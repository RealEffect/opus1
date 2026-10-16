//! Robustness of the SILK encoder, driven directly.
//!
//! `OpusEncoder` reaches only part of it: it never passes odd lengths. Here
//! `SilkEncoder` is called the way `OpusEncoder` calls it, with settings that
//! change at random between packets, in every packet duration and at every
//! internal rate, into buffers from one byte up. The packet functions below
//! it get odd lengths and states. Nothing may panic, and every packet that
//! fits its buffer must be read to its end by the libopus decoder: the
//! decoder's final range is the encoder's.
// `RangeCoder::finish` needs the `std` feature.
#![cfg(feature = "std")]

mod silkenc_common;
use silkenc_common::*;

use opus1::range_coder::RangeCoder;
use opus1::silk::control_codec::{
    SilkBandwidthControl, silk_control_audio_bandwidth, silk_control_encoder,
};
use opus1::silk::decode_pulses::silk_decode_pulses;
use opus1::silk::enc_api::{
    SilkControl, SilkEncoder, SilkInput, SilkPrefill, silk_average_first_mono_frame,
    silk_downmix_stereo, silk_encode_packet, silk_encode_prefill,
};
use opus1::silk::encode_pulses::silk_encode_pulses;
use opus1::silk::init_encoder::silk_init_encoder;
use opus1::silk::resampler::SilkResampler;
use opus1::silk::structs::SilkEncoderState;

const API_FS: usize = 48_000;

impl Rng {
    fn below(&mut self, n: u32) -> u32 {
        self.next() % n
    }
    fn chance(&mut self, percent: u32) -> bool {
        self.below(100) < percent
    }
    fn pick<T: Copy>(&mut self, items: &[T]) -> T {
        items[self.below(items.len() as u32) as usize]
    }
}

/// A SILK encoder for 48 kHz stereo input and the settings of its packets,
/// used the way `OpusEncoder` uses it for SILK-only packets.
struct Driver {
    fs_khz: i32,
    ms: i32,
    bitrate: i32,
    complexity: i32,
    fec: bool,
    loss: i32,
    silk: SilkEncoder,
}

struct Packet {
    bytes: Vec<u8>,
    final_range: u32,
}

/// A restart before a packet: after CELT-only packets (`clear_all`) or for
/// a rate switch.
struct Restart<'a> {
    clear_all: bool,
    /// The 10 ms before the packet, interleaved stereo.
    pcm: &'a [i16],
}

impl Driver {
    fn new(fs_khz: i32, ms: i32, bitrate: i32, complexity: i32) -> Self {
        Self {
            fs_khz,
            ms,
            bitrate,
            complexity,
            fec: false,
            loss: 0,
            silk: SilkEncoder::new(API_FS as i32, 2),
        }
    }

    /// Code one packet of interleaved stereo input in `channels` channels
    /// into a buffer of `buffer` bytes. `None` when the packet did not fit.
    #[allow(clippy::too_many_arguments)]
    fn encode(
        &mut self,
        pcm: &[i16],
        channels: usize,
        to_mono: bool,
        buffer: usize,
        max_bits: i32,
        cbr: bool,
        activity: i32,
        restart: Option<Restart<'_>>,
    ) -> Option<Packet> {
        // The rate asked for is the only one allowed: a change takes effect
        // at once.
        let mut ctl = SilkControl {
            channels_internal: channels,
            payload_size_ms: self.ms,
            min_khz: self.fs_khz,
            max_khz: self.fs_khz,
            desired_khz: self.fs_khz,
            bit_rate: self.bitrate,
            max_bits,
            use_cbr: cbr,
            complexity: self.complexity,
            packet_loss_perc: self.loss,
            lbrr_coded: self.fec && self.loss > 0,
            activity,
            to_mono,
            ..SilkControl::default()
        };
        let prefill = restart.map(|r| {
            if r.clear_all {
                self.silk.reset();
            }
            SilkPrefill { rate_switch: !r.clear_all, input: Some(SilkInput::Int16(&r.pcm[..2 * API_FS / 100])) }
        });
        let mut rc = RangeCoder::new_encoder(buffer as u32);
        self.silk.encode(&mut ctl, SilkInput::Int16(pcm), prefill, &mut rc).expect("SILK error");
        if rc.error != 0 || rc.tell() > 8 * buffer as i32 {
            return None;
        }
        assert_eq!(ctl.internal_khz, self.fs_khz);
        let final_range = rc.rng;
        // SILK-only configurations: narrowband 0 to 3, mediumband 4 to 7,
        // wideband 8 to 11, each for 10, 20, 40 and 60 ms.
        let config = (self.fs_khz as u8 / 4 - 2) * 4 + [10, 20, 40, 60].iter().position(|&m| m == self.ms).unwrap() as u8;
        let mut bytes = vec![(config << 3) | (((channels == 2) as u8) << 2)];
        bytes.extend(rc.finish());
        Some(Packet { bytes, final_range })
    }
}

/// Two SILK channel states at `fs_khz`, set up for packets of `ms`.
fn channels(fs_khz: i32, ms: i32) -> (Box<SilkEncoderState>, Box<SilkEncoderState>) {
    let mut mid = Box::new(SilkEncoderState::default());
    let mut side = Box::new(SilkEncoderState::default());
    for enc in [&mut mid, &mut side] {
        assert_eq!(silk_init_encoder(enc, 0), 0);
        assert_eq!(silk_control_encoder(enc, fs_khz, ms, 24_000, 5), 0);
    }
    (mid, side)
}

/// 48 kHz stereo input for the walks: speech in several shapes.
fn input(kind: u32, n: usize) -> Vec<i16> {
    let to_i16 = |x: f32| (x * 32768.0).round().clamp(-32768.0, 32767.0) as i16;
    match kind {
        // Speech, the right channel a delayed and weaker copy.
        0 => speech(API_FS, 2, n, 0.5).iter().map(|&x| to_i16(x)).collect(),
        // Two unrelated channels: a strong side.
        1 => {
            let (a, b) = (speech(API_FS, 1, n, 0.5), speech(API_FS, 1, n, 7.0));
            a.iter().zip(&b).flat_map(|(&l, &r)| [to_i16(l), to_i16(r)]).collect()
        }
        // Very quiet.
        2 => speech(API_FS, 2, n, 3.0).iter().map(|&x| to_i16(x * 0.004)).collect(),
        // Clipped at full scale, the channels in opposite phase.
        3 => speech(API_FS, 1, n, 1.0)
            .iter()
            .flat_map(|&x| {
                let v = to_i16(x * 40.0);
                [v, v.saturating_neg()]
            })
            .collect(),
        // Silence, then full-scale noise.
        _ => {
            let mut rng = Rng(77);
            (0..2 * n).map(|i| if i < n { 0 } else { (rng.next() >> 16) as i16 }).collect()
        }
    }
}

/// Settings that change at random in a running stream.
#[test]
fn every_packet_of_a_random_walk_is_valid_for_the_libopus_decoder() {
    let seconds = 3;
    let signals: Vec<Vec<i16>> = (0..5).map(|k| input(k, seconds * API_FS)).collect();
    let (mut packets, mut too_large, mut one_channel, mut with_fec) = (0, 0, 0, 0);
    for stream in 0..240u32 {
        let mut rng = Rng(0x9E37_79B9 ^ stream.wrapping_mul(2_654_435_761) | 1);
        let pcm = &signals[(stream % 5) as usize];
        let mut d = Driver::new(rng.pick(&[8, 12, 16]), rng.pick(&[10, 20, 40, 60]), 24_000, rng.below(11) as i32);
        let mut dec = CDecoder::new(API_FS, 2);
        let mut fec_dec = CDecoder::new(API_FS, 2);
        // FEC is on from the start in two streams of three.
        (d.fec, d.loss) = (stream % 3 != 0, 10);
        let (mut want, mut cbr, mut small) = (2usize, false, false);
        let mut prev_stream_channels = 2;
        let mut pos = 0;
        for i in 0..40 {
            // One or two settings change before most packets.
            for _ in 0..rng.below(3) {
                match rng.below(12) {
                    0 | 1 => {
                        d.bitrate = match rng.below(8) {
                            0 => 500,
                            1 => 300_000,
                            2 => 2_000 + rng.below(6_000) as i32,
                            _ => 6_000 + rng.below(60_000) as i32,
                        }
                    }
                    2 => d.ms = rng.pick(&[10, 20, 40, 60]),
                    3 => d.fs_khz = rng.pick(&[8, 12, 16]),
                    4 => d.complexity = rng.below(11) as i32,
                    5 | 6 => d.fec = !d.fec,
                    7 => d.loss = rng.pick(&[0, 1, 5, 10, 25, 60, 100]),
                    8 | 9 => want = 3 - want,
                    10 => cbr = !cbr,
                    _ => small = !small,
                }
            }
            let frame = API_FS / 1000 * d.ms as usize;
            if pos + frame + API_FS / 100 > pcm.len() / 2 {
                pos = 0;
            }
            let restart = rng.chance(6).then(|| Restart { clear_all: rng.chance(50), pcm: &pcm[2 * pos..] });
            // libopus's rule for the change to one channel
            // (opus_encoder.c:1560-1571), broken now and then.
            let mut channels = want;
            let mut to_mono = false;
            if channels == 1 && prev_stream_channels == 2 && !rng.chance(15) {
                channels = 2;
                to_mono = true;
            } else if rng.chance(3) {
                to_mono = channels == 2;
            }
            let buffer = if small { 1 + rng.below(40) as usize } else { 1275 };
            let max_bits = match rng.below(10) {
                0 => 0,
                1 => -64,
                2 => 80,
                3 => 1 << 20,
                _ => 8 * buffer as i32,
            };
            let chunk = &pcm[2 * pos..2 * (pos + frame)];
            pos += frame;
            let what = format!(
                "stream {stream}, packet {i}: {} kHz, {} ms, {} b/s, {channels} ch, to_mono {to_mono}, \
                 fec {} at {} %, cbr {cbr}, buffer {buffer}, max_bits {max_bits}, complexity {}",
                d.fs_khz, d.ms, d.bitrate, d.fec, d.loss, d.complexity
            );
            let activity = rng.below(8).min(1) as i32;
            let packet = d.encode(chunk, channels, to_mono, buffer, max_bits, cbr, activity, restart);
            prev_stream_channels = channels;
            packets += 1;
            one_channel += (channels == 1) as usize;
            let Some(packet) = packet else {
                too_large += 1;
                continue;
            };
            let header = u16::from_be_bytes([packet.bytes[1], *packet.bytes.get(2).unwrap_or(&0)]);
            let n = (d.ms as usize / 20).max(1);
            with_fec += ((header >> (15 - n)) & 1) as usize;
            if let Err(e) = dec.decode(Some(&packet.bytes), frame, false) {
                panic!("{what}: libopus decoder error {e}");
            }
            assert_eq!(dec.final_range(), packet.final_range, "{what}: final range");
            // The FEC data of the packet, then the packet itself.
            if let Err(e) = fec_dec.decode(Some(&packet.bytes), frame, true) {
                panic!("{what}: libopus decoder error {e} on the FEC data");
            }
            fec_dec.decode(Some(&packet.bytes), frame, false).unwrap();
            assert_eq!(fec_dec.final_range(), packet.final_range, "{what}: final range after FEC");
        }
    }
    // The walks must reach what they are meant to reach.
    assert!(too_large > 100 && too_large < packets / 2, "{too_large} of {packets} packets did not fit");
    assert!(one_channel > packets / 5, "{one_channel} of {packets} packets code one channel");
    assert!(with_fec > packets / 20, "{with_fec} of {packets} packets carry FEC data");
}

/// The excitation is read back as it was coded, for every frame length SILK
/// uses, whatever lies in the buffer after the frame.
#[test]
fn pulses_are_decoded_as_coded_for_every_frame_length() {
    let mut rng = Rng(4242);
    for frame_length in [80usize, 120, 160, 240, 320] {
        for round in 0..300 {
            let (signal_type, quant_offset_type) = (round % 3, round / 3 % 2);
            // Small values mostly; some blocks so large that their least
            // significant bits are coded separately, up to the largest value.
            let mut pulses = [0i8; 336];
            let mut amp = 1 + rng.below(4);
            for (k, p) in pulses.iter_mut().enumerate() {
                if k % 16 == 0 && rng.chance(20) {
                    amp = rng.pick(&[1, 2, 6, 30, 127]);
                }
                *p = (rng.below(2 * amp + 1) as i32 - amp as i32) as i8;
                if rng.chance(40) {
                    *p = 0;
                }
            }
            let want: Vec<i16> = pulses[..frame_length].iter().map(|&p| p as i16).collect();
            let mut enc = RangeCoder::new_encoder(1275);
            silk_encode_pulses(&mut enc, signal_type, quant_offset_type, &mut pulses, frame_length);
            let (bits, final_range) = (enc.tell(), enc.rng);
            let bytes = enc.finish();

            let mut dec = RangeCoder::new_decoder(&bytes);
            let mut got = [0i16; 336];
            silk_decode_pulses(&mut dec, &mut got, signal_type, quant_offset_type, frame_length as i32);
            let what = format!("{frame_length} samples, round {round}");
            assert_eq!(&got[..frame_length], &want[..], "{what}");
            assert_eq!((dec.tell(), dec.rng), (bits, final_range), "{what}: bits read and final range");
        }
    }
}

/// Lengths that SILK never uses, and buffers shorter than the frame.
#[test]
fn the_pulse_coder_takes_any_length() {
    let mut rng = Rng(7);
    for frame_length in [0usize, 1, 7, 15, 16, 17, 119, 121, 319, 320, 321, 336, 5000, usize::MAX] {
        for buffer in [0usize, 1, 15, 16, 17, 120, 128, 320, 336, 400] {
            let mut pulses: Vec<i8> = (0..buffer).map(|_| (rng.below(255) as i32 - 128) as i8).collect();
            let mut rc = RangeCoder::new_encoder(1275);
            silk_encode_pulses(&mut rc, (frame_length % 3) as i32, (buffer % 2) as i32, &mut pulses, frame_length);
        }
    }
}

/// Calls with a state that is not set up, with too little input and with odd
/// lengths give an error code or do nothing.
#[test]
fn calls_that_make_no_sense_do_not_panic() {
    let pcm = vec![1000i16; 4000];
    let mut rc = RangeCoder::new_encoder(1275);

    // Not set up.
    let mut mid = Box::new(SilkEncoderState::default());
    let mut side = Box::new(SilkEncoderState::default());
    assert!(silk_encode_packet(&mut mid, None, [&pcm, &[]], &mut rc, 20_000, 8000, false, 1, false) < 0);
    assert!(silk_encode_packet(&mut mid, Some(&mut side), [&pcm, &pcm], &mut rc, 20_000, 8000, false, 1, true) < 0);
    silk_encode_prefill(&mut mid, None, [&pcm, &[]], 20_000, 1);
    silk_encode_prefill(&mut mid, Some(&mut side), [&pcm, &pcm], 20_000, 1);

    // Set up, with too little input.
    let (mut mid, mut side) = channels(16, 40);
    for n in [0usize, 1, 319, 320, 639] {
        assert!(silk_encode_packet(&mut mid, None, [&pcm[..n], &[]], &mut rc, 20_000, 8000, false, 1, false) < 0);
        assert!(silk_encode_packet(&mut mid, Some(&mut side), [&pcm[..n], &pcm], &mut rc, 20_000, 8000, false, 1, false) < 0);
        assert!(silk_encode_packet(&mut mid, Some(&mut side), [&pcm, &pcm[..n]], &mut rc, 20_000, 8000, false, 1, false) < 0);
        silk_encode_prefill(&mut mid, None, [&pcm[..n.min(159)], &[]], 20_000, 1);
        silk_encode_prefill(&mut mid, Some(&mut side), [&pcm[..n.min(159)], &pcm], 20_000, 1);
    }
    // With more than a packet: the packet is coded.
    let (mut mid, _) = channels(16, 40);
    for n in [640usize, 641, 4000] {
        let mut rc = RangeCoder::new_encoder(1275);
        assert_eq!(silk_encode_packet(&mut mid, None, [&pcm[..n], &[]], &mut rc, 20_000, 8000, false, 1, false), 0, "{n} samples");
    }
    // A side encoder at another rate than the mid.
    let (mut mid, mut side) = channels(16, 20);
    assert_eq!(silk_encode_packet(&mut mid, Some(&mut side), [&pcm, &pcm], &mut rc, 30_000, 8000, false, 1, false), 0);
    assert_eq!(silk_control_encoder(&mut side, 8, 20, 20_000, 5), 0);
    assert!(silk_encode_packet(&mut mid, Some(&mut side), [&pcm, &pcm], &mut rc, 30_000, 8000, false, 1, false) < 0);
    silk_encode_prefill(&mut mid, Some(&mut side), [&pcm, &pcm], 20_000, 1);

    // The whole encoder with input longer than a packet, and with input
    // that is not a whole number of milliseconds.
    for n in [2 * 2881usize, 10, 0] {
        let mut silk = SilkEncoder::new(48_000, 2);
        let mut ctl = SilkControl {
            channels_internal: 2,
            payload_size_ms: 20,
            min_khz: 16,
            max_khz: 16,
            desired_khz: 16,
            bit_rate: 30_000,
            max_bits: 8000,
            complexity: 5,
            activity: 1,
            ..SilkControl::default()
        };
        let input = vec![1000i16; n];
        assert!(silk.encode(&mut ctl, SilkInput::Int16(&input), None, &mut rc).is_err(), "{n} samples");
    }

    // The downmix and the average of the first mono frame with odd lengths.
    for (n_in, n_out) in [(0usize, 0usize), (1, 1), (7, 9), (960, 0), (0, 960), (961, 480), (4000, 4000)] {
        let mut mono = vec![0i16; n_out];
        silk_downmix_stereo(&pcm[..n_in], &mut mono);
        let mut started = SilkResampler::default();
        assert_eq!(started.init_for_enc(48_000, 16_000), 0);
        for mut resampler in [SilkResampler::default(), started] {
            silk_average_first_mono_frame(&mut resampler, &mut mono, &pcm[..n_in]);
        }
    }
}

/// A rate that SILK does not have and a packet duration that it does not
/// have are refused, and the encoder stays as it was. libopus refuses them
/// in silk/check_control_input.c and silk_setup_fs (silk/control_codec.c).
/// opus1 set the encoder up for them: for one subframe per frame, or for
/// frames longer than its buffers.
#[test]
fn settings_that_libopus_refuses_leave_the_encoder_as_it_was() {
    let pcm = vec![1000i16; 4000];
    let refused = [(0, 20), (24, 20), (-8, 20), (16, 0), (16, 5), (16, 30), (16, 100), (16, -20), (i32::MAX, i32::MAX)];
    for (fs_khz, ms) in refused {
        let mut enc = Box::new(SilkEncoderState::default());
        silk_init_encoder(&mut enc, 0);
        // A new encoder stays unusable.
        assert!(silk_control_encoder(&mut enc, fs_khz, ms, 20_000, 5) < 0, "{fs_khz} kHz, {ms} ms");
        let mut rc = RangeCoder::new_encoder(1275);
        assert!(silk_encode_packet(&mut enc, None, [&pcm, &[]], &mut rc, 20_000, 8000, false, 1, false) < 0);
        // A running one keeps its settings.
        assert_eq!(silk_control_encoder(&mut enc, 16, 20, 20_000, 5), 0);
        assert!(silk_control_encoder(&mut enc, fs_khz, ms, 20_000, 9) < 0, "{fs_khz} kHz, {ms} ms");
        let cmn = &enc.s_cmn;
        assert_eq!((cmn.fs_khz, cmn.packet_size_ms, cmn.frame_length, cmn.nb_subfr, cmn.complexity), (16, 20, 320, 4, 5));
        let mut rc = RangeCoder::new_encoder(1275);
        assert_eq!(silk_encode_packet(&mut enc, None, [&pcm, &[]], &mut rc, 20_000, 8000, false, 1, false), 0);
    }
    // Every pair that libopus accepts is accepted.
    for fs_khz in [8, 12, 16] {
        for ms in [10, 20, 40, 60] {
            let mut enc = Box::new(SilkEncoderState::default());
            silk_init_encoder(&mut enc, 0);
            assert_eq!(silk_control_encoder(&mut enc, fs_khz, ms, 20_000, 5), 0);
            assert_eq!(enc.s_cmn.frame_length, fs_khz * ms.min(20));
        }
    }
}

/// Bitrates, bit limits and control values far outside what Opus allows
/// overflow nothing and divide by nothing.
///
/// libopus computes with them in 32 bits and wraps. In a build with overflow
/// checks (`cargo test` without `--release`) opus1 stopped with a panic on
/// the first product, and in every build the rate switch state machine
/// divided by zero for a packet duration of -5 ms.
#[test]
fn extreme_rates_and_limits_do_not_overflow() {
    let pcm = vec![1000i16; 4000];
    let ramp: Vec<i16> = (0..4000).map(|i| (i * 37 % 20_000 - 10_000) as i16).collect();
    for complexity in [i32::MIN, -1, 11, 1000, i32::MAX] {
        for bitrate in [i32::MIN, -1, 0, 1, i32::MAX] {
            let mut enc = Box::new(SilkEncoderState::default());
            silk_init_encoder(&mut enc, 0);
            assert_eq!(silk_control_encoder(&mut enc, 12, 60, bitrate, complexity), 0);
            for max_bits in [i32::MIN, 0, i32::MAX] {
                let mut rc = RangeCoder::new_encoder(1275);
                assert_eq!(silk_encode_packet(&mut enc, None, [&pcm, &[]], &mut rc, bitrate, max_bits, true, 1, false), 0);
            }
            // The same with two channels.
            let mut side = Box::new(SilkEncoderState::default());
            silk_init_encoder(&mut side, 0);
            for max_bits in [i32::MIN, 0, i32::MAX] {
                let mut rc = RangeCoder::new_encoder(1275);
                let ret = silk_encode_packet(&mut enc, Some(&mut side), [&pcm, &ramp], &mut rc, bitrate, max_bits, true, 1, false);
                assert_eq!(ret, 0);
            }
            silk_encode_prefill(&mut enc, Some(&mut side), [&pcm, &ramp], bitrate, 1);
        }
    }

    // The rate switch state machine with any control values.
    let mut rng = Rng(31);
    let values = |rng: &mut Rng| rng.pick(&[i32::MIN, -16_000, -5, -1, 0, 1, 5, 10, 20, 8_000, 12_000, 16_000, 48_000, i32::MAX]);
    for _ in 0..20_000 {
        let mut enc = Box::new(SilkEncoderState::default());
        enc.s_cmn.fs_khz = rng.pick(&[i32::MIN, -1, 0, 8, 12, 16, 24, i32::MAX]);
        enc.s_cmn.s_lp.saved_fs_khz = rng.pick(&[i32::MIN, 0, 8, 12, 16, i32::MAX]);
        enc.s_cmn.s_lp.mode = rng.pick(&[i32::MIN, -2, 0, 1, i32::MAX]);
        enc.s_cmn.s_lp.transition_frame_no = values(&mut rng);
        let mut ctl = SilkBandwidthControl {
            api_fs_hz: values(&mut rng),
            max_internal_fs_hz: values(&mut rng),
            min_internal_fs_hz: values(&mut rng),
            desired_internal_fs_hz: values(&mut rng),
            payload_size_ms: values(&mut rng),
            allow_bandwidth_switch: rng.chance(50),
            opus_can_switch: rng.chance(50),
            max_bits: values(&mut rng),
            switch_ready: rng.chance(50),
        };
        silk_control_audio_bandwidth(&mut enc.s_cmn, &mut ctl);
    }
}
