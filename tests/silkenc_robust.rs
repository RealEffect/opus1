//! Robustness of the SILK encoder's packet functions, driven directly.
//!
//! `OpusEncoder` reaches only part of them: it never passes odd lengths.
//! Here the functions are
//! called the way `OpusEncoder` calls them, with settings that change at
//! random between packets, in every packet duration and at every internal
//! rate, into buffers from one byte up. Nothing may panic, and every packet
//! that fits its buffer must be read to its end by the libopus decoder: the
//! decoder's final range is the encoder's.
// `RangeCoder::finish` needs the `std` feature.
#![cfg(feature = "std")]

mod silkenc_common;
use silkenc_common::*;

use opus_rs::range_coder::RangeCoder;
use opus_rs::silk::control_codec::{
    SilkBandwidthControl, silk_control_audio_bandwidth, silk_control_encoder, silk_setup_lbrr,
};
use opus_rs::silk::decode_pulses::silk_decode_pulses;
use opus_rs::silk::define::MAX_FRAMES_PER_PACKET;
use opus_rs::silk::enc_api::{
    silk_average_first_mono_frame, silk_downmix_stereo, silk_encode, silk_encode_prefill,
    silk_encode_prefill_stereo, silk_encode_stereo_packet_to_mono, silk_prev_channels_internal,
};
use opus_rs::silk::encode_pulses::silk_encode_pulses;
use opus_rs::silk::init_encoder::silk_init_encoder;
use opus_rs::silk::resampler::SilkResampler;
use opus_rs::silk::structs::SilkEncoderState;

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

/// Two SILK encoders and the two input resamplers, used the way
/// `OpusEncoder` uses them for SILK-only packets of 48 kHz stereo input.
struct Driver {
    fs_khz: i32,
    ms: i32,
    bitrate: i32,
    complexity: i32,
    fec: bool,
    loss: i32,
    mid: Box<SilkEncoderState>,
    side: Box<SilkEncoderState>,
    resamplers: [SilkResampler; 2],
}

struct Packet {
    bytes: Vec<u8>,
    final_range: u32,
}

impl Driver {
    fn new(fs_khz: i32, ms: i32, bitrate: i32, complexity: i32) -> Self {
        let mut d = Self {
            fs_khz,
            ms,
            bitrate,
            complexity,
            fec: false,
            loss: 0,
            mid: Box::new(SilkEncoderState::default()),
            side: Box::new(SilkEncoderState::default()),
            resamplers: [SilkResampler::default(), SilkResampler::default()],
        };
        assert_eq!(silk_init_encoder(&mut d.mid, 0), 0);
        assert_eq!(silk_init_encoder(&mut d.side, 0), 0);
        d.control();
        d.start_resamplers();
        d
    }

    fn start_resamplers(&mut self) {
        assert_eq!(self.resamplers[0].init_for_enc(API_FS as i32, self.fs_khz * 1000), 0);
        self.resamplers[1] = self.resamplers[0].clone();
    }

    /// What `OpusEncoder` does when the rate, the packet size or the
    /// complexity changes.
    fn control(&mut self) {
        for enc in [&mut self.mid, &mut self.side] {
            assert_eq!(silk_control_encoder(enc, self.fs_khz, self.ms, self.bitrate, self.complexity), 0);
            enc.s_cmn.lbrr_flags = [0; MAX_FRAMES_PER_PACKET];
        }
    }

    /// The settings of every packet.
    fn per_packet(&mut self) {
        let coded = (self.fec && self.loss > 0) as i32;
        for enc in [&mut self.mid, &mut self.side] {
            enc.s_cmn.use_in_band_fec = self.fec as i32;
            enc.s_cmn.packet_loss_perc = self.loss;
            silk_setup_lbrr(enc, coded);
        }
    }

    fn internal_len(&self, api_len: usize) -> usize {
        api_len * self.fs_khz as usize * 1000 / API_FS
    }

    /// Restart the encoders and feed them 10 ms, as after CELT-only packets
    /// (`clear_all`) or for a rate switch.
    fn prefill(&mut self, pcm: &[i16], channels: usize, clear_all: bool, activity: i32) {
        if clear_all {
            *self.mid = SilkEncoderState::default();
            *self.side = SilkEncoderState::default();
        }
        silk_init_encoder(&mut self.mid, 0);
        silk_init_encoder(&mut self.side, 0);
        self.control();
        self.per_packet();
        self.start_resamplers();
        let n_api = API_FS / 100;
        let n = self.internal_len(n_api);
        let mut internal = [vec![0i16; n], vec![0i16; n]];
        for c in 0..2 {
            let channel: Vec<i16> = pcm[..2 * n_api].as_chunks::<2>().0.iter().map(|lr| lr[c]).collect();
            assert_eq!(self.resamplers[c].process(&mut internal[c], &channel, n_api as i32), 0);
        }
        if channels == 2 {
            let (l, r) = (&internal[0], &internal[1]);
            silk_encode_prefill_stereo(&mut self.mid, &mut self.side, l, r, self.bitrate, activity);
        } else {
            silk_encode_prefill(&mut self.mid, &internal[0], activity);
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
    ) -> Option<Packet> {
        self.per_packet();
        let frame = pcm.len() / 2;
        let n = self.internal_len(frame);
        let prev = silk_prev_channels_internal(&self.mid);
        let mut rc = RangeCoder::new_encoder(buffer as u32);
        let mut n_bytes = 0;
        let ret = if channels == 2 {
            if prev == 1 {
                self.resamplers[1] = self.resamplers[0].clone();
            }
            let mut internal = [vec![0i16; n], vec![0i16; n]];
            for c in 0..2 {
                let channel: Vec<i16> = pcm.as_chunks::<2>().0.iter().map(|lr| lr[c]).collect();
                assert_eq!(self.resamplers[c].process(&mut internal[c], &channel, frame as i32), 0);
            }
            silk_encode_stereo_packet_to_mono(
                &mut self.mid,
                &mut self.side,
                &internal[0],
                &internal[1],
                n,
                &mut rc,
                &mut n_bytes,
                self.bitrate,
                max_bits,
                cbr as i32,
                activity,
                to_mono,
            )
        } else {
            let mut mono_api = vec![0i16; frame];
            silk_downmix_stereo(pcm, &mut mono_api);
            let mut mono = vec![0i16; n];
            assert_eq!(self.resamplers[0].process(&mut mono, &mono_api, frame as i32), 0);
            if prev == 2 {
                let fl = self.mid.s_cmn.frame_length as usize;
                let first_api = &mono_api[..fl * API_FS / (self.fs_khz as usize * 1000)];
                silk_average_first_mono_frame(&mut self.resamplers[1], &mut mono[..fl], first_api);
            }
            silk_encode(&mut self.mid, &mono, n, &mut rc, &mut n_bytes, self.bitrate, max_bits, cbr as i32, activity)
        };
        assert_eq!(ret, 0, "SILK error");
        if rc.error != 0 || rc.tell() > 8 * buffer as i32 {
            return None;
        }
        let final_range = rc.rng;
        // SILK-only configurations: narrowband 0 to 3, mediumband 4 to 7,
        // wideband 8 to 11, each for 10, 20, 40 and 60 ms.
        let config = (self.fs_khz as u8 / 4 - 2) * 4 + [10, 20, 40, 60].iter().position(|&m| m == self.ms).unwrap() as u8;
        let mut bytes = vec![(config << 3) | (((channels == 2) as u8) << 2)];
        bytes.extend(rc.finish());
        Some(Packet { bytes, final_range })
    }
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
                    2 => {
                        d.ms = rng.pick(&[10, 20, 40, 60]);
                        d.control();
                    }
                    3 => {
                        d.fs_khz = rng.pick(&[8, 12, 16]);
                        d.control();
                        d.start_resamplers();
                    }
                    4 => {
                        d.complexity = rng.below(11) as i32;
                        d.control();
                    }
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
            if rng.chance(6) {
                let clear_all = rng.chance(50);
                d.prefill(&pcm[2 * pos..], want, clear_all, rng.below(2) as i32);
            }
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
            let packet = d.encode(chunk, channels, to_mono, buffer, max_bits, cbr, rng.below(8).min(1) as i32);
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
    let mut n_bytes = 0;

    // Not set up.
    let mut mid = Box::new(SilkEncoderState::default());
    let mut side = Box::new(SilkEncoderState::default());
    assert!(silk_encode(&mut mid, &pcm, 320, &mut rc, &mut n_bytes, 20_000, 8000, 0, 1) < 0);
    assert!(
        silk_encode_stereo_packet_to_mono(&mut mid, &mut side, &pcm, &pcm, 320, &mut rc, &mut n_bytes, 20_000, 8000, 0, 1, true)
            < 0
    );
    silk_encode_prefill(&mut mid, &pcm, 1);
    silk_encode_prefill_stereo(&mut mid, &mut side, &pcm, &pcm, 20_000, 1);

    // Set up, with too little input.
    let mut d = Driver::new(16, 40, 24_000, 5);
    for n in [0usize, 1, 319, 320, 639] {
        assert!(silk_encode(&mut d.mid, &pcm[..n], n, &mut rc, &mut n_bytes, 20_000, 8000, 0, 1) < 0);
        assert!(silk_encode(&mut d.mid, &pcm[..n], 640, &mut rc, &mut n_bytes, 20_000, 8000, 0, 1) < 0);
        assert!(silk_encode(&mut d.mid, &pcm, n, &mut rc, &mut n_bytes, 20_000, 8000, 0, 1) < 0);
        let (mid, side) = (&mut d.mid, &mut d.side);
        assert!(
            silk_encode_stereo_packet_to_mono(mid, side, &pcm[..n], &pcm, 640, &mut rc, &mut n_bytes, 20_000, 8000, 0, 1, false)
                < 0
        );
        silk_encode_prefill(&mut d.mid, &pcm[..n.min(159)], 1);
        silk_encode_prefill_stereo(&mut d.mid, &mut d.side, &pcm[..n.min(159)], &pcm, 20_000, 1);
    }
    // With more than a packet: the packet is coded.
    for n in [640usize, 641, 4000, usize::MAX] {
        let mut rc = RangeCoder::new_encoder(1275);
        assert_eq!(silk_encode(&mut d.mid, &pcm, n, &mut rc, &mut n_bytes, 20_000, 8000, 0, 1), 0, "{n} samples");
    }
    // A side encoder at another rate than the mid.
    let mut d = Driver::new(16, 20, 24_000, 5);
    let (mid, side) = (&mut d.mid, &mut d.side);
    assert_eq!(silk_encode_stereo_packet_to_mono(mid, side, &pcm, &pcm, 320, &mut rc, &mut n_bytes, 30_000, 8000, 0, 1, false), 0);
    assert_eq!(silk_control_encoder(&mut d.side, 8, 20, 20_000, 5), 0);
    let (mid, side) = (&mut d.mid, &mut d.side);
    assert!(silk_encode_stereo_packet_to_mono(mid, side, &pcm, &pcm, 320, &mut rc, &mut n_bytes, 30_000, 8000, 0, 1, false) < 0);
    silk_encode_prefill_stereo(&mut d.mid, &mut d.side, &pcm, &pcm, 20_000, 1);

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
/// opus-rs set the encoder up for them: for one subframe per frame, or for
/// frames longer than its buffers.
#[test]
fn settings_that_libopus_refuses_leave_the_encoder_as_it_was() {
    let pcm = vec![1000i16; 4000];
    let mut n_bytes = 0;
    let refused = [(0, 20), (24, 20), (-8, 20), (16, 0), (16, 5), (16, 30), (16, 100), (16, -20), (i32::MAX, i32::MAX)];
    for (fs_khz, ms) in refused {
        let mut enc = Box::new(SilkEncoderState::default());
        silk_init_encoder(&mut enc, 0);
        // A new encoder stays unusable.
        assert!(silk_control_encoder(&mut enc, fs_khz, ms, 20_000, 5) < 0, "{fs_khz} kHz, {ms} ms");
        let mut rc = RangeCoder::new_encoder(1275);
        assert!(silk_encode(&mut enc, &pcm, 320, &mut rc, &mut n_bytes, 20_000, 8000, 0, 1) < 0);
        // A running one keeps its settings.
        assert_eq!(silk_control_encoder(&mut enc, 16, 20, 20_000, 5), 0);
        assert!(silk_control_encoder(&mut enc, fs_khz, ms, 20_000, 9) < 0, "{fs_khz} kHz, {ms} ms");
        let cmn = &enc.s_cmn;
        assert_eq!((cmn.fs_khz, cmn.packet_size_ms, cmn.frame_length, cmn.nb_subfr, cmn.complexity), (16, 20, 320, 4, 5));
        let mut rc = RangeCoder::new_encoder(1275);
        assert_eq!(silk_encode(&mut enc, &pcm, 320, &mut rc, &mut n_bytes, 20_000, 8000, 0, 1), 0);
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
/// checks (`cargo test` without `--release`) opus-rs stopped with a panic on
/// the first product, and in every build the rate switch state machine
/// divided by zero for a packet duration of -5 ms.
#[test]
fn extreme_rates_and_limits_do_not_overflow() {
    let pcm = vec![1000i16; 4000];
    let ramp: Vec<i16> = (0..4000).map(|i| (i * 37 % 20_000 - 10_000) as i16).collect();
    let mut n_bytes = 0;
    for complexity in [i32::MIN, -1, 11, 1000, i32::MAX] {
        for bitrate in [i32::MIN, -1, 0, 1, i32::MAX] {
            let mut enc = Box::new(SilkEncoderState::default());
            silk_init_encoder(&mut enc, 0);
            assert_eq!(silk_control_encoder(&mut enc, 12, 60, bitrate, complexity), 0);
            for max_bits in [i32::MIN, 0, i32::MAX] {
                let mut rc = RangeCoder::new_encoder(1275);
                assert_eq!(silk_encode(&mut enc, &pcm, 720, &mut rc, &mut n_bytes, bitrate, max_bits, 1, 1), 0);
            }
            // The same with two channels.
            let mut side = Box::new(SilkEncoderState::default());
            silk_init_encoder(&mut side, 0);
            for max_bits in [i32::MIN, 0, i32::MAX] {
                let mut rc = RangeCoder::new_encoder(1275);
                let ret = silk_encode_stereo_packet_to_mono(
                    &mut enc, &mut side, &pcm, &ramp, 720, &mut rc, &mut n_bytes, bitrate, max_bits, 1, 1, false,
                );
                assert_eq!(ret, 0);
            }
            silk_encode_prefill_stereo(&mut enc, &mut side, &pcm, &ramp, bitrate, 1);
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
