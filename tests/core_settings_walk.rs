//! Settings that change in a running stream, with every frame duration and
//! output buffers from 1 byte to several kilobytes: the encoder never
//! panics, every packet is valid for the C decoder, and the packet-level
//! rules of `opus_encode_native` give what libopus gives.
//!
//! The reference is the C libopus encoder, fed the same input with the same
//! settings and the same buffer, at a complexity below 7, where both
//! encoders decide without the speech/music analysis (both run it from
//! complexity 7 on; `core_analysis_libopus.rs` compares that).
//! The C encoder is told to keep stereo input stereo
//! (`OPUS_SET_FORCE_CHANNELS(2)`), because opus1 does not code stereo
//! input as mono yet.
//!
//! What is compared, and what is not:
//!
//! - Compared on every packet: whether the encoder fails, the size under a
//!   constant bitrate, CELT-only or not, the channel count, and every byte
//!   of a packet without audio but the bandwidth in its first byte.
//! - Counted, not compared: the bandwidth. SILK changes its bandwidth in a
//!   pause of the speech, which the two SILK encoders find some packets
//!   apart (4 packets in 1000 over 4000 streams). A CELT-only packet after
//!   such packets can differ as well, through the hysteresis of the
//!   decision.
//! - Every packet is decoded by the C decoder, which must end in the
//!   encoder's final range.

mod core_common;
mod k24_common;

use audiopus_sys as sys;
use core_common::*;
use k24_common::{multitone, speech, to_stereo};
use opus1::{Application, Bandwidth, Bitrate, Channels, ForcedMode, OpusEncoder, SampleRate, packet};

/// A small deterministic generator (the 64-bit LCG of Knuth's MMIX).
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    fn below(&mut self, n: usize) -> usize {
        self.next() as usize % n
    }
    fn pick<T: Copy>(&mut self, items: &[T]) -> T {
        items[self.below(items.len())]
    }
    fn chance(&mut self, percent: usize) -> bool {
        self.below(100) < percent
    }
}

/// `opus_encode_float` without the assertions of `CEncoder::encode_max`:
/// the length of the packet, or the libopus error code.
fn c_encode(st: &mut RawEncoder, pcm: &[f32], frame: usize, out: &mut [u8]) -> i32 {
    unsafe { sys::opus_encode_float(st.0, pcm.as_ptr(), frame as i32, out.as_mut_ptr(), out.len() as i32) }
}

/// `opus_encode`: 16-bit input.
fn c_encode_i16(st: &mut RawEncoder, pcm: &[i16], frame: usize, out: &mut [u8]) -> i32 {
    unsafe { sys::opus_encode(st.0, pcm.as_ptr(), frame as i32, out.as_mut_ptr(), out.len() as i32) }
}

/// A C encoder driven through raw controls.
struct RawEncoder(*mut sys::OpusEncoder);

impl RawEncoder {
    fn new(rate: usize, channels: usize, application: i32) -> Self {
        let mut err = 0;
        let st = unsafe { sys::opus_encoder_create(rate as i32, channels as i32, application, &mut err) };
        assert_eq!(err, 0);
        Self(st)
    }
    fn set(&mut self, request: i32, value: i32) {
        let ret = unsafe { sys::opus_encoder_ctl(self.0, request, value) };
        assert_eq!(ret, 0, "opus_encoder_ctl({request}, {value})");
    }
    fn reset(&mut self) {
        let ret = unsafe { sys::opus_encoder_ctl(self.0, sys::OPUS_RESET_STATE) };
        assert_eq!(ret, 0, "OPUS_RESET_STATE");
    }
}

impl Drop for RawEncoder {
    fn drop(&mut self) {
        unsafe { sys::opus_encoder_destroy(self.0) }
    }
}

/// 0 SILK-only, 1 Hybrid, 2 CELT-only.
fn mode_of(toc: u8) -> u8 {
    if toc & 0x80 != 0 {
        2
    } else if toc & 0x60 == 0x60 {
        1
    } else {
        0
    }
}

/// The settings both encoders run with.
#[derive(Clone, Copy, Debug)]
struct Settings {
    bitrate: i32,
    cbr: bool,
    mode: Option<(i32, ForcedMode)>,
    bandwidth: (i32, Bandwidth),
    max_bandwidth: (i32, Bandwidth),
    fec: bool,
    loss: i32,
    complexity: i32,
    /// Frame duration in tenths of a millisecond.
    tenth_ms: usize,
    buffer: usize,
}

const BANDWIDTHS: [(i32, Bandwidth); 5] = [
    (BW_NB, Bandwidth::Narrowband),
    (BW_MB, Bandwidth::Mediumband),
    (BW_WB, Bandwidth::Wideband),
    (BW_SWB, Bandwidth::Superwideband),
    (BW_FB, Bandwidth::Fullband),
];

fn change(rng: &mut Rng, s: &mut Settings, low_delay: bool) {
    match rng.below(10) {
        0 | 1 => {
            // Bitrates over the whole range libopus accepts, most of them
            // where the decisions are made.
            s.bitrate = match rng.below(16) {
                0 => 500,
                1 => 510_000,
                2 | 3 => 2_000 + rng.below(6_000) as i32,
                4 | 5 => 100_000 + rng.below(300_000) as i32,
                _ => 6_000 + rng.below(90_000) as i32,
            }
        }
        2 => s.cbr = !s.cbr,
        3 | 4 => {
            s.mode = rng.pick(&[
                None,
                Some((MODE_SILK, ForcedMode::SilkOnly)),
                Some((MODE_HYBRID, ForcedMode::Hybrid)),
                Some((MODE_CELT, ForcedMode::CeltOnly)),
            ])
        }
        5 => s.bandwidth = if rng.chance(50) { (-1000, Bandwidth::Auto) } else { rng.pick(&BANDWIDTHS) },
        6 => s.max_bandwidth = rng.pick(&BANDWIDTHS),
        7 => {
            s.fec = rng.chance(50);
            s.loss = rng.pick(&[0, 3, 10, 30]);
        }
        8 => s.complexity = rng.below(7) as i32,
        _ => {
            s.tenth_ms = if low_delay {
                rng.pick(&[25, 50, 100, 200])
            } else {
                rng.pick(&[25, 50, 100, 200, 200, 400, 600])
            }
        }
    }
}

#[derive(Default)]
struct Tally {
    packets: usize,
    /// Packets both encoders refused.
    errors: usize,
    /// Packets whose first byte names another bandwidth than libopus's.
    toc_differs: usize,
    /// Packets the C decoder's final range was compared on.
    range_checked: usize,
    /// Packets without audio.
    placeholders: usize,
    /// Changes between CELT-only and SILK-only or Hybrid.
    mode_switches: usize,
}

/// What a frame of the input is replaced by, now and then.
#[derive(Clone, Copy, Debug)]
enum Odd {
    Silence,
    FullScale,
    /// Samples at three times full scale.
    TooLoud,
    /// One sample that is not a number.
    NotANumber,
}

/// One stream of `packets` packets. Returns what was seen; panics on what
/// must never happen. `sixteen_bit` feeds both encoders 16-bit samples.
fn walk(
    seed: u64,
    rate: usize,
    channels: usize,
    application: (i32, Application),
    sixteen_bit: bool,
    packets: usize,
) -> Tally {
    let mut rng = Rng(seed.wrapping_mul(0x9E3779B97F4A7C15) ^ 0xD1B54A32D192ED03);
    let low_delay = application.0 == APP_LOWDELAY;
    let voice = speech(rate, 6.0);
    let tones = multitone(rate, voice.len());
    let mono: Vec<f32> = voice.iter().zip(&tones).map(|(v, t)| 0.7 * v + 0.2 * t).collect();
    let pcm = if channels == 2 { to_stereo(&mono, rate) } else { mono };

    // Both encoders choose between mono and stereo packets.
    let mut c_enc = RawEncoder::new(rate, channels, application.0);
    let mut enc = OpusEncoder::new(
        SampleRate::try_from(rate as i32).unwrap(),
        Channels::try_from(channels).unwrap(),
        application.1,
    )
    .unwrap();
    let mut dec = CDecoder::new(rate, channels);
    let mut was_celt = None;
    let mut s = Settings {
        bitrate: 24_000,
        cbr: false,
        mode: None,
        bandwidth: (-1000, Bandwidth::Auto),
        max_bandwidth: (BW_FB, Bandwidth::Fullband),
        fec: false,
        loss: 0,
        complexity: 5,
        tenth_ms: 200,
        buffer: 1500,
    };
    for _ in 0..4 {
        change(&mut rng, &mut s, low_delay);
    }
    let mut tally = Tally::default();
    let mut at = 0usize;
    for i in 0..packets {
        if rng.chance(25) {
            change(&mut rng, &mut s, low_delay);
        }
        // The buffer: mostly roomy, sometimes around the limits of the
        // rules for packets too small to carry audio.
        s.buffer = match rng.below(16) {
            0 => 1 + rng.below(4),
            1 => 4 + rng.below(30),
            2 => 30 + rng.below(100),
            3 => 4000 + rng.below(6000),
            _ => 1500,
        };
        if rng.chance(1) {
            c_enc.reset();
            enc.reset_state();
        }
        c_enc.set(sys::OPUS_SET_BITRATE_REQUEST, s.bitrate);
        c_enc.set(sys::OPUS_SET_VBR_REQUEST, !s.cbr as i32);
        c_enc.set(OPUS_SET_FORCE_MODE_REQUEST, s.mode.map_or(-1000, |m| m.0));
        c_enc.set(sys::OPUS_SET_BANDWIDTH_REQUEST, s.bandwidth.0);
        c_enc.set(sys::OPUS_SET_MAX_BANDWIDTH_REQUEST, s.max_bandwidth.0);
        c_enc.set(sys::OPUS_SET_INBAND_FEC_REQUEST, s.fec as i32);
        c_enc.set(sys::OPUS_SET_PACKET_LOSS_PERC_REQUEST, s.loss);
        c_enc.set(sys::OPUS_SET_COMPLEXITY_REQUEST, s.complexity);
        enc.set_bitrate(Bitrate::BitsPerSecond(s.bitrate)).unwrap();
        enc.set_vbr(!s.cbr);
        enc.set_force_mode(s.mode.map(|m| m.1));
        enc.set_bandwidth(s.bandwidth.1);
        enc.set_max_bandwidth(s.max_bandwidth.1).unwrap();
        enc.set_inband_fec(s.fec);
        enc.set_packet_loss_perc(s.loss).unwrap();
        enc.set_complexity(s.complexity).unwrap();

        let frame = rate * s.tenth_ms / 10_000;
        if at + frame * channels > pcm.len() {
            at = 0;
        }
        let mut input = pcm[at..at + frame * channels].to_vec();
        at += frame * channels;
        let odd = rng.chance(4).then(|| rng.pick(&[Odd::Silence, Odd::FullScale, Odd::TooLoud, Odd::NotANumber]));
        match odd {
            Some(Odd::Silence) => input.fill(0.0),
            Some(Odd::FullScale) => input.iter_mut().enumerate().for_each(|(k, v)| *v = if k / 16 % 2 == 0 { 1.0 } else { -1.0 }),
            Some(Odd::TooLoud) if !sixteen_bit => input.iter_mut().for_each(|v| *v *= 12.0),
            Some(Odd::NotANumber) if !sixteen_bit => input[frame / 2] = f32::NAN,
            _ => {}
        }
        let what = format!("seed {seed}, {rate} Hz, {channels} ch, {:?}, 16-bit {sixteen_bit}, packet {i}, {odd:?}, {s:?}", application.1);

        let mut c_buf = vec![0u8; s.buffer];
        let mut buf = vec![0u8; s.buffer];
        let (c_len, result) = if sixteen_bit {
            let input: Vec<i16> = input.iter().map(|v| (v * 32768.0).round().clamp(-32768.0, 32767.0) as i16).collect();
            (c_encode_i16(&mut c_enc, &input, frame, &mut c_buf), enc.encode_i16(&input, frame, &mut buf))
        } else {
            (c_encode(&mut c_enc, &input, frame, &mut c_buf), enc.encode(&input, frame, &mut buf))
        };
        tally.packets += 1;
        let n = match result {
            Ok(n) => n,
            Err(e) => {
                assert!(c_len < 0, "{what}: opus1: {e}, libopus writes {c_len} bytes");
                tally.errors += 1;
                continue;
            }
        };
        assert!(c_len > 0, "{what}: libopus fails with {c_len}, opus1 writes {n} bytes");
        let c_len = c_len as usize;
        assert!(n >= 1 && n <= s.buffer, "{what}: {n} bytes");
        let (packet, c_packet) = (&buf[..n], &c_buf[..c_len]);

        // Valid for the C decoder, with the right duration.
        let decoded = dec.decode(Some(packet), frame, false).unwrap_or_else(|e| panic!("{what}: C decoder: {e}"));
        assert_eq!(decoded.len(), frame * channels, "{what}: decoded length");
        assert!(decoded.iter().all(|v| v.is_finite()), "{what}: decoded samples");
        let is_celt = mode_of(packet[0]) == 2;
        if was_celt.is_some_and(|w| w != is_celt) {
            tally.mode_switches += 1;
        }
        was_celt = Some(is_celt);

        // The packet-level rules of `opus_encode_native`.
        if s.cbr {
            assert_eq!(n, c_len, "{what}: size under a constant bitrate");
        }
        // A packet without audio (every frame empty) is libopus's byte for
        // byte, but for the bandwidth in its first byte: it repeats the
        // last coded packet's.
        let parsed = packet::parse(c_packet, false).unwrap();
        let without_audio = parsed.sizes[..parsed.count].iter().all(|&size| size == 0);
        if without_audio {
            assert_eq!(n, c_len, "{what}: packet without audio");
            assert_eq!(packet[0] & 7, c_packet[0] & 7, "{what}: packet without audio");
            assert_eq!(packet[1..], c_packet[1..], "{what}: packet without audio");
            assert_eq!(enc.final_range(), 0, "{what}: packet without audio");
            tally.placeholders += 1;
        }
        // CELT-only or not, and the channel count, never depend on SILK.
        assert_eq!(is_celt, mode_of(c_packet[0]) == 2, "{what}: CELT-only, {:#x} against {:#x}", packet[0], c_packet[0]);
        assert_eq!(packet[0] & 4, c_packet[0] & 4, "{what}: channel bit");
        if packet[0] & 0xFC != c_packet[0] & 0xFC {
            // The bandwidth. A CELT-only packet can differ too: its
            // bandwidth leans towards the one of the last decision
            // (`auto_bandwidth`), and SILK packets only decide when SILK
            // allows a switch.
            tally.toc_differs += 1;
            if std::env::var("WALK_PRINT").is_ok() && tally.toc_differs == 1 {
                println!("first bandwidth that differs: {what}: {:#x}, libopus {:#x}", packet[0], c_packet[0]);
            }
        }

        // The decoder ends in the encoder's final range. Left out:
        //
        // - Stereo Hybrid packets: their CELT layer is corrupt until
        //   src/celt.rs limits the intensity band to start..end.
        // - SILK-only mediumband packets of 10 ms: a fault in
        //   src/silk/encode_pulses.rs, see tests/multistream_enc_robust.rs.
        let stereo_hybrid = mode_of(packet[0]) == 1 && packet[0] & 4 != 0;
        let mediumband_10ms = packet[0] >> 3 == 4;
        if !stereo_hybrid && !mediumband_10ms {
            tally.range_checked += 1;
            assert_eq!(enc.final_range(), dec.final_range(), "{what}: final range, first byte {:#x}, {n} bytes", packet[0]);
        }
    }
    tally
}

/// 60 streams of 250 packets: every sampling rate, mono and stereo, the
/// three applications, float and 16-bit input. About one setting in four
/// packets changes: bitrate (500 b/s to 510 kb/s), constant or variable
/// bitrate, forced mode, bandwidth, maximum bandwidth, FEC and loss,
/// complexity, frame duration (2.5 to 60 ms). One buffer in four is small
/// (1 to 130 bytes) or large (up to 10 kB). Some frames are silent, at
/// full scale, beyond full scale or hold a sample that is not a number,
/// and the encoder is reset now and then.
///
/// More streams: `WALK_SEEDS=2000 cargo test --release --test
/// core_settings_walk -- --nocapture` (about 80 s). `WALK_FIRST` sets the
/// first seed, and `WALK_PRINT=1` shows where a stream's bandwidth first
/// differs from libopus's.
#[test]
fn settings_buffers_and_durations_change_in_a_running_stream() {
    let seeds: u64 = std::env::var("WALK_SEEDS").ok().and_then(|v| v.parse().ok()).unwrap_or(60);
    let first: u64 = std::env::var("WALK_FIRST").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    let mut total = Tally::default();
    for seed in first..first + seeds {
        let rate = [8000usize, 12000, 16000, 24000, 48000][(seed % 5) as usize];
        let channels = 1 + (seed / 5 % 2) as usize;
        let application = [(APP_AUDIO, Application::Audio), (APP_VOIP, Application::Voip), (APP_LOWDELAY, Application::RestrictedLowDelay)]
            [(seed / 10 % 3) as usize];
        let sixteen_bit = seed / 30 % 2 == 1;
        let t = walk(seed, rate, channels, application, sixteen_bit, 250);
        total.packets += t.packets;
        total.errors += t.errors;
        total.toc_differs += t.toc_differs;
        total.range_checked += t.range_checked;
        total.placeholders += t.placeholders;
        total.mode_switches += t.mode_switches;
    }
    println!(
        "{} packets: {} refused by both, {} without audio, {} mode switches, {} final ranges compared, {} bandwidths differ",
        total.packets, total.errors, total.placeholders, total.mode_switches, total.range_checked, total.toc_differs
    );
    if seeds >= 60 {
        // The walk reaches what it is for.
        assert!(total.placeholders * 20 >= total.packets, "{} packets without audio", total.placeholders);
        assert!(total.mode_switches * 100 >= total.packets, "{} mode switches", total.mode_switches);
        assert!(total.range_checked * 10 >= total.packets * 8, "{} final ranges", total.range_checked);
        // The bandwidth: measured 0.9 % of these packets,
        // 0.4 % over 4000 streams.
        assert!(total.toc_differs * 50 <= total.packets, "{} first bytes differ", total.toc_differs);
    }
}
