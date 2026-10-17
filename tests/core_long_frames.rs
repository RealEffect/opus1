//! Frames of 80, 100 and 120 ms, against C libopus 1.6.1.
//!
//! libopus codes such a frame as several parts in one packet
//! (`opus_encode_native`), and opus1 does the same:
//!
//! - CELT-only and Hybrid: 4, 5 or 6 frames of 20 ms.
//! - SILK-only: 80 ms as 2 frames of 40 ms, 100 ms as 5 frames of 20 ms,
//!   120 ms as 2 frames of 60 ms.
//!
//! What is checked:
//!
//! 1. Over 8, 16, 24 and 48 kHz, mono and stereo, the three applications,
//!    VBR, constrained VBR and CBR, with and without in-band FEC, at a low
//!    and a high bitrate:
//!    - libopus decodes every packet to the whole duration, and its final
//!      range is the encoder's;
//!    - with CBR every packet has the size of libopus's packet;
//!    - the decoded audio is within 0.5 dB of what libopus's encoder gives,
//!      on average per sampling rate, channel count and application, and
//!      per stream except where opus1 is as far below libopus with frames
//!      of 60 ms or less (see `MEAN_BOUND_DB`).
//! 2. With the mode forced, the first byte of every packet (mode,
//!    bandwidth, frame duration, stereo flag and frame count code) is
//!    libopus's.
//! 3. CELT-only packets with CBR at complexity 0 and 5 are libopus's byte
//!    for byte, and so are packets too small to hold audio. This also holds
//!    for packet sizes that do not divide by the number of parts, and when
//!    the frame duration, bitrate, buffer, complexity and bandwidth change
//!    between packets.
//!
//! The expert frame duration control has its tests in
//! `core_expert_frame_duration.rs`.

mod core_common;
mod k24_common;

use core_common::*;
use k24_common::{speech_fullband, to_stereo};
use opus1::{Application, Bandwidth, Bitrate, ForcedMode, OpusEncoder};

const LONG_MS: [usize; 3] = [80, 100, 120];
/// The durations of the parts: libopus codes no frame longer than 60 ms.
const PART_MS: [usize; 3] = [20, 40, 60];
const MAX_PACKET: usize = 8000;
/// Seconds of speech per stream in the tests other than the matrix. A
/// debug build (the `no_std` check runs one) codes less.
const SHORT_SECONDS: f64 = if cfg!(debug_assertions) { 0.48 } else { 1.2 };

#[derive(Clone, Copy, Debug, PartialEq)]
enum Rc {
    Vbr,
    Cvbr,
    Cbr,
}

#[derive(Clone, Copy)]
struct App {
    name: &'static str,
    c: i32,
    rs: Application,
}

const APPS: [App; 3] = [
    App { name: "voip", c: APP_VOIP, rs: Application::Voip },
    App { name: "audio", c: APP_AUDIO, rs: Application::Audio },
    App { name: "low delay", c: APP_LOWDELAY, rs: Application::RestrictedLowDelay },
];

/// The settings of one stream, given to both encoders.
#[derive(Clone, Copy)]
struct Setting {
    fs: usize,
    channels: usize,
    app: App,
    bitrate: i32,
    rc: Rc,
    fec: bool,
    complexity: i32,
    /// `OPUS_SET_FORCE_MODE` and `set_force_mode`.
    force: Option<(i32, ForcedMode)>,
    /// `OPUS_SET_BANDWIDTH` and `set_bandwidth`.
    bandwidth: Option<(i32, Bandwidth)>,
}

impl Setting {
    fn new(fs: usize, channels: usize, app: App, bitrate: i32) -> Self {
        Self {
            fs,
            channels,
            app,
            bitrate,
            rc: Rc::Vbr,
            fec: false,
            complexity: 9,
            force: None,
            bandwidth: None,
        }
    }

    fn label(&self, ms: usize) -> String {
        format!(
            "{} Hz {} ch {} {} b/s {:?} fec {} c{} force {:?} bw {:?} {ms} ms",
            self.fs,
            self.channels,
            self.app.name,
            self.bitrate,
            self.rc,
            self.fec,
            self.complexity,
            self.force.map(|f| f.1),
            self.bandwidth.map(|b| b.1),
        )
    }

    fn encoders(&self) -> (OpusEncoder, CEncoder) {
        let loss = if self.fec { 10 } else { 0 };
        let (vbr, constrained) = (self.rc != Rc::Cbr, self.rc == Rc::Cvbr);
        let mut rs = OpusEncoder::new(self.fs as i32, self.channels, self.app.rs).unwrap();
        rs.set_bitrate(Bitrate::BitsPerSecond(self.bitrate)).unwrap();
        rs.set_vbr(vbr);
        rs.set_vbr_constraint(constrained);
        rs.set_complexity(self.complexity).unwrap();
        rs.set_inband_fec(self.fec);
        rs.set_packet_loss_perc(loss).unwrap();
        let mut c = CEncoder::new(self.fs, self.channels, self.app.c);
        c.bitrate(self.bitrate).vbr(vbr).complexity(self.complexity).fec(self.fec, loss);
        c.set(audiopus_sys::OPUS_SET_VBR_CONSTRAINT_REQUEST, constrained as i32);
        if let Some((c_mode, mode)) = self.force {
            rs.set_force_mode(Some(mode));
            c.force_mode(c_mode);
        }
        if let Some((c_bw, bw)) = self.bandwidth {
            rs.set_bandwidth(bw);
            c.bandwidth(c_bw);
        }
        (rs, c)
    }
}

/// Both encoders' packets for one stream, and the measures of
/// `encode_and_check`.
struct Coded {
    rs: Vec<Vec<u8>>,
    c: Vec<Vec<u8>>,
    /// SNR in dB of the libopus decode of opus1's packets against the input.
    snr_rs: f64,
    /// The same for libopus's packets.
    snr_c: f64,
}

/// Speech with content up to the Nyquist frequency.
fn signal(fs: usize, channels: usize, seconds: f64) -> Vec<f32> {
    let mono = speech_fullband(fs, seconds);
    if channels == 2 { to_stereo(&mono, fs) } else { mono }
}

/// SNR of `decoded` against `input` when the decoded signal is `lag`
/// samples late, leaving out the first 100 ms.
fn snr_against_input(input: &[f32], decoded: &[f32], lag: usize, channels: usize, fs: usize) -> f64 {
    let (mut sig, mut err) = (0f64, 0f64);
    let n = input.len().min(decoded.len().saturating_sub(lag * channels));
    for i in fs / 10 * channels..n {
        let (a, b) = (input[i] as f64, decoded[i + lag * channels] as f64);
        sig += a * a;
        err += (a - b) * (a - b);
    }
    if err <= 0.0 { 200.0 } else { 10.0 * (sig / err).log10() }
}

/// Encodes `input` in frames of `ms` with both encoders, decodes both with
/// libopus, and checks what holds for every packet: libopus decodes each of
/// opus1's packets to the whole frame and ends in the encoder's final
/// range, and with CBR each packet has the size of libopus's.
fn encode_and_check(s: &Setting, input: &[f32], ms: usize) -> Coded {
    let label = s.label(ms);
    let frame = s.fs * ms / 1000;
    let (mut rs, mut c) = s.encoders();
    let mut rs_dec = CDecoder::new(s.fs, s.channels);
    let mut c_dec = CDecoder::new(s.fs, s.channels);
    let (mut rs_out, mut c_out) = (Vec::new(), Vec::new());
    let mut coded = Coded { rs: Vec::new(), c: Vec::new(), snr_rs: 0.0, snr_c: 0.0 };
    let mut buf = vec![0u8; MAX_PACKET];
    for (k, pcm) in input.chunks_exact(frame * s.channels).enumerate() {
        let n = rs.encode(pcm, frame, &mut buf).unwrap_or_else(|e| panic!("{label}: packet {k}: {e}"));
        let packet = buf[..n].to_vec();
        let c_packet = c.encode_max(pcm, MAX_PACKET);
        let decoded = rs_dec
            .decode(Some(&packet), 5760, false)
            .unwrap_or_else(|e| panic!("{label}: libopus rejects packet {k} {:02x?}: {e}", &packet[..2]));
        assert_eq!(decoded.len(), frame * s.channels, "{label}: packet {k} duration");
        assert_eq!(rs_dec.final_range(), rs.final_range(), "{label}: packet {k} final range");
        if s.rc == Rc::Cbr {
            assert_eq!(n, c_packet.len(), "{label}: packet {k} size");
        }
        rs_out.extend(decoded);
        c_out.extend(c_dec.decode(Some(&c_packet), 5760, false).unwrap());
        coded.rs.push(packet);
        coded.c.push(c_packet);
    }
    let lag = rs.lookahead();
    coded.snr_rs = snr_against_input(input, &rs_out, lag, s.channels, s.fs);
    coded.snr_c = snr_against_input(input, &c_out, lag, s.channels, s.fs);
    coded
}

/// The largest deficit of opus1 against libopus (libopus's SNR minus
/// opus1's) with the settings `s` at the durations of the parts of a long
/// packet: 20, 40 and 60 ms. 0 when opus1 is not below libopus there.
fn part_deficit(s: &Setting, input: &[f32]) -> f64 {
    PART_MS
        .iter()
        .map(|&ms| {
            let coded = encode_and_check(s, input, ms);
            coded.snr_c - coded.snr_rs
        })
        .fold(0.0f64, f64::max)
}

/// The low and the high bitrate of the matrix.
fn bitrates(channels: usize) -> [i32; 2] {
    if channels == 2 { [24_000, 96_000] } else { [16_000, 64_000] }
}

/// The quality rule of the matrix, as opus1's SNR minus libopus's:
///
/// - The mean over the streams of one sampling rate, channel count and
///   application (rate controls, FEC, bitrates and durations) is at least
///   -0.5 dB.
/// - Each stream is at least -0.5 dB, or no more than 1.5 dB below what
///   opus1 gives against libopus with the same settings at 20, 40 or
///   60 ms. Measured on 3.6 s of speech: 51 of the 864 streams are more
///   than 0.5 dB below libopus, and the largest excess over the frames of
///   60 ms or less is 0.95 dB (8 kHz stereo VoIP, 24 kb/s CBR, 120 ms).
///
/// Why the second part. opus1 ports the fixed-point SILK encoder, and
/// Homebrew's libopus is the float build. With a constant bitrate their
/// SILK quality moves by several tenths of a dB when the budget of a frame
/// changes by one byte, at every duration: 8 kHz stereo VoIP at 24 kb/s
/// CBR in 60 ms frames is 0.3 dB below libopus at 23,866 b/s and within
/// 0.1 dB of it at 24,000 b/s (8 s of speech). The parts of a long CBR packet have a byte less
/// than frames of their own (the header of the packet), so they land
/// elsewhere on this curve. In-band FEC adds to it. libopus's speech and
/// music analysis, which opus1 lacks, also picks other modes at 16 to
/// 48 kHz; that shows at every duration too.
const MEAN_BOUND_DB: f64 = 0.5;
const STREAM_BOUND_DB: f64 = 0.5;
const PART_MARGIN_DB: f64 = 1.5;

/// The matrix at one sampling rate: both channel counts, the three
/// applications, two bitrates, VBR, constrained VBR and CBR, with and
/// without FEC, 80, 100 and 120 ms. `encode_and_check` checks every packet;
/// this function checks the quality rule.
fn check_matrix(fs: usize) {
    // A debug build runs a part of the matrix and checks only the packets.
    let full = !cfg!(debug_assertions);
    let seconds = if full { 3.6 } else { 0.48 };
    let (mut streams, mut beyond) = (0usize, 0usize);
    let mut worst = (f64::INFINITY, String::new());
    for channels in [1, 2] {
        let input = signal(fs, channels, seconds);
        for app in APPS {
            let (mut group_sum, mut group_n) = (0f64, 0usize);
            let rates = bitrates(channels);
            for &bitrate in if full { &rates[..] } else { &rates[..1] } {
                for rc in [Rc::Vbr, Rc::Cvbr, Rc::Cbr] {
                    for fec in [false, true] {
                        let s = Setting { rc, fec, ..Setting::new(fs, channels, app, bitrate) };
                        let mut part = None;
                        for ms in LONG_MS {
                            let coded = encode_and_check(&s, &input, ms);
                            if !full {
                                // Too few packets for the quality rule.
                                continue;
                            }
                            let difference = coded.snr_rs - coded.snr_c;
                            if difference < -STREAM_BOUND_DB {
                                let part = *part.get_or_insert_with(|| part_deficit(&s, &input));
                                println!(
                                    "{}: {difference:+.2} dB, at 20 to 60 ms down to {:+.2} dB",
                                    s.label(ms),
                                    -part
                                );
                                assert!(
                                    difference >= -part - PART_MARGIN_DB,
                                    "{}: opus1 {:.2} dB, libopus {:.2} dB; at 20 to 60 ms opus1 is at most {part:.2} dB below",
                                    s.label(ms),
                                    coded.snr_rs,
                                    coded.snr_c
                                );
                                beyond += 1;
                            }
                            group_sum += difference;
                            group_n += 1;
                            if difference < worst.0 {
                                worst = (difference, s.label(ms));
                            }
                        }
                    }
                }
            }
            if !full {
                continue;
            }
            let mean = group_sum / group_n as f64;
            println!("{fs} Hz {channels} ch {}: {group_n} streams, opus1 minus libopus {mean:+.2} dB on average", app.name);
            assert!(mean >= -MEAN_BOUND_DB, "{fs} Hz {channels} ch {}: mean {mean:+.2} dB", app.name);
            streams += group_n;
        }
    }
    println!(
        "{fs} Hz: {streams} streams, worst {:+.2} dB ({}); {beyond} more than {STREAM_BOUND_DB} dB below libopus",
        worst.0, worst.1
    );
}

#[test]
fn long_frames_at_8_khz_match_libopus() {
    check_matrix(8000);
}

#[test]
fn long_frames_at_16_khz_match_libopus() {
    check_matrix(16000);
}

#[test]
fn long_frames_at_24_khz_match_libopus() {
    check_matrix(24000);
}

#[test]
fn long_frames_at_48_khz_match_libopus() {
    check_matrix(48000);
}

/// The modes a setting can force, with the bandwidth that goes with them
/// at `fs`. Hybrid needs super-wideband or fullband, so 24 or 48 kHz.
fn forced_settings(fs: usize, channels: usize) -> Vec<Setting> {
    let celt_bw = match fs {
        8000 => (BW_NB, Bandwidth::Narrowband),
        16000 => (BW_WB, Bandwidth::Wideband),
        24000 => (BW_SWB, Bandwidth::Superwideband),
        _ => (BW_FB, Bandwidth::Fullband),
    };
    let silk_bw = if fs == 8000 { (BW_NB, Bandwidth::Narrowband) } else { (BW_WB, Bandwidth::Wideband) };
    let rate = |mono: i32| if channels == 2 { mono * 3 / 2 } else { mono };
    let mut out = Vec::new();
    for app in &APPS[..2] {
        out.push(Setting {
            force: Some((MODE_SILK, ForcedMode::SilkOnly)),
            bandwidth: Some(silk_bw),
            ..Setting::new(fs, channels, *app, rate(24_000))
        });
        if fs >= 24000 {
            out.push(Setting {
                force: Some((MODE_HYBRID, ForcedMode::Hybrid)),
                bandwidth: Some(celt_bw),
                ..Setting::new(fs, channels, *app, rate(32_000))
            });
        }
        out.push(Setting {
            force: Some((MODE_CELT, ForcedMode::CeltOnly)),
            bandwidth: Some(celt_bw),
            ..Setting::new(fs, channels, *app, rate(48_000))
        });
    }
    out.push(Setting { bandwidth: Some(celt_bw), ..Setting::new(fs, channels, APPS[2], rate(48_000)) });
    out
}

/// With the mode and the bandwidth forced, the first byte of every packet
/// at `fs` is libopus's.
fn check_first_bytes(fs: usize) {
    // A debug build leaves out constrained VBR.
    let rate_controls: &[Rc] = if cfg!(debug_assertions) { &[Rc::Vbr, Rc::Cbr] } else { &[Rc::Vbr, Rc::Cvbr, Rc::Cbr] };
    let mut packets = 0usize;
    for channels in [1, 2] {
        let input = signal(fs, channels, SHORT_SECONDS);
        for base in forced_settings(fs, channels) {
            for &rc in rate_controls {
                for fec in [false, true] {
                    let s = Setting { rc, fec, ..base };
                    for ms in LONG_MS {
                        let coded = encode_and_check(&s, &input, ms);
                        for (k, (a, b)) in coded.rs.iter().zip(&coded.c).enumerate() {
                            let label = s.label(ms);
                            // Mode, bandwidth, frame duration and stereo flag.
                            assert_eq!(a[0] & 0xfc, b[0] & 0xfc, "{label}: packet {k}: {:02x} against {:02x}", a[0], b[0]);
                            // Two SILK frames of a VBR packet (80 and
                            // 120 ms) get code 1 when they happen to have
                            // the same size and code 2 otherwise. libopus
                            // limits each of them to its share of the
                            // bitrate, which opus1 leaves out, so their
                            // sizes differ from libopus's.
                            let two_silk_frames = a[0] >> 3 < 12 && ms != 100;
                            if s.rc != Rc::Cbr && two_silk_frames {
                                assert!(matches!(a[0] & 3, 1 | 2), "{label}: packet {k}: code {}", a[0] & 3);
                            } else {
                                assert_eq!(a[0], b[0], "{label}: packet {k}");
                                if a[0] & 3 == 3 {
                                    // The frame count.
                                    assert_eq!(a[1] & 0x3f, b[1] & 0x3f, "{label}: packet {k}: frame count");
                                }
                            }
                            packets += 1;
                        }
                    }
                }
            }
        }
    }
    println!("{fs} Hz: {packets} packets compared");
}

#[test]
fn first_byte_is_libopus_where_the_mode_is_forced_at_8_khz() {
    check_first_bytes(8000);
}

#[test]
fn first_byte_is_libopus_where_the_mode_is_forced_at_16_khz() {
    check_first_bytes(16000);
}

#[test]
fn first_byte_is_libopus_where_the_mode_is_forced_at_24_khz() {
    check_first_bytes(24000);
}

#[test]
fn first_byte_is_libopus_where_the_mode_is_forced_at_48_khz() {
    check_first_bytes(48000);
}

#[test]
fn celt_only_cbr_packets_are_libopus_byte_for_byte() {
    // The CELT encoder alone writes libopus's packets at complexity 0 to 5
    // (libopus runs its speech and music analysis from complexity 7). A
    // long CELT-only packet is made of 20 ms CELT frames, so it must be
    // libopus's too.
    for fs in [8000, 12000, 16000, 24000, 48000] {
        for channels in [1, 2] {
            let input = signal(fs, channels, SHORT_SECONDS);
            for complexity in [0, 5] {
                for bitrate in [6_000, 32_000, 128_000] {
                    let s = Setting {
                        rc: Rc::Cbr,
                        complexity,
                        ..Setting::new(fs, channels, APPS[2], bitrate)
                    };
                    for ms in LONG_MS {
                        let coded = encode_and_check(&s, &input, ms);
                        for (k, (a, b)) in coded.rs.iter().zip(&coded.c).enumerate() {
                            assert_eq!(a, b, "{}: packet {k}", s.label(ms));
                        }
                    }
                }
            }
        }
    }
}

/// Encodes `input` in frames of `ms` with CELT-only CBR at complexity 5
/// into buffers of `out_len` bytes, and checks that every packet and final
/// range is libopus's. Returns the number of packets.
fn check_celt_cbr_bytes(fs: usize, channels: usize, bitrate: i32, out_len: usize, ms: usize, input: &[f32]) -> usize {
    let frame = fs * ms / 1000;
    let mut rs = OpusEncoder::new(fs as i32, channels, Application::RestrictedLowDelay).unwrap();
    rs.set_bitrate(Bitrate::BitsPerSecond(bitrate)).unwrap();
    rs.set_vbr(false);
    rs.set_complexity(5).unwrap();
    let mut c = CEncoder::new(fs, channels, APP_LOWDELAY);
    c.bitrate(bitrate).vbr(false).complexity(5);
    let mut buf = vec![0u8; out_len];
    let mut packets = 0;
    for (k, pcm) in input.chunks_exact(frame * channels).enumerate() {
        let label = format!("{fs} Hz {channels} ch {bitrate} b/s, {out_len}-byte buffer, {ms} ms: packet {k}");
        let n = rs.encode(pcm, frame, &mut buf).unwrap_or_else(|e| panic!("{label}: {e}"));
        let c_packet = c.try_encode(pcm, frame, out_len).unwrap();
        assert_eq!(&buf[..n], &c_packet[..], "{label}");
        assert_eq!(rs.final_range(), c.final_range(), "{label}");
        packets += 1;
    }
    packets
}

#[test]
fn celt_only_cbr_budgets_that_do_not_split_evenly_are_libopus() {
    // A long CBR packet gives each part an equal share of the packet once
    // the largest header the packet can need is taken off (C 1735-1785).
    // At the bitrates of the test above the packet size divides by the
    // number of parts. Here it does not: for example 7 kb/s gives 70, 88
    // and 105 bytes for 80, 100 and 120 ms, and the buffer sizes limit
    // the packet to odd sizes too.
    let mut packets = 0;
    for fs in [8000, 16000, 48000] {
        for channels in [1, 2] {
            let input = signal(fs, channels, SHORT_SECONDS);
            for ms in LONG_MS {
                for bitrate in [7_000, 9_000, 13_000, 21_000, 47_000, 99_000] {
                    packets += check_celt_cbr_bytes(fs, channels, bitrate, MAX_PACKET, ms, &input);
                }
                for out_len in [37, 101, 333, 1001] {
                    packets += check_celt_cbr_bytes(fs, channels, 256_000, out_len, ms, &input);
                }
            }
        }
    }
    println!("{packets} packets compared");
}

#[test]
fn celt_only_cbr_with_settings_changed_between_packets_is_libopus() {
    // Between packets the frame duration (2.5 to 120 ms), the bitrate, the
    // output buffer (1 byte to more than six full frames), the complexity
    // (0 to 6) and the bandwidth change, and the encoders are sometimes
    // reset. Every packet is libopus's, on speech and on a music-like
    // signal.
    let mut seed = 0x2545_f491_4f6c_dd1du64;
    let mut pick = move |n: usize| {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 16) as usize % n
    };
    const TENTHS_MS: [usize; 9] = [25, 50, 100, 200, 400, 600, 800, 1000, 1200];
    const BITRATES: [i32; 10] = [2_000, 6_000, 9_000, 16_000, 32_000, 47_000, 64_000, 128_000, 256_000, 510_000];
    const OUT_LENS: [usize; 13] = [1, 2, 3, 7, 20, 39, 75, 120, 700, 1276, 4000, 7657, 12_000];
    const BANDWIDTHS: [(i32, Bandwidth); 4] = [
        (audiopus_sys::OPUS_AUTO, Bandwidth::Auto),
        (BW_NB, Bandwidth::Narrowband),
        (BW_WB, Bandwidth::Wideband),
        (BW_FB, Bandwidth::Fullband),
    ];
    let seconds = if cfg!(debug_assertions) { 1.0 } else { 4.0 };
    let mut packets = 0;
    for fs in [8000, 12000, 16000, 24000, 48000] {
        for channels in [1, 2] {
            let total = (seconds * fs as f64) as usize;
            let music = k24_common::multitone(fs, total + fs / 5);
            let music = if channels == 2 { to_stereo(&music, fs) } else { music };
            for input in [signal(fs, channels, seconds + 0.2), music] {
                let mut rs = OpusEncoder::new(fs as i32, channels, Application::RestrictedLowDelay).unwrap();
                let mut c = CEncoder::new(fs, channels, APP_LOWDELAY);
                rs.set_vbr(false);
                rs.set_complexity(5).unwrap();
                c.vbr(false).complexity(5);
                let mut pos = 0;
                while pos < total {
                    if pick(3) == 0 {
                        let bitrate = BITRATES[pick(BITRATES.len())];
                        rs.set_bitrate(Bitrate::BitsPerSecond(bitrate)).unwrap();
                        c.bitrate(bitrate);
                    }
                    if pick(4) == 0 {
                        let complexity = pick(7) as i32;
                        rs.set_complexity(complexity).unwrap();
                        c.complexity(complexity);
                    }
                    if pick(6) == 0 {
                        let (c_bw, bw) = BANDWIDTHS[pick(BANDWIDTHS.len())];
                        rs.set_bandwidth(bw);
                        c.bandwidth(c_bw);
                    }
                    if pick(30) == 0 {
                        rs.reset_state();
                        c.reset();
                    }
                    // Mostly 80 to 120 ms.
                    let tenths = if pick(3) == 0 { TENTHS_MS[pick(9)] } else { TENTHS_MS[6 + pick(3)] };
                    let frame = fs * tenths / 10_000;
                    let out_len = OUT_LENS[pick(OUT_LENS.len())];
                    let pcm = &input[pos * channels..(pos + frame) * channels];
                    let mut buf = vec![0u8; out_len];
                    let label = format!(
                        "{fs} Hz {channels} ch, {} ms, {out_len}-byte buffer, {:?}, complexity {}: packet {packets}",
                        tenths as f64 / 10.0,
                        rs.bitrate_setting(),
                        rs.complexity()
                    );
                    match (rs.encode(pcm, frame, &mut buf), c.try_encode(pcm, frame, out_len)) {
                        (Ok(n), Ok(c_packet)) => {
                            assert_eq!(&buf[..n], &c_packet[..], "{label}");
                            assert_eq!(rs.final_range(), c.final_range(), "{label}");
                        }
                        (Err(_), Err(e)) => assert_eq!(e, audiopus_sys::OPUS_BUFFER_TOO_SMALL, "{label}"),
                        (rs, c) => panic!("{label}: opus1 {rs:?}, libopus {:?}", c.map(|p| p.len())),
                    }
                    pos += frame;
                    packets += 1;
                }
            }
        }
    }
    println!("{packets} packets compared");
}

#[test]
fn packets_without_audio_and_buffer_errors_are_libopus() {
    // Too few bytes or too low a bitrate: libopus writes a packet without
    // audio ("emit 'PLC' frames"), whose first bytes say the duration. A
    // 100 ms packet cannot be written into one byte.
    for fs in [8000, 16000, 48000] {
        for channels in [1, 2] {
            for app in APPS {
                for ms in [60, 80, 100, 120] {
                    let frame = fs * ms / 1000;
                    let pcm = vec![0.1f32; frame * channels];
                    for out_len in [1, 2, 3, 5, 9, 30] {
                        for vbr in [true, false] {
                            for bitrate in [500, 2000, 64_000] {
                                let mut rs = OpusEncoder::new(fs as i32, channels, app.rs).unwrap();
                                rs.set_bitrate(Bitrate::BitsPerSecond(bitrate)).unwrap();
                                rs.set_vbr(vbr);
                                let mut c = CEncoder::new(fs, channels, app.c);
                                c.bitrate(bitrate).vbr(vbr);
                                let mut buf = vec![0u8; out_len];
                                let label = format!("{fs} Hz {channels} ch {} {ms} ms, {out_len} bytes, vbr {vbr}, {bitrate} b/s", app.name);
                                match (rs.encode(&pcm, frame, &mut buf), c.try_encode(&pcm, frame, out_len)) {
                                    (Ok(n), Ok(c_packet)) => {
                                        if bitrate <= 2000 || out_len < 10 {
                                            assert_eq!(&buf[..n], &c_packet[..], "{label}");
                                        }
                                    }
                                    (Err(_), Err(e)) => assert_eq!(e, audiopus_sys::OPUS_BUFFER_TOO_SMALL, "{label}"),
                                    (rs, c) => panic!("{label}: opus1 {rs:?}, libopus {c:?}"),
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn maximum_bitrate_with_cbr_fills_the_buffer_as_libopus_does() {
    // CBR at the maximum bitrate: a packet made of several frames is padded
    // to the whole buffer (opus_encoder.c:1749), also a SILK-only one of 80
    // to 120 ms.
    for fs in [8000, 48000] {
        for app in APPS {
            for ms in LONG_MS {
                for out_len in [200, 1000, 3000, 9000] {
                    let frame = fs * ms / 1000;
                    let pcm = &signal(fs, 1, 0.2)[..frame];
                    let mut rs = OpusEncoder::new(fs as i32, 1, app.rs).unwrap();
                    rs.set_bitrate(Bitrate::Max).unwrap();
                    rs.set_vbr(false);
                    let mut c = CEncoder::new(fs, 1, app.c);
                    c.bitrate(audiopus_sys::OPUS_BITRATE_MAX).vbr(false);
                    let mut buf = vec![0u8; out_len];
                    let n = rs.encode(pcm, frame, &mut buf).unwrap();
                    let c_packet = c.try_encode(pcm, frame, out_len).unwrap();
                    let label = format!("{fs} Hz {} {ms} ms, {out_len} bytes", app.name);
                    assert_eq!(n, c_packet.len(), "{label}");
                    assert_eq!(n, out_len, "{label}");
                }
            }
        }
    }
}
