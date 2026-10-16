//! The multistream encoder against C libopus when libopus does not use its
//! analysis of speech against music.
//!
//! libopus runs that analysis from complexity 7 on (opus_encoder.c:1252).
//! It then picks modes and bandwidths that opus1, which has no such
//! analysis, cannot pick. tests/multistream_e2e_matrix.rs runs at the
//! default complexity 9, so its limits leave room for that. The tests here
//! run at complexity 6 and below. Both encoders then go by the same rules,
//! and the limits can be strict.
//!
//! The signal is another one than in the other end-to-end tests: `music`
//! has content up to the Nyquist frequency, repeated attacks, and channels
//! of different levels, so that the energy masks are far from zero.
//!
//! What the matrix checks for every setting (all layouts of the other
//! matrix, 48, 24 and 16 kHz, 2.5 to 60 ms, variable and constant bitrate,
//! 12, 32 and 64 kb/s per channel):
//!
//! - Structure. In a surround layout every stream packet has the mode, the
//!   audio bandwidth and the channel count of the libopus stream. That
//!   includes the streams that choose their own mode. Outside surround mode
//!   a few packets differ, for two reasons that lie in the stream encoder
//!   (see the list below).
//! - Bytes per stream with variable bitrate. Up to 20 ms a coupled stream
//!   and the LFE stream of a surround layout use the bytes of the libopus
//!   stream within 3 % (measured: within 0.8 %). This shows that the
//!   energy mask and the LFE flag act in the stream encoder as they do in
//!   libopus.
//! - Constant bitrate: the packets have the size of the libopus packets.
//! - Quality: the SNR of every channel against the libopus encoder, both
//!   decoded by libopus, and the two decoders against each other.
//!
//! The last tests take both encoders through the same changes of settings
//! in a running stream: bitrate, variable or constant bitrate, frame
//! duration, complexity, forced mode, bandwidth and resets. After every
//! frame the streams are compared.
//!
//! Differences that remain, all in the stream encoder (`OpusEncoder`):
//!
//! - libopus codes a stereo stream as mono at low bitrates, outside
//!   surround mode. opus1 always codes two channels. The stereo layout
//!   and the 6-channel layout of family 2 show it. With one channel
//!   libopus also picks a wider audio bandwidth.
//! - A stream that changes between CELT and SILK or Hybrid does so one
//!   packet earlier than in libopus, which first sends a frame with
//!   redundancy.
//! - 40 and 60 ms packets of CELT and Hybrid streams are up to 27 % larger
//!   than in libopus with variable bitrate (opus_encoder.c:1785).
//! - A Hybrid stream with an energy mask: libopus codes up to 8.4 times
//!   the bitrate of the stream, opus1 up to 1.25 times. See
//!   `OpusEncoder::set_energy_mask`.
//! - The channels of a stereo CELT stream are 0.5 to 2.0 dB below libopus
//!   on average on this signal: 0.4 dB at 64 kb/s per channel, 1.5 to
//!   2.2 dB at 32 kb/s, and up to 8.3 dB in single settings at 16 kHz.
//!   Mono streams are within 0.7 dB on average. The gap is the same at
//!   complexity 4, where neither encoder uses the pitch pre-filter.
//!
//! Two faults of the stream encoder are known and were tried in a copy of
//! the source. With the pulse search for 2 to 4 pulses on aarch64 changed
//! to compare what libopus compares (src/pvq.rs), the average gap over the
//! whole matrix goes from 0.28 to 0.13 dB for mono channels, from 1.11 to
//! 1.00 dB for the channels of stereo streams, and from 0.18 dB below
//! libopus to 0.21 dB above for the LFE channel. With the intensity band
//! limited to the first coded band (src/celt.rs) no number changes. So
//! about 1 dB of the stereo gap has another cause.
//!
//! With the environment variable `MS_E2E_CASES` set, the matrix prints one
//! line per setting (run with `--nocapture`).
#![cfg(feature = "std")]

mod ms_support;
mod multistream_e2e_common;
mod multistream_enc_common;

use ms_support::*;
use multistream_e2e_common::*;
use multistream_enc_common::*;
use opus1::multistream::encoder::Bitrate;
use opus1::{Application, ForcedMode};

/// libopus runs its analysis from complexity 7 on.
const COMPLEXITY: i32 = 6;

/// Signal length per setting in seconds. The first 0.2 s are not measured.
const SECONDS: f64 = 0.6;

/// Level of channel `c` of `music` against the loudest channel.
const LEVELS: [f32; 9] = [1.0, 0.35, 0.7, 0.1, 0.2, 0.5, 0.05, 0.8, 0.15];

/// A music-like signal with content up to the Nyquist frequency. Every
/// channel holds three notes with eight overtones each. The notes are
/// struck anew four times a second, each channel a little later than the
/// one before. Under them lies white noise, about 30 dB below. The LFE
/// channel holds a 35 Hz tone that is struck twice a second.
///
/// The channels differ in level by up to 26 dB (`LEVELS`), so that loud
/// channels mask quiet ones and the energy masks of a surround layout are
/// far from zero.
fn music(fs: i32, case: &Case, samples: usize) -> Vec<f32> {
    let channels = case.channels;
    let lfe = lfe_channel(channels, case.tone_family());
    let two_pi = 2.0 * std::f64::consts::PI;
    let nyquist = fs as f64 / 2.0;
    let hiss = noise(channels, samples, 0.006, 31);
    let mut pcm = vec![0f32; samples * channels];
    for (i, frame) in pcm.chunks_exact_mut(channels).enumerate() {
        let t = i as f64 / fs as f64;
        for (c, sample) in frame.iter_mut().enumerate() {
            if Some(c) == lfe {
                let since = (t * 2.0).fract() / 2.0;
                let env = (-since * 9.0).exp();
                *sample = (0.4 * env * (two_pi * 35.0 * t).sin()) as f32;
                continue;
            }
            let since = ((t + 0.031 * c as f64) * 4.0).fract() / 4.0;
            let env = 0.15 + 0.85 * (-since * 14.0).exp();
            let mut v = 0.0;
            for (n, base) in [220.0, 277.2, 329.6].iter().enumerate() {
                let f0 = base * (1.0 + 0.11 * c as f64) * (1.0 + 0.5 * n as f64);
                for h in 1..=8 {
                    // The overtones are stretched, so that they do not
                    // fall on one pitch.
                    let f = f0 * h as f64 * (1.0 + 0.9 * (h - 1) as f64);
                    if f < 0.92 * nyquist {
                        v += (two_pi * f * t + (c + n) as f64).sin() / (h as f64).sqrt();
                    }
                }
            }
            *sample = LEVELS[c % LEVELS.len()]
                * ((0.035 * env * v) as f32 + hiss[i * channels + c]);
        }
    }
    pcm
}

/// Whether the layout is coded in surround mode: family 1 with more than
/// two channels.
fn is_surround(case: &Case) -> bool {
    case.tone_family() == 1 && case.channels > 2
}

/// The limits that hold for every single setting.
fn check_setting(m: &Measured, wrong: &mut Vec<String>) {
    let label = m.setting.label();
    let mut fail = |text: String| wrong.push(format!("{label}: {text}"));
    let example = |kind: &str| {
        m.examples
            .iter()
            .find(|e| e.ends_with(kind))
            .cloned()
            .unwrap_or_default()
    };
    let case = &m.setting.case;
    let surround = is_surround(case);
    let per_channel = m.setting.bitrate / case.channels as i32;
    let long = m.setting.frame_size * 25 >= m.setting.fs as usize;

    // --- Structure ---
    let s = &m.structure;
    if s.fixed_mode != 0 {
        fail(format!(
            "{} stream packets with another mode than libopus fixes: {}",
            s.fixed_mode,
            example("fixed mode")
        ));
    }
    if surround {
        if s.free_mode != 0 || s.bandwidth != 0 || s.channels != 0 {
            fail(format!(
                "{} stream packets with another mode, {} with another bandwidth, {} with another \
                 channel count: {} {} {}",
                s.free_mode,
                s.bandwidth,
                s.channels,
                example(": mode"),
                example("bandwidth"),
                example("channels")
            ));
        }
    } else {
        // Measured: at most 3 packets of a setting change their mode one
        // packet earlier than in libopus (family 255, 12 kb/s per channel,
        // constant bitrate).
        if s.free_mode > 6 {
            fail(format!(
                "{} stream packets with another mode: {}",
                s.free_mode,
                example(": mode")
            ));
        }
        // Another bandwidth comes with a stereo stream that libopus codes
        // as mono. Measured otherwise: up to 5 packets of a setting, SILK
        // streams of family 255 that reach the next bandwidth a little
        // earlier or later than in libopus.
        if s.channels == 0 && s.bandwidth > 10 {
            fail(format!(
                "{} stream packets with another bandwidth: {}",
                s.bandwidth,
                example("bandwidth")
            ));
        }
        // Only a layout with a stereo stream can differ in channel count.
        let has_stereo_stream = m.streams.iter().any(|st| st.kind == StreamKind::Coupled);
        if s.channels != 0 && !has_stereo_stream {
            fail(format!("{} stream packets with another channel count", s.channels));
        }
    }
    // A stream with audio in one encoder only. Measured in one setting:
    // 7.1 at 16 kHz, 10 ms, constant bitrate, 12 kb/s per channel, where
    // the SILK centre stream of opus1 takes a few bytes more and the LFE
    // stream is left none (23 and 10 of 300 stream packets).
    if s.audio + s.empty_toc != 0 {
        let expected = !m.setting.vbr && per_channel == 12_000;
        if !expected || s.audio + s.empty_toc > 60 {
            fail(format!(
                "{} stream packets with audio in one encoder only, {} without audio and another \
                 TOC byte: {} {}",
                s.audio,
                s.empty_toc,
                example("audio in one only"),
                example("without audio")
            ));
        }
    }

    // --- Bitrates ---
    for (i, stream) in m.streams.iter().enumerate() {
        let last = i + 1 == m.streams.len();
        // With constant bitrate the last stream is set to what the others
        // left, so its setting is not compared.
        if (m.setting.vbr || !last) && stream.set[RUST] != stream.set[C] {
            fail(format!(
                "stream {i} is set to {} b/s, in libopus to {} b/s",
                stream.set[RUST], stream.set[C]
            ));
        }
        if !m.setting.vbr {
            continue;
        }
        // Bytes of the stream against the libopus stream.
        let ratio = stream.used[RUST] / stream.used[C].max(1.0);
        // A Hybrid stream with an energy mask: libopus codes many times
        // the bitrate of the stream (see the head of this file). Measured
        // on the centre stream at 12 kb/s per channel with 10 and 20 ms
        // frames: libopus up to 8.4 times the bitrate of the stream,
        // opus1 up to 1.25 times. With all channels at one level libopus
        // reaches 30 times and opus1 1.36 times.
        let libopus_far_above = surround
            && stream.kind == StreamKind::Mono
            && m.setting.frame_size * 100 >= m.setting.fs as usize
            && stream.used[C] > 1.5 * stream.set[C] as f64;
        if libopus_far_above {
            println!(
                "{label}: stream {i} is set to {} b/s: libopus codes {:.0} b/s, opus1 {:.0} b/s",
                stream.set[C], stream.used[C], stream.used[RUST]
            );
            if stream.used[RUST] > 1.45 * stream.set[RUST] as f64 {
                fail(format!(
                    "stream {i} ({}): {:.0} b/s, set to {} b/s",
                    stream.kind.name(),
                    stream.used[RUST],
                    stream.set[RUST]
                ));
            }
            continue;
        }
        let short = m.setting.frame_size * 100 < m.setting.fs as usize;
        let limits = if long {
            // 40 and 60 ms. Measured: 0.961 to 1.239.
            0.93..=1.30
        } else if !surround {
            // Measured: 0.877 to 1.047. The low end is SILK at 12 kb/s.
            0.85..=1.08
        } else if stream.kind != StreamKind::Mono {
            // Coupled and LFE streams, always CELT. Measured: 0.994 to
            // 1.008.
            0.97..=1.03
        } else if short {
            // The centre stream with 2.5 and 5 ms frames, always CELT.
            // Measured: 0.998 to 1.012.
            0.97..=1.04
        } else {
            // The centre stream with 10 and 20 ms frames, which is SILK at
            // 12 kb/s per channel. Measured: 0.975 to 1.104.
            0.90..=1.14
        };
        if !limits.contains(&ratio) {
            fail(format!(
                "stream {i} ({}): {:.0} b/s, libopus {:.0} b/s",
                stream.kind.name(),
                stream.used[RUST],
                stream.used[C]
            ));
        }
    }
    if !m.setting.vbr && m.rate[RUST] != m.rate[C] {
        fail(format!(
            "constant bitrate: {:.0} b/s, libopus {:.0} b/s",
            m.rate[RUST], m.rate[C]
        ));
    }

    // --- Quality and the decoder ---
    for (ch, row) in m.channels.iter().enumerate() {
        let gap = row.snr[RUST_TO_C] - row.snr[C_TO_C];
        // Measured at worst: mono -3.9 dB, coupled -8.3 dB, LFE -6.1 dB.
        let limit = match row.kind {
            StreamKind::Mono => -5.5,
            StreamKind::Coupled => -10.0,
            StreamKind::Lfe => -7.5,
        };
        if gap < limit {
            fail(format!(
                "channel {ch} ({}): {:.1} dB, libopus {:.1} dB",
                row.kind.name(),
                row.snr[RUST_TO_C],
                row.snr[C_TO_C]
            ));
        }
        // The two decoders give the same SNR for the same packets.
        for (ours, theirs, whose) in [
            (RUST_TO_RUST, RUST_TO_C, "opus1"),
            (C_TO_RUST, C_TO_C, "libopus"),
        ] {
            if (row.snr[ours] - row.snr[theirs]).abs() > 0.2 {
                fail(format!(
                    "channel {ch}: the opus1 decoder gives {:.2} dB on {whose} packets, the \
                     libopus decoder {:.2} dB",
                    row.snr[ours], row.snr[theirs]
                ));
            }
        }
    }
}

/// The limits on the 18 settings of one layout together: the average gap
/// to libopus per kind of stream. Measured at worst: mono -0.6 dB, coupled
/// -2.0 dB (the stereo stream of family 2 with 6 channels at 24 kHz), LFE
/// -0.9 dB.
fn check_layout(name: &str, results: &[Measured], wrong: &mut Vec<String>) {
    for kind in StreamKind::ALL {
        let limit = match kind {
            StreamKind::Mono => -1.2,
            StreamKind::Coupled => -2.7,
            StreamKind::Lfe => -1.6,
        };
        if let Some(gap) = snr_differences(results, kind, RUST_TO_C, C_TO_C)
            && gap.mean < limit
        {
            wrong.push(format!(
                "{name}: {} channels are {:.1} dB below libopus on average",
                kind.name(),
                -gap.mean
            ));
        }
    }
}

/// One line of text for a setting: rates, stream bytes, structure and the
/// SNR of every channel on the four paths. Printed when the environment
/// variable `MS_E2E_CASES` is set.
fn case_line(m: &Measured) -> String {
    let snr: Vec<String> = m
        .channels
        .iter()
        .map(|ch| {
            format!(
                "{} {:.1}/{:.1}/{:.1}/{:.1}",
                ch.kind.name(),
                ch.snr[RUST_TO_C],
                ch.snr[C_TO_C],
                ch.snr[RUST_TO_RUST],
                ch.snr[C_TO_RUST]
            )
        })
        .collect();
    let rates: Vec<String> = m
        .streams
        .iter()
        .map(|s| format!("{:.0}/{:.0}", s.used[RUST], s.used[C]))
        .collect();
    format!(
        "CASE {}: rate {:.0}/{:.0}; streams {}; {:?}; snr {}",
        m.setting.label(),
        m.rate[RUST],
        m.rate[C],
        rates.join(" "),
        m.structure,
        snr.join(", ")
    )
}

/// Runs the layouts at one sampling rate with variable or constant bitrate
/// and checks the limits.
fn assert_run(fs: i32, vbr: bool, layouts: &[Case]) {
    let mut wrong = Vec::new();
    for case in layouts {
        let mut results = Vec::new();
        for frame_size in frame_sizes(fs) {
            for per_channel in RATES_PER_CHANNEL {
                let setting = Setting {
                    case: *case,
                    fs,
                    application: Application::Audio,
                    frame_size,
                    vbr,
                    bitrate: per_channel * case.channels as i32,
                    complexity: COMPLEXITY,
                };
                let frames = ((SECONDS * fs as f64) as usize / frame_size).max(1);
                let pcm = music(fs, case, frames * frame_size);
                let m = measure_signal(setting, &pcm);
                if std::env::var_os("MS_E2E_CASES").is_some() {
                    println!("{}", case_line(&m));
                }
                check_setting(&m, &mut wrong);
                results.push(m);
            }
        }
        let name = format!("{} Hz, {}, {}", fs, if vbr { "VBR" } else { "CBR" }, case.name);
        println!("{}", summary(&name, &results));
        check_layout(&name, &results, &mut wrong);
    }
    assert!(wrong.is_empty(), "{} limits broken: {wrong:#?}", wrong.len());
}

/// Family 1 up to 5 channels.
const SMALL: &[Case] = &[LAYOUTS[0], LAYOUTS[1], LAYOUTS[2], LAYOUTS[3], LAYOUTS[4]];
/// Family 1 with an LFE channel: 5.1, 6.1 and 7.1.
const WITH_LFE: &[Case] = &[LAYOUTS[5], LAYOUTS[6], LAYOUTS[7]];
/// Families 255 and 2.
const OTHER: &[Case] = &[LAYOUTS[8], LAYOUTS[9], LAYOUTS[10], LAYOUTS[11], LAYOUTS[12]];

macro_rules! matrix {
    ($($name:ident: $fs:expr, $vbr:expr, $layouts:expr;)*) => {
        $(
            #[test]
            fn $name() {
                assert_run($fs, $vbr, $layouts);
            }
        )*
    };
}

matrix! {
    music_family_1_up_to_5_channels_48k_vbr: 48_000, true, SMALL;
    music_family_1_up_to_5_channels_48k_cbr: 48_000, false, SMALL;
    music_family_1_up_to_5_channels_24k_vbr: 24_000, true, SMALL;
    music_family_1_up_to_5_channels_24k_cbr: 24_000, false, SMALL;
    music_family_1_up_to_5_channels_16k_vbr: 16_000, true, SMALL;
    music_family_1_up_to_5_channels_16k_cbr: 16_000, false, SMALL;
    music_family_1_with_lfe_48k_vbr: 48_000, true, WITH_LFE;
    music_family_1_with_lfe_48k_cbr: 48_000, false, WITH_LFE;
    music_family_1_with_lfe_24k_vbr: 24_000, true, WITH_LFE;
    music_family_1_with_lfe_24k_cbr: 24_000, false, WITH_LFE;
    music_family_1_with_lfe_16k_vbr: 16_000, true, WITH_LFE;
    music_family_1_with_lfe_16k_cbr: 16_000, false, WITH_LFE;
    music_families_255_and_2_48k_vbr: 48_000, true, OTHER;
    music_families_255_and_2_48k_cbr: 48_000, false, OTHER;
    music_families_255_and_2_24k_vbr: 24_000, true, OTHER;
    music_families_255_and_2_24k_cbr: 24_000, false, OTHER;
    music_families_255_and_2_16k_vbr: 16_000, true, OTHER;
    music_families_255_and_2_16k_cbr: 16_000, false, OTHER;
}

// ---------------------------------------------------------------------------
// Both encoders through the same changes of settings
// ---------------------------------------------------------------------------

/// A small deterministic random number generator.
struct Lcg(u32);

impl Lcg {
    fn below(&mut self, n: u32) -> u32 {
        self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
        (self.0 >> 8) % n
    }

    fn pick<T: Copy>(&mut self, items: &[T]) -> T {
        items[self.below(items.len() as u32) as usize]
    }
}

/// Requests of the C API that `ms_support` has no name for.
const OPUS_RESET_STATE: i32 = 4028;
const OPUS_SET_BANDWIDTH_REQUEST: i32 = 4008;
/// `OPUS_AUTO` as the value of a forced mode or a bandwidth.
const AUTO: i32 = -1000;

/// What a walk saw. Every field but `frames` counts stream packets.
#[derive(Default, Debug)]
struct Walk {
    frames: usize,
    stream_packets: usize,
    /// Streams whose mode libopus fixes, with audio in both packets.
    fixed: usize,
    /// Of those: another mode, bandwidth, frame duration or channel count.
    fixed_differs: usize,
    /// Streams that choose their mode, with audio in both packets.
    free: usize,
    /// Of those: another mode or bandwidth.
    free_differs: usize,
    /// Without audio in both packets.
    empty: usize,
    /// Of those: another TOC byte (mode, bandwidth, duration, channels).
    empty_differs: usize,
    /// Audio in the packet of one encoder only.
    audio_in_one: usize,
}

/// Takes both encoders through `frames` frames. About every fourth frame a
/// setting changes, on both encoders alike. Every frame is coded by both,
/// and the streams of the two packets are compared. The opus1 packet is
/// decoded by libopus and the libopus packet by opus1.
fn walk(case: &Case, fs: i32, seed: u32, frames: usize) -> Walk {
    let mut rng = Lcg(seed);
    let (mut rust, mut c) = encoders(case, fs, Application::Audio);
    let streams = c.streams;
    let channels = case.channels;
    rust.set_complexity(5).unwrap();
    c.set_complexity(5);
    // An explicit bitrate from the first frame on. With the automatic
    // bitrate libopus codes narrowband in surround mode (difference D1 of
    // src/multistream/encoder.rs).
    let mut per_channel = 32_000;
    rust.set_bitrate(Bitrate::BitsPerSecond(per_channel * channels as i32))
        .unwrap();
    c.set_bitrate(per_channel * channels as i32);
    let sizes = frame_sizes(fs);
    let longest = sizes[5];
    let pcm = music(fs, case, longest * 10);
    let mut c_dec = c_decoder(&rust);
    let mut rust_dec = rust_decoder(&rust);
    let mut out = vec![0u8; max_packet_bytes(streams)];
    let mut decoded = vec![0f32; longest * channels];
    let mut w = Walk::default();
    let mut frame_size = sizes[3];
    let mut vbr = true;
    let mut history: Vec<String> = Vec::new();
    for f in 0..frames {
        if rng.below(4) == 0 {
            match rng.below(8) {
                0 | 1 => {
                    per_channel = rng.pick(&[
                        4_000, 6_000, 9_000, 12_000, 16_000, 24_000, 48_000, 96_000,
                    ]);
                    let bitrate = per_channel * channels as i32;
                    rust.set_bitrate(Bitrate::BitsPerSecond(bitrate)).unwrap();
                    c.set_bitrate(bitrate);
                    history.push(format!("frame {f}: {per_channel} b/s per channel"));
                }
                2 => {
                    vbr = rng.below(2) == 0;
                    rust.set_vbr(vbr);
                    c.set_vbr(vbr);
                    history.push(format!("frame {f}: vbr {vbr}"));
                }
                3 | 4 => {
                    frame_size = rng.pick(&sizes);
                    history.push(format!("frame {f}: {}", ms(frame_size, fs)));
                }
                5 => {
                    rust.reset_state();
                    assert_eq!(c.try_ctl(OPUS_RESET_STATE, 0), 0);
                    history.push(format!("frame {f}: reset"));
                }
                6 => {
                    // Up to 6: libopus does not use its analysis.
                    let complexity = rng.below(7) as i32;
                    rust.set_complexity(complexity).unwrap();
                    c.set_complexity(complexity);
                    history.push(format!("frame {f}: complexity {complexity}"));
                }
                _ => {
                    let (mode, c_mode) = rng.pick(&[
                        (None, AUTO),
                        (None, AUTO),
                        (Some(ForcedMode::SilkOnly), MODE_SILK_ONLY),
                        (Some(ForcedMode::Hybrid), MODE_HYBRID),
                        (Some(ForcedMode::CeltOnly), MODE_CELT_ONLY),
                    ]);
                    rust.set_force_mode(mode);
                    assert_eq!(c.try_ctl(OPUS_SET_FORCE_MODE_REQUEST, c_mode), 0);
                    let (bandwidth, c_bandwidth) = rng.pick(&[
                        (opus1::Bandwidth::Auto, AUTO),
                        (opus1::Bandwidth::Narrowband, 1101),
                        (opus1::Bandwidth::Wideband, 1103),
                        (opus1::Bandwidth::Fullband, 1105),
                    ]);
                    rust.set_bandwidth(bandwidth);
                    assert_eq!(c.try_ctl(OPUS_SET_BANDWIDTH_REQUEST, c_bandwidth), 0);
                    history.push(format!("frame {f}: force {mode:?}, {bandwidth:?}"));
                }
            }
        }
        let start = (rng.below(8 * longest as u32) as usize) * channels;
        let frame = &pcm[start..start + frame_size * channels];
        let at = |history: &[String]| {
            format!(
                "{}, {fs} Hz, frame {f} of {}: {}",
                case.name,
                ms(frame_size, fs),
                history[history.len().saturating_sub(8)..].join("; ")
            )
        };

        // Both encoders write a packet for every frame.
        let n = rust
            .encode(frame, frame_size, &mut out)
            .unwrap_or_else(|e| panic!("{}: opus1 encode: {e}", at(&history)));
        let rust_packet = out[..n].to_vec();
        let c_packet = c
            .encode_float(frame, frame_size, out.len())
            .unwrap_or_else(|e| panic!("{}: libopus encode: {e}", at(&history)));
        w.frames += 1;
        if !vbr {
            assert_eq!(
                rust_packet.len(),
                c_packet.len(),
                "{}: packet size with constant bitrate",
                at(&history)
            );
        }

        // Stream by stream.
        let r_parts = split_streams(&rust_packet, streams).unwrap();
        let c_parts = split_streams(&c_packet, streams).unwrap();
        for (s, (r, cs)) in r_parts.iter().zip(&c_parts).enumerate() {
            w.stream_packets += 1;
            assert_eq!(r.samples(fs), frame_size, "{}: stream {s}", at(&history));
            let (t, ct) = (toc(r.toc), toc(cs.toc));
            // Everything of the TOC byte but the frame count code: mode,
            // bandwidth, frame duration and channel count.
            let same = r.toc & 0xFC == cs.toc & 0xFC;
            if r.payload_len == 0 && cs.payload_len == 0 {
                w.empty += 1;
                w.empty_differs += usize::from(!same);
            } else if r.payload_len == 0 || cs.payload_len == 0 {
                w.audio_in_one += 1;
            } else if mode_is_fixed(&rust, s) {
                w.fixed += 1;
                w.fixed_differs += usize::from(!same);
            } else {
                w.free += 1;
                w.free_differs += usize::from((t.mode, t.bandwidth) != (ct.mode, ct.bandwidth));
            }
        }

        // The opus1 packet in libopus, with the encoder's final range.
        let samples = c_dec
            .decode_float(Some(&rust_packet), frame_size, false)
            .unwrap_or_else(|e| panic!("{}: libopus decode: {e}", at(&history)));
        assert_eq!(samples.len(), frame_size * channels, "{}", at(&history));
        // Two kinds of stream have a wrong final range in the stream
        // encoder (see tests/multistream_enc_robust.rs).
        let known_fault = r_parts.iter().any(|p| {
            let t = toc(p.toc);
            (t.mode == Mode::Hybrid && t.stereo)
                || (t.mode == Mode::Silk
                    && t.bandwidth == ms_support::Bandwidth::Mb
                    && t.frame_48k == 480)
        });
        if !known_fault {
            assert_eq!(
                rust.final_range(),
                c_dec.final_range(),
                "{}: final range: {}",
                at(&history),
                describe_packet(&rust_packet, streams)
            );
        }
        // The libopus packet in opus1.
        let n = rust_dec
            .decode(&c_packet, frame_size, &mut decoded)
            .unwrap_or_else(|e| panic!("{}: opus1 decode: {e}", at(&history)));
        assert_eq!(n, frame_size, "{}", at(&history));
        assert!(
            decoded[..n * channels].iter().all(|v| v.is_finite()),
            "{}: opus1 decoded samples that are not finite",
            at(&history)
        );
    }
    w
}

/// Runs the walk on every layout at one sampling rate and checks what it
/// saw.
///
/// Measured over all three sampling rates, 300 frames per layout, 11700
/// frames and 44100 stream packets in all:
///
/// - A stream whose mode libopus fixes (coupled and LFE streams of a
///   surround layout, all streams of ambisonics) has the mode, bandwidth,
///   frame duration and channel count of the libopus stream in every
///   packet, through every change and reset. The same holds for the TOC
///   byte of a stream without audio, which names the last frame the stream
///   coded. That is 24689 stream packets with audio and 1856 without.
/// - The two layouts with a stereo stream outside surround mode are left
///   out of that: libopus codes such a stream as mono at low bitrates.
/// - A stream that chooses its mode differs in up to 6 % of its packets.
///   Most layouts have 1 to 4 such packets in 300 frames, at a change of
///   mode. The mono layout at 48 kHz has a run of 13: after CELT frames
///   and a drop to 12 kb/s libopus codes SILK narrowband and opus1 SILK
///   wideband.
/// - Up to 2 % of the stream packets of a layout have audio in one encoder
///   only (25 of 44100 in all), with constant bitrate at 4 to 6 kb/s per
///   channel, where a stream is left a byte or two.
fn assert_walks(fs: i32) {
    let mut total = Walk::default();
    for (i, case) in LAYOUTS.iter().enumerate() {
        let w = walk(case, fs, 77 + 13 * i as u32 + fs as u32 / 1000, 300);
        println!("{}, {fs} Hz: {w:?}", case.name);
        let stereo_outside_surround = !is_surround(case)
            && matches!((case.tone_family(), case.channels), (1, 2) | (2, 6));
        if !stereo_outside_surround {
            assert_eq!(w.fixed_differs, 0, "{}, {fs} Hz: {w:?}", case.name);
            assert_eq!(w.empty_differs, 0, "{}, {fs} Hz: {w:?}", case.name);
        }
        assert!(
            10 * w.free_differs <= w.free,
            "{}, {fs} Hz: {w:?}",
            case.name
        );
        assert!(
            25 * w.audio_in_one <= w.stream_packets,
            "{}, {fs} Hz: {w:?}",
            case.name
        );
        total.frames += w.frames;
        total.fixed += w.fixed;
        total.empty += w.empty;
        total.free += w.free;
    }
    // The walks reach what they are meant to compare.
    assert_eq!(total.frames, 300 * LAYOUTS.len());
    assert!(total.fixed > 8_000, "{total:?}");
    assert!(total.empty > 300, "{total:?}");
    assert!(total.free > 3_000, "{total:?}");
}

#[test]
fn changing_settings_keep_the_streams_of_libopus_48k() {
    assert_walks(48_000);
}

#[test]
fn changing_settings_keep_the_streams_of_libopus_24k() {
    assert_walks(24_000);
}

#[test]
fn changing_settings_keep_the_streams_of_libopus_16k() {
    assert_walks(16_000);
}
