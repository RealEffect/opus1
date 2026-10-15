//! The multistream encoder with everything switched on, end to end against
//! C libopus: surround masking, the LFE tuning, forced modes and the rate
//! allocation together.
//!
//! The matrix: every layout of mapping family 1 (1 to 8 channels), two
//! layouts of family 255 and three of family 2; 48, 24 and 16 kHz; frames
//! of 2.5, 5, 10, 20, 40 and 60 ms; variable and constant bitrate; 12, 32
//! and 64 kb/s per channel. That is 1404 settings. Each one is coded by
//! both encoders, and both packet sequences are decoded by both decoders
//! (`multistream_e2e_common::measure`).
//!
//! What is checked for every setting:
//!
//! - Structure: a stream whose mode libopus fixes has the mode of the
//!   libopus stream in every frame. In a surround layout every stream also
//!   has its bandwidth and channel count.
//! - Bitrates: every stream encoder is set to the bitrate libopus sets.
//!   With constant bitrate the packets have the size of the libopus
//!   packets. With variable bitrate the total stays near the request and
//!   near libopus.
//! - Quality: the SNR of every channel with the opus-rs encoder, against
//!   the libopus encoder, both decoded by libopus.
//! - The opus-rs decoder: its output has the SNR of the libopus decoder's
//!   output, for the packets of either encoder.
//!
//! Every test prints the measurements of its layouts; run with
//! `--nocapture` to see them. With the environment variable `MS_E2E_CASES`
//! set, it prints one line per setting as well.
//!
//! Where opus-rs differs, the cause is in the stream encoder, not in the
//! multistream layer: tests/multistream_enc_streams.rs shows that every
//! stream holds the bytes a single `OpusEncoder` writes. The differences
//! that the limits below leave room for:
//!
//! - A stream that picks its own mode can pick another one than libopus,
//!   in up to 7 % of its frames. libopus goes by its analysis of the
//!   signal, which opus-rs does not have
//!   (tests/multistream_e2e_modes.rs).
//! - A stereo stream outside surround mode is coded as mono by libopus at
//!   12 kb/s per channel. opus-rs always codes two channels.
//! - Single channels are up to 9 dB below libopus. The largest gaps: mono
//!   CELT streams with 2.5 ms frames at 32 kb/s and 48 kHz (7 to 9 dB), and
//!   the right channel of a stereo CELT stream with 5 ms frames at 12 kb/s
//!   per channel (5 to 9 dB). Averaged over a layout the gap is 0.2 to
//!   0.9 dB.
//! - Of the average gap of the stereo streams at 48 kHz, 0.7 to 0.85 dB
//!   come from the pitch pre-filter: libopus runs it on mono and stereo
//!   streams from complexity 5 on, opus-rs on mono streams only
//!   (`pf_enabled` in src/celt.rs). Measured on this matrix at 12, 32 and
//!   64 kb/s per channel: 0.6, 0.0 and 0.2 dB below libopus at complexity
//!   4, and 1.3, 0.8 and 1.0 dB at complexity 5. At 24 and 16 kHz the two
//!   complexities give the same gap.
//! - The pulse search for 2 to 4 pulses on aarch64 (src/pvq.rs) costs 0.2
//!   to 0.7 dB at 12 kb/s per channel with frames from 20 ms on, most on
//!   the LFE channel. tests/multistream_e2e_no_analysis.rs has the numbers
//!   for a second signal.
//! - 7.1 at 16 kHz, 10 ms, constant bitrate, 96 kb/s: the centre stream is
//!   SILK and takes a few bytes more than in libopus, so in 32 of 80
//!   frames the LFE stream is left too few bytes for audio.
#![cfg(feature = "std")]

mod ms_support;
mod multistream_e2e_common;
mod multistream_enc_common;

use multistream_e2e_common::*;
use multistream_enc_common::*;
use opus_rs::Application;
use opus_rs::multistream::MappingType;

/// Signal length per setting in seconds. The first 0.2 s are not measured.
const SECONDS: f64 = 0.8;

/// Family 1 up to 5 channels.
const SMALL: &[Case] = &[LAYOUTS[0], LAYOUTS[1], LAYOUTS[2], LAYOUTS[3], LAYOUTS[4]];
/// Family 1 with an LFE channel: 5.1, 6.1 and 7.1.
const WITH_LFE: &[Case] = &[LAYOUTS[5], LAYOUTS[6], LAYOUTS[7]];
/// Families 255 and 2.
const OTHER: &[Case] = &[LAYOUTS[8], LAYOUTS[9], LAYOUTS[10], LAYOUTS[11], LAYOUTS[12]];

/// One line of text for a setting: rates, stream bytes, structure and the
/// SNR of every channel on the four paths.
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
/// and checks the limits. Returns what is wrong.
fn run(fs: i32, vbr: bool, layouts: &[Case]) -> Vec<String> {
    let mut wrong = Vec::new();
    // Settings where one encoder has audio in a stream and the other none.
    let mut audio_in_one_only = Vec::new();
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
                    complexity: 9,
                };
                let m = measure(setting, SECONDS);
                if std::env::var_os("MS_E2E_CASES").is_some() {
                    println!("{}", case_line(&m));
                    for e in &m.examples {
                        println!("  EX {e}");
                    }
                }
                check_setting(&m, &mut wrong, &mut audio_in_one_only);
                results.push(m);
            }
        }
        let name = format!("{} Hz, {}, {}", fs, if vbr { "VBR" } else { "CBR" }, case.name);
        println!("{}", summary(&name, &results));
        check_layout(&name, &results, &mut wrong);
    }
    // Measured: one setting of the whole matrix, 7.1 at 16 kHz, 10 ms,
    // constant bitrate, 96 kb/s.
    if audio_in_one_only.len() > 1 {
        wrong.push(format!(
            "audio in one encoder's stream only, in {} settings: {audio_in_one_only:?}",
            audio_in_one_only.len()
        ));
    }
    wrong
}

/// The limits that hold for every single setting.
fn check_setting(m: &Measured, wrong: &mut Vec<String>, audio_in_one_only: &mut Vec<String>) {
    let label = m.setting.label();
    let mut fail = |text: String| wrong.push(format!("{label}: {text}"));
    let example = |kind: &str| {
        m.examples
            .iter()
            .find(|e| e.ends_with(kind))
            .cloned()
            .unwrap_or_default()
    };
    let surround = m.setting.case.channels > 2 && m.setting.case.tone_family() == 1;

    // --- Structure ---
    let s = &m.structure;
    if s.fixed_mode != 0 {
        fail(format!(
            "{} stream packets with another mode than libopus fixes: {}",
            s.fixed_mode,
            example("fixed mode")
        ));
    }
    if surround && (s.bandwidth != 0 || s.channels != 0) {
        fail(format!(
            "{} stream packets with another bandwidth, {} with another channel count: {} {}",
            s.bandwidth,
            s.channels,
            example("bandwidth"),
            example("channels")
        ));
    }
    if s.audio != 0 || s.empty_toc != 0 {
        audio_in_one_only.push(label.clone());
        // Measured in the one setting: 32 and 23 of 400 stream packets.
        if s.audio + s.empty_toc > 80 {
            fail(format!(
                "{} stream packets with audio in one encoder only, {} without audio and another \
                 TOC byte",
                s.audio, s.empty_toc
            ));
        }
    }

    // --- Bitrates ---
    for (i, stream) in m.streams.iter().enumerate() {
        // With constant bitrate the last stream is set to what the others
        // left (opus_multistream_encoder.c:1025-1026). That depends on the
        // bytes the streams before it wrote, so it is not compared.
        if !m.setting.vbr && i + 1 == m.streams.len() {
            continue;
        }
        if stream.set[RUST] != stream.set[C] {
            fail(format!(
                "stream {i} is set to {} b/s, in libopus to {} b/s",
                stream.set[RUST], stream.set[C]
            ));
        }
    }
    let request = m.setting.bitrate as f64;
    if !m.setting.vbr {
        // The same bytes in total. Rounding to whole bytes per frame puts
        // the rate up to 6.7 % off the request (2.5 ms frames).
        if m.rate[RUST] != m.rate[C] {
            fail(format!(
                "constant bitrate: {:.0} b/s, libopus {:.0} b/s",
                m.rate[RUST], m.rate[C]
            ));
        }
    } else {
        // Measured for 10 ms and longer: 90 to 113 % of the request. For
        // shorter frames the headers of the streams come on top: up to
        // 182 % for both encoders (2.5 ms, 12 kb/s per channel).
        let long = m.setting.frame_size * 100 >= m.setting.fs as usize;
        if long && !(0.87..=1.15).contains(&(m.rate[RUST] / request)) {
            fail(format!(
                "variable bitrate: {:.0} b/s, libopus {:.0} b/s",
                m.rate[RUST], m.rate[C]
            ));
        }
        // Against libopus, measured: 89 to 112 %. Not where libopus is far
        // above the request itself: with a Hybrid centre stream and an
        // energy mask libopus uses up to 4.2 times the request (5.1 at
        // 24 kHz, 10 ms, 12 kb/s per channel; see
        // tests/multistream_e2e_modes.rs).
        let libopus_is_sane = m.rate[C] <= 1.3 * request || !long;
        if libopus_is_sane && !(0.86..=1.15).contains(&(m.rate[RUST] / m.rate[C])) {
            fail(format!(
                "variable bitrate: {:.0} b/s, libopus {:.0} b/s",
                m.rate[RUST], m.rate[C]
            ));
        }
    }

    // --- Quality and the decoder ---
    for (ch, row) in m.channels.iter().enumerate() {
        let gap = row.snr[RUST_TO_C] - row.snr[C_TO_C];
        // Measured at worst: mono -8.9 dB, coupled -8.8 dB, LFE -3.9 dB.
        let limit = match row.kind {
            StreamKind::Mono | StreamKind::Coupled => -10.5,
            StreamKind::Lfe => -5.5,
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
        // Measured: within 0.05 dB in every setting.
        for (ours, theirs, whose) in [
            (RUST_TO_RUST, RUST_TO_C, "opus-rs"),
            (C_TO_RUST, C_TO_C, "libopus"),
        ] {
            if (row.snr[ours] - row.snr[theirs]).abs() > 0.2 {
                fail(format!(
                    "channel {ch}: the opus-rs decoder gives {:.2} dB on {whose} packets, the \
                     libopus decoder {:.2} dB",
                    row.snr[ours], row.snr[theirs]
                ));
            }
        }
    }
}

/// The limits on the 18 settings of one layout together.
fn check_layout(name: &str, results: &[Measured], wrong: &mut Vec<String>) {
    // The average gap to libopus per kind of stream. Measured at worst:
    // -1.6 dB (coupled streams of the 3-channel layout at 48 kHz).
    for kind in StreamKind::ALL {
        if let Some(gap) = snr_differences(results, kind, RUST_TO_C, C_TO_C)
            && gap.mean < -2.2
        {
            wrong.push(format!(
                "{name}: {} channels are {:.1} dB below libopus on average",
                kind.name(),
                -gap.mean
            ));
        }
    }
    // Streams that pick their own mode. Measured at worst: 7.0 % of the
    // stream packets of a layout have another mode than in libopus.
    let mut structure = Structure::default();
    for m in results {
        structure.add(&m.structure);
    }
    if 100 * structure.free_mode > 10 * structure.compared {
        wrong.push(format!(
            "{name}: {} of {} stream packets have another mode than in libopus",
            structure.free_mode, structure.compared
        ));
    }
}

fn assert_run(fs: i32, vbr: bool, layouts: &[Case]) {
    let wrong = run(fs, vbr, layouts);
    assert!(wrong.is_empty(), "{} limits broken: {wrong:#?}", wrong.len());
}

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
    family_1_up_to_5_channels_48k_vbr: 48_000, true, SMALL;
    family_1_up_to_5_channels_48k_cbr: 48_000, false, SMALL;
    family_1_up_to_5_channels_24k_vbr: 24_000, true, SMALL;
    family_1_up_to_5_channels_24k_cbr: 24_000, false, SMALL;
    family_1_up_to_5_channels_16k_vbr: 16_000, true, SMALL;
    family_1_up_to_5_channels_16k_cbr: 16_000, false, SMALL;
    family_1_with_lfe_48k_vbr: 48_000, true, WITH_LFE;
    family_1_with_lfe_48k_cbr: 48_000, false, WITH_LFE;
    family_1_with_lfe_24k_vbr: 24_000, true, WITH_LFE;
    family_1_with_lfe_24k_cbr: 24_000, false, WITH_LFE;
    family_1_with_lfe_16k_vbr: 16_000, true, WITH_LFE;
    family_1_with_lfe_16k_cbr: 16_000, false, WITH_LFE;
    families_255_and_2_48k_vbr: 48_000, true, OTHER;
    families_255_and_2_48k_cbr: 48_000, false, OTHER;
    families_255_and_2_24k_vbr: 24_000, true, OTHER;
    families_255_and_2_24k_cbr: 24_000, false, OTHER;
    families_255_and_2_16k_vbr: 16_000, true, OTHER;
    families_255_and_2_16k_cbr: 16_000, false, OTHER;
}

/// The layouts of the matrix are the ones the task names, and the surround
/// constructor gives each the kind of layout libopus gives it.
#[test]
fn the_layouts_of_the_matrix_have_the_mapping_types_of_libopus() {
    for case in LAYOUTS {
        let enc = rust_encoder(case, 48_000, Application::Audio);
        let want = match (case.tone_family(), case.channels) {
            (1, 3..) => MappingType::Surround,
            (2, _) => MappingType::Ambisonics,
            _ => MappingType::None,
        };
        assert_eq!(enc.mapping_type(), want, "{}", case.name);
        assert_eq!(
            enc.lfe_stream().is_some(),
            case.tone_family() == 1 && case.channels >= 6,
            "{}",
            case.name
        );
    }
}

/// The test signal has a little noise so that both encoders code the same
/// audio bandwidth (see `multistream_e2e_common::program`). Without the
/// noise the signal has nothing above 2.5 kHz. The analysis of libopus then
/// lowers the bandwidth of a stream that picks its own
/// (opus_encoder.c:1652-1673), and libopus puts its bits where the signal
/// is. opus-rs has no analysis and codes the bandwidth the bitrate allows.
///
/// Shown on four CELT-only mono streams (family 2) at 24 kHz with 2.5 ms
/// frames and 32 kb/s per channel. Measured without the noise: libopus
/// codes narrowband in 964 of 1280 stream packets where opus-rs codes
/// wideband, and opus-rs is 3.4 dB below libopus on average (8.6 dB when
/// this test was written; opus-rs has improved since). With the noise: the
/// same bandwidth in every packet, and opus-rs is 0.8 dB above.
///
/// The bound on the gap without the noise is relative to the gap with it,
/// where both code the same bandwidth: the narrower bandwidth gains libopus
/// at least 2 dB against opus-rs (measured 4.2 dB). A fixed bound on the gap
/// itself held opus-rs to being far behind.
#[test]
fn a_signal_without_high_frequencies_gets_a_narrower_bandwidth_in_libopus() {
    let fs = 24_000;
    let case = LAYOUTS[10];
    let setting = Setting {
        case,
        fs,
        application: Application::Audio,
        frame_size: fs as usize / 400,
        vbr: true,
        bitrate: 32_000 * case.channels as i32,
        complexity: 9,
    };
    let samples = (SECONDS * fs as f64) as usize;
    let gap = |m: &Measured| {
        snr_differences(std::slice::from_ref(m), StreamKind::Mono, RUST_TO_C, C_TO_C)
            .unwrap()
            .mean
    };

    let without = measure_signal(setting, &program_without_noise(fs, &case, samples));
    println!(
        "without noise: another bandwidth in {} of {} stream packets, opus-rs {:+.1} dB",
        without.structure.bandwidth,
        without.structure.compared,
        gap(&without)
    );
    assert!(2 * without.structure.bandwidth > without.structure.compared);

    let with = measure_signal(setting, &program(fs, &case, samples));
    println!(
        "with noise: another bandwidth in {} of {} stream packets, opus-rs {:+.1} dB",
        with.structure.bandwidth,
        with.structure.compared,
        gap(&with)
    );
    assert_eq!(with.structure.bandwidth, 0);
    assert!(gap(&with).abs() < 2.0);
    assert!(
        gap(&without) < gap(&with) - 2.0,
        "the narrower bandwidth gains libopus {:.1} dB",
        gap(&with) - gap(&without)
    );
}

/// A single CELT stream of opus-rs against libopus where it was furthest
/// behind: mono, 48 kHz, 2.5 ms frames, 32 kb/s. Mode, bandwidth and packet
/// sizes are those of libopus.
///
/// It is not the multistream layer: the layout is one mono stream.
///
/// Both encoders use the pitch pre-filter of CELT from complexity 5 on.
/// When this test was written opus-rs was 0.5 dB behind libopus at
/// complexity 4 and 8 dB behind with the pre-filter (11.5 / 19.3 dB); it was
/// then ignored because a merge left 17.8 dB at complexity 4. Measured now,
/// on the integration branch as on this one, opus-rs / libopus:
///
/// - complexity 4, no pre-filter: 20.4 / 20.4 dB;
/// - complexity 5: 19.3 / 19.3 dB;
/// - complexity 9: 19.3 / 19.1 dB.
///
/// The test holds all three to 0.5 dB of libopus.
#[test]
fn mono_2_5_ms_frames_reach_libopus_with_and_without_the_pre_filter() {
    let fs = 48_000;
    let case = LAYOUTS[0];
    let setting = Setting {
        case,
        fs,
        application: Application::Audio,
        frame_size: fs as usize / 400,
        vbr: true,
        bitrate: 32_000,
        complexity: 9,
    };
    for (complexity, limit) in [(4, -0.5), (5, -0.5), (9, -0.5)] {
        let m = measure(Setting { complexity, ..setting }, 2.0);
        let (rust, c) = (m.channels[0].snr[RUST_TO_C], m.channels[0].snr[C_TO_C]);
        println!(
            "complexity {complexity}: opus-rs {rust:.1} dB, libopus {c:.1} dB; {:.0} / {:.0} b/s",
            m.rate[RUST], m.rate[C]
        );
        // The same mode and bandwidth in every packet, and the same rate.
        let s = &m.structure;
        assert_eq!(
            (s.free_mode, s.bandwidth, s.audio),
            (0, 0, 0),
            "complexity {complexity}"
        );
        assert!((m.rate[RUST] / m.rate[C] - 1.0).abs() < 0.01);
        assert!(
            rust - c > limit,
            "complexity {complexity}: opus-rs {rust:.1} dB, libopus {c:.1} dB"
        );
    }
}
