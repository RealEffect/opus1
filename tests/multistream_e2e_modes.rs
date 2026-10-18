//! The coding mode of the streams that choose it themselves, against C
//! libopus, and what it does to the bitrate.
//!
//! libopus fixes the mode of a stream from the multistream layer in two
//! cases only (opus_multistream_encoder.c:953-962): the coupled streams of a
//! surround layout and all streams of an ambisonics layout are CELT-only.
//! Every other stream, such as the centre channel of 5.1 or any stream of
//! mapping family 255, is an ordinary encoder that picks SILK, Hybrid or
//! CELT for each frame (opus_encoder.c:1466-1539). The multistream layer
//! hands it the input for its analysis and nothing else.
//!
//! That choice goes by the bitrate and by an estimate of how likely the
//! input is speech. From complexity 7 on libopus takes the estimate from
//! its analysis of the signal (analysis.c), which each stream encoder runs
//! on the stream's channels of the whole input. opus1 does the same. The
//! tests show:
//!
//! - Without the analysis (complexity 6) every stream of opus1 has the
//!   mode of the libopus stream in every frame.
//! - With it (complexity 9) both start in the same mode and, on the test
//!   signal, change the stream to CELT at the same frame, after about
//!   0.4 s. Every stream of opus1 has the mode of the libopus stream in
//!   every frame here too.
//!
//! A second difference shows in the same numbers and has nothing to do
//! with the analysis: in the same mode, a SILK or Hybrid stream of opus1
//! uses up to 19 % fewer bits than the libopus stream.
#![cfg(feature = "std")]
// Some helpers serve only the tests that need the speech and music analysis.
#![cfg_attr(not(feature = "analysis"), allow(dead_code))]

mod ms_support;
mod multistream_e2e_common;
mod multistream_enc_common;

use ms_support::*;
use multistream_e2e_common::*;
use multistream_enc_common::*;
use opus1::Application;

/// 20 ms frames per run: 2 seconds.
const FRAMES: usize = 100;

/// What one run of both encoders shows.
struct Run {
    label: String,
    /// Bits per second of the packets as a fraction of the request:
    /// opus1, libopus.
    rate: [f64; 2],
    /// Stream packets of streams that choose their own mode.
    compared: usize,
    /// Of those, the ones with another mode than the libopus stream.
    mode_differs: usize,
    /// Per stream that chooses its own mode: the first frame from which
    /// libopus codes CELT to the end, if it does not code CELT from the
    /// start, and whether opus1 codes the same mode in every frame.
    to_celt: Vec<(Option<usize>, bool)>,
}

#[allow(clippy::needless_range_loop)]
fn run(
    case: &Case,
    fs: i32,
    application: Application,
    per_channel: i32,
    complexity: i32,
) -> Run {
    let frame_size = fs as usize / 50;
    let bitrate = per_channel * case.channels as i32;
    let (mut rust, mut c) = encoders(case, fs, application);
    configure(&mut rust, &mut c, bitrate, true);
    rust.set_complexity(complexity).unwrap();
    c.set_complexity(complexity);
    let streams = c.streams;
    // The signal without noise: libopus takes it for music, and it has
    // nothing above 2.5 kHz.
    let pcm = program_without_noise(fs, case, FRAMES * frame_size);
    let packets = [
        rust_encode_all(&mut rust, &pcm, frame_size),
        c.encode_all(&pcm, frame_size),
    ];
    // The mode of every stream in every frame.
    let modes: Vec<Vec<Vec<Mode>>> = packets
        .iter()
        .map(|packets| {
            packets
                .iter()
                .map(|p| {
                    split_streams(p, streams)
                        .unwrap()
                        .iter()
                        .map(|part| toc(part.toc).mode)
                        .collect()
                })
                .collect()
        })
        .collect();
    let free: Vec<usize> = (0..streams)
        .filter(|&s| !mode_is_fixed(&rust, s))
        .collect();
    let mut mode_differs = 0;
    for f in 0..FRAMES {
        for &s in &free {
            mode_differs += usize::from(modes[RUST][f][s] != modes[C][f][s]);
        }
    }
    let to_celt = free
        .iter()
        .map(|&s| {
            let first_celt_run = (0..FRAMES)
                .rev()
                .take_while(|&f| modes[C][f][s] == Mode::Celt)
                .last();
            (
                first_celt_run.filter(|&f| f > 0),
                (0..FRAMES).all(|f| modes[RUST][f][s] == modes[RUST][0][s]),
            )
        })
        .collect();
    let seconds = FRAMES as f64 / 50.0;
    let fraction = |packets: &[Vec<u8>]| {
        8.0 * packets.iter().map(|p| p.len()).sum::<usize>() as f64 / seconds / bitrate as f64
    };
    Run {
        label: format!(
            "{}, {fs} Hz, {application:?}, {per_channel} b/s per channel, complexity {complexity}",
            case.name
        ),
        rate: [fraction(&packets[RUST]), fraction(&packets[C])],
        compared: FRAMES * free.len(),
        mode_differs,
        to_celt,
    }
}

/// Layouts with streams that choose their own mode: all streams of the
/// first two, the centre stream of the others.
const CASES: [Case; 4] = [
    Case::family("mono", 1, 1),
    Case::family("family 255, 3 channels", 3, 255),
    Case::family("5.1", 6, 1),
    Case::family("7.1", 8, 1),
];

/// Sampling rate, application and bitrates per channel.
const SETTINGS: [(i32, Application, &[i32]); 4] = [
    (16_000, Application::Audio, &[12_000, 16_000, 32_000, 64_000]),
    (16_000, Application::Voip, &[16_000, 32_000, 64_000]),
    (48_000, Application::Voip, &[16_000, 32_000, 64_000]),
    (48_000, Application::Audio, &[12_000, 16_000, 32_000]),
];

fn run_all(complexity: i32) -> Vec<Run> {
    let mut runs = Vec::new();
    for (fs, application, rates) in SETTINGS {
        for case in &CASES {
            for &per_channel in rates {
                let r = run(case, fs, application, per_channel, complexity);
                println!(
                    "{}: {:.0} % of the request, libopus {:.0} %; mode differs in {} of {} stream packets",
                    r.label,
                    100.0 * r.rate[RUST],
                    100.0 * r.rate[C],
                    r.mode_differs,
                    r.compared
                );
                runs.push(r);
            }
        }
    }
    runs
}

/// libopus runs its analysis from complexity 7 (opus_encoder.c:1252). At
/// complexity 6 it goes by the application alone, and so does opus1.
///
/// Measured over 52 settings: no stream packet of 7800 has another mode.
/// Also at 16 kHz with the VoIP application and 64 kb/s per channel, where
/// both encoders code SILK and use about half of the request (one mono
/// stream: libopus 50 %, opus1 45 %).
///
/// The rate of opus1 is 81 to 100 % of the rate of libopus. The low end
/// is SILK at 16 kHz: in the same mode, the SILK encoder of opus1 spends
/// fewer bits. That is a difference inside the stream encoder.
///
/// In one setting libopus uses about twice what was asked for: 5.1
/// at 48 kHz and 12 kb/s per channel, where the centre stream is in Hybrid
/// mode. The energy mask raises the SILK part of the stream to about the
/// stream's whole bitrate (opus_encoder.c:2069-2104). The CELT part is
/// set to what is left (opus_encoder.c:2454). The CELT encoder refuses a
/// bitrate of 500 b/s or less (celt_encoder.c:3008) and keeps the one it
/// started with, the maximum. opus1 deliberately does not follow that
/// and stays at the request, so the rates are not compared there.
#[test]
fn without_the_analysis_of_libopus_every_stream_has_the_mode_of_libopus() {
    let runs = run_all(6);
    let mut compared = 0;
    let mut libopus_far_above = 0;
    let (mut lowest, mut highest) = (f64::INFINITY, 0f64);
    for r in &runs {
        assert_eq!(r.mode_differs, 0, "{}", r.label);
        compared += r.compared;
        if r.rate[C] > 1.2 {
            libopus_far_above += 1;
            assert!(
                (0.85..=1.10).contains(&r.rate[RUST]),
                "{}: {:.0} % of the request",
                r.label,
                100.0 * r.rate[RUST]
            );
            continue;
        }
        let ratio = r.rate[RUST] / r.rate[C];
        lowest = lowest.min(ratio);
        highest = highest.max(ratio);
        assert!(
            (0.77..=1.03).contains(&ratio),
            "{}: {:.0} % of the request, libopus {:.0} %",
            r.label,
            100.0 * r.rate[RUST],
            100.0 * r.rate[C]
        );
    }
    println!(
        "{compared} stream packets; opus1 uses {:.0} to {:.0} % of the bits of libopus; \
         libopus far above the request in {libopus_far_above} settings",
        100.0 * lowest,
        100.0 * highest
    );
    assert_eq!(compared, 7800);
    assert!(libopus_far_above <= 2, "{libopus_far_above} settings");
}

/// With the analysis both encoders take the test signal for music. A
/// stream that starts in SILK or Hybrid then changes to CELT, which uses
/// the whole bitrate. Measured: in 24 of the 52 settings libopus changes
/// streams to CELT, 36 streams in all, at frame 17 to 21, which is 0.34 to
/// 0.42 s. opus1 does the same in the same frames: no stream packet of
/// 16,900 has another mode. 16 of the 52 settings give libopus's packets
/// byte for byte in all 100 frames; in most others the frames before the
/// switch differ, where SILK codes them.
///
/// Before opus1 ran the analysis, its streams kept the mode of their first
/// frame: 16 of the 24 settings stayed in SILK or Hybrid, at 49 % of the
/// request at worst.
///
/// libopus itself is at 78 to 110 % of the request, apart from 5.1 at
/// 48 kHz and 12 kb/s per channel (146 %, see the test above). opus1 is at
/// 63 to 104 %; its lowest rates are SILK at 16 kHz, as in the test above.
#[cfg(feature = "analysis")]
#[test]
fn with_the_analysis_every_stream_has_the_mode_of_libopus() {
    let runs = run_all(9);
    let (mut moved, mut settings_moved) = (0, 0);
    let (mut first, mut last) = (FRAMES, 0);
    let mut lowest = f64::INFINITY;
    let mut libopus_lowest = f64::INFINITY;
    for r in &runs {
        lowest = lowest.min(r.rate[RUST]);
        libopus_lowest = libopus_lowest.min(r.rate[C]);
        let before = moved;
        assert_eq!(r.mode_differs, 0, "{}", r.label);
        for &(frame, _) in &r.to_celt {
            let Some(frame) = frame else { continue };
            moved += 1;
            first = first.min(frame);
            last = last.max(frame);
            assert!(
                (15..=25).contains(&frame),
                "{}: libopus changes to CELT at frame {frame}",
                r.label
            );
        }
        settings_moved += usize::from(moved > before);
        // Never far above the request, and not below the floor.
        assert!(
            (0.45..=1.10).contains(&r.rate[RUST]),
            "{}: {:.0} % of the request",
            r.label,
            100.0 * r.rate[RUST]
        );
    }
    println!(
        "libopus changed {moved} streams to CELT in {settings_moved} settings, at frame {first} \
         to {last}; opus1 uses {:.0} % of the request at worst, libopus {:.0} %",
        100.0 * lowest,
        100.0 * libopus_lowest
    );
    assert!(moved >= 30, "{moved} streams changed to CELT");
    assert!(libopus_lowest > 0.70, "libopus: {libopus_lowest}");
}
