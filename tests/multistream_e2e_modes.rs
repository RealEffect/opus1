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
//! its analysis of the signal (analysis.c). opus-rs has no such analysis
//! and uses the fixed estimate libopus uses without it. The tests show:
//!
//! - Without the analysis of libopus (complexity 6) every stream of
//!   opus-rs has the mode of the libopus stream in every frame.
//! - With it (complexity 9) libopus starts in the same mode and, on the
//!   test signal, changes the stream to CELT after about 0.4 s. CELT fills
//!   the bitrate; SILK and Hybrid do not. opus-rs stays where it started.
//!
//! So the multistream layer has nothing to follow here. Forcing CELT on
//! these streams would not be what libopus does: on a signal its analysis
//! takes for speech, libopus stays with SILK as well.
//!
//! A second difference shows in the same numbers and has nothing to do
//! with the analysis: in the same mode, a SILK or Hybrid stream of opus-rs
//! uses up to 19 % fewer bits than the libopus stream.
#![cfg(feature = "std")]

mod ms_support;
mod multistream_e2e_common;
mod multistream_enc_common;

use ms_support::*;
use multistream_e2e_common::*;
use multistream_enc_common::*;
use opus_rs::Application;

/// 20 ms frames per run: 2 seconds.
const FRAMES: usize = 100;

/// What one run of both encoders shows.
struct Run {
    label: String,
    /// Bits per second of the packets as a fraction of the request:
    /// opus-rs, libopus.
    rate: [f64; 2],
    /// Stream packets of streams that choose their own mode.
    compared: usize,
    /// Of those, the ones with another mode than the libopus stream.
    mode_differs: usize,
    /// Per stream that chooses its own mode: the first frame from which
    /// libopus codes CELT to the end, if it does not code CELT from the
    /// start, and whether opus-rs codes the same mode in every frame.
    to_celt: Vec<(Option<usize>, bool)>,
}

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
/// complexity 6 it goes by the application alone, as opus-rs always does.
///
/// Measured over 52 settings: no stream packet of 7800 has another mode.
/// Also at 16 kHz with the VoIP application and 64 kb/s per channel, where
/// both encoders code SILK and use about half of the request (one mono
/// stream: libopus 50 %, opus-rs 45 %).
///
/// The rate of opus-rs is 81 to 100 % of the rate of libopus. The low end
/// is SILK at 16 kHz: in the same mode, the SILK encoder of opus-rs spends
/// fewer bits. That is a difference inside the stream encoder.
///
/// In one setting libopus uses about twice what was asked for: 5.1
/// at 48 kHz and 12 kb/s per channel, where the centre stream is in Hybrid
/// mode. The energy mask raises the SILK part of the stream to about the
/// stream's whole bitrate (opus_encoder.c:2069-2104). The CELT part is
/// set to what is left (opus_encoder.c:2454). The CELT encoder refuses a
/// bitrate of 500 b/s or less (celt_encoder.c:3008) and keeps the one it
/// started with, the maximum. opus-rs deliberately does not follow that
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
        "{compared} stream packets; opus-rs uses {:.0} to {:.0} % of the bits of libopus; \
         libopus far above the request in {libopus_far_above} settings",
        100.0 * lowest,
        100.0 * highest
    );
    assert_eq!(compared, 7800);
    assert!(libopus_far_above <= 2, "{libopus_far_above} settings");
}

/// With its analysis libopus takes the test signal for music. A stream
/// that starts in SILK or Hybrid then changes to CELT, which uses the whole
/// bitrate. Measured: in 24 of the 52 settings libopus changes streams to
/// CELT, 36 streams in all, at frame 17 to 21, which is 0.34 to 0.42 s.
/// A stream of opus-rs codes the mode of its first frame in every frame.
///
/// - In 16 of the 24 settings opus-rs starts in SILK or Hybrid, as libopus
///   does, and stays there. Its packets then hold less than was asked for:
///   49 % of the request at worst (three mono streams at 16 kHz, VoIP,
///   32 kb/s each), where libopus has 94 %.
/// - In the other 8 opus-rs codes CELT from the first frame, where libopus
///   codes SILK or Hybrid for its first 17 or 18 frames.
///
/// At 48 kHz with 12 kb/s per channel the analysis has another effect: it
/// finds no signal above 8 kHz, so libopus codes SILK wideband where
/// opus-rs codes Hybrid.
///
/// libopus itself is at 78 to 110 % of the request, apart from 5.1 at
/// 48 kHz and 12 kb/s per channel (146 %, see the test above).
#[test]
fn the_analysis_of_libopus_moves_streams_to_celt_and_opus_rs_stays() {
    let runs = run_all(9);
    let (mut moved, mut settings_moved) = (0, 0);
    let (mut first, mut last) = (FRAMES, 0);
    let mut lowest = f64::INFINITY;
    let mut libopus_lowest = f64::INFINITY;
    for r in &runs {
        lowest = lowest.min(r.rate[RUST]);
        libopus_lowest = libopus_lowest.min(r.rate[C]);
        let before = moved;
        for &(frame, rust_keeps_its_mode) in &r.to_celt {
            assert!(
                rust_keeps_its_mode,
                "{}: opus-rs changed its mode; is the analysis ported?",
                r.label
            );
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
         to {last}; opus-rs uses {:.0} % of the request at worst, libopus {:.0} %",
        100.0 * lowest,
        100.0 * libopus_lowest
    );
    assert!(moved >= 30, "{moved} streams changed to CELT");
    assert!(libopus_lowest > 0.70, "libopus: {libopus_lowest}");
}
