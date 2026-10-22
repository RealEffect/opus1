//! The redundant 5 ms CELT frame that goes with a switch between CELT-only
//! and SILK-only or Hybrid, and with a change of SILK's internal rate
//! (opus_encoder.c: the `redundancy`, `celt_to_silk` and `to_celt`
//! decisions of `opus_encode_native`, and their coding in
//! `opus_encode_frame_native`).
//!
//! The reference is the C libopus encoder with the same settings, at
//! complexity 5, where both encoders decide without the speech/music
//! analysis (both run it from complexity 7 on; `core_analysis_libopus.rs`
//! compares that). Every packet is decoded by the C decoder.
//!
//! What a missing or wrong redundant frame looks like here:
//!
//! - The first bytes differ from libopus's: libopus keeps the packet that
//!   announces a switch to CELT-only in the old mode, so the switch shows
//!   one packet later.
//! - The 20 ms around the switch decode worse than libopus's stream does.
//!   Before the frames were written, such a window was 6 to 14 dB below
//!   libopus at most switches to CELT-only.
//! - The decoder does not end in the encoder's final range.

mod core_common;
mod k24_common;

use core_common::*;
use k24_common::{speech, to_stereo};
use opus1::{Application, Bandwidth, Bitrate, Channels, ForcedMode, OpusEncoder, SampleRate};

/// Speech, mono or stereo. No steady tones: on those the SILK encoder of
/// opus1 is known to be several dB below libopus (tests/core_switch.rs),
/// which would hide what a switch does.
///
/// The recording is repeated when `seconds` is more than the 12 s used of it.
fn signal(rate: usize, channels: usize, seconds: f64) -> Vec<f32> {
    let clip = speech(rate, seconds.min(12.0));
    let mono: Vec<f32> = clip.iter().copied().cycle().take((seconds * rate as f64) as usize).collect();
    if channels == 2 { to_stereo(&mono, rate) } else { mono }
}

/// What a stream is asked to do from a given packet on.
#[derive(Clone, Copy)]
struct Step {
    mode: Option<(i32, ForcedMode)>,
    bitrate: i32,
    bandwidth: (i32, Bandwidth),
}

const AUTO_BW: (i32, Bandwidth) = (-1000, Bandwidth::Auto);
const SILK: Option<(i32, ForcedMode)> = Some((MODE_SILK, ForcedMode::SilkOnly));
const HYBRID: Option<(i32, ForcedMode)> = Some((MODE_HYBRID, ForcedMode::Hybrid));
const CELT: Option<(i32, ForcedMode)> = Some((MODE_CELT, ForcedMode::CeltOnly));

struct Stream {
    rate: usize,
    channels: usize,
    frame: usize,
    /// Packets per step of the schedule.
    hold: usize,
    cbr: bool,
    /// Silence the last 300 ms of every second. SILK changes its internal
    /// rate in a pause of the speech.
    pauses: bool,
    schedule: Vec<Step>,
    /// The application of both encoders.
    application: (i32, Application),
    /// A constant added to every input sample.
    offset: f32,
}

const AUDIO: (i32, Application) = (APP_AUDIO, Application::Audio);
const VOIP: (i32, Application) = (APP_VOIP, Application::Voip);

struct Outcome {
    /// First bytes of the opus1 and of the libopus packets, without the
    /// frame count code.
    tocs: Vec<u8>,
    c_tocs: Vec<u8>,
    /// Packets where the C decoder did not end in the encoder's final range.
    range_mismatches: Vec<usize>,
    /// Packets at which the mode or the bandwidth in the first byte changes.
    switches: Vec<usize>,
    /// For each switch: the lowest signal-to-noise ratio of the 20 ms
    /// windows from 40 ms before it to 60 ms after it, for the opus1
    /// stream and for the libopus stream, both decoded by libopus.
    worst_window: Vec<(f64, f64)>,
    /// Signal-to-noise ratio of the two decoded streams over everything.
    snr: (f64, f64),
    /// Packet sizes in bytes, opus1 and libopus.
    sizes: Vec<usize>,
    c_sizes: Vec<usize>,
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

fn snr(reference: &[f32], test: &[f32]) -> f64 {
    let (mut sig, mut err) = (0f64, 0f64);
    for (r, t) in reference.iter().zip(test) {
        sig += (*r as f64).powi(2);
        err += (*r as f64 - *t as f64).powi(2);
    }
    10.0 * (sig.max(1e-12) / err.max(1e-12)).log10()
}

fn run(stream: &Stream) -> Outcome {
    let Stream { rate, channels, frame, hold, cbr, pauses, ref schedule, application, offset } = *stream;
    let packets_total = hold * schedule.len();
    let mut pcm = signal(rate, channels, (packets_total * frame) as f64 / rate as f64 + 0.05);
    if pauses {
        for (i, sample) in pcm.chunks_exact_mut(channels).enumerate() {
            if i % rate >= rate * 7 / 10 {
                sample.fill(0.0);
            }
        }
    }
    for sample in &mut pcm {
        *sample += offset;
    }
    let mut c_enc = CEncoder::new(rate, channels, application.0);
    c_enc.complexity(5).vbr(!cbr);
    let mut enc = OpusEncoder::new(
        SampleRate::try_from(rate as i32).unwrap(),
        Channels::try_from(channels).unwrap(),
        application.1,
    )
    .unwrap();
    enc.set_complexity(5).unwrap();
    enc.set_vbr(!cbr);
    // One C decoder for each stream.
    let mut dec = CDecoder::new(rate, channels);
    let mut c_dec = CDecoder::new(rate, channels);
    let mut buf = vec![0u8; 4000];
    let (mut tocs, mut c_tocs, mut range_mismatches) = (Vec::new(), Vec::new(), Vec::new());
    let (mut decoded, mut c_decoded) = (Vec::new(), Vec::new());
    let (mut sizes, mut c_sizes) = (Vec::new(), Vec::new());
    for (i, input) in pcm.chunks_exact(frame * channels).take(packets_total).enumerate() {
        let step = schedule[i / hold];
        c_enc.bitrate(step.bitrate).bandwidth(step.bandwidth.0);
        enc.set_bitrate(Bitrate::BitsPerSecond(step.bitrate)).unwrap();
        enc.set_bandwidth(step.bandwidth.1);
        match step.mode {
            Some((c_mode, forced)) => {
                c_enc.force_mode(c_mode);
                enc.set_force_mode(Some(forced));
            }
            None => {
                c_enc.force_mode(-1000);
                enc.set_force_mode(None);
            }
        }
        let c_packet = c_enc.encode(input);
        let n = enc.encode(input, frame, &mut buf).unwrap_or_else(|e| panic!("packet {i}: {e}"));
        let pcm_out = dec
            .decode(Some(&buf[..n]), frame, false)
            .unwrap_or_else(|e| panic!("packet {i}: C decoder: {e}"));
        decoded.extend(pcm_out);
        c_decoded.extend(c_dec.decode(Some(&c_packet), frame, false).unwrap());
        if enc.final_range() != dec.final_range() {
            range_mismatches.push(i);
        }
        tocs.push(buf[0] & 0xFC);
        c_tocs.push(c_packet[0] & 0xFC);
        sizes.push(n);
        c_sizes.push(c_packet.len());
    }
    // The decoder's output lags the input by the encoder's look-ahead.
    let lag = enc.lookahead() * channels;
    let reference = &pcm[..decoded.len() - lag];
    let (decoded, c_decoded) = (&decoded[lag..], &c_decoded[lag..]);
    let switches: Vec<usize> = (1..tocs.len()).filter(|&i| tocs[i] != tocs[i - 1]).collect();
    let window = rate / 50 * channels;
    let worst_window = switches
        .iter()
        .map(|&s| {
            let at = s * frame * channels;
            let (mut worst, mut c_worst) = (f64::INFINITY, f64::INFINITY);
            for w in 0..5 {
                let start = (at + w * window).saturating_sub(2 * window);
                let end = start + window;
                if end > reference.len() {
                    break;
                }
                worst = worst.min(snr(&reference[start..end], &decoded[start..end]));
                c_worst = c_worst.min(snr(&reference[start..end], &c_decoded[start..end]));
            }
            (worst, c_worst)
        })
        .collect();
    Outcome {
        tocs,
        c_tocs,
        range_mismatches,
        switches,
        worst_window,
        snr: (snr(reference, decoded), snr(reference, c_decoded)),
        sizes,
        c_sizes,
    }
}

/// How far below libopus a stream may be, in dB: in the worst 20 ms around
/// any switch, around the switches on average, and over the whole stream.
struct Margins {
    window: f64,
    mean: f64,
    stream: f64,
}

/// For switches between CELT-only and the other modes. Measured: 4.9 dB at
/// most around a switch, 1.6 dB on average, 1.4 dB over a stream. Before
/// the redundant frames were written the worst window was 6 to 14 dB below
/// libopus at most switches to CELT-only.
const MODE_SWITCH: Margins = Margins { window: 6.0, mean: 2.0, stream: 1.5 };

/// The checks every stream of this file must pass.
///
/// - The C decoder ends every packet in the encoder's final range.
/// - The stream decodes within `margins` of libopus's own stream. What is
///   left around a switch comes from the first SILK frame after a restart,
///   which the SILK encoder of opus1 codes up to 5 dB worse than libopus,
///   also at the start of a stream.
fn check(what: &str, outcome: &Outcome, min_switches: usize, margins: &Margins) {
    assert!(outcome.range_mismatches.is_empty(), "{what}: final range of packets {:?}", outcome.range_mismatches);
    assert!(outcome.switches.len() >= min_switches, "{what}: switches at {:?}", outcome.switches);
    let deficits: Vec<f64> = outcome.worst_window.iter().map(|(worst, c_worst)| c_worst - worst).collect();
    let mean_deficit = deficits.iter().sum::<f64>() / deficits.len().max(1) as f64;
    if std::env::var("CORE_REDUNDANCY_PRINT").is_ok() {
        let worst_deficit = deficits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        println!(
            "{what}: {} switches, worst deficit {worst_deficit:.1} dB, mean {mean_deficit:.1} dB, stream {:.1}/{:.1} dB",
            outcome.switches.len(),
            outcome.snr.0,
            outcome.snr.1
        );
    }
    for (&at, &(worst, c_worst)) in outcome.switches.iter().zip(&outcome.worst_window) {
        assert!(worst >= c_worst - margins.window, "{what}: switch at packet {at}: {worst:.1} dB, libopus {c_worst:.1} dB");
    }
    assert!(mean_deficit <= margins.mean, "{what}: {mean_deficit:.1} dB below libopus around the switches on average");
    assert!(outcome.snr.0 >= outcome.snr.1 - margins.stream, "{what}: {:.1} dB, libopus {:.1} dB", outcome.snr.0, outcome.snr.1);
}

/// Every order of two forced modes, each held for 200 ms. Above 16 kHz a
/// forced SILK-only stream is Hybrid unless the bandwidth is limited, so
/// the Hybrid steps differ from the SILK ones only with `silk_bandwidth`
/// set to wideband.
fn forced_schedule(bitrate: i32, silk_bandwidth: (i32, Bandwidth)) -> Vec<Step> {
    [SILK, CELT, HYBRID, CELT, SILK, HYBRID, SILK, CELT, SILK]
        .into_iter()
        .map(|mode| Step { mode, bitrate, bandwidth: if mode == SILK { silk_bandwidth } else { AUTO_BW } })
        .collect()
}

/// A forced mode that changes every 200 ms, mono, at 16, 24 and 48 kHz,
/// with 10 and 20 ms frames, at a variable and a constant bitrate. Every
/// packet has libopus's first byte: the same mode, bandwidth and channel
/// count, and so the same packet for every switch.
#[test]
fn forced_mode_switches_match_libopus_mono() {
    let wideband = (BW_WB, Bandwidth::Wideband);
    for (rate, bitrate, silk_bandwidth) in [
        (48000usize, 32000, AUTO_BW),
        (48000, 32000, wideband),
        (24000, 28000, AUTO_BW),
        (16000, 24000, AUTO_BW),
    ] {
        for frame_ms in [10usize, 20] {
            for cbr in [false, true] {
                let what = format!("{rate} Hz, {frame_ms} ms, CBR {cbr}, SILK bandwidth {:?}", silk_bandwidth.1);
                let outcome = run(&Stream {
                    rate,
                    channels: 1,
                    frame: rate * frame_ms / 1000,
                    hold: 200 / frame_ms,
                    cbr,
                    pauses: false,
                    schedule: forced_schedule(bitrate, silk_bandwidth),
                    application: AUDIO,
                    offset: 0.0,
                });
                assert_eq!(outcome.tocs, outcome.c_tocs, "{what}: first bytes");
                check(&what, &outcome, 6, &MODE_SWITCH);
                // The packet after a request for CELT-only is still in the
                // old mode: it carries the redundant frame.
                let hold = 200 / frame_ms;
                assert_ne!(mode_of(outcome.tocs[hold]), 2, "{what}: packet {hold}");
                assert_eq!(mode_of(outcome.tocs[hold + 1]), 2, "{what}: packet {}", hold + 1);
            }
        }
    }
}

/// Stereo input at a bitrate where libopus codes stereo. `hybrid` says
/// whether the schedule has Hybrid steps besides SILK-only and CELT-only.
fn stereo_switches(hybrid: bool) {
    let wideband = (BW_WB, Bandwidth::Wideband);
    for (rate, bitrate) in [(48000usize, 64000), (24000, 48000), (16000, 40000)] {
        for frame_ms in [10usize, 20] {
            for cbr in [false, true] {
                let what = format!("{rate} Hz stereo, {frame_ms} ms, CBR {cbr}, Hybrid steps {hybrid}");
                let mut schedule = forced_schedule(bitrate, wideband);
                if !hybrid {
                    for step in &mut schedule {
                        if step.mode == HYBRID {
                            *step = Step { mode: SILK, bitrate, bandwidth: wideband };
                        }
                    }
                }
                let outcome = run(&Stream {
                    rate,
                    channels: 2,
                    frame: rate * frame_ms / 1000,
                    hold: 200 / frame_ms,
                    cbr,
                    pauses: false,
                    schedule,
                    application: AUDIO,
                    offset: 0.0,
                });
                assert_eq!(outcome.tocs, outcome.c_tocs, "{what}: first bytes");
                check(&what, &outcome, 4, &MODE_SWITCH);
            }
        }
    }
}

/// The same with stereo input: SILK-only, limited to wideband, and
/// CELT-only.
#[test]
fn forced_mode_switches_match_libopus_stereo() {
    stereo_switches(false);
}

/// Stereo with Hybrid steps. The CELT layer of a stereo Hybrid packet was
/// corrupt while src/celt.rs limited the intensity band to 0..end instead
/// of start..end (celt_encoder.c 2408). In a packet that also carries a
/// redundant frame the CELT layer has few bits, and the fault then showed
/// as a burst 29 dB above the signal (24 kHz, 10 ms, constant bitrate).
#[test]
fn forced_mode_switches_match_libopus_stereo_hybrid() {
    stereo_switches(true);
}

/// The automatic mode with a bitrate that jumps between speech rates and
/// music rates every 300 ms: the stream moves between SILK-only or Hybrid
/// and CELT-only. Every packet has libopus's first byte.
#[test]
fn bitrate_changes_switch_the_mode_like_libopus() {
    for rate in [48000usize, 24000, 16000] {
        for frame_ms in [10usize, 20] {
            let what = format!("{rate} Hz, {frame_ms} ms");
            let schedule = [12000, 96000, 20000, 80000, 14000, 64000, 10000, 96000]
                .into_iter()
                .map(|bitrate| Step { mode: None, bitrate, bandwidth: AUTO_BW })
                .collect();
            let outcome = run(&Stream {
                rate,
                channels: 1,
                frame: rate * frame_ms / 1000,
                hold: 300 / frame_ms,
                cbr: false,
                pauses: false,
                schedule,
                application: AUDIO,
                offset: 0.0,
            });
            // The switches to and from CELT-only come at libopus's packets.
            let celt: Vec<bool> = outcome.tocs.iter().map(|&t| mode_of(t) == 2).collect();
            let c_celt: Vec<bool> = outcome.c_tocs.iter().map(|&t| mode_of(t) == 2).collect();
            assert_eq!(celt, c_celt, "{what}: CELT-only packets");
            assert!(celt.windows(2).filter(|w| w[0] != w[1]).count() >= 5, "{what}");
            // A switch between SILK-only and Hybrid waits for a pause in
            // the speech, which the two SILK encoders can find some packets
            // apart.
            let differ = (0..outcome.tocs.len()).filter(|&i| outcome.tocs[i] != outcome.c_tocs[i]).count();
            assert!(differ * 50 <= outcome.tocs.len(), "{what}: {differ} first bytes differ");
            check(&what, &outcome, 5, &MODE_SWITCH);
        }
    }
}

/// SILK-only with a bandwidth that changes: SILK moves to another internal
/// rate, and libopus bridges that with a redundant frame at the end of the
/// last packet at the old rate and one at the start of the first packet at
/// the new rate (`silk_bw_switch`). Every packet has libopus's first byte.
#[test]
fn silk_bandwidth_switches_match_libopus() {
    let steps = [
        (BW_WB, Bandwidth::Wideband),
        (BW_NB, Bandwidth::Narrowband),
        (BW_WB, Bandwidth::Wideband),
        (BW_MB, Bandwidth::Mediumband),
        (BW_NB, Bandwidth::Narrowband),
    ];
    // 20 ms frames only: a 10 ms SILK frame at 12 kHz that follows frames
    // of another size has a known fault in src/silk/encode_pulses.rs (the
    // decoder does not reach the encoder's final range; see the ignored
    // test in tests/multistream_enc_robust.rs).
    for rate in [16000usize, 48000] {
        for frame_ms in [20usize] {
            for cbr in [false, true] {
                let what = format!("{rate} Hz, {frame_ms} ms, CBR {cbr}");
                let schedule =
                    steps.into_iter().map(|bandwidth| Step { mode: SILK, bitrate: 24000, bandwidth }).collect();
                // 6 s per step: SILK changes its rate in a pause of the
                // speech, after it has closed its input filter over 2.5 s.
                let outcome = run(&Stream {
                    rate,
                    channels: 1,
                    frame: rate * frame_ms / 1000,
                    hold: 6000 / frame_ms,
                    cbr,
                    pauses: true,
                    schedule,
                    application: AUDIO,
                    offset: 0.0,
                });
                let differ = (0..outcome.tocs.len()).filter(|&i| outcome.tocs[i] != outcome.c_tocs[i]).count();
                if std::env::var("CORE_REDUNDANCY_PRINT").is_ok() {
                    println!("{what}: {differ} first bytes differ, switches {:?}", outcome.switches);
                }
                if !cbr {
                    // The two packets of a switch each carry a redundant
                    // frame: 30 bytes at 24 kb/s (`compute_redundancy_bytes`),
                    // behind at least two bytes of SILK and the first byte.
                    // Most switches fall into a pause, where the packets
                    // around them have 6 to 16 bytes.
                    let c_switches = (1..outcome.c_tocs.len()).filter(|&i| outcome.c_tocs[i] != outcome.c_tocs[i - 1]);
                    for at in c_switches {
                        let pair = &outcome.c_sizes[at - 1..=at];
                        assert!(pair[0] >= 33 && pair[1] >= 33, "{what}: libopus's switch at packet {at}: {pair:?} bytes");
                    }
                    for &at in &outcome.switches {
                        let pair = &outcome.sizes[at - 1..=at];
                        assert!(pair[0] >= 33 && pair[1] >= 33, "{what}: switch at packet {at}: {pair:?} bytes");
                    }
                }
                assert!(differ * 100 <= outcome.tocs.len(), "{what}: {differ} first bytes differ");
                // SILK changes its rate in a pause, where the level is low and
                // the signal-to-noise ratio says little. libopus restarts
                // SILK there, and so does opus1, on the same packets. The
                // windows around a switch then hold digital silence, so their
                // "SNR" compares how much each SILK encoder leaves in the
                // decoded silence (-88 to -111 dB against a reference of
                // 1e-12). Measured against libopus: 5.5 dB at most around a
                // switch (48 kHz, VBR), 1.8 dB on average (48 kHz, VBR;
                // 16 kHz VBR 0.3 dB, CBR -1.4 and -1.5 dB), 0.3 dB over the
                // stream. Before the frames of digital silence told SILK
                // that there was no voice activity, the switches came one
                // packet after libopus's, and the windows were others.
                //
                // What is left is the difference between the fixed-point SILK
                // encoder, which opus1 ports, and the float one of the
                // linked libopus. The fixed-point build of libopus 1.6.1,
                // given the same audio as 16-bit samples, is itself below
                // the float build in these windows: 7.0 dB at most and 2.0 dB
                // on average at 16 kHz with VBR, 0.2 dB on average with CBR,
                // 0.4 and -0.7 dB at 48 kHz. opus1 does not write that
                // build's packets here: with the `Audio` application its
                // SILK reads the frame that the float `dc_reject` filtered
                // (see `OpusEncoder::encode_impl`). With `Voip` and 16-bit
                // input, where its SILK packets are those of the
                // fixed-point build, 23 of the 24 worst windows of opus1
                // equal that build's to 0.1 dB. The other one (48 kHz, CBR)
                // is 1.2 dB off: the redundant CELT frames of a switch
                // differ, because that build's CELT encoder is fixed-point.
                //
                // The redundant frame at the end of the last packet at the
                // old rate starts from a reset CELT encoder, which first
                // codes the 2.5 ms before it into two bytes it throws away
                // (opus_encoder.c:2523-2538). Without those 2.5 ms the
                // frame was coded against an empty overlap: 3.3 dB below
                // libopus on average (48 kHz, CBR).
                check(&what, &outcome, 4, &Margins { window: 9.0, mean: 3.0, stream: 3.0 });
            }
        }
    }
}

/// A stream that moves between SILK-only (Hybrid above 16 kHz) and
/// CELT-only every 500 ms, coded from input with a constant `offset`.
/// Returns, for every step from the second on, the mean of what the C
/// decoder gives for the opus1 stream and for the libopus stream. The
/// first 100 ms of a step are left out: the switch itself is not the
/// subject here.
fn step_means(application: (i32, Application), rate: usize, offset: f32) -> Vec<(bool, f32, f32)> {
    let frame = rate / 50;
    let schedule = [SILK, CELT, SILK, CELT, SILK, CELT];
    let hold = 25;
    let mut pcm = signal(rate, 1, (hold * schedule.len() * frame) as f64 / rate as f64 + 0.05);
    for sample in &mut pcm {
        *sample = 0.5 * *sample + offset;
    }
    let mut c_enc = CEncoder::new(rate, 1, application.0);
    c_enc.complexity(5).bitrate(24000);
    let mut enc = OpusEncoder::new(
        SampleRate::try_from(rate as i32).unwrap(),
        Channels::Mono,
        application.1,
    )
    .unwrap();
    enc.set_complexity(5).unwrap();
    enc.set_bitrate(Bitrate::BitsPerSecond(24000)).unwrap();
    let (mut dec, mut c_dec) = (CDecoder::new(rate, 1), CDecoder::new(rate, 1));
    let mut buf = vec![0u8; 1500];
    let mut sums = vec![(0f64, 0f64); schedule.len()];
    for (i, input) in pcm.chunks_exact(frame).take(hold * schedule.len()).enumerate() {
        let (c_mode, forced) = schedule[i / hold].unwrap();
        c_enc.force_mode(c_mode);
        enc.set_force_mode(Some(forced));
        let c_packet = c_enc.encode(input);
        let n = enc.encode(input, frame, &mut buf).unwrap();
        assert_eq!(buf[0] & 0xFC, c_packet[0] & 0xFC, "packet {i}: first byte");
        let decoded = dec.decode(Some(&buf[..n]), frame, false).unwrap();
        let c_decoded = c_dec.decode(Some(&c_packet), frame, false).unwrap();
        if i % hold >= 5 {
            sums[i / hold].0 += decoded.iter().map(|v| *v as f64).sum::<f64>();
            sums[i / hold].1 += c_decoded.iter().map(|v| *v as f64).sum::<f64>();
        }
    }
    let samples = ((hold - 5) * frame) as f64;
    (1..schedule.len())
        .map(|step| (schedule[step] == CELT, (sums[step].0 / samples) as f32, (sums[step].1 / samples) as f32))
        .collect()
}

/// Input with a constant offset, in a stream that moves between SILK-only
/// or Hybrid and CELT-only. libopus filters the offset out of the frame
/// before either coder sees it (`dc_reject`, or `hp_cutoff` for VoIP;
/// opus_encoder.c:1978-2010), so the decoded stream has none in any mode.
///
/// An encoder that hands SILK the unfiltered input keeps the offset in the
/// SILK packets and not in the CELT-only ones: the decoded signal steps by
/// the offset at every switch. Measured before SILK was fed the filtered
/// frame, with the `Audio` application: a mean of 0.0200 over a SILK-only
/// step for an offset of 0.02, libopus 0.0000.
#[test]
fn an_input_offset_is_removed_in_every_mode_like_libopus() {
    let offset = 0.02;
    for application in [AUDIO, VOIP] {
        for rate in [16000usize, 48000] {
            for (celt, mean, c_mean) in step_means(application, rate, offset) {
                let what = format!("{:?}, {rate} Hz, CELT-only {celt}", application.1);
                if std::env::var("CORE_REDUNDANCY_PRINT").is_ok() {
                    println!("{what}: mean {mean:.4}, libopus {c_mean:.4}");
                }
                // A twentieth of the offset.
                assert!(c_mean.abs() < offset / 20.0, "{what}: libopus {c_mean:.4}");
                assert!(mean.abs() < offset / 20.0, "{what}: mean {mean:.4}, libopus {c_mean:.4}");
            }
        }
    }
}

/// The same switches in streams the tests above leave out: 40 and 60 ms
/// packets, the `Voip` application, and 8 and 12 kHz input.
///
/// A 40 or 60 ms Hybrid or CELT-only packet is made of 20 ms frames. The
/// redundant frame of a switch to CELT-only goes with the last of them, the
/// one of a switch from CELT-only with the first (opus_encoder.c:
/// 1771-1773). With the frame at the wrong end the stream is still valid,
/// but the switch decodes worse: measured 8.2 dB below libopus (48 kHz,
/// 40 ms, the frame with the first 20 ms instead of the last) against
/// 0.4 dB.
#[test]
fn forced_mode_switches_with_long_packets_voip_and_low_rates_match_libopus() {
    let wideband = (BW_WB, Bandwidth::Wideband);
    for (name, rate, channels, frame_ms, bitrate, application, cbr, silk_bandwidth) in [
        ("48 kHz, 40 ms", 48000usize, 1usize, 40usize, 32000, AUDIO, false, AUTO_BW),
        ("48 kHz, 60 ms", 48000, 1, 60, 32000, AUDIO, false, AUTO_BW),
        ("48 kHz, 60 ms, SILK at wideband, CBR", 48000, 1, 60, 32000, AUDIO, true, wideband),
        ("24 kHz, 40 ms, CBR", 24000, 1, 40, 28000, AUDIO, true, AUTO_BW),
        ("16 kHz, 60 ms", 16000, 1, 60, 24000, AUDIO, false, AUTO_BW),
        ("48 kHz stereo, 40 ms, SILK at wideband", 48000, 2, 40, 64000, AUDIO, false, wideband),
        ("VoIP, 48 kHz, 20 ms", 48000, 1, 20, 32000, VOIP, false, AUTO_BW),
        ("VoIP, 16 kHz, 10 ms, CBR", 16000, 1, 10, 24000, VOIP, true, AUTO_BW),
        ("VoIP, 48 kHz, 40 ms", 48000, 1, 40, 32000, VOIP, false, AUTO_BW),
        ("8 kHz, 20 ms", 8000, 1, 20, 16000, AUDIO, false, AUTO_BW),
        ("8 kHz, 10 ms, CBR", 8000, 1, 10, 16000, AUDIO, true, AUTO_BW),
        ("12 kHz, 20 ms", 12000, 1, 20, 20000, AUDIO, false, AUTO_BW),
    ] {
        let schedule = forced_schedule(bitrate, silk_bandwidth);
        let outcome = run(&Stream {
            rate,
            channels,
            frame: rate * frame_ms / 1000,
            hold: 240 / frame_ms,
            cbr,
            pauses: false,
            schedule,
            application,
            offset: 0.0,
        });
        assert_eq!(outcome.tocs, outcome.c_tocs, "{name}: first bytes");
        check(name, &outcome, 4, &MODE_SWITCH);
    }
}

/// The level of the upper half of the spectrum, in dB: the signal's second
/// difference.
fn high_band_level(x: &[f32]) -> f64 {
    let e: f64 = x.windows(3).map(|w| ((w[0] - 2.0 * w[1] + w[2]) as f64).powi(2)).sum();
    10.0 * (e / x.len().max(1) as f64).max(1e-12).log10()
}

/// SILK-only at wideband and Hybrid in turns, with no CELT-only packet in
/// between. The CELT layer of the first Hybrid packet after SILK-only
/// starts afresh: the decoder resets its CELT state when the mode changes
/// (opus_decoder.c, "Make sure to discard any previous CELT state"), and
/// libopus resets the encoder's (opus_encoder.c:2478-2486).
///
/// An encoder that goes on from the energies of the last Hybrid packets
/// predicts from a state the decoder no longer has. Measured without the
/// reset: the band above 8 kHz of the first Hybrid packets decodes up to
/// 14.4 dB below libopus's. With a reset alone: within 1.5 dB. libopus
/// also codes the 2.5 ms before the frame into two bytes it throws away,
/// and codes the frame without prediction (C 2478-2486). With that: within
/// 0.2 dB at a variable bitrate and 0.7 dB at a constant one.
#[test]
fn hybrid_after_silk_only_starts_its_celt_layer_afresh() {
    let rate = 48000usize;
    let frame = rate / 50;
    let hold = 10;
    let steps = 8;
    let pcm = k24_common::speech_fullband(rate, (hold * steps * frame) as f64 / rate as f64 + 0.05);
    for cbr in [false, true] {
        let mut c_enc = CEncoder::new(rate, 1, APP_AUDIO);
        c_enc.complexity(5).vbr(!cbr).bitrate(32000).force_mode(MODE_SILK);
        let mut enc = OpusEncoder::new(
            SampleRate::try_from(rate as i32).unwrap(),
            Channels::Mono,
            Application::Audio,
        )
        .unwrap();
        enc.set_complexity(5).unwrap();
        enc.set_vbr(!cbr);
        enc.set_bitrate(Bitrate::BitsPerSecond(32000)).unwrap();
        enc.set_force_mode(Some(ForcedMode::SilkOnly));
        let (mut dec, mut c_dec) = (CDecoder::new(rate, 1), CDecoder::new(rate, 1));
        let mut buf = vec![0u8; 1500];
        let mut worst = 0f64;
        let mut hybrid_starts = 0;
        for (i, input) in pcm.chunks_exact(frame).take(hold * steps).enumerate() {
            // Forced SILK-only is Hybrid at fullband and SILK-only at
            // wideband. The steps start with Hybrid.
            let hybrid = (i / hold) % 2 == 0;
            let bandwidth = if hybrid { (BW_FB, Bandwidth::Fullband) } else { (BW_WB, Bandwidth::Wideband) };
            c_enc.bandwidth(bandwidth.0);
            enc.set_bandwidth(bandwidth.1);
            let c_packet = c_enc.encode(input);
            let n = enc.encode(input, frame, &mut buf).unwrap();
            assert_eq!(buf[0] & 0xFC, c_packet[0] & 0xFC, "CBR {cbr}, packet {i}: first byte");
            assert_eq!(mode_of(buf[0]) == 1, hybrid, "CBR {cbr}, packet {i}");
            let decoded = dec.decode(Some(&buf[..n]), frame, false).unwrap();
            let c_decoded = c_dec.decode(Some(&c_packet), frame, false).unwrap();
            assert_eq!(enc.final_range(), dec.final_range(), "CBR {cbr}, packet {i}: final range");
            // The first three Hybrid packets after SILK-only packets.
            if hybrid && i >= hold && i % hold < 3 {
                hybrid_starts += (i % hold == 0) as usize;
                let (level, c_level) = (high_band_level(&decoded), high_band_level(&c_decoded));
                if std::env::var("CORE_REDUNDANCY_PRINT").is_ok() {
                    println!("CBR {cbr}, packet {i}: high band {level:.1} dB, libopus {c_level:.1} dB");
                }
                worst = worst.max((level - c_level).abs());
            }
        }
        assert_eq!(hybrid_starts, 3);
        assert!(worst < 1.0, "CBR {cbr}: the high band is {worst:.1} dB from libopus's after a switch to Hybrid");
    }
}
