//! Quality and bitrate of the opus1 multistream encoder against the C
//! libopus encoder under the same settings.
//!
//! Quality: the same signal is encoded by both encoders and both packet
//! sequences are decoded by the C decoder. For every channel the
//! signal-to-noise ratio (SNR) against the input is measured, at the delay
//! of the codec. The limits depend on the kind of stream a channel is in:
//!
//! - A mono stream: opus1 is at most 6 dB below libopus.
//! - A coupled (stereo) stream: opus1 stereo CELT is behind libopus, by up
//!   to 20 dB on pure tones and up to 4 dB on a speech-like signal. The
//!   limits are those gaps with a small margin, so that they catch a further
//!   loss. They are a record of where the stereo encoder stands, not a
//!   target. The same gap shows for a single stereo `OpusEncoder`: the
//!   multistream encoder writes the very packets that encoder writes.
//! - The LFE stream: see `lfe_tone_survives`.
//!
//! Every test prints its measurements; run with `--nocapture` to see them.
//!
//! Bitrate: with variable bitrate the average rate stays near the request.
#![cfg(feature = "std")]

mod ms_support;
mod multistream_enc_common;

use ms_support::*;
use multistream_enc_common::*;
use opus1::multistream::encoder::MultistreamEncoder;
use opus1::multistream::surround::SurroundProbe;
use opus1::{Application, Bandwidth, ForcedMode, OpusEncoder};

const FS: i32 = 48_000;
/// 20 ms.
const FRAME: usize = 960;

/// The layouts of the quality tests with the total bitrate of each.
fn quality_cases() -> Vec<(Case, i32)> {
    vec![
        (Case::family("mono", 1, 1), 64_000),
        (Case::family("stereo", 2, 1), 96_000),
        (Case::family("3 channels", 3, 1), 144_000),
        (Case::family("quad", 4, 1), 192_000),
        (Case::family("5 channels", 5, 1), 240_000),
        (Case::family("5.1", 6, 1), 256_000),
        (Case::family("6.1", 7, 1), 288_000),
        (Case::family("7.1", 8, 1), 320_000),
        (Case::family("family 255, 3 channels", 3, 255), 144_000),
        (Case::family("family 2, 4 channels", 4, 2), 192_000),
        (Case::family("family 2, 6 channels", 6, 2), 256_000),
        (
            Case::generic("generic 3 channels", 3, 2, 1, &[0, 2, 1]),
            144_000,
        ),
        (
            Case::generic("generic 4 channels", 4, 3, 1, &[2, 0, 1, 3]),
            192_000,
        ),
    ]
}

/// The kind of stream an input channel is coded in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StreamKind {
    Mono,
    Coupled,
    Lfe,
}

fn stream_kind(enc: &MultistreamEncoder, channel: usize) -> StreamKind {
    let layout = enc.layout();
    let coded = layout.mapping()[channel] as usize;
    if coded < 2 * layout.coupled_streams() {
        StreamKind::Coupled
    } else if Some(coded - layout.coupled_streams()) == enc.lfe_stream() {
        StreamKind::Lfe
    } else {
        StreamKind::Mono
    }
}

/// SNR of one channel with both encoders, in dB.
struct ChannelSnr {
    kind: StreamKind,
    rust: f64,
    c: f64,
}

/// Encodes `pcm` with both encoders, decodes both with C, and measures every
/// channel against the input. The first 200 ms are left out.
fn measure(case: &Case, bitrate: i32, vbr: bool, pcm: &[f32]) -> Vec<ChannelSnr> {
    let (mut rust, mut c) = encoders(case, FS, Application::Audio);
    configure(&mut rust, &mut c, bitrate, vbr);
    let rust_out = c_decoder(&rust).decode_all(&rust_encode_all(&mut rust, pcm, FRAME));
    let c_out = c_decoder(&rust).decode_all(&c.encode_all(pcm, FRAME));
    // The decoder's output is the input, late by the encoder's lookahead.
    // Searching for the best delay instead goes wrong on a periodic signal,
    // where a whole period more or less fits as well for a while.
    assert_eq!(rust.lookahead() as i32, c.lookahead());
    let lag = rust.lookahead() as isize;
    let skip = FS as usize / 5;
    (0..case.channels)
        .map(|ch| {
            let input = channel(pcm, case.channels, ch);
            ChannelSnr {
                kind: stream_kind(&rust, ch),
                rust: snr_at_lag(&input, &channel(&rust_out, case.channels, ch), lag, skip),
                c: snr_at_lag(&input, &channel(&c_out, case.channels, ch), lag, skip),
            }
        })
        .collect()
}

/// Text for the measurements of one case: `kind opus1/C (difference)`.
fn table(snrs: &[ChannelSnr]) -> String {
    snrs.iter()
        .map(|s| {
            let kind = match s.kind {
                StreamKind::Mono => "mono",
                StreamKind::Coupled => "coupled",
                StreamKind::Lfe => "LFE",
            };
            format!("{kind} {:.1}/{:.1} ({:+.1})", s.rust, s.c, s.rust - s.c)
        })
        .collect::<Vec<_>>()
        .join(", ")
}

/// Runs every layout on one signal and checks the limits.
///
/// - `coupled_gap`: how far below libopus a channel of a coupled stream may
///   be, in dB.
/// - `coupled_floor`: the least SNR of such a channel, in dB.
fn check_quality(signal: &str, coupled_gap: f64, coupled_floor: f64) {
    let mut failures = Vec::new();
    for (case, bitrate) in quality_cases() {
        let samples = 2 * FS as usize;
        let pcm = match signal {
            "tones" => tones(FS, case.channels, case.tone_family(), samples),
            _ => voiced(FS, case.channels, samples),
        };
        for vbr in [true, false] {
            let snrs = measure(&case, bitrate, vbr, &pcm);
            let label = format!(
                "{} {bitrate} b/s {signal} {}",
                case.name,
                if vbr { "VBR" } else { "CBR" }
            );
            println!("{label}: {}", table(&snrs));
            let worst = |kind| {
                snrs.iter()
                    .filter(|s| s.kind == kind)
                    .map(|s| s.rust - s.c)
                    .fold(f64::INFINITY, f64::min)
            };
            if worst(StreamKind::Coupled).is_finite() {
                println!(
                    "    largest gap to libopus in a coupled stream: {:.1} dB",
                    -worst(StreamKind::Coupled)
                );
            }
            for (ch, s) in snrs.iter().enumerate() {
                match s.kind {
                    StreamKind::Mono => {
                        if s.rust < s.c - 6.0 {
                            failures.push(format!(
                                "{label}: channel {ch} (mono stream): {:.1} dB, libopus {:.1} dB",
                                s.rust, s.c
                            ));
                        }
                    }
                    StreamKind::Coupled => {
                        if s.rust < s.c - coupled_gap || s.rust < coupled_floor {
                            failures.push(format!(
                                "{label}: channel {ch} (coupled stream): {:.1} dB, libopus {:.1} dB",
                                s.rust, s.c
                            ));
                        }
                    }
                    // The 80 Hz tone of the LFE channel has its own test.
                    StreamKind::Lfe => {}
                }
            }
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// One steady sine per channel. Measured: mono streams 5.9 dB below libopus
/// at worst (41.8 dB against 47.7 dB); coupled streams up to 19.8 dB below,
/// and never under 23.7 dB.
#[test]
fn snr_on_tones_against_libopus() {
    check_quality("tones", 21.0, 21.0);
}

/// A speech-like signal with a different pitch on every channel. Measured:
/// mono streams 2.5 dB below libopus at worst; coupled streams up to 4.2 dB
/// below, and never under 8.8 dB.
#[test]
fn snr_on_a_speech_like_signal_against_libopus() {
    check_quality("voiced", 7.0, 7.0);
}

/// How the coupled streams of the 7.1 layout compare with a single stereo
/// encoder that codes the two channels of the stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MaskGiven {
    /// The mask of the stream's left channel, then that of its right
    /// channel, as libopus hands it over
    /// (opus_multistream_encoder.c:989-995).
    LeftThenRight,
    /// The mask of the two channels of the next coupled stream.
    OfAnotherStream,
    /// No mask, as for a stream outside surround mode.
    Nothing,
}

/// Codes one second on the 7.1 layout and codes the channels of every
/// coupled stream again with a stereo encoder that has the settings libopus
/// gives a coupled surround stream. The mask comes from an analysis of the
/// same input that is separate from the multistream encoder. Returns how
/// many stream packets are the same bytes, and how many were compared.
fn coupled_streams_equal_to_a_stereo_encoder(vbr: bool, given: MaskGiven) -> (usize, usize) {
    let case = Case::family("7.1", 8, 1);
    let mut enc = rust_encoder(&case, FS, Application::Audio);
    enc.set_bitrate(rust_bitrate(320_000)).unwrap();
    enc.set_vbr(vbr);
    let pcm = voiced(FS, 8, FS as usize);
    let packets = rust_encode_all(&mut enc, &pcm, FRAME);
    let layout = enc.layout().clone();

    // The mask of every frame: 21 values per input channel.
    let mut probe = SurroundProbe::new(8, FS);
    let masks: Vec<Vec<f32>> = pcm
        .chunks_exact(FRAME * 8)
        .map(|frame| {
            let mut band_smr = vec![0f32; 21 * 8];
            assert!(probe.run(frame, FRAME, &mut band_smr));
            band_smr
        })
        .collect();

    let (mut equal, mut compared) = (0, 0);
    for stream in 0..layout.coupled_streams() {
        let left = layout.left_channel(stream, None).unwrap();
        let right = layout.right_channel(stream, None).unwrap();
        let mut single = OpusEncoder::new(FS, 2, Application::Audio).unwrap();
        single.bitrate_bps = enc.stream_encoder(stream).unwrap().bitrate_bps;
        single.use_cbr = !vbr;
        single.set_force_mode(Some(ForcedMode::CeltOnly));
        single.set_bandwidth(Bandwidth::Fullband);
        let mut out = vec![0u8; 1500];
        for (f, frame) in pcm.chunks_exact(FRAME * 8).enumerate() {
            let stereo: Vec<f32> = frame
                .chunks_exact(8)
                .flat_map(|x| [x[left], x[right]])
                .collect();
            let of = |c: usize| &masks[f][21 * c..21 * c + 21];
            let mask = match given {
                MaskGiven::LeftThenRight => [of(left), of(right)].concat(),
                MaskGiven::OfAnotherStream => {
                    let other = (stream + 1) % layout.coupled_streams();
                    [
                        of(layout.left_channel(other, None).unwrap()),
                        of(layout.right_channel(other, None).unwrap()),
                    ]
                    .concat()
                }
                MaskGiven::Nothing => Vec::new(),
            };
            single.set_energy_mask((given != MaskGiven::Nothing).then_some(&mask[..]));
            let n = single.encode(&stereo, FRAME, &mut out).unwrap();
            let parts = split_streams(&packets[f], layout.streams()).unwrap();
            equal += usize::from(parts[stream].packet == out[..n]);
            compared += 1;
        }
    }
    (equal, compared)
}

/// The gap of the coupled streams is not made by the multistream encoder: a
/// coupled stream holds exactly the packets a single stereo encoder writes
/// for the two channels, when that encoder has the settings and the energy
/// mask libopus gives a coupled surround stream.
#[test]
fn a_coupled_stream_is_what_a_stereo_encoder_with_its_mask_writes() {
    for vbr in [true, false] {
        let (equal, compared) = coupled_streams_equal_to_a_stereo_encoder(vbr, MaskGiven::LeftThenRight);
        assert_eq!(compared, 150);
        assert_eq!(equal, compared, "vbr {vbr}");
    }
}

/// The control for the test above: it can tell the masks apart. Without a
/// mask, or with the mask of another stream, the stereo encoder writes
/// other packets in at least four of five frames.
///
/// Some packets stay the same: in a frame where the mask does not change the
/// allocation, the packet does not depend on it. libopus does the same in
/// this experiment (`libopus_also_writes_some_packets_without_the_mask`):
/// without a mask 10 of 150 packets are the same with a variable bitrate,
/// 25 of 150 with a constant one. opus1, measured: 2 and 1 of 150 with a
/// variable bitrate (no mask, mask of another stream), 20 and 10 of 150 with
/// a constant one. The bound was one in ten, below libopus's own figure.
///
/// Exchanging the masks of the left and the right channel is not a control:
/// CELT uses the two halves alike (celt_encoder.c:2114-2165), so the packets
/// stay the same. tests/multistream_enc_streams.rs compares the mask of
/// every stream value by value instead.
#[test]
fn a_coupled_stream_differs_from_a_stereo_encoder_with_another_mask() {
    for vbr in [true, false] {
        for given in [MaskGiven::Nothing, MaskGiven::OfAnotherStream] {
            let (equal, compared) = coupled_streams_equal_to_a_stereo_encoder(vbr, given);
            println!("vbr {vbr}, {given:?}: {equal} of {compared} packets are the same");
            assert!(
                5 * equal < compared,
                "vbr {vbr}, {given:?}: {equal} of {compared} packets are the same"
            );
        }
    }
}

/// The figure behind the bound of the test above: the same experiment with
/// libopus. Its 7.1 encoder codes one second; a libopus stereo encoder with
/// the settings of the test above and no mask codes the channels of each
/// coupled stream again. Measured: 10 of 150 packets are the same with a
/// variable bitrate, 25 of 150 with a constant one, so fewer than one in
/// five.
#[test]
fn libopus_also_writes_some_packets_without_the_mask() {
    use audiopus_sys as sys;
    for vbr in [true, false] {
        let case = Case::family("7.1", 8, 1);
        let layout = rust_encoder(&case, FS, Application::Audio).layout().clone();
        let mut c = c_encoder(&case, FS, Application::Audio);
        c.set_bitrate(320_000);
        c.set_vbr(vbr);
        let pcm = voiced(FS, 8, FS as usize);
        let packets = c.encode_all(&pcm, FRAME);
        let (mut equal, mut compared) = (0, 0);
        for stream in 0..layout.coupled_streams() {
            let left = layout.left_channel(stream, None).unwrap();
            let right = layout.right_channel(stream, None).unwrap();
            let rate = c.stream_bitrate(stream);
            let mut err = 0;
            // SAFETY: plain calls of the C API; the encoder is destroyed
            // below, and every buffer holds what the call reads or writes.
            let single =
                unsafe { sys::opus_encoder_create(FS, 2, sys::OPUS_APPLICATION_AUDIO, &mut err) };
            assert!(!single.is_null(), "opus_encoder_create: {err}");
            unsafe {
                sys::opus_encoder_ctl(single, sys::OPUS_SET_BITRATE_REQUEST, rate);
                sys::opus_encoder_ctl(single, sys::OPUS_SET_VBR_REQUEST, vbr as i32);
                sys::opus_encoder_ctl(single, OPUS_SET_FORCE_MODE_REQUEST, MODE_CELT_ONLY);
                sys::opus_encoder_ctl(
                    single,
                    sys::OPUS_SET_BANDWIDTH_REQUEST,
                    sys::OPUS_BANDWIDTH_FULLBAND,
                );
            }
            let mut out = vec![0u8; 1500];
            for (f, frame) in pcm.as_chunks::<{ FRAME * 8 }>().0.iter().enumerate() {
                let stereo: Vec<f32> = (0..FRAME)
                    .flat_map(|i| [frame[i * 8 + left], frame[i * 8 + right]])
                    .collect();
                let n = unsafe {
                    sys::opus_encode_float(single, stereo.as_ptr(), FRAME as i32, out.as_mut_ptr(), 1500)
                };
                assert!(n > 0, "opus_encode_float: {n}");
                let parts = split_streams(&packets[f], layout.streams()).unwrap();
                equal += usize::from(parts[stream].packet == out[..n as usize]);
                compared += 1;
            }
            unsafe { sys::opus_encoder_destroy(single) };
        }
        println!("libopus, vbr {vbr}, no mask: {equal} of {compared} packets are the same");
        assert_eq!(compared, 150);
        assert!(
            5 * equal < compared,
            "vbr {vbr}: {equal} of {compared} packets are the same"
        );
    }
}

/// The 80 Hz tone on the LFE channel of 5.1, 6.1 and 7.1.
///
/// libopus codes the LFE stream with a tuning of its own inside CELT, and
/// so does opus1: the stream encoder passes the LFE flag on to its CELT
/// layer. Measured: 18.2 to 27.8 dB, where libopus has 18.2 to 27.9 dB.
/// Without the tuning it was 10.8 to 13.4 dB. The limit here is a floor of
/// 15 dB, which the untuned stream does not reach.
/// `lfe_tone_is_within_2_db_of_libopus` holds the limit against libopus.
#[test]
fn lfe_tone_survives() {
    for (channels, bitrate) in [(6usize, 256_000), (7, 288_000), (8, 320_000)] {
        let case = Case::family("surround", channels, 1);
        let pcm = tones(FS, channels, 1, 2 * FS as usize);
        for vbr in [true, false] {
            let snrs = measure(&case, bitrate, vbr, &pcm);
            let lfe = &snrs[channels - 1];
            assert_eq!(lfe.kind, StreamKind::Lfe);
            println!(
                "{channels} channels {}: LFE tone {:.1} dB, libopus {:.1} dB",
                if vbr { "VBR" } else { "CBR" },
                lfe.rust,
                lfe.c
            );
            assert!(
                lfe.rust >= 15.0,
                "{channels} channels: LFE tone {:.1} dB",
                lfe.rust
            );
        }
    }
}

/// Measured: the LFE tone is 1.3 dB below libopus at worst (6.1, constant
/// bitrate) and equal within 0.1 dB in four of the six cases.
#[test]
fn lfe_tone_is_within_2_db_of_libopus() {
    for (channels, bitrate) in [(6usize, 256_000), (7, 288_000), (8, 320_000)] {
        let case = Case::family("surround", channels, 1);
        let pcm = tones(FS, channels, 1, 2 * FS as usize);
        for vbr in [true, false] {
            let snrs = measure(&case, bitrate, vbr, &pcm);
            let lfe = &snrs[channels - 1];
            assert!(
                lfe.rust >= lfe.c - 2.0,
                "{channels} channels, vbr {vbr}: LFE tone {:.1} dB, libopus {:.1} dB",
                lfe.rust,
                lfe.c
            );
        }
    }
}

/// Average bitrate of a variable-bitrate stream over 5 seconds, both
/// encoders. Returns bits per second of opus1 and of libopus.
fn average_rates(case: &Case, bitrate: i32, pcm: &[f32]) -> (f64, f64) {
    let (mut rust, mut c) = encoders(case, FS, Application::Audio);
    configure(&mut rust, &mut c, bitrate, true);
    let seconds = (pcm.len() / case.channels) as f64 / FS as f64;
    let bits = |packets: &[Vec<u8>]| 8.0 * packets.iter().map(|p| p.len()).sum::<usize>() as f64;
    (
        bits(&rust_encode_all(&mut rust, pcm, FRAME)) / seconds,
        bits(&c.encode_all(pcm, FRAME)) / seconds,
    )
}

/// Checks the average rate of every layout at five bitrates on one signal.
///
/// The rate is compared with the request, not with libopus: libopus itself
/// can be far off. 5.1 at 60 kb/s on the speech-like signal uses 150 kb/s
/// there, an effect of its surround masking on a Hybrid stream.
///
/// Measured for opus1: 9.7 % under to 4.2 % over the request from 16 kb/s
/// per channel on, and 17.8 % under to 25.7 % over below that.
fn check_rates(signal: &str) {
    let cases = [
        Case::family("mono", 1, 1),
        Case::family("stereo", 2, 1),
        Case::family("quad", 4, 1),
        Case::family("5.1", 6, 1),
        Case::family("7.1", 8, 1),
        Case::family("family 255, 3 channels", 3, 255),
        Case::family("family 2, 6 channels", 6, 2),
    ];
    let mut failures = Vec::new();
    for case in &cases {
        let samples = 5 * FS as usize;
        let pcm = match signal {
            "tones" => tones(FS, case.channels, case.tone_family(), samples),
            _ => voiced(FS, case.channels, samples),
        };
        for per_channel in [6_000, 10_000, 16_000, 32_000, 64_000] {
            let bitrate = per_channel * case.channels as i32;
            let (rust, c) = average_rates(case, bitrate, &pcm);
            let ratio = rust / bitrate as f64;
            println!(
                "{} {signal} {bitrate} b/s: opus1 {rust:.0} ({:+.1} %), libopus {c:.0} ({:+.1} %)",
                case.name,
                100.0 * (ratio - 1.0),
                100.0 * (c / bitrate as f64 - 1.0)
            );
            // From 16 kb/s per channel: 15 % under to 10 % over. Below
            // that the fixed cost per stream weighs more; there the rate
            // may reach 1.6 times the request.
            let ok = if per_channel >= 16_000 {
                (0.85..=1.10).contains(&ratio)
            } else {
                (0.5..=1.6).contains(&ratio)
            };
            if !ok {
                failures.push(format!(
                    "{} {signal} {bitrate} b/s: {rust:.0} b/s",
                    case.name
                ));
            }
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

#[test]
fn variable_bitrate_stays_near_the_request_on_tones() {
    check_rates("tones");
}

#[test]
fn variable_bitrate_stays_near_the_request_on_a_speech_like_signal() {
    check_rates("voiced");
}
