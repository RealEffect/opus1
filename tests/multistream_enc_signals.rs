//! The opus1 multistream encoder on signals and settings the other test
//! files do not use, against C libopus.
//!
//! - One channel sounds at a time, at all five sampling rates and with
//!   16-bit input: the sound comes out of that channel only, at its level.
//! - In-band forward error correction (FEC): the C decoder rebuilds a lost
//!   frame of a SILK stream from the next packet.
//! - The average bitrate at 16 kHz and with the application `Voip`, where
//!   the stream encoders choose SILK. tests/multistream_enc_quality.rs
//!   measures it at 48 kHz with the application `Audio` only.
#![cfg(feature = "std")]

mod ms_support;
mod multistream_enc_common;

use ms_support::*;
use multistream_enc_common::*;
use opus1::Application;
use opus1::multistream::MultistreamEncoder;

/// Sum of squares.
fn energy(x: &[f32]) -> f64 {
    x.iter().map(|&v| v as f64 * v as f64).sum()
}

/// `channels` slots of `slot` samples each. Channel `c` sounds in slot `c`
/// and is silent in the others: a sine that rises from 300 Hz to 3 kHz, or
/// to 0.3 times the sampling rate when that is lower. The LFE channel has a
/// 60 Hz sine instead, because its stream only keeps low frequencies.
fn one_channel_at_a_time(fs: i32, channels: usize, lfe: Option<usize>, slot: usize) -> Vec<f32> {
    let mut pcm = vec![0f32; channels * slot * channels];
    let top = (0.3 * fs as f64).min(3000.0);
    for c in 0..channels {
        let mut phase = 0f64;
        for i in 0..slot {
            let hz = if lfe == Some(c) {
                60.0
            } else {
                300.0 + (top - 300.0) * i as f64 / slot as f64
            };
            phase += 2.0 * std::f64::consts::PI * hz / fs as f64;
            // Fade in and out over 5 ms.
            let edge = (i.min(slot - 1 - i) as f64 / (fs as f64 * 0.005)).min(1.0);
            pcm[(c * slot + i) * channels + c] = (0.3 * edge * phase.sin()) as f32;
        }
    }
    pcm
}

/// What `isolation` measured for one encoder, in dB.
struct Isolation {
    /// The level of a channel's own sound after decoding, against the
    /// input: the value farthest from 0 over all channels.
    own: f64,
    /// The strongest sound on any other channel while one channel sounds,
    /// against that channel's level.
    leak: f64,
}

/// Decodes `packets` with C and measures, for every coded channel, the
/// middle three fifths of its slot.
fn isolation(
    case: &Case,
    enc: &MultistreamEncoder,
    pcm: &[f32],
    packets: &[Vec<u8>],
    slot: usize,
) -> Isolation {
    let channels = case.channels;
    let out = c_decoder(enc).decode_all(packets);
    assert_eq!(out.len(), pcm.len(), "{}: decoded length", case.name);
    let lag = enc.lookahead();
    let mapping = enc.layout().mapping();
    let mut result = Isolation {
        own: 0.0,
        leak: f64::NEG_INFINITY,
    };
    for src in case.coded_channels() {
        let (from, to) = (src * slot + slot / 5, (src + 1) * slot - slot / 5);
        let input = energy(&channel(pcm, channels, src)[from..to]);
        // The output is late by the encoder's lookahead.
        let to_out = (to + lag).min(out.len() / channels);
        let level: Vec<f64> = (0..channels)
            .map(|j| energy(&channel(&out, channels, j)[from + lag..to_out]))
            .collect();
        let own = 10.0 * (level[src] / input).log10();
        if own.abs() > result.own.abs() {
            result.own = own;
        }
        for j in 0..channels {
            // A generic layout can feed two outputs from one coded channel.
            if j != src && mapping[j] != mapping[src] {
                result.leak = result
                    .leak
                    .max(10.0 * (level[j].max(1e-30) / level[src]).log10());
            }
        }
    }
    result
}

/// Measured over all layouts and sampling rates: the own level is within
/// 0.6 dB of the input (libopus: 0.05 dB), and the other channels are at
/// -91 dB or below (libopus: -117 dB, the noise floor of its decoder).
///
/// The limits hold for the LFE channel with and without the LFE tuning of
/// CELT: libopus, which has it, keeps the 60 Hz sine within 0.05 dB.
#[test]
fn one_channel_at_a_time_comes_out_of_that_channel_only() {
    let mut worst = Isolation {
        own: 0.0,
        leak: f64::NEG_INFINITY,
    };
    for fs in [48_000, 24_000, 16_000, 12_000, 8_000] {
        for case in CASES {
            let channels = case.channels;
            let slot = fs as usize * 3 / 10;
            let frame_size = fs as usize / 50;
            let lfe = lfe_channel(channels, case.tone_family());
            let pcm = one_channel_at_a_time(fs, channels, lfe, slot);
            let (mut rust, mut c) = encoders(case, fs, Application::Audio);
            configure(&mut rust, &mut c, 48_000 * channels as i32, true);
            let packets = rust_encode_all(&mut rust, &pcm, frame_size);
            let r = isolation(case, &rust, &pcm, &packets, slot);
            let l = isolation(case, &rust, &pcm, &c.encode_all(&pcm, frame_size), slot);
            println!(
                "{fs} Hz {}: own level {:+.2} dB (libopus {:+.2}), other channels {:.1} dB \
                 (libopus {:.1})",
                case.name, r.own, l.own, r.leak, l.leak
            );
            assert!(
                r.own.abs() < 1.5,
                "{fs} Hz {}: own level {:+.2} dB",
                case.name,
                r.own
            );
            assert!(
                r.leak < -60.0,
                "{fs} Hz {}: other channels at {:.1} dB",
                case.name,
                r.leak
            );
            if r.own.abs() > worst.own.abs() {
                worst.own = r.own;
            }
            worst.leak = worst.leak.max(r.leak);
        }
    }
    println!(
        "worst: own level {:+.2} dB, other channels {:.1} dB",
        worst.own, worst.leak
    );
}

/// The same with 16-bit input and other frame sizes.
#[test]
fn one_channel_at_a_time_with_16_bit_input() {
    let fs = 48_000;
    for case in CASES {
        for frame_size in [240usize, 1920] {
            let channels = case.channels;
            let slot = fs as usize * 3 / 10;
            let lfe = lfe_channel(channels, case.tone_family());
            let pcm16 = to_i16(&one_channel_at_a_time(fs, channels, lfe, slot));
            let pcm: Vec<f32> = pcm16.iter().map(|&s| s as f32 / 32768.0).collect();
            let mut enc = rust_encoder(case, fs, Application::RestrictedLowDelay);
            // A 5 ms frame needs more bits to keep a 60 Hz sine.
            let per_channel = if frame_size < 480 { 96_000 } else { 48_000 };
            enc.set_bitrate(rust_bitrate(per_channel * channels as i32))
                .unwrap();
            let mut out = vec![0u8; max_packet_bytes(enc.layout().streams())];
            let packets: Vec<Vec<u8>> = pcm16
                .chunks_exact(frame_size * channels)
                .map(|frame| {
                    let n = enc.encode_i16(frame, frame_size, &mut out).unwrap();
                    out[..n].to_vec()
                })
                .collect();
            let whole = packets.len() * frame_size * channels;
            let r = isolation(case, &enc, &pcm[..whole], &packets, slot);
            println!(
                "{} {}: own level {:+.2} dB, other channels {:.1} dB",
                case.name,
                ms(frame_size, fs),
                r.own,
                r.leak
            );
            assert!(
                r.own.abs() < 1.5,
                "{}: own level {:+.2} dB",
                case.name,
                r.own
            );
            assert!(
                r.leak < -60.0,
                "{}: other channels at {:.1} dB",
                case.name,
                r.leak
            );
        }
    }
}

/// A speech-like signal with pauses: words of 150 to 400 ms with a gliding
/// pitch and a few harmonics, a little noise, and silence between them.
/// Every channel has its own pitch and its own timing.
fn words(fs: i32, channels: usize, samples: usize) -> Vec<f32> {
    let mut pcm = vec![0f32; samples * channels];
    for c in 0..channels {
        let mut state = 77u32.wrapping_mul(c as u32 + 1);
        let mut rand = move || {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            (state >> 8) as f64 / (1u32 << 24) as f64
        };
        let mut i = 0;
        let mut phase = 0f64;
        while i < samples {
            let word = ((0.15 + 0.25 * rand()) * fs as f64) as usize;
            let pause = ((0.03 + 0.07 * rand()) * fs as f64) as usize;
            let (f0, glide) = (95.0 + 22.0 * c as f64 + 60.0 * rand(), 0.7 + 0.6 * rand());
            for k in 0..word.min(samples - i) {
                let x = k as f64 / word as f64;
                phase += 2.0 * std::f64::consts::PI * f0 * (1.0 + (glide - 1.0) * x) / fs as f64;
                let envelope = (std::f64::consts::PI * x).sin().powf(0.5);
                let mut v = 0.0;
                for h in 1..=6 {
                    v += (h as f64 * phase).sin() / (h as f64).powf(1.2);
                }
                v += 0.05 * (2.0 * rand() - 1.0);
                pcm[(i + k) * channels + c] = (0.2 * envelope * v) as f32;
            }
            i += word + pause;
        }
    }
    pcm
}

/// Decodes `packets` three times with C: all packets; every fifth packet
/// lost and rebuilt from the FEC data of the next one; every fifth packet
/// lost and concealed. Returns the SNR of the lost frames against the
/// decoding without loss, in dB: rebuilt, concealed.
fn fec_gain(enc: &MultistreamEncoder, packets: &[Vec<u8>], frame_size: usize) -> (f64, f64) {
    let (mut full, mut with_fec, mut concealed) = (c_decoder(enc), c_decoder(enc), c_decoder(enc));
    let (mut signal, mut fec_error, mut plc_error) = (0f64, 0f64, 0f64);
    let error = |a: &[f32], b: &[f32]| -> f64 {
        a.iter().zip(b).map(|(x, y)| ((x - y) as f64).powi(2)).sum()
    };
    for f in 0..packets.len() - 1 {
        let good = full
            .decode_float(Some(&packets[f]), frame_size, false)
            .unwrap();
        if f % 5 == 4 {
            let rebuilt = with_fec
                .decode_float(Some(&packets[f + 1]), frame_size, true)
                .unwrap();
            let hidden = concealed.decode_float(None, frame_size, false).unwrap();
            assert_eq!(rebuilt.len(), good.len());
            assert_eq!(hidden.len(), good.len());
            signal += energy(&good);
            fec_error += error(&good, &rebuilt);
            plc_error += error(&good, &hidden);
        } else {
            with_fec
                .decode_float(Some(&packets[f]), frame_size, false)
                .unwrap();
            concealed
                .decode_float(Some(&packets[f]), frame_size, false)
                .unwrap();
        }
    }
    (snr_db(signal, fec_error), snr_db(signal, plc_error))
}

/// `set_inband_fec` and `set_packet_loss_perc` reach the stream encoders:
/// with both set, a layout of mono SILK streams carries data from which the
/// C decoder rebuilds a lost frame far better than it conceals it.
///
/// Measured over the five sampling rates, SNR of the lost frames against
/// the decoding without loss: rebuilt 17.3 to 21.7 dB, concealed 5.3 to
/// 7.0 dB. With the packets of the libopus encoder: rebuilt 17.0 to 20.9 dB.
#[test]
fn inband_fec_of_silk_streams_rebuilds_a_lost_frame_in_libopus() {
    // Bitrates per channel at which a stream is SILK or Hybrid and has
    // room for the FEC data.
    for (fs, per_channel) in [
        (16_000, 24_000),
        (48_000, 20_000),
        (8_000, 24_000),
        (12_000, 20_000),
        (24_000, 24_000),
    ] {
        // Three mono streams, and a single one.
        for case in [
            Case::family("family 255, 3 channels", 3, 255),
            Case::family("mono", 1, 0),
        ] {
            let frame_size = fs as usize / 50;
            let channels = case.channels;
            let pcm = words(fs, channels, 150 * frame_size);
            let label = format!("{} {fs} Hz", case.name);
            let mut snr = [(0f64, 0f64); 2];
            let mut c_snr = (0f64, 0f64);
            for fec in [false, true] {
                let (mut rust, mut c) = encoders(&case, fs, Application::Voip);
                configure(&mut rust, &mut c, per_channel * channels as i32, true);
                rust.set_inband_fec(fec);
                rust.set_packet_loss_perc(if fec { 25 } else { 0 }).unwrap();
                c.set_inband_fec(fec)
                    .set_packet_loss_perc(if fec { 25 } else { 0 });
                // Speech: without FEC the speech and music analysis of both
                // encoders moves some of these streams to CELT (at 16 and
                // 24 kHz, the same packets in both).
                rust.set_signal(opus1::Signal::Voice);
                c.ctl(audiopus_sys::OPUS_SET_SIGNAL_REQUEST, audiopus_sys::OPUS_SIGNAL_VOICE);
                assert_eq!(rust.inband_fec(), fec);
                let packets = rust_encode_all(&mut rust, &pcm, frame_size);
                // The test is about SILK streams.
                let silk = packets
                    .iter()
                    .flat_map(|p| split_streams(p, rust.layout().streams()).unwrap())
                    .filter(|part| toc(part.toc).mode != Mode::Celt)
                    .count();
                assert!(
                    silk >= packets.len() * rust.layout().streams() * 9 / 10,
                    "{label}: only {silk} SILK or Hybrid streams"
                );
                snr[fec as usize] = fec_gain(&rust, &packets, frame_size);
                if fec {
                    c_snr = fec_gain(&rust, &c.encode_all(&pcm, frame_size), frame_size);
                }
            }
            let (off, on) = (snr[0], snr[1]);
            println!(
                "{label}: lost frames rebuilt {:.1} dB, concealed {:.1} dB; without FEC \
                 {:.1} / {:.1} dB; libopus encoder with FEC {:.1} / {:.1} dB",
                on.0, on.1, off.0, off.1, c_snr.0, c_snr.1
            );
            // With FEC the rebuilt frame is far better than concealment.
            assert!(
                on.0 > on.1 + 8.0,
                "{label}: rebuilt {:.1} dB, concealed {:.1} dB",
                on.0,
                on.1
            );
            // Without FEC there is nothing to rebuild from: the decoder
            // conceals, and gains nothing.
            assert!(
                off.0 < off.1 + 1.0,
                "{label}: without FEC {:.1} / {:.1} dB",
                off.0,
                off.1
            );
            // The data is as useful as that of the libopus encoder.
            assert!(
                on.0 > c_snr.0 - 3.0,
                "{label}: rebuilt {:.1} dB, with the libopus encoder {:.1} dB",
                on.0,
                c_snr.0
            );
        }
    }
}

/// Average bitrate over 5 s of 20 ms frames with variable bitrate, as a
/// fraction of the request: opus1, libopus.
fn rate_ratio(
    case: &Case,
    fs: i32,
    application: Application,
    bitrate: i32,
    pcm: &[f32],
) -> (f64, f64) {
    let frame_size = fs as usize / 50;
    let (mut rust, mut c) = encoders(case, fs, application);
    configure(&mut rust, &mut c, bitrate, true);
    let seconds = (pcm.len() / case.channels) as f64 / fs as f64;
    let bits = |packets: &[Vec<u8>]| 8.0 * packets.iter().map(|p| p.len()).sum::<usize>() as f64;
    (
        bits(&rust_encode_all(&mut rust, pcm, frame_size)) / seconds / bitrate as f64,
        bits(&c.encode_all(pcm, frame_size)) / seconds / bitrate as f64,
    )
}

/// The layouts, settings and bitrates per channel of the two rate tests
/// below.
fn rate_grid() -> Vec<(Case, i32, Application, i32)> {
    let cases = [
        Case::family("mono", 1, 0),
        Case::family("stereo", 2, 0),
        Case::family("family 255, 3 channels", 3, 255),
        Case::family("5.1", 6, 1),
        Case::family("family 2, 6 channels", 6, 2),
    ];
    let settings: [(i32, Application, &[i32]); 3] = [
        (16_000, Application::Audio, &[16_000, 64_000]),
        (16_000, Application::Voip, &[64_000]),
        (48_000, Application::Voip, &[16_000, 32_000]),
    ];
    let mut grid = Vec::new();
    for (fs, application, rates) in settings {
        for case in cases {
            for &per_channel in rates {
                grid.push((case, fs, application, per_channel));
            }
        }
    }
    grid
}

/// Runs the grid on the speech-like signal of ms_support. Returns the
/// lowest and the highest fraction of the request opus1 used, and prints
/// every measurement.
fn measure_rates() -> Vec<(String, f64, f64)> {
    let mut rows = Vec::new();
    for (case, fs, application, per_channel) in rate_grid() {
        let pcm = voiced(fs, case.channels, 5 * fs as usize);
        let bitrate = per_channel * case.channels as i32;
        let (rust, c) = rate_ratio(&case, fs, application, bitrate, &pcm);
        let label = format!("{} {fs} Hz {application:?} {bitrate} b/s", case.name);
        println!(
            "{label}: opus1 {:+.1} %, libopus {:+.1} %",
            100.0 * (rust - 1.0),
            100.0 * (c - 1.0)
        );
        rows.push((label, rust, c));
    }
    rows
}

/// The speech and music analysis of both encoders moves the streams that
/// start in SILK or Hybrid to CELT after about 0.4 s on this signal, and
/// CELT fills the bitrate (tests/multistream_e2e_modes.rs). Before opus1
/// ran the analysis, its streams stayed in SILK or Hybrid, and SILK, which
/// does not spend more than it needs, left up to 34 % of the request
/// unused.
///
/// This test keeps the wider limits of that time: the rate never exceeds
/// the request by more than 10 %, the limit of the design, and is never
/// below 60 % of it. Measured now: 87 % (three mono streams at 48 kHz,
/// VoIP, 16 kb/s each; libopus 102 %) to 104 %.
#[test]
fn average_rate_at_16_khz_and_for_voip_stays_below_the_request() {
    let rows = measure_rates();
    let lowest = rows.iter().map(|r| r.1).fold(f64::INFINITY, f64::min);
    let highest = rows.iter().map(|r| r.1).fold(0.0, f64::max);
    println!(
        "opus1 uses {:.0} % to {:.0} % of the request",
        100.0 * lowest,
        100.0 * highest
    );
    for (label, rust, _) in &rows {
        assert!(
            (0.60..=1.10).contains(rust),
            "{label}: {:.0} % of the request",
            100.0 * rust
        );
    }
    // Ambisonics streams are CELT whatever the signal: within the limits
    // of the design already.
    for (label, rust, _) in rows.iter().filter(|r| r.0.starts_with("family 2,")) {
        assert!(
            (0.85..=1.10).contains(rust),
            "{label}: {:.0} %",
            100.0 * rust
        );
    }
}

/// The limits of the design: 85 % to 110 % of the request.
#[cfg(feature = "analysis")]
#[test]
fn average_rate_at_16_khz_and_for_voip_is_within_the_design_limits() {
    let low: Vec<String> = measure_rates()
        .into_iter()
        .filter(|(_, rust, _)| !(0.85..=1.10).contains(rust))
        .map(|(label, rust, c)| {
            format!(
                "{label}: {:.0} % of the request, libopus {:.0} %",
                100.0 * rust,
                100.0 * c
            )
        })
        .collect();
    assert!(low.is_empty(), "{low:#?}");
}
