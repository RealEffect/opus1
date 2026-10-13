//! Signals and encoder settings other than speech at the encoder's
//! defaults.
//!
//! The other `silkdec_*` tests use one speech recording and a C encoder
//! with default settings. The streams here come from synthetic signals that
//! push the decoder to its limits (the longest pitch periods, full-scale
//! and clipping signals, channels in opposite phase, silence) and from
//! encoders set up differently: the audio application, 48 kHz input,
//! complexity 0 and 10, constant bitrate, very low and high bitrates,
//! forward error correction (FEC) tuned for 5 to 60 % loss, discontinuous
//! transmission (DTX).
//!
//! Losses are drawn at random from a fixed seed. Every test requires
//! 16-bit output identical to C libopus for received, concealed and
//! FEC-recovered packets.

mod silkdec_common;
use silkdec_common::*;

use audiopus_sys as sys;

const SECONDS: f64 = 1.5;

/// Every test signal: mono ones, and stereo ones built from two of them.
/// Returns the name, the channel count and the interleaved samples.
fn signals(rate: usize) -> Vec<(&'static str, usize, Vec<i16>)> {
    let n = (rate as f64 * SECONDS) as usize;
    let negated = |x: &[i16]| x.iter().map(|&v| v.saturating_neg()).collect::<Vec<i16>>();
    let delayed = |x: &[i16]| {
        let mut out = vec![0i16; 1];
        out.extend_from_slice(&x[..x.len() - 1]);
        out
    };
    let (voice, tone, noise, square, click, quiet) = (
        low_voice(rate, n),
        sweep(rate, n),
        noise_bursts(rate, n),
        clipped(rate, n),
        clicks(rate, n),
        mostly_silence(rate, n),
    );
    vec![
        ("low voice", 1, voice.clone()),
        ("sweep", 1, tone.clone()),
        ("noise bursts", 1, noise.clone()),
        ("clipped tone", 1, square.clone()),
        ("clicks", 1, click.clone()),
        ("mostly silence", 1, quiet.clone()),
        ("low voice left, noise bursts right", 2, stereo_from(&voice, &noise)),
        ("low voice in opposite phase", 2, stereo_from(&voice, &negated(&voice))),
        ("clipped tone left, silence right", 2, stereo_from(&square, &vec![0; n])),
        ("clicks, right one sample late", 2, stereo_from(&click, &delayed(&click))),
        ("mostly silence left, clipped tone right", 2, stereo_from(&quiet, &square)),
    ]
}

/// How one encoder is set up.
struct Setup {
    name: &'static str,
    application: i32,
    input_rate: usize,
    complexity: i32,
    vbr: bool,
    /// Bitrate per channel.
    bitrate: i32,
    /// Expected loss in percent for FEC, or `None` for no FEC.
    fec_loss: Option<i32>,
    dtx: bool,
}

const SETUPS: [Setup; 6] = [
    Setup {
        name: "audio, 48 kHz input, complexity 10, FEC for 25 % loss",
        application: sys::OPUS_APPLICATION_AUDIO,
        input_rate: 48000,
        complexity: 10,
        vbr: true,
        bitrate: 32000,
        fec_loss: Some(25),
        dtx: false,
    },
    Setup {
        name: "voip, complexity 0, constant bitrate, FEC for 5 % loss",
        application: sys::OPUS_APPLICATION_VOIP,
        input_rate: 16000,
        complexity: 0,
        vbr: false,
        bitrate: 28000,
        fec_loss: Some(5),
        dtx: false,
    },
    Setup {
        name: "voip, 6 kbit/s",
        application: sys::OPUS_APPLICATION_VOIP,
        input_rate: 16000,
        complexity: 5,
        vbr: true,
        bitrate: 6000,
        fec_loss: None,
        dtx: false,
    },
    Setup {
        name: "audio, constant bitrate 40 kbit/s, FEC for 60 % loss",
        application: sys::OPUS_APPLICATION_AUDIO,
        input_rate: 16000,
        complexity: 10,
        vbr: false,
        bitrate: 40000,
        fec_loss: Some(60),
        dtx: false,
    },
    Setup {
        name: "voip, DTX, FEC for 25 % loss",
        application: sys::OPUS_APPLICATION_VOIP,
        input_rate: 16000,
        complexity: 10,
        vbr: true,
        bitrate: 30000,
        fec_loss: Some(25),
        dtx: true,
    },
    Setup {
        name: "voip, 48 kHz input, 80 kbit/s",
        application: sys::OPUS_APPLICATION_VOIP,
        input_rate: 48000,
        complexity: 7,
        vbr: true,
        bitrate: 80000,
        fec_loss: Some(15),
        dtx: false,
    },
];

/// A SILK-only encoder set up as `setup` says.
fn encoder(setup: &Setup, channels: usize, bandwidth: i32) -> CEnc {
    let mut enc = CEnc::with(setup.input_rate, channels, setup.application);
    enc.ctl(OPUS_SET_FORCE_MODE_REQUEST, MODE_SILK)
        .ctl(sys::OPUS_SET_BANDWIDTH_REQUEST, bandwidth)
        .ctl(sys::OPUS_SET_BITRATE_REQUEST, setup.bitrate * channels as i32)
        .ctl(sys::OPUS_SET_COMPLEXITY_REQUEST, setup.complexity)
        .ctl(sys::OPUS_SET_VBR_REQUEST, setup.vbr as i32)
        .ctl(sys::OPUS_SET_DTX_REQUEST, setup.dtx as i32)
        .force_channels(channels as i32);
    if let Some(loss) = setup.fec_loss {
        enc.fec(loss);
    }
    enc
}

/// Every signal with every encoder setup. Bandwidth and packet duration
/// change from stream to stream, so that all twelve combinations occur.
#[test]
fn synthetic_signals_and_encoder_settings_match_libopus() {
    let mut failures = Vec::new();
    let mut total = Stats::default();
    let mut k = 0usize;
    let mut seen = std::collections::BTreeSet::new();
    for setup in &SETUPS {
        for (signal, channels, pcm) in signals(setup.input_rate) {
            let bandwidth = [BW_NB, BW_MB, BW_WB][k % 3];
            let frame_ms = [20, 60, 10, 40][k % 4];
            let extra_rate = [8000, 12000, 16000, 24000][(k / 3) % 4];
            k += 1;
            seen.insert((bandwidth, frame_ms));
            let packets = encoder(setup, channels, bandwidth).encode_all(&pcm, frame_ms);
            let name = format!(
                "{signal}; {}; {} {frame_ms} ms",
                setup.name,
                bandwidth_name(bandwidth)
            );
            assert!(packets.iter().all(|p| is_silk(p)), "{name}: not SILK-only");
            assert!(
                packets
                    .iter()
                    .filter(|p| p.len() > 2)
                    .all(|p| toc(p).silk_rate == silk_rate(bandwidth) && toc(p).channels == channels),
                "{name}: {}",
                stream_summary(&packets)
            );
            for decoder_channels in [1, 2] {
                for rate in [silk_rate(bandwidth), 48000, extra_rate] {
                    for (seed, fec) in [(k as u32, true), (k as u32 + 1000, false)] {
                        let fates = random_fates(seed, packets.len(), fec);
                        let (diff, stats) = run(&packets, rate, decoder_channels, |i| fates[i]);
                        failures.extend(diff.failure(&format!(
                            "{name}, {decoder_channels} ch decoder at {rate} Hz, seed {seed}"
                        )));
                        total.add(&stats);
                    }
                }
            }
        }
    }
    assert_eq!(seen.len(), 12, "not every bandwidth and packet duration occurred");
    println!("{k} streams: {total:?}");
    assert_no_failures(&failures);
    // The streams must have exercised every kind of frame.
    assert!(
        total.lbrr_mid > 500
            && total.lbrr_side > 100
            && total.lbrr_mid_only > 20
            && total.lbrr_missing > 500
            && total.mid_only > 500
            && total.with_side > 500
            && total.concealed > 500,
        "{total:?}"
    );
}

/// One encoder whose bitrate changes while it runs, the way an application
/// reacts to the network. The bitrate moves between 8 and 64 kbit/s, and
/// the encoder itself decides when to code stereo and when mono. Nothing is
/// forced except the SILK mode and the bandwidth.
///
/// The bandwidth stays fixed: when libopus changes the SILK bandwidth of a
/// running stream it adds a CELT frame to the packet (redundancy), and the
/// C decoder's output is then no longer the SILK decoder's alone.
#[test]
fn an_encoder_that_changes_its_bitrate_matches_libopus() {
    let n = 16000 * 5;
    let right: Vec<i16> = noise_bursts(16000, n).iter().map(|&v| v / 4).collect();
    let pcm = stereo_from(&low_voice(16000, n), &right);
    let mut failures = Vec::new();
    for (frame_ms, bandwidth) in [(10usize, BW_WB), (20, BW_NB), (40, BW_MB), (60, BW_WB)] {
        let mut enc = CEnc::with(16000, 2, sys::OPUS_APPLICATION_VOIP);
        enc.ctl(OPUS_SET_FORCE_MODE_REQUEST, MODE_SILK)
            .ctl(sys::OPUS_SET_BANDWIDTH_REQUEST, bandwidth)
            .fec(20);
        let mut packets = Vec::new();
        for (i, frame) in pcm.chunks_exact(16 * frame_ms * 2).enumerate() {
            let bitrate = [64000, 8000, 40000, 12000, 56000][(i * frame_ms / 480) % 5];
            enc.ctl(sys::OPUS_SET_BITRATE_REQUEST, bitrate);
            packets.push(enc.encode(frame));
        }
        assert!(packets.iter().all(|p| is_silk(p)));
        let summary = stream_summary(&packets);
        let count = |channels| packets.iter().filter(|p| toc(p).channels == channels).count();
        assert!(
            count(1) > 5 && count(2) > 5,
            "{frame_ms} ms: the encoder did not change the channel count: {summary}"
        );
        assert!(
            packets.iter().all(|p| toc(p).silk_rate == silk_rate(bandwidth)),
            "{frame_ms} ms: the bandwidth changed: {summary}"
        );
        for channels in [2, 1] {
            for rate in [8000, 16000, 24000, 48000] {
                let what = format!(
                    "bitrate changes, {} {frame_ms} ms, {channels} ch decoder at {rate} Hz",
                    bandwidth_name(bandwidth)
                );
                for (seed, fec) in [(31, true), (32, false), (33, true)] {
                    let fates = random_fates(seed + frame_ms as u32, packets.len(), fec);
                    let (diff, _) = run(&packets, rate, channels, |i| fates[i]);
                    failures.extend(diff.failure(&format!("{what}, seed {seed}")));
                }
                let (diff, _) = run(&packets, rate, channels, |_| Fate::Received);
                failures.extend(diff.failure(&format!("{what}, no loss")));
            }
        }
    }
    assert_no_failures(&failures);
}

/// Comfort noise right after a decoder reset.
///
/// The signal is steady background noise, which SILK codes as frames
/// without voice activity. From such frames the decoder builds its picture
/// of the background (level and spectrum) and adds matching noise to
/// concealed frames. A CELT-only packet in the stream makes libopus reset
/// the SILK decoder, and with it that picture. The packets lost shortly
/// after each reset are concealed with a picture that is one or two frames
/// old, so the output shows whether the reset cleared it.
#[test]
fn comfort_noise_after_a_reset_matches_libopus() {
    let mut rng = Lcg(5);
    let mut colored = 0i32;
    // Low-pass filtered noise at about -40 dB.
    let mono: Vec<i16> = (0..16000 * 4)
        .map(|_| {
            let white = (rng.next() & 0x3ff) as i32 - 512;
            colored += (white - colored) / 4;
            colored as i16
        })
        .collect();
    let stereo = stereo_from(&mono, &mono.iter().rev().copied().collect::<Vec<_>>());
    let mut celt = CEnc::celt(1, BW_WB, 32000);

    let mut failures = Vec::new();
    for (channels, bandwidth, frame_ms) in
        [(1usize, BW_WB, 20usize), (2, BW_WB, 20), (1, BW_NB, 10), (2, BW_MB, 40), (1, BW_MB, 60)]
    {
        let pcm = if channels == 2 { &stereo } else { &mono };
        let mut enc = CEnc::silk(channels, bandwidth, 16000 * channels as i32);
        enc.force_channels(channels as i32);
        let silk = enc.encode_all(pcm, frame_ms);
        // A CELT-only packet after every eighth SILK packet.
        let mut packets = Vec::new();
        let mut celt_at = Vec::new();
        for (i, p) in silk.into_iter().enumerate() {
            if i > 0 && i % 8 == 0 {
                celt_at.push(packets.len());
                packets.push(celt.encode(&mono[..320]));
            }
            packets.push(p);
        }
        // Lose the second and third SILK packet after each CELT-only
        // packet, or the third to the sixth.
        for (pattern, first, last) in [("2 lost soon after each reset", 2, 3), ("4 lost after each reset", 3, 6)] {
            let lost = |i: usize| celt_at.iter().any(|&c| i >= c + first && i <= c + last);
            for decoder_channels in [1, 2] {
                for rate in [silk_rate(bandwidth), 48000] {
                    let (diff, stats) = run(&packets, rate, decoder_channels, fate_without_fec(lost));
                    assert!(stats.concealed > 10, "{stats:?}");
                    failures.extend(diff.failure(&format!(
                        "noise, {} {frame_ms} ms {channels} ch, {decoder_channels} ch decoder at {rate} Hz, {pattern}",
                        bandwidth_name(bandwidth)
                    )));
                }
            }
        }
    }
    assert_no_failures(&failures);
}
