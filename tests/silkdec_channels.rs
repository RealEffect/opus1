//! Packets that do not code as many channels as the decoder outputs.
//!
//! A stereo decoder also gets mono packets: the C encoder codes stereo
//! input as mono at low bitrates, and a stream may switch between the two
//! at any packet. A mono decoder may get a stereo stream. libopus converts
//! inside `silk_Decode`, which keeps the channel count the caller wants
//! (`nChannelsAPI`) apart from the count the packet codes
//! (`nChannelsInternal`):
//!
//! - Mono packet, stereo output: the decoded signal goes to both channels.
//! - Stereo packet, mono output: the output is the mid channel.
//! - When a stream changes from mono to stereo, the second channel starts
//!   afresh and its resampler continues from the first channel's state.
//! - In the first mono frame after stereo, the right channel still goes
//!   through its own resampler.
//!
//! `SilkDecoder::decode_with_control` is the port. Every test compares it
//! with a C decoder of the same output channel count, sample by sample,
//! for received, concealed and FEC-recovered packets.

mod silkdec_common;
use silkdec_common::*;

#[allow(clippy::type_complexity)]
const LOSSES: [(&str, fn(usize) -> bool); 4] = [
    ("no loss", |_| false),
    ("first packet lost", |i| i == 0),
    ("one packet in five lost", |i| i % 5 == 2),
    ("three in a row lost", |i| (4..7).contains(&(i % 10))),
];

/// Bitrate per channel at which the C encoder keeps the bandwidth and, when
/// asked, the FEC data.
fn bitrate(bandwidth: i32, frame_ms: usize) -> i32 {
    let base = match bandwidth {
        BW_NB => 20000,
        BW_MB => 24000,
        _ => 30000,
    };
    base + if frame_ms == 10 { 6000 } else { 0 }
}

/// 3 s of stereo with a different signal on each channel.
fn stereo_pcm() -> Vec<i16> {
    stereo_from(&speech(0.5, 3.0), &speech(7.0, 3.0))
}

/// Decode `packets` with a decoder of `channels` channels at each of
/// `rates`, under every loss pattern, with and without FEC recovery.
fn check(name: &str, packets: &[Vec<u8>], channels: usize, rates: &[usize], failures: &mut Vec<String>) -> Stats {
    let mut total = Stats::default();
    for &rate in rates {
        for (pattern, lost) in LOSSES {
            let what = format!("{name}, {channels} ch decoder at {rate} Hz, {pattern}");
            let (diff, stats) = run(packets, rate, channels, fate_without_fec(lost));
            failures.extend(diff.failure(&what));
            total.add(&stats);
            let (diff, stats) = run(packets, rate, channels, fate_with_fec(lost));
            failures.extend(diff.failure(&format!("{what}, FEC")));
            total.add(&stats);
        }
    }
    total
}

/// Mono streams in a stereo decoder: all bandwidths and packet durations.
#[test]
fn mono_stream_in_a_stereo_decoder_matches_libopus() {
    let pcm = speech(0.5, 3.0);
    let mut failures = Vec::new();
    let mut total = Stats::default();
    for bandwidth in [BW_NB, BW_MB, BW_WB] {
        for frame_ms in [10, 20, 40, 60] {
            let mut enc = CEnc::silk(1, bandwidth, bitrate(bandwidth, frame_ms));
            enc.fec(20);
            let packets = enc.encode_all(&pcm, frame_ms);
            let want = Toc { silk_rate: silk_rate(bandwidth), frame_ms, channels: 1 };
            assert!(packets.iter().all(|p| toc(p) == want), "{}", stream_summary(&packets));
            let name = format!("mono {} {frame_ms} ms", bandwidth_name(bandwidth));
            let rates = [silk_rate(bandwidth), 48000];
            total.add(&check(&name, &packets, 2, &rates, &mut failures));
        }
    }
    assert_no_failures(&failures);
    assert!(total.lbrr_mid > 100 && total.concealed > 100, "{total:?}");
}

/// Stereo streams in a mono decoder: all bandwidths and packet durations.
/// The stream starts with the same signal on both channels, which the
/// encoder codes without a side channel, so the mono decoder also meets
/// mid-only frames.
#[test]
fn stereo_stream_in_a_mono_decoder_matches_libopus() {
    let mut pcm = stereo_pcm();
    for frame in pcm[..2 * 16000].as_chunks_mut::<2>().0 {
        frame[1] = frame[0];
    }
    let mut failures = Vec::new();
    let mut total = Stats::default();
    for bandwidth in [BW_NB, BW_MB, BW_WB] {
        for frame_ms in [10, 20, 40, 60] {
            let mut enc = CEnc::silk(2, bandwidth, 2 * bitrate(bandwidth, frame_ms));
            enc.fec(20).force_channels(2);
            let packets = enc.encode_all(&pcm, frame_ms);
            let want = Toc { silk_rate: silk_rate(bandwidth), frame_ms, channels: 2 };
            assert!(packets.iter().all(|p| toc(p) == want), "{}", stream_summary(&packets));
            let name = format!("stereo {} {frame_ms} ms", bandwidth_name(bandwidth));
            let rates = [silk_rate(bandwidth), 48000];
            total.add(&check(&name, &packets, 1, &rates, &mut failures));
        }
    }
    assert_no_failures(&failures);
    assert!(
        total.lbrr_mid > 100 && total.lbrr_side > 100 && total.mid_only > 100 && total.with_side > 100,
        "{total:?}"
    );
}

/// One C encoder that is told to change between stereo and mono coding
/// every `period_ms`. It makes the change the way a real stream does: the
/// last stereo frames are narrowed towards mono first.
fn encoder_switching_stream(bandwidth: i32, frame_ms: usize, period_ms: usize) -> Vec<Vec<u8>> {
    let pcm = stereo_pcm();
    let mut enc = CEnc::silk(2, bandwidth, 2 * bitrate(bandwidth, frame_ms));
    enc.fec(20);
    let mut packets = Vec::new();
    for (i, frame) in pcm.chunks_exact(16 * frame_ms * 2).enumerate() {
        let stereo = (i * frame_ms / period_ms).is_multiple_of(2);
        enc.force_channels(if stereo { 2 } else { 1 });
        packets.push(enc.encode(frame));
    }
    let count = |channels| packets.iter().filter(|p| toc(p).channels == channels).count();
    assert!(
        count(1) * frame_ms > 500 && count(2) * frame_ms > 500,
        "the C encoder did not switch: {}",
        stream_summary(&packets)
    );
    packets
}

/// Streams that switch between stereo and mono packets, in a stereo and in
/// a mono decoder.
#[test]
fn streams_that_switch_between_stereo_and_mono_match_libopus() {
    let mut failures = Vec::new();
    let mut total = Stats::default();
    for (bandwidth, frame_ms, period_ms) in [
        (BW_NB, 20, 400),
        (BW_MB, 20, 300),
        (BW_WB, 20, 400),
        (BW_WB, 10, 250),
        (BW_WB, 40, 480),
        (BW_NB, 60, 540),
        (BW_MB, 10, 200),
        (BW_WB, 60, 480),
    ] {
        let packets = encoder_switching_stream(bandwidth, frame_ms, period_ms);
        let name = format!(
            "{} {frame_ms} ms, channel count changes every {period_ms} ms",
            bandwidth_name(bandwidth)
        );
        let rates = [silk_rate(bandwidth), 48000];
        for channels in [2, 1] {
            total.add(&check(&name, &packets, channels, &rates, &mut failures));
        }
    }
    assert_no_failures(&failures);
    assert!(total.lbrr_mid > 100 && total.concealed > 100, "{total:?}");
}

/// Packets of independent encoders put one after the other. Nothing
/// prepares these switches: the channel count, the SILK rate and the packet
/// duration change from one packet to the next, in every combination. The
/// loss patterns of `check` are joined by one that loses the packets on
/// both sides of every switch.
#[test]
fn abrupt_switches_match_libopus() {
    // (channels, bandwidth, packet duration) of each segment of 12 packets.
    let segments = [
        (2, BW_WB, 20),
        (1, BW_WB, 20),
        (2, BW_WB, 20),
        (1, BW_NB, 20),
        (2, BW_NB, 40),
        (1, BW_MB, 10),
        (2, BW_MB, 60),
        (1, BW_MB, 60),
        (1, BW_WB, 10),
        (2, BW_WB, 10),
        (2, BW_NB, 20),
        (1, BW_NB, 40),
        (2, BW_WB, 40),
        (1, BW_WB, 40),
    ];
    const SEGMENT: usize = 12;
    let (mono, stereo) = (speech(0.5, 1.0), stereo_from(&speech(0.5, 1.0), &speech(7.0, 1.0)));
    let mut packets = Vec::new();
    for (channels, bandwidth, frame_ms) in segments {
        let mut enc = CEnc::silk(channels, bandwidth, channels as i32 * bitrate(bandwidth, frame_ms));
        enc.fec(20).force_channels(channels as i32);
        let all = enc.encode_all(if channels == 2 { &stereo } else { &mono }, frame_ms);
        // Skip the encoder's first packets, which carry no FEC data yet.
        packets.extend(all.into_iter().skip(2).take(SEGMENT));
    }
    assert_eq!(packets.len(), segments.len() * SEGMENT);

    let mut failures = Vec::new();
    let mut total = Stats::default();
    for channels in [2, 1] {
        for rate in [8000, 12000, 16000, 24000, 48000] {
            total.add(&check("abrupt switches", &packets, channels, &[rate], &mut failures));
            // Lose the last packet of every segment and the first of the
            // next one. Without FEC both are concealed. With FEC the first
            // one is concealed with the old channel count and the second
            // one is recovered from a packet with the new one.
            let around_switch = |i: usize| i % SEGMENT == SEGMENT - 1 || (i.is_multiple_of(SEGMENT) && i > 0);
            let last_before = |i: usize| i % SEGMENT == SEGMENT - 1;
            for (pattern, lost, fec) in [
                ("both packets at a switch lost", &around_switch as &dyn Fn(usize) -> bool, false),
                ("both packets at a switch lost, FEC", &around_switch, true),
                ("last packet before a switch lost, FEC", &last_before, true),
            ] {
                let what = format!("abrupt switches, {channels} ch decoder at {rate} Hz, {pattern}");
                let (diff, stats) = if fec {
                    run(&packets, rate, channels, fate_with_fec(lost))
                } else {
                    run(&packets, rate, channels, fate_without_fec(lost))
                };
                failures.extend(diff.failure(&what));
                total.add(&stats);
            }
        }
    }
    assert_no_failures(&failures);
    assert!(total.lbrr_mid > 100 && total.lbrr_side > 100 && total.concealed > 100, "{total:?}");
}
