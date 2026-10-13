//! In-band forward error correction (FEC) in the SILK decoder.
//!
//! A SILK packet can carry a low-bitrate copy of the packet before it
//! (libopus calls the copy LBRR). When a packet is lost, the receiver hands
//! the next packet to `opus_decode` with `decode_fec = 1` and gets the copy
//! back. `SilkDecoder` does the same with `FLAG_DECODE_LBRR`.
//!
//! Every test decodes one stream from the C encoder twice, with C libopus
//! and with opus-rs, under the same loss pattern, and requires identical
//! 16-bit output for every packet: received, concealed and recovered.

mod silkdec_common;
use silkdec_common::*;

/// The loss patterns every stream is decoded with.
const PATTERNS: [(&str, fn(usize) -> bool); 4] = [
    ("first packet lost", |i| i == 0),
    ("one packet in seven lost", |i| i % 7 == 3),
    ("three in a row lost", |i| (5..8).contains(&(i % 12))),
    ("first two and every fifth lost", |i| i < 2 || i % 5 == 4),
];

/// Bitrate that keeps the C encoder's FEC on at 20 % expected loss.
fn fec_bitrate(bandwidth: i32, channels: usize, frame_ms: usize) -> i32 {
    let base = match bandwidth {
        BW_NB => 20000,
        BW_MB => 24000,
        _ => 30000,
    };
    // 10 ms packets spend more of the rate on overhead.
    (base + if frame_ms == 10 { 6000 } else { 0 }) * if channels == 2 { 2 } else { 1 }
}

/// Encode 3 s of speech with FEC and check the stream is what the test
/// needs: SILK-only at the wanted bandwidth, duration and channel count.
fn fec_stream(bandwidth: i32, channels: usize, frame_ms: usize) -> Vec<Vec<u8>> {
    let mono = speech(0.5, 3.0);
    let pcm = if channels == 2 { stereo_delayed(&mono) } else { mono };
    let mut enc = CEnc::silk(channels, bandwidth, fec_bitrate(bandwidth, channels, frame_ms));
    enc.fec(20).force_channels(channels as i32);
    let packets = enc.encode_all(&pcm, frame_ms);
    let want = Toc { silk_rate: silk_rate(bandwidth), frame_ms, channels };
    assert!(
        packets.iter().all(|p| toc(p) == want && frames(p).len() == 1),
        "the C encoder did not produce {want:?} throughout: {}",
        stream_summary(&packets)
    );
    packets
}

/// All bandwidths, packet durations and loss patterns for one channel
/// count, at the SILK rate and at 48 kHz output.
fn fec_matrix(channels: usize) {
    let mut failures = Vec::new();
    let mut total = Stats::default();
    for bandwidth in [BW_NB, BW_MB, BW_WB] {
        for frame_ms in [10, 20, 40, 60] {
            let packets = fec_stream(bandwidth, channels, frame_ms);
            for rate in [silk_rate(bandwidth), 48000] {
                for (pattern, lost) in PATTERNS {
                    let (diff, stats) = run(&packets, rate, channels, fate_with_fec(lost));
                    let name = format!(
                        "{} {frame_ms} ms {channels} ch at {rate} Hz, {pattern}",
                        bandwidth_name(bandwidth)
                    );
                    assert!(stats.lbrr_mid > 0, "{name}: no frame was recovered from FEC data");
                    // The pitch lag comparison needs voiced frames.
                    assert!(diff.voiced > 10, "{name}: {} voiced packets", diff.voiced);
                    failures.extend(diff.failure(&name));
                    total.add(&stats);
                }
            }
        }
    }
    println!("{channels} ch: {total:?}");
    assert_no_failures(&failures);
    // The matrix must have met frames with a copy and frames without one.
    assert!(total.lbrr_missing > 0, "no frame without an LBRR copy was asked for");
    if channels == 2 {
        assert!(total.lbrr_side > 0, "no side channel was recovered from FEC data");
    }
}

#[test]
fn mono_fec_matches_libopus() {
    fec_matrix(1);
}

#[test]
fn stereo_fec_matches_libopus() {
    fec_matrix(2);
}
