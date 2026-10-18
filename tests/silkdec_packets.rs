//! Packet layouts and loss handling that the other `silkdec_*` tests do
//! not use.
//!
//! - Packets that hold several Opus frames (made with the libopus
//!   repacketizer), up to 120 ms, and packets with padding. The forward
//!   error correction (FEC) data of such a packet is in its first frame and
//!   covers only the last frame of the packet before it.
//! - Lost audio that the caller asks for in pieces that are not the packet
//!   duration: 2.5 ms to 120 ms per call. For a piece shorter than 10 ms
//!   the SILK decoder still conceals 10 ms.
//! - FEC recovery where the caller asks for more or less than the next
//!   packet covers.
//! - Loss bursts of up to a second.
//!
//! Every test requires 16-bit output identical to C libopus.

mod silkdec_common;
use silkdec_common::*;

/// Bitrate per channel at which the C encoder keeps FEC data in the stream.
fn bitrate(bandwidth: i32, frame_ms: usize) -> i32 {
    let base = match bandwidth {
        BW_NB => 20000,
        BW_MB => 24000,
        _ => 30000,
    };
    base + if frame_ms == 10 { 6000 } else { 0 }
}

/// `seconds` of speech coded as SILK with FEC data, mono or stereo.
fn stream(channels: usize, bandwidth: i32, frame_ms: usize, seconds: f64) -> Vec<Vec<u8>> {
    let mono = speech(1.0, seconds);
    let pcm = if channels == 2 { stereo_from(&mono, &speech(8.0, seconds)) } else { mono };
    let mut enc = CEnc::silk(channels, bandwidth, channels as i32 * bitrate(bandwidth, frame_ms));
    enc.fec(20).force_channels(channels as i32);
    let packets = enc.encode_all(&pcm, frame_ms);
    let want = Toc { silk_rate: silk_rate(bandwidth), frame_ms, channels };
    assert!(packets.iter().all(|p| toc(p) == want), "{}", stream_summary(&packets));
    packets
}

/// Packets with 2 to 4 frames, with and without padding.
#[test]
fn packets_with_several_frames_match_libopus() {
    let mut failures = Vec::new();
    let mut total = Stats::default();
    // (channels, bandwidth, frame duration, frames per packet)
    for (channels, bandwidth, frame_ms, group) in [
        (1usize, BW_WB, 20usize, 2usize),
        (2, BW_WB, 20, 3),
        (1, BW_NB, 10, 4),
        (2, BW_MB, 40, 3),
        (1, BW_WB, 60, 2),
        (2, BW_NB, 10, 2),
        (1, BW_MB, 20, 4),
        (2, BW_WB, 60, 2),
    ] {
        let single = stream(channels, bandwidth, frame_ms, 3.0);
        let mut packets: Vec<Vec<u8>> = single.chunks_exact(group).map(repacketize).collect();
        // Every third packet also gets padding.
        for (i, p) in packets.iter_mut().enumerate() {
            if i % 3 == 1 {
                *p = pad(p, 1 + 7 * i % 300);
            }
        }
        assert!(
            packets.iter().all(|p| frames(p).len() == group),
            "the repacketizer did not make {group} frames per packet"
        );
        let name = format!(
            "{group} x {frame_ms} ms per packet, {} {channels} ch",
            bandwidth_name(bandwidth)
        );
        for decoder_channels in [1, 2] {
            for rate in [silk_rate(bandwidth), 48000] {
                for (pattern, fates) in [
                    ("no loss", vec![Fate::Received; packets.len()]),
                    ("random losses", random_fates(11, packets.len(), false)),
                    ("random losses, FEC", random_fates(12, packets.len(), true)),
                    ("first packet lost, FEC", random_fates(14, packets.len(), true)),
                ] {
                    let (diff, stats) = run(&packets, rate, decoder_channels, |i| fates[i]);
                    failures.extend(diff.failure(&format!(
                        "{name}, {decoder_channels} ch decoder at {rate} Hz, {pattern}"
                    )));
                    total.add(&stats);
                }
            }
        }
    }
    println!("{total:?}");
    assert_no_failures(&failures);
    assert!(total.lbrr_mid > 100 && total.lbrr_side > 20 && total.concealed > 500, "{total:?}");
}

/// A single-frame packet with padding and nothing else changed.
#[test]
fn padded_packets_match_libopus() {
    let mut failures = Vec::new();
    for (channels, bandwidth, frame_ms) in [(1usize, BW_WB, 20usize), (2, BW_NB, 60), (2, BW_MB, 10)] {
        let packets: Vec<Vec<u8>> = stream(channels, bandwidth, frame_ms, 2.0)
            .iter()
            .enumerate()
            .map(|(i, p)| pad(p, 1 + (i * 37) % 600))
            .collect();
        let fates = random_fates(21, packets.len(), true);
        for rate in [silk_rate(bandwidth), 24000] {
            let (diff, stats) = run(&packets, rate, channels, |i| fates[i]);
            assert!(stats.lbrr_mid > 0, "{stats:?}");
            failures.extend(diff.failure(&format!(
                "padded {} {frame_ms} ms {channels} ch at {rate} Hz",
                bandwidth_name(bandwidth)
            )));
        }
    }
    assert_no_failures(&failures);
}

/// Lost audio asked for in pieces of many sizes.
///
/// After every sixth packet the caller asks both decoders for a series of
/// pieces, given in units of 2.5 ms. The series do not add up to a packet
/// duration: the next packet simply follows, as it would after a jitter
/// buffer changed its mind.
#[test]
fn concealment_in_pieces_matches_libopus() {
    // Pieces in units of 2.5 ms.
    const SERIES: [&[usize]; 10] = [
        &[4, 4],
        &[8],
        &[2, 2, 2, 2],
        &[1, 1, 1, 5],
        &[24],
        &[12],
        &[3, 5, 16],
        &[48],
        &[1],
        &[7, 9, 4, 4, 8, 16],
    ];
    let mut failures = Vec::new();
    for (channels, bandwidth, frame_ms) in [
        (1usize, BW_WB, 20usize),
        (2, BW_WB, 20),
        (1, BW_NB, 10),
        (2, BW_MB, 10),
        (1, BW_MB, 40),
        (2, BW_NB, 60),
        (1, BW_WB, 60),
        (2, BW_WB, 40),
    ] {
        let packets = stream(channels, bandwidth, frame_ms, 4.0);
        for decoder_channels in [1, 2] {
            for rate in [silk_rate(bandwidth), 48000] {
                let unit = rate / 400;
                let mut c = CDec::new(rate, decoder_channels);
                let mut rs = RsDec::new(rate, decoder_channels);
                let mut diff = Diff::default();
                let mut series = 0;
                for (i, p) in packets.iter().enumerate() {
                    // The first series comes before any packet: silence.
                    if i % 6 == 0 {
                        for (k, &piece) in SERIES[series % SERIES.len()].iter().enumerate() {
                            let reference = c.decode(None, piece * unit, false);
                            let ours = rs.lost(piece * unit);
                            diff.add(&reference, &ours, || {
                                format!("piece {k} of series {series} before packet {i}")
                            });
                            diff.add_pitch(c.pitch(), rs.dec.prev_pitch_lag(), || {
                                format!("piece {k} of series {series} before packet {i}")
                            });
                        }
                        series += 1;
                    }
                    let frame_size = rate * frame_ms / 1000;
                    diff.add(&c.decode(Some(p), frame_size, false), &rs.decode(p), || {
                        format!("packet {i}")
                    });
                    diff.add_pitch(c.pitch(), rs.dec.prev_pitch_lag(), || format!("packet {i}"));
                }
                assert!(series >= SERIES.len(), "the stream is too short for every series");
                failures.extend(diff.failure(&format!(
                    "pieces, {} {frame_ms} ms {channels} ch stream, {decoder_channels} ch decoder at {rate} Hz",
                    bandwidth_name(bandwidth)
                )));
            }
        }
    }
    assert_no_failures(&failures);
}

/// FEC recovery where the duration the caller asks for is not the duration
/// of the next packet.
///
/// - More than the next packet: the first part is concealed, the last part
///   comes from the FEC data.
/// - Less than the next packet: libopus conceals, and does not use the FEC
///   data at all.
/// - 2.5 ms more than the next packet: the SILK decoder conceals 10 ms for
///   the 2.5 ms, then decodes the FEC data.
#[test]
fn fec_for_another_duration_than_the_next_packet_matches_libopus() {
    // Duration asked for, in units of 2.5 ms, as a function of the packet
    // duration in the same units.
    #[allow(clippy::type_complexity)]
    let asks: [(&str, fn(usize) -> usize); 5] = [
        ("twice the packet", |d| 2 * d),
        ("three times the packet", |d| 3 * d),
        ("half the packet", |d| d / 2),
        ("the packet and 2.5 ms", |d| d + 1),
        ("the packet and 10 ms", |d| d + 4),
    ];
    let mut failures = Vec::new();
    let mut total = Stats::default();
    for (channels, bandwidth, frame_ms) in [
        (1usize, BW_WB, 20usize),
        (2, BW_WB, 20),
        (2, BW_NB, 10),
        (1, BW_MB, 40),
        (2, BW_MB, 60),
        (1, BW_NB, 60),
    ] {
        let packets = stream(channels, bandwidth, frame_ms, 4.0);
        for rate in [silk_rate(bandwidth), 48000] {
            let unit = rate / 400;
            let mut c = CDec::new(rate, channels);
            let mut rs = RsDec::new(rate, channels);
            let mut diff = Diff::default();
            for (i, p) in packets.iter().enumerate() {
                // Before every fourth packet: a FEC call on that packet.
                // The audio it returns stands for packets that never
                // existed, which is fine for both decoders.
                if i % 4 == 2 {
                    let (what, ask) = asks[(i / 4) % asks.len()];
                    let frame_size = ask(frame_ms * 2 / 5) * unit;
                    let reference = c.decode(Some(p), frame_size, true);
                    let ours = rs.fec(p, frame_size);
                    diff.add(&reference, &ours, || format!("FEC from packet {i}, {what}"));
                    diff.add_pitch(c.pitch(), rs.dec.prev_pitch_lag(), || {
                        format!("FEC from packet {i}, {what}")
                    });
                }
                let frame_size = rate * frame_ms / 1000;
                diff.add(&c.decode(Some(p), frame_size, false), &rs.decode(p), || {
                    format!("packet {i}")
                });
                diff.add_pitch(c.pitch(), rs.dec.prev_pitch_lag(), || format!("packet {i}"));
            }
            total.add(&rs.stats);
            failures.extend(diff.failure(&format!(
                "FEC for another duration, {} {frame_ms} ms {channels} ch at {rate} Hz",
                bandwidth_name(bandwidth)
            )));
        }
    }
    println!("{total:?}");
    assert_no_failures(&failures);
    assert!(total.lbrr_mid > 50 && total.concealed > 50, "{total:?}");
}

/// Bursts of 0.3 to 1 second. The concealment fades out over the first
/// lost frames and then stays silent apart from comfort noise. The frames
/// after a burst start from that state.
#[test]
fn long_loss_bursts_match_libopus() {
    let mut failures = Vec::new();
    for (channels, bandwidth, frame_ms) in [
        (1usize, BW_WB, 20usize),
        (2, BW_WB, 20),
        (1, BW_NB, 10),
        (2, BW_MB, 40),
        (1, BW_MB, 60),
        (2, BW_NB, 60),
    ] {
        let packets = stream(channels, bandwidth, frame_ms, 5.0);
        // Bursts of 300, 600 and 1000 ms, each after 600 ms of received
        // packets.
        let lost = |i: usize| {
            let ms = i * frame_ms;
            (600..900).contains(&ms) || (1500..2100).contains(&ms) || (2700..3700).contains(&ms)
        };
        for decoder_channels in [1, 2] {
            for rate in [silk_rate(bandwidth), 48000] {
                let name = format!(
                    "bursts, {} {frame_ms} ms {channels} ch stream, {decoder_channels} ch decoder at {rate} Hz",
                    bandwidth_name(bandwidth)
                );
                let (diff, stats) = run(&packets, rate, decoder_channels, fate_without_fec(lost));
                // 1.9 s are lost, concealed 10 or 20 ms at a time.
                assert!(stats.concealed >= 90, "{name}: {stats:?}");
                failures.extend(diff.failure(&name));
                let (diff, stats) = run(&packets, rate, decoder_channels, fate_with_fec(lost));
                // The last packet of each burst is asked for with the FEC
                // flag. Not every packet carries a copy.
                assert!(stats.lbrr_mid >= 1 && stats.lbrr_mid + stats.lbrr_missing >= 3, "{name}: {stats:?}");
                failures.extend(diff.failure(&format!("{name}, FEC")));
            }
        }
    }
    assert_no_failures(&failures);
}
