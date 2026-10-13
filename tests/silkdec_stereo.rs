//! Stereo SILK streams with and without a coded side channel.
//!
//! SILK codes stereo as a mid and a side channel. The encoder leaves the
//! side channel out of a frame and sets the frame's mid-only flag in two
//! situations: left and right are the same, or the bitrate is too low for
//! two channels. The decoder has rules for the switches between the two
//! kinds of frame, for losing a frame of either kind, and for the mid-only
//! flag inside the forward error correction (FEC) data.
//!
//! The streams here contain both kinds of frame and both switches. Every
//! test requires 16-bit output identical to C libopus.

mod silkdec_common;
use silkdec_common::*;

/// 4 s of stereo: one talker on both channels for 1.2 s, then two
/// different talkers. The C encoder codes the first part mid-only, at any
/// bitrate.
fn pcm() -> Vec<i16> {
    let (left, right) = (speech(0.5, 4.0), speech(7.0, 4.0));
    let same = 19200;
    let mut out = Vec::with_capacity(left.len() * 2);
    for i in 0..left.len() {
        out.push(left[i]);
        out.push(if i < same { left[i] } else { right[i] });
    }
    out
}

/// Bitrate at which the C encoder codes both channels of different
/// talkers (and keeps FEC on when it is asked for).
fn high_bitrate(bandwidth: i32, frame_ms: usize) -> i32 {
    let base = match bandwidth {
        BW_NB => 36000,
        BW_MB => 44000,
        _ => 52000,
    };
    base + if frame_ms == 10 { 10000 } else { 0 }
}

/// Bitrate at which the C encoder gives up the side channel.
fn low_bitrate(bandwidth: i32) -> i32 {
    match bandwidth {
        BW_NB => 8000,
        BW_MB => 10000,
        _ => 12000,
    }
}

/// Encode `pcm()` as stereo SILK. With `dip`, the bitrate is low from
/// 2.4 s to 3.2 s, which gives a second run of mid-only frames, this time
/// after frames with a side channel. `fec_loss` switches FEC on for that
/// expected loss percentage.
fn stream(bandwidth: i32, frame_ms: usize, dip: bool, fec_loss: Option<i32>) -> Vec<Vec<u8>> {
    let high = high_bitrate(bandwidth, frame_ms);
    let mut enc = CEnc::silk(2, bandwidth, high);
    enc.force_channels(2);
    if let Some(loss) = fec_loss {
        enc.fec(loss);
    }
    let pcm = pcm();
    let mut packets = Vec::new();
    for (i, frame) in pcm.chunks_exact(16 * frame_ms * 2).enumerate() {
        let ms = i * frame_ms;
        let low = dip && (2400..3200).contains(&ms);
        enc.ctl(audiopus_sys::OPUS_SET_BITRATE_REQUEST, if low { low_bitrate(bandwidth) } else { high });
        packets.push(enc.encode(frame));
    }
    let want = Toc { silk_rate: silk_rate(bandwidth), frame_ms, channels: 2 };
    assert!(
        packets.iter().all(|p| toc(p) == want),
        "the C encoder did not produce {want:?} throughout: {}",
        stream_summary(&packets)
    );
    packets
}

const LOSSES: [(&str, fn(usize) -> bool); 4] = [
    ("no loss", |_| false),
    ("first packet lost", |i| i == 0),
    ("one packet in five lost", |i| i % 5 == 2),
    ("four in a row lost", |i| (3..7).contains(&(i % 11))),
];

/// Received and concealed frames, all bandwidths and packet durations.
#[test]
fn mid_only_and_side_frames_match_libopus() {
    let mut failures = Vec::new();
    for bandwidth in [BW_NB, BW_MB, BW_WB] {
        for frame_ms in [10, 20, 40, 60] {
            let packets = stream(bandwidth, frame_ms, true, None);
            for rate in [silk_rate(bandwidth), 24000, 48000] {
                for (pattern, lost) in LOSSES {
                    let (diff, stats) = run(&packets, rate, 2, fate_without_fec(lost));
                    let name = format!(
                        "{} {frame_ms} ms at {rate} Hz, {pattern}",
                        bandwidth_name(bandwidth)
                    );
                    // The stream must contain what this test is about.
                    assert!(
                        stats.mid_only > 5 && stats.with_side > 5,
                        "{name}: {} mid-only frames and {} frames with a side channel",
                        stats.mid_only,
                        stats.with_side
                    );
                    failures.extend(diff.failure(&name));
                }
            }
        }
    }
    assert_no_failures(&failures);
}

/// The same streams with FEC data, and every lost packet recovered from
/// the next packet where that one arrived. The FEC copy of a frame has its
/// own stereo predictor and its own mid-only flag.
#[test]
fn fec_of_mid_only_and_side_frames_matches_libopus() {
    let mut failures = Vec::new();
    let mut total = Stats::default();
    for bandwidth in [BW_NB, BW_MB, BW_WB] {
        for frame_ms in [10, 20, 40, 60] {
            let packets = stream(bandwidth, frame_ms, false, Some(20));
            for rate in [silk_rate(bandwidth), 48000] {
                for (pattern, lost) in &LOSSES[1..] {
                    let (diff, stats) = run(&packets, rate, 2, fate_with_fec(lost));
                    let name = format!(
                        "{} {frame_ms} ms at {rate} Hz with FEC, {pattern}",
                        bandwidth_name(bandwidth)
                    );
                    assert!(stats.lbrr_mid > 0, "{name}: no frame was recovered from FEC data");
                    failures.extend(diff.failure(&name));
                    total.add(&stats);
                }
            }
        }
    }
    println!("{total:?}");
    assert_no_failures(&failures);
    // Copies with a side channel, and copies that say "mid only", must both
    // have been decoded.
    assert!(total.lbrr_side > 100 && total.lbrr_mid_only > 100, "{total:?}");
}

/// A stream without FEC data, decoded with the FEC flag: the packet has no
/// copy of the lost frame, so the frame is concealed. This path differs
/// from a plain loss in what state it updates afterwards.
#[test]
fn fec_flag_without_fec_data_matches_libopus() {
    for channels in [1usize, 2] {
        for (bandwidth, frame_ms) in [(BW_NB, 20), (BW_MB, 60), (BW_WB, 10), (BW_WB, 40)] {
            let packets: Vec<Vec<u8>> = if channels == 2 {
                stream(bandwidth, frame_ms, true, None)
            } else {
                CEnc::silk(1, bandwidth, 20000).encode_all(&speech(0.5, 3.0), frame_ms)
            };
            for rate in [silk_rate(bandwidth), 48000] {
                let fate = |i| match i % 6 {
                    0 | 4 => Fate::Recovered,
                    3 => Fate::Lost,
                    _ => Fate::Received,
                };
                let (diff, stats) = run(&packets, rate, channels, fate);
                let name = format!(
                    "{} {frame_ms} ms {channels} ch at {rate} Hz, FEC flag without FEC data",
                    bandwidth_name(bandwidth)
                );
                assert_eq!(stats.lbrr_mid + stats.lbrr_side, 0, "{name}: the stream has FEC data");
                assert!(stats.lbrr_missing > 10, "{name}: {stats:?}");
                diff.assert_identical(&name);
            }
        }
    }
}

/// Speech, 3 s of faint noise, speech again, encoded with discontinuous
/// transmission (DTX): during the noise the encoder sends packets without
/// data, apart from one frame every 400 ms. The decoder conceals the
/// missing frames with comfort noise, for seconds. Mono and stereo streams
/// in mono and stereo decoders.
#[test]
fn dtx_pauses_match_libopus() {
    let mut seed = 99u32;
    let mut noise = |n: usize| -> Vec<i16> {
        (0..n)
            .map(|_| {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                ((seed >> 8) as i32 % 65 - 32) as i16
            })
            .collect()
    };
    let mut left = speech(0.5, 1.0);
    left.extend(noise(48000));
    left.extend(speech(3.0, 1.0));
    let mut right = speech(7.0, 1.0);
    right.extend(noise(48000));
    right.extend(speech(9.0, 1.0));
    let stereo = stereo_from(&left, &right);

    let mut failures = Vec::new();
    for stream_channels in [1usize, 2] {
        for (bandwidth, frame_ms) in [(BW_NB, 20), (BW_WB, 20), (BW_MB, 60), (BW_WB, 10)] {
            let mut enc = CEnc::silk(stream_channels, bandwidth, 24000 * stream_channels as i32);
            enc.force_channels(stream_channels as i32).ctl(audiopus_sys::OPUS_SET_DTX_REQUEST, 1);
            let packets = enc.encode_all(if stream_channels == 2 { &stereo } else { &left }, frame_ms);
            let empty = packets.iter().filter(|p| p.len() <= 2).count();
            println!("{} {frame_ms} ms {stream_channels} ch: {empty} of {} packets empty", bandwidth_name(bandwidth), packets.len());
            assert!(empty * frame_ms > 600, "expected DTX packets, got {empty}");
            for channels in [1, 2] {
                for rate in [silk_rate(bandwidth), 48000] {
                    let name = format!(
                        "DTX, {} {frame_ms} ms {stream_channels} ch stream, {channels} ch decoder at {rate} Hz",
                        bandwidth_name(bandwidth)
                    );
                    let (diff, stats) = run(&packets, rate, channels, |_| Fate::Received);
                    assert!(stats.concealed * 10 > 600, "{name}: {stats:?}");
                    failures.extend(diff.failure(&name));
                    // One packet in six lost on top of the pauses.
                    let (diff, _) = run(&packets, rate, channels, fate_without_fec(|i| i % 6 == 1));
                    failures.extend(diff.failure(&format!("{name}, one packet in six lost")));
                }
            }
        }
    }
    assert_no_failures(&failures);
}
