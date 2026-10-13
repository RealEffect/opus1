//! The older entry point `SilkDecoder::decode`, which `OpusDecoder` in
//! src/lib.rs uses today.
//!
//! It returns the decoded channels at the SILK rate and the caller
//! resamples them. The helper `RsLegacyDec` drives it exactly the way
//! `OpusDecoder::decode_silk_frames` does. The forward error correction
//! (FEC) rules are shared with `decode_with_control`, so stereo streams and
//! 40 and 60 ms packets must come out identical to C libopus here too.
//!
//! The same loss pattern also runs through `decode_with_control`. Both
//! entry points must give the same samples.

mod silkdec_common;
use silkdec_common::*;

/// 3 s of speech with FEC data, mono or stereo.
fn stream(bandwidth: i32, channels: usize, frame_ms: usize) -> Vec<Vec<u8>> {
    let mono = speech(0.5, 3.0);
    let pcm = if channels == 2 { stereo_from(&mono, &speech(7.0, 3.0)) } else { mono };
    let base = match bandwidth {
        BW_NB => 20000,
        BW_MB => 24000,
        _ => 30000,
    } + if frame_ms == 10 { 6000 } else { 0 };
    let mut enc = CEnc::silk(channels, bandwidth, base * channels as i32);
    enc.fec(20).force_channels(channels as i32);
    let packets = enc.encode_all(&pcm, frame_ms);
    let want = Toc { silk_rate: silk_rate(bandwidth), frame_ms, channels };
    assert!(packets.iter().all(|p| toc(p) == want), "{}", stream_summary(&packets));
    packets
}

#[test]
fn decode_at_the_silk_rate_matches_libopus_with_losses_and_fec() {
    let lost = |i: usize| i == 0 || i % 6 == 3 || (8..10).contains(&(i % 13));
    let mut failures = Vec::new();
    let mut recovered = [0usize; 2];
    for channels in [1usize, 2] {
        for bandwidth in [BW_NB, BW_MB, BW_WB] {
            for frame_ms in [10, 20, 40, 60] {
                let packets = stream(bandwidth, channels, frame_ms);
                for rate in [silk_rate(bandwidth), 48000] {
                    let name = format!(
                        "{} {frame_ms} ms {channels} ch at {rate} Hz",
                        bandwidth_name(bandwidth)
                    );
                    let frame_size = rate * frame_ms / 1000;
                    let mut c = CDec::new(rate, channels);
                    let mut legacy = RsLegacyDec::new(rate, channels);
                    let mut current = RsDec::new(rate, channels);
                    let (mut diff, mut diff_entry_points) = (Diff::default(), Diff::default());
                    let fate = fate_with_fec(lost);
                    for (i, p) in packets.iter().enumerate() {
                        let mut f = fate(i);
                        if f == Fate::Recovered && i + 1 == packets.len() {
                            f = Fate::Lost;
                        }
                        let (reference, old, new) = match f {
                            Fate::Received => (
                                c.decode(Some(p), frame_size, false),
                                legacy.decode(p),
                                current.decode(p),
                            ),
                            Fate::Lost => (
                                c.decode(None, frame_size, false),
                                legacy.lost(frame_ms),
                                current.lost(frame_size),
                            ),
                            Fate::Recovered => {
                                let next = &packets[i + 1];
                                (
                                    c.decode(Some(next), frame_size, true),
                                    legacy.fec(next),
                                    current.fec(next, frame_size),
                                )
                            }
                        };
                        diff.add(&reference, &old, || format!("packet {i} ({f:?})"));
                        diff_entry_points.add(&new, &old, || format!("packet {i} ({f:?})"));
                    }
                    assert!(legacy.lbrr_frames > 0, "{name}: no frame was recovered from FEC data");
                    recovered[channels - 1] += legacy.lbrr_frames;
                    failures.extend(diff.failure(&name));
                    failures.extend(
                        diff_entry_points.failure(&format!("{name}, decode against decode_with_control")),
                    );
                }
            }
        }
    }
    println!("frames recovered from FEC data: mono {}, stereo {}", recovered[0], recovered[1]);
    assert_no_failures(&failures);
}
