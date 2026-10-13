//! `SilkDecoder::reset` and `SilkDecoder::init` next to
//! `decode_with_control`.
//!
//! libopus resets its SILK decoder before the first SILK frame that follows
//! a CELT-only frame (`silk_ResetDecoder`). The reset clears both channels
//! completely, the sampling rate and the resamplers included, but keeps the
//! channel counts of the last call. What the decoder does on the next
//! channel count change depends on both facts.
//!
//! C libopus has no call for this reset alone, so the streams here contain
//! CELT-only packets. The C decoder decodes them. The opus-rs SILK decoder
//! never sees them and is reset instead. In the first SILK frame after a
//! CELT-only packet libopus cross-fades 5 ms with CELT concealment; those
//! 5 ms are left out, everything else must be identical.

mod silkdec_common;
use silkdec_common::*;

/// Segments of 10 SILK packets each, with one CELT-only packet between two
/// segments wherever `celt_between` says so. Channel count, SILK rate and
/// packet duration change from segment to segment.
fn stream(celt_between: impl Fn(usize) -> bool) -> Vec<Vec<u8>> {
    let segments = [
        (2, BW_WB, 20),
        (1, BW_WB, 20),
        (2, BW_WB, 20),
        (2, BW_NB, 20),
        (1, BW_NB, 40),
        (1, BW_MB, 10),
        (2, BW_MB, 60),
        (2, BW_WB, 10),
        (1, BW_WB, 60),
        (1, BW_WB, 20),
        (2, BW_WB, 40),
        (1, BW_NB, 20),
    ];
    let (mono, stereo) = (speech(0.5, 1.0), stereo_from(&speech(0.5, 1.0), &speech(7.0, 1.0)));
    let mut celt = [CEnc::celt(1, BW_WB, 32000), CEnc::celt(2, BW_WB, 48000)];
    let mut packets = Vec::new();
    for (k, (channels, bandwidth, frame_ms)) in segments.into_iter().enumerate() {
        if k > 0 && celt_between(k) {
            // Mono and stereo CELT packets in turn.
            let enc = &mut celt[k % 2];
            let pcm = if enc.channels == 2 { &stereo } else { &mono };
            let packet = enc.encode(&pcm[..320 * enc.channels]);
            assert!(!is_silk(&packet));
            packets.push(packet);
        }
        let base = match bandwidth {
            BW_NB => 20000,
            BW_MB => 24000,
            _ => 30000,
        } + if frame_ms == 10 { 6000 } else { 0 };
        let mut enc = CEnc::silk(channels, bandwidth, base * channels as i32);
        enc.fec(20).force_channels(channels as i32);
        let all = enc.encode_all(if channels == 2 { &stereo } else { &mono }, frame_ms);
        packets.extend(all.into_iter().skip(2).take(10));
    }
    packets
}

const LOSSES: [(&str, fn(usize) -> bool); 3] = [
    ("no loss", |_| false),
    ("one packet in four lost", |i| i % 4 == 2),
    ("two in a row lost", |i| (5..7).contains(&(i % 9))),
];

/// A reset between any two segments, and a reset between some of them only.
#[test]
fn reset_after_a_celt_only_packet_matches_libopus() {
    let mut failures = Vec::new();
    for (layout, packets) in [
        ("CELT packet between all segments", stream(|_| true)),
        ("CELT packet between every third pair of segments", stream(|k| k % 3 == 1)),
    ] {
        assert!(packets.iter().any(|p| !is_silk(p)));
        for channels in [2, 1] {
            for rate in [8000, 16000, 48000] {
                for (pattern, lost) in LOSSES {
                    let name = format!("{layout}, {channels} ch decoder at {rate} Hz, {pattern}");
                    let (diff, _) = run(&packets, rate, channels, fate_without_fec(lost));
                    failures.extend(diff.failure(&name));
                    let (diff, stats) = run(&packets, rate, channels, fate_with_fec(lost));
                    failures.extend(diff.failure(&format!("{name}, FEC")));
                    assert!(pattern == "no loss" || stats.lbrr_mid > 0, "{name}: {stats:?}");
                }
            }
        }
    }
    assert_no_failures(&failures);
}

/// `OpusDecoder::new` calls `SilkDecoder::init`, which sets both channels
/// up for the decoder's own channel count before any packet has arrived.
/// libopus has no such state. `decode_with_control` must still behave like
/// libopus afterwards, whatever the first packets code: here a stereo
/// decoder that starts with mono packets, and the reverse.
#[test]
fn decoder_set_up_with_init_matches_libopus() {
    let mut failures = Vec::new();
    for (layout, packets) in [
        ("no CELT packets", stream(|_| false)),
        ("CELT packet between all segments", stream(|_| true)),
    ] {
        for channels in [2, 1] {
            // Start at segment 0 (stereo) and at segment 1 (mono).
            for start in [0, packets.len() / 12] {
                let packets = &packets[start..];
                for rate in [8000, 16000, 24000, 48000] {
                    let name = format!(
                        "{layout}, from packet {start}, {channels} ch decoder at {rate} Hz after init()"
                    );
                    let dec = RsDec::with_init(rate, channels);
                    let lost = |i: usize| i % 5 == 3;
                    let (diff, _) = run_with(dec, packets, rate, channels, fate_with_fec(lost));
                    failures.extend(diff.failure(&name));
                }
            }
        }
    }
    assert_no_failures(&failures);
}
