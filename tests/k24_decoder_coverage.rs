//! Decoder coverage at a 24 kHz output rate: frame durations, packets with
//! several frames, padding, other signals, discontinuous transmission and
//! the `frame_size` argument. Everything is compared with libopus decoding
//! the same libopus-encoded packets.
//!
//! Required agreement: SILK-only bit-exact, everything else better than
//! 80 dB in the worst 20 ms (CELT is floating-point).

mod k24_common;
use k24_common::*;

const MODES: [(i32, i32, &str, i32); 9] = [
    (MODE_SILK, BW_NB, "SILK-NB", 16000),
    (MODE_SILK, BW_MB, "SILK-MB", 20000),
    (MODE_SILK, BW_WB, "SILK-WB", 24000),
    (MODE_HYBRID, BW_SWB, "Hybrid-SWB", 32000),
    (MODE_HYBRID, BW_FB, "Hybrid-FB", 40000),
    (MODE_CELT, BW_NB, "CELT-NB", 32000),
    (MODE_CELT, BW_WB, "CELT-WB", 40000),
    (MODE_CELT, BW_SWB, "CELT-SWB", 48000),
    (MODE_CELT, BW_FB, "CELT-FB", 64000),
];

/// Compare both decoders on `packets` in 20 ms windows and check the
/// requirement. Returns the worst-window SNR.
fn check(label: &str, packets: &[Vec<u8>], rate: usize, failures: &mut Vec<String>) -> f64 {
    let reference = c_decode_all(packets, rate, 1);
    let ours = match rs_decode_all(packets, rate, 1) {
        Ok(v) => v,
        Err((i, e)) => {
            failures.push(format!("{label}: packet {i}: {e}"));
            return f64::NEG_INFINITY;
        }
    };
    let cmp = compare(&reference, &ours, 1, rate / 50, rate);
    let silk_only = packets.iter().all(|p| toc(p[0]).mode == "SILK");
    let ok = ours.len() == reference.len()
        && cmp.lag == 0
        && if silk_only { cmp.max_abs_diff == 0.0 } else { cmp.worst_frame_snr_db > 80.0 };
    if !ok {
        failures.push(format!("{label} [{}]: {cmp}", toc_summary(packets)));
    }
    cmp.worst_frame_snr_db
}

/// Requested frame durations from 2.5 to 60 ms in every mode. libopus maps
/// what a mode cannot code: SILK and Hybrid below 10 ms become CELT, and
/// Hybrid or CELT above 20 ms becomes a packet of several 20 ms frames.
#[test]
fn frame_durations_from_2_5_to_60_ms() {
    let pcm = speech_fullband(48000, 1.5);
    let mut failures = Vec::new();
    let mut lowest = f64::INFINITY;
    let mut seen = std::collections::BTreeSet::new();
    for &(mode, bw, name, bitrate) in &MODES {
        for tenth_ms in [25usize, 50, 100, 200, 400, 600] {
            let mut enc = CEnc::audio(48000, 1);
            enc.force_mode(mode).bandwidth(bw).bitrate(bitrate);
            let packets = enc.encode_all(&pcm, 48000 * tenth_ms / 10000);
            let t = toc(packets[packets.len() / 2][0]);
            seen.insert((t.mode, t.frame_tenth_ms, t.code));
            for rate in [24000usize, 48000] {
                let label = format!("{name}, {} ms requested, at {rate} Hz", tenth_ms as f32 / 10.0);
                lowest = lowest.min(check(&label, &packets, rate, &mut failures));
            }
        }
    }
    println!("frame durations: lowest worst-window SNR {lowest:.1} dB; packet kinds seen: {seen:?}");
    // The streams must really contain every frame duration and all four
    // frame-count codes.
    for d in [25, 50, 100, 200, 400, 600] {
        assert!(seen.iter().any(|s| s.1 == d), "no {d}/10 ms frames in the test streams");
    }
    for code in 0..4u8 {
        assert!(seen.iter().any(|s| s.2 == code), "no code {code} packets in the test streams");
    }
    assert!(failures.is_empty(), "{} failures:\n{}", failures.len(), failures.join("\n"));
}

/// Packets merged by the libopus repacketizer (frame-count codes 1, 2 and
/// 3, constant and variable frame sizes) and packets with padding.
#[test]
fn repacketized_and_padded_packets() {
    let pcm = speech_fullband(48000, 1.5);
    let mut failures = Vec::new();
    let mut codes = std::collections::BTreeSet::new();
    for &(mode, bw, name, bitrate) in &MODES {
        for cbr in [false, true] {
            let mut enc = CEnc::audio(48000, 1);
            enc.force_mode(mode).bandwidth(bw).bitrate(bitrate).vbr(!cbr);
            let packets = enc.encode_all(&pcm, 960);
            let kind = if cbr { "CBR" } else { "VBR" };
            for group in [2usize, 3, 6] {
                let merged: Vec<Vec<u8>> = packets
                    .chunks_exact(group)
                    .map(|g| repacketize(&g.iter().map(|p| p.as_slice()).collect::<Vec<_>>()))
                    .collect();
                for p in &merged {
                    codes.insert(p[0] & 3);
                }
                check(&format!("{name} {kind}, {group} frames per packet"), &merged, 24000, &mut failures);
            }
            let padded: Vec<Vec<u8>> = packets.iter().map(|p| pad_packet(p, p.len() + 300)).collect();
            check(&format!("{name} {kind}, 300 bytes of padding"), &padded, 24000, &mut failures);
        }
    }
    assert_eq!(codes.len(), 3, "expected codes 1, 2 and 3, got {codes:?}");
    assert!(failures.is_empty(), "{} failures:\n{}", failures.len(), failures.join("\n"));
}

/// Chirp, music-like tones, an isolated impulse and silence through every
/// mode a 24 kHz encoder can produce. The impulse also checks the delay:
/// the comparison requires zero lag between the two decoders.
#[test]
fn other_signals_from_a_24k_encoder() {
    let n = 36000;
    let signals: Vec<(&str, Vec<f32>)> = vec![
        ("chirp", chirp(24000, n)),
        ("multitone", multitone(24000, n)),
        ("impulse", impulse(n, 4800)),
        ("silence", vec![0.0; n]),
    ];
    let mut failures = Vec::new();
    let mut streams = 0;
    for (signal, pcm) in &signals {
        for &(mode, bw, name, bitrate) in &MODES {
            let mut enc = CEnc::voip(24000, 1);
            enc.force_mode(mode).bandwidth(bw).bitrate(bitrate);
            let packets = enc.encode_all(pcm, 480);
            let t = toc(packets[packets.len() / 2][0]);
            if format!("{}-{}", t.mode, t.bandwidth) != name {
                continue; // fullband needs a 48 kHz encoder
            }
            streams += 1;
            check(&format!("{signal} {name}"), &packets, 24000, &mut failures);
        }
    }
    assert_eq!(streams, 4 * 7);
    assert!(failures.is_empty(), "{} failures:\n{}", failures.len(), failures.join("\n"));
}

/// The impulse comes out of both decoders at the same sample.
#[test]
fn impulse_position_matches_libopus() {
    let pcm = impulse(24000, 4800);
    for &(mode, bw, name, bitrate) in &MODES[..4] {
        let mut enc = CEnc::voip(24000, 1);
        enc.force_mode(mode).bandwidth(bw).bitrate(bitrate);
        let packets = enc.encode_all(&pcm, 480);
        let peak = |x: &[f32]| x.iter().enumerate().fold((0, 0f32), |m, (i, v)| if v.abs() > m.1 { (i, v.abs()) } else { m }).0;
        let reference = c_decode_all(&packets, 24000, 1);
        let ours = rs_decode_all(&packets, 24000, 1).unwrap();
        println!("{name}: impulse in at 4800, out at {} (libopus {})", peak(&ours), peak(&reference));
        assert_eq!(peak(&ours), peak(&reference), "{name}");
    }
}

/// Discontinuous transmission in Hybrid and CELT-only streams (SILK-only
/// DTX is covered bit-exactly in k24_silk_loss_cng.rs).
#[test]
fn dtx_in_hybrid_and_celt_streams() {
    let mut pcm = speech_fullband(24000, 1.0);
    pcm.extend(vec![0.0; 48000]);
    pcm.extend(speech_fullband(24000, 1.0));
    let mut failures = Vec::new();
    for &(mode, bw, name, bitrate) in &[(MODE_HYBRID, BW_SWB, "Hybrid-SWB", 32000), (MODE_CELT, BW_SWB, "CELT-SWB", 48000)] {
        let mut enc = CEnc::voip(24000, 1);
        enc.force_mode(mode).bandwidth(bw).bitrate(bitrate).dtx(true);
        let packets = enc.encode_all(&pcm, 480);
        let empty = packets.iter().filter(|p| p.len() <= 2).count();
        assert!(empty > 30, "{name}: expected DTX packets, got {empty}");
        check(&format!("{name} with DTX"), &packets, 24000, &mut failures);
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// The `frame_size` argument: too small is an error, larger is fine, and a
/// lost packet may be concealed in any multiple of 2.5 ms. Both decoders
/// must agree on the result of every call.
#[test]
fn frame_size_argument_matches_libopus() {
    let pcm = speech_fullband(24000, 0.5);
    let mut enc = CEnc::voip(24000, 1);
    enc.force_mode(MODE_HYBRID).bandwidth(BW_SWB).bitrate(32000);
    let packets = enc.encode_all(&pcm, 480);
    for frame_size in [120usize, 240, 479, 480, 481, 960, 2880, 5760] {
        let mut c_dec = CDec::new(24000, 1);
        let mut rs_dec = opus_rs::OpusDecoder::new(24000, 1).unwrap();
        let c = c_dec.decode(Some(&packets[3]), frame_size, false).map(|v| v.len()).ok();
        let mut buf = vec![0f32; frame_size];
        let rs = rs_dec.decode(&packets[3], frame_size, &mut buf).ok();
        assert_eq!(rs, c, "good packet, frame_size {frame_size}");
    }
    for frame_size in [59usize, 60, 120, 240, 300, 360, 480, 600, 960, 1440, 2880, 5760] {
        let mut c_dec = CDec::new(24000, 1);
        let mut rs_dec = opus_rs::OpusDecoder::new(24000, 1).unwrap();
        c_dec.decode(Some(&packets[3]), 480, false).unwrap();
        let mut buf = vec![0f32; 5760];
        rs_dec.decode(&packets[3], 480, &mut buf).unwrap();
        let c = c_dec.decode(None, frame_size, false).map(|v| v.len()).ok();
        let rs = rs_dec.decode(&[], frame_size, &mut buf[..frame_size]).ok();
        assert_eq!(rs, c, "lost packet, frame_size {frame_size}");
    }
}
