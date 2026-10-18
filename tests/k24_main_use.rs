//! The main use of the 24 kHz path, end to end: mono speech, VoIP, 16 to
//! 40 kb/s, 20 ms frames, encoded from and decoded to 24 kHz.
//!
//! The other `k24_*` tests force one coding mode per stream. Here libopus
//! chooses the mode itself, as it does in a real call: SILK-only, Hybrid
//! and CELT-only frames follow each other in one stream, and the choice
//! depends on the bitrate and on what the signal contains. The speech is a
//! part of the fixture that the other tests do not use (from 5 s on).
//!
//! Three checks, all against libopus's decode of the same packets:
//!
//! 1. libopus encoder at a fixed bitrate.
//! 2. libopus encoder whose bitrate changes every 250 ms, which makes it
//!    switch modes on its own.
//! 3. opus1's own encoder, through both of its entry points (`f32` and
//!    16-bit integer input).
//!
//! Measured: above 132 dB in the worst 20 ms frame of every stream (the
//! SILK-only frames are bit-exact), and the two decoders never differ in
//! delay. Each check requires 100 dB.

mod k24_common;
use k24_common::*;
use opus1::Bitrate;

/// `seconds` of the speech fixture from `start_s` on, at 24 kHz, in two
/// versions: as it is (nothing above 8 kHz, the fixture is 16 kHz speech),
/// and with envelope-following noise added so the band up to 12 kHz is used.
fn speech_from(start_s: f64, seconds: f64) -> [(&'static str, Vec<f32>); 2] {
    let src = load_speech_16k();
    let a = (start_s * 16000.0) as usize;
    let n = (seconds * 16000.0) as usize;
    let plain = resample(&src[a..a + n], 16000, 24000);
    let mut seed = 0x0bad_5eedu32;
    let full: Vec<f32> = plain
        .iter()
        .map(|&s| {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            let white = (seed >> 8) as f32 / (1u32 << 23) as f32 - 1.0;
            s + 0.25 * s * white
        })
        .collect();
    [("speech", plain), ("speech with content up to 12 kHz", full)]
}

/// Decode with both decoders and require the same length, no delay
/// difference and at least 100 dB in the worst 20 ms frame. Returns the
/// comparison.
fn check(label: &str, packets: &[Vec<u8>]) -> Cmp {
    let reference = c_decode_all(packets, 24000, 1);
    let ours = rs_decode_all(packets, 24000, 1).unwrap_or_else(|(i, e)| panic!("{label}: packet {i}: {e}"));
    assert_eq!(ours.len(), reference.len(), "{label}");
    // The speech here stays below full scale, so the clamp of opus1's
    // float output to [-1, 1] (see k24_known_gaps.rs) plays no part.
    assert!(reference.iter().all(|v| v.abs() <= 1.0), "{label}: reference clips");
    let cmp = compare(&reference, &ours, 1, 480, 24000);
    println!("{label} [{}]: {cmp}", toc_summary(packets));
    assert!(cmp.worst_frame_snr_db > 100.0 && cmp.lag == 0, "{label}: {cmp}");
    cmp
}

#[test]
fn libopus_voip_stream_decodes_like_libopus() {
    let mut modes_seen = std::collections::BTreeSet::new();
    for (signal, pcm) in &speech_from(5.0, 4.0) {
        for bitrate in [16000, 20000, 24000, 32000, 40000] {
            let mut enc = CEnc::voip(24000, 1);
            enc.bitrate(bitrate);
            let packets = enc.encode_all(pcm, 480);
            for p in &packets {
                modes_seen.insert(toc(p[0]).mode);
            }
            check(&format!("libopus encoder, {signal}, {bitrate} b/s"), &packets);
        }
    }
    // The point of the test: libopus used all three modes by itself.
    assert_eq!(modes_seen.len(), 3, "modes in the test streams: {modes_seen:?}");
}

#[test]
fn libopus_stream_with_changing_bitrate_decodes_like_libopus() {
    let [_, (signal, pcm)] = speech_from(5.0, 6.0);
    let plan = [9000, 40000, 13000, 28000, 64000, 16000, 48000, 11000, 36000, 20000, 80000, 24000];
    let mut enc = CEnc::voip(24000, 1);
    let mut packets = Vec::new();
    for (i, frame) in pcm.as_chunks::<480>().0.iter().enumerate() {
        // A new bitrate every 250 ms.
        enc.bitrate(plan[(i * 20 / 250) % plan.len()]);
        packets.push(enc.encode(frame));
    }
    let switches = packets.windows(2).filter(|w| toc(w[0][0]).mode != toc(w[1][0]).mode).count();
    assert!(switches >= 6, "libopus switched modes only {switches} times: {}", toc_summary(&packets));
    check(&format!("libopus encoder, {signal}, bitrate changing every 250 ms, {switches} mode switches"), &packets);
}

#[test]
fn own_encoder_stream_decodes_like_libopus() {
    for (signal, pcm) in &speech_from(5.0, 4.0) {
        // The same samples as 16-bit integers, and as the floats they stand for.
        let pcm_i16: Vec<i16> = pcm.iter().map(|v| (v * 32768.0).round().clamp(-32768.0, 32767.0) as i16).collect();
        let pcm_f32: Vec<f32> = pcm_i16.iter().map(|v| *v as f32 / 32768.0).collect();
        for bitrate in [16000, 24000, 32000, 40000] {
            let mut enc_f = opus1::OpusEncoder::new(24000, 1, opus1::Application::Voip).unwrap();
            let mut enc_i = opus1::OpusEncoder::new(24000, 1, opus1::Application::Voip).unwrap();
            enc_f.set_bitrate(Bitrate::BitsPerSecond(bitrate)).unwrap();
            enc_i.set_bitrate(Bitrate::BitsPerSecond(bitrate)).unwrap();
            // `encode_i16` codes at a depth of 16 bits, as libopus
            // `opus_encode` does; the float input says so to give the
            // same packets.
            enc_f.set_lsb_depth(16).unwrap();
            let mut out_f = vec![0u8; 4000];
            let mut out_i = vec![0u8; 4000];
            let mut packets = Vec::new();
            for (f, i) in pcm_f32.as_chunks::<480>().0.iter().zip(pcm_i16.as_chunks::<480>().0) {
                let n = enc_f.encode(f, 480, &mut out_f).unwrap();
                let m = enc_i.encode_i16(i, 480, &mut out_i).unwrap();
                assert_eq!(out_f[..n], out_i[..m], "{signal}, {bitrate} b/s: encode and encode_i16 differ");
                packets.push(out_f[..n].to_vec());
            }
            check(&format!("opus1 encoder, {signal}, {bitrate} b/s"), &packets);
        }
    }
}
