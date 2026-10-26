//! Packets of `OpusEncoder` against the fixed-point build of libopus
//! 1.6.1, byte for byte: SILK-only VoIP, mono, 16-bit input.
//!
//! opus1's SILK encoder is a port of libopus's fixed-point encoder
//! (`silk/fixed`). The libopus that the tests link is the float build, so
//! the reference here is the fixed-point build of the official libopus
//! 1.6.1 release that the dev-dependency `libopus-fixed` compiles
//! (tools/libopus-fixed). Both encoders get the same samples and the same
//! settings, packet by packet.
//!
//! libopus's main branch adds 1 to the frame energy of the Burg analysis
//! (line 72 of silk/fixed/burg_modified_FIX.c). Against its packets,
//! packets 4 to 19 of `test_wb_c10` differed, and no packet of the other
//! streams.

use libopus_fixed::consts::*;
use std::num::NonZeroU32;

/// A 440 Hz sine at half of full scale, as 16-bit samples. The phase is
/// reduced with integers first, so that `sin` gets an argument in
/// [0, 2 pi): f64 `sin` there is the same on every IEEE-754 platform.
fn sine_440(fs: i32, num_samples: usize) -> Vec<i16> {
    (0..num_samples)
        .map(|i| {
            let cycle = (440i64 * i as i64) % fs as i64;
            let phase = cycle as f64 / fs as f64;
            (f64::sin(2.0 * std::f64::consts::PI * phase) * 16384.0) as i16
        })
        .collect()
}

/// 20 packets of 20 ms with a constant bitrate. opus1 gets the float
/// samples `s / 32768` through `encode`, libopus the 16-bit samples `s`
/// through `opus_encode`.
fn run_config(fs: i32, bitrate: i32, complexity: i32, label: &str) {
    use opus1::{Application, Bitrate, Channels, OpusEncoder, SampleRate};
    let frame_size = fs as usize / 50;
    let frames = 20;
    let pcm = sine_440(fs, frame_size * frames);

    let mut encoder = OpusEncoder::new(
        SampleRate::try_from(fs).unwrap(),
        Channels::Mono,
        Application::Voip,
    )
    .expect("encoder");
    encoder.set_complexity(complexity as u8).unwrap();
    encoder.set_bitrate(Bitrate::BitsPerSecond(NonZeroU32::new(bitrate as u32).unwrap()));
    encoder.set_vbr(false);
    let mut c_enc = libopus_fixed::Encoder::new(fs, 1, OPUS_APPLICATION_VOIP);
    c_enc.set(OPUS_SET_COMPLEXITY_REQUEST, complexity);
    c_enc.set(OPUS_SET_BITRATE_REQUEST, bitrate);
    c_enc.set(OPUS_SET_VBR_REQUEST, 0);

    let mut mismatches = 0;
    for (i, input) in pcm.chunks_exact(frame_size).enumerate() {
        let c_ref = c_enc.encode(input, frame_size, 1275).unwrap();
        let float: Vec<f32> = input.iter().map(|&s| s as f32 / 32768.0).collect();
        let mut out = vec![0u8; 1275];
        let n = encoder.encode(&float, frame_size, &mut out).unwrap();
        let pkt = &out[..n];
        if pkt != c_ref.as_slice() {
            mismatches += 1;
            println!("[{label}] Frame {i}: MISMATCH");
            println!("  C:    {c_ref:02x?}");
            println!("  Rust: {pkt:02x?}");
            if let Some(j) = pkt.iter().zip(&c_ref).position(|(a, b)| a != b) {
                println!("  First diff byte {j}: C=0x{:02x} Rust=0x{:02x}", c_ref[j], pkt[j]);
            }
        }
    }
    println!("[{label}] {}/{frames} frames match", frames - mismatches);
    assert_eq!(mismatches, 0, "{label}: {mismatches} frames mismatched");
}

#[test]
fn test_nb_c0() {
    run_config(8000, 10000, 0, "nb_c0");
}

#[test]
fn test_nb_c1() {
    run_config(8000, 10000, 1, "nb_c1");
}

#[test]
fn test_nb_c2() {
    run_config(8000, 10000, 2, "nb_c2");
}

#[test]
fn test_nb_c5() {
    run_config(8000, 10000, 5, "nb_c5");
}

#[test]
fn test_nb_c10() {
    run_config(8000, 10000, 10, "nb_c10");
}

#[test]
fn test_nb_6k_c5() {
    run_config(8000, 6000, 5, "nb_6k_c5");
}

#[test]
fn test_nb_16k_c5() {
    run_config(8000, 16000, 5, "nb_16k_c5");
}

#[test]
fn test_mb_c0() {
    run_config(12000, 14000, 0, "mb_c0");
}

#[test]
fn test_mb_c5() {
    run_config(12000, 14000, 5, "mb_c5");
}

#[test]
fn test_mb_c10() {
    run_config(12000, 14000, 10, "mb_c10");
}

#[test]
fn test_wb_c0() {
    run_config(16000, 20000, 0, "wb_c0");
}

#[test]
fn test_wb_c5() {
    run_config(16000, 20000, 5, "wb_c5");
}

#[test]
fn test_wb_c10() {
    run_config(16000, 20000, 10, "wb_c10");
}

#[test]
fn test_wb_32k_c5() {
    run_config(16000, 32000, 5, "wb_32k_c5");
}

/// A voiced signal for `long_voiced_stream_is_the_stream_of_fixed_point_libopus`:
/// a pulse train whose period changes every quarter second (100 to 159
/// samples at 16 kHz) through two resonators, with a little noise.
fn voiced_signal(samples: usize) -> Vec<i16> {
    let mut seed: u32 = 0x600d_c0de;
    let (mut y1, mut y2, mut z1, mut z2) = (0i64, 0i64, 0i64, 0i64);
    let mut phase = 0usize;
    (0..samples)
        .map(|n| {
            let period = 100 + (n / 4000) * 37 % 60;
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            let noise = ((seed >> 8) & 0xffff) as i64 - 32768;
            phase += 1;
            if phase >= period {
                phase = 0;
            }
            let e = if phase == 0 { 20000 } else { noise >> 9 };
            let y = e + ((26000 * y1) >> 14) + ((-11000 * y2) >> 14);
            y2 = y1;
            y1 = y;
            let z = y + ((18000 * z1) >> 14) + ((-12500 * z2) >> 14);
            z2 = z1;
            z1 = z;
            (z >> 2).clamp(-30000, 30000) as i16
        })
        .collect()
}

/// Six seconds of loud voiced input: 300 packets against the fixed-point
/// build of libopus 1.6.1 (16 kHz mono, VoIP, 20 kb/s, variable bitrate,
/// complexity 5, 16-bit input).
///
/// The streams above are too short to show what this one shows. After a
/// voiced frame the encoder moves the cutoff of its input high-pass filter
/// towards the pitch frequency, through two smoothers. Both had a
/// coefficient that was one unit off, so after a few seconds the cutoff
/// changed by 1 Hz in another frame than in libopus, and from there on every
/// packet differed:
///
/// - `SILK_FIX_CONST( VARIABLE_HP_SMTH_COEF1, 16 )` is 6554;
///   src/silk/hp_variable_cutoff.rs had 6553. Corrected.
/// - `SILK_FIX_CONST( VARIABLE_HP_SMTH_COEF2, 16 )` is 983 (libopus
///   src/opus_encoder.c, line 1975); src/lib.rs had
///   `VARIABLE_HP_SMTH_COEF2_Q16 = 984`. Corrected.
///
/// Measured against the fixed-point build of libopus's main branch, from
/// which this test first took recorded hashes: 206 of the 300 packets
/// equal libopus with both constants off, 238 with the first one corrected,
/// 300 with both.
#[test]
fn long_voiced_stream_is_the_stream_of_fixed_point_libopus() {
    use opus1::{Application, Bitrate, Channels, OpusEncoder, SampleRate};
    let frame_size = 320;
    let pcm = voiced_signal(frame_size * 300);
    let mut encoder =
        OpusEncoder::new(SampleRate::Hz16000, Channels::Mono, Application::Voip).expect("encoder");
    encoder.set_complexity(5).unwrap();
    encoder.set_bitrate(Bitrate::BitsPerSecond(NonZeroU32::new(20000).unwrap()));
    encoder.set_vbr(true);
    let mut c_enc = libopus_fixed::Encoder::new(16000, 1, OPUS_APPLICATION_VOIP);
    c_enc.set(OPUS_SET_COMPLEXITY_REQUEST, 5);
    c_enc.set(OPUS_SET_BITRATE_REQUEST, 20000);
    c_enc.set(OPUS_SET_VBR_REQUEST, 1);

    let mut out = vec![0u8; 1275];
    let mut differ = Vec::new();
    for (i, frame) in pcm.chunks_exact(frame_size).enumerate() {
        let c_packet = c_enc.encode(frame, frame_size, 1275).unwrap();
        let n = encoder.encode_i16(frame, frame_size, &mut out).unwrap();
        if out[..n] != c_packet[..] {
            differ.push(i);
        }
    }
    assert!(differ.is_empty(), "{} of 300 packets differ from libopus: {differ:?}", differ.len());
}
