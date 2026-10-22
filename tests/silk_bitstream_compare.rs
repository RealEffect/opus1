// Verification scripts intentionally mirror the C reference style
// (index-based loops, grouped hex tables) — keep clippy quiet.
#![allow(clippy::needless_range_loop, clippy::unreadable_literal)]

/// Test to compare Rust SILK encoder output with C reference
///
/// Note: `test_silk_bitstream_vs_c_reference` was removed because it compared our
/// fixed-point Rust encoder against a mismatched floating-point reference, which
/// can never produce byte-identical output. Fixed-point compatibility coverage
/// remains in `tests/silk_fixed_point_compare.rs`.
use std::num::NonZeroU32;
use opus1::{Application, Bitrate, Channels, OpusEncoder, SampleRate};
use std::f32::consts::PI;

/// Test TOC byte structure is correct for SILK NB 20ms CBR
#[test]
fn test_silk_toc_byte_structure() {
    let frame_size = 160;

    let mut encoder =
        OpusEncoder::new(SampleRate::Hz8000, Channels::Mono, Application::Voip).expect("Failed to create encoder");
    encoder.set_complexity(0).unwrap();
    encoder.set_bitrate(Bitrate::BitsPerSecond(NonZeroU32::new(10000).unwrap()));
    encoder.set_vbr(false);

    let mut input = vec![0.0f32; frame_size];
    for i in 0..frame_size {
        input[i] = (2.0f32 * PI * 440.0f32 * i as f32 / 8000.0f32).sin();
    }

    let mut output = vec![0u8; 25];
    let bytes = encoder
        .encode(&input, frame_size, &mut output)
        .expect("Encode failed");

    assert!(bytes >= 3, "Packet too short: {}", bytes);

    // TOC byte: SILK-only NB 20ms mono = 0x0b
    // bits[7:3] = config (NB 20ms = 0b00001)
    // bit[2]    = stereo = 0
    // bits[1:0] = frame_count_code = 3 (Code 3)
    assert_eq!(
        output[0], 0x0b,
        "TOC byte mismatch: got 0x{:02x}",
        output[0]
    );

    // Code 3 second byte: count byte, 1 frame, no padding = 0x01
    assert_eq!(
        output[1], 0x01,
        "Count byte mismatch: got 0x{:02x}",
        output[1]
    );
}

/// Test that multiple consecutive frames produce correct output sizes
#[test]
fn test_silk_multi_frame_sizes() {
    let sample_rate = 8000;
    let frame_size = 160;

    let mut encoder =
        OpusEncoder::new(SampleRate::Hz8000, Channels::Mono, Application::Voip).expect("Failed to create encoder");
    encoder.set_complexity(0).unwrap();
    encoder.set_bitrate(Bitrate::BitsPerSecond(NonZeroU32::new(10000).unwrap()));
    encoder.set_vbr(false);

    for frame_idx in 0..5 {
        let mut input = vec![0.0f32; frame_size];
        for i in 0..frame_size {
            let t = (frame_idx * frame_size + i) as f32 / sample_rate as f32;
            input[i] = (2.0f32 * PI * 440.0f32 * t).sin();
        }

        let mut output = vec![0u8; 25];
        let bytes = encoder
            .encode(&input, frame_size, &mut output)
            .expect("Encode failed");

        assert!(
            bytes >= 3,
            "Frame {}: packet too short: {}",
            frame_idx,
            bytes
        );
        assert!(
            bytes <= 25,
            "Frame {}: packet too large: {}",
            frame_idx,
            bytes
        );
        println!(
            "Frame {}: {} bytes, hex: {}",
            frame_idx,
            bytes,
            hex::encode(&output[..bytes])
        );
    }
}

// Helper to display hex
mod hex {
    pub fn encode(data: &[u8]) -> String {
        let mut s = String::with_capacity(data.len() * 2);
        for &b in data {
            let _ = std::fmt::Write::write_fmt(&mut s, format_args!("{:02x}", b));
        }
        s
    }
}
