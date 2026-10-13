//! The soft clipper's memory when 16-bit and float decoding are mixed in
//! one decoder, against C libopus.
//!
//! `opus_decode` soft-clips a packet and keeps, per channel, the curve that
//! was being applied at the end, so that the next packet continues it.
//! `opus_decode_float` does not clip and clears that memory
//! (`opus_decode_native`: `st->softclip_mem[0] = st->softclip_mem[1] = 0`).

mod core_common;

use core_common::*;
use opus_rs::OpusDecoder;
use std::f64::consts::PI;

/// A low tone far beyond full scale: every 2.5 ms frame ends inside a
/// stretch that the soft clipper is bending.
fn loud_low_tone(rate: usize, channels: usize, seconds: f64) -> Vec<f32> {
    let n = (rate as f64 * seconds) as usize;
    (0..n)
        .flat_map(|i| {
            let t = i as f64 / rate as f64;
            let v = 1.8 * (2.0 * PI * 70.0 * t).sin() + 0.1 * (2.0 * PI * 1900.0 * t).sin();
            (0..channels).map(move |c| if c == 0 { v as f32 } else { -0.9 * v as f32 })
        })
        .collect()
}

/// Largest difference between two 16-bit signals.
fn max_diff(reference: &[i16], ours: &[i16]) -> i32 {
    assert_eq!(reference.len(), ours.len());
    reference.iter().zip(ours).map(|(a, b)| (*a as i32 - *b as i32).abs()).max().unwrap_or(0)
}

/// 16-bit and float decoding in turn, in several patterns. Every 16-bit
/// packet is libopus's within one step, and every float packet matches
/// libopus's unclipped samples.
///
/// If the float decode left the memory alone, the next 16-bit packet would
/// start by bending samples with a curve from two packets earlier, and come
/// out hundreds of steps away from libopus.
#[test]
fn float_decoding_clears_the_soft_clip_memory() {
    for (rate, channels, frame_2_5ms) in [(48000usize, 1usize, 1usize), (48000, 2, 2), (24000, 1, 4), (16000, 2, 1)] {
        let frame = rate / 400 * frame_2_5ms;
        let pcm = loud_low_tone(rate, channels, 0.5);
        let mut enc = CEncoder::new(rate, channels, APP_AUDIO);
        enc.force_mode(MODE_CELT).bitrate(96000 * channels as i32);
        let packets = enc.encode_all(&pcm, frame);
        // Which packets go through the 16-bit path: every second one, two
        // of every three, one of every four.
        for pattern in [[true, false, true, false], [true, true, false, true], [true, false, false, false]] {
            let mut c_dec = CDecoder::new(rate, channels);
            let mut dec = OpusDecoder::new(rate as i32, channels).unwrap();
            let mut buf16 = vec![0i16; frame * channels];
            let mut buf = vec![0f32; frame * channels];
            let (mut worst16, mut worst_float, mut clipped, mut over) = (0, 0f32, 0, 0);
            for (i, p) in packets.iter().enumerate() {
                if pattern[i % 4] {
                    let reference = c_dec.decode_i16(Some(p), frame, false).unwrap();
                    assert_eq!(dec.decode_i16(p, frame, &mut buf16), Ok(frame));
                    worst16 = worst16.max(max_diff(&reference, &buf16));
                    clipped += reference.iter().filter(|&&v| v == i16::MAX || v == i16::MIN).count();
                } else {
                    let reference = c_dec.decode(Some(p), frame, false).unwrap();
                    assert_eq!(dec.decode(p, frame, &mut buf), Ok(frame));
                    let diff = reference.iter().zip(&buf).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
                    worst_float = worst_float.max(diff);
                    over += buf.iter().filter(|v| v.abs() > 1.0).count();
                }
            }
            let what = format!("{rate} Hz, {channels} ch, {} ms, pattern {pattern:?}", frame_2_5ms as f32 * 2.5);
            println!("{what}: 16-bit packets differ by at most {worst16}, float packets by {worst_float:.1e}; {clipped} samples at full scale, {over} float samples beyond 1.0");
            assert!(clipped > 100 && over > 1000, "{what}: the signal was meant to clip");
            assert!(worst16 <= 1, "{what}: 16-bit output differs from libopus by {worst16}");
            assert!(worst_float < 1e-4, "{what}: float output differs from libopus by {worst_float}");
        }
    }
}
