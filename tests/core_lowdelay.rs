//! The `RestrictedLowDelay` application: libopus filters the input of a
//! low-delay encoder like that of any other (`dc_reject` at 3 Hz,
//! opus_encoder.c:1996-2010). Only the look-ahead is left out.

mod core_common;
mod k24_common;

use core_common::*;
use k24_common::{speech_fullband, to_stereo};
use opus1::{Application, Bitrate, OpusEncoder};

fn mean(pcm: &[f32]) -> f64 {
    pcm.iter().map(|&v| v as f64).sum::<f64>() / pcm.len() as f64
}

/// Speech on top of a constant offset of 0.1. The filter removes the offset
/// within a fraction of a second, so the last second of the decoded signal
/// has none, as in libopus's stream. Without the filter the offset stays
/// (it is coded and decoded like any other low frequency).
#[test]
fn low_delay_input_is_dc_rejected_like_libopus() {
    for (rate, channels, frame_ms, bitrate) in [(48000usize, 1usize, 10usize, 64000), (48000, 2, 5, 128000), (24000, 1, 20, 48000), (16000, 2, 20, 64000)] {
        let what = format!("{rate} Hz, {channels} ch, {frame_ms} ms");
        let frame = rate * frame_ms / 1000;
        let mono = speech_fullband(rate, 3.0);
        let mut input = if channels == 2 { to_stereo(&mono, rate) } else { mono };
        for v in &mut input {
            *v = 0.5 * *v + 0.1;
        }
        let mut c_enc = CEncoder::new(rate, channels, APP_LOWDELAY);
        c_enc.bitrate(bitrate).complexity(5);
        let mut enc = OpusEncoder::new(rate as i32, channels, Application::RestrictedLowDelay).unwrap();
        enc.set_bitrate(Bitrate::BitsPerSecond(bitrate)).unwrap();
        enc.set_complexity(5).unwrap();
        let (mut dec, mut c_dec) = (CDecoder::new(rate, channels), CDecoder::new(rate, channels));
        let mut buf = vec![0u8; 4000];
        let (mut out, mut c_out) = (Vec::new(), Vec::new());
        let (mut bytes, mut c_bytes) = (0usize, 0usize);
        for (i, pcm) in input.chunks_exact(frame * channels).enumerate() {
            let n = enc.encode(pcm, frame, &mut buf).unwrap();
            let c_packet = c_enc.encode(pcm);
            assert_eq!(buf[0] & 0xFC, c_packet[0] & 0xFC, "{what}: first byte of packet {i}");
            out.extend(dec.decode(Some(&buf[..n]), frame, false).unwrap());
            assert_eq!(enc.final_range(), dec.final_range(), "{what}: final range of packet {i}");
            c_out.extend(c_dec.decode(Some(&c_packet), frame, false).unwrap());
            bytes += n;
            c_bytes += c_packet.len();
        }
        let tail = rate * channels;
        let (dc, c_dc) = (mean(&out[out.len() - tail..]), mean(&c_out[c_out.len() - tail..]));
        let input_dc = mean(&input[input.len() - tail..]);
        println!("{what}: offset {dc:.4}, libopus {c_dc:.4}, input {input_dc:.4}; {bytes} bytes, libopus {c_bytes}");
        assert!(input_dc > 0.09, "{what}: input offset {input_dc:.4}");
        assert!(c_dc.abs() < 0.01, "{what}: libopus keeps an offset of {c_dc:.4}");
        assert!((dc - c_dc).abs() < 0.005, "{what}: offset {dc:.4}, libopus {c_dc:.4}");
        // The streams take the same number of bytes, within 5 %.
        assert!((bytes as f64 / c_bytes as f64 - 1.0).abs() < 0.05, "{what}: {bytes} bytes, libopus {c_bytes}");
    }
}
