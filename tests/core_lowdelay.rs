//! The `RestrictedLowDelay` application: libopus filters the input of a
//! low-delay encoder like that of any other (`dc_reject` at 3 Hz,
//! opus_encoder.c:1996-2010). Only the look-ahead is left out.

mod core_common;
mod k24_common;

use std::num::NonZeroU32;
use core_common::*;
use k24_common::{speech_fullband, to_stereo};
use opus1::{Application, Bitrate, Channels, OpusEncoder, SampleRate};

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
        let mut enc = OpusEncoder::new(
            SampleRate::try_from(rate as i32).unwrap(),
            Channels::try_from(channels).unwrap(),
            Application::RestrictedLowDelay,
        )
        .unwrap();
        enc.set_bitrate(Bitrate::BitsPerSecond(NonZeroU32::new(bitrate as u32).unwrap()));
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

/// After a stereo signal stops, the output of the DC filter decays for
/// about half a second before it falls below CELT's silence threshold. Its
/// two channels are then scaled copies of each other, at peak levels from
/// about 1e-3 down to 6e-8. Every packet of these frames must be libopus's:
/// CELT-only at 8, 16 and 48 kHz, 10 and 20 ms, constant and variable
/// bitrate, with output buffers from 25 bytes (a mono stream with a
/// narrowed stereo image, `stereo_fade`) to 1275 bytes (stereo at
/// 510 kb/s).
///
/// Before, a few packets in each stream differed (up to 6 of 600). The
/// causes were the `EPSILON` terms of `intensity_stereo` and of the
/// transient analysis, and the rounding of the gain in `stereo_fade`.
#[test]
#[cfg_attr(not(target_arch = "aarch64"), ignore = "bit-exact with libopus only on 64-bit ARM (docs/guide/10, Platforms)")]
fn the_decaying_output_of_the_dc_filter_is_coded_as_libopus_codes_it() {
    let mut seed = 0x6c07_8965_d1b5_4a32u64;
    let mut pick = move |n: usize| {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 16) as usize % n
    };
    // A debug build (the `no_std` check runs one) codes a part of the matrix.
    let debug = cfg!(debug_assertions);
    let seconds = if debug { 2.0 } else { 12.0 };
    let rates: &[usize] = if debug { &[8000, 48000] } else { &[8000, 16000, 48000] };
    let durations: &[usize] = if debug { &[20] } else { &[10, 20] };
    let buffers: &[usize] = if debug { &[25, 1275] } else { &[25, 50, 100, 1275] };
    let mut packets = 0;
    for &rate in rates {
        // Speech in stretches of 50 to 400 ms, with digital silence of 0.8
        // to 1.6 s between them.
        let mut input = to_stereo(&speech_fullband(rate, seconds), rate);
        let samples = input.len() / 2;
        let (mut pos, mut silent) = (0, false);
        while pos < samples {
            let ms = if silent { 800 + pick(801) } else { 50 + pick(351) };
            let len = (rate * ms / 1000).min(samples - pos);
            if silent {
                input[2 * pos..2 * (pos + len)].fill(0.0);
            }
            silent = !silent;
            pos += len;
        }
        for &frame_ms in durations {
            let frame = rate * frame_ms / 1000;
            for &out_len in buffers {
                for vbr in [false, true] {
                    let what = format!("{rate} Hz, {frame_ms} ms, {out_len}-byte buffer, vbr {vbr}");
                    let mut enc = OpusEncoder::new(
                        SampleRate::try_from(rate as i32).unwrap(),
                        Channels::Stereo,
                        Application::RestrictedLowDelay,
                    )
                    .unwrap();
                    enc.set_bitrate(Bitrate::BitsPerSecond(NonZeroU32::new(510_000).unwrap()));
                    enc.set_complexity(5).unwrap();
                    enc.set_vbr(vbr);
                    let mut c_enc = CEncoder::new(rate, 2, APP_LOWDELAY);
                    c_enc.bitrate(510_000).vbr(vbr).complexity(5);
                    let mut buf = vec![0u8; out_len];
                    for (i, pcm) in input.chunks_exact(frame * 2).enumerate() {
                        let n = enc.encode(pcm, frame, &mut buf).unwrap();
                        let c_packet = c_enc.try_encode(pcm, frame, out_len).unwrap();
                        assert_eq!(&buf[..n], &c_packet[..], "{what}: packet {i}");
                        packets += 1;
                    }
                }
            }
        }
    }
    println!("{packets} packets, all equal to libopus's");
}
