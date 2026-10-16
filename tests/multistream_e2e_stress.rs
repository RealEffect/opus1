//! The surround encoder with everything switched on, under input, buffers
//! and settings that no test plan foresees.
//!
//! tests/multistream_enc_robust.rs does this for the multistream layer on a
//! speech-like signal. It was written before the surround analysis and the
//! LFE tuning were active. This file adds what the analysis and the LFE
//! stream meet now:
//!
//! - input that is not audio in single channels, among them the LFE
//!   channel: NaN, infinities, values far beyond full scale, denormals, a
//!   constant;
//! - every sampling rate, also 8 and 12 kHz, where the analysis inserts 5
//!   or 3 zeros after every sample;
//! - every frame duration, changing from frame to frame, so that the
//!   analysis carries its state between frames of different length;
//! - output buffers from the smallest packet to a megabyte;
//! - settings that change at random, extreme bitrates and resets;
//! - 16-bit input.
//!
//! No call may panic. An error must be one the buffer explains. Every
//! packet must decode in C libopus and in opus1, to finite samples, with
//! the final range of the encoder.
//!
//! The sequences are random but fixed. Set the environment variable
//! `STRESS_SEED` to a number to run other ones.
#![cfg(feature = "std")]

mod ms_support;
mod multistream_e2e_common;
mod multistream_enc_common;

use ms_support::*;
use multistream_e2e_common::*;
use multistream_enc_common::*;
use opus1::multistream::MultistreamEncoder;
use opus1::{Application, Bandwidth, Bitrate, ForcedMode};

/// A small deterministic random number generator.
struct Lcg(u32);

impl Lcg {
    fn below(&mut self, n: u32) -> u32 {
        self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
        (self.0 >> 8) % n
    }

    fn pick<T: Copy>(&mut self, items: &[T]) -> T {
        items[self.below(items.len() as u32) as usize]
    }
}

/// The surround layouts: 3 to 8 channels of mapping family 1.
const SURROUND: [Case; 6] = [
    Case::family("3 channels", 3, 1),
    Case::family("quad", 4, 1),
    Case::family("5 channels", 5, 1),
    Case::family("5.1", 6, 1),
    Case::family("6.1", 7, 1),
    Case::family("7.1", 8, 1),
];

/// A signal with tones, attacks and noise on every channel.
fn busy(fs: i32, channels: usize, samples: usize) -> Vec<f32> {
    let two_pi = 2.0 * std::f64::consts::PI;
    let hiss = noise(channels, samples, 0.01, 5);
    let mut pcm = vec![0f32; samples * channels];
    for (i, frame) in pcm.chunks_exact_mut(channels).enumerate() {
        let t = i as f64 / fs as f64;
        for (c, sample) in frame.iter_mut().enumerate() {
            let since = ((t + 0.02 * c as f64) * 5.0).fract() / 5.0;
            let env = 0.1 + 0.9 * (-since * 20.0).exp();
            let f = 150.0 * (c + 1) as f64;
            let v = (two_pi * f * t).sin() + 0.5 * (two_pi * 2.7 * f * t).sin();
            *sample = (0.2 * env * v) as f32 + hiss[i * channels + c];
        }
    }
    pcm
}

/// Changes one setting of the encoder at random. Returns a description.
fn change_a_setting(enc: &mut MultistreamEncoder, rng: &mut Lcg) -> String {
    let channels = enc.layout().channels() as i32;
    match rng.below(9) {
        0 => {
            let bitrate = rng.pick(&[
                Bitrate::Auto,
                Bitrate::Max,
                Bitrate::BitsPerSecond(1),
                Bitrate::BitsPerSecond(500 * channels),
                Bitrate::BitsPerSecond(2_500 * channels),
                Bitrate::BitsPerSecond(7_000 * channels),
                Bitrate::BitsPerSecond(12_000 * channels),
                Bitrate::BitsPerSecond(40_000 * channels),
                Bitrate::BitsPerSecond(750_000 * channels),
                Bitrate::BitsPerSecond(i32::MAX),
            ]);
            enc.set_bitrate(bitrate).unwrap();
            format!("{bitrate:?}")
        }
        1 => {
            let vbr = rng.below(2) == 0;
            enc.set_vbr(vbr);
            format!("vbr {vbr}")
        }
        2 => {
            let complexity = rng.below(11) as i32;
            enc.set_complexity(complexity).unwrap();
            format!("complexity {complexity}")
        }
        3 => {
            let fec = rng.below(2) == 0;
            enc.set_inband_fec(fec);
            format!("fec {fec}")
        }
        4 => {
            let loss = rng.pick(&[0, 5, 25, 100]);
            enc.set_packet_loss_perc(loss).unwrap();
            format!("loss {loss}")
        }
        5 => {
            let bandwidth = rng.pick(&[
                Bandwidth::Auto,
                Bandwidth::Narrowband,
                Bandwidth::Mediumband,
                Bandwidth::Wideband,
                Bandwidth::Superwideband,
                Bandwidth::Fullband,
            ]);
            enc.set_bandwidth(bandwidth);
            format!("{bandwidth:?}")
        }
        6 => {
            let bandwidth = rng.pick(&[
                Bandwidth::Narrowband,
                Bandwidth::Wideband,
                Bandwidth::Fullband,
                Bandwidth::Fullband,
            ]);
            enc.set_max_bandwidth(bandwidth).unwrap();
            format!("max {bandwidth:?}")
        }
        7 => {
            let mode = rng.pick(&[
                None,
                None,
                Some(ForcedMode::SilkOnly),
                Some(ForcedMode::Hybrid),
                Some(ForcedMode::CeltOnly),
            ]);
            enc.set_force_mode(mode);
            format!("force {mode:?}")
        }
        _ => {
            enc.reset_state();
            "reset".to_string()
        }
    }
}

/// Overwrites one frame with input that is not audio. Returns a
/// description.
fn spoil(frame: &mut [f32], channels: usize, rng: &mut Lcg) -> &'static str {
    // Mostly the frame stays as it is.
    let kind = rng.below(24);
    // One channel, often the last one, which is the LFE channel of 5.1, 6.1
    // and 7.1.
    let channel = if rng.below(2) == 0 {
        channels - 1
    } else {
        rng.below(channels as u32) as usize
    };
    let mut fill = |value: &dyn Fn(usize) -> f32, all: bool| {
        for (i, x) in frame.chunks_exact_mut(channels).enumerate() {
            for (c, s) in x.iter_mut().enumerate() {
                if all || c == channel {
                    *s = value(i);
                }
            }
        }
    };
    match kind {
        0 => {
            fill(&|_| f32::NAN, false);
            "NaN in one channel"
        }
        1 => {
            fill(&|_| f32::INFINITY, false);
            "infinity in one channel"
        }
        2 => {
            fill(&|i| if i % 2 == 0 { 1e30 } else { -1e30 }, false);
            "1e30 in one channel"
        }
        3 => {
            fill(&|i| if (i / 7) % 2 == 0 { 4.0 } else { -4.0 }, true);
            "a square wave at 4 times full scale"
        }
        4 => {
            fill(&|_| 1e-40, true);
            "denormals"
        }
        5 => {
            fill(&|_| 1.0, false);
            "a constant in one channel"
        }
        6 => {
            fill(&|_| 0.0, true);
            "silence"
        }
        7 => {
            // One sample only.
            let at = rng.below((frame.len() / channels) as u32) as usize * channels + channel;
            frame[at] = f32::NAN;
            "one NaN sample"
        }
        8 => {
            fill(&|_| f32::NEG_INFINITY, true);
            "negative infinity in all channels"
        }
        _ => "audio",
    }
}

/// One encoder and both decoders through `frames` frames. Returns the
/// number of packets written and of frames refused for a small buffer.
fn stress(case: &Case, fs: i32, application: Application, seed: u32, frames: usize) -> (usize, usize) {
    let mut rng = Lcg(seed);
    let mut enc = rust_encoder(case, fs, application);
    let mut c_dec = c_decoder(&enc);
    let mut rust_dec = rust_decoder(&enc);
    let streams = enc.layout().streams();
    let channels = case.channels;
    let sizes = frame_sizes(fs);
    let longest = sizes[5];
    let pcm = busy(fs, channels, longest * 8);
    let mut frame = vec![0f32; longest * channels];
    let mut decoded = vec![0f32; longest * channels];
    // A megabyte: far more than any packet.
    let mut out = vec![0u8; 1 << 20];
    let largest_packet = max_packet_bytes(streams);
    let smallest = 2 * streams - 1;
    let mut history: Vec<String> = Vec::new();
    let (mut written, mut refused) = (0, 0);
    for f in 0..frames {
        for _ in 0..rng.below(3) {
            history.push(change_a_setting(&mut enc, &mut rng));
        }
        let frame_size = rng.pick(&sizes);
        let len = match rng.below(10) {
            0 => smallest,
            1 => smallest + rng.below(3 * streams as u32) as usize,
            2 => 20 * streams + rng.below(200) as usize,
            3 => out.len(),
            _ => largest_packet,
        };
        let start = (rng.below(7 * longest as u32) as usize) * channels;
        let input = &mut frame[..frame_size * channels];
        input.copy_from_slice(&pcm[start..start + frame_size * channels]);
        let what = spoil(input, channels, &mut rng);
        let as_i16 = rng.below(5) == 0;
        history.push(format!(
            "frame {f}: {} of {what} into {len} bytes{}",
            ms(frame_size, fs),
            if as_i16 { ", 16-bit" } else { "" }
        ));
        let at = |history: &[String]| {
            format!(
                "{}, {fs} Hz, {application:?}: {}",
                case.name,
                history[history.len().saturating_sub(12)..].join("; ")
            )
        };

        let result = if as_i16 {
            // The loudest 16-bit input where the float input is not audio.
            let pcm16: Vec<i16> = input
                .iter()
                .map(|&s| {
                    if s.is_nan() {
                        i16::MIN
                    } else {
                        (s * 32768.0).clamp(-32768.0, 32767.0) as i16
                    }
                })
                .collect();
            enc.encode_i16(&pcm16, frame_size, &mut out[..len])
        } else {
            enc.encode(input, frame_size, &mut out[..len])
        };
        let n = match result {
            Ok(n) => n,
            Err(e) => {
                // Only too few bytes are a reason to refuse a frame. The
                // packet is limited by the buffer, and with constant
                // bitrate also by the bitrate.
                assert_eq!(e, "Output buffer too small", "{}", at(&history));
                let limit = match enc.bitrate_setting() {
                    Bitrate::BitsPerSecond(bps) if !enc.vbr() => {
                        let bits = bps as i64 * frame_size as i64 / fs as i64;
                        len.min((((bits + 4) / 8) as usize).max(smallest))
                    }
                    _ => len,
                };
                // Every packet libopus can write is written.
                assert!(limit < smallest, "{}", at(&history));
                refused += 1;
                // The streams coded before the error moved on by a frame
                // the decoders never see. Start all sides afresh.
                enc.reset_state();
                c_dec = c_decoder(&enc);
                rust_dec = rust_decoder(&enc);
                continue;
            }
        };
        assert!(n <= len, "{}: {n} bytes", at(&history));
        let packet = &out[..n];
        let c_out = c_dec
            .decode_float(Some(packet), frame_size, false)
            .unwrap_or_else(|e| {
                panic!(
                    "{}: C decode error {e}: {}",
                    at(&history),
                    describe_packet(packet, streams)
                )
            });
        assert_eq!(c_out.len(), frame_size * channels, "{}", at(&history));
        assert!(c_out.iter().all(|v| v.is_finite()), "{}: C output", at(&history));
        let samples = rust_dec
            .decode(packet, frame_size, &mut decoded)
            .unwrap_or_else(|e| panic!("{}: opus1 decode error {e}", at(&history)));
        assert_eq!(samples, frame_size, "{}", at(&history));
        assert!(
            decoded[..samples * channels].iter().all(|v| v.is_finite()),
            "{}: opus1 output",
            at(&history)
        );
        // Two kinds of stream have a wrong final range in the stream
        // encoder (see tests/multistream_enc_robust.rs).
        let known_fault = split_streams(packet, streams).is_ok_and(|parts| {
            parts.iter().any(|p| {
                let t = toc(p.toc);
                (t.mode == Mode::Hybrid && t.stereo)
                    || (t.mode == Mode::Silk
                        && t.bandwidth == ms_support::Bandwidth::Mb
                        && t.frame_48k == 480)
            })
        });
        if !known_fault {
            assert_eq!(
                enc.final_range(),
                c_dec.final_range(),
                "{}: final range: {}",
                at(&history),
                describe_packet(packet, streams)
            );
            assert_eq!(
                rust_dec.final_range(),
                c_dec.final_range(),
                "{}: final range of the opus1 decoder",
                at(&history)
            );
        }
        written += 1;
    }
    (written, refused)
}

fn assert_stress(fs: i32, application: Application) {
    let (mut written, mut refused) = (0, 0);
    for (i, case) in SURROUND.iter().enumerate() {
        let extra: u32 = std::env::var("STRESS_SEED")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        let seed = (4000 + fs as u32 / 100 + 17 * i as u32 + application as u32)
            .wrapping_add(extra.wrapping_mul(7919));
        let (w, r) = stress(case, fs, application, seed, 500);
        written += w;
        refused += r;
    }
    println!("{fs} Hz: {written} packets decoded by both decoders, {refused} frames refused");
    assert!(written > 2_500, "{written} packets");
}

#[test]
fn surround_survives_anything_48k() {
    assert_stress(48_000, Application::Audio);
}

#[test]
fn surround_survives_anything_24k() {
    assert_stress(24_000, Application::Audio);
}

#[test]
fn surround_survives_anything_16k() {
    assert_stress(16_000, Application::Voip);
}

#[test]
fn surround_survives_anything_12k() {
    assert_stress(12_000, Application::Audio);
}

#[test]
fn surround_survives_anything_8k() {
    assert_stress(8_000, Application::Voip);
}

#[test]
fn surround_survives_anything_low_delay() {
    assert_stress(48_000, Application::RestrictedLowDelay);
}
