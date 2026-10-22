//! The encoder and decoder controls that opus1 0.2 added, against libopus
//! 1.6.1: phase inversion disabling, the decoder's output gain and pitch,
//! the 24-bit integer calls, the application change, the voice ratio, the
//! bandwidth getters of both and the decoder's `nb_samples`.
//!
//! Every test runs the same input through opus1 and through libopus with
//! the same settings. Decoded samples must be bit-identical, getters must
//! return the same values, and a setter must refuse exactly the values
//! libopus refuses. Packets are compared byte for byte where opus1's
//! encoder is libopus's byte for byte: CELT-only frames of up to 20 ms
//! (see docs/10-differences-from-libopus.md). Elsewhere a test compares
//! the mode of each packet.
// Without `analysis`, the tests that compare encoded packets are left out.
#![cfg_attr(not(feature = "analysis"), allow(dead_code, unused_imports))]

mod core_common;
mod k24_common;

use std::num::NonZeroU32;
use audiopus_sys as sys;
use core_common::*;
use k24_common::{multitone, speech};
use opus1::{
    Application, Bandwidth, Bitrate, Channels, ErrorKind, ForcedMode, OpusDecoder, OpusEncoder, SampleRate,
};

/// Stereo music whose right channel is mostly the inverted left channel.
/// CELT codes such bands as intensity stereo with phase inversion, so
/// the setting changes the packets.
fn anti_phase_stereo(rate: usize, n: usize) -> Vec<f32> {
    let x = multitone(rate, n);
    let y = multitone(rate, n + 97);
    let mut out = Vec::with_capacity(2 * n);
    for i in 0..n {
        out.push(x[i]);
        out.push(-0.8 * x[i] + 0.1 * y[i + 97]);
    }
    out
}

fn same_bytes(a: &[u8], b: &[u8], label: &str) {
    if a != b {
        let i = a.iter().zip(b).position(|(x, y)| x != y).unwrap_or(a.len().min(b.len()));
        panic!("{label}: {} bytes in opus1, {} in libopus, first difference at {i}", a.len(), b.len());
    }
}

fn same_f32(a: &[f32], b: &[f32], label: &str) {
    assert_eq!(a.len(), b.len(), "{label}: length");
    if let Some(i) = a.iter().zip(b).position(|(x, y)| x.to_bits() != y.to_bits()) {
        panic!("{label}: sample {i} is {} in opus1 and {} in libopus", a[i], b[i]);
    }
}

fn rust_mode(mode: i32) -> ForcedMode {
    match mode {
        MODE_SILK => ForcedMode::SilkOnly,
        MODE_HYBRID => ForcedMode::Hybrid,
        _ => ForcedMode::CeltOnly,
    }
}

/// libopus packets of 20 ms frames that change mode every 6 frames:
/// SILK, Hybrid, CELT, SILK, CELT, Hybrid. Below 24 kHz Hybrid is
/// replaced by SILK. The mode changes give transition frames and
/// redundancy frames.
fn mixed_stream(rate: usize, channels: usize, pcm: &[f32]) -> Vec<Vec<u8>> {
    let frame = rate / 50;
    let mut c = CEncoder::new(rate, channels, APP_AUDIO);
    c.bitrate(if channels == 2 { 48_000 } else { 32_000 });
    let modes = [MODE_SILK, MODE_HYBRID, MODE_CELT, MODE_SILK, MODE_CELT, MODE_HYBRID];
    pcm.chunks_exact(frame * channels)
        .enumerate()
        .map(|(i, f)| {
            let mut mode = modes[(i / 6) % modes.len()];
            if mode == MODE_HYBRID && rate < 24_000 {
                mode = MODE_SILK;
            }
            let bw = match mode {
                MODE_SILK => BW_WB,
                MODE_HYBRID => BW_SWB,
                _ => BW_FB,
            };
            c.force_mode(mode).bandwidth(bw);
            c.encode(f)
        })
        .collect()
}

/// One decoder call: a packet, a lost packet, or a lost packet rebuilt
/// from the forward error correction (FEC) data of the next one.
enum Step<'a> {
    Packet(&'a [u8]),
    Lost,
    Fec(&'a [u8]),
}

/// The packets with losses: every ninth packet from the fifth is rebuilt
/// with FEC, every ninth from the eighth is concealed.
fn steps(packets: &[Vec<u8>]) -> Vec<Step<'_>> {
    (0..packets.len())
        .map(|i| match i % 9 {
            4 if i + 1 < packets.len() => Step::Fec(&packets[i + 1]),
            7 => Step::Lost,
            _ => Step::Packet(&packets[i]),
        })
        .collect()
}

/// Runs both decoders over `steps` with float output and compares every
/// call. `check` runs after each call. The samples must be equal only on
/// 64-bit ARM: elsewhere CELT and Hybrid output is not bit-exact with
/// libopus (docs/10, Platforms), so only the lengths are compared there.
fn decode_both(
    rust: &mut OpusDecoder,
    c: &mut CDecoder,
    steps: &[Step<'_>],
    frame: usize,
    label: &str,
    mut check: impl FnMut(&mut OpusDecoder, &mut CDecoder, &str),
) {
    let channels = c.channels;
    let mut out = vec![0f32; 5760 * channels];
    for (i, step) in steps.iter().enumerate() {
        let label = format!("{label}, call {i}");
        let (n, expected) = match step {
            Step::Packet(p) => (rust.decode(p, 5760, &mut out), c.decode(Some(p), 5760, false)),
            Step::Lost => (rust.decode(&[], frame, &mut out), c.decode(None, frame, false)),
            Step::Fec(p) => (rust.decode_fec(p, frame, &mut out), c.decode(Some(p), frame, true)),
        };
        let expected = expected.unwrap_or_else(|e| panic!("{label}: libopus error {e}"));
        let n = n.unwrap_or_else(|e| panic!("{label}: {e}"));
        if cfg!(target_arch = "aarch64") {
            same_f32(&out[..n * channels], &expected, &label);
        } else {
            assert_eq!(n * channels, expected.len(), "{label}: length");
        }
        check(rust, c, &label);
    }
}

// --- Phase inversion ---

/// CELT-only stereo at bitrates that use intensity stereo, VBR and CBR:
/// the packets of opus1 equal libopus's with and without phase inversion,
/// and disabling it changes the packets.
// libopus runs its speech and music analysis at the default complexity.
#[cfg(feature = "analysis")]
#[test]
#[cfg_attr(not(target_arch = "aarch64"), ignore = "bit-exact with libopus only on 64-bit ARM (docs/10, Platforms)")]
fn phase_inversion_disabled_encodes_as_libopus() {
    let rate = 48_000;
    let pcm = anti_phase_stereo(rate, rate);
    let mode = MODE_CELT;
    for vbr in [true, false] {
        for bitrate in [12_000, 24_000, 32_000, 48_000, 64_000] {
            for frame in [rate / 100, rate / 50] {
                let mut packets_by_setting = Vec::new();
                for disabled in [false, true] {
                    let label = format!("VBR {vbr}, {bitrate} b/s, frame {frame}, disabled {disabled}");
                    let mut c = CEncoder::new(rate, 2, APP_AUDIO);
                    c.bitrate(bitrate).force_mode(mode).vbr(vbr);
                    c.set(sys::OPUS_SET_PHASE_INVERSION_DISABLED_REQUEST, disabled as i32);
                    let mut rust = OpusEncoder::new(
                        SampleRate::try_from(rate as i32).unwrap(),
                        Channels::Stereo,
                        Application::Audio,
                    )
                    .unwrap();
                    rust.set_bitrate(Bitrate::BitsPerSecond(NonZeroU32::new(bitrate as u32).unwrap()));
                    rust.set_vbr(vbr);
                    rust.set_force_mode(Some(rust_mode(mode)));
                    rust.set_phase_inversion_disabled(disabled);
                    assert_eq!(
                        rust.phase_inversion_disabled() as i32,
                        c.get(sys::OPUS_GET_PHASE_INVERSION_DISABLED_REQUEST),
                        "{label}: getter"
                    );
                    let mut out = vec![0u8; 1500];
                    let mut packets = Vec::new();
                    for (i, f) in pcm.chunks_exact(2 * frame).enumerate() {
                        let expected = c.encode(f);
                        let n = rust.encode(f, frame, &mut out).unwrap();
                        same_bytes(&out[..n], &expected, &format!("{label}: packet {i}"));
                        assert_eq!(rust.final_range(), c.final_range(), "{label}: range {i}");
                        packets.push(expected);
                    }
                    // A reset keeps the setting, in both.
                    rust.reset_state();
                    c.reset();
                    assert_eq!(
                        rust.phase_inversion_disabled() as i32,
                        c.get(sys::OPUS_GET_PHASE_INVERSION_DISABLED_REQUEST),
                        "{label}: getter after reset"
                    );
                    packets_by_setting.push(packets);
                }
                // At 12 kb/s the stream is coded as mono; from 48 kb/s on
                // fewer bands are intensity stereo.
                if (24_000..=32_000).contains(&bitrate) {
                    assert!(
                        packets_by_setting[0] != packets_by_setting[1],
                        "VBR {vbr}, {bitrate} b/s, frame {frame}: the setting changed nothing"
                    );
                }
            }
        }
    }
}

/// Stereo and mono decoders, with phase inversion disabled or not, decode
/// a CELT stream that uses phase inversion to libopus's samples. The
/// defaults are libopus's: on for a mono decoder, off for a stereo one.
#[test]
#[cfg_attr(not(target_arch = "aarch64"), ignore = "bit-exact with libopus only on 64-bit ARM (docs/10, Platforms)")]
fn phase_inversion_disabled_decodes_as_libopus() {
    let rate = 48_000;
    let frame = rate / 50;
    let pcm = anti_phase_stereo(rate, rate);
    let mut c_enc = CEncoder::new(rate, 2, APP_AUDIO);
    c_enc.bitrate(24_000).force_mode(MODE_CELT);
    let packets = c_enc.encode_all(&pcm, frame);
    let steps = steps(&packets);
    for channels in [1, 2] {
        let mut outputs = Vec::new();
        for setting in [None, Some(false), Some(true)] {
            let label = format!("{channels} channels, setting {setting:?}");
            let mut rust = OpusDecoder::new(
                SampleRate::try_from(rate as i32).unwrap(),
                Channels::try_from(channels).unwrap(),
            )
            .unwrap();
            let mut c = CDecoder::new(rate, channels);
            let get = sys::OPUS_GET_PHASE_INVERSION_DISABLED_REQUEST;
            assert_eq!(rust.phase_inversion_disabled() as i32, c.get(get), "{label}: default");
            assert_eq!(rust.phase_inversion_disabled(), channels == 1, "{label}: default");
            if let Some(disabled) = setting {
                rust.set_phase_inversion_disabled(disabled);
                let ret = c.try_set(sys::OPUS_SET_PHASE_INVERSION_DISABLED_REQUEST, disabled as i32);
                assert_eq!(ret, 0, "{label}");
            }
            let mut all = Vec::new();
            let mut out = vec![0f32; 5760 * channels];
            decode_both(&mut rust, &mut c, &steps, frame, &label, |r, c, label| {
                assert_eq!(r.phase_inversion_disabled() as i32, c.get(get), "{label}: getter");
            });
            for p in &packets[..20] {
                let n = rust.decode(p, 5760, &mut out).unwrap();
                all.extend_from_slice(&out[..n * channels]);
            }
            rust.reset_state();
            c.reset();
            assert_eq!(rust.phase_inversion_disabled() as i32, c.get(get), "{label}: after reset");
            outputs.push(all);
        }
        // For two channels, disabling the inversion changes the output.
        if channels == 2 {
            assert_ne!(outputs[1], outputs[2], "the setting changed nothing");
        }
    }
    // libopus takes only 0 and 1. The Rust setter takes a `bool`, so no
    // other value can reach it.
    let mut c = CDecoder::new(rate, 2);
    for v in [-1, 2, i32::MAX] {
        assert_eq!(c.try_set(sys::OPUS_SET_PHASE_INVERSION_DISABLED_REQUEST, v), sys::OPUS_BAD_ARG);
    }
}

// --- Output gain ---

const GAINS: [i16; 13] = [-32768, -32767, -12000, -1536, -256, -1, 0, 1, 255, 1536, 6000, 20000, 32767];

/// The gain over SILK, Hybrid and CELT frames with mode changes, losses
/// and FEC, with float, 16-bit and 24-bit output. Out-of-range values are
/// refused by both, and the setting stays.
#[test]
#[cfg_attr(not(target_arch = "aarch64"), ignore = "bit-exact with libopus only on 64-bit ARM (docs/10, Platforms)")]
fn gain_matches_libopus() {
    for (rate, channels) in [(48_000, 2), (48_000, 1), (16_000, 1), (24_000, 2)] {
        let frame = rate / 50;
        let mono = speech(rate, 4.0);
        let pcm: Vec<f32> = if channels == 2 {
            k24_common::to_stereo(&mono, rate)
        } else {
            mono
        };
        let packets = mixed_stream(rate, channels, &pcm);
        let steps = steps(&packets);
        for gain in GAINS {
            let label = format!("{rate} Hz, {channels} ch, gain {gain}");
            let mut rust = OpusDecoder::new(
                SampleRate::try_from(rate as i32).unwrap(),
                Channels::try_from(channels).unwrap(),
            )
            .unwrap();
            let mut c = CDecoder::new(rate, channels);
            rust.set_gain(gain);
            assert_eq!(c.try_set(sys::OPUS_SET_GAIN_REQUEST, i32::from(gain)), 0);
            assert_eq!(i32::from(rust.gain()), c.get(sys::OPUS_GET_GAIN_REQUEST), "{label}: getter");
            decode_both(&mut rust, &mut c, &steps, frame, &label, |_, _, _| {});

            // 16-bit and 24-bit output, after a reset that keeps the gain.
            rust.reset_state();
            c.reset();
            assert_eq!(i32::from(rust.gain()), c.get(sys::OPUS_GET_GAIN_REQUEST), "{label}: after reset");
            let mut out16 = vec![0i16; 5760 * channels];
            let mut out24 = vec![0i32; 5760 * channels];
            let mut c24 = CDecoder::new(rate, channels);
            assert_eq!(c24.try_set(sys::OPUS_SET_GAIN_REQUEST, i32::from(gain)), 0);
            let mut rust24 = OpusDecoder::new(
                SampleRate::try_from(rate as i32).unwrap(),
                Channels::try_from(channels).unwrap(),
            )
            .unwrap();
            rust24.set_gain(gain);
            for (i, step) in steps.iter().enumerate() {
                let label = format!("{label}, call {i}");
                let (n16, e16, n24, e24) = match step {
                    Step::Packet(p) => (
                        rust.decode_i16(p, 5760, &mut out16),
                        c.decode_i16(Some(p), 5760, false),
                        rust24.decode24(p, 5760, &mut out24),
                        c24.decode24(Some(p), 5760, false),
                    ),
                    Step::Lost => (
                        rust.decode_i16(&[], frame, &mut out16),
                        c.decode_i16(None, frame, false),
                        rust24.decode24(&[], frame, &mut out24),
                        c24.decode24(None, frame, false),
                    ),
                    Step::Fec(p) => (
                        rust.decode_fec_i16(p, frame, &mut out16),
                        c.decode_i16(Some(p), frame, true),
                        rust24.decode_fec24(p, frame, &mut out24),
                        c24.decode24(Some(p), frame, true),
                    ),
                };
                let n16 = n16.unwrap();
                assert_eq!(&out16[..n16 * channels], &e16.unwrap()[..], "{label}: 16-bit");
                let n24 = n24.unwrap();
                assert_eq!(&out24[..n24 * channels], &e24.unwrap()[..], "{label}: 24-bit");
            }
        }
    }
    let mut rust = OpusDecoder::new(SampleRate::Hz48000, Channels::Stereo).unwrap();
    let mut c = CDecoder::new(48_000, 2);
    rust.set_gain(1000);
    assert_eq!(c.try_set(sys::OPUS_SET_GAIN_REQUEST, 1000), 0);
    // libopus refuses the values outside the range of an `i16`, which
    // opus1's `set_gain` cannot be given.
    for v in [i32::MIN, -32769, 32768, i32::MAX] {
        assert_eq!(c.try_set(sys::OPUS_SET_GAIN_REQUEST, v), sys::OPUS_BAD_ARG, "gain {v}");
        assert_eq!(i32::from(rust.gain()), c.get(sys::OPUS_GET_GAIN_REQUEST), "gain {v}: kept");
    }
}

// --- Pitch ---

/// `pitch` after every decoder call, over SILK, Hybrid and CELT frames,
/// losses, FEC and resets, equals libopus's `OPUS_GET_PITCH`. Both the SILK
/// lag and the CELT post-filter period show up with nonzero values.
#[test]
#[cfg_attr(not(target_arch = "aarch64"), ignore = "bit-exact with libopus only on 64-bit ARM (docs/10, Platforms)")]
fn pitch_matches_libopus() {
    for (rate, channels) in [(48_000, 1), (48_000, 2), (16_000, 1), (8_000, 2), (12_000, 1)] {
        let frame = rate / 50;
        let mono = speech(rate, 6.0);
        let pcm: Vec<f32> = if channels == 2 {
            k24_common::to_stereo(&mono, rate)
        } else {
            mono
        };
        let packets = mixed_stream(rate, channels, &pcm);
        let steps = steps(&packets);
        let label = format!("{rate} Hz, {channels} ch");
        let mut rust = OpusDecoder::new(
            SampleRate::try_from(rate as i32).unwrap(),
            Channels::try_from(channels).unwrap(),
        )
        .unwrap();
        let mut c = CDecoder::new(rate, channels);
        assert_eq!(rust.pitch(), c.get(sys::OPUS_GET_PITCH_REQUEST), "{label}: new");
        let mut seen = Vec::new();
        for half in [&steps[..steps.len() / 2], &steps[steps.len() / 2..]] {
            decode_both(&mut rust, &mut c, half, frame, &label, |r, c, label| {
                let p = c.get(sys::OPUS_GET_PITCH_REQUEST);
                assert_eq!(r.pitch(), p, "{label}: pitch");
                seen.push(p);
            });
            // libopus keeps the SILK pitch over a reset.
            rust.reset_state();
            c.reset();
            assert_eq!(rust.pitch(), c.get(sys::OPUS_GET_PITCH_REQUEST), "{label}: reset");
        }
        assert!(seen.iter().filter(|&&p| p != 0).count() > 20, "{label}: few pitch values: {seen:?}");
    }
    // A CELT-only stream: the post-filter period, 0 when it is off, and the
    // values after concealment.
    let rate = 48_000;
    let frame = rate / 50;
    let pcm = speech(rate, 4.0);
    let mut c_enc = CEncoder::new(rate, 1, APP_AUDIO);
    c_enc.bitrate(32_000).force_mode(MODE_CELT);
    let packets = c_enc.encode_all(&pcm, frame);
    let mut rust =
        OpusDecoder::new(SampleRate::try_from(rate as i32).unwrap(), Channels::Mono).unwrap();
    let mut c = CDecoder::new(rate, 1);
    let mut celt_seen = Vec::new();
    decode_both(&mut rust, &mut c, &steps(&packets), frame, "CELT", |r, c, label| {
        let p = c.get(sys::OPUS_GET_PITCH_REQUEST);
        assert_eq!(r.pitch(), p, "{label}: pitch");
        celt_seen.push(p);
    });
    assert!(celt_seen.contains(&0), "no frame without the post-filter");
    assert!(celt_seen.iter().any(|&p| p > 15), "no post-filter period");
}

// --- 24-bit calls ---

/// A 24-bit test signal: the music scaled to 2^23 with odd values, and
/// stretches at the 24-bit limits and beyond them, up to the 32-bit limits.
fn pcm24(rate: usize, channels: usize, seconds: f64) -> Vec<i32> {
    let n = (rate as f64 * seconds) as usize;
    let x = multitone(rate, n * channels);
    let mut seed = 0x2468_ace1u32;
    x.iter()
        .enumerate()
        .map(|(i, &v)| {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            let block = i / (rate / 10);
            match block % 8 {
                // At and just beyond the 24-bit range.
                3 => [8_388_607, -8_388_608, 8_388_608, -8_388_609][i % 4],
                // Far beyond it.
                5 => [i32::MAX, i32::MIN, 1 << 28, -(1 << 26) - 1][i % 4],
                _ => ((v * 8_388_608.0 * 1.4) as i32) | 1 | (seed >> 31) as i32,
            }
        })
        .collect()
}

/// `encode24` gives libopus's packets for VoIP and Audio, mono and stereo,
/// at several rates, with input at and beyond the 24-bit limits: CELT-only
/// frames of 10 and 20 ms with VBR, 40 and 60 ms with CBR (opus1 codes
/// longer VBR packets differently on purpose). `decode24` of those packets
/// gives libopus's samples, values beyond 24 bits included.
// libopus runs its speech and music analysis at the default complexity.
#[cfg(feature = "analysis")]
#[test]
#[cfg_attr(not(target_arch = "aarch64"), ignore = "bit-exact with libopus only on 64-bit ARM (docs/10, Platforms)")]
fn encode24_and_decode24_match_libopus() {
    for (rate, channels, app) in [
        (48_000, 2, APP_AUDIO),
        (48_000, 1, APP_VOIP),
        (16_000, 1, APP_VOIP),
        (24_000, 2, APP_AUDIO),
        (8_000, 1, APP_AUDIO),
    ] {
        let application = if app == APP_VOIP { Application::Voip } else { Application::Audio };
        let pcm = pcm24(rate, channels, 2.0);
        for (frame_ms10, vbr) in [(100usize, true), (200, true), (400, false), (600, false)] {
            let frame = rate * frame_ms10 / 10_000;
            let label = format!("{rate} Hz, {channels} ch, app {app}, {frame} samples");
            let mut c = CEncoder::new(rate, channels, app);
            c.force_mode(MODE_CELT).vbr(vbr);
            let mut rust = OpusEncoder::new(
                SampleRate::try_from(rate as i32).unwrap(),
                Channels::try_from(channels).unwrap(),
                application,
            )
            .unwrap();
            rust.set_force_mode(Some(ForcedMode::CeltOnly));
            rust.set_vbr(vbr);
            let mut out = vec![0u8; 1500];
            let mut packets = Vec::new();
            for (i, f) in pcm.chunks_exact(frame * channels).enumerate() {
                let expected = c.try_encode24(f, frame, 1500).unwrap();
                let n = rust.encode24(f, frame, &mut out).unwrap();
                same_bytes(&out[..n], &expected, &format!("{label}: packet {i}"));
                packets.push(expected);
            }
            let mut rust_dec = OpusDecoder::new(
                SampleRate::try_from(rate as i32).unwrap(),
                Channels::try_from(channels).unwrap(),
            )
            .unwrap();
            let mut c_dec = CDecoder::new(rate, channels);
            let mut out24 = vec![0i32; 5760 * channels];
            let mut beyond = false;
            for (i, p) in packets.iter().enumerate() {
                let expected = c_dec.decode24(Some(p), 5760, false).unwrap();
                let n = rust_dec.decode24(p, 5760, &mut out24).unwrap();
                assert!(out24[..n * channels] == expected[..], "{label}: decoded {i}");
                beyond |= expected.iter().any(|v| v.unsigned_abs() > 8_388_608);
            }
            assert!(beyond, "{label}: no decoded value beyond 24 bits");
        }
    }
}

/// In every mode and frame size, `encode24` codes what `encode` codes for
/// the same samples divided by 2^23 (libopus `INT24TORES`), as libopus's
/// float build does. The input stays within 24 bits, where the speech and
/// music analysis reads both inputs alike.
#[test]
fn encode24_codes_as_encode_of_the_scaled_samples() {
    for (rate, channels, application) in [
        (48_000, 2, Application::Audio),
        (48_000, 1, Application::Voip),
        (16_000, 1, Application::Voip),
        (24_000, 2, Application::Audio),
        (12_000, 2, Application::Voip),
    ] {
        let x = speech(rate, 2.0);
        let pcm: Vec<i32> = (0..x.len() * channels)
            .map(|i| ((x[i / channels] * (6_000_000 + 1_000_000 * (i % channels)) as f32) as i32) | 1)
            .collect();
        let scaled: Vec<f32> = pcm.iter().map(|&v| v as f32 * (1.0 / 8_388_608.0)).collect();
        for frame_ms10 in [100usize, 200, 400, 600] {
            let frame = rate * frame_ms10 / 10_000;
            let label = format!("{rate} Hz, {channels} ch, {application:?}, {frame} samples");
            let mut a = OpusEncoder::new(
                SampleRate::try_from(rate as i32).unwrap(),
                Channels::try_from(channels).unwrap(),
                application,
            )
            .unwrap();
            let mut b = OpusEncoder::new(
                SampleRate::try_from(rate as i32).unwrap(),
                Channels::try_from(channels).unwrap(),
                application,
            )
            .unwrap();
            let (mut out_a, mut out_b) = (vec![0u8; 1500], vec![0u8; 1500]);
            let chunks = pcm.chunks_exact(frame * channels).zip(scaled.chunks_exact(frame * channels));
            for (i, (p24, pf)) in chunks.enumerate() {
                let na = a.encode24(p24, frame, &mut out_a).unwrap();
                let nb = b.encode(pf, frame, &mut out_b).unwrap();
                same_bytes(&out_a[..na], &out_b[..nb], &format!("{label}: packet {i}"));
            }
        }
    }
}

/// The errors of the 24-bit calls: a frame size libopus refuses, a frame
/// size of 0 and an invalid packet.
#[test]
fn errors_of_the_24_bit_calls_follow_libopus() {
    let pcm = vec![1000i32; 2 * 5760];
    let mut c = CEncoder::new(48_000, 2, APP_AUDIO);
    let mut rust =
        OpusEncoder::new(SampleRate::Hz48000, Channels::Stereo, Application::Audio).unwrap();
    let mut out = vec![0u8; 1500];
    for frame in [0usize, 1, 119, 121, 480, 959, 2880, 5760] {
        let c_ok = c.try_encode24(&pcm, frame, 1500).is_ok();
        assert_eq!(rust.encode24(&pcm, frame, &mut out).is_ok(), c_ok, "frame {frame}");
    }
    let mut c = CDecoder::new(48_000, 2);
    let mut rust = OpusDecoder::new(SampleRate::Hz48000, Channels::Stereo).unwrap();
    let mut out24 = vec![0i32; 2 * 5760];
    assert!(c.decode24(None, 0, false).is_err());
    assert!(rust.decode24(&[], 0, &mut out24).is_err());
    let bad = [0xff, 0xff, 0xff];
    assert!(c.decode24(Some(&bad), 960, false).is_err());
    assert!(rust.decode24(&bad, 960, &mut out24).is_err());
}

// --- Application change and voice ratio ---

/// `set_application` before the first frame gives libopus's packets, also
/// a change to and from `RestrictedLowDelay`. After the first frame only
/// the current application is accepted; after a reset any again.
// libopus runs its speech and music analysis at the default complexity.
#[cfg(feature = "analysis")]
#[test]
#[cfg_attr(not(target_arch = "aarch64"), ignore = "bit-exact with libopus only on 64-bit ARM (docs/10, Platforms)")]
fn set_application_follows_libopus() {
    let apps = [
        (APP_VOIP, Application::Voip),
        (APP_AUDIO, Application::Audio),
        (APP_LOWDELAY, Application::RestrictedLowDelay),
    ];
    let rate = 48_000;
    let frame = rate / 50;
    let pcm = k24_common::to_stereo(&speech(rate, 1.0), rate);
    for (c_from, from) in apps {
        for (c_to, to) in apps {
            let label = format!("{from:?} to {to:?}");
            let mut c = CEncoder::new(rate, 2, c_from);
            let mut rust = OpusEncoder::new(
                SampleRate::try_from(rate as i32).unwrap(),
                Channels::Stereo,
                from,
            )
            .unwrap();
            c.set(sys::OPUS_SET_APPLICATION_REQUEST, c_to);
            rust.set_application(to).unwrap();
            assert_eq!(rust.application() as i32, c.get(sys::OPUS_GET_APPLICATION_REQUEST), "{label}");
            assert_eq!(rust.lookahead() as i32, c.lookahead(), "{label}: lookahead");
            let mut out = vec![0u8; 1500];
            for (i, f) in pcm.chunks_exact(2 * frame).take(20).enumerate() {
                let expected = c.encode(f);
                let n = rust.encode(f, frame, &mut out).unwrap();
                same_bytes(&out[..n], &expected, &format!("{label}: packet {i}"));
            }
            for (c_other, other) in apps {
                let c_ok = c.try_set(sys::OPUS_SET_APPLICATION_REQUEST, c_other) == 0;
                assert_eq!(rust.set_application(other).is_ok(), c_ok, "{label}, then {other:?}");
                assert_eq!(c_ok, other == to, "{label}, then {other:?}");
            }
            rust.reset_state();
            c.reset();
            let c_ok = c.try_set(sys::OPUS_SET_APPLICATION_REQUEST, c_from) == 0;
            assert!(c_ok);
            rust.set_application(from).unwrap();
        }
    }
}

/// opus1's voice ratio for the libopus value `v` (-1 is `None`), or `None`
/// for a value that `set_voice_ratio` cannot be given.
fn rust_ratio(v: i32) -> Option<Option<u8>> {
    if v == -1 { Some(None) } else { u8::try_from(v).ok().map(Some) }
}

/// The libopus value of opus1's voice ratio.
fn c_ratio(ratio: Option<u8>) -> i32 {
    ratio.map_or(-1, i32::from)
}

/// The voice ratio: the values libopus takes, the value after each frame
/// (set by the caller, kept over silent frames, or replaced by the
/// analysis), and the configuration of each packet.
// libopus runs its speech and music analysis at the default complexity.
#[cfg(feature = "analysis")]
#[test]
fn voice_ratio_follows_libopus() {
    let rate = 48_000;
    let frame = rate / 50;
    let mut c = CEncoder::new(rate, 1, APP_AUDIO);
    let mut rust = OpusEncoder::new(
        SampleRate::try_from(rate as i32).unwrap(),
        Channels::Mono,
        Application::Audio,
    )
    .unwrap();
    let get = OPUS_GET_VOICE_RATIO_REQUEST;
    assert_eq!(c_ratio(rust.voice_ratio()), c.get(get));
    for v in [i32::MIN, -2, -1, 0, 1, 50, 99, 100, 101, 255, i32::MAX] {
        let c_ok = c.try_set(OPUS_SET_VOICE_RATIO_REQUEST, v) == 0;
        match rust_ratio(v) {
            Some(ratio) => assert_eq!(rust.set_voice_ratio(ratio).is_ok(), c_ok, "ratio {v}"),
            None => assert!(!c_ok, "ratio {v}"),
        }
        assert_eq!(c_ratio(rust.voice_ratio()), c.get(get), "ratio {v}");
    }
    // Speech, silence and music, with a voice ratio set before every fifth
    // frame.
    let mut pcm = speech(rate, 1.0);
    pcm.extend(std::iter::repeat_n(0.0, rate));
    pcm.extend(multitone(rate, rate));
    c.bitrate(20_000);
    rust.set_bitrate(Bitrate::BitsPerSecond(NonZeroU32::new(20_000).unwrap()));
    let mut out = vec![0u8; 1500];
    for (i, f) in pcm.chunks_exact(frame).enumerate() {
        if i % 5 == 0 {
            let v = [0, 100, 30, -1][(i / 5) % 4];
            c.set(OPUS_SET_VOICE_RATIO_REQUEST, v);
            rust.set_voice_ratio(rust_ratio(v).unwrap()).unwrap();
        }
        let expected = c.encode(f);
        let n = rust.encode(f, frame, &mut out).unwrap();
        // SILK packets are not libopus's byte for byte; the mode, the
        // bandwidth and the frame size in the first byte are.
        assert_eq!(out[0] >> 3, expected[0] >> 3, "configuration of packet {i}");
        assert!(n > 0);
        assert_eq!(c_ratio(rust.voice_ratio()), c.get(get), "ratio after packet {i}");
    }
    rust.reset_state();
    c.reset();
    assert_eq!(c_ratio(rust.voice_ratio()), c.get(get), "after reset");
}

// --- Bandwidth getters and the packet duration ---

/// The libopus number of the decoder's bandwidth: 0 for `None`.
fn c_bandwidth(bandwidth: Option<Bandwidth>) -> i32 {
    bandwidth.map_or(0, Bandwidth::to_raw)
}

/// `OpusEncoder::bandwidth` (the bandwidth of the last packet) and
/// `max_bandwidth` equal libopus's `OPUS_GET_BANDWIDTH` and
/// `OPUS_GET_MAX_BANDWIDTH`: before the first packet, after every packet
/// while the bitrate, the bandwidth setting and the maximum bandwidth
/// change, for packets too small to code audio, and after a reset.
// libopus runs its speech and music analysis at the default complexity.
#[cfg(feature = "analysis")]
#[test]
#[cfg_attr(not(target_arch = "aarch64"), ignore = "bit-exact with libopus only on 64-bit ARM (docs/10, Platforms)")]
fn encoder_bandwidth_matches_libopus() {
    let configs = [
        (48_000, 1, APP_AUDIO, Application::Audio),
        (48_000, 2, APP_VOIP, Application::Voip),
        (24_000, 2, APP_AUDIO, Application::Audio),
        (16_000, 1, APP_VOIP, Application::Voip),
        (8_000, 1, APP_VOIP, Application::Voip),
        (48_000, 1, APP_LOWDELAY, Application::RestrictedLowDelay),
    ];
    let bitrates = [6_000, 10_000, 16_000, 24_000, 40_000, 96_000];
    let settings = [None, Some(Bandwidth::Wideband), None, Some(Bandwidth::Fullband), Some(Bandwidth::Narrowband)];
    let maxima = [Bandwidth::Fullband, Bandwidth::Superwideband, Bandwidth::Mediumband, Bandwidth::Fullband];
    for (rate, channels, c_app, app) in configs {
        let frame = rate / 50;
        let mono = speech(rate, 4.0);
        let mut pcm: Vec<f32> = if channels == 2 { k24_common::to_stereo(&mono, rate) } else { mono };
        pcm.extend(multitone(rate, 2 * rate * channels));
        let label = format!("{rate} Hz, {channels} ch, {app:?}");
        let mut c = CEncoder::new(rate, channels, c_app);
        let mut rust = OpusEncoder::new(
            SampleRate::try_from(rate as i32).unwrap(),
            Channels::try_from(channels).unwrap(),
            app,
        )
        .unwrap();
        let check = |rust: &OpusEncoder, c: &mut CEncoder, label: &str| {
            assert_eq!(rust.bandwidth().to_raw(), c.get(sys::OPUS_GET_BANDWIDTH_REQUEST), "{label}: bandwidth");
            assert_eq!(rust.max_bandwidth().to_raw(), c.get(sys::OPUS_GET_MAX_BANDWIDTH_REQUEST), "{label}: max");
        };
        check(&rust, &mut c, &format!("{label}, new"));
        let mut seen = Vec::new();
        let mut out = vec![0u8; 1500];
        for (i, f) in pcm.chunks_exact(frame * channels).enumerate() {
            if i % 8 == 0 {
                let step = i / 8;
                let bitrate = bitrates[step % bitrates.len()];
                c.bitrate(bitrate);
                rust.set_bitrate(Bitrate::BitsPerSecond(NonZeroU32::new(bitrate as u32).unwrap()));
                let setting = settings[(step / 2) % settings.len()];
                c.bandwidth(setting.map_or(sys::OPUS_AUTO, Bandwidth::to_raw));
                rust.set_bandwidth(setting);
                let max = maxima[(step / 3) % maxima.len()];
                c.max_bandwidth(max.to_raw());
                rust.set_max_bandwidth(max);
            }
            // Every 13th packet has room for 2 bytes only: too small to
            // code audio.
            let max_bytes = if i % 13 == 12 { 2 } else { 1500 };
            let expected = c.try_encode(f, frame, max_bytes).unwrap();
            let n = rust.encode(f, frame, &mut out[..max_bytes]).unwrap();
            let label = format!("{label}, packet {i}");
            // SILK packets are not libopus's byte for byte; the mode, the
            // bandwidth and the frame size in the first byte are.
            assert!(n > 0, "{label}");
            assert_eq!(out[0] >> 3, expected[0] >> 3, "{label}: configuration");
            check(&rust, &mut c, &label);
            seen.push(rust.bandwidth());
        }
        rust.reset_state();
        c.reset();
        check(&rust, &mut c, &format!("{label}, reset"));
        if app != Application::RestrictedLowDelay && rate == 48_000 {
            for bw in [Bandwidth::Narrowband, Bandwidth::Wideband, Bandwidth::Fullband] {
                assert!(seen.contains(&bw), "{label}: no packet of {bw:?} in {seen:?}");
            }
        }
    }
}

/// `OpusDecoder::bandwidth` equals libopus's `OPUS_GET_BANDWIDTH` (0 for
/// `None`) after every decoder call: packets of every mode and bandwidth,
/// lost packets and FEC calls, and after a reset. `nb_samples` equals
/// `opus_decoder_get_nb_samples`, also for packets that libopus refuses.
#[test]
fn decoder_bandwidth_and_nb_samples_match_libopus() {
    for (rate, channels) in [(48_000, 2), (16_000, 1), (8_000, 1)] {
        let frame = rate / 50;
        let mono = speech(rate, 3.0);
        let pcm: Vec<f32> = if channels == 2 { k24_common::to_stereo(&mono, rate) } else { mono };
        let packets = mixed_stream(rate, channels, &pcm);
        let label = format!("{rate} Hz, {channels} ch");
        let mut rust = OpusDecoder::new(
            SampleRate::try_from(rate as i32).unwrap(),
            Channels::try_from(channels).unwrap(),
        )
        .unwrap();
        let mut c = CDecoder::new(rate, channels);
        assert_eq!(rust.bandwidth(), None, "{label}: new");
        assert_eq!(c.get(sys::OPUS_GET_BANDWIDTH_REQUEST), 0, "{label}: new");
        let mut seen = Vec::new();
        decode_both(&mut rust, &mut c, &steps(&packets), frame, &label, |r, c, label| {
            assert_eq!(c_bandwidth(r.bandwidth()), c.get(sys::OPUS_GET_BANDWIDTH_REQUEST), "{label}: bandwidth");
            seen.push(r.bandwidth());
        });
        if rate == 48_000 {
            assert!(seen.iter().collect::<std::collections::HashSet<_>>().len() >= 3, "{label}: {seen:?}");
        }
        rust.reset_state();
        c.reset();
        assert_eq!(c_bandwidth(rust.bandwidth()), c.get(sys::OPUS_GET_BANDWIDTH_REQUEST), "{label}: reset");
        assert_eq!(rust.bandwidth(), None, "{label}: reset");

        // The packet durations, also of packets made up here: a code 3
        // packet without its count byte, one of 48 frames of 20 ms (more
        // than 120 ms) and an empty one.
        let mut made_up: Vec<Vec<u8>> = packets.iter().take(12).cloned().collect();
        made_up.extend([vec![0x0b], vec![0x0b, 0x02], vec![0x0b, 0x30], vec![0xfb, 0x06], vec![]]);
        for (i, p) in made_up.iter().enumerate() {
            let expected = c.nb_samples(p);
            match rust.nb_samples(p) {
                Ok(n) => assert_eq!(n as i32, expected, "{label}: packet {i}"),
                Err(e) => assert_eq!(e.to_raw(), expected, "{label}: packet {i}"),
            }
        }
        assert_eq!(rust.nb_samples(&[]).map_err(|e| e.kind()), Err(ErrorKind::BadArg));
    }
}
