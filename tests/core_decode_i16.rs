//! 16-bit output: `OpusDecoder::decode_i16` against C `opus_decode`, and
//! `pcm_soft_clip` against C `opus_pcm_soft_clip`.
//!
//! libopus (float build) decodes to floating point, soft-clips the packet
//! and rounds to 16 bits. The soft clipper only acts where the decoded
//! signal goes beyond full scale, so the tests use both ordinary speech and
//! signals that overshoot on purpose.

mod core_common;
mod k24_common;

use audiopus_sys as sys;
use core_common::*;
use k24_common::{full_scale, speech_fullband, to_stereo};
use opus1::{OpusDecoder, pcm_soft_clip};
use std::f64::consts::PI;

/// A signal that overshoots full scale in several ways:
///
/// - a loud tone whose level swells from 0.5 to 2.5 and back,
/// - a stretch with an offset, so that the signal stays above 1 without
///   crossing zero from one frame into the next,
/// - short bursts beyond 2, the limit of the soft clipper's curve.
fn overdriven(rate: usize, seconds: f64) -> Vec<f32> {
    let n = (rate as f64 * seconds) as usize;
    (0..n)
        .map(|i| {
            let t = i as f64 / rate as f64;
            let swell = 0.5 + 2.0 * (PI * t / seconds).sin().powi(2);
            let mut v = swell * (2.0 * PI * 220.0 * t).sin() + 0.2 * (2.0 * PI * 1370.0 * t).sin();
            if (0.8..1.1).contains(&t) {
                v = 1.3 + 0.2 * (2.0 * PI * 30.0 * t).sin();
            }
            if (t * 10.0) as usize % 7 == 3 {
                v *= 1.6;
            }
            v as f32
        })
        .collect()
}

struct Diff {
    samples: usize,
    /// Samples that differ by exactly 1.
    off_by_one: usize,
    /// Largest difference.
    max: i32,
    /// Samples at the limits of the 16-bit range, in the libopus output.
    at_full_scale: usize,
}

fn diff(reference: &[i16], test: &[i16]) -> Diff {
    assert_eq!(reference.len(), test.len());
    let mut d = Diff { samples: reference.len(), off_by_one: 0, max: 0, at_full_scale: 0 };
    for (&a, &b) in reference.iter().zip(test) {
        let e = (a as i32 - b as i32).abs();
        d.off_by_one += (e == 1) as usize;
        d.max = d.max.max(e);
        d.at_full_scale += (a == i16::MAX || a == i16::MIN) as usize;
    }
    d
}

/// Encode `pcm` with the C encoder and decode the packets to 16 bits with
/// both decoders.
fn decode_both(rate: usize, channels: usize, mode: i32, bandwidth: i32, bitrate: i32, frame: usize, pcm: &[f32]) -> (Vec<i16>, Vec<i16>) {
    let mut enc = CEncoder::new(rate, channels, APP_AUDIO);
    enc.force_mode(mode).bandwidth(bandwidth).bitrate(bitrate);
    let packets = enc.encode_all(pcm, frame);
    let mut c_dec = CDecoder::new(rate, channels);
    let mut dec = OpusDecoder::new(rate as i32, channels).unwrap();
    let (mut c, mut rs) = (Vec::new(), Vec::new());
    let max = rate * 120 / 1000;
    let mut buf = vec![0i16; max * channels];
    for p in &packets {
        c.extend(c_dec.decode_i16(Some(p), max, false).unwrap());
        let n = dec.decode_i16(p, max, &mut buf).unwrap();
        rs.extend_from_slice(&buf[..n * channels]);
    }
    (c, rs)
}

/// Rates, modes and frame durations for the comparisons: (name, rate,
/// channels, mode, bandwidth, bitrate, frame duration in units of 2.5 ms).
const STREAMS: [(&str, usize, usize, i32, i32, i32, usize); 9] = [
    ("SILK-NB 8k mono 20ms", 8000, 1, MODE_SILK, BW_NB, 16000, 8),
    ("SILK-WB 16k stereo 20ms", 16000, 2, MODE_SILK, BW_WB, 48000, 8),
    ("SILK-WB 48k mono 60ms", 48000, 1, MODE_SILK, BW_WB, 32000, 24),
    ("Hybrid-SWB 24k mono 20ms", 24000, 1, MODE_HYBRID, BW_SWB, 40000, 8),
    ("Hybrid-FB 48k stereo 10ms", 48000, 2, MODE_HYBRID, BW_FB, 96000, 4),
    ("CELT-FB 48k mono 20ms", 48000, 1, MODE_CELT, BW_FB, 96000, 8),
    ("CELT-FB 48k stereo 2.5ms", 48000, 2, MODE_CELT, BW_FB, 256000, 1),
    ("CELT-SWB 24k stereo 60ms", 24000, 2, MODE_CELT, BW_SWB, 128000, 24),
    ("CELT-WB 16k mono 40ms", 16000, 1, MODE_CELT, BW_WB, 64000, 16),
];

/// Whether the 16-bit output must be libopus's in every sample: on 64-bit
/// ARM, where the float output and the rounding are libopus's. Elsewhere a
/// sample may be one step away.
const EXACT: bool = cfg!(target_arch = "aarch64");

/// Ordinary speech: every sample is libopus's (on other targets than
/// 64-bit ARM, or one step away).
#[test]
fn speech_decodes_to_the_samples_of_opus_decode() {
    for (name, rate, channels, mode, bandwidth, bitrate, frame_2_5) in STREAMS {
        let mono = speech_fullband(rate, 2.0);
        let pcm = if channels == 2 { to_stereo(&mono, rate) } else { mono };
        let (c, rs) = decode_both(rate, channels, mode, bandwidth, bitrate, rate / 400 * frame_2_5, &pcm);
        let d = diff(&c, &rs);
        println!("{name}: {} samples, {} differ by 1, largest difference {}", d.samples, d.off_by_one, d.max);
        assert!(d.max <= if EXACT { 0 } else { 1 }, "{name}: largest difference {}", d.max);
        assert!(d.off_by_one * 1000 < d.samples, "{name}: {} of {} samples differ", d.off_by_one, d.samples);
    }
}

/// Input that overshoots: the decoded signal goes beyond full scale and the
/// soft clipper bends it back. The result is still libopus's (on other
/// targets than 64-bit ARM, within one step).
#[test]
fn overdriven_input_is_soft_clipped_like_opus_decode() {
    for (name, rate, channels, mode, bandwidth, bitrate, frame_2_5) in STREAMS {
        for (signal, mono) in [("overdriven", overdriven(rate, 2.0)), ("full scale", full_scale(rate, rate))] {
            let pcm = if channels == 2 {
                // The right channel is the left one inverted and louder, so
                // that the two channels clip at different places.
                mono.iter().flat_map(|&v| [v, -1.2 * v]).collect()
            } else {
                mono
            };
            let (c, rs) = decode_both(rate, channels, mode, bandwidth, bitrate, rate / 400 * frame_2_5, &pcm);
            let d = diff(&c, &rs);
            println!("{name}, {signal}: {} samples, {} at full scale, {} differ by 1, largest difference {}", d.samples, d.at_full_scale, d.off_by_one, d.max);
            assert!(d.max <= if EXACT { 0 } else { 1 }, "{name}, {signal}: largest difference {}", d.max);
            assert!(d.off_by_one * 100 < d.samples, "{name}, {signal}: {} of {} samples differ", d.off_by_one, d.samples);
            if signal == "overdriven" && mode == MODE_CELT {
                assert!(d.at_full_scale > 0, "{name}: the signal was meant to reach full scale");
            }
        }
    }
}

/// The float path of the same streams is not clipped at all: the samples
/// beyond ±1 that the soft clipper works on are really there.
#[test]
fn float_output_keeps_the_overshoot() {
    let rate = 48000;
    let pcm = overdriven(rate, 2.0);
    let mut enc = CEncoder::new(rate, 1, APP_AUDIO);
    enc.force_mode(MODE_CELT).bitrate(96000);
    let packets = enc.encode_all(&pcm, 960);
    let mut c_dec = CDecoder::new(rate, 1);
    let mut dec = OpusDecoder::new(rate as i32, 1).unwrap();
    let mut buf = vec![0f32; 960];
    let (mut c, mut rs) = (Vec::new(), Vec::new());
    for p in &packets {
        c.extend(c_dec.decode(Some(p), 960, false).unwrap());
        dec.decode(p, 960, &mut buf).unwrap();
        rs.extend_from_slice(&buf);
    }
    let peak = |x: &[f32]| x.iter().fold(0f32, |m, v| m.max(v.abs()));
    println!("peak libopus {:.3}, opus1 {:.3}, {:.1} dB", peak(&c), peak(&rs), snr(&c, &rs));
    assert!(peak(&rs) > 1.5 && (peak(&rs) - peak(&c)).abs() < 1e-3);
    assert!(snr(&c, &rs) > 100.0);
}

/// Lost packets and FEC with 16-bit output. libopus does not soft-clip
/// concealed or recovered frames; it only rounds them.
#[test]
fn lost_packets_and_fec_decode_like_opus_decode() {
    let rate = 16000;
    let frame = 320;
    let pcm = speech_fullband(rate, 3.0);
    let mut enc = CEncoder::new(rate, 1, APP_VOIP);
    enc.force_mode(MODE_SILK).bandwidth(BW_WB).bitrate(28000).fec(true, 20);
    let packets = enc.encode_all(&pcm, frame);
    let mut c_dec = CDecoder::new(rate, 1);
    let mut dec = OpusDecoder::new(rate as i32, 1).unwrap();
    let mut buf = vec![0i16; 12 * frame];
    let (mut c, mut rs) = (Vec::new(), Vec::new());
    for i in 0..packets.len() - 1 {
        let (reference, n) = match i % 10 {
            // Lost, recovered from the next packet.
            3 => (c_dec.decode_i16(Some(&packets[i + 1]), frame, true), dec.decode_fec_i16(&packets[i + 1], frame, &mut buf)),
            // Lost, concealed.
            6 => (c_dec.decode_i16(None, frame, false), dec.decode_i16(&[], frame, &mut buf)),
            // A table-of-contents byte alone: one concealed frame.
            8 => (c_dec.decode_i16(Some(&packets[i][..1]), frame, false), dec.decode_i16(&packets[i][..1], frame, &mut buf)),
            _ => (c_dec.decode_i16(Some(&packets[i]), frame, false), dec.decode_i16(&packets[i], frame, &mut buf)),
        };
        let reference = reference.unwrap();
        assert_eq!(n, Ok(reference.len()), "packet {i}");
        c.extend(reference);
        rs.extend_from_slice(&buf[..frame]);
    }
    let d = diff(&c, &rs);
    println!("{} samples, {} differ by 1, largest difference {}", d.samples, d.off_by_one, d.max);
    assert_eq!(d.max, 0, "SILK output is 16-bit to begin with");

    // A loss longer than the 120 ms one packet can hold is concealed in
    // full, 400 ms here.
    let long = 20 * frame;
    let mut buf = vec![0i16; long];
    let reference = c_dec.decode_i16(None, long, false).unwrap();
    assert_eq!(dec.decode_i16(&[], long, &mut buf), Ok(long));
    assert_eq!(diff(&reference, &buf).max, 0);
    // So is a long loss whose end is recovered from FEC data.
    let last = packets.len() - 1;
    let reference = c_dec.decode_i16(Some(&packets[last]), long, true).unwrap();
    assert_eq!(dec.decode_fec_i16(&packets[last], long, &mut buf), Ok(long));
    assert_eq!(diff(&reference, &buf).max, 0);
}

/// Arguments that do not fit are errors.
#[test]
fn bad_arguments_are_errors() {
    let mut enc = CEncoder::new(48000, 2, APP_AUDIO);
    let packet = enc.encode(&vec![0.1f32; 960 * 2]);
    let mut dec = OpusDecoder::new(48000, 2).unwrap();
    let mut buf = vec![0i16; 960 * 2];
    assert!(dec.decode_i16(&packet, 0, &mut buf).is_err());
    assert!(dec.decode_i16(&packet, 480, &mut buf).is_err());
    assert!(dec.decode_i16(&packet, 960, &mut buf[..959 * 2]).is_err());
    assert!(dec.decode_i16(&[], 961, &mut buf).is_err());
    assert!(dec.decode_i16(&[], 1920, &mut buf).is_err());
    assert!(dec.decode_i16(&[0xFF], 960, &mut buf).is_err());
    // The caller's frame_size may be larger than the packet when the buffer
    // holds the packet.
    assert_eq!(dec.decode_i16(&packet, 5760, &mut buf), Ok(960));
}

/// `opus_pcm_soft_clip` on a copy of `pcm`.
fn c_soft_clip(pcm: &[f32], frame_size: usize, channels: usize, mem: &mut [f32]) -> Vec<f32> {
    let mut out = pcm.to_vec();
    unsafe { sys::opus_pcm_soft_clip(out.as_mut_ptr(), frame_size as i32, channels as i32, mem.as_mut_ptr()) };
    out
}

/// `pcm_soft_clip` gives the output of `opus_pcm_soft_clip`, buffer after
/// buffer with the memory carried along, for buffers of many lengths. On
/// 64-bit ARM, where opus1 fuses the multiply-adds as clang does, the bits
/// are equal.
#[test]
fn pcm_soft_clip_matches_libopus() {
    let mut seed = 12345u32;
    let mut rand = move || {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        (seed >> 8) as f32 / (1u32 << 23) as f32 - 1.0
    };
    let mut worst = 0f32;
    let mut clipped_buffers = 0;
    for channels in [1usize, 2] {
        for (gain, offset) in [(0.9f32, 0.0f32), (1.5, 0.0), (3.0, 0.0), (0.6, 0.9), (1.0, -1.4), (6.0, 0.5)] {
            let (mut c_mem, mut rs_mem) = (vec![0f32; channels], vec![0f32; channels]);
            let mut phase = [0f32; 2];
            for round in 0..60 {
                let n = [1usize, 2, 3, 7, 40, 120, 480, 960][round % 8];
                let mut pcm = vec![0f32; n * channels];
                for i in 0..n {
                    for c in 0..channels {
                        // A tone with some noise; the second channel has
                        // another pitch.
                        phase[c] += 0.013 + 0.02 * c as f32 + 0.0005 * round as f32;
                        pcm[i * channels + c] = gain * (phase[c].sin() + 0.1 * rand()) + offset;
                    }
                }
                let expected = c_soft_clip(&pcm, n, channels, &mut c_mem);
                let mut ours = pcm.clone();
                pcm_soft_clip(&mut ours, n, channels, &mut rs_mem);
                let exact = cfg!(target_arch = "aarch64");
                for (i, (a, b)) in expected.iter().zip(&ours).enumerate() {
                    worst = worst.max((a - b).abs());
                    let ok = if exact { a.to_bits() == b.to_bits() } else { (a - b).abs() <= 2e-6 };
                    assert!(ok, "gain {gain}, offset {offset}, {channels} ch, round {round}, sample {i}: libopus {a}, opus1 {b}");
                    assert!(b.abs() <= 1.0, "sample {i} is {b}");
                }
                for (a, b) in c_mem.iter().zip(&rs_mem) {
                    let ok = if exact { a.to_bits() == b.to_bits() } else { (a - b).abs() <= 1e-6 * a.abs().max(1.0) };
                    assert!(ok, "memory: libopus {a}, opus1 {b}");
                }
                clipped_buffers += (expected != pcm) as usize;
            }
        }
    }
    println!("largest difference from libopus {worst:e}; {clipped_buffers} buffers were changed by the clipper");
    assert!(clipped_buffers > 300);

    // Arguments that do not fit change nothing and do not panic.
    let mut pcm = [1.5f32, -1.5, 1.5];
    pcm_soft_clip(&mut pcm, 0, 1, &mut [0.0]);
    pcm_soft_clip(&mut pcm, 3, 0, &mut [0.0]);
    pcm_soft_clip(&mut pcm, 4, 1, &mut [0.0]);
    pcm_soft_clip(&mut pcm, 3, 2, &mut [0.0, 0.0]);
    pcm_soft_clip(&mut pcm, 3, 1, &mut []);
    assert_eq!(pcm, [1.5, -1.5, 1.5]);
    // Not-a-number samples pass through without a panic or an endless loop.
    let mut pcm = [0.5f32, f32::NAN, 1.7, f32::INFINITY, -0.3];
    pcm_soft_clip(&mut pcm, 5, 1, &mut [0.0]);
    assert!(pcm[1].is_nan());
}
