//! The residual energies that set SILK's quantization gains.
//!
//! libopus's fixed-point SILK encoder (`silk_find_pred_coefs_FIX`) scales
//! each subframe by the smallest gain of the frame over the subframe's
//! gain, times 1/4, and rounds the result to 16 bits. It then computes the
//! residual energy of each subframe from that signal
//! (`silk_residual_energy_FIX`). In a frame where one subframe is near
//! digital silence, the other subframes are scaled down by a factor of 50
//! or more, and the rounding error makes their energies several times too
//! large. At a low SNR the energy decides the gain, so the gain comes out
//! too large as well, and the decoder plays the frame's noise too loudly.
//!
//! A stereo stream whose bitrate drops below the stereo threshold shows it.
//! In the packet of the drop the stereo width falls to 0: the side channel
//! fades out over 8 ms and then holds zeros, and it gets almost no bits.
//! In the example of docs/guide/10-differences-from-libopus.md (section SILK) the
//! gain of that frame came out 15 dB above the float build's, and the decoded
//! left and right channels held a burst of energy below 200 Hz.
//!
//! opus1 also computes the energies from the unscaled signal, and takes the
//! gain they give when it is less than half of the gain from libopus's
//! energies. The test control `internals::silk_input::set_fixed_point_res_nrg`
//! turns that off, so that SILK's gains are those of the fixed-point build.

mod k24_common;

use std::num::NonZeroU32;
use k24_common::*;
use opus1::internals::silk_input::set_fixed_point_res_nrg;
use opus1::{Application, Bitrate, Channels, OpusEncoder, SampleRate};

/// A 2.5 s excerpt of the TSP speech database at 48 kHz (`fixtures/tsp/`).
fn tsp_excerpt(name: &str) -> Vec<i16> {
    let path = format!("{}/fixtures/tsp/{name}.wav", env!("CARGO_MANIFEST_DIR"));
    let bytes = std::fs::read(&path).unwrap();
    let mut pos = 12;
    while pos + 8 <= bytes.len() {
        let size = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into().unwrap()) as usize;
        if &bytes[pos..pos + 4] == b"data" {
            let data = &bytes[pos + 8..pos + 8 + size];
            return data.as_chunks::<2>().0.iter().map(|&c| i16::from_le_bytes(c)).collect();
        }
        pos += 8 + size + (size & 1);
    }
    panic!("{path}: no data chunk");
}

const EXCERPTS: [&str; 4] = ["FA01_05", "MD19_04", "FJ55_02", "MH43_04"];

/// 32 kb/s and 8 kb/s, taking turns every 10 packets of 20 ms. Each step
/// down ends the stereo coding.
fn bitrate(packet: usize) -> i32 {
    if (packet / 10).is_multiple_of(2) { 32_000 } else { 8_000 }
}

/// The energy of the DFT bins 0 to 4 (0 to 200 Hz) of a 20 ms Hann window,
/// every 10 ms, in dB.
fn low_band_db(x: &[f32]) -> Vec<f64> {
    let (hop, win) = (480usize, 960usize);
    let w: Vec<f64> =
        (0..win).map(|i| 0.5 - 0.5 * (std::f64::consts::TAU * (i as f64 + 0.5) / win as f64).cos()).collect();
    (0..x.len() / hop)
        .map(|f| {
            let start = (f * hop) as isize - (hop / 2) as isize;
            let mut energy = 0.0;
            for k in 0..=4 {
                let (mut re, mut im) = (0.0f64, 0.0f64);
                for (i, wi) in w.iter().enumerate() {
                    let j = start + i as isize;
                    if j < 0 || j as usize >= x.len() {
                        continue;
                    }
                    let v = x[j as usize] as f64 * wi;
                    let ph = std::f64::consts::TAU * (k * i) as f64 / win as f64;
                    re += v * ph.cos();
                    im -= v * ph.sin();
                }
                energy += re * re + im * im;
            }
            10.0 * (energy + 1.0).log10()
        })
        .collect()
}

fn channel(x: &[f32], c: usize) -> Vec<f32> {
    x.iter().skip(c).step_by(2).copied().collect()
}

/// libopus decodes the packets at 48 kHz, in 16-bit units, without the
/// look-ahead, cut to `len` samples per channel.
fn decoded(packets: &[Vec<u8>], len: usize) -> Vec<f32> {
    let mut dec = CDec::new(48000, 2);
    let mut out = Vec::new();
    for p in packets {
        out.extend(dec.decode(Some(p), 960, false).unwrap().iter().map(|v| v * 32768.0));
    }
    out[312 * 2..(312 + len) * 2].to_vec()
}

/// Stereo `Voip` at 48 kHz with the bitrates of [`bitrate`]: opus1's
/// packets, with SILK's gains from libopus's energies when `fixed_point`.
fn opus1_packets(pcm: &[i16], complexity: i32, fixed_point: bool) -> Vec<Vec<u8>> {
    let mut enc =
        OpusEncoder::new(SampleRate::Hz48000, Channels::Stereo, Application::Voip).unwrap();
    enc.set_complexity(complexity as u8).unwrap();
    set_fixed_point_res_nrg(&mut enc, fixed_point);
    let mut buf = vec![0u8; 4000];
    pcm.as_chunks::<{ 960 * 2 }>()
        .0
        .iter()
        .enumerate()
        .map(|(k, x)| {
            enc.set_bitrate(Bitrate::BitsPerSecond(NonZeroU32::new(bitrate(k) as u32).unwrap()));
            let n = enc.encode_i16(x, 960, &mut buf).unwrap();
            buf[..n].to_vec()
        })
        .collect()
}

fn libopus_packets(pcm: &[i16], complexity: i32) -> Vec<Vec<u8>> {
    let mut enc = CEnc::voip(48000, 2);
    enc.complexity(complexity);
    pcm.as_chunks::<{ 960 * 2 }>()
        .0
        .iter()
        .enumerate()
        .map(|(k, x)| {
            enc.bitrate(bitrate(k));
            enc.encode(&x.iter().map(|&v| v as f32 / 32768.0).collect::<Vec<_>>())
        })
        .collect()
}

/// Frames of 10 ms whose energy below 200 Hz lies more than 10 dB above
/// both the input's and that of libopus's output.
fn bursts(input: &[f32], out: &[f32], libopus: &[f32]) -> usize {
    (0..2)
        .map(|c| {
            let (x, o, l) = (low_band_db(&channel(input, c)), low_band_db(&channel(out, c)), low_band_db(&channel(libopus, c)));
            (0..x.len()).filter(|&i| o[i] > x[i] + 10.0 && o[i] > l[i] + 10.0).count()
        })
        .sum()
}

/// The 12 ordered pairs of the excerpts as stereo (one talker on each
/// channel), at complexity 5 and 9, 6,000 frames of 10 ms per complexity.
/// Measured when this test was written, frames with a burst:
///
/// | | complexity 5 | complexity 9 |
/// |---|---:|---:|
/// | gains from libopus's energies (the fixed-point build) | 15 | 10 |
/// | opus1 | 5 | 2 |
///
/// docs/guide/10-differences-from-libopus.md (section SILK) describes the bursts
/// that stay.
#[test]
fn a_drop_of_the_stereo_bitrate_plays_fewer_low_frequency_bursts() {
    for complexity in [5, 9] {
        let (mut with_unscaled, mut fixed_point) = (0, 0);
        for left in EXCERPTS {
            for right in EXCERPTS.into_iter().filter(|&right| right != left) {
                let (l, r) = (tsp_excerpt(left), tsp_excerpt(right));
                let n = l.len().min(r.len());
                let pcm: Vec<i16> = (0..2 * n).map(|i| if i % 2 == 0 { l[i / 2] } else { r[i / 2] }).collect();
                // The look-ahead of 6.5 ms, then whole packets.
                let mut padded = pcm.clone();
                padded.resize((n + 312).div_ceil(960) * 960 * 2, 0);
                let input: Vec<f32> = pcm.iter().map(|&v| v as f32).collect();
                let libopus = decoded(&libopus_packets(&padded, complexity), n);
                with_unscaled += bursts(&input, &decoded(&opus1_packets(&padded, complexity, false), n), &libopus);
                fixed_point += bursts(&input, &decoded(&opus1_packets(&padded, complexity, true), n), &libopus);
            }
        }
        println!("complexity {complexity}: frames with a burst: {with_unscaled}, with libopus's energies {fixed_point}");
        assert!(fixed_point >= 5, "complexity {complexity}: the streams no longer show the bursts ({fixed_point})");
        assert!(
            2 * with_unscaled < fixed_point,
            "complexity {complexity}: {with_unscaled} frames with a burst, {fixed_point} with libopus's energies"
        );
    }
}

/// Speech that never falls to digital silence within a frame gets the
/// packets of the fixed-point energies: the energies of the unscaled
/// signal only take over in rare frames.
#[test]
fn speech_at_a_constant_bitrate_codes_the_packets_of_the_fixed_point_energies() {
    for (rate, bitrate) in [(16000, 20_000), (24000, 24_000), (48000, 32_000)] {
        for name in EXCERPTS {
            let x = tsp_excerpt(name);
            let pcm: Vec<i16> = resample(&x.iter().map(|&v| v as f32).collect::<Vec<_>>(), 48000, rate as usize)
                .iter()
                .map(|v| v.round().clamp(-32768.0, 32767.0) as i16)
                .collect();
            let frame = rate as usize / 50;
            let code = |fixed_point: bool| -> Vec<Vec<u8>> {
                let mut enc = OpusEncoder::new(
                    SampleRate::try_from(rate).unwrap(),
                    Channels::Mono,
                    Application::Voip,
                )
                .unwrap();
                enc.set_complexity(9).unwrap();
                enc.set_bitrate(Bitrate::BitsPerSecond(NonZeroU32::new(bitrate as u32).unwrap()));
                set_fixed_point_res_nrg(&mut enc, fixed_point);
                let mut buf = vec![0u8; 4000];
                pcm.chunks_exact(frame)
                    .map(|f| {
                        let n = enc.encode_i16(f, frame, &mut buf).unwrap();
                        buf[..n].to_vec()
                    })
                    .collect()
            };
            assert!(code(false) == code(true), "{name} at {rate} Hz, {bitrate} b/s");
        }
    }
}
