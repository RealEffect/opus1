//! The high-pass filter in front of SILK with the `Voip` application.
//!
//! libopus's fixed-point build filters the 16-bit input with an integer
//! second-order filter (`hp_cutoff` in opus_encoder.c, which calls
//! `silk_biquad_alt_stride1`). Its state updates round down, and the poles
//! of the filter lie close to 1, so the rounding adds a constant to the
//! output. At the lowest cutoff, 60 Hz, the constant is about -9 (in 16-bit
//! units) at 48 kHz, -2 at 24 kHz and -0.5 at 16 kHz. SILK codes it, and the
//! decoder plays it as energy below 200 Hz in quiet frames. The float
//! build filters in float and adds no such constant.
//!
//! opus1 gives SILK the float filter's output at 48 kHz, as the float build
//! does, and the integer filter's output at 8 to 24 kHz, as the fixed-point
//! build does. The test control `internals::silk_input::set_integer_hp_at_48k`
//! brings back the integer filter at 48 kHz.
//!
//! The integer filter runs on every frame, CELT-only frames included, as in
//! the fixed-point build. When it ran on SILK's frames only, the first SILK
//! frame after CELT-only frames started from the filter state of the last
//! SILK frame, and SILK coded that step as energy below 200 Hz.

mod k24_common;

use std::num::NonZeroU32;
use k24_common::{resample, BW_FB, BW_WB, CDec, CEnc, MODE_CELT, MODE_HYBRID, MODE_SILK};
use opus1::internals::silk_input::set_integer_hp_at_48k;
use opus1::{Application, Bandwidth, Bitrate, Channels, ForcedMode, OpusEncoder, SampleRate};

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

/// The settings of one stream: forced mode (libopus's number), forced
/// bandwidth (libopus's number) and bitrate. VoIP, mono, 20 ms frames,
/// complexity 5, variable bitrate.
#[derive(Clone, Copy)]
struct Stream {
    mode: i32,
    bandwidth: i32,
    bitrate: i32,
}

const FRAME: usize = 960;

/// The input followed by zeros, so that the decoder outputs every input
/// sample after the encoder's look-ahead.
fn padded(pcm: &[i16], lookahead: usize) -> Vec<i16> {
    let mut x = pcm.to_vec();
    x.resize((pcm.len() + lookahead).div_ceil(FRAME) * FRAME, 0);
    x
}

fn opus1_packets(pcm: &[i16], rate: i32, s: Stream, integer_hp: bool) -> Vec<Vec<u8>> {
    let frame = rate as usize / 50;
    let mut enc = OpusEncoder::new(
        SampleRate::try_from(rate).unwrap(),
        Channels::Mono,
        Application::Voip,
    )
    .unwrap();
    set_integer_hp_at_48k(&mut enc, integer_hp);
    enc.set_bitrate(Bitrate::BitsPerSecond(NonZeroU32::new(s.bitrate as u32).unwrap()));
    enc.set_complexity(5).unwrap();
    enc.set_force_mode(Some(if s.mode == MODE_SILK { ForcedMode::SilkOnly } else { ForcedMode::Hybrid }));
    enc.set_bandwidth(Some(if s.bandwidth == BW_WB { Bandwidth::Wideband } else { Bandwidth::Fullband }));
    let mut buf = vec![0u8; 1500];
    pcm.chunks_exact(frame)
        .map(|x| {
            let n = enc.encode_i16(x, frame, &mut buf).unwrap();
            buf[..n].to_vec()
        })
        .collect()
}

/// The packets of the fixed-point build of libopus 1.6.1, which the
/// dev-dependency `libopus-fixed` compiles (tools/libopus-fixed), for the
/// same 16-bit input and settings, at 48 kHz.
fn fixed_point_packets(pcm: &[i16], s: Stream) -> Vec<Vec<u8>> {
    use libopus_fixed::consts::*;
    let mut enc = libopus_fixed::Encoder::new(48000, 1, OPUS_APPLICATION_VOIP);
    enc.set(OPUS_SET_BITRATE_REQUEST, s.bitrate);
    enc.set(OPUS_SET_COMPLEXITY_REQUEST, 5);
    enc.set(OPUS_SET_FORCE_MODE_REQUEST, s.mode);
    enc.set(OPUS_SET_BANDWIDTH_REQUEST, s.bandwidth);
    pcm.as_chunks::<FRAME>().0.iter().map(|x| enc.encode(x, FRAME, 1500).unwrap()).collect()
}

fn libopus_packets(pcm: &[i16], s: Stream) -> Vec<Vec<u8>> {
    let mut enc = CEnc::voip(48000, 1);
    enc.bitrate(s.bitrate).complexity(5).force_mode(s.mode).bandwidth(s.bandwidth);
    let x: Vec<f32> = pcm.iter().map(|&v| v as f32 / 32768.0).collect();
    enc.encode_all(&x, FRAME)
}

/// libopus decodes the packets; the look-ahead is dropped and the output
/// cut to `len` samples, in 16-bit units.
fn decoded(packets: &[Vec<u8>], lookahead: usize, len: usize) -> Vec<f32> {
    let mut dec = CDec::new(48000, 1);
    let mut out = Vec::new();
    for p in packets {
        out.extend(dec.decode(Some(p), FRAME, false).unwrap().iter().map(|v| v * 32768.0));
    }
    out[lookahead..lookahead + len].to_vec()
}

/// The energy (dB) at 0 to 200 Hz of each 10 ms frame of a 48 kHz signal:
/// the DFT bins 0 to 4 of a 20 ms Hann window centred on the frame.
fn low_band_db(x: &[f32]) -> Vec<f64> {
    let (hop, win) = (480usize, 960usize);
    (0..x.len() / hop)
        .map(|f| {
            let start = (f * hop) as isize - (hop / 2) as isize;
            let mut energy = 0.0;
            for k in 0..=4 {
                let (mut re, mut im) = (0.0f64, 0.0f64);
                for i in 0..win {
                    let j = start + i as isize;
                    if j < 0 || j as usize >= x.len() {
                        continue;
                    }
                    let w = 0.5 - 0.5 * (std::f64::consts::TAU * (i as f64 + 0.5) / win as f64).cos();
                    let phase = std::f64::consts::TAU * (k * i) as f64 / win as f64;
                    let v = x[j as usize] as f64 * w;
                    re += v * phase.cos();
                    im -= v * phase.sin();
                }
                energy += re * re + im * im;
            }
            10.0 * (energy + 1.0).log10()
        })
        .collect()
}

/// The 10 ms frames where `decoded` holds more than 10 dB more energy at
/// 0 to 200 Hz than both the original and `reference`.
fn low_band_bursts(original: &[i16], decoded: &[f32], reference: &[f32]) -> Vec<usize> {
    let x: Vec<f32> = original.iter().map(|&v| v as f32).collect();
    let (ex, ea, eb) = (low_band_db(&x), low_band_db(decoded), low_band_db(reference));
    (0..ex.len()).filter(|&i| ea[i] > ex[i] + 10.0 && ea[i] > eb[i] + 10.0).collect()
}

const SILK_12K: Stream = Stream { mode: MODE_SILK, bandwidth: BW_WB, bitrate: 12000 };
const HYBRID_32K: Stream = Stream { mode: MODE_HYBRID, bandwidth: BW_FB, bitrate: 32000 };

/// With the float filter no frame of opus1's output holds such energy
/// (more than the original and than the output of the linked float
/// libopus). With the integer filter opus1 writes the SILK-only packets of
/// the fixed-point build of libopus 1.6.1 (`fixed_point_packets`), and its
/// output shows such energy in the same frames as that build's, also in
/// Hybrid: when measured, 11 (`MD19_04`) and 18 (`MH43_04`) of the 250
/// frames of SILK at 12 kb/s, 11 and 23 of Hybrid at 32 kb/s.
#[test]
#[cfg_attr(not(target_arch = "aarch64"), ignore = "bit-exact with libopus only on 64-bit ARM (docs/10, Platforms)")]
fn voip_at_48_khz_adds_no_low_frequency_energy_in_quiet_frames() {
    let lookahead = 312; // 2.5 ms + 4 ms at 48 kHz, for every encoder
    for (name, s) in [("MD19_04", SILK_12K), ("MD19_04", HYBRID_32K), ("MH43_04", SILK_12K), ("MH43_04", HYBRID_32K)] {
        let what = format!("{name}, {} b/s", s.bitrate);
        let x = tsp_excerpt(name);
        let pcm = padded(&x, lookahead);
        let float_build = decoded(&libopus_packets(&pcm, s), lookahead, x.len());
        let bursts = |packets: &[Vec<u8>]| low_band_bursts(&x, &decoded(packets, lookahead, x.len()), &float_build);
        assert_eq!(bursts(&opus1_packets(&pcm, 48000, s, false)), Vec::<usize>::new(), "{what}");

        let integer_filter = opus1_packets(&pcm, 48000, s, true);
        let fixed_point = fixed_point_packets(&pcm, s);
        if s.mode == MODE_SILK {
            assert!(integer_filter == fixed_point, "{what}: the packets are not those of the fixed-point build");
        }
        let want = bursts(&fixed_point);
        println!("{what}: {} frames with the integer filter", want.len());
        assert!(!want.is_empty(), "{what}: the fixed-point build adds no energy");
        assert_eq!(bursts(&integer_filter), want, "{what}, integer filter");
    }
}

/// The control changes SILK's input only at 48 kHz. At 16 and 24 kHz the
/// integer filter of the fixed-point build feeds SILK either way.
#[test]
fn the_integer_filter_feeds_silk_below_48_khz() {
    let x48 = tsp_excerpt("MH43_04");
    for rate in [16000, 24000, 48000] {
        let pcm: Vec<i16> = x48.iter().step_by(48000 / rate as usize).copied().collect();
        let a = opus1_packets(&pcm, rate, SILK_12K, false);
        let b = opus1_packets(&pcm, rate, SILK_12K, true);
        let equal = a.iter().zip(&b).filter(|(p, q)| p == q).count();
        if rate == 48000 {
            assert_eq!(equal, 0, "48 kHz");
        } else {
            assert_eq!(equal, a.len(), "{rate} Hz");
        }
    }
}

/// One stream at `rate` in which SILK-only packets (16 kb/s) and CELT-only
/// packets (48 kb/s) take turns every 5 packets of 20 ms. VoIP, mono,
/// complexity 9. `x48` is the 48 kHz original; the encoders get it
/// resampled to `rate`. Returns the frames of 10 ms in which opus1's output
/// holds more than 10 dB more energy at 0 to 200 Hz than both the original
/// and libopus's output (both streams decoded by libopus at 48 kHz).
fn switching_bursts(x48: &[i16], rate: i32) -> usize {
    let frame = rate as usize / 50;
    let lookahead = rate as usize / 400 + rate as usize / 250;
    let x: Vec<f32> = x48.iter().map(|&v| v as f32).collect();
    let mut pcm: Vec<i16> =
        resample(&x, 48000, rate as usize).iter().map(|v| v.round().clamp(-32768.0, 32767.0) as i16).collect();
    pcm.resize((pcm.len() + lookahead).div_ceil(frame) * frame, 0);
    let turn = |k: usize| if (k / 5).is_multiple_of(2) { (MODE_SILK, 16000) } else { (MODE_CELT, 48000) };

    let mut enc = OpusEncoder::new(
        SampleRate::try_from(rate).unwrap(),
        Channels::Mono,
        Application::Voip,
    )
    .unwrap();
    enc.set_complexity(9).unwrap();
    let mut cenc = CEnc::voip(rate as usize, 1);
    cenc.complexity(9);
    let mut buf = vec![0u8; 1500];
    let (mut a, mut b) = (Vec::new(), Vec::new());
    for (k, f) in pcm.chunks_exact(frame).enumerate() {
        let (mode, bitrate) = turn(k);
        enc.set_force_mode(Some(if mode == MODE_SILK { ForcedMode::SilkOnly } else { ForcedMode::CeltOnly }));
        enc.set_bitrate(Bitrate::BitsPerSecond(NonZeroU32::new(bitrate as u32).unwrap()));
        let n = enc.encode_i16(f, frame, &mut buf).unwrap();
        a.push(buf[..n].to_vec());
        cenc.force_mode(mode).bitrate(bitrate);
        b.push(cenc.encode(&f.iter().map(|&v| v as f32 / 32768.0).collect::<Vec<_>>()));
    }
    let la48 = 312; // 2.5 ms + 4 ms at 48 kHz
    let (a, b) = (decoded(&a, la48, x48.len()), decoded(&b, la48, x48.len()));
    let (ex, ea, eb) = (low_band_db(&x), low_band_db(&a), low_band_db(&b));
    (0..ex.len()).filter(|&i| ea[i] > ex[i] + 10.0 && ea[i] > eb[i] + 10.0).count()
}

/// The integer filter runs on every frame at 16 and 24 kHz, as in the
/// fixed-point build. When it ran on SILK's frames only, 2 to 4 of the 250
/// frames of each of these streams held such energy (`MD19_04` 4 at both
/// rates, `MH43_04` 2, `FA01_05` 3, `FJ55_02` 2).
#[test]
fn voip_below_48_khz_adds_no_low_frequency_energy_after_celt_frames() {
    for name in ["MD19_04", "MH43_04", "FA01_05", "FJ55_02"] {
        let x = tsp_excerpt(name);
        for rate in [16000, 24000] {
            assert_eq!(switching_bursts(&x, rate), 0, "{name}, {rate} Hz");
        }
    }
}
