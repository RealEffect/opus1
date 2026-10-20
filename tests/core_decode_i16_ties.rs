//! The rounding of halfway values in the 16-bit output, against libopus.
//!
//! A float sample `x` becomes the 16-bit sample nearest to `x * 32768`.
//! When `x * 32768` lies exactly halfway between two integers (a "tie"),
//! the rule that breaks the tie depends on the libopus function and on the
//! processor:
//!
//! - `opus_decode` converts all the samples of one call with
//!   `celt_float2int16`. On ARM that is `celt_float2int16_neon`: it rounds
//!   ties away from zero in blocks of 16 values, and to the even integer in
//!   the values after the last whole block. On x86 and other targets it
//!   rounds every tie to the even integer.
//! - `opus_multistream_decode` converts each sample with `FLOAT2INT16`,
//!   which rounds ties to the even integer on every target.
//!
//! The tests decode loud CELT and Hybrid streams with libopus and with
//! opus1, at every output rate and with one and two channels, and require
//! the same 16-bit samples. A third libopus decoder gives the float output
//! of the same packets, which shows where the ties are. Only ties where the
//! two rules give different results count. The tests require such ties in
//! the blocks and after them, so that a wrong rule would fail.
//!
//! The soft clipper of the 16-bit path changes calls that go beyond full
//! scale. Their ties are not counted (see `check_call`).

mod core_common;
mod ms_support;

use core_common::*;
use opus1::OpusDecoder;
use opus1::multistream::MultistreamDecoder;
use std::f64::consts::PI;

const RATES: [usize; 5] = [8000, 12000, 16000, 24000, 48000];

/// Ties seen, where rounding away from zero and rounding to even differ.
#[derive(Default, Debug, Clone, Copy)]
struct Ties {
    /// In a block of 16 values of `celt_float2int16`.
    block: usize,
    /// After the last whole block of a call.
    rest: usize,
    /// Calls whose ties were not counted because of the soft clipper.
    skipped: usize,
    /// Whether the last call went beyond full scale.
    clipped: bool,
}

/// One decode call. `float` is libopus's float output of the call, `c` its
/// 16-bit output, `ours` opus1's 16-bit output. `blocks` says whether the
/// call converts in blocks of 16 (`opus_decode`) or each sample alone
/// (`opus_multistream_decode`).
///
/// A call that goes beyond full scale is soft-clipped before it is
/// rounded, and the clipper carries its bend into the start of the next
/// call. Then `float` is not the value that is rounded, and the ties of
/// both calls are not counted. The 16-bit samples must still be equal.
fn check_call(label: &str, float: &[f32], c: &[i16], ours: &[i16], blocks: bool, ties: &mut Ties) {
    assert_eq!(c.len(), float.len(), "{label}: float and 16-bit calls differ in length");
    if let Some(i) = (0..c.len()).find(|&i| ours[i] != c[i]) {
        panic!(
            "{label}: sample {i} of {}: libopus {}, opus1 {}, float value {} ({} after scaling)",
            c.len(),
            c[i],
            ours[i],
            float[i],
            float[i] * 32768.0
        );
    }
    let clipped = float.iter().any(|x| x.abs() > 1.0);
    let skip = clipped || ties.clipped;
    ties.clipped = clipped;
    if skip {
        ties.skipped += 1;
        return;
    }
    let whole = if blocks { float.len() / 16 * 16 } else { 0 };
    for (i, (&x, &c)) in float.iter().zip(c).enumerate() {
        let v = x * 32768.0;
        if (v - v.trunc()).abs() != 0.5 {
            continue;
        }
        let (away, even) = (v.round() as i16, v.round_ties_even() as i16);
        if away == even {
            continue;
        }
        // The rule of the libopus these tests link. On ARM its
        // `opus_decode` calls `celt_float2int16_neon`.
        let arm = cfg!(any(target_arch = "aarch64", target_arch = "arm"));
        let expected = if i < whole && arm { away } else { even };
        assert_eq!(c, expected, "{label}: libopus rounds the tie {v} at {i} of {}", float.len());
        if i < whole {
            ties.block += 1;
        } else {
            ties.rest += 1;
        }
    }
}

/// A loud signal: three tones per channel, a slow swell, and some noise.
/// Its peak is about 0.9, so that many samples lie where a float has room
/// for a fraction of one half after the scaling by 32768. The decoded
/// signal goes beyond full scale in places.
fn loud(rate: usize, channels: usize, samples: usize, seed: u32) -> Vec<f32> {
    let mut state = seed;
    let mut pcm = vec![0f32; samples * channels];
    for i in 0..samples {
        let t = i as f64 / rate as f64;
        let swell = 0.75 + 0.25 * (2.0 * PI * 0.4 * t).sin();
        for c in 0..channels {
            let f = 1.0 + 0.13 * c as f64;
            let tones = 0.32 * (2.0 * PI * 220.0 * f * t).sin()
                + 0.22 * (2.0 * PI * 1250.0 * f * t).sin()
                + 0.12 * (2.0 * PI * 5100.0 * f * t).sin();
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            let white = (state >> 8) as f64 / (1u64 << 23) as f64 - 1.0;
            pcm[i * channels + c] = (1.3 * swell * tones + 0.05 * white) as f32;
        }
    }
    pcm
}

/// Frame durations of the streams, in units of 2.5 ms. 2.5 ms frames leave
/// values after the last whole block at most rates (see `has_rest`).
const DURATIONS: [usize; 10] = [1, 1, 1, 1, 2, 1, 1, 1, 4, 8];

/// Whether a call of `samples` values per channel leaves values after the
/// last whole block.
fn has_rest(samples: usize, channels: usize) -> bool {
    !(samples * channels).is_multiple_of(16)
}

/// Packets of `seconds` of `loud`, coded by libopus at 48 kHz in `mode`,
/// with the frame durations of `durations` in turn.
fn c_packets(mode: i32, channels: usize, seconds: f64, durations: &[usize]) -> Vec<(Vec<u8>, usize)> {
    let rate = 48000;
    let pcm = loud(rate, channels, (seconds * rate as f64) as usize, 0x2468_ace1 + channels as u32);
    let mut enc = CEncoder::new(rate, channels, APP_AUDIO);
    enc.force_mode(mode).bandwidth(BW_FB).bitrate(if mode == MODE_CELT { 96000 } else { 48000 } * channels as i32);
    let mut packets = Vec::new();
    let mut pos = 0;
    for &units in durations.iter().cycle() {
        let n = units * rate / 400;
        if (pos + n) * channels > pcm.len() {
            break;
        }
        packets.push((enc.encode(&pcm[pos * channels..(pos + n) * channels]), units));
        pos += n;
    }
    packets
}

/// Decode `packets` at `rate` with libopus (float and 16 bits) and opus1
/// (16 bits). Every 13th packet is lost. Returns the ties seen.
fn decode_single(label: &str, packets: &[(Vec<u8>, usize)], rate: usize, channels: usize) -> Ties {
    let mut c_float = CDecoder::new(rate, channels);
    let mut c_i16 = CDecoder::new(rate, channels);
    let mut ours = OpusDecoder::new(rate as i32, channels).unwrap();
    let mut buf = vec![0i16; rate / 400 * 48 * channels];
    let mut ties = Ties::default();
    for (k, (p, units)) in packets.iter().enumerate() {
        let n = units * rate / 400;
        let label = format!("{label}, {rate} Hz, packet {k}");
        let lost = k % 13 == 12;
        let (float, c, got) = if lost {
            (c_float.decode(None, n, false), c_i16.decode_i16(None, n, false), ours.decode_i16(&[], n, &mut buf))
        } else {
            (c_float.decode(Some(p), n, false), c_i16.decode_i16(Some(p), n, false), ours.decode_i16(p, n, &mut buf))
        };
        assert_eq!(got, Ok(n), "{label}");
        check_call(&label, &float.unwrap(), &c.unwrap(), &buf[..n * channels], true, &mut ties);
    }
    ties
}

/// CELT: ties in the blocks and after them, at every output rate, mono and
/// stereo.
#[test]
fn celt_ties_round_as_opus_decode() {
    for channels in [1, 2] {
        let packets = c_packets(MODE_CELT, channels, 6.0, &DURATIONS);
        for rate in RATES {
            let label = format!("CELT, {channels} ch");
            let ties = decode_single(&label, &packets, rate, channels);
            println!("{label}, {rate} Hz: {ties:?}");
            assert!(ties.block > 0, "{label}, {rate} Hz: no tie in a block");
            // A stream with 2.5 ms frames leaves values after the last
            // block, except at 16 and 48 kHz in stereo, where every
            // duration of a call is a whole number of blocks.
            if has_rest(rate / 400, channels) {
                assert!(ties.rest > 0, "{label}, {rate} Hz: no tie after the last block");
            } else {
                assert_eq!(ties.rest, 0);
            }
        }
    }
}

/// Hybrid (SILK plus CELT, 10 and 20 ms frames), mono and stereo, at
/// every output rate. Ties are required at 24 and 48 kHz. At 8 to 16 kHz
/// the CELT band of Hybrid (above 8 kHz) is beyond the output band, so the
/// output is SILK's 16-bit samples and has no ties.
#[test]
fn hybrid_ties_round_as_opus_decode() {
    for channels in [1, 2] {
        let packets = c_packets(MODE_HYBRID, channels, 3.0, &[4, 8]);
        for rate in RATES {
            let label = format!("Hybrid, {channels} ch");
            let ties = decode_single(&label, &packets, rate, channels);
            println!("{label}, {rate} Hz: {ties:?}");
            if rate > 16000 {
                assert!(ties.block > 0, "{label}, {rate} Hz: no tie in a block");
            }
        }
    }
}

/// A loss longer than 120 ms. opus1 conceals it 60 ms at a time and
/// converts each piece alone; libopus converts the whole call at once. A
/// piece of 60 ms is a whole number of blocks, so the blocks fall at the
/// same places. A loss of 122.5 ms leaves values after the last block at
/// 48 kHz in mono.
#[test]
fn long_loss_converts_like_one_call() {
    let (rate, channels) = (48000, 1);
    let packets = c_packets(MODE_CELT, channels, 1.0, &[8]);
    let mut c_float = CDecoder::new(rate, channels);
    let mut c_i16 = CDecoder::new(rate, channels);
    let mut ours = OpusDecoder::new(rate as i32, channels).unwrap();
    let mut buf = vec![0i16; rate / 4];
    let mut ties = Ties::default();
    for (k, (p, _)) in packets.iter().enumerate() {
        let label = format!("packet {k}");
        let n = 960;
        assert_eq!(ours.decode_i16(p, n, &mut buf), Ok(n));
        check_call(
            &label,
            &c_float.decode(Some(p), n, false).unwrap(),
            &c_i16.decode_i16(Some(p), n, false).unwrap(),
            &buf[..n],
            true,
            &mut ties,
        );
        if k % 10 == 9 {
            let n = 49 * rate / 400;
            assert!(has_rest(n, channels));
            assert_eq!(ours.decode_i16(&[], n, &mut buf), Ok(n));
            let label = format!("loss after packet {k}");
            check_call(
                &label,
                &c_float.decode(None, n, false).unwrap(),
                &c_i16.decode_i16(None, n, false).unwrap(),
                &buf[..n],
                true,
                &mut ties,
            );
        }
    }
    println!("{ties:?}");
}

/// `opus_multistream_decode` rounds every tie to the even integer, on
/// every target. Three channels in two streams (one stereo, one mono), at
/// every output rate.
#[test]
fn multistream_ties_round_to_even() {
    let (fs, channels, streams, coupled, mapping) = (48000, 3, 2, 1, [0u8, 1, 2]);
    let pcm = loud(fs as usize, channels, 4 * fs as usize, 0x1357_9bdf);
    let mut enc = ms_support::CMsEncoder::generic(fs, channels, streams, coupled, &mapping, APP_AUDIO).unwrap();
    enc.set_bitrate(288000);
    let mut packets = Vec::new();
    let mut pos = 0;
    for &units in DURATIONS.iter().cycle() {
        let n = units * fs as usize / 400;
        if (pos + n) * channels > pcm.len() {
            break;
        }
        let frame = &pcm[pos * channels..(pos + n) * channels];
        packets.push((enc.encode_float(frame, n, ms_support::max_packet_bytes(streams)).unwrap(), n));
        pos += n;
    }
    for rate in RATES {
        let mut c_float = ms_support::CMsDecoder::new(rate as i32, channels, streams, coupled, &mapping).unwrap();
        let mut c_i16 = ms_support::CMsDecoder::new(rate as i32, channels, streams, coupled, &mapping).unwrap();
        let mut ours = MultistreamDecoder::with_layout(rate as i32, channels, streams, coupled, &mapping).unwrap();
        let mut buf = vec![0i16; rate / 50 * channels];
        let mut ties = Ties::default();
        for (k, (p, n48)) in packets.iter().enumerate() {
            let n = n48 * rate / 48000;
            let label = format!("multistream, {rate} Hz, packet {k}");
            let lost = k % 13 == 12;
            let (float, c, got) = if lost {
                (c_float.decode_float(None, n, false), c_i16.decode_i16(None, n, false), ours.decode_i16(&[], n, &mut buf))
            } else {
                (c_float.decode_float(Some(p), n, false), c_i16.decode_i16(Some(p), n, false), ours.decode_i16(p, n, &mut buf))
            };
            assert_eq!(got, Ok(n), "{label}");
            check_call(&label, &float.unwrap(), &c.unwrap(), &buf[..n * channels], false, &mut ties);
        }
        println!("multistream, {rate} Hz: {} ties", ties.rest);
        assert!(ties.rest > 0, "multistream, {rate} Hz: no tie");
    }
}
