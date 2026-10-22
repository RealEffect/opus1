//! DRED (deep redundancy) parsing and decoding: `OpusDredDecoder::parse`,
//! `process` and `process_in_place`, and `OpusDecoder::dred_decode_float`,
//! `dred_decode` and `dred_decode24`.
//!
//! The first input byte picks the packet:
//!
//! - the input as it is;
//! - a packet of opus1's encoder with DRED data (`common/pool.rs`), whose
//!   DRED payload is changed (bytes flipped, cut short or lengthened, its
//!   header changed, moved to another frame) and written back with
//!   `packet::pad_with_extensions`;
//! - a packet made from the input: a TOC byte, frames, and a DRED
//!   extension whose payload is input bytes, after the DRED header or not.
//!
//! The packet is parsed with a random `max_dred_samples` (sometimes any
//! `i32`), a random sampling rate and deferred processing or not, then decoded several times with `dred_decode*` at
//! random offsets (sometimes any `i32`) and lengths, into buffers of random
//! length, by a decoder at a random rate, channel count and complexity 0 to
//! 10, with the bandwidth extension on or off.
//!
//! What must hold: nothing panics or hangs. `parse` gives no negative
//! durations; after an error the data is not parsed, without deferral it is
//! processed. `process` processes exactly what is parsed. Processed
//! features are finite. A DRED decode that succeeds returns `frame_size`,
//! fits its buffer, writes no sample past `frame_size * channels`, and
//! gives finite `f32` samples. So does decoding the packet afterwards.
//!
//! Needs the `dred` feature of this package; `fuzz.sh` builds the target
//! with it.

#![no_main]

mod common;

use common::pool::{DRED_ID, pool};
use common::RATES;
use libfuzzer_sys::arbitrary::{Result, Unstructured};
use libfuzzer_sys::fuzz_target;
use opus1::extensions::{self, Extension};
use opus1::packet::{self, Framing, Padding};
use opus1::{Channels, DredProcessing, OpusDecoder, OpusDred, OpusDredDecoder, SampleRate};
use std::sync::OnceLock;

/// The DRED header: "D" and the experimental version.
const HEADER: [u8; 2] = [b'D', 12];

/// A NaN with a payload no decoder writes: marks samples not written.
const UNWRITTEN: u32 = 0x7FC0_5A5A;

fuzz_target!(|data: &[u8]| {
    let Some((&mode, rest)) = data.split_first() else { return };
    let mut u = Unstructured::new(rest);
    let _ = run(mode, &mut u);
});

fn dred_decoder() -> &'static OpusDredDecoder {
    static DEC: OnceLock<OpusDredDecoder> = OnceLock::new();
    DEC.get_or_init(OpusDredDecoder::new)
}

/// `toc` and `frames` as a packet whose padding holds `exts`.
fn build(toc: u8, frames: &[&[u8]], exts: &[Extension]) -> Option<Vec<u8>> {
    let mut out = vec![0u8; 48 * 1277 + 8];
    let n = packet::write(toc, frames, &mut out, Framing::Normal, Padding::Shortest).ok()?;
    let ext_len = extensions::generated_len(1 << 20, exts, frames.len()).ok()?;
    let new_len = n + ext_len + 2 * 48 + 16 + ext_len / 254;
    out.resize(new_len.max(n), 0);
    let m = packet::pad_with_extensions(&mut out, n, new_len, Padding::Shortest, exts).ok()?;
    out.truncate(if m == 0 { n } else { m });
    Some(out)
}

/// A packet of the pool with its DRED payload changed.
fn changed_pool_packet(u: &mut Unstructured) -> Result<Vec<u8>> {
    let pool = &pool().dred;
    let src = &pool[u.choose_index(pool.len())?];
    let p = packet::parse(src, Framing::Normal).unwrap();
    let frames: Vec<&[u8]> = p.frames(src).collect();
    let padding = p.padding(src);
    let mut exts = vec![Extension::default(); extensions::count(padding, p.nb_frames()).unwrap()];
    let n = extensions::parse(padding, &mut exts, p.nb_frames()).unwrap();
    exts.truncate(n);
    let at = exts.iter().position(|e| e.id == DRED_ID).unwrap();
    let mut payload = exts[at].data.to_vec();
    for _ in 0..u.int_in_range(0..=8)? {
        match u.int_in_range(0..=7)? {
            0 => {
                let len = u.int_in_range(0..=payload.len())?;
                payload.truncate(len);
            }
            1 => {
                let n = u.int_in_range(0..=64usize)?.min(u.len());
                payload.extend_from_slice(u.bytes(n)?);
            }
            2 if !payload.is_empty() => {
                let i = u.int_in_range(0..=payload.len().min(2) - 1)?;
                payload[i] = u.arbitrary()?;
            }
            _ if !payload.is_empty() => {
                let i = u.choose_index(payload.len())?;
                payload[i] ^= u.arbitrary::<u8>()?;
            }
            _ => {}
        }
    }
    let frame = if u.ratio(1, 4)? { u.int_in_range(0..=p.nb_frames() - 1)? } else { exts[at].frame };
    exts[at] = Extension::new(DRED_ID, frame, &payload);
    Ok(build(p.toc(), &frames, &exts).unwrap_or_else(|| src.clone()))
}

/// A packet from the input with a DRED extension.
fn made_packet(u: &mut Unstructured) -> Result<Vec<u8>> {
    let toc: u8 = u.arbitrary()?;
    let max = 48 / (packet::samples_per_frame(toc, SampleRate::Hz48000) / 120);
    let count = u.int_in_range(1..=max.min(3))?;
    let mut frames = Vec::with_capacity(count);
    for _ in 0..count {
        let n = u.int_in_range(0..=40usize)?.min(u.len());
        frames.push(u.bytes(n)?);
    }
    let mut payloads = Vec::new();
    for _ in 0..u.int_in_range(1..=2)? {
        let mut payload = if u.ratio(7, 8)? { HEADER.to_vec() } else { vec![u.arbitrary()?, u.arbitrary()?] };
        let n = u.int_in_range(0..=300usize)?.min(u.len());
        payload.extend_from_slice(u.bytes(n)?);
        payloads.push((u.int_in_range(0..=count - 1)?, payload));
    }
    let exts: Vec<Extension> = payloads.iter().map(|(f, d)| Extension::new(DRED_ID, *f, d)).collect();
    Ok(build(toc, &frames, &exts).unwrap_or_default())
}

fn finite(x: &[f32], what: &str) {
    if let Some(i) = x.iter().position(|v| !v.is_finite()) {
        panic!("{what}: sample {i} is {}", x[i]);
    }
}

fn run(mode: u8, u: &mut Unstructured) -> Result<()> {
    let data = match mode % 3 {
        0 => {
            let n = u.int_in_range(0..=1500usize)?.min(u.len());
            u.bytes(n)?.to_vec()
        }
        1 => changed_pool_packet(u)?,
        _ => made_packet(u)?,
    };

    // Parse.
    let parse_rate = *u.choose(SampleRate::ALL)?;
    let parse_fs = parse_rate.hz() as i32;
    let max_samples = match u.int_in_range(0..=5)? {
        0 => u.arbitrary()?,
        _ => u.int_in_range(0..=parse_fs)?,
    };
    let defer: bool = u.arbitrary()?;
    let processing = if defer { DredProcessing::Deferred } else { DredProcessing::Now };
    let dec = dred_decoder();
    let mut dred = OpusDred::new();
    let what = format!("parse of {} bytes, max {max_samples} at {parse_fs} Hz, defer {defer}", data.len());
    let found = match dec.parse(&mut dred, &data, max_samples, parse_rate, processing) {
        Ok(r) => {
            assert!(r.samples >= 0 && r.end >= 0, "{what}: {r:?}");
            if dred.is_parsed() && !defer {
                assert!(dred.is_processed(), "{what}: not processed");
            }
            r
        }
        Err(_) => {
            assert!(!dred.is_parsed(), "{what}: parsed after an error");
            Default::default()
        }
    };
    if dred.is_processed() {
        finite(dred.features(), &what);
    }

    // Process into a second state, and in place.
    let mut processed = OpusDred::new();
    match dec.process(&dred, &mut processed) {
        Ok(()) => {
            assert!(dred.is_parsed(), "{what}: processed data that is not parsed");
            assert!(processed.is_processed(), "{what}: process");
            finite(processed.features(), &what);
            assert_eq!(processed.nb_latents(), dred.nb_latents(), "{what}");
            if dred.is_processed() {
                assert_eq!(processed.features(), dred.features(), "{what}: processed twice");
            }
        }
        Err(e) => {
            assert!(!dred.is_parsed(), "{what}: process failed: {e}");
            assert!(!processed.is_parsed(), "{what}: the target changed after an error");
        }
    }
    if u.arbitrary()? {
        let mut again = dred.clone();
        let _ = dec.process_in_place(&mut again);
        assert_eq!(again.is_processed(), dred.is_parsed(), "{what}: process_in_place");
        if again.is_processed() {
            finite(again.features(), &what);
            assert_eq!(again.features(), processed.features(), "{what}: process_in_place and process");
        }
    }
    let dred = if u.arbitrary()? { &processed } else { &dred };

    // Decode.
    let fs = *u.choose(&RATES)?;
    let channels = u.int_in_range(1..=2)?;
    let mut decoder = OpusDecoder::new(
        SampleRate::try_from(fs).unwrap(),
        Channels::try_from(channels).unwrap(),
    )
    .unwrap();
    decoder.set_complexity(u.int_in_range(0..=10)?).unwrap();
    decoder.set_osce_bwe(u.arbitrary()?).unwrap();
    let max_frame = fs as usize * 120 / 1000;
    let mut out = vec![0f32; max_frame * channels];
    for _ in 0..u.int_in_range(0..=3)? {
        let p = &pool().packets[u.choose_index(pool().packets.len())?];
        if let Ok(n) = decoder.decode(p, max_frame, &mut out) {
            finite(&out[..n * channels], "a packet of the pool");
        }
    }
    let unit = fs as usize / 400;
    for step in 0..u.int_in_range(1..=4)? {
        let frame_size = match u.int_in_range(0..=9)? {
            0 => u.int_in_range(0..=fs as usize / 5)?,
            1 => unit * u.int_in_range(1..=48)?,
            _ => unit * u.int_in_range(1..=8)?,
        };
        let offset = match u.int_in_range(0..=5)? {
            0 => u.arbitrary()?,
            _ => {
                let reach = found.samples.saturating_mul(fs) / parse_fs;
                u.int_in_range(-(fs / 50)..=reach.clamp(0, 10 * fs) + fs / 50)?
            }
        };
        let len = match u.int_in_range(0..=5)? {
            0 => u.int_in_range(0..=frame_size * channels + 8)?,
            1 => frame_size * channels + u.int_in_range(1..=64)?,
            _ => frame_size * channels,
        };
        let what = format!("step {step}: DRED decode of {frame_size} at {offset}, {fs} Hz, {channels} channels, into {len}");
        match u.int_in_range(0..=3)? {
            0 => {
                let mut pcm = vec![0i16; len];
                if let Ok(n) = decoder.dred_decode(dred, offset, frame_size, &mut pcm) {
                    assert_eq!(n, frame_size, "{what}");
                    assert!(n * channels <= len, "{what}");
                }
            }
            1 => {
                let mut pcm = vec![0i32; len];
                if let Ok(n) = decoder.dred_decode24(dred, offset, frame_size, &mut pcm) {
                    assert_eq!(n, frame_size, "{what}");
                    assert!(n * channels <= len, "{what}");
                }
            }
            _ => {
                let mut pcm = vec![f32::from_bits(UNWRITTEN); len];
                if let Ok(n) = decoder.dred_decode_float(dred, offset, frame_size, &mut pcm) {
                    assert_eq!(n, frame_size, "{what}");
                    assert!(n * channels <= len, "{what}");
                    finite(&pcm[..n * channels], &what);
                    assert!(pcm[n * channels..].iter().all(|v| v.to_bits() == UNWRITTEN), "{what}: wrote past the frame");
                }
            }
        }
    }
    // The packet itself after the losses.
    if let Ok(n) = decoder.decode(&data, max_frame, &mut out) {
        assert!(n <= max_frame);
        finite(&out[..n * channels], "the packet after DRED decoding");
    }
    Ok(())
}
