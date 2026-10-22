//! `OpusDecoder` and `MultistreamDecoder` with the neural network features
//! (deep PLC, LACE, NoLACE, the bandwidth extension and DRED).
//!
//! The input picks a decoder: an `OpusDecoder` at a random rate and channel
//! count, or a `MultistreamDecoder` of 1 to 3 streams with a random
//! mapping. Then up to 24 steps, each one of:
//!
//! - decode a packet of opus1's encoder (`common/pool.rs`: SILK wideband
//!   20 ms, Hybrid, SILK in 10 to 60 ms, CELT, with FEC and DRED data),
//!   with `f32` or 16-bit output, as it is or with bytes changed, or the
//!   input as a packet. A multistream decoder gets packets of the pool
//!   joined into one multistream packet;
//! - a lost packet, of a random multiple of 2.5 ms;
//! - `decode_fec` with a packet of the pool;
//! - `set_complexity` with -1 to 11, `set_osce_bwe` on or off,
//!   `reset_state`;
//! - (`OpusDecoder` only) DRED: a packet of the pool with DRED data is
//!   parsed by `OpusDredDecoder`, and `dred_decode_float` conceals a loss at
//!   a random offset.
//!
//! The output buffer is sometimes shorter or longer than the frame.
//!
//! What must hold: nothing panics or hangs. `set_complexity` takes exactly
//! 0 to 10. A decode that succeeds returns at most the room given, fits its
//! buffer, writes no sample past what it returns, and gives finite samples.
//! A lost packet, an FEC decode and a DRED decode that succeed return the
//! requested duration (a multistream decoder at most 120 ms).
//!
//! Needs the `dred` feature of this package; `fuzz.sh` builds the target
//! with it.

#![no_main]

mod common;

use common::RATES;
use common::pool::pool;
use libfuzzer_sys::arbitrary::{Result, Unstructured};
use libfuzzer_sys::fuzz_target;
use opus1::multistream::MultistreamDecoder;
use opus1::packet::{self, Repacketizer};
use opus1::{OpusDecoder, OpusDred, OpusDredDecoder};
use std::sync::OnceLock;

/// A NaN with a payload no decoder writes: marks samples not written.
const UNWRITTEN: u32 = 0x7FC0_5A5A;

fuzz_target!(|data: &[u8]| {
    let mut u = Unstructured::new(data);
    let _ = run(&mut u);
});

fn dred_decoder() -> &'static OpusDredDecoder {
    static DEC: OnceLock<OpusDredDecoder> = OnceLock::new();
    DEC.get_or_init(OpusDredDecoder::new)
}

enum Dec {
    One(OpusDecoder),
    Multi(MultistreamDecoder),
}

impl Dec {
    fn decode(&mut self, p: &[u8], frame_size: usize, out: &mut [f32]) -> std::result::Result<usize, opus1::Error> {
        match self {
            Dec::One(d) => d.decode(p, frame_size, out),
            Dec::Multi(d) => d.decode(p, frame_size, out),
        }
    }

    fn decode_i16(&mut self, p: &[u8], frame_size: usize, out: &mut [i16]) -> std::result::Result<usize, opus1::Error> {
        match self {
            Dec::One(d) => d.decode_i16(p, frame_size, out),
            Dec::Multi(d) => d.decode_i16(p, frame_size, out),
        }
    }

    fn decode_fec(&mut self, p: &[u8], frame_size: usize, out: &mut [f32]) -> std::result::Result<usize, opus1::Error> {
        match self {
            Dec::One(d) => d.decode_fec(p, frame_size, out),
            Dec::Multi(d) => d.decode_fec(p, frame_size, out),
        }
    }

    fn set_complexity(&mut self, c: i32) -> std::result::Result<(), opus1::Error> {
        match self {
            Dec::One(d) => d.set_complexity(c),
            Dec::Multi(d) => d.set_complexity(c),
        }
    }

    fn set_osce_bwe(&mut self, on: bool) -> std::result::Result<(), opus1::Error> {
        match self {
            Dec::One(d) => d.set_osce_bwe(on),
            Dec::Multi(d) => d.set_osce_bwe(on),
        }
    }

    fn reset_state(&mut self) {
        match self {
            Dec::One(d) => d.reset_state(),
            Dec::Multi(d) => d.reset_state(),
        }
    }
}

/// A packet for the decoder: one of the pool, or for a multistream decoder
/// `streams` packets of the pool with the same duration joined, all but
/// the last self-delimited.
fn pool_packet(u: &mut Unstructured, streams: Option<usize>) -> Result<Vec<u8>> {
    let packets = &pool().packets;
    let first = &packets[u.choose_index(packets.len())?];
    let Some(streams) = streams else { return Ok(first.clone()) };
    let duration = packet::nb_samples(first, 48000).unwrap();
    let same: Vec<&Vec<u8>> = packets.iter().filter(|p| packet::nb_samples(p, 48000) == Ok(duration)).collect();
    let mut out = Vec::new();
    for s in 0..streams {
        let p = if s == 0 { first } else { same[u.choose_index(same.len())?] };
        let mut rp = Repacketizer::new();
        rp.cat(p).unwrap();
        let mut buf = vec![0u8; 2 * p.len() + 64];
        let n = rp.out_range_ext(0, rp.nb_frames(), &mut buf, s + 1 < streams, false, &[]).unwrap();
        out.extend_from_slice(&buf[..n]);
    }
    Ok(out)
}

fn check_f32(out: &[f32], n: usize, channels: usize, what: &str) {
    assert!(n * channels <= out.len(), "{what}: {n} samples into {}", out.len());
    if let Some(i) = out[..n * channels].iter().position(|v| !v.is_finite()) {
        panic!("{what}: sample {i} is {}", out[i]);
    }
    assert!(out[n * channels..].iter().all(|v| v.to_bits() == UNWRITTEN), "{what}: wrote past {n} samples");
}

fn run(u: &mut Unstructured) -> Result<()> {
    let fs = *u.choose(&RATES)?;
    let (mut dec, channels, streams) = if u.ratio(1, 4)? {
        let streams = u.int_in_range(1..=3)?;
        let coupled = u.int_in_range(0..=streams)?;
        let channels = u.int_in_range(1..=4)?;
        let mut mapping = Vec::with_capacity(channels);
        for _ in 0..channels {
            mapping.push(if u.ratio(1, 10)? { 255 } else { u.int_in_range(0..=(streams + coupled - 1) as u8)? });
        }
        let Ok(d) = MultistreamDecoder::with_layout(fs, channels, streams, coupled, &mapping) else { return Ok(()) };
        (Dec::Multi(d), channels, Some(streams))
    } else {
        let channels = u.int_in_range(1..=2)?;
        (Dec::One(OpusDecoder::new(fs, channels).unwrap()), channels, None)
    };
    // Mostly the complexities that run the networks.
    dec.set_complexity(if u.ratio(3, 4)? { u.int_in_range(5..=10)? } else { u.int_in_range(0..=4)? }).unwrap();
    dec.set_osce_bwe(u.arbitrary()?).unwrap();
    let max_frame = fs as usize * 120 / 1000;
    let unit = fs as usize / 400;

    for step in 0..u.int_in_range(1..=24)? {
        // The room for the frame, and the buffer.
        let room = match u.int_in_range(0..=9)? {
            0 => u.int_in_range(0..=max_frame + unit)?,
            1 => unit * u.int_in_range(1..=8)?,
            _ => max_frame,
        };
        let len = match u.int_in_range(0..=9)? {
            0 => u.int_in_range(0..=room * channels + 8)?,
            1 => room * channels + u.int_in_range(1..=64)?,
            _ => room * channels,
        };
        let mut out = vec![f32::from_bits(UNWRITTEN); len];
        let what = format!("step {step}: {fs} Hz, {channels} channels, room {room}, buffer {len}");
        match u.int_in_range(0..=15)? {
            0..=4 => {
                let p = pool_packet(u, streams)?;
                if let Ok(n) = dec.decode(&p, room, &mut out) {
                    assert!(n <= room, "{what}: {n} samples");
                    check_f32(&out, n, channels, &what);
                }
            }
            5 => {
                // A packet of the pool with bytes changed.
                let mut p = pool_packet(u, streams)?;
                for _ in 0..u.int_in_range(1..=8)? {
                    let i = u.choose_index(p.len())?;
                    p[i] ^= u.arbitrary::<u8>()?;
                }
                if let Ok(n) = dec.decode(&p, room, &mut out) {
                    assert!(n <= room, "{what}: {n} samples");
                    check_f32(&out, n, channels, &format!("{what}: changed packet"));
                }
            }
            6 => {
                let n = u.int_in_range(0..=400usize)?.min(u.len());
                let p = u.bytes(n)?;
                if let Ok(n) = dec.decode(p, room, &mut out) {
                    assert!(n <= room, "{what}: {n} samples");
                    check_f32(&out, n, channels, &format!("{what}: input packet"));
                }
            }
            7 | 8 => {
                // A lost packet.
                let lost = unit * u.int_in_range(1..=48)?;
                let lost = if u.ratio(1, 10)? { lost + 1 } else { lost };
                let mut out = vec![f32::from_bits(UNWRITTEN); lost * channels + if u.ratio(1, 4)? { 16 } else { 0 }];
                // A multistream decoder first limits the duration to
                // 120 ms, as libopus does.
                let want = if streams.is_some() { lost.min(max_frame) } else { lost };
                if let Ok(n) = dec.decode(&[], lost, &mut out) {
                    assert_eq!(n, want, "{what}: loss of {lost}");
                    check_f32(&out, n, channels, &format!("{what}: loss of {lost}"));
                }
            }
            9 => {
                let p = pool_packet(u, streams)?;
                let fec = match packet::nb_samples(&p, fs) {
                    Ok(d) if u.ratio(3, 4)? => d,
                    _ => unit * u.int_in_range(1..=48)?,
                };
                let mut out = vec![f32::from_bits(UNWRITTEN); fec * channels];
                if let Ok(n) = dec.decode_fec(&p, fec, &mut out) {
                    assert_eq!(n, fec, "{what}: FEC of {fec}");
                    check_f32(&out, n, channels, &format!("{what}: FEC of {fec}"));
                }
            }
            10 => {
                let c = u.int_in_range(-1..=11)?;
                assert_eq!(dec.set_complexity(c).is_ok(), (0..=10).contains(&c), "complexity {c}");
            }
            11 => dec.set_osce_bwe(u.arbitrary()?).unwrap(),
            12 => {
                let p = pool_packet(u, streams)?;
                let mut pcm = vec![0i16; len];
                if let Ok(n) = dec.decode_i16(&p, room, &mut pcm) {
                    assert!(n <= room && n * channels <= len, "{what}: {n} samples, 16 bits");
                }
            }
            13 => {
                if u.ratio(1, 4)? {
                    dec.reset_state();
                }
            }
            _ => {
                // DRED: conceal a loss from the DRED data of a packet.
                let Dec::One(d) = &mut dec else { continue };
                let dred_packets = &pool().dred;
                let p = &dred_packets[u.choose_index(dred_packets.len())?];
                let mut dred = OpusDred::new();
                let max = u.int_in_range(0..=fs)?;
                let Ok(found) = dred_decoder().parse(&mut dred, p, max, fs, u.arbitrary()?) else { continue };
                if !dred.is_processed() && u.arbitrary()? {
                    let _ = dred_decoder().process_in_place(&mut dred);
                }
                let lost = unit * u.int_in_range(1..=8)?;
                let offset = u.int_in_range(-(unit as i32)..=found.samples + unit as i32)?;
                let mut out = vec![f32::from_bits(UNWRITTEN); lost * channels];
                if let Ok(n) = d.dred_decode_float(&dred, offset, lost, &mut out) {
                    assert_eq!(n, lost, "{what}: DRED decode of {lost} at {offset}");
                    check_f32(&out, n, channels, &format!("{what}: DRED decode of {lost} at {offset}"));
                }
            }
        }
    }
    Ok(())
}
