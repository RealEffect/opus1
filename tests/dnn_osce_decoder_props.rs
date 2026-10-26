#![cfg(all(feature = "osce-bwe", feature = "std"))]
//! Properties of the Opus Speech Coding Enhancement (OSCE) inside the
//! decoder that need no libopus: LACE (complexity 6), NoLACE (7 and
//! above) and the bandwidth extension (BWE). `dnn_osce_decoder_libopus.rs`
//! compares the output with libopus.
//!
//! - Decode calls allocate nothing and fit a small stack.
//! - OSCE changes only SILK frames at 16 kHz: LACE and NoLACE only 20 ms
//!   frames, BWE also 10 ms frames, and BWE only at 48 kHz output and from
//!   complexity 4.
//! - CELT-only frames after frames in BWE mode are decoded. libopus 1.6.1
//!   outputs silence there; opus1 has the fix of libopus main (commit
//!   228a0f85). Their output is that of a decoder without BWE.
//! - The final range does not depend on OSCE.
//! - `MultistreamDecoder::set_osce_bwe` reaches every stream.
//!
//! The binary has its own global allocator, which counts the calls a
//! thread makes while that thread has counting switched on (as
//! `dnn_plc_decoder_props.rs` does).

mod dnn_plc_common;

use std::num::NonZeroU32;
use opus1::multistream::MultistreamDecoder;
use opus1::{
    Application, Bandwidth, Bitrate, Channels, ForcedMode, OpusDecoder, OpusEncoder, SampleRate,
};
use std::alloc::{GlobalAlloc, Layout as MemoryLayout, System};
use std::cell::Cell;

thread_local! {
    /// Whether this thread counts its allocator calls.
    static COUNTING: Cell<bool> = const { Cell::new(false) };
    /// Calls that take memory: `alloc`, `alloc_zeroed`, `realloc`.
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

struct CountingAllocator;

fn count() {
    if COUNTING.try_with(|c| c.get()).unwrap_or(false) {
        let _ = ALLOCATIONS.try_with(|n| n.set(n.get() + 1));
    }
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: MemoryLayout) -> *mut u8 {
        count();
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: MemoryLayout) -> *mut u8 {
        count();
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: MemoryLayout, new_size: usize) -> *mut u8 {
        count();
        unsafe { System.realloc(ptr, layout, new_size) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: MemoryLayout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

/// The allocator calls `f` makes on this thread.
fn allocations_of(f: impl FnOnce()) -> usize {
    ALLOCATIONS.with(|n| n.set(0));
    COUNTING.with(|c| c.set(true));
    f();
    COUNTING.with(|c| c.set(false));
    ALLOCATIONS.with(|n| n.get())
}

/// 3 s of the speech fixture at 48 kHz, half of full scale, in `channels`
/// channels (the second one delayed).
fn speech48(channels: usize) -> Vec<f32> {
    let x16 = dnn_plc_common::fixture();
    let mut out = Vec::new();
    for i in 0..3 * 48000 {
        for c in 0..channels {
            let k = (i + 997 * c) / 3;
            out.push(0.5 * x16[k] as f32 / 32768.0);
        }
    }
    out
}

/// How a test stream is coded by opus1's encoder at 48 kHz.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Kind {
    /// SILK wideband, 20 ms.
    SilkWb,
    /// SILK wideband, 10 ms.
    SilkWb10,
    /// SILK narrowband, 20 ms.
    SilkNb,
    /// SILK mediumband, 20 ms.
    SilkMb,
    /// Hybrid fullband, 20 ms.
    Hybrid,
    /// CELT-only fullband, 20 ms.
    Celt,
    /// SILK wideband and CELT-only wideband, switching every 15 packets.
    /// The encoder makes the switches with redundancy frames.
    SilkCelt,
}

/// The packets of a stream of `kind`.
fn packets(kind: Kind, channels: usize) -> Vec<Vec<u8>> {
    let pcm = speech48(channels);
    let mut enc = OpusEncoder::new(
        SampleRate::Hz48000,
        Channels::try_from(channels).unwrap(),
        Application::Audio,
    )
    .unwrap();
    let ch = channels as i32;
    let (mode, bandwidth, bitrate, frame) = match kind {
        Kind::SilkWb | Kind::SilkCelt => (ForcedMode::SilkOnly, Bandwidth::Wideband, 16000 * ch, 960),
        Kind::SilkWb10 => (ForcedMode::SilkOnly, Bandwidth::Wideband, 16000 * ch, 480),
        Kind::SilkNb => (ForcedMode::SilkOnly, Bandwidth::Narrowband, 10000 * ch, 960),
        Kind::SilkMb => (ForcedMode::SilkOnly, Bandwidth::Mediumband, 12000 * ch, 960),
        Kind::Hybrid => (ForcedMode::Hybrid, Bandwidth::Fullband, 32000 * ch, 960),
        Kind::Celt => (ForcedMode::CeltOnly, Bandwidth::Fullband, 48000 * ch, 960),
    };
    enc.set_bandwidth(Some(bandwidth));
    let mut buf = vec![0u8; 1500];
    pcm.chunks_exact(frame * channels)
        .enumerate()
        .map(|(k, f)| {
            let celt = kind == Kind::SilkCelt && (k / 15) % 2 == 1;
            enc.set_force_mode(Some(if celt { ForcedMode::CeltOnly } else { mode }));
            enc.set_bitrate(Bitrate::BitsPerSecond(NonZeroU32::new((if celt { 48000 * ch } else { bitrate }) as u32).unwrap()));
            let n = enc.encode(f, frame, &mut buf).unwrap();
            buf[..n].to_vec()
        })
        .collect()
}

/// Whether a packet is CELT-only, from its first byte.
fn is_celt(packet: &[u8]) -> bool {
    packet[0] & 0x80 != 0
}

/// A decoder at `rate` with `complexity` and the BWE switch.
fn decoder(rate: i32, channels: usize, complexity: i32, bwe: bool) -> OpusDecoder {
    let mut d = OpusDecoder::new(
        SampleRate::try_from(rate).unwrap(),
        Channels::try_from(channels).unwrap(),
    )
    .unwrap();
    d.set_complexity(complexity as u8).unwrap();
    d.set_osce_bwe(bwe);
    d
}

/// Decodes every packet, `lost(k)` packets concealed. Returns the output
/// of each call and the final range after it.
fn decode(d: &mut OpusDecoder, packets: &[Vec<u8>], lost: impl Fn(usize) -> bool) -> Vec<(Vec<f32>, u32)> {
    let channels = d.channels().count();
    let rate = d.sampling_rate();
    let mut buf = vec![0f32; 960 * channels];
    packets
        .iter()
        .enumerate()
        .map(|(k, p)| {
            let n = opus1::packet::nb_samples(p, rate).unwrap();
            let packet: &[u8] = if lost(k) { &[] } else { p };
            let got = d.decode(packet, n, &mut buf).unwrap();
            (buf[..got * channels].to_vec(), d.final_range())
        })
        .collect()
}

fn same(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

/// The output of a whole decoding, as one signal.
fn signal(calls: &[(Vec<f32>, u32)]) -> Vec<f32> {
    calls.iter().flat_map(|(x, _)| x.iter().copied()).collect()
}

const KINDS: [Kind; 7] = [Kind::SilkWb, Kind::SilkWb10, Kind::SilkNb, Kind::SilkMb, Kind::Hybrid, Kind::Celt, Kind::SilkCelt];

/// Decode calls with LACE, NoLACE and BWE allocate nothing: losses, FEC
/// calls, resets, complexity and BWE changes, of `OpusDecoder` and
/// `MultistreamDecoder`, mono and stereo.
#[test]
fn decoding_with_osce_allocates_nothing() {
    for kind in KINDS {
        for channels in [1, 2] {
            let packets = packets(kind, channels);
            let mut d = decoder(48000, channels, 7, true);
            let mut buf = vec![0f32; 960 * channels];
            let mut buf16 = vec![0i16; 960 * channels];
            let n = allocations_of(|| {
                for (k, p) in packets.iter().enumerate() {
                    let n = opus1::packet::nb_samples(p, SampleRate::Hz48000).unwrap();
                    match k % 11 {
                        3..=5 => d.decode(&[], n, &mut buf).map(|_| ()).unwrap(),
                        7 => d.decode_fec(p, n, &mut buf).map(|_| ()).unwrap(),
                        9 => d.decode_i16(&[], n, &mut buf16).map(|_| ()).unwrap(),
                        _ => d.decode(p, n, &mut buf).map(|_| ()).unwrap(),
                    }
                    match k {
                        40 => d.reset_state(),
                        60 => d.set_complexity(6).unwrap(),
                        80 => d.set_osce_bwe(false),
                        100 => {
                            d.set_complexity(10).unwrap();
                            d.set_osce_bwe(true);
                        }
                        _ => {}
                    }
                }
            });
            assert_eq!(n, 0, "{kind:?} {channels} ch: {n} allocations");
        }
    }
    // Two mono streams that carry the same packets.
    let ms_packets: Vec<Vec<u8>> = packets(Kind::SilkWb, 1).iter().map(|p| ms_packet(p)).collect();
    let mut ms = MultistreamDecoder::new(SampleRate::Hz48000, 2, 2, 0, &[0, 1]).unwrap();
    ms.set_complexity(7).unwrap();
    ms.set_osce_bwe(true);
    let mut buf = vec![0f32; 960 * 2];
    let n = allocations_of(|| {
        for (k, p) in ms_packets.iter().enumerate() {
            let packet: &[u8] = if k % 5 == 2 { &[] } else { p };
            ms.decode(packet, 960, &mut buf).unwrap();
        }
    });
    assert_eq!(n, 0, "multistream: {n} allocations");
}

/// A packet of two mono streams that both carry `p`: the first one
/// self-delimited (RFC 6716 appendix B), the second one as it is.
fn ms_packet(p: &[u8]) -> Vec<u8> {
    let len = p.len() - 1;
    let mut out = vec![p[0]];
    if len < 252 {
        out.push(len as u8);
    } else {
        let b0 = 252 + (len & 3);
        out.push(b0 as u8);
        out.push(((len - b0) >> 2) as u8);
    }
    out.extend_from_slice(&p[1..]);
    out.extend_from_slice(p);
    out
}

/// Decode calls with NoLACE, BWE and deep PLC run on a thread with a
/// 64 KB stack in a release build (256 KB in a debug build): SILK
/// wideband, Hybrid and the switch to CELT-only, mono and stereo, with
/// losses. On Windows the thread gets more, so that the decoder can use
/// that much (see `thread_stack_size`).
#[test]
fn decoding_with_osce_fits_a_small_stack() {
    let claim = if cfg!(debug_assertions) { 256 * 1024 } else { 64 * 1024 };
    let stack = dnn_plc_common::thread_stack_size(claim);
    let kinds = [Kind::SilkWb, Kind::SilkWb10, Kind::Hybrid, Kind::SilkCelt];
    let cases: Vec<(usize, Vec<Vec<u8>>)> = kinds.iter().flat_map(|&k| [1, 2].map(|c| (c, packets(k, c)))).collect();
    let mut decoders: Vec<OpusDecoder> = cases.iter().map(|(channels, _)| decoder(48000, *channels, 10, true)).collect();
    std::thread::Builder::new()
        .stack_size(stack)
        .spawn(move || {
            for ((channels, packets), d) in cases.iter().zip(&mut decoders) {
                let mut buf = vec![0f32; 960 * channels];
                for (k, p) in packets.iter().enumerate().take(50) {
                    let n = opus1::packet::nb_samples(p, SampleRate::Hz48000).unwrap();
                    let packet: &[u8] = if k % 20 >= 17 { &[] } else { p };
                    d.decode(packet, n, &mut buf).unwrap();
                }
            }
        })
        .unwrap()
        .join()
        .unwrap();
}

/// Without loss, LACE and NoLACE change only SILK frames of 20 ms at
/// 16 kHz: SILK wideband and Hybrid. SILK narrowband and mediumband,
/// 10 ms SILK frames and CELT decode at complexity 6 and 7 as at 0.
#[test]
fn lace_and_nolace_change_only_20_ms_silk_frames_at_16_khz() {
    for kind in KINDS {
        for channels in [1, 2] {
            let packets = packets(kind, channels);
            let plain = signal(&decode(&mut decoder(48000, channels, 0, false), &packets, |_| false));
            for complexity in [6, 7] {
                let osce = signal(&decode(&mut decoder(48000, channels, complexity, false), &packets, |_| false));
                let changes = matches!(kind, Kind::SilkWb | Kind::Hybrid | Kind::SilkCelt);
                assert_eq!(!same(&plain, &osce), changes, "{kind:?} {channels} ch, complexity {complexity}");
            }
        }
    }
}

/// BWE runs only at 48 kHz output, from complexity 4, on SILK at 16 kHz
/// (also 10 ms frames) and not on Hybrid frames that arrive.
#[test]
fn bwe_needs_48_khz_complexity_4_and_silk_wideband() {
    for kind in KINDS {
        for channels in [1, 2] {
            let packets = packets(kind, channels);
            for rate in [8000, 12000, 16000, 24000, 48000] {
                for complexity in [3, 4, 7] {
                    let off = signal(&decode(&mut decoder(rate, channels, complexity, false), &packets, |_| false));
                    let on = signal(&decode(&mut decoder(rate, channels, complexity, true), &packets, |_| false));
                    let changes =
                        rate == 48000 && complexity >= 4 && matches!(kind, Kind::SilkWb | Kind::SilkWb10 | Kind::SilkCelt);
                    assert_eq!(!same(&off, &on), changes, "{kind:?} {channels} ch, {rate} Hz, complexity {complexity}");
                }
            }
        }
    }
}

/// libopus 1.6.1 outputs silence for CELT-only frames after frames in BWE
/// mode, until the next SILK frame. opus1 decodes them, as libopus main
/// does: with BWE on, each CELT-only packet decodes as with BWE off. Only
/// the first 5 ms after a switch without a redundancy frame differ, which
/// fade from 5 ms of SILK concealment (extended by BWE) into the CELT
/// frame. Lost CELT-only packets are concealed as with BWE off too.
#[test]
fn celt_only_frames_after_bwe_frames_are_decoded() {
    for channels in [1, 2] {
        // With redundancy frames (one encoder), and without them (two
        // encoders, spliced).
        let switched = packets(Kind::SilkCelt, channels);
        let (silk, celt) = (packets(Kind::SilkWb, channels), packets(Kind::Celt, channels));
        let spliced: Vec<Vec<u8>> = (0..silk.len()).map(|k| if (k / 15) % 2 == 1 { celt[k].clone() } else { silk[k].clone() }).collect();
        for (label, stream) in [("switched", &switched), ("spliced", &spliced)] {
            let lost = |k: usize| k % 15 == 9;
            let off = decode(&mut decoder(48000, channels, 7, false), stream, lost);
            let on = decode(&mut decoder(48000, channels, 7, true), stream, lost);
            assert!(!same(&signal(&off), &signal(&on)), "{label}: BWE did not run");
            let f5 = 240 * channels;
            let mut celt_calls = 0;
            for k in 0..stream.len() {
                let (want, got) = (&off[k], &on[k]);
                assert_eq!(want.1, got.1, "{label} {channels} ch, packet {k}: final range");
                if !is_celt(&stream[k]) {
                    continue;
                }
                celt_calls += 1;
                // A switch from SILK without redundancy.
                let from = if k > 0 && !is_celt(&stream[k - 1]) && label == "spliced" { f5 } else { 0 };
                assert!(same(&want.0[from..], &got.0[from..]), "{label} {channels} ch, packet {k}");
                assert!(got.0.iter().any(|&v| v != 0.0), "{label} {channels} ch, packet {k}: silence");
            }
            assert!(celt_calls >= 60, "{label}: {celt_calls} CELT-only packets");
        }
    }
}

/// OSCE does not read the range decoder, so the final range of every
/// call is the same at complexity 0 without BWE and at 6, 7 and 10 with
/// BWE, with losses (a concealed call has 0).
#[test]
fn the_final_range_does_not_depend_on_osce() {
    for kind in KINDS {
        for channels in [1, 2] {
            let packets = packets(kind, channels);
            let lost = |k: usize| k % 7 == 4;
            let ranges = |complexity, bwe| -> Vec<u32> {
                decode(&mut decoder(48000, channels, complexity, bwe), &packets, lost).into_iter().map(|(_, r)| r).collect()
            };
            let want = ranges(0, false);
            for complexity in [6, 7, 10] {
                assert_eq!(want, ranges(complexity, true), "{kind:?} {channels} ch, complexity {complexity}");
            }
        }
    }
}

/// `MultistreamDecoder::set_osce_bwe` and `set_complexity` reach every
/// stream: two mono streams carrying the same packets decode, channel by
/// channel, as one `OpusDecoder` with the same settings.
#[test]
fn the_multistream_decoder_forwards_bwe_to_every_stream() {
    let packets = packets(Kind::SilkWb, 1);
    for (complexity, bwe) in [(7, true), (4, true), (6, false)] {
        let want = signal(&decode(&mut decoder(48000, 1, complexity, bwe), &packets, |k| k % 9 == 4));
        let mut ms =
            MultistreamDecoder::new(SampleRate::Hz48000, 2, 2, 0, &[0, 1]).unwrap();
        ms.set_complexity(complexity as u8).unwrap();
        ms.set_osce_bwe(bwe);
        let mut buf = vec![0f32; 960 * 2];
        let mut got = [Vec::new(), Vec::new()];
        for (k, p) in packets.iter().enumerate() {
            let packet = if k % 9 == 4 { Vec::new() } else { ms_packet(p) };
            let n = ms.decode(&packet, 960, &mut buf).unwrap();
            for i in 0..n {
                got[0].push(buf[2 * i]);
                got[1].push(buf[2 * i + 1]);
            }
        }
        assert!(same(&want, &got[0]) && same(&want, &got[1]), "complexity {complexity}, BWE {bwe}");
    }
}
