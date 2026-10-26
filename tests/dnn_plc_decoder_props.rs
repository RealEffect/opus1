#![cfg(all(feature = "deep-plc", feature = "std"))]
//! Properties of deep packet loss concealment (deep PLC) inside the
//! decoder that need no libopus: decode calls allocate nothing and fit a
//! small stack, the complexity changes only concealed audio, a reset
//! decoder is a new decoder until deep PLC has run, and long losses fade
//! out. `dnn_plc_decoder_libopus.rs` compares the output with libopus.
//!
//! The binary has its own global allocator, which counts the calls a
//! thread makes while that thread has counting switched on (as
//! `dnn_no_alloc.rs` does).

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

/// 20 ms packets of opus1's encoder at 48 kHz in `mode`.
fn packets(mode: ForcedMode, channels: usize) -> Vec<Vec<u8>> {
    let pcm = speech48(channels);
    let mut enc = OpusEncoder::new(
        SampleRate::Hz48000,
        Channels::try_from(channels).unwrap(),
        Application::Voip,
    )
    .unwrap();
    enc.set_force_mode(Some(mode));
    enc.set_bandwidth(Some(if mode == ForcedMode::SilkOnly { Bandwidth::Wideband } else { Bandwidth::Fullband }));
    enc.set_bitrate(Bitrate::BitsPerSecond(NonZeroU32::new(24000 * channels as u32).unwrap()));
    let mut buf = vec![0u8; 1500];
    pcm.chunks_exact(960 * channels)
        .map(|frame| {
            let n = enc.encode(frame, 960, &mut buf).unwrap();
            buf[..n].to_vec()
        })
        .collect()
}

/// Decodes `packets` with `lost(k)` packets concealed, and returns the
/// output.
fn decode(d: &mut OpusDecoder, packets: &[Vec<u8>], lost: impl Fn(usize) -> bool) -> Vec<f32> {
    let channels = d.channels().count();
    let mut out = Vec::new();
    let mut buf = vec![0f32; 960 * channels];
    for (k, p) in packets.iter().enumerate() {
        let packet: &[u8] = if lost(k) { &[] } else { p };
        let n = d.decode(packet, 960, &mut buf).unwrap();
        out.extend_from_slice(&buf[..n * channels]);
    }
    out
}

fn same(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

const MODES: [ForcedMode; 3] = [ForcedMode::SilkOnly, ForcedMode::Hybrid, ForcedMode::CeltOnly];

/// Decode calls with deep PLC allocate nothing: losses, FEC calls and
/// resets of `OpusDecoder` and `MultistreamDecoder`, mono and stereo.
#[test]
fn decoding_with_deep_plc_allocates_nothing() {
    for mode in MODES {
        for channels in [1, 2] {
            let packets = packets(mode, channels);
            let mut d = OpusDecoder::new(
                SampleRate::Hz48000,
                Channels::try_from(channels).unwrap(),
            )
            .unwrap();
            d.set_complexity(5).unwrap();
            let mut buf = vec![0f32; 960 * channels];
            let mut buf16 = vec![0i16; 960 * channels];
            let n = allocations_of(|| {
                for (k, p) in packets.iter().enumerate() {
                    match k % 11 {
                        3..=5 => d.decode(&[], 960, &mut buf).map(|_| ()).unwrap(),
                        7 => d.decode_fec(p, 960, &mut buf).map(|_| ()).unwrap(),
                        9 => d.decode_i16(&[], 960, &mut buf16).map(|_| ()).unwrap(),
                        _ => d.decode(p, 960, &mut buf).map(|_| ()).unwrap(),
                    }
                    if k == 60 {
                        d.reset_state();
                    }
                }
            });
            assert_eq!(n, 0, "{mode:?} {channels} ch: {n} allocations");
        }
    }
    // Two mono streams that carry the same packets.
    let ms_packets: Vec<Vec<u8>> = packets(ForcedMode::CeltOnly, 1).iter().map(|p| ms_packet(p)).collect();
    let mut ms = MultistreamDecoder::new(SampleRate::Hz48000, 2, 2, 0, &[0, 1]).unwrap();
    ms.set_complexity(5).unwrap();
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

/// Each decode call with deep PLC runs on a thread with a 64 KB stack in a
/// release build (256 KB in a debug build): the first lost frame of SILK
/// wideband and of CELT, mono and stereo, which analyses the whole history
/// and starts FARGAN. On Windows the thread gets more, so that the decoder
/// can use that much (see `thread_stack_size`).
#[test]
fn decoding_with_deep_plc_fits_a_small_stack() {
    let claim = if cfg!(debug_assertions) { 256 * 1024 } else { 64 * 1024 };
    let stack = dnn_plc_common::thread_stack_size(claim);
    let cases: Vec<(ForcedMode, usize, Vec<Vec<u8>>)> =
        MODES.iter().flat_map(|&m| [1, 2].map(|c| (m, c, packets(m, c)))).collect();
    let mut decoders: Vec<OpusDecoder> = cases
        .iter()
        .map(|(_, channels, _)| {
            let mut d = OpusDecoder::new(
                SampleRate::Hz48000,
                Channels::try_from(*channels).unwrap(),
            )
            .unwrap();
            d.set_complexity(5).unwrap();
            d
        })
        .collect();
    std::thread::Builder::new()
        .stack_size(stack)
        .spawn(move || {
            for ((_, channels, packets), d) in cases.iter().zip(&mut decoders) {
                let mut buf = vec![0f32; 960 * channels];
                for (k, p) in packets.iter().enumerate().take(60) {
                    let packet: &[u8] = if k % 20 >= 17 { &[] } else { p };
                    d.decode(packet, 960, &mut buf).unwrap();
                }
            }
        })
        .unwrap()
        .join()
        .unwrap();
}

/// Without loss, deep PLC only appends the decoded frames to its history:
/// a stream of one mode decodes to the same output at every complexity.
/// With the feature `osce`, LACE and NoLACE change SILK wideband and
/// Hybrid frames from complexity 6 (`dnn_osce_decoder_props.rs`), so
/// those are checked at 5 only.
#[test]
fn without_loss_the_complexity_changes_nothing() {
    for mode in MODES {
        for channels in [1, 2] {
            let packets = packets(mode, channels);
            let mut plain = OpusDecoder::new(
                SampleRate::Hz48000,
                Channels::try_from(channels).unwrap(),
            )
            .unwrap();
            let want = decode(&mut plain, &packets, |_| false);
            let complexities: &[i32] =
                if cfg!(feature = "osce") && mode != ForcedMode::CeltOnly { &[5] } else { &[5, 10] };
            for &complexity in complexities {
                let mut d = OpusDecoder::new(
                    SampleRate::Hz48000,
                    Channels::try_from(channels).unwrap(),
                )
                .unwrap();
                d.set_complexity(complexity as u8).unwrap();
                assert!(same(&want, &decode(&mut d, &packets, |_| false)), "{mode:?} {channels} ch, complexity {complexity}");
            }
        }
    }
}

/// With losses, complexity 5 conceals SILK wideband and CELT-only frames
/// with deep PLC (other output than complexity 4), but not the CELT layer
/// of Hybrid frames. Complexity 4 conceals as complexity 0.
#[test]
fn the_complexity_switches_deep_plc() {
    let lost = |k: usize| k > 5 && k.is_multiple_of(9);
    for mode in MODES {
        let packets = packets(mode, 1);
        let run = |complexity| {
            let mut d = OpusDecoder::new(SampleRate::Hz48000, Channels::Mono).unwrap();
            d.set_complexity(complexity).unwrap();
            decode(&mut d, &packets, lost)
        };
        let (c0, c4, c5) = (run(0), run(4), run(5));
        assert!(same(&c0, &c4), "{mode:?}: complexity 4 is not complexity 0");
        assert!(!same(&c4, &c5), "{mode:?}: complexity 5 does not run deep PLC");
    }
}

/// A reset decoder decodes as a new one when deep PLC has not concealed
/// anything yet. After deep PLC has run, the state of FARGAN survives the
/// reset, as in libopus, so the concealment after the reset differs from
/// that of a new decoder.
#[test]
fn a_reset_keeps_only_the_fargan_state() {
    let packets = packets(ForcedMode::SilkOnly, 1);
    let lost = |k: usize| k % 13 == 7;
    let fresh = {
        let mut d = OpusDecoder::new(SampleRate::Hz48000, Channels::Mono).unwrap();
        d.set_complexity(5).unwrap();
        decode(&mut d, &packets, lost)
    };
    // Packets decoded before the reset, no loss: deep PLC has only
    // appended history, which the reset clears.
    let mut d = OpusDecoder::new(SampleRate::Hz48000, Channels::Mono).unwrap();
    d.set_complexity(5).unwrap();
    decode(&mut d, &packets[..20], |_| false);
    d.reset_state();
    assert!(same(&fresh, &decode(&mut d, &packets, lost)), "a reset before any loss");
    // With losses before the reset.
    let mut d = OpusDecoder::new(SampleRate::Hz48000, Channels::Mono).unwrap();
    d.set_complexity(5).unwrap();
    decode(&mut d, &packets[..20], |k| k % 5 == 3);
    d.reset_state();
    assert!(!same(&fresh, &decode(&mut d, &packets, lost)), "FARGAN's state did not survive the reset");
}

/// A loss of 2 s stays finite and no louder than the speech before it:
/// no 20 ms of the loss has more than twice the energy of the loudest
/// 20 ms of the last 400 ms decoded. (Deep PLC lowers its energy feature
/// with the length of the loss; SILK adds comfort noise to concealed
/// frames, and CELT turns to noise after 200 ms, so the level does not go
/// to zero.)
#[test]
fn a_long_loss_stays_bounded() {
    for mode in MODES {
        for channels in [1, 2] {
            let packets = packets(mode, channels);
            let mut d = OpusDecoder::new(
                SampleRate::Hz48000,
                Channels::try_from(channels).unwrap(),
            )
            .unwrap();
            d.set_complexity(5).unwrap();
            let energy = |x: &[f32]| x.iter().map(|&v| (v as f64).powi(2)).sum::<f64>();
            let mut buf = vec![0f32; 960 * channels];
            let mut speech = 0f64;
            for (k, p) in packets[..40].iter().enumerate() {
                let n = d.decode(p, 960, &mut buf).unwrap();
                if k >= 20 {
                    speech = speech.max(energy(&buf[..n * channels]));
                }
            }
            let mut loudest = 0f64;
            for _ in 0..100 {
                let n = d.decode(&[], 960, &mut buf).unwrap();
                let frame = &buf[..n * channels];
                assert!(frame.iter().all(|v| v.is_finite()), "{mode:?} {channels} ch");
                loudest = loudest.max(energy(frame));
            }
            assert!(loudest <= 2.0 * speech, "{mode:?} {channels} ch: {loudest} against {speech}");
        }
    }
}
