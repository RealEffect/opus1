//! `MultistreamDecoder` allocates memory in its constructor only.
//!
//! This test binary has its own global allocator, which counts the calls a
//! thread makes while that thread has counting switched on. The test builds
//! the decoders and the packets first, switches counting on, and then runs
//! every decoding function: on the first packet, on SILK, Hybrid and CELT
//! packets, on lost packets, on malformed packets, and after a reset.

mod ms_support;
mod multistream_dec_common;

use multistream_dec_common::*;
use std::alloc::{GlobalAlloc, Layout as MemoryLayout, System};
use std::cell::Cell;

thread_local! {
    /// Whether this thread counts its allocator calls.
    static COUNTING: Cell<bool> = const { Cell::new(false) };
    /// Calls that take memory: `alloc`, `alloc_zeroed`, `realloc`.
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
    /// Calls that give memory back.
    static FREES: Cell<usize> = const { Cell::new(0) };
}

struct CountingAllocator;

fn count(counter: &'static std::thread::LocalKey<Cell<usize>>) {
    // `try_with`: the allocator also runs while a thread shuts down, when
    // its thread-local values are gone.
    if COUNTING.try_with(|c| c.get()).unwrap_or(false) {
        let _ = counter.try_with(|n| n.set(n.get() + 1));
    }
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: MemoryLayout) -> *mut u8 {
        count(&ALLOCATIONS);
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: MemoryLayout) -> *mut u8 {
        count(&ALLOCATIONS);
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: MemoryLayout, new_size: usize) -> *mut u8 {
        count(&ALLOCATIONS);
        unsafe { System.realloc(ptr, layout, new_size) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: MemoryLayout) {
        count(&FREES);
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

/// Runs `work` with counting on and returns the allocator calls it made:
/// (calls that take memory, calls that give memory back).
fn allocator_calls(work: impl FnOnce()) -> (usize, usize) {
    ALLOCATIONS.with(|n| n.set(0));
    FREES.with(|n| n.set(0));
    COUNTING.with(|c| c.set(true));
    work();
    COUNTING.with(|c| c.set(false));
    (ALLOCATIONS.with(|n| n.get()), FREES.with(|n| n.get()))
}

#[test]
fn the_counter_sees_an_allocation() {
    let (allocations, frees) = allocator_calls(|| {
        let v = std::hint::black_box(vec![1u8; 100]);
        drop(v);
    });
    assert_eq!((allocations, frees), (1, 1));
    let (allocations, frees) = allocator_calls(|| {
        std::hint::black_box([1u8; 100]);
    });
    assert_eq!((allocations, frees), (0, 0));
}

#[test]
fn decoding_does_not_allocate() {
    // (layout, settings, bitrate per channel, frame size at 48 kHz)
    let cases = [
        // CELT streams.
        (SURROUND_5_1, AUDIO, 32, 960usize),
        (SURROUND_7_1, AUDIO, 24, 120),
        // SILK streams with error correction data.
        (DISCRETE_3, VOIP_FEC, 12, 2880),
        // SILK, Hybrid and CELT streams, and changes between them.
        (DISCRETE_5, VOIP_FEC, 24, 480),
        (MONO, VOIP_FEC, 16, 960),
        (Layout::Generic, AUDIO, 64, 1920),
        // Constant bitrate: padded packets.
        (AMBISONICS_4, AUDIO_CBR, 32, 240),
    ];
    let mut calls = 0usize;
    for (layout, settings, kbps, frame) in cases {
        // A signal beyond full scale, so that the soft clip of the 16-bit
        // output has work.
        // At least 25 packets.
        let milliseconds = (frame * 25 / 48).max(600);
        let e = encode(layout, settings, kbps, frame, Signal::Voiced, milliseconds, 5.0);
        assert!(!e.has_mono_coded_stereo_stream(), "{}", e.label);
        for fs in [8000, 16000, 48000] {
            // Everything that needs memory is made before counting starts:
            // four decoders, one per decoding function, and the outputs.
            let (_, mut float_dec) = e.decoders(fs);
            let (_, mut i16_dec) = e.decoders(fs);
            let (_, mut fec_dec) = e.decoders(fs);
            let (_, mut fec_i16_dec) = e.decoders(fs);
            let frame_dec = frame * fs as usize / FS as usize;
            let max_frame = fs as usize / 25 * 3;
            let mut out = vec![0f32; max_frame * e.channels];
            let mut out16 = vec![0i16; max_frame * e.channels];
            let garbage = [0xFFu8; 3];
            let mut done = 0usize;
            let mut samples = 0usize;

            let (allocations, frees) = allocator_calls(|| {
                for round in 0..2 {
                    for (i, packet) in e.packets.iter().enumerate() {
                        // Every 4th packet is lost.
                        if i % 4 == 3 {
                            // Concealed by two of the decoders.
                            samples += float_dec.decode(&[], frame_dec, &mut out).unwrap();
                            samples += i16_dec.decode_i16(&[], frame_dec, &mut out16).unwrap();
                            done += 2;
                            continue;
                        }
                        if i % 4 == 0 && i > 0 {
                            // Rebuilt by the other two from the packet after it.
                            samples += fec_dec.decode_fec(packet, frame_dec, &mut out).unwrap();
                            samples += fec_i16_dec.decode_fec_i16(packet, frame_dec, &mut out16).unwrap();
                            done += 2;
                        }
                        samples += float_dec.decode(packet, max_frame, &mut out).unwrap();
                        samples += i16_dec.decode_i16(packet, max_frame, &mut out16).unwrap();
                        samples += fec_dec.decode(packet, max_frame, &mut out).unwrap();
                        samples += fec_i16_dec.decode_i16(packet, max_frame, &mut out16).unwrap();
                        done += 4;
                        // Calls that fail: a malformed packet, too little
                        // room, a lost duration that is no multiple of 2.5 ms.
                        assert!(float_dec.decode(&garbage, max_frame, &mut out).is_err());
                        assert!(i16_dec.decode_i16(packet, frame_dec / 2, &mut out16).is_err());
                        assert!(fec_dec.decode_fec(&[], frame_dec + 1, &mut out).is_err());
                        assert!(fec_i16_dec.decode_fec_i16(packet, 0, &mut out16).is_err());
                        std::hint::black_box(float_dec.final_range());
                        std::hint::black_box((float_dec.layout().channels(), float_dec.sampling_rate()));
                        std::hint::black_box(float_dec.stream_decoder(0).is_some());
                    }
                    // The second round starts from a reset decoder.
                    if round == 0 {
                        float_dec.reset_state();
                        i16_dec.reset_state();
                        fec_dec.reset_state();
                        fec_i16_dec.reset_state();
                    }
                }
            });
            assert!(done >= 100 && samples > 0, "{}: {done} calls", e.label);
            assert_eq!(
                (allocations, frees),
                (0, 0),
                "{} at {fs} Hz: {allocations} allocations and {frees} frees in {done} decoding calls",
                e.label
            );
            calls += done;
        }
    }
    println!("{calls} decoding calls without an allocation");
}
