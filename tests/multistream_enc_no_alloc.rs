//! The multistream encoder allocates nothing after construction.
//!
//! This test binary replaces the global allocator with one that counts the
//! allocations of each thread. `encode` and `encode_i16` must not add to the
//! count, from the very first frame on: no buffer is grown on first use.
//!
//! The surround layouts run with everything a surround encoder does per
//! frame: the masking analysis reads every channel through the same input
//! as the streams, and every stream gets its energy mask. The test checks
//! that the streams have a mask after the measured frames, so the analysis
//! was part of what was measured.
//!
//! The setters are covered too. `reset_state` is not: `OpusEncoder`
//! builds a new encoder in its `reset_state`, which allocates with the
//! `heap` feature. It is not called per frame.
#![cfg(feature = "std")]


use opus1::multistream::MultistreamEncoder;
use opus1::multistream::MappingType;
use opus1::{Application, Bandwidth, Bitrate, SampleRate};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

thread_local! {
    /// Allocations and reallocations made by this thread.
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

struct CountingAllocator;

fn count_one() {
    // The value is gone while a thread shuts down; nothing is measured then.
    let _ = ALLOCATIONS.try_with(|n| n.set(n.get() + 1));
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count_one();
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        count_one();
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        count_one();
        unsafe { System.realloc(ptr, layout, new_size) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

fn allocations() -> usize {
    ALLOCATIONS.with(|n| n.get())
}

/// A layout: the surround constructor with a mapping family, or the generic
/// constructor.
enum Kind {
    Family(usize, u8),
    Generic(usize, usize, usize, &'static [u8]),
}

fn encoder(kind: &Kind, fs: i32, application: Application) -> MultistreamEncoder {
    match *kind {
        Kind::Family(channels, family) => {
            MultistreamEncoder::new_surround(SampleRate::try_from(fs).unwrap(), channels, family, application)
        }
        Kind::Generic(channels, streams, coupled, mapping) => {
            MultistreamEncoder::new(SampleRate::try_from(fs).unwrap(), channels, streams, coupled, mapping, application)
        }
    }
    .unwrap()
}

/// A signal with a different content on every channel: a sine sweep plus a
/// little noise. Built before anything is measured.
fn signal(fs: i32, channels: usize, samples: usize) -> Vec<f32> {
    let mut state = 12345u32;
    let mut pcm = vec![0f32; samples * channels];
    for (i, frame) in pcm.chunks_exact_mut(channels).enumerate() {
        let t = i as f32 / fs as f32;
        for (c, s) in frame.iter_mut().enumerate() {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            let noise = (state >> 8) as f32 / (1u32 << 24) as f32 - 0.5;
            let hz = 150.0 + 90.0 * c as f32 + 300.0 * t;
            *s = 0.3 * (2.0 * std::f32::consts::PI * hz * t).sin() + 0.02 * noise;
        }
    }
    pcm
}

#[test]
fn encode_does_not_allocate() {
    let kinds = [
        Kind::Family(1, 1),
        Kind::Family(2, 1),
        Kind::Family(3, 1),
        Kind::Family(4, 1),
        Kind::Family(6, 1),
        Kind::Family(7, 1),
        Kind::Family(8, 1),
        Kind::Family(3, 255),
        Kind::Family(6, 2),
        Kind::Generic(4, 3, 1, &[2, 0, 1, 3]),
        Kind::Generic(5, 2, 1, &[0, 1, 255, 2, 2]),
    ];
    let mut frames = 0;
    // Frames coded by an encoder that runs the surround analysis.
    let mut analysed_frames = 0;
    for kind in &kinds {
        for fs in [48_000, 24_000, 16_000] {
            for application in [Application::Audio, Application::Voip] {
                // Frames of 2.5, 10, 20, 40 and 60 ms.
                for frame_size in [fs / 400, fs / 100, fs / 50, fs / 25, 3 * fs / 50] {
                    let frame_size = frame_size as usize;
                    // A low bitrate brings SILK and Hybrid streams, a high
                    // one CELT streams.
                    for (per_channel, vbr) in [(8_000, true), (12_000, false), (64_000, true)] {
                        let mut enc = encoder(kind, fs, application);
                        let channels = enc.layout().channels();
                        let pcm = signal(fs, channels, 6 * frame_size);
                        let pcm16: Vec<i16> = pcm.iter().map(|&s| (s * 32767.0) as i16).collect();
                        let mut out = vec![0u8; 4000 * enc.layout().streams()];
                        let label = format!(
                            "{channels} channels, {fs} Hz, {application:?}, frame {frame_size}, \
                             {per_channel} b/s per channel, vbr {vbr}"
                        );

                        // Everything from here on is measured.
                        let before = allocations();
                        enc.set_bitrate(Bitrate::BitsPerSecond(per_channel * channels as i32))
                            .unwrap();
                        enc.set_vbr(vbr);
                        enc.set_complexity(9).unwrap();
                        let step = frame_size * channels;
                        let mut failed = None;
                        for (f, (frame, frame16)) in pcm
                            .chunks_exact(step)
                            .zip(pcm16.chunks_exact(step))
                            .enumerate()
                        {
                            // Float and 16-bit input in turn, and a change
                            // of settings in the running stream.
                            let result = if f % 2 == 0 {
                                enc.encode(frame, frame_size, &mut out)
                            } else {
                                enc.encode_i16(frame16, frame_size, &mut out)
                            };
                            if f == 3 {
                                enc.set_inband_fec(true);
                                let _ = enc.set_packet_loss_perc(10);
                                enc.set_bandwidth(Bandwidth::Wideband);
                                let _ = enc.set_max_bandwidth(Bandwidth::Superwideband);
                                let _ = enc.set_bitrate(Bitrate::Auto);
                            }
                            let _ = (enc.bitrate(), enc.final_range(), enc.lookahead());
                            if result.is_err() {
                                failed = Some(f);
                                break;
                            }
                            frames += 1;
                        }
                        let after = allocations();
                        // Reported only now, so that a failure message does
                        // not count as an allocation of the encoder.
                        assert_eq!(failed, None, "{label}: encode failed");
                        assert_eq!(after - before, 0, "{label}: allocations");
                        // The analysis ran in the measured frames: every
                        // stream of a surround layout was given a mask.
                        if enc.mapping_type() == MappingType::Surround {
                            for s in 0..enc.layout().streams() {
                                let stream = enc.stream_encoder(s).unwrap();
                                assert_eq!(
                                    stream.energy_mask().map(|m| m.len()),
                                    Some(21 * stream.channels().count()),
                                    "{label}: stream {s} has no mask"
                                );
                            }
                            analysed_frames += 6;
                        }
                    }
                }
            }
        }
    }
    println!("{frames} frames encoded, {analysed_frames} of them with the surround analysis");
    assert!(frames > 5000, "only {frames} frames encoded");
    assert!(
        analysed_frames > 2500,
        "only {analysed_frames} frames with the surround analysis"
    );

    // The counter works: it sees an allocation when there is one.
    let before = allocations();
    let v = vec![0u8; 100];
    assert_eq!(allocations() - before, 1);
    drop(v);
}
