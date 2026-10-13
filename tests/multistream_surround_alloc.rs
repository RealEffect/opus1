//! The surround masking analysis allocates nothing per frame.
//!
//! All its buffers are made in `new`. This file replaces the allocator of
//! the test program with one that counts, and looks at the count around the
//! calls of `run`. The count is kept per thread, so tests that run at the
//! same time do not disturb each other.

mod ms_support;

use ms_support::noise;
use opus_rs::multistream::surround::SurroundProbe;
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

thread_local! {
    /// Allocations and reallocations made by this thread.
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

fn count_one() {
    // `try_with` because a thread can still free memory while it ends.
    let _ = ALLOCATIONS.try_with(|n| n.set(n.get() + 1));
}

fn allocations() -> usize {
    ALLOCATIONS.with(|n| n.get())
}

/// The system allocator, counting every request for memory.
struct Counting;

unsafe impl GlobalAlloc for Counting {
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
static ALLOCATOR: Counting = Counting;

const NB_BANDS: usize = 21;

#[test]
fn the_counter_sees_allocations() {
    let before = allocations();
    let v = std::hint::black_box(vec![1u8; 100]);
    assert!(allocations() > before);
    drop(v);
}

#[test]
fn run_allocates_nothing() {
    // The shortest and the longest frame, rates with and without inserted
    // zeros, the smallest and the largest layout.
    for (channels, rate, frame_size) in [
        (6usize, 48_000, 960usize),
        (8, 48_000, 5760),
        (8, 48_000, 120),
        (3, 8_000, 20),
        (7, 16_000, 1920),
        (5, 12_000, 240),
        (4, 24_000, 1440),
    ] {
        let frames = 4;
        let mut pcm = noise(channels, frames * frame_size, 0.2, 157);
        // A sample that makes libopus, and this port, clear a channel.
        pcm[(2 * frame_size + 7) * channels + 1] = f32::NAN;
        let mut mask = vec![0f32; NB_BANDS * channels];
        let mut probe = SurroundProbe::new(channels, rate);
        let before = allocations();
        let mut ok = true;
        for frame in pcm.chunks_exact(frame_size * channels) {
            ok &= probe.run(frame, frame_size, &mut mask);
            // Calls that give no mask.
            ok &= !probe.run(frame, frame_size + 1, &mut mask);
            ok &= !probe.run(frame, frame_size, &mut mask[..NB_BANDS]);
        }
        probe.reset();
        let made = allocations() - before;
        assert!(ok, "{channels} channels, {rate} Hz, frame {frame_size}");
        assert_eq!(
            made, 0,
            "{channels} channels, {rate} Hz, frame {frame_size}: {made} allocations in {frames} frames"
        );
    }
}

#[test]
fn an_analysis_that_cannot_run_holds_no_buffers() {
    // 255 channels, as the largest multistream layout has, or a sampling
    // rate CELT does not take: no mask, and no memory set aside for one.
    for (channels, rate) in [
        (255usize, 48_000),
        (2, 48_000),
        (6, 44_100),
        (usize::MAX, 8_000),
    ] {
        let before = allocations();
        let mut probe = SurroundProbe::new(channels, rate);
        assert!(!probe.run(&[], 960, &mut []));
        probe.reset();
        drop(probe);
        assert_eq!(allocations() - before, 0, "{channels} channels, {rate} Hz");
    }
}
