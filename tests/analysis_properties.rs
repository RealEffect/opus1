//! Properties of the speech and music analysis that hold for any input,
//! checked with random inputs (`proptest`):
//!
//! - Any `f32` input (NaN, infinities, subnormal values included), any
//!   valid frame size and buffer length, any channel layout and bit depth:
//!   the analysis does not panic, and every valid result has no NaN and
//!   its fields in their ranges (probabilities and tonality in 0 to 1,
//!   bandwidth in 0 to 20).
//! - Calls of 40 or 60 ms give the same state as calls of 20 ms over the
//!   same samples.
//! - `reset` gives the state of `new`.
//! - `run` allocates nothing (this binary counts the allocations of each
//!   thread).
#![cfg(all(feature = "std", feature = "analysis"))]

use opus1::internals::analysis::{
    AnalysisInfo, AnalysisPcm, PcmRef, SecondChannel, TonalityAnalysisState,
};
use proptest::prelude::*;
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

const RATES: [i32; 3] = [16000, 24000, 48000];
/// Frame durations in steps of 2.5 ms.
const DURATIONS: [usize; 9] = [1, 2, 4, 8, 16, 24, 32, 40, 48];
/// (channels, c1, c2) as libopus passes them.
const LAYOUTS: [(usize, usize, SecondChannel); 6] = [
    (1, 0, SecondChannel::AllOthers),
    (2, 0, SecondChannel::AllOthers),
    (3, 0, SecondChannel::AllOthers),
    (3, 1, SecondChannel::Channel(2)),
    (3, 2, SecondChannel::None),
    (2, 1, SecondChannel::None),
];

/// One call: the coded frame duration, extra samples of look-ahead in the
/// buffer, and how to fill the buffer.
#[derive(Clone, Debug)]
struct Call {
    duration: usize,
    extra_steps: usize,
    /// One more sample in the buffer: an odd length, which the analysis
    /// rounds down.
    odd: bool,
    fill: Fill,
}

#[derive(Clone, Debug)]
enum Fill {
    /// Every sample from this list, in turn.
    Values(Vec<f32>),
    /// A sine of this frequency (Hz) and amplitude.
    Sine(f32, f32),
    /// Digital silence.
    Zero,
}

/// Any `f32`, with more weight on the special values and the normal range.
fn sample() -> impl Strategy<Value = f32> {
    prop_oneof![
        3 => -1.0f32..1.0,
        1 => any::<f32>(),
        1 => prop::sample::select(vec![
            0.0f32, -0.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY, f32::MAX, f32::MIN,
            f32::MIN_POSITIVE, 1e-40, -1e-40, 2.0, -2.0, 1.0, -1.0,
        ]),
    ]
}

fn call() -> impl Strategy<Value = Call> {
    (
        0..DURATIONS.len(),
        prop_oneof![3 => Just(0usize), 1 => 0usize..20],
        any::<bool>(),
        prop_oneof![
            2 => prop::collection::vec(sample(), 1..50).prop_map(Fill::Values),
            2 => (20.0f32..20000.0, 0.0f32..1.5).prop_map(|(f, a)| Fill::Sine(f, a)),
            1 => Just(Fill::Zero),
        ],
    )
        .prop_map(|(d, extra_steps, odd, fill)| Call { duration: DURATIONS[d], extra_steps, odd, fill })
}

/// The interleaved buffer of one call.
fn buffer(fill: &Fill, samples: usize, channels: usize, fs: i32, start: usize) -> Vec<f32> {
    (0..samples * channels)
        .map(|k| match fill {
            Fill::Values(v) => v[k % v.len()],
            Fill::Sine(f, a) => {
                let t = (start + k / channels) as f32 / fs as f32;
                a * (2.0 * std::f32::consts::PI * f * t + (k % channels) as f32).sin()
            }
            Fill::Zero => 0.0,
        })
        .collect()
}

fn to_i16(x: &[f32]) -> Vec<i16> {
    x.iter()
        .map(|v| if v.is_nan() { 0 } else { (v * 32768.0).round().clamp(-32768.0, 32767.0) as i16 })
        .collect()
}

fn check_ranges(info: &AnalysisInfo) -> Result<(), TestCaseError> {
    if !info.valid {
        return Ok(());
    }
    let unit = |name: &str, v: f32| -> Result<(), TestCaseError> {
        prop_assert!((0.0..=1.0).contains(&v), "{name} = {v} in {info:?}");
        Ok(())
    };
    unit("tonality", info.tonality)?;
    unit("music_prob", info.music_prob)?;
    unit("music_prob_min", info.music_prob_min)?;
    unit("music_prob_max", info.music_prob_max)?;
    unit("activity_probability", info.activity_probability)?;
    unit("max_pitch_ratio", info.max_pitch_ratio)?;
    for (name, v) in [
        ("tonality_slope", info.tonality_slope),
        ("noisiness", info.noisiness),
        ("activity", info.activity),
    ] {
        prop_assert!(v.is_finite(), "{name} = {v} in {info:?}");
    }
    prop_assert!((0..=20).contains(&info.bandwidth), "bandwidth in {info:?}");
    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    /// Any input and any valid call sequence: no panic, no allocation, and
    /// valid results in range.
    #[test]
    fn any_input_gives_results_in_range(
        rate in 0..RATES.len(),
        layout in 0..LAYOUTS.len(),
        float in any::<bool>(),
        lsb_depth in 8i32..=24,
        calls in prop::collection::vec(call(), 1..40),
        reset_at in prop::option::of(0usize..40),
    ) {
        let fs = RATES[rate];
        let (channels, c1, c2) = LAYOUTS[layout];
        let mut st = TonalityAnalysisState::new(fs);
        let mut start = 0;
        for (i, c) in calls.iter().enumerate() {
            let frame_size = fs as usize / 400 * c.duration;
            let samples = frame_size + fs as usize / 400 * c.extra_steps + c.odd as usize;
            let x = buffer(&c.fill, samples, channels, fs, start);
            let xi = to_i16(&x);
            let pcm = if float { PcmRef::F32(&x) } else { PcmRef::I16(&xi) };
            let before = allocations();
            let info = st.run(
                AnalysisPcm { pcm, channels, c1, c2, frame_size: samples },
                frame_size,
                lsb_depth,
            );
            prop_assert_eq!(allocations(), before, "run allocated");
            check_ranges(&info)?;
            for slot in st.info.iter() {
                check_ranges(slot)?;
            }
            if reset_at == Some(i) {
                st.reset();
            }
            start += frame_size;
        }
    }

    /// Calls of 40 or 60 ms leave the same state as calls of 20 ms over the
    /// same samples: the analysis works on its own 20 ms grid.
    #[test]
    fn longer_calls_on_the_20_ms_grid_change_nothing(
        rate in 0..RATES.len(),
        parts in 2usize..=3,
        packets in 1usize..12,
        fills in prop::collection::vec(call(), 1..12),
    ) {
        let fs = RATES[rate];
        let n20 = fs as usize / 50;
        let total = n20 * parts * packets;
        let mut x = Vec::with_capacity(total);
        let mut k = 0;
        while x.len() < total {
            let piece = buffer(&fills[k % fills.len()].fill, n20, 1, fs, x.len());
            x.extend_from_slice(&piece);
            k += 1;
        }
        let mut a = TonalityAnalysisState::new(fs);
        let mut b = TonalityAnalysisState::new(fs);
        let long = n20 * parts;
        for p in 0..packets {
            let s = &x[p * long..(p + 1) * long];
            a.run(mono(s), long, 24);
            for q in 0..parts {
                b.run(mono(&s[q * n20..(q + 1) * n20]), n20, 24);
            }
        }
        prop_assert!(same_state(&a, &b));
    }

    /// `reset` after any history gives the state of `new`.
    #[test]
    fn reset_gives_the_state_of_new(
        rate in 0..RATES.len(),
        calls in prop::collection::vec(call(), 1..20),
    ) {
        let fs = RATES[rate];
        let mut st = TonalityAnalysisState::new(fs);
        for c in &calls {
            let n = fs as usize / 400 * c.duration;
            let x = buffer(&c.fill, n, 2, fs, 0);
            let pcm = AnalysisPcm {
                pcm: PcmRef::F32(&x),
                channels: 2,
                c1: 0,
                c2: SecondChannel::AllOthers,
                frame_size: n,
            };
            st.run(pcm, n, 24);
        }
        st.reset();
        prop_assert!(same_state(&st, &TonalityAnalysisState::new(fs)));
    }
}

/// One channel of float samples, as `OpusEncoder` passes it.
fn mono(s: &[f32]) -> AnalysisPcm<'_> {
    AnalysisPcm { pcm: PcmRef::F32(s), channels: 1, c1: 0, c2: SecondChannel::AllOthers, frame_size: s.len() }
}

/// Every field of the two states, bit for bit (NaN counts as one value).
fn same_state(a: &TonalityAnalysisState, b: &TonalityAnalysisState) -> bool {
    fn bits(x: &[f32]) -> Vec<u32> {
        x.iter().map(|v| if v.is_nan() { 0x7fc0_0000 } else { v.to_bits() }).collect()
    }
    let infos = |s: &TonalityAnalysisState| -> Vec<(bool, Vec<u32>, i32, [u8; 19])> {
        s.info
            .iter()
            .map(|i| {
                (
                    i.valid,
                    bits(&[
                        i.tonality,
                        i.tonality_slope,
                        i.noisiness,
                        i.activity,
                        i.music_prob,
                        i.music_prob_min,
                        i.music_prob_max,
                        i.activity_probability,
                        i.max_pitch_ratio,
                    ]),
                    i.bandwidth,
                    i.leak_boost,
                )
            })
            .collect()
    };
    a.fs == b.fs
        && bits(&a.angle) == bits(&b.angle)
        && bits(&a.d_angle) == bits(&b.d_angle)
        && bits(&a.d2_angle) == bits(&b.d2_angle)
        && bits(&a.inmem) == bits(&b.inmem)
        && a.mem_fill == b.mem_fill
        && bits(&a.prev_band_tonality) == bits(&b.prev_band_tonality)
        && bits(&[a.prev_tonality, a.e_tracker, a.low_e_count, a.hp_ener_accum])
            == bits(&[b.prev_tonality, b.e_tracker, b.low_e_count, b.hp_ener_accum])
        && a.prev_bandwidth == b.prev_bandwidth
        && bits(a.e.as_flattened()) == bits(b.e.as_flattened())
        && bits(a.log_e.as_flattened()) == bits(b.log_e.as_flattened())
        && bits(&a.low_e) == bits(&b.low_e)
        && bits(&a.high_e) == bits(&b.high_e)
        && bits(&a.mean_e) == bits(&b.mean_e)
        && bits(&a.mem) == bits(&b.mem)
        && bits(&a.cmean) == bits(&b.cmean)
        && bits(&a.std) == bits(&b.std)
        && (a.e_count, a.count, a.analysis_offset) == (b.e_count, b.count, b.analysis_offset)
        && (a.write_pos, a.read_pos, a.read_subframe) == (b.write_pos, b.read_pos, b.read_subframe)
        && a.initialized == b.initialized
        && bits(&a.rnn_state) == bits(&b.rnn_state)
        && bits(&a.downmix_state) == bits(&b.downmix_state)
        && infos(a) == infos(b)
}
