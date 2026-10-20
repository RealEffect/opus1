//! Properties of deep PLC (`opus1::internals::dnn::plc`) that need no C
//! reference: finite output on silence, full scale, noise and garbage DRED
//! features; long losses fade out; every `DnnArch` gives the same bits; a
//! reset clears all but FARGAN; no allocation; a bounded stack.
#![cfg(all(feature = "deep-plc", feature = "std"))]

mod dnn_plc_common;

use dnn_plc_common::{Lcg, fixture};
use opus1::internals::dnn::arch::{self, DnnArch};
use opus1::internals::dnn::builtin;
use opus1::internals::dnn::models::DeepPlcModels;
use opus1::internals::dnn::plc::fargan::{self, FARGAN_CONT_SAMPLES, FARGAN_FRAME_SIZE, FarganState};
use opus1::internals::dnn::plc::features::{
    LpcnetEncState, compute_single_frame_features, inner_prod_portable, pitch_xcorr_portable,
};
use opus1::internals::dnn::plc::lpcnet_plc::CELT_DECODE_BUFFER_SIZE;
use opus1::internals::dnn::plc::{
    CONT_VECTORS, DeepPlc, LPCNET_FRAME_SIZE, NB_FEATURES, NB_TOTAL_FEATURES, PLC_MAX_FEC,
};
use std::alloc::{GlobalAlloc, Layout as MemoryLayout, System};
use std::cell::Cell;

const FRAME: usize = LPCNET_FRAME_SIZE;

thread_local! {
    static COUNTING: Cell<bool> = const { Cell::new(false) };
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
static ALLOCATOR: CountingAllocator = CountingAllocator;

/// Runs `f` with counting on and returns the number of allocations.
fn allocations(f: impl FnOnce()) -> usize {
    ALLOCATIONS.with(|n| n.set(0));
    COUNTING.with(|c| c.set(true));
    f();
    COUNTING.with(|c| c.set(false));
    ALLOCATIONS.with(|n| n.get())
}

fn models() -> &'static DeepPlcModels<'static> {
    builtin::models().deep_plc.as_ref().expect("the built-in deep PLC models")
}

fn frame(s: &[i16]) -> [i16; FRAME] {
    let mut f = [0i16; FRAME];
    f.copy_from_slice(&s[..FRAME]);
    f
}

/// Asserts that the state holds only finite values in the places that
/// feed the output.
#[track_caller]
fn assert_finite(st: &DeepPlc, what: &str) {
    assert!(st.features().iter().all(|v| v.is_finite()), "{what}: features {:?}", st.features());
    assert!(st.cont_features().iter().all(|v| v.is_finite()), "{what}: cont_features");
    assert!(st.history().iter().all(|v| v.is_finite() && v.abs() <= 1.0), "{what}: history");
    let (net, bak) = st.plc_states();
    for s in [net, &bak[0], &bak[1]] {
        assert!(s.gru1_state.iter().chain(&s.gru2_state).all(|v| v.is_finite()), "{what}: PLC states");
    }
    let f = st.fargan();
    let (g1, g2, g3) = f.gru_states();
    assert!(g1.iter().chain(g2).chain(g3).all(|v| v.is_finite()), "{what}: FARGAN GRU states");
    assert!(f.pitch_buf().iter().all(|v| v.is_finite()), "{what}: FARGAN pitch buffer");
    assert!(f.deemph_mem().is_finite(), "{what}: FARGAN de-emphasis");
}

/// The energy of a frame, per sample.
fn energy(x: &[i16]) -> f64 {
    x.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / x.len() as f64
}

/// Feeds `good` frames of `signal` (repeated as needed), then conceals
/// `lost` frames, on every available `DnnArch`; checks that all variants
/// give the same frames and that the state stays finite. Returns the
/// concealed frames of the first variant.
fn run(signal: &[i16], good: usize, lost: usize, what: &str) -> Vec<[i16; FRAME]> {
    let m = models();
    let mut first: Option<Vec<[i16; FRAME]>> = None;
    for &a in arch::available() {
        let mut st = DeepPlc::new();
        for k in 0..good {
            let start = (k * FRAME) % (signal.len() - FRAME + 1);
            st.update(&frame(&signal[start..]));
        }
        let mut out = Vec::with_capacity(lost);
        for j in 0..lost {
            let mut pcm = [0i16; FRAME];
            st.conceal(m, a, &mut pcm);
            assert_finite(&st, &format!("{what} {a:?} lost frame {j}"));
            out.push(pcm);
        }
        match &first {
            None => first = Some(out),
            Some(f) => assert!(f == &out, "{what}: {a:?} differs from {:?}", arch::available()[0]),
        }
    }
    first.unwrap_or_default()
}

#[test]
fn silence_gives_silence_like_output() {
    let out = run(&[0i16; FRAME], 20, 30, "silence");
    // The network makes up a little noise at most.
    for (j, f) in out.iter().enumerate() {
        assert!(energy(f) < 1e4, "silence, lost frame {j}: energy {}", energy(f));
    }
}

#[test]
fn a_loss_at_the_start_works() {
    let out = run(&[0i16; FRAME], 0, 12, "no history");
    assert_eq!(out.len(), 12);
}

#[test]
fn full_scale_squares_stay_finite() {
    for period in [2usize, 16, 50, 160, 333] {
        let s: Vec<i16> = (0..4 * 4000).map(|i| if (i / period) % 2 == 0 { 32767 } else { -32767 }).collect();
        run(&s, 20, 15, &format!("square of period {period}"));
    }
    // One full-scale impulse per frame.
    let mut s = vec![0i16; 3200];
    for k in 0..20 {
        s[k * FRAME + 7] = i16::MIN;
    }
    run(&s, 20, 10, "impulses");
}

#[test]
fn noise_stays_finite() {
    let mut rng = Lcg(9);
    for level in [1.0f32, 300.0, 32767.0] {
        let s: Vec<i16> = (0..6400).map(|_| (rng.uniform() * level) as i16).collect();
        run(&s, 25, 12, &format!("noise at {level}"));
    }
}

#[test]
fn a_long_loss_fades_out() {
    let s = fixture();
    let out = run(&s[100 * FRAME..], 40, 150, "speech, 1.5 s loss");
    let early: f64 = out[..5].iter().map(|f| energy(f)).sum::<f64>() / 5.0;
    let late: f64 = out[140..].iter().map(|f| energy(f)).sum::<f64>() / 10.0;
    assert!(early > 1e4, "the first concealed frames carry speech energy: {early}");
    assert!(late < early * 1e-2, "after 1.4 s the energy is down by 20 dB: {early} -> {late}");
}

#[test]
fn speech_with_losses_is_deterministic_and_the_same_on_every_arch() {
    let s = fixture();
    let m = models();
    let mut outs: Vec<Vec<i16>> = Vec::new();
    for &a in arch::available() {
        for _ in 0..2 {
            let mut st = DeepPlc::new();
            let mut out = Vec::new();
            let mut rng = Lcg(3);
            for k in 0..s.len() / FRAME {
                if rng.below(100) < 15 {
                    let mut pcm = [0i16; FRAME];
                    st.conceal(m, a, &mut pcm);
                    out.extend_from_slice(&pcm);
                } else {
                    st.update(&frame(&s[k * FRAME..]));
                }
            }
            outs.push(out);
        }
    }
    for o in &outs[1..] {
        assert!(o == &outs[0]);
    }
}

#[test]
fn reset_clears_everything_but_fargan() {
    let s = fixture();
    let m = models();
    let a = arch::detect();
    let mut st = DeepPlc::new();
    for k in 0..30 {
        st.update(&frame(&s[(200 + k) * FRAME..]));
    }
    let mut pcm = [0i16; FRAME];
    for _ in 0..3 {
        st.conceal(m, a, &mut pcm);
    }
    st.fec_add(Some(&[0.5; NB_FEATURES]));
    st.fec_add(None);
    st.reset();
    let new = DeepPlc::new();
    assert_eq!(st.positions(), new.positions());
    assert_eq!(st.fec_cursor(), (0, 0));
    assert_eq!(st.fec_fill_pos(), 0);
    assert!(!st.blend());
    assert_eq!(st.history(), new.history());
    assert_eq!(st.features(), new.features());
    assert_eq!(st.cont_features(), new.cont_features());
    assert_eq!(st.plc_states(), new.plc_states());
    assert_eq!(st.analysis().features(), new.analysis().features());
    assert_eq!(st.analysis().pitchdnn().gru_state(), new.analysis().pitchdnn().gru_state());
    // FARGAN keeps its state, as in libopus.
    assert!(st.fargan().cont_initialized());
    assert_ne!(st.fargan().gru_states().0, new.fargan().gru_states().0);

    // So the same input after the reset conceals differently from a new
    // state, while two new states agree.
    let run = |st: &mut DeepPlc| {
        for k in 0..20 {
            st.update(&frame(&s[(400 + k) * FRAME..]));
        }
        let mut out = [0i16; FRAME];
        st.conceal(m, a, &mut out);
        out
    };
    let (mut b, mut c) = (DeepPlc::new(), DeepPlc::new());
    assert_eq!(run(&mut b), run(&mut c));
    assert_ne!(run(&mut st), run(&mut DeepPlc::new()));
}

#[test]
fn a_reset_before_any_loss_gives_a_new_state() {
    let s = fixture();
    let m = models();
    let a = arch::detect();
    let mut st = DeepPlc::new();
    for k in 0..50 {
        st.update(&frame(&s[k * FRAME..]));
    }
    st.reset();
    let mut new = DeepPlc::new();
    for k in 0..30 {
        let f = frame(&s[(300 + k) * FRAME..]);
        st.update(&f);
        new.update(&f);
    }
    for _ in 0..5 {
        let (mut x, mut y) = ([0i16; FRAME], [0i16; FRAME]);
        st.conceal(m, a, &mut x);
        new.conceal(m, a, &mut y);
        assert_eq!(x, y);
    }
}

#[test]
fn the_dred_queue_is_bounded_and_takes_any_value() {
    let s = fixture();
    let m = models();
    let a = arch::detect();
    let mut st = DeepPlc::new();
    for k in 0..20 {
        st.update(&frame(&s[(500 + k) * FRAME..]));
    }
    // More vectors than the queue holds: the rest are dropped.
    for i in 0..PLC_MAX_FEC + 10 {
        st.fec_add(Some(&[i as f32 / 100.0; NB_FEATURES]));
    }
    assert_eq!(st.fec_fill_pos(), PLC_MAX_FEC as i32);
    st.fec_clear();
    // Garbage: NaN, infinities, huge values, pitches that give the periods
    // 0, 1 and the largest `int`.
    let garbage = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 1e30, -1e30, 8.0, 7.0, -40.0];
    for &g in &garbage {
        let mut f = [0.1f32; NB_FEATURES];
        f[18] = g;
        st.fec_add(Some(&f));
        let mut f = [g; NB_FEATURES];
        f[18] = 0.0;
        st.fec_add(Some(&f));
    }
    for _ in 0..2 * garbage.len() + 5 {
        let mut pcm = [0i16; FRAME];
        st.conceal(m, a, &mut pcm);
    }
    // The state recovers after a reset of the queue and good frames? The
    // NaN stays in the recurrent states, as in libopus; only check that
    // nothing panics and the output stays 16-bit.
    st.fec_clear();
    for k in 0..20 {
        st.update(&frame(&s[(700 + k) * FRAME..]));
    }
    let mut pcm = [0i16; FRAME];
    st.conceal(m, a, &mut pcm);
}

#[test]
fn update_from_celt_takes_any_value() {
    let m = models();
    let a = arch::detect();
    let mut st = DeepPlc::new();
    let mut mem = vec![vec![0.0f32; CELT_DECODE_BUFFER_SIZE]; 2];
    let mut rng = Lcg(4);
    for (i, v) in mem[0].iter_mut().enumerate() {
        *v = match i % 97 {
            0 => f32::NAN,
            1 => f32::INFINITY,
            2 => -1e38,
            _ => 1e5 * rng.uniform(),
        };
    }
    let mut pre = 0.0;
    st.update_from_celt(&[&mem[0], &mem[1]], &mut pre);
    st.update_from_celt(&[&mem[1]], &mut pre);
    let mut pcm = [0i16; FRAME];
    st.conceal(m, a, &mut pcm);
}

#[test]
fn fargan_output_is_bounded_on_speech() {
    let s = fixture();
    let m = models();
    let a = arch::detect();
    let mut enc = LpcnetEncState::new();
    let mut feats = Vec::new();
    for k in 0..s.len() / FRAME {
        let mut x = [0.0f32; FRAME];
        for (d, &v) in x.iter_mut().zip(&s[k * FRAME..]) {
            *d = v as f32;
        }
        let mut f = [0.0f32; NB_TOTAL_FEATURES];
        compute_single_frame_features(&mut enc, &x, &mut f, &m.pitchdnn, a);
        assert!(f.iter().all(|v| v.is_finite()), "frame {k}: {f:?}");
        assert!((-1.5..=1.5).contains(&f[18]), "frame {k}: pitch {}", f[18]);
        assert!((-0.5..=0.5).contains(&f[19]), "frame {k}: correlation {}", f[19]);
        feats.push(f);
    }
    let mut st = FarganState::new();
    let mut pcm0 = [0.0f32; FARGAN_CONT_SAMPLES];
    for (d, &v) in pcm0.iter_mut().zip(&s[10 * FRAME - FARGAN_CONT_SAMPLES..]) {
        *d = v as f32 / 32768.0;
    }
    let mut f0 = [0.0f32; CONT_VECTORS * NB_FEATURES];
    for (j, v) in f0.as_chunks_mut::<NB_FEATURES>().0.iter_mut().enumerate() {
        v.copy_from_slice(&feats[5 + j][..NB_FEATURES]);
    }
    fargan::fargan_cont(&mut st, &pcm0, &f0, &m.fargan, a);
    let mut peak = 0.0f32;
    for f in &feats[10..] {
        let mut ff = [0.0f32; NB_FEATURES];
        ff.copy_from_slice(&f[..NB_FEATURES]);
        let mut out = [0.0f32; FARGAN_FRAME_SIZE];
        fargan::fargan_synthesize(&mut st, &mut out, &ff, &m.fargan, a);
        for &v in &out {
            assert!(v.is_finite());
            peak = peak.max(v.abs());
        }
    }
    assert!(peak > 0.05 && peak < 4.0, "peak {peak}");
}

#[test]
fn the_portable_helpers_give_the_neon_bits() {
    // On 64-bit ARM, `DnnArch::Portable` uses the plain inner product and
    // cross-correlation of `features.rs`, and the other variants the NEON
    // ones of `crate::pitch`; the feature analysis must not depend on it.
    let mut rng = Lcg(11);
    for len in [1usize, 4, 7, 8, 160] {
        let x: Vec<f32> = (0..len).map(|_| rng.uniform() * 1000.0).collect();
        let y: Vec<f32> = (0..len + 223).map(|_| rng.uniform() * 1000.0).collect();
        let mut got = vec![0.0f32; 224];
        pitch_xcorr_portable(&x, &y, &mut got);
        let mut want = vec![0.0f32; 224];
        opus1::internals::pitch::pitch_xcorr(&x, &y, &mut want, len, 224);
        if cfg!(target_arch = "aarch64") {
            assert_eq!(
                got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                want.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "pitch_xcorr, len {len}"
            );
            let a = inner_prod_portable(&x, &y);
            let b = opus1::internals::pitch::inner_prod(&x, &y, len);
            assert_eq!(a.to_bits(), b.to_bits(), "inner_prod, len {len}");
        }
    }
    // And the whole analysis.
    let s = fixture();
    let m = models();
    let mut p = LpcnetEncState::new();
    let mut n = LpcnetEncState::new();
    for k in 0..s.len() / FRAME {
        let mut x = [0.0f32; FRAME];
        for (d, &v) in x.iter_mut().zip(&s[k * FRAME..]) {
            *d = v as f32;
        }
        let (mut fp, mut fd) = ([0.0f32; NB_TOTAL_FEATURES], [0.0f32; NB_TOTAL_FEATURES]);
        compute_single_frame_features(&mut p, &x, &mut fp, &m.pitchdnn, DnnArch::Portable);
        compute_single_frame_features(&mut n, &x, &mut fd, &m.pitchdnn, arch::detect());
        assert_eq!(
            fp.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            fd.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "frame {k}"
        );
    }
}

#[test]
fn conceal_and_update_do_not_allocate() {
    let s = fixture();
    let m = models();
    let mut st = Box::new(DeepPlc::new());
    let frames: Vec<[i16; FRAME]> = (0..60).map(|k| frame(&s[(300 + k) * FRAME..])).collect();
    let mem = vec![vec![10.0f32; CELT_DECODE_BUFFER_SIZE]; 2];
    let _ = models();
    for &a in arch::available() {
        let n = allocations(|| {
            for f in &frames[..40] {
                st.update(f);
            }
            let mut pcm = [0i16; FRAME];
            for _ in 0..4 {
                st.conceal(m, a, &mut pcm);
            }
            st.fec_add(Some(&[0.0; NB_FEATURES]));
            st.conceal(m, a, &mut pcm);
            st.fec_clear();
            let mut pre = 0.0;
            st.update_from_celt(&[&mem[0], &mem[1]], &mut pre);
            st.conceal(m, a, &mut pcm);
            st.reset();
        });
        assert_eq!(n, 0, "{a:?}: allocations");
    }
}

#[test]
fn conceal_fits_in_a_small_stack() {
    // The first lost frame of a loss is the deepest call: the batch of
    // analyses and `fargan_cont`. Measured on the M4: release builds run
    // in 16 KB (the smallest thread stack macOS gives), debug builds need
    // about 100 KB.
    let stack = if cfg!(debug_assertions) { 256 << 10 } else { 64 << 10 };
    let s = fixture();
    let handle = std::thread::Builder::new()
        .stack_size(stack)
        .spawn(move || {
            let m = models();
            let mut st = Box::new(DeepPlc::new());
            for k in 0..40 {
                st.update(&frame(&s[(300 + k) * FRAME..]));
            }
            let mut pcm = [0i16; FRAME];
            st.conceal(m, arch::detect(), &mut pcm);
            st.conceal(m, DnnArch::Portable, &mut pcm);
            pcm
        })
        .unwrap();
    handle.join().unwrap();
}

#[test]
fn the_state_has_the_expected_size() {
    // 9,296 bytes of PitchDNN state, 5,748 more of the analysis, 3,868 of
    // FARGAN, 8,320 of the DRED queue, 9,600 of history, 4,608 of PLC
    // network states, and 572 of features and positions. (libopus's
    // `LPCNetPLCState` has 67,992 bytes with its model structs.)
    assert_eq!(std::mem::size_of::<DeepPlc>(), 42_012);
}
