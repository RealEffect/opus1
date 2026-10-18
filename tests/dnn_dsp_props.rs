#![cfg(feature = "dnn")]
//! Properties of the signal processing of the neural network code
//! (`opus1::internals::dnn::freq` and `burg`) that hold without libopus:
//! the transforms keep the energy and are linear, and the LPC filters of the
//! Burg analysis and of `lpc_from_cepstrum` are stable.
//!
//! `tests/dnn_freq_burg.rs` checks the exact bits against libopus.

use opus1::internals::dnn::burg::silk_burg_analysis;
use opus1::internals::dnn::freq::{
    self, FRAME_SIZE, FREQ_SIZE, LPC_ORDER, NB_BANDS, WINDOW_SIZE,
};
use opus1::internals::kiss_fft::KissCpx;
use proptest::prelude::*;

/// True when the filter `1 + c[1]·z^-1 + ... + c[m]·z^-m` has all its zeros
/// inside the unit circle: every reflection coefficient of the step-down
/// recursion has a magnitude below 1.
fn is_minimum_phase(c: &[f64]) -> bool {
    let mut c = c.to_vec();
    for m in (1..c.len()).rev() {
        let k = c[m];
        // NaN fails too.
        if k.abs() >= 1.0 || k.is_nan() {
            return false;
        }
        let prev = c.clone();
        for i in 1..m {
            c[i] = (prev[i] - k * prev[m - i]) / (1.0 - k * k);
        }
    }
    true
}

fn frame<const N: usize>(scale: f32) -> impl Strategy<Value = [f32; N]> {
    proptest::collection::vec(-1.0f32..1.0, N).prop_map(move |v| {
        let mut out = [0.0f32; N];
        for (o, x) in out.iter_mut().zip(v) {
            *o = scale * x;
        }
        out
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// Parseval's theorem for the transform scaled by 1/320: the energy of
    /// the 161 kept bins, counted twice except for the first and the last,
    /// is the signal energy divided by 320.
    #[test]
    fn forward_transform_keeps_the_energy(x in frame::<WINDOW_SIZE>(1000.0)) {
        let mut spec = [KissCpx::default(); FREQ_SIZE];
        freq::forward_transform(&mut spec, &x);
        let e_time: f64 = x.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / 320.0;
        let mut e_freq = 0.0f64;
        for (k, b) in spec.iter().enumerate() {
            let e = (b.r as f64).powi(2) + (b.i as f64).powi(2);
            e_freq += if k == 0 || k == FREQ_SIZE - 1 { e } else { 2.0 * e };
        }
        prop_assert!((e_time - e_freq).abs() <= 1e-5 * e_time + 1e-9, "{e_time} {e_freq}");
        // The first bin is the mean, and real.
        let mean = x.iter().map(|&v| v as f64).sum::<f64>() / 320.0;
        prop_assert!((spec[0].r as f64 - mean).abs() <= 1e-3, "{} {mean}", spec[0].r);
        prop_assert!(spec[0].i.abs() <= 1e-3 && spec[FREQ_SIZE - 1].i.abs() <= 1e-3);
    }

    /// The DCT is linear, up to rounding.
    #[test]
    fn dct_is_linear(a in frame::<NB_BANDS>(10.0), b in frame::<NB_BANDS>(10.0)) {
        let mut sum = [0.0f32; NB_BANDS];
        for i in 0..NB_BANDS {
            sum[i] = a[i] + b[i];
        }
        let (mut da, mut db, mut ds) = ([0.0f32; NB_BANDS], [0.0f32; NB_BANDS], [0.0f32; NB_BANDS]);
        freq::dct(&mut da, &a);
        freq::dct(&mut db, &b);
        freq::dct(&mut ds, &sum);
        for i in 0..NB_BANDS {
            prop_assert!((da[i] + db[i] - ds[i]).abs() <= 1e-4, "{i}: {} {} {}", da[i], db[i], ds[i]);
        }
    }

    /// The Burg analysis gives a stable predictor and a residual energy
    /// between 0 and the signal energy, for every order and gain limit.
    #[test]
    fn burg_gives_a_stable_predictor(
        x in frame::<FRAME_SIZE>(1.0),
        d in 0usize..=16,
        nb_subfr in 1usize..=4,
        min_inv_gain in prop_oneof![Just(1e-3f32), Just(1e-2f32), Just(0.5f32)],
    ) {
        let subfr_length = FRAME_SIZE / nb_subfr;
        prop_assume!(d <= subfr_length);
        let mut a = [f32::NAN; 16];
        let nrg = silk_burg_analysis(&mut a, &x, min_inv_gain, subfr_length, nb_subfr, d);
        let energy: f64 = x[..subfr_length * nb_subfr].iter().map(|&v| (v as f64).powi(2)).sum();
        prop_assert!(nrg >= 0.0 && (nrg as f64) <= 1.001 * energy + 1e-6, "{nrg} {energy}");
        // `a` predicts x[n] from the samples before it: the error filter is
        // 1 - a[0]·z^-1 - ... .
        let mut c = vec![1.0f64];
        c.extend(a[..d].iter().map(|&v| -(v as f64)));
        prop_assert!(is_minimum_phase(&c), "{:?}", &a[..d]);
    }

    /// `lpc_from_cepstrum` gives finite coefficients of a stable filter
    /// and a positive error.
    #[test]
    fn lpc_from_cepstrum_gives_a_stable_filter(ceps in frame::<NB_BANDS>(2.0), level in -4.0f32..4.0) {
        let mut ceps = ceps;
        ceps[0] += level;
        let mut lpc = [f32::NAN; LPC_ORDER];
        let err = freq::lpc_from_cepstrum(&mut lpc, &ceps);
        prop_assert!(err > 0.0 && err.is_finite(), "{err}");
        // The filter is 1 + lpc[0]·z^-1 + ... .
        let mut c = vec![1.0f64];
        c.extend(lpc.iter().map(|&v| v as f64));
        prop_assert!(is_minimum_phase(&c), "{lpc:?}");
    }

    /// The Burg cepstrum of any 16-bit signal is finite, and the second
    /// half (the difference of the two 5 ms halves) is zero when both halves
    /// are the same.
    #[test]
    fn burg_cepstrum_is_finite_and_symmetric(half in frame::<{ FRAME_SIZE / 2 }>(32767.0)) {
        let mut x = [0.0f32; FRAME_SIZE];
        x[..FRAME_SIZE / 2].copy_from_slice(&half);
        x[FRAME_SIZE / 2..].copy_from_slice(&half);
        let mut ceps = [0.0f32; 2 * NB_BANDS];
        freq::burg_cepstral_analysis(&mut ceps, &x);
        prop_assert!(ceps.iter().all(|c| c.is_finite()), "{ceps:?}");
        prop_assert!(ceps[NB_BANDS..].iter().all(|&c| c == 0.0), "{ceps:?}");
    }
}
