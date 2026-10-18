//! Property tests of the inference kernels (`proptest`):
//!
//! - every `DnnArch` this processor can run gives the bits of `Portable`
//!   for random int8, float and sparse layers of random shapes;
//! - the activations never give NaN for finite inputs below the limits
//!   where libopus itself gives NaN (|x| about 2.31e9 for TANH, 4.63e9 for
//!   SIGMOID; `nnet.md` section 3.4), stay within their ranges, and
//!   saturate exactly for large inputs;
//! - the int8 quantization follows `round(127·x)` with ties to even.
#![cfg(feature = "dnn")]

use opus1::internals::dnn::nnet::{
    LinearLayer, compute_activation, compute_generic_gru, compute_linear,
};
use opus1::internals::dnn::vec_portable::{quantize, sigmoid_approx, tanh_approx};
use opus1::internals::dnn::{Activation, DnnArch, arch};
use proptest::prelude::*;

fn same(a: &[f32], b: &[f32]) -> bool {
    a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan()))
}

/// Runs a layer on every available variant and compares with `Portable`.
fn all_archs_agree(l: &LinearLayer, x: &[f32]) -> Result<(), TestCaseError> {
    let mut want = vec![0.0; l.nb_outputs];
    compute_linear(l, &mut want, x, DnnArch::Portable);
    for &a in arch::available() {
        let mut got = vec![0.0; l.nb_outputs];
        compute_linear(l, &mut got, x, a);
        prop_assert!(same(&got, &want), "{:?} {}x{}", a, l.nb_inputs, l.nb_outputs);
    }
    Ok(())
}

/// A random sparse index table for `nout` rows and `nin` inputs, and its
/// block count.
fn sparse_table(nin: usize, nout: usize, picks: &[bool]) -> (Vec<i32>, usize) {
    let mut idx = Vec::new();
    let mut blocks = 0;
    let mut p = 0;
    for _ in 0..nout / 8 {
        let at = idx.len();
        idx.resize(at + 1, 0);
        for c in (0..nin / 4 * 4).step_by(4) {
            if picks[p % picks.len()] {
                idx.push(c as i32);
                idx[at] += 1;
                blocks += 1;
            }
            p += 1;
        }
    }
    (idx, blocks)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    #[test]
    fn int8_layers_agree(
        g in 1usize..64, b in 1usize..12, seed in any::<u64>(),
        xs in prop::collection::vec(-2.0f32..2.0, 256),
    ) {
        let (nin, nout) = (4 * g, 8 * b);
        let mut s = seed | 1;
        let mut next = || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; s };
        let w: Vec<i8> = (0..nin * nout).map(|_| ((next() % 255) as i32 - 127) as i8).collect();
        let scale: Vec<f32> = (0..nout).map(|i| 1e-4 * (1 + i % 7) as f32).collect();
        let bias: Vec<f32> = (0..nout).map(|i| i as f32 * 0.01).collect();
        let l = LinearLayer { weights: Some(&w), scale: Some(&scale), bias: Some(&bias), nb_inputs: nin, nb_outputs: nout, ..Default::default() };
        all_archs_agree(&l, &xs[..nin])?;
    }

    #[test]
    fn sparse_layers_agree(
        g in 1usize..40, b in 1usize..8, picks in prop::collection::vec(any::<bool>(), 1..64),
        xs in prop::collection::vec(-3.0f32..3.0, 160), float in any::<bool>(),
    ) {
        let (nin, nout) = (4 * g, 8 * b);
        let (idx, blocks) = sparse_table(nin, nout, &picks);
        let w: Vec<i8> = (0..32 * blocks).map(|i| ((i * 37 % 255) as i32 - 127) as i8).collect();
        let fw: Vec<f32> = (0..32 * blocks).map(|i| ((i * 37 % 255) as f32 - 127.0) / 300.0).collect();
        let scale: Vec<f32> = vec![1e-3; nout];
        let l = if float {
            LinearLayer { float_weights: Some(&fw), weights_idx: Some(&idx), nb_inputs: nin, nb_outputs: nout, ..Default::default() }
        } else {
            LinearLayer { weights: Some(&w), scale: Some(&scale), weights_idx: Some(&idx), nb_inputs: nin, nb_outputs: nout, ..Default::default() }
        };
        all_archs_agree(&l, &xs[..nin])?;
    }

    #[test]
    fn float_layers_agree(
        nin in 1usize..200, nout in 1usize..41,
        ws in prop::collection::vec(-1.0f32..1.0, 200 * 40),
        xs in prop::collection::vec(-10.0f32..10.0, 200),
    ) {
        let l = LinearLayer { float_weights: Some(&ws[..nin * nout]), nb_inputs: nin, nb_outputs: nout, ..Default::default() };
        all_archs_agree(&l, &xs[..nin])?;
    }

    #[test]
    fn gru_states_stay_finite_and_agree(
        steps in prop::collection::vec(prop::collection::vec(-5.0f32..5.0, 16), 1..20),
    ) {
        let n = 8;
        let wi: Vec<i8> = (0..16 * 3 * n).map(|i| ((i * 53 % 255) as i32 - 127) as i8).collect();
        let wr: Vec<f32> = (0..n * 3 * n).map(|i| ((i * 29 % 97) as f32 - 48.0) / 60.0).collect();
        let scale = vec![2e-3f32; 3 * n];
        let li = LinearLayer { weights: Some(&wi), scale: Some(&scale), nb_inputs: 16, nb_outputs: 3 * n, ..Default::default() };
        let lr = LinearLayer { float_weights: Some(&wr), nb_inputs: n, nb_outputs: 3 * n, ..Default::default() };
        let mut want = vec![0.0; n];
        let mut states: Vec<Vec<f32>> = arch::available().iter().map(|_| vec![0.0; n]).collect();
        for x in &steps {
            compute_generic_gru(&li, &lr, &mut want, x, DnnArch::Portable);
            prop_assert!(want.iter().all(|v| v.is_finite() && v.abs() <= 1.0));
            for (s, &a) in states.iter_mut().zip(arch::available()) {
                compute_generic_gru(&li, &lr, s, x, a);
                prop_assert!(same(s, &want));
            }
        }
    }

    #[test]
    fn activations_of_finite_inputs_are_not_nan(
        xs in prop::collection::vec(-2.0e9f32..2.0e9, 1..40),
        small in prop::collection::vec(-20.0f32..20.0, 1..40),
    ) {
        for x in [&xs, &small] {
            for &a in arch::available() {
                let mut y = vec![0.0; x.len()];
                compute_activation(&mut y, x, Activation::Tanh, a);
                prop_assert!(y.iter().all(|v| (-1.0..=1.0).contains(v)), "tanh {:?}", y);
                compute_activation(&mut y, x, Activation::Sigmoid, a);
                prop_assert!(y.iter().all(|v| (0.0..=1.0).contains(v)), "sigmoid {:?}", y);
                compute_activation(&mut y, x, Activation::Exp, a);
                prop_assert!(y.iter().all(|v| v.is_finite() && *v >= 0.0), "exp {:?}", y);
                compute_activation(&mut y, x, Activation::Softmax, a);
                prop_assert!(y.iter().all(|v| !v.is_nan() && *v >= 0.0), "softmax {:?}", y);
                compute_activation(&mut y, x, Activation::Swish, a);
                prop_assert!(y.iter().all(|v| !v.is_nan()), "swish {:?}", y);
            }
        }
    }

    #[test]
    fn group_formulas_saturate(x in 50.0f32..2.3e9) {
        prop_assert_eq!(tanh_approx(x), 1.0);
        prop_assert_eq!(tanh_approx(-x), -1.0);
        prop_assert_eq!(sigmoid_approx(x), 1.0);
        prop_assert_eq!(sigmoid_approx(-x), 0.0);
    }

    #[test]
    fn quantization_rounds_to_nearest_even(k in -127i32..=127, frac in 0.0f32..1.0) {
        // Inputs whose 127·x rounds to k.
        let x = (k as f32 + frac - 0.5) / 127.0;
        let p = 127.0 * x;
        let want = p.round_ties_even() as i32;
        prop_assert_eq!(quantize(x) as i32, want);
    }
}

#[test]
fn quantization_wraps_and_handles_specials() {
    assert_eq!(quantize(1.5), -66);
    assert_eq!(quantize(-1.5), 66);
    assert_eq!(quantize(f32::NAN), 0);
    assert_eq!(quantize(f32::INFINITY), -1);
    assert_eq!(quantize(f32::NEG_INFINITY), 0);
    assert_eq!(quantize(0.5 / 127.0), 0);
    assert_eq!(quantize(-1.5 / 127.0), -2);
}

#[test]
fn nan_limits_of_tanh_and_sigmoid_follow_libopus() {
    // libopus gives NaN above these limits (nnet.md section 3.4, Review
    // item 1); opus1 reproduces it.
    assert_eq!(tanh_approx(2.313e9), 1.0);
    assert!(tanh_approx(2.3132e9).is_nan());
    assert!(tanh_approx(7.4e9).is_nan());
    assert_eq!(sigmoid_approx(4.626e9), 1.0);
    assert!(sigmoid_approx(4.6263e9).is_nan());
}
