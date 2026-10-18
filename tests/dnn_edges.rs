//! Edge cases of the inference kernels that the random tests reach rarely
//! or never, checked on every `DnnArch` this processor can run:
//!
//! - sparse index tables with empty row blocks, repeated positions and
//!   positions in falling order, against a plain scalar computation of the
//!   rule of `nnet.md` section 3 (one fused multiply-add chain per output
//!   in the order of the table; exact int8 sums);
//! - int8 layers at the extremes: every weight ±127, inputs that quantize
//!   to -128 and 127, up to `MAX_INPUTS` inputs, so that the 16-bit pair
//!   sums of the kernels without DotProd reach their largest value and the
//!   32-bit sums exceed 2^24;
//! - NaN, infinities and subnormal numbers as inputs of float and int8
//!   layers;
//! - GRU and GLU layers whose sizes are odd, so that every activation runs
//!   its tail formulas;
//! - layers without inputs or outputs;
//! - invalid layers and short slices: the safe functions must panic before
//!   any `unsafe` kernel runs, never read out of bounds;
//! - the arithmetic of the x86 kernels (`vec_x86`), which cannot run on
//!   the development machine: their FRECPE through a float division and
//!   their fix of the x86 float-to-int conversion, in scalar form, against
//!   `vec_portable` on every input class.
#![cfg(feature = "dnn")]

use std::panic::{AssertUnwindSafe, catch_unwind};

use opus1::internals::dnn::nnet::{
    Conv2dLayer, LinearLayer, compute_activation, compute_conv2d, compute_generic_conv1d,
    compute_generic_conv1d_dilation, compute_generic_gru, compute_glu, compute_glu_in_place,
    compute_linear,
};
use opus1::internals::dnn::vec_portable::{frecpe_bits, quantize};
use opus1::internals::dnn::{Activation, DnnArch, MAX_INPUTS, arch};

fn same(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan()))
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 16) as u32
    }
    fn f(&mut self, a: f32) -> f32 {
        a * (2.0 * (self.next() as f32 / 4_294_967_296.0) - 1.0)
    }
    /// An int8 weight in [-127, 127].
    fn w8(&mut self) -> i8 {
        ((self.next() % 255) as i32 - 127) as i8
    }
    /// A number in `0..n`.
    fn below(&mut self, n: usize) -> usize {
        self.next() as usize % n
    }
}

/// The int8 input rule of `nnet.md` section 3.2, written independently of
/// `vec_portable`: `127·x` rounded to `f32`, to the nearest integer with
/// ties to even, saturated (NaN gives 0), then the low 8 bits.
fn q(x: f32) -> i8 {
    (127.0f32 * x).round_ties_even() as i32 as i16 as i8
}

/// A sparse index table with every kind of row block: empty, one block,
/// repeated positions, falling positions, every position.
fn awkward_table(nin: usize, nout: usize, rng: &mut Rng) -> (Vec<i32>, usize) {
    let groups = nin / 4;
    let mut idx = Vec::new();
    let mut blocks = 0;
    for b in 0..nout / 8 {
        let positions: Vec<i32> = match b % 6 {
            0 => vec![],
            1 => vec![4 * rng.below(groups) as i32],
            2 => vec![0, 0, 0],
            3 => (0..groups as i32).rev().map(|g| 4 * g).collect(),
            4 => (0..groups as i32).map(|g| 4 * g).chain([4 * (groups as i32 - 1), 0]).collect(),
            _ => (0..5).map(|_| 4 * rng.below(groups) as i32).collect(),
        };
        idx.push(positions.len() as i32);
        blocks += positions.len();
        idx.extend(positions);
    }
    (idx, blocks)
}

/// The sparse float product, one fused multiply-add chain per output in
/// the order of the table (`nnet.md` section 3.3).
fn sparse_float_oracle(w: &[f32], idx: &[i32], nout: usize, x: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; nout];
    let (mut p, mut wp) = (0, 0);
    for b in 0..nout / 8 {
        let count = idx[p] as usize;
        p += 1;
        for _ in 0..count {
            let pos = idx[p] as usize;
            p += 1;
            for c in 0..4 {
                for k in 0..8 {
                    out[8 * b + k] = w[wp + 8 * c + k].mul_add(x[pos + c], out[8 * b + k]);
                }
            }
            wp += 32;
        }
    }
    out
}

/// The sparse int8 product with exact sums.
fn sparse_int8_oracle(w: &[i8], idx: &[i32], scale: &[f32], nout: usize, x: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; nout];
    let (mut p, mut wp) = (0, 0);
    for b in 0..nout / 8 {
        let mut acc = [0i64; 8];
        let count = idx[p] as usize;
        p += 1;
        for _ in 0..count {
            let pos = idx[p] as usize;
            p += 1;
            for (k, a) in acc.iter_mut().enumerate() {
                for c in 0..4 {
                    *a += w[wp + 4 * k + c] as i64 * q(x[pos + c]) as i64;
                }
            }
            wp += 32;
        }
        for k in 0..8 {
            out[8 * b + k] = scale[8 * b + k] * acc[k] as i32 as f32;
        }
    }
    out
}

/// The dense int8 product with exact sums (blocks of 8 rows by 4 columns).
fn dense_int8_oracle(w: &[i8], scale: &[f32], nin: usize, nout: usize, x: &[f32]) -> Vec<f32> {
    let groups = nin / 4;
    (0..nout)
        .map(|i| {
            let (b, k) = (i / 8, i % 8);
            let mut acc = 0i64;
            for g in 0..groups {
                for c in 0..4 {
                    acc += w[(b * groups + g) * 32 + 4 * k + c] as i64 * q(x[4 * g + c]) as i64;
                }
            }
            scale[i] * acc as i32 as f32
        })
        .collect()
}

/// Runs `l` on every available variant and returns the outputs.
fn run_all(l: &LinearLayer, x: &[f32]) -> Vec<(DnnArch, Vec<f32>)> {
    arch::available()
        .iter()
        .map(|&a| {
            let mut out = vec![0.0f32; l.nb_outputs];
            compute_linear(l, &mut out, x, a);
            (a, out)
        })
        .collect()
}

#[test]
fn awkward_sparse_tables_follow_the_table_order() {
    let mut rng = Rng(0x5eed_1234_9876_abcd);
    for &(nin, nout) in &[(4usize, 48usize), (8, 48), (12, 96), (64, 48), (128, 96), (2048, 48)] {
        let (idx, blocks) = awkward_table(nin, nout, &mut rng);
        let fw: Vec<f32> = (0..32 * blocks).map(|_| rng.f(0.5)).collect();
        let w8v: Vec<i8> = (0..32 * blocks).map(|_| rng.w8()).collect();
        let scale: Vec<f32> = (0..nout).map(|_| rng.f(0.01)).collect();
        for trial in 0..8 {
            let x: Vec<f32> = (0..nin).map(|_| rng.f(if trial < 4 { 1.0 } else { 2.0 })).collect();
            let fl = LinearLayer { float_weights: Some(&fw), weights_idx: Some(&idx), nb_inputs: nin, nb_outputs: nout, ..Default::default() };
            let want = sparse_float_oracle(&fw, &idx, nout, &x);
            for (a, got) in run_all(&fl, &x) {
                assert!(same(&got, &want), "float sparse {a:?} {nin}x{nout}");
            }
            let il = LinearLayer { weights: Some(&w8v), scale: Some(&scale), weights_idx: Some(&idx), nb_inputs: nin, nb_outputs: nout, ..Default::default() };
            let want = sparse_int8_oracle(&w8v, &idx, &scale, nout, &x);
            for (a, got) in run_all(&il, &x) {
                assert!(same(&got, &want), "int8 sparse {a:?} {nin}x{nout}");
            }
        }
    }
}

#[test]
fn a_sparse_layer_without_blocks_gives_its_bias() {
    let idx = [0i32; 4];
    let bias: Vec<f32> = (0..32).map(|i| i as f32 - 3.5).collect();
    let scale = vec![1.0f32; 32];
    let x = vec![1.0f32; 16];
    let fl = LinearLayer { float_weights: Some(&[]), weights_idx: Some(&idx), bias: Some(&bias), nb_inputs: 16, nb_outputs: 32, ..Default::default() };
    let il = LinearLayer { weights: Some(&[]), scale: Some(&scale), weights_idx: Some(&idx), bias: Some(&bias), nb_inputs: 16, nb_outputs: 32, ..Default::default() };
    for l in [fl, il] {
        for (a, got) in run_all(&l, &x) {
            assert!(same(&got, &bias), "{a:?}");
        }
    }
}

#[test]
fn int8_layers_at_the_extremes_are_exact() {
    // Inputs that quantize to -128, 127, -127, 0 and the wrap-around of
    // values above 1.
    let specials = [-1.003_937f32, -1.0, 1.0, 0.0, -1.0 / 254.0, 1.5, -1.5, 1.007_874, -1.007_874];
    for &nin in &[4usize, 8, 12, 2044, MAX_INPUTS] {
        for &nout in &[8usize, 16] {
            for pattern in 0..6 {
                let w: Vec<i8> = (0..nin * nout)
                    .map(|i| match pattern {
                        0 => 127,
                        1 => -127,
                        2 => if i % 2 == 0 { 127 } else { -127 },
                        3 => if (i / 4) % 2 == 0 { -127 } else { 127 },
                        4 => if i % 3 == 0 { 0 } else { -127 },
                        _ => 1,
                    })
                    .collect();
                let scale = vec![1.0f32; nout];
                for &x0 in &specials {
                    let x: Vec<f32> = (0..nin).map(|j| if pattern == 5 { specials[j % specials.len()] } else { x0 }).collect();
                    let l = LinearLayer { weights: Some(&w), scale: Some(&scale), nb_inputs: nin, nb_outputs: nout, ..Default::default() };
                    let want = dense_int8_oracle(&w, &scale, nin, nout, &x);
                    for (a, got) in run_all(&l, &x) {
                        assert!(same(&got, &want), "{a:?} {nin}x{nout} pattern {pattern} x {x0}: {:?} vs {:?}", &got[..2], &want[..2]);
                    }
                }
            }
        }
    }
    // Sums above 2^24 whose conversion to float rounds (an odd sum of 25
    // bits lies halfway between two floats): every variant rounds to even.
    let w = vec![-127i8; MAX_INPUTS * 8];
    let mut x = vec![-1.003_937f32; MAX_INPUTS];
    x[0] = 1.0;
    assert_eq!((q(x[1]), q(x[0])), (-128, 127));
    let sum: i64 = 2047 * 127 * 128 - 127 * 127;
    assert!(sum > 1 << 24 && sum % 2 == 1 && (sum as f32) as i64 != sum);
    let l = LinearLayer { weights: Some(&w), scale: Some(&[1.0; 8]), nb_inputs: MAX_INPUTS, nb_outputs: 8, ..Default::default() };
    for (a, got) in run_all(&l, &x) {
        assert_eq!(got, vec![sum as f32; 8], "{a:?}");
    }
}

#[test]
fn special_floats_in_layers_agree() {
    let specials = [
        f32::NAN,
        -f32::NAN,
        f32::from_bits(0x7f80_0001),
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::MAX,
        f32::MIN,
        f32::MIN_POSITIVE,
        1e-45,
        -1e-45,
        1e-40,
        -0.0,
        0.0,
        2.147_483_6e9,
        2_147_483_648.0,
        -2_147_483_648.0,
        3e9,
    ];
    let mut rng = Rng(77);
    for &(nin, nout) in &[(16usize, 8usize), (20, 16), (17, 3), (64, 1), (5, 40)] {
        let fw: Vec<f32> = (0..nin * nout).map(|_| rng.f(1.0)).collect();
        let w: Vec<i8> = (0..nin * nout).map(|_| rng.w8()).collect();
        let scale: Vec<f32> = (0..nout).map(|_| rng.f(0.01)).collect();
        for shift in 0..specials.len() {
            let x: Vec<f32> = (0..nin).map(|j| if j % 3 == 0 { specials[(j + shift) % specials.len()] } else { rng.f(1.0) }).collect();
            let fl = LinearLayer { float_weights: Some(&fw), nb_inputs: nin, nb_outputs: nout, ..Default::default() };
            let all = run_all(&fl, &x);
            for (a, got) in &all {
                assert!(same(got, &all[0].1), "float {a:?} {nin}x{nout}");
            }
            if nin % 4 == 0 && nout % 8 == 0 {
                let il = LinearLayer { weights: Some(&w), scale: Some(&scale), nb_inputs: nin, nb_outputs: nout, ..Default::default() };
                let want = dense_int8_oracle(&w, &scale, nin, nout, &x);
                for (a, got) in run_all(&il, &x) {
                    assert!(same(&got, &want), "int8 {a:?} {nin}x{nout} shift {shift}");
                }
            }
            for act in [Activation::Tanh, Activation::Sigmoid, Activation::Exp, Activation::Swish, Activation::Softmax, Activation::Relu] {
                let mut want = vec![0.0f32; nin];
                compute_activation(&mut want, &x, act, DnnArch::Portable);
                for &a in arch::available() {
                    let mut got = vec![0.0f32; nin];
                    compute_activation(&mut got, &x, act, a);
                    assert!(same(&got, &want), "{act:?} {a:?} n {nin}");
                }
            }
        }
    }
}

#[test]
fn odd_sized_gru_and_glu_agree() {
    let mut rng = Rng(99);
    for &n in &[1usize, 2, 3, 5, 6, 7, 9, 13, 191, 192] {
        let nin = 2 * n + 1;
        let wi: Vec<f32> = (0..nin * 3 * n).map(|_| rng.f(0.4)).collect();
        let wr: Vec<f32> = (0..n * 3 * n).map(|_| rng.f(0.4)).collect();
        let bi: Vec<f32> = (0..3 * n).map(|_| rng.f(0.2)).collect();
        let diag: Vec<f32> = (0..3 * n).map(|_| rng.f(0.2)).collect();
        let li = LinearLayer { float_weights: Some(&wi), bias: Some(&bi), nb_inputs: nin, nb_outputs: 3 * n, ..Default::default() };
        let lr = LinearLayer { float_weights: Some(&wr), diag: Some(&diag), nb_inputs: n, nb_outputs: 3 * n, ..Default::default() };
        let mut want = vec![0.0f32; n];
        let mut states: Vec<Vec<f32>> = arch::available().iter().map(|_| vec![0.0; n]).collect();
        for step in 0..12 {
            let x: Vec<f32> = (0..nin).map(|_| rng.f(if step % 3 == 0 { 8.0 } else { 1.0 })).collect();
            compute_generic_gru(&li, &lr, &mut want, &x, DnnArch::Portable);
            assert!(want.iter().all(|v| v.is_finite() && v.abs() <= 1.0), "GRU {n}: {want:?}");
            for (s, &a) in states.iter_mut().zip(arch::available()) {
                compute_generic_gru(&li, &lr, s, &x, a);
                assert!(same(s, &want), "GRU {a:?} n {n} step {step}");
            }
        }
        let wg: Vec<f32> = (0..n * n).map(|_| rng.f(0.4)).collect();
        let lg = LinearLayer { float_weights: Some(&wg), nb_inputs: n, nb_outputs: n, ..Default::default() };
        let x: Vec<f32> = (0..n).map(|_| rng.f(2.0)).collect();
        let mut want = vec![0.0f32; n];
        compute_glu(&lg, &mut want, &x, DnnArch::Portable);
        for &a in arch::available() {
            let mut got = vec![0.0f32; n];
            compute_glu(&lg, &mut got, &x, a);
            let mut inplace = x.clone();
            compute_glu_in_place(&lg, &mut inplace, a);
            assert!(same(&got, &want) && same(&inplace, &want), "GLU {a:?} n {n}");
        }
    }
}

#[test]
fn layers_without_inputs_or_outputs() {
    let bias = [0.25f32, -0.5, 1.0];
    for &a in arch::available() {
        // No inputs: the output is the bias (float) or 0 plus the bias.
        let l = LinearLayer { float_weights: Some(&[]), bias: Some(&bias), nb_inputs: 0, nb_outputs: 3, ..Default::default() };
        let mut out = [9.0f32; 3];
        compute_linear(&l, &mut out, &[], a);
        assert_eq!(out, bias, "{a:?}");
        let scale = [1.0f32; 8];
        let l = LinearLayer { weights: Some(&[]), scale: Some(&scale), nb_inputs: 0, nb_outputs: 8, ..Default::default() };
        let mut out = [9.0f32; 8];
        compute_linear(&l, &mut out, &[], a);
        assert_eq!(out, [0.0; 8], "{a:?}");
        // No outputs: nothing is written.
        let l = LinearLayer { float_weights: Some(&[]), nb_inputs: 4, nb_outputs: 0, ..Default::default() };
        let mut out = [9.0f32; 2];
        compute_linear(&l, &mut out, &[1.0; 4], a);
        assert_eq!(out, [9.0; 2], "{a:?}");
        // A layer without any weights writes its bias, as in libopus.
        let l = LinearLayer { bias: Some(&bias), nb_inputs: 2, nb_outputs: 3, ..Default::default() };
        let mut out = [9.0f32; 3];
        compute_linear(&l, &mut out, &[1.0; 2], a);
        assert_eq!(out, bias, "{a:?}");
        let mut empty: [f32; 0] = [];
        compute_activation(&mut empty, &[], Activation::Softmax, a);
    }
}

/// Runs `f` and says whether it panicked.
fn panics(f: impl FnOnce()) -> bool {
    catch_unwind(AssertUnwindSafe(f)).is_err()
}

#[test]
fn invalid_layers_and_short_slices_panic_on_every_variant() {
    let fw = vec![0.5f32; 64 * 32];
    let w8v = vec![1i8; 64 * 32];
    let scale = vec![1.0f32; 64];
    let x = vec![0.25f32; 64];
    for &a in arch::available() {
        let lin = |l: LinearLayer, nin: usize, nout: usize| {
            panics(|| {
                let mut out = vec![0.0f32; nout];
                compute_linear(&l, &mut out, &x[..nin], a);
            })
        };
        let float = LinearLayer { float_weights: Some(&fw[..16 * 8]), nb_inputs: 16, nb_outputs: 8, ..Default::default() };
        let int8 = LinearLayer { weights: Some(&w8v[..16 * 8]), scale: Some(&scale[..8]), nb_inputs: 16, nb_outputs: 8, ..Default::default() };
        // The valid layers run.
        assert!(!lin(float, 16, 8) && !lin(int8, 16, 8), "{a:?}");
        // Short input, short output.
        assert!(lin(float, 15, 8) && lin(float, 16, 7), "{a:?}");
        assert!(lin(int8, 15, 8) && lin(int8, 16, 7), "{a:?}");
        // Short weights, scale and bias.
        assert!(lin(LinearLayer { float_weights: Some(&fw[..16 * 8 - 1]), ..float }, 16, 8), "{a:?}");
        assert!(lin(LinearLayer { weights: Some(&w8v[..16 * 8 - 1]), ..int8 }, 16, 8), "{a:?}");
        assert!(lin(LinearLayer { scale: Some(&scale[..7]), ..int8 }, 16, 8), "{a:?}");
        assert!(lin(LinearLayer { scale: None, ..int8 }, 16, 8), "{a:?}");
        assert!(lin(LinearLayer { bias: Some(&scale[..7]), ..float }, 16, 8), "{a:?}");
        // int8 shapes the kernels cannot run.
        assert!(lin(LinearLayer { nb_inputs: 14, weights: Some(&w8v[..14 * 8]), ..int8 }, 14, 8), "{a:?}");
        assert!(lin(LinearLayer { nb_outputs: 12, weights: Some(&w8v[..16 * 12]), scale: Some(&scale[..12]), ..int8 }, 16, 12), "{a:?}");
        let big = vec![0i8; (MAX_INPUTS + 4) * 8];
        let bigx = vec![0.0f32; MAX_INPUTS + 4];
        let l = LinearLayer { weights: Some(&big), scale: Some(&scale[..8]), nb_inputs: MAX_INPUTS + 4, nb_outputs: 8, ..Default::default() };
        assert!(panics(|| compute_linear(&l, &mut [0.0; 8], &bigx, a)), "{a:?}");
        // A diagonal needs nb_outputs == 3·nb_inputs.
        assert!(lin(LinearLayer { diag: Some(&scale[..8]), ..float }, 16, 8), "{a:?}");
        // Sparse tables: valid, then each kind of damage.
        let sparse = |idx: &[i32], nw: usize, float: bool| {
            let l = if float {
                LinearLayer { float_weights: Some(&fw[..nw]), weights_idx: Some(idx), nb_inputs: 16, nb_outputs: 16, ..Default::default() }
            } else {
                LinearLayer { weights: Some(&w8v[..nw]), scale: Some(&scale[..16]), weights_idx: Some(idx), nb_inputs: 16, nb_outputs: 16, ..Default::default() }
            };
            panics(|| compute_linear(&l, &mut [0.0; 16], &x[..16], a))
        };
        for float in [false, true] {
            assert!(!sparse(&[1, 12, 2, 0, 4], 96, float), "{a:?}");
            assert!(sparse(&[1, 12, 2, 0, 4], 95, float), "short weights {a:?}");
            assert!(sparse(&[1, 12, 2, 0], 96, float), "short table {a:?}");
            assert!(sparse(&[1, 12], 96, float), "missing block {a:?}");
            assert!(sparse(&[-1, 12, 2, 0, 4], 96, float), "negative count {a:?}");
            assert!(sparse(&[1, -4, 2, 0, 4], 96, float), "negative position {a:?}");
            assert!(sparse(&[1, i32::MIN, 2, 0, 4], 96, float), "i32::MIN {a:?}");
            assert!(sparse(&[1, 13, 2, 0, 4], 96, float), "position past the input {a:?}");
            assert!(sparse(&[1, 16, 2, 0, 4], 96, float), "position at the end {a:?}");
            assert!(sparse(&[1, 12, i32::MAX, 0, 4], 96, float), "huge count {a:?}");
        }
        // A sparse layer needs a multiple of 8 outputs.
        let l = LinearLayer { float_weights: Some(&fw[..32]), weights_idx: Some(&[1, 0]), nb_inputs: 16, nb_outputs: 4, ..Default::default() };
        assert!(panics(|| compute_linear(&l, &mut [0.0; 4], &x[..16], a)), "{a:?}");
        // GRU: a short state, sizes that do not fit, too many neurons.
        let wi = vec![0.1f32; 8 * 12];
        let wr = vec![0.1f32; 4 * 12];
        let gi = LinearLayer { float_weights: Some(&wi), nb_inputs: 8, nb_outputs: 12, ..Default::default() };
        let gr = LinearLayer { float_weights: Some(&wr), nb_inputs: 4, nb_outputs: 12, ..Default::default() };
        assert!(!panics(|| compute_generic_gru(&gi, &gr, &mut [0.0; 4], &x[..8], a)), "{a:?}");
        assert!(panics(|| compute_generic_gru(&gi, &gr, &mut [0.0; 3], &x[..8], a)), "{a:?}");
        assert!(panics(|| compute_generic_gru(&gi, &gr, &mut [0.0; 4], &x[..7], a)), "{a:?}");
        let gr3 = LinearLayer { nb_inputs: 3, ..gr };
        assert!(panics(|| compute_generic_gru(&gi, &gr3, &mut [0.0; 4], &x[..8], a)), "{a:?}");
        let big_r = LinearLayer { float_weights: Some(&[]), nb_inputs: 193, nb_outputs: 579, ..Default::default() };
        let big_i = LinearLayer { float_weights: Some(&[]), nb_inputs: 0, nb_outputs: 579, ..Default::default() };
        assert!(panics(|| compute_generic_gru(&big_i, &big_r, &mut [0.0; 193], &[], a)), "{a:?}");
        // GLU: inputs and outputs differ, short output.
        let g = LinearLayer { float_weights: Some(&fw[..64]), nb_inputs: 8, nb_outputs: 8, ..Default::default() };
        assert!(!panics(|| compute_glu(&g, &mut [0.0; 8], &x[..8], a)), "{a:?}");
        assert!(panics(|| compute_glu(&g, &mut [0.0; 7], &x[..8], a)), "{a:?}");
        assert!(panics(|| compute_glu(&LinearLayer { nb_inputs: 4, ..g }, &mut [0.0; 8], &x[..8], a)), "{a:?}");
        // conv1d: memory too short, input longer than the layer, a
        // dilation with a layer that is not a whole number of frames.
        let c = LinearLayer { float_weights: Some(&fw[..24 * 8]), nb_inputs: 24, nb_outputs: 8, ..Default::default() };
        let mut out = [0.0f32; 8];
        assert!(!panics(|| compute_generic_conv1d(&c, &mut out, &mut [0.0; 16], &x[..8], Activation::Tanh, a)), "{a:?}");
        assert!(panics(|| compute_generic_conv1d(&c, &mut out, &mut [0.0; 15], &x[..8], Activation::Tanh, a)), "{a:?}");
        assert!(panics(|| compute_generic_conv1d(&c, &mut out, &mut [], &x[..25], Activation::Tanh, a)), "{a:?}");
        assert!(panics(|| compute_generic_conv1d_dilation(&c, &mut out, &mut [0.0; 64], &x[..7], 2, Activation::Tanh, a)), "{a:?}");
        assert!(panics(|| compute_generic_conv1d_dilation(&c, &mut out, &mut [0.0; 31], &x[..8], 2, Activation::Tanh, a)), "{a:?}");
        // conv2d: no weights, short weights, short input, memory, output.
        let w2 = vec![0.1f32; 2 * 9];
        let c2 = Conv2dLayer { float_weights: Some(&w2), in_channels: 1, out_channels: 2, ktime: 3, kheight: 3, ..Default::default() };
        let run2 = |c: Conv2dLayer, out_len: usize, mem_len: usize, in_len: usize| {
            panics(|| {
                let mut out = vec![0.0f32; out_len];
                let mut mem = vec![0.0f32; mem_len];
                compute_conv2d(&c, &mut out, &mut mem, &x[..in_len], 8, 10, Activation::Tanh, a);
            })
        };
        assert!(!run2(c2, 18, 20, 10), "{a:?}");
        assert!(run2(Conv2dLayer { float_weights: None, ..c2 }, 18, 20, 10), "{a:?}");
        assert!(run2(Conv2dLayer { float_weights: Some(&w2[..17]), ..c2 }, 18, 20, 10), "{a:?}");
        assert!(run2(c2, 17, 20, 10) && run2(c2, 18, 19, 10) && run2(c2, 18, 20, 9), "{a:?}");
        assert!(run2(Conv2dLayer { ktime: 0, ..c2 }, 18, 20, 10), "{a:?}");
    }
}

/// `vec_x86::frecpe8` in scalar form: the 8-bit estimate through a float
/// division, for exponent fields 1 to 252, and the special values.
fn frecpe_by_division(x: u32) -> Option<u32> {
    let sign = x & 0x8000_0000;
    let e = (x >> 23) & 0xff;
    if e == 0xff {
        return Some(if x & 0x007f_ffff == 0 { sign } else { x | 0x0040_0000 });
    }
    if e == 0 || e > 252 {
        // The x86 kernel sends these lanes to `frecpe_bits`.
        return None;
    }
    let a = 0x100 | ((x >> 15) & 0xff);
    let b = (524_288.0f32 / (2 * a + 1) as f32) as u32;
    let est = (b + 1) >> 1;
    Some(sign | ((253 - e) << 23) | ((est & 0xff) << 15))
}

#[test]
fn the_x86_frecpe_arithmetic_matches_frecpe_bits() {
    // The estimate depends only on the sign, the exponent and the top 8
    // fraction bits, so every class is checked, each with 3 low parts.
    let mut checked = 0;
    for sign in [0u32, 1] {
        for e in 0..256u32 {
            for top in 0..256u32 {
                for low in [0u32, 0x3a5c, 0x7fff] {
                    let x = sign << 31 | e << 23 | top << 15 | low;
                    if let Some(got) = frecpe_by_division(x) {
                        let want = frecpe_bits(x);
                        let nan = f32::from_bits(want).is_nan();
                        assert!(got == want || (nan && f32::from_bits(got).is_nan()), "{x:08x}: {got:08x} vs {want:08x}");
                        checked += 1;
                    }
                }
            }
        }
    }
    assert!(checked > 380_000);
}

/// `vec_x86::quantize` in scalar form: `cvtps_epi32` gives `i32::MIN` for
/// NaN and for values out of range; values of 2^31 and more are then set
/// to all ones. The low byte is the result.
fn quantize_like_x86(x: f32) -> i8 {
    let p = 127.0f32 * x;
    let cvt = if p.is_nan() || p >= 2_147_483_648.0 || p < -2_147_483_648.0 { i32::MIN } else { p.round_ties_even() as i32 };
    let fixed = if p >= 2_147_483_648.0 { cvt | -1 } else { cvt };
    fixed as i8
}

#[test]
fn the_x86_quantization_matches_quantize() {
    let mut x: Vec<f32> = Vec::new();
    for sign in [0u32, 1] {
        for e in 0..256u32 {
            for f in [0u32, 1, 0x40_0000, 0x7f_ffff, 0x12_3456, 0x55_5555] {
                x.push(f32::from_bits(sign << 31 | e << 23 | f));
            }
        }
    }
    for k in -300..=300 {
        let v = (k as f32 + 0.5) / 127.0;
        x.extend([v, f32::from_bits(v.to_bits() + 1), f32::from_bits(v.to_bits() - 1)]);
    }
    for v in [2_147_483_648.0f32 / 127.0, -2_147_483_648.0 / 127.0, 2.147_483_5e9 / 127.0] {
        x.extend([v, f32::from_bits(v.to_bits() + 1), f32::from_bits(v.to_bits() - 1)]);
    }
    for &v in &x {
        assert_eq!(quantize_like_x86(v), quantize(v), "x = {v:e} ({:08x})", v.to_bits());
        assert_eq!(q(v), quantize(v), "x = {v:e} ({:08x})", v.to_bits());
    }
}
