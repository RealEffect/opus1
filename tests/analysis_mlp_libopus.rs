//! The network of the analysis against `analysis_compute_dense` and
//! `analysis_compute_gru` of libopus, called directly.
//!
//! The weight tables must be the arrays of libopus byte for byte, and each
//! layer must give the same bits as the C function for the same input:
//! random inputs of every scale, inputs where the `tanh` approximation
//! reaches its limits, infinities, NaN, and a long run of the recurrent
//! layer. Layers with other weights (all int8 values, other sizes) reach
//! the approximations of `tanh` and the sigmoid more directly.
//!
//! As for the whole analysis, the bits are those of the Homebrew build on
//! 64-bit ARM, so the tests run only there.
#![cfg(all(feature = "std", feature = "analysis", target_arch = "aarch64"))]

mod analysis_common;

use analysis_common::*;
use opus1::internals::analysis::mlp::{
    DenseLayer, GruLayer, LAYER0, LAYER1, LAYER2, compute_dense, compute_gru,
};
use opus1::internals::analysis::mlp_data::*;

fn c_slice<'a>(p: *const i8, n: usize) -> &'a [i8] {
    // SAFETY: the C layer points to arrays of at least `n` values, which
    // live as long as the program (or as the leaked vectors below).
    unsafe { std::slice::from_raw_parts(p, n) }
}

fn same(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| fbits(*x) == fbits(*y))
}

/// An input value of any scale, and sometimes a special value.
fn input(rng: &mut Lcg) -> f32 {
    match rng.below(20) {
        0 => [0.0, -0.0, f32::INFINITY, f32::NEG_INFINITY, f32::NAN, 1e-40, -1e-40][rng.below(7) as usize],
        1 => rng.uniform() * 1e3,
        2 => rng.uniform() * 1e30,
        3..=6 => rng.uniform() * 10.0,
        7..=9 => rng.uniform() * 0.01,
        _ => rng.uniform(),
    }
}

#[test]
fn weights_are_the_tables_of_libopus() {
    // SAFETY: reading the constant layer structs of libopus.
    let (l0, l1, l2) = unsafe { (&layer0, &layer1, &layer2) };
    assert_eq!((l0.nb_inputs, l0.nb_neurons, l0.sigmoid), (25, 32, 0));
    assert_eq!((l1.nb_inputs, l1.nb_neurons), (32, 24));
    assert_eq!((l2.nb_inputs, l2.nb_neurons, l2.sigmoid), (24, 2, 1));
    assert_eq!(c_slice(l0.input_weights, 800), &LAYER0_WEIGHTS[..]);
    assert_eq!(c_slice(l0.bias, 32), &LAYER0_BIAS[..]);
    assert_eq!(c_slice(l1.input_weights, 2304), &LAYER1_WEIGHTS[..]);
    assert_eq!(c_slice(l1.recurrent_weights, 1728), &LAYER1_RECUR_WEIGHTS[..]);
    assert_eq!(c_slice(l1.bias, 72), &LAYER1_BIAS[..]);
    assert_eq!(c_slice(l2.input_weights, 48), &LAYER2_WEIGHTS[..]);
    assert_eq!(c_slice(l2.bias, 2), &LAYER2_BIAS[..]);

    // The f32 tables of opus1 hold the same values.
    let pairs: [(&[f32], &[i8]); 7] = [
        (LAYER0.input_weights, &LAYER0_WEIGHTS),
        (LAYER0.bias, &LAYER0_BIAS),
        (LAYER1.input_weights, &LAYER1_WEIGHTS),
        (LAYER1.recurrent_weights, &LAYER1_RECUR_WEIGHTS),
        (LAYER1.bias, &LAYER1_BIAS),
        (LAYER2.input_weights, &LAYER2_WEIGHTS),
        (LAYER2.bias, &LAYER2_BIAS),
    ];
    for (f, i) in pairs {
        assert_eq!(f.len(), i.len());
        assert!(f.iter().zip(i).all(|(a, b)| *a == *b as f32));
    }
    assert_eq!((LAYER0.nb_inputs, LAYER0.nb_neurons, LAYER0.sigmoid), (25, 32, false));
    assert_eq!((LAYER1.nb_inputs, LAYER1.nb_neurons), (32, 24));
    assert_eq!((LAYER2.nb_inputs, LAYER2.nb_neurons, LAYER2.sigmoid), (24, 2, true));
}

/// The two dense layers of libopus on 200,000 random inputs each.
#[test]
fn dense_layers_match_libopus() {
    let mut rng = Lcg(1);
    for trial in 0..200_000 {
        let x0: Vec<f32> = (0..25).map(|_| input(&mut rng)).collect();
        let mut c = [0.0f32; 32];
        let mut r = [0.0f32; 32];
        unsafe { analysis_compute_dense(&layer0, c.as_mut_ptr(), x0.as_ptr()) };
        compute_dense(&LAYER0, &mut r, &x0);
        assert!(same(&c, &r), "layer0, trial {trial}: {x0:?}\nC {c:?}\nopus1 {r:?}");

        let x2: Vec<f32> = (0..24).map(|_| input(&mut rng)).collect();
        let mut c = [0.0f32; 2];
        let mut r = [0.0f32; 2];
        unsafe { analysis_compute_dense(&layer2, c.as_mut_ptr(), x2.as_ptr()) };
        compute_dense(&LAYER2, &mut r, &x2);
        assert!(same(&c, &r), "layer2, trial {trial}: {x2:?}\nC {c:?}\nopus1 {r:?}");
    }
}

/// The recurrent layer: 200,000 random single steps, then 10,000 steps that
/// carry the state, as the analysis does, with inputs from layer 0.
#[test]
fn recurrent_layer_matches_libopus() {
    let mut rng = Lcg(2);
    for trial in 0..200_000 {
        let x: Vec<f32> = (0..32).map(|_| input(&mut rng)).collect();
        let s: Vec<f32> = (0..24)
            .map(|_| if rng.below(4) == 0 { input(&mut rng) } else { rng.uniform() })
            .collect();
        let mut c = s.clone();
        let mut r = s.clone();
        unsafe { analysis_compute_gru(&layer1, c.as_mut_ptr(), x.as_ptr()) };
        compute_gru(&LAYER1, &mut r, &x);
        assert!(same(&c, &r), "trial {trial}: x {x:?}, state {s:?}\nC {c:?}\nopus1 {r:?}");
    }
    let mut c = [0.0f32; 32];
    let mut r = [0.0f32; 32];
    for step in 0..10_000 {
        let f: Vec<f32> = (0..25).map(|_| rng.uniform() * 3.0).collect();
        let mut h = [0.0f32; 32];
        compute_dense(&LAYER0, &mut h, &f);
        unsafe { analysis_compute_gru(&layer1, c.as_mut_ptr(), h.as_ptr()) };
        compute_gru(&LAYER1, &mut r, &h);
        assert!(same(&c, &r), "step {step}");
        let mut pc = [0.0f32; 2];
        let mut pr = [0.0f32; 2];
        unsafe { analysis_compute_dense(&layer2, pc.as_mut_ptr(), c.as_ptr()) };
        compute_dense(&LAYER2, &mut pr, &r);
        assert!(same(&pc, &pr), "step {step}");
    }
}

/// A dense layer with given int8 weights for both sides. The vectors are
/// leaked: the opus1 layer needs `'static` slices.
fn dense_pair(
    inputs: usize,
    neurons: usize,
    sigmoid: bool,
    rng: &mut Lcg,
) -> (CDenseLayer, DenseLayer) {
    let w: Vec<i8> = (0..inputs * neurons).map(|_| rng.next_u32() as i8).collect();
    let b: Vec<i8> = (0..neurons).map(|_| rng.next_u32() as i8).collect();
    let wf: &'static [f32] = Box::leak(w.iter().map(|&v| v as f32).collect::<Vec<_>>().into_boxed_slice());
    let bf: &'static [f32] = Box::leak(b.iter().map(|&v| v as f32).collect::<Vec<_>>().into_boxed_slice());
    let w: &'static [i8] = Box::leak(w.into_boxed_slice());
    let b: &'static [i8] = Box::leak(b.into_boxed_slice());
    (
        CDenseLayer {
            bias: b.as_ptr(),
            input_weights: w.as_ptr(),
            nb_inputs: inputs as i32,
            nb_neurons: neurons as i32,
            sigmoid: sigmoid as i32,
        },
        DenseLayer { bias: bf, input_weights: wf, nb_inputs: inputs, nb_neurons: neurons, sigmoid },
    )
}

/// Layers with random int8 weights (the whole range, -128 included) and
/// sizes 1 to 32, so that the output of the `tanh` and sigmoid
/// approximations is seen directly: with one input and a weight of 127,
/// the output is `tanh(127 * x / 128)`. The inputs sweep the range where
/// the rational approximation reaches ±1.
#[test]
fn other_dense_layers_match_libopus() {
    let mut rng = Lcg(3);
    for layer in 0..400 {
        let inputs = 1 + rng.below(32) as usize;
        let neurons = [1, 2, 3, 24, 32, 5, 17][rng.below(7) as usize];
        let sigmoid = rng.below(2) == 0;
        let (c_layer, r_layer) = dense_pair(inputs, neurons, sigmoid, &mut rng);
        for trial in 0..500 {
            let x: Vec<f32> = (0..inputs)
                .map(|_| match rng.below(4) {
                    // Around the point where num * x / den reaches 1.
                    0 => (rng.uniform() * 6.0).clamp(-6.0, 6.0),
                    _ => input(&mut rng),
                })
                .collect();
            let mut c = vec![0.0f32; neurons];
            let mut r = vec![0.0f32; neurons];
            unsafe { analysis_compute_dense(&c_layer, c.as_mut_ptr(), x.as_ptr()) };
            compute_dense(&r_layer, &mut r, &x);
            assert!(same(&c, &r), "layer {layer} ({inputs}x{neurons}), trial {trial}: {x:?}\nC {c:?}\nopus1 {r:?}");
        }
    }
    // One input, weight 127, bias 0: a fine sweep of the approximations.
    for sigmoid in [false, true] {
        let w: &'static [i8] = Box::leak(vec![127i8].into_boxed_slice());
        let b: &'static [i8] = Box::leak(vec![0i8].into_boxed_slice());
        let c_layer = CDenseLayer {
            bias: b.as_ptr(),
            input_weights: w.as_ptr(),
            nb_inputs: 1,
            nb_neurons: 1,
            sigmoid: sigmoid as i32,
        };
        let r_layer = DenseLayer {
            bias: &[0.0],
            input_weights: &[127.0],
            nb_inputs: 1,
            nb_neurons: 1,
            sigmoid,
        };
        let mut x = -20.0f32;
        while x < 20.0 {
            let mut c = [0.0f32];
            let mut r = [0.0f32];
            unsafe { analysis_compute_dense(&c_layer, c.as_mut_ptr(), &x) };
            compute_dense(&r_layer, &mut r, &[x]);
            assert!(same(&c, &r), "x {x}: C {c:?}, opus1 {r:?}");
            x += 0.000_37;
        }
    }
}

/// Recurrent layers with random int8 weights and sizes 1 to 32.
#[test]
fn other_recurrent_layers_match_libopus() {
    let mut rng = Lcg(4);
    for layer in 0..200 {
        let m = 1 + rng.below(32) as usize;
        let n = [1, 2, 7, 24, 32][rng.below(5) as usize];
        let random_i8 = |len: usize, rng: &mut Lcg| -> Vec<i8> { (0..len).map(|_| rng.next_u32() as i8).collect() };
        let w = random_i8(m * 3 * n, &mut rng);
        let rw = random_i8(n * 3 * n, &mut rng);
        let b = random_i8(3 * n, &mut rng);
        let leak_f = |v: &[i8]| -> &'static [f32] {
            Box::leak(v.iter().map(|&x| x as f32).collect::<Vec<_>>().into_boxed_slice())
        };
        let r_layer = GruLayer {
            bias: leak_f(&b),
            input_weights: leak_f(&w),
            recurrent_weights: leak_f(&rw),
            nb_inputs: m,
            nb_neurons: n,
        };
        let (w, rw, b): (&'static [i8], &'static [i8], &'static [i8]) = (
            Box::leak(w.into_boxed_slice()),
            Box::leak(rw.into_boxed_slice()),
            Box::leak(b.into_boxed_slice()),
        );
        let c_layer = CGruLayer {
            bias: b.as_ptr(),
            input_weights: w.as_ptr(),
            recurrent_weights: rw.as_ptr(),
            nb_inputs: m as i32,
            nb_neurons: n as i32,
        };
        let mut c = vec![0.0f32; n];
        let mut r = vec![0.0f32; n];
        for step in 0..300 {
            let x: Vec<f32> = (0..m).map(|_| input(&mut rng)).collect();
            unsafe { analysis_compute_gru(&c_layer, c.as_mut_ptr(), x.as_ptr()) };
            compute_gru(&r_layer, &mut r, &x);
            assert!(same(&c, &r), "layer {layer} ({m}x{n}), step {step}: {x:?}\nC {c:?}\nopus1 {r:?}");
            if rng.below(50) == 0 {
                // Start again from a random state.
                for (a, b) in c.iter_mut().zip(r.iter_mut()) {
                    let v = rng.uniform();
                    *a = v;
                    *b = v;
                }
            }
        }
    }
}
