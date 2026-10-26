//! The small network of the analysis (libopus `src/mlp.c` and `mlp.h`).
//!
//! It maps the 25 features of a frame to two probabilities: music rather
//! than speech, and sound rather than silence or noise. It has three
//! layers:
//!
//! 1. a dense layer, 25 inputs, 32 outputs, `tanh`,
//! 2. a gated recurrent unit (GRU), 32 inputs, 24 outputs, whose state
//!    carries over from frame to frame,
//! 3. a dense layer, 24 inputs, 2 outputs, sigmoid.
//!
//! libopus stores the weights as int8 values (`mlp_data.rs`) and scales the
//! sums by 1/128. Here the weights are converted to `f32` once, when the
//! crate is compiled. Every int8 value is exact in `f32`, so the products
//! and sums are the same bits as in C, and the inner loops skip the
//! conversion.

use super::mlp_data::{
    LAYER0_BIAS, LAYER0_WEIGHTS, LAYER1_BIAS, LAYER1_RECUR_WEIGHTS, LAYER1_WEIGHTS, LAYER2_BIAS,
    LAYER2_WEIGHTS,
};
use crate::mathops::mac;

/// libopus `WEIGHTS_SCALE`: the sums of int8 weights are scaled by this.
pub const WEIGHTS_SCALE: f32 = 1.0 / 128.0;

/// libopus `MAX_NEURONS`: the most outputs of a layer, and the size of the
/// state of the recurrent layer.
pub const MAX_NEURONS: usize = 32;

/// A dense layer (libopus `AnalysisDenseLayer`).
#[derive(Debug)]
pub struct DenseLayer {
    /// One per output.
    pub bias: &'static [f32],
    /// `nb_inputs * nb_neurons`. Input `j`, output `i` at `j * nb_neurons + i`.
    pub input_weights: &'static [f32],
    pub nb_inputs: usize,
    pub nb_neurons: usize,
    /// Sigmoid output when true, `tanh` output when false.
    pub sigmoid: bool,
}

/// A gated recurrent unit (libopus `AnalysisGRULayer`).
///
/// The three gates (update, reset, output) share each row of weights:
/// input `j`, gate `g`, output `i` is at `j * 3 * nb_neurons + g * nb_neurons
/// + i`.
#[derive(Debug)]
pub struct GruLayer {
    /// `3 * nb_neurons`.
    pub bias: &'static [f32],
    /// `nb_inputs * 3 * nb_neurons`.
    pub input_weights: &'static [f32],
    /// `nb_neurons * 3 * nb_neurons`.
    pub recurrent_weights: &'static [f32],
    pub nb_inputs: usize,
    pub nb_neurons: usize,
}

/// The int8 weights as `f32`, at compile time.
const fn to_f32<const N: usize>(w: &[i8; N]) -> [f32; N] {
    let mut out = [0.0f32; N];
    let mut i = 0;
    while i < N {
        out[i] = w[i] as f32;
        i += 1;
    }
    out
}

static L0_BIAS: [f32; 32] = to_f32(&LAYER0_BIAS);
static L0_WEIGHTS: [f32; 800] = to_f32(&LAYER0_WEIGHTS);
static L1_BIAS: [f32; 72] = to_f32(&LAYER1_BIAS);
static L1_WEIGHTS: [f32; 2304] = to_f32(&LAYER1_WEIGHTS);
static L1_RECUR_WEIGHTS: [f32; 1728] = to_f32(&LAYER1_RECUR_WEIGHTS);
static L2_BIAS: [f32; 2] = to_f32(&LAYER2_BIAS);
static L2_WEIGHTS: [f32; 48] = to_f32(&LAYER2_WEIGHTS);

/// libopus `layer0`: dense, 25 inputs, 32 outputs, `tanh`.
pub static LAYER0: DenseLayer = DenseLayer {
    bias: &L0_BIAS,
    input_weights: &L0_WEIGHTS,
    nb_inputs: 25,
    nb_neurons: 32,
    sigmoid: false,
};

/// libopus `layer1`: GRU, 32 inputs, 24 outputs.
pub static LAYER1: GruLayer = GruLayer {
    bias: &L1_BIAS,
    input_weights: &L1_WEIGHTS,
    recurrent_weights: &L1_RECUR_WEIGHTS,
    nb_inputs: 32,
    nb_neurons: 24,
};

/// libopus `layer2`: dense, 24 inputs, 2 outputs, sigmoid.
pub static LAYER2: DenseLayer = DenseLayer {
    bias: &L2_BIAS,
    input_weights: &L2_WEIGHTS,
    nb_inputs: 24,
    nb_neurons: 2,
    sigmoid: true,
};

/// The coefficients of `tansig_approx`, with the digits of libopus: the
/// numerator `N0 + N1 x^2 + N2 x^4` and the denominator `D0 + D1 x^2 +
/// D2 x^4`.
#[allow(clippy::excessive_precision)]
pub(super) const TANSIG_N: [f32; 3] = [952.52801514, 96.39235687, 0.60863042];
#[allow(clippy::excessive_precision)]
pub(super) const TANSIG_D: [f32; 3] = [952.72399902, 413.36801147, 11.88600922];

/// libopus `tansig_approx` (mlp.c:39-53): a rational approximation of
/// `tanh`, limited to -1 to 1.
///
/// The two polynomials are fused multiply-adds, as clang builds the `fmadd`
/// macro of mlp.c. The limit is written as the C ternaries, so that a NaN
/// stays a NaN.
#[inline]
pub fn tansig_approx(x: f32) -> f32 {
    let [n0, n1, n2] = TANSIG_N;
    let [d0, d1, d2] = TANSIG_D;
    let x2 = x * x;
    let num = mac(n0, mac(n1, n2, x2), x2);
    let den = mac(d0, mac(d1, d2, x2), x2);
    let num = num * x / den;
    // MAX32(-1.f, MIN32(1.f, num))
    let num = if 1.0 < num { 1.0 } else { num };
    if -1.0 > num { -1.0 } else { num }
}

/// libopus `sigmoid_approx` (mlp.c:55-58): `.5 + .5 * tanh(.5 * x)`, the
/// last step fused.
#[inline]
pub fn sigmoid_approx(x: f32) -> f32 {
    mac(0.5, 0.5, tansig_approx(0.5 * x))
}

/// libopus `gemm_accum` (mlp.c:60-68): `out[i] += w[j * stride + i] * x[j]`
/// for `i < out.len()` and `j < x.len()`, one fused multiply-add each.
///
/// Each `out[i]` adds its products in the order of `j`, as in C. C runs the
/// loop over `i` outside; here it runs inside, which gives the same bits,
/// because the outputs do not depend on each other.
///
/// The layer sizes of libopus (32, 24 and 2 outputs) get code with a fixed
/// size, which keeps the sums in registers. Measured on the Apple M4 Pro,
/// the whole network then takes about 0.31 µs per frame instead of
/// 0.85 µs. (libopus takes about 1.5 µs.)
#[inline]
fn gemm_accum(out: &mut [f32], w: &[f32], stride: usize, x: &[f32]) {
    match out.len() {
        32 => gemm_accum_fixed::<32>(out, w, stride, x),
        24 => gemm_accum_fixed::<24>(out, w, stride, x),
        2 => gemm_accum_fixed::<2>(out, w, stride, x),
        rows => {
            for (j, &xj) in x.iter().enumerate() {
                let row = &w[j * stride..j * stride + rows];
                for (o, &wi) in out.iter_mut().zip(row) {
                    *o = mac(*o, wi, xj);
                }
            }
        }
    }
}

/// `gemm_accum` for `R` outputs.
#[inline(always)]
fn gemm_accum_fixed<const R: usize>(out: &mut [f32], w: &[f32], stride: usize, x: &[f32]) {
    let out: &mut [f32; R] = out.try_into().expect("R outputs");
    let mut acc = *out;
    for (j, &xj) in x.iter().enumerate() {
        let row: &[f32; R] = w[j * stride..j * stride + R].try_into().expect("R weights");
        for i in 0..R {
            acc[i] = mac(acc[i], row[i], xj);
        }
    }
    *out = acc;
}

/// libopus `analysis_compute_dense` (mlp.c:70-90).
///
/// `output` gets `layer.nb_neurons` values; `input` holds `layer.nb_inputs`.
pub fn compute_dense(layer: &DenseLayer, output: &mut [f32], input: &[f32]) {
    let n = layer.nb_neurons;
    let m = layer.nb_inputs;
    let output = &mut output[..n];
    output.copy_from_slice(&layer.bias[..n]);
    gemm_accum(output, layer.input_weights, n, &input[..m]);
    // SAFETY: NEON is part of the baseline of every aarch64 target.
    #[cfg(target_arch = "aarch64")]
    let output = unsafe { super::neon::scale_and_activate(output, layer.sigmoid) };
    // SAFETY: SSE2 is part of this target (see the `cfg`).
    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    let output = unsafe { super::x86::scale_and_activate(output, layer.sigmoid) };
    for o in output.iter_mut() {
        *o *= WEIGHTS_SCALE;
    }
    if layer.sigmoid {
        for o in output.iter_mut() {
            *o = sigmoid_approx(*o);
        }
    } else {
        for o in output.iter_mut() {
            *o = tansig_approx(*o);
        }
    }
}

/// libopus `analysis_compute_gru` (mlp.c:92-131): one step of the gated
/// recurrent unit. `state` holds `gru.nb_neurons` values and is updated;
/// `input` holds `gru.nb_inputs`.
pub fn compute_gru(gru: &GruLayer, state: &mut [f32], input: &[f32]) {
    #[cfg(target_arch = "aarch64")]
    if gru.nb_inputs == super::neon::GRU_INPUTS && gru.nb_neurons == super::neon::GRU_NEURONS {
        let state = (&mut state[..super::neon::GRU_NEURONS]).try_into().expect("24 values");
        // SAFETY: NEON is part of the baseline of every aarch64 target.
        unsafe { super::neon::gru(gru, state, input) };
        return;
    }
    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    if gru.nb_inputs == super::x86::GRU_INPUTS && gru.nb_neurons == super::x86::GRU_NEURONS {
        let state = (&mut state[..super::x86::GRU_NEURONS]).try_into().expect("24 values");
        // SAFETY: SSE2 is part of this target (see the `cfg`).
        unsafe { super::x86::gru(gru, state, input) };
        return;
    }
    compute_gru_scalar(gru, state, input);
}

/// `compute_gru` one output at a time. 64-bit ARM runs a NEON version and
/// x86-64 an SSE version for the size of the layer of libopus; this one is
/// their reference in the tests.
pub fn compute_gru_scalar(gru: &GruLayer, state: &mut [f32], input: &[f32]) {
    let n = gru.nb_neurons;
    let m = gru.nb_inputs;
    let stride = 3 * n;
    let input = &input[..m];
    let state = &mut state[..n];
    let mut z = [0.0f32; MAX_NEURONS];
    let mut r = [0.0f32; MAX_NEURONS];
    let mut h = [0.0f32; MAX_NEURONS];
    let mut tmp = [0.0f32; MAX_NEURONS];
    let (z, r, h, tmp) = (&mut z[..n], &mut r[..n], &mut h[..n], &mut tmp[..n]);

    // Update gate.
    z.copy_from_slice(&gru.bias[..n]);
    gemm_accum(z, gru.input_weights, stride, input);
    gemm_accum(z, gru.recurrent_weights, stride, state);
    for v in z.iter_mut() {
        *v = sigmoid_approx(WEIGHTS_SCALE * *v);
    }

    // Reset gate.
    r.copy_from_slice(&gru.bias[n..2 * n]);
    gemm_accum(r, &gru.input_weights[n..], stride, input);
    gemm_accum(r, &gru.recurrent_weights[n..], stride, state);
    for v in r.iter_mut() {
        *v = sigmoid_approx(WEIGHTS_SCALE * *v);
    }

    // Output.
    h.copy_from_slice(&gru.bias[2 * n..3 * n]);
    for i in 0..n {
        tmp[i] = state[i] * r[i];
    }
    gemm_accum(h, &gru.input_weights[2 * n..], stride, input);
    gemm_accum(h, &gru.recurrent_weights[2 * n..], stride, tmp);
    for i in 0..n {
        // z*state + (1-z)*tanh(h): the left product fused, the right one
        // rounded, as clang contracts the C expression.
        h[i] = mac((1.0 - z[i]) * tansig_approx(WEIGHTS_SCALE * h[i]), z[i], state[i]);
    }
    state.copy_from_slice(h);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layer_sizes_match_the_tables() {
        for l in [&LAYER0, &LAYER2] {
            assert_eq!(l.bias.len(), l.nb_neurons);
            assert_eq!(l.input_weights.len(), l.nb_inputs * l.nb_neurons);
            assert!(l.nb_neurons <= MAX_NEURONS);
        }
        let g = &LAYER1;
        assert_eq!(g.bias.len(), 3 * g.nb_neurons);
        assert_eq!(g.input_weights.len(), 3 * g.nb_inputs * g.nb_neurons);
        assert_eq!(g.recurrent_weights.len(), 3 * g.nb_neurons * g.nb_neurons);
        assert_eq!(LAYER0.nb_neurons, LAYER1.nb_inputs);
        assert_eq!(LAYER1.nb_neurons, LAYER2.nb_inputs);
    }

    #[test]
    fn f32_tables_hold_the_int8_values() {
        for (a, b) in L1_WEIGHTS.iter().zip(LAYER1_WEIGHTS.iter()) {
            assert_eq!(*a, *b as f32);
        }
        for (a, b) in L0_WEIGHTS.iter().zip(LAYER0_WEIGHTS.iter()) {
            assert_eq!(*a, *b as f32);
        }
    }

    #[test]
    fn tansig_is_limited_and_odd() {
        assert_eq!(tansig_approx(0.0), 0.0);
        assert_eq!(tansig_approx(100.0), 1.0);
        assert_eq!(tansig_approx(-100.0), -1.0);
        assert!(tansig_approx(f32::NAN).is_nan());
        for i in 0..1000 {
            let x = i as f32 * 0.01;
            assert_eq!(tansig_approx(-x), -tansig_approx(x));
        }
        assert_eq!(sigmoid_approx(0.0), 0.5);
    }
}
