//! The layer types and layer functions of libopus (`dnn/nnet.h`,
//! `dnn/nnet.c`, `dnn/nnet_arch.h`).
//!
//! Each function checks its slices with `assert!`, then runs the kernels of
//! the given `DnnArch`. Every `DnnArch` gives the same bits, and a variant
//! that the processor cannot run is replaced by one it can run
//! (`arch::usable`). Nothing here allocates: the scratch buffers are
//! fixed-size arrays on the stack, sized by the constants of `crate::dnn`.
//!
//! Slice rules (`nnet.md`, Review item 8). libopus callers often pass
//! buffers that are longer than a layer reads, so the checks are "at
//! least":
//!
//! - inputs may be longer than `nb_inputs`; only the first `nb_inputs`
//!   values are read;
//! - outputs may be longer than `nb_outputs`; only the first `nb_outputs`
//!   values are written;
//! - the memory of `compute_generic_conv1d` may be empty (libopus passes
//!   `NULL` when the layer has no memory);
//! - the sizes of `compute_conv2d` are minimums.
//!
//! The roundings follow libopus 1.6.1 on 64-bit ARM, built without loop
//! vectorization (`nnet.md` sections 3.4 and 3.5): the steps that clang
//! contracts into a fused multiply-add use `fma`, every other step rounds
//! on its own.

use super::arch::{self, DnnArch};
use super::vec_portable::{self, fma};
use super::{Activation, MAX_CONV_INPUTS_ALL, MAX_INPUTS, MAX_RNN_NEURONS_ALL};

/// One affine layer `y = W·x + b` (libopus `LinearLayer`, `dnn/nnet.h:66-76`).
///
/// The layer borrows its arrays from a weight blob. A layer has int8
/// weights (`weights` with `scale`), float weights (`float_weights`), or
/// both; the float weights win when both are present, as in libopus.
///
/// Layout of the weights:
/// - `weights` (int8): blocks of 8 rows by 4 columns, 32 bytes each; byte
///   `4·k + c` of a block is the weight of row `i + k` and column `j + c`.
///   A dense layer stores the blocks of rows 0-7 for columns 0, 4, 8 and so
///   on, then those of rows 8-15. A sparse layer stores one block for each
///   position of its index table, in the order of the table.
/// - `float_weights`: column-major, `w[j · nb_outputs + i]` for row `i` and
///   column `j`. For a sparse layer, 32 floats for each position of the
///   index table; float `8·c + k` of a block is the weight of row `i + k`
///   and column `pos + c`.
/// - `weights_idx`: for a sparse layer, for each block of 8 rows a count `n`
///   and then `n` column positions (multiples of 4).
///
/// The int8 weights must lie in [-127, 127], as the weight loader checks.
/// With -128 the kernel variants can disagree (`nnet.md` section 6.5), but
/// nothing reads out of bounds.
///
/// The model code also reads `float_weights` directly as an embedding table
/// for three layers (`lace_pitch_embedding`, `nolace_pitch_embedding`,
/// `cond_net_pembed`), so the fields are public.
#[derive(Clone, Copy, Default)]
pub struct LinearLayer<'w> {
    /// One value per output, added after the product. `None` for a few GRU
    /// projections of FARGAN.
    pub bias: Option<&'w [f32]>,
    /// The bias for unsigned inputs that libopus uses on x86 only. opus1
    /// never reads it; it is kept so that a layer matches the C struct.
    pub subias: Option<&'w [f32]>,
    /// int8 weights, 8 × 4 blocked.
    pub weights: Option<&'w [i8]>,
    /// float weights, column-major (or 8 × 4 blocked when sparse).
    pub float_weights: Option<&'w [f32]>,
    /// The index table of a sparse layer.
    pub weights_idx: Option<&'w [i32]>,
    /// The diagonal of a GRU recurrent matrix (`3 · nb_inputs` values). No
    /// 1.6.1 model has one.
    pub diag: Option<&'w [f32]>,
    /// One factor per output for the int8 weights.
    pub scale: Option<&'w [f32]>,
    /// The input count `M`.
    pub nb_inputs: usize,
    /// The output count `N`.
    pub nb_outputs: usize,
}

/// A two-dimensional convolution (libopus `Conv2dLayer`,
/// `dnn/nnet.h:79-86`). Only PitchDNN uses one (two 3 × 3 layers).
#[derive(Clone, Copy, Default)]
pub struct Conv2dLayer<'w> {
    /// One value per output channel.
    pub bias: Option<&'w [f32]>,
    /// `out_channels · in_channels · ktime · kheight` weights: weight `k` of
    /// output channel `i` and input channel `m` is
    /// `float_weights[i·in_channels·ktime·kheight + m·ktime·kheight + k]`,
    /// with `k = kheight·t + h`.
    pub float_weights: Option<&'w [f32]>,
    /// Input channel count.
    pub in_channels: usize,
    /// Output channel count.
    pub out_channels: usize,
    /// Kernel size along time (the frames kept in the memory plus one).
    pub ktime: usize,
    /// Kernel size along the height (frequency) axis.
    pub kheight: usize,
}

// ---------------------------------------------------------------------------
// Kernel dispatch
// ---------------------------------------------------------------------------

/// The kernels of one `DnnArch` that this processor can run.
///
/// It can only be made with `Kernels::new`, which replaces a variant the
/// processor cannot run (`arch::usable`). So its methods may call the
/// kernels of its variant, which are compiled for the instructions of that
/// variant (`#[target_feature]`): the `unsafe` block around each call only
/// states that the processor has them. The methods check the slices first;
/// the kernels check them again, and the kernels of sparse layers check
/// the index table (`check_sparse`).
#[derive(Clone, Copy)]
struct Kernels(DnnArch);

/// Checks a sparse index table (the layout of `sparse_sgemv8x4`) for `rows`
/// rows and inputs of length `cols`, and returns the number of blocks it
/// lists. Panics when a count or a position is negative, a block would read
/// past `cols`, or the table is too short.
fn sparse_blocks(idx: &[i32], rows: usize, cols: usize) -> usize {
    assert!(rows.is_multiple_of(8), "a sparse layer needs a multiple of 8 outputs");
    let mut p = 0;
    let mut total = 0;
    for _ in 0..rows / 8 {
        let count = idx[p];
        assert!(count >= 0, "negative block count in a sparse index table");
        p += 1;
        for &pos in &idx[p..p + count as usize] {
            assert!(
                pos >= 0 && pos as usize + 4 <= cols,
                "position out of range in a sparse index table"
            );
        }
        p += count as usize;
        total += count as usize;
    }
    total
}

/// Checks the index table of a sparse layer with `sparse_blocks` (`rows`
/// rows, inputs of length `cols`), and that `w_len` weights hold 32 for
/// each block the table lists. The sparse kernels of every variant run
/// only after this check: those of `vec_neon` and `vec_x86` call it
/// themselves and then read the table, the weights and the inputs at the
/// listed positions without index checks.
///
/// Panics when the table is invalid or the weights are too short.
#[inline]
pub(crate) fn check_sparse(idx: &[i32], rows: usize, cols: usize, w_len: usize) {
    let blocks = sparse_blocks(idx, rows, cols);
    assert!(blocks.checked_mul(32).is_some_and(|n| w_len >= n), "sparse weights shorter than the index table");
}

impl Kernels {
    #[inline]
    fn new(arch: DnnArch) -> Kernels {
        Kernels(arch::usable(arch))
    }

    /// `vec_portable::sgemv` on this variant.
    fn sgemv(self, out: &mut [f32], w: &[f32], rows: usize, cols: usize, stride: usize, x: &[f32]) {
        assert!(out.len() >= rows && x.len() >= cols);
        if rows == 0 {
            return;
        }
        if cols == 0 {
            // No products: every chain stays at +0. The kernels are not
            // called.
            out[..rows].fill(0.0);
            return;
        }
        // Checked, so that a huge shape cannot wrap around and pass.
        let need = (cols - 1).checked_mul(stride).and_then(|n| n.checked_add(rows));
        assert!(stride >= rows && need.is_some_and(|n| w.len() >= n), "float weights shorter than the layer");
        match self.0 {
            #[cfg(target_arch = "aarch64")]
            // SAFETY: NEON is part of every aarch64 target.
            DnnArch::Neon | DnnArch::NeonDotprod => unsafe {
                super::vec_neon::sgemv(out, w, rows, cols, stride, x)
            },
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // SAFETY: `Kernels::new` keeps these variants only on a
            // processor with AVX2 and FMA.
            DnnArch::Avx2Fma | DnnArch::AvxVnni | DnnArch::Avx512Vnni => unsafe {
                super::vec_x86::sgemv_avx2(out, w, rows, cols, stride, x)
            },
            _ => vec_portable::sgemv(out, w, rows, cols, stride, x),
        }
    }

    /// `vec_portable::sparse_sgemv8x4` on this variant.
    fn sparse_sgemv8x4(self, out: &mut [f32], w: &[f32], idx: &[i32], rows: usize, x: &[f32]) {
        assert!(out.len() >= rows);
        match self.0 {
            #[cfg(target_arch = "aarch64")]
            // SAFETY: NEON is part of every aarch64 target.
            DnnArch::Neon | DnnArch::NeonDotprod => unsafe {
                super::vec_neon::sparse_sgemv8x4(out, w, idx, rows, x)
            },
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // SAFETY: `Kernels::new` keeps these variants only on a
            // processor with AVX2 and FMA.
            DnnArch::Avx2Fma | DnnArch::AvxVnni | DnnArch::Avx512Vnni => unsafe {
                super::vec_x86::sparse_sgemv8x4_avx2(out, w, idx, rows, x)
            },
            _ => {
                check_sparse(idx, rows, x.len(), w.len());
                vec_portable::sparse_sgemv8x4(out, w, idx, rows, x)
            }
        }
    }

    /// `vec_portable::cgemv8x4` on this variant.
    #[allow(clippy::too_many_arguments)]
    fn cgemv8x4(self, out: &mut [f32], w: &[i8], scale: &[f32], rows: usize, cols: usize, x: &[f32]) {
        assert!(rows.is_multiple_of(8) && cols.is_multiple_of(4), "an int8 layer needs 8·k outputs and 4·k inputs");
        assert!(cols <= MAX_INPUTS);
        assert!(out.len() >= rows && x.len() >= cols && scale.len() >= rows);
        assert!(rows.checked_mul(cols).is_some_and(|n| w.len() >= n), "int8 weights shorter than the layer");
        match self.0 {
            #[cfg(target_arch = "aarch64")]
            // SAFETY: NEON is part of every aarch64 target.
            DnnArch::Neon => unsafe { super::vec_neon::cgemv8x4(out, w, scale, rows, cols, x) },
            #[cfg(target_arch = "aarch64")]
            // SAFETY: as above; `Kernels::new` keeps `NeonDotprod` only on a
            // processor with DotProd.
            DnnArch::NeonDotprod => unsafe {
                super::vec_neon::cgemv8x4_dotprod(out, w, scale, rows, cols, x)
            },
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // SAFETY: `Kernels::new` keeps `Avx2Fma` only on a processor with
            // AVX2 and FMA.
            DnnArch::Avx2Fma => unsafe { super::vec_x86::cgemv8x4_avx2(out, w, scale, rows, cols, x) },
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // SAFETY: `Kernels::new` keeps `AvxVnni` only on a processor with
            // AVX2, FMA and AVX-VNNI.
            DnnArch::AvxVnni => unsafe {
                super::vec_x86::cgemv8x4_avxvnni(out, w, scale, rows, cols, x)
            },
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // SAFETY: `Kernels::new` keeps `Avx512Vnni` only on a processor
            // with AVX2, FMA and AVX-512 F, BW and VNNI.
            DnnArch::Avx512Vnni => unsafe {
                super::vec_x86::cgemv8x4_avx512vnni(out, w, scale, rows, cols, x)
            },
            _ => vec_portable::cgemv8x4(out, w, scale, rows, cols, x),
        }
    }

    /// `vec_portable::sparse_cgemv8x4` on this variant.
    #[allow(clippy::too_many_arguments)]
    fn sparse_cgemv8x4(
        self,
        out: &mut [f32],
        w: &[i8],
        idx: &[i32],
        scale: &[f32],
        rows: usize,
        cols: usize,
        x: &[f32],
    ) {
        assert!(cols <= MAX_INPUTS);
        assert!(out.len() >= rows && x.len() >= cols && scale.len() >= rows);
        match self.0 {
            #[cfg(target_arch = "aarch64")]
            // SAFETY: NEON is part of every aarch64 target.
            DnnArch::Neon => unsafe {
                super::vec_neon::sparse_cgemv8x4(out, w, idx, scale, rows, cols, x)
            },
            #[cfg(target_arch = "aarch64")]
            // SAFETY: as above; `Kernels::new` keeps `NeonDotprod` only on a
            // processor with DotProd.
            DnnArch::NeonDotprod => unsafe {
                super::vec_neon::sparse_cgemv8x4_dotprod(out, w, idx, scale, rows, cols, x)
            },
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // SAFETY: `Kernels::new` keeps `Avx2Fma` only on a processor with
            // AVX2 and FMA.
            DnnArch::Avx2Fma => unsafe {
                super::vec_x86::sparse_cgemv8x4_avx2(out, w, idx, scale, rows, cols, x)
            },
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // SAFETY: `Kernels::new` keeps `AvxVnni` only on a processor with
            // AVX2, FMA and AVX-VNNI.
            DnnArch::AvxVnni => unsafe {
                super::vec_x86::sparse_cgemv8x4_avxvnni(out, w, idx, scale, rows, cols, x)
            },
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // SAFETY: `Kernels::new` keeps `Avx512Vnni` only on a processor
            // with AVX2, FMA and AVX-512 F, BW and VNNI.
            DnnArch::Avx512Vnni => unsafe {
                super::vec_x86::sparse_cgemv8x4_avx512vnni(out, w, idx, scale, rows, cols, x)
            },
            _ => {
                check_sparse(idx, rows, cols, w.len());
                vec_portable::sparse_cgemv8x4(out, w, idx, scale, rows, cols, x)
            }
        }
    }

    /// `vec_portable::vec_exp` on this variant.
    fn vec_exp(self, buf: &mut [f32]) {
        match self.0 {
            #[cfg(target_arch = "aarch64")]
            // SAFETY: NEON is part of every aarch64 target.
            DnnArch::Neon | DnnArch::NeonDotprod => unsafe { super::vec_neon::vec_exp(buf) },
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // SAFETY: `Kernels::new` keeps these variants only on a
            // processor with AVX2 and FMA.
            DnnArch::Avx2Fma | DnnArch::AvxVnni | DnnArch::Avx512Vnni => unsafe {
                super::vec_x86::vec_exp_avx2(buf)
            },
            _ => vec_portable::vec_exp(buf),
        }
    }

    /// `vec_portable::vec_tanh` on this variant.
    fn vec_tanh(self, buf: &mut [f32]) {
        match self.0 {
            #[cfg(target_arch = "aarch64")]
            // SAFETY: NEON is part of every aarch64 target.
            DnnArch::Neon | DnnArch::NeonDotprod => unsafe { super::vec_neon::vec_tanh(buf) },
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // SAFETY: `Kernels::new` keeps these variants only on a
            // processor with AVX2 and FMA.
            DnnArch::Avx2Fma | DnnArch::AvxVnni | DnnArch::Avx512Vnni => unsafe {
                super::vec_x86::vec_tanh_avx2(buf)
            },
            _ => vec_portable::vec_tanh(buf),
        }
    }

    /// `vec_portable::vec_sigmoid` on this variant.
    fn vec_sigmoid(self, buf: &mut [f32]) {
        match self.0 {
            #[cfg(target_arch = "aarch64")]
            // SAFETY: NEON is part of every aarch64 target.
            DnnArch::Neon | DnnArch::NeonDotprod => unsafe { super::vec_neon::vec_sigmoid(buf) },
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // SAFETY: `Kernels::new` keeps these variants only on a
            // processor with AVX2 and FMA.
            DnnArch::Avx2Fma | DnnArch::AvxVnni | DnnArch::Avx512Vnni => unsafe {
                super::vec_x86::vec_sigmoid_avx2(buf)
            },
            _ => vec_portable::vec_sigmoid(buf),
        }
    }

    /// libopus `compute_linear` on `out[..N]` and `input[..M]`.
    fn linear(self, l: &LinearLayer, out: &mut [f32], input: &[f32]) {
        let m = l.nb_inputs;
        let n = l.nb_outputs;
        assert!(input.len() >= m, "input shorter than the layer's nb_inputs");
        assert!(out.len() >= n, "output shorter than the layer's nb_outputs");
        let input = &input[..m];
        let out = &mut out[..n];
        if let Some(fw) = l.float_weights {
            match l.weights_idx {
                Some(idx) => self.sparse_sgemv8x4(out, fw, idx, n, input),
                None => self.sgemv(out, fw, n, m, n, input),
            }
        } else if let Some(w) = l.weights {
            let scale = l.scale.expect("an int8 layer needs a scale");
            match l.weights_idx {
                Some(idx) => self.sparse_cgemv8x4(out, w, idx, scale, n, m, input),
                None => self.cgemv8x4(out, w, scale, n, m, input),
            }
            // libopus uses `subias` here only on x86 (`USE_SU_BIAS`); the
            // 64-bit ARM build keeps `bias`.
        } else {
            out.fill(0.0);
        }
        if let Some(bias) = l.bias {
            for (o, &b) in out.iter_mut().zip(&bias[..n]) {
                *o += b;
            }
        }
        if let Some(diag) = l.diag {
            // Only GRU recurrent weights have a diagonal.
            assert!(3 * m == n, "a diagonal needs nb_outputs == 3·nb_inputs");
            let diag = &diag[..n];
            for (i, &x) in input.iter().enumerate() {
                out[i] = fma(diag[i], x, out[i]);
                out[i + m] = fma(diag[i + m], x, out[i + m]);
                out[i + 2 * m] = fma(diag[i + 2 * m], x, out[i + 2 * m]);
            }
        }
    }

    /// libopus `compute_activation` with `output == input`.
    fn activation(self, buf: &mut [f32], act: Activation) {
        match act {
            Activation::Sigmoid => self.vec_sigmoid(buf),
            Activation::Tanh => self.vec_tanh(buf),
            Activation::Swish => {
                // libopus computes the sigmoid of all values into a buffer,
                // then `y = x · sigmoid(x)`. Pieces of a multiple of 4 values
                // give the same groups of 4 and the same tail, without the
                // buffer of 4,096 values.
                const PIECE: usize = 64;
                for piece in buf.chunks_mut(PIECE) {
                    let mut s = [0.0f32; PIECE];
                    let s = &mut s[..piece.len()];
                    s.copy_from_slice(piece);
                    self.vec_sigmoid(s);
                    for (y, &g) in piece.iter_mut().zip(s.iter()) {
                        *y *= g;
                    }
                }
            }
            Activation::Relu => {
                for v in buf.iter_mut() {
                    // `x < 0 ? 0 : x`: -0.0 and NaN pass through.
                    if *v < 0.0 {
                        *v = 0.0;
                    }
                }
            }
            Activation::Softmax => {
                // `SOFTMAX_HACK` of nnet.c has no effect on nnet_arch.h, so
                // this is a real softmax (nnet.md section 3.4).
                self.vec_exp(buf);
                let mut sum = 0.0f32;
                for &v in buf.iter() {
                    sum += v;
                }
                // The sum is a float, added in order. In
                // `sum = 1.f/(sum+1e-30)`, `1e-30` is a double, so the
                // addition and the division are in double precision.
                let s = (1.0f64 / (sum as f64 + 1e-30)) as f32;
                for v in buf.iter_mut() {
                    *v *= s;
                }
            }
            Activation::Exp => self.vec_exp(buf),
            Activation::Linear => {}
        }
    }
}

// ---------------------------------------------------------------------------
// Layer functions
// ---------------------------------------------------------------------------

/// `out = W·input + bias` (libopus `compute_linear`, `dnn/nnet_arch.h:122-154`),
/// plus `diag[i]·input[i]` for each gate when the layer has a diagonal.
///
/// Reads `input[..nb_inputs]` and writes `out[..nb_outputs]`; both slices
/// may be longer. A layer without weights writes only its bias (or zeros).
///
/// Panics when a slice is too short, when an int8 layer does not have a
/// multiple of 8 outputs and of 4 inputs (dense) or more than `MAX_INPUTS`
/// inputs, when an int8 layer has no `scale`, or when a sparse index table
/// is invalid.
pub fn compute_linear(l: &LinearLayer, out: &mut [f32], input: &[f32], arch: DnnArch) {
    Kernels::new(arch).linear(l, out, input);
}

/// `out = act(input)` (libopus `compute_activation`,
/// `dnn/nnet_arch.h:69-119`), for `input.len()` values. Requires
/// `out.len() >= input.len()`.
///
/// The values are processed in groups of 4, then the last `len % 4` values
/// with the tail formulas of libopus, which round differently.
pub fn compute_activation(out: &mut [f32], input: &[f32], act: Activation, arch: DnnArch) {
    let out = &mut out[..input.len()];
    out.copy_from_slice(input);
    Kernels::new(arch).activation(out, act);
}

/// `buf = act(buf)`: `compute_activation` with the output in place of the
/// input, as libopus calls it with `output == input`.
pub fn compute_activation_in_place(buf: &mut [f32], act: Activation, arch: DnnArch) {
    Kernels::new(arch).activation(buf, act);
}

/// `out = act(W·input + bias)` (libopus `compute_generic_dense`,
/// `dnn/nnet.c:60-64`). Reads `input[..nb_inputs]` and writes
/// `out[..nb_outputs]`.
pub fn compute_generic_dense(
    l: &LinearLayer,
    out: &mut [f32],
    input: &[f32],
    act: Activation,
    arch: DnnArch,
) {
    let k = Kernels::new(arch);
    k.linear(l, out, input);
    k.activation(&mut out[..l.nb_outputs], act);
}

/// One step of a gated recurrent unit (libopus `compute_generic_gru`,
/// `dnn/nnet.c:76-105`). `state[..N]` is updated in place, with
/// `N = recurrent_w.nb_inputs`.
///
/// Requires `recurrent_w.nb_outputs == input_w.nb_outputs == 3·N`,
/// `N <= MAX_RNN_NEURONS_ALL`, `state.len() >= N` and
/// `input.len() >= input_w.nb_inputs`.
///
/// The roundings (`nnet.md` section 3.5): `z` and `r` are
/// `sigmoid(input part + recurrent part)` with a plain add;
/// `h = tanh(fma(recur_h, r, input_h))`; the new state is
/// `fma(z, state, (1 - z)·h)`.
pub fn compute_generic_gru(
    input_w: &LinearLayer,
    recurrent_w: &LinearLayer,
    state: &mut [f32],
    input: &[f32],
    arch: DnnArch,
) {
    let n = recurrent_w.nb_inputs;
    assert!(3 * n == recurrent_w.nb_outputs, "a GRU needs recurrent nb_outputs == 3·nb_inputs");
    assert!(input_w.nb_outputs == recurrent_w.nb_outputs, "the two GRU layers differ in size");
    assert!(n <= MAX_RNN_NEURONS_ALL, "GRU larger than MAX_RNN_NEURONS_ALL");
    let k = Kernels::new(arch);
    let state = &mut state[..n];
    let mut zrh = [0.0f32; 3 * MAX_RNN_NEURONS_ALL];
    let mut recur = [0.0f32; 3 * MAX_RNN_NEURONS_ALL];
    let zrh = &mut zrh[..3 * n];
    let recur = &mut recur[..3 * n];
    k.linear(input_w, zrh, input);
    k.linear(recurrent_w, recur, state);
    for (a, &b) in zrh[..2 * n].iter_mut().zip(&recur[..2 * n]) {
        *a += b;
    }
    k.activation(&mut zrh[..2 * n], Activation::Sigmoid);
    let (zr, h) = zrh.split_at_mut(2 * n);
    let (z, r) = zr.split_at(n);
    for i in 0..n {
        h[i] = fma(recur[2 * n + i], r[i], h[i]);
    }
    k.activation(h, Activation::Tanh);
    for i in 0..n {
        // `h = z·state + (1 - z)·h`: clang fuses the first product.
        state[i] = fma(z[i], state[i], (1.0 - z[i]) * h[i]);
    }
}

/// Gated linear unit: `out[i] = input[i] · sigmoid(W·input + b)[i]`
/// (libopus `compute_glu`, `dnn/nnet.c:107-120`). Requires
/// `l.nb_inputs == l.nb_outputs <= MAX_INPUTS`; reads `input[..N]` and
/// writes `out[..N]`.
pub fn compute_glu(l: &LinearLayer, out: &mut [f32], input: &[f32], arch: DnnArch) {
    let n = l.nb_outputs;
    assert!(l.nb_inputs == n, "a GLU needs nb_inputs == nb_outputs");
    assert!(n <= MAX_INPUTS);
    let k = Kernels::new(arch);
    let mut act = [0.0f32; MAX_INPUTS];
    let act = &mut act[..n];
    k.linear(l, act, input);
    k.activation(act, Activation::Sigmoid);
    for ((o, &x), &a) in out[..n].iter_mut().zip(&input[..n]).zip(act.iter()) {
        *o = x * a;
    }
}

/// `compute_glu` in place, as FARGAN calls it (`dnn/fargan.c:106`, `:132`).
/// Uses `buf[..N]`.
pub fn compute_glu_in_place(l: &LinearLayer, buf: &mut [f32], arch: DnnArch) {
    let n = l.nb_outputs;
    assert!(l.nb_inputs == n, "a GLU needs nb_inputs == nb_outputs");
    assert!(n <= MAX_INPUTS);
    let k = Kernels::new(arch);
    let mut act = [0.0f32; MAX_INPUTS];
    let act = &mut act[..n];
    k.linear(l, act, buf);
    k.activation(act, Activation::Sigmoid);
    for (o, &a) in buf[..n].iter_mut().zip(act.iter()) {
        *o *= a;
    }
}

/// One-dimensional convolution over time (libopus `compute_generic_conv1d`,
/// `dnn/nnet.c:124-134`): a dense layer over `[mem, input]`, after which
/// `mem` keeps the last `nb_inputs - input.len()` values of `[mem, input]`.
///
/// The input size is `input.len()`. Requires
/// `input.len() <= l.nb_inputs <= MAX_CONV_INPUTS_ALL`,
/// `mem.len() >= l.nb_inputs - input.len()` (it may be empty when they are
/// equal; libopus then passes `NULL`) and `out.len() >= l.nb_outputs`. Only
/// `mem[..nb_inputs - input.len()]` is read and written.
pub fn compute_generic_conv1d(
    l: &LinearLayer,
    out: &mut [f32],
    mem: &mut [f32],
    input: &[f32],
    act: Activation,
    arch: DnnArch,
) {
    compute_generic_conv1d_dilation(l, out, mem, input, 1, act, arch);
}

/// `compute_generic_conv1d` with a dilation (libopus
/// `compute_generic_conv1d_dilation`, `dnn/nnet.c:136-153`). With
/// `ksize = l.nb_inputs / input.len()`, `mem` holds
/// `input.len() · dilation · (ksize - 1)` values (at least). Dilation 1 is
/// the plain conv1d. A dilation above 1 requires `nb_inputs` to be a
/// multiple of `input.len()`.
pub fn compute_generic_conv1d_dilation(
    l: &LinearLayer,
    out: &mut [f32],
    mem: &mut [f32],
    input: &[f32],
    dilation: usize,
    act: Activation,
    arch: DnnArch,
) {
    let input_size = input.len();
    let nb_inputs = l.nb_inputs;
    assert!(nb_inputs <= MAX_CONV_INPUTS_ALL, "conv1d larger than MAX_CONV_INPUTS_ALL");
    assert!(input_size <= nb_inputs, "conv1d input longer than the layer's nb_inputs");
    assert!(dilation >= 1);
    let hist = nb_inputs - input_size;
    let k = Kernels::new(arch);
    let mut tmp = [0.0f32; MAX_CONV_INPUTS_ALL];
    let tmp = &mut tmp[..nb_inputs];
    if dilation == 1 {
        tmp[..hist].copy_from_slice(&mem[..hist]);
    } else {
        assert!(input_size > 0 && nb_inputs.is_multiple_of(input_size));
        let ksize = nb_inputs / input_size;
        assert!(mem.len() >= input_size * dilation * (ksize - 1));
        for i in 0..ksize - 1 {
            let src = i * input_size * dilation;
            tmp[i * input_size..(i + 1) * input_size].copy_from_slice(&mem[src..src + input_size]);
        }
    }
    tmp[hist..].copy_from_slice(input);
    k.linear(l, out, tmp);
    k.activation(&mut out[..l.nb_outputs], act);
    if dilation == 1 {
        mem[..hist].copy_from_slice(&tmp[input_size..]);
    } else {
        // With a kernel of one frame there is no memory. (libopus would
        // copy a negative count here.)
        let len = input_size * dilation * (nb_inputs / input_size - 1);
        if len > 0 {
            mem.copy_within(input_size..len, 0);
            mem[len - input_size..len].copy_from_slice(input);
        }
    }
}

/// Two-dimensional convolution of one time step (libopus `compute_conv2d`,
/// `dnn/nnet_arch.h:217-243`).
///
/// One time step of the input holds `in_channels · (height + kheight - 1)`
/// values (`time_stride`). `mem` holds the last `ktime - 1` time steps and
/// is updated. Output channel `i` is written to
/// `out[i·hstride .. i·hstride + height]`.
///
/// The checks are "at least", because PitchDNN passes larger buffers
/// (`dnn/pitchdnn.c:22-37`): `input.len() >= time_stride`,
/// `mem.len() >= (ktime - 1)·time_stride` and
/// `out.len() >= (out_channels - 1)·hstride + height`. The layer must have
/// its weights (`out_channels · in_channels · ktime · kheight` values).
///
/// The roundings (`nnet.md` section 3.5): for 3 × 3 kernels, the nine
/// products of one input channel are summed as
/// `fma(w8, x8, ... fma(w2, x2, fma(w0, x0, w1·x1)))` and added to the
/// output with a plain add; other kernels use one fused multiply-add per
/// weight. Then the bias is added and the activation runs on each output
/// channel.
#[allow(clippy::too_many_arguments)]
pub fn compute_conv2d(
    c: &Conv2dLayer,
    out: &mut [f32],
    mem: &mut [f32],
    input: &[f32],
    height: usize,
    hstride: usize,
    act: Activation,
    arch: DnnArch,
) {
    let (cin, cout, ktime, kheight) = (c.in_channels, c.out_channels, c.ktime, c.kheight);
    assert!(ktime >= 1 && kheight >= 1);
    let in_stride = height + kheight - 1;
    let time_stride = cin * in_stride;
    let hist = (ktime - 1) * time_stride;
    assert!(input.len() >= time_stride && mem.len() >= hist);
    if cout > 0 {
        assert!(hstride >= height && out.len() >= (cout - 1) * hstride + height);
    }
    let w = c.float_weights.expect("a conv2d layer needs its weights");
    assert!(w.len() >= cout * cin * ktime * kheight);
    let k = Kernels::new(arch);
    // libopus copies `[mem, input]` into one buffer; time step `t` of that
    // buffer is `mem[t·time_stride..]` for `t < ktime - 1` and `input` for
    // the last one. The memory is shifted after the products.
    let step = |t: usize| -> &[f32] {
        if t + 1 < ktime {
            &mem[t * time_stride..(t + 1) * time_stride]
        } else {
            &input[..time_stride]
        }
    };
    for i in 0..cout {
        let o = &mut out[i * hstride..i * hstride + height];
        o.fill(0.0);
        for m in 0..cin {
            let wm = &w[(i * cin + m) * ktime * kheight..(i * cin + m + 1) * ktime * kheight];
            if ktime == 3 && kheight == 3 {
                let x0 = &step(0)[m * in_stride..(m + 1) * in_stride];
                let x1 = &step(1)[m * in_stride..(m + 1) * in_stride];
                let x2 = &step(2)[m * in_stride..(m + 1) * in_stride];
                for (j, oj) in o.iter_mut().enumerate() {
                    let mut s = fma(wm[0], x0[j], wm[1] * x0[j + 1]);
                    s = fma(wm[2], x0[j + 2], s);
                    s = fma(wm[3], x1[j], s);
                    s = fma(wm[4], x1[j + 1], s);
                    s = fma(wm[5], x1[j + 2], s);
                    s = fma(wm[6], x2[j], s);
                    s = fma(wm[7], x2[j + 1], s);
                    s = fma(wm[8], x2[j + 2], s);
                    *oj += s;
                }
            } else {
                for t in 0..ktime {
                    let xt = &step(t)[m * in_stride..(m + 1) * in_stride];
                    for h in 0..kheight {
                        let wv = wm[t * kheight + h];
                        for (j, oj) in o.iter_mut().enumerate() {
                            *oj = fma(wv, xt[j + h], *oj);
                        }
                    }
                }
            }
        }
    }
    if let Some(bias) = c.bias {
        for i in 0..cout {
            let b = bias[i];
            for v in &mut out[i * hstride..i * hstride + height] {
                *v += b;
            }
        }
    }
    for i in 0..cout {
        k.activation(&mut out[i * hstride..i * hstride + height], act);
    }
    if hist > 0 {
        mem.copy_within(time_stride..hist, 0);
        mem[hist - time_stride..hist].copy_from_slice(&input[..time_stride]);
    }
}

#[cfg(test)]
mod tests {
    //! Every `DnnArch` of `available()` against `Portable`, bit for bit, on
    //! random and edge inputs. Two NaNs count as equal (x86 and ARM make
    //! different NaN bits).
    extern crate std;
    use super::*;
    use std::vec;
    use std::vec::Vec;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 16) as u32
        }
        fn below(&mut self, n: u32) -> u32 {
            self.next() % n
        }
        /// Uniform in (-a, a).
        fn f(&mut self, a: f32) -> f32 {
            a * (2.0 * (self.next() as f32 / 4_294_967_296.0) - 1.0)
        }
        /// int8 weights in [-127, 127].
        fn w8(&mut self) -> i8 {
            (self.below(255) as i32 - 127) as i8
        }
    }

    const EDGE: [f32; 34] = [
        0.0,
        -0.0,
        1.0,
        -1.0,
        0.5 / 127.0,
        1.5 / 127.0,
        2.5 / 127.0,
        -0.5 / 127.0,
        -2.5 / 127.0,
        1.003_937,
        1.007_874,
        1.5,
        -1.5,
        2.0,
        -2.0,
        3.0,
        88.0,
        -88.0,
        89.0,
        -89.0,
        1e10,
        -1e10,
        1e20,
        -1e20,
        f32::INFINITY,
        f32::NEG_INFINITY,
        1e-40,
        -1e-40,
        7.3e9,
        5.4e7,
        2.313e9,
        2.3132e9,
        4.626e9,
        4.6263e9,
    ];

    fn input(rng: &mut Rng, n: usize, trial: usize) -> Vec<f32> {
        (0..n)
            .map(|i| match trial {
                0..=2 => EDGE[(i * 7 + trial) % EDGE.len()],
                3..=19 => rng.f(1.0),
                20..=24 => rng.f(1.6),
                _ => rng.f(30.0),
            })
            .collect()
    }

    fn same(a: &[f32], b: &[f32]) -> bool {
        a.len() == b.len()
            && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan()))
    }

    /// The arrays of one random layer.
    struct Owned {
        bias: Vec<f32>,
        fw: Vec<f32>,
        w8: Vec<i8>,
        idx: Vec<i32>,
        scale: Vec<f32>,
        diag: Vec<f32>,
        kind: u8,
        nin: usize,
        nout: usize,
        with_bias: bool,
        with_diag: bool,
    }

    /// kind: 0 float, 1 int8, 2 int8 sparse, 3 float sparse.
    fn make(rng: &mut Rng, nin: usize, nout: usize, kind: u8, with_bias: bool, with_diag: bool) -> Owned {
        let bias = (0..nout).map(|_| rng.f(0.5)).collect();
        let scale = (0..nout).map(|_| (1.0 / 128.0 / 127.0) * (0.5 + rng.below(1000) as f32 / 1000.0)).collect();
        let diag = (0..nout).map(|_| rng.f(0.3)).collect();
        let mut idx = Vec::new();
        let mut blocks = 0;
        if kind >= 2 {
            for _ in (0..nout).step_by(8) {
                let at = idx.len();
                idx.push(0);
                let mut count = 0;
                for c in (0..nin.saturating_sub(3)).step_by(4) {
                    if rng.below(3) != 0 {
                        idx.push(c as i32);
                        count += 1;
                    }
                }
                idx[at] = count;
                blocks += count as usize;
            }
        }
        let nw = if kind >= 2 { 32 * blocks } else { nin * nout };
        let fw = (0..nw).map(|_| rng.f(0.3)).collect();
        let w8 = (0..nw).map(|_| rng.w8()).collect();
        Owned { bias, fw, w8, idx, scale, diag, kind, nin, nout, with_bias, with_diag }
    }

    impl Owned {
        fn layer(&self) -> LinearLayer<'_> {
            let mut l = LinearLayer { nb_inputs: self.nin, nb_outputs: self.nout, ..Default::default() };
            if self.with_bias {
                l.bias = Some(&self.bias);
            }
            if self.with_diag {
                l.diag = Some(&self.diag);
            }
            match self.kind {
                0 => l.float_weights = Some(&self.fw),
                1 => {
                    l.weights = Some(&self.w8);
                    l.scale = Some(&self.scale);
                }
                2 => {
                    l.weights = Some(&self.w8);
                    l.scale = Some(&self.scale);
                    l.weights_idx = Some(&self.idx);
                }
                _ => {
                    l.float_weights = Some(&self.fw);
                    l.weights_idx = Some(&self.idx);
                }
            }
            l
        }
    }

    const SHAPES: [(usize, usize, u8); 64] = [
        (88, 64, 1), (64, 64, 1), (288, 64, 1), (64, 192, 1), (57, 128, 0), (192, 20, 0),
        (128, 576, 1), (192, 576, 1), (224, 12, 0), (32, 64, 0), (80, 1, 0), (328, 192, 1),
        (688, 128, 1), (128, 40, 1), (192, 4, 0), (40, 64, 0), (544, 32, 1), (128, 56, 1),
        (544, 128, 2), (96, 64, 2), (480, 64, 2), (64, 96, 2), (26, 96, 0), (50, 128, 0),
        (576, 80, 2), (128, 320, 2), (301, 64, 0), (173, 96, 0), (768, 128, 1), (128, 512, 1),
        (128, 16, 1), (128, 1, 0), (160, 2, 0), (42, 80, 0), (160, 80, 0), (320, 160, 1),
        (342, 128, 0), (384, 128, 1), (256, 120, 1), (42, 120, 0), (240, 120, 0), (128, 3, 0),
        (128, 288, 1), (3, 1, 0), (4, 1, 0), (5, 1, 0), (7, 1, 0), (16, 1, 0), (17, 1, 0),
        (19, 1, 0), (20, 1, 0), (2, 8, 0), (32, 1, 0), (8, 48, 1), (1, 3, 0), (12, 12, 0),
        (64, 96, 3), (160, 24, 3), (2040, 8, 1), (8, 8, 1), (12, 40, 1), (20, 8, 1), (36, 21, 0),
        (9, 7, 0),
    ];

    /// Sparse int8 layers whose input count is not a multiple of 4 or 8,
    /// so the quantization runs its shorter loops.
    const ODD_SPARSE: [(usize, usize, u8); 3] = [(30, 16, 2), (13, 8, 2), (22, 8, 3)];

    #[test]
    fn linear_layers_match_portable() {
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        for &(nin, nout, kind) in SHAPES.iter().chain(ODD_SPARSE.iter()) {
            for trial in 0..30 {
                let with_diag = trial % 3 == 2 && nout == 3 * nin;
                let o = make(&mut rng, nin, nout, kind, trial % 4 != 3, with_diag);
                let l = o.layer();
                let x = input(&mut rng, nin + trial % 3, trial);
                let mut want = vec![0.0; nout];
                compute_linear(&l, &mut want, &x, DnnArch::Portable);
                for &a in arch::available() {
                    let mut got = vec![0.0; nout + 2];
                    compute_linear(&l, &mut got, &x, a);
                    assert!(same(&got[..nout], &want), "{a:?} {nin}x{nout} kind {kind} trial {trial}");
                    assert_eq!(&got[nout..], &[0.0, 0.0]);
                    let mut dense = vec![0.0; nout];
                    compute_generic_dense(&l, &mut dense, &x, Activation::Tanh, a);
                    let mut want_t = want.clone();
                    compute_activation_in_place(&mut want_t, Activation::Tanh, DnnArch::Portable);
                    assert!(same(&dense, &want_t), "dense {a:?} {nin}x{nout} kind {kind}");
                }
            }
        }
    }

    #[test]
    fn int8_quantization_of_every_float_class_matches() {
        // Row `i` of the layer has weight 1 at column `i` and scale 1, so
        // the output is the quantized input.
        let mut x = Vec::new();
        for sign in [0u32, 1] {
            for e in 0..256u32 {
                for f in [0u32, 1, 0x40_0000, 0x7f_ffff, 0x12_3456] {
                    x.push(f32::from_bits(sign << 31 | e << 23 | f));
                }
            }
        }
        for k in -260..=260 {
            for d in [-0.5f32, 0.5] {
                let v = (k as f32 + d) / 127.0;
                x.extend([v, f32::from_bits(v.to_bits() + 1), f32::from_bits(v.to_bits() - 1)]);
            }
        }
        x.truncate(x.len() / 8 * 8);
        let mut pieces: Vec<(usize, usize)> = (0..x.len()).step_by(2048).map(|s| (s, (x.len() - s).min(2048))).collect();
        pieces.extend([(8, 24), (0, 16), (16, 8)]);
        for (start, n) in pieces {
            let x = &x[start..start + n];
            let mut w = vec![0i8; n * n];
            for i in 0..n {
                // Block of rows i/8, column group i/4; byte 4·(i%8) + i%4.
                w[(i / 8 * (n / 4) + i / 4) * 32 + 4 * (i % 8) + i % 4] = 1;
            }
            let scale = vec![1.0f32; n];
            let l = LinearLayer { weights: Some(&w), scale: Some(&scale), nb_inputs: n, nb_outputs: n, ..Default::default() };
            for &a in arch::available() {
                let mut y = vec![0.0f32; n];
                compute_linear(&l, &mut y, &x[..n], a);
                for i in 0..n {
                    assert_eq!(y[i], vec_portable::quantize(x[i]) as f32, "{a:?} x = {:e} ({:08x})", x[i], x[i].to_bits());
                }
            }
        }
    }

    #[test]
    fn a_sparse_layer_with_every_block_equals_the_dense_layer() {
        let mut rng = Rng(7);
        for &(nin, nout) in &[(64usize, 32usize), (96, 64), (8, 8), (544, 128)] {
            let dense8: Vec<i8> = (0..nin * nout).map(|_| rng.w8()).collect();
            let scale: Vec<f32> = (0..nout).map(|_| rng.f(0.01)).collect();
            let mut idx = Vec::new();
            for _ in 0..nout / 8 {
                idx.push((nin / 4) as i32);
                idx.extend((0..nin as i32).step_by(4));
            }
            // Float weights: dense column-major, sparse 8 × 4 blocks with
            // float `8·c + k` for row `k` and column `c`.
            let densef: Vec<f32> = (0..nin * nout).map(|_| rng.f(0.3)).collect();
            let mut sparsef = Vec::new();
            for b in 0..nout / 8 {
                for g in 0..nin / 4 {
                    for c in 0..4 {
                        for k in 0..8 {
                            sparsef.push(densef[(4 * g + c) * nout + 8 * b + k]);
                        }
                    }
                }
            }
            let x = input(&mut rng, nin, 5);
            for &a in arch::available() {
                let d = LinearLayer { weights: Some(&dense8), scale: Some(&scale), nb_inputs: nin, nb_outputs: nout, ..Default::default() };
                let s = LinearLayer { weights_idx: Some(&idx), ..d };
                let (mut yd, mut ys) = (vec![0.0; nout], vec![0.0; nout]);
                compute_linear(&d, &mut yd, &x, a);
                compute_linear(&s, &mut ys, &x, a);
                assert!(same(&yd, &ys), "int8 {a:?} {nin}x{nout}");
                let d = LinearLayer { float_weights: Some(&densef), nb_inputs: nin, nb_outputs: nout, ..Default::default() };
                let s = LinearLayer { float_weights: Some(&sparsef), weights_idx: Some(&idx), ..d };
                compute_linear(&d, &mut yd, &x, a);
                compute_linear(&s, &mut ys, &x, a);
                assert!(same(&yd, &ys), "float {a:?} {nin}x{nout}");
            }
        }
    }

    #[test]
    fn activations_match_portable() {
        let mut rng = Rng(3);
        let acts = [
            Activation::Linear,
            Activation::Sigmoid,
            Activation::Tanh,
            Activation::Relu,
            Activation::Softmax,
            Activation::Swish,
            Activation::Exp,
        ];
        for act in acts {
            for n in (1..=41).chain([576, 130]) {
                for trial in 0..40 {
                    let x = input(&mut rng, n, trial % 30);
                    let mut want = vec![0.0; n];
                    compute_activation(&mut want, &x, act, DnnArch::Portable);
                    for &a in arch::available() {
                        let mut got = vec![0.0; n];
                        compute_activation(&mut got, &x, act, a);
                        assert!(same(&got, &want), "{act:?} {a:?} n {n} trial {trial}");
                        let mut inplace = x.clone();
                        compute_activation_in_place(&mut inplace, act, a);
                        assert!(same(&inplace, &want), "in place {act:?} {a:?} n {n}");
                    }
                }
            }
        }
    }

    #[test]
    fn activation_tails_use_the_division_formulas() {
        // One value: always the tail formula.
        let x = [0.3f32];
        let mut y = [0.0];
        compute_activation(&mut y, &x, Activation::Tanh, DnnArch::Portable);
        assert_eq!(y[0].to_bits(), vec_portable::tanh_tail(0.3).to_bits());
        compute_activation(&mut y, &x, Activation::Sigmoid, DnnArch::Portable);
        assert_eq!(y[0].to_bits(), vec_portable::sigmoid_tail(0.3).to_bits());
        // Five values: four with the group formula, one with the tail.
        let x = [0.3f32; 5];
        let mut y = [0.0; 5];
        for &a in arch::available() {
            compute_activation(&mut y, &x, Activation::Tanh, a);
            assert_eq!(y[3].to_bits(), vec_portable::tanh_approx(0.3).to_bits());
            assert_eq!(y[4].to_bits(), vec_portable::tanh_tail(0.3).to_bits());
        }
        assert_ne!(vec_portable::tanh_tail(0.3).to_bits(), vec_portable::tanh_approx(0.3).to_bits());
        // RELU keeps -0.0 and NaN.
        let x = [-0.0f32, f32::NAN, -1.0, 2.0];
        let mut y = [0.0; 4];
        compute_activation(&mut y, &x, Activation::Relu, DnnArch::Portable);
        assert_eq!(y[0].to_bits(), (-0.0f32).to_bits());
        assert!(y[1].is_nan());
        assert_eq!(&y[2..], &[0.0, 2.0]);
    }

    #[test]
    fn gru_glu_and_conv1d_match_portable() {
        let mut rng = Rng(11);
        let grus = [(128usize, 192usize, 1u8), (192, 192, 1), (64, 64, 1), (272, 160, 1), (64, 32, 2), (160, 128, 0), (57, 4, 0), (8, 16, 1)];
        for &(nin, n, kind) in &grus {
            let li = make(&mut rng, nin, 3 * n, kind, true, false);
            let lr = make(&mut rng, n, 3 * n, if kind == 2 { 1 } else { kind }, true, kind == 0);
            let mut want = vec![0.0; n];
            let mut states: Vec<Vec<f32>> = arch::available().iter().map(|_| vec![0.0; n]).collect();
            for step in 0..30 {
                let x = input(&mut rng, nin, 3 + step % 25);
                compute_generic_gru(&li.layer(), &lr.layer(), &mut want, &x, DnnArch::Portable);
                for (s, &a) in states.iter_mut().zip(arch::available()) {
                    compute_generic_gru(&li.layer(), &lr.layer(), s, &x, a);
                    assert!(same(s, &want), "GRU {a:?} {nin}x{n} step {step}");
                }
            }
        }
        for &(n, kind) in &[(192usize, 1u8), (160, 1), (128, 1), (64, 2), (24, 0), (20, 0)] {
            let l = make(&mut rng, n, n, kind, true, false);
            let x = input(&mut rng, n, 7);
            let mut want = vec![0.0; n];
            compute_glu(&l.layer(), &mut want, &x, DnnArch::Portable);
            for &a in arch::available() {
                let mut got = vec![0.0; n];
                compute_glu(&l.layer(), &mut got, &x, a);
                assert!(same(&got, &want), "GLU {a:?} {n}");
                let mut inplace = x.clone();
                compute_glu_in_place(&l.layer(), &mut inplace, a);
                assert!(same(&inplace, &want), "GLU in place {a:?} {n}");
            }
        }
        // conv1d: (input size, kernel size, outputs, kind, dilation).
        let convs = [(64usize, 3usize, 32usize, 1u8, 1usize), (64, 2, 64, 1, 2), (32, 4, 64, 2, 2), (173, 1, 96, 0, 1), (16, 3, 8, 0, 3), (40, 2, 16, 3, 1)];
        for &(isz, ksize, nout, kind, dil) in &convs {
            let l = make(&mut rng, isz * ksize, nout, kind, true, false);
            let memlen = isz * dil * (ksize - 1);
            let mut want_mem = vec![0.0; memlen];
            let mut mems: Vec<Vec<f32>> = arch::available().iter().map(|_| vec![0.0; memlen]).collect();
            for step in 0..20 {
                let x = input(&mut rng, isz, 3 + step);
                let mut want = vec![0.0; nout];
                compute_generic_conv1d_dilation(&l.layer(), &mut want, &mut want_mem, &x, dil, Activation::Tanh, DnnArch::Portable);
                for (m, &a) in mems.iter_mut().zip(arch::available()) {
                    let mut got = vec![0.0; nout];
                    if dil == 1 {
                        compute_generic_conv1d(&l.layer(), &mut got, m, &x, Activation::Tanh, a);
                    } else {
                        compute_generic_conv1d_dilation(&l.layer(), &mut got, m, &x, dil, Activation::Tanh, a);
                    }
                    assert!(same(&got, &want) && same(m, &want_mem), "conv1d {a:?} {isz}x{ksize} dil {dil} step {step}");
                }
            }
        }
    }

    #[test]
    fn conv1d_without_memory_is_a_dense_layer() {
        let mut rng = Rng(5);
        let l = make(&mut rng, 173, 96, 0, true, false);
        let x = input(&mut rng, 173, 4);
        let (mut a, mut b, mut c) = (vec![0.0; 96], vec![0.0; 96], vec![0.0; 96]);
        compute_generic_conv1d(&l.layer(), &mut a, &mut [], &x, Activation::Tanh, arch::detect());
        compute_generic_dense(&l.layer(), &mut b, &x, Activation::Tanh, arch::detect());
        compute_generic_conv1d_dilation(&l.layer(), &mut c, &mut [], &x, 2, Activation::Tanh, arch::detect());
        assert!(same(&a, &b) && same(&c, &b));
    }

    #[test]
    fn conv2d_matches_portable() {
        let mut rng = Rng(13);
        // (in channels, out channels, ktime, kheight, height, hstride)
        let shapes = [(1usize, 4usize, 3usize, 3usize, 224usize, 226usize), (4, 1, 3, 3, 224, 224), (2, 3, 2, 5, 17, 20), (3, 2, 1, 1, 9, 9)];
        for &(cin, cout, kt, kh, height, hstride) in &shapes {
            let w: Vec<f32> = (0..cout * cin * kt * kh).map(|_| rng.f(0.5)).collect();
            let bias: Vec<f32> = (0..cout).map(|_| rng.f(0.5)).collect();
            let c = Conv2dLayer { bias: Some(&bias), float_weights: Some(&w), in_channels: cin, out_channels: cout, ktime: kt, kheight: kh };
            let ts = cin * (height + kh - 1);
            let mut want_mem = vec![0.0; (kt - 1) * ts + 3];
            let mut mems: Vec<Vec<f32>> = arch::available().iter().map(|_| want_mem.clone()).collect();
            for step in 0..8 {
                let x = input(&mut rng, ts + 5, 3 + step);
                let olen = (cout - 1) * hstride + height + 2;
                let mut want = vec![0.0; olen];
                compute_conv2d(&c, &mut want, &mut want_mem, &x, height, hstride, Activation::Tanh, DnnArch::Portable);
                for (m, &a) in mems.iter_mut().zip(arch::available()) {
                    let mut got = vec![0.0; olen];
                    compute_conv2d(&c, &mut got, m, &x, height, hstride, Activation::Tanh, a);
                    assert!(same(&got, &want) && same(m, &want_mem), "conv2d {a:?} step {step}");
                }
            }
        }
    }
}
