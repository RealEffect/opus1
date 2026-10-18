//! The kernels in plain Rust. They are the specification of every other
//! kernel variant: `vec_neon` and `vec_x86` must give the same bits.
//!
//! The numbers are those of libopus 1.6.1 on 64-bit ARM (`dnn/vec_neon.h`),
//! built without loop vectorization:
//!
//! - Every `fma` of `nnet.md` section 3 is a fused multiply-add (`fma`
//!   below). Rust never fuses `a * b + c` by itself.
//! - FRECPE (the ARM reciprocal estimate) is computed in software.
//! - FMIN and FMAX return NaN when an input is NaN, unlike `f32::min` and
//!   `f32::max`.
//! - Float products: each output is one chain of fused multiply-adds over
//!   the inputs in order, starting from +0.
//! - int8 products: inputs quantized as `round(127·x)` with ties to even,
//!   then truncated to 8 bits; exact integer sums; `scale[i] · (sum as f32)`.
//!
//! The constants are the C `float` literals of `vec_neon.h`, written with
//! the same digits, so they give the same `f32` values.

// The constants keep the digits of the C source (one of them is close to
// LOG2_E).
#![allow(clippy::excessive_precision, clippy::approx_constant)]

#[cfg(all(not(feature = "std"), not(target_arch = "aarch64")))]
use crate::compat::Math;

use super::MAX_INPUTS;

/// `a · b + c` with one rounding (a fused multiply-add) on every target.
///
/// On 64-bit ARM this is one `fmadd` instruction. Elsewhere it is
/// `f32::mul_add` (with `std`) or `libm::fmaf` (without), which give the
/// same result; they are slow on x86 processors without FMA.
#[inline(always)]
pub fn fma(a: f32, b: f32, c: f32) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        use core::arch::aarch64::{vdup_n_f32, vfma_f32, vget_lane_f32};
        // SAFETY: NEON is part of every 64-bit ARM processor. The compiler
        // turns this into one scalar `fmadd`.
        unsafe { vget_lane_f32::<0>(vfma_f32(vdup_n_f32(c), vdup_n_f32(a), vdup_n_f32(b))) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        a.mul_add(b, c)
    }
}

/// The ARM instruction FRECPE (`FPRecipEstimate`) on the bits of an `f32`:
/// an 8-bit estimate of `1/x`. Assumes FPCR.FZ = 0, FPCR.DN = 0 and
/// FPCR.AH = 0, the state of user programs on macOS, iOS, Linux and
/// Android.
///
/// - NaN: the same NaN, quieted.
/// - ±infinity: ±0.
/// - ±0 and |x| < 2^-128: ±infinity.
/// - otherwise the 8-bit estimate of the Arm Architecture Reference Manual,
///   which can be a subnormal number when |x| ≥ 2^126.
///
/// It matched the instruction on all 2^32 inputs on the Apple M4, the
/// Cortex-A76 and the Cortex-A55 (`nnet.md` section 6.4).
pub fn frecpe_bits(x: u32) -> u32 {
    let sign = x & 0x8000_0000;
    let exp = ((x >> 23) & 0xff) as i32;
    let frac = x & 0x007f_ffff;
    if exp == 0xff {
        // NaN: quieted. Infinity: zero of the same sign.
        return if frac != 0 { x | 0x0040_0000 } else { sign };
    }
    if exp == 0 && frac < (1 << 21) {
        // Zero, and values so small that 1/x overflows: infinity.
        return sign | 0x7f80_0000;
    }
    // The 52-bit fraction of the pseudocode.
    let mut f = (frac as u64) << 29;
    let mut e = exp;
    if e == 0 {
        // Subnormal input: normalize.
        if (f >> 51) & 1 == 0 {
            e = -1;
            f = (f & ((1 << 50) - 1)) << 2;
        } else {
            f = (f & ((1 << 51) - 1)) << 1;
        }
    }
    // The top 8 fraction bits with the leading one: 256..511.
    let a = 0x100 | ((f >> 44) & 0xff) as u32;
    let b = (1u32 << 19) / (2 * a + 1);
    // The 8-bit estimate with its leading one: 256..511. This is
    // `(b + 1) / 2` of the pseudocode.
    let est = b.div_ceil(2);
    let mut rexp = 253 - e;
    let mut rf = ((est & 0xff) as u64) << 44;
    if rexp == 0 {
        rf = (1 << 51) | (rf >> 1);
    } else if rexp == -1 {
        rf = (1 << 50) | (rf >> 2);
        rexp = 0;
    }
    sign | (((rexp & 0xff) as u32) << 23) | (rf >> 29) as u32
}

/// `frecpe_bits` on an `f32`.
#[inline]
pub fn frecpe(x: f32) -> f32 {
    f32::from_bits(frecpe_bits(x.to_bits()))
}

/// The NaN that the ARM instructions FMIN and FMAX return when an input is
/// NaN (FPCR.DN = 0): the first signaling NaN, else the first quiet NaN,
/// quieted.
#[inline]
fn arm_nan(a: f32, b: f32) -> f32 {
    const QUIET: u32 = 0x0040_0000;
    let (ab, bb) = (a.to_bits(), b.to_bits());
    let a_snan = a.is_nan() && ab & QUIET == 0;
    let b_snan = b.is_nan() && bb & QUIET == 0;
    let pick = if a_snan {
        ab
    } else if b_snan {
        bb
    } else if a.is_nan() {
        ab
    } else {
        bb
    };
    f32::from_bits(pick | QUIET)
}

/// The ARM instruction FMIN: the smaller value, or NaN when `a` or `b` is
/// NaN. (-0.0 is smaller than +0.0.)
#[inline]
pub fn fmin(a: f32, b: f32) -> f32 {
    if a.is_nan() || b.is_nan() {
        arm_nan(a, b)
    } else if a == b {
        // Equal values have equal bits, except +0.0 and -0.0.
        f32::from_bits(a.to_bits() | b.to_bits())
    } else if a < b {
        a
    } else {
        b
    }
}

/// The ARM instruction FMAX: the larger value, or NaN when `a` or `b` is
/// NaN. (+0.0 is larger than -0.0.)
#[inline]
pub fn fmax(a: f32, b: f32) -> f32 {
    if a.is_nan() || b.is_nan() {
        arm_nan(a, b)
    } else if a == b {
        f32::from_bits(a.to_bits() & b.to_bits())
    } else if a > b {
        a
    } else {
        b
    }
}

/// The ARM instruction FCVTNS: the nearest `i32`, with ties to even,
/// saturated; NaN gives 0.
#[inline]
fn round_to_i32(p: f32) -> i32 {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is part of every 64-bit ARM processor.
        unsafe { core::arch::aarch64::vcvtns_s32_f32(p) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        // `as` saturates and turns NaN into 0, like FCVTNS.
        p.round_ties_even() as i32
    }
}

/// The int8 input of an int8 product (`dnn/vec_neon.h:385-392`): `127·x`
/// rounded to `f32`, then to the nearest integer with ties to even,
/// saturated to `i32` (NaN gives 0), then truncated to 16 and to 8 bits. So
/// `|127·x| > 127.5` wraps around: `x = 1.5` gives -66.
#[inline]
pub fn quantize(x: f32) -> i8 {
    round_to_i32(127.0 * x) as i16 as i8
}

/// One lane of `exp4_approx` (`dnn/vec_neon.h:59-84`), which is libopus
/// `lpcnet_exp`.
#[inline]
pub fn lpcnet_exp(x: f32) -> f32 {
    let x = fmax(fmin(x, 88.0), -88.0);
    // exp(x) = 2^(x / ln 2); 127 is the exponent bias, added here.
    let x = fma(x, 1.44269504, 127.0);
    // FCVTZS: truncation toward zero, saturated; NaN gives 0.
    let i = x as i32;
    let x = x - i as f32;
    let y = fma(x, fma(x, fma(0.078024523, x, 0.22606716), 0.69583354), 0.99992522);
    // 2^(i - 127); i = 0 gives 0.0.
    y * f32::from_bits((i as u32) << 23)
}

/// One lane of `tanh4_approx` (`dnn/vec_neon.h:86-104`), which is libopus
/// `tanh_approx`. This is the formula for the groups of 4 values, not the
/// tail formula of `vec_tanh`.
#[inline]
pub fn tanh_approx(x: f32) -> f32 {
    let x2 = x * x;
    let num = fma(x2, fma(0.60863042, x2, 96.39235687), 952.52801514);
    let den = fma(x2, fma(11.88600922, x2, 413.36801147), 952.72399902);
    let num = num * x;
    let num = num * frecpe(den);
    fmax(-1.0, fmin(1.0, num))
}

/// One lane of `sigmoid4_approx` (`dnn/vec_neon.h:106-125`), which is
/// libopus `sigmoid_approx`. This is the formula for the groups of 4
/// values, not the tail formula of `vec_sigmoid`.
#[inline]
pub fn sigmoid_approx(x: f32) -> f32 {
    let x2 = x * x;
    let num = fma(x2, fma(0.00950985, x2, 6.02452230), 238.13200378);
    let den = fma(x2, fma(0.74287558, x2, 103.34200287), 952.72399902);
    let num = num * x;
    let num = fma(num, frecpe(den), 0.5);
    fmax(0.0, fmin(1.0, num))
}

/// The tail formula of `vec_tanh` (`dnn/vec_neon.h:181-186`), for the last
/// `len % 4` values: `(e - 1) / (e + 1)` with `e = lpcnet_exp(2·x)`.
#[inline]
pub fn tanh_tail(x: f32) -> f32 {
    let ex2 = lpcnet_exp(2.0 * x);
    (ex2 - 1.0) / (ex2 + 1.0)
}

/// The tail formula of `vec_sigmoid` (`dnn/vec_neon.h:199-204`), for the
/// last `len % 4` values: `e / (e + 1)` with `e = lpcnet_exp(x)`.
#[inline]
pub fn sigmoid_tail(x: f32) -> f32 {
    let ex = lpcnet_exp(x);
    ex / (ex + 1.0)
}

/// `buf[i] = exp(buf[i])` with `exp4_approx` for every value (libopus
/// `softmax` in `dnn/vec_neon.h:157-169`, which computes only the
/// exponentials).
pub fn vec_exp(buf: &mut [f32]) {
    for v in buf.iter_mut() {
        *v = lpcnet_exp(*v);
    }
}

/// `buf[i] = tanh(buf[i])` (libopus `vec_tanh`, `dnn/vec_neon.h:171-187`):
/// `tanh4_approx` on the first `4·floor(len/4)` values, then
/// `(e - 1) / (e + 1)` with `e = lpcnet_exp(2·x)` on the last `len % 4`.
pub fn vec_tanh(buf: &mut [f32]) {
    let groups = buf.len() & !3;
    let (head, tail) = buf.split_at_mut(groups);
    for v in head.iter_mut() {
        *v = tanh_approx(*v);
    }
    for v in tail.iter_mut() {
        *v = tanh_tail(*v);
    }
}

/// `buf[i] = sigmoid(buf[i])` (libopus `vec_sigmoid`,
/// `dnn/vec_neon.h:189-205`): `sigmoid4_approx` on the first
/// `4·floor(len/4)` values, then `e / (e + 1)` with `e = lpcnet_exp(x)` on
/// the last `len % 4`.
pub fn vec_sigmoid(buf: &mut [f32]) {
    let groups = buf.len() & !3;
    let (head, tail) = buf.split_at_mut(groups);
    for v in head.iter_mut() {
        *v = sigmoid_approx(*v);
    }
    for v in tail.iter_mut() {
        *v = sigmoid_tail(*v);
    }
}

/// Dense float product (libopus `sgemv`, `dnn/vec_neon.h:208-299`):
/// `out[i] = sum over j of weights[j·col_stride + i] · x[j]` for `i < rows`,
/// `j < cols`, as one fused multiply-add chain per output, in the order of
/// `j`, starting from +0.
///
/// Requires `out.len() >= rows`, `x.len() >= cols` and, when `cols > 0`,
/// `weights.len() >= (cols - 1)·col_stride + rows`.
pub fn sgemv(
    out: &mut [f32],
    weights: &[f32],
    rows: usize,
    cols: usize,
    col_stride: usize,
    x: &[f32],
) {
    let out = &mut out[..rows];
    out.fill(0.0);
    // The column loop is outside, so that the rows of one column are read
    // in a row; each output still runs its own chain in the order of `j`.
    for (j, &xj) in x[..cols].iter().enumerate() {
        let w = &weights[j * col_stride..j * col_stride + rows];
        for (o, &wij) in out.iter_mut().zip(w) {
            *o = fma(wij, xj, *o);
        }
    }
}

/// Sparse float product (libopus `sparse_sgemv8x4`,
/// `dnn/vec_neon.h:302-359`). `rows` is a multiple of 8.
///
/// For each block of 8 rows, the index table holds a count `n` and then `n`
/// column positions. The weights hold 32 floats for each listed block, in
/// order: weight `8·c + k` multiplies `x[pos + c]` into row `k` of the
/// block. Each output is one fused multiply-add chain in the order of the
/// table, starting from +0.
///
/// Panics when the table or the weights are too short or a position is out
/// of range of `x`.
pub fn sparse_sgemv8x4(out: &mut [f32], weights: &[f32], idx: &[i32], rows: usize, x: &[f32]) {
    let mut p = 0;
    let mut w = 0;
    for block in out[..rows].as_chunks_mut::<8>().0 {
        let mut y = [0.0f32; 8];
        let count = idx[p] as usize;
        p += 1;
        for _ in 0..count {
            let pos = idx[p] as usize;
            p += 1;
            let wb = &weights[w..w + 32];
            let xb = &x[pos..pos + 4];
            for c in 0..4 {
                for k in 0..8 {
                    y[k] = fma(wb[8 * c + k], xb[c], y[k]);
                }
            }
            w += 32;
        }
        block.copy_from_slice(&y);
    }
}

/// Quantizes `x` into `xq` with `quantize`.
#[inline]
pub fn quantize_slice(xq: &mut [i8], x: &[f32]) {
    for (q, &v) in xq.iter_mut().zip(x) {
        *q = quantize(v);
    }
}

/// The sums of one 8 × 4 int8 block: `acc[k] += Σ_c w[4·k + c] · x[c]`.
#[inline(always)]
fn block_dot(acc: &mut [i32; 8], w: &[i8], x: &[i8]) {
    for k in 0..8 {
        let mut s = 0;
        for c in 0..4 {
            s += w[4 * k + c] as i32 * x[c] as i32;
        }
        acc[k] += s;
    }
}

/// Dense int8 product (libopus `cgemv8x4`, `dnn/vec_neon.h:379-432`).
/// `rows` is a multiple of 8 and `cols` a multiple of 4, at most
/// `MAX_INPUTS`; the input is never read past `cols`.
///
/// The weights are 8 × 4 blocks of 32 bytes: the blocks of rows 0-7 for
/// columns 0, 4, 8 and so on, then those of rows 8-15. Byte `4·k + c` of a
/// block is the weight of row `k` and column `c` of the block.
///
/// `out[i] = scale[i] · (Σ_j w_ij · q(x_j) as f32)`, where `q` is
/// `quantize`. The sum is exact.
pub fn cgemv8x4(
    out: &mut [f32],
    weights: &[i8],
    scale: &[f32],
    rows: usize,
    cols: usize,
    x: &[f32],
) {
    let mut xq = [0i8; MAX_INPUTS];
    let xq = &mut xq[..cols];
    quantize_slice(xq, &x[..cols]);
    let groups = cols / 4;
    for (b, block) in out[..rows].as_chunks_mut::<8>().0.iter_mut().enumerate() {
        let mut acc = [0i32; 8];
        let wrow = &weights[b * groups * 32..(b + 1) * groups * 32];
        for (g, wb) in wrow.as_chunks::<32>().0.iter().enumerate() {
            block_dot(&mut acc, wb, &xq[4 * g..4 * g + 4]);
        }
        for k in 0..8 {
            block[k] = scale[8 * b + k] * acc[k] as f32;
        }
    }
}

/// Sparse int8 product (libopus `sparse_cgemv8x4`,
/// `dnn/vec_neon.h:434-470`). `rows` is a multiple of 8 and `cols` is at
/// most `MAX_INPUTS`.
///
/// The index table has the layout of `sparse_sgemv8x4`. The weights hold
/// one 8 × 4 int8 block of `cgemv8x4`'s layout for each listed position.
///
/// Panics when the table or the weights are too short or a position is out
/// of range.
pub fn sparse_cgemv8x4(
    out: &mut [f32],
    weights: &[i8],
    idx: &[i32],
    scale: &[f32],
    rows: usize,
    cols: usize,
    x: &[f32],
) {
    let mut xq = [0i8; MAX_INPUTS];
    let xq = &mut xq[..cols];
    quantize_slice(xq, &x[..cols]);
    let mut p = 0;
    let mut w = 0;
    for (b, block) in out[..rows].as_chunks_mut::<8>().0.iter_mut().enumerate() {
        let mut acc = [0i32; 8];
        let count = idx[p] as usize;
        p += 1;
        for _ in 0..count {
            let pos = idx[p] as usize;
            p += 1;
            block_dot(&mut acc, &weights[w..w + 32], &xq[pos..pos + 4]);
            w += 32;
        }
        for k in 0..8 {
            block[k] = scale[8 * b + k] * acc[k] as f32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantize_rounds_ties_to_even_and_wraps() {
        // 127·x exactly k + 0.5 rounds to even.
        assert_eq!(quantize(0.5 / 127.0), 0);
        assert_eq!(quantize(1.5 / 127.0), 2);
        assert_eq!(quantize(2.5 / 127.0), 2);
        assert_eq!(quantize(-2.5 / 127.0), -2);
        assert_eq!(quantize(1.0), 127);
        assert_eq!(quantize(-1.0), -127);
        // round(127 · 1.5) = 190 (190.5 rounds to even), which wraps to -66.
        assert_eq!(quantize(1.5), -66);
        assert_eq!(quantize(f32::NAN), 0);
        assert_eq!(quantize(f32::INFINITY), -1);
        assert_eq!(quantize(f32::NEG_INFINITY), 0);
    }

    #[test]
    fn fmin_fmax_follow_arm() {
        assert!(fmin(f32::NAN, 1.0).is_nan());
        assert!(fmin(1.0, f32::NAN).is_nan());
        assert!(fmax(f32::NAN, 1.0).is_nan());
        assert_eq!(fmin(-0.0, 0.0).to_bits(), (-0.0f32).to_bits());
        assert_eq!(fmin(0.0, -0.0).to_bits(), (-0.0f32).to_bits());
        assert_eq!(fmax(-0.0, 0.0).to_bits(), 0);
        assert_eq!(fmax(0.0, -0.0).to_bits(), 0);
        assert_eq!(fmin(2.0, 3.0), 2.0);
        assert_eq!(fmax(2.0, 3.0), 3.0);
    }

    #[test]
    fn frecpe_special_values() {
        assert_eq!(frecpe(1.0), 0.998_046_9);
        assert_eq!(frecpe(f32::INFINITY).to_bits(), 0);
        assert_eq!(frecpe(f32::NEG_INFINITY).to_bits(), 0x8000_0000);
        assert_eq!(frecpe(0.0), f32::INFINITY);
        assert_eq!(frecpe(-0.0), f32::NEG_INFINITY);
        assert!(frecpe(f32::NAN).is_nan());
    }

    #[test]
    fn activations_saturate_and_pass_nan() {
        assert_eq!(tanh_approx(100.0), 1.0);
        assert_eq!(tanh_approx(-100.0), -1.0);
        assert_eq!(sigmoid_approx(100.0), 1.0);
        assert_eq!(sigmoid_approx(-100.0), 0.0);
        // libopus gives NaN for |x| above about 2.31e9 (TANH) and 4.63e9
        // (SIGMOID): the denominator overflows (nnet.md section 3.4).
        assert!(tanh_approx(7.4e9).is_nan());
        assert_eq!(tanh_approx(2.3e9), 1.0);
        assert!(sigmoid_approx(4.7e9).is_nan());
        assert_eq!(sigmoid_approx(4.6e9), 1.0);
        assert!(lpcnet_exp(f32::NAN).is_nan());
        assert_eq!(lpcnet_exp(-1000.0), lpcnet_exp(-88.0));
    }
}
