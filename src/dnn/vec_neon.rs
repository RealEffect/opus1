//! The kernels for 64-bit ARM: NEON, and NEON with the DotProd extension
//! (libopus `dnn/vec_neon.h`, `dnn/arm/nnet_dotprod.c`).
//!
//! Each kernel gives the bits of the function of the same name in
//! `vec_portable`. `vfmaq_f32` stands wherever libopus writes `vmlaq_f32`:
//! libopus redefines `vmlaq_f32` as the fused `vfmaq_f32`, while the Rust
//! `vmlaq_f32` rounds the product first.
//!
//! How they are faster than libopus without changing a bit (`nnet.md`
//! section 10):
//!
//! - Float products: vector lanes stand for output rows, and each lane runs
//!   its own chain of fused multiply-adds in the order of the inputs, as
//!   every libopus variant does. A pass takes up to 32 rows (8 vectors), so
//!   8 independent chains hide the latency of `fmla`. libopus computes an
//!   output count that is not a multiple of 8 one row at a time.
//! - int8 products: integer sums do not depend on their order. Four column
//!   groups share one 16-byte load of the inputs (`sdot` by element) and
//!   run on 8 accumulators. The weights are read as one stream, as in
//!   libopus.
//! - Activations: 4 vectors per loop step.
//!
//! The kernels read the slices through raw pointers. The callers in
//! `nnet` check the lengths and the index tables first.

use core::arch::aarch64::*;
use core::ptr;

use super::MAX_INPUTS;
use super::vec_portable;

// ---------------------------------------------------------------------------
// Float products
// ---------------------------------------------------------------------------

/// One pass of `sgemv` over `V` vectors of 4 rows, starting at the rows
/// that `out` and `w` point to.
///
/// # Safety
///
/// NEON. `out` has `4·V` values; `w` has `4·V` values at each multiple of
/// `stride` below `cols·stride`; `x` has `cols` values.
#[inline(always)]
unsafe fn sgemv_pass<const V: usize>(
    out: *mut f32,
    w: *const f32,
    cols: usize,
    stride: usize,
    x: *const f32,
) {
    // SAFETY: the caller guarantees the lengths; NEON is enabled in the
    // calling function.
    unsafe {
        let mut acc = [vdupq_n_f32(0.0); V];
        let mut wj = w;
        for j in 0..cols {
            let xj = vld1q_dup_f32(x.add(j));
            for (v, a) in acc.iter_mut().enumerate() {
                *a = vfmaq_f32(*a, vld1q_f32(wj.add(4 * v)), xj);
            }
            // After the last column this points past the weights; it is
            // not read then. `wrapping_add` makes that pointer allowed.
            wj = wj.wrapping_add(stride);
        }
        for (v, a) in acc.iter().enumerate() {
            vst1q_f32(out.add(4 * v), *a);
        }
    }
}

/// NEON `sgemv`. Same contract as `vec_portable::sgemv`.
///
/// # Safety
///
/// The processor must support NEON. The slices must have the lengths that
/// `vec_portable::sgemv` requires, and `col_stride >= rows`.
#[target_feature(enable = "neon")]
pub unsafe fn sgemv(
    out: &mut [f32],
    weights: &[f32],
    rows: usize,
    cols: usize,
    col_stride: usize,
    x: &[f32],
) {
    debug_assert!(out.len() >= rows && x.len() >= cols);
    let o = out.as_mut_ptr();
    let w = weights.as_ptr();
    let xp = x.as_ptr();
    let mut i = 0;
    // SAFETY: the caller guarantees the lengths: each pass reads `w` at rows
    // below `rows` of the columns below `cols` and writes `out` below
    // `rows`.
    unsafe {
        while rows - i >= 32 {
            sgemv_pass::<8>(o.add(i), w.add(i), cols, col_stride, xp);
            i += 32;
        }
        match (rows - i) / 4 {
            7 => sgemv_pass::<7>(o.add(i), w.add(i), cols, col_stride, xp),
            6 => sgemv_pass::<6>(o.add(i), w.add(i), cols, col_stride, xp),
            5 => sgemv_pass::<5>(o.add(i), w.add(i), cols, col_stride, xp),
            4 => sgemv_pass::<4>(o.add(i), w.add(i), cols, col_stride, xp),
            3 => sgemv_pass::<3>(o.add(i), w.add(i), cols, col_stride, xp),
            2 => sgemv_pass::<2>(o.add(i), w.add(i), cols, col_stride, xp),
            1 => sgemv_pass::<1>(o.add(i), w.add(i), cols, col_stride, xp),
            _ => {}
        }
        i += (rows - i) & !3;
        // The last 1 to 3 rows: one scalar chain each, side by side.
        match rows - i {
            3 => sgemv_rows::<3>(o.add(i), w.add(i), cols, col_stride, xp),
            2 => sgemv_rows::<2>(o.add(i), w.add(i), cols, col_stride, xp),
            1 => sgemv_rows::<1>(o.add(i), w.add(i), cols, col_stride, xp),
            _ => {}
        }
    }
}

/// `R` rows of `sgemv` (1 to 3) as scalar chains, side by side. The chain
/// of a row is fixed, so a layer with one output runs at the latency of
/// `fmadd`, as in libopus.
///
/// # Safety
///
/// NEON. `out` has `R` values; `w` has `R` values at each multiple of
/// `stride` below `cols·stride`; `x` has `cols` values.
#[inline(always)]
unsafe fn sgemv_rows<const R: usize>(
    out: *mut f32,
    w: *const f32,
    cols: usize,
    stride: usize,
    x: *const f32,
) {
    // SAFETY: the caller guarantees the lengths.
    unsafe {
        let mut acc = [0.0f32; R];
        let mut wj = w;
        for j in 0..cols {
            let xj = *x.add(j);
            for (k, a) in acc.iter_mut().enumerate() {
                *a = vec_portable::fma(*wj.add(k), xj, *a);
            }
            // After the last column this points past the weights; it is
            // not read then. `wrapping_add` makes that pointer allowed.
            wj = wj.wrapping_add(stride);
        }
        ptr::copy_nonoverlapping(acc.as_ptr(), out, R);
    }
}

/// NEON `sparse_sgemv8x4`. Same contract as `vec_portable::sparse_sgemv8x4`.
///
/// # Safety
///
/// The processor must support NEON. The index table must be valid for
/// `rows` and the length of `x`, and `weights` must hold 32 floats for each
/// listed block, as `nnet` checks.
#[target_feature(enable = "neon")]
pub unsafe fn sparse_sgemv8x4(out: &mut [f32], weights: &[f32], idx: &[i32], rows: usize, x: &[f32]) {
    let mut ip = idx.as_ptr();
    let mut w = weights.as_ptr();
    let xp = x.as_ptr();
    let o = out.as_mut_ptr();
    // SAFETY: the caller guarantees that the table lists positions with
    // `pos + 4 <= x.len()`, that the weights cover the listed blocks and
    // that `out` holds `rows` values.
    unsafe {
        for i in (0..rows).step_by(8) {
            let mut y0 = vdupq_n_f32(0.0);
            let mut y1 = vdupq_n_f32(0.0);
            let count = *ip as usize;
            ip = ip.add(1);
            for _ in 0..count {
                let pos = *ip as usize;
                ip = ip.add(1);
                let xv = vld1q_f32(xp.add(pos));
                y0 = vfmaq_laneq_f32::<0>(y0, vld1q_f32(w), xv);
                y1 = vfmaq_laneq_f32::<0>(y1, vld1q_f32(w.add(4)), xv);
                y0 = vfmaq_laneq_f32::<1>(y0, vld1q_f32(w.add(8)), xv);
                y1 = vfmaq_laneq_f32::<1>(y1, vld1q_f32(w.add(12)), xv);
                y0 = vfmaq_laneq_f32::<2>(y0, vld1q_f32(w.add(16)), xv);
                y1 = vfmaq_laneq_f32::<2>(y1, vld1q_f32(w.add(20)), xv);
                y0 = vfmaq_laneq_f32::<3>(y0, vld1q_f32(w.add(24)), xv);
                y1 = vfmaq_laneq_f32::<3>(y1, vld1q_f32(w.add(28)), xv);
                w = w.add(32);
            }
            vst1q_f32(o.add(i), y0);
            vst1q_f32(o.add(i + 4), y1);
        }
    }
}

// ---------------------------------------------------------------------------
// int8 products
// ---------------------------------------------------------------------------

/// The quantized input of an int8 product, aligned for 4-byte loads.
#[repr(C, align(16))]
struct XBuf([i8; MAX_INPUTS]);

/// `xq[j] = quantize(x[j])` for `j < cols` (`dnn/vec_neon.h:385-392`):
/// `vcvtnq_s32_f32(127·x)`, then the low 16 and the low 8 bits.
///
/// # Safety
///
/// NEON. `x` has `cols` values and `xq` room for them.
#[target_feature(enable = "neon")]
#[inline]
unsafe fn quantize(xq: *mut i8, x: *const f32, cols: usize) {
    // SAFETY: the caller guarantees the lengths.
    unsafe {
        let c127 = vdupq_n_f32(127.0);
        let mut j = 0;
        while j + 16 <= cols {
            let a = vcvtnq_s32_f32(vmulq_f32(c127, vld1q_f32(x.add(j))));
            let b = vcvtnq_s32_f32(vmulq_f32(c127, vld1q_f32(x.add(j + 4))));
            let c = vcvtnq_s32_f32(vmulq_f32(c127, vld1q_f32(x.add(j + 8))));
            let d = vcvtnq_s32_f32(vmulq_f32(c127, vld1q_f32(x.add(j + 12))));
            let ab = vmovn_high_s32(vmovn_s32(a), b);
            let cd = vmovn_high_s32(vmovn_s32(c), d);
            vst1q_s8(xq.add(j), vmovn_high_s16(vmovn_s16(ab), cd));
            j += 16;
        }
        while j + 4 <= cols {
            let a = vcvtnq_s32_f32(vmulq_f32(c127, vld1q_f32(x.add(j))));
            let a8 = vmovn_s16(vcombine_s16(vmovn_s32(a), vdup_n_s16(0)));
            vst1_lane_s32::<0>(xq.add(j) as *mut i32, vreinterpret_s32_s8(a8));
            j += 4;
        }
        while j < cols {
            *xq.add(j) = vec_portable::quantize(*x.add(j));
            j += 1;
        }
    }
}

/// The 4 quantized inputs at `q`, repeated in each 32-bit lane.
///
/// # Safety
///
/// NEON; `q` has 4 bytes.
#[target_feature(enable = "neon")]
#[inline]
unsafe fn load_x4(q: *const i8) -> int8x16_t {
    // SAFETY: the caller guarantees 4 readable bytes; the read does not
    // need alignment.
    unsafe { vreinterpretq_s8_s32(vdupq_n_s32(ptr::read_unaligned(q as *const i32))) }
}

/// `out[k] = scale[k] · acc[k]` for the 8 rows of one block.
///
/// # Safety
///
/// NEON; `out` and `scale` have 8 values.
#[target_feature(enable = "neon")]
#[inline]
unsafe fn store8(out: *mut f32, scale: *const f32, a0: int32x4_t, a1: int32x4_t) {
    // SAFETY: the caller guarantees the lengths.
    unsafe {
        vst1q_f32(out, vmulq_f32(vld1q_f32(scale), vcvtq_f32_s32(a0)));
        vst1q_f32(out.add(4), vmulq_f32(vld1q_f32(scale.add(4)), vcvtq_f32_s32(a1)));
    }
}

/// The dot products of 4 rows by 4 columns with the `sdot` instruction.
///
/// # Safety
///
/// NEON and DotProd.
#[target_feature(enable = "neon,dotprod")]
#[inline]
unsafe fn dot_sdot(acc: int32x4_t, w: int8x16_t, x: int8x16_t) -> int32x4_t {
    vdotq_s32(acc, w, x)
}

/// The same dot products without DotProd (`dnn/vec_neon.h:373-376`):
/// products in 16 bits, pairs added in 16 bits, then added to the 32-bit
/// lanes. A pair cannot overflow while the weights lie in [-127, 127].
///
/// # Safety
///
/// NEON.
#[target_feature(enable = "neon")]
#[inline]
unsafe fn dot_neon(acc: int32x4_t, w: int8x16_t, x: int8x16_t) -> int32x4_t {
    vpadalq_s16(acc, vpaddq_s16(vmull_s8(vget_low_s8(w), vget_low_s8(x)), vmull_high_s8(w, x)))
}

/// The dot products of 4 rows by 4 columns with `sdot`, the inputs taken
/// from 32-bit lane `L` of `x` (`sdot` by element).
///
/// # Safety
///
/// NEON and DotProd.
#[target_feature(enable = "neon,dotprod")]
#[inline]
unsafe fn dot_lane_sdot<const L: i32>(acc: int32x4_t, w: int8x16_t, x: int8x16_t) -> int32x4_t {
    vdotq_laneq_s32::<L>(acc, w, x)
}

/// `dot_neon` with the inputs taken from 32-bit lane `L` of `x`.
///
/// # Safety
///
/// NEON.
#[target_feature(enable = "neon")]
#[inline]
unsafe fn dot_lane_neon<const L: i32>(acc: int32x4_t, w: int8x16_t, x: int8x16_t) -> int32x4_t {
    // SAFETY: NEON is enabled.
    unsafe { dot_neon(acc, w, vreinterpretq_s8_s32(vdupq_laneq_s32::<L>(vreinterpretq_s32_s8(x)))) }
}

/// Defines a dense and a sparse int8 product on one dot product function
/// (`$dot`, inputs repeated in every lane) and its by-lane form (`$dotl`).
macro_rules! int8_kernels {
    ($features:literal, $dot:ident, $dotl:ident, $dense:ident, $sparse:ident, $what:literal) => {
        #[doc = concat!("NEON `cgemv8x4` ", $what, ". Same contract as `vec_portable::cgemv8x4`.")]
        ///
        /// One block of 8 rows at a time, so the weights are read as one
        /// sequential stream, as in libopus: with several blocks at a time
        /// (several streams), layers whose weights are not in the caches
        /// ran up to 1.6 times slower on the Cortex-A55 and 1.4 times on
        /// the Cortex-A76. Four column groups share one 16-byte load of
        /// the inputs and run on separate accumulators.
        ///
        /// # Safety
        ///
        /// The processor must support the instructions named above. The
        /// slices must have the lengths that `vec_portable::cgemv8x4`
        /// requires.
        #[target_feature(enable = $features)]
        pub unsafe fn $dense(
            out: &mut [f32],
            weights: &[i8],
            scale: &[f32],
            rows: usize,
            cols: usize,
            x: &[f32],
        ) {
            debug_assert!(cols <= MAX_INPUTS && rows % 8 == 0 && cols % 4 == 0);
            let mut xq = core::mem::MaybeUninit::<XBuf>::uninit();
            let q = xq.as_mut_ptr() as *mut i8;
            let o = out.as_mut_ptr();
            let s = scale.as_ptr();
            let groups = cols / 4;
            // SAFETY: the caller guarantees `x.len() >= cols`,
            // `weights.len() >= rows·cols`, `scale.len() >= rows` and
            // `out.len() >= rows`; `xq` holds `MAX_INPUTS >= cols` bytes,
            // and only the `cols` bytes written by `quantize` are read.
            unsafe {
                quantize(q, x.as_ptr(), cols);
                let mut w = weights.as_ptr();
                for b in 0..rows / 8 {
                    let mut acc = [vdupq_n_s32(0); 8];
                    let mut g = 0;
                    while g + 4 <= groups {
                        let xv = vld1q_s8(q.add(4 * g));
                        let w01 = vld1q_s8_x4(w);
                        let w23 = vld1q_s8_x4(w.add(64));
                        acc[0] = $dotl::<0>(acc[0], w01.0, xv);
                        acc[1] = $dotl::<0>(acc[1], w01.1, xv);
                        acc[2] = $dotl::<1>(acc[2], w01.2, xv);
                        acc[3] = $dotl::<1>(acc[3], w01.3, xv);
                        acc[4] = $dotl::<2>(acc[4], w23.0, xv);
                        acc[5] = $dotl::<2>(acc[5], w23.1, xv);
                        acc[6] = $dotl::<3>(acc[6], w23.2, xv);
                        acc[7] = $dotl::<3>(acc[7], w23.3, xv);
                        w = w.add(128);
                        g += 4;
                    }
                    while g < groups {
                        let vx = load_x4(q.add(4 * g));
                        acc[0] = $dot(acc[0], vld1q_s8(w), vx);
                        acc[1] = $dot(acc[1], vld1q_s8(w.add(16)), vx);
                        w = w.add(32);
                        g += 1;
                    }
                    let a0 = vaddq_s32(vaddq_s32(acc[0], acc[2]), vaddq_s32(acc[4], acc[6]));
                    let a1 = vaddq_s32(vaddq_s32(acc[1], acc[3]), vaddq_s32(acc[5], acc[7]));
                    store8(o.add(8 * b), s.add(8 * b), a0, a1);
                }
            }
        }

        #[doc = concat!("NEON `sparse_cgemv8x4` ", $what, ". Same contract as `vec_portable::sparse_cgemv8x4`.")]
        ///
        /// # Safety
        ///
        /// The processor must support the instructions named above. The
        /// index table must be valid for `rows` and `cols`, and `weights`
        /// must hold 32 bytes for each listed block, as `nnet` checks.
        #[target_feature(enable = $features)]
        pub unsafe fn $sparse(
            out: &mut [f32],
            weights: &[i8],
            idx: &[i32],
            scale: &[f32],
            rows: usize,
            cols: usize,
            x: &[f32],
        ) {
            debug_assert!(cols <= MAX_INPUTS && rows % 8 == 0);
            let mut xq = core::mem::MaybeUninit::<XBuf>::uninit();
            let q = xq.as_mut_ptr() as *mut i8;
            let o = out.as_mut_ptr();
            let s = scale.as_ptr();
            let mut ip = idx.as_ptr();
            let mut w = weights.as_ptr();
            // SAFETY: the caller guarantees the lengths and that each listed
            // position has `pos + 4 <= cols`, so only bytes written by
            // `quantize` are read.
            unsafe {
                quantize(q, x.as_ptr(), cols);
                for i in (0..rows).step_by(8) {
                    let mut acc = [vdupq_n_s32(0); 4];
                    let count = *ip as usize;
                    ip = ip.add(1);
                    let mut c = 0;
                    while c + 2 <= count {
                        let vx0 = load_x4(q.add(*ip as usize));
                        let vx1 = load_x4(q.add(*ip.add(1) as usize));
                        acc[0] = $dot(acc[0], vld1q_s8(w), vx0);
                        acc[1] = $dot(acc[1], vld1q_s8(w.add(16)), vx0);
                        acc[2] = $dot(acc[2], vld1q_s8(w.add(32)), vx1);
                        acc[3] = $dot(acc[3], vld1q_s8(w.add(48)), vx1);
                        ip = ip.add(2);
                        w = w.add(64);
                        c += 2;
                    }
                    if c < count {
                        let vx = load_x4(q.add(*ip as usize));
                        acc[0] = $dot(acc[0], vld1q_s8(w), vx);
                        acc[1] = $dot(acc[1], vld1q_s8(w.add(16)), vx);
                        ip = ip.add(1);
                        w = w.add(32);
                    }
                    let a0 = vaddq_s32(acc[0], acc[2]);
                    let a1 = vaddq_s32(acc[1], acc[3]);
                    store8(o.add(i), s.add(i), a0, a1);
                }
            }
        }
    };
}

int8_kernels!(
    "neon",
    dot_neon,
    dot_lane_neon,
    cgemv8x4,
    sparse_cgemv8x4,
    "without DotProd: the dot products with `vmull_s8`, `vpaddq_s16` and `vpadalq_s16`, as libopus on processors without DotProd"
);
int8_kernels!(
    "neon,dotprod",
    dot_sdot,
    dot_lane_sdot,
    cgemv8x4_dotprod,
    sparse_cgemv8x4_dotprod,
    "with the `sdot` instruction"
);

// ---------------------------------------------------------------------------
// Activations
// ---------------------------------------------------------------------------

/// `exp4_approx` (`dnn/vec_neon.h:59-84`).
///
/// # Safety
///
/// NEON.
#[target_feature(enable = "neon")]
#[inline]
#[allow(clippy::excessive_precision, clippy::approx_constant)]
unsafe fn exp4(x: float32x4_t) -> float32x4_t {
    let x = vmaxq_f32(vminq_f32(x, vdupq_n_f32(88.0)), vdupq_n_f32(-88.0));
    let x = vfmaq_f32(vdupq_n_f32(127.0), x, vdupq_n_f32(1.44269504));
    let i = vcvtq_s32_f32(x);
    let x = vsubq_f32(x, vcvtq_f32_s32(i));
    let k0 = vdupq_n_f32(0.99992522);
    let k1 = vdupq_n_f32(0.69583354);
    let k2 = vdupq_n_f32(0.22606716);
    let k3 = vdupq_n_f32(0.078024523);
    let y = vfmaq_f32(k0, x, vfmaq_f32(k1, x, vfmaq_f32(k2, k3, x)));
    vmulq_f32(y, vreinterpretq_f32_s32(vshlq_n_s32::<23>(i)))
}

/// `tanh4_approx` (`dnn/vec_neon.h:86-104`).
///
/// # Safety
///
/// NEON.
#[target_feature(enable = "neon")]
#[inline]
#[allow(clippy::excessive_precision)]
unsafe fn tanh4(x: float32x4_t) -> float32x4_t {
    let x2 = vmulq_f32(x, x);
    let num = vfmaq_f32(
        vdupq_n_f32(952.52801514),
        x2,
        vfmaq_f32(vdupq_n_f32(96.39235687), vdupq_n_f32(0.60863042), x2),
    );
    let den = vfmaq_f32(
        vdupq_n_f32(952.72399902),
        x2,
        vfmaq_f32(vdupq_n_f32(413.36801147), vdupq_n_f32(11.88600922), x2),
    );
    let num = vmulq_f32(num, x);
    let num = vmulq_f32(num, vrecpeq_f32(den));
    vmaxq_f32(vdupq_n_f32(-1.0), vminq_f32(vdupq_n_f32(1.0), num))
}

/// `sigmoid4_approx` (`dnn/vec_neon.h:106-125`).
///
/// # Safety
///
/// NEON.
#[target_feature(enable = "neon")]
#[inline]
#[allow(clippy::excessive_precision)]
unsafe fn sigmoid4(x: float32x4_t) -> float32x4_t {
    let x2 = vmulq_f32(x, x);
    let num = vfmaq_f32(
        vdupq_n_f32(238.13200378),
        x2,
        vfmaq_f32(vdupq_n_f32(6.02452230), vdupq_n_f32(0.00950985), x2),
    );
    let den = vfmaq_f32(
        vdupq_n_f32(952.72399902),
        x2,
        vfmaq_f32(vdupq_n_f32(103.34200287), vdupq_n_f32(0.74287558), x2),
    );
    let num = vmulq_f32(num, x);
    let num = vfmaq_f32(vdupq_n_f32(0.5), num, vrecpeq_f32(den));
    vmaxq_f32(vdupq_n_f32(0.0), vminq_f32(vdupq_n_f32(1.0), num))
}

/// Applies the vector function `$f` to the first `4·floor(len/4)` values
/// of `$buf`, 4 vectors per loop step (16 values), so that an in-order
/// core such as the Cortex-A55 can overlap their chains of dependent
/// instructions. Evaluates to the number of values done.
macro_rules! map_groups {
    ($buf:expr, $f:ident) => {{
        let buf: &mut [f32] = $buf;
        let groups = buf.len() & !3;
        let p = buf.as_mut_ptr();
        let mut i = 0;
        // SAFETY: NEON is enabled in the calling function, and every
        // access lies below `groups <= buf.len()`.
        unsafe {
            while i + 16 <= groups {
                let a = $f(vld1q_f32(p.add(i)));
                let b = $f(vld1q_f32(p.add(i + 4)));
                let c = $f(vld1q_f32(p.add(i + 8)));
                let d = $f(vld1q_f32(p.add(i + 12)));
                vst1q_f32(p.add(i), a);
                vst1q_f32(p.add(i + 4), b);
                vst1q_f32(p.add(i + 8), c);
                vst1q_f32(p.add(i + 12), d);
                i += 16;
            }
            while i + 4 <= groups {
                vst1q_f32(p.add(i), $f(vld1q_f32(p.add(i))));
                i += 4;
            }
        }
        groups
    }};
}

/// NEON `vec_exp`. Same contract as `vec_portable::vec_exp`.
///
/// # Safety
///
/// The processor must support NEON.
#[target_feature(enable = "neon")]
pub unsafe fn vec_exp(buf: &mut [f32]) {
    let groups = map_groups!(buf, exp4);
    for v in &mut buf[groups..] {
        *v = vec_portable::lpcnet_exp(*v);
    }
}

/// NEON `vec_tanh` (with `vrecpeq_f32`). Same contract as
/// `vec_portable::vec_tanh`.
///
/// # Safety
///
/// The processor must support NEON.
#[target_feature(enable = "neon")]
pub unsafe fn vec_tanh(buf: &mut [f32]) {
    let groups = map_groups!(buf, tanh4);
    for v in &mut buf[groups..] {
        *v = vec_portable::tanh_tail(*v);
    }
}

/// NEON `vec_sigmoid` (with `vrecpeq_f32`). Same contract as
/// `vec_portable::vec_sigmoid`.
///
/// # Safety
///
/// The processor must support NEON.
#[target_feature(enable = "neon")]
pub unsafe fn vec_sigmoid(buf: &mut [f32]) {
    let groups = map_groups!(buf, sigmoid4);
    for v in &mut buf[groups..] {
        *v = vec_portable::sigmoid_tail(*v);
    }
}
