#[cfg(not(feature = "std"))]
use crate::compat::Math;
use crate::celt_lpc::{autocorr, lpc};
use crate::fixedvec::FixedVec;

/// Capacity for pitch-search working buffers (x_lp4 / y_lp4 / xcorr). Bounded by
/// `COMBFILTER_MAXPERIOD = 1024` and the maximum pitch lag.
const PITCH_BUF_MAX: usize = 1024;

// How the sums of products in this file are rounded
// --------------------------------------------------
// The encoder output depends on the exact rounding of these sums, so every
// kernel keeps the order of additions it has always had. Two orders exist:
//
// * "By sample": one running sum per lag, and the products are added one
//   sample after the other. `pitch_xcorr` uses it for groups of four lags on
//   aarch64, and for every lag in the portable code.
// * "Four lanes": four running sums that take samples 4m, 4m+1, 4m+2 and 4m+3.
//   They are added as (s0 + s1) + (s2 + s3), and the last `n % 4` products are
//   added one by one after that. `inner_prod` uses it on aarch64.
//
// The aarch64 kernels add each product with a fused multiply-add, which rounds
// once. The portable code multiplies and adds separately, which rounds twice.
// The tail of the four-lane order also multiplies and adds separately.
//
// Several lags can be computed in the same pass without changing any result,
// as long as every lag keeps its own running sum. The kernels below do that:
// the processor then works on many sums at the same time instead of waiting
// for one addition after the other.

/// `acc + a * b`.
///
/// With `FUSED` the product is not rounded before the addition (fused
/// multiply-add). The aarch64 kernels work that way. Without `FUSED` the
/// product is rounded first, as the portable code has always done.
#[allow(dead_code)] // unused on targets that have a vector kernel
#[inline(always)]
fn mul_acc<const FUSED: bool>(acc: f32, a: f32, b: f32) -> f32 {
    if FUSED {
        a.mul_add(b, acc)
    } else {
        acc + a * b
    }
}

/// Lags per pass of `xcorr_by_sample_scalar`.
#[allow(dead_code)] // unused on targets that have a vector kernel
const SCALAR_LAGS_PER_PASS: usize = 8;

/// Cross-correlation in the by-sample order, portable code.
///
/// `out[k] = sum over j < len of x[j] * y[j + k]` for every `k < out.len()`.
/// Each lag has its own running sum and adds its products in rising order of
/// `j`. Eight lags are handled per pass, which gives the same result as one
/// lag after the other and is faster.
///
/// `FUSED = false` is the portable fallback. `FUSED = true` gives the same bits
/// as `xcorr_by_sample_neon` and is its reference in the tests.
///
/// Needs `x.len() >= len` and `y.len() >= len + out.len() - 1`.
#[allow(dead_code)] // unused on targets that have a vector kernel
fn xcorr_by_sample_scalar<const FUSED: bool>(x: &[f32], y: &[f32], out: &mut [f32], len: usize) {
    const L: usize = SCALAR_LAGS_PER_PASS;
    let lags = out.len();
    if lags == 0 {
        return;
    }
    let x = &x[..len];
    let y = &y[..len + lags - 1];

    if lags < L {
        for (k, o) in out.iter_mut().enumerate() {
            let mut acc = 0.0f32;
            for (&xj, &yj) in x.iter().zip(&y[k..]) {
                acc = mul_acc::<FUSED>(acc, xj, yj);
            }
            *o = acc;
        }
        return;
    }

    let mut pass = |k: usize| {
        let mut acc = [0.0f32; L];
        // `windows(L)` gives y[j + k..j + k + L] for j = 0, 1, ..; the zip
        // stops after `len` of them.
        for (&xj, yw) in x.iter().zip(y[k..].windows(L)) {
            for l in 0..L {
                acc[l] = mul_acc::<FUSED>(acc[l], xj, yw[l]);
            }
        }
        out[k..k + L].copy_from_slice(&acc);
    };
    let mut k = 0;
    while k + L <= lags {
        pass(k);
        k += L;
    }
    if k < lags {
        // The last pass starts early enough to end at the last lag. It
        // computes some lags a second time, with the same result.
        pass(lags - L);
    }
}

/// Sum of `x[i] * y[i]` for `i < n`. Panics when `x` or `y` is shorter than
/// `n`.
pub fn inner_prod(x: &[f32], y: &[f32], n: usize) -> f32 {
    // The vector kernels read `n` samples through pointers without a check,
    // so the lengths are checked here.
    let (x, y) = (&x[..n], &y[..n]);
    // SAFETY (all three blocks): `x` and `y` hold exactly `n` samples. The
    // AVX kernel runs only when the CPU has AVX and FMA; NEON and SSE are
    // part of the baseline of their targets.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    unsafe {
        if crate::compat::x86_has_avx_fma() {
            return inner_prod_avx(x, y, n);
        }
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
        inner_prod_neon(x, y, n)
    }
    #[cfg(all(target_arch = "x86_64", target_feature = "sse"))]
    unsafe {
        inner_prod_sse(x, y, n)
    }
    #[cfg(not(any(
        target_arch = "aarch64",
        all(target_arch = "x86_64", target_feature = "sse")
    )))]
    {
        let mut sum = 0.0f32;
        for i in 0..n {
            sum += x[i] * y[i];
        }
        sum
    }
}

/// The inner products of `x` with `y1` and with `y2` over `n` samples. Panics
/// when one of the three is shorter than `n`.
pub fn dual_inner_prod(x: &[f32], y1: &[f32], y2: &[f32], n: usize) -> (f32, f32) {
    // As in `inner_prod`: the kernels read `n` samples without a check.
    let (x, y1, y2) = (&x[..n], &y1[..n], &y2[..n]);
    // SAFETY (all three blocks): the three slices hold exactly `n` samples.
    // The AVX kernel runs only when the CPU has AVX and FMA; NEON and SSE
    // are part of the baseline of their targets.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    unsafe {
        if crate::compat::x86_has_avx_fma() {
            return dual_inner_prod_avx(x, y1, y2, n);
        }
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
        dual_inner_prod_neon(x, y1, y2, n)
    }
    #[cfg(all(target_arch = "x86_64", target_feature = "sse"))]
    unsafe {
        dual_inner_prod_sse(x, y1, y2, n)
    }
    #[cfg(not(any(
        target_arch = "aarch64",
        all(target_arch = "x86_64", target_feature = "sse")
    )))]
    {
        let mut xy1 = 0.0f32;
        let mut xy2 = 0.0f32;
        for i in 0..n {
            xy1 += x[i] * y1[i];
            xy2 += x[i] * y2[i];
        }
        (xy1, xy2)
    }
}

pub fn pitch_xcorr(x: &[f32], y: &[f32], xcorr: &mut [f32], len: usize, max_pitch: usize) {
    // Every lag window reads y[i..i+len]; the SIMD kernels peek up to 3
    // elements past the window. The combined contract is
    // y.len() >= len + max_pitch - 1; clamp so that direct callers with a
    // short `y` degrade gracefully instead of reading past the slice
    // (issue #27 deep scan). In-repo callers always satisfy the contract and
    // are unaffected.
    let max_pitch = if len > 0 {
        max_pitch.min((y.len() + 1).saturating_sub(len))
    } else {
        0
    };
    if max_pitch == 0 {
        return;
    }
    // The vector kernels read `len` samples of `x` through pointers without
    // a check, so its length is checked here.
    let x = &x[..len];
    // SAFETY (both blocks): `x` holds exactly `len` samples and `y` at least
    // `len + max_pitch - 1` (`max_pitch` was limited to that above). `xcorr`
    // is written with index checks. The AVX kernel runs only when the CPU
    // has AVX and FMA.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    unsafe {
        if crate::compat::x86_has_avx_fma() {
            return pitch_xcorr_avx(x, y, xcorr, len, max_pitch);
        }
    }
    #[cfg(target_arch = "aarch64")]
    pitch_xcorr_neon(x, y, &mut xcorr[..max_pitch]);
    #[cfg(all(target_arch = "x86_64", target_feature = "sse"))]
    unsafe {
        pitch_xcorr_sse(x, y, xcorr, len, max_pitch)
    }
    #[cfg(not(any(
        target_arch = "aarch64",
        all(target_arch = "x86_64", target_feature = "sse")
    )))]
    xcorr_by_sample_scalar::<false>(x, y, &mut xcorr[..max_pitch], len);
}

/// Most inner products `inner_prod_many` computes in one pass on aarch64.
#[cfg(target_arch = "aarch64")]
const MANY_PER_PASS: usize = 16;

/// Several inner products of the same `x` with windows of `y`:
/// `out[k] = inner_prod(x, &y[offsets[k]..], n)` for every `k`.
///
/// The result has the same bits as one call of `inner_prod` per window. On
/// aarch64 up to 16 windows are computed in one pass, each with its own
/// running sums, which is several times faster than one window after the
/// other. Other targets call `inner_prod` once per window.
///
/// Panics when `x` is shorter than `n`, when a window does not fit in `y`, or
/// when `offsets` and `out` differ in length.
fn inner_prod_many(x: &[f32], y: &[f32], offsets: &[usize], n: usize, out: &mut [f32]) {
    assert_eq!(offsets.len(), out.len());
    #[cfg(target_arch = "aarch64")]
    {
        let x = &x[..n];
        for (offs, out) in offsets
            .chunks(MANY_PER_PASS)
            .zip(out.chunks_mut(MANY_PER_PASS))
        {
            // The windows of this pass. Slicing checks that each one fits.
            // Entries past the last window repeat the first one; their sums
            // are not used.
            let ys: [&[f32]; MANY_PER_PASS] = core::array::from_fn(|k| {
                let off = offs[if k < offs.len() { k } else { 0 }];
                &y[off..off + n]
            });
            // The kernel is only as wide as the windows need.
            match offs.len().div_ceil(4) {
                1 => inner_prod_pass_neon::<4>(x, &ys, out),
                2 => inner_prod_pass_neon::<8>(x, &ys, out),
                3 => inner_prod_pass_neon::<12>(x, &ys, out),
                _ => inner_prod_pass_neon::<16>(x, &ys, out),
            }
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    for (o, &off) in out.iter_mut().zip(offsets) {
        *o = inner_prod(x, &y[off..], n);
    }
}

/// One pass of `inner_prod_many`: stores the first `out.len()` of the `K`
/// products of `x` with the windows `ys[..K]`. Needs `out.len() <= K <= 16`.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn inner_prod_pass_neon<const K: usize>(x: &[f32], ys: &[&[f32]; MANY_PER_PASS], out: &mut [f32]) {
    let ys_k: [&[f32]; K] = core::array::from_fn(|k| ys[k]);
    let sums = inner_prod_k_neon::<K>(x, &ys_k);
    out.copy_from_slice(&sums[..out.len()]);
}

/// `inner_prod` on aarch64, in the four-lane order.
///
/// # Safety
///
/// `x` and `y` must hold at least `n` samples each. Every load covers
/// `[i..i + 4]` with `i + 4 <= n`.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn inner_prod_neon(x: &[f32], y: &[f32], n: usize) -> f32 {
    use core::arch::aarch64::*;

    debug_assert!(x.len() >= n && y.len() >= n);
    let mut xy = vdupq_n_f32(0.0);
    let mut i = 0;

    while i + 8 <= n {
        let x0 = vld1q_f32(x.as_ptr().add(i));
        let y0 = vld1q_f32(y.as_ptr().add(i));
        xy = vfmaq_f32(xy, x0, y0);

        let x1 = vld1q_f32(x.as_ptr().add(i + 4));
        let y1 = vld1q_f32(y.as_ptr().add(i + 4));
        xy = vfmaq_f32(xy, x1, y1);

        i += 8;
    }

    if i + 4 <= n {
        let x0 = vld1q_f32(x.as_ptr().add(i));
        let y0 = vld1q_f32(y.as_ptr().add(i));
        xy = vfmaq_f32(xy, x0, y0);
        i += 4;
    }

    let mut sum = vaddvq_f32(xy);

    for j in i..n {
        sum += x[j] * y[j];
    }

    sum
}

/// `dual_inner_prod` on aarch64: two inner products in the four-lane order.
///
/// # Safety
///
/// `x`, `y1` and `y2` must hold at least `n` samples each. Every load covers
/// `[i..i + 4]` with `i + 4 <= n`.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn dual_inner_prod_neon(x: &[f32], y1: &[f32], y2: &[f32], n: usize) -> (f32, f32) {
    use core::arch::aarch64::*;

    debug_assert!(x.len() >= n && y1.len() >= n && y2.len() >= n);
    let mut xy1 = vdupq_n_f32(0.0);
    let mut xy2 = vdupq_n_f32(0.0);
    let mut i = 0;

    while i + 8 <= n {
        let x0 = vld1q_f32(x.as_ptr().add(i));
        let x4 = vld1q_f32(x.as_ptr().add(i + 4));

        let y1_0 = vld1q_f32(y1.as_ptr().add(i));
        let y1_4 = vld1q_f32(y1.as_ptr().add(i + 4));
        let y2_0 = vld1q_f32(y2.as_ptr().add(i));
        let y2_4 = vld1q_f32(y2.as_ptr().add(i + 4));

        xy1 = vfmaq_f32(xy1, x0, y1_0);
        xy2 = vfmaq_f32(xy2, x0, y2_0);
        xy1 = vfmaq_f32(xy1, x4, y1_4);
        xy2 = vfmaq_f32(xy2, x4, y2_4);

        i += 8;
    }

    if i + 4 <= n {
        let x0 = vld1q_f32(x.as_ptr().add(i));
        let y1_0 = vld1q_f32(y1.as_ptr().add(i));
        let y2_0 = vld1q_f32(y2.as_ptr().add(i));
        xy1 = vfmaq_f32(xy1, x0, y1_0);
        xy2 = vfmaq_f32(xy2, x0, y2_0);
        i += 4;
    }

    let sum1 = vaddvq_f32(xy1);
    let sum2 = vaddvq_f32(xy2);

    let mut s1 = sum1;
    let mut s2 = sum2;
    for j in i..n {
        s1 += x[j] * y1[j];
        s2 += x[j] * y2[j];
    }

    (s1, s2)
}

/// `K` inner products of `x` with `K` other signals in one pass, in the
/// four-lane order. `sums[k]` has the same bits as
/// `inner_prod_neon(x, ys[k], x.len())`.
///
/// Each product has its own register of running sums. One product alone has
/// to wait four cycles for each fused multiply-add; with eight of them the
/// processor works on the others in the meantime.
///
/// Panics when one of the signals is shorter than `x`.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn inner_prod_k_neon<const K: usize>(x: &[f32], ys: &[&[f32]; K]) -> [f32; K] {
    use core::arch::aarch64::*;

    let n = x.len();
    assert!(ys.iter().all(|y| y.len() >= n));
    let x = x.as_ptr();
    let ys: [*const f32; K] = core::array::from_fn(|k| ys[k].as_ptr());

    // SAFETY: `x` and every signal in `ys` hold at least `n` samples (checked
    // above). The vector loop reads x[i..i + 4] and ys[k][i..i + 4] with
    // i + 4 <= n; the scalar loop reads index j < n.
    unsafe {
        let mut acc = [vdupq_n_f32(0.0); K];
        let mut i = 0;
        while i + 4 <= n {
            let xv = vld1q_f32(x.add(i));
            for (a, &y) in acc.iter_mut().zip(&ys) {
                *a = vfmaq_f32(*a, xv, vld1q_f32(y.add(i)));
            }
            i += 4;
        }
        let mut sums = [0.0f32; K];
        for ((sum, a), &y) in sums.iter_mut().zip(&acc).zip(&ys) {
            // (lane 0 + lane 1) + (lane 2 + lane 3)
            let mut s = vaddvq_f32(*a);
            for j in i..n {
                s += *x.add(j) * *y.add(j);
            }
            *sum = s;
        }
        sums
    }
}

/// Cross-correlation in the by-sample order for `4 * G` neighbouring lags.
///
/// `out[k] = sum over j < len of x[j] * y[j + k]` for `k < 4 * G`. Lane `l` of
/// running sum `g` belongs to lag `4 * g + l`. Every lag adds its products in
/// rising order of `j`, each with one fused multiply-add. The result has the
/// same bits as `xcorr_by_sample_scalar::<true>`.
///
/// One lag alone has to wait four cycles for each addition before it can
/// start the next one. With 64 lags in 16 registers (`G = 16`) the processor
/// always has other additions to work on. On an Apple M4 that is 6 times
/// faster than four lags per pass, and 1.4 times faster than `G = 8`.
///
/// `len` is `x.len()`. Only the first `4 * G` elements of `out` are written.
/// Panics unless `y` holds `len + 4 * G - 1` samples and `out` holds `4 * G`
/// lags. The last sample of the last lag is `y[len + 4 * G - 2]`, and no load
/// goes past it.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn xcorr_by_sample_neon<const G: usize>(x: &[f32], y: &[f32], out: &mut [f32]) {
    use core::arch::aarch64::*;

    let len = x.len();
    assert!(y.len() >= len + 4 * G - 1 && out.len() >= 4 * G);
    let (x, y, out) = (x.as_ptr(), y.as_ptr(), out.as_mut_ptr());

    // SAFETY: `x` holds `len` samples, `y` at least len + 4G - 1 and `out` at
    // least 4G (checked above). The four-sample loop reads x[j..j + 4] with
    // j + 4 <= len, and y[j + 4g + s..][..4] with s <= 3 and g < G. The last
    // element of such a load is
    // y[j + 4g + s + 3] <= y[len - 4 + 4(G - 1) + 6] = y[len + 4G - 2]. The
    // one-sample loop reads x[j] and y[j + 4g..][..4] with j < len, which
    // ends at y[len + 4G - 2] as well. The stores write out[..4G].
    unsafe {
        let mut acc = [vdupq_n_f32(0.0); G];
        let mut j = 0;
        while j + 4 <= len {
            let xv = vld1q_f32(x.add(j));
            for (g, a) in acc.iter_mut().enumerate() {
                let yp = y.add(j + 4 * g);
                let mut s = *a;
                s = vfmaq_laneq_f32::<0>(s, vld1q_f32(yp), xv);
                s = vfmaq_laneq_f32::<1>(s, vld1q_f32(yp.add(1)), xv);
                s = vfmaq_laneq_f32::<2>(s, vld1q_f32(yp.add(2)), xv);
                s = vfmaq_laneq_f32::<3>(s, vld1q_f32(yp.add(3)), xv);
                *a = s;
            }
            j += 4;
        }
        while j < len {
            let xv = vld1q_dup_f32(x.add(j));
            for (g, a) in acc.iter_mut().enumerate() {
                *a = vfmaq_f32(*a, vld1q_f32(y.add(j + 4 * g)), xv);
            }
            j += 1;
        }
        for (g, a) in acc.iter().enumerate() {
            vst1q_f32(out.add(4 * g), *a);
        }
    }
}

/// Fills `out` with `xcorr_by_sample_neon::<G>`, `4 * G` lags per pass.
///
/// Panics unless `out` holds at least `4 * G` lags and `y` at least
/// `x.len() + out.len() - 1` samples.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn xcorr_by_sample_passes_neon<const G: usize>(x: &[f32], y: &[f32], out: &mut [f32]) {
    let per_pass = 4 * G;
    let lags = out.len();
    assert!(lags >= per_pass);

    let mut i = 0;
    while i < lags {
        // When fewer than 4G lags are left, the last pass starts early
        // enough to end at the last lag. It computes some lags a second
        // time, with the same result. A shorter pass would not be faster:
        // the time of a pass is set by the chain of additions of one lag,
        // not by the number of lags.
        let first = i.min(lags - per_pass);
        // The kernel checks that `y` is long enough for these lags.
        xcorr_by_sample_neon::<G>(x, &y[first..], &mut out[first..]);
        i += per_pass;
    }
}

/// `pitch_xcorr` for aarch64: `xcorr[k] = sum over j of x[j] * y[j + k]` with
/// `len = x.len()` and one lag per element of `xcorr`.
///
/// With 32 lags or more, the lags below the largest multiple of four are
/// summed by sample and the remaining one to three lags in four lanes. With
/// fewer than 32 lags every lag is summed in four lanes. This split is what
/// the function has always done, and the output of the encoder depends on it.
#[cfg(target_arch = "aarch64")]
fn pitch_xcorr_neon(x: &[f32], y: &[f32], xcorr: &mut [f32]) {
    let len = x.len();
    let lags = xcorr.len();
    if lags == 0 {
        return;
    }
    // The last lag reads y[lags - 1..lags - 1 + len].
    assert!(len > 0 && y.len() >= len + lags - 1);

    let by_sample = if lags >= 32 { lags & !3 } else { 0 };
    if by_sample >= 64 {
        xcorr_by_sample_passes_neon::<16>(x, y, &mut xcorr[..by_sample]);
    } else if by_sample >= 32 {
        xcorr_by_sample_passes_neon::<8>(x, y, &mut xcorr[..by_sample]);
    }

    // At most 31 lags are left: all of them when fewer than 32 were asked
    // for, else the one to three after the last group of four.
    let rest = lags - by_sample;
    let mut offsets = [0usize; 31];
    for (k, off) in offsets[..rest].iter_mut().enumerate() {
        *off = by_sample + k;
    }
    inner_prod_many(x, y, &offsets[..rest], len, &mut xcorr[by_sample..]);
}

/// # Safety
///
/// `x` and `y` must hold at least `n` samples each, and the CPU must have the
/// features this function is compiled for.
#[cfg(all(target_arch = "x86_64", target_feature = "sse"))]
#[inline(always)]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn inner_prod_sse(x: &[f32], y: &[f32], n: usize) -> f32 {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;

    let mut sum0 = _mm_setzero_ps();
    let mut sum1 = _mm_setzero_ps();
    let mut i = 0;

    while i + 8 <= n {
        let x0 = _mm_loadu_ps(x.as_ptr().add(i));
        let y0 = _mm_loadu_ps(y.as_ptr().add(i));
        sum0 = _mm_add_ps(sum0, _mm_mul_ps(x0, y0));
        let x1 = _mm_loadu_ps(x.as_ptr().add(i + 4));
        let y1 = _mm_loadu_ps(y.as_ptr().add(i + 4));
        sum1 = _mm_add_ps(sum1, _mm_mul_ps(x1, y1));
        i += 8;
    }

    if i + 4 <= n {
        let x0 = _mm_loadu_ps(x.as_ptr().add(i));
        let y0 = _mm_loadu_ps(y.as_ptr().add(i));
        sum0 = _mm_add_ps(sum0, _mm_mul_ps(x0, y0));
        i += 4;
    }

    let sum = _mm_add_ps(sum0, sum1);
    let tmp = _mm_movehl_ps(sum, sum);
    let sum = _mm_add_ps(sum, tmp);
    let tmp2 = _mm_shuffle_ps(sum, sum, 0x55);
    let sum = _mm_add_ss(sum, tmp2);
    let mut result = _mm_cvtss_f32(sum);

    while i < n {
        result += x[i] * y[i];
        i += 1;
    }

    result
}

/// # Safety
///
/// `x`, `y1` and `y2` must hold at least `n` samples each, and the CPU must
/// have the features this function is compiled for.
#[cfg(all(target_arch = "x86_64", target_feature = "sse"))]
#[inline(always)]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn dual_inner_prod_sse(x: &[f32], y1: &[f32], y2: &[f32], n: usize) -> (f32, f32) {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;

    let mut xy1 = _mm_setzero_ps();
    let mut xy2 = _mm_setzero_ps();
    let mut i = 0;

    while i + 4 <= n {
        let xi = _mm_loadu_ps(x.as_ptr().add(i));
        let y1i = _mm_loadu_ps(y1.as_ptr().add(i));
        let y2i = _mm_loadu_ps(y2.as_ptr().add(i));
        xy1 = _mm_add_ps(xy1, _mm_mul_ps(xi, y1i));
        xy2 = _mm_add_ps(xy2, _mm_mul_ps(xi, y2i));
        i += 4;
    }

    let tmp = _mm_movehl_ps(xy1, xy1);
    let xy1 = _mm_add_ps(xy1, tmp);
    let tmp2 = _mm_shuffle_ps(xy1, xy1, 0x55);
    let xy1 = _mm_add_ss(xy1, tmp2);
    let mut s1 = _mm_cvtss_f32(xy1);

    let tmp = _mm_movehl_ps(xy2, xy2);
    let xy2 = _mm_add_ps(xy2, tmp);
    let tmp2 = _mm_shuffle_ps(xy2, xy2, 0x55);
    let xy2 = _mm_add_ss(xy2, tmp2);
    let mut s2 = _mm_cvtss_f32(xy2);

    while i < n {
        s1 += x[i] * y1[i];
        s2 += x[i] * y2[i];
        i += 1;
    }

    (s1, s2)
}

/// # Safety
///
/// `x` must hold at least `len` samples and `y` at least `len + 3`, and the
/// CPU must have the features this function is compiled for. The loads read
/// up to `y[len + 2]`.
#[cfg(all(target_arch = "x86_64", target_feature = "sse"))]
#[inline(always)]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn xcorr_kernel_sse(x: &[f32], y: &[f32], sum: &mut [f32; 4], len: usize) {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;

    let mut xsum1 = _mm_loadu_ps(sum.as_ptr());
    let mut xsum2 = _mm_setzero_ps();

    let mut j = 0;

    while j + 4 <= len {
        let x0 = _mm_loadu_ps(x.as_ptr().add(j));
        let yj = _mm_loadu_ps(y.as_ptr().add(j));
        let y3 = _mm_loadu_ps(y.as_ptr().add(j + 3));

        xsum1 = _mm_add_ps(xsum1, _mm_mul_ps(_mm_shuffle_ps(x0, x0, 0x00), yj));

        xsum2 = _mm_add_ps(
            xsum2,
            _mm_mul_ps(_mm_shuffle_ps(x0, x0, 0x55), _mm_shuffle_ps(yj, y3, 0x49)),
        );

        xsum1 = _mm_add_ps(
            xsum1,
            _mm_mul_ps(_mm_shuffle_ps(x0, x0, 0xaa), _mm_shuffle_ps(yj, y3, 0x9e)),
        );

        xsum2 = _mm_add_ps(xsum2, _mm_mul_ps(_mm_shuffle_ps(x0, x0, 0xff), y3));

        j += 4;
    }

    if j < len {
        xsum1 = _mm_add_ps(
            xsum1,
            _mm_mul_ps(
                _mm_set1_ps(*x.as_ptr().add(j)),
                _mm_loadu_ps(y.as_ptr().add(j)),
            ),
        );
        j += 1;
        if j < len {
            xsum2 = _mm_add_ps(
                xsum2,
                _mm_mul_ps(
                    _mm_set1_ps(*x.as_ptr().add(j)),
                    _mm_loadu_ps(y.as_ptr().add(j)),
                ),
            );
            j += 1;
            if j < len {
                xsum1 = _mm_add_ps(
                    xsum1,
                    _mm_mul_ps(
                        _mm_set1_ps(*x.as_ptr().add(j)),
                        _mm_loadu_ps(y.as_ptr().add(j)),
                    ),
                );
            }
        }
    }

    _mm_storeu_ps(sum.as_mut_ptr(), _mm_add_ps(xsum1, xsum2));
}

/// # Safety
///
/// `x` must hold at least `len` samples and `y` at least
/// `len + max_pitch - 1`, and the CPU must have the features this function is
/// compiled for. `xcorr` is written with index checks.
#[cfg(all(target_arch = "x86_64", target_feature = "sse"))]
#[inline(always)]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn pitch_xcorr_sse(x: &[f32], y: &[f32], xcorr: &mut [f32], len: usize, max_pitch: usize) {
    let mut i = 0;

    while i + 4 <= max_pitch {
        let mut sum = [0.0f32; 4];
        xcorr_kernel_sse(x, &y[i..], &mut sum, len);
        xcorr[i] = sum[0];
        xcorr[i + 1] = sum[1];
        xcorr[i + 2] = sum[2];
        xcorr[i + 3] = sum[3];
        i += 4;
    }

    for j in i..max_pitch {
        xcorr[j] = inner_prod_sse(x, &y[j..], len);
    }
}

/// # Safety
///
/// `x` and `y` must hold at least `n` samples each, and the CPU must have the
/// features this function is compiled for.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx,fma")]
unsafe fn inner_prod_avx(x: &[f32], y: &[f32], n: usize) -> f32 {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;

    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut i = 0usize;

    while i + 16 <= n {
        let x0 = _mm256_loadu_ps(x.as_ptr().add(i));
        let y0 = _mm256_loadu_ps(y.as_ptr().add(i));
        acc0 = _mm256_fmadd_ps(x0, y0, acc0);

        let x1 = _mm256_loadu_ps(x.as_ptr().add(i + 8));
        let y1 = _mm256_loadu_ps(y.as_ptr().add(i + 8));
        acc1 = _mm256_fmadd_ps(x1, y1, acc1);
        i += 16;
    }

    while i + 8 <= n {
        let x0 = _mm256_loadu_ps(x.as_ptr().add(i));
        let y0 = _mm256_loadu_ps(y.as_ptr().add(i));
        acc0 = _mm256_fmadd_ps(x0, y0, acc0);
        i += 8;
    }

    let acc = _mm256_add_ps(acc0, acc1);
    let hi = _mm256_extractf128_ps(acc, 1);
    let lo = _mm256_castps256_ps128(acc);
    let sum4 = _mm_add_ps(lo, hi);
    let tmp = _mm_movehl_ps(sum4, sum4);
    let sum2 = _mm_add_ps(sum4, tmp);
    let tmp2 = _mm_shuffle_ps(sum2, sum2, 0x55);
    let mut result = _mm_cvtss_f32(_mm_add_ss(sum2, tmp2));

    while i < n {
        result += x[i] * y[i];
        i += 1;
    }

    result
}

/// # Safety
///
/// `x`, `y1` and `y2` must hold at least `n` samples each, and the CPU must
/// have the features this function is compiled for.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx,fma")]
unsafe fn dual_inner_prod_avx(x: &[f32], y1: &[f32], y2: &[f32], n: usize) -> (f32, f32) {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;

    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc1b = _mm256_setzero_ps();
    let mut acc2b = _mm256_setzero_ps();
    let mut i = 0usize;

    while i + 16 <= n {
        let xv0 = _mm256_loadu_ps(x.as_ptr().add(i));
        let xv1 = _mm256_loadu_ps(x.as_ptr().add(i + 8));
        let y1v0 = _mm256_loadu_ps(y1.as_ptr().add(i));
        let y2v0 = _mm256_loadu_ps(y2.as_ptr().add(i));
        let y1v1 = _mm256_loadu_ps(y1.as_ptr().add(i + 8));
        let y2v1 = _mm256_loadu_ps(y2.as_ptr().add(i + 8));
        acc1 = _mm256_fmadd_ps(xv0, y1v0, acc1);
        acc2 = _mm256_fmadd_ps(xv0, y2v0, acc2);
        acc1b = _mm256_fmadd_ps(xv1, y1v1, acc1b);
        acc2b = _mm256_fmadd_ps(xv1, y2v1, acc2b);
        i += 16;
    }

    while i + 8 <= n {
        let xv = _mm256_loadu_ps(x.as_ptr().add(i));
        let y1v = _mm256_loadu_ps(y1.as_ptr().add(i));
        let y2v = _mm256_loadu_ps(y2.as_ptr().add(i));
        acc1 = _mm256_fmadd_ps(xv, y1v, acc1);
        acc2 = _mm256_fmadd_ps(xv, y2v, acc2);
        i += 8;
    }

    let acc1 = _mm256_add_ps(acc1, acc1b);
    let acc2 = _mm256_add_ps(acc2, acc2b);

    let hi1 = _mm256_extractf128_ps(acc1, 1);
    let lo1 = _mm256_castps256_ps128(acc1);
    let sum41 = _mm_add_ps(lo1, hi1);
    let t11 = _mm_movehl_ps(sum41, sum41);
    let s21 = _mm_add_ps(sum41, t11);
    let t12 = _mm_shuffle_ps(s21, s21, 0x55);
    let mut s1 = _mm_cvtss_f32(_mm_add_ss(s21, t12));

    let hi2 = _mm256_extractf128_ps(acc2, 1);
    let lo2 = _mm256_castps256_ps128(acc2);
    let sum42 = _mm_add_ps(lo2, hi2);
    let t21 = _mm_movehl_ps(sum42, sum42);
    let s22 = _mm_add_ps(sum42, t21);
    let t22 = _mm_shuffle_ps(s22, s22, 0x55);
    let mut s2 = _mm_cvtss_f32(_mm_add_ss(s22, t22));

    while i < n {
        s1 += x[i] * y1[i];
        s2 += x[i] * y2[i];
        i += 1;
    }

    (s1, s2)
}

/// # Safety
///
/// `x` must hold at least `len` samples and `y` at least
/// `len + max_pitch - 1`, and the CPU must have the features this function is
/// compiled for. `xcorr` is written with index checks.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx,fma")]
unsafe fn pitch_xcorr_avx(x: &[f32], y: &[f32], xcorr: &mut [f32], len: usize, max_pitch: usize) {
    let mut i = 0;

    while i + 4 <= max_pitch {
        let mut sum = [0.0f32; 4];
        xcorr_kernel_avx(x, &y[i..], &mut sum, len);
        xcorr[i] = sum[0];
        xcorr[i + 1] = sum[1];
        xcorr[i + 2] = sum[2];
        xcorr[i + 3] = sum[3];
        i += 4;
    }

    for j in i..max_pitch {
        xcorr[j] = inner_prod_avx(x, &y[j..], len);
    }
}

/// # Safety
///
/// `x` must hold at least `len` samples and `y` at least `len + 3`, and the
/// CPU must have the features this function is compiled for. The loads read
/// up to `y[len + 2]`.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx,fma")]
unsafe fn xcorr_kernel_avx(x: &[f32], y: &[f32], sum: &mut [f32; 4], len: usize) {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;

    let mut xsum1 = _mm_loadu_ps(sum.as_ptr());
    let mut xsum2 = _mm_setzero_ps();

    let mut j = 0;

    while j + 8 <= len {
        let x0 = _mm_loadu_ps(x.as_ptr().add(j));
        let yj = _mm_loadu_ps(y.as_ptr().add(j));
        let y3 = _mm_loadu_ps(y.as_ptr().add(j + 3));

        xsum1 = _mm_fmadd_ps(_mm_shuffle_ps(x0, x0, 0x00), yj, xsum1);
        xsum2 = _mm_fmadd_ps(
            _mm_shuffle_ps(x0, x0, 0x55),
            _mm_shuffle_ps(yj, y3, 0x49),
            xsum2,
        );
        xsum1 = _mm_fmadd_ps(
            _mm_shuffle_ps(x0, x0, 0xaa),
            _mm_shuffle_ps(yj, y3, 0x9e),
            xsum1,
        );
        xsum2 = _mm_fmadd_ps(_mm_shuffle_ps(x0, x0, 0xff), y3, xsum2);

        let x1 = _mm_loadu_ps(x.as_ptr().add(j + 4));
        let yj4 = _mm_loadu_ps(y.as_ptr().add(j + 4));
        let y7 = _mm_loadu_ps(y.as_ptr().add(j + 7));

        xsum1 = _mm_fmadd_ps(_mm_shuffle_ps(x1, x1, 0x00), yj4, xsum1);
        xsum2 = _mm_fmadd_ps(
            _mm_shuffle_ps(x1, x1, 0x55),
            _mm_shuffle_ps(yj4, y7, 0x49),
            xsum2,
        );
        xsum1 = _mm_fmadd_ps(
            _mm_shuffle_ps(x1, x1, 0xaa),
            _mm_shuffle_ps(yj4, y7, 0x9e),
            xsum1,
        );
        xsum2 = _mm_fmadd_ps(_mm_shuffle_ps(x1, x1, 0xff), y7, xsum2);

        j += 8;
    }

    if j + 4 <= len {
        let x0 = _mm_loadu_ps(x.as_ptr().add(j));
        let yj = _mm_loadu_ps(y.as_ptr().add(j));
        let y3 = _mm_loadu_ps(y.as_ptr().add(j + 3));

        xsum1 = _mm_fmadd_ps(_mm_shuffle_ps(x0, x0, 0x00), yj, xsum1);
        xsum2 = _mm_fmadd_ps(
            _mm_shuffle_ps(x0, x0, 0x55),
            _mm_shuffle_ps(yj, y3, 0x49),
            xsum2,
        );
        xsum1 = _mm_fmadd_ps(
            _mm_shuffle_ps(x0, x0, 0xaa),
            _mm_shuffle_ps(yj, y3, 0x9e),
            xsum1,
        );
        xsum2 = _mm_fmadd_ps(_mm_shuffle_ps(x0, x0, 0xff), y3, xsum2);

        j += 4;
    }

    if j < len {
        xsum1 = _mm_fmadd_ps(
            _mm_set1_ps(*x.as_ptr().add(j)),
            _mm_loadu_ps(y.as_ptr().add(j)),
            xsum1,
        );
        j += 1;
        if j < len {
            xsum2 = _mm_fmadd_ps(
                _mm_set1_ps(*x.as_ptr().add(j)),
                _mm_loadu_ps(y.as_ptr().add(j)),
                xsum2,
            );
            j += 1;
            if j < len {
                xsum1 = _mm_fmadd_ps(
                    _mm_set1_ps(*x.as_ptr().add(j)),
                    _mm_loadu_ps(y.as_ptr().add(j)),
                    xsum1,
                );
            }
        }
    }

    _mm_storeu_ps(sum.as_mut_ptr(), _mm_add_ps(xsum1, xsum2));
}

pub fn pitch_downsample(x: &[&[f32]], x_lp: &mut [f32], len: usize, c: usize, factor: usize) {
    let offset = factor / 2;

    if x_lp.len() < len {
        return;
    }

    // Outputs below `full` have all three inputs in every channel. For a
    // decimation by two of one or two channels they are computed without a
    // test per sample. The encoder passes channels of exactly `2 * len`
    // samples, so that is every output.
    let mut full = 1;
    if factor == 2 && (c == 1 || c == 2) {
        let shortest = x[..c].iter().map(|x_k| x_k.len()).min().unwrap_or(0);
        // 2 * i + 1 < shortest is the same as i < shortest / 2.
        let avail = len.min(shortest / 2);
        if avail >= 2 {
            let out = &mut x_lp[..avail];
            if c == 1 {
                decimate2([x[0]], out);
            } else {
                decimate2([x[0], x[1]], out);
            }
            full = avail;
        }
    }

    // The general case, and the outputs near the end of a short channel. A
    // channel that lacks an input contributes nothing to that output.
    for i in full..len {
        let mut val = 0.0f32;
        for k in 0..c {
            let x_k = x[k];

            let idx_m = factor * i - offset;
            let idx_p = factor * i + offset;
            let idx_c = factor * i;

            if idx_p < x_k.len() {
                val += 0.25 * x_k[idx_m] + 0.25 * x_k[idx_p] + 0.5 * x_k[idx_c];
            }
        }
        x_lp[i] = val;
    }

    pitch_downsample_boundary(x, x_lp, c, offset);

    flatten_spectrum(&mut x_lp[..len]);
}

/// Flattens the spectrum of the decimated signal: the second half of
/// libopus's pitch_downsample (celt/pitch.c).
///
/// A fourth-order LPC analysis gives the spectral envelope, and a filter
/// built from it removes most of that envelope. The pitch search then
/// compares the fine structure of the signal instead of following its
/// strongest formant.
fn flatten_spectrum(x_lp: &mut [f32]) {
    let len = x_lp.len();
    if len < 5 {
        // The autocorrelation needs more samples than lags. libopus is
        // never called with so few.
        return;
    }
    let mut ac = [0.0f32; 5];
    autocorr(x_lp, &mut ac, None, 0, 4, len);

    // "Noise floor -40 dB"
    ac[0] *= 1.0001;
    // "Lag windowing"
    for (i, a) in ac.iter_mut().enumerate().skip(1) {
        let f = 0.008 * i as f32;
        *a -= *a * f * f;
    }

    let mut coef = [0.0f32; 4];
    lpc(&mut coef, &ac, 4);
    let mut tmp = 1.0f32;
    for v in coef.iter_mut() {
        tmp *= 0.9;
        *v *= tmp;
    }
    // "Add a zero"
    let c1 = 0.8f32;
    let lpc2 = [
        coef[0] + 0.8,
        coef[1] + c1 * coef[0],
        coef[2] + c1 * coef[1],
        coef[3] + c1 * coef[2],
        c1 * coef[3],
    ];
    celt_fir5(x_lp, &lpc2);
}

/// Filter with five feed-forward taps, in place (libopus: celt_fir5):
/// `x[i] += num[0] * x[i - 1] + .. + num[4] * x[i - 5]`, with the samples
/// before `x[0]` taken as zero and the old values of `x` on the right.
fn celt_fir5(x: &mut [f32], num: &[f32; 5]) {
    let mut mem = [0.0f32; 5];
    for v in x.iter_mut() {
        let mut sum = *v;
        sum += num[0] * mem[0];
        sum += num[1] * mem[1];
        sum += num[2] * mem[2];
        sum += num[3] * mem[3];
        sum += num[4] * mem[4];
        mem[4] = mem[3];
        mem[3] = mem[2];
        mem[2] = mem[1];
        mem[1] = mem[0];
        mem[0] = *v;
        *v = sum;
    }
}

/// Decimation by two with the low-pass filter (0.25, 0.5, 0.25), summed over
/// `C` channels: for `i` from 1 to `out.len() - 1`
///
/// `out[i] = sum over channels of (0.25 * x[2i - 1] + 0.25 * x[2i + 1]) + 0.5 * x[2i]`
///
/// The sum starts at 0.0 and adds the channels in order. `out[0]` is not
/// written. Every channel must hold at least `2 * out.len()` samples.
#[inline(always)]
fn decimate2<const C: usize>(x: [&[f32]; C], out: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    decimate2_neon(x, out);
    #[cfg(not(target_arch = "aarch64"))]
    decimate2_scalar(x, out);
}

/// Portable version of `decimate2`. It is also what `decimate2_neon` uses for
/// the outputs that do not fill a vector, and its reference in the tests.
#[inline(always)]
fn decimate2_scalar<const C: usize>(x: [&[f32]; C], out: &mut [f32]) {
    let n = out.len();
    if n < 2 {
        return;
    }
    for (k, x_k) in x.iter().enumerate() {
        // Pairs (x[2i - 1], x[2i]) and (x[2i], x[2i + 1]) for i = 1..n.
        let (before, _) = x_k[1..2 * n - 1].as_chunks::<2>();
        let (after, _) = x_k[2..2 * n].as_chunks::<2>();
        for ((o, b), a) in out[1..].iter_mut().zip(before).zip(after) {
            let val = 0.25 * b[0] + 0.25 * a[1] + 0.5 * a[0];
            *o = if k == 0 { 0.0 + val } else { *o + val };
        }
    }
}

/// NEON version of `decimate2`: four outputs per pass, with the same
/// operations per output as `decimate2_scalar` (no fused multiply-add).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn decimate2_neon<const C: usize>(x: [&[f32]; C], out: &mut [f32]) {
    use core::arch::aarch64::*;

    let n = out.len();
    // Checked here once, so that the loads below need no check.
    assert!(x.iter().all(|x_k| x_k.len() >= 2 * n));
    let mut i = 1;
    if n >= 5 {
        // SAFETY: a pass at output `i` runs only when i + 4 <= n. It reads
        // x_k[2i..2i + 8], which ends at 2n - 1 at most, and writes
        // out[i..i + 4]. Before the loop x_k[1] is read; n >= 5 means the
        // channel has at least 10 samples.
        unsafe {
            let quarter = vdupq_n_f32(0.25);
            let half = vdupq_n_f32(0.5);
            // 0.25 * x[2i + 1] of the pass before. Its last lane is
            // 0.25 * x[2i - 1] for the first output of this pass.
            let mut q_prev = [vdupq_n_f32(0.0); C];
            for (q, x_k) in q_prev.iter_mut().zip(&x) {
                *q = vmulq_f32(vld1q_dup_f32(x_k.as_ptr().add(1)), quarter);
            }
            while i + 4 <= n {
                let mut val = vdupq_n_f32(0.0);
                for (q, x_k) in q_prev.iter_mut().zip(&x) {
                    // .0 = x[2i], x[2i + 2], ..; .1 = x[2i + 1], x[2i + 3], ..
                    let pair = vld2q_f32(x_k.as_ptr().add(2 * i));
                    let q_after = vmulq_f32(pair.1, quarter);
                    // 0.25 * (x[2i - 1], x[2i + 1], x[2i + 3], x[2i + 5])
                    let q_before = vextq_f32::<3>(*q, q_after);
                    let centre = vmulq_f32(pair.0, half);
                    val = vaddq_f32(val, vaddq_f32(vaddq_f32(q_before, q_after), centre));
                    *q = q_after;
                }
                vst1q_f32(out.as_mut_ptr().add(i), val);
                i += 4;
            }
        }
    }
    // The last outputs: the same formula on the samples from 2 * (i - 1) on,
    // where output `i` is the second one.
    decimate2_scalar(x.map(|x_k| &x_k[2 * (i - 1)..]), &mut out[i - 1..]);
}

#[inline]
fn pitch_downsample_boundary(x: &[&[f32]], x_lp: &mut [f32], c: usize, offset: usize) {
    {
        let mut val = 0.0f32;
        for k in 0..c {
            let x_k = x[k];

            let idx_offset = offset;
            let idx_0 = 0;
            if idx_offset < x_k.len() {
                val += 0.25 * x_k[idx_offset] + 0.5 * x_k[idx_0];
            }
        }
        x_lp[0] = val;
    }
}

/// One plus the energy of `y`: the start value of the window energy in
/// `find_best_pitch`.
///
/// Each target adds in its own order, and that order has to stay:
/// - aarch64: the squares in four lanes with fused multiply-add, then
///   `1.0 + ((s0 + s1) + (s2 + s3))`, then the last `len % 4` squares one by
///   one with a separate multiply and add.
/// - x86 with AVX: see `window_energy_plus_one_avx`.
/// - everything else: starts at 1.0 and adds one square after the other.
#[inline(always)]
fn window_energy_plus_one(y: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        use core::arch::aarch64::*;
        let (quads, tail) = y.as_chunks::<4>();
        // SAFETY: NEON is always present on aarch64. Each load reads one
        // `[f32; 4]` of the slice.
        let lanes = unsafe {
            let mut acc = vdupq_n_f32(0.0);
            for q in quads {
                let v = vld1q_f32(q.as_ptr());
                acc = vfmaq_f32(acc, v, v);
            }
            vaddvq_f32(acc)
        };
        let mut sum = 1.0f32 + lanes;
        for &v in tail {
            sum += v * v;
        }
        sum
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if crate::compat::x86_has_avx() {
            // SAFETY: the processor has AVX (checked on the line above).
            return unsafe { window_energy_plus_one_avx(y) };
        }
        let mut sum = 1.0f32;
        for &v in y {
            sum += v * v;
        }
        sum
    }
}

/// `window_energy_plus_one` for x86 with AVX: two registers of eight lanes
/// take turns over blocks of eight samples, with a separate multiply and add.
/// Their lanes are added pairwise, 1.0 is added to that, and the last
/// `len % 8` squares follow one by one.
///
/// This code used to sit inside `find_best_pitch`, which does not enable AVX.
/// Every intrinsic was then a call to a separate function. Here it is
/// compiled with AVX enabled, so the intrinsics become single instructions.
/// The operations and their order are the same.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx")]
fn window_energy_plus_one_avx(y: &[f32]) -> f32 {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;

    let len = y.len();
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut j = 0;
    while j + 16 <= len {
        // SAFETY: j + 16 <= len, so both loads of eight samples are inside `y`.
        let (y0, y1) = unsafe {
            (
                _mm256_loadu_ps(y.as_ptr().add(j)),
                _mm256_loadu_ps(y.as_ptr().add(j + 8)),
            )
        };
        acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(y0, y0));
        acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(y1, y1));
        j += 16;
    }
    while j + 8 <= len {
        // SAFETY: j + 8 <= len.
        let y0 = unsafe { _mm256_loadu_ps(y.as_ptr().add(j)) };
        acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(y0, y0));
        j += 8;
    }
    let acc = _mm256_add_ps(acc0, acc1);
    let hi = _mm256_extractf128_ps(acc, 1);
    let lo = _mm256_castps256_ps128(acc);
    let sum4 = _mm_add_ps(lo, hi);
    let tmp = _mm_movehl_ps(sum4, sum4);
    let sum2 = _mm_add_ps(sum4, tmp);
    let tmp2 = _mm_shuffle_ps(sum2, sum2, 0x55);
    let mut sum = 1.0f32 + _mm_cvtss_f32(_mm_add_ss(sum2, tmp2));
    while j < len {
        sum += y[j] * y[j];
        j += 1;
    }
    sum
}

#[inline(always)]
fn find_best_pitch(
    xcorr: &[f32],
    y: &[f32],
    len: usize,
    max_pitch: usize,
    best_pitch: &mut [usize; 2],
) {
    let mut best_num = [-1.0f32, -1.0f32];
    let mut best_den = [0.0f32, 0.0f32];

    best_pitch[0] = 0;
    best_pitch[1] = 1;

    // `syy` is the energy of y[i..i + len] plus one. Each step adds the sample
    // that enters the window and removes the one that leaves it.
    let mut syy = window_energy_plus_one(&y[..len]);
    let leaving = &y[..max_pitch];
    let entering = &y[len..len + max_pitch];
    for (i, ((&xc, &old), &new)) in xcorr[..max_pitch]
        .iter()
        .zip(leaving)
        .zip(entering)
        .enumerate()
    {
        if xc > 0.0 {
            // libopus scales the correlation before it squares it ("this
            // should avoid both underflows and overflows (inf) when squaring
            // xcorr16", celt/pitch.c, find_best_pitch). The signal is at the
            // scale of 16-bit samples, so a correlation reaches 5e11 for a
            // full-scale frame. Without the scaling its square times a
            // window energy no longer fits an f32 from about four times
            // full scale on (float input may exceed full scale). The
            // products then became infinite and the comparisons below false.
            let xc16 = xc * 1e-12;
            let num = xc16 * xc16;
            if num * best_den[1] > best_num[1] * syy {
                if num * best_den[0] > best_num[0] * syy {
                    best_num[1] = best_num[0];
                    best_den[1] = best_den[0];
                    best_pitch[1] = best_pitch[0];
                    best_num[0] = num;
                    best_den[0] = syy;
                    best_pitch[0] = i;
                } else {
                    best_num[1] = num;
                    best_den[1] = syy;
                    best_pitch[1] = i;
                }
            }
        }
        syy += new * new - old * old;
        if syy < 1.0 {
            // The lower limit is almost never reached. Marking it as the rare
            // case keeps it a predicted branch. Otherwise the compiler turns
            // it into a compare and select, which every step of the running
            // sum would have to wait for (8 cycles per step instead of 3).
            core::hint::cold_path();
            syy = 1.0;
        }
    }
}

/// Second step of `pitch_search`: the correlation at full resolution, for the
/// lags that are at most two away from twice a coarse candidate. Every other
/// lag gets -1, and no lag gets less than -1.
///
/// There are at most ten such lags. Their inner products are computed
/// together (`inner_prod_many`).
fn fine_xcorr(x_lp: &[f32], y: &[f32], len: usize, best_pitch: &[usize; 2], xcorr: &mut [f32]) {
    let max_pitch = xcorr.len();
    xcorr.fill(-1.0);

    let near = |i: usize, coarse: usize| (i as i32 - 2 * coarse as i32).abs() <= 2;
    let mut lags = [0usize; 10];
    let mut count = 0;
    for (c, &coarse) in best_pitch.iter().enumerate() {
        let first = (2 * coarse).saturating_sub(2);
        let end = (2 * coarse + 3).min(max_pitch);
        for i in first..end {
            // The windows of the two candidates can overlap.
            if c == 1 && near(i, best_pitch[0]) {
                continue;
            }
            lags[count] = i;
            count += 1;
        }
    }

    let mut sums = [0.0f32; 10];
    inner_prod_many(x_lp, y, &lags[..count], len, &mut sums[..count]);
    for (&i, &sum) in lags[..count].iter().zip(&sums) {
        xcorr[i] = if sum < -1.0 { -1.0 } else { sum };
    }
}

/// `dst[j] = src[2 * j]` for every `j < dst.len()`.
///
/// Needs `src.len() >= 2 * dst.len() - 1`. Written over pairs of samples, so
/// that the compiler can turn it into vector code.
#[inline(always)]
fn take_even(src: &[f32], dst: &mut [f32]) {
    let n = dst.len();
    if n == 0 {
        return;
    }
    let (pairs, last) = src[..2 * n - 1].as_chunks::<2>();
    for (d, pair) in dst.iter_mut().zip(pairs) {
        *d = pair[0];
    }
    dst[n - 1] = last[0];
}

/// Pitch search: finds the lag at which `y` is most similar to `x_lp`.
///
/// `x_lp` and `y` are signals that were decimated by two. `len` and
/// `max_pitch` count samples before that decimation, and so does the lag
/// that is returned.
///
/// The search first decimates both signals by two again and correlates them
/// at every lag (the coarse search, `pitch_xcorr`). It then correlates
/// `x_lp` and `y` themselves, only around the two best coarse lags.
pub fn pitch_search(x_lp: &[f32], y: &[f32], len: usize, max_pitch: usize) -> usize {
    let mut best_pitch = [0, 0];

    let max_pitch = max_pitch >> 1;
    let len = len >> 1;
    let lag = len + max_pitch;

    // The buffers hold up to 1024 values each, but only the part that is
    // used is cleared. Longer inputs panic in `resize`.
    let mut x_lp4: FixedVec<f32, PITCH_BUF_MAX> = FixedVec::new();
    let mut y_lp4: FixedVec<f32, PITCH_BUF_MAX> = FixedVec::new();
    let mut xcorr: FixedVec<f32, PITCH_BUF_MAX> = FixedVec::new();
    x_lp4.resize(len >> 1, 0.0);
    y_lp4.resize(lag >> 1, 0.0);
    xcorr.resize(max_pitch, 0.0);

    take_even(x_lp, &mut x_lp4);
    take_even(y, &mut y_lp4);

    pitch_xcorr(&x_lp4, &y_lp4, &mut xcorr, len >> 1, max_pitch >> 1);

    find_best_pitch(&xcorr, &y_lp4, len >> 1, max_pitch >> 1, &mut best_pitch);

    fine_xcorr(x_lp, y, len, &best_pitch, &mut xcorr);

    find_best_pitch(&xcorr, y, len, max_pitch, &mut best_pitch);

    let mut offset = 0;
    if best_pitch[0] > 0 && best_pitch[0] < max_pitch - 1 {
        let a = xcorr[best_pitch[0] - 1];
        let b = xcorr[best_pitch[0]];
        let c = xcorr[best_pitch[0] + 1];
        if (c - a) > 0.7 * (b - a) {
            offset = 1;
        } else if (a - c) > 0.7 * (b - c) {
            offset = -1;
        }
    }

    ((2 * best_pitch[0]) as isize).wrapping_sub(offset as isize) as usize
}

fn compute_pitch_gain(xy: f32, xx: f32, yy: f32) -> f32 {
    if xy <= 0.0 || xx <= 0.0 || yy <= 0.0 {
        return 0.0;
    }
    xy / (1.0 + xx * yy).sqrt()
}

static SECOND_CHECK: [usize; 16] = [0, 0, 3, 2, 3, 2, 5, 2, 3, 2, 3, 2, 5, 2, 3, 2];

/// Energy of `x[..n]`, one product after the other. aarch64 does not use it:
/// there `doubling_products` computes the energy in the four-lane order,
/// together with the other inner products.
#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
fn sum_squares(x: &[f32], n: usize) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..n {
        sum += x[i] * x[i];
    }
    sum
}

/// `remove_doubling` tries the periods `t0 / k` for `k` from 2 to 15.
const MAX_DOUBLING_CANDS: usize = 14;

/// The inner products `remove_doubling` needs before it decides anything.
///
/// With `target = x[max_period..]` and `n` samples per product, it returns
/// `(xx, xy)`: the energy of `target` and its product with the signal `t0`
/// samples earlier. For every candidate `(t1, t1b)` it stores in `pair_xy`
/// the products of `target` with the signal `t1` and `t1b` samples earlier.
///
/// On aarch64 all of them (up to 30) are computed eight at a time. Each has
/// the same bits as the separate call it replaces. Other targets make the
/// same calls as before, because their kernels for one product and for a
/// pair of products round differently.
fn doubling_products(
    x: &[f32],
    max_period: usize,
    n: usize,
    t0: usize,
    cands: &[(usize, usize)],
    pair_xy: &mut [(f32, f32); MAX_DOUBLING_CANDS],
) -> (f32, f32) {
    let x_target = &x[max_period..];
    #[cfg(target_arch = "aarch64")]
    {
        const TOTAL: usize = 2 + 2 * MAX_DOUBLING_CANDS;
        let mut offsets = [0usize; TOTAL];
        offsets[0] = max_period;
        offsets[1] = max_period - t0;
        for (o, &(t1, t1b)) in offsets[2..].as_chunks_mut::<2>().0.iter_mut().zip(cands) {
            *o = [max_period - t1, max_period - t1b];
        }
        let count = 2 + 2 * cands.len();
        let mut sums = [0.0f32; TOTAL];
        inner_prod_many(x_target, x, &offsets[..count], n, &mut sums[..count]);
        for (p, s) in pair_xy.iter_mut().zip(sums[2..count].as_chunks::<2>().0) {
            *p = (s[0], s[1]);
        }
        (sums[0], sums[1])
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let xx = sum_squares(x_target, n);
        let xy = inner_prod(x_target, &x[max_period - t0..], n);
        for (p, &(t1, t1b)) in pair_xy.iter_mut().zip(cands) {
            *p = dual_inner_prod(x_target, &x[max_period - t1..], &x[max_period - t1b..], n);
        }
        (xx, xy)
    }
}

pub fn remove_doubling(
    x: &[f32],
    mut max_period: usize,
    mut min_period: usize,
    mut n: usize,
    t0_ptr: &mut usize,
    mut prev_period: usize,
    prev_gain: f32,
) -> f32 {
    let min_period0 = min_period;
    max_period /= 2;
    min_period /= 2;
    *t0_ptr /= 2;
    prev_period /= 2;
    n /= 2;

    let x_target = &x[max_period..];

    if *t0_ptr >= max_period {
        *t0_ptr = max_period - 1;
    }

    let mut t = *t0_ptr;
    let t0 = *t0_ptr;

    // The shorter periods the loop below tries: `t1` is about `t0 / k`, and
    // `t1b` is a second period that is checked together with it. Both depend
    // on `t0` and `k` only. So all inner products of the loop can be computed
    // before its decisions, several at a time.
    let mut cands = [(0usize, 0usize); MAX_DOUBLING_CANDS];
    let mut n_cands = 0;
    for k in 2..=15 {
        let t1 = (2 * t0 + k) / (2 * k);
        if t1 < min_period {
            break;
        }
        let t1b = if k == 2 {
            if t1 + t0 > max_period { t0 } else { t0 + t1 }
        } else {
            (2 * SECOND_CHECK[k] * t0 + k) / (2 * k)
        };
        cands[n_cands] = (t1, t1b);
        n_cands += 1;
    }
    let cands = &cands[..n_cands];

    let mut pair_xy = [(0.0f32, 0.0f32); MAX_DOUBLING_CANDS];
    let (xx, xy) = doubling_products(x, max_period, n, t0, cands, &mut pair_xy);

    // yy_lookup[i] is the energy of the `n` samples that start `i` samples
    // before the target. Each entry follows from the one before it, so the
    // table is filled in order. It is only read at `t0` and at the periods
    // of the candidates, so it ends at the largest of them.
    let last = cands.iter().fold(t0, |m, &(t1, t1b)| m.max(t1).max(t1b));
    const MAX_YY_SIZE: usize = 1024;
    // `new` plus `resize` clears the used entries only and builds the table
    // in place. (`FixedVec::from_value` would build it and then copy all 4 KB.)
    let mut yy_lookup: FixedVec<f32, MAX_YY_SIZE> = FixedVec::new();
    yy_lookup.resize(last + 1, 0.0);
    yy_lookup[0] = xx;
    let mut yy_curr = xx;
    // Entry `i` adds x[max_period - i] and removes x[max_period + n - i].
    let entering = x[max_period - last..max_period].iter().rev();
    let leaving = x[max_period + n - last..max_period + n].iter().rev();
    for ((yy, &new), &old) in yy_lookup[1..].iter_mut().zip(entering).zip(leaving) {
        yy_curr = yy_curr + new * new - old * old;
        if yy_curr < 0.0 {
            // Rare: see the same limit in `find_best_pitch`.
            core::hint::cold_path();
            yy_curr = 0.0;
        }
        *yy = yy_curr;
    }

    let mut best_xy = xy;
    let mut best_yy = yy_lookup[t0];
    let mut g = compute_pitch_gain(best_xy, xx, best_yy);
    let g0 = g;

    for (i, (&(t1, t1b), &(xy_a, xy_b))) in cands.iter().zip(&pair_xy).enumerate() {
        let k = i + 2;
        let xy_new = 0.5 * (xy_a + xy_b);
        let yy_new = 0.5 * (yy_lookup[t1] + yy_lookup[t1b]);
        let g1 = compute_pitch_gain(xy_new, xx, yy_new);

        let mut cont = 0.0f32;
        if (t1 as i32 - prev_period as i32).abs() <= 1 {
            cont = prev_gain;
        } else if (t1 as i32 - prev_period as i32).abs() <= 2 && 5 * k * k < t0 {
            cont = 0.5 * prev_gain;
        }

        let mut thresh = (0.7 * g0 - cont).max(0.3);
        if t1 < 3 * min_period {
            thresh = (0.85 * g0 - cont).max(0.4);
        } else if t1 < 2 * min_period {
            thresh = (0.9 * g0 - cont).max(0.5);
        }

        if g1 > thresh {
            best_xy = xy_new;
            best_yy = yy_new;
            t = t1;
            g = g1;
        }
    }

    best_xy = best_xy.max(0.0);
    let pg = if best_yy <= best_xy {
        1.0f32
    } else {
        best_xy / (best_yy + 1.0)
    };

    // The correlation at the periods t - 1, t and t + 1.
    let offsets: [usize; 3] = core::array::from_fn(|k_idx| {
        let lag = (t as i32 + k_idx as i32 - 1) as usize;
        max_period - lag
    });
    let mut xcorr_res = [0.0f32; 3];
    inner_prod_many(x_target, x, &offsets, n, &mut xcorr_res);

    let mut offset = 0;
    if (xcorr_res[2] - xcorr_res[0]) > 0.7 * (xcorr_res[1] - xcorr_res[0]) {
        offset = 1;
    } else if (xcorr_res[0] - xcorr_res[2]) > 0.7 * (xcorr_res[1] - xcorr_res[2]) {
        offset = -1;
    }

    let pg = pg.min(g);
    *t0_ptr = (2 * t as i32 + offset) as usize;
    if *t0_ptr < min_period0 {
        *t0_ptr = min_period0;
    }

    pg
}

#[cfg(test)]
pub(crate) mod tests {
    #[cfg(not(feature = "std"))]
    use std::vec::Vec;
    use super::*;

    /// Small deterministic generator, so that a failure can be repeated.
    pub(crate) struct Rng(u64);

    impl Rng {
        pub(crate) fn new(seed: u64) -> Self {
            Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
        }

        pub(crate) fn next_u32(&mut self) -> u32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 33) as u32
        }

        /// Uniform in [-1, 1).
        pub(crate) fn unit(&mut self) -> f32 {
            self.next_u32() as f32 / (1u32 << 30) as f32 - 1.0
        }
    }

    /// Kinds of test input, from ordinary samples to values that overflow.
    #[derive(Clone, Copy, Debug)]
    pub(crate) enum Kind {
        /// Samples in the range the encoder works with.
        Audio,
        /// Random signs and magnitudes between 1e-30 and 1e15.
        WideRange,
        /// Zeros of both signs, subnormal numbers and the smallest normal one.
        Tiny,
        /// The largest finite values: products overflow to infinity.
        Huge,
        /// A few infinities and not-a-number values among ordinary samples.
        NonFinite,
    }

    pub(crate) const KINDS: [Kind; 5] = [
        Kind::Audio,
        Kind::WideRange,
        Kind::Tiny,
        Kind::Huge,
        Kind::NonFinite,
    ];

    pub(crate) fn samples(rng: &mut Rng, n: usize, kind: Kind) -> Vec<f32> {
        (0..n)
            .map(|_| match kind {
                Kind::Audio => 32768.0 * rng.unit(),
                Kind::WideRange => {
                    let exp = (rng.next_u32() % 46) as i32 - 30;
                    rng.unit() * 10f32.powi(exp)
                }
                Kind::Tiny => match rng.next_u32() % 5 {
                    0 => 0.0,
                    1 => -0.0,
                    2 => f32::MIN_POSITIVE,
                    3 => -f32::from_bits(1 + rng.next_u32() % 1000),
                    _ => f32::from_bits(1 + rng.next_u32() % 0x0080_0000),
                },
                Kind::Huge => match rng.next_u32() % 4 {
                    0 => f32::MAX,
                    1 => f32::MIN,
                    _ => 1e30 * rng.unit(),
                },
                Kind::NonFinite => match rng.next_u32() % 16 {
                    0 => f32::INFINITY,
                    1 => f32::NEG_INFINITY,
                    2 => f32::NAN,
                    _ => 100.0 * rng.unit(),
                },
            })
            .collect()
    }

    /// Bit-for-bit comparison. Two not-a-number values count as equal: which
    /// payload an operation returns is not something the codec relies on.
    pub(crate) fn same_bits(a: f32, b: f32) -> bool {
        a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
    }

    #[track_caller]
    pub(crate) fn assert_same(a: &[f32], b: &[f32], what: &str) {
        assert_eq!(a.len(), b.len(), "{what}: lengths differ");
        for (i, (&p, &q)) in a.iter().zip(b).enumerate() {
            assert!(
                same_bits(p, q),
                "{what}: element {i} differs: {p:e} ({:#010x}) and {q:e} ({:#010x})",
                p.to_bits(),
                q.to_bits()
            );
        }
    }

    /// One lag in the by-sample order, written as plainly as possible.
    fn lag_by_sample<const FUSED: bool>(x: &[f32], y: &[f32], len: usize) -> f32 {
        let mut acc = 0.0f32;
        for j in 0..len {
            acc = if FUSED {
                x[j].mul_add(y[j], acc)
            } else {
                acc + x[j] * y[j]
            };
        }
        acc
    }

    /// One lag in the four-lane order of `inner_prod_neon`, written as plainly
    /// as possible: fused additions in the lanes, then (s0 + s1) + (s2 + s3),
    /// then the last `n % 4` products with a separate multiply and add.
    fn lag_four_lanes(x: &[f32], y: &[f32], n: usize) -> f32 {
        let mut s = [0.0f32; 4];
        let whole = n / 4 * 4;
        for j in 0..whole {
            s[j % 4] = x[j].mul_add(y[j], s[j % 4]);
        }
        let mut sum = (s[0] + s[1]) + (s[2] + s[3]);
        for j in whole..n {
            sum += x[j] * y[j];
        }
        sum
    }

    #[test]
    fn xcorr_by_sample_scalar_matches_one_lag_at_a_time() {
        let mut rng = Rng::new(1);
        for kind in KINDS {
            for len in (0..=20).chain([31, 32, 33, 61, 240]) {
                for lags in (0..=20).chain([31, 32, 33, 40, 61]) {
                    let x = samples(&mut rng, len, kind);
                    let y = samples(&mut rng, (len + lags).saturating_sub(1), kind);
                    let what = format!("{kind:?} len {len} lags {lags}");

                    let mut got = vec![f32::NAN; lags];
                    xcorr_by_sample_scalar::<false>(&x, &y, &mut got, len);
                    let want: Vec<f32> = (0..lags)
                        .map(|k| lag_by_sample::<false>(&x, &y[k..], len))
                        .collect();
                    assert_same(&got, &want, &what);

                    let mut got = vec![f32::NAN; lags];
                    xcorr_by_sample_scalar::<true>(&x, &y, &mut got, len);
                    let want: Vec<f32> = (0..lags)
                        .map(|k| lag_by_sample::<true>(&x, &y[k..], len))
                        .collect();
                    assert_same(&got, &want, &what);
                }
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    mod neon {
        use super::*;

        /// The kernel `pitch_xcorr` used before `xcorr_by_sample_neon`: four
        /// lags per pass with one running sum. Kept here unchanged, as proof
        /// that the new kernel gives the same bits.
        #[allow(unsafe_op_in_unsafe_fn)]
        unsafe fn xcorr_kernel_neon_old(x: &[f32], y: &[f32], sum: &mut [f32; 4], mut len: usize) {
            use core::arch::aarch64::*;

            assert!(len >= 1 && x.len() >= len && y.len() >= len + 3);

            let mut summ = vdupq_n_f32(0.0);
            let mut xi = x.as_ptr();
            let mut yi = y.as_ptr();

            let mut yy = vld1q_f32(yi);

            while len > 8 {
                yi = yi.add(4);
                let yy1 = vld1q_f32(yi);
                yi = yi.add(4);
                let yy2 = vld1q_f32(yi);

                let xx0 = vld1q_f32(xi);
                xi = xi.add(4);
                let xx1 = vld1q_f32(xi);
                xi = xi.add(4);

                summ = vfmaq_lane_f32(summ, yy, vget_low_f32(xx0), 0);
                let yext = vextq_f32(yy, yy1, 1);
                summ = vfmaq_lane_f32(summ, yext, vget_low_f32(xx0), 1);
                let yext = vextq_f32(yy, yy1, 2);
                summ = vfmaq_lane_f32(summ, yext, vget_high_f32(xx0), 0);
                let yext = vextq_f32(yy, yy1, 3);
                summ = vfmaq_lane_f32(summ, yext, vget_high_f32(xx0), 1);

                summ = vfmaq_lane_f32(summ, yy1, vget_low_f32(xx1), 0);
                let yext = vextq_f32(yy1, yy2, 1);
                summ = vfmaq_lane_f32(summ, yext, vget_low_f32(xx1), 1);
                let yext = vextq_f32(yy1, yy2, 2);
                summ = vfmaq_lane_f32(summ, yext, vget_high_f32(xx1), 0);
                let yext = vextq_f32(yy1, yy2, 3);
                summ = vfmaq_lane_f32(summ, yext, vget_high_f32(xx1), 1);

                yy = yy2;
                len -= 8;
            }

            if len > 4 {
                yi = yi.add(4);
                let yy1 = vld1q_f32(yi);

                let xx0 = vld1q_f32(xi);
                xi = xi.add(4);

                summ = vfmaq_lane_f32(summ, yy, vget_low_f32(xx0), 0);
                let yext = vextq_f32(yy, yy1, 1);
                summ = vfmaq_lane_f32(summ, yext, vget_low_f32(xx0), 1);
                let yext = vextq_f32(yy, yy1, 2);
                summ = vfmaq_lane_f32(summ, yext, vget_high_f32(xx0), 0);
                let yext = vextq_f32(yy, yy1, 3);
                summ = vfmaq_lane_f32(summ, yext, vget_high_f32(xx0), 1);

                yy = yy1;
                len -= 4;
            }

            while len > 1 {
                let xx = vld1_dup_f32(xi);
                xi = xi.add(1);
                summ = vfmaq_lane_f32(summ, yy, xx, 0);
                yi = yi.add(1);
                yy = vld1q_f32(yi);
                len -= 1;
            }

            let xx = vld1_dup_f32(xi);
            summ = vfmaq_lane_f32(summ, yy, xx, 0);

            vst1q_f32(sum.as_mut_ptr(), summ);
        }

        /// `pitch_xcorr` as it was before the 32-lag kernel, for `len >= 1`.
        fn pitch_xcorr_old(x: &[f32], y: &[f32], len: usize, lags: usize) -> Vec<f32> {
            let mut out = vec![f32::NAN; lags];
            if lags < 32 {
                for (k, o) in out.iter_mut().enumerate() {
                    *o = unsafe { inner_prod_neon(x, &y[k..], len) };
                }
                return out;
            }
            let mut i = 0;
            while i + 4 <= lags {
                let mut sum = [0.0f32; 4];
                unsafe { xcorr_kernel_neon_old(x, &y[i..], &mut sum, len) };
                out[i..i + 4].copy_from_slice(&sum);
                i += 4;
            }
            for k in i..lags {
                out[k] = unsafe { inner_prod_neon(x, &y[k..], len) };
            }
            out
        }

        #[test]
        fn xcorr_by_sample_neon_matches_scalar() {
            let mut rng = Rng::new(2);
            for kind in KINDS {
                // Every tail length (len % 4), lengths 0 and 1, and the
                // lengths the encoder uses (30, 60, 120, 240).
                for len in (0..=40).chain([59, 60, 61, 62, 63, 120, 240, 241]) {
                    let x = samples(&mut rng, len, kind);
                    // Exactly as long as the 64 lags need.
                    let y = samples(&mut rng, len + 63, kind);
                    let what = format!("{kind:?} len {len}");

                    let mut got = [f32::NAN; 64];
                    xcorr_by_sample_neon::<16>(&x, &y, &mut got);
                    let mut want = [f32::NAN; 64];
                    xcorr_by_sample_scalar::<true>(&x, &y, &mut want, len);
                    assert_same(&got, &want, &what);

                    // 32 lags, with `y` cut to what they need.
                    let y = &y[..len + 31];
                    let mut got = [f32::NAN; 32];
                    xcorr_by_sample_neon::<8>(&x, y, &mut got);
                    assert_same(&got, &want[..32], &what);
                }
            }
        }

        #[test]
        fn window_energy_plus_one_follows_the_documented_order() {
            let mut rng = Rng::new(12);
            for kind in KINDS {
                for len in (0..=40).chain([60, 61, 240, 480]) {
                    let y = samples(&mut rng, len, kind);
                    let got = window_energy_plus_one(&y);

                    let mut s = [0.0f32; 4];
                    let whole = len / 4 * 4;
                    for j in 0..whole {
                        s[j % 4] = y[j].mul_add(y[j], s[j % 4]);
                    }
                    let mut want = 1.0 + ((s[0] + s[1]) + (s[2] + s[3]));
                    for j in whole..len {
                        want += y[j] * y[j];
                    }
                    assert!(same_bits(got, want), "{kind:?} len {len}: {got} and {want}");
                }
            }
        }

        #[test]
        fn decimate2_neon_matches_scalar() {
            let mut rng = Rng::new(10);
            for kind in KINDS {
                // Every tail length, the lengths below one vector pass, and
                // the lengths the encoder uses (572 to 992).
                for n in (0..=40).chain([572, 632, 752, 991, 992]) {
                    // Exactly as long as the outputs need.
                    let x0 = samples(&mut rng, 2 * n, kind);
                    let x1 = samples(&mut rng, 2 * n, kind);

                    let mut got = vec![f32::NAN; n];
                    decimate2_neon([&x0[..]], &mut got);
                    let mut want = vec![f32::NAN; n];
                    decimate2_scalar([&x0[..]], &mut want);
                    assert_same(&got, &want, &format!("{kind:?} n {n} mono"));

                    let mut got = vec![f32::NAN; n];
                    decimate2_neon([&x0[..], &x1[..]], &mut got);
                    let mut want = vec![f32::NAN; n];
                    decimate2_scalar([&x0[..], &x1[..]], &mut want);
                    assert_same(&got, &want, &format!("{kind:?} n {n} stereo"));
                }
            }
        }

        #[test]
        fn inner_prod_k_neon_matches_single_products() {
            let mut rng = Rng::new(5);
            for kind in KINDS {
                // Every tail length (n % 4), lengths 0 and 1, and the lengths
                // the encoder uses (60 to 480).
                for n in (0..=21).chain([60, 61, 62, 63, 120, 240, 480]) {
                    let x = samples(&mut rng, n, kind);
                    let ys: Vec<Vec<f32>> = (0..16).map(|_| samples(&mut rng, n, kind)).collect();
                    let what = format!("{kind:?} n {n}");

                    // The single kernel, and the order written out in plain code.
                    let single: Vec<f32> = ys
                        .iter()
                        .map(|y| unsafe { inner_prod_neon(&x, y, n) })
                        .collect();
                    let plain: Vec<f32> = ys.iter().map(|y| lag_four_lanes(&x, y, n)).collect();
                    assert_same(&single, &plain, &what);

                    // Every width `inner_prod_many` uses.
                    let p16: [&[f32]; 16] = core::array::from_fn(|k| &ys[k][..]);
                    let got16 = inner_prod_k_neon::<16>(&x, &p16);
                    assert_same(&got16, &single, &what);

                    let p12: [&[f32]; 12] = core::array::from_fn(|k| &ys[k][..]);
                    let got12 = inner_prod_k_neon::<12>(&x, &p12);
                    assert_same(&got12, &single[..12], &what);

                    let p8: [&[f32]; 8] = core::array::from_fn(|k| &ys[k][..]);
                    let got8 = inner_prod_k_neon::<8>(&x, &p8);
                    assert_same(&got8, &single[..8], &what);

                    let p4: [&[f32]; 4] = core::array::from_fn(|k| &ys[k][..]);
                    let got4 = inner_prod_k_neon::<4>(&x, &p4);
                    assert_same(&got4, &single[..4], &what);

                    // A pair of products is two single products.
                    let (a, b) = unsafe { dual_inner_prod_neon(&x, &ys[0], &ys[1], n) };
                    assert_same(&[a, b], &single[..2], &what);
                }
            }
        }

        #[test]
        fn inner_prod_many_matches_one_call_per_window() {
            let mut rng = Rng::new(6);
            for kind in KINDS {
                for n in [0, 1, 3, 4, 5, 62, 63, 240] {
                    let x = samples(&mut rng, n, kind);
                    let y = samples(&mut rng, n + 40, kind);
                    // Every number of windows from 0 to 30, in no particular order.
                    for count in 0..=30 {
                        let offsets: Vec<usize> =
                            (0..count).map(|_| rng.next_u32() as usize % 41).collect();
                        let mut got = vec![f32::NAN; count];
                        inner_prod_many(&x, &y, &offsets, n, &mut got);
                        let want: Vec<f32> = offsets
                            .iter()
                            .map(|&off| inner_prod(&x, &y[off..], n))
                            .collect();
                        assert_same(&got, &want, &format!("{kind:?} n {n} count {count}"));
                    }
                }
            }
        }

        #[test]
        fn pitch_xcorr_matches_the_old_kernels() {
            let mut rng = Rng::new(3);
            for kind in KINDS {
                for len in (1..=12).chain([30, 31, 60, 61, 62, 63, 120, 240]) {
                    for lags in (1..=70).chain([127, 128, 129, 244, 245, 246, 247, 489]) {
                        let x = samples(&mut rng, len, kind);
                        // Exactly as long as the lags need.
                        let y = samples(&mut rng, len + lags - 1, kind);
                        let mut got = vec![f32::NAN; lags];
                        pitch_xcorr(&x, &y, &mut got, len, lags);
                        let want = pitch_xcorr_old(&x, &y, len, lags);
                        assert_same(&got, &want, &format!("{kind:?} len {len} lags {lags}"));
                    }
                }
            }
        }

        #[test]
        fn pitch_xcorr_follows_the_documented_orders() {
            let mut rng = Rng::new(4);
            for kind in KINDS {
                for len in [1, 2, 3, 4, 5, 7, 8, 9, 30, 61, 240] {
                    for lags in [1, 2, 5, 25, 31, 32, 33, 34, 35, 36, 63, 64, 65, 66, 67, 244] {
                        let x = samples(&mut rng, len, kind);
                        let y = samples(&mut rng, len + lags - 1, kind);
                        let mut got = vec![f32::NAN; lags];
                        pitch_xcorr(&x, &y, &mut got, len, lags);
                        let by_sample = if lags >= 32 { lags & !3 } else { 0 };
                        let want: Vec<f32> = (0..lags)
                            .map(|k| {
                                if k < by_sample {
                                    lag_by_sample::<true>(&x, &y[k..], len)
                                } else {
                                    lag_four_lanes(&x, &y[k..], len)
                                }
                            })
                            .collect();
                        assert_same(&got, &want, &format!("{kind:?} len {len} lags {lags}"));
                    }
                }
            }
        }
    }

    /// A signal with a clear period, some noise, and a level that depends on
    /// `kind`: 0 is ordinary audio, 1 is silence, 2 is very quiet, 3 is noise
    /// only.
    fn periodic(rng: &mut Rng, n: usize, period: usize, kind: usize) -> Vec<f32> {
        let cycle: Vec<f32> = (0..period).map(|_| rng.unit()).collect();
        (0..n)
            .map(|i| {
                let v = 8000.0 * cycle[i % period] + 800.0 * rng.unit();
                match kind {
                    0 => v,
                    1 => 0.0,
                    2 => v * 1e-7,
                    _ => 3000.0 * rng.unit(),
                }
            })
            .collect()
    }

    /// `remove_doubling` as it was written before its inner products were
    /// computed together and its energy table was cut short: one call per
    /// product, the whole table. The public kernels are the same ones.
    fn remove_doubling_plain(
        x: &[f32],
        mut max_period: usize,
        mut min_period: usize,
        mut n: usize,
        t0_ptr: &mut usize,
        mut prev_period: usize,
        prev_gain: f32,
    ) -> f32 {
        let min_period0 = min_period;
        max_period /= 2;
        min_period /= 2;
        *t0_ptr /= 2;
        prev_period /= 2;
        n /= 2;

        let x_target = &x[max_period..];
        if *t0_ptr >= max_period {
            *t0_ptr = max_period - 1;
        }
        let mut t = *t0_ptr;
        let t0 = *t0_ptr;

        let mut yy_lookup = vec![0.0f32; max_period + 1];
        // The energy was a four-lane inner product on aarch64 and a plain
        // loop elsewhere.
        let xx = if cfg!(target_arch = "aarch64") {
            inner_prod(x_target, x_target, n)
        } else {
            x_target[..n].iter().fold(0.0f32, |s, &v| s + v * v)
        };
        let xy = inner_prod(x_target, &x[max_period - t0..], n);

        yy_lookup[0] = xx;
        let mut yy_curr = xx;
        for i in 1..=max_period {
            yy_curr = yy_curr + x[max_period - i] * x[max_period - i]
                - x[max_period + n - i] * x[max_period + n - i];
            if yy_curr < 0.0 {
                yy_curr = 0.0;
            }
            yy_lookup[i] = yy_curr;
        }

        let mut best_xy = xy;
        let mut best_yy = yy_lookup[t0];
        let mut g = compute_pitch_gain(best_xy, xx, best_yy);
        let g0 = g;

        for k in 2..=15 {
            let t1 = (2 * t0 + k) / (2 * k);
            if t1 < min_period {
                break;
            }
            let t1b = if k == 2 {
                if t1 + t0 > max_period { t0 } else { t0 + t1 }
            } else {
                (2 * SECOND_CHECK[k] * t0 + k) / (2 * k)
            };
            let (xy_a, xy_b) =
                dual_inner_prod(x_target, &x[max_period - t1..], &x[max_period - t1b..], n);
            let xy_new = 0.5 * (xy_a + xy_b);
            let yy_new = 0.5 * (yy_lookup[t1] + yy_lookup[t1b]);
            let g1 = compute_pitch_gain(xy_new, xx, yy_new);

            let mut cont = 0.0f32;
            if (t1 as i32 - prev_period as i32).abs() <= 1 {
                cont = prev_gain;
            } else if (t1 as i32 - prev_period as i32).abs() <= 2 && 5 * k * k < t0 {
                cont = 0.5 * prev_gain;
            }
            let mut thresh = (0.7 * g0 - cont).max(0.3);
            if t1 < 3 * min_period {
                thresh = (0.85 * g0 - cont).max(0.4);
            } else if t1 < 2 * min_period {
                thresh = (0.9 * g0 - cont).max(0.5);
            }
            if g1 > thresh {
                best_xy = xy_new;
                best_yy = yy_new;
                t = t1;
                g = g1;
            }
        }

        best_xy = best_xy.max(0.0);
        let pg = if best_yy <= best_xy {
            1.0f32
        } else {
            best_xy / (best_yy + 1.0)
        };

        let mut xcorr_res = [0.0f32; 3];
        for k_idx in 0..3 {
            let lag = (t as i32 + k_idx as i32 - 1) as usize;
            xcorr_res[k_idx] = inner_prod(x_target, &x[max_period - lag..], n);
        }
        let mut offset = 0;
        if (xcorr_res[2] - xcorr_res[0]) > 0.7 * (xcorr_res[1] - xcorr_res[0]) {
            offset = 1;
        } else if (xcorr_res[0] - xcorr_res[2]) > 0.7 * (xcorr_res[1] - xcorr_res[2]) {
            offset = -1;
        }

        let pg = pg.min(g);
        *t0_ptr = (2 * t as i32 + offset) as usize;
        if *t0_ptr < min_period0 {
            *t0_ptr = min_period0;
        }
        pg
    }

    #[test]
    fn remove_doubling_matches_the_plain_version() {
        let mut rng = Rng::new(7);
        let max_period = 1024;
        let min_period = 15;
        let mut picked_shorter = 0;
        for n in [120usize, 240, 480, 960] {
            for kind in 0..4 {
                for period in [16, 37, 100, 163, 300, 451] {
                    let x = periodic(&mut rng, (max_period + n) / 2, period, kind);
                    // Start periods from the smallest to past the largest,
                    // among them multiples of the true period.
                    for t0 in [
                        2,
                        15,
                        16,
                        31,
                        64,
                        2 * period,
                        4 * period,
                        6 * period + 1,
                        1021,
                        1022,
                        1023,
                        1024,
                        3000,
                    ] {
                        for (prev_period, prev_gain) in [(0, 0.0), (2 * period, 0.8), (t0 / 3, 0.4)]
                        {
                            let mut got_t = t0;
                            let got = remove_doubling(
                                &x,
                                max_period,
                                min_period,
                                n,
                                &mut got_t,
                                prev_period,
                                prev_gain,
                            );
                            let mut want_t = t0;
                            let want = remove_doubling_plain(
                                &x,
                                max_period,
                                min_period,
                                n,
                                &mut want_t,
                                prev_period,
                                prev_gain,
                            );
                            let what = format!(
                                "n {n} kind {kind} period {period} t0 {t0} prev {prev_period}"
                            );
                            assert_eq!(got_t, want_t, "{what}");
                            assert!(same_bits(got, want), "{what}: gain {got} and {want}");
                            if got_t + 2 < t0.min(1023) {
                                picked_shorter += 1;
                            }
                        }
                    }
                }
            }
        }
        // The inputs must reach the branch that replaces the period.
        assert!(
            picked_shorter > 50,
            "only {picked_shorter} cases picked a shorter period"
        );
    }

    /// `find_best_pitch` as it was written before its loop was changed: with
    /// indexing, and with the lower limit of the energy as a plain assignment.
    fn find_best_pitch_plain(xcorr: &[f32], y: &[f32], len: usize, max_pitch: usize) -> [usize; 2] {
        let mut best_num = [-1.0f32, -1.0f32];
        let mut best_den = [0.0f32, 0.0f32];
        let mut best_pitch = [0usize, 1];
        let mut syy = window_energy_plus_one(&y[..len]);
        for i in 0..max_pitch {
            if xcorr[i] > 0.0 {
                let xc16 = xcorr[i] * 1e-12;
                let num = xc16 * xc16;
                if num * best_den[1] > best_num[1] * syy {
                    if num * best_den[0] > best_num[0] * syy {
                        best_num[1] = best_num[0];
                        best_den[1] = best_den[0];
                        best_pitch[1] = best_pitch[0];
                        best_num[0] = num;
                        best_den[0] = syy;
                        best_pitch[0] = i;
                    } else {
                        best_num[1] = num;
                        best_den[1] = syy;
                        best_pitch[1] = i;
                    }
                }
            }
            syy += y[i + len] * y[i + len] - y[i] * y[i];
            if syy < 1.0 {
                syy = 1.0;
            }
        }
        best_pitch
    }

    #[test]
    fn find_best_pitch_matches_the_plain_version() {
        let mut rng = Rng::new(11);
        let mut differ_from_first_guess = 0;
        for kind in KINDS {
            for len in [0usize, 1, 3, 4, 5, 30, 61, 240] {
                for max_pitch in [0usize, 1, 2, 3, 17, 244, 489] {
                    for shape in 0..3 {
                        let mut y = samples(&mut rng, len + max_pitch, kind);
                        if shape == 1 {
                            // Loud at the start, silent after it: the window
                            // energy falls to its lower limit of 1.
                            for v in y.iter_mut().skip(len / 2 + 1) {
                                *v = 0.0;
                            }
                        }
                        let mut xcorr = samples(&mut rng, max_pitch, kind);
                        if shape == 2 {
                            // As in the second search: -1 almost everywhere.
                            for (i, v) in xcorr.iter_mut().enumerate() {
                                if i % 50 > 4 {
                                    *v = -1.0;
                                }
                            }
                        }
                        let mut got = [7usize, 7];
                        find_best_pitch(&xcorr, &y, len, max_pitch, &mut got);
                        let want = find_best_pitch_plain(&xcorr, &y, len, max_pitch);
                        assert_eq!(
                            got, want,
                            "{kind:?} len {len} max_pitch {max_pitch} shape {shape}"
                        );
                        if got != [0, 1] {
                            differ_from_first_guess += 1;
                        }
                    }
                }
            }
        }
        // The inputs must reach the branches that replace the best lags.
        assert!(differ_from_first_guess > 100);
    }

    /// `pitch_downsample` as it was written before `decimate2`: one output at
    /// a time, with a test per channel and output.
    fn pitch_downsample_plain(x: &[&[f32]], x_lp: &mut [f32], len: usize, c: usize, factor: usize) {
        let offset = factor / 2;
        for i in 1..len {
            let mut val = 0.0f32;
            for x_k in &x[..c] {
                if factor * i + offset < x_k.len() {
                    val += 0.25 * x_k[factor * i - offset]
                        + 0.25 * x_k[factor * i + offset]
                        + 0.5 * x_k[factor * i];
                }
            }
            x_lp[i] = val;
        }
        let mut val = 0.0f32;
        for x_k in &x[..c] {
            if offset < x_k.len() {
                val += 0.25 * x_k[offset] + 0.5 * x_k[0];
            }
        }
        x_lp[0] = val;
    }

    #[test]
    fn decimate2_scalar_matches_the_formula() {
        let mut rng = Rng::new(8);
        for kind in KINDS {
            for n in 0..=40 {
                // Exactly as long as the outputs need.
                let x0 = samples(&mut rng, 2 * n, kind);
                let x1 = samples(&mut rng, 2 * n, kind);
                for c in 1..=2 {
                    let mut got = vec![f32::NAN; n];
                    if c == 1 {
                        decimate2_scalar([&x0[..]], &mut got);
                    } else {
                        decimate2_scalar([&x0[..], &x1[..]], &mut got);
                    }
                    for i in 1..n {
                        let mut want = 0.0f32;
                        for x_k in [&x0, &x1].iter().take(c) {
                            want +=
                                0.25 * x_k[2 * i - 1] + 0.25 * x_k[2 * i + 1] + 0.5 * x_k[2 * i];
                        }
                        assert!(
                            same_bits(got[i], want),
                            "{kind:?} n {n} channels {c} output {i}: {} and {want}",
                            got[i]
                        );
                    }
                    // The first output is not written.
                    assert!(n == 0 || got[0].is_nan());
                }
            }
        }
    }

    #[test]
    fn pitch_downsample_matches_the_plain_version() {
        let mut rng = Rng::new(9);
        for kind in KINDS {
            for len in [1usize, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 33, 62, 572, 992] {
                for c in 0..=3 {
                    for factor in [1usize, 2, 3, 4] {
                        // With no channel at all, aarch64 takes its separate
                        // path for long inputs, which needs two channels.
                        if cfg!(target_arch = "aarch64") && c == 0 && factor == 2 {
                            continue;
                        }
                        // Channels of exactly factor * len samples (what the
                        // encoder passes), shorter ones, and one that is a
                        // sample shorter than the other.
                        for (cut0, cut1) in [(0, 0), (1, 0), (0, 3), (5, 5), (factor * len, 0)] {
                            let full = factor * len;
                            let x0 = samples(&mut rng, full - cut0.min(full), kind);
                            let x1 = samples(&mut rng, full - cut1.min(full), kind);
                            let x2 = samples(&mut rng, full, kind);
                            let chans: [&[f32]; 3] = [&x0, &x1, &x2];
                            let mut got = vec![f32::NAN; len];
                            pitch_downsample(&chans[..c], &mut got, len, c, factor);
                            let mut want = vec![f32::NAN; len];
                            pitch_downsample_plain(&chans[..c], &mut want, len, c, factor);
                            flatten_spectrum(&mut want);
                            assert_same(
                                &got,
                                &want,
                                &format!(
                                    "{kind:?} len {len} c {c} factor {factor} cut {cut0} {cut1}"
                                ),
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn pitch_xcorr_handles_short_and_empty_inputs() {
        // No lag fits: nothing is written.
        let x = [1.0f32; 8];
        let y = [1.0f32; 4];
        let mut out = [7.0f32; 4];
        pitch_xcorr(&x, &y, &mut out, 8, 4);
        assert_eq!(out, [7.0; 4]);
        // Zero length: nothing is written.
        pitch_xcorr(&x, &y, &mut out, 0, 4);
        assert_eq!(out, [7.0; 4]);
        // `y` is long enough for two of the four lags only.
        let y = [1.0f32; 9];
        pitch_xcorr(&x, &y, &mut out, 8, 4);
        assert_eq!(out, [8.0, 8.0, 7.0, 7.0]);
    }
}
