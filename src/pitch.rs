//! Pitch analysis of the CELT pre-filter and of the CELT loss concealment
//! (libopus celt/pitch.c and celt/pitch.h, float build), and the inner
//! products they share with the rest of CELT.
use crate::celt_lpc::{autocorr, lpc};
#[cfg(not(feature = "std"))]
use crate::compat::Math;
use crate::fixedvec::FixedVec;
use crate::mathops::mac;
use core::mem::MaybeUninit;

/// Capacity for pitch-search working buffers (x_lp4 / y_lp4 / xcorr). Bounded by
/// `COMBFILTER_MAXPERIOD = 1024` and the maximum pitch lag.
const PITCH_BUF_MAX: usize = 1024;

// How the sums of products in this file are rounded
// --------------------------------------------------
// The output of the encoder, and of the decoder's loss concealment, depends
// on the exact rounding of these sums. Every function adds in the order of
// libopus 1.6 on the same processor:
//
// * `celt_inner_prod` on 64-bit ARM is libopus `celt_inner_prod_neon`
//   (celt/arm/pitch_neon_intr.c). Four running sums take the samples 4m,
//   4m + 1, 4m + 2 and 4m + 3. They are added as (s0 + s2) + (s1 + s3), and
//   the last `n % 4` products are added one by one after that.
// * `celt_inner_prod` on x86-64 is libopus `celt_inner_prod_sse`
//   (celt/x86/pitch_sse.c): the same order as on 64-bit ARM, with each
//   product rounded before its addition.
// * `pitch_xcorr` on 64-bit ARM is libopus `celt_pitch_xcorr_float_neon`
//   (celt/arm/celt_neon_intr.c). The lags below the largest multiple of four
//   add their products one sample after the other ("by sample"). The one to
//   three lags after them are `celt_inner_prod`.
// * The other sums of this file are loops of libopus's C code, written here
//   with `mac`.
//
// clang builds libopus for 64-bit ARM with fused multiply-add (`mac`
// explains why), and its NEON kernels ask for the fused instruction. Every
// product here is therefore added with one rounding on 64-bit ARM. On other
// processors `mac` rounds the product first, as libopus's C code does
// there. The other x86 kernels of this file (`inner_prod`, `pitch_xcorr`
// and the ones they call) are separate: they are what this crate has used
// on x86 so far, and they do not follow libopus's x86 kernels in every
// detail.
//
// Several sums can be computed in the same pass without changing any
// result, as long as every sum keeps its own order. The kernels below do
// that: the processor then works on many sums at the same time instead of
// waiting for one addition after the other.

/// Cross-correlation in the by-sample order, portable code.
///
/// `out[k] = sum over j < len of x[j] * y[j + k]` for every `k < out.len()`.
/// Each lag has its own running sum and adds its products in rising order of
/// `j`, each with `mac`. Eight lags are handled per pass, which gives the
/// same result as one lag after the other and is faster.
///
/// This is libopus `celt_pitch_xcorr_c` with `celt_inner_prod_c`, and the
/// reference of the NEON kernel in the tests.
///
/// Needs `x.len() >= len` and `y.len() >= len + out.len() - 1`.
#[allow(dead_code)] // unused on targets that have a vector kernel
fn xcorr_by_sample_scalar(x: &[f32], y: &[f32], out: &mut [f32], len: usize) {
    const L: usize = 8;
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
                acc = mac(acc, xj, yj);
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
                acc[l] = mac(acc[l], xj, yw[l]);
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

/// libopus `celt_inner_prod` of the float build: the sum of `x[i] * y[i]`.
///
/// On 64-bit ARM this is `celt_inner_prod_neon` (celt/arm/pitch_neon_intr.c).
/// It has one accumulator of four lanes, updated with fused multiply-adds.
/// The lanes are added low half plus high half, then the two results:
/// (s0 + s2) + (s1 + s3). The last one to three values are added one at a
/// time with fused multiply-adds.
///
/// On x86-64 this is `celt_inner_prod_sse` (celt/x86/pitch_sse.c), which
/// libopus's x86-64 builds call without a run-time check. It has the steps
/// of the NEON kernel, with each product rounded before its addition: one
/// accumulator of four lanes, the lanes added as (s0 + s2) + (s1 + s3), and
/// the last one to three values added one at a time.
///
/// Elsewhere this is `celt_inner_prod_c` (celt/pitch.h): one value at a time
/// from 0, with `mac`.
///
/// `y` must be at least as long as `x`.
#[inline(always)]
pub(crate) fn celt_inner_prod(x: &[f32], y: &[f32]) -> f32 {
    let n = x.len();
    // The only bounds check. The loads below rely on it.
    let y = &y[..n];
    #[cfg(target_arch = "aarch64")]
    {
        use core::arch::aarch64::*;
        // SAFETY: every load reads `x[i..i + 4]` or `y[i..i + 4]` with
        // `i + 4 <= n`; both slices hold exactly `n` values.
        unsafe {
            let xp = x.as_ptr();
            let yp = y.as_ptr();
            let mut acc = vdupq_n_f32(0.0);
            let mut i = 0;
            while i + 8 <= n {
                acc = vfmaq_f32(acc, vld1q_f32(xp.add(i)), vld1q_f32(yp.add(i)));
                acc = vfmaq_f32(acc, vld1q_f32(xp.add(i + 4)), vld1q_f32(yp.add(i + 4)));
                i += 8;
            }
            if n - i >= 4 {
                acc = vfmaq_f32(acc, vld1q_f32(xp.add(i)), vld1q_f32(yp.add(i)));
                i += 4;
            }
            let mut xy = lanes_sum(acc);
            while i < n {
                xy = mac(xy, *xp.add(i), *yp.add(i));
                i += 1;
            }
            xy
        }
    }
    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    {
        use core::arch::x86_64::*;
        // SAFETY: SSE is part of this target (see the `cfg`). Every load
        // reads `x[i..i + 4]` or `y[i..i + 4]` with `i + 4 <= n`; both
        // slices hold exactly `n` values.
        unsafe {
            let xp = x.as_ptr();
            let yp = y.as_ptr();
            let mut sum = _mm_setzero_ps();
            let mut i = 0;
            while i + 4 <= n {
                sum = _mm_add_ps(sum, _mm_mul_ps(_mm_loadu_ps(xp.add(i)), _mm_loadu_ps(yp.add(i))));
                i += 4;
            }
            // (s0 + s2) + (s1 + s3), as libopus adds the lanes.
            let sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
            let sum = _mm_add_ss(sum, _mm_shuffle_ps::<0x55>(sum, sum));
            let mut xy = _mm_cvtss_f32(sum);
            while i < n {
                xy = mac(xy, *xp.add(i), *yp.add(i));
                i += 1;
            }
            xy
        }
    }
    #[cfg(not(any(target_arch = "aarch64", all(target_arch = "x86_64", target_feature = "sse2"))))]
    {
        let mut xy = 0.0f32;
        for (&a, &b) in x.iter().zip(y) {
            xy = mac(xy, a, b);
        }
        xy
    }
}

/// The four lanes of `v` added as libopus's NEON kernels add them:
/// `vadd_f32` of the low and the high half, then `vpadd_f32`, which is
/// (v0 + v2) + (v1 + v3).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn lanes_sum(v: core::arch::aarch64::float32x4_t) -> f32 {
    use core::arch::aarch64::*;
    // SAFETY: these vector instructions are part of every 64-bit ARM
    // processor.
    unsafe {
        let half = vadd_f32(vget_low_f32(v), vget_high_f32(v));
        vget_lane_f32::<0>(vpadd_f32(half, half))
    }
}

/// Sum of `x[i] * y[i]` for `i < n`: libopus `celt_inner_prod`. Panics when
/// `x` or `y` is shorter than `n`.
///
/// On x86 this uses this crate's own SSE and AVX kernels.
pub fn inner_prod(x: &[f32], y: &[f32], n: usize) -> f32 {
    // The vector kernels read `n` samples through pointers without a check,
    // so the lengths are checked here.
    let (x, y) = (&x[..n], &y[..n]);
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if crate::compat::x86_has_avx_fma() {
        // SAFETY: `x` and `y` hold exactly `n` samples, and the CPU has AVX
        // and FMA (checked just above).
        return unsafe { inner_prod_avx(x, y, n) };
    }
    // SAFETY: `x` and `y` hold exactly `n` samples, and SSE is part of the
    // baseline of x86-64.
    #[cfg(all(target_arch = "x86_64", target_feature = "sse"))]
    unsafe {
        inner_prod_sse(x, y, n)
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "sse")))]
    celt_inner_prod(x, y)
}

/// The inner products of `x` with `y1` and with `y2` over `n` samples:
/// libopus `dual_inner_prod`. Panics when one of the three is shorter than
/// `n`.
///
/// On 64-bit ARM each of the two has the bits of `celt_inner_prod`, as in
/// libopus `dual_inner_prod_neon`. Other targets without an x86 kernel add
/// one value at a time, as libopus `dual_inner_prod_c` does.
pub fn dual_inner_prod(x: &[f32], y1: &[f32], y2: &[f32], n: usize) -> (f32, f32) {
    // As in `inner_prod`: the kernels read `n` samples without a check.
    let (x, y1, y2) = (&x[..n], &y1[..n], &y2[..n]);
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if crate::compat::x86_has_avx_fma() {
        // SAFETY: the three slices hold exactly `n` samples, and the CPU has
        // AVX and FMA (checked just above).
        return unsafe { dual_inner_prod_avx(x, y1, y2, n) };
    }
    // SAFETY: the three slices hold exactly `n` samples, and SSE is part of
    // the baseline of x86-64.
    #[cfg(all(target_arch = "x86_64", target_feature = "sse"))]
    unsafe {
        dual_inner_prod_sse(x, y1, y2, n)
    }
    #[cfg(target_arch = "aarch64")]
    {
        let [a, b] = inner_prod_k_neon::<2>(x, &[y1, y2]);
        (a, b)
    }
    #[cfg(not(any(
        target_arch = "aarch64",
        all(target_arch = "x86_64", target_feature = "sse")
    )))]
    (celt_inner_prod(x, y1), celt_inner_prod(x, y2))
}

/// libopus `celt_pitch_xcorr`: `xcorr[k] = sum over j < len of x[j] * y[j + k]`
/// for `k < max_pitch`.
///
/// Lag `k` reads `y[k..k + len]`, so `y` should hold `len + max_pitch - 1`
/// samples. With a shorter `y`, only the lags it holds are computed. The
/// callers in the codec always pass a long enough `y`.
pub fn pitch_xcorr(x: &[f32], y: &[f32], xcorr: &mut [f32], len: usize, max_pitch: usize) {
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
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if crate::compat::x86_has_avx_fma() {
        // SAFETY: `x` holds exactly `len` samples and `y` at least
        // `len + max_pitch - 1` (`max_pitch` was limited to that above).
        // `xcorr` is written with index checks. The CPU has AVX and FMA
        // (checked just above).
        return unsafe { pitch_xcorr_avx(x, y, xcorr, len, max_pitch) };
    }
    #[cfg(target_arch = "aarch64")]
    pitch_xcorr_neon(x, y, &mut xcorr[..max_pitch]);
    // SAFETY: `x` holds exactly `len` samples and `y` at least
    // `len + max_pitch - 1` (`max_pitch` was limited to that above).
    // `xcorr` is written with index checks. SSE is part of the baseline of
    // x86-64.
    #[cfg(all(target_arch = "x86_64", target_feature = "sse"))]
    unsafe {
        pitch_xcorr_sse(x, y, xcorr, len, max_pitch)
    }
    #[cfg(not(any(
        target_arch = "aarch64",
        all(target_arch = "x86_64", target_feature = "sse")
    )))]
    xcorr_by_sample_scalar(x, y, &mut xcorr[..max_pitch], len);
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
    inner_prod_many_side(x, y, offsets, n, out, &mut || {});
}

/// `inner_prod_many` that calls `side` once per four samples of every pass
/// on aarch64, and never elsewhere.
///
/// The passes are limited by their loads and leave time for one chain of
/// additions that waits for itself, such as the window energies of
/// `remove_doubling`. `side` can do a few steps of such a chain.
fn inner_prod_many_side<F: FnMut()>(
    x: &[f32],
    y: &[f32],
    offsets: &[usize],
    n: usize,
    out: &mut [f32],
    side: &mut F,
) {
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
                1 => inner_prod_pass_neon::<4, F>(x, &ys, out, side),
                2 => inner_prod_pass_neon::<8, F>(x, &ys, out, side),
                3 => inner_prod_pass_neon::<12, F>(x, &ys, out, side),
                _ => inner_prod_pass_neon::<16, F>(x, &ys, out, side),
            }
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let _ = side;
        for (o, &off) in out.iter_mut().zip(offsets) {
            *o = inner_prod(x, &y[off..], n);
        }
    }
}

/// One pass of `inner_prod_many`: stores the first `out.len()` of the `K`
/// products of `x` with the windows `ys[..K]`. Needs `out.len() <= K <= 16`.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn inner_prod_pass_neon<const K: usize, F: FnMut()>(
    x: &[f32],
    ys: &[&[f32]; MANY_PER_PASS],
    out: &mut [f32],
    side: &mut F,
) {
    let ys_k: [&[f32]; K] = core::array::from_fn(|k| ys[k]);
    let sums = inner_prod_k_side_neon::<K, F>(x, &ys_k, side);
    out.copy_from_slice(&sums[..out.len()]);
}

/// `K` inner products of `x` with `K` other signals in one pass.
/// `sums[k]` has the bits of `celt_inner_prod(x, ys[k])`.
///
/// Each product has its own register of running sums. One product alone has
/// to wait for each fused multiply-add before the next; with several of them
/// the processor works on the others in the meantime.
///
/// Panics when one of the signals is shorter than `x`.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn inner_prod_k_neon<const K: usize>(x: &[f32], ys: &[&[f32]; K]) -> [f32; K] {
    inner_prod_k_side_neon::<K, _>(x, ys, &mut || {})
}

/// `inner_prod_k_neon` that calls `side` once per four samples.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn inner_prod_k_side_neon<const K: usize, F: FnMut()>(
    x: &[f32],
    ys: &[&[f32]; K],
    side: &mut F,
) -> [f32; K] {
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
            side();
            let xv = vld1q_f32(x.add(i));
            for (a, &y) in acc.iter_mut().zip(&ys) {
                *a = vfmaq_f32(*a, xv, vld1q_f32(y.add(i)));
            }
            i += 4;
        }
        let mut sums = [0.0f32; K];
        for ((sum, a), &y) in sums.iter_mut().zip(&acc).zip(&ys) {
            let mut s = lanes_sum(*a);
            for j in i..n {
                s = mac(s, *x.add(j), *y.add(j));
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
/// same bits as `xcorr_by_sample_scalar`, and as libopus
/// `xcorr_kernel_neon_float` for each group of four lags.
///
/// One lag alone has to wait for each addition before it can start the next
/// one. With 64 lags in 16 registers (`G = 16`) the processor always has
/// other additions to work on.
///
/// The main loop takes eight samples per step. Running sum `g` then needs
/// the eight windows `y[j + 4g + t..][..4]` for `t < 8`, and the last four of
/// them are the first four of sum `g + 1`. Each window is loaded once per
/// step and used by two sums, which halves the loads of a step of four
/// samples. On an Apple M4 the loads limited the steps of four samples; the
/// step of eight is limited by the multiply-adds and runs about 1.8 times
/// as fast.
///
/// `len` is `x.len()`. Only the first `4 * G` elements of `out` are written.
/// Panics unless `y` holds `len + 4 * G - 1` samples and `out` holds `4 * G`
/// lags. The last sample of the last lag is `y[len + 4 * G - 2]`, and no load
/// goes past it.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn xcorr_by_sample_neon<const G: usize>(x: &[f32], y: &[f32], out: &mut [f32]) {
    xcorr_by_sample_side_neon::<G, _>(x, y, out, &mut || {});
}

/// `xcorr_by_sample_neon` that also calls `side` once per step of eight
/// samples.
///
/// The kernel keeps the multiply-add units busy but leaves time between the
/// steps of a single chain of additions. Work that is one such chain, for
/// example the energy of a signal summed one sample after the other, then
/// costs almost nothing when `side` does a few of its steps at a time.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn xcorr_by_sample_side_neon<const G: usize, F: FnMut()>(
    x: &[f32],
    y: &[f32],
    out: &mut [f32],
    side: &mut F,
) {
    use core::arch::aarch64::*;

    let len = x.len();
    assert!(y.len() >= len + 4 * G - 1 && out.len() >= 4 * G);
    let (x, y, out) = (x.as_ptr(), y.as_ptr(), out.as_mut_ptr());

    // SAFETY: `x` holds `len` samples, `y` at least len + 4G - 1 and `out` at
    // least 4G (checked above).
    // - The eight-sample loop reads x[j..j + 8] with j + 8 <= len. Its
    //   windows start at y[j + t] with t <= 4G + 3, so the last element read
    //   is y[j + 4G + 6] <= y[len - 8 + 4G + 6] = y[len + 4G - 2].
    // - The four-sample step reads x[j..j + 4] with j + 4 <= len, and
    //   y[j + 4g + s..][..4] with s <= 3 and g < G, which ends at
    //   y[len - 4 + 4(G - 1) + 6] = y[len + 4G - 2].
    // - The one-sample loop reads x[j] and y[j + 4g..][..4] with j < len,
    //   which ends at y[len + 4G - 2] as well.
    // The stores write out[..4G].
    unsafe {
        let mut acc = [vdupq_n_f32(0.0); G];
        let mut j = 0;
        while j + 8 <= len {
            side();
            let x0 = vld1q_f32(x.add(j));
            let x1 = vld1q_f32(x.add(j + 4));
            let mut yp = y.add(j);
            // The windows y[j + 4g + t..] for t < 4 of the sum `g` at hand.
            let mut w = [
                vld1q_f32(yp),
                vld1q_f32(yp.add(1)),
                vld1q_f32(yp.add(2)),
                vld1q_f32(yp.add(3)),
            ];
            for a in acc.iter_mut() {
                let q = yp.add(4);
                let next = [
                    vld1q_f32(q),
                    vld1q_f32(q.add(1)),
                    vld1q_f32(q.add(2)),
                    vld1q_f32(q.add(3)),
                ];
                let mut s = *a;
                s = vfmaq_laneq_f32::<0>(s, w[0], x0);
                s = vfmaq_laneq_f32::<1>(s, w[1], x0);
                s = vfmaq_laneq_f32::<2>(s, w[2], x0);
                s = vfmaq_laneq_f32::<3>(s, w[3], x0);
                s = vfmaq_laneq_f32::<0>(s, next[0], x1);
                s = vfmaq_laneq_f32::<1>(s, next[1], x1);
                s = vfmaq_laneq_f32::<2>(s, next[2], x1);
                s = vfmaq_laneq_f32::<3>(s, next[3], x1);
                *a = s;
                w = next;
                yp = q;
            }
            j += 8;
        }
        if j + 4 <= len {
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
    xcorr_by_sample_passes_side_neon::<G, _>(x, y, out, &mut || {});
}

/// `xcorr_by_sample_passes_neon` that calls `side` once per step of eight
/// samples of every pass (see `xcorr_by_sample_side_neon`).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn xcorr_by_sample_passes_side_neon<const G: usize, F: FnMut()>(
    x: &[f32],
    y: &[f32],
    out: &mut [f32],
    side: &mut F,
) {
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
        xcorr_by_sample_side_neon::<G, F>(x, &y[first..], &mut out[first..], side);
        i += per_pass;
    }
}

// Cross-correlation in the by-sample order with shared windows
// -------------------------------------------------------------
// `xcorr_by_sample_neon` loads the windows of y for each group of four lags
// on its own. On the Cortex-A55, a core that runs the instructions in
// program order, those loads cost more than the multiply-adds: a 16-byte
// load takes two cycles, four when it does not start on a multiple of eight
// bytes, and it does not pair with a multiply-add.
//
// `xcorr_skewed_neon` shares the windows between groups. Group `g` (lags
// 4g to 4g + 3) works on sample `j - 4g` while group 0 works on sample `j`:
// all groups then multiply the same window y[j..j + 4], each with its own
// sample of x. Every lag still adds its products in rising order of the
// sample, so the sums are those of `xcorr_by_sample_neon`.
//
// A pass handles 64 lags in 16 running sums: two halves of eight groups, the
// second half 32 lags after the first. The two halves share the samples of
// x. A block is four samples. In the middle of a pass ("steady" blocks) all
// groups are busy. At the start of a pass group `g` starts in block `g`, and
// at its end group `g` stops in block `M + g - 1`, where `M` is the number of
// whole blocks of x. While one pass ends, the next one starts in the groups
// that are done, so every block keeps all 16 sums busy. The first pass
// starts with sums that are thrown away, and the last one ends with them.
//
// The steady blocks are written in assembly, so that the order of the
// instructions is fixed: the loads of x for the next step of four samples
// stand between the multiply-adds of this one, and each window is made with
// one `ext` from two loads. The rest is written with intrinsics.

/// Lags per pass of `xcorr_skewed_neon`.
#[cfg(target_arch = "aarch64")]
const SKEW_PASS: usize = 64;

/// Fewest whole blocks of four samples of x that `xcorr_skewed_neon` takes.
/// With fewer, the start and the end of a pass are most of its work, and
/// `xcorr_by_sample_neon` is faster.
#[cfg(target_arch = "aarch64")]
const SKEW_MIN_BLOCKS: usize = 8;

/// Whether `pitch_xcorr` and the coarse search of `pitch_search` use
/// `xcorr_skewed_neon` for 64 lags or more. It is built for cores that run
/// the instructions in program order, such as the Cortex-A55 and the other
/// small cores of Android phones, where it takes about half the time of
/// `xcorr_by_sample_neon`. Apple's cores all reorder the instructions: there
/// the shared windows save nothing, and its six `ext` per 64 multiply-adds
/// make it about 10 % slower (Apple M4 Pro). The two give the same bits.
#[cfg(target_arch = "aarch64")]
const USE_SKEW: bool = !cfg!(target_vendor = "apple");

/// Steady blocks per call of the assembly loop, two at a time. `side` runs
/// after each call.
#[cfg(target_arch = "aarch64")]
const SKEW_PAIRS_PER_CALL: usize = 2;

/// The 16 running sums of a pass that starts at lag `o`: element `g` holds the
/// lags o + 4g to o + 4g + 3, element 8 + g the lags o + 32 + 4g to
/// o + 32 + 4g + 3.
#[cfg(target_arch = "aarch64")]
type SkewSums = [core::arch::aarch64::float32x4_t; 16];

/// Two multiply-adds of one group in one step of a steady block: sum `$a`
/// (first half) with window `$wa`, sum `$b` (second half) with window `$wb`,
/// both with lane `$lane` of the x register `$x`.
#[cfg(target_arch = "aarch64")]
#[rustfmt::skip]
macro_rules! skew_fmas {
    ($a:literal, $b:literal, $x:literal, $wa:literal, $wb:literal, $lane:literal) => {
        concat!(
            "fmla v", $a, ".4s, v", $wa, ".4s, v", $x, ".s[", $lane, "]\n",
            "fmla v", $b, ".4s, v", $wb, ".4s, v", $x, ".s[", $lane, "]\n",
        )
    };
}

/// `ext` that makes the window of step `$shift / 4` from the window registers
/// `$cur` (y at the block) and `$next` (four samples later).
#[cfg(target_arch = "aarch64")]
#[rustfmt::skip]
macro_rules! skew_ext {
    ($dst:literal, $cur:literal, $next:literal, $shift:literal) => {
        concat!("ext v", $dst, ".16b, v", $cur, ".16b, v", $next, ".16b, #", $shift, "\n")
    };
}

/// One step of four in a steady block: the 16 multiply-adds of one sample
/// for each group. `$mid` stands after the fourth group (the windows of the
/// next step). With offsets, x register `24 + g` is loaded again after the
/// last use of its two values, from `{xm}` plus the offset in bytes.
#[cfg(target_arch = "aarch64")]
macro_rules! skew_step {
    ($wa:literal, $wb:literal, $lane:literal, [$($mid:tt)*]) => {
        concat!(
            skew_fmas!(0, 8, 24, $wa, $wb, $lane),
            skew_fmas!(1, 9, 25, $wa, $wb, $lane),
            skew_fmas!(2, 10, 26, $wa, $wb, $lane),
            skew_fmas!(3, 11, 27, $wa, $wb, $lane),
            $($mid)*
            skew_fmas!(4, 12, 28, $wa, $wb, $lane),
            skew_fmas!(5, 13, 29, $wa, $wb, $lane),
            skew_fmas!(6, 14, 30, $wa, $wb, $lane),
            skew_fmas!(7, 15, 31, $wa, $wb, $lane),
        )
    };
    ($wa:literal, $wb:literal, $lane:literal, [$($mid:tt)*],
     $o0:literal, $o1:literal, $o2:literal, $o3:literal,
     $o4:literal, $o5:literal, $o6:literal, $o7:literal) => {
        concat!(
            skew_fmas!(0, 8, 24, $wa, $wb, $lane),
            "ldr d24, [{xm}, #", $o0, "]\n",
            skew_fmas!(1, 9, 25, $wa, $wb, $lane),
            "ldr d25, [{xm}, #", $o1, "]\n",
            skew_fmas!(2, 10, 26, $wa, $wb, $lane),
            "ldr d26, [{xm}, #", $o2, "]\n",
            skew_fmas!(3, 11, 27, $wa, $wb, $lane),
            "ldr d27, [{xm}, #", $o3, "]\n",
            $($mid)*
            skew_fmas!(4, 12, 28, $wa, $wb, $lane),
            "ldr d28, [{xm}, #", $o4, "]\n",
            skew_fmas!(5, 13, 29, $wa, $wb, $lane),
            "ldr d29, [{xm}, #", $o5, "]\n",
            skew_fmas!(6, 14, 30, $wa, $wb, $lane),
            "ldr d30, [{xm}, #", $o6, "]\n",
            skew_fmas!(7, 15, 31, $wa, $wb, $lane),
            "ldr d31, [{xm}, #", $o7, "]\n",
        )
    };
}

/// One steady block of four samples. On entry `$cy` and `$cz` hold the
/// windows of the two halves at the block (y at {ym} and {ym} + 128 bytes),
/// and v24 to v31 the first two samples of x for each group (x at
/// {xm} - 16g bytes). The block loads the windows four samples later into
/// `$ny` and `$nz`, makes the windows of the steps 1 to 3 in v20 to v23,
/// loads the last two samples of x for each group after step 1, and the
/// first two of the next block after step 3. It ends with {xm} and {ym} one
/// block further.
#[cfg(target_arch = "aarch64")]
#[rustfmt::skip]
macro_rules! skew_block {
    ($cy:literal, $cz:literal, $ny:literal, $nz:literal) => {
        concat!(
            "ldr q", $ny, ", [{ym}, #16]\n",
            "ldr q", $nz, ", [{ym}, #144]\n",
            skew_step!($cy, $cz, 0,
                [skew_ext!(20, $cy, $ny, 4), skew_ext!(21, $cz, $nz, 4),]),
            skew_step!(20, 21, 1,
                [skew_ext!(22, $cy, $ny, 8), skew_ext!(23, $cz, $nz, 8),],
                8, -8, -24, -40, -56, -72, -88, -104),
            skew_step!(22, 23, 0,
                [skew_ext!(20, $cy, $ny, 12), skew_ext!(21, $cz, $nz, 12),]),
            skew_step!(20, 21, 1, [], 16, 0, -16, -32, -48, -64, -80, -96),
            "add {xm}, {xm}, #16\n",
            "add {ym}, {ym}, #16\n",
        )
    };
}

/// Step I of the change from the pass at lag `old` to the pass at lag `new`
/// (both may be the same pass): the groups g <= I run block I of `new`, the
/// others block `m_full + I` of `old`.
///
/// # Safety
///
/// As `xcorr_skewed_neon` explains for its pointers.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn skew_change_step<const I: usize>(
    acc: &mut SkewSums,
    xp: *const f32,
    yp: *const f32,
    m_full: usize,
    old: usize,
    new: usize,
) {
    use core::arch::aarch64::*;
    // SAFETY: see `xcorr_skewed_neon`.
    unsafe {
        let bn = yp.add(new + 4 * I);
        let bo = yp.add(old + 4 * (m_full + I));
        let xq: [*const f32; 8] = core::array::from_fn(|g| {
            if g <= I {
                xp.add(4 * (I - g))
            } else {
                xp.add(4 * (m_full + I - g))
            }
        });
        let lo: [float32x2_t; 8] = core::array::from_fn(|g| vld1_f32(xq[g]));
        let hi: [float32x2_t; 8] = core::array::from_fn(|g| vld1_f32(xq[g].add(2)));
        macro_rules! step {
            ($t:literal, $x:ident, $lane:literal) => {{
                let (wn, vn) = (vld1q_f32(bn.add($t)), vld1q_f32(bn.add(32 + $t)));
                let (wo, vo) = (vld1q_f32(bo.add($t)), vld1q_f32(bo.add(32 + $t)));
                for g in 0..8 {
                    let (w, v) = if g <= I { (wn, vn) } else { (wo, vo) };
                    acc[g] = vfmaq_lane_f32::<$lane>(acc[g], w, $x[g]);
                    acc[8 + g] = vfmaq_lane_f32::<$lane>(acc[8 + g], v, $x[g]);
                }
            }};
        }
        step!(0, lo, 0);
        step!(1, lo, 1);
        step!(2, hi, 0);
        step!(3, hi, 1);
    }
}

/// Ends group G of the pass at lag `o`, which has run its last block: adds
/// the products of the last `len % 4` samples, stores the two sums when
/// `store` is set, and clears them for the next pass.
///
/// # Safety
///
/// As `xcorr_skewed_neon` explains for its pointers.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn skew_finish_group<const G: usize>(
    acc: &mut SkewSums,
    xp: *const f32,
    yp: *const f32,
    len: usize,
    o: usize,
    out: *mut f32,
    store: bool,
) {
    use core::arch::aarch64::*;
    // SAFETY: see `xcorr_skewed_neon`.
    unsafe {
        if store {
            for j in len & !3..len {
                let xv = vdupq_n_f32(*xp.add(j));
                acc[G] = vfmaq_f32(acc[G], vld1q_f32(yp.add(o + 4 * G + j)), xv);
                acc[8 + G] = vfmaq_f32(acc[8 + G], vld1q_f32(yp.add(o + 32 + 4 * G + j)), xv);
            }
            vst1q_f32(out.add(o + 4 * G), acc[G]);
            vst1q_f32(out.add(o + 32 + 4 * G), acc[8 + G]);
        }
        acc[G] = vdupq_n_f32(0.0);
        acc[8 + G] = vdupq_n_f32(0.0);
    }
}

/// The change from the pass at lag `old` to the pass at lag `new`: seven
/// blocks in which the groups of `old` end one after the other and those of
/// `new` start. The sums of `old` are stored when `store` is set; otherwise
/// they are thrown away (the start of the first pass).
///
/// # Safety
///
/// As `xcorr_skewed_neon` explains for its pointers.
#[cfg(target_arch = "aarch64")]
#[inline(never)]
#[allow(clippy::too_many_arguments)]
unsafe fn skew_change(
    acc: &mut SkewSums,
    xp: *const f32,
    yp: *const f32,
    len: usize,
    old: usize,
    new: usize,
    out: *mut f32,
    store: bool,
) {
    let m_full = len / 4;
    // SAFETY: see `xcorr_skewed_neon`.
    unsafe {
        skew_finish_group::<0>(acc, xp, yp, len, old, out, store);
        skew_change_step::<0>(acc, xp, yp, m_full, old, new);
        skew_finish_group::<1>(acc, xp, yp, len, old, out, store);
        skew_change_step::<1>(acc, xp, yp, m_full, old, new);
        skew_finish_group::<2>(acc, xp, yp, len, old, out, store);
        skew_change_step::<2>(acc, xp, yp, m_full, old, new);
        skew_finish_group::<3>(acc, xp, yp, len, old, out, store);
        skew_change_step::<3>(acc, xp, yp, m_full, old, new);
        skew_finish_group::<4>(acc, xp, yp, len, old, out, store);
        skew_change_step::<4>(acc, xp, yp, m_full, old, new);
        skew_finish_group::<5>(acc, xp, yp, len, old, out, store);
        skew_change_step::<5>(acc, xp, yp, m_full, old, new);
        skew_finish_group::<6>(acc, xp, yp, len, old, out, store);
        skew_change_step::<6>(acc, xp, yp, m_full, old, new);
        // Group 7 ran its last block of `old` in step 6.
        skew_finish_group::<7>(acc, xp, yp, len, old, out, store);
    }
}

/// One steady block `m` of the pass at lag `o`, with intrinsics.
///
/// # Safety
///
/// As `xcorr_skewed_neon` explains for its pointers.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn skew_block(acc: &mut SkewSums, xp: *const f32, yp: *const f32, o: usize, m: usize) {
    use core::arch::aarch64::*;
    // SAFETY: see `xcorr_skewed_neon`.
    unsafe {
        let b = yp.add(o + 4 * m);
        let lo: [float32x2_t; 8] = core::array::from_fn(|g| vld1_f32(xp.add(4 * (m - g))));
        let hi: [float32x2_t; 8] = core::array::from_fn(|g| vld1_f32(xp.add(4 * (m - g) + 2)));
        macro_rules! step {
            ($t:literal, $x:ident, $lane:literal) => {{
                let (w, v) = (vld1q_f32(b.add($t)), vld1q_f32(b.add(32 + $t)));
                for g in 0..8 {
                    acc[g] = vfmaq_lane_f32::<$lane>(acc[g], w, $x[g]);
                    acc[8 + g] = vfmaq_lane_f32::<$lane>(acc[8 + g], v, $x[g]);
                }
            }};
        }
        step!(0, lo, 0);
        step!(1, lo, 1);
        step!(2, hi, 0);
        step!(3, hi, 1);
    }
}

/// The steady blocks 7 to m_full - 1 of the pass at lag `o`. Pairs of blocks
/// run in assembly, `SKEW_PAIRS_PER_CALL` pairs per call, and `side` runs
/// after each call. The last one or two blocks run with intrinsics, so that
/// the assembly never loads x past the last block.
///
/// # Safety
///
/// As `xcorr_skewed_neon` explains for its pointers.
#[cfg(target_arch = "aarch64")]
#[inline(never)]
unsafe fn skew_steady<F: FnMut()>(
    acc: &mut SkewSums,
    xp: *const f32,
    yp: *const f32,
    m_full: usize,
    o: usize,
    side: &mut F,
) {
    use core::arch::aarch64::*;
    // SAFETY: see `xcorr_skewed_neon`.
    unsafe {
        let mut m = 7;
        let mut pairs = (m_full - SKEW_MIN_BLOCKS) / 2;
        while pairs > 0 {
            let count = pairs.min(SKEW_PAIRS_PER_CALL);
            let mut xm = xp.add(4 * m);
            let mut ym = yp.add(o + 4 * m);
            let mut n = count;
            let lo: [float32x2_t; 8] = core::array::from_fn(|g| vld1_f32(xp.add(4 * (m - g))));
            let ycur = vld1q_f32(ym);
            let zcur = vld1q_f32(ym.add(32));
            core::arch::asm!(
                "2:",
                skew_block!(16, 17, 18, 19),
                skew_block!(18, 19, 16, 17),
                "subs {n}, {n}, #1",
                "b.ne 2b",
                xm = inout(reg) xm,
                ym = inout(reg) ym,
                n = inout(reg) n,
                inout("v0") acc[0],
                inout("v1") acc[1],
                inout("v2") acc[2],
                inout("v3") acc[3],
                inout("v4") acc[4],
                inout("v5") acc[5],
                inout("v6") acc[6],
                inout("v7") acc[7],
                inout("v8") acc[8],
                inout("v9") acc[9],
                inout("v10") acc[10],
                inout("v11") acc[11],
                inout("v12") acc[12],
                inout("v13") acc[13],
                inout("v14") acc[14],
                inout("v15") acc[15],
                inout("v16") ycur => _,
                inout("v17") zcur => _,
                out("v18") _,
                out("v19") _,
                out("v20") _,
                out("v21") _,
                out("v22") _,
                out("v23") _,
                inout("v24") lo[0] => _,
                inout("v25") lo[1] => _,
                inout("v26") lo[2] => _,
                inout("v27") lo[3] => _,
                inout("v28") lo[4] => _,
                inout("v29") lo[5] => _,
                inout("v30") lo[6] => _,
                inout("v31") lo[7] => _,
                options(nostack, readonly),
            );
            let _ = (xm, ym, n);
            m += 2 * count;
            pairs -= count;
            side();
        }
        while m < m_full {
            skew_block(acc, xp, yp, o, m);
            m += 1;
        }
    }
}

/// How many times `xcorr_skewed_neon` calls `side` for `len` samples and
/// `lags` lags.
#[cfg(target_arch = "aarch64")]
fn skew_side_calls(len: usize, lags: usize) -> usize {
    let pairs = (len / 4).saturating_sub(SKEW_MIN_BLOCKS) / 2;
    lags.div_ceil(SKEW_PASS) * pairs.div_ceil(SKEW_PAIRS_PER_CALL)
}

/// Cross-correlation in the by-sample order with windows shared between
/// groups of lags (see above): `out[k] = sum over j < len of x[j] * y[j + k]`
/// for every `k < out.len()`, with `len = x.len()`. The result has the same
/// bits as `xcorr_by_sample_neon`.
///
/// `side` runs `skew_side_calls(len, out.len())` times, between the calls of
/// the assembly loop.
///
/// Panics unless `out.len()` is a multiple of 4 and at least 64, `x` holds
/// `4 * SKEW_MIN_BLOCKS` samples or more, and `y` holds
/// `len + out.len() - 1`.
#[cfg(target_arch = "aarch64")]
fn xcorr_skewed_neon<F: FnMut()>(x: &[f32], y: &[f32], out: &mut [f32], side: &mut F) {
    use core::arch::aarch64::*;

    let len = x.len();
    let lags = out.len();
    let m_full = len / 4;
    assert!(
        lags.is_multiple_of(4)
            && lags >= SKEW_PASS
            && m_full >= SKEW_MIN_BLOCKS
            && y.len() >= len + lags - 1
    );
    let passes = lags.div_ceil(SKEW_PASS);
    // The last pass ends at the last lag; it computes some lags of the pass
    // before it again, with the same result.
    let start = |p: usize| (SKEW_PASS * p).min(lags - SKEW_PASS);
    // SAFETY: with L = lags, M = m_full and a pass at lag o <= L - 64, the
    // highest index each load reads is:
    // - windows of steady block m < M (assembly and intrinsics): o + 4m + 38
    //   <= L + 4M - 30; the assembly loads the windows of block m + 2 at
    //   most, and it runs only while one more block is left after it.
    // - windows of block I < 7 of a pass: o + 4I + 38 <= L - 2; windows of
    //   block M + I: o + 4(M + I) + 38 <= L + 4M - 2.
    // - the last samples of a group: o + 32 + 28 + (len - 1) + 3 = o + len + 62
    //   <= L + len - 2.
    // All are at most len + L - 2, the last index of `y`. x is read at
    // 4(b - g) to 4(b - g) + 3 for the block b - g of group g, which is
    // between 0 and M - 1, so below len; the last samples at j < len.
    // Stores write out[o..o + 64] with o + 64 <= L.
    unsafe {
        let (xp, yp, op) = (x.as_ptr(), y.as_ptr(), out.as_mut_ptr());
        let mut acc: SkewSums = [vdupq_n_f32(0.0); 16];
        for p in 0..passes {
            // When the first pass starts, the groups that have not started
            // run a copy of it whose sums are thrown away.
            let old = start(p.saturating_sub(1));
            skew_change(&mut acc, xp, yp, len, old, start(p), op, p > 0);
            skew_steady(&mut acc, xp, yp, m_full, start(p), side);
        }
        // The last pass ends; the groups that are done run a copy of it.
        let last = start(passes - 1);
        skew_change(&mut acc, xp, yp, len, last, last, op, true);
    }
}

/// `pitch_xcorr` for aarch64, libopus `celt_pitch_xcorr_float_neon`:
/// `xcorr[k] = sum over j of x[j] * y[j + k]` with `len = x.len()` and one lag
/// per element of `xcorr`.
///
/// The lags below the largest multiple of four are summed by sample, the
/// remaining one to three with `celt_inner_prod`.
#[cfg(target_arch = "aarch64")]
fn pitch_xcorr_neon(x: &[f32], y: &[f32], xcorr: &mut [f32]) {
    let len = x.len();
    let lags = xcorr.len();
    if lags == 0 {
        return;
    }
    // The last lag reads y[lags - 1..lags - 1 + len].
    assert!(len > 0 && y.len() >= len + lags - 1);

    if lags == 5 {
        // The autocorrelation of `pitch_downsample`.
        xcorr5_neon(x, y, xcorr);
        return;
    }
    let by_sample = lags & !3;
    let (grouped, rest) = xcorr.split_at_mut(by_sample);
    if USE_SKEW && by_sample >= SKEW_PASS && len >= 4 * SKEW_MIN_BLOCKS {
        xcorr_skewed_neon(x, y, grouped, &mut || {});
    } else {
        xcorr_by_sample_groups_neon(x, y, grouped);
    }

    // The one to three lags after the last group of four.
    let mut offsets = [0usize; 3];
    for (k, off) in offsets[..rest.len()].iter_mut().enumerate() {
        *off = by_sample + k;
    }
    inner_prod_many(x, y, &offsets[..rest.len()], len, rest);
}

/// `xcorr_by_sample_neon` for `out.len()` lags, a multiple of four: one call
/// for up to 28 lags, passes of 32 or 64 lags for more.
#[cfg(target_arch = "aarch64")]
fn xcorr_by_sample_groups_neon(x: &[f32], y: &[f32], grouped: &mut [f32]) {
    match grouped.len() / 4 {
        0 => {}
        1 => xcorr_by_sample_neon::<1>(x, y, grouped),
        2 => xcorr_by_sample_neon::<2>(x, y, grouped),
        3 => xcorr_by_sample_neon::<3>(x, y, grouped),
        4 => xcorr_by_sample_neon::<4>(x, y, grouped),
        5 => xcorr_by_sample_neon::<5>(x, y, grouped),
        6 => xcorr_by_sample_neon::<6>(x, y, grouped),
        7 => xcorr_by_sample_neon::<7>(x, y, grouped),
        8..16 => xcorr_by_sample_passes_neon::<8>(x, y, grouped),
        _ => xcorr_by_sample_passes_neon::<16>(x, y, grouped),
    }
}

/// `pitch_xcorr_neon` for five lags, as the autocorrelation of
/// `pitch_downsample` asks for: lags 0 to 3 by sample and lag 4 as
/// `celt_inner_prod`, in one loop.
///
/// The four lags by sample are one chain of fused multiply-adds, one per
/// sample, and the loop waits for each of them. The inner product of lag 4
/// does not depend on that chain, so in the same loop it costs almost no
/// time. Computed after the chain, as `pitch_xcorr_neon` does for other
/// counts of lags, it took a quarter as long again.
///
/// `len` is `x.len()`. Panics unless `y` holds `len + 4` samples and `xcorr`
/// holds 5 lags.
#[cfg(target_arch = "aarch64")]
fn xcorr5_neon(x: &[f32], y: &[f32], xcorr: &mut [f32]) {
    use core::arch::aarch64::*;

    let len = x.len();
    let y = &y[..len + 4];
    let xcorr = &mut xcorr[..5];
    // SAFETY: `x` holds `len` samples and `y` len + 4 (sliced above). The
    // four-sample loop reads x[j..j + 4] and y[j + t..][..4] for t <= 4, with
    // j + 4 <= len, which ends at y[j + 7] <= y[len + 3]. The one-sample loop
    // reads x[j] and y[j..j + 4] with j < len, which ends at y[len + 2].
    unsafe {
        let (xp, yp) = (x.as_ptr(), y.as_ptr());
        let mut by_sample = vdupq_n_f32(0.0);
        let mut lag4 = vdupq_n_f32(0.0);
        let mut j = 0;
        while j + 4 <= len {
            let xv = vld1q_f32(xp.add(j));
            let w = yp.add(j);
            let mut s = by_sample;
            s = vfmaq_laneq_f32::<0>(s, vld1q_f32(w), xv);
            s = vfmaq_laneq_f32::<1>(s, vld1q_f32(w.add(1)), xv);
            s = vfmaq_laneq_f32::<2>(s, vld1q_f32(w.add(2)), xv);
            s = vfmaq_laneq_f32::<3>(s, vld1q_f32(w.add(3)), xv);
            by_sample = s;
            lag4 = vfmaq_f32(lag4, xv, vld1q_f32(w.add(4)));
            j += 4;
        }
        let mut lag4 = lanes_sum(lag4);
        while j < len {
            let xj = *xp.add(j);
            by_sample = vfmaq_f32(by_sample, vld1q_f32(yp.add(j)), vdupq_n_f32(xj));
            lag4 = mac(lag4, xj, *yp.add(j + 4));
            j += 1;
        }
        vst1q_f32(xcorr.as_mut_ptr(), by_sample);
        xcorr[4] = lag4;
    }
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
    assert!(x.len() >= n && y.len() >= n);

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
    assert!(x.len() >= n && y1.len() >= n && y2.len() >= n);

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
    // The loads read `x[..len]` and `y[..len + 3]`.
    assert!(x.len() >= len && y.len() >= len + 3);

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
    // Lag `i` reads `y[i..i + len + 3]` for `i + 4 <= max_pitch`.
    assert!(x.len() >= len && y.len() + 1 >= len + max_pitch);
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

    assert!(x.len() >= n && y.len() >= n);

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

    assert!(x.len() >= n && y1.len() >= n && y2.len() >= n);

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
    // Lag `i` reads `y[i..i + len + 3]` for `i + 4 <= max_pitch`.
    assert!(x.len() >= len && y.len() + 1 >= len + max_pitch);
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

    // The loads read `x[..len]` and `y[..len + 3]`.
    assert!(x.len() >= len && y.len() >= len + 3);

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

/// libopus `pitch_downsample` (celt/pitch.c) of the float build: decimates
/// the `c` channels of `x` by `factor` with a low-pass filter, sums them into
/// `x_lp[..len]`, and then flattens the spectrum of the result.
///
/// For a factor of 2, output `i` is
/// `0.25 * x[2i - 1] + 0.25 * x[2i + 1] + 0.5 * x[2i]` per channel. A channel
/// shorter than `factor * len` contributes nothing to the outputs whose
/// samples it lacks; libopus is never called with such a channel. The
/// flattening is a fourth-order linear prediction of `x_lp` and a filter with
/// five taps built from it: the pitch search then follows the pitch and not
/// the strongest formant.
pub fn pitch_downsample(x: &[&[f32]], x_lp: &mut [f32], len: usize, c: usize, factor: usize) {
    let offset = factor / 2;

    if x_lp.len() < len || len == 0 {
        return;
    }

    #[cfg(target_arch = "aarch64")]
    if factor == 2
        && (c == 1 || c == 2)
        && len >= 8
        && x[..c].iter().all(|x_k| x_k.len() >= 2 * len)
    {
        let mut ac = [0.0f32; 5];
        if c == 1 {
            decimate2_autocorr5_neon([x[0]], &mut x_lp[..len], &mut ac);
        } else {
            decimate2_autocorr5_neon([x[0], x[1]], &mut x_lp[..len], &mut ac);
        }
        flatten_with(&mut x_lp[..len], ac);
        return;
    }

    // Outputs below `full` have all three inputs in every channel. For a
    // decimation by two of one or two channels they are computed without a
    // test per sample. The encoder and the concealment pass channels of
    // exactly `2 * len` samples, so that is every output.
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

    // The general case, and the outputs near the end of a short channel.
    for i in full..len {
        let mut val = 0.0f32;
        for (k, x_k) in x[..c].iter().enumerate() {
            let idx_p = factor * i + offset;
            if idx_p < x_k.len() {
                let v = lowpass3(x_k[factor * i - offset], x_k[factor * i], x_k[idx_p]);
                val = if k == 0 { v } else { val + v };
            }
        }
        x_lp[i] = val;
    }

    // Output 0 has no sample before it: 0.25 * x[offset] + 0.5 * x[0].
    let mut val = 0.0f32;
    for (k, x_k) in x[..c].iter().enumerate() {
        if offset < x_k.len() {
            let v = mac(0.5 * x_k[0], x_k[offset], 0.25);
            val = if k == 0 { v } else { val + v };
        }
    }
    x_lp[0] = val;

    flatten_spectrum(&mut x_lp[..len]);
}

/// `0.25 * before + 0.25 * after + 0.5 * centre`, rounded as libopus rounds
/// that line of `pitch_downsample`: the product with `after` first, then the
/// other two added with `mac` (clang turns them into fused multiply-adds on
/// 64-bit ARM). The products with 0.25 and 0.5 are exact unless they are
/// subnormal numbers.
#[inline(always)]
fn lowpass3(before: f32, centre: f32, after: f32) -> f32 {
    mac(mac(0.25 * after, before, 0.25), centre, 0.5)
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
    flatten_with(x_lp, ac);
}

/// `flatten_spectrum` from the autocorrelation `ac` of `x_lp` at the lags 0
/// to 4.
fn flatten_with(x_lp: &mut [f32], mut ac: [f32; 5]) {
    // "Noise floor -40 dB"
    ac[0] *= 1.0001;
    // "Lag windowing": C `ac[i] -= ac[i]*(.008f*i)*(.008f*i)`.
    for (i, a) in ac.iter_mut().enumerate().skip(1) {
        let f = 0.008 * i as f32;
        *a = mac(*a, -(*a * f), f);
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
        mac(coef[1], c1, coef[0]),
        mac(coef[2], c1, coef[1]),
        mac(coef[3], c1, coef[2]),
        c1 * coef[3],
    ];
    celt_fir5(x_lp, &lpc2);
}

/// Filter with five feed-forward taps, in place (libopus `celt_fir5`):
/// `x[i] += num[0] * x[i - 1] + .. + num[4] * x[i - 5]`, with the samples
/// before `x[0]` taken as zero and the old values of `x` on the right. The
/// products are added in that order, each with `mac`.
fn celt_fir5(x: &mut [f32], num: &[f32; 5]) {
    #[cfg(target_arch = "aarch64")]
    celt_fir5_neon(x, num);
    #[cfg(not(target_arch = "aarch64"))]
    celt_fir5_scalar(x, num, [0.0; 5]);
}

/// Portable version of `celt_fir5`, starting from the history `mem`
/// (`mem[k]` is the input `k + 1` samples before `x[0]`). It is also the
/// tail of `celt_fir5_neon` and its reference in the tests.
#[inline(always)]
fn celt_fir5_scalar(x: &mut [f32], num: &[f32; 5], mut mem: [f32; 5]) {
    for v in x.iter_mut() {
        let mut sum = *v;
        for (&n, &m) in num.iter().zip(&mem) {
            sum = mac(sum, n, m);
        }
        mem = [*v, mem[0], mem[1], mem[2], mem[3]];
        *v = sum;
    }
}

/// NEON version of `celt_fir5`: four outputs per block, each with the same
/// five fused multiply-adds in the same order as `celt_fir5_scalar`.
///
/// The five multiply-adds of a block wait for each other. The main loop
/// therefore takes four blocks at a time and does the first multiply-add of
/// all four, then the second of all four, and so on. A core that runs the
/// instructions in program order, such as the Cortex-A55, then has three
/// other multiply-adds to issue while each one finishes.
#[cfg(target_arch = "aarch64")]
fn celt_fir5_neon(x: &mut [f32], num: &[f32; 5]) {
    use core::arch::aarch64::*;

    let n = x.len();
    let blocks = n / 4;
    // SAFETY: block `b` reads and writes x[4b..4b + 4] with 4b + 4 <= n. The
    // inputs of the samples before a block are kept in registers, and a step
    // loads all its blocks before it stores any, so every load reads an
    // input that has not been overwritten yet.
    unsafe {
        let p = x.as_mut_ptr();
        let taps = [
            vdupq_n_f32(num[0]),
            vdupq_n_f32(num[1]),
            vdupq_n_f32(num[2]),
            vdupq_n_f32(num[3]),
            vdupq_n_f32(num[4]),
        ];
        // x[i - 1 - k] for the four outputs i of the block `cur`, for k from
        // 0 to 4. `prev` and `prev2` are the inputs of the two blocks before.
        let delayed = |prev2: float32x4_t, prev: float32x4_t, cur: float32x4_t| {
            [
                vextq_f32::<3>(prev, cur),
                vextq_f32::<2>(prev, cur),
                vextq_f32::<1>(prev, cur),
                prev,
                vextq_f32::<3>(prev2, prev),
            ]
        };
        // The inputs of the two blocks before this one; zero before x[0].
        let mut prev2 = vdupq_n_f32(0.0);
        let mut prev = vdupq_n_f32(0.0);
        let mut b = 0;
        while b + 4 <= blocks {
            let cur: [float32x4_t; 4] = core::array::from_fn(|k| vld1q_f32(p.add(4 * (b + k))));
            let m = [
                delayed(prev2, prev, cur[0]),
                delayed(prev, cur[0], cur[1]),
                delayed(cur[0], cur[1], cur[2]),
                delayed(cur[1], cur[2], cur[3]),
            ];
            let mut sum = cur;
            for (tap, &t) in taps.iter().enumerate() {
                for k in 0..4 {
                    sum[k] = vfmaq_f32(sum[k], t, m[k][tap]);
                }
            }
            for (k, s) in sum.iter().enumerate() {
                vst1q_f32(p.add(4 * (b + k)), *s);
            }
            prev2 = cur[2];
            prev = cur[3];
            b += 4;
        }
        while b < blocks {
            let cur = vld1q_f32(p.add(4 * b));
            let m = delayed(prev2, prev, cur);
            let mut sum = cur;
            for (&t, &d) in taps.iter().zip(&m) {
                sum = vfmaq_f32(sum, t, d);
            }
            vst1q_f32(p.add(4 * b), sum);
            prev2 = prev;
            prev = cur;
            b += 1;
        }
        // The history of the last samples: the five inputs before x[4 * blocks].
        let mem = [
            vgetq_lane_f32::<3>(prev),
            vgetq_lane_f32::<2>(prev),
            vgetq_lane_f32::<1>(prev),
            vgetq_lane_f32::<0>(prev),
            vgetq_lane_f32::<3>(prev2),
        ];
        celt_fir5_scalar(&mut x[4 * blocks..], num, mem);
    }
}

/// Decimation by two with the low-pass filter (0.25, 0.5, 0.25), summed over
/// `C` channels: `out[i]` is the sum over the channels of `lowpass3` of
/// x[2i - 1], x[2i] and x[2i + 1], for `i` from 1 to `out.len() - 1`.
///
/// The first channel's value is stored, the others are added to it in
/// order. `out[0]` is not written. Every channel must hold at least
/// `2 * out.len()` samples.
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
            let val = lowpass3(b[0], a[0], a[1]);
            *o = if k == 0 { val } else { *o + val };
        }
    }
}

/// NEON version of `decimate2`: four outputs per pass, with the same
/// operations per output as `decimate2_scalar`.
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
            // x[2i + 1] of the pass before. Its last lane is x[2i - 1] for
            // the first output of this pass.
            let mut odd_prev = [vdupq_n_f32(0.0); C];
            for (o, x_k) in odd_prev.iter_mut().zip(&x) {
                *o = vld1q_dup_f32(x_k.as_ptr().add(1));
            }
            while i + 4 <= n {
                let mut val = vdupq_n_f32(0.0);
                for (k, (o, x_k)) in odd_prev.iter_mut().zip(&x).enumerate() {
                    // .0 = x[2i], x[2i + 2], ..; .1 = x[2i + 1], x[2i + 3], ..
                    let pair = vld2q_f32(x_k.as_ptr().add(2 * i));
                    // x[2i - 1], x[2i + 1], x[2i + 3], x[2i + 5]
                    let before = vextq_f32::<3>(*o, pair.1);
                    let v = vfmaq_f32(
                        vfmaq_f32(vmulq_f32(pair.1, quarter), before, quarter),
                        pair.0,
                        half,
                    );
                    val = if k == 0 { v } else { vaddq_f32(val, v) };
                    *o = pair.1;
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

/// `decimate2` of all of `x_lp` together with libopus's autocorrelation of
/// the result at the lags 0 to 4 (`_celt_autocorr(x_lp, ac, NULL, 0, 4, len)`),
/// in one loop.
///
/// The autocorrelation is one chain of fused multiply-adds, one per sample
/// (see `xcorr5_neon`), and its loop waits for each of them. The decimation
/// does not depend on that chain, so in the same loop it costs almost no
/// time. The outputs are computed four at a time, and the windows of the
/// autocorrelation are built from them in registers.
///
/// `x_lp[0]` is libopus's first output, `0.25 * x[1] + 0.5 * x[0]` per
/// channel. Panics unless `x_lp` holds 8 samples or more and every channel
/// `2 * x_lp.len()`.
#[cfg(target_arch = "aarch64")]
fn decimate2_autocorr5_neon<const C: usize>(x: [&[f32]; C], x_lp: &mut [f32], ac: &mut [f32; 5]) {
    use core::arch::aarch64::*;

    let n = x_lp.len();
    assert!(n >= 8 && x.iter().all(|x_k| x_k.len() >= 2 * n));
    let blocks = n / 4;
    let out = x_lp.as_mut_ptr();

    // SAFETY: block `m < blocks` reads x_k[8m..8m + 8] with 8m + 8 <= 2n and
    // writes x_lp[4m..4m + 4] with 4m + 4 <= n. Before the loop x_k[0] and
    // x_k[1] are read; every channel holds at least 16 samples.
    unsafe {
        let quarter = vdupq_n_f32(0.25);
        let half = vdupq_n_f32(0.5);
        // x[8m - 1] in lane 3, for the first output of block m.
        let mut odd_prev = [vdupq_n_f32(0.0); C];
        let block = |m: usize, odd_prev: &mut [float32x4_t; C]| {
            let mut val = vdupq_n_f32(0.0);
            for (k, (o, x_k)) in odd_prev.iter_mut().zip(&x).enumerate() {
                // .0 = x[8m], x[8m + 2], ..; .1 = x[8m + 1], x[8m + 3], ..
                let pair = vld2q_f32(x_k.as_ptr().add(8 * m));
                let before = vextq_f32::<3>(*o, pair.1);
                let v = vfmaq_f32(
                    vfmaq_f32(vmulq_f32(pair.1, quarter), before, quarter),
                    pair.0,
                    half,
                );
                val = if k == 0 { v } else { vaddq_f32(val, v) };
                *o = pair.1;
            }
            val
        };

        // Block 0, whose first output has no sample before it.
        let mut first = 0.0f32;
        for (k, x_k) in x.iter().enumerate() {
            let v = mac(0.5 * x_k[0], x_k[1], 0.25);
            first = if k == 0 { v } else { first + v };
        }
        let mut cur = vsetq_lane_f32::<0>(first, block(0, &mut odd_prev));
        vst1q_f32(out, cur);

        // Step `m` adds the products of the samples 4(m - 1) to 4m - 1 to
        // the autocorrelation: lags 0 to 3 by sample, lag 4 in four lanes as
        // `celt_inner_prod`. The last of these samples is fast_n - 1 = n - 5
        // or before. It needs the blocks m - 1 (`cur`) and m (`next`).
        //
        // The four multiply-adds of lags 0 to 3 wait for each other. So step
        // `m` also decimates block m + 1, which the step after it needs: that
        // work does not depend on the chain, and a core that runs the
        // instructions in program order can do it while the chain waits.
        let mut by_sample = vdupq_n_f32(0.0);
        let mut lag4 = vdupq_n_f32(0.0);
        let step = |by_sample: &mut float32x4_t,
                    lag4: &mut float32x4_t,
                    cur: float32x4_t,
                    next: float32x4_t| {
            let mut s = *by_sample;
            s = vfmaq_laneq_f32::<0>(s, cur, cur);
            s = vfmaq_laneq_f32::<1>(s, vextq_f32::<1>(cur, next), cur);
            s = vfmaq_laneq_f32::<2>(s, vextq_f32::<2>(cur, next), cur);
            s = vfmaq_laneq_f32::<3>(s, vextq_f32::<3>(cur, next), cur);
            *by_sample = s;
            *lag4 = vfmaq_f32(*lag4, cur, next);
        };
        // `blocks` is 2 or more, because n >= 8.
        let mut next = block(1, &mut odd_prev);
        vst1q_f32(out.add(4), next);
        for m in 1..blocks - 1 {
            let after = block(m + 1, &mut odd_prev);
            step(&mut by_sample, &mut lag4, cur, next);
            vst1q_f32(out.add(4 * (m + 1)), after);
            cur = next;
            next = after;
        }
        step(&mut by_sample, &mut lag4, cur, next);

        // The last n % 4 outputs.
        for i in 4 * blocks..n {
            let mut val = 0.0f32;
            for (k, x_k) in x.iter().enumerate() {
                let v = lowpass3(x_k[2 * i - 1], x_k[2 * i], x_k[2 * i + 1]);
                val = if k == 0 { v } else { val + v };
            }
            x_lp[i] = val;
        }

        // The samples of the autocorrelation after the vector steps: from
        // 4 * (blocks - 1) to fast_n - 1, one at a time.
        let fast_n = n - 4;
        let mut lag4 = lanes_sum(lag4);
        for j in 4 * (blocks - 1)..fast_n {
            let xj = x_lp[j];
            by_sample = vfmaq_f32(by_sample, vld1q_f32(x_lp.as_ptr().add(j)), vdupq_n_f32(xj));
            lag4 = mac(lag4, xj, x_lp[j + 4]);
        }
        vst1q_f32(ac.as_mut_ptr(), by_sample);
        ac[4] = lag4;
    }

    // The last products of each lag, as in `celt_lpc::autocorr`.
    let fast_n = n - 4;
    for (k, out) in ac.iter_mut().enumerate() {
        let mut d = 0.0f32;
        for (&a, &b) in x_lp[k + fast_n..].iter().zip(&x_lp[fast_n..]) {
            d = mac(d, a, b);
        }
        *out += d;
    }
}

/// The two best lags of libopus `find_best_pitch` (float build), and the
/// normalised correlations they won with.
///
/// A lag is compared as `xcorr^2 / energy`, without the division: lag `i`
/// beats a kept lag when `num_i * den_kept > num_kept * syy_i`.
struct BestPitch {
    num: [f32; 2],
    den: [f32; 2],
    pitch: [usize; 2],
}

impl BestPitch {
    fn new() -> Self {
        Self {
            num: [-1.0, -1.0],
            den: [0.0, 0.0],
            pitch: [0, 1],
        }
    }

    /// The second half of `consider`, for lag `i` with the scaled square
    /// `num` of its correlation, which beats the second best lag: it
    /// becomes the best or the second best.
    #[inline(always)]
    fn take(&mut self, i: usize, num: f32, syy: f32) {
        if num * self.den[0] > self.num[0] * syy {
            self.num[1] = self.num[0];
            self.den[1] = self.den[0];
            self.pitch[1] = self.pitch[0];
            self.num[0] = num;
            self.den[0] = syy;
            self.pitch[0] = i;
        } else {
            self.num[1] = num;
            self.den[1] = syy;
            self.pitch[1] = i;
        }
    }

    /// One step of the loop of `find_best_pitch`: lag `i` with the
    /// correlation `xc` and the window energy `syy`.
    #[inline(always)]
    fn consider(&mut self, i: usize, xc: f32, syy: f32) {
        if xc > 0.0 {
            // libopus scales the correlation before it squares it ("this
            // should avoid both underflows and overflows (inf) when squaring
            // xcorr16", celt/pitch.c, find_best_pitch).
            let xc16 = xc * 1e-12;
            let num = xc16 * xc16;
            if num * self.den[1] > self.num[1] * syy {
                self.take(i, num, syy);
            }
        }
    }
}

/// libopus `find_best_pitch` of the float build: the two lags with the
/// highest normalised correlation, for the lags `i < xcorr.len()`.
///
/// `syy[i]` is the window energy of lag `i` (`WindowEnergies`).
///
/// Most lags change nothing. Their test does not depend on the lags before
/// them, so on aarch64 eight lags are tested at a time against the second
/// best lag as it is before them. The lanes before the first lane that
/// beats it change nothing. That lane is taken, and the lanes after it are
/// tested again against the new second best lag. The products and
/// comparisons are those of the loop, so the result is the same.
fn best_pitch_from(xcorr: &[f32], syy: &[f32], best_pitch: &mut [usize; 2]) {
    let n = xcorr.len();
    let syy = &syy[..n];
    let mut best = BestPitch::new();
    // Only the aarch64 code moves `i` on.
    #[allow(unused_mut)]
    let mut i = 0;
    #[cfg(target_arch = "aarch64")]
    {
        use core::arch::aarch64::*;
        // SAFETY: the loads read xcorr[i..i + 8] and syy[i..i + 8] with
        // i + 8 <= n; both slices hold `n` values.
        unsafe {
            let (xp, sp) = (xcorr.as_ptr(), syy.as_ptr());
            let zero = vdupq_n_f32(0.0);
            // Bit `l` of the mask of lane `l` of the two vectors.
            let bits = [
                vld1q_u32([1u32, 2, 4, 8].as_ptr()),
                vld1q_u32([16u32, 32, 64, 128].as_ptr()),
            ];
            while i + 8 <= n {
                let xc = [vld1q_f32(xp.add(i)), vld1q_f32(xp.add(i + 4))];
                let s = [vld1q_f32(sp.add(i)), vld1q_f32(sp.add(i + 4))];
                let num = xc.map(|v| {
                    let xc16 = vmulq_n_f32(v, 1e-12);
                    vmulq_f32(xc16, xc16)
                });
                let positive = [vcgtq_f32(xc[0], zero), vcgtq_f32(xc[1], zero)];
                // The lanes from `from` on that beat the second best lag as
                // it is now. Lanes before the first of them change nothing.
                let mut from = 0;
                loop {
                    let mut mask = 0u32;
                    for h in 0..2 {
                        let beats = vcgtq_f32(
                            vmulq_n_f32(num[h], best.den[1]),
                            vmulq_n_f32(s[h], best.num[1]),
                        );
                        mask |= vaddvq_u32(vandq_u32(vandq_u32(beats, positive[h]), bits[h]));
                    }
                    mask &= !0u32 << from;
                    if mask == 0 {
                        break;
                    }
                    let lane = mask.trailing_zeros() as usize;
                    let (h, l) = (lane / 4, lane % 4);
                    let mut nums = [0.0f32; 4];
                    vst1q_f32(nums.as_mut_ptr(), num[h]);
                    best.take(i + lane, nums[l], syy[i + lane]);
                    from = lane + 1;
                }
                i += 8;
            }
        }
    }
    for k in i..n {
        best.consider(k, xcorr[k], syy[k]);
    }
    *best_pitch = best.pitch;
}

/// C `Syy` of libopus `find_best_pitch` at every lag, filled in parts.
///
/// `table[i]` is the energy of the window y[i..i + len] plus one, as libopus
/// carries it from lag to lag. Entry 0 is the energy of the first window
/// plus one, summed one sample after the other (`Energy`). Each step then
/// adds the square of the sample that enters the window minus the square of
/// the one that leaves it (C `Syy += y[i+len]*y[i+len] - y[i]*y[i]`; clang
/// fuses the first of the two products into the subtraction). The energy is
/// held at 1 or more.
///
/// The entries are one chain of additions that wait for each other. The
/// increments do not depend on the chain, so they are computed first, four
/// at a time. The chain is then one addition per step. The lower limit is
/// almost never reached: the chain runs without it, and when any value of
/// the chain is below 1, the entries are computed again with it. Until the
/// first value below 1 the two chains are the same, so the result is too.
///
/// The table is not cleared before use: the entries below `next` hold
/// values, and the others are written before they are read.
struct WindowEnergies<'a> {
    /// y[..n]: the sample that leaves the window at each lag.
    leaving: &'a [f32],
    /// y[len..len + n]: the sample that enters it.
    entering: &'a [f32],
    /// C `Syy` at the lag `next`.
    syy: f32,
    /// The next entry to fill. The entries below it are filled.
    next: usize,
    table: &'a mut [MaybeUninit<f32>],
}

impl<'a> WindowEnergies<'a> {
    /// The table of `table.len()` lags of `y`, with `len` samples per window
    /// and `first` the energy of the first window plus one. `y` must hold
    /// `len + table.len()` samples.
    fn new(y: &'a [f32], len: usize, first: f32, table: &'a mut [MaybeUninit<f32>]) -> Self {
        let n = table.len();
        Self {
            leaving: &y[..n],
            entering: &y[len..len + n],
            syy: first,
            next: 0,
            table,
        }
    }

    /// Fills the entries up to `end` one step after the other, with the
    /// lower limit at each step, as libopus does.
    fn fill_exact(&mut self, end: usize) {
        for i in self.next..end {
            self.table[i].write(self.syy);
            let (old, new) = (self.leaving[i], self.entering[i]);
            self.syy += mac(-(old * old), new, new);
            if self.syy < 1.0 {
                self.syy = 1.0;
            }
        }
        self.next = end;
    }

    /// Writes the increment of each step from `next` to `end` into the
    /// table entry of that step: `new^2 - old^2`, with the first product
    /// fused into the subtraction (`mac`).
    fn increments(&mut self, end: usize) {
        let range = self.next..end;
        let (old, new) = (&self.leaving[range.clone()], &self.entering[range.clone()]);
        let out = &mut self.table[range];
        // Only the aarch64 code moves `done` on.
        #[allow(unused_mut)]
        let mut done = 0;
        #[cfg(target_arch = "aarch64")]
        {
            use core::arch::aarch64::*;
            let n = out.len();
            // SAFETY: the three slices hold `n` values each; the loads and
            // the store touch [i, i + 4) with i + 4 <= n.
            unsafe {
                while done + 4 <= n {
                    let o = vld1q_f32(old.as_ptr().add(done));
                    let w = vld1q_f32(new.as_ptr().add(done));
                    let d = vfmaq_f32(vnegq_f32(vmulq_f32(o, o)), w, w);
                    vst1q_f32(out.as_mut_ptr().add(done).cast(), d);
                    done += 4;
                }
            }
        }
        for ((d, &o), &w) in out[done..].iter_mut().zip(&old[done..]).zip(&new[done..]) {
            d.write(mac(-(o * o), w, w));
        }
    }

    /// The filled entries, `table[..next]`.
    fn filled(&self) -> &[f32] {
        // SAFETY: the entries below `next` hold values: `fill_exact` writes
        // each entry before it moves `next` past it, and `fill_to` and
        // `fill_both` move it only over entries that `increments` wrote.
        unsafe { assume_written(&self.table[..self.next]) }
    }

    /// Fills the entries up to `end`, or does nothing when they are filled.
    /// Every `B::EVERY` steps also fill one entry of `beside`.
    fn fill_to<B: Beside>(&mut self, end: usize, beside: &mut B) {
        if end <= self.next {
            return;
        }
        let (start, first) = (self.next, self.syy);
        self.increments(end);
        // SAFETY: `increments` wrote every entry from `start` to `end`.
        let table = unsafe { assume_written_mut(&mut self.table[start..end]) };
        let mut s = first;
        for part in table.chunks_mut(B::EVERY) {
            for t in part {
                let d = *t;
                *t = s;
                s += d;
            }
            beside.fill(1);
        }
        self.syy = s;
        self.next = end;
        if below_one(table) || s < 1.0 {
            core::hint::cold_path();
            (self.next, self.syy) = (start, first);
            self.fill_exact(end);
        }
    }

    /// Fills the next `count` entries of `a` and of `b`, which must both
    /// have that many left, side by side: the processor then works on one
    /// chain while the other waits. Every `B::EVERY` steps also fill one
    /// entry of `beside`.
    fn fill_both<B: Beside>(a: &mut Self, b: &mut Self, count: usize, beside: &mut B) {
        let (ia, ib) = (a.next, b.next);
        let (fa, fb) = (a.syy, b.syy);
        a.increments(ia + count);
        b.increments(ib + count);
        // SAFETY: `a.increments(ia + count)` above wrote these entries of
        // the table of `a`.
        let ta = unsafe { assume_written_mut(&mut a.table[ia..ia + count]) };
        // SAFETY: `b.increments(ib + count)` above wrote these entries of
        // the table of `b`.
        let tb = unsafe { assume_written_mut(&mut b.table[ib..ib + count]) };
        let (mut sa, mut sb) = (fa, fb);
        for (pa, pb) in ta.chunks_mut(B::EVERY).zip(tb.chunks_mut(B::EVERY)) {
            for (ta, tb) in pa.iter_mut().zip(pb) {
                let (da, db) = (*ta, *tb);
                *ta = sa;
                *tb = sb;
                sa += da;
                sb += db;
            }
            beside.fill(1);
        }
        (a.syy, b.syy) = (sa, sb);
        a.next += count;
        b.next += count;
        let (redo_a, redo_b) = (below_one(ta) || sa < 1.0, below_one(tb) || sb < 1.0);
        if redo_a {
            core::hint::cold_path();
            (a.next, a.syy) = (ia, fa);
            a.fill_exact(ia + count);
        }
        if redo_b {
            core::hint::cold_path();
            (b.next, b.syy) = (ib, fb);
            b.fill_exact(ib + count);
        }
    }
}

/// `v` as the values it holds.
///
/// # Safety
///
/// Every element of `v` must have been written.
#[inline(always)]
unsafe fn assume_written(v: &[MaybeUninit<f32>]) -> &[f32] {
    // SAFETY: `MaybeUninit<f32>` has the layout of `f32`, and the caller
    // promises that every element holds a value.
    unsafe { core::slice::from_raw_parts(v.as_ptr().cast(), v.len()) }
}

/// `v` as the values it holds, to change them.
///
/// # Safety
///
/// Every element of `v` must have been written.
#[inline(always)]
unsafe fn assume_written_mut(v: &mut [MaybeUninit<f32>]) -> &mut [f32] {
    // SAFETY: as in `assume_written`. Only values are written through the
    // result, so the elements keep holding values.
    unsafe { core::slice::from_raw_parts_mut(v.as_mut_ptr().cast(), v.len()) }
}

/// Whether any value of `v` is below 1. Not-a-number values are not.
fn below_one(v: &[f32]) -> bool {
    // Only the aarch64 code moves `done` on.
    #[allow(unused_mut)]
    let mut done = 0;
    #[cfg(target_arch = "aarch64")]
    {
        use core::arch::aarch64::*;
        // SAFETY: the loads read v[i..i + 4] with i + 4 <= v.len().
        unsafe {
            let one = vdupq_n_f32(1.0);
            let mut any = vdupq_n_u32(0);
            while done + 4 <= v.len() {
                any = vorrq_u32(any, vcltq_f32(vld1q_f32(v.as_ptr().add(done)), one));
                done += 4;
            }
            if vmaxvq_u32(any) != 0 {
                return true;
            }
        }
    }
    v[done..].iter().any(|&x| x < 1.0)
}

/// Second step of `pitch_search`: the correlation at full resolution, for the
/// lags that are at most two away from twice a coarse candidate. libopus
/// sets every other lag to 0, and no lag below -1.
///
/// There are at most ten such lags. Their inner products are computed
/// together (`inner_prod_many`). On aarch64 the pass fills one entry of
/// `beside` per four samples.
fn fine_xcorr<B: Beside>(
    x_lp: &[f32],
    y: &[f32],
    len: usize,
    best_pitch: &[usize; 2],
    max_pitch: usize,
    beside: &mut B,
) -> FineLags {
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
    if B::ACTIVE {
        inner_prod_many_side(
            x_lp,
            y,
            &lags[..count],
            len,
            &mut sums[..count],
            &mut || beside.fill(1),
        );
    } else {
        inner_prod_many(x_lp, y, &lags[..count], len, &mut sums[..count]);
    }
    let mut fine = FineLags {
        lags,
        xcorr: [0.0; 10],
        count,
        end: 0,
    };
    for (k, (&i, &sum)) in lags[..count].iter().zip(&sums).enumerate() {
        // C `MAX32(-1, sum)`, which keeps a not-a-number sum.
        fine.xcorr[k] = if -1.0 > sum { -1.0 } else { sum };
        if sum > 0.0 {
            fine.end = fine.end.max(i + 1);
        }
    }
    // Each candidate's lags are in rising order; the second candidate can be
    // the lower one. An insertion sort of at most ten values.
    for i in 1..count {
        let mut j = i;
        while j > 0 && fine.lags[j - 1] > fine.lags[j] {
            fine.lags.swap(j - 1, j);
            fine.xcorr.swap(j - 1, j);
            j -= 1;
        }
    }
    fine
}

/// The lags of the fine search of `pitch_search` (`fine_xcorr`) and their
/// correlations, C `xcorr[lags[k]] = xcorr[k]` for `k < count`. libopus
/// sets the correlation of every other lag to 0.
struct FineLags {
    /// In rising order.
    lags: [usize; 10],
    xcorr: [f32; 10],
    count: usize,
    /// One more than the last lag whose correlation is above 0, 0 when there
    /// is none.
    end: usize,
}

impl FineLags {
    /// C `xcorr[i]` after the fine search.
    fn at(&self, i: usize) -> f32 {
        self.lags[..self.count]
            .iter()
            .position(|&lag| lag == i)
            .map_or(0.0, |k| self.xcorr[k])
    }
}

/// The coarse search of `pitch_search`: `pitch_xcorr(x4, y4, xcorr, ..)` for
/// `xcorr.len()` lags, with `x4.len()` samples per lag, and the energies of
/// the first windows of the two calls of `find_best_pitch`
/// (`best_pitch_from`): `1 + sum of y4[j]^2` over `j < x4.len()`, and
/// `1 + sum of y[j]^2` over all of `y`.
///
/// The two energies are chains of additions that wait for each other. On
/// aarch64 they are summed inside the passes of the correlation, which
/// leave time for them, so they take almost no time of their own. The
/// passes also fill entries of `beside`.
fn coarse_xcorr<B: Beside>(
    x4: &[f32],
    y4: &[f32],
    y: &[f32],
    xcorr: &mut [f32],
    beside: &mut B,
) -> (f32, f32) {
    let len4 = x4.len();
    // Only the aarch64 code takes steps; elsewhere `finish` does all of them.
    #[allow(unused_mut)]
    let (mut coarse, mut fine) = (Energy::new(&y4[..len4]), Energy::new(y));
    #[cfg(not(target_arch = "aarch64"))]
    let _ = beside;
    #[cfg(target_arch = "aarch64")]
    if xcorr.len() >= 64 && len4 >= 8 {
        // libopus `celt_pitch_xcorr_float_neon`, as in `pitch_xcorr_neon`.
        let lags = xcorr.len();
        assert!(y4.len() >= len4 + lags - 1);
        let by_sample = lags & !3;
        let (grouped, rest) = xcorr.split_at_mut(by_sample);
        if USE_SKEW && by_sample >= SKEW_PASS && len4 >= 4 * SKEW_MIN_BLOCKS {
            // Enough steps per call of `side` that both energies are done in
            // the kernel.
            let calls = skew_side_calls(len4, by_sample).max(1);
            let per_call = y.len().max(len4).div_ceil(calls);
            let mut side = || {
                Energy::step_both(&mut fine, &mut coarse, per_call);
                beside.fill(BESIDE_PER_SKEW_CALL);
            };
            xcorr_skewed_neon(x4, y4, grouped, &mut side);
        } else {
            // `side` runs once per step of eight samples of every pass of 64
            // lags. Enough steps that both energies are done in the passes
            // of the encoder's 20 ms frames.
            let mut side = || {
                fine.step(4);
                coarse.step(2);
                beside.fill(BESIDE_PER_STEP);
            };
            xcorr_by_sample_passes_side_neon::<16, _>(x4, y4, grouped, &mut side);
        }
        let mut offsets = [0usize; 3];
        for (k, off) in offsets[..rest.len()].iter_mut().enumerate() {
            *off = by_sample + k;
        }
        inner_prod_many(x4, y4, &offsets[..rest.len()], len4, rest);
        return (coarse.finish(), fine.finish());
    }
    pitch_xcorr(x4, y4, xcorr, len4, xcorr.len());
    (coarse.finish(), fine.finish())
}

/// `1 + sum of y[j]^2`, summed one sample after the other with `mac`, in
/// parts: `step` adds the next few samples.
struct Energy<'a> {
    y: &'a [f32],
    pos: usize,
    sum: f32,
}

impl<'a> Energy<'a> {
    fn new(y: &'a [f32]) -> Self {
        Self {
            y,
            pos: 0,
            sum: 1.0,
        }
    }

    /// Adds the squares of the next `count` samples, or of all that are left.
    #[inline(always)]
    fn step(&mut self, count: usize) {
        let end = (self.pos + count).min(self.y.len());
        for &v in &self.y[self.pos..end] {
            self.sum = mac(self.sum, v, v);
        }
        self.pos = end;
    }

    /// Adds the squares of all samples that are left and returns the sum.
    fn finish(mut self) -> f32 {
        self.step(self.y.len());
        self.sum
    }

    /// Adds the squares of the next `count` samples of `a` and of `b`, or of
    /// all that are left, side by side: the processor then works on one
    /// chain while the other waits.
    #[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
    #[inline(always)]
    fn step_both(a: &mut Self, b: &mut Self, count: usize) {
        let na = count.min(a.y.len() - a.pos);
        let nb = count.min(b.y.len() - b.pos);
        let both = na.min(nb);
        let (mut sa, mut sb) = (a.sum, b.sum);
        for (&va, &vb) in a.y[a.pos..a.pos + both]
            .iter()
            .zip(&b.y[b.pos..b.pos + both])
        {
            sa = mac(sa, va, va);
            sb = mac(sb, vb, vb);
        }
        (a.sum, b.sum) = (sa, sb);
        a.pos += both;
        b.pos += both;
        a.step(na - both);
        b.step(nb - both);
    }
}

/// Writes `src[2 * j]` into `dst[j]` for every `j < dst.len()` and returns
/// `dst` with its values.
///
/// Needs `src.len() >= 2 * dst.len() - 1`. Written over pairs of samples, so
/// that the compiler can turn it into vector code.
#[inline(always)]
fn take_even<'b>(src: &[f32], dst: &'b mut [MaybeUninit<f32>]) -> &'b [f32] {
    let n = dst.len();
    if n == 0 {
        return &[];
    }
    let (pairs, last) = src[..2 * n - 1].as_chunks::<2>();
    assert!(pairs.len() == n - 1);
    for (d, pair) in dst.iter_mut().zip(pairs) {
        d.write(pair[0]);
    }
    dst[n - 1].write(last[0]);
    // SAFETY: the loop wrote dst[..n - 1], one element per pair, and the
    // line above dst[n - 1].
    unsafe { assume_written(dst) }
}

/// libopus `pitch_search`: finds the lag at which `y` is most similar to
/// `x_lp`.
///
/// `x_lp` and `y` are signals that were decimated by two. `len` and
/// `max_pitch` count samples before that decimation, and so does the lag
/// that is returned. `x_lp` must hold `len / 2` samples and `y`
/// `(len + max_pitch) / 2`.
///
/// The search first decimates both signals by two again and correlates them
/// at every lag (the coarse search, `pitch_xcorr`). It then correlates
/// `x_lp` and `y` themselves, only around the two best coarse lags. The lag
/// is refined by one sample from the correlation at its two neighbours.
pub fn pitch_search(x_lp: &[f32], y: &[f32], len: usize, max_pitch: usize) -> usize {
    pitch_search_beside(x_lp, y, len, max_pitch, &mut ())
}

/// A chain of additions that `pitch_search_beside` advances in the gaps of
/// its own work: the window energies of `remove_doubling` when the
/// encoder's pre-filter calls the two (`prefilter_pitch`), and nothing
/// (`()`) otherwise.
trait Beside {
    /// Whether there is a chain at all.
    const ACTIVE: bool;

    /// The loops of the window energies of `find_best_pitch` fill one entry
    /// per this many steps of their own.
    const EVERY: usize;

    /// Fills up to `count` more entries.
    fn fill(&mut self, count: usize);

    /// Called when the coarse search has found its two lags (in samples
    /// of `y` decimated by two, `best_pitch_from`).
    fn coarse_found(&mut self, best_pitch: &[usize; 2]);
}

impl Beside for () {
    const ACTIVE: bool = false;
    const EVERY: usize = usize::MAX;

    #[inline(always)]
    fn fill(&mut self, _count: usize) {}

    #[inline(always)]
    fn coarse_found(&mut self, _best_pitch: &[usize; 2]) {}
}

/// Entries of `beside` that the coarse search fills per call of its side
/// work, in the kernel for shared windows (`xcorr_skewed_neon`) and in the
/// other one (`xcorr_by_sample_passes_side_neon`).
#[cfg(target_arch = "aarch64")]
const BESIDE_PER_SKEW_CALL: usize = 4;
#[cfg(target_arch = "aarch64")]
const BESIDE_PER_STEP: usize = 3;

/// `pitch_search` that also advances the chain `beside`.
fn pitch_search_beside<B: Beside>(
    x_lp: &[f32],
    y: &[f32],
    len: usize,
    max_pitch: usize,
    beside: &mut B,
) -> usize {
    let mut best_pitch = [0, 0];

    let max_pitch = max_pitch >> 1;
    let len = len >> 1;
    let lag = len + max_pitch;
    let coarse_lags = max_pitch >> 1;

    // The buffers hold up to 1024 values each. Longer inputs panic. Only the
    // correlations are cleared: every other value is written before it is
    // read.
    let mut x4_buf = [const { MaybeUninit::<f32>::uninit() }; PITCH_BUF_MAX];
    let mut y4_buf = [const { MaybeUninit::<f32>::uninit() }; PITCH_BUF_MAX];
    let x_lp4 = take_even(x_lp, &mut x4_buf[..len >> 1]);
    let y_lp4 = take_even(y, &mut y4_buf[..lag >> 1]);
    let mut xcorr: FixedVec<f32, PITCH_BUF_MAX> = FixedVec::new();
    xcorr.resize(coarse_lags, 0.0);

    let (syy_coarse, syy_fine) = coarse_xcorr(x_lp4, y_lp4, &y[..len], &mut xcorr, beside);

    // The window energies of the two calls of `find_best_pitch`. Both tables
    // are chains of additions that wait for each other, so they are filled
    // side by side, up to the last coarse lag.
    let mut tables = [const { MaybeUninit::<f32>::uninit() }; PITCH_BUF_MAX + PITCH_BUF_MAX / 2];
    let (coarse_table, fine_table) = tables[..coarse_lags + max_pitch].split_at_mut(coarse_lags);
    let mut coarse = WindowEnergies::new(y_lp4, len >> 1, syy_coarse, coarse_table);
    let mut fine = WindowEnergies::new(y, len, syy_fine, fine_table);
    WindowEnergies::fill_both(&mut coarse, &mut fine, coarse_lags.min(max_pitch), beside);
    coarse.fill_to(coarse_lags, beside);
    best_pitch_from(&xcorr, coarse.filled(), &mut best_pitch);
    beside.coarse_found(&best_pitch);

    let found = fine_xcorr(x_lp, y, len, &best_pitch, max_pitch, beside);
    // libopus compares every lag up to `max_pitch`. Only the lags of the
    // fine correlation have a correlation above 0, and only they can win;
    // the window energy at each of them is the same as in libopus's loop.
    fine.fill_to(found.end, beside);
    let syy = fine.filled();
    let mut best = BestPitch::new();
    for (&i, &xc) in found.lags[..found.count].iter().zip(&found.xcorr) {
        if i < found.end {
            best.consider(i, xc, syy[i]);
        }
    }
    best_pitch = best.pitch;

    let mut offset = 0;
    if best_pitch[0] > 0 && best_pitch[0] < max_pitch - 1 {
        let a = found.at(best_pitch[0] - 1);
        let b = found.at(best_pitch[0]);
        let c = found.at(best_pitch[0] + 1);
        if (c - a) > 0.7 * (b - a) {
            offset = 1;
        } else if (a - c) > 0.7 * (b - c) {
            offset = -1;
        }
    }

    ((2 * best_pitch[0]) as isize).wrapping_sub(offset as isize) as usize
}

/// libopus `compute_pitch_gain` of the float build:
/// `xy / sqrt(1 + xx * yy)`. It is negative when `xy` is.
fn compute_pitch_gain(xy: f32, xx: f32, yy: f32) -> f32 {
    xy / mac(1.0, xx, yy).sqrt()
}

static SECOND_CHECK: [usize; 16] = [0, 0, 3, 2, 3, 2, 5, 2, 3, 2, 3, 2, 5, 2, 3, 2];

/// `remove_doubling` tries the periods `t0 / k` for `k` from 2 to 15.
const MAX_DOUBLING_CANDS: usize = 14;

/// Capacity of the table of window energies of `remove_doubling`. It holds
/// `max_period / 2 + 1` entries at most; the encoder's `max_period` is
/// `COMBFILTER_MAXPERIOD = 1024`.
const MAX_YY_SIZE: usize = 1024;

/// The inner products `remove_doubling` needs before it decides anything,
/// and its table of window energies.
///
/// With `target = x[max_period..]` and `n` samples per product, it returns
/// `(xx, xy)`: the energy of `target` and its product with the signal `t0`
/// samples earlier (libopus computes the two with one `dual_inner_prod`).
/// For every candidate `(t1, t1b)` it stores in `pair_xy` the products of
/// `target` with the signal `t1` and `t1b` samples earlier. Each product has
/// the bits of `celt_inner_prod`, as each output of libopus
/// `dual_inner_prod_neon` has.
///
/// The returned table holds `end` entries: entry `i` is the energy of the
/// `n` samples that start `i` samples before the target, held at 0 or more.
/// Each entry follows from the running sum of the one before it (C `yy`,
/// which is not held at 0), starting from `xx`.
///
/// `started` is that chain when the caller has started it already
/// (`prefilter_pitch`, aarch64 only); `xx` is then its entry 0. Otherwise
/// the chain starts here, in `own_table`.
///
/// On aarch64 the products are computed together, up to 16 per pass. The
/// energies are one chain of additions that waits for itself, and they
/// start from `xx`. So `xx` comes first, and the energies are summed during
/// the passes of the other products, which leave time for them.
#[allow(clippy::too_many_arguments)]
fn doubling_products<'a>(
    x: &'a [f32],
    max_period: usize,
    n: usize,
    t0: usize,
    cands: &[(usize, usize)],
    pair_xy: &mut [(f32, f32); MAX_DOUBLING_CANDS],
    end: usize,
    started: Option<Energies<'a>>,
    own_table: &'a mut [MaybeUninit<f32>; MAX_YY_SIZE],
    around_t0: &mut Option<[f32; 3]>,
) -> (f32, f32, Energies<'a>) {
    let x_target = &x[max_period..];
    #[cfg(target_arch = "aarch64")]
    {
        // Only `prefilter_pitch` on Apple's processors starts the chain.
        #[cfg(not(target_vendor = "apple"))]
        let started: Option<Energies> = {
            let _ = started;
            None
        };
        const TOTAL: usize = 4 + 2 * MAX_DOUBLING_CANDS;
        // The energy of the target, then its products at the periods t0,
        // t0 - 1 and t0 + 1 (`remove_doubling` needs the last two when it
        // keeps t0), then those of the candidates.
        let mut offsets = [0usize; TOTAL];
        offsets[..4].copy_from_slice(&[
            max_period,
            max_period - t0,
            max_period - (t0 - 1),
            max_period - (t0 + 1),
        ]);
        for (o, &(t1, t1b)) in offsets[4..].as_chunks_mut::<2>().0.iter_mut().zip(cands) {
            *o = [max_period - t1, max_period - t1b];
        }
        let count = 4 + 2 * cands.len();
        let mut sums = [0.0f32; TOTAL];
        // The products that the passes with the chain compute: all but the
        // energy when the chain has started, else all after the short first
        // pass, which gives the energy that starts the chain.
        let (mut energies, first) = match started {
            Some(energies) => {
                sums[0] = energies.filled()[0];
                (energies, 1)
            }
            None => {
                inner_prod_many(x_target, x, &offsets[..4], n, &mut sums[..4]);
                let table = &mut own_table[..end];
                (Energies::new(x, max_period, n, sums[0], table), 4)
            }
        };
        energies.set_end(end);
        let steps = n / 4;
        for (offs, out) in offsets[first..count]
            .chunks(MANY_PER_PASS)
            .zip(sums[first..count].chunks_mut(MANY_PER_PASS))
        {
            // A pass calls `side` once per four samples (`steps` times).
            if energies.left() >= 2 * steps {
                // SAFETY: the table has room for two entries per call
                // (checked above).
                inner_prod_many_side(x_target, x, offs, n, out, &mut || unsafe {
                    energies.step2_unchecked()
                });
            } else {
                // The last entries, spread over the pass.
                let per_call = energies.left().div_ceil(steps.max(1));
                inner_prod_many_side(x_target, x, offs, n, out, &mut || {
                    // On the Apple M4 Pro the runs are 1-3 % faster for
                    // 10 and 20 ms frames, on the Cortex-A76 the steps
                    // one by one.
                    #[cfg(target_vendor = "apple")]
                    energies.step_runs(per_call);
                    #[cfg(not(target_vendor = "apple"))]
                    energies.step(per_call);
                });
            }
        }
        energies.finish();
        for (p, s) in pair_xy.iter_mut().zip(sums[4..count].as_chunks::<2>().0) {
            *p = (s[0], s[1]);
        }
        *around_t0 = Some([sums[2], sums[1], sums[3]]);
        (sums[0], sums[1], energies)
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let _ = (around_t0, started);
        let (xx, xy) = dual_inner_prod(x_target, x_target, &x[max_period - t0..], n);
        for (p, &(t1, t1b)) in pair_xy.iter_mut().zip(cands) {
            *p = dual_inner_prod(x_target, &x[max_period - t1..], &x[max_period - t1b..], n);
        }
        let mut energies = Energies::new(x, max_period, n, xx, &mut own_table[..end]);
        energies.finish();
        (xx, xy, energies)
    }
}

/// The table of window energies of `remove_doubling`, filled in parts:
/// C `yy = yy+MULT16_16(x[-i],x[-i])-MULT16_16(x[N-i],x[N-i])` and
/// `yy_lookup[i] = MAX32(0, yy)` for `i` from 1 on. clang turns the update
/// into two fused multiply-adds (`mac`).
///
/// The table is not cleared before use: the entries below `next` hold
/// values, and no other entry is read.
struct Energies<'a> {
    /// The decimated signal; the target starts at `x[max_period]`.
    x: &'a [f32],
    max_period: usize,
    n: usize,
    /// The running sum C `yy`.
    yy: f32,
    /// The next entry to fill. The entries below it are filled.
    next: usize,
    /// The entries are filled up to here, at most to the end of `table`.
    end: usize,
    table: &'a mut [MaybeUninit<f32>],
}

impl<'a> Energies<'a> {
    /// The chain that starts at `table[0] = xx`, to be filled up to the end
    /// of `table`.
    fn new(
        x: &'a [f32],
        max_period: usize,
        n: usize,
        xx: f32,
        table: &'a mut [MaybeUninit<f32>],
    ) -> Self {
        // Entry `i` reads x[max_period - i] and x[max_period + n - i].
        assert!(table.len() <= max_period + 1 && max_period + n <= x.len());
        table[0].write(xx);
        Self {
            x,
            max_period,
            n,
            yy: xx,
            next: 1,
            end: table.len(),
            table,
        }
    }

    /// Fills the entries up to `end` (at most the length of the table) and
    /// no further. Entries that are filled already stay filled.
    #[cfg(target_arch = "aarch64")]
    fn set_end(&mut self, end: usize) {
        self.end = end.min(self.table.len());
    }

    /// Fills the next `count` entries, or all that are left.
    #[inline(always)]
    fn step(&mut self, count: usize) {
        let end = (self.next + count).min(self.end);
        for i in self.next..end {
            let new = self.x[self.max_period - i];
            let old = self.x[self.max_period + self.n - i];
            self.yy = mac(mac(self.yy, new, new), -old, old);
            // C `MAX32(0, yy)`.
            self.table[i].write(if 0.0 > self.yy { 0.0 } else { self.yy });
        }
        self.next = self.next.max(end);
    }

    /// `step`, with the samples of all its entries taken as two runs first.
    /// It has fewer instructions per entry when `count` is large, and more
    /// per call.
    #[cfg(all(target_arch = "aarch64", target_vendor = "apple"))]
    #[inline(always)]
    fn step_runs(&mut self, count: usize) {
        let (start, end) = (self.next, (self.next + count).min(self.end));
        if end <= start {
            return;
        }
        // Entry i reads x[max_period - i] and x[max_period + n - i]: from
        // `start` to `end`, the two runs below from their last sample down.
        let (at, n) = (self.max_period, self.n);
        let entering = &self.x[at + 1 - end..=at - start];
        let leaving = &self.x[at + n + 1 - end..=at + n - start];
        let mut yy = self.yy;
        for ((t, &new), &old) in self.table[start..end]
            .iter_mut()
            .zip(entering.iter().rev())
            .zip(leaving.iter().rev())
        {
            yy = mac(mac(yy, new, new), -old, old);
            // C `MAX32(0, yy)`.
            t.write(if 0.0 > yy { 0.0 } else { yy });
        }
        self.yy = yy;
        self.next = end;
    }

    /// The filled entries, `table[..next]`.
    fn filled(&self) -> &[f32] {
        // SAFETY: `new` writes entry 0, and `step` and `step2_unchecked`
        // write each entry before they move `next` past it.
        unsafe { assume_written(&self.table[..self.next]) }
    }

    /// Fills all entries that are left.
    fn finish(&mut self) {
        self.step(self.left());
    }

    /// How many entries are left to fill.
    fn left(&self) -> usize {
        self.end.saturating_sub(self.next)
    }

    /// Fills the next two entries, as `step(2)`, without checks and without
    /// a branch. The code then stays in the block of the caller, where the
    /// compiler can place its multiply-adds between other work.
    ///
    /// # Safety
    ///
    /// At least two entries must be left (`left() >= 2`).
    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    unsafe fn step2_unchecked(&mut self) {
        let i = self.next;
        debug_assert!(i + 2 <= self.end && self.end <= self.table.len());
        // SAFETY: entries i and i + 1 exist (the caller's promise, and
        // `set_end` keeps `end` within the table), and `new` asserted
        // table.len() <= max_period + 1 and max_period + n <= x.len(). So
        // max_period - i - 1 >= 0 and max_period + n - i < x.len() for
        // i >= 1.
        unsafe {
            let at = self.x.as_ptr().add(self.max_period - i);
            let (new0, new1) = (*at, *at.sub(1));
            let (old0, old1) = (*at.add(self.n), *at.add(self.n - 1));
            let yy0 = mac(mac(self.yy, new0, new0), -old0, old0);
            let yy1 = mac(mac(yy0, new1, new1), -old1, old1);
            let table = self.table.as_mut_ptr().add(i).cast::<f32>();
            // C `MAX32(0, yy)`.
            table.write(if 0.0 > yy0 { 0.0 } else { yy0 });
            table.add(1).write(if 0.0 > yy1 { 0.0 } else { yy1 });
            self.yy = yy1;
        }
        self.next = i + 2;
    }
}

/// The candidate periods of `remove_doubling` for the period `t0`: `t1` is
/// about `t0 / k`, and `t1b` is a second period that is checked together
/// with it. Both depend on `t0` and `k` only. Returns the candidates and
/// how many there are. All periods are in samples of the decimated signal.
fn doubling_candidates(
    t0: usize,
    min_period: usize,
    max_period: usize,
) -> ([(usize, usize); MAX_DOUBLING_CANDS], usize) {
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
    (cands, n_cands)
}

/// libopus `remove_doubling` (celt/pitch.c) of the float build: checks
/// whether the period `*t0_ptr` that the pitch search found is a multiple
/// of the true period, and returns the pitch gain.
///
/// `x` is the decimated signal of `pitch_downsample`, with `max_period / 2`
/// samples of history before the frame. All periods are in samples before
/// the decimation. On return `*t0_ptr` holds the chosen period, at least
/// `min_period`.
pub fn remove_doubling(
    x: &[f32],
    max_period: usize,
    min_period: usize,
    n: usize,
    t0_ptr: &mut usize,
    prev_period: usize,
    prev_gain: f32,
) -> f32 {
    remove_doubling_from(
        x,
        max_period,
        min_period,
        n,
        t0_ptr,
        prev_period,
        prev_gain,
        None,
    )
}

/// `remove_doubling`, with the chain of window energies that the caller
/// may have started (`doubling_products`).
#[allow(clippy::too_many_arguments)]
fn remove_doubling_from<'a>(
    x: &'a [f32],
    mut max_period: usize,
    mut min_period: usize,
    mut n: usize,
    t0_ptr: &mut usize,
    mut prev_period: usize,
    prev_gain: f32,
    started: Option<Energies<'a>>,
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

    // All inner products of the loop below can be computed before its
    // decisions, several at a time.
    let (cands, n_cands) = doubling_candidates(t0, min_period, max_period);
    let cands = &cands[..n_cands];

    // libopus fills the table of window energies up to `max_period`. It is
    // only read at `t0` and at the periods of the candidates, so here it ends
    // at the largest of them.
    let last = cands.iter().fold(t0, |m, &(t1, t1b)| m.max(t1).max(t1b));
    // Not cleared: `Energies` writes each entry before it is read.
    let mut own_table = [const { MaybeUninit::<f32>::uninit() }; MAX_YY_SIZE];

    let mut pair_xy = [(0.0f32, 0.0f32); MAX_DOUBLING_CANDS];
    let mut around_t0 = None;
    let (xx, xy, energies) = doubling_products(
        x,
        max_period,
        n,
        t0,
        cands,
        &mut pair_xy,
        last + 1,
        started,
        &mut own_table,
        &mut around_t0,
    );
    let yy_lookup = &energies.filled()[..last + 1];

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

        // C `MAX16(.3, .7*g0 - cont)`, and its bias against very short
        // periods. (libopus's third case, `T1<2*minperiod`, is never
        // reached, because it comes after `T1<3*minperiod`.)
        let thresh = if t1 < 3 * min_period {
            max_first(0.4, mac(-cont, 0.85, g0))
        } else {
            max_first(0.3, mac(-cont, 0.7, g0))
        };

        if g1 > thresh {
            best_xy = xy_new;
            best_yy = yy_new;
            t = t1;
            g = g1;
        }
    }

    best_xy = max_first(0.0, best_xy);
    let pg = if best_yy <= best_xy {
        1.0f32
    } else {
        best_xy / (best_yy + 1.0)
    };

    // The correlation at the periods t - 1, t and t + 1.
    let xcorr_res = match around_t0 {
        Some(products) if t == t0 => products,
        _ => {
            let offsets: [usize; 3] = core::array::from_fn(|k_idx| {
                let lag = (t as i32 + k_idx as i32 - 1) as usize;
                max_period - lag
            });
            let mut products = [0.0f32; 3];
            inner_prod_many(x_target, x, &offsets, n, &mut products);
            products
        }
    };

    let mut offset = 0;
    if (xcorr_res[2] - xcorr_res[0]) > 0.7 * (xcorr_res[1] - xcorr_res[0]) {
        offset = 1;
    } else if (xcorr_res[0] - xcorr_res[2]) > 0.7 * (xcorr_res[1] - xcorr_res[2]) {
        offset = -1;
    }

    let pg = if pg > g { g } else { pg };
    *t0_ptr = (2 * t as i32 + offset) as usize;
    if *t0_ptr < min_period0 {
        *t0_ptr = min_period0;
    }

    pg
}

/// The pitch analysis of the encoder's pre-filter after `pitch_downsample`
/// (libopus `run_prefilter`, celt/celt_encoder.c): `pitch_search` over
/// `max_period - 3 * min_period` lags, then `remove_doubling` on the period
/// it finds. `x` is the decimated signal (C `pitch_buf`), with
/// `max_period / 2` samples of history before the frame of `n / 2`
/// samples. Returns the period C `pitch_index` and the gain of
/// `remove_doubling`, as the two calls give them.
///
/// The chain of window energies of `remove_doubling` (`Energies`) waits for
/// itself at every step, and it does not depend on the period. On Apple's
/// 64-bit ARM processors it therefore starts before the search and runs in
/// the gaps of the search's own work (`DoublingChain`). For a 2.5 ms frame
/// the chain takes longer than the products of `remove_doubling`. These
/// cores run many instructions at once, out of program order, and the
/// search plus the chain takes 12 % less time than the two calls on an
/// Apple M4 Pro at 2.5 ms. The Cortex-A55 and Cortex-A76 of an Android
/// phone take 8 to 20 % more time that way, so elsewhere this is the two
/// calls.
pub(crate) fn prefilter_pitch(
    x: &[f32],
    max_period: usize,
    min_period: usize,
    n: usize,
    prev_period: usize,
    prev_gain: f32,
) -> (usize, f32) {
    let search_max = max_period - 3 * min_period;
    // C `pitch_index` from the lag of `pitch_search`.
    let period = |lag: usize| (max_period - lag).min(max_period - 2);
    let half = max_period >> 1;
    #[cfg(all(target_arch = "aarch64", target_vendor = "apple"))]
    {
        let target = &x[half..half + n / 2];
        let xx = celt_inner_prod(target, target);
        let mut table = [const { MaybeUninit::<f32>::uninit() }; MAX_YY_SIZE];
        let mut energies = Energies::new(x, half, n / 2, xx, &mut table[..half + 1]);
        energies.set_end(DOUBLING_EARLY);
        let mut chain = DoublingChain {
            energies,
            max_period,
            min_period,
        };
        let mut t0 = period(pitch_search_beside(
            &x[half..],
            x,
            n,
            search_max,
            &mut chain,
        ));
        let gain = remove_doubling_from(
            x,
            max_period,
            min_period,
            n,
            &mut t0,
            prev_period,
            prev_gain,
            Some(chain.energies),
        );
        (t0, gain)
    }
    #[cfg(not(all(target_arch = "aarch64", target_vendor = "apple")))]
    {
        let mut t0 = period(pitch_search(&x[half..], x, n, search_max));
        let gain = remove_doubling(
            x,
            max_period,
            min_period,
            n,
            &mut t0,
            prev_period,
            prev_gain,
        );
        (t0, gain)
    }
}

/// Entries of the chain of `remove_doubling` that `prefilter_pitch` fills
/// before the coarse search has found its lags. Until then the period is
/// not known, and neither is how many entries `remove_doubling` reads.
#[cfg(all(target_arch = "aarch64", target_vendor = "apple"))]
const DOUBLING_EARLY: usize = 512;

/// The chain of window energies of `remove_doubling` while `pitch_search`
/// runs (`prefilter_pitch`). `max_period` and `min_period` are the
/// arguments of `remove_doubling`.
#[cfg(all(target_arch = "aarch64", target_vendor = "apple"))]
struct DoublingChain<'a> {
    energies: Energies<'a>,
    max_period: usize,
    min_period: usize,
}

#[cfg(all(target_arch = "aarch64", target_vendor = "apple"))]
impl Beside for DoublingChain<'_> {
    const ACTIVE: bool = true;
    const EVERY: usize = 3;

    #[inline(always)]
    fn fill(&mut self, count: usize) {
        self.energies.step_runs(count);
    }

    /// Limits the chain to the entries `remove_doubling` can read. The fine
    /// search tests the lags `i` within 2 of `2 * c` for each coarse lag
    /// `c`, and returns `2 * i` moved by at most one. The table ends after
    /// the longest candidate period of the periods that follow from that.
    /// (When no lag correlates above 0, the search returns 0 and the table
    /// needs more entries; `remove_doubling` then fills them.)
    fn coarse_found(&mut self, best_pitch: &[usize; 2]) {
        let half = self.max_period / 2;
        let min_half = self.min_period / 2;
        let mut end = 0;
        for &c in best_pitch {
            for lag in (4 * c).saturating_sub(5)..=4 * c + 5 {
                let index = self.max_period.saturating_sub(lag).min(self.max_period - 2);
                let t0 = (index / 2).min(half - 1);
                // The longest candidate period: the second one of k = 2 (the
                // others are at most t0).
                let t1 = (2 * t0 + 2) / 4;
                let last = if t1 >= min_half && t1 + t0 <= half {
                    t0 + t1
                } else {
                    t0
                };
                end = end.max(last + 1);
            }
        }
        self.energies.set_end(end);
    }
}

/// libopus `MAX16(a, b)` of the float build, `a > b ? a : b`: `b` when the
/// two compare equal or `b` is not a number.
#[inline(always)]
fn max_first(a: f32, b: f32) -> f32 {
    if a > b { a } else { b }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    #[cfg(not(feature = "std"))]
    use std::vec::Vec;

    /// `v` as a table for `WindowEnergies`, which writes only values into it,
    /// so `v` keeps holding values.
    fn as_table(v: &mut [f32]) -> &mut [MaybeUninit<f32>] {
        // SAFETY: `MaybeUninit<f32>` has the layout of `f32`. The code under
        // test writes only values through the result.
        unsafe { core::slice::from_raw_parts_mut(v.as_mut_ptr().cast(), v.len()) }
    }

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
    fn lag_by_sample(x: &[f32], y: &[f32], len: usize) -> f32 {
        let mut acc = 0.0f32;
        for j in 0..len {
            acc = mac(acc, x[j], y[j]);
        }
        acc
    }

    /// `celt_inner_prod` written as plainly as possible: on aarch64 and
    /// x86-64 four lanes, then (s0 + s2) + (s1 + s3), then the last `n % 4`
    /// products; elsewhere one running sum.
    fn inner_prod_model(x: &[f32], y: &[f32]) -> f32 {
        let n = x.len();
        if cfg!(any(target_arch = "aarch64", all(target_arch = "x86_64", target_feature = "sse2"))) {
            let whole = n / 4 * 4;
            let mut s = [0.0f32; 4];
            for j in 0..whole {
                s[j % 4] = mac(s[j % 4], x[j], y[j]);
            }
            let mut sum = (s[0] + s[2]) + (s[1] + s[3]);
            for j in whole..n {
                sum = mac(sum, x[j], y[j]);
            }
            sum
        } else {
            lag_by_sample(x, y, n)
        }
    }

    #[test]
    fn celt_inner_prod_matches_its_model() {
        let mut rng = Rng::new(3);
        for kind in KINDS {
            for n in (0..=40).chain([63, 64, 65, 240, 481]) {
                let x = samples(&mut rng, n, kind);
                let y = samples(&mut rng, n + 3, kind);
                let got = celt_inner_prod(&x, &y);
                let want = inner_prod_model(&x, &y[..n]);
                assert_same(&[got], &[want], &format!("{kind:?} n {n}"));
            }
        }
    }

    /// On x86-64, `celt_inner_prod` against `celt_inner_prod_sse` of the
    /// libopus these tests link (celt/x86/pitch_sse.c, exported by the
    /// static library of an x86-64 build), on the same values.
    #[cfg(all(feature = "std", target_arch = "x86_64"))]
    #[test]
    fn celt_inner_prod_matches_libopus_sse() {
        use audiopus_sys as _;
        unsafe extern "C" {
            /// `opus_val32 celt_inner_prod_sse(const opus_val16 *x,
            /// const opus_val16 *y, int N)`, float build.
            #[link_name = "celt_inner_prod_sse"]
            fn c_celt_inner_prod_sse(x: *const f32, y: *const f32, n: i32) -> f32;
        }
        let mut rng = Rng::new(31);
        for kind in KINDS {
            for n in (0..=40).chain([63, 64, 65, 240, 481]) {
                let x = samples(&mut rng, n, kind);
                let y = samples(&mut rng, n, kind);
                // SAFETY: `x` and `y` hold `n` values each.
                let want = unsafe { c_celt_inner_prod_sse(x.as_ptr(), y.as_ptr(), n as i32) };
                let got = celt_inner_prod(&x, &y);
                assert_same(&[got], &[want], &format!("{kind:?} n {n}"));
            }
        }
    }

    #[test]
    fn xcorr_by_sample_scalar_matches_one_lag_at_a_time() {
        let mut rng = Rng::new(1);
        for kind in KINDS {
            for len in (0..=20).chain([31, 32, 33, 61, 240]) {
                for lags in (0..=20).chain([31, 32, 33, 40, 61]) {
                    let x = samples(&mut rng, len, kind);
                    let y = samples(&mut rng, (len + lags).saturating_sub(1), kind);
                    let mut got = vec![f32::NAN; lags];
                    xcorr_by_sample_scalar(&x, &y, &mut got, len);
                    let want: Vec<f32> =
                        (0..lags).map(|k| lag_by_sample(&x, &y[k..], len)).collect();
                    assert_same(&got, &want, &format!("{kind:?} len {len} lags {lags}"));
                }
            }
        }
    }

    /// The window energy of `find_best_pitch` summed in parts of any size
    /// is the energy summed in one loop.
    #[test]
    fn energy_in_parts_is_the_energy_in_one_loop() {
        let mut rng = Rng::new(4);
        for kind in KINDS {
            for n in [0usize, 1, 7, 60, 480] {
                let y = samples(&mut rng, n, kind);
                let mut want = 1.0f32;
                for &v in &y {
                    want = mac(want, v, v);
                }
                for part in [1usize, 2, 3, 4, 8, 1000] {
                    let mut e = Energy::new(&y);
                    let mut calls = 0;
                    while e.pos < n && calls < 3 {
                        e.step(part);
                        calls += 1;
                    }
                    assert_same(
                        &[e.finish()],
                        &[want],
                        &format!("{kind:?} n {n} part {part}"),
                    );

                    // Side by side with a shorter second energy.
                    let half = &y[..n / 2];
                    let mut want_half = 1.0f32;
                    for &v in half {
                        want_half = mac(want_half, v, v);
                    }
                    let (mut a, mut b) = (Energy::new(&y), Energy::new(half));
                    for _ in 0..2 {
                        Energy::step_both(&mut a, &mut b, part);
                    }
                    assert_same(
                        &[a.finish(), b.finish()],
                        &[want, want_half],
                        &format!("{kind:?} n {n} part {part} side by side"),
                    );
                }
            }
        }
    }

    #[test]
    fn celt_fir5_matches_the_filter_written_out() {
        let mut rng = Rng::new(5);
        let num = [0.5f32, -0.25, 0.125, 0.75, -0.0625];
        for kind in KINDS {
            for n in (0..=20).chain([572, 991, 992]) {
                let x = samples(&mut rng, n, kind);
                let mut want = x.clone();
                let mut past = [0.0f32; 5];
                for v in want.iter_mut() {
                    let mut sum = *v;
                    for j in 0..5 {
                        sum = mac(sum, num[j], past[j]);
                    }
                    past.copy_within(0..4, 1);
                    past[0] = *v;
                    *v = sum;
                }
                let mut got = x.clone();
                celt_fir5(&mut got, &num);
                assert_same(&got, &want, &format!("{kind:?} n {n}"));
            }
        }
    }

    /// libopus `find_best_pitch` written as plainly as possible: the energy
    /// of the first window, then one loop that tests each lag and moves the
    /// window on.
    fn find_best_pitch_model(xcorr: &[f32], y: &[f32], len: usize) -> [usize; 2] {
        let mut syy = 1.0f32;
        for &v in &y[..len] {
            syy = mac(syy, v, v);
        }
        let mut best_num = [-1.0f32, -1.0];
        let mut best_den = [0.0f32, 0.0];
        let mut best_pitch = [0, 1];
        for (i, &xc) in xcorr.iter().enumerate() {
            if xc > 0.0 {
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
            syy += mac(-(y[i] * y[i]), y[i + len], y[i + len]);
            if syy < 1.0 {
                syy = 1.0;
            }
        }
        best_pitch
    }

    /// The window energies filled in parts, alone or two side by side, and
    /// the search that tests eight lags at a time, give the lags of the
    /// plain loop.
    #[test]
    fn best_pitch_from_matches_the_plain_loop() {
        let mut rng = Rng::new(13);
        for kind in KINDS {
            for (len, lags) in [
                (1usize, 1usize),
                (5, 7),
                (30, 8),
                (30, 244),
                (60, 489),
                (240, 244),
            ] {
                let y = samples(&mut rng, len + lags, kind);
                // Mostly negative correlations, so that most lags change
                // nothing, and some that grow, so that some do.
                let xcorr: Vec<f32> = (0..lags)
                    .map(|i| {
                        let r = rng.unit();
                        if r > 0.6 { r * (1 + i) as f32 * 1e9 } else { r }
                    })
                    .collect();
                let what = format!("{kind:?} len {len} lags {lags}");
                let want = find_best_pitch_model(&xcorr, &y, len);
                let first = Energy::new(&y[..len]).finish();

                let mut exact = vec![f32::NAN; lags];
                WindowEnergies::new(&y, len, first, as_table(&mut exact)).fill_exact(lags);
                for part in [1usize, 3, 8, 1000] {
                    let mut table = vec![f32::NAN; lags];
                    let mut e = WindowEnergies::new(&y, len, first, as_table(&mut table));
                    while e.next < lags {
                        let end = (e.next + part).min(lags);
                        e.fill_to(end, &mut ());
                    }
                    assert_same(&table, &exact, &format!("{what} part {part}"));
                    let mut got = [7, 7];
                    best_pitch_from(&xcorr, &table, &mut got);
                    assert_eq!(got, want, "{what} part {part}");
                }

                // Two chains side by side: this one and a second one that
                // starts from another energy and one entry later.
                let mut table = vec![f32::NAN; lags];
                let mut other = vec![f32::NAN; lags];
                let mut a = WindowEnergies::new(&y, len, first, as_table(&mut table));
                let mut b = WindowEnergies::new(&y, len, 2.5, as_table(&mut other));
                b.fill_to(1, &mut ());
                WindowEnergies::fill_both(&mut a, &mut b, lags / 2, &mut ());
                a.fill_to(lags, &mut ());
                b.fill_to(lags, &mut ());
                assert_same(&table, &exact, &format!("{what} side by side"));
                let mut alone = vec![f32::NAN; lags];
                WindowEnergies::new(&y, len, 2.5, as_table(&mut alone)).fill_exact(lags);
                assert_same(&other, &alone, &format!("{what} second chain"));
            }
        }
    }

    /// Windows whose energy drops to about 0 after a loud start: the running
    /// sum goes below 1 through rounding, and the lower limit applies. The
    /// fast chain must then give the entries of libopus's loop.
    #[test]
    fn window_energies_reach_the_lower_limit_as_libopus() {
        let mut rng = Rng::new(15);
        let mut limited = 0;
        for trial in 0..60 {
            let (len, lags) = ([4usize, 7, 60][trial % 3], [1usize, 5, 30, 244][trial % 4]);
            {
                let mut y = vec![0.0f32; len + lags];
                for (i, v) in y.iter_mut().enumerate() {
                    *v = if i < len {
                        1e4 * rng.unit()
                    } else {
                        1e-3 * rng.unit()
                    };
                }
                let first = Energy::new(&y[..len]).finish();
                let mut want = vec![f32::NAN; lags];
                WindowEnergies::new(&y, len, first, as_table(&mut want)).fill_exact(lags);
                let mut got = vec![f32::NAN; lags];
                WindowEnergies::new(&y, len, first, as_table(&mut got)).fill_to(lags, &mut ());
                assert_same(&got, &want, &format!("len {len} lags {lags}"));
                let mut other = vec![f32::NAN; lags];
                let mut a = WindowEnergies::new(&y, len, first, as_table(&mut got));
                let mut b = WindowEnergies::new(&y, len, first, as_table(&mut other));
                WindowEnergies::fill_both(&mut a, &mut b, lags, &mut ());
                assert_same(&got, &want, &format!("len {len} lags {lags} first"));
                assert_same(&other, &want, &format!("len {len} lags {lags} second"));
                if want.contains(&1.0) {
                    limited += 1;
                }
            }
        }
        // The test is only worth something when the limit was reached.
        assert!(limited > 5, "{limited}");
    }

    /// `prefilter_pitch` gives the period and the gain of `pitch_search`
    /// followed by `remove_doubling`, for every frame size of the encoder:
    /// on periodic signals of many periods, where the table of
    /// `remove_doubling` has from a few to all of its entries, and on the
    /// kinds of `KINDS`. Silence and signals without a positive correlation
    /// make the search return 0 and need the whole table.
    #[test]
    fn prefilter_pitch_is_pitch_search_then_remove_doubling() {
        let mut rng = Rng::new(21);
        for frame in [120usize, 240, 480, 960] {
            let buf_len = (1024 + frame) >> 1;
            for trial in 0..60 {
                let kind = KINDS[trial % KINDS.len()];
                let buf: Vec<f32> = if trial % 3 == 0 {
                    samples(&mut rng, buf_len, kind)
                } else {
                    // A period from 8 to 520 samples of the decimated signal.
                    let period = 8.0 + 512.0 * (0.5 + 0.5 * rng.unit());
                    let noise = samples(&mut rng, buf_len, kind);
                    (0..buf_len)
                        .map(|i| {
                            let phase = core::f32::consts::TAU * i as f32 / period;
                            1e4 * (phase.sin() + 0.3 * (2.0 * phase).sin()) + 0.1 * noise[i]
                        })
                        .collect()
                };
                let prev_period = 15 + (rng.next_u32() % 1000) as usize;
                let prev_gain = 0.5 + 0.5 * rng.unit();
                let found = pitch_search(&buf[512..], &buf, frame, 979);
                let mut want = (1024 - found).min(1022);
                let want_gain =
                    remove_doubling(&buf, 1024, 15, frame, &mut want, prev_period, prev_gain);
                let (got, gain) = prefilter_pitch(&buf, 1024, 15, frame, prev_period, prev_gain);
                let what = format!("frame {frame} trial {trial} {kind:?}");
                assert_eq!(got, want, "{what}");
                assert_same(&[gain], &[want_gain], &what);
            }
            // Silence: no lag correlates above 0.
            let silence = vec![0.0f32; buf_len];
            let found = pitch_search(&silence[512..], &silence, frame, 979);
            let mut want = (1024 - found).min(1022);
            let want_gain = remove_doubling(&silence, 1024, 15, frame, &mut want, 100, 0.5);
            let (got, gain) = prefilter_pitch(&silence, 1024, 15, frame, 100, 0.5);
            assert_eq!(got, want, "silence, frame {frame}");
            assert_same(&[gain], &[want_gain], &format!("silence, frame {frame}"));
        }
    }

    /// The pitch analysis of the encoder's pre-filter, as `run_prefilter`
    /// in celt.rs calls it: `pitch_search` and `remove_doubling` on the
    /// output of `pitch_downsample`. Returns C `pitch_index` and the gain.
    fn prefilter_search(
        pitch_buf: &[f32],
        frame: usize,
        prev_period: usize,
        prev_gain: f32,
    ) -> (usize, f32) {
        prefilter_pitch(pitch_buf, 1024, 15, frame, prev_period, prev_gain)
    }

    /// The time the pitch analysis of the encoder's pre-filter takes per
    /// frame, for each frame size, mono and stereo. Not a check: it prints
    /// the times and a hash of the results, which must not change when the
    /// code gets faster.
    ///
    /// The input is the WAV file (48 kHz, 16-bit, one or two channels) named
    /// by `PITCH_TIMING_WAV`, or a synthetic signal without it. It runs
    /// through the encoder's pre-emphasis. `PITCH_TIMING_REPS` sets how often
    /// each loop runs (default 15); the times are the fastest run.
    ///
    /// `PITCH_TIMING_WAV=music.wav cargo test --release --lib
    /// prefilter_timing -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn prefilter_timing() {
        use std::time::Instant;
        const MAX_PERIOD: usize = 1024;

        // On macOS, ask for a performance core (as examples/perf_harness.rs).
        #[cfg(target_os = "macos")]
        {
            unsafe extern "C" {
                fn pthread_set_qos_class_self_np(qos_class: u32, relative_priority: i32) -> i32;
            }
            // SAFETY: changes the scheduling class of the calling thread only.
            unsafe { pthread_set_qos_class_self_np(0x21, 0) };
        }

        let wav = std::env::var("PITCH_TIMING_WAV").ok();
        let mut input: Vec<Vec<f32>> = match &wav {
            Some(path) => {
                let bytes = std::fs::read(path).expect("read PITCH_TIMING_WAV");
                // The chunks after the 12-byte RIFF header.
                let (mut at, mut channels, mut data) = (12, 0usize, &bytes[..0]);
                while at + 8 <= bytes.len() {
                    let size = u32::from_le_bytes(bytes[at + 4..at + 8].try_into().unwrap());
                    let body = &bytes[at + 8..(at + 8 + size as usize).min(bytes.len())];
                    match &bytes[at..at + 4] {
                        b"fmt " => channels = u16::from_le_bytes([body[2], body[3]]) as usize,
                        b"data" => data = body,
                        _ => {}
                    }
                    at += 8 + size as usize + (size as usize & 1);
                }
                assert!(channels == 1 || channels == 2, "{channels} channels");
                let samples: Vec<f32> = data
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|b| i16::from_le_bytes(*b) as f32)
                    .collect();
                (0..channels)
                    .map(|c| samples.iter().skip(c).step_by(channels).copied().collect())
                    .collect()
            }
            None => {
                // Two voices of harmonics whose pitch changes every 0.25 s.
                let mut rng = Rng::new(5);
                (0..2)
                    .map(|c| {
                        (0..5 * 48000)
                            .map(|i| {
                                let note = [110.0f32, 147.0, 196.0, 131.0][(i / 12000) % 4];
                                let f0 = note * (1.0 + 0.5 * c as f32);
                                let t = i as f32 / 48000.0;
                                let mut s = 0.0;
                                for h in 1..8 {
                                    s += (core::f32::consts::TAU * f0 * h as f32 * t).sin()
                                        / h as f32;
                                }
                                6000.0 * s + 200.0 * rng.unit()
                            })
                            .collect()
                    })
                    .collect()
            }
        };
        // The encoder's pre-emphasis at 48 kHz (`celt_preemphasis`).
        for ch in &mut input {
            let mut mem = 0.0f32;
            for v in ch.iter_mut() {
                let x = *v;
                *v = x - mem;
                mem = 0.850_006_1 * x;
            }
        }
        let reps: usize = std::env::var("PITCH_TIMING_REPS")
            .ok()
            .and_then(|r| r.parse().ok())
            .unwrap_or(15);
        let len = input[0].len();
        println!(
            "input {} ({} ch, {:.1} s), {reps} reps",
            wav.as_deref().unwrap_or("synthetic"),
            input.len(),
            len as f64 / 48000.0
        );
        println!(
            "frame ch frames | downsample  search  doubling  search+doubling (us/frame) | hash"
        );
        let fastest = |f: &mut dyn FnMut()| {
            (0..reps)
                .map(|_| {
                    let t = Instant::now();
                    f();
                    t.elapsed().as_secs_f64()
                })
                .fold(f64::INFINITY, f64::min)
        };
        for frame in [120usize, 240, 480, 960] {
            for channels in 1..=input.len().min(2) {
                let frames = ((len - MAX_PERIOD) / frame).min(400);
                let buf_len = (MAX_PERIOD + frame) >> 1;
                let mut bufs = vec![0.0f32; frames * buf_len];
                let t_down = fastest(&mut || {
                    for (f, buf) in bufs.chunks_exact_mut(buf_len).enumerate() {
                        let at = f * frame;
                        let pre: Vec<&[f32]> = input[..channels]
                            .iter()
                            .map(|ch| &ch[at..at + MAX_PERIOD + frame])
                            .collect();
                        pitch_downsample(&pre, buf, buf_len, channels, 2);
                    }
                });
                // C `pitch_index` before `remove_doubling`.
                let mut found = vec![0usize; frames];
                let t_search = fastest(&mut || {
                    for (buf, out) in bufs.chunks_exact(buf_len).zip(found.iter_mut()) {
                        let lag = pitch_search(&buf[MAX_PERIOD >> 1..], buf, frame, 979);
                        *out = (MAX_PERIOD - lag).min(MAX_PERIOD - 2);
                    }
                });
                let t_doubling = fastest(&mut || {
                    let (mut period, mut gain) = (15usize, 0.0f32);
                    for (buf, &t0) in bufs.chunks_exact(buf_len).zip(&found) {
                        let mut index = t0;
                        let g =
                            remove_doubling(buf, MAX_PERIOD, 15, frame, &mut index, period, gain);
                        (period, gain) = (index, 0.7 * g);
                    }
                });
                let mut hash = 0u64;
                let t_both = fastest(&mut || {
                    let (mut period, mut gain) = (15usize, 0.0f32);
                    hash = 0;
                    for buf in bufs.chunks_exact(buf_len) {
                        let (index, g) = prefilter_search(buf, frame, period, gain);
                        (period, gain) = (index, 0.7 * g);
                        hash = hash.wrapping_mul(0x100_0000_01b3)
                            ^ (index as u64)
                            ^ ((g.to_bits() as u64) << 16);
                    }
                });
                let us = |t: f64| t * 1e6 / frames as f64;
                println!(
                    "{frame:5} {channels:2} {frames:6} | {:10.2} {:7.2} {:9.2} {:16.2} | {hash:016x}",
                    us(t_down),
                    us(t_search),
                    us(t_doubling),
                    us(t_both)
                );
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

    #[cfg(target_arch = "aarch64")]
    mod neon {
        use super::*;

        #[test]
        fn xcorr_by_sample_neon_matches_scalar_and_calls_the_side_work() {
            let mut rng = Rng::new(2);
            for kind in KINDS {
                for len in (1..=40).chain([60, 61, 120, 240, 241]) {
                    let x = samples(&mut rng, len, kind);
                    let y = samples(&mut rng, len + 4 * 16 - 1, kind);
                    let what = format!("{kind:?} len {len}");
                    macro_rules! check {
                        ($g:literal) => {{
                            let mut got = vec![f32::NAN; 4 * $g];
                            let mut calls = 0;
                            xcorr_by_sample_side_neon::<$g, _>(&x, &y, &mut got, &mut || {
                                calls += 1
                            });
                            assert_eq!(calls, len / 8, "{what} G {}", $g);
                            let mut want = vec![f32::NAN; 4 * $g];
                            xcorr_by_sample_scalar(&x, &y, &mut want, len);
                            assert_same(&got, &want, &format!("{what} G {}", $g));
                        }};
                    }
                    check!(1);
                    check!(2);
                    check!(3);
                    check!(5);
                    check!(7);
                    check!(8);
                    check!(16);
                }
            }
        }

        /// The kernel with shared windows gives the sums of the plain
        /// by-sample loop for every lag, and calls `side` as often as
        /// `skew_side_calls` says.
        #[test]
        fn xcorr_skewed_neon_matches_scalar_and_calls_the_side_work() {
            let mut rng = Rng::new(14);
            for kind in KINDS {
                for len in (32..=72).chain([240, 241, 242, 243, 332, 664]) {
                    for lags in [64usize, 68, 124, 128, 132, 244] {
                        let x = samples(&mut rng, len, kind);
                        let y = samples(&mut rng, len + lags - 1, kind);
                        let what = format!("{kind:?} len {len} lags {lags}");
                        let mut got = vec![f32::NAN; lags];
                        let mut calls = 0;
                        xcorr_skewed_neon(&x, &y, &mut got, &mut || calls += 1);
                        assert_eq!(calls, skew_side_calls(len, lags), "{what}");
                        let mut want = vec![f32::NAN; lags];
                        xcorr_by_sample_scalar(&x, &y, &mut want, len);
                        assert_same(&got, &want, &what);
                    }
                }
            }
        }

        #[test]
        fn pitch_xcorr_neon_follows_libopus_for_every_count_of_lags() {
            let mut rng = Rng::new(6);
            for kind in KINDS {
                for len in [1usize, 3, 4, 9, 30, 60, 240] {
                    for lags in (1..=40).chain([63, 64, 65, 66, 67, 155, 244, 245]) {
                        let x = samples(&mut rng, len, kind);
                        let y = samples(&mut rng, len + lags - 1, kind);
                        let mut got = vec![f32::NAN; lags];
                        pitch_xcorr_neon(&x, &y, &mut got);
                        let by_sample = lags & !3;
                        let want: Vec<f32> = (0..lags)
                            .map(|k| {
                                if k < by_sample {
                                    lag_by_sample(&x, &y[k..], len)
                                } else {
                                    inner_prod_model(&x, &y[k..k + len])
                                }
                            })
                            .collect();
                        assert_same(&got, &want, &format!("{kind:?} len {len} lags {lags}"));
                    }
                }
            }
        }

        #[test]
        fn inner_prod_many_matches_one_call_per_window() {
            let mut rng = Rng::new(7);
            for kind in KINDS {
                for n in [0usize, 1, 5, 8, 61, 480] {
                    for count in [0usize, 1, 3, 4, 9, 16, 17, 30] {
                        let x = samples(&mut rng, n, kind);
                        let y = samples(&mut rng, n + 40, kind);
                        let offsets: Vec<usize> = (0..count).map(|k| (7 * k) % 41).collect();
                        let mut got = vec![f32::NAN; count];
                        let mut calls = 0;
                        inner_prod_many_side(&x, &y, &offsets, n, &mut got, &mut || calls += 1);
                        assert_eq!(calls, count.div_ceil(MANY_PER_PASS) * (n / 4));
                        let want: Vec<f32> = offsets
                            .iter()
                            .map(|&o| inner_prod_model(&x, &y[o..o + n]))
                            .collect();
                        assert_same(&got, &want, &format!("{kind:?} n {n} count {count}"));
                        let (a, b) = dual_inner_prod(&x, &y, &y[3..], n);
                        let want = [
                            inner_prod_model(&x, &y[..n]),
                            inner_prod_model(&x, &y[3..3 + n]),
                        ];
                        assert_same(&[a, b], &want, &format!("{kind:?} n {n} dual"));
                    }
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

        /// The fused loop gives the decimation of `decimate2_scalar` (with
        /// libopus's first output) and the autocorrelation of
        /// `celt_lpc::autocorr` on it.
        #[test]
        fn decimate2_autocorr5_neon_matches_the_separate_steps() {
            let mut rng = Rng::new(11);
            for kind in KINDS {
                for n in (8..=40).chain([572, 632, 752, 991, 992, 1024]) {
                    let x0 = samples(&mut rng, 2 * n, kind);
                    let x1 = samples(&mut rng, 2 * n, kind);
                    for stereo in [false, true] {
                        let mut got = vec![f32::NAN; n];
                        let mut got_ac = [f32::NAN; 5];
                        let mut want = vec![f32::NAN; n];
                        if stereo {
                            decimate2_autocorr5_neon([&x0[..], &x1[..]], &mut got, &mut got_ac);
                            decimate2_scalar([&x0[..], &x1[..]], &mut want);
                            want[0] = mac(0.5 * x0[0], x0[1], 0.25) + mac(0.5 * x1[0], x1[1], 0.25);
                        } else {
                            decimate2_autocorr5_neon([&x0[..]], &mut got, &mut got_ac);
                            decimate2_scalar([&x0[..]], &mut want);
                            want[0] = mac(0.5 * x0[0], x0[1], 0.25);
                        }
                        let what = format!("{kind:?} n {n} stereo {stereo}");
                        assert_same(&got, &want, &what);
                        let mut want_ac = [f32::NAN; 5];
                        autocorr(&want, &mut want_ac, None, 0, 4, n);
                        assert_same(&got_ac, &want_ac, &what);
                    }
                }
            }
        }

        #[test]
        fn xcorr5_neon_matches_the_general_kernels() {
            let mut rng = Rng::new(12);
            for kind in KINDS {
                for len in (1..=20).chain([988, 1020]) {
                    let x = samples(&mut rng, len, kind);
                    let y = samples(&mut rng, len + 4, kind);
                    let mut got = [f32::NAN; 5];
                    xcorr5_neon(&x, &y, &mut got);
                    let mut want = [f32::NAN; 5];
                    xcorr_by_sample_scalar(&x, &y, &mut want[..4], len);
                    want[4] = inner_prod_model(&x, &y[4..4 + len]);
                    assert_same(&got, &want, &format!("{kind:?} len {len}"));
                }
            }
        }
    }
}
