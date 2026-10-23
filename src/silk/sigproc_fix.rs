use crate::silk::define::*;
use crate::silk::macros::*;

/// The second operand that makes the NEON multiply `vqdmulhq_s32` compute
/// `silk_smulwb`: `vqdmulhq_s32(a, smulwb_operand(b))` is `(a * b) >> 16`.
///
/// The instruction returns the upper half of `2 * a * (b << 15)`, which is
/// exactly `(a * b) >> 16`. The doubled product saturates only when both
/// factors are `i32::MIN`, and `b << 15` never is.
#[inline(always)]
pub(crate) const fn smulwb_operand(b: i16) -> i32 {
    (b as i32) << 15
}

#[inline]
pub fn silk_k2a_q16(a_q24: &mut [i32], rc_q16: &[i32], order: usize) {
    for k in 0..order {
        let rc = rc_q16[k];
        for n in 0..((k + 1) >> 1) {
            let tmp1 = a_q24[n];
            let tmp2 = a_q24[k - n - 1];
            a_q24[n] = silk_smlaww(tmp1, tmp2, rc);
            a_q24[k - n - 1] = silk_smlaww(tmp2, tmp1, rc);
        }
        a_q24[k] = -(rc << 8);
    }
}

/// Schur recursion with 64-bit multiplies (libopus `silk_schur64`): turns
/// the autocorrelation `c[..=order]` into the reflection coefficients
/// `rc_q16[..order]` and returns the residual energy.
///
/// Step `k` divides `C[k + 1][0]` by `C[0][1]` (libopus names) and then
/// updates both columns with the quotient. The next step cannot start its
/// division before these two values are updated, so the steps form one long
/// chain. The two values are therefore kept in local variables and updated
/// first; the rest of the columns, which later steps need, follows in a
/// plain loop over the two arrays.
pub fn silk_schur64(rc_q16: &mut [i32], c: &[i32], order: usize) -> i32 {
    if c[0] <= 0 {
        for v in rc_q16.iter_mut().take(order) {
            *v = 0;
        }
        return 0;
    }

    // `upper[i]` is libopus `C[i][0]` and `lower[i]` is `C[i][1]`.
    let mut upper = [0i32; MAX_SHAPE_LPC_ORDER + 1];
    let mut lower = [0i32; MAX_SHAPE_LPC_ORDER + 1];
    upper[..=order].copy_from_slice(&c[..=order]);
    lower[..=order].copy_from_slice(&c[..=order]);
    let rc_q16 = &mut rc_q16[..order];

    // `C[k + 1][0]` and `C[0][1]` of the current step.
    let mut num = upper[1];
    let mut den = lower[0];

    let mut k = 0;
    while k < order {
        if num.wrapping_abs() >= den {
            rc_q16[k] = if num > 0 { -64880 } else { 64880 };
            k += 1;
            break;
        }

        let rc_tmp_q31 = silk_div32_varq(num.wrapping_neg(), den, 31);
        rc_q16[k] = silk_rshift_round(rc_tmp_q31, 15);

        // libopus updates the pairs `(C[n + k + 1][0], C[n][1])` for
        // `n = 0..order - k`. Pair 0 is `(num, den)`; its first value is
        // never read again.
        den = den.wrapping_add(silk_smmul(num << 1, rc_tmp_q31));
        // Pair 1 gives the `num` of the next step. The last step has none.
        let upper_split = upper[k + 2..=order].split_first_mut();
        let lower_split = lower[1..order - k].split_first_mut();
        if let (Some((u1, upper_rest)), Some((l1, lower_rest))) = (upper_split, lower_split) {
            let (c_tmp1_q30, c_tmp2_q30) = (*u1, *l1);
            num = c_tmp1_q30.wrapping_add(silk_smmul(c_tmp2_q30 << 1, rc_tmp_q31));
            *l1 = c_tmp2_q30.wrapping_add(silk_smmul(c_tmp1_q30 << 1, rc_tmp_q31));
            for (u, l) in upper_rest.iter_mut().zip(lower_rest) {
                let (c_tmp1_q30, c_tmp2_q30) = (*u, *l);
                *u = c_tmp1_q30.wrapping_add(silk_smmul(c_tmp2_q30 << 1, rc_tmp_q31));
                *l = c_tmp2_q30.wrapping_add(silk_smmul(c_tmp1_q30 << 1, rc_tmp_q31));
            }
        }
        k += 1;
    }

    rc_q16[k..].fill(0);

    den.max(1)
}

/// Second-order filter on one channel (libopus `silk_biquad_alt_stride1`),
/// in place on the first `len` samples of `input_output`. `s` holds the two
/// state values.
pub fn silk_biquad_alt_stride1(
    input_output: &mut [i16],
    b_q28: &[i32],
    a_q28: &[i32],
    s: &mut [i32],
    len: usize,
) {
    // Fixed sizes, so that the loop has no bounds checks.
    let b_q28: &[i32; 3] = b_q28[..3].try_into().unwrap();
    let a_q28: &[i32; 2] = a_q28[..2].try_into().unwrap();
    let s: &mut [i32; 2] = (&mut s[..2]).try_into().unwrap();
    let a0_l_q28 = (-a_q28[0]) & 0x00003FFF;
    let a0_u_q28 = -a_q28[0] >> 14;
    let a1_l_q28 = (-a_q28[1]) & 0x00003FFF;
    let a1_u_q28 = -a_q28[1] >> 14;

    for item in input_output.iter_mut().take(len) {
        let inval = *item as i32;
        let out32_q14 = silk_smlawb(s[0], b_q28[0], inval) << 2;

        s[0] = s[1] + silk_rshift_round(silk_smulwb(out32_q14, a0_l_q28), 14);
        s[0] = silk_smlawb(s[0], out32_q14, a0_u_q28);
        s[0] = silk_smlawb(s[0], b_q28[1], inval);

        s[1] = silk_rshift_round(silk_smulwb(out32_q14, a1_l_q28), 14);
        s[1] = silk_smlawb(s[1], out32_q14, a1_u_q28);
        s[1] = silk_smlawb(s[1], b_q28[2], inval);

        *item = silk_sat16(silk_rshift(out32_q14 + (1 << 14) - 1, 14)) as i16;
    }
}

/// Second-order filter on two interleaved channels (libopus
/// `silk_biquad_alt_stride2`). `input_output` holds `len` pairs of samples,
/// `s` the two state values of channel 0 followed by those of channel 1.
pub fn silk_biquad_alt_stride2(
    input_output: &mut [i16],
    b_q28: &[i32],
    a_q28: &[i32],
    s: &mut [i32],
    len: usize,
) {
    // The kernels read and write exactly these elements.
    let input_output = &mut input_output[..2 * len];
    let b_q28: &[i32; 3] = b_q28[..3].try_into().unwrap();
    let a_q28: &[i32; 2] = a_q28[..2].try_into().unwrap();
    let s: &mut [i32; 4] = (&mut s[..4]).try_into().unwrap();

    #[cfg(target_arch = "aarch64")]
    // SAFETY: NEON is part of the aarch64 baseline. The kernel requires
    // `input_output.len()` to be even, which the slicing above guarantees.
    unsafe {
        silk_biquad_alt_stride2_neon(input_output, b_q28, a_q28, s);
    }
    #[cfg(not(target_arch = "aarch64"))]
    silk_biquad_alt_stride2_scalar(input_output, b_q28, a_q28, s);
}

/// Portable form of [`silk_biquad_alt_stride2`]. `input_output` holds pairs
/// of samples.
#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
fn silk_biquad_alt_stride2_scalar(
    input_output: &mut [i16],
    b_q28: &[i32; 3],
    a_q28: &[i32; 2],
    s: &mut [i32; 4],
) {
    let a0_l_q28 = (-a_q28[0]) & 0x00003FFF;
    let a0_u_q28 = -a_q28[0] >> 14;
    let a1_l_q28 = (-a_q28[1]) & 0x00003FFF;
    let a1_u_q28 = -a_q28[1] >> 14;

    let (pairs, _) = input_output.as_chunks_mut::<2>();
    for pair in pairs {
        let (in0, in1) = (pair[0] as i32, pair[1] as i32);
        let out32_q14_0 = silk_smlawb(s[0], b_q28[0], in0) << 2;
        let out32_q14_1 = silk_smlawb(s[2], b_q28[0], in1) << 2;

        s[0] = s[1].wrapping_add(silk_rshift_round(silk_smulwb(out32_q14_0, a0_l_q28), 14));
        s[2] = s[3].wrapping_add(silk_rshift_round(silk_smulwb(out32_q14_1, a0_l_q28), 14));
        s[0] = silk_smlawb(s[0], out32_q14_0, a0_u_q28);
        s[2] = silk_smlawb(s[2], out32_q14_1, a0_u_q28);
        s[0] = silk_smlawb(s[0], b_q28[1], in0);
        s[2] = silk_smlawb(s[2], b_q28[1], in1);

        s[1] = silk_rshift_round(silk_smulwb(out32_q14_0, a1_l_q28), 14);
        s[3] = silk_rshift_round(silk_smulwb(out32_q14_1, a1_l_q28), 14);
        s[1] = silk_smlawb(s[1], out32_q14_0, a1_u_q28);
        s[3] = silk_smlawb(s[3], out32_q14_1, a1_u_q28);
        s[1] = silk_smlawb(s[1], b_q28[2], in0);
        s[3] = silk_smlawb(s[3], b_q28[2], in1);

        pair[0] = silk_sat16(silk_rshift(out32_q14_0.wrapping_add((1 << 14) - 1), 14)) as i16;
        pair[1] = silk_sat16(silk_rshift(out32_q14_1.wrapping_add((1 << 14) - 1), 14)) as i16;
    }
}

/// Inner product of two equally long 16-bit vectors as a 32-bit sum that
/// wraps on overflow. Written so that the compiler vectorizes it.
#[inline(always)]
fn inner_prod16_wrapping(a: &[i16], b: &[i16]) -> i32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .fold(0i32, |sum, (&a, &b)| mac16_16(sum, a, b))
}

/// Cross-correlation at 4 neighbouring lags (libopus `xcorr_kernel`):
/// `sum[k] += x[0] * y[k] + x[1] * y[k + 1] + ... + x[len - 1] * y[k + len - 1]`
/// for `k = 0..4`. Reads `x[..len]` and `y[..len + 3]`. The sums wrap on
/// overflow.
#[inline(always)]
fn xcorr_kernel(x: &[i16], y: &[i16], sum: &mut [i32; 4], len: usize) {
    // The AVX2 kernel loads 8 elements of `y` at a time. Its last load ends
    // at `y[len + 3]`, one element after those the sums use.
    #[cfg(target_arch = "x86_64")]
    if x.len() >= len && y.len() > len + 3 && crate::compat::x86_has_avx2() {
        // SAFETY: AVX2 was detected. `x` has `len` elements and `y` has
        // `len + 4`, which covers every load of the kernel.
        unsafe { xcorr_kernel_avx2(x, y, sum, len) };
        return;
    }

    let x = &x[..len];
    let y = &y[..len + 3];

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        // SAFETY: this block is compiled only when the `neon` target feature
        // is enabled, which is all the kernel requires.
        unsafe { xcorr_kernel_neon(x, y, sum) }
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    xcorr_kernel_scalar(x, y, sum);
}

/// Portable form of [`xcorr_kernel`]: one plain inner product per lag.
/// `y` has 3 more elements than `x`.
#[cfg_attr(
    all(target_arch = "aarch64", target_feature = "neon"),
    allow(dead_code)
)]
#[inline(always)]
fn xcorr_kernel_scalar(x: &[i16], y: &[i16], sum: &mut [i32; 4]) {
    debug_assert_eq!(y.len(), x.len() + 3);
    for (k, s) in sum.iter_mut().enumerate() {
        *s = s.wrapping_add(inner_prod16_wrapping(x, &y[k..k + x.len()]));
    }
}

/// NEON form of [`xcorr_kernel`]. `y` has 3 more elements than `x`.
///
/// 16 samples of `x` are loaded once and multiplied with `y` at the 4 lags.
/// Each lag has two sums, one for the first 8 samples of such a block and
/// one for the last 8, so that consecutive multiply-accumulates do not wait
/// for each other.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[target_feature(enable = "neon")]
#[inline]
fn xcorr_kernel_neon(x: &[i16], y: &[i16], sum: &mut [i32; 4]) {
    use core::arch::aarch64::*;

    #[target_feature(enable = "neon")]
    #[inline]
    fn load(v: &[i16], at: usize) -> int16x8_t {
        let lanes: &[i16; 8] = v[at..at + 8].try_into().unwrap();
        // SAFETY: `lanes` refers to eight readable `i16`.
        unsafe { vld1q_s16(lanes.as_ptr()) }
    }

    /// Adds the 8 products `a[i] * b[i]` to `acc`. Lane `i % 4` receives
    /// product `i`.
    #[target_feature(enable = "neon")]
    #[inline]
    fn mac(acc: int32x4_t, a: int16x8_t, b: int16x8_t) -> int32x4_t {
        vmlal_high_s16(vmlal_s16(acc, vget_low_s16(a), vget_low_s16(b)), a, b)
    }

    /// Loading 8 lanes at offset `n` gives `8 - n` zero lanes followed by
    /// `n` lanes with all bits set.
    const LAST_LANES: [i16; 16] = [0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1];

    assert!(y.len() == x.len() + 3);

    if x.len() < 8 {
        for (j, &xj) in x.iter().enumerate() {
            for k in 0..4 {
                sum[k] = mac16_16(sum[k], xj, y[j + k]);
            }
        }
        return;
    }

    let mut acc_a = [vdupq_n_s32(0); 4];
    let mut acc_b = [vdupq_n_s32(0); 4];

    // Lag `k` multiplies `x[j..j + 8]` with `y[j + k..j + k + 8]`. With
    // `j + 8 <= x.len()` and `k <= 3` that stays inside `y`.
    let (blocks, rest) = x.as_chunks::<16>();
    for (i, xb) in blocks.iter().enumerate() {
        let yb = &y[16 * i..16 * i + 19];
        let x0 = load(xb, 0);
        let x1 = load(xb, 8);
        for k in 0..4 {
            acc_a[k] = mac(acc_a[k], x0, load(yb, k));
            acc_b[k] = mac(acc_b[k], x1, load(yb, k + 8));
        }
    }

    let mut left = rest.len();
    if left >= 8 {
        let at = x.len() - left;
        let yb = &y[at..at + 11];
        let x0 = load(x, at);
        for k in 0..4 {
            acc_a[k] = mac(acc_a[k], x0, load(yb, k));
        }
        left -= 8;
    }

    if left > 0 {
        // The 1 to 7 samples that are left are the end of the last 8
        // samples of `x`. The samples before them are already in the sums,
        // so they are replaced by zeros.
        let at = x.len() - 8;
        let yb = &y[at..at + 11];
        let x_last = vandq_s16(load(x, at), load(&LAST_LANES, left));
        for k in 0..4 {
            acc_b[k] = mac(acc_b[k], x_last, load(yb, k));
        }
    }

    for k in 0..4 {
        sum[k] = sum[k].wrapping_add(vaddvq_s32(vaddq_s32(acc_a[k], acc_b[k])));
    }
}

#[inline(always)]
fn mac16_16(a: i32, b: i16, c: i16) -> i32 {
    a.wrapping_add((b as i32).wrapping_mul(c as i32))
}

/// `results[k]` = sum of `x[j] * x[j + k]` over `j = 0..x.len() - k`, for
/// every lag `k` in `results`. The sums wrap on overflow. `results` has at
/// most `x.len() + 1` elements.
#[inline]
fn autocorr_sums(results: &mut [i32], x: &[i16]) {
    #[cfg(target_arch = "x86_64")]
    if crate::compat::x86_has_avx2() {
        // SAFETY: the processor has AVX2.
        unsafe { autocorr_sums_avx2(results, x) };
        return;
    }
    autocorr_sums_portable(results, x);
}

/// [`autocorr_sums`] compiled with AVX2 enabled.
///
/// AVX2 only changes how the compiler encodes and vectorizes the code. The
/// default x86-64 target has no 32-bit and no 32x32-to-64-bit vector
/// multiplication, which fixed-point code needs; AVX2 has both. The results
/// are the same bits.
///
/// # Safety
///
/// The processor must have AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn autocorr_sums_avx2(results: &mut [i32], x: &[i16]) {
    autocorr_sums_portable(results, x);
}

#[inline(always)]
fn autocorr_sums_portable(results: &mut [i32], x: &[i16]) {
    let n = x.len();
    let count = results.len();
    let mut k = 0;
    while k + 4 <= count {
        // The kernel adds the `len` products that all 4 lags have.
        let len = n - (k + 3);
        let mut sum = [0i32; 4];
        xcorr_kernel(x, &x[k..], &mut sum, len);
        // Lag `k + m` has `3 - m` more.
        for m in 0..3 {
            for j in len..n - (k + m) {
                sum[m] = mac16_16(sum[m], x[j], x[j + k + m]);
            }
        }
        results[k..k + 4].copy_from_slice(&sum);
        k += 4;
    }
    while k < count {
        results[k] = silk_inner_prod_aligned(x, &x[k..], n - k);
        k += 1;
    }
}

/// Autocorrelation of `input_data[..input_data_size]` at the lags
/// `0..correlation_count` (libopus `silk_autocorr`, which is CELT's
/// `_celt_autocorr` without window). The input is scaled down first if its
/// energy would overflow 32 bits, and the results are scaled so that
/// `results[0]` uses 29 or 30 bits. `scale` receives the total right shift.
pub fn silk_autocorr(
    results: &mut [i32],
    scale: &mut i32,
    input_data: &[i16],
    input_data_size: usize,
    correlation_count: usize,
) {
    #[inline]
    fn ec_ilog(x: u32) -> i32 {
        if x == 0 {
            0
        } else {
            32 - x.leading_zeros() as i32
        }
    }

    let n = input_data_size;
    // Lag `k` needs `n - k` products.
    assert!(correlation_count <= n + 1);
    let input = &input_data[..n];
    let results = &mut results[..correlation_count];

    // A first estimate of the energy decides the scaling of the input.
    let ac0_shift = ec_ilog((n + (n >> 4)) as u32) - 1;
    let mut ac0: i32 = 1 + ((n as i32) << 7);
    for &sample in input {
        ac0 += ((sample as i32) * (sample as i32)) >> ac0_shift;
    }
    ac0 += ac0 >> 7;

    let ac0_log2 = ec_ilog(ac0 as u32) - 1;
    let mut shift = (ac0_log2 - 30 + ac0_shift + 1) / 2;

    if shift > 0 {
        // The buffer exists only on this path, so a quiet input does not
        // pay for clearing it.
        let mut scaled = [0i16; PE_MAX_FRAME_LENGTH];
        for (dst, &sample) in scaled[..n].iter_mut().zip(input) {
            *dst = silk_rshift_round(sample as i32, shift) as i16;
        }
        autocorr_sums(results, &scaled[..n]);
    } else {
        shift = 0;
        autocorr_sums(results, input);
    }

    shift *= 2;
    if shift <= 0 {
        let add_shift = (-shift).min(30);
        results[0] += 1i32 << add_shift;
    }
    if results[0] > 0 && results[0] < 268435456 {
        let shift2 = 29 - ec_ilog(results[0] as u32);
        for v in results.iter_mut() {
            *v <<= shift2;
        }
        shift -= shift2;
    } else if results[0] >= 536870912 {
        let mut shift2 = 1;
        if results[0] >= 1073741824 {
            shift2 += 1;
        }
        for v in results.iter_mut() {
            *v >>= shift2;
        }
        shift += shift2;
    }

    *scale = shift;
}

/// Energy of `x[..len]` with the right shift that makes it fit 32 bits with
/// 2 bits of headroom (libopus `silk_sum_sqr_shift`, silk/sum_sqr_shift.c).
/// A first pass with a shift that is safe for `len` samples measures the
/// energy, a second pass uses the smallest shift that fits.
///
/// Both passes add the squares in pairs and shift the sum of each pair, as
/// libopus does. The result is the same on every target.
pub fn silk_sum_sqr_shift(energy: &mut i32, shift: &mut i32, x: &[i16], len: usize) {
    let x = &x[..len];

    let mut shft = 31 - silk_clz32(len as i32);
    let nrg = sum_sqr_pairs_shifted(x, len as i32, shft);
    shft = (shft + 3 - silk_clz32(nrg)).max(0);

    #[cfg(target_arch = "x86_64")]
    let nrg = if crate::compat::x86_has_avx2() {
        // SAFETY: AVX2 was detected, and `x` has exactly `len` elements.
        unsafe { silk_sum_sqr_shift_avx2(x, len, shft) }
    } else {
        sum_sqr_pairs_shifted(x, 0, shft)
    };
    #[cfg(not(target_arch = "x86_64"))]
    let nrg = sum_sqr_pairs_shifted(x, 0, shft);

    *shift = shft;
    *energy = nrg;
}

/// One pass of libopus's `silk_sum_sqr_shift`: squares summed in pairs, each
/// pair's sum shifted right by `shft` (as unsigned) and added to `init`. A
/// last sample without partner is shifted alone. The sum wraps at 32 bits.
#[inline(always)]
fn sum_sqr_pairs_shifted(x: &[i16], init: i32, shft: i32) -> i32 {
    let (pairs, last) = x.as_chunks::<2>();
    let mut nrg = init;
    for &[a, b] in pairs {
        let nrg_tmp = (silk_smulbb(a as i32, a as i32) as u32)
            .wrapping_add(silk_smulbb(b as i32, b as i32) as u32);
        nrg = nrg.wrapping_add((nrg_tmp >> shft) as i32);
    }
    if let [a] = last {
        let nrg_tmp = silk_smulbb(*a as i32, *a as i32) as u32;
        nrg = nrg.wrapping_add((nrg_tmp >> shft) as i32);
    }
    nrg
}

/// Inner product with each product shifted right by `scale` before it is
/// accumulated (libopus `silk_inner_prod_aligned_scale`).
pub fn silk_inner_prod_aligned_scale(
    in_vec1: &[i16],
    in_vec2: &[i16],
    scale: i32,
    len: usize,
) -> i32 {
    let mut sum: i32 = 0;
    for (&a, &b) in in_vec1[..len].iter().zip(&in_vec2[..len]) {
        sum = sum.wrapping_add(silk_smulbb(a as i32, b as i32) >> scale);
    }
    sum
}

/// Inner product of `ptr1[..len]` and `ptr2[..len]` as a 32-bit sum that
/// wraps on overflow (libopus `silk_inner_prod_aligned`).
#[inline(always)]
pub fn silk_inner_prod_aligned(ptr1: &[i16], ptr2: &[i16], len: usize) -> i32 {
    // The vector kernels read `len` elements through pointers, so both
    // lengths are checked here.
    let ptr1 = &ptr1[..len];
    let ptr2 = &ptr2[..len];

    #[cfg(target_arch = "aarch64")]
    // SAFETY: NEON is part of the aarch64 baseline. Both slices have `len`
    // elements, which is what the kernel requires.
    unsafe {
        silk_inner_prod_aligned_neon(ptr1, ptr2, len)
    }
    #[cfg(target_arch = "x86_64")]
    if crate::compat::x86_has_avx2() {
        // SAFETY: AVX2 was detected. Both slices have `len` elements, which
        // is what the kernel requires.
        return unsafe { silk_inner_prod_aligned_avx2(ptr1, ptr2, len) };
    }
    #[cfg(not(target_arch = "aarch64"))]
    silk_inner_prod_aligned_scalar(ptr1, ptr2, len)
}

/// Portable form of [`silk_inner_prod_aligned`].
#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
#[inline(always)]
fn silk_inner_prod_aligned_scalar(ptr1: &[i16], ptr2: &[i16], len: usize) -> i32 {
    let ptr1 = &ptr1[..len];
    let ptr2 = &ptr2[..len];
    let mut i = 0;
    let mut sum0 = 0i32;
    let mut sum1 = 0i32;
    let mut sum2 = 0i32;
    let mut sum3 = 0i32;
    let len4 = (len / 4) * 4;

    while i < len4 {
        sum0 = sum0.wrapping_add((ptr1[i] as i32).wrapping_mul(ptr2[i] as i32));
        sum1 = sum1.wrapping_add((ptr1[i + 1] as i32).wrapping_mul(ptr2[i + 1] as i32));
        sum2 = sum2.wrapping_add((ptr1[i + 2] as i32).wrapping_mul(ptr2[i + 2] as i32));
        sum3 = sum3.wrapping_add((ptr1[i + 3] as i32).wrapping_mul(ptr2[i + 3] as i32));
        i += 4;
    }

    while i < len {
        sum0 = sum0.wrapping_add((ptr1[i] as i32).wrapping_mul(ptr2[i] as i32));
        i += 1;
    }

    sum0.wrapping_add(sum1)
        .wrapping_add(sum2)
        .wrapping_add(sum3)
}

/// NEON form of [`silk_inner_prod_aligned`].
///
/// # Safety
///
/// Both slices must have at least `len` elements: the loads read 8 elements
/// at an offset `i` with `i + 8 <= len`.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn silk_inner_prod_aligned_neon(ptr1: &[i16], ptr2: &[i16], len: usize) -> i32 {
    use core::arch::aarch64::*;

    debug_assert!(ptr1.len() >= len && ptr2.len() >= len);

    let mut acc0 = vdupq_n_s32(0i32);
    let mut acc1 = vdupq_n_s32(0i32);
    let mut acc2 = vdupq_n_s32(0i32);
    let mut acc3 = vdupq_n_s32(0i32);

    let mut i = 0;

    while i + 32 <= len {
        let a0 = vld1q_s16(ptr1.as_ptr().add(i));
        let b0 = vld1q_s16(ptr2.as_ptr().add(i));
        let a1 = vld1q_s16(ptr1.as_ptr().add(i + 8));
        let b1 = vld1q_s16(ptr2.as_ptr().add(i + 8));
        let a2 = vld1q_s16(ptr1.as_ptr().add(i + 16));
        let b2 = vld1q_s16(ptr2.as_ptr().add(i + 16));
        let a3 = vld1q_s16(ptr1.as_ptr().add(i + 24));
        let b3 = vld1q_s16(ptr2.as_ptr().add(i + 24));

        acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
        acc0 = vmlal_high_s16(acc0, a0, b0);
        acc1 = vmlal_s16(acc1, vget_low_s16(a1), vget_low_s16(b1));
        acc1 = vmlal_high_s16(acc1, a1, b1);
        acc2 = vmlal_s16(acc2, vget_low_s16(a2), vget_low_s16(b2));
        acc2 = vmlal_high_s16(acc2, a2, b2);
        acc3 = vmlal_s16(acc3, vget_low_s16(a3), vget_low_s16(b3));
        acc3 = vmlal_high_s16(acc3, a3, b3);

        i += 32;
    }

    while i + 8 <= len {
        let a0 = vld1q_s16(ptr1.as_ptr().add(i));
        let b0 = vld1q_s16(ptr2.as_ptr().add(i));
        acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
        acc0 = vmlal_high_s16(acc0, a0, b0);
        i += 8;
    }

    let sum01 = vpaddq_s32(acc0, acc1);
    let sum23 = vpaddq_s32(acc2, acc3);
    let sum = vpaddq_s32(sum01, sum23);
    let wide = vpaddlq_s32(sum);
    let mut result = vaddvq_s64(wide);

    while i < len {
        result += (ptr1[i] as i64) * (ptr2[i] as i64);
        i += 1;
    }

    result as i32
}

pub fn silk_corr_vector_fix(
    x: &[i16],
    t: &[i16],
    l: usize,
    order: usize,
    xt: &mut [i32],
    rshifts: i32,
) {
    #[cfg(target_arch = "x86_64")]
    if crate::compat::x86_has_avx2() {
        // SAFETY: the processor has AVX2.
        unsafe { corr_vector_avx2(x, t, l, order, xt, rshifts) };
        return;
    }
    corr_vector_portable(x, t, l, order, xt, rshifts);
}

/// [`silk_corr_vector_fix`] compiled with AVX2 enabled. The results are the
/// same bits; see [`autocorr_sums_avx2`].
///
/// # Safety
///
/// The processor must have AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn corr_vector_avx2(
    x: &[i16],
    t: &[i16],
    l: usize,
    order: usize,
    xt: &mut [i32],
    rshifts: i32,
) {
    corr_vector_portable(x, t, l, order, xt, rshifts);
}

#[inline(always)]
fn corr_vector_portable(
    x: &[i16],
    t: &[i16],
    l: usize,
    order: usize,
    xt: &mut [i32],
    rshifts: i32,
) {
    let mut ptr1_idx = order - 1;
    if rshifts > 0 {
        for xt_val in xt[..order].iter_mut() {
            let mut inner_prod: i32 = 0;
            for i in 0..l {
                inner_prod = silk_add_rshift32(
                    inner_prod,
                    silk_smulbb(x[ptr1_idx + i] as i32, t[i] as i32),
                    rshifts,
                );
            }
            *xt_val = inner_prod;
            ptr1_idx = ptr1_idx.saturating_sub(1);
        }
    } else {
        for xt_val in xt[..order].iter_mut() {
            *xt_val = silk_inner_prod_aligned(&x[ptr1_idx..], t, l);
            ptr1_idx = ptr1_idx.saturating_sub(1);
        }
    }
}

pub fn silk_corr_matrix_fix(
    x: &[i16],
    l: usize,
    order: usize,
    xx: &mut [i32],
    nrg: &mut i32,
    rshifts: &mut i32,
) {
    #[cfg(target_arch = "x86_64")]
    if crate::compat::x86_has_avx2() {
        // SAFETY: the processor has AVX2.
        unsafe { corr_matrix_avx2(x, l, order, xx, nrg, rshifts) };
        return;
    }
    corr_matrix_portable(x, l, order, xx, nrg, rshifts);
}

/// [`silk_corr_matrix_fix`] compiled with AVX2 enabled. The results are the
/// same bits; see [`autocorr_sums_avx2`].
///
/// # Safety
///
/// The processor must have AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn corr_matrix_avx2(
    x: &[i16],
    l: usize,
    order: usize,
    xx: &mut [i32],
    nrg: &mut i32,
    rshifts: &mut i32,
) {
    corr_matrix_portable(x, l, order, xx, nrg, rshifts);
}

#[inline(always)]
fn corr_matrix_portable(
    x: &[i16],
    l: usize,
    order: usize,
    xx: &mut [i32],
    nrg: &mut i32,
    rshifts: &mut i32,
) {
    silk_sum_sqr_shift(nrg, rshifts, x, l + order - 1);
    let mut energy = *nrg;

    for xi in x.iter().take(order - 1) {
        energy -= silk_rshift32(silk_smulbb(*xi as i32, *xi as i32), *rshifts);
    }

    xx[0] = energy;
    let ptr1_start_idx = order - 1;
    for j in 1..order {
        energy = silk_sub32(
            energy,
            silk_rshift32(
                silk_smulbb(
                    x[ptr1_start_idx + l - j] as i32,
                    x[ptr1_start_idx + l - j] as i32,
                ),
                *rshifts,
            ),
        );
        energy = silk_add32(
            energy,
            silk_rshift32(
                silk_smulbb(x[ptr1_start_idx - j] as i32, x[ptr1_start_idx - j] as i32),
                *rshifts,
            ),
        );
        xx[j * order + j] = energy;
    }

    for lag in 1..order {
        let ptr1_idx = ptr1_start_idx;
        let ptr2_idx = ptr1_start_idx - lag;
        let mut inner_prod: i32 = 0;
        if *rshifts > 0 {
            for i in 0..l {
                inner_prod = silk_add_rshift32(
                    inner_prod,
                    silk_smulbb(x[ptr1_idx + i] as i32, x[ptr2_idx + i] as i32),
                    *rshifts,
                );
            }
        } else {
            inner_prod = silk_inner_prod_aligned(&x[ptr1_idx..], &x[ptr2_idx..], l);
        }
        xx[lag] = inner_prod;
        xx[lag * order] = inner_prod;

        for j in 1..(order - lag) {
            inner_prod = silk_sub32(
                inner_prod,
                silk_rshift32(
                    silk_smulbb(x[ptr1_idx + l - j] as i32, x[ptr2_idx + l - j] as i32),
                    *rshifts,
                ),
            );
            inner_prod = silk_add32(
                inner_prod,
                silk_rshift32(
                    silk_smulbb(x[ptr1_idx - j] as i32, x[ptr2_idx - j] as i32),
                    *rshifts,
                ),
            );
            xx[j * order + (lag + j)] = inner_prod;
            xx[(lag + j) * order + j] = inner_prod;
        }
    }
}

const FREQ_TABLE_Q16: [i16; 27] = [
    12111, 9804, 8235, 7100, 6239, 5565, 5022, 4575, 4202, 3885, 3612, 3375, 3167, 2984, 2820,
    2674, 2542, 2422, 2313, 2214, 2123, 2038, 1961, 1889, 1822, 1760, 1702,
];

/// `silk_smulwb` for constant evaluation.
const fn smulwb_const(a: i32, b: i32) -> i32 {
    (((a as i64) * (b as i16 as i64)) >> 16) as i32
}

/// Start of the sine recurrence for a window of `length` samples: the two
/// state values and the coefficient, all in Q16. `win_type` 1 is the rising
/// quarter of a sine, any other value the falling quarter.
const fn sine_window_start(win_type: i32, length: usize) -> (i32, i32, i32) {
    let f_q16 = FREQ_TABLE_Q16[(length >> 2) - 4] as i32;
    let c_q16 = smulwb_const(f_q16, -f_q16);
    if win_type == 1 {
        (0, f_q16 + (length as i32 >> 3), c_q16)
    } else {
        (
            1 << 16,
            (1 << 16) + (c_q16 >> 1) + (length as i32 >> 4),
            c_q16,
        )
    }
}

/// One step of the sine recurrence: the window factors of the next 4 samples
/// in Q16, and the state after them.
const fn sine_window_step(s0_q16: i32, s1_q16: i32, c_q16: i32) -> ([i32; 4], i32, i32) {
    let limit = 1 << 16;
    let f0 = (s0_q16 + s1_q16) >> 1;
    let f1 = s1_q16;
    let mut s0_q16 = smulwb_const(s1_q16, c_q16) + (s1_q16 << 1) - s0_q16 + 1;
    if s0_q16 > limit {
        s0_q16 = limit;
    }
    let f2 = (s0_q16 + s1_q16) >> 1;
    let f3 = s0_q16;
    let mut s1_q16 = smulwb_const(s0_q16, c_q16) + (s0_q16 << 1) - s1_q16;
    if s1_q16 > limit {
        s1_q16 = limit;
    }
    ([f0, f1, f2, f3], s0_q16, s1_q16)
}

/// All window factors for `N` samples: index 0 is `win_type` 1, index 1 is
/// `win_type` 2.
const fn sine_window_factors<const N: usize>() -> [[i32; N]; 2] {
    let mut tables = [[0i32; N]; 2];
    let mut t = 0;
    while t < 2 {
        let (mut s0_q16, mut s1_q16, c_q16) = sine_window_start(t as i32 + 1, N);
        let mut k = 0;
        while k < N {
            let (factors, next0, next1) = sine_window_step(s0_q16, s1_q16, c_q16);
            let mut i = 0;
            while i < 4 {
                // `silk_apply_sine_window` multiplies these in 32 bits.
                assert!(factors[i] >= 0 && factors[i] <= 1 << 16);
                tables[t][k + i] = factors[i];
                i += 1;
            }
            s0_q16 = next0;
            s1_q16 = next1;
            k += 4;
        }
        t += 1;
    }
    tables
}

// The window factors do not depend on the signal. They are computed at
// compile time for the lengths the encoder uses: 2 ms for the pitch analysis
// and 4 ms or 6 ms for the noise shaping analysis, at 8, 12 and 16 kHz.
static SINE_WINDOW_16: [[i32; 16]; 2] = sine_window_factors();
static SINE_WINDOW_24: [[i32; 24]; 2] = sine_window_factors();
static SINE_WINDOW_32: [[i32; 32]; 2] = sine_window_factors();
static SINE_WINDOW_48: [[i32; 48]; 2] = sine_window_factors();
static SINE_WINDOW_64: [[i32; 64]; 2] = sine_window_factors();
static SINE_WINDOW_72: [[i32; 72]; 2] = sine_window_factors();
static SINE_WINDOW_96: [[i32; 96]; 2] = sine_window_factors();

/// The precomputed window factors for `length` samples, if there are any.
#[inline]
fn sine_window_table(win_type: i32, length: usize) -> Option<&'static [i32]> {
    let t = if win_type == 1 { 0 } else { 1 };
    Some(match length {
        16 => &SINE_WINDOW_16[t],
        24 => &SINE_WINDOW_24[t],
        32 => &SINE_WINDOW_32[t],
        48 => &SINE_WINDOW_48[t],
        64 => &SINE_WINDOW_64[t],
        72 => &SINE_WINDOW_72[t],
        96 => &SINE_WINDOW_96[t],
        _ => return None,
    })
}

/// Multiplies `px[..length]` with a quarter of a sine wave and writes the
/// result to `px_win` (libopus `silk_apply_sine_window`). `win_type` 1 rises
/// from 0 to 1, `win_type` 2 falls from 1 to 0. `length` is a multiple of 4
/// from 16 to 120.
///
/// libopus computes the window with a recurrence while it multiplies. Each
/// value needs the two before it, so that loop cannot run faster than the
/// recurrence. The values depend only on `win_type` and `length`, so for the
/// lengths the encoder uses they come from a table.
pub fn silk_apply_sine_window(px_win: &mut [i16], px: &[i16], win_type: i32, length: usize) {
    if let Some(factors) = sine_window_table(win_type, length) {
        let px_win = &mut px_win[..length];
        let px = &px[..length];
        for ((win, &x), &factor_q16) in px_win.iter_mut().zip(px).zip(factors) {
            // `factor_q16` is at most 1 << 16, so the product fits 32 bits
            // and this equals `silk_smulwb(factor_q16, x)`.
            *win = (factor_q16.wrapping_mul(x as i32) >> 16) as i16;
        }
        return;
    }

    // Any other length: the same factors, computed on the way. As in
    // libopus, samples are handled in groups of 4.
    let (mut s0_q16, mut s1_q16, c_q16) = sine_window_start(win_type, length);
    let groups = length.div_ceil(4);
    let px_win = &mut px_win[..4 * groups];
    let px = &px[..4 * groups];
    let (px_win, _) = px_win.as_chunks_mut::<4>();
    let (px, _) = px.as_chunks::<4>();
    for (win, x) in px_win.iter_mut().zip(px) {
        let (factors, next0, next1) = sine_window_step(s0_q16, s1_q16, c_q16);
        for i in 0..4 {
            win[i] = silk_smulwb(factors[i], x[i] as i32) as i16;
        }
        s0_q16 = next0;
        s1_q16 = next1;
    }
}

/// Cross-correlation of `x[..len]` with `y` at the lags `0..max_pitch`
/// (libopus `celt_pitch_xcorr` on 16-bit data):
/// `xcorr[i] = x[0] * y[i] + x[1] * y[i + 1] + ... + x[len - 1] * y[i + len - 1]`.
/// The sums wrap on overflow. `y` should have `len + max_pitch - 1` elements;
/// if it is shorter, a sum stops where `y` ends.
#[inline(always)]
pub fn silk_pitch_xcorr(x: &[i16], y: &[i16], xcorr: &mut [i32], len: usize, max_pitch: usize) {
    debug_assert!(max_pitch > 0);
    debug_assert!(x.len() >= len);
    debug_assert!(xcorr.len() >= max_pitch);

    let y_len = y.len();

    // Four lags at a time, as long as `y` covers all four.
    let mut i = 0;
    while i + 3 < max_pitch && y_len >= i + len + 3 {
        let mut sum = [0i32; 4];
        xcorr_kernel(x, &y[i..], &mut sum, len);
        xcorr[i..i + 4].copy_from_slice(&sum);
        i += 4;
    }

    while i < max_pitch {
        let avail = len.min(y_len.saturating_sub(i));
        // `silk_inner_prod_aligned` does not check the slice lengths before
        // its kernel reads `avail` elements. `y[i..]` has them by the line
        // above; slicing `x` checks the other side.
        xcorr[i] = if avail > 0 {
            silk_inner_prod_aligned(&x[..avail], &y[i..], avail)
        } else {
            0
        };
        i += 1;
    }
}

/// Q-domain of the warped autocorrelation sums (libopus `QC`).
const WARPED_QC: i32 = 10;
/// Q-domain of the warped filter state (libopus `QS`).
const WARPED_QS: i32 = 13;

/// Autocorrelation of `input[..length]` on a warped frequency axis (libopus
/// `silk_warped_autocorrelation_FIX`). Writes `corr[0..=order]` and the
/// scaling of these values. `order` must be even, as in libopus.
///
/// The input runs through a chain of `order` first-order all-pass sections.
/// With `t[n][i]` the output of section `i` for sample `n`:
///
/// ```text
/// t[n][0]     = input[n] << 13
/// t[n][i + 1] = t[n - 1][i] + (((t[n - 1][i + 1] - t[n][i]) * warping_q16) >> 16)
/// corr[i]     = sum over n of (t[n][i] * t[n][0]) >> 16
/// ```
///
/// Values with `n = -1` are zero.
pub fn silk_warped_autocorrelation_fix(
    corr: &mut [i32],
    scale: &mut i32,
    input: &[i16],
    warping_q16: i32,
    length: usize,
    order: usize,
) {
    debug_assert!((order & 1) == 0);

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    if (order & 1) == 0
        && order <= MAX_SHAPE_LPC_ORDER
        && length <= input.len()
        && warping_q16.unsigned_abs() <= 65535
    {
        let input = &input[..length];
        // SAFETY: this block is compiled only when the `neon` target feature
        // is enabled, which is all the kernel requires.
        unsafe {
            match order.div_ceil(4) {
                0..=4 => warped_autocorr_neon::<4>(corr, scale, input, warping_q16, order),
                5 => warped_autocorr_neon::<5>(corr, scale, input, warping_q16, order),
                _ => warped_autocorr_neon::<6>(corr, scale, input, warping_q16, order),
            }
        }
        return;
    }

    // The x86 kernels take warping factors up to 2^15 - 1 only; the encoder
    // uses 0 to 16383.
    #[cfg(target_arch = "x86_64")]
    if (order & 1) == 0
        && order <= MAX_SHAPE_LPC_ORDER
        && length <= input.len()
        && (0..=i16::MAX as i32).contains(&warping_q16)
    {
        let input = &input[..length];
        if crate::compat::x86_has_avx2() {
            // SAFETY: the processor has AVX2.
            unsafe {
                match order.div_ceil(8) {
                    0..=2 => warped_autocorr_avx2::<2>(corr, scale, input, warping_q16, order),
                    _ => warped_autocorr_avx2::<3>(corr, scale, input, warping_q16, order),
                }
            }
            return;
        }
        if crate::compat::x86_has_sse41() {
            // SAFETY: the processor has SSE4.1.
            unsafe {
                match order.div_ceil(4) {
                    0..=4 => warped_autocorr_sse41::<4>(corr, scale, input, warping_q16, order),
                    5 => warped_autocorr_sse41::<5>(corr, scale, input, warping_q16, order),
                    _ => warped_autocorr_sse41::<6>(corr, scale, input, warping_q16, order),
                }
            }
            return;
        }
    }

    silk_warped_autocorrelation_scalar(corr, scale, input, warping_q16, length, order);
}

/// NEON form of [`silk_warped_autocorrelation_fix`] for up to `4 * K`
/// sections. Each value is computed as in libopus
/// `silk_warped_autocorrelation_FIX_neon`; the arrangement in the registers
/// differs.
///
/// `t[n][i]` needs `t[n - 1][i - 1]`, `t[n - 1][i]` and `t[n][i - 1]`. All
/// values with the same `n + i` are therefore independent of each other. At
/// the start of step `m` the registers hold `t[m - i][i]` for the sections
/// `i = 1..=4 * K`. The step adds their products with the samples `m - i` to
/// the sums. Then it computes the values of step `m + 1` from them and from
/// those of step `m - 1`. The filter state and the sums stay in registers for
/// the whole call.
///
/// Lane `l` of register `r` holds section `4 * K - K * l - r`. The section
/// below it is then in the same lane of register `r + 1`, so a register is
/// updated without moving lanes. Only the last register needs register 0
/// moved by one lane, with the new input sample as section 0.
///
/// The samples are needed in the same arrangement. In step `m + 1` a section
/// uses the sample that the section below it used in step `m`. The vector of
/// samples for register `r + 1` is therefore the vector for register `r` one
/// step later, and each step builds one new vector. The last 8 vectors are
/// kept in memory.
///
/// The caller guarantees `order <= 4 * K` and `|warping_q16| <= 65535`. With
/// that bound `vqdmulhq_s32(d, warping_q16 << 15)` never saturates and equals
/// `(d * warping_q16) >> 16`.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[target_feature(enable = "neon")]
#[inline(never)]
fn warped_autocorr_neon<const K: usize>(
    corr: &mut [i32],
    scale: &mut i32,
    input: &[i16],
    warping_q16: i32,
    order: usize,
) {
    use core::arch::aarch64::*;

    /// Number of sample vectors kept. A power of two above `K`.
    const RING: usize = 8;

    assert!(K >= 1 && K < RING && 4 * K <= MAX_SHAPE_LPC_ORDER && order <= 4 * K);
    debug_assert!(warping_q16.unsigned_abs() <= 65535);

    #[target_feature(enable = "neon")]
    #[inline]
    fn load(lanes: &[i32; 4]) -> int32x4_t {
        // SAFETY: `lanes` refers to four readable `i32`.
        unsafe { vld1q_s32(lanes.as_ptr()) }
    }

    #[target_feature(enable = "neon")]
    #[inline]
    fn store(lanes: &mut [i32; 4], v: int32x4_t) {
        // SAFETY: `lanes` refers to four writable `i32`.
        unsafe { vst1q_s32(lanes.as_mut_ptr(), v) }
    }

    let warping = vdupq_n_s32(warping_q16 << 15);
    let zero = vdupq_n_s32(0);
    // Sections of the current step.
    let mut cur = [zero; K];
    // For each lane of `cur`: the section below it, one step earlier.
    let mut below_before = [zero; K];
    let mut sum_lo = [vdupq_n_s64(0); K];
    let mut sum_hi = [vdupq_n_s64(0); K];
    // Vector `j` holds the samples `j - 3 * K`, `j - 2 * K`, `j - K` and `j`,
    // shifted left by 13; samples outside the input are zero. It is stored
    // in `samples[j % RING]`. Register `r` is multiplied with vector
    // `m - K + r` in step `m`.
    let mut samples = [[0i32; 4]; RING];

    // `newest` has sample `m`, shifted left by 13, in lane 0.
    let mut step = |m: usize, newest: int32x4_t| {
        // Index of the vector for register 0. It wraps below zero in the
        // first `K` steps; the ring is all zeros there, as it must be.
        let oldest = m.wrapping_sub(K);
        // The sections below those of the last register: register 0 moved by
        // one lane, then section 0. The same move turns vector `m - K` into
        // vector `m`.
        let below_last = vextq_s32::<1>(cur[0], newest);
        let x_new = vextq_s32::<1>(load(&samples[oldest % RING]), newest);
        for r in 0..K {
            let below = if r + 1 < K { cur[r + 1] } else { below_last };
            let x = load(&samples[oldest.wrapping_add(r) % RING]);
            sum_lo[r] =
                vsraq_n_s64::<16>(sum_lo[r], vmull_s32(vget_low_s32(cur[r]), vget_low_s32(x)));
            sum_hi[r] = vsraq_n_s64::<16>(sum_hi[r], vmull_high_s32(cur[r], x));
            cur[r] = vaddq_s32(
                below_before[r],
                vqdmulhq_s32(vsubq_s32(cur[r], below), warping),
            );
            below_before[r] = below;
        }
        store(&mut samples[m % RING], x_new);
    };
    for (m, &sample) in input.iter().enumerate() {
        step(m, vdupq_n_s32((sample as i32) << WARPED_QS));
    }
    // The last sample reaches section `order` after `order` more steps.
    for m in input.len()..input.len() + order {
        step(m, zero);
    }

    let mut sums = [[0i64; 4]; K];
    for r in 0..K {
        sums[r] = [
            vgetq_lane_s64::<0>(sum_lo[r]),
            vgetq_lane_s64::<1>(sum_lo[r]),
            vgetq_lane_s64::<0>(sum_hi[r]),
            vgetq_lane_s64::<1>(sum_hi[r]),
        ];
    }
    warped_autocorr_finish(corr, scale, input, order, &sums, 0);
}

/// Last part of the vector kernels: `sums[r][l]` is the sum of lane `l` of
/// register `r`, with `excess_per_step` too much per step (0 for NEON,
/// `2^47` for the x86 kernels, see the comment above `WARPED_RING`).
#[cfg(any(all(target_arch = "aarch64", target_feature = "neon"), target_arch = "x86_64"))]
#[inline]
fn warped_autocorr_finish<const L: usize, const K: usize>(
    corr: &mut [i32],
    scale: &mut i32,
    input: &[i16],
    order: usize,
    sums: &[[i64; L]; K],
    excess_per_step: i64,
) {
    let steps = (input.len() + order) as i64;
    let excess = steps.wrapping_mul(excess_per_step);

    // Section 0 is the input itself: ((x << 13) * (x << 13)) >> 16 = (x * x) << 10.
    let mut energy = 0i64;
    for &sample in input {
        energy += (sample as i32 * sample as i32) as i64;
    }
    let mut corr_qc = [0i64; MAX_SHAPE_LPC_ORDER + 1];
    corr_qc[0] = energy << WARPED_QC;
    for i in 1..=order {
        // Section `i` is in lane `d / K` of register `d % K`.
        let d = L * K - i;
        corr_qc[i] = sums[d % K][d / K].wrapping_sub(excess);
    }

    warped_autocorr_output(corr, scale, &corr_qc[..=order]);
}

// The x86 kernels of `silk_warped_autocorrelation_fix` follow the steps of
// `warped_autocorr_neon`, with `L` sections per register (4 for SSE4.1, 8
// for AVX2) and `K` registers. Lane `l` of register `r` holds section
// `L * K - K * l - r`, and vector `j` of samples holds the samples
// `j - (L - 1) * K`, ..., `j - K`, `j`.
//
// x86 has neither `vqdmulhq_s32` nor a 64-bit arithmetic shift, so two steps
// are done differently:
//
// - The filter update `(d * warping_q16) >> 16`: see `warped_mulw_sse41`.
// - The sums. `_mm_mul_epi32` multiplies lanes 0 and 2 into two 64-bit
//   products, and lanes 1 and 3 after a shuffle. Each product `p` must be
//   added as `p >> 16`, rounded down. `p ^ 2^63` is `p + 2^63` in 64 bits, so
//   a logical shift of it gives `(p >> 16) + 2^47`. The kernels add this
//   value, and at the end subtract `2^47` once for every step. The sums wrap
//   around in 64 bits, so the result is exact.

/// Number of sample vectors the x86 kernels keep. A power of two above `K`.
#[cfg(target_arch = "x86_64")]
const WARPED_RING: usize = 8;

/// `(d * w) >> 16` per 32-bit lane, rounded down, for `0 <= w < 2^15`.
/// `w_low` holds `w` in the low 16 bits of every lane, `w_high` in the high
/// 16 bits.
///
/// With `d = h * 2^16 + l`, `h = d >> 16` and `l` the low 16 bits taken as
/// unsigned: `(d * w) >> 16 = h * w + ((l * w) >> 16)`. `_mm_madd_epi16`
/// with `w_high` gives `l * 0 + h * w`; `_mm_mulhi_epu16` with `w_low` gives
/// `(l * w) >> 16` in the low half and `0` in the high half. Neither part
/// overflows, so the sum is the exact result.
///
/// The kernels pass `w_low` through `black_box` in every step.
/// `_mm_mulhi_epu16` is written in Rust as a product of zero-extended
/// values. Without the barrier the compiler moves the zero extension of the
/// constant `w_low` out of the loop and no longer recognizes the
/// instruction: with AVX2 it emitted 10 instructions in place of
/// `vpmulhuw`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
#[inline]
fn warped_mulw_sse41(d: __m128i, w_low: __m128i, w_high: __m128i) -> __m128i {
    _mm_add_epi32(_mm_madd_epi16(d, w_high), _mm_mulhi_epu16(d, w_low))
}

/// [`warped_mulw_sse41`] for 8 lanes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
fn warped_mulw_avx2(d: __m256i, w_low: __m256i, w_high: __m256i) -> __m256i {
    _mm256_add_epi32(_mm256_madd_epi16(d, w_high), _mm256_mulhi_epu16(d, w_low))
}

/// SSE4.1 kernel of [`silk_warped_autocorrelation_fix`], 4 sections per
/// register, for up to `4 * K` sections, and warping factors from 0 to
/// `2^15 - 1`. See the comment above [`WARPED_RING`].
///
/// # Safety
///
/// The processor must have SSE4.1.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
#[inline(never)]
unsafe fn warped_autocorr_sse41<const K: usize>(
    corr: &mut [i32],
    scale: &mut i32,
    input: &[i16],
    warping_q16: i32,
    order: usize,
) {
    assert!(K >= 1 && K < WARPED_RING && 4 * K <= MAX_SHAPE_LPC_ORDER && order <= 4 * K);
    assert!((0..=i16::MAX as i32).contains(&warping_q16));

    let w_low = _mm_set1_epi32(warping_q16);
    let w_high = _mm_set1_epi32(warping_q16 << 16);
    let bias = _mm_set1_epi64x(i64::MIN);
    let zero = _mm_setzero_si128();
    // Sections of the current step.
    let mut cur = [zero; K];
    // For each lane of `cur`: the section below it, one step earlier.
    let mut below_before = [zero; K];
    // Sums of the sections in lanes 0 and 2 (`even`) and 1 and 3 (`odd`).
    let mut sum_even = [zero; K];
    let mut sum_odd = [zero; K];
    // `samples[j % WARPED_RING]` holds vector `j`, shifted left by 13, and
    // the same vector with lanes 1 and 3 moved to lanes 0 and 2. Samples
    // outside the input are zero.
    let mut samples = [[zero; 2]; WARPED_RING];

    // `newest` has sample `m`, shifted left by 13, in lane 0.
    let mut step = |m: usize, newest: __m128i| {
        let oldest = m.wrapping_sub(K);
        // See `warped_mulw_sse41`.
        let w_low = core::hint::black_box(w_low);
        // Register 0 moved down by one lane, with section 0 on top; the
        // same move turns vector `m - K` into vector `m`.
        let below_last = _mm_alignr_epi8::<4>(newest, cur[0]);
        let x_new = _mm_alignr_epi8::<4>(newest, samples[oldest % WARPED_RING][0]);
        for r in 0..K {
            let below = if r + 1 < K { cur[r + 1] } else { below_last };
            let [x, x_odd] = samples[oldest.wrapping_add(r) % WARPED_RING];
            let even = _mm_mul_epi32(cur[r], x);
            let odd = _mm_mul_epi32(_mm_shuffle_epi32::<0xF5>(cur[r]), x_odd);
            sum_even[r] =
                _mm_add_epi64(sum_even[r], _mm_srli_epi64::<16>(_mm_xor_si128(even, bias)));
            sum_odd[r] = _mm_add_epi64(sum_odd[r], _mm_srli_epi64::<16>(_mm_xor_si128(odd, bias)));
            let d = _mm_sub_epi32(cur[r], below);
            cur[r] = _mm_add_epi32(below_before[r], warped_mulw_sse41(d, w_low, w_high));
            below_before[r] = below;
        }
        samples[m % WARPED_RING] = [x_new, _mm_shuffle_epi32::<0xF5>(x_new)];
    };
    for (m, &sample) in input.iter().enumerate() {
        step(m, _mm_cvtsi32_si128((sample as i32) << WARPED_QS));
    }
    // The last sample reaches section `order` after `order` more steps.
    for m in input.len()..input.len() + order {
        step(m, zero);
    }

    let mut sums = [[0i64; 4]; K];
    for r in 0..K {
        sums[r] = [
            _mm_cvtsi128_si64(sum_even[r]),
            _mm_cvtsi128_si64(sum_odd[r]),
            _mm_extract_epi64::<1>(sum_even[r]),
            _mm_extract_epi64::<1>(sum_odd[r]),
        ];
    }
    warped_autocorr_finish(corr, scale, input, order, &sums, 1 << 47);
}

/// AVX2 kernel of [`silk_warped_autocorrelation_fix`], 8 sections per
/// register, for up to `8 * K` sections. The same steps as
/// [`warped_autocorr_sse41`]; only the move by one lane crosses the two
/// 128-bit halves.
///
/// # Safety
///
/// The processor must have AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline(never)]
unsafe fn warped_autocorr_avx2<const K: usize>(
    corr: &mut [i32],
    scale: &mut i32,
    input: &[i16],
    warping_q16: i32,
    order: usize,
) {
    assert!(K >= 1 && K < WARPED_RING && 8 * K <= MAX_SHAPE_LPC_ORDER && order <= 8 * K);
    assert!((0..=i16::MAX as i32).contains(&warping_q16));

    /// Lanes 1 to 7 of `v` in lanes 0 to 6, and lane 0 of `newest` in
    /// lane 7.
    #[target_feature(enable = "avx2")]
    #[inline]
    fn shift_in(v: __m256i, newest: __m256i) -> __m256i {
        // The upper half of `v` and the lower half of `newest`.
        let upper = _mm256_permute2x128_si256::<0x21>(v, newest);
        _mm256_alignr_epi8::<4>(upper, v)
    }

    let w_low = _mm256_set1_epi32(warping_q16);
    let w_high = _mm256_set1_epi32(warping_q16 << 16);
    let bias = _mm256_set1_epi64x(i64::MIN);
    let zero = _mm256_setzero_si256();
    let mut cur = [zero; K];
    let mut below_before = [zero; K];
    // Sums of the sections in lanes 0, 2, 4 and 6 (`even`) and 1, 3, 5 and
    // 7 (`odd`).
    let mut sum_even = [zero; K];
    let mut sum_odd = [zero; K];
    let mut samples = [[zero; 2]; WARPED_RING];

    let mut step = |m: usize, newest: __m256i| {
        let oldest = m.wrapping_sub(K);
        // See `warped_mulw_sse41`.
        let w_low = core::hint::black_box(w_low);
        let below_last = shift_in(cur[0], newest);
        let x_new = shift_in(samples[oldest % WARPED_RING][0], newest);
        for r in 0..K {
            let below = if r + 1 < K { cur[r + 1] } else { below_last };
            let [x, x_odd] = samples[oldest.wrapping_add(r) % WARPED_RING];
            let even = _mm256_mul_epi32(cur[r], x);
            let odd = _mm256_mul_epi32(_mm256_shuffle_epi32::<0xF5>(cur[r]), x_odd);
            sum_even[r] = _mm256_add_epi64(
                sum_even[r],
                _mm256_srli_epi64::<16>(_mm256_xor_si256(even, bias)),
            );
            sum_odd[r] =
                _mm256_add_epi64(sum_odd[r], _mm256_srli_epi64::<16>(_mm256_xor_si256(odd, bias)));
            let d = _mm256_sub_epi32(cur[r], below);
            cur[r] = _mm256_add_epi32(below_before[r], warped_mulw_avx2(d, w_low, w_high));
            below_before[r] = below;
        }
        samples[m % WARPED_RING] = [x_new, _mm256_shuffle_epi32::<0xF5>(x_new)];
    };
    for (m, &sample) in input.iter().enumerate() {
        step(m, _mm256_castsi128_si256(_mm_cvtsi32_si128((sample as i32) << WARPED_QS)));
    }
    for m in input.len()..input.len() + order {
        step(m, zero);
    }

    let mut sums = [[0i64; 8]; K];
    for r in 0..K {
        sums[r] = [
            _mm256_extract_epi64::<0>(sum_even[r]),
            _mm256_extract_epi64::<0>(sum_odd[r]),
            _mm256_extract_epi64::<1>(sum_even[r]),
            _mm256_extract_epi64::<1>(sum_odd[r]),
            _mm256_extract_epi64::<2>(sum_even[r]),
            _mm256_extract_epi64::<2>(sum_odd[r]),
            _mm256_extract_epi64::<3>(sum_even[r]),
            _mm256_extract_epi64::<3>(sum_odd[r]),
        ];
    }
    warped_autocorr_finish(corr, scale, input, order, &sums, 1 << 47);
}

/// Portable form of [`silk_warped_autocorrelation_fix`].
///
/// One section needs the result of the section before it, so a single
/// sample is a chain of `order` dependent steps. Four input samples are
/// therefore run through the sections together: their chains do not wait for
/// each other, and the processor works on all four at the same time.
fn silk_warped_autocorrelation_scalar(
    corr: &mut [i32],
    scale: &mut i32,
    input: &[i16],
    warping_q16: i32,
    length: usize,
    order: usize,
) {
    let mut state_qs = [0i32; MAX_SHAPE_LPC_ORDER + 1];
    let mut corr_qc = [0i64; MAX_SHAPE_LPC_ORDER + 1];
    let state_qs = &mut state_qs[..=order];
    let corr_qc = &mut corr_qc[..=order];

    let input = &input[..length.min(input.len())];
    let (blocks, rest) = input.as_chunks::<4>();
    for x in blocks {
        let x_qs = x.map(|sample| (sample as i32) << WARPED_QS);
        warped_autocorr_pass(state_qs, corr_qc, x_qs, warping_q16);
    }
    for &x in rest {
        warped_autocorr_pass(state_qs, corr_qc, [(x as i32) << WARPED_QS], warping_q16);
    }

    warped_autocorr_output(corr, scale, corr_qc);
}

/// Runs `N` consecutive input samples `x_qs` (already shifted left by 13)
/// through all sections of the warped filter and adds their products to the
/// sums. `state_qs[i]` holds `t[n - 1][i]` of the last sample before `x_qs`
/// on entry and of the last sample of `x_qs` on return. Both slices have
/// `order + 1` elements.
#[inline(always)]
fn warped_autocorr_pass<const N: usize>(
    state_qs: &mut [i32],
    corr_qc: &mut [i64],
    x_qs: [i32; N],
    warping_q16: i32,
) {
    let order = state_qs.len() - 1;
    let corr_qc = &mut corr_qc[..=order];

    // `cur[k]` is `t[n + k][i]`: section `i` of the `k`-th sample.
    let mut cur = x_qs;
    // Section `i` of the sample before `x_qs`.
    let mut before = state_qs[0];
    for i in 0..order {
        let before_next = state_qs[i + 1];
        let mut next = [0i32; N];
        next[0] = silk_smlaww(before, before_next.wrapping_sub(cur[0]), warping_q16);
        for k in 1..N {
            next[k] = silk_smlaww(cur[k - 1], next[k - 1].wrapping_sub(cur[k]), warping_q16);
        }

        let mut sum = corr_qc[i];
        for k in 0..N {
            sum += silk_rshift64(silk_smull(cur[k], x_qs[k]), 2 * WARPED_QS - WARPED_QC);
        }
        corr_qc[i] = sum;
        state_qs[i] = cur[N - 1];

        cur = next;
        before = before_next;
    }

    let mut sum = corr_qc[order];
    for k in 0..N {
        sum += silk_rshift64(silk_smull(cur[k], x_qs[k]), 2 * WARPED_QS - WARPED_QC);
    }
    corr_qc[order] = sum;
    state_qs[order] = cur[N - 1];
}

/// Scales the 64-bit sums `corr_qc[0..=order]` to 32 bits and writes them to
/// `corr[0..=order]`. `scale` receives the scaling that was applied.
#[inline(always)]
fn warped_autocorr_output(corr: &mut [i32], scale: &mut i32, corr_qc: &[i64]) {
    let corr = &mut corr[..corr_qc.len()];
    let mut lsh = silk_clz64(corr_qc[0]) - 35;
    lsh = silk_limit_32(lsh, -12 - WARPED_QC, 30 - WARPED_QC);
    *scale = -(WARPED_QC + lsh);
    if lsh >= 0 {
        for (c, &c_qc) in corr.iter_mut().zip(corr_qc) {
            *c = (c_qc << lsh) as i32;
        }
    } else {
        for (c, &c_qc) in corr.iter_mut().zip(corr_qc) {
            *c = (c_qc >> (-lsh)) as i32;
        }
    }
}

pub fn silk_schur(rc_q15: &mut [i16], c: &[i32], order: usize) -> i32 {
    let mut c_inner = [[0i32; 2]; MAX_LPC_ORDER + 1];
    let mut ctmp1: i32;
    let mut ctmp2: i32;
    let mut rc_tmp_q15: i32;

    assert!(order <= MAX_LPC_ORDER);

    let lz = c[0].leading_zeros() as i32;

    if lz < 2 {
        for i in 0..=order {
            c_inner[i][0] = c[i] >> 1;
            c_inner[i][1] = c[i] >> 1;
        }
    } else if lz > 2 {
        let lz_adj = lz - 2;
        for i in 0..=order {
            c_inner[i][0] = c[i] << lz_adj;
            c_inner[i][1] = c[i] << lz_adj;
        }
    } else {
        for i in 0..=order {
            c_inner[i][0] = c[i];
            c_inner[i][1] = c[i];
        }
    }

    for k in 0..order {
        if c_inner[k + 1][0].abs() >= c_inner[0][1] {
            if c_inner[k + 1][0] > 0 {
                rc_q15[k] = -32440;
            } else {
                rc_q15[k] = 32440;
            }
            return c_inner[0][1];
        }

        rc_tmp_q15 = -silk_div32_16(c_inner[k + 1][0], (c_inner[0][1] >> 15).max(1));

        rc_tmp_q15 = silk_sat16(rc_tmp_q15);

        rc_q15[k] = rc_tmp_q15 as i16;

        for n in 0..order - k {
            ctmp1 = c_inner[n + k + 1][0];
            ctmp2 = c_inner[n][1];
            c_inner[n + k + 1][0] = silk_smlawb(ctmp1, ctmp2 << 1, rc_tmp_q15);
            c_inner[n][1] = silk_smlawb(ctmp2, ctmp1 << 1, rc_tmp_q15);
        }
    }

    c_inner[0][1]
}

pub fn silk_k2a(a_q24: &mut [i32], rc_q15: &[i16], order: usize) {
    for k in 0..order {
        let rc = rc_q15[k] as i32;
        for n in 0..(k + 1) >> 1 {
            let tmp1 = a_q24[n];
            let tmp2 = a_q24[k - n - 1];
            a_q24[n] = silk_smlawb(tmp1, tmp2 << 1, rc);
            a_q24[k - n - 1] = silk_smlawb(tmp2, tmp1 << 1, rc);
        }
        a_q24[k] = -(rc << 9);
    }
}

pub fn silk_bwexpander(ar: &mut [i16], d: usize, mut chirp_q16: i32) {
    let chirp_minus_one_q16 = chirp_q16 - 65536;

    for ar_val in ar[..d - 1].iter_mut() {
        *ar_val = silk_rshift_round((*ar_val as i32).wrapping_mul(chirp_q16), 16) as i16;
        chirp_q16 += silk_rshift_round(chirp_q16.wrapping_mul(chirp_minus_one_q16), 16);
    }
    ar[d - 1] = silk_rshift_round((ar[d - 1] as i32).wrapping_mul(chirp_q16), 16) as i16;
}

/// LPC analysis filter (libopus `silk_LPC_analysis_filter`): the prediction
/// residual of `input[..len]` for the `d` coefficients `b` in Q12.
///
/// ```text
/// out[ix] = 0                                                   for ix < d
/// out[ix] = sat16(round((input[ix] << 12) - sum_j b[j] * input[ix - 1 - j]) >> 12))
/// ```
///
/// The sum runs over `j = 0..d` and wraps at 32 bits. `d` is even and at
/// least 6, as in libopus.
pub fn silk_lpc_analysis_filter(
    out: &mut [i16],
    input: &[i16],
    b: &[i16],
    len: usize,
    d: usize,
    _arch: i32,
) {
    assert!(d >= 6);
    assert!((d & 1) == 0);
    assert!(d <= len);

    // Every implementation below reads and writes exactly `len` samples.
    let out = &mut out[..len];
    let input = &input[..len];

    #[cfg(target_arch = "x86_64")]
    if d <= 16 && crate::compat::x86_has_avx2() {
        // SAFETY: AVX2 was detected. `out` and `input` have `len` elements
        // and `d` is at most 16; the kernel checks `b` itself.
        unsafe {
            silk_lpc_analysis_filter_avx2(out, input, b, len, d);
        }
        return;
    }

    // One copy of the loop per order the codec uses. With the number of
    // taps known, the compiler unrolls the sum and vectorizes over the
    // outputs.
    match d {
        6 => lpc_analysis_filter_taps(out, input, &b[..6]),
        8 => lpc_analysis_filter_taps(out, input, &b[..8]),
        10 => lpc_analysis_filter_taps(out, input, &b[..10]),
        12 => lpc_analysis_filter_taps(out, input, &b[..12]),
        14 => lpc_analysis_filter_taps(out, input, &b[..14]),
        16 => lpc_analysis_filter_taps(out, input, &b[..16]),
        _ => lpc_analysis_filter_taps(out, input, &b[..d]),
    }
}

/// [`silk_lpc_analysis_filter`] for `b.len()` taps. `out` and `input` have
/// the same length, at least `b.len()`.
#[inline(always)]
fn lpc_analysis_filter_taps(out: &mut [i16], input: &[i16], b: &[i16]) {
    let d = b.len();
    debug_assert_eq!(out.len(), input.len());
    let (zeros, out) = out.split_at_mut(d);
    zeros.fill(0);
    // Each window is the `d` samples before an output sample, then the
    // sample itself.
    for (o, window) in out.iter_mut().zip(input.windows(d + 1)) {
        let (history, x) = window.split_at(d);
        let mut sum = 0i32;
        for (&sample, &coef) in history.iter().rev().zip(b) {
            sum = sum.wrapping_add(silk_smulbb(sample as i32, coef as i32));
        }
        let out32_q12 = ((x[0] as i32) << 12).wrapping_sub(sum);
        *o = silk_sat16(silk_rshift_round(out32_q12, 12)) as i16;
    }
}

pub fn silk_scale_copy_vector16(
    data_out: &mut [i16],
    data_in: &[i16],
    gain_q16: i32,
    data_size: usize,
) {
    for i in 0..data_size {
        let tmp32 = silk_smulwb(gain_q16, data_in[i] as i32);
        data_out[i] = silk_sat16(tmp32) as i16;
    }
}

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn silk_inner_prod_aligned_avx2(ptr1: &[i16], ptr2: &[i16], len: usize) -> i32 {
    // The caller guarantees this; checked here too. Every load below stays inside `len`.
    assert!(ptr1.len() >= len && ptr2.len() >= len);
    let mut acc0 = _mm256_setzero_si256();
    let mut acc1 = _mm256_setzero_si256();
    let mut i = 0;

    while i + 32 <= len {
        let a0 = _mm256_loadu_si256(ptr1.as_ptr().add(i) as *const __m256i);
        let b0 = _mm256_loadu_si256(ptr2.as_ptr().add(i) as *const __m256i);
        let a1 = _mm256_loadu_si256(ptr1.as_ptr().add(i + 16) as *const __m256i);
        let b1 = _mm256_loadu_si256(ptr2.as_ptr().add(i + 16) as *const __m256i);

        acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(a0, b0));
        acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(a1, b1));
        i += 32;
    }
    while i + 16 <= len {
        let a = _mm256_loadu_si256(ptr1.as_ptr().add(i) as *const __m256i);
        let b = _mm256_loadu_si256(ptr2.as_ptr().add(i) as *const __m256i);
        acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(a, b));
        i += 16;
    }

    let acc = _mm256_add_epi32(acc0, acc1);
    let hi = _mm256_extracti128_si256(acc, 1);
    let lo = _mm256_castsi256_si128(acc);
    let sum4 = _mm_add_epi32(lo, hi);
    let sum2 = _mm_add_epi32(sum4, _mm_srli_si128(sum4, 8));
    let sum1 = _mm_add_epi32(sum2, _mm_srli_si128(sum2, 4));
    let mut result = _mm_cvtsi128_si32(sum1);

    if i + 8 <= len {
        let a = _mm_loadu_si128(ptr1.as_ptr().add(i) as *const __m128i);
        let b = _mm_loadu_si128(ptr2.as_ptr().add(i) as *const __m128i);
        let p = _mm_madd_epi16(a, b);
        let p2 = _mm_add_epi32(p, _mm_srli_si128(p, 8));
        let p1 = _mm_add_epi32(p2, _mm_srli_si128(p2, 4));
        result = result.wrapping_add(_mm_cvtsi128_si32(p1));
        i += 8;
    }

    while i < len {
        result = result.wrapping_add((ptr1[i] as i32).wrapping_mul(ptr2[i] as i32));
        i += 1;
    }

    result
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn xcorr_kernel_avx2(x: &[i16], y: &[i16], sum: &mut [i32; 4], len: usize) {
    // Each 8-sample load of `y` starts at `i <= len - 4`, so the last one
    // ends at `y[len + 3]`: `y` needs `len + 4` samples, as the caller
    // checks.
    assert!(x.len() >= len && y.len() >= len + 4);

    let mut acc = _mm_setzero_si128();
    let mut i = 0;

    while i + 4 <= len {
        let x4 = _mm_loadl_epi64(x.as_ptr().add(i) as *const __m128i);
        let xdup = _mm_unpacklo_epi64(x4, x4);

        let y8 = _mm_loadu_si128(y.as_ptr().add(i) as *const __m128i);
        let y1 = _mm_bsrli_si128(y8, 2);
        let y2 = _mm_bsrli_si128(y8, 4);
        let y3 = _mm_bsrli_si128(y8, 6);

        let y4_01 = _mm_unpacklo_epi64(y8, y1);

        let y4_23 = _mm_unpacklo_epi64(y2, y3);

        let p01 = _mm_madd_epi16(xdup, y4_01);
        let p23 = _mm_madd_epi16(xdup, y4_23);

        acc = _mm_add_epi32(acc, _mm_hadd_epi32(p01, p23));

        i += 4;
    }

    let mut tmp = [0i32; 4];
    _mm_storeu_si128(tmp.as_mut_ptr() as *mut __m128i, acc);
    sum[0] = sum[0].wrapping_add(tmp[0]);
    sum[1] = sum[1].wrapping_add(tmp[1]);
    sum[2] = sum[2].wrapping_add(tmp[2]);
    sum[3] = sum[3].wrapping_add(tmp[3]);

    let mut y_0 = y[i];
    let mut y_1 = y[i + 1];
    let mut y_2 = y[i + 2];
    for j in i..len {
        let y_3 = y[j + 3];
        let xv = x[j] as i32;
        sum[0] = sum[0].wrapping_add(xv.wrapping_mul(y_0 as i32));
        sum[1] = sum[1].wrapping_add(xv.wrapping_mul(y_1 as i32));
        sum[2] = sum[2].wrapping_add(xv.wrapping_mul(y_2 as i32));
        sum[3] = sum[3].wrapping_add(xv.wrapping_mul(y_3 as i32));
        y_0 = y_1;
        y_1 = y_2;
        y_2 = y_3;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn silk_sum_sqr_shift_avx2(x: &[i16], len: usize, shft: i32) -> i32 {
    // The caller guarantees this; checked here too. Every load below stays inside `len`.
    assert!(x.len() >= len);
    let mut acc = _mm256_setzero_si256();
    let mut i = 0;

    while i + 16 <= len {
        let v = _mm256_loadu_si256(x.as_ptr().add(i) as *const __m256i);

        let sq = _mm256_madd_epi16(v, v);

        let vshift = _mm256_set1_epi32(shft);
        let shifted = if shft > 0 {
            _mm256_srlv_epi32(sq, vshift)
        } else {
            sq
        };
        acc = _mm256_add_epi32(acc, shifted);
        i += 16;
    }

    let hi = _mm256_extracti128_si256(acc, 1);
    let lo = _mm256_castsi256_si128(acc);
    let sum4 = _mm_add_epi32(lo, hi);
    let sum2 = _mm_add_epi32(sum4, _mm_srli_si128(sum4, 8));
    let sum1 = _mm_add_epi32(sum2, _mm_srli_si128(sum2, 4));
    let nrg = _mm_cvtsi128_si32(sum1);

    // The samples after the last block of 16: in pairs with the sum of each
    // pair shifted, as in the blocks above and in libopus. `i` is even, so
    // the pairs are the same ones.
    sum_sqr_pairs_shifted(&x[i..len], nrg, shft)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn silk_lpc_analysis_filter_avx2(
    out: &mut [i16],
    input: &[i16],
    b: &[i16],
    len: usize,
    d: usize,
) {
    debug_assert!(d >= 6 && (d & 1) == 0 && d <= 16);
    // The caller guarantees this; checked here too. The vector loads read `input[ix - 16..ix]`
    // with `ix < len`.
    assert!(input.len() >= len && out.len() >= len);

    let mut b_rev_pad = [0i16; 16];
    for k in 0..d {
        b_rev_pad[15 - k] = b[k];
    }

    for v in out[..d].iter_mut() {
        *v = 0;
    }

    let avx_start = 16usize.max(d);
    for ix in d..avx_start.min(len) {
        let mut s: i32 = 0;
        for k in 0..d {
            s = s.wrapping_add((input[ix - 1 - k] as i32).wrapping_mul(b[k] as i32));
        }
        let out32_q12 = ((input[ix] as i32) << 12).wrapping_sub(s);
        out[ix] = silk_sat16(silk_rshift_round(out32_q12, 12)) as i16;
    }

    if avx_start >= len {
        return;
    }

    let b_lo = _mm256_cvtepi16_epi32(_mm_loadu_si128(b_rev_pad.as_ptr() as *const __m128i));
    let b_hi = _mm256_cvtepi16_epi32(_mm_loadu_si128(b_rev_pad.as_ptr().add(8) as *const __m128i));

    for ix in avx_start..len {
        let base = input.as_ptr().add(ix - 16);
        let inp_lo = _mm256_cvtepi16_epi32(_mm_loadu_si128(base as *const __m128i));
        let inp_hi = _mm256_cvtepi16_epi32(_mm_loadu_si128(base.add(8) as *const __m128i));

        let prod_lo = _mm256_mullo_epi32(inp_lo, b_lo);
        let prod_hi = _mm256_mullo_epi32(inp_hi, b_hi);

        let sum256 = _mm256_add_epi32(prod_lo, prod_hi);
        let hi128 = _mm256_extracti128_si256(sum256, 1);
        let lo128 = _mm256_castsi256_si128(sum256);
        let sum128 = _mm_add_epi32(lo128, hi128);
        let sum64 = _mm_add_epi32(sum128, _mm_srli_si128(sum128, 8));
        let sum32 = _mm_add_epi32(sum64, _mm_srli_si128(sum64, 4));
        let s = _mm_cvtsi128_si32(sum32);

        let out32_q12 = ((input[ix] as i32) << 12).wrapping_sub(s);
        out[ix] = silk_sat16(silk_rshift_round(out32_q12, 12)) as i16;
    }
}

/// NEON form of [`silk_biquad_alt_stride2`], as libopus
/// `silk_biquad_alt_stride2_neon`: two pairs of samples per pass.
///
/// # Safety
///
/// `input_output.len()` must be even. Each pass reads and writes the 4
/// samples from `2 * k` on, with `2 * k + 4 <= input_output.len()`. The
/// loads of the coefficients and of the state read whole arrays.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn silk_biquad_alt_stride2_neon(
    input_output: &mut [i16],
    b_q28: &[i32; 3],
    a_q28: &[i32; 2],
    s: &mut [i32; 4],
) {
    use core::arch::aarch64::*;

    debug_assert!(input_output.len().is_multiple_of(2));
    let len = input_output.len() / 2;

    let offset_s32x2 = vdup_n_s32((1 << 14) - 1);
    let offset_s32x4 = vcombine_s32(offset_s32x2, offset_s32x2);

    let a_q28_s32x2 = vneg_s32(vld1_s32(a_q28.as_ptr()));
    let a_l_s32x2 = vreinterpret_s32_u32(vshr_n_u32(
        vreinterpret_u32_s32(vshl_n_s32(a_q28_s32x2, 18)),
        3,
    ));
    let mut a_u_s32x2 = vshr_n_s32(a_q28_s32x2, 14);
    a_u_s32x2 = vshr_n_s32(vshl_n_s32(a_u_s32x2, 16), 1);

    let b_q28_s32x2 = vld1_s32(b_q28.as_ptr());
    let t_s32x2 = vld1_s32(b_q28.as_ptr().add(1));

    let t0_s32x2x2 = vzip_s32(a_l_s32x2, a_l_s32x2);
    let t1_s32x2x2 = vzip_s32(a_u_s32x2, a_u_s32x2);
    let t2_s32x2x2 = vzip_s32(t_s32x2, t_s32x2);

    let a_l_s32x4 = vcombine_s32(t0_s32x2x2.0, t0_s32x2x2.1);
    let a_u_s32x4 = vcombine_s32(t1_s32x2x2.0, t1_s32x2x2.1);
    let b_q28_s32x4 = vcombine_s32(t2_s32x2x2.0, t2_s32x2x2.1);

    let mut s_s32x4 = vld1q_s32(s.as_ptr());
    let s_s32x2x2 = vtrn_s32(vget_low_s32(s_s32x4), vget_high_s32(s_s32x4));
    s_s32x4 = vcombine_s32(s_s32x2x2.0, s_s32x2x2.1);

    let mut k = 0;
    while k + 1 < len {
        let in_s16x4 = vld1_s16(input_output.as_ptr().add(2 * k));
        let in_s32x4 = vshll_n_s16(in_s16x4, 15);

        let t_s32x4 = vqdmulhq_lane_s32(in_s32x4, b_q28_s32x2, 0);

        let in_low = vcombine_s32(vget_low_s32(in_s32x4), vget_low_s32(in_s32x4));
        let t_low = vget_low_s32(t_s32x4);

        let mut out32_q14_s32x2 = vadd_s32(vget_low_s32(s_s32x4), t_low);
        out32_q14_s32x2 = vshl_n_s32(out32_q14_s32x2, 2);

        s_s32x4 = vcombine_s32(vget_high_s32(s_s32x4), vdup_n_s32(0));

        let out32_q14_s32x4 = vcombine_s32(out32_q14_s32x2, out32_q14_s32x2);

        let tmp_s32x4 = vqdmulhq_s32(out32_q14_s32x4, a_l_s32x4);
        s_s32x4 = vrsraq_n_s32(s_s32x4, tmp_s32x4, 14);

        let tmp_s32x4 = vqdmulhq_s32(out32_q14_s32x4, a_u_s32x4);
        s_s32x4 = vaddq_s32(s_s32x4, tmp_s32x4);

        let tmp_s32x4 = vqdmulhq_s32(in_low, b_q28_s32x4);
        s_s32x4 = vaddq_s32(s_s32x4, tmp_s32x4);

        let in_high = vcombine_s32(vget_high_s32(in_s32x4), vget_high_s32(in_s32x4));
        let t_high = vget_high_s32(t_s32x4);

        let mut out32_q14_1_s32x2 = vadd_s32(vget_low_s32(s_s32x4), t_high);
        out32_q14_1_s32x2 = vshl_n_s32(out32_q14_1_s32x2, 2);

        s_s32x4 = vcombine_s32(vget_high_s32(s_s32x4), vdup_n_s32(0));

        let out32_q14_1_s32x4 = vcombine_s32(out32_q14_1_s32x2, out32_q14_1_s32x2);

        let tmp_s32x4 = vqdmulhq_s32(out32_q14_1_s32x4, a_l_s32x4);
        s_s32x4 = vrsraq_n_s32(s_s32x4, tmp_s32x4, 14);

        let tmp_s32x4 = vqdmulhq_s32(out32_q14_1_s32x4, a_u_s32x4);
        s_s32x4 = vaddq_s32(s_s32x4, tmp_s32x4);

        let tmp_s32x4 = vqdmulhq_s32(in_high, b_q28_s32x4);
        s_s32x4 = vaddq_s32(s_s32x4, tmp_s32x4);

        let out32_q14_combined = vcombine_s32(out32_q14_s32x2, out32_q14_1_s32x2);
        let out_s32x4 = vaddq_s32(out32_q14_combined, offset_s32x4);
        let out_s16x4 = vqshrn_n_s32(out_s32x4, 14);

        vst1_s16(input_output.as_mut_ptr().add(2 * k), out_s16x4);

        k += 2;
    }

    if k < len {
        // One pair is left. The register holds the state as
        // `[s[0], s[2], s[1], s[3]]`.
        let in0 = input_output[2 * k] as i32;
        let in1 = input_output[2 * k + 1] as i32;

        let out32_q14_0 = silk_smlawb(vgetq_lane_s32(s_s32x4, 0), b_q28[0], in0) << 2;
        let out32_q14_1 = silk_smlawb(vgetq_lane_s32(s_s32x4, 1), b_q28[0], in1) << 2;

        let a0_l = (-a_q28[0]) & 0x00003FFF;
        let a0_u = -a_q28[0] >> 14;
        let a1_l = (-a_q28[1]) & 0x00003FFF;
        let a1_u = -a_q28[1] >> 14;

        let s0_new = vgetq_lane_s32(s_s32x4, 2)
            .wrapping_add(silk_rshift_round(silk_smulwb(out32_q14_0, a0_l), 14));
        let s2_new = vgetq_lane_s32(s_s32x4, 3)
            .wrapping_add(silk_rshift_round(silk_smulwb(out32_q14_1, a0_l), 14));

        let s0_new = silk_smlawb(s0_new, out32_q14_0, a0_u);
        let s2_new = silk_smlawb(s2_new, out32_q14_1, a0_u);
        let s0_new = silk_smlawb(s0_new, b_q28[1], in0);
        let s2_new = silk_smlawb(s2_new, b_q28[1], in1);

        let s1_new = silk_rshift_round(silk_smulwb(out32_q14_0, a1_l), 14);
        let s3_new = silk_rshift_round(silk_smulwb(out32_q14_1, a1_l), 14);
        let s1_new = silk_smlawb(s1_new, out32_q14_0, a1_u);
        let s3_new = silk_smlawb(s3_new, out32_q14_1, a1_u);
        let s1_new = silk_smlawb(s1_new, b_q28[2], in0);
        let s3_new = silk_smlawb(s3_new, b_q28[2], in1);

        input_output[2 * k] =
            silk_sat16(silk_rshift(out32_q14_0.wrapping_add((1 << 14) - 1), 14)) as i16;
        input_output[2 * k + 1] =
            silk_sat16(silk_rshift(out32_q14_1.wrapping_add((1 << 14) - 1), 14)) as i16;

        s[0] = s0_new;
        s[1] = s1_new;
        s[2] = s2_new;
        s[3] = s3_new;
    } else {
        vst1q_lane_s32(&mut s[0], s_s32x4, 0);
        vst1q_lane_s32(&mut s[1], s_s32x4, 2);
        vst1q_lane_s32(&mut s[2], s_s32x4, 1);
        vst1q_lane_s32(&mut s[3], s_s32x4, 3);
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;

    fn lpc_analysis_filter_ref(out: &mut [i16], input: &[i16], b: &[i16], len: usize, d: usize) {
        out[..d].fill(0);
        for ix in d..len {
            let mut s = 0i32;
            for k in 0..d {
                s = s.wrapping_add((input[ix - 1 - k] as i32).wrapping_mul(b[k] as i32));
            }
            let out32_q12 = ((input[ix] as i32) << 12).wrapping_sub(s);
            out[ix] = silk_sat16(silk_rshift_round(out32_q12, 12)) as i16;
        }
    }

    /// Small deterministic generator for the kernel tests.
    struct TestRng(u64);

    impl TestRng {
        fn next_u32(&mut self) -> u32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 16) as u32
        }

        fn below(&mut self, n: u32) -> u32 {
            self.next_u32() % n
        }

        fn i16(&mut self) -> i16 {
            self.next_u32() as i16
        }
    }

    /// Test signals of `len` samples. `kind` selects: 0 = full-range noise,
    /// 1 = noise at a random lower level, 2 = random choice of the two
    /// extreme values, 3 = all `i16::MIN`, 4 = all `i16::MAX`, 5 = extremes
    /// with alternating sign.
    fn test_signal(rng: &mut TestRng, len: usize, kind: u32) -> Vec<i16> {
        let shift = rng.below(12);
        (0..len)
            .map(|i| match kind {
                0 => rng.i16(),
                1 => rng.i16() >> shift,
                2 => {
                    if rng.below(2) == 0 {
                        i16::MAX
                    } else {
                        i16::MIN
                    }
                }
                3 => i16::MIN,
                4 => i16::MAX,
                _ => {
                    if i & 1 == 0 {
                        i16::MIN
                    } else {
                        i16::MAX
                    }
                }
            })
            .collect()
    }

    /// The loop of libopus `silk_warped_autocorrelation_FIX_c`, statement by
    /// statement: one sample at a time, two sections per pass.
    fn warped_autocorrelation_ref(
        corr: &mut [i32],
        scale: &mut i32,
        input: &[i16],
        warping_q16: i32,
        length: usize,
        order: usize,
    ) {
        const QC: i32 = 10;
        const QS: i32 = 13;
        let mut state_qs = [0i32; MAX_SHAPE_LPC_ORDER + 1];
        let mut corr_qc = [0i64; MAX_SHAPE_LPC_ORDER + 1];
        for &input_n in &input[..length] {
            let mut tmp1_qs = (input_n as i32) << QS;
            let mut i = 0;
            while i < order {
                let tmp2_qs = silk_smlaww(
                    state_qs[i],
                    state_qs[i + 1].wrapping_sub(tmp1_qs),
                    warping_q16,
                );
                state_qs[i] = tmp1_qs;
                corr_qc[i] += silk_rshift64(silk_smull(tmp1_qs, state_qs[0]), 2 * QS - QC);
                tmp1_qs = silk_smlaww(
                    state_qs[i + 1],
                    state_qs[i + 2].wrapping_sub(tmp2_qs),
                    warping_q16,
                );
                state_qs[i + 1] = tmp2_qs;
                corr_qc[i + 1] += silk_rshift64(silk_smull(tmp2_qs, state_qs[0]), 2 * QS - QC);
                i += 2;
            }
            state_qs[order] = tmp1_qs;
            corr_qc[order] += silk_rshift64(silk_smull(tmp1_qs, state_qs[0]), 2 * QS - QC);
        }
        let lsh = silk_limit_32(silk_clz64(corr_qc[0]) - 35, -12 - QC, 30 - QC);
        *scale = -(QC + lsh);
        for i in 0..=order {
            corr[i] = if lsh >= 0 {
                (corr_qc[i] << lsh) as i32
            } else {
                (corr_qc[i] >> (-lsh)) as i32
            };
        }
    }

    type WarpedFn = fn(&mut [i32], &mut i32, &[i16], i32, usize, usize);

    /// Calls `f` and the reference on the same input and compares
    /// `corr[0..=order]` and `scale`.
    fn check_warped(name: &str, f: WarpedFn, input: &[i16], warping_q16: i32, order: usize) {
        let length = input.len();
        let mut corr = [0i32; MAX_SHAPE_LPC_ORDER + 1];
        let mut corr_ref = [0i32; MAX_SHAPE_LPC_ORDER + 1];
        let (mut scale, mut scale_ref) = (0i32, 0i32);
        f(&mut corr, &mut scale, input, warping_q16, length, order);
        warped_autocorrelation_ref(
            &mut corr_ref,
            &mut scale_ref,
            input,
            warping_q16,
            length,
            order,
        );
        assert_eq!(
            (&corr[..=order], scale),
            (&corr_ref[..=order], scale_ref),
            "{name}: order={order} length={length} warping_q16={warping_q16}"
        );
    }

    /// Warping coefficients for the tests: the values the encoder uses
    /// (0.015 per kHz in Q16, scaled by the coding quality), zero, and the
    /// limits of the range a 16-bit multiplier covers.
    fn test_warping(rng: &mut TestRng, case: u32) -> i32 {
        match case % 8 {
            0 => 15729,
            1 => 7864,
            2 => 0,
            3 => 65535,
            4 => -65535,
            5 => rng.below(16385) as i32,
            6 => rng.below(65536) as i32,
            _ => -(rng.below(65536) as i32),
        }
    }

    fn check_warped_all(name: &str, f: WarpedFn) {
        let mut rng = TestRng(0x5eed_1234_9abc_def1);
        // Every even order with every length up to 40: all block
        // remainders and the lengths 0 and 1.
        for order in (0..=MAX_SHAPE_LPC_ORDER).step_by(2) {
            for length in 0..=40 {
                let kind = (order + length) as u32 % 6;
                let input = test_signal(&mut rng, length, kind);
                let warping_q16 = test_warping(&mut rng, (order / 2 + length) as u32);
                check_warped(name, f, &input, warping_q16, order);
            }
        }
        // Random cases up to the longest analysis window.
        for case in 0..3000u32 {
            let order = 2 * rng.below(MAX_SHAPE_LPC_ORDER as u32 / 2 + 1) as usize;
            let length = match case % 4 {
                0 => 240,
                1 => 120,
                _ => rng.below(241) as usize,
            };
            let input = test_signal(&mut rng, length, case % 6);
            let warping_q16 = test_warping(&mut rng, case / 6);
            check_warped(name, f, &input, warping_q16, order);
        }
    }

    #[test]
    fn warped_autocorrelation_scalar_matches_reference() {
        check_warped_all("scalar", silk_warped_autocorrelation_scalar);
    }

    #[test]
    fn warped_autocorrelation_matches_reference() {
        check_warped_all("dispatch", silk_warped_autocorrelation_fix);
    }

    /// An x86 kernel of the warped autocorrelation: `(corr, scale, input,
    /// warping_q16, order)`.
    #[cfg(target_arch = "x86_64")]
    type WarpedKernel = dyn Fn(&mut [i32], &mut i32, &[i16], i32, usize);

    /// One x86 kernel against the portable code, for every even order up to
    /// `max_order`, lengths 0 to 40, the window lengths of the encoder (120,
    /// 180 and 240 samples) and random lengths up to 480, with the warping
    /// factors of the encoder, the limits 0 and 2^15 - 1 and random ones in
    /// between.
    #[cfg(target_arch = "x86_64")]
    fn check_warped_x86(name: &str, max_order: usize, run: &WarpedKernel) {
        let check = |input: &[i16], warping_q16: i32, order: usize| {
            let mut corr = [0i32; MAX_SHAPE_LPC_ORDER + 1];
            let mut corr_ref = [0i32; MAX_SHAPE_LPC_ORDER + 1];
            let (mut scale, mut scale_ref) = (0i32, 0i32);
            run(&mut corr, &mut scale, input, warping_q16, order);
            silk_warped_autocorrelation_scalar(
                &mut corr_ref,
                &mut scale_ref,
                input,
                warping_q16,
                input.len(),
                order,
            );
            assert_eq!(
                (&corr[..=order], scale),
                (&corr_ref[..=order], scale_ref),
                "{name}: order={order} length={} warping_q16={warping_q16}",
                input.len()
            );
        };
        let warping = |rng: &mut TestRng, case: usize| match case % 6 {
            0 => 15728,
            1 => 16383,
            2 => 0,
            3 => i16::MAX as i32,
            4 => 1,
            _ => rng.below(1 << 15) as i32,
        };
        let mut rng = TestRng(0x3c6e_f372_fe94_f82b);
        for order in (0..=max_order).step_by(2) {
            for length in (0..=40).chain([120, 180, 240]) {
                let input = test_signal(&mut rng, length, (order + length) as u32 % 6);
                check(&input, warping(&mut rng, order / 2 + length), order);
            }
        }
        for case in 0..2000 {
            let order = 2 * rng.below(max_order as u32 / 2 + 1) as usize;
            let length = rng.below(481) as usize;
            let input = test_signal(&mut rng, length, case as u32 % 6);
            check(&input, warping(&mut rng, case / 6), order);
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn warped_autocorrelation_sse41_matches_scalar() {
        if !crate::compat::x86_has_sse41() {
            return;
        }
        // Every register count, also with fewer sections than it holds.
        check_warped_x86("sse4.1 K=4", 16, &|corr, scale, input, warping_q16, order| {
            // SAFETY: the processor has SSE4.1 (checked above).
            unsafe { warped_autocorr_sse41::<4>(corr, scale, input, warping_q16, order) }
        });
        check_warped_x86("sse4.1 K=5", 20, &|corr, scale, input, warping_q16, order| {
            // SAFETY: the processor has SSE4.1 (checked above).
            unsafe { warped_autocorr_sse41::<5>(corr, scale, input, warping_q16, order) }
        });
        check_warped_x86("sse4.1 K=6", 24, &|corr, scale, input, warping_q16, order| {
            // SAFETY: the processor has SSE4.1 (checked above).
            unsafe { warped_autocorr_sse41::<6>(corr, scale, input, warping_q16, order) }
        });
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn warped_autocorrelation_avx2_matches_scalar() {
        if !crate::compat::x86_has_avx2() {
            return;
        }
        check_warped_x86("avx2 K=2", 16, &|corr, scale, input, warping_q16, order| {
            // SAFETY: the processor has AVX2 (checked above).
            unsafe { warped_autocorr_avx2::<2>(corr, scale, input, warping_q16, order) }
        });
        check_warped_x86("avx2 K=3", 24, &|corr, scale, input, warping_q16, order| {
            // SAFETY: the processor has AVX2 (checked above).
            unsafe { warped_autocorr_avx2::<3>(corr, scale, input, warping_q16, order) }
        });
    }

    /// The loop of libopus `silk_apply_sine_window`, statement by statement.
    fn apply_sine_window_ref(px_win: &mut [i16], px: &[i16], win_type: i32, length: usize) {
        let f_q16 = FREQ_TABLE_Q16[(length >> 2) - 4] as i32;
        let c_q16 = silk_smulwb(f_q16, -f_q16);
        let (mut s0_q16, mut s1_q16) = if win_type == 1 {
            (0, f_q16 + (length as i32 >> 3))
        } else {
            (1 << 16, (1 << 16) + (c_q16 >> 1) + (length as i32 >> 4))
        };
        for k in (0..length).step_by(4) {
            px_win[k] = silk_smulwb((s0_q16 + s1_q16) >> 1, px[k] as i32) as i16;
            px_win[k + 1] = silk_smulwb(s1_q16, px[k + 1] as i32) as i16;
            s0_q16 = silk_smulwb(s1_q16, c_q16) + (s1_q16 << 1) - s0_q16 + 1;
            s0_q16 = s0_q16.min(1 << 16);
            px_win[k + 2] = silk_smulwb((s0_q16 + s1_q16) >> 1, px[k + 2] as i32) as i16;
            px_win[k + 3] = silk_smulwb(s0_q16, px[k + 3] as i32) as i16;
            s1_q16 = silk_smulwb(s0_q16, c_q16) + (s0_q16 << 1) - s1_q16;
            s1_q16 = s1_q16.min(1 << 16);
        }
    }

    #[test]
    fn apply_sine_window_matches_reference() {
        let mut rng = TestRng(0x0f1e_2d3c_4b5a_6978);
        // Every length the frequency table covers: the 7 with a table and
        // the 20 that run the recurrence.
        for length in (16..=120usize).step_by(4) {
            for win_type in [1, 2] {
                for kind in 0..6 {
                    let px = test_signal(&mut rng, length, kind);
                    let mut win = vec![0x1234i16; length];
                    let mut win_ref = vec![0x1234i16; length];
                    silk_apply_sine_window(&mut win, &px, win_type, length);
                    apply_sine_window_ref(&mut win_ref, &px, win_type, length);
                    assert_eq!(
                        win, win_ref,
                        "length={length} win_type={win_type} kind={kind}"
                    );
                }
            }
        }
    }

    /// `silk_autocorr` with one plain loop per lag: the same scaling
    /// decisions, and every sum over all `n - k` products in order.
    fn autocorr_ref(results: &mut [i32], scale: &mut i32, input: &[i16], n: usize, count: usize) {
        let ilog = |x: u32| 32 - x.leading_zeros() as i32;
        let ac0_shift = ilog((n + (n >> 4)) as u32) - 1;
        let mut ac0: i32 = 1 + ((n as i32) << 7);
        for &sample in &input[..n] {
            ac0 += (sample as i32 * sample as i32) >> ac0_shift;
        }
        ac0 += ac0 >> 7;
        let mut shift = (ilog(ac0 as u32) - 1 - 30 + ac0_shift + 1) / 2;
        let x: Vec<i16> = if shift > 0 {
            input[..n]
                .iter()
                .map(|&sample| silk_rshift_round(sample as i32, shift) as i16)
                .collect()
        } else {
            shift = 0;
            input[..n].to_vec()
        };
        for k in 0..count {
            let mut sum = 0i32;
            for j in 0..n - k {
                sum = sum.wrapping_add((x[j] as i32).wrapping_mul(x[j + k] as i32));
            }
            results[k] = sum;
        }
        shift *= 2;
        if shift <= 0 {
            results[0] += 1i32 << (-shift).min(30);
        }
        if results[0] > 0 && results[0] < 1 << 28 {
            let shift2 = 29 - ilog(results[0] as u32);
            results[..count].iter_mut().for_each(|v| *v <<= shift2);
            shift -= shift2;
        } else if results[0] >= 1 << 29 {
            let shift2 = if results[0] >= 1 << 30 { 2 } else { 1 };
            results[..count].iter_mut().for_each(|v| *v >>= shift2);
            shift += shift2;
        }
        *scale = shift;
    }

    #[test]
    fn autocorr_matches_reference() {
        let mut rng = TestRng(0x7777_1234_8888_4321);
        let sizes = [
            0usize, 1, 2, 3, 4, 7, 8, 16, 17, 31, 100, 120, 192, 240, 384, 640,
        ];
        for &n in &sizes {
            // Up to 25 lags (shaping order 24), and never more than `n + 1`.
            for count in 1..=25.min(n + 1) {
                // Kinds 0 and 2 to 5 are loud, so the input is scaled down;
                // kind 1 is often quiet enough to be used as it is.
                for kind in 0..7 {
                    let mut input = if kind == 6 {
                        vec![0i16; n]
                    } else {
                        test_signal(&mut rng, n, kind)
                    };
                    // Samples after `n` must not be read.
                    input.extend_from_slice(&[i16::MAX; 5]);
                    let mut results = [0x0bad_beefi32; 26];
                    let mut results_ref = [0x0bad_beefi32; 26];
                    let (mut scale, mut scale_ref) = (0, 0);
                    silk_autocorr(&mut results, &mut scale, &input, n, count);
                    autocorr_ref(&mut results_ref, &mut scale_ref, &input, n, count);
                    assert_eq!(
                        (results, scale),
                        (results_ref, scale_ref),
                        "n={n} count={count} kind={kind}"
                    );
                }
            }
        }
    }

    /// The AVX2 copies of `autocorr_sums`, `silk_corr_vector_fix` and
    /// `silk_corr_matrix_fix` against the portable bodies, whatever the
    /// dispatch chooses: lengths 0 to 40, the subframe lengths of the
    /// encoder and random ones, orders 1 to 16, every signal kind and right
    /// shifts of the correlation vector from 0 to 6.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_copies_match_portable() {
        if !crate::compat::x86_has_avx2() {
            return;
        }
        let mut rng = TestRng(0x6a09_e667_f3bc_c908);
        let lengths: Vec<usize> = (0..=40)
            .chain([80, 120, 160, 240, 640])
            .chain((0..60).map(|_| rng.below(700) as usize))
            .collect();
        for (case, &n) in lengths.iter().enumerate() {
            for kind in 0..6 {
                let x = test_signal(&mut rng, n + 16, kind);
                let t = test_signal(&mut rng, n, (kind + 1) % 6);

                let count = (rng.below(25) as usize + 1).min(n + 1);
                let mut want = [0x0bad_beefi32; 25];
                let mut got = [0x0bad_beefi32; 25];
                autocorr_sums_portable(&mut want[..count], &x[..n]);
                // SAFETY: the processor has AVX2 (checked above).
                unsafe { autocorr_sums_avx2(&mut got[..count], &x[..n]) };
                assert_eq!(got, want, "autocorr_sums n={n} count={count} kind={kind}");

                if n == 0 {
                    continue;
                }
                let order = 1 + (case + kind as usize) % 16;
                let rshifts = rng.below(7) as i32;
                let mut want = [0x0bad_beefi32; 16];
                let mut got = [0x0bad_beefi32; 16];
                corr_vector_portable(&x, &t, n, order, &mut want, rshifts);
                // SAFETY: the processor has AVX2 (checked above).
                unsafe { corr_vector_avx2(&x, &t, n, order, &mut got, rshifts) };
                assert_eq!(got, want, "corr_vector n={n} order={order} rshifts={rshifts}");

                let mut want = ([0x0bad_beefi32; 256], 0i32, 0i32);
                let mut got = ([0x0bad_beefi32; 256], 0i32, 0i32);
                corr_matrix_portable(&x, n, order, &mut want.0, &mut want.1, &mut want.2);
                // SAFETY: the processor has AVX2 (checked above).
                unsafe { corr_matrix_avx2(&x, n, order, &mut got.0, &mut got.1, &mut got.2) };
                assert_eq!(got, want, "corr_matrix n={n} order={order} kind={kind}");
            }
        }
    }

    /// libopus `silk_schur64`, statement by statement, with the two columns
    /// in one array.
    fn schur64_ref(rc_q16: &mut [i32], c: &[i32], order: usize) -> i32 {
        let mut c_matrix = [[0i32; 2]; MAX_SHAPE_LPC_ORDER + 1];
        if c[0] <= 0 {
            rc_q16[..order].fill(0);
            return 0;
        }
        for k in 0..=order {
            c_matrix[k] = [c[k], c[k]];
        }
        let mut k = 0;
        while k < order {
            if c_matrix[k + 1][0].wrapping_abs() >= c_matrix[0][1] {
                rc_q16[k] = if c_matrix[k + 1][0] > 0 {
                    -64880
                } else {
                    64880
                };
                k += 1;
                break;
            }
            let rc_tmp_q31 = silk_div32_varq(c_matrix[k + 1][0].wrapping_neg(), c_matrix[0][1], 31);
            rc_q16[k] = silk_rshift_round(rc_tmp_q31, 15);
            for n in 0..(order - k) {
                let c_tmp1_q30 = c_matrix[n + k + 1][0];
                let c_tmp2_q30 = c_matrix[n][1];
                c_matrix[n + k + 1][0] =
                    c_tmp1_q30.wrapping_add(silk_smmul(c_tmp2_q30 << 1, rc_tmp_q31));
                c_matrix[n][1] = c_tmp2_q30.wrapping_add(silk_smmul(c_tmp1_q30 << 1, rc_tmp_q31));
            }
            k += 1;
        }
        rc_q16[k..order].fill(0);
        c_matrix[0][1].max(1)
    }

    #[test]
    fn schur64_matches_reference() {
        let mut rng = TestRng(0x9e37_79b9_7f4a_7c15);
        let check = |c: &[i32], order: usize, what: &str| {
            let mut rc = [0x1111_1111i32; MAX_SHAPE_LPC_ORDER];
            let mut rc_ref = [0x1111_1111i32; MAX_SHAPE_LPC_ORDER];
            let nrg = silk_schur64(&mut rc, c, order);
            let nrg_ref = schur64_ref(&mut rc_ref, c, order);
            assert_eq!(
                (nrg, rc),
                (nrg_ref, rc_ref),
                "{what}: order={order} c={c:?}"
            );
        };
        for order in 0..=MAX_SHAPE_LPC_ORDER {
            for case in 0..60u32 {
                // Autocorrelations of real signals, as the encoder passes
                // them: the recursion runs through all steps.
                let mut signal = test_signal(&mut rng, 240, case % 2);
                if case % 3 == 0 {
                    // Low-pass: reflection coefficients close to -1.
                    for i in 1..signal.len() {
                        signal[i] = ((signal[i] as i32 + 7 * signal[i - 1] as i32) / 8) as i16;
                    }
                }
                let mut corr = [0i32; MAX_SHAPE_LPC_ORDER + 1];
                let mut scale = 0;
                silk_autocorr(&mut corr, &mut scale, &signal, 240, order + 1);
                if case % 4 != 0 {
                    // The white-noise term of the noise shaping analysis.
                    corr[0] = corr[0].wrapping_add(silk_smulwb(corr[0] >> 4, 31).max(1));
                }
                check(&corr, order, "autocorrelation");

                // Arbitrary values: the recursion stops early when a
                // correlation is not smaller than the energy.
                let mut c = [0i32; MAX_SHAPE_LPC_ORDER + 1];
                for v in c.iter_mut() {
                    *v = (rng.next_u32() as i32) >> rng.below(8);
                }
                c[0] = match case % 5 {
                    0 => i32::MAX,
                    1 => (rng.next_u32() >> 1) as i32,
                    2 => 1,
                    3 => 0,
                    _ => -((rng.next_u32() >> 1) as i32),
                };
                check(&c, order, "arbitrary");
                if case % 7 == 0 {
                    c[1 + rng.below(MAX_SHAPE_LPC_ORDER as u32) as usize] = i32::MIN;
                    check(&c, order, "with i32::MIN");
                }
            }
        }
    }

    /// libopus `silk_sum_sqr_shift` written out: the shift, and the energy
    /// with the sum of each pair of squares shifted.
    fn sum_sqr_shift_ref(x: &[i16]) -> (i32, i32) {
        let len = x.len();
        let pass = |init: i32, shft: i32| {
            let mut nrg = init;
            let mut i = 0;
            while i < len {
                let a = (x[i] as i32 * x[i] as i32) as u32;
                if i + 1 < len {
                    let b = (x[i + 1] as i32 * x[i + 1] as i32) as u32;
                    nrg = nrg.wrapping_add((a.wrapping_add(b) >> shft) as i32);
                } else {
                    nrg = nrg.wrapping_add((a >> shft) as i32);
                }
                i += 2;
            }
            nrg
        };
        let shft = 31 - (len as i32).leading_zeros() as i32;
        let first = if len == 0 { 0 } else { pass(len as i32, shft) };
        let shft = (shft + 3 - first.leading_zeros() as i32).max(0);
        (shft, pass(0, shft))
    }

    #[test]
    fn sum_sqr_shift_matches_reference() {
        let mut rng = TestRng(0x4d59_5df4_d0f3_3173);
        // Every length up to 70 (all remainders of a vector loop), then
        // lengths of real calls.
        for len in (0..=70usize).chain([100, 160, 320, 656]) {
            for kind in 0..7 {
                let mut x = if kind == 6 {
                    vec![0i16; len]
                } else {
                    test_signal(&mut rng, len, kind)
                };
                let (shft, nrg) = sum_sqr_shift_ref(&x);
                // Samples after `len` must not be read.
                x.extend_from_slice(&[i16::MIN; 3]);

                let (mut energy, mut shift) = (-1, -1);
                silk_sum_sqr_shift(&mut energy, &mut shift, &x, len);
                assert_eq!((energy, shift), (nrg, shft), "len={len} kind={kind}");
            }
        }
    }

    /// `x[0] * y[lag] + x[1] * y[lag + 1] + ...` over `len` products, or
    /// fewer where `y` ends earlier. The sum wraps on overflow.
    fn xcorr_one_lag_ref(x: &[i16], y: &[i16], len: usize, lag: usize) -> i32 {
        let mut sum = 0i32;
        for j in 0..len {
            if lag + j < y.len() {
                sum = sum.wrapping_add((x[j] as i32).wrapping_mul(y[lag + j] as i32));
            }
        }
        sum
    }

    type XcorrKernelFn = fn(&[i16], &[i16], &mut [i32; 4], usize);

    fn check_xcorr_kernel(name: &str, kernel: XcorrKernelFn) {
        let mut rng = TestRng(0x0bad_cafe_1357_9bdf);
        // Lengths 0 to 70 cover every count of whole 16-sample blocks up to
        // 4, the extra 8-sample block and every remainder from 0 to 7.
        for len in 0..=70usize {
            for kind in 0..6 {
                // Slices of exactly the length the kernel may read.
                let x = test_signal(&mut rng, len, kind);
                let y = test_signal(&mut rng, len + 3, (kind + len as u32) % 6);
                let start = [rng.next_u32() as i32, 0, i32::MAX, i32::MIN];
                let mut sum = start;
                kernel(&x, &y, &mut sum, len);
                for k in 0..4 {
                    let expected = start[k].wrapping_add(xcorr_one_lag_ref(&x, &y, len, k));
                    assert_eq!(sum[k], expected, "{name}: len={len} kind={kind} lag={k}");
                }
            }
        }
    }

    #[test]
    fn xcorr_kernel_scalar_matches_reference() {
        check_xcorr_kernel("scalar", |x, y, sum, len| {
            xcorr_kernel_scalar(&x[..len], &y[..len + 3], sum)
        });
    }

    #[test]
    fn xcorr_kernel_matches_reference() {
        check_xcorr_kernel("dispatch", xcorr_kernel);
    }

    /// An `x` shorter than `len` must stop with a panic, in a group of 4
    /// lags and in the lags after the groups, and never be read past its end.
    #[test]
    #[should_panic]
    fn pitch_xcorr_short_x_panics_in_single_lag() {
        let (x, y) = ([1i16; 15], [1i16; 16]);
        let mut xcorr = [0i32; 1];
        silk_pitch_xcorr(&x, &y, &mut xcorr, 16, 1);
    }

    #[test]
    #[should_panic]
    fn pitch_xcorr_short_x_panics_in_group() {
        let (x, y) = ([1i16; 15], [1i16; 19]);
        let mut xcorr = [0i32; 4];
        silk_pitch_xcorr(&x, &y, &mut xcorr, 16, 4);
    }

    #[test]
    fn pitch_xcorr_matches_reference() {
        let mut rng = TestRng(0x2468_ace0_1122_3344);
        let lens = [0usize, 1, 2, 7, 8, 9, 15, 16, 17, 31, 40, 41, 80, 147, 320];
        for &len in &lens {
            for max_pitch in 1..=21usize {
                for kind in 0..6 {
                    let x = test_signal(&mut rng, len, kind);
                    // `y` with exactly the elements the lags need, and two
                    // cases where it ends early.
                    let full = len + max_pitch - 1;
                    for y_len in [full, full.saturating_sub(1), full / 2] {
                        let y = test_signal(&mut rng, y_len, (kind + 1) % 6);
                        let mut xcorr = [0x5a5a_5a5ai32; 21];
                        silk_pitch_xcorr(&x, &y, &mut xcorr[..max_pitch], len, max_pitch);
                        for lag in 0..max_pitch {
                            assert_eq!(
                                xcorr[lag],
                                xcorr_one_lag_ref(&x, &y, len, lag),
                                "len={len} max_pitch={max_pitch} y_len={y_len} lag={lag}"
                            );
                        }
                    }
                }
            }
        }
    }

    /// `silk_inner_prod_aligned` by its definition.
    fn inner_prod_ref(a: &[i16], b: &[i16], len: usize) -> i32 {
        let mut sum = 0i32;
        for i in 0..len {
            sum = sum.wrapping_add((a[i] as i32).wrapping_mul(b[i] as i32));
        }
        sum
    }

    #[test]
    fn inner_prod_aligned_matches_reference() {
        let mut rng = TestRng(0x1bad_f00d_5eed_0001);
        // Lengths 0 to 100 cover every count of 32-sample and 8-sample
        // blocks of the vector loops and every remainder.
        for len in (0..=100usize).chain([160, 320, 641]) {
            for kind in 0..6 {
                let mut a = test_signal(&mut rng, len, kind);
                let mut b = test_signal(&mut rng, len, (kind + len as u32) % 6);
                let expected = inner_prod_ref(&a, &b, len);
                assert_eq!(
                    silk_inner_prod_aligned_scalar(&a, &b, len),
                    expected,
                    "scalar: len={len} kind={kind}"
                );
                // Samples after `len` must not be read.
                a.extend_from_slice(&[i16::MAX; 9]);
                b.extend_from_slice(&[i16::MIN; 9]);
                assert_eq!(
                    silk_inner_prod_aligned(&a, &b, len),
                    expected,
                    "dispatch: len={len} kind={kind}"
                );
            }
        }
    }

    /// A slice shorter than `len` must stop with a panic and never be read
    /// past its end.
    #[test]
    #[should_panic]
    fn inner_prod_aligned_short_first_slice_panics() {
        silk_inner_prod_aligned(&[1i16; 39], &[1i16; 40], 40);
    }

    #[test]
    #[should_panic]
    fn inner_prod_aligned_short_second_slice_panics() {
        silk_inner_prod_aligned(&[1i16; 40], &[1i16; 39], 40);
    }

    #[test]
    fn biquad_alt_stride2_matches_scalar() {
        let mut rng = TestRng(0x0123_4567_89ab_cdef);
        for case in 0..600u32 {
            // Pair counts 0 to 9 cover the two-pair vector loop with and
            // without the single pair that is left, then longer blocks.
            let len = match case % 12 {
                10 => 160,
                11 => 481,
                n => n as usize,
            };
            // High-pass coefficients as the encoder computes them: `r` close
            // to 1 in Q28, `b = [r, -2r, r]`, `a = [-2r(1 - fc^2 / 2), r^2]`.
            let fc_q19 = 1 + rng.below(80_000) as i64;
            let r_q28 = (1i64 << 28) - 471 * fc_q19 / 2;
            let r_q22 = r_q28 >> 6;
            let fc2_q28 = (fc_q19 * fc_q19) >> 10;
            let b = [r_q28 as i32, (-2 * r_q28) as i32, r_q28 as i32];
            let a = [
                ((r_q22 * (fc2_q28 - (2i64 << 28))) >> 22) as i32,
                ((r_q22 * r_q22) >> 16) as i32,
            ];
            let kind = case % 6;
            let input = test_signal(&mut rng, 2 * len, kind);
            // A state as a running filter has it, or zero.
            let start: [i32; 4] = if case % 5 == 0 {
                [0; 4]
            } else {
                core::array::from_fn(|_| (rng.next_u32() as i32) >> (6 + rng.below(8)))
            };

            let mut expected = input.clone();
            let mut state_expected = start;
            silk_biquad_alt_stride2_scalar(&mut expected, &b, &a, &mut state_expected);

            // Longer slices than the call names: nothing after them may
            // change.
            let mut got = input.clone();
            got.extend_from_slice(&[0x55aai16; 5]);
            let mut b_long = b.to_vec();
            b_long.push(i32::MIN);
            let mut a_long = a.to_vec();
            a_long.push(i32::MIN);
            let mut state = [0x0bad_beefi32; 6];
            state[..4].copy_from_slice(&start);
            silk_biquad_alt_stride2(&mut got, &b_long, &a_long, &mut state, len);

            assert_eq!(&got[..2 * len], &expected[..], "len={len} kind={kind}");
            assert_eq!(&got[2 * len..], &[0x55aai16; 5], "len={len} kind={kind}");
            assert_eq!(&state[..4], &state_expected, "len={len} kind={kind}");
            assert_eq!(&state[4..], &[0x0bad_beefi32; 2], "len={len} kind={kind}");
        }
    }

    #[test]
    #[should_panic]
    fn biquad_alt_stride2_short_input_panics() {
        let mut samples = [1i16; 7];
        let mut state = [0i32; 4];
        silk_biquad_alt_stride2(&mut samples, &[1 << 28, -2 << 28, 1 << 28], &[0, 0], &mut state, 4);
    }

    #[test]
    fn test_silk_lpc_analysis_filter_matches_reference() {
        let mut rng = TestRng(0x1357_2468_aaaa_5555);
        // Every even order: 6 to 16 have their own copy of the loop, the
        // orders above share the general one.
        for d in (6usize..=24).step_by(2) {
            // No output at all, then every count of outputs up to 40 (all
            // remainders of a vector loop), then the lengths of real calls.
            let lens = (d..=d + 40).chain([160, 320, 672]);
            for (case, len) in lens.enumerate() {
                let kind = case as u32 % 6;
                let input = test_signal(&mut rng, len, kind);
                // Full-range coefficients drive the output into saturation;
                // the smaller ones keep most of it inside the 16-bit range.
                let b = test_signal(&mut rng, d, if case % 2 == 0 { 0 } else { 1 });

                // Longer slices than `len` and `d`: only the first `len`
                // outputs may be written.
                let mut out_opt = vec![0x55aai16; len + 3];
                let mut out_ref = vec![0x55aai16; len + 3];
                let mut input_long = input.clone();
                input_long.extend_from_slice(&[i16::MAX; 3]);
                let mut b_long = b.clone();
                b_long.push(i16::MIN);

                silk_lpc_analysis_filter(&mut out_opt, &input_long, &b_long, len, d, 0);
                lpc_analysis_filter_ref(&mut out_ref, &input, &b, len, d);

                assert_eq!(
                    out_opt, out_ref,
                    "mismatch for d={d}, len={len}, kind={kind}"
                );
            }
        }
    }
}
