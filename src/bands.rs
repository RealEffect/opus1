use crate::mathops::{celt_exp2, mac};
use crate::modes::CeltMode;
use crate::pitch::celt_inner_prod;
use crate::pvq::*;
use crate::range_coder::RangeCoder;
use crate::rate::{BITRES, bits2pulses, get_pulses, pulses2bits};
use crate::range_coder::tell_frac_inline;
use core::mem::MaybeUninit;

#[cfg(not(feature = "std"))]
use crate::compat::Math;

const MIN_STEREO_ENERGY: f32 = 1e-10;

pub struct BandCtx<'a> {
    pub encode: bool,
    pub m: &'a CeltMode,
    pub i: usize,
    pub band_e: &'a [f32],
    pub rc: &'a mut RangeCoder,
    pub spread: i32,
    pub remaining_bits: i32,
    pub resynth: bool,
    pub tf_change: i32,
    pub intensity: usize,
    pub theta_round: i32,
    pub avoid_split_noise: bool,
    pub disable_inv: bool,
    pub seed: u32,
    /// The encoder's angle of the stereo band in Q30, once computed. Both
    /// passes of `quant_band_stereo_rdo` start from the same input, so the
    /// second pass takes the angle of the first.
    pub stereo_angle: Option<i32>,
    /// The encoder's cache of rotation angles (see `RotationCache`). `None`
    /// computes each angle.
    pub rotations: Option<&'a mut RotationCache>,
}

#[inline]
fn bitexact_cos(x: i16) -> i16 {
    #[inline(always)]
    fn frac_mul16(a: i16, b: i16) -> i16 {
        ((16384i32 + (a as i32) * (b as i32)) >> 15) as i16
    }

    let tmp = (4096i32 + (x as i32) * (x as i32)) >> 13;
    let x2 = tmp as i16;
    let x2 = (32767 - x2 as i32
        + frac_mul16(x2, -7651 + frac_mul16(x2, 8277 + frac_mul16(-626, x2))) as i32)
        as i16;
    1 + x2
}

#[inline]
pub fn bitexact_log2tan(isin: i32, icos: i32) -> i32 {
    let ec_ilog = |x: u32| -> i32 {
        if x == 0 {
            0
        } else {
            32 - x.leading_zeros() as i32
        }
    };
    let lc = ec_ilog(icos.max(0) as u32);
    let ls = ec_ilog(isin.max(0) as u32);
    let icos_shifted = if lc > 0 {
        icos.max(0) << (15 - lc).max(0)
    } else {
        0
    };
    let isin_shifted = if ls > 0 {
        isin.max(0) << (15 - ls).max(0)
    } else {
        0
    };
    let fract_mul = |a: i32, b: i32| -> i32 { (a * b + 16384) >> 15 };
    (ls - lc) * (1 << 11) + fract_mul(isin_shifted, fract_mul(isin_shifted, -2597) + 7932)
        - fract_mul(icos_shifted, fract_mul(icos_shifted, -2597) + 7932)
}

#[inline(always)]
fn celt_sudiv(n: i32, d: i32) -> i32 {
    n / d
}

/// libopus `isqrt32` (celt/mathops.c): the integer part of the square root
/// of `val`.
///
/// libopus finds it one bit at a time, in a loop of up to 16 steps whose
/// branch depends on the data. The square root in double precision gives
/// the same value with one instruction: it is correctly rounded, and for
/// `val` below 2^32 that is not a square, the root lies at least 2^-17
/// below the next integer, far more than its rounding error of at most
/// 2^-37. So cutting off the fraction gives the integer part. A test
/// compares it with the loop of libopus (`isqrt32_bits`).
#[inline]
fn isqrt32(val: u32) -> u32 {
    (val as f64).sqrt() as u32
}

/// The loop of libopus `isqrt32`, for `val` of at least 1. The reference
/// for `isqrt32` in the tests.
#[cfg(test)]
fn isqrt32_bits(mut val: u32) -> u32 {
    let mut g = 0u32;
    let mut bshift = ((32 - val.leading_zeros()) as i32 - 1) >> 1;
    let mut b = 1u32 << bshift;
    while bshift >= 0 {
        let t = (((g << 1) + b) as u64) << bshift;
        if t <= val as u64 {
            g += b;
            val -= t as u32;
        }
        b >>= 1;
        bshift -= 1;
    }
    g
}

pub const SPREAD_NONE: i32 = 0;
pub const SPREAD_LIGHT: i32 = 1;
pub const SPREAD_NORMAL: i32 = 2;
pub const SPREAD_AGGRESSIVE: i32 = 3;

#[allow(clippy::too_many_arguments)]
pub fn spreading_decision(
    m: &CeltMode,
    x_buf: &[f32],
    average: &mut i32,
    last_decision: i32,
    hf_average: &mut i32,
    tapset_decision: &mut i32,
    update_hf: bool,
    end: usize,
    channels: usize,
    m_val: usize,
    spread_weight: &[i32],
) -> i32 {
    let mut sum = 0;
    let mut nb_bands = 0;
    let n0 = m_val * m.short_mdct_size;
    let mut hf_sum = 0;

    if m_val * (m.e_bands[end] as usize - m.e_bands[end - 1] as usize) <= 8 {
        return SPREAD_NONE;
    }

    for c in 0..channels {
        for (i, &sw) in spread_weight[..end].iter().enumerate() {
            let n = m_val * (m.e_bands[i + 1] as usize - m.e_bands[i] as usize);
            if n <= 8 {
                continue;
            }

            let mut tcount = [0; 3];
            let offset = m_val * m.e_bands[i] as usize + c * n0;
            let x = &x_buf[offset..offset + n];

            for xv in x.iter().copied() {
                let x2n = xv * xv * (n as f32);
                if x2n < 0.25 {
                    tcount[0] += 1;
                }
                if x2n < 0.0625 {
                    tcount[1] += 1;
                }
                if x2n < 0.015625 {
                    tcount[2] += 1;
                }
            }

            if i > m.nb_ebands - 4 {
                hf_sum += 32 * (tcount[1] + tcount[0]) / (n as i32);
            }

            let tmp = (if 2 * tcount[2] >= (n as i32) { 1 } else { 0 })
                + (if 2 * tcount[1] >= (n as i32) { 1 } else { 0 })
                + (if 2 * tcount[0] >= (n as i32) { 1 } else { 0 });
            sum += tmp * sw;
            nb_bands += sw;
        }
    }

    if update_hf {
        if hf_sum > 0 {
            hf_sum /= (channels as i32) * (4 - m.nb_ebands as i32 + end as i32);
        }
        *hf_average = (*hf_average + hf_sum) >> 1;
        hf_sum = *hf_average;

        if *tapset_decision == 2 {
            hf_sum += 4;
        } else if *tapset_decision == 0 {
            hf_sum -= 4;
        }

        if hf_sum > 22 {
            *tapset_decision = 2;
        } else if hf_sum > 18 {
            *tapset_decision = 1;
        } else {
            *tapset_decision = 0;
        }
    }

    if nb_bands == 0 {
        return SPREAD_NORMAL;
    }

    let mut sum_scaled = (sum << 8) / nb_bands;
    sum_scaled = (sum_scaled + *average) >> 1;
    *average = sum_scaled;

    let sum_final = (3 * sum_scaled + (((3 - last_decision) << 7) + 64) + 2) >> 2;

    if sum_final < 80 {
        SPREAD_AGGRESSIVE
    } else if sum_final < 256 {
        SPREAD_NORMAL
    } else if sum_final < 384 {
        SPREAD_LIGHT
    } else {
        SPREAD_NONE
    }
}

/// Scale factor of one Haar step, `1 / sqrt(2)`.
const HAAR_SCALE: f32 = core::f32::consts::FRAC_1_SQRT_2;

/// One step of the Haar transform, in place.
///
/// `x` is read as `n0` rows of `stride` values. Rows `2j` and `2j + 1` are
/// replaced by their sum and their difference, both scaled by `1 / sqrt(2)`.
/// A last row without a partner (odd `n0`) is left alone.
pub fn haar1(x: &mut [f32], n0: usize, stride: usize) {
    // The only bounds check. Every kernel below works on this slice, whose
    // length is a multiple of `2 * stride`.
    let x = &mut x[..2 * (n0 >> 1) * stride];
    // SAFETY: NEON is available: src/lib.rs refuses to build for aarch64
    // without it.
    #[cfg(target_arch = "aarch64")]
    unsafe {
        haar1_neon(x, stride);
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        // The kernel needs a 64-bit lane permute, which only AVX2 has.
        // Processors without AVX2 use the portable code.
        if stride == 1 && n0 >= 16 && crate::compat::x86_has_avx2() {
            // SAFETY: the CPU has AVX2, the feature of the kernel.
            unsafe { haar1_avx2(x, n0) };
            return;
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    haar1_scalar(x, stride);
}

/// Haar step of `haar1` for rows of one value (`stride == 1`), with AVX2.
///
/// Panics when `x` holds fewer than `2 * (n0 / 2)` values.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
fn haar1_avx2(x: &mut [f32], n0: usize) {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;
    use crate::simd_mem::x86::{load_f32x8, store_f32x8};
    let n = n0 >> 1;
    // The loads and stores below cover `x[..2 * n]`.
    assert!(x.len() >= 2 * n);
    let x = &mut x[..2 * n];
    let scale = _mm256_set1_ps(core::f32::consts::FRAC_1_SQRT_2);
    let fixup = _mm256_set_epi32(7, 6, 3, 2, 5, 4, 1, 0);
    // Eight pairs per pass, in two vectors of eight values.
    let (octets, _) = x.as_chunks_mut::<8>();
    let (blocks, _) = octets.as_chunks_mut::<2>();
    for [lo, hi] in blocks {
        let a = load_f32x8(lo);
        let b = load_f32x8(hi);

        let t0 = _mm256_unpacklo_ps(a, b);
        let t1 = _mm256_unpackhi_ps(a, b);

        // unpacklo/hi leave the two middle quartets swapped; fix the 32-bit
        // lane order so even/odd hold pairs 0-7 in order.
        let even = _mm256_permutevar8x32_ps(_mm256_unpacklo_ps(t0, t1), fixup);
        let odd = _mm256_permutevar8x32_ps(_mm256_unpackhi_ps(t0, t1), fixup);

        // libopus: tmp1 = s*x0, tmp2 = s*x1, then tmp1 + tmp2 and tmp1 - tmp2.
        let a = _mm256_mul_ps(even, scale);
        let b = _mm256_mul_ps(odd, scale);
        let sum = _mm256_add_ps(a, b);
        let diff = _mm256_sub_ps(a, b);

        let r0 = _mm256_unpacklo_ps(sum, diff);
        let r1 = _mm256_unpackhi_ps(sum, diff);

        let out0 = _mm256_permute2f128_ps(r0, r1, 0x20);
        let out1 = _mm256_permute2f128_ps(r0, r1, 0x31);

        store_f32x8(lo, out0);
        store_f32x8(hi, out1);
    }

    // The pairs after the last pass of eight, one by one.
    let scale = core::f32::consts::FRAC_1_SQRT_2;
    for pair in x[16 * (n / 8)..].as_chunks_mut::<2>().0 {
        let tmp1 = scale * pair[0];
        let tmp2 = scale * pair[1];
        pair[0] = tmp1 + tmp2;
        pair[1] = tmp1 - tmp2;
    }
}

/// Haar step for rows of `S` values. `x.len()` must be a multiple of
/// `2 * S`; a remainder would be left alone.
///
/// The row width is a constant so that the compiler turns the inner loop
/// into vector instructions on every target.
#[inline(always)]
fn haar1_rows<const S: usize>(x: &mut [f32]) {
    let (rows, _) = x.as_chunks_mut::<S>();
    let (row_pairs, _) = rows.as_chunks_mut::<2>();
    for [a, b] in row_pairs {
        for k in 0..S {
            let tmp1 = HAAR_SCALE * a[k];
            let tmp2 = HAAR_SCALE * b[k];
            a[k] = tmp1 + tmp2;
            b[k] = tmp1 - tmp2;
        }
    }
}

/// Portable Haar step. `x.len()` must be a multiple of `2 * stride`.
///
/// This is the fallback on targets without a hand-written kernel and the
/// reference for the unit tests. Each value is computed as in libopus:
/// both inputs are scaled first, then added and subtracted.
#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
#[inline]
fn haar1_scalar(x: &mut [f32], stride: usize) {
    debug_assert!(stride > 0 && x.len().is_multiple_of(2 * stride));
    match stride {
        // The codec uses the widths 1, 2, 4 and 8 (one row per short block).
        1 => haar1_rows::<1>(x),
        2 => haar1_rows::<2>(x),
        4 => haar1_rows::<4>(x),
        _ => haar1_wide(x, stride),
    }
}

/// Portable Haar step for rows of 8 values, and for any other width.
/// `x.len()` must be a multiple of `2 * stride`.
#[inline]
fn haar1_wide(x: &mut [f32], stride: usize) {
    if stride == 8 {
        return haar1_rows::<8>(x);
    }
    for row_pair in x.chunks_exact_mut(2 * stride) {
        let (a, b) = row_pair.split_at_mut(stride);
        for (p, q) in a.iter_mut().zip(b.iter_mut()) {
            let tmp1 = HAAR_SCALE * *p;
            let tmp2 = HAAR_SCALE * *q;
            *p = tmp1 + tmp2;
            *q = tmp1 - tmp2;
        }
    }
}

/// NEON Haar step. `x.len()` must be a multiple of `2 * stride`.
///
/// Row widths 1, 2 and 4 have their own loops. Wider rows use the portable
/// `haar1_wide`, which the compiler already turns into full vectors. The values are
/// bit-identical to `haar1_scalar`: the same multiply, add and subtract per
/// value, with no fused multiply-add.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
fn haar1_neon(x: &mut [f32], stride: usize) {
    use crate::simd_mem::neon::{load_f32x4, load2_f32x4, store_f32x4, store2_f32x4};
    use core::arch::aarch64::*;

    debug_assert!(stride > 0 && x.len().is_multiple_of(2 * stride));
    match stride {
        1 => {
            // Four pairs per pass: split even and odd positions.
            let (blocks, tail) = x.as_chunks_mut::<8>();
            for block in blocks {
                let scale = vdupq_n_f32(HAAR_SCALE);
                let pairs = load2_f32x4(block);
                let even = vmulq_f32(pairs.0, scale);
                let odd = vmulq_f32(pairs.1, scale);
                let out = float32x4x2_t(vaddq_f32(even, odd), vsubq_f32(even, odd));
                store2_f32x4(block, out);
            }
            // At most three pairs are left.
            for pair in tail.as_chunks_mut::<2>().0.iter_mut().take(3) {
                let tmp1 = HAAR_SCALE * pair[0];
                let tmp2 = HAAR_SCALE * pair[1];
                pair[0] = tmp1 + tmp2;
                pair[1] = tmp1 - tmp2;
            }
        }
        2 => {
            // One row pair per vector: the low half is row `2j`, the high
            // half is row `2j + 1`.
            let (blocks, _) = x.as_chunks_mut::<4>();
            for block in blocks {
                let v = vmulq_f32(load_f32x4(block), vdupq_n_f32(HAAR_SCALE));
                let a = vget_low_f32(v);
                let b = vget_high_f32(v);
                store_f32x4(block, vcombine_f32(vadd_f32(a, b), vsub_f32(a, b)));
            }
        }
        4 => {
            // Row `2j` is values 0..4 of a block, row `2j + 1` values 4..8.
            let (rows, _) = x.as_chunks_mut::<4>();
            let (row_pairs, _) = rows.as_chunks_mut::<2>();
            for [row_a, row_b] in row_pairs {
                let scale = vdupq_n_f32(HAAR_SCALE);
                let a = vmulq_f32(load_f32x4(row_a), scale);
                let b = vmulq_f32(load_f32x4(row_b), scale);
                store_f32x4(row_a, vaddq_f32(a, b));
                store_f32x4(row_b, vsubq_f32(a, b));
            }
        }
        _ => haar1_wide(x, stride),
    }
}

#[inline(always)]
pub fn compute_qn(n: usize, b: i32, offset: i32, pulse_cap: i32, stereo: bool) -> i32 {
    static EXP2_TABLE8: [i16; 8] = [16384, 17866, 19483, 21247, 23170, 25267, 27554, 30048];
    let mut n2 = (2 * n as i32) - 1;
    if stereo && n == 2 {
        n2 -= 1;
    }
    let mut qb = celt_sudiv(b + n2 * offset, n2);
    qb = qb.min(b - pulse_cap - (4 << BITRES));
    qb = qb.min(8 << BITRES);
    if qb < (1i32 << BITRES >> 1) {
        1
    } else {
        let val = EXP2_TABLE8[(qb & 0x7) as usize] as i32;
        let shift = 14 - (qb >> BITRES);
        let raw = if (0..32).contains(&shift) {
            val >> shift
        } else {
            0
        };
        let qn = (raw + 1) >> 1 << 1;
        qn.min(256)
    }
}

/// libopus `stereo_itheta` (celt/vq.c 724-754, float build): the angle
/// between `x[..n]` and `y[..n]`, `atan2(side, mid)` in Q30, so that a
/// quarter turn is `1 << 30`.
///
/// With `stereo`, mid and side are the sum and the difference of the two
/// channels, summed one value at a time. Otherwise they are `x` and `y`
/// themselves, summed with `celt_inner_prod`. Both energies start at 0, so
/// two silent halves give the angle 0. The angle is rounded down to a
/// multiple of 64 or finer, from `floor(0.5 + 2^30 * theta)` in single
/// precision. `compute_theta` keeps the top 14 bits (`itheta_q30 >> 16`).
#[inline(always)]
pub fn stereo_itheta(x: &[f32], y: &[f32], stereo: bool, n: usize) -> i32 {
    // The only bounds checks. The loops get slices of exactly `n` values.
    let x = &x[..n];
    let y = &y[..n];
    let (emid, eside) = if stereo {
        // C: `Emid = MAC16_16(Emid, m, m)` in a plain loop, which no
        // compiler may reorder. On aarch64 each step is one `fmadd`.
        let mut emid = 0.0f32;
        let mut eside = 0.0f32;
        for (&xv, &yv) in x.iter().zip(y) {
            let m = xv + yv;
            let s = xv - yv;
            emid = mac(emid, m, m);
            eside = mac(eside, s, s);
        }
        (emid, eside)
    } else {
        // C: `Emid += celt_inner_prod_norm_shift(X, X, N, arch)`, which is
        // `celt_inner_prod` in the float build. Adding it to 0 changes no
        // bits of a sum of squares.
        (celt_inner_prod(x, x), celt_inner_prod(y, y))
    };
    let mid = emid.sqrt();
    let side = eside.sqrt();
    // C: `(int)floor(.5f+65536.f*16384*celt_atan2p_norm(side,mid))`. The
    // product with 2^30 is exact, so fused or not it rounds once, in the
    // add. The value is not negative, so the cast rounds down like floor.
    (0.5 + 1_073_741_824.0 * celt_atan2p_norm(side, mid)) as i32
}

/// libopus `celt_atan_norm` (celt/mathops.h 144-168, float build):
/// `atan(x) * 2 / pi` for `x` in `[0, 1]`, from an odd polynomial.
#[inline(always)]
fn celt_atan_norm(x: f32) -> f32 {
    // The values of the C constants, written as short as each float allows.
    const ATAN2_2_OVER_PI: f32 = core::f32::consts::FRAC_2_PI;
    const A03: f32 = -3.333_166e-1;
    const A05: f32 = 1.996_270_4e-1;
    const A07: f32 = -1.397_658_3e-1;
    const A09: f32 = 9.794_234_5e-2;
    const A11: f32 = -5.777_359e-2;
    const A13: f32 = 2.304_013_6e-2;
    const A15: f32 = -4.355_406e-3;
    let x_sq = x * x;
    // C: `ATAN2_2_OVER_PI * (x + x * x_sq * (A03 + x_sq * (A05 + ...)))`.
    // Each `c + x_sq * p` and the final `x + (x * x_sq) * p` are one fused
    // step on aarch64 (see `mac`); the product with the constant is not.
    let mut p = mac(A13, x_sq, A15);
    p = mac(A11, x_sq, p);
    p = mac(A09, x_sq, p);
    p = mac(A07, x_sq, p);
    p = mac(A05, x_sq, p);
    p = mac(A03, x_sq, p);
    ATAN2_2_OVER_PI * mac(x, x * x_sq, p)
}

/// libopus `celt_atan2p_norm` (celt/mathops.h 172-188, float build):
/// `atan2(y, x) * 2 / pi` for `x, y >= 0`, so a quarter turn is 1.
#[inline(always)]
fn celt_atan2p_norm(y: f32, x: f32) -> f32 {
    // C: `(x*x + y*y) < 1e-18f`. clang computes `y*y` and fuses `x*x` into
    // the add on aarch64.
    if mac(y * y, x, x) < 1e-18 {
        return 0.0;
    }
    if y < x {
        celt_atan_norm(y / x)
    } else {
        1.0 - celt_atan_norm(x / y)
    }
}

pub struct SplitCtx {
    pub inv: bool,
    pub imid: i32,
    pub iside: i32,
    pub delta: i32,
    pub itheta: i32,
    pub qalloc: i32,
}

/// Computes and codes the angle between `x` and `y`: the two halves of a
/// mono band (`stereo == false`) or the two channels of a stereo band.
///
/// The stereo encoder also changes the band in place, as libopus does. After
/// the call `x` holds the mid signal and `y` the side signal, or `x` holds
/// the intensity-stereo mix. This is why both slices are `&mut`.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub fn compute_theta(
    ctx: &mut BandCtx,
    sctx: &mut SplitCtx,
    x: &mut [f32],
    y: &mut [f32],
    n: usize,
    b: &mut i32,
    b_blocks: i32,
    b0: i32,
    lm: i32,
    stereo: bool,
    fill: &mut u32,
) {
    let pulse_cap = ctx.m.log_n[ctx.i] as i32 + (lm << BITRES);
    let offset = (pulse_cap >> 1) - if stereo && n == 2 { 16 } else { 4 };
    let mut qn = compute_qn(n, *b, offset, pulse_cap, stereo);

    if stereo && ctx.i >= ctx.intensity {
        qn = 1;
    }

    let mut itheta = 0;
    if ctx.encode {
        // C (bands.c 741-742): the angle in Q30, rounded down to Q14.
        let q30 = if stereo {
            *ctx.stereo_angle.get_or_insert_with(|| stereo_itheta(x, y, true, n))
        } else {
            stereo_itheta(x, y, false, n)
        };
        itheta = q30 >> 16;
    }

    let tell_start = tell_frac_inline!(ctx.rc);

    if qn != 1 {
        if ctx.encode {
            if !stereo || ctx.theta_round == 0 {
                itheta = (itheta * qn + 8192) >> 14;
                if !stereo && ctx.avoid_split_noise && itheta > 0 && itheta < qn {
                    let unquantized = (itheta * 16384) / qn;
                    let imid = bitexact_cos(unquantized as i16) as i32;
                    let iside = bitexact_cos((16384 - unquantized) as i16) as i32;
                    let delta =
                        (((n as i32 - 1) << 7) * bitexact_log2tan(iside, imid) + 16384) >> 15;
                    if delta > *b {
                        itheta = qn;
                    } else if delta < -*b {
                        itheta = 0;
                    }
                }
            } else {
                let bias = if itheta > 8192 {
                    32767 / qn
                } else {
                    -32767 / qn
                };
                let down = (itheta * qn + bias) >> 14;
                let down = down.clamp(0, qn - 1);
                if ctx.theta_round < 0 {
                    itheta = down;
                } else {
                    itheta = down + 1;
                }
            }
        }

        if stereo && n > 2 {
            let p0 = 3;
            let x0 = qn / 2;
            let ft = p0 * (x0 + 1) + x0;
            if ctx.encode {
                let fl = if itheta <= x0 {
                    p0 * itheta
                } else {
                    (itheta - 1 - x0) + (x0 + 1) * p0
                };
                let fh = if itheta <= x0 {
                    p0 * (itheta + 1)
                } else {
                    (itheta - x0) + (x0 + 1) * p0
                };
                ctx.rc.encode(fl as u32, fh as u32, ft as u32);
            } else {
                let fs = ctx.rc.decode(ft as u32);
                if fs < (x0 + 1) as u32 * p0 as u32 {
                    itheta = fs as i32 / p0;
                } else {
                    itheta = (x0 + 1) + (fs as i32 - (x0 + 1) * p0);
                }
                let fl = if itheta <= x0 {
                    p0 * itheta
                } else {
                    (itheta - 1 - x0) + (x0 + 1) * p0
                };
                let fh = if itheta <= x0 {
                    p0 * (itheta + 1)
                } else {
                    (itheta - x0) + (x0 + 1) * p0
                };
                ctx.rc.update(fl as u32, fh as u32, ft as u32);
            }
        } else if b0 > 1 || stereo {
            if ctx.encode {
                ctx.rc.enc_uint(itheta as u32, (qn + 1) as u32);
            } else {
                itheta = ctx.rc.dec_uint((qn + 1) as u32) as i32;
            }
        } else {
            let ft = ((qn >> 1) + 1) * ((qn >> 1) + 1);
            if ctx.encode {
                let fs = if itheta <= (qn >> 1) {
                    itheta + 1
                } else {
                    qn + 1 - itheta
                };
                let fl = if itheta <= (qn >> 1) {
                    (itheta * (itheta + 1)) >> 1
                } else {
                    ft - (((qn + 1 - itheta) * (qn + 2 - itheta)) >> 1)
                };
                ctx.rc.encode(fl as u32, (fl + fs) as u32, ft as u32);
            } else {
                let fm = ctx.rc.decode(ft as u32) as i32;
                if fm < (((qn >> 1) * ((qn >> 1) + 1)) >> 1) {
                    itheta = (isqrt32((8 * fm + 1) as u32) as i32 - 1) >> 1;
                    let fl = (itheta * (itheta + 1)) >> 1;
                    let fs = itheta + 1;
                    ctx.rc.update(fl as u32, (fl + fs) as u32, ft as u32);
                } else {
                    itheta = (2 * (qn + 1) - isqrt32((8 * (ft - fm - 1) + 1) as u32) as i32) >> 1;
                    let fs = qn + 1 - itheta;
                    let fl = ft - (((qn + 1 - itheta) * (qn + 2 - itheta)) >> 1);
                    ctx.rc.update(fl as u32, (fl + fs) as u32, ft as u32);
                }
            }
        }
        itheta = (itheta as u32 * 16384 / qn as u32) as i32;
        if ctx.encode && stereo {
            if itheta == 0 {
                intensity_stereo(ctx.m, x, y, ctx.band_e, ctx.i, n);
            } else {
                stereo_split(x, y, n);
            }
        }
    } else if stereo {
        if ctx.encode {
            let inv = itheta > 8192 && !ctx.disable_inv;
            if inv {
                // The two channels are out of phase. Negate the right one,
                // so that the mix below does not cancel. The decoder negates
                // it again when it reads the `inv` bit. Same as libopus
                // (celt/bands.c, compute_theta).
                for yv in y[..n].iter_mut() {
                    *yv = -*yv;
                }
            }
            intensity_stereo(ctx.m, x, y, ctx.band_e, ctx.i, n);
            // C: the flag is coded only when there are bits for it, and is 0
            // otherwise, for the encoder too. Its resynthesis then keeps the
            // right channel as the decoder will have it.
            let inv = if *b > (2 << BITRES) && ctx.remaining_bits > (2 << BITRES) {
                ctx.rc.encode_bit_logp(inv, 2);
                inv
            } else {
                false
            };
            itheta = 0;
            sctx.inv = inv;
        } else {
            if *b > (2 << BITRES) && ctx.remaining_bits > (2 << BITRES) {
                sctx.inv = ctx.rc.decode_bit_logp(2);
            } else {
                sctx.inv = false;
            }
            if ctx.disable_inv {
                sctx.inv = false;
            }
            itheta = 0;
        }
    }

    sctx.itheta = itheta;

    sctx.qalloc = tell_frac_inline!(ctx.rc) - tell_start;
    *b -= sctx.qalloc; // matches C: *b -= qalloc

    if itheta == 0 {
        sctx.imid = 32767;
        sctx.iside = 0;
        sctx.delta = -16384;
        *fill &= (1 << b_blocks) - 1;
    } else if itheta == 16384 {
        sctx.imid = 0;
        sctx.iside = 32767;
        sctx.delta = 16384;
        *fill &= ((1 << b_blocks) - 1) << b_blocks;
    } else {
        let imid = bitexact_cos(itheta as i16);
        sctx.imid = imid as i32;
        let iside = bitexact_cos((16384 - itheta) as i16);
        sctx.iside = iside as i32;
        sctx.delta =
            (((n as i32 - 1) << 7) * bitexact_log2tan(sctx.iside, sctx.imid) + 16384) >> 15;
    }
}

/// Whether a band part of `n` values with `b` bits (in 1/8 bit) is split in
/// two. It is when `b` exceeds, by more than 12, the cost of the largest
/// pulse count the mode's table has for this band size (libopus:
/// `b > cache[cache[0]] + 12` in quant_partition).
#[inline(always)]
fn band_should_split(m: &CeltMode, band: usize, lm: i32, n: usize, b: i32) -> bool {
    if lm < 0 || n <= 2 {
        return false;
    }
    let cache_idx = (lm + 1) as usize * m.nb_ebands + band;
    debug_assert!(cache_idx < m.cache.index.len());
    // SAFETY: `cache.index` has one entry per pair of band size and band,
    // `(max_lm + 2) * nb_ebands` entries in all. `lm` is at most `max_lm`
    // and `band` is below `nb_ebands` for every caller (quant_all_bands
    // walks the bands of the mode).
    let cache_base = unsafe { *m.cache.index.get_unchecked(cache_idx) };
    if cache_base < 0 {
        return false;
    }
    let cache_base = cache_base as usize;
    debug_assert!(cache_base < m.cache.bits.len());
    // SAFETY: a non-negative entry of `cache.index` is the start of one
    // record in `cache.bits`. The first byte of a record is its largest
    // pulse count `max_q`, and the record holds the bytes `0..=max_q`. The
    // tables in modes.rs are the cache tables of libopus and have this
    // layout.
    unsafe {
        let cache_ptr = m.cache.bits.as_ptr().add(cache_base);
        let max_q = *cache_ptr as usize;
        debug_assert!(cache_base + max_q < m.cache.bits.len());
        b > (*cache_ptr.add(max_q) as i32) + 12
    }
}

/// `LCG_AHEAD[k - 1]` takes the noise generator `k` steps ahead, for `k`
/// from 1 to 8: the seed after `k` steps of `celt_lcg_rand` is
/// `seed * multiplier + increment` (wrapping).
///
/// One step waits for the step before it. With these constants eight
/// values are computed from the same seed at once.
#[cfg_attr(not(any(target_arch = "aarch64", all(target_arch = "x86_64", target_feature = "sse2"))), allow(dead_code))]
const LCG_AHEAD: [(u32, u32); 8] = {
    let mut table = [(0u32, 0u32); 8];
    let (mut mul, mut add) = (1u32, 0u32);
    let mut k = 0;
    while k < 8 {
        // One more step on top of `seed * mul + add`.
        mul = mul.wrapping_mul(1664525);
        add = add.wrapping_mul(1664525).wrapping_add(1013904223);
        table[k] = (mul, add);
        k += 1;
    }
    table
};

/// Portable form of `fold_with_noise`; `x` and `lb` have the same length.
#[cfg_attr(any(target_arch = "aarch64", all(target_arch = "x86_64", target_feature = "sse2")), allow(dead_code))]
#[inline(always)]
fn fold_with_noise_scalar(x: &mut [f32], lb: &[f32], mut seed: u32) -> u32 {
    debug_assert_eq!(x.len(), lb.len());
    for (xv, &l) in x.iter_mut().zip(lb) {
        seed = celt_lcg_rand(seed);
        *xv = l + if seed & 0x8000 != 0 {
            1.0 / 256.0
        } else {
            -1.0 / 256.0
        };
    }
    seed
}

/// Portable form of `fill_with_noise`.
#[cfg_attr(any(target_arch = "aarch64", all(target_arch = "x86_64", target_feature = "sse2")), allow(dead_code))]
#[inline(always)]
fn fill_with_noise_scalar(x: &mut [f32], mut seed: u32) -> u32 {
    for xv in x.iter_mut() {
        seed = celt_lcg_rand(seed);
        *xv = ((seed as i32 >> 20) as f32) / 16384.0;
    }
    seed
}

/// The seeds of the next eight values, as two vectors, and the seed after
/// them in every lane of a third one.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
fn lcg_next8(
    vseed: core::arch::aarch64::uint32x4_t,
) -> (
    core::arch::aarch64::uint32x4_t,
    core::arch::aarch64::uint32x4_t,
    core::arch::aarch64::uint32x4_t,
) {
    use crate::simd_mem::neon::load_u32x4;
    use core::arch::aarch64::*;
    const MUL: [u32; 8] = {
        let mut m = [0u32; 8];
        let mut k = 0;
        while k < 8 {
            m[k] = LCG_AHEAD[k].0;
            k += 1;
        }
        m
    };
    const ADD: [u32; 8] = {
        let mut a = [0u32; 8];
        let mut k = 0;
        while k < 8 {
            a[k] = LCG_AHEAD[k].1;
            k += 1;
        }
        a
    };
    let (mul, _) = MUL.as_chunks::<4>();
    let (add, _) = ADD.as_chunks::<4>();
    let lo = vmlaq_u32(load_u32x4(&add[0]), vseed, load_u32x4(&mul[0]));
    let hi = vmlaq_u32(load_u32x4(&add[1]), vseed, load_u32x4(&mul[1]));
    // The seed eight steps ahead, computed in all lanes, so that the
    // next pass does not wait for a move between registers.
    let next = vmlaq_u32(vdupq_n_u32(ADD[7]), vseed, vdupq_n_u32(MUL[7]));
    (lo, hi, next)
}

/// NEON form of `fold_with_noise`: eight values per pass. `lb` must be at
/// least as long as `x`.
///
/// The loop is here, in a function that is always inlined, and one pass is
/// `fold_with_noise8_neon`. A `#[target_feature]` function cannot be
/// `#[inline(always)]`, and with the whole loop in one the compiler may
/// inline it later and lay out the code around it differently.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn fold_with_noise_neon(x: &mut [f32], lb: &[f32], seed: u32) -> u32 {
    use core::arch::aarch64::*;

    let lb = &lb[..x.len()];
    let (x8, x_tail) = x.as_chunks_mut::<8>();
    let (lb8, lb_tail) = lb.as_chunks::<8>();
    // SAFETY: NEON is available: src/lib.rs refuses to build for aarch64
    // without it.
    let mut vseed = unsafe { vdupq_n_u32(seed) };
    for (xo, l) in x8.iter_mut().zip(lb8) {
        // SAFETY: NEON is available: src/lib.rs refuses to build for aarch64
        // without it.
        vseed = unsafe { fold_with_noise8_neon(xo, l, vseed) };
    }
    // SAFETY: NEON is available: src/lib.rs refuses to build for aarch64
    // without it.
    let seed = unsafe { vgetq_lane_u32::<0>(vseed) };
    fold_with_noise_scalar(x_tail, lb_tail, seed)
}

/// One pass of `fold_with_noise_neon`: `xo = l + noise` for eight values,
/// with the seeds `lcg_next8(vseed)`. Returns the seed after them, in every
/// lane.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
fn fold_with_noise8_neon(
    xo: &mut [f32; 8],
    l: &[f32; 8],
    vseed: core::arch::aarch64::uint32x4_t,
) -> core::arch::aarch64::uint32x4_t {
    use crate::simd_mem::neon::{load_f32x4, store_f32x4};
    use core::arch::aarch64::*;

    let bit = vdupq_n_u32(0x8000);
    let plus = vdupq_n_f32(1.0 / 256.0);
    let minus = vdupq_n_f32(-1.0 / 256.0);
    let (s_lo, s_hi, next) = lcg_next8(vseed);
    let n_lo = vbslq_f32(vtstq_u32(s_lo, bit), plus, minus);
    let n_hi = vbslq_f32(vtstq_u32(s_hi, bit), plus, minus);
    let (l, _) = l.as_chunks::<4>();
    let (xo, _) = xo.as_chunks_mut::<4>();
    store_f32x4(&mut xo[0], vaddq_f32(load_f32x4(&l[0]), n_lo));
    store_f32x4(&mut xo[1], vaddq_f32(load_f32x4(&l[1]), n_hi));
    next
}

/// NEON form of `fill_with_noise`: eight values per pass. The loop is
/// here and one pass is `fill_with_noise8_neon`, for the reason given at
/// `fold_with_noise_neon`.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn fill_with_noise_neon(x: &mut [f32], seed: u32) -> u32 {
    use core::arch::aarch64::*;

    let (x8, x_tail) = x.as_chunks_mut::<8>();
    // SAFETY: NEON is available: src/lib.rs refuses to build for aarch64
    // without it.
    let mut vseed = unsafe { vdupq_n_u32(seed) };
    for xo in x8.iter_mut() {
        // SAFETY: NEON is available: src/lib.rs refuses to build for aarch64
        // without it.
        vseed = unsafe { fill_with_noise8_neon(xo, vseed) };
    }
    // SAFETY: NEON is available: src/lib.rs refuses to build for aarch64
    // without it.
    let seed = unsafe { vgetq_lane_u32::<0>(vseed) };
    fill_with_noise_scalar(x_tail, seed)
}

/// One pass of `fill_with_noise_neon`: eight values from the seeds
/// `lcg_next8(vseed)`. Returns the seed after them, in every lane.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
fn fill_with_noise8_neon(
    xo: &mut [f32; 8],
    vseed: core::arch::aarch64::uint32x4_t,
) -> core::arch::aarch64::uint32x4_t {
    use crate::simd_mem::neon::store_f32x4;
    use core::arch::aarch64::*;

    let (s_lo, s_hi, next) = lcg_next8(vseed);
    // The top 12 bits as a signed number, divided by 2^14. The conversion
    // with 14 fraction bits does the division, exactly.
    let v_lo = vcvtq_n_f32_s32::<14>(vshrq_n_s32::<20>(vreinterpretq_s32_u32(s_lo)));
    let v_hi = vcvtq_n_f32_s32::<14>(vshrq_n_s32::<20>(vreinterpretq_s32_u32(s_hi)));
    let (xo, _) = xo.as_chunks_mut::<4>();
    store_f32x4(&mut xo[0], v_lo);
    store_f32x4(&mut xo[1], v_hi);
    next
}

/// The seeds of the next eight values after `seed`, as two vectors, for
/// SSE2. `_mm_mul_epu32` multiplies lanes 0 and 2 into 64-bit products;
/// the lower halves of those, for the even and the odd steps, are the
/// wrapping products of the scalar code.
#[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
#[target_feature(enable = "sse2")]
#[inline]
fn lcg_next8_sse2(seed: u32) -> (core::arch::x86_64::__m128i, core::arch::x86_64::__m128i) {
    use core::arch::x86_64::*;
    const fn lanes(first: usize, field: usize) -> [i32; 4] {
        let mut v = [0i32; 4];
        let mut k = 0;
        while k < 4 {
            let (mul, add) = LCG_AHEAD[first + k];
            v[k] = if field == 0 { mul as i32 } else { add as i32 };
            k += 1;
        }
        v
    }
    /// `seed * mul[k] + add[k]` for the four lanes `k`, wrapping.
    #[target_feature(enable = "sse2")]
    #[inline]
    fn four(s: __m128i, mul: [i32; 4], add: [i32; 4]) -> __m128i {
        // Steps 0 and 2 in the 64-bit lanes of `even`, 1 and 3 in `odd`.
        let even = _mm_mul_epu32(s, _mm_setr_epi32(mul[0], 0, mul[2], 0));
        let odd = _mm_mul_epu32(s, _mm_setr_epi32(mul[1], 0, mul[3], 0));
        let low = _mm_unpacklo_epi32(
            _mm_shuffle_epi32::<0b11_11_10_00>(even),
            _mm_shuffle_epi32::<0b11_11_10_00>(odd),
        );
        _mm_add_epi32(low, _mm_setr_epi32(add[0], add[1], add[2], add[3]))
    }
    let s = _mm_set1_epi32(seed as i32);
    (four(s, lanes(0, 0), lanes(0, 1)), four(s, lanes(4, 0), lanes(4, 1)))
}

/// SSE2 form of `fold_with_noise`: eight values per pass, with the seed
/// eight steps ahead computed in a general register. `lb` must be at least
/// as long as `x`.
///
/// The loop is here and one pass is `fold_with_noise8_sse2`, for the
/// reason given at `fold_with_noise_neon`.
#[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
#[inline(always)]
fn fold_with_noise_sse2(x: &mut [f32], lb: &[f32], mut seed: u32) -> u32 {
    let lb = &lb[..x.len()];
    let (x8, x_tail) = x.as_chunks_mut::<8>();
    let (lb8, lb_tail) = lb.as_chunks::<8>();
    let (mul8, add8) = LCG_AHEAD[7];
    for (xo, l) in x8.iter_mut().zip(lb8) {
        // SAFETY: SSE2 is part of this target (see the `cfg`).
        unsafe { fold_with_noise8_sse2(xo, l, seed) };
        seed = seed.wrapping_mul(mul8).wrapping_add(add8);
    }
    fold_with_noise_scalar(x_tail, lb_tail, seed)
}

/// One pass of `fold_with_noise_sse2`: `xo = l + noise` for eight values,
/// with the seeds `lcg_next8_sse2(seed)`.
///
/// The noise is -1/256 with its sign bit flipped where bit 15 of the seed
/// is set: bit 15 shifted to bit 31 and added with an exclusive or. The
/// sum is the scalar code's.
#[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
#[target_feature(enable = "sse2")]
#[inline]
fn fold_with_noise8_sse2(xo: &mut [f32; 8], l: &[f32; 8], seed: u32) {
    use crate::simd_mem::x86::{load_f32x4, store_f32x4};
    use core::arch::x86_64::*;

    let minus = _mm_set1_epi32((-1.0f32 / 256.0).to_bits() as i32);
    let bit = _mm_set1_epi32(0x8000);
    let noise = |s: __m128i| _mm_castsi128_ps(_mm_xor_si128(minus, _mm_slli_epi32::<16>(_mm_and_si128(s, bit))));
    let (s_lo, s_hi) = lcg_next8_sse2(seed);
    let (l, _) = l.as_chunks::<4>();
    let (xo, _) = xo.as_chunks_mut::<4>();
    store_f32x4(&mut xo[0], _mm_add_ps(load_f32x4(&l[0]), noise(s_lo)));
    store_f32x4(&mut xo[1], _mm_add_ps(load_f32x4(&l[1]), noise(s_hi)));
}

/// SSE2 form of `fill_with_noise`: eight values per pass. The loop is here
/// and one pass is `fill_with_noise8_sse2`, for the reason given at
/// `fold_with_noise_neon`.
#[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
#[inline(always)]
fn fill_with_noise_sse2(x: &mut [f32], mut seed: u32) -> u32 {
    let (x8, x_tail) = x.as_chunks_mut::<8>();
    let (mul8, add8) = LCG_AHEAD[7];
    for xo in x8.iter_mut() {
        // SAFETY: SSE2 is part of this target (see the `cfg`).
        unsafe { fill_with_noise8_sse2(xo, seed) };
        seed = seed.wrapping_mul(mul8).wrapping_add(add8);
    }
    fill_with_noise_scalar(x_tail, seed)
}

/// One pass of `fill_with_noise_sse2`: eight values from the seeds
/// `lcg_next8_sse2(seed)`. The top 12 bits as a signed number, converted
/// to `f32` (exact) and multiplied by 2^-14, which is exact as the division
/// by 16384 of the scalar code.
#[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
#[target_feature(enable = "sse2")]
#[inline]
fn fill_with_noise8_sse2(xo: &mut [f32; 8], seed: u32) {
    use crate::simd_mem::x86::store_f32x4;
    use core::arch::x86_64::*;

    let scale = _mm_set1_ps(1.0 / 16384.0);
    let value = |s: __m128i| _mm_mul_ps(_mm_cvtepi32_ps(_mm_srai_epi32::<20>(s)), scale);
    let (s_lo, s_hi) = lcg_next8_sse2(seed);
    let (xo, _) = xo.as_chunks_mut::<4>();
    store_f32x4(&mut xo[0], value(s_lo));
    store_f32x4(&mut xo[1], value(s_hi));
}

/// Fills a band that got no pulses with the folded band `lb` plus noise:
/// `x[j] = lb[j] + 1/256` or `lb[j] - 1/256`, by bit 15 of one step of the
/// noise generator per value (libopus: quant_partition, the `lowband`
/// case). Returns the seed after the last value. `x` and `lb` must have the
/// same length.
#[inline(always)]
fn fold_with_noise(x: &mut [f32], lb: &[f32], seed: u32) -> u32 {
    #[cfg(target_arch = "aarch64")]
    {
        fold_with_noise_neon(x, lb, seed)
    }
    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    {
        fold_with_noise_sse2(x, lb, seed)
    }
    #[cfg(not(any(target_arch = "aarch64", all(target_arch = "x86_64", target_feature = "sse2"))))]
    {
        fold_with_noise_scalar(x, lb, seed)
    }
}

/// Fills a band that got no pulses and has no band to fold with noise: the
/// top 12 bits of one step of the noise generator per value, as a signed
/// number divided by 16384 (libopus: quant_partition, the noise case).
/// Returns the seed after the last value.
#[inline(always)]
fn fill_with_noise(x: &mut [f32], seed: u32) -> u32 {
    #[cfg(target_arch = "aarch64")]
    {
        fill_with_noise_neon(x, seed)
    }
    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    {
        fill_with_noise_sse2(x, seed)
    }
    #[cfg(not(any(target_arch = "aarch64", all(target_arch = "x86_64", target_feature = "sse2"))))]
    {
        fill_with_noise_scalar(x, seed)
    }
}

/// libopus `quant_partition` (celt/bands.c): codes or decodes one band part
/// of `n` values with `b` bits (in 1/8 bit). A part with more bits than its
/// largest pulse count needs is split in two halves, recursively; the angle
/// between the halves decides how the bits are shared.
///
/// `ENCODE_ONLY` is the encoder without resynthesis (`ctx.encode` set and
/// `ctx.resynth` not set). With it set, the compiler knows both flags, and
/// the leaves of the sizes 2, 4, 8 and 16 get a pulse search for that size.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn quant_partition<const ENCODE_ONLY: bool>(
    ctx: &mut BandCtx,
    x: &mut [f32],
    n: usize,
    b: i32,
    b_blocks: i32,
    lowband: Option<&mut [f32]>,
    lm: i32,
    gain: f32,
    fill: u32,
) -> u32 {
    if !band_should_split(ctx.m, ctx.i, lm, n, b) {
        if !ENCODE_ONLY {
            return quant_partition_leaf::<false>(ctx, x, n, b, b_blocks, lowband, lm, gain, fill);
        }
        let leaf = quant_partition_leaf::<true>;
        return match n {
            2 => leaf(ctx, x, 2, b, b_blocks, lowband, lm, gain, fill),
            4 => leaf(ctx, x, 4, b, b_blocks, lowband, lm, gain, fill),
            8 => leaf(ctx, x, 8, b, b_blocks, lowband, lm, gain, fill),
            16 => leaf(ctx, x, 16, b, b_blocks, lowband, lm, gain, fill),
            _ => leaf(ctx, x, n, b, b_blocks, lowband, lm, gain, fill),
        };
    }

    let mut sctx = SplitCtx {
        inv: false,
        imid: 0,
        iside: 0,
        delta: 0,
        itheta: 0,
        qalloc: 0,
    };
    let mut b = b;
    let mut fill = fill;
    let mid = n / 2;
    let lm = lm - 1;
    let b0 = b_blocks;
    if b_blocks == 1 {
        fill = (fill & 1) | (fill << 1);
    }
    let b_blocks = (b_blocks + 1) >> 1;
    let (x_mid, x_side) = x.split_at_mut(mid);

    compute_theta(
        ctx, &mut sctx, x_mid, x_side, mid, &mut b, b_blocks, b0, lm, false, &mut fill,
    );

    ctx.remaining_bits -= sctx.qalloc;
    let mut delta = sctx.delta;
    // C: "Give more bits to low-energy MDCTs than they would otherwise
    // deserve".
    if b0 > 1 && (sctx.itheta & 0x3fff) != 0 {
        if sctx.itheta > 8192 {
            delta -= delta >> (4 - lm);
        } else {
            delta = 0.min(delta + ((mid as i32) << BITRES >> (5 - lm)));
        }
    }
    let mut mbits = 0.max((b - delta) / 2).min(b);
    let mut sbits = b - mbits;
    let mid_gain = gain * (sctx.imid as f32 / 32768.0);
    let side_gain = gain * (sctx.iside as f32 / 32768.0);
    let (lb_mid, lb_side) = match lowband {
        Some(lb) => {
            let (lb_mid, lb_side) = lb.split_at_mut(mid);
            (Some(lb_mid), Some(lb_side))
        }
        None => (None, None),
    };

    // The half with more bits goes first. The bits it leaves unused go to
    // the other half.
    let rebalance = ctx.remaining_bits;
    if mbits >= sbits {
        let mut cm = quant_partition::<ENCODE_ONLY>(
            ctx, x_mid, mid, mbits, b_blocks, lb_mid, lm, mid_gain, fill,
        );
        let rebalance = mbits - (rebalance - ctx.remaining_bits);
        if rebalance > (3 << 3) && sctx.itheta != 0 {
            sbits += rebalance - (3 << 3);
        }
        cm |= quant_partition::<ENCODE_ONLY>(
            ctx,
            x_side,
            mid,
            sbits,
            b_blocks,
            lb_side,
            lm,
            side_gain,
            fill >> b_blocks,
        ) << (b0 >> 1);
        cm
    } else {
        let mut cm = quant_partition::<ENCODE_ONLY>(
            ctx,
            x_side,
            mid,
            sbits,
            b_blocks,
            lb_side,
            lm,
            side_gain,
            fill >> b_blocks,
        ) << (b0 >> 1);
        let rebalance = sbits - (rebalance - ctx.remaining_bits);
        if rebalance > (3 << 3) && sctx.itheta != 16384 {
            mbits += rebalance - (3 << 3);
        }
        cm |= quant_partition::<ENCODE_ONLY>(
            ctx, x_mid, mid, mbits, b_blocks, lb_mid, lm, mid_gain, fill,
        );
        cm
    }
}

/// A band part that `quant_partition` does not split: the pulses its bits
/// pay for are coded or decoded. Without pulses and with resynthesis, the
/// part is filled from the folded band `lowband` or with noise. Returns the
/// collapse mask of the part. See `quant_partition` for `ENCODE_ONLY`.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn quant_partition_leaf<const ENCODE_ONLY: bool>(
    ctx: &mut BandCtx,
    x: &mut [f32],
    n: usize,
    b: i32,
    b_blocks: i32,
    lowband: Option<&mut [f32]>,
    lm: i32,
    gain: f32,
    fill: u32,
) -> u32 {
    let (encode, resynth) = if ENCODE_ONLY {
        (true, false)
    } else {
        (ctx.encode, ctx.resynth)
    };
    let mut q = bits2pulses(ctx.m, ctx.i, lm, b);
    let mut curr_bits = pulses2bits(ctx.m, ctx.i, lm, q);
    ctx.remaining_bits -= curr_bits;
    // C: "Ensures we can never bust the budget".
    while ctx.remaining_bits < 0 && q > 0 {
        ctx.remaining_bits += curr_bits;
        q -= 1;
        curr_bits = pulses2bits(ctx.m, ctx.i, lm, q);
        ctx.remaining_bits -= curr_bits;
    }

    if q != 0 {
        let k = get_pulses(q);
        return if encode {
            alg_quant_with(
                x,
                n,
                k,
                ctx.spread,
                b_blocks as usize,
                ctx.rc,
                gain,
                resynth,
                ctx.rotations.as_deref_mut(),
            )
        } else {
            alg_unquant(x, n, k, ctx.spread, b_blocks as usize, ctx.rc, gain)
        };
    }
    // Without resynthesis the part is not filled and its collapse mask
    // stays 0.
    if !resynth {
        return 0;
    }
    let cm_mask = (1u32 << b_blocks) - 1;
    let fill = fill & cm_mask;
    // One bounds check; the kernels need none.
    let x = &mut x[..n];
    if fill == 0 {
        x.fill(0.0);
        0
    } else if let Some(lb) = lowband {
        ctx.seed = fold_with_noise(x, &lb[..n], ctx.seed);
        renormalise_vector(x, n, gain);
        fill
    } else {
        ctx.seed = fill_with_noise(x, ctx.seed);
        renormalise_vector(x, n, gain);
        cm_mask
    }
}

/// Hadamard row order for `stride` rows. `ORDERY_TABLE` holds the orders
/// for 2, 4, 8 and 16 rows, one after the other.
///
/// Panics for any other number of rows: the table has no order for it.
#[inline(always)]
fn hadamard_order(stride: usize) -> &'static [i32] {
    let offset = match stride {
        2 => 0,
        4 => 2,
        8 => 6,
        16 => 14,
        _ => panic!("no Hadamard order for {stride} rows"),
    };
    &ORDERY_TABLE[offset..offset + stride]
}

// Each order in `ORDERY_TABLE` names every row exactly once.
const _: () = {
    let mut stride = 2;
    let mut offset = 0;
    while stride <= 16 {
        let mut seen = 0u32;
        let mut i = 0;
        while i < stride {
            seen |= 1 << ORDERY_TABLE[offset + i];
            i += 1;
        }
        assert!(seen == (1u32 << stride) - 1);
        offset += stride;
        stride *= 2;
    }
};

/// Scratch buffer of the reordering functions below. They write all the
/// values they use before they read them.
type HadamardScratch = [core::mem::MaybeUninit<f32>; MAX_PVQ_N];

/// The first `n` values of `tmp` as initialised values.
///
/// # Safety
///
/// The caller must have written `tmp[..n]` completely.
#[inline(always)]
unsafe fn hadamard_written(tmp: &HadamardScratch, n: usize) -> &[f32] {
    let tmp = &tmp[..n];
    // SAFETY: `MaybeUninit<f32>` has the layout of `f32`, and the caller
    // wrote all `n` values.
    unsafe { &*(tmp as *const [core::mem::MaybeUninit<f32>] as *const [f32]) }
}

/// `deinterleave_hadamard` for `S` rows; `x` holds exactly `n0 * S` values
/// and `n0 >= 1`. Value `i` of every group goes to row `order[i]`, or to row
/// `i` without an order.
///
/// The number of rows is a constant so that the compiler can use vector
/// loads that skip `S - 1` values.
#[inline(always)]
fn deinterleave_rows<const S: usize>(x: &mut [f32], n0: usize, order: Option<&[i32]>) {
    let n = x.len();
    debug_assert!(n0 >= 1 && n == n0 * S);
    let mut tmp_buf: HadamardScratch = [core::mem::MaybeUninit::uninit(); MAX_PVQ_N];
    {
        // A band never has more than MAX_PVQ_N values; a longer one panics here.
        let mut rows_left = tmp_buf[..n].chunks_exact_mut(n0);
        let mut rows: [Option<&mut [core::mem::MaybeUninit<f32>]>; S] =
            core::array::from_fn(|_| rows_left.next());
        let (groups, _) = x.as_chunks::<S>();
        for i in 0..S {
            let row = order.map_or(i, |o| o[i] as usize);
            // Each row is handed out once. A row that is asked for twice
            // means the order is not a permutation.
            let dst = rows[row].take().expect("each row is used once");
            for (d, g) in dst.iter_mut().zip(groups) {
                d.write(g[i]);
            }
        }
    }
    // SAFETY: the loop above ran `S` times and took a different row of
    // `tmp_buf[..n]` each time, so it used all `S` rows. `x` has `n0` groups,
    // so every row of `n0` values was written completely.
    x.copy_from_slice(unsafe { hadamard_written(&tmp_buf, n) });
}

/// `interleave_hadamard` for `S` rows; `x` holds exactly `n0 * S` values.
/// Value `i` of every group comes from row `order[i]`, or from row `i`
/// without an order.
#[inline(always)]
fn interleave_rows<const S: usize>(x: &mut [f32], n0: usize, order: Option<&[i32]>) {
    let n = x.len();
    debug_assert!(n == n0 * S);
    let mut tmp_buf: HadamardScratch = [core::mem::MaybeUninit::uninit(); MAX_PVQ_N];
    {
        // A band never has more than MAX_PVQ_N values; a longer one panics here.
        let (groups, _) = tmp_buf[..n].as_chunks_mut::<S>();
        for i in 0..S {
            let row = order.map_or(i, |o| o[i] as usize);
            let src = &x[row * n0..(row + 1) * n0];
            for (g, &v) in groups.iter_mut().zip(src) {
                g[i].write(v);
            }
        }
    }
    // SAFETY: `tmp_buf[..n]` is `n0` groups of `S` values. Every row of `x`
    // has `n0` values, so pass `i` of the loop above wrote value `i` of
    // every group, and the `S` passes wrote all of them.
    x.copy_from_slice(unsafe { hadamard_written(&tmp_buf, n) });
}

/// Reorders `x` from `n0` groups of `stride` values (one value per short
/// block) to `stride` rows of `n0` values (one row per short block). With
/// `hadamard`, the rows are stored in the order of the Hadamard transform.
///
/// Panics when `x` has fewer than `n0 * stride` values, when that product
/// exceeds `MAX_PVQ_N`, and with `hadamard` when `stride` is not 2, 4, 8 or
/// 16.
pub fn deinterleave_hadamard(x: &mut [f32], n0: usize, stride: usize, hadamard: bool) {
    let n = n0 * stride;
    let x = &mut x[..n];
    if n == 0 {
        return;
    }
    let order = if hadamard {
        Some(hadamard_order(stride))
    } else {
        None
    };
    match stride {
        // The codec uses 2, 4 and 8 rows (one row per short block).
        2 => deinterleave_rows::<2>(x, n0, order),
        4 => deinterleave_rows::<4>(x, n0, order),
        8 => deinterleave_rows::<8>(x, n0, order),
        16 => deinterleave_rows::<16>(x, n0, order),
        _ => {
            // Any other number of rows comes without an order
            // (`hadamard_order` panics otherwise).
            let mut tmp_buf: HadamardScratch = [core::mem::MaybeUninit::uninit(); MAX_PVQ_N];
            let tmp = &mut tmp_buf[..n];
            for i in 0..stride {
                for j in 0..n0 {
                    tmp[i * n0 + j].write(x[j * stride + i]);
                }
            }
            // SAFETY: `i * n0 + j` takes every value below `n0 * stride`
            // once, so all of `tmp_buf[..n]` was written.
            x.copy_from_slice(unsafe { hadamard_written(&tmp_buf, n) });
        }
    }
}

/// The reverse of `deinterleave_hadamard`. It panics in the same cases.
pub fn interleave_hadamard(x: &mut [f32], n0: usize, stride: usize, hadamard: bool) {
    let n = n0 * stride;
    let x = &mut x[..n];
    if n == 0 {
        return;
    }
    let order = if hadamard {
        Some(hadamard_order(stride))
    } else {
        None
    };
    match stride {
        2 => interleave_rows::<2>(x, n0, order),
        4 => interleave_rows::<4>(x, n0, order),
        8 => interleave_rows::<8>(x, n0, order),
        16 => interleave_rows::<16>(x, n0, order),
        _ => {
            let mut tmp_buf: HadamardScratch = [core::mem::MaybeUninit::uninit(); MAX_PVQ_N];
            let tmp = &mut tmp_buf[..n];
            for i in 0..stride {
                for j in 0..n0 {
                    tmp[j * stride + i].write(x[i * n0 + j]);
                }
            }
            // SAFETY: `j * stride + i` takes every value below
            // `n0 * stride` once, so all of `tmp_buf[..n]` was written.
            x.copy_from_slice(unsafe { hadamard_written(&tmp_buf, n) });
        }
    }
}

const ORDERY_TABLE: [i32; 30] = [
    1, 0, 3, 0, 2, 1, 7, 0, 4, 3, 6, 1, 5, 2, 15, 0, 8, 7, 12, 3, 11, 4, 14, 1, 9, 6, 13, 2, 10, 5,
];

fn quant_band_n1(
    ctx: &mut BandCtx,
    x: &mut [f32],
    y: Option<&mut [f32]>,
    lowband_out: Option<&mut [f32]>,
) -> u32 {
    let mut sign = 0;
    if ctx.remaining_bits >= 1 << BITRES {
        if ctx.encode {
            sign = if x[0] < 0.0 { 1 } else { 0 };
            ctx.rc.enc_bits(sign as u32, 1);
        } else {
            sign = ctx.rc.dec_bits(1) as i32;
        }
        ctx.remaining_bits -= 1 << BITRES;
    }
    if ctx.resynth {
        x[0] = if sign != 0 { -1.0 } else { 1.0 };
    }
    if let Some(y_val) = y {
        let mut y_sign = 0;
        if ctx.remaining_bits >= 1 << BITRES {
            if ctx.encode {
                y_sign = if y_val[0] < 0.0 { 1 } else { 0 };
                ctx.rc.enc_bits(y_sign as u32, 1);
            } else {
                y_sign = ctx.rc.dec_bits(1) as i32;
            }
            ctx.remaining_bits -= 1 << BITRES;
        }
        if ctx.resynth {
            y_val[0] = if y_sign != 0 { -1.0 } else { 1.0 };
        }
    }
    if let Some(l_out) = lowband_out {
        // libopus: `lowband_out[0] = SHR32(X[0], 4)`. `SHR32` shifts only in
        // the fixed-point build; in the floating-point build, which this
        // port follows, it leaves the value alone. Dividing by 16 here gave
        // one-bin bands 1/16 of their weight as folding source material.
        l_out[0] = x[0];
    }
    1
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub fn quant_band(
    ctx: &mut BandCtx,
    x: &mut [f32],
    n: usize,
    b: i32,
    b_blocks: i32,
    lowband: Option<&mut [f32]>,
    lm: i32,
    lowband_out: Option<&mut [f32]>,
    gain: f32,
    fill: u32,
) -> u32 {
    let n0 = n;
    let b0 = b_blocks;
    let long_blocks = b0 == 1;

    if n == 1 {
        return quant_band_n1(ctx, x, None, lowband_out);
    }

    let mut b_blocks = b_blocks;
    let mut n_b = n / b_blocks as usize;
    let mut time_divide = 0;
    let mut recombine = 0;
    let mut tf_change_local = ctx.tf_change;
    let mut fill = fill;

    if tf_change_local > 0 {
        recombine = tf_change_local;
    }

    let mut lowband_buf = lowband;

    static BIT_INTERLEAVE_TABLE: [u8; 16] = [0, 1, 1, 1, 2, 3, 3, 3, 2, 3, 3, 3, 2, 3, 3, 3];

    for k in 0..recombine {
        if ctx.encode {
            haar1(x, n >> k, 1 << k);
        }
        if let Some(ref mut lb) = lowband_buf {
            haar1(lb, n >> k, 1 << k);
        }
        fill = (BIT_INTERLEAVE_TABLE[(fill & 0xF) as usize] as u32)
            | ((BIT_INTERLEAVE_TABLE[(fill >> 4) as usize] as u32) << 2);
    }
    b_blocks >>= recombine;
    n_b <<= recombine;

    while n_b & 1 == 0 && tf_change_local < 0 {
        if ctx.encode {
            haar1(x, n_b, b_blocks as usize);
        }
        if let Some(ref mut lb) = lowband_buf {
            haar1(lb, n_b, b_blocks as usize);
        }
        fill |= fill << b_blocks;
        b_blocks <<= 1;
        n_b >>= 1;
        time_divide += 1;
        tf_change_local += 1;
    }

    let b0_after = b_blocks;
    let n_b0 = n_b;

    if b_blocks > 1 {
        if ctx.encode {
            deinterleave_hadamard(
                x,
                n_b >> recombine as usize,
                (b_blocks << recombine) as usize,
                long_blocks,
            );
        }
        if let Some(ref mut lb) = lowband_buf {
            deinterleave_hadamard(
                lb,
                n_b >> recombine as usize,
                (b_blocks << recombine) as usize,
                long_blocks,
            );
        }
    }

    let cm = if ctx.encode && !ctx.resynth {
        quant_partition::<true>(ctx, x, n, b, b_blocks, lowband_buf, lm, gain, fill)
    } else {
        quant_partition::<false>(ctx, x, n, b, b_blocks, lowband_buf, lm, gain, fill)
    };

    if ctx.resynth {
        let mut cm = cm;

        if b_blocks > 1 {
            interleave_hadamard(
                x,
                n_b >> recombine as usize,
                (b0_after << recombine) as usize,
                long_blocks,
            );
        }

        let mut n_b_undo = n_b0;
        let mut b_undo = b0_after;
        for _ in 0..time_divide {
            b_undo >>= 1;
            n_b_undo <<= 1;
            cm |= cm >> b_undo;
            haar1(x, n_b_undo, b_undo as usize);
        }

        static BIT_DEINTERLEAVE_TABLE: [u8; 16] = [
            0x00, 0x03, 0x0C, 0x0F, 0x30, 0x33, 0x3C, 0x3F, 0xC0, 0xC3, 0xCC, 0xCF, 0xF0, 0xF3,
            0xFC, 0xFF,
        ];
        for k in 0..recombine {
            cm = BIT_DEINTERLEAVE_TABLE[cm as usize & 0xF] as u32;
            haar1(x, n0 >> k, 1 << k);
        }
        let mut b_final = b_undo;
        b_final <<= recombine;

        if let Some(lb_out) = lowband_out {
            let scale = (n0 as f32).sqrt();
            for (out, &xv) in lb_out[..n0].iter_mut().zip(&x[..n0]) {
                *out = scale * xv;
            }
        }
        cm &= (1u32 << b_final) - 1;
        return cm;
    }

    cm
}

/// Turns the mid signal in `x` and the side signal in `y` back into the left
/// and the right channel, each with unit energy: libopus `stereo_merge`
/// (celt/bands.c, float build).
///
/// The two sums are `celt_inner_prod` (on arm64 the NEON form). clang fuses
/// `mid * mid + side` (`MULT32_32_Q31(mid, mid) + side`, the shift is none in
/// the float build); `2 * xp` is exact, so the second step rounds alike
/// either way.
pub fn stereo_merge(x: &mut [f32], y: &mut [f32], mid: f32, n: usize) {
    // One bounds check per slice; the loops below need none.
    let x = &mut x[..n];
    let y = &mut y[..n];

    // C: `xp = celt_inner_prod_norm_shift(Y, X, N, arch)` and
    // `side = celt_inner_prod_norm_shift(Y, Y, N, arch)`.
    let xp = mid * celt_inner_prod(y, x);
    let side_e = celt_inner_prod(y, y);
    let t = mac(side_e, mid, mid);
    let el = t - 2.0 * xp;
    let er = t + 2.0 * xp;

    if er < 6e-4f32 || el < 6e-4f32 {
        y.copy_from_slice(x);
        return;
    }

    let lgain = 1.0 / el.sqrt();
    let rgain = 1.0 / er.sqrt();

    for (xv, yv) in x.iter_mut().zip(y.iter_mut()) {
        let l = mid * *xv;
        let r = *yv;
        *xv = lgain * (l - r);
        *yv = rgain * (l + r);
    }
}

#[inline(always)]
fn stereo_split(x: &mut [f32], y: &mut [f32], n: usize) {
    let scale = core::f32::consts::FRAC_1_SQRT_2;
    for (xv, yv) in x[..n].iter_mut().zip(y[..n].iter_mut()) {
        let l = scale * *xv;
        let r = scale * *yv;
        *xv = l + r;
        *yv = r - l;
    }
}

/// libopus `intensity_stereo` (celt/bands.c, lines 379-403, float build):
/// `x` becomes the mix of both channels that the band codes, weighted by
/// the band energies of the two channels. The side is not coded.
///
/// `EPSILON` (1e-15) is added under the square root and to the root, as
/// libopus does. It matters in bands whose energies are near 1e-7 or
/// below, such as the decaying output of the DC filter after a signal
/// stops. clang fuses the sum of squares into two fused multiply-adds, and
/// the mix into one product and one fused multiply-add, on 64-bit ARM.
/// `mac` rounds the same way there.
#[inline(always)]
fn intensity_stereo(
    m: &CeltMode,
    x: &mut [f32],
    y: &mut [f32],
    band_e: &[f32],
    band: usize,
    n: usize,
) {
    const EPSILON: f32 = 1e-15;
    let left = band_e[band];
    let right = band_e[m.nb_ebands + band];
    let norm = EPSILON + mac(mac(EPSILON, left, left), right, right).sqrt();
    let a1 = left / norm;
    let a2 = right / norm;
    for (xv, &yv) in x[..n].iter_mut().zip(&y[..n]) {
        *xv = mac(a2 * yv, a1, *xv);
    }
}

#[inline(always)]
fn special_hybrid_folding(m: &CeltMode, norm: &mut [f32], start: usize, m_val: usize) {
    if start + 2 >= m.e_bands.len() {
        return;
    }
    let n1 = m_val * (m.e_bands[start + 1] - m.e_bands[start]) as usize;
    let n2 = m_val * (m.e_bands[start + 2] - m.e_bands[start + 1]) as usize;
    if n2 <= n1 {
        return;
    }
    let len = n2 - n1;
    let src_start = 2 * n1 - n2;
    if src_start + len <= norm.len() && n1 + len <= norm.len() {
        norm.copy_within(src_start..src_start + len, n1);
    }
}

/// For the band of `n` values at `norm_pos`: the lowband it folds from,
/// which starts at `effective_lowband` (no lowband when that is negative),
/// and, with `want_out`, the part of `norm` the band writes. With
/// `allow_lowband_scratch` the lowband is a copy in `lowband_scratch`, so
/// it may overlap the part the band writes.
fn prepare_lowband_views<'a>(
    norm: &'a mut [f32],
    lowband_scratch: &'a mut [MaybeUninit<f32>],
    allow_lowband_scratch: bool,
    effective_lowband: i32,
    norm_pos: usize,
    n: usize,
    want_out: bool,
) -> (Option<&'a mut [f32]>, Option<&'a mut [f32]>) {
    let len = norm.len();
    let out_range = if want_out && norm_pos + n <= len {
        Some((norm_pos, norm_pos + n))
    } else {
        None
    };

    let Some(lb_start) = (if effective_lowband >= 0 {
        Some(effective_lowband as usize)
    } else {
        None
    }) else {
        let lb_out = out_range.map(|(s, e)| &mut norm[s..e]);
        return (None, lb_out);
    };
    let lb_end = lb_start + n;
    if lb_end > len {
        let lb_out = out_range.map(|(s, e)| &mut norm[s..e]);
        return (None, lb_out);
    }

    if allow_lowband_scratch {
        // The scratch buffer of quant_all_bands holds MAX_PVQ_N values; a
        // longer band panics here.
        let lb = lowband_scratch[..n].write_copy_of_slice(&norm[lb_start..lb_end]);
        let lb_out = out_range.map(|(s, e)| &mut norm[s..e]);
        return (Some(lb), lb_out);
    }

    if let Some((out_start, out_end)) = out_range {
        if lb_end <= out_start {
            let (left, right) = norm.split_at_mut(out_start);
            let lb = Some(&mut left[lb_start..lb_end]);
            let lb_out = Some(&mut right[..(out_end - out_start)]);
            return (lb, lb_out);
        }
        if out_end <= lb_start {
            let (left, right) = norm.split_at_mut(lb_start);
            let lb_out = Some(&mut left[out_start..out_end]);
            let lb = Some(&mut right[..n]);
            return (lb, lb_out);
        }
        return (Some(&mut norm[lb_start..lb_end]), None);
    }

    (Some(&mut norm[lb_start..lb_end]), None)
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub fn quant_band_stereo(
    ctx: &mut BandCtx,
    x: &mut [f32],
    y: &mut [f32],
    n: usize,
    b: i32,
    b_blocks: i32,
    lowband: Option<&mut [f32]>,
    lm: i32,
    lowband_out: Option<&mut [f32]>,
    fill: u32,
) -> u32 {
    if n == 1 {
        return quant_band_n1(ctx, x, Some(y), lowband_out);
    }

    if ctx.encode
        && (ctx.band_e[ctx.i] < MIN_STEREO_ENERGY
            || ctx.band_e[ctx.m.nb_ebands + ctx.i] < MIN_STEREO_ENERGY)
    {
        if ctx.band_e[ctx.i] > ctx.band_e[ctx.m.nb_ebands + ctx.i] {
            y.copy_from_slice(x);
        } else {
            x.copy_from_slice(y);
        }
    }

    let mut sctx = SplitCtx {
        inv: false,
        imid: 0,
        iside: 0,
        delta: 0,
        itheta: 0,
        qalloc: 0,
    };
    let mut b_mut = b;
    let mut fill_mut = fill;
    compute_theta(
        ctx,
        &mut sctx,
        x,
        y,
        n,
        &mut b_mut,
        b_blocks,
        b_blocks,
        lm,
        true,
        &mut fill_mut,
    );

    let mid_gain = sctx.imid as f32 / 32768.0;
    let side_gain = sctx.iside as f32 / 32768.0;

    if n == 2 {
        let orig_fill = fill;
        let mut mbits = b_mut;
        let mut sbits = 0;
        if sctx.itheta != 0 && sctx.itheta != 16384 {
            sbits = 1 << BITRES;
        }
        mbits -= sbits;
        let c = sctx.itheta > 8192;
        ctx.remaining_bits -= sctx.qalloc + sbits;

        let mut sign = 0;
        if sbits != 0 {
            if ctx.encode {
                sign = if c {
                    if (y[0] * x[1] - y[1] * x[0]) < 0.0 {
                        1
                    } else {
                        0
                    }
                } else if (x[0] * y[1] - x[1] * y[0]) < 0.0 {
                    1
                } else {
                    0
                };
                ctx.rc.enc_bits(sign as u32, 1);
            } else {
                sign = ctx.rc.dec_bits(1) as i32;
            }
        }
        let sign_val = (1 - 2 * sign) as f32;
        let cm = if c {
            let cm = quant_band(
                ctx,
                y,
                n,
                mbits,
                b_blocks,
                lowband,
                lm,
                lowband_out,
                1.0,
                orig_fill,
            );
            x[0] = -sign_val * y[1];
            x[1] = sign_val * y[0];
            cm
        } else {
            let cm = quant_band(
                ctx,
                x,
                n,
                mbits,
                b_blocks,
                lowband,
                lm,
                lowband_out,
                1.0,
                orig_fill,
            );
            y[0] = -sign_val * x[1];
            y[1] = sign_val * x[0];
            cm
        };

        if ctx.resynth {
            let x0 = x[0];
            let x1 = x[1];
            let y0 = y[0];
            let y1 = y[1];
            let mx0 = mid_gain * x0;
            let mx1 = mid_gain * x1;
            let sy0 = side_gain * y0;
            let sy1 = side_gain * y1;
            x[0] = mx0 - sy0;
            x[1] = mx1 - sy1;
            y[0] = mx0 + sy0;
            y[1] = mx1 + sy1;
            // libopus applies the phase-inversion flag to two-bin bands
            // too (the `if (inv)` at the end of quant_band_stereo is
            // outside the `N != 2` test). Returning before it left the
            // right channel of such a band with the wrong sign.
            if sctx.inv {
                y[0] = -y[0];
                y[1] = -y[1];
            }
        }
        return cm;
    }

    ctx.remaining_bits -= sctx.qalloc;
    let mut mbits = (0).max((b_mut - sctx.delta) / 2).min(b_mut);
    let mut sbits = b_mut - mbits;

    let mut rebalance = ctx.remaining_bits;
    let mut cm;

    if mbits >= sbits {
        cm = quant_band(
            ctx,
            x,
            n,
            mbits,
            b_blocks,
            lowband,
            lm,
            lowband_out,
            1.0,
            fill_mut,
        );
        rebalance = mbits - (rebalance - ctx.remaining_bits);
        if rebalance > (3 << 3) && sctx.itheta != 0 {
            sbits += rebalance - (3 << 3);
        }
        cm |= quant_band(
            ctx,
            y,
            n,
            sbits,
            b_blocks,
            None,
            lm,
            None,
            side_gain,
            fill_mut >> b_blocks,
        );
    } else {
        cm = quant_band(
            ctx,
            y,
            n,
            sbits,
            b_blocks,
            None,
            lm,
            None,
            side_gain,
            fill_mut >> b_blocks,
        );
        rebalance = sbits - (rebalance - ctx.remaining_bits);
        if rebalance > (3 << 3) && sctx.itheta != 16384 {
            mbits += rebalance - (3 << 3);
        }
        cm |= quant_band(
            ctx,
            x,
            n,
            mbits,
            b_blocks,
            lowband,
            lm,
            lowband_out,
            1.0,
            fill_mut,
        );
    }

    if ctx.resynth {
        stereo_merge(x, y, mid_gain, n);
        if sctx.inv {
            for yv in y[..n].iter_mut() {
                *yv = -*yv;
            }
        }
    }
    cm
}

/// libopus `compute_channel_weights` (celt/bands.c 362-377, float build):
/// the weights of the two channels in the distortion of theta_rdo.
#[inline(always)]
fn compute_channel_weights(ex: f32, ey: f32) -> [f32; 2] {
    // C: `minE = MIN32(Ex, Ey)`, then `Ex = ADD32(Ex, minE/3)`.
    let min_e = if ex < ey { ex } else { ey };
    [ex + min_e / 3.0, ey + min_e / 3.0]
}

/// One stereo band coded twice, with the angle rounded down and up, keeping
/// the result with the smaller distortion: the `theta_rdo` branch of libopus
/// `quant_all_bands` (celt/bands.c 1809-1880).
///
/// The first pass codes the band in place with `theta_round = -1`. Its coder
/// state, its bytes and its folding output in `norm` are kept. The second
/// pass starts from the same coder state with `theta_round = 1`, on a copy
/// of the input. It takes the angle that the first pass computed
/// (`BandCtx::stereo_angle`). When the first pass has the smaller or equal
/// distortion, its coder state, bytes and folding output are put back.
/// Otherwise the band of the second pass is copied over the first one.
///
/// libopus runs both passes in place: it copies the band of the first pass
/// aside and copies it back when that pass wins. The order here gives the
/// same results with fewer copies.
///
/// libopus copies the coder bytes from the start offset to the end of the
/// buffer. A pass writes only bytes from `offs` up and raw bits from the
/// end down, so this copies just those two ranges; the bytes in between are
/// written again later or cleared by `RangeCoder::done`.
///
/// `hybrid_fold` is `Some((start, m_val))` for the band after the first one:
/// libopus folds its first band into `norm` again before the second pass
/// (`special_hybrid_folding`), because the first pass overwrote it.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn quant_band_stereo_rdo(
    ctx: &mut BandCtx,
    x: &mut [f32],
    y: &mut [f32],
    n: usize,
    b: i32,
    b_blocks: i32,
    lm: i32,
    fill: u32,
    norm: &mut [f32],
    lowband_scratch: &mut [MaybeUninit<f32>],
    allow_lowband_scratch: bool,
    effective_lowband: i32,
    norm_pos: usize,
    last: bool,
    hybrid_fold: Option<(usize, usize)>,
) -> u32 {
    // A band never has more than MAX_PVQ_N values; the copies below go to
    // buffers of that size.
    assert!(n <= MAX_PVQ_N, "band of {n} values exceeds MAX_PVQ_N");
    let x = &mut x[..n];
    let y = &mut y[..n];
    let mut x_save_mu = [MaybeUninit::<f32>::uninit(); MAX_PVQ_N];
    let mut y_save_mu = [MaybeUninit::<f32>::uninit(); MAX_PVQ_N];
    let x_save: &[f32] = x_save_mu[..n].write_copy_of_slice(x);
    let y_save: &[f32] = y_save_mu[..n].write_copy_of_slice(y);

    let w = compute_channel_weights(ctx.band_e[ctx.i], ctx.band_e[ctx.i + ctx.m.nb_ebands]);
    // C: `MULT16_32_Q15(w[0], ip(X_save, X)) + MULT16_32_Q15(w[1], ip(Y_save,
    // Y))`. clang fuses the first product into the add (fmadd).
    let distortion = |x: &[f32], y: &[f32], x_save: &[f32], y_save: &[f32]| {
        mac(w[1] * celt_inner_prod(y_save, y), w[0], celt_inner_prod(x_save, x))
    };

    let state0 = ctx.rc.save_state();
    let (offs0, end_offs0) = (ctx.rc.offs as usize, ctx.rc.end_offs as usize);
    let (seed0, remaining0) = (ctx.seed, ctx.remaining_bits);

    // Encode and round down, in place.
    ctx.theta_round = -1;
    let (lb, lb_out) = prepare_lowband_views(
        norm,
        lowband_scratch,
        allow_lowband_scratch,
        effective_lowband,
        norm_pos,
        n,
        !last,
    );
    let cm_down = quant_band_stereo(ctx, x, y, n, b, b_blocks, lb, lm, lb_out, fill);
    let dist_down = distortion(x, y, x_save, y_save);

    // Keep what the first pass wrote outside the band.
    let state_down = ctx.rc.save_state();
    let (seed_down, remaining_down) = (ctx.seed, ctx.remaining_bits);
    let mut norm_down_mu = [MaybeUninit::<f32>::uninit(); MAX_PVQ_N];
    let norm_down: Option<&[f32]> = if !last && norm_pos + n <= norm.len() {
        Some(norm_down_mu[..n].write_copy_of_slice(&norm[norm_pos..norm_pos + n]))
    } else {
        None
    };
    // The bytes the first pass wrote: range coder bytes from `offs0` up, raw
    // bits from the end down. Both ranges lie inside the coder's buffer of
    // `storage` bytes and do not overlap, so together they hold at most
    // RANGE_BUF_MAX bytes.
    let storage = ctx.rc.storage as usize;
    let front = offs0..(ctx.rc.offs as usize).max(offs0);
    let back = storage - (ctx.rc.end_offs as usize).max(end_offs0)..storage - end_offs0;
    let mut bytes_mu = [MaybeUninit::<u8>::uninit(); crate::range_coder::RANGE_BUF_MAX];
    let (front_mu, back_mu) = bytes_mu[..front.len() + back.len()].split_at_mut(front.len());
    let front_bytes: &[u8] = front_mu.write_copy_of_slice(&ctx.rc.buf[front.clone()]);
    let back_bytes: &[u8] = back_mu.write_copy_of_slice(&ctx.rc.buf[back.clone()]);

    // Back to the coder state before the first pass.
    ctx.rc.restore_state(&state0);
    ctx.seed = seed0;
    ctx.remaining_bits = remaining0;
    if let Some((start, m_val)) = hybrid_fold {
        special_hybrid_folding(ctx.m, norm, start, m_val);
    }

    // Encode and round up, on a copy of the input.
    let mut x_up_mu = [MaybeUninit::<f32>::uninit(); MAX_PVQ_N];
    let mut y_up_mu = [MaybeUninit::<f32>::uninit(); MAX_PVQ_N];
    let x_up = x_up_mu[..n].write_copy_of_slice(x_save);
    let y_up = y_up_mu[..n].write_copy_of_slice(y_save);
    ctx.theta_round = 1;
    let (lb, lb_out) = prepare_lowband_views(
        norm,
        lowband_scratch,
        allow_lowband_scratch,
        effective_lowband,
        norm_pos,
        n,
        !last,
    );
    let cm_up = quant_band_stereo(ctx, x_up, y_up, n, b, b_blocks, lb, lm, lb_out, fill);
    let dist_up = distortion(x_up, y_up, x_save, y_save);

    if dist_down >= dist_up {
        // The first pass wins. Its band is still in `x` and `y`.
        ctx.rc.restore_state(&state_down);
        ctx.rc.buf[front].copy_from_slice(front_bytes);
        ctx.rc.buf[back].copy_from_slice(back_bytes);
        ctx.seed = seed_down;
        ctx.remaining_bits = remaining_down;
        ctx.theta_round = -1;
        if let Some(band) = norm_down {
            norm[norm_pos..norm_pos + n].copy_from_slice(band);
        }
        cm_down
    } else {
        x.copy_from_slice(x_up);
        y.copy_from_slice(y_up);
        cm_up
    }
}

/// The encoder's call of `quant_all_bands` as libopus makes it
/// (celt/celt_encoder.c 2674-2679), with the encoder's complexity and
/// phase-inversion setting. `seed` is the encoder's `rng`: the final range
/// of its previous frame, or 0 after a reset.
///
/// At complexity 8 and above a stereo frame that is not dual stereo is coded
/// with `theta_rdo`: the encoder resynthesises every band, and codes each
/// band below the intensity band twice (`quant_band_stereo_rdo`).
#[allow(clippy::too_many_arguments)]
pub fn quant_all_bands_enc(
    m: &CeltMode,
    start: usize,
    end: usize,
    x: &mut [f32],
    y: Option<&mut [f32]>,
    collapse_masks: &mut [u32],
    band_e: &[f32],
    pulses: &[i32],
    short_blocks: bool,
    spread: i32,
    dual_stereo: &mut bool,
    intensity: usize,
    tf_res: &[i32],
    total_bits: i32,
    balance: &mut i32,
    rc: &mut RangeCoder,
    lm: i32,
    coded_bands: i32,
    seed: &mut u32,
    complexity: i32,
    disable_inv: bool,
) {
    quant_all_bands_enc_with(
        m,
        start,
        end,
        x,
        y,
        collapse_masks,
        band_e,
        pulses,
        short_blocks,
        spread,
        dual_stereo,
        intensity,
        tf_res,
        total_bits,
        balance,
        rc,
        lm,
        coded_bands,
        seed,
        complexity,
        disable_inv,
        &mut RotationCache::new(),
    );
}

/// `quant_all_bands_enc` with the encoder's cache of rotation angles. The
/// cache only saves time: the results are those of `quant_all_bands_enc`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn quant_all_bands_enc_with(
    m: &CeltMode,
    start: usize,
    end: usize,
    x: &mut [f32],
    y: Option<&mut [f32]>,
    collapse_masks: &mut [u32],
    band_e: &[f32],
    pulses: &[i32],
    short_blocks: bool,
    spread: i32,
    dual_stereo: &mut bool,
    intensity: usize,
    tf_res: &[i32],
    total_bits: i32,
    balance: &mut i32,
    rc: &mut RangeCoder,
    lm: i32,
    coded_bands: i32,
    seed: &mut u32,
    complexity: i32,
    disable_inv: bool,
    rotations: &mut RotationCache,
) {
    quant_all_bands_impl(
        true,
        m,
        start,
        end,
        x,
        y,
        collapse_masks,
        band_e,
        pulses,
        short_blocks,
        spread,
        dual_stereo,
        intensity,
        tf_res,
        total_bits,
        balance,
        rc,
        lm,
        coded_bands,
        false,
        disable_inv,
        seed,
        complexity,
        Some(rotations),
    );
}

/// libopus `quant_all_bands` without the encoder's complexity: no
/// `theta_rdo`. The decoder calls it with `resynth` set.
#[allow(clippy::too_many_arguments)]
pub fn quant_all_bands(
    encode: bool,
    m: &CeltMode,
    start: usize,
    end: usize,
    x: &mut [f32],
    y: Option<&mut [f32]>,
    collapse_masks: &mut [u32],
    band_e: &[f32],
    pulses: &[i32],
    short_blocks: bool,
    spread: i32,
    dual_stereo: &mut bool,
    intensity: usize,
    tf_res: &[i32],
    total_bits: i32,
    balance: &mut i32,
    rc: &mut RangeCoder,
    lm: i32,
    coded_bands: i32,
    resynth: bool,
    disable_inv: bool,
    seed: &mut u32,
) {
    quant_all_bands_impl(
        encode,
        m,
        start,
        end,
        x,
        y,
        collapse_masks,
        band_e,
        pulses,
        short_blocks,
        spread,
        dual_stereo,
        intensity,
        tf_res,
        total_bits,
        balance,
        rc,
        lm,
        coded_bands,
        resynth,
        disable_inv,
        seed,
        0,
        None,
    );
}

/// libopus `quant_all_bands` (celt/bands.c 1589-1925).
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn quant_all_bands_impl(
    encode: bool,
    m: &CeltMode,
    start: usize,
    end: usize,
    x: &mut [f32],
    mut y: Option<&mut [f32]>,
    collapse_masks: &mut [u32],
    band_e: &[f32],
    pulses: &[i32],
    short_blocks: bool,
    spread: i32,
    dual_stereo: &mut bool,
    intensity: usize,
    tf_res: &[i32],
    total_bits: i32,
    balance: &mut i32,
    rc: &mut RangeCoder,
    lm: i32,
    coded_bands: i32,
    resynth: bool,
    disable_inv: bool,
    seed: &mut u32,
    complexity: i32,
    mut rotations: Option<&mut RotationCache>,
) {
    // C 1618-1623: at complexity 8 and above the encoder codes each stereo
    // band twice and resynthesises every band for it.
    let theta_rdo = encode && y.is_some() && !*dual_stereo && complexity >= 8;
    let resynth = resynth || theta_rdo;
    let mut balance_val = *balance;
    let b_blocks = if short_blocks { 1 << lm } else { 1 };
    let c_channels = if y.is_some() { 2 } else { 1 };
    let m_val = 1usize << lm as usize;

    let norm_offset = m_val * (m.e_bands[start] as usize);
    let norm_size = m_val * (m.e_bands[m.nb_ebands - 1] as usize) - norm_offset;

    const MAX_NORM_SIZE: usize = 800;
    // Checked in release builds too: the two slices below must fit in their
    // buffers. The 48 kHz mode needs 624 values at most.
    assert!(norm_size <= MAX_NORM_SIZE, "norm buffer too small for this mode");

    // `norm` and `norm2` hold the decoded bands of the frame so far, for
    // folding. A band is written (as `lowband_out`) before a later band
    // reads it as its `lowband`: the read range ends at or before the start
    // of the current band, or within the values that
    // `special_hybrid_folding` copied there. The values are set to 0 first
    // only because Rust needs memory to hold values before it can be a
    // `&mut [f32]`; a fill once per frame costs less than tracking which
    // values the bands wrote. Without resynthesis no band folds, so `norm`
    // stays empty. `norm2` stays empty unless the frame starts in dual
    // stereo, the only case that uses it.
    let mut norm_buf = [MaybeUninit::<f32>::uninit(); MAX_NORM_SIZE];
    let norm = write_filled(&mut norm_buf[..if resynth { norm_size } else { 0 }], 0.0);
    let mut norm2_buf = [MaybeUninit::<f32>::uninit(); MAX_NORM_SIZE];
    let norm2_size = if resynth && *dual_stereo { norm_size } else { 0 };
    let norm2 = write_filled(&mut norm2_buf[..norm2_size], 0.0);

    let mut lowband_scratch = [MaybeUninit::<f32>::uninit(); MAX_PVQ_N];

    let mut lowband_offset: usize = 0;
    let mut update_lowband = true;
    let mut avoid_split_noise = b_blocks > 1;

    let e_bands = &m.e_bands;
    let mut ctx_seed = *seed;

    for i in start..end {
        let e_band_i = e_bands[i] as usize;
        let e_band_i1 = e_bands[i + 1] as usize;
        let offset = m_val * e_band_i;
        let n = m_val * (e_band_i1 - e_band_i);
        let last = i == end - 1;

        let tell = tell_frac_inline!(rc);
        if i != start {
            balance_val -= tell;
        }
        let remaining_bits = total_bits - tell - 1;

        let mut b = 0i32;
        if i < coded_bands as usize {
            let curr_balance = celt_sudiv(balance_val, 3i32.min(coded_bands - i as i32));
            b = 0i32.max(16383i32.min((remaining_bits + 1).min(pulses[i] + curr_balance)));
        }

        let norm_pos = m_val * e_band_i - norm_offset;
        let tf_change = tf_res[i];

        let mut effective_lowband: i32 = -1;
        let mut x_cm: u32;
        let mut y_cm: u32;

        let band_start_abs = m_val * e_band_i;
        let start_abs = m_val * (e_bands[start] as usize);
        if resynth
            && ((band_start_abs as isize - n as isize >= start_abs as isize) || i == start + 1)
            && (update_lowband || lowband_offset == 0)
        {
            lowband_offset = i;
        }

        if resynth && i == start + 1 {
            special_hybrid_folding(m, norm, start, m_val);
            if *dual_stereo {
                special_hybrid_folding(m, norm2, start, m_val);
            }
        }

        if lowband_offset != 0 && (spread != SPREAD_AGGRESSIVE || b_blocks > 1 || tf_change < 0) {
            effective_lowband = 0i32.max(
                (m_val * e_bands[lowband_offset] as usize) as i32 - norm_offset as i32 - n as i32,
            );
            let el_abs = effective_lowband as usize + norm_offset;

            let mut fold_start = lowband_offset;
            while fold_start > 0 {
                fold_start -= 1;
                if m_val * (e_bands[fold_start] as usize) <= el_abs {
                    break;
                }
            }

            let mut fold_end = lowband_offset.saturating_sub(1);
            loop {
                fold_end += 1;
                if fold_end >= i || m_val * (e_bands[fold_end] as usize) >= el_abs + n {
                    break;
                }
            }

            x_cm = 0;
            y_cm = 0;
            let mut fi = fold_start;
            loop {
                x_cm |= collapse_masks[fi * c_channels];
                y_cm |= collapse_masks[fi * c_channels + c_channels - 1];
                fi += 1;
                if fi >= fold_end {
                    break;
                }
            }
        } else {
            x_cm = (1u32 << b_blocks) - 1;
            y_cm = (1u32 << b_blocks) - 1;
        }

        let mut ctx = BandCtx {
            encode,
            m,
            i,
            band_e,
            rc,
            spread,
            remaining_bits,
            resynth,
            tf_change,
            intensity,
            theta_round: 0,
            avoid_split_noise,
            disable_inv,
            seed: ctx_seed,
            stereo_angle: None,
            rotations: rotations.as_deref_mut(),
        };

        let x_slice = &mut x[offset..offset + n];
        let band_uses_direct_norm = i >= m.eff_ebands;
        // C 1749-1758: no scratch for bands past `effEBands`, nor for the last
        // band unless theta_rdo codes it twice.
        let allow_lowband_scratch = !(band_uses_direct_norm || (last && !theta_rdo));
        if *dual_stereo && i == intensity {
            *dual_stereo = false;
            if resynth {
                for j in 0..norm_pos {
                    norm[j] = 0.5 * (norm[j] + norm2[j]);
                }
            }
        }

        if *dual_stereo {
            let y_slice = &mut y.as_mut().unwrap()[offset..offset + n];

            let (lb_x, lb_out_x) = prepare_lowband_views(
                norm,
                &mut lowband_scratch,
                allow_lowband_scratch,
                effective_lowband,
                norm_pos,
                n,
                !last,
            );
            x_cm = quant_band(
                &mut ctx,
                x_slice,
                n,
                b / 2,
                b_blocks,
                lb_x,
                lm,
                lb_out_x,
                1.0,
                x_cm,
            );

            let (lb_y, lb_out_y) = prepare_lowband_views(
                norm2,
                &mut lowband_scratch,
                allow_lowband_scratch,
                effective_lowband,
                norm_pos,
                n,
                !last,
            );
            y_cm = quant_band(
                &mut ctx,
                y_slice,
                n,
                b / 2,
                b_blocks,
                lb_y,
                lm,
                lb_out_y,
                1.0,
                y_cm,
            );
        } else if let Some(y_all) = y.as_mut() {
            let y_slice = &mut y_all[offset..offset + n];
            if theta_rdo && i < intensity {
                x_cm = quant_band_stereo_rdo(
                    &mut ctx,
                    x_slice,
                    y_slice,
                    n,
                    b,
                    b_blocks,
                    lm,
                    x_cm | y_cm,
                    norm,
                    &mut lowband_scratch,
                    allow_lowband_scratch,
                    effective_lowband,
                    norm_pos,
                    last,
                    if i == start + 1 { Some((start, m_val)) } else { None },
                );
            } else {
                let (lb, lb_out) = prepare_lowband_views(
                    norm,
                    &mut lowband_scratch,
                    allow_lowband_scratch,
                    effective_lowband,
                    norm_pos,
                    n,
                    !last,
                );
                x_cm = quant_band_stereo(
                    &mut ctx,
                    x_slice,
                    y_slice,
                    n,
                    b,
                    b_blocks,
                    lb,
                    lm,
                    lb_out,
                    x_cm | y_cm,
                );
            }
            y_cm = x_cm;
        } else {
            let (lb, lb_out) = prepare_lowband_views(
                norm,
                &mut lowband_scratch,
                allow_lowband_scratch,
                effective_lowband,
                norm_pos,
                n,
                !last,
            );
            x_cm = quant_band(&mut ctx, x_slice, n, b, b_blocks, lb, lm, lb_out, 1.0, x_cm);
            y_cm = x_cm;
        }

        collapse_masks[i * c_channels] = (x_cm & 0xFF) as u8 as u32;
        if c_channels == 2 {
            collapse_masks[i * c_channels + 1] = (y_cm & 0xFF) as u8 as u32;
        }

        balance_val += pulses[i] + tell;
        ctx_seed = ctx.seed;
        update_lowband = b > ((n as i32) << BITRES);

        avoid_split_noise = false;
    }
    *balance = balance_val;
    *seed = ctx_seed;
}

/// The amplitude of one band, `sqrt(1e-27 + sum of x^2)`, with AVX2 and
/// FMA: sixteen values per pass in two sums of eight lanes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
fn compute_band_energy_avx2(band: &[f32]) -> f32 {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;
    use crate::simd_mem::x86::load_f32x8;

    let (octets, tail) = band.as_chunks::<8>();
    let (pairs, last) = octets.as_chunks::<2>();

    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();

    for [a, b] in pairs {
        let v0 = load_f32x8(a);
        let v1 = load_f32x8(b);
        acc0 = _mm256_fmadd_ps(v0, v0, acc0);
        acc1 = _mm256_fmadd_ps(v1, v1, acc1);
    }

    // At most one group of eight is left.
    if let [a] = last {
        let v0 = load_f32x8(a);
        acc0 = _mm256_fmadd_ps(v0, v0, acc0);
    }

    let acc = _mm256_add_ps(acc0, acc1);
    let hi = _mm256_extractf128_ps(acc, 1);
    let lo = _mm256_castps256_ps128(acc);
    let s4 = _mm_add_ps(lo, hi);
    let t1 = _mm_movehl_ps(s4, s4);
    let s2 = _mm_add_ps(s4, t1);
    let t2 = _mm_shuffle_ps(s2, s2, 0x55);
    let mut sum = 1e-27f32 + _mm_cvtss_f32(_mm_add_ss(s2, t2));

    for &v in tail {
        sum += v * v;
    }

    sum.sqrt()
}

/// libopus `compute_band_energies` (celt/bands.c 151-166, float build): the
/// amplitude of each band, `sqrt(1e-27 + sum of x^2)`.
///
/// The sum is `celt_inner_prod` of the band with itself, in the order of
/// libopus on this processor (see `celt_inner_prod`); the offset is added
/// after it. On x86-64 with AVX2 the bands of 8 values and more use a wider
/// kernel of their own.
pub fn compute_band_energies(
    m: &CeltMode,
    x: &[f32],
    band_e: &mut [f32],
    end: usize,
    channels: usize,
    lm: usize,
) {
    let frame_size = m.short_mdct_size << lm;

    #[cfg(target_arch = "x86_64")]
    let use_avx2 = crate::compat::x86_has_avx2_fma();

    for c in 0..channels {
        let ch = &x[c * frame_size..(c + 1) * frame_size];
        for i in 0..end {
            let offset = (m.e_bands[i] as usize) << lm;
            let n = ((m.e_bands[i + 1] - m.e_bands[i]) as usize) << lm;
            let band = &ch[offset..offset + n];

            #[cfg(target_arch = "x86_64")]
            if n >= 8 && use_avx2 {
                // SAFETY: the CPU has AVX2 and FMA (`use_avx2`).
                band_e[c * m.nb_ebands + i] = unsafe { compute_band_energy_avx2(band) };
                continue;
            }
            // C: `sum = 1e-27f + celt_inner_prod(X, X, N, arch)`.
            band_e[c * m.nb_ebands + i] = (1e-27f32 + celt_inner_prod(band, band)).sqrt();
        }
    }
}

pub fn amp2log2(
    m: &CeltMode,
    start: usize,
    end: usize,
    band_e: &[f32],
    band_log_e: &mut [f32],
    channels: usize,
) {
    for c in 0..channels {
        for i in 0..start {
            band_log_e[c * m.nb_ebands + i] = -14.0;
        }
        for i in start..end {
            let val = band_e[c * m.nb_ebands + i].max(1e-10);
            band_log_e[c * m.nb_ebands + i] = val.log2() - m.e_means[i];
        }
    }
}

pub fn log2amp(m: &CeltMode, end: usize, band_e: &mut [f32], band_log_e: &[f32], channels: usize) {
    for c in 0..channels {
        for i in 0..end {
            band_e[c * m.nb_ebands + i] = band_log_e[c * m.nb_ebands + i] + m.e_means[i];
        }
    }
}

pub fn normalise_bands(
    m: &CeltMode,
    freq: &[f32],
    x: &mut [f32],
    band_e: &[f32],
    end: usize,
    channels: usize,
    m_val: usize,
) {
    let lm = m_val.trailing_zeros() as usize;
    let frame_size = m.short_mdct_size << lm;
    #[cfg(target_arch = "x86_64")]
    let use_avx2 = crate::compat::x86_has_avx2();
    for c in 0..channels {
        for i in 0..end {
            let base = c * frame_size + ((m.e_bands[i] as usize) << lm);
            let n = ((m.e_bands[i + 1] - m.e_bands[i]) as usize) << lm;
            let norm = 1.0 / (1e-27 + band_e[c * m.nb_ebands + i]);
            let src = &freq[base..base + n];
            let dst = &mut x[base..base + n];
            #[cfg(target_arch = "x86_64")]
            if n >= 8 && use_avx2 {
                // SAFETY: the CPU has AVX2, the feature of the kernel.
                unsafe { scale_slice_avx2(src, dst, norm, n) };
                continue;
            }
            #[cfg(target_arch = "aarch64")]
            if n >= 8 {
                scale_slice_neon(src, dst, norm, n);
                continue;
            }
            for (d, &s) in dst.iter_mut().zip(src) {
                *d = s * norm;
            }
        }
    }
}

/// `dst[i] = src[i] * scale` for `i < n`, with AVX2.
///
/// Panics when `src` or `dst` holds fewer than `n` values.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
fn scale_slice_avx2(src: &[f32], dst: &mut [f32], scale: f32, n: usize) {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;
    // The loads and stores below cover `src[..n]` and `dst[..n]`.
    assert!(src.len() >= n && dst.len() >= n);
    let vscale = _mm256_set1_ps(scale);
    let mut i = 0;

    // The loops keep the pointer loads and stores of the original code:
    // with the groups of eight taken from `as_chunks`, the compiler counted
    // the steps with an extra compare.
    // SAFETY: each load and store covers `src` or `dst` at `i..i + 8` or
    // `i + 8..i + 16` with `i + 8 <= n` or `i + 16 <= n`, and both hold at
    // least `n` values (checked above).
    unsafe {
        while i + 16 <= n {
            let s0 = _mm256_loadu_ps(src.as_ptr().add(i));
            let s1 = _mm256_loadu_ps(src.as_ptr().add(i + 8));
            _mm256_storeu_ps(dst.as_mut_ptr().add(i), _mm256_mul_ps(s0, vscale));
            _mm256_storeu_ps(dst.as_mut_ptr().add(i + 8), _mm256_mul_ps(s1, vscale));
            i += 16;
        }
        while i + 8 <= n {
            let sv = _mm256_loadu_ps(src.as_ptr().add(i));
            _mm256_storeu_ps(dst.as_mut_ptr().add(i), _mm256_mul_ps(sv, vscale));
            i += 8;
        }
    }
    for j in i..n {
        dst[j] = src[j] * scale;
    }
}

/// `dst[i] = src[i] * scale` for `i < n`.
///
/// Panics when `src` or `dst` holds fewer than `n` values.
///
/// The loops keep their pointer loads and stores, for the reason given at
/// `crate::pitch::celt_inner_prod`: the function is inlined into
/// `CeltEncoder::encode`, whose loops got longer with the safe form.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn scale_slice_neon(src: &[f32], dst: &mut [f32], scale: f32, n: usize) {
    use core::arch::aarch64::*;
    let (src, dst) = (&src[..n], &mut dst[..n]);
    // SAFETY: NEON is available: src/lib.rs refuses to build for aarch64
    // without it. `src` and `dst` hold `n` values (sliced above), and every
    // load and store covers `[i..i + 4]` with `i + 4 <= n`.
    let i = unsafe {
        let vscale = vdupq_n_f32(scale);
        let mut i = 0;

        while i + 16 <= n {
            let s0 = vld1q_f32(src.as_ptr().add(i));
            let s1 = vld1q_f32(src.as_ptr().add(i + 4));
            let s2 = vld1q_f32(src.as_ptr().add(i + 8));
            let s3 = vld1q_f32(src.as_ptr().add(i + 12));
            vst1q_f32(dst.as_mut_ptr().add(i), vmulq_f32(s0, vscale));
            vst1q_f32(dst.as_mut_ptr().add(i + 4), vmulq_f32(s1, vscale));
            vst1q_f32(dst.as_mut_ptr().add(i + 8), vmulq_f32(s2, vscale));
            vst1q_f32(dst.as_mut_ptr().add(i + 12), vmulq_f32(s3, vscale));
            i += 16;
        }
        while i + 8 <= n {
            let s0 = vld1q_f32(src.as_ptr().add(i));
            let s1 = vld1q_f32(src.as_ptr().add(i + 4));
            vst1q_f32(dst.as_mut_ptr().add(i), vmulq_f32(s0, vscale));
            vst1q_f32(dst.as_mut_ptr().add(i + 4), vmulq_f32(s1, vscale));
            i += 8;
        }
        while i + 4 <= n {
            let s0 = vld1q_f32(src.as_ptr().add(i));
            vst1q_f32(dst.as_mut_ptr().add(i), vmulq_f32(s0, vscale));
            i += 4;
        }
        i
    };
    for j in i..n {
        dst[j] = src[j] * scale;
    }
}

#[allow(clippy::too_many_arguments)]
pub fn denormalise_bands(
    m: &CeltMode,
    x: &[f32],
    freq: &mut [f32],
    band_e: &[f32],
    start: usize,
    end: usize,
    channels: usize,
    m_val: usize,
) {
    let lm = m_val.trailing_zeros() as usize;
    let frame_size = m.short_mdct_size << lm;
    #[cfg(target_arch = "x86_64")]
    let use_avx2 = crate::compat::x86_has_avx2();

    for c in 0..channels {
        for i in start..end {
            let base = c * frame_size + ((m.e_bands[i] as usize) << lm);
            let n = ((m.e_bands[i + 1] - m.e_bands[i]) as usize) << lm;
            let band_log = band_e[c * m.nb_ebands + i];

            // C: `g = celt_exp2_db(MIN32(32.f, lg))`, the polynomial
            // `celt_exp2` of the float build. The cap keeps the gain finite.
            let g = celt_exp2(if 32.0 < band_log { 32.0 } else { band_log });
            let src = &x[base..base + n];
            let dst = &mut freq[base..base + n];
            #[cfg(target_arch = "x86_64")]
            if n >= 8 && use_avx2 {
                // SAFETY: the CPU has AVX2, the feature of the kernel.
                unsafe { scale_slice_avx2(src, dst, g, n) };
                continue;
            }
            #[cfg(target_arch = "aarch64")]
            if n >= 8 {
                scale_slice_neon(src, dst, g, n);
                continue;
            }
            for (d, &s) in dst.iter_mut().zip(src) {
                *d = s * g;
            }
        }
    }
}

pub fn celt_lcg_rand(seed: u32) -> u32 {
    seed.wrapping_mul(1664525).wrapping_add(1013904223)
}

/// `renormalise_vector` with AVX2 and FMA: the energy of `x[..n]` in two
/// sums of eight lanes, then every value times `(1 / sqrt(E)) * gain`.
///
/// Panics when `x` holds fewer than `n` values.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
fn renormalise_vector_avx2(x: &mut [f32], n: usize, gain: f32) {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;
    use crate::simd_mem::x86::{load_f32x8, store_f32x8};

    // The loads and stores below cover `x[..n]`.
    assert!(x.len() >= n);
    let x = &mut x[..n];

    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();

    let (octets, tail) = x.as_chunks::<8>();
    let (pairs, last) = octets.as_chunks::<2>();
    for [a, b] in pairs {
        let v0 = load_f32x8(a);
        let v1 = load_f32x8(b);
        acc0 = _mm256_fmadd_ps(v0, v0, acc0);
        acc1 = _mm256_fmadd_ps(v1, v1, acc1);
    }

    // At most one group of eight is left.
    if let [a] = last {
        let v0 = load_f32x8(a);
        acc0 = _mm256_fmadd_ps(v0, v0, acc0);
    }

    let acc = _mm256_add_ps(acc0, acc1);
    let hi = _mm256_extractf128_ps(acc, 1);
    let lo = _mm256_castps256_ps128(acc);
    let s4 = _mm_add_ps(lo, hi);
    let t1 = _mm_movehl_ps(s4, s4);
    let s2 = _mm_add_ps(s4, t1);
    let t2 = _mm_shuffle_ps(s2, s2, 0x55);
    let mut e = 1e-15f32 + _mm_cvtss_f32(_mm_add_ss(s2, t2));

    for &v in tail {
        e += v * v;
    }

    // C (renormalise_vector): the reciprocal first, then the gain.
    let norm = (1.0 / e.sqrt()) * gain;
    let vnorm = _mm256_set1_ps(norm);

    let (octets, tail) = x.as_chunks_mut::<8>();
    let (pairs, last) = octets.as_chunks_mut::<2>();
    for [a, b] in pairs {
        let v0 = load_f32x8(a);
        let v1 = load_f32x8(b);
        store_f32x8(a, _mm256_mul_ps(v0, vnorm));
        store_f32x8(b, _mm256_mul_ps(v1, vnorm));
    }
    // At most one group of eight is left.
    if let [a] = last {
        let v = load_f32x8(a);
        store_f32x8(a, _mm256_mul_ps(v, vnorm));
    }
    for v in tail {
        *v *= norm;
    }
}

/// libopus `renormalise_vector` (celt/vq.c, float build): `E = EPSILON +
/// celt_inner_prod(X, X)`, then every value times `(1 / sqrt(E)) * gain`.
/// The sum has the order of libopus on this processor (`celt_inner_prod`);
/// the offset is added after it.
#[inline]
fn renormalise_vector_libopus(x: &mut [f32], gain: f32) {
    let e = 1e-15f32 + celt_inner_prod(x, x);
    let g = (1.0 / e.sqrt()) * gain;
    for xv in x.iter_mut() {
        *xv *= g;
    }
}

/// Scales `x[..n]` so that its energy becomes `gain * gain`, as libopus's
/// `renormalise_vector` does (see `renormalise_vector_libopus`). x86-64 with
/// AVX2 sums bands of 16 values and more with its own kernel.
pub fn renormalise_vector(x: &mut [f32], n: usize, gain: f32) {
    // The only bounds check: the vector kernel works on `n` values.
    let x = &mut x[..n];
    #[cfg(target_arch = "x86_64")]
    {
        if n >= 16 && crate::compat::x86_has_avx2_fma() {
            // SAFETY: the CPU has AVX2 and FMA, the features of the kernel.
            unsafe { renormalise_vector_avx2(x, n, gain) };
            return;
        }
    }
    renormalise_vector_libopus(x, gain);
}

#[allow(clippy::too_many_arguments)]
pub fn anti_collapse(
    m: &CeltMode,
    x_buf: &mut [f32],
    collapse_masks: &[u32],
    lm: i32,
    channels: usize,
    size: usize,
    start: usize,
    end: usize,
    log_e: &[f32],
    prev1_log_e: &[f32],
    prev2_log_e: &[f32],
    pulses: &[i32],
    mut seed: u32,
) -> u32 {
    for i in start..end {
        let n0 = (m.e_bands[i + 1] - m.e_bands[i]) as usize;
        let depth = if n0 > 0 {
            ((1 + pulses[i]) / n0 as i32) >> lm
        } else {
            0
        };

        // C: `thresh = .5f*celt_exp2(-.125f*depth)`.
        let thresh = 0.5 * celt_exp2(-0.125 * depth as f32);
        let sqrt_1 = 1.0 / ((n0 << lm) as f32).sqrt();

        for c in 0..channels {
            let p1 = prev1_log_e[c * m.nb_ebands + i];
            let p2 = prev2_log_e[c * m.nb_ebands + i];

            let (p1_adj, p2_adj) = if channels == 1 && prev1_log_e.len() >= 2 * m.nb_ebands {
                (
                    p1.max(prev1_log_e[m.nb_ebands + i]),
                    p2.max(prev2_log_e[m.nb_ebands + i]),
                )
            } else {
                (p1, p2)
            };

            let e_diff = log_e[c * m.nb_ebands + i] - p1_adj.min(p2_adj);
            let e_diff = e_diff.max(0.0);

            // C: `r = 2.f*celt_exp2(-Ediff)`, then `r *= 1.41421356f` for LM 3.
            let mut r = 2.0 * celt_exp2(-e_diff);
            if lm == 3 {
                r *= core::f32::consts::SQRT_2;
            }
            r = r.min(thresh);
            r *= sqrt_1;

            let x_offset = c * size + ((m.e_bands[i] as usize) << lm);
            let mut renormalize = false;
            for k in 0..(1 << lm) {
                if (collapse_masks[i * channels + c] & (1 << k)) == 0 {
                    for j in 0..n0 {
                        seed = celt_lcg_rand(seed);
                        x_buf[x_offset + (j << lm) + k] = if (seed & 0x8000) != 0 { r } else { -r };
                    }
                    renormalize = true;
                }
            }
            if renormalize {
                renormalise_vector(&mut x_buf[x_offset..x_offset + (n0 << lm)], n0 << lm, 1.0);
            }
        }
    }
    seed
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;

    /// `isqrt32` gives the result of the loop of libopus for every value up
    /// to 2^24, for every square up to 2^32 and its two neighbours, and for
    /// the largest values.
    #[test]
    fn isqrt32_matches_the_bit_loop() {
        for val in 1..1u32 << 24 {
            assert_eq!(isqrt32(val), isqrt32_bits(val), "val {val}");
        }
        for root in 1..=65535u32 {
            let sq = root * root;
            for val in [sq - 1, sq, sq.saturating_add(1)] {
                if val >= 1 {
                    assert_eq!(isqrt32(val), isqrt32_bits(val), "val {val}");
                }
            }
        }
        for val in (u32::MAX - 1000)..=u32::MAX {
            assert_eq!(isqrt32(val), isqrt32_bits(val), "val {val}");
        }
    }

    #[test]
    fn test_bitexact_primitives_reference_values() {
        assert_eq!(bitexact_cos(64), 32767);
        assert_eq!(bitexact_cos(8192), 23171);
        assert_eq!(bitexact_cos(16320), 200);

        assert_eq!(bitexact_log2tan(32767, 200), 15059);
        assert_eq!(bitexact_log2tan(30274, 12540), 2611);
        assert_eq!(bitexact_log2tan(23171, 23171), 0);
        assert_eq!(bitexact_log2tan(200, 32767), -15059);
    }

    /// Values that stress float kernels: both zeros, the smallest and the
    /// largest magnitudes, and infinities.
    const EXTREMES: [f32; 10] = [
        0.0,
        -0.0,
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
        1.0e-45,
        f32::MAX,
        -f32::MAX,
        f32::INFINITY,
        f32::NEG_INFINITY,
        1.0,
    ];

    /// `len` values in -1..1 from a fixed generator. With `extremes`, about
    /// every seventh value is replaced by one of `EXTREMES`.
    fn test_values(len: usize, seed: &mut u32, extremes: bool) -> Vec<f32> {
        (0..len)
            .map(|_| {
                *seed = celt_lcg_rand(*seed);
                let pick = (*seed >> 8) as usize;
                if extremes && pick.is_multiple_of(7) {
                    EXTREMES[(pick / 7) % EXTREMES.len()]
                } else {
                    (*seed >> 8) as f32 / (1u32 << 23) as f32 - 1.0
                }
            })
            .collect()
    }

    /// Bit-for-bit equality. Two NaN values count as equal: which NaN an
    /// operation returns is not part of what the kernels promise.
    fn assert_same_bits(got: &[f32], want: &[f32], what: &str) {
        assert_eq!(got.len(), want.len(), "{what}: length");
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert!(
                g.to_bits() == w.to_bits() || (g.is_nan() && w.is_nan()),
                "{what}: value {i} is {g:e} ({:#010x}), expected {w:e} ({:#010x})",
                g.to_bits(),
                w.to_bits()
            );
        }
    }

    /// The Haar step as libopus writes it (celt/bands.c, haar1).
    fn haar1_reference(x: &mut [f32], n0: usize, stride: usize) {
        let n = n0 >> 1;
        for i in 0..stride {
            for j in 0..n {
                let tmp1 = core::f32::consts::FRAC_1_SQRT_2 * x[stride * 2 * j + i];
                let tmp2 = core::f32::consts::FRAC_1_SQRT_2 * x[stride * (2 * j + 1) + i];
                x[stride * 2 * j + i] = tmp1 + tmp2;
                x[stride * (2 * j + 1) + i] = tmp1 - tmp2;
            }
        }
    }

    #[test]
    fn haar1_kernels_match_reference() {
        let mut seed = 0x0bad_5eedu32;
        let row_counts = (0..=41).chain([64, 65, 88, 127, 176]);
        for n0 in row_counts {
            for stride in [1usize, 2, 3, 4, 5, 8, 16] {
                for extremes in [false, true] {
                    // Three values more than the kernel may touch. They
                    // must come back unchanged.
                    let input = test_values(n0 * stride + 3, &mut seed, extremes);
                    let what = format!("n0 {n0} stride {stride} extremes {extremes}");
                    let mut want = input.clone();
                    haar1_reference(&mut want, n0, stride);

                    let mut got = input.clone();
                    haar1(&mut got, n0, stride);
                    assert_same_bits(&got, &want, &format!("haar1, {what}"));

                    let used = 2 * (n0 >> 1) * stride;
                    let mut got = input.clone();
                    haar1_scalar(&mut got[..used], stride);
                    assert_same_bits(&got, &want, &format!("haar1_scalar, {what}"));

                    #[cfg(target_arch = "aarch64")]
                    {
                        let mut got = input.clone();
                        // SAFETY: NEON is available: src/lib.rs refuses to
                        // build for aarch64 without it.
                        unsafe { haar1_neon(&mut got[..used], stride) };
                        assert_same_bits(&got, &want, &format!("haar1_neon, {what}"));
                    }
                }
            }
        }
    }

    /// stereo_merge with one indexed loop per step, as libopus writes it
    /// (celt/bands.c, stereo_merge, with the scalar inner products).
    fn stereo_merge_reference(x: &mut [f32], y: &mut [f32], mid: f32, n: usize) {
        let xp = mid * inner_prod_model(&y[..n], &x[..n]);
        let side_e = inner_prod_model(&y[..n], &y[..n]);
        let t = mac_model(side_e, mid, mid);
        let el = t - 2.0 * xp;
        let er = t + 2.0 * xp;
        if er < 6e-4f32 || el < 6e-4f32 {
            y[..n].copy_from_slice(&x[..n]);
            return;
        }
        let lgain = 1.0 / el.sqrt();
        let rgain = 1.0 / er.sqrt();
        for i in 0..n {
            let l = mid * x[i];
            let r = y[i];
            x[i] = lgain * (l - r);
            y[i] = rgain * (l + r);
        }
    }

    #[test]
    fn stereo_helpers_match_indexed_loops() {
        let mode = crate::modes::default_mode();
        let mut seed = 0x5eed_0001u32;
        for n in (0..=40).chain([64, 88, 96, 144, 176]) {
            for extremes in [false, true] {
                for mid in [0.0f32, 1.0e-3, 0.3, 0.8, 1.0] {
                    // Two values more than `n`: they must stay unchanged.
                    let x0 = test_values(n + 2, &mut seed, extremes);
                    let y0 = test_values(n + 2, &mut seed, extremes);
                    let what = format!("n {n} extremes {extremes} mid {mid}");

                    let (mut x, mut y) = (x0.clone(), y0.clone());
                    let (mut xr, mut yr) = (x0.clone(), y0.clone());
                    stereo_merge(&mut x, &mut y, mid, n);
                    stereo_merge_reference(&mut xr, &mut yr, mid, n);
                    assert_same_bits(&x, &xr, &format!("stereo_merge x, {what}"));
                    assert_same_bits(&y, &yr, &format!("stereo_merge y, {what}"));

                    let (mut x, mut y) = (x0.clone(), y0.clone());
                    let (mut xr, mut yr) = (x0.clone(), y0.clone());
                    stereo_split(&mut x, &mut y, n);
                    let scale = core::f32::consts::FRAC_1_SQRT_2;
                    for i in 0..n {
                        let l = scale * xr[i];
                        let r = scale * yr[i];
                        xr[i] = l + r;
                        yr[i] = r - l;
                    }
                    assert_same_bits(&x, &xr, &format!("stereo_split x, {what}"));
                    assert_same_bits(&y, &yr, &format!("stereo_split y, {what}"));

                    // Band energies for intensity_stereo: left then right.
                    let mut band_e = vec![0.0f32; 2 * mode.nb_ebands];
                    band_e[3] = mid + 0.25;
                    band_e[mode.nb_ebands + 3] = 0.5;
                    let (mut x, mut y) = (x0.clone(), y0.clone());
                    let mut xr = x0.clone();
                    intensity_stereo(mode, &mut x, &mut y, &band_e, 3, n);
                    // celt/bands.c 389-401 with EPSILON 1e-15.
                    let left = band_e[3];
                    let right = band_e[mode.nb_ebands + 3];
                    let norm = 1e-15 + mac_model(mac_model(1e-15, left, left), right, right).sqrt();
                    let (a1, a2) = (left / norm, right / norm);
                    for i in 0..n {
                        xr[i] = mac_model(a2 * y0[i], a1, xr[i]);
                    }
                    assert_same_bits(&x, &xr, &format!("intensity_stereo x, {what}"));
                    assert_same_bits(&y, &y0, &format!("intensity_stereo y, {what}"));
                }
            }
        }
    }

    /// Lengths that cover every tail of the 16, 8 and 4 wide loops.
    fn kernel_lengths() -> impl Iterator<Item = usize> {
        (0..=41).chain([63, 64, 65, 88, 127, 176])
    }

    /// `acc + a * b` as `mac` rounds it: fused on aarch64 only.
    fn mac_model(acc: f32, a: f32, b: f32) -> f32 {
        if cfg!(target_arch = "aarch64") {
            a.mul_add(b, acc)
        } else {
            acc + a * b
        }
    }

    /// Scalar model of `celt_inner_prod`. On aarch64 and x86-64: four lanes
    /// over the complete groups of four, lanes added as (0 + 2) + (1 + 3),
    /// then the rest one value at a time. Elsewhere one value at a time
    /// from 0.
    fn inner_prod_model(x: &[f32], y: &[f32]) -> f32 {
        let n = x.len();
        if !cfg!(any(target_arch = "aarch64", all(target_arch = "x86_64", target_feature = "sse2"))) {
            return (0..n).fold(0.0, |acc, i| mac_model(acc, x[i], y[i]));
        }
        let mut lanes = [0.0f32; 4];
        for i in 0..(n & !3) {
            lanes[i % 4] = mac_model(lanes[i % 4], x[i], y[i]);
        }
        let mut sum = (lanes[0] + lanes[2]) + (lanes[1] + lanes[3]);
        for i in (n & !3)..n {
            sum = mac_model(sum, x[i], y[i]);
        }
        sum
    }

    #[test]
    fn celt_inner_prod_matches_its_scalar_model() {
        let mut seed = 0x17e7_a003u32;
        for n in kernel_lengths() {
            for extremes in [false, true] {
                let x = test_values(n, &mut seed, extremes);
                let y = test_values(n + 3, &mut seed, extremes);
                let got = celt_inner_prod(&x, &y);
                let want = inner_prod_model(&x, &y[..n]);
                assert_same_bits(&[got], &[want], &format!("n {n} extremes {extremes}"));
            }
        }
    }

    /// The energies of `stereo_itheta` as the model sums them.
    fn itheta_energies_model(x: &[f32], y: &[f32], stereo: bool) -> (f32, f32) {
        if stereo {
            let (mut emid, mut eside) = (0.0f32, 0.0f32);
            for i in 0..x.len() {
                let (m, s) = (x[i] + y[i], x[i] - y[i]);
                emid = mac_model(emid, m, m);
                eside = mac_model(eside, s, s);
            }
            (emid, eside)
        } else {
            (inner_prod_model(x, x), inner_prod_model(y, y))
        }
    }

    #[test]
    fn stereo_itheta_sums_as_libopus_does() {
        let mut seed = 0x17e7_a001u32;
        for n in kernel_lengths() {
            for stereo in [false, true] {
                for round in 0..8 {
                    let x = test_values(n, &mut seed, false);
                    let mut y = test_values(n, &mut seed, false);
                    // Some rounds with a much weaker second signal, so that
                    // small and large angles are both covered.
                    if round % 2 == 1 {
                        y.iter_mut().for_each(|v| *v *= 1.0e-3);
                    }
                    let (emid, eside) = itheta_energies_model(&x, &y, stereo);
                    let want =
                        (0.5 + 1_073_741_824.0 * celt_atan2p_norm(eside.sqrt(), emid.sqrt())) as i32;
                    let got = stereo_itheta(&x, &y, stereo, n);
                    assert_eq!(got, want, "n {n} stereo {stereo} round {round}");
                    assert!((0..=1 << 30).contains(&got), "n {n}: angle {got}");
                }
            }
        }
    }

    /// Two silent halves give the angle 0, as in libopus 1.6, whose sums
    /// start at 0. A start at 1e-15 gave two equal energies, a quarter turn
    /// halved: 8192 in Q14.
    #[test]
    fn silent_halves_have_the_angle_zero() {
        let zeros = [0.0f32; 24];
        for stereo in [false, true] {
            assert_eq!(stereo_itheta(&zeros, &zeros, stereo, 24), 0);
        }
    }

    #[test]
    #[should_panic]
    fn stereo_itheta_rejects_a_slice_that_is_too_short() {
        let x = [0.5f32; 16];
        let y = [0.25f32; 15];
        stereo_itheta(&x, &y, true, 16);
    }

    #[test]
    fn band_energies_add_the_offset_after_the_sum() {
        let mode = crate::modes::default_mode();
        let mut seed = 0xe4e2_0001u32;
        for lm in 0..=3usize {
            for extremes in [false, true] {
                let n = mode.short_mdct_size << lm;
                let x = test_values(2 * n, &mut seed, extremes);
                let mut band_e = vec![0.0f32; 2 * mode.nb_ebands];
                compute_band_energies(mode, &x, &mut band_e, mode.nb_ebands, 2, lm);
                for c in 0..2 {
                    for i in 0..mode.nb_ebands {
                        let lo = c * n + ((mode.e_bands[i] as usize) << lm);
                        let hi = c * n + ((mode.e_bands[i + 1] as usize) << lm);
                        let band = &x[lo..hi];
                        // The AVX2 and FMA kernel of x86-64 sums in another
                        // order; without it, `celt_inner_prod` runs.
                        if band.len() >= 8 && crate::compat::x86_has_avx2_fma() {
                            continue;
                        }
                        let want = (1e-27f32 + inner_prod_model(band, band)).sqrt();
                        let got = band_e[c * mode.nb_ebands + i];
                        assert_same_bits(&[got], &[want], &format!("lm {lm} band {i}"));
                    }
                }
            }
        }
    }

    /// `renormalise_vector` sums as `celt_inner_prod` does, adds the offset
    /// after the sum and scales by `(1 / sqrt(E)) * gain`, as libopus.
    #[test]
    fn renormalise_vector_matches_its_model() {
        let mut seed = 0x4e04_0001u32;
        for n in kernel_lengths() {
            for extremes in [false, true] {
                for gain in [1.0f32, 0.37, 0.0] {
                    // Two values more than `n`: they must stay unchanged.
                    let input = test_values(n + 2, &mut seed, extremes);
                    let what = format!("n {n} extremes {extremes} gain {gain}");
                    let mut want = input.clone();
                    let e = 1e-15f32 + inner_prod_model(&want[..n], &want[..n]);
                    let g = (1.0 / e.sqrt()) * gain;
                    for v in want[..n].iter_mut() {
                        *v *= g;
                    }
                    let mut got = input.clone();
                    renormalise_vector(&mut got, n, gain);
                    // As above: only the AVX2 and FMA kernel sums otherwise.
                    if n >= 16 && crate::compat::x86_has_avx2_fma() {
                        continue;
                    }
                    assert_same_bits(&got, &want, &what);
                }
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn scale_slice_neon_matches_scalar_multiply() {
        let mut seed = 0x5ca1_e001u32;
        for n in kernel_lengths() {
            for extremes in [false, true] {
                let src = test_values(n, &mut seed, extremes);
                let scale = 1.0 + test_values(1, &mut seed, false)[0];
                // Two values more than `n`: they must stay unchanged.
                let mut dst = vec![7.0f32; n + 2];
                scale_slice_neon(&src, &mut dst, scale, n);
                let mut want: Vec<f32> = src.iter().map(|&v| v * scale).collect();
                want.extend([7.0, 7.0]);
                assert_same_bits(&dst, &want, &format!("n {n} extremes {extremes}"));
            }
        }
    }

    /// The row orders of the Hadamard transform as libopus lists them
    /// (celt/bands.c, ordery_table), written out a second time so that the
    /// test does not read the table it checks.
    const HADAMARD_ORDERS: [(usize, &[usize]); 4] = [
        (2, &[1, 0]),
        (4, &[3, 0, 2, 1]),
        (8, &[7, 0, 4, 3, 6, 1, 5, 2]),
        (16, &[15, 0, 8, 7, 12, 3, 11, 4, 14, 1, 9, 6, 13, 2, 10, 5]),
    ];

    #[test]
    fn hadamard_reorder_matches_its_definition_and_round_trips() {
        for stride in 1usize..=16 {
            let order = HADAMARD_ORDERS.iter().find(|(s, _)| *s == stride).map(|(_, o)| *o);
            // Every band size up to the largest one that fits the scratch
            // buffer, and the empty band.
            for n0 in 0..=(MAX_PVQ_N / stride).min(48) {
                // The Hadamard order exists for 2, 4, 8 and 16 rows only.
                let orders: &[bool] = if order.is_some() { &[false, true] } else { &[false] };
                for &hadamard in orders {
                    let n = n0 * stride;
                    // Two values more than `n`: they must stay unchanged.
                    let input: Vec<f32> = (0..n + 2).map(|i| i as f32 + 0.5).collect();
                    let what = format!("n0 {n0} stride {stride} hadamard {hadamard}");

                    // Row of the reordered layout that takes column `i`.
                    let row = |i: usize| -> usize { if hadamard { order.unwrap()[i] } else { i } };
                    let mut want = input.clone();
                    for i in 0..stride {
                        for j in 0..n0 {
                            want[row(i) * n0 + j] = input[j * stride + i];
                        }
                    }
                    let mut got = input.clone();
                    deinterleave_hadamard(&mut got, n0, stride, hadamard);
                    assert_same_bits(&got, &want, &format!("deinterleave, {what}"));

                    interleave_hadamard(&mut got, n0, stride, hadamard);
                    assert_same_bits(&got, &input, &format!("round trip, {what}"));
                }
            }
        }
    }

    /// With a row count that has no Hadamard order the two functions must
    /// panic. Otherwise some of these counts would make
    /// `deinterleave_hadamard` copy scratch values that it never wrote.
    #[test]
    fn hadamard_reorder_rejects_a_row_count_without_an_order() {
        for stride in [1usize, 3, 5, 6, 7, 9, 12, 15, 17, 32] {
            for deinterleave in [true, false] {
                let result = std::panic::catch_unwind(|| {
                    let mut x = vec![1.0f32; 4 * stride];
                    if deinterleave {
                        deinterleave_hadamard(&mut x, 4, stride, true);
                    } else {
                        interleave_hadamard(&mut x, 4, stride, true);
                    }
                });
                assert!(result.is_err(), "stride {stride} deinterleave {deinterleave}");
            }
        }
    }

    #[test]
    fn hadamard_reorder_rejects_a_band_that_is_too_long() {
        for stride in [2usize, 3, 8] {
            for deinterleave in [true, false] {
                let result = std::panic::catch_unwind(|| {
                    let n0 = MAX_PVQ_N / stride + 1;
                    let mut x = vec![0.0f32; n0 * stride];
                    if deinterleave {
                        deinterleave_hadamard(&mut x, n0, stride, false);
                    } else {
                        interleave_hadamard(&mut x, n0, stride, false);
                    }
                });
                assert!(result.is_err(), "stride {stride} deinterleave {deinterleave}");
            }
        }
    }

    #[test]
    fn hadamard_reorder_rejects_a_slice_that_is_too_short() {
        for deinterleave in [true, false] {
            let result = std::panic::catch_unwind(|| {
                let mut x = vec![0.0f32; 31];
                if deinterleave {
                    deinterleave_hadamard(&mut x, 4, 8, true);
                } else {
                    interleave_hadamard(&mut x, 4, 8, true);
                }
            });
            assert!(result.is_err(), "deinterleave {deinterleave}");
        }
    }

    #[test]
    #[should_panic]
    fn haar1_rejects_a_slice_that_is_too_short() {
        let mut x = [0.0f32; 15];
        haar1(&mut x, 16, 1);
    }

    #[test]
    fn lcg_ahead_matches_single_steps() {
        for seed in [0u32, 1, 0x8000, 0xffff_ffff, 0x1234_5678, 0xdead_beef] {
            let mut s = seed;
            for (mul, add) in LCG_AHEAD {
                s = celt_lcg_rand(s);
                assert_eq!(seed.wrapping_mul(mul).wrapping_add(add), s, "seed {seed:#x}");
            }
        }
    }

    #[test]
    fn noise_kernels_match_the_loops_they_replace() {
        let mut gen_seed = 0x0153_0001u32;
        for n in kernel_lengths() {
            for round in 0..12u32 {
                gen_seed = celt_lcg_rand(gen_seed);
                let seed = match round {
                    0 => 0,
                    1 => 0xffff_ffff,
                    2 => 0x8000,
                    _ => gen_seed,
                };
                let lb = test_values(n, &mut gen_seed, round % 2 == 1);
                let what = format!("n {n} seed {seed:#x}");

                // The folded band plus noise, as libopus writes it.
                let mut want = vec![0.0f32; n];
                let mut s = seed;
                for j in 0..n {
                    s = celt_lcg_rand(s);
                    want[j] = lb[j] + if s & 0x8000 != 0 { 1.0 / 256.0 } else { -1.0 / 256.0 };
                }
                let mut got = vec![7.0f32; n];
                assert_eq!(fold_with_noise_scalar(&mut got, &lb, seed), s, "scalar seed, {what}");
                assert_same_bits(&got, &want, &format!("fold scalar, {what}"));
                let mut got = vec![7.0f32; n];
                assert_eq!(fold_with_noise(&mut got, &lb, seed), s, "seed, {what}");
                assert_same_bits(&got, &want, &format!("fold, {what}"));
                #[cfg(target_arch = "aarch64")]
                {
                    // A longer `lb`: only the first `n` values count.
                    let mut lb_long = lb.clone();
                    lb_long.extend([9.0, 9.0, 9.0]);
                    let mut got = vec![7.0f32; n];
                    assert_eq!(fold_with_noise_neon(&mut got, &lb_long, seed), s, "neon seed, {what}");
                    assert_same_bits(&got, &want, &format!("fold neon, {what}"));
                }
                #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
                {
                    // A longer `lb`: only the first `n` values count.
                    let mut lb_long = lb.clone();
                    lb_long.extend([9.0, 9.0, 9.0]);
                    let mut got = vec![7.0f32; n];
                    assert_eq!(fold_with_noise_sse2(&mut got, &lb_long, seed), s, "sse2 seed, {what}");
                    assert_same_bits(&got, &want, &format!("fold sse2, {what}"));
                }

                // Noise alone.
                let mut want = vec![0.0f32; n];
                let mut s = seed;
                for v in want.iter_mut() {
                    s = celt_lcg_rand(s);
                    *v = ((s as i32 >> 20) as f32) / 16384.0;
                }
                let mut got = vec![7.0f32; n];
                assert_eq!(fill_with_noise_scalar(&mut got, seed), s, "scalar seed, {what}");
                assert_same_bits(&got, &want, &format!("fill scalar, {what}"));
                let mut got = vec![7.0f32; n];
                assert_eq!(fill_with_noise(&mut got, seed), s, "seed, {what}");
                assert_same_bits(&got, &want, &format!("fill, {what}"));
                #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
                {
                    let mut got = vec![7.0f32; n];
                    assert_eq!(fill_with_noise_sse2(&mut got, seed), s, "sse2 seed, {what}");
                    assert_same_bits(&got, &want, &format!("fill sse2, {what}"));
                }
            }
        }
    }
}
