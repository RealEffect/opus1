//! The fixed-point helpers of `src/silk/macros.rs` against the libopus macros
//! they port (silk/SigProc_FIX.h, silk/macros.h, silk/Inlines.h).
//!
//! Each `c_*` function below is the C expression written out in Rust: the
//! same operations in the same order, with the casts the C macro has. Where
//! libopus has two forms of a macro (with and without `OPUS_FAST_INT64`), the
//! reference is the form the Rust code does not use, so that the two are
//! independent.
//!
//! Signed overflow is undefined in C. Where a macro can overflow, the
//! reference wraps, which is what the compilers libopus is built with do.
//! Inputs for which the C macro has no defined meaning at all (a shift count
//! out of range, a division by zero) are left out and named at the test.

use opus_rs::silk::macros::*;

// ---------------------------------------------------------------------------
// Inputs
// ---------------------------------------------------------------------------

/// Values around every place where a 32-bit helper can change behaviour.
fn extremes() -> Vec<i32> {
    let mut v = vec![
        i32::MIN,
        i32::MIN + 1,
        i32::MIN + 2,
        -0x4000_0001,
        -0x4000_0000,
        -0x3FFF_FFFF,
        -0x0800_0001,
        -0x0800_0000,
        -0x07FF_FFFF,
        -65537,
        -65536,
        -65535,
        -32769,
        -32768,
        -32767,
        -2,
        -1,
        0,
        1,
        2,
        127,
        128,
        32767,
        32768,
        32769,
        65535,
        65536,
        65537,
        0x07FF_FFFF,
        0x0800_0000,
        0x0800_0001,
        0x3FFF_FFFF,
        0x4000_0000,
        0x4000_0001,
        i32::MAX - 2,
        i32::MAX - 1,
        i32::MAX,
    ];
    for bit in 0..31 {
        let p = 1i32 << bit;
        v.extend([p - 1, p, p.wrapping_add(1), -p, -p - 1, (-p).wrapping_add(1)]);
    }
    v.sort_unstable();
    v.dedup();
    v
}

/// A small deterministic generator (xorshift64*).
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// A 32-bit value. Half of them have a random number of leading sign
    /// bits, so that small and large magnitudes are equally likely.
    fn next_i32(&mut self) -> i32 {
        let r = self.next_u64();
        let v = (r >> 32) as i32;
        if r & 1 == 0 {
            v
        } else {
            v >> ((r >> 1) % 32)
        }
    }

    fn next_i64(&mut self) -> i64 {
        let r = self.next_u64();
        let v = self.next_u64() as i64;
        if r & 1 == 0 {
            v
        } else {
            v >> ((r >> 1) % 64)
        }
    }

    fn below(&mut self, n: u32) -> i32 {
        ((self.next_u64() >> 33) % n as u64) as i32
    }
}

const RANDOM_CASES: usize = 200_000;

/// Call `f` with every pair of extreme values and with random pairs.
fn for_pairs(seed: u64, mut f: impl FnMut(i32, i32)) {
    let ext = extremes();
    for &a in &ext {
        for &b in &ext {
            f(a, b);
        }
    }
    let mut rng = Rng(seed);
    for _ in 0..RANDOM_CASES {
        f(rng.next_i32(), rng.next_i32());
    }
}

/// Call `f` with every extreme value and each shift, and with random ones.
fn for_value_and_shift(seed: u64, shifts: core::ops::RangeInclusive<i32>, mut f: impl FnMut(i32, i32)) {
    for &a in &extremes() {
        for shift in shifts.clone() {
            f(a, shift);
        }
    }
    let mut rng = Rng(seed);
    let span = (shifts.end() - shifts.start() + 1) as u32;
    for _ in 0..RANDOM_CASES {
        f(rng.next_i32(), shifts.start() + rng.below(span));
    }
}

/// Call `f` with triples: all extremes for the first two values with a few
/// third values, and random triples.
fn for_triples(seed: u64, mut f: impl FnMut(i32, i32, i32)) {
    let ext = extremes();
    let third = [i32::MIN, -65536, -32768, -1, 0, 1, 32767, 65536, i32::MAX];
    for &a in &ext {
        for &b in &ext {
            for &c in &third {
                f(a, b, c);
                f(c, a, b);
            }
        }
    }
    let mut rng = Rng(seed);
    for _ in 0..RANDOM_CASES {
        f(rng.next_i32(), rng.next_i32(), rng.next_i32());
    }
}

// ---------------------------------------------------------------------------
// The C expressions
// ---------------------------------------------------------------------------

/// `silk_LIMIT(a, limit1, limit2)`:
/// `((limit1) > (limit2) ? ((a) > (limit1) ? (limit1) : ((a) < (limit2) ? (limit2) : (a)))
///                       : ((a) > (limit2) ? (limit2) : ((a) < (limit1) ? (limit1) : (a))))`
fn c_limit(a: i32, limit1: i32, limit2: i32) -> i32 {
    if limit1 > limit2 {
        if a > limit1 {
            limit1
        } else if a < limit2 {
            limit2
        } else {
            a
        }
    } else if a > limit2 {
        limit2
    } else if a < limit1 {
        limit1
    } else {
        a
    }
}

/// `silk_LSHIFT32(a, shift)`: `(opus_int32)((opus_uint32)(a) << (shift))`.
fn c_lshift32(a: i32, shift: i32) -> i32 {
    ((a as u32) << shift) as i32
}

/// `silk_LSHIFT_SAT32(a, shift)`:
/// `silk_LSHIFT32(silk_LIMIT((a), silk_RSHIFT32(silk_int32_MIN, (shift)),
///                                silk_RSHIFT32(silk_int32_MAX, (shift))), (shift))`
fn c_lshift_sat32(a: i32, shift: i32) -> i32 {
    c_lshift32(c_limit(a, i32::MIN >> shift, i32::MAX >> shift), shift)
}

/// `silk_ADD_POS_SAT32(a, b)`:
/// `((((opus_uint32)(a)+(opus_uint32)(b)) & 0x80000000) ? silk_int32_MAX : ((a)+(b)))`
fn c_add_pos_sat32(a: i32, b: i32) -> i32 {
    if ((a as u32).wrapping_add(b as u32) & 0x8000_0000) != 0 {
        i32::MAX
    } else {
        a.wrapping_add(b)
    }
}

/// `silk_ADD_SAT32(a, b)` (silk/macros.h):
/// `((((opus_uint32)(a) + (opus_uint32)(b)) & 0x80000000) == 0 ?
///     ((((a) & (b)) & 0x80000000) != 0 ? silk_int32_MIN : (a)+(b)) :
///     ((((a) | (b)) & 0x80000000) == 0 ? silk_int32_MAX : (a)+(b)))`
fn c_add_sat32(a: i32, b: i32) -> i32 {
    let sign = 0x8000_0000u32;
    if ((a as u32).wrapping_add(b as u32) & sign) == 0 {
        if ((a & b) as u32 & sign) != 0 {
            i32::MIN
        } else {
            a.wrapping_add(b)
        }
    } else if ((a | b) as u32 & sign) == 0 {
        i32::MAX
    } else {
        a.wrapping_add(b)
    }
}

/// `silk_SUB_SAT32(a, b)` (silk/macros.h):
/// `((((opus_uint32)(a)-(opus_uint32)(b)) & 0x80000000) == 0 ?
///     (( (a) & ((b)^0x80000000) & 0x80000000) ? silk_int32_MIN : (a)-(b)) :
///     ((((a)^0x80000000) & (b)  & 0x80000000) ? silk_int32_MAX : (a)-(b)))`
fn c_sub_sat32(a: i32, b: i32) -> i32 {
    let sign = 0x8000_0000u32;
    let (ua, ub) = (a as u32, b as u32);
    if (ua.wrapping_sub(ub) & sign) == 0 {
        if (ua & (ub ^ sign) & sign) != 0 {
            i32::MIN
        } else {
            a.wrapping_sub(b)
        }
    } else if ((ua ^ sign) & ub & sign) != 0 {
        i32::MAX
    } else {
        a.wrapping_sub(b)
    }
}

/// `silk_SAT16(a)`:
/// `((a) > silk_int16_MAX ? silk_int16_MAX : ((a) < silk_int16_MIN ? silk_int16_MIN : (a)))`
fn c_sat16(a: i32) -> i32 {
    if a > 32767 {
        32767
    } else if a < -32768 {
        -32768
    } else {
        a
    }
}

/// `silk_ADD_SAT16(a, b)`: `(opus_int16)silk_SAT16(silk_ADD32((opus_int32)(a), (b)))`
fn c_add_sat16(a: i16, b: i16) -> i16 {
    c_sat16(a as i32 + b as i32) as i16
}

/// `silk_RSHIFT_ROUND(a, shift)`:
/// `((shift) == 1 ? ((a) >> 1) + ((a) & 1) : (((a) >> ((shift) - 1)) + 1) >> 1)`
fn c_rshift_round(a: i32, shift: i32) -> i32 {
    if shift == 1 {
        (a >> 1) + (a & 1)
    } else {
        ((a >> (shift - 1)) + 1) >> 1
    }
}

/// `silk_RSHIFT_ROUND64(a, shift)`: the same expression on 64-bit values.
fn c_rshift_round64(a: i64, shift: i32) -> i64 {
    if shift == 1 {
        (a >> 1) + (a & 1)
    } else {
        ((a >> (shift - 1)) + 1) >> 1
    }
}

/// `silk_SMULBB(a32, b32)`:
/// `((opus_int32)((opus_int16)(a32)) * (opus_int32)((opus_int16)(b32)))`
fn c_smulbb(a: i32, b: i32) -> i32 {
    (a as i16 as i32).wrapping_mul(b as i16 as i32)
}

/// `silk_SMULWB(a32, b32)` without `OPUS_FAST_INT64`:
/// `((((a32) >> 16) * (opus_int32)((opus_int16)(b32))) +
///   ((((a32) & 0x0000FFFF) * (opus_int32)((opus_int16)(b32))) >> 16))`
fn c_smulwb(a: i32, b: i32) -> i32 {
    let b16 = b as i16 as i32;
    ((a >> 16).wrapping_mul(b16)).wrapping_add(((a & 0xFFFF).wrapping_mul(b16)) >> 16)
}

/// `silk_SMLAWB(a32, b32, c32)` without `OPUS_FAST_INT64`: `a32` plus the
/// expression of `silk_SMULWB(b32, c32)`.
fn c_smlawb(a: i32, b: i32, c: i32) -> i32 {
    a.wrapping_add(c_smulwb(b, c))
}

/// `silk_SMULWT(a32, b32)` without `OPUS_FAST_INT64`:
/// `(((a32) >> 16) * ((b32) >> 16) + ((((a32) & 0x0000FFFF) * ((b32) >> 16)) >> 16))`
fn c_smulwt(a: i32, b: i32) -> i32 {
    let bt = b >> 16;
    ((a >> 16).wrapping_mul(bt)).wrapping_add(((a & 0xFFFF).wrapping_mul(bt)) >> 16)
}

/// `silk_SMLAWT(a32, b32, c32)` without `OPUS_FAST_INT64`.
fn c_smlawt(a: i32, b: i32, c: i32) -> i32 {
    a.wrapping_add(c_smulwt(b, c))
}

/// `silk_SMLABB(a32, b32, c32)`:
/// `((a32) + ((opus_int32)((opus_int16)(b32))) * (opus_int32)((opus_int16)(c32)))`
fn c_smlabb(a: i32, b: i32, c: i32) -> i32 {
    a.wrapping_add((b as i16 as i32).wrapping_mul(c as i16 as i32))
}

/// `silk_SMULWW(a32, b32)` without `OPUS_FAST_INT64`:
/// `silk_MLA(silk_SMULWB((a32), (b32)), (a32), silk_RSHIFT_ROUND((b32), 16))`
fn c_smulww(a: i32, b: i32) -> i32 {
    c_smulwb(a, b).wrapping_add(a.wrapping_mul(c_rshift_round(b, 16)))
}

/// `silk_SMLAWW(a32, b32, c32)` without `OPUS_FAST_INT64`:
/// `silk_MLA(silk_SMLAWB((a32), (b32), (c32)), (b32), silk_RSHIFT_ROUND((c32), 16))`
fn c_smlaww(a: i32, b: i32, c: i32) -> i32 {
    c_smlawb(a, b, c).wrapping_add(b.wrapping_mul(c_rshift_round(c, 16)))
}

/// `silk_SMMUL(a32, b32)`: `(opus_int32)silk_RSHIFT64(silk_SMULL((a32), (b32)), 32)`
fn c_smmul(a: i32, b: i32) -> i32 {
    ((a as i64 * b as i64) >> 32) as i32
}

/// `EC_ILOG(x)` (celt/ecintrin.h): the number of bits needed to write `x`,
/// 0 for 0. Counted bit by bit here.
fn c_ec_ilog(x: u32) -> i32 {
    let mut bits = 0;
    let mut x = x;
    while x != 0 {
        bits += 1;
        x >>= 1;
    }
    bits
}

/// `silk_CLZ32(in32)`: `in32 ? 32 - EC_ILOG(in32) : 32`
fn c_clz32(x: i32) -> i32 {
    if x != 0 { 32 - c_ec_ilog(x as u32) } else { 32 }
}

/// `silk_CLZ64(in)` (silk/Inlines.h).
fn c_clz64(x: i64) -> i32 {
    let upper = (x >> 32) as i32;
    if upper == 0 {
        32 + c_clz32(x as i32)
    } else {
        c_clz32(upper)
    }
}

/// `silk_ROR32(a32, rot)` (silk/SigProc_FIX.h).
fn c_ror32(a: i32, rot: i32) -> i32 {
    let x = a as u32;
    let r = rot as u32;
    let m = rot.wrapping_neg() as u32;
    if rot == 0 {
        a
    } else if rot < 0 {
        ((x << m) | (x >> (32 - m))) as i32
    } else {
        ((x << (32 - r)) | (x >> r)) as i32
    }
}

/// `silk_CLZ_FRAC(in, lz, frac_Q7)` (silk/Inlines.h).
fn c_clz_frac(x: i32) -> (i32, i32) {
    let lz = c_clz32(x);
    (lz, c_ror32(x, 24 - lz) & 0x7f)
}

/// `silk_SQRT_APPROX(x)` (silk/Inlines.h).
fn c_sqrt_approx(x: i32) -> i32 {
    if x <= 0 {
        return 0;
    }
    let (lz, frac_q7) = c_clz_frac(x);
    let mut y: i32 = if lz & 1 != 0 { 32768 } else { 46214 };
    y >>= lz >> 1;
    c_smlawb(y, y, c_smulbb(213, frac_q7))
}

/// `silk_abs(a)`: `(((a) > 0) ? (a) : -(a))`.
fn c_abs(a: i32) -> i32 {
    if a > 0 { a } else { a.wrapping_neg() }
}

/// `silk_DIV32_varQ(a32, b32, Qres)` (silk/Inlines.h).
fn c_div32_varq(a32: i32, b32: i32, qres: i32) -> i32 {
    let a_headrm = c_clz32(c_abs(a32)) - 1;
    let mut a32_nrm = c_lshift32(a32, a_headrm);
    let b_headrm = c_clz32(c_abs(b32)) - 1;
    let b32_nrm = c_lshift32(b32, b_headrm);
    let b32_inv = (i32::MAX >> 2) / (b32_nrm >> 16);
    let mut result = c_smulwb(a32_nrm, b32_inv);
    a32_nrm = a32_nrm.wrapping_sub(c_lshift32(c_smmul(b32_nrm, result), 3));
    result = c_smlawb(result, a32_nrm, b32_inv);
    let lshift = 29 + a_headrm - b_headrm - qres;
    if lshift < 0 {
        c_lshift_sat32(result, -lshift)
    } else if lshift < 32 {
        result >> lshift
    } else {
        0
    }
}

/// `silk_INVERSE32_varQ(b32, Qres)` (silk/Inlines.h).
fn c_inverse32_varq(b32: i32, qres: i32) -> i32 {
    let b_headrm = c_clz32(c_abs(b32)) - 1;
    let b32_nrm = c_lshift32(b32, b_headrm);
    let b32_inv = (i32::MAX >> 2) / (b32_nrm >> 16);
    let mut result = c_lshift32(b32_inv, 16);
    let err_q32 = c_lshift32((1i32 << 29) - c_smulwb(b32_nrm, b32_inv), 3);
    result = c_smlaww(result, err_q32, b32_inv);
    let lshift = 61 - b_headrm - qres;
    if lshift <= 0 {
        c_lshift_sat32(result, -lshift)
    } else if lshift < 32 {
        result >> lshift
    } else {
        0
    }
}

/// `silk_lin2log(inLin)` (silk/lin2log.c).
fn c_lin2log(in_lin: i32) -> i32 {
    let (lz, frac_q7) = c_clz_frac(in_lin);
    c_smlawb(frac_q7, frac_q7.wrapping_mul(128 - frac_q7), 179).wrapping_add(c_lshift32(31 - lz, 7))
}

/// `silk_log2lin(inLog_Q7)` (silk/log2lin.c).
fn c_log2lin(in_log_q7: i32) -> i32 {
    if in_log_q7 < 0 {
        return 0;
    } else if in_log_q7 >= 3967 {
        return i32::MAX;
    }
    let out = c_lshift32(1, in_log_q7 >> 7);
    let frac_q7 = in_log_q7 & 0x7F;
    let curve = c_smlawb(frac_q7, c_smulbb(frac_q7, 128 - frac_q7), -174);
    if in_log_q7 < 2048 {
        out.wrapping_add(out.wrapping_mul(curve) >> 7)
    } else {
        out.wrapping_add((out >> 7).wrapping_mul(curve))
    }
}

/// `silk_RAND(seed)`: `silk_MLA_ovflw(907633515, (seed), 196314165)`.
fn c_rand(seed: i32) -> i32 {
    907_633_515u32.wrapping_add((seed as u32).wrapping_mul(196_314_165)) as i32
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// The values the decoder differed by before: the largest result of a
/// saturating shift is not `i32::MAX`.
#[test]
fn lshift_sat32_keeps_the_low_bits_clear() {
    assert_eq!(silk_lshift_sat32(i32::MAX, 4), 0x7FFF_FFF0);
    assert_eq!(silk_lshift_sat32(0x0800_0000, 4), 0x7FFF_FFF0);
    assert_eq!(silk_lshift_sat32(0x07FF_FFFF, 4), 0x7FFF_FFF0);
    assert_eq!(silk_lshift_sat32(0x07FF_FFFE, 4), 0x7FFF_FFE0);
    assert_eq!(silk_lshift_sat32(i32::MIN, 4), i32::MIN);
    assert_eq!(silk_lshift_sat32(-0x0800_0001, 4), i32::MIN);
    assert_eq!(silk_lshift_sat32(i32::MAX, 8), 0x7FFF_FF00);
    assert_eq!(silk_lshift_sat32(i32::MAX, 16), 0x7FFF_0000);
    assert_eq!(silk_lshift_sat32(12345, 0), 12345);
    // With a shift of 31 only the sign is left.
    assert_eq!(silk_lshift_sat32(i32::MAX, 31), 0);
    assert_eq!(silk_lshift_sat32(1, 31), 0);
    assert_eq!(silk_lshift_sat32(-1, 31), i32::MIN);
}

/// Shifts that libopus does not define must not panic, in debug builds
/// either.
#[test]
fn lshift_sat32_takes_any_shift() {
    for &a in &extremes() {
        for shift in [-40, -1, 32, 33, 63, 64, 1000, i32::MIN, i32::MAX] {
            let nearest = shift.clamp(0, 31);
            assert_eq!(silk_lshift_sat32(a, shift), c_lshift_sat32(a, nearest), "a {a}, shift {shift}");
        }
    }
}

#[test]
fn lshift_sat32_matches_the_c_macro() {
    for_value_and_shift(1, 0..=31, |a, shift| {
        assert_eq!(silk_lshift_sat32(a, shift), c_lshift_sat32(a, shift), "a {a}, shift {shift}");
    });
}

#[test]
fn limit_matches_the_c_macro_for_limits_in_either_order() {
    for_triples(2, |a, l1, l2| {
        let expected = c_limit(a, l1, l2);
        assert_eq!(silk_limit_32(a, l1, l2), expected, "a {a}, limits {l1} {l2}");
        assert_eq!(silk_limit(a, l1, l2), expected, "a {a}, limits {l1} {l2}");
        assert_eq!(silk_limit_int(a, l1, l2), expected, "a {a}, limits {l1} {l2}");
    });
    // The order of the limits does not matter.
    assert_eq!(silk_limit_32(5, 10, 0), 5);
    assert_eq!(silk_limit_32(-5, 10, 0), 0);
    assert_eq!(silk_limit_32(15, 10, 0), 10);
    assert_eq!(silk_limit_32(5, 0, 10), 5);
}

#[test]
fn saturating_adds_and_subtractions_match_the_c_macros() {
    for_pairs(3, |a, b| {
        assert_eq!(silk_add_pos_sat32(a, b), c_add_pos_sat32(a, b), "add_pos_sat32({a}, {b})");
        assert_eq!(silk_add_sat32(a, b), c_add_sat32(a, b), "add_sat32({a}, {b})");
        assert_eq!(silk_sub_sat32(a, b), c_sub_sat32(a, b), "sub_sat32({a}, {b})");
        // The aarch64 build of libopus uses the saturating instructions
        // (silk/arm/macros_arm64.h), which give the same values.
        assert_eq!(c_add_sat32(a, b), a.saturating_add(b));
        assert_eq!(c_sub_sat32(a, b), a.saturating_sub(b));
        assert_eq!(silk_sat16(a), c_sat16(a), "sat16({a})");
        let (a16, b16) = (a as i16, b as i16);
        assert_eq!(silk_add_sat16(a16, b16), c_add_sat16(a16, b16), "add_sat16({a16}, {b16})");
    });
}

/// `silk_ADD_POS_SAT32` is meant for values that are not negative, and
/// tests only the sign of the sum. libopus also calls it with negative
/// values (the rate-distortion sum in quant_LTP_gains.c), where a sum below
/// zero becomes `i32::MAX`.
#[test]
fn add_pos_sat32_tests_the_sign_of_the_sum() {
    assert_eq!(silk_add_pos_sat32(3, 4), 7);
    assert_eq!(silk_add_pos_sat32(i32::MAX, 1), i32::MAX);
    assert_eq!(silk_add_pos_sat32(0x4000_0000, 0x4000_0000), i32::MAX);
    assert_eq!(silk_add_pos_sat32(0x4000_0000, 0x3FFF_FFFF), i32::MAX);
    assert_eq!(silk_add_pos_sat32(10, -3), 7);
    assert_eq!(silk_add_pos_sat32(3, -10), i32::MAX);
    assert_eq!(silk_add_pos_sat32(-3, -10), i32::MAX);
    assert_eq!(silk_add_pos_sat32(i32::MIN, i32::MIN), 0);
    assert_eq!(silk_add_pos_sat32(i32::MAX, -5), i32::MAX - 5);
}

/// libopus requires `0 < shift < 32` (64 for the 64-bit form).
#[test]
fn rounding_shifts_match_the_c_macros() {
    for_value_and_shift(4, 1..=31, |a, shift| {
        assert_eq!(silk_rshift_round(a, shift), c_rshift_round(a, shift), "a {a}, shift {shift}");
    });
    // A shift of 31 leaves -1, 0 or 1.
    assert_eq!(silk_rshift_round(i32::MAX, 31), 1);
    assert_eq!(silk_rshift_round(0x4000_0000, 31), 1);
    assert_eq!(silk_rshift_round(0x3FFF_FFFF, 31), 0);
    assert_eq!(silk_rshift_round(-0x4000_0000, 31), 0);
    assert_eq!(silk_rshift_round(-0x4000_0001, 31), -1);
    assert_eq!(silk_rshift_round(i32::MIN, 31), -1);
    // Halfway cases go up.
    assert_eq!(silk_rshift_round(3, 1), 2);
    assert_eq!(silk_rshift_round(-3, 1), -1);
    assert_eq!(silk_rshift_round(6, 2), 2);
    assert_eq!(silk_rshift_round(-6, 2), -1);

    let mut rng = Rng(5);
    let mut wide: Vec<i64> = extremes().iter().map(|&v| v as i64).collect();
    wide.extend([i64::MIN, i64::MIN + 1, i64::MAX - 1, i64::MAX, 1 << 40, -(1 << 40), (1 << 62) - 1]);
    for &a in &wide {
        for shift in 1..=63 {
            // The C expression overflows for `i64::MAX` and a shift of 1.
            if shift == 1 && a == i64::MAX {
                continue;
            }
            assert_eq!(silk_rshift_round64(a, shift), c_rshift_round64(a, shift), "a {a}, shift {shift}");
        }
    }
    for _ in 0..RANDOM_CASES {
        let (a, shift) = (rng.next_i64(), 1 + rng.below(63));
        if shift == 1 && a == i64::MAX {
            continue;
        }
        assert_eq!(silk_rshift_round64(a, shift), c_rshift_round64(a, shift), "a {a}, shift {shift}");
    }
}

#[test]
fn multiplications_match_the_c_macros() {
    for_pairs(6, |a, b| {
        assert_eq!(silk_smulbb(a, b), c_smulbb(a, b), "smulbb({a}, {b})");
        assert_eq!(silk_smulwb(a, b), c_smulwb(a, b), "smulwb({a}, {b})");
        assert_eq!(silk_smulwt(a, b), c_smulwt(a, b), "smulwt({a}, {b})");
        assert_eq!(silk_smulww(a, b), c_smulww(a, b), "smulww({a}, {b})");
        assert_eq!(silk_smmul(a, b), c_smmul(a, b), "smmul({a}, {b})");
        assert_eq!(silk_smull(a, b), a as i64 * b as i64, "smull({a}, {b})");
        assert_eq!(silk_mul(a, b), a.wrapping_mul(b), "mul({a}, {b})");
    });
    for_triples(7, |a, b, c| {
        assert_eq!(silk_smlawb(a, b, c), c_smlawb(a, b, c), "smlawb({a}, {b}, {c})");
        assert_eq!(silk_smlawt(a, b, c), c_smlawt(a, b, c), "smlawt({a}, {b}, {c})");
        assert_eq!(silk_smlaww(a, b, c), c_smlaww(a, b, c), "smlaww({a}, {b}, {c})");
        assert_eq!(silk_smlabb(a, b, c), c_smlabb(a, b, c), "smlabb({a}, {b}, {c})");
        assert_eq!(silk_mla(a, b, c), a.wrapping_add(b.wrapping_mul(c)), "mla({a}, {b}, {c})");
    });
}

#[test]
fn shifts_and_shifted_sums_match_the_c_macros() {
    for_value_and_shift(8, 0..=31, |a, shift| {
        assert_eq!(silk_lshift(a, shift), c_lshift32(a, shift), "lshift({a}, {shift})");
        assert_eq!(silk_rshift(a, shift), a >> shift);
        assert_eq!(silk_rshift32(a, shift), a >> shift);
    });
    let mut rng = Rng(9);
    for _ in 0..RANDOM_CASES {
        let (a, b, shift) = (rng.next_i32(), rng.next_i32(), rng.below(32));
        assert_eq!(silk_add_lshift32(a, b, shift), a.wrapping_add(c_lshift32(b, shift)));
        assert_eq!(silk_sub_lshift32(a, b, shift), a.wrapping_sub(c_lshift32(b, shift)));
        assert_eq!(silk_add_rshift32(a, b, shift), a.wrapping_add(b >> shift));
        assert_eq!(silk_add_rshift(a, b, shift), a.wrapping_add(b >> shift));
        assert_eq!(silk_sub_rshift32(a, b, shift), a.wrapping_sub(b >> shift));
        assert_eq!(silk_add32_ovflw(a, b), (a as u32).wrapping_add(b as u32) as i32);
        assert_eq!(silk_sub32_ovflw(a, b), (a as u32).wrapping_sub(b as u32) as i32);
        let wide = rng.next_i64();
        let shift64 = rng.below(64);
        assert_eq!(silk_rshift64(wide, shift64), wide >> shift64);
    }
}

#[test]
fn bit_counts_and_minima_match_the_c_functions() {
    let mut rng = Rng(10);
    let mut values = extremes();
    values.extend((0..RANDOM_CASES).map(|_| rng.next_i32()));
    for &a in &values {
        assert_eq!(silk_clz32(a), c_clz32(a), "clz32({a})");
        let (mut lz, mut frac) = (0, 0);
        silk_clz_frac(a, &mut lz, &mut frac);
        assert_eq!((lz, frac), c_clz_frac(a), "clz_frac({a})");
        assert_eq!(silk_sqrt_approx(a), c_sqrt_approx(a), "sqrt_approx({a})");
        assert_eq!(silk_rand(a), c_rand(a), "rand({a})");
        assert_eq!(silk_lin2log(a), c_lin2log(a), "lin2log({a})");
    }
    for &a in &values {
        let wide = ((a as i64) << 32) | (rng.next_u64() & 0xFFFF_FFFF) as i64;
        for v in [wide, a as i64, a as u32 as i64, 0, -1, i64::MIN, i64::MAX] {
            assert_eq!(silk_clz64(v), c_clz64(v), "clz64({v})");
        }
    }
    for_pairs(11, |a, b| {
        assert_eq!(silk_min_int(a, b), a.min(b));
        assert_eq!(silk_min_32(a, b), a.min(b));
        assert_eq!(silk_max_int(a, b), a.max(b));
        assert_eq!(silk_max_32(a, b), a.max(b));
    });
    // The square root is within 10 % of the true one for results above 15.
    for x in [300i32, 1000, 65536, 1 << 20, 123_456_789, i32::MAX] {
        let exact = (x as f64).sqrt();
        let approx = silk_sqrt_approx(x) as f64;
        assert!((approx - exact).abs() <= 0.1 * exact, "sqrt({x}): {approx} against {exact}");
    }
}

#[test]
fn logarithm_and_its_inverse_match_libopus() {
    // Every input of the inverse that libopus treats on its own, and all
    // that lie between.
    for q7 in -300..4200 {
        assert_eq!(silk_log2lin(q7), c_log2lin(q7), "log2lin({q7})");
    }
    for q7 in [i32::MIN, -1, 0, 127, 128, 2047, 2048, 3966, 3967, i32::MAX] {
        assert_eq!(silk_log2lin(q7), c_log2lin(q7), "log2lin({q7})");
    }
    // What libopus returns outside the range the logarithm is meant for.
    assert_eq!(silk_lin2log(0), -128);
    assert_eq!(silk_lin2log(1), 0);
    assert_eq!(silk_lin2log(2), 128);
    assert_eq!(silk_lin2log(65536), 16 * 128);
    assert_eq!(silk_lin2log(-1), c_lin2log(-1));
    // The two are inverses of each other to within 1 %.
    for x in [1i32, 3, 10, 100, 1000, 65536, 1_000_000, 1 << 30] {
        let back = silk_log2lin(silk_lin2log(x)) as f64;
        assert!((back - x as f64).abs() <= 0.01 * x as f64 + 1.0, "{x} came back as {back}");
    }
}

/// The two divisions. libopus requires a divisor that is not zero and a
/// Q-domain of the result of at least 0 (above 0 for the inverse). Left out
/// here: `i32::MIN`, whose magnitude `silk_abs` cannot give, and results
/// that need a shift of more than 31, which libopus does not define.
#[test]
fn divisions_match_the_c_functions() {
    let ext: Vec<i32> = extremes().into_iter().filter(|&v| v != i32::MIN).collect();
    let mut cases = 0u32;
    let mut check_div = |a: i32, b: i32, qres: i32| {
        if b == 0 || a == i32::MIN || b == i32::MIN {
            return;
        }
        let a_headrm = c_clz32(c_abs(a)) - 1;
        let b_headrm = c_clz32(c_abs(b)) - 1;
        if -(29 + a_headrm - b_headrm - qres) > 31 {
            return;
        }
        assert_eq!(silk_div32_varq(a, b, qres), c_div32_varq(a, b, qres), "div32_varq({a}, {b}, {qres})");
        cases += 1;
    };
    for &a in &ext {
        for &b in &ext {
            for qres in [0, 1, 7, 14, 16, 24, 30, 31] {
                check_div(a, b, qres);
            }
        }
    }
    let mut rng = Rng(12);
    for _ in 0..RANDOM_CASES {
        check_div(rng.next_i32(), rng.next_i32(), rng.below(32));
    }
    assert!(cases > 300_000, "{cases} divisions compared");

    let mut cases = 0u32;
    let mut check_inv = |b: i32, qres: i32| {
        if b == 0 || b == i32::MIN {
            return;
        }
        let b_headrm = c_clz32(c_abs(b)) - 1;
        if -(61 - b_headrm - qres) > 31 {
            return;
        }
        assert_eq!(silk_inverse32_varq(b, qres), c_inverse32_varq(b, qres), "inverse32_varq({b}, {qres})");
        cases += 1;
    };
    for &b in &ext {
        for qres in 1..=62 {
            check_inv(b, qres);
        }
    }
    for _ in 0..RANDOM_CASES {
        check_inv(rng.next_i32(), 1 + rng.below(62));
    }
    assert!(cases > 150_000, "{cases} inverses compared");

    // The results are what they are meant to approximate, to about 14 bits.
    let near = |got: i32, exact: i32| (got - exact).abs() <= (exact.abs() >> 13) + 1;
    assert!(near(silk_div32_varq(1, 2, 16), 32768));
    assert!(near(silk_div32_varq(-3, 4, 16), -49152));
    assert!(near(silk_div32_varq(1_000_000, 7, 8), 36_571_428));
    assert!(near(silk_inverse32_varq(3, 30), (1 << 30) / 3));
    assert!(near(silk_inverse32_varq(-1000, 40), -1_099_511_627));
    // A quotient too large for 32 bits saturates like the shift it ends with.
    assert_eq!(silk_div32_varq(i32::MAX, 1, 16), c_div32_varq(i32::MAX, 1, 16));
    assert_eq!(silk_div32_varq(i32::MAX, 1, 16) & 0xFFFF, 0);
}
