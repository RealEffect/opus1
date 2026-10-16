#[inline(always)]
pub fn silk_rshift64(a: i64, shift: i32) -> i64 {
    a >> shift
}

/// libopus `silk_LSHIFT_SAT32` (silk/SigProc_FIX.h): limit `a` to the values
/// that survive the shift, then shift.
///
/// The largest result is `(i32::MAX >> shift) << shift`, which is
/// `0x7FFF_FFF0` for a shift of 4. It is not `i32::MAX`. The smallest result
/// is `i32::MIN`.
///
/// libopus allows shifts from 0 to 31. A shift outside that range is taken
/// as the nearest one inside it, so that no input can panic.
#[inline(always)]
pub fn silk_lshift_sat32(a: i32, shift: i32) -> i32 {
    let shift = shift.clamp(0, 31);
    silk_lshift(
        silk_limit_32(a, i32::MIN >> shift, i32::MAX >> shift),
        shift,
    )
}

/// libopus `silk_ADD_POS_SAT32` (silk/SigProc_FIX.h): add two values that
/// are not negative. A sum beyond `i32::MAX` gives `i32::MAX`.
///
/// The macro tests the sign bit of the 32-bit sum. For inputs that are not
/// negative that is the overflow test. With a negative input, any sum below
/// zero also gives `i32::MAX`, as in libopus.
#[inline(always)]
pub fn silk_add_pos_sat32(a: i32, b: i32) -> i32 {
    let sum = (a as u32).wrapping_add(b as u32);
    if sum & 0x8000_0000 != 0 {
        i32::MAX
    } else {
        sum as i32
    }
}

#[inline(always)]
pub fn silk_smlaww_64(a: i64, b: i64, c: i32) -> i64 {
    a.wrapping_add((b * (c as i64)) >> 16)
}

#[inline(always)]
pub fn silk_smulww(a: i32, b: i32) -> i32 {
    (((a as i64) * (b as i64)) >> 16) as i32
}

#[inline(always)]
pub fn silk_smlaww(a: i32, b: i32, c: i32) -> i32 {
    a.wrapping_add(silk_smulww(b, c))
}

/// libopus `silk_RSHIFT_ROUND` (silk/SigProc_FIX.h): `a / 2^shift`, rounded
/// to the nearest integer, halfway cases towards plus infinity.
///
/// libopus requires `0 < shift < 32`. For a shift of 31 the result is -1, 0
/// or 1. Outside the range of libopus this function shifts left for a shift
/// of zero or less, and returns 0 for a shift of 32 or more.
#[inline(always)]
pub fn silk_rshift_round(a: i32, shift: i32) -> i32 {
    if shift <= 0 {
        return a << -shift;
    }
    if shift >= 32 {
        return 0;
    }
    // The same value as the C expression
    // `shift == 1 ? (a >> 1) + (a & 1) : ((a >> (shift - 1)) + 1) >> 1`.
    (a >> shift) + ((a >> (shift - 1)) & 1)
}

/// libopus `silk_RSHIFT_ROUND64`: [`silk_rshift_round`] for 64-bit values,
/// with `0 < shift < 64`. A shift of zero or less shifts left.
#[inline(always)]
pub fn silk_rshift_round64(a: i64, shift: i32) -> i64 {
    if shift <= 0 {
        return a << -shift;
    }
    (a >> shift) + ((a >> (shift - 1)) & 1)
}

#[inline(always)]
pub fn silk_add_rshift(a: i32, b: i32, shift: i32) -> i32 {
    a.wrapping_add(b >> shift)
}

#[inline(always)]
pub fn silk_div32_16(a: i32, b: i32) -> i32 {
    a / b
}

#[inline(always)]
pub fn silk_div32_varq(a32: i32, b32: i32, qres: i32) -> i32 {
    debug_assert!(b32 != 0);
    debug_assert!(qres >= 0);

    let a_headrm = if a32 == 0 {
        31
    } else {
        (a32.wrapping_abs().leading_zeros() as i32) - 1
    };
    let a32_nrm = if a_headrm >= 0 {
        a32 << a_headrm
    } else {
        a32 >> (-a_headrm)
    };
    let b_headrm = if b32 == 0 {
        31
    } else {
        (b32.wrapping_abs().leading_zeros() as i32) - 1
    };
    let b32_nrm = if b_headrm >= 0 {
        b32 << b_headrm
    } else {
        b32 >> (-b_headrm)
    };

    let b32_inv = silk_div32_16(i32::MAX >> 2, b32_nrm >> 16);

    let mut result = silk_smulwb(a32_nrm, b32_inv);

    let a32_nrm2 =
        a32_nrm.wrapping_sub((silk_smmul(b32_nrm, result) as u32).wrapping_shl(3) as i32);

    result = silk_smlawb(result, a32_nrm2, b32_inv);

    let lshift = 29 + a_headrm - b_headrm - qres;
    if lshift < 0 {
        silk_lshift_sat32(result, -lshift)
    } else if lshift < 32 {
        result >> lshift
    } else {
        0
    }
}

#[inline(always)]
pub fn silk_div32(a: i32, b: i32) -> i32 {
    a / b
}

#[inline(always)]
pub fn silk_limit(a: i32, limit_low: i32, limit_high: i32) -> i32 {
    silk_limit_32(a, limit_low, limit_high)
}

#[inline(always)]
pub fn silk_limit_int(a: i32, limit_low: i32, limit_high: i32) -> i32 {
    silk_limit_32(a, limit_low, limit_high)
}

#[inline(always)]
pub fn silk_smulbb(a: i32, b: i32) -> i32 {
    (a as i16 as i32) * (b as i16 as i32)
}

#[inline(always)]
pub fn silk_smulwb(a: i32, b: i32) -> i32 {
    (((a as i64) * (b as i16 as i64)) >> 16) as i32
}

#[inline(always)]
pub fn silk_smlawb(a: i32, b: i32, c: i32) -> i32 {
    a.wrapping_add(silk_smulwb(b, c))
}

/// libopus `silk_SMULTT`: the top 16 bits of `a` times the top 16 bits of
/// `b`. The product wraps at 32 bits.
#[inline(always)]
pub(crate) fn silk_smultt(a: i32, b: i32) -> i32 {
    (a >> 16).wrapping_mul(b >> 16)
}

#[inline(always)]
pub fn silk_smulwt(a: i32, b: i32) -> i32 {
    ((a as i64 * (b as i64 >> 16)) >> 16) as i32
}

#[inline(always)]
pub fn silk_smlawt(a: i32, b: i32, c: i32) -> i32 {
    a.wrapping_add(silk_smulwt(b, c))
}

#[inline(always)]
pub fn silk_smlabb(a: i32, b: i32, c: i32) -> i32 {
    a.wrapping_add((b as i16 as i32).wrapping_mul(c as i16 as i32))
}

#[inline(always)]
pub fn silk_mla(a: i32, b: i32, c: i32) -> i32 {
    a.wrapping_add(b.wrapping_mul(c))
}

#[inline(always)]
pub fn silk_mul(a: i32, b: i32) -> i32 {
    a.wrapping_mul(b)
}

#[inline(always)]
pub fn silk_rshift(a: i32, shift: i32) -> i32 {
    a >> shift
}

#[inline(always)]
pub fn silk_rshift32(a: i32, shift: i32) -> i32 {
    a >> shift
}

#[inline(always)]
pub fn silk_lshift(a: i32, shift: i32) -> i32 {
    a << shift
}

#[inline(always)]
pub fn silk_sub_rshift32(a: i32, b: i32, shift: i32) -> i32 {
    a.wrapping_sub(b >> shift)
}

#[inline(always)]
pub fn silk_smulll(a: i64, b: i64) -> i64 {
    a.wrapping_mul(b)
}

#[inline(always)]
pub fn silk_smull(a: i32, b: i32) -> i64 {
    (a as i64) * (b as i64)
}

#[inline(always)]
pub fn silk_smmul(a: i32, b: i32) -> i32 {
    (((a as i64) * (b as i64)) >> 32) as i32
}

#[inline(always)]
pub fn silk_clz32(a: i32) -> i32 {
    a.leading_zeros() as i32
}

#[inline(always)]
pub fn silk_clz64(a: i64) -> i32 {
    a.leading_zeros() as i32
}

#[inline(always)]
pub fn silk_inverse32_varq(b32: i32, qres: i32) -> i32 {
    if b32 == 0 {
        return i32::MAX;
    }

    let b_headrm = (b32.wrapping_abs().leading_zeros() as i32) - 1;
    let b32_nrm = if b_headrm >= 0 {
        b32 << b_headrm
    } else {
        b32 >> (-b_headrm)
    };

    let b32_inv = silk_div32_16(i32::MAX >> 2, b32_nrm >> 16);

    let result = b32_inv << 16;

    let err_q32 = ((1i32 << 29) - silk_smulwb(b32_nrm, b32_inv)) << 3;
    let result = silk_smlaww(result, err_q32, b32_inv);

    let lshift = 61 - b_headrm - qres;
    if lshift <= 0 {
        silk_lshift_sat32(result, -lshift)
    } else if lshift < 32 {
        result >> lshift
    } else {
        0
    }
}

#[inline(always)]
pub fn silk_clz_frac(input: i32, lz: &mut i32, frac_q7: &mut i32) {
    let lzeros = silk_clz32(input);
    *lz = lzeros;
    *frac_q7 = (input.rotate_right((24 - lzeros) as u32)) & 0x7f;
}

#[inline(always)]
pub fn silk_sqrt_approx(x: i32) -> i32 {
    let mut y: i32;
    let mut lz = 0;
    let mut frac_q7 = 0;

    if x <= 0 {
        return 0;
    }

    silk_clz_frac(x, &mut lz, &mut frac_q7);

    if (lz & 1) != 0 {
        y = 32768;
    } else {
        y = 46214;
    }

    y >>= lz >> 1;

    y = silk_smlawb(y, y, silk_smulbb(213, frac_q7));

    y
}

#[inline(always)]
pub fn silk_min_int(a: i32, b: i32) -> i32 {
    if a < b { a } else { b }
}

#[inline(always)]
pub fn silk_max_int(a: i32, b: i32) -> i32 {
    if a > b { a } else { b }
}

#[inline(always)]
pub fn silk_max_32(a: i32, b: i32) -> i32 {
    silk_max_int(a, b)
}

#[inline(always)]
pub fn silk_sat16(a: i32) -> i32 {
    if a > i16::MAX as i32 {
        i16::MAX as i32
    } else if a < i16::MIN as i32 {
        i16::MIN as i32
    } else {
        a
    }
}

#[inline(always)]
pub fn silk_add_sat16(a: i16, b: i16) -> i16 {
    let res = (a as i32) + (b as i32);
    if res > i16::MAX as i32 {
        i16::MAX
    } else if res < i16::MIN as i32 {
        i16::MIN
    } else {
        res as i16
    }
}

#[inline(always)]
pub fn silk_add_sat32(a: i32, b: i32) -> i32 {
    let res = (a as i64) + (b as i64);
    if res > i32::MAX as i64 {
        i32::MAX
    } else if res < i32::MIN as i64 {
        i32::MIN
    } else {
        res as i32
    }
}

#[inline(always)]
pub fn silk_rand(seed: i32) -> i32 {
    (907633515u32.wrapping_add((seed as u32).wrapping_mul(196314165u32))) as i32
}

#[inline(always)]
pub fn silk_add32_ovflw(a: i32, b: i32) -> i32 {
    a.wrapping_add(b)
}

#[inline(always)]
pub fn silk_sub32_ovflw(a: i32, b: i32) -> i32 {
    a.wrapping_sub(b)
}

#[inline(always)]
pub fn silk_add32(a: i32, b: i32) -> i32 {
    a.wrapping_add(b)
}

#[inline(always)]
pub fn silk_sub32(a: i32, b: i32) -> i32 {
    a.wrapping_sub(b)
}

#[inline(always)]
pub fn silk_sub_sat32(a1: i32, a2: i32) -> i32 {
    let res = (a1 as i64) - (a2 as i64);
    if res > i32::MAX as i64 {
        i32::MAX
    } else if res < i32::MIN as i64 {
        i32::MIN
    } else {
        res as i32
    }
}

#[inline(always)]
pub fn silk_sub_lshift32(a: i32, b: i32, shift: i32) -> i32 {
    a.wrapping_sub(b << shift)
}

/// libopus `silk_LIMIT` (silk/SigProc_FIX.h): bring `a` into the range
/// between the two limits. The limits may be given in either order.
#[inline(always)]
pub fn silk_limit_32(a: i32, limit1: i32, limit2: i32) -> i32 {
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

#[inline(always)]
pub fn silk_add_lshift32(a: i32, b: i32, shift: i32) -> i32 {
    a.wrapping_add(b << shift)
}

#[inline(always)]
pub fn silk_min_32(a: i32, b: i32) -> i32 {
    if a < b { a } else { b }
}

#[inline(always)]
pub fn silk_add_rshift32(a: i32, b: i32, shift: i32) -> i32 {
    a.wrapping_add(b >> shift)
}

/// libopus `silk_lin2log` (silk/lin2log.c): approximately
/// `128 * log2(in_lin)`.
///
/// The input is meant to be positive. libopus does not check it, and
/// neither does this function: 0 gives -128, and a negative value is read
/// as a number with no leading zero bit.
#[inline(always)]
pub fn silk_lin2log(in_lin: i32) -> i32 {
    let mut lz = 0;
    let mut frac_q7 = 0;
    silk_clz_frac(in_lin, &mut lz, &mut frac_q7);

    // Piece-wise parabolic approximation.
    silk_add_lshift32(
        silk_smlawb(frac_q7, silk_mul(frac_q7, 128 - frac_q7), 179),
        31 - lz,
        7,
    )
}

/// libopus `SILK_FIX_CONST( c, q )` (silk/SigProc_FIX.h): `c` in Q`q`,
/// rounded to the nearest integer. The C macro is
/// `(opus_int32)( c * ( (opus_int64)1 << q ) + 0.5 )`, computed in double
/// precision.
///
/// The 0.5 matters: `VARIABLE_HP_SMTH_COEF1` (0.1) in Q16 is 6554 with it
/// and 6553 without, and the input high-pass filter of the encoder then
/// changes its cutoff in other frames than libopus.
#[inline(always)]
pub(crate) const fn silk_fix_const(c: f64, q: u32) -> i32 {
    (c * (1i64 << q) as f64 + 0.5) as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The integers libopus's `SILK_FIX_CONST` gives for the constants this
    /// crate converts with `silk_fix_const`.
    #[test]
    fn fix_const_gives_the_libopus_integers() {
        assert_eq!(silk_fix_const(0.8, 16), 52429);
        assert_eq!(silk_fix_const(0.7, 16), 45875);
        assert_eq!(silk_fix_const(0.2, 16), 13107);
        assert_eq!(silk_fix_const(0.1f32 as f64, 16), 6554);
        assert_eq!(silk_fix_const(0.4f32 as f64, 7), 51);
        assert_eq!(silk_fix_const(60.0, 16), 3932160);
        assert_eq!(silk_fix_const(250.0f32 as f64 / 6.0, 7), 5333);
        assert_eq!(silk_fix_const(1.5 * core::f64::consts::PI / 1000.0, 19), 2471);
        // C converts to an integer by cutting off the fraction.
        assert_eq!(silk_fix_const(-0.004, 13), -32);
    }
}
