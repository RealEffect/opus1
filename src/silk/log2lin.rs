use crate::silk::macros::*;

/// libopus `silk_log2lin` (silk/log2lin.c): `2^(in_log_q7 / 128)`, the inverse
/// of `silk_lin2log`. 0 for a negative input and `i32::MAX` from 3967 on.
#[inline]
pub fn silk_log2lin(in_log_q7: i32) -> i32 {
    if in_log_q7 < 0 {
        return 0;
    }
    if in_log_q7 >= 3967 {
        return i32::MAX;
    }
    let out = silk_lshift(1, in_log_q7 >> 7);
    let frac_q7 = in_log_q7 & 0x7F;

    // Piece-wise parabolic approximation.
    let val = silk_smlawb(frac_q7, silk_smulbb(frac_q7, 128 - frac_q7), -174);
    if in_log_q7 < 2048 {
        silk_add_rshift32(out, silk_mul(out, val), 7)
    } else {
        silk_mla(out, out >> 7, val)
    }
}
