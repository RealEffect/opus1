use crate::silk::macros::*;

const A_FB1_20: i32 = 5394 << 1;
const A_FB1_21: i32 = -24290;

/// `y + ((y * A_FB1_21) >> 16)`, the all-pass step for the even samples
/// (`silk_smlawb(y, y, A_FB1_21)`), as one multiply:
/// `y * 65536 + y * A_FB1_21` is `y * (65536 + A_FB1_21)`, and the first
/// term is a multiple of 65536.
#[inline(always)]
fn all_pass_even(y: i32) -> i32 {
    ((y as i64 * (65536 + A_FB1_21) as i64) >> 16) as i32
}

/// `(y * A_FB1_20) >> 16`, the all-pass step for the odd samples.
#[inline(always)]
fn all_pass_odd(y: i32) -> i32 {
    silk_smulwb(y, A_FB1_20)
}

/// The low-band and the high-band output sample of one pair of input
/// samples. `out_1` and `out_2` are the outputs of the two all-pass filters.
#[inline(always)]
fn band_outputs(out_1: i32, out_2: i32) -> (i16, i16) {
    (
        silk_sat16(out_2.wrapping_add(out_1).wrapping_add(1024) >> 11) as i16,
        silk_sat16(out_2.wrapping_sub(out_1).wrapping_add(1024) >> 11) as i16,
    )
}

/// Split the first `n` samples of `input` into a low band and a high band
/// at half the sampling rate (libopus `silk_ana_filt_bank_1`): two first-order
/// all-pass filters, one for the even and one for the odd samples.
///
/// `s` holds the two filter states. `out_l` and `out_h` each receive `n / 2`
/// samples. The function does nothing when `n` is outside 2..=4096 or a
/// buffer is too short.
///
/// libopus writes each filter like this, per pair of samples (`in` is the
/// sample shifted left by 10, `x` is the all-pass step of `y`):
///
/// ```text
/// y = in - s;   out = s + x;   s = in + x
/// ```
///
/// The state `s` has to wait for its previous value through subtract,
/// multiply, shift and add. The loop below computes the same values in
/// another order. It carries `y` from pair to pair instead of the state:
///
/// ```text
/// out = in - y + x;   y' = (in' - in) - x
/// ```
///
/// (`in'` is the next sample of the same filter.) Now only multiply, shift
/// and one subtract lie between one `y` and the next. All additions wrap
/// around, so the order does not change any result.
pub fn silk_ana_filt_bank_1(
    input: &[i16],
    s: &mut [i32],
    out_l: &mut [i16],
    out_h: &mut [i16],
    n: usize,
) {
    if !(2..=4096).contains(&n) || n > input.len() {
        return;
    }

    let n2 = n >> 1;

    if n2 > out_l.len() || n2 > out_h.len() {
        return;
    }

    let (pairs, _) = input[..2 * n2].as_chunks::<2>();
    let (out_l, out_h) = (&mut out_l[..n2], &mut out_h[..n2]);
    let scaled = |pair: &[i16; 2]| [(pair[0] as i32) << 10, (pair[1] as i32) << 10];

    // All pairs but the last: the next pair is known.
    let [mut in_even, mut in_odd] = scaled(&pairs[0]);
    let mut y_even = in_even.wrapping_sub(s[0]);
    let mut y_odd = in_odd.wrapping_sub(s[1]);
    for ((next, low), high) in pairs[1..]
        .iter()
        .zip(out_l.iter_mut())
        .zip(out_h.iter_mut())
    {
        let [next_even, next_odd] = scaled(next);
        let x_even = all_pass_even(y_even);
        let x_odd = all_pass_odd(y_odd);
        let out_1 = in_even.wrapping_sub(y_even).wrapping_add(x_even);
        let out_2 = in_odd.wrapping_sub(y_odd).wrapping_add(x_odd);
        (*low, *high) = band_outputs(out_1, out_2);

        y_even = next_even.wrapping_sub(in_even).wrapping_sub(x_even);
        y_odd = next_odd.wrapping_sub(in_odd).wrapping_sub(x_odd);
        in_even = next_even;
        in_odd = next_odd;
    }

    // The last pair, and the states for the next call.
    let x_even = all_pass_even(y_even);
    let x_odd = all_pass_odd(y_odd);
    let out_1 = in_even.wrapping_sub(y_even).wrapping_add(x_even);
    let out_2 = in_odd.wrapping_sub(y_odd).wrapping_add(x_odd);
    (out_l[n2 - 1], out_h[n2 - 1]) = band_outputs(out_1, out_2);
    s[0] = in_even.wrapping_add(x_even);
    s[1] = in_odd.wrapping_add(x_odd);
}
