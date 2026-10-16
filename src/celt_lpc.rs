//! Linear prediction of CELT (libopus celt/celt_lpc.c, float build): the
//! autocorrelation, the Levinson-Durbin recursion and the two filters of the
//! loss concealment.
//!
//! Every product is added with `mathops::mac`, which rounds as libopus does
//! on this processor (one fused multiply-add on 64-bit ARM).

use crate::fixedvec::FixedVec;
use crate::mathops::mac;
use crate::pitch::pitch_xcorr;

/// libopus `_celt_lpc`: the linear predictor of order `p` for the
/// autocorrelation `ac[..=p]`, by the Levinson-Durbin recursion, written to
/// `lpc[..p]`.
///
/// The coefficients start at 0. When `ac[0]` is not above 1e-10 they stay
/// 0. The recursion stops once the prediction error is down to 0.001 of
/// `ac[0]` (30 dB of prediction gain); the coefficients of the higher
/// orders then stay 0.
pub fn lpc(lpc: &mut [f32], ac: &[f32], p: usize) {
    let lpc = &mut lpc[..p];
    lpc.fill(0.0);
    let mut error = ac[0];
    // C `if (ac[0] > 1e-10f)`: not-a-number gives zeros as well.
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    if !(ac[0] > 1e-10) {
        return;
    }

    for i in 0..p {
        // Sum up this iteration's reflection coefficient.
        let mut rr = 0.0f32;
        for j in 0..i {
            rr = mac(rr, lpc[j], ac[i - j]);
        }
        rr += ac[i + 1];
        let r = -(rr / error);

        // Update the coefficients and the prediction error.
        lpc[i] = r;
        for j in 0..(i + 1) >> 1 {
            let tmp1 = lpc[j];
            let tmp2 = lpc[i - 1 - j];
            lpc[j] = mac(tmp1, r, tmp2);
            lpc[i - 1 - j] = mac(tmp2, r, tmp1);
        }
        error = mac(error, -(r * r), error);

        // "Bail out once we get 30 dB gain"
        if error <= 0.001 * ac[0] {
            break;
        }
    }
}

/// Largest `n` that `autocorr` accepts together with a window.
const AUTOCORR_WINDOWED_MAX: usize = 2880;

/// libopus `_celt_autocorr`: the autocorrelation of `x[..n]` for the lags 0
/// to `lag`, written to `ac[..=lag]`.
///
/// With a window, the first and the last `overlap` samples are multiplied by
/// `window[..overlap]` first (the last ones in reverse order). That needs a
/// copy of the signal, which is limited to 2880 samples.
pub fn autocorr(
    x: &[f32],
    ac: &mut [f32],
    window: Option<&[f32]>,
    overlap: usize,
    lag: usize,
    n: usize,
) {
    match window {
        Some(win) => autocorr_windowed(x, ac, win, overlap, lag, n),
        None => autocorr_of(&x[..n], ac, lag),
    }
}

/// `autocorr` with a window. This is a separate function, never inlined, so
/// that the copy of the signal (11.5 KB) takes stack space only in calls that
/// use a window.
#[inline(never)]
fn autocorr_windowed(x: &[f32], ac: &mut [f32], win: &[f32], overlap: usize, lag: usize, n: usize) {
    if x.len() < n {
        return;
    }
    // Built in place, and only `n` samples of it are written.
    // (`FixedVec::from_slice` builds the buffer and then copies all 11.5 KB
    // to its final place; `extend_from_slice` stores one sample at a time.)
    let mut xx: FixedVec<f32, AUTOCORR_WINDOWED_MAX> = FixedVec::new();
    xx.resize(n, 0.0);
    xx.copy_from_slice(&x[..n]);
    for i in 0..overlap {
        xx[i] *= win[i];
        xx[n - 1 - i] *= win[i];
    }
    autocorr_of(&xx, ac, lag);
}

/// Autocorrelation of all of `xx` for the lags 0 to `lag`.
fn autocorr_of(xx: &[f32], ac: &mut [f32], lag: usize) {
    let n = xx.len();
    let fast_n = n - lag;

    // The first `fast_n` products of every lag, several lags at a time.
    pitch_xcorr(xx, xx, ac, fast_n, lag + 1);

    // Lag `k` has `lag - k` more products: xx[i] * xx[i - k] for i from
    // `k + fast_n` on. They are summed from 0 and then added to the rest.
    for (k, out) in ac[..=lag].iter_mut().enumerate() {
        let mut d = 0.0f32;
        for (&a, &b) in xx[k + fast_n..].iter().zip(&xx[fast_n..]) {
            d = mac(d, a, b);
        }
        *out += d;
    }
}

/// libopus `celt_fir` (all-zero filter, celt/celt_lpc.c):
/// `y[i] = x[at + i] + sum over j of num[j] * x[at + i - 1 - j]`.
///
/// The `num.len()` samples before `x[at]` are the filter's history. The
/// products are added from the oldest sample to the newest, as in libopus.
pub(crate) fn celt_fir(x: &[f32], at: usize, num: &[f32], y: &mut [f32]) {
    let ord = num.len();
    for (i, out) in y.iter_mut().enumerate() {
        let hist = &x[at + i - ord..at + i];
        let mut sum = x[at + i];
        for j in 0..ord {
            sum = mac(sum, num[ord - 1 - j], hist[j]);
        }
        *out = sum;
    }
}

/// libopus `celt_iir` (all-pole filter, celt/celt_lpc.c), run in place over
/// `buf[at..at + n]`: `buf[at + i] -= sum over j of den[j] * buf[at + i - 1 - j]`.
///
/// The `den.len()` samples before `buf[at]` are the filter's memory.
/// libopus computes four outputs at a time: each one first adds the
/// products with the samples before the group, oldest first, then the
/// products with the outputs of its own group, newest first. The same order
/// is kept here so that rounding matches.
pub(crate) fn celt_iir(buf: &mut [f32], at: usize, n: usize, den: &[f32]) {
    let ord = den.len();
    let mut i = 0;
    while i + 4 <= n {
        let p = at + i;
        let mut sum = [buf[p], buf[p + 1], buf[p + 2], buf[p + 3]];
        for (k, s) in sum.iter_mut().enumerate() {
            for j in 0..ord - k {
                *s = mac(*s, den[ord - 1 - j], -buf[p - ord + j + k]);
            }
        }
        buf[p] = sum[0];
        sum[1] = mac(sum[1], -buf[p], den[0]);
        buf[p + 1] = sum[1];
        sum[2] = mac(sum[2], -buf[p + 1], den[0]);
        sum[2] = mac(sum[2], -buf[p], den[1]);
        buf[p + 2] = sum[2];
        sum[3] = mac(sum[3], -buf[p + 2], den[0]);
        sum[3] = mac(sum[3], -buf[p + 1], den[1]);
        sum[3] = mac(sum[3], -buf[p], den[2]);
        buf[p + 3] = sum[3];
        i += 4;
    }
    // The concealment always filters a multiple of four samples. This loop
    // covers any other length.
    while i < n {
        let p = at + i;
        let mut sum = buf[p];
        for j in 0..ord {
            sum = mac(sum, den[ord - 1 - j], -buf[p - ord + j]);
        }
        buf[p] = sum;
        i += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pitch::tests::{KINDS, Rng, assert_same, samples};
    #[cfg(not(feature = "std"))]
    use std::vec::Vec;

    /// `autocorr` written as libopus writes `_celt_autocorr`: a windowed copy
    /// built one sample at a time, the cross-correlation, and the last
    /// products of each lag summed with `mac` and added.
    fn autocorr_plain(
        x: &[f32],
        ac: &mut [f32],
        window: Option<&[f32]>,
        overlap: usize,
        lag: usize,
        n: usize,
    ) {
        let mut xx = x[..n].to_vec();
        if let Some(win) = window {
            for i in 0..overlap {
                xx[i] = x[i] * win[i];
                xx[n - i - 1] = x[n - i - 1] * win[i];
            }
        }
        let fast_n = n - lag;
        pitch_xcorr(&xx, &xx, ac, fast_n, lag + 1);
        for k in 0..=lag {
            let mut d = 0.0f32;
            for i in (k + fast_n)..n {
                d = mac(d, xx[i], xx[i - k]);
            }
            ac[k] += d;
        }
    }

    #[test]
    fn autocorr_matches_the_plain_version() {
        let mut rng = Rng::new(28);
        let win: Vec<f32> = (0..120).map(|i| ((i as f32 + 0.5) / 120.0).sin()).collect();
        for kind in KINDS {
            for (n, lag, overlap) in [
                (5usize, 4usize, 0usize),
                (24, 4, 0),
                (25, 24, 0),
                (64, 8, 20),
                (240, 24, 120),
                (988, 4, 0),
                (1024, 24, 120),
                (2880, 24, 120),
            ] {
                let x = samples(&mut rng, n, kind);
                for window in [None, Some(&win[..])] {
                    if window.is_none() && overlap > 0 {
                        continue;
                    }
                    let mut got = vec![f32::NAN; lag + 1];
                    autocorr(&x, &mut got, window, overlap, lag, n);
                    let mut want = vec![f32::NAN; lag + 1];
                    autocorr_plain(&x, &mut want, window, overlap, lag, n);
                    assert_same(
                        &got,
                        &want,
                        &format!("{kind:?} n {n} lag {lag} window {}", window.is_some()),
                    );
                }
            }
        }
    }

    /// `lpc` stops at 30 dB of prediction gain and leaves the higher orders
    /// at 0, and gives all zeros for a signal without energy.
    #[test]
    fn lpc_clears_the_orders_it_does_not_reach() {
        // A first-order process: the second coefficient and those after it
        // add nothing, and the recursion stops after the first one.
        let ac = [1.0f32, 0.9999, 0.9998, 0.9997, 0.9996];
        let mut coef = [7.0f32; 4];
        lpc(&mut coef, &ac, 4);
        assert!(coef[0] < -0.99);
        assert!(coef[1..].iter().all(|&c| c == 0.0), "{coef:?}");

        let mut coef = [7.0f32; 4];
        lpc(&mut coef, &[0.0; 5], 4);
        assert_eq!(coef, [0.0; 4]);
    }
}
