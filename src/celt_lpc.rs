use crate::fixedvec::FixedVec;
use crate::pitch::pitch_xcorr;

pub fn lpc(lpc: &mut [f32], ac: &[f32], p: usize) {
    let mut error = ac[0];
    if error <= 1e-10 {
        for x in lpc.iter_mut() {
            *x = 0.0;
        }
        return;
    }

    for i in 0..p {
        let mut rr = 0.0f32;
        for j in 0..i {
            rr += lpc[j] * ac[i - j];
        }
        rr += ac[i + 1];
        let r = -rr / error;

        lpc[i] = r;
        for j in 0..i.div_ceil(2) {
            let tmp1 = lpc[j];
            let tmp2 = lpc[i - 1 - j];
            lpc[j] = tmp1 + r * tmp2;
            lpc[i - 1 - j] = tmp2 + r * tmp1;
        }

        error = error - r * r * error;

        if error <= 0.001 * ac[0] {
            break;
        }
    }
}

/// Largest `n` that `autocorr` accepts together with a window.
const AUTOCORR_WINDOWED_MAX: usize = 2880;

/// Autocorrelation of `x[..n]` for the lags 0 to `lag`, written to
/// `ac[..=lag]`.
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
    // `k + fast_n` on.
    for (k, out) in ac[..=lag].iter_mut().enumerate() {
        let mut d = 0.0f32;
        for (a, b) in xx[k + fast_n..].iter().zip(&xx[fast_n..]) {
            d += a * b;
        }
        *out += d;
    }
}

#[cfg(test)]
mod tests {
    #[cfg(not(feature = "std"))]
    use std::vec::Vec;
    use super::*;
    use crate::pitch::tests::{KINDS, Rng, assert_same, samples};

    /// `autocorr` as it was before: the windowed signal is built with
    /// `FixedVec::from_slice`, and the last products use indexing.
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
                xx[i] *= win[i];
                xx[n - 1 - i] *= win[i];
            }
        }
        let fast_n = n - lag;
        pitch_xcorr(&xx, &xx, ac, fast_n, lag + 1);
        for k in 0..=lag {
            let mut d = 0.0f32;
            for i in (k + fast_n)..n {
                d += xx[i] * xx[i - k];
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
}
