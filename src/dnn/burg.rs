//! The Burg LPC analysis of libopus `dnn/burg.c`, in double precision.
//! The feature analysis of deep PLC uses it through
//! `freq::burg_cepstral_analysis`.
//!
//! This is not the fixed-point Burg analysis of the SILK encoder
//! (`silk::lpc_analysis`).
//!
//! Rounding, as in the reference build (libopus 1.6.1 for 64-bit ARM, built
//! without loop vectorization; checked in its assembly):
//!
//! - The sums `C_first_row[k] -= x[n]·x[n-k-1]` and the `C_last_row` line
//!   multiply two floats in `f32`. The product is rounded to `f32`, then
//!   subtracted in `f64` (`deep_plc.md`, Review item 2).
//! - A product with an `f64` operand is fused with the sum it is written
//!   next to (`freq::fma`).
//! - The energies and inner products multiply two floats in `f64`. These
//!   products are exact, so only the order of the sums matters; it is the
//!   order of the C code, unrolled by 4.
//! - `sqrt` is the exact square root of IEEE 754, with and without `std`.
//!
//! The C code checks its intermediate values with `assert` (active in the
//! reference build). Here these checks are `debug_assert!`: they fail only
//! for input that makes libopus abort, such as NaN.

#[cfg(not(feature = "std"))]
use crate::compat::Math;

use super::freq::fma;

/// `MAX_FRAME_SIZE` (`dnn/burg.c:39`): the largest
/// `subfr_length · nb_subfr`.
pub const MAX_FRAME_SIZE: usize = 384;
/// `SILK_MAX_ORDER_LPC` (`dnn/burg.c:40`): the largest order.
pub const SILK_MAX_ORDER_LPC: usize = 16;
/// `FIND_LPC_COND_FAC` (`dnn/burg.c:41`): the white-noise fraction added to
/// the energy.
pub const FIND_LPC_COND_FAC: f32 = 1e-5;

/// The Burg analysis of `nb_subfr` stacked subframes of `subfr_length`
/// samples (libopus `silk_burg_analysis`, `dnn/burg.c:99-246`). Writes `d`
/// prediction coefficients to `a` and returns the residual energy (at
/// least 0). `min_inv_gain` limits the prediction gain.
///
/// Requires `a.len() >= d`, `d <= SILK_MAX_ORDER_LPC`,
/// `1 <= subfr_length`, `d <= subfr_length`,
/// `x.len() >= subfr_length · nb_subfr` and
/// `subfr_length · nb_subfr <= MAX_FRAME_SIZE`. It panics otherwise.
pub fn silk_burg_analysis(
    a: &mut [f32],
    x: &[f32],
    min_inv_gain: f32,
    subfr_length: usize,
    nb_subfr: usize,
    d: usize,
) -> f32 {
    assert!(subfr_length * nb_subfr <= MAX_FRAME_SIZE);
    assert!(subfr_length >= 1 && d <= SILK_MAX_ORDER_LPC && d <= subfr_length);
    let x = &x[..subfr_length * nb_subfr];
    let a = &mut a[..d];
    let min_inv_gain = min_inv_gain as f64;

    // The autocorrelations, added over the subframes.
    let mut c0 = silk_energy_flp(x);
    let mut c_first_row = [0.0f64; SILK_MAX_ORDER_LPC];
    for x_ptr in x.chunks_exact(subfr_length) {
        for n in 1..d + 1 {
            c_first_row[n - 1] += silk_inner_product_flp(&x_ptr[..subfr_length - n], &x_ptr[n..]);
        }
    }
    let mut c_last_row = c_first_row;

    // `C0 + FIND_LPC_COND_FAC * C0 + 1e-9f`.
    let start = fma(FIND_LPC_COND_FAC as f64, c0, c0) + 1e-9f32 as f64;
    let mut caf = [0.0f64; SILK_MAX_ORDER_LPC + 1];
    let mut cab = [0.0f64; SILK_MAX_ORDER_LPC + 1];
    caf[0] = start;
    cab[0] = start;
    let mut inv_gain = 1.0f64;
    let mut reached_max_gain = false;
    let mut af = [0.0f64; SILK_MAX_ORDER_LPC];

    for n in 0..d {
        // Update the first row of the correlation matrix (without its
        // first element), the last row (without its last element, stored
        // in reverse order), C * Af, and C * flipud(Af) (stored in reverse
        // order).
        for x_ptr in x.chunks_exact(subfr_length) {
            let mut tmp1 = x_ptr[n] as f64;
            let mut tmp2 = x_ptr[subfr_length - n - 1] as f64;
            for k in 0..n {
                c_first_row[k] -= (x_ptr[n] * x_ptr[n - k - 1]) as f64;
                c_last_row[k] -=
                    (x_ptr[subfr_length - n - 1] * x_ptr[subfr_length - n + k]) as f64;
                let atmp = af[k];
                tmp1 = fma(x_ptr[n - k - 1] as f64, atmp, tmp1);
                tmp2 = fma(x_ptr[subfr_length - n + k] as f64, atmp, tmp2);
            }
            for k in 0..=n {
                caf[k] = fma(-tmp1, x_ptr[n - k] as f64, caf[k]);
                cab[k] = fma(-tmp2, x_ptr[subfr_length - n + k - 1] as f64, cab[k]);
            }
        }
        let mut tmp1 = c_first_row[n];
        let mut tmp2 = c_last_row[n];
        for k in 0..n {
            let atmp = af[k];
            tmp1 = fma(c_last_row[n - k - 1], atmp, tmp1);
            tmp2 = fma(c_first_row[n - k - 1], atmp, tmp2);
        }
        caf[n + 1] = tmp1;
        cab[n + 1] = tmp2;

        // The numerator and the denominator of the next reflection
        // coefficient.
        let mut num = cab[n + 1];
        let mut nrg_b = cab[0];
        let mut nrg_f = caf[0];
        for k in 0..n {
            let atmp = af[k];
            num = fma(cab[n - k], atmp, num);
            nrg_b = fma(cab[k + 1], atmp, nrg_b);
            nrg_f = fma(caf[k + 1], atmp, nrg_f);
        }
        debug_assert!(nrg_f > 0.0);
        debug_assert!(nrg_b > 0.0);

        // The next reflection coefficient.
        let mut rc = -2.0 * num / (nrg_f + nrg_b);
        debug_assert!(rc > -1.0 && rc < 1.0);

        // Update the inverse prediction gain.
        let tmp1 = inv_gain * fma(-rc, rc, 1.0);
        if tmp1 <= min_inv_gain {
            // The maximum prediction gain is exceeded: choose the reflection
            // coefficient that reaches it exactly, with the original sign.
            rc = (1.0 - min_inv_gain / inv_gain).sqrt();
            if num > 0.0 {
                rc = -rc;
            }
            inv_gain = min_inv_gain;
            reached_max_gain = true;
        } else {
            inv_gain = tmp1;
        }

        // Update the AR coefficients.
        for k in 0..(n + 1) >> 1 {
            let tmp1 = af[k];
            let tmp2 = af[n - k - 1];
            af[k] = fma(rc, tmp2, tmp1);
            af[n - k - 1] = fma(rc, tmp1, tmp2);
        }
        af[n] = rc;

        if reached_max_gain {
            // The remaining coefficients are zero.
            af[n + 1..d].fill(0.0);
            break;
        }

        // Update C * Af and C * Ab.
        for k in 0..=n + 1 {
            let tmp1 = caf[k];
            caf[k] = fma(rc, cab[n + 1 - k], caf[k]);
            cab[n + 1 - k] = fma(rc, tmp1, cab[n + 1 - k]);
        }
    }

    let nrg_f = if reached_max_gain {
        for (ak, &f) in a.iter_mut().zip(af.iter()) {
            *ak = (-f) as f32;
        }
        // Subtract the energy of the first `d` samples of each subframe.
        for x_ptr in x.chunks_exact(subfr_length) {
            c0 -= silk_energy_flp(&x_ptr[..d]);
        }
        // An approximation of the residual energy.
        c0 * inv_gain
    } else {
        // The residual energy, and the coefficients as `f32`.
        let mut nrg_f = caf[0];
        let mut tmp1 = 1.0f64;
        for (k, ak) in a.iter_mut().enumerate() {
            let atmp = af[k];
            nrg_f = fma(caf[k + 1], atmp, nrg_f);
            tmp1 = fma(atmp, atmp, tmp1);
            *ak = (-atmp) as f32;
        }
        fma(-(FIND_LPC_COND_FAC as f64 * c0), tmp1, nrg_f)
    };

    // `MAX32(0, (float)nrg_f)`.
    let nrg = nrg_f as f32;
    if 0.0 > nrg { 0.0 } else { nrg }
}

/// `silk_energy_FLP` (`dnn/burg.c:44-68`): the sum of squares in `f64`,
/// unrolled by 4 as in the C code.
fn silk_energy_flp(data: &[f32]) -> f64 {
    let mut result = 0.0f64;
    let (blocks, rest) = data.as_chunks::<4>();
    for b in blocks {
        let (d0, d1, d2, d3) = (b[0] as f64, b[1] as f64, b[2] as f64, b[3] as f64);
        result += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
    }
    for &v in rest {
        let v = v as f64;
        result += v * v;
    }
    debug_assert!(result >= 0.0);
    result
}

/// `silk_inner_product_FLP` (`dnn/burg.c:71-95`): the inner product of
/// `data1` and the first `data1.len()` values of `data2` in `f64`, unrolled
/// by 4 as in the C code.
fn silk_inner_product_flp(data1: &[f32], data2: &[f32]) -> f64 {
    let data2 = &data2[..data1.len()];
    let mut result = 0.0f64;
    let (blocks1, rest1) = data1.as_chunks::<4>();
    let (blocks2, rest2) = data2.as_chunks::<4>();
    for (a, b) in blocks1.iter().zip(blocks2.iter()) {
        result += a[0] as f64 * b[0] as f64
            + a[1] as f64 * b[1] as f64
            + a[2] as f64 * b[2] as f64
            + a[3] as f64 * b[3] as f64;
    }
    for (&a, &b) in rest1.iter().zip(rest2.iter()) {
        result += a as f64 * b as f64;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn energy_and_inner_product_follow_the_c_order() {
        let x: [f32; 7] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        assert_eq!(silk_energy_flp(&x), 140.0);
        assert_eq!(silk_inner_product_flp(&x[..6], &x[1..]), 1.0 * 2.0 + 6.0 + 12.0 + 20.0 + 30.0 + 42.0);
        assert_eq!(silk_energy_flp(&[]), 0.0);
    }

    #[test]
    fn a_first_order_process_gives_its_coefficient() {
        // x[n] = 0.9 * x[n-1] + small noise: the first coefficient is close
        // to 0.9 (libopus returns the predictor with this sign: A[0] > 0
        // predicts x[n] from +x[n-1]).
        let mut x = [0.0f32; 160];
        let mut state = 1u32;
        let mut prev = 0.0f32;
        for v in x.iter_mut() {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let noise = (state >> 8) as f32 / (1u32 << 24) as f32 - 0.5;
            prev = 0.9 * prev + noise;
            *v = prev;
        }
        let mut a = [0.0f32; 16];
        let nrg = silk_burg_analysis(&mut a, &x, 1e-3, 160, 1, 16);
        assert!((a[0] - 0.9).abs() < 0.1, "{a:?}");
        assert!(nrg > 0.0);
    }

    #[test]
    fn a_pure_sine_reaches_the_maximum_gain() {
        // An undamped sine is perfectly predictable with order 2, so the
        // gain limit stops the recursion and the later coefficients are 0.
        let mut x = [0.0f32; 80];
        for (n, v) in x.iter_mut().enumerate() {
            *v = (0.3 * n as f64).sin() as f32 * 1000.0;
        }
        let mut a = [1.0f32; 16];
        let nrg = silk_burg_analysis(&mut a, &x, 1e-3, 80, 1, 16);
        assert!(nrg >= 0.0 && nrg.is_finite());
        assert!(a.iter().all(|v| v.is_finite()), "{a:?}");
        assert!(a[8..].iter().all(|&v| v == 0.0), "{a:?}");
    }

    #[test]
    fn silence_gives_zero_coefficients() {
        let mut a = [1.0f32; 16];
        let nrg = silk_burg_analysis(&mut a, &[0.0; 79], 1e-3, 79, 1, 16);
        assert!(a.iter().all(|&v| v == 0.0), "{a:?}");
        // Only the 1e-9 conditioning term is left.
        assert_eq!(nrg, 1e-9f32 as f64 as f32);
    }
}
