//! `silk_nlsf_vq` computes the weighted error of a codebook vector in two
//! loops of independent steps for the LPC orders 10 and 16. These tests
//! compare it with the loop as libopus writes it, which carries a value from
//! one coefficient to the next.

use opus_rs::silk::macros::silk_smulbb;
use opus_rs::silk::nlsf::silk_nlsf_vq;
use opus_rs::silk::tables_nlsf::{SILK_NLSF_CB_NB_MB, SILK_NLSF_CB_WB};

/// libopus `silk_NLSF_VQ`, line by line.
fn reference(
    err_q24: &mut [i32],
    in_q15: &[i16],
    pcb_q8: &[u8],
    pwght_q9: &[i16],
    k: usize,
    lpc_order: usize,
) {
    for i in 0..k {
        let mut sum_error_q24 = 0i32;
        let mut pred_q24 = 0i32;
        let cb_ptr = &pcb_q8[i * lpc_order..];
        let w_ptr = &pwght_q9[i * lpc_order..];

        for m in (0..lpc_order).step_by(2).rev() {
            let diff_q15 = (in_q15[m + 1] as i32) - ((cb_ptr[m + 1] as i32) << 7);
            let diffw_q24 = silk_smulbb(diff_q15, w_ptr[m + 1] as i32);
            sum_error_q24 = sum_error_q24.wrapping_add((diffw_q24 - (pred_q24 >> 1)).abs());
            pred_q24 = diffw_q24;

            let diff_q15_m = (in_q15[m] as i32) - ((cb_ptr[m] as i32) << 7);
            let diffw_q24_m = silk_smulbb(diff_q15_m, w_ptr[m] as i32);
            sum_error_q24 = sum_error_q24.wrapping_add((diffw_q24_m - (pred_q24 >> 1)).abs());
            pred_q24 = diffw_q24_m;
        }
        err_q24[i] = sum_error_q24;
    }
}

/// xorshift64 generator with a fixed seed, so the test data never changes.
struct XorShift(u64);

impl XorShift {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    /// Uniform integer in `low..=high`.
    fn range(&mut self, low: i32, high: i32) -> i32 {
        low + (self.next() % (high as i64 - low as i64 + 1) as u64) as i32
    }
}

/// Run both functions and require the same errors. Entries from `k` on must
/// stay untouched.
fn check(in_q15: &[i16], pcb_q8: &[u8], pwght_q9: &[i16], k: usize, order: usize, what: &str) {
    let mut expected = [0x5a5a_5a5ai32; 40];
    let mut got = expected;
    reference(&mut expected, in_q15, pcb_q8, pwght_q9, k, order);
    silk_nlsf_vq(&mut got, in_q15, pcb_q8, pwght_q9, k, order);
    assert_eq!(got, expected, "{what}: order {order}, {k} vectors");
}

#[test]
fn codebooks_of_the_codec_match_reference() {
    let mut rng = XorShift(0x9E37_79B9_7F4A_7C15);
    for cb in [&SILK_NLSF_CB_NB_MB, &SILK_NLSF_CB_WB] {
        let order = cb.order as usize;
        for _ in 0..2_000 {
            // Increasing values between 0 and 32767, like real NLSFs.
            let mut nlsf = [0i16; 16];
            let mut value = 0i32;
            for slot in nlsf[..order].iter_mut() {
                value += rng.range(100, 32000 / order as i32 * 2);
                *slot = value.min(32767) as i16;
            }
            for k in [cb.n_vectors as usize, 31, 17, 1, 0] {
                check(
                    &nlsf[..order],
                    cb.cb1_nlsf_q8,
                    cb.cb1_wght_q9,
                    k,
                    order,
                    "codebook",
                );
            }
        }
    }
}

#[test]
fn extreme_values_match_reference() {
    let mut rng = XorShift(0x0123_4567_89AB_CDEF);
    let extremes = [i16::MIN, -32767, -1, 0, 1, 32767];
    // The orders with loops of fixed length (10 and 16) and others.
    for order in [2usize, 4, 8, 10, 12, 14, 16] {
        for round in 0..1_500 {
            let k = rng.range(0, 36) as usize;
            let pick = |rng: &mut XorShift| {
                if round % 3 == 0 {
                    extremes[rng.range(0, 5) as usize]
                } else {
                    rng.range(-32768, 32767) as i16
                }
            };
            let in_q15: Vec<i16> = (0..order).map(|_| pick(&mut rng)).collect();
            let pwght_q9: Vec<i16> = (0..k * order).map(|_| pick(&mut rng)).collect();
            let pcb_q8: Vec<u8> = (0..k * order)
                .map(|_| match rng.range(0, 3) {
                    0 => 0,
                    1 => 255,
                    _ => rng.range(0, 255) as u8,
                })
                .collect();
            check(&in_q15, &pcb_q8, &pwght_q9, k, order, "extreme values");
        }
    }
}
