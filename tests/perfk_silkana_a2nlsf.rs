//! `silk_a2nlsf` computes the values of its two polynomials at all points of
//! the cosine table first, and the values of the three halving steps of a
//! root at once. These tests compare it with the loop as libopus writes it,
//! which evaluates one point after the other.

use opus1::internals::silk::macros::{silk_add_rshift, silk_smlaww};
use opus1::internals::silk::nlsf::{silk_a2nlsf, silk_bwexpander_32, silk_nlsf2a};
use opus1::internals::silk::tables::{LSF_COS_TAB_SZ_FIX, SILK_LSF_COS_TAB_FIX_Q12};

fn trans_poly(p: &mut [i32], dd: usize) {
    for k in 2..=dd {
        for n in (k + 1..=dd).rev() {
            p[n - 2] -= p[n];
        }
        p[k - 2] -= p[k] << 1;
    }
}

fn eval_poly(p: &[i32], x: i32, dd: usize) -> i32 {
    let mut y32 = p[dd];
    let x_q16 = x << 4;
    for n in (0..dd).rev() {
        y32 = silk_smlaww(p[n], y32, x_q16);
    }
    y32
}

fn init(a_q16: &[i32], p: &mut [i32], q: &mut [i32], dd: usize) {
    p[dd] = 1 << 16;
    q[dd] = 1 << 16;
    for k in 0..dd {
        p[k] = -a_q16[dd - k - 1] - a_q16[dd + k];
        q[k] = -a_q16[dd - k - 1] + a_q16[dd + k];
    }
    for k in (1..=dd).rev() {
        p[k - 1] -= p[k];
        q[k - 1] += q[k];
    }
    trans_poly(p, dd);
    trans_poly(q, dd);
}

/// libopus `silk_A2NLSF`, line by line. Returns how often the bandwidth was
/// widened because not all roots were found.
fn reference(nlsf: &mut [i16], a_q16: &mut [i32], d: usize) -> i32 {
    const BIN_DIV_STEPS: i32 = 3;
    const MAX_ITERATIONS: i32 = 16;

    let mut p = [0i32; 9];
    let mut q = [0i32; 9];
    let dd = d >> 1;

    init(a_q16, &mut p, &mut q, dd);

    let mut xlo = SILK_LSF_COS_TAB_FIX_Q12[0] as i32;
    let mut ylo = eval_poly(&p, xlo, dd);

    let mut root_ix: usize;
    let mut use_q: bool;
    if ylo < 0 {
        nlsf[0] = 0;
        use_q = true;
        ylo = eval_poly(&q, xlo, dd);
        root_ix = 1;
    } else {
        use_q = false;
        root_ix = 0;
    }

    let mut k = 1;
    let mut i = 0;
    let mut thr = 0;

    loop {
        let poly = if use_q { &q } else { &p };
        let xhi = SILK_LSF_COS_TAB_FIX_Q12[k] as i32;
        let yhi = eval_poly(poly, xhi, dd);

        if (ylo <= 0 && yhi >= thr) || (ylo >= 0 && yhi <= -thr) {
            thr = if yhi == 0 { 1 } else { 0 };

            let mut ffrac = -256;
            let mut xlo_div = xlo;
            let mut xhi_div = xhi;
            let mut ylo_div = ylo;
            let mut yhi_div = yhi;
            for m in 0..BIN_DIV_STEPS {
                let xmid = (xlo_div + xhi_div + 1) >> 1;
                let ymid = eval_poly(poly, xmid, dd);
                if (ylo_div <= 0 && ymid >= 0) || (ylo_div >= 0 && ymid <= 0) {
                    xhi_div = xmid;
                    yhi_div = ymid;
                } else {
                    xlo_div = xmid;
                    ylo_div = ymid;
                    ffrac = silk_add_rshift(ffrac, 128, m);
                }
            }

            if ylo_div.abs() < 65536 {
                let den = ylo_div - yhi_div;
                let nom = (ylo_div << (8 - BIN_DIV_STEPS)) + (den >> 1);
                if den != 0 {
                    ffrac += nom / den;
                }
            } else {
                ffrac += ylo_div / ((ylo_div - yhi_div) >> (8 - BIN_DIV_STEPS));
            }
            nlsf[root_ix] = ((k as i32) << 8).wrapping_add(ffrac).min(i16::MAX as i32) as i16;

            root_ix += 1;
            if root_ix >= d {
                return i;
            }
            use_q = root_ix & 1 == 1;
            xlo = SILK_LSF_COS_TAB_FIX_Q12[k - 1] as i32;
            ylo = (1 - (root_ix as i32 & 2)) << 12;
        } else {
            k += 1;
            xlo = xhi;
            ylo = yhi;
            thr = 0;

            if k > LSF_COS_TAB_SZ_FIX {
                i += 1;
                if i > MAX_ITERATIONS {
                    let val = (1i32 << 15) / (d as i32 + 1);
                    nlsf[0] = val as i16;
                    for j in 1..d {
                        nlsf[j] = nlsf[j - 1].wrapping_add(val as i16);
                    }
                    return i;
                }
                silk_bwexpander_32(a_q16, d, 65536 - (1 << i));
                init(a_q16, &mut p, &mut q, dd);
                xlo = SILK_LSF_COS_TAB_FIX_Q12[0] as i32;
                ylo = eval_poly(&p, xlo, dd);
                if ylo < 0 {
                    nlsf[0] = 0;
                    use_q = true;
                    ylo = eval_poly(&q, xlo, dd);
                    root_ix = 1;
                } else {
                    use_q = false;
                    root_ix = 0;
                }
                k = 1;
            }
        }
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

/// Run both functions on the same coefficients. The frequencies and the
/// coefficients (which the function widens in place when it does not find
/// all roots) must be the same. Returns how often the bandwidth was widened.
fn check(a_q16: &[i32], d: usize, what: &str) -> i32 {
    let mut expected_a = a_q16.to_vec();
    let mut got_a = a_q16.to_vec();
    let mut expected = [0x5a5ai16; 16];
    let mut got = expected;
    let widened = reference(&mut expected, &mut expected_a, d);
    silk_a2nlsf(&mut got, &mut got_a, d);
    assert_eq!(got, expected, "{what}: order {d}, {a_q16:?}");
    assert_eq!(got_a, expected_a, "{what}: order {d}, {a_q16:?}");
    widened
}

/// Coefficients of a stable filter: random increasing frequencies turned
/// into LPC coefficients (Q12) and widened to Q16. `silk_nlsf2a` supports the
/// orders 10 and 16.
fn filter_from_random_nlsf(rng: &mut XorShift, d: usize, min_gap: i32) -> Vec<i32> {
    let mut nlsf = [0i16; 16];
    let mut value = 0i32;
    for slot in nlsf[..d].iter_mut() {
        value += rng.range(min_gap, 2 * 32000 / d as i32);
        *slot = value.min(32767) as i16;
    }
    let mut a_q12 = [0i16; 16];
    silk_nlsf2a(&mut a_q12, &nlsf, d);
    a_q12[..d].iter().map(|&a| (a as i32) << 4).collect()
}

#[test]
fn stable_filters_match_reference() {
    let mut rng = XorShift(0x9E37_79B9_7F4A_7C15);
    for d in [10usize, 16] {
        for round in 0..6_000 {
            // Frequencies far apart, and frequencies that almost touch.
            let min_gap = [400, 40, 3][round % 3];
            let a_q16 = filter_from_random_nlsf(&mut rng, d, min_gap);
            check(&a_q16, d, "stable filter");
        }
    }
}

#[test]
fn coefficients_between_q12_steps_match_reference() {
    let mut rng = XorShift(0x0123_4567_89AB_CDEF);
    for d in [10usize, 16] {
        for _ in 0..3_000 {
            let mut a_q16 = filter_from_random_nlsf(&mut rng, d, 60);
            // The encoder passes coefficients with all 16 fractional bits.
            for a in a_q16.iter_mut() {
                *a += rng.range(-8, 8);
            }
            check(&a_q16, d, "fine coefficients");
        }
    }
}

#[test]
fn every_even_order_matches_reference() {
    // Coefficients whose magnitudes add up to less than one (65536 in Q16)
    // always give a stable filter.
    let mut rng = XorShift(0x1357_9BDF_0246_8ACE);
    for d in [2usize, 4, 6, 8, 10, 12, 14, 16] {
        for _ in 0..1_500 {
            let limit = 60000 / d as i32;
            let a_q16: Vec<i32> = (0..d).map(|_| rng.range(-limit, limit)).collect();
            check(&a_q16, d, "even order");
        }
    }
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "libopus relies on 32-bit wrap-around for such coefficients; a debug build stops at the first overflow"
)]
fn filters_without_all_roots_match_reference() {
    // Random coefficients seldom have all roots on the cosine table. The
    // function then widens the bandwidth and searches again, up to 16 times.
    let mut rng = XorShift(0x0F1E_2D3C_4B5A_6978);
    let mut searched_again = 0;
    let mut most_widened = 0;
    for d in [10usize, 16] {
        for round in 0..3_000 {
            let limit = [1 << 12, 1 << 15, 1 << 17, 1 << 19][round % 4];
            let a_q16: Vec<i32> = (0..d).map(|_| rng.range(-limit, limit)).collect();
            let widened = check(&a_q16, d, "random coefficients");
            if widened > 0 {
                searched_again += 1;
            }
            most_widened = most_widened.max(widened);
        }
    }
    // The test must reach the second search, and the largest number of
    // searches: the 16th one sets all coefficients to zero, which always
    // gives all roots.
    assert!(
        searched_again > 1_000,
        "{searched_again} cases searched again"
    );
    assert_eq!(most_widened, 16);
    // All coefficients zero, and a single large one.
    check(&[0; 16], 16, "zero filter");
    check(&[0; 10], 10, "zero filter");
    let mut one = [0i32; 16];
    one[0] = 1 << 18;
    check(&one, 16, "one coefficient");
}
