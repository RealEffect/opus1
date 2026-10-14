//! `silk_burg_modified_fix` against the fixed-point function of libopus
//! (silk/fixed/burg_modified_FIX.c).
//!
//! The libopus that the tests link is the floating-point build, which does
//! not contain this function. The reference values in
//! `perfk_silk_burg.expected` were therefore computed once by the C function
//! itself: the libopus 1.6.1 source compiled with `FIXED_POINT`, called on
//! the signals that `Generator` below produces. The generator uses integers
//! only, so the C program and this test produce the same samples; each row
//! carries a checksum of its input to make sure of that.
//!
//! The function is fixed-point: the residual energy, its Q value and every
//! coefficient must be identical.

use opus_rs::silk::lpc_analysis::silk_burg_modified_fix;

/// (checksum of the input, residual energy, its Q value, checksum of all
/// coefficients, coefficients 0, 1 and the last one).
type Expected = (u64, i32, i32, u64, [i32; 3]);

static EXPECTED: [Expected; 1000] = include!("perfk_silk_burg.expected");

/// The generator of the C program that produced `EXPECTED`.
struct Generator(u32);

impl Generator {
    fn rnd(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
        self.0 >> 8
    }
}

/// Two-pole resonators: `2 r cos(w)` and `-r^2` in Q14.
const RES: [[i64; 2]; 6] = [
    [0, 0],
    [26000, -11000],
    [18000, -12500],
    [-9000, -10000],
    [29000, -13500],
    [31000, -15800],
];

/// Limits of the prediction gain as the encoder sets them, from loose to
/// tight, and one that the first coefficient already reaches.
const MIN_INV_GAIN_Q30: [i32; 5] = [10737418, 107374, 1073742, 53687, 536870912];

struct Case {
    x: Vec<i16>,
    min_inv_gain_q30: i32,
    subfr_length: usize,
    nb_subfr: usize,
    order: usize,
}

/// The next case: noise, pulse trains, sawtooth and square waves or almost
/// silence, through two resonators, at levels from the highest the encoder
/// passes down by up to 13 bits. Frames of 2 or 4 subframes at 8, 12 and
/// 16 kHz with the prediction order of that rate.
///
/// The encoder scales the input of this function by the inverse gains, which
/// are at most 0.25, so no sample is above a quarter of full scale. The
/// samples here keep that limit. Above it the energy of a frame does not fit
/// the 32 bits libopus gives it, and the C code is no reference any more.
fn next_case(g: &mut Generator) -> Case {
    let fs_khz = [8usize, 12, 16][(g.rnd() % 3) as usize];
    let nb_subfr = if g.rnd().is_multiple_of(3) { 2 } else { 4 };
    let order = if fs_khz == 16 { 16 } else { 10 };
    let subfr_length = 5 * fs_khz + order;
    let kind = g.rnd() % 5;
    let amp_shift = g.rnd() % 14;
    let res = RES[(g.rnd() % 6) as usize];
    let res2 = RES[(g.rnd() % 6) as usize];
    let min_inv_gain_q30 = MIN_INV_GAIN_Q30[(g.rnd() % 5) as usize];
    let period = 16 + (g.rnd() % 200) as i64;
    let (mut y1, mut y2, mut z1, mut z2, mut phase) = (0i64, 0i64, 0i64, 0i64, 0i64);
    let limit = 1i64 << 28;
    let x = (0..subfr_length * nb_subfr)
        .map(|_| {
            let noise = (g.rnd() & 0xffff) as i64 - 32768;
            phase += 1;
            if phase >= period {
                phase -= period;
            }
            let e = match kind {
                0 => noise >> 2,
                1 => {
                    if phase == 0 { 24000 } else { noise >> 9 }
                }
                2 => (phase * 16000) / period - 8000,
                3 => {
                    if phase < period / 2 { 30000 } else { -30000 }
                }
                _ => noise >> 12,
            };
            let y = (e + ((res[0] * y1) >> 14) + ((res[1] * y2) >> 14)).clamp(-limit, limit);
            y2 = y1;
            y1 = y;
            let z = (y + ((res2[0] * z1) >> 14) + ((res2[1] * z2) >> 14)).clamp(-limit, limit);
            z2 = z1;
            z1 = z;
            (z >> amp_shift).clamp(-8192, 8191) as i16
        })
        .collect();
    Case {
        x,
        min_inv_gain_q30,
        subfr_length,
        nb_subfr,
        order,
    }
}

#[test]
fn burg_modified_matches_libopus_fixed_point() {
    let mut g = Generator(0x2468_ace1);
    let mut failures = Vec::new();
    let mut q_values = std::collections::BTreeSet::new();
    for (number, expected) in EXPECTED.iter().enumerate() {
        let case = next_case(&mut g);
        let checksum = case
            .x
            .iter()
            .fold(0u64, |sum, &s| sum.wrapping_mul(31).wrapping_add(s as i64 as u64))
            & 0xffff_ffff_ffff;
        assert_eq!(checksum, expected.0, "case {number}: the input differs from the C program's");

        let (mut res_nrg, mut res_nrg_q) = (0i32, 0i32);
        let mut a_q16 = [0i32; 16];
        silk_burg_modified_fix(
            &mut res_nrg,
            &mut res_nrg_q,
            &mut a_q16,
            &case.x,
            case.min_inv_gain_q30,
            case.subfr_length,
            case.nb_subfr,
            case.order,
        );
        let d = case.order;
        let a_checksum = a_q16[..d]
            .iter()
            .fold(0u64, |sum, &a| sum.wrapping_mul(1000003).wrapping_add(a as i64 as u64))
            & 0xffff_ffff_ffff;
        let got: Expected = (checksum, res_nrg, res_nrg_q, a_checksum, [a_q16[0], a_q16[1], a_q16[d - 1]]);
        if got != *expected {
            failures.push(format!(
                "case {number} (order {d}, {} subframes of {}): opus-rs {:?}, libopus {:?}",
                case.nb_subfr, case.subfr_length, got, expected
            ));
        }
        q_values.insert(expected.2);
    }
    // The cases cover every scaling of the correlations: Q values from -7
    // (loud) to 16 (quiet).
    assert_eq!(q_values.len(), 24, "{q_values:?}");
    assert!(
        failures.is_empty(),
        "{} of {} cases differ from libopus:\n{}",
        failures.len(),
        EXPECTED.len(),
        failures.iter().take(10).cloned().collect::<Vec<_>>().join("\n")
    );
}
