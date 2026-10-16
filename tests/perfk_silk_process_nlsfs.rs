//! `silk_process_nlsfs` against libopus `silk_process_NLSFs`
//! (silk/process_NLSFs.c).
//!
//! The C function takes the whole encoder state, whose layout depends on how
//! libopus was built, so it is not called through the linked library. The
//! reference values in `perfk_silk_process_nlsfs.expected` were computed
//! once by the C function itself (the libopus 1.6.1 source, compiled) on the
//! vectors that `Generator` below produces. The generator uses integers
//! only, so the C program and this test produce the same vectors; each row
//! carries a checksum of its input to make sure of that.
//!
//! The function is fixed-point: the indices, the quantized vector and both
//! sets of prediction coefficients must be identical.

// The generator repeats the index-based loops of the C program.
#![allow(clippy::needless_range_loop)]

use opus1::silk::nlsf::silk_process_nlsfs;
use opus1::silk::structs::{SilkEncoderControl, SilkEncoderState};

/// (checksum of the input, checksum of all outputs, first index, first
/// quantized value, first prediction coefficient).
type Expected = (u64, u64, [i32; 3]);

static EXPECTED: [Expected; 1200] = include!("perfk_silk_process_nlsfs.expected");

/// The generator of the C program that produced `EXPECTED`.
struct Generator(u32);

impl Generator {
    fn rnd(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
        self.0 >> 8
    }

    /// An increasing vector of `order` values in 1..=32767. Kind 0: steps
    /// of random size. Kind 1: a quarter of the steps are 1 to 8, closer
    /// than the codebooks allow, as the analysis of a steady tone gives
    /// them. Kind 2: random values, sorted.
    fn nlsf(&mut self, order: usize, kind: u32) -> [i16; 16] {
        let mut v = [0i16; 16];
        if kind == 2 {
            for i in 0..order {
                let x = (1 + self.rnd() % 32767) as i16;
                let mut j = i;
                while j > 0 && v[j - 1] > x {
                    v[j] = v[j - 1];
                    j -= 1;
                }
                v[j] = x;
            }
        } else {
            let avg = 32767 / (order as u32 + 1);
            let mut acc = 0i32;
            for i in 0..order {
                let mut step = 1 + self.rnd() % (2 * avg);
                if kind == 1 && self.rnd().is_multiple_of(4) {
                    step = 1 + self.rnd() % 8;
                }
                acc = (acc + step as i32).min(32767 - (order - i) as i32);
                v[i] = acc as i16;
            }
        }
        v
    }
}

fn checksum(sum: u64, values: impl IntoIterator<Item = i64>, factor: u64) -> u64 {
    values.into_iter().fold(sum, |s, v| s.wrapping_mul(factor).wrapping_add(v as u64))
}

#[test]
fn process_nlsfs_matches_libopus() {
    let mut g = Generator(0x0f1e_2d3c);
    let mut failures = Vec::new();
    let mut close_pairs = 0;
    for (number, expected) in EXPECTED.iter().enumerate() {
        let mut enc = Box::new(SilkEncoderState::default());
        let mut ctrl = SilkEncoderControl::default();
        let wideband = g.rnd() % 2 == 1;
        let order = if wideband { 16 } else { 10 };
        enc.s_cmn.fs_khz = if wideband { 16 } else { 8 };
        enc.s_cmn.predict_lpc_order = order as i32;
        enc.s_cmn.nb_subfr = if g.rnd().is_multiple_of(4) { 2 } else { 4 };
        enc.s_cmn.speech_activity_q8 = (g.rnd() % 257) as i32;
        enc.s_cmn.n_nlsf_survivors = [2, 3, 4, 6, 8, 16][(g.rnd() % 6) as usize];
        enc.s_cmn.indices.signal_type = (g.rnd() % 3) as i8;
        if enc.s_cmn.nb_subfr == 4 && g.rnd().is_multiple_of(2) {
            enc.s_cmn.use_interpolated_nlsfs = 1;
            enc.s_cmn.indices.nlsf_interp_coef_q2 = (g.rnd() % 5) as i8;
        } else {
            enc.s_cmn.use_interpolated_nlsfs = (g.rnd() % 2) as i32;
            enc.s_cmn.indices.nlsf_interp_coef_q2 = 4;
        }
        let kind = g.rnd() % 3;
        let mut nlsf = g.nlsf(order, kind);
        let prev = g.nlsf(order, 0);
        enc.s_cmn.prev_nlsf_q15 = prev;
        if nlsf[..order].windows(2).any(|w| w[1] - w[0] < 9) {
            close_pairs += 1;
        }

        let input = checksum(0, nlsf[..order].iter().map(|&v| v as i64), 31);
        let input = checksum(input, prev[..order].iter().map(|&v| v as i64), 31) & 0xffff_ffff_ffff;
        assert_eq!(input, expected.0, "case {number}: the input differs from the C program's");

        silk_process_nlsfs(&mut enc, &mut ctrl, &mut nlsf);

        let indices = enc.s_cmn.indices.nlsf_indices;
        let mut out = checksum(0, ctrl.pred_coef_q12[0][..order].iter().map(|&v| v as i64), 1000003);
        out = checksum(out, ctrl.pred_coef_q12[1][..order].iter().map(|&v| v as i64), 1000003);
        out = checksum(out, nlsf[..order].iter().map(|&v| v as i64), 1000003);
        out = checksum(out, indices[..=order].iter().map(|&v| v as i64), 1000003) & 0xffff_ffff_ffff;
        let got: Expected = (
            input,
            out,
            [indices[0] as i32, nlsf[0] as i32, ctrl.pred_coef_q12[1][0] as i32],
        );
        if got != *expected {
            failures.push(format!(
                "case {number} (order {order}, kind {kind}, {} subframes, interpolation {}): opus1 {:?}, libopus {:?}",
                enc.s_cmn.nb_subfr, enc.s_cmn.indices.nlsf_interp_coef_q2, got, expected
            ));
        }
    }
    // Many vectors have values closer together than the codebooks allow.
    assert!(close_pairs > 300, "{close_pairs}");
    assert!(
        failures.is_empty(),
        "{} of {} cases differ from libopus:\n{}",
        failures.len(),
        EXPECTED.len(),
        failures.iter().take(10).cloned().collect::<Vec<_>>().join("\n")
    );
}
