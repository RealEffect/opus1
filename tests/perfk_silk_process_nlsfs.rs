//! `silk_process_nlsfs` against libopus `silk_process_NLSFs`
//! (silk/process_NLSFs.c).
//!
//! The C function takes the whole encoder state, whose layout depends on how
//! libopus was built, so it is not called in the float build the tests
//! link. The test calls it in the fixed-point build of the official libopus
//! 1.6.1 release that the dev-dependency `libopus-fixed` compiles
//! (tools/libopus-fixed), on the vectors that `Generator` below produces.
//!
//! The function is fixed-point: the indices, the quantized vector and both
//! sets of prediction coefficients must be identical.

// The generator uses index-based loops.
#![allow(clippy::needless_range_loop)]

use opus1::internals::silk::nlsf::silk_process_nlsfs;
use opus1::internals::silk::structs::{SilkEncoderControl, SilkEncoderState};

/// Number of cases.
const CASES: usize = 1200;

/// Small deterministic generator: integers only.
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

#[test]
fn process_nlsfs_matches_libopus() {
    let mut g = Generator(0x0f1e_2d3c);
    let mut failures = Vec::new();
    let mut close_pairs = 0;
    for number in 0..CASES {
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
        let s = &enc.s_cmn;
        let input = libopus_fixed::NlsfInput {
            fs_khz: s.fs_khz,
            predict_lpc_order: s.predict_lpc_order,
            nb_subfr: s.nb_subfr,
            speech_activity_q8: s.speech_activity_q8,
            n_survivors: s.n_nlsf_survivors,
            signal_type: s.indices.signal_type as i32,
            use_interpolated_nlsfs: s.use_interpolated_nlsfs,
            nlsf_interp_coef_q2: s.indices.nlsf_interp_coef_q2 as i32,
        };
        let c = libopus_fixed::process_nlsfs(&input, &nlsf, &prev);

        silk_process_nlsfs(&mut enc, &mut ctrl, &mut nlsf);

        let s = &enc.s_cmn;
        let got = (
            &ctrl.pred_coef_q12[0][..order],
            &ctrl.pred_coef_q12[1][..order],
            &nlsf[..order],
            &s.indices.nlsf_indices[..=order],
            s.indices.nlsf_interp_coef_q2 as i32,
        );
        let want = (
            &c.pred_coef_q12[0][..order],
            &c.pred_coef_q12[1][..order],
            &c.nlsf_q15[..order],
            &c.nlsf_indices[..=order],
            c.nlsf_interp_coef_q2,
        );
        if got != want {
            failures.push(format!(
                "case {number} (order {order}, kind {kind}, {} subframes, interpolation {}): opus1 {:?}, libopus {:?}",
                s.nb_subfr, input.nlsf_interp_coef_q2, got, want
            ));
        }
    }
    // Many vectors have values closer together than the codebooks allow.
    assert!(close_pairs > 300, "{close_pairs}");
    assert!(
        failures.is_empty(),
        "{} of {CASES} cases differ from libopus:\n{}",
        failures.len(),
        failures.iter().take(10).cloned().collect::<Vec<_>>().join("\n")
    );
}
