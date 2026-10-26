//! `silk_hp_variable_cutoff` against `silk_HP_variable_cutoff` of libopus
//! (silk/HP_variable_cutoff.c).
//!
//! The function keeps a smoothed cutoff frequency for the high-pass filter
//! at the input of the encoder. The test calls the C function in the
//! fixed-point build of the official libopus 1.6.1 release that the
//! dev-dependency `libopus-fixed` compiles (tools/libopus-fixed), on the
//! same sequence of encoder states. The function is integer arithmetic, so
//! the values must be equal.
//!
//! The sequence found one constant: the smoothing coefficient
//! `SILK_FIX_CONST( VARIABLE_HP_SMTH_COEF1, 16 )` is 6554 in libopus, and
//! opus1 had 6553 (the product cut off instead of rounded). 23 of the
//! 240 calls below then gave another value than libopus, each call starting
//! from the state of libopus.

use opus1::internals::silk::define::{TYPE_UNVOICED, TYPE_VOICED};
use opus1::internals::silk::hp_variable_cutoff::silk_hp_variable_cutoff;
use opus1::internals::silk::lin2log::silk_lin2log;
use opus1::internals::silk::structs::SilkEncoderState;

/// Number of calls.
const CALLS: usize = 240;

/// Small deterministic generator.
struct Lcg(u32);

impl Lcg {
    fn rnd(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
        self.0 >> 8
    }
}

#[test]
fn the_smoothed_cutoff_follows_libopus() {
    let mut g = Lcg(0x0bad_cafe);
    let mut enc = Box::new(SilkEncoderState::default());
    let mut different = 0;
    for c in 0..CALLS {
        let s = &mut enc.s_cmn;
        let fs = [8, 12, 16][(g.rnd() % 3) as usize];
        s.fs_khz = fs;
        s.prev_signal_type = if g.rnd() % 7 == 3 { TYPE_UNVOICED } else { TYPE_VOICED };
        s.prev_lag = 2 * fs + (g.rnd() % (16 * fs as u32)) as i32;
        s.input_quality_bands_q15[0] = (g.rnd() % 32768) as i32;
        s.speech_activity_q8 = (g.rnd() % 257) as i32;
        if c % 40 == 0 {
            // A new stream: the smoother starts at a cutoff of 60 to 100 Hz.
            s.variable_hp_smth1_q15 = silk_lin2log(60 + (g.rnd() % 41) as i32) << 8;
        }
        let expected = libopus_fixed::hp_variable_cutoff(&libopus_fixed::HpCutoffInput {
            fs_khz: s.fs_khz,
            prev_signal_type: s.prev_signal_type,
            prev_lag: s.prev_lag,
            input_quality_band0_q15: s.input_quality_bands_q15[0],
            speech_activity_q8: s.speech_activity_q8,
            variable_hp_smth1_q15: s.variable_hp_smth1_q15,
        });
        silk_hp_variable_cutoff(s);
        if s.variable_hp_smth1_q15 != expected {
            different += 1;
            if different <= 3 {
                println!("call {c}: {} where libopus has {expected}", s.variable_hp_smth1_q15);
            }
            // Continue from the value of libopus, so that one difference
            // does not hide the calls after it.
            s.variable_hp_smth1_q15 = expected;
        }
    }
    assert_eq!(different, 0, "of {CALLS} calls");
}
