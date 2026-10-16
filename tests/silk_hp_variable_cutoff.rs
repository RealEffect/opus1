//! `silk_hp_variable_cutoff` against `silk_HP_variable_cutoff` of libopus
//! (silk/HP_variable_cutoff.c).
//!
//! The function keeps a smoothed cutoff frequency for the high-pass filter
//! at the input of the encoder. The expected values below were printed by
//! the C function itself, compiled from the libopus 1.6 source with
//! FIXED_POINT and run on the same sequence of encoder states (the generator
//! is repeated here; the C program is hp_gen.c in
//! target/perf/silk-enc-fixed-ref/). The function is integer arithmetic, so
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

/// The generator of hp_gen.c.
struct Lcg(u32);

impl Lcg {
    fn rnd(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
        self.0 >> 8
    }
}

/// `variable_HP_smth1_Q15` after each of the 240 calls of the C function.
#[rustfmt::skip]
const EXPECTED: [i32; 240] = [
    205098, 205098, 205179, 205979, 205979, 205147, 205400, 204359, 204359, 203224,
    203667, 202677, 203702, 203702, 203217, 202344, 203017, 203017, 203017, 203813,
    203873, 204122, 202816, 202596, 202208, 201175, 202424, 203395, 202941, 203935,
    203949, 204116, 204269, 203794, 204064, 204386, 205548, 206358, 206358, 207627,
    199784, 201053, 201069, 201519, 202219, 202253, 202253, 202137, 202825, 203722,
    203722, 203841, 203871, 203141, 204380, 205201, 206368, 207014, 206565, 206565,
    206565, 207401, 207401, 207501, 207705, 207805, 207815, 207146, 205896, 205896,
    206247, 206449, 206892, 206892, 207356, 207356, 207620, 207181, 207956, 207027,
    211456, 211456, 211420, 211154, 210317, 210289, 210289, 209722, 210397, 209427,
    210052, 209990, 208837, 208439, 208305, 209340, 209666, 209875, 209875, 210339,
    211598, 212076, 212681, 212293, 211777, 211777, 211496, 210277, 210113, 210113,
    211199, 209903, 209545, 210743, 210318, 210501, 210501, 209664, 208470, 207261,
    203074, 203525, 204258, 204258, 204186, 204228, 204199, 204338, 203569, 202742,
    202537, 203745, 203745, 203546, 203018, 203553, 203686, 203686, 203541, 203806,
    203005, 204111, 204289, 204477, 204788, 203936, 204359, 205562, 205898, 206396,
    205646, 206788, 207966, 207190, 206710, 206710, 206710, 205730, 206785, 207733,
    203867, 203214, 202097, 202000, 200990, 200990, 202117, 202372, 203386, 204176,
    204284, 203916, 203500, 203460, 204480, 204608, 205801, 205581, 205617, 206203,
    206141, 205360, 205196, 205114, 204155, 202849, 202849, 203345, 203459, 203459,
    202892, 203376, 202213, 201167, 201926, 201926, 201926, 201979, 202818, 202818,
    198048, 198098, 197906, 198704, 198959, 199611, 200478, 199738, 200758, 201798,
    202251, 202251, 201756, 201602, 202081, 202081, 202081, 201596, 201531, 201959,
    201586, 201586, 201356, 201496, 201103, 202337, 201566, 200696, 200696, 200777,
    201262, 201501, 201939, 201607, 201668, 201738, 201309, 201462, 201498, 202074,
];

#[test]
fn the_smoothed_cutoff_follows_libopus() {
    let mut g = Lcg(0x0bad_cafe);
    let mut enc = Box::new(SilkEncoderState::default());
    let mut different = 0;
    for (c, &expected) in EXPECTED.iter().enumerate() {
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
    assert_eq!(different, 0, "of {} calls", EXPECTED.len());
}
