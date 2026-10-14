use crate::silk::define::TYPE_VOICED;

use crate::silk::lin2log::silk_lin2log;
use crate::silk::macros::*;
use crate::silk::structs::SilkEncoderStateCommon;
use crate::silk::tuning_parameters::*;

/// Port of `silk_HP_variable_cutoff` (silk/HP_variable_cutoff.c): moves the
/// smoothed cutoff of the input high-pass filter towards the low end of the
/// pitch frequency, after a voiced frame.
pub fn silk_hp_variable_cutoff(ps_enc: &mut SilkEncoderStateCommon) {
    if ps_enc.prev_signal_type == TYPE_VOICED {
        let pitch_freq_hz_q16 =
            ((ps_enc.fs_khz as i64 * 1000 * (1 << 16)) / ps_enc.prev_lag as i64) as i32;
        let pitch_freq_log_q7 = silk_lin2log(pitch_freq_hz_q16) - (16 << 7);

        let quality_q15 = ps_enc.input_quality_bands_q15[0];
        let min_cutoff_log_q7 = silk_lin2log(fix_const_q16(VARIABLE_HP_MIN_CUTOFF_HZ)) - (16 << 7);
        let pitch_freq_log_q7 = silk_smlawb(
            pitch_freq_log_q7,
            silk_smulwb((-quality_q15) << 2, quality_q15),
            pitch_freq_log_q7 - min_cutoff_log_q7,
        );

        let mut delta_freq_q7 = pitch_freq_log_q7 - silk_rshift(ps_enc.variable_hp_smth1_q15, 8);
        if delta_freq_q7 < 0 {
            delta_freq_q7 *= 3;
        }

        let max_delta = fix_const_q7(VARIABLE_HP_MAX_DELTA_FREQ);
        delta_freq_q7 = silk_limit_32(delta_freq_q7, -max_delta, max_delta);

        let smth_coef1 = fix_const_q16(VARIABLE_HP_SMTH_COEF1);
        ps_enc.variable_hp_smth1_q15 = silk_smlawb(
            ps_enc.variable_hp_smth1_q15,
            silk_smulbb(ps_enc.speech_activity_q8, delta_freq_q7),
            smth_coef1,
        );

        let min_q15 = silk_lin2log(VARIABLE_HP_MIN_CUTOFF_HZ) << 8;
        let max_q15 = silk_lin2log(VARIABLE_HP_MAX_CUTOFF_HZ) << 8;
        ps_enc.variable_hp_smth1_q15 =
            silk_limit_32(ps_enc.variable_hp_smth1_q15, min_q15, max_q15);
    }
}

/// libopus `SILK_FIX_CONST( c, 16 )` (silk/SigProc_FIX.h): `c` in Q16,
/// rounded to the nearest integer. The C macro is
/// `(opus_int32)( c * ( 1 << 16 ) + 0.5 )`.
///
/// Without the 0.5 the smoothing coefficient `VARIABLE_HP_SMTH_COEF1` (0.1)
/// came out as 6553; libopus has 6554. The cutoff then drifted away from the
/// one libopus computes, and the input filter changed its cutoff by 1 Hz in
/// other frames than in libopus.
fn fix_const_q16(x: impl Into<f64>) -> i32 {
    (x.into() * 65536.0 + 0.5) as i32
}

/// libopus `SILK_FIX_CONST( c, 7 )`, see `fix_const_q16`.
fn fix_const_q7(x: impl Into<f64>) -> i32 {
    (x.into() * 128.0 + 0.5) as i32
}
