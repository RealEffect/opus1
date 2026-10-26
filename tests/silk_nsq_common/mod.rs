//! The SILK quantizers of opus1 against those of the fixed-point libopus
//! 1.6.1 (`silk_NSQ_c` and `silk_NSQ_del_dec_c`), which the dev-dependency
//! `libopus-fixed` builds: the conversions between the two sets of types.

#![allow(dead_code)]

use libopus_fixed::{NsqFrame, NsqState};
use opus1::internals::silk::structs::{SideInfoIndices, SilkEncoderStateCommon, SilkNSQState};

/// The fields of `enc` and `ind` that the quantizers read.
pub fn frame_of(enc: &SilkEncoderStateCommon, ind: &SideInfoIndices) -> NsqFrame {
    NsqFrame {
        nb_subfr: enc.nb_subfr,
        subfr_length: enc.subfr_length,
        frame_length: enc.frame_length,
        ltp_mem_length: enc.ltp_mem_length,
        predict_lpc_order: enc.predict_lpc_order,
        shaping_lpc_order: enc.shaping_lpc_order,
        n_states_delayed_decision: enc.n_states_delayed_decision,
        warping_q16: enc.warping_q16,
        signal_type: ind.signal_type as i32,
        quant_offset_type: ind.quant_offset_type as i32,
        nlsf_interp_coef_q2: ind.nlsf_interp_coef_q2 as i32,
        seed: ind.seed as i32,
    }
}

/// libopus's quantizer state with the values of `s`.
pub fn state_of(s: &SilkNSQState) -> Box<NsqState> {
    let mut c = Box::<NsqState>::default();
    c.xq = s.xq;
    c.s_ltp_shp_q14 = s.s_ltp_shp_q14;
    c.s_lpc_q14 = s.s_lpc_q14;
    c.s_ar2_q14 = s.s_ar2_q14;
    c.s_lf_ar_shp_q14 = s.s_lf_ar_q14;
    c.s_diff_shp_q14 = s.s_diff_shp_q14;
    c.lag_prev = s.lag_prev;
    c.s_ltp_buf_idx = s.s_ltp_buf_idx;
    c.s_ltp_shp_buf_idx = s.s_ltp_shp_buf_idx;
    c.rand_seed = s.rand_seed;
    c.prev_gain_q16 = s.prev_gain_q16;
    c.rewhite_flag = s.rewhite_flag;
    c
}

/// The names of the fields in which the two states differ (opus1's
/// `prev_sig_type` has no counterpart in libopus).
pub fn differences(s: &SilkNSQState, c: &NsqState) -> Vec<&'static str> {
    let mut names = Vec::new();
    let mut check = |name, equal: bool| {
        if !equal {
            names.push(name);
        }
    };
    check("xq", s.xq == c.xq);
    check("sLTP_shp_Q14", s.s_ltp_shp_q14 == c.s_ltp_shp_q14);
    check("sLPC_Q14", s.s_lpc_q14 == c.s_lpc_q14);
    check("sAR2_Q14", s.s_ar2_q14 == c.s_ar2_q14);
    check("sLF_AR_shp_Q14", s.s_lf_ar_q14 == c.s_lf_ar_shp_q14);
    check("sDiff_shp_Q14", s.s_diff_shp_q14 == c.s_diff_shp_q14);
    check("lagPrev", s.lag_prev == c.lag_prev);
    check("sLTP_buf_idx", s.s_ltp_buf_idx == c.s_ltp_buf_idx);
    check("sLTP_shp_buf_idx", s.s_ltp_shp_buf_idx == c.s_ltp_shp_buf_idx);
    check("rand_seed", s.rand_seed == c.rand_seed);
    check("prev_gain_Q16", s.prev_gain_q16 == c.prev_gain_q16);
    check("rewhite_flag", s.rewhite_flag == c.rewhite_flag);
    names
}
