//! Replay tests for the SILK noise-shaping quantizers (`silk_nsq_del_dec`
//! and `silk_nsq`).
//!
//! The quantizers were rewritten for speed. Their decisions must not change,
//! not even where two choices cost the same. This file keeps the
//! implementations from before the rewrite as reference code (the modules
//! `reference_del_dec` and `reference_nsq`, copied unchanged except for the
//! `use` paths). The tests run generated frames through the current and the
//! previous implementation and require equal pulses, an equal returned seed
//! and an equal quantizer state after every frame.
//!
//! The frames are generated, not recorded: the generator covers every state
//! count, filter order, sampling rate, frame length and signal type, and
//! values far outside what the encoder produces.

// The reference modules are kept as they were written.
#![allow(clippy::all)]
#![allow(dead_code)]

use opus_rs::silk::define::*;
use opus_rs::silk::nsq::silk_nsq;
use opus_rs::silk::nsq_del_dec::silk_nsq_del_dec;
use opus_rs::silk::structs::*;

/// `src/silk/nsq_del_dec.rs` before the rewrite.
mod reference_del_dec {
    use opus_rs::silk::define::*;
    use opus_rs::silk::macros::*;
    use opus_rs::silk::sigproc_fix::*;
    use opus_rs::silk::structs::*;
    use opus_rs::silk::tables::*;

    #[derive(Copy, Clone)]
    pub struct NSQDelDecStruct {
        pub s_lpc_q14: [i32; MAX_SUB_FRAME_LENGTH + NSQ_LPC_BUF_LENGTH],
        pub rand_state: [i32; DECISION_DELAY],
        pub q_q10: [i32; DECISION_DELAY],
        pub xq_q14: [i32; DECISION_DELAY],
        pub pred_q15: [i32; DECISION_DELAY],
        pub shape_q14: [i32; DECISION_DELAY],
        pub s_ar2_q14: [i32; MAX_SHAPE_LPC_ORDER],
        pub lf_ar_q14: i32,
        pub diff_q14: i32,
        pub seed: i32,
        pub seed_init: i32,
        pub rd_q10: i32,
    }

    impl Default for NSQDelDecStruct {
        fn default() -> Self {
            Self {
                s_lpc_q14: [0; MAX_SUB_FRAME_LENGTH + NSQ_LPC_BUF_LENGTH],
                rand_state: [0; DECISION_DELAY],
                q_q10: [0; DECISION_DELAY],
                xq_q14: [0; DECISION_DELAY],
                pred_q15: [0; DECISION_DELAY],
                shape_q14: [0; DECISION_DELAY],
                s_ar2_q14: [0; MAX_SHAPE_LPC_ORDER],
                lf_ar_q14: 0,
                diff_q14: 0,
                seed: 0,
                seed_init: 0,
                rd_q10: 0,
            }
        }
    }

    #[derive(Copy, Clone, Default)]
    pub struct NSQSampleStruct {
        pub q_q10: i32,
        pub rd_q10: i32,
        pub xq_q14: i32,
        pub lf_ar_q14: i32,
        pub diff_q14: i32,
        pub s_ltp_shp_q14: i32,
        pub lpc_exc_q14: i32,
    }

    pub type NSQSamplePair = [NSQSampleStruct; 2];

    #[inline]
    fn silk_nsq_del_dec_scale_states(
        ps_enc_c: &SilkEncoderStateCommon,
        nsq: &mut SilkNSQState,
        ps_del_dec: &mut [NSQDelDecStruct],
        x16: &[i16],
        x_sc_q10: &mut [i32],
        s_ltp: &[i16],
        s_ltp_q15: &mut [i32],
        subfr: usize,
        n_states_delayed_decision: i32,
        ltp_scale_q14: i32,
        gains_q16: &[i32],
        pitch_l: &[i32],
        signal_type: i32,
        decision_delay: i32,
    ) {
        let lag = pitch_l[subfr] as usize;
        let inv_gain_q31 = silk_inverse32_varq(gains_q16[subfr].max(1), 47);

        let inv_gain_q26 = silk_rshift_round(inv_gain_q31, 5);
        let n = ps_enc_c.subfr_length as usize;
        for (x_out, &x_in) in x_sc_q10[..n].iter_mut().zip(x16[..n].iter()) {
            *x_out = silk_smulww(x_in as i32, inv_gain_q26);
        }

        if nsq.rewhite_flag != 0 {
            let mut inv_gain_q31_scaled = inv_gain_q31;
            if subfr == 0 {
                inv_gain_q31_scaled = silk_lshift(silk_smulwb(inv_gain_q31, ltp_scale_q14), 2);
            }
            for i in
                (nsq.s_ltp_buf_idx as usize - lag - LTP_ORDER / 2)..(nsq.s_ltp_buf_idx as usize)
            {
                s_ltp_q15[i] = silk_smulwb(inv_gain_q31_scaled, s_ltp[i] as i32);
            }
        }

        if gains_q16[subfr] != nsq.prev_gain_q16 {
            let gain_adj_q16 = silk_div32_varq(nsq.prev_gain_q16, gains_q16[subfr], 16);

            for i in (nsq.s_ltp_shp_buf_idx as usize - ps_enc_c.ltp_mem_length as usize)
                ..(nsq.s_ltp_shp_buf_idx as usize)
            {
                nsq.s_ltp_shp_q14[i] = silk_smulww(gain_adj_q16, nsq.s_ltp_shp_q14[i]);
            }

            if signal_type == TYPE_VOICED && nsq.rewhite_flag == 0 {
                let ltp_start = nsq.s_ltp_buf_idx as usize - lag - LTP_ORDER / 2;
                let ltp_end = nsq.s_ltp_buf_idx as usize - decision_delay as usize;
                for v in s_ltp_q15[ltp_start..ltp_end].iter_mut() {
                    *v = silk_smulww(gain_adj_q16, *v);
                }
            }

            let n_states = (n_states_delayed_decision as usize).min(NSQ_MAX_STATES_OPERATING);
            for k in 0..n_states {
                let ps_dd = &mut ps_del_dec[k];
                ps_dd.lf_ar_q14 = silk_smulww(gain_adj_q16, ps_dd.lf_ar_q14);
                ps_dd.diff_q14 = silk_smulww(gain_adj_q16, ps_dd.diff_q14);
                for i in 0..NSQ_LPC_BUF_LENGTH {
                    ps_dd.s_lpc_q14[i] = silk_smulww(gain_adj_q16, ps_dd.s_lpc_q14[i]);
                }
                for i in 0..MAX_SHAPE_LPC_ORDER {
                    ps_dd.s_ar2_q14[i] = silk_smulww(gain_adj_q16, ps_dd.s_ar2_q14[i]);
                }
                for i in 0..DECISION_DELAY {
                    ps_dd.pred_q15[i] = silk_smulww(gain_adj_q16, ps_dd.pred_q15[i]);
                    ps_dd.shape_q14[i] = silk_smulww(gain_adj_q16, ps_dd.shape_q14[i]);
                }
            }
            nsq.prev_gain_q16 = gains_q16[subfr];
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    #[allow(unsafe_op_in_unsafe_fn)]
    unsafe fn silk_lpc_prediction_neon(
        ps_lpc_q14: &[i32],
        idx: usize,
        a_q12: &[i16],
        predict_lpc_order: i32,
    ) -> i32 {
        use core::arch::aarch64::*;

        let order = predict_lpc_order as usize;
        let lpc_base = ps_lpc_q14.as_ptr().add(idx);
        let a_ptr = a_q12.as_ptr();

        let mut acc = vdupq_n_s32(0i32);
        let neon_taps = order & !3;

        let mut j = 0usize;
        while j < neon_taps {
            let lpc_asc = vld1q_s32(lpc_base.sub(j + 3));

            let coef_narrow = vld1_s16(a_ptr.add(j));
            let coef_wide = vmovl_s16(coef_narrow);

            let coef_rev = vrev64q_s32(vcombine_s32(
                vget_high_s32(coef_wide),
                vget_low_s32(coef_wide),
            ));

            let prod_lo = vmull_s32(vget_low_s32(lpc_asc), vget_low_s32(coef_rev));
            let prod_hi = vmull_s32(vget_high_s32(lpc_asc), vget_high_s32(coef_rev));

            let shr_lo = vshrn_n_s64::<16>(prod_lo); // int32x2
            let shr_hi = vshrn_n_s64::<16>(prod_hi); // int32x2

            acc = vaddq_s32(acc, vcombine_s32(shr_lo, shr_hi));
            j += 4;
        }

        let mut out = silk_rshift(predict_lpc_order, 1) + vaddvq_s32(acc);

        while j < order {
            out = silk_smlawb(out, ps_lpc_q14[idx - j], a_q12[j] as i32);
            j += 1;
        }

        out
    }

    #[inline]
    fn silk_noise_shape_quantizer_short_prediction(
        ps_lpc_q14: &[i32],
        idx: usize,
        a_q12: &[i16],
        predict_lpc_order: i32,
    ) -> i32 {
        #[cfg(target_arch = "aarch64")]
        // SAFETY: aarch64 always has NEON; bounds are guaranteed by SILK frame sizing.
        unsafe {
            return silk_lpc_prediction_neon(ps_lpc_q14, idx, a_q12, predict_lpc_order);
        }
        #[allow(unreachable_code)]
        {
            let mut out = silk_rshift(predict_lpc_order, 1);
            for j in 0..predict_lpc_order as usize {
                out = silk_smlawb(out, ps_lpc_q14[idx - j], a_q12[j] as i32);
            }
            out
        }
    }

    pub fn silk_noise_shape_quantizer_del_dec(
        nsq: &mut SilkNSQState,
        ps_del_dec: &mut [NSQDelDecStruct],
        signal_type: i32,
        x_q10: &[i32],
        pulses: &mut [i8],
        pulses_offset: i32,
        xq_ptr: i32,
        s_ltp_q15: &mut [i32],
        delayed_ga_q10: &mut [i32],
        a_q12: &[i16],
        b_q14: &[i16],
        ar_shp_q13: &[i16],
        lag: i32,
        harm_shape_fir_packed_q14: i32,
        tilt_q14: i32,
        lf_shp_q14: i32,
        gain_q16: i32,
        lambda_q10: i32,
        offset_q10: i32,
        length: i32,
        subfr: i32,
        shaping_lpc_order: i32,
        predict_lpc_order: i32,
        warping_q16: i32,
        n_states_delayed_decision: i32,
        smpl_buf_idx: &mut i32,
        decision_delay: i32,
        _frame_counter: i32,
    ) {
        let mut ps_sample_state = [[NSQSampleStruct::default(); 2]; NSQ_MAX_STATES_OPERATING];
        let gain_q10 = silk_rshift(gain_q16, 6);

        let n_states = (n_states_delayed_decision as usize).min(NSQ_MAX_STATES_OPERATING);

        let pred_lag_ptr_base = nsq.s_ltp_buf_idx - lag + LTP_ORDER as i32 / 2;
        let shp_lag_ptr_base = nsq.s_ltp_shp_buf_idx - lag + HARM_SHAPE_FIR_TAPS as i32 / 2;

        for i in 0..length {
            let idx = i as usize;
            let mut ltp_pred_q14 = 0;
            if signal_type == TYPE_VOICED {
                let pred_lag_idx_calc = pred_lag_ptr_base + i;
                if pred_lag_idx_calc >= LTP_ORDER as i32
                    && pred_lag_idx_calc < s_ltp_q15.len() as i32
                {
                    let pred_lag_idx = pred_lag_idx_calc as usize;
                    ltp_pred_q14 = 2;
                    ltp_pred_q14 =
                        silk_smlawb(ltp_pred_q14, s_ltp_q15[pred_lag_idx], b_q14[0] as i32);
                    ltp_pred_q14 =
                        silk_smlawb(ltp_pred_q14, s_ltp_q15[pred_lag_idx - 1], b_q14[1] as i32);
                    ltp_pred_q14 =
                        silk_smlawb(ltp_pred_q14, s_ltp_q15[pred_lag_idx - 2], b_q14[2] as i32);
                    ltp_pred_q14 =
                        silk_smlawb(ltp_pred_q14, s_ltp_q15[pred_lag_idx - 3], b_q14[3] as i32);
                    ltp_pred_q14 =
                        silk_smlawb(ltp_pred_q14, s_ltp_q15[pred_lag_idx - 4], b_q14[4] as i32);
                    ltp_pred_q14 = silk_lshift(ltp_pred_q14, 1);
                }
            }

            let mut n_ltp_q14 = 0;
            if lag > 0 {
                let shp_lag_idx_calc = shp_lag_ptr_base + i;
                if shp_lag_idx_calc >= HARM_SHAPE_FIR_TAPS as i32
                    && shp_lag_idx_calc < nsq.s_ltp_shp_q14.len() as i32
                {
                    let shp_lag_idx = shp_lag_idx_calc as usize;
                    n_ltp_q14 = silk_smulwb(
                        silk_add_sat32(
                            nsq.s_ltp_shp_q14[shp_lag_idx],
                            nsq.s_ltp_shp_q14[shp_lag_idx - 2],
                        ),
                        harm_shape_fir_packed_q14,
                    );
                    n_ltp_q14 = silk_smlawt(
                        n_ltp_q14,
                        nsq.s_ltp_shp_q14[shp_lag_idx - 1],
                        harm_shape_fir_packed_q14,
                    );
                    n_ltp_q14 = silk_sub_lshift32(ltp_pred_q14, n_ltp_q14, 2);
                }
            }

            for k in 0..n_states {
                let ps_dd = &mut ps_del_dec[k];
                let ps_ss = &mut ps_sample_state[k];

                ps_dd.seed = silk_rand(ps_dd.seed);
                let ps_lpc_q14_idx = NSQ_LPC_BUF_LENGTH - 1 + idx;
                let lpc_pred_q14 = silk_lshift(
                    silk_noise_shape_quantizer_short_prediction(
                        &ps_dd.s_lpc_q14,
                        ps_lpc_q14_idx,
                        a_q12,
                        predict_lpc_order,
                    ),
                    4,
                );

                let mut tmp2 = silk_smlawb(ps_dd.diff_q14, ps_dd.s_ar2_q14[0], warping_q16);
                let mut tmp1 = silk_smlawb(
                    ps_dd.s_ar2_q14[0],
                    silk_sub32_ovflw(ps_dd.s_ar2_q14[1], tmp2),
                    warping_q16,
                );
                ps_dd.s_ar2_q14[0] = tmp2;
                let mut n_ar_q14 = silk_rshift(shaping_lpc_order, 1);
                n_ar_q14 = silk_smlawb(n_ar_q14, tmp2, ar_shp_q13[0] as i32);
                for j in (2..shaping_lpc_order as usize).step_by(2) {
                    tmp2 = silk_smlawb(
                        ps_dd.s_ar2_q14[j - 1],
                        silk_sub32_ovflw(ps_dd.s_ar2_q14[j], tmp1),
                        warping_q16,
                    );
                    ps_dd.s_ar2_q14[j - 1] = tmp1;
                    n_ar_q14 = silk_smlawb(n_ar_q14, tmp1, ar_shp_q13[j - 1] as i32);
                    tmp1 = silk_smlawb(
                        ps_dd.s_ar2_q14[j],
                        silk_sub32_ovflw(ps_dd.s_ar2_q14[j + 1], tmp2),
                        warping_q16,
                    );
                    ps_dd.s_ar2_q14[j] = tmp2;
                    n_ar_q14 = silk_smlawb(n_ar_q14, tmp2, ar_shp_q13[j] as i32);
                }
                ps_dd.s_ar2_q14[shaping_lpc_order as usize - 1] = tmp1;
                n_ar_q14 = silk_smlawb(
                    n_ar_q14,
                    tmp1,
                    ar_shp_q13[shaping_lpc_order as usize - 1] as i32,
                );

                n_ar_q14 = silk_lshift(n_ar_q14, 1);
                n_ar_q14 = silk_smlawb(n_ar_q14, ps_dd.lf_ar_q14, tilt_q14);
                n_ar_q14 = silk_lshift(n_ar_q14, 2);

                let smpl_idx = (*smpl_buf_idx as usize).min(DECISION_DELAY - 1);
                let mut n_lf_q14 = silk_smulwb(ps_dd.shape_q14[smpl_idx], lf_shp_q14);
                n_lf_q14 = silk_smlawt(n_lf_q14, ps_dd.lf_ar_q14, lf_shp_q14);
                n_lf_q14 = silk_lshift(n_lf_q14, 2);

                let tmp1_val = silk_sub_sat32(
                    silk_add32_ovflw(n_ltp_q14, lpc_pred_q14),
                    silk_add_sat32(n_ar_q14, n_lf_q14),
                );
                let r_q10 = x_q10[idx] - silk_rshift_round(tmp1_val, 4);

                let r_q10_signed = if ps_dd.seed < 0 { -r_q10 } else { r_q10 };
                let r_q10_signed = silk_limit_32(r_q10_signed, -(31 << 10), 30 << 10);

                let q1_q10_in = r_q10_signed - offset_q10;
                let mut q1_q0 = silk_rshift(q1_q10_in, 10);
                if lambda_q10 > 2048 {
                    let rdo_offset = lambda_q10 / 2 - 512;
                    if q1_q10_in > rdo_offset {
                        q1_q0 = silk_rshift(q1_q10_in - rdo_offset, 10);
                    } else if q1_q10_in < -rdo_offset {
                        q1_q0 = silk_rshift(q1_q10_in + rdo_offset, 10);
                    } else if q1_q10_in < 0 {
                        q1_q0 = -1;
                    } else {
                        q1_q0 = 0;
                    }
                }

                let (rd1_q10, rd2_q10, q1_q10_val, q2_q10_val);
                if q1_q0 > 0 {
                    q1_q10_val =
                        silk_sub32(silk_lshift(q1_q0, 10), QUANT_LEVEL_ADJUST_Q10) + offset_q10;
                    q2_q10_val = q1_q10_val + 1024;
                    rd1_q10 = silk_smulbb(q1_q10_val, lambda_q10);
                    rd2_q10 = silk_smulbb(q2_q10_val, lambda_q10);
                } else if q1_q0 == 0 {
                    q1_q10_val = offset_q10;
                    q2_q10_val = q1_q10_val + 1024 - QUANT_LEVEL_ADJUST_Q10;
                    rd1_q10 = silk_smulbb(q1_q10_val, lambda_q10);
                    rd2_q10 = silk_smulbb(q2_q10_val, lambda_q10);
                } else if q1_q0 == -1 {
                    q2_q10_val = offset_q10;
                    q1_q10_val = q2_q10_val - (1024 - QUANT_LEVEL_ADJUST_Q10);
                    rd1_q10 = silk_smulbb(-q1_q10_val, lambda_q10);
                    rd2_q10 = silk_smulbb(q2_q10_val, lambda_q10);
                } else {
                    q1_q10_val =
                        silk_add32(silk_lshift(q1_q0, 10), QUANT_LEVEL_ADJUST_Q10) + offset_q10;
                    q2_q10_val = q1_q10_val + 1024;
                    rd1_q10 = silk_smulbb(-q1_q10_val, lambda_q10);
                    rd2_q10 = silk_smulbb(-q2_q10_val, lambda_q10);
                }

                let mut rr_q10 = r_q10_signed - q1_q10_val;
                let rd1_q10_final = silk_rshift(silk_smlabb(rd1_q10, rr_q10, rr_q10), 10);
                rr_q10 = r_q10_signed - q2_q10_val;
                let rd2_q10_final = silk_rshift(silk_smlabb(rd2_q10, rr_q10, rr_q10), 10);

                if rd1_q10_final < rd2_q10_final {
                    ps_ss[0].rd_q10 = ps_dd.rd_q10 + rd1_q10_final;
                    ps_ss[1].rd_q10 = ps_dd.rd_q10 + rd2_q10_final;
                    ps_ss[0].q_q10 = q1_q10_val;
                    ps_ss[1].q_q10 = q2_q10_val;
                } else {
                    ps_ss[0].rd_q10 = ps_dd.rd_q10 + rd2_q10_final;
                    ps_ss[1].rd_q10 = ps_dd.rd_q10 + rd1_q10_final;
                    ps_ss[0].q_q10 = q2_q10_val;
                    ps_ss[1].q_q10 = q1_q10_val;
                }

                for j in 0..2 {
                    let mut exc_q14 = silk_lshift(ps_ss[j].q_q10, 4);
                    if ps_dd.seed < 0 {
                        exc_q14 = -exc_q14;
                    }
                    let lpc_exc_q14 = silk_add32(exc_q14, ltp_pred_q14);
                    let xq_q14 = silk_add32_ovflw(lpc_exc_q14, lpc_pred_q14);
                    ps_ss[j].diff_q14 = silk_sub32_ovflw(xq_q14, silk_lshift(x_q10[idx], 4));
                    let s_lf_ar_shp_q14 = silk_sub32_ovflw(ps_ss[j].diff_q14, n_ar_q14);
                    ps_ss[j].s_ltp_shp_q14 = silk_sub_sat32(s_lf_ar_shp_q14, n_lf_q14);
                    ps_ss[j].lf_ar_q14 = s_lf_ar_shp_q14;
                    ps_ss[j].lpc_exc_q14 = lpc_exc_q14;
                    ps_ss[j].xq_q14 = xq_q14;
                }
            }

            *smpl_buf_idx = (*smpl_buf_idx - 1 + DECISION_DELAY as i32) % DECISION_DELAY as i32;
            let last_smple_idx =
                ((*smpl_buf_idx + decision_delay) % DECISION_DELAY as i32) as usize;

            let mut winner_ind = 0;
            let mut rd_min_q10 = ps_sample_state[0][0].rd_q10;
            for k in 1..n_states {
                if ps_sample_state[k][0].rd_q10 < rd_min_q10 {
                    rd_min_q10 = ps_sample_state[k][0].rd_q10;
                    winner_ind = k;
                }
            }

            let winner_rand_state = ps_del_dec[winner_ind].rand_state[last_smple_idx];
            for k in 0..n_states {
                if ps_del_dec[k].rand_state[last_smple_idx] != winner_rand_state {
                    ps_sample_state[k][0].rd_q10 =
                        ps_sample_state[k][0].rd_q10.saturating_add(i32::MAX >> 4);
                    ps_sample_state[k][1].rd_q10 =
                        ps_sample_state[k][1].rd_q10.saturating_add(i32::MAX >> 4);
                }
            }

            let mut rd_max_q10 = ps_sample_state[0][0].rd_q10;
            let mut rd_max_ind = 0;
            let mut rd_min_q10_2 = ps_sample_state[0][1].rd_q10;
            let mut rd_min_ind = 0;
            for k in 1..n_states {
                if ps_sample_state[k][0].rd_q10 > rd_max_q10 {
                    rd_max_q10 = ps_sample_state[k][0].rd_q10;
                    rd_max_ind = k;
                }
                if ps_sample_state[k][1].rd_q10 < rd_min_q10_2 {
                    rd_min_q10_2 = ps_sample_state[k][1].rd_q10;
                    rd_min_ind = k;
                }
            }

            if rd_min_q10_2 < rd_max_q10 {
                if rd_min_ind != rd_max_ind {
                    let (min_state, max_state) = if rd_min_ind < rd_max_ind {
                        let (left, right) = ps_del_dec.split_at_mut(rd_max_ind);
                        (&left[rd_min_ind], &mut right[0])
                    } else {
                        let (left, right) = ps_del_dec.split_at_mut(rd_min_ind);
                        (&right[0], &mut left[rd_max_ind])
                    };
                    max_state.s_lpc_q14[idx..].copy_from_slice(&min_state.s_lpc_q14[idx..]);
                    max_state.rand_state = min_state.rand_state;
                    max_state.q_q10 = min_state.q_q10;
                    max_state.xq_q14 = min_state.xq_q14;
                    max_state.pred_q15 = min_state.pred_q15;
                    max_state.shape_q14 = min_state.shape_q14;
                    max_state.s_ar2_q14 = min_state.s_ar2_q14;
                    max_state.lf_ar_q14 = min_state.lf_ar_q14;
                    max_state.diff_q14 = min_state.diff_q14;
                    max_state.seed = min_state.seed;
                    max_state.seed_init = min_state.seed_init;
                    max_state.rd_q10 = min_state.rd_q10;
                }

                ps_sample_state[rd_max_ind][0] = ps_sample_state[rd_min_ind][1];
            }

            let ps_dd = &ps_del_dec[winner_ind];
            if subfr > 0 || i >= decision_delay {
                let pulse_idx = (pulses_offset + i - decision_delay) as isize;
                let xq_idx = (xq_ptr + i - decision_delay) as isize;
                let shp_idx = (nsq.s_ltp_shp_buf_idx - decision_delay) as isize;
                let ltp_idx = (nsq.s_ltp_buf_idx - decision_delay) as isize;

                if pulse_idx >= 0 && pulse_idx < pulses.len() as isize {
                    pulses[pulse_idx as usize] =
                        silk_rshift_round(ps_dd.q_q10[last_smple_idx], 10) as i8;
                }
                if xq_idx >= 0 && xq_idx < nsq.xq.len() as isize {
                    nsq.xq[xq_idx as usize] = silk_sat16(silk_rshift_round(
                        silk_smulww(ps_dd.xq_q14[last_smple_idx], delayed_ga_q10[last_smple_idx]),
                        8,
                    )) as i16;
                }
                if shp_idx >= 0 && shp_idx < nsq.s_ltp_shp_q14.len() as isize {
                    nsq.s_ltp_shp_q14[shp_idx as usize] = ps_dd.shape_q14[last_smple_idx];
                }
                if ltp_idx >= 0 && ltp_idx < s_ltp_q15.len() as isize {
                    s_ltp_q15[ltp_idx as usize] = ps_dd.pred_q15[last_smple_idx];
                }
            }
            nsq.s_ltp_shp_buf_idx += 1;
            nsq.s_ltp_buf_idx += 1;

            for k in 0..n_states {
                let ps_ss = &ps_sample_state[k][0];
                let ps_dd = &mut ps_del_dec[k];
                ps_dd.lf_ar_q14 = ps_ss.lf_ar_q14;
                ps_dd.diff_q14 = ps_ss.diff_q14;

                let lpc_idx = NSQ_LPC_BUF_LENGTH + idx;
                if lpc_idx < ps_dd.s_lpc_q14.len() {
                    ps_dd.s_lpc_q14[lpc_idx] = ps_ss.xq_q14;
                }
                let smpl_idx = (*smpl_buf_idx as usize).min(DECISION_DELAY - 1);
                ps_dd.xq_q14[smpl_idx] = ps_ss.xq_q14;
                ps_dd.q_q10[smpl_idx] = ps_ss.q_q10;
                ps_dd.pred_q15[smpl_idx] = silk_lshift(ps_ss.lpc_exc_q14, 1);
                ps_dd.shape_q14[smpl_idx] = ps_ss.s_ltp_shp_q14;
                ps_dd.seed = silk_add32_ovflw(ps_dd.seed, silk_rshift_round(ps_ss.q_q10, 10));
                ps_dd.rand_state[smpl_idx] = ps_dd.seed;
                ps_dd.rd_q10 = ps_ss.rd_q10;
            }
            let smpl_idx = (*smpl_buf_idx as usize).min(DECISION_DELAY - 1);
            delayed_ga_q10[smpl_idx] = gain_q10;
        }
        for k in 0..n_states {
            let ps_dd = &mut ps_del_dec[k];
            let mut tmp = [0i32; NSQ_LPC_BUF_LENGTH];
            tmp.copy_from_slice(
                &ps_dd.s_lpc_q14[length as usize..length as usize + NSQ_LPC_BUF_LENGTH],
            );
            ps_dd.s_lpc_q14[..NSQ_LPC_BUF_LENGTH].copy_from_slice(&tmp);
        }
    }

    pub fn silk_nsq_del_dec(
        ps_common: &SilkEncoderStateCommon,
        ps_nsq: &mut SilkNSQState,
        ps_indices: &SideInfoIndices,

        x16: &[i16],
        pulses: &mut [i8],
        pred_coef_q12: &[i16],
        ltp_coef_q14: &[i16],
        ar_q13: &[i16],
        harm_shape_gain_q14: &[i32],
        tilt_q14: &[i32],
        lf_shp_q14: &[i32],
        gains_q16: &[i32],
        pitch_l: &[i32],
        lambda_q10: i32,
        ltp_scale_q14: i32,
    ) -> i32 {
        let mut x_sc_q10 = [0i32; MAX_SUB_FRAME_LENGTH];
        let mut delayed_ga_q10 = [0i32; DECISION_DELAY];
        let mut s_ltp_q15 = [0i32; LTP_MEM_LENGTH_MS * MAX_FS_KHZ + MAX_FRAME_LENGTH];
        let mut s_ltp = [0i16; LTP_MEM_LENGTH_MS * MAX_FS_KHZ + MAX_FRAME_LENGTH];
        let mut ps_del_dec = [NSQDelDecStruct::default(); NSQ_MAX_STATES_OPERATING];

        let mut lag = ps_nsq.lag_prev;

        let n_states = (ps_common.n_states_delayed_decision as usize).min(NSQ_MAX_STATES_OPERATING);

        for k in 0..n_states {
            let ps_dd = &mut ps_del_dec[k];
            ps_dd.seed = (k as i32 + ps_indices.seed as i32) & 3;
            ps_dd.seed_init = ps_dd.seed;
            ps_dd.rd_q10 = 0;
            ps_dd.lf_ar_q14 = ps_nsq.s_lf_ar_q14;
            ps_dd.diff_q14 = ps_nsq.s_diff_shp_q14;

            if ps_common.ltp_mem_length > 0 {
                ps_dd.shape_q14[0] = ps_nsq.s_ltp_shp_q14[ps_common.ltp_mem_length as usize - 1];
            }
            ps_dd.s_lpc_q14[..NSQ_LPC_BUF_LENGTH]
                .copy_from_slice(&ps_nsq.s_lpc_q14[..NSQ_LPC_BUF_LENGTH]);
            ps_dd.s_ar2_q14.copy_from_slice(&ps_nsq.s_ar2_q14);
        }

        let offset_q10 = SILK_QUANT_OFFSETS_Q10[(ps_indices.signal_type >> 1) as usize]
            [ps_indices.quant_offset_type as usize] as i32;
        let mut smpl_buf_idx = 0i32;
        let mut decision_delay = (DECISION_DELAY as i32).min(ps_common.subfr_length);

        if ps_indices.signal_type as i32 == TYPE_VOICED {
            for k in 0..ps_common.nb_subfr as usize {
                let pitch_constraint = pitch_l[k] - LTP_ORDER as i32 / 2 - 1;
                if pitch_constraint > 0 {
                    decision_delay = decision_delay.min(pitch_constraint);
                }
            }
        } else if lag > 0 {
            let lag_constraint = lag - LTP_ORDER as i32 / 2 - 1;
            if lag_constraint > 0 {
                decision_delay = decision_delay.min(lag_constraint);
            }
        }

        let lsf_interpolation_flag = if ps_indices.nlsf_interp_coef_q2 == 4 {
            0
        } else {
            1
        };

        ps_nsq.s_ltp_shp_buf_idx = ps_common.ltp_mem_length;
        ps_nsq.s_ltp_buf_idx = ps_common.ltp_mem_length;

        let mut x_ptr = 0;
        let mut pulses_ptr = 0;
        let mut xq_ptr = ps_common.ltp_mem_length as usize;
        let mut subfr_nsq = 0;

        for k in 0..ps_common.nb_subfr as usize {
            let a_q12 = &pred_coef_q12
                [((k >> 1) | (1 - lsf_interpolation_flag as usize)) * MAX_LPC_ORDER..];
            let b_q14 = &ltp_coef_q14[k * LTP_ORDER..];
            let ar_shp_q13 = &ar_q13[k * MAX_SHAPE_LPC_ORDER..];

            let harm_shape_gain = harm_shape_gain_q14[k];
            let mut harm_shape_fir_packed_q14 = silk_rshift(harm_shape_gain, 2);
            harm_shape_fir_packed_q14 |= silk_lshift(silk_rshift(harm_shape_gain, 1), 16);

            ps_nsq.rewhite_flag = 0;
            if ps_indices.signal_type as i32 == TYPE_VOICED {
                lag = pitch_l[k];
                if (k & (3 - silk_lshift(lsf_interpolation_flag, 1) as usize)) == 0 {
                    if k == 2 {
                        let mut rd_min_q10 = ps_del_dec[0].rd_q10;
                        let mut winner_ind = 0;
                        for i in 1..n_states {
                            if ps_del_dec[i].rd_q10 < rd_min_q10 {
                                rd_min_q10 = ps_del_dec[i].rd_q10;
                                winner_ind = i;
                            }
                        }
                        for i in 0..n_states {
                            if i != winner_ind {
                                ps_del_dec[i].rd_q10 =
                                    ps_del_dec[i].rd_q10.saturating_add(i32::MAX >> 4);
                            }
                        }

                        let ps_dd = &ps_del_dec[winner_ind];
                        let mut last_smple_idx =
                            (smpl_buf_idx + decision_delay) % DECISION_DELAY as i32;
                        for i in 0..decision_delay {
                            last_smple_idx = (last_smple_idx - 1 + DECISION_DELAY as i32)
                                % DECISION_DELAY as i32;
                            let pulse_idx = (pulses_ptr as i32 + i - decision_delay) as isize;
                            let xq_idx = (xq_ptr as i32 + i - decision_delay) as isize;
                            let shp_idx = (ps_nsq.s_ltp_shp_buf_idx + i - decision_delay) as isize;
                            if pulse_idx >= 0 && xq_idx >= 0 && shp_idx >= 0 {
                                pulses[pulse_idx as usize] =
                                    silk_rshift_round(ps_dd.q_q10[last_smple_idx as usize], 10)
                                        as i8;
                                ps_nsq.xq[xq_idx as usize] = silk_sat16(silk_rshift_round(
                                    silk_smulww(
                                        ps_dd.xq_q14[last_smple_idx as usize],
                                        gains_q16[1],
                                    ),
                                    14,
                                ))
                                    as i16;
                                ps_nsq.s_ltp_shp_q14[shp_idx as usize] =
                                    ps_dd.shape_q14[last_smple_idx as usize];
                            }
                        }
                        subfr_nsq = 0;
                    }

                    let start_idx_calc = ps_common.ltp_mem_length
                        - lag
                        - ps_common.predict_lpc_order
                        - LTP_ORDER as i32 / 2;
                    if start_idx_calc < 0 {
                        continue;
                    }
                    let start_idx = start_idx_calc as usize;
                    let xq_start = start_idx + k * ps_common.subfr_length as usize;
                    let filter_len = ps_common.ltp_mem_length as usize - start_idx;

                    if start_idx + filter_len > s_ltp.len()
                        || xq_start + filter_len > ps_nsq.xq.len()
                    {
                        continue;
                    }
                    silk_lpc_analysis_filter(
                        &mut s_ltp[start_idx..],
                        &ps_nsq.xq[xq_start..],
                        a_q12,
                        filter_len,
                        ps_common.predict_lpc_order as usize,
                        0,
                    );
                    ps_nsq.s_ltp_buf_idx = ps_common.ltp_mem_length;
                    ps_nsq.rewhite_flag = 1;
                }
            }

            silk_nsq_del_dec_scale_states(
                ps_common,
                ps_nsq,
                &mut ps_del_dec,
                &x16[x_ptr..],
                &mut x_sc_q10,
                &s_ltp,
                &mut s_ltp_q15,
                k,
                ps_common.n_states_delayed_decision,
                ltp_scale_q14,
                gains_q16,
                pitch_l,
                ps_indices.signal_type as i32,
                decision_delay,
            );

            silk_noise_shape_quantizer_del_dec(
                ps_nsq,
                &mut ps_del_dec,
                ps_indices.signal_type as i32,
                &x_sc_q10,
                pulses,
                pulses_ptr as i32,
                xq_ptr as i32,
                &mut s_ltp_q15,
                &mut delayed_ga_q10,
                a_q12,
                b_q14,
                ar_shp_q13,
                lag,
                harm_shape_fir_packed_q14,
                tilt_q14[k],
                lf_shp_q14[k],
                gains_q16[k],
                lambda_q10,
                offset_q10,
                ps_common.subfr_length,
                subfr_nsq,
                ps_common.shaping_lpc_order,
                ps_common.predict_lpc_order,
                ps_common.warping_q16,
                ps_common.n_states_delayed_decision,
                &mut smpl_buf_idx,
                decision_delay,
                ps_common.frame_counter,
            );

            x_ptr += ps_common.subfr_length as usize;
            pulses_ptr += ps_common.subfr_length as usize;
            xq_ptr += ps_common.subfr_length as usize;
            subfr_nsq += 1;
        }

        let mut rd_min_q10 = ps_del_dec[0].rd_q10;
        let mut winner_ind = 0;
        for k in 1..n_states {
            if ps_del_dec[k].rd_q10 < rd_min_q10 {
                rd_min_q10 = ps_del_dec[k].rd_q10;
                winner_ind = k;
            }
        }

        let ps_dd = &ps_del_dec[winner_ind];
        ps_nsq.s_lf_ar_q14 = ps_dd.lf_ar_q14;
        ps_nsq.s_diff_shp_q14 = ps_dd.diff_q14;
        ps_nsq.lag_prev = pitch_l[ps_common.nb_subfr as usize - 1];

        let gain_q10_final = silk_rshift(gains_q16[ps_common.nb_subfr as usize - 1], 6);
        let mut last_smple_idx = smpl_buf_idx + decision_delay;
        for i in 0..decision_delay {
            last_smple_idx = (last_smple_idx - 1 + DECISION_DELAY as i32) % DECISION_DELAY as i32;
            let pulse_idx = (pulses_ptr as i32 + i - decision_delay) as isize;
            let xq_idx = (xq_ptr as i32 + i - decision_delay) as isize;
            if pulse_idx >= 0 && (pulse_idx as usize) < pulses.len() {
                pulses[pulse_idx as usize] =
                    silk_rshift_round(ps_dd.q_q10[last_smple_idx as usize], 10) as i8;
            }
            if xq_idx >= 0 && (xq_idx as usize) < ps_nsq.xq.len() {
                ps_nsq.xq[xq_idx as usize] = silk_sat16(silk_rshift_round(
                    silk_smulww(ps_dd.xq_q14[last_smple_idx as usize], gain_q10_final),
                    8,
                )) as i16;
            }
            let shp_idx = (ps_nsq.s_ltp_shp_buf_idx - decision_delay + i) as isize;
            if shp_idx >= 0 && (shp_idx as usize) < ps_nsq.s_ltp_shp_q14.len() {
                ps_nsq.s_ltp_shp_q14[shp_idx as usize] = ps_dd.shape_q14[last_smple_idx as usize];
            }
        }

        let subfr_len = ps_common.subfr_length as usize;
        ps_nsq.s_lpc_q14[..NSQ_LPC_BUF_LENGTH]
            .copy_from_slice(&ps_dd.s_lpc_q14[subfr_len..subfr_len + NSQ_LPC_BUF_LENGTH]);
        ps_nsq.s_ar2_q14.copy_from_slice(&ps_dd.s_ar2_q14);

        let ltp_mem_len = ps_common.ltp_mem_length as usize;
        let frame_len = ps_common.frame_length as usize;
        ps_nsq.xq.copy_within(frame_len..frame_len + ltp_mem_len, 0);
        ps_nsq
            .s_ltp_shp_q14
            .copy_within(frame_len..frame_len + ltp_mem_len, 0);

        ps_del_dec[winner_ind].seed_init
    }
}

/// `src/silk/nsq.rs` before the rewrite.
mod reference_nsq {
    use opus_rs::silk::define::*;
    use opus_rs::silk::macros::*;
    use opus_rs::silk::sigproc_fix::*;
    use opus_rs::silk::structs::*;
    use opus_rs::silk::tables::*;

    pub fn silk_nsq(
        ps_enc_c: &SilkEncoderStateCommon,
        nsq: &mut SilkNSQState,
        ps_indices: &SideInfoIndices,
        x16: &[i16],
        pulses: &mut [i8],
        pred_coef_q12: &[i16],
        ltp_coef_q14: &[i16],
        ar_q13: &[i16],
        harm_shape_gain_q14: &[i32],
        tilt_q14: &[i32],
        lf_shp_q14: &[i32],
        gains_q16: &[i32],
        pitch_l: &[i32],
        lambda_q10: i32,
        ltp_scale_q14: i32,
    ) {
        let mut lag: usize;
        let mut start_idx: usize;
        let mut a_q12: &[i16];
        let mut b_q14: &[i16];
        let mut ar_shp_q13: &[i16];
        let mut harm_shape_fir_packed_q14: i32;
        let mut offset_q10: i32;

        let mut s_ltp_q15 = [0i32; LTP_MEM_LENGTH_MS * MAX_FS_KHZ + MAX_FRAME_LENGTH];
        let mut s_ltp = [0i16; LTP_MEM_LENGTH_MS * MAX_FS_KHZ + MAX_FRAME_LENGTH];
        let mut x_sc_q10 = [0i32; MAX_SUB_FRAME_LENGTH];

        nsq.rand_seed = ps_indices.seed as i32;

        lag = nsq.lag_prev as usize;

        let lsf_interpolation_flag: i32 = if ps_indices.nlsf_interp_coef_q2 == 4 {
            0
        } else {
            1
        };

        nsq.s_ltp_shp_buf_idx = ps_enc_c.ltp_mem_length;
        nsq.s_ltp_buf_idx = ps_enc_c.ltp_mem_length;

        let pxq_idx = ps_enc_c.ltp_mem_length as usize;

        for k in 0..ps_enc_c.nb_subfr as usize {
            a_q12 = &pred_coef_q12
                [((k >> 1) | (1 - lsf_interpolation_flag as usize)) * MAX_LPC_ORDER..];
            b_q14 = &ltp_coef_q14[k * LTP_ORDER..];
            ar_shp_q13 = &ar_q13[k * MAX_SHAPE_LPC_ORDER..];

            nsq.rewhite_flag = 0;
            if ps_indices.signal_type == TYPE_VOICED as i8 {
                lag = pitch_l[k] as usize;

                if (k & (3 - (lsf_interpolation_flag << 1) as usize)) == 0 {
                    // Clamp: lag/order are geometry-bounded for valid streams; a
                    // corrupted state must not yield a negative start index
                    // (issue #27 deep-scan).
                    let start_idx_signed = (ps_enc_c.ltp_mem_length
                        - lag as i32
                        - ps_enc_c.predict_lpc_order
                        - (LTP_ORDER as i32 / 2))
                        .max(0);
                    start_idx = start_idx_signed as usize;

                    silk_lpc_analysis_filter(
                        &mut s_ltp[start_idx..],
                        &nsq.xq[start_idx + k * ps_enc_c.subfr_length as usize..],
                        a_q12,
                        (ps_enc_c.ltp_mem_length as usize) - start_idx,
                        ps_enc_c.predict_lpc_order as usize,
                        0,
                    );
                    nsq.rewhite_flag = 1;
                    nsq.s_ltp_buf_idx = ps_enc_c.ltp_mem_length;
                }
            }

            harm_shape_fir_packed_q14 = harm_shape_gain_q14[k] >> 2;
            harm_shape_fir_packed_q14 |= (harm_shape_gain_q14[k] >> 1) << 16;

            offset_q10 = SILK_QUANT_OFFSETS_Q10[(ps_indices.signal_type >> 1) as usize]
                [ps_indices.quant_offset_type as usize] as i32;

            silk_nsq_scale_states(
                ps_enc_c,
                nsq,
                &x16[k * ps_enc_c.subfr_length as usize..],
                &mut x_sc_q10,
                &s_ltp,
                &mut s_ltp_q15,
                k,
                ltp_scale_q14,
                gains_q16,
                pitch_l,
                ps_indices.signal_type as i32,
            );

            silk_noise_shape_quantizer(
                nsq,
                ps_indices.signal_type as i32,
                &x_sc_q10,
                &mut pulses[k * ps_enc_c.subfr_length as usize..],
                pxq_idx + k * ps_enc_c.subfr_length as usize,
                &mut s_ltp_q15,
                a_q12,
                b_q14,
                ar_shp_q13,
                lag,
                harm_shape_fir_packed_q14,
                tilt_q14[k],
                lf_shp_q14[k],
                gains_q16[k],
                lambda_q10,
                offset_q10,
                ps_enc_c.subfr_length as usize,
                ps_enc_c.shaping_lpc_order as usize,
                ps_enc_c.predict_lpc_order as usize,
                k,
            );
        }

        nsq.lag_prev = pitch_l[ps_enc_c.nb_subfr as usize - 1];

        let frame_length = ps_enc_c.frame_length as usize;
        let ltp_mem_length = ps_enc_c.ltp_mem_length as usize;

        nsq.xq
            .copy_within(frame_length..frame_length + ltp_mem_length, 0);
        nsq.s_ltp_shp_q14
            .copy_within(frame_length..frame_length + ltp_mem_length, 0);
    }

    pub fn silk_nsq_scale_states(
        ps_enc_c: &SilkEncoderStateCommon,
        nsq: &mut SilkNSQState,
        x16: &[i16],
        x_sc_q10: &mut [i32],
        s_ltp: &[i16],
        s_ltp_q15: &mut [i32],
        k: usize,
        ltp_scale_q14: i32,
        gains_q16: &[i32],
        pitch_l: &[i32],
        signal_type: i32,
    ) {
        let subfr_length = ps_enc_c.subfr_length as usize;
        let ltp_mem_length = ps_enc_c.ltp_mem_length as usize;
        let lag = pitch_l[k] as usize;

        let mut inv_gain_q31 =
            silk_inverse32_varq(if gains_q16[k] > 1 { gains_q16[k] } else { 1 }, 47);

        let inv_gain_q26 = silk_rshift_round(inv_gain_q31, 5);
        for i in 0..subfr_length {
            x_sc_q10[i] = silk_smulww(x16[i] as i32, inv_gain_q26);
        }

        if nsq.rewhite_flag != 0 {
            if k == 0 {
                inv_gain_q31 = silk_lshift(silk_smulwb(inv_gain_q31, ltp_scale_q14), 2);
            }
            let start = (nsq.s_ltp_buf_idx as usize)
                .wrapping_sub(lag)
                .wrapping_sub(LTP_ORDER / 2);
            for i in start..nsq.s_ltp_buf_idx as usize {
                s_ltp_q15[i] = silk_smulwb(inv_gain_q31, s_ltp[i] as i32);
            }
        }

        if gains_q16[k] != nsq.prev_gain_q16 {
            let gain_adj_q16 = silk_div32_varq(
                nsq.prev_gain_q16,
                if gains_q16[k] > 1 { gains_q16[k] } else { 1 },
                16,
            );

            let shp_start = (nsq.s_ltp_shp_buf_idx - ltp_mem_length as i32) as usize;
            for i in shp_start..nsq.s_ltp_shp_buf_idx as usize {
                nsq.s_ltp_shp_q14[i] = silk_smulww(gain_adj_q16, nsq.s_ltp_shp_q14[i]);
            }

            if signal_type == TYPE_VOICED && nsq.rewhite_flag == 0 {
                let ltp_start = (nsq.s_ltp_buf_idx as usize)
                    .wrapping_sub(lag)
                    .wrapping_sub(LTP_ORDER / 2);
                for i in ltp_start..nsq.s_ltp_buf_idx as usize {
                    s_ltp_q15[i] = silk_smulww(gain_adj_q16, s_ltp_q15[i]);
                }
            }

            nsq.s_lf_ar_q14 = silk_smulww(gain_adj_q16, nsq.s_lf_ar_q14);
            nsq.s_diff_shp_q14 = silk_smulww(gain_adj_q16, nsq.s_diff_shp_q14);

            for i in 0..NSQ_LPC_BUF_LENGTH {
                nsq.s_lpc_q14[i] = silk_smulww(gain_adj_q16, nsq.s_lpc_q14[i]);
            }
            for i in 0..MAX_SHAPE_LPC_ORDER {
                nsq.s_ar2_q14[i] = silk_smulww(gain_adj_q16, nsq.s_ar2_q14[i]);
            }

            nsq.prev_gain_q16 = gains_q16[k];
        }

        nsq.prev_sig_type = signal_type as i8;
    }

    #[inline(always)]
    fn silk_nsq_noise_shape_feedback_loop(
        data0_val: i32,
        data1: &mut [i32],
        coef: &[i16],
        order: usize,
    ) -> i32 {
        if order == 12 {
            return silk_nsq_noise_shape_feedback_loop_order12(data0_val, data1, coef);
        }

        unsafe {
            let mut tmp2 = data0_val;
            let mut tmp1 = *data1.get_unchecked(0);
            *data1.get_unchecked_mut(0) = tmp2;

            let mut out = (order as i32) >> 1;
            out = silk_smlawb(out, tmp2, *coef.get_unchecked(0) as i32);

            let mut j = 2;
            while j < order {
                tmp2 = *data1.get_unchecked(j - 1);
                *data1.get_unchecked_mut(j - 1) = tmp1;
                out = silk_smlawb(out, tmp1, *coef.get_unchecked(j - 1) as i32);
                tmp1 = *data1.get_unchecked(j);
                *data1.get_unchecked_mut(j) = tmp2;
                out = silk_smlawb(out, tmp2, *coef.get_unchecked(j) as i32);
                j += 2;
            }
            *data1.get_unchecked_mut(order - 1) = tmp1;
            out = silk_smlawb(out, tmp1, *coef.get_unchecked(order - 1) as i32);

            out <<= 1;
            out
        }
    }

    #[inline(always)]
    fn silk_nsq_noise_shape_feedback_loop_order12(
        data0_val: i32,
        data1: &mut [i32],
        coef: &[i16],
    ) -> i32 {
        unsafe {
            let d0 = data0_val;
            let d1 = *data1.get_unchecked(0);
            let d2 = *data1.get_unchecked(1);
            let d3 = *data1.get_unchecked(2);
            let d4 = *data1.get_unchecked(3);
            let d5 = *data1.get_unchecked(4);
            let d6 = *data1.get_unchecked(5);
            let d7 = *data1.get_unchecked(6);
            let d8 = *data1.get_unchecked(7);
            let d9 = *data1.get_unchecked(8);
            let d10 = *data1.get_unchecked(9);
            let d11 = *data1.get_unchecked(10);

            *data1.get_unchecked_mut(0) = d0;
            *data1.get_unchecked_mut(1) = d1;
            *data1.get_unchecked_mut(2) = d2;
            *data1.get_unchecked_mut(3) = d3;
            *data1.get_unchecked_mut(4) = d4;
            *data1.get_unchecked_mut(5) = d5;
            *data1.get_unchecked_mut(6) = d6;
            *data1.get_unchecked_mut(7) = d7;
            *data1.get_unchecked_mut(8) = d8;
            *data1.get_unchecked_mut(9) = d9;
            *data1.get_unchecked_mut(10) = d10;
            *data1.get_unchecked_mut(11) = d11;

            let mut out = 6i32;
            out = silk_smlawb(out, d0, *coef.get_unchecked(0) as i32);
            out = silk_smlawb(out, d1, *coef.get_unchecked(1) as i32);
            out = silk_smlawb(out, d2, *coef.get_unchecked(2) as i32);
            out = silk_smlawb(out, d3, *coef.get_unchecked(3) as i32);
            out = silk_smlawb(out, d4, *coef.get_unchecked(4) as i32);
            out = silk_smlawb(out, d5, *coef.get_unchecked(5) as i32);
            out = silk_smlawb(out, d6, *coef.get_unchecked(6) as i32);
            out = silk_smlawb(out, d7, *coef.get_unchecked(7) as i32);
            out = silk_smlawb(out, d8, *coef.get_unchecked(8) as i32);
            out = silk_smlawb(out, d9, *coef.get_unchecked(9) as i32);
            out = silk_smlawb(out, d10, *coef.get_unchecked(10) as i32);
            out = silk_smlawb(out, d11, *coef.get_unchecked(11) as i32);

            out <<= 1;
            out
        }
    }

    fn silk_noise_shape_quantizer(
        nsq: &mut SilkNSQState,
        signal_type: i32,
        x_sc_q10: &[i32],
        pulses: &mut [i8],
        xq_offset: usize,
        s_ltp_q15: &mut [i32],
        a_q12: &[i16],
        b_q14: &[i16],
        ar_shp_q13: &[i16],
        lag: usize,
        harm_shape_fir_packed_q14: i32,
        tilt_q14: i32,
        lf_shp_q14: i32,
        gain_q16: i32,
        lambda_q10: i32,
        offset_q10: i32,
        subfr_length: usize,
        shaping_lpc_order: usize,
        predict_lpc_order: usize,
        _subfr_idx: usize,
    ) {
        let gain_q10 = gain_q16 >> 6;

        let shp_lag_base =
            (nsq.s_ltp_shp_buf_idx as usize).wrapping_sub(lag) + HARM_SHAPE_FIR_TAPS / 2;
        let pred_lag_base = (nsq.s_ltp_buf_idx as usize).wrapping_sub(lag) + LTP_ORDER / 2;

        if signal_type == TYPE_VOICED {
            silk_noise_shape_quantizer_voiced(
                nsq,
                x_sc_q10,
                pulses,
                xq_offset,
                s_ltp_q15,
                a_q12,
                b_q14,
                ar_shp_q13,
                lag,
                harm_shape_fir_packed_q14,
                tilt_q14,
                lf_shp_q14,
                gain_q10,
                lambda_q10,
                offset_q10,
                subfr_length,
                shaping_lpc_order,
                predict_lpc_order,
                shp_lag_base,
                pred_lag_base,
            );
        } else {
            silk_noise_shape_quantizer_unvoiced(
                nsq,
                x_sc_q10,
                pulses,
                xq_offset,
                s_ltp_q15,
                a_q12,
                ar_shp_q13,
                tilt_q14,
                lf_shp_q14,
                gain_q10,
                lambda_q10,
                offset_q10,
                subfr_length,
                shaping_lpc_order,
                predict_lpc_order,
            );
        }

        nsq.s_lpc_q14
            .copy_within(subfr_length..subfr_length + NSQ_LPC_BUF_LENGTH, 0);
    }

    #[inline(always)]
    fn lpc_pred_order6(s_lpc: &[i32], a_q12: &[i16], idx: usize) -> i32 {
        let mut pred = 3i32;

        unsafe {
            pred = silk_smlawb(
                pred,
                *s_lpc.get_unchecked(idx),
                *a_q12.get_unchecked(0) as i32,
            );
            pred = silk_smlawb(
                pred,
                *s_lpc.get_unchecked(idx - 1),
                *a_q12.get_unchecked(1) as i32,
            );
            pred = silk_smlawb(
                pred,
                *s_lpc.get_unchecked(idx - 2),
                *a_q12.get_unchecked(2) as i32,
            );
            pred = silk_smlawb(
                pred,
                *s_lpc.get_unchecked(idx - 3),
                *a_q12.get_unchecked(3) as i32,
            );
            pred = silk_smlawb(
                pred,
                *s_lpc.get_unchecked(idx - 4),
                *a_q12.get_unchecked(4) as i32,
            );
            pred = silk_smlawb(
                pred,
                *s_lpc.get_unchecked(idx - 5),
                *a_q12.get_unchecked(5) as i32,
            );
        }
        pred
    }

    #[inline(always)]
    fn lpc_pred_order10(s_lpc: &[i32], a_q12: &[i16], idx: usize) -> i32 {
        let mut pred = 5i32;

        unsafe {
            pred = silk_smlawb(
                pred,
                *s_lpc.get_unchecked(idx),
                *a_q12.get_unchecked(0) as i32,
            );
            pred = silk_smlawb(
                pred,
                *s_lpc.get_unchecked(idx - 1),
                *a_q12.get_unchecked(1) as i32,
            );
            pred = silk_smlawb(
                pred,
                *s_lpc.get_unchecked(idx - 2),
                *a_q12.get_unchecked(2) as i32,
            );
            pred = silk_smlawb(
                pred,
                *s_lpc.get_unchecked(idx - 3),
                *a_q12.get_unchecked(3) as i32,
            );
            pred = silk_smlawb(
                pred,
                *s_lpc.get_unchecked(idx - 4),
                *a_q12.get_unchecked(4) as i32,
            );
            pred = silk_smlawb(
                pred,
                *s_lpc.get_unchecked(idx - 5),
                *a_q12.get_unchecked(5) as i32,
            );
            pred = silk_smlawb(
                pred,
                *s_lpc.get_unchecked(idx - 6),
                *a_q12.get_unchecked(6) as i32,
            );
            pred = silk_smlawb(
                pred,
                *s_lpc.get_unchecked(idx - 7),
                *a_q12.get_unchecked(7) as i32,
            );
            pred = silk_smlawb(
                pred,
                *s_lpc.get_unchecked(idx - 8),
                *a_q12.get_unchecked(8) as i32,
            );
            pred = silk_smlawb(
                pred,
                *s_lpc.get_unchecked(idx - 9),
                *a_q12.get_unchecked(9) as i32,
            );
        }
        pred
    }

    #[inline(always)]
    fn lpc_pred_order16(s_lpc: &[i32], a_q12: &[i16], idx: usize) -> i32 {
        let mut pred = 8i32;

        unsafe {
            pred = silk_smlawb(
                pred,
                *s_lpc.get_unchecked(idx),
                *a_q12.get_unchecked(0) as i32,
            );
            pred = silk_smlawb(
                pred,
                *s_lpc.get_unchecked(idx - 1),
                *a_q12.get_unchecked(1) as i32,
            );
            pred = silk_smlawb(
                pred,
                *s_lpc.get_unchecked(idx - 2),
                *a_q12.get_unchecked(2) as i32,
            );
            pred = silk_smlawb(
                pred,
                *s_lpc.get_unchecked(idx - 3),
                *a_q12.get_unchecked(3) as i32,
            );
            pred = silk_smlawb(
                pred,
                *s_lpc.get_unchecked(idx - 4),
                *a_q12.get_unchecked(4) as i32,
            );
            pred = silk_smlawb(
                pred,
                *s_lpc.get_unchecked(idx - 5),
                *a_q12.get_unchecked(5) as i32,
            );
            pred = silk_smlawb(
                pred,
                *s_lpc.get_unchecked(idx - 6),
                *a_q12.get_unchecked(6) as i32,
            );
            pred = silk_smlawb(
                pred,
                *s_lpc.get_unchecked(idx - 7),
                *a_q12.get_unchecked(7) as i32,
            );
            pred = silk_smlawb(
                pred,
                *s_lpc.get_unchecked(idx - 8),
                *a_q12.get_unchecked(8) as i32,
            );
            pred = silk_smlawb(
                pred,
                *s_lpc.get_unchecked(idx - 9),
                *a_q12.get_unchecked(9) as i32,
            );
            pred = silk_smlawb(
                pred,
                *s_lpc.get_unchecked(idx - 10),
                *a_q12.get_unchecked(10) as i32,
            );
            pred = silk_smlawb(
                pred,
                *s_lpc.get_unchecked(idx - 11),
                *a_q12.get_unchecked(11) as i32,
            );
            pred = silk_smlawb(
                pred,
                *s_lpc.get_unchecked(idx - 12),
                *a_q12.get_unchecked(12) as i32,
            );
            pred = silk_smlawb(
                pred,
                *s_lpc.get_unchecked(idx - 13),
                *a_q12.get_unchecked(13) as i32,
            );
            pred = silk_smlawb(
                pred,
                *s_lpc.get_unchecked(idx - 14),
                *a_q12.get_unchecked(14) as i32,
            );
            pred = silk_smlawb(
                pred,
                *s_lpc.get_unchecked(idx - 15),
                *a_q12.get_unchecked(15) as i32,
            );
        }
        pred
    }

    #[inline(always)]
    fn lpc_pred_generic(s_lpc: &[i32], a_q12: &[i16], idx: usize, order: usize) -> i32 {
        let mut pred = (order as i32) >> 1;
        for j in 0..order {
            pred = unsafe {
                silk_smlawb(
                    pred,
                    *s_lpc.get_unchecked(idx - j),
                    *a_q12.get_unchecked(j) as i32,
                )
            };
        }
        pred
    }

    #[inline(always)]
    fn silk_noise_shape_quantizer_voiced(
        nsq: &mut SilkNSQState,
        x_sc_q10: &[i32],
        pulses: &mut [i8],
        xq_offset: usize,
        s_ltp_q15: &mut [i32],
        a_q12: &[i16],
        b_q14: &[i16],
        ar_shp_q13: &[i16],
        _lag: usize,
        harm_shape_fir_packed_q14: i32,
        tilt_q14: i32,
        lf_shp_q14: i32,
        gain_q10: i32,
        lambda_q10: i32,
        offset_q10: i32,
        subfr_length: usize,
        shaping_lpc_order: usize,
        predict_lpc_order: usize,
        shp_lag_base: usize,
        pred_lag_base: usize,
    ) {
        let mut ltp_pred_q13: i32;
        let mut lpc_pred_q10: i32;
        let mut n_ar_q12: i32;
        let mut n_lf_q12: i32;
        let mut n_ltp_q13: i32;
        let mut r_q10: i32;
        let mut rr_q10: i32;
        let mut q1_q0: i32;
        let mut q1_q10: i32;
        let mut q2_q10: i32;
        let mut rd1_q20: i32;
        let mut rd2_q20: i32;
        let mut exc_q14: i32;
        let mut lpc_exc_q14: i32;
        let mut xq_q14: i32;
        let mut tmp1: i32;
        let mut tmp2: i32;
        let mut s_lf_ar_shp_q14: i32;

        let mut rand_seed = nsq.rand_seed;
        let mut s_ltp_shp_buf_idx = nsq.s_ltp_shp_buf_idx as usize;
        let mut s_ltp_buf_idx = nsq.s_ltp_buf_idx as usize;
        let mut s_lf_ar_q14 = nsq.s_lf_ar_q14;
        let mut s_diff_shp_q14 = nsq.s_diff_shp_q14;

        for i in 0..subfr_length {
            rand_seed = silk_rand(rand_seed);

            let ps_lpc_idx = NSQ_LPC_BUF_LENGTH - 1 + i;
            lpc_pred_q10 = if predict_lpc_order == 16 {
                lpc_pred_order16(&nsq.s_lpc_q14, a_q12, ps_lpc_idx)
            } else if predict_lpc_order == 10 {
                lpc_pred_order10(&nsq.s_lpc_q14, a_q12, ps_lpc_idx)
            } else if predict_lpc_order == 6 {
                lpc_pred_order6(&nsq.s_lpc_q14, a_q12, ps_lpc_idx)
            } else {
                lpc_pred_generic(&nsq.s_lpc_q14, a_q12, ps_lpc_idx, predict_lpc_order)
            };

            ltp_pred_q13 = 2;
            ltp_pred_q13 = silk_smlawb(ltp_pred_q13, s_ltp_q15[pred_lag_base + i], b_q14[0] as i32);
            ltp_pred_q13 = silk_smlawb(
                ltp_pred_q13,
                s_ltp_q15[pred_lag_base + i - 1],
                b_q14[1] as i32,
            );
            ltp_pred_q13 = silk_smlawb(
                ltp_pred_q13,
                s_ltp_q15[pred_lag_base + i - 2],
                b_q14[2] as i32,
            );
            ltp_pred_q13 = silk_smlawb(
                ltp_pred_q13,
                s_ltp_q15[pred_lag_base + i - 3],
                b_q14[3] as i32,
            );
            ltp_pred_q13 = silk_smlawb(
                ltp_pred_q13,
                s_ltp_q15[pred_lag_base + i - 4],
                b_q14[4] as i32,
            );

            n_ar_q12 = silk_nsq_noise_shape_feedback_loop(
                s_diff_shp_q14,
                &mut nsq.s_ar2_q14,
                ar_shp_q13,
                shaping_lpc_order,
            );
            n_ar_q12 = silk_smlawb(n_ar_q12, s_lf_ar_q14, tilt_q14);

            n_lf_q12 = silk_smulwb(nsq.s_ltp_shp_q14[s_ltp_shp_buf_idx - 1], lf_shp_q14);
            n_lf_q12 = silk_smlawt(n_lf_q12, s_lf_ar_q14, lf_shp_q14);

            tmp1 = (lpc_pred_q10 << 2).wrapping_sub(n_ar_q12);
            tmp1 = tmp1.wrapping_sub(n_lf_q12);

            let shp_idx = shp_lag_base + i;
            n_ltp_q13 = silk_smulwb(
                silk_add_sat32(nsq.s_ltp_shp_q14[shp_idx], nsq.s_ltp_shp_q14[shp_idx - 2]),
                harm_shape_fir_packed_q14,
            );
            n_ltp_q13 = silk_smlawt(
                n_ltp_q13,
                nsq.s_ltp_shp_q14[shp_idx - 1],
                harm_shape_fir_packed_q14,
            );
            n_ltp_q13 <<= 1;

            tmp2 = ltp_pred_q13 - n_ltp_q13;
            tmp1 = tmp2.wrapping_add(tmp1 << 1);
            tmp1 = silk_rshift_round(tmp1, 3);

            r_q10 = x_sc_q10[i] - tmp1;

            if rand_seed < 0 {
                r_q10 = -r_q10;
            }
            r_q10 = silk_limit_32(r_q10, -(31 << 10), 30 << 10);

            q1_q10 = r_q10 - offset_q10;
            q1_q0 = q1_q10 >> 10;
            if lambda_q10 > 2048 {
                let rdo_offset = lambda_q10 / 2 - 512;
                if q1_q10 > rdo_offset {
                    q1_q0 = (q1_q10 - rdo_offset) >> 10;
                } else if q1_q10 < -rdo_offset {
                    q1_q0 = (q1_q10 + rdo_offset) >> 10;
                } else if q1_q10 < 0 {
                    q1_q0 = -1;
                } else {
                    q1_q0 = 0;
                }
            }

            if q1_q0 > 0 {
                q1_q10 = (q1_q0 << 10) - QUANT_LEVEL_ADJUST_Q10 + offset_q10;
                q2_q10 = q1_q10 + 1024;
                rd1_q20 = silk_smulbb(q1_q10, lambda_q10);
                rd2_q20 = silk_smulbb(q2_q10, lambda_q10);
            } else if q1_q0 == 0 {
                q1_q10 = offset_q10;
                q2_q10 = q1_q10 + 1024 - QUANT_LEVEL_ADJUST_Q10;
                rd1_q20 = silk_smulbb(q1_q10, lambda_q10);
                rd2_q20 = silk_smulbb(q2_q10, lambda_q10);
            } else if q1_q0 == -1 {
                q2_q10 = offset_q10;
                q1_q10 = q2_q10 - (1024 - QUANT_LEVEL_ADJUST_Q10);
                rd1_q20 = silk_smulbb(-q1_q10, lambda_q10);
                rd2_q20 = silk_smulbb(q2_q10, lambda_q10);
            } else {
                q1_q10 = (q1_q0 << 10) + QUANT_LEVEL_ADJUST_Q10 + offset_q10;
                q2_q10 = q1_q10 + 1024;
                rd1_q20 = silk_smulbb(-q1_q10, lambda_q10);
                rd2_q20 = silk_smulbb(-q2_q10, lambda_q10);
            }
            rr_q10 = r_q10 - q1_q10;
            rd1_q20 = silk_smlabb(rd1_q20, rr_q10, rr_q10);
            rr_q10 = r_q10 - q2_q10;
            rd2_q20 = silk_smlabb(rd2_q20, rr_q10, rr_q10);

            if rd2_q20 < rd1_q20 {
                q1_q10 = q2_q10;
            }

            pulses[i] = silk_rshift_round(q1_q10, 10) as i8;

            exc_q14 = q1_q10 << 4;
            if rand_seed < 0 {
                exc_q14 = -exc_q14;
            }

            lpc_exc_q14 = silk_add_lshift32(exc_q14, ltp_pred_q13, 1);
            xq_q14 = lpc_exc_q14.wrapping_add(lpc_pred_q10 << 4);

            nsq.xq[xq_offset + i] =
                silk_sat16(silk_rshift_round(silk_smulww(xq_q14, gain_q10), 8)) as i16;

            nsq.s_lpc_q14[NSQ_LPC_BUF_LENGTH + i] = xq_q14;
            s_diff_shp_q14 = xq_q14.wrapping_sub(x_sc_q10[i] << 4);
            s_lf_ar_shp_q14 = s_diff_shp_q14.wrapping_sub(n_ar_q12 << 2);
            s_lf_ar_q14 = s_lf_ar_shp_q14;

            nsq.s_ltp_shp_q14[s_ltp_shp_buf_idx] = s_lf_ar_shp_q14.wrapping_sub(n_lf_q12 << 2);
            s_ltp_q15[s_ltp_buf_idx] = lpc_exc_q14 << 1;
            s_ltp_shp_buf_idx += 1;
            s_ltp_buf_idx += 1;

            rand_seed = rand_seed.wrapping_add(pulses[i] as i32);
        }

        nsq.rand_seed = rand_seed;
        nsq.s_ltp_shp_buf_idx = s_ltp_shp_buf_idx as i32;
        nsq.s_ltp_buf_idx = s_ltp_buf_idx as i32;
        nsq.s_lf_ar_q14 = s_lf_ar_q14;
        nsq.s_diff_shp_q14 = s_diff_shp_q14;
    }

    #[inline(always)]
    fn silk_noise_shape_quantizer_unvoiced(
        nsq: &mut SilkNSQState,
        x_sc_q10: &[i32],
        pulses: &mut [i8],
        xq_offset: usize,
        s_ltp_q15: &mut [i32],
        a_q12: &[i16],
        ar_shp_q13: &[i16],
        tilt_q14: i32,
        lf_shp_q14: i32,
        gain_q10: i32,
        lambda_q10: i32,
        offset_q10: i32,
        subfr_length: usize,
        shaping_lpc_order: usize,
        predict_lpc_order: usize,
    ) {
        let mut lpc_pred_q10: i32;
        let mut n_ar_q12: i32;
        let mut n_lf_q12: i32;
        let mut r_q10: i32;
        let mut rr_q10: i32;
        let mut q1_q0: i32;
        let mut q1_q10: i32;
        let mut q2_q10: i32;
        let mut rd1_q20: i32;
        let mut rd2_q20: i32;
        let mut exc_q14: i32;
        let mut lpc_exc_q14: i32;
        let mut xq_q14: i32;
        let mut tmp1: i32;
        let mut s_lf_ar_shp_q14: i32;

        let mut rand_seed = nsq.rand_seed;
        let mut s_ltp_shp_buf_idx = nsq.s_ltp_shp_buf_idx as usize;
        let mut s_ltp_buf_idx = nsq.s_ltp_buf_idx as usize;
        let mut s_lf_ar_q14 = nsq.s_lf_ar_q14;
        let mut s_diff_shp_q14 = nsq.s_diff_shp_q14;

        for i in 0..subfr_length {
            rand_seed = silk_rand(rand_seed);

            let ps_lpc_idx = NSQ_LPC_BUF_LENGTH - 1 + i;
            lpc_pred_q10 = if predict_lpc_order == 16 {
                lpc_pred_order16(&nsq.s_lpc_q14, a_q12, ps_lpc_idx)
            } else if predict_lpc_order == 10 {
                lpc_pred_order10(&nsq.s_lpc_q14, a_q12, ps_lpc_idx)
            } else if predict_lpc_order == 6 {
                lpc_pred_order6(&nsq.s_lpc_q14, a_q12, ps_lpc_idx)
            } else {
                lpc_pred_generic(&nsq.s_lpc_q14, a_q12, ps_lpc_idx, predict_lpc_order)
            };

            let ltp_pred_q13: i32 = 0;

            n_ar_q12 = silk_nsq_noise_shape_feedback_loop(
                s_diff_shp_q14,
                &mut nsq.s_ar2_q14,
                ar_shp_q13,
                shaping_lpc_order,
            );
            n_ar_q12 = silk_smlawb(n_ar_q12, s_lf_ar_q14, tilt_q14);

            n_lf_q12 = silk_smulwb(nsq.s_ltp_shp_q14[s_ltp_shp_buf_idx - 1], lf_shp_q14);
            n_lf_q12 = silk_smlawt(n_lf_q12, s_lf_ar_q14, lf_shp_q14);

            tmp1 = (lpc_pred_q10 << 2).wrapping_sub(n_ar_q12);
            tmp1 = tmp1.wrapping_sub(n_lf_q12);

            tmp1 = silk_rshift_round(tmp1, 2);

            r_q10 = x_sc_q10[i] - tmp1;

            if rand_seed < 0 {
                r_q10 = -r_q10;
            }
            r_q10 = silk_limit_32(r_q10, -(31 << 10), 30 << 10);

            q1_q10 = r_q10 - offset_q10;
            q1_q0 = q1_q10 >> 10;
            if lambda_q10 > 2048 {
                let rdo_offset = lambda_q10 / 2 - 512;
                if q1_q10 > rdo_offset {
                    q1_q0 = (q1_q10 - rdo_offset) >> 10;
                } else if q1_q10 < -rdo_offset {
                    q1_q0 = (q1_q10 + rdo_offset) >> 10;
                } else if q1_q10 < 0 {
                    q1_q0 = -1;
                } else {
                    q1_q0 = 0;
                }
            }

            if q1_q0 > 0 {
                q1_q10 = (q1_q0 << 10) - QUANT_LEVEL_ADJUST_Q10 + offset_q10;
                q2_q10 = q1_q10 + 1024;
                rd1_q20 = silk_smulbb(q1_q10, lambda_q10);
                rd2_q20 = silk_smulbb(q2_q10, lambda_q10);
            } else if q1_q0 == 0 {
                q1_q10 = offset_q10;
                q2_q10 = q1_q10 + 1024 - QUANT_LEVEL_ADJUST_Q10;
                rd1_q20 = silk_smulbb(q1_q10, lambda_q10);
                rd2_q20 = silk_smulbb(q2_q10, lambda_q10);
            } else if q1_q0 == -1 {
                q2_q10 = offset_q10;
                q1_q10 = q2_q10 - (1024 - QUANT_LEVEL_ADJUST_Q10);
                rd1_q20 = silk_smulbb(-q1_q10, lambda_q10);
                rd2_q20 = silk_smulbb(q2_q10, lambda_q10);
            } else {
                q1_q10 = (q1_q0 << 10) + QUANT_LEVEL_ADJUST_Q10 + offset_q10;
                q2_q10 = q1_q10 + 1024;
                rd1_q20 = silk_smulbb(-q1_q10, lambda_q10);
                rd2_q20 = silk_smulbb(-q2_q10, lambda_q10);
            }
            rr_q10 = r_q10 - q1_q10;
            rd1_q20 = silk_smlabb(rd1_q20, rr_q10, rr_q10);
            rr_q10 = r_q10 - q2_q10;
            rd2_q20 = silk_smlabb(rd2_q20, rr_q10, rr_q10);

            if rd2_q20 < rd1_q20 {
                q1_q10 = q2_q10;
            }

            pulses[i] = silk_rshift_round(q1_q10, 10) as i8;

            exc_q14 = q1_q10 << 4;
            if rand_seed < 0 {
                exc_q14 = -exc_q14;
            }

            lpc_exc_q14 = silk_add_lshift32(exc_q14, ltp_pred_q13, 1);
            xq_q14 = lpc_exc_q14.wrapping_add(lpc_pred_q10 << 4);

            nsq.xq[xq_offset + i] =
                silk_sat16(silk_rshift_round(silk_smulww(xq_q14, gain_q10), 8)) as i16;

            nsq.s_lpc_q14[NSQ_LPC_BUF_LENGTH + i] = xq_q14;
            s_diff_shp_q14 = xq_q14.wrapping_sub(x_sc_q10[i] << 4);
            s_lf_ar_shp_q14 = s_diff_shp_q14.wrapping_sub(n_ar_q12 << 2);
            s_lf_ar_q14 = s_lf_ar_shp_q14;

            nsq.s_ltp_shp_q14[s_ltp_shp_buf_idx] = s_lf_ar_shp_q14.wrapping_sub(n_lf_q12 << 2);
            s_ltp_q15[s_ltp_buf_idx] = lpc_exc_q14 << 1;
            s_ltp_shp_buf_idx += 1;
            s_ltp_buf_idx += 1;

            rand_seed = rand_seed.wrapping_add(pulses[i] as i32);
        }

        nsq.rand_seed = rand_seed;
        nsq.s_ltp_shp_buf_idx = s_ltp_shp_buf_idx as i32;
        nsq.s_ltp_buf_idx = s_ltp_buf_idx as i32;
        nsq.s_lf_ar_q14 = s_lf_ar_q14;
        nsq.s_diff_shp_q14 = s_diff_shp_q14;
    }
}

// ---------------------------------------------------------------------------
// Generated inputs
// ---------------------------------------------------------------------------

/// Small deterministic generator (splitmix64), so that the test needs no
/// dependency and repeats exactly.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `lo..=hi`.
    fn range(&mut self, lo: i64, hi: i64) -> i64 {
        lo + (self.next_u64() % (hi - lo + 1) as u64) as i64
    }

    /// Uniform in `0.0..1.0`.
    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn chance(&mut self, percent: i64) -> bool {
        self.range(0, 99) < percent
    }
}

/// Coefficients of a stable all-pole filter, made from random reflection
/// coefficients with the step-up recursion. `out[j]` multiplies the sample
/// `j + 1` steps back, scaled by `2^q`.
fn random_filter(rng: &mut Rng, order: usize, max_reflection: f64, q: u32, out: &mut [i16]) {
    let mut a = [0f64; 24];
    for m in 0..order {
        let k = (rng.unit() * 2.0 - 1.0) * max_reflection / (1.0 + m as f64 * 0.15);
        let prev = a;
        for j in 0..m {
            a[j] = prev[j] - k * prev[m - 1 - j];
        }
        a[m] = k;
    }
    for j in 0..order {
        let v = (a[j] * (1u32 << q) as f64).round();
        out[j] = v.clamp(i16::MIN as f64, i16::MAX as f64) as i16;
    }
}

/// How the values of a frame are drawn.
#[derive(Clone, Copy, PartialEq)]
enum Kind {
    /// Values as the encoder produces them.
    Plausible,
    /// Full-range random coefficients and gains. The filters are unstable and
    /// the arithmetic wraps and saturates. Only used where the build does not
    /// check for overflow, because the previous implementation uses plain
    /// `+` and `-` in places.
    Extreme,
}

#[derive(Clone, Copy)]
struct Config {
    fs_khz: i32,
    nb_subfr: i32,
    n_states: i32,
    shaping_order: i32,
    predict_order: i32,
    warping_q16: i32,
}

impl Config {
    fn common(&self) -> Box<SilkEncoderStateCommon> {
        let mut c = Box::new(SilkEncoderStateCommon::default());
        c.fs_khz = self.fs_khz;
        c.nb_subfr = self.nb_subfr;
        c.subfr_length = 5 * self.fs_khz;
        c.frame_length = c.subfr_length * self.nb_subfr;
        c.ltp_mem_length = 20 * self.fs_khz;
        c.predict_lpc_order = self.predict_order;
        c.shaping_lpc_order = self.shaping_order;
        c.warping_q16 = self.warping_q16;
        c.n_states_delayed_decision = self.n_states;
        c
    }
}

struct Frame {
    indices: SideInfoIndices,
    x16: [i16; MAX_FRAME_LENGTH],
    pred_coef_q12: [i16; 2 * MAX_LPC_ORDER],
    ltp_coef_q14: [i16; MAX_NB_SUBFR * LTP_ORDER],
    ar_q13: [i16; MAX_NB_SUBFR * MAX_SHAPE_LPC_ORDER],
    harm_shape_gain_q14: [i32; MAX_NB_SUBFR],
    tilt_q14: [i32; MAX_NB_SUBFR],
    lf_shp_q14: [i32; MAX_NB_SUBFR],
    gains_q16: [i32; MAX_NB_SUBFR],
    pitch_l: [i32; MAX_NB_SUBFR],
    lambda_q10: i32,
    ltp_scale_q14: i32,
}

/// Draws the inputs of one frame. `phase` carries the oscillator of the test
/// signal from frame to frame.
fn random_frame(rng: &mut Rng, cfg: &Config, kind: Kind, phase: &mut f64) -> Frame {
    let fs = cfg.fs_khz as f64 * 1000.0;
    let frame_length = (5 * cfg.fs_khz * cfg.nb_subfr) as usize;
    let mut f = Frame {
        indices: SideInfoIndices::default(),
        x16: [0; MAX_FRAME_LENGTH],
        pred_coef_q12: [0; 2 * MAX_LPC_ORDER],
        ltp_coef_q14: [0; MAX_NB_SUBFR * LTP_ORDER],
        ar_q13: [0; MAX_NB_SUBFR * MAX_SHAPE_LPC_ORDER],
        harm_shape_gain_q14: [0; MAX_NB_SUBFR],
        tilt_q14: [0; MAX_NB_SUBFR],
        lf_shp_q14: [0; MAX_NB_SUBFR],
        gains_q16: [0; MAX_NB_SUBFR],
        pitch_l: [0; MAX_NB_SUBFR],
        lambda_q10: rng.range(900, 2800) as i32,
        ltp_scale_q14: [15565, 12288, 8192][rng.range(0, 2) as usize],
    };

    let signal_type = [0, 1, 2, 2][rng.range(0, 3) as usize];
    let voiced = signal_type == TYPE_VOICED;
    f.indices.signal_type = signal_type as i8;
    f.indices.quant_offset_type = rng.range(0, 1) as i8;
    f.indices.nlsf_interp_coef_q2 = if rng.chance(60) {
        4
    } else {
        rng.range(0, 3) as i8
    };
    f.indices.seed = rng.range(0, 3) as i8;

    // Input: silence, a full-scale square wave, or two sines with noise.
    let shape = rng.range(0, 19);
    let amplitude = 30.0 * (400.0f64).powf(rng.unit());
    let f0 = 80.0 + 300.0 * rng.unit();
    for x in f.x16[..frame_length].iter_mut() {
        *phase += f0 / fs;
        let noise = rng.unit() * 2.0 - 1.0;
        let v = match shape {
            0 => 0.0,
            1 => {
                if (*phase).fract() < 0.5 {
                    32767.0
                } else {
                    -32768.0
                }
            }
            _ => {
                let t = *phase * core::f64::consts::TAU;
                amplitude * (0.6 * t.sin() + 0.3 * (3.1 * t).sin() + 0.25 * noise)
            }
        };
        *x = v.clamp(-32768.0, 32767.0) as i16;
    }

    if kind == Kind::Extreme {
        for v in f.pred_coef_q12.iter_mut() {
            *v = rng.range(i16::MIN as i64, i16::MAX as i64) as i16;
        }
        for v in f.ar_q13.iter_mut() {
            *v = rng.range(i16::MIN as i64, i16::MAX as i64) as i16;
        }
        for v in f.ltp_coef_q14.iter_mut() {
            *v = rng.range(i16::MIN as i64, i16::MAX as i64) as i16;
        }
        for k in 0..MAX_NB_SUBFR {
            f.harm_shape_gain_q14[k] = rng.range(i32::MIN as i64, i32::MAX as i64) as i32;
            f.tilt_q14[k] = rng.range(i32::MIN as i64, i32::MAX as i64) as i32;
            f.lf_shp_q14[k] = rng.range(i32::MIN as i64, i32::MAX as i64) as i32;
            f.gains_q16[k] = if rng.chance(50) {
                rng.range(1, i32::MAX as i64) as i32
            } else {
                rng.range(1, 1 << 16) as i32
            };
        }
        f.lambda_q10 = rng.range(-70000, 70000) as i32;
        for x in f.x16.iter_mut() {
            if rng.chance(30) {
                *x = rng.range(i16::MIN as i64, i16::MAX as i64) as i16;
            }
        }
    } else {
        let order = cfg.predict_order as usize;
        let (first, second) = f.pred_coef_q12.split_at_mut(MAX_LPC_ORDER);
        random_filter(rng, order, 0.9, 12, first);
        random_filter(rng, order, 0.9, 12, second);
        let gain_level = 8.0e4 * (5.0e3f64).powf(rng.unit());
        for k in 0..MAX_NB_SUBFR {
            random_filter(
                rng,
                cfg.shaping_order as usize,
                0.7,
                13,
                &mut f.ar_q13[k * MAX_SHAPE_LPC_ORDER..(k + 1) * MAX_SHAPE_LPC_ORDER],
            );
            // Sometimes the same gain for all subframes, so that the states
            // are not rescaled.
            let step = if rng.chance(30) {
                1.0
            } else {
                0.5 + rng.unit()
            };
            f.gains_q16[k] = (gain_level * step).clamp(82570.0, 1.6e9) as i32;
            f.tilt_q14[k] = -(rng.unit() * 0.5 * 16384.0) as i32;
            let b_q14 = (rng.unit() * 0.3 * 16384.0) as i32;
            let high = 16384 - b_q14 - rng.range(0, (b_q14 / 2) as i64) as i32;
            f.lf_shp_q14[k] = (high << 16) | ((b_q14 - 16384) as u16 as i32);
            if voiced {
                f.harm_shape_gain_q14[k] = (rng.unit() * 0.35 * 16384.0) as i32;
                let scale = 0.2 + 0.75 * rng.unit();
                let taps = [0.06, 0.22, 0.44, 0.22, 0.06];
                for (j, tap) in taps.iter().enumerate() {
                    let jitter = 1.0 + 0.4 * (rng.unit() - 0.5);
                    f.ltp_coef_q14[k * LTP_ORDER + j] = (tap * scale * jitter * 16384.0) as i16;
                }
            }
        }
    }

    if voiced {
        // 2 ms to 18 ms, with small steps between the subframes. Now and
        // then a lag near the lower end, which shortens the decision delay.
        let low = 2 * cfg.fs_khz as i64;
        let high = 18 * cfg.fs_khz as i64;
        let base = if rng.chance(25) {
            rng.range(low, low + 12)
        } else {
            rng.range(low, high)
        };
        for k in 0..MAX_NB_SUBFR {
            f.pitch_l[k] = (base + rng.range(-4, 4)).clamp(low, high) as i32;
        }
    }
    f
}

// ---------------------------------------------------------------------------
// Comparison
// ---------------------------------------------------------------------------

fn assert_same_state(new: &SilkNSQState, old: &SilkNSQState, what: &str) {
    assert_eq!(new.xq[..], old.xq[..], "{what}: xq");
    assert_eq!(
        new.s_ltp_shp_q14[..],
        old.s_ltp_shp_q14[..],
        "{what}: s_ltp_shp_q14"
    );
    assert_eq!(new.s_lpc_q14[..], old.s_lpc_q14[..], "{what}: s_lpc_q14");
    assert_eq!(new.s_ar2_q14, old.s_ar2_q14, "{what}: s_ar2_q14");
    assert_eq!(new.s_lf_ar_q14, old.s_lf_ar_q14, "{what}: s_lf_ar_q14");
    assert_eq!(
        new.s_diff_shp_q14, old.s_diff_shp_q14,
        "{what}: s_diff_shp_q14"
    );
    assert_eq!(new.lag_prev, old.lag_prev, "{what}: lag_prev");
    assert_eq!(
        new.s_ltp_buf_idx, old.s_ltp_buf_idx,
        "{what}: s_ltp_buf_idx"
    );
    assert_eq!(
        new.s_ltp_shp_buf_idx, old.s_ltp_shp_buf_idx,
        "{what}: s_ltp_shp_buf_idx"
    );
    assert_eq!(new.rand_seed, old.rand_seed, "{what}: rand_seed");
    assert_eq!(
        new.prev_gain_q16, old.prev_gain_q16,
        "{what}: prev_gain_q16"
    );
    assert_eq!(new.rewhite_flag, old.rewhite_flag, "{what}: rewhite_flag");
    assert_eq!(
        new.prev_sig_type, old.prev_sig_type,
        "{what}: prev_sig_type"
    );
}

#[derive(Clone, Copy, PartialEq)]
enum Quantizer {
    DelayedDecision,
    Plain,
}

/// Runs `frames` frames of one configuration through the current and the
/// previous implementation. Both start from the same state and each keeps
/// its own state from frame to frame. After every frame the pulses, the
/// returned seed and every field of the state must be equal.
fn replay(quantizer: Quantizer, cfg: Config, kind: Kind, frames: usize, seed: u64) {
    let common = cfg.common();
    let mut rng = Rng(seed);
    let mut new_state = Box::new(SilkNSQState::default());
    // The encoder starts with a gain of 1.
    new_state.prev_gain_q16 = 65536;
    let mut old_state = new_state.clone();
    let mut phase = 0.0;

    for frame_no in 0..frames {
        let f = random_frame(&mut rng, &cfg, kind, &mut phase);
        let what = format!(
            "{} fs {} kHz, {} subframes, {} states, shaping {}, lpc {}, warping {}, {} values, seed {seed:#x}, frame {frame_no}, signal type {}",
            if quantizer == Quantizer::Plain {
                "silk_nsq"
            } else {
                "silk_nsq_del_dec"
            },
            cfg.fs_khz,
            cfg.nb_subfr,
            cfg.n_states,
            cfg.shaping_order,
            cfg.predict_order,
            cfg.warping_q16,
            if kind == Kind::Extreme {
                "extreme"
            } else {
                "plausible"
            },
            f.indices.signal_type,
        );
        let mut new_pulses = [0i8; MAX_FRAME_LENGTH];
        let mut old_pulses = [0i8; MAX_FRAME_LENGTH];
        let (new_seed, old_seed) = match quantizer {
            Quantizer::DelayedDecision => (
                silk_nsq_del_dec(
                    &common,
                    &mut new_state,
                    &f.indices,
                    &f.x16,
                    &mut new_pulses,
                    &f.pred_coef_q12,
                    &f.ltp_coef_q14,
                    &f.ar_q13,
                    &f.harm_shape_gain_q14,
                    &f.tilt_q14,
                    &f.lf_shp_q14,
                    &f.gains_q16,
                    &f.pitch_l,
                    f.lambda_q10,
                    f.ltp_scale_q14,
                ),
                reference_del_dec::silk_nsq_del_dec(
                    &common,
                    &mut old_state,
                    &f.indices,
                    &f.x16,
                    &mut old_pulses,
                    &f.pred_coef_q12,
                    &f.ltp_coef_q14,
                    &f.ar_q13,
                    &f.harm_shape_gain_q14,
                    &f.tilt_q14,
                    &f.lf_shp_q14,
                    &f.gains_q16,
                    &f.pitch_l,
                    f.lambda_q10,
                    f.ltp_scale_q14,
                ),
            ),
            Quantizer::Plain => {
                silk_nsq(
                    &common,
                    &mut new_state,
                    &f.indices,
                    &f.x16,
                    &mut new_pulses,
                    &f.pred_coef_q12,
                    &f.ltp_coef_q14,
                    &f.ar_q13,
                    &f.harm_shape_gain_q14,
                    &f.tilt_q14,
                    &f.lf_shp_q14,
                    &f.gains_q16,
                    &f.pitch_l,
                    f.lambda_q10,
                    f.ltp_scale_q14,
                );
                reference_nsq::silk_nsq(
                    &common,
                    &mut old_state,
                    &f.indices,
                    &f.x16,
                    &mut old_pulses,
                    &f.pred_coef_q12,
                    &f.ltp_coef_q14,
                    &f.ar_q13,
                    &f.harm_shape_gain_q14,
                    &f.tilt_q14,
                    &f.lf_shp_q14,
                    &f.gains_q16,
                    &f.pitch_l,
                    f.lambda_q10,
                    f.ltp_scale_q14,
                );
                (0, 0)
            }
        };
        assert_eq!(new_seed, old_seed, "{what}: returned seed");
        assert_eq!(new_pulses[..], old_pulses[..], "{what}: pulses");
        assert_same_state(&new_state, &old_state, &what);
    }
}

/// The combinations of shaping order and warping the encoder uses, by
/// complexity, and a few it does not use.
const SHAPING: [(i32, bool); 8] = [
    (12, false),
    (14, false),
    (16, true),
    (20, true),
    (24, true),
    (24, false),
    (2, true),
    (8, true),
];

fn configs(n_states: &[i32], shaping: &[(i32, bool)]) -> Vec<Config> {
    let mut all = Vec::new();
    for &fs_khz in &[8, 12, 16] {
        for &nb_subfr in &[4, 2] {
            for &n in n_states {
                for &(shaping_order, warped) in shaping {
                    all.push(Config {
                        fs_khz,
                        nb_subfr,
                        n_states: n,
                        shaping_order,
                        predict_order: if fs_khz == 16 { 16 } else { 10 },
                        warping_q16: if warped { fs_khz * 983 } else { 0 },
                    });
                }
            }
        }
    }
    all
}

#[test]
fn delayed_decision_matches_previous_implementation() {
    for (n, cfg) in configs(&[1, 2, 3, 4], &SHAPING).into_iter().enumerate() {
        replay(
            Quantizer::DelayedDecision,
            cfg,
            Kind::Plausible,
            10,
            0x1000 + n as u64,
        );
    }
}

/// State counts outside 1 to 4: negative and too large values mean 4 states,
/// and 0 means no state at all (the outputs then come from a state that is
/// never updated). The encoder uses none of them, but the function accepted
/// them before.
#[test]
fn delayed_decision_matches_for_unusual_state_counts() {
    for (n, cfg) in configs(&[-1, 0, 5, 9], &SHAPING[2..5])
        .into_iter()
        .enumerate()
    {
        replay(
            Quantizer::DelayedDecision,
            cfg,
            Kind::Plausible,
            6,
            0x2000 + n as u64,
        );
    }
}

/// Prediction orders other than 10 and 16.
#[test]
fn delayed_decision_matches_for_other_prediction_orders() {
    let mut n = 0;
    for &predict_order in &[6, 8, 12, 14] {
        for &n_states in &[1, 2, 3, 4] {
            for &fs_khz in &[8, 16] {
                let cfg = Config {
                    fs_khz,
                    nb_subfr: 4,
                    n_states,
                    shaping_order: 16,
                    predict_order,
                    warping_q16: fs_khz * 983,
                };
                replay(
                    Quantizer::DelayedDecision,
                    cfg,
                    Kind::Plausible,
                    6,
                    0x3000 + n,
                );
                n += 1;
            }
        }
    }
}

/// Long runs of one configuration each: more frames give more state
/// replacements, ties and gain changes per state count.
#[test]
fn delayed_decision_matches_over_long_runs() {
    for (n, &(n_states, shaping_order)) in [(2, 16), (3, 20), (4, 24)].iter().enumerate() {
        for &fs_khz in &[8, 16] {
            let cfg = Config {
                fs_khz,
                nb_subfr: 4,
                n_states,
                shaping_order,
                predict_order: if fs_khz == 16 { 16 } else { 10 },
                warping_q16: fs_khz * 983,
            };
            replay(
                Quantizer::DelayedDecision,
                cfg,
                Kind::Plausible,
                150,
                0x4000 + n as u64,
            );
        }
    }
}

#[test]
fn delayed_decision_matches_on_extreme_values() {
    if cfg!(debug_assertions) {
        // See `Kind::Extreme`.
        return;
    }
    for (n, cfg) in configs(&[1, 2, 3, 4], &SHAPING[..6])
        .into_iter()
        .enumerate()
    {
        replay(
            Quantizer::DelayedDecision,
            cfg,
            Kind::Extreme,
            6,
            0x5000 + n as u64,
        );
    }
}

#[test]
fn plain_quantizer_matches_previous_implementation() {
    let shaping: Vec<(i32, bool)> = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]
        .iter()
        .map(|&order| (order, false))
        .collect();
    for (n, cfg) in configs(&[1], &shaping).into_iter().enumerate() {
        replay(
            Quantizer::Plain,
            cfg,
            Kind::Plausible,
            12,
            0x6000 + n as u64,
        );
    }
}

#[test]
fn plain_quantizer_matches_for_other_prediction_orders() {
    let mut n = 0;
    for &predict_order in &[6, 8, 12, 14] {
        for &shaping_order in &[12, 14] {
            for &fs_khz in &[8, 16] {
                let cfg = Config {
                    fs_khz,
                    nb_subfr: 4,
                    n_states: 1,
                    shaping_order,
                    predict_order,
                    warping_q16: 0,
                };
                replay(Quantizer::Plain, cfg, Kind::Plausible, 8, 0x7000 + n);
                n += 1;
            }
        }
    }
}

#[test]
fn plain_quantizer_matches_over_long_runs() {
    for (n, &shaping_order) in [12, 14].iter().enumerate() {
        for &fs_khz in &[8, 12, 16] {
            let cfg = Config {
                fs_khz,
                nb_subfr: 4,
                n_states: 1,
                shaping_order,
                predict_order: if fs_khz == 16 { 16 } else { 10 },
                warping_q16: 0,
            };
            replay(
                Quantizer::Plain,
                cfg,
                Kind::Plausible,
                150,
                0x8000 + n as u64,
            );
        }
    }
}

#[test]
fn plain_quantizer_matches_on_extreme_values() {
    if cfg!(debug_assertions) {
        // See `Kind::Extreme`.
        return;
    }
    let shaping: Vec<(i32, bool)> = [12, 14, 16, 24].iter().map(|&o| (o, false)).collect();
    for (n, cfg) in configs(&[1], &shaping).into_iter().enumerate() {
        replay(Quantizer::Plain, cfg, Kind::Extreme, 6, 0x9000 + n as u64);
    }
}

/// `silk_noise_shape_quantizer_del_dec` works on states in the libopus
/// layout. It must leave them as the previous implementation did, except for
/// entries that no later call reads: the leftovers in `s_lpc_q14` and the
/// part of `s_ar2_q14` above the shaping order.
#[test]
fn subframe_function_matches_previous_implementation() {
    use opus_rs::silk::nsq_del_dec::{NSQDelDecStruct, silk_noise_shape_quantizer_del_dec};

    let mut rng = Rng(0xA000);
    for case in 0..200 {
        let fs_khz = [8, 12, 16][rng.range(0, 2) as usize];
        let n_states = rng.range(0, 4) as i32;
        let cfg = Config {
            fs_khz,
            nb_subfr: 4,
            n_states,
            shaping_order: [12, 16, 20, 24][rng.range(0, 3) as usize],
            predict_order: if fs_khz == 16 { 16 } else { 10 },
            warping_q16: if rng.chance(70) { fs_khz * 983 } else { 0 },
        };
        let common = cfg.common();
        let mut phase = 0.0;
        let f = random_frame(&mut rng, &cfg, Kind::Plausible, &mut phase);
        let length = common.subfr_length as usize;
        let ltp_mem = common.ltp_mem_length;

        // A quantizer state and delayed-decision states with history.
        let mut new_nsq = Box::new(SilkNSQState::default());
        for v in new_nsq.s_ltp_shp_q14.iter_mut() {
            *v = rng.range(-(1 << 22), 1 << 22) as i32;
        }
        new_nsq.s_ltp_shp_buf_idx = ltp_mem;
        new_nsq.s_ltp_buf_idx = ltp_mem;
        let mut old_nsq = new_nsq.clone();
        let mut new_dd = [NSQDelDecStruct::default(); 4];
        let mut old_dd = [reference_del_dec::NSQDelDecStruct::default(); 4];
        for k in 0..4 {
            macro_rules! fill {
                ($field:ident, $bits:expr) => {
                    for j in 0..new_dd[k].$field.len() {
                        let v = rng.range(-(1 << $bits), 1 << $bits) as i32;
                        new_dd[k].$field[j] = v;
                        old_dd[k].$field[j] = v;
                    }
                };
            }
            fill!(s_lpc_q14, 22);
            fill!(rand_state, 31);
            fill!(q_q10, 14);
            fill!(xq_q14, 24);
            fill!(pred_q15, 24);
            fill!(shape_q14, 22);
            fill!(s_ar2_q14, 22);
            macro_rules! set {
                ($field:ident, $bits:expr) => {
                    let v = rng.range(-(1 << $bits), 1 << $bits) as i32;
                    new_dd[k].$field = v;
                    old_dd[k].$field = v;
                };
            }
            set!(lf_ar_q14, 22);
            set!(diff_q14, 22);
            set!(seed, 31);
            set!(seed_init, 2);
            set!(rd_q10, 20);
        }
        let mut x_q10 = [0i32; MAX_SUB_FRAME_LENGTH];
        for v in x_q10.iter_mut() {
            *v = rng.range(-(1 << 20), 1 << 20) as i32;
        }
        let mut s_ltp_q15 = [0i32; 2 * MAX_FRAME_LENGTH];
        for v in s_ltp_q15.iter_mut() {
            *v = rng.range(-(1 << 22), 1 << 22) as i32;
        }
        let mut old_s_ltp_q15 = s_ltp_q15;
        let mut delayed_ga_q10 = [0i32; DECISION_DELAY];
        for v in delayed_ga_q10.iter_mut() {
            *v = rng.range(1 << 6, 1 << 20) as i32;
        }
        let mut old_delayed_ga_q10 = delayed_ga_q10;
        let smpl_buf_idx = rng.range(0, DECISION_DELAY as i64 - 1) as i32;
        let decision_delay = rng.range(1, (DECISION_DELAY as i64).min(length as i64)) as i32;
        let lag = if f.indices.signal_type as i32 == TYPE_VOICED {
            f.pitch_l[0].max(decision_delay + 3)
        } else {
            0
        };
        let subfr = rng.range(0, 3) as i32;
        let pulses_offset = subfr * length as i32;
        let xq_ptr = ltp_mem + pulses_offset;
        let harm = (f.harm_shape_gain_q14[0] >> 2) | ((f.harm_shape_gain_q14[0] >> 1) << 16);

        let mut new_pulses = [0i8; MAX_FRAME_LENGTH];
        let mut old_pulses = [0i8; MAX_FRAME_LENGTH];
        let mut new_idx = smpl_buf_idx;
        let mut old_idx = smpl_buf_idx;
        silk_noise_shape_quantizer_del_dec(
            &mut new_nsq,
            &mut new_dd,
            f.indices.signal_type as i32,
            &x_q10,
            &mut new_pulses,
            pulses_offset,
            xq_ptr,
            &mut s_ltp_q15,
            &mut delayed_ga_q10,
            &f.pred_coef_q12,
            &f.ltp_coef_q14,
            &f.ar_q13,
            lag,
            harm,
            f.tilt_q14[0],
            f.lf_shp_q14[0],
            f.gains_q16[0],
            f.lambda_q10,
            100,
            length as i32,
            subfr,
            cfg.shaping_order,
            cfg.predict_order,
            cfg.warping_q16,
            n_states,
            &mut new_idx,
            decision_delay,
            0,
        );
        reference_del_dec::silk_noise_shape_quantizer_del_dec(
            &mut old_nsq,
            &mut old_dd,
            f.indices.signal_type as i32,
            &x_q10,
            &mut old_pulses,
            pulses_offset,
            xq_ptr,
            &mut old_s_ltp_q15,
            &mut old_delayed_ga_q10,
            &f.pred_coef_q12,
            &f.ltp_coef_q14,
            &f.ar_q13,
            lag,
            harm,
            f.tilt_q14[0],
            f.lf_shp_q14[0],
            f.gains_q16[0],
            f.lambda_q10,
            100,
            length as i32,
            subfr,
            cfg.shaping_order,
            cfg.predict_order,
            cfg.warping_q16,
            n_states,
            &mut old_idx,
            decision_delay,
            0,
        );

        let what = format!("case {case}, {n_states} states, fs {fs_khz} kHz");
        assert_eq!(new_idx, old_idx, "{what}: smpl_buf_idx");
        assert_eq!(new_pulses[..], old_pulses[..], "{what}: pulses");
        assert_eq!(s_ltp_q15[..], old_s_ltp_q15[..], "{what}: s_ltp_q15");
        assert_eq!(delayed_ga_q10, old_delayed_ga_q10, "{what}: delayed gains");
        assert_same_state(&new_nsq, &old_nsq, &what);
        for k in 0..4 {
            let (n, o) = (&new_dd[k], &old_dd[k]);
            let what = format!("{what}, state {k}");
            // The prediction history of the next subframe and the samples of
            // this one. Above that the array holds leftovers that differ.
            let live = length + NSQ_LPC_BUF_LENGTH;
            assert_eq!(
                n.s_lpc_q14[..live],
                o.s_lpc_q14[..live],
                "{what}: s_lpc_q14"
            );
            assert_eq!(n.rand_state, o.rand_state, "{what}: rand_state");
            assert_eq!(n.q_q10, o.q_q10, "{what}: q_q10");
            assert_eq!(n.xq_q14, o.xq_q14, "{what}: xq_q14");
            assert_eq!(n.pred_q15, o.pred_q15, "{what}: pred_q15");
            assert_eq!(n.shape_q14, o.shape_q14, "{what}: shape_q14");
            // The filter state up to the shaping order. The entries above
            // it are not used by the filter. The previous implementation
            // copied them along when a state was replaced; now every state
            // keeps its own. In the encoder they are equal in all states.
            let order = cfg.shaping_order as usize;
            assert_eq!(
                n.s_ar2_q14[..order],
                o.s_ar2_q14[..order],
                "{what}: s_ar2_q14"
            );
            assert_eq!(n.lf_ar_q14, o.lf_ar_q14, "{what}: lf_ar_q14");
            assert_eq!(n.diff_q14, o.diff_q14, "{what}: diff_q14");
            assert_eq!(n.seed, o.seed, "{what}: seed");
            assert_eq!(n.seed_init, o.seed_init, "{what}: seed_init");
            assert_eq!(n.rd_q10, o.rd_q10, "{what}: rd_q10");
        }
    }
}
