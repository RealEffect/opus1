use crate::silk::define::*;
use crate::silk::lin2log::silk_lin2log;
use crate::silk::log2lin::silk_log2lin;
use crate::silk::macros::*;
use crate::silk::tables::*;
use crate::silk::tuning_parameters::MAX_SUM_LOG_GAIN_dB;
use crate::silk::vq_wmat_ec::silk_vq_wmat_ec_codebook;

pub const OFFSET: i32 = (MIN_QGAIN_DB * 128) / 6 + 16 * 128;
pub const SCALE_Q16: i32 =
    (65536 * (N_LEVELS_QGAIN - 1)) / (((MAX_QGAIN_DB - MIN_QGAIN_DB) * 128) / 6);

pub const INV_SCALE_Q16: i32 = 1907825;

#[inline(always)]
pub fn silk_gains_quant(
    ind: &mut [i8; MAX_NB_SUBFR],
    gain_q16: &mut [i32; MAX_NB_SUBFR],
    prev_ind: &mut i8,
    conditional: i32,
    nb_subfr: usize,
) {
    let mut double_step_size_threshold: i32;

    for k in 0..nb_subfr {
        let lin2log_val = silk_lin2log(gain_q16[k]);
        ind[k] = silk_smulwb(SCALE_Q16, lin2log_val - OFFSET) as i8;

        if ind[k] < *prev_ind {
            ind[k] += 1;
        }
        ind[k] = silk_limit_int(ind[k] as i32, 0, N_LEVELS_QGAIN - 1) as i8;

        if k == 0 && conditional == 0 {
            ind[k] = silk_limit_int(
                ind[k] as i32,
                (*prev_ind as i32) + MIN_DELTA_GAIN_QUANT,
                N_LEVELS_QGAIN - 1,
            ) as i8;
            *prev_ind = ind[k];
        } else {
            ind[k] -= *prev_ind;

            double_step_size_threshold =
                2 * MAX_DELTA_GAIN_QUANT - N_LEVELS_QGAIN + (*prev_ind as i32);
            if (ind[k] as i32) > double_step_size_threshold {
                ind[k] = (double_step_size_threshold
                    + silk_rshift((ind[k] as i32) - double_step_size_threshold + 1, 1))
                    as i8;
            }

            ind[k] =
                silk_limit_int(ind[k] as i32, MIN_DELTA_GAIN_QUANT, MAX_DELTA_GAIN_QUANT) as i8;

            if (ind[k] as i32) > double_step_size_threshold {
                *prev_ind = ((*prev_ind as i32) + silk_lshift(ind[k] as i32, 1)
                    - double_step_size_threshold) as i8;
                *prev_ind = silk_min_int(*prev_ind as i32, N_LEVELS_QGAIN - 1) as i8;
            } else {
                *prev_ind = ((*prev_ind as i32) + (ind[k] as i32)) as i8;
            }

            ind[k] -= MIN_DELTA_GAIN_QUANT as i8;
        }

        gain_q16[k] = silk_log2lin(silk_min_32(
            silk_smulwb(INV_SCALE_Q16, *prev_ind as i32) + OFFSET,
            3967,
        ));
    }
}

pub fn silk_gains_dequant(
    gain_q16: &mut [i32; MAX_NB_SUBFR],
    ind: &[i8; MAX_NB_SUBFR],
    prev_ind: &mut i8,
    conditional: i32,
    nb_subfr: usize,
) {
    let mut double_step_size_threshold: i32;
    let mut ind_tmp: i32;

    for k in 0..nb_subfr {
        if k == 0 && conditional == 0 {
            *prev_ind = core::cmp::max(ind[k] as i32, (*prev_ind as i32) - 16) as i8;
        } else {
            ind_tmp = (ind[k] as i32) + MIN_DELTA_GAIN_QUANT;

            double_step_size_threshold =
                2 * MAX_DELTA_GAIN_QUANT - N_LEVELS_QGAIN + (*prev_ind as i32);
            if ind_tmp > double_step_size_threshold {
                *prev_ind = ((*prev_ind as i32) + silk_lshift(ind_tmp, 1)
                    - double_step_size_threshold) as i8;
            } else {
                *prev_ind = ((*prev_ind as i32) + ind_tmp) as i8;
            }
        }
        *prev_ind = silk_limit_int(*prev_ind as i32, 0, N_LEVELS_QGAIN - 1) as i8;

        gain_q16[k] = silk_log2lin(silk_min_32(
            silk_smulwb(INV_SCALE_Q16, *prev_ind as i32) + OFFSET,
            3967,
        ));
    }
}

pub fn silk_quant_ltp_gains(
    b_q14: &mut [i16],
    cbk_index: &mut [i8],
    periodicity_index: &mut i8,
    sum_log_gain_q7: &mut i32,
    pred_gain_db_q7: &mut i32,
    xx_q17: &[i32],
    xx_in_q17: &[i32],
    subfr_len: i32,
    nb_subfr: usize,
    _arch: i32,
) {
    let mut temp_idx = [0i8; MAX_NB_SUBFR];
    let mut res_nrg_total_q15: i32 = 0;
    let mut rate_dist_total_q7: i32;
    let mut min_rate_dist_q7: i32;
    let mut sum_log_gain_tmp_q7: i32;
    let mut best_sum_log_gain_q7: i32 = 0;
    let mut max_gain_q7: i32;
    let mut gain_q7: i32 = 0;

    min_rate_dist_q7 = i32::MAX;
    for k in 0..3 {
        let gain_safety = 51;

        res_nrg_total_q15 = 0;
        rate_dist_total_q7 = 0;
        sum_log_gain_tmp_q7 = *sum_log_gain_q7;
        for j in 0..nb_subfr {
            max_gain_q7 = silk_log2lin(
                (silk_float_to_fixed_q7(MAX_SUM_LOG_GAIN_dB / 6.0) - sum_log_gain_tmp_q7) + 896,
            ) - gain_safety;

            // Correlation matrix and vector of subframe `j`.
            let xx_subfr_q17 = xx_q17[j * LTP_ORDER * LTP_ORDER..]
                .first_chunk()
                .expect("25 values per subframe");
            let xx_in_subfr_q17 = xx_in_q17[j * LTP_ORDER..]
                .first_chunk()
                .expect("5 values per subframe");
            let choice =
                silk_vq_wmat_ec_codebook(k, xx_subfr_q17, xx_in_subfr_q17, subfr_len, max_gain_q7);
            temp_idx[j] = choice.ind;
            // When no codebook vector qualified, the gain of the search
            // before stays in use (as in libopus).
            if let Some(found_gain_q7) = choice.gain_q7 {
                gain_q7 = found_gain_q7;
            }

            res_nrg_total_q15 = add_pos_sat32(res_nrg_total_q15, choice.res_nrg_q15);
            rate_dist_total_q7 = add_pos_sat32(rate_dist_total_q7, choice.rate_dist_q7);
            sum_log_gain_tmp_q7 =
                0.max(sum_log_gain_tmp_q7 + silk_lin2log(gain_q7 + gain_safety) - 896);
        }

        if rate_dist_total_q7 <= min_rate_dist_q7 {
            min_rate_dist_q7 = rate_dist_total_q7;
            best_sum_log_gain_q7 = sum_log_gain_tmp_q7;
            *periodicity_index = k as i8;
            cbk_index[..nb_subfr].copy_from_slice(&temp_idx[..nb_subfr]);
        }
    }

    *sum_log_gain_q7 = best_sum_log_gain_q7;

    for j in 0..nb_subfr {
        let cbk_ptr = &SILK_LTP_VQ_PTRS_Q7[*periodicity_index as usize][cbk_index[j] as usize];
        for i in 0..LTP_ORDER {
            b_q14[j * LTP_ORDER + i] = silk_lshift(cbk_ptr[i] as i32, 7) as i16;
        }
    }

    // libopus (silk/quant_LTP_gains.c) computes the prediction gain from
    // the residual energy of the codebook it tried last, which is codebook
    // 2, and not from that of the codebook it chose. The decoder never sees
    // this value, but the encoder's later decisions depend on it, so the
    // same value is used here.
    let res_nrg_shift = if nb_subfr == 2 {
        silk_rshift32(res_nrg_total_q15, 1)
    } else {
        silk_rshift32(res_nrg_total_q15, 2)
    };
    *pred_gain_db_q7 = silk_smulbb(-3, silk_lin2log(res_nrg_shift) - 1920);
}

/// libopus `silk_ADD_POS_SAT32`: the sum of two numbers, or `i32::MAX` when
/// the 32-bit sum has its top bit set.
///
/// The macro is meant for two positive numbers, where this is an addition
/// that saturates. `silk_quant_ltp_gains` also calls it with the bit count of
/// a subframe, which is negative when the predictor works well. A negative
/// sum then turns into `i32::MAX`. libopus chooses its codebook from these
/// sums, so the same is done here.
#[inline(always)]
fn add_pos_sat32(a: i32, b: i32) -> i32 {
    let sum = (a as u32).wrapping_add(b as u32);
    if sum & 0x8000_0000 != 0 {
        i32::MAX
    } else {
        sum as i32
    }
}

fn silk_float_to_fixed_q7(f: f32) -> i32 {
    (f * 128.0 + 0.5) as i32
}

#[inline(always)]
pub fn silk_gains_id(ind: &[i8; MAX_NB_SUBFR], nb_subfr: i32) -> i32 {
    let mut gains_id: i32 = 0;
    for k in 0..nb_subfr as usize {
        gains_id = (ind[k] as i32) + (gains_id << 8);
    }
    gains_id
}
