use crate::silk::define::MAX_LPC_ORDER;
use crate::silk::lin2log::silk_lin2log;
use crate::silk::macros::{silk_rshift, silk_smlabb, silk_smulbb};
use crate::silk::nlsf::{silk_nlsf_stabilize, silk_nlsf_vq};
use crate::silk::nlsf_decode::silk_nlsf_decode;
use crate::silk::nlsf_del_dec_quant::{
    NLSF_EC_TABLE_LEN, NLSF_QUANT_LEVELS, NlsfQuantInput, silk_nlsf_del_dec_quant_group,
    silk_nlsf_quant_levels, silk_nlsf_rate_cost,
};
use crate::silk::nlsf_unpack::silk_nlsf_unpack;
use crate::silk::pitch_analysis::silk_div32_varq_batch;
use crate::silk::sort::silk_insertion_sort_increasing;
use crate::silk::structs::NLSFCodebook;

/// Entropy tables a codebook can have: `silk_nlsf_unpack` reads a 3-bit
/// table number.
const NLSF_EC_TABLES: usize = 8;

/// Largest number of survivors whose residual searches run side by side (see
/// `silk_nlsf_del_dec_quant_group`). Measured on an Apple M4: 4 is twice as
/// good as 2, and 8 is no better than 4.
const SEARCH_GROUP: usize = 4;

/// Largest number of survivors of the first stage.
const MAX_SURVIVORS: usize = 16;

pub fn silk_nlsf_encode(
    nlsf_indices: &mut [i8],
    p_nlsf_q15: &mut [i16],
    ps_nlsf_cb: &NLSFCodebook,
    p_w_q2: &[i16],
    nlsf_mu_q20: i32,
    n_survivors: usize,
    signal_type: i32,
) -> i32 {
    let order = ps_nlsf_cb.order as usize;

    assert!(n_survivors <= MAX_SURVIVORS);

    let mut err_q24 = [0i32; 32];
    let mut temp_indices1 = [0i32; MAX_SURVIVORS];
    let mut rd_q25 = [0i32; MAX_SURVIVORS];
    let mut temp_indices2 = [[0i8; MAX_LPC_ORDER]; MAX_SURVIVORS];

    // The inputs of the residual search, one set per survivor of a group.
    let mut res_q10 = [[0i16; MAX_LPC_ORDER]; SEARCH_GROUP];
    let mut w_adj_q5 = [[0i16; MAX_LPC_ORDER]; SEARCH_GROUP];
    let mut pred_q8 = [[0u8; MAX_LPC_ORDER]; SEARCH_GROUP];
    let mut rate_row = [[0u8; MAX_LPC_ORDER]; SEARCH_GROUP];
    let mut ec_ix = [0i16; MAX_LPC_ORDER];

    silk_nlsf_stabilize(
        p_nlsf_q15,
        ps_nlsf_cb.delta_min_q15,
        ps_nlsf_cb.order as usize,
    );

    silk_nlsf_vq(
        &mut err_q24,
        p_nlsf_q15,
        ps_nlsf_cb.cb1_nlsf_q8,
        ps_nlsf_cb.cb1_wght_q9,
        ps_nlsf_cb.n_vectors as usize,
        ps_nlsf_cb.order as usize,
    );

    silk_insertion_sort_increasing(
        &mut err_q24,
        &mut temp_indices1,
        ps_nlsf_cb.n_vectors as usize,
        n_survivors,
    );

    // Tables for the residual quantiser. They are the same for every
    // survivor: the output levels depend on the codebook, and the rate cost
    // of an index depends on the entropy table (at most 8 per codebook) and
    // on `nlsf_mu_q20`.
    let levels = silk_nlsf_quant_levels(ps_nlsf_cb.quant_step_size_q16);
    let mut rate_cost = [[0i32; NLSF_QUANT_LEVELS]; NLSF_EC_TABLES];
    let (tables, _) = ps_nlsf_cb.ec_rates_q5.as_chunks::<NLSF_EC_TABLE_LEN>();
    for (cost, table_rates_q5) in rate_cost.iter_mut().zip(tables) {
        *cost = silk_nlsf_rate_cost(nlsf_mu_q20, table_rates_q5);
    }

    // The dividends of the adjusted weights are the same for every survivor.
    let mut w_q2 = [0i32; MAX_LPC_ORDER];
    for (wide, &w) in w_q2.iter_mut().zip(&p_w_q2[..order]) {
        *wide = w as i32;
    }
    let mut w_tmp_squared = [1i32; MAX_LPC_ORDER];
    let mut w_adj = [0i32; MAX_LPC_ORDER];

    // The survivors are searched in groups: the searches of one group run
    // side by side. Each survivor's result does not depend on the others.
    let mut s = 0;
    while s < n_survivors {
        let group = (n_survivors - s).min(SEARCH_GROUP);

        for g in 0..group {
            let ind1 = temp_indices1[s + g] as usize;

            let p_cb_element = &ps_nlsf_cb.cb1_nlsf_q8[ind1 * order..];
            let p_cb_wght_q9 = &ps_nlsf_cb.cb1_wght_q9[ind1 * order..];
            for i in 0..order {
                let nlsf_tmp_q15 = (p_cb_element[i] as i16) << 7;
                let w_tmp_q9 = p_cb_wght_q9[i] as i32;
                res_q10[g][i] =
                    (silk_smulbb(p_nlsf_q15[i] as i32 - nlsf_tmp_q15 as i32, w_tmp_q9) >> 14) as i16;
                w_tmp_squared[i] = w_tmp_q9 * w_tmp_q9;
            }
            // Weight of each coefficient divided by the squared codebook
            // weight, all coefficients in one batch.
            silk_div32_varq_batch(
                &mut w_adj[..order],
                &w_q2[..order],
                &w_tmp_squared[..order],
                21,
            );
            for (narrow, &wide) in w_adj_q5[g].iter_mut().zip(&w_adj[..order]) {
                *narrow = wide as i16;
            }

            silk_nlsf_unpack(&mut ec_ix, &mut pred_q8[g], ps_nlsf_cb, ind1);

            // `ec_ix[i]` is the offset of the entropy table of coefficient
            // `i` in `ec_rates_q5`: a table number from 0 to 7 times the
            // table length.
            for (row, &offset) in rate_row[g].iter_mut().zip(&ec_ix[..order]) {
                debug_assert!(offset >= 0 && (offset as usize).is_multiple_of(NLSF_EC_TABLE_LEN));
                *row = (offset as usize / NLSF_EC_TABLE_LEN) as u8;
            }
        }

        let input = |g: usize| NlsfQuantInput {
            x_q10: &res_q10[g],
            w_q5: &w_adj_q5[g],
            pred_coef_q8: &pred_q8[g],
            rate_cost: &rate_cost,
            rate_row: &rate_row[g],
            levels: &levels,
            inv_quant_step_size_q6: ps_nlsf_cb.inv_quant_step_size_q6,
            order,
        };
        // Survivors `s + from..s + from + N` of the group, side by side.
        macro_rules! search {
            ($n:literal, $from:expr) => {{
                let from: usize = $from;
                let rows = temp_indices2[s + from..].first_chunk_mut::<$n>().unwrap();
                let rd: [i32; $n] = silk_nlsf_del_dec_quant_group(
                    rows,
                    &core::array::from_fn(|g| input(from + g)),
                );
                rd_q25[s + from..s + from + $n].copy_from_slice(&rd);
            }};
        }
        match group {
            SEARCH_GROUP => search!(4, 0),
            3 => {
                search!(2, 0);
                search!(1, 2);
            }
            2 => search!(2, 0),
            _ => search!(1, 0),
        }

        for g in 0..group {
            let ind1 = temp_indices1[s + g] as usize;
            let i_cdf_ptr = &ps_nlsf_cb.cb1_icdf
                [((signal_type >> 1) as usize) * ps_nlsf_cb.n_vectors as usize..];
            let prob_q8 = if ind1 == 0 {
                256 - i_cdf_ptr[ind1] as i32
            } else {
                i_cdf_ptr[ind1 - 1] as i32 - i_cdf_ptr[ind1] as i32
            };
            let bits_q7 = (8 << 7) - silk_lin2log(prob_q8);
            rd_q25[s + g] = silk_smlabb(rd_q25[s + g], bits_q7, silk_rshift(nlsf_mu_q20, 2));
        }

        s += group;
    }

    let mut best_index = [0i32; 1];
    silk_insertion_sort_increasing(&mut rd_q25, &mut best_index, n_survivors, 1);
    let best_idx = best_index[0] as usize;

    nlsf_indices[0] = temp_indices1[best_idx] as i8;
    nlsf_indices[1..1 + order].copy_from_slice(&temp_indices2[best_idx][..order]);

    silk_nlsf_decode(p_nlsf_q15, nlsf_indices, ps_nlsf_cb);

    rd_q25[0]
}
