//! Delayed-decision quantiser for the NLSF residual (libopus
//! `silk_NLSF_del_dec_quant`).
//!
//! NLSF = normalised line spectral frequencies, the form in which SILK codes
//! its prediction filter. The residual of each coefficient is quantised to
//! one of two neighbouring levels. The search follows up to four paths of
//! such choices ("states") and keeps the ones with the lowest rate-distortion
//! value.
//!
//! This file has two versions of the search that give the same result:
//!
//! - `silk_nlsf_del_dec_quant_reference` follows the libopus code line by
//!   line.
//! - `silk_nlsf_del_dec_quant_tables` is the fast version. It keeps each
//!   candidate in one 64-bit word and prunes without branches. It hands over
//!   to the reference version in the one case it does not cover (a negative
//!   rate-distortion value).

use core::hint::select_unpredictable as select;

use crate::silk::define::{
    MAX_LPC_ORDER, NLSF_QUANT_DEL_DEC_STATES, NLSF_QUANT_DEL_DEC_STATES_LOG2, NLSF_QUANT_LEVEL_ADJ,
    NLSF_QUANT_MAX_AMPLITUDE, NLSF_QUANT_MAX_AMPLITUDE_EXT,
};
use crate::silk::macros::{silk_mla, silk_smulbb};

/// Number of states the search keeps (4).
const N_STATES: usize = NLSF_QUANT_DEL_DEC_STATES;

/// Number of values a residual index can take, plus one: indices run from
/// -10 to 9, and the search also needs the level above the chosen index.
pub(crate) const NLSF_QUANT_LEVELS: usize = 2 * NLSF_QUANT_MAX_AMPLITUDE_EXT as usize + 1;

/// Entries of one entropy table in `NLSFCodebook::ec_rates_q5`: one rate for
/// each index from -4 to 4.
pub(crate) const NLSF_EC_TABLE_LEN: usize = 2 * NLSF_QUANT_MAX_AMPLITUDE as usize + 1;

/// Output level (Q10) of every residual index from -10 to 10.
///
/// Entry `k + 10` is the level of index `k`. The levels depend only on the
/// quantisation step of the codebook.
pub(crate) fn silk_nlsf_quant_levels(quant_step_size_q16: i32) -> [i32; NLSF_QUANT_LEVELS] {
    let mut levels = [0i32; NLSF_QUANT_LEVELS];
    for (k, level) in (-NLSF_QUANT_MAX_AMPLITUDE_EXT..).zip(levels.iter_mut()) {
        let mut out_q10 = k << 10;
        if k > 0 {
            out_q10 -= NLSF_QUANT_LEVEL_ADJ;
        } else if k < 0 {
            out_q10 += NLSF_QUANT_LEVEL_ADJ;
        }
        *level = silk_smulbb(out_q10, quant_step_size_q16) >> 16;
    }
    levels
}

/// Rate (Q5) of the residual indices that are coded with an escape code:
/// 108 + 43 * |k| for |k| >= 4. Entry `k + 10` belongs to index `k`. The
/// entries for -3..=3 are zero; their rates come from the entropy table.
const ESCAPE_RATES_Q5: [i32; NLSF_QUANT_LEVELS] = {
    let mut rates = [0i32; NLSF_QUANT_LEVELS];
    let mut slot = 0;
    while slot < NLSF_QUANT_LEVELS {
        let magnitude = (slot as i32 - NLSF_QUANT_MAX_AMPLITUDE_EXT).abs();
        if magnitude >= NLSF_QUANT_MAX_AMPLITUDE {
            rates[slot] = 280 - 43 * NLSF_QUANT_MAX_AMPLITUDE + 43 * magnitude;
        }
        slot += 1;
    }
    rates
};

/// Rate cost of every residual index from -10 to 10 for one entropy table.
///
/// Entry `k + 10` is `mu_q20` times the rate (Q5) of index `k`.
/// `table_rates_q5` is one table of `NLSFCodebook::ec_rates_q5`: entry
/// `k + 4` is the rate of index `k`. Only the entries for -3..=3 are used;
/// the other indices cost `ESCAPE_RATES_Q5`.
pub(crate) fn silk_nlsf_rate_cost(
    mu_q20: i32,
    table_rates_q5: &[u8; NLSF_EC_TABLE_LEN],
) -> [i32; NLSF_QUANT_LEVELS] {
    const FIRST: usize = (NLSF_QUANT_MAX_AMPLITUDE_EXT - NLSF_QUANT_MAX_AMPLITUDE + 1) as usize;
    let mut rates_q5 = ESCAPE_RATES_Q5;
    for (rate, &from_table) in rates_q5[FIRST..]
        .iter_mut()
        .zip(&table_rates_q5[1..NLSF_EC_TABLE_LEN - 1])
    {
        *rate = from_table as i32;
    }
    rates_q5.map(|rate| silk_smulbb(mu_q20, rate))
}

/// Values that are the same for all states of one coefficient.
struct Coefficient<'a> {
    in_q10: i32,
    pred_coef_q8: i32,
    w_q5: i32,
    inv_quant_step_size_q6: i32,
    /// Output level of each residual index, at position index + 10.
    levels: &'a [i32; NLSF_QUANT_LEVELS],
    /// Rate cost of each residual index, at position index + 10.
    cost: &'a [i32; NLSF_QUANT_LEVELS],
}

/// The two quantisation candidates of one state for one coefficient.
struct Candidates {
    /// Lower residual index, -10..=9. The upper candidate has index `k + 1`.
    k: i32,
    out0_q10: i32,
    out1_q10: i32,
    rd0_q25: i32,
    rd1_q25: i32,
}

/// Quantise one coefficient from one state with previous output
/// `prev_out_q10` (16-bit range) and rate-distortion value `rd_q25`.
#[inline(always)]
fn candidates(prev_out_q10: i32, rd_q25: i32, c: &Coefficient) -> Candidates {
    let pred_q10 = silk_smulbb(c.pred_coef_q8, prev_out_q10) >> 8;
    let res_q10 = c.in_q10 - pred_q10;
    let k = (silk_smulbb(c.inv_quant_step_size_q6, res_q10) >> 16).clamp(
        -NLSF_QUANT_MAX_AMPLITUDE_EXT,
        NLSF_QUANT_MAX_AMPLITUDE_EXT - 1,
    );
    let slot = (k + NLSF_QUANT_MAX_AMPLITUDE_EXT) as usize;
    let level0_q10 = c.levels[slot];
    let level1_q10 = c.levels[slot + 1];
    // The quantisation error is the input minus the output, which equals
    // the residual minus the level.
    let diff0_q10 = res_q10 - level0_q10;
    let diff1_q10 = res_q10 - level1_q10;
    Candidates {
        k,
        out0_q10: level0_q10 + pred_q10,
        out1_q10: level1_q10 + pred_q10,
        rd0_q25: silk_mla(rd_q25, silk_smulbb(diff0_q10, diff0_q10), c.w_q5)
            .wrapping_add(c.cost[slot]),
        rd1_q25: silk_mla(rd_q25, silk_smulbb(diff1_q10, diff1_q10), c.w_q5)
            .wrapping_add(c.cost[slot + 1]),
    }
}

/// The arguments of the search that `silk_nlsf_encode` prepares.
///
/// - `rate_cost[rate_row[i]]` is the table from `silk_nlsf_rate_cost` for
///   coefficient `i`.
/// - `levels` is the table from `silk_nlsf_quant_levels`.
pub(crate) struct NlsfQuantInput<'a> {
    pub x_q10: &'a [i16; MAX_LPC_ORDER],
    pub w_q5: &'a [i16; MAX_LPC_ORDER],
    pub pred_coef_q8: &'a [u8; MAX_LPC_ORDER],
    pub rate_cost: &'a [[i32; NLSF_QUANT_LEVELS]],
    pub rate_row: &'a [u8; MAX_LPC_ORDER],
    pub levels: &'a [i32; NLSF_QUANT_LEVELS],
    pub inv_quant_step_size_q6: i16,
    pub order: usize,
}

impl NlsfQuantInput<'_> {
    #[inline(always)]
    fn coefficient(&self, i: usize) -> Coefficient<'_> {
        Coefficient {
            in_q10: self.x_q10[i] as i32,
            pred_coef_q8: self.pred_coef_q8[i] as i32,
            w_q5: self.w_q5[i] as i32,
            inv_quant_step_size_q6: self.inv_quant_step_size_q6 as i32,
            levels: self.levels,
            cost: &self.rate_cost[self.rate_row[i] as usize],
        }
    }
}

/// Position and value of the lowest of the eight final rate-distortion
/// values. The first one wins a tie.
#[inline(always)]
fn lowest(rd_q25: &[i32; 2 * N_STATES]) -> (usize, i32) {
    let mut best = 0usize;
    let mut min_q25 = i32::MAX;
    for (j, &rd) in rd_q25.iter().enumerate() {
        if min_q25 > rd {
            min_q25 = rd;
            best = j;
        }
    }
    (best, min_q25)
}

/// The search as libopus writes it. `silk_nlsf_del_dec_quant_tables` gives
/// the same result faster and uses this version for the input it does not
/// cover.
///
/// Returns the rate-distortion value (Q25) of the chosen path and writes its
/// `order` residual indices to `indices`.
pub(crate) fn silk_nlsf_del_dec_quant_reference(indices: &mut [i8], input: &NlsfQuantInput) -> i32 {
    let order = input.order;
    assert!(order <= MAX_LPC_ORDER);

    let mut n_states: usize = 1;
    let mut ind_sort = [0usize; N_STATES];
    let mut ind = [[0i8; MAX_LPC_ORDER]; N_STATES];
    let mut prev_out_q10 = [0i16; 2 * N_STATES];
    let mut rd_q25 = [0i32; 2 * N_STATES];
    let mut rd_min_q25 = [0i32; N_STATES];
    let mut rd_max_q25 = [0i32; N_STATES];

    for i in (0..order).rev() {
        let coef = input.coefficient(i);
        for j in 0..n_states {
            let c = candidates(prev_out_q10[j] as i32, rd_q25[j], &coef);
            ind[j][i] = c.k as i8;
            prev_out_q10[j] = c.out0_q10 as i16;
            prev_out_q10[j + n_states] = c.out1_q10 as i16;
            rd_q25[j] = c.rd0_q25;
            rd_q25[j + n_states] = c.rd1_q25;
        }

        if n_states <= N_STATES / 2 {
            // Double the number of states.
            for j in 0..n_states {
                ind[j + n_states][i] = ind[j][i] + 1;
            }
            n_states <<= 1;
            for j in n_states..N_STATES {
                ind[j][i] = ind[j - n_states][i];
            }
        } else {
            // Order the two candidates of each state: the better one stays
            // in the place of the state.
            for j in 0..N_STATES {
                if rd_q25[j] > rd_q25[j + N_STATES] {
                    rd_max_q25[j] = rd_q25[j];
                    rd_min_q25[j] = rd_q25[j + N_STATES];
                    rd_q25[j] = rd_min_q25[j];
                    rd_q25[j + N_STATES] = rd_max_q25[j];
                    prev_out_q10.swap(j, j + N_STATES);
                    ind_sort[j] = j + N_STATES;
                } else {
                    rd_min_q25[j] = rd_q25[j];
                    rd_max_q25[j] = rd_q25[j + N_STATES];
                    ind_sort[j] = j;
                }
            }
            // While the best of the worse candidates beats the worst of the
            // better ones, it takes that one's place.
            loop {
                let mut min_max_q25 = i32::MAX;
                let mut max_min_q25 = 0;
                let mut ind_min_max = 0;
                let mut ind_max_min = 0;
                for j in 0..N_STATES {
                    if min_max_q25 > rd_max_q25[j] {
                        min_max_q25 = rd_max_q25[j];
                        ind_min_max = j;
                    }
                    if max_min_q25 < rd_min_q25[j] {
                        max_min_q25 = rd_min_q25[j];
                        ind_max_min = j;
                    }
                }
                if min_max_q25 >= max_min_q25 {
                    break;
                }
                ind_sort[ind_max_min] = ind_sort[ind_min_max] ^ N_STATES;
                rd_q25[ind_max_min] = rd_q25[ind_min_max + N_STATES];
                prev_out_q10[ind_max_min] = prev_out_q10[ind_min_max + N_STATES];
                rd_min_q25[ind_max_min] = 0;
                rd_max_q25[ind_min_max] = i32::MAX;
                ind[ind_max_min] = ind[ind_min_max];
            }
            for j in 0..N_STATES {
                ind[j][i] += (ind_sort[j] >> NLSF_QUANT_DEL_DEC_STATES_LOG2) as i8;
            }
        }
    }

    let (best, min_q25) = lowest(&rd_q25);
    indices[..order].copy_from_slice(&ind[best & (N_STATES - 1)][..order]);
    indices[0] += (best >> NLSF_QUANT_DEL_DEC_STATES_LOG2) as i8;
    min_q25
}

// ---------------------------------------------------------------------------
// Fast version: one candidate packed into a 64-bit key
// ---------------------------------------------------------------------------
//
// bits 63-32  rate-distortion value (Q25, signed)
// bits 31-30  place: the number of the state the candidate was made from
// bits 23-16  residual index plus 10 (0..=20)
// bits 15-0   output (Q10), 16 bits as the reference version stores it
//
// Comparing two keys as signed 64-bit numbers compares the rate-distortion
// values first. Candidates with equal values are ordered by place, and the
// two candidates of one state by their residual index. This is the order in
// which the reference version looks at the candidates, so the smallest key
// is the candidate the reference version finds first.

const KEY_PLACE_SHIFT: u32 = 30;
const KEY_SLOT_SHIFT: u32 = 16;
/// Inverts the place bits. With it the largest key among equal
/// rate-distortion values belongs to the lowest place.
const KEY_FLIP_PLACE: i64 = 3 << KEY_PLACE_SHIFT;

#[inline(always)]
fn key(rd_q25: i32, place: usize, slot: usize, out_q10: i32) -> i64 {
    let low =
        (place as u32) << KEY_PLACE_SHIFT | (slot as u32) << KEY_SLOT_SHIFT | out_q10 as u16 as u32;
    (rd_q25 as i64) << 32 | low as i64
}

#[inline(always)]
fn key_rd_q25(key: i64) -> i32 {
    (key >> 32) as i32
}

#[inline(always)]
fn key_place(key: i64) -> usize {
    (key >> KEY_PLACE_SHIFT) as usize & (N_STATES - 1)
}

/// Residual index plus 10.
#[inline(always)]
fn key_slot(key: i64) -> u8 {
    (key >> KEY_SLOT_SHIFT) as u8
}

#[inline(always)]
fn key_out_q10(key: i64) -> i32 {
    key as i16 as i32
}

/// Both candidates of the state in `place` for one coefficient: the better
/// one first. `state` is the key of the candidate the state continues from.
#[inline(always)]
fn ordered_candidates(place: usize, state: i64, c: &Coefficient) -> (i64, i64) {
    let both = candidates(key_out_q10(state), key_rd_q25(state), c);
    let slot = (both.k + NLSF_QUANT_MAX_AMPLITUDE_EXT) as usize;
    let lower = key(both.rd0_q25, place, slot, both.out0_q10);
    let upper = key(both.rd1_q25, place, slot + 1, both.out1_q10);
    (lower.min(upper), lower.max(upper))
}

/// The smallest and the second smallest of four keys. No two keys are equal.
#[inline(always)]
fn two_smallest(k: &[i64; N_STATES]) -> (i64, i64) {
    let (low01, high01) = (k[0].min(k[1]), k[0].max(k[1]));
    let (low23, high23) = (k[2].min(k[3]), k[2].max(k[3]));
    // The second smallest is the larger of the two pair minima, or the other
    // key of the pair that holds the smallest.
    let other = select(low01 < low23, high01, high23);
    (low01.min(low23), low01.max(low23).min(other))
}

/// The largest and the second largest of four keys. No two keys are equal.
#[inline(always)]
fn two_largest(k: &[i64; N_STATES]) -> (i64, i64) {
    let (low01, high01) = (k[0].min(k[1]), k[0].max(k[1]));
    let (low23, high23) = (k[2].min(k[3]), k[2].max(k[3]));
    let other = select(high01 > high23, low01, low23);
    (high01.max(high23), high01.min(high23).max(other))
}

/// Index paths of the four places, one byte per coefficient (the residual
/// index plus 10). `paths[0][j]` holds coefficients 0 to 7 of place `j` and
/// `paths[1][j]` coefficients 8 to 15. Every access is one whole word, so a
/// load always reads exactly what one store wrote.
type Paths = [[u64; N_STATES]; 2];

/// Prune eight candidates to four, with the result of the reference version:
/// `better[j]` and `worse[j]` are the two candidates of state `j`. Returns
/// the candidate that ends up in each place. `paths` are moved along.
///
/// The reference version starts with the better candidate of every state.
/// Then, as long as the best of the worse candidates that has not moved has
/// a lower rate-distortion value than the worst candidate that is still in
/// its own place, the first replaces the second.
///
/// When no rate-distortion value is negative, this happens at most twice.
/// The reason: a state whose worse candidate moves has a better candidate
/// with a value that is lower still, so that state is not replaced later,
/// and a state that was replaced has no candidate left to give. Each
/// replacement therefore uses up two of the four states. The pairs are the
/// smallest key of `worse` with the largest of `better`, then the second
/// smallest with the second largest.
///
/// The selections are done without a branch, because their outcome is hard
/// to predict.
#[inline(always)]
fn prune(better: &[i64; N_STATES], worse: &[i64; N_STATES], paths: &mut Paths) -> [i64; N_STATES] {
    let (move1, move2) = two_smallest(worse);
    let (leave1, leave2) = two_largest(&better.map(|key| key ^ KEY_FLIP_PLACE));
    let replace1 = key_rd_q25(move1) < key_rd_q25(leave1);
    let replace2 = replace1 & (key_rd_q25(move2) < key_rd_q25(leave2));
    let to1 = key_place(leave1 ^ KEY_FLIP_PLACE);
    let to2 = key_place(leave2 ^ KEY_FLIP_PLACE);
    // Without a replacement the target is 4 or more: it matches no place.
    let target1 = to1 | (!replace1 as usize) << NLSF_QUANT_DEL_DEC_STATES_LOG2;
    let target2 = to2 | (!replace2 as usize) << NLSF_QUANT_DEL_DEC_STATES_LOG2;
    let mut survivors = [0i64; N_STATES];
    for j in 0..N_STATES {
        let stay_or_second = select(target2 == j, move2, better[j]);
        survivors[j] = select(target1 == j, move1, stay_or_second);
    }
    // Without a replacement a path is copied onto itself.
    let from1 = select(replace1, key_place(move1), to1);
    let from2 = select(replace2, key_place(move2), to2);
    for half in paths.iter_mut() {
        half[to1] = half[from1];
        half[to2] = half[from2];
    }
    survivors
}

/// The search, with the tables prepared by the caller (see
/// `NlsfQuantInput`).
///
/// Returns the rate-distortion value (Q25) of the chosen path and writes its
/// `order` residual indices to `indices`. The result is that of
/// `silk_nlsf_del_dec_quant_reference`.
pub(crate) fn silk_nlsf_del_dec_quant_tables(indices: &mut [i8], input: &NlsfQuantInput) -> i32 {
    let order = input.order;
    assert!(order <= MAX_LPC_ORDER);

    // Index path of each state. Entry `i` is the index of coefficient `i`.
    let mut ind = [[0i8; MAX_LPC_ORDER]; N_STATES];
    // Entries 0..4 belong to the states. Entries 4..8 hold the second
    // candidate of each state.
    let mut prev_out_q10 = [0i16; 2 * N_STATES];
    let mut rd_q25 = [0i32; 2 * N_STATES];

    // The first coefficients: every state splits into two, until there are
    // four states. This part is the reference code unchanged.
    let mut n_states: usize = 1;
    let mut i = order;
    while i > 0 && n_states <= N_STATES / 2 {
        i -= 1;
        let coef = input.coefficient(i);
        for j in 0..n_states {
            let c = candidates(prev_out_q10[j] as i32, rd_q25[j], &coef);
            ind[j][i] = c.k as i8;
            prev_out_q10[j] = c.out0_q10 as i16;
            prev_out_q10[j + n_states] = c.out1_q10 as i16;
            rd_q25[j] = c.rd0_q25;
            rd_q25[j + n_states] = c.rd1_q25;
        }
        for j in 0..n_states {
            ind[j + n_states][i] = ind[j][i] + 1;
        }
        n_states <<= 1;
        for j in n_states..N_STATES {
            ind[j][i] = ind[j - n_states][i];
        }
    }
    if i == 0 {
        // Fewer than three coefficients: there is nothing to prune.
        let (best, min_q25) = lowest(&rd_q25);
        indices[..order].copy_from_slice(&ind[best & (N_STATES - 1)][..order]);
        indices[0] += (best >> NLSF_QUANT_DEL_DEC_STATES_LOG2) as i8;
        return min_q25;
    }

    // The other coefficients: four states give eight candidates, and the four
    // with the lowest rate-distortion value survive.
    //
    // `state[j]` is the candidate in place `j`. `worse[j]` is the worse
    // candidate of state `j` for the last coefficient done. These and
    // `better` are only read and written at fixed positions, so the compiler
    // keeps them in registers; the next coefficient waits for them.
    let mut state: [i64; N_STATES] =
        core::array::from_fn(|j| key(rd_q25[j], j, 0, prev_out_q10[j] as i32));
    let mut worse = [0i64; N_STATES];
    // Coefficients that are not done yet have a zero byte.
    let mut paths: Paths = [[0; N_STATES]; 2];
    for j in 0..N_STATES {
        let mut bytes = [0u8; MAX_LPC_ORDER];
        for (byte, &index) in bytes[i..order].iter_mut().zip(&ind[j][i..order]) {
            *byte = (index as i32 + NLSF_QUANT_MAX_AMPLITUDE_EXT) as u8;
        }
        paths[0][j] = u64::from_le_bytes(*bytes.first_chunk().unwrap());
        paths[1][j] = u64::from_le_bytes(*bytes.last_chunk().unwrap());
    }

    while i > 0 {
        i -= 1;
        let coef = input.coefficient(i);

        let mut better = [0i64; N_STATES];
        for j in 0..N_STATES {
            (better[j], worse[j]) = ordered_candidates(j, state[j], &coef);
        }
        // `prune` needs rate-distortion values that are not negative. A
        // negative one can only come from a sum that overflowed, which valid
        // encoder input does not produce. The sign bit of a key is the sign
        // of its value.
        //
        // These are two tests and not one over all four keys on purpose:
        // with a single test the compiler merges the four candidate
        // computations into vector instructions. Each state has to wait for
        // its own result, and the moves into vector registers and back make
        // that wait longer (measured: 10 % more time for the whole search).
        if (better[0] | better[1]) < 0 || (better[2] | better[3]) < 0 {
            #[cfg(test)]
            tests::count_hand_over();
            return silk_nlsf_del_dec_quant_reference(indices, input);
        }
        state = prune(&better, &worse, &mut paths);

        // The paths were moved without the index of this coefficient.
        let half = &mut paths[i / 8];
        let shift = 8 * (i % 8);
        for j in 0..N_STATES {
            half[j] |= (key_slot(state[j]) as u64) << shift;
        }
    }

    for j in 0..N_STATES {
        rd_q25[j] = key_rd_q25(state[j]);
        rd_q25[j + N_STATES] = key_rd_q25(worse[j]);
    }
    let (best, min_q25) = lowest(&rd_q25);
    let place = best & (N_STATES - 1);
    let path = [paths[0][place].to_le_bytes(), paths[1][place].to_le_bytes()];
    for (index, &slot) in indices[..order].iter_mut().zip(path.as_flattened()) {
        *index = (slot as i32 - NLSF_QUANT_MAX_AMPLITUDE_EXT) as i8;
    }
    indices[0] += (best >> NLSF_QUANT_DEL_DEC_STATES_LOG2) as i8;
    min_q25
}

/// Delayed-decision quantiser for the NLSF residual (libopus
/// `silk_NLSF_del_dec_quant`).
///
/// This entry point builds the level table and one rate table per
/// coefficient on every call. `silk_nlsf_encode` builds the tables once per
/// frame and calls `silk_nlsf_del_dec_quant_tables` directly.
pub fn silk_nlsf_del_dec_quant(
    indices: &mut [i8],
    x_q10: &[i16],
    w_q5: &[i16],
    pred_coef_q8: &[u8],
    ec_ix: &[i16],
    ec_rates_q5: &[u8],
    quant_step_size_q16: i32,
    inv_quant_step_size_q6: i16,
    mu_q20: i32,
    order: i16,
) -> i32 {
    let order = order as usize;
    assert!(order <= MAX_LPC_ORDER);

    let mut x = [0i16; MAX_LPC_ORDER];
    let mut w = [0i16; MAX_LPC_ORDER];
    let mut pred = [0u8; MAX_LPC_ORDER];
    x[..order].copy_from_slice(&x_q10[..order]);
    w[..order].copy_from_slice(&w_q5[..order]);
    pred[..order].copy_from_slice(&pred_coef_q8[..order]);

    let mut rate_cost = [[0i32; NLSF_QUANT_LEVELS]; MAX_LPC_ORDER];
    let mut rate_row = [0u8; MAX_LPC_ORDER];
    for i in 0..order {
        // A rate outside `ec_rates_q5` is never used by a valid codebook;
        // it counts as zero here.
        let mut table_rates_q5 = [0u8; NLSF_EC_TABLE_LEN];
        if let Some(table) = usize::try_from(ec_ix[i])
            .ok()
            .and_then(|start| ec_rates_q5.get(start..))
        {
            for (rate, &from_table) in table_rates_q5.iter_mut().zip(table) {
                *rate = from_table;
            }
        }
        rate_cost[i] = silk_nlsf_rate_cost(mu_q20, &table_rates_q5);
        rate_row[i] = i as u8;
    }

    silk_nlsf_del_dec_quant_tables(
        indices,
        &NlsfQuantInput {
            x_q10: &x,
            w_q5: &w,
            pred_coef_q8: &pred,
            rate_cost: &rate_cost,
            rate_row: &rate_row,
            levels: &silk_nlsf_quant_levels(quant_step_size_q16),
            inv_quant_step_size_q6,
            order,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::silk::define::{NLSF_QUANT_MAX_AMPLITUDE, NLSF_QUANT_MAX_AMPLITUDE_EXT};
    use crate::silk::macros::{silk_limit, silk_smlabb};
    use crate::silk::tables_nlsf::{SILK_NLSF_CB_NB_MB, SILK_NLSF_CB_WB};

    /// The function as it was before the two versions in this file replaced
    /// it: a direct port of libopus `silk_NLSF_del_dec_quant`. The tests
    /// require that every version gives its result.
    #[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
    fn original(
        indices: &mut [i8],
        x_q10: &[i16],
        w_q5: &[i16],
        pred_coef_q8: &[u8],
        ec_ix: &[i16],
        ec_rates_q5: &[u8],
        quant_step_size_q16: i32,
        inv_quant_step_size_q6: i16,
        mu_q20: i32,
        order: i16,
    ) -> i32 {
        let mut n_states: usize = 1;
        let mut ind_tmp: i32;
        let mut ind_min_max: usize;
        let mut ind_max_min: usize;
        let mut in_q10: i32;
        let mut res_q10: i32;
        let mut pred_q10: i32;
        let mut diff_q10: i32;
        let mut rate0_q5: i32;
        let mut rate1_q5: i32;
        let mut out0_q10: i32;
        let mut out1_q10: i32;
        let mut rd_tmp_q25: i32;
        let mut min_q25: i32;
        let mut min_max_q25: i32;
        let mut max_min_q25: i32;

        let mut ind_sort = [0usize; NLSF_QUANT_DEL_DEC_STATES];
        let mut ind = [[0i8; MAX_LPC_ORDER]; NLSF_QUANT_DEL_DEC_STATES];
        let mut prev_out_q10 = [0i16; 2 * NLSF_QUANT_DEL_DEC_STATES];
        let mut rd_q25 = [0i32; 2 * NLSF_QUANT_DEL_DEC_STATES];
        let mut rd_min_q25 = [0i32; NLSF_QUANT_DEL_DEC_STATES];
        let mut rd_max_q25 = [0i32; NLSF_QUANT_DEL_DEC_STATES];

        let mut out0_q10_table = [0i32; 2 * NLSF_QUANT_MAX_AMPLITUDE_EXT as usize];
        let mut out1_q10_table = [0i32; 2 * NLSF_QUANT_MAX_AMPLITUDE_EXT as usize];

        for i in -(NLSF_QUANT_MAX_AMPLITUDE_EXT)..NLSF_QUANT_MAX_AMPLITUDE_EXT {
            let mut tmp_out0_q10 = i << 10;
            let mut tmp_out1_q10 = tmp_out0_q10 + 1024;
            if i > 0 {
                tmp_out0_q10 -= NLSF_QUANT_LEVEL_ADJ;
                tmp_out1_q10 -= NLSF_QUANT_LEVEL_ADJ;
            } else if i == 0 {
                tmp_out1_q10 -= NLSF_QUANT_LEVEL_ADJ;
            } else if i == -1 {
                tmp_out0_q10 += NLSF_QUANT_LEVEL_ADJ;
            } else {
                tmp_out0_q10 += NLSF_QUANT_LEVEL_ADJ;
                tmp_out1_q10 += NLSF_QUANT_LEVEL_ADJ;
            }
            out0_q10_table[(i + NLSF_QUANT_MAX_AMPLITUDE_EXT) as usize] =
                (silk_smulbb(tmp_out0_q10, quant_step_size_q16)) >> 16;
            out1_q10_table[(i + NLSF_QUANT_MAX_AMPLITUDE_EXT) as usize] =
                (silk_smulbb(tmp_out1_q10, quant_step_size_q16)) >> 16;
        }

        rd_q25[0] = 0;
        prev_out_q10[0] = 0;

        for i in (0..order as usize).rev() {
            let rates_q5_ptr = &ec_rates_q5[ec_ix[i] as usize..];
            in_q10 = x_q10[i] as i32;

            for j in 0..n_states {
                pred_q10 = silk_smulbb(pred_coef_q8[i] as i32, prev_out_q10[j] as i32) >> 8;
                res_q10 = in_q10 - pred_q10;
                ind_tmp = (silk_smulbb(inv_quant_step_size_q6 as i32, res_q10)) >> 16;
                ind_tmp = silk_limit(
                    ind_tmp,
                    -NLSF_QUANT_MAX_AMPLITUDE_EXT,
                    NLSF_QUANT_MAX_AMPLITUDE_EXT - 1,
                );
                ind[j][i] = ind_tmp as i8;

                out0_q10 = out0_q10_table[(ind_tmp + NLSF_QUANT_MAX_AMPLITUDE_EXT) as usize];
                out1_q10 = out1_q10_table[(ind_tmp + NLSF_QUANT_MAX_AMPLITUDE_EXT) as usize];

                out0_q10 += pred_q10;
                out1_q10 += pred_q10;
                prev_out_q10[j] = out0_q10 as i16;
                prev_out_q10[j + n_states] = out1_q10 as i16;

                if ind_tmp + 1 >= NLSF_QUANT_MAX_AMPLITUDE {
                    if ind_tmp + 1 == NLSF_QUANT_MAX_AMPLITUDE {
                        rate0_q5 =
                            rates_q5_ptr[(ind_tmp + NLSF_QUANT_MAX_AMPLITUDE) as usize] as i32;
                        rate1_q5 = 280;
                    } else {
                        rate0_q5 = silk_smlabb(280 - 43 * NLSF_QUANT_MAX_AMPLITUDE, 43, ind_tmp);
                        rate1_q5 = rate0_q5 + 43;
                    }
                } else if ind_tmp <= -NLSF_QUANT_MAX_AMPLITUDE {
                    if ind_tmp == -NLSF_QUANT_MAX_AMPLITUDE {
                        rate0_q5 = 280;
                        rate1_q5 =
                            rates_q5_ptr[(ind_tmp + 1 + NLSF_QUANT_MAX_AMPLITUDE) as usize] as i32;
                    } else {
                        rate0_q5 = silk_smlabb(280 - 43 * NLSF_QUANT_MAX_AMPLITUDE, -43, ind_tmp);
                        rate1_q5 = rate0_q5 - 43;
                    }
                } else {
                    rate0_q5 = rates_q5_ptr[(ind_tmp + NLSF_QUANT_MAX_AMPLITUDE) as usize] as i32;
                    rate1_q5 =
                        rates_q5_ptr[(ind_tmp + 1 + NLSF_QUANT_MAX_AMPLITUDE) as usize] as i32;
                }

                rd_tmp_q25 = rd_q25[j];
                diff_q10 = in_q10 - out0_q10;
                rd_q25[j] = silk_smlabb(
                    silk_mla(rd_tmp_q25, silk_smulbb(diff_q10, diff_q10), w_q5[i] as i32),
                    mu_q20,
                    rate0_q5,
                );
                diff_q10 = in_q10 - out1_q10;
                rd_q25[j + n_states] = silk_smlabb(
                    silk_mla(rd_tmp_q25, silk_smulbb(diff_q10, diff_q10), w_q5[i] as i32),
                    mu_q20,
                    rate1_q5,
                );
            }

            if n_states <= NLSF_QUANT_DEL_DEC_STATES / 2 {
                for j in 0..n_states {
                    ind[j + n_states][i] = ind[j][i] + 1;
                }
                n_states <<= 1;
                for j in n_states..NLSF_QUANT_DEL_DEC_STATES {
                    ind[j][i] = ind[j - n_states][i];
                }
            } else {
                for j in 0..NLSF_QUANT_DEL_DEC_STATES {
                    if rd_q25[j] > rd_q25[j + NLSF_QUANT_DEL_DEC_STATES] {
                        rd_max_q25[j] = rd_q25[j];
                        rd_min_q25[j] = rd_q25[j + NLSF_QUANT_DEL_DEC_STATES];
                        rd_q25[j] = rd_min_q25[j];
                        rd_q25[j + NLSF_QUANT_DEL_DEC_STATES] = rd_max_q25[j];

                        prev_out_q10.swap(j, j + NLSF_QUANT_DEL_DEC_STATES);
                        ind_sort[j] = j + NLSF_QUANT_DEL_DEC_STATES;
                    } else {
                        rd_min_q25[j] = rd_q25[j];
                        rd_max_q25[j] = rd_q25[j + NLSF_QUANT_DEL_DEC_STATES];
                        ind_sort[j] = j;
                    }
                }

                loop {
                    min_max_q25 = i32::MAX;
                    max_min_q25 = 0;
                    ind_min_max = 0;
                    ind_max_min = 0;
                    for j in 0..NLSF_QUANT_DEL_DEC_STATES {
                        if min_max_q25 > rd_max_q25[j] {
                            min_max_q25 = rd_max_q25[j];
                            ind_min_max = j;
                        }
                        if max_min_q25 < rd_min_q25[j] {
                            max_min_q25 = rd_min_q25[j];
                            ind_max_min = j;
                        }
                    }
                    if min_max_q25 >= max_min_q25 {
                        break;
                    }

                    ind_sort[ind_max_min] = ind_sort[ind_min_max] ^ NLSF_QUANT_DEL_DEC_STATES;
                    rd_q25[ind_max_min] = rd_q25[ind_min_max + NLSF_QUANT_DEL_DEC_STATES];
                    prev_out_q10[ind_max_min] =
                        prev_out_q10[ind_min_max + NLSF_QUANT_DEL_DEC_STATES];
                    rd_min_q25[ind_max_min] = 0;
                    rd_max_q25[ind_min_max] = i32::MAX;
                    ind[ind_max_min] = ind[ind_min_max];
                }

                for j in 0..NLSF_QUANT_DEL_DEC_STATES {
                    ind[j][i] += (ind_sort[j] >> NLSF_QUANT_DEL_DEC_STATES_LOG2) as i8;
                }
            }
        }

        ind_tmp = 0;
        min_q25 = i32::MAX;
        for j in 0..2 * NLSF_QUANT_DEL_DEC_STATES {
            if min_q25 > rd_q25[j] {
                min_q25 = rd_q25[j];
                ind_tmp = j as i32;
            }
        }
        indices[..(order as usize)].copy_from_slice(
            &ind[(ind_tmp & (NLSF_QUANT_DEL_DEC_STATES as i32 - 1)) as usize][..(order as usize)],
        );
        indices[0] += (ind_tmp >> NLSF_QUANT_DEL_DEC_STATES_LOG2) as i8;

        min_q25
    }

    std::thread_local! {
        /// How often the fast version handed over to the reference version
        /// in the current test (each test runs in its own thread).
        static HANDED_OVER: core::cell::Cell<usize> = const { core::cell::Cell::new(0) };
    }

    pub(super) fn count_hand_over() {
        HANDED_OVER.with(|count| count.set(count.get() + 1));
    }

    fn handed_over() -> usize {
        HANDED_OVER.with(|count| count.get())
    }

    /// xorshift64 generator with a fixed seed, so the test data never
    /// changes.
    struct XorShift(u64);

    impl XorShift {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }

        /// Uniform integer in `low..=high`.
        fn range(&mut self, low: i32, high: i32) -> i32 {
            low + (self.next() % (high as i64 - low as i64 + 1) as u64) as i32
        }
    }

    /// One call of the quantiser, in the form of the public entry point.
    struct Case {
        x_q10: [i16; MAX_LPC_ORDER],
        w_q5: [i16; MAX_LPC_ORDER],
        pred_coef_q8: [u8; MAX_LPC_ORDER],
        /// Offset of the entropy table of each coefficient in `ec_rates_q5`.
        ec_ix: [i16; MAX_LPC_ORDER],
        ec_rates_q5: &'static [u8],
        quant_step_size_q16: i32,
        inv_quant_step_size_q6: i16,
        mu_q20: i32,
        order: usize,
    }

    /// Run the original function and the three entry points of this file on
    /// one case and require equal results. `indices` starts with a marker
    /// value, which shows that nothing past `order` is written.
    fn check(case: &Case, what: &str) {
        const MARK: i8 = 0x55;
        let order = case.order;

        let mut expected = [MARK; MAX_LPC_ORDER + 2];
        let expected_rd = original(
            &mut expected,
            &case.x_q10,
            &case.w_q5,
            &case.pred_coef_q8,
            &case.ec_ix,
            case.ec_rates_q5,
            case.quant_step_size_q16,
            case.inv_quant_step_size_q6,
            case.mu_q20,
            order as i16,
        );

        let mut indices = [MARK; MAX_LPC_ORDER + 2];
        let rd = silk_nlsf_del_dec_quant(
            &mut indices,
            &case.x_q10,
            &case.w_q5,
            &case.pred_coef_q8,
            &case.ec_ix,
            case.ec_rates_q5,
            case.quant_step_size_q16,
            case.inv_quant_step_size_q6,
            case.mu_q20,
            order as i16,
        );
        assert_eq!(
            (rd, indices),
            (expected_rd, expected),
            "public entry point, {what}"
        );

        // The two versions with prepared tables. The tables are laid out as
        // `silk_nlsf_encode` does it when every offset is a whole number of
        // tables, and one table per coefficient otherwise.
        let levels = silk_nlsf_quant_levels(case.quant_step_size_q16);
        let mut rate_cost = [[0i32; NLSF_QUANT_LEVELS]; MAX_LPC_ORDER];
        let mut rate_row = [0u8; MAX_LPC_ORDER];
        let whole_tables = case.ec_ix[..order]
            .iter()
            .all(|&offset| offset as usize % NLSF_EC_TABLE_LEN == 0);
        if whole_tables {
            let (tables, _) = case.ec_rates_q5.as_chunks::<NLSF_EC_TABLE_LEN>();
            for (cost, table) in rate_cost.iter_mut().zip(tables) {
                *cost = silk_nlsf_rate_cost(case.mu_q20, table);
            }
            for i in 0..order {
                rate_row[i] = (case.ec_ix[i] as usize / NLSF_EC_TABLE_LEN) as u8;
            }
        } else {
            for i in 0..order {
                let start = case.ec_ix[i] as usize;
                let table = case.ec_rates_q5[start..start + NLSF_EC_TABLE_LEN]
                    .try_into()
                    .unwrap();
                rate_cost[i] = silk_nlsf_rate_cost(case.mu_q20, table);
                rate_row[i] = i as u8;
            }
        }
        let input = NlsfQuantInput {
            x_q10: &case.x_q10,
            w_q5: &case.w_q5,
            pred_coef_q8: &case.pred_coef_q8,
            rate_cost: &rate_cost,
            rate_row: &rate_row,
            levels: &levels,
            inv_quant_step_size_q6: case.inv_quant_step_size_q6,
            order,
        };

        let mut indices = [MARK; MAX_LPC_ORDER + 2];
        let rd = silk_nlsf_del_dec_quant_reference(&mut indices, &input);
        assert_eq!(
            (rd, indices),
            (expected_rd, expected),
            "reference version, {what}"
        );

        let mut indices = [MARK; MAX_LPC_ORDER + 2];
        let rd = silk_nlsf_del_dec_quant_tables(&mut indices, &input);
        assert_eq!(
            (rd, indices),
            (expected_rd, expected),
            "fast version, {what}"
        );
    }

    /// A case with the value ranges of the encoder: one of the two codebooks
    /// with its step sizes and entropy tables. (In 6 s of speech the residual
    /// stayed within +-1012, the weights within 6..=1950 and `mu_q20` within
    /// 2101..=3137; the largest rate-distortion value was 14.9 million.)
    fn encoder_case(rng: &mut XorShift, wideband: bool, order: usize) -> Case {
        let cb = if wideband {
            &SILK_NLSF_CB_WB
        } else {
            &SILK_NLSF_CB_NB_MB
        };
        let tables = (cb.ec_rates_q5.len() / NLSF_EC_TABLE_LEN) as i32;
        Case {
            x_q10: core::array::from_fn(|_| rng.range(-1100, 1100) as i16),
            w_q5: core::array::from_fn(|_| rng.range(1, 2000) as i16),
            pred_coef_q8: core::array::from_fn(|_| rng.range(40, 200) as u8),
            ec_ix: core::array::from_fn(|_| {
                (rng.range(0, tables - 1) * NLSF_EC_TABLE_LEN as i32) as i16
            }),
            ec_rates_q5: cb.ec_rates_q5,
            quant_step_size_q16: cb.quant_step_size_q16,
            inv_quant_step_size_q6: cb.inv_quant_step_size_q6,
            mu_q20: rng.range(1, 6000),
            order,
        }
    }

    #[test]
    fn encoder_ranges_match_original() {
        let mut rng = XorShift(0x9E37_79B9_7F4A_7C15);
        for round in 0..20_000 {
            let wideband = round % 2 == 0;
            let order = if wideband { 16 } else { 10 };
            check(&encoder_case(&mut rng, wideband, order), "encoder ranges");
        }
        // All of these went through the fast version.
        assert_eq!(handed_over(), 0);
    }

    #[test]
    fn every_order_matches_original() {
        let mut rng = XorShift(0x0123_4567_89AB_CDEF);
        for order in 0..=MAX_LPC_ORDER {
            for round in 0..500 {
                check(
                    &encoder_case(&mut rng, round % 2 == 0, order),
                    "every order",
                );
            }
        }
    }

    /// Equal rate-distortion values: the choice then depends only on the
    /// order in which the candidates are looked at.
    #[test]
    fn ties_match_original() {
        let mut rng = XorShift(0xDEAD_BEEF_CAFE_F00D);
        for round in 0..20_000 {
            let mut case = encoder_case(&mut rng, round % 2 == 0, 3 + round % 14);
            // No weight on the error, no weight on the rate, or both; or
            // values so coarse that many sums coincide.
            match round % 5 {
                0 => case.w_q5 = [0; MAX_LPC_ORDER],
                1 => case.mu_q20 = 0,
                2 => {
                    case.w_q5 = [0; MAX_LPC_ORDER];
                    case.mu_q20 = 0;
                }
                3 => {
                    case.w_q5 = [0; MAX_LPC_ORDER];
                    case.mu_q20 = 1;
                }
                _ => {
                    case.x_q10 = core::array::from_fn(|_| rng.range(-2, 2) as i16 * 1024);
                    case.w_q5 = core::array::from_fn(|_| rng.range(0, 1) as i16);
                    case.mu_q20 = rng.range(0, 2);
                }
            }
            check(&case, "ties");
        }
        assert_eq!(handed_over(), 0);
    }

    /// Values outside what the encoder produces, but without overflow: the
    /// whole 16-bit range of predictor outputs, any table offset, any step
    /// size. These stay in the fast version.
    #[test]
    fn wide_values_match_original() {
        let mut rng = XorShift(0x2468_ACE0_1357_9BDF);
        for round in 0..20_000 {
            let mut case = encoder_case(&mut rng, round % 2 == 0, 1 + round % 16);
            case.x_q10 = core::array::from_fn(|_| rng.range(-8000, 8000) as i16);
            case.w_q5 = core::array::from_fn(|_| rng.range(0, 2) as i16);
            case.pred_coef_q8 = core::array::from_fn(|_| rng.range(0, 255) as u8);
            // Any offset that keeps the seven rates that are read inside
            // the table.
            let last = (case.ec_rates_q5.len() - NLSF_EC_TABLE_LEN) as i32;
            case.ec_ix = core::array::from_fn(|_| rng.range(0, last) as i16);
            if round % 4 == 0 {
                case.quant_step_size_q16 = rng.range(1, 32767);
                case.inv_quant_step_size_q6 = rng.range(1, 2000) as i16;
            }
            case.mu_q20 = rng.range(0, 3000);
            check(&case, "wide values");
        }
        assert_eq!(handed_over(), 0);
    }

    /// Values far outside what the encoder produces. The sums overflow and
    /// rate-distortion values become negative, which the fast version hands
    /// over to the reference version.
    #[test]
    fn overflowing_values_match_original() {
        let mut rng = XorShift(0x0F1E_2D3C_4B5A_6978);
        let extreme_i16 = [i16::MIN, -32767, -1, 0, 1, 255, 32767];
        let rounds = 20_000;
        for round in 0..rounds {
            let mut case = encoder_case(&mut rng, round % 2 == 0, 1 + round % 16);
            let pick = |rng: &mut XorShift| {
                if rng.range(0, 2) == 0 {
                    extreme_i16[rng.range(0, extreme_i16.len() as i32 - 1) as usize]
                } else {
                    rng.range(i16::MIN as i32, i16::MAX as i32) as i16
                }
            };
            case.x_q10 = core::array::from_fn(|_| pick(&mut rng));
            if round % 3 != 0 {
                case.w_q5 = core::array::from_fn(|_| pick(&mut rng));
            }
            case.pred_coef_q8 = core::array::from_fn(|_| rng.range(0, 255) as u8);
            let last = (case.ec_rates_q5.len() - NLSF_EC_TABLE_LEN) as i32;
            case.ec_ix = core::array::from_fn(|_| rng.range(0, last) as i16);
            if round % 4 == 0 {
                case.quant_step_size_q16 = rng.range(i32::MIN / 2, i32::MAX / 2);
                case.inv_quant_step_size_q6 = pick(&mut rng);
            }
            case.mu_q20 = match round % 5 {
                0 => rng.range(i32::MIN / 2, i32::MAX / 2),
                1 => i32::MAX,
                2 => -1,
                _ => rng.range(0, 1 << 20),
            };
            check(&case, "overflowing values");
        }
        // `check` runs the fast version twice per case. Most of these cases
        // must have reached the hand-over.
        let handed_over = handed_over();
        assert!(
            handed_over > rounds,
            "only {handed_over} hand-overs in {rounds} cases"
        );
    }
}
