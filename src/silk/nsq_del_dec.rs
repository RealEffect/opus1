use crate::silk::define::*;
use crate::silk::macros::*;
use crate::silk::sigproc_fix::*;
use crate::silk::structs::*;
use crate::silk::tables::*;
use core::hint::select_unpredictable as select;

/// One delayed-decision state in the layout of libopus (`NSQ_del_dec_struct`).
///
/// The quantizer itself keeps its states in `PathState` and `DelayLines`.
/// This type is the interface of `silk_noise_shape_quantizer_del_dec`.
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

/// What one delayed-decision state carries besides its delay lines.
#[derive(Copy, Clone)]
struct PathState {
    s_lpc_q14: [i32; MAX_SUB_FRAME_LENGTH + NSQ_LPC_BUF_LENGTH],
    s_ar2_q14: [i32; MAX_SHAPE_LPC_ORDER],
    lf_ar_q14: i32,
    diff_q14: i32,
    seed: i32,
    seed_init: i32,
    rd_q10: i32,
}

impl PathState {
    const ZERO: Self = Self {
        s_lpc_q14: [0; MAX_SUB_FRAME_LENGTH + NSQ_LPC_BUF_LENGTH],
        s_ar2_q14: [0; MAX_SHAPE_LPC_ORDER],
        lf_ar_q14: 0,
        diff_q14: 0,
        seed: 0,
        seed_init: 0,
        rd_q10: 0,
    };
}

/// The newest `DECISION_DELAY` samples of all delayed-decision states.
///
/// libopus keeps five arrays of `DECISION_DELAY` entries in every state and
/// copies all of them when one state continues as a copy of another. That
/// happens for about every second sample. Here every sample is stored once:
/// entry `[slot][column]` of an array holds what one state wrote for the
/// sample in `slot`. `column[k][slot]` names the column that belongs to the
/// history of state `k`. A state that continues as a copy of another state
/// takes over that state's `column` row, 40 bytes, and nothing else moves.
///
/// A state writes a new sample into its own column, so
/// `column[k][slot] == k` for the newest slot.
#[derive(Clone)]
struct DelayLines {
    rand_state: [[i32; NSQ_MAX_STATES_OPERATING]; DECISION_DELAY],
    q_q10: [[i32; NSQ_MAX_STATES_OPERATING]; DECISION_DELAY],
    xq_q14: [[i32; NSQ_MAX_STATES_OPERATING]; DECISION_DELAY],
    pred_q15: [[i32; NSQ_MAX_STATES_OPERATING]; DECISION_DELAY],
    shape_q14: [[i32; NSQ_MAX_STATES_OPERATING]; DECISION_DELAY],
    /// One row per state, and a last row that takes the copy when no state
    /// is replaced (see `copy_history`).
    column: [[u8; DECISION_DELAY]; NSQ_MAX_STATES_OPERATING + 1],
}

impl DelayLines {
    /// All samples zero, every state reading its own column.
    fn new() -> Self {
        Self {
            rand_state: [[0; NSQ_MAX_STATES_OPERATING]; DECISION_DELAY],
            q_q10: [[0; NSQ_MAX_STATES_OPERATING]; DECISION_DELAY],
            xq_q14: [[0; NSQ_MAX_STATES_OPERATING]; DECISION_DELAY],
            pred_q15: [[0; NSQ_MAX_STATES_OPERATING]; DECISION_DELAY],
            shape_q14: [[0; NSQ_MAX_STATES_OPERATING]; DECISION_DELAY],
            column: core::array::from_fn(|k| {
                [(k % NSQ_MAX_STATES_OPERATING) as u8; DECISION_DELAY]
            }),
        }
    }

    /// The column that holds the sample of `state` in `slot`.
    #[inline(always)]
    fn column(&self, state: usize, slot: usize) -> usize {
        // The stored values are below NSQ_MAX_STATES_OPERATING. The remainder
        // tells the compiler so.
        self.column[state][slot] as usize % NSQ_MAX_STATES_OPERATING
    }
}

#[inline]
fn silk_nsq_del_dec_scale_states(
    ps_enc_c: &SilkEncoderStateCommon,
    nsq: &mut SilkNSQState,
    states: &mut [PathState; NSQ_MAX_STATES_OPERATING],
    delay_lines: &mut DelayLines,
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
        for i in (nsq.s_ltp_buf_idx as usize - lag - LTP_ORDER / 2)..(nsq.s_ltp_buf_idx as usize) {
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
        for state in states[..n_states].iter_mut() {
            state.lf_ar_q14 = silk_smulww(gain_adj_q16, state.lf_ar_q14);
            state.diff_q14 = silk_smulww(gain_adj_q16, state.diff_q14);
            for v in state.s_lpc_q14[..NSQ_LPC_BUF_LENGTH].iter_mut() {
                *v = silk_smulww(gain_adj_q16, *v);
            }
            for v in state.s_ar2_q14.iter_mut() {
                *v = silk_smulww(gain_adj_q16, *v);
            }
        }
        // Every stored sample is scaled once. States that share a sample see
        // the same value libopus computes in each of their copies.
        for (pred, shape) in delay_lines
            .pred_q15
            .iter_mut()
            .zip(delay_lines.shape_q14.iter_mut())
        {
            for k in 0..n_states {
                pred[k] = silk_smulww(gain_adj_q16, pred[k]);
                shape[k] = silk_smulww(gain_adj_q16, shape[k]);
            }
        }
        nsq.prev_gain_q16 = gains_q16[subfr];
    }
}

/// Short-term prediction coefficients of one subframe, prepared once so that
/// the per-sample kernel does no conversion work.
struct ShortPredictionCoefs<'a> {
    /// The coefficients as given: `a_q12[j]` multiplies the sample `j` steps
    /// back. Its length is the prediction order.
    a_q12: &'a [i16],
    /// Entry `15 - j` holds `smulwb_operand(a_q12[j])`; the entries of taps at
    /// and above the prediction order are zero. This is the form the NEON
    /// kernel reads.
    #[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
    reversed_q27: [i32; MAX_LPC_ORDER],
}

impl<'a> ShortPredictionCoefs<'a> {
    /// `a_q12` must hold at least `order` coefficients and `order` must not
    /// exceed `MAX_LPC_ORDER`.
    #[inline]
    fn new(a_q12: &'a [i16], order: usize) -> Self {
        assert!(order <= MAX_LPC_ORDER);
        let a_q12 = &a_q12[..order];
        let mut reversed_q27 = [0i32; MAX_LPC_ORDER];
        for (dst, &a) in reversed_q27.iter_mut().rev().zip(a_q12) {
            *dst = smulwb_operand(a);
        }
        Self {
            a_q12,
            reversed_q27,
        }
    }

    /// The constant that the reference adds to the prediction: half the order.
    #[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
    #[inline(always)]
    fn bias(&self) -> i32 {
        (self.a_q12.len() as i32) >> 1
    }

    /// Prediction from the 16 newest samples; `window[15]` is the newest.
    /// The result has the unit of the samples shifted right by 4 (Q10 from Q14).
    #[inline(always)]
    fn predict(&self, window: &[i32; MAX_LPC_ORDER]) -> i32 {
        #[cfg(target_arch = "aarch64")]
        {
            short_prediction_neon(window, &self.reversed_q27, self.a_q12.len())
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            short_prediction_scalar(window, self.a_q12)
        }
    }
}

/// Portable short-term prediction: `order / 2 + sum((window[15 - j] * a_q12[j]) >> 16)`
/// with wrap-around additions, as libopus `silk_noise_shape_quantizer_short_prediction_c`.
/// The order is the length of `a_q12`.
#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
#[inline(always)]
fn short_prediction_scalar(window: &[i32; MAX_LPC_ORDER], a_q12: &[i16]) -> i32 {
    let mut out = (a_q12.len() as i32) >> 1;
    for (&s, &a) in window.iter().rev().zip(a_q12) {
        out = silk_smlawb(out, s, a as i32);
    }
    out
}

/// The 16 products of the NEON short-term prediction, added up to 4 partial
/// sums. The prediction is the sum of the 4 lanes plus half the order.
///
/// The coefficients are `smulwb_operand` values, so each `vqdmulhq_s32` is
/// `(x * a) >> 16` exactly. The additions wrap around, so their order does
/// not change the result.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
fn short_prediction_products_neon(
    window: &[i32; MAX_LPC_ORDER],
    reversed_q27: &[i32; MAX_LPC_ORDER],
) -> core::arch::aarch64::int32x4_t {
    use crate::simd_mem::neon::load_i32x4;
    use core::arch::aarch64::*;

    // The 4 values from position `i` on.
    let w = |i: usize| load_i32x4(window[i..i + 4].try_into().unwrap());
    let c = |i: usize| load_i32x4(reversed_q27[i..i + 4].try_into().unwrap());
    let t0 = vqdmulhq_s32(w(0), c(0));
    let t1 = vqdmulhq_s32(w(4), c(4));
    let t2 = vqdmulhq_s32(w(8), c(8));
    let t3 = vqdmulhq_s32(w(12), c(12));
    vaddq_s32(vaddq_s32(t0, t1), vaddq_s32(t2, t3))
}

/// NEON short-term prediction, as libopus
/// `silk_noise_shape_quantizer_short_prediction_neon`. Same result as
/// `short_prediction_scalar` for the coefficients `reversed_q27` was made from.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn short_prediction_neon(
    window: &[i32; MAX_LPC_ORDER],
    reversed_q27: &[i32; MAX_LPC_ORDER],
    order: usize,
) -> i32 {
    use core::arch::aarch64::*;

    // SAFETY: NEON is available: src/lib.rs refuses to build for aarch64
    // without it.
    let sum = unsafe { vaddvq_s32(short_prediction_products_neon(window, reversed_q27)) };
    ((order as i32) >> 1).wrapping_add(sum)
}

/// The shaping filter states of all delayed-decision states of one subframe.
type ShapingStates<const N: usize> = [[i32; MAX_SHAPE_LPC_ORDER]; N];

/// One `i32` per state, `N` at most 4, packed into 128 bits: state `k` is in
/// bits `32 * k` to `32 * k + 31`.
///
/// `warped_shaping_filter` takes and returns its per-state values in this
/// form. With arrays the compiler kept the sums of that function in vector
/// registers and moved them to integer registers and back for every filter
/// tap. Packed, the values stay in integer registers until the filter is
/// done.
#[derive(Copy, Clone)]
struct PerState<const N: usize>(u128);

impl<const N: usize> PerState<N> {
    #[inline(always)]
    fn pack(values: [i32; N]) -> Self {
        const { assert!(N <= 4) };
        let mut packed = 0u128;
        for (k, &v) in values.iter().enumerate() {
            packed |= (v as u32 as u128) << (32 * k);
        }
        Self(packed)
    }

    #[inline(always)]
    fn unpack(self) -> [i32; N] {
        core::array::from_fn(|k| (self.0 >> (32 * k)) as u32 as i32)
    }
}

/// Warped noise-shaping filter of one sample for `N` delayed-decision states.
///
/// Per state this is libopus's allpass chain. With `s` the filter state,
/// `s[-1]` the newest difference signal `diff` and `t[-1] = 0`, tap `m` is
/// `t[m] = s[m - 1] + ((s[m] - t[m - 1]) * warping) >> 16`. Tap `m` needs tap
/// `m - 1`, so one state alone is a chain of dependent steps. The states do
/// not depend on each other. Each tap is therefore computed for all states
/// before the next tap, which lets the processor work on the `N` chains at
/// the same time. The operations per state and their order are unchanged.
///
/// The new state is `s[m] = t[m]`. The return value is, per state,
/// `order / 2` plus the sum of `(t[m] * ar_shp_q13[m]) >> 16`.
///
/// State `k` reads its old filter state from `states[source[k]]` and writes
/// the new one to `states[k]`. `source[k]` is `k`, except for a state that
/// has just become a copy of another state: it names that state. This saves
/// copying the filter state.
///
/// The order is the length of `ar_shp_q13`. It must be even and at most
/// `MAX_SHAPE_LPC_ORDER`.
#[inline(always)]
fn warped_shaping_filter<const N: usize>(
    states: &mut ShapingStates<N>,
    source: [u8; N],
    diff_q14: PerState<N>,
    ar_shp_q13: &[i16],
    warping_q16: i32,
) -> PerState<N> {
    let order = ar_shp_q13.len();
    debug_assert!(order.is_multiple_of(2) && order <= MAX_SHAPE_LPC_ORDER);
    // Out-of-range values cannot occur; the clamp lets the compiler drop the
    // bounds checks in the loop.
    let source = source.map(|s| (s as usize).min(N - 1));
    let mut n_ar = [(order as i32) >> 1; N];
    let mut prev = diff_q14.unpack();
    let mut t = [0i32; N];
    // Two taps per pass: the compiler then needs no register moves to hand
    // `prev` and `t` to the next pass.
    let (coef_pairs, _) = ar_shp_q13.as_chunks::<2>();
    let coef_pairs = &coef_pairs[..coef_pairs.len().min(MAX_SHAPE_LPC_ORDER / 2)];

    for (p, coefs) in coef_pairs.iter().enumerate() {
        let c0 = coefs[0] as i32;
        let c1 = coefs[1] as i32;
        // Read the old values of all states first: a state that reads from
        // another state must see them before that state overwrites them.
        let mut a = [0i32; N];
        let mut b = [0i32; N];
        for k in 0..N {
            a[k] = states[source[k]][2 * p];
            b[k] = states[source[k]][2 * p + 1];
        }
        for k in 0..N {
            let t0 = silk_smlawb(prev[k], silk_sub32_ovflw(a[k], t[k]), warping_q16);
            let t1 = silk_smlawb(a[k], silk_sub32_ovflw(b[k], t0), warping_q16);
            states[k][2 * p] = t0;
            states[k][2 * p + 1] = t1;
            n_ar[k] = silk_smlawb(silk_smlawb(n_ar[k], t0, c0), t1, c1);
            prev[k] = b[k];
            t[k] = t1;
        }
    }
    PerState::pack(n_ar)
}

/// The filter states at the start of a subframe, and the `source` argument
/// of `warped_shaping_filter` that makes every state read its own.
#[inline(always)]
fn load_shaping_states<const N: usize>(states: &[PathState; N]) -> (ShapingStates<N>, [u8; N]) {
    (
        core::array::from_fn(|k| states[k].s_ar2_q14),
        core::array::from_fn(|k| k as u8),
    )
}

/// Writes the filter states back at the end of a subframe. A state that
/// became a copy of another state in the last sample takes that state's.
#[inline(always)]
fn store_shaping_states<const N: usize>(
    states: &mut [PathState; N],
    shaping: &ShapingStates<N>,
    source: [u8; N],
) {
    for k in 0..N {
        states[k].s_ar2_q14 = shaping[(source[k] as usize).min(N - 1)];
    }
}

/// Values that are the same for every sample of one subframe, as passed in.
struct SubframeParams<'a> {
    signal_type: i32,
    pulses_offset: i32,
    xq_ptr: i32,
    a_q12: &'a [i16],
    b_q14: &'a [i16],
    ar_shp_q13: &'a [i16],
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
    decision_delay: i32,
}

/// Values of one subframe that the level decision needs.
struct LevelParams {
    tilt_q14: i32,
    lf_shp_q14: i32,
    lambda_q10: i32,
    offset_q10: i32,
    /// `lambda_q10 / 2 - 512` when rate-distortion optimisation of the level
    /// is on (`lambda_q10 > 2048`), otherwise 0. With 0 the three-way choice
    /// of the level gives `q >> 10`, the value used when it is off.
    rdo_offset: i32,
}

/// What both sample loops derive from `SubframeParams` before the first sample.
struct SubframeSetup<'a> {
    x_q10: &'a [i32],
    ar_shp_q13: &'a [i16],
    lpc: ShortPredictionCoefs<'a>,
    levels: LevelParams,
    gain_q10: i32,
    pred_lag_ptr_base: i32,
    shp_lag_ptr_base: i32,
}

impl<'a> SubframeSetup<'a> {
    /// Checks the sizes the sample loops rely on. Returns `None` for an empty
    /// subframe, which changes nothing.
    fn new(nsq: &SilkNSQState, x_q10: &'a [i32], p: &SubframeParams<'a>) -> Option<Self> {
        // A longer subframe would run past `s_lpc_q14`.
        assert!(p.length >= 0 && p.length as usize <= MAX_SUB_FRAME_LENGTH);
        if p.length == 0 {
            return None;
        }
        // libopus requires an even shaping order too (`celt_assert`).
        assert!(p.shaping_lpc_order >= 2 && p.shaping_lpc_order % 2 == 0);
        assert!(p.shaping_lpc_order as usize <= MAX_SHAPE_LPC_ORDER);
        assert!(p.predict_lpc_order >= 0);
        Some(Self {
            x_q10: &x_q10[..p.length as usize],
            ar_shp_q13: &p.ar_shp_q13[..p.shaping_lpc_order as usize],
            lpc: ShortPredictionCoefs::new(p.a_q12, p.predict_lpc_order as usize),
            levels: LevelParams {
                tilt_q14: p.tilt_q14,
                lf_shp_q14: p.lf_shp_q14,
                lambda_q10: p.lambda_q10,
                offset_q10: p.offset_q10,
                rdo_offset: if p.lambda_q10 > 2048 {
                    p.lambda_q10 / 2 - 512
                } else {
                    0
                },
            },
            gain_q10: silk_rshift(p.gain_q16, 6),
            pred_lag_ptr_base: nsq.s_ltp_buf_idx - p.lag + LTP_ORDER as i32 / 2,
            shp_lag_ptr_base: nsq.s_ltp_shp_buf_idx - p.lag + HARM_SHAPE_FIR_TAPS as i32 / 2,
        })
    }
}

/// Long-term prediction and long-term shaping of sample `i`. Both are the
/// same for all states. Returns `(ltp_pred_q14, n_ltp_q14)`.
#[inline(always)]
fn long_term_terms(
    nsq: &SilkNSQState,
    s_ltp_q15: &[i32],
    p: &SubframeParams,
    setup: &SubframeSetup,
    i: i32,
) -> (i32, i32) {
    let mut ltp_pred_q14 = 0;
    if p.signal_type == TYPE_VOICED {
        let pred_lag_idx_calc = setup.pred_lag_ptr_base + i;
        if pred_lag_idx_calc >= LTP_ORDER as i32 && pred_lag_idx_calc < s_ltp_q15.len() as i32 {
            let pred_lag_idx = pred_lag_idx_calc as usize;
            let b_q14 = p.b_q14;
            ltp_pred_q14 = 2;
            ltp_pred_q14 = silk_smlawb(ltp_pred_q14, s_ltp_q15[pred_lag_idx], b_q14[0] as i32);
            ltp_pred_q14 = silk_smlawb(ltp_pred_q14, s_ltp_q15[pred_lag_idx - 1], b_q14[1] as i32);
            ltp_pred_q14 = silk_smlawb(ltp_pred_q14, s_ltp_q15[pred_lag_idx - 2], b_q14[2] as i32);
            ltp_pred_q14 = silk_smlawb(ltp_pred_q14, s_ltp_q15[pred_lag_idx - 3], b_q14[3] as i32);
            ltp_pred_q14 = silk_smlawb(ltp_pred_q14, s_ltp_q15[pred_lag_idx - 4], b_q14[4] as i32);
            ltp_pred_q14 = silk_lshift(ltp_pred_q14, 1);
        }
    }

    let mut n_ltp_q14 = 0;
    if p.lag > 0 {
        let shp_lag_idx_calc = setup.shp_lag_ptr_base + i;
        if shp_lag_idx_calc >= HARM_SHAPE_FIR_TAPS as i32
            && shp_lag_idx_calc < nsq.s_ltp_shp_q14.len() as i32
        {
            let shp_lag_idx = shp_lag_idx_calc as usize;
            n_ltp_q14 = silk_smulwb(
                silk_add_sat32(
                    nsq.s_ltp_shp_q14[shp_lag_idx],
                    nsq.s_ltp_shp_q14[shp_lag_idx - 2],
                ),
                p.harm_shape_fir_packed_q14,
            );
            n_ltp_q14 = silk_smlawt(
                n_ltp_q14,
                nsq.s_ltp_shp_q14[shp_lag_idx - 1],
                p.harm_shape_fir_packed_q14,
            );
            n_ltp_q14 = silk_sub_lshift32(ltp_pred_q14, n_ltp_q14, 2);
        }
    }
    (ltp_pred_q14, n_ltp_q14)
}

/// Quantizes one sample of one state. Returns the two candidates, the cheaper
/// one first.
///
/// The choices of the levels depend on the data in a way the processor cannot
/// predict, so they are written as selections and not as branches.
#[inline(always)]
fn quantize_sample(
    lv: &LevelParams,
    x_q10: i32,
    ltp_pred_q14: i32,
    n_ltp_q14: i32,
    n_ar: i32,
    lpc_pred_q14: i32,
    lf_ar_q14: i32,
    shape_q14: i32,
    seed: i32,
    rd_q10: i32,
) -> NSQSamplePair {
    let mut n_ar_q14 = silk_lshift(n_ar, 1);
    n_ar_q14 = silk_smlawb(n_ar_q14, lf_ar_q14, lv.tilt_q14);
    n_ar_q14 = silk_lshift(n_ar_q14, 2);

    let mut n_lf_q14 = silk_smulwb(shape_q14, lv.lf_shp_q14);
    n_lf_q14 = silk_smlawt(n_lf_q14, lf_ar_q14, lv.lf_shp_q14);
    n_lf_q14 = silk_lshift(n_lf_q14, 2);

    let tmp1 = silk_sub_sat32(
        silk_add32_ovflw(n_ltp_q14, lpc_pred_q14),
        silk_add_sat32(n_ar_q14, n_lf_q14),
    );
    let r_q10 = x_q10 - silk_rshift_round(tmp1, 4);

    // The dither flips the sign of the residual and of the excitation.
    let negative = seed < 0;
    let r_q10 = select(negative, r_q10.wrapping_neg(), r_q10);
    let r_q10 = silk_limit_32(r_q10, -(31 << 10), 30 << 10);

    let offset_q10 = lv.offset_q10;
    let lambda_q10 = lv.lambda_q10;
    let rdo_offset = lv.rdo_offset;
    let q1_q10_in = r_q10 - offset_q10;
    let q1_q0 = select(
        q1_q10_in > rdo_offset,
        silk_rshift(q1_q10_in.wrapping_sub(rdo_offset), 10),
        select(
            q1_q10_in < -rdo_offset,
            silk_rshift(q1_q10_in.wrapping_add(rdo_offset), 10),
            q1_q10_in >> 31,
        ),
    );
    // Level `q1_q0` at Q10 with the offset, before the adjustment that pulls
    // every level towards zero.
    let base_q10 = silk_lshift(q1_q0, 10).wrapping_add(offset_q10);
    let q1_q10 = select(
        q1_q0 > 0,
        base_q10.wrapping_sub(QUANT_LEVEL_ADJUST_Q10),
        select(
            q1_q0 == 0,
            base_q10,
            base_q10.wrapping_add(QUANT_LEVEL_ADJUST_Q10),
        ),
    );
    // The two levels next to zero are closer together.
    let near_zero = q1_q0 == 0 || q1_q0 == -1;
    let q2_q10 = q1_q10.wrapping_add(select(near_zero, 1024 - QUANT_LEVEL_ADJUST_Q10, 1024));
    let rd1_q10 = silk_smulbb(
        select(q1_q0 >= 0, q1_q10, q1_q10.wrapping_neg()),
        lambda_q10,
    );
    let rd2_q10 = silk_smulbb(
        select(q1_q0 >= -1, q2_q10, q2_q10.wrapping_neg()),
        lambda_q10,
    );

    let rr_q10 = r_q10 - q1_q10;
    let rd1_q10 = silk_rshift(silk_smlabb(rd1_q10, rr_q10, rr_q10), 10);
    let rr_q10 = r_q10 - q2_q10;
    let rd2_q10 = silk_rshift(silk_smlabb(rd2_q10, rr_q10, rr_q10), 10);

    let first_is_cheaper = rd1_q10 < rd2_q10;
    let q = [
        select(first_is_cheaper, q1_q10, q2_q10),
        select(first_is_cheaper, q2_q10, q1_q10),
    ];
    let rd = [
        rd_q10 + select(first_is_cheaper, rd1_q10, rd2_q10),
        rd_q10 + select(first_is_cheaper, rd2_q10, rd1_q10),
    ];

    let x_q14 = silk_lshift(x_q10, 4);
    core::array::from_fn(|j| {
        let exc_q14 = silk_lshift(q[j], 4);
        let exc_q14 = select(negative, exc_q14.wrapping_neg(), exc_q14);
        let lpc_exc_q14 = silk_add32(exc_q14, ltp_pred_q14);
        let xq_q14 = silk_add32_ovflw(lpc_exc_q14, lpc_pred_q14);
        let diff_q14 = silk_sub32_ovflw(xq_q14, x_q14);
        let s_lf_ar_shp_q14 = silk_sub32_ovflw(diff_q14, n_ar_q14);
        NSQSampleStruct {
            q_q10: q[j],
            rd_q10: rd[j],
            xq_q14,
            lf_ar_q14: s_lf_ar_shp_q14,
            diff_q14,
            s_ltp_shp_q14: silk_sub_sat32(s_lf_ar_shp_q14, n_lf_q14),
            lpc_exc_q14,
        }
    })
}

/// Result of comparing the candidates of all states for one sample.
struct Survivors {
    /// State with the lowest cost. The delayed outputs are taken from it.
    winner_ind: usize,
    /// State whose first candidate is the most expensive.
    rd_max_ind: usize,
    /// State whose second candidate is the cheapest, and that cost.
    rd_min_ind: usize,
    rd_min_q10: i32,
    /// True when state `rd_max_ind` continues as a copy of state `rd_min_ind`
    /// with that state's second candidate.
    replace: bool,
}

impl Survivors {
    /// True when the history of state `rd_min_ind` has to become the history
    /// of another state.
    #[inline(always)]
    fn copies_history(&self) -> bool {
        self.replace && self.rd_min_ind != self.rd_max_ind
    }

    /// The `source` argument of `warped_shaping_filter` for the next sample.
    #[inline(always)]
    fn shaping_source<const N: usize>(&self) -> [u8; N] {
        core::array::from_fn(|k| {
            select(self.replace && k == self.rd_max_ind, self.rd_min_ind, k) as u8
        })
    }
}

/// Chooses the winner and the state to replace, as the reference does.
///
/// `rd0[k]` and `rd1[k]` are the costs of the first and second candidate of
/// state `k`. `rand_state[k]` is the dither state of state `k` at the sample
/// that is written out now. A state whose value differs from the winner's is
/// on another path than the output: its costs are raised in place so that it
/// is replaced soon. Ties keep the state with the lowest index, because the
/// reference compares with `<` and `>` while it walks up the states.
#[inline(always)]
fn select_survivors<const N: usize>(
    rd0: &mut [i32; N],
    rd1: &mut [i32; N],
    rand_state: &[i32; N],
) -> Survivors {
    let mut winner_ind = 0;
    let mut rd_winner_q10 = rd0[0];
    let mut winner_rand_state = rand_state[0];
    for k in 1..N {
        let lower = rd0[k] < rd_winner_q10;
        rd_winner_q10 = select(lower, rd0[k], rd_winner_q10);
        winner_ind = select(lower, k, winner_ind);
        winner_rand_state = select(lower, rand_state[k], winner_rand_state);
    }

    for k in 0..N {
        let penalty = select(rand_state[k] != winner_rand_state, i32::MAX >> 4, 0);
        rd0[k] = rd0[k].saturating_add(penalty);
        rd1[k] = rd1[k].saturating_add(penalty);
    }

    let mut rd_max_q10 = rd0[0];
    let mut rd_max_ind = 0;
    let mut rd_min_q10 = rd1[0];
    let mut rd_min_ind = 0;
    for k in 1..N {
        let higher = rd0[k] > rd_max_q10;
        rd_max_q10 = select(higher, rd0[k], rd_max_q10);
        rd_max_ind = select(higher, k, rd_max_ind);
        let lower = rd1[k] < rd_min_q10;
        rd_min_q10 = select(lower, rd1[k], rd_min_q10);
        rd_min_ind = select(lower, k, rd_min_ind);
    }

    Survivors {
        winner_ind,
        rd_max_ind,
        rd_min_ind,
        rd_min_q10,
        replace: rd_min_q10 < rd_max_q10,
    }
}

/// Makes the stored history of state `sv.rd_max_ind` a copy of that of state
/// `sv.rd_min_ind` when `sv` asks for it.
///
/// Whether a state is replaced changes from sample to sample in a way the
/// processor cannot predict, and a wrong guess costs more than the copy. The
/// copy is therefore always done: when no state is replaced it goes to a
/// spare row and to `spare_lpc`, which nothing reads.
///
/// Called while sample `idx` of a subframe is processed, before the states
/// are extended by that sample.
/// - Delay lines: the state takes over the columns of the other state.
/// - `s_lpc_q14`: later predictions and the hand-over to the next subframe
///   read entries `idx + 1` and above. Entries `idx + 16` and above are
///   written, for every state, before they are read. Copying the 16 entries
///   from `idx` on is therefore enough.
/// - `lf_ar_q14`, `diff_q14` and `rd_q10` get new values for every state
///   right after the copy, the callers handle `seed` and `seed_init`
///   themselves, and the shaping filter state is handled by
///   `warped_shaping_filter`. These are not copied here.
#[inline(always)]
fn copy_history<const N: usize>(
    states: &mut [PathState; N],
    delay_lines: &mut DelayLines,
    spare_lpc: &mut [i32; NSQ_LPC_BUF_LENGTH],
    sv: &Survivors,
    idx: usize,
) {
    let copies = sv.copies_history();
    let row = select(copies, sv.rd_max_ind, NSQ_MAX_STATES_OPERATING);
    delay_lines.column[row] = delay_lines.column[sv.rd_min_ind];

    let window: [i32; NSQ_LPC_BUF_LENGTH] = states[sv.rd_min_ind].s_lpc_q14
        [idx..idx + NSQ_LPC_BUF_LENGTH]
        .try_into()
        .unwrap();
    let dst: &mut [i32; NSQ_LPC_BUF_LENGTH] = (&mut states[sv.rd_max_ind].s_lpc_q14
        [idx..idx + NSQ_LPC_BUF_LENGTH])
        .try_into()
        .unwrap();
    *select(copies, dst, spare_lpc) = window;
}

/// Writes the outputs that are `decision_delay` samples old, taken from the
/// state `winner`, which currently has the lowest cost.
#[inline(always)]
fn write_delayed_outputs(
    nsq: &mut SilkNSQState,
    delay_lines: &DelayLines,
    winner: usize,
    pulses: &mut [i8],
    s_ltp_q15: &mut [i32],
    delayed_ga_q10: &[i32],
    p: &SubframeParams,
    i: i32,
    last_smple_idx: usize,
) {
    let pulse_idx = (p.pulses_offset + i - p.decision_delay) as isize;
    let xq_idx = (p.xq_ptr + i - p.decision_delay) as isize;
    let shp_idx = (nsq.s_ltp_shp_buf_idx - p.decision_delay) as isize;
    let ltp_idx = (nsq.s_ltp_buf_idx - p.decision_delay) as isize;
    let column = delay_lines.column(winner, last_smple_idx);

    if pulse_idx >= 0 && pulse_idx < pulses.len() as isize {
        pulses[pulse_idx as usize] =
            silk_rshift_round(delay_lines.q_q10[last_smple_idx][column], 10) as i8;
    }
    if xq_idx >= 0 && xq_idx < nsq.xq.len() as isize {
        nsq.xq[xq_idx as usize] = silk_sat16(silk_rshift_round(
            silk_smulww(
                delay_lines.xq_q14[last_smple_idx][column],
                delayed_ga_q10[last_smple_idx],
            ),
            8,
        )) as i16;
    }
    if shp_idx >= 0 && shp_idx < nsq.s_ltp_shp_q14.len() as isize {
        nsq.s_ltp_shp_q14[shp_idx as usize] = delay_lines.shape_q14[last_smple_idx][column];
    }
    if ltp_idx >= 0 && ltp_idx < s_ltp_q15.len() as isize {
        s_ltp_q15[ltp_idx as usize] = delay_lines.pred_q15[last_smple_idx][column];
    }
}

/// One subframe of the delayed-decision quantizer for exactly `N` states,
/// portable version.
///
/// `N` is a compile-time constant so that the loops over the states have a
/// fixed length.
fn quantize_subframe<const N: usize>(
    nsq: &mut SilkNSQState,
    states: &mut [PathState; N],
    delay_lines: &mut DelayLines,
    x_q10: &[i32],
    pulses: &mut [i8],
    s_ltp_q15: &mut [i32],
    delayed_ga_q10: &mut [i32],
    smpl_buf_idx: &mut i32,
    p: &SubframeParams,
) {
    let Some(setup) = SubframeSetup::new(nsq, x_q10, p) else {
        return;
    };
    let decision_delay = p.decision_delay;
    let (mut shaping, mut shaping_src) = load_shaping_states(states);
    let mut spare_lpc = [0i32; NSQ_LPC_BUF_LENGTH];

    for (idx, &x) in setup.x_q10.iter().enumerate() {
        let i = idx as i32;
        let (ltp_pred_q14, n_ltp_q14) = long_term_terms(nsq, s_ltp_q15, p, &setup, i);

        // Dither and short-term prediction of every state.
        let mut lpc_pred_q14 = [0i32; N];
        let mut diff_q14 = [0i32; N];
        for k in 0..N {
            let state = &mut states[k];
            state.seed = silk_rand(state.seed);
            let window: &[i32; NSQ_LPC_BUF_LENGTH] = state.s_lpc_q14[idx..idx + NSQ_LPC_BUF_LENGTH]
                .try_into()
                .unwrap();
            lpc_pred_q14[k] = silk_lshift(setup.lpc.predict(window), 4);
            diff_q14[k] = state.diff_q14;
        }

        // Noise shaping feedback of every state.
        let n_ar = warped_shaping_filter(
            &mut shaping,
            shaping_src,
            PerState::pack(diff_q14),
            setup.ar_shp_q13,
            p.warping_q16,
        )
        .unpack();

        // The two quantization candidates of every state, the cheaper one
        // first. Their costs are kept apart because the selection of the
        // survivors changes them.
        let smpl_idx = (*smpl_buf_idx as usize).min(DECISION_DELAY - 1);
        let mut cand = [[NSQSampleStruct::default(); 2]; N];
        let mut rd0 = [0i32; N];
        let mut rd1 = [0i32; N];
        for k in 0..N {
            let state = &states[k];
            cand[k] = quantize_sample(
                &setup.levels,
                x,
                ltp_pred_q14,
                n_ltp_q14,
                n_ar[k],
                lpc_pred_q14[k],
                state.lf_ar_q14,
                delay_lines.shape_q14[smpl_idx][delay_lines.column(k, smpl_idx)],
                state.seed,
                state.rd_q10,
            );
            rd0[k] = cand[k][0].rd_q10;
            rd1[k] = cand[k][1].rd_q10;
        }

        *smpl_buf_idx = (*smpl_buf_idx - 1 + DECISION_DELAY as i32) % DECISION_DELAY as i32;
        let last_smple_idx = ((*smpl_buf_idx + decision_delay) % DECISION_DELAY as i32) as usize;

        let rand_state: [i32; N] = core::array::from_fn(|k| {
            delay_lines.rand_state[last_smple_idx][delay_lines.column(k, last_smple_idx)]
        });
        let sv = select_survivors(&mut rd0, &mut rd1, &rand_state);
        shaping_src = sv.shaping_source();

        // Replace the worst state by a copy of the best one. Its candidate
        // is exchanged below.
        copy_history(states, delay_lines, &mut spare_lpc, &sv, idx);
        let seed_from = select(sv.replace, sv.rd_min_ind, sv.rd_max_ind);
        states[sv.rd_max_ind].seed = states[seed_from].seed;
        states[sv.rd_max_ind].seed_init = states[seed_from].seed_init;

        if p.subfr > 0 || i >= decision_delay {
            write_delayed_outputs(
                nsq,
                delay_lines,
                sv.winner_ind,
                pulses,
                s_ltp_q15,
                delayed_ga_q10,
                p,
                i,
                last_smple_idx,
            );
        }
        nsq.s_ltp_shp_buf_idx += 1;
        nsq.s_ltp_buf_idx += 1;

        // Append the chosen candidate to every state.
        let best_second = cand[sv.rd_min_ind][1];
        let smpl_idx = (*smpl_buf_idx as usize).min(DECISION_DELAY - 1);
        for k in 0..N {
            let replaced = sv.replace && k == sv.rd_max_ind;
            let c = if replaced { best_second } else { cand[k][0] };
            let state = &mut states[k];
            state.lf_ar_q14 = c.lf_ar_q14;
            state.diff_q14 = c.diff_q14;
            state.s_lpc_q14[NSQ_LPC_BUF_LENGTH + idx] = c.xq_q14;
            state.seed = silk_add32_ovflw(state.seed, silk_rshift_round(c.q_q10, 10));
            state.rd_q10 = if replaced { sv.rd_min_q10 } else { rd0[k] };
            delay_lines.xq_q14[smpl_idx][k] = c.xq_q14;
            delay_lines.q_q10[smpl_idx][k] = c.q_q10;
            delay_lines.pred_q15[smpl_idx][k] = silk_lshift(c.lpc_exc_q14, 1);
            delay_lines.shape_q14[smpl_idx][k] = c.s_ltp_shp_q14;
            delay_lines.rand_state[smpl_idx][k] = state.seed;
            delay_lines.column[k][smpl_idx] = k as u8;
        }
        delayed_ga_q10[smpl_idx] = setup.gain_q10;
    }

    store_shaping_states(states, &shaping, shaping_src);
    // The newest samples become the prediction history of the next subframe.
    let length = setup.x_q10.len();
    for state in states.iter_mut() {
        state
            .s_lpc_q14
            .copy_within(length..length + NSQ_LPC_BUF_LENGTH, 0);
    }
}

/// What the long-term terms of `long_term_terms` need, prepared once per
/// subframe so that the sample loop computes them without branches.
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
struct LongTerm {
    /// The five LTP coefficients.
    b_q14: [i32; LTP_ORDER],
    /// The subframe is voiced: it has long-term prediction.
    voiced: bool,
    /// Index into `s_ltp_q15` of the newest LTP tap of sample 0.
    pred_base: i32,
    /// The subframe has a lag: it has harmonic noise shaping.
    has_lag: bool,
    /// Index into `s_ltp_shp_q14` of the newest shaping tap of sample 0.
    shp_base: i32,
    harm_shape_fir_packed_q14: i32,
}

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
impl LongTerm {
    fn new(p: &SubframeParams, setup: &SubframeSetup) -> Self {
        let mut b_q14 = [0i32; LTP_ORDER];
        let voiced = p.signal_type == TYPE_VOICED;
        if voiced {
            for (dst, &b) in b_q14.iter_mut().zip(&p.b_q14[..LTP_ORDER]) {
                *dst = b as i32;
            }
        }
        Self {
            b_q14,
            voiced,
            pred_base: setup.pred_lag_ptr_base,
            has_lag: p.lag > 0,
            shp_base: setup.shp_lag_ptr_base,
            harm_shape_fir_packed_q14: p.harm_shape_fir_packed_q14,
        }
    }

    /// `long_term_terms` of sample `i`, without branches. Both buffers
    /// must be longer than `LTP_ORDER`. An index out of range gives a
    /// zero term, as in `long_term_terms`; the values are then read at
    /// index `LTP_ORDER` and not used.
    #[inline(always)]
    fn at(&self, s_ltp_q15: &[i32], s_ltp_shp_q14: &[i32], i: i32) -> (i32, i32) {
        debug_assert!(s_ltp_q15.len() > LTP_ORDER && s_ltp_shp_q14.len() > LTP_ORDER);
        let pred_idx = self.pred_base.wrapping_add(i);
        let pred_ok =
            self.voiced & (pred_idx >= LTP_ORDER as i32) & (pred_idx < s_ltp_q15.len() as i32);
        let j = select(pred_ok, pred_idx as usize, LTP_ORDER);
        // SAFETY: with `pred_ok`, `LTP_ORDER <= j < s_ltp_q15.len()`;
        // otherwise `j == LTP_ORDER < s_ltp_q15.len()`. The 5 entries
        // up to `j` are inside the slice.
        let taps = unsafe {
            &*s_ltp_q15
                .as_ptr()
                .add(j - (LTP_ORDER - 1))
                .cast::<[i32; LTP_ORDER]>()
        };
        let mut ltp_pred_q14 = 2;
        for (&s, &b) in taps.iter().rev().zip(&self.b_q14) {
            ltp_pred_q14 = silk_smlawb(ltp_pred_q14, s, b);
        }
        let ltp_pred_q14 = select(pred_ok, silk_lshift(ltp_pred_q14, 1), 0);

        let shp_idx = self.shp_base.wrapping_add(i);
        let shp_ok = self.has_lag
            & (shp_idx >= HARM_SHAPE_FIR_TAPS as i32)
            & (shp_idx < s_ltp_shp_q14.len() as i32);
        let j = select(shp_ok, shp_idx as usize, HARM_SHAPE_FIR_TAPS);
        // SAFETY: as above, with `HARM_SHAPE_FIR_TAPS < LTP_ORDER`.
        let shp = unsafe {
            &*s_ltp_shp_q14
                .as_ptr()
                .add(j - (HARM_SHAPE_FIR_TAPS - 1))
                .cast::<[i32; HARM_SHAPE_FIR_TAPS]>()
        };
        let harm = self.harm_shape_fir_packed_q14;
        let n_ltp_q14 = silk_smulwb(silk_add_sat32(shp[2], shp[0]), harm);
        let n_ltp_q14 = silk_smlawt(n_ltp_q14, shp[1], harm);
        let n_ltp_q14 = select(shp_ok, silk_sub_lshift32(ltp_pred_q14, n_ltp_q14, 2), 0);
        (ltp_pred_q14, n_ltp_q14)
    }
}

/// Whether the vector sample loops (`neon::quantize_subframe_neon` and
/// `x86::quantize_subframe_x86`) can run a subframe: they rely on these
/// ranges. The encoder stays inside them; other values take the portable
/// loop, which handles them as the code before did.
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
#[inline(always)]
fn vector_loop_applies(
    smpl_buf_idx: i32,
    s_ltp_q15: &[i32],
    delayed_ga_q10: &[i32],
    p: &SubframeParams,
) -> bool {
    (0..DECISION_DELAY as i32).contains(&smpl_buf_idx)
        && (0..=DECISION_DELAY as i32).contains(&p.decision_delay)
        && s_ltp_q15.len() > LTP_ORDER
        && delayed_ga_q10.len() >= DECISION_DELAY
        && (p.signal_type != TYPE_VOICED || p.b_q14.len() >= LTP_ORDER)
}

/// Stores `value` at `buf[idx]` when `ok` is true and `idx` is inside
/// `buf`; otherwise stores it in `spare`, which nothing reads. A branch
/// here would split the sample loop into pieces that the compiler orders
/// separately.
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
#[inline(always)]
fn store_if<T: Copy>(buf: &mut [T], idx: isize, ok: bool, spare: &mut T, value: T) {
    let in_range = ok & (idx >= 0) & ((idx as usize) < buf.len());
    let dst = select(
        in_range,
        buf.as_mut_ptr().wrapping_offset(idx),
        spare as *mut T,
    );
    // SAFETY: with `in_range`, `dst` is element `idx` of `buf`;
    // otherwise it is `spare`.
    unsafe { dst.write(value) };
}

/// The sample loop with the values of all states in the lanes of NEON
/// registers: lane `k` belongs to state `k`.
#[cfg(target_arch = "aarch64")]
mod neon {
    use super::*;
    // `load` and `store` read and write one row of four `i32` values.
    use crate::simd_mem::neon::{load_i32x4 as load, load_u8x16, load_u32x4, store_i32x4 as store};
    use core::arch::aarch64::*;

    /// `LANE_MASK[k]` has all bits of lane `k` set. Entry 4 selects no lane.
    static LANE_MASK: [[u32; 4]; 5] = [
        [!0, 0, 0, 0],
        [0, !0, 0, 0],
        [0, 0, !0, 0],
        [0, 0, 0, !0],
        [0, 0, 0, 0],
    ];

    /// `BROADCAST[k]` is the byte table for `vqtbl1q_u8` that copies lane `k`
    /// into all four lanes.
    static BROADCAST: [[u8; 16]; 4] = [
        [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3],
        [4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7],
        [8, 9, 10, 11, 8, 9, 10, 11, 8, 9, 10, 11, 8, 9, 10, 11],
        [
            12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15,
        ],
    ];

    /// The first `N` lanes of `v`.
    #[target_feature(enable = "neon")]
    #[inline]
    fn to_lanes<const N: usize>(v: int32x4_t) -> [i32; N] {
        let mut out = [0i32; N];
        if N > 0 {
            out[0] = vgetq_lane_s32::<0>(v);
        }
        if N > 1 {
            out[1] = vgetq_lane_s32::<1>(v);
        }
        if N > 2 {
            out[2] = vgetq_lane_s32::<2>(v);
        }
        if N > 3 {
            out[3] = vgetq_lane_s32::<3>(v);
        }
        out
    }

    /// A vector with `a[k]` in lane `k`. Lanes from `N` on are zero.
    #[target_feature(enable = "neon")]
    #[inline]
    fn from_lanes<const N: usize>(a: [i32; N]) -> int32x4_t {
        from_packed(PerState::pack(a))
    }

    /// A vector with state `k` of `p` in lane `k`. Lanes from `N` on are zero.
    #[target_feature(enable = "neon")]
    #[inline]
    fn from_packed<const N: usize>(p: PerState<N>) -> int32x4_t {
        vcombine_s32(vcreate_s32(p.0 as u64), vcreate_s32((p.0 >> 64) as u64))
    }

    /// `-v` in the lanes where `negative` is all ones, `v` elsewhere.
    #[target_feature(enable = "neon")]
    #[inline]
    fn negate_if(v: int32x4_t, negative: int32x4_t) -> int32x4_t {
        vsubq_s32(veorq_s32(v, negative), negative)
    }

    /// `(a as i16 as i32) * (b as i16 as i32)` per lane, as `silk_smulbb`.
    #[target_feature(enable = "neon")]
    #[inline]
    fn smulbb(a: int32x4_t, b: int16x4_t) -> int32x4_t {
        vmull_s16(vmovn_s32(a), b)
    }

    /// `acc + (a as i16 as i32) * (a as i16 as i32)` per lane, as
    /// `silk_smlabb(acc, a, a)`.
    #[target_feature(enable = "neon")]
    #[inline]
    fn add_square(acc: int32x4_t, a: int32x4_t) -> int32x4_t {
        let a = vmovn_s32(a);
        vmlal_s16(acc, a, a)
    }

    /// The constant `c` for which `vqdmulhq_s32(x, c)` is `silk_smulwb(x, b)`:
    /// `(x * (b as i16)) >> 16`.
    #[target_feature(enable = "neon")]
    #[inline]
    fn smulwb_factor(b: i32) -> int32x4_t {
        vdupq_n_s32(smulwb_operand(b as i16))
    }

    /// The constant `c` for which `vqdmulhq_s32(x, c)` is `silk_smulwt(x, b)`:
    /// `(x * (b >> 16)) >> 16`.
    #[target_feature(enable = "neon")]
    #[inline]
    fn smulwt_factor(b: i32) -> int32x4_t {
        vdupq_n_s32(smulwb_operand((b >> 16) as i16))
    }

    /// The two candidates of every state for one sample, the cheaper one
    /// first. The costs are returned as the amounts to add to the cost so far.
    struct Candidates {
        q_q10: [int32x4_t; 2],
        rd_added_q10: [int32x4_t; 2],
        xq_q14: [int32x4_t; 2],
        lf_ar_q14: [int32x4_t; 2],
        diff_q14: [int32x4_t; 2],
        s_ltp_shp_q14: [int32x4_t; 2],
        lpc_exc_q14: [int32x4_t; 2],
    }

    /// Constants of one subframe for `quantize_sample_neon`.
    struct LevelVectors {
        tilt: int32x4_t,
        lf_shp_low: int32x4_t,
        lf_shp_high: int32x4_t,
        lambda: int16x4_t,
        offset_q10: int32x4_t,
        rdo_offset: int32x4_t,
        minus_rdo_offset: int32x4_t,
    }

    impl LevelVectors {
        #[target_feature(enable = "neon")]
        #[inline]
        fn new(lv: &LevelParams) -> Self {
            Self {
                tilt: smulwb_factor(lv.tilt_q14),
                lf_shp_low: smulwb_factor(lv.lf_shp_q14),
                lf_shp_high: smulwt_factor(lv.lf_shp_q14),
                lambda: vdup_n_s16(lv.lambda_q10 as i16),
                offset_q10: vdupq_n_s32(lv.offset_q10),
                rdo_offset: vdupq_n_s32(lv.rdo_offset),
                minus_rdo_offset: vdupq_n_s32(-lv.rdo_offset),
            }
        }
    }

    /// `quantize_sample` for four states at once. Every step is the same
    /// operation as in the scalar function, done per lane.
    #[target_feature(enable = "neon")]
    #[inline]
    fn quantize_sample_neon(
        lv: &LevelVectors,
        x_q10: i32,
        ltp_pred_q14: i32,
        n_ltp_q14: i32,
        n_ar: int32x4_t,
        lpc_pred_q14: int32x4_t,
        lf_ar_q14: int32x4_t,
        shape_q14: int32x4_t,
        seed: int32x4_t,
    ) -> Candidates {
        let n_ar_q14 = vshlq_n_s32::<2>(vaddq_s32(
            vshlq_n_s32::<1>(n_ar),
            vqdmulhq_s32(lf_ar_q14, lv.tilt),
        ));
        let n_lf_q14 = vshlq_n_s32::<2>(vaddq_s32(
            vqdmulhq_s32(shape_q14, lv.lf_shp_low),
            vqdmulhq_s32(lf_ar_q14, lv.lf_shp_high),
        ));

        let tmp1 = vqsubq_s32(
            vaddq_s32(vdupq_n_s32(n_ltp_q14), lpc_pred_q14),
            vqaddq_s32(n_ar_q14, n_lf_q14),
        );
        // A rounding shift is `silk_rshift_round`.
        let r_q10 = vsubq_s32(vdupq_n_s32(x_q10), vrshrq_n_s32::<4>(tmp1));

        let negative = vreinterpretq_s32_u32(vcltzq_s32(seed));
        let r_q10 = negate_if(r_q10, negative);
        let r_q10 = vminq_s32(
            vmaxq_s32(r_q10, vdupq_n_s32(-(31 << 10))),
            vdupq_n_s32(30 << 10),
        );

        let q1_q10_in = vsubq_s32(r_q10, lv.offset_q10);
        let q1_q0 = vbslq_s32(
            vcgtq_s32(q1_q10_in, lv.rdo_offset),
            vshrq_n_s32::<10>(vsubq_s32(q1_q10_in, lv.rdo_offset)),
            vbslq_s32(
                vcltq_s32(q1_q10_in, lv.minus_rdo_offset),
                vshrq_n_s32::<10>(vaddq_s32(q1_q10_in, lv.rdo_offset)),
                vshrq_n_s32::<31>(q1_q10_in),
            ),
        );
        let base_q10 = vaddq_s32(vshlq_n_s32::<10>(q1_q0), lv.offset_q10);
        let adjust = vdupq_n_s32(QUANT_LEVEL_ADJUST_Q10);
        let is_zero = vceqzq_s32(q1_q0);
        let q1_q10 = vbslq_s32(
            vcgtzq_s32(q1_q0),
            vsubq_s32(base_q10, adjust),
            vbslq_s32(is_zero, base_q10, vaddq_s32(base_q10, adjust)),
        );
        let near_zero = vorrq_u32(is_zero, vceqq_s32(q1_q0, vdupq_n_s32(-1)));
        let q2_q10 = vaddq_s32(
            q1_q10,
            vbslq_s32(
                near_zero,
                vdupq_n_s32(1024 - QUANT_LEVEL_ADJUST_Q10),
                vdupq_n_s32(1024),
            ),
        );
        // The rate term uses the magnitude of the level: negate q1 when
        // q1_q0 < 0 and q2 when q1_q0 < -1.
        let q1_negative = vreinterpretq_s32_u32(vcltzq_s32(q1_q0));
        let q2_negative = vreinterpretq_s32_u32(vcltq_s32(q1_q0, vdupq_n_s32(-1)));
        let rd1_q10 = smulbb(negate_if(q1_q10, q1_negative), lv.lambda);
        let rd2_q10 = smulbb(negate_if(q2_q10, q2_negative), lv.lambda);
        let rd1_q10 = vshrq_n_s32::<10>(add_square(rd1_q10, vsubq_s32(r_q10, q1_q10)));
        let rd2_q10 = vshrq_n_s32::<10>(add_square(rd2_q10, vsubq_s32(r_q10, q2_q10)));

        let first_is_cheaper = vcltq_s32(rd1_q10, rd2_q10);
        let q = [
            vbslq_s32(first_is_cheaper, q1_q10, q2_q10),
            vbslq_s32(first_is_cheaper, q2_q10, q1_q10),
        ];

        let x_q14 = vdupq_n_s32(silk_lshift(x_q10, 4));
        let ltp_pred_q14 = vdupq_n_s32(ltp_pred_q14);
        let lpc_exc = q.map(|q| vaddq_s32(negate_if(vshlq_n_s32::<4>(q), negative), ltp_pred_q14));
        let xq = lpc_exc.map(|e| vaddq_s32(e, lpc_pred_q14));
        let diff = xq.map(|xq| vsubq_s32(xq, x_q14));
        let lf_ar = diff.map(|d| vsubq_s32(d, n_ar_q14));
        Candidates {
            q_q10: q,
            rd_added_q10: [vminq_s32(rd1_q10, rd2_q10), vmaxq_s32(rd1_q10, rd2_q10)],
            xq_q14: xq,
            lf_ar_q14: lf_ar,
            diff_q14: diff,
            s_ltp_shp_q14: lf_ar.map(|l| vqsubq_s32(l, n_lf_q14)),
            lpc_exc_q14: lpc_exc,
        }
    }

    /// The candidates of `quantize_sample_neon` as arrays, for the tests.
    #[cfg(test)]
    pub(super) struct CandidateLanes {
        pub(super) q_q10: [i32; 4],
        /// The amount added to the cost so far.
        pub(super) rd_q10: [i32; 4],
        pub(super) xq_q14: [i32; 4],
        pub(super) lf_ar_q14: [i32; 4],
        pub(super) diff_q14: [i32; 4],
        pub(super) s_ltp_shp_q14: [i32; 4],
        pub(super) lpc_exc_q14: [i32; 4],
    }

    /// `quantize_sample_neon` on arrays, for the tests.
    #[cfg(test)]
    pub(super) fn level_decision_lanes(
        lv: &LevelParams,
        x_q10: i32,
        ltp_pred_q14: i32,
        n_ltp_q14: i32,
        n_ar: [i32; 4],
        lpc_pred_q14: [i32; 4],
        lf_ar_q14: [i32; 4],
        shape_q14: [i32; 4],
        seed: [i32; 4],
    ) -> [CandidateLanes; 2] {
        // SAFETY: NEON is available: src/lib.rs refuses to build for aarch64
        // without it.
        unsafe {
            let c = quantize_sample_neon(
                &LevelVectors::new(lv),
                x_q10,
                ltp_pred_q14,
                n_ltp_q14,
                load(&n_ar),
                load(&lpc_pred_q14),
                load(&lf_ar_q14),
                load(&shape_q14),
                load(&seed),
            );
            core::array::from_fn(|j| CandidateLanes {
                q_q10: to_lanes(c.q_q10[j]),
                rd_q10: to_lanes(c.rd_added_q10[j]),
                xq_q14: to_lanes(c.xq_q14[j]),
                lf_ar_q14: to_lanes(c.lf_ar_q14[j]),
                diff_q14: to_lanes(c.diff_q14[j]),
                s_ltp_shp_q14: to_lanes(c.s_ltp_shp_q14[j]),
                lpc_exc_q14: to_lanes(c.lpc_exc_q14[j]),
            })
        }
    }

    /// The shaping filter states of all delayed-decision states, one row per
    /// tap: lane `k` of row `m` is tap `m` of state `k`.
    pub(super) type ShapingRows = [[i32; 4]; MAX_SHAPE_LPC_ORDER];

    /// The byte table for `vqtbl1q_u8` that keeps every lane in place.
    static IDENTITY: [u8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];

    /// Constants of the shaping filter of one subframe for
    /// `shaping_filter_rows`.
    pub(super) struct ShapingCoefs {
        /// `smulwb_operand(warping_q16 as i16)` in every lane, so that
        /// `vqdmulhq_s32` gives `silk_smulwb(x, warping_q16)`.
        warping: int32x4_t,
        /// Lane `m % 4` of `coef[m / 4]` is `smulwb_operand(ar_shp_q13[m])`.
        /// The entries from the order on are zero.
        coef: [[i32; 4]; MAX_SHAPE_LPC_ORDER / 4],
        /// `order / 2` in every lane: the start of the feedback sum.
        bias: int32x4_t,
        /// The shaping order: the length of `ar_shp_q13`.
        order: usize,
        /// See `coef_vectors`.
        keep_lanes: bool,
    }

    impl ShapingCoefs {
        #[target_feature(enable = "neon")]
        #[inline]
        pub(super) fn new(ar_shp_q13: &[i16], warping_q16: i32, keep_lanes: bool) -> Self {
            let order = ar_shp_q13.len().min(MAX_SHAPE_LPC_ORDER);
            let mut coef = [[0i32; 4]; MAX_SHAPE_LPC_ORDER / 4];
            for (dst, &c) in coef.as_flattened_mut().iter_mut().zip(&ar_shp_q13[..order]) {
                *dst = smulwb_operand(c);
            }
            Self {
                warping: smulwb_factor(warping_q16),
                coef,
                bias: vdupq_n_s32((order as i32) >> 1),
                order,
                keep_lanes,
            }
        }

        /// The coefficient vectors for the filter taps of `later_taps` and
        /// `shaping_filter_both`, which multiply by one lane of them.
        ///
        /// The compiler turns each such lane into a vector of its own and
        /// makes these 16 to 24 vectors once, before the sample loop. They
        /// do not fit in the registers, so every sample reloads each of
        /// them from the stack. With `keep_lanes`, an empty `asm!`
        /// statement, after which the compiler must assume that the 4-lane
        /// vectors have changed, stops that: the multiplications then read
        /// the lanes of at most 6 vectors.
        #[target_feature(enable = "neon")]
        #[inline]
        fn coef_vectors<const ORDER: usize>(&self) -> [int32x4_t; MAX_SHAPE_LPC_ORDER / 4] {
            let mut coef: [int32x4_t; MAX_SHAPE_LPC_ORDER / 4] =
                core::array::from_fn(|j| load(&self.coef[j]));
            if self.keep_lanes {
                for v in coef.iter_mut().take(ORDER.div_ceil(4)) {
                    // SAFETY: the statement is empty. It reads and writes
                    // only the register that holds `v`, and no memory.
                    unsafe {
                        core::arch::asm!(
                            "/* {0:v} */",
                            inout(vreg) * v,
                            options(nomem, nostack, preserves_flags)
                        )
                    };
                }
            }
            coef
        }
    }

    /// Whether `ShapingCoefs::coef_vectors` keeps the lanes, for `n` states.
    ///
    /// Measured with the quantizer alone, 3 pairs: on the Cortex-A55 2.3 to
    /// 4.0 % less time for 2, 3 and 4 states. On the Cortex-A76, 4 states
    /// -0.6 %, 2 states -0.3 to -2.0 %, but 3 states +0.7 to +1.0 %. On
    /// Apple M4, 2 states +1.4 to +1.7 %. So it is used for 2 and 4 states,
    /// and not on Apple cores.
    const fn keep_coef_lanes(n: usize) -> bool {
        !cfg!(target_vendor = "apple") && n != 3
    }

    /// The rows of `shaping_filter_rows` for the filter states of `states`.
    /// Lanes from `N` on are zero.
    pub(super) fn load_shaping_rows<const N: usize>(states: &[PathState; N]) -> ShapingRows {
        let mut rows = [[0i32; 4]; MAX_SHAPE_LPC_ORDER];
        for (m, row) in rows.iter_mut().enumerate() {
            for k in 0..N {
                row[k] = states[k].s_ar2_q14[m];
            }
        }
        rows
    }

    /// Writes the rows back to the filter states of `states`. `perm` is the
    /// pending lane table of `shaping_filter_rows`: state `k` takes the lane
    /// that `perm` moves to lane `k`, in every row, also in the rows from the
    /// shaping order on. This is what the scalar loop does with its `source`.
    #[target_feature(enable = "neon")]
    pub(super) fn store_shaping_rows<const N: usize>(
        states: &mut [PathState; N],
        rows: &ShapingRows,
        perm: uint8x16_t,
    ) {
        for (m, row) in rows.iter().enumerate() {
            let row: [i32; N] = to_lanes(permute(load(row), perm));
            for k in 0..N {
                states[k].s_ar2_q14[m] = row[k];
            }
        }
    }

    /// The lanes of `v` rearranged by the byte table `perm`.
    #[target_feature(enable = "neon")]
    #[inline]
    fn permute(v: int32x4_t, perm: uint8x16_t) -> int32x4_t {
        vreinterpretq_s32_u8(vqtbl1q_u8(vreinterpretq_u8_s32(v), perm))
    }

    /// The two candidates of every state for the last sample, and which
    /// state continues as a copy of which. This is what the shaping filter
    /// of the next sample needs from the last one.
    #[derive(Copy, Clone)]
    pub(super) struct LastChoice {
        /// `diff_q14` of the first candidate of every state.
        pub(super) diff_first: int32x4_t,
        /// `diff_q14` of the second candidate of every state.
        pub(super) diff_second: int32x4_t,
        /// All bits set in the lane of the state that continues as a copy
        /// of the best state, with that state's second candidate. No bit is
        /// set when no state is replaced.
        pub(super) replaced: uint32x4_t,
        /// The byte table for `vqtbl1q_u8` that copies the lane of the best
        /// state into all lanes.
        pub(super) from_best: uint8x16_t,
        /// The byte table that moves the lane of the best state to the
        /// replaced lane and keeps the other lanes in place.
        pub(super) perm: uint8x16_t,
    }

    impl LastChoice {
        /// For the first sample of a subframe: every state keeps its own
        /// `diff_q14` and filter state.
        #[target_feature(enable = "neon")]
        #[inline]
        pub(super) fn keep(diff_q14: int32x4_t) -> Self {
            let identity = load_u8x16(&IDENTITY);
            Self {
                diff_first: diff_q14,
                diff_second: diff_q14,
                replaced: vdupq_n_u32(0),
                from_best: identity,
                perm: identity,
            }
        }

        /// The chosen value of every state from the values `first` and
        /// `second` of the two candidates of every state.
        #[target_feature(enable = "neon")]
        #[inline]
        pub(super) fn take(&self, first: int32x4_t, second: int32x4_t) -> int32x4_t {
            vbslq_s32(self.replaced, permute(second, self.from_best), first)
        }
    }

    /// How many taps of the shaping filter `shaping_filter_both` computes
    /// for both candidates of every state, before the choice of the
    /// survivors of the last sample is known.
    ///
    /// On the wide out-of-order cores of Apple the sample loop waits for its
    /// longest chain: the shaping filter, the level decision, the choice of
    /// the survivors, and again the filter. The first taps then run at the
    /// same time as the choice. Measured on Apple M4 with 5 taps: 3 states
    /// 9 to 12 % less time, 4 states 7 to 8 %, 2 states 3 to 8 %. On the
    /// Cortex-A55 and A76 the loop does not wait for that chain (removing
    /// the filter from it saves 0 to 3 %), and the extra work made it 1 to
    /// 6 % slower; there the filter waits for the choice (0 taps).
    pub(super) const FIRST_TAPS: usize = if cfg!(target_vendor = "apple") { 5 } else { 0 };

    /// Warped noise-shaping filter of one sample for all states at once,
    /// state `k` in lane `k`. Same results as `warped_shaping_filter`.
    ///
    /// Tap `m` of a state is
    /// `t[m] = s[m - 1] + ((s[m] - t[m - 1]) * warping) >> 16`, with `s` the
    /// old filter state, `s[-1] = diff` and `t[-1] = 0`. The new state is
    /// `t`. The return value is `order / 2` plus the sum of
    /// `(t[m] * ar_shp_q13[m]) >> 16`, with wrap-around additions, so the
    /// order of the additions does not change it.
    ///
    /// Each row is read through the byte table `perm` first. It moves the
    /// lane of a state that has just become a copy of another state to the
    /// lane of that other state, so the copy costs nothing else.
    ///
    /// `ORDER` is the shaping order when it is known at compile time; the
    /// loop then has no branches. With `ORDER == 0` the order is `c.order`.
    #[target_feature(enable = "neon")]
    #[inline]
    pub(super) fn shaping_filter_rows<const ORDER: usize>(
        rows: &mut ShapingRows,
        perm: uint8x16_t,
        diff_q14: int32x4_t,
        c: &ShapingCoefs,
    ) -> int32x4_t {
        debug_assert!(ORDER == 0 || ORDER == c.order);
        let zero = vdupq_n_s32(0);
        if ORDER == 0 {
            let order = c.order.min(MAX_SHAPE_LPC_ORDER);
            let mut prev = diff_q14;
            let mut t = zero;
            // Two partial sums, one for the even and one for the odd taps,
            // so that their additions do not wait for each other.
            let mut even = c.bias;
            let mut odd = zero;
            for m in 0..order {
                let old = permute(load(&rows[m]), perm);
                t = vaddq_s32(prev, vqdmulhq_s32(vsubq_s32(old, t), c.warping));
                store(&mut rows[m], t);
                let coef = vdupq_n_s32(c.coef.as_flattened()[m]);
                even = vaddq_s32(even, vqdmulhq_s32(t, coef));
                prev = old;
                core::mem::swap(&mut even, &mut odd);
            }
            // After an odd number of taps the sums are swapped; the sum of
            // both does not depend on that.
            return vaddq_s32(even, odd);
        }
        let coef = c.coef_vectors::<ORDER>();
        later_taps::<ORDER>(rows, perm, c, &coef, 0, diff_q14, zero, c.bias, zero)
    }

    /// Taps `start` to `ORDER - 1` of `shaping_filter_rows`, for an order
    /// known at compile time. `prev` is `s[start - 1]` read through `perm`,
    /// `t` is `t[start - 1]`, and `even` and `odd` are the partial sums of
    /// the even and the odd taps before `start`.
    ///
    /// Every tap is written out, so the whole filter is one block of code
    /// without branches.
    #[target_feature(enable = "neon")]
    #[inline]
    fn later_taps<const ORDER: usize>(
        rows: &mut ShapingRows,
        perm: uint8x16_t,
        c: &ShapingCoefs,
        coef: &[int32x4_t; MAX_SHAPE_LPC_ORDER / 4],
        start: usize,
        mut prev: int32x4_t,
        mut t: int32x4_t,
        mut even: int32x4_t,
        mut odd: int32x4_t,
    ) -> int32x4_t {
        macro_rules! tap {
            ($m:expr, $sum:ident, $lane:literal) => {
                if $m >= start && $m < ORDER && $m < MAX_SHAPE_LPC_ORDER {
                    let old = permute(load(&rows[$m]), perm);
                    t = vaddq_s32(prev, vqdmulhq_s32(vsubq_s32(old, t), c.warping));
                    store(&mut rows[$m], t);
                    let product = vqdmulhq_laneq_s32::<$lane>(t, coef[$m / 4]);
                    $sum = vaddq_s32($sum, product);
                    prev = old;
                }
            };
        }
        tap!(0, even, 0);
        tap!(1, odd, 1);
        tap!(2, even, 2);
        tap!(3, odd, 3);
        tap!(4, even, 0);
        tap!(5, odd, 1);
        tap!(6, even, 2);
        tap!(7, odd, 3);
        tap!(8, even, 0);
        tap!(9, odd, 1);
        tap!(10, even, 2);
        tap!(11, odd, 3);
        tap!(12, even, 0);
        tap!(13, odd, 1);
        tap!(14, even, 2);
        tap!(15, odd, 3);
        tap!(16, even, 0);
        tap!(17, odd, 1);
        tap!(18, even, 2);
        tap!(19, odd, 3);
        tap!(20, even, 0);
        tap!(21, odd, 1);
        tap!(22, even, 2);
        tap!(23, odd, 3);
        let _ = prev;
        vaddq_s32(even, odd)
    }

    /// `shaping_filter_rows` for the next sample, started before the choice
    /// of the survivors of the last sample is used. `last` holds both
    /// candidates of every state and that choice.
    ///
    /// The first `FIRST` taps are computed twice, from the first and from
    /// the second candidate of every state, each on the state's own rows. Then the replaced lane takes the second result of the best
    /// state, the other lanes their first result. Only then are the other
    /// taps computed, as in `shaping_filter_rows`, with `last.perm`.
    ///
    /// With `ORDER == 0` or `FIRST == 0` this is `shaping_filter_rows` on
    /// the chosen `diff_q14`. The sample loop uses `FIRST_TAPS` for `FIRST`.
    #[target_feature(enable = "neon")]
    #[inline]
    pub(super) fn shaping_filter_both<const ORDER: usize, const FIRST: usize>(
        rows: &mut ShapingRows,
        last: &LastChoice,
        c: &ShapingCoefs,
    ) -> int32x4_t {
        let first_taps = FIRST.min(ORDER).min(MAX_SHAPE_LPC_ORDER);
        if first_taps == 0 {
            let diff_q14 = last.take(last.diff_first, last.diff_second);
            return shaping_filter_rows::<ORDER>(rows, last.perm, diff_q14, c);
        }
        debug_assert!(ORDER == c.order);
        let zero = vdupq_n_s32(0);
        let coef = c.coef_vectors::<ORDER>();
        let mut even = c.bias;
        let mut odd = zero;
        // `t` of the two candidates, and the chosen one.
        let mut first = zero;
        let mut second = zero;
        let mut t = zero;
        // `s[m - 1]` of the states' own rows; `s[-1]` is `diff`.
        let mut own_prev = zero;
        macro_rules! first_tap {
            ($m:expr, $sum:ident, $lane:literal) => {
                if $m < first_taps {
                    let old = load(&rows[$m]);
                    let (prev_first, prev_second) = if $m == 0 {
                        (last.diff_first, last.diff_second)
                    } else {
                        (own_prev, own_prev)
                    };
                    first = vaddq_s32(prev_first, vqdmulhq_s32(vsubq_s32(old, first), c.warping));
                    second =
                        vaddq_s32(prev_second, vqdmulhq_s32(vsubq_s32(old, second), c.warping));
                    t = last.take(first, second);
                    store(&mut rows[$m], t);
                    let product = vqdmulhq_laneq_s32::<$lane>(t, coef[$m / 4]);
                    $sum = vaddq_s32($sum, product);
                    own_prev = old;
                }
            };
        }
        const { assert!(FIRST <= 8) };
        first_tap!(0, even, 0);
        first_tap!(1, odd, 1);
        first_tap!(2, even, 2);
        first_tap!(3, odd, 3);
        first_tap!(4, even, 0);
        first_tap!(5, odd, 1);
        first_tap!(6, even, 2);
        first_tap!(7, odd, 3);
        let prev = permute(own_prev, last.perm);
        later_taps::<ORDER>(rows, last.perm, c, &coef, first_taps, prev, t, even, odd)
    }

    /// `shaping_filter_rows` on arrays, for the tests. `source` is the
    /// `source` argument of `warped_shaping_filter`; at most one state may
    /// read another state's filter state. With `both`, the
    /// function tested is `shaping_filter_both::<ORDER, FIRST>`.
    /// `keep_lanes` is the setting of `ShapingCoefs::coef_vectors`.
    #[cfg(test)]
    pub(super) fn shaping_filter_rows_lanes<
        const N: usize,
        const ORDER: usize,
        const FIRST: usize,
    >(
        states: &mut ShapingStates<N>,
        source: [u8; N],
        diff_q14: [i32; N],
        ar_shp_q13: &[i16],
        warping_q16: i32,
        both: bool,
        keep_lanes: bool,
    ) -> [i32; N] {
        // SAFETY: NEON is available: src/lib.rs refuses to build for aarch64
        // without it.
        unsafe {
            let c = ShapingCoefs::new(ar_shp_q13, warping_q16, keep_lanes);
            let mut rows = [[0i32; 4]; MAX_SHAPE_LPC_ORDER];
            for (m, row) in rows.iter_mut().enumerate() {
                for k in 0..N {
                    row[k] = states[k][m];
                }
            }
            // At most one state reads another state's filter state: that is
            // the replaced state, and the state it reads is the best one.
            let replaced = (0..N).find(|&k| source[k] as usize != k);
            assert!((0..N).all(|k| Some(k) == replaced || source[k] as usize == k));
            let mut last = LastChoice::keep(from_lanes(diff_q14));
            // The second candidates of the other states must not matter.
            let mut second: [i32; 4] = core::array::from_fn(|k| (k as i32 + 1) * 0x0123_4567);
            if let Some(dst) = replaced {
                let best = source[dst] as usize;
                second[best] = diff_q14[dst];
                let mut mask = [0u32; 4];
                mask[dst] = !0;
                last.replaced = load_u32x4(&mask);
                last.from_best = load_u8x16(&BROADCAST[best]);
                let mut table = IDENTITY;
                table[4 * dst..4 * dst + 4].copy_from_slice(&BROADCAST[best][..4]);
                last.perm = load_u8x16(&table);
            }
            last.diff_second = load(&second);
            let n_ar = if both {
                shaping_filter_both::<ORDER, FIRST>(&mut rows, &last, &c)
            } else {
                shaping_filter_rows::<ORDER>(&mut rows, last.perm, from_lanes(diff_q14), &c)
            };
            // Only the taps below the order change.
            for (m, row) in rows.iter().enumerate().take(c.order) {
                for k in 0..N {
                    states[k][m] = row[k];
                }
            }
            to_lanes(n_ar)
        }
    }

    /// Short-term prediction of all states at once, state `k` in lane `k`,
    /// with the newest sample of every state taken from a register instead
    /// of from memory.
    struct NewestFromRegister {
        /// The coefficients of window positions 0 to 3, 4 to 7, 8 to 11 and
        /// 11 to 14. Position 11 is in two of these loads; its coefficient is
        /// in the third vector only.
        without_newest: [int32x4_t; 4],
        /// The coefficient of position 15, the newest sample, in every lane.
        newest: int32x4_t,
        /// Half the order, the constant of the sum, in every lane.
        bias: int32x4_t,
    }

    impl NewestFromRegister {
        #[target_feature(enable = "neon")]
        #[inline]
        fn new(coefs: &ShortPredictionCoefs) -> Self {
            let r = &coefs.reversed_q27;
            let last = [0, r[12], r[13], r[14]];
            Self {
                without_newest: [
                    load(r[0..4].try_into().unwrap()),
                    load(r[4..8].try_into().unwrap()),
                    load(r[8..12].try_into().unwrap()),
                    load(&last),
                ],
                newest: vdupq_n_s32(r[15]),
                bias: vdupq_n_s32(coefs.bias()),
            }
        }

        /// The prediction of every state, shifted to Q14 as the quantizer
        /// uses it. `windows[k]` holds the 16 newest samples of state `k`;
        /// its last entry is not read, lane `k` of `newest` holds it instead.
        #[target_feature(enable = "neon")]
        #[inline]
        fn predict<const N: usize>(
            &self,
            windows: [&[i32; NSQ_LPC_BUF_LENGTH]; N],
            newest: int32x4_t,
        ) -> int32x4_t {
            let zero = vdupq_n_s32(0);
            let mut partial = [zero; 4];
            for k in 0..N {
                let w = windows[k];
                let (head, _) = w.as_chunks::<4>();
                let last: &[i32; 4] = w[11..15].try_into().unwrap();
                let c = &self.without_newest;
                let t0 = vqdmulhq_s32(load(&head[0]), c[0]);
                let t1 = vqdmulhq_s32(load(&head[1]), c[1]);
                let t2 = vqdmulhq_s32(load(&head[2]), c[2]);
                let t3 = vqdmulhq_s32(load(last), c[3]);
                partial[k] = vaddq_s32(vaddq_s32(t0, t1), vaddq_s32(t2, t3));
            }
            let sum = vpaddq_s32(
                vpaddq_s32(partial[0], partial[1]),
                vpaddq_s32(partial[2], partial[3]),
            );
            let sum = vaddq_s32(sum, vaddq_s32(self.bias, vqdmulhq_s32(newest, self.newest)));
            vshlq_n_s32::<4>(sum)
        }
    }

    /// One subframe of the delayed-decision quantizer for `N` states, 2 to 4.
    /// Same results as `quantize_subframe`.
    ///
    /// All states are computed at once, state `k` in lane `k` of NEON
    /// registers, the shaping filter included. Lanes from `N` on are computed
    /// but never used. The values that go from one sample to the next stay in
    /// registers.
    ///
    /// The loop over the samples has no branches besides its own: the
    /// conditions on the long-term terms and on the delayed outputs are
    /// selections, and the bounds are checked once per subframe. The compiler
    /// orders instructions only within a block between two branches. A core
    /// that issues instructions in program order, such as the Cortex-A55,
    /// runs a long block with fewer stalls than many short ones.
    ///
    /// `ORDER` is the shaping order when it is known at compile time, or 0.
    #[target_feature(enable = "neon")]
    pub(super) fn quantize_subframe_neon<const N: usize, const ORDER: usize>(
        nsq: &mut SilkNSQState,
        states: &mut [PathState; N],
        delay_lines: &mut DelayLines,
        x_q10: &[i32],
        pulses: &mut [i8],
        s_ltp_q15: &mut [i32],
        delayed_ga_q10: &mut [i32],
        smpl_buf_idx: &mut i32,
        p: &SubframeParams,
    ) {
        debug_assert!((2..=4).contains(&N));
        let dd = p.decision_delay;
        if !vector_loop_applies(*smpl_buf_idx, s_ltp_q15, delayed_ga_q10, p) {
            quantize_subframe::<N>(
                nsq,
                states,
                delay_lines,
                x_q10,
                pulses,
                s_ltp_q15,
                delayed_ga_q10,
                smpl_buf_idx,
                p,
            );
            return;
        }
        let Some(setup) = SubframeSetup::new(nsq, x_q10, p) else {
            return;
        };
        assert!(ORDER == 0 || ORDER == setup.ar_shp_q13.len());
        let delayed_ga_q10: &mut [i32; DECISION_DELAY] =
            (&mut delayed_ga_q10[..DECISION_DELAY]).try_into().unwrap();
        let length = setup.x_q10.len().min(MAX_SUB_FRAME_LENGTH);
        let levels = LevelVectors::new(&setup.levels);
        let long_term = LongTerm::new(p, &setup);
        let lpc_bias = vdupq_n_s32(setup.lpc.bias());
        let lpc_newest = NewestFromRegister::new(&setup.lpc);
        // Measured on Apple M4: with 2 states the loop takes 8 % less time
        // when the newest sample of the short-term prediction comes from a
        // register; with 3 or 4 states it takes more. On the Cortex-A55 and
        // A76 the difference is within 1 %.
        let newest_from_register = N == 2;
        let mut newest_xq_q14 = from_lanes::<N>(core::array::from_fn(|k| {
            states[k].s_lpc_q14[NSQ_LPC_BUF_LENGTH - 1]
        }));
        let rand_multiplier = vdupq_n_s32(196314165);
        let rand_increment = vdupq_n_s32(907633515);
        let shp_buf_start = nsq.s_ltp_shp_buf_idx;
        let ltp_buf_start = nsq.s_ltp_buf_idx;

        // The shaping filter states, one row per tap, and the byte table
        // that tells the filter which state each state continues.
        let shaping_coefs = ShapingCoefs::new(setup.ar_shp_q13, p.warping_q16, keep_coef_lanes(N));
        let mut shaping_rows = load_shaping_rows(states);
        let identity = load_u8x16(&IDENTITY);
        let mut perm = identity;

        // A store that has no place to go goes to one of these. The compiler
        // must not see that nothing reads them: it would turn each such
        // store into a branch around the store.
        let mut spare_lpc = [0i32; NSQ_LPC_BUF_LENGTH];
        let (mut spare_i8, mut spare_i16, mut spare_i32) = (0i8, 0i16, 0i32);
        core::hint::black_box((
            &mut spare_lpc,
            &mut spare_i8,
            &mut spare_i16,
            &mut spare_i32,
        ));

        // The per-state values that change with every sample.
        let mut seed = from_lanes::<N>(core::array::from_fn(|k| states[k].seed));
        let mut seed_init = from_lanes::<N>(core::array::from_fn(|k| states[k].seed_init));
        let mut lf_ar_q14 = from_lanes::<N>(core::array::from_fn(|k| states[k].lf_ar_q14));
        let mut diff_q14 = from_lanes::<N>(core::array::from_fn(|k| states[k].diff_q14));
        // What `shaping_filter_both` needs from the last sample.
        let mut last = LastChoice::keep(diff_q14);
        let mut rd_q10: [i32; N] = core::array::from_fn(|k| states[k].rd_q10);
        // The slot of the newest sample of the delay lines, and its
        // `shape_q14` row.
        let mut newest = (*smpl_buf_idx as usize).min(DECISION_DELAY - 1);
        // The newest slot holds each state's sample in its own column.
        debug_assert!((0..N).all(|k| delay_lines.column(k, newest) == k));
        let mut shape_q14 = load(&delay_lines.shape_q14[newest]);

        for idx in 0..length {
            let i = idx as i32;
            let x = setup.x_q10[idx];

            // Noise shaping feedback of every state. It is the long chain of
            // the sample and only needs `diff_q14`. Everything else up to the
            // level decision does not depend on it.
            // `!= 0`, not `> 0`: where `FIRST_TAPS` is 0, clippy denies a
            // comparison that is always false (`absurd_extreme_comparisons`).
            let n_ar = if FIRST_TAPS != 0 {
                shaping_filter_both::<ORDER, FIRST_TAPS>(&mut shaping_rows, &last, &shaping_coefs)
            } else {
                shaping_filter_rows::<ORDER>(&mut shaping_rows, perm, diff_q14, &shaping_coefs)
            };

            let (ltp_pred_q14, n_ltp_q14) = long_term.at(s_ltp_q15, &nsq.s_ltp_shp_q14, i);

            // Dither: `silk_rand`.
            seed = vmlaq_s32(rand_increment, seed, rand_multiplier);

            // Short-term prediction.
            let windows: [&[i32; NSQ_LPC_BUF_LENGTH]; N] = core::array::from_fn(|k| {
                states[k].s_lpc_q14[idx..idx + NSQ_LPC_BUF_LENGTH]
                    .try_into()
                    .unwrap()
            });
            let lpc_pred_q14 = if newest_from_register {
                lpc_newest.predict::<N>(windows, newest_xq_q14)
            } else {
                // 4 partial sums per state, then the sums of all states side
                // by side. Written here and not as a function: as a function,
                // on Apple M4 the compiler rotated the 4-state loop so that
                // its end test fell in the middle of the body, and the loop
                // took 13 % more time.
                let mut partial = [vdupq_n_s32(0); 4];
                for k in 0..N {
                    partial[k] =
                        short_prediction_products_neon(windows[k], &setup.lpc.reversed_q27);
                }
                let lpc_pred = vpaddq_s32(
                    vpaddq_s32(partial[0], partial[1]),
                    vpaddq_s32(partial[2], partial[3]),
                );
                vshlq_n_s32::<4>(vaddq_s32(lpc_pred, lpc_bias))
            };

            let c = quantize_sample_neon(
                &levels,
                x,
                ltp_pred_q14,
                n_ltp_q14,
                n_ar,
                lpc_pred_q14,
                lf_ar_q14,
                shape_q14,
                seed,
            );

            // The slot of this sample, and the slot of the sample that is
            // `dd` samples older.
            newest = select(newest == 0, DECISION_DELAY - 1, newest.wrapping_sub(1))
                .min(DECISION_DELAY - 1);
            let oldest = newest + dd as usize;
            let oldest = select(
                oldest >= DECISION_DELAY,
                oldest.wrapping_sub(DECISION_DELAY),
                oldest,
            )
            .min(DECISION_DELAY - 1);

            // The survivors are chosen on scalars, as in the portable loop.
            let added0: [i32; N] = to_lanes(c.rd_added_q10[0]);
            let added1: [i32; N] = to_lanes(c.rd_added_q10[1]);
            let mut rd0: [i32; N] = core::array::from_fn(|k| rd_q10[k].wrapping_add(added0[k]));
            let mut rd1: [i32; N] = core::array::from_fn(|k| rd_q10[k].wrapping_add(added1[k]));
            let rand_state: [i32; N] = core::array::from_fn(|k| {
                delay_lines.rand_state[oldest][delay_lines.column(k, oldest)]
            });
            let sv = select_survivors(&mut rd0, &mut rd1, &rand_state);

            // Replace the worst state by a copy of the best one that takes
            // the second candidate. `replaced` marks the lane of the worst
            // state and `from_best` reads the lane of the best state.
            copy_history(states, delay_lines, &mut spare_lpc, &sv, idx);
            let replaced_lane = select(sv.replace, sv.rd_max_ind, 4);
            let (replaced, from_best) = (
                load_u32x4(&LANE_MASK[replaced_lane]),
                load_u8x16(&BROADCAST[sv.rd_min_ind]),
            );
            let take = |first: int32x4_t, second: int32x4_t| {
                let second = vqtbl1q_u8(vreinterpretq_u8_s32(second), from_best);
                vbslq_s32(replaced, vreinterpretq_s32_u8(second), first)
            };
            seed = take(seed, seed);
            seed_init = take(seed_init, seed_init);
            let q_q10 = take(c.q_q10[0], c.q_q10[1]);
            let xq_q14 = take(c.xq_q14[0], c.xq_q14[1]);
            newest_xq_q14 = xq_q14;
            let lpc_exc_q14 = take(c.lpc_exc_q14[0], c.lpc_exc_q14[1]);
            shape_q14 = take(c.s_ltp_shp_q14[0], c.s_ltp_shp_q14[1]);
            lf_ar_q14 = take(c.lf_ar_q14[0], c.lf_ar_q14[1]);
            diff_q14 = take(c.diff_q14[0], c.diff_q14[1]);
            // In the next sample the replaced state reads the filter state
            // of the best state.
            perm = vbslq_u8(vreinterpretq_u8_u32(replaced), from_best, identity);
            last = LastChoice {
                diff_first: c.diff_q14[0],
                diff_second: c.diff_q14[1],
                replaced,
                from_best,
                perm,
            };
            for k in 0..N {
                let replaced = sv.replace && k == sv.rd_max_ind;
                rd_q10[k] = select(replaced, sv.rd_min_q10, rd0[k]);
            }

            // The outputs that are `dd` samples old, from the state with the
            // lowest cost. The first `dd` samples of a frame have none.
            let active = p.subfr > 0 || i >= dd;
            let column = delay_lines.column(sv.winner_ind, oldest);
            store_if(
                pulses,
                (p.pulses_offset + i - dd) as isize,
                active,
                &mut spare_i8,
                silk_rshift_round(delay_lines.q_q10[oldest][column], 10) as i8,
            );
            store_if(
                &mut nsq.xq,
                (p.xq_ptr + i - dd) as isize,
                active,
                &mut spare_i16,
                silk_sat16(silk_rshift_round(
                    silk_smulww(delay_lines.xq_q14[oldest][column], delayed_ga_q10[oldest]),
                    8,
                )) as i16,
            );
            store_if(
                &mut nsq.s_ltp_shp_q14,
                (shp_buf_start + i - dd) as isize,
                active,
                &mut spare_i32,
                delay_lines.shape_q14[oldest][column],
            );
            store_if(
                s_ltp_q15,
                (ltp_buf_start + i - dd) as isize,
                active,
                &mut spare_i32,
                delay_lines.pred_q15[oldest][column],
            );

            // Append the chosen candidate to every state: one row per delay
            // line. The lanes from `N` on of a row are never read.
            seed = vaddq_s32(seed, vrshrq_n_s32::<10>(q_q10));
            store(&mut delay_lines.xq_q14[newest], xq_q14);
            store(&mut delay_lines.q_q10[newest], q_q10);
            store(
                &mut delay_lines.pred_q15[newest],
                vshlq_n_s32::<1>(lpc_exc_q14),
            );
            store(&mut delay_lines.shape_q14[newest], shape_q14);
            store(&mut delay_lines.rand_state[newest], seed);
            let xq: [i32; N] = to_lanes(xq_q14);
            for k in 0..N {
                states[k].s_lpc_q14[NSQ_LPC_BUF_LENGTH + idx] = xq[k];
                delay_lines.column[k][newest] = k as u8;
            }
            delayed_ga_q10[newest] = setup.gain_q10;
        }
        *smpl_buf_idx = newest as i32;
        nsq.s_ltp_shp_buf_idx = shp_buf_start + length as i32;
        nsq.s_ltp_buf_idx = ltp_buf_start + length as i32;

        store_shaping_rows(states, &shaping_rows, perm);
        let seed: [i32; N] = to_lanes(seed);
        let seed_init: [i32; N] = to_lanes(seed_init);
        let lf_ar_q14: [i32; N] = to_lanes(lf_ar_q14);
        let diff_q14: [i32; N] = to_lanes(diff_q14);
        for k in 0..N {
            let state = &mut states[k];
            state.seed = seed[k];
            state.seed_init = seed_init[k];
            state.lf_ar_q14 = lf_ar_q14[k];
            state.diff_q14 = diff_q14[k];
            state.rd_q10 = rd_q10[k];
            // The newest samples become the prediction history of the next
            // subframe.
            state
                .s_lpc_q14
                .copy_within(length..length + NSQ_LPC_BUF_LENGTH, 0);
        }
    }
}

/// The sample loop with the values of all states in the lanes of SSE
/// registers, for x86-64: lane `k` belongs to state `k`. It follows
/// `mod neon` step by step. The comments there explain the steps; the
/// comments here say where and why this module differs.
///
/// The loop is compiled twice: `quantize_subframe_sse41` for processors with
/// SSE4.1, and `quantize_subframe_avx2` for processors with AVX2. The AVX2
/// copy runs the shaping filter of 3 and 4 states in 256-bit registers
/// (`wide`) and the short-term prediction with 256-bit loads.
///
/// x86 has no instruction like NEON's `vqdmulhq_s32`, which multiplies the
/// 32-bit values of 4 lanes by 16-bit factors and keeps bits 16 to 47 of each
/// product (`silk_smulwb`). `_mm_mul_epi32` multiplies lanes 0 and 2 only,
/// into two 64-bit products. It is used in two ways:
///
/// - `mulw` gives the 4 products of 4 lanes: one multiplication for lanes 0
///   and 2, one for lanes 1 and 3 after a shift, and a blend.
/// - The shaping filter is the longest chain of dependent steps of a sample.
///   There each register holds two states, in lanes 0 and 2 (see `pair`).
///   One multiplication then gives the products of both states, and nothing
///   has to be merged: a filter tap is a subtraction, a multiplication, a
///   shift and an addition.
///
/// The large functions that run once per sample are `#[inline(always)]`,
/// which rules out target features of their own. Both copies of the loop
/// use them, and the compiler does not inline a large function that has two
/// callers; a call per sample would cost more than the function saves. They
/// are `unsafe` because they need the target features of the loop that
/// inlines them.
///
/// The intrinsics in the bodies of these `unsafe fn` are not wrapped in
/// `unsafe` blocks of their own: what makes each call sound is the
/// contract of the function, that the processor has SSE4.1. This module is
/// the only place of the library that allows `unsafe_op_in_unsafe_fn`.
#[cfg(target_arch = "x86_64")]
#[allow(unsafe_op_in_unsafe_fn)]
mod x86 {
    use super::*;
    // `load` and `store` read and write one row of four `i32` values,
    // `load_table` a byte table for `_mm_shuffle_epi8`.
    use crate::simd_mem::x86::{load_i32x4 as load, load_u8x16 as load_table, store_i32x4 as store};
    use core::arch::x86_64::*;

    /// `LANE_MASK[k]` has all bits of lane `k` set. Entry 4 selects no lane.
    static LANE_MASK: [[i32; 4]; 5] = [
        [-1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, -1],
        [0, 0, 0, 0],
    ];

    /// `BROADCAST[k]` is the byte table for `_mm_shuffle_epi8` that copies
    /// lane `k` into all four lanes.
    static BROADCAST: [[u8; 16]; 4] = [
        [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3],
        [4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7],
        [8, 9, 10, 11, 8, 9, 10, 11, 8, 9, 10, 11, 8, 9, 10, 11],
        [
            12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15,
        ],
    ];

    /// `PERM[r][b]` is the byte table for `_mm_shuffle_epi8` that copies lane
    /// `b` into lane `r` and keeps the other lanes in place. `PERM[4][b]`
    /// keeps every lane.
    static PERM: [[[u8; 16]; 4]; 5] = perm_tables();

    /// `PAIRS[r][b][p]` reads a row through `PERM[r][b]` and puts its lanes
    /// `2 * p` and `2 * p + 1` into lanes 0 and 2, as `pair::<p>` does.
    static PAIRS: [[[[u8; 16]; 2]; 4]; 5] = pair_tables();

    const fn perm_tables() -> [[[u8; 16]; 4]; 5] {
        let mut tables = [[[0u8; 16]; 4]; 5];
        let mut r = 0;
        while r < 5 {
            let mut b = 0;
            while b < 4 {
                let mut byte = 0;
                while byte < 16 {
                    let lane = byte / 4;
                    let from = if lane == r { b } else { lane };
                    tables[r][b][byte] = (4 * from + byte % 4) as u8;
                    byte += 1;
                }
                b += 1;
            }
            r += 1;
        }
        tables
    }

    const fn pair_tables() -> [[[[u8; 16]; 2]; 4]; 5] {
        let perm = perm_tables();
        let mut tables = [[[[0u8; 16]; 2]; 4]; 5];
        let mut r = 0;
        while r < 5 {
            let mut b = 0;
            while b < 4 {
                let mut p = 0;
                while p < 2 {
                    let mut byte = 0;
                    while byte < 16 {
                        // Lanes 0 and 1 take lane `2 * p`, lanes 2 and 3
                        // lane `2 * p + 1`, of the permuted row.
                        let lane = 2 * p + byte / 8;
                        tables[r][b][p][byte] = perm[r][b][4 * lane + byte % 4];
                        byte += 1;
                    }
                    p += 1;
                }
                b += 1;
            }
            r += 1;
        }
        tables
    }

    /// The first `N` lanes of `v`.
    #[target_feature(enable = "sse4.1")]
    #[inline]
    fn to_lanes<const N: usize>(v: __m128i) -> [i32; N] {
        const { assert!(N <= 4) };
        let mut all = [0i32; 4];
        store(&mut all, v);
        core::array::from_fn(|k| all[k])
    }

    /// A vector with `a[k]` in lane `k`. Lanes from `N` on are zero.
    #[target_feature(enable = "sse4.1")]
    #[inline]
    fn from_lanes<const N: usize>(a: [i32; N]) -> __m128i {
        const { assert!(N <= 4) };
        let mut all = [0i32; 4];
        all[..N].copy_from_slice(&a);
        load(&all)
    }

    /// The factor `f` for which `mulw(x, f)` is `silk_smulwb(x, c)`:
    /// `c << 16` in every lane.
    #[target_feature(enable = "sse4.1")]
    #[inline]
    fn mulw_factor(c: i16) -> __m128i {
        _mm_set1_epi32((c as i32) << 16)
    }

    /// `(x * c) >> 16` per lane, as `silk_smulwb(x, c)` and NEON's
    /// `vqdmulhq_s32(x, smulwb_operand(c))`, for `f = mulw_factor(c)`.
    ///
    /// `_mm_mul_epi32` gives the 64-bit products `x * c << 16` of lanes 0 and
    /// 2. Their upper halves, in lanes 1 and 3, are the results. The second
    /// multiplication does the same for lanes 1 and 3 moved down.
    #[target_feature(enable = "sse4.1")]
    #[inline]
    fn mulw(x: __m128i, f: __m128i) -> __m128i {
        let even = _mm_mul_epi32(x, f);
        let odd = _mm_mul_epi32(_mm_srli_epi64::<32>(x), f);
        _mm_blend_epi16::<0b1100_1100>(_mm_srli_epi64::<32>(even), odd)
    }

    /// Saturating `a + b` per lane, as `vqaddq_s32` and `silk_add_sat32`.
    ///
    /// `a` is first clamped to the range in which `a + b` does not overflow.
    /// The bounds depend on `b` only: when `b` is known early, the result
    /// follows `a` after three steps.
    #[target_feature(enable = "sse4.1")]
    #[inline]
    fn adds(a: __m128i, b: __m128i) -> __m128i {
        let zero = _mm_setzero_si128();
        // Overflow needs b > 0 and a > MAX - b; underflow b < 0 and a < MIN - b.
        let high = _mm_sub_epi32(_mm_set1_epi32(i32::MAX), _mm_max_epi32(b, zero));
        let low = _mm_sub_epi32(_mm_set1_epi32(i32::MIN), _mm_min_epi32(b, zero));
        _mm_add_epi32(_mm_min_epi32(_mm_max_epi32(a, low), high), b)
    }

    /// Saturating `a - b` per lane, as `vqsubq_s32` and `silk_sub_sat32`.
    ///
    /// `b` is first clamped to the range in which `a - b` does not overflow.
    /// The bounds depend on `a` only: when `a` is known early, the result
    /// follows `b` after three steps.
    #[target_feature(enable = "sse4.1")]
    #[inline]
    fn subs(a: __m128i, b: __m128i) -> __m128i {
        let minus_one = _mm_set1_epi32(-1);
        // a >= 0: overflow needs b < a - MAX. a < 0: underflow needs
        // b > a - MIN. With m = max(a, -1), m - MAX is a - MAX for a >= 0
        // and MIN for a < 0; with m = min(a, -1), m - MIN is MAX for a >= 0
        // and a - MIN for a < 0.
        let low = _mm_sub_epi32(_mm_max_epi32(a, minus_one), _mm_set1_epi32(i32::MAX));
        let high = _mm_sub_epi32(_mm_min_epi32(a, minus_one), _mm_set1_epi32(i32::MIN));
        _mm_sub_epi32(a, _mm_min_epi32(_mm_max_epi32(b, low), high))
    }

    /// `silk_rshift_round(v, PRE + 1)` per lane, as `vrshrq_n_s32`:
    /// `((v >> PRE) + 1) >> 1`, which cannot overflow. `PRE` is at least 1.
    #[target_feature(enable = "sse4.1")]
    #[inline]
    fn round_shift<const PRE: i32>(v: __m128i) -> __m128i {
        _mm_srai_epi32::<1>(_mm_add_epi32(_mm_srai_epi32::<PRE>(v), _mm_set1_epi32(1)))
    }

    /// `v` in the lanes where `s >= 0`, `-v` where `s < 0`. The lowest bit
    /// of `s` is set first, so that `_mm_sign_epi32` never sees a zero (it
    /// would give zero).
    #[target_feature(enable = "sse4.1")]
    #[inline]
    fn apply_sign(v: __m128i, s: __m128i) -> __m128i {
        _mm_sign_epi32(v, _mm_or_si128(s, _mm_set1_epi32(1)))
    }

    /// `a` in the lanes where `mask` is all ones, `b` where it is zero, as
    /// `vbslq_s32`.
    #[target_feature(enable = "sse4.1")]
    #[inline]
    fn bsl(mask: __m128i, a: __m128i, b: __m128i) -> __m128i {
        _mm_blendv_epi8(b, a, mask)
    }

    /// `(a as i16 as i32) * (b as i16 as i32)` per lane, as `silk_smulbb`,
    /// for `b_low` holding `b as i16` in the low half of every lane and zero
    /// in the high half.
    #[target_feature(enable = "sse4.1")]
    #[inline]
    fn smulbb(a: __m128i, b_low: __m128i) -> __m128i {
        _mm_madd_epi16(a, b_low)
    }

    /// `acc + (a as i16 as i32) * (a as i16 as i32)` per lane, as
    /// `silk_smlabb(acc, a, a)`.
    #[target_feature(enable = "sse4.1")]
    #[inline]
    fn add_square(acc: __m128i, a: __m128i) -> __m128i {
        let a_low = _mm_and_si128(a, _mm_set1_epi32(0xFFFF));
        _mm_add_epi32(acc, _mm_madd_epi16(a, a_low))
    }

    /// The two candidates of every state for one sample, the cheaper one
    /// first. The costs are returned as the amounts to add to the cost so far.
    struct Candidates {
        q_q10: [__m128i; 2],
        rd_added_q10: [__m128i; 2],
        xq_q14: [__m128i; 2],
        lf_ar_q14: [__m128i; 2],
        diff_q14: [__m128i; 2],
        s_ltp_shp_q14: [__m128i; 2],
        lpc_exc_q14: [__m128i; 2],
    }

    /// Constants of one subframe for `quantize_sample_x86`.
    struct LevelVectors {
        tilt: __m128i,
        lf_shp_low: __m128i,
        lf_shp_high: __m128i,
        /// `lambda_q10 as i16` in the low half of every lane, zero in the
        /// high half: the operand of `smulbb`.
        lambda: __m128i,
        offset_q10: __m128i,
        rdo_offset: __m128i,
        minus_rdo_offset: __m128i,
    }

    impl LevelVectors {
        #[target_feature(enable = "sse4.1")]
        #[inline]
        fn new(lv: &LevelParams) -> Self {
            // `quantize_sample_x86` relies on it; `SubframeSetup::new` makes
            // it so.
            debug_assert!(lv.rdo_offset >= 0);
            Self {
                tilt: mulw_factor(lv.tilt_q14 as i16),
                lf_shp_low: mulw_factor(lv.lf_shp_q14 as i16),
                lf_shp_high: mulw_factor((lv.lf_shp_q14 >> 16) as i16),
                lambda: _mm_set1_epi32(lv.lambda_q10 & 0xFFFF),
                offset_q10: _mm_set1_epi32(lv.offset_q10),
                rdo_offset: _mm_set1_epi32(lv.rdo_offset),
                minus_rdo_offset: _mm_set1_epi32(-lv.rdo_offset),
            }
        }
    }

    /// `quantize_sample` for four states at once, as `quantize_sample_neon`.
    ///
    /// The value of the shaping filter, `n_ar`, comes last: everything up to
    /// the next sample waits for this function. Some steps are therefore
    /// written with fewer dependent instructions than in the NEON version;
    /// each such place says what it computes.
    ///
    /// # Safety
    ///
    /// The processor must have SSE4.1.
    #[inline(always)]
    unsafe fn quantize_sample_x86(
        lv: &LevelVectors,
        x_q10: i32,
        ltp_pred_q14: i32,
        n_ltp_q14: i32,
        n_ar: __m128i,
        lpc_pred_q14: __m128i,
        lf_ar_q14: __m128i,
        shape_q14: __m128i,
        seed: __m128i,
    ) -> Candidates {
        let zero = _mm_setzero_si128();
        let minus_one = _mm_set1_epi32(-1);
        let n_ar_q14 = _mm_slli_epi32::<2>(_mm_add_epi32(
            _mm_slli_epi32::<1>(n_ar),
            mulw(lf_ar_q14, lv.tilt),
        ));
        let n_lf_q14 = _mm_slli_epi32::<2>(_mm_add_epi32(
            mulw(shape_q14, lv.lf_shp_low),
            mulw(lf_ar_q14, lv.lf_shp_high),
        ));

        // `adds` and `subs` are fast in the operand that comes from `n_ar`.
        let tmp1 = subs(
            _mm_add_epi32(_mm_set1_epi32(n_ltp_q14), lpc_pred_q14),
            adds(n_ar_q14, n_lf_q14),
        );
        let r_q10 = _mm_sub_epi32(_mm_set1_epi32(x_q10), round_shift::<3>(tmp1));

        // The dither flips the sign where the seed is negative.
        let r_q10 = apply_sign(r_q10, seed);
        let r_q10 = _mm_min_epi32(
            _mm_max_epi32(r_q10, _mm_set1_epi32(-(31 << 10))),
            _mm_set1_epi32(30 << 10),
        );

        // NEON chooses between `(q1_q10_in - rdo) >> 10` above `rdo`,
        // `(q1_q10_in + rdo) >> 10` below `-rdo` and `q1_q10_in >> 31`
        // between them. With `rdo >= 0` that is
        // `(q1_q10_in - clamp(q1_q10_in, -rdo, rdo) - m) >> 10`, where `m`
        // is 1 for `-rdo <= q1_q10_in < 0` and 0 otherwise.
        let q1_q10_in = _mm_sub_epi32(r_q10, lv.offset_q10);
        let middle_negative = _mm_andnot_si128(
            _mm_cmplt_epi32(q1_q10_in, lv.minus_rdo_offset),
            _mm_cmplt_epi32(q1_q10_in, zero),
        );
        let clamped = _mm_min_epi32(
            _mm_max_epi32(q1_q10_in, lv.minus_rdo_offset),
            lv.rdo_offset,
        );
        // `middle_negative` is -1 where `m` is 1.
        let q1_q0 = _mm_srai_epi32::<10>(_mm_add_epi32(
            _mm_sub_epi32(q1_q10_in, clamped),
            middle_negative,
        ));
        let base_q10 = _mm_add_epi32(_mm_slli_epi32::<10>(q1_q0), lv.offset_q10);
        // `base - adjust` for q1_q0 > 0, `base` for 0 and `base + adjust`
        // below: `adjust` takes the sign of `q1_q0`, and 0 for 0.
        let adjust = _mm_set1_epi32(QUANT_LEVEL_ADJUST_Q10);
        let q1_q10 = _mm_sub_epi32(base_q10, _mm_sign_epi32(adjust, q1_q0));
        let near_zero = _mm_or_si128(
            _mm_cmpeq_epi32(q1_q0, zero),
            _mm_cmpeq_epi32(q1_q0, minus_one),
        );
        let q2_q10 = _mm_sub_epi32(
            _mm_add_epi32(q1_q10, _mm_set1_epi32(1024)),
            _mm_and_si128(near_zero, adjust),
        );
        // The rate term uses the magnitude of the level: negate q1 when
        // q1_q0 < 0 and q2 when q1_q0 < -1, that is when q1_q0 + 1 < 0.
        let one = _mm_set1_epi32(1);
        let rd1_q10 = smulbb(apply_sign(q1_q10, q1_q0), lv.lambda);
        let rd2_q10 = smulbb(
            apply_sign(q2_q10, _mm_add_epi32(q1_q0, one)),
            lv.lambda,
        );
        let rd1_q10 = _mm_srai_epi32::<10>(add_square(rd1_q10, _mm_sub_epi32(r_q10, q1_q10)));
        let rd2_q10 = _mm_srai_epi32::<10>(add_square(rd2_q10, _mm_sub_epi32(r_q10, q2_q10)));

        let first_is_cheaper = _mm_cmplt_epi32(rd1_q10, rd2_q10);
        let q = [
            bsl(first_is_cheaper, q1_q10, q2_q10),
            bsl(first_is_cheaper, q2_q10, q1_q10),
        ];

        let x_q14 = _mm_set1_epi32(silk_lshift(x_q10, 4));
        let ltp_pred_q14 = _mm_set1_epi32(ltp_pred_q14);
        let lpc_exc = [
            _mm_add_epi32(apply_sign(_mm_slli_epi32::<4>(q[0]), seed), ltp_pred_q14),
            _mm_add_epi32(apply_sign(_mm_slli_epi32::<4>(q[1]), seed), ltp_pred_q14),
        ];
        let xq = [
            _mm_add_epi32(lpc_exc[0], lpc_pred_q14),
            _mm_add_epi32(lpc_exc[1], lpc_pred_q14),
        ];
        let diff = [_mm_sub_epi32(xq[0], x_q14), _mm_sub_epi32(xq[1], x_q14)];
        let lf_ar = [
            _mm_sub_epi32(diff[0], n_ar_q14),
            _mm_sub_epi32(diff[1], n_ar_q14),
        ];
        Candidates {
            q_q10: q,
            rd_added_q10: [
                _mm_min_epi32(rd1_q10, rd2_q10),
                _mm_max_epi32(rd1_q10, rd2_q10),
            ],
            xq_q14: xq,
            lf_ar_q14: lf_ar,
            diff_q14: diff,
            s_ltp_shp_q14: [subs(lf_ar[0], n_lf_q14), subs(lf_ar[1], n_lf_q14)],
            lpc_exc_q14: lpc_exc,
        }
    }

    /// The candidates of `quantize_sample_x86` as arrays, for the tests.
    #[cfg(test)]
    pub(super) struct CandidateLanes {
        pub(super) q_q10: [i32; 4],
        /// The amount added to the cost so far.
        pub(super) rd_q10: [i32; 4],
        pub(super) xq_q14: [i32; 4],
        pub(super) lf_ar_q14: [i32; 4],
        pub(super) diff_q14: [i32; 4],
        pub(super) s_ltp_shp_q14: [i32; 4],
        pub(super) lpc_exc_q14: [i32; 4],
    }

    /// `quantize_sample_x86` on arrays, for the tests. Panics when the
    /// processor does not have SSE4.1.
    #[cfg(test)]
    pub(super) fn level_decision_lanes(
        lv: &LevelParams,
        x_q10: i32,
        ltp_pred_q14: i32,
        n_ltp_q14: i32,
        n_ar: [i32; 4],
        lpc_pred_q14: [i32; 4],
        lf_ar_q14: [i32; 4],
        shape_q14: [i32; 4],
        seed: [i32; 4],
    ) -> [CandidateLanes; 2] {
        assert!(crate::compat::x86_has_sse41());
        // SAFETY: the processor has SSE4.1 (checked above).
        unsafe {
            let c = quantize_sample_x86(
                &LevelVectors::new(lv),
                x_q10,
                ltp_pred_q14,
                n_ltp_q14,
                load(&n_ar),
                load(&lpc_pred_q14),
                load(&lf_ar_q14),
                load(&shape_q14),
                load(&seed),
            );
            core::array::from_fn(|j| CandidateLanes {
                q_q10: to_lanes(c.q_q10[j]),
                rd_q10: to_lanes(c.rd_added_q10[j]),
                xq_q14: to_lanes(c.xq_q14[j]),
                lf_ar_q14: to_lanes(c.lf_ar_q14[j]),
                diff_q14: to_lanes(c.diff_q14[j]),
                s_ltp_shp_q14: to_lanes(c.s_ltp_shp_q14[j]),
                lpc_exc_q14: to_lanes(c.lpc_exc_q14[j]),
            })
        }
    }

    /// The vector helpers on arrays, for the tests: `[mulw, adds, subs,
    /// round_shift::<3>, round_shift::<9>, apply_sign]` of `a` and `b`, with
    /// `mulw` by `b as i16`. Panics when the processor does not have SSE4.1.
    #[cfg(test)]
    pub(super) fn helpers_lanes(a: [i32; 4], b: [i32; 4]) -> [[i32; 4]; 6] {
        assert!(crate::compat::x86_has_sse41());
        // SAFETY: the processor has SSE4.1 (checked above).
        unsafe {
            let (va, vb) = (load(&a), load(&b));
            let products: [i32; 4] =
                core::array::from_fn(|k| to_lanes::<4>(mulw(va, mulw_factor(b[k] as i16)))[k]);
            [
                products,
                to_lanes(adds(va, vb)),
                to_lanes(subs(va, vb)),
                to_lanes(round_shift::<3>(va)),
                to_lanes(round_shift::<9>(va)),
                to_lanes(apply_sign(va, vb)),
            ]
        }
    }

    /// The shaping filter states of all delayed-decision states, one row per
    /// tap: lane `k` of row `m` is tap `m` of state `k`.
    pub(super) type ShapingRows = [[i32; 4]; MAX_SHAPE_LPC_ORDER];

    /// The number of registers that hold the shaping filter of `n` states,
    /// two states in each (see `pair`).
    const fn pair_count(n: usize) -> usize {
        if n > 2 { 2 } else { 1 }
    }

    /// States `2 * P` and `2 * P + 1` of `v` (one state per lane) in lanes
    /// 0 and 2, the lanes that `_mm_mul_epi32` multiplies. Lanes 1 and 3
    /// hold copies.
    #[target_feature(enable = "sse4.1")]
    #[inline]
    fn pair<const P: usize>(v: __m128i) -> __m128i {
        if P == 0 {
            _mm_shuffle_epi32::<0b01_01_00_00>(v)
        } else {
            _mm_shuffle_epi32::<0b11_11_10_10>(v)
        }
    }

    /// Both pairs of `v`.
    #[target_feature(enable = "sse4.1")]
    #[inline]
    fn pairs(v: __m128i) -> [__m128i; 2] {
        [pair::<0>(v), pair::<1>(v)]
    }

    /// The values in lanes 0 and 2 of the pairs of `N` states, one state per
    /// lane. For `N <= 2` lanes 2 and 3 hold copies.
    #[target_feature(enable = "sse4.1")]
    #[inline]
    fn unpair<const N: usize>(p: [__m128i; 2]) -> __m128i {
        if N > 2 {
            let (a, b) = (_mm_castsi128_ps(p[0]), _mm_castsi128_ps(p[1]));
            _mm_castps_si128(_mm_shuffle_ps::<0b10_00_10_00>(a, b))
        } else {
            _mm_shuffle_epi32::<0b10_00_10_00>(p[0])
        }
    }

    /// The values in lanes 1 and 3 of the pairs of `N` states (the sums of
    /// `tap`), one state per lane.
    #[target_feature(enable = "sse4.1")]
    #[inline]
    fn unpair_sums<const N: usize>(p: [__m128i; 2]) -> __m128i {
        if N > 2 {
            let (a, b) = (_mm_castsi128_ps(p[0]), _mm_castsi128_ps(p[1]));
            _mm_castps_si128(_mm_shuffle_ps::<0b11_01_11_01>(a, b))
        } else {
            _mm_shuffle_epi32::<0b11_01_11_01>(p[0])
        }
    }

    /// Constants of the shaping filter of one subframe.
    pub(super) struct ShapingCoefs {
        /// `warping_q16 as i16` in every lane: `_mm_mul_epi32` by it and a
        /// shift right by 16 give `silk_smulwb(x, warping_q16)` in lanes 0
        /// and 2.
        warping: __m128i,
        /// Row `m` is `ar_shp_q13[m] << 16` in every lane: `_mm_mul_epi32`
        /// by it gives `silk_smulwb(t, ar_shp_q13[m])` in lanes 1 and 3. The
        /// rows from the order on are zero.
        coef: [[i32; 4]; MAX_SHAPE_LPC_ORDER],
        /// `order / 2` in every lane: the start of the feedback sum.
        bias: __m128i,
        /// The shaping order: the length of `ar_shp_q13`.
        order: usize,
    }

    impl ShapingCoefs {
        #[target_feature(enable = "sse4.1")]
        #[inline]
        pub(super) fn new(ar_shp_q13: &[i16], warping_q16: i32) -> Self {
            let order = ar_shp_q13.len().min(MAX_SHAPE_LPC_ORDER);
            let mut coef = [[0i32; 4]; MAX_SHAPE_LPC_ORDER];
            for (row, &c) in coef.iter_mut().zip(&ar_shp_q13[..order]) {
                *row = [(c as i32) << 16; 4];
            }
            Self {
                warping: _mm_set1_epi32(warping_q16 as i16 as i32),
                coef,
                bias: _mm_set1_epi32((order as i32) >> 1),
                order,
            }
        }
    }

    /// The rows of the shaping filter for the filter states of `states`.
    /// Lanes from `N` on are zero.
    pub(super) fn load_shaping_rows<const N: usize>(states: &[PathState; N]) -> ShapingRows {
        let mut rows = [[0i32; 4]; MAX_SHAPE_LPC_ORDER];
        for (m, row) in rows.iter_mut().enumerate() {
            for k in 0..N {
                row[k] = states[k].s_ar2_q14[m];
            }
        }
        rows
    }

    /// Writes the rows back to the filter states of `states`, every row read
    /// through the byte table `perm` (see `neon::store_shaping_rows`).
    #[target_feature(enable = "sse4.1")]
    pub(super) fn store_shaping_rows<const N: usize>(
        states: &mut [PathState; N],
        rows: &ShapingRows,
        perm: __m128i,
    ) {
        for (m, row) in rows.iter().enumerate() {
            let row: [i32; N] = to_lanes(_mm_shuffle_epi8(load(row), perm));
            for k in 0..N {
                states[k].s_ar2_q14[m] = row[k];
            }
        }
    }

    /// The two candidates of every state for the last sample, and which
    /// state continues as a copy of which (see `neon::LastChoice`).
    #[derive(Copy, Clone)]
    pub(super) struct LastChoice {
        /// `diff_q14` of the first candidate of every state.
        pub(super) diff_first: __m128i,
        /// `diff_q14` of the second candidate of every state.
        pub(super) diff_second: __m128i,
        /// All bits set in the lane of the state that continues as a copy
        /// of the best state, with that state's second candidate. No bit is
        /// set when no state is replaced.
        pub(super) replaced: __m128i,
        /// The byte table for `_mm_shuffle_epi8` that copies the lane of the
        /// best state into all lanes.
        pub(super) from_best: __m128i,
        /// The byte tables that read a row through `PERM[replaced_lane][best]`
        /// (which moves the lane of the best state to the replaced lane and
        /// keeps the other lanes in place) into the layout of `pair::<0>` and
        /// `pair::<1>`.
        pub(super) pairs: [__m128i; 2],
        /// The lane of the replaced state, 4 when no state is replaced, and
        /// the lane of the best state: the indices of the tables, for
        /// `wide`.
        pub(super) replaced_lane: usize,
        pub(super) best: usize,
    }

    impl LastChoice {
        /// `replaced_lane` is the lane of the state that continues as a copy
        /// of state `best`, or 4 when no state is replaced.
        #[target_feature(enable = "sse4.1")]
        #[inline]
        pub(super) fn new(
            diff_first: __m128i,
            diff_second: __m128i,
            replaced_lane: usize,
            best: usize,
        ) -> Self {
            // Out-of-range values cannot occur; the limits let the compiler
            // drop the bounds checks.
            let replaced_lane = replaced_lane.min(4);
            let best = best.min(3);
            let pairs = &PAIRS[replaced_lane][best];
            Self {
                diff_first,
                diff_second,
                replaced: load(&LANE_MASK[replaced_lane]),
                from_best: load_table(&BROADCAST[best]),
                pairs: [load_table(&pairs[0]), load_table(&pairs[1])],
                replaced_lane,
                best,
            }
        }

        /// For the first sample of a subframe: every state keeps its own
        /// `diff_q14` and filter state.
        #[target_feature(enable = "sse4.1")]
        #[inline]
        pub(super) fn keep(diff_q14: __m128i) -> Self {
            Self::new(diff_q14, diff_q14, 4, 0)
        }

        /// The chosen value of every state from the values `first` and
        /// `second` of the two candidates of every state.
        #[target_feature(enable = "sse4.1")]
        #[inline]
        pub(super) fn take(&self, first: __m128i, second: __m128i) -> __m128i {
            let second = _mm_shuffle_epi8(second, self.from_best);
            bsl(self.replaced, second, first)
        }
    }

    /// How many taps of the shaping filter `shaping_filter_both` computes for
    /// both candidates of every state (see `neon::FIRST_TAPS`).
    ///
    /// Like the Apple cores, for which NEON uses 5, recent x86 cores execute
    /// far ahead of an instruction that waits. The choice of the survivors
    /// takes an estimated 25 to 35 cycles, and a tap 8 (the multiplication
    /// alone takes 5), so 4 taps cover it. The value is an estimate, not a
    /// measurement: compare 0, 2, 4 and 6 on the target processor with
    /// `perf_harness pairs`.
    pub(super) const FIRST_TAPS: usize = 4;

    /// Tap `m` of the shaping filter for `N` states in the layout of `pair`:
    /// `old` is row `m` read through the byte tables `tables`, then
    /// `t = prev + ((old - t) * warping) >> 16`, the new row is `t`, `sum`
    /// gets `(t * ar_shp_q13[m]) >> 16` in lanes 1 and 3, and `prev = old`.
    ///
    /// # Safety
    ///
    /// The processor must have SSE4.1.
    #[inline(always)]
    unsafe fn tap<const N: usize>(
        rows: &mut ShapingRows,
        m: usize,
        tables: &[__m128i; 2],
        c: &ShapingCoefs,
        prev: &mut [__m128i; 2],
        t: &mut [__m128i; 2],
        sum: &mut [__m128i; 2],
    ) {
        let row = load(&rows[m]);
        let coef = load(&c.coef[m]);
        for p in 0..pair_count(N) {
            let old = _mm_shuffle_epi8(row, tables[p]);
            let product = _mm_mul_epi32(_mm_sub_epi32(old, t[p]), c.warping);
            t[p] = _mm_add_epi32(prev[p], _mm_srli_epi64::<16>(product));
            sum[p] = _mm_add_epi32(sum[p], _mm_mul_epi32(t[p], coef));
            prev[p] = old;
        }
        store(&mut rows[m], unpair::<N>(*t));
    }

    /// Taps `start` to the order minus 1 of the shaping filter, every row
    /// read through `tables`. `prev` is `s[start - 1]` and `t` is
    /// `t[start - 1]`, both in the layout of `pair`. Returns the sum of the
    /// products of these taps, one state per lane, without `order / 2`.
    ///
    /// `ORDER` is the shaping order when it is known at compile time; every
    /// tap is then written out. With `ORDER == 0` the order is `c.order`.
    ///
    /// # Safety
    ///
    /// The processor must have SSE4.1.
    #[inline(always)]
    unsafe fn later_taps<const N: usize, const ORDER: usize>(
        rows: &mut ShapingRows,
        tables: &[__m128i; 2],
        c: &ShapingCoefs,
        start: usize,
        mut prev: [__m128i; 2],
        mut t: [__m128i; 2],
    ) -> __m128i {
        let mut sum = [_mm_setzero_si128(); 2];
        if ORDER == 0 {
            for m in start..c.order.min(MAX_SHAPE_LPC_ORDER) {
                tap::<N>(rows, m, tables, c, &mut prev, &mut t, &mut sum);
            }
        } else {
            macro_rules! taps {
                ($($m:literal)*) => {$(
                    if $m >= start && $m < ORDER && $m < MAX_SHAPE_LPC_ORDER {
                        tap::<N>(rows, $m, tables, c, &mut prev, &mut t, &mut sum);
                    }
                )*};
            }
            taps!(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23);
        }
        unpair_sums::<N>(sum)
    }

    /// Warped noise-shaping filter of one sample for all states at once,
    /// state `k` in lane `k` of `diff_q14` and of the result. Same results as
    /// `warped_shaping_filter` and `neon::shaping_filter_rows`. Each row is
    /// read through the byte tables `tables` (`LastChoice::pairs`).
    ///
    /// # Safety
    ///
    /// The processor must have SSE4.1.
    #[inline(always)]
    pub(super) unsafe fn shaping_filter_rows<const N: usize, const ORDER: usize>(
        rows: &mut ShapingRows,
        tables: &[__m128i; 2],
        diff_q14: __m128i,
        c: &ShapingCoefs,
    ) -> __m128i {
        debug_assert!(ORDER == 0 || ORDER == c.order);
        let zero = _mm_setzero_si128();
        let sum = later_taps::<N, ORDER>(rows, tables, c, 0, pairs(diff_q14), [zero; 2]);
        _mm_add_epi32(sum, c.bias)
    }

    /// `shaping_filter_rows` for the next sample, started before the choice
    /// of the survivors of the last sample is known, as
    /// `neon::shaping_filter_both`. The first `FIRST` taps are computed for
    /// both candidates of every state on the state's own rows; the chosen
    /// values are stored, and the later taps continue from them.
    ///
    /// The products of the first taps are summed for each candidate apart.
    /// `take` chooses lane by lane, so taking the sum is the same as summing
    /// the taken products.
    ///
    /// # Safety
    ///
    /// The processor must have SSE4.1.
    #[inline(always)]
    pub(super) unsafe fn shaping_filter_both<
        const N: usize,
        const ORDER: usize,
        const FIRST: usize,
    >(
        rows: &mut ShapingRows,
        last: &LastChoice,
        c: &ShapingCoefs,
    ) -> __m128i {
        let first_taps = FIRST.min(ORDER).min(MAX_SHAPE_LPC_ORDER);
        if first_taps == 0 {
            let diff_q14 = last.take(last.diff_first, last.diff_second);
            return shaping_filter_rows::<N, ORDER>(rows, &last.pairs, diff_q14, c);
        }
        debug_assert!(ORDER == c.order);
        let zero = _mm_setzero_si128();
        // `t` of the two candidates, and their sums.
        let mut first = [zero; 2];
        let mut second = [zero; 2];
        let mut sum_first = [zero; 2];
        let mut sum_second = [zero; 2];
        // Row `m - 1` of the states' own rows, and the chosen `t` of tap
        // `m - 1`, one state per lane.
        let mut own_prev = zero;
        let mut t = zero;
        macro_rules! first_taps {
            ($($m:literal)*) => {$(
                if $m < first_taps {
                    let row = load(&rows[$m]);
                    let coef = load(&c.coef[$m]);
                    let old = pairs(row);
                    // `s[-1]` is `diff`.
                    let (prev_first, prev_second) = if $m == 0 {
                        (pairs(last.diff_first), pairs(last.diff_second))
                    } else {
                        (pairs(own_prev), pairs(own_prev))
                    };
                    for p in 0..pair_count(N) {
                        let product = _mm_mul_epi32(_mm_sub_epi32(old[p], first[p]), c.warping);
                        first[p] = _mm_add_epi32(prev_first[p], _mm_srli_epi64::<16>(product));
                        let product = _mm_mul_epi32(_mm_sub_epi32(old[p], second[p]), c.warping);
                        second[p] = _mm_add_epi32(prev_second[p], _mm_srli_epi64::<16>(product));
                        let product = _mm_mul_epi32(first[p], coef);
                        sum_first[p] = _mm_add_epi32(sum_first[p], product);
                        let product = _mm_mul_epi32(second[p], coef);
                        sum_second[p] = _mm_add_epi32(sum_second[p], product);
                    }
                    t = last.take(unpair::<N>(first), unpair::<N>(second));
                    store(&mut rows[$m], t);
                    own_prev = row;
                }
            )*};
        }
        const { assert!(FIRST <= 8) };
        first_taps!(0 1 2 3 4 5 6 7);
        // The later taps read `s[first_taps - 1]` through the last choice.
        let prev = [
            _mm_shuffle_epi8(own_prev, last.pairs[0]),
            _mm_shuffle_epi8(own_prev, last.pairs[1]),
        ];
        let later = later_taps::<N, ORDER>(rows, &last.pairs, c, first_taps, prev, pairs(t));
        let first_sums = last.take(unpair_sums::<N>(sum_first), unpair_sums::<N>(sum_second));
        _mm_add_epi32(_mm_add_epi32(first_sums, later), c.bias)
    }

    /// `shaping_filter_rows` on arrays, for the tests, as
    /// `neon::shaping_filter_rows_lanes`. With `wide`, the functions tested
    /// are those of `wide`. Panics when the processor does not have SSE4.1,
    /// or AVX2 for `wide`.
    #[cfg(test)]
    pub(super) fn shaping_filter_rows_lanes<
        const N: usize,
        const ORDER: usize,
        const FIRST: usize,
    >(
        states: &mut ShapingStates<N>,
        source: [u8; N],
        diff_q14: [i32; N],
        ar_shp_q13: &[i16],
        warping_q16: i32,
        both: bool,
        wide: bool,
    ) -> [i32; N] {
        assert!(crate::compat::x86_has_sse41());
        assert!(!wide || crate::compat::x86_has_avx2());
        // SAFETY: the processor has SSE4.1, and AVX2 for `wide` (checked
        // above).
        unsafe {
            let c = ShapingCoefs::new(ar_shp_q13, warping_q16);
            let mut rows = [[0i32; 4]; MAX_SHAPE_LPC_ORDER];
            for (m, row) in rows.iter_mut().enumerate() {
                for k in 0..N {
                    row[k] = states[k][m];
                }
            }
            // At most one state reads another state's filter state: that is
            // the replaced state, and the state it reads is the best one.
            let replaced = (0..N).find(|&k| source[k] as usize != k);
            assert!((0..N).all(|k| Some(k) == replaced || source[k] as usize == k));
            // The second candidates of the other states must not matter.
            let mut second: [i32; 4] = core::array::from_fn(|k| (k as i32 + 1) * 0x0123_4567);
            let (replaced_lane, best) = match replaced {
                Some(dst) => {
                    let best = source[dst] as usize;
                    second[best] = diff_q14[dst];
                    (dst, best)
                }
                None => (4, 0),
            };
            let last = LastChoice::new(from_lanes(diff_q14), load(&second), replaced_lane, best);
            let n_ar = match (wide, both) {
                (false, true) => shaping_filter_both::<N, ORDER, FIRST>(&mut rows, &last, &c),
                (false, false) => shaping_filter_rows::<N, ORDER>(
                    &mut rows,
                    &last.pairs,
                    from_lanes(diff_q14),
                    &c,
                ),
                (true, true) => wide::shaping_filter_both::<ORDER, FIRST>(&mut rows, &last, &c),
                (true, false) => {
                    wide::shaping_filter_rows::<ORDER>(&mut rows, &last, from_lanes(diff_q14), &c)
                }
            };
            // Only the taps below the order change.
            for (m, row) in rows.iter().enumerate().take(c.order) {
                for k in 0..N {
                    states[k][m] = row[k];
                }
            }
            to_lanes(n_ar)
        }
    }

    /// The shaping filter of up to 4 states in one 256-bit register, for
    /// `quantize_subframe_avx2`: state `k` in the 64-bit lane `k`, its value
    /// in the lower half (32-bit lane `2 * k`), as libopus's AVX2 quantizer
    /// keeps its states. One `_mm256_mul_epi32` then multiplies all states.
    /// The steps are those of `tap` and `shaping_filter_both` above, and the
    /// rows keep their layout, one state per 32-bit lane.
    mod wide {
        use super::*;
        // `load8` reads one table of eight `i32` values.
        use crate::simd_mem::x86::load_i32x8 as load8;

        /// `WIDEN[r][b]` is the index table for `_mm256_permutevar8x32_epi32`
        /// that reads a row through `PERM[r][b]` into the layout of this
        /// module: the 32-bit lanes `2 * k` and `2 * k + 1` take lane `k` of
        /// the permuted row.
        static WIDEN: [[[i32; 8]; 4]; 5] = widen_tables();

        /// The index tables that take the 32-bit lanes 0, 2, 4 and 6 (the
        /// values) and 1, 3, 5 and 7 (the sums of `tap`) into lanes 0 to 3.
        static NARROW: [i32; 8] = [0, 2, 4, 6, 0, 2, 4, 6];
        static NARROW_SUMS: [i32; 8] = [1, 3, 5, 7, 1, 3, 5, 7];

        const fn widen_tables() -> [[[i32; 8]; 4]; 5] {
            let mut tables = [[[0i32; 8]; 4]; 5];
            let mut r = 0;
            while r < 5 {
                let mut b = 0;
                while b < 4 {
                    let mut lane = 0;
                    while lane < 8 {
                        let state = lane / 2;
                        tables[r][b][lane] = if state == r { b as i32 } else { state as i32 };
                        lane += 1;
                    }
                    b += 1;
                }
                r += 1;
            }
            tables
        }

        /// `row`, one state per 32-bit lane, in the layout of this module,
        /// read through the index table `index`.
        #[target_feature(enable = "avx2")]
        #[inline]
        fn widen(row: __m128i, index: __m256i) -> __m256i {
            // The indices are below 4: only the lower half is read.
            _mm256_permutevar8x32_epi32(_mm256_castsi128_si256(row), index)
        }

        /// The lanes of `v` that the first 4 entries of `index` name.
        #[target_feature(enable = "avx2")]
        #[inline]
        fn narrow(v: __m256i, index: __m256i) -> __m128i {
            _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(v, index))
        }

        /// `super::tap` for all states in one register: `index` is the
        /// `WIDEN` table of the last choice.
        ///
        /// # Safety
        ///
        /// The processor must have AVX2.
        #[inline(always)]
        unsafe fn tap(
            rows: &mut ShapingRows,
            m: usize,
            index: __m256i,
            c: &ShapingCoefs,
            prev: &mut __m256i,
            t: &mut __m256i,
            sum: &mut __m256i,
        ) {
            let old = widen(load(&rows[m]), index);
            let coef = _mm256_broadcastsi128_si256(load(&c.coef[m]));
            let warping = _mm256_broadcastsi128_si256(c.warping);
            let product = _mm256_mul_epi32(_mm256_sub_epi32(old, *t), warping);
            *t = _mm256_add_epi32(*prev, _mm256_srli_epi64::<16>(product));
            *sum = _mm256_add_epi32(*sum, _mm256_mul_epi32(*t, coef));
            *prev = old;
            store(&mut rows[m], narrow(*t, load8(&NARROW)));
        }

        /// `super::later_taps` for all states in one register.
        ///
        /// # Safety
        ///
        /// The processor must have AVX2.
        #[inline(always)]
        unsafe fn later_taps<const ORDER: usize>(
            rows: &mut ShapingRows,
            index: __m256i,
            c: &ShapingCoefs,
            start: usize,
            mut prev: __m256i,
            mut t: __m256i,
        ) -> __m128i {
            let mut sum = _mm256_setzero_si256();
            if ORDER == 0 {
                for m in start..c.order.min(MAX_SHAPE_LPC_ORDER) {
                    tap(rows, m, index, c, &mut prev, &mut t, &mut sum);
                }
            } else {
                macro_rules! taps {
                    ($($m:literal)*) => {$(
                        if $m >= start && $m < ORDER && $m < MAX_SHAPE_LPC_ORDER {
                            tap(rows, $m, index, c, &mut prev, &mut t, &mut sum);
                        }
                    )*};
                }
                taps!(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23);
            }
            narrow(sum, load8(&NARROW_SUMS))
        }

        /// `super::shaping_filter_rows` for all states in one register, with
        /// every row read through the tables of `last`.
        ///
        /// # Safety
        ///
        /// The processor must have AVX2.
        #[inline(always)]
        pub(super) unsafe fn shaping_filter_rows<const ORDER: usize>(
            rows: &mut ShapingRows,
            last: &LastChoice,
            diff_q14: __m128i,
            c: &ShapingCoefs,
        ) -> __m128i {
            debug_assert!(ORDER == 0 || ORDER == c.order);
            let index = load8(&WIDEN[last.replaced_lane.min(4)][last.best.min(3)]);
            let diff_q14 = widen(diff_q14, load8(&WIDEN[4][0]));
            let sum = later_taps::<ORDER>(rows, index, c, 0, diff_q14, _mm256_setzero_si256());
            _mm_add_epi32(sum, c.bias)
        }

        /// `super::shaping_filter_both` for all states in one register.
        ///
        /// # Safety
        ///
        /// The processor must have AVX2.
        #[inline(always)]
        pub(super) unsafe fn shaping_filter_both<const ORDER: usize, const FIRST: usize>(
            rows: &mut ShapingRows,
            last: &LastChoice,
            c: &ShapingCoefs,
        ) -> __m128i {
            let first_taps = FIRST.min(ORDER).min(MAX_SHAPE_LPC_ORDER);
            if first_taps == 0 {
                let diff_q14 = last.take(last.diff_first, last.diff_second);
                return shaping_filter_rows::<ORDER>(rows, last, diff_q14, c);
            }
            debug_assert!(ORDER == c.order);
            // The states' own rows, and the rows through the last choice.
            let identity = load8(&WIDEN[4][0]);
            let index = load8(&WIDEN[last.replaced_lane.min(4)][last.best.min(3)]);
            let narrow_values = load8(&NARROW);
            let narrow_sums = load8(&NARROW_SUMS);
            let warping = _mm256_broadcastsi128_si256(c.warping);
            let zero = _mm256_setzero_si256();
            // `t` of the two candidates, and their sums.
            let (mut first, mut second) = (zero, zero);
            let (mut sum_first, mut sum_second) = (zero, zero);
            // Row `m - 1` of the states' own rows, and the chosen `t` of tap
            // `m - 1`, one state per lane.
            let mut own_prev = _mm_setzero_si128();
            let mut t = _mm_setzero_si128();
            macro_rules! first_taps {
                ($($m:literal)*) => {$(
                    if $m < first_taps {
                        let row = load(&rows[$m]);
                        let coef = _mm256_broadcastsi128_si256(load(&c.coef[$m]));
                        let old = widen(row, identity);
                        // `s[-1]` is `diff`.
                        let (prev_first, prev_second) = if $m == 0 {
                            (widen(last.diff_first, identity), widen(last.diff_second, identity))
                        } else {
                            (widen(own_prev, identity), widen(own_prev, identity))
                        };
                        let product = _mm256_mul_epi32(_mm256_sub_epi32(old, first), warping);
                        first = _mm256_add_epi32(prev_first, _mm256_srli_epi64::<16>(product));
                        let product = _mm256_mul_epi32(_mm256_sub_epi32(old, second), warping);
                        second = _mm256_add_epi32(prev_second, _mm256_srli_epi64::<16>(product));
                        sum_first = _mm256_add_epi32(sum_first, _mm256_mul_epi32(first, coef));
                        sum_second = _mm256_add_epi32(sum_second, _mm256_mul_epi32(second, coef));
                        t = last.take(narrow(first, narrow_values), narrow(second, narrow_values));
                        store(&mut rows[$m], t);
                        own_prev = row;
                    }
                )*};
            }
            const { assert!(FIRST <= 8) };
            first_taps!(0 1 2 3 4 5 6 7);
            // The later taps read `s[first_taps - 1]` through the last choice.
            let prev = widen(own_prev, index);
            let later = later_taps::<ORDER>(rows, index, c, first_taps, prev, widen(t, identity));
            let first_sums = last.take(
                narrow(sum_first, narrow_sums),
                narrow(sum_second, narrow_sums),
            );
            _mm_add_epi32(_mm_add_epi32(first_sums, later), c.bias)
        }

        /// `ShortPrediction::predict` with 256-bit loads: 4 products per
        /// multiplication instead of 2.
        ///
        /// # Safety
        ///
        /// The processor must have AVX2.
        #[inline(always)]
        pub(super) unsafe fn predict<const N: usize>(
            lpc: &ShortPrediction,
            windows: [&[i32; NSQ_LPC_BUF_LENGTH]; N],
            newest: __m128i,
            second: __m128i,
        ) -> __m128i {
            const { assert!(N <= 4) };
            let zero = _mm256_setzero_si256();
            // The odd 32-bit lanes of `partial[k]` hold four partial sums of
            // state k.
            let mut partial = [zero; 4];
            for k in 0..N {
                let w = windows[k];
                let mut sum = zero;
                for (row, &offset) in lpc.older_wide.iter().zip(&WIDE_OFFSETS) {
                    let v = load8(w[offset..offset + 8].try_into().unwrap());
                    sum = _mm256_add_epi32(sum, _mm256_mul_epi32(v, load8(row)));
                }
                partial[k] = sum;
            }
            // As in `ShortPrediction::predict`, in both 128-bit halves; then
            // the halves are added.
            let moved = [
                _mm256_srli_epi64::<32>(partial[0]),
                _mm256_srli_epi64::<32>(partial[2]),
            ];
            let low = _mm256_blend_epi32::<0b1010_1010>(moved[0], partial[1]);
            let high = _mm256_blend_epi32::<0b1010_1010>(moved[1], partial[3]);
            let sum = _mm256_add_epi32(
                _mm256_unpacklo_epi64(low, high),
                _mm256_unpackhi_epi64(low, high),
            );
            let halves = (_mm256_castsi256_si128(sum), _mm256_extracti128_si256::<1>(sum));
            lpc.finish(_mm_add_epi32(halves.0, halves.1), newest, second)
        }
    }

    /// Window positions of the loads of `ShortPrediction`. The load at
    /// position `o` puts positions `o` and `o + 2` into lanes 0 and 2.
    /// Positions 10 and 11 are in two loads each.
    const OLDER_OFFSETS: [usize; 8] = [0, 1, 4, 5, 8, 9, 10, 11];

    /// The same for the 256-bit loads of `wide::predict`: the load at
    /// position `o` puts positions `o`, `o + 2`, `o + 4` and `o + 6` into
    /// the 32-bit lanes 0, 2, 4 and 6. Positions 6 and 7 are in two loads
    /// each.
    const WIDE_OFFSETS: [usize; 4] = [0, 1, 6, 7];

    /// Short-term prediction of all states at once, state `k` in lane `k`.
    ///
    /// The two newest samples of every state come from registers. The
    /// older ones were stored at least two samples before: an x86 core
    /// forwards a recent store to a load only when the load reads no other
    /// bytes, and a 16-byte load over the newest samples would wait until
    /// those stores are written to the cache.
    struct ShortPrediction {
        /// Lanes 0 and 2 of row `i`: the factors (coefficient `<< 16`) of
        /// the window positions `OLDER_OFFSETS[i]` and `OLDER_OFFSETS[i] + 2`,
        /// so that `_mm_mul_epi32` puts the products into lanes 1 and 3. The
        /// second load of positions 10 and 11 has the factor 0.
        older: [[i32; 4]; 8],
        /// The same for the loads at `WIDE_OFFSETS`: lanes `2 * j` and
        /// `2 * j + 1` of row `i` hold the factor of position
        /// `WIDE_OFFSETS[i] + 2 * j`. The second load of positions 6 and 7
        /// has the factor 0.
        older_wide: [[i32; 8]; 4],
        /// `mulw_factor` of the coefficients of window positions 15 (the
        /// newest sample) and 14.
        newest: __m128i,
        second: __m128i,
        /// Half the order, the constant of the sum, in every lane.
        bias: __m128i,
    }

    impl ShortPrediction {
        #[target_feature(enable = "sse4.1")]
        #[inline]
        fn new(coefs: &ShortPredictionCoefs) -> Self {
            let a = coefs.a_q12;
            // Window position 15 is the newest sample: `a_q12[0]`.
            let factor = |position: usize| match a.get(NSQ_LPC_BUF_LENGTH - 1 - position) {
                Some(&c) => (c as i32) << 16,
                None => 0,
            };
            let mut older = [[0i32; 4]; 8];
            for (i, (row, &offset)) in older.iter_mut().zip(&OLDER_OFFSETS).enumerate() {
                let low = if i < 6 { factor(offset) } else { 0 };
                let high = factor(offset + 2);
                *row = [low, low, high, high];
            }
            let mut older_wide = [[0i32; 8]; 4];
            for (i, (row, &offset)) in older_wide.iter_mut().zip(&WIDE_OFFSETS).enumerate() {
                for (j, pair) in row.as_chunks_mut::<2>().0.iter_mut().enumerate() {
                    let f = if i >= 2 && j == 0 { 0 } else { factor(offset + 2 * j) };
                    *pair = [f; 2];
                }
            }
            Self {
                older,
                older_wide,
                newest: _mm_set1_epi32(factor(NSQ_LPC_BUF_LENGTH - 1)),
                second: _mm_set1_epi32(factor(NSQ_LPC_BUF_LENGTH - 2)),
                bias: _mm_set1_epi32(coefs.bias()),
            }
        }

        /// The prediction of every state, shifted to Q14 as the quantizer
        /// uses it. `windows[k]` holds the 16 newest samples of state `k`;
        /// its last two entries are not read: lane `k` of `newest` and of
        /// `second` hold them.
        ///
        /// # Safety
        ///
        /// The processor must have SSE4.1.
        #[inline(always)]
        unsafe fn predict<const N: usize>(
            &self,
            windows: [&[i32; NSQ_LPC_BUF_LENGTH]; N],
            newest: __m128i,
            second: __m128i,
        ) -> __m128i {
            const { assert!(N <= 4) };
            let zero = _mm_setzero_si128();
            // Lanes 1 and 3 of `partial[k]` hold two partial sums of state k.
            let mut partial = [zero; 4];
            for k in 0..N {
                let w = windows[k];
                let mut sum = zero;
                for (row, &offset) in self.older.iter().zip(&OLDER_OFFSETS) {
                    let v = load(w[offset..offset + 4].try_into().unwrap());
                    sum = _mm_add_epi32(sum, _mm_mul_epi32(v, load(row)));
                }
                partial[k] = sum;
            }
            // Lanes 1 and 3 of states 0 and 1, then of states 2 and 3:
            // [s0 lane 1, s1 lane 1, s0 lane 3, s1 lane 3].
            let low = _mm_blend_epi16::<0b1100_1100>(_mm_srli_epi64::<32>(partial[0]), partial[1]);
            let high = _mm_blend_epi16::<0b1100_1100>(_mm_srli_epi64::<32>(partial[2]), partial[3]);
            let sum = _mm_add_epi32(_mm_unpacklo_epi64(low, high), _mm_unpackhi_epi64(low, high));
            self.finish(sum, newest, second)
        }

        /// The prediction from `older`, the sum of the products of the 14
        /// older samples of every state: adds half the order and the
        /// products of the two newest samples, and shifts to Q14.
        ///
        /// # Safety
        ///
        /// The processor must have SSE4.1.
        #[inline(always)]
        unsafe fn finish(&self, older: __m128i, newest: __m128i, second: __m128i) -> __m128i {
            let two_newest = _mm_add_epi32(mulw(newest, self.newest), mulw(second, self.second));
            _mm_slli_epi32::<4>(_mm_add_epi32(_mm_add_epi32(older, self.bias), two_newest))
        }
    }

    /// `ShortPrediction::predict`, or with `wide` `wide::predict`, on
    /// arrays, for the tests: the prediction of every state from its whole
    /// window. Panics when the processor does not have SSE4.1, or AVX2 for
    /// `wide`.
    #[cfg(test)]
    pub(super) fn short_prediction_lanes<const N: usize>(
        windows: &[[i32; NSQ_LPC_BUF_LENGTH]; N],
        a_q12: &[i16],
        order: usize,
        wide: bool,
    ) -> [i32; N] {
        assert!(crate::compat::x86_has_sse41());
        assert!(!wide || crate::compat::x86_has_avx2());
        // SAFETY: the processor has SSE4.1, and AVX2 for `wide` (checked
        // above).
        unsafe {
            let lpc = ShortPrediction::new(&ShortPredictionCoefs::new(a_q12, order));
            // The last two entries come from the registers only.
            let mut stale = *windows;
            for w in stale.iter_mut() {
                w[NSQ_LPC_BUF_LENGTH - 2..].copy_from_slice(&[0x0bad_f00d, -0x0bad_f00d]);
            }
            let newest = from_lanes::<N>(core::array::from_fn(|k| {
                windows[k][NSQ_LPC_BUF_LENGTH - 1]
            }));
            let second = from_lanes::<N>(core::array::from_fn(|k| {
                windows[k][NSQ_LPC_BUF_LENGTH - 2]
            }));
            let stale = core::array::from_fn(|k| &stale[k]);
            to_lanes(if wide {
                wide::predict::<N>(&lpc, stale, newest, second)
            } else {
                lpc.predict::<N>(stale, newest, second)
            })
        }
    }

    /// One subframe of the delayed-decision quantizer for `N` states, 2 to 4,
    /// on x86-64 with SSE4.1. Same results as `quantize_subframe`.
    ///
    /// `ORDER` is the shaping order when it is known at compile time, or 0.
    #[target_feature(enable = "sse4.1")]
    pub(super) fn quantize_subframe_sse41<const N: usize, const ORDER: usize>(
        nsq: &mut SilkNSQState,
        states: &mut [PathState; N],
        delay_lines: &mut DelayLines,
        x_q10: &[i32],
        pulses: &mut [i8],
        s_ltp_q15: &mut [i32],
        delayed_ga_q10: &mut [i32],
        smpl_buf_idx: &mut i32,
        p: &SubframeParams,
    ) {
        // SAFETY: the processor has SSE4.1, the feature of this function,
        // and that is all `WIDE == false` needs.
        unsafe {
            quantize_subframe_x86::<N, ORDER, false>(
                nsq,
                states,
                delay_lines,
                x_q10,
                pulses,
                s_ltp_q15,
                delayed_ga_q10,
                smpl_buf_idx,
                p,
            )
        }
    }

    /// `quantize_subframe_sse41` for processors with AVX2. The shaping filter
    /// of 3 and 4 states and the short-term prediction run in 256-bit
    /// registers (`wide`). Every other step uses the same instructions in the
    /// VEX encoding, which has a separate destination register and so needs
    /// fewer copies between registers.
    #[target_feature(enable = "avx2")]
    pub(super) fn quantize_subframe_avx2<const N: usize, const ORDER: usize>(
        nsq: &mut SilkNSQState,
        states: &mut [PathState; N],
        delay_lines: &mut DelayLines,
        x_q10: &[i32],
        pulses: &mut [i8],
        s_ltp_q15: &mut [i32],
        delayed_ga_q10: &mut [i32],
        smpl_buf_idx: &mut i32,
        p: &SubframeParams,
    ) {
        // SAFETY: the processor has AVX2, the feature of this function,
        // which includes SSE4.1.
        unsafe {
            quantize_subframe_x86::<N, ORDER, true>(
                nsq,
                states,
                delay_lines,
                x_q10,
                pulses,
                s_ltp_q15,
                delayed_ga_q10,
                smpl_buf_idx,
                p,
            )
        }
    }

    /// The sample loop of `quantize_subframe_sse41` and, with `WIDE`, of
    /// `quantize_subframe_avx2`. The steps are those of
    /// `neon::quantize_subframe_neon`.
    ///
    /// It is inlined into both, so that it is compiled once with the target
    /// features of each.
    ///
    /// # Safety
    ///
    /// The processor must have SSE4.1, and AVX2 when `WIDE` is true.
    #[inline(always)]
    unsafe fn quantize_subframe_x86<const N: usize, const ORDER: usize, const WIDE: bool>(
        nsq: &mut SilkNSQState,
        states: &mut [PathState; N],
        delay_lines: &mut DelayLines,
        x_q10: &[i32],
        pulses: &mut [i8],
        s_ltp_q15: &mut [i32],
        delayed_ga_q10: &mut [i32],
        smpl_buf_idx: &mut i32,
        p: &SubframeParams,
    ) {
        debug_assert!((2..=4).contains(&N));
        let dd = p.decision_delay;
        if !vector_loop_applies(*smpl_buf_idx, s_ltp_q15, delayed_ga_q10, p) {
            quantize_subframe::<N>(
                nsq,
                states,
                delay_lines,
                x_q10,
                pulses,
                s_ltp_q15,
                delayed_ga_q10,
                smpl_buf_idx,
                p,
            );
            return;
        }
        let Some(setup) = SubframeSetup::new(nsq, x_q10, p) else {
            return;
        };
        assert!(ORDER == 0 || ORDER == setup.ar_shp_q13.len());
        let delayed_ga_q10: &mut [i32; DECISION_DELAY] =
            (&mut delayed_ga_q10[..DECISION_DELAY]).try_into().unwrap();
        let length = setup.x_q10.len().min(MAX_SUB_FRAME_LENGTH);
        let levels = LevelVectors::new(&setup.levels);
        let long_term = LongTerm::new(p, &setup);
        let lpc = ShortPrediction::new(&setup.lpc);
        // The two newest samples of every state, for `ShortPrediction`.
        let mut newest_xq_q14 = from_lanes::<N>(core::array::from_fn(|k| {
            states[k].s_lpc_q14[NSQ_LPC_BUF_LENGTH - 1]
        }));
        let mut second_xq_q14 = from_lanes::<N>(core::array::from_fn(|k| {
            states[k].s_lpc_q14[NSQ_LPC_BUF_LENGTH - 2]
        }));
        let rand_multiplier = _mm_set1_epi32(196314165);
        let rand_increment = _mm_set1_epi32(907633515);
        let shp_buf_start = nsq.s_ltp_shp_buf_idx;
        let ltp_buf_start = nsq.s_ltp_buf_idx;

        let shaping_coefs = ShapingCoefs::new(setup.ar_shp_q13, p.warping_q16);
        let mut shaping_rows = load_shaping_rows(states);

        // A store that has no place to go goes to one of these (see
        // `neon::quantize_subframe_neon`).
        let mut spare_lpc = [0i32; NSQ_LPC_BUF_LENGTH];
        let (mut spare_i8, mut spare_i16, mut spare_i32) = (0i8, 0i16, 0i32);
        core::hint::black_box((
            &mut spare_lpc,
            &mut spare_i8,
            &mut spare_i16,
            &mut spare_i32,
        ));

        // The per-state values that change with every sample.
        let mut seed = from_lanes::<N>(core::array::from_fn(|k| states[k].seed));
        let mut seed_init = from_lanes::<N>(core::array::from_fn(|k| states[k].seed_init));
        let mut lf_ar_q14 = from_lanes::<N>(core::array::from_fn(|k| states[k].lf_ar_q14));
        let diff_q14 = from_lanes::<N>(core::array::from_fn(|k| states[k].diff_q14));
        // What the shaping filter needs from the last sample. `diff_q14` of
        // every state is `last.take(last.diff_first, last.diff_second)`.
        let mut last = LastChoice::keep(diff_q14);
        let mut rd_q10: [i32; N] = core::array::from_fn(|k| states[k].rd_q10);
        // The slot of the newest sample of the delay lines, and its
        // `shape_q14` row.
        let mut newest = (*smpl_buf_idx as usize).min(DECISION_DELAY - 1);
        debug_assert!((0..N).all(|k| delay_lines.column(k, newest) == k));
        let mut shape_q14 = load(&delay_lines.shape_q14[newest]);

        for idx in 0..length {
            let i = idx as i32;
            let x = setup.x_q10[idx];

            // Noise shaping feedback of every state. With 2 states, one
            // 128-bit register holds the filter of both.
            let n_ar = if WIDE && N > 2 {
                wide::shaping_filter_both::<ORDER, FIRST_TAPS>(
                    &mut shaping_rows,
                    &last,
                    &shaping_coefs,
                )
            } else {
                shaping_filter_both::<N, ORDER, FIRST_TAPS>(
                    &mut shaping_rows,
                    &last,
                    &shaping_coefs,
                )
            };

            let (ltp_pred_q14, n_ltp_q14) = long_term.at(s_ltp_q15, &nsq.s_ltp_shp_q14, i);

            // Dither: `silk_rand`.
            seed = _mm_add_epi32(_mm_mullo_epi32(seed, rand_multiplier), rand_increment);

            // Short-term prediction.
            let windows: [&[i32; NSQ_LPC_BUF_LENGTH]; N] = core::array::from_fn(|k| {
                states[k].s_lpc_q14[idx..idx + NSQ_LPC_BUF_LENGTH]
                    .try_into()
                    .unwrap()
            });
            let lpc_pred_q14 = if WIDE {
                wide::predict::<N>(&lpc, windows, newest_xq_q14, second_xq_q14)
            } else {
                lpc.predict::<N>(windows, newest_xq_q14, second_xq_q14)
            };

            let c = quantize_sample_x86(
                &levels,
                x,
                ltp_pred_q14,
                n_ltp_q14,
                n_ar,
                lpc_pred_q14,
                lf_ar_q14,
                shape_q14,
                seed,
            );

            // The slot of this sample, and the slot of the sample that is
            // `dd` samples older.
            newest = select(newest == 0, DECISION_DELAY - 1, newest.wrapping_sub(1))
                .min(DECISION_DELAY - 1);
            let oldest = newest + dd as usize;
            let oldest = select(
                oldest >= DECISION_DELAY,
                oldest.wrapping_sub(DECISION_DELAY),
                oldest,
            )
            .min(DECISION_DELAY - 1);

            // The survivors are chosen on scalars, as in the portable loop.
            let added0: [i32; N] = to_lanes(c.rd_added_q10[0]);
            let added1: [i32; N] = to_lanes(c.rd_added_q10[1]);
            let mut rd0: [i32; N] = core::array::from_fn(|k| rd_q10[k].wrapping_add(added0[k]));
            let mut rd1: [i32; N] = core::array::from_fn(|k| rd_q10[k].wrapping_add(added1[k]));
            let rand_state: [i32; N] = core::array::from_fn(|k| {
                delay_lines.rand_state[oldest][delay_lines.column(k, oldest)]
            });
            let sv = select_survivors(&mut rd0, &mut rd1, &rand_state);

            // Replace the worst state by a copy of the best one that takes
            // the second candidate.
            copy_history(states, delay_lines, &mut spare_lpc, &sv, idx);
            let replaced_lane = select(sv.replace, sv.rd_max_ind, 4);
            last = LastChoice::new(c.diff_q14[0], c.diff_q14[1], replaced_lane, sv.rd_min_ind);
            seed = last.take(seed, seed);
            seed_init = last.take(seed_init, seed_init);
            let q_q10 = last.take(c.q_q10[0], c.q_q10[1]);
            let xq_q14 = last.take(c.xq_q14[0], c.xq_q14[1]);
            // The replaced state continues the history of the best state.
            second_xq_q14 = last.take(newest_xq_q14, newest_xq_q14);
            newest_xq_q14 = xq_q14;
            let lpc_exc_q14 = last.take(c.lpc_exc_q14[0], c.lpc_exc_q14[1]);
            shape_q14 = last.take(c.s_ltp_shp_q14[0], c.s_ltp_shp_q14[1]);
            lf_ar_q14 = last.take(c.lf_ar_q14[0], c.lf_ar_q14[1]);
            for k in 0..N {
                let replaced = sv.replace && k == sv.rd_max_ind;
                rd_q10[k] = select(replaced, sv.rd_min_q10, rd0[k]);
            }

            // The outputs that are `dd` samples old, from the state with the
            // lowest cost. The first `dd` samples of a frame have none.
            let active = p.subfr > 0 || i >= dd;
            let column = delay_lines.column(sv.winner_ind, oldest);
            store_if(
                pulses,
                (p.pulses_offset + i - dd) as isize,
                active,
                &mut spare_i8,
                silk_rshift_round(delay_lines.q_q10[oldest][column], 10) as i8,
            );
            store_if(
                &mut nsq.xq,
                (p.xq_ptr + i - dd) as isize,
                active,
                &mut spare_i16,
                silk_sat16(silk_rshift_round(
                    silk_smulww(delay_lines.xq_q14[oldest][column], delayed_ga_q10[oldest]),
                    8,
                )) as i16,
            );
            store_if(
                &mut nsq.s_ltp_shp_q14,
                (shp_buf_start + i - dd) as isize,
                active,
                &mut spare_i32,
                delay_lines.shape_q14[oldest][column],
            );
            store_if(
                s_ltp_q15,
                (ltp_buf_start + i - dd) as isize,
                active,
                &mut spare_i32,
                delay_lines.pred_q15[oldest][column],
            );

            // Append the chosen candidate to every state: one row per delay
            // line. The lanes from `N` on of a row are never read.
            seed = _mm_add_epi32(seed, round_shift::<9>(q_q10));
            store(&mut delay_lines.xq_q14[newest], xq_q14);
            store(&mut delay_lines.q_q10[newest], q_q10);
            store(
                &mut delay_lines.pred_q15[newest],
                _mm_slli_epi32::<1>(lpc_exc_q14),
            );
            store(&mut delay_lines.shape_q14[newest], shape_q14);
            store(&mut delay_lines.rand_state[newest], seed);
            let xq: [i32; N] = to_lanes(xq_q14);
            for k in 0..N {
                states[k].s_lpc_q14[NSQ_LPC_BUF_LENGTH + idx] = xq[k];
                delay_lines.column[k][newest] = k as u8;
            }
            delayed_ga_q10[newest] = setup.gain_q10;
        }
        *smpl_buf_idx = newest as i32;
        nsq.s_ltp_shp_buf_idx = shp_buf_start + length as i32;
        nsq.s_ltp_buf_idx = ltp_buf_start + length as i32;

        let perm = load_table(&PERM[last.replaced_lane][last.best]);
        store_shaping_rows(states, &shaping_rows, perm);
        let seed: [i32; N] = to_lanes(seed);
        let seed_init: [i32; N] = to_lanes(seed_init);
        let lf_ar_q14: [i32; N] = to_lanes(lf_ar_q14);
        let diff_q14: [i32; N] = to_lanes(last.take(last.diff_first, last.diff_second));
        for k in 0..N {
            let state = &mut states[k];
            state.seed = seed[k];
            state.seed_init = seed_init[k];
            state.lf_ar_q14 = lf_ar_q14[k];
            state.diff_q14 = diff_q14[k];
            state.rd_q10 = rd_q10[k];
            // The newest samples become the prediction history of the next
            // subframe.
            state
                .s_lpc_q14
                .copy_within(length..length + NSQ_LPC_BUF_LENGTH, 0);
        }
    }
}

/// One subframe with no state at all. This is not a setting the encoder
/// uses; it only keeps what the function did before for a state count of 0:
/// the delayed outputs are taken from state 0, which nothing updates.
fn quantize_subframe_without_states(
    nsq: &mut SilkNSQState,
    delay_lines: &DelayLines,
    pulses: &mut [i8],
    s_ltp_q15: &mut [i32],
    delayed_ga_q10: &mut [i32],
    smpl_buf_idx: &mut i32,
    p: &SubframeParams,
) {
    let gain_q10 = silk_rshift(p.gain_q16, 6);
    for i in 0..p.length {
        *smpl_buf_idx = (*smpl_buf_idx - 1 + DECISION_DELAY as i32) % DECISION_DELAY as i32;
        let last_smple_idx = ((*smpl_buf_idx + p.decision_delay) % DECISION_DELAY as i32) as usize;
        if p.subfr > 0 || i >= p.decision_delay {
            write_delayed_outputs(
                nsq,
                delay_lines,
                0,
                pulses,
                s_ltp_q15,
                delayed_ga_q10,
                p,
                i,
                last_smple_idx,
            );
        }
        nsq.s_ltp_shp_buf_idx += 1;
        nsq.s_ltp_buf_idx += 1;
        let smpl_idx = (*smpl_buf_idx as usize).min(DECISION_DELAY - 1);
        delayed_ga_q10[smpl_idx] = gain_q10;
    }
}

/// Runs one subframe with the sample loop for `n_states` states. The sample
/// loops are compiled once per state count, and on aarch64 and x86-64 also
/// once per shaping order the encoder uses with that count; this is the only
/// place that chooses between them. On x86-64 the vector loop needs SSE4.1,
/// checked once per subframe; without it the portable loop runs.
fn quantize_subframe_dispatch(
    nsq: &mut SilkNSQState,
    states: &mut [PathState; NSQ_MAX_STATES_OPERATING],
    delay_lines: &mut DelayLines,
    n_states: usize,
    x_q10: &[i32],
    pulses: &mut [i8],
    s_ltp_q15: &mut [i32],
    delayed_ga_q10: &mut [i32],
    smpl_buf_idx: &mut i32,
    p: &SubframeParams,
) {
    macro_rules! portable {
        ($n:literal) => {
            quantize_subframe::<$n>(
                nsq,
                states.first_chunk_mut::<$n>().unwrap(),
                delay_lines,
                x_q10,
                pulses,
                s_ltp_q15,
                delayed_ga_q10,
                smpl_buf_idx,
                p,
            )
        };
    }
    // The shaping orders the encoder uses with each state count get their
    // own copy of the loop, with the filter taps unrolled.
    #[cfg(target_arch = "aarch64")]
    macro_rules! fastest {
        ($n:literal, $order:literal) => {
            // SAFETY: NEON is available: src/lib.rs refuses to build for
            // aarch64 without it.
            unsafe {
                neon::quantize_subframe_neon::<$n, $order>(
                    nsq,
                    states.first_chunk_mut::<$n>().unwrap(),
                    delay_lines,
                    x_q10,
                    pulses,
                    s_ltp_q15,
                    delayed_ga_q10,
                    smpl_buf_idx,
                    p,
                )
            }
        };
    }
    #[cfg(target_arch = "x86_64")]
    let (avx2, sse41) = (
        crate::compat::x86_has_avx2(),
        crate::compat::x86_has_sse41(),
    );
    #[cfg(target_arch = "x86_64")]
    macro_rules! x86_loop {
        ($loop:ident, $n:literal, $order:literal) => {
            x86::$loop::<$n, $order>(
                nsq,
                states.first_chunk_mut::<$n>().unwrap(),
                delay_lines,
                x_q10,
                pulses,
                s_ltp_q15,
                delayed_ga_q10,
                smpl_buf_idx,
                p,
            )
        };
    }
    #[cfg(target_arch = "x86_64")]
    macro_rules! fastest {
        ($n:literal, $order:literal) => {
            if avx2 {
                // SAFETY: the processor has AVX2 (detected above).
                unsafe { x86_loop!(quantize_subframe_avx2, $n, $order) }
            } else if sse41 {
                // SAFETY: the processor has SSE4.1 (detected above).
                unsafe { x86_loop!(quantize_subframe_sse41, $n, $order) }
            } else {
                portable!($n)
            }
        };
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    macro_rules! fastest {
        ($n:literal, $order:literal) => {
            portable!($n)
        };
    }
    match (n_states, p.shaping_lpc_order) {
        (0, _) => quantize_subframe_without_states(
            nsq,
            delay_lines,
            pulses,
            s_ltp_q15,
            delayed_ga_q10,
            smpl_buf_idx,
            p,
        ),
        (1, _) => portable!(1),
        (2, 12) => fastest!(2, 12),
        (2, 14) => fastest!(2, 14),
        (2, 16) => fastest!(2, 16),
        (2, _) => fastest!(2, 0),
        (3, 20) => fastest!(3, 20),
        (3, _) => fastest!(3, 0),
        (_, 24) => fastest!(4, 24),
        _ => fastest!(4, 0),
    }
}

/// Delayed-decision noise-shaping quantizer for one subframe, on states in
/// the layout of libopus.
///
/// `n_states_delayed_decision` selects how many of the states in `ps_del_dec`
/// take part (at most `NSQ_MAX_STATES_OPERATING`). `silk_nsq_del_dec` does not
/// call this function: it keeps the states in the internal layout for the
/// whole frame. This function converts on entry and on exit.
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
    let p = SubframeParams {
        signal_type,
        pulses_offset,
        xq_ptr,
        a_q12,
        b_q14,
        ar_shp_q13,
        lag,
        harm_shape_fir_packed_q14,
        tilt_q14,
        lf_shp_q14,
        gain_q16,
        lambda_q10,
        offset_q10,
        length,
        subfr,
        shaping_lpc_order,
        predict_lpc_order,
        warping_q16,
        decision_delay,
    };
    let n_states = (n_states_delayed_decision as usize).min(NSQ_MAX_STATES_OPERATING);
    // Without any state the outputs are still read from state 0.
    let converted = &mut ps_del_dec[..n_states.max(1)];

    let mut states = [PathState::ZERO; NSQ_MAX_STATES_OPERATING];
    let mut delay_lines = DelayLines::new();
    for (k, dd) in converted.iter().enumerate() {
        states[k] = PathState {
            s_lpc_q14: dd.s_lpc_q14,
            s_ar2_q14: dd.s_ar2_q14,
            lf_ar_q14: dd.lf_ar_q14,
            diff_q14: dd.diff_q14,
            seed: dd.seed,
            seed_init: dd.seed_init,
            rd_q10: dd.rd_q10,
        };
        for slot in 0..DECISION_DELAY {
            delay_lines.rand_state[slot][k] = dd.rand_state[slot];
            delay_lines.q_q10[slot][k] = dd.q_q10[slot];
            delay_lines.xq_q14[slot][k] = dd.xq_q14[slot];
            delay_lines.pred_q15[slot][k] = dd.pred_q15[slot];
            delay_lines.shape_q14[slot][k] = dd.shape_q14[slot];
        }
    }

    quantize_subframe_dispatch(
        nsq,
        &mut states,
        &mut delay_lines,
        n_states,
        x_q10,
        pulses,
        s_ltp_q15,
        delayed_ga_q10,
        smpl_buf_idx,
        &p,
    );

    for (k, dd) in converted.iter_mut().enumerate() {
        let state = &states[k];
        // Only the 16 entries from `idx` on are copied when one state
        // replaces another, so the entries of `s_lpc_q14` below the last 16
        // of the subframe are not meaningful afterwards.
        dd.s_lpc_q14 = state.s_lpc_q14;
        dd.s_ar2_q14 = state.s_ar2_q14;
        dd.lf_ar_q14 = state.lf_ar_q14;
        dd.diff_q14 = state.diff_q14;
        dd.seed = state.seed;
        dd.seed_init = state.seed_init;
        dd.rd_q10 = state.rd_q10;
        for slot in 0..DECISION_DELAY {
            let column = delay_lines.column(k, slot);
            dd.rand_state[slot] = delay_lines.rand_state[slot][column];
            dd.q_q10[slot] = delay_lines.q_q10[slot][column];
            dd.xq_q14[slot] = delay_lines.xq_q14[slot][column];
            dd.pred_q15[slot] = delay_lines.pred_q15[slot][column];
            dd.shape_q14[slot] = delay_lines.shape_q14[slot][column];
        }
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
    let mut states = [PathState::ZERO; NSQ_MAX_STATES_OPERATING];
    let mut delay_lines = DelayLines::new();

    let mut lag = ps_nsq.lag_prev;

    let n_states = (ps_common.n_states_delayed_decision as usize).min(NSQ_MAX_STATES_OPERATING);

    for (k, state) in states[..n_states].iter_mut().enumerate() {
        state.seed = (k as i32 + ps_indices.seed as i32) & 3;
        state.seed_init = state.seed;
        state.rd_q10 = 0;
        state.lf_ar_q14 = ps_nsq.s_lf_ar_q14;
        state.diff_q14 = ps_nsq.s_diff_shp_q14;

        if ps_common.ltp_mem_length > 0 {
            delay_lines.shape_q14[0][k] =
                ps_nsq.s_ltp_shp_q14[ps_common.ltp_mem_length as usize - 1];
        }
        state.s_lpc_q14[..NSQ_LPC_BUF_LENGTH]
            .copy_from_slice(&ps_nsq.s_lpc_q14[..NSQ_LPC_BUF_LENGTH]);
        state.s_ar2_q14.copy_from_slice(&ps_nsq.s_ar2_q14);
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
        let a_q12 =
            &pred_coef_q12[((k >> 1) | (1 - lsf_interpolation_flag as usize)) * MAX_LPC_ORDER..];
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
                    let mut rd_min_q10 = states[0].rd_q10;
                    let mut winner_ind = 0;
                    for i in 1..n_states {
                        if states[i].rd_q10 < rd_min_q10 {
                            rd_min_q10 = states[i].rd_q10;
                            winner_ind = i;
                        }
                    }
                    for i in 0..n_states {
                        if i != winner_ind {
                            states[i].rd_q10 = states[i].rd_q10.saturating_add(i32::MAX >> 4);
                        }
                    }

                    let mut last_smple_idx =
                        (smpl_buf_idx + decision_delay) % DECISION_DELAY as i32;
                    for i in 0..decision_delay {
                        last_smple_idx =
                            (last_smple_idx - 1 + DECISION_DELAY as i32) % DECISION_DELAY as i32;
                        let pulse_idx = (pulses_ptr as i32 + i - decision_delay) as isize;
                        let xq_idx = (xq_ptr as i32 + i - decision_delay) as isize;
                        let shp_idx = (ps_nsq.s_ltp_shp_buf_idx + i - decision_delay) as isize;
                        if pulse_idx >= 0 && xq_idx >= 0 && shp_idx >= 0 {
                            let slot = last_smple_idx as usize;
                            let column = delay_lines.column(winner_ind, slot);
                            pulses[pulse_idx as usize] =
                                silk_rshift_round(delay_lines.q_q10[slot][column], 10) as i8;
                            ps_nsq.xq[xq_idx as usize] = silk_sat16(silk_rshift_round(
                                silk_smulww(delay_lines.xq_q14[slot][column], gains_q16[1]),
                                14,
                            )) as i16;
                            ps_nsq.s_ltp_shp_q14[shp_idx as usize] =
                                delay_lines.shape_q14[slot][column];
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

                if start_idx + filter_len > s_ltp.len() || xq_start + filter_len > ps_nsq.xq.len() {
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
            &mut states,
            &mut delay_lines,
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

        let p = SubframeParams {
            signal_type: ps_indices.signal_type as i32,
            pulses_offset: pulses_ptr as i32,
            xq_ptr: xq_ptr as i32,
            a_q12,
            b_q14,
            ar_shp_q13,
            lag,
            harm_shape_fir_packed_q14,
            tilt_q14: tilt_q14[k],
            lf_shp_q14: lf_shp_q14[k],
            gain_q16: gains_q16[k],
            lambda_q10,
            offset_q10,
            length: ps_common.subfr_length,
            subfr: subfr_nsq,
            shaping_lpc_order: ps_common.shaping_lpc_order,
            predict_lpc_order: ps_common.predict_lpc_order,
            warping_q16: ps_common.warping_q16,
            decision_delay,
        };
        quantize_subframe_dispatch(
            ps_nsq,
            &mut states,
            &mut delay_lines,
            n_states,
            &x_sc_q10,
            pulses,
            &mut s_ltp_q15,
            &mut delayed_ga_q10,
            &mut smpl_buf_idx,
            &p,
        );

        x_ptr += ps_common.subfr_length as usize;
        pulses_ptr += ps_common.subfr_length as usize;
        xq_ptr += ps_common.subfr_length as usize;
        subfr_nsq += 1;
    }

    let mut rd_min_q10 = states[0].rd_q10;
    let mut winner_ind = 0;
    for k in 1..n_states {
        if states[k].rd_q10 < rd_min_q10 {
            rd_min_q10 = states[k].rd_q10;
            winner_ind = k;
        }
    }

    let winner = &states[winner_ind];
    ps_nsq.s_lf_ar_q14 = winner.lf_ar_q14;
    ps_nsq.s_diff_shp_q14 = winner.diff_q14;
    ps_nsq.lag_prev = pitch_l[ps_common.nb_subfr as usize - 1];

    let gain_q10_final = silk_rshift(gains_q16[ps_common.nb_subfr as usize - 1], 6);
    let mut last_smple_idx = smpl_buf_idx + decision_delay;
    for i in 0..decision_delay {
        last_smple_idx = (last_smple_idx - 1 + DECISION_DELAY as i32) % DECISION_DELAY as i32;
        let slot = last_smple_idx as usize;
        let pulse_idx = (pulses_ptr as i32 + i - decision_delay) as isize;
        let xq_idx = (xq_ptr as i32 + i - decision_delay) as isize;
        if pulse_idx >= 0 && (pulse_idx as usize) < pulses.len() {
            let column = delay_lines.column(winner_ind, slot);
            pulses[pulse_idx as usize] =
                silk_rshift_round(delay_lines.q_q10[slot][column], 10) as i8;
        }
        if xq_idx >= 0 && (xq_idx as usize) < ps_nsq.xq.len() {
            let column = delay_lines.column(winner_ind, slot);
            ps_nsq.xq[xq_idx as usize] = silk_sat16(silk_rshift_round(
                silk_smulww(delay_lines.xq_q14[slot][column], gain_q10_final),
                8,
            )) as i16;
        }
        let shp_idx = (ps_nsq.s_ltp_shp_buf_idx - decision_delay + i) as isize;
        if shp_idx >= 0 && (shp_idx as usize) < ps_nsq.s_ltp_shp_q14.len() {
            let column = delay_lines.column(winner_ind, slot);
            ps_nsq.s_ltp_shp_q14[shp_idx as usize] = delay_lines.shape_q14[slot][column];
        }
    }

    let subfr_len = ps_common.subfr_length as usize;
    ps_nsq.s_lpc_q14[..NSQ_LPC_BUF_LENGTH]
        .copy_from_slice(&winner.s_lpc_q14[subfr_len..subfr_len + NSQ_LPC_BUF_LENGTH]);
    ps_nsq.s_ar2_q14.copy_from_slice(&winner.s_ar2_q14);

    let ltp_mem_len = ps_common.ltp_mem_length as usize;
    let frame_len = ps_common.frame_length as usize;
    ps_nsq.xq.copy_within(frame_len..frame_len + ltp_mem_len, 0);
    ps_nsq
        .s_ltp_shp_q14
        .copy_within(frame_len..frame_len + ltp_mem_len, 0);

    winner.seed_init
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Small deterministic generator (splitmix64).
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

        /// A value of at most `bits` bits plus sign, or, one time in eight,
        /// one of the values at the edges of `i32`.
        fn value(&mut self, bits: u32) -> i32 {
            const EDGES: [i32; 6] = [i32::MIN, i32::MIN + 1, -1, 0, 1, i32::MAX];
            if self.range(0, 7) == 0 {
                EDGES[self.range(0, 5) as usize]
            } else {
                self.range(-(1i64 << bits), 1i64 << bits) as i32
            }
        }

        fn coef(&mut self) -> i16 {
            const EDGES: [i16; 5] = [i16::MIN, -1, 0, 1, i16::MAX];
            if self.range(0, 7) == 0 {
                EDGES[self.range(0, 4) as usize]
            } else {
                self.range(i16::MIN as i64, i16::MAX as i64) as i16
            }
        }
    }

    #[test]
    fn per_state_keeps_every_value() {
        let mut rng = Rng(1);
        for _ in 0..200 {
            let v4: [i32; 4] = core::array::from_fn(|_| rng.value(30));
            assert_eq!(PerState::pack(v4).unpack(), v4);
            let v3 = [v4[0], v4[1], v4[2]];
            assert_eq!(PerState::pack(v3).unpack(), v3);
            assert_eq!(PerState::pack(v3).0 >> 96, 0);
            let v2 = [v4[3], v4[0]];
            assert_eq!(PerState::pack(v2).unpack(), v2);
            let v1 = [v4[2]];
            assert_eq!(PerState::pack(v1).unpack(), v1);
        }
    }

    /// The prediction as libopus writes it: one tap after the other.
    fn prediction_by_definition(window: &[i32; MAX_LPC_ORDER], a_q12: &[i16], order: usize) -> i32 {
        let mut out = (order as i32) >> 1;
        for j in 0..order {
            out = silk_smlawb(out, window[MAX_LPC_ORDER - 1 - j], a_q12[j] as i32);
        }
        out
    }

    #[test]
    fn short_prediction_kernels_match_the_definition() {
        let mut rng = Rng(2);
        for case in 0..4000 {
            // Every order, also the ones the codec does not use.
            let order = case % (MAX_LPC_ORDER + 1);
            let window: [i32; MAX_LPC_ORDER] = core::array::from_fn(|_| rng.value(30));
            let mut a_q12 = [0i16; MAX_LPC_ORDER];
            for a in a_q12.iter_mut() {
                *a = rng.coef();
            }
            let want = prediction_by_definition(&window, &a_q12, order);
            let coefs = ShortPredictionCoefs::new(&a_q12, order);
            assert_eq!(
                short_prediction_scalar(&window, &a_q12[..order]),
                want,
                "scalar, order {order}"
            );
            assert_eq!(coefs.predict(&window), want, "dispatch, order {order}");
            #[cfg(target_arch = "aarch64")]
            assert_eq!(
                short_prediction_neon(&window, &coefs.reversed_q27, order),
                want,
                "NEON, order {order}"
            );
        }
    }

    /// The shaping filter of one state as libopus writes it (two taps per
    /// pass, `tmp1` and `tmp2` alternating).
    fn shaping_filter_by_definition(
        s_ar2_q14: &mut [i32; MAX_SHAPE_LPC_ORDER],
        diff_q14: i32,
        ar_shp_q13: &[i16],
        warping_q16: i32,
    ) -> i32 {
        let order = ar_shp_q13.len();
        let mut tmp2 = silk_smlawb(diff_q14, s_ar2_q14[0], warping_q16);
        let mut tmp1 = silk_smlawb(
            s_ar2_q14[0],
            silk_sub32_ovflw(s_ar2_q14[1], tmp2),
            warping_q16,
        );
        s_ar2_q14[0] = tmp2;
        let mut n_ar_q14 = silk_rshift(order as i32, 1);
        n_ar_q14 = silk_smlawb(n_ar_q14, tmp2, ar_shp_q13[0] as i32);
        for j in (2..order).step_by(2) {
            tmp2 = silk_smlawb(
                s_ar2_q14[j - 1],
                silk_sub32_ovflw(s_ar2_q14[j], tmp1),
                warping_q16,
            );
            s_ar2_q14[j - 1] = tmp1;
            n_ar_q14 = silk_smlawb(n_ar_q14, tmp1, ar_shp_q13[j - 1] as i32);
            tmp1 = silk_smlawb(
                s_ar2_q14[j],
                silk_sub32_ovflw(s_ar2_q14[j + 1], tmp2),
                warping_q16,
            );
            s_ar2_q14[j] = tmp2;
            n_ar_q14 = silk_smlawb(n_ar_q14, tmp2, ar_shp_q13[j] as i32);
        }
        s_ar2_q14[order - 1] = tmp1;
        silk_smlawb(n_ar_q14, tmp1, ar_shp_q13[order - 1] as i32)
    }

    fn check_shaping_filter<const N: usize>(rng: &mut Rng) {
        for case in 0..600 {
            let order = 2 * (1 + case % (MAX_SHAPE_LPC_ORDER / 2));
            let mut coefs = [0i16; MAX_SHAPE_LPC_ORDER];
            for c in coefs.iter_mut() {
                *c = rng.coef();
            }
            let warping_q16 = if case % 5 == 0 { 0 } else { rng.coef() as i32 };
            let mut states: ShapingStates<N> =
                core::array::from_fn(|_| core::array::from_fn(|_| rng.value(28)));
            let diff: [i32; N] = core::array::from_fn(|_| rng.value(28));
            // Sometimes one state reads the filter state of another one.
            let mut source: [u8; N] = core::array::from_fn(|k| k as u8);
            if N > 1 && case % 2 == 1 {
                let dst = rng.range(0, N as i64 - 1) as usize;
                source[dst] = rng.range(0, N as i64 - 1) as u8;
            }

            let mut want_states = states;
            let mut want_n_ar = [0i32; N];
            for k in 0..N {
                // The filter state above the order stays the state's own.
                let mut s = states[source[k] as usize];
                s[order..].copy_from_slice(&states[k][order..]);
                want_n_ar[k] =
                    shaping_filter_by_definition(&mut s, diff[k], &coefs[..order], warping_q16);
                want_states[k] = s;
            }

            // Only the vector checks below read it.
            #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
            let start_states = states;
            let n_ar = warped_shaping_filter(
                &mut states,
                source,
                PerState::pack(diff),
                &coefs[..order],
                warping_q16,
            )
            .unpack();
            assert_eq!(n_ar, want_n_ar, "{N} states, order {order}, case {case}");
            assert_eq!(
                states, want_states,
                "{N} states, order {order}, case {case}"
            );

            // The NEON filter on rows, with the order known at run time and,
            // for the orders the encoder uses, at compile time. Each also
            // with some taps computed for both candidates.
            #[cfg(target_arch = "aarch64")]
            {
                macro_rules! check_neon_with {
                    ($order:literal, $first:literal, $both:literal) => {{
                        for keep_lanes in [false, true] {
                            let mut states = start_states;
                            let n_ar = neon::shaping_filter_rows_lanes::<N, $order, $first>(
                                &mut states,
                                source,
                                diff,
                                &coefs[..order],
                                warping_q16,
                                $both,
                                keep_lanes,
                            );
                            let what = format!(
                                "NEON {}, both {} {}, keep {keep_lanes}, {N} states, order {order}, case {case}",
                                $order, $both, $first
                            );
                            assert_eq!(n_ar, want_n_ar, "{what}");
                            assert_eq!(states, want_states, "{what}");
                        }
                    }};
                }
                macro_rules! check_neon {
                    ($order:literal) => {{
                        check_neon_with!($order, 0, false);
                        check_neon_with!($order, 0, true);
                        check_neon_with!($order, 1, true);
                        check_neon_with!($order, 2, true);
                        check_neon_with!($order, 5, true);
                        check_neon_with!($order, 8, true);
                    }};
                }
                check_neon!(0);
                match order {
                    12 => check_neon!(12),
                    14 => check_neon!(14),
                    16 => check_neon!(16),
                    20 => check_neon!(20),
                    24 => check_neon!(24),
                    _ => {}
                }
            }

            // The same for the x86 filters.
            #[cfg(target_arch = "x86_64")]
            if crate::compat::x86_has_sse41() {
                // The 128-bit filter, and the 256-bit one where the
                // processor has AVX2.
                let widths: &[bool] = if crate::compat::x86_has_avx2() {
                    &[false, true]
                } else {
                    &[false]
                };
                macro_rules! check_x86_with {
                    ($order:literal, $first:literal, $both:literal) => {{
                        for &wide in widths {
                            let mut states = start_states;
                            let n_ar = x86::shaping_filter_rows_lanes::<N, $order, $first>(
                                &mut states,
                                source,
                                diff,
                                &coefs[..order],
                                warping_q16,
                                $both,
                                wide,
                            );
                            let what = format!(
                                "x86 {}, both {} {}, wide {wide}, {N} states, order {order}, case {case}",
                                $order, $both, $first
                            );
                            assert_eq!(n_ar, want_n_ar, "{what}");
                            assert_eq!(states, want_states, "{what}");
                        }
                    }};
                }
                macro_rules! check_x86 {
                    ($order:literal) => {{
                        check_x86_with!($order, 0, false);
                        check_x86_with!($order, 0, true);
                        check_x86_with!($order, 1, true);
                        check_x86_with!($order, 2, true);
                        check_x86_with!($order, 4, true);
                        check_x86_with!($order, 5, true);
                        check_x86_with!($order, 8, true);
                    }};
                }
                check_x86!(0);
                match order {
                    12 => check_x86!(12),
                    14 => check_x86!(14),
                    16 => check_x86!(16),
                    20 => check_x86!(20),
                    24 => check_x86!(24),
                    _ => {}
                }
            }
        }
    }

    #[test]
    fn shaping_filter_matches_the_definition() {
        let mut rng = Rng(3);
        check_shaping_filter::<1>(&mut rng);
        check_shaping_filter::<2>(&mut rng);
        check_shaping_filter::<3>(&mut rng);
        check_shaping_filter::<4>(&mut rng);
    }

    /// The survivor selection as libopus writes it
    /// (`silk_noise_shape_quantizer_del_dec`, silk/NSQ_del_dec.c 565-604):
    /// `(Winner_ind, RDmax_ind, RDmin_ind, RDmin_Q10, replace)`. The costs
    /// must leave room for the penalty: libopus adds it without saturation.
    fn select_survivors_by_definition<const N: usize>(
        rd0: &mut [i32; N],
        rd1: &mut [i32; N],
        rand_state: &[i32; N],
    ) -> (usize, usize, usize, i32, bool) {
        // Find winner.
        let mut rd_min_q10 = rd0[0];
        let mut winner_ind = 0;
        for k in 1..N {
            if rd0[k] < rd_min_q10 {
                rd_min_q10 = rd0[k];
                winner_ind = k;
            }
        }
        // Increase RD values of expired states.
        let winner_rand_state = rand_state[winner_ind];
        for k in 0..N {
            if rand_state[k] != winner_rand_state {
                rd0[k] += i32::MAX >> 4;
                rd1[k] += i32::MAX >> 4;
            }
        }
        // Find worst in first set and best in second set.
        let mut rd_max_q10 = rd0[0];
        let mut rd_min_q10 = rd1[0];
        let (mut rd_max_ind, mut rd_min_ind) = (0, 0);
        for k in 1..N {
            if rd0[k] > rd_max_q10 {
                rd_max_q10 = rd0[k];
                rd_max_ind = k;
            }
            if rd1[k] < rd_min_q10 {
                rd_min_q10 = rd1[k];
                rd_min_ind = k;
            }
        }
        (winner_ind, rd_max_ind, rd_min_ind, rd_min_q10, rd_min_q10 < rd_max_q10)
    }

    fn check_survivors<const N: usize>(rng: &mut Rng) {
        for case in 0..4000 {
            // Costs from a few values only, so that many are equal, and
            // dither states from two or three values: ties and states on
            // other paths in most cases. Every cost leaves room for the
            // penalty.
            let top = [3, 40, 1 << 20, (i32::MAX - (i32::MAX >> 4)) as i64][case % 4];
            let mut rd0: [i32; N] = core::array::from_fn(|_| rng.range(0, top) as i32);
            let mut rd1: [i32; N] = core::array::from_fn(|_| rng.range(0, top) as i32);
            let paths = 1 + case % 3;
            let rand_state: [i32; N] = core::array::from_fn(|_| rng.range(0, paths as i64 - 1) as i32);
            let (mut want0, mut want1) = (rd0, rd1);
            let want = select_survivors_by_definition(&mut want0, &mut want1, &rand_state);
            let sv = select_survivors(&mut rd0, &mut rd1, &rand_state);
            let what = format!("{N} states, case {case}: {rd0:?} {rd1:?} {rand_state:?}");
            assert_eq!((sv.winner_ind, sv.rd_max_ind, sv.rd_min_ind, sv.rd_min_q10, sv.replace), want, "{what}");
            assert_eq!((rd0, rd1), (want0, want1), "{what}");
        }
    }

    #[test]
    fn survivors_keep_the_lowest_index_on_ties() {
        let mut rng = Rng(5);
        check_survivors::<1>(&mut rng);
        check_survivors::<2>(&mut rng);
        check_survivors::<3>(&mut rng);
        check_survivors::<4>(&mut rng);

        // Where libopus's addition would overflow, the penalty saturates:
        // the cost of a state on another path never wraps around to a
        // small value.
        let (mut rd0, mut rd1) = ([0, i32::MAX - 5], [1, i32::MAX - 1]);
        select_survivors(&mut rd0, &mut rd1, &[1, 2]);
        assert!(rd0[1] >= i32::MAX - 5 && rd1[1] >= i32::MAX - 1, "{rd0:?} {rd1:?}");
    }

    /// The tests of the vector sample loops: `mod neon` on aarch64, `mod x86`
    /// on x86-64. On an x86-64 processor without SSE4.1 they check nothing.
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    mod simd_tests {
        #[cfg(target_arch = "aarch64")]
        use super::super::neon::*;
        #[cfg(target_arch = "x86_64")]
        use super::super::x86::*;
        use super::super::*;
        use super::Rng;

        /// Whether the processor runs the vector loop of this target.
        fn simd_available() -> bool {
            cfg!(target_arch = "aarch64") || crate::compat::x86_has_sse41()
        }

        #[test]
        fn level_decision_matches_scalar() {
            if !simd_available() {
                return;
            }
            let mut rng = Rng(4);
            for case in 0..20000 {
                // Small values exercise the levels around zero, large ones the
                // limiter and the saturating steps.
                let bits = [8, 12, 16, 22, 30][case % 5];
                let lambda_q10 = rng.range(-5000, 70000) as i32;
                let lv = LevelParams {
                    tilt_q14: rng.value(16),
                    lf_shp_q14: rng.value(30),
                    lambda_q10,
                    offset_q10: [32, 100, 240, 0, -100][rng.range(0, 4) as usize],
                    rdo_offset: if lambda_q10 > 2048 {
                        lambda_q10 / 2 - 512
                    } else {
                        0
                    },
                };
                // The scalar function subtracts and adds with overflow checks
                // in a debug build: keep `x` and the cost away from the edges.
                let x_q10 = rng.range(-(1 << 26), 1 << 26) as i32;
                let ltp_pred_q14 = rng.value(bits);
                let n_ltp_q14 = rng.value(bits);
                let n_ar: [i32; 4] = core::array::from_fn(|_| rng.value(bits));
                let lpc_pred_q14: [i32; 4] = core::array::from_fn(|_| rng.value(bits));
                let lf_ar_q14: [i32; 4] = core::array::from_fn(|_| rng.value(bits));
                let shape_q14: [i32; 4] = core::array::from_fn(|_| rng.value(bits));
                let seed: [i32; 4] = core::array::from_fn(|_| rng.value(30));
                let rd_q10: [i32; 4] = core::array::from_fn(|_| rng.range(0, 1 << 28) as i32);

                let got = level_decision_lanes(
                    &lv,
                    x_q10,
                    ltp_pred_q14,
                    n_ltp_q14,
                    n_ar,
                    lpc_pred_q14,
                    lf_ar_q14,
                    shape_q14,
                    seed,
                );
                for k in 0..4 {
                    let want = quantize_sample(
                        &lv,
                        x_q10,
                        ltp_pred_q14,
                        n_ltp_q14,
                        n_ar[k],
                        lpc_pred_q14[k],
                        lf_ar_q14[k],
                        shape_q14[k],
                        seed[k],
                        rd_q10[k],
                    );
                    for j in 0..2 {
                        let what = format!("case {case}, lane {k}, candidate {j}");
                        let got = &got[j];
                        assert_eq!(got.q_q10[k], want[j].q_q10, "{what}: q_q10");
                        assert_eq!(
                            rd_q10[k].wrapping_add(got.rd_q10[k]),
                            want[j].rd_q10,
                            "{what}: rd_q10"
                        );
                        assert_eq!(got.xq_q14[k], want[j].xq_q14, "{what}: xq_q14");
                        assert_eq!(got.lf_ar_q14[k], want[j].lf_ar_q14, "{what}: lf_ar_q14");
                        assert_eq!(got.diff_q14[k], want[j].diff_q14, "{what}: diff_q14");
                        assert_eq!(
                            got.s_ltp_shp_q14[k], want[j].s_ltp_shp_q14,
                            "{what}: s_ltp_shp_q14"
                        );
                        assert_eq!(
                            got.lpc_exc_q14[k], want[j].lpc_exc_q14,
                            "{what}: lpc_exc_q14"
                        );
                    }
                }
            }
        }

        /// Everything one subframe reads and writes.
        #[derive(Clone)]
        struct World {
            nsq: SilkNSQState,
            states: [PathState; NSQ_MAX_STATES_OPERATING],
            delay_lines: DelayLines,
            pulses: [i8; MAX_FRAME_LENGTH],
            s_ltp_q15: [i32; 2 * MAX_FRAME_LENGTH],
            delayed_ga_q10: [i32; DECISION_DELAY],
            smpl_buf_idx: i32,
        }

        fn random_world(rng: &mut Rng, n_states: usize, ltp_mem: i32) -> World {
            let mut w = World {
                nsq: SilkNSQState::default(),
                states: [PathState::ZERO; NSQ_MAX_STATES_OPERATING],
                delay_lines: DelayLines::new(),
                pulses: [0; MAX_FRAME_LENGTH],
                s_ltp_q15: [0; 2 * MAX_FRAME_LENGTH],
                delayed_ga_q10: [0; DECISION_DELAY],
                smpl_buf_idx: rng.range(0, DECISION_DELAY as i64 - 1) as i32,
            };
            for v in w.nsq.s_ltp_shp_q14.iter_mut() {
                *v = rng.range(-(1 << 22), 1 << 22) as i32;
            }
            w.nsq.s_ltp_shp_buf_idx = ltp_mem;
            w.nsq.s_ltp_buf_idx = ltp_mem;
            for v in w.s_ltp_q15.iter_mut() {
                *v = rng.range(-(1 << 22), 1 << 22) as i32;
            }
            for v in w.delayed_ga_q10.iter_mut() {
                *v = rng.range(1 << 6, 1 << 20) as i32;
            }
            for state in w.states.iter_mut() {
                for v in state.s_lpc_q14.iter_mut() {
                    *v = rng.range(-(1 << 22), 1 << 22) as i32;
                }
                for v in state.s_ar2_q14.iter_mut() {
                    *v = rng.range(-(1 << 22), 1 << 22) as i32;
                }
                state.lf_ar_q14 = rng.range(-(1 << 22), 1 << 22) as i32;
                state.diff_q14 = rng.range(-(1 << 22), 1 << 22) as i32;
                state.seed = rng.value(30);
                state.seed_init = rng.range(0, 3) as i32;
                state.rd_q10 = rng.range(0, 1 << 20) as i32;
            }
            let newest = (w.smpl_buf_idx as usize).min(DECISION_DELAY - 1);
            for slot in 0..DECISION_DELAY {
                for k in 0..NSQ_MAX_STATES_OPERATING {
                    // Few distinct dither states, so that states agree and
                    // disagree with the winner.
                    w.delay_lines.rand_state[slot][k] = rng.range(0, 2) as i32;
                    w.delay_lines.q_q10[slot][k] = rng.range(-(1 << 14), 1 << 14) as i32;
                    w.delay_lines.xq_q14[slot][k] = rng.range(-(1 << 24), 1 << 24) as i32;
                    w.delay_lines.pred_q15[slot][k] = rng.range(-(1 << 24), 1 << 24) as i32;
                    w.delay_lines.shape_q14[slot][k] = rng.range(-(1 << 22), 1 << 22) as i32;
                }
                for k in 0..n_states {
                    // Histories that share samples, except in the newest slot.
                    w.delay_lines.column[k][slot] = if slot == newest {
                        k as u8
                    } else {
                        rng.range(0, n_states as i64 - 1) as u8
                    };
                }
            }
            w
        }

        fn assert_same_world(
            neon: &World,
            portable: &World,
            n_states: usize,
            length: usize,
            what: &str,
        ) {
            assert_eq!(
                neon.smpl_buf_idx, portable.smpl_buf_idx,
                "{what}: smpl_buf_idx"
            );
            assert_eq!(neon.pulses, portable.pulses, "{what}: pulses");
            assert_eq!(neon.s_ltp_q15, portable.s_ltp_q15, "{what}: s_ltp_q15");
            assert_eq!(
                neon.delayed_ga_q10, portable.delayed_ga_q10,
                "{what}: delayed gains"
            );
            assert_eq!(neon.nsq.xq, portable.nsq.xq, "{what}: xq");
            assert_eq!(
                neon.nsq.s_ltp_shp_q14, portable.nsq.s_ltp_shp_q14,
                "{what}: s_ltp_shp_q14"
            );
            assert_eq!(
                neon.nsq.s_ltp_buf_idx, portable.nsq.s_ltp_buf_idx,
                "{what}: s_ltp_buf_idx"
            );
            assert_eq!(
                neon.nsq.s_ltp_shp_buf_idx, portable.nsq.s_ltp_shp_buf_idx,
                "{what}: s_ltp_shp_buf_idx"
            );
            for k in 0..n_states {
                let (a, b) = (&neon.states[k], &portable.states[k]);
                let what = format!("{what}, state {k}");
                // Above this the array holds leftovers of copies.
                let live = if length == 0 {
                    a.s_lpc_q14.len()
                } else {
                    length + NSQ_LPC_BUF_LENGTH
                };
                assert_eq!(
                    a.s_lpc_q14[..live],
                    b.s_lpc_q14[..live],
                    "{what}: s_lpc_q14"
                );
                assert_eq!(a.s_ar2_q14, b.s_ar2_q14, "{what}: s_ar2_q14");
                assert_eq!(a.lf_ar_q14, b.lf_ar_q14, "{what}: lf_ar_q14");
                assert_eq!(a.diff_q14, b.diff_q14, "{what}: diff_q14");
                assert_eq!(a.seed, b.seed, "{what}: seed");
                assert_eq!(a.seed_init, b.seed_init, "{what}: seed_init");
                assert_eq!(a.rd_q10, b.rd_q10, "{what}: rd_q10");
                for slot in 0..DECISION_DELAY {
                    let (da, db) = (&neon.delay_lines, &portable.delay_lines);
                    let (ca, cb) = (da.column(k, slot), db.column(k, slot));
                    let what = format!("{what}, slot {slot}");
                    assert_eq!(
                        da.rand_state[slot][ca], db.rand_state[slot][cb],
                        "{what}: rand_state"
                    );
                    assert_eq!(da.q_q10[slot][ca], db.q_q10[slot][cb], "{what}: q_q10");
                    assert_eq!(da.xq_q14[slot][ca], db.xq_q14[slot][cb], "{what}: xq_q14");
                    assert_eq!(
                        da.pred_q15[slot][ca], db.pred_q15[slot][cb],
                        "{what}: pred_q15"
                    );
                    assert_eq!(
                        da.shape_q14[slot][ca], db.shape_q14[slot][cb],
                        "{what}: shape_q14"
                    );
                }
            }
        }

        fn check_subframes<const N: usize>(rng: &mut Rng) {
            // Every length up to 5, the three subframe lengths of the codec
            // and their neighbours.
            let lengths = [0usize, 1, 2, 3, 4, 5, 39, 40, 41, 59, 60, 61, 79, 80];
            for case in 0..(lengths.len() * 12) {
                let length = lengths[case % lengths.len()];
                let ltp_mem = 320;
                let mut neon = random_world(rng, N, ltp_mem);
                let mut portable = neon.clone();

                let mut x_q10 = [0i32; MAX_SUB_FRAME_LENGTH];
                for v in x_q10.iter_mut() {
                    *v = rng.range(-(1 << 20), 1 << 20) as i32;
                }
                let mut a_q12 = [0i16; MAX_LPC_ORDER];
                for v in a_q12.iter_mut() {
                    *v = rng.range(-1500, 1500) as i16;
                }
                let mut ar_shp_q13 = [0i16; MAX_SHAPE_LPC_ORDER];
                for v in ar_shp_q13.iter_mut() {
                    *v = rng.range(-1500, 1500) as i16;
                }
                let b_q14: [i16; LTP_ORDER] =
                    core::array::from_fn(|_| rng.range(-3000, 6000) as i16);
                let voiced = case % 3 != 0;
                let longest_delay = (DECISION_DELAY as i64).min(length.max(1) as i64);
                let decision_delay = rng.range(1, longest_delay) as i32;
                let subfr = rng.range(0, 3) as i32;
                let p = SubframeParams {
                    signal_type: if voiced { TYPE_VOICED } else { 1 },
                    pulses_offset: subfr * 80,
                    xq_ptr: ltp_mem + subfr * 80,
                    a_q12: &a_q12,
                    b_q14: &b_q14,
                    ar_shp_q13: &ar_shp_q13,
                    // An unvoiced subframe keeps the lag of the last voiced
                    // one for the harmonic noise shaping (libopus
                    // `lagPrev`), so half of them have a lag.
                    lag: if voiced || case % 2 == 0 {
                        rng.range(decision_delay as i64 + 3, 288) as i32
                    } else {
                        0
                    },
                    harm_shape_fir_packed_q14: rng.range(-(1 << 30), 1 << 30) as i32,
                    tilt_q14: rng.range(-8192, 0) as i32,
                    lf_shp_q14: rng.range(-(1 << 30), 1 << 30) as i32,
                    gain_q16: rng.range(1 << 12, 1 << 28) as i32,
                    lambda_q10: rng.range(900, 2800) as i32,
                    offset_q10: [32, 100, 240][rng.range(0, 2) as usize],
                    length: length as i32,
                    subfr,
                    shaping_lpc_order: [2, 12, 14, 16, 20, 24][rng.range(0, 5) as usize],
                    predict_lpc_order: [10, 16, 6, 0][rng.range(0, 3) as usize],
                    warping_q16: if case % 4 == 0 { 0 } else { 16 * 983 },
                    decision_delay,
                };

                let start = neon.clone();
                let mut runtime_order = neon.clone();
                // `wide` selects the AVX2 loop on x86-64.
                let world = |w: &mut World, order: usize, wide: bool| {
                    macro_rules! call {
                        ($loop:ident, $order:literal) => {
                            // SAFETY: NEON is available: src/lib.rs refuses to
                            // build for aarch64 without it; on x86-64 the test
                            // runs only with SSE4.1, and with `wide` only with
                            // AVX2.
                            unsafe {
                                $loop::<N, $order>(
                                    &mut w.nsq,
                                    w.states.first_chunk_mut::<N>().unwrap(),
                                    &mut w.delay_lines,
                                    &x_q10,
                                    &mut w.pulses,
                                    &mut w.s_ltp_q15,
                                    &mut w.delayed_ga_q10,
                                    &mut w.smpl_buf_idx,
                                    &p,
                                )
                            }
                        };
                    }
                    macro_rules! run {
                        ($order:literal) => {{
                            #[cfg(target_arch = "aarch64")]
                            {
                                assert!(!wide);
                                call!(quantize_subframe_neon, $order)
                            }
                            #[cfg(target_arch = "x86_64")]
                            if wide {
                                call!(quantize_subframe_avx2, $order)
                            } else {
                                call!(quantize_subframe_sse41, $order)
                            }
                        }};
                    }
                    // The instances `quantize_subframe_dispatch` uses.
                    match (N, order) {
                        (2, 12) => run!(12),
                        (2, 14) => run!(14),
                        (2, 16) => run!(16),
                        (3, 20) => run!(20),
                        (4, 24) => run!(24),
                        _ => run!(0),
                    }
                };
                world(&mut neon, p.shaping_lpc_order as usize, false);
                world(&mut runtime_order, 0, false);
                quantize_subframe::<N>(
                    &mut portable.nsq,
                    portable.states.first_chunk_mut::<N>().unwrap(),
                    &mut portable.delay_lines,
                    &x_q10,
                    &mut portable.pulses,
                    &mut portable.s_ltp_q15,
                    &mut portable.delayed_ga_q10,
                    &mut portable.smpl_buf_idx,
                    &p,
                );
                let what = format!("{N} states, length {length}, case {case}");
                assert_same_world(&neon, &portable, N, length, &what);
                let what = format!("{what}, order at run time");
                assert_same_world(&runtime_order, &portable, N, length, &what);

                // The AVX2 loop, where the processor has it.
                if cfg!(target_arch = "x86_64") && crate::compat::x86_has_avx2() {
                    for order in [p.shaping_lpc_order as usize, 0] {
                        let mut wide = start.clone();
                        world(&mut wide, order, true);
                        let what = format!("{what}, AVX2, order {order}");
                        assert_same_world(&wide, &portable, N, length, &what);
                    }
                }
            }
        }

        #[test]
        fn subframe_matches_portable_loop() {
            if !simd_available() {
                return;
            }
            let mut rng = Rng(5);
            check_subframes::<2>(&mut rng);
            check_subframes::<3>(&mut rng);
            check_subframes::<4>(&mut rng);
        }

        /// The x86 forms of `vqdmulhq_s32`, of the saturating and rounding
        /// steps and of the sign flip, against the scalar macros, at the
        /// edges of `i32` and on random values.
        #[cfg(target_arch = "x86_64")]
        #[test]
        fn x86_helpers_match_scalar() {
            if !simd_available() {
                return;
            }
            const EDGES: [i32; 12] = [
                i32::MIN,
                i32::MIN + 1,
                i32::MIN / 2,
                -65536,
                -32768,
                -1,
                0,
                1,
                32767,
                65535,
                i32::MAX - 1,
                i32::MAX,
            ];
            let mut rng = Rng(6);
            // Every pair of edges, then random pairs, four at a time.
            let edge_pairs = EDGES.len() * EDGES.len();
            for group in 0..(edge_pairs + 20000) / 4 {
                let mut a = [0i32; 4];
                let mut b = [0i32; 4];
                for k in 0..4 {
                    let n = 4 * group + k;
                    (a[k], b[k]) = if n < edge_pairs {
                        (EDGES[n / EDGES.len()], EDGES[n % EDGES.len()])
                    } else {
                        (rng.value(30), rng.value(30))
                    };
                }
                let [products, sums, differences, round4, round10, signed] = helpers_lanes(a, b);
                for k in 0..4 {
                    let what = format!("a {}, b {}", a[k], b[k]);
                    assert_eq!(products[k], silk_smulwb(a[k], b[k]), "mulw, {what}");
                    assert_eq!(sums[k], silk_add_sat32(a[k], b[k]), "adds, {what}");
                    assert_eq!(differences[k], silk_sub_sat32(a[k], b[k]), "subs, {what}");
                    assert_eq!(round4[k], silk_rshift_round(a[k], 4), "round 4, {what}");
                    assert_eq!(round10[k], silk_rshift_round(a[k], 10), "round 10, {what}");
                    let want = if b[k] < 0 { a[k].wrapping_neg() } else { a[k] };
                    assert_eq!(signed[k], want, "sign, {what}");
                }
            }
        }

        /// The short-term prediction of the x86 loop, two samples of it
        /// from registers, against the definition: every order, 2 to 4
        /// states.
        #[cfg(target_arch = "x86_64")]
        #[test]
        fn x86_short_prediction_matches_the_definition() {
            if !simd_available() {
                return;
            }
            fn check<const N: usize>(rng: &mut Rng) {
                for case in 0..2000 {
                    let order = case % (MAX_LPC_ORDER + 1);
                    let windows: [[i32; MAX_LPC_ORDER]; N] =
                        core::array::from_fn(|_| core::array::from_fn(|_| rng.value(30)));
                    let a_q12: [i16; MAX_LPC_ORDER] = core::array::from_fn(|_| rng.coef());
                    for wide in [false, true] {
                        if wide && !crate::compat::x86_has_avx2() {
                            continue;
                        }
                        let got = short_prediction_lanes::<N>(&windows, &a_q12, order, wide);
                        for k in 0..N {
                            let want = silk_lshift(
                                super::prediction_by_definition(&windows[k], &a_q12, order),
                                4,
                            );
                            let what = format!("{N} states, order {order}, state {k}, wide {wide}");
                            assert_eq!(got[k], want, "{what}");
                        }
                    }
                }
            }
            let mut rng = Rng(7);
            check::<2>(&mut rng);
            check::<3>(&mut rng);
            check::<4>(&mut rng);
        }
    }
}
