use crate::silk::define::*;
use crate::silk::macros::*;
use crate::silk::sigproc_fix::*;
use crate::silk::structs::*;
use crate::silk::tables::*;

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
    #[cfg(target_arch = "x86_64")]
    if crate::compat::x86_has_avx2() {
        // SAFETY: the processor has AVX2.
        unsafe {
            nsq_avx2(
                ps_enc_c,
                nsq,
                ps_indices,
                x16,
                pulses,
                pred_coef_q12,
                ltp_coef_q14,
                ar_q13,
                harm_shape_gain_q14,
                tilt_q14,
                lf_shp_q14,
                gains_q16,
                pitch_l,
                lambda_q10,
                ltp_scale_q14,
            )
        };
        return;
    }
    nsq_portable(
        ps_enc_c,
        nsq,
        ps_indices,
        x16,
        pulses,
        pred_coef_q12,
        ltp_coef_q14,
        ar_q13,
        harm_shape_gain_q14,
        tilt_q14,
        lf_shp_q14,
        gains_q16,
        pitch_l,
        lambda_q10,
        ltp_scale_q14,
    );
}

/// [`silk_nsq`] compiled with AVX2 enabled, for the scaling of the states
/// and the setup of each subframe. The sample loop has its own SSE4.1
/// version (`x86::quantize_subframe_sse41`).
///
/// AVX2 only changes how the compiler encodes and vectorizes the code. The
/// default x86-64 target has no 32-bit and no 32x32-to-64-bit vector
/// multiplication, which fixed-point code needs; AVX2 has both. The results
/// are the same bits.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
fn nsq_avx2(
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
    nsq_portable(
        ps_enc_c,
        nsq,
        ps_indices,
        x16,
        pulses,
        pred_coef_q12,
        ltp_coef_q14,
        ar_q13,
        harm_shape_gain_q14,
        tilt_q14,
        lf_shp_q14,
        gains_q16,
        pitch_l,
        lambda_q10,
        ltp_scale_q14,
    );
}

#[inline(always)]
fn nsq_portable(
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

    // The quantization offset is the same for all subframes of a frame.
    let offset_q10 = SILK_QUANT_OFFSETS_Q10[(ps_indices.signal_type >> 1) as usize]
        [ps_indices.quant_offset_type as usize] as i32;

    for k in 0..ps_enc_c.nb_subfr as usize {
        a_q12 =
            &pred_coef_q12[((k >> 1) | (1 - lsf_interpolation_flag as usize)) * MAX_LPC_ORDER..];
        b_q14 = &ltp_coef_q14[k * LTP_ORDER..];
        ar_shp_q13 = &ar_q13[k * MAX_SHAPE_LPC_ORDER..];

        nsq.rewhite_flag = 0;
        if ps_indices.signal_type == TYPE_VOICED as i8 {
            lag = pitch_l[k] as usize;

            if (k & (3 - (lsf_interpolation_flag << 1) as usize)) == 0 {
                // Clamp: lag/order are geometry-bounded for valid streams; a
                // corrupted state must not yield a negative start index.
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

/// Scales the input of subframe `k` by the inverse of its gain, and the
/// states of the quantizer by the change of the gain (libopus
/// `silk_nsq_scale_states`). On x86-64 it is inlined, so that the AVX2 copy
/// of `silk_nsq` compiles it with AVX2 too.
#[cfg_attr(target_arch = "x86_64", inline(always))]
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

    let mut inv_gain_q31 = silk_inverse32_varq(if gains_q16[k] > 1 { gains_q16[k] } else { 1 }, 47);

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

/// Noise shaping feedback of one sample (libopus
/// `silk_NSQ_noise_shape_feedback_loop`). `state` and `coef` have the same
/// length, the shaping order.
///
/// The state is a shift register: every value moves up by one place and
/// `newest` enters at place 0. The return value is twice
/// `order / 2 + sum((state[j] * coef[j]) >> 16)` over the new state.
///
/// The product with `newest` is added last. `newest` is the value the caller
/// has just computed, so the other products do not have to wait for it.
#[inline(always)]
fn noise_shape_feedback(newest: i32, state: &mut [i32], coef: &[i16]) -> i32 {
    debug_assert_eq!(state.len(), coef.len());
    let mut out = (coef.len() as i32) >> 1;
    let mut moving = newest;
    for (j, (s, &c)) in state.iter_mut().zip(coef).enumerate() {
        let old = *s;
        *s = moving;
        if j > 0 {
            out = silk_smlawb(out, moving, c as i32);
        }
        moving = old;
    }
    if let Some(&c) = coef.first() {
        out = silk_smlawb(out, newest, c as i32);
    }
    out << 1
}

/// [`noise_shape_feedback`] for the shaping order `ORDER`, known at compile
/// time so that the loop is unrolled. `state` and `coef` must have at least
/// `ORDER` elements.
#[inline(always)]
fn noise_shape_feedback_order<const ORDER: usize>(
    newest: i32,
    state: &mut [i32],
    coef: &[i16],
) -> i32 {
    let state: &mut [i32; ORDER] = state.first_chunk_mut().unwrap();
    let coef: &[i16; ORDER] = coef.first_chunk().unwrap();
    noise_shape_feedback(newest, state, coef)
}

/// Short-term prediction of one sample (libopus
/// `silk_noise_shape_quantizer_short_prediction`):
/// `order / 2 + sum((sample j + 1 steps back * a_q12[j]) >> 16)` for
/// `j < order`.
///
/// `window` holds the 16 newest samples with the newest one last. That last
/// entry is not read: the caller passes the newest sample in `newest`, the
/// value it has just computed, and its product is added last. `order` is at
/// most 16; `a_q12` holds at least `order` coefficients.
///
/// On aarch64 the quantizer uses `short_prediction_neon`; this version is
/// its reference in the tests.
#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
#[inline(always)]
fn short_prediction(
    window: &[i32; NSQ_LPC_BUF_LENGTH],
    a_q12: &[i16; MAX_LPC_ORDER],
    order: usize,
    newest: i32,
) -> i32 {
    let mut pred = (order as i32) >> 1;
    for j in 1..order.min(MAX_LPC_ORDER) {
        pred = silk_smlawb(pred, window[NSQ_LPC_BUF_LENGTH - 1 - j], a_q12[j] as i32);
    }
    if order > 0 {
        pred = silk_smlawb(pred, newest, a_q12[0] as i32);
    }
    pred
}

/// The coefficients of `short_prediction` in the form `short_prediction_neon`
/// reads, prepared once per subframe.
///
/// Lane `i` of `older[v]` multiplies window position `4 * v + i` for
/// `v < 3`; `older[3]` multiplies positions 11 to 14. Position 11 is in two
/// of these loads, so lane 0 of `older[3]` is zero. Each entry is
/// `smulwb_operand(a_q12[15 - position])`, so that `vqdmulhq_s32` gives the
/// product of `silk_smlawb`. Position 15, the newest sample, is not here.
#[cfg(target_arch = "aarch64")]
struct ShortPredictionNeon {
    older: [[i32; 4]; 4],
}

#[cfg(target_arch = "aarch64")]
impl ShortPredictionNeon {
    fn new(a_q12: &[i16; MAX_LPC_ORDER]) -> Self {
        let operand = |position: usize| smulwb_operand(a_q12[MAX_LPC_ORDER - 1 - position]);
        let mut older = [[0i32; 4]; 4];
        for (v, row) in older.iter_mut().enumerate() {
            for (i, c) in row.iter_mut().enumerate() {
                let position = if v < 3 { 4 * v + i } else { 11 + i };
                if v < 3 || i > 0 {
                    *c = operand(position);
                }
            }
        }
        Self { older }
    }
}

/// `short_prediction` with NEON: the 15 older samples in 4 vector products,
/// then the newest one with `silk_smlawb`. The additions wrap around, so
/// their order does not change the result. `a0` is `a_q12[0]` and `bias` is
/// `order / 2`; the coefficients from the order on are zero.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn short_prediction_neon(
    window: &[i32; NSQ_LPC_BUF_LENGTH],
    c: &ShortPredictionNeon,
    bias: i32,
    a0: i16,
    newest: i32,
) -> i32 {
    use crate::simd_mem::neon::load_i32x4;
    use core::arch::aarch64::*;
    // Window positions 0 to 3, 4 to 7 and 8 to 11, then 11 to 14.
    let (w, _) = window.as_chunks::<4>();
    // SAFETY: NEON is part of the aarch64 baseline.
    let sum = unsafe {
        let t0 = vqdmulhq_s32(load_i32x4(&w[0]), load_i32x4(&c.older[0]));
        let t1 = vqdmulhq_s32(load_i32x4(&w[1]), load_i32x4(&c.older[1]));
        let t2 = vqdmulhq_s32(load_i32x4(&w[2]), load_i32x4(&c.older[2]));
        let last = window[11..15].try_into().unwrap();
        let t3 = vqdmulhq_s32(load_i32x4(last), load_i32x4(&c.older[3]));
        vaddvq_s32(vaddq_s32(vaddq_s32(t0, t1), vaddq_s32(t2, t3)))
    };
    silk_smlawb(bias.wrapping_add(sum), newest, a0 as i32)
}

/// Shaping orders up to this one keep the state of the noise shaping
/// feedback in NEON registers (`FeedbackNeon`).
#[cfg(target_arch = "aarch64")]
const FEEDBACK_NEON_MAX_ORDER: usize = 16;

/// `noise_shape_feedback` with its state in 4 NEON registers instead of
/// memory, for shaping orders up to `FEEDBACK_NEON_MAX_ORDER`.
///
/// Lane `i` of `state[v]` is entry `4 * v + i` of the filter state. The
/// lanes from the order on take part in the shifts but are never added up
/// or written back.
#[cfg(target_arch = "aarch64")]
struct FeedbackNeon {
    state: [core::arch::aarch64::int32x4_t; 4],
    /// Lane `i` of `coef[v]` multiplies state entry `4 * v + i`, which
    /// becomes entry `4 * v + i + 1` in the step: it is
    /// `smulwb_operand(coef[4 * v + i + 1])`, zero from the order on.
    coef: [[i32; 4]; 4],
    /// The coefficient of the newest value, `coef[0]`.
    coef0: i16,
    /// `order / 2`, the start of the sum.
    bias: i32,
}

#[cfg(target_arch = "aarch64")]
impl FeedbackNeon {
    /// `state` and `coef` have the same length, the shaping order, at most
    /// `FEEDBACK_NEON_MAX_ORDER`.
    fn new(state: &[i32], coef: &[i16]) -> Self {
        use crate::simd_mem::neon::load_i32x4;
        let order = coef.len().min(FEEDBACK_NEON_MAX_ORDER);
        debug_assert_eq!(state.len(), coef.len());
        debug_assert!(coef.len() <= FEEDBACK_NEON_MAX_ORDER);
        let mut lanes = [0i32; FEEDBACK_NEON_MAX_ORDER];
        lanes[..order].copy_from_slice(&state[..order]);
        let mut c = [[0i32; 4]; 4];
        for (j, dst) in c.as_flattened_mut().iter_mut().enumerate() {
            if j + 1 < order {
                *dst = smulwb_operand(coef[j + 1]);
            }
        }
        let (rows, _) = lanes.as_chunks::<4>();
        // SAFETY: NEON is part of the aarch64 baseline.
        let state = unsafe { core::array::from_fn(|v| load_i32x4(&rows[v])) };
        Self {
            state,
            coef: c,
            coef0: coef.first().copied().unwrap_or(0),
            bias: (order as i32) >> 1,
        }
    }

    /// One step of `noise_shape_feedback`: the same return value and the
    /// same new state.
    #[inline(always)]
    fn step(&mut self, newest: i32) -> i32 {
        use crate::simd_mem::neon::load_i32x4;
        use core::arch::aarch64::*;
        let s = &mut self.state;
        // SAFETY: NEON is part of the aarch64 baseline.
        let sum = unsafe {
            let c = &self.coef;
            let t0 = vqdmulhq_s32(s[0], load_i32x4(&c[0]));
            let t1 = vqdmulhq_s32(s[1], load_i32x4(&c[1]));
            let t2 = vqdmulhq_s32(s[2], load_i32x4(&c[2]));
            let t3 = vqdmulhq_s32(s[3], load_i32x4(&c[3]));
            vaddvq_s32(vaddq_s32(vaddq_s32(t0, t1), vaddq_s32(t2, t3)))
        };
        // Every entry moves up by one place and `newest` enters at place 0.
        // SAFETY: NEON is part of the aarch64 baseline.
        unsafe {
            s[3] = vextq_s32::<3>(s[2], s[3]);
            s[2] = vextq_s32::<3>(s[1], s[2]);
            s[1] = vextq_s32::<3>(s[0], s[1]);
            s[0] = vextq_s32::<3>(vdupq_n_s32(newest), s[0]);
        }
        silk_smlawb(self.bias.wrapping_add(sum), newest, self.coef0 as i32) << 1
    }

    /// Writes the state back; `state` has the length of the order.
    fn store(&self, state: &mut [i32]) {
        use crate::simd_mem::neon::store_i32x4;
        let mut lanes = [0i32; FEEDBACK_NEON_MAX_ORDER];
        let (rows, _) = lanes.as_chunks_mut::<4>();
        for (dst, &row) in rows.iter_mut().zip(&self.state) {
            // SAFETY: NEON is part of the aarch64 baseline.
            unsafe { store_i32x4(dst, row) };
        }
        let order = state.len().min(FEEDBACK_NEON_MAX_ORDER);
        state[..order].copy_from_slice(&lanes[..order]);
    }
}

/// The vector kernels of the plain quantizer for x86-64 processors with
/// SSE4.1: `short_prediction_neon` and `FeedbackNeon` with SSE instructions.
///
/// x86 has no instruction like NEON's `vqdmulhq_s32`. `_mm_mul_epi32`
/// multiplies the 32-bit lanes 0 and 2 into two 64-bit products. With the
/// factor `c << 16`, the upper halves of the products, lanes 1 and 3, are
/// `(x * c) >> 16`: the product of `silk_smlawb`. The sums below collect
/// lanes 1 and 3; lanes 0 and 2 hold the lower halves and are not used.
#[cfg(target_arch = "x86_64")]
mod x86 {
    use super::*;
    use crate::simd_mem::x86::{load_i32x4, store_i32x4};
    use core::arch::x86_64::*;

    /// Shaping orders up to this one keep the feedback state in registers
    /// (`Feedback`), as on aarch64.
    pub(super) const FEEDBACK_MAX_ORDER: usize = 16;

    /// Window positions of the loads of `short_prediction`. The load at
    /// position `o` puts positions `o` and `o + 2` into lanes 0 and 2. The
    /// last load is used twice: as it is (positions 10 and 12) and shifted
    /// by one lane (positions 11 and 13). Positions 10 and 11 are in two
    /// loads each.
    const OFFSETS: [usize; 7] = [0, 1, 4, 5, 8, 9, 10];

    /// The coefficients of `short_prediction` in the form it reads,
    /// prepared once per subframe.
    pub(super) struct ShortPrediction {
        /// Lanes 0 and 2 of row `i` are the factors `a << 16` of the window
        /// positions `OFFSETS[i]` and `OFFSETS[i] + 2`, with `a` the
        /// coefficient that multiplies the position. Row 7 is for the last
        /// load shifted by one lane: positions 11 and 13. The second loads
        /// of positions 10 and 11 have the factor 0.
        older: [[i32; 4]; 8],
    }

    impl ShortPrediction {
        /// `a_q12` holds the coefficients, zero from the prediction order
        /// on.
        pub(super) fn new(a_q12: &[i16; MAX_LPC_ORDER]) -> Self {
            // Position 15 is the newest sample: `a_q12[0]`.
            let factor = |position: usize| (a_q12[MAX_LPC_ORDER - 1 - position] as i32) << 16;
            let rows = [(0, 2), (1, 3), (4, 6), (5, 7), (8, 10), (9, 11)];
            let mut older = [[0i32; 4]; 8];
            for (row, (low, high)) in older.iter_mut().zip(rows) {
                *row = [factor(low), factor(low), factor(high), factor(high)];
            }
            older[6] = [0, 0, factor(12), factor(12)];
            older[7] = [0, 0, factor(13), factor(13)];
            Self { older }
        }
    }

    /// `short_prediction` with SSE4.1: window positions 0 to 13 in vector
    /// products. Positions 14 and 15 come in `second` and `newest`, the
    /// values the caller has computed in the last two samples: their stores
    /// may not have reached the cache yet, and an x86 core does not forward
    /// a recent store to a load that also reads other bytes. The newest
    /// product is added last. `bias` is `order / 2`; `a_q12` holds the
    /// coefficients, zero from the order on.
    #[target_feature(enable = "sse4.1")]
    #[inline]
    pub(super) fn short_prediction(
        window: &[i32; NSQ_LPC_BUF_LENGTH],
        c: &ShortPrediction,
        bias: i32,
        a_q12: &[i16; MAX_LPC_ORDER],
        second: i32,
        newest: i32,
    ) -> i32 {
        let mut sum = _mm_setzero_si128();
        let mut last = sum;
        for (row, &offset) in c.older.iter().zip(&OFFSETS) {
            last = load_i32x4(window[offset..offset + 4].try_into().unwrap());
            sum = _mm_add_epi32(sum, _mm_mul_epi32(last, load_i32x4(row)));
        }
        let shifted = _mm_srli_epi64::<32>(last);
        sum = _mm_add_epi32(sum, _mm_mul_epi32(shifted, load_i32x4(&c.older[7])));
        // Lane 1 plus lane 3.
        let sum = _mm_add_epi32(sum, _mm_unpackhi_epi64(sum, sum));
        let older = _mm_extract_epi32::<1>(sum);
        let pred = silk_smlawb(bias.wrapping_add(older), second, a_q12[1] as i32);
        silk_smlawb(pred, newest, a_q12[0] as i32)
    }

    /// `noise_shape_feedback` with its state in 4 SSE registers instead of
    /// memory, as `FeedbackNeon`, for shaping orders up to
    /// `FEEDBACK_MAX_ORDER`.
    ///
    /// Lane `i` of `state[v]` is entry `4 * v + i` of the filter state. The
    /// lanes from the order on take part in the shifts but are never added
    /// up or written back.
    pub(super) struct Feedback {
        state: [__m128i; 4],
        /// Lanes 0 and 2 of `even[v]` multiply the state entries `4 * v` and
        /// `4 * v + 2`; lanes 0 and 2 of `odd[v]` the entries `4 * v + 1` and
        /// `4 * v + 3`, after a shift by one lane. Entry `j` becomes entry
        /// `j + 1` in the step, so its factor is `coef[j + 1] << 16`, zero
        /// from the order on.
        even: [[i32; 4]; 4],
        odd: [[i32; 4]; 4],
        /// The coefficient of the newest value, `coef[0]`.
        coef0: i16,
        /// `order / 2`, the start of the sum.
        bias: i32,
    }

    impl Feedback {
        /// `state` and `coef` have the same length, the shaping order, at
        /// most `FEEDBACK_MAX_ORDER`. Needs only SSE2.
        pub(super) fn new(state: &[i32], coef: &[i16]) -> Self {
            let order = coef.len().min(FEEDBACK_MAX_ORDER);
            debug_assert_eq!(state.len(), coef.len());
            debug_assert!(coef.len() <= FEEDBACK_MAX_ORDER);
            let mut lanes = [[0i32; 4]; 4];
            lanes.as_flattened_mut()[..order].copy_from_slice(&state[..order]);
            let factor = |j: usize| {
                if j + 1 < order {
                    (coef[j + 1] as i32) << 16
                } else {
                    0
                }
            };
            let mut even = [[0i32; 4]; 4];
            let mut odd = [[0i32; 4]; 4];
            for v in 0..4 {
                let (e0, e2) = (factor(4 * v), factor(4 * v + 2));
                let (o1, o3) = (factor(4 * v + 1), factor(4 * v + 3));
                even[v] = [e0, e0, e2, e2];
                odd[v] = [o1, o1, o3, o3];
            }
            Self {
                // SAFETY: SSE2 is part of the x86-64 baseline.
                state: unsafe { lanes.map(|row| load_i32x4(&row)) },
                even,
                odd,
                coef0: coef.first().copied().unwrap_or(0),
                bias: (order as i32) >> 1,
            }
        }

        /// One step of `noise_shape_feedback`: the same return value and the
        /// same new state.
        #[target_feature(enable = "sse4.1")]
        #[inline]
        pub(super) fn step(&mut self, newest: i32) -> i32 {
            let s = &mut self.state;
            let mut sum = _mm_setzero_si128();
            for v in 0..4 {
                let even = _mm_mul_epi32(s[v], load_i32x4(&self.even[v]));
                let odd = _mm_mul_epi32(_mm_srli_epi64::<32>(s[v]), load_i32x4(&self.odd[v]));
                sum = _mm_add_epi32(sum, _mm_add_epi32(even, odd));
            }
            // Lane 1 plus lane 3.
            let sum = _mm_add_epi32(sum, _mm_unpackhi_epi64(sum, sum));
            let older = _mm_extract_epi32::<1>(sum);
            // Every entry moves up by one place and `newest` enters at place
            // 0: `_mm_alignr_epi8::<12>(a, b)` is `[b[3], a[0], a[1], a[2]]`.
            s[3] = _mm_alignr_epi8::<12>(s[3], s[2]);
            s[2] = _mm_alignr_epi8::<12>(s[2], s[1]);
            s[1] = _mm_alignr_epi8::<12>(s[1], s[0]);
            s[0] = _mm_alignr_epi8::<12>(s[0], _mm_set1_epi32(newest));
            silk_smlawb(self.bias.wrapping_add(older), newest, self.coef0 as i32) << 1
        }

        /// Writes the state back; `state` has the length of the order.
        pub(super) fn store(&self, state: &mut [i32]) {
            let mut lanes = [[0i32; 4]; 4];
            for (row, &v) in lanes.iter_mut().zip(&self.state) {
                // SAFETY: SSE2 is part of the x86-64 baseline.
                unsafe { store_i32x4(row, v) };
            }
            let order = state.len().min(FEEDBACK_MAX_ORDER);
            state[..order].copy_from_slice(&lanes.as_flattened()[..order]);
        }
    }

    /// The sample loop of `silk_noise_shape_quantizer` with the vector
    /// kernels, one copy for each combination of long-term prediction and
    /// harmonic shaping (`quantize_subframe_cases`).
    ///
    /// Panics when the shaping order exceeds `FEEDBACK_MAX_ORDER`.
    #[target_feature(enable = "sse4.1")]
    pub(super) fn quantize_subframe_sse41(
        nsq: &mut SilkNSQState,
        x_sc_q10: &[i32],
        pulses: &mut [i8],
        xq_offset: usize,
        s_ltp_q15: &mut [i32],
        p: &SubframeParams,
        prediction: Option<&LongTermPrediction>,
        shaping: Option<&HarmonicShaping>,
    ) {
        assert!(p.ar_shp_q13.len() <= FEEDBACK_MAX_ORDER);
        quantize_subframe_cases::<true>(
            nsq, x_sc_q10, pulses, xq_offset, s_ltp_q15, p, prediction, shaping,
        );
    }
}

/// The quantization level of a residual as libopus `silk_noise_shape_quantizer`
/// chooses it: the two levels next to the residual `r_q10` are compared by
/// rate-distortion cost, and the cheaper one is returned (Q10).
#[inline(always)]
fn choose_level(r_q10: i32, offset_q10: i32, lambda_q10: i32) -> i32 {
    let mut q1_q10 = r_q10 - offset_q10;
    let mut q1_q0 = q1_q10 >> 10;
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

    let q2_q10: i32;
    let mut rd1_q20: i32;
    let mut rd2_q20: i32;
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
    let rr_q10 = r_q10 - q1_q10;
    rd1_q20 = silk_smlabb(rd1_q20, rr_q10, rr_q10);
    let rr_q10 = r_q10 - q2_q10;
    rd2_q20 = silk_smlabb(rd2_q20, rr_q10, rr_q10);

    if rd2_q20 < rd1_q20 { q2_q10 } else { q1_q10 }
}

/// Lowest and highest value of the residual that reaches the level decision.
const R_MIN_Q10: i32 = -(31 << 10);
const R_MAX_Q10: i32 = 30 << 10;

/// One subframe of the plain quantizer (libopus `silk_noise_shape_quantizer`).
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
) {
    // The sample loop reads and writes these sizes.
    assert!(subfr_length <= MAX_SUB_FRAME_LENGTH);
    assert!(predict_lpc_order <= MAX_LPC_ORDER);
    // libopus requires an even shaping order too (`celt_assert`).
    assert!(shaping_lpc_order >= 2 && shaping_lpc_order.is_multiple_of(2));
    assert!(shaping_lpc_order <= MAX_SHAPE_LPC_ORDER);

    let mut a = [0i16; MAX_LPC_ORDER];
    a[..predict_lpc_order].copy_from_slice(&a_q12[..predict_lpc_order]);
    let p = SubframeParams {
        a_q12: &a,
        ar_shp_q13: &ar_shp_q13[..shaping_lpc_order],
        tilt_q14,
        lf_shp_q14,
        gain_q10: gain_q16 >> 6,
        lambda_q10,
        offset_q10,
        predict_lpc_order,
        #[cfg(target_arch = "aarch64")]
        lpc_neon: ShortPredictionNeon::new(&a),
        #[cfg(target_arch = "x86_64")]
        lpc_x86: x86::ShortPrediction::new(&a),
    };
    let x_sc_q10 = &x_sc_q10[..subfr_length];

    // Long-term prediction belongs to voiced subframes. Harmonic noise
    // shaping runs whenever there is a lag, as in libopus: an unvoiced frame
    // keeps the lag of the frame before it, and its shaping gain is the
    // smoothed gain, which is not zero right after a voiced frame.
    let prediction = if signal_type == TYPE_VOICED {
        let b_q14: &[i16; LTP_ORDER] = b_q14.first_chunk().unwrap();
        Some(LongTermPrediction {
            b_q14,
            pred_lag_base: (nsq.s_ltp_buf_idx as usize).wrapping_sub(lag) + LTP_ORDER / 2,
        })
    } else {
        None
    };
    let shaping = if lag > 0 {
        Some(HarmonicShaping {
            fir_packed_q14: harm_shape_fir_packed_q14,
            shp_lag_base: (nsq.s_ltp_shp_buf_idx as usize).wrapping_sub(lag)
                + HARM_SHAPE_FIR_TAPS / 2,
        })
    } else {
        None
    };
    macro_rules! run {
        ($in_registers:literal) => {
            quantize_subframe_cases::<$in_registers>(
                nsq,
                x_sc_q10,
                pulses,
                xq_offset,
                s_ltp_q15,
                &p,
                prediction.as_ref(),
                shaping.as_ref(),
            )
        };
    }
    #[cfg(target_arch = "aarch64")]
    if shaping_lpc_order <= FEEDBACK_NEON_MAX_ORDER {
        run!(true);
    } else {
        run!(false);
    }
    #[cfg(target_arch = "x86_64")]
    if shaping_lpc_order <= x86::FEEDBACK_MAX_ORDER && crate::compat::x86_has_sse41() {
        // SAFETY: the processor has SSE4.1 (detected above).
        unsafe {
            x86::quantize_subframe_sse41(
                nsq,
                x_sc_q10,
                pulses,
                xq_offset,
                s_ltp_q15,
                &p,
                prediction.as_ref(),
                shaping.as_ref(),
            )
        };
    } else {
        run!(false);
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    run!(false);

    nsq.s_lpc_q14
        .copy_within(subfr_length..subfr_length + NSQ_LPC_BUF_LENGTH, 0);
}

/// Values that are the same for every sample of one subframe.
struct SubframeParams<'a> {
    /// `a_q12[j]` multiplies the sample `j + 1` steps back. Entries from the
    /// prediction order on are zero.
    a_q12: &'a [i16; MAX_LPC_ORDER],
    /// Its length is the shaping order.
    ar_shp_q13: &'a [i16],
    tilt_q14: i32,
    lf_shp_q14: i32,
    gain_q10: i32,
    lambda_q10: i32,
    offset_q10: i32,
    predict_lpc_order: usize,
    #[cfg(target_arch = "aarch64")]
    lpc_neon: ShortPredictionNeon,
    #[cfg(target_arch = "x86_64")]
    lpc_x86: x86::ShortPrediction,
}

/// Long-term prediction of a voiced subframe.
struct LongTermPrediction<'a> {
    b_q14: &'a [i16; LTP_ORDER],
    /// Index into `s_ltp_q15` of the newest of the five samples the
    /// prediction of sample 0 reads.
    pred_lag_base: usize,
}

/// Harmonic noise shaping of a subframe that has a lag.
struct HarmonicShaping {
    /// The two filter coefficients in one word (libopus
    /// `HarmShapeFIRPacked_Q14`).
    fir_packed_q14: i32,
    /// Index into `s_ltp_shp_q14` of the newest of the three samples the
    /// shaping of sample 0 reads.
    shp_lag_base: usize,
}

/// `quantize_subframe` with one call per combination of long-term
/// prediction and harmonic shaping, so that each combination has its own
/// copy of the loop.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn quantize_subframe_cases<const FEEDBACK_IN_REGISTERS: bool>(
    nsq: &mut SilkNSQState,
    x_sc_q10: &[i32],
    pulses: &mut [i8],
    xq_offset: usize,
    s_ltp_q15: &mut [i32],
    p: &SubframeParams,
    prediction: Option<&LongTermPrediction>,
    shaping: Option<&HarmonicShaping>,
) {
    match (prediction, shaping) {
        (Some(prediction), Some(shaping)) => quantize_subframe::<FEEDBACK_IN_REGISTERS>(
            nsq,
            x_sc_q10,
            pulses,
            xq_offset,
            s_ltp_q15,
            p,
            Some(prediction),
            Some(shaping),
        ),
        (None, Some(shaping)) => quantize_subframe::<FEEDBACK_IN_REGISTERS>(
            nsq,
            x_sc_q10,
            pulses,
            xq_offset,
            s_ltp_q15,
            p,
            None,
            Some(shaping),
        ),
        (None, None) => quantize_subframe::<FEEDBACK_IN_REGISTERS>(
            nsq, x_sc_q10, pulses, xq_offset, s_ltp_q15, p, None, None,
        ),
        // A voiced subframe without a lag does not occur (libopus asserts
        // it); this is what the libopus loop computes for it.
        (Some(prediction), None) => quantize_subframe::<FEEDBACK_IN_REGISTERS>(
            nsq,
            x_sc_q10,
            pulses,
            xq_offset,
            s_ltp_q15,
            p,
            Some(prediction),
            None,
        ),
    }
}

/// The sample loop of [`silk_noise_shape_quantizer`]. `prediction` is `Some`
/// for a voiced subframe, `shaping` for a subframe with a lag. The function
/// is inlined into every call, so each combination has its own copy of the
/// loop.
///
/// With `FEEDBACK_IN_REGISTERS` (aarch64, shaping order at most
/// `FEEDBACK_NEON_MAX_ORDER`) the shaping feedback runs on `FeedbackNeon`.
/// On x86-64 it selects the SSE4.1 kernels of `mod x86`, for the short-term
/// prediction too; `x86::quantize_subframe_sse41` passes it.
#[inline(always)]
fn quantize_subframe<const FEEDBACK_IN_REGISTERS: bool>(
    nsq: &mut SilkNSQState,
    x_sc_q10: &[i32],
    pulses: &mut [i8],
    xq_offset: usize,
    s_ltp_q15: &mut [i32],
    p: &SubframeParams,
    prediction: Option<&LongTermPrediction>,
    shaping: Option<&HarmonicShaping>,
) {
    let subfr_length = x_sc_q10.len();
    let shaping_lpc_order = p.ar_shp_q13.len();
    let pulses = &mut pulses[..subfr_length];
    let xq = &mut nsq.xq[xq_offset..xq_offset + subfr_length];
    let s_lpc_q14 = &mut nsq.s_lpc_q14[..NSQ_LPC_BUF_LENGTH + subfr_length];
    let s_ar2_q14 = &mut nsq.s_ar2_q14[..shaping_lpc_order];
    #[cfg(target_arch = "aarch64")]
    let mut feedback = FeedbackNeon::new(
        &s_ar2_q14[..shaping_lpc_order.min(FEEDBACK_NEON_MAX_ORDER)],
        &p.ar_shp_q13[..shaping_lpc_order.min(FEEDBACK_NEON_MAX_ORDER)],
    );
    // The SSE4.1 kernels below rely on this check.
    #[cfg(target_arch = "x86_64")]
    assert!(!FEEDBACK_IN_REGISTERS || crate::compat::x86_has_sse41());
    #[cfg(target_arch = "x86_64")]
    let mut feedback = x86::Feedback::new(
        &s_ar2_q14[..shaping_lpc_order.min(x86::FEEDBACK_MAX_ORDER)],
        &p.ar_shp_q13[..shaping_lpc_order.min(x86::FEEDBACK_MAX_ORDER)],
    );

    let mut rand_seed = nsq.rand_seed;
    let mut s_ltp_shp_buf_idx = nsq.s_ltp_shp_buf_idx as usize;
    let mut s_ltp_buf_idx = nsq.s_ltp_buf_idx as usize;
    let mut s_lf_ar_q14 = nsq.s_lf_ar_q14;
    let mut s_diff_shp_q14 = nsq.s_diff_shp_q14;
    // Copies of the two values the next sample needs first. The sample loop
    // is one long chain of dependent operations from each sample to the
    // next. Reading these two back from the arrays they were just stored to
    // would put a store and a load on that chain.
    let mut newest_xq_q14 = s_lpc_q14[NSQ_LPC_BUF_LENGTH - 1];
    // The SSE4.1 short-term prediction takes the sample before it from a
    // register as well.
    #[cfg(target_arch = "x86_64")]
    let mut second_xq_q14 = s_lpc_q14[NSQ_LPC_BUF_LENGTH - 2];
    let mut newest_shp_q14 = if subfr_length > 0 {
        nsq.s_ltp_shp_q14[s_ltp_shp_buf_idx - 1]
    } else {
        0
    };

    for i in 0..subfr_length {
        rand_seed = silk_rand(rand_seed);

        let window: &[i32; NSQ_LPC_BUF_LENGTH] =
            s_lpc_q14[i..i + NSQ_LPC_BUF_LENGTH].try_into().unwrap();
        // One copy of the sum per order the codec uses, with the number of
        // taps known at compile time.
        #[cfg(not(target_arch = "aarch64"))]
        let scalar_prediction = || match p.predict_lpc_order {
            16 => short_prediction(window, p.a_q12, 16, newest_xq_q14),
            10 => short_prediction(window, p.a_q12, 10, newest_xq_q14),
            6 => short_prediction(window, p.a_q12, 6, newest_xq_q14),
            order => short_prediction(window, p.a_q12, order, newest_xq_q14),
        };
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        let lpc_pred_q10 = scalar_prediction();
        #[cfg(target_arch = "x86_64")]
        let lpc_pred_q10 = if FEEDBACK_IN_REGISTERS {
            // SAFETY: the processor has SSE4.1 (asserted above).
            unsafe {
                x86::short_prediction(
                    window,
                    &p.lpc_x86,
                    (p.predict_lpc_order as i32) >> 1,
                    p.a_q12,
                    second_xq_q14,
                    newest_xq_q14,
                )
            }
        } else {
            scalar_prediction()
        };
        #[cfg(target_arch = "aarch64")]
        let lpc_pred_q10 = short_prediction_neon(
            window,
            &p.lpc_neon,
            (p.predict_lpc_order as i32) >> 1,
            p.a_q12[0],
            newest_xq_q14,
        );

        let scalar_feedback = |state: &mut [i32]| match shaping_lpc_order {
            12 => noise_shape_feedback_order::<12>(s_diff_shp_q14, state, p.ar_shp_q13),
            14 => noise_shape_feedback_order::<14>(s_diff_shp_q14, state, p.ar_shp_q13),
            16 => noise_shape_feedback_order::<16>(s_diff_shp_q14, state, p.ar_shp_q13),
            _ => noise_shape_feedback(s_diff_shp_q14, state, p.ar_shp_q13),
        };
        #[cfg(target_arch = "aarch64")]
        let mut n_ar_q12 = if FEEDBACK_IN_REGISTERS {
            feedback.step(s_diff_shp_q14)
        } else {
            scalar_feedback(s_ar2_q14)
        };
        #[cfg(target_arch = "x86_64")]
        let mut n_ar_q12 = if FEEDBACK_IN_REGISTERS {
            // SAFETY: the processor has SSE4.1 (asserted above).
            unsafe { feedback.step(s_diff_shp_q14) }
        } else {
            scalar_feedback(s_ar2_q14)
        };
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        let mut n_ar_q12 = scalar_feedback(s_ar2_q14);
        n_ar_q12 = silk_smlawb(n_ar_q12, s_lf_ar_q14, p.tilt_q14);

        let mut n_lf_q12 = silk_smulwb(newest_shp_q14, p.lf_shp_q14);
        n_lf_q12 = silk_smlawt(n_lf_q12, s_lf_ar_q14, p.lf_shp_q14);

        let mut tmp1 = (lpc_pred_q10 << 2).wrapping_sub(n_ar_q12);
        tmp1 = tmp1.wrapping_sub(n_lf_q12);

        let mut ltp_pred_q13 = 0i32;
        if let Some(lt) = prediction {
            let pred_idx = lt.pred_lag_base + i;
            let taps: &[i32; LTP_ORDER] = s_ltp_q15[pred_idx - (LTP_ORDER - 1)..=pred_idx]
                .try_into()
                .unwrap();
            ltp_pred_q13 = 2;
            for (&sample, &b) in taps.iter().rev().zip(lt.b_q14) {
                ltp_pred_q13 = silk_smlawb(ltp_pred_q13, sample, b as i32);
            }
        }

        if let Some(hs) = shaping {
            let shp_idx = hs.shp_lag_base + i;
            let shp: &[i32; HARM_SHAPE_FIR_TAPS] = nsq.s_ltp_shp_q14
                [shp_idx - (HARM_SHAPE_FIR_TAPS - 1)..=shp_idx]
                .try_into()
                .unwrap();
            let mut n_ltp_q13 = silk_smulwb(silk_add_sat32(shp[2], shp[0]), hs.fir_packed_q14);
            n_ltp_q13 = silk_smlawt(n_ltp_q13, shp[1], hs.fir_packed_q14);
            n_ltp_q13 <<= 1;

            let tmp2 = ltp_pred_q13.wrapping_sub(n_ltp_q13);
            tmp1 = tmp2.wrapping_add(tmp1 << 1);
            tmp1 = silk_rshift_round(tmp1, 3);
        } else {
            tmp1 = silk_rshift_round(tmp1, 2);
        }

        let mut r_q10 = x_sc_q10[i] - tmp1;
        if rand_seed < 0 {
            r_q10 = -r_q10;
        }
        r_q10 = silk_limit_32(r_q10, R_MIN_Q10, R_MAX_Q10);

        let q_q10 = choose_level(r_q10, p.offset_q10, p.lambda_q10);

        let pulse = silk_rshift_round(q_q10, 10) as i8;
        pulses[i] = pulse;

        let mut exc_q14 = q_q10 << 4;
        if rand_seed < 0 {
            exc_q14 = -exc_q14;
        }

        let lpc_exc_q14 = silk_add_lshift32(exc_q14, ltp_pred_q13, 1);
        let xq_q14 = lpc_exc_q14.wrapping_add(lpc_pred_q10 << 4);

        xq[i] = silk_sat16(silk_rshift_round(silk_smulww(xq_q14, p.gain_q10), 8)) as i16;

        s_lpc_q14[NSQ_LPC_BUF_LENGTH + i] = xq_q14;
        #[cfg(target_arch = "x86_64")]
        {
            second_xq_q14 = newest_xq_q14;
        }
        newest_xq_q14 = xq_q14;
        s_diff_shp_q14 = xq_q14.wrapping_sub(x_sc_q10[i] << 4);
        let s_lf_ar_shp_q14 = s_diff_shp_q14.wrapping_sub(n_ar_q12 << 2);
        s_lf_ar_q14 = s_lf_ar_shp_q14;

        newest_shp_q14 = s_lf_ar_shp_q14.wrapping_sub(n_lf_q12 << 2);
        nsq.s_ltp_shp_q14[s_ltp_shp_buf_idx] = newest_shp_q14;
        s_ltp_q15[s_ltp_buf_idx] = lpc_exc_q14 << 1;
        s_ltp_shp_buf_idx += 1;
        s_ltp_buf_idx += 1;

        rand_seed = rand_seed.wrapping_add(pulse as i32);
    }

    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    if FEEDBACK_IN_REGISTERS {
        feedback.store(s_ar2_q14);
    }
    nsq.rand_seed = rand_seed;
    nsq.s_ltp_shp_buf_idx = s_ltp_shp_buf_idx as i32;
    nsq.s_ltp_buf_idx = s_ltp_buf_idx as i32;
    nsq.s_lf_ar_q14 = s_lf_ar_q14;
    nsq.s_diff_shp_q14 = s_diff_shp_q14;
}

#[cfg(test)]
mod tests {
    #[cfg(not(feature = "std"))]
    use std::vec::Vec;
    use super::*;

    /// xorshift64 generator with a fixed seed.
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }

        /// A value of up to 30 bits plus sign, or one of the edges of `i32`.
        fn value(&mut self) -> i32 {
            match self.next() % 8 {
                0 => i32::MIN,
                1 => i32::MAX,
                2 => 0,
                n => (self.next() as i32) >> (n * 3),
            }
        }

        fn coef(&mut self) -> i16 {
            match self.next() % 8 {
                0 => i16::MIN,
                1 => i16::MAX,
                _ => self.next() as i16,
            }
        }
    }

    /// libopus `silk_NSQ_noise_shape_feedback_loop_c`, statement by
    /// statement.
    fn feedback_ref(data0: i32, data1: &mut [i32], coef: &[i16], order: usize) -> i32 {
        let mut tmp2 = data0;
        let mut tmp1 = data1[0];
        data1[0] = tmp2;
        let mut out = (order as i32) >> 1;
        out = silk_smlawb(out, tmp2, coef[0] as i32);
        let mut j = 2;
        while j < order {
            tmp2 = data1[j - 1];
            data1[j - 1] = tmp1;
            out = silk_smlawb(out, tmp1, coef[j - 1] as i32);
            tmp1 = data1[j];
            data1[j] = tmp2;
            out = silk_smlawb(out, tmp2, coef[j] as i32);
            j += 2;
        }
        data1[order - 1] = tmp1;
        out = silk_smlawb(out, tmp1, coef[order - 1] as i32);
        out << 1
    }

    #[test]
    fn noise_shape_feedback_matches_reference() {
        let mut rng = Rng(0x243F_6A88_85A3_08D3);
        for case in 0..3000usize {
            // Every even order, also the ones the codec does not use.
            let order = 2 * (1 + case % (MAX_SHAPE_LPC_ORDER / 2));
            let coef: Vec<i16> = (0..order).map(|_| rng.coef()).collect();
            let start: Vec<i32> = (0..MAX_SHAPE_LPC_ORDER).map(|_| rng.value()).collect();
            let newest = rng.value();

            let mut want_state = start.clone();
            let want = feedback_ref(newest, &mut want_state, &coef, order);

            let mut state = start.clone();
            let got = noise_shape_feedback(newest, &mut state[..order], &coef);
            assert_eq!((got, &state), (want, &want_state), "order {order}");

            // The copies with the order known at compile time.
            let mut state = start.clone();
            let got = match order {
                12 => noise_shape_feedback_order::<12>(newest, &mut state[..order], &coef),
                14 => noise_shape_feedback_order::<14>(newest, &mut state[..order], &coef),
                16 => noise_shape_feedback_order::<16>(newest, &mut state[..order], &coef),
                _ => continue,
            };
            assert_eq!((got, &state), (want, &want_state), "fixed order {order}");
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn feedback_in_registers_matches_reference() {
        let mut rng = Rng(0x0A2F_31B5_77C3_9E01);
        for case in 0..2000usize {
            // Every even order up to the limit of the register version.
            let order = 2 * (1 + case % (FEEDBACK_NEON_MAX_ORDER / 2));
            let coef: Vec<i16> = (0..order).map(|_| rng.coef()).collect();
            let start: Vec<i32> = (0..order).map(|_| rng.value()).collect();

            let mut want_state = start.clone();
            let mut neon = FeedbackNeon::new(&start, &coef);
            // Several steps, so that the shifts carry values between the
            // registers.
            for step in 0..20 {
                let newest = rng.value();
                let want = feedback_ref(newest, &mut want_state, &coef, order);
                let got = neon.step(newest);
                assert_eq!(got, want, "order {order}, case {case}, step {step}");
            }
            let mut got_state = vec![0x0bad_f00d; order];
            neon.store(&mut got_state);
            assert_eq!(got_state, want_state, "order {order}, case {case}");
        }
    }

    #[test]
    fn short_prediction_matches_definition() {
        let mut rng = Rng(0x1319_8A2E_0370_7344);
        for case in 0..4000usize {
            let order = case % (MAX_LPC_ORDER + 1);
            let window: [i32; NSQ_LPC_BUF_LENGTH] = core::array::from_fn(|_| rng.value());
            let mut a_q12 = [0i16; MAX_LPC_ORDER];
            for a in a_q12[..order].iter_mut() {
                *a = rng.coef();
            }
            // libopus: one tap after the other, the newest sample first.
            let mut want = (order as i32) >> 1;
            for j in 0..order {
                want = silk_smlawb(want, window[NSQ_LPC_BUF_LENGTH - 1 - j], a_q12[j] as i32);
            }
            // The last entry of the window is not read: the newest sample
            // comes in separately.
            let mut window_without_newest = window;
            window_without_newest[NSQ_LPC_BUF_LENGTH - 1] = 0x0bad_f00d;
            let got = short_prediction(
                &window_without_newest,
                &a_q12,
                order,
                window[NSQ_LPC_BUF_LENGTH - 1],
            );
            assert_eq!(got, want, "order {order}");
            #[cfg(target_arch = "aarch64")]
            {
                let got = short_prediction_neon(
                    &window_without_newest,
                    &ShortPredictionNeon::new(&a_q12),
                    (order as i32) >> 1,
                    a_q12[0],
                    window[NSQ_LPC_BUF_LENGTH - 1],
                );
                assert_eq!(got, want, "NEON, order {order}");
            }
            #[cfg(target_arch = "x86_64")]
            if crate::compat::x86_has_sse41() {
                // Positions 14 and 15 come in separately.
                let mut stale = window_without_newest;
                stale[NSQ_LPC_BUF_LENGTH - 2] = -0x0bad_f00d;
                // SAFETY: the processor has SSE4.1 (checked above).
                let got = unsafe {
                    x86::short_prediction(
                        &stale,
                        &x86::ShortPrediction::new(&a_q12),
                        (order as i32) >> 1,
                        &a_q12,
                        window[NSQ_LPC_BUF_LENGTH - 2],
                        window[NSQ_LPC_BUF_LENGTH - 1],
                    )
                };
                assert_eq!(got, want, "SSE4.1, order {order}");
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn x86_feedback_matches_reference() {
        if !crate::compat::x86_has_sse41() {
            return;
        }
        let mut rng = Rng(0x7C1E_5D0A_2B94_6F83);
        for case in 0..2000usize {
            // Every even order up to the limit of the register version.
            let order = 2 * (1 + case % (x86::FEEDBACK_MAX_ORDER / 2));
            let coef: Vec<i16> = (0..order).map(|_| rng.coef()).collect();
            let start: Vec<i32> = (0..order).map(|_| rng.value()).collect();

            let mut want_state = start.clone();
            let mut feedback = x86::Feedback::new(&start, &coef);
            // Several steps, so that the shifts carry values between the
            // registers.
            for step in 0..20 {
                let newest = rng.value();
                let want = feedback_ref(newest, &mut want_state, &coef, order);
                // SAFETY: the processor has SSE4.1 (checked above).
                let got = unsafe { feedback.step(newest) };
                assert_eq!(got, want, "order {order}, case {case}, step {step}");
            }
            let mut got_state = vec![0x0bad_f00d; order];
            feedback.store(&mut got_state);
            assert_eq!(got_state, want_state, "order {order}, case {case}");
        }
    }

    /// The sample loop with the SSE4.1 kernels against the scalar loop, for
    /// every combination of long-term prediction and harmonic shaping, the
    /// shaping orders up to the limit of the register version, and lengths
    /// from 0 to the longest subframe.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn x86_subframe_matches_scalar_loop() {
        if !crate::compat::x86_has_sse41() {
            return;
        }
        let mut rng = Rng(0x3F84_D5B5_B547_0917);
        let lengths = [0usize, 1, 2, 3, 5, 39, 40, 41, 79, 80];
        for case in 0..800usize {
            let length = lengths[case % lengths.len()];
            let shaping_order = [2, 4, 12, 14, 16][(rng.next() % 5) as usize];
            let predict_order = [0, 6, 10, 16][(rng.next() % 4) as usize];
            let voiced = case % 4 < 2;
            let lag = if voiced || case % 4 == 2 {
                40 + (rng.next() % 260) as usize
            } else {
                0
            };
            // Moderate values where the scalar loop adds without wrapping,
            // values up to the edges of `i32` elsewhere.
            let moderate = |rng: &mut Rng| rng.value() >> 9;
            let ltp_mem = 320;
            let mut nsq = SilkNSQState::default();
            for v in nsq.s_ltp_shp_q14.iter_mut() {
                *v = moderate(&mut rng);
            }
            for v in nsq.s_lpc_q14.iter_mut() {
                *v = rng.value();
            }
            for v in nsq.s_ar2_q14.iter_mut() {
                *v = rng.value();
            }
            nsq.s_lf_ar_q14 = rng.value();
            nsq.s_diff_shp_q14 = rng.value();
            nsq.rand_seed = rng.value();
            nsq.s_ltp_buf_idx = ltp_mem;
            nsq.s_ltp_shp_buf_idx = ltp_mem;
            let mut s_ltp_q15 = [0i32; 2 * MAX_FRAME_LENGTH];
            for v in s_ltp_q15.iter_mut() {
                *v = moderate(&mut rng);
            }
            let x_sc_q10: Vec<i32> = (0..length).map(|_| rng.value() >> 11).collect();
            let mut a_q12 = [0i16; MAX_LPC_ORDER];
            for a in a_q12[..predict_order].iter_mut() {
                *a = rng.coef();
            }
            let ar_shp_q13: Vec<i16> = (0..shaping_order).map(|_| rng.coef()).collect();
            let b_q14: [i16; LTP_ORDER] = core::array::from_fn(|_| rng.coef());
            let lambda_q10 = (rng.next() % 4000) as i32;
            let p = SubframeParams {
                a_q12: &a_q12,
                ar_shp_q13: &ar_shp_q13,
                tilt_q14: rng.value(),
                lf_shp_q14: rng.value(),
                gain_q10: rng.value() >> 6,
                lambda_q10,
                offset_q10: [32, 100, 240][(rng.next() % 3) as usize],
                predict_lpc_order: predict_order,
                lpc_x86: x86::ShortPrediction::new(&a_q12),
            };
            let prediction = voiced.then(|| LongTermPrediction {
                b_q14: &b_q14,
                pred_lag_base: ltp_mem as usize - lag + LTP_ORDER / 2,
            });
            let shaping = (lag > 0).then(|| HarmonicShaping {
                fir_packed_q14: rng.value(),
                shp_lag_base: ltp_mem as usize - lag + HARM_SHAPE_FIR_TAPS / 2,
            });

            let xq_offset = ltp_mem as usize;
            let (mut nsq_vec, mut s_ltp_vec) = (nsq, s_ltp_q15);
            let mut pulses = [0i8; MAX_SUB_FRAME_LENGTH];
            let mut pulses_vec = pulses;
            quantize_subframe::<false>(
                &mut nsq,
                &x_sc_q10,
                &mut pulses,
                xq_offset,
                &mut s_ltp_q15,
                &p,
                prediction.as_ref(),
                shaping.as_ref(),
            );
            // SAFETY: the processor has SSE4.1 (checked above).
            unsafe {
                x86::quantize_subframe_sse41(
                    &mut nsq_vec,
                    &x_sc_q10,
                    &mut pulses_vec,
                    xq_offset,
                    &mut s_ltp_vec,
                    &p,
                    prediction.as_ref(),
                    shaping.as_ref(),
                )
            };
            let what = format!(
                "case {case}, length {length}, orders {shaping_order} {predict_order}, lag {lag}"
            );
            assert_eq!(pulses_vec, pulses, "{what}: pulses");
            assert_eq!(s_ltp_vec, s_ltp_q15, "{what}: s_ltp_q15");
            assert_eq!(nsq_vec.xq, nsq.xq, "{what}: xq");
            assert_eq!(nsq_vec.s_ltp_shp_q14, nsq.s_ltp_shp_q14, "{what}: s_ltp_shp_q14");
            assert_eq!(nsq_vec.s_lpc_q14, nsq.s_lpc_q14, "{what}: s_lpc_q14");
            assert_eq!(nsq_vec.s_ar2_q14, nsq.s_ar2_q14, "{what}: s_ar2_q14");
            let scalars = |n: &SilkNSQState| {
                [
                    n.s_lf_ar_q14,
                    n.s_diff_shp_q14,
                    n.rand_seed,
                    n.s_ltp_buf_idx,
                    n.s_ltp_shp_buf_idx,
                ]
            };
            assert_eq!(scalars(&nsq_vec), scalars(&nsq), "{what}: state");
        }
    }
}
