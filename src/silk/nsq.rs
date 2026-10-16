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
    // One call per combination, so that each has its own copy of the loop.
    match (&prediction, &shaping) {
        (Some(prediction), Some(shaping)) => quantize_subframe(
            nsq,
            x_sc_q10,
            pulses,
            xq_offset,
            s_ltp_q15,
            &p,
            Some(prediction),
            Some(shaping),
        ),
        (None, Some(shaping)) => {
            quantize_subframe(nsq, x_sc_q10, pulses, xq_offset, s_ltp_q15, &p, None, Some(shaping))
        }
        (None, None) => quantize_subframe(nsq, x_sc_q10, pulses, xq_offset, s_ltp_q15, &p, None, None),
        // A voiced subframe without a lag does not occur (libopus asserts
        // it); this is what the libopus loop computes for it.
        (Some(prediction), None) => {
            quantize_subframe(nsq, x_sc_q10, pulses, xq_offset, s_ltp_q15, &p, Some(prediction), None)
        }
    }

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

/// The sample loop of [`silk_noise_shape_quantizer`]. `prediction` is `Some`
/// for a voiced subframe, `shaping` for a subframe with a lag. The function
/// is inlined into every call, so each combination has its own copy of the
/// loop.
#[inline(always)]
fn quantize_subframe(
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
        let lpc_pred_q10 = match p.predict_lpc_order {
            16 => short_prediction(window, p.a_q12, 16, newest_xq_q14),
            10 => short_prediction(window, p.a_q12, 10, newest_xq_q14),
            6 => short_prediction(window, p.a_q12, 6, newest_xq_q14),
            order => short_prediction(window, p.a_q12, order, newest_xq_q14),
        };

        let mut n_ar_q12 = match shaping_lpc_order {
            12 => noise_shape_feedback_order::<12>(s_diff_shp_q14, s_ar2_q14, p.ar_shp_q13),
            14 => noise_shape_feedback_order::<14>(s_diff_shp_q14, s_ar2_q14, p.ar_shp_q13),
            16 => noise_shape_feedback_order::<16>(s_diff_shp_q14, s_ar2_q14, p.ar_shp_q13),
            _ => noise_shape_feedback(s_diff_shp_q14, s_ar2_q14, p.ar_shp_q13),
        };
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
        }
    }
}
