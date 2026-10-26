use crate::silk::decoder_structs::{SilkDecoderControl, SilkDecoderState};
use crate::silk::define::*;
use crate::silk::macros::*;
use crate::silk::tables::SILK_QUANTIZATION_OFFSETS_Q10;
use core::mem::MaybeUninit;

/// Longest long-term-prediction buffer: 20 ms of history and one 20 ms frame
/// at 16 kHz.
const MAX_S_LTP_Q15: usize = 2 * MAX_FRAME_LENGTH;

pub fn silk_decode_core(
    ps_dec: &mut SilkDecoderState,
    ps_dec_ctrl: &mut SilkDecoderControl,
    xq: &mut [i16],
    pulses: &[i16],
) {
    let nlsf_interpolation_flag = ps_dec.indices.nlsf_interp_coef_q2 < 4;

    let offset_q10 = SILK_QUANTIZATION_OFFSETS_Q10[(ps_dec.indices.signal_type >> 1) as usize]
        [ps_dec.indices.quant_offset_type as usize] as i32;

    let frame_length = ps_dec.frame_length as usize;
    let subfr_length = ps_dec.subfr_length as usize;
    let ltp_mem_length = ps_dec.ltp_mem_length as usize;
    let lpc_order = ps_dec.lpc_order as usize;

    decode_excitation(
        &mut ps_dec.exc_q14[..frame_length],
        &pulses[..frame_length],
        ps_dec.indices.seed as i32,
        offset_q10,
    );

    let mut s_lpc_q14 = [0i32; MAX_SUB_FRAME_LENGTH + MAX_LPC_ORDER];
    s_lpc_q14[..MAX_LPC_ORDER].copy_from_slice(&ps_dec.s_lpc_q14_buf);
    let mut res_q14 = [0i32; MAX_SUB_FRAME_LENGTH];

    // State of the long-term predictor, in Q15. Index `ltp_mem_length` is
    // the first sample of this frame. libopus leaves this buffer
    // uninitialized as well; clearing its 2.5 kB on every frame is wasted
    // work, and unvoiced frames do not use it at all.
    //
    // The elements from `s_ltp_init_from` up to `s_ltp_buf_idx` are
    // initialized. The code below keeps that true, and it reads no element
    // outside that range.
    let mut s_ltp_q15_buf = [MaybeUninit::<i32>::uninit(); MAX_S_LTP_Q15];
    let s_ltp_q15 = &mut s_ltp_q15_buf[..ltp_mem_length + frame_length];
    let mut s_ltp_buf_idx = ltp_mem_length;
    let mut s_ltp_init_from = ltp_mem_length;

    let mut exc_idx: usize = 0;

    for k in 0..ps_dec.nb_subfr as usize {
        debug_assert!(lpc_order == 10 || lpc_order == 16);
        let a_q12: [i16; MAX_LPC_ORDER] = ps_dec_ctrl.pred_coef_q12[k >> 1];
        let gain_q16 = ps_dec_ctrl.gains_q16[k];

        let mut inv_gain_q31 = silk_inverse32_varq(gain_q16, 47);

        let gain_adj_q16 = if gain_q16 != ps_dec.prev_gain_q16 {
            let adj = silk_div32_varq(ps_dec.prev_gain_q16, gain_q16, 16);

            for s in s_lpc_q14[..MAX_LPC_ORDER].iter_mut() {
                *s = silk_smulww(adj, *s);
            }
            adj
        } else {
            1 << 16
        };

        ps_dec.prev_gain_q16 = gain_q16;

        // libopus: "Avoid abrupt transition from voiced PLC to unvoiced
        // normal decoding". The first two subframes after a concealed
        // voiced frame are decoded as voiced, with one prediction tap of
        // 0.25 at the concealment's pitch lag. libopus writes both into the
        // control structure, so the pitch lag is also what the frame leaves
        // behind as `lag_prev` when it has only two subframes.
        let mut eff_signal_type = ps_dec.indices.signal_type as i32;
        if ps_dec.loss_cnt > 0
            && ps_dec.prev_signal_type == TYPE_VOICED
            && ps_dec.indices.signal_type as i32 != TYPE_VOICED
            && k < MAX_NB_SUBFR / 2
        {
            ps_dec_ctrl.ltp_coef_q14[k * LTP_ORDER..(k + 1) * LTP_ORDER]
                .copy_from_slice(&[0, 0, 1 << 12, 0, 0]);
            ps_dec_ctrl.pitch_l[k] = ps_dec.lag_prev;
            eff_signal_type = TYPE_VOICED;
        }

        let exc_q14 = &ps_dec.exc_q14[exc_idx..exc_idx + subfr_length];
        let res: &[i32] = if eff_signal_type == TYPE_VOICED {
            let mut b_q14 = [0i16; LTP_ORDER];
            b_q14.copy_from_slice(&ps_dec_ctrl.ltp_coef_q14[k * LTP_ORDER..(k + 1) * LTP_ORDER]);

            // Clamp the lag to the geometry of the CURRENT frame rate. The
            // loss-recovery path above can surface `lag_prev` from a previous
            // higher-bandwidth frame (e.g. 288 at 16 kHz); without this clamp
            // the re-whitening and the predictor below index out of bounds
            // after a WB -> NB switch under packet loss.
            // libopus keeps this invariant via the lagPrev reset in
            // silk_decoder_set_fs.
            let lag =
                ps_dec_ctrl.pitch_l[k].clamp(2 * ps_dec.fs_khz, 18 * ps_dec.fs_khz) as usize;

            // The predictor reads this many samples before the write
            // position: the lag and half the filter length.
            let hist_len = lag + LTP_ORDER / 2;
            let hist_start = s_ltp_buf_idx - hist_len;

            if k == 0 || (k == 2 && nlsf_interpolation_flag) {
                // Re-whitening: the predictor state is rebuilt from the
                // decoded signal with the current LPC coefficients.
                if k == 2 {
                    ps_dec.out_buf[ltp_mem_length..ltp_mem_length + 2 * subfr_length]
                        .copy_from_slice(&xq[..2 * subfr_length]);
                }
                if k == 0 {
                    // Scale the state down to reduce the dependency on the
                    // previous packet.
                    inv_gain_q31 =
                        silk_lshift(silk_smulwb(inv_gain_q31, ps_dec_ctrl.ltp_scale_q14), 2);
                }

                // The state covers the last `hist_len` samples of the 20 ms
                // history; `first` is the index of its first sample there.
                // libopus filters `lpc_order` more samples in front of it,
                // only to have the filter memory. Here the kernel reads that
                // memory straight from `out_buf`. Where the history is too
                // short for a full memory the result is zero. This needs a
                // lag that the clamp above rules out.
                let first = ltp_mem_length - hist_len;
                let zeros = lpc_order.saturating_sub(first).min(hist_len);
                // Both calls together write every element of `state`.
                let state = &mut s_ltp_q15[hist_start..s_ltp_buf_idx];
                state[..zeros].fill(MaybeUninit::new(0));
                let in_start = k * subfr_length + first + zeros - lpc_order;
                let input = &ps_dec.out_buf[in_start..k * subfr_length + ltp_mem_length];
                if lpc_order == MAX_LPC_ORDER {
                    rewhiten::<16>(&mut state[zeros..], input, &a_q12, inv_gain_q31);
                } else {
                    rewhiten::<10>(&mut state[zeros..], input, &a_q12, inv_gain_q31);
                }
            } else {
                if hist_start < s_ltp_init_from {
                    // Not reached by a valid stream: the lag grew by more
                    // than a subframe. The samples in between count as zero.
                    s_ltp_q15[hist_start..s_ltp_init_from].fill(MaybeUninit::new(0));
                }
                if gain_adj_q16 != (1 << 16) {
                    // Update the predictor state when the gain changes.
                    // SAFETY: the elements from `s_ltp_init_from` up to
                    // `s_ltp_buf_idx` are initialized, and so are those
                    // from `hist_start` on if that is lower: they were
                    // filled just above.
                    let state =
                        unsafe { s_ltp_q15[hist_start..s_ltp_buf_idx].assume_init_mut() };
                    for s in state.iter_mut() {
                        *s = silk_smulww(gain_adj_q16, *s);
                    }
                }
            }
            // In both branches everything from `hist_start` up to
            // `s_ltp_buf_idx` is initialized now. The re-whitening wrote all
            // of it. The other branch found it written by the subframes
            // before, or filled what was missing.
            s_ltp_init_from = s_ltp_init_from.min(hist_start);

            let res = &mut res_q14[..subfr_length];
            // SAFETY: the slice starts with the `hist_len` elements from
            // `hist_start` on, which are initialized (see above).
            unsafe {
                ltp_synthesis(
                    &mut s_ltp_q15[hist_start..s_ltp_buf_idx + subfr_length],
                    lag,
                    &b_q14,
                    exc_q14,
                    res,
                );
            }
            // `ltp_synthesis` has written the `subfr_length` new elements.
            s_ltp_buf_idx += subfr_length;
            res
        } else {
            exc_q14
        };

        // Short-term prediction: the new samples go behind the 16 samples of
        // filter state.
        let s_lpc = &mut s_lpc_q14[..MAX_LPC_ORDER + subfr_length];
        let bias = ps_dec.lpc_order >> 1;
        if lpc_order == MAX_LPC_ORDER {
            lpc_synthesis::<16>(s_lpc, res, &a_q12, bias);
        } else {
            lpc_synthesis::<10>(s_lpc, res, &a_q12, bias);
        }

        // Scale with the gain.
        scale_output(
            &mut xq[exc_idx..exc_idx + subfr_length],
            &s_lpc[MAX_LPC_ORDER..],
            gain_q16 >> 6,
        );

        s_lpc_q14.copy_within(subfr_length..subfr_length + MAX_LPC_ORDER, 0);
        exc_idx += subfr_length;
    }

    ps_dec
        .s_lpc_q14_buf
        .copy_from_slice(&s_lpc_q14[..MAX_LPC_ORDER]);
}

/// Multiplier and increment of `silk_rand`.
const RAND_MUL: u32 = 196314165;
const RAND_ADD: u32 = 907633515;

/// Number of random numbers `decode_excitation` computes per step.
const RAND_STEP: usize = 8;

/// `RAND_MUL` to the powers 1 to `RAND_STEP`.
const RAND_MUL_POW: [u32; RAND_STEP] = {
    let mut pow = [0u32; RAND_STEP];
    let mut m = 1u32;
    let mut j = 0;
    while j < RAND_STEP {
        m = m.wrapping_mul(RAND_MUL);
        pow[j] = m;
        j += 1;
    }
    pow
};

/// One excitation sample in Q14 from its pulse and from the random number
/// that decides the sign.
#[inline(always)]
fn excitation_sample(pulse: i16, rand: i32, offset_q14: i32) -> i32 {
    let mut exc = (pulse as i32) << 14;
    if exc > 0 {
        exc -= QUANT_LEVEL_ADJUST_Q10 << 4;
    } else if exc < 0 {
        exc += QUANT_LEVEL_ADJUST_Q10 << 4;
    }
    exc += offset_q14;
    if rand < 0 { -exc } else { exc }
}

/// Decode the excitation: `exc_q14[i]` from `pulses[i]`.
///
/// The sign of each sample comes from a random number generator whose seed
/// is updated with every pulse: `rand = silk_rand(seed)`, then
/// `seed = rand + pulse`. Computed one sample at a time, each sample has to
/// wait for the multiplication of the one before it.
///
/// The first loop below computes the random numbers, `RAND_STEP` per step.
/// Written out, the numbers of a step are
/// `rand[j] = seed * RAND_MUL^(j+1) + c[j]`, where `c[j]` depends on the
/// pulses of the step only. All arithmetic wraps at 32 bits, so these are
/// exactly the numbers the sample-by-sample loop produces. Only one
/// multiplication per step now depends on the step before.
///
/// The second loop turns pulses and random numbers into the excitation. Its
/// samples do not depend on each other.
fn decode_excitation(exc_q14: &mut [i32], pulses: &[i16], seed: i32, offset_q10: i32) {
    assert!(exc_q14.len() == pulses.len());
    let mut seed = seed as u32;

    let (rand_steps, rand_rest) = exc_q14.as_chunks_mut::<RAND_STEP>();
    let (pulse_steps, pulse_rest) = pulses.as_chunks::<RAND_STEP>();
    for (rand, pulse) in rand_steps.iter_mut().zip(pulse_steps) {
        let mut c = RAND_ADD;
        let mut next_seed = seed;
        for j in 0..RAND_STEP {
            let r = seed.wrapping_mul(RAND_MUL_POW[j]).wrapping_add(c);
            let p = pulse[j] as i32 as u32;
            rand[j] = r as i32;
            c = c.wrapping_add(p).wrapping_mul(RAND_MUL).wrapping_add(RAND_ADD);
            next_seed = r.wrapping_add(p);
        }
        seed = next_seed;
    }
    // Frame lengths are multiples of `RAND_STEP`; this is for other lengths.
    for (rand, &pulse) in rand_rest.iter_mut().zip(pulse_rest) {
        seed = silk_rand(seed as i32) as u32;
        *rand = seed as i32;
        seed = seed.wrapping_add(pulse as i32 as u32);
    }

    let offset_q14 = offset_q10 << 4;
    for (exc, &pulse) in exc_q14.iter_mut().zip(pulses) {
        *exc = excitation_sample(pulse, *exc, offset_q14);
    }
}

/// Re-whitening of the long-term predictor state (libopus:
/// `silk_LPC_analysis_filter` followed by the scaling loop in
/// `silk_decode_core`).
///
/// `input` holds `ORDER` samples of filter memory followed by one sample per
/// output. `out[m]` is the prediction residual of `input[ORDER + m]`, rounded
/// to 16 bits and multiplied by `inv_gain_q31`. Every element of `out` is
/// written.
#[inline(always)]
fn rewhiten<const ORDER: usize>(
    out: &mut [MaybeUninit<i32>],
    input: &[i16],
    a_q12: &[i16; MAX_LPC_ORDER],
    inv_gain_q31: i32,
) {
    #[cfg(target_arch = "aarch64")]
    rewhiten_neon::<ORDER>(out, input, a_q12, inv_gain_q31);
    #[cfg(target_arch = "x86_64")]
    if crate::compat::x86_has_sse41() {
        // SAFETY: the processor has SSE4.1.
        unsafe { x86::rewhiten_sse41::<ORDER>(out, input, a_q12, inv_gain_q31) };
        return;
    }
    #[cfg(not(target_arch = "aarch64"))]
    rewhiten_scalar::<ORDER>(out, input, a_q12, inv_gain_q31);
}

#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
#[inline(always)]
fn rewhiten_scalar<const ORDER: usize>(
    out: &mut [MaybeUninit<i32>],
    input: &[i16],
    a_q12: &[i16; MAX_LPC_ORDER],
    inv_gain_q31: i32,
) {
    assert!(input.len() == out.len() + ORDER);
    // Coefficients in the order of the input samples: oldest sample first.
    let mut a_rev = [0i16; ORDER];
    for (j, a) in a_rev.iter_mut().enumerate() {
        *a = a_q12[ORDER - 1 - j];
    }

    for (o, x) in out.iter_mut().zip(input.windows(ORDER + 1)) {
        let mut pred_q12 = 0i32;
        for j in 0..ORDER {
            pred_q12 = pred_q12.wrapping_add(silk_smulbb(x[j] as i32, a_rev[j] as i32));
        }
        let res_q12 = ((x[ORDER] as i32) << 12).wrapping_sub(pred_q12);
        let res = silk_sat16(silk_rshift_round(res_q12, 12));
        o.write(silk_smulwb(inv_gain_q31, res));
    }
}

/// `rewhiten_scalar` with four outputs per step.
///
/// The loop keeps its pointer loads and stores. The function is inlined
/// into the SILK decoder. With the ranges taken from `as_chunks` and
/// windows, or checked once and taken with `get_unchecked`, the compiler
/// assigned the registers of the decoder differently on Android, and
/// loops of the decoder got longer.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn rewhiten_neon<const ORDER: usize>(
    out: &mut [MaybeUninit<i32>],
    input: &[i16],
    a_q12: &[i16; MAX_LPC_ORDER],
    inv_gain_q31: i32,
) {
    use core::arch::aarch64::*;

    let n = out.len();
    assert!(input.len() == n + ORDER);
    let n4 = n & !3;

    // SAFETY: NEON is part of the aarch64 baseline. A step at `m` has
    // `m + 4 <= n`. It reads `input[m + j..m + j + 4]` for `j <= ORDER`; the
    // last index is `m + ORDER + 3 < n + ORDER`, the length of `input`
    // (checked above). It writes `out[m..m + 4]`.
    unsafe {
        let gain = vdupq_n_s32(inv_gain_q31);
        let mut m = 0;
        while m < n4 {
            let x = input.as_ptr().add(m);
            let mut pred_q12 = vdupq_n_s32(0);
            for j in 0..ORDER {
                // `x[j]` is the sample `ORDER - j` steps before the one
                // that is filtered.
                pred_q12 = vmlal_n_s16(pred_q12, vld1_s16(x.add(j)), a_q12[ORDER - 1 - j]);
            }
            let res_q12 = vsubq_s32(vshll_n_s16::<12>(vld1_s16(x.add(ORDER))), pred_q12);
            // Round to Q0 and saturate to 16 bits.
            let res = vqmovn_s32(vrshrq_n_s32::<12>(res_q12));
            // silk_smulwb(gain, res) per lane: `vshll_n_s16::<15>` makes
            // `smulwb_operand(res)`.
            let scaled = vqdmulhq_s32(gain, vshll_n_s16::<15>(res));
            vst1q_s32(out.as_mut_ptr().add(m) as *mut i32, scaled);
            m += 4;
        }
    }
    rewhiten_scalar::<ORDER>(&mut out[n4..], &input[n4..], a_q12, inv_gain_q31);
}

/// Long-term prediction for one subframe.
///
/// `state` is the part of the predictor state that the subframe uses: the
/// `lag + LTP_ORDER / 2` samples before the subframe, then one element per
/// sample of the subframe. Those receive the new predictor state, and
/// `res_q14` the excitation with the prediction added. Sample `i` is
/// predicted from the five state samples around the one `lag` samples
/// before it, which are `state[i..i + 5]`.
///
/// The newest sample a prediction reads lies `lag - 2` samples before the
/// one it writes. The subframe is therefore processed in blocks of at most
/// `lag - 2` samples: inside a block no sample depends on another sample of
/// the block, so several can be computed at once.
///
/// # Safety
/// The first `lag + LTP_ORDER / 2` elements of `state` must be initialized.
/// The function initializes all others.
#[inline(always)]
unsafe fn ltp_synthesis(
    state: &mut [MaybeUninit<i32>],
    lag: usize,
    b_q14: &[i16; LTP_ORDER],
    exc_q14: &[i32],
    res_q14: &mut [i32],
) {
    let n = exc_q14.len();
    let hist_len = lag + LTP_ORDER / 2;
    assert!(res_q14.len() == n && state.len() == hist_len + n);
    // The lag is at least 2 ms, which is 16 samples or more.
    debug_assert!(lag > LTP_ORDER / 2);
    let block = lag.saturating_sub(LTP_ORDER / 2).max(1);
    // One check for the whole subframe.
    #[cfg(target_arch = "x86_64")]
    let sse41 = crate::compat::x86_has_sse41();

    let mut done = 0;
    while done < n {
        let len = block.min(n - done);
        let (known, new_state) = state.split_at_mut(hist_len + done);
        // Sample `i` of the block reads `taps[i..i + 5]`, newest sample last.
        // SAFETY: `known` holds the history, which the caller guarantees to
        // be initialized, and behind it the `done` elements that the blocks
        // before this one have written.
        let taps = unsafe { known[done..done + len + LTP_ORDER - 1].assume_init_ref() };
        let exc = &exc_q14[done..done + len];
        let res = &mut res_q14[done..done + len];
        let new_state = &mut new_state[..len];

        #[cfg(target_arch = "aarch64")]
        ltp_block_neon(taps, b_q14, exc, res, new_state);
        #[cfg(target_arch = "x86_64")]
        if sse41 {
            // SAFETY: the processor has SSE4.1.
            unsafe { x86::ltp_block_sse41(taps, b_q14, exc, res, new_state) };
        } else {
            ltp_block_scalar(taps, b_q14, exc, res, new_state);
        }
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        ltp_block_scalar(taps, b_q14, exc, res, new_state);

        done += len;
    }
}

/// One block of `ltp_synthesis`. `taps` holds four samples more than the
/// other slices: sample `i` is predicted from `taps[i..i + 5]`. Every
/// element of `res_q14` and of `new_state` is written.
#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
#[inline(always)]
fn ltp_block_scalar(
    taps: &[i32],
    b_q14: &[i16; LTP_ORDER],
    exc_q14: &[i32],
    res_q14: &mut [i32],
    new_state: &mut [MaybeUninit<i32>],
) {
    let n = exc_q14.len();
    assert!(taps.len() == n + LTP_ORDER - 1 && res_q14.len() == n && new_state.len() == n);

    for (((x, &e), r), s) in taps
        .windows(LTP_ORDER)
        .zip(exc_q14)
        .zip(res_q14.iter_mut())
        .zip(new_state.iter_mut())
    {
        // The 2 avoids a bias: silk_smlawb always rounds down.
        let mut pred_q13 = 2i32;
        pred_q13 = silk_smlawb(pred_q13, x[4], b_q14[0] as i32);
        pred_q13 = silk_smlawb(pred_q13, x[3], b_q14[1] as i32);
        pred_q13 = silk_smlawb(pred_q13, x[2], b_q14[2] as i32);
        pred_q13 = silk_smlawb(pred_q13, x[1], b_q14[3] as i32);
        pred_q13 = silk_smlawb(pred_q13, x[0], b_q14[4] as i32);

        *r = silk_add_lshift32(e, pred_q13, 1);
        s.write(*r << 1);
    }
}

/// `ltp_block_scalar` with four samples per step.
///
/// The loop keeps its pointer loads and stores, for the reason given at
/// `rewhiten_neon`.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn ltp_block_neon(
    taps: &[i32],
    b_q14: &[i16; LTP_ORDER],
    exc_q14: &[i32],
    res_q14: &mut [i32],
    new_state: &mut [MaybeUninit<i32>],
) {
    use crate::silk::sigproc_fix::smulwb_operand;
    use core::arch::aarch64::*;

    let n = exc_q14.len();
    assert!(taps.len() == n + LTP_ORDER - 1 && res_q14.len() == n && new_state.len() == n);
    let n4 = n & !3;

    // SAFETY: NEON is part of the aarch64 baseline. A step at `i` has
    // `i + 4 <= n`. It reads `taps[i + k..i + k + 4]` for `k <= 4`; the last
    // index is `i + 7 < n + 4`, the length of `taps` (checked above). It
    // reads `exc_q14[i..i + 4]` and writes `res_q14[i..i + 4]` and
    // `new_state[i..i + 4]`, which all have `n` elements.
    unsafe {
        let b0 = vdupq_n_s32(smulwb_operand(b_q14[0]));
        let b1 = vdupq_n_s32(smulwb_operand(b_q14[1]));
        let b2 = vdupq_n_s32(smulwb_operand(b_q14[2]));
        let b3 = vdupq_n_s32(smulwb_operand(b_q14[3]));
        let b4 = vdupq_n_s32(smulwb_operand(b_q14[4]));
        let bias = vdupq_n_s32(2);
        let mut i = 0;
        while i < n4 {
            let x = taps.as_ptr().add(i);
            let mut pred_q13 = vaddq_s32(bias, vqdmulhq_s32(vld1q_s32(x.add(4)), b0));
            pred_q13 = vaddq_s32(pred_q13, vqdmulhq_s32(vld1q_s32(x.add(3)), b1));
            pred_q13 = vaddq_s32(pred_q13, vqdmulhq_s32(vld1q_s32(x.add(2)), b2));
            pred_q13 = vaddq_s32(pred_q13, vqdmulhq_s32(vld1q_s32(x.add(1)), b3));
            pred_q13 = vaddq_s32(pred_q13, vqdmulhq_s32(vld1q_s32(x), b4));

            let exc = vld1q_s32(exc_q14.as_ptr().add(i));
            let res = vaddq_s32(exc, vshlq_n_s32::<1>(pred_q13));
            vst1q_s32(res_q14.as_mut_ptr().add(i), res);
            vst1q_s32(new_state.as_mut_ptr().add(i) as *mut i32, vshlq_n_s32::<1>(res));
            i += 4;
        }
    }
    ltp_block_scalar(
        &taps[n4..],
        b_q14,
        &exc_q14[n4..],
        &mut res_q14[n4..],
        &mut new_state[n4..],
    );
}

/// Short-term (LPC) synthesis of one subframe.
///
/// `s_lpc_q14` holds `MAX_LPC_ORDER` samples of filter state followed by room
/// for one output sample per sample of `res_q14`. `bias` is the rounding
/// offset the prediction starts from.
///
/// Each output sample is the input of the next step, so the time per sample
/// is the time from one output to the next. In libopus (and here before)
/// that path runs through the whole sum of `ORDER` products and two
/// saturating operations. Both kernels shorten it in the same two ways:
///
/// 1. Everything that does not depend on the newest sample is computed one
///    step ahead. What is left on the path is one multiplication and the
///    additions that follow it. The additions wrap, so the order in which
///    the products are added does not change the sum.
/// 2. The two saturating operations are replaced by a test whether they
///    would saturate. Speech never saturates here, so the processor predicts
///    the test and the next step does not wait for it. If the test fires,
///    `lpc_synthesis_saturated` computes the sample the slow way.
#[inline(always)]
fn lpc_synthesis<const ORDER: usize>(
    s_lpc_q14: &mut [i32],
    res_q14: &[i32],
    a_q12: &[i16; MAX_LPC_ORDER],
    bias: i32,
) {
    #[cfg(target_arch = "aarch64")]
    lpc_synthesis_neon::<ORDER>(s_lpc_q14, res_q14, a_q12, bias);
    #[cfg(target_arch = "x86_64")]
    if crate::compat::x86_has_sse41() {
        // SAFETY: the processor has SSE4.1.
        unsafe { x86::lpc_synthesis_sse41::<ORDER>(s_lpc_q14, res_q14, a_q12, bias) };
        return;
    }
    #[cfg(not(target_arch = "aarch64"))]
    lpc_synthesis_scalar::<ORDER>(s_lpc_q14, res_q14, a_q12, bias);
}

/// Number of samples behind the two newest ones that the filter reads.
#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
const OLDER: usize = MAX_LPC_ORDER - 2;

#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
#[inline(always)]
fn lpc_synthesis_scalar<const ORDER: usize>(
    s_lpc_q14: &mut [i32],
    res_q14: &[i32],
    a_q12: &[i16; MAX_LPC_ORDER],
    bias: i32,
) {
    let n = res_q14.len();
    assert!(s_lpc_q14.len() == MAX_LPC_ORDER + n);

    // `bias` plus the products with the samples 3 to ORDER steps back.
    // `w[m]` is the sample `MAX_LPC_ORDER - m` steps back.
    let older_taps = |w: &[i32; OLDER]| -> i32 {
        let mut sum = bias;
        for m in MAX_LPC_ORDER - ORDER..OLDER {
            sum = silk_smlawb(sum, w[m], a_q12[MAX_LPC_ORDER - 1 - m] as i32);
        }
        sum
    };
    let a0 = a_q12[0] as i32;
    let a1 = a_q12[1] as i32;

    // What a step needs from the steps before it: the newest sample, the
    // product with the sample 2 steps back, and the sum of the older ones.
    let mut newest = s_lpc_q14[MAX_LPC_ORDER - 1];
    let mut tap2 = silk_smulwb(s_lpc_q14[MAX_LPC_ORDER - 2], a1);
    let mut older = older_taps(s_lpc_q14[..OLDER].try_into().unwrap());

    for i in 0..n {
        let pred_q10 = older
            .wrapping_add(tap2)
            .wrapping_add(silk_smulwb(newest, a0));

        // silk_add_sat32(res, silk_lshift_sat32(pred, 4)) without saturation.
        let pred_q14 = pred_q10 << 4;
        let (mut sample, overflow) = res_q14[i].overflowing_add(pred_q14);
        if (pred_q14 >> 4) != pred_q10 || overflow {
            sample = lpc_synthesis_saturated(res_q14[i], pred_q10);
        }

        // Prepare the next step. Nothing here depends on `sample`.
        tap2 = silk_smulwb(newest, a1);
        older = older_taps(s_lpc_q14[i + 1..i + 1 + OLDER].try_into().unwrap());

        newest = sample;
        s_lpc_q14[MAX_LPC_ORDER + i] = sample;
    }
}

/// `(x * c) >> 16` for a 16-bit `c` that is passed as `c << 48`: the upper
/// half of a 64 by 64 bit product, which is one instruction on 64-bit ARM.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn smulwb_high(x: i64, c_shl48: i64) -> i64 {
    ((x as i128 * c_shl48 as i128) >> 64) as i64
}

/// The aarch64 kernel keeps the whole filter state in registers.
///
/// The four newest samples are in general registers. The older ones are in
/// vector registers that take in one sample per step; their products are
/// vector multiplications. The path from one output sample to the next is
/// one multiplication (`smulh`) and one addition.
///
/// Sample values are held as 64-bit integers. In 64 bits neither the sum nor
/// the shift can overflow, so the test for saturation is a test whether the
/// results fit: the prediction in 28 bits, the sample in 32 bits.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn lpc_synthesis_neon<const ORDER: usize>(
    s_lpc_q14: &mut [i32],
    res_q14: &[i32],
    a_q12: &[i16; MAX_LPC_ORDER],
    bias: i32,
) {
    use crate::silk::sigproc_fix::smulwb_operand;
    use crate::simd_mem::{array_at, neon::load_i32x4};
    use core::arch::aarch64::*;

    let n = res_q14.len();
    assert!(s_lpc_q14.len() == MAX_LPC_ORDER + n);
    let (state, out) = s_lpc_q14.split_at_mut(MAX_LPC_ORDER);
    let state: &[i32; MAX_LPC_ORDER] = (&*state).try_into().unwrap();

    // Number of the newest samples that are kept in general registers.
    const NEWEST: usize = 4;
    // Number of samples in the vector registers: 5 to 16 steps back.
    const IN_VECTORS: usize = MAX_LPC_ORDER - NEWEST;

    let c_high: [i64; NEWEST] = core::array::from_fn(|k| (a_q12[k] as i64) << 48);
    // Lane `k` of the vector registers holds the sample `16 - k` steps back,
    // and lane `k` of `c_vec` the `smulwb_operand` of its coefficient. Order
    // 10 has no taps 11 to 16; their coefficients stay zero, and the first
    // vector is not used at all.
    let mut c_vec = [0i32; IN_VECTORS];
    for k in MAX_LPC_ORDER - ORDER..IN_VECTORS {
        c_vec[k] = smulwb_operand(a_q12[MAX_LPC_ORDER - 1 - k]);
    }
    // SAFETY: NEON is part of the aarch64 baseline.
    let (c0, c1, c2, mut h0, mut h1, mut h2) = unsafe {
        (
            load_i32x4(array_at(&c_vec, 0)),
            load_i32x4(array_at(&c_vec, 4)),
            load_i32x4(array_at(&c_vec, 8)),
            load_i32x4(array_at(state, 0)),
            load_i32x4(array_at(state, 4)),
            load_i32x4(array_at(state, 8)),
        )
    };

    // q[k] is the sample k + 1 steps back.
    let mut q: [i64; NEWEST] = core::array::from_fn(|k| state[MAX_LPC_ORDER - 1 - k] as i64);
    // `bias` and all products but the one with the newest sample, for the
    // first output sample.
    let mut partial = bias;
    for k in 1..ORDER {
        partial = silk_smlawb(partial, state[MAX_LPC_ORDER - 1 - k], a_q12[k] as i32);
    }
    // The output sample without the product with the newest sample.
    let mut base = match res_q14.first() {
        Some(&res) => res as i64 + ((partial as i64) << 4),
        None => 0,
    };

    for i in 0..n {
        let tap1 = smulwb_high(q[0], c_high[0]);
        let mut sample = base + (tap1 << 4);
        let pred_q10 = partial as i64 + tap1;
        if ((pred_q10 + (1 << 27)) as u64) >> 28 != 0 || sample != sample as i32 as i64 {
            sample = lpc_synthesis_saturated(res_q14[i], pred_q10 as i32) as i64;
        }

        // Prepare the next step. Nothing here depends on `sample`.
        // SAFETY: NEON is part of the aarch64 baseline; no memory is accessed.
        let vec_sum = unsafe {
            // The oldest sample leaves, `q[NEWEST - 1]` enters.
            if ORDER == MAX_LPC_ORDER {
                h0 = vextq_s32::<1>(h0, h1);
            }
            h1 = vextq_s32::<1>(h1, h2);
            h2 = vextq_s32::<1>(h2, vdupq_n_s32(q[NEWEST - 1] as i32));
            let mut products = vaddq_s32(vqdmulhq_s32(h2, c2), vqdmulhq_s32(h1, c1));
            if ORDER == MAX_LPC_ORDER {
                products = vaddq_s32(products, vqdmulhq_s32(h0, c0));
            }
            vaddvq_s32(products)
        };
        let mut sum = bias as i64 + vec_sum as i64;
        for k in 1..NEWEST {
            sum += smulwb_high(q[k - 1], c_high[k]);
        }
        partial = sum as i32;
        let res_next = if i + 1 < n { res_q14[i + 1] } else { 0 };
        base = res_next as i64 + ((partial as i64) << 4);

        q = [sample, q[0], q[1], q[2]];
        out[i] = sample as i32;
    }
}

/// The x86-64 kernels of the decoder synthesis, for processors with
/// SSE4.1. Each computes what the NEON kernel of the same name computes,
/// with the same integer arithmetic, so the results are the bits of the
/// portable code.
///
/// `silk_smulwb(x, c)`, `(x * c) >> 16` for a 16-bit `c`, is the upper half
/// of the 64-bit product of `x` and `c << 16`. `_mm_mul_epi32` gives that
/// product for lanes 0 and 2, so the results land in lanes 1 and 3.
///
/// `rewhiten_sse41` and `ltp_block_sse41` keep their loops (`while m < n4`)
/// and take the ranges they read and write with `get_unchecked` and
/// `get_unchecked_mut`. They are inlined into the SILK decoder. With the
/// ranges taken from `as_chunks` and windows, the loops counted
/// differently, the compiler assigned the registers of the decoder
/// differently, and the loops of the range decoder there got longer.
#[cfg(target_arch = "x86_64")]
mod x86 {
    use super::*;
    use crate::simd_mem::{
        array_at,
        x86::{load_i16x4, load_i16x8, load_i32x4, store_i32x4, store_i32x4_uninit},
    };
    use core::arch::x86_64::*;

    /// The sums of `silk_smulwb(x, c)` over several pairs, as two vectors:
    /// `even` collects the products of lanes 0 and 2 of each `x`, `odd`
    /// those of lanes 1 and 3. The results are in lanes 1 and 3 of both;
    /// the 32-bit additions of the other lanes are not used.
    #[derive(Clone, Copy)]
    struct Sums {
        even: __m128i,
        odd: __m128i,
    }

    /// Lanes 1 and 3 of `x` in lanes 0 and 2, where `_mm_mul_epi32` reads
    /// them (and also in lanes 1 and 3).
    ///
    /// A shuffle, not `_mm_srli_epi64::<32>`: after the shift LLVM sees a
    /// 64-bit arithmetic shift in the operands of the multiplication and
    /// emits an emulated 64-bit multiplication, three `pmuludq` with
    /// shifts, instead of one `pmuldq`.
    #[target_feature(enable = "sse4.1")]
    #[inline]
    fn odd_lanes(x: __m128i) -> __m128i {
        _mm_shuffle_epi32::<0b11_11_01_01>(x)
    }

    impl Sums {
        #[target_feature(enable = "sse4.1")]
        #[inline]
        fn new() -> Self {
            Sums { even: _mm_setzero_si128(), odd: _mm_setzero_si128() }
        }

        /// Adds `silk_smulwb(x[l], c[l])` for each lane `l`, with
        /// `c[l] << 16` in lanes 0 and 2 of `f_even` for lanes 0 and 2, and
        /// in lanes 0 and 2 of `f_odd` for lanes 1 and 3 (`odd_lanes`).
        #[target_feature(enable = "sse4.1")]
        #[inline]
        fn add(&mut self, x: __m128i, f_even: __m128i, f_odd: __m128i) {
            self.even = _mm_add_epi32(self.even, _mm_mul_epi32(x, f_even));
            self.odd = _mm_add_epi32(self.odd, _mm_mul_epi32(odd_lanes(x), f_odd));
        }

        /// The four sums, one per lane.
        #[target_feature(enable = "sse4.1")]
        #[inline]
        fn lanes(self) -> __m128i {
            _mm_blend_epi16::<0b1100_1100>(_mm_srli_epi64::<32>(self.even), self.odd)
        }

        /// The sum of all four lanes.
        #[target_feature(enable = "sse4.1")]
        #[inline]
        fn total(self) -> i32 {
            let both = _mm_add_epi32(self.even, self.odd);
            _mm_extract_epi32::<1>(_mm_add_epi32(both, _mm_unpackhi_epi64(both, both)))
        }
    }

    /// `rewhiten_scalar` with four outputs per step, as `rewhiten_neon`.
    ///
    /// The prediction of one output is a sum of `ORDER` products of 16-bit
    /// samples and coefficients: two `_mm_madd_epi16` of 8 products each,
    /// the second one at sample `ORDER - 8` with the taps that the first
    /// one covered set to zero. The products and their sums wrap at 32 bits
    /// as in the scalar code, so their order does not matter.
    ///
    /// Panics when `input` does not hold `ORDER` samples more than `out`.
    #[target_feature(enable = "sse4.1")]
    pub(super) fn rewhiten_sse41<const ORDER: usize>(
        out: &mut [MaybeUninit<i32>],
        input: &[i16],
        a_q12: &[i16; MAX_LPC_ORDER],
        inv_gain_q31: i32,
    ) {
        const { assert!(ORDER >= 8 && ORDER <= MAX_LPC_ORDER) };
        let n = out.len();
        assert!(input.len() == n + ORDER);
        let n4 = n & !3;

        // Coefficients in the order of the input samples: oldest first.
        let mut a_rev = [0i16; 16];
        for j in 0..ORDER {
            a_rev[j] = a_q12[ORDER - 1 - j];
        }
        let mut second = [0i16; 8];
        for k in 8 - (ORDER - 8)..8 {
            second[k] = a_rev[ORDER - 8 + k];
        }

        // The step at `m` writes `out[m..m + 4]` and reads
        // `input[m..m + ORDER + 4]`: for output `m + l` the samples from
        // `m + l` to `m + l + 8`, from `m + l + ORDER - 8` to
        // `m + l + ORDER` and at `m + l + ORDER`.
        let taps0 = load_i16x8(array_at(&a_rev, 0));
        let taps1 = load_i16x8(&second);
        let gain = _mm_set1_epi32(inv_gain_q31);
        let one = _mm_set1_epi32(1);
        let mut m = 0;
        while m < n4 {
            // SAFETY: `m + 4 <= n`, so the range ends at most at
            // `n + ORDER`, the length of `input` (checked above).
            let x = unsafe { input.get_unchecked(m..m + ORDER + 4) };
            // The prediction of output `m + l`, in four lanes.
            let pred = |l: usize| {
                let first = _mm_madd_epi16(load_i16x8(array_at(x, l)), taps0);
                let second = _mm_madd_epi16(load_i16x8(array_at(x, l + ORDER - 8)), taps1);
                _mm_add_epi32(first, second)
            };
            let (p0, p1, p2, p3) = (pred(0), pred(1), pred(2), pred(3));
            let t01 = _mm_add_epi32(_mm_unpacklo_epi32(p0, p1), _mm_unpackhi_epi32(p0, p1));
            let t23 = _mm_add_epi32(_mm_unpacklo_epi32(p2, p3), _mm_unpackhi_epi32(p2, p3));
            let pred_q12 = _mm_add_epi32(_mm_unpacklo_epi64(t01, t23), _mm_unpackhi_epi64(t01, t23));

            let newest = _mm_cvtepi16_epi32(load_i16x4(array_at(x, ORDER)));
            let res_q12 = _mm_sub_epi32(_mm_slli_epi32::<12>(newest), pred_q12);
            // Round to Q0, ((x >> 11) + 1) >> 1 as silk_rshift_round,
            // and saturate to 16 bits.
            let rounded = _mm_srai_epi32::<1>(_mm_add_epi32(_mm_srai_epi32::<11>(res_q12), one));
            let res = _mm_packs_epi32(rounded, rounded);
            // silk_smulwb(gain, res) per lane: the factor `res << 16` is
            // `res` in the upper half of each 32-bit lane.
            let factor = _mm_unpacklo_epi16(_mm_setzero_si128(), res);
            let mut sums = Sums::new();
            sums.add(gain, factor, odd_lanes(factor));
            // SAFETY: `m + 4 <= n`, the length of `out`.
            let o = unsafe { out.get_unchecked_mut(m..m + 4) };
            store_i32x4_uninit(o.try_into().unwrap(), sums.lanes());
            m += 4;
        }
        rewhiten_scalar::<ORDER>(&mut out[n4..], &input[n4..], a_q12, inv_gain_q31);
    }

    /// `ltp_block_scalar` with four samples per step, as `ltp_block_neon`.
    ///
    /// Panics when `taps` does not hold four samples more than `exc_q14`,
    /// or `res_q14` or `new_state` not as many.
    #[target_feature(enable = "sse4.1")]
    pub(super) fn ltp_block_sse41(
        taps: &[i32],
        b_q14: &[i16; LTP_ORDER],
        exc_q14: &[i32],
        res_q14: &mut [i32],
        new_state: &mut [MaybeUninit<i32>],
    ) {
        let n = exc_q14.len();
        assert!(taps.len() == n + LTP_ORDER - 1 && res_q14.len() == n && new_state.len() == n);
        let n4 = n & !3;

        // The step at `i` reads `taps[i..i + 8]` and `exc_q14[i..i + 4]`,
        // and writes `res_q14` and `new_state` there.
        //
        // `b_q14[k] << 16` in every lane: the factor of the sample
        // `4 - k` places into the five.
        let f = |k: usize| _mm_set1_epi32((b_q14[k] as i32) << 16);
        let f = [f(0), f(1), f(2), f(3), f(4)];
        // The 2 avoids a bias: silk_smlawb always rounds down.
        let bias = _mm_set1_epi32(2);
        let mut i = 0;
        while i < n4 {
            // SAFETY: `i + 4 <= n`; `taps` has `n + 4` elements, the other
            // slices `n` (checked above).
            let (x, exc, res_out, state_out) = unsafe {
                (
                    taps.get_unchecked(i..i + 8),
                    exc_q14.get_unchecked(i..i + 4),
                    res_q14.get_unchecked_mut(i..i + 4),
                    new_state.get_unchecked_mut(i..i + 4),
                )
            };
            let mut sums = Sums::new();
            for (k, &f) in f.iter().enumerate() {
                sums.add(load_i32x4(array_at(x, LTP_ORDER - 1 - k)), f, f);
            }
            let pred_q13 = _mm_add_epi32(bias, sums.lanes());

            let exc = load_i32x4(array_at(exc, 0));
            let res = _mm_add_epi32(exc, _mm_slli_epi32::<1>(pred_q13));
            store_i32x4(res_out.try_into().unwrap(), res);
            store_i32x4_uninit(state_out.try_into().unwrap(), _mm_slli_epi32::<1>(res));
            i += 4;
        }
        ltp_block_scalar(
            &taps[n4..],
            b_q14,
            &exc_q14[n4..],
            &mut res_q14[n4..],
            &mut new_state[n4..],
        );
    }

    /// `lpc_synthesis` for x86-64 with SSE4.1, in the way of
    /// `lpc_synthesis_neon`: the whole filter state in registers.
    ///
    /// The four newest samples are in general registers, as 64-bit
    /// integers. The older ones are in vector registers that take in one
    /// sample per step; their products are vector multiplications that do
    /// not wait for the newest sample. The path from one output sample to
    /// the next is one multiplication, one shift and one addition.
    ///
    /// In 64 bits neither the sum nor the shift can overflow, so the test
    /// for saturation is a test whether the results fit: the prediction in
    /// 28 bits, the sample in 32 bits. Where they do not, the sample is
    /// computed the slow way from the prediction wrapped to 32 bits, as the
    /// scalar code does; that gives the scalar code's result also when it
    /// would not have saturated.
    ///
    /// Panics when `s_lpc_q14` does not hold `MAX_LPC_ORDER` samples more
    /// than `res_q14`.
    #[target_feature(enable = "sse4.1")]
    pub(super) fn lpc_synthesis_sse41<const ORDER: usize>(
        s_lpc_q14: &mut [i32],
        res_q14: &[i32],
        a_q12: &[i16; MAX_LPC_ORDER],
        bias: i32,
    ) {
        // The first vector register (taps 13 to 16) is used only for the
        // full order. Below it, the order must leave those taps out.
        const { assert!(ORDER > 4 && (ORDER <= MAX_LPC_ORDER - 4 || ORDER == MAX_LPC_ORDER)) };
        let n = res_q14.len();
        assert!(s_lpc_q14.len() == MAX_LPC_ORDER + n);
        let (state, out) = s_lpc_q14.split_at_mut(MAX_LPC_ORDER);
        let state: &[i32; MAX_LPC_ORDER] = (&*state).try_into().unwrap();

        // Number of the newest samples that are kept in general registers.
        const NEWEST: usize = 4;
        // Number of samples in the vector registers: 5 to 16 steps back.
        const IN_VECTORS: usize = MAX_LPC_ORDER - NEWEST;

        let c_newest: [i64; NEWEST] = core::array::from_fn(|k| a_q12[k] as i64);
        // Lane `k` of the vector registers holds the sample `16 - k` steps
        // back, and lane `k` of `c_vec` its coefficient. Order 10 has no
        // taps 11 to 16; their coefficients stay zero, and the first vector
        // is not used at all.
        let mut c_vec = [0i32; IN_VECTORS];
        for k in MAX_LPC_ORDER - ORDER..IN_VECTORS {
            c_vec[k] = (a_q12[MAX_LPC_ORDER - 1 - k] as i32) << 16;
        }
        let (f, mut h0, mut h1, mut h2) = {
            let c0 = load_i32x4(array_at(&c_vec, 0));
            let c1 = load_i32x4(array_at(&c_vec, 4));
            let c2 = load_i32x4(array_at(&c_vec, 8));
            (
                [(c0, odd_lanes(c0)), (c1, odd_lanes(c1)), (c2, odd_lanes(c2))],
                load_i32x4(array_at(state, 0)),
                load_i32x4(array_at(state, 4)),
                load_i32x4(array_at(state, 8)),
            )
        };

        // q[k] is the sample k + 1 steps back.
        let mut q: [i64; NEWEST] = core::array::from_fn(|k| state[MAX_LPC_ORDER - 1 - k] as i64);
        // `bias` and all products but the one with the newest sample, for
        // the first output sample.
        let mut partial = bias;
        for k in 1..ORDER {
            partial = silk_smlawb(partial, state[MAX_LPC_ORDER - 1 - k], a_q12[k] as i32);
        }
        // The output sample without the product with the newest sample.
        let mut base = match res_q14.first() {
            Some(&res) => res as i64 + ((partial as i64) << 4),
            None => 0,
        };

        for i in 0..n {
            let tap1 = (q[0] * c_newest[0]) >> 16;
            let mut sample = base + (tap1 << 4);
            let pred_q10 = partial as i64 + tap1;
            if ((pred_q10 + (1 << 27)) as u64) >> 28 != 0 || sample != sample as i32 as i64 {
                sample = lpc_synthesis_saturated(res_q14[i], pred_q10 as i32) as i64;
            }

            // Prepare the next step. Nothing here depends on `sample`.
            // The oldest sample leaves, `q[NEWEST - 1]` enters.
            if ORDER == MAX_LPC_ORDER {
                h0 = _mm_alignr_epi8::<4>(h1, h0);
            }
            h1 = _mm_alignr_epi8::<4>(h2, h1);
            h2 = _mm_alignr_epi8::<4>(_mm_cvtsi32_si128(q[NEWEST - 1] as i32), h2);
            let mut sums = Sums::new();
            if ORDER == MAX_LPC_ORDER {
                sums.add(h0, f[0].0, f[0].1);
            }
            sums.add(h1, f[1].0, f[1].1);
            sums.add(h2, f[2].0, f[2].1);
            let mut sum = bias as i64 + sums.total() as i64;
            for k in 1..NEWEST {
                sum += (q[k - 1] * c_newest[k]) >> 16;
            }
            partial = sum as i32;
            let res_next = if i + 1 < n { res_q14[i + 1] } else { 0 };
            base = res_next as i64 + ((partial as i64) << 4);

            q = [sample, q[0], q[1], q[2]];
            out[i] = sample as i32;
        }
    }
}

/// The saturating form of the last step of `lpc_synthesis`. It is kept out
/// of line so that the compiler leaves the test in front of it as a branch.
#[cold]
#[inline(never)]
fn lpc_synthesis_saturated(res_q14: i32, pred_q10: i32) -> i32 {
    silk_add_sat32(res_q14, silk_lshift_sat32(pred_q10, 4))
}

/// Output samples from the synthesis filter state: multiply by the gain and
/// round to 16 bits.
#[inline(always)]
fn scale_output(xq: &mut [i16], s_q14: &[i32], gain_q10: i32) {
    for (out, &s) in xq.iter_mut().zip(s_q14) {
        *out = silk_sat16(silk_rshift_round(silk_smulww(s, gain_q10), 8)) as i16;
    }
}

#[cfg(test)]
mod tests {
    #[cfg(not(feature = "std"))]
    use std::vec::Vec;
    use super::*;
    use crate::silk::init_decoder::silk_decoder_set_fs;

    /// Small deterministic pseudo-random generator.
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 32) as u32
        }
        fn range(&mut self, lo: i32, hi: i32) -> i32 {
            lo + (self.next() as u64 % (hi as i64 - lo as i64 + 1) as u64) as i32
        }
        /// A value that is often at or near a limit of `i16`.
        fn i16_edgy(&mut self) -> i16 {
            match self.next() % 8 {
                0 => i16::MAX,
                1 => i16::MIN,
                2 => 0,
                _ => self.next() as i16,
            }
        }
        /// A value that is often at or near a limit of `i32`.
        fn i32_edgy(&mut self) -> i32 {
            match self.next() % 8 {
                0 => i32::MAX,
                1 => i32::MIN,
                2 => 0,
                _ => self.next() as i32,
            }
        }
    }

    /// The view of an output slice that the kernels write to.
    fn uninit(out: &mut [i32]) -> &mut [MaybeUninit<i32>] {
        // SAFETY: `MaybeUninit<i32>` has the layout of `i32`, and the
        // kernels only store initialized values.
        unsafe { &mut *(out as *mut [i32] as *mut [MaybeUninit<i32>]) }
    }

    /// The lengths every kernel test runs: empty, one sample, every tail
    /// length of the 4- and 8-sample steps, and the subframe lengths.
    const LENGTHS: [usize; 24] = [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 31, 33, 40, 60, 79, 80, 81,
    ];

    // -- excitation --------------------------------------------------------

    /// The loop of libopus `silk_decode_core`, one sample at a time.
    fn excitation_reference(exc_q14: &mut [i32], pulses: &[i16], seed: i32, offset_q10: i32) {
        let mut rand_seed = seed;
        for i in 0..pulses.len() {
            rand_seed = silk_rand(rand_seed);
            exc_q14[i] = (pulses[i] as i32) << 14;
            if exc_q14[i] > 0 {
                exc_q14[i] -= QUANT_LEVEL_ADJUST_Q10 << 4;
            } else if exc_q14[i] < 0 {
                exc_q14[i] += QUANT_LEVEL_ADJUST_Q10 << 4;
            }
            exc_q14[i] += offset_q10 << 4;
            if rand_seed < 0 {
                exc_q14[i] = -exc_q14[i];
            }
            rand_seed = silk_add32_ovflw(rand_seed, pulses[i] as i32);
        }
    }

    #[test]
    fn excitation_matches_the_sample_by_sample_loop() {
        let mut rng = Lcg(11);
        let mut lengths = LENGTHS.to_vec();
        lengths.extend([120, 160, 240, 320]);
        for &n in &lengths {
            for trial in 0..40 {
                let pulses: Vec<i16> = (0..n)
                    .map(|_| match trial % 4 {
                        0 => rng.range(-3, 3) as i16,
                        1 => rng.range(-40, 40) as i16,
                        2 => rng.i16_edgy(),
                        _ => 0,
                    })
                    .collect();
                let seed = if trial % 2 == 0 { rng.range(0, 3) } else { rng.i32_edgy() };
                let offset_q10 = [25, 60, 8, 25, 0, -100, 4000][trial % 7];
                let mut expect = vec![0i32; n];
                excitation_reference(&mut expect, &pulses, seed, offset_q10);
                let mut got = vec![0x5555_5555i32; n];
                decode_excitation(&mut got, &pulses, seed, offset_q10);
                assert_eq!(got, expect, "n={n} trial={trial} seed={seed}");
            }
        }
    }

    // -- re-whitening --------------------------------------------------------

    /// libopus `silk_LPC_analysis_filter` followed by the scaling loop of
    /// `silk_decode_core`, as this file had them before.
    fn rewhiten_reference(
        out: &mut [i32],
        input: &[i16],
        a_q12: &[i16; MAX_LPC_ORDER],
        inv_gain_q31: i32,
        order: usize,
    ) {
        for ix in order..input.len() {
            let mut out32_q12: i32 = 0;
            for j in 0..order {
                out32_q12 = out32_q12
                    .wrapping_add(silk_smulbb(input[ix - j - 1] as i32, a_q12[j] as i32));
            }
            out32_q12 = ((input[ix] as i32) << 12).wrapping_sub(out32_q12);
            let s = silk_sat16(silk_rshift_round(out32_q12, 12)) as i16;
            out[ix - order] = silk_smulwb(inv_gain_q31, s as i32);
        }
    }

    fn check_rewhiten<const ORDER: usize>(rng: &mut Lcg) {
        let mut lengths = LENGTHS.to_vec();
        lengths.extend([146, 290, 306]);
        for &n in &lengths {
            for trial in 0..24 {
                let input: Vec<i16> = (0..n + ORDER)
                    .map(|_| if trial % 3 == 0 { rng.i16_edgy() } else { rng.range(-9000, 9000) as i16 })
                    .collect();
                let mut a_q12 = [0i16; MAX_LPC_ORDER];
                for a in a_q12.iter_mut() {
                    *a = if trial % 2 == 0 { rng.i16_edgy() } else { rng.range(-4096, 4096) as i16 };
                }
                let inv_gain_q31 = if trial % 4 == 0 { rng.i32_edgy() } else { rng.range(1, 1 << 30) };

                let mut expect = vec![0i32; n];
                rewhiten_reference(&mut expect, &input, &a_q12, inv_gain_q31, ORDER);
                let mut scalar = vec![0x5555_5555i32; n];
                rewhiten_scalar::<ORDER>(uninit(&mut scalar), &input, &a_q12, inv_gain_q31);
                assert_eq!(scalar, expect, "scalar: order={ORDER} n={n} trial={trial}");
                #[cfg(target_arch = "aarch64")]
                {
                    let mut neon = vec![0x5555_5555i32; n];
                    rewhiten_neon::<ORDER>(uninit(&mut neon), &input, &a_q12, inv_gain_q31);
                    assert_eq!(neon, expect, "NEON: order={ORDER} n={n} trial={trial}");
                }
                #[cfg(target_arch = "x86_64")]
                if crate::compat::x86_has_sse41() {
                    let mut sse = vec![0x5555_5555i32; n];
                    // SAFETY: the processor has SSE4.1.
                    unsafe { x86::rewhiten_sse41::<ORDER>(uninit(&mut sse), &input, &a_q12, inv_gain_q31) };
                    assert_eq!(sse, expect, "SSE4.1: order={ORDER} n={n} trial={trial}");
                }
                let mut dispatched = vec![0x5555_5555i32; n];
                rewhiten::<ORDER>(uninit(&mut dispatched), &input, &a_q12, inv_gain_q31);
                assert_eq!(dispatched, expect, "dispatch: order={ORDER} n={n} trial={trial}");
            }
        }
    }

    #[test]
    fn rewhiten_kernels_match_the_reference() {
        let mut rng = Lcg(12);
        check_rewhiten::<10>(&mut rng);
        check_rewhiten::<16>(&mut rng);
    }

    // -- long-term prediction ------------------------------------------------

    /// The loop of libopus `silk_decode_core`: one sample at a time, reading
    /// and writing the same buffer.
    fn ltp_reference(
        s_ltp_q15: &mut [i32],
        buf_idx: usize,
        lag: usize,
        b_q14: &[i16; LTP_ORDER],
        exc_q14: &[i32],
        res_q14: &mut [i32],
    ) {
        let start = buf_idx - lag + LTP_ORDER / 2;
        for i in 0..exc_q14.len() {
            let mut pred: i32 = 2;
            for t in 0..LTP_ORDER {
                pred = silk_smlawb(pred, s_ltp_q15[start + i - t], b_q14[t] as i32);
            }
            res_q14[i] = silk_add_lshift32(exc_q14[i], pred, 1);
            s_ltp_q15[buf_idx + i] = res_q14[i] << 1;
        }
    }

    #[test]
    fn ltp_block_kernels_match_the_reference() {
        let mut rng = Lcg(13);
        for &n in &LENGTHS {
            for trial in 0..24 {
                let taps: Vec<i32> = (0..n + LTP_ORDER - 1)
                    .map(|_| if trial % 3 == 0 { rng.i32_edgy() } else { rng.range(-(1 << 26), 1 << 26) })
                    .collect();
                let exc: Vec<i32> = (0..n)
                    .map(|_| if trial % 3 == 1 { rng.i32_edgy() } else { rng.range(-(1 << 24), 1 << 24) })
                    .collect();
                let mut b_q14 = [0i16; LTP_ORDER];
                for b in b_q14.iter_mut() {
                    *b = if trial % 2 == 0 { rng.i16_edgy() } else { rng.range(-8000, 8000) as i16 };
                }

                // Reference: the taps as history, a lag that puts every read
                // in front of the first write.
                let lag = n + LTP_ORDER / 2 + 2;
                let buf_idx = lag + LTP_ORDER / 2;
                let mut state = vec![0i32; buf_idx + n];
                state[buf_idx - lag - 2..buf_idx - lag - 2 + taps.len()].copy_from_slice(&taps);
                let mut expect_res = vec![0i32; n];
                ltp_reference(&mut state, buf_idx, lag, &b_q14, &exc, &mut expect_res);
                let expect_state = state[buf_idx..].to_vec();

                let mut res = vec![0x5555_5555i32; n];
                let mut new_state = vec![0x5555_5555i32; n];
                ltp_block_scalar(&taps, &b_q14, &exc, &mut res, uninit(&mut new_state));
                assert_eq!((&res, &new_state), (&expect_res, &expect_state), "scalar: n={n} trial={trial}");
                #[cfg(target_arch = "aarch64")]
                {
                    let mut res = vec![0x5555_5555i32; n];
                    let mut new_state = vec![0x5555_5555i32; n];
                    ltp_block_neon(&taps, &b_q14, &exc, &mut res, uninit(&mut new_state));
                    assert_eq!((&res, &new_state), (&expect_res, &expect_state), "NEON: n={n} trial={trial}");
                }
                #[cfg(target_arch = "x86_64")]
                if crate::compat::x86_has_sse41() {
                    let mut res = vec![0x5555_5555i32; n];
                    let mut new_state = vec![0x5555_5555i32; n];
                    // SAFETY: the processor has SSE4.1.
                    unsafe { x86::ltp_block_sse41(&taps, &b_q14, &exc, &mut res, uninit(&mut new_state)) };
                    assert_eq!((&res, &new_state), (&expect_res, &expect_state), "SSE4.1: n={n} trial={trial}");
                }
            }
        }
    }

    /// `ltp_synthesis` splits a subframe into blocks. Every lag from the
    /// shortest one that works (3) upwards must give the result of the
    /// sample-by-sample loop, also when a sample depends on samples of the
    /// same subframe.
    #[test]
    fn ltp_synthesis_matches_the_reference_for_every_lag() {
        let mut rng = Lcg(14);
        for lag in (3..=40).chain([41, 79, 80, 81, 82, 83, 144, 200, 288]) {
            for &n in &[1usize, 5, 40, 60, 80] {
                let buf_idx = 320usize;
                let mut state: Vec<i32> = (0..buf_idx + n)
                    .map(|_| rng.range(-(1 << 27), 1 << 27))
                    .collect();
                let exc: Vec<i32> = (0..n).map(|_| rng.range(-(1 << 24), 1 << 24)).collect();
                let mut b_q14 = [0i16; LTP_ORDER];
                for b in b_q14.iter_mut() {
                    *b = rng.range(-6000, 6000) as i16;
                }
                let mut expect_state = state.clone();
                let mut expect_res = vec![0i32; n];
                ltp_reference(&mut expect_state, buf_idx, lag, &b_q14, &exc, &mut expect_res);

                let mut res = vec![0x5555_5555i32; n];
                // SAFETY: the whole of `state` is initialized.
                unsafe {
                    let used = uninit(&mut state[buf_idx - lag - LTP_ORDER / 2..]);
                    ltp_synthesis(used, lag, &b_q14, &exc, &mut res);
                }
                assert_eq!(res, expect_res, "lag={lag} n={n}");
                assert_eq!(state, expect_state, "lag={lag} n={n}");
            }
        }
    }

    // -- short-term synthesis ------------------------------------------------

    /// The loop of libopus `silk_decode_core`, with the products added in the
    /// order of the source and both saturating operations.
    fn lpc_reference(
        s_lpc_q14: &mut [i32],
        res_q14: &[i32],
        a_q12: &[i16; MAX_LPC_ORDER],
        bias: i32,
        order: usize,
    ) {
        for i in 0..res_q14.len() {
            let mut pred = bias;
            for j in 0..order {
                pred = silk_smlawb(pred, s_lpc_q14[MAX_LPC_ORDER + i - 1 - j], a_q12[j] as i32);
            }
            s_lpc_q14[MAX_LPC_ORDER + i] =
                silk_add_sat32(res_q14[i], silk_lshift_sat32(pred, 4));
        }
    }

    /// Returns how many output samples sit at a limit of `i32`.
    fn check_lpc_synthesis<const ORDER: usize>(rng: &mut Lcg) -> usize {
        let mut saturated = 0;
        for &n in &LENGTHS {
            for trial in 0..60 {
                // Trials 0..20: a decaying filter and a moderate signal, as
                // in speech. Trials 20..40: large coefficients, so that the
                // output grows until it saturates. Trials 40..60: values at
                // the limits everywhere.
                let kind = trial / 20;
                let mut a_q12 = [0i16; MAX_LPC_ORDER];
                for (j, a) in a_q12.iter_mut().enumerate() {
                    *a = match kind {
                        0 => rng.range(-1500, 1500) as i16 >> (j / 4),
                        1 => rng.range(-20000, 20000) as i16,
                        _ => rng.i16_edgy(),
                    };
                }
                let mut state = vec![0i32; MAX_LPC_ORDER + n];
                for s in state[..MAX_LPC_ORDER].iter_mut() {
                    *s = match kind {
                        0 => rng.range(-(1 << 24), 1 << 24),
                        1 => rng.range(-(1 << 28), 1 << 28),
                        _ => rng.i32_edgy(),
                    };
                }
                let res: Vec<i32> = (0..n)
                    .map(|_| match kind {
                        0 => rng.range(-(1 << 22), 1 << 22),
                        1 => rng.range(-(1 << 27), 1 << 27),
                        _ => rng.i32_edgy(),
                    })
                    .collect();
                let bias = (ORDER / 2) as i32;

                let mut expect = state.clone();
                lpc_reference(&mut expect, &res, &a_q12, bias, ORDER);
                saturated += expect[MAX_LPC_ORDER..]
                    .iter()
                    .filter(|&&s| s == i32::MAX || s == i32::MIN)
                    .count();

                let mut scalar = state.clone();
                lpc_synthesis_scalar::<ORDER>(&mut scalar, &res, &a_q12, bias);
                assert_eq!(scalar, expect, "scalar: order={ORDER} n={n} trial={trial}");
                #[cfg(target_arch = "aarch64")]
                {
                    let mut neon = state.clone();
                    lpc_synthesis_neon::<ORDER>(&mut neon, &res, &a_q12, bias);
                    assert_eq!(neon, expect, "NEON: order={ORDER} n={n} trial={trial}");
                }
                #[cfg(target_arch = "x86_64")]
                if crate::compat::x86_has_sse41() {
                    let mut sse = state.clone();
                    // SAFETY: the processor has SSE4.1.
                    unsafe { x86::lpc_synthesis_sse41::<ORDER>(&mut sse, &res, &a_q12, bias) };
                    assert_eq!(sse, expect, "SSE4.1: order={ORDER} n={n} trial={trial}");
                }
                let mut dispatched = state.clone();
                lpc_synthesis::<ORDER>(&mut dispatched, &res, &a_q12, bias);
                assert_eq!(dispatched, expect, "dispatch: order={ORDER} n={n} trial={trial}");
            }
        }
        saturated
    }

    #[test]
    fn lpc_synthesis_kernels_match_the_reference() {
        let mut rng = Lcg(15);
        let saturated = check_lpc_synthesis::<10>(&mut rng) + check_lpc_synthesis::<16>(&mut rng);
        // The saturating path must have been exercised.
        assert!(saturated > 1000, "only {saturated} saturated samples");
    }

    /// Predictions just inside and just outside the range that a shift by 4
    /// keeps in 32 bits, and sums at the limits of 32 bits.
    #[test]
    fn lpc_synthesis_at_the_saturation_limits() {
        let limit = i32::MAX >> 4;
        for order in [10usize, 16] {
            for pred_target in [limit - 1, limit, limit + 1, -limit - 2, -limit - 1, -limit] {
                for res in [0, 15, 16, -1, -16, -17, i32::MAX, i32::MIN, 1 << 30, -(1 << 30)] {
                    // One coefficient of 1.0 in Q16 terms is not available
                    // (the coefficients are Q12 in 16 bits), so the newest
                    // state sample is chosen such that a[0] * s >> 16 gives
                    // the wanted prediction.
                    let mut a_q12 = [0i16; MAX_LPC_ORDER];
                    a_q12[0] = 16384;
                    let bias = (order / 2) as i32;
                    let mut state = vec![0i32; MAX_LPC_ORDER + 3];
                    state[MAX_LPC_ORDER - 1] = (pred_target - bias).wrapping_mul(4);
                    let res_q14 = [res, 0, 0];

                    let mut expect = state.clone();
                    lpc_reference(&mut expect, &res_q14, &a_q12, bias, order);
                    let mut scalar = state.clone();
                    let mut dispatched = state.clone();
                    if order == 16 {
                        lpc_synthesis_scalar::<16>(&mut scalar, &res_q14, &a_q12, bias);
                        lpc_synthesis::<16>(&mut dispatched, &res_q14, &a_q12, bias);
                    } else {
                        lpc_synthesis_scalar::<10>(&mut scalar, &res_q14, &a_q12, bias);
                        lpc_synthesis::<10>(&mut dispatched, &res_q14, &a_q12, bias);
                    }
                    assert_eq!(scalar, expect, "scalar: order={order} pred={pred_target} res={res}");
                    assert_eq!(dispatched, expect, "dispatch: order={order} pred={pred_target} res={res}");
                }
            }
        }
    }

    // -- the whole function ----------------------------------------------------

    /// A direct port of libopus `silk_decode_core`, the reference for the
    /// kernels above.
    fn decode_core_reference(
        ps_dec: &mut SilkDecoderState,
        ps_dec_ctrl: &mut SilkDecoderControl,
        xq: &mut [i16],
        pulses: &[i16],
    ) {
        let nlsf_interpolation_flag = ps_dec.indices.nlsf_interp_coef_q2 < 4;
        let offset_q10 = SILK_QUANTIZATION_OFFSETS_Q10[(ps_dec.indices.signal_type >> 1) as usize]
            [ps_dec.indices.quant_offset_type as usize] as i32;
        let frame_length = ps_dec.frame_length as usize;
        let subfr_length = ps_dec.subfr_length as usize;
        let ltp_mem_length = ps_dec.ltp_mem_length as usize;
        let lpc_order = ps_dec.lpc_order as usize;

        excitation_reference(
            &mut ps_dec.exc_q14[..frame_length],
            &pulses[..frame_length],
            ps_dec.indices.seed as i32,
            offset_q10,
        );

        let mut s_lpc_q14 = [0i32; MAX_SUB_FRAME_LENGTH + MAX_LPC_ORDER];
        s_lpc_q14[..MAX_LPC_ORDER].copy_from_slice(&ps_dec.s_lpc_q14_buf);
        let mut pexc = 0usize;
        let mut s_ltp_buf_idx = ltp_mem_length;
        let mut s_ltp_q15 = vec![0i32; ltp_mem_length + frame_length];
        let mut s_ltp = vec![0i16; ltp_mem_length];

        for k in 0..ps_dec.nb_subfr as usize {
            let a_q12: [i16; MAX_LPC_ORDER] = ps_dec_ctrl.pred_coef_q12[k >> 1];
            let mut inv_gain_q31 = silk_inverse32_varq(ps_dec_ctrl.gains_q16[k], 47);
            let gain_adj_q16 = if ps_dec_ctrl.gains_q16[k] != ps_dec.prev_gain_q16 {
                let adj = silk_div32_varq(ps_dec.prev_gain_q16, ps_dec_ctrl.gains_q16[k], 16);
                for s in s_lpc_q14[..MAX_LPC_ORDER].iter_mut() {
                    *s = silk_smulww(adj, *s);
                }
                adj
            } else {
                1 << 16
            };
            ps_dec.prev_gain_q16 = ps_dec_ctrl.gains_q16[k];

            let mut eff_signal_type = ps_dec.indices.signal_type as i32;
            if ps_dec.loss_cnt > 0
                && ps_dec.prev_signal_type == TYPE_VOICED
                && ps_dec.indices.signal_type as i32 != TYPE_VOICED
                && k < MAX_NB_SUBFR / 2
            {
                ps_dec_ctrl.ltp_coef_q14[k * LTP_ORDER..(k + 1) * LTP_ORDER]
                    .copy_from_slice(&[0, 0, 1 << 12, 0, 0]);
                ps_dec_ctrl.pitch_l[k] = ps_dec.lag_prev;
                eff_signal_type = TYPE_VOICED;
            }
            let mut b_q14 = [0i16; LTP_ORDER];
            b_q14.copy_from_slice(&ps_dec_ctrl.ltp_coef_q14[k * LTP_ORDER..(k + 1) * LTP_ORDER]);

            let mut res_q14 = [0i32; MAX_SUB_FRAME_LENGTH];
            if eff_signal_type == TYPE_VOICED {
                let lag = ps_dec_ctrl.pitch_l[k].clamp(2 * ps_dec.fs_khz, 18 * ps_dec.fs_khz);
                if k == 0 || (k == 2 && nlsf_interpolation_flag) {
                    let start_idx =
                        (ps_dec.ltp_mem_length - lag - ps_dec.lpc_order - (LTP_ORDER / 2) as i32)
                            .max(0) as usize;
                    if k == 2 {
                        ps_dec.out_buf[ltp_mem_length..ltp_mem_length + 2 * subfr_length]
                            .copy_from_slice(&xq[..2 * subfr_length]);
                    }
                    // silk_LPC_analysis_filter: the first `lpc_order`
                    // outputs are zero.
                    let input = &ps_dec.out_buf[start_idx + k * subfr_length..];
                    let len = ltp_mem_length - start_idx;
                    for ix in 0..lpc_order {
                        s_ltp[start_idx + ix] = 0;
                    }
                    for ix in lpc_order..len {
                        let mut out32_q12: i32 = 0;
                        for j in 0..lpc_order {
                            out32_q12 = out32_q12.wrapping_add(silk_smulbb(
                                input[ix - j - 1] as i32,
                                a_q12[j] as i32,
                            ));
                        }
                        out32_q12 = ((input[ix] as i32) << 12).wrapping_sub(out32_q12);
                        s_ltp[start_idx + ix] =
                            silk_sat16(silk_rshift_round(out32_q12, 12)) as i16;
                    }
                    if k == 0 {
                        inv_gain_q31 =
                            silk_lshift(silk_smulwb(inv_gain_q31, ps_dec_ctrl.ltp_scale_q14), 2);
                    }
                    for i in 0..(lag as usize + LTP_ORDER / 2) {
                        s_ltp_q15[s_ltp_buf_idx - i - 1] =
                            silk_smulwb(inv_gain_q31, s_ltp[ltp_mem_length - i - 1] as i32);
                    }
                } else if gain_adj_q16 != (1 << 16) {
                    for i in 0..(lag as usize + LTP_ORDER / 2) {
                        s_ltp_q15[s_ltp_buf_idx - i - 1] =
                            silk_smulww(gain_adj_q16, s_ltp_q15[s_ltp_buf_idx - i - 1]);
                    }
                }
                ltp_reference(
                    &mut s_ltp_q15,
                    s_ltp_buf_idx,
                    lag as usize,
                    &b_q14,
                    &ps_dec.exc_q14[pexc..pexc + subfr_length],
                    &mut res_q14[..subfr_length],
                );
                s_ltp_buf_idx += subfr_length;
            } else {
                res_q14[..subfr_length]
                    .copy_from_slice(&ps_dec.exc_q14[pexc..pexc + subfr_length]);
            }

            lpc_reference(
                &mut s_lpc_q14[..MAX_LPC_ORDER + subfr_length],
                &res_q14[..subfr_length],
                &a_q12,
                ps_dec.lpc_order >> 1,
                if lpc_order == 16 { 16 } else { 10 },
            );
            let gain_q10 = ps_dec_ctrl.gains_q16[k] >> 6;
            for i in 0..subfr_length {
                let product = silk_smulww(s_lpc_q14[MAX_LPC_ORDER + i], gain_q10);
                xq[pexc + i] = silk_sat16(silk_rshift_round(product, 8)) as i16;
            }
            for i in 0..MAX_LPC_ORDER {
                s_lpc_q14[i] = s_lpc_q14[subfr_length + i];
            }
            pexc += subfr_length;
        }
        ps_dec
            .s_lpc_q14_buf
            .copy_from_slice(&s_lpc_q14[..MAX_LPC_ORDER]);
    }

    /// A decoder state and frame parameters as a decoded frame could have
    /// them, and beyond: `wild` trials use pitch lags outside the valid
    /// range, lags that jump between subframes, and filter coefficients
    /// that make the output saturate.
    fn random_frame(
        rng: &mut Lcg,
        fs_khz: i32,
        nb_subfr: i32,
        wild: bool,
    ) -> (SilkDecoderState, SilkDecoderControl, [i16; MAX_FRAME_LENGTH]) {
        let mut dec = SilkDecoderState { nb_subfr, ..Default::default() };
        assert_eq!(silk_decoder_set_fs(&mut dec, fs_khz, 48000), 0);

        for v in dec.out_buf.iter_mut() {
            *v = if wild { rng.i16_edgy() } else { rng.range(-12000, 12000) as i16 };
        }
        for v in dec.s_lpc_q14_buf.iter_mut() {
            *v = if wild { rng.range(-(1 << 29), 1 << 29) } else { rng.range(-(1 << 24), 1 << 24) };
        }
        dec.prev_gain_q16 = rng.range(1 << 16, 1 << 28);
        dec.lag_prev = if wild { rng.range(0, 400) } else { rng.range(2 * fs_khz, 18 * fs_khz) };
        dec.loss_cnt = if rng.next().is_multiple_of(3) { 1 } else { 0 };
        dec.prev_signal_type = rng.range(0, 2);
        dec.indices.signal_type = rng.range(0, 2) as i8;
        dec.indices.quant_offset_type = rng.range(0, 1) as i8;
        dec.indices.seed = rng.range(0, 3) as i8;
        dec.indices.nlsf_interp_coef_q2 = rng.range(0, 4) as i8;

        let mut ctrl = SilkDecoderControl::new();
        let base_lag = rng.range(2 * fs_khz, 18 * fs_khz);
        for k in 0..MAX_NB_SUBFR {
            ctrl.pitch_l[k] = if wild {
                rng.range(-20, 400)
            } else {
                (base_lag + rng.range(-9, 9)).clamp(2 * fs_khz, 18 * fs_khz)
            };
            ctrl.gains_q16[k] = if rng.next().is_multiple_of(4) {
                dec.prev_gain_q16
            } else {
                rng.range(1 << 16, 1 << 28)
            };
        }
        for coefs in ctrl.pred_coef_q12.iter_mut() {
            for (j, a) in coefs.iter_mut().enumerate() {
                *a = if wild {
                    rng.range(-12000, 12000) as i16
                } else {
                    rng.range(-1500, 1500) as i16 >> (j / 4)
                };
            }
        }
        for b in ctrl.ltp_coef_q14.iter_mut() {
            *b = if wild { rng.i16_edgy() } else { rng.range(-3000, 3000) as i16 };
        }
        ctrl.ltp_scale_q14 = [15565, 12288, 8192][rng.range(0, 2) as usize];

        let mut pulses = [0i16; MAX_FRAME_LENGTH];
        for p in pulses[..dec.frame_length as usize].iter_mut() {
            *p = if wild { rng.range(-200, 200) as i16 } else { rng.range(-4, 4) as i16 };
        }
        (dec, ctrl, pulses)
    }

    #[test]
    fn decode_core_matches_the_reference() {
        let mut rng = Lcg(16);
        let mut voiced = 0;
        let mut transition = 0;
        for trial in 0..6000 {
            let fs_khz = [8, 12, 16][trial % 3];
            let nb_subfr = if trial % 5 == 0 { 2 } else { 4 };
            let wild = trial % 4 == 3;
            let (dec, ctrl, pulses) = random_frame(&mut rng, fs_khz, nb_subfr, wild);
            let n = dec.frame_length as usize;
            voiced += usize::from(dec.indices.signal_type as i32 == TYPE_VOICED);
            transition += usize::from(
                dec.loss_cnt > 0
                    && dec.prev_signal_type == TYPE_VOICED
                    && dec.indices.signal_type as i32 != TYPE_VOICED,
            );

            let (mut dec_ref, mut ctrl_ref) = (dec.clone(), ctrl.clone());
            let mut xq_ref = vec![0i16; n];
            decode_core_reference(&mut dec_ref, &mut ctrl_ref, &mut xq_ref, &pulses);

            let (mut dec_new, mut ctrl_new) = (dec.clone(), ctrl.clone());
            let mut xq_new = vec![0x5555i16; n];
            silk_decode_core(&mut dec_new, &mut ctrl_new, &mut xq_new, &pulses);

            let what = format!("trial {trial}: {fs_khz} kHz, {nb_subfr} subframes, wild={wild}");
            assert_eq!(xq_new, xq_ref, "output, {what}");
            assert_eq!(dec_new.exc_q14, dec_ref.exc_q14, "excitation, {what}");
            assert_eq!(dec_new.s_lpc_q14_buf, dec_ref.s_lpc_q14_buf, "LPC state, {what}");
            assert_eq!(dec_new.out_buf, dec_ref.out_buf, "out_buf, {what}");
            assert_eq!(dec_new.prev_gain_q16, dec_ref.prev_gain_q16, "gain, {what}");
            assert_eq!(ctrl_new.pitch_l, ctrl_ref.pitch_l, "pitch lags, {what}");
            assert_eq!(ctrl_new.ltp_coef_q14, ctrl_ref.ltp_coef_q14, "LTP coefficients, {what}");
        }
        assert!(voiced > 1000 && transition > 100, "voiced {voiced}, transitions {transition}");
    }
}
