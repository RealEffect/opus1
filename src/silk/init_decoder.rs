use crate::silk::cng::silk_cng_reset;
use crate::silk::decoder_structs::SilkDecoderState;
use crate::silk::define::*;
use crate::silk::plc::silk_plc_reset;
use crate::silk::resampler::SilkResampler;
use crate::silk::tables_nlsf::*;

/// Set the sampling rate of one channel (libopus `silk_decoder_set_fs`).
///
/// `fs_khz` is the SILK rate of the packet, `fs_api_hz` the output rate.
/// Returns 0, or a negative value when a rate is not supported.
pub fn silk_decoder_set_fs(dec: &mut SilkDecoderState, fs_khz: i32, fs_api_hz: i32) -> i32 {
    if fs_khz != 8 && fs_khz != 12 && fs_khz != 16 {
        return -1;
    }
    let mut ret = 0;

    // libopus decoder_set_fs.c: the caller sets `nb_subfr` first (4 for a
    // 20 ms SILK frame, 2 for a 10 ms one) and the frame length and the
    // pitch contour table below follow from it, so a 10 ms frame gets the
    // 10 ms table. A state whose `nb_subfr` was never set counts as 20 ms.
    if dec.nb_subfr != (MAX_NB_SUBFR / 2) as i32 {
        dec.nb_subfr = MAX_NB_SUBFR as i32;
    }
    dec.subfr_length = (SUB_FRAME_LENGTH_MS as i32) * fs_khz;
    let frame_length = dec.subfr_length * dec.nb_subfr;

    // The resampler starts afresh when either of its two rates changes.
    // libopus compares with `fs_API_hz`; see `resampler_api_hz` for why
    // this compares with the rate the resampler really has.
    if dec.fs_khz != fs_khz || dec.resampler_api_hz != fs_api_hz {
        let err = dec.resampler_state.init(fs_khz * 1000, fs_api_hz);
        dec.resampler_api_hz = if err == 0 { fs_api_hz } else { 0 };
        ret += err;
    }
    dec.fs_api_hz = fs_api_hz;

    if dec.fs_khz != fs_khz || frame_length != dec.frame_length {
        dec.pitch_contour_icdf = if fs_khz == 8 {
            if dec.nb_subfr == MAX_NB_SUBFR as i32 {
                &crate::silk::tables::SILK_PITCH_CONTOUR_NB_ICDF
            } else {
                &crate::silk::tables::SILK_PITCH_CONTOUR_10_MS_NB_ICDF
            }
        } else if dec.nb_subfr == MAX_NB_SUBFR as i32 {
            &crate::silk::tables::SILK_PITCH_CONTOUR_ICDF
        } else {
            &crate::silk::tables::SILK_PITCH_CONTOUR_10_MS_ICDF
        };

        if dec.fs_khz != fs_khz {
            // Bandwidth switch (libopus decoder_set_fs.c): reset the state that
            // is tied to the old rate. A `lag_prev` from the old rate (e.g.
            // 288 at 16 kHz) would index out of bounds in silk_decode_core
            // at 8 kHz.
            dec.ltp_mem_length = (LTP_MEM_LENGTH_MS as i32) * fs_khz;
            // NB and MB share the order-10 NLSF codebook; only WB codes 16
            // coefficients. Set both from one branch, as libopus does: the
            // decoder reads `ps_nlsf_cb.order` NLSF indices and filters with
            // `lpc_order` coefficients, so the two must match.
            let (lpc_order, nlsf_cb) = if fs_khz == 8 || fs_khz == 12 {
                (MIN_LPC_ORDER, &SILK_NLSF_CB_NB_MB)
            } else {
                (MAX_LPC_ORDER, &SILK_NLSF_CB_WB)
            };
            dec.lpc_order = lpc_order as i32;
            dec.ps_nlsf_cb = Some(nlsf_cb);
            dec.pitch_lag_low_bits_icdf = match fs_khz {
                8 => &crate::silk::tables::SILK_UNIFORM4_ICDF,
                12 => &crate::silk::tables::SILK_UNIFORM6_ICDF,
                _ => &crate::silk::tables::SILK_UNIFORM8_ICDF,
            };
            dec.first_frame_after_reset = 1;
            dec.lag_prev = 100;
            dec.last_gain_index = 10;
            dec.prev_signal_type = TYPE_NO_VOICE_ACTIVITY;
            dec.out_buf.fill(0);
            dec.s_lpc_q14_buf.fill(0);
        }

        dec.fs_khz = fs_khz;
        dec.frame_length = frame_length;
    }

    ret
}

/// Put one channel back into its start state (libopus `silk_reset_decoder`).
///
/// libopus clears the whole state, the sampling rate included. The next
/// `silk_decoder_set_fs` therefore sets up everything that depends on the
/// rate again: the tables, the filter memories and the resampler.
///
/// Three kinds of fields get a usable value instead of the zero that
/// libopus writes, so that no call order can index out of range:
/// `lag_prev` and `last_gain_index` (to the values `silk_decoder_set_fs`
/// gives them anyway) and the table references.
pub fn silk_reset_decoder(ps_dec: &mut SilkDecoderState) -> i32 {
    ps_dec.prev_gain_q16 = 1 << 16;
    ps_dec.exc_q14.fill(0);
    ps_dec.s_lpc_q14_buf.fill(0);
    ps_dec.out_buf.fill(0);
    ps_dec.lag_prev = 100;
    ps_dec.last_gain_index = 10;
    ps_dec.fs_khz = 0;
    ps_dec.fs_api_hz = 0;
    ps_dec.nb_subfr = 0;
    ps_dec.frame_length = 0;
    ps_dec.subfr_length = 0;
    ps_dec.ltp_mem_length = 0;
    ps_dec.lpc_order = 0;
    ps_dec.prev_nlsf_q15.fill(0);
    ps_dec.first_frame_after_reset = 1;
    ps_dec.pitch_lag_low_bits_icdf = &crate::silk::tables::SILK_UNIFORM4_ICDF;
    ps_dec.pitch_contour_icdf = &crate::silk::tables::SILK_PITCH_CONTOUR_ICDF;
    ps_dec.n_frames_decoded = 0;
    ps_dec.n_frames_per_packet = 0;
    ps_dec.ec_prev_signal_type = 0;
    ps_dec.ec_prev_lag_index = 0;
    ps_dec.vad_flags.fill(0);
    ps_dec.lbrr_flag = 0;
    ps_dec.lbrr_flags.fill(0);
    ps_dec.resampler_state = SilkResampler::default();
    ps_dec.resampler_api_hz = 0;
    ps_dec.ps_nlsf_cb = Some(&SILK_NLSF_CB_WB);
    ps_dec.indices = Default::default();
    ps_dec.loss_cnt = 0;
    ps_dec.prev_signal_type = TYPE_NO_VOICE_ACTIVITY;

    // Comfort noise state (libopus clears it, then calls silk_CNG_Reset).
    let cng = &mut ps_dec.s_cng;
    cng.cng_exc_buf_q14.fill(0);
    cng.cng_smth_nlsf_q15.fill(0);
    cng.cng_synth_state.fill(0);
    cng.fs_khz = 0;
    silk_cng_reset(ps_dec);

    // Concealment state (libopus clears it, then calls silk_PLC_Reset).
    let plc = &mut ps_dec.s_plc;
    plc.ltp_coef_q14.fill(0);
    plc.prev_lpc_q12.fill(0);
    plc.last_frame_lost = 0;
    plc.rand_seed = 0;
    plc.rand_scale_q14 = 0;
    plc.conc_energy = 0;
    plc.conc_energy_shift = 0;
    plc.prev_ltp_scale_q14 = 0;
    plc.fs_khz = 0;
    plc.enable_deep_plc = 0;
    silk_plc_reset(ps_dec);

    0
}

/// Initialise one channel (libopus `silk_init_decoder`). In libopus this
/// clears the state and then calls `silk_reset_decoder`, which clears it
/// too, so the two do the same.
pub fn silk_init_decoder(ps_dec: &mut SilkDecoderState) -> i32 {
    silk_reset_decoder(ps_dec)
}

pub fn silk_create_decoder() -> SilkDecoderState {
    let mut dec = SilkDecoderState::default();
    silk_init_decoder(&mut dec);
    dec
}
