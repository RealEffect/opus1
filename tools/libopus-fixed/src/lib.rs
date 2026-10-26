//! The fixed-point build of libopus 1.6.1, for the tests of opus1.
//!
//! TEST-ONLY. opus1's SILK encoder is a port of libopus's fixed-point
//! encoder (`silk/fixed`). The libopus that the tests link through
//! `audiopus_sys` is the float build, which runs the float SILK encoder and
//! lacks the fixed-point analysis functions. This crate builds the
//! fixed-point libopus 1.6.1 from the source of the official release, so
//! that the tests can compare with it live, call for call.
//!
//! Where the source comes from (see `build.rs`): the folder that
//! `LIBOPUS_SRC_1_6_1` names, else `opus-1.6.1/` in the repository, else
//! `target/perf/libopus-1.6.1-src/opus-1.6.1/` (where
//! scripts/android/build-libopus.sh looks too). The x86 test scripts set
//! `LIBOPUS_SRC_1_6_1` to the release they download. It must be the release
//! (https://downloads.xiph.org/releases/opus/opus-1.6.1.tar.gz, SHA-256
//! 6ffcb593207be92584df15b32466ed64bbec99109f007c82205f0194572411a1), not
//! libopus's main branch, which adds 1 to the frame energy of the Burg
//! analysis and so codes other packets.
//!
//! Without the source the crate still builds, and every function panics
//! with the reason.

use std::os::raw::{c_int, c_uchar, c_void};

/// libopus constants (include/opus_defines.h, src/opus_private.h).
pub mod consts {
    pub const OPUS_AUTO: i32 = -1000;
    pub const OPUS_BITRATE_MAX: i32 = -1;
    pub const OPUS_APPLICATION_VOIP: i32 = 2048;
    pub const OPUS_APPLICATION_AUDIO: i32 = 2049;
    pub const OPUS_APPLICATION_RESTRICTED_LOWDELAY: i32 = 2051;
    pub const OPUS_BANDWIDTH_NARROWBAND: i32 = 1101;
    pub const OPUS_BANDWIDTH_MEDIUMBAND: i32 = 1102;
    pub const OPUS_BANDWIDTH_WIDEBAND: i32 = 1103;
    pub const OPUS_BANDWIDTH_SUPERWIDEBAND: i32 = 1104;
    pub const OPUS_BANDWIDTH_FULLBAND: i32 = 1105;
    pub const MODE_SILK_ONLY: i32 = 1000;
    pub const MODE_HYBRID: i32 = 1001;
    pub const MODE_CELT_ONLY: i32 = 1002;

    pub const OPUS_SET_BITRATE_REQUEST: i32 = 4002;
    pub const OPUS_SET_VBR_REQUEST: i32 = 4006;
    pub const OPUS_SET_BANDWIDTH_REQUEST: i32 = 4008;
    pub const OPUS_SET_COMPLEXITY_REQUEST: i32 = 4010;
    pub const OPUS_SET_INBAND_FEC_REQUEST: i32 = 4012;
    pub const OPUS_SET_PACKET_LOSS_PERC_REQUEST: i32 = 4014;
    pub const OPUS_SET_DTX_REQUEST: i32 = 4016;
    pub const OPUS_SET_FORCE_CHANNELS_REQUEST: i32 = 4022;
    pub const OPUS_GET_FINAL_RANGE_REQUEST: i32 = 4031;
    pub const OPUS_SET_PREDICTION_DISABLED_REQUEST: i32 = 4042;
    pub const OPUS_GET_IN_DTX_REQUEST: i32 = 4049;
    /// Private to libopus (src/opus_private.h): `MODE_SILK_ONLY`,
    /// `MODE_HYBRID`, `MODE_CELT_ONLY` or `OPUS_AUTO`.
    pub const OPUS_SET_FORCE_MODE_REQUEST: i32 = 11002;
}

/// The arguments of `fixed_ref_nsq` that come from the SILK encoder state
/// and the side information (`fixed_ref_nsq_frame` in shim.c).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NsqFrame {
    pub nb_subfr: c_int,
    pub subfr_length: c_int,
    pub frame_length: c_int,
    pub ltp_mem_length: c_int,
    pub predict_lpc_order: c_int,
    pub shaping_lpc_order: c_int,
    pub n_states_delayed_decision: c_int,
    pub warping_q16: c_int,
    pub signal_type: c_int,
    pub quant_offset_type: c_int,
    pub nlsf_interp_coef_q2: c_int,
    /// Read, and written by the delayed-decision quantizer.
    pub seed: c_int,
}

/// `silk_nsq_state` (silk/structs.h).
#[repr(C)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NsqState {
    pub xq: [i16; 2 * 320],
    pub s_ltp_shp_q14: [i32; 2 * 320],
    pub s_lpc_q14: [i32; 80 + 16],
    pub s_ar2_q14: [i32; 24],
    pub s_lf_ar_shp_q14: i32,
    pub s_diff_shp_q14: i32,
    pub lag_prev: c_int,
    pub s_ltp_buf_idx: c_int,
    pub s_ltp_shp_buf_idx: c_int,
    pub rand_seed: i32,
    pub prev_gain_q16: i32,
    pub rewhite_flag: c_int,
}

impl Default for NsqState {
    fn default() -> Self {
        NsqState {
            xq: [0; 640],
            s_ltp_shp_q14: [0; 640],
            s_lpc_q14: [0; 96],
            s_ar2_q14: [0; 24],
            s_lf_ar_shp_q14: 0,
            s_diff_shp_q14: 0,
            lag_prev: 0,
            s_ltp_buf_idx: 0,
            s_ltp_shp_buf_idx: 0,
            rand_seed: 0,
            prev_gain_q16: 0,
            rewhite_flag: 0,
        }
    }
}

/// Declares the functions of shim.c. Without the library each becomes a
/// function that panics with the reason the build script gave.
macro_rules! shim_functions {
    ($( fn $name:ident($($arg:ident: $ty:ty),* $(,)?) $(-> $ret:ty)?; )*) => {
        #[cfg(not(libopus_fixed_missing))]
        unsafe extern "C" {
            $( fn $name($($arg: $ty),*) $(-> $ret)?; )*
        }
        $(
            #[cfg(libopus_fixed_missing)]
            #[allow(unused_variables, clippy::too_many_arguments)]
            unsafe fn $name($($arg: $ty),*) $(-> $ret)? {
                panic!("the fixed-point libopus 1.6.1 was not built: {}", env!("LIBOPUS_FIXED_MISSING"))
            }
        )*
    };
}

shim_functions! {
    fn fixed_ref_encoder_create(fs: i32, channels: c_int, application: c_int, error: *mut c_int) -> *mut c_void;
    fn fixed_ref_encoder_destroy(st: *mut c_void);
    fn fixed_ref_encoder_set(st: *mut c_void, request: c_int, value: i32) -> c_int;
    fn fixed_ref_encoder_get(st: *mut c_void, request: c_int, value: *mut i32) -> c_int;
    fn fixed_ref_encode(st: *mut c_void, pcm: *const i16, frame_size: c_int, data: *mut c_uchar, max_data_bytes: i32) -> i32;
    fn fixed_ref_burg_modified(
        res_nrg: *mut i32,
        res_nrg_q: *mut c_int,
        a_q16: *mut i32,
        x: *const i16,
        min_inv_gain_q30: i32,
        subfr_length: c_int,
        nb_subfr: c_int,
        order: c_int,
    );
    fn fixed_ref_pitch_analysis_core(
        frame: *const i16,
        pitch_out: *mut c_int,
        lag_index: *mut i16,
        contour_index: *mut i8,
        ltp_corr_q15: *mut c_int,
        prev_lag: c_int,
        search_thres1_q16: i32,
        search_thres2_q13: c_int,
        fs_khz: c_int,
        complexity: c_int,
        nb_subfr: c_int,
    ) -> c_int;
    fn fixed_ref_hp_variable_cutoff(
        fs_khz: c_int,
        prev_signal_type: c_int,
        prev_lag: c_int,
        input_quality_band0_q15: c_int,
        speech_activity_q8: c_int,
        variable_hp_smth1_q15: i32,
    ) -> i32;
    fn fixed_ref_control_audio_bandwidth(b: *mut BandwidthState) -> c_int;
    fn fixed_ref_nsq_state_size() -> c_int;
    fn fixed_ref_nsq(
        del_dec: c_int,
        frame: *mut NsqFrame,
        state: *mut NsqState,
        x16: *const i16,
        pulses: *mut i8,
        pred_coef_q12: *const i16,
        ltp_coef_q14: *const i16,
        ar_q13: *const i16,
        harm_shape_gain_q14: *const c_int,
        tilt_q14: *const c_int,
        lf_shp_q14: *const i32,
        gains_q16: *const i32,
        pitch_l: *const c_int,
        lambda_q10: c_int,
        ltp_scale_q14: c_int,
    );
    fn fixed_ref_process_nlsfs(
        fs_khz: c_int,
        predict_lpc_order: c_int,
        nb_subfr: c_int,
        speech_activity_q8: c_int,
        n_survivors: c_int,
        signal_type: c_int,
        use_interpolated_nlsfs: c_int,
        nlsf_interp_coef_q2: *mut c_int,
        pred_coef_q12: *mut i16,
        nlsf_q15: *mut i16,
        prev_nlsfq_q15: *const i16,
        nlsf_indices: *mut i8,
    );
}

/// An `OpusEncoder` of the fixed-point build.
pub struct Encoder {
    st: *mut c_void,
    channels: usize,
}

impl Encoder {
    /// `opus_encoder_create`. Panics if libopus refuses the arguments.
    pub fn new(fs: i32, channels: usize, application: i32) -> Encoder {
        let mut error = 0;
        // SAFETY: plain values and a pointer to a local integer.
        let st = unsafe { fixed_ref_encoder_create(fs, channels as c_int, application, &mut error) };
        assert!(error == 0 && !st.is_null(), "opus_encoder_create({fs}, {channels}, {application}): {error}");
        Encoder { st, channels }
    }

    /// `opus_encoder_ctl` with a control that takes one integer, such as
    /// `OPUS_SET_BITRATE_REQUEST`. Panics if libopus refuses it.
    pub fn set(&mut self, request: i32, value: i32) {
        // SAFETY: `st` is a live encoder; the control takes an integer.
        let ret = unsafe { fixed_ref_encoder_set(self.st, request, value) };
        assert_eq!(ret, 0, "opus_encoder_ctl({request}, {value})");
    }

    /// `opus_encoder_ctl` with a control that returns one integer, such as
    /// `OPUS_GET_IN_DTX_REQUEST`.
    pub fn get(&mut self, request: i32) -> i32 {
        let mut value = 0;
        // SAFETY: `st` is a live encoder; the control writes one integer.
        let ret = unsafe { fixed_ref_encoder_get(self.st, request, &mut value) };
        assert_eq!(ret, 0, "opus_encoder_ctl({request})");
        value
    }

    /// `opus_encode`: one packet of at most `max_bytes` bytes from
    /// `frame_size` samples per channel, or libopus's error code.
    pub fn encode(&mut self, pcm: &[i16], frame_size: usize, max_bytes: usize) -> Result<Vec<u8>, i32> {
        assert!(pcm.len() >= frame_size * self.channels, "{} samples for a frame of {frame_size}", pcm.len());
        let mut data = vec![0u8; max_bytes];
        // SAFETY: `pcm` holds `frame_size` samples per channel and `data`
        // `max_bytes` bytes, which is what libopus reads and writes.
        let n = unsafe {
            fixed_ref_encode(self.st, pcm.as_ptr(), frame_size as c_int, data.as_mut_ptr(), max_bytes as i32)
        };
        if n < 0 {
            return Err(n);
        }
        data.truncate(n as usize);
        Ok(data)
    }
}

impl Drop for Encoder {
    fn drop(&mut self) {
        // SAFETY: `st` came from `opus_encoder_create` and is freed once.
        unsafe { fixed_ref_encoder_destroy(self.st) };
    }
}

/// `silk_burg_modified_c` (silk/fixed/burg_modified_FIX.c): the residual
/// energy, its Q value and the `order` prediction coefficients (Q16).
pub fn burg_modified(
    x: &[i16],
    min_inv_gain_q30: i32,
    subfr_length: usize,
    nb_subfr: usize,
    order: usize,
) -> (i32, i32, Vec<i32>) {
    assert!(x.len() >= subfr_length * nb_subfr && order <= 24);
    let (mut res_nrg, mut res_nrg_q) = (0, 0);
    let mut a_q16 = [0i32; 24];
    // SAFETY: `x` holds `nb_subfr * subfr_length` samples, `a_q16` room
    // for `order` coefficients.
    unsafe {
        fixed_ref_burg_modified(
            &mut res_nrg,
            &mut res_nrg_q,
            a_q16.as_mut_ptr(),
            x.as_ptr(),
            min_inv_gain_q30,
            subfr_length as c_int,
            nb_subfr as c_int,
            order as c_int,
        )
    };
    (res_nrg, res_nrg_q, a_q16[..order].to_vec())
}

/// What `silk_pitch_analysis_core` returns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PitchCore {
    /// 0: voiced, 1: unvoiced.
    pub unvoiced: i32,
    pub pitch_out: [i32; 4],
    pub lag_index: i16,
    pub contour_index: i8,
    pub ltp_corr_q15: i32,
}

/// `silk_pitch_analysis_core` (silk/fixed/pitch_analysis_core_FIX.c).
/// `ltp_corr_q15` is the correlation of the previous frame.
#[allow(clippy::too_many_arguments)]
pub fn pitch_analysis_core(
    frame: &[i16],
    ltp_corr_q15: i32,
    prev_lag: i32,
    search_thres1_q16: i32,
    search_thres2_q13: i32,
    fs_khz: i32,
    complexity: i32,
    nb_subfr: usize,
) -> PitchCore {
    // 20 ms of history and 5 ms per subframe (pitch_analysis_core_FIX.c).
    let frame_ms = 20 + 5 * nb_subfr;
    assert!(frame.len() >= frame_ms * fs_khz as usize, "{} samples", frame.len());
    let mut out = PitchCore { unvoiced: 0, pitch_out: [0; 4], lag_index: 0, contour_index: 0, ltp_corr_q15 };
    // SAFETY: `frame` holds a whole analysis frame; the outputs are locals.
    out.unvoiced = unsafe {
        fixed_ref_pitch_analysis_core(
            frame.as_ptr(),
            out.pitch_out.as_mut_ptr(),
            &mut out.lag_index,
            &mut out.contour_index,
            &mut out.ltp_corr_q15,
            prev_lag,
            search_thres1_q16,
            search_thres2_q13,
            fs_khz,
            complexity,
            nb_subfr as c_int,
        )
    };
    out
}

/// What `silk_HP_variable_cutoff` (silk/HP_variable_cutoff.c) reads from
/// the encoder state.
#[derive(Clone, Copy, Debug)]
pub struct HpCutoffInput {
    pub fs_khz: i32,
    pub prev_signal_type: i32,
    pub prev_lag: i32,
    pub input_quality_band0_q15: i32,
    pub speech_activity_q8: i32,
    pub variable_hp_smth1_q15: i32,
}

/// `silk_HP_variable_cutoff`: the new `variable_HP_smth1_Q15`.
pub fn hp_variable_cutoff(s: &HpCutoffInput) -> i32 {
    // SAFETY: plain values.
    unsafe {
        fixed_ref_hp_variable_cutoff(
            s.fs_khz,
            s.prev_signal_type,
            s.prev_lag,
            s.input_quality_band0_q15,
            s.speech_activity_q8,
            s.variable_hp_smth1_q15,
        )
    }
}

/// What `silk_control_audio_bandwidth` (silk/control_audio_bandwidth.c)
/// reads from the encoder state and the control structure, and writes back
/// (`fixed_ref_bandwidth` in shim.c).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BandwidthState {
    pub fs_khz: c_int,
    pub allow_bandwidth_switch: c_int,
    pub api_fs_hz: i32,
    pub max_internal_fs_hz: c_int,
    pub min_internal_fs_hz: c_int,
    pub desired_internal_fs_hz: c_int,
    /// Written back.
    pub in_lp_state: [i32; 2],
    /// Written back.
    pub transition_frame_no: i32,
    /// Written back.
    pub mode: c_int,
    pub saved_fs_khz: i32,
    pub opus_can_switch: c_int,
    pub payload_size_ms: c_int,
    /// Written back.
    pub max_bits: c_int,
    /// Written back.
    pub switch_ready: c_int,
}

/// `silk_control_audio_bandwidth`: the internal rate in kHz. `b` gets the
/// new low-pass state, `maxBits` and `switchReady`.
pub fn control_audio_bandwidth(b: &mut BandwidthState) -> i32 {
    // SAFETY: `b` has the layout of `fixed_ref_bandwidth`.
    unsafe { fixed_ref_control_audio_bandwidth(b) }
}

/// The parameters of one frame of `silk_NSQ_c` and `silk_NSQ_del_dec_c`,
/// with the lengths libopus gives them.
pub struct NsqParams<'a> {
    /// 2 * 16 (MAX_LPC_ORDER) values.
    pub pred_coef_q12: &'a [i16],
    /// 4 * 5 (MAX_NB_SUBFR * LTP_ORDER) values.
    pub ltp_coef_q14: &'a [i16],
    /// 4 * 24 (MAX_NB_SUBFR * MAX_SHAPE_LPC_ORDER) values.
    pub ar_q13: &'a [i16],
    pub harm_shape_gain_q14: &'a [i32],
    pub tilt_q14: &'a [i32],
    pub lf_shp_q14: &'a [i32],
    pub gains_q16: &'a [i32],
    pub pitch_l: &'a [i32],
    pub lambda_q10: i32,
    pub ltp_scale_q14: i32,
}

/// One frame of `silk_NSQ_c` (`del_dec` false) or `silk_NSQ_del_dec_c`
/// (true) of silk/NSQ.c and silk/NSQ_del_dec.c. `pulses` gets
/// `frame.frame_length` values; `frame.seed` may change.
pub fn nsq(del_dec: bool, frame: &mut NsqFrame, state: &mut NsqState, x16: &[i16], pulses: &mut [i8], p: &NsqParams) {
    // SAFETY: no arguments.
    let size = unsafe { fixed_ref_nsq_state_size() };
    assert_eq!(size as usize, size_of::<NsqState>(), "layout of silk_nsq_state");
    let n = frame.frame_length as usize;
    assert!(x16.len() >= n && pulses.len() >= n && n <= 320);
    assert!(p.pred_coef_q12.len() >= 32 && p.ltp_coef_q14.len() >= 20 && p.ar_q13.len() >= 96);
    for v in [p.harm_shape_gain_q14, p.tilt_q14, p.lf_shp_q14, p.gains_q16, p.pitch_l] {
        assert!(v.len() >= 4);
    }
    // SAFETY: every buffer has at least the length libopus reads or
    // writes (checked above), and `state` has the layout of the C struct.
    unsafe {
        fixed_ref_nsq(
            del_dec as c_int,
            frame,
            state,
            x16.as_ptr(),
            pulses.as_mut_ptr(),
            p.pred_coef_q12.as_ptr(),
            p.ltp_coef_q14.as_ptr(),
            p.ar_q13.as_ptr(),
            p.harm_shape_gain_q14.as_ptr(),
            p.tilt_q14.as_ptr(),
            p.lf_shp_q14.as_ptr(),
            p.gains_q16.as_ptr(),
            p.pitch_l.as_ptr(),
            p.lambda_q10,
            p.ltp_scale_q14,
        )
    }
}

/// What `silk_process_NLSFs` (silk/process_NLSFs.c) reads from the
/// encoder state.
#[derive(Clone, Copy, Debug)]
pub struct NlsfInput {
    /// 8, 12 or 16; 16 takes the wideband codebook.
    pub fs_khz: i32,
    pub predict_lpc_order: i32,
    pub nb_subfr: i32,
    pub speech_activity_q8: i32,
    pub n_survivors: i32,
    pub signal_type: i32,
    pub use_interpolated_nlsfs: i32,
    pub nlsf_interp_coef_q2: i32,
}

/// What `silk_process_NLSFs` returns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NlsfOutput {
    pub pred_coef_q12: [[i16; 16]; 2],
    /// The quantized vector.
    pub nlsf_q15: [i16; 16],
    pub nlsf_indices: [i8; 17],
    pub nlsf_interp_coef_q2: i32,
}

/// `silk_process_NLSFs` on `nlsf_q15`.
pub fn process_nlsfs(input: &NlsfInput, nlsf_q15: &[i16; 16], prev_nlsfq_q15: &[i16; 16]) -> NlsfOutput {
    let mut out = NlsfOutput {
        pred_coef_q12: [[0; 16]; 2],
        nlsf_q15: *nlsf_q15,
        nlsf_indices: [0; 17],
        nlsf_interp_coef_q2: input.nlsf_interp_coef_q2,
    };
    // SAFETY: the arrays have the sizes the function reads and writes
    // ([2][MAX_LPC_ORDER], MAX_LPC_ORDER and MAX_LPC_ORDER + 1).
    unsafe {
        fixed_ref_process_nlsfs(
            input.fs_khz,
            input.predict_lpc_order,
            input.nb_subfr,
            input.speech_activity_q8,
            input.n_survivors,
            input.signal_type,
            input.use_interpolated_nlsfs,
            &mut out.nlsf_interp_coef_q2,
            out.pred_coef_q12.as_mut_ptr().cast(),
            out.nlsf_q15.as_mut_ptr(),
            prev_nlsfq_q15.as_ptr(),
            out.nlsf_indices.as_mut_ptr(),
        )
    };
    out
}
