#[cfg(feature = "osce-bwe")]
use crate::dnn::osce::OsceBweChannel;
#[cfg(feature = "osce")]
use crate::dnn::osce::OsceChannel;
use crate::silk::define::*;
use crate::silk::resampler::SilkResampler;
use crate::silk::structs::{NLSFCodebook, SideInfoIndices};

/// Where a channel keeps its OSCE states (14 KB and 8 KB): behind a `Box`
/// with the feature `heap`, inline without it. A `SilkDecoder` with them
/// inline would be built on the stack before `OpusDecoder` boxes it, which
/// doubles the stack that building a decoder needs.
#[cfg(all(feature = "osce", feature = "heap"))]
pub type OsceHeld<T> = Box<T>;
/// Where a channel keeps its OSCE states: inline without `heap`.
#[cfg(all(feature = "osce", not(feature = "heap")))]
pub type OsceHeld<T> = T;

#[cfg(all(feature = "osce", feature = "heap"))]
fn osce_held<T>(value: T) -> OsceHeld<T> {
    Box::new(value)
}
#[cfg(all(feature = "osce", not(feature = "heap")))]
fn osce_held<T>(value: T) -> OsceHeld<T> {
    value
}

#[derive(Clone)]
pub struct SilkCNGState {
    pub cng_exc_buf_q14: [i32; MAX_FRAME_LENGTH],
    pub cng_smth_nlsf_q15: [i16; MAX_LPC_ORDER],
    pub cng_synth_state: [i32; MAX_LPC_ORDER],
    pub cng_smth_gain_q16: i32,
    pub rand_seed: i32,
    pub fs_khz: i32,
}

impl Default for SilkCNGState {
    fn default() -> Self {
        Self {
            cng_exc_buf_q14: [0; MAX_FRAME_LENGTH],
            cng_smth_nlsf_q15: [0; MAX_LPC_ORDER],
            cng_synth_state: [0; MAX_LPC_ORDER],
            cng_smth_gain_q16: 0,
            rand_seed: 0,
            fs_khz: 0,
        }
    }
}

#[derive(Clone, Default)]
pub struct SilkPLCState {
    pub pitch_l_q8: i32,
    pub ltp_coef_q14: [i16; LTP_ORDER],
    pub prev_lpc_q12: [i16; MAX_LPC_ORDER],
    pub last_frame_lost: i32,
    pub rand_seed: i32,
    pub rand_scale_q14: i16,
    pub conc_energy: i32,
    pub conc_energy_shift: i32,
    pub prev_ltp_scale_q14: i16,
    pub prev_gain_q16: [i32; 2],
    pub fs_khz: i32,
    pub nb_subfr: i32,
    pub subfr_length: i32,
    pub enable_deep_plc: i32,
}

#[derive(Clone)]
pub struct SilkDecoderState {
    pub prev_gain_q16: i32,
    pub exc_q14: [i32; MAX_FRAME_LENGTH],
    pub s_lpc_q14_buf: [i32; MAX_LPC_ORDER],
    pub out_buf: [i16; MAX_FRAME_LENGTH + 2 * MAX_SUB_FRAME_LENGTH],
    pub lag_prev: i32,
    pub last_gain_index: i8,
    pub fs_khz: i32,
    pub fs_api_hz: i32,
    pub nb_subfr: i32,
    pub frame_length: i32,
    pub subfr_length: i32,
    pub ltp_mem_length: i32,
    pub lpc_order: i32,
    pub prev_nlsf_q15: [i16; MAX_LPC_ORDER],
    pub first_frame_after_reset: i32,
    pub pitch_lag_low_bits_icdf: &'static [u8],
    pub pitch_contour_icdf: &'static [u8],
    pub n_frames_decoded: i32,
    pub n_frames_per_packet: i32,
    pub ec_prev_signal_type: i32,
    pub ec_prev_lag_index: i16,
    pub vad_flags: [i32; MAX_FRAMES_PER_PACKET],
    pub lbrr_flag: i32,
    pub lbrr_flags: [i32; MAX_FRAMES_PER_PACKET],
    /// Resampler from the SILK internal rate (`fs_khz`) to the output rate,
    /// libopus `resampler_state`. `silk_decoder_set_fs` sets it up. Only
    /// `SilkDecoder::decode_with_control` runs it.
    pub resampler_state: SilkResampler,
    /// The output rate `resampler_state` was set up for, 0 if it was not
    /// set up. In libopus this is `fs_API_hz` itself. It is kept apart here
    /// because callers of `SilkDecoder::decode` write the rate they want
    /// into `fs_api_hz` directly, and the resampler must not miss a change
    /// of rate that was made that way.
    pub resampler_api_hz: i32,
    pub ps_nlsf_cb: Option<&'static NLSFCodebook>,
    pub indices: SideInfoIndices,
    pub s_cng: SilkCNGState,
    pub loss_cnt: i32,
    pub prev_signal_type: i32,
    pub s_plc: SilkPLCState,
    /// The OSCE state of the channel (libopus `psDec->osce`): its method,
    /// the feature history and the state of LACE or NoLACE.
    /// `silk_reset_decoder` resets it to `OSCE_DEFAULT_METHOD`.
    #[cfg(feature = "osce")]
    pub osce: OsceHeld<OsceChannel>,
    /// The bandwidth extension state of the channel (libopus
    /// `psDec->osce_bwe`). `silk_init_decoder` clears it to zero;
    /// `silk_reset_decoder` keeps it, as libopus does.
    #[cfg(feature = "osce-bwe")]
    pub osce_bwe: OsceHeld<OsceBweChannel>,
}

impl Default for SilkDecoderState {
    fn default() -> Self {
        Self {
            prev_gain_q16: 65536,
            exc_q14: [0; MAX_FRAME_LENGTH],
            s_lpc_q14_buf: [0; MAX_LPC_ORDER],
            out_buf: [0; MAX_FRAME_LENGTH + 2 * MAX_SUB_FRAME_LENGTH],
            lag_prev: 100,
            last_gain_index: 10,
            fs_khz: 0,
            fs_api_hz: 0,
            nb_subfr: 0,
            frame_length: 0,
            subfr_length: 0,
            ltp_mem_length: 0,
            lpc_order: 0,
            prev_nlsf_q15: [0; MAX_LPC_ORDER],
            first_frame_after_reset: 1,
            pitch_lag_low_bits_icdf: &crate::silk::tables::SILK_UNIFORM4_ICDF,
            pitch_contour_icdf: &crate::silk::tables::SILK_PITCH_CONTOUR_ICDF,
            n_frames_decoded: 0,
            n_frames_per_packet: 0,
            ec_prev_signal_type: 0,
            ec_prev_lag_index: 0,
            vad_flags: [0; MAX_FRAMES_PER_PACKET],
            lbrr_flag: 0,
            lbrr_flags: [0; MAX_FRAMES_PER_PACKET],
            resampler_state: SilkResampler::default(),
            resampler_api_hz: 0,
            // Default to the WB codebook so that decoding before an explicit
            // `silk_decoder_set_fs` cannot panic on `unwrap`. `set_fs` always
            // overrides this on the normal path.
            ps_nlsf_cb: Some(&crate::silk::tables_nlsf::SILK_NLSF_CB_WB),
            indices: SideInfoIndices::default(),
            s_cng: SilkCNGState::default(),
            loss_cnt: 0,
            prev_signal_type: TYPE_NO_VOICE_ACTIVITY,
            s_plc: SilkPLCState::default(),
            #[cfg(feature = "osce")]
            osce: osce_held(OsceChannel::new()),
            #[cfg(feature = "osce-bwe")]
            osce_bwe: osce_held(OsceBweChannel::new()),
        }
    }
}

/// What the caller tells `SilkDecoder::decode_with_control` about the
/// frame to decode (libopus `silk_DecControlStruct`).
///
/// The two channel counts are separate because a packet does not have to
/// code as many channels as the caller wants: a stereo decoder also gets
/// mono packets, and a mono decoder also gets stereo packets.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SilkDecControl {
    /// Channels in the output buffer: 1 or 2 (`nChannelsAPI`).
    pub n_channels_api: i32,
    /// Channels coded in the packet: 1 or 2 (`nChannelsInternal`).
    pub n_channels_internal: i32,
    /// Output sampling rate in Hz: 8000, 12000, 16000, 24000 or 48000
    /// (`API_sampleRate`).
    pub api_sample_rate: i32,
    /// SILK sampling rate of the packet in Hz: 8000, 12000 or 16000
    /// (`internalSampleRate`).
    pub internal_sample_rate: i32,
    /// Duration of the Opus frame in ms: 10, 20, 40 or 60. 0 counts as 10,
    /// as in libopus (`payloadSize_ms`).
    pub payload_size_ms: i32,
}

#[derive(Clone, Default)]
pub struct SilkDecoderControl {
    pub pitch_l: [i32; MAX_NB_SUBFR],
    pub gains_q16: [i32; MAX_NB_SUBFR],
    pub pred_coef_q12: [[i16; MAX_LPC_ORDER]; 2],
    pub ltp_coef_q14: [i16; LTP_ORDER * MAX_NB_SUBFR],
    pub ltp_scale_q14: i32,
}

impl SilkDecoderControl {
    pub fn new() -> Self {
        Self {
            pitch_l: [0; MAX_NB_SUBFR],
            gains_q16: [0; MAX_NB_SUBFR],
            pred_coef_q12: [[0; MAX_LPC_ORDER]; 2],
            ltp_coef_q14: [0; LTP_ORDER * MAX_NB_SUBFR],
            ltp_scale_q14: 0,
        }
    }
}
