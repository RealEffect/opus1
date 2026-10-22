//! Internal parts of the codec, for the tests, benchmarks, examples and
//! fuzz targets of this repository.
//!
//! This module is not part of the stable API. It is hidden from the
//! documentation, and any release may change or remove what it holds
//! without notice. Applications use the items at the crate root and the
//! modules `packet`, `extensions` and `multistream`.
//!
//! Each module here has the name and the place of the internal module whose
//! items it re-exports, and only the items that code outside the crate
//! uses. The exception is `silk_input`, which holds a test control of the
//! encoder.

pub mod analysis {
    pub use crate::analysis::{
        AnalysisGate, AnalysisInfo, AnalysisPcm, LEAK_BANDS, PcmRef, SecondChannel,
        is_digital_silence,
    };

    /// Sets when `enc` runs the speech and music analysis. Tests that
    /// compare with a fixed-point build of libopus use
    /// [`AnalysisGate::FixedPointBuild`].
    pub fn set_analysis_gate(enc: &mut crate::OpusEncoder, gate: AnalysisGate) {
        enc.set_analysis_gate(gate);
    }

    /// The gate [`set_analysis_gate`] set.
    pub fn analysis_gate(enc: &crate::OpusEncoder) -> AnalysisGate {
        enc.analysis_gate()
    }

    /// The analysis the last frame `enc` coded went by: the `AnalysisInfo`
    /// libopus hands to `opus_encode_frame_native`. Invalid when the
    /// analysis did not run.
    pub fn last_analysis_info(enc: &crate::OpusEncoder) -> AnalysisInfo {
        enc.last_analysis_info()
    }
    #[cfg(feature = "analysis")]
    pub use crate::analysis::{
        ANALYSIS_BUF_SIZE, ANALYSIS_COUNT_MAX, DETECT_SIZE, NB_FRAMES, NB_TBANDS,
        TonalityAnalysisState, fast_atan2f,
    };
    #[cfg(feature = "analysis")]
    pub mod mlp {
        pub use crate::analysis::mlp::{
            DenseLayer, GruLayer, LAYER0, LAYER1, LAYER2, MAX_NEURONS, WEIGHTS_SCALE,
            compute_dense, compute_gru, sigmoid_approx, tansig_approx,
        };
    }
    #[cfg(feature = "analysis")]
    pub mod mlp_data {
        pub use crate::analysis::mlp_data::*;
    }
}
pub mod bands {
    pub use crate::bands::{
        amp2log2, anti_collapse, compute_band_energies, denormalise_bands, log2amp,
        normalise_bands, quant_all_bands, quant_all_bands_enc, stereo_itheta,
    };
}
pub mod celt {
    pub use crate::celt::{CeltDecoder, CeltEncoder, PcmLayout};
}
pub mod celt_lpc {
    pub use crate::celt_lpc::{autocorr, lpc};
}
/// The evaluation tools of `src/eval/`: `opus_compare`, MOC and the
/// `opus_demo -d` decoding loop.
#[cfg(feature = "std")]
pub mod eval {
    pub use crate::eval::*;
}
/// The neural network code (`src/dnn/`), whole: every public item of every
/// submodule, for the tests that compare it with a DNN-enabled libopus.
#[cfg(feature = "dnn")]
pub mod dnn {
    pub use crate::dnn::*;
}
/// libopus `compute_stereo_width` of the encoder, for the tests that compare
/// it with the C function.
pub mod stereo_width {
    /// C `StereoWidthState` (src/opus_encoder.c), with the C layout.
    #[repr(C)]
    #[derive(Clone, Copy, Debug, Default, PartialEq)]
    pub struct State {
        pub xx: f32,
        pub xy: f32,
        pub yy: f32,
        pub smoothed_width: f32,
        pub max_follower: f32,
    }

    /// The width of a stereo frame of `frame_size` samples per channel at
    /// `fs`, which also updates `state`.
    pub fn compute(pcm: &[f32], frame_size: usize, fs: i32, state: &mut State) -> f32 {
        let mut mem = crate::StereoWidthState {
            xx: state.xx,
            xy: state.xy,
            yy: state.yy,
            smoothed_width: state.smoothed_width,
            max_follower: state.max_follower,
        };
        let width = crate::compute_stereo_width(pcm, frame_size, fs, &mut mem);
        *state = State {
            xx: mem.xx,
            xy: mem.xy,
            yy: mem.yy,
            smoothed_width: mem.smoothed_width,
            max_follower: mem.max_follower,
        };
        width
    }
}
pub mod hp_cutoff {
    pub use crate::hp_cutoff::{dc_reject_float, hp_cutoff};
}
pub mod kiss_fft {
    pub use crate::kiss_fft::{
        FFT_TWIDDLES_480, KissCpx, KissFftState, opus_fft, opus_fft_impl, opus_fft_impl_fused,
        opus_fft_impl_scalar,
    };
}
pub mod mdct {
    pub use crate::mdct::MdctLookup;
}
pub mod modes {
    pub use crate::modes::{
        CeltMode, SPREAD_ICDF, TAPSET_ICDF, TF_SELECT_TABLE, TRIM_ICDF, default_mode,
    };
}
#[cfg(feature = "std")]
pub mod multistream {
    pub mod rate_alloc {
        pub use crate::multistream::rate_alloc::{
            MS_FRAME_TMP, cbr_last_stream_bitrate, cbr_max_data_bytes, clamp_user_bitrate,
            rate_allocation, smallest_packet, stream_max_bytes,
        };
    }
    pub mod surround {
        pub use crate::multistream::surround::SurroundProbe;
    }
}
pub mod pitch {
    pub use crate::pitch::{
        dual_inner_prod, inner_prod, pitch_downsample, pitch_search, pitch_xcorr,
        remove_doubling,
    };
}
pub mod pvq {
    pub use crate::pvq::{
        MAX_PVQ_K, alg_quant, alg_unquant, celt_pvq_u, celt_pvq_v, cwrsi, decode_pulses,
        encode_pulses, exp_rotation, icwrs, pvq_search,
    };
}
pub mod quant_bands {
    pub use crate::quant_bands::{
        E_PROB_MODEL, quant_coarse_energy, quant_coarse_energy_advanced, quant_energy_finalise,
        quant_fine_energy, unquant_coarse_energy,
    };
}
pub mod range_coder {
    pub use crate::range_coder::{BITRES, RANGE_BUF_MAX, RangeCoder};
}
pub mod rate {
    pub use crate::rate::{BITRES, bits2pulses, clt_compute_allocation, pulses2bits};
}
pub mod silk {
    pub use crate::silk::SilkResampler;
    pub mod ana_filt_bank_1 {
        pub use crate::silk::ana_filt_bank_1::silk_ana_filt_bank_1;
    }
    pub mod control_codec {
        pub use crate::silk::control_codec::*;
    }
    pub mod dec_api {
        pub use crate::silk::dec_api::SilkDecoder;
    }
    /// `silk_decode_core`: the OSCE tests decode SILK frames step by step
    /// to read the parameters that OSCE takes.
    pub mod decode_core {
        pub use crate::silk::decode_core::silk_decode_core;
    }
    pub mod decode_frame {
        pub use crate::silk::decode_frame::{
            FLAG_DECODE_LBRR, FLAG_DECODE_NORMAL, FLAG_PACKET_LOST,
        };
    }
    pub mod decode_indices {
        pub use crate::silk::decode_indices::{
            silk_decode_indices, silk_stereo_decode_mid_only, silk_stereo_decode_pred,
        };
    }
    /// `silk_decode_parameters`, for the OSCE tests (see `decode_core`).
    pub mod decode_parameters {
        pub use crate::silk::decode_parameters::silk_decode_parameters;
    }
    pub mod decode_pulses {
        pub use crate::silk::decode_pulses::silk_decode_pulses;
    }
    pub mod decoder_structs {
        pub use crate::silk::decoder_structs::SilkDecControl;
        /// The decoder state and control of one SILK channel, for the OSCE
        /// tests (see `decode_core`).
        pub use crate::silk::decoder_structs::{SilkDecoderControl, SilkDecoderState};
    }
    pub mod define {
        pub use crate::silk::define::*;
    }
    pub mod enc_api {
        pub use crate::silk::enc_api::{
            SilkControl, SilkEncoder, SilkInput, SilkPrefill, silk_average_first_mono_frame,
            silk_downmix_stereo, silk_encode_do_vad, silk_encode_frame, silk_encode_packet,
            silk_encode_prefill,
        };
    }
    pub mod encode_indices {
        pub use crate::silk::encode_indices::silk_encode_indices;
    }
    pub mod encode_pulses {
        pub use crate::silk::encode_pulses::silk_encode_pulses;
    }
    pub mod gain_quant {
        pub use crate::silk::gain_quant::silk_quant_ltp_gains;
    }
    pub mod hp_variable_cutoff {
        pub use crate::silk::hp_variable_cutoff::silk_hp_variable_cutoff;
    }
    pub mod init_decoder {
        pub use crate::silk::init_decoder::{silk_create_decoder, silk_decoder_set_fs};
    }
    pub mod init_encoder {
        pub use crate::silk::init_encoder::silk_init_encoder;
    }
    pub mod lin2log {
        pub use crate::silk::lin2log::silk_lin2log;
    }
    pub mod log2lin {
        pub use crate::silk::log2lin::silk_log2lin;
    }
    pub mod lpc_analysis {
        pub use crate::silk::lpc_analysis::{
            silk_burg_modified_fix, silk_find_lpc_fix, silk_lpc_inverse_pred_gain,
        };
    }
    pub mod ltp_analysis {
        pub use crate::silk::ltp_analysis::{silk_find_ltp_fix, silk_ltp_analysis_filter_fix};
    }
    pub mod macros {
        pub use crate::silk::macros::*;
    }
    pub mod nlsf {
        pub use crate::silk::nlsf::{
            silk_a2nlsf, silk_bwexpander_32, silk_nlsf2a, silk_nlsf_vq, silk_process_nlsfs,
        };
    }
    pub mod nlsf_decode {
        pub use crate::silk::nlsf_decode::silk_nlsf_decode;
    }
    pub mod nlsf_encode {
        pub use crate::silk::nlsf_encode::silk_nlsf_encode;
    }
    pub mod nsq {
        pub use crate::silk::nsq::silk_nsq;
    }
    pub mod nsq_del_dec {
        pub use crate::silk::nsq_del_dec::{
            NSQDelDecStruct, silk_noise_shape_quantizer_del_dec, silk_nsq_del_dec,
        };
    }
    pub mod pitch_analysis {
        pub use crate::silk::pitch_analysis::silk_pitch_analysis_core;
    }
    pub mod resampler {
        pub use crate::silk::resampler::{
            SILK_RESAMPLER_2_3_COEFS_LQ, SilkResampler, SilkResamplerDown1_6,
            silk_resampler_down2, silk_resampler_down2_3, silk_resampler_down_1_6,
            silk_resampler_private_ar2, silk_resampler_private_up2_hq,
        };
    }
    pub mod sigproc_fix {
        pub use crate::silk::sigproc_fix::*;
    }
    pub mod stereo_lr_to_ms {
        pub use crate::silk::stereo_lr_to_ms::silk_stereo_lr_to_ms;
    }
    pub mod structs {
        pub use crate::silk::structs::*;
    }
    pub mod tables {
        pub use crate::silk::tables::*;
    }
    pub mod tables_nlsf {
        pub use crate::silk::tables_nlsf::{SILK_NLSF_CB_NB_MB, SILK_NLSF_CB_WB};
    }
}

/// Test controls of `OpusEncoder`, not part of the API: which high-pass
/// filter feeds SILK with the `Voip` application at 48 kHz, and which
/// residual energies set SILK's quantization gains.
pub mod silk_input {
    /// With `true`, SILK reads the output of the integer high-pass filter
    /// `hp_cutoff` of libopus's fixed-point build at 48 kHz too, as it does
    /// at 8 to 24 kHz. Its SILK packets are then those of that build. The
    /// default, `false`, gives SILK the float filter's output at 48 kHz, as
    /// the float build does: the integer filter adds a constant of about -9
    /// to the 16-bit signal there. For tests that compare packets with the
    /// fixed-point build. The setting survives `reset_state`.
    pub fn set_integer_hp_at_48k(enc: &mut crate::OpusEncoder, on: bool) {
        enc.set_integer_hp_at_48k(on);
    }

    /// With `true`, SILK takes its quantization gains from libopus's
    /// residual energies alone, as libopus's fixed-point build does. These
    /// energies come from the signal scaled by the inverse subframe gains
    /// and rounded to 16 bits. The default, `false`, also computes them
    /// from the unscaled signal and takes the gain they give when it is
    /// less than half of libopus's: in a frame with a subframe near digital
    /// silence the rounding error inflates libopus's energies. For tests
    /// that compare packets with the fixed-point build. The setting
    /// survives `reset_state`.
    pub fn set_fixed_point_res_nrg(enc: &mut crate::OpusEncoder, on: bool) {
        enc.set_fixed_point_res_nrg(on);
    }
}
