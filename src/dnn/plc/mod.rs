//! Deep packet loss concealment (deep PLC) of libopus 1.6.1, compiled with
//! the `deep-plc` feature.
//!
//! When a packet is lost, the decoder of libopus 1.5 and later can make up
//! the missing speech with three neural networks, at decoder complexity 5
//! and above:
//!
//! - PitchDNN estimates the pitch of each 10 ms of past audio
//!   (`pitchdnn`, libopus `dnn/pitchdnn.c`);
//! - the feature analysis describes each 10 ms of past audio with 20 values
//!   (`features`, libopus `dnn/lpcnet_enc.c`);
//! - the PLC network predicts the 20 values of the lost frames from the
//!   past ones (`compute_plc_pred` in `lpcnet_plc`);
//! - FARGAN, a vocoder, turns those values into 16 kHz audio (`fargan`,
//!   libopus `dnn/fargan.c`).
//!
//! `DeepPlc` (`lpcnet_plc`, libopus `dnn/lpcnet_plc.c`) holds the state of
//! all of them for one decoder: 150 ms of past audio at 16 kHz, the
//! positions of the analysis, the recurrent states and the queue of
//! features that the deep redundancy (DRED) decoder would fill. It works in
//! frames of 10 ms at 16 kHz (`LPCNET_FRAME_SIZE`, 160 samples):
//!
//! - `DeepPlc::update` appends a frame that was decoded;
//! - `DeepPlc::conceal` makes up a lost frame and appends it;
//! - `DeepPlc::update_from_celt` appends the last 40 ms of a CELT decoder's
//!   history, resampled from 48 kHz to 16 kHz (libopus `update_plc_state`).
//!
//! The weights are not part of the state: `conceal` takes the models
//! (`DeepPlcModels`, from `DecoderDnn::deep_plc`) and the kernel variant
//! (`DnnArch`) as arguments.
//!
//! # Rounding
//!
//! Every function gives the bits of libopus 1.6.1 built for 64-bit ARM
//! without loop vectorization (`-O2 -fno-vectorize -fno-slp-vectorize`), on
//! every target. The rules are those of `freq.rs`: a C expression
//! `a * b + c` that clang contracts is one fused multiply-add; a `double`
//! constant or a `double` math function makes the expression `double`;
//! sums keep the order of the C loops. Each place was checked in the
//! assembly of that build.
//!
//! The math library functions (`exp`, `exp2`, `log`, `sqrt`, `floor`) are
//! those of the platform with `std`, as in libopus, and those of the `libm`
//! crate without `std`. Each result is rounded to `f32` or to an integer
//! right after, so a last-bit difference between libraries shows only in
//! rare cases.

pub mod fargan;
pub mod features;
pub mod lpcnet_plc;
pub mod pitchdnn;

pub use fargan::FarganState;
pub use features::LpcnetEncState;
pub use lpcnet_plc::{DeepPlc, PlcNetState, compute_plc_pred};
pub use pitchdnn::PitchDnnState;

/// `NB_FEATURES` (`dnn/lpcnet.h:32`): the features of one 10 ms frame that
/// the PLC network predicts and FARGAN reads: 18 cepstral values, the pitch
/// and the pitch correlation.
pub const NB_FEATURES: usize = 20;

/// `NB_TOTAL_FEATURES` (`dnn/lpcnet.h:33`): `NB_FEATURES` followed by the
/// 16 LPC coefficients.
pub const NB_TOTAL_FEATURES: usize = 36;

/// `LPCNET_FRAME_SIZE` (`dnn/lpcnet.h:36`): 10 ms at 16 kHz, the unit of
/// `DeepPlc::update` and `DeepPlc::conceal`.
pub const LPCNET_FRAME_SIZE: usize = 160;

/// `PITCH_MIN_PERIOD` (`dnn/pitchdnn.h:10`).
pub const PITCH_MIN_PERIOD: usize = 32;

/// `PITCH_MAX_PERIOD` (`dnn/pitchdnn.h:11`).
pub const PITCH_MAX_PERIOD: usize = 256;

/// `NB_XCORR_FEATURES` (`dnn/pitchdnn.h:14`): the cross-correlation inputs
/// of PitchDNN, one per lag from 32 to 255.
pub const NB_XCORR_FEATURES: usize = PITCH_MAX_PERIOD - PITCH_MIN_PERIOD;

/// `PITCH_IF_MAX_FREQ` (`dnn/lpcnet_private.h:18`): the bins of the
/// instantaneous frequency inputs of PitchDNN.
pub const PITCH_IF_MAX_FREQ: usize = 30;

/// `PITCH_IF_FEATURES` (`dnn/lpcnet_private.h:19`): the instantaneous
/// frequency inputs of PitchDNN.
pub const PITCH_IF_FEATURES: usize = 3 * PITCH_IF_MAX_FREQ - 2;

/// `CONT_VECTORS` (`dnn/lpcnet_private.h:21`): the feature vectors that
/// FARGAN reads when it starts on past audio.
pub const CONT_VECTORS: usize = 5;

/// `PLC_MAX_FEC` (`dnn/lpcnet_private.h:15`): the room of the queue of
/// features that the DRED decoder fills.
pub const PLC_MAX_FEC: usize = 104;

/// `PLC_BUF_SIZE` (`dnn/lpcnet_private.h:49`): the past audio that
/// `DeepPlc` keeps, 150 ms at 16 kHz.
pub const PLC_BUF_SIZE: usize = (CONT_VECTORS + 10) * LPCNET_FRAME_SIZE;

/// The input of the PLC network (`2*NB_BANDS+NB_FEATURES+1` in
/// `dnn/lpcnet_plc.c`): 36 Burg cepstral values, 20 features and a flag
/// (1 for analysed audio, -1 for DRED features, 0 for a pure prediction).
pub const PLC_INPUT_SIZE: usize = 2 * crate::dnn::freq::NB_BANDS + NB_FEATURES + 1;

/// `MIN16` and `MIN32` of the float build: `a < b ? a : b`, in this
/// argument order (it decides the result when one value is NaN).
#[inline(always)]
pub(crate) fn min_c(a: f32, b: f32) -> f32 {
    if a < b { a } else { b }
}

/// `MAX16` and `MAX32` of the float build: `a > b ? a : b`, in this
/// argument order.
#[inline(always)]
pub(crate) fn max_c(a: f32, b: f32) -> f32 {
    if a > b { a } else { b }
}

/// `(int)floor(.5+256./pow(2.f,((1./60.)*((f+1.5)*60))))`: the pitch
/// period in samples of the pitch feature `f` (`dnn/lpcnet_enc.c:188`,
/// `dnn/fargan.c:154`, `:205`). All in `double`; clang turns `pow(2, x)`
/// into `exp2(x)`. A NaN gives 0 and values out of range saturate, as the
/// `fcvtms` instruction does.
#[inline]
pub(crate) fn period_from_pitch(f: f32) -> i32 {
    #[cfg(not(feature = "std"))]
    use crate::compat::Math;
    let x = (1.0f64 / 60.0) * ((f as f64 + 1.5) * 60.0);
    (0.5 + 256.0 / x.exp2()).floor() as i32
}
