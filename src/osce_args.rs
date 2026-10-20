//! What the SILK decoder needs to run the Opus Speech Coding Enhancement
//! (OSCE): the LACE and NoLACE enhancers and the bandwidth extension
//! (BWE).
//!
//! libopus passes the OSCE controls in `silk_DecControlStruct`
//! (`osce_method`, `osce_extended_mode`, `prev_osce_extended_mode`) and the
//! models in `silk_decoder` (`osce_model`). opus1 keeps the models in
//! `OpusDecoder` (`DecoderDnn`) and passes `Option<&OsceArgs>` to
//! `SilkDecoder`, built for every SILK frame from the decoder complexity
//! and the BWE switch (libopus `src/opus_decoder.c:443-466`). The OSCE
//! state of each channel lives in `SilkDecoderState`, as in libopus.
//!
//! `None` stands for a libopus `DecControl` without OSCE settings: method
//! NONE and no BWE. With the feature `osce` method NONE still clips
//! -32768 to -32767 in 20 ms SILK frames at 16 kHz, as every libopus
//! built with OSCE does. The public entry points of `SilkDecoder` pass
//! `None`.
//!
//! Without the feature `osce`, `OsceArgs` is a type without values, so
//! every argument of that type is `None` and SILK behaves as in a libopus
//! built without OSCE.

#[cfg(feature = "osce")]
use crate::dnn::osce::{OsceMethod, OsceModels};
#[cfg(feature = "osce-bwe")]
use crate::dnn::osce::OsceExtMode;

/// The OSCE controls of one SILK frame and the models they use.
#[cfg(feature = "osce")]
pub(crate) struct OsceArgs<'a> {
    /// The layers of LACE, NoLACE and BBWENet and the kernel variant.
    pub(crate) models: OsceModels<'a>,
    /// libopus `DecControl.osce_method`: NONE below complexity 6, LACE at
    /// 6, NoLACE from 7.
    pub(crate) method: OsceMethod,
    /// libopus `DecControl.osce_extended_mode`: `SilkBbwe` when the
    /// bandwidth extension replaces the SILK resampler for this frame,
    /// otherwise `SilkOnly` or `Hybrid`, the mode of the frame.
    #[cfg(feature = "osce-bwe")]
    pub(crate) extended_mode: OsceExtMode,
}

/// A type without values: builds without the feature `osce` have no OSCE
/// controls to pass.
#[cfg(not(feature = "osce"))]
pub(crate) enum NoOsce {}

/// Without the feature `osce`, `Option<&OsceArgs>` is always `None`.
#[cfg(not(feature = "osce"))]
pub(crate) type OsceArgs<'a> = NoOsce;

/// The models `silk_decode_frame` enhances with: `OsceModels` with the
/// feature `osce`, a type without values otherwise.
#[cfg(feature = "osce")]
pub(crate) type OsceFrameModels<'a> = OsceModels<'a>;

/// Without the feature `osce`, `Option<&OsceFrameModels>` is always
/// `None`.
#[cfg(not(feature = "osce"))]
pub(crate) type OsceFrameModels<'a> = NoOsce;
