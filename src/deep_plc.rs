//! What the SILK and CELT decoders need to run deep packet loss
//! concealment (deep PLC).
//!
//! libopus passes its deep PLC state (`LPCNetPLCState *lpcnet`) from
//! `opus_decode_frame` down to `silk_Decode`, `silk_decode_frame`,
//! `silk_PLC` and `celt_decode_with_ec_dred`. Only the mid channel of SILK
//! and the main CELT call of a frame get it; the CELT redundancy frames
//! and the CELT fade-out frame get `NULL`. opus1 passes
//! `Option<&mut DeepPlcArgs>` along the same calls, and `None` where
//! libopus passes `NULL`.
//!
//! With the cargo feature `deep-plc`, `DeepPlcArgs` holds the state of one
//! `OpusDecoder`, its models and its kernel variant. Without the feature
//! it is a type without values, so every argument of that type is `None`
//! and the decoders behave as a libopus built without deep PLC.

#[cfg(feature = "deep-plc")]
use crate::dnn::{DnnArch, models::DeepPlcModels, plc::DeepPlc};

/// The deep PLC state of one decoder and what it runs on.
#[cfg(feature = "deep-plc")]
pub(crate) struct DeepPlcArgs<'a> {
    /// The state (libopus `st->lpcnet`).
    pub(crate) state: &'a mut DeepPlc,
    /// The models, or `None` when they are not loaded (libopus
    /// `lpcnet->loaded == 0`). The history is then still updated, as in
    /// libopus, but nothing is concealed with it.
    pub(crate) models: Option<&'a DeepPlcModels<'a>>,
    /// The kernel variant (libopus `arch`).
    pub(crate) arch: DnnArch,
}

/// A type without values: builds without the feature `deep-plc` have no
/// deep PLC state to pass.
#[cfg(not(feature = "deep-plc"))]
pub(crate) enum NoDeepPlc {}

/// Without the feature `deep-plc`, `Option<&mut DeepPlcArgs>` is always
/// `None`.
#[cfg(not(feature = "deep-plc"))]
pub(crate) type DeepPlcArgs<'a> = NoDeepPlc;
