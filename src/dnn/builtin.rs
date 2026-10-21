//! The libopus 1.6.1 weights embedded in the crate.
//!
//! `tools/dnn-gen/` writes one blob per model in the libopus format into
//! `dnn-weights/` at the repository root, with a `SHA256SUMS` file. This
//! module includes them with `include_bytes!` inside a 64-byte aligned
//! wrapper, so that every array of a blob is aligned and can be borrowed in
//! place. Each blob is included only by the feature that needs it:
//!
//! | Feature | Blobs | Bytes |
//! |---|---|---:|
//! | `deep-plc` | `pitchdnn.bin`, `plc.bin`, `fargan.bin` | 1,432,768 |
//! | `dred` | `dred_rdovae_enc.bin`, `dred_rdovae_dec.bin` | 565,440 |
//! | `osce` | `lace.bin`, `nolace.bin` | 1,573,440 |
//! | `osce-bwe` | `bbwenet.bin` | 685,888 |
//!
//! The blobs are parsed by the same code as a blob given at run time
//! (`WeightList::parse`), once per process: `weights()` parses them on its
//! first call, and `models()` builds the layers of every model once, so
//! that every decoder can share them.

use super::models::ModelSet;
use super::weights::WeightList;
use crate::compat::OnceCell;

/// The libopus release the weights come from.
pub const LIBOPUS_VERSION: &str = "1.6.1";

/// A byte array that starts on a 64-byte boundary, for `include_bytes!`:
///
/// ```text
/// static PITCHDNN: Aligned<{ include_bytes!("...").len() }> =
///     Aligned(*include_bytes!("..."));
/// ```
#[repr(C, align(64))]
pub struct Aligned<const N: usize>(pub [u8; N]);

/// Includes one blob of `dnn-weights/` in a 64-byte aligned static.
#[cfg(feature = "deep-plc")]
macro_rules! blob {
    ($name:ident, $file:literal) => {
        static $name: Aligned<{ include_bytes!(concat!("../../dnn-weights/", $file)).len() }> =
            Aligned(*include_bytes!(concat!("../../dnn-weights/", $file)));
    };
}

#[cfg(feature = "deep-plc")]
blob!(PITCHDNN, "pitchdnn.bin");
#[cfg(feature = "deep-plc")]
blob!(PLC, "plc.bin");
#[cfg(feature = "deep-plc")]
blob!(FARGAN, "fargan.bin");
#[cfg(feature = "dred")]
blob!(DRED_RDOVAE_ENC, "dred_rdovae_enc.bin");
#[cfg(feature = "dred")]
blob!(DRED_RDOVAE_DEC, "dred_rdovae_dec.bin");
#[cfg(feature = "osce")]
blob!(LACE, "lace.bin");
#[cfg(feature = "osce")]
blob!(NOLACE, "nolace.bin");
#[cfg(feature = "osce-bwe")]
blob!(BBWENET, "bbwenet.bin");

/// The PitchDNN blob (`pitchdnn.bin`).
#[cfg(feature = "deep-plc")]
pub fn pitchdnn() -> &'static [u8] {
    &PITCHDNN.0
}

/// The PLC model blob (`plc.bin`).
#[cfg(feature = "deep-plc")]
pub fn plc() -> &'static [u8] {
    &PLC.0
}

/// The FARGAN blob (`fargan.bin`).
#[cfg(feature = "deep-plc")]
pub fn fargan() -> &'static [u8] {
    &FARGAN.0
}

/// The blob of DRED's RDOVAE encoder (`dred_rdovae_enc.bin`).
#[cfg(feature = "dred")]
pub fn dred_rdovae_enc() -> &'static [u8] {
    &DRED_RDOVAE_ENC.0
}

/// The blob of DRED's RDOVAE decoder (`dred_rdovae_dec.bin`).
#[cfg(feature = "dred")]
pub fn dred_rdovae_dec() -> &'static [u8] {
    &DRED_RDOVAE_DEC.0
}

/// The LACE blob (`lace.bin`).
#[cfg(feature = "osce")]
pub fn lace() -> &'static [u8] {
    &LACE.0
}

/// The NoLACE blob (`nolace.bin`).
#[cfg(feature = "osce")]
pub fn nolace() -> &'static [u8] {
    &NOLACE.0
}

/// The BBWENet blob (`bbwenet.bin`).
#[cfg(feature = "osce-bwe")]
pub fn bbwenet() -> &'static [u8] {
    &BBWENET.0
}

/// Every built-in blob of this build, with its file name, in the order of
/// libopus's blob writer (PitchDNN, FARGAN, the PLC model, the RDOVAE
/// encoder and decoder, LACE, NoLACE), then BBWENet. Empty in a build with only `dnn`.
pub fn blobs() -> &'static [(&'static str, &'static [u8])] {
    static BLOBS: &[(&str, &[u8])] = &[
        #[cfg(feature = "deep-plc")]
        ("pitchdnn.bin", &PITCHDNN.0),
        #[cfg(feature = "deep-plc")]
        ("fargan.bin", &FARGAN.0),
        #[cfg(feature = "deep-plc")]
        ("plc.bin", &PLC.0),
        #[cfg(feature = "dred")]
        ("dred_rdovae_enc.bin", &DRED_RDOVAE_ENC.0),
        #[cfg(feature = "dred")]
        ("dred_rdovae_dec.bin", &DRED_RDOVAE_DEC.0),
        #[cfg(feature = "osce")]
        ("lace.bin", &LACE.0),
        #[cfg(feature = "osce")]
        ("nolace.bin", &NOLACE.0),
        #[cfg(feature = "osce-bwe")]
        ("bbwenet.bin", &BBWENET.0),
    ];
    BLOBS
}

/// The built-in blobs as one weight list, parsed on the first call.
///
/// The tests check that every built-in blob parses. Should one ever fail,
/// the list is empty and no model loads, instead of a panic inside the
/// lazy initializer.
pub fn weights() -> &'static WeightList<'static> {
    static LIST: OnceCell<WeightList<'static>> = OnceCell::new();
    LIST.get(|| {
        let mut list = WeightList::empty();
        for &(_, blob) in blobs() {
            if list.push(blob).is_err() {
                return WeightList::empty();
            }
        }
        list
    })
}

/// The layers of every built-in model, built on the first call from
/// `weights()` and shared by every caller.
///
/// The first call also reads one byte of every 4 KiB page of the blobs,
/// so that the first use of a model (often in a lost packet, inside an
/// audio callback) does not take page faults.
pub fn models() -> &'static ModelSet<'static> {
    static MODELS: OnceCell<ModelSet<'static>> = OnceCell::new();
    MODELS.get(|| {
        let mut set = ModelSet::new();
        // A group that fails stays unloaded; the tests check that all load.
        let _ = set.load(weights());
        for &(_, blob) in blobs() {
            super::holder::prefault(blob);
        }
        set
    })
}

/// The layers of the built-in RDOVAE decoder of DRED, built on the first
/// call from `weights()` and shared by every `OpusDredDecoder`. `None` if
/// they fail to load (the tests check that they load).
///
/// The first call also reads one byte of every 4 KiB page of its blob, as
/// `models()` does for the decoder's models.
#[cfg(feature = "dred")]
pub fn rdovae_dec() -> Option<&'static super::models::dred_rdovae_dec::RdovaeDec<'static>> {
    static MODEL: OnceCell<Option<super::models::dred_rdovae_dec::RdovaeDec<'static>>> = OnceCell::new();
    MODEL
        .get(|| {
            let model = super::models::dred_rdovae_dec::RdovaeDec::init(weights()).ok();
            super::holder::prefault(dred_rdovae_dec());
            model
        })
        .as_ref()
}

/// The models of the DRED encoder (the RDOVAE encoder and PitchDNN), built
/// on the first call from `weights()` and shared by every Opus encoder.
/// `None` if they fail to load (the tests check that they load).
///
/// The first call also reads one byte of every 4 KiB page of the RDOVAE
/// encoder's blob and of PitchDNN's, as `models()` does for the decoder's
/// models.
#[cfg(feature = "dred")]
pub fn dred_enc() -> Option<&'static super::models::DredEncModels<'static>> {
    static MODEL: OnceCell<Option<super::models::DredEncModels<'static>>> = OnceCell::new();
    MODEL
        .get(|| {
            let model = super::models::DredEncModels::init(weights()).ok();
            super::holder::prefault(dred_rdovae_enc());
            super::holder::prefault(pitchdnn());
            model
        })
        .as_ref()
}
