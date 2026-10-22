//! The weights and the models that a decoder holds: `DnnBlob` (a validated
//! weight blob), `DnnModels` (which model groups are loaded) and
//! `DecoderDnn` (what an `OpusDecoder` stores for its neural network
//! features).
//!
//! libopus loads its built-in models into every decoder (`lpcnet_plc_init`,
//! `silk_InitDecoder`), or, when built with `USE_WEIGHTS_FILE`, from a blob
//! given with `OPUS_SET_DNN_BLOB`. opus1 parses the built-in weights with
//! the same code as a blob given at run time (`plan.md` section 3), and
//! offers `set_dnn_blob` in every build with a neural network feature.
//!
//! Where the layers live:
//!
//! - The built-in models are built once per process (`builtin::models()`).
//!   Every decoder points at them, so they cost no memory per decoder.
//! - A model loaded from a blob is built for the decoder that loads it.
//!   With `heap` its layers are in a `Box` and the decoder keeps a handle
//!   to the blob, whose bytes the layers borrow. Without `heap` only blobs
//!   that live for the whole program exist, and the layers are stored in
//!   the decoder itself.

use super::arch::{self, DnnArch};
use super::weights::WeightList;
use crate::in_place::write_fields;
#[cfg(feature = "deep-plc")]
use super::{builtin, models};

#[cfg(feature = "heap")]
use std::boxed::Box;
#[cfg(feature = "heap")]
use std::sync::Arc;

use super::models::ERR_NO_MODEL;

/// A validated weight blob in the libopus format (`OPUS_SET_DNN_BLOB`).
/// Cloning is cheap: the bytes are not copied.
///
/// A blob may hold any set of models, in any order, like the file that
/// libopus's `dump_weights_blob` writes. Every array must start 4-byte
/// aligned in memory, so that the layers can use the bytes in place.
#[derive(Clone)]
pub struct DnnBlob {
    inner: Inner,
}

#[derive(Clone)]
enum Inner {
    /// The weights built into this build (`builtin::weights()`).
    #[cfg(feature = "deep-plc")]
    Builtin,
    /// A blob that lives for the whole program, already parsed.
    Static(WeightList<'static>),
    /// A copy in 64-byte aligned heap memory, shared by every clone.
    #[cfg(feature = "heap")]
    Shared(Arc<SharedBlob>),
}

/// The heap copy behind `DnnBlob::from_bytes`.
#[cfg(feature = "heap")]
struct SharedBlob {
    /// The parsed list. Its slices point into the allocation of `bytes`;
    /// they are typed `'static`, which is sound because that allocation is
    /// never changed or freed while this struct exists (see
    /// `DnnBlob::from_bytes`). The list is never handed out with that
    /// lifetime.
    list: WeightList<'static>,
    /// The blob, in blocks of 64 bytes so that it starts 64-byte aligned.
    /// It is owned through a raw pointer (from `Box::into_raw`) and freed
    /// in `Drop`. A `Box` field would not do: moving a `Box` asserts that
    /// it is the only pointer to its memory, which the slices of `list`
    /// contradict (Rust's aliasing rules, as Miri checks them).
    bytes: *mut [Block],
}

// SAFETY: `SharedBlob` owns its allocation like a `Box<[Block]>` and never
// writes to it after `from_bytes`, so sharing it or moving it to another
// thread is as safe as for `Box<[Block]>` and `WeightList<'static>`.
#[cfg(feature = "heap")]
unsafe impl Send for SharedBlob {}
// SAFETY: see `Send`; every access after construction is a read.
#[cfg(feature = "heap")]
unsafe impl Sync for SharedBlob {}

#[cfg(feature = "heap")]
impl Drop for SharedBlob {
    fn drop(&mut self) {
        // SAFETY: `bytes` comes from `Box::into_raw` in `from_bytes` and is
        // freed only here, once. `list` borrows it, but it is dropped with
        // this struct and never used again.
        drop(unsafe { Box::from_raw(self.bytes) });
    }
}

/// 64 bytes on a 64-byte boundary.
#[cfg(feature = "heap")]
#[derive(Clone, Copy)]
#[repr(C, align(64))]
struct Block([u8; 64]);

impl DnnBlob {
    /// A blob that lives for the whole program, for example from
    /// `include_bytes!` in a 64-byte aligned wrapper. Fails with
    /// `"DNN blob is not aligned"` when an array does not start 4-byte
    /// aligned, and with `"Invalid DNN blob"` when a record is malformed.
    pub fn from_static(bytes: &'static [u8]) -> Result<DnnBlob, &'static str> {
        Ok(DnnBlob { inner: Inner::Static(WeightList::parse(bytes)?) })
    }

    /// Copies `bytes` into 64-byte aligned memory and checks it as
    /// `from_static` does. The copy is shared by the clones of the blob and
    /// by every decoder that loads it.
    #[cfg(feature = "heap")]
    pub fn from_bytes(bytes: &[u8]) -> Result<DnnBlob, &'static str> {
        let blocks = std::vec![Block([0; 64]); bytes.len().div_ceil(64)].into_boxed_slice();
        // From here on the allocation is owned through this raw pointer
        // only. `SharedBlob::drop` frees it, and so does the failure path
        // below.
        let raw: *mut [Block] = Box::into_raw(blocks);
        // SAFETY: `raw` holds `64 · len >= bytes.len()` bytes in one live
        // allocation, and `Block` is plain bytes without padding.
        let copy: &mut [u8] = unsafe { core::slice::from_raw_parts_mut(raw.cast::<u8>(), bytes.len()) };
        copy.copy_from_slice(bytes);
        // SAFETY: the allocation stays where it is and is never written
        // again; it is freed only when the `SharedBlob` that owns it is
        // dropped (or right below, if the blob is refused, after the last
        // use of `view`). The `'static` list is only reachable through
        // `DnnBlob` methods that tie it to a borrow of the blob, or through
        // `DecoderDnn`, which keeps a clone of the blob (and so the
        // allocation) next to every layer it builds from it.
        let view: &'static [u8] = unsafe { core::slice::from_raw_parts(raw.cast::<u8>(), bytes.len()) };
        let list = match WeightList::parse(view) {
            Ok(list) => list,
            Err(e) => {
                // SAFETY: `raw` comes from `Box::into_raw` above and nothing
                // refers to it any more.
                drop(unsafe { Box::from_raw(raw) });
                return Err(e);
            }
        };
        Ok(DnnBlob { inner: Inner::Shared(Arc::new(SharedBlob { list, bytes: raw })) })
    }

    /// The libopus 1.6.1 weights of every model this build contains
    /// (`crate::dnn::builtin`). Loading it into a decoder brings back the
    /// built-in models after another blob replaced them.
    #[cfg(feature = "deep-plc")]
    pub fn builtin() -> DnnBlob {
        DnnBlob { inner: Inner::Builtin }
    }

    /// The parsed arrays of the blob.
    ///
    /// Hidden from the documentation: `WeightList` is a type of the
    /// internal `dnn` module, which applications cannot name (tests reach
    /// it through `opus1::internals::dnn`).
    #[doc(hidden)]
    pub fn weights(&self) -> &WeightList<'_> {
        self.list()
    }

    /// The list with the lifetime of its bytes. Private: for a heap blob
    /// that lifetime is only valid while a clone of this blob is kept.
    fn list(&self) -> &WeightList<'static> {
        match &self.inner {
            #[cfg(feature = "deep-plc")]
            Inner::Builtin => builtin::weights(),
            Inner::Static(list) => list,
            #[cfg(feature = "heap")]
            Inner::Shared(shared) => &shared.list,
        }
    }

    /// Reads one byte of every 4 KiB page, so that the first use of the
    /// weights (often in a lost packet, inside an audio callback) does not
    /// take page faults.
    pub fn prefault(&self) {
        for blob in self.list().blobs() {
            prefault(blob);
        }
    }
}

/// Reads one byte of every 4 KiB page of `bytes`.
pub(crate) fn prefault(bytes: &[u8]) {
    for i in (0..bytes.len()).step_by(4096) {
        // SAFETY: `i < bytes.len()`, so the pointer is in bounds and the
        // byte is initialized. The volatile read keeps the compiler from
        // removing it.
        unsafe { core::ptr::read_volatile(bytes.as_ptr().add(i)) };
    }
}

impl core::fmt::Debug for DnnBlob {
    /// The kind of blob and the size of its arrays, without the data.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let kind = match &self.inner {
            #[cfg(feature = "deep-plc")]
            Inner::Builtin => "built-in",
            Inner::Static(_) => "static",
            #[cfg(feature = "heap")]
            Inner::Shared(_) => "copied",
        };
        f.debug_struct("DnnBlob").field("kind", &kind).field("weights", self.list()).finish()
    }
}

/// A set of model groups, as returned by `OpusDecoder::set_dnn_blob`.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct DnnModels(u8);

impl DnnModels {
    /// No model.
    pub const NONE: DnnModels = DnnModels(0);
    /// Deep PLC: PitchDNN, the PLC model and FARGAN, which load together.
    pub const DEEP_PLC: DnnModels = DnnModels(1 << 0);
    /// LACE.
    pub const LACE: DnnModels = DnnModels(1 << 1);
    /// NoLACE.
    pub const NOLACE: DnnModels = DnnModels(1 << 2);
    /// BBWENet, the network of the bandwidth extension.
    pub const BBWENET: DnnModels = DnnModels(1 << 3);
    // Bits 4 and 5 are kept for the DRED encoder and decoder of a later
    // `dred` feature.

    /// The bits of the set.
    pub const fn bits(self) -> u8 {
        self.0
    }

    /// Whether every group of `other` is in the set.
    pub const fn contains(self, other: DnnModels) -> bool {
        self.0 & other.0 == other.0
    }

    /// The groups of both sets.
    pub const fn union(self, other: DnnModels) -> DnnModels {
        DnnModels(self.0 | other.0)
    }

    /// Whether the set is empty.
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }
}

/// Where the layers of one model group come from.
#[cfg(feature = "deep-plc")]
enum Slot<T: 'static> {
    /// Not loaded.
    Empty,
    /// The built-in model, shared by every decoder.
    Builtin(&'static T),
    /// A model built from a blob, with a clone of the blob, which keeps the
    /// bytes its layers borrow alive.
    #[cfg(feature = "heap")]
    Loaded { model: Box<T>, _blob: DnnBlob },
    /// A model built from a blob that lives for the whole program.
    #[cfg(not(feature = "heap"))]
    Loaded(T),
}

#[cfg(feature = "deep-plc")]
impl<T: 'static> Slot<T> {
    fn builtin(model: Option<&'static T>) -> Self {
        model.map_or(Slot::Empty, Slot::Builtin)
    }

    #[cfg(feature = "heap")]
    fn loaded(model: T, blob: &DnnBlob) -> Self {
        Slot::Loaded { model: Box::new(model), _blob: blob.clone() }
    }

    #[cfg(not(feature = "heap"))]
    fn loaded(model: T, _blob: &DnnBlob) -> Self {
        Slot::Loaded(model)
    }

    /// The model, borrowed for as long as the slot is not changed. The
    /// layers come out with that shorter lifetime, so none of them can be
    /// kept after the blob they borrow from is released.
    fn get(&self) -> Option<&T> {
        match self {
            Slot::Empty => None,
            Slot::Builtin(m) => Some(m),
            #[cfg(feature = "heap")]
            Slot::Loaded { model, .. } => Some(model),
            #[cfg(not(feature = "heap"))]
            Slot::Loaded(m) => Some(m),
        }
    }
}

/// What a decoder stores for its neural network features: the kernel
/// variant, chosen once at creation as libopus does (`opus_select_arch`),
/// and the layers of every loaded model group.
///
/// `OpusDecoder::reset_state` keeps all of it: unlike libopus, a reset does
/// not reload PitchDNN (`nnet.md`, Review item 4).
pub struct DecoderDnn {
    arch: DnnArch,
    #[cfg(feature = "deep-plc")]
    deep_plc: Slot<models::DeepPlcModels<'static>>,
    #[cfg(feature = "osce")]
    lace: Slot<models::lace::LaceLayers<'static>>,
    #[cfg(feature = "osce")]
    nolace: Slot<models::nolace::NolaceLayers<'static>>,
    #[cfg(feature = "osce-bwe")]
    bbwenet: Slot<models::bbwenet::BbwenetLayers<'static>>,
}

impl DecoderDnn {
    /// A holder with the detected kernel variant and the built-in models of
    /// this build (none in a build with only `dnn`), as a new libopus
    /// decoder has.
    pub fn new() -> DecoderDnn {
        let mut d = core::mem::MaybeUninit::<DecoderDnn>::uninit();
        // SAFETY: `write_new` initializes every field of `d`.
        unsafe {
            DecoderDnn::write_new(d.as_mut_ptr());
            d.assume_init()
        }
    }

    /// Writes the holder of [`new`](Self::new) to `this`, which may be
    /// uninitialized memory, without building it on the stack first (see
    /// `crate::in_place`). Without `heap` a holder has room for models
    /// loaded from a blob, 12 KB with every feature.
    ///
    /// # Safety
    /// `this` must be valid for writes of `Self`.
    pub(crate) unsafe fn write_new(this: *mut DecoderDnn) {
        #[cfg(feature = "deep-plc")]
        let set = builtin::models();
        // SAFETY: the caller guarantees that `this` is valid for writes.
        unsafe {
            write_fields!(this, {
                arch: arch::detect(),
                #[cfg(feature = "deep-plc")]
                deep_plc: Slot::builtin(set.deep_plc.as_ref()),
                #[cfg(feature = "osce")]
                lace: Slot::builtin(set.lace.as_ref()),
                #[cfg(feature = "osce")]
                nolace: Slot::builtin(set.nolace.as_ref()),
                #[cfg(feature = "osce-bwe")]
                bbwenet: Slot::builtin(set.bbwenet.as_ref()),
            });
        }
    }

    /// The kernel variant the decoder uses.
    pub fn arch(&self) -> DnnArch {
        self.arch
    }

    /// The model groups that are loaded.
    pub fn models(&self) -> DnnModels {
        #[allow(unused_mut)]
        let mut m = DnnModels::NONE;
        #[cfg(feature = "deep-plc")]
        if self.deep_plc.get().is_some() {
            m = m.union(DnnModels::DEEP_PLC);
        }
        #[cfg(feature = "osce")]
        if self.lace.get().is_some() {
            m = m.union(DnnModels::LACE);
        }
        #[cfg(feature = "osce")]
        if self.nolace.get().is_some() {
            m = m.union(DnnModels::NOLACE);
        }
        #[cfg(feature = "osce-bwe")]
        if self.bbwenet.get().is_some() {
            m = m.union(DnnModels::BBWENET);
        }
        m
    }

    /// Loads every model group of this build that `blob` holds completely.
    /// Each group is replaced only when all of its arrays load; the others
    /// keep what they had. Returns the groups loaded now, or `"DNN blob
    /// holds no model of this build"` (and nothing changes).
    ///
    /// libopus differs (`nnet.md` sections 4.3 and 4.4): it returns 1 on
    /// any failure, can leave a model half-replaced, and a blob without
    /// BBWENet also leaves LACE and NoLACE unloaded in a BWE build.
    pub fn load(&mut self, blob: &DnnBlob) -> Result<DnnModels, &'static str> {
        #[allow(unused_mut)]
        let mut loaded = DnnModels::NONE;
        #[cfg(feature = "deep-plc")]
        {
            if let Inner::Builtin = blob.inner {
                let set = builtin::models();
                if let Some(m) = set.deep_plc.as_ref() {
                    self.deep_plc = Slot::Builtin(m);
                    loaded = loaded.union(DnnModels::DEEP_PLC);
                }
                #[cfg(feature = "osce")]
                if let Some(m) = set.lace.as_ref() {
                    self.lace = Slot::Builtin(m);
                    loaded = loaded.union(DnnModels::LACE);
                }
                #[cfg(feature = "osce")]
                if let Some(m) = set.nolace.as_ref() {
                    self.nolace = Slot::Builtin(m);
                    loaded = loaded.union(DnnModels::NOLACE);
                }
                #[cfg(feature = "osce-bwe")]
                if let Some(m) = set.bbwenet.as_ref() {
                    self.bbwenet = Slot::Builtin(m);
                    loaded = loaded.union(DnnModels::BBWENET);
                }
            } else {
                let list = blob.list();
                if let Ok(m) = models::DeepPlcModels::init(list) {
                    self.deep_plc = Slot::loaded(m, blob);
                    loaded = loaded.union(DnnModels::DEEP_PLC);
                }
                #[cfg(feature = "osce")]
                if let Ok(m) = models::lace::LaceLayers::init(list) {
                    self.lace = Slot::loaded(m, blob);
                    loaded = loaded.union(DnnModels::LACE);
                }
                #[cfg(feature = "osce")]
                if let Ok(m) = models::nolace::NolaceLayers::init(list) {
                    self.nolace = Slot::loaded(m, blob);
                    loaded = loaded.union(DnnModels::NOLACE);
                }
                #[cfg(feature = "osce-bwe")]
                if let Ok(m) = models::bbwenet::BbwenetLayers::init(list) {
                    self.bbwenet = Slot::loaded(m, blob);
                    loaded = loaded.union(DnnModels::BBWENET);
                }
            }
        }
        if loaded.is_empty() {
            return Err(ERR_NO_MODEL);
        }
        blob.prefault();
        Ok(loaded)
    }

    /// The deep PLC models, when loaded.
    #[cfg(feature = "deep-plc")]
    pub fn deep_plc(&self) -> Option<&models::DeepPlcModels<'_>> {
        self.deep_plc.get()
    }

    /// LACE, when loaded.
    #[cfg(feature = "osce")]
    pub fn lace(&self) -> Option<&models::lace::LaceLayers<'_>> {
        self.lace.get()
    }

    /// NoLACE, when loaded.
    #[cfg(feature = "osce")]
    pub fn nolace(&self) -> Option<&models::nolace::NolaceLayers<'_>> {
        self.nolace.get()
    }

    /// BBWENet, when loaded.
    #[cfg(feature = "osce-bwe")]
    pub fn bbwenet(&self) -> Option<&models::bbwenet::BbwenetLayers<'_>> {
        self.bbwenet.get()
    }
}

impl Default for DecoderDnn {
    fn default() -> DecoderDnn {
        DecoderDnn::new()
    }
}

/// What an `OpusDredDecoder` stores: the kernel variant and the layers of
/// the RDOVAE decoder (libopus: `model`, `loaded` and `arch` of
/// `OpusDREDDecoder`).
#[cfg(feature = "dred")]
pub(crate) struct DredDecoderDnn {
    arch: DnnArch,
    model: Slot<models::dred_rdovae_dec::RdovaeDec<'static>>,
}

#[cfg(feature = "dred")]
impl DredDecoderDnn {
    /// The detected kernel variant and the built-in model.
    pub(crate) fn new() -> Self {
        DredDecoderDnn { arch: arch::detect(), model: Slot::builtin(builtin::rdovae_dec()) }
    }

    /// Loads the RDOVAE decoder from `blob`. When the blob lacks it, or one
    /// of its arrays does not load, returns `"DNN blob holds no model of
    /// this build"` and keeps the model it had.
    pub(crate) fn load(&mut self, blob: &DnnBlob) -> Result<(), &'static str> {
        if let Inner::Builtin = blob.inner {
            self.model = Slot::Builtin(builtin::rdovae_dec().ok_or(ERR_NO_MODEL)?);
        } else {
            let model = models::dred_rdovae_dec::RdovaeDec::init(blob.list()).map_err(|_| ERR_NO_MODEL)?;
            self.model = Slot::loaded(model, blob);
        }
        blob.prefault();
        Ok(())
    }

    /// The model, when loaded.
    pub(crate) fn model(&self) -> Option<&models::dred_rdovae_dec::RdovaeDec<'_>> {
        self.model.get()
    }

    /// The kernel variant.
    pub(crate) fn arch(&self) -> DnnArch {
        self.arch
    }
}

/// What the DRED part of an Opus encoder stores: the kernel variant and
/// the models of the DRED encoder (libopus: `model`, `loaded` and the
/// PitchDNN model of `lpcnet_enc_state` in `DREDEnc`, and the encoder's
/// `arch`).
#[cfg(feature = "dred")]
pub(crate) struct DredEncoderDnn {
    arch: DnnArch,
    model: Slot<models::DredEncModels<'static>>,
}

#[cfg(feature = "dred")]
impl DredEncoderDnn {
    /// The detected kernel variant and the built-in models.
    pub(crate) fn new() -> Self {
        DredEncoderDnn { arch: arch::detect(), model: Slot::builtin(builtin::dred_enc()) }
    }

    /// Loads the RDOVAE encoder and PitchDNN from `blob`. When the blob
    /// lacks one of them, or one of their arrays does not load, returns
    /// `"DNN blob holds no model of this build"` and keeps the models it
    /// had.
    pub(crate) fn load(&mut self, blob: &DnnBlob) -> Result<(), &'static str> {
        if let Inner::Builtin = blob.inner {
            self.model = Slot::Builtin(builtin::dred_enc().ok_or(ERR_NO_MODEL)?);
        } else {
            let model = models::DredEncModels::init(blob.list()).map_err(|_| ERR_NO_MODEL)?;
            self.model = Slot::loaded(model, blob);
        }
        blob.prefault();
        Ok(())
    }

    /// The models, when loaded.
    pub(crate) fn model(&self) -> Option<&models::DredEncModels<'_>> {
        self.model.get()
    }

    /// The kernel variant.
    pub(crate) fn arch(&self) -> DnnArch {
        self.arch
    }
}

#[cfg(all(test, feature = "deep-plc"))]
mod tests {
    use super::*;

    #[test]
    fn a_new_holder_has_every_builtin_model() {
        let d = DecoderDnn::new();
        assert_eq!(d.models(), builtin::models().models());
        assert!(d.models().contains(DnnModels::DEEP_PLC));
        assert!(d.deep_plc().is_some());
    }

    #[test]
    fn a_failed_load_changes_nothing() {
        let mut d = DecoderDnn::new();
        let before = d.models();
        let empty = DnnBlob::from_static(&[]).unwrap();
        assert_eq!(d.load(&empty), Err(ERR_NO_MODEL));
        assert_eq!(d.models(), before);
        // The deep PLC models still point at the built-in ones.
        let shared = builtin::models().deep_plc.as_ref().unwrap();
        assert!(core::ptr::eq(d.deep_plc().unwrap(), shared));
    }

    #[cfg(feature = "heap")]
    #[test]
    fn a_copied_blob_outlives_its_source_and_is_kept_by_the_decoder() {
        let mut d = DecoderDnn::new();
        let mut bytes = std::vec::Vec::new();
        for &(_, blob) in builtin::blobs() {
            bytes.extend_from_slice(blob);
        }
        // Start the copy at an odd address of the source, to show that the
        // source alignment does not matter.
        let mut shifted = std::vec![0u8; 1];
        shifted.extend_from_slice(&bytes);
        let blob = DnnBlob::from_bytes(&shifted[1..]).unwrap();
        drop(shifted);
        let got = d.load(&blob).unwrap();
        assert_eq!(got, builtin::models().models());
        drop(blob);
        // The layers still read the copy, which the decoder keeps alive.
        let plc = &d.deep_plc().unwrap().plc;
        let builtin_plc = &builtin::models().deep_plc.as_ref().unwrap().plc;
        assert_eq!(plc.plc_dense_in.bias, builtin_plc.plc_dense_in.bias);
        assert!(!core::ptr::eq(plc, builtin_plc));
    }

    #[test]
    fn the_builtin_blob_brings_back_the_shared_models() {
        let mut d = DecoderDnn::new();
        let blob = DnnBlob::from_static(builtin::plc()).unwrap();
        // The PLC blob alone is not a complete deep PLC group.
        assert_eq!(d.load(&blob), Err(ERR_NO_MODEL));
        assert_eq!(d.load(&DnnBlob::builtin()), Ok(builtin::models().models()));
    }
}
