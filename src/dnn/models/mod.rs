//! The models of libopus 1.6.1: one file per model with its layer struct,
//! its `init` function and the constants of its C header, and here the
//! types those files share and the loading of whole model groups.
//!
//! `tools/dnn-gen/` generates the model files from libopus's
//! `dnn/*_data.c` and `dnn/*_data.h`; this file is written by hand. Each
//! model file has:
//!
//! - `LAYERS`: one `LayerSpec` per layer, in the order of the C init
//!   function: the names of the arrays it reads and its shape;
//! - the layer struct, with the C field names in the C order, and `init`,
//!   which builds every layer from a `WeightList` (libopus `init_*`);
//! - `layers()`, every layer of a built model in the order of `LAYERS`.
//!
//! With the feature `dred`, the two models of deep redundancy (DRED) are
//! here too: the RDOVAE (rate-distortion-optimized variational autoencoder)
//! encoder and decoder. `dred_rdovae_stats.rs`, also generated, holds the
//! constants of `dnn/dred_rdovae_constants.h` and the tables of DRED's
//! quantizer. `ModelSet` does not load the DRED models: in libopus the
//! encoder and the DRED decoder hold them, not the Opus decoder.

use super::holder::DnnModels;
use super::nnet::{Conv2dLayer, LinearLayer};
use super::weights::{WeightList, conv2d_init, linear_init};

// PitchDNN, the pitch estimator of deep PLC.
#[cfg(feature = "deep-plc")]
pub mod pitchdnn;
// The PLC model, which predicts the features of a lost frame.
#[cfg(feature = "deep-plc")]
pub mod plc;
// FARGAN, the vocoder of deep PLC.
#[cfg(feature = "deep-plc")]
pub mod fargan;
// The RDOVAE encoder of DRED, which the Opus encoder runs.
#[cfg(feature = "dred")]
pub mod dred_rdovae_enc;
// The RDOVAE decoder of DRED, which the DRED decoder runs.
#[cfg(feature = "dred")]
pub mod dred_rdovae_dec;
// The constants of the RDOVAE and the tables of DRED's quantizer.
#[cfg(feature = "dred")]
pub mod dred_rdovae_stats;
// LACE, the speech enhancer of decoder complexity 6.
#[cfg(feature = "osce")]
pub mod lace;
// NoLACE, the speech enhancer of decoder complexity 7 and above.
#[cfg(feature = "osce")]
pub mod nolace;
// BBWENet, the network of the bandwidth extension.
#[cfg(feature = "osce-bwe")]
pub mod bbwenet;

/// The error of `LayerSpec::linear` on a conv2d layer and of
/// `LayerSpec::conv2d` on a linear layer.
pub const ERR_LAYER_KIND: &str = "DNN layer has another kind";
/// The error of `ModelSet::load` when no model group of this build loads.
pub const ERR_NO_MODEL: &str = "DNN blob holds no model of this build";

/// The arguments of one `linear_init` call of a C init function.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LinearSpec {
    /// The field of the C struct.
    pub name: &'static str,
    /// The array names of the layer's parts (see `LinearLayer`); `None`
    /// where the C call passes `NULL`.
    pub bias: Option<&'static str>,
    /// See `bias`.
    pub subias: Option<&'static str>,
    /// See `bias`.
    pub weights: Option<&'static str>,
    /// See `bias`.
    pub float_weights: Option<&'static str>,
    /// See `bias`.
    pub weights_idx: Option<&'static str>,
    /// See `bias`.
    pub diag: Option<&'static str>,
    /// See `bias`.
    pub scale: Option<&'static str>,
    /// The input count.
    pub nb_inputs: usize,
    /// The output count.
    pub nb_outputs: usize,
}

/// The arguments of one `conv2d_init` call of a C init function.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Conv2dSpec {
    /// The field of the C struct.
    pub name: &'static str,
    /// The array names of the bias and of the weights (see `Conv2dLayer`);
    /// `None` where the C call passes `NULL`.
    pub bias: Option<&'static str>,
    /// See `bias`.
    pub float_weights: Option<&'static str>,
    /// The input channel count.
    pub in_channels: usize,
    /// The output channel count.
    pub out_channels: usize,
    /// The kernel size along time.
    pub ktime: usize,
    /// The kernel size along the height (frequency) axis.
    pub kheight: usize,
}

/// One layer of a model: which arrays it reads, and its shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerSpec {
    /// A `LinearLayer`, built by `linear_init`.
    Linear(LinearSpec),
    /// A `Conv2dLayer`, built by `conv2d_init`.
    Conv2d(Conv2dSpec),
}

impl LayerSpec {
    /// The field name of the layer.
    pub const fn name(&self) -> &'static str {
        match self {
            LayerSpec::Linear(s) => s.name,
            LayerSpec::Conv2d(s) => s.name,
        }
    }

    /// Builds the layer with `linear_init`. Fails with `ERR_LAYER_KIND`
    /// for a conv2d layer.
    pub fn linear<'w>(&self, list: &WeightList<'w>) -> Result<LinearLayer<'w>, &'static str> {
        match self {
            LayerSpec::Linear(s) => linear_init(
                list,
                s.bias,
                s.subias,
                s.weights,
                s.float_weights,
                s.weights_idx,
                s.diag,
                s.scale,
                s.nb_inputs,
                s.nb_outputs,
            ),
            LayerSpec::Conv2d(_) => Err(ERR_LAYER_KIND),
        }
    }

    /// Builds the layer with `conv2d_init`. Fails with `ERR_LAYER_KIND`
    /// for a linear layer.
    pub fn conv2d<'w>(&self, list: &WeightList<'w>) -> Result<Conv2dLayer<'w>, &'static str> {
        match self {
            LayerSpec::Conv2d(s) => {
                conv2d_init(list, s.bias, s.float_weights, s.in_channels, s.out_channels, s.ktime, s.kheight)
            }
            LayerSpec::Linear(_) => Err(ERR_LAYER_KIND),
        }
    }
}

/// One layer of a built model, as `layers()` of a model returns it.
#[derive(Clone, Copy)]
pub enum Layer<'a, 'w> {
    /// A dense, GRU, GLU or conv1d layer.
    Linear(&'a LinearLayer<'w>),
    /// A conv2d layer.
    Conv2d(&'a Conv2dLayer<'w>),
}

/// The three models of deep PLC, which load together, as in libopus's
/// `lpcnet_plc_load_model` (`dnn/lpcnet_plc.c:75-89`).
#[cfg(feature = "deep-plc")]
#[derive(Clone, Copy, Default)]
pub struct DeepPlcModels<'w> {
    /// The PLC model.
    pub plc: plc::PlcModel<'w>,
    /// PitchDNN.
    pub pitchdnn: pitchdnn::PitchDnn<'w>,
    /// FARGAN.
    pub fargan: fargan::Fargan<'w>,
}

#[cfg(feature = "deep-plc")]
impl<'w> DeepPlcModels<'w> {
    /// Builds the three models, in the order of `lpcnet_plc_load_model`:
    /// the PLC model, PitchDNN, FARGAN. Fails when one of them fails.
    pub fn init(list: &WeightList<'w>) -> Result<Self, &'static str> {
        Ok(DeepPlcModels {
            plc: plc::PlcModel::init(list)?,
            pitchdnn: pitchdnn::PitchDnn::init(list)?,
            fargan: fargan::Fargan::init(list)?,
        })
    }
}

/// The two models of the DRED encoder, which load together, as in libopus's
/// `dred_encoder_load_model` (`dnn/dred_encoder.c:46-57`): the RDOVAE
/// encoder, and PitchDNN for the feature analysis.
#[cfg(feature = "dred")]
#[derive(Clone, Copy, Default)]
pub struct DredEncModels<'w> {
    /// The RDOVAE encoder.
    pub rdovae: dred_rdovae_enc::RdovaeEnc<'w>,
    /// PitchDNN.
    pub pitchdnn: pitchdnn::PitchDnn<'w>,
}

#[cfg(feature = "dred")]
impl<'w> DredEncModels<'w> {
    /// Builds both models, the RDOVAE encoder first. Fails when one of them
    /// fails.
    pub fn init(list: &WeightList<'w>) -> Result<Self, &'static str> {
        Ok(DredEncModels {
            rdovae: dred_rdovae_enc::RdovaeEnc::init(list)?,
            pitchdnn: pitchdnn::PitchDnn::init(list)?,
        })
    }
}

/// The models of every group that this build contains, each loaded or not.
///
/// The groups are those of `DnnModels`: deep PLC (PitchDNN, the PLC model
/// and FARGAN together), LACE, NoLACE and BBWENet. `load` replaces a group
/// only when all of its arrays load, so a group is never half-loaded. This
/// differs from libopus, where one failing model makes the whole request
/// fail and can leave layers half-replaced (`nnet.md` section 4.4).
#[derive(Clone, Copy, Default)]
pub struct ModelSet<'w> {
    /// Deep PLC: PitchDNN, the PLC model and FARGAN.
    #[cfg(feature = "deep-plc")]
    pub deep_plc: Option<DeepPlcModels<'w>>,
    /// LACE.
    #[cfg(feature = "osce")]
    pub lace: Option<lace::LaceLayers<'w>>,
    /// NoLACE.
    #[cfg(feature = "osce")]
    pub nolace: Option<nolace::NolaceLayers<'w>>,
    /// BBWENet.
    #[cfg(feature = "osce-bwe")]
    pub bbwenet: Option<bbwenet::BbwenetLayers<'w>>,
    /// Keeps the lifetime parameter in a build with only `dnn`.
    _weights: core::marker::PhantomData<&'w [u8]>,
}

impl<'w> ModelSet<'w> {
    /// A set with no model loaded.
    pub const fn new() -> Self {
        ModelSet {
            #[cfg(feature = "deep-plc")]
            deep_plc: None,
            #[cfg(feature = "osce")]
            lace: None,
            #[cfg(feature = "osce")]
            nolace: None,
            #[cfg(feature = "osce-bwe")]
            bbwenet: None,
            _weights: core::marker::PhantomData,
        }
    }

    /// The groups that are loaded.
    pub fn models(&self) -> DnnModels {
        #[allow(unused_mut)]
        let mut m = DnnModels::NONE;
        #[cfg(feature = "deep-plc")]
        if self.deep_plc.is_some() {
            m = m.union(DnnModels::DEEP_PLC);
        }
        #[cfg(feature = "osce")]
        if self.lace.is_some() {
            m = m.union(DnnModels::LACE);
        }
        #[cfg(feature = "osce")]
        if self.nolace.is_some() {
            m = m.union(DnnModels::NOLACE);
        }
        #[cfg(feature = "osce-bwe")]
        if self.bbwenet.is_some() {
            m = m.union(DnnModels::BBWENET);
        }
        m
    }

    /// Loads every group of this build whose arrays are all in `list`, and
    /// replaces those groups. A group that fails keeps what it had. Returns
    /// the groups loaded by this call, or `ERR_NO_MODEL` when none loads.
    pub fn load(&mut self, list: &WeightList<'w>) -> Result<DnnModels, &'static str> {
        #[allow(unused_mut)]
        let mut loaded = DnnModels::NONE;
        #[cfg(feature = "deep-plc")]
        if let Ok(m) = DeepPlcModels::init(list) {
            self.deep_plc = Some(m);
            loaded = loaded.union(DnnModels::DEEP_PLC);
        }
        #[cfg(feature = "osce")]
        if let Ok(m) = lace::LaceLayers::init(list) {
            self.lace = Some(m);
            loaded = loaded.union(DnnModels::LACE);
        }
        #[cfg(feature = "osce")]
        if let Ok(m) = nolace::NolaceLayers::init(list) {
            self.nolace = Some(m);
            loaded = loaded.union(DnnModels::NOLACE);
        }
        #[cfg(feature = "osce-bwe")]
        if let Ok(m) = bbwenet::BbwenetLayers::init(list) {
            self.bbwenet = Some(m);
            loaded = loaded.union(DnnModels::BBWENET);
        }
        #[cfg(not(feature = "deep-plc"))]
        let _ = list;
        if loaded.is_empty() { Err(ERR_NO_MODEL) } else { Ok(loaded) }
    }
}
