//! The neural network code of libopus 1.6.1 (its `dnn/` folder).
//!
//! libopus 1.5 and later use small neural networks in the decoder when the
//! decoder complexity is high enough:
//!
//! - complexity 5 and above: deep packet loss concealment (deep PLC), with the
//!   PitchDNN, PLC and FARGAN models (cargo feature `deep-plc`, module `plc`);
//! - complexity 6: LACE, and complexity 7 and above: NoLACE, two speech
//!   enhancers for SILK (feature `osce`, module `osce`);
//! - the blind bandwidth extension of SILK wideband to 48 kHz, BBWENet
//!   (feature `osce-bwe`).
//!
//! This module holds what these features share:
//!
//! - `nnet`: the layer types and the layer functions (dense, GRU, GLU,
//!   conv1d, conv2d, activations);
//! - `arch`: the processor variants and their detection;
//! - `vec_portable`, `vec_neon`, `vec_x86`: the kernels of each variant;
//! - `weights`: the parser of libopus weight blobs;
//! - `models`: one layer struct per model;
//! - `builtin`: the 1.6.1 weights embedded in the crate;
//! - `holder`: the weights and models that a decoder holds;
//! - `freq`, `burg`, `lpcnet_tables`: the signal processing of `freq.c`,
//!   `burg.c` and `lpcnet_tables.c`.
//!
//! All kernels compute the bits of libopus 1.6.1 on 64-bit ARM, built
//! without loop vectorization, on every target (see
//! `target/perf/next/plan.md` and `nnet.md` in the repository's working
//! notes).
//!
//! The models themselves (`plc` for deep PLC, `osce` for LACE, NoLACE and
//! the bandwidth extension) are empty modules until they are ported. The
//! decoder already stores the complexity, the bandwidth extension switch
//! and the loaded weights (`holder`), but nothing reads them during
//! decoding yet.

#[cfg(target_endian = "big")]
compile_error!(
    "the opus1 neural network features need a little-endian target: \
     the weight blobs of libopus are little-endian"
);

pub mod arch;
pub mod builtin;
pub mod burg;
pub mod freq;
pub mod holder;
pub mod lpcnet_tables;
pub mod models;
pub mod nnet;
pub mod vec_portable;
#[cfg(target_arch = "aarch64")]
pub mod vec_neon;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod vec_x86;
pub mod weights;

// Filled in the second wave.
#[cfg(feature = "deep-plc")]
pub mod plc;
#[cfg(feature = "osce")]
pub mod osce;

pub use arch::DnnArch;
pub use holder::{DecoderDnn, DnnBlob, DnnModels};
pub use nnet::{Conv2dLayer, LinearLayer};
pub use weights::{WeightArray, WeightList};

/// The activation function applied after a layer. The numbers are those
/// of libopus (`ACTIVATION_*` in `dnn/nnet.h:34-40`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(i32)]
pub enum Activation {
    /// The identity.
    Linear = 0,
    /// `1 / (1 + exp(-x))`, computed with a rational approximation.
    Sigmoid = 1,
    /// `tanh(x)`, computed with a rational approximation.
    Tanh = 2,
    /// `max(x, 0)`; -0.0 and NaN pass through unchanged.
    Relu = 3,
    /// `exp(x)` of every value, then each value divided by the sum.
    Softmax = 4,
    /// `x * sigmoid(x)`.
    Swish = 5,
    /// `exp(x)`, computed with a polynomial approximation.
    Exp = 6,
}

/// The largest input count of an int8 layer and of `compute_glu`
/// (`MAX_INPUTS`, `dnn/vec_neon.h:365`). The 1.6.1 models need 768.
pub const MAX_INPUTS: usize = 2048;

/// The largest GRU state (`MAX_RNN_NEURONS_ALL`, `dnn/nnet.c:66-74`): 192,
/// for the PLC model.
pub const MAX_RNN_NEURONS_ALL: usize = 192;

/// The largest input of a conv1d layer, memory included
/// (`MAX_CONV_INPUTS_ALL`, `dnn/nnet.c:122`).
pub const MAX_CONV_INPUTS_ALL: usize = 1024;

/// The largest input buffer of `compute_conv2d` (`MAX_CONV2D_INPUTS`,
/// `dnn/nnet_arch.h:215`). PitchDNN, the only conv2d user, needs
/// 3 · 4 · 226 = 2,712 values. The layer code may size its buffer from the
/// model maximum instead.
pub const MAX_CONV2D_INPUTS: usize = 8192;

/// The largest value count of a SWISH activation (`MAX_ACTIVATIONS`,
/// `dnn/nnet_arch.h:50`). No 1.6.1 model uses SWISH.
pub const MAX_ACTIVATIONS: usize = 4096;
