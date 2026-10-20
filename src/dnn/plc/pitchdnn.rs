//! PitchDNN, the pitch estimator of deep PLC (libopus `dnn/pitchdnn.c`,
//! `dnn/pitchdnn.h`).
//!
//! Each frame, two inputs go through the network: the 88 instantaneous
//! frequency values (two dense layers) and the 224 normalized
//! cross-correlation values (two 3 × 3 convolutions over time and lag).
//! A dense layer, a GRU and a final dense layer give 180 scores, one per
//! pitch bin; the result is a weighted mean of the bins around the best
//! one.

#[cfg(not(feature = "std"))]
use crate::compat::Math;
use crate::dnn::Activation;
use crate::dnn::arch::DnnArch;
use crate::dnn::freq::fmaf;
use crate::dnn::models::pitchdnn::{
    DENSE_DOWNSAMPLER_OUT_SIZE, DENSE_FINAL_UPSAMPLER_OUT_SIZE, DENSE_IF_UPSAMPLER_1_OUT_SIZE,
    DENSE_IF_UPSAMPLER_2_OUT_SIZE, GRU_1_STATE_SIZE, PitchDnn,
};
use crate::dnn::nnet::{compute_conv2d, compute_generic_dense, compute_generic_gru};

use super::{NB_XCORR_FEATURES, PITCH_IF_FEATURES};

/// The height of one row of the convolution inputs: the 224 lags and one
/// value of zero padding on each side.
const PADDED: usize = NB_XCORR_FEATURES + 2;

/// The output channels of `conv2d_1` (the input channels of `conv2d_2`).
const CONV1_CHANNELS: usize = 4;

/// The pitch bins that the soft arg-max reads (of the 192 outputs of the
/// last layer).
const PITCH_BINS: usize = 180;

/// The state of PitchDNN (libopus `PitchDNNState`, `dnn/pitchdnn.h:15-21`)
/// without the weights.
#[derive(Clone)]
pub struct PitchDnnState {
    /// The GRU state.
    gru_state: [f32; GRU_1_STATE_SIZE],
    /// The last two input rows of `conv2d_1` (one channel).
    xcorr_mem1: [f32; PADDED * 2],
    /// The last two input rows of `conv2d_2` (four channels). libopus
    /// declares twice this size and `xcorr_mem3`, which nothing uses.
    xcorr_mem2: [f32; PADDED * 2 * CONV1_CHANNELS],
}

impl Default for PitchDnnState {
    fn default() -> Self {
        Self::new()
    }
}

impl PitchDnnState {
    /// A cleared state (libopus `pitchdnn_init` without the model).
    pub const fn new() -> Self {
        PitchDnnState {
            gru_state: [0.0; GRU_1_STATE_SIZE],
            xcorr_mem1: [0.0; PADDED * 2],
            xcorr_mem2: [0.0; PADDED * 2 * CONV1_CHANNELS],
        }
    }

    /// Clears the state in place.
    pub fn reset(&mut self) {
        self.gru_state.fill(0.0);
        self.xcorr_mem1.fill(0.0);
        self.xcorr_mem2.fill(0.0);
    }

    /// The GRU state.
    pub fn gru_state(&self) -> &[f32; GRU_1_STATE_SIZE] {
        &self.gru_state
    }

    /// The memory of `conv2d_1`.
    pub fn xcorr_mem1(&self) -> &[f32] {
        &self.xcorr_mem1
    }

    /// The memory of `conv2d_2`.
    pub fn xcorr_mem2(&self) -> &[f32] {
        &self.xcorr_mem2
    }
}

/// `compute_pitchdnn` (`dnn/pitchdnn.c:12-56`): the pitch of one frame, as
/// `(1/60)·bin - 1.5` with `bin` in [0, 179].
///
/// The last step is a soft arg-max: `pos` is the first bin with the
/// largest score (scores must exceed -1); the result is
/// `sum(p·i) / sum(p)` over the bins `pos-2 ..= pos+2` (limited to
/// [0, 179]), with `p = exp(score)`. It rounds as the reference build
/// does: `exp` in `double`, rounded to `float`; `sum += p·i` fused; the
/// division and the product by `1/60` in `float`; the subtraction of 1.5
/// in `double`.
pub fn compute_pitchdnn(
    st: &mut PitchDnnState,
    if_features: &[f32; PITCH_IF_FEATURES],
    xcorr_features: &[f32; NB_XCORR_FEATURES],
    model: &PitchDnn,
    arch: DnnArch,
) -> f32 {
    let mut if1_out = [0.0f32; DENSE_IF_UPSAMPLER_1_OUT_SIZE];
    let mut downsampler_in = [0.0f32; NB_XCORR_FEATURES + DENSE_IF_UPSAMPLER_2_OUT_SIZE];
    let mut downsampler_out = [0.0f32; DENSE_DOWNSAMPLER_OUT_SIZE];
    // libopus sizes both buffers `(NB_XCORR_FEATURES + 2)*8`; the layers
    // read one and four rows.
    let mut conv1_tmp1 = [0.0f32; PADDED];
    let mut conv1_tmp2 = [0.0f32; PADDED * CONV1_CHANNELS];
    let mut output = [0.0f32; DENSE_FINAL_UPSAMPLER_OUT_SIZE];

    // The instantaneous frequency branch.
    compute_generic_dense(
        &model.dense_if_upsampler_1,
        &mut if1_out,
        if_features,
        Activation::Tanh,
        arch,
    );
    compute_generic_dense(
        &model.dense_if_upsampler_2,
        &mut downsampler_in[NB_XCORR_FEATURES..],
        &if1_out,
        Activation::Tanh,
        arch,
    );
    // The cross-correlation branch: the padded row, then the two
    // convolutions. The first writes each channel between its padding
    // (stride `PADDED`, from index 1).
    conv1_tmp1[1..=NB_XCORR_FEATURES].copy_from_slice(xcorr_features);
    compute_conv2d(
        &model.conv2d_1,
        &mut conv1_tmp2[1..],
        &mut st.xcorr_mem1,
        &conv1_tmp1,
        NB_XCORR_FEATURES,
        PADDED,
        Activation::Tanh,
        arch,
    );
    compute_conv2d(
        &model.conv2d_2,
        &mut downsampler_in[..NB_XCORR_FEATURES],
        &mut st.xcorr_mem2,
        &conv1_tmp2,
        NB_XCORR_FEATURES,
        NB_XCORR_FEATURES,
        Activation::Tanh,
        arch,
    );

    compute_generic_dense(
        &model.dense_downsampler,
        &mut downsampler_out,
        &downsampler_in,
        Activation::Tanh,
        arch,
    );
    compute_generic_gru(
        &model.gru_1_input,
        &model.gru_1_recurrent,
        &mut st.gru_state,
        &downsampler_out,
        arch,
    );
    compute_generic_dense(
        &model.dense_final_upsampler,
        &mut output,
        &st.gru_state,
        Activation::Linear,
        arch,
    );

    let mut pos = 0usize;
    let mut maxval = -1.0f32;
    for (i, &v) in output[..PITCH_BINS].iter().enumerate() {
        if v > maxval {
            pos = i;
            maxval = v;
        }
    }
    let mut sum = 0.0f32;
    let mut count = 0.0f32;
    for i in pos.saturating_sub(2)..=(pos + 2).min(PITCH_BINS - 1) {
        let p = (output[i] as f64).exp() as f32;
        sum = fmaf(p, i as f32, sum);
        count += p;
    }
    (((1.0f32 / 60.0) * (sum / count)) as f64 - 1.5) as f32
}
