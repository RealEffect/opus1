//! LACE, the linear adaptive coding enhancer of SILK at decoder complexity
//! 6 (the LACE parts of libopus `dnn/osce.c`, lines 112-380).
//!
//! Per 20 ms frame: pre-emphasis; the feature network (4 subframe
//! convolutions, a convolution over the frame, a transposed convolution
//! and a GRU per subframe) gives one conditioning vector of 128 values per
//! 5 ms subframe; two adaptive comb filters and one adaptive convolution
//! filter the signal subframe by subframe; de-emphasis.

use super::features::OSCE_FEATURE_DIM;
use super::math::{fma, ln, sin};
use super::nndsp::{
    AdaCombParams, AdaCombState, AdaConvParams, AdaConvState, OVERLAP_WINDOW_40, adacomb_process_frame,
    adaconv_process_frame,
};
use crate::dnn::models::lace::*;
use crate::dnn::nnet::{compute_generic_conv1d, compute_generic_dense, compute_generic_gru};
use crate::dnn::{Activation, DnnArch};

/// The state of LACE for one SILK channel (libopus `LACEState`).
#[derive(Clone)]
pub struct LaceState {
    feature_net_conv2_state: [f32; LACE_FNET_CONV2_STATE_SIZE],
    feature_net_gru_state: [f32; LACE_COND_DIM],
    cf1_state: AdaCombState,
    cf2_state: AdaCombState,
    af1_state: AdaConvState<{ LACE_AF1_KERNEL_SIZE * LACE_AF1_IN_CHANNELS }, LACE_AF1_KERNEL_OUT_SIZE>,
    preemph_mem: f32,
    deemph_mem: f32,
}

impl Default for LaceState {
    /// The cleared state of `reset_lace_state`.
    fn default() -> Self {
        LaceState {
            feature_net_conv2_state: [0.0; LACE_FNET_CONV2_STATE_SIZE],
            feature_net_gru_state: [0.0; LACE_COND_DIM],
            cf1_state: AdaCombState::default(),
            cf2_state: AdaCombState::default(),
            af1_state: AdaConvState::default(),
            preemph_mem: 0.0,
            deemph_mem: 0.0,
        }
    }
}

/// `(float)log(LACE_NUMBITS_RANGE_LOW)`, as clang folds it.
pub(super) const LOG_NUMBITS_LOW: f32 = f32::from_bits(0x407a_5e96);
/// `(float)log(LACE_NUMBITS_RANGE_HIGH)`.
pub(super) const LOG_NUMBITS_HIGH: f32 = f32::from_bits(0x40cf_435c);

/// libopus `compute_lace_numbits_embedding` and
/// `compute_nolace_numbits_embedding` (`dnn/osce.c:122-138`, `388-404`):
/// eight sine embeddings of the log of a bit count.
///
/// The bit count is clamped from above only: the C macro `CLIP` returns
/// its argument when it is below the lower limit.
pub(super) fn numbits_embedding(emb: &mut [f32], numbits: f32, scales: &[f32; 8]) {
    let (min_val, max_val) = (LOG_NUMBITS_LOW, LOG_NUMBITS_HIGH);
    let n = ln(numbits as f64) as f32;
    let clipped = if (if n < min_val { min_val } else { n }) > max_val { max_val } else { n };
    let x = clipped - (max_val + min_val) / 2.0;
    for (e, &s) in emb.iter_mut().zip(scales) {
        *e = sin(fma(x, s, -0.5) as f64) as f32;
    }
}

const NUMBITS_SCALES: [f32; 8] = [
    LACE_NUMBITS_SCALE_0,
    LACE_NUMBITS_SCALE_1,
    LACE_NUMBITS_SCALE_2,
    LACE_NUMBITS_SCALE_3,
    LACE_NUMBITS_SCALE_4,
    LACE_NUMBITS_SCALE_5,
    LACE_NUMBITS_SCALE_6,
    LACE_NUMBITS_SCALE_7,
];

const CF1: AdaCombParams = AdaCombParams {
    frame_size: LACE_FRAME_SIZE,
    overlap_size: LACE_OVERLAP_SIZE,
    kernel_size: LACE_CF1_KERNEL_SIZE,
    left_padding: LACE_CF1_LEFT_PADDING,
    filter_gain_a: LACE_CF1_FILTER_GAIN_A,
    filter_gain_b: LACE_CF1_FILTER_GAIN_B,
    log_gain_limit: LACE_CF1_LOG_GAIN_LIMIT,
};

const CF2: AdaCombParams = AdaCombParams {
    frame_size: LACE_FRAME_SIZE,
    overlap_size: LACE_OVERLAP_SIZE,
    kernel_size: LACE_CF2_KERNEL_SIZE,
    left_padding: LACE_CF2_LEFT_PADDING,
    filter_gain_a: LACE_CF2_FILTER_GAIN_A,
    filter_gain_b: LACE_CF2_FILTER_GAIN_B,
    log_gain_limit: LACE_CF2_LOG_GAIN_LIMIT,
};

const AF1: AdaConvParams = AdaConvParams {
    frame_size: LACE_FRAME_SIZE,
    overlap_size: LACE_OVERLAP_SIZE,
    in_channels: LACE_AF1_IN_CHANNELS,
    out_channels: LACE_AF1_OUT_CHANNELS,
    kernel_size: LACE_AF1_KERNEL_SIZE,
    left_padding: LACE_AF1_LEFT_PADDING,
    filter_gain_a: LACE_AF1_FILTER_GAIN_A,
    filter_gain_b: LACE_AF1_FILTER_GAIN_B,
    shape_gain: LACE_AF1_SHAPE_GAIN,
};

/// The input size of `lace_fnet_conv1`: the features, the pitch embedding
/// and the two bit count embeddings.
const CONV1_IN: usize = LACE_NUM_FEATURES + LACE_PITCH_EMBEDDING_DIM + 2 * LACE_NUMBITS_EMBEDDING_DIM;

/// libopus `lace_feature_net` (`dnn/osce.c:163-235`): the conditioning
/// vectors of the 4 subframes.
fn lace_feature_net(
    layers: &LaceLayers<'_>,
    st: &mut LaceState,
    output: &mut [f32; 4 * LACE_COND_DIM],
    features: &[f32; 4 * OSCE_FEATURE_DIM],
    numbits: &[f32; 2],
    periods: &[i32; 4],
    arch: DnnArch,
) {
    let mut numbits_embedded = [0.0f32; 2 * LACE_NUMBITS_EMBEDDING_DIM];
    let (e0, e1) = numbits_embedded.split_at_mut(LACE_NUMBITS_EMBEDDING_DIM);
    numbits_embedding(e0, numbits[0], &NUMBITS_SCALES);
    numbits_embedding(e1, numbits[1], &NUMBITS_SCALES);
    let pitch_embedding = layers.lace_pitch_embedding.float_weights.unwrap_or(&[]);

    // Scaling and dimensionality reduction, per subframe.
    let mut hidden = [0.0f32; 4 * LACE_HIDDEN_FEATURE_DIM];
    let mut input = [0.0f32; CONV1_IN];
    for sf in 0..4 {
        input[..LACE_NUM_FEATURES].copy_from_slice(&features[sf * OSCE_FEATURE_DIM..(sf + 1) * OSCE_FEATURE_DIM]);
        let p = periods[sf] as usize * LACE_PITCH_EMBEDDING_DIM;
        input[LACE_NUM_FEATURES..LACE_NUM_FEATURES + LACE_PITCH_EMBEDDING_DIM]
            .copy_from_slice(&pitch_embedding[p..p + LACE_PITCH_EMBEDDING_DIM]);
        input[LACE_NUM_FEATURES + LACE_PITCH_EMBEDDING_DIM..].copy_from_slice(&numbits_embedded);
        compute_generic_conv1d(
            &layers.lace_fnet_conv1,
            &mut hidden[sf * LACE_HIDDEN_FEATURE_DIM..],
            &mut [],
            &input,
            Activation::Tanh,
            arch,
        );
    }

    // Subframe accumulation.
    let mut cond = [0.0f32; LACE_COND_DIM];
    compute_generic_conv1d(&layers.lace_fnet_conv2, &mut cond, &mut st.feature_net_conv2_state, &hidden, Activation::Tanh, arch);

    // Upsampling with the transposed convolution.
    let mut up = [0.0f32; 4 * LACE_COND_DIM];
    compute_generic_dense(&layers.lace_fnet_tconv, &mut up, &cond, Activation::Tanh, arch);

    // The GRU, per subframe.
    for sf in 0..4 {
        compute_generic_gru(
            &layers.lace_fnet_gru_input,
            &layers.lace_fnet_gru_recurrent,
            &mut st.feature_net_gru_state,
            &up[sf * LACE_COND_DIM..(sf + 1) * LACE_COND_DIM],
            arch,
        );
        output[sf * LACE_COND_DIM..(sf + 1) * LACE_COND_DIM].copy_from_slice(&st.feature_net_gru_state);
    }
}

/// libopus `lace_process_20ms_frame` (`dnn/osce.c:238-380`): enhances one
/// 20 ms frame of 16 kHz samples in [-1, 1].
///
/// `features`, `numbits` and `periods` come from
/// `features::osce_calculate_features`. Pre-emphasis is
/// `fma(-0.85, prev, x)`, de-emphasis `fma(0.85, prev, y)`.
#[allow(clippy::too_many_arguments)]
pub fn lace_process_20ms_frame(
    layers: &LaceLayers<'_>,
    st: &mut LaceState,
    x_out: &mut [f32; 4 * LACE_FRAME_SIZE],
    x_in: &[f32; 4 * LACE_FRAME_SIZE],
    features: &[f32; 4 * OSCE_FEATURE_DIM],
    numbits: &[f32; 2],
    periods: &[i32; 4],
    arch: DnnArch,
) {
    let mut x = [0.0f32; 4 * LACE_FRAME_SIZE];
    for (o, &v) in x.iter_mut().zip(x_in.iter()) {
        *o = fma(-LACE_PREEMPH, st.preemph_mem, v);
        st.preemph_mem = v;
    }

    let mut cond = [0.0f32; 4 * LACE_COND_DIM];
    lace_feature_net(layers, st, &mut cond, features, numbits, periods, arch);

    let window = &OVERLAP_WINDOW_40;
    for sf in 0..4 {
        adacomb_process_frame(
            &mut st.cf1_state,
            &mut x[sf * LACE_FRAME_SIZE..(sf + 1) * LACE_FRAME_SIZE],
            &cond[sf * LACE_COND_DIM..(sf + 1) * LACE_COND_DIM],
            &layers.lace_cf1_kernel,
            &layers.lace_cf1_gain,
            &layers.lace_cf1_global_gain,
            periods[sf] as usize,
            &CF1,
            window,
            arch,
        );
    }
    for sf in 0..4 {
        adacomb_process_frame(
            &mut st.cf2_state,
            &mut x[sf * LACE_FRAME_SIZE..(sf + 1) * LACE_FRAME_SIZE],
            &cond[sf * LACE_COND_DIM..(sf + 1) * LACE_COND_DIM],
            &layers.lace_cf2_kernel,
            &layers.lace_cf2_gain,
            &layers.lace_cf2_global_gain,
            periods[sf] as usize,
            &CF2,
            window,
            arch,
        );
    }
    for sf in 0..4 {
        let sub = &mut x[sf * LACE_FRAME_SIZE..(sf + 1) * LACE_FRAME_SIZE];
        let mut input = [0.0f32; LACE_FRAME_SIZE];
        input.copy_from_slice(sub);
        adaconv_process_frame(
            &mut st.af1_state,
            sub,
            &input,
            &cond[sf * LACE_COND_DIM..(sf + 1) * LACE_COND_DIM],
            &layers.lace_af1_kernel,
            &layers.lace_af1_gain,
            &AF1,
            window,
            arch,
        );
    }

    for (o, &v) in x_out.iter_mut().zip(x.iter()) {
        *o = fma(LACE_PREEMPH, st.deemph_mem, v);
        st.deemph_mem = *o;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_folded_log_limits_are_the_logs_of_the_bit_range() {
        assert_eq!(LOG_NUMBITS_LOW, (LACE_NUMBITS_RANGE_LOW as f64).ln() as f32);
        assert_eq!(LOG_NUMBITS_HIGH, (LACE_NUMBITS_RANGE_HIGH as f64).ln() as f32);
    }

    #[test]
    fn the_bit_count_is_clamped_from_above_only() {
        let mut lo = [0.0f32; 8];
        let mut tiny = [0.0f32; 8];
        let mut hi = [0.0f32; 8];
        let mut huge = [0.0f32; 8];
        numbits_embedding(&mut lo, 50.0, &NUMBITS_SCALES);
        numbits_embedding(&mut tiny, 5.0, &NUMBITS_SCALES);
        numbits_embedding(&mut hi, 650.0, &NUMBITS_SCALES);
        numbits_embedding(&mut huge, 6500.0, &NUMBITS_SCALES);
        assert_ne!(lo, tiny);
        assert_eq!(hi, huge);
    }
}
