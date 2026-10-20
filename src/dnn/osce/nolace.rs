//! NoLACE, the non-linear adaptive coding enhancer of SILK at decoder
//! complexity 7 and above (the NoLACE parts of libopus `dnn/osce.c`,
//! lines 384-848).
//!
//! As LACE, with a conditioning vector of 160 values, a post-filter
//! convolution on the conditioning vector after each filter stage, and
//! three rounds of adaptive shaping (AdaShape on the second of two
//! channels) followed by an adaptive convolution that mixes the channels.

use super::features::OSCE_FEATURE_DIM;
use super::lace::numbits_embedding;
use super::math::fma;
use super::nndsp::{
    AdaCombParams, AdaCombState, AdaConvParams, AdaConvState, AdaShapeParams, AdaShapeState, OVERLAP_WINDOW_40,
    adacomb_process_frame, adaconv_process_frame, adashape_process_frame,
};
use crate::dnn::models::nolace::*;
use crate::dnn::nnet::{LinearLayer, compute_generic_conv1d, compute_generic_dense, compute_generic_gru};
use crate::dnn::{Activation, DnnArch};

/// The state of NoLACE for one SILK channel (libopus `NoLACEState`).
#[derive(Clone)]
pub struct NolaceState {
    feature_net_conv2_state: [f32; NOLACE_FNET_CONV2_STATE_SIZE],
    feature_net_gru_state: [f32; NOLACE_COND_DIM],
    post_cf1_state: [f32; NOLACE_COND_DIM],
    post_cf2_state: [f32; NOLACE_COND_DIM],
    post_af1_state: [f32; NOLACE_COND_DIM],
    post_af2_state: [f32; NOLACE_COND_DIM],
    post_af3_state: [f32; NOLACE_COND_DIM],
    cf1_state: AdaCombState,
    cf2_state: AdaCombState,
    af1_state: AdaConvState<{ NOLACE_AF1_KERNEL_SIZE * NOLACE_AF1_IN_CHANNELS }, NOLACE_AF1_KERNEL_OUT_SIZE>,
    af2_state: AdaConvState<{ NOLACE_AF2_KERNEL_SIZE * NOLACE_AF2_IN_CHANNELS }, NOLACE_AF2_KERNEL_OUT_SIZE>,
    af3_state: AdaConvState<{ NOLACE_AF3_KERNEL_SIZE * NOLACE_AF3_IN_CHANNELS }, NOLACE_AF3_KERNEL_OUT_SIZE>,
    af4_state: AdaConvState<{ NOLACE_AF4_KERNEL_SIZE * NOLACE_AF4_IN_CHANNELS }, NOLACE_AF4_KERNEL_OUT_SIZE>,
    tdshape1_state: AdaShapeState,
    tdshape2_state: AdaShapeState,
    tdshape3_state: AdaShapeState,
    preemph_mem: f32,
    deemph_mem: f32,
}

impl Default for NolaceState {
    /// The cleared state of `reset_nolace_state`.
    fn default() -> Self {
        NolaceState {
            feature_net_conv2_state: [0.0; NOLACE_FNET_CONV2_STATE_SIZE],
            feature_net_gru_state: [0.0; NOLACE_COND_DIM],
            post_cf1_state: [0.0; NOLACE_COND_DIM],
            post_cf2_state: [0.0; NOLACE_COND_DIM],
            post_af1_state: [0.0; NOLACE_COND_DIM],
            post_af2_state: [0.0; NOLACE_COND_DIM],
            post_af3_state: [0.0; NOLACE_COND_DIM],
            cf1_state: AdaCombState::default(),
            cf2_state: AdaCombState::default(),
            af1_state: AdaConvState::default(),
            af2_state: AdaConvState::default(),
            af3_state: AdaConvState::default(),
            af4_state: AdaConvState::default(),
            tdshape1_state: AdaShapeState::default(),
            tdshape2_state: AdaShapeState::default(),
            tdshape3_state: AdaShapeState::default(),
            preemph_mem: 0.0,
            deemph_mem: 0.0,
        }
    }
}

const NUMBITS_SCALES: [f32; 8] = [
    NOLACE_NUMBITS_SCALE_0,
    NOLACE_NUMBITS_SCALE_1,
    NOLACE_NUMBITS_SCALE_2,
    NOLACE_NUMBITS_SCALE_3,
    NOLACE_NUMBITS_SCALE_4,
    NOLACE_NUMBITS_SCALE_5,
    NOLACE_NUMBITS_SCALE_6,
    NOLACE_NUMBITS_SCALE_7,
];

const fn comb(kernel_size: usize, left_padding: usize, a: f32, b: f32, limit: f32) -> AdaCombParams {
    AdaCombParams {
        frame_size: NOLACE_FRAME_SIZE,
        overlap_size: NOLACE_OVERLAP_SIZE,
        kernel_size,
        left_padding,
        filter_gain_a: a,
        filter_gain_b: b,
        log_gain_limit: limit,
    }
}

const CF1: AdaCombParams = comb(
    NOLACE_CF1_KERNEL_SIZE,
    NOLACE_CF1_LEFT_PADDING,
    NOLACE_CF1_FILTER_GAIN_A,
    NOLACE_CF1_FILTER_GAIN_B,
    NOLACE_CF1_LOG_GAIN_LIMIT,
);
const CF2: AdaCombParams = comb(
    NOLACE_CF2_KERNEL_SIZE,
    NOLACE_CF2_LEFT_PADDING,
    NOLACE_CF2_FILTER_GAIN_A,
    NOLACE_CF2_FILTER_GAIN_B,
    NOLACE_CF2_LOG_GAIN_LIMIT,
);

#[allow(clippy::too_many_arguments)]
const fn conv(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    left_padding: usize,
    a: f32,
    b: f32,
    shape_gain: f32,
) -> AdaConvParams {
    AdaConvParams {
        frame_size: NOLACE_FRAME_SIZE,
        overlap_size: NOLACE_OVERLAP_SIZE,
        in_channels,
        out_channels,
        kernel_size,
        left_padding,
        filter_gain_a: a,
        filter_gain_b: b,
        shape_gain,
    }
}

const AF1: AdaConvParams = conv(
    NOLACE_AF1_IN_CHANNELS,
    NOLACE_AF1_OUT_CHANNELS,
    NOLACE_AF1_KERNEL_SIZE,
    NOLACE_AF1_LEFT_PADDING,
    NOLACE_AF1_FILTER_GAIN_A,
    NOLACE_AF1_FILTER_GAIN_B,
    NOLACE_AF1_SHAPE_GAIN,
);
const AF2: AdaConvParams = conv(
    NOLACE_AF2_IN_CHANNELS,
    NOLACE_AF2_OUT_CHANNELS,
    NOLACE_AF2_KERNEL_SIZE,
    NOLACE_AF2_LEFT_PADDING,
    NOLACE_AF2_FILTER_GAIN_A,
    NOLACE_AF2_FILTER_GAIN_B,
    NOLACE_AF2_SHAPE_GAIN,
);
const AF3: AdaConvParams = conv(
    NOLACE_AF3_IN_CHANNELS,
    NOLACE_AF3_OUT_CHANNELS,
    NOLACE_AF3_KERNEL_SIZE,
    NOLACE_AF3_LEFT_PADDING,
    NOLACE_AF3_FILTER_GAIN_A,
    NOLACE_AF3_FILTER_GAIN_B,
    NOLACE_AF3_SHAPE_GAIN,
);
const AF4: AdaConvParams = conv(
    NOLACE_AF4_IN_CHANNELS,
    NOLACE_AF4_OUT_CHANNELS,
    NOLACE_AF4_KERNEL_SIZE,
    NOLACE_AF4_LEFT_PADDING,
    NOLACE_AF4_FILTER_GAIN_A,
    NOLACE_AF4_FILTER_GAIN_B,
    NOLACE_AF4_SHAPE_GAIN,
);

/// The three shaping layers have the same shape; `interpolate_k` is 1, as
/// `nolace_process_20ms_frame` passes it.
const TDSHAPE: AdaShapeParams = AdaShapeParams {
    feature_dim: NOLACE_TDSHAPE1_FEATURE_DIM,
    frame_size: NOLACE_TDSHAPE1_FRAME_SIZE,
    avg_pool_k: NOLACE_TDSHAPE1_AVG_POOL_K,
    interpolate_k: 1,
};
const _: () = assert!(
    NOLACE_TDSHAPE2_FEATURE_DIM == NOLACE_TDSHAPE1_FEATURE_DIM
        && NOLACE_TDSHAPE3_FEATURE_DIM == NOLACE_TDSHAPE1_FEATURE_DIM
        && NOLACE_TDSHAPE2_FRAME_SIZE == NOLACE_TDSHAPE1_FRAME_SIZE
        && NOLACE_TDSHAPE3_FRAME_SIZE == NOLACE_TDSHAPE1_FRAME_SIZE
        && NOLACE_TDSHAPE2_AVG_POOL_K == NOLACE_TDSHAPE1_AVG_POOL_K
        && NOLACE_TDSHAPE3_AVG_POOL_K == NOLACE_TDSHAPE1_AVG_POOL_K
);
const _: () = assert!(
    NOLACE_AF1_OUT_CHANNELS == 2
        && NOLACE_AF2_IN_CHANNELS == 2
        && NOLACE_AF2_OUT_CHANNELS == 2
        && NOLACE_AF3_OUT_CHANNELS == 2
        && NOLACE_AF4_OUT_CHANNELS == 1
);

/// The input size of `nolace_fnet_conv1`.
const CONV1_IN: usize = NOLACE_NUM_FEATURES + NOLACE_PITCH_EMBEDDING_DIM + 2 * NOLACE_NUMBITS_EMBEDDING_DIM;
/// The samples of a 20 ms frame.
const FRAME: usize = 4 * NOLACE_FRAME_SIZE;
/// The values of the conditioning vectors of a 20 ms frame.
const COND: usize = 4 * NOLACE_COND_DIM;

/// libopus `nolace_feature_net` (`dnn/osce.c:434-506`).
fn nolace_feature_net(
    layers: &NolaceLayers<'_>,
    st: &mut NolaceState,
    output: &mut [f32; COND],
    features: &[f32; 4 * OSCE_FEATURE_DIM],
    numbits: &[f32; 2],
    periods: &[i32; 4],
    arch: DnnArch,
) {
    let mut numbits_embedded = [0.0f32; 2 * NOLACE_NUMBITS_EMBEDDING_DIM];
    let (e0, e1) = numbits_embedded.split_at_mut(NOLACE_NUMBITS_EMBEDDING_DIM);
    numbits_embedding(e0, numbits[0], &NUMBITS_SCALES);
    numbits_embedding(e1, numbits[1], &NUMBITS_SCALES);
    let pitch_embedding = layers.nolace_pitch_embedding.float_weights.unwrap_or(&[]);

    let mut hidden = [0.0f32; 4 * NOLACE_HIDDEN_FEATURE_DIM];
    let mut input = [0.0f32; CONV1_IN];
    for sf in 0..4 {
        input[..NOLACE_NUM_FEATURES].copy_from_slice(&features[sf * OSCE_FEATURE_DIM..(sf + 1) * OSCE_FEATURE_DIM]);
        let p = periods[sf] as usize * NOLACE_PITCH_EMBEDDING_DIM;
        input[NOLACE_NUM_FEATURES..NOLACE_NUM_FEATURES + NOLACE_PITCH_EMBEDDING_DIM]
            .copy_from_slice(&pitch_embedding[p..p + NOLACE_PITCH_EMBEDDING_DIM]);
        input[NOLACE_NUM_FEATURES + NOLACE_PITCH_EMBEDDING_DIM..].copy_from_slice(&numbits_embedded);
        compute_generic_conv1d(
            &layers.nolace_fnet_conv1,
            &mut hidden[sf * NOLACE_HIDDEN_FEATURE_DIM..],
            &mut [],
            &input,
            Activation::Tanh,
            arch,
        );
    }

    let mut cond = [0.0f32; NOLACE_COND_DIM];
    compute_generic_conv1d(&layers.nolace_fnet_conv2, &mut cond, &mut st.feature_net_conv2_state, &hidden, Activation::Tanh, arch);

    let mut up = [0.0f32; COND];
    compute_generic_dense(&layers.nolace_fnet_tconv, &mut up, &cond, Activation::Tanh, arch);

    for sf in 0..4 {
        compute_generic_gru(
            &layers.nolace_fnet_gru_input,
            &layers.nolace_fnet_gru_recurrent,
            &mut st.feature_net_gru_state,
            &up[sf * NOLACE_COND_DIM..(sf + 1) * NOLACE_COND_DIM],
            arch,
        );
        output[sf * NOLACE_COND_DIM..(sf + 1) * NOLACE_COND_DIM].copy_from_slice(&st.feature_net_gru_state);
    }
}

/// One post filter: the conditioning vector of each subframe through a
/// convolution with memory (tanh), as `compute_generic_conv1d(&post_*,
/// feature_transform_buffer + i, state, feature_buffer + i, ...)` does in
/// each loop of `nolace_process_20ms_frame`. The result replaces `cond`
/// after the loop.
fn post_filter(
    layer: &LinearLayer<'_>,
    mem: &mut [f32; NOLACE_COND_DIM],
    cond: &[f32; COND],
    sf: usize,
    next: &mut [f32; COND],
    arch: DnnArch,
) {
    let r = sf * NOLACE_COND_DIM..(sf + 1) * NOLACE_COND_DIM;
    compute_generic_conv1d(layer, &mut next[r.clone()], mem, &cond[r], Activation::Tanh, arch);
}

/// libopus `nolace_process_20ms_frame` (`dnn/osce.c:509-848`): enhances one
/// 20 ms frame of 16 kHz samples in [-1, 1].
#[allow(clippy::too_many_arguments)]
pub fn nolace_process_20ms_frame(
    layers: &NolaceLayers<'_>,
    st: &mut NolaceState,
    x_out: &mut [f32; FRAME],
    x_in: &[f32; FRAME],
    features: &[f32; 4 * OSCE_FEATURE_DIM],
    numbits: &[f32; 2],
    periods: &[i32; 4],
    arch: DnnArch,
) {
    const FS: usize = NOLACE_FRAME_SIZE;
    const CD: usize = NOLACE_COND_DIM;
    let mut x1 = [0.0f32; 2 * FRAME];
    let mut x2 = [0.0f32; 2 * FRAME];
    for (o, &v) in x1.iter_mut().zip(x_in.iter()) {
        *o = fma(-NOLACE_PREEMPH, st.preemph_mem, v);
        st.preemph_mem = v;
    }

    let mut cond = [0.0f32; COND];
    nolace_feature_net(layers, st, &mut cond, features, numbits, periods, arch);
    let mut next = [0.0f32; COND];
    let window = &OVERLAP_WINDOW_40;

    // First comb filter (in place).
    for sf in 0..4 {
        adacomb_process_frame(
            &mut st.cf1_state,
            &mut x1[sf * FS..(sf + 1) * FS],
            &cond[sf * CD..(sf + 1) * CD],
            &layers.nolace_cf1_kernel,
            &layers.nolace_cf1_gain,
            &layers.nolace_cf1_global_gain,
            periods[sf] as usize,
            &CF1,
            window,
            arch,
        );
        post_filter(&layers.nolace_post_cf1, &mut st.post_cf1_state, &cond, sf, &mut next, arch);
    }
    cond = next;

    // Second comb filter (in place).
    for sf in 0..4 {
        adacomb_process_frame(
            &mut st.cf2_state,
            &mut x1[sf * FS..(sf + 1) * FS],
            &cond[sf * CD..(sf + 1) * CD],
            &layers.nolace_cf2_kernel,
            &layers.nolace_cf2_gain,
            &layers.nolace_cf2_global_gain,
            periods[sf] as usize,
            &CF2,
            window,
            arch,
        );
        post_filter(&layers.nolace_post_cf2, &mut st.post_cf2_state, &cond, sf, &mut next, arch);
    }
    cond = next;

    // First adaptive convolution: one channel to two.
    for sf in 0..4 {
        adaconv_process_frame(
            &mut st.af1_state,
            &mut x2[sf * 2 * FS..(sf + 1) * 2 * FS],
            &x1[sf * FS..(sf + 1) * FS],
            &cond[sf * CD..(sf + 1) * CD],
            &layers.nolace_af1_kernel,
            &layers.nolace_af1_gain,
            &AF1,
            window,
            arch,
        );
        post_filter(&layers.nolace_post_af1, &mut st.post_af1_state, &cond, sf, &mut next, arch);
    }
    cond = next;

    // First shape-and-mix round: shape the second channel, then mix the two.
    for sf in 0..4 {
        adashape_process_frame(
            &mut st.tdshape1_state,
            &mut x2[sf * 2 * FS + FS..(sf + 1) * 2 * FS],
            &cond[sf * CD..(sf + 1) * CD],
            &layers.nolace_tdshape1_alpha1_f,
            &layers.nolace_tdshape1_alpha1_t,
            &layers.nolace_tdshape1_alpha2,
            &TDSHAPE,
            arch,
        );
        adaconv_process_frame(
            &mut st.af2_state,
            &mut x1[sf * 2 * FS..(sf + 1) * 2 * FS],
            &x2[sf * 2 * FS..(sf + 1) * 2 * FS],
            &cond[sf * CD..(sf + 1) * CD],
            &layers.nolace_af2_kernel,
            &layers.nolace_af2_gain,
            &AF2,
            window,
            arch,
        );
        post_filter(&layers.nolace_post_af2, &mut st.post_af2_state, &cond, sf, &mut next, arch);
    }
    cond = next;

    // Second shape-and-mix round.
    for sf in 0..4 {
        adashape_process_frame(
            &mut st.tdshape2_state,
            &mut x1[sf * 2 * FS + FS..(sf + 1) * 2 * FS],
            &cond[sf * CD..(sf + 1) * CD],
            &layers.nolace_tdshape2_alpha1_f,
            &layers.nolace_tdshape2_alpha1_t,
            &layers.nolace_tdshape2_alpha2,
            &TDSHAPE,
            arch,
        );
        adaconv_process_frame(
            &mut st.af3_state,
            &mut x2[sf * 2 * FS..(sf + 1) * 2 * FS],
            &x1[sf * 2 * FS..(sf + 1) * 2 * FS],
            &cond[sf * CD..(sf + 1) * CD],
            &layers.nolace_af3_kernel,
            &layers.nolace_af3_gain,
            &AF3,
            window,
            arch,
        );
        post_filter(&layers.nolace_post_af3, &mut st.post_af3_state, &cond, sf, &mut next, arch);
    }
    cond = next;

    // Third round: shape, then mix two channels into one.
    for sf in 0..4 {
        adashape_process_frame(
            &mut st.tdshape3_state,
            &mut x2[sf * 2 * FS + FS..(sf + 1) * 2 * FS],
            &cond[sf * CD..(sf + 1) * CD],
            &layers.nolace_tdshape3_alpha1_f,
            &layers.nolace_tdshape3_alpha1_t,
            &layers.nolace_tdshape3_alpha2,
            &TDSHAPE,
            arch,
        );
        adaconv_process_frame(
            &mut st.af4_state,
            &mut x1[sf * FS..(sf + 1) * FS],
            &x2[sf * 2 * FS..(sf + 1) * 2 * FS],
            &cond[sf * CD..(sf + 1) * CD],
            &layers.nolace_af4_kernel,
            &layers.nolace_af4_gain,
            &AF4,
            window,
            arch,
        );
    }

    for (o, &v) in x_out.iter_mut().zip(x1.iter()) {
        *o = fma(NOLACE_PREEMPH, st.deemph_mem, v);
        st.deemph_mem = *o;
    }
}
