//! BBWENet, the blind bandwidth extension of SILK wideband to 48 kHz (the
//! BBWENet parts of libopus `dnn/osce.c`, lines 850-1374).
//!
//! Per 10 ms frame the feature network (two convolutions, a transposed
//! convolution with stride 2 and a GRU per 5 ms) gives one conditioning
//! vector of 128 values per 5 ms subframe. The signal network then runs per
//! subframe:
//!
//! 1. an adaptive convolution from one channel to three (16 kHz);
//! 2. per channel, 2x upsampling with all-pass filters (32 kHz); the second
//!    channel is shaped (AdaShape), the third goes through
//!    `x·celt_sin(celt_log(|x| + 1e-6))`;
//! 3. an adaptive convolution from three channels to three;
//! 4. per channel, 3/2 interpolation (48 kHz); shaping of the second
//!    channel and the same non-linear function on the third;
//! 5. an adaptive convolution from three channels to one.
//!
//! libopus runs each step over all subframes of the call before the next
//! step. Every step has its own state, so running all steps for one
//! subframe before the next subframe performs the same operations on the
//! same values; this port does that, which keeps the buffers small.

use super::math::{celt_log, celt_sin, fma};
use super::nndsp::{
    AdaConvParams, AdaConvState, AdaShapeParams, AdaShapeState, OVERLAP_WINDOW_120, OVERLAP_WINDOW_40,
    OVERLAP_WINDOW_80, adaconv_process_frame, adashape_process_frame,
};
use crate::dnn::models::bbwenet::*;
use crate::dnn::nnet::{compute_generic_conv1d, compute_generic_dense, compute_generic_gru};
use crate::dnn::{Activation, DnnArch};

/// `DELAY_SAMPLES` of `interpol_3_2`.
const DELAY_SAMPLES: usize = 8;

/// The state of the two resamplers of one channel (libopus
/// `resamp_state`): the all-pass memories of `upsamp_2x` and the last input
/// samples of `interpol_3_2`.
#[derive(Clone, Copy, Default)]
struct ResampState {
    upsamp_buffer: [[f32; 3]; 2],
    interpol_buffer: [f32; DELAY_SAMPLES],
}

/// The state of BBWENet for one SILK channel (libopus `BBWENetState`).
#[derive(Clone)]
pub struct BbwenetState {
    feature_net_conv1_state: [f32; BBWENET_FNET_CONV1_STATE_SIZE],
    feature_net_conv2_state: [f32; BBWENET_FNET_CONV2_STATE_SIZE],
    feature_net_gru_state: [f32; BBWENET_FNET_GRU_STATE_SIZE],
    /// The last 21 output samples, which `osce_bwe` returns at the start of
    /// the next call (libopus `outbut_buffer`).
    pub(super) output_buffer: [i16; super::OSCE_BWE_OUTPUT_DELAY],
    af1_state: AdaConvState<{ BBWENET_AF1_KERNEL_SIZE * BBWENET_AF1_IN_CHANNELS }, BBWENET_AF1_KERNEL_OUT_SIZE>,
    af2_state: AdaConvState<{ BBWENET_AF2_KERNEL_SIZE * BBWENET_AF2_IN_CHANNELS }, BBWENET_AF2_KERNEL_OUT_SIZE>,
    af3_state: AdaConvState<{ BBWENET_AF3_KERNEL_SIZE * BBWENET_AF3_IN_CHANNELS }, BBWENET_AF3_KERNEL_OUT_SIZE>,
    tdshape1_state: AdaShapeState,
    tdshape2_state: AdaShapeState,
    resampler_state: [ResampState; 3],
}

impl Default for BbwenetState {
    /// The cleared state of `reset_bbwenet_state` and `silk_init_decoder`.
    fn default() -> Self {
        BbwenetState {
            feature_net_conv1_state: [0.0; BBWENET_FNET_CONV1_STATE_SIZE],
            feature_net_conv2_state: [0.0; BBWENET_FNET_CONV2_STATE_SIZE],
            feature_net_gru_state: [0.0; BBWENET_FNET_GRU_STATE_SIZE],
            output_buffer: [0; super::OSCE_BWE_OUTPUT_DELAY],
            af1_state: AdaConvState::default(),
            af2_state: AdaConvState::default(),
            af3_state: AdaConvState::default(),
            tdshape1_state: AdaShapeState::default(),
            tdshape2_state: AdaShapeState::default(),
            resampler_state: [ResampState::default(); 3],
        }
    }
}

// The layout that the code below assumes (the asserts of libopus).
const _: () = assert!(
    BBWENET_FNET_GRU_STATE_SIZE == BBWENET_FNET_TCONV_OUT_CHANNELS
        && BBWENET_FNET_TCONV_OUT_CHANNELS == BBWENET_FNET_CONV2_OUT_SIZE
        && BBWENET_FNET_CONV2_OUT_SIZE == BBWENET_FNET_CONV1_OUT_SIZE
        && BBWENET_FNET_TCONV_STRIDE == 2
        && BBWENET_COND_DIM == BBWENET_FNET_GRU_STATE_SIZE
);
const _: () = assert!(
    BBWENET_AF1_OUT_CHANNELS == 3
        && BBWENET_AF2_IN_CHANNELS == 3
        && BBWENET_AF2_OUT_CHANNELS == 3
        && BBWENET_AF3_IN_CHANNELS == 3
        && BBWENET_AF3_OUT_CHANNELS == 1
        && 2 * BBWENET_AF1_FRAME_SIZE == BBWENET_TDSHAPE1_FRAME_SIZE
        && BBWENET_AF2_FRAME_SIZE == BBWENET_TDSHAPE1_FRAME_SIZE
        && 3 * BBWENET_AF2_FRAME_SIZE == 2 * BBWENET_TDSHAPE2_FRAME_SIZE
        && BBWENET_AF3_FRAME_SIZE == BBWENET_TDSHAPE2_FRAME_SIZE
);

/// The samples of one subframe at 16, 32 and 48 kHz.
const F16: usize = BBWENET_AF1_FRAME_SIZE;
const F32: usize = BBWENET_AF2_FRAME_SIZE;
const F48: usize = BBWENET_AF3_FRAME_SIZE;

const AF1: AdaConvParams = AdaConvParams {
    frame_size: BBWENET_AF1_FRAME_SIZE,
    overlap_size: BBWENET_AF1_OVERLAP_SIZE,
    in_channels: BBWENET_AF1_IN_CHANNELS,
    out_channels: BBWENET_AF1_OUT_CHANNELS,
    kernel_size: BBWENET_AF1_KERNEL_SIZE,
    left_padding: BBWENET_AF1_LEFT_PADDING,
    filter_gain_a: BBWENET_AF1_FILTER_GAIN_A,
    filter_gain_b: BBWENET_AF1_FILTER_GAIN_B,
    shape_gain: BBWENET_AF1_SHAPE_GAIN,
};
const AF2: AdaConvParams = AdaConvParams {
    frame_size: BBWENET_AF2_FRAME_SIZE,
    overlap_size: BBWENET_AF2_OVERLAP_SIZE,
    in_channels: BBWENET_AF2_IN_CHANNELS,
    out_channels: BBWENET_AF2_OUT_CHANNELS,
    kernel_size: BBWENET_AF2_KERNEL_SIZE,
    left_padding: BBWENET_AF2_LEFT_PADDING,
    filter_gain_a: BBWENET_AF2_FILTER_GAIN_A,
    filter_gain_b: BBWENET_AF2_FILTER_GAIN_B,
    shape_gain: BBWENET_AF2_SHAPE_GAIN,
};
const AF3: AdaConvParams = AdaConvParams {
    frame_size: BBWENET_AF3_FRAME_SIZE,
    overlap_size: BBWENET_AF3_OVERLAP_SIZE,
    in_channels: BBWENET_AF3_IN_CHANNELS,
    out_channels: BBWENET_AF3_OUT_CHANNELS,
    kernel_size: BBWENET_AF3_KERNEL_SIZE,
    left_padding: BBWENET_AF3_LEFT_PADDING,
    filter_gain_a: BBWENET_AF3_FILTER_GAIN_A,
    filter_gain_b: BBWENET_AF3_FILTER_GAIN_B,
    shape_gain: BBWENET_AF3_SHAPE_GAIN,
};
const TDSHAPE1: AdaShapeParams = AdaShapeParams {
    feature_dim: BBWENET_TDSHAPE1_FEATURE_DIM,
    frame_size: BBWENET_TDSHAPE1_FRAME_SIZE,
    avg_pool_k: BBWENET_TDSHAPE1_AVG_POOL_K,
    interpolate_k: BBWENET_TDSHAPE1_INTERPOLATE_K,
};
const TDSHAPE2: AdaShapeParams = AdaShapeParams {
    feature_dim: BBWENET_TDSHAPE2_FEATURE_DIM,
    frame_size: BBWENET_TDSHAPE2_FRAME_SIZE,
    avg_pool_k: BBWENET_TDSHAPE2_AVG_POOL_K,
    interpolate_k: BBWENET_TDSHAPE2_INTERPOLATE_K,
};

/// The coefficients of `upsamp_2x` (`hq_2x_even`, `hq_2x_odd`). The C arrays
/// are `float` arrays written with `double` literals.
const HQ_2X_EVEN: [f32; 3] = [0.026641845703125_f64 as f32, 0.228668212890625_f64 as f32, -0.4036407470703125_f64 as f32];
const HQ_2X_ODD: [f32; 3] = [0.104583740234375_f64 as f32, 0.3932037353515625_f64 as f32, -0.152496337890625_f64 as f32];

/// The three 8-tap filters of `interpol_3_2` (`frac_01_24`, `frac_17_24`,
/// `frac_09_24`), also `double` literals rounded to `float`.
#[allow(clippy::excessive_precision)]
const FRAC_01_24: [f32; 8] = [
    0.00576782_f64 as f32,
    -0.01831055_f64 as f32,
    0.01882935_f64 as f32,
    0.9328308_f64 as f32,
    0.09143066_f64 as f32,
    -0.04196167_f64 as f32,
    0.01296997_f64 as f32,
    -0.00140381_f64 as f32,
];
#[allow(clippy::excessive_precision)]
const FRAC_17_24: [f32; 8] = [
    -3.14331055e-03_f64 as f32,
    2.73437500e-02_f64 as f32,
    -1.06414795e-01_f64 as f32,
    3.64685059e-01_f64 as f32,
    8.03863525e-01_f64 as f32,
    -1.02233887e-01_f64 as f32,
    1.61437988e-02_f64 as f32,
    -1.22070312e-04_f64 as f32,
];
#[allow(clippy::excessive_precision)]
const FRAC_09_24: [f32; 8] = [
    -0.00146484_f64 as f32,
    0.02313232_f64 as f32,
    -0.12072754_f64 as f32,
    0.7315979_f64 as f32,
    0.4621277_f64 as f32,
    -0.12075806_f64 as f32,
    0.0295105_f64 as f32,
    -0.00326538_f64 as f32,
];

/// libopus `bbwe_feature_net` (`dnn/osce.c:855-954`): the conditioning
/// vectors of the `2·num_frames` subframes.
fn bbwe_feature_net(
    layers: &BbwenetLayers<'_>,
    st: &mut BbwenetState,
    output: &mut [f32; 4 * BBWENET_COND_DIM],
    features: &[f32],
    num_frames: usize,
    arch: DnnArch,
) {
    const C: usize = BBWENET_FNET_CONV1_OUT_SIZE;
    const T: usize = BBWENET_FNET_TCONV_OUT_CHANNELS * BBWENET_FNET_TCONV_STRIDE;
    let mut a = [0.0f32; 2 * C];
    let mut b = [0.0f32; 2 * C];
    for fr in 0..num_frames {
        compute_generic_conv1d(
            &layers.bbwenet_fnet_conv1,
            &mut a[fr * C..(fr + 1) * C],
            &mut st.feature_net_conv1_state,
            &features[fr * BBWENET_FEATURE_DIM..(fr + 1) * BBWENET_FEATURE_DIM],
            Activation::Tanh,
            arch,
        );
    }
    for fr in 0..num_frames {
        compute_generic_conv1d(
            &layers.bbwenet_fnet_conv2,
            &mut b[fr * C..(fr + 1) * C],
            &mut st.feature_net_conv2_state,
            &a[fr * C..(fr + 1) * C],
            Activation::Tanh,
            arch,
        );
    }
    let mut up = [0.0f32; 2 * T];
    for fr in 0..num_frames {
        compute_generic_dense(&layers.bbwenet_fnet_tconv, &mut up[fr * T..(fr + 1) * T], &b[fr * C..(fr + 1) * C], Activation::Tanh, arch);
    }
    const G: usize = BBWENET_FNET_GRU_STATE_SIZE;
    for sf in 0..BBWENET_FNET_TCONV_STRIDE * num_frames {
        compute_generic_gru(
            &layers.bbwenet_fnet_gru_input,
            &layers.bbwenet_fnet_gru_recurrent,
            &mut st.feature_net_gru_state,
            &up[sf * BBWENET_FNET_TCONV_OUT_CHANNELS..(sf + 1) * BBWENET_FNET_TCONV_OUT_CHANNELS],
            arch,
        );
        output[sf * G..(sf + 1) * G].copy_from_slice(&st.feature_net_gru_state);
    }
}

/// libopus `apply_valin_activation` (`dnn/osce.c:974-991`):
/// `x = x·celt_sin(celt_log(|x| + 1e-6))`, in place.
fn apply_valin_activation(x: &mut [f32]) {
    for v in x {
        let y = celt_log(v.abs() + 1e-6);
        *v *= celt_sin(y);
    }
}

/// libopus `interpol_3_2` (`dnn/osce.c:994-1039`): 3/2 resampling with three
/// 8-tap filters. Each output is `fma(b0, f0, b1·f1)`, then
/// `fma(b_j, f_j, ..)` for `j` = 2 to 7, as clang contracts the C sum.
fn interpol_3_2(st: &mut ResampState, x_out: &mut [f32], x_in: &[f32]) {
    let n = x_in.len();
    debug_assert!(n > 1 && n.is_multiple_of(2) && n <= F32);
    let mut buffer = [0.0f32; F32 + DELAY_SAMPLES];
    buffer[..DELAY_SAMPLES].copy_from_slice(&st.interpol_buffer);
    buffer[DELAY_SAMPLES..DELAY_SAMPLES + n].copy_from_slice(x_in);
    let tap8 = |b: &[f32], f: &[f32; 8]| {
        let mut s = fma(b[0], f[0], b[1] * f[1]);
        for j in 2..8 {
            s = fma(b[j], f[j], s);
        }
        s
    };
    for (k, i) in (0..n).step_by(2).enumerate() {
        x_out[3 * k] = tap8(&buffer[i..i + 8], &FRAC_01_24);
        x_out[3 * k + 1] = tap8(&buffer[i..i + 8], &FRAC_17_24);
        x_out[3 * k + 2] = tap8(&buffer[i + 1..i + 9], &FRAC_09_24);
    }
    st.interpol_buffer.copy_from_slice(&buffer[n..n + DELAY_SAMPLES]);
}

/// libopus `upsamp_2x` (`dnn/osce.c:1041-1097`): 2x upsampling with two
/// branches of three first-order all-pass sections. Every step is its own
/// C statement, so nothing is fused.
fn upsamp_2x(st: &mut ResampState, x_out: &mut [f32], x_in: &[f32]) {
    for (k, &x) in x_in.iter().enumerate() {
        for (branch, hq) in [&HQ_2X_EVEN, &HQ_2X_ODD].into_iter().enumerate() {
            let s = &mut st.upsamp_buffer[branch];
            let y = x - s[0];
            let xx = y * hq[0];
            let tmp1 = s[0] + xx;
            s[0] = x + xx;

            let y = tmp1 - s[1];
            let xx = y * hq[1];
            let tmp2 = s[1] + xx;
            s[1] = tmp1 + xx;

            let y = tmp2 - s[2];
            let xx = y * (1.0 + hq[2]);
            let tmp3 = s[2] + xx;
            s[2] = tmp2 + xx;

            x_out[2 * k + branch] = tmp3;
        }
    }
}

/// libopus `bbwenet_process_frames` (`dnn/osce.c:1099-1348`): extends
/// `num_frames` frames of 10 ms (160 samples per frame in `x_in`, in
/// [-1, 1]) to 48 kHz (480 samples per frame in `x_out`). `features` holds
/// 114 values per frame.
pub fn bbwenet_process_frames(
    layers: &BbwenetLayers<'_>,
    st: &mut BbwenetState,
    x_out: &mut [f32],
    x_in: &[f32],
    features: &[f32],
    num_frames: usize,
    arch: DnnArch,
) {
    assert!(num_frames == 1 || num_frames == 2);
    let num_subframes = 2 * num_frames;
    let mut latent = [0.0f32; 4 * BBWENET_COND_DIM];
    bbwe_feature_net(layers, st, &mut latent, features, num_frames, arch);

    for sf in 0..num_subframes {
        let cond = &latent[sf * BBWENET_COND_DIM..(sf + 1) * BBWENET_COND_DIM];

        // Adaptive filtering: one channel to three.
        let mut a1 = [0.0f32; 3 * F16];
        adaconv_process_frame(
            &mut st.af1_state,
            &mut a1,
            &x_in[sf * F16..(sf + 1) * F16],
            cond,
            &layers.bbwenet_af1_kernel,
            &layers.bbwenet_af1_gain,
            &AF1,
            &OVERLAP_WINDOW_40,
            arch,
        );

        // First non-linear extension at 32 kHz.
        let mut u2 = [0.0f32; 3 * F32];
        for c in 0..3 {
            upsamp_2x(&mut st.resampler_state[c], &mut u2[c * F32..(c + 1) * F32], &a1[c * F16..(c + 1) * F16]);
        }
        adashape_process_frame(
            &mut st.tdshape1_state,
            &mut u2[F32..2 * F32],
            cond,
            &layers.bbwenet_tdshape1_alpha1_f,
            &layers.bbwenet_tdshape1_alpha1_t,
            &layers.bbwenet_tdshape1_alpha2,
            &TDSHAPE1,
            arch,
        );
        apply_valin_activation(&mut u2[2 * F32..3 * F32]);

        // Mixing.
        let mut a2 = [0.0f32; 3 * F32];
        adaconv_process_frame(
            &mut st.af2_state,
            &mut a2,
            &u2,
            cond,
            &layers.bbwenet_af2_kernel,
            &layers.bbwenet_af2_gain,
            &AF2,
            &OVERLAP_WINDOW_80,
            arch,
        );

        // Second non-linear extension at 48 kHz.
        let mut u3 = [0.0f32; 3 * F48];
        for c in 0..3 {
            interpol_3_2(&mut st.resampler_state[c], &mut u3[c * F48..(c + 1) * F48], &a2[c * F32..(c + 1) * F32]);
        }
        adashape_process_frame(
            &mut st.tdshape2_state,
            &mut u3[F48..2 * F48],
            cond,
            &layers.bbwenet_tdshape2_alpha1_f,
            &layers.bbwenet_tdshape2_alpha1_t,
            &layers.bbwenet_tdshape2_alpha2,
            &TDSHAPE2,
            arch,
        );
        apply_valin_activation(&mut u3[2 * F48..3 * F48]);

        // Final mixing: three channels to one.
        adaconv_process_frame(
            &mut st.af3_state,
            &mut x_out[sf * F48..(sf + 1) * F48],
            &u3,
            cond,
            &layers.bbwenet_af3_kernel,
            &layers.bbwenet_af3_gain,
            &AF3,
            &OVERLAP_WINDOW_120,
            arch,
        );
    }
}
