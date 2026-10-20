//! The adaptive filters of libopus `dnn/nndsp.c`, which LACE, NoLACE and
//! BBWENet run on the signal: AdaConv (an adaptive convolution), AdaComb
//! (an adaptive comb filter around the pitch lag) and AdaShape (an adaptive
//! temporal envelope). A small network computes the filter of each 5 ms
//! subframe from the conditioning vector of the model; the signal itself
//! does not pass through a network.
//!
//! The roundings are those of libopus 1.6.1 on 64-bit ARM, built without
//! loop vectorization (read from the disassembly of the reference build):
//!
//! - the kernel norm is a chain of fused multiply-adds (`fma`); the scale
//!   factor `1/(1e-6 + sqrt(norm))` is computed in double precision;
//! - the gains are `exp` in double precision of a fused multiply-add;
//! - the overlap of the old and the new filter output uses the fused
//!   multiply-adds listed at each function;
//! - the correlations are one chain of fused multiply-adds per lag, as
//!   `celt_pitch_xcorr_float_neon` computes them.

use super::math::{celt_log, exp, fma, fma64, sqrt64};
use crate::dnn::nnet::{
    LinearLayer, compute_activation_in_place, compute_generic_conv1d, compute_generic_dense,
};
use crate::dnn::{Activation, DnnArch};

/// `ADACONV_MAX_KERNEL_SIZE`.
pub const ADACONV_MAX_KERNEL_SIZE: usize = 32;
/// `ADACONV_MAX_INPUT_CHANNELS`.
pub const ADACONV_MAX_INPUT_CHANNELS: usize = 3;
/// `ADACONV_MAX_OUTPUT_CHANNELS`.
pub const ADACONV_MAX_OUTPUT_CHANNELS: usize = 3;
/// `ADACONV_MAX_FRAME_SIZE`.
pub const ADACONV_MAX_FRAME_SIZE: usize = 240;
/// `ADACONV_MAX_OVERLAP_SIZE`.
pub const ADACONV_MAX_OVERLAP_SIZE: usize = 120;
/// `ADACOMB_MAX_LAG`.
pub const ADACOMB_MAX_LAG: usize = 300;
/// `ADACOMB_MAX_KERNEL_SIZE`.
pub const ADACOMB_MAX_KERNEL_SIZE: usize = 16;
/// `ADACOMB_MAX_FRAME_SIZE`.
pub const ADACOMB_MAX_FRAME_SIZE: usize = 80;
/// `ADACOMB_MAX_OVERLAP_SIZE`.
pub const ADACOMB_MAX_OVERLAP_SIZE: usize = 40;
/// `ADASHAPE_MAX_INPUT_DIM`.
pub const ADASHAPE_MAX_INPUT_DIM: usize = 512;
/// `ADASHAPE_MAX_FRAME_SIZE`.
pub const ADASHAPE_MAX_FRAME_SIZE: usize = 240;

/// The largest memory of the first AdaShape layer (`alpha1_f`): the
/// feature dimension of NoLACE, 160.
const ADASHAPE_MAX_ALPHA1F_STATE: usize = 160;
/// The memory of the second AdaShape layer (`alpha1_t`): one envelope of
/// 20 values plus its mean.
const ADASHAPE_ALPHA1T_STATE: usize = 21;
/// The largest memory of the third AdaShape layer (`alpha2`): 120 for the
/// second shaping layer of BBWENet.
const ADASHAPE_MAX_ALPHA2_STATE: usize = 120;

/// The state of one AdaConv filter (libopus `AdaConvState`), sized for one
/// model: `HIST = kernel_size · in_channels` input samples of the previous
/// subframe and `KERN = kernel_size · in_channels · out_channels` taps of
/// the previous kernel. libopus also has a field `last_gain` that it never
/// uses.
#[derive(Clone)]
pub struct AdaConvState<const HIST: usize, const KERN: usize> {
    /// The last `kernel_size` input samples of each input channel.
    pub history: [f32; HIST],
    /// The kernel of the previous subframe, `[out][in][tap]`.
    pub last_kernel: [f32; KERN],
}

impl<const HIST: usize, const KERN: usize> Default for AdaConvState<HIST, KERN> {
    /// The cleared state of `init_adaconv_state`.
    fn default() -> Self {
        AdaConvState { history: [0.0; HIST], last_kernel: [0.0; KERN] }
    }
}

/// The state of one AdaComb filter (libopus `AdaCombState`).
#[derive(Clone)]
pub struct AdaCombState {
    /// The last `kernel_size + ADACOMB_MAX_LAG` input samples.
    pub history: [f32; ADACOMB_MAX_KERNEL_SIZE + ADACOMB_MAX_LAG],
    /// The kernel of the previous subframe.
    pub last_kernel: [f32; ADACOMB_MAX_KERNEL_SIZE],
    /// The global gain of the previous subframe.
    pub last_global_gain: f32,
    /// The pitch lag of the previous subframe.
    pub last_pitch_lag: usize,
}

impl Default for AdaCombState {
    /// The cleared state of `init_adacomb_state`.
    fn default() -> Self {
        AdaCombState {
            history: [0.0; ADACOMB_MAX_KERNEL_SIZE + ADACOMB_MAX_LAG],
            last_kernel: [0.0; ADACOMB_MAX_KERNEL_SIZE],
            last_global_gain: 0.0,
            last_pitch_lag: 0,
        }
    }
}

/// The state of one AdaShape filter (libopus `AdaShapeState`): the
/// memories of its three convolution layers and the last value of the
/// interpolation. libopus sizes the memories 512, 512 and 240; the 1.6.1
/// models need at most 160, 21 and 120.
#[derive(Clone)]
pub struct AdaShapeState {
    /// The memory of `alpha1_f` (the feature branch).
    pub conv_alpha1f_state: [f32; ADASHAPE_MAX_ALPHA1F_STATE],
    /// The memory of `alpha1_t` (the envelope branch).
    pub conv_alpha1t_state: [f32; ADASHAPE_ALPHA1T_STATE],
    /// The memory of `alpha2`.
    pub conv_alpha2_state: [f32; ADASHAPE_MAX_ALPHA2_STATE],
    /// The last output of `alpha2`, the start of the next interpolation.
    pub interpolate_state: f32,
}

impl Default for AdaShapeState {
    /// The cleared state of `init_adashape_state`.
    fn default() -> Self {
        AdaShapeState {
            conv_alpha1f_state: [0.0; ADASHAPE_MAX_ALPHA1F_STATE],
            conv_alpha1t_state: [0.0; ADASHAPE_ALPHA1T_STATE],
            conv_alpha2_state: [0.0; ADASHAPE_MAX_ALPHA2_STATE],
            interpolate_state: 0.0,
        }
    }
}

/// The constant arguments of one AdaConv filter (the `*_AF*_` constants of
/// the model headers).
#[derive(Clone, Copy, Debug)]
pub struct AdaConvParams {
    /// Samples per subframe and channel.
    pub frame_size: usize,
    /// Samples at the start of the subframe where the old and the new
    /// kernel are cross-faded.
    pub overlap_size: usize,
    /// Input channels.
    pub in_channels: usize,
    /// Output channels.
    pub out_channels: usize,
    /// Taps of the kernel.
    pub kernel_size: usize,
    /// Must be `kernel_size - 1` (a causal filter), as libopus asserts.
    pub left_padding: usize,
    /// `filter_gain_a` of `transform_gains`.
    pub filter_gain_a: f32,
    /// `filter_gain_b` of `transform_gains`.
    pub filter_gain_b: f32,
    /// Must be 1, as libopus asserts.
    pub shape_gain: f32,
}

/// The constant arguments of one AdaComb filter.
#[derive(Clone, Copy, Debug)]
pub struct AdaCombParams {
    /// Samples per subframe (at most `ADACOMB_MAX_FRAME_SIZE`).
    pub frame_size: usize,
    /// Samples of the cross-fade (at most `ADACOMB_MAX_OVERLAP_SIZE`).
    pub overlap_size: usize,
    /// Taps of the kernel (at most `ADACOMB_MAX_KERNEL_SIZE`).
    pub kernel_size: usize,
    /// Taps of the kernel before the pitch lag.
    pub left_padding: usize,
    /// `filter_gain_a` of the global gain.
    pub filter_gain_a: f32,
    /// `filter_gain_b` of the global gain.
    pub filter_gain_b: f32,
    /// The largest log gain of the kernel.
    pub log_gain_limit: f32,
}

/// The constant arguments of one AdaShape filter.
#[derive(Clone, Copy, Debug)]
pub struct AdaShapeParams {
    /// The length of the conditioning vector.
    pub feature_dim: usize,
    /// Samples per subframe.
    pub frame_size: usize,
    /// Samples averaged into one value of the envelope.
    pub avg_pool_k: usize,
    /// The upsampling factor of the shape (`frame_size / interpolate_k`
    /// values are computed, then interpolated).
    pub interpolate_k: usize,
}

/// libopus `compute_overlap_window` (`dnn/nndsp.c:63-70`): the raised
/// cosine `0.5 + 0.5·cos(π·(i + 0.5)/n)`, computed in double precision
/// with a fused multiply-add and rounded to float.
///
/// The models use the tables `OVERLAP_WINDOW_40`, `_80` and `_120`, which
/// hold the values of this function computed with the C library of macOS
/// on 64-bit ARM (the unit tests check them on 64-bit ARM macOS).
pub fn compute_overlap_window(window: &mut [f32]) {
    let n = window.len();
    for (i, w) in window.iter_mut().enumerate() {
        let angle = core::f64::consts::PI * (i as f32 + 0.5) as f64 / n as f64;
        *w = fma64(super::math::cos(angle), 0.5, 0.5) as f32;
    }
}

/// The overlap window of 40 samples: LACE, NoLACE and the first filter of
/// BBWENet. Values of
/// `compute_overlap_window` as `f32` bits.
pub static OVERLAP_WINDOW_40: [f32; 40] = window_from_bits(&WINDOW_40_BITS);
/// The overlap window of 80 samples (the second filter of BBWENet).
pub static OVERLAP_WINDOW_80: [f32; 80] = window_from_bits(&WINDOW_80_BITS);
/// The overlap window of 120 samples (the third filter of BBWENet).
pub static OVERLAP_WINDOW_120: [f32; 120] = window_from_bits(&WINDOW_120_BITS);

const fn window_from_bits<const N: usize>(bits: &[u32; N]) -> [f32; N] {
    let mut w = [0.0f32; N];
    let mut i = 0;
    while i < N {
        w[i] = f32::from_bits(bits[i]);
        i += 1;
    }
    w
}

/// `compute_overlap_window` for 40 samples, as `f32` bits.
const WINDOW_40_BITS: [u32; 40] = [
    0x3f7fe6bd, 0x3f7f1cde, 0x3f7d8a5f, 0x3f7b31bc, 0x3f7816a7, 0x3f743e09, 0x3f6fadf3,
    0x3f6a6d99, 0x3f648544, 0x3f5dfe48, 0x3f56e2f1, 0x3f4f3e78, 0x3f471ced, 0x3f3e8b24,
    0x3f3596a4, 0x3f2c4d90, 0x3f22be90, 0x3f18f8b8, 0x3f0f0b77, 0x3f050677, 0x3ef5f312,
    0x3ee1e912, 0x3ece0e90, 0x3eba82e1, 0x3ea764df, 0x3e94d2b7, 0x3e82e9b8, 0x3e638c4c,
    0x3e43061e, 0x3e24743b, 0x3e0806e1, 0x3ddbd5e1, 0x3dac933b, 0x3d829068, 0x3d3c1f6f,
    0x3cfd2b15, 0x3c99c88a, 0x3c1d6830, 0x3b6321ff, 0x39ca1a81,
];
/// `compute_overlap_window` for 80 samples, as `f32` bits.
const WINDOW_80_BITS: [u32; 80] = [
    0x3f7ff9af, 0x3f7fc72b, 0x3f7f6237, 0x3f7ecafa, 0x3f7e01b1, 0x3f7d06aa, 0x3f7bda49,
    0x3f7a7d05, 0x3f78ef68, 0x3f77320d, 0x3f7545a6, 0x3f732af4, 0x3f70e2cc, 0x3f6e6e15,
    0x3f6bcdc6, 0x3f6902ea, 0x3f660e9a, 0x3f62f202, 0x3f5fae5a, 0x3f5c44ee, 0x3f58b716,
    0x3f550639, 0x3f5133cd, 0x3f4d4152, 0x3f493059, 0x3f45027c, 0x3f40b961, 0x3f3c56ba,
    0x3f37dc42, 0x3f334bbd, 0x3f2ea6f8, 0x3f29efc9, 0x3f25280c, 0x3f2051a5, 0x3f1b6e7b,
    0x3f16807e, 0x3f11899f, 0x3f0c8bd3, 0x3f078914, 0x3f02835b, 0x3efaf949, 0x3ef0edd8,
    0x3ee6e859, 0x3edcecc2, 0x3ed2ff04, 0x3ec92309, 0x3ebf5cb6, 0x3eb5afe7, 0x3eac206e,
    0x3ea2b210, 0x3e996886, 0x3e90477c, 0x3e87528b, 0x3e7d1a7a, 0x3e6bf610, 0x3e5b3e9c,
    0x3e4afab7, 0x3e3b30ce, 0x3e2be71b, 0x3e1d23a7, 0x3e0eec47, 0x3e014697, 0x3de86ff3,
    0x3dcf8b2d, 0x3db7e8af, 0x3da191ce, 0x3d8c8f5b, 0x3d71d344, 0x3d4d50c6, 0x3d2ba5a6,
    0x3d0cdf2f, 0x3ce2130e, 0x3cb05f55, 0x3c84b6d0, 0x3c3e557a, 0x3bff27b5, 0x3b9a830e,
    0x3b1dc971, 0x3a635477, 0x38ca1f7d,
];
/// `compute_overlap_window` for 120 samples, as `f32` bits.
const WINDOW_120_BITS: [u32; 120] = [
    0x3f7ffd31, 0x3f7fe6bd, 0x3f7fb9d7, 0x3f7f7689, 0x3f7f1cde, 0x3f7eace6, 0x3f7e26b3,
    0x3f7d8a5f, 0x3f7cd804, 0x3f7c0fc2, 0x3f7b31bc, 0x3f7a3e18, 0x3f793502, 0x3f7816a7,
    0x3f76e33b, 0x3f759af3, 0x3f743e09, 0x3f72ccba, 0x3f714746, 0x3f6fadf3, 0x3f6e0107,
    0x3f6c40cf, 0x3f6a6d99, 0x3f6887b6, 0x3f668f7c, 0x3f648544, 0x3f626969, 0x3f603c49,
    0x3f5dfe48, 0x3f5bafc8, 0x3f595133, 0x3f56e2f1, 0x3f546572, 0x3f51d923, 0x3f4f3e78,
    0x3f4c95e7, 0x3f49dfe5, 0x3f471ced, 0x3f444d7b, 0x3f41720d, 0x3f3e8b24, 0x3f3b9942,
    0x3f389cea, 0x3f3596a4, 0x3f3286f7, 0x3f2f6e6d, 0x3f2c4d90, 0x3f2924ee, 0x3f25f513,
    0x3f22be90, 0x3f1f81f3, 0x3f1c3fd0, 0x3f18f8b8, 0x3f15ad3f, 0x3f125df7, 0x3f0f0b77,
    0x3f0bb653, 0x3f085f21, 0x3f050677, 0x3f01aceb, 0x3efca629, 0x3ef5f312, 0x3eef41be,
    0x3ee8935a, 0x3ee1e912, 0x3edb4411, 0x3ed4a583, 0x3ece0e90, 0x3ec7805f, 0x3ec0fc19,
    0x3eba82e1, 0x3eb415da, 0x3eadb625, 0x3ea764df, 0x3ea12326, 0x3e9af211, 0x3e94d2b7,
    0x3e8ec62b, 0x3e88cd7d, 0x3e82e9b8, 0x3e7a37cb, 0x3e6eca14, 0x3e638c4c, 0x3e58806d,
    0x3e4da866, 0x3e43061e, 0x3e389b73, 0x3e2e6a3a, 0x3e24743b, 0x3e1abb35, 0x3e1140df,
    0x3e0806e1, 0x3dfe1db5, 0x3decb4ba, 0x3ddbd5e1, 0x3dcb841f, 0x3dbbc251, 0x3dac933b,
    0x3d9df987, 0x3d8ff7c4, 0x3d829068, 0x3d6b8b9a, 0x3d533462, 0x3d3c1f6f, 0x3d2650cb,
    0x3d11cc4c, 0x3cfd2b15, 0x3cd95fcb, 0x3cb83d02, 0x3c99c88a, 0x3c7c0f76, 0x3c49fee7,
    0x3c1d6830, 0x3beca647, 0x3ba98d3b, 0x3b6321ff, 0x3b0976d1, 0x3a8c514f, 0x39ca1a81,
    0x3833ab09,
];

/// `celt_pitch_xcorr` as libopus computes it on 64-bit ARM
/// (`celt_pitch_xcorr_float_neon`): `out[l] = Σ_j x[j]·y[l + j]` for each
/// lag `l`, one chain of fused multiply-adds per lag, starting from +0.
///
/// libopus passes a kernel of `ADACONV_MAX_KERNEL_SIZE` (32) taps whose
/// taps past `kernel_size` are zero. A zero tap adds `±0` to a sum that is
/// never `-0` (it starts at `+0`, and an exact cancellation rounds to
/// `+0`), so it changes no result; only the real taps are computed here.
#[inline]
fn xcorr(x: &[f32], y: &[f32], out: &mut [f32]) {
    let len = x.len();
    let lags = out.len();
    assert!(y.len() + 1 >= len + lags);
    #[cfg(target_arch = "aarch64")]
    if lags.is_multiple_of(4) {
        // The NEON kernel of the codec computes these bits when the lag
        // count is a multiple of 4 (`pitch.rs`, test
        // `pitch_xcorr_neon_follows_libopus_for_every_count_of_lags`).
        crate::pitch::pitch_xcorr(x, y, out, len, lags);
        return;
    }
    for (l, o) in out.iter_mut().enumerate() {
        let mut s = 0.0f32;
        for (j, &xj) in x.iter().enumerate() {
            s = fma(xj, y[l + j], s);
        }
        *o = s;
    }
}

/// libopus `transform_gains` (`dnn/nndsp.c:119-131`):
/// `gain = exp(a·gain + b)`, the exponential in double precision.
fn transform_gains(gains: &mut [f32], a: f32, b: f32) {
    for g in gains {
        *g = exp(fma(a, *g, b) as f64) as f32;
    }
}

/// libopus `scale_kernel` (`dnn/nndsp.c:82-117`): normalizes the kernel of
/// each output channel to unit 2-norm over its input channels and taps,
/// then multiplies it by the gain of that output channel.
fn scale_kernel(kernel: &mut [f32], in_channels: usize, out_channels: usize, kernel_size: usize, gain: &[f32]) {
    let per_out = in_channels * kernel_size;
    for o in 0..out_channels {
        let k = &mut kernel[o * per_out..(o + 1) * per_out];
        let mut norm = 0.0f32;
        for &v in k.iter() {
            norm = fma(v, v, norm);
        }
        // `1.f / (1e-6f + sqrt(norm))`: double precision, rounded to float.
        let norm = (1.0 / (1e-6f32 as f64 + sqrt64(norm as f64))) as f32;
        let factor = norm * gain[o];
        for v in k.iter_mut() {
            *v *= factor;
        }
    }
}

/// libopus `adaconv_process_frame` (`dnn/nndsp.c:133-241`): one subframe
/// of an adaptive convolution with `in_channels` inputs and
/// `out_channels` outputs.
///
/// `x_in` holds `in_channels` blocks of `frame_size` samples and `x_out`
/// receives `out_channels` blocks. `features` is the conditioning vector
/// of the subframe. The kernel is computed from it (`kernel_layer`, linear)
/// together with one gain per output channel (`gain_layer`, tanh, then
/// `exp(a·g + b)`), and normalized. Over the first `overlap_size` samples
/// the outputs of the previous and of the new kernel are cross-faded with
/// `window`: `out = fma(w, old, out)`, then `out = fma(1 - w, new, out)`,
/// summed over the input channels in order; after the overlap
/// `out += new`.
///
/// libopus calls it in place for LACE; the input is copied first here too,
/// so a caller with one buffer copies it into `x_in` before the call.
#[allow(clippy::too_many_arguments)]
pub fn adaconv_process_frame<const HIST: usize, const KERN: usize>(
    st: &mut AdaConvState<HIST, KERN>,
    x_out: &mut [f32],
    x_in: &[f32],
    features: &[f32],
    kernel_layer: &LinearLayer,
    gain_layer: &LinearLayer,
    p: &AdaConvParams,
    window: &[f32],
    arch: DnnArch,
) {
    let (fs, ov, ic, oc, ks) = (p.frame_size, p.overlap_size, p.in_channels, p.out_channels, p.kernel_size);
    assert!(p.shape_gain == 1.0, "AdaConv supports only a shape gain of 1");
    assert!(p.left_padding + 1 == ks, "AdaConv supports only causal kernels");
    assert!(ks < fs && fs <= ADACONV_MAX_FRAME_SIZE && ov <= fs && ov <= ADACONV_MAX_OVERLAP_SIZE);
    assert!(ks <= ADACONV_MAX_KERNEL_SIZE && ic <= ADACONV_MAX_INPUT_CHANNELS && oc <= ADACONV_MAX_OUTPUT_CHANNELS);
    assert!(HIST >= ks * ic && KERN >= ks * ic * oc);
    let window = &window[..ov];

    // Each input channel: the last `kernel_size` samples, then the subframe.
    let seg = ks + fs;
    let mut input = [0.0f32; ADACONV_MAX_INPUT_CHANNELS * (ADACONV_MAX_FRAME_SIZE + ADACONV_MAX_KERNEL_SIZE)];
    for c in 0..ic {
        input[c * seg..c * seg + ks].copy_from_slice(&st.history[c * ks..(c + 1) * ks]);
        input[c * seg + ks..(c + 1) * seg].copy_from_slice(&x_in[c * fs..(c + 1) * fs]);
    }

    // The new kernel and gains.
    let mut kernel = [0.0f32; ADACONV_MAX_KERNEL_SIZE * ADACONV_MAX_INPUT_CHANNELS * ADACONV_MAX_OUTPUT_CHANNELS];
    let mut gain = [0.0f32; ADACONV_MAX_OUTPUT_CHANNELS];
    compute_generic_dense(kernel_layer, &mut kernel, features, Activation::Linear, arch);
    compute_generic_dense(gain_layer, &mut gain, features, Activation::Tanh, arch);
    transform_gains(&mut gain[..oc], p.filter_gain_a, p.filter_gain_b);
    scale_kernel(&mut kernel, ic, oc, ks, &gain);

    let x_out = &mut x_out[..oc * fs];
    x_out.fill(0.0);
    let mut old = [0.0f32; ADACONV_MAX_OVERLAP_SIZE];
    let mut new = [0.0f32; ADACONV_MAX_FRAME_SIZE];
    for o in 0..oc {
        let out = &mut x_out[o * fs..(o + 1) * fs];
        for c in 0..ic {
            let k = (o * ic + c) * ks;
            // `p_input + c·(frame + kernel) - left_padding`.
            let y = &input[c * seg + ks - p.left_padding..(c + 1) * seg];
            xcorr(&st.last_kernel[k..k + ks], y, &mut old[..ov]);
            xcorr(&kernel[k..k + ks], y, &mut new[..fs]);
            for i in 0..ov {
                out[i] = fma(window[i], old[i], out[i]);
                out[i] = fma(1.0 - window[i], new[i], out[i]);
            }
            for i in ov..fs {
                out[i] += new[i];
            }
        }
    }

    for c in 0..ic {
        st.history[c * ks..(c + 1) * ks].copy_from_slice(&input[c * seg + fs..(c + 1) * seg]);
    }
    st.last_kernel[..ks * ic * oc].copy_from_slice(&kernel[..ks * ic * oc]);
}

/// libopus `adacomb_process_frame` (`dnn/nndsp.c:243-336`): one subframe
/// of an adaptive comb filter `out = g·(x + Σ_j k[j]·x[n - lag - pad + j])`,
/// in place on `x` (`frame_size` samples).
///
/// The kernel comes from `kernel_layer` (linear), normalized and scaled by
/// `exp(log_gain_limit - relu(gain_layer))`; the global gain is
/// `exp(a·tanh(global_gain_layer) + b)`. Over the overlap, with `w` the
/// window, `lg` and `g` the old and new global gains, `old` and `new` the
/// comb outputs of the old and new kernels and lags:
/// `out = fma(lg·w, old, (g·(1 - w))·new)`, then
/// `out = fma(fma(w, lg, (1 - w)·g), x, out)`. After it,
/// `out = g·(new + x)`.
///
/// `pitch_lag` must be at most `ADACOMB_MAX_LAG + kernel_size - left_padding`
/// (OSCE passes lags of 7 to 288).
#[allow(clippy::too_many_arguments)]
pub fn adacomb_process_frame(
    st: &mut AdaCombState,
    x: &mut [f32],
    features: &[f32],
    kernel_layer: &LinearLayer,
    gain_layer: &LinearLayer,
    global_gain_layer: &LinearLayer,
    pitch_lag: usize,
    p: &AdaCombParams,
    window: &[f32],
    arch: DnnArch,
) {
    let (fs, ov, ks, lp) = (p.frame_size, p.overlap_size, p.kernel_size, p.left_padding);
    assert!(fs <= ADACOMB_MAX_FRAME_SIZE && ov <= fs && ks <= ADACOMB_MAX_KERNEL_SIZE);
    let hist = ks + ADACOMB_MAX_LAG;
    assert!(lp + pitch_lag <= hist && lp + st.last_pitch_lag <= hist, "AdaComb pitch lag too long");
    let window = &window[..ov];
    let x = &mut x[..fs];

    let mut input = [0.0f32; ADACOMB_MAX_FRAME_SIZE + ADACOMB_MAX_LAG + ADACOMB_MAX_KERNEL_SIZE];
    input[..hist].copy_from_slice(&st.history[..hist]);
    input[hist..hist + fs].copy_from_slice(x);
    // `p_input` is `input[hist..]`.

    let mut kernel = [0.0f32; ADACOMB_MAX_KERNEL_SIZE];
    let mut gain = [0.0f32; 1];
    let mut global_gain = [0.0f32; 1];
    compute_generic_dense(kernel_layer, &mut kernel, features, Activation::Linear, arch);
    compute_generic_dense(gain_layer, &mut gain, features, Activation::Relu, arch);
    compute_generic_dense(global_gain_layer, &mut global_gain, features, Activation::Tanh, arch);
    let gain = exp((p.log_gain_limit - gain[0]) as f64) as f32;
    let global_gain = exp(fma(p.filter_gain_a, global_gain[0], p.filter_gain_b) as f64) as f32;
    scale_kernel(&mut kernel, 1, 1, ks, &[gain]);

    let mut old = [0.0f32; ADACOMB_MAX_OVERLAP_SIZE];
    let mut new = [0.0f32; ADACOMB_MAX_FRAME_SIZE];
    let old_start = hist - lp - st.last_pitch_lag;
    xcorr(&st.last_kernel[..ks], &input[old_start..old_start + ks + ov - 1], &mut old[..ov]);
    let new_start = hist - lp - pitch_lag;
    xcorr(&kernel[..ks], &input[new_start..new_start + ks + fs - 1], &mut new[..fs]);

    let lg = st.last_global_gain;
    let xin = &input[hist..hist + fs];
    for i in 0..ov {
        let w = window[i];
        let mixed = fma(lg * w, old[i], (global_gain * (1.0 - w)) * new[i]);
        x[i] = fma(fma(w, lg, (1.0 - w) * global_gain), xin[i], mixed);
    }
    for i in ov..fs {
        x[i] = global_gain * (new[i] + xin[i]);
    }

    st.last_kernel[..ks].copy_from_slice(&kernel[..ks]);
    st.history[..hist].copy_from_slice(&input[fs..fs + hist]);
    st.last_pitch_lag = pitch_lag;
    st.last_global_gain = global_gain;
}

/// libopus `adashape_process_frame` (`dnn/nndsp.c:339-441`): multiplies one
/// subframe of `x` (`frame_size` samples), in place, by an adaptive
/// envelope.
///
/// The envelope: the log of the mean absolute value over blocks of
/// `avg_pool_k` samples (`celt_log` of `fma(sum, 1/k, 2^-16)`), minus its
/// mean, with the mean appended; two convolutions (`alpha1_f` on the
/// features, `alpha1_t` on the envelope), their sum through a leaky ReLU
/// (`0.2·t` in double precision for `t < 0`), a third convolution
/// (`alpha2`), linear interpolation by `interpolate_k`
/// (`fma(α, t, (1 - α)·prev)`), and the polynomial exponential of the
/// layers.
#[allow(clippy::too_many_arguments)]
pub fn adashape_process_frame(
    st: &mut AdaShapeState,
    x: &mut [f32],
    features: &[f32],
    alpha1f: &LinearLayer,
    alpha1t: &LinearLayer,
    alpha2: &LinearLayer,
    p: &AdaShapeParams,
    arch: DnnArch,
) {
    let (fd, fs, pool, ik) = (p.feature_dim, p.frame_size, p.avg_pool_k, p.interpolate_k);
    assert!(fs <= ADASHAPE_MAX_FRAME_SIZE && fs.is_multiple_of(pool) && fs.is_multiple_of(ik));
    assert!(fd + fs / pool + 1 < ADASHAPE_MAX_INPUT_DIM);
    let x = &mut x[..fs];
    let hidden_dim = fs / ik;
    let tenv_size = fs / pool;
    let f = 1.0f32 / pool as f32;

    let mut in_buffer = [0.0f32; ADASHAPE_MAX_INPUT_DIM + ADASHAPE_MAX_FRAME_SIZE];
    let mut out_buffer = [0.0f32; ADASHAPE_MAX_FRAME_SIZE];
    let mut tmp_buffer = [0.0f32; ADASHAPE_MAX_FRAME_SIZE];
    in_buffer[..fd].copy_from_slice(&features[..fd]);

    // The temporal envelope, after the features in `in_buffer`.
    {
        let tenv = &mut in_buffer[fd..fd + tenv_size + 1];
        let mut mean = 0.0f32;
        for i in 0..tenv_size {
            let mut t = 0.0f32;
            for &v in &x[i * pool..(i + 1) * pool] {
                t += v.abs();
            }
            // `1.52587890625e-05f`, 2^-16.
            tenv[i] = celt_log(fma(t, f, 1.0 / 65536.0));
            mean += tenv[i];
        }
        mean /= tenv_size as f32;
        for t in &mut tenv[..tenv_size] {
            *t -= mean;
        }
        tenv[tenv_size] = mean;
    }

    // The temporal weights.
    compute_generic_conv1d(alpha1f, &mut out_buffer, &mut st.conv_alpha1f_state, &in_buffer[..fd], Activation::Linear, arch);
    compute_generic_conv1d(
        alpha1t,
        &mut tmp_buffer,
        &mut st.conv_alpha1t_state,
        &in_buffer[fd..fd + tenv_size + 1],
        Activation::Linear,
        arch,
    );
    // Leaky ReLU, by hand as in libopus.
    for i in 0..hidden_dim {
        let t = out_buffer[i] + tmp_buffer[i];
        in_buffer[i] = if t >= 0.0 { t } else { (0.2 * t as f64) as f32 };
    }
    compute_generic_conv1d(alpha2, &mut tmp_buffer, &mut st.conv_alpha2_state, &in_buffer[..hidden_dim], Activation::Linear, arch);

    // Upsampling by linear interpolation.
    for i in 0..hidden_dim {
        for k in 0..ik {
            let alpha = (k + 1) as f32 / ik as f32;
            out_buffer[i * ik + k] = fma(alpha, tmp_buffer[i], (1.0 - alpha) * st.interpolate_state);
        }
        st.interpolate_state = tmp_buffer[i];
    }

    compute_activation_in_place(&mut out_buffer[..fs], Activation::Exp, arch);
    for (v, &g) in x.iter_mut().zip(out_buffer.iter()) {
        *v *= g;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The tables hold the values of `compute_overlap_window`. libopus
    /// computes them with the C library's `cos`; Rust's `f64::cos` calls
    /// the same library on macOS.
    #[cfg(all(feature = "std", target_arch = "aarch64", target_os = "macos"))]
    #[test]
    fn the_window_tables_are_compute_overlap_window() {
        let mut w = [0.0f32; 120];
        for (n, table) in [(40, &OVERLAP_WINDOW_40[..]), (80, &OVERLAP_WINDOW_80[..]), (120, &OVERLAP_WINDOW_120[..])] {
            compute_overlap_window(&mut w[..n]);
            for i in 0..n {
                assert_eq!(w[i].to_bits(), table[i].to_bits(), "window {n}, value {i}");
            }
        }
    }

    #[test]
    fn the_window_tables_are_close_to_the_raised_cosine() {
        for table in [&OVERLAP_WINDOW_40[..], &OVERLAP_WINDOW_80[..], &OVERLAP_WINDOW_120[..]] {
            let n = table.len();
            let mut w = [0.0f32; 120];
            compute_overlap_window(&mut w[..n]);
            for i in 0..n {
                assert!((w[i] - table[i]).abs() <= f32::EPSILON, "window {n}, value {i}");
            }
        }
    }

    /// On 64-bit ARM, `pitch_xcorr` must give the bits of the plain chain
    /// of fused multiply-adds for the lag counts that OSCE uses.
    #[test]
    fn xcorr_is_one_chain_of_fused_multiply_adds_per_lag() {
        let mut seed = 1u32;
        let mut next = || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (seed >> 8) as f32 / (1u32 << 24) as f32 - 0.5
        };
        for &taps in &[16usize, 32] {
            for &lags in &[40usize, 80, 120, 160, 240] {
                let x: [f32; 32] = core::array::from_fn(|_| next());
                let y: [f32; 280] = core::array::from_fn(|_| next());
                let mut got = [0.0f32; 240];
                xcorr(&x[..taps], &y[..taps + lags - 1], &mut got[..lags]);
                for l in 0..lags {
                    let mut s = 0.0f32;
                    for j in 0..taps {
                        s = fma(x[j], y[l + j], s);
                    }
                    assert_eq!(got[l].to_bits(), s.to_bits(), "taps {taps}, lags {lags}, lag {l}");
                }
            }
        }
    }
}
