//! The feature analysis of deep PLC (libopus `dnn/lpcnet_enc.c`, the part
//! that `lpcnet_compute_single_frame_features_float` runs).
//!
//! Each 10 ms frame at 16 kHz gives 36 values (`NB_TOTAL_FEATURES`):
//!
//! - 0 to 17: the cepstrum of the band energies (an 18-band DCT of their
//!   base-10 logarithms, minus 4 in the first value);
//! - 18: the pitch from PitchDNN, `(1/60)·bin - 1.5`, where the period in
//!   samples is `256 / 2^(value + 1.5)`;
//! - 19: the pitch correlation, mapped to [-0.5, 0.5];
//! - 20 to 35: the LPC coefficients computed from the cepstrum.
//!
//! The state keeps the overlap of the analysis window, the spectrum of the
//! last frame (for the instantaneous frequency inputs of PitchDNN), the
//! LPC residual and its low-passed copy over the last 26 ms (for the
//! cross-correlation inputs and the pitch correlation), and PitchDNN's own
//! state.
//!
//! # Rounding
//!
//! See `super` and `crate::dnn::freq`. The libopus functions that this code
//! calls from the CELT folder are rewritten here with the rounding of the
//! 64-bit ARM build: `celt_inner_prod` (`celt_inner_prod_neon`: four lanes
//! of fused multiply-adds, added as `(l0 + l2) + (l1 + l3)`),
//! `celt_pitch_xcorr` (`celt_pitch_xcorr_float_neon`: one chain of fused
//! multiply-adds per lag), `celt_fir` (`celt_fir_c` with fused
//! multiply-adds) and `celt_log2` (with `FLOAT_APPROX` and fused
//! multiply-adds). On 64-bit ARM, every `DnnArch` but `Portable` uses the
//! NEON versions of `crate::pitch` for the first two, which give the same
//! bits; `DnnArch::Portable` and other targets use the plain versions here.

#[cfg(not(feature = "std"))]
use crate::compat::Math;
use crate::dnn::arch::DnnArch;
use crate::dnn::freq::{
    self, FRAME_SIZE, FREQ_SIZE, LPC_ORDER, NB_BANDS, OVERLAP_SIZE, PREEMPHASIS, TRAINING_OFFSET,
    WINDOW_SIZE, fmaf,
};
use crate::dnn::models::pitchdnn::PitchDnn;
use crate::kiss_fft::KissCpx;

use super::pitchdnn::{PitchDnnState, compute_pitchdnn};
use super::{
    NB_FEATURES, NB_TOTAL_FEATURES, NB_XCORR_FEATURES, PITCH_IF_FEATURES, PITCH_IF_MAX_FREQ,
    PITCH_MAX_PERIOD, max_c, min_c, period_from_pitch,
};

/// The part of `exc_buf` and `lp_buf` that libopus uses: the last
/// `PITCH_MAX_PERIOD` samples and the current frame. (libopus declares
/// `PITCH_BUF_SIZE`, 576 values; the last 160 are never touched.)
const PITCH_BUF_USED: usize = PITCH_MAX_PERIOD + FRAME_SIZE;

/// `0.3010299957f`, the factor of `celt_log10` (`dnn/lpcnet_enc.c:107`),
/// written by its bits.
const LOG10_2: f32 = f32::from_bits(0x3e9a_209b);

/// `log(1+exp(5.f))` (`dnn/lpcnet_enc.c:194`), as clang computes it at
/// compile time.
const LOG_1P_EXP5: f64 = f64::from_bits(0x4014_06e0_6368_916a);

/// The low-pass filter of the pitch correlation, `[b,a]=ellip(2, 2, 20,
/// 1200/8000)` (`dnn/lpcnet_enc.c:125-126`).
const LP_B: [f32; 2] = [-0.84946, 1.0];
/// See `LP_B`.
const LP_A: [f32; 2] = [-1.54220, 0.70781];

/// The state of the feature analysis (libopus `LPCNetEncState`,
/// `dnn/lpcnet_private.h:25-42`), without the fields that deep PLC never
/// uses (`sig_mem`, `burg_cepstrum`) and without the PitchDNN weights,
/// which the functions take as an argument.
#[derive(Clone)]
pub struct LpcnetEncState {
    /// PitchDNN's recurrent and convolution state.
    pitchdnn: PitchDnnState,
    /// The last `OVERLAP_SIZE` input samples (the first half of the next
    /// analysis window).
    analysis_mem: [f32; OVERLAP_SIZE],
    /// The memory of the pre-emphasis filter.
    mem_preemph: f32,
    /// The first `PITCH_IF_MAX_FREQ` bins of the last frame's spectrum.
    prev_if: [KissCpx; PITCH_IF_MAX_FREQ],
    /// The instantaneous frequency inputs of PitchDNN.
    if_features: [f32; PITCH_IF_FEATURES],
    /// The cross-correlation inputs of PitchDNN.
    xcorr_features: [f32; NB_XCORR_FEATURES],
    /// The last output of PitchDNN.
    dnn_pitch: f32,
    /// The last `LPC_ORDER` samples of the delayed input (the memory of
    /// the LPC analysis filter).
    pitch_mem: [f32; LPC_ORDER],
    /// The last LPC residual sample.
    pitch_filt: f32,
    /// The LPC residual with a slight pitch emphasis, last 26 ms.
    exc_buf: [f32; PITCH_BUF_USED],
    /// The low-passed LPC residual, last 26 ms.
    lp_buf: [f32; PITCH_BUF_USED],
    /// The memory of the low-pass filter. (libopus declares 4 values and
    /// uses 2.)
    lp_mem: [f32; 2],
    /// The LPC coefficients of the last frame.
    lpc: [f32; LPC_ORDER],
    /// The features of the last frame.
    features: [f32; NB_TOTAL_FEATURES],
}

impl Default for LpcnetEncState {
    fn default() -> Self {
        Self::new()
    }
}

impl LpcnetEncState {
    /// A cleared state (libopus `lpcnet_encoder_init`, without the model
    /// initialization).
    pub const fn new() -> Self {
        LpcnetEncState {
            pitchdnn: PitchDnnState::new(),
            analysis_mem: [0.0; OVERLAP_SIZE],
            mem_preemph: 0.0,
            prev_if: [KissCpx { r: 0.0, i: 0.0 }; PITCH_IF_MAX_FREQ],
            if_features: [0.0; PITCH_IF_FEATURES],
            xcorr_features: [0.0; NB_XCORR_FEATURES],
            dnn_pitch: 0.0,
            pitch_mem: [0.0; LPC_ORDER],
            pitch_filt: 0.0,
            exc_buf: [0.0; PITCH_BUF_USED],
            lp_buf: [0.0; PITCH_BUF_USED],
            lp_mem: [0.0; 2],
            lpc: [0.0; LPC_ORDER],
            features: [0.0; NB_TOTAL_FEATURES],
        }
    }

    /// Clears the state in place (libopus `lpcnet_encoder_init`).
    pub fn reset(&mut self) {
        self.pitchdnn.reset();
        self.analysis_mem.fill(0.0);
        self.mem_preemph = 0.0;
        self.prev_if.fill(KissCpx::default());
        self.if_features.fill(0.0);
        self.xcorr_features.fill(0.0);
        self.dnn_pitch = 0.0;
        self.pitch_mem.fill(0.0);
        self.pitch_filt = 0.0;
        self.exc_buf.fill(0.0);
        self.lp_buf.fill(0.0);
        self.lp_mem.fill(0.0);
        self.lpc.fill(0.0);
        self.features.fill(0.0);
    }

    /// The features of the last frame.
    pub fn features(&self) -> &[f32; NB_TOTAL_FEATURES] {
        &self.features
    }

    /// The instantaneous frequency inputs of PitchDNN of the last frame.
    pub fn if_features(&self) -> &[f32; PITCH_IF_FEATURES] {
        &self.if_features
    }

    /// The cross-correlation inputs of PitchDNN of the last frame.
    pub fn xcorr_features(&self) -> &[f32; NB_XCORR_FEATURES] {
        &self.xcorr_features
    }

    /// The last output of PitchDNN.
    pub fn dnn_pitch(&self) -> f32 {
        self.dnn_pitch
    }

    /// PitchDNN's state.
    pub fn pitchdnn(&self) -> &PitchDnnState {
        &self.pitchdnn
    }
}

/// `celt_log2` of the float build with `FLOAT_APPROX` (`celt/mathops.h`),
/// with the polynomial in fused multiply-adds as on 64-bit ARM. Like
/// libopus, it does not handle 0, negative values, infinities or NaN (it
/// returns some finite or NaN value for them).
#[allow(clippy::excessive_precision)]
#[inline]
fn log2_approx(x: f32) -> f32 {
    const X_NORM: [f32; 8] = [
        1.0,
        8.888_888_955_116_272e-1,
        8.0e-1,
        7.272_727_489_471_436e-1,
        6.666_666_865_348_816e-1,
        6.153_846_383_094_788e-1,
        5.714_285_969_734_192e-1,
        5.333_333_611_488_342e-1,
    ];
    const Y_NORM: [f32; 8] = [
        0.0,
        1.699_250_042_438_507e-1,
        3.219_280_838_966_369_6e-1,
        4.594_316_184_520_721_4e-1,
        5.849_624_872_207_642e-1,
        7.004_396_915_435_791e-1,
        8.073_549_270_629_883e-1,
        9.068_905_711_174_011e-1,
    ];
    let bits = x.to_bits() as i32;
    let integer = (bits >> 23).wrapping_sub(127);
    let bits = bits.wrapping_sub(((integer as u32) << 23) as i32);
    let range_idx = ((bits >> 20) & 0x7) as usize;
    let m = fmaf(f32::from_bits(bits as u32), X_NORM[range_idx], -1.0625);
    let p = fmaf(m, -2.841_544_449_329_376e-1, 4.019_712_507_724_762e-1);
    let p = fmaf(m, p, -6.389_770_507_812_5e-1);
    let p = fmaf(m, p, 1.357_829_570_770_263_7);
    let p = fmaf(m, p, 8.746_284_246_444_702e-2);
    integer as f32 + p + Y_NORM[range_idx]
}

/// `MAX16(-1.f, MIN16(1.f, (1.f/64)*(10.f*celt_log10(e)-6.f)))`: one
/// log-energy input of PitchDNN. `10*x - 6` is one fused multiply-add.
#[inline]
fn if_log_energy(e: f32) -> f32 {
    let v = (1.0f32 / 64.0) * fmaf(10.0, LOG10_2 * log2_approx(e), -6.0);
    max_c(-1.0, min_c(1.0, v))
}

/// `celt_inner_prod_neon` of the float build (`celt/arm/pitch_neon_intr.c`):
/// four running sums of fused multiply-adds over the samples `4k + l`,
/// added as `(s0 + s2) + (s1 + s3)`, then the last `n % 4` products one at
/// a time. `y` must be at least as long as `x`.
pub fn inner_prod_portable(x: &[f32], y: &[f32]) -> f32 {
    let n = x.len();
    let y = &y[..n];
    let mut s = [0.0f32; 4];
    let blocks = n / 4;
    for k in 0..blocks {
        for l in 0..4 {
            s[l] = fmaf(x[4 * k + l], y[4 * k + l], s[l]);
        }
    }
    let mut xy = (s[0] + s[2]) + (s[1] + s[3]);
    for i in 4 * blocks..n {
        xy = fmaf(x[i], y[i], xy);
    }
    xy
}

/// `celt_pitch_xcorr_float_neon` (`celt/arm/celt_neon_intr.c`) for a count
/// of lags that is a multiple of 4: `xcorr[k]` is one chain of fused
/// multiply-adds over `x[j]·y[k + j]`, `j` from 0, starting from 0. `y`
/// must hold `x.len() + xcorr.len() - 1` samples.
pub fn pitch_xcorr_portable(x: &[f32], y: &[f32], xcorr: &mut [f32]) {
    let len = x.len();
    debug_assert!(xcorr.len().is_multiple_of(4));
    let y = &y[..len + xcorr.len() - 1];
    for (k, out) in xcorr.iter_mut().enumerate() {
        let yk = &y[k..k + len];
        let mut s = 0.0f32;
        for (&a, &b) in x.iter().zip(yk) {
            s = fmaf(b, a, s);
        }
        *out = s;
    }
}

/// `celt_inner_prod` as the reference build runs it, with the NEON kernel
/// of `crate::pitch` where it gives those bits.
#[inline]
fn inner_prod(x: &[f32], y: &[f32], arch: DnnArch) -> f32 {
    #[cfg(target_arch = "aarch64")]
    if arch != DnnArch::Portable {
        return crate::pitch::celt_inner_prod(x, y);
    }
    let _ = arch;
    inner_prod_portable(x, y)
}

/// `celt_pitch_xcorr` as the reference build runs it, with the NEON kernel
/// of `crate::pitch` where it gives those bits.
#[inline]
fn pitch_xcorr(x: &[f32], y: &[f32], xcorr: &mut [f32], arch: DnnArch) {
    #[cfg(target_arch = "aarch64")]
    if arch != DnnArch::Portable {
        let lags = xcorr.len();
        crate::pitch::pitch_xcorr(x, y, xcorr, x.len(), lags);
        return;
    }
    let _ = arch;
    pitch_xcorr_portable(x, y, xcorr);
}

/// `celt_fir_c` (`celt/celt_lpc.c`) of the float build for `y.len()`
/// outputs: `y[i] = x[ord + i] + sum_j num[ord-1-j]·x[i + j]`, the sum one
/// chain of fused multiply-adds from the oldest sample. `x` holds `ord`
/// samples of memory and then the input.
fn celt_fir(x: &[f32], num: &[f32; LPC_ORDER], y: &mut [f32]) {
    let ord = LPC_ORDER;
    let mut rnum = [0.0f32; LPC_ORDER];
    for (i, r) in rnum.iter_mut().enumerate() {
        *r = num[ord - 1 - i];
    }
    for (i, out) in y.iter_mut().enumerate() {
        let hist = &x[i..i + ord];
        let mut sum = x[i + ord];
        for (&r, &h) in rnum.iter().zip(hist) {
            sum = fmaf(r, h, sum);
        }
        *out = sum;
    }
}

/// `biquad` (`dnn/lpcnet_enc.c:84-105`) in place, with the coefficients of
/// `LP_B` and `LP_A`: the form libopus rewrote to shorten its dependency
/// chains, `mem0 = (b0-a0)·x + mem1 - a0·mem0` and
/// `mem1 = (b1-a1)·x + 1e-30 - a1·mem0`, each two fused multiply-adds.
fn biquad(buf: &mut [f32], mem: &mut [f32; 2]) {
    // clang folds the differences in `float`.
    let b0a0 = LP_B[0] - LP_A[0];
    let b1a1 = LP_B[1] - LP_A[1];
    let (mut mem0, mut mem1) = (mem[0], mem[1]);
    for v in buf.iter_mut() {
        let xi = *v;
        let yi = xi + mem0;
        let mem00 = mem0;
        mem0 = fmaf(-LP_A[0], mem0, fmaf(b0a0, xi, mem1));
        mem1 = fmaf(-LP_A[1], mem00, fmaf(b1a1, xi, 1e-30));
        *v = yi;
    }
    *mem = [mem0, mem1];
}

/// `frame_analysis` (`dnn/lpcnet_enc.c:74-82`): the windowed spectrum of
/// the last two frames and its band energies.
fn frame_analysis(
    st: &mut LpcnetEncState,
    x_spec: &mut [KissCpx; FREQ_SIZE],
    ex: &mut [f32; NB_BANDS],
    input: &[f32; FRAME_SIZE],
) {
    let mut x = [0.0f32; WINDOW_SIZE];
    x[..OVERLAP_SIZE].copy_from_slice(&st.analysis_mem);
    x[OVERLAP_SIZE..].copy_from_slice(input);
    st.analysis_mem.copy_from_slice(&input[FRAME_SIZE - OVERLAP_SIZE..]);
    freq::apply_window(&mut x);
    freq::forward_transform(x_spec, &x);
    freq::lpcn_compute_band_energy(ex, x_spec);
}

/// `compute_frame_features` (`dnn/lpcnet_enc.c:109-197`): the features of
/// one frame of pre-emphasized 16 kHz audio (scaled like 16-bit samples).
/// They are left in the state (`LpcnetEncState::features`).
pub fn compute_frame_features(
    st: &mut LpcnetEncState,
    input: &[f32; FRAME_SIZE],
    pitchdnn: &PitchDnn,
    arch: DnnArch,
) {
    let mut aligned_in = [0.0f32; FRAME_SIZE];
    aligned_in[..TRAINING_OFFSET]
        .copy_from_slice(&st.analysis_mem[OVERLAP_SIZE - TRAINING_OFFSET..]);
    let mut x_spec = [KissCpx::default(); FREQ_SIZE];
    let mut ex = [0.0f32; NB_BANDS];
    frame_analysis(st, &mut x_spec, &mut ex, input);

    // The instantaneous frequency inputs of PitchDNN: the log energy of
    // bin 0, then for bins 1 to 29 the phase change since the last frame
    // (as a unit complex number) and the log energy.
    st.if_features[0] = if_log_energy(fmaf(x_spec[0].r, x_spec[0].r, 1e-15));
    for i in 1..PITCH_IF_MAX_FREQ {
        let a = x_spec[i];
        let b = st.prev_if[i];
        // `C_MULC(prod, X[i], prev_if[i])`; the left product of each line
        // is fused.
        let pr = fmaf(a.r, b.r, a.i * b.i);
        let pi = fmaf(a.i, b.r, -(a.r * b.i));
        // `1.f/sqrt(1e-15f + pr*pr + pi*pi)`: the sum in `float`, the
        // square root and the division in `double`.
        let n = fmaf(pi, pi, fmaf(pr, pr, 1e-15));
        let norm_1 = (1.0 / (n as f64).sqrt()) as f32;
        st.if_features[3 * i - 2] = pr * norm_1;
        st.if_features[3 * i - 1] = pi * norm_1;
        st.if_features[3 * i] = if_log_energy(fmaf(a.i, a.i, fmaf(a.r, a.r, 1e-15)));
    }
    st.prev_if.copy_from_slice(&x_spec[..PITCH_IF_MAX_FREQ]);

    // The cepstrum of the band energies, with a floor 8 below the largest
    // band and a decay of 2.5 per band from the previous band.
    let mut ly = [0.0f32; NB_BANDS];
    let mut log_max = -2.0f32;
    let mut follow = -2.0f32;
    for i in 0..NB_BANDS {
        let l = LOG10_2 * log2_approx(1e-2 + ex[i]);
        let l = max_c(log_max - 8.0, max_c(follow - 2.5, l));
        ly[i] = l;
        log_max = max_c(log_max, l);
        follow = max_c(follow - 2.5, l);
    }
    let mut cepstrum = [0.0f32; NB_BANDS];
    freq::dct(&mut cepstrum, &ly);
    cepstrum[0] -= 4.0;
    st.features[..NB_BANDS].copy_from_slice(&cepstrum);
    freq::lpc_from_cepstrum(&mut st.lpc, &cepstrum);
    st.features[NB_BANDS + 2..].copy_from_slice(&st.lpc);

    // The LPC residual of the input, delayed by TRAINING_OFFSET samples,
    // and its low-passed copy.
    st.exc_buf.copy_within(FRAME_SIZE.., 0);
    st.lp_buf.copy_within(FRAME_SIZE.., 0);
    aligned_in[TRAINING_OFFSET..].copy_from_slice(&input[..FRAME_SIZE - TRAINING_OFFSET]);
    let mut x = [0.0f32; FRAME_SIZE + LPC_ORDER];
    x[..LPC_ORDER].copy_from_slice(&st.pitch_mem);
    x[LPC_ORDER..].copy_from_slice(&aligned_in);
    st.pitch_mem.copy_from_slice(&aligned_in[FRAME_SIZE - LPC_ORDER..]);
    celt_fir(&x, &st.lpc, &mut st.lp_buf[PITCH_MAX_PERIOD..]);
    for i in 0..FRAME_SIZE {
        let lp = st.lp_buf[PITCH_MAX_PERIOD + i];
        st.exc_buf[PITCH_MAX_PERIOD + i] = fmaf(0.7, st.pitch_filt, lp);
        st.pitch_filt = lp;
    }
    biquad(&mut st.lp_buf[PITCH_MAX_PERIOD..], &mut st.lp_mem);

    // The cross-correlation inputs of PitchDNN: the normalized correlation
    // of the residual with itself at the lags 0 to 223 before the frame.
    {
        let buf = &st.exc_buf;
        let cur = &buf[PITCH_MAX_PERIOD..];
        let mut xcorr = [0.0f32; NB_XCORR_FEATURES];
        pitch_xcorr(cur, buf, &mut xcorr, arch);
        let ener0 = inner_prod(cur, cur, arch);
        let mut ener1 = inner_prod(&buf[..FRAME_SIZE], &buf[..FRAME_SIZE], arch) as f64;
        let mut ener_norm = [0.0f32; NB_XCORR_FEATURES];
        for i in 0..NB_XCORR_FEATURES {
            // `1 + ener0 + ener1`: the first sum in `float`, the second in
            // `double`.
            ener_norm[i] = ((1.0 + ener0) as f64 + ener1) as f32;
            st.xcorr_features[i] = 2.0 * xcorr[i];
            // `buf[i+160]*(double)buf[i+160] - buf[i]*(double)buf[i]`: the
            // left product is fused.
            let a = buf[i + FRAME_SIZE] as f64;
            let b = buf[i] as f64;
            ener1 += freq::fma(a, a, -(b * b));
        }
        for (v, &e) in st.xcorr_features.iter_mut().zip(ener_norm.iter()) {
            *v /= e;
        }
    }
    st.dnn_pitch = compute_pitchdnn(
        &mut st.pitchdnn,
        &st.if_features,
        &st.xcorr_features,
        pitchdnn,
        arch,
    );

    // The pitch correlation of the low-passed residual at the period of
    // PitchDNN. The period lies in [32, 256] for every output of PitchDNN
    // but NaN, which gives 0; the clamp only keeps the indexes in the
    // buffer for values libopus never produces.
    let pitch = period_from_pitch(st.dnn_pitch).clamp(0, PITCH_MAX_PERIOD as i32) as usize;
    let lp = &st.lp_buf;
    let cur = &lp[PITCH_MAX_PERIOD..];
    let past = &lp[PITCH_MAX_PERIOD - pitch..PITCH_MAX_PERIOD - pitch + FRAME_SIZE];
    let xx = inner_prod(cur, cur, arch);
    let yy = inner_prod(past, past, arch);
    let xy = inner_prod(cur, past, arch);
    // `xy/sqrt(1+xx*yy)`: `1+xx*yy` is a fused multiply-add in `float`,
    // the rest is in `double`.
    let frame_corr = (xy as f64 / (fmaf(xx, yy, 1.0) as f64).sqrt()) as f32;
    // `log(1.f+exp(5.f*frame_corr))/log(1+exp(5.f))`.
    let frame_corr = ((1.0 + ((5.0 * frame_corr) as f64).exp()).ln() / LOG_1P_EXP5) as f32;
    st.features[NB_BANDS] = st.dnn_pitch;
    st.features[NB_BANDS + 1] = frame_corr - 0.5;
}

/// `preemphasis` (`dnn/lpcnet_enc.c:199-207`) in place:
/// `y[i] = x[i] + mem`, `mem = -coef·x[i]`.
pub fn preemphasis(x: &mut [f32], mem: &mut f32, coef: f32) {
    for v in x.iter_mut() {
        let yi = *v + *mem;
        *mem = -coef * *v;
        *v = yi;
    }
}

/// `lpcnet_compute_single_frame_features_float` (`dnn/lpcnet_enc.c:224-230`):
/// the features of one 10 ms frame of 16 kHz audio, scaled like 16-bit
/// samples (full scale is ±32768). The input is pre-emphasized with
/// `PREEMPHASIS` first.
pub fn compute_single_frame_features(
    st: &mut LpcnetEncState,
    pcm: &[f32; FRAME_SIZE],
    features: &mut [f32; NB_TOTAL_FEATURES],
    pitchdnn: &PitchDnn,
    arch: DnnArch,
) {
    let mut x = *pcm;
    preemphasis(&mut x, &mut st.mem_preemph, PREEMPHASIS);
    compute_frame_features(st, &x, pitchdnn, arch);
    *features = st.features;
}

// `NB_FEATURES` features come first in `features`; the rest are the LPC
// coefficients.
const _: () = assert!(NB_FEATURES + LPC_ORDER == NB_TOTAL_FEATURES);
const _: () = assert!(NB_FEATURES == NB_BANDS + 2);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constants_have_the_bits_of_the_reference() {
        // From the assembly of the reference build (`dnn_lpcnet_enc.s`).
        assert_eq!(LOG10_2.to_bits(), 0x3e9a_209b);
        assert_eq!((LP_B[0] - LP_A[0]).to_bits(), 0x3f31_5768);
        assert_eq!((LP_B[1] - LP_A[1]).to_bits(), 0x3e95_99ee);
        assert_eq!((-LP_A[0]).to_bits(), 0x3fc5_66cf);
        assert_eq!((-LP_A[1]).to_bits(), 0xbf35_3309);
        assert_eq!(1e-30f32.to_bits(), 0x0da2_4260);
        assert_eq!(0.7f32.to_bits(), 0x3f33_3333);
    }

    #[test]
    fn log2_approx_matches_the_celt_one_on_aarch64() {
        // On 64-bit ARM `mathops::celt_log2` fuses the same steps.
        #[cfg(target_arch = "aarch64")]
        {
            let mut x = 1e-15f32;
            while x < 1e12 {
                assert_eq!(log2_approx(x).to_bits(), crate::mathops::celt_log2(x).to_bits(), "{x}");
                x *= 1.013;
            }
        }
        assert!((log2_approx(8.0) - 3.0).abs() < 1e-6);
    }
}
