//! FARGAN, the vocoder of deep PLC (libopus `dnn/fargan.c`,
//! `dnn/fargan.h`).
//!
//! FARGAN turns one vector of `NB_FEATURES` features into 10 ms of 16 kHz
//! audio, in four subframes of 2.5 ms. A conditioning network runs once per
//! frame on the features and an embedding of the pitch period. Each
//! subframe then runs three GRUs and several gated layers on the
//! conditioning, on the last subframe and on a pitch prediction read one
//! period back in its own past output (`pitch_buf`). The output is
//! pre-emphasized; a de-emphasis filter follows.
//!
//! Before the first frame, `fargan_cont` primes the state with 20 ms of
//! past audio and five feature vectors.
//!
//! The model (`crate::dnn::models::fargan::Fargan`) and the kernel variant
//! are arguments of each function; the state holds only what changes.

#[cfg(not(feature = "std"))]
use crate::compat::Math;
use crate::dnn::Activation;
use crate::dnn::arch::DnnArch;
use crate::dnn::freq::{NB_BANDS, fmaf};
use crate::dnn::models::fargan::{
    COND_NET_FCONV1_IN_SIZE, COND_NET_FCONV1_OUT_SIZE, COND_NET_FCONV1_STATE_SIZE,
    COND_NET_FDENSE2_OUT_SIZE, COND_NET_PEMBED_OUT_SIZE, Fargan, SIG_NET_FWC0_CONV_OUT_SIZE,
    SIG_NET_FWC0_GLU_GATE_OUT_SIZE, SIG_NET_GRU1_OUT_SIZE, SIG_NET_GRU1_STATE_SIZE,
    SIG_NET_GRU2_OUT_SIZE, SIG_NET_GRU2_STATE_SIZE, SIG_NET_GRU3_OUT_SIZE,
    SIG_NET_GRU3_STATE_SIZE, SIG_NET_SKIP_DENSE_OUT_SIZE,
};
use crate::dnn::nnet::{
    compute_generic_conv1d, compute_generic_dense, compute_generic_gru, compute_glu,
    compute_glu_in_place,
};

use super::{LPCNET_FRAME_SIZE, NB_FEATURES, PITCH_MAX_PERIOD, max_c, min_c, period_from_pitch};

/// `FARGAN_CONT_SAMPLES`: the past audio that `fargan_cont` reads.
pub const FARGAN_CONT_SAMPLES: usize = 320;
/// `FARGAN_NB_SUBFRAMES`.
pub const FARGAN_NB_SUBFRAMES: usize = 4;
/// `FARGAN_SUBFRAME_SIZE`.
pub const FARGAN_SUBFRAME_SIZE: usize = 40;
/// `FARGAN_FRAME_SIZE`: 10 ms at 16 kHz.
pub const FARGAN_FRAME_SIZE: usize = FARGAN_NB_SUBFRAMES * FARGAN_SUBFRAME_SIZE;
/// `FARGAN_COND_SIZE`: the conditioning values of one subframe.
pub const FARGAN_COND_SIZE: usize = COND_NET_FDENSE2_OUT_SIZE / FARGAN_NB_SUBFRAMES;
/// `FARGAN_DEEMPHASIS`.
pub const FARGAN_DEEMPHASIS: f32 = 0.85;
/// `SIG_NET_INPUT_SIZE`: the input of the first subframe layer: the
/// conditioning, the pitch prediction (44 values) and the last subframe.
pub const SIG_NET_INPUT_SIZE: usize = FARGAN_COND_SIZE + 2 * FARGAN_SUBFRAME_SIZE + 4;

/// The pitch prediction of a subframe: 2 samples before it, the subframe,
/// 2 after.
const PRED_SIZE: usize = FARGAN_SUBFRAME_SIZE + 4;

/// The input of `sig_net_skip_dense`: the outputs of the three GRU gates,
/// of the first gate, the gated pitch prediction and the last subframe.
const SKIP_CAT_SIZE: usize = SIG_NET_GRU1_OUT_SIZE
    + SIG_NET_GRU2_OUT_SIZE
    + SIG_NET_GRU3_OUT_SIZE
    + SIG_NET_FWC0_CONV_OUT_SIZE
    + 2 * FARGAN_SUBFRAME_SIZE;

const _: () = assert!(FARGAN_FRAME_SIZE == LPCNET_FRAME_SIZE);
const _: () = assert!(SKIP_CAT_SIZE == 688);

/// The state of FARGAN (libopus `FARGANState`, `dnn/fargan.h:45-57`)
/// without the model and the kernel variant.
///
/// `DeepPlc::reset` does not clear it, as `lpcnet_plc_reset` does not: the
/// GRU states, the convolution memories and the start of `pitch_buf`
/// carry over from one loss to the next and across a decoder reset.
#[derive(Clone)]
pub struct FarganState {
    /// Whether `fargan_cont` ran.
    cont_initialized: bool,
    /// The memory of the de-emphasis filter.
    deemph_mem: f32,
    /// The last 256 output samples, before de-emphasis.
    pitch_buf: [f32; PITCH_MAX_PERIOD],
    /// The memory of `cond_net_fconv1`.
    cond_conv1_state: [f32; COND_NET_FCONV1_STATE_SIZE],
    /// The memory of `sig_net_fwc0_conv`. (libopus declares twice this
    /// size and uses this much.)
    fwc0_mem: [f32; SIG_NET_INPUT_SIZE],
    /// The state of the first GRU.
    gru1_state: [f32; SIG_NET_GRU1_STATE_SIZE],
    /// The state of the second GRU.
    gru2_state: [f32; SIG_NET_GRU2_STATE_SIZE],
    /// The state of the third GRU.
    gru3_state: [f32; SIG_NET_GRU3_STATE_SIZE],
    /// The pitch period of the previous frame, which this frame's
    /// subframes use.
    last_period: i32,
}

impl Default for FarganState {
    fn default() -> Self {
        Self::new()
    }
}

impl FarganState {
    /// A cleared state (libopus `fargan_init` without the model).
    pub const fn new() -> Self {
        FarganState {
            cont_initialized: false,
            deemph_mem: 0.0,
            pitch_buf: [0.0; PITCH_MAX_PERIOD],
            cond_conv1_state: [0.0; COND_NET_FCONV1_STATE_SIZE],
            fwc0_mem: [0.0; SIG_NET_INPUT_SIZE],
            gru1_state: [0.0; SIG_NET_GRU1_STATE_SIZE],
            gru2_state: [0.0; SIG_NET_GRU2_STATE_SIZE],
            gru3_state: [0.0; SIG_NET_GRU3_STATE_SIZE],
            last_period: 0,
        }
    }

    /// Whether `fargan_cont` ran since the state was made.
    pub fn cont_initialized(&self) -> bool {
        self.cont_initialized
    }

    /// The memory of the de-emphasis filter.
    pub fn deemph_mem(&self) -> f32 {
        self.deemph_mem
    }

    /// The last 256 output samples before de-emphasis.
    pub fn pitch_buf(&self) -> &[f32; PITCH_MAX_PERIOD] {
        &self.pitch_buf
    }

    /// The memory of the conditioning convolution.
    pub fn cond_conv1_state(&self) -> &[f32; COND_NET_FCONV1_STATE_SIZE] {
        &self.cond_conv1_state
    }

    /// The memory of the first subframe layer.
    pub fn fwc0_mem(&self) -> &[f32; SIG_NET_INPUT_SIZE] {
        &self.fwc0_mem
    }

    /// The states of the three GRUs.
    pub fn gru_states(&self) -> (&[f32], &[f32], &[f32]) {
        (&self.gru1_state, &self.gru2_state, &self.gru3_state)
    }

    /// The pitch period of the last frame.
    pub fn last_period(&self) -> i32 {
        self.last_period
    }

    /// `pitch_buf[i]` as libopus reads it. For `i` past the end it returns
    /// what libopus reads there: the next field of `FARGANState`,
    /// `cond_conv1_state` (`dnn/fargan.h:51-52`). Only a period of 0 makes
    /// `run_fargan_subframe` read past the end (`dnn/fargan.c:92-96`); the
    /// PLC network and the feature analysis never give one, but DRED
    /// features can.
    #[inline]
    fn pitch_buf_at(&self, i: usize) -> f32 {
        if i < PITCH_MAX_PERIOD {
            self.pitch_buf[i]
        } else {
            self.cond_conv1_state.get(i - PITCH_MAX_PERIOD).copied().unwrap_or(0.0)
        }
    }
}

/// `compute_fargan_cond` (`dnn/fargan.c:43-59`): the conditioning of one
/// frame, `FARGAN_COND_SIZE` values per subframe.
fn compute_fargan_cond(
    st: &mut FarganState,
    cond: &mut [f32; COND_NET_FDENSE2_OUT_SIZE],
    features: &[f32; NB_FEATURES],
    period: i32,
    model: &Fargan,
    arch: DnnArch,
) {
    let mut dense_in = [0.0f32; NB_FEATURES + COND_NET_PEMBED_OUT_SIZE];
    let mut conv1_in = [0.0f32; COND_NET_FCONV1_IN_SIZE];
    let mut fdense2_in = [0.0f32; COND_NET_FCONV1_OUT_SIZE];
    // The pitch embedding: row `clamp(period - 32, 0, 223)` of the table.
    // (The loader checks the size of the table; a row it does not hold
    // stays zero.)
    let row = period.saturating_sub(32).clamp(0, 223) as usize;
    let embed = &mut dense_in[NB_FEATURES..];
    if let Some(table) = model.cond_net_pembed.float_weights
        && let Some(r) =
            table.get(row * COND_NET_PEMBED_OUT_SIZE..(row + 1) * COND_NET_PEMBED_OUT_SIZE)
    {
        embed.copy_from_slice(r);
    }
    dense_in[..NB_FEATURES].copy_from_slice(features);
    compute_generic_dense(
        &model.cond_net_fdense1,
        &mut conv1_in,
        &dense_in,
        Activation::Tanh,
        arch,
    );
    compute_generic_conv1d(
        &model.cond_net_fconv1,
        &mut fdense2_in,
        &mut st.cond_conv1_state,
        &conv1_in,
        Activation::Tanh,
        arch,
    );
    compute_generic_dense(&model.cond_net_fdense2, cond, &fdense2_in, Activation::Tanh, arch);
}

/// `fargan_deemphasis` (`dnn/fargan.c:61-67`): `pcm[i] += 0.85·mem`, one
/// fused multiply-add per sample.
fn fargan_deemphasis(pcm: &mut [f32; FARGAN_SUBFRAME_SIZE], deemph_mem: &mut f32) {
    for v in pcm.iter_mut() {
        *v = fmaf(FARGAN_DEEMPHASIS, *deemph_mem, *v);
        *deemph_mem = *v;
    }
}

/// `run_fargan_subframe` (`dnn/fargan.c:69-140`): one subframe of output,
/// de-emphasized, with the pitch period `period`.
fn run_fargan_subframe(
    st: &mut FarganState,
    pcm: &mut [f32; FARGAN_SUBFRAME_SIZE],
    cond: &[f32],
    period: i32,
    model: &Fargan,
    arch: DnnArch,
) {
    debug_assert!(st.cont_initialized, "fargan_cont must run before the first subframe");
    let cond = &cond[..FARGAN_COND_SIZE];

    let mut gain = [0.0f32; 1];
    compute_generic_dense(
        &model.sig_net_cond_gain_dense,
        &mut gain,
        cond,
        Activation::Linear,
        arch,
    );
    let gain = (gain[0] as f64).exp() as f32;
    let gain_1 = 1.0 / (1e-5 + gain);

    // The pitch prediction: the past output one period back, wrapped at
    // the end of the buffer, scaled by 1/gain and clamped to [-1, 1].
    let mut pred = [0.0f32; PRED_SIZE];
    let mut pos = (PITCH_MAX_PERIOD as i32 - 2).saturating_sub(period);
    for p in pred.iter_mut() {
        let v = gain_1 * st.pitch_buf_at(pos.max(0) as usize);
        *p = min_c(1.0, max_c(-1.0, v));
        pos = pos.saturating_add(1);
        if pos == PITCH_MAX_PERIOD as i32 {
            pos -= period;
        }
    }
    let mut prev = [0.0f32; FARGAN_SUBFRAME_SIZE];
    for (i, p) in prev.iter_mut().enumerate() {
        let v = gain_1 * st.pitch_buf[PITCH_MAX_PERIOD - FARGAN_SUBFRAME_SIZE + i];
        *p = max_c(-1.0, min_c(1.0, v));
    }

    let mut fwc0_in = [0.0f32; SIG_NET_INPUT_SIZE];
    fwc0_in[..FARGAN_COND_SIZE].copy_from_slice(cond);
    fwc0_in[FARGAN_COND_SIZE..FARGAN_COND_SIZE + PRED_SIZE].copy_from_slice(&pred);
    fwc0_in[FARGAN_COND_SIZE + PRED_SIZE..].copy_from_slice(&prev);

    let mut gru1_in = [0.0f32; SIG_NET_FWC0_CONV_OUT_SIZE + 2 * FARGAN_SUBFRAME_SIZE];
    let mut gru2_in = [0.0f32; SIG_NET_GRU1_OUT_SIZE + 2 * FARGAN_SUBFRAME_SIZE];
    let mut gru3_in = [0.0f32; SIG_NET_GRU2_OUT_SIZE + 2 * FARGAN_SUBFRAME_SIZE];
    let mut skip_cat = [0.0f32; SKIP_CAT_SIZE];
    let mut skip_out = [0.0f32; SIG_NET_SKIP_DENSE_OUT_SIZE];
    let mut pitch_gate = [0.0f32; 4];

    compute_generic_conv1d(
        &model.sig_net_fwc0_conv,
        &mut gru1_in,
        &mut st.fwc0_mem,
        &fwc0_in,
        Activation::Tanh,
        arch,
    );
    compute_glu_in_place(&model.sig_net_fwc0_glu_gate, &mut gru1_in, arch);
    compute_generic_dense(
        &model.sig_net_gain_dense_out,
        &mut pitch_gate,
        &gru1_in,
        Activation::Sigmoid,
        arch,
    );

    /// `in[n..n+40] = gate·pred[2..42]`, `in[n+40..n+80] = prev`.
    fn append(buf: &mut [f32], n: usize, gate: f32, pred: &[f32; PRED_SIZE], prev: &[f32]) {
        for i in 0..FARGAN_SUBFRAME_SIZE {
            buf[n + i] = gate * pred[i + 2];
        }
        buf[n + FARGAN_SUBFRAME_SIZE..n + 2 * FARGAN_SUBFRAME_SIZE].copy_from_slice(prev);
    }

    append(&mut gru1_in, SIG_NET_FWC0_GLU_GATE_OUT_SIZE, pitch_gate[0], &pred, &prev);
    compute_generic_gru(
        &model.sig_net_gru1_input,
        &model.sig_net_gru1_recurrent,
        &mut st.gru1_state,
        &gru1_in,
        arch,
    );
    compute_glu(&model.sig_net_gru1_glu_gate, &mut gru2_in, &st.gru1_state, arch);

    append(&mut gru2_in, SIG_NET_GRU1_OUT_SIZE, pitch_gate[1], &pred, &prev);
    compute_generic_gru(
        &model.sig_net_gru2_input,
        &model.sig_net_gru2_recurrent,
        &mut st.gru2_state,
        &gru2_in,
        arch,
    );
    compute_glu(&model.sig_net_gru2_glu_gate, &mut gru3_in, &st.gru2_state, arch);

    append(&mut gru3_in, SIG_NET_GRU2_OUT_SIZE, pitch_gate[2], &pred, &prev);
    compute_generic_gru(
        &model.sig_net_gru3_input,
        &model.sig_net_gru3_recurrent,
        &mut st.gru3_state,
        &gru3_in,
        arch,
    );
    const G12: usize = SIG_NET_GRU1_OUT_SIZE + SIG_NET_GRU2_OUT_SIZE;
    const G123: usize = G12 + SIG_NET_GRU3_OUT_SIZE;
    compute_glu(&model.sig_net_gru3_glu_gate, &mut skip_cat[G12..G123], &st.gru3_state, arch);

    skip_cat[..SIG_NET_GRU1_OUT_SIZE].copy_from_slice(&gru2_in[..SIG_NET_GRU1_OUT_SIZE]);
    skip_cat[SIG_NET_GRU1_OUT_SIZE..G12].copy_from_slice(&gru3_in[..SIG_NET_GRU2_OUT_SIZE]);
    skip_cat[G123..G123 + SIG_NET_FWC0_CONV_OUT_SIZE]
        .copy_from_slice(&gru1_in[..SIG_NET_FWC0_CONV_OUT_SIZE]);
    append(&mut skip_cat, G123 + SIG_NET_FWC0_CONV_OUT_SIZE, pitch_gate[3], &pred, &prev);

    compute_generic_dense(
        &model.sig_net_skip_dense,
        &mut skip_out,
        &skip_cat,
        Activation::Tanh,
        arch,
    );
    compute_glu_in_place(&model.sig_net_skip_glu_gate, &mut skip_out, arch);

    compute_generic_dense(&model.sig_net_sig_dense_out, pcm, &skip_out, Activation::Tanh, arch);
    for v in pcm.iter_mut() {
        *v *= gain;
    }

    st.pitch_buf.copy_within(FARGAN_SUBFRAME_SIZE.., 0);
    st.pitch_buf[PITCH_MAX_PERIOD - FARGAN_SUBFRAME_SIZE..].copy_from_slice(pcm);
    fargan_deemphasis(pcm, &mut st.deemph_mem);
}

/// `fargan_cont` (`dnn/fargan.c:142-171`): primes FARGAN with the last
/// `FARGAN_CONT_SAMPLES` samples of past audio (`pcm0`, full scale ±1) and
/// the features of the last `CONT_VECTORS` frames (`features0`, one vector
/// of `NB_FEATURES` after the other).
///
/// The conditioning runs on the five vectors; the subframe network then
/// runs on the last 10 ms of `pcm0` (pre-emphasized) with the conditioning
/// of the fifth vector and the period of the fourth.
pub fn fargan_cont(
    st: &mut FarganState,
    pcm0: &[f32; FARGAN_CONT_SAMPLES],
    features0: &[f32; super::CONT_VECTORS * NB_FEATURES],
    model: &Fargan,
    arch: DnnArch,
) {
    let mut cond = [0.0f32; COND_NET_FDENSE2_OUT_SIZE];
    let mut period = 0;
    for features in features0.as_chunks::<NB_FEATURES>().0 {
        st.last_period = period;
        period = period_from_pitch(features[NB_BANDS]);
        compute_fargan_cond(st, &mut cond, features, period, model, arch);
    }

    let mut x0 = [0.0f32; FARGAN_CONT_SAMPLES];
    for i in 1..FARGAN_CONT_SAMPLES {
        x0[i] = fmaf(-FARGAN_DEEMPHASIS, pcm0[i - 1], pcm0[i]);
    }
    st.pitch_buf[PITCH_MAX_PERIOD - FARGAN_FRAME_SIZE..].copy_from_slice(&x0[..FARGAN_FRAME_SIZE]);
    st.cont_initialized = true;

    for i in 0..FARGAN_NB_SUBFRAMES {
        let mut dummy = [0.0f32; FARGAN_SUBFRAME_SIZE];
        let period = st.last_period;
        run_fargan_subframe(
            st,
            &mut dummy,
            &cond[i * FARGAN_COND_SIZE..],
            period,
            model,
            arch,
        );
        let start = FARGAN_FRAME_SIZE + i * FARGAN_SUBFRAME_SIZE;
        st.pitch_buf[PITCH_MAX_PERIOD - FARGAN_SUBFRAME_SIZE..]
            .copy_from_slice(&x0[start..start + FARGAN_SUBFRAME_SIZE]);
    }
    st.deemph_mem = pcm0[FARGAN_CONT_SAMPLES - 1];
}

/// `fargan_synthesize` (`dnn/fargan.c:197-216`): 10 ms of 16 kHz audio
/// (full scale ±1) from one feature vector. The subframes use the period
/// of the previous call (or of `fargan_cont`); this call's period is kept
/// for the next one.
pub fn fargan_synthesize(
    st: &mut FarganState,
    pcm: &mut [f32; FARGAN_FRAME_SIZE],
    features: &[f32; NB_FEATURES],
    model: &Fargan,
    arch: DnnArch,
) {
    let mut cond = [0.0f32; COND_NET_FDENSE2_OUT_SIZE];
    let period = period_from_pitch(features[NB_BANDS]);
    compute_fargan_cond(st, &mut cond, features, period, model, arch);
    for (sub_pcm, sub_cond) in pcm
        .as_chunks_mut::<FARGAN_SUBFRAME_SIZE>()
        .0
        .iter_mut()
        .zip(cond.as_chunks::<FARGAN_COND_SIZE>().0)
    {
        let last_period = st.last_period;
        run_fargan_subframe(st, sub_pcm, sub_cond, last_period, model, arch);
    }
    st.last_period = period;
}

/// `fargan_synthesize_int` (`dnn/fargan.c:218-225`): `fargan_synthesize`,
/// then each sample times 32768, clamped to ±32767 and rounded with
/// `floor(.5 + x)` in `double`.
pub fn fargan_synthesize_int(
    st: &mut FarganState,
    pcm: &mut [i16; LPCNET_FRAME_SIZE],
    features: &[f32; NB_FEATURES],
    model: &Fargan,
    arch: DnnArch,
) {
    let mut fpcm = [0.0f32; FARGAN_FRAME_SIZE];
    fargan_synthesize(st, &mut fpcm, features, model, arch);
    for (out, &v) in pcm.iter_mut().zip(fpcm.iter()) {
        let v = min_c(32767.0, max_c(-32767.0, 32768.0 * v));
        *out = (0.5 + v as f64).floor() as i32 as i16;
    }
}
