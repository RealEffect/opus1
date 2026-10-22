//! The deep PLC state machine (libopus `dnn/lpcnet_plc.c`, and
//! `update_plc_state` of `celt/celt_decoder.c`).
//!
//! `DeepPlc` keeps the last 150 ms of audio at 16 kHz (`pcm`, 15 frames of
//! 10 ms). Frames that arrive are only appended (`update`): libopus defers
//! the analysis of good frames (`PLC_SKIP_UPDATES`). The first lost frame
//! of a loss runs the deferred work as one batch:
//!
//! 1. every frame of the history that was not analysed yet goes through
//!    the Burg cepstrum and the feature analysis, from `analysis_pos` to
//!    the end; frames appended by `update` since the last lost frame
//!    (`analysis_pos >= predict_pos`) also run the PLC network, with
//!    their features as input;
//! 2. the PLC network predicts two more feature vectors;
//! 3. FARGAN starts on the last 20 ms of history and the last five feature
//!    vectors (`fargan_cont`).
//!
//! Then every lost frame (the first included) runs the PLC network once
//! more, lowers the energy feature with the length of the loss, runs
//! FARGAN, and appends the result to the history.
//!
//! `plc_bak` keeps the states of the PLC network from one and two steps
//! back; it rotates before every step. At the start of each loss the
//! network goes back to `plc_bak[0]`, its state from before the last two
//! steps of the previous loss (`dnn/lpcnet_plc.c:162`).
//!
//! The deep redundancy (DRED) decoder of libopus fills a queue of feature
//! vectors (`fec_add`, `fec_clear`) that replace the predictions of lost
//! frames. opus1 does not decode DRED yet; the queue is ported so that it
//! can.

#[cfg(not(feature = "std"))]
use crate::compat::Math;
use crate::dnn::Activation;
use crate::dnn::arch::DnnArch;
use crate::in_place::{write_fields, write_zero};
use crate::dnn::freq::{self, FRAME_SIZE, NB_BANDS, fmaf};
use crate::dnn::models::DeepPlcModels;
use crate::dnn::models::plc::{PLC_DENSE_IN_OUT_SIZE, PLC_GRU1_STATE_SIZE, PLC_GRU2_STATE_SIZE, PlcModel};
use crate::dnn::nnet::{compute_generic_dense, compute_generic_gru};

use super::fargan::{FARGAN_CONT_SAMPLES, FarganState, fargan_cont, fargan_synthesize_int};
use super::features::{LpcnetEncState, compute_single_frame_features};
use super::{
    CONT_VECTORS, LPCNET_FRAME_SIZE, NB_FEATURES, NB_TOTAL_FEATURES, PLC_BUF_SIZE,
    PLC_INPUT_SIZE, PLC_MAX_FEC, max_c, min_c,
};

/// The recurrent state of the PLC network (libopus `PLCNetState`,
/// `dnn/lpcnet_private.h:44-47`).
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct PlcNetState {
    /// The state of the first GRU.
    pub gru1_state: [f32; PLC_GRU1_STATE_SIZE],
    /// The state of the second GRU.
    pub gru2_state: [f32; PLC_GRU2_STATE_SIZE],
}

impl PlcNetState {
    /// A cleared state.
    pub const fn new() -> Self {
        PlcNetState { gru1_state: [0.0; PLC_GRU1_STATE_SIZE], gru2_state: [0.0; PLC_GRU2_STATE_SIZE] }
    }
}

impl Default for PlcNetState {
    fn default() -> Self {
        Self::new()
    }
}

/// `compute_plc_pred` (`dnn/lpcnet_plc.c:106-115`): one step of the PLC
/// network. `input` is `[Burg cepstrum (36), features (20), flag]`; the
/// `NB_FEATURES` predicted features of the next frame go to
/// `out[..NB_FEATURES]`.
pub fn compute_plc_pred(
    net: &mut PlcNetState,
    out: &mut [f32],
    input: &[f32; PLC_INPUT_SIZE],
    model: &PlcModel,
    arch: DnnArch,
) {
    let mut tmp = [0.0f32; PLC_DENSE_IN_OUT_SIZE];
    compute_generic_dense(&model.plc_dense_in, &mut tmp, input, Activation::Tanh, arch);
    compute_generic_gru(&model.plc_gru1_input, &model.plc_gru1_recurrent, &mut net.gru1_state, &tmp, arch);
    compute_generic_gru(
        &model.plc_gru2_input,
        &model.plc_gru2_recurrent,
        &mut net.gru2_state,
        &net.gru1_state,
        arch,
    );
    compute_generic_dense(&model.plc_dense_out, &mut out[..NB_FEATURES], &net.gru2_state, Activation::Linear, arch);
}

/// The attenuation of the energy feature for the first ten lost frames of
/// a loss (`att_table`, `dnn/lpcnet_plc.c:156`).
const ATT_TABLE: [f32; 10] = [0.0, 0.0, -0.2, -0.2, -0.4, -0.4, -0.8, -0.8, -1.6, -1.6];

/// `DECODE_BUFFER_SIZE` of the CELT decoder (`celt/celt_decoder.c:79`): the
/// history per channel that `update_from_celt` reads.
pub const CELT_DECODE_BUFFER_SIZE: usize = 2048;

/// `SINC_ORDER` (`celt/celt_decoder.c:625`).
pub const SINC_ORDER: usize = 48;

/// `PLC_UPDATE_FRAMES` (`celt/celt_decoder.c:81`): the frames that
/// `update_from_celt` appends.
const PLC_UPDATE_FRAMES: usize = 4;

/// `PLC_UPDATE_SAMPLES`.
const PLC_UPDATE_SAMPLES: usize = PLC_UPDATE_FRAMES * FRAME_SIZE;

/// `PREEMPHASIS` of CELT (`celt/celt.h`).
const CELT_PREEMPHASIS: f32 = 0.85;

/// `sinc_filter` (`celt/celt_decoder.c:629-637`): the 49-tap low-pass
/// filter of the resampling between 48 kHz and 16 kHz, in both directions.
/// The CELT side of deep PLC also uses it to upsample the concealed audio.
#[allow(clippy::excessive_precision)]
pub const SINC_FILTER: [f32; SINC_ORDER + 1] = [
    4.2931e-05, -0.000190293, -0.000816132, -0.000637162, 0.00141662, 0.00354764, 0.00184368,
    -0.00428274, -0.00856105, -0.0034003, 0.00930201, 0.0159616, 0.00489785, -0.0169649,
    -0.0259484, -0.00596856, 0.0286551, 0.0405872, 0.00649994, -0.0509284, -0.0716655,
    -0.00665212, 0.134336, 0.278927, 0.339995, 0.278927, 0.134336, -0.00665212, -0.0716655,
    -0.0509284, 0.00649994, 0.0405872, 0.0286551, -0.00596856, -0.0259484, -0.0169649, 0.00489785,
    0.0159616, 0.00930201, -0.0034003, -0.00856105, -0.00428274, 0.00184368, 0.00354764,
    0.00141662, -0.000637162, -0.000816132, -0.000190293, 4.2931e-05,
];

/// The state of deep PLC for one decoder (libopus `LPCNetPLCState`,
/// `dnn/lpcnet_private.h:50-72`), without the weights, the `loaded` flag
/// and the kernel variant: `conceal` takes the models and the variant as
/// arguments, and "loaded" is whether the decoder has the models
/// (`DecoderDnn::deep_plc`).
///
/// Size: 42,012 bytes (the DRED queue is 8,320 of them). It allocates
/// nothing. On the Apple M4, a release build conceals a first lost frame
/// on a thread with a 16 KB stack.
#[derive(Clone)]
pub struct DeepPlc {
    /// FARGAN. `reset` leaves it as it is.
    fargan: FarganState,
    /// The feature analysis.
    enc: LpcnetEncState,
    // Everything below is cleared by `reset` (`LPCNET_PLC_RESET_START`).
    /// The queue of DRED features.
    fec: [[f32; NB_FEATURES]; PLC_MAX_FEC],
    /// Whether frames left the history without being analysed, so the
    /// first analysed frame of the next batch has no valid past and does
    /// not feed the PLC network.
    analysis_gap: bool,
    /// The next queued DRED vector.
    fec_read_pos: i32,
    /// The number of queued DRED vectors.
    fec_fill_pos: i32,
    /// Lost frames to predict before the queue is read.
    fec_skip: i32,
    /// The first sample of the history that was not analysed.
    analysis_pos: i32,
    /// The first sample of the history appended by `update` since the last
    /// lost frame (`PLC_BUF_SIZE` when none).
    predict_pos: i32,
    /// The history, full scale ±1.
    pcm: [f32; PLC_BUF_SIZE],
    /// Whether the last frame was concealed (FARGAN is running).
    blend: bool,
    /// The features of the last analysed or concealed frame.
    features: [f32; NB_TOTAL_FEATURES],
    /// The last `CONT_VECTORS` feature vectors, the oldest first.
    cont_features: [f32; CONT_VECTORS * NB_FEATURES],
    /// The lost frames since the last good frame or DRED vector.
    loss_count: i32,
    /// The PLC network's state.
    plc_net: PlcNetState,
    /// The PLC network's state one and two steps back.
    plc_bak: [PlcNetState; 2],
}

// `DeepPlc::write_new` writes these states in place as zero bytes.
crate::in_place::zeroable_new!(PlcNetState);

impl Default for DeepPlc {
    fn default() -> Self {
        Self::new()
    }
}

impl DeepPlc {
    /// A new state (libopus `lpcnet_plc_init`: FARGAN and the feature
    /// analysis cleared, then `lpcnet_plc_reset`).
    pub fn new() -> Self {
        let mut st = core::mem::MaybeUninit::<Self>::uninit();
        // SAFETY: `write_new` initializes every field of `st`.
        unsafe {
            Self::write_new(st.as_mut_ptr());
            st.assume_init()
        }
    }

    /// Writes the state of [`new`](Self::new) to `this`, which may be
    /// uninitialized memory, without building it on the stack first (see
    /// `crate::in_place`): the state is 42 KB.
    ///
    /// # Safety
    /// `this` must be valid for writes of `Self`.
    pub(crate) unsafe fn write_new(this: *mut Self) {
        // SAFETY: the caller guarantees that `this` is valid for writes.
        unsafe {
            write_fields!(this, {
                fargan: FarganState::new(),
                enc = write_zero,
                fec = write_zero,
                analysis_gap: true,
                fec_read_pos: 0,
                fec_fill_pos: 0,
                fec_skip: 0,
                analysis_pos: PLC_BUF_SIZE as i32,
                predict_pos: PLC_BUF_SIZE as i32,
                pcm = write_zero,
                blend: false,
                features = write_zero,
                cont_features = write_zero,
                loss_count: 0,
                plc_net = write_zero,
                plc_bak = write_zero,
            });
        }
    }

    /// `lpcnet_plc_reset` (`dnn/lpcnet_plc.c:45-56`), what
    /// `OPUS_RESET_STATE` runs: clears the history, the positions, the DRED
    /// queue, the features, the PLC network's states and the feature
    /// analysis (PitchDNN's state included).
    ///
    /// FARGAN is not cleared, as in libopus: its GRU states, its
    /// convolution memories, its period and its pitch buffer carry over
    /// (`fargan_cont` rewrites only the last 160 samples of the buffer), so
    /// concealment after a reset can depend on losses concealed before it.
    pub fn reset(&mut self) {
        self.enc.reset();
        for v in self.fec.iter_mut() {
            v.fill(0.0);
        }
        self.analysis_gap = true;
        self.fec_read_pos = 0;
        self.fec_fill_pos = 0;
        self.fec_skip = 0;
        self.analysis_pos = PLC_BUF_SIZE as i32;
        self.predict_pos = PLC_BUF_SIZE as i32;
        self.pcm.fill(0.0);
        self.blend = false;
        self.features.fill(0.0);
        self.cont_features.fill(0.0);
        self.loss_count = 0;
        self.plc_net = PlcNetState::new();
        self.plc_bak = [PlcNetState::new(); 2];
    }

    /// `lpcnet_plc_update` (`dnn/lpcnet_plc.c:144-154`): appends 10 ms of
    /// decoded 16 kHz audio to the history. Ends a loss: the next lost
    /// frame starts a new one.
    pub fn update(&mut self, pcm: &[i16; LPCNET_FRAME_SIZE]) {
        let n = FRAME_SIZE as i32;
        if self.analysis_pos - n >= 0 {
            self.analysis_pos -= n;
        } else {
            self.analysis_gap = true;
        }
        if self.predict_pos - n >= 0 {
            self.predict_pos -= n;
        }
        self.append(pcm);
        self.loss_count = 0;
        self.blend = false;
    }

    /// Shifts the history by one frame and appends `pcm` as `pcm / 32768`.
    fn append(&mut self, pcm: &[i16; LPCNET_FRAME_SIZE]) {
        self.pcm.copy_within(FRAME_SIZE.., 0);
        for (h, &s) in self.pcm[PLC_BUF_SIZE - FRAME_SIZE..].iter_mut().zip(pcm.iter()) {
            *h = (1.0 / 32768.0) * s as f32;
        }
    }

    /// `lpcnet_plc_conceal` (`dnn/lpcnet_plc.c:157-207`): makes up 10 ms of
    /// 16 kHz audio for a lost frame, writes it to `pcm` and appends it to
    /// the history.
    ///
    /// The first call of a loss also analyses the part of the history that
    /// was not analysed yet (up to 15 frames, with up to 17 steps of the
    /// PLC network) and starts FARGAN on it. After 15 or more good frames
    /// it costs about seven times a later call (313 µs against 45 µs on the
    /// Apple M4).
    pub fn conceal(
        &mut self,
        models: &DeepPlcModels,
        arch: DnnArch,
        pcm: &mut [i16; LPCNET_FRAME_SIZE],
    ) {
        let buf_size = PLC_BUF_SIZE as i32;
        let n = FRAME_SIZE as i32;
        if !self.blend {
            let mut count = 0;
            self.plc_net = self.plc_bak[0];
            while self.analysis_pos + n <= buf_size {
                debug_assert!(self.analysis_pos >= 0);
                let start = self.analysis_pos as usize;
                let mut x = [0.0f32; FRAME_SIZE];
                for (v, &h) in x.iter_mut().zip(&self.pcm[start..start + FRAME_SIZE]) {
                    *v = 32768.0 * h;
                }
                let mut plc_features = [0.0f32; PLC_INPUT_SIZE];
                let mut burg = [0.0f32; 2 * NB_BANDS];
                freq::burg_cepstral_analysis(&mut burg, &x);
                plc_features[..2 * NB_BANDS].copy_from_slice(&burg);
                compute_single_frame_features(
                    &mut self.enc,
                    &x,
                    &mut self.features,
                    &models.pitchdnn,
                    arch,
                );
                if (!self.analysis_gap || count > 0) && self.analysis_pos >= self.predict_pos {
                    let mut f = [0.0f32; NB_FEATURES];
                    f.copy_from_slice(&self.features[..NB_FEATURES]);
                    self.queue_features(&f);
                    plc_features[2 * NB_BANDS..2 * NB_BANDS + NB_FEATURES].copy_from_slice(&f);
                    plc_features[2 * NB_BANDS + NB_FEATURES] = 1.0;
                    self.rotate_backups();
                    compute_plc_pred(
                        &mut self.plc_net,
                        &mut self.features,
                        &plc_features,
                        &models.plc,
                        arch,
                    );
                }
                self.analysis_pos += n;
                count += 1;
            }
            for _ in 0..2 {
                self.rotate_backups();
                self.get_fec_or_pred(&models.plc, arch);
                let f = self.first_features();
                self.queue_features(&f);
            }
            let pcm0: &[f32; FARGAN_CONT_SAMPLES] =
                match self.pcm[PLC_BUF_SIZE - FARGAN_CONT_SAMPLES..].try_into() {
                    Ok(p) => p,
                    Err(_) => unreachable!("the last FARGAN_CONT_SAMPLES of the history"),
                };
            fargan_cont(&mut self.fargan, pcm0, &self.cont_features, &models.fargan, arch);
            self.analysis_gap = false;
        }
        self.rotate_backups();
        if self.get_fec_or_pred(&models.plc, arch) {
            self.loss_count = 0;
        } else {
            self.loss_count = self.loss_count.saturating_add(1);
        }
        // Lower the energy: by the table for the first ten frames, then by
        // 2 more per frame, down to -15.
        let f0 = if self.loss_count >= 10 {
            // `2*(loss_count-9)`, an `int` converted to `float`.
            let ramp = (2 * (self.loss_count as i64 - 9)) as f32;
            (self.features[0] + ATT_TABLE[9]) - ramp
        } else {
            self.features[0] + ATT_TABLE[self.loss_count as usize]
        };
        self.features[0] = max_c(-15.0, f0);
        let f = self.first_features();
        fargan_synthesize_int(&mut self.fargan, pcm, &f, &models.fargan, arch);
        self.queue_features(&f);
        if self.analysis_pos - n >= 0 {
            self.analysis_pos -= n;
        } else {
            self.analysis_gap = true;
        }
        self.predict_pos = buf_size;
        self.append(pcm);
        self.blend = true;
    }

    /// `update_plc_state` (`celt/celt_decoder.c:639-672`): appends the last
    /// 40 ms of a CELT decoder's history to the deep PLC history, as four
    /// frames.
    ///
    /// `decode_mem` holds one slice per channel (one or two), each with the
    /// last `CELT_DECODE_BUFFER_SIZE` samples of the CELT history at 48 kHz,
    /// the oldest first, as CELT keeps it: before de-emphasis (libopus
    /// `decode_mem[c][0..DECODE_BUFFER_SIZE]`). Two channels are averaged.
    /// The history is then de-emphasized (`preemph_mem` gets its last
    /// sample, libopus `plc_preemphasis_mem`), low-pass filtered with
    /// `SINC_FILTER`, decimated by 3 and rounded to 16-bit samples.
    ///
    /// libopus saves and restores the DRED queue cursor around the four
    /// updates; `update` does not touch it, so there is nothing to restore.
    ///
    /// Panics when `decode_mem` holds no slice, more than two, or a slice
    /// shorter than `CELT_DECODE_BUFFER_SIZE`.
    pub fn update_from_celt(&mut self, decode_mem: &[&[f32]], preemph_mem: &mut f32) {
        assert!(matches!(decode_mem.len(), 1 | 2), "one or two channels");
        let mut buf48k = [0.0f32; CELT_DECODE_BUFFER_SIZE];
        if decode_mem.len() == 1 {
            buf48k.copy_from_slice(&decode_mem[0][..CELT_DECODE_BUFFER_SIZE]);
        } else {
            let (c0, c1) =
                (&decode_mem[0][..CELT_DECODE_BUFFER_SIZE], &decode_mem[1][..CELT_DECODE_BUFFER_SIZE]);
            for ((b, &a0), &a1) in buf48k.iter_mut().zip(c0).zip(c1) {
                // `.5*(a + b)`: the product in `double`, which is exact, so
                // the same as in `float`.
                *b = 0.5 * (a0 + a1);
            }
        }
        for i in 1..CELT_DECODE_BUFFER_SIZE {
            buf48k[i] = fmaf(CELT_PREEMPHASIS, buf48k[i - 1], buf48k[i]);
        }
        *preemph_mem = buf48k[CELT_DECODE_BUFFER_SIZE - 1];
        let offset = CELT_DECODE_BUFFER_SIZE - SINC_ORDER - 1 - 3 * (PLC_UPDATE_SAMPLES - 1);
        let mut buf16k = [0i16; PLC_UPDATE_SAMPLES];
        for (i, out) in buf16k.iter_mut().enumerate() {
            let taps = &buf48k[3 * i + offset..3 * i + offset + SINC_ORDER + 1];
            let mut sum = 0.0f32;
            for (&x, &h) in taps.iter().zip(SINC_FILTER.iter()) {
                sum = fmaf(x, h, sum);
            }
            // `float2int(MIN32(32767.f, MAX32(-32767.f, sum)))`: rounded to
            // the nearest, ties to even (`lrintf`).
            *out = min_c(32767.0, max_c(-32767.0, sum)).round_ties_even() as i32 as i16;
        }
        for frame in buf16k.as_chunks::<LPCNET_FRAME_SIZE>().0 {
            self.update(frame);
        }
    }

    /// `lpcnet_plc_fec_add` (`dnn/lpcnet_plc.c:91-99`): queues one DRED
    /// feature vector, or with `None` one lost frame without DRED data,
    /// which the PLC network predicts instead. A vector beyond the room of
    /// the queue (`PLC_MAX_FEC`) is dropped; libopus asserts there.
    pub fn fec_add(&mut self, features: Option<&[f32; NB_FEATURES]>) {
        match features {
            None => self.fec_skip = self.fec_skip.saturating_add(1),
            Some(f) => {
                if let Some(slot) = self.fec.get_mut(self.fec_fill_pos as usize) {
                    *slot = *f;
                    self.fec_fill_pos += 1;
                }
            }
        }
    }

    /// `lpcnet_plc_fec_clear` (`dnn/lpcnet_plc.c:101-103`): empties the
    /// DRED queue.
    pub fn fec_clear(&mut self) {
        self.fec_read_pos = 0;
        self.fec_fill_pos = 0;
        self.fec_skip = 0;
    }

    /// The number of DRED vectors queued since the last `fec_clear`
    /// (libopus `fec_fill_pos`). SILK runs deep PLC below complexity 5 when
    /// it is not 0.
    pub fn fec_fill_pos(&self) -> i32 {
        self.fec_fill_pos
    }

    /// The next DRED vector to read and the lost frames to predict before
    /// it (libopus `fec_read_pos`, `fec_skip`).
    pub fn fec_cursor(&self) -> (i32, i32) {
        (self.fec_read_pos, self.fec_skip)
    }

    /// Whether the last frame was concealed by `conceal` (libopus
    /// `blend`): the next `conceal` continues the loss.
    pub fn blend(&self) -> bool {
        self.blend
    }

    /// Ends a loss without appending audio (libopus `lpcnet->blend = 0`,
    /// which the CELT decoder runs for every frame it decodes): the next
    /// `conceal` starts a new loss.
    pub fn clear_blend(&mut self) {
        self.blend = false;
    }

    /// The features of the last analysed or concealed frame (libopus
    /// `features`).
    pub fn features(&self) -> &[f32; NB_TOTAL_FEATURES] {
        &self.features
    }

    /// The feature vectors that FARGAN would start on, the oldest first.
    pub fn cont_features(&self) -> &[f32; CONT_VECTORS * NB_FEATURES] {
        &self.cont_features
    }

    /// The history: the last 150 ms at 16 kHz, full scale ±1.
    pub fn history(&self) -> &[f32; PLC_BUF_SIZE] {
        &self.pcm
    }

    /// `(analysis_pos, predict_pos, analysis_gap, loss_count)`.
    pub fn positions(&self) -> (i32, i32, bool, i32) {
        (self.analysis_pos, self.predict_pos, self.analysis_gap, self.loss_count)
    }

    /// The PLC network's state and its two backups.
    pub fn plc_states(&self) -> (&PlcNetState, &[PlcNetState; 2]) {
        (&self.plc_net, &self.plc_bak)
    }

    /// FARGAN's state.
    pub fn fargan(&self) -> &FarganState {
        &self.fargan
    }

    /// The feature analysis' state.
    pub fn analysis(&self) -> &LpcnetEncState {
        &self.enc
    }

    /// `plc_bak[0] = plc_bak[1]; plc_bak[1] = plc_net`.
    #[inline]
    fn rotate_backups(&mut self) {
        self.plc_bak[0] = self.plc_bak[1];
        self.plc_bak[1] = self.plc_net;
    }

    /// The first `NB_FEATURES` values of `features`.
    #[inline]
    fn first_features(&self) -> [f32; NB_FEATURES] {
        let mut f = [0.0f32; NB_FEATURES];
        f.copy_from_slice(&self.features[..NB_FEATURES]);
        f
    }

    /// `queue_features` (`dnn/lpcnet_plc.c:136-139`).
    fn queue_features(&mut self, features: &[f32; NB_FEATURES]) {
        self.cont_features.copy_within(NB_FEATURES.., 0);
        self.cont_features[(CONT_VECTORS - 1) * NB_FEATURES..].copy_from_slice(features);
    }

    /// `get_fec_or_pred` (`dnn/lpcnet_plc.c:117-134`): the features of the
    /// next lost frame into `features[..NB_FEATURES]`, from the DRED queue
    /// (true) or predicted (false). A queued vector also runs the PLC
    /// network on it, to keep the network's state current.
    fn get_fec_or_pred(&mut self, model: &PlcModel, arch: DnnArch) -> bool {
        if self.fec_read_pos != self.fec_fill_pos && self.fec_skip == 0 {
            let f = self.fec[self.fec_read_pos as usize];
            self.features[..NB_FEATURES].copy_from_slice(&f);
            self.fec_read_pos += 1;
            let mut plc_features = [0.0f32; PLC_INPUT_SIZE];
            plc_features[2 * NB_BANDS..2 * NB_BANDS + NB_FEATURES].copy_from_slice(&f);
            plc_features[2 * NB_BANDS + NB_FEATURES] = -1.0;
            let mut discard = [0.0f32; NB_FEATURES];
            compute_plc_pred(&mut self.plc_net, &mut discard, &plc_features, model, arch);
            true
        } else {
            let zeros = [0.0f32; PLC_INPUT_SIZE];
            compute_plc_pred(&mut self.plc_net, &mut self.features, &zeros, model, arch);
            if self.fec_skip > 0 {
                self.fec_skip -= 1;
            }
            false
        }
    }
}
