//! The DRED encoder: libopus `dnn/dred_encoder.c`, and the fields and
//! steps of DRED in `src/opus_encoder.c`.
//!
//! For every frame the Opus encoder codes, the DRED encoder:
//!
//! 1. converts the filtered input to 16 kHz mono (`convert_to_16k`);
//! 2. for every 20 ms of it, computes the features of two 10 ms frames
//!    with the feature analysis of deep PLC (`LpcnetEncState`), and runs
//!    the RDOVAE encoder on them, which gives one latent vector and one
//!    initial state vector (`process_frame`);
//! 3. keeps the last `DRED_MAX_FRAMES` of both, and the voice activity of
//!    the last `4 * DRED_MAX_FRAMES` periods of 2.5 ms.
//!
//! In the first coded frame of a packet it then quantizes one initial
//! state and up to `DRED_NUM_REDUNDANCY_FRAMES / 2` latent vectors, newest
//! first, with a range coder (`encode_silk_frame`). The Opus encoder puts
//! the result into the padding of the packet as the extension with ID
//! `DRED_EXTENSION_ID` (126).

#[cfg(not(feature = "std"))]
use crate::compat::Math;

use super::coding::compute_quantizer;
use super::rdovae_enc::RdovaeEncState;
use super::{
    DRED_DFRAME_SIZE, DRED_EXPERIMENTAL_BYTES, DRED_EXPERIMENTAL_VERSION, DRED_FRAME_SIZE, DRED_MAX_DATA_SIZE,
    DRED_MAX_FRAMES, DRED_MIN_BYTES, DRED_NUM_REDUNDANCY_FRAMES, DRED_SILK_ENCODER_DELAY,
};
use crate::dnn::Activation;
use crate::dnn::DnnBlob;
use crate::dnn::arch::DnnArch;
use crate::dnn::holder::DredEncoderDnn;
use crate::dnn::models::DredEncModels;
use crate::dnn::models::dred_rdovae_stats::{
    DRED_LATENT_DEAD_ZONE_Q8, DRED_LATENT_DIM, DRED_LATENT_P0_Q8, DRED_LATENT_QUANT_SCALES_Q8, DRED_LATENT_R_Q8,
    DRED_NUM_FEATURES, DRED_STATE_DEAD_ZONE_Q8, DRED_STATE_DIM, DRED_STATE_P0_Q8, DRED_STATE_QUANT_SCALES_Q8,
    DRED_STATE_R_Q8,
};
use crate::dnn::nnet::compute_activation_in_place;
use crate::dnn::plc::NB_TOTAL_FEATURES;
use crate::dnn::plc::features::{LpcnetEncState, compute_single_frame_features};
use crate::range_coder::RangeCoder;
use crate::{bitrate_to_bits, bits_to_bitrate, float2int16};

/// The order of the low-pass filters of `convert_to_16k`.
const RESAMPLING_ORDER: usize = 8;

/// The size of the buffer of `convert_to_16k` (libopus
/// `MAX_DOWNMIX_BUFFER` without QEXT): 20 ms of stereo at 48 kHz.
const MAX_DOWNMIX_BUFFER: usize = 960 * 2;

/// libopus `VERY_SMALL` (celt/arch.h, float build).
const VERY_SMALL: f32 = 1e-30;

/// The approximate size in bits of the initial state (by its quantizer)
/// and of one latent vector (by its quantizer level), from libopus
/// `dred_bits_table` (opus_encoder.c:668).
const DRED_BITS_TABLE: [f32; 16] =
    [73.2, 68.1, 62.5, 57.0, 51.5, 45.7, 39.9, 32.4, 26.4, 20.4, 16.3, 13., 9.3, 8.2, 7.2, 6.4];

/// The state of libopus's `DREDEnc` after `DREDENC_RESET_START`, plus the
/// feature analysis and the RDOVAE encoder state.
#[derive(Clone)]
struct DredEnc {
    /// The feature analysis (libopus `lpcnet_enc_state`).
    lpcnet: LpcnetEncState,
    /// The RDOVAE encoder state (libopus `rdovae_enc`).
    rdovae: RdovaeEncState,
    /// The input rate in Hz.
    fs: i32,
    /// The input channels, 1 or 2.
    channels: usize,
    /// 16 kHz mono audio not yet analyzed.
    input_buffer: [f32; 2 * DRED_DFRAME_SIZE],
    /// How many samples of `input_buffer` are filled.
    input_buffer_fill: usize,
    /// Where the newest latent vector ends relative to the end of the
    /// frame, in 2.5 ms units (libopus `dred_offset`).
    dred_offset: i32,
    /// How many latent vectors the last frame added after the first one
    /// that `dred_offset` counts with (libopus `latent_offset`).
    latent_offset: i32,
    /// How many latent vectors of silence the last DRED data skipped
    /// (libopus `last_extra_dred_offset`).
    last_extra_dred_offset: i32,
    /// The latent vectors, newest first.
    latents_buffer: [f32; DRED_MAX_FRAMES * DRED_LATENT_DIM],
    /// How many latent vectors are valid, at most
    /// `DRED_NUM_REDUNDANCY_FRAMES`.
    latents_buffer_fill: usize,
    /// The initial state vectors that go with the latent vectors.
    state_buffer: [f32; DRED_MAX_FRAMES * DRED_STATE_DIM],
    /// The memory of the low-pass filter of `convert_to_16k`.
    resample_mem: [f32; RESAMPLING_ORDER + 1],
}

/// The DRED part of an Opus encoder: libopus `DREDEnc` and the DRED
/// fields of `OpusEncoder` (`dred_duration`, `dred_q0`, `dred_dQ`,
/// `dred_qmax`, `dred_target_chunks`, `activity_mem`).
pub(crate) struct DredEncoder {
    /// The models (libopus `model`, `loaded` and the PitchDNN model of
    /// `lpcnet_enc_state`) and the kernel variant.
    dnn: DredEncoderDnn,
    /// How much audio to keep redundant, in 10 ms frames (libopus
    /// `OPUS_SET_DRED_DURATION`). 0 turns DRED off.
    duration: i32,
    /// The quantizer of the initial state and of the first latent vector.
    q0: i32,
    /// How fast the quantizer grows with the age of the latent vector.
    dq: i32,
    /// The largest quantizer.
    qmax: i32,
    /// How many latent vectors the bitrate of DRED pays for.
    target_chunks: i32,
    /// The bitrate taken from the packet for DRED (a local of libopus
    /// `opus_encode_native`, `dred_bitrate_bps`).
    bitrate_bps: i32,
    /// The voice activity of the last frames, one value per 2.5 ms, newest
    /// first: 1 active, 0 not.
    activity_mem: [u8; 4 * DRED_MAX_FRAMES],
    /// The analysis and the latent vectors.
    enc: DredEnc,
}

impl DredEncoder {
    /// libopus `dred_encoder_init` with the built-in models, and DRED off.
    pub(crate) fn new(fs: i32, channels: usize) -> Self {
        DredEncoder {
            dnn: DredEncoderDnn::new(),
            duration: 0,
            q0: 0,
            dq: 0,
            qmax: 0,
            target_chunks: 0,
            bitrate_bps: 0,
            activity_mem: [0; 4 * DRED_MAX_FRAMES],
            enc: DredEnc::new(fs, channels),
        }
    }

    /// Swaps the models with those of `other`. `OpusEncoder::reset_state`
    /// gives a new encoder the models of the old one: libopus's
    /// `OPUS_RESET_STATE` clears the DRED state and turns DRED off, but
    /// keeps the RDOVAE encoder.
    pub(crate) fn swap_models(&mut self, other: &mut DredEncoder) {
        core::mem::swap(&mut self.dnn, &mut other.dnn);
    }

    /// libopus `OPUS_SET_DRED_DURATION`: 0 to `DRED_MAX_FRAMES` frames of
    /// 10 ms.
    pub(crate) fn set_duration(&mut self, frames: i32) -> Result<(), &'static str> {
        if !(0..=DRED_MAX_FRAMES as i32).contains(&frames) {
            return Err("Invalid DRED duration");
        }
        self.duration = frames;
        Ok(())
    }

    /// libopus `OPUS_GET_DRED_DURATION`.
    pub(crate) fn duration(&self) -> i32 {
        self.duration
    }

    /// The bitrate taken from the current packet for DRED.
    pub(crate) fn bitrate_bps(&self) -> i32 {
        self.bitrate_bps
    }

    /// libopus `dred_encoder_load_model`: the RDOVAE encoder and PitchDNN
    /// from `blob`. When one of them is missing, returns `"DNN blob holds
    /// no model of this build"` and keeps the models it had.
    pub(crate) fn load(&mut self, blob: &DnnBlob) -> Result<(), &'static str> {
        self.dnn.load(blob)
    }

    /// libopus `loaded`.
    pub(crate) fn is_loaded(&self) -> bool {
        self.dnn.model().is_some()
    }

    /// libopus `compute_dred_bitrate` (opus_encoder.c:687-730): the part of
    /// `bitrate_bps` that DRED takes for a packet of `frame_size` samples,
    /// and the quantizers and the number of latent vectors that go with
    /// it. `fec` is whether in-band FEC is on, `loss` the expected packet
    /// loss in percent.
    pub(crate) fn compute_bitrate(&mut self, bitrate_bps: i32, frame_size: usize, fec: bool, loss: i32) -> i32 {
        let fs = self.enc.fs;
        // C `MIN16` of the float build.
        let min16 = |a: f32, b: f32| if a < b { a } else { b };
        let (mut dred_frac, bitrate_offset) = if fec {
            (min16(0.7, 3.0 * loss as f32 / 100.0), 20000)
        } else if loss > 5 {
            (min16(0.8, 0.55 + loss as f32 / 100.0), 12000)
        } else {
            ((12 * loss) as f32 / 100.0, 12000)
        };
        // "Account for the fact that longer packets require less
        // redundancy."
        dred_frac = dred_frac / (dred_frac + (1.0 - dred_frac) * (frame_size as f32 * 50.0) / fs as f32);
        // "Approximate fit based on a few experiments. Could probably be
        // improved."
        let rate = bitrate_bps - bitrate_offset;
        let q0 = (51 - 3 * ec_ilog(rate.max(1))).clamp(4, 15);
        let dq = if rate > 36000 { 3 } else { 5 };
        let qmax = 15;
        let target_dred_bitrate = ((dred_frac * rate as f32) as i32).max(0);
        let (max_dred_bits, target_chunks) = if self.duration > 0 {
            let target_bits = bitrate_to_bits(target_dred_bitrate, fs, frame_size);
            estimate_bitrate(q0, dq, qmax, self.duration, target_bits)
        } else {
            (0, 0)
        };
        let mut dred_bitrate = target_dred_bitrate.min(bits_to_bitrate(max_dred_bits, fs, frame_size));
        // "If we can't afford enough bits, don't bother with DRED at all."
        if target_chunks < 2 {
            dred_bitrate = 0;
        }
        self.q0 = q0;
        self.dq = dq;
        self.qmax = qmax;
        self.target_chunks = target_chunks;
        self.bitrate_bps = dred_bitrate;
        dred_bitrate
    }

    /// opus_encoder.c:2027-2040, before SILK: with DRED on and the models
    /// loaded, analyzes the frame `pcm` (`frame_size` samples per channel,
    /// interleaved) and records its `activity` (C `VAD_NO_DECISION`, -1,
    /// is kept as 255 until [`set_activity`](Self::set_activity) replaces
    /// it). `extra_delay` is the encoder's look-ahead in samples. With
    /// DRED off, forgets the latent vectors and the activity.
    pub(crate) fn analyze_frame(&mut self, pcm: &[f32], frame_size: usize, extra_delay: usize, activity: i32) {
        if self.duration > 0
            && let Some(model) = self.dnn.model()
        {
            self.enc.compute_latents(model, pcm, frame_size, extra_delay, self.dnn.arch());
            let n = frame_size * 400 / self.enc.fs as usize;
            self.activity_mem.copy_within(..4 * DRED_MAX_FRAMES - n, n);
            self.activity_mem[..n].fill(activity as u8);
        } else {
            self.enc.latents_buffer_fill = 0;
            self.activity_mem.fill(0);
        }
    }

    /// opus_encoder.c:2237-2240: the frame's activity once SILK decided
    /// it.
    pub(crate) fn set_activity(&mut self, frame_size: usize, activity: i32) {
        let n = frame_size * 400 / self.enc.fs as usize;
        self.activity_mem[..n].fill(activity as u8);
    }

    /// opus_encoder.c:2603-2625: the DRED data for the first coded frame
    /// of a packet, written to `buf` with the two experimental bytes in
    /// front. `room` is C `orig_max_data_bytes - ret - 3`: the bytes left
    /// in the packet after the frame, less the code 3 byte, the padding
    /// length and the extension ID. Returns the length of the extension
    /// data, or 0 when there is none: DRED off, no model, too little room,
    /// or nothing worth sending. `rc` is used as the range coder.
    pub(crate) fn write_extension(
        &mut self,
        rc: &mut RangeCoder,
        buf: &mut [u8; DRED_MAX_DATA_SIZE],
        room: i32,
        use_vbr: bool,
    ) -> usize {
        if self.duration <= 0 || !self.is_loaded() {
            return 0;
        }
        let mut dred_chunks = ((self.duration + 5) / 4).min(DRED_NUM_REDUNDANCY_FRAMES as i32 / 2);
        if use_vbr {
            dred_chunks = dred_chunks.min(self.target_chunks);
        }
        let mut bytes_left = (DRED_MAX_DATA_SIZE as i32).min(room);
        // "Account for the extra bytes required to signal large padding
        // length."
        bytes_left -= (bytes_left + 1 + DRED_EXPERIMENTAL_BYTES as i32) / 255;
        // "Check whether we actually have something to encode."
        if dred_chunks < 1 || bytes_left < (DRED_MIN_BYTES + DRED_EXPERIMENTAL_BYTES) as i32 {
            return 0;
        }
        // "Add temporary extension type and version."
        buf[0] = b'D';
        buf[1] = DRED_EXPERIMENTAL_VERSION;
        let max_bytes = bytes_left as usize - DRED_EXPERIMENTAL_BYTES;
        let n = self.encode_silk_frame(rc, &mut buf[DRED_EXPERIMENTAL_BYTES..], dred_chunks, max_bytes);
        if n == 0 { 0 } else { n + DRED_EXPERIMENTAL_BYTES }
    }

    /// libopus `dred_encode_silk_frame`: codes the initial state and up to
    /// `max_chunks` latent vectors (every second one, newest first) in at
    /// most `max_bytes` bytes, written to `out`. Returns the number of
    /// bytes, or 0 when not even one latent vector fits or when no voice
    /// activity is near.
    fn encode_silk_frame(&mut self, rc: &mut RangeCoder, out: &mut [u8], max_chunks: i32, max_bytes: usize) -> usize {
        let (q0, dq, qmax) = (self.q0, self.dq, self.qmax);
        let arch = self.dnn.arch();
        let enc = &mut self.enc;
        let activity_mem = &self.activity_mem;
        let fill = enc.latents_buffer_fill as i32;
        let mut latent_offset = enc.latent_offset;
        let mut extra_dred_offset = 0i32;
        let mut delayed_dred = false;
        // "Delaying new DRED data when just out of silence because we
        // already have the main Opus payload for that frame."
        if activity_mem[0] != 0 && enc.last_extra_dred_offset > 0 {
            latent_offset = enc.last_extra_dred_offset;
            delayed_dred = true;
            enc.last_extra_dred_offset = 0;
        }
        while latent_offset < fill && !voice_active(activity_mem, latent_offset) {
            latent_offset += 1;
            extra_dred_offset += 1;
        }
        if !delayed_dred {
            enc.last_extra_dred_offset = extra_dred_offset;
        }

        // "entropy coding of state and latents"
        rc.reset_for_encode(max_bytes as u32);
        rc.enc_uint(q0 as u32, 16);
        rc.enc_uint(dq as u32, 8);
        let total_offset = 16 - (enc.dred_offset - extra_dred_offset * 8);
        debug_assert!(total_offset >= 0);
        if total_offset > 31 {
            rc.enc_uint(1, 2);
            rc.enc_uint((total_offset >> 5) as u32, 256);
            rc.enc_uint((total_offset & 31) as u32, 32);
        } else {
            rc.enc_uint(0, 2);
            rc.enc_uint(total_offset as u32, 32);
        }
        debug_assert!(qmax >= q0);
        if q0 < 14 && dq > 0 {
            // "If you want to use qmax == q0, you should have set dQ = 0."
            let nvals = 15 - (q0 + 1);
            let (fl, fh) = if qmax >= 15 { (0, nvals) } else { (nvals + qmax - (q0 + 1), nvals + qmax - q0) };
            rc.encode(fl as u32, fh as u32, 2 * nvals as u32);
        }
        let state_q = q0 as usize * DRED_STATE_DIM;
        let state_at = latent_offset as usize * DRED_STATE_DIM;
        encode_latents(
            rc,
            &enc.state_buffer[state_at..state_at + DRED_STATE_DIM],
            &DRED_STATE_QUANT_SCALES_Q8[state_q..],
            &DRED_STATE_DEAD_ZONE_Q8[state_q..],
            &DRED_STATE_R_Q8[state_q..],
            &DRED_STATE_P0_Q8[state_q..],
            arch,
        );
        if rc.tell() > 8 * max_bytes as i32 {
            return 0;
        }
        let mut bak = rc.save_state();
        let mut prev_active = false;
        let mut dred_encoded = 0;
        let end = (2 * max_chunks).min(fill - latent_offset - 1);
        let mut i = 0;
        while i < end {
            let q_level = compute_quantizer(q0, dq as usize, qmax, i / 2) as usize;
            let offset = q_level * DRED_LATENT_DIM;
            let at = (i + latent_offset) as usize * DRED_LATENT_DIM;
            encode_latents(
                rc,
                &enc.latents_buffer[at..at + DRED_LATENT_DIM],
                &DRED_LATENT_QUANT_SCALES_Q8[offset..],
                &DRED_LATENT_DEAD_ZONE_Q8[offset..],
                &DRED_LATENT_R_Q8[offset..],
                &DRED_LATENT_P0_Q8[offset..],
                arch,
            );
            if rc.tell() > 8 * max_bytes as i32 {
                // "If we haven't been able to code one chunk, give up on
                // DRED completely."
                if i == 0 {
                    return 0;
                }
                break;
            }
            let active = voice_active(activity_mem, i + latent_offset);
            if active || prev_active {
                bak = rc.save_state();
                dred_encoded = i + 2;
            }
            prev_active = active;
            i += 2;
        }
        // "Avoid sending empty DRED packets."
        if dred_encoded == 0 || (dred_encoded <= 2 && extra_dred_offset != 0) {
            return 0;
        }
        rc.restore_state(&bak);
        let n = ((rc.tell() + 7) / 8) as usize;
        rc.shrink(n as u32);
        rc.done();
        out[..n].copy_from_slice(&rc.buf[..n]);
        n
    }
}

impl DredEnc {
    /// libopus `dred_encoder_reset` on a new state.
    fn new(fs: i32, channels: usize) -> Self {
        DredEnc {
            lpcnet: LpcnetEncState::new(),
            rdovae: RdovaeEncState::new(),
            fs,
            channels,
            input_buffer: [0.0; 2 * DRED_DFRAME_SIZE],
            input_buffer_fill: DRED_SILK_ENCODER_DELAY,
            dred_offset: 0,
            latent_offset: 0,
            last_extra_dred_offset: 0,
            latents_buffer: [0.0; DRED_MAX_FRAMES * DRED_LATENT_DIM],
            latents_buffer_fill: 0,
            state_buffer: [0.0; DRED_MAX_FRAMES * DRED_STATE_DIM],
            resample_mem: [0.0; RESAMPLING_ORDER + 1],
        }
    }

    /// libopus `dred_compute_latents`: adds the frame `pcm` to the
    /// 16 kHz buffer and runs `process_frame` on every 20 ms of it.
    fn compute_latents(
        &mut self,
        model: &DredEncModels,
        pcm: &[f32],
        frame_size: usize,
        extra_delay: usize,
        arch: DnnArch,
    ) {
        let fs = self.fs as usize;
        let mut frame_size16k = frame_size * 16000 / fs;
        let curr_offset16k = 40 + (extra_delay * 16000 / fs) as i32 - self.input_buffer_fill as i32;
        self.dred_offset = ((curr_offset16k as f32 + 20.0) / 40.0).floor() as i32;
        self.latent_offset = 0;
        let mut pos = 0;
        while frame_size16k > 0 {
            let process_size16k = (2 * DRED_FRAME_SIZE).min(frame_size16k);
            let process_size = process_size16k * fs / 16000;
            let fill = self.input_buffer_fill;
            self.convert_to_16k(&pcm[pos..], process_size, fill, process_size16k);
            self.input_buffer_fill += process_size16k;
            if self.input_buffer_fill >= 2 * DRED_FRAME_SIZE {
                self.process_frame(model, arch);
                self.input_buffer_fill -= 2 * DRED_FRAME_SIZE;
                self.input_buffer
                    .copy_within(2 * DRED_FRAME_SIZE..2 * DRED_FRAME_SIZE + self.input_buffer_fill, 0);
                // "15 ms (6*2.5 ms) is the ideal offset for DRED because it
                // corresponds to our vocoder look-ahead."
                if self.dred_offset < 6 {
                    self.dred_offset += 8;
                } else {
                    self.latent_offset += 1;
                }
            }
            // libopus moves on by `process_size` values, not by
            // `process_size` samples of every channel. Only frames longer
            // than 20 ms take this loop twice (SILK-only frames of 40 and
            // 60 ms); in stereo the next 20 ms are then read from the middle
            // of the first ones. Kept for the same packets.
            pos += process_size;
            frame_size16k -= process_size16k;
        }
    }

    /// libopus `dred_process_frame`: the features of the 20 ms at the start
    /// of `input_buffer` (two 10 ms frames), and one step of the RDOVAE
    /// encoder on them. The new latent vector and initial state go first.
    fn process_frame(&mut self, model: &DredEncModels, arch: DnnArch) {
        // "shift latents buffer"
        self.latents_buffer.copy_within(..(DRED_MAX_FRAMES - 1) * DRED_LATENT_DIM, DRED_LATENT_DIM);
        self.state_buffer.copy_within(..(DRED_MAX_FRAMES - 1) * DRED_STATE_DIM, DRED_STATE_DIM);
        // "calculate LPCNet features"
        let mut features = [[0.0f32; NB_TOTAL_FEATURES]; 2];
        for (k, f) in features.iter_mut().enumerate() {
            let frame = self.input_buffer[k * DRED_FRAME_SIZE..(k + 1) * DRED_FRAME_SIZE].try_into().unwrap();
            compute_single_frame_features(&mut self.lpcnet, frame, f, &model.pitchdnn, arch);
        }
        // "prepare input buffer (discard LPC coefficients)"
        let mut input = [0.0f32; 2 * DRED_NUM_FEATURES];
        input[..DRED_NUM_FEATURES].copy_from_slice(&features[0][..DRED_NUM_FEATURES]);
        input[DRED_NUM_FEATURES..].copy_from_slice(&features[1][..DRED_NUM_FEATURES]);
        // "run RDOVAE encoder"
        self.rdovae.encode_dframe(
            &model.rdovae,
            &mut self.latents_buffer[..DRED_LATENT_DIM],
            &mut self.state_buffer[..DRED_STATE_DIM],
            &input,
            arch,
        );
        self.latents_buffer_fill = (self.latents_buffer_fill + 1).min(DRED_NUM_REDUNDANCY_FRAMES);
    }

    /// libopus `dred_convert_to_16k`: `in_len` samples per channel of
    /// `pcm` to `out_len` samples of 16 kHz mono, written to
    /// `input_buffer[at..]`. The input is scaled to 16 bits and rounded,
    /// and stereo is averaged. Then:
    ///
    /// - 16 kHz is copied.
    /// - 48 kHz is low-pass filtered, and every third sample is kept.
    /// - 24 and 12 kHz first get 1 and 3 zeros after each sample (to 48 kHz),
    ///   then they are treated as 48 kHz, with a lower cutoff for 12 kHz.
    /// - 8 kHz gets one zero after each sample (to 16 kHz) and is low-pass
    ///   filtered.
    ///
    /// The filter coefficients are libopus's decimal literals, which have
    /// more digits than a float holds; both languages round them the same.
    #[allow(clippy::excessive_precision)]
    fn convert_to_16k(&mut self, pcm: &[f32], in_len: usize, at: usize, out_len: usize) {
        let up = match self.fs {
            8000 => 2,
            12000 => 4,
            16000 => 1,
            24000 => 2,
            48000 => 1,
            _ => unreachable!("the Opus encoder takes no other rate"),
        };
        debug_assert!(self.channels * in_len <= MAX_DOWNMIX_BUFFER && up * in_len <= MAX_DOWNMIX_BUFFER);
        let mut downmix = [0.0f32; MAX_DOWNMIX_BUFFER];
        if self.channels == 1 {
            for i in 0..in_len {
                downmix[up * i] = float2int16(up as f32 * pcm[i]) as f32 + VERY_SMALL;
            }
        } else {
            for i in 0..in_len {
                // C `.5*up*(in[2*i]+in[2*i+1])`: the product in double
                // precision.
                let x = (0.5 * up as f64 * (pcm[2 * i] + pcm[2 * i + 1]) as f64) as f32;
                downmix[up * i] = float2int16(x) as f32 + VERY_SMALL;
            }
        }
        let out = &mut self.input_buffer[at..at + out_len];
        match self.fs {
            16000 => out.copy_from_slice(&downmix[..out_len]),
            48000 | 24000 => {
                // "ellip(7, .2, 70, 7750/24000)"
                const B: [f32; 8] = [
                    0.005873358047, 0.012980854831, 0.014531340042, 0.014531340042, 0.012980854831,
                    0.005873358047, 0.004523418224, 0.,
                ];
                const A: [f32; 8] = [
                    -3.878718597768, 7.748834257468, -9.653651699533, 8.007342726666, -4.379450178552,
                    1.463182111810, -0.231720677804, 0.,
                ];
                let b0 = 0.004523418224f32;
                filter_df2t(&mut downmix[..up * in_len], b0, &B, &A, &mut self.resample_mem);
                for (i, o) in out.iter_mut().enumerate() {
                    *o = downmix[3 * i];
                }
            }
            12000 => {
                // "ellip(7, .2, 70, 5800/24000)"
                const B: [f32; 8] = [
                    -0.001017101081, 0.003673127243, 0.001009165267, 0.001009165267, 0.003673127243,
                    -0.001017101081, 0.002033596776, 0.,
                ];
                const A: [f32; 8] = [
                    -4.930414411612, 11.291643096504, -15.322037343815, 13.216403930898, -7.220409219553,
                    2.310550142771, -0.334338618782, 0.,
                ];
                let b0 = 0.002033596776f32;
                filter_df2t(&mut downmix[..up * in_len], b0, &B, &A, &mut self.resample_mem);
                for (i, o) in out.iter_mut().enumerate() {
                    *o = downmix[3 * i];
                }
            }
            _ => {
                // 8 kHz. "ellip(7, .2, 70, 3900/8000)"
                const B: [f32; 8] = [
                    0.081670120929, 0.180401598565, 0.259391051971, 0.259391051971, 0.180401598565,
                    0.081670120929, 0.020109185709, 0.,
                ];
                const A: [f32; 8] = [
                    -1.393651933659, 2.609789872676, -2.403541968806, 2.056814957331, -1.148908574570,
                    0.473001413788, -0.110359852412, 0.,
                ];
                let b0 = 0.020109185709f32;
                out.copy_from_slice(&downmix[..out_len]);
                filter_df2t(out, b0, &B, &A, &mut self.resample_mem);
            }
        }
    }
}

/// libopus `filter_df2t`: an IIR filter in transposed direct form II, in
/// place. clang fuses each product with the sum it goes into (one `fmadd`
/// or `fmsub` each), as `mul_add` does here.
fn filter_df2t(x: &mut [f32], b0: f32, b: &[f32; 8], a: &[f32; 8], mem: &mut [f32; RESAMPLING_ORDER + 1]) {
    for v in x.iter_mut() {
        let xi = *v;
        let yi = xi.mul_add(b0, mem[0]);
        let nyi = -yi;
        for j in 0..RESAMPLING_ORDER {
            mem[j] = a[j].mul_add(nyi, b[j].mul_add(xi, mem[j + 1]));
        }
        *v = yi;
    }
}

/// libopus `dred_encode_latents`: quantizes the `x.len()` values of `x`
/// with a dead zone and codes them with the Laplace code of DRED. A value
/// whose table entries make it impossible (`r` 0 or `p0` 255) is not
/// coded.
fn encode_latents(rc: &mut RangeCoder, x: &[f32], scale: &[u8], dzone: &[u8], r: &[u8], p0: &[u8], arch: DnnArch) {
    const MAX_DIM: usize = if DRED_LATENT_DIM > DRED_STATE_DIM { DRED_LATENT_DIM } else { DRED_STATE_DIM };
    let dim = x.len();
    let eps = 0.1f32;
    let mut xq = [0.0f32; MAX_DIM];
    let mut delta = [0.0f32; MAX_DIM];
    let mut deadzone = [0.0f32; MAX_DIM];
    for i in 0..dim {
        delta[i] = dzone[i] as f32 * (1.0 / 256.0);
        xq[i] = x[i] * scale[i] as f32 * (1.0 / 256.0);
        deadzone[i] = xq[i] / (delta[i] + eps);
    }
    compute_activation_in_place(&mut deadzone[..dim], Activation::Tanh, arch);
    for i in 0..dim {
        // C `xq[i] - delta[i]*deadzone[i]`, one `fmsub`.
        xq[i] = (-delta[i]).mul_add(deadzone[i], xq[i]);
        let q = (0.5 + xq[i]).floor() as i32;
        // "Make the impossible actually impossible."
        if r[i] != 0 && p0[i] != 255 {
            rc.laplace_encode_p0(q, (p0[i] as u16) << 7, (r[i] as u16) << 7);
        }
    }
}

/// libopus `dred_voice_active`: whether one of the 16 periods of 2.5 ms
/// around the latent vector `offset` was active.
///
/// For `offset` 51 libopus reads 8 bytes past `activity_mem` (the encoder
/// fields `nonfinal_frame` and `rangeFinal`). Here they count as not
/// active. Only the loop that skips silence reads there, when the latent
/// buffer is full; with `offset` 51 or 52 no latent vector is left to
/// code either way, so the packets are the same.
fn voice_active(activity_mem: &[u8; 4 * DRED_MAX_FRAMES], offset: i32) -> bool {
    let start = 8 * offset as usize;
    (start..start + 16).any(|i| activity_mem.get(i) == Some(&1))
}

/// libopus `estimate_dred_bitrate` (opus_encoder.c:669-685): the bits of
/// the DRED data with the initial state and the latent vectors of
/// `duration` frames of 10 ms, and how many latent vectors fit in
/// `target_bits`.
fn estimate_bitrate(q0: i32, dq: i32, qmax: i32, duration: i32, target_bits: i32) -> (i32, i32) {
    // "Signaling DRED costs 3 bytes."
    let mut bits = (8 * (3 + DRED_EXPERIMENTAL_BYTES)) as f32;
    // "Approximation for the size of the IS."
    bits += 50.0 + DRED_BITS_TABLE[q0 as usize];
    let dred_chunks = ((duration + 5) / 4).min(DRED_NUM_REDUNDANCY_FRAMES as i32 / 2);
    let mut target_chunks = 0;
    for i in 0..dred_chunks {
        let q = compute_quantizer(q0, dq as usize, qmax, i);
        bits += DRED_BITS_TABLE[q as usize];
        if bits < target_bits as f32 {
            target_chunks = i + 1;
        }
    }
    ((0.5 + bits).floor() as i32, target_chunks)
}

/// libopus `EC_ILOG` for a positive value: the number of bits it needs.
fn ec_ilog(x: i32) -> i32 {
    32 - (x as u32).leading_zeros() as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bits_table_and_ilog() {
        assert_eq!(ec_ilog(1), 1);
        assert_eq!(ec_ilog(36000), 16);
        // 64 kb/s without FEC: q0 = 51 - 3*16 = 3, limited to 4.
        let (bits, chunks) = estimate_bitrate(4, 5, 15, 100, 1000);
        assert!(bits > 0 && chunks > 0);
    }
}
