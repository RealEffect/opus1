//! The DRED decoder: `OpusDredDecoder` and `OpusDred` (libopus
//! `OpusDREDDecoder`, `OpusDRED`, `opus_dred_parse` and
//! `opus_dred_process` of `src/opus_decoder.c`, `dnn/dred_decoder.c`).
//!
//! Decoding DRED takes two steps:
//!
//! 1. `parse` finds the DRED extension in a packet and reads its range
//!    coded data: the initial state, the quantizer levels and the latent
//!    vectors. This is cheap.
//! 2. `process` runs the RDOVAE decoder, which turns the latent vectors
//!    into features, four 10 ms frames per vector. `parse` does it at once
//!    unless it is told to defer it.
//!
//! The processed `OpusDred` then goes to `OpusDecoder::dred_decode` (and
//! its float and 24-bit variants), which queues the features for deep PLC
//! and conceals the lost audio with them.

use super::coding::compute_quantizer;
use super::rdovae_dec;
use super::{DRED_EXPERIMENTAL_BYTES, DRED_EXPERIMENTAL_VERSION, DRED_EXTENSION_ID, DRED_NUM_REDUNDANCY_FRAMES};
use crate::dnn::DnnBlob;
use crate::dnn::holder::DredDecoderDnn;
use crate::dnn::models::dred_rdovae_stats::{
    DRED_LATENT_DIM, DRED_LATENT_P0_Q8, DRED_LATENT_QUANT_SCALES_Q8, DRED_LATENT_R_Q8, DRED_NUM_FEATURES,
    DRED_STATE_DIM, DRED_STATE_P0_Q8, DRED_STATE_QUANT_SCALES_Q8, DRED_STATE_R_Q8,
};
use crate::extensions::ExtensionIterator;
use crate::packet;
use crate::range_coder::RangeCoder;

const BAD_ARG: &str = "Invalid argument";
const UNIMPLEMENTED: &str = "Request not implemented";

/// The number of feature values an `OpusDred` holds: 104 frames of 10 ms.
const FEC_FEATURES: usize = 2 * DRED_NUM_REDUNDANCY_FRAMES * DRED_NUM_FEATURES;
/// The number of latent values an `OpusDred` holds: 26 vectors, each with
/// its quantizer level as one more value.
const LATENTS: usize = (DRED_NUM_REDUNDANCY_FRAMES / 2) * (DRED_LATENT_DIM + 1);

/// The DRED data of one packet (libopus `OpusDRED`, 11,236 bytes).
///
/// [`OpusDredDecoder::parse`] fills it from a packet and
/// [`OpusDredDecoder::process`] turns it into features. It can then be
/// passed to [`OpusDecoder::dred_decode`](crate::OpusDecoder::dred_decode)
/// as often as needed: decoding does not change it.
#[derive(Clone)]
pub struct OpusDred {
    /// Features, 20 per 10 ms frame. Frame `k` is the `k`-th frame back in
    /// time from the newest DRED frame.
    pub(crate) fec_features: [f32; FEC_FEATURES],
    pub(crate) state: [f32; DRED_STATE_DIM],
    pub(crate) latents: [f32; LATENTS],
    pub(crate) nb_latents: i32,
    /// -1: nothing parsed (or the last parse failed or found no DRED),
    /// 1: parsed, 2: processed.
    pub(crate) process_stage: i32,
    /// The position of the newest DRED frame, in units of 2.5 ms before
    /// the start of the packet's first frame (negative: after it).
    pub(crate) dred_offset: i32,
}

impl OpusDred {
    /// An empty DRED state (libopus `opus_dred_alloc`). It holds nothing
    /// until [`OpusDredDecoder::parse`] fills it.
    pub fn new() -> Self {
        OpusDred {
            fec_features: [0.0; FEC_FEATURES],
            state: [0.0; DRED_STATE_DIM],
            latents: [0.0; LATENTS],
            nb_latents: 0,
            process_stage: -1,
            dred_offset: 0,
        }
    }

    /// The number of latent vectors the last parse read. Each stands for
    /// 40 ms (four frames of 10 ms).
    pub fn nb_latents(&self) -> usize {
        self.nb_latents.max(0) as usize
    }

    /// The position of the newest DRED data, in units of 2.5 ms before the
    /// start of the packet (libopus `dred_offset`). It is negative when the
    /// encoder left out the newest frames, for example in silence.
    pub fn dred_offset(&self) -> i32 {
        self.dred_offset
    }

    /// True when the state holds parsed data (libopus `process_stage` 1 or 2).
    pub fn is_parsed(&self) -> bool {
        self.process_stage == 1 || self.process_stage == 2
    }

    /// True when the state holds processed features, ready for
    /// `OpusDecoder::dred_decode` (libopus `process_stage == 2`).
    pub fn is_processed(&self) -> bool {
        self.process_stage == 2
    }

    /// The decoded features: `4 * nb_latents()` frames of 20 values, frame
    /// `k` being the `k`-th 10 ms frame back in time from the newest. Only
    /// meaningful after processing.
    pub fn features(&self) -> &[f32] {
        &self.fec_features[..4 * self.nb_latents() * DRED_NUM_FEATURES]
    }

    /// The initial state vector the last parse read (50 values).
    pub fn state(&self) -> &[f32; DRED_STATE_DIM] {
        &self.state
    }

    /// The latent vectors the last parse read: `nb_latents()` vectors of 25
    /// values, each followed by its quantizer level `q` as `q / 8 - 1`.
    pub fn latents(&self) -> &[f32] {
        &self.latents[..self.nb_latents() * (DRED_LATENT_DIM + 1)]
    }
}

impl Default for OpusDred {
    fn default() -> Self {
        OpusDred::new()
    }
}

/// What [`OpusDredDecoder::parse`] found in a packet.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DredParse {
    /// libopus's return value: how far back the DRED data reaches, in
    /// samples at the `sampling_rate` of the call, counted back from the
    /// start of the packet. 0 when the packet carries no DRED data.
    pub samples: i32,
    /// libopus `dred_end`: the samples between the start of the packet and
    /// the newest DRED data, when the encoder left out the newest frames.
    /// Usually 0.
    pub end: i32,
}

/// The DRED decoder (libopus `OpusDREDDecoder`): the RDOVAE decoder's
/// weights and the kernel variant. It keeps no state between packets, so
/// one decoder can serve any number of streams.
pub struct OpusDredDecoder {
    dnn: DredDecoderDnn,
}

impl OpusDredDecoder {
    /// A DRED decoder with the built-in weights (libopus
    /// `opus_dred_decoder_create` and `opus_dred_decoder_init`).
    pub fn new() -> Self {
        OpusDredDecoder { dnn: DredDecoderDnn::new() }
    }

    /// libopus `OPUS_SET_DNN_BLOB` on a DRED decoder: loads the RDOVAE
    /// decoder from a blob in the libopus format. When the blob does not
    /// hold all of its arrays, returns `"DNN blob holds no model of this
    /// build"` and keeps the weights it had. [`DnnBlob::builtin`] brings
    /// back the built-in weights.
    ///
    /// libopus offers the request only when built with `USE_WEIGHTS_FILE`,
    /// returns `OPUS_BAD_ARG` on a failure, and can leave the model half
    /// replaced.
    pub fn set_dnn_blob(&mut self, blob: &DnnBlob) -> Result<(), &'static str> {
        self.dnn.load(blob)
    }

    /// True when the RDOVAE decoder is loaded (libopus `loaded`). Without
    /// it, `parse` and `process` return `"Request not implemented"`.
    pub fn is_loaded(&self) -> bool {
        self.dnn.model().is_some()
    }

    /// Reads the DRED data of a packet into `dred` (libopus
    /// `opus_dred_parse`).
    ///
    /// - `data`: a whole Opus packet, not self-delimited.
    /// - `max_dred_samples`: how much audio before the packet may be
    ///   needed, in samples at `sampling_rate`. Only the latent vectors
    ///   that cover it are read, plus up to two 10 ms frames of margin.
    /// - `sampling_rate`: the rate of `max_dred_samples` and of the result.
    ///   It need not be the rate of the decoder.
    /// - `defer_processing`: when true, the RDOVAE decoder does not run;
    ///   call [`process`](Self::process) before decoding audio.
    ///
    /// Returns what was found ([`DredParse`]). A packet without DRED data
    /// gives zeros and leaves `dred` unusable for decoding. Errors:
    /// `"Request not implemented"` without a loaded model; `"Invalid
    /// packet"` when the packet or its extensions are malformed;
    /// `"Invalid argument"` when `sampling_rate` is not positive (libopus
    /// divides by it). Every call with a loaded model first marks `dred` as
    /// not parsed, as libopus does, so after an error it cannot be decoded.
    pub fn parse(
        &self,
        dred: &mut OpusDred,
        data: &[u8],
        max_dred_samples: i32,
        sampling_rate: i32,
        defer_processing: bool,
    ) -> Result<DredParse, &'static str> {
        if !self.is_loaded() {
            return Err(UNIMPLEMENTED);
        }
        dred.process_stage = -1;
        if sampling_rate <= 0 {
            return Err(BAD_ARG);
        }
        let Some((payload, dred_frame_offset)) = find_payload(data)? else {
            return Ok(DredParse::default());
        };
        let offset = 100i32.wrapping_mul(max_dred_samples) / sampling_rate;
        let min_feature_frames = 2i32.wrapping_add(offset).min(2 * DRED_NUM_REDUNDANCY_FRAMES as i32);
        ec_decode(dred, payload, min_feature_frames, dred_frame_offset);
        if !defer_processing {
            self.process_in_place(dred)?;
        }
        let end = (-dred.dred_offset).wrapping_mul(sampling_rate) / 400;
        let samples = (dred.nb_latents.wrapping_mul(sampling_rate) / 25)
            .wrapping_sub(dred.dred_offset.wrapping_mul(sampling_rate) / 400);
        Ok(DredParse { samples: samples.max(0), end: end.max(0) })
    }

    /// Runs the RDOVAE decoder on a parsed `dred` (libopus
    /// `opus_dred_process` with `src == dst`). A state already processed
    /// is left as it is. Errors: `"Invalid argument"` when `dred` holds no
    /// parsed data; `"Request not implemented"` without a loaded model.
    pub fn process_in_place(&self, dred: &mut OpusDred) -> Result<(), &'static str> {
        if !dred.is_parsed() {
            return Err(BAD_ARG);
        }
        let Some(model) = self.dnn.model() else {
            return Err(UNIMPLEMENTED);
        };
        if dred.process_stage == 2 {
            return Ok(());
        }
        rdovae_dec::decode_all(
            model,
            &mut dred.fec_features,
            &dred.state,
            &dred.latents,
            dred.nb_latents.max(0) as usize,
            self.dnn.arch(),
        );
        dred.process_stage = 2;
        Ok(())
    }

    /// libopus `opus_dred_process` with two states: copies `src` to `dst`
    /// and processes `dst`. `src` stays as it is, so a deferred parse can
    /// be processed later, or more than once. The errors are those of
    /// [`process_in_place`](Self::process_in_place), and then `dst` is
    /// unchanged.
    pub fn process(&self, src: &OpusDred, dst: &mut OpusDred) -> Result<(), &'static str> {
        if !src.is_parsed() {
            return Err(BAD_ARG);
        }
        if !self.is_loaded() {
            return Err(UNIMPLEMENTED);
        }
        dst.clone_from(src);
        self.process_in_place(dst)
    }
}

impl Default for OpusDredDecoder {
    fn default() -> Self {
        OpusDredDecoder::new()
    }
}

/// libopus `dred_find_payload`: the DRED payload of a packet, without its
/// 2-byte header, and the position of the frame that carries it in units of
/// 2.5 ms. The first DRED extension with the right header wins.
fn find_payload(data: &[u8]) -> Result<Option<(&[u8], i32)>, &'static str> {
    let parsed = packet::parse(data, false)?;
    let frame_size = packet::samples_per_frame(data[0], 48000);
    let mut iter = ExtensionIterator::new(parsed.padding(data), parsed.count);
    while let Some(ext) = iter.find(DRED_EXTENSION_ID)? {
        // DRED position in the packet, in units of 2.5 ms like the offset
        // that the DRED data signals.
        let dred_frame_offset = (ext.frame * frame_size / 120) as i32;
        // The check of the experimental header, as long as DRED has no ID
        // of its own.
        if ext.data.len() > DRED_EXPERIMENTAL_BYTES && ext.data[0] == b'D' && ext.data[1] == DRED_EXPERIMENTAL_VERSION {
            return Ok(Some((&ext.data[DRED_EXPERIMENTAL_BYTES..], dred_frame_offset)));
        }
    }
    Ok(None)
}

/// libopus `dred_decode_latents`: one vector of Laplace-coded values,
/// scaled back by the quantizer's scales. A value with `r == 0` or
/// `p0 == 255` is not coded and is 0.
fn decode_latents(ec: &mut RangeCoder, x: &mut [f32], scale: &[u8], r: &[u8], p0: &[u8]) {
    for (i, x) in x.iter_mut().enumerate() {
        let q = if r[i] == 0 || p0[i] == 255 {
            0
        } else {
            ec.laplace_decode_p0((p0[i] as u16) << 7, (r[i] as u16) << 7)
        };
        let s = if scale[i] == 0 { 1 } else { scale[i] };
        *x = q as f32 * 256.0 / s as f32;
    }
}

/// libopus `dred_ec_decode`: reads the range-coded DRED payload into
/// `dred` and marks it as parsed. At most `(min_feature_frames + 1) / 2`
/// frames are read, in latent vectors of two 20 ms frames, newest first,
/// and reading stops when fewer than 8 bits are left. Returns the number
/// of latent vectors read.
///
/// The header holds the quantizer level `q0` of the newest vector, the
/// step index `dq` by which the level rises, the largest level `qmax`
/// (coded only when it can be below 15) and the offset of the newest
/// frame. `dred_frame_offset` adds the position of the extension's frame
/// in the packet.
pub(crate) fn ec_decode(dred: &mut OpusDred, bytes: &[u8], min_feature_frames: i32, dred_frame_offset: i32) -> i32 {
    let mut ec = RangeCoder::new_decoder(bytes);
    let q0 = ec.dec_uint(16) as i32;
    let dq = ec.dec_uint(8) as i32;
    let extra_offset = if ec.dec_uint(2) != 0 { 32 * ec.dec_uint(256) as i32 } else { 0 };
    // The total offset, with the position of DRED in a multiframe packet.
    dred.dred_offset = 16 - ec.dec_uint(32) as i32 - extra_offset + dred_frame_offset;
    let mut qmax = 15;
    if q0 < 14 && dq > 0 {
        // One symbol says whether qmax is 15 (half of the range) or which
        // of the levels between q0 + 1 and 14 it is (the other half, each
        // value equally likely).
        let nvals = 15 - (q0 + 1);
        let ft = (2 * nvals) as u32;
        let s = ec.decode(ft) as i32;
        if s >= nvals {
            qmax = q0 + (s - nvals) + 1;
            ec.update(s as u32, s as u32 + 1, ft);
        } else {
            ec.update(0, nvals as u32, ft);
        }
    }
    let state_qoffset = q0 as usize * DRED_STATE_DIM;
    let state_tables = state_qoffset..state_qoffset + DRED_STATE_DIM;
    decode_latents(
        &mut ec,
        &mut dred.state,
        &DRED_STATE_QUANT_SCALES_Q8[state_tables.clone()],
        &DRED_STATE_R_Q8[state_tables.clone()],
        &DRED_STATE_P0_Q8[state_tables],
    );
    // Decode the newest vector first and store them in that order.
    let total_bits = 8 * bytes.len() as i64;
    let limit = (DRED_NUM_REDUNDANCY_FRAMES as i32).min((min_feature_frames.wrapping_add(1)) / 2);
    let mut i = 0i32;
    while i < limit {
        // libopus: "FIXME: Figure out how to avoid missing a last frame
        // that would take up < 8 bits."
        if total_bits - ec.tell() as i64 <= 7 {
            break;
        }
        let q_level = compute_quantizer(q0, dq as usize, qmax, i / 2);
        let offset = q_level as usize * DRED_LATENT_DIM;
        let tables = offset..offset + DRED_LATENT_DIM;
        let base = (i / 2) as usize * (DRED_LATENT_DIM + 1);
        decode_latents(
            &mut ec,
            &mut dred.latents[base..base + DRED_LATENT_DIM],
            &DRED_LATENT_QUANT_SCALES_Q8[tables.clone()],
            &DRED_LATENT_R_Q8[tables.clone()],
            &DRED_LATENT_P0_Q8[tables],
        );
        dred.latents[base + DRED_LATENT_DIM] = (q_level as f64 * 0.125 - 1.0) as f32;
        i += 2;
    }
    dred.process_stage = 1;
    dred.nb_latents = i / 2;
    i / 2
}
