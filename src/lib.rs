#![cfg_attr(not(feature = "std"), no_std)]
#![allow(unsafe_op_in_unsafe_fn)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]

// The unit tests use `vec!`, `format!` and `std::` paths. In a `no_std`
// build they get them from `std`, which a test binary links anyway; the
// library code itself stays `no_std`. With `std` in the test binary its
// float methods are found before `compat::Math`, whose imports then look
// unused.
#![cfg_attr(all(test, not(feature = "std")), allow(unused_imports))]
#[cfg(all(test, not(feature = "std")))]
#[macro_use]
extern crate std;

mod compat;
mod fixedvec;
mod mathops;

mod bands;
mod celt;
mod celt_lpc;
mod hp_cutoff;
#[doc(hidden)]
pub mod internals;
mod kiss_fft;
mod mdct;
mod modes;
/// Multichannel encoding and decoding (several Opus streams in one packet).
/// It allocates in its constructors, so it needs the `std` feature (on by
/// default). The single-stream codec stays `no_std` without `alloc`.
#[cfg(feature = "std")]
pub mod multistream;
pub mod packet;
mod pitch;
mod pvq;
mod quant_bands;
mod range_coder;
mod rate;
mod silk;

/// The Rust examples in `README.md` and in `docs/`. `cargo test --doc`
/// compiles and runs every ` ```rust ` block in these files, so the
/// examples stay in step with the code. Nothing here is part of the
/// library: these items exist only when rustdoc collects doc tests. Some
/// examples use the `multistream` module, so they need the `std` feature.
#[cfg(all(doctest, feature = "std"))]
mod doc_examples {
    #[doc = include_str!("../README.md")]
    struct Readme;

    #[doc = include_str!("../docs/01-what-is-opus.md")]
    struct WhatIsOpus;

    #[doc = include_str!("../docs/02-packets.md")]
    struct Packets;

    #[doc = include_str!("../docs/03-encoder.md")]
    struct Encoder;

    #[doc = include_str!("../docs/04-decoder.md")]
    struct Decoder;

    #[doc = include_str!("../docs/05-multichannel.md")]
    struct Multichannel;

    #[doc = include_str!("../docs/06-using-the-library.md")]
    struct UsingTheLibrary;

    #[doc = include_str!("../docs/07-architecture.md")]
    struct Architecture;

    #[doc = include_str!("../docs/08-performance.md")]
    struct Performance;

    #[doc = include_str!("../docs/09-testing.md")]
    struct Testing;

    #[doc = include_str!("../docs/10-differences-from-libopus.md")]
    struct DifferencesFromLibopus;
}

#[cfg(not(feature = "std"))]
use crate::compat::Math;
use crate::fixedvec::FixedVec;
use celt::{CeltDecoder, CeltEncoder};
use hp_cutoff::{dc_reject_float, hp_cutoff, hp_cutoff_float};
use range_coder::RangeCoder;
use silk::enc_api::{SilkControl, SilkEncoder, SilkInput, SilkPrefill};
use silk::lin2log::silk_lin2log;
use silk::log2lin::silk_log2lin;
use silk::macros::*;

// --- Heap-free buffer capacity constants (worst case: 2 channels). ---
const OPUS_MAX_CHANNELS: usize = 2;
/// Largest API frame in samples/channel (120 ms at 48 kHz). The encoder's
/// SILK input buffer (`buf_filtered`) has this size too: it holds one part
/// of a packet, at most 60 ms, of both channels.
const OPUS_MAX_FRAME: usize = packet::MAX_PACKET_SAMPLES;
/// Largest frame in samples per channel: 60 ms at 48 kHz.
const OPUS_MAX_SUBFRAME: usize = 2880;
/// High-pass filter state memory (`channels * 2`).
const OPUS_HP_MEM: usize = OPUS_MAX_CHANNELS * 2;
/// Decoder `w_pcm_i16` and `w_silk_out` cap (`960 * max_channels`): one
/// SILK frame of 20 ms at 48 kHz, interleaved.
const OPUS_PCM_I16: usize = 960 * OPUS_MAX_CHANNELS;
/// Largest packet of interleaved samples: 120 ms at 48 kHz, stereo. The
/// decoder's output and `encode_i16`'s converted input have this size.
const OPUS_PCM_PACKET: usize = OPUS_MAX_FRAME * OPUS_MAX_CHANNELS;
/// Largest 5 ms of interleaved decoder output (`240 * max_channels`): the
/// size of a redundancy frame or of the concealment that bridges a mode
/// switch.
const OPUS_PCM_TAIL: usize = 240 * OPUS_MAX_CHANNELS;
/// The 2.5 ms a CELT encoder codes in front of a frame after a reset, at
/// 48 kHz, for each channel: C `tmp_prefill`.
const CELT_PREFILL: usize = 120 * OPUS_MAX_CHANNELS;
/// The largest packet of one frame: the TOC byte and a frame of
/// `packet::MAX_FRAME_BYTES` bytes (RFC 6716 §3.1).
pub(crate) const OPUS_MAX_PACKET_BYTES: usize = 1 + packet::MAX_FRAME_BYTES;
/// C `OpusEncoder.delay_buffer[MAX_ENCODER_BUFFER*2]` (480 samples * 2 ch).
/// Holds the delay-compensation ring feeding the CELT encoder.
const OPUS_DELAY_BUF: usize = 480 * OPUS_MAX_CHANNELS;
/// CELT/hybrid input staging cap: `(delay_compensation + frame)*channels`.
/// Worst case 60 ms stereo @ 48 kHz: (192 + 2880) * 2 = 6144.
const OPUS_PCM_BUF: usize = (192 + OPUS_MAX_SUBFRAME) * OPUS_MAX_CHANNELS;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Application {
    Voip = 2048,
    Audio = 2049,
    RestrictedLowDelay = 2051,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bandwidth {
    Auto = OPUS_AUTO as isize,
    Narrowband = 1101,
    Mediumband = 1102,
    Wideband = 1103,
    Superwideband = 1104,
    Fullband = 1105,
}

impl Bandwidth {
    /// The number of CELT bands a packet of this bandwidth codes (libopus
    /// `CELT_SET_END_BAND` in `opus_encode_frame_native` and
    /// `opus_decode_frame`). `Auto` counts as fullband.
    ///
    /// This holds for Hybrid packets as well as CELT-only ones: a Hybrid
    /// super-wideband packet codes bands 17 and 18 only.
    #[inline]
    pub(crate) fn celt_end_band(self) -> usize {
        match self {
            Bandwidth::Narrowband => 13,
            Bandwidth::Mediumband | Bandwidth::Wideband => 17,
            Bandwidth::Superwideband => 19,
            Bandwidth::Fullband | Bandwidth::Auto => 21,
        }
    }

    /// The internal sampling rate of SILK for a SILK-only packet of this
    /// bandwidth, in kHz: 8 for narrowband, 12 for mediumband, 16 otherwise.
    #[inline]
    pub(crate) fn silk_khz(self) -> i32 {
        match self {
            Bandwidth::Narrowband => 8,
            Bandwidth::Mediumband => 12,
            _ => 16,
        }
    }
}

/// The first CELT band of a Hybrid packet. SILK codes the bands below it,
/// up to 8 kHz.
const HYBRID_START_BAND: usize = 17;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OpusMode {
    SilkOnly,
    Hybrid,
    CeltOnly,
}

/// A coding mode the caller asks the encoder to use: libopus
/// `MODE_SILK_ONLY`, `MODE_HYBRID` and `MODE_CELT_ONLY` for
/// `OPUS_SET_FORCE_MODE`. See [`OpusEncoder::set_force_mode`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ForcedMode {
    /// SILK alone: speech up to wideband (8 kHz of audio).
    SilkOnly,
    /// SILK below 8 kHz and CELT above: super-wideband and fullband speech.
    Hybrid,
    /// CELT alone: music, and frames shorter than 10 ms.
    CeltOnly,
}

/// The bitrate setting of an encoder: libopus `OPUS_SET_BITRATE`. See
/// [`OpusEncoder::set_bitrate`]; `MultistreamEncoder::set_bitrate` takes it
/// too.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bitrate {
    /// libopus `OPUS_AUTO`, the default: the encoder picks the bitrate from
    /// the sampling rate, the channel count and the frame duration. An
    /// [`OpusEncoder`] takes `60 * rate / frame_size + rate * channels` bits
    /// per second: 19,000 b/s for 16 kHz mono and 99,000 b/s for 48 kHz
    /// stereo in 20 ms frames.
    Auto,
    /// libopus `OPUS_BITRATE_MAX`: as many bits as the output buffer holds.
    /// An [`OpusEncoder`] takes at most 1,500,000 b/s.
    Max,
    /// Bits per second for all channels together.
    BitsPerSecond(i32),
}

/// The frame duration setting of an encoder: libopus
/// `OPUS_SET_EXPERT_FRAME_DURATION`. See
/// [`OpusEncoder::set_expert_frame_duration`];
/// `MultistreamEncoder::set_expert_frame_duration` takes it too.
///
/// The number of each value is the libopus constant of the same name
/// (`OPUS_FRAMESIZE_ARG` is 5000, `OPUS_FRAMESIZE_2_5_MS` is 5001, and so
/// on). [`FrameDuration::try_from`] takes those numbers and refuses every
/// other one, as libopus's encoder control does.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FrameDuration {
    /// libopus `OPUS_FRAMESIZE_ARG`, the default: each frame lasts as long
    /// as the `frame_size` argument of the encode call says.
    Argument = 5000,
    /// 2.5 ms (`OPUS_FRAMESIZE_2_5_MS`).
    Ms2_5 = 5001,
    /// 5 ms (`OPUS_FRAMESIZE_5_MS`).
    Ms5 = 5002,
    /// 10 ms (`OPUS_FRAMESIZE_10_MS`).
    Ms10 = 5003,
    /// 20 ms (`OPUS_FRAMESIZE_20_MS`).
    Ms20 = 5004,
    /// 40 ms (`OPUS_FRAMESIZE_40_MS`).
    Ms40 = 5005,
    /// 60 ms (`OPUS_FRAMESIZE_60_MS`).
    Ms60 = 5006,
    /// 80 ms (`OPUS_FRAMESIZE_80_MS`).
    Ms80 = 5007,
    /// 100 ms (`OPUS_FRAMESIZE_100_MS`).
    Ms100 = 5008,
    /// 120 ms (`OPUS_FRAMESIZE_120_MS`).
    Ms120 = 5009,
}

impl FrameDuration {
    /// Every value, in the order of their libopus numbers.
    pub const ALL: [FrameDuration; 10] = [
        FrameDuration::Argument,
        FrameDuration::Ms2_5,
        FrameDuration::Ms5,
        FrameDuration::Ms10,
        FrameDuration::Ms20,
        FrameDuration::Ms40,
        FrameDuration::Ms60,
        FrameDuration::Ms80,
        FrameDuration::Ms100,
        FrameDuration::Ms120,
    ];

    /// The libopus number of the value, for example 5000 for
    /// [`Argument`](Self::Argument).
    pub fn to_raw(self) -> i32 {
        self as i32
    }

    /// The frame size in samples per channel at `sampling_rate`, or `None`
    /// for [`Argument`](Self::Argument). The arithmetic is that of libopus
    /// `frame_size_select`, and gives a whole number at every sampling rate
    /// Opus has.
    pub fn samples(self, sampling_rate: i32) -> Option<usize> {
        let fs = sampling_rate as usize;
        Some(match self {
            FrameDuration::Argument => return None,
            FrameDuration::Ms2_5 => fs / 400,
            FrameDuration::Ms5 => fs / 400 * 2,
            FrameDuration::Ms10 => fs / 400 * 4,
            FrameDuration::Ms20 => fs / 400 * 8,
            FrameDuration::Ms40 => fs / 400 * 16,
            FrameDuration::Ms60 => 3 * fs / 50,
            FrameDuration::Ms80 => 4 * fs / 50,
            FrameDuration::Ms100 => 5 * fs / 50,
            FrameDuration::Ms120 => 6 * fs / 50,
        })
    }
}

impl TryFrom<i32> for FrameDuration {
    type Error = &'static str;

    /// The value with the libopus number `value` (5000 to 5009). Any other
    /// number is refused, as libopus's `OPUS_SET_EXPERT_FRAME_DURATION`
    /// refuses it with `OPUS_BAD_ARG`.
    fn try_from(value: i32) -> Result<Self, Self::Error> {
        FrameDuration::ALL
            .into_iter()
            .find(|d| d.to_raw() == value)
            .ok_or("Invalid frame duration")
    }
}

/// The decisions `opus_encode_native` makes once per packet and hands to
/// `opus_encode_frame_native` for each frame of the packet.
#[derive(Debug, Clone, Copy)]
struct FrameDecisions {
    mode: OpusMode,
    bandwidth: Bandwidth,
    /// Whether the packet codes in-band FEC (libopus `LBRR_coded`).
    lbrr_coded: i32,
    /// C `redundancy`: the frame carries a 5 ms CELT frame that bridges a
    /// switch between CELT-only and SILK-only or Hybrid.
    redundancy: bool,
    /// C `celt_to_silk`: the redundant frame comes before the main frame
    /// (the stream leaves CELT-only). Otherwise it comes after it.
    celt_to_silk: bool,
    /// C `to_celt`: the next packet is CELT-only. This one is still coded
    /// in the previous mode, with the redundant frame at its end.
    to_celt: bool,
    /// C `prefill`: the SILK encoder restarts after CELT-only packets and
    /// is first fed the 10 ms before this frame.
    prefill: bool,
    /// C `equiv_rate`: the bitrate the decisions went by.
    equiv_rate: i32,
}

/// libopus `compute_redundancy_bytes` (opus_encoder.c:1142): the size of the
/// redundant 5 ms CELT frame, or 0 when the packet has no room for one that
/// is worth its bits.
fn compute_redundancy_bytes(
    max_data_bytes: usize,
    bitrate_bps: i32,
    frame_rate: i32,
    channels: usize,
) -> usize {
    let channels = channels as i32;
    let base_bits = 40 * channels + 20;
    // Equivalent rate for 5 ms frames.
    let redundancy_rate = bitrate_bps + base_bits * (200 - frame_rate);
    // "For VBR, further increase the bitrate if we can afford it."
    let redundancy_rate = 3 * redundancy_rate / 2;
    let mut redundancy_bytes = redundancy_rate / 1600;
    // The largest rate that fits, given CBR or VBR with a cap.
    let available_bits = max_data_bytes as i32 * 8 - 2 * base_bits;
    let cap = (available_bits * 240 / (240 + 48000 / frame_rate) + base_bits) / 8;
    redundancy_bytes = redundancy_bytes.min(cap);
    // "If we can't get enough bits for redundancy to be worth it, rely on
    // the decoder PLC."
    if redundancy_bytes > 4 + 8 * channels {
        redundancy_bytes.min(257) as usize
    } else {
        0
    }
}

/// libopus `stereo_fade` (opus_encoder.c:548): narrow the stereo image of an
/// interleaved stereo frame. A gain of 1 keeps the signal and a gain of 0
/// leaves the average of the two channels in both. The gain moves from `g1`
/// to `g2` over CELT's overlap (2.5 ms), following the squared CELT window.
fn stereo_fade(pcm: &mut [f32], g1: f32, g2: f32, frame_size: usize, fs: i32) {
    let window = modes::default_mode().window;
    let inc = celt::resampling_factor(fs).unwrap_or(1);
    let overlap = (window.len() / inc).min(frame_size);
    let (g1, g2) = (1.0 - g1, 1.0 - g2);
    let pcm = &mut pcm[..2 * frame_size];
    for i in 0..frame_size {
        let g = if i < overlap {
            let w = window[i * inc] * window[i * inc];
            w * g2 + (1.0 - w) * g1
        } else {
            g2
        };
        let diff = g * (0.5 * (pcm[2 * i] - pcm[2 * i + 1]));
        pcm[2 * i] -= diff;
        pcm[2 * i + 1] += diff;
    }
}

/// The test libopus makes on the filtered input of a frame
/// (opus_encoder.c:2016-2028): "This should filter out both NaNs and
/// ridiculous signals that could cause NaNs further down". The frame is
/// sane when the sum of its squares is below 1e9 and is a number.
///
/// The sum is first taken in eight partial sums, which the compiler turns
/// into vector code. The squares are not negative, so the order of the
/// additions changes the result by a few parts in a million at most. Only
/// when that sum is within 1 % of the limit, or not a number, is the sum
/// taken again one sample after the other, which is the sum the decision
/// was always made on.
#[inline]
fn input_is_sane(x: &[f32]) -> bool {
    const LIMIT: f32 = 1e9;
    let mut partial = [0.0f32; 8];
    let (blocks, rest) = x.split_at(x.len() - x.len() % 8);
    for b in 0..blocks.len() / 8 {
        let block = &blocks[8 * b..8 * b + 8];
        for k in 0..8 {
            partial[k] += block[k] * block[k];
        }
    }
    let mut quick = 0.0f32;
    for sum in partial {
        quick += sum;
    }
    for &v in rest {
        quick += v * v;
    }
    if quick < 0.99 * LIMIT {
        return true;
    }
    let mut sum = 0.0f32;
    for &v in x {
        sum += v * v;
    }
    // C: `!(sum < 1e9f) || celt_isnan(sum)` clears the frame.
    sum < LIMIT
}

/// Code `n` samples per channel of interleaved `pcm` as one CELT frame of
/// exactly `out.len()` bytes, all bands, at a constant bitrate: the
/// redundant frame of a mode switch (opus_encoder.c:2428-2441 and
/// 2513-2538), and the 2.5 ms in front of the one at the end of a packet.
/// Returns the final range of the frame's range coder.
fn encode_celt_redundancy(
    celt: &mut CeltEncoder,
    pcm: &[f32],
    n: usize,
    out: &mut [u8],
) -> Result<u32, &'static str> {
    let mut rc = RangeCoder::new_encoder(out.len() as u32);
    let _ = celt.set_start_band(0);
    celt.set_vbr(false);
    let _ = celt.set_bitrate(OPUS_BITRATE_MAX);
    celt.encode(pcm, n, &mut rc);
    rc.done();
    if rc.error != 0 || rc.buf.len() < out.len() {
        return Err("CELT redundancy frame does not fit");
    }
    out.copy_from_slice(&rc.buf[..out.len()]);
    Ok(rc.rng)
}

pub struct OpusEncoder {
    #[cfg(not(feature = "heap"))]
    celt_enc: CeltEncoder,
    #[cfg(feature = "heap")]
    celt_enc: Box<CeltEncoder>,
    /// The SILK encoder, with a channel state and an input resampler for
    /// each input channel.
    silk: SilkEncoder,
    application: Application,
    sampling_rate: i32,
    channels: usize,
    /// C `st->bandwidth`: the audio bandwidth of the last packet. It is
    /// decided again for each packet, but SILK and Hybrid only reconsider
    /// the rate-dependent choice when SILK allows a switch.
    bandwidth: Bandwidth,
    /// C `st->auto_bandwidth`: the bandwidth the bitrate alone called for
    /// last time, kept for the hysteresis of the next decision.
    auto_bandwidth: Bandwidth,
    /// C `st->user_bandwidth`: a bandwidth the caller asked for
    /// (`set_bandwidth`), or `None` for the automatic choice.
    user_bandwidth: Option<Bandwidth>,
    /// C `st->max_bandwidth`: the widest bandwidth the automatic choice may
    /// pick (`set_max_bandwidth`).
    max_bandwidth: Bandwidth,
    /// C `st->user_forced_mode`: a mode the caller asked for
    /// (`set_force_mode`), or `None` for the automatic choice.
    user_forced_mode: Option<OpusMode>,
    /// C `st->lfe`: this encoder codes a low-frequency-effects channel.
    lfe: bool,
    /// C `st->energy_masking`: 21 values per channel that say how far each
    /// band lies above what the other channels of a surround stream mask,
    /// or `None` without a mask. libopus keeps a pointer to the caller's
    /// array; this is a copy.
    energy_mask: Option<[f32; 21 * OPUS_MAX_CHANNELS]>,
    /// C `st->rangeFinal`: the state of the range coder after the last
    /// frame, 0 when the packet carries no range-coded data.
    range_final: u32,
    /// C `st->first`: no packet has been coded yet.
    first: bool,
    /// C `silk_mode.allowBandwidthSwitch`: SILK's speech activity is low
    /// enough for a bandwidth switch to go unheard.
    silk_allow_bandwidth_switch: bool,
    /// C `silk_mode.inWBmodeWithoutVariableLP`: SILK runs at 16 kHz and is
    /// not in a low-pass transition.
    silk_in_wb_mode_without_variable_lp: bool,
    /// C `silk_mode.opusCanSwitch`: SILK reported on the last packet that it
    /// is ready to change its internal rate, so it may on this one.
    silk_opus_can_switch: bool,
    /// C `st->silk_mode.LBRR_coded`: the last packet was to carry in-band
    /// FEC. The next decision leans the same way (`decide_fec`). CELT-only
    /// packets clear it. libopus keeps it across `OPUS_RESET_STATE`.
    silk_lbrr_coded: i32,
    /// C `st->nonfinal_frame`: this part of a packet made of several frames
    /// is not the last one.
    nonfinal_frame: bool,
    /// C `st->stream_channels`: the channel count the packets code. Stereo
    /// input is coded as mono at low bitrates (`decide_stream_channels`).
    stream_channels: usize,
    /// C `st->force_channels` (`OPUS_SET_FORCE_CHANNELS`): the channel count
    /// the caller asks the packets to code, or `None` for the automatic
    /// choice.
    force_channels: Option<usize>,
    /// C `st->prev_channels`: the channel count of the last frame that was
    /// coded, 0 before the first one.
    prev_channels: usize,
    /// C `st->silk_mode.toMono`: this packet is the last one with two
    /// channels before the stream goes on in mono. SILK fades its side
    /// signal out over it.
    silk_to_mono: bool,
    /// C `st->silk_bw_switch`: the last packet ended with a redundant CELT
    /// frame because SILK changes its internal rate on this one.
    silk_bw_switch: bool,
    /// C `st->hybrid_stereo_width_Q14`: the stereo width the last frame's
    /// CELT input was reduced to (16384 is full width).
    hybrid_stereo_width_q14: i32,
    /// C `st->silk_mode.stereoWidth_Q14`: the stereo width for this frame's
    /// CELT input.
    silk_stereo_width_q14: i32,
    /// C `st->width_mem`: the state of the stereo width estimate.
    width_mem: StereoWidthState,
    /// C `stereo_width` of `opus_encode_native`: the estimate for the
    /// current packet, 0 for mono input.
    stereo_width: f32,
    /// C `st->user_bitrate_bps`: the bitrate setting
    /// ([`set_bitrate`](Self::set_bitrate)).
    user_bitrate: Bitrate,
    /// C `st->bitrate_bps`: the bitrate the current packet is coded at. It
    /// is the rate of `user_bitrate`, or the rate the output buffer stands
    /// for when that is lower (libopus `user_bitrate_to_bitrate`). With a
    /// constant bitrate it is the rate of the whole bytes the packet takes.
    /// Each part of a packet made of several frames keeps the packet's
    /// value.
    packet_bitrate_bps: i32,
    /// C `st->silk_mode.complexity`: 0 to 10
    /// ([`set_complexity`](Self::set_complexity)).
    complexity: i32,
    /// C `!st->use_vbr`: every packet has the size the bitrate gives it
    /// ([`set_vbr`](Self::set_vbr)).
    use_cbr: bool,
    /// C `st->vbr_constraint`: with a variable bitrate, CELT-only packets
    /// stay close to the bitrate
    /// ([`set_vbr_constraint`](Self::set_vbr_constraint)).
    vbr_constraint: bool,
    /// C `st->silk_mode.useInBandFEC`
    /// ([`set_inband_fec`](Self::set_inband_fec)).
    use_inband_fec: bool,
    /// C `st->silk_mode.packetLossPercentage`: 0 to 100
    /// ([`set_packet_loss_perc`](Self::set_packet_loss_perc)).
    packet_loss_perc: i32,
    /// C `st->prev_framesize`: the duration in samples of the last frame
    /// that was coded, 0 before the first one. [`bitrate`](Self::bitrate)
    /// reads it.
    prev_frame_size: usize,
    /// C `silk_mode.reducedDependency` (`OPUS_SET_PREDICTION_DISABLED`):
    /// every frame is coded without prediction from the frames before it.
    prediction_disabled: bool,
    /// C `st->variable_duration` (`OPUS_SET_EXPERT_FRAME_DURATION`): the
    /// frame duration the caller set, or `Argument` when the `frame_size`
    /// of each encode call decides it.
    variable_duration: FrameDuration,
    /// The 2.5 ms that the CELT encoder codes, and throws away, after a
    /// reset for a change of mode (C `tmp_prefill`), interleaved.
    #[cfg(not(feature = "heap"))]
    celt_prefill: [f32; CELT_PREFILL],
    #[cfg(feature = "heap")]
    celt_prefill: Box<[f32; CELT_PREFILL]>,
    /// C `st->mode`: the mode decided for the last packet that was coded.
    /// Hybrid before the first one, as `opus_encoder_init` sets it. Only a
    /// packet too small to code anything reads it, for its first byte.
    mode: OpusMode,
    /// C `st->prev_mode`: the mode of the last frame that was coded, or
    /// CELT-only after a frame that announced the switch to it. `None`
    /// before the first frame.
    prev_enc_mode: Option<OpusMode>,

    variable_hp_smth2_q15: i32,
    hp_mem: FixedVec<i32, OPUS_HP_MEM>,

    #[cfg(not(feature = "heap"))]
    buf_filtered: FixedVec<i16, OPUS_MAX_FRAME>,
    #[cfg(feature = "heap")]
    buf_filtered: Box<FixedVec<i16, OPUS_MAX_FRAME>>,
    /// C `delay_buffer[MAX_ENCODER_BUFFER*2]`: delay-compensation ring feeding
    /// the CELT encoder (see `encoder_buffer` / `delay_compensation`).
    #[cfg(not(feature = "heap"))]
    delay_buffer: FixedVec<f32, OPUS_DELAY_BUF>,
    #[cfg(feature = "heap")]
    delay_buffer: Box<FixedVec<f32, OPUS_DELAY_BUF>>,
    /// Interleaved `pcm_buf` staging: `delay prefix + filtered frame` exactly
    /// like C `opus_encode_frame_native()` builds it before CELT encoding.
    #[cfg(not(feature = "heap"))]
    buf_celt_pcm: FixedVec<f32, OPUS_PCM_BUF>,
    #[cfg(feature = "heap")]
    buf_celt_pcm: Box<FixedVec<f32, OPUS_PCM_BUF>>,
    /// C `encoder_buffer` (= Fs/100): samples of ring history kept between
    /// frames (0 for restricted-latency applications).
    encoder_buffer: usize,
    /// C `delay_compensation` (= Fs/250): 4 ms lookahead prefix prepended to
    /// each CELT frame (0 for restricted-latency applications).
    delay_compensation: usize,
    /// Float filter state for `dc_reject_float` / `hp_cutoff_float`
    /// (C `st->hp_mem[4]`, opus_val32 in the float build).
    hp_mem_float: [f32; 4],
    /// C `prev_HB_gain`: the gain the previous frame applied to CELT's input.
    /// Hybrid attenuates the high band when it gets few bits.
    prev_hb_gain: f32,

    rc: RangeCoder,
}

/// libopus `gain_fade` (opus_encoder.c:581): scale an interleaved frame by a
/// gain that moves from `g1` to `g2` over CELT's overlap (2.5 ms), following
/// the squared CELT window, and stays at `g2` after it.
fn gain_fade(pcm: &mut [f32], g1: f32, g2: f32, frame_size: usize, channels: usize, fs: i32) {
    let window = modes::default_mode().window;
    let inc = celt::resampling_factor(fs).unwrap_or(1);
    let overlap = (window.len() / inc).min(frame_size);
    for i in 0..overlap {
        let w = window[i * inc] * window[i * inc];
        let g = w * g2 + (1.0 - w) * g1;
        for c in 0..channels {
            pcm[i * channels + c] *= g;
        }
    }
    for v in &mut pcm[overlap * channels..frame_size * channels] {
        *v *= g2;
    }
}

/// libopus `compute_equiv_rate` (opus_encoder.c:1027): the bitrate a 20 ms
/// VBR frame at complexity 10 would need for the same quality. `mode` is
/// `None` while the mode is not decided yet.
fn compute_equiv_rate(
    bitrate: i32,
    channels: usize,
    frame_rate: i32,
    vbr: bool,
    mode: Option<OpusMode>,
    complexity: i32,
    loss: i32,
) -> i32 {
    let mut equiv = bitrate;
    // Take into account overhead from smaller frames.
    if frame_rate > 50 {
        equiv -= (40 * channels as i32 + 20) * (frame_rate - 50);
    }
    // CBR is about a 8% penalty for both SILK and CELT.
    if !vbr {
        equiv -= equiv / 12;
    }
    // Complexity makes about 10% difference (from 0 to 10) in general.
    equiv = equiv * (90 + complexity) / 100;
    match mode {
        Some(OpusMode::SilkOnly) | Some(OpusMode::Hybrid) => {
            // SILK complexity 0-1 uses the non-delayed-decision NSQ, which
            // costs about 20%.
            if complexity < 2 {
                equiv = equiv * 4 / 5;
            }
            equiv -= equiv * loss / (6 * loss + 10);
        }
        Some(OpusMode::CeltOnly) => {
            // CELT complexity 0-4 doesn't have the pitch filter, which costs
            // about 10%.
            if complexity < 5 {
                equiv = equiv * 9 / 10;
            }
        }
        // Mode not known yet: half the SILK loss.
        None => equiv -= equiv * loss / (12 * loss + 20),
    }
    equiv
}

/// `mono_voice_bandwidth_thresholds` (opus_encoder.c:149): for each step
/// between two bandwidths, the equivalent bitrate at which speech moves to
/// the wider one, and the hysteresis around it. libopus 1.6 uses the same
/// numbers for stereo.
const VOICE_BANDWIDTH_THRESHOLDS: [i32; 8] = [
    9000, 700, // NB <-> MB
    9000, 700, // MB <-> WB
    13500, 1000, // WB <-> SWB
    14000, 2000, // SWB <-> FB
];
/// `mono_music_bandwidth_thresholds` (opus_encoder.c:155).
const MUSIC_BANDWIDTH_THRESHOLDS: [i32; 8] = [
    9000, 700, // NB <-> MB
    9000, 700, // MB <-> WB
    11000, 1000, // WB <-> SWB
    12000, 2000, // SWB <-> FB
];

/// `fec_thresholds` (opus_encoder.c:160): LBRR rate threshold and hysteresis
/// per bandwidth, starting at narrowband.
const FEC_THRESHOLDS: [i32; 10] = [
    12000, 1000, // NB
    14000, 1000, // MB
    16000, 1000, // WB
    20000, 1000, // SWB
    22000, 1000, // FB
];

fn bandwidth_fec_index(bw: Bandwidth) -> usize {
    match bw {
        Bandwidth::Narrowband => 0,
        Bandwidth::Mediumband => 1,
        Bandwidth::Wideband => 2,
        Bandwidth::Superwideband => 3,
        Bandwidth::Fullband | Bandwidth::Auto => 4,
    }
}

fn bandwidth_from_fec_index(idx: usize) -> Bandwidth {
    match idx {
        0 => Bandwidth::Narrowband,
        1 => Bandwidth::Mediumband,
        2 => Bandwidth::Wideband,
        3 => Bandwidth::Superwideband,
        _ => Bandwidth::Fullband,
    }
}

/// Port of opus_encoder.c `decide_fec()`: gate in-band FEC on the packet loss
/// setting and on whether the equivalent rate can afford LBRR at the current
/// bandwidth. With loss > 5%, the bandwidth is reduced until FEC fits; the
/// caller's per-frame bandwidth is updated accordingly.
fn decide_fec(
    use_in_band_fec: i32,
    packet_loss_perc: i32,
    last_fec: i32,
    mode: OpusMode,
    bandwidth: &mut Bandwidth,
    rate: i32,
) -> i32 {
    if use_in_band_fec == 0 || packet_loss_perc == 0 || mode == OpusMode::CeltOnly {
        return 0;
    }
    let orig_bandwidth = *bandwidth;
    loop {
        let idx = bandwidth_fec_index(*bandwidth);
        let mut lbrr_rate_thres_bps = FEC_THRESHOLDS[2 * idx];
        let hysteresis = FEC_THRESHOLDS[2 * idx + 1];
        if last_fec == 1 {
            lbrr_rate_thres_bps -= hysteresis;
        }
        if last_fec == 0 {
            lbrr_rate_thres_bps += hysteresis;
        }
        // silk_SMULWB(threshold * (125 - min(loss, 25)), 0.01 in Q16)
        lbrr_rate_thres_bps = silk_smulwb(
            lbrr_rate_thres_bps * (125 - packet_loss_perc.min(25)),
            655,
        );
        // If loss <= 5%, we look at whether we have enough rate to enable FEC.
        // If loss > 5%, we decrease the bandwidth until we can enable FEC.
        if rate > lbrr_rate_thres_bps {
            return 1;
        } else if packet_loss_perc <= 5 {
            return 0;
        } else if bandwidth_fec_index(*bandwidth) > 0 {
            let lower = bandwidth_fec_index(*bandwidth) - 1;
            *bandwidth = bandwidth_from_fec_index(lower);
        } else {
            break;
        }
    }
    // Couldn't find any bandwidth to enable FEC, keep original bandwidth.
    *bandwidth = orig_bandwidth;
    0
}

/// C `StereoWidthState` (opus_encoder.c): the running estimates behind
/// [`compute_stereo_width`].
#[derive(Clone, Copy, Default)]
struct StereoWidthState {
    /// Smoothed power of the left and of the right channel, and their
    /// smoothed product.
    xx: f32,
    xy: f32,
    yy: f32,
    smoothed_width: f32,
    max_follower: f32,
}

/// libopus `compute_stereo_width` (opus_encoder.c:854, float build): how
/// far apart the two channels of stereo input are, from 0 (the same signal
/// in both) to 1.
///
/// The width of a frame is small when the channels are correlated or
/// equally loud. It is smoothed over one second, and the result follows its
/// peaks, which decay slowly.
fn compute_stereo_width(
    pcm: &[f32],
    frame_size: usize,
    fs: i32,
    mem: &mut StereoWidthState,
) -> f32 {
    let frame_rate = fs / frame_size as i32;
    let short_alpha = 25.0 / frame_rate.max(50) as f32;
    let (mut xx, mut xy, mut yy) = (0.0f32, 0.0f32, 0.0f32);
    // Four samples at a time, as in C. The frame size is a multiple of 4
    // except for 2.5 ms at 12 kHz, where the last two samples are left out.
    let mut i = 0;
    while i + 3 < frame_size {
        let (mut pxx, mut pxy, mut pyy) = (0.0f32, 0.0f32, 0.0f32);
        for k in 0..4 {
            let (x, y) = (pcm[2 * (i + k)], pcm[2 * (i + k) + 1]);
            pxx += x * x;
            pxy += x * y;
            pyy += y * y;
        }
        xx += pxx;
        xy += pxy;
        yy += pyy;
        i += 4;
    }
    // C shape kept (`!(xx < 1e9f) || celt_isnan(xx)`): it also catches NaN.
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    if !(xx < 1e9) || xx.is_nan() || !(yy < 1e9) || yy.is_nan() {
        xx = 0.0;
        xy = 0.0;
        yy = 0.0;
    }
    mem.xx += short_alpha * (xx - mem.xx);
    // "Rewritten to avoid overflows on abrupt sign change."
    mem.xy = (1.0 - short_alpha) * mem.xy + short_alpha * xy;
    mem.yy += short_alpha * (yy - mem.yy);
    mem.xx = mem.xx.max(0.0);
    mem.xy = mem.xy.max(0.0);
    mem.yy = mem.yy.max(0.0);
    if mem.xx.max(mem.yy) > 8e-4 {
        let sqrt_xx = mem.xx.sqrt();
        let sqrt_yy = mem.yy.sqrt();
        let qrrt_xx = sqrt_xx.sqrt();
        let qrrt_yy = sqrt_yy.sqrt();
        // Inter-channel correlation.
        mem.xy = mem.xy.min(sqrt_xx * sqrt_yy);
        let corr = mem.xy / (1e-15 + sqrt_xx * sqrt_yy);
        // Approximate loudness difference.
        let ldiff = (qrrt_xx - qrrt_yy).abs() / (1e-15 + qrrt_xx + qrrt_yy);
        let width = (1.0 - corr * corr).sqrt().min(1.0) * ldiff;
        // Smoothing over one second.
        mem.smoothed_width += (width - mem.smoothed_width) / frame_rate as f32;
        // Peak follower.
        mem.max_follower = (mem.max_follower - 0.02 / frame_rate as f32).max(mem.smoothed_width);
    }
    (20.0 * mem.max_follower).min(1.0)
}

/// The equivalent bitrate from which the automatic choice takes CELT-only
/// instead of SILK (opus_encoder.c:1492-1511).
///
/// `stereo_width` is 0 for mono input and what [`compute_stereo_width`]
/// gives for stereo input. Wide stereo moves the threshold for speech from
/// 64 to 44 kb/s (C `mode_thresholds`).
fn compute_mode_threshold(
    application: Application,
    stereo_width: f32,
    prev_was_celt: bool,
    has_prev_mode: bool,
    voice_est: i32,
) -> i32 {
    // "Interpolate based on stereo width", with the float arithmetic of C.
    let mode_voice = ((1.0 - stereo_width) * 64000.0 + stereo_width * 44000.0) as i32;
    let mode_music = ((1.0 - stereo_width) * 10000.0 + stereo_width * 10000.0) as i32;

    let diff = mode_voice - mode_music;
    let offset = (voice_est * voice_est * diff) >> 14;
    let mut threshold = mode_music + offset;

    if application == Application::Voip {
        threshold += 8000;
    }

    if has_prev_mode {
        if prev_was_celt {
            threshold -= 4000;
        } else {
            threshold += 4000;
        }
    }

    if application == Application::RestrictedLowDelay {
        threshold = 0;
    }

    threshold
}

/// libopus `OPUS_AUTO`: the encoder picks the value itself. As a bitrate,
/// the encoder picks it from the sampling rate and the channel count.
pub(crate) const OPUS_AUTO: i32 = -1000;

/// libopus `OPUS_BITRATE_MAX`: no bitrate limit. The frame fills the bytes
/// it is given.
pub(crate) const OPUS_BITRATE_MAX: i32 = -1;

/// Six times the number of frames per second, the divisor of
/// `bitrate_to_bits` and `bits_to_bitrate`. With six times the rate, 60 ms
/// frames (16.67 per second) round as in libopus. A frame size above the
/// `i32` range counts as `i32::MAX`.
///
/// The frame must be shorter than 6 seconds and `frame_size` must not be 0:
/// otherwise the conversions divide by 0.
#[inline]
fn frame_rate_x6(fs: i32, frame_size: usize) -> i64 {
    6 * fs as i64 / frame_size.min(i32::MAX as usize) as i64
}

/// libopus `bitrate_to_bits`: bits per second to bits per frame. The
/// arithmetic is 64 bits wide; a result outside the `i32` range is clamped.
#[inline]
pub(crate) fn bitrate_to_bits(bitrate: i32, fs: i32, frame_size: usize) -> i32 {
    let bits = bitrate as i64 * 6 / frame_rate_x6(fs, frame_size);
    bits.clamp(i32::MIN as i64, i32::MAX as i64) as i32
}

/// libopus `bits_to_bitrate`: bits per frame to bits per second. The
/// arithmetic is 64 bits wide; a result outside the `i32` range is clamped.
#[inline]
pub(crate) fn bits_to_bitrate(bits: i32, fs: i32, frame_size: usize) -> i32 {
    let bitrate = (bits as i64).saturating_mul(frame_rate_x6(fs, frame_size)) / 6;
    bitrate.clamp(i32::MIN as i64, i32::MAX as i64) as i32
}

/// SILK's share of a Hybrid packet's `rate_bps`: libopus
/// `compute_silk_rate_for_hybrid` (opus_encoder.c:973). The rate is
/// allocated per channel, since stereo codes the side with SILK too.
fn compute_silk_rate_for_hybrid(
    rate_bps: i32,
    bandwidth: Bandwidth,
    frame20ms: bool,
    vbr: bool,
    fec: bool,
    channels: usize,
) -> i32 {
    //  total |  SILK, no FEC  |   SILK, FEC    |
    //        | 10 ms   20 ms  | 10 ms   20 ms  |
    const RATE_TABLE: [[i32; 5]; 7] = [
        [0, 0, 0, 0, 0],
        [12000, 10000, 10000, 11000, 11000],
        [16000, 13500, 13500, 15000, 15000],
        [20000, 16000, 16000, 18000, 18000],
        [24000, 18000, 18000, 21000, 21000],
        [32000, 22000, 22000, 28000, 28000],
        [64000, 38000, 38000, 50000, 50000],
    ];
    let channels = channels as i32;
    // Do the allocation per channel.
    let rate = rate_bps / channels;
    let entry = 1 + frame20ms as usize + 2 * fec as usize;
    let n = RATE_TABLE.len();
    let mut i = 1;
    while i < n && RATE_TABLE[i][0] <= rate {
        i += 1;
    }
    let mut silk_rate = if i == n {
        // Above the table, SILK gets half of the extra bits.
        RATE_TABLE[n - 1][entry] + (rate - RATE_TABLE[n - 1][0]) / 2
    } else {
        let (lo, hi) = (RATE_TABLE[i - 1][entry], RATE_TABLE[i][entry]);
        let (x0, x1) = (RATE_TABLE[i - 1][0], RATE_TABLE[i][0]);
        (lo * (x1 - rate) + hi * (rate - x0)) / (x1 - x0)
    };
    if !vbr {
        // Tiny boost to SILK for CBR.
        silk_rate += 100;
    }
    if bandwidth == Bandwidth::Superwideband {
        silk_rate += 300;
    }
    silk_rate *= channels;
    // Small adjustment for stereo (libopus: "calibrated for 32 kb/s").
    if channels == 2 && rate >= 12000 {
        silk_rate -= 1000;
    }
    silk_rate
}

#[cfg(all(test, feature = "std"))]
mod silk_rate_tests {
    use super::{Bandwidth, compute_silk_rate_for_hybrid};

    /// Fullband, 20 ms, VBR, no FEC: the plain table.
    fn plain(rate: i32, channels: usize) -> i32 {
        compute_silk_rate_for_hybrid(rate, Bandwidth::Fullband, true, true, false, channels)
    }

    #[test]
    fn test_reference_table_exact_entries() {
        assert_eq!(plain(12000, 1), 10000);
        assert_eq!(plain(16000, 1), 13500);
        assert_eq!(plain(20000, 1), 16000);
        assert_eq!(plain(24000, 1), 18000);
        assert_eq!(plain(32000, 1), 22000);
        assert_eq!(plain(64000, 1), 38000);
    }

    #[test]
    fn test_interpolation_between_table_entries() {
        assert_eq!(plain(18000, 1), 14750);
    }

    #[test]
    fn test_above_table_max_gives_half_extra() {
        assert_eq!(plain(72000, 1), 38000 + (72000 - 64000) / 2);
    }

    /// Stereo looks the rate up per channel, doubles it and, from 12 kb/s per
    /// channel, takes 1 kb/s off (libopus opus_encoder.c).
    #[test]
    fn test_stereo_allocates_per_channel() {
        assert_eq!(plain(32000, 2), 2 * 13500 - 1000);
        assert_eq!(plain(24000, 2), 2 * 10000 - 1000);
        assert_eq!(plain(20000, 2), 2 * 8333);
        assert_eq!(plain(64000, 2), 2 * 22000 - 1000);
    }

    /// Super-wideband adds 300 b/s, CBR 100 b/s, and FEC uses its own column.
    #[test]
    fn test_boosts_and_fec_column() {
        let swb = Bandwidth::Superwideband;
        let fb = Bandwidth::Fullband;
        assert_eq!(compute_silk_rate_for_hybrid(24000, swb, true, true, false, 1), 18300);
        assert_eq!(compute_silk_rate_for_hybrid(24000, fb, true, false, false, 1), 18100);
        assert_eq!(compute_silk_rate_for_hybrid(24000, swb, true, false, false, 1), 18400);
        assert_eq!(compute_silk_rate_for_hybrid(24000, fb, true, true, true, 1), 21000);
        assert_eq!(compute_silk_rate_for_hybrid(32000, swb, true, true, true, 1), 28300);
        // Stereo: the boosts are per channel, before the 1 kb/s adjustment.
        assert_eq!(
            compute_silk_rate_for_hybrid(32000, swb, true, true, false, 2),
            2 * (13500 + 300) - 1000
        );
    }
}

/// Mutably borrow a codec-state field that is a `Box<T>` under the `heap`
/// feature and a plain `T` otherwise, so call sites don't need `#[cfg]` on
/// every `&mut self.field`.
///
/// Under `heap` the field is a `Box<...>`, so `T: DerefMut` resolves to the
/// box itself and the `Target` is the inner type. Without `heap` the field is
/// the inner type directly; `?Sized` lets `&mut [i16]`-style coercion choose
/// the slice target.
#[cfg(feature = "heap")]
#[inline(always)]
fn state_mut<T: core::ops::DerefMut>(b: &mut T) -> &mut T::Target {
    b
}
#[cfg(not(feature = "heap"))]
#[inline(always)]
fn state_mut<T: ?Sized>(b: &mut T) -> &mut T {
    b
}

impl OpusEncoder {
    pub fn new(
        sampling_rate: i32,
        channels: usize,
        application: Application,
    ) -> Result<Self, &'static str> {
        if ![8000, 12000, 16000, 24000, 48000].contains(&sampling_rate) {
            return Err("Invalid sampling rate");
        }
        if ![1, 2].contains(&channels) {
            return Err("Invalid number of channels");
        }

        let mode = modes::default_mode();
        #[cfg(feature = "heap")]
        let celt_enc = Box::new(CeltEncoder::with_sampling_rate(mode, channels, sampling_rate));
        #[cfg(not(feature = "heap"))]
        let celt_enc = CeltEncoder::with_sampling_rate(mode, channels, sampling_rate);

        // C `opus_encoder_init`: `st->mode = MODE_HYBRID` and
        // `st->bandwidth = OPUS_BANDWIDTH_FULLBAND`, whatever the application
        // and the sampling rate. Both are decided again for the first packet
        // that is coded.
        let (opus_mode, bw) = (OpusMode::Hybrid, Bandwidth::Fullband);

        use silk::lin2log::silk_lin2log;
        let variable_hp_smth2_q15 = silk_lin2log(60) << 8;

        Ok(Self {
            celt_enc,
            silk: SilkEncoder::new(sampling_rate, channels),
            application,
            sampling_rate,
            channels,
            bandwidth: bw,
            auto_bandwidth: bw,
            user_bandwidth: None,
            max_bandwidth: Bandwidth::Fullband,
            user_forced_mode: None,
            lfe: false,
            energy_mask: None,
            range_final: 0,
            first: true,
            silk_allow_bandwidth_switch: false,
            silk_in_wb_mode_without_variable_lp: false,
            silk_opus_can_switch: false,
            silk_lbrr_coded: 0,
            nonfinal_frame: false,
            stream_channels: channels,
            force_channels: None,
            prev_channels: 0,
            silk_to_mono: false,
            silk_bw_switch: false,
            // C `opus_encoder_init`: `hybrid_stereo_width_Q14 = 1 << 14`.
            hybrid_stereo_width_q14: 1 << 14,
            silk_stereo_width_q14: 1 << 14,
            width_mem: StereoWidthState::default(),
            stereo_width: 0.0,
            // C `opus_encoder_init`: `user_bitrate_bps = OPUS_AUTO` and
            // `bitrate_bps = 3000 + Fs * channels`. Each packet sets the
            // second again before it is read.
            user_bitrate: Bitrate::Auto,
            packet_bitrate_bps: 3000 + sampling_rate * channels as i32,
            complexity: 9,
            use_cbr: false,
            vbr_constraint: true,
            use_inband_fec: false,
            packet_loss_perc: 0,
            prev_frame_size: 0,
            prediction_disabled: false,
            variable_duration: FrameDuration::Argument,
            #[cfg(not(feature = "heap"))]
            celt_prefill: [0.0; CELT_PREFILL],
            #[cfg(feature = "heap")]
            celt_prefill: Box::new([0.0; CELT_PREFILL]),
            prev_enc_mode: None,
            mode: opus_mode,
            variable_hp_smth2_q15,
            hp_mem: FixedVec::from_value(0, channels * 2),

            #[cfg(not(feature = "heap"))]
            buf_filtered: FixedVec::new(),
            #[cfg(feature = "heap")]
            buf_filtered: Box::new(FixedVec::new()),
            #[cfg(not(feature = "heap"))]
            delay_buffer: FixedVec::from_value(0.0, OPUS_DELAY_BUF),
            #[cfg(feature = "heap")]
            delay_buffer: Box::new(FixedVec::from_value(0.0, OPUS_DELAY_BUF)),
            #[cfg(not(feature = "heap"))]
            buf_celt_pcm: FixedVec::new(),
            #[cfg(feature = "heap")]
            buf_celt_pcm: Box::new(FixedVec::new()),
            encoder_buffer: if matches!(application, Application::RestrictedLowDelay) {
                0
            } else {
                (sampling_rate / 100) as usize
            },
            delay_compensation: if matches!(application, Application::RestrictedLowDelay) {
                0
            } else {
                (sampling_rate / 250) as usize
            },
            hp_mem_float: [0.0; 4],
            prev_hb_gain: 1.0,
            rc: RangeCoder::new_encoder(1),
        })
    }

    /// The decisions libopus's `opus_encode_native` makes for a packet
    /// (opus_encoder.c:1408-1695): the mode, the audio bandwidth, whether
    /// the packet carries in-band FEC, and whether a redundant CELT frame
    /// goes with a change of mode. `n_bytes` is the largest packet allowed.
    ///
    /// Not ported: the speech/music and bandwidth detection of libopus's
    /// tonality analysis (it runs from complexity 7).
    fn decide_mode_and_bandwidth(
        &mut self,
        frame_size: usize,
        frame_rate: i32,
        n_bytes: usize,
    ) -> FrameDecisions {
        // C 1325-1334, `st->bitrate_bps`: the bitrate asked for, or what the
        // output buffer can hold when that is less. CBR codes whole bytes,
        // so its bitrate is the packet's.
        let bitrate = self.packet_bitrate_bps;
        // C 1407: the rate the largest packet allowed stands for.
        let max_rate = bits_to_bitrate(8 * n_bytes as i32, self.sampling_rate, frame_size);
        // C 1413-1426: how likely the input is speech, in Q7. Without the
        // tonality analysis this depends on the application alone.
        let voice_est = if self.application == Application::Voip { 115 } else { 48 };

        // C 1428-1456: the channels the packet codes, then the rate the
        // decisions go by for that many channels.
        self.decide_stream_channels(bitrate, frame_rate, voice_est);

        // --- Mode: SILK or CELT (C 1466-1539) ---
        let mut mode = if self.application == Application::RestrictedLowDelay {
            OpusMode::CeltOnly
        } else if let Some(forced) = self.user_forced_mode {
            // C 1528-1530: the mode the caller asked for. The rules below
            // can still change it: frames under 10 ms are CELT-only, SILK
            // above wideband is Hybrid, Hybrid at wideband or below is SILK.
            forced
        } else {
            let equiv = compute_equiv_rate(
                bitrate,
                self.stream_channels,
                frame_rate,
                !self.use_cbr,
                None,
                self.complexity,
                self.packet_loss_perc,
            );
            let threshold = compute_mode_threshold(
                self.application,
                self.stereo_width,
                self.prev_enc_mode == Some(OpusMode::CeltOnly),
                self.prev_enc_mode.is_some(),
                voice_est,
            );
            // C 1513, at every input rate: 8, 12 and 16 kHz input is coded
            // by CELT too when the bitrate is high for speech.
            let mut mode = if equiv >= threshold {
                OpusMode::CeltOnly
            } else {
                OpusMode::SilkOnly
            };
            // C 1515-1518: with FEC on and enough packet loss, use SILK: only
            // SILK carries the FEC.
            if self.use_inband_fec && self.packet_loss_perc > (128 - voice_est) >> 4 {
                mode = OpusMode::SilkOnly;
            }
            // C 1525-1527: a budget representing less than 6 kb/s (9 kb/s for
            // frames shorter than 20 ms) cannot carry SILK. CELT-only codes
            // it as a small valid packet instead.
            let tiny_budget_rate = if frame_rate > 50 { 9000 } else { 6000 };
            if (n_bytes as i32) < bitrate_to_bits(tiny_budget_rate, self.sampling_rate, frame_size) / 8 {
                mode = OpusMode::CeltOnly;
            }
            mode
        };
        // C 1532-1537: SILK frames are 10 ms or longer; 2.5 and 5 ms frames
        // are CELT-only.
        if mode != OpusMode::CeltOnly && frame_size < self.sampling_rate as usize / 100 {
            mode = OpusMode::CeltOnly;
        }
        // C 1538-1539: a low-frequency-effects channel is coded by CELT.
        if self.lfe {
            mode = OpusMode::CeltOnly;
        }

        // C 1541-1558: a switch between CELT-only and SILK-only or Hybrid
        // is bridged by a redundant 5 ms CELT frame.
        //
        // - Leaving CELT-only: the frame goes in front of the first packet
        //   of the new mode.
        // - Entering CELT-only: this packet stays in the old mode and gets
        //   the frame at its end; the next packet is CELT-only. A packet
        //   shorter than 10 ms cannot stay, so it switches at once, without
        //   the frame.
        let (mut redundancy, mut celt_to_silk, mut to_celt) = (false, false, false);
        if let Some(prev) = self.prev_enc_mode {
            let (is_celt, was_celt) = (mode == OpusMode::CeltOnly, prev == OpusMode::CeltOnly);
            if is_celt != was_celt {
                redundancy = true;
                celt_to_silk = !is_celt;
                if !celt_to_silk {
                    if frame_size >= self.sampling_rate as usize / 100 {
                        mode = prev;
                        to_celt = true;
                    } else {
                        redundancy = false;
                    }
                }
            }
        }
        // C 1560-1571: "Delay stereo->mono transition by two frames so that
        // SILK can do a smooth downmix". The last packet before mono still
        // codes two channels, with SILK's side signal faded out. Not around
        // CELT-only packets, which change the channel count at once.
        if self.stream_channels == 1
            && self.prev_channels == 2
            && !self.silk_to_mono
            && mode != OpusMode::CeltOnly
            && self.prev_enc_mode != Some(OpusMode::CeltOnly)
        {
            self.silk_to_mono = true;
            self.stream_channels = 2;
        } else {
            self.silk_to_mono = false;
        }

        // C 1576-1581: SILK starts afresh after CELT-only packets.
        let prefill =
            mode != OpusMode::CeltOnly && self.prev_enc_mode == Some(OpusMode::CeltOnly);

        // C 1573: the rate the next decisions go by counts the mode.
        let equiv_rate = compute_equiv_rate(
            bitrate,
            self.stream_channels,
            frame_rate,
            !self.use_cbr,
            Some(mode),
            self.complexity,
            self.packet_loss_perc,
        );

        // --- Bandwidth from the rate (C 1583-1627) ---
        // SILK and Hybrid only reconsider it when SILK allows a switch, which
        // is in a pause or after 5 s at the latest.
        if mode == OpusMode::CeltOnly || self.first || self.silk_allow_bandwidth_switch {
            // Indices: 0 narrowband .. 4 fullband.
            let mut bandwidth = 4usize;
            loop {
                // Thresholds between this bandwidth and the next lower one,
                // interpolated between music and voice by `voice_est`.
                let entry = 2 * (bandwidth - 1);
                let interpolate = |i: usize| {
                    MUSIC_BANDWIDTH_THRESHOLDS[i]
                        + ((voice_est
                            * voice_est
                            * (VOICE_BANDWIDTH_THRESHOLDS[i] - MUSIC_BANDWIDTH_THRESHOLDS[i]))
                            >> 14)
                };
                let (mut threshold, hysteresis) = (interpolate(entry), interpolate(entry + 1));
                if !self.first {
                    if bandwidth_fec_index(self.auto_bandwidth) >= bandwidth {
                        threshold -= hysteresis;
                    } else {
                        threshold += hysteresis;
                    }
                }
                if equiv_rate >= threshold {
                    break;
                }
                bandwidth -= 1;
                if bandwidth == 0 {
                    break;
                }
            }
            // Mediumband is only used on request or during transitions.
            if bandwidth == 1 {
                bandwidth = 2;
            }
            self.auto_bandwidth = bandwidth_from_fec_index(bandwidth);
            self.bandwidth = self.auto_bandwidth;
            // No switch above wideband until SILK itself runs at wideband.
            if !self.first
                && mode != OpusMode::CeltOnly
                && !self.silk_in_wb_mode_without_variable_lp
                && bandwidth > 2
            {
                self.bandwidth = Bandwidth::Wideband;
            }
        }
        // C 1629-1630: nothing above the maximum the caller set.
        if bandwidth_fec_index(self.bandwidth) > bandwidth_fec_index(self.max_bandwidth) {
            self.bandwidth = self.max_bandwidth;
        }
        // C 1632-1633: a bandwidth the caller asked for wins.
        if let Some(user) = self.user_bandwidth {
            self.bandwidth = user;
        }
        // C 1636-1639: no Hybrid when the packet cannot exceed 15 kb/s.
        if mode != OpusMode::CeltOnly
            && max_rate < 15000
            && bandwidth_fec_index(self.bandwidth) > 2
        {
            self.bandwidth = Bandwidth::Wideband;
        }
        // C 1643-1650: nothing above the input's Nyquist frequency.
        let nyquist_bandwidth = match self.sampling_rate {
            8000 => Bandwidth::Narrowband,
            12000 => Bandwidth::Mediumband,
            16000 => Bandwidth::Wideband,
            24000 => Bandwidth::Superwideband,
            _ => Bandwidth::Fullband,
        };
        if bandwidth_fec_index(self.bandwidth) > bandwidth_fec_index(nyquist_bandwidth) {
            self.bandwidth = nyquist_bandwidth;
        }

        // --- In-band FEC (C 1675) ---
        // decide_fec can lower the bandwidth when a high loss rate asks for
        // FEC at any cost.
        let lbrr_coded = decide_fec(
            if self.use_inband_fec { 1 } else { 0 },
            self.packet_loss_perc,
            self.silk_lbrr_coded,
            mode,
            &mut self.bandwidth,
            equiv_rate,
        );
        self.silk_lbrr_coded = lbrr_coded;

        // C 1680-1695, after decide_fec: CELT has no mediumband; SILK above
        // wideband is Hybrid; Hybrid at wideband or below is SILK-only.
        // Running this before decide_fec left Hybrid with a wideband frame
        // when FEC lowered the bandwidth: CELT then had no band to code and
        // divided by zero.
        if mode == OpusMode::CeltOnly && self.bandwidth == Bandwidth::Mediumband {
            self.bandwidth = Bandwidth::Wideband;
        }
        // C 1683-1684: a low-frequency-effects channel is narrowband,
        // whatever was asked for.
        if self.lfe {
            self.bandwidth = Bandwidth::Narrowband;
        }
        let above_wideband = bandwidth_fec_index(self.bandwidth) > 2;
        if mode == OpusMode::SilkOnly && above_wideband {
            mode = OpusMode::Hybrid;
        }
        if mode == OpusMode::Hybrid && !above_wideband {
            mode = OpusMode::SilkOnly;
        }
        // C `st->mode`.
        self.mode = mode;
        FrameDecisions {
            mode,
            bandwidth: self.bandwidth,
            lbrr_coded,
            redundancy,
            celt_to_silk,
            to_celt,
            prefill,
            equiv_rate,
        }
    }

    /// libopus's choice of the channels a packet codes
    /// (opus_encoder.c:1408-1451): `stream_channels`.
    ///
    /// - A forced count wins for stereo input.
    /// - Otherwise stereo input is coded in stereo when the equivalent rate
    ///   for two channels is above a threshold between 17 kb/s (music) and
    ///   19 kb/s (speech), with 1 kb/s of hysteresis.
    /// - Mono input is coded in mono.
    fn decide_stream_channels(&mut self, bitrate: i32, frame_rate: i32, voice_est: i32) {
        // C `stereo_voice_threshold` and `stereo_music_threshold`.
        const STEREO_VOICE_THRESHOLD: i32 = 19000;
        const STEREO_MUSIC_THRESHOLD: i32 = 17000;
        if let (Some(forced), 2) = (self.force_channels, self.channels) {
            self.stream_channels = forced;
        } else if self.channels == 2 {
            // C 1409: the rate for the input's channels, before the mode is
            // known.
            let equiv_rate = compute_equiv_rate(
                bitrate,
                self.channels,
                frame_rate,
                !self.use_cbr,
                None,
                self.complexity,
                self.packet_loss_perc,
            );
            let mut threshold = STEREO_MUSIC_THRESHOLD
                + ((voice_est * voice_est * (STEREO_VOICE_THRESHOLD - STEREO_MUSIC_THRESHOLD))
                    >> 14);
            if self.stream_channels == 2 {
                threshold -= 1000;
            } else {
                threshold += 1000;
            }
            self.stream_channels = if equiv_rate > threshold { 2 } else { 1 };
        } else {
            self.stream_channels = self.channels;
        }
    }

    /// libopus `OPUS_SET_BITRATE`: the bitrate of the packets, for all
    /// channels together. The default is [`Bitrate::Auto`], as in libopus:
    /// `60 * rate / frame_size + rate * channels` b/s, for example
    /// 19,000 b/s for 16 kHz mono in 20 ms frames.
    ///
    /// As in libopus, an explicit bitrate must be positive. It is raised to
    /// at least 500 bits per second and limited to at most 750,000 bits per
    /// second per channel. The length of the output buffer of each
    /// [`encode`](Self::encode) call can lower it further.
    pub fn set_bitrate(&mut self, bitrate: Bitrate) -> Result<(), &'static str> {
        self.user_bitrate = match bitrate {
            Bitrate::BitsPerSecond(bps) if bps <= 0 => return Err("Invalid bitrate"),
            Bitrate::BitsPerSecond(bps) => {
                Bitrate::BitsPerSecond(bps.clamp(500, 750_000 * self.channels as i32))
            }
            other => other,
        };
        Ok(())
    }

    /// The bitrate that was set, after its limits were applied.
    pub fn bitrate_setting(&self) -> Bitrate {
        self.user_bitrate
    }

    /// libopus `OPUS_GET_BITRATE`: the bitrate in bits per second that the
    /// setting stands for. [`Bitrate::Auto`] and [`Bitrate::Max`] depend
    /// on the duration of the last frame that was coded (2.5 ms before the
    /// first one), and every setting counts with at most what a 1276-byte
    /// packet per frame carries.
    pub fn bitrate(&self) -> i32 {
        self.user_bitrate_to_bitrate(self.prev_frame_size, OPUS_MAX_PACKET_BYTES)
    }

    /// libopus `user_bitrate_to_bitrate` (opus_encoder.c:733): the bitrate
    /// of the setting for frames of `frame_size` samples (0 counts as
    /// 2.5 ms), at most what `max_data_bytes` bytes per frame carry.
    fn user_bitrate_to_bitrate(&self, frame_size: usize, max_data_bytes: usize) -> i32 {
        let fs = self.sampling_rate;
        let frame_size = if frame_size == 0 { fs as usize / 400 } else { frame_size };
        let max_bitrate = bits_to_bitrate(8 * max_data_bytes as i32, fs, frame_size);
        let user_bitrate = match self.user_bitrate {
            Bitrate::Auto => 60 * fs / frame_size as i32 + fs * self.channels as i32,
            Bitrate::Max => 1_500_000,
            Bitrate::BitsPerSecond(bps) => bps,
        };
        user_bitrate.min(max_bitrate)
    }

    /// libopus `OPUS_SET_COMPLEXITY`: 0 to 10. Higher values spend more
    /// processor time for better quality. The default is 9.
    pub fn set_complexity(&mut self, complexity: i32) -> Result<(), &'static str> {
        if !(0..=10).contains(&complexity) {
            return Err("Invalid complexity");
        }
        self.complexity = complexity;
        Ok(())
    }

    /// The complexity set with [`set_complexity`](Self::set_complexity).
    pub fn complexity(&self) -> i32 {
        self.complexity
    }

    /// libopus `OPUS_SET_VBR`: a variable bitrate (`true`, the default) or a
    /// constant one (`false`). With a constant bitrate every packet of the
    /// same duration has the same size.
    pub fn set_vbr(&mut self, vbr: bool) {
        self.use_cbr = !vbr;
    }

    /// Whether the bitrate is variable ([`set_vbr`](Self::set_vbr)).
    pub fn vbr(&self) -> bool {
        !self.use_cbr
    }

    /// libopus `OPUS_SET_VBR_CONSTRAINT`: with a variable bitrate, keep
    /// CELT-only packets close to the bitrate (`true`, the default). Without
    /// the constraint a CELT-only packet can take more bytes for a hard
    /// frame and fewer for an easy one. It has no effect with a constant
    /// bitrate. Hybrid packets are never constrained, as in libopus.
    pub fn set_vbr_constraint(&mut self, constrained: bool) {
        self.vbr_constraint = constrained;
    }

    /// Whether the variable bitrate is constrained
    /// ([`set_vbr_constraint`](Self::set_vbr_constraint)).
    pub fn vbr_constraint(&self) -> bool {
        self.vbr_constraint
    }

    /// libopus `OPUS_SET_INBAND_FEC`: let SILK and Hybrid packets carry a
    /// low-bitrate copy of the frame before them (in-band forward error
    /// correction, FEC), so that a decoder can rebuild a lost packet. The
    /// copy is coded only when [`set_packet_loss_perc`](Self::set_packet_loss_perc)
    /// is above 0 and the bitrate leaves room for it. Off by default.
    pub fn set_inband_fec(&mut self, enabled: bool) {
        self.use_inband_fec = enabled;
    }

    /// Whether in-band FEC is on ([`set_inband_fec`](Self::set_inband_fec)).
    pub fn inband_fec(&self) -> bool {
        self.use_inband_fec
    }

    /// libopus `OPUS_SET_PACKET_LOSS_PERC`: the packet loss the network is
    /// expected to have, 0 to 100 percent. The default is 0. Higher values
    /// make the packets more robust against loss, at the cost of quality
    /// without loss.
    pub fn set_packet_loss_perc(&mut self, percent: i32) -> Result<(), &'static str> {
        if !(0..=100).contains(&percent) {
            return Err("Invalid packet loss percentage");
        }
        self.packet_loss_perc = percent;
        Ok(())
    }

    /// The expected packet loss in percent
    /// ([`set_packet_loss_perc`](Self::set_packet_loss_perc)).
    pub fn packet_loss_perc(&self) -> i32 {
        self.packet_loss_perc
    }

    /// libopus `OPUS_SET_FORCE_MODE`: ask for a coding mode instead of the
    /// automatic choice. `None` gives the choice back to the encoder.
    ///
    /// The mode is a request, as in libopus. The rules that keep a packet
    /// valid still apply, in this order:
    ///
    /// - A `RestrictedLowDelay` encoder codes CELT-only whatever is asked.
    /// - Frames shorter than 10 ms are CELT-only.
    /// - A low-frequency-effects encoder ([`set_lfe`](Self::set_lfe)) codes
    ///   CELT-only.
    /// - SILK-only above wideband becomes Hybrid, and Hybrid at wideband or
    ///   below becomes SILK-only. To get SILK-only from 24 or 48 kHz input,
    ///   also limit the bandwidth with
    ///   [`set_bandwidth`](Self::set_bandwidth) or
    ///   [`set_max_bandwidth`](Self::set_max_bandwidth).
    pub fn set_force_mode(&mut self, mode: Option<ForcedMode>) {
        self.user_forced_mode = mode.map(|mode| match mode {
            ForcedMode::SilkOnly => OpusMode::SilkOnly,
            ForcedMode::Hybrid => OpusMode::Hybrid,
            ForcedMode::CeltOnly => OpusMode::CeltOnly,
        });
    }

    /// libopus `OPUS_SET_BANDWIDTH`: code this audio bandwidth instead of
    /// the one the bitrate calls for. `Bandwidth::Auto` gives the choice
    /// back to the encoder.
    ///
    /// The bandwidth never exceeds what the input sampling rate can carry.
    /// CELT has no mediumband and codes wideband instead. SILK codes at the
    /// internal rate of the bandwidth (8, 12 or 16 kHz); a running SILK
    /// encoder moves there one step at a time, when the signal lets it.
    pub fn set_bandwidth(&mut self, bandwidth: Bandwidth) {
        self.user_bandwidth = match bandwidth {
            Bandwidth::Auto => None,
            fixed => Some(fixed),
        };
    }

    /// libopus `OPUS_SET_MAX_BANDWIDTH`: the widest audio bandwidth the
    /// automatic choice may take. A bandwidth set with
    /// [`set_bandwidth`](Self::set_bandwidth) is not limited by it.
    /// `Bandwidth::Auto` is not a bandwidth and is rejected.
    pub fn set_max_bandwidth(&mut self, bandwidth: Bandwidth) -> Result<(), &'static str> {
        if bandwidth == Bandwidth::Auto {
            return Err("Invalid maximum bandwidth");
        }
        self.max_bandwidth = bandwidth;
        Ok(())
    }

    /// libopus `OPUS_SET_LFE`: this encoder codes the low-frequency-effects
    /// channel of a surround stream. Its packets are CELT-only and
    /// narrowband, whatever mode and bandwidth are set.
    pub fn set_lfe(&mut self, lfe: bool) {
        self.lfe = lfe;
    }

    /// Whether [`set_lfe`](Self::set_lfe) is on.
    pub fn lfe(&self) -> bool {
        self.lfe
    }

    /// libopus `OPUS_SET_PREDICTION_DISABLED`: code every frame so that a
    /// decoder can start with it, at the cost of quality.
    ///
    /// - SILK codes every frame as the first one after a reset
    ///   (enc_API.c:172-177).
    /// - CELT runs without its pitch pre-filter and codes the coarse band
    ///   energies without prediction from the frame before
    ///   (`CELT_SET_PREDICTION(0)`, opus_encoder.c:2288-2294).
    pub fn set_prediction_disabled(&mut self, disabled: bool) {
        self.prediction_disabled = disabled;
    }

    /// Whether [`set_prediction_disabled`](Self::set_prediction_disabled)
    /// is on.
    pub fn prediction_disabled(&self) -> bool {
        self.prediction_disabled
    }

    /// libopus `OPUS_SET_EXPERT_FRAME_DURATION`: the duration of the frames
    /// the encoder codes.
    ///
    /// - [`FrameDuration::Argument`], the default: the `frame_size` of each
    ///   [`encode`](Self::encode) call is the frame.
    /// - Any other value: each call codes a frame of that duration. Its
    ///   `frame_size` must be at least that long, and `input` must still
    ///   hold `frame_size * channels` samples. Only the first frame of
    ///   them is coded. A shorter `frame_size` is refused.
    ///
    /// libopus calls this "variable duration": the encoder may code less
    /// audio than the call passes. The caller learns how much from the
    /// packet ([`packet::nb_samples`]) and passes the rest again in the
    /// next call. libopus 1.6 makes no choice of its own here; the
    /// duration is the one that was set.
    ///
    /// The setting stays across [`reset_state`](Self::reset_state), as in
    /// libopus.
    pub fn set_expert_frame_duration(&mut self, duration: FrameDuration) {
        self.variable_duration = duration;
    }

    /// libopus `OPUS_GET_EXPERT_FRAME_DURATION`: the value set with
    /// [`set_expert_frame_duration`](Self::set_expert_frame_duration).
    pub fn expert_frame_duration(&self) -> FrameDuration {
        self.variable_duration
    }

    /// libopus `OPUS_SET_FORCE_CHANNELS`: code `Some(1)` or `Some(2)`
    /// channels, or let the bitrate decide with `None`. The count may not
    /// exceed the input's. A mono encoder codes mono whatever is set.
    ///
    /// Without a forced count, stereo input is coded as mono below about
    /// 17 kb/s for music and 19 kb/s for speech, with 1 kb/s of hysteresis
    /// (opus_encoder.c:1428-1451).
    ///
    /// As in libopus, a packet made of several frames (40 to 120 ms of
    /// CELT or Hybrid, 80 to 120 ms of SILK) that would lead over to mono
    /// sets this to `Some(1)` itself (opus_encoder.c:1763-1765). The stream
    /// then stays mono until the caller sets the count again.
    pub fn set_force_channels(&mut self, channels: Option<usize>) -> Result<(), &'static str> {
        match channels {
            Some(n) if n < 1 || n > self.channels => Err("Invalid forced channel count"),
            _ => {
                self.force_channels = channels;
                Ok(())
            }
        }
    }

    /// The count set with [`set_force_channels`](Self::set_force_channels).
    pub fn force_channels(&self) -> Option<usize> {
        self.force_channels
    }

    /// libopus `OPUS_SET_ENERGY_MASK`: how far each band of each channel
    /// lies above what the other channels of a surround stream mask. A
    /// multistream encoder computes it for every frame.
    ///
    /// `mask` holds 21 values per channel of this encoder, the first
    /// channel first, in units of 6 dB (log2 of an amplitude ratio). A
    /// negative value says the band is masked. `None` clears the mask, and
    /// so does a slice that is too short. libopus keeps a pointer to the
    /// caller's array; this function copies the values, so call it again
    /// when they change.
    ///
    /// What the encoder does with a mask, as libopus does:
    ///
    /// - CELT-only packets: the CELT encoder spends fewer bits on masked
    ///   bands and, with a variable bitrate, on the whole frame.
    /// - SILK-only and Hybrid packets with a variable bitrate: the SILK
    ///   bitrate goes down with the average depth of the mask over SILK's
    ///   bands, by at most two thirds. A mask of all zeros raises it a
    ///   little.
    /// - Hybrid packets: the band above 8 kHz is not attenuated at low
    ///   bitrates.
    /// - With a constant bitrate, and for a low-frequency-effects encoder,
    ///   the mask changes no packet size.
    ///
    /// One difference from libopus, on purpose. In a Hybrid packet the CELT
    /// layer's bitrate is what the SILK bitrate leaves of the total. When a
    /// mask raises the SILK bitrate to within 500 b/s of the total, libopus
    /// fails to set the CELT bitrate and the CELT layer runs at its maximum:
    /// a 10 kb/s stream comes out at about 500 kb/s. This encoder gives the
    /// CELT layer the lowest bitrate it takes instead (501 b/s).
    pub fn set_energy_mask(&mut self, mask: Option<&[f32]>) {
        let n = 21 * self.channels;
        self.energy_mask = mask.filter(|m| m.len() >= n).map(|m| {
            let mut values = [0.0; 21 * OPUS_MAX_CHANNELS];
            values[..n].copy_from_slice(&m[..n]);
            values
        });
    }

    /// The mask set with [`set_energy_mask`](Self::set_energy_mask): 21
    /// values per channel, or `None` when there is none.
    pub fn energy_mask(&self) -> Option<&[f32]> {
        self.energy_mask.as_ref().map(|m| &m[..21 * self.channels])
    }

    /// libopus `OPUS_GET_FINAL_RANGE`: the state of the range coder after
    /// the last frame of the last packet. A decoder that reads the packet
    /// correctly ends in the same state
    /// ([`OpusDecoder::final_range`]). The value is 0 for a packet that
    /// carries no range-coded data: one too small to code anything, or one
    /// whose frame did not fit and was replaced by an empty frame.
    pub fn final_range(&self) -> u32 {
        self.range_final
    }

    /// libopus `OPUS_RESET_STATE`: forget the signal coded so far. The
    /// encoder then behaves like a new one with the same settings.
    ///
    /// The settings that stay are the bitrate, the complexity, CBR or VBR
    /// and the VBR constraint, in-band FEC, the packet loss percentage, the
    /// forced mode, the bandwidth and maximum bandwidth, the forced channel
    /// count, the prediction switch, the expert frame duration and the LFE
    /// flag. The energy mask is cleared, as in libopus.
    ///
    /// As in libopus, three things SILK reported for the last packet also
    /// stay: whether the packet was to carry FEC, whether SILK allowed a
    /// bandwidth switch, and whether it ran at wideband. With FEC on and
    /// more than 5 % loss, the first packets after a reset can therefore
    /// have another bandwidth than those of a new encoder.
    pub fn reset_state(&mut self) {
        // `new` accepted this sampling rate, channel count and application
        // before, so it does again.
        let Ok(mut fresh) = Self::new(self.sampling_rate, self.channels, self.application) else {
            return;
        };
        fresh.user_bitrate = self.user_bitrate;
        fresh.complexity = self.complexity;
        fresh.use_cbr = self.use_cbr;
        fresh.vbr_constraint = self.vbr_constraint;
        fresh.use_inband_fec = self.use_inband_fec;
        fresh.packet_loss_perc = self.packet_loss_perc;
        fresh.user_forced_mode = self.user_forced_mode;
        fresh.user_bandwidth = self.user_bandwidth;
        fresh.max_bandwidth = self.max_bandwidth;
        fresh.lfe = self.lfe;
        fresh.prediction_disabled = self.prediction_disabled;
        fresh.variable_duration = self.variable_duration;
        // C `force_channels` and `silk_mode.toMono` lie before
        // `OPUS_ENCODER_RESET_START` too.
        fresh.force_channels = self.force_channels;
        fresh.silk_to_mono = self.silk_to_mono;
        // libopus clears its state from `OPUS_ENCODER_RESET_START` on. The
        // SILK control structure `silk_mode` lies before that, so what SILK
        // reported for the last packet stays (opus_encoder.c:3243-3271).
        // The first of them shows after a reset: it is the hysteresis of
        // the next FEC decision.
        fresh.silk_lbrr_coded = self.silk_lbrr_coded;
        fresh.silk_allow_bandwidth_switch = self.silk_allow_bandwidth_switch;
        fresh.silk_in_wb_mode_without_variable_lp = self.silk_in_wb_mode_without_variable_lp;
        *self = fresh;
    }

    /// The sampling rate of the input in Hz (`OPUS_GET_SAMPLE_RATE`).
    pub fn sampling_rate(&self) -> i32 {
        self.sampling_rate
    }

    /// The number of input channels.
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// libopus `OPUS_GET_LOOKAHEAD`: the delay the encoder adds, in samples
    /// per channel at the input sampling rate. A decoder's output lags the
    /// encoder's input by this much. An Ogg Opus file stores it, scaled to
    /// 48 kHz, as the pre-skip.
    pub fn lookahead(&self) -> usize {
        // 2.5 ms of CELT overlap, and the 4 ms that align CELT with SILK
        // unless the application is `RestrictedLowDelay`.
        self.sampling_rate as usize / 400 + self.delay_compensation
    }

    /// Encode a frame of interleaved f32 PCM (scale ±1.0; C `opus_encode_float`
    /// convention where 1.0 maps to PCM16 32768). `input` must hold
    /// `frame_size * channels` samples.
    ///
    /// `frame_size` is in samples per channel: 2.5, 5, 10, 20, 40, 60, 80,
    /// 100 or 120 ms. With an expert frame duration set
    /// ([`set_expert_frame_duration`](Self::set_expert_frame_duration)),
    /// only the first frame of that duration is coded, and `frame_size`
    /// must be at least that long.
    ///
    /// The packet never exceeds `output`. As in libopus, the length of
    /// `output` also counts as a limit on the bitrate: when it holds less
    /// than the bitrate gives the frame, the encoder chooses mode and
    /// bandwidth for the lower rate. Pass a buffer of 1276 bytes per 20 ms
    /// or more when the bitrate alone should decide.
    pub fn encode(
        &mut self,
        input: &[f32],
        frame_size: usize,
        output: &mut [u8],
    ) -> Result<usize, &'static str> {
        let frame_size = self.select_frame_size(input.len(), frame_size)?;
        self.encode_impl(input, frame_size, output)
    }

    /// Encode a frame of interleaved 16-bit PCM (libopus `opus_encode`).
    /// The packets are those of [`encode`](Self::encode) with each sample
    /// converted to `sample as f32 / 32768.0`, which is how this function
    /// converts it.
    pub fn encode_i16(
        &mut self,
        input: &[i16],
        frame_size: usize,
        output: &mut [u8],
    ) -> Result<usize, &'static str> {
        let frame_size = self.select_frame_size(input.len(), frame_size)?;
        // libopus `INT16TORES`, over the frame that is coded. The
        // conversion is exact, and the 16-bit high-pass filter of the
        // `Voip` application rounds the float samples back to the same
        // 16-bit values.
        let n = frame_size * self.channels;
        let mut pcm: FixedVec<f32, OPUS_PCM_PACKET> = FixedVec::from_value(0.0, n);
        for (dst, &src) in pcm.iter_mut().zip(&input[..n]) {
            *dst = int16_to_res(src);
        }
        self.encode_impl(&pcm, frame_size, output)
    }

    /// The frame an encode call codes: libopus `frame_size_select`, which
    /// `opus_encode` and `opus_encode_float` call first. `frame_size` is
    /// the caller's (C `analysis_frame_size`), and `input_len` the number
    /// of samples the caller passed. Returns the frame size in samples per
    /// channel.
    fn select_frame_size(&self, input_len: usize, frame_size: usize) -> Result<usize, &'static str> {
        let selected = frame_size_select(frame_size, self.variable_duration, self.sampling_rate)
            .ok_or("Invalid frame size for sampling rate")?;
        // The caller passes `frame_size` samples per channel, as libopus
        // reads them. Short input is rejected here, so that it cannot
        // panic inside the high-pass filter or the conversion loops.
        if input_len < frame_size.saturating_mul(self.channels) {
            return Err("Input buffer too small for frame");
        }
        Ok(selected)
    }

    /// One packet: libopus `opus_encode_native` (opus_encoder.c:1182-1853).
    /// `frame_size` is the frame [`select_frame_size`](Self::select_frame_size)
    /// chose. The multistream encoder calls this directly, as libopus
    /// calls `opus_encode_native`, so the expert frame duration of a
    /// stream encoder plays no part there.
    ///
    /// What this function, [`decide_mode_and_bandwidth`](Self::decide_mode_and_bandwidth)
    /// and [`encode_frame`](Self::encode_frame) still do differently from
    /// libopus:
    ///
    /// 1. No speech/music analysis (analysis.c). The chance of speech is a
    ///    constant per application, the bandwidth the analysis detects is
    ///    not applied (C 1651-1673), and the CELT encoder gets no analysis.
    /// 2. No discontinuous transmission (C 1455-1461, 1904-1927, 2564-2577).
    /// 3. SILK's input with the `Voip` application. The float build of
    ///    libopus feeds SILK the float `pcm_buf`, filtered by `hp_cutoff`
    ///    and rounded to 16 bits. Here SILK gets the output of the integer
    ///    `hp_cutoff` of the fixed-point build, which has its own filter
    ///    state and only runs on frames that SILK codes. The other
    ///    applications follow the float build.
    /// 4. Packets made of several frames (CELT and Hybrid at 40 to 120 ms,
    ///    SILK-only at 80 to 120 ms) with a variable bitrate: the limit of
    ///    each frame to its share of the bitrate (C 1785) is left out on
    ///    purpose, see `encode_multiframe`.
    /// 5. The Hybrid CELT bitrate has a floor of 501 b/s, on purpose.
    /// 6. 16-bit input is coded with the 24-bit silence threshold of float
    ///    input (libopus: `lsb_depth` 16), so that both entry points give
    ///    the same packets.
    /// 7. The controls for signal type and phase inversion are not
    ///    offered.
    /// 8. On purpose: a frame that announces the switch to CELT-only
    ///    without room for the redundant frame still resets the CELT
    ///    encoder. The decoder starts the CELT-only packets from the reset
    ///    state then; libopus's encoder goes on from its last CELT frame.
    /// 9. The energy mask is copied. After a CELT reset libopus has lost
    ///    its pointer to the mask until the caller sets it again; here the
    ///    copy is applied again on the next frame.
    pub(crate) fn encode_impl(
        &mut self,
        input: &[f32],
        frame_size: usize,
        output: &mut [u8],
    ) -> Result<usize, &'static str> {
        // C 1223: no final range until a frame is coded.
        self.range_final = 0;
        let out_data_bytes = output.len();
        // C 1221: "Just avoid insane packet sizes here, but the real bounds
        // are applied later on."
        let mut max_data_bytes = out_data_bytes.min(6 * OPUS_MAX_PACKET_BYTES);
        if max_data_bytes == 0 {
            return Err("Output buffer too small");
        }

        let frame_rate = frame_rate_from_params(self.sampling_rate, frame_size)
            .ok_or("Invalid frame size for sampling rate")?;
        // C 1231-1236: "Cannot encode 100 ms in 1 byte". A 100 ms packet
        // without audio needs a frame count byte: it is five 20 ms frames.
        if max_data_bytes == 1 && frame_size * 10 == self.sampling_rate as usize {
            return Err("Output buffer too small");
        }

        // The encoder consumes exactly frame_size * channels samples per call.
        // Short input is rejected here, so that it cannot panic inside the
        // high-pass filter or the float-to-integer conversion loops.
        if input.len() < frame_size * self.channels {
            return Err("Input buffer too small for frame");
        }

        // C 1321-1324: the stereo width of the input, once per packet,
        // whatever is done with the packet. Not when the caller asks for
        // mono packets.
        self.stereo_width = if self.channels == 2 && self.force_channels != Some(1) {
            compute_stereo_width(input, frame_size, self.sampling_rate, &mut self.width_mem)
        } else {
            0.0
        };

        // C 1325-1334. `user_bitrate_to_bitrate`: the packet is coded at the
        // bitrate asked for, or at the rate the output buffer stands for
        // when that is lower. A constant bitrate codes whole bytes, and the
        // bitrate becomes the rate of those bytes.
        let fs = self.sampling_rate;
        let mut bitrate = self.user_bitrate_to_bitrate(frame_size, max_data_bytes);
        let mut cbr_bytes = 0usize;
        if self.use_cbr {
            let nominal = (bitrate_to_bits(bitrate, fs, frame_size) + 4) / 8;
            cbr_bytes = (nominal.max(0) as usize).min(max_data_bytes);
            bitrate = bits_to_bitrate(8 * cbr_bytes as i32, fs, frame_size);
            // "Make sure we provide at least one byte to avoid failing."
            max_data_bytes = cbr_bytes.max(1);
        }
        self.packet_bitrate_bps = bitrate;

        // C 1340-1406: "If the space is too low to do something useful,
        // emit 'PLC' frames". This comes before every decision, and before
        // a frame of 40 ms or more is split into parts.
        if max_data_bytes < 3
            || bitrate < 3 * frame_rate * 8
            || (frame_rate < 50 && (max_data_bytes as i32 * frame_rate < 300 || bitrate < 2400))
        {
            return self.write_placeholder(frame_rate, max_data_bytes, output);
        }

        let decisions = self.decide_mode_and_bandwidth(frame_size, frame_rate, max_data_bytes);
        let mode = decisions.mode;

        // C 1697-1838: "Can't support higher than >60 ms frames, and
        // >20 ms when in Hybrid or CELT-only modes". Longer packets are
        // made of several frames.
        let fs_u = fs as usize;
        if (mode != OpusMode::SilkOnly && frame_size > fs_u / 50) || frame_size > 3 * fs_u / 50 {
            return self.encode_multiframe(input, frame_size, decisions, cbr_bytes, output);
        }
        self.encode_frame(input, frame_size, &mut output[..max_data_bytes], decisions)
    }

    /// One frame of a packet: libopus `opus_encode_frame_native`
    /// (opus_encoder.c:1855-2655).
    ///
    /// `output.len()` is the largest size the frame's packet may have (C
    /// `orig_max_data_bytes`). With a constant bitrate the packet is padded
    /// to exactly that size.
    fn encode_frame(
        &mut self,
        input: &[f32],
        frame_size: usize,
        output: &mut [u8],
        d: FrameDecisions,
    ) -> Result<usize, &'static str> {
        let orig_max_data_bytes = output.len();
        // C `max_data_bytes`: one frame takes 1275 bytes at most.
        let n_bytes = orig_max_data_bytes.min(OPUS_MAX_PACKET_BYTES);
        self.range_final = 0;
        // The range coder needs room. libopus never gets here with less:
        // `opus_encode_native` writes a packet without audio instead.
        if n_bytes < 3 {
            return Err("Output buffer too small");
        }
        let fs = self.sampling_rate;
        // C `st->channels` (the input's, SILK's `nChannelsAPI`) and
        // `st->stream_channels` (the packet's, SILK's `nChannelsInternal`).
        let ch = self.channels;
        let sch = self.stream_channels;
        let frame_rate = fs / frame_size as i32;
        let mode = d.mode;
        let frame_bandwidth = d.bandwidth;
        let lbrr_coded = d.lbrr_coded;
        let f2_5 = fs as usize / 400;
        let f5 = 2 * f2_5;

        match (mode, frame_rate) {
            (OpusMode::CeltOnly, 400 | 200 | 100 | 50) => {}
            (OpusMode::CeltOnly, _) => return Err("Unsupported frame size for CELT-only mode"),
            (OpusMode::Hybrid, 100 | 50) => {}
            (OpusMode::Hybrid, _) => return Err("Unsupported frame size for Hybrid mode"),
            // 2.5/5 ms are CELT-only durations: SILK frames are 10 ms or
            // longer (libopus), and sub-10 ms frames break the SILK
            // analysis geometry (e.g. the VAD band split at 12 kHz).
            (OpusMode::SilkOnly, 100 | 50 | 25 | 16) => {}
            (OpusMode::SilkOnly, _) => return Err("Unsupported frame size for SILK-only mode"),
        }

        // C 1933-1954: what goes with a change of mode or of SILK's rate.
        let (mut redundancy, mut celt_to_silk) = (d.redundancy, d.celt_to_silk);
        // C `prefill`: 1 after CELT-only packets, 2 for the first frame at
        // a new SILK rate.
        let mut prefill = d.prefill as u8;
        if self.silk_bw_switch {
            redundancy = true;
            celt_to_silk = true;
            self.silk_bw_switch = false;
            // "Do a prefill without resetting the sampling rate control."
            // After CELT-only packets SILK has no rate to keep.
            if prefill == 0 {
                prefill = 2;
            }
        }
        // "If we decided to go with CELT, make sure redundancy is off, no
        // matter what we decided earlier."
        if mode == OpusMode::CeltOnly {
            redundancy = false;
        }
        let mut redundancy_bytes = 0usize;
        if redundancy {
            redundancy_bytes = compute_redundancy_bytes(
                n_bytes,
                self.packet_bitrate_bps,
                frame_rate,
                self.stream_channels,
            );
            if redundancy_bytes == 0 {
                redundancy = false;
            }
        }

        self.rc.reset_for_encode((n_bytes - 1) as u32);

        // Hybrid: SILK's base rate, and the gain on CELT's input (C `HB_gain`).
        let mut hybrid_silk_rate = 0i32;
        let mut hb_gain = 1.0f32;

        // C opus_encode_frame_native: high-pass cutoff state is updated for ALL
        // modes (celt_encoder.c:1969-1977); the filtered signal is what feeds
        // both SILK (hybrid) and CELT. Compute it here so CELT-only frames also
        // keep `variable_hp_smth2_q15` in sync with libopus.
        let hp_freq_smth1 = if mode == OpusMode::CeltOnly {
            silk_lin2log(60) << 8
        } else {
            self.silk.mid().s_cmn.variable_hp_smth1_q15
        };

        // SILK_FIX_CONST(VARIABLE_HP_SMTH_COEF2, 16) with 0.015: 983 (opus_encoder.c).
        const VARIABLE_HP_SMTH_COEF2_Q16: i32 = 983;
        self.variable_hp_smth2_q15 = silk_smlawb(
            self.variable_hp_smth2_q15,
            hp_freq_smth1 - self.variable_hp_smth2_q15,
            VARIABLE_HP_SMTH_COEF2_Q16,
        );

        let cutoff_hz = silk_log2lin(silk_rshift(self.variable_hp_smth2_q15, 8));

        let delay = self.delay_compensation;
        let ebuf = self.encoder_buffer;

        // C 1966-2028, `pcm_buf`: the look-ahead taken from the delay
        // buffer, then the filtered frame. libopus builds it in every mode:
        // the CELT layer, the redundant CELT frames of a SILK-only packet
        // and the delay buffer read it. A `RestrictedLowDelay` encoder has
        // no look-ahead and no delay buffer (`delay` and `ebuf` are 0); its
        // frame is filtered all the same.
        let prefix = delay * ch;
        // The frame was not a sane signal and is coded as silence.
        let mut frame_cleared = false;
        {
            self.buf_celt_pcm.resize((delay + frame_size) * ch, 0.0);
            // 1. delay prefix from the ring (C: OPUS_COPY at 1967)
            let src_start = (ebuf - delay) * ch;
            self.buf_celt_pcm[..prefix]
                .copy_from_slice(&self.delay_buffer[src_start..src_start + prefix]);
            // 2. filter current frame into pcm_buf[delay..] (C: 2002-2010)
            let out = &mut self.buf_celt_pcm[prefix..];
            if self.application == Application::Voip {
                hp_cutoff_float(
                    input,
                    cutoff_hz,
                    out,
                    &mut self.hp_mem_float,
                    frame_size,
                    ch,
                    fs,
                );
            } else {
                dc_reject_float(input, 3, out, &mut self.hp_mem_float, frame_size, ch, fs);
            }
            // 3. float NaN guard (C: 2016-2028)
            if !input_is_sane(&self.buf_celt_pcm[prefix..]) {
                self.buf_celt_pcm[prefix..].fill(0.0);
                self.hp_mem_float = [0.0; 4];
                frame_cleared = true;
            }
        }

        // C `internalSampleRate` of the SILK encoder, 0 for CELT-only frames.
        let mut silk_fs_khz = 0;
        if mode == OpusMode::SilkOnly || mode == OpusMode::Hybrid {
            let frame_ms = (frame_size as i32 * 1000) / fs;

            // C opus_encoder.c `bits_target` / `total_bitRate` (1.6): SILK
            // codes at the packet's bitrate, capped by the packet budget
            // without the redundant frame, less the TOC byte. Under VBR the
            // budget is the caller's output buffer, which must cap the
            // rate, not set it.
            let rate_bits = bitrate_to_bits(self.packet_bitrate_bps, fs, frame_size);
            let bits_target = (8 * (n_bytes - redundancy_bytes) as i32).min(rate_bits) - 8;
            let total_bitrate = bits_to_bitrate(bits_target, fs, frame_size);

            let silk_bitrate = if mode == OpusMode::Hybrid {
                // C opus_encoder.c:2049: SILK's base rate out of the rate
                // the packet codes at.
                let rate = compute_silk_rate_for_hybrid(
                    total_bitrate,
                    frame_bandwidth,
                    frame_ms == 20,
                    !self.use_cbr,
                    lbrr_coded != 0,
                    self.stream_channels,
                );
                // C opus_encoder.c:2057-2062: the fewer bits the high band
                // gets, the more it is attenuated before CELT codes it. Not
                // in a surround stream (`if (!st->energy_masking)`).
                //
                // `HB_gain = Q15ONE - SHR32(celt_exp2(-celt_rate/1024), 1)`:
                // the shift turns the Q16 result of the fixed-point
                // `celt_exp2` into Q15. It is not a halving, and in the float
                // build it does nothing. The gain is 1 - 2^(-celt_rate/1024).
                if self.energy_mask.is_none() {
                    let celt_rate = total_bitrate - rate;
                    // C float `celt_exp2(x)`: `(float)exp(0.6931471805599453094*(x))`.
                    let x = -celt_rate as f32 * (1.0 / 1024.0);
                    hb_gain = 1.0 - (core::f64::consts::LN_2 * x as f64).exp() as f32;
                }
                rate
            } else {
                // C: SILK gets all bits, at the requested rate.
                total_bitrate
            };
            // C opus_encoder.c:2068-2108, "Surround masking for SILK": with a
            // variable bitrate the SILK rate follows how deeply the other
            // channels mask this one, over the bands SILK codes.
            let silk_bitrate = if let Some(energy_mask) = &self.energy_mask
                && !self.use_cbr
                && !self.lfe
            {
                let (end, srate) = match frame_bandwidth {
                    Bandwidth::Narrowband => (13usize, 8000.0f32),
                    Bandwidth::Mediumband => (15, 12000.0),
                    _ => (17, 16000.0),
                };
                let mut mask_sum = 0.0f32;
                for c in 0..ch {
                    for i in 0..end {
                        // C `MAX16(MIN16(mask, .5f), -2.0f)`. Not `clamp`: a
                        // value that is not a number becomes 0.5 here, as in
                        // C, and `clamp` would pass it on.
                        #[allow(clippy::manual_clamp)]
                        let mut mask = energy_mask[21 * c + i].min(0.5).max(-2.0);
                        if mask > 0.0 {
                            mask *= 0.5;
                        }
                        mask_sum += mask;
                    }
                }
                // "Conservative rate reduction, we cut the masking in half".
                let mut masking_depth = mask_sum / end as f32 * ch as f32;
                masking_depth += 0.2;
                let rate_offset = ((srate * masking_depth) as i32).max(-2 * silk_bitrate / 3);
                // "Split the rate change between the SILK and CELT part for
                // hybrid."
                if matches!(frame_bandwidth, Bandwidth::Superwideband | Bandwidth::Fullband) {
                    silk_bitrate + 3 * rate_offset / 5
                } else {
                    silk_bitrate + rate_offset
                }
            } else {
                silk_bitrate
            };
            if mode == OpusMode::Hybrid {
                hybrid_silk_rate = silk_bitrate;
            }

            // C 2154-2189, `silk_mode.maxBits`: "Max bits for SILK, counting
            // ToC, redundancy bytes, and optionally redundancy."
            let mut silk_max_bits = ((n_bytes - 1) * 8) as i32;
            if redundancy && redundancy_bytes >= 2 {
                // "Counting 1 bit for redundancy position and 20 bits for
                // flag+size (only for hybrid)."
                silk_max_bits -= redundancy_bytes as i32 * 8 + 1;
                if mode == OpusMode::Hybrid {
                    silk_max_bits -= 20;
                }
            }
            if mode == OpusMode::Hybrid {
                if self.use_cbr {
                    // "Allow SILK to steal up to 25% of the remaining bits".
                    let silk_bits =
                        (silk_bitrate as i64 * frame_size as i64 / fs as i64) as i32;
                    let other_bits = 0i32.max(silk_max_bits - silk_bits);
                    silk_max_bits = 0i32.max(silk_max_bits - other_bits * 3 / 4);
                } else {
                    // C opus_encoder.c:2176-2181: the SILK rate that goes
                    // with the largest packet allowed.
                    let max_bit_rate = compute_silk_rate_for_hybrid(
                        silk_max_bits * fs / frame_size as i32,
                        frame_bandwidth,
                        frame_ms == 20,
                        true,
                        lbrr_coded != 0,
                        self.stream_channels,
                    );
                    silk_max_bits = bitrate_to_bits(max_bit_rate, fs, frame_size);
                }
            }
            // C 2112-2147: the internal rates SILK may choose from. Hybrid's
            // CELT layer starts at 8 kHz, so its SILK layer is wideband. A
            // SILK-only packet with few bytes limits the rate.
            let mut desired_khz = frame_bandwidth.silk_khz();
            let min_khz = if mode == OpusMode::Hybrid { 16 } else { 8 };
            let mut max_khz = 16;
            if mode == OpusMode::SilkOnly {
                let mut effective_max_rate = bits_to_bitrate(8 * n_bytes as i32, fs, frame_size);
                if frame_rate > 50 {
                    effective_max_rate = effective_max_rate * 2 / 3;
                }
                if effective_max_rate < 8000 {
                    max_khz = 12;
                    desired_khz = desired_khz.min(12);
                }
                if effective_max_rate < 7000 {
                    max_khz = 8;
                    desired_khz = desired_khz.min(8);
                }
            }

            // SILK's input: the filtered frame.
            //
            // - VoIP: the fixed-point build of libopus, whose `hp_cutoff`
            //   is an integer filter on the 16-bit input. SILK-only VoIP
            //   packets are byte for byte those of that build
            //   (tests/silk_fixed_point_compare.rs).
            // - Otherwise: the float build. SILK reads the frame of
            //   `pcm_buf` that `dc_reject` filtered (opus_encoder.c:2211).
            //   An offset in the input is then gone in SILK packets as it is
            //   in CELT ones, and the decoded signal does not step by the
            //   offset when the mode changes.
            let required_size = frame_size * ch;
            let silk_input = if self.application == Application::Voip {
                self.buf_filtered.resize(required_size, 0);
                if frame_cleared {
                    // C 2023-2027: the frame is cleared for SILK as well, and
                    // the filter starts again.
                    state_mut(&mut self.buf_filtered)[..required_size].fill(0);
                    self.hp_mem.fill(0);
                } else {
                    hp_cutoff(
                        input,
                        cutoff_hz,
                        state_mut(&mut self.buf_filtered),
                        &mut self.hp_mem,
                        frame_size,
                        ch,
                        fs,
                    );
                }
                SilkInput::Int16(&self.buf_filtered[..required_size])
            } else {
                SilkInput::Float(&self.buf_celt_pcm[prefix..prefix + required_size])
            };

            // C 1576-1581, 2190-2208: SILK starts afresh.
            //
            // - After CELT-only packets (`prefill == 1`): everything is
            //   cleared, once per packet (`silk_InitEncoder`). The later
            //   parts of a packet made of several frames clear the
            //   channels again.
            // - For the first frame at a new SILK rate that follows a
            //   redundant frame (`prefill == 2`): the channels are cleared,
            //   the state of the input low-pass filter stays.
            //
            // The encoder is then fed the 10 ms in the delay buffer.
            // Everything before the last `delay_compensation + 2.5 ms` is
            // cleared, and the next 2.5 ms fade in, "to avoid the encoder
            // trying to encode a discontinuity".
            //
            // As in libopus, the fade and the clearing change the delay
            // buffer itself. A Hybrid frame after CELT-only frames reads
            // the faded 2.5 ms again: its CELT encoder starts from them
            // (`tmp_prefill`, C 2297-2302).
            let mut silk_prefill = None;
            if prefill != 0 {
                if prefill == 1 && self.prev_enc_mode == Some(OpusMode::CeltOnly) {
                    self.silk.reset();
                }
                let mut prefill_input = None;
                if ebuf > 0 && delay > 0 && ebuf >= delay + f2_5 {
                    let offset = ebuf - delay - f2_5;
                    let faded = &mut self.delay_buffer[..ebuf * ch];
                    gain_fade(&mut faded[offset * ch..], 0.0, 1.0, f2_5, ch, fs);
                    faded[..offset * ch].fill(0.0);
                    // The prefill reads the float delay buffer whatever the
                    // application, so it follows the float build.
                    prefill_input = Some(SilkInput::Float(&self.delay_buffer[..ebuf * ch]));
                }
                silk_prefill = Some(SilkPrefill { rate_switch: prefill == 2, input: prefill_input });
            }

            let mut silk_ctl = SilkControl {
                channels_internal: sch,
                payload_size_ms: frame_ms,
                min_khz,
                max_khz,
                desired_khz,
                bit_rate: silk_bitrate,
                max_bits: silk_max_bits,
                // C opus_encoder.c:2133-2177: SILK runs at a constant
                // bitrate only in SILK-only packets. A Hybrid packet with a
                // constant bitrate codes SILK at a variable one, with a cap,
                // and CELT takes up the rest ("switch SILK to VBR with
                // cap").
                use_cbr: self.use_cbr && mode != OpusMode::Hybrid,
                complexity: self.complexity,
                packet_loss_perc: self.packet_loss_perc,
                lbrr_coded: lbrr_coded != 0,
                activity: 1,
                to_mono: self.silk_to_mono,
                reduced_dependency: self.prediction_disabled,
                opus_can_switch: self.silk_opus_can_switch,
                allow_bandwidth_switch: self.silk_allow_bandwidth_switch,
                ..SilkControl::default()
            };
            let result = self.silk.encode(&mut silk_ctl, silk_input, silk_prefill, &mut self.rc);
            // C 2207: "Prevent a second switch in the real encode call."
            // A restart for a rate switch has made that switch.
            if prefill != 0 {
                self.silk_opus_can_switch = false;
            }
            result?;

            // In a packet made of several frames only the last part leaves
            // its answer, as in libopus.
            self.silk_allow_bandwidth_switch = silk_ctl.allow_bandwidth_switch;
            self.silk_in_wb_mode_without_variable_lp = silk_ctl.in_wb_mode_without_variable_lp;
            silk_fs_khz = silk_ctl.internal_khz;
            // enc_API.c:584: the stereo width SILK settled on, 0 for the
            // packet that leads over to mono. The CELT layer of a stereo
            // Hybrid frame is narrowed to it below.
            if ch == 2 {
                self.silk_stereo_width_q14 = silk_ctl.stereo_width_q14;
            }
            // C opus_encoder.c:2233: a rate switch SILK is ready for happens
            // on the next packet, not in the middle of this one.
            self.silk_opus_can_switch = silk_ctl.switch_ready && !self.nonfinal_frame;
            // C 2250-2259: that packet starts with a redundant CELT frame
            // (`silk_bw_switch`), and this one ends with one, when it has
            // room for it.
            if self.silk_opus_can_switch {
                redundancy_bytes = compute_redundancy_bytes(
                    n_bytes,
                    self.packet_bitrate_bps,
                    frame_rate,
                    self.stream_channels,
                );
                redundancy = redundancy_bytes != 0;
                celt_to_silk = false;
                self.silk_bw_switch = true;
            }
        }

        // The rate SILK coded this frame at. A SILK-only packet signals that
        // rate's bandwidth, not the one asked for (C opus_encoder.c:2223-2232):
        // SILK does not follow every bandwidth request at once, and a TOC
        // that names another rate makes the decoder read the frame wrongly.
        // C `curr_bandwidth`.
        let toc_bandwidth = match (mode, silk_fs_khz) {
            (OpusMode::SilkOnly, 8) => Bandwidth::Narrowband,
            (OpusMode::SilkOnly, 12) => Bandwidth::Mediumband,
            (OpusMode::SilkOnly, _) => Bandwidth::Wideband,
            _ => frame_bandwidth,
        };
        let toc = gen_toc(mode, frame_rate, toc_bandwidth, self.stream_channels);

        // C 2262-2283: settings the CELT encoder gets in every mode. A
        // SILK-only packet needs them for its redundant frame.
        self.celt_enc.complexity = self.complexity;
        // libopus `CELT_SET_END_BAND`: match the TOC bandwidth so the
        // decoder reads the same number of coded bands.
        self.celt_enc.set_end_band(toc_bandwidth.celt_end_band());
        // C 2285: the channels the packet codes. The value is 1 or the
        // input's channel count, which the CELT encoder accepts.
        let _ = self.celt_enc.set_stream_channels(self.stream_channels);
        // C 2288-2294: "We may still decide to disable prediction later".
        if mode != OpusMode::SilkOnly {
            let _ = self
                .celt_enc
                .set_prediction(if self.prediction_disabled { 0 } else { 2 });
        }
        // C 2979-2989: OPUS_SET_PACKET_LOSS_PERC goes to the CELT encoder
        // too. The value was clamped to 0..=100 above.
        let _ = self.celt_enc.set_packet_loss_perc(self.packet_loss_perc);
        // C opus_encoder.c:3283-3297: OPUS_SET_LFE and OPUS_SET_ENERGY_MASK
        // go to the CELT encoder as well. libopus hands the mask over as a
        // pointer; here the values are copied before every frame.
        self.celt_enc.set_lfe(self.lfe);
        let mask = self.energy_mask.as_ref().map(|m| &m[..21 * ch]);
        self.celt_enc.set_energy_mask(mask);

        // C 2297-2302: a CELT or Hybrid frame in another mode than the
        // frame before restarts its CELT encoder, which first codes the
        // 2.5 ms before the frame's look-ahead (below). They are taken from
        // the delay buffer before it moves on.
        let celt_restart = mode != OpusMode::SilkOnly
            && self.prev_enc_mode.is_some()
            && self.prev_enc_mode != Some(mode);
        if celt_restart {
            // `RestrictedLowDelay` keeps no delay buffer, and is always
            // CELT-only: the start never comes before the buffer.
            let start = ebuf.saturating_sub(delay + f2_5) * ch;
            self.celt_prefill[..f2_5 * ch]
                .copy_from_slice(&self.delay_buffer[start..start + f2_5 * ch]);
        }

        {
            // C 2300-2312: the delay buffer takes the end of `pcm_buf`.
            let keep = ebuf as i64 - (frame_size as i64 + delay as i64);
            if keep > 0 {
                let keep = keep as usize;
                self.delay_buffer
                    .copy_within(ch * frame_size..ch * (frame_size + keep), 0);
                let dst = ch * keep;
                let n = (frame_size + delay) * ch;
                self.delay_buffer[dst..dst + n].copy_from_slice(&self.buf_celt_pcm[..n]);
            } else {
                let n = ebuf * ch;
                let src = (frame_size + delay - ebuf) * ch;
                self.delay_buffer[..n].copy_from_slice(&self.buf_celt_pcm[src..src + n]);
            }
            // C 2313-2320: Hybrid attenuates the high band when it gets
            // few bits. The gain fades in over CELT's overlap. "gain_fade()
            // and stereo_fade() need to be after the buffer copying because
            // we don't want any of this to affect the SILK part".
            if self.prev_hb_gain < 1.0 || hb_gain < 1.0 {
                gain_fade(
                    &mut self.buf_celt_pcm[..frame_size * ch],
                    self.prev_hb_gain,
                    hb_gain,
                    frame_size,
                    ch,
                    fs,
                );
            }
        }
        self.prev_hb_gain = hb_gain;

        // C 2321-2329: the stereo width goes down with the bitrate, from
        // full width at 32 kb/s to none at 16 kb/s. A stereo Hybrid frame
        // takes the width SILK settled on instead.
        if mode != OpusMode::Hybrid || self.stream_channels == 1 {
            self.silk_stereo_width_q14 = if d.equiv_rate > 32000 {
                16384
            } else if d.equiv_rate < 16000 {
                0
            } else {
                16384 - 2048 * (32000 - d.equiv_rate) / (d.equiv_rate - 14000)
            };
        }
        // C 2330-2349: "Apply stereo width reduction (at low bitrates)",
        // not in a surround stream.
        if self.energy_mask.is_none()
            && ch == 2
            && (self.hybrid_stereo_width_q14 < (1 << 14) || self.silk_stereo_width_q14 < (1 << 14))
        {
            let g1 = self.hybrid_stereo_width_q14 as f32 * (1.0 / 16384.0);
            let g2 = self.silk_stereo_width_q14 as f32 * (1.0 / 16384.0);
            stereo_fade(&mut self.buf_celt_pcm, g1, g2, frame_size, fs);
            self.hybrid_stereo_width_q14 = self.silk_stereo_width_q14;
        }

        // C 2351-2378: say whether a redundant CELT frame follows, where it
        // lies and how long it is.
        //
        // - A Hybrid packet has a flag for it, but only when it has room
        //   for the flag and for the smallest frame that could follow
        //   (37 bits). The decoder makes the same test
        //   (opus_decoder.c:509-514).
        // - In a SILK-only packet the bytes after the SILK data are the
        //   redundant frame, so nothing but its position is written.
        if mode != OpusMode::CeltOnly
            && self.rc.tell() + 17 + 20 * (mode == OpusMode::Hybrid) as i32
                <= 8 * (n_bytes as i32 - 1)
        {
            if mode == OpusMode::Hybrid {
                self.rc.encode_bit_logp(redundancy, 12);
            }
            if redundancy {
                self.rc.encode_bit_logp(celt_to_silk, 1);
                let max_redundancy = if mode == OpusMode::Hybrid {
                    // "Reserve the 8 bits needed for the redundancy length,
                    // and at least a few bits for CELT if possible".
                    (n_bytes as i32 - 1) - ((self.rc.tell() + 8 + 3 + 7) >> 3)
                } else {
                    (n_bytes as i32 - 1) - ((self.rc.tell() + 7) >> 3)
                };
                // "Target the same bit-rate for redundancy as for the rest,
                // up to a max of 257 bytes".
                let bytes = (redundancy_bytes as i32).min(max_redundancy);
                redundancy_bytes = bytes.clamp(2, 257) as usize;
                if mode == OpusMode::Hybrid {
                    self.rc.enc_uint(redundancy_bytes as u32 - 2, 256);
                }
            }
        } else {
            redundancy = false;
        }
        if !redundancy {
            self.silk_bw_switch = false;
            redundancy_bytes = 0;
        }
        // C 2386-2413: the bytes of the main frame.
        let mut nb_compr_bytes = if mode == OpusMode::SilkOnly {
            (((self.rc.tell() + 7) >> 3).max(0) as usize).min(n_bytes - 1 - redundancy_bytes)
        } else {
            let nb = n_bytes - 1 - redundancy_bytes;
            self.rc.shrink(nb as u32);
            nb
        };
        // "If false, we already busted the budget and we'll end up with a
        // 'PLC frame'" (C 2484).
        let fits = self.rc.error == 0 && self.rc.tell() <= 8 * nb_compr_bytes as i32;

        // The redundant frame's bytes and final range.
        let mut redundant = [0u8; 257];
        let mut redundant_rng = 0u32;

        // C 2419-2425 (CELT_SET_SILK_INFO): the SILK frame's signal type and
        // quantization offset steer CELT's bit target and its handling of
        // transients. A reset of the CELT encoder below clears them again,
        // as in libopus: the first Hybrid frame after a reset is coded with
        // signal type 0 and offset 0.
        if mode == OpusMode::Hybrid {
            const QUANTIZATION_OFFSETS_Q10: [[i32; 2]; 2] = [[100, 240], [32, 100]];
            let indices = &self.silk.mid().s_cmn.indices;
            self.celt_enc.set_silk_info(
                indices.signal_type as i32,
                QUANTIZATION_OFFSETS_Q10[(indices.signal_type >> 1) as usize & 1]
                    [indices.quant_offset_type as usize & 1],
            );
        }

        // C 2427-2441: "5 ms redundant frame for CELT->SILK". It continues
        // the CELT stream of the packets before, so it is coded first, from
        // the start of `pcm_buf`. Then the CELT encoder forgets that stream.
        if redundancy && celt_to_silk && fits {
            redundant_rng = encode_celt_redundancy(
                state_mut(&mut self.celt_enc),
                &self.buf_celt_pcm[..f5 * ch],
                f5,
                &mut redundant[..redundancy_bytes],
            )?;
            self.celt_enc.reset_state();
        }

        if mode != OpusMode::SilkOnly {
            let start_band = if mode == OpusMode::Hybrid { HYBRID_START_BAND } else { 0 };
            // C `st->bitrate_bps` (opus_encoder.c:2454, 2462): CELT's
            // variable bitrate goes by the packet's bitrate, which the
            // output buffer can lower. In Hybrid it is what SILK's rate
            // leaves of the bitrate. With a constant bitrate libopus leaves
            // CELT at `OPUS_BITRATE_MAX` (C 2286): CELT fills the bytes it
            // is given, and its decisions go by the rate of the whole
            // packet.
            //
            // CELT takes no bitrate under 501 b/s. In libopus such a
            // request fails, and CELT keeps `OPUS_BITRATE_MAX`: the packet
            // grows to whatever the buffer holds. opus1 gives CELT 501 b/s
            // instead, a difference from libopus on purpose. In Hybrid an
            // energy mask can bring this about, because it can raise the
            // SILK rate above the total (C 2105). A CELT-only packet has a
            // bitrate of at least 500 b/s, and CELT codes the same frames at
            // 500 and 501 b/s.
            let celt_bitrate = if self.use_cbr {
                OPUS_BITRATE_MAX
            } else if mode == OpusMode::Hybrid {
                (self.packet_bitrate_bps - hybrid_silk_rate).max(501)
            } else {
                self.packet_bitrate_bps.max(501)
            };
            // The value is accepted: it is `OPUS_BITRATE_MAX` or above 500.
            let _ = self.celt_enc.set_bitrate(celt_bitrate);
            self.celt_enc.set_vbr(!self.use_cbr);
            // C 2453-2462: Hybrid VBR is unconstrained (CELT can take bits
            // SILK left), CELT-only VBR follows the setting.
            self.celt_enc
                .set_constrained_vbr(mode == OpusMode::CeltOnly && self.vbr_constraint);
            // C `CELT_SET_START_BAND`; 0 and 17 are both valid.
            let _ = self.celt_enc.set_start_band(start_band);

            // C 2478-2486: the CELT state of another mode is of no use, and
            // the decoder resets its CELT state at the same place. The
            // encoder starts again, codes the 2.5 ms before the frame into
            // 2 bytes that are thrown away ("Prefilling"), and codes the
            // frame without prediction from them.
            if celt_restart {
                self.celt_enc.reset_state();
                let mut prefill_rc = RangeCoder::new_encoder(2);
                self.celt_enc
                    .encode(&self.celt_prefill[..f2_5 * ch], f2_5, &mut prefill_rc);
                let _ = self.celt_enc.set_prediction(0);
            }

            // C 2493: CELT codes the start of `pcm_buf`. The range coder
            // holds `nb_compr_bytes`.
            if fits {
                self.celt_enc
                    .encode(&self.buf_celt_pcm[..frame_size * ch], frame_size, &mut self.rc);
            }
        }

        self.rc.done();

        // C 2555-2560: every coded frame leaves its mode behind for the
        // next one. After a frame that announced the switch to CELT-only,
        // that mode is CELT-only.
        self.prev_enc_mode = Some(if d.to_celt { OpusMode::CeltOnly } else { mode });
        self.prev_channels = sch;
        self.prev_frame_size = frame_size;
        // C 2562: from here on a frame is coded. The packets too small to
        // code anything leave `first` alone.
        self.first = false;

        // C 2582-2594: "In the unlikely case that the SILK encoder busted
        // its target, tell the decoder to call the PLC". The packet holds
        // one frame of one zero byte, which a decoder conceals. With FEC the
        // LBRR section can squeeze the main frame below what even the
        // damage-control re-encode fits, which is the case this exists for.
        if !fits || self.rc.error != 0 || self.rc.tell() > 8 * nb_compr_bytes as i32 {
            output[0] = toc;
            output[1] = 0;
            // No redundant frame goes with it, in libopus neither: a frame
            // this full has no room for the 17 bits (37 for Hybrid) that
            // the redundancy needs (C 2351), so `redundancy` is off and
            // C 2379-2383 clear `silk_bw_switch`.
            self.silk_bw_switch = false;
            if d.to_celt {
                self.celt_enc.reset_state();
            }
            let mut ret = 2usize;
            if self.use_cbr {
                packet::pad(output, ret, orig_max_data_bytes)?;
                ret = orig_max_data_bytes;
            }
            return Ok(ret);
        }

        // C 2509-2511: the state of the range coder once the frame is
        // coded. `done` above does not change it.
        self.range_final = self.rc.rng;

        if mode == OpusMode::SilkOnly {
            if !redundancy {
                // C 2595-2603: "it's perfectly reasonable to strip off
                // trailing zero bytes as the required range decoder
                // behavior is to fill these in". Not with a redundant
                // frame behind: the decoder finds that by the length.
                while nb_compr_bytes > 2 && self.rc.buf[nb_compr_bytes - 1] == 0 {
                    nb_compr_bytes -= 1;
                }
            }
        } else {
            // CELT shrinks the range coder to what a variable bitrate used.
            nb_compr_bytes = (self.rc.storage as usize).min(nb_compr_bytes);
        }

        // C 2513-2538: "5 ms redundant frame for SILK->CELT". It starts
        // from a fresh CELT state, as the decoder does, and codes the last
        // 5 ms of the frame. The CELT-only packets that follow continue
        // from it. In front of it, the 2.5 ms before those 5 ms are coded
        // into 2 bytes that are thrown away, to fill the CELT encoder's
        // overlap and pre-emphasis memories ("NOTE: We could speed this up
        // slightly ... by just adding a function that prefills the
        // buffer"). Both frames are coded without prediction; the next
        // CELT-only frame sets it back to 2 (C 2288-2294).
        //
        // A frame that announces the switch without a redundant frame (the
        // packet had no room) resets the CELT encoder all the same. The
        // decoder starts the CELT-only packets from the reset state then,
        // and libopus's encoder, which keeps the state of its last CELT
        // frame, predicts from energies the decoder does not have.
        if redundancy && !celt_to_silk {
            self.celt_enc.reset_state();
            let _ = self.celt_enc.set_prediction(0);
            // Frames that announce a switch to CELT-only are 10 ms or
            // longer, so the 7.5 ms lie inside the frame.
            let warm_up = (frame_size - f5 - f2_5) * ch..(frame_size - f5) * ch;
            let mut dummy = [0u8; 2];
            // The 2 bytes are thrown away, and so is any error in them.
            let _ = encode_celt_redundancy(
                state_mut(&mut self.celt_enc),
                &self.buf_celt_pcm[warm_up],
                f2_5,
                &mut dummy,
            );
            redundant_rng = encode_celt_redundancy(
                state_mut(&mut self.celt_enc),
                &self.buf_celt_pcm[(frame_size - f5) * ch..frame_size * ch],
                f5,
                &mut redundant[..redundancy_bytes],
            )?;
        } else if d.to_celt {
            self.celt_enc.reset_state();
        }

        // C 2543-2546: the TOC byte, the main frame, the redundant frame.
        output[0] = toc;
        output[1..1 + nb_compr_bytes].copy_from_slice(&self.rc.buf[..nb_compr_bytes]);
        let mut ret = 1 + nb_compr_bytes;
        output[ret..ret + redundancy_bytes].copy_from_slice(&redundant[..redundancy_bytes]);
        ret += redundancy_bytes;
        self.range_final ^= redundant_rng;

        // C 2643-2651: a constant bitrate pads the packet to its full size.
        if self.use_cbr {
            packet::pad(output, ret, orig_max_data_bytes)?;
            ret = orig_max_data_bytes;
        }
        Ok(ret)
    }

    /// Write a packet that codes nothing: only its first byte, and a frame
    /// count when it holds three to six 20 ms frames (60 to 120 ms of CELT
    /// or Hybrid, 100 ms of SILK). A decoder conceals such a
    /// packet like a lost one. This is libopus `opus_encode_native`, "If the
    /// space is too low to do something useful, emit 'PLC' frames"
    /// (opus_encoder.c:1344-1405).
    ///
    /// The first byte names the mode and bandwidth of the last packet that
    /// was coded (C `st->mode`, `st->bandwidth`). With a constant bitrate
    /// the packet is padded to `max_data_bytes`, as `opus_packet_pad` does.
    /// Returns the packet length. No encoder state changes.
    fn write_placeholder(
        &self,
        frame_rate: i32,
        max_data_bytes: usize,
        output: &mut [u8],
    ) -> Result<usize, &'static str> {
        let mut toc_mode = self.mode;
        let mut bw = self.bandwidth;
        let mut frame_rate = frame_rate;
        let mut packet_code = 0u8;
        let mut num_multiframes = 0u8;
        if frame_rate > 100 {
            toc_mode = OpusMode::CeltOnly;
        }
        // 40 ms -> 2 x 20 ms if in CELT_ONLY or HYBRID mode.
        if frame_rate == 25 && toc_mode != OpusMode::SilkOnly {
            frame_rate = 50;
            packet_code = 1;
        }
        // ">= 60 ms frames": frame rates 16 (60 ms), 12 (80 ms), 10 (100 ms)
        // and 8 (120 ms).
        if frame_rate <= 16 {
            // "1 x 60 ms, 2 x 40 ms, 2 x 60 ms": SILK frames, as one frame or
            // two of equal size (code 1), which the first byte alone can
            // say. So can a buffer of one byte, whatever the mode. 100 ms
            // has no such form and never gets here with one byte
            // (`encode_impl` refuses it).
            if output.len() == 1 || (toc_mode == OpusMode::SilkOnly && frame_rate != 10) {
                toc_mode = OpusMode::SilkOnly;
                packet_code = (frame_rate <= 12) as u8;
                frame_rate = if frame_rate == 12 { 25 } else { 16 };
            } else {
                // 3, 4, 5 or 6 frames of 20 ms (code 3).
                num_multiframes = (50 / frame_rate) as u8;
                frame_rate = 50;
                packet_code = 3;
            }
        }
        let index = bandwidth_fec_index(bw);
        if toc_mode == OpusMode::SilkOnly && index > 2 {
            bw = Bandwidth::Wideband;
        } else if toc_mode == OpusMode::CeltOnly && bw == Bandwidth::Mediumband {
            bw = Bandwidth::Narrowband;
        } else if toc_mode == OpusMode::Hybrid && index <= 3 {
            bw = Bandwidth::Superwideband;
        }
        output[0] = gen_toc(toc_mode, frame_rate, bw, self.stream_channels) | packet_code;
        let mut ret = if packet_code <= 1 { 1 } else { 2 };
        let max_data_bytes = max_data_bytes.max(ret);
        if packet_code == 3 {
            output[1] = num_multiframes;
        }
        if self.use_cbr {
            packet::pad(output, ret, max_data_bytes)?;
            ret = max_data_bytes;
        }
        Ok(ret)
    }

    /// A packet made of several frames (libopus `opus_encode_native`,
    /// opus_encoder.c:1697-1838):
    ///
    /// - CELT-only and Hybrid frames are 20 ms at most, so a packet of 40
    ///   to 120 ms is made of 2 to 6 frames of 20 ms.
    /// - SILK-only frames are 60 ms at most. 80 ms is coded as 2 frames of
    ///   40 ms, 100 ms as 5 frames of 20 ms, 120 ms as 2 frames of 60 ms.
    ///
    /// Every frame is coded with the packet's `decisions`, so the packet
    /// has one configuration. The frames are joined into one packet with
    /// frame count code 1, 2 or 3, as libopus's repacketizer does.
    ///
    /// `cbr_bytes` is the packet size a constant bitrate gives.
    fn encode_multiframe(
        &mut self,
        input: &[f32],
        frame_size: usize,
        decisions: FrameDecisions,
        cbr_bytes: usize,
        output: &mut [u8],
    ) -> Result<usize, &'static str> {
        // Up to 120 ms of 20 ms frames.
        const MAX_SUB: usize = 6;
        let ch = self.channels;
        let fs = self.sampling_rate;
        let fs_u = fs as usize;
        // C 1712-1722: the duration of each frame.
        let enc_frame = if decisions.mode == OpusMode::SilkOnly {
            if frame_size == 2 * fs_u / 25 {
                // 80 ms -> 2 x 40 ms
                fs_u / 25
            } else if frame_size == 3 * fs_u / 25 {
                // 120 ms -> 2 x 60 ms
                3 * fs_u / 50
            } else {
                // 100 ms -> 5 x 20 ms
                fs_u / 50
            }
        } else {
            fs_u / 50
        };
        if enc_frame == 0 || !frame_size.is_multiple_of(enc_frame) {
            return Err("Invalid multi-frame size");
        }
        let nb = frame_size / enc_frame;
        if !(2..=MAX_SUB).contains(&nb) {
            return Err("Unsupported multi-frame size");
        }

        // C 1735-1795: how large the packet and each of its frames may be.
        //
        // - The packet: the caller's buffer, or under CBR the constant size.
        // - A frame: its share of the packet once the worst-case header is
        //   taken off. Under CBR also never more than the bitrate gives
        //   one frame.
        //
        // Not ported on purpose: libopus 1.6 applies the "never more than
        // the bitrate gives one frame" limit under VBR too (`curr_max`,
        // opus_encoder.c:1785). That turns VBR into a hard cap per frame and
        // costs libopus itself up to 23 dB when in-band FEC is on (measured:
        // 24 kHz stereo Hybrid, 128 kb/s, 40 ms, FEC: 3.9 dB against 27.2 dB
        // for its own 20 ms packets). libopus 1.5 and earlier had no such
        // limit under VBR, and neither has opus1.
        let max_header_bytes = if nb == 2 { 3 } else { 2 + (nb - 1) * 2 };
        // C 1749: VBR, and CBR at the maximum bitrate, may fill the buffer.
        let repacketize_len = if self.use_cbr && self.user_bitrate != Bitrate::Max {
            cbr_bytes.min(output.len())
        } else {
            output.len()
        };
        let max_len_sum = (nb + repacketize_len).saturating_sub(max_header_bytes);
        let nominal_part =
            (bitrate_to_bits(self.packet_bitrate_bps, fs, enc_frame) / 8).max(0) as usize;

        let mut sub = [[0u8; OPUS_MAX_PACKET_BYTES]; MAX_SUB];
        let mut lens = [0usize; MAX_SUB];
        let mut tot_size = 0usize;
        let mut result = Ok(());
        // C 1762-1767. A packet that leads over to mono is not coded so: its
        // frames code two channels without SILK's fade, and the
        // caller's channel setting becomes mono, so that the next packet is
        // mono at once. The setting stays (a libopus behavior, ported as it
        // is).
        let bak_to_mono = self.silk_to_mono;
        if bak_to_mono {
            self.force_channels = Some(1);
        } else {
            self.prev_channels = self.stream_channels;
        }
        for i in 0..nb {
            let start = i * enc_frame * ch;
            let end = start + enc_frame * ch;
            let sub_input = &input[start..end];
            self.silk_to_mono = false;
            self.nonfinal_frame = i + 1 < nb;
            // C 1771-1773: "When switching from SILK/Hybrid to CELT, only
            // ask for a switch at the last frame". A redundant frame for
            // leaving CELT-only goes with the first frame.
            let frame_to_celt = decisions.to_celt && i == nb - 1;
            let frame_decisions = FrameDecisions {
                to_celt: frame_to_celt,
                redundancy: decisions.redundancy
                    && (frame_to_celt || (!decisions.to_celt && i == 0)),
                ..decisions
            };
            let mut curr_max = (max_len_sum / nb).min(max_len_sum.saturating_sub(tot_size));
            if self.use_cbr {
                curr_max = curr_max.min(nominal_part);
            }
            let curr_max = curr_max.min(OPUS_MAX_PACKET_BYTES);
            match self.encode_frame(sub_input, enc_frame, &mut sub[i][..curr_max], frame_decisions)
            {
                Ok(n) => {
                    lens[i] = n;
                    tot_size += n;
                }
                Err(e) => {
                    result = Err(e);
                    break;
                }
            }
        }
        self.nonfinal_frame = false;
        self.silk_to_mono = bak_to_mono;
        result?;

        // C 1826: the frames go into one packet (libopus's repacketizer).
        // A constant bitrate pads it to its full size.
        let mut frames: [&[u8]; MAX_SUB] = [&[]; MAX_SUB];
        for ((frame, part), &len) in frames.iter_mut().zip(&sub).zip(&lens).take(nb) {
            let parsed = packet::parse(&part[..len], false)?;
            if parsed.count != 1 {
                return Err("Unexpected frame count in a part of a packet");
            }
            *frame = parsed.frame(&part[..len], 0);
        }
        packet::write(
            sub[0][0],
            &frames[..nb],
            &mut output[..repacketize_len],
            false,
            self.use_cbr,
        )
    }
}

use celt::PcmLayout;

pub struct OpusDecoder {
    #[cfg(not(feature = "heap"))]
    celt_dec: CeltDecoder,
    #[cfg(feature = "heap")]
    celt_dec: Box<CeltDecoder>,
    #[cfg(not(feature = "heap"))]
    silk_dec: silk::dec_api::SilkDecoder,
    #[cfg(feature = "heap")]
    silk_dec: Box<silk::dec_api::SilkDecoder>,
    sampling_rate: i32,
    channels: usize,

    /// C `st->mode`: the mode of the last packet that was decoded, lost
    /// frames not counted. `None` before the first packet.
    mode: Option<OpusMode>,
    /// C `st->prev_mode`: the mode the last frame was decoded or concealed
    /// in.
    prev_mode: Option<OpusMode>,

    /// Whether the previous frame had redundancy (mode transition marker).
    prev_redundancy: bool,
    /// C `st->frame_size`: samples per channel in one frame of the last
    /// packet.
    frame_size: usize,

    /// C `st->stream_channels`: the channel count the last packet codes,
    /// from its first byte. It may differ from `channels`. The CELT layer
    /// converts by itself once it is told (`CELT_SET_CHANNELS`).
    stream_channels: usize,

    /// C `st->rangeFinal`: the state of the range decoder after the last
    /// frame, 0 after a lost frame.
    range_final: u32,
    /// C `st->softclip_mem`: what the soft clipper of `decode_i16` carries
    /// from one packet to the next, per channel.
    softclip_mem: [f32; OPUS_MAX_CHANNELS],

    /// C `st->DecControl.nChannelsInternal`: the channel count of the last
    /// SILK frame that carried data. A lost frame is concealed with it,
    /// whatever the packets in between said. 0 before the first SILK frame.
    silk_channels_internal: usize,
    /// C `st->DecControl.internalSampleRate`: the SILK rate in Hz of the
    /// last SILK frame that carried data, kept for concealment in the same
    /// way. 0 before the first SILK frame.
    silk_internal_rate: i32,

    #[cfg(not(feature = "heap"))]
    w_pcm_i16: FixedVec<i16, OPUS_PCM_I16>,
    #[cfg(feature = "heap")]
    w_pcm_i16: Box<FixedVec<i16, OPUS_PCM_I16>>,
    /// SILK output of one concealed piece, interleaved. SILK conceals at
    /// least 10 ms, also for a shorter piece, and at most 20 ms.
    #[cfg(not(feature = "heap"))]
    w_silk_out: FixedVec<f32, OPUS_PCM_I16>,
    #[cfg(feature = "heap")]
    w_silk_out: Box<FixedVec<f32, OPUS_PCM_I16>>,
    /// Float output of one packet for `decode_i16`. It is allocated in
    /// `new`, so that no decode call allocates, and filled at first use.
    /// `None` while a call uses it.
    #[cfg(feature = "heap")]
    w_pcm_float: Option<Box<FixedVec<f32, OPUS_PCM_PACKET>>>,
}

impl OpusDecoder {
    pub fn new(sampling_rate: i32, channels: usize) -> Result<Self, &'static str> {
        if ![8000, 12000, 16000, 24000, 48000].contains(&sampling_rate) {
            return Err("Invalid sampling rate");
        }
        if ![1, 2].contains(&channels) {
            return Err("Invalid number of channels");
        }

        let mode = modes::default_mode();
        #[cfg(feature = "heap")]
        let celt_dec = Box::new(CeltDecoder::new(mode, channels, sampling_rate));
        #[cfg(not(feature = "heap"))]
        let celt_dec = CeltDecoder::new(mode, channels, sampling_rate);

        // C `silk_InitDecoder`. The SILK decoder learns its rates and
        // channel counts with every frame (`SilkDecControl`).
        #[cfg(feature = "heap")]
        let silk_dec = Box::new(silk::dec_api::SilkDecoder::new());
        #[cfg(not(feature = "heap"))]
        let silk_dec = silk::dec_api::SilkDecoder::new();

        Ok(Self {
            celt_dec,
            silk_dec,
            sampling_rate,
            channels,
            mode: None,
            prev_mode: None,
            prev_redundancy: false,
            // C `opus_decoder_init`: `st->frame_size = Fs/400`.
            frame_size: sampling_rate as usize / 400,
            stream_channels: channels,
            range_final: 0,
            softclip_mem: [0.0; OPUS_MAX_CHANNELS],
            silk_channels_internal: 0,
            silk_internal_rate: 0,

            #[cfg(not(feature = "heap"))]
            w_pcm_i16: FixedVec::from_value(0i16, 960 * channels),
            #[cfg(feature = "heap")]
            w_pcm_i16: Box::new(FixedVec::from_value(0i16, 960 * channels)),

            #[cfg(not(feature = "heap"))]
            w_silk_out: FixedVec::from_value(0.0f32, 960 * channels),
            #[cfg(feature = "heap")]
            w_silk_out: Box::new(FixedVec::from_value(0.0f32, 960 * channels)),
            #[cfg(feature = "heap")]
            w_pcm_float: Some(Box::default()),
        })
    }

    /// Decode one Opus packet into interleaved `f32` PCM, returning the
    /// number of samples per channel written.
    ///
    /// `frame_size` is the room in `output`, in samples per channel. The
    /// packet says how many samples it holds. Only the first
    /// `samples * channels` values of `output` are written, where `samples`
    /// is the result; the rest of `output` is left as it was, as in libopus.
    ///
    /// The packet may code another channel count than the decoder has, as
    /// in libopus. A mono packet is written to both channels of a stereo
    /// decoder. A stereo packet is mixed down by a mono decoder. The count
    /// may change from one packet to the next.
    ///
    /// The output is not clipped: like libopus `opus_decode_float`, a loud
    /// signal comes back with peaks beyond ±1.0. Use
    /// [`decode_i16`](Self::decode_i16) or [`pcm_soft_clip`] where the
    /// samples must stay in range.
    ///
    /// Packet loss follows libopus: pass an empty packet (`&[]`) to conceal
    /// `frame_size` samples (a multiple of 2.5 ms), or a ToC-only 1-byte
    /// packet to conceal the duration that ToC describes. Either way the
    /// frame is concealed in the previous packet's mode; the placeholder's
    /// mode, bandwidth and channel bits are ignored. A 1-byte packet whose
    /// frame count code is 2 or 3 lacks the bytes that code needs and is
    /// invalid, as in libopus.
    ///
    /// Inside a longer packet, a frame of 0 or 1 bytes stands for a lost
    /// frame and is concealed in the same way.
    pub fn decode(
        &mut self,
        input: &[u8],
        frame_size: usize,
        output: &mut [f32],
    ) -> Result<usize, &'static str> {
        let (samples, _) = self.decode_native(input, frame_size, output, false, false, false)?;
        Ok(samples)
    }

    /// Rebuild a lost packet from the in-band forward error correction
    /// (FEC) data of the packet that follows it: libopus `opus_decode_float`
    /// with `decode_fec = 1`.
    ///
    /// `data` is the packet received after the lost one. `frame_size` is the
    /// duration that was lost, in samples per channel, a multiple of 2.5 ms.
    /// Exactly that many samples are written. Decode `data` itself with
    /// [`decode`](Self::decode) afterwards.
    ///
    /// A SILK-only or Hybrid packet can carry a low-bitrate copy of the
    /// frame before it. That copy fills the end of the lost duration, and
    /// anything before it is concealed. The whole duration is concealed
    /// when `data` cannot hold such a copy (it is CELT-only, the packet
    /// before the loss was CELT-only, or `frame_size` is shorter than a
    /// frame of `data`) or holds none. In a Hybrid stream the copy covers
    /// the band below 8 kHz; the band above is concealed.
    pub fn decode_fec(
        &mut self,
        data: &[u8],
        frame_size: usize,
        output: &mut [f32],
    ) -> Result<usize, &'static str> {
        self.decode_native(data, frame_size, output, true, false, false)
            .map(|(samples, _)| samples)
    }

    /// Decode one Opus packet into interleaved 16-bit PCM: libopus
    /// `opus_decode` in the float build.
    ///
    /// The packet is decoded to floating point and soft-clipped, so that
    /// peaks beyond full scale are rounded off instead of cut
    /// ([`pcm_soft_clip`]). The result is rounded to 16 bits. A lost packet
    /// is given as in [`decode`](Self::decode).
    pub fn decode_i16(
        &mut self,
        input: &[u8],
        frame_size: usize,
        output: &mut [i16],
    ) -> Result<usize, &'static str> {
        self.decode_i16_native(input, frame_size, output, false)
    }

    /// [`decode_fec`](Self::decode_fec) with 16-bit output: libopus
    /// `opus_decode` with `decode_fec = 1`.
    pub fn decode_fec_i16(
        &mut self,
        data: &[u8],
        frame_size: usize,
        output: &mut [i16],
    ) -> Result<usize, &'static str> {
        self.decode_i16_native(data, frame_size, output, true)
    }

    /// libopus `opus_decode` (opus_decoder.c, float build): decode into a
    /// float buffer with the soft clipper on, then round to 16 bits.
    ///
    /// The soft clipper needs the whole packet at once, so the float buffer
    /// holds 120 ms, the longest packet. A longer loss is concealed 60 ms at
    /// a time, which gives the same samples as concealing it in one call. At least 60 ms stay for the last call,
    /// enough for the longest frame an FEC copy can stand for.
    fn decode_i16_native(
        &mut self,
        data: &[u8],
        frame_size: usize,
        output: &mut [i16],
        decode_fec: bool,
    ) -> Result<usize, &'static str> {
        let channels = self.channels;
        let lost = decode_fec || data.is_empty();
        let max_samples = packet::max_packet_samples(self.sampling_rate);
        let step = max_samples / 2;
        if lost && frame_size > max_samples {
            // `decode_native` checks each call, but the whole loss must be
            // checked before any of it is concealed. libopus also reads the
            // packet before it conceals anything, so an invalid packet is
            // an error that leaves the decoder as it was.
            if !is_whole_2_5ms(frame_size, self.sampling_rate) {
                return Err("Lost frame: frame_size must be a multiple of 2.5 ms");
            }
            // A product that overflows cannot fit any buffer.
            if frame_size.checked_mul(channels).is_none_or(|len| len > output.len()) {
                return Err("Output buffer too small for packet");
            }
            if !data.is_empty() {
                packet::parse(data, false)?;
            }
        }

        // The buffer leaves the decoder during the call, because the decoder
        // is borrowed while it fills the buffer. Without `heap` it is an
        // uninitialized array on the stack; only the part a call fills is
        // touched.
        #[cfg(feature = "heap")]
        let mut pcm = self.w_pcm_float.take().unwrap_or_default();
        #[cfg(not(feature = "heap"))]
        let mut pcm: FixedVec<f32, OPUS_PCM_PACKET> = FixedVec::new();
        let mut done = 0;
        let result = loop {
            let last = !(lost && frame_size - done > max_samples);
            let (data, piece, decode_fec) = if last {
                (data, frame_size - done, decode_fec)
            } else {
                (&[][..], step, false)
            };
            let output = &mut output[done * channels..];
            // The float buffer gets the length of `output`, so that
            // `decode_native` checks the caller's buffer. It need not be
            // longer than the call or the packet.
            let needed = if lost {
                piece
            } else {
                match packet::nb_samples(data, self.sampling_rate) {
                    Ok(samples) => piece.min(samples),
                    Err(e) => break Err(e),
                }
            };
            let len = output.len().min(needed * channels);
            if pcm.len() < len {
                pcm.resize(len, 0.0);
            }
            match self.decode_native(data, piece, &mut pcm[..len], decode_fec, false, true) {
                Ok((samples, _)) => {
                    for (o, &x) in output.iter_mut().zip(&pcm[..samples * channels]) {
                        *o = float2int16(x);
                    }
                    done += samples;
                }
                Err(e) => break Err(e),
            }
            if last {
                break Ok(done);
            }
        };
        #[cfg(feature = "heap")]
        {
            self.w_pcm_float = Some(pcm);
        }
        result
    }

    /// Decode a packet that may be followed by other data: libopus
    /// `opus_decode_native` (opus_decoder.c).
    ///
    /// - `data` empty: the packet is lost. `frame_size` samples are
    ///   concealed.
    /// - `decode_fec`: `data` is the packet after a lost one, see
    ///   [`decode_fec`](Self::decode_fec).
    /// - `self_delimited`: `data` starts with a self-delimited packet
    ///   (RFC 6716 appendix B), as every stream but the last of a
    ///   multistream packet does.
    /// - `soft_clip`: a decoded packet is soft-clipped with this decoder's
    ///   memory ([`pcm_soft_clip`]), as libopus does for 16-bit output.
    ///   Concealed and FEC output is left as it is, as in libopus. The
    ///   caller converts to 16 bits with [`float2int16`].
    ///
    /// Returns the samples per channel written to `output`, and the number
    /// of bytes of `data` the packet takes (0 for a lost packet).
    pub(crate) fn decode_native(
        &mut self,
        data: &[u8],
        frame_size: usize,
        output: &mut [f32],
        decode_fec: bool,
        self_delimited: bool,
        soft_clip: bool,
    ) -> Result<(usize, usize), &'static str> {
        let channels = self.channels;
        if (decode_fec || data.is_empty()) && !is_whole_2_5ms(frame_size, self.sampling_rate) {
            return Err("Lost frame: frame_size must be a multiple of 2.5 ms");
        }
        // A lost duration so long that `frame_size * channels` overflows
        // cannot fit any buffer.
        let lost_len = frame_size.checked_mul(channels);
        if data.is_empty() {
            let lost_len = lost_len
                .filter(|&len| len <= output.len())
                .ok_or("Output buffer too small for packet")?;
            self.conceal(frame_size, &mut output[..lost_len])?;
            return Ok((frame_size, 0));
        }

        let toc = data[0];
        let packet_mode = mode_from_toc(toc);
        let packet_bandwidth = bandwidth_from_toc(toc);
        let packet_frame_size = packet::samples_per_frame(toc, self.sampling_rate);
        let packet_channels = channels_from_toc(toc);
        let parsed = packet::parse(data, self_delimited)?;

        if decode_fec {
            if lost_len.is_none_or(|len| len > output.len()) {
                return Err("Output buffer too small for packet");
            }
            // If no FEC can be present, run the loss concealment.
            if frame_size < packet_frame_size
                || packet_mode == OpusMode::CeltOnly
                || self.mode == Some(OpusMode::CeltOnly)
            {
                self.conceal(frame_size, &mut output[..frame_size * channels])?;
                return Ok((frame_size, parsed.packet_offset));
            }
            let frame = parsed.frame(data, 0);
            // Otherwise, conceal everything except the duration the FEC
            // data stands for.
            let lead = frame_size - packet_frame_size;
            if lead != 0 {
                self.conceal(lead, &mut output[..lead * channels])?;
            }
            // Complete with FEC.
            self.mode = Some(packet_mode);
            self.frame_size = packet_frame_size;
            self.stream_channels = packet_channels;
            self.decode_or_conceal_frame(
                packet_mode,
                packet_bandwidth,
                frame,
                packet_frame_size,
                &mut output[lead * channels..frame_size * channels],
                true,
            )?;
            return Ok((frame_size, parsed.packet_offset));
        }

        // The duration comes from the ToC, not from the caller's
        // `frame_size`, which only says how much room there is.
        let decoded_total = parsed.count * packet_frame_size;
        if decoded_total > frame_size {
            return Err("frame_size too small for packet");
        }
        if output.len() < decoded_total * channels {
            return Err("Output buffer too small for packet");
        }
        // Update the state as the last step, so that an invalid packet
        // leaves it alone.
        self.mode = Some(packet_mode);
        self.frame_size = packet_frame_size;
        self.stream_channels = packet_channels;

        // Each frame of the packet goes through the same steps as libopus
        // `opus_decode_frame`.
        let frame_len = packet_frame_size * channels;
        for (frame, out) in parsed.frames(data).zip(output.chunks_exact_mut(frame_len)) {
            self.decode_or_conceal_frame(
                packet_mode,
                packet_bandwidth,
                frame,
                packet_frame_size,
                out,
                false,
            )?;
        }
        // C: the soft clipper runs on a decoded packet only. Concealed and
        // FEC output returns above, without it, and leaves its memory alone.
        if soft_clip {
            pcm_soft_clip(
                &mut output[..decoded_total * channels],
                decoded_total,
                channels,
                &mut self.softclip_mem,
            );
        } else {
            self.softclip_mem = [0.0; OPUS_MAX_CHANNELS];
        }
        Ok((decoded_total, parsed.packet_offset))
    }

    /// One frame of a packet: decode it, or conceal it when it carries no
    /// data (libopus `opus_decode_frame`: "Payloads of 1 (2 including ToC)
    /// or 0 trigger the PLC/DTX").
    fn decode_or_conceal_frame(
        &mut self,
        mode: OpusMode,
        bandwidth: Bandwidth,
        payload: &[u8],
        frame_size: usize,
        out: &mut [f32],
        decode_fec: bool,
    ) -> Result<(), &'static str> {
        if payload.len() <= 1 {
            self.conceal(frame_size, out)
        } else {
            self.decode_frame(mode, bandwidth, payload, frame_size, out, decode_fec)
        }
    }

    /// libopus `OPUS_GET_FINAL_RANGE`: the state of the range decoder after
    /// the last frame, or 0 when that frame was lost. For a packet that
    /// encoder and decoder read alike it equals the encoder's
    /// [`OpusEncoder::final_range`].
    ///
    /// After [`decode_fec`](Self::decode_fec) of a Hybrid packet the value
    /// is the random state of the CELT loss concealment, as in libopus: the
    /// CELT layer of the recovered frame is concealed, and libopus reports
    /// what the CELT decoder ends with.
    pub fn final_range(&self) -> u32 {
        self.range_final
    }

    /// libopus `OPUS_RESET_STATE`: forget the stream decoded so far. The
    /// decoder then behaves like a new one with the same sampling rate and
    /// channel count.
    pub fn reset_state(&mut self) {
        // C opus_decoder_ctl, OPUS_RESET_STATE: clear the stream state,
        // reset both layers, and put back the two values that do not start
        // at zero. `st->DecControl` is not part of what libopus clears, so
        // the SILK channel count and rate of the last frame stay. They are
        // not used before the next SILK frame sets them again.
        self.celt_dec.reset_state();
        self.silk_dec.reset();
        self.mode = None;
        self.prev_mode = None;
        self.prev_redundancy = false;
        self.frame_size = self.sampling_rate as usize / 400;
        self.stream_channels = self.channels;
        self.range_final = 0;
        self.softclip_mem = [0.0; OPUS_MAX_CHANNELS];
    }

    /// The sampling rate of the output in Hz (`OPUS_GET_SAMPLE_RATE`).
    pub fn sampling_rate(&self) -> i32 {
        self.sampling_rate
    }

    /// The number of output channels.
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// Decode one frame of a packet: a port of libopus `opus_decode_frame`
    /// for a frame that carries data.
    ///
    /// Besides the SILK and CELT layers this handles what happens when the
    /// coding mode changes inside a stream:
    ///
    /// - A SILK-only or Hybrid frame can end with a 5 ms CELT frame that
    ///   covers the whole band (the "redundancy" frame). The encoder adds it
    ///   next to a switch to or from CELT-only, and the decoder cross-fades
    ///   it with the main frame so the switch is not audible.
    /// - A switch without such a frame is bridged with 5 ms of concealment
    ///   in the previous mode, cross-faded into the new frame.
    /// - A switch between SILK-only and Hybrid needs neither.
    ///
    /// With `decode_fec` the frame is not decoded itself. Its SILK layer
    /// yields the redundant copy it carries of the frame before it (or
    /// conceals, when it carries none), and the CELT layer of a Hybrid frame
    /// is concealed. A redundancy frame is not read.
    fn decode_frame(
        &mut self,
        mode: OpusMode,
        bandwidth: Bandwidth,
        payload: &[u8],
        frame_size: usize,
        out: &mut [f32],
        decode_fec: bool,
    ) -> Result<(), &'static str> {
        let channels = self.channels;
        let f2_5 = self.sampling_rate as usize / 400;
        let f5 = 2 * f2_5;
        let frame_len = frame_size * channels;
        let prev_mode = self.prev_mode;
        let mut len = payload.len();
        let mut rc = RangeCoder::new_decoder(payload);

        // A switch to or from CELT-only needs a cross-fade, unless the
        // previous frame ended with a redundancy frame.
        let mut transition = match prev_mode {
            Some(prev) => {
                (mode == OpusMode::CeltOnly
                    && prev != OpusMode::CeltOnly
                    && !self.prev_redundancy)
                    || (mode != OpusMode::CeltOnly && prev == OpusMode::CeltOnly)
            }
            None => false,
        };
        // 5 ms of concealment for the cross-fade. The buffer is filled only
        // on a switch that needs it; every other frame leaves it untouched.
        let mut pcm_transition: FixedVec<f32, OPUS_PCM_TAIL> = FixedVec::new();
        let transition_len = f5.min(frame_size);
        if transition && mode == OpusMode::CeltOnly {
            pcm_transition.resize(f5 * channels, 0.0);
            // 5 ms of concealment in the previous (SILK or Hybrid) mode.
            self.conceal(transition_len, &mut pcm_transition[..transition_len * channels])?;
        }

        // SILK layer (C opus_decoder.c:402-497).
        if mode != OpusMode::CeltOnly {
            if prev_mode == Some(OpusMode::CeltOnly) {
                // libopus `silk_ResetDecoder`: the SILK state is stale after
                // CELT-only frames. Its resamplers start afresh as well.
                self.silk_dec.reset();
            }
            // C 424-442: a frame that carries data says how many channels
            // SILK codes and at which rate. Lost frames keep both.
            self.silk_channels_internal = self.stream_channels;
            self.silk_internal_rate = if mode == OpusMode::SilkOnly {
                1000 * bandwidth.silk_khz()
            } else {
                16000
            };
            // C `lost_flag = 2 * decode_fec`: read the frame itself, or the
            // redundant copy (LBRR) of the frame before it.
            let lost_flag = if decode_fec {
                silk::decode_frame::FLAG_DECODE_LBRR
            } else {
                silk::decode_frame::FLAG_DECODE_NORMAL
            };
            self.decode_silk(&mut rc, lost_flag, frame_size, Some(&mut *out))?;
        }

        // Redundancy frame: present if enough bits are left after SILK.
        // A Hybrid frame says so with a flag; in a SILK-only frame any such
        // remainder is one.
        let mut redundancy = false;
        let mut celt_to_silk = false;
        let mut redundancy_bytes = 0usize;
        // C `redundant_rng`: the final range of the redundancy frame.
        let mut redundant_rng = 0u32;
        let hybrid = mode == OpusMode::Hybrid;
        if !decode_fec
            && mode != OpusMode::CeltOnly
            && rc.tell() + 17 + if hybrid { 20 } else { 0 } <= 8 * len as i32
        {
            redundancy = if hybrid {
                rc.decode_bit_logp(12)
            } else {
                true
            };
            if redundancy {
                celt_to_silk = rc.decode_bit_logp(1);
                redundancy_bytes = if hybrid {
                    rc.dec_uint(256) as usize + 2
                } else {
                    len - ((rc.tell() + 7) >> 3) as usize
                };
                if redundancy_bytes > len || ((len - redundancy_bytes) * 8) < rc.tell() as usize {
                    // Not possible in a valid packet; ignore the frame.
                    len = 0;
                    redundancy_bytes = 0;
                    redundancy = false;
                } else {
                    len -= redundancy_bytes;
                    // The CELT layer reads its raw bits from the end of the
                    // main frame, not from the end of the redundancy frame.
                    rc.storage -= redundancy_bytes as u32;
                }
            }
        }
        let start_band = if mode != OpusMode::CeltOnly { HYBRID_START_BAND } else { 0 };
        if redundancy {
            transition = false;
        }
        if transition && mode != OpusMode::CeltOnly {
            pcm_transition.resize(f5 * channels, 0.0);
            // 5 ms of concealment in the previous (CELT-only) mode.
            self.conceal(transition_len, &mut pcm_transition[..transition_len * channels])?;
        }

        // C 556-581, `CELT_SET_END_BAND` and `CELT_SET_CHANNELS`: the CELT
        // frames of this Opus frame code the bands of the packet's
        // bandwidth, and as many channels as the packet says. That holds
        // for the main frame, a redundancy frame and the fade-out frame
        // alike, and for the concealment of an FEC decode. The CELT decoder
        // converts to the output channel count. Each CELT call below sets
        // the first band itself (`CELT_SET_START_BAND`).
        self.celt_dec.set_end_band(bandwidth.celt_end_band())?;
        self.celt_dec.set_stream_channels(self.stream_channels)?;
        // The decoded redundancy frame. Filled only when the frame has one.
        let mut redundant_audio: FixedVec<f32, OPUS_PCM_TAIL> = FixedVec::new();
        if redundancy {
            redundant_audio.resize(f5 * channels, 0.0);
        }

        // Redundancy frame for a switch from CELT-only: it continues the
        // CELT state of the previous frame, so it is decoded first. The
        // CELT decoder accepts every redundancy frame a packet can hold. A
        // frame it refused would leave silence.
        if redundancy && celt_to_silk {
            let mut rc_red = RangeCoder::new_decoder(&payload[len..len + redundancy_bytes]);
            let bits = (redundancy_bytes * 8) as i32;
            self.celt_dec.set_start_band(0)?;
            let layout = PcmLayout::Interleaved;
            self.celt_dec.decode(&mut rc_red, bits, f5, &mut redundant_audio, layout);
            redundant_rng = self.celt_dec.final_range();
        }

        // The SILK layer is in `out` already. The CELT layer of a Hybrid
        // frame is added to it (libopus `celt_accum`).
        if mode != OpusMode::SilkOnly {
            // Discard the CELT state of another mode.
            if prev_mode.is_some() && prev_mode != Some(mode) && !self.prev_redundancy {
                self.celt_dec.reset_state();
            }
            self.celt_dec.set_start_band(start_band)?;
            let layout = if hybrid { PcmLayout::InterleavedAdd } else { PcmLayout::Interleaved };
            if decode_fec {
                // libopus hands CELT no data here (`decode_fec ? NULL :
                // data`): the redundant copy has no CELT layer, so the band
                // above 8 kHz is concealed on top of SILK's output.
                //
                // The band range of this packet is set above, so the
                // concealment covers the bands of this packet, not those of
                // the last CELT frame decoded. The two differ when the
                // bandwidth changes across the loss, or when SILK-only
                // packets came before it.
                self.celt_dec.decode_lost(frame_size, out, layout)?;
            } else {
                let bits = (len * 8) as i32;
                let decoded = self.celt_dec.decode(&mut rc, bits, frame_size, out, layout);
                // C: `return celt_ret < 0 ? celt_ret : audiosize`. The CELT
                // decoder refuses a frame it cannot decode at all (a frame
                // size or band range that is not valid), which the packet
                // parser rules out.
                if decoded != frame_size {
                    return Err("CELT decoding failed");
                }
            }
        } else if prev_mode == Some(OpusMode::Hybrid)
            && !(redundancy && celt_to_silk && self.prev_redundancy)
        {
            // After a Hybrid frame the CELT layer still holds half a window
            // of signal. Decoding 2.5 ms of CELT silence on top of the SILK
            // output lets it fade out.
            let mut rc_silence = RangeCoder::new_decoder(&[0xFF, 0xFF]);
            let fade = &mut out[..f2_5 * channels];
            self.celt_dec.set_start_band(0)?;
            self.celt_dec.decode(&mut rc_silence, 16, f2_5, fade, PcmLayout::InterleavedAdd);
        }
        // C 621, 636 `st->rangeFinal`: what the CELT decoder ends with when
        // the frame has a CELT layer, and the range decoder's state after
        // SILK otherwise. For a decoded CELT layer the two are the same
        // value. For an FEC decode of a Hybrid frame the CELT layer is
        // concealed, and the value is the random state of that concealment.
        let frame_rng = if mode != OpusMode::SilkOnly {
            self.celt_dec.final_range()
        } else {
            rc.rng
        };

        let window = modes::default_mode().window;
        let inc = celt::resampling_factor(self.sampling_rate).unwrap_or(1);
        let f2_5_ch = f2_5 * channels;

        // Redundancy frame for a switch to CELT-only: it starts from a
        // fresh CELT state and replaces the end of this frame.
        if redundancy && !celt_to_silk {
            self.celt_dec.reset_state();
            let mut rc_red = RangeCoder::new_decoder(&payload[len..len + redundancy_bytes]);
            let bits = (redundancy_bytes * 8) as i32;
            self.celt_dec.set_start_band(0)?;
            let layout = PcmLayout::Interleaved;
            self.celt_dec.decode(&mut rc_red, bits, f5, &mut redundant_audio, layout);
            redundant_rng = self.celt_dec.final_range();
            let tail = frame_len - f2_5_ch;
            // The end of the main frame fades out.
            let fade = &redundant_audio[f2_5_ch..];
            smooth_fade(fade, &mut out[tail..], false, f2_5, channels, window, inc);
        }
        // Redundancy frame for a switch from CELT-only: it supplies the
        // start of this frame. It is ignored if the previous frame did not
        // use CELT (the first frame of the switch was lost).
        if redundancy
            && celt_to_silk
            && (prev_mode != Some(OpusMode::SilkOnly) || self.prev_redundancy)
        {
            out[..f2_5_ch].copy_from_slice(&redundant_audio[..f2_5_ch]);
            // The main frame fades in.
            let fade = &redundant_audio[f2_5_ch..];
            smooth_fade(fade, &mut out[f2_5_ch..2 * f2_5_ch], true, f2_5, channels, window, inc);
        }
        if transition {
            if frame_size >= f5 {
                out[..f2_5_ch].copy_from_slice(&pcm_transition[..f2_5_ch]);
                let fade = &pcm_transition[f2_5_ch..];
                let main = &mut out[f2_5_ch..2 * f2_5_ch];
                smooth_fade(fade, main, true, f2_5, channels, window, inc);
            } else {
                // A 2.5 ms frame: no room for a clean cross-fade.
                let main = &mut out[..f2_5_ch];
                smooth_fade(&pcm_transition, main, true, f2_5, channels, window, inc);
            }
        }

        // The output is not clipped: libopus `opus_decode_float` does not
        // clip, and `decode_i16` soft-clips the whole packet.

        // C: `if (len <= 1) st->rangeFinal = 0; else st->rangeFinal ^=
        // redundant_rng;`, with `len` the bytes left for the main frame.
        self.range_final = if len <= 1 { 0 } else { frame_rng ^ redundant_rng };

        self.prev_mode = Some(mode);
        self.prev_redundancy = redundancy && !celt_to_silk;
        Ok(())
    }

    /// Fill `output` with `total` samples per channel of concealment in the
    /// mode of the previous frame: what libopus `opus_decode_native` does
    /// for a lost packet, a loop over `opus_decode_frame` without data
    /// (opus_decoder.c:774-788 and 316-366). After a frame that ended with
    /// a redundancy frame for a switch to CELT-only, that mode is CELT-only.
    ///
    /// The loss is concealed in pieces, and the pieces are audible: every
    /// piece of a pitch-concealed run is quieter than the one before. The
    /// rules for a piece are libopus's:
    ///
    /// - One call of `opus_decode_frame` covers at most one frame of the
    ///   last packet (`frame_size = IMIN(frame_size, st->frame_size)`). A
    ///   loss after 10 ms packets is concealed 10 ms at a time.
    /// - A call longer than 20 ms is cut into pieces of 20 ms, and a last
    ///   shorter piece.
    /// - A piece is 2.5, 5, 10 or 20 ms: the next shorter one when what is
    ///   left lies between two of them. SILK-only also takes 7.5 ms.
    fn conceal(&mut self, total: usize, output: &mut [f32]) -> Result<(), &'static str> {
        let f2_5 = self.sampling_rate as usize / 400;
        let (f5, f10, f20) = (2 * f2_5, 4 * f2_5, 8 * f2_5);
        let channels = self.channels;
        // C 331: "Run PLC using last used mode (CELT if we ended with CELT
        // redundancy)".
        let plc_mode = if self.prev_redundancy {
            Some(OpusMode::CeltOnly)
        } else {
            self.prev_mode
        };
        let Some(plc_mode) = plc_mode else {
            // C 334-341: no packet yet, all there is to return is zeros. The
            // state stays as it is.
            output[..total * channels].fill(0.0);
            return Ok(());
        };

        let mut pos = 0usize;
        while pos < total {
            // One call of `opus_decode_frame`. A frame of a packet is 60 ms
            // at most, which is also below the 120 ms that libopus limits a
            // call to.
            let mut left = (total - pos).min(self.frame_size);
            while left > 0 {
                // C 345-365. The inner calls of a call longer than 20 ms get
                // `IMIN(audiosize, F20)`.
                let limit = left.min(f20);
                let piece = if limit > f10 && limit < f20 {
                    f10
                } else if plc_mode != OpusMode::SilkOnly && limit > f5 && limit < f10 {
                    f5
                } else {
                    limit
                };
                let dst = &mut output[pos * channels..(pos + piece) * channels];
                if plc_mode == OpusMode::CeltOnly {
                    self.conceal_celt(piece, 0, dst, PcmLayout::Interleaved)?;
                } else {
                    // C 402-497: SILK conceals 10 or 20 ms, with the channel
                    // count and rate of the last frame that carried data. A
                    // shorter piece is the head of 10 ms.
                    let mut rc = RangeCoder::new_decoder(&[]);
                    self.decode_silk(&mut rc, silk::decode_frame::FLAG_PACKET_LOST, piece, None)?;
                    dst.copy_from_slice(&self.w_silk_out[..piece * channels]);
                    if plc_mode == OpusMode::Hybrid {
                        // C 610: CELT conceals the piece itself
                        // (`celt_frame_size = IMIN(F20, frame_size)`), also
                        // when SILK had to produce 10 ms for a shorter one.
                        // It adds its output to SILK's.
                        let layout = PcmLayout::InterleavedAdd;
                        self.conceal_celt(piece, HYBRID_START_BAND, dst, layout)?;
                    }
                }
                pos += piece;
                left -= piece;
            }
        }
        // C 706-712: a concealed frame has no final range and no redundancy
        // frame, and leaves its mode behind.
        self.prev_mode = Some(plc_mode);
        self.prev_redundancy = false;
        self.range_final = 0;
        Ok(())
    }

    /// CELT concealment for `n` samples per channel into `out` (libopus
    /// `opus_decode_frame` with `data == NULL`, which calls
    /// `celt_decode_with_ec` without data).
    ///
    /// The band range starts at `start_band`: 0 for a CELT-only stream and
    /// `HYBRID_START_BAND` for the CELT layer of a Hybrid stream. It ends
    /// where the last frame with data set it.
    ///
    /// libopus sets the CELT decoder's stream channel count before every
    /// CELT call, this one included (`CELT_SET_CHANNELS(st->stream_channels)`,
    /// opus_decoder.c:581), and so does this function. The concealment
    /// itself works on the output channels and does not read the count.
    fn conceal_celt(
        &mut self,
        n: usize,
        start_band: usize,
        out: &mut [f32],
        layout: PcmLayout,
    ) -> Result<(), &'static str> {
        self.celt_dec.set_stream_channels(self.stream_channels)?;
        self.celt_dec.set_start_band(start_band)?;
        self.celt_dec.decode_lost(n, out, layout)?;
        Ok(())
    }
}

impl OpusDecoder {
    /// The SILK layer of one Opus frame: decode, conceal or recover
    /// `frame_size` samples per channel into `out`, interleaved at the
    /// output rate with the decoder's channel count (libopus
    /// `opus_decode_frame`, "SILK processing", opus_decoder.c:402-497).
    /// `None` stands for `w_silk_out`, which concealment uses.
    ///
    /// SILK itself works in frames of 10 or 20 ms. A 40 or 60 ms Opus frame
    /// holds 2 or 3 of them in one range-coded stream, so `silk_Decode` is
    /// called until the duration is there, with the "new packet" flag in
    /// the first call only.
    ///
    /// The channel count and the rate SILK codes at are those of the last
    /// frame that carried data (`silk_channels_internal`,
    /// `silk_internal_rate`). `SilkDecoder::decode_with_control` resamples
    /// to the output rate and converts between channel counts.
    ///
    /// `lost_flag` is one of `FLAG_DECODE_NORMAL`, `FLAG_PACKET_LOST`
    /// (conceal) and `FLAG_DECODE_LBRR` (the redundant copy of the frame
    /// before, for FEC).
    fn decode_silk(
        &mut self,
        rc: &mut RangeCoder,
        lost_flag: i32,
        frame_size: usize,
        out: Option<&mut [f32]>,
    ) -> Result<(), &'static str> {
        let channels = self.channels;
        let f10 = self.sampling_rate as usize / 100;
        let out = match out {
            Some(out) => out,
            None => &mut self.w_silk_out[..],
        };
        let control = silk::decoder_structs::SilkDecControl {
            n_channels_api: channels as i32,
            n_channels_internal: self.silk_channels_internal as i32,
            api_sample_rate: self.sampling_rate,
            internal_sample_rate: self.silk_internal_rate,
            // C 421-422: "The SILK PLC cannot produce frames of less than
            // 10 ms".
            payload_size_ms: (1000 * frame_size as i32 / self.sampling_rate).max(10),
        };
        // C 409-416: a piece shorter than 10 ms is the head of a 10 ms
        // frame. Only concealment asks for such a piece, and `w_silk_out`
        // has room for 10 ms. A frame with data lasts 10 ms or more.
        let room = frame_size.max(f10);
        let mut done = 0usize;
        while done < frame_size {
            let decoded = {
                let (silk_dec, pcm) = (
                    state_mut(&mut self.silk_dec),
                    state_mut(&mut self.w_pcm_i16),
                );
                silk_dec.decode_with_control(&control, rc, pcm, lost_flag, done == 0)
            };
            match decoded {
                Ok(n) if n > 0 && done + n <= room => {
                    // C `INT16TORES`: SILK's 16-bit samples as floats.
                    let silk_out = &mut out[done * channels..(done + n) * channels];
                    for (o, &v) in silk_out.iter_mut().zip(self.w_pcm_i16.iter()) {
                        *o = int16_to_res(v);
                    }
                    done += n;
                }
                _ if lost_flag != silk::decode_frame::FLAG_DECODE_NORMAL => {
                    // C 480-485: "PLC failure should not be fatal". The
                    // rest of the frame is silence.
                    out[done * channels..room * channels].fill(0.0);
                    done = frame_size;
                }
                _ => return Err("SILK decoding failed"),
            }
        }
        Ok(())
    }
}

/// libopus `INT16TORES` of the float build: a 16-bit sample as a float in
/// [-1, 1). Dividing by a power of two is exact.
#[inline(always)]
pub(crate) fn int16_to_res(x: i16) -> f32 {
    x as f32 * (1.0 / 32768.0)
}

/// libopus `FLOAT2INT16` (float_cast.h): scale a float sample to 16 bits,
/// limit it to the 16-bit range and round it to the nearest integer. A
/// value halfway between two integers goes to the even one.
#[inline]
pub(crate) fn float2int16(x: f32) -> i16 {
    let x = x * 32768.0;
    // The comparisons are written as in C, so that a NaN gives -32768.
    let x = if x > -32768.0 { x } else { -32768.0 };
    let x = if x < 32767.0 { x } else { 32767.0 };
    // C `lrintf`. A float at or beyond 2^23 has no fraction bits, so adding
    // 2^23 rounds to an integer, to nearest and ties to even.
    const TWO_23: f32 = 8_388_608.0;
    let rounded = if x >= 0.0 { (x + TWO_23) - TWO_23 } else { (x - TWO_23) + TWO_23 };
    rounded as i16
}

/// libopus `opus_pcm_soft_clip` (opus.c): bring a float signal into
/// [-1, 1] without hard clipping.
///
/// Where the signal goes beyond ±1, the stretch between the zero crossings
/// around that place is bent by `x + a * x * x`. `a` is chosen so that the
/// peak of the stretch lands on ±1. Samples beyond ±2 are cut to ±2 first.
///
/// - `pcm`: interleaved samples, `frame_size * channels` of them.
/// - `declip_mem`: one value per channel, all 0 at the start of a stream.
///   It carries the bend that is under way from one call to the next, so
///   that consecutive buffers join without a step.
///
/// Arguments that do not fit (no samples, a slice too short) leave `pcm`
/// unchanged.
pub fn pcm_soft_clip(pcm: &mut [f32], frame_size: usize, channels: usize, declip_mem: &mut [f32]) {
    let (n, nc) = (frame_size, channels);
    // A sample count that overflows cannot fit `pcm` either.
    let fits = n.checked_mul(nc).is_some_and(|len| len <= pcm.len());
    if nc < 1 || n < 1 || !fits || declip_mem.len() < nc {
        return;
    }
    // Clamp everything to [-2, 2], the domain of the non-linearity
    // (`opus_limit2_checkwithin1`). Beyond it the derivative is zero, so
    // this adds no discontinuity.
    for v in pcm[..n * nc].iter_mut() {
        *v = v.clamp(-2.0, 2.0);
    }
    for c in 0..nc {
        // Channel `c`: sample `i` is `x[i * nc]`.
        let x = &mut pcm[c..];
        let mut a = declip_mem[c];
        // Continue applying the non-linearity from the previous frame to
        // avoid any discontinuity.
        for i in 0..n {
            let v = x[i * nc];
            if v * a >= 0.0 {
                break;
            }
            x[i * nc] = v + a * v * v;
        }

        let mut curr = 0usize;
        let x0 = x[0];
        loop {
            let mut i = curr;
            while i < n {
                let v = x[i * nc];
                if v > 1.0 || v < -1.0 {
                    break;
                }
                i += 1;
            }
            if i == n {
                a = 0.0;
                break;
            }
            let clip = x[i * nc];
            let mut peak_pos = i;
            let (mut start, mut end) = (i, i);
            let mut maxval = clip.abs();
            // Look for first zero crossing before clipping.
            while start > 0 && clip * x[(start - 1) * nc] >= 0.0 {
                start -= 1;
            }
            // Look for first zero crossing after clipping.
            while end < n && clip * x[end * nc] >= 0.0 {
                // Look for other peaks until the next zero-crossing.
                let v = x[end * nc].abs();
                if v > maxval {
                    maxval = v;
                    peak_pos = end;
                }
                end += 1;
            }
            // Detect the special case where we clip before the first zero
            // crossing.
            let special = start == 0 && clip * x[0] >= 0.0;

            // Compute a such that maxval + a*maxval^2 = 1.
            a = (maxval - 1.0) / (maxval * maxval);
            // Slightly boost "a" by 2^-22. libopus does this so that a
            // fast-math build cannot give values beyond +/-1. It is too
            // small to matter even for 24-bit output.
            a += a * 2.4e-7;
            if clip > 0.0 {
                a = -a;
            }
            // Apply soft clipping.
            for k in start..end {
                let v = x[k * nc];
                x[k * nc] = v + a * v * v;
            }

            if special && peak_pos >= 2 {
                // Add a linear ramp from the first sample to the signal
                // peak. This avoids a discontinuity at the beginning of the
                // frame.
                let mut offset = x0 - x[0];
                let delta = offset / peak_pos as f32;
                for k in curr..peak_pos {
                    offset -= delta;
                    x[k * nc] = (x[k * nc] + offset).clamp(-1.0, 1.0);
                }
            }
            curr = end;
            if curr == n {
                break;
            }
        }
        declip_mem[c] = a;
    }
}

/// Whether `frame_size` samples at `fs` samples per second are a whole number
/// of 2.5 ms, and not 0. libopus requires this of the frame size of a lost
/// frame and of an FEC decode.
#[inline]
fn is_whole_2_5ms(frame_size: usize, fs: i32) -> bool {
    frame_size != 0 && frame_size.is_multiple_of(fs as usize / 400)
}

/// The number of frames per second (rounded down) for frames of
/// `frame_size` samples at `sampling_rate`, or `None` when the encoder does
/// not take that frame size. A frame is 2.5, 5, 10, 20, 40, 60, 80, 100 or
/// 120 ms: the durations libopus `frame_size_select` accepts.
pub(crate) fn frame_rate_from_params(sampling_rate: i32, frame_size: usize) -> Option<i32> {
    // A size that does not fit an `i32` is no frame size. Cutting it to 32
    // bits could turn it into one. No valid frame is longer than a second,
    // and checking this first keeps the products below in range.
    let frame_size = i32::try_from(frame_size).ok()?;
    if frame_size <= 0 || frame_size > sampling_rate {
        return None;
    }
    // The check goes by duration, as in libopus, not by
    // `sampling_rate % frame_size == 0`: 60 ms is 16.67 frames per second.
    let (fs, n) = (sampling_rate, frame_size);
    let valid = 400 * n == fs
        || 200 * n == fs
        || 100 * n == fs
        || 50 * n == fs
        || 25 * n == fs
        || 50 * n == 3 * fs
        || 50 * n == 4 * fs
        || 50 * n == 5 * fs
        || 50 * n == 6 * fs;
    if !valid {
        return None;
    }
    Some(fs / frame_size)
}

/// libopus `frame_size_select` (opus_encoder.c:827): the frame an encode
/// call codes, in samples per channel, or `None` when the call is refused.
///
/// - `frame_size` is what the caller passed (C `analysis_frame_size`).
/// - With [`FrameDuration::Argument`] the frame is `frame_size`.
/// - With any other duration the frame has that duration. The caller must
///   pass at least that much.
///
/// The frame must then be one of the durations Opus has (see
/// [`frame_rate_from_params`]). libopus also refuses frames under 10 ms
/// for its `RESTRICTED_SILK` application, which opus1 does not have.
pub(crate) fn frame_size_select(
    frame_size: usize,
    duration: FrameDuration,
    sampling_rate: i32,
) -> Option<usize> {
    if frame_size < sampling_rate as usize / 400 {
        return None;
    }
    let new_size = duration.samples(sampling_rate).unwrap_or(frame_size);
    if new_size > frame_size {
        return None;
    }
    frame_rate_from_params(sampling_rate, new_size)?;
    Some(new_size)
}

fn gen_toc(mode: OpusMode, frame_rate: i32, bandwidth: Bandwidth, channels: usize) -> u8 {
    let mut rate = frame_rate;
    let mut period = 0;
    while rate < 400 {
        rate <<= 1;
        period += 1;
    }

    let mut toc = match mode {
        OpusMode::SilkOnly => {
            let bw = (bandwidth as i32 - Bandwidth::Narrowband as i32) << 5;
            let per = (period - 2) << 3;
            (bw | per) as u8
        }
        OpusMode::CeltOnly => {
            let mut tmp = bandwidth as i32 - Bandwidth::Mediumband as i32;
            if tmp < 0 {
                tmp = 0;
            }
            let per = period << 3;
            (0x80 | (tmp << 5) | per) as u8
        }
        OpusMode::Hybrid => {
            let base_config = if bandwidth == Bandwidth::Superwideband {
                12
            } else {
                14
            };
            let period_offset = if frame_rate >= 100 { 0 } else { 1 };
            ((base_config + period_offset) << 3) as u8
        }
    };

    if channels == 2 {
        toc |= 0x04;
    }
    toc
}

fn mode_from_toc(toc: u8) -> OpusMode {
    if toc & 0x80 != 0 {
        OpusMode::CeltOnly
    } else if toc & 0x60 == 0x60 {
        OpusMode::Hybrid
    } else {
        OpusMode::SilkOnly
    }
}

fn bandwidth_from_toc(toc: u8) -> Bandwidth {
    let mode = mode_from_toc(toc);
    match mode {
        OpusMode::SilkOnly => {
            let bw_bits = (toc >> 5) & 0x03;
            match bw_bits {
                0 => Bandwidth::Narrowband,
                1 => Bandwidth::Mediumband,
                2 => Bandwidth::Wideband,
                _ => Bandwidth::Wideband,
            }
        }
        OpusMode::Hybrid => {
            let bw_bit = (toc >> 4) & 0x01;
            if bw_bit == 0 {
                Bandwidth::Superwideband
            } else {
                Bandwidth::Fullband
            }
        }
        OpusMode::CeltOnly => {
            // CELT has no mediumband: configurations 16..19 are narrowband
            // (libopus `opus_packet_get_bandwidth`).
            let bw_bits = (toc >> 5) & 0x03;
            match bw_bits {
                0 => Bandwidth::Narrowband,
                1 => Bandwidth::Wideband,
                2 => Bandwidth::Superwideband,
                3 => Bandwidth::Fullband,
                _ => Bandwidth::Fullband,
            }
        }
    }
}

#[cfg(all(test, feature = "std"))]
fn frame_duration_ms_from_toc(toc: u8) -> i32 {
    let mode = mode_from_toc(toc);
    match mode {
        OpusMode::SilkOnly => {
            let config = (toc >> 3) & 0x03;
            match config {
                0 => 10,
                1 => 20,
                2 => 40,
                3 => 60,
                _ => 20,
            }
        }
        OpusMode::Hybrid => {
            let config = (toc >> 3) & 0x01;
            if config == 0 { 10 } else { 20 }
        }
        OpusMode::CeltOnly => {
            let config = (toc >> 3) & 0x03;
            match config {
                0 => 2,
                1 => 5,
                2 => 10,
                3 => 20,
                _ => 20,
            }
        }
    }
}

fn channels_from_toc(toc: u8) -> usize {
    if toc & 0x04 != 0 { 2 } else { 1 }
}

/// Cross-fade `overlap` interleaved samples per channel between `out` and
/// `other`, in place (libopus `smooth_fade`, with `out` the same buffer as
/// one of its inputs). The weight of the signal that fades in rises from 0
/// to 1 as the square of the CELT window. `window` is the 120-sample CELT
/// window at 48 kHz, and `inc` = 48000 / Fs strides it.
///
/// With `out_fades_in`, `other` fades out and `out` fades in. Otherwise
/// `out` fades out and `other` fades in.
fn smooth_fade(
    other: &[f32],
    out: &mut [f32],
    out_fades_in: bool,
    overlap: usize,
    channels: usize,
    window: &[f32],
    inc: usize,
) {
    for (i, &win) in window.iter().step_by(inc).take(overlap).enumerate() {
        let w = win * win;
        for c in 0..channels {
            let k = i * channels + c;
            let (in1, in2) = if out_fades_in { (other[k], out[k]) } else { (out[k], other[k]) };
            out[k] = w * in2 + (1.0 - w) * in1;
        }
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;

    fn frame_size_from_toc(toc: u8, sampling_rate: i32) -> Option<usize> {
        let mode = mode_from_toc(toc);
        match mode {
            OpusMode::CeltOnly => {
                let period = ((toc >> 3) & 0x03) as i32;
                let frame_rate = 400 >> period;
                if frame_rate == 0 || sampling_rate % frame_rate != 0 {
                    return None;
                }
                Some((sampling_rate / frame_rate) as usize)
            }
            OpusMode::SilkOnly => {
                let duration_ms = frame_duration_ms_from_toc(toc);
                Some((sampling_rate as i64 * duration_ms as i64 / 1000) as usize)
            }
            OpusMode::Hybrid => {
                let duration_ms = frame_duration_ms_from_toc(toc);
                Some((sampling_rate as i64 * duration_ms as i64 / 1000) as usize)
            }
        }
    }

    #[test]
    fn gen_toc_matches_celt_reference_values() {
        let sampling_rate = 48_000;
        let cases = [
            (120usize, 0xE0u8),
            (240usize, 0xE8u8),
            (480usize, 0xF0u8),
            (960usize, 0xF8u8),
        ];

        for (frame_size, expected_toc) in cases {
            let frame_rate = frame_rate_from_params(sampling_rate, frame_size).unwrap();
            let toc = gen_toc(OpusMode::CeltOnly, frame_rate, Bandwidth::Fullband, 1);
            assert_eq!(
                toc, expected_toc,
                "frame_size {} expected TOC {:02X} got {:02X}",
                frame_size, expected_toc, toc
            );
            let decoded_size = frame_size_from_toc(toc, sampling_rate).unwrap();
            assert_eq!(decoded_size, frame_size);
        }

        let stereo_toc = gen_toc(
            OpusMode::CeltOnly,
            frame_rate_from_params(sampling_rate, 960).unwrap(),
            Bandwidth::Fullband,
            2,
        );
        assert_eq!(channels_from_toc(stereo_toc), 2);
    }

    #[test]
    fn test_celt_decoder_large_frame_sizes() {
        let sampling_rate = 48000;
        let channels = 1;

        let mut decoder = OpusDecoder::new(sampling_rate, channels).unwrap();

        let frame_sizes = [120, 240, 480, 960];

        for frame_size in frame_sizes {
            let toc = gen_toc(
                OpusMode::CeltOnly,
                frame_rate_from_params(sampling_rate, frame_size).unwrap(),
                Bandwidth::Fullband,
                channels,
            );
            let packet = [toc, 0, 0, 0, 0];

            let mut output = vec![0.0f32; frame_size * channels];

            let _ = decoder.decode(&packet, frame_size, &mut output);
        }

        let channels = 2;
        let mut decoder = OpusDecoder::new(sampling_rate, channels).unwrap();

        for frame_size in frame_sizes {
            let toc = gen_toc(
                OpusMode::CeltOnly,
                frame_rate_from_params(sampling_rate, frame_size).unwrap(),
                Bandwidth::Fullband,
                channels,
            );
            let packet = [toc, 0, 0, 0, 0];

            let mut output = vec![0.0f32; frame_size * channels];
            let _ = decoder.decode(&packet, frame_size, &mut output);
        }
    }

    #[test]
    fn test_celt_decoder_edge_case_frame_sizes() {
        let sampling_rate = 48000;
        let channels = 1;
        let mut decoder = OpusDecoder::new(sampling_rate, channels).unwrap();

        let edge_sizes = [2048, 2167, 2168, 2169, 2880, 3072];

        for frame_size in edge_sizes {
            let mut output = vec![0.0f32; frame_size * channels];

            let _ = decoder.decode(&[0x80, 0, 0, 0], frame_size, &mut output);
        }
    }

    // Regression test for: "index out of bounds: the len is 48 but the index is 119"
    // Root cause: frame_size=48 at 48kHz gives frame_rate=1000, which is not a valid
    // Hybrid-mode frame rate but was not validated.  CELT's lm-search then silently
    // fell back to lm=0, computed n2=120, and wrote output[119] into a 48-element
    // slice.  Triggered via G.729-decoded PCM (8kHz) passed to a 48kHz Opus encoder
    // without proper resampling, so the encoder received 48 samples instead of 480.
    #[test]
    fn test_invalid_small_frame_size_returns_error_not_panic() {
        let mut enc = OpusEncoder::new(48000, 2, Application::Voip).unwrap();
        enc.set_bitrate(Bitrate::BitsPerSecond(64000)).unwrap();
        enc.set_complexity(5).unwrap();
        enc.set_vbr(false);

        // 48 samples at 48kHz = 1ms → frame_rate=1000, invalid for Hybrid mode.
        let input = vec![0.0f32; 48 * 2]; // stereo interleaved
        let mut output = vec![0u8; 256];

        let result = enc.encode(&input, 48, &mut output);
        assert!(
            result.is_err(),
            "encode with invalid frame_size=48 should return Err, not panic"
        );
    }

    /// libopus `FLOAT2INT16`: scale, limit to 16 bits, round to nearest
    /// with halfway cases to the even integer.
    #[test]
    fn float2int16_rounds_and_saturates() {
        let lsb = 1.0 / 32768.0;
        assert_eq!(float2int16(0.0), 0);
        assert_eq!(float2int16(100.4 * lsb), 100);
        assert_eq!(float2int16(100.6 * lsb), 101);
        assert_eq!(float2int16(-100.6 * lsb), -101);
        // Halfway cases go to the even neighbour.
        assert_eq!(float2int16(0.5 * lsb), 0);
        assert_eq!(float2int16(1.5 * lsb), 2);
        assert_eq!(float2int16(2.5 * lsb), 2);
        assert_eq!(float2int16(-0.5 * lsb), 0);
        assert_eq!(float2int16(-1.5 * lsb), -2);
        // The 16-bit range is not symmetric.
        assert_eq!(float2int16(1.0), 32767);
        assert_eq!(float2int16(32767.4 * lsb), 32767);
        assert_eq!(float2int16(-1.0), -32768);
        assert_eq!(float2int16(7.0), 32767);
        assert_eq!(float2int16(-7.0), -32768);
        assert_eq!(float2int16(f32::INFINITY), 32767);
        assert_eq!(float2int16(f32::NEG_INFINITY), -32768);
        assert_eq!(float2int16(f32::NAN), -32768);
    }

    /// Encode a tone with `enc`, `frames` frames of `frame_size` samples.
    fn encode_tone(enc: &mut OpusEncoder, frame_size: usize, frames: usize) -> Vec<Vec<u8>> {
        let mut packets = Vec::new();
        for f in 0..frames {
            let pcm: Vec<f32> = (0..frame_size)
                .map(|i| 0.4 * ((f * frame_size + i) as f32 * 0.05).sin())
                .collect();
            let mut out = vec![0u8; 1500];
            let n = enc.encode(&pcm, frame_size, &mut out).unwrap();
            out.truncate(n);
            packets.push(out);
        }
        packets
    }

    /// `decode_native` reads a self-delimited packet that is followed by
    /// other data, decodes what `decode` decodes from the plain packet, and
    /// says where the packet ends.
    #[test]
    fn decode_native_reads_self_delimited_packets() {
        // 20 ms packets hold one frame, 60 ms CELT packets three.
        for frame_size in [960usize, 2880] {
            let mut enc = OpusEncoder::new(48000, 1, Application::RestrictedLowDelay).unwrap();
            let mut plain = OpusDecoder::new(48000, 1).unwrap();
            let mut delimited = OpusDecoder::new(48000, 1).unwrap();
            let (mut a, mut b) = (vec![0f32; frame_size], vec![0f32; frame_size]);
            for p in encode_tone(&mut enc, frame_size, 12) {
                let parsed = packet::parse(&p, false).unwrap();
                assert_eq!(parsed.count, frame_size / 960);
                let frames: Vec<&[u8]> = parsed.frames(&p).collect();
                let mut stream = vec![0u8; p.len() + 16];
                let len = packet::write(parsed.toc, &frames, &mut stream, true, false).unwrap();
                // The next stream of a multistream packet would follow here.
                stream.truncate(len);
                stream.extend_from_slice(&[0xF8, 1, 2, 3, 4]);

                assert_eq!(plain.decode(&p, frame_size, &mut a), Ok(frame_size));
                assert_eq!(
                    delimited.decode_native(&stream, frame_size, &mut b, false, true, false),
                    Ok((frame_size, len))
                );
                assert_eq!(a, b);
                assert_eq!(plain.final_range(), delimited.final_range());
                assert_ne!(plain.final_range(), 0);
            }
            // A lost packet takes no bytes.
            assert_eq!(delimited.decode_native(&[], 960, &mut b, false, true, false), Ok((960, 0)));
            assert_eq!(delimited.final_range(), 0);
            // FEC decode of a CELT-only packet conceals, and still says how
            // long the packet is.
            let stream = [0xF8, 3, 1, 2, 3, 0xF8, 9];
            assert_eq!(
                delimited.decode_native(&stream, 960, &mut b, true, true, false),
                Ok((960, 5))
            );
            // A self-delimited packet that is cut short is an error.
            let cut = [0xF8, 3, 1, 2];
            assert!(delimited.decode_native(&cut, 960, &mut b, false, true, false).is_err());
        }
    }

    /// Self-delimited SILK packets with in-band FEC, some of them padded, as
    /// the streams of a multistream packet are: a normal decode, an FEC
    /// decode and a soft-clipped decode through `decode_native` give what
    /// the public calls give for the plain packet, and each says where the
    /// self-delimited packet ends.
    #[test]
    fn decode_native_handles_fec_and_padding_in_self_delimited_packets() {
        let frame_size = 320usize;
        let mut enc = OpusEncoder::new(16000, 1, Application::Voip).unwrap();
        enc.set_bitrate(Bitrate::BitsPerSecond(28000)).unwrap();
        enc.set_inband_fec(true);
        enc.set_packet_loss_perc(25).unwrap();
        let packets = encode_tone(&mut enc, frame_size, 40);
        // SILK-only packets, half of them or more with a copy of the frame
        // before: the second bit of a mono SILK frame is its FEC flag.
        assert!(packets.iter().all(|p| p[0] & 0x80 == 0));
        let with_copy = packets.iter().filter(|p| p[1] & 0x40 != 0).count();
        assert!(with_copy >= 15, "{with_copy} of 40 packets carry FEC data");
        let mut plain = OpusDecoder::new(16000, 1).unwrap();
        let mut delimited = OpusDecoder::new(16000, 1).unwrap();
        let (mut a, mut b) = (vec![0f32; 2 * frame_size], vec![0f32; 2 * frame_size]);
        let mut a16 = vec![0i16; frame_size];
        let mut fec_calls = 0;
        for (i, p) in packets.iter().enumerate() {
            let parsed = packet::parse(p, false).unwrap();
            let frames: Vec<&[u8]> = parsed.frames(p).collect();
            // Every third packet is padded by 1, 2 or 300 bytes.
            let extra = if i % 3 == 0 { [1usize, 2, 300][(i / 3) % 3] } else { 0 };
            let mut stream = vec![0u8; p.len() + 2 + extra];
            let len = if extra == 0 {
                packet::write(parsed.toc, &frames, &mut stream, true, false).unwrap()
            } else {
                packet::write(parsed.toc, &frames, &mut stream, true, true).unwrap()
            };
            stream.truncate(len);
            stream.extend_from_slice(&[0x0B, 0xFF, 7]);

            if i % 7 == 5 {
                // The packet before this one was lost, together with the
                // one before that on every second occasion.
                let lost = if fec_calls % 2 == 0 { frame_size } else { 2 * frame_size };
                assert_eq!(plain.decode_fec(p, lost, &mut a), Ok(lost));
                assert_eq!(
                    delimited.decode_native(&stream, lost, &mut b, true, true, false),
                    Ok((lost, len))
                );
                assert_eq!(a[..lost], b[..lost], "packet {i}: FEC decode");
                assert_eq!(plain.final_range(), delimited.final_range());
                fec_calls += 1;
            }
            if i % 4 == 1 {
                // The 16-bit path: soft clipping, then rounding.
                assert_eq!(plain.decode_i16(p, frame_size, &mut a16), Ok(frame_size));
                assert_eq!(
                    delimited.decode_native(&stream, frame_size, &mut b, false, true, true),
                    Ok((frame_size, len))
                );
                let converted: Vec<i16> = b[..frame_size].iter().map(|&v| float2int16(v)).collect();
                assert_eq!(a16, converted, "packet {i}: 16-bit decode");
            } else {
                assert_eq!(plain.decode(p, frame_size, &mut a), Ok(frame_size));
                assert_eq!(
                    delimited.decode_native(&stream, frame_size, &mut b, false, true, false),
                    Ok((frame_size, len))
                );
                assert_eq!(a[..frame_size], b[..frame_size], "packet {i}");
            }
            assert_eq!(plain.final_range(), delimited.final_range(), "packet {i}");
            assert_ne!(plain.final_range(), 0, "packet {i}");
        }
        assert!(fec_calls >= 4);
    }

    // Also verify that the Audio application path (always Hybrid at 48 kHz) rejects
    // the same bad frame size.
    #[test]
    fn test_invalid_small_frame_size_audio_application_returns_error() {
        let mut enc = OpusEncoder::new(48000, 1, Application::Audio).unwrap();
        let input = vec![0.0f32; 48];
        let mut output = vec![0u8; 256];

        let result = enc.encode(&input, 48, &mut output);
        assert!(
            result.is_err(),
            "Audio/48kHz encoder with frame_size=48 should return Err"
        );
    }
}
