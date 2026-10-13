#![cfg_attr(not(feature = "std"), no_std)]
#![allow(unsafe_op_in_unsafe_fn)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]

mod compat;
mod fixedvec;

pub mod bands;
pub mod celt;
pub mod celt_lpc;
pub mod hp_cutoff;
pub mod kiss_fft;
pub mod mdct;
pub mod modes;
/// Multistream decode uses `Vec`-based plumbing, so it is only available
/// with the `std` (default) feature set; the core codec stays `no_std`
/// + no-`alloc` capable (see `split_self_delimited` docs).
#[cfg(feature = "std")]
pub mod multistream;
pub mod pitch;
pub mod pvq;
pub mod quant_bands;
pub mod range_coder;
pub mod rate;
pub mod silk;

pub use silk::{SilkResampler, SilkResamplerDown1_3, SilkResamplerDown1_6};

#[cfg(not(feature = "std"))]
use crate::compat::Math;
use crate::fixedvec::FixedVec;
pub use celt::{CeltDecoder, CeltEncoder};
use hp_cutoff::{dc_reject_float, hp_cutoff, hp_cutoff_float, hp_cutoff_i16};
use range_coder::RangeCoder;
use silk::control_codec::{silk_control_encoder, silk_setup_lbrr};
use silk::enc_api::{
    silk_encode, silk_encode_prefill, silk_encode_prefill_stereo, silk_encode_stereo_packet,
};
use silk::init_encoder::silk_init_encoder;
use silk::lin2log::silk_lin2log;
use silk::log2lin::silk_log2lin;
use silk::macros::*;
use silk::structs::SilkEncoderState;

// --- Heap-free buffer capacity constants (worst case: 2 channels). ---
const OPUS_MAX_CHANNELS: usize = 2;
/// Largest API frame in samples/channel (120 ms @ 48 kHz = 5760). Used by the
/// encoder's per-frame input buffers (sized `frame_size * channels`).
const OPUS_MAX_FRAME: usize = 5760;
/// Largest SILK packet at the internal rate: 60 ms at 16 kHz.
const SILK_MAX_PACKET_SAMPLES: usize = 3 * silk::define::MAX_FRAME_LENGTH;
/// Largest *single-frame* samples/channel (60 ms @ 48 kHz = 2880). The decoder's
/// staging buffers hold one sub-frame at a time, so they're sized to this — not
/// the full packet. (Halves the decoder footprint vs. a naive 5760/channel.)
const OPUS_MAX_SUBFRAME: usize = 2880;
/// Decoder per-sub-frame staging cap: `OPUS_MAX_SUBFRAME * max_channels`.
const OPUS_SUBFRAME_SCRATCH: usize = OPUS_MAX_SUBFRAME * OPUS_MAX_CHANNELS;
/// High-pass filter state memory (`channels * 2`).
const OPUS_HP_MEM: usize = OPUS_MAX_CHANNELS * 2;
/// Decoder `w_pcm_i16` cap (`960 * max_channels`).
const OPUS_PCM_I16: usize = 960 * OPUS_MAX_CHANNELS;
/// Largest 5 ms of interleaved decoder output (`240 * max_channels`): the
/// size of a redundancy frame or of the concealment that bridges a mode
/// switch.
const OPUS_PCM_TAIL: usize = 240 * OPUS_MAX_CHANNELS;
/// Max number of frames encoded in one Opus packet (RFC 6716 caps at 48 for
/// 2.5 ms codes in a 120 ms packet).
const OPUS_MAX_PACKET_FRAMES: usize = 48;
/// RFC 6716 §3.1: a single Opus packet carries at most 1276 bytes of data.
const OPUS_MAX_PACKET_BYTES: usize = 1276;
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
    Auto = -1000,
    Narrowband = 1101,
    Mediumband = 1102,
    Wideband = 1103,
    Superwideband = 1104,
    Fullband = 1105,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OpusMode {
    SilkOnly,
    Hybrid,
    CeltOnly,
}

/// The decisions `opus_encode_native` makes once per packet.
#[derive(Debug, Clone, Copy)]
struct ForcedFrame {
    mode: OpusMode,
    bandwidth: Bandwidth,
    /// Whether the packet codes in-band FEC (libopus `LBRR_coded`).
    lbrr_coded: i32,
}

pub struct OpusEncoder {
    #[cfg(not(feature = "heap"))]
    celt_enc: CeltEncoder,
    #[cfg(feature = "heap")]
    celt_enc: Box<CeltEncoder>,
    /// SILK encoder for mono input, or for the mid of stereo input (it also
    /// holds the stereo state).
    #[cfg(not(feature = "heap"))]
    silk_enc: SilkEncoderState,
    #[cfg(feature = "heap")]
    silk_enc: Box<SilkEncoderState>,
    /// SILK encoder for the side of stereo input (libopus `state_Fxx[1]`).
    #[cfg(not(feature = "heap"))]
    silk_enc_side: SilkEncoderState,
    #[cfg(feature = "heap")]
    silk_enc_side: Box<SilkEncoderState>,
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
    /// (`enable_hybrid_mode`), or `None` for the automatic choice.
    user_bandwidth: Option<Bandwidth>,
    /// C `st->first`: no packet has been coded yet.
    first: bool,
    /// C `silk_mode.allowBandwidthSwitch`: SILK's speech activity is low
    /// enough for a bandwidth switch to go unheard.
    silk_allow_bandwidth_switch: bool,
    /// C `timeSinceSwitchAllowed_ms` (enc_API.c): how long SILK has refused
    /// a bandwidth switch. The bar drops with it and is gone after 5 s.
    silk_time_since_switch_allowed_ms: i32,
    /// C `silk_mode.inWBmodeWithoutVariableLP`: SILK runs at 16 kHz and is
    /// not in a low-pass transition.
    silk_in_wb_mode_without_variable_lp: bool,
    /// C `silk_mode.opusCanSwitch`: SILK reported on the last packet that it
    /// is ready to change its internal rate, so it may on this one.
    silk_opus_can_switch: bool,
    /// C `st->nonfinal_frame`: this 20 ms part of a 40/60 ms packet is not
    /// the last one.
    nonfinal_frame: bool,
    pub bitrate_bps: i32,
    pub complexity: i32,
    pub use_cbr: bool,

    pub use_inband_fec: bool,

    pub packet_loss_perc: i32,
    silk_initialized: bool,
    mode: OpusMode,
    prev_enc_mode: Option<OpusMode>,
    /// When set, `encode_impl` uses these decisions instead of making its
    /// own. It keeps every 20 ms part of a 40/60 ms packet in the packet's
    /// mode and bandwidth (libopus codes the parts with
    /// `opus_encode_frame_native`, after `opus_encode_native` has decided).
    forced_frame: Option<ForcedFrame>,

    variable_hp_smth2_q15: i32,
    hp_mem: FixedVec<i32, OPUS_HP_MEM>,

    #[cfg(not(feature = "heap"))]
    buf_filtered: FixedVec<i16, OPUS_MAX_FRAME>,
    #[cfg(feature = "heap")]
    buf_filtered: Box<FixedVec<i16, OPUS_MAX_FRAME>>,
    #[cfg(not(feature = "heap"))]
    buf_silk_input: FixedVec<i16, OPUS_MAX_FRAME>,
    #[cfg(feature = "heap")]
    buf_silk_input: Box<FixedVec<i16, OPUS_MAX_FRAME>>,
    /// Right channel of a stereo frame at the SILK internal rate, beside the
    /// left in `buf_silk_input`.
    #[cfg(not(feature = "heap"))]
    buf_silk_input_right: FixedVec<i16, SILK_MAX_PACKET_SAMPLES>,
    #[cfg(feature = "heap")]
    buf_silk_input_right: Box<FixedVec<i16, SILK_MAX_PACKET_SAMPLES>>,
    /// One channel of a stereo frame at the API rate, de-interleaved for its
    /// SILK resampler.
    #[cfg(not(feature = "heap"))]
    buf_silk_api_channel: FixedVec<i16, OPUS_MAX_FRAME>,
    #[cfg(feature = "heap")]
    buf_silk_api_channel: Box<FixedVec<i16, OPUS_MAX_FRAME>>,
    #[cfg(not(feature = "heap"))]
    buf_celt_input: FixedVec<f32, OPUS_MAX_FRAME>,
    #[cfg(feature = "heap")]
    buf_celt_input: Box<FixedVec<f32, OPUS_MAX_FRAME>>,
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
    /// f32 staging of the current frame for the CELT/Hybrid float pipeline
    /// when the caller feeds `encode_i16()` (issue #28).
    #[cfg(not(feature = "heap"))]
    buf_f32_frame: FixedVec<f32, OPUS_MAX_FRAME>,
    #[cfg(feature = "heap")]
    buf_f32_frame: Box<FixedVec<f32, OPUS_MAX_FRAME>>,
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
    /// Encoder-direction SILK input resampler (C `resampler_state` with
    /// `forEnc = 1`): API rate -> SILK internal rate, carrying the
    /// `delay_matrix_enc` input delay and the private down-FIR chain. It
    /// resamples mono input, or the left channel of stereo input.
    silk_resampler_enc: silk::resampler::SilkResampler,
    /// The right channel's SILK input resampler: libopus resamples each
    /// stereo channel separately and splits mid/side at the internal rate.
    silk_resampler_enc_right: silk::resampler::SilkResampler,

    rc: RangeCoder,
}

/// Format-agnostic PCM input for `encode_impl` (issue #28): `F32` is the
/// classic float path; `I16` reaches the SILK integer pipeline directly
/// and is converted to f32 only where the CELT pipeline requires it.
#[derive(Clone, Copy)]
enum PcmInput<'a> {
    F32(&'a [f32]),
    I16(&'a [i16]),
}

impl PcmInput<'_> {
    fn len(&self) -> usize {
        match self {
            PcmInput::F32(s) => s.len(),
            PcmInput::I16(s) => s.len(),
        }
    }
}

/// libopus `silk_setup_resamplers` (control_codec.c:134) for a running SILK
/// encoder that changes its internal rate to `new_khz`: the analysis history
/// `x_buf` is resampled to the input rate and from there to the new rate.
/// The second step runs through the encoder's own input resampler, freshly
/// set up for the new rate, so that its filter state continues from the
/// history too.
fn silk_carry_history_to_rate(
    enc: &mut SilkEncoderState,
    resampler: &mut silk::resampler::SilkResampler,
    api_hz: i32,
    new_khz: i32,
) -> Result<(), &'static str> {
    let old_khz = enc.s_cmn.fs_khz;
    // 2 frames of history and the noise-shaping look-ahead: 45 ms, or 25 ms
    // for 10 ms packets.
    let buf_length_ms = ((enc.s_cmn.nb_subfr * 5) << 1) + silk::define::LA_SHAPE_MS as i32;
    let old_n = (buf_length_ms * old_khz) as usize;
    let new_n = (buf_length_ms * new_khz) as usize;
    let api_n = (buf_length_ms * (api_hz / 1000)) as usize;
    let mut at_api_rate = [0i16; 45 * 48];
    let mut at_new_rate = [0i16; 45 * 16];
    if old_n > enc.s_cmn.x_buf.len() || api_n > at_api_rate.len() || new_n > at_new_rate.len() {
        return Err("Unsupported SILK rate switch");
    }
    let mut to_api = silk::resampler::SilkResampler::default();
    if to_api.init(old_khz * 1000, api_hz) != 0
        || to_api.process(&mut at_api_rate[..api_n], &enc.s_cmn.x_buf[..old_n], old_n as i32) != 0
        || resampler.init_for_enc(api_hz, new_khz * 1000) != 0
        || resampler.process(&mut at_new_rate[..new_n], &at_api_rate[..api_n], api_n as i32) != 0
    {
        return Err("Unsupported SILK rate switch");
    }
    enc.s_cmn.x_buf[..new_n].copy_from_slice(&at_new_rate[..new_n]);
    Ok(())
}

/// libopus `gain_fade` (opus_encoder.c:581): scale an interleaved frame by a
/// gain that moves from `g1` to `g2` over CELT's overlap (2.5 ms), following
/// the squared CELT window, and stays at `g2` after it.
fn gain_fade(pcm: &mut [f32], g1: f32, g2: f32, frame_size: usize, channels: usize, fs: i32) {
    let window = modes::default_mode().window;
    let inc = (48000 / fs).max(1) as usize;
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

fn compute_mode_threshold(
    application: Application,
    channels: usize,
    prev_was_celt: bool,
    has_prev_mode: bool,
    voice_est: i32,
) -> i32 {
    let mode_voice = if channels == 1 { 64000 } else { 44000 };
    let mode_music = 10000;

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

/// C `bits_to_bitrate` (opus_private.h, libopus 1.6): bits per frame to b/s,
/// through the 6x frame rate so 60 ms frames (16.67 Hz) round like libopus.
fn bits_to_bitrate(bits: i32, fs: i32, frame_size: usize) -> i32 {
    (bits as i64 * (6 * fs as i64 / frame_size as i64) / 6) as i32
}

/// C `bitrate_to_bits` (opus_private.h, libopus 1.6): b/s to bits per frame.
fn bitrate_to_bits(bitrate: i32, fs: i32, frame_size: usize) -> i32 {
    (bitrate as i64 * 6 / (6 * fs as i64 / frame_size as i64)) as i32
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

/// Uniformly borrow a codec-state field that is a `Box<T>` under the `heap`
/// feature and a plain `T` otherwise, so call sites don't need `#[cfg]` on
/// every `&self.field` / `&mut self.field`.
///
/// Under `heap` the field is a `Box<...>`, so `T: Deref(DerefMut)` resolves to
/// the box itself and the `Target` is the inner type; without `heap` the field
/// is the inner type directly (`?Sized` lets `&mut [i16]`-style coercion choose
/// the slice target).
#[cfg(feature = "heap")]
#[inline(always)]
fn state_ref<T: core::ops::Deref>(b: &T) -> &T::Target {
    b
}
#[cfg(not(feature = "heap"))]
#[inline(always)]
fn state_ref<T: ?Sized>(b: &T) -> &T {
    b
}

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

        #[cfg(feature = "heap")]
        let mut silk_enc = Box::new(SilkEncoderState::default());
        #[cfg(not(feature = "heap"))]
        let mut silk_enc = SilkEncoderState::default();
        if silk_init_encoder(state_mut(&mut silk_enc), 0) != 0 {
            return Err("SILK encoder initialization failed");
        }
        #[cfg(feature = "heap")]
        let mut silk_enc_side = Box::new(SilkEncoderState::default());
        #[cfg(not(feature = "heap"))]
        let mut silk_enc_side = SilkEncoderState::default();
        if silk_init_encoder(state_mut(&mut silk_enc_side), 0) != 0 {
            return Err("SILK encoder initialization failed");
        }

        let (opus_mode, bw) = match application {
            Application::Voip => {
                let bw = match sampling_rate {
                    8000 => Bandwidth::Narrowband,
                    12000 => Bandwidth::Mediumband,
                    16000 => Bandwidth::Wideband,
                    24000 => Bandwidth::Superwideband,
                    48000 => Bandwidth::Fullband,
                    _ => Bandwidth::Narrowband,
                };

                let mode = if sampling_rate > 16000 {
                    OpusMode::Hybrid
                } else {
                    OpusMode::SilkOnly
                };
                (mode, bw)
            }
            Application::RestrictedLowDelay => {
                let bw = match sampling_rate {
                    8000 => Bandwidth::Narrowband,
                    12000 => Bandwidth::Mediumband,
                    16000 => Bandwidth::Wideband,
                    24000 => Bandwidth::Superwideband,
                    _ => Bandwidth::Fullband,
                };
                (OpusMode::CeltOnly, bw)
            }
            Application::Audio => {
                if sampling_rate <= 16000 {
                    let bw = match sampling_rate {
                        8000 => Bandwidth::Narrowband,
                        12000 => Bandwidth::Mediumband,
                        _ => Bandwidth::Wideband,
                    };
                    (OpusMode::SilkOnly, bw)
                } else {
                    let bw = match sampling_rate {
                        24000 => Bandwidth::Superwideband,
                        _ => Bandwidth::Fullband,
                    };
                    (OpusMode::Hybrid, bw)
                }
            }
        };

        use silk::lin2log::silk_lin2log;
        let variable_hp_smth2_q15 = silk_lin2log(60) << 8;

        Ok(Self {
            celt_enc,
            silk_enc,
            silk_enc_side,
            application,
            sampling_rate,
            channels,
            bandwidth: bw,
            auto_bandwidth: bw,
            user_bandwidth: None,
            first: true,
            silk_allow_bandwidth_switch: false,
            silk_time_since_switch_allowed_ms: 0,
            silk_in_wb_mode_without_variable_lp: false,
            silk_opus_can_switch: false,
            nonfinal_frame: false,
            bitrate_bps: 64000,
            complexity: 9,
            use_cbr: false,
            use_inband_fec: false,
            packet_loss_perc: 0,
            silk_initialized: false,
            prev_enc_mode: None,
            forced_frame: None,
            mode: opus_mode,
            variable_hp_smth2_q15,
            hp_mem: FixedVec::from_value(0, channels * 2),

            #[cfg(not(feature = "heap"))]
            buf_filtered: FixedVec::new(),
            #[cfg(feature = "heap")]
            buf_filtered: Box::new(FixedVec::new()),
            #[cfg(not(feature = "heap"))]
            buf_silk_input: FixedVec::new(),
            #[cfg(feature = "heap")]
            buf_silk_input: Box::new(FixedVec::new()),
            #[cfg(not(feature = "heap"))]
            buf_silk_input_right: FixedVec::new(),
            #[cfg(feature = "heap")]
            buf_silk_input_right: Box::new(FixedVec::new()),
            #[cfg(not(feature = "heap"))]
            buf_silk_api_channel: FixedVec::new(),
            #[cfg(feature = "heap")]
            buf_silk_api_channel: Box::new(FixedVec::new()),
            #[cfg(not(feature = "heap"))]
            buf_celt_input: FixedVec::new(),
            #[cfg(feature = "heap")]
            buf_celt_input: Box::new(FixedVec::new()),
            #[cfg(not(feature = "heap"))]
            delay_buffer: FixedVec::from_value(0.0, OPUS_DELAY_BUF),
            #[cfg(feature = "heap")]
            delay_buffer: Box::new(FixedVec::from_value(0.0, OPUS_DELAY_BUF)),
            #[cfg(not(feature = "heap"))]
            buf_celt_pcm: FixedVec::new(),
            #[cfg(feature = "heap")]
            buf_celt_pcm: Box::new(FixedVec::new()),
            #[cfg(not(feature = "heap"))]
            buf_f32_frame: FixedVec::new(),
            #[cfg(feature = "heap")]
            buf_f32_frame: Box::new(FixedVec::new()),
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
            silk_resampler_enc: silk::resampler::SilkResampler::default(),
            silk_resampler_enc_right: silk::resampler::SilkResampler::default(),
            rc: RangeCoder::new_encoder(1),
        })
    }

    /// The decisions libopus's `opus_encode_native` makes for a packet
    /// (opus_encoder.c:1408-1695): the mode, the audio bandwidth and whether
    /// the packet carries in-band FEC. `n_bytes` is the largest packet
    /// allowed.
    ///
    /// Not ported: the speech/music and bandwidth detection of libopus's
    /// tonality analysis (it runs from complexity 7), the redundant CELT
    /// frames around a switch to or from CELT-only, and the stereo to mono
    /// downmix at low bitrates.
    fn decide_mode_and_bandwidth(
        &mut self,
        frame_size: usize,
        frame_rate: i32,
        n_bytes: usize,
    ) -> (OpusMode, Bandwidth, i32) {
        // C 1327-1334: CBR codes whole bytes, so its bitrate is the packet's.
        let bitrate = if self.use_cbr {
            bits_to_bitrate(8 * n_bytes as i32, self.sampling_rate, frame_size)
        } else {
            self.bitrate_bps
        };
        // C 1407: the rate the largest packet allowed stands for.
        let max_rate = bits_to_bitrate(8 * n_bytes as i32, self.sampling_rate, frame_size);
        // C 1413-1426: how likely the input is speech, in Q7. Without the
        // tonality analysis this depends on the application alone.
        let voice_est = if self.application == Application::Voip { 115 } else { 48 };

        // --- Mode: SILK or CELT (C 1466-1539) ---
        let mut mode = if self.application == Application::RestrictedLowDelay {
            OpusMode::CeltOnly
        } else {
            let equiv = compute_equiv_rate(
                bitrate,
                self.channels,
                frame_rate,
                !self.use_cbr,
                None,
                self.complexity,
                self.packet_loss_perc,
            );
            let threshold = compute_mode_threshold(
                self.application,
                self.channels,
                self.prev_enc_mode == Some(OpusMode::CeltOnly),
                self.prev_enc_mode.is_some(),
                voice_est,
            );
            let mut mode = if equiv >= threshold && self.sampling_rate >= 24000 {
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
            // it as a small valid packet instead (issue #46).
            let tiny_budget_rate: i64 = if frame_rate > 50 { 9000 } else { 6000 };
            if ((n_bytes as i64) * 8 * (self.sampling_rate as i64))
                < (tiny_budget_rate * (frame_size as i64))
            {
                mode = OpusMode::CeltOnly;
            }
            mode
        };
        // C 1532-1537: SILK frames are 10 ms or longer; 2.5 and 5 ms frames
        // are CELT-only.
        if mode != OpusMode::CeltOnly && frame_size < self.sampling_rate as usize / 100 {
            mode = OpusMode::CeltOnly;
        }

        // C 1573: the rate the next decisions go by counts the mode.
        let equiv_rate = compute_equiv_rate(
            bitrate,
            self.channels,
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
            self.silk_enc.s_cmn.lbrr_enabled,
            mode,
            &mut self.bandwidth,
            equiv_rate,
        );

        // C 1680-1695, after decide_fec: CELT has no mediumband; SILK above
        // wideband is Hybrid; Hybrid at wideband or below is SILK-only.
        // Running this before decide_fec left Hybrid with a wideband frame
        // when FEC lowered the bandwidth: CELT then had no band to code and
        // divided by zero.
        if mode == OpusMode::CeltOnly && self.bandwidth == Bandwidth::Mediumband {
            self.bandwidth = Bandwidth::Wideband;
        }
        let above_wideband = bandwidth_fec_index(self.bandwidth) > 2;
        if mode == OpusMode::SilkOnly && above_wideband {
            mode = OpusMode::Hybrid;
        }
        if mode == OpusMode::Hybrid && !above_wideband {
            mode = OpusMode::SilkOnly;
        }
        (mode, self.bandwidth, lbrr_coded)
    }

    /// The rate SILK codes this frame at, in kHz: libopus
    /// `silk_control_audio_bandwidth` (control_audio_bandwidth.c) with the
    /// limits `opus_encode_frame_native` sets (opus_encoder.c:2112-2147).
    ///
    /// A SILK encoder that starts, or restarts after CELT-only frames, takes
    /// the rate of the requested bandwidth. A running one moves towards it
    /// one step at a time (8, 12, 16 kHz), and only when SILK allows a
    /// switch:
    ///
    /// - Down: SILK first closes its input low-pass filter over 2.56 s, then
    ///   reports that it is ready, and switches on the next frame.
    /// - Up: SILK reports that it is ready, switches on the next frame, and
    ///   opens the low-pass filter over 5.12 s.
    ///
    /// Returns the rate and whether SILK is ready to switch on the next
    /// frame (libopus `switchReady`). Not ported: the redundant CELT frames
    /// libopus adds around the switch when the packet has room for them.
    fn silk_internal_rate_khz(
        &mut self,
        mode: OpusMode,
        bandwidth: Bandwidth,
        n_bytes: usize,
        frame_size: usize,
        frame_rate: i32,
    ) -> (i32, bool) {
        let api_khz = self.sampling_rate / 1000;
        let mut desired_khz = match bandwidth {
            Bandwidth::Narrowband => 8,
            Bandwidth::Mediumband => 12,
            _ => 16,
        };
        // Hybrid's CELT layer starts at 8 kHz, so its SILK layer is wideband.
        let min_khz = if mode == OpusMode::Hybrid { 16 } else { 8 };
        let mut max_khz = 16;
        if mode == OpusMode::SilkOnly {
            let mut effective_max_rate =
                bits_to_bitrate(8 * n_bytes as i32, self.sampling_rate, frame_size);
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
        let restarting = !self.silk_initialized || self.prev_enc_mode == Some(OpusMode::CeltOnly);
        let current_khz = if restarting { 0 } else { self.silk_enc.s_cmn.fs_khz };
        if current_khz == 0 {
            // The encoder has just been initialized.
            return (desired_khz.min(api_khz), false);
        }
        if current_khz > api_khz || current_khz > max_khz || current_khz < min_khz {
            // Not above the input rate or the maximum, not below the minimum.
            return (api_khz.min(max_khz).max(min_khz), false);
        }

        // The state machine of the rate switch. Each SILK channel keeps its
        // own low-pass state; both follow the same steps.
        let allowed = self.silk_allow_bandwidth_switch;
        let can_switch = self.silk_opus_can_switch;
        let mut fs_khz = current_khz;
        let mut switch_ready = false;
        for enc in [&mut self.silk_enc, &mut self.silk_enc_side]
            .into_iter()
            .take(self.channels)
        {
            let lp = &mut enc.s_cmn.s_lp;
            if lp.transition_frame_no >= silk::define::TRANSITION_FRAMES {
                // Stop the transition phase.
                lp.mode = 0;
            }
            if !(allowed || can_switch) {
                continue;
            }
            if current_khz > desired_khz {
                // Switch down.
                if lp.mode == 0 {
                    // New transition.
                    lp.transition_frame_no = silk::define::TRANSITION_FRAMES;
                    lp.in_lp_state = [0; 2];
                }
                if can_switch {
                    lp.mode = 0;
                    fs_khz = if current_khz == 16 { 12 } else { 8 };
                } else if lp.transition_frame_no <= 0 {
                    switch_ready = true;
                } else {
                    // Direction: down, at double speed.
                    lp.mode = -2;
                }
            } else if current_khz < desired_khz {
                // Switch up.
                if can_switch {
                    fs_khz = if current_khz == 8 { 12 } else { 16 };
                    // New transition, direction: up.
                    lp.transition_frame_no = 0;
                    lp.in_lp_state = [0; 2];
                    lp.mode = 1;
                } else if lp.mode == 0 {
                    switch_ready = true;
                } else {
                    lp.mode = 1;
                }
            } else if lp.mode < 0 {
                lp.mode = 1;
            }
        }
        (fs_khz, switch_ready)
    }

    pub fn enable_hybrid_mode(&mut self) -> Result<(), &'static str> {
        if self.sampling_rate != 24000 && self.sampling_rate != 48000 {
            return Err("Hybrid mode requires 24kHz or 48kHz sampling rate");
        }
        let bw = if self.sampling_rate == 48000 {
            Bandwidth::Fullband
        } else {
            Bandwidth::Superwideband
        };
        self.mode = OpusMode::Hybrid;
        self.bandwidth = bw;
        // Like libopus's OPUS_SET_BANDWIDTH: the automatic, rate-dependent
        // choice no longer applies. Low bitrates then stay Hybrid instead of
        // dropping to wideband SILK.
        self.user_bandwidth = Some(bw);
        self.silk_initialized = false;
        Ok(())
    }

    /// Encode a frame of interleaved f32 PCM (scale ±1.0; C `opus_encode_float`
    /// convention where 1.0 maps to PCM16 32768). Samples are consumed exactly
    /// up to `frame_size * channels`.
    pub fn encode(
        &mut self,
        input: &[f32],
        frame_size: usize,
        output: &mut [u8],
    ) -> Result<usize, &'static str> {
        self.encode_impl(PcmInput::F32(input), frame_size, output)
    }

    /// Encode a frame of interleaved PCM16 (`opus_int16` C `opus_encode()`
    /// convention). Produces byte-for-byte identical packets to
    /// `encode()` called with the same audio converted via
    /// `sample as f32 / 32768.0`, while skipping the f32 round-trip in the
    /// SILK/VoIP path: the integer high-pass biquad and SILK pipeline are fed
    /// natively, with no temporary conversion buffer (issue #28).
    pub fn encode_i16(
        &mut self,
        input: &[i16],
        frame_size: usize,
        output: &mut [u8],
    ) -> Result<usize, &'static str> {
        self.encode_impl(PcmInput::I16(input), frame_size, output)
    }

    fn encode_impl(
        &mut self,
        input: PcmInput<'_>,
        frame_size: usize,
        output: &mut [u8],
    ) -> Result<usize, &'static str> {
        if output.len() < 2 {
            return Err("Output buffer too small");
        }

        let frame_rate = frame_rate_from_params(self.sampling_rate, frame_size)
            .ok_or("Invalid frame size for sampling rate")?;

        // Clamp caller-provided control fields the way libopus's ctl
        // interface does (OPUS_SET_BITRATE clamps to [500, 3000000];
        // complexity to [0, 10]). Unclamped extremes overflow the bitrate
        // arithmetic downstream (bitrate*6 in celt, `equiv*(90+complexity)`
        // in the mode selection) — issue #27 deep scan.
        self.bitrate_bps = self.bitrate_bps.clamp(500, 3_000_000);
        self.complexity = self.complexity.clamp(0, 10);
        self.packet_loss_perc = self.packet_loss_perc.clamp(0, 100);

        // The encoder consumes exactly frame_size * channels samples per call;
        // reject short input up front instead of panicking inside the HP filter
        // or the float→int conversion loops (issue #27 deep-scan).
        if input.len() < frame_size * self.channels {
            return Err("Input buffer too small for frame");
        }

        // Per-frame packet budget (C computes this before the mode-dependent
        // encode; the starved-budget rules below need it too).
        let target_bits =
            (self.bitrate_bps as i64 * frame_size as i64 / self.sampling_rate as i64) as i32;
        let cbr_bytes = ((target_bits + 4) / 8) as usize;
        let max_data_bytes = output.len();

        // C parity (opus_encoder.c): the nominal-size cap applies in CBR mode
        // only. In VBR mode the CELT-internal bound (vbr_rate/reservoir) decides
        // the per-frame size, allowing overshoot (borrowing) and undershoot.
        // Capping VBR at nominal defeats the reservoir and starves complex frames.
        let mut n_bytes = if self.use_cbr {
            cbr_bytes
                .min(max_data_bytes)
                .clamp(1, OPUS_MAX_PACKET_BYTES)
        } else {
            max_data_bytes.clamp(1, OPUS_MAX_PACKET_BYTES)
        };

        // Mode selection: match C's opus_encode_native() behavior.
        // C reference auto-selects between SILK_ONLY and CELT_ONLY; Hybrid is
        // produced afterwards by bandwidth overrides (SILK-only + FB/SWB → Hybrid).
        let (mode, frame_bandwidth, lbrr_coded) = if let Some(forced) = self.forced_frame {
            // A 20 ms part of a 40/60 ms packet: the packet's decisions hold.
            (forced.mode, forced.bandwidth, forced.lbrr_coded)
        } else {
            self.decide_mode_and_bandwidth(frame_size, frame_rate, n_bytes)
        };

        // 40/60 ms frames are only legal as a single Opus frame in SILK-only
        // mode. CELT and Hybrid cap a frame at 20 ms, so libopus encodes 20 ms
        // sub-frames and repacketizes them into one code 1/2/3 packet
        // (opus_encoder.c `encode_multiframe_packet`). (issue #35)
        if self.forced_frame.is_none()
            && mode != OpusMode::SilkOnly
            && frame_size > self.sampling_rate as usize / 50
        {
            let decisions = ForcedFrame {
                mode,
                bandwidth: frame_bandwidth,
                lbrr_coded,
            };
            return self.encode_multiframe(input, frame_size, decisions, output);
        }

        if mode == OpusMode::CeltOnly {
            // silk_mode.LBRR_coded persists across frames even while SILK is
            // idle; it feeds decide_fec's hysteresis on the next SILK frame.
            self.silk_enc.s_cmn.lbrr_enabled = 0;
            self.silk_enc_side.s_cmn.lbrr_enabled = 0;
        }

        if mode == OpusMode::CeltOnly {
            match frame_rate {
                400 | 200 | 100 | 50 => {}
                _ => return Err("Unsupported frame size for CELT-only mode"),
            }
        }

        if mode == OpusMode::Hybrid {
            match frame_rate {
                100 | 50 => {}
                _ => return Err("Unsupported frame size for Hybrid mode"),
            }
        }

        if mode == OpusMode::SilkOnly {
            match frame_rate {
                100 | 50 | 25 | 16 => {}
                // 2.5/5 ms are CELT-only durations: SILK frames are 10 ms or
                // longer (libopus), and sub-10 ms frames break the SILK
                // analysis geometry (e.g. the VAD band split at 12 kHz).
                _ => return Err("Unsupported frame size for SILK-only mode"),
            }
        }

        // The rate SILK codes this frame at. A SILK-only packet signals that
        // rate's bandwidth, not the one asked for (C opus_encoder.c:2223-2232):
        // SILK does not follow every bandwidth request at once, and a TOC
        // that names another rate makes the decoder read the frame wrongly.
        let (silk_fs_khz, silk_switch_ready) = if mode == OpusMode::CeltOnly {
            (0, false)
        } else {
            self.silk_internal_rate_khz(mode, frame_bandwidth, n_bytes, frame_size, frame_rate)
        };
        let toc_bandwidth = match (mode, silk_fs_khz) {
            (OpusMode::SilkOnly, 8) => Bandwidth::Narrowband,
            (OpusMode::SilkOnly, 12) => Bandwidth::Mediumband,
            (OpusMode::SilkOnly, _) => Bandwidth::Wideband,
            _ => frame_bandwidth,
        };
        let toc = gen_toc(mode, frame_rate, toc_bandwidth, self.channels);
        output[0] = toc;

        // C opus_encoder.c:1226-1243: a budget too small to code anything
        // useful produces a TOC-only "PLC" packet — decoders conceal it as a
        // loss. Encoding into 0-2 bytes would otherwise overflow the range
        // coder (issue #45). Under CBR the packet is padded to the nominal
        // size, exactly like libopus's opus_packet_pad.
        let sub_3_bytes_per_frame =
            (self.bitrate_bps as i64) < 3 * frame_rate as i64 * 8;
        let tiny_long_frame =
            frame_rate < 50
                && (n_bytes * (frame_rate as usize) < 300 || self.bitrate_bps < 2400);
        if n_bytes < 3 || sub_3_bytes_per_frame || tiny_long_frame {
            let mut toc_mode = mode;
            let mut packet_code = 0u8;
            let mut num_multiframes = 0u8;
            let mut frame_rate_adj = frame_rate;
            if frame_rate > 100 {
                toc_mode = OpusMode::CeltOnly;
            }
            // 40 ms -> 2 x 20 ms in CELT/hybrid mode.
            if frame_rate == 25 && toc_mode != OpusMode::SilkOnly {
                frame_rate_adj = 50;
                packet_code = 1;
            }
            // >= 60 ms frames: 1 x 60 ms, or 2 x 40 ms / 2 x 60 ms multiframes.
            if frame_rate <= 16 {
                if n_bytes == 1 || (toc_mode == OpusMode::SilkOnly && frame_rate != 10) {
                    toc_mode = OpusMode::SilkOnly;
                    packet_code = if frame_rate <= 12 { 1 } else { 0 };
                    frame_rate_adj = if frame_rate == 12 { 25 } else { 16 };
                } else {
                    num_multiframes = (50 / frame_rate) as u8;
                    frame_rate_adj = 50;
                    packet_code = 3;
                }
            }
            let mut plc_bw = frame_bandwidth;
            if toc_mode == OpusMode::SilkOnly && bandwidth_fec_index(plc_bw) > 2 {
                // SilkOnly is wideband at most (C: bw > OPUS_BANDWIDTH_WIDEBAND).
                plc_bw = Bandwidth::Wideband;
            }
            output[0] = gen_toc(toc_mode, frame_rate_adj, plc_bw, self.channels) | packet_code;
            let mut ret = if packet_code <= 1 { 1 } else { 2 };
            if packet_code == 3 {
                output[1] = num_multiframes;
            }
            if self.use_cbr {
                let target_total = n_bytes.min(output.len());
                if target_total >= 2 {
                    // opus_packet_pad of the TOC-only packet: code 3 with
                    // `m` empty frames and zero padding.
                    let m: usize = if packet_code == 3 {
                        num_multiframes as usize
                    } else if packet_code == 1 {
                        2
                    } else {
                        1
                    };
                    output[0] = (output[0] & 0xFC) | 0x03;
                    // CBR, m frames. The "padding follows" bit is set only
                    // when padding does follow (C repacketizer.c:259-290:
                    // `pad_amount != 0`). A two-byte packet has none, and
                    // with the bit set a decoder rejects it as corrupt.
                    output[1] = m as u8;
                    if target_total > 2 {
                        output[1] |= 0x40;
                        let pad_amount = target_total - 2;
                        let nb_255s = (pad_amount - 1) / 255;
                        let mut ptr = 2;
                        for _ in 0..nb_255s {
                            output[ptr] = 255;
                            ptr += 1;
                        }
                        output[ptr] = (pad_amount - 255 * nb_255s - 1) as u8;
                        ptr += 1;
                        for b in &mut output[ptr..target_total] {
                            *b = 0;
                        }
                    }
                    ret = target_total;
                }
            }
            // C leaves st->prev_mode untouched on this path.
            return Ok(ret);
        }

        let init_rc_size = n_bytes - 1;
        self.rc.reset_for_encode(init_rc_size as u32);
        // C opus_encoder.c:2562: from here on a frame is coded. The packets
        // above, too small to code anything, leave `first` alone.
        self.first = false;

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
            self.silk_enc.s_cmn.variable_hp_smth1_q15
        };

        const VARIABLE_HP_SMTH_COEF2_Q16: i32 = 984;
        self.variable_hp_smth2_q15 = silk_smlawb(
            self.variable_hp_smth2_q15,
            hp_freq_smth1 - self.variable_hp_smth2_q15,
            VARIABLE_HP_SMTH_COEF2_Q16,
        );

        let cutoff_hz = silk_log2lin(silk_rshift(self.variable_hp_smth2_q15, 8));

        if mode == OpusMode::SilkOnly || mode == OpusMode::Hybrid {
            let frame_ms = (frame_size as i32 * 1000) / self.sampling_rate;

            // C opus_encoder.c `bits_target` / `total_bitRate` (1.6): SILK
            // codes at the requested bitrate, capped by the packet budget,
            // less the TOC byte. Under VBR the budget is the caller's output
            // buffer, which must cap the rate, not set it (issue #43). CBR
            // first re-quantizes the bitrate to the whole bytes it codes, as
            // opus_encode_native does, so the cap is the packet itself.
            let rate_bps = if self.use_cbr {
                bits_to_bitrate(8 * n_bytes as i32, self.sampling_rate, frame_size)
            } else {
                self.bitrate_bps
            };
            let rate_bits = bitrate_to_bits(rate_bps, self.sampling_rate, frame_size);
            let bits_target = (8 * n_bytes as i32).min(rate_bits) - 8;
            let total_bitrate = bits_to_bitrate(bits_target, self.sampling_rate, frame_size);

            // C opus_encoder.c:1448-1455: (re)entering SILK/hybrid from CELT
            // resets the SILK encoder and prefills it with recent (ramped)
            // audio, so the first SILK frame starts from valid history.
            let silk_restart_after_celt =
                self.prev_enc_mode == Some(OpusMode::CeltOnly);
            if silk_restart_after_celt {
                silk_init_encoder(state_mut(&mut self.silk_enc), 0);
                // C silk_InitEncoder also clears the packet-level rate state.
                self.silk_enc.n_bits_used_lbrr = 0;
                self.silk_enc.n_bits_exceeded = 0;
                self.silk_time_since_switch_allowed_ms = 0;
                // C opus_encoder.c:2208: no rate switch right after a restart.
                self.silk_opus_can_switch = false;
                // C silk_InitEncoder clears the whole state. A low-pass
                // transition left running from before the CELT-only frames
                // would keep filtering the input and hold the stream at
                // wideband.
                self.silk_enc.s_cmn.s_lp = Default::default();
                if self.channels == 2 {
                    // C silk_InitEncoder clears the side encoder and the
                    // stereo state too; the next stereo frame starts afresh.
                    *state_mut(&mut self.silk_enc_side) = SilkEncoderState::default();
                    silk_init_encoder(state_mut(&mut self.silk_enc_side), 0);
                    self.silk_enc.stereo.reset();
                }
                self.silk_initialized = false;
            }
            let new_rate = !self.silk_initialized || self.silk_enc.s_cmn.fs_khz != silk_fs_khz;
            // A running SILK encoder that changes its internal rate takes its
            // analysis history along (C silk_setup_resamplers).
            let running_switch = self.silk_initialized && new_rate;
            if running_switch {
                silk_carry_history_to_rate(
                    state_mut(&mut self.silk_enc),
                    &mut self.silk_resampler_enc,
                    self.sampling_rate,
                    silk_fs_khz,
                )?;
                if self.channels == 2 {
                    silk_carry_history_to_rate(
                        state_mut(&mut self.silk_enc_side),
                        &mut self.silk_resampler_enc_right,
                        self.sampling_rate,
                        silk_fs_khz,
                    )?;
                }
            }
            // C silk_control_encoder runs every packet and follows a change
            // of packet duration (control_codec.c silk_setup_fs). A switch
            // between SILK-only and Hybrid changes it for 40/60 ms input:
            // SILK-only codes one 40/60 ms packet, Hybrid 20 ms parts.
            let new_packet_size = self.silk_enc.s_cmn.packet_size_ms != frame_ms;
            if new_rate || new_packet_size {
                silk_control_encoder(
                    state_mut(&mut self.silk_enc),
                    silk_fs_khz,
                    frame_ms,
                    total_bitrate,
                    self.complexity,
                );
                self.silk_enc.s_cmn.use_cbr = if self.use_cbr { 1 } else { 0 };

                self.silk_enc.s_cmn.n_channels = self.channels as i32;
                self.silk_initialized = true;
                // enc_API.c:207, 267: the redundant frames kept for the next
                // packet belong to the old packet layout.
                self.silk_enc.s_cmn.lbrr_flags = [0; silk::define::MAX_FRAMES_PER_PACKET];
                if new_rate && !running_switch {
                    // C silk_setup_resamplers: a fresh encoder initializes
                    // the encoder-direction resampler; the input delay comes
                    // from delay_matrix_enc.
                    let ret = self
                        .silk_resampler_enc
                        .init_for_enc(self.sampling_rate as i32, silk_fs_khz * 1000);
                    if ret != 0 {
                        return Err("Unsupported SILK resampling ratio");
                    }
                }
                if self.channels == 2 {
                    if new_rate && !running_switch {
                        self.silk_resampler_enc_right = self.silk_resampler_enc.clone();
                    }
                    // The side runs at the mid's rate and frame size (C
                    // silk_control_encoder with force_fs_kHz).
                    let side = state_mut(&mut self.silk_enc_side);
                    silk_control_encoder(
                        side,
                        silk_fs_khz,
                        frame_ms,
                        total_bitrate,
                        self.complexity,
                    );
                    side.s_cmn.use_cbr = self.silk_enc.s_cmn.use_cbr;
                    side.s_cmn.n_channels = 2;
                    side.s_cmn.lbrr_flags = [0; silk::define::MAX_FRAMES_PER_PACKET];
                }
            }

            // C silk_setup_LBRR (control_codec.c): per-packet FEC state and
            // LBRR gain increase (7 on the first FEC packet, then
            // max(7 - 0.2*loss, 2)), for each coded channel.
            for enc in [&mut self.silk_enc, &mut self.silk_enc_side]
                .into_iter()
                .take(self.channels)
            {
                enc.s_cmn.use_in_band_fec = if self.use_inband_fec { 1 } else { 0 };
                enc.s_cmn.packet_loss_perc = self.packet_loss_perc.clamp(0, 100);
                silk_setup_lbrr(state_mut(enc), lbrr_coded);
            }

            let silk_frame_len_samples =
                frame_size * silk_fs_khz as usize / (self.sampling_rate.max(1000) / 1000) as usize;
            let silk_bitrate = if mode == OpusMode::Hybrid {
                // C opus_encoder.c:2049: SILK's base rate out of the rate
                // the packet codes at.
                let rate = compute_silk_rate_for_hybrid(
                    total_bitrate,
                    frame_bandwidth,
                    frame_ms == 20,
                    !self.use_cbr,
                    lbrr_coded != 0,
                    self.channels,
                );
                // C opus_encoder.c:2054-2055: the fewer bits the high band
                // gets, the more it is attenuated before CELT codes it.
                let celt_rate = total_bitrate - rate;
                hb_gain = 1.0 - 0.5 * (-celt_rate as f32 / 1024.0).exp2();
                hybrid_silk_rate = rate;
                rate
            } else {
                // C: SILK gets all bits, at the requested rate (issue #43).
                total_bitrate
            };

            // C opus_encoder.c:1806-1826: prefill the fresh SILK encoder with
            // the tail of the delay buffer, zeroed before a 2.5 ms fade-in,
            // resampled through the fresh encoder resampler so its delay
            // state also starts consistent.
            if silk_restart_after_celt && self.encoder_buffer > 0 && self.delay_compensation > 0
            {
                let ch = self.channels;
                let f2_5 = self.sampling_rate / 400;
                let eb = self.encoder_buffer;
                let offset = eb - self.delay_compensation.min(eb) - f2_5 as usize;
                let pre_resampled_len = eb * (silk_fs_khz * 1000) as usize / self.sampling_rate as usize;
                // Stack-fixed: encoder_buffer = Fs/100 <= 480, and the
                // resampled prefill is at most the same length (Copy at equal
                // rates). No allocation: this crate is heap-free without the
                // `heap` feature.
                if ch == 2 {
                    // Each channel through its own resampler, then mid/side
                    // at the internal rate, as C silk_Encode prefills two
                    // internal channels.
                    let (mut pre_l, mut pre_r) = ([0i16; 480], [0i16; 480]);
                    if offset < eb {
                        for i in offset..eb {
                            let fade = (((i - offset) as f32) / f2_5 as f32).min(1.0);
                            pre_l[i] = (self.delay_buffer[2 * i] * fade * 32768.0) as i16;
                            pre_r[i] = (self.delay_buffer[2 * i + 1] * fade * 32768.0) as i16;
                        }
                    }
                    let (mut int_l, mut int_r) = ([0i16; 480], [0i16; 480]);
                    let int_l = &mut int_l[..pre_resampled_len];
                    let int_r = &mut int_r[..pre_resampled_len];
                    self.silk_resampler_enc
                        .process(int_l, &pre_l[..eb], eb as i32);
                    self.silk_resampler_enc_right
                        .process(int_r, &pre_r[..eb], eb as i32);
                    silk_encode_prefill_stereo(
                        state_mut(&mut self.silk_enc),
                        state_mut(&mut self.silk_enc_side),
                        int_l,
                        int_r,
                        silk_bitrate,
                        1,
                    );
                } else {
                    let mut pre = [0i16; 480];
                    let pre = &mut pre[..eb];
                    if offset < eb {
                        for i in offset..eb {
                            let fade = (((i - offset) as f32) / f2_5 as f32).min(1.0);
                            pre[i] = (self.delay_buffer[i] * fade * 32768.0) as i16;
                        }
                    }
                    let mut pre_internal = [0i16; 480];
                    let pre_internal = &mut pre_internal[..pre_resampled_len];
                    self.silk_resampler_enc
                        .process(pre_internal, pre, eb as i32);
                    silk_encode_prefill(state_mut(&mut self.silk_enc), pre_internal, 1);
                }
            }

            let required_size = frame_size * self.channels;
            self.buf_filtered.resize(required_size, 0);
            let voip = self.application == Application::Voip;
            match input {
                PcmInput::I16(samples) => {
                    if voip {
                        // Native i16 path: feed the integer HP biquad directly,
                        // skipping the f32 round-trip and its temporary
                        // conversion buffer (issue #28).
                        hp_cutoff_i16(
                            samples,
                            cutoff_hz,
                            state_mut(&mut self.buf_filtered),
                            &mut self.hp_mem,
                            frame_size,
                            self.channels,
                            self.sampling_rate,
                        );
                    } else {
                        let filtered = state_mut(&mut self.buf_filtered);
                        filtered[..required_size].copy_from_slice(&samples[..required_size]);
                    }
                }
                PcmInput::F32(samples) => {
                    if voip {
                        hp_cutoff(
                            samples,
                            cutoff_hz,
                            state_mut(&mut self.buf_filtered),
                            &mut self.hp_mem,
                            frame_size,
                            self.channels,
                            self.sampling_rate,
                        );
                    } else {
                        let filtered = state_mut(&mut self.buf_filtered);
                        for (i, &x) in samples.iter().enumerate().take(required_size) {
                            filtered[i] = (x * 32768.0).clamp(-32768.0, 32767.0) as i16;
                        }
                    }
                }
            }

            let input_i16 = state_ref(&self.buf_filtered);

            // SILK input resampling (C silk_Encode always runs silk_resampler
            // on the API-rate input; the Copy path still applies its
            // delay_matrix_enc input delay at equal rates). Stereo resamples
            // each channel on its own and splits mid/side at the internal
            // rate (enc_API.c, nChannelsInternal 2), in SILK-only and Hybrid
            // alike.
            self.buf_silk_input.resize(silk_frame_len_samples, 0);
            if self.channels == 2 {
                self.buf_silk_input_right.resize(silk_frame_len_samples, 0);
                for c in 0..2 {
                    self.buf_silk_api_channel.resize(frame_size, 0);
                    for (s, lr) in self
                        .buf_silk_api_channel
                        .iter_mut()
                        .zip(input_i16.chunks_exact(2))
                    {
                        *s = lr[c];
                    }
                    let (resampler, out) = if c == 0 {
                        (
                            &mut self.silk_resampler_enc,
                            state_mut(&mut self.buf_silk_input).as_mut_slice(),
                        )
                    } else {
                        (
                            &mut self.silk_resampler_enc_right,
                            state_mut(&mut self.buf_silk_input_right).as_mut_slice(),
                        )
                    };
                    if resampler.process(out, &self.buf_silk_api_channel, frame_size as i32) != 0 {
                        return Err("SILK input resampling failed");
                    }
                }
            } else {
                let got = self.silk_resampler_enc.process(
                    state_mut(&mut self.buf_silk_input),
                    input_i16,
                    frame_size as i32,
                );
                if got != 0 {
                    return Err("SILK input resampling failed");
                }
            }
            let silk_input: &[i16] = state_ref(&self.buf_silk_input);

            let mut pn_bytes = 0;

            let silk_max_bits = if mode == OpusMode::Hybrid {
                let total_max_bits = ((n_bytes - 1) * 8) as i32;
                if self.use_cbr {
                    let silk_bits = (silk_bitrate as i64 * frame_size as i64
                        / self.sampling_rate as i64) as i32;
                    let other_bits = 0i32.max(total_max_bits - silk_bits);
                    0i32.max(total_max_bits - other_bits * 3 / 4)
                } else {
                    // C opus_encoder.c:2176-2181: the SILK rate that goes
                    // with the largest packet allowed.
                    let max_bit_rate = compute_silk_rate_for_hybrid(
                        total_max_bits * self.sampling_rate / frame_size as i32,
                        frame_bandwidth,
                        frame_ms == 20,
                        true,
                        lbrr_coded != 0,
                        self.channels,
                    );
                    bitrate_to_bits(max_bit_rate, self.sampling_rate, frame_size)
                }
            } else {
                ((n_bytes - 1) * 8) as i32
            };
            // control_audio_bandwidth.c:88, 116: a packet that announces a
            // rate switch keeps room for the redundant frame of the switch.
            let silk_max_bits = if silk_switch_ready {
                silk_max_bits - silk_max_bits * 5 / (frame_ms + 5)
            } else {
                silk_max_bits
            };
            let silk_use_cbr = if mode == OpusMode::Hybrid && self.use_cbr {
                0
            } else if self.use_cbr {
                1
            } else {
                0
            };
            let ret = if self.channels == 2 {
                silk_encode_stereo_packet(
                    state_mut(&mut self.silk_enc),
                    state_mut(&mut self.silk_enc_side),
                    silk_input,
                    state_ref(&self.buf_silk_input_right),
                    silk_input.len(),
                    &mut self.rc,
                    &mut pn_bytes,
                    silk_bitrate,
                    silk_max_bits,
                    silk_use_cbr,
                    1,
                )
            } else {
                silk_encode(
                    state_mut(&mut self.silk_enc),
                    silk_input,
                    silk_input.len(),
                    &mut self.rc,
                    &mut pn_bytes,
                    silk_bitrate,
                    silk_max_bits,
                    silk_use_cbr,
                    1,
                )
            };
            if ret != 0 {
                return Err("SILK encoding failed");
            }

            // enc_API.c:559-583: SILK allows a bandwidth switch while its
            // speech activity is under a bar that starts at 5 % and is gone
            // after 5 s (SPEECH_ACTIVITY_DTX_THRES,
            // MAX_BANDWIDTH_SWITCH_DELAY_MS). In a 40/60 ms Hybrid packet
            // only the last 20 ms part leaves its answer, as in libopus.
            let bar_q8 = silk_smlawb(13, 3188, self.silk_time_since_switch_allowed_ms);
            if self.silk_enc.s_cmn.speech_activity_q8 < bar_q8 {
                self.silk_allow_bandwidth_switch = true;
                self.silk_time_since_switch_allowed_ms = 0;
            } else {
                self.silk_allow_bandwidth_switch = false;
                self.silk_time_since_switch_allowed_ms += frame_ms;
            }
            self.silk_in_wb_mode_without_variable_lp =
                self.silk_enc.s_cmn.fs_khz == 16 && self.silk_enc.s_cmn.s_lp.mode == 0;
            // C opus_encoder.c:2233: a rate switch SILK is ready for happens
            // on the next packet, not in the middle of this one.
            self.silk_opus_can_switch = silk_switch_ready && !self.nonfinal_frame;
        }

        // C opus_encoder.c:2351-2355: a Hybrid packet says with one flag that
        // no redundant CELT frame follows, but only when it has room for the
        // flag and for the redundancy header that could follow it (37 bits).
        // The decoder makes the same test (opus_decoder.c:509-514). Writing
        // the flag into a packet SILK had nearly filled shifted every CELT
        // symbol by one for libopus's decoder.
        if mode == OpusMode::Hybrid && self.rc.tell() + 17 + 20 <= 8 * (n_bytes as i32 - 1) {
            self.rc.encode_bit_logp(false, 12); // redundancy = 0
        }

        if mode == OpusMode::Hybrid {
            let nb_compr_bytes = (n_bytes - 1) as u32;
            self.rc.shrink(nb_compr_bytes);
        }

        let silk_ret_bytes = if mode == OpusMode::SilkOnly {
            ((self.rc.tell() + 7) >> 3) as usize
        } else {
            0
        };

        // libopus parity: adjust nbCompressedBytes for CELT/Hybrid based on tell (celt_encoder.c:1913-1921)
        // tmp = bitrate*frame_size + tell*Fs; nbCompressed = (tmp+4*Fs)/(8*Fs)
        // This is the CBR branch (vbr==0) in celt_encoder.c; for VBR the encoder
        // does *not* do this adjustment - it uses the vbr_bound logic instead.
        // Doing it for VBR would double-shrink and corrupt the budget.
        if mode != OpusMode::SilkOnly && self.use_cbr {
            let tell = self.rc.tell();
            if tell > 1 {
                let tmp = self.bitrate_bps as i64 * frame_size as i64
                    + tell as i64 * self.sampling_rate as i64;
                let adjusted = ((tmp + 4 * self.sampling_rate as i64)
                    / (8 * self.sampling_rate as i64)) as usize;
                let new_n = adjusted.clamp(1, OPUS_MAX_PACKET_BYTES).min(max_data_bytes);
                if new_n < n_bytes {
                    n_bytes = new_n;
                    // shrink range coder to new size (keep SILK bytes, trim tail)
                    let new_payload = n_bytes - 1;
                    self.rc.shrink(new_payload as u32);
                }
            }
        }
        if mode == OpusMode::SilkOnly {
            // C 2320: `prev_HB_gain` follows every frame; SILK-only has no
            // high band to attenuate.
            self.prev_hb_gain = 1.0;
        }
        if mode == OpusMode::CeltOnly || mode == OpusMode::Hybrid {
            self.celt_enc.complexity = self.complexity;
            let start_band = if mode == OpusMode::Hybrid { 17 } else { 0 };
            // libopus `CELT_SET_END_BAND`: match the TOC bandwidth so the
            // decoder reads the same number of coded bands (issue #37).
            let end_band = match frame_bandwidth {
                Bandwidth::Narrowband => 13,
                Bandwidth::Mediumband | Bandwidth::Wideband => 17,
                Bandwidth::Superwideband => 19,
                Bandwidth::Fullband | Bandwidth::Auto => 21,
            };
            self.celt_enc.set_end_band(end_band);
            let total_packet_bits = ((n_bytes - 1) * 8) as i32;
            // Propagate bitrate/VBR to CeltEncoder for accurate VBR handling (libopus parity)
            let celt_bitrate = if mode == OpusMode::Hybrid {
                // C opus_encoder.c:2453: CELT's target is what SILK's base
                // rate leaves of the bitrate. A floor of 8 kb/s here made a
                // 16 kb/s Hybrid stream come out at 24 kb/s. CELT's bitrate
                // control accepts nothing under 501 b/s.
                (self.bitrate_bps - hybrid_silk_rate).max(501)
            } else {
                self.bitrate_bps
            };
            self.celt_enc.set_bitrate(celt_bitrate);
            if mode == OpusMode::Hybrid {
                // C opus_encoder.c:2419-2424 (CELT_SET_SILK_INFO): the SILK
                // frame's signal type and quantization offset steer CELT's
                // bit target and its handling of transients.
                const QUANTIZATION_OFFSETS_Q10: [[i32; 2]; 2] = [[100, 240], [32, 100]];
                let indices = &self.silk_enc.s_cmn.indices;
                self.celt_enc.set_silk_info(
                    indices.signal_type as i32,
                    QUANTIZATION_OFFSETS_Q10[(indices.signal_type >> 1) as usize & 1]
                        [indices.quant_offset_type as usize & 1],
                );
            }
            self.celt_enc.set_vbr(!self.use_cbr);
            // libopus: Hybrid VBR is unconstrained (can steal from SILK), CELT-only constrained
            self.celt_enc
                .set_constrained_vbr(mode == OpusMode::CeltOnly);

            // Float view of the current frame for the CELT pipeline. For the
            // i16 entry point this converts once into a scratch buffer with
            // the exact per-sample scaling a caller of `encode()` performs
            // (`s as f32 / 32768.0`), so packets stay byte-for-byte identical
            // across the two entry points (issue #28).
            let input_f32: &[f32] = match input {
                PcmInput::F32(samples) => samples,
                PcmInput::I16(samples) => {
                    let n = frame_size * self.channels;
                    self.buf_f32_frame.resize(n, 0.0);
                    let staging = state_mut(&mut self.buf_f32_frame);
                    for (dst, &src) in staging[..n].iter_mut().zip(samples[..n].iter()) {
                        *dst = src as f32 / 32768.0;
                    }
                    state_ref(&self.buf_f32_frame)
                }
            };

            // Build the CELT input exactly like C `opus_encode_frame_native`:
            //   pcm_buf = [delay-compensation prefix from ring]
            //             [dc_reject / hp_cutoff'd current frame]
            // CELT then reads pcm_buf[0..frame_size*channels] (opus_encoder.c:
            // 1966-2010, 2493). This delay + filter step is what libopus feeds
            // its CELT encoder; passing the raw input diverges from 1.6.
            let celt_input: &[f32] = if self.delay_compensation > 0 {
                let delay = self.delay_compensation;
                let ebuf = self.encoder_buffer;
                let ch = self.channels;
                let total = (delay + frame_size) * ch;
                self.buf_celt_pcm.resize(total, 0.0);
                // 1. delay prefix from the ring (C: OPUS_COPY at 1967)
                let prefix = delay * ch;
                let src_start = (ebuf - delay) * ch;
                self.buf_celt_pcm[..prefix]
                    .copy_from_slice(&self.delay_buffer[src_start..src_start + prefix]);
                // 2. filter current frame into pcm_buf[delay..] (C: 2002-2010)
                let out = &mut self.buf_celt_pcm[prefix..];
                if self.application == Application::Voip {
                    hp_cutoff_float(
                        input_f32,
                        cutoff_hz,
                        out,
                        &mut self.hp_mem_float,
                        frame_size,
                        ch,
                        self.sampling_rate,
                    );
                } else {
                    dc_reject_float(
                        input_f32,
                        3,
                        out,
                        &mut self.hp_mem_float,
                        frame_size,
                        ch,
                        self.sampling_rate,
                    );
                }
                // 3. float NaN guard (C: 2016-2028)
                let mut sum = 0.0f32;
                for &v in &self.buf_celt_pcm[prefix..] {
                    sum += v * v;
                }
                // C shape kept verbatim (`!(sum < 1e9f) || celt_isnan(sum)`):
                // the negated comparison is deliberate NaN handling.
                #[allow(clippy::neg_cmp_op_on_partial_ord)]
                if !(sum < 1e9) || sum.is_nan() {
                    self.buf_celt_pcm[prefix..].fill(0.0);
                    self.hp_mem_float = [0.0; 4];
                }
                // 4. delay ring update (C: 2300-2312)
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
                // few bits. The gain fades in over CELT's overlap. It comes
                // after the ring update so that SILK's input is untouched.
                if self.prev_hb_gain < 1.0 || hb_gain < 1.0 {
                    gain_fade(
                        &mut self.buf_celt_pcm[..frame_size * ch],
                        self.prev_hb_gain,
                        hb_gain,
                        frame_size,
                        ch,
                        self.sampling_rate,
                    );
                }
                self.prev_hb_gain = hb_gain;
                // 5. deinterleave pcm_buf[0..frame_size*ch] (interleaved) to
                //    channel-major for the Rust CELT encoder.
                let n = frame_size * ch;
                self.buf_celt_input.resize(n, 0.0);
                for i in 0..frame_size {
                    for c in 0..ch {
                        self.buf_celt_input[c * frame_size + i] = self.buf_celt_pcm[i * ch + c];
                    }
                }
                state_ref(&self.buf_celt_input)
            } else if self.channels == 1 {
                input_f32
            } else {
                let n = frame_size * self.channels;
                self.buf_celt_input.resize(n, 0.0);
                for i in 0..frame_size {
                    for ch in 0..self.channels {
                        self.buf_celt_input[ch * frame_size + i] =
                            input_f32[i * self.channels + ch];
                    }
                }
                state_ref(&self.buf_celt_input)
            };

            if self.rc.tell() <= total_packet_bits {
                let is_vbr = !self.use_cbr;
                self.celt_enc.encode_with_budget_vbr(
                    celt_input,
                    frame_size,
                    &mut self.rc,
                    start_band,
                    total_packet_bits,
                    is_vbr,
                );
            }
        }

        self.rc.done();

        // C opus_encoder.c:2170-2179: "In the unlikely case that the SILK
        // encoder busted its target, tell the decoder to call the PLC" — emit
        // a minimal packet with an EMPTY frame instead of failing hard. The
        // zero-length frame makes the decoder conceal (libopus pads the same
        // 2-byte packet with opus_packet_pad under CBR). With FEC the LBRR
        // section can squeeze the main frame below what even the
        // damage-control re-encode fits, which is exactly the case this
        // fallback exists for.
        if self.rc.error != 0 || self.rc.tell() > (n_bytes as i32 - 1) * 8 {
            if output.len() < 2 {
                return Err("Output buffer too small");
            }
            let mut ret = 2usize;
            if self.use_cbr && n_bytes >= 2 {
                // CBR: pad to the nominal size as a code-3 packet holding a
                // single EMPTY frame (opus_packet_pad of the 2-byte packet).
                let target_total = n_bytes.min(output.len());
                if target_total > 2 {
                    output[0] = toc | 0x03;
                    output[1] = 0x41; // 1 frame, padding present, CBR
                    let pad_amount = target_total - 2;
                    let nb_255s = (pad_amount - 1) / 255;
                    let mut ptr = 2;
                    for _ in 0..nb_255s {
                        output[ptr] = 255;
                        ptr += 1;
                    }
                    output[ptr] = (pad_amount - 255 * nb_255s - 1) as u8;
                    ptr += 1;
                    for b in &mut output[ptr..target_total] {
                        *b = 0;
                    }
                    self.prev_enc_mode = Some(mode);
                    return Ok(target_total);
                }
                ret = target_total.max(1);
            }
            // VBR: a 2-byte code-0 packet whose frame length byte is 0.
            output[0] = toc;
            output[1] = 0;
            self.prev_enc_mode = Some(mode);
            return Ok(ret);
        }

        // C opus_encoder.c:2555-2558: every coded frame leaves its mode behind
        // for the next one. The variable-bitrate SILK-only return below used
        // to skip this, so after CELT-only frames the mode before stayed
        // "CELT-only" and SILK restarted on every packet.
        self.prev_enc_mode = Some(mode);

        if mode == OpusMode::SilkOnly {
            let mut ret = silk_ret_bytes.min(self.rc.storage as usize);
            while ret > 2 && self.rc.buf[ret - 1] == 0 {
                ret -= 1;
            }

            let target_total = if self.use_cbr {
                n_bytes.min(output.len())
            } else {
                (ret + 1).min(output.len())
            };

            let silk_len = ret;

            if !self.use_cbr || silk_len + 1 >= target_total {
                // VBR or payload fills the target: simple code 0 packet
                output[0] = toc;
                let copy_len = silk_len.min(target_total - 1);
                output[1..1 + copy_len].copy_from_slice(&self.rc.buf[..copy_len]);
                return Ok((copy_len + 1).min(output.len()));
            }

            output[0] = toc | 0x03;

            if silk_len + 2 >= target_total {
                output[1] = 0x01;
                let copy_len = (target_total - 2).min(silk_len);
                output[2..2 + copy_len].copy_from_slice(&self.rc.buf[..copy_len]);
                self.prev_enc_mode = Some(mode);
                return Ok(target_total.min(output.len()));
            }

            let pad_amount = target_total - silk_len - 2;
            output[1] = 0x41;

            let nb_255s = (pad_amount - 1) / 255;
            let mut ptr = 2;
            for _ in 0..nb_255s {
                output[ptr] = 255;
                ptr += 1;
            }
            output[ptr] = (pad_amount - 255 * nb_255s - 1) as u8;
            ptr += 1;

            output[ptr..ptr + silk_len].copy_from_slice(&self.rc.buf[..silk_len]);
            ptr += silk_len;

            let fill_end = target_total.min(output.len());
            for byte in output[ptr..fill_end].iter_mut() {
                *byte = 0;
            }

            self.prev_enc_mode = Some(mode);
            return Ok(target_total.min(output.len()));
        }

        // For CELT/Hybrid, respect possible VBR shrink performed by CeltEncoder (e.g. silence)
        let payload_len = (self.rc.storage as usize).min(output.len() - 1);
        output[1..1 + payload_len].copy_from_slice(&self.rc.buf[..payload_len]);
        self.prev_enc_mode = Some(mode);
        Ok(payload_len + 1)
    }

    /// Repacketize 40/60 ms CELT/Hybrid input as 20 ms sub-frames in one code
    /// 1/2/3 Opus packet (libopus `encode_multiframe_packet`). Every sub-frame
    /// is forced to the packet's `decisions` so the packet carries a single
    /// config.
    fn encode_multiframe(
        &mut self,
        input: PcmInput<'_>,
        frame_size: usize,
        decisions: ForcedFrame,
        output: &mut [u8],
    ) -> Result<usize, &'static str> {
        let mode = decisions.mode;
        const MAX_SUB: usize = 3; // up to 60 ms of 20 ms frames
        let ch = self.channels;
        let enc_frame = (self.sampling_rate / 50) as usize; // 20 ms
        if enc_frame == 0 || !frame_size.is_multiple_of(enc_frame) {
            return Err("Invalid multi-frame size");
        }
        let nb = frame_size / enc_frame;
        if !(2..=MAX_SUB).contains(&nb) {
            return Err("Unsupported multi-frame size");
        }

        // RFC 6716 size field: one byte below 252, else two.
        fn put_size(out: &mut [u8], pos: &mut usize, len: usize) -> Result<(), &'static str> {
            if len < 252 {
                if *pos >= out.len() {
                    return Err("Output buffer too small");
                }
                out[*pos] = len as u8;
                *pos += 1;
            } else {
                if *pos + 1 >= out.len() {
                    return Err("Output buffer too small");
                }
                let first = 252 + (len & 3);
                out[*pos] = first as u8;
                out[*pos + 1] = ((len - first) >> 2) as u8;
                *pos += 2;
            }
            Ok(())
        }

        // C opus_encoder.c:1735-1795: how large the packet and each of its
        // 20 ms parts may be.
        //
        // - The packet: the caller's buffer, or under CBR the constant size.
        // - A part: its share of the packet once the worst-case header is
        //   taken off. Under CBR also never more than the bitrate gives
        //   20 ms.
        //
        // Not ported on purpose: libopus 1.6 applies the "never more than
        // the bitrate gives 20 ms" limit under VBR too (`curr_max`,
        // opus_encoder.c:1785). That turns VBR into a hard cap per part and
        // costs libopus itself up to 23 dB when in-band FEC is on (measured:
        // 24 kHz stereo Hybrid, 128 kb/s, 40 ms, FEC: 3.9 dB against 27.2 dB
        // for its own 20 ms packets). libopus 1.5 and earlier had no such
        // limit under VBR, and neither has opus-rs.
        let max_header_bytes = if nb == 2 { 3 } else { 2 + (nb - 1) * 2 };
        let cbr_bytes = ((bitrate_to_bits(self.bitrate_bps, self.sampling_rate, frame_size) + 4)
            / 8)
            .max(0) as usize;
        let (repacketize_len, packet_bitrate) = if self.use_cbr {
            let len = cbr_bytes.min(output.len());
            let rate = bits_to_bitrate(8 * len as i32, self.sampling_rate, frame_size);
            (len, rate)
        } else {
            (output.len(), self.bitrate_bps)
        };
        let max_len_sum = (nb + repacketize_len).saturating_sub(max_header_bytes);
        let nominal_part =
            (bitrate_to_bits(packet_bitrate, self.sampling_rate, enc_frame) / 8).max(0) as usize;

        let mut sub = [[0u8; 1276]; MAX_SUB];
        let mut lens = [0usize; MAX_SUB];
        self.forced_frame = Some(decisions);
        let enc_res = (|| -> Result<(), &'static str> {
            let mut tot_size = 0usize;
            for i in 0..nb {
                let start = i * enc_frame * ch;
                let end = start + enc_frame * ch;
                let sub_input = match input {
                    PcmInput::F32(s) => PcmInput::F32(&s[start..end]),
                    PcmInput::I16(s) => PcmInput::I16(&s[start..end]),
                };
                self.nonfinal_frame = i + 1 < nb;
                let mut curr_max = (max_len_sum / nb).min(max_len_sum.saturating_sub(tot_size));
                if self.use_cbr {
                    curr_max = curr_max.min(nominal_part);
                }
                // A part needs its TOC and one byte at least.
                let curr_max = curr_max.clamp(2, 1276);
                let n = self.encode_impl(sub_input, enc_frame, &mut sub[i][..curr_max])?;
                if n == 0 {
                    return Err("Empty sub-frame");
                }
                lens[i] = n - 1;
                tot_size += n;
            }
            Ok(())
        })();
        self.forced_frame = None;
        self.nonfinal_frame = false;
        enc_res?;

        // Pack the parts like libopus's repacketizer
        // (repacketizer.c opus_repacketizer_out_range_impl):
        //
        // - Two parts: code 1 when they are the same size, else code 2.
        // - Three parts, or a CBR packet that needs padding: code 3, with the
        //   sizes of all parts but the last when they differ.
        // - CBR: padded to the constant size.
        let toc = sub[0][0] & 0xFC;
        let lens = &lens[..nb];
        let payload_sum: usize = lens.iter().sum();
        let size_bytes = |len: usize| 1 + (len >= 252) as usize;
        let same_size = lens.iter().all(|&l| l == lens[0]);
        let simple_size = if same_size {
            1 + payload_sum
        } else {
            1 + size_bytes(lens[0]) + payload_sum
        };
        let needs_padding = self.use_cbr && simple_size < repacketize_len;
        let mut pos = 1usize;
        let packet_len;
        if nb == 2 && !needs_padding {
            packet_len = simple_size;
            if packet_len > output.len() {
                return Err("Output buffer too small");
            }
            if same_size {
                output[0] = toc | 0x01;
            } else {
                output[0] = toc | 0x02;
                put_size(output, &mut pos, lens[0])?;
            }
        } else {
            let mut tot_size = 2 + payload_sum;
            if !same_size {
                tot_size += lens[..nb - 1].iter().map(|&l| size_bytes(l)).sum::<usize>();
            }
            // Padding: length bytes and zeros, together `pad_amount` bytes.
            let pad_amount = if self.use_cbr {
                repacketize_len.saturating_sub(tot_size)
            } else {
                0
            };
            packet_len = tot_size + pad_amount;
            if packet_len > output.len() {
                return Err("Output buffer too small");
            }
            output[0] = toc | 0x03;
            output[1] = nb as u8;
            if !same_size {
                output[1] |= 0x80;
            }
            pos = 2;
            if pad_amount != 0 {
                output[1] |= 0x40;
                let nb_255s = (pad_amount - 1) / 255;
                for _ in 0..nb_255s {
                    output[pos] = 255;
                    pos += 1;
                }
                output[pos] = (pad_amount - 255 * nb_255s - 1) as u8;
                pos += 1;
            }
            if !same_size {
                for &len in &lens[..nb - 1] {
                    put_size(output, &mut pos, len)?;
                }
            }
        }
        for (part, &len) in sub.iter().zip(lens) {
            output[pos..pos + len].copy_from_slice(&part[1..1 + len]);
            pos += len;
        }
        output[pos..packet_len].fill(0);
        let pos = packet_len;
        self.prev_enc_mode = Some(mode);
        Ok(pos)
    }
}

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

    prev_mode: Option<OpusMode>,

    /// Whether the previous frame had redundancy (mode transition marker).
    prev_redundancy: bool,
    frame_size: usize,

    bandwidth: Bandwidth,

    stream_channels: usize,

    silk_resampler: silk::resampler::SilkResampler,

    /// Second resampler instance for stereo channel 1.
    silk_resampler_2: silk::resampler::SilkResampler,

    prev_internal_rate: i32,

    pub hybrid_skip_celt: bool,

    #[cfg(not(feature = "heap"))]
    w_pcm_i16: FixedVec<i16, OPUS_PCM_I16>,
    #[cfg(feature = "heap")]
    w_pcm_i16: Box<FixedVec<i16, OPUS_PCM_I16>>,
    #[cfg(not(feature = "heap"))]
    w_silk_out: FixedVec<f32, OPUS_SUBFRAME_SCRATCH>,
    #[cfg(feature = "heap")]
    w_silk_out: Box<FixedVec<f32, OPUS_SUBFRAME_SCRATCH>>,
    #[cfg(not(feature = "heap"))]
    w_pcm_resampled: FixedVec<i16, OPUS_SUBFRAME_SCRATCH>,
    #[cfg(feature = "heap")]
    w_pcm_resampled: Box<FixedVec<i16, OPUS_SUBFRAME_SCRATCH>>,
    #[cfg(not(feature = "heap"))]
    w_celt_planar: FixedVec<f32, OPUS_SUBFRAME_SCRATCH>,
    #[cfg(feature = "heap")]
    w_celt_planar: Box<FixedVec<f32, OPUS_SUBFRAME_SCRATCH>>,
    #[cfg(not(feature = "heap"))]
    w_celt_out: FixedVec<f32, OPUS_SUBFRAME_SCRATCH>,
    #[cfg(feature = "heap")]
    w_celt_out: Box<FixedVec<f32, OPUS_SUBFRAME_SCRATCH>>,
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

        #[cfg(feature = "heap")]
        let mut silk_dec = Box::new(silk::dec_api::SilkDecoder::new());
        #[cfg(not(feature = "heap"))]
        let mut silk_dec = silk::dec_api::SilkDecoder::new();
        silk_dec.init(sampling_rate.min(16000), channels as i32);
        silk_dec.channel_state[0].fs_api_hz = sampling_rate;

        Ok(Self {
            celt_dec,
            silk_dec,
            sampling_rate,
            channels,
            prev_mode: None,
            prev_redundancy: false,
            frame_size: 0,
            bandwidth: Bandwidth::Auto,
            stream_channels: channels,
            silk_resampler: silk::resampler::SilkResampler::default(),
            silk_resampler_2: silk::resampler::SilkResampler::default(),
            prev_internal_rate: 0,
            hybrid_skip_celt: false,

            #[cfg(not(feature = "heap"))]
            w_pcm_i16: FixedVec::from_value(0i16, 960 * channels),
            #[cfg(feature = "heap")]
            w_pcm_i16: Box::new(FixedVec::from_value(0i16, 960 * channels)),

            #[cfg(not(feature = "heap"))]
            w_silk_out: FixedVec::from_value(0.0f32, OPUS_MAX_SUBFRAME * channels),
            #[cfg(feature = "heap")]
            w_silk_out: Box::new(FixedVec::from_value(0.0f32, OPUS_MAX_SUBFRAME * channels)),
            #[cfg(not(feature = "heap"))]
            w_pcm_resampled: FixedVec::from_value(0i16, OPUS_MAX_SUBFRAME * channels),
            #[cfg(feature = "heap")]
            w_pcm_resampled: Box::new(FixedVec::from_value(0i16, OPUS_MAX_SUBFRAME * channels)),
            #[cfg(not(feature = "heap"))]
            w_celt_planar: FixedVec::from_value(0.0f32, OPUS_MAX_SUBFRAME * channels),
            #[cfg(feature = "heap")]
            w_celt_planar: Box::new(FixedVec::from_value(0.0f32, OPUS_MAX_SUBFRAME * channels)),
            #[cfg(not(feature = "heap"))]
            w_celt_out: FixedVec::from_value(0.0f32, OPUS_MAX_SUBFRAME * channels),
            #[cfg(feature = "heap")]
            w_celt_out: Box::new(FixedVec::from_value(0.0f32, OPUS_MAX_SUBFRAME * channels)),
        })
    }

    /// Decode one Opus packet into interleaved `f32` PCM, returning the
    /// number of samples per channel written.
    ///
    /// Packet loss follows libopus: pass an empty packet (`&[]`) to conceal
    /// `frame_size` samples (a multiple of 2.5 ms), or a ToC-only 1-byte
    /// packet to conceal the duration that ToC describes. Either way the
    /// frame is concealed in the previous packet's mode; the placeholder's
    /// mode, bandwidth and channel bits are ignored.
    pub fn decode(
        &mut self,
        input: &[u8],
        frame_size: usize,
        output: &mut [f32],
    ) -> Result<usize, &'static str> {
        // A packet of 0 or 1 bytes (ToC only) is a lost/DTX frame: conceal it
        // in the previous mode without touching the mode state (issue #15).
        if input.len() <= 1 {
            return self.decode_lost(input.first().copied(), frame_size, output);
        }

        let toc = input[0];
        let mode = mode_from_toc(toc);
        let packet_channels = channels_from_toc(toc);
        let bandwidth = bandwidth_from_toc(toc);
        let frame_duration_ms = frame_duration_ms_from_toc(toc);

        if packet_channels != self.channels {
            return Err("Channel count mismatch between packet and decoder");
        }

        let code = toc & 0x03;
        let frame_count: usize;
        let frame_payloads: FixedVec<&[u8], OPUS_MAX_PACKET_FRAMES>;

        match code {
            0 => {
                frame_count = 1;
                frame_payloads = FixedVec::from_slice(&[&input[1..]]);
            }
            1 => {
                frame_count = 2;
                let data_len = input.len() - 1;
                // RFC 6716 §3.2.1: code 1 carries two equal-size (CBR) frames,
                // so the payload length must be even. libopus rejects odd lengths.
                if !data_len.is_multiple_of(2) {
                    return Err("Code 1: payload length must be even");
                }
                let half = data_len / 2;
                if half == 0 {
                    return Err("Code 1: empty frame");
                }
                frame_payloads = FixedVec::from_slice(&[&input[1..1 + half], &input[1 + half..]]);
            }
            2 => {
                frame_count = 2;
                let data = &input[1..];
                if data.is_empty() {
                    return Err("Code 2 packet has no data");
                }
                let (first_len, header_size) = parse_frame_size(data)?;
                if header_size + first_len > data.len() {
                    return Err("Code 2: first frame size exceeds packet");
                }
                frame_payloads = FixedVec::from_slice(&[
                    &data[header_size..header_size + first_len],
                    &data[header_size + first_len..],
                ]);
            }
            3 => {
                if input.len() < 2 {
                    return Err("Code 3 packet too short");
                }
                let count_byte = input[1];
                let n_frames = (count_byte & 0x3F) as usize;
                if !(1..=48).contains(&n_frames) {
                    return Err("Code 3: invalid frame count");
                }
                frame_count = n_frames;
                // Bit 6 = padding flag, bit 7 = VBR flag (RFC 6716 §3.2.1).
                let padding_flag = (count_byte & 0x40) != 0;
                let vbr = (count_byte & 0x80) != 0;

                // Parse the optional padding length bytes that follow the count
                // byte. The padding *content* (pad_len bytes) lives at the end of
                // the packet and is not part of any frame.
                let mut ptr = 2usize;
                let mut pad_len = 0usize;
                if padding_flag {
                    loop {
                        if ptr >= input.len() {
                            return Err("Code 3: padding overflow");
                        }
                        let p = input[ptr] as usize;
                        ptr += 1;
                        if p == 255 {
                            pad_len += 254;
                        } else {
                            pad_len += p;
                            break;
                        }
                    }
                }
                if ptr + pad_len > input.len() {
                    return Err("Code 3: padding exceeds packet");
                }
                let payload_end = input.len() - pad_len;
                let payload = &input[ptr..payload_end];

                let mut payloads: FixedVec<&[u8], OPUS_MAX_PACKET_FRAMES> = FixedVec::new();
                if frame_count == 1 {
                    // Single frame: the entire payload region is the frame, both
                    // for VBR and CBR (no length prefix is present).
                    payloads.push(payload);
                } else if vbr {
                    // VBR (V=1): the per-frame length fields for all frames
                    // except the last come first, then all frame payloads
                    // (RFC 6716 §3.2.1; libopus `opus_packet_parse_impl`).
                    // Interleaving length and data — as this used to — parsed
                    // any libopus code-3 VBR packet wrong.
                    let mut cursor = 0usize;
                    let mut frame_bytes = 0usize;
                    let mut sizes = [0usize; OPUS_MAX_PACKET_FRAMES];
                    for i in 0..frame_count - 1 {
                        if cursor >= payload.len() {
                            return Err("Code 3: unexpected end in VBR header");
                        }
                        let (frame_len, header_bytes) = parse_frame_size(&payload[cursor..])?;
                        cursor += header_bytes;
                        sizes[i] = frame_len;
                        frame_bytes += frame_len;
                    }
                    if frame_bytes > payload.len() - cursor {
                        return Err("Code 3: frame length exceeds packet");
                    }
                    sizes[frame_count - 1] = payload.len() - cursor - frame_bytes;
                    let mut data_pos = cursor;
                    for i in 0..frame_count {
                        payloads.push(&payload[data_pos..data_pos + sizes[i]]);
                        data_pos += sizes[i];
                    }
                } else {
                    // CBR (V=0): remaining bytes are split equally into M frames
                    // (RFC 6716 §3.2.1: "the remaining bytes are split into M
                    // equal chunks").
                    if !payload.len().is_multiple_of(frame_count) {
                        return Err("Code 3 CBR: payload not divisible by frame count");
                    }
                    let frame_len = payload.len() / frame_count;
                    for i in 0..frame_count {
                        payloads.push(&payload[i * frame_len..(i + 1) * frame_len]);
                    }
                }
                frame_payloads = payloads;
            }
            _ => unreachable!(),
        }

        self.frame_size = frame_size;
        self.bandwidth = bandwidth;
        self.stream_channels = packet_channels;

        // Derive the actual per-frame sample count from the TOC, not from the
        // caller's frame_size. This prevents panics in bands.rs/celt.rs when
        // the caller passes a mismatched frame_size (issue #7 sub-item 1):
        // the internal decoders always get the correct geometry.
        let toc_frame_size = frame_samples_from_toc(toc, self.sampling_rate)
            .ok_or("Invalid TOC for sampling rate")?;
        let decoded_total = toc_frame_size * frame_count;
        if frame_size < decoded_total {
            return Err("frame_size too small for packet");
        }
        if output.len() < decoded_total * self.channels {
            return Err("Output buffer too small for packet");
        }
        // Zero-fill any extra space the caller provided beyond what the packet
        // actually produces, so stale data is never left in the buffer.
        if output.len() > decoded_total * self.channels {
            for v in &mut output[decoded_total * self.channels..] {
                *v = 0.0;
            }
        }

        // Each frame of the packet goes through the same steps as libopus
        // `opus_decode_frame`.
        let frame_len = toc_frame_size * self.channels;
        for (fi, payload) in frame_payloads.iter().enumerate() {
            let out = &mut output[fi * frame_len..(fi + 1) * frame_len];
            if payload.len() <= 1 {
                // A frame of 0 or 1 bytes is a lost or DTX frame
                // (opus_decode_frame: `if (len<=1) data = NULL`).
                self.conceal(toc_frame_size, out)?;
            } else {
                self.decode_frame(
                    mode,
                    bandwidth,
                    frame_duration_ms,
                    toc,
                    payload,
                    toc_frame_size,
                    out,
                )?;
            }
        }
        Ok(decoded_total)
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
    fn decode_frame(
        &mut self,
        mode: OpusMode,
        bandwidth: Bandwidth,
        frame_duration_ms: i32,
        toc: u8,
        payload: &[u8],
        frame_size: usize,
        out: &mut [f32],
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
        let mut pcm_transition: FixedVec<f32, OPUS_PCM_TAIL> =
            FixedVec::from_value(0.0, f5 * channels);
        let transition_len = f5.min(frame_size);
        if transition && mode == OpusMode::CeltOnly {
            // 5 ms of concealment in the previous (SILK or Hybrid) mode.
            self.conceal(transition_len, &mut pcm_transition[..transition_len * channels])?;
        }

        // SILK layer.
        if mode != OpusMode::CeltOnly {
            if prev_mode == Some(OpusMode::CeltOnly) {
                // libopus `silk_ResetDecoder`: the SILK state is stale after
                // CELT-only frames. Its resamplers start afresh as well.
                self.silk_dec.reset();
                self.silk_dec.channel_state[0].fs_api_hz = self.sampling_rate;
                self.prev_internal_rate = 0;
            }
            let internal_sample_rate = if mode == OpusMode::SilkOnly {
                match bandwidth {
                    Bandwidth::Narrowband => 8000,
                    Bandwidth::Mediumband => 12000,
                    _ => 16000,
                }
            } else {
                16000
            };
            if internal_sample_rate != self.prev_internal_rate
                || !self.silk_resampler.is_initialized()
            {
                self.silk_resampler
                    .init(internal_sample_rate, self.sampling_rate);
                self.silk_resampler_2
                    .init(internal_sample_rate, self.sampling_rate);
            }
            self.prev_internal_rate = internal_sample_rate;
            self.decode_silk_frames(
                &mut rc,
                silk::decode_frame::FLAG_DECODE_NORMAL,
                frame_duration_ms,
                internal_sample_rate,
                frame_size,
            )?;
        }

        // Redundancy frame: present if enough bits are left after SILK.
        // A Hybrid frame says so with a flag; in a SILK-only frame any such
        // remainder is one.
        let mut redundancy = false;
        let mut celt_to_silk = false;
        let mut redundancy_bytes = 0usize;
        let hybrid = mode == OpusMode::Hybrid;
        if mode != OpusMode::CeltOnly
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
        let start_band = if mode != OpusMode::CeltOnly { 17 } else { 0 };
        if redundancy {
            transition = false;
        }
        if transition && mode != OpusMode::CeltOnly {
            // 5 ms of concealment in the previous (CELT-only) mode.
            self.conceal(transition_len, &mut pcm_transition[..transition_len * channels])?;
        }

        let celt_end_band = self.celt_end_band_from_toc(toc);
        let mut redundant_audio: FixedVec<f32, OPUS_PCM_TAIL> =
            FixedVec::from_value(0.0, f5 * channels);

        // Redundancy frame for a switch from CELT-only: it continues the
        // CELT state of the previous frame, so it is decoded first.
        if redundancy && celt_to_silk {
            let mut rc_red = RangeCoder::new_decoder(&payload[len..len + redundancy_bytes]);
            self.celt_decode_interleaved(
                &mut rc_red,
                (redundancy_bytes * 8) as i32,
                f5,
                0,
                celt_end_band,
                &mut redundant_audio,
            );
        }

        if mode != OpusMode::SilkOnly {
            // Discard the CELT state of another mode.
            if prev_mode.is_some() && prev_mode != Some(mode) && !self.prev_redundancy {
                self.celt_dec.reset_state();
            }
            let (celt_dec, celt_planar) = (
                state_mut(&mut self.celt_dec),
                state_mut(&mut self.w_celt_planar),
            );
            celt_dec.decode_from_range_coder_with_band_range(
                &mut rc,
                (len * 8) as i32,
                frame_size,
                &mut celt_planar[..frame_len],
                start_band,
                celt_end_band,
            );
            for i in 0..frame_size {
                for ch in 0..channels {
                    let celt = self.w_celt_planar[ch * frame_size + i];
                    let idx = i * channels + ch;
                    out[idx] = if hybrid {
                        self.w_silk_out[idx] + celt
                    } else {
                        celt
                    };
                }
            }
        } else {
            out.copy_from_slice(&self.w_silk_out[..frame_len]);
            // After a Hybrid frame the CELT layer still holds half a window
            // of signal. Decoding 2.5 ms of CELT silence lets it fade out.
            if prev_mode == Some(OpusMode::Hybrid)
                && !(redundancy && celt_to_silk && self.prev_redundancy)
            {
                let mut fade: FixedVec<f32, OPUS_PCM_TAIL> =
                    FixedVec::from_value(0.0, f2_5 * channels);
                let mut rc_silence = RangeCoder::new_decoder(&[0xFF, 0xFF]);
                self.celt_decode_interleaved(&mut rc_silence, 16, f2_5, 0, celt_end_band, &mut fade);
                for (o, f) in out.iter_mut().zip(fade.iter()) {
                    *o += *f;
                }
            }
        }

        let window = modes::default_mode().window;
        let inc = (48000 / self.sampling_rate) as usize;
        let f2_5_ch = f2_5 * channels;

        // Redundancy frame for a switch to CELT-only: it starts from a
        // fresh CELT state and replaces the end of this frame.
        if redundancy && !celt_to_silk {
            self.celt_dec.reset_state();
            let mut rc_red = RangeCoder::new_decoder(&payload[len..len + redundancy_bytes]);
            self.celt_decode_interleaved(
                &mut rc_red,
                (redundancy_bytes * 8) as i32,
                f5,
                0,
                celt_end_band,
                &mut redundant_audio,
            );
            let tail = frame_len - f2_5_ch;
            let main: FixedVec<f32, OPUS_PCM_TAIL> = FixedVec::from_slice(&out[tail..]);
            smooth_fade(
                &main,
                &redundant_audio[f2_5_ch..],
                &mut out[tail..],
                f2_5,
                channels,
                window,
                inc,
            );
        }
        // Redundancy frame for a switch from CELT-only: it supplies the
        // start of this frame. It is ignored if the previous frame did not
        // use CELT (the first frame of the switch was lost).
        if redundancy
            && celt_to_silk
            && (prev_mode != Some(OpusMode::SilkOnly) || self.prev_redundancy)
        {
            out[..f2_5_ch].copy_from_slice(&redundant_audio[..f2_5_ch]);
            let main: FixedVec<f32, OPUS_PCM_TAIL> = FixedVec::from_slice(&out[f2_5_ch..2 * f2_5_ch]);
            smooth_fade(
                &redundant_audio[f2_5_ch..],
                &main,
                &mut out[f2_5_ch..2 * f2_5_ch],
                f2_5,
                channels,
                window,
                inc,
            );
        }
        if transition {
            if frame_size >= f5 {
                out[..f2_5_ch].copy_from_slice(&pcm_transition[..f2_5_ch]);
                let main: FixedVec<f32, OPUS_PCM_TAIL> =
                    FixedVec::from_slice(&out[f2_5_ch..2 * f2_5_ch]);
                smooth_fade(
                    &pcm_transition[f2_5_ch..],
                    &main,
                    &mut out[f2_5_ch..2 * f2_5_ch],
                    f2_5,
                    channels,
                    window,
                    inc,
                );
            } else {
                // A 2.5 ms frame: no room for a clean cross-fade.
                let main: FixedVec<f32, OPUS_PCM_TAIL> = FixedVec::from_slice(&out[..f2_5_ch]);
                smooth_fade(&pcm_transition, &main, &mut out[..f2_5_ch], f2_5, channels, window, inc);
            }
        }

        for sample in out.iter_mut() {
            *sample = sample.clamp(-1.0, 1.0);
        }

        self.prev_mode = Some(mode);
        self.prev_redundancy = redundancy && !celt_to_silk;
        Ok(())
    }

    /// Decode one CELT frame of `n` samples per channel into `dst`,
    /// interleaved.
    fn celt_decode_interleaved(
        &mut self,
        rc: &mut RangeCoder,
        total_bits: i32,
        n: usize,
        start_band: usize,
        end_band: usize,
        dst: &mut [f32],
    ) {
        let channels = self.channels;
        let (celt_dec, planar) = (
            state_mut(&mut self.celt_dec),
            state_mut(&mut self.w_celt_planar),
        );
        celt_dec.decode_from_range_coder_with_band_range(
            rc,
            total_bits,
            n,
            &mut planar[..n * channels],
            start_band,
            end_band,
        );
        for i in 0..n {
            for ch in 0..channels {
                dst[i * channels + ch] = self.w_celt_planar[ch * n + i];
            }
        }
    }

    /// Conceal a lost frame (libopus `opus_decode_frame` with `data == NULL`).
    ///
    /// Only the duration comes from the caller: the placeholder ToC of a
    /// 1-byte packet, or `frame_size` for an empty one. Mode, bandwidth, SILK
    /// rate and resamplers stay the previous packet's, so a placeholder such
    /// as `[0]` (SILK NB 10 ms) can no longer switch a CELT or wideband stream
    /// into another mode and fake two mode transitions (issue #15).
    fn decode_lost(
        &mut self,
        toc: Option<u8>,
        frame_size: usize,
        output: &mut [f32],
    ) -> Result<usize, &'static str> {
        let f2_5 = self.sampling_rate as usize / 400;

        let total = match toc {
            None => {
                if frame_size == 0 || !frame_size.is_multiple_of(f2_5) {
                    return Err("Lost frame: frame_size must be a multiple of 2.5 ms");
                }
                frame_size
            }
            Some(toc) => {
                // A ToC-only packet: code 0 is one empty frame, codes 1 and 2
                // are two; code 3 lacks its frame-count byte (libopus rejects).
                let count = match toc & 0x03 {
                    0 => 1,
                    1 | 2 => 2,
                    _ => return Err("Code 3 packet too short"),
                };
                let per_frame = frame_samples_from_toc(toc, self.sampling_rate)
                    .ok_or("Invalid TOC for sampling rate")?;
                let total = per_frame * count;
                if frame_size < total {
                    return Err("frame_size too small for packet");
                }
                total
            }
        };
        let channels = self.channels;
        if output.len() < total * channels {
            return Err("Output buffer too small for packet");
        }
        output[total * channels..].fill(0.0);
        self.conceal(total, &mut output[..total * channels])?;
        Ok(total)
    }

    /// Fill `output` with `total` samples per channel of concealment in the
    /// mode of the previous frame (libopus `opus_decode_frame` with
    /// `data == NULL`). After a frame that ended with a redundancy frame
    /// for a switch to CELT-only, that mode is CELT-only.
    fn conceal(&mut self, total: usize, output: &mut [f32]) -> Result<(), &'static str> {
        let f2_5 = self.sampling_rate as usize / 400;
        let (f5, f10, f20) = (2 * f2_5, 4 * f2_5, 8 * f2_5);
        let channels = self.channels;
        let plc_mode = if self.prev_redundancy {
            Some(OpusMode::CeltOnly)
        } else {
            self.prev_mode
        };

        let mut pos = 0usize;
        while pos < total {
            let remaining = total - pos;
            let dst = &mut output[pos * channels..total * channels];
            let produced = match plc_mode {
                None => {
                    dst.fill(0.0);
                    remaining
                }
                Some(OpusMode::CeltOnly) => {
                    // Largest CELT frame that fits; `remaining` is a multiple
                    // of 2.5 ms, so one always does.
                    let n = [f20, f10, f5, f2_5]
                        .into_iter()
                        .find(|&c| c <= remaining)
                        .unwrap_or(f2_5);
                    self.conceal_celt(n, 0)?;
                    for (d, s) in dst.iter_mut().zip(&self.w_celt_out[..n * channels]) {
                        *d = s.clamp(-1.0, 1.0);
                    }
                    n
                }
                Some(prev @ (OpusMode::SilkOnly | OpusMode::Hybrid)) => {
                    // SILK conceals in 10/20 ms frames; a shorter request
                    // takes the head of a 10 ms frame, as libopus does.
                    let (ms, n) = if remaining >= f20 {
                        (20, f20)
                    } else {
                        (10, f10)
                    };
                    let hybrid = prev == OpusMode::Hybrid;
                    let rate = if hybrid {
                        16000
                    } else {
                        self.prev_internal_rate
                    };
                    let mut rc = RangeCoder::new_decoder(&[]);
                    self.decode_silk_frames(
                        &mut rc,
                        silk::decode_frame::FLAG_PACKET_LOST,
                        ms,
                        rate,
                        n,
                    )?;
                    if hybrid {
                        self.conceal_celt(n, 17)?;
                    } else {
                        self.w_celt_out[..n * channels].fill(0.0);
                    }
                    let take = n.min(remaining);
                    for (i, d) in dst[..take * channels].iter_mut().enumerate() {
                        *d = (self.w_silk_out[i] + self.w_celt_out[i]).clamp(-1.0, 1.0);
                    }
                    take
                }
            };
            pos += produced;
        }
        // A concealed frame carries no redundancy frame.
        if plc_mode.is_some() {
            self.prev_mode = plc_mode;
        }
        self.prev_redundancy = false;
        Ok(())
    }

    /// CELT concealment for `n` samples: decode an empty payload, which takes
    /// the silence path — the previous frame's MDCT overlap decays into
    /// silence with the decoder state kept continuous. Output lands
    /// interleaved in `w_celt_out[..n * channels]`.
    fn conceal_celt(&mut self, n: usize, start_band: usize) -> Result<(), &'static str> {
        let len = n * self.channels;
        let end_band = modes::default_mode().eff_ebands;
        let mut rc = RangeCoder::new_decoder(&[]);
        let decoded = {
            let (celt_dec, planar) = (
                state_mut(&mut self.celt_dec),
                state_mut(&mut self.w_celt_planar),
            );
            celt_dec.decode_from_range_coder_with_band_range(
                &mut rc,
                0,
                n,
                &mut planar[..len],
                start_band,
                end_band,
            )
        };
        if decoded != n {
            return Err("CELT concealment failed");
        }
        for i in 0..n {
            for ch in 0..self.channels {
                self.w_celt_out[i * self.channels + ch] = self.w_celt_planar[ch * n + i];
            }
        }
        Ok(())
    }
}

impl OpusDecoder {
    /// Decode one Opus frame of SILK audio — `duration_ms` at `internal_rate`
    /// — into `w_silk_out[..api_len * channels]`, interleaved at the API rate.
    ///
    /// A 40/60 ms payload carries 2/3 internal SILK frames in one range-coded
    /// stream; libopus loops silk_Decode until the full duration is produced,
    /// with new_packet set only on the first call (issue #27). With
    /// `FLAG_PACKET_LOST` the same loop drives SILK packet-loss concealment.
    fn decode_silk_frames(
        &mut self,
        rc: &mut RangeCoder,
        lost_flag: i32,
        duration_ms: i32,
        internal_rate: i32,
        api_len: usize,
    ) -> Result<(), &'static str> {
        let internal_frame_size = (duration_ms * internal_rate / 1000) as usize;
        let pcm_i16_len = internal_frame_size * self.channels;
        debug_assert!(pcm_i16_len <= self.w_pcm_i16.len());
        let out_len = api_len * self.channels;
        self.w_silk_out[..out_len].fill(0.0);

        let mut frame_pos = 0usize; // internal-rate samples decoded so far
        let mut new_packet = true;
        while frame_pos < internal_frame_size {
            let ret = {
                let (silk_dec, pcm_i16) = (
                    state_mut(&mut self.silk_dec),
                    state_mut(&mut self.w_pcm_i16),
                );
                silk_dec.decode(
                    rc,
                    &mut pcm_i16[..pcm_i16_len],
                    lost_flag,
                    new_packet,
                    duration_ms,
                    internal_rate,
                )
            };
            new_packet = false;

            if ret < 0 {
                return Err("SILK decoding failed");
            }
            let decoded_samples = ret as usize;
            if decoded_samples == 0 {
                break;
            }

            // SILK decoder outputs planar for THIS frame:
            // ch0 at [0..fl], ch1 at [fl..2*fl].
            //
            // libopus always runs `silk_resampler`, even when the API and SILK
            // internal rates are equal: the Copy path still carries the
            // resampler's `inputDelay` (12 samples at 16k, 4 at 8k). Bypassing
            // it made equal-rate SILK output early versus libopus (and versus
            // opus-rs's own resampled paths).
            let resampled_len =
                decoded_samples * self.sampling_rate as usize / internal_rate as usize;
            let api_pos = frame_pos * self.sampling_rate as usize / internal_rate as usize;
            let copy_len = resampled_len.min(api_len - api_pos);
            debug_assert!(resampled_len * self.channels <= self.w_pcm_resampled.len());
            // Resample channel 0.
            {
                let (res, inp, out) = (
                    &mut self.silk_resampler,
                    state_ref(&self.w_pcm_i16),
                    state_mut(&mut self.w_pcm_resampled),
                );
                res.process(
                    &mut out[..resampled_len],
                    &inp[..decoded_samples],
                    decoded_samples as i32,
                );
            }
            // Resample channel 1 (stereo only).
            if self.channels == 2 {
                let (res, inp, out) = (
                    &mut self.silk_resampler_2,
                    state_ref(&self.w_pcm_i16),
                    state_mut(&mut self.w_pcm_resampled),
                );
                res.process(
                    &mut out[resampled_len..2 * resampled_len],
                    &inp[decoded_samples..2 * decoded_samples],
                    decoded_samples as i32,
                );
            }
            for i in 0..copy_len {
                for ch in 0..self.channels {
                    let v = self.w_pcm_resampled[ch * resampled_len + i] as f32 / 32768.0;
                    let idx = (api_pos + i) * self.channels + ch;
                    if idx < out_len {
                        self.w_silk_out[idx] = v;
                    }
                }
            }
            frame_pos += decoded_samples;
        }
        Ok(())
    }

    /// Number of CELT bands a packet codes, from the bandwidth in its ToC
    /// byte (libopus `opus_decode_frame`, the `CELT_SET_END_BAND` switch).
    ///
    /// This applies to Hybrid packets as well as CELT-only ones: a Hybrid
    /// super-wideband packet codes bands 17 and 18 only. Reading 21 bands
    /// from it, as this used to, desynchronises the range decoder and turns
    /// everything above 8 kHz into noise.
    #[inline(always)]
    fn celt_end_band_from_toc(&self, toc: u8) -> usize {
        match bandwidth_from_toc(toc) {
            Bandwidth::Narrowband => 13,
            Bandwidth::Mediumband | Bandwidth::Wideband => 17,
            Bandwidth::Superwideband => 19,
            Bandwidth::Fullband | Bandwidth::Auto => 21,
        }
    }
}

fn frame_rate_from_params(sampling_rate: i32, frame_size: usize) -> Option<i32> {
    let frame_size = frame_size as i32;
    if frame_size <= 0 {
        return None;
    }
    // libopus `frame_size_select`: a frame is 2.5, 5, 10, 20, 40 or 60 ms.
    // Validating by duration (not by `sampling_rate % frame_size == 0`) is what
    // lets 60 ms through: its rate is 16.67 frames/s, so the old divisibility
    // test rejected it before encoding could start (issue #35). Longer frames
    // (>60 ms) are handled by the caller's multiframe path.
    let fs = sampling_rate;
    let valid = frame_size == fs / 400
        || frame_size == fs / 200
        || frame_size == fs / 100
        || frame_size == fs / 50
        || frame_size == fs / 25
        || 50 * frame_size == 3 * fs;
    if !valid {
        return None;
    }
    Some(fs / frame_size)
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

/// Compute the per-frame sample count implied by the TOC byte at a given
/// sampling rate. For CELT this uses the frame-rate derivation (which handles
/// the 2.5 ms case correctly, unlike integer millisecond arithmetic).
fn frame_samples_from_toc(toc: u8, sampling_rate: i32) -> Option<usize> {
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
        OpusMode::SilkOnly | OpusMode::Hybrid => {
            let duration_ms = frame_duration_ms_from_toc(toc);
            Some((sampling_rate as i64 * duration_ms as i64 / 1000) as usize)
        }
    }
}

fn channels_from_toc(toc: u8) -> usize {
    if toc & 0x04 != 0 { 2 } else { 1 }
}

/// Crossfade two signals using a squared-sine window (libopus smooth_fade).
/// `window` is the 120-sample CELT window at 48 kHz; `inc` = 48000/Fs strides it.
fn smooth_fade(
    in1: &[f32],
    in2: &[f32],
    out: &mut [f32],
    overlap: usize,
    channels: usize,
    window: &[f32],
    inc: usize,
) {
    for c in 0..channels {
        for i in 0..overlap {
            let wi = i * inc;
            if wi >= window.len() {
                break;
            }
            let w = window[wi] * window[wi];
            out[i * channels + c] = w * in2[i * channels + c] + (1.0 - w) * in1[i * channels + c];
        }
    }
}

/// Parse an Opus frame length per RFC 6716 §3.2.1, identical to libopus
/// `parse_size()`:
///   - `0`: no frame (DTX / lost packet)
///   - `1..=251`: length of the frame in bytes (one byte consumed)
///   - `252..=255`: a second byte is read; length = `second*4 + first`
///
/// Returns `(length, bytes_consumed)`.
pub(crate) fn parse_frame_size(data: &[u8]) -> Result<(usize, usize), &'static str> {
    let first = *data.first().ok_or("truncated frame length")? as usize;
    if first < 252 {
        Ok((first, 1))
    } else {
        let second = *data.get(1).ok_or("truncated frame length")? as usize;
        Ok((second * 4 + first, 2))
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
        enc.bitrate_bps = 64000;
        enc.complexity = 5;
        enc.use_cbr = true;

        // 48 samples at 48kHz = 1ms → frame_rate=1000, invalid for Hybrid mode.
        let input = vec![0.0f32; 48 * 2]; // stereo interleaved
        let mut output = vec![0u8; 256];

        let result = enc.encode(&input, 48, &mut output);
        assert!(
            result.is_err(),
            "encode with invalid frame_size=48 should return Err, not panic"
        );
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
