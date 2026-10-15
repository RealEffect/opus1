//! Shared helpers for the CELT encoder tests (`tests/celtenc_*.rs`).
//!
//! These tests drive `CeltEncoder` directly, through its public functions.
//! The CELT payload gets the matching CELT-only TOC byte in front, which
//! makes it a valid Opus packet. C libopus decodes every packet, from
//! opus-rs and from the C encoder alike, so a difference in the numbers
//! comes from the encoder alone.
#![allow(dead_code)]

use opus_rs::celt::CeltEncoder;
use opus_rs::hp_cutoff::dc_reject_float;
use opus_rs::modes::{SPREAD_ICDF, TAPSET_ICDF, TF_SELECT_TABLE, TRIM_ICDF, default_mode};
use opus_rs::quant_bands::E_PROB_MODEL;
use opus_rs::range_coder::RangeCoder;
use std::os::raw::{c_int, c_uchar};

pub mod c_celt;

/// Sampling rate of every test in this group.
pub const FS: usize = 48_000;
/// Samples in a 20 ms frame.
pub const FRAME: usize = 960;
/// Largest CELT payload of one frame, in bytes.
pub const MAX_PAYLOAD: usize = 1275;
/// Delay of a CELT-only stream through encoder and decoder, in samples.
pub const CELT_DELAY: usize = 120;

// ---------------------------------------------------------------------------
// The C encoder
// ---------------------------------------------------------------------------

/// Opaque C `OpusEncoder`.
#[repr(C)]
pub struct COpusEncoder {
    _private: [u8; 0],
}

// The raw encoder functions of libopus. The safe wrapper of the `opus`
// dev-dependency has no call for the private requests below. That crate
// links the library, so the declarations here need no dependency of their
// own.
unsafe extern "C" {
    fn opus_encoder_create(
        fs: i32,
        channels: c_int,
        application: c_int,
        error: *mut c_int,
    ) -> *mut COpusEncoder;
    fn opus_encoder_ctl(st: *mut COpusEncoder, request: c_int, ...) -> c_int;
    fn opus_encode_float(
        st: *mut COpusEncoder,
        pcm: *const f32,
        frame_size: c_int,
        data: *mut c_uchar,
        max_data_bytes: i32,
    ) -> i32;
    fn opus_encoder_destroy(st: *mut COpusEncoder);
}

const OPUS_APPLICATION_RESTRICTED_LOWDELAY: c_int = 2051;
const OPUS_SET_BITRATE_REQUEST: c_int = 4002;
const OPUS_SET_VBR_REQUEST: c_int = 4006;
const OPUS_SET_BANDWIDTH_REQUEST: c_int = 4008;
const OPUS_SET_COMPLEXITY_REQUEST: c_int = 4010;
const OPUS_SET_VBR_CONSTRAINT_REQUEST: c_int = 4020;
// Private requests (celt.h, opus_private.h).
const OPUS_SET_LFE_REQUEST: c_int = 10024;
const OPUS_SET_ENERGY_MASK_REQUEST: c_int = 10026;
const OPUS_SET_FORCE_MODE_REQUEST: c_int = 11002;
const MODE_CELT_ONLY: c_int = 1002;

const OPUS_BANDWIDTH_NARROWBAND: c_int = 1101;
const OPUS_BANDWIDTH_FULLBAND: c_int = 1105;

/// Sampling rate and frame size of a stream.
#[derive(Clone, Copy, Debug)]
pub struct Format {
    /// Sampling rate of the input, in Hz.
    pub fs: usize,
    /// Samples per channel in one frame, at `fs`.
    pub frame: usize,
}

impl Format {
    /// 48 kHz, 20 ms: what the helpers without a `Format` use.
    pub const DEFAULT: Format = Format {
        fs: FS,
        frame: FRAME,
    };

    /// `frame_ms_x2` is the frame duration in half milliseconds: 5 for
    /// 2.5 ms, 10, 20 or 40 for 20 ms.
    pub fn new(fs: usize, frame_ms_x2: usize) -> Self {
        Format {
            fs,
            frame: fs * frame_ms_x2 / 2000,
        }
    }

    /// CELT frame size index: 0 for 2.5 ms up to 3 for 20 ms.
    pub fn lm(&self) -> usize {
        match self.frame * 400 / self.fs {
            1 => 0,
            2 => 1,
            4 => 2,
            8 => 3,
            other => panic!("not a CELT frame size: {other} / 400 s"),
        }
    }

    /// Delay of a CELT-only stream through encoder and decoder, in samples
    /// at `fs` (2.5 ms).
    pub fn delay(&self) -> usize {
        CELT_DELAY * self.fs / FS
    }

    /// Bytes of a constant-bitrate packet, TOC byte included
    /// (opus_encoder.c: `cbr_bytes`).
    pub fn cbr_packet_bytes(&self, bitrate: i32) -> usize {
        let bits = bitrate as usize * self.frame / self.fs;
        (bits + 4) / 8
    }
}

/// Settings shared by both encoders.
#[derive(Clone, Debug)]
pub struct Cfg {
    pub channels: usize,
    pub bitrate: i32,
    pub vbr: bool,
    pub constrained_vbr: bool,
    /// CELT end band: 13 narrowband, 17 wideband, 19 super-wideband, 21 fullband.
    pub end_band: usize,
    pub lfe: bool,
    /// 21 values per channel, log2 units.
    pub energy_mask: Option<Vec<f32>>,
}

impl Cfg {
    pub fn new(channels: usize, bitrate: i32) -> Self {
        Cfg {
            channels,
            bitrate,
            vbr: true,
            constrained_vbr: true,
            end_band: 21,
            lfe: false,
            energy_mask: None,
        }
    }

    /// The stream the multistream encoder makes of an LFE channel: mono,
    /// CELT-only, narrowband, LFE flag set.
    pub fn lfe(bitrate: i32) -> Self {
        Cfg {
            end_band: 13,
            lfe: true,
            ..Cfg::new(1, bitrate)
        }
    }

    /// Bytes of a constant-bitrate packet, TOC byte included
    /// (opus_encoder.c: `cbr_bytes`).
    pub fn cbr_packet_bytes(&self) -> usize {
        let bits = self.bitrate as usize * FRAME / FS;
        (bits + 4) / 8
    }
}

/// Complexity of both encoders. From 7 up libopus runs its tonality
/// analysis, which opus-rs does not have. CELT itself works the same at
/// complexity 5, 6 and 7.
pub const COMPLEXITY: i32 = 6;

/// Encode `pcm` (interleaved) with C libopus in CELT-only mode, 20 ms frames.
///
/// The application is "restricted low delay": the input reaches CELT without
/// the 4 ms delay the other applications add, as it does in the opus-rs half
/// of each test.
pub fn enc_c(cfg: &Cfg, pcm: &[f32]) -> Vec<Vec<u8>> {
    enc_c_fmt(cfg, Format::DEFAULT, pcm, None)
}

/// `enc_c` at another sampling rate or frame size.
///
/// With `masks`, frame `k` is coded with the energy mask `masks[k]`, the way
/// the multistream encoder gives each frame its own mask. `cfg.energy_mask`
/// is then not used.
pub fn enc_c_fmt(cfg: &Cfg, fmt: Format, pcm: &[f32], masks: Option<&[Vec<f32>]>) -> Vec<Vec<u8>> {
    enc_c_schedule(cfg, fmt, pcm, masks, None)
}

/// `enc_c_fmt` with the LFE flag set anew before every frame: frame `k` is
/// coded with `OPUS_SET_LFE(lfe[k])`. `cfg.lfe` is then not used.
pub fn enc_c_schedule(
    cfg: &Cfg,
    fmt: Format,
    pcm: &[f32],
    masks: Option<&[Vec<f32>]>,
    lfe: Option<&[bool]>,
) -> Vec<Vec<u8>> {
    let mut err: c_int = 0;
    let st = unsafe {
        opus_encoder_create(
            fmt.fs as i32,
            cfg.channels as c_int,
            OPUS_APPLICATION_RESTRICTED_LOWDELAY,
            &mut err,
        )
    };
    assert_eq!(err, 0, "opus_encoder_create");
    let ctl = |request: c_int, value: c_int| {
        let ret = unsafe { opus_encoder_ctl(st, request, value) };
        assert_eq!(ret, 0, "opus_encoder_ctl({request}, {value})");
    };
    ctl(OPUS_SET_BITRATE_REQUEST, cfg.bitrate);
    ctl(OPUS_SET_VBR_REQUEST, cfg.vbr as c_int);
    ctl(
        OPUS_SET_VBR_CONSTRAINT_REQUEST,
        cfg.constrained_vbr as c_int,
    );
    ctl(OPUS_SET_COMPLEXITY_REQUEST, COMPLEXITY);
    ctl(OPUS_SET_FORCE_MODE_REQUEST, MODE_CELT_ONLY);
    let bandwidth = match cfg.end_band {
        13 => OPUS_BANDWIDTH_NARROWBAND,
        21 => OPUS_BANDWIDTH_FULLBAND,
        other => panic!("no bandwidth for end band {other}"),
    };
    ctl(OPUS_SET_BANDWIDTH_REQUEST, bandwidth);
    if cfg.lfe {
        ctl(OPUS_SET_LFE_REQUEST, 1);
    }
    // libopus keeps this pointer and reads the values on every frame. The
    // buffer lives until the encoder is destroyed.
    let mut mask_buf: Vec<f32> = match (masks, &cfg.energy_mask) {
        (Some(masks), _) => masks[0].clone(),
        (None, Some(mask)) => mask.clone(),
        (None, None) => Vec::new(),
    };
    if !mask_buf.is_empty() {
        assert_eq!(mask_buf.len(), 21 * cfg.channels);
        let ret = unsafe { opus_encoder_ctl(st, OPUS_SET_ENERGY_MASK_REQUEST, mask_buf.as_ptr()) };
        assert_eq!(ret, 0, "OPUS_SET_ENERGY_MASK");
    }

    let step = fmt.frame * cfg.channels;
    let mut out = Vec::new();
    let mut pkt = vec![0u8; 4000];
    for (k, chunk) in pcm.chunks_exact(step).enumerate() {
        if let Some(masks) = masks {
            mask_buf.copy_from_slice(&masks[k]);
        }
        if let Some(lfe) = lfe {
            ctl(OPUS_SET_LFE_REQUEST, lfe[k] as c_int);
        }
        let n = unsafe {
            opus_encode_float(
                st,
                chunk.as_ptr(),
                fmt.frame as c_int,
                pkt.as_mut_ptr(),
                pkt.len() as i32,
            )
        };
        assert!(n > 0, "libopus encode failed at packet {k}: {n}");
        out.push(pkt[..n as usize].to_vec());
    }
    unsafe { opus_encoder_destroy(st) };
    out
}

// ---------------------------------------------------------------------------
// The opus-rs CELT encoder
// ---------------------------------------------------------------------------

/// TOC byte of a CELT-only packet with one 20 ms frame (RFC 6716 section 3.1).
pub fn celt_toc(end_band: usize, channels: usize) -> u8 {
    celt_toc_lm(end_band, channels, 3)
}

/// TOC byte of a CELT-only packet with one frame of 2.5 ms (`lm` 0), 5 ms,
/// 10 ms or 20 ms (`lm` 3).
pub fn celt_toc_lm(end_band: usize, channels: usize, lm: usize) -> u8 {
    let config: u8 = match end_band {
        13 => 16,
        17 => 20,
        19 => 24,
        _ => 28,
    } + lm as u8;
    (config << 3) | (((channels == 2) as u8) << 2)
}

/// `CeltEncoder` set up from `cfg`, the way `OpusEncoder` sets up its CELT
/// encoder for a CELT-only frame.
pub fn rs_encoder(cfg: &Cfg) -> CeltEncoder {
    rs_encoder_fmt(cfg, Format::DEFAULT)
}

/// `rs_encoder` at another sampling rate.
pub fn rs_encoder_fmt(cfg: &Cfg, fmt: Format) -> CeltEncoder {
    let mut enc = CeltEncoder::with_sampling_rate(default_mode(), cfg.channels, fmt.fs as i32);
    enc.complexity = COMPLEXITY;
    enc.set_end_band(cfg.end_band);
    enc.set_bitrate(cfg.bitrate);
    enc.set_vbr(cfg.vbr);
    enc.set_constrained_vbr(cfg.constrained_vbr);
    enc.set_lfe(cfg.lfe);
    enc.set_energy_mask(cfg.energy_mask.as_deref());
    enc
}

/// Encode one 20 ms frame into an Opus packet: the CELT payload behind the
/// TOC byte. `frame` is interleaved. `payload_bytes` is the room for the
/// CELT payload.
pub fn rs_encode_frame(
    enc: &mut CeltEncoder,
    cfg: &Cfg,
    frame: &[f32],
    payload_bytes: usize,
) -> Vec<u8> {
    rs_encode_frame_fmt(enc, cfg, Format::DEFAULT, frame, payload_bytes)
}

/// `rs_encode_frame` for a frame of `fmt.frame` samples.
pub fn rs_encode_frame_fmt(
    enc: &mut CeltEncoder,
    cfg: &Cfg,
    fmt: Format,
    frame: &[f32],
    payload_bytes: usize,
) -> Vec<u8> {
    // `CeltEncoder` takes one channel after the other.
    let mut planar = vec![0.0f32; fmt.frame * cfg.channels];
    for i in 0..fmt.frame {
        for c in 0..cfg.channels {
            planar[c * fmt.frame + i] = frame[i * cfg.channels + c];
        }
    }
    let mut rc = RangeCoder::new_encoder(payload_bytes as u32);
    enc.encode_with_budget_vbr(
        &planar,
        fmt.frame,
        &mut rc,
        0,
        (payload_bytes * 8) as i32,
        cfg.vbr,
    );
    rc.done();
    assert_eq!(rc.error, 0, "range coder error");
    let n = rc.storage as usize;
    assert!(n <= payload_bytes);
    let mut pkt = Vec::with_capacity(n + 1);
    pkt.push(celt_toc_lm(cfg.end_band, cfg.channels, fmt.lm()));
    pkt.extend_from_slice(&rc.buf[..n]);
    pkt
}

/// Encode `pcm` (interleaved) with the opus-rs CELT encoder, 20 ms frames.
///
/// The input first passes the 3 Hz DC filter that libopus and `OpusEncoder`
/// put in front of CELT.
pub fn enc_rs(cfg: &Cfg, pcm: &[f32]) -> Vec<Vec<u8>> {
    enc_rs_with(&mut rs_encoder(cfg), cfg, pcm)
}

/// `enc_rs` with an encoder the caller has set up.
pub fn enc_rs_with(enc: &mut CeltEncoder, cfg: &Cfg, pcm: &[f32]) -> Vec<Vec<u8>> {
    enc_rs_with_fmt(enc, cfg, Format::DEFAULT, pcm, None)
}

/// `enc_rs` at another sampling rate or frame size. With `masks`, frame `k`
/// is coded with the energy mask `masks[k]`.
pub fn enc_rs_fmt(cfg: &Cfg, fmt: Format, pcm: &[f32], masks: Option<&[Vec<f32>]>) -> Vec<Vec<u8>> {
    enc_rs_with_fmt(&mut rs_encoder_fmt(cfg, fmt), cfg, fmt, pcm, masks)
}

/// `enc_rs_fmt` with an encoder the caller has set up.
pub fn enc_rs_with_fmt(
    enc: &mut CeltEncoder,
    cfg: &Cfg,
    fmt: Format,
    pcm: &[f32],
    masks: Option<&[Vec<f32>]>,
) -> Vec<Vec<u8>> {
    enc_rs_schedule(enc, cfg, fmt, pcm, masks, None)
}

/// `enc_rs_with_fmt` with the LFE flag set anew before every frame: frame
/// `k` is coded with `set_lfe(lfe[k])`.
pub fn enc_rs_schedule(
    enc: &mut CeltEncoder,
    cfg: &Cfg,
    fmt: Format,
    pcm: &[f32],
    masks: Option<&[Vec<f32>]>,
    lfe: Option<&[bool]>,
) -> Vec<Vec<u8>> {
    let payload_bytes = if cfg.vbr {
        MAX_PAYLOAD
    } else {
        fmt.cbr_packet_bytes(cfg.bitrate) - 1
    };
    let step = fmt.frame * cfg.channels;
    let mut hp_mem = [0.0f32; 4];
    let mut filtered = vec![0.0f32; step];
    pcm.chunks_exact(step)
        .enumerate()
        .map(|(k, chunk)| {
            dc_reject_float(
                chunk,
                3,
                &mut filtered,
                &mut hp_mem,
                fmt.frame,
                cfg.channels,
                fmt.fs as i32,
            );
            if let Some(masks) = masks {
                enc.set_energy_mask(Some(&masks[k]));
            }
            if let Some(lfe) = lfe {
                enc.set_lfe(lfe[k]);
            }
            rs_encode_frame_fmt(enc, cfg, fmt, &filtered, payload_bytes)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Decoding and measuring
// ---------------------------------------------------------------------------

/// Decode a stream with C libopus at 48 kHz. Returns interleaved samples.
pub fn dec_c(packets: &[Vec<u8>], channels: usize) -> Vec<f32> {
    dec_c_fmt(packets, channels, Format::DEFAULT)
}

/// `dec_c` at another sampling rate or frame size.
pub fn dec_c_fmt(packets: &[Vec<u8>], channels: usize, fmt: Format) -> Vec<f32> {
    let ch = if channels == 2 {
        opus::Channels::Stereo
    } else {
        opus::Channels::Mono
    };
    let mut dec = opus::Decoder::new(fmt.fs as u32, ch).expect("opus decoder");
    let mut out = Vec::with_capacity(packets.len() * fmt.frame * channels);
    let mut pcm = vec![0f32; fmt.frame * channels];
    for (k, p) in packets.iter().enumerate() {
        let n = dec
            .decode_float(p, &mut pcm, false)
            .unwrap_or_else(|e| panic!("libopus rejected packet {k} ({} bytes): {e}", p.len()));
        assert_eq!(n, fmt.frame, "packet {k} decoded to {n} samples");
        out.extend_from_slice(&pcm);
    }
    out
}

/// Signal-to-noise ratio in dB of `decoded` against `reference`, both
/// interleaved, with `decoded` late by `CELT_DELAY` samples. The first
/// `skip_frames` frames are left out.
pub fn snr_db(reference: &[f32], decoded: &[f32], channels: usize, skip_frames: usize) -> f64 {
    snr_db_delay(
        reference,
        decoded,
        channels,
        skip_frames * FRAME,
        CELT_DELAY,
    )
}

/// `snr_db` with `decoded` late by `delay` samples and without the first
/// `skip` samples.
pub fn snr_db_delay(
    reference: &[f32],
    decoded: &[f32],
    channels: usize,
    skip: usize,
    delay: usize,
) -> f64 {
    let n = reference.len().min(decoded.len()) / channels;
    let (mut s, mut e) = (0f64, 0f64);
    for i in skip..n - delay {
        for c in 0..channels {
            let r = f64::from(reference[i * channels + c]);
            let d = f64::from(decoded[(i + delay) * channels + c]);
            s += r * r;
            e += (r - d) * (r - d);
        }
    }
    10.0 * (s / e.max(1e-30)).log10()
}

/// Mean packet size in bytes, TOC byte included, without the first
/// `skip_frames` packets.
pub fn mean_bytes(packets: &[Vec<u8>], skip_frames: usize) -> f64 {
    let p = &packets[skip_frames..];
    p.iter().map(|x| x.len() as f64).sum::<f64>() / p.len() as f64
}

/// Energy of `x` (one channel) in each of `edges.len() - 1` frequency
/// ranges, `edges` in Hz. Computed from 20 ms blocks with a Hann window and
/// a plain discrete Fourier transform over the bins that are needed.
pub fn range_energies(x: &[f32], edges: &[f64]) -> Vec<f64> {
    let n = FRAME;
    let bin_hz = FS as f64 / n as f64;
    let last_bin = (edges[edges.len() - 1] / bin_hz).ceil() as usize;
    let window: Vec<f64> = (0..n)
        .map(|i| 0.5 - 0.5 * (std::f64::consts::TAU * i as f64 / n as f64).cos())
        .collect();
    // One rotation table serves every bin: cos and sin of TAU * m / n.
    let table: Vec<(f64, f64)> = (0..n)
        .map(|m| (std::f64::consts::TAU * m as f64 / n as f64).sin_cos())
        .map(|(s, c)| (c, s))
        .collect();
    let mut out = vec![0f64; edges.len() - 1];
    for block in x.chunks_exact(n) {
        for k in 0..=last_bin {
            let f = k as f64 * bin_hz;
            let Some(r) = edges.windows(2).position(|w| f >= w[0] && f < w[1]) else {
                continue;
            };
            let (mut re, mut im) = (0f64, 0f64);
            for (i, &v) in block.iter().enumerate() {
                let (c, s) = table[(i * k) % n];
                let w = f64::from(v) * window[i];
                re += w * c;
                im -= w * s;
            }
            out[r] += re * re + im * im;
        }
    }
    out
}

/// Signal-to-noise ratio in dB of one channel of `decoded` against
/// `reference`, separately for each frequency range of `edges` (Hz). The
/// first `skip_frames` frames are left out.
pub fn range_snr_db(
    reference: &[f32],
    decoded: &[f32],
    channels: usize,
    channel: usize,
    skip_frames: usize,
    edges: &[f64],
) -> Vec<f64> {
    let n = reference.len().min(decoded.len()) / channels - CELT_DELAY;
    let range = skip_frames * FRAME..n;
    let signal: Vec<f32> = range
        .clone()
        .map(|i| reference[i * channels + channel])
        .collect();
    let error: Vec<f32> = range
        .map(|i| decoded[(i + CELT_DELAY) * channels + channel] - reference[i * channels + channel])
        .collect();
    let s = range_energies(&signal, edges);
    let e = range_energies(&error, edges);
    s.iter()
        .zip(&e)
        .map(|(s, e)| 10.0 * (s / e.max(1e-30)).log10())
        .collect()
}

// ---------------------------------------------------------------------------
// Reading what an encoder wrote
// ---------------------------------------------------------------------------

/// The first symbols of a CELT frame, in the order the decoder reads them
/// (libopus celt_decode_with_ec). They show the encoder's decisions
/// directly, where the decoded audio only shows their sum.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Header {
    /// The frame is flagged as silence. Nothing else is coded then.
    pub silence: bool,
    /// Pitch post-filter: period, gain index (0 to 7) and tap set.
    pub postfilter: Option<(u32, u32, i32)>,
    pub transient: bool,
    /// Coarse energy coded without prediction from the frame before.
    pub intra: bool,
    /// Coarse energy step of each band, `coarse[21 * channel + band]`.
    pub coarse: Vec<i32>,
    /// Time-frequency resolution change of each band.
    pub tf: Vec<i32>,
    /// Spreading decision (0 none to 3 aggressive), `None` when the packet
    /// has no room for it.
    pub spread: Option<i32>,
    /// Dynamic allocation: how many boost steps each band got.
    pub boost: Vec<i32>,
    /// Allocation trim (0 to 10), `None` when the packet has no room for it.
    pub trim: Option<i32>,
}

/// Read the header of a CELT-only frame. `payload` is the packet without
/// its TOC byte, `end` the number of coded bands, `lm` the frame size index
/// (3 for 20 ms).
pub fn parse_header(payload: &[u8], channels: usize, end: usize, lm: usize) -> Header {
    parse_header_rc(payload, channels, 0, end, lm).0
}

/// `parse_header` for a frame that starts at band `start`: the CELT part
/// of a Hybrid frame (start 17), coded without the SILK part in front of
/// it. Such a frame has no post-filter, and the bands below `start` keep
/// their default values in the result.
pub fn parse_header_from(payload: &[u8], channels: usize, start: usize, end: usize, lm: usize) -> Header {
    parse_header_rc(payload, channels, start, end, lm).0
}

/// `parse_header`, and the range decoder as the header leaves it.
fn parse_header_rc(
    payload: &[u8],
    channels: usize,
    start: usize,
    end: usize,
    lm: usize,
) -> (Header, RangeCoder) {
    let mode = default_mode();
    let nb = mode.nb_ebands;
    let total_bits = (payload.len() * 8) as i32;
    let mut rc = RangeCoder::new_decoder(payload);
    let mut h = Header::default();
    let tell = rc.tell();
    h.silence = if tell >= total_bits {
        true
    } else if tell == 1 {
        rc.decode_bit_logp(15)
    } else {
        false
    };
    if h.silence {
        return (h, rc);
    }
    if start == 0 && rc.tell() + 16 <= total_bits && rc.decode_bit_logp(1) {
        let octave = rc.dec_uint(6);
        let period = (16 << octave) + rc.dec_bits(4 + octave) - 1;
        let gain = rc.dec_bits(3);
        let tapset = if rc.tell() + 2 <= total_bits {
            rc.decode_icdf(&TAPSET_ICDF, 2)
        } else {
            0
        };
        h.postfilter = Some((period, gain, tapset));
    }
    if lm > 0 && rc.tell() + 3 <= total_bits {
        h.transient = rc.decode_bit_logp(3);
    }
    if rc.tell() + 3 <= total_bits {
        h.intra = rc.decode_bit_logp(3);
    }

    // Coarse energy (unquant_coarse_energy).
    let model = &E_PROB_MODEL[lm][h.intra as usize];
    h.coarse = vec![0; nb * channels];
    for i in start..end {
        for c in 0..channels {
            let left = total_bits - rc.tell();
            h.coarse[c * nb + i] = if left >= 15 {
                let pi = 2 * i.min(20);
                rc.laplace_decode((model[pi] as u32) << 7, (model[pi + 1] as i32) << 6)
            } else if left >= 2 {
                let s = rc.decode_icdf(&[2, 1, 0], 2);
                (s >> 1) ^ -(s & 1)
            } else if left >= 1 {
                -(rc.decode_bit_logp(1) as i32)
            } else {
                -1
            };
        }
    }

    // Time-frequency resolution (tf_decode).
    {
        let mut logp = if h.transient { 2 } else { 4 };
        let mut budget = total_bits;
        let mut tell = rc.tell();
        let select_reserved = (lm > 0 && tell + logp < budget) as i32;
        budget -= select_reserved;
        let (mut current, mut changed) = (0, 0);
        h.tf = vec![0; end];
        for v in h.tf[start..].iter_mut() {
            if tell + logp <= budget {
                current ^= rc.decode_bit_logp(logp as u32) as i32;
                tell = rc.tell();
                changed |= current;
            }
            *v = current;
            logp = if h.transient { 4 } else { 5 };
        }
        let row = &TF_SELECT_TABLE[lm];
        let t = 4 * h.transient as usize;
        let mut select = 0;
        if select_reserved > 0 && row[t + changed as usize] != row[t + 2 + changed as usize] {
            select = rc.decode_bit_logp(1) as usize;
        }
        for v in h.tf[start..].iter_mut() {
            *v = row[t + 2 * select + *v as usize] as i32;
        }
    }

    if rc.tell() + 4 <= total_bits {
        h.spread = Some(rc.decode_icdf(&SPREAD_ICDF, 5));
    }

    // Dynamic allocation, in units of 1/8 bit.
    h.boost = vec![0; end];
    let mut logp = 6i32;
    let mut total = total_bits << 3;
    let mut tell = rc.tell_frac();
    for i in start..end {
        let n = ((mode.e_bands[i + 1] - mode.e_bands[i]) as i32) << lm;
        let cap =
            ((mode.cache.caps[nb * (2 * lm + channels - 1) + i] as i32 + 64) * channels as i32 * n)
                >> 2;
        let width = channels as i32 * n;
        let quanta = (width << 3).min((6 << 3).max(width));
        let mut loop_logp = logp;
        let mut boost = 0;
        while tell + (loop_logp << 3) < total && boost < cap {
            let flag = rc.decode_bit_logp(loop_logp as u32);
            tell = rc.tell_frac();
            if !flag {
                break;
            }
            boost += quanta;
            total -= quanta;
            loop_logp = 1;
            h.boost[i] += 1;
        }
        if boost > 0 {
            logp = (logp - 1).max(2);
        }
    }
    if rc.tell_frac() + (6 << 3) <= total {
        h.trim = Some(rc.decode_icdf(&TRIM_ICDF, 7));
    }
    (h, rc)
}

/// What a CELT frame says about the bit allocation, read after the header
/// (libopus celt_decode_with_ec: clt_compute_allocation and
/// unquant_fine_energy).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Allocation {
    /// First band coded with intensity stereo. 0 for mono.
    pub intensity: i32,
    /// Left and right are coded separately, not as mid and side.
    pub dual_stereo: bool,
    /// Number of the band after the last one that gets pulses.
    pub coded_bands: i32,
    /// Bits for the shape of each band, in units of 1/8 bit.
    pub pulses: Vec<i32>,
    /// Fine energy bits of each band, per channel.
    pub fine_bits: Vec<i32>,
    /// Fine energy value of each band, `fine[21 * channel + band]`.
    pub fine: Vec<u32>,
}

/// Read the header and the allocation of a CELT-only frame. `None` for a
/// frame flagged as silence, which has no allocation.
pub fn parse_frame(
    payload: &[u8],
    channels: usize,
    end: usize,
    lm: usize,
) -> (Header, Option<Allocation>) {
    let (h, rest) = parse_frame_rc(payload, channels, end, lm);
    (h, rest.map(|r| r.allocation))
}

/// What `parse_frame_rc` leaves for reading the rest of a frame.
struct FrameRest {
    allocation: Allocation,
    /// The range decoder after the fine energy.
    rc: RangeCoder,
    /// The allocator's balance, which the band decoder goes on with.
    balance: i32,
    /// 1/8 bit kept for the anti-collapse flag, or 0.
    anti_collapse_rsv: i32,
}

/// The anti-collapse flag of a CELT-only frame (libopus
/// celt_decode_with_ec: the raw bit after the bands). `None` when the
/// frame has no such flag: it is not a transient, is shorter than 10 ms, or
/// has no room for it.
///
/// The flag comes after the shapes of all bands, so they are decoded here
/// with the band decoder of opus-rs.
pub fn anti_collapse_flag(payload: &[u8], channels: usize, end: usize, lm: usize) -> Option<bool> {
    let mode = default_mode();
    let (h, rest) = parse_frame_rc(payload, channels, end, lm);
    let FrameRest {
        allocation: a,
        mut rc,
        mut balance,
        anti_collapse_rsv,
    } = rest?;
    if anti_collapse_rsv == 0 {
        return None;
    }
    let n = mode.short_mdct_size << lm;
    // Room past the last band for the vector kernels, as the decoder has.
    let mut x = vec![0.0f32; n * channels + 512];
    let (xs, ys) = x.split_at_mut(n);
    let mut masks = vec![0u32; mode.nb_ebands * channels];
    let band_e = vec![1.0f32; mode.nb_ebands * channels];
    let mut dual_stereo = a.dual_stereo;
    opus_rs::bands::quant_all_bands(
        false,
        mode,
        0,
        end,
        xs,
        if channels == 2 { Some(ys) } else { None },
        &mut masks,
        &band_e,
        &a.pulses,
        h.transient,
        h.spread.unwrap_or(2),
        &mut dual_stereo,
        a.intensity as usize,
        &h.tf,
        ((payload.len() as i32 * 8) << 3) - anti_collapse_rsv,
        &mut balance,
        &mut rc,
        lm as i32,
        a.coded_bands,
        true,
        false,
        &mut 0u32,
    );
    Some(rc.dec_bits(1) != 0)
}

/// `parse_frame`, and what is needed to read on.
fn parse_frame_rc(
    payload: &[u8],
    channels: usize,
    end: usize,
    lm: usize,
) -> (Header, Option<FrameRest>) {
    let mode = default_mode();
    let nb = mode.nb_ebands;
    let (h, mut rc) = parse_header_rc(payload, channels, 0, end, lm);
    if h.silence {
        return (h, None);
    }
    let mut cap = vec![0i32; nb];
    let mut offsets = vec![0i32; nb];
    for i in 0..nb {
        let n = ((mode.e_bands[i + 1] - mode.e_bands[i]) as i32) << lm;
        cap[i] =
            ((mode.cache.caps[nb * (2 * lm + channels - 1) + i] as i32 + 64) * channels as i32 * n)
                >> 2;
        if i < end {
            let width = channels as i32 * n;
            offsets[i] = h.boost[i] * (width << 3).min((6 << 3).max(width));
        }
    }
    let mut bits = ((payload.len() as i32 * 8) << 3) - rc.tell_frac() - 1;
    let anti_collapse_rsv = if h.transient && lm >= 2 && bits >= ((lm as i32 + 2) << 3) {
        1 << 3
    } else {
        0
    };
    bits -= anti_collapse_rsv;
    let mut a = Allocation {
        pulses: vec![0; nb],
        fine_bits: vec![0; nb],
        fine: vec![0; nb * channels],
        ..Allocation::default()
    };
    let mut fine_priority = vec![0i32; nb];
    let (mut dual_stereo, mut balance) = (0, 0);
    a.coded_bands = opus_rs::rate::clt_compute_allocation(
        mode,
        0,
        end,
        &offsets,
        &cap,
        h.trim.unwrap_or(5),
        &mut a.intensity,
        &mut dual_stereo,
        bits,
        &mut balance,
        &mut a.pulses,
        &mut a.fine_bits,
        &mut fine_priority,
        channels as i32,
        lm as i32,
        &mut rc,
        false,
        0,
        0,
    );
    a.dual_stereo = dual_stereo != 0;
    for i in 0..end {
        if a.fine_bits[i] > 0 {
            for c in 0..channels {
                a.fine[c * nb + i] = rc.dec_bits(a.fine_bits[i] as u32);
            }
        }
    }
    let rest = FrameRest {
        allocation: a,
        rc,
        balance,
        anti_collapse_rsv,
    };
    (h, Some(rest))
}

/// The header of every packet of a stream of single CELT-only frames.
pub fn headers(packets: &[Vec<u8>], channels: usize, end: usize, lm: usize) -> Vec<Header> {
    packets
        .iter()
        .map(|p| parse_header(&p[1..], channels, end, lm))
        .collect()
}

// ---------------------------------------------------------------------------
// Signals
// ---------------------------------------------------------------------------

/// Deterministic white noise in [-1, 1).
pub struct Noise(pub u32);

impl Noise {
    pub fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (self.0 >> 8) as f32 / 8_388_608.0 - 1.0
    }
}

/// What an LFE channel carries, mono. Twice a second a thump that sweeps
/// from 90 Hz down to 45 Hz and dies away, over a steady 35 Hz rumble and a
/// little noise below about 100 Hz.
pub fn lfe_signal(frames: usize) -> Vec<f32> {
    let mut noise = Noise(7);
    let mut lowpass = 0.0f64;
    (0..frames * FRAME)
        .map(|i| {
            let t = i as f64 / FS as f64;
            let tb = t % 0.5;
            let phase = std::f64::consts::TAU * (90.0 * tb - 45.0 * tb * tb);
            let thump = 0.5 * (-6.0 * tb).exp() * phase.sin();
            let rumble = 0.08 * (std::f64::consts::TAU * 35.0 * t).sin();
            lowpass += 0.012 * (f64::from(noise.next()) - lowpass);
            (thump + rumble + 0.2 * lowpass) as f32
        })
        .collect()
}

/// Signals an LFE channel can carry, mono, `seconds` long at `fs` Hz.
///
/// - "thumps": `lfe_signal` at any sampling rate.
/// - "50 Hz tone": steady and as periodic as a signal gets.
/// - "bursts": 60 Hz for 100 ms, then 300 ms of digital silence. The level
///   jumps up and down, and whole frames are silent.
/// - "white noise": content in every band. LFE mode has to hold all of it
///   down except the lowest 400 Hz.
pub fn lfe_signals(fs: usize, seconds: usize) -> Vec<(&'static str, Vec<f32>)> {
    use std::f64::consts::TAU;
    let n = fs * seconds;
    let t = |i: usize| i as f64 / fs as f64;
    let mut noise = Noise(7);
    let mut lowpass = 0.0f64;
    let thumps = (0..n)
        .map(|i| {
            let tb = t(i) % 0.5;
            let thump = 0.5 * (-6.0 * tb).exp() * (TAU * (90.0 * tb - 45.0 * tb * tb)).sin();
            let rumble = 0.08 * (TAU * 35.0 * t(i)).sin();
            lowpass += 0.012 * 48_000.0 / fs as f64 * (f64::from(noise.next()) - lowpass);
            (thump + rumble + 0.2 * lowpass) as f32
        })
        .collect();
    let tone = (0..n)
        .map(|i| (0.4 * (TAU * 50.0 * t(i)).sin()) as f32)
        .collect();
    let bursts = (0..n)
        .map(|i| {
            let tb = t(i) % 0.4;
            if tb < 0.1 {
                (0.6 * (TAU * 60.0 * tb).sin()) as f32
            } else {
                0.0
            }
        })
        .collect();
    let mut noise = Noise(3);
    let white = (0..n).map(|_| 0.2 * noise.next()).collect();
    vec![
        ("thumps", thumps),
        ("50 Hz tone", tone),
        ("bursts", bursts),
        ("white noise", white),
    ]
}

/// A signal with content in every band, interleaved. Each channel has its
/// own chord of 24 harmonics and its own noise, louder at low frequencies,
/// with a slow change of level.
pub fn broadband(frames: usize, channels: usize) -> Vec<f32> {
    let n = frames * FRAME;
    let mut out = vec![0f32; n * channels];
    for c in 0..channels {
        let mut noise = Noise(100 + c as u32);
        let (mut lp1, mut lp2) = (0f64, 0f64);
        let base = [220.0, 277.18][c % 2];
        for i in 0..n {
            let t = i as f64 / FS as f64;
            let mut v = 0.0;
            for h in 1..=24 {
                let f = base * h as f64;
                if f < 18_000.0 {
                    v += (std::f64::consts::TAU * f * t).sin() / (h as f64).powf(0.9);
                }
            }
            let w = f64::from(noise.next());
            lp1 += 0.25 * (w - lp1);
            lp2 += 0.03 * (w - lp2);
            let level = 0.7 + 0.3 * (std::f64::consts::TAU * 1.5 * t).sin();
            out[i * channels + c] = (level * (0.08 * v + 0.05 * w + 0.1 * lp1 + 0.2 * lp2)) as f32;
        }
    }
    out
}

/// `fixtures/answer_16k.wav`: 18 s of mono speech, 16-bit, 16 kHz.
pub fn speech_16k() -> Vec<f32> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/answer_16k.wav");
    let bytes = std::fs::read(path).expect("fixtures/answer_16k.wav");
    assert_eq!(&bytes[0..4], b"RIFF");
    let mut pos = 12;
    while pos + 8 <= bytes.len() {
        let len = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into().unwrap()) as usize;
        if &bytes[pos..pos + 4] == b"data" {
            let data = &bytes[pos + 8..(pos + 8 + len).min(bytes.len())];
            return data
                .chunks_exact(2)
                .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0)
                .collect();
        }
        pos += 8 + len + (len & 1);
    }
    panic!("no data chunk in answer_16k.wav");
}

/// `seconds` of the speech fixture at `fs` Hz (16000 or 48000), mono,
/// starting 1 s into the recording. The 48 kHz version is the fixture with
/// two zeros after each sample and a low-pass filter at 8 kHz.
pub fn speech(fs: usize, seconds: usize) -> Vec<f32> {
    let src = speech_16k();
    let src = &src[16_000..16_000 * (1 + seconds)];
    match fs {
        16_000 => src.to_vec(),
        48_000 => {
            // Windowed sinc, 8 input samples to each side.
            const HALF: isize = 24;
            let taps: Vec<f64> = (-HALF..=HALF)
                .map(|k| {
                    let x = k as f64 / 3.0;
                    let sinc = if k == 0 {
                        1.0
                    } else {
                        (std::f64::consts::PI * x).sin() / (std::f64::consts::PI * x)
                    };
                    let w = 0.5 + 0.5 * (std::f64::consts::PI * k as f64 / (HALF + 1) as f64).cos();
                    sinc * w
                })
                .collect();
            (0..src.len() as isize * 3)
                .map(|i| {
                    let mut acc = 0.0f64;
                    for (t, &h) in taps.iter().enumerate() {
                        let j = i + t as isize - HALF;
                        if j >= 0 && j % 3 == 0 && ((j / 3) as usize) < src.len() {
                            acc += h * f64::from(src[(j / 3) as usize]);
                        }
                    }
                    acc as f32
                })
                .collect()
        }
        other => panic!("no speech signal at {other} Hz"),
    }
}

/// A music-like signal, mono, `seconds` long at `fs` Hz: a new three-note
/// chord every 250 ms, each note with decaying harmonics and a short
/// attack, over a little noise. `voice` picks the notes, so two voices give
/// two related but different channels.
pub fn music(fs: usize, seconds: usize, voice: usize) -> Vec<f32> {
    let notes = [220.0, 277.18, 329.63, 392.0, 440.0, 523.25, 659.25, 783.99];
    let note_len = fs / 4;
    let mut noise = Noise(11 + voice as u32);
    (0..fs * seconds)
        .map(|i| {
            let k = i / note_len;
            let t = (i % note_len) as f64 / fs as f64;
            let env = (t / 0.004).min(1.0) * (-3.0 * t).exp();
            let mut v = 0.0;
            for j in 0..3 {
                let f = notes[(k * 3 + j * 2 + voice * 3) % notes.len()];
                for h in 1..=8 {
                    let fh = f * h as f64;
                    if fh < 0.45 * fs as f64 {
                        v += (std::f64::consts::TAU * fh * i as f64 / fs as f64).sin() / h as f64;
                    }
                }
            }
            (0.12 * env * v + 0.003 * f64::from(noise.next())) as f32
        })
        .collect()
}

/// Two mono signals as the left and right channel of an interleaved stereo
/// signal. Each channel also gets `leak` of the other one.
pub fn stereo_of(left: &[f32], right: &[f32], leak: f32) -> Vec<f32> {
    left.iter()
        .zip(right)
        .flat_map(|(&l, &r)| [l + leak * r, r + leak * l])
        .collect()
}
