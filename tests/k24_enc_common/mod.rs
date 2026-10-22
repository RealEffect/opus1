//! Shared helpers for the 24 kHz encoder tests (`tests/k24_enc_*.rs`).
//!
//! Every measurement puts opus1's encoder next to C libopus's encoder under
//! the same settings. Both streams are decoded by C libopus, so a difference
//! in the numbers comes from the encoder alone.
#![allow(dead_code)]

use audiopus_sys as sys;
use opus1::{Application, Bandwidth, Bitrate, Channels, ForcedMode, OpusEncoder, SampleRate};
use std::os::raw::c_int;

/// The input rate under test.
pub const FS: usize = 24_000;

// libopus private control request (opus_private.h): force the coding mode.
pub const OPUS_SET_FORCE_MODE_REQUEST: c_int = 11002;
pub const MODE_SILK_ONLY: c_int = 1000;
pub const MODE_HYBRID: c_int = 1001;
pub const MODE_CELT_ONLY: c_int = 1002;

// ---------------------------------------------------------------------------
// Signals
// ---------------------------------------------------------------------------

/// `fixtures/answer_16k.wav`: 18 s of mono speech, 16-bit, 16 kHz.
pub fn load_speech_16k() -> Vec<f32> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/answer_16k.wav");
    let bytes = std::fs::read(path).expect("fixtures/answer_16k.wav");
    assert_eq!(&bytes[0..4], b"RIFF");
    assert_eq!(&bytes[8..12], b"WAVE");
    let mut pos = 12;
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let len = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into().unwrap()) as usize;
        if id == b"data" {
            let data = &bytes[pos + 8..(pos + 8 + len).min(bytes.len())];
            return data
                .as_chunks::<2>().0.iter()
                .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0)
                .collect();
        }
        pos += 8 + len + (len & 1);
    }
    panic!("no data chunk in answer_16k.wav");
}

/// Modified Bessel function of the first kind, order 0 (for the Kaiser window).
fn bessel_i0(x: f64) -> f64 {
    let (mut sum, mut term, mut k) = (1.0, 1.0, 1.0);
    loop {
        term *= (x / (2.0 * k)) * (x / (2.0 * k));
        sum += term;
        if term < 1e-12 * sum {
            return sum;
        }
        k += 1.0;
    }
}

/// Rational resampler by `up / down`: a Kaiser-windowed sinc low-pass
/// (stop-band about -90 dB, 48 zero crossings per side), evaluated in f64.
/// The output is time-aligned with the input: output sample `n` sits at input
/// time `n * down / up`.
pub fn resample(x: &[f32], up: usize, down: usize) -> Vec<f32> {
    const HALF_ZEROS: usize = 48;
    const BETA: f64 = 9.0;
    // Cut-off as a fraction of the input Nyquist frequency, leaving a small
    // transition band below the lower of the two Nyquist frequencies.
    let cutoff = if up >= down {
        0.96
    } else {
        0.96 * up as f64 / down as f64
    };
    let half_width = HALF_ZEROS as f64 / cutoff;
    let i0_beta = bessel_i0(BETA);
    // The fractional position of an output sample repeats every `up` output
    // samples, so one coefficient set per phase is enough.
    let half = half_width.ceil() as isize;
    let taps = (2 * half + 1) as usize;
    let mut table = vec![0.0f64; up * taps];
    for phase in 0..up {
        let frac = phase as f64 / up as f64;
        for j in 0..taps {
            // Input sample `base + j - half` against output time `base + frac`.
            let d = frac - (j as isize - half) as f64;
            let r = d / half_width;
            if r.abs() >= 1.0 {
                continue;
            }
            let sinc = if d.abs() < 1e-12 {
                1.0
            } else {
                let a = std::f64::consts::PI * d * cutoff;
                a.sin() / a
            };
            let w = bessel_i0(BETA * (1.0 - r * r).sqrt()) / i0_beta;
            table[phase * taps + j] = cutoff * sinc * w;
        }
    }
    let out_len = x.len() * up / down;
    let mut y = Vec::with_capacity(out_len);
    for n in 0..out_len {
        let base = (n * down / up) as isize;
        let coeffs = &table[((n * down) % up) * taps..][..taps];
        let mut acc = 0.0f64;
        for (j, &c) in coeffs.iter().enumerate() {
            let k = base + j as isize - half;
            if k >= 0 && (k as usize) < x.len() {
                acc += f64::from(x[k as usize]) * c;
            }
        }
        y.push(acc as f32);
    }
    y
}

/// The speech fixture resampled 16 kHz -> 24 kHz. It has no content above
/// 8 kHz, like any narrower recording converted to 24 kHz.
pub fn speech_24k() -> Vec<f32> {
    resample(&load_speech_16k(), 3, 2)
}

/// The speech fixture's 16 kHz samples played at 24 kHz (1.5x faster). The
/// pitch and formants move up by 1.5x, so it is a speech-like signal with
/// real content up to 12 kHz: the stand-in for 24 kHz speech synthesis.
pub fn speech_fast_24k() -> Vec<f32> {
    load_speech_16k()
}

/// Linear chirp from `f0` to `f1` Hz over `n` samples.
pub fn chirp(n: usize, f0: f64, f1: f64, amp: f64) -> Vec<f32> {
    let dur = n as f64 / FS as f64;
    (0..n)
        .map(|i| {
            let t = i as f64 / FS as f64;
            let phase = std::f64::consts::TAU * (f0 * t + 0.5 * (f1 - f0) * t * t / dur);
            (amp * phase.sin()) as f32
        })
        .collect()
}

/// A music-like signal: a new three-note chord every 250 ms, each note with
/// six decaying harmonics and a short attack.
pub fn music(n: usize) -> Vec<f32> {
    let notes = [220.0, 277.18, 329.63, 392.0, 440.0, 523.25, 659.25, 783.99];
    let mut x = vec![0.0f32; n];
    let note_len = FS / 4;
    for (i, s) in x.iter_mut().enumerate() {
        let k = i / note_len;
        let t = (i % note_len) as f64 / FS as f64;
        let env = (t / 0.004).min(1.0) * (-3.0 * t).exp();
        let mut v = 0.0;
        for j in 0..3 {
            let f = notes[(k * 3 + j * 2) % notes.len()];
            for h in 1..=6 {
                let fh = f * h as f64;
                if fh < 11_500.0 {
                    v += (std::f64::consts::TAU * fh * i as f64 / FS as f64).sin() / h as f64;
                }
            }
        }
        *s = (0.12 * env * v) as f32;
    }
    x
}

/// Silence with one full-scale sample at `at`.
pub fn impulse(n: usize, at: usize, amp: f32) -> Vec<f32> {
    let mut x = vec![0.0f32; n];
    x[at] = amp;
    x
}

/// Mono -> interleaved stereo with different gains and a small delay on the
/// right channel, so the two channels are related but not identical.
pub fn to_stereo(mono: &[f32]) -> Vec<f32> {
    let mut out = Vec::with_capacity(mono.len() * 2);
    for i in 0..mono.len() {
        let l = mono[i];
        let r = if i >= 12 { 0.6 * mono[i - 12] } else { 0.0 };
        out.push(l);
        out.push(r);
    }
    out
}

/// Two different sources: one per channel.
pub fn two_sources(a: &[f32], b: &[f32]) -> Vec<f32> {
    let n = a.len().min(b.len());
    let mut out = Vec::with_capacity(2 * n);
    for i in 0..n {
        out.push(a[i]);
        out.push(b[i]);
    }
    out
}

// ---------------------------------------------------------------------------
// Encoder settings
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum App {
    Voip,
    Audio,
    LowDelay,
}

impl App {
    pub fn name(self) -> &'static str {
        match self {
            App::Voip => "voip",
            App::Audio => "audio",
            App::LowDelay => "rld",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Cfg {
    pub app: App,
    pub channels: usize,
    pub bitrate: i32,
    /// Samples per channel per packet at 24 kHz (480 = 20 ms).
    pub frame: usize,
    pub cbr: bool,
    pub complexity: i32,
    pub fec: bool,
    pub loss: i32,
    /// Feed `encode_i16` / `opus_encode` instead of the float entry points.
    pub i16_input: bool,
    /// libopus only: OPUS_SET_FORCE_MODE (`MODE_*`).
    pub c_force_mode: Option<c_int>,
    /// libopus only: OPUS_SET_BANDWIDTH (`OPUS_BANDWIDTH_*`).
    pub c_bandwidth: Option<c_int>,
    /// libopus only: OPUS_SET_MAX_BANDWIDTH.
    pub c_max_bandwidth: Option<c_int>,
    /// libopus only: OPUS_SET_SIGNAL (`OPUS_SIGNAL_*`).
    pub c_signal: Option<c_int>,
    /// libopus only: complexity, when it must differ from `complexity`
    /// (complexity 6 and below turns libopus's tonality analysis off).
    pub c_complexity: Option<i32>,
    /// opus1 only: `set_force_mode`. Since opus1 runs libopus's speech and
    /// music analysis, it takes the same decisions as libopus without
    /// forcing, and a test that forces libopus forces opus1 the same way.
    pub rs_force_mode: Option<ForcedMode>,
    /// opus1 only: `set_bandwidth`.
    pub rs_bandwidth: Option<Bandwidth>,
}

impl Cfg {
    pub fn new(app: App, bitrate: i32) -> Self {
        Cfg {
            app,
            channels: 1,
            bitrate,
            frame: 480,
            cbr: false,
            complexity: 9,
            fec: false,
            loss: 0,
            i16_input: false,
            c_force_mode: None,
            c_bandwidth: None,
            c_max_bandwidth: None,
            c_signal: None,
            c_complexity: None,
            rs_force_mode: None,
            rs_bandwidth: None,
        }
    }
}

pub fn to_i16(pcm: &[f32]) -> Vec<i16> {
    pcm.iter()
        .map(|&x| (x * 32768.0).round().clamp(-32768.0, 32767.0) as i16)
        .collect()
}

/// Encode with opus1. `pcm` is interleaved and is cut into whole packets.
pub fn enc_rs(cfg: &Cfg, pcm: &[f32]) -> Result<Vec<Vec<u8>>, String> {
    let mut enc = rs_encoder(cfg);
    enc_rs_with(&mut enc, cfg, pcm)
}

pub fn rs_encoder(cfg: &Cfg) -> OpusEncoder {
    let app = match cfg.app {
        App::Voip => Application::Voip,
        App::Audio => Application::Audio,
        App::LowDelay => Application::RestrictedLowDelay,
    };
    let mut enc = OpusEncoder::new(
        SampleRate::try_from(FS as i32).unwrap(),
        Channels::try_from(cfg.channels).unwrap(),
        app,
    )
    .unwrap();
    enc.set_bitrate(Bitrate::BitsPerSecond(cfg.bitrate)).unwrap();
    enc.set_vbr(!cfg.cbr);
    enc.set_complexity(cfg.complexity).unwrap();
    enc.set_inband_fec(cfg.fec);
    enc.set_packet_loss_perc(cfg.loss).unwrap();
    enc.set_force_mode(cfg.rs_force_mode);
    if let Some(bandwidth) = cfg.rs_bandwidth {
        enc.set_bandwidth(bandwidth);
    }
    enc
}

/// Encode `pcm` with an existing opus1 encoder (for mid-stream changes).
pub fn enc_rs_with(
    enc: &mut OpusEncoder,
    cfg: &Cfg,
    pcm: &[f32],
) -> Result<Vec<Vec<u8>>, String> {
    let step = cfg.frame * cfg.channels;
    let mut out = Vec::new();
    let mut pkt = vec![0u8; 4000];
    let pcm_i16 = if cfg.i16_input { to_i16(pcm) } else { Vec::new() };
    for (k, chunk) in pcm.chunks_exact(step).enumerate() {
        let n = if cfg.i16_input {
            enc.encode_i16(&pcm_i16[k * step..(k + 1) * step], cfg.frame, &mut pkt)
        } else {
            enc.encode(chunk, cfg.frame, &mut pkt)
        }
        .map_err(|e| format!("opus1 encode failed at packet {k}: {e}"))?;
        out.push(pkt[..n].to_vec());
    }
    Ok(out)
}

/// Owning handle of a C libopus encoder.
pub struct CEncoder {
    st: *mut sys::OpusEncoder,
}

impl CEncoder {
    pub fn new(cfg: &Cfg) -> Self {
        let app = match cfg.app {
            App::Voip => sys::OPUS_APPLICATION_VOIP,
            App::Audio => sys::OPUS_APPLICATION_AUDIO,
            App::LowDelay => sys::OPUS_APPLICATION_RESTRICTED_LOWDELAY,
        };
        let mut err: c_int = 0;
        let st =
            unsafe { sys::opus_encoder_create(FS as i32, cfg.channels as c_int, app, &mut err) };
        assert_eq!(err, 0, "opus_encoder_create");
        let enc = CEncoder { st };
        enc.ctl(sys::OPUS_SET_BITRATE_REQUEST, cfg.bitrate);
        enc.ctl(sys::OPUS_SET_VBR_REQUEST, if cfg.cbr { 0 } else { 1 });
        enc.ctl(
            sys::OPUS_SET_COMPLEXITY_REQUEST,
            cfg.c_complexity.unwrap_or(cfg.complexity),
        );
        enc.ctl(sys::OPUS_SET_INBAND_FEC_REQUEST, if cfg.fec { 1 } else { 0 });
        enc.ctl(sys::OPUS_SET_PACKET_LOSS_PERC_REQUEST, cfg.loss);
        if let Some(m) = cfg.c_force_mode {
            enc.ctl(OPUS_SET_FORCE_MODE_REQUEST, m);
        }
        if let Some(b) = cfg.c_bandwidth {
            enc.ctl(sys::OPUS_SET_BANDWIDTH_REQUEST, b);
        }
        if let Some(b) = cfg.c_max_bandwidth {
            enc.ctl(sys::OPUS_SET_MAX_BANDWIDTH_REQUEST, b);
        }
        if let Some(s) = cfg.c_signal {
            enc.ctl(sys::OPUS_SET_SIGNAL_REQUEST, s);
        }
        enc
    }

    pub fn ctl(&self, request: c_int, value: c_int) {
        let ret = unsafe { sys::opus_encoder_ctl(self.st, request, value) };
        assert_eq!(ret, 0, "opus_encoder_ctl({request}, {value})");
    }

    pub fn lookahead(&self) -> i32 {
        let mut v: c_int = 0;
        let ret = unsafe {
            sys::opus_encoder_ctl(self.st, sys::OPUS_GET_LOOKAHEAD_REQUEST, &mut v as *mut c_int)
        };
        assert_eq!(ret, 0);
        v
    }

    pub fn encode(&mut self, cfg: &Cfg, pcm: &[f32]) -> Vec<Vec<u8>> {
        let step = cfg.frame * cfg.channels;
        let mut out = Vec::new();
        let mut pkt = vec![0u8; 4000];
        let pcm_i16 = if cfg.i16_input { to_i16(pcm) } else { Vec::new() };
        for (k, chunk) in pcm.chunks_exact(step).enumerate() {
            let n = unsafe {
                if cfg.i16_input {
                    sys::opus_encode(
                        self.st,
                        pcm_i16[k * step..].as_ptr(),
                        cfg.frame as c_int,
                        pkt.as_mut_ptr(),
                        pkt.len() as i32,
                    )
                } else {
                    sys::opus_encode_float(
                        self.st,
                        chunk.as_ptr(),
                        cfg.frame as c_int,
                        pkt.as_mut_ptr(),
                        pkt.len() as i32,
                    )
                }
            };
            assert!(n > 0, "libopus encode failed at packet {k}: {n}");
            out.push(pkt[..n as usize].to_vec());
        }
        out
    }
}

impl Drop for CEncoder {
    fn drop(&mut self) {
        unsafe { sys::opus_encoder_destroy(self.st) }
    }
}

/// Encode with C libopus.
pub fn enc_c(cfg: &Cfg, pcm: &[f32]) -> Vec<Vec<u8>> {
    CEncoder::new(cfg).encode(cfg, pcm)
}

/// Decode a stream with C libopus at `rate` Hz. `lost(k)` marks packet `k` as
/// lost; a lost packet is concealed, or recovered from the next packet's
/// in-band FEC when `use_fec` is set.
pub fn dec_c_lossy(
    packets: &[Vec<u8>],
    rate: usize,
    channels: usize,
    frame_at_24k: usize,
    lost: &dyn Fn(usize) -> bool,
    use_fec: bool,
) -> Result<Vec<f32>, String> {
    let mut err: c_int = 0;
    let st = unsafe { sys::opus_decoder_create(rate as i32, channels as c_int, &mut err) };
    assert_eq!(err, 0, "opus_decoder_create");
    let frame = frame_at_24k * rate / FS;
    let mut pcm = vec![0f32; 5760 * channels];
    let mut out = Vec::with_capacity(packets.len() * frame * channels);
    let mut result = Ok(());
    for (k, p) in packets.iter().enumerate() {
        let n = unsafe {
            if lost(k) {
                if use_fec && k + 1 < packets.len() && !lost(k + 1) {
                    let next = &packets[k + 1];
                    sys::opus_decode_float(
                        st,
                        next.as_ptr(),
                        next.len() as i32,
                        pcm.as_mut_ptr(),
                        frame as c_int,
                        1,
                    )
                } else {
                    sys::opus_decode_float(
                        st,
                        std::ptr::null(),
                        0,
                        pcm.as_mut_ptr(),
                        frame as c_int,
                        0,
                    )
                }
            } else {
                sys::opus_decode_float(
                    st,
                    p.as_ptr(),
                    p.len() as i32,
                    pcm.as_mut_ptr(),
                    5760,
                    0,
                )
            }
        };
        if n < 0 {
            result = Err(format!("libopus rejected packet {k} ({} bytes): error {n}", p.len()));
            break;
        }
        if n as usize != frame {
            result = Err(format!(
                "packet {k} decoded to {n} samples, expected {frame} (TOC {:#04x})",
                p[0]
            ));
            break;
        }
        out.extend_from_slice(&pcm[..n as usize * channels]);
    }
    unsafe { sys::opus_decoder_destroy(st) };
    result.map(|_| out)
}

/// Decode a stream with C libopus at `rate` Hz, no loss.
pub fn dec_c(
    packets: &[Vec<u8>],
    rate: usize,
    channels: usize,
    frame_at_24k: usize,
) -> Result<Vec<f32>, String> {
    dec_c_lossy(packets, rate, channels, frame_at_24k, &|_| false, false)
}

/// Decode a stream whose packets may differ in duration with one C libopus
/// decoder at 24 kHz, mono.
pub fn dec_c_any(packets: &[Vec<u8>]) -> Vec<f32> {
    let mut err: c_int = 0;
    let st = unsafe { sys::opus_decoder_create(FS as i32, 1, &mut err) };
    assert_eq!(err, 0, "opus_decoder_create");
    let mut pcm = vec![0f32; 5760];
    let mut out = Vec::new();
    for (k, p) in packets.iter().enumerate() {
        let n = unsafe {
            sys::opus_decode_float(st, p.as_ptr(), p.len() as i32, pcm.as_mut_ptr(), 5760, 0)
        };
        assert!(n > 0, "libopus rejected packet {k} ({} bytes): error {n}", p.len());
        out.extend_from_slice(&pcm[..n as usize]);
    }
    unsafe { sys::opus_decoder_destroy(st) };
    out
}

/// How well the frames of lost packets come back, in dB against the loss-free
/// decode of the same stream, over the lost frames only. Every 7th packet is
/// lost. Returns `(with in-band FEC, with concealment only)`.
pub fn fec_recovery_db(packets: &[Vec<u8>], channels: usize, frame: usize) -> (f64, f64) {
    let lost = |k: usize| k % 7 == 3;
    let clean = dec_c(packets, FS, channels, frame).unwrap();
    let fec = dec_c_lossy(packets, FS, channels, frame, &lost, true).unwrap();
    let plc = dec_c_lossy(packets, FS, channels, frame, &lost, false).unwrap();
    let step = frame * channels;
    let db = |x: &[f32]| -> f64 {
        let (mut s, mut e) = (0f64, 0f64);
        // The last packet has no successor to carry its FEC.
        for k in (0..packets.len() - 1).filter(|&k| lost(k)) {
            for i in k * step..(k + 1) * step {
                s += f64::from(clean[i]).powi(2);
                e += f64::from(clean[i] - x[i]).powi(2);
            }
        }
        10.0 * (s / e.max(1e-30)).log10()
    };
    (db(&fec), db(&plc))
}

/// One channel of an interleaved buffer.
pub fn channel(pcm: &[f32], channels: usize, c: usize) -> Vec<f32> {
    pcm.iter().skip(c).step_by(channels).copied().collect()
}

// ---------------------------------------------------------------------------
// Packet inspection
// ---------------------------------------------------------------------------

/// What the TOC byte (RFC 6716 section 3.1) says about a packet.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Toc {
    /// "SILK", "HYB" or "CELT".
    pub mode: &'static str,
    /// "NB", "MB", "WB", "SWB" or "FB".
    pub bw: &'static str,
    /// Duration of one coded frame in units of 0.1 ms (25 = 2.5 ms).
    pub frame_tenth_ms: u32,
    pub stereo: bool,
    /// Frames in the packet.
    pub frames: u8,
}

pub fn toc(p: &[u8]) -> Toc {
    let t = p[0];
    let config = t >> 3;
    let (mode, bw, dur) = match config {
        0..=11 => (
            "SILK",
            ["NB", "MB", "WB"][(config / 4) as usize],
            [100, 200, 400, 600][(config % 4) as usize],
        ),
        12..=15 => (
            "HYB",
            ["SWB", "FB"][((config - 12) / 2) as usize],
            [100, 200][((config - 12) % 2) as usize],
        ),
        _ => (
            "CELT",
            ["NB", "WB", "SWB", "FB"][((config - 16) / 4) as usize],
            [25, 50, 100, 200][((config - 16) % 4) as usize],
        ),
    };
    let frames = match t & 3 {
        0 => 1,
        1 | 2 => 2,
        _ => p.get(1).map_or(0, |b| b & 0x3f),
    };
    Toc {
        mode,
        bw,
        frame_tenth_ms: dur,
        stereo: t & 4 != 0,
        frames,
    }
}

/// The distinct TOCs of a stream with their packet counts, most frequent
/// first, as text: `HYB/SWB/20ms x298, SILK/WB/20ms x2`.
pub fn toc_summary(packets: &[Vec<u8>]) -> String {
    let mut counts: std::collections::BTreeMap<Toc, usize> = Default::default();
    for p in packets {
        *counts.entry(toc(p)).or_default() += 1;
    }
    let mut v: Vec<_> = counts.into_iter().collect();
    v.sort_by_key(|a| std::cmp::Reverse(a.1));
    v.iter()
        .map(|(t, n)| {
            let frames = if t.frames == 1 {
                String::new()
            } else {
                format!("*{}", t.frames)
            };
            format!(
                "{}/{}/{}ms{}{} x{}",
                t.mode,
                t.bw,
                t.frame_tenth_ms as f32 / 10.0,
                frames,
                if t.stereo { "/st" } else { "" },
                n
            )
        })
        .collect::<Vec<_>>()
        .join(", ")
}

/// The most frequent (mode, bandwidth) of a stream.
pub fn main_mode(packets: &[Vec<u8>]) -> (&'static str, &'static str) {
    let mut counts: std::collections::BTreeMap<(&str, &str), usize> = Default::default();
    for p in packets {
        let t = toc(p);
        *counts.entry((t.mode, t.bw)).or_default() += 1;
    }
    counts.into_iter().max_by_key(|e| e.1).unwrap().0
}

/// Average bitrate of a stream in bits per second.
pub fn mean_bitrate(packets: &[Vec<u8>], frame_at_24k: usize) -> f64 {
    let bytes: usize = packets.iter().map(|p| p.len()).sum();
    bytes as f64 * 8.0 * FS as f64 / (packets.len() * frame_at_24k) as f64
}

pub fn size_range(packets: &[Vec<u8>]) -> (usize, usize) {
    let min = packets.iter().map(|p| p.len()).min().unwrap_or(0);
    let max = packets.iter().map(|p| p.len()).max().unwrap_or(0);
    (min, max)
}

// ---------------------------------------------------------------------------
// Measurements
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
pub struct Metrics {
    /// Delay of the decoded signal against the reference, in samples of the
    /// decoder rate, at the largest cross-correlation.
    pub lag: isize,
    /// Signal-to-noise ratio in dB at that lag, over the whole signal.
    pub snr: f64,
    /// Correlation over the whole signal at that lag.
    pub corr: f64,
    /// Lowest correlation of any 100 ms window that carries signal.
    pub worst_corr: f64,
    /// Lowest SNR in dB of any 20 ms window that carries signal.
    pub worst_snr20: f64,
    /// Decoded level over reference level, in dB.
    pub level_db: f64,
}

fn energy(x: &[f32]) -> f64 {
    x.iter().map(|&v| f64::from(v) * f64::from(v)).sum()
}

fn dot(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| f64::from(x) * f64::from(y))
        .sum()
}

/// The lag in `0..=max_lag` at which `out[lag..]` best matches `reference`.
pub fn best_lag(reference: &[f32], out: &[f32], max_lag: usize) -> usize {
    let n = reference.len().min(out.len().saturating_sub(max_lag));
    let (mut best, mut best_c) = (0usize, f64::NEG_INFINITY);
    for lag in 0..=max_lag {
        let c = dot(&reference[..n], &out[lag..lag + n]);
        if c > best_c {
            best_c = c;
            best = lag;
        }
    }
    best
}

/// Compare a decoded signal with its reference. `rate` is the rate of both.
/// `skip` reference samples are left out at the start (encoder start-up).
pub fn measure(reference: &[f32], out: &[f32], rate: usize, skip: usize) -> Metrics {
    let max_lag = rate / 25; // 40 ms
    let lag = best_lag(reference, out, max_lag);
    measure_at(reference, out, rate, skip, lag)
}

pub fn measure_at(reference: &[f32], out: &[f32], rate: usize, skip: usize, lag: usize) -> Metrics {
    let n = reference.len().min(out.len() - lag);
    let r = &reference[skip.min(n)..n];
    let o = &out[lag + skip.min(n)..lag + n];
    let (er, eo) = (energy(r), energy(o));
    let err: f64 = r
        .iter()
        .zip(o)
        .map(|(&a, &b)| {
            let d = f64::from(a) - f64::from(b);
            d * d
        })
        .sum();
    let snr = 10.0 * (er / err.max(1e-30)).log10();
    let corr = dot(r, o) / (er * eo).sqrt().max(1e-30);
    // A window "carries signal" when its level is within 30 dB of the mean.
    let floor = er / r.len().max(1) as f64 * 1e-3;
    let mut worst_corr = 1.0f64;
    let w100 = rate / 10;
    for (a, b) in r.chunks_exact(w100).zip(o.chunks_exact(w100)) {
        let (ea, eb) = (energy(a), energy(b));
        if ea / w100 as f64 > floor {
            worst_corr = worst_corr.min(dot(a, b) / (ea * eb).sqrt().max(1e-30));
        }
    }
    let mut worst_snr20 = f64::INFINITY;
    let w20 = rate / 50;
    for (a, b) in r.chunks_exact(w20).zip(o.chunks_exact(w20)) {
        let ea = energy(a);
        if ea / w20 as f64 > floor {
            let e: f64 = a
                .iter()
                .zip(b)
                .map(|(&x, &y)| {
                    let d = f64::from(x) - f64::from(y);
                    d * d
                })
                .sum();
            worst_snr20 = worst_snr20.min(10.0 * (ea / e.max(1e-30)).log10());
        }
    }
    Metrics {
        lag: lag as isize,
        snr,
        corr,
        worst_corr,
        worst_snr20,
        level_db: 10.0 * (eo / er.max(1e-30)).log10(),
    }
}

/// In-place radix-2 complex FFT.
fn fft(re: &mut [f64], im: &mut [f64]) {
    let n = re.len();
    let mut j = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        let ang = -std::f64::consts::TAU / len as f64;
        for start in (0..n).step_by(len) {
            for k in 0..len / 2 {
                let (wr, wi) = ((ang * k as f64).cos(), (ang * k as f64).sin());
                let (a, b) = (start + k, start + k + len / 2);
                let (tr, ti) = (re[b] * wr - im[b] * wi, re[b] * wi + im[b] * wr);
                re[b] = re[a] - tr;
                im[b] = im[a] - ti;
                re[a] += tr;
                im[a] += ti;
            }
        }
        len <<= 1;
    }
}

/// Energy of `x` (sampled at `rate`) in each band `[lo, hi)` Hz, averaged over
/// Hann-windowed 1024-point frames with half overlap.
pub fn band_energies(x: &[f32], rate: usize, bands: &[(f64, f64)]) -> Vec<f64> {
    const N: usize = 1024;
    let mut acc = vec![0.0f64; bands.len()];
    let mut frames = 0usize;
    let mut start = 0;
    while start + N <= x.len() {
        let mut re: Vec<f64> = (0..N)
            .map(|i| {
                let w = 0.5 - 0.5 * (std::f64::consts::TAU * i as f64 / N as f64).cos();
                f64::from(x[start + i]) * w
            })
            .collect();
        let mut im = vec![0.0f64; N];
        fft(&mut re, &mut im);
        for (b, &(lo, hi)) in bands.iter().enumerate() {
            let k0 = (lo * N as f64 / rate as f64).ceil() as usize;
            let k1 = ((hi * N as f64 / rate as f64).ceil() as usize).min(N / 2);
            for k in k0..k1 {
                acc[b] += re[k] * re[k] + im[k] * im[k];
            }
        }
        frames += 1;
        start += N / 2;
    }
    for a in &mut acc {
        *a /= frames.max(1) as f64;
    }
    acc
}

/// Decoded energy over input energy in a band, in dB. A coded band reads
/// near 0 dB; a band the encoder dropped reads far below.
pub fn band_gain_db(reference: &[f32], out: &[f32], rate: usize, lo: f64, hi: f64) -> f64 {
    let r = band_energies(reference, rate, &[(lo, hi)])[0];
    let o = band_energies(out, rate, &[(lo, hi)])[0];
    10.0 * (o.max(1e-30) / r.max(1e-30)).log10()
}
