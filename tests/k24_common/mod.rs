//! Shared helpers for the `k24_*` tests.
//!
//! These tests check opus1's decoder against C libopus when the decoder
//! output rate is 24 kHz (and, for contrast, 8 / 12 / 16 / 48 kHz). The
//! helpers here do three things:
//!
//! 1. Build test signals: real speech resampled with a windowed-sinc filter,
//!    chirps, tones, an impulse, silence and full-scale input.
//! 2. Wrap the raw libopus C API (`audiopus_sys`), because the tests need
//!    controls the `opus` crate does not expose: forcing the coding mode,
//!    the decoder's forward-error-correction flag, and the repacketizer.
//! 3. Compare two decoded signals sample by sample.

#![allow(dead_code)]

use audiopus_sys as sys;
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Signals
// ---------------------------------------------------------------------------

/// Read `fixtures/answer_16k.wav` (16 kHz mono 16-bit speech) as `f32`.
pub fn load_speech_16k() -> Vec<f32> {
    let bytes = std::fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/fixtures/answer_16k.wav"
    ))
    .expect("fixtures/answer_16k.wav");
    // Find the "data" chunk instead of assuming a 44-byte header.
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

/// Zeroth-order modified Bessel function of the first kind (for the Kaiser
/// window).
fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0;
    let mut term = 1.0;
    let mut k = 1.0;
    loop {
        term *= (x / (2.0 * k)) * (x / (2.0 * k));
        sum += term;
        if term < 1e-12 * sum {
            return sum;
        }
        k += 1.0;
    }
}

/// Windowed-sinc sample-rate converter (Kaiser window, 48 zero crossings on
/// each side, stop band below -100 dB). The cut-off sits at 46% of the lower
/// of the two rates, so nothing aliases.
pub fn resample(x: &[f32], from: usize, to: usize) -> Vec<f32> {
    if from == to {
        return x.to_vec();
    }
    let g = gcd(from, to);
    let (up, down) = (to / g, from / g);
    // Cut-off as a fraction of the input sampling rate.
    let fc = 0.46 * (to.min(from) as f64) / from as f64;
    let half = (48.0 * 0.5 / fc).ceil() as isize; // taps on each side, in input samples
    let beta = 10.0;
    let i0_beta = bessel_i0(beta);
    // One filter per output phase.
    let taps = (2 * half + 1) as usize;
    let mut table = vec![0f64; up * taps];
    for p in 0..up {
        let frac = p as f64 / up as f64; // position between input samples
        for (t, k) in (-half..=half).enumerate() {
            let d = k as f64 - frac; // distance in input samples
            let w_arg = d / (half as f64 + 1.0);
            let w = if w_arg.abs() >= 1.0 {
                0.0
            } else {
                bessel_i0(beta * (1.0 - w_arg * w_arg).sqrt()) / i0_beta
            };
            let s = if d == 0.0 {
                1.0
            } else {
                (2.0 * PI * fc * d).sin() / (2.0 * PI * fc * d)
            };
            table[p * taps + t] = 2.0 * fc * s * w;
        }
    }
    let n_out = x.len() * up / down;
    let mut y = Vec::with_capacity(n_out);
    for n in 0..n_out {
        let pos = n * down; // in units of 1/up input samples
        let base = (pos / up) as isize;
        let phase = pos % up;
        let h = &table[phase * taps..(phase + 1) * taps];
        let mut acc = 0f64;
        for (t, k) in (-half..=half).enumerate() {
            let idx = base + k;
            if idx >= 0 && (idx as usize) < x.len() {
                acc += x[idx as usize] as f64 * h[t];
            }
        }
        y.push(acc as f32);
    }
    y
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// `seconds` of real speech at `rate` Hz, mono. The excerpt starts 0.5 s into
/// the fixture, where the talker is already speaking.
pub fn speech(rate: usize, seconds: f64) -> Vec<f32> {
    let src = load_speech_16k();
    let start = 8000;
    let len16 = (seconds * 16000.0) as usize;
    resample(&src[start..start + len16], 16000, rate)
}

/// Speech with energy up to the Nyquist frequency of `rate`.
///
/// The fixture is 16 kHz speech, so `speech()` has nothing above 8 kHz and
/// cannot show a fault in the 8-12 kHz band that a 24 kHz stream carries.
/// This adds noise that follows the speech waveform's envelope (the speech
/// multiplied by white noise, 12 dB down), which fills the whole band the
/// way breath and fricative noise do in real wideband speech.
pub fn speech_fullband(rate: usize, seconds: f64) -> Vec<f32> {
    let mut seed = 0x1234_5678u32;
    speech(rate, seconds)
        .iter()
        .map(|&s| {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            let white = (seed >> 8) as f32 / (1u32 << 23) as f32 - 1.0;
            s + 0.25 * s * white
        })
        .collect()
}

/// Linear chirp from 100 Hz up to 45% of `rate`.
pub fn chirp(rate: usize, n: usize) -> Vec<f32> {
    let f0 = 100.0;
    let f1 = 0.45 * rate as f64;
    let dur = n as f64 / rate as f64;
    (0..n)
        .map(|i| {
            let t = i as f64 / rate as f64;
            (0.4 * (2.0 * PI * (f0 * t + 0.5 * (f1 - f0) * t * t / dur)).sin()) as f32
        })
        .collect()
}

/// A music-like signal: a few harmonic notes that change every 250 ms.
pub fn multitone(rate: usize, n: usize) -> Vec<f32> {
    let notes = [220.0, 277.18, 329.63, 440.0, 554.37, 659.25];
    (0..n)
        .map(|i| {
            let t = i as f64 / rate as f64;
            let step = (t / 0.25) as usize;
            let mut v = 0.0;
            for (k, base) in notes.iter().enumerate() {
                if (step + k).is_multiple_of(3) {
                    continue;
                }
                for h in 1..=4 {
                    let f = base * h as f64;
                    if f < 0.45 * rate as f64 {
                        v += (2.0 * PI * f * t + k as f64).sin() / (h as f64 * 8.0);
                    }
                }
            }
            v as f32
        })
        .collect()
}

/// Silence with one full-scale-ish sample at `pos`.
pub fn impulse(n: usize, pos: usize) -> Vec<f32> {
    let mut v = vec![0f32; n];
    v[pos] = 0.9;
    v
}

/// A 440 Hz + 1 kHz mix that peaks at exactly +-1.0.
pub fn full_scale(rate: usize, n: usize) -> Vec<f32> {
    let v: Vec<f64> = (0..n)
        .map(|i| {
            let t = i as f64 / rate as f64;
            (2.0 * PI * 440.0 * t).sin() + 0.6 * (2.0 * PI * 1000.0 * t).sin()
        })
        .collect();
    let peak = v.iter().fold(0f64, |m, x| m.max(x.abs()));
    v.iter().map(|x| (x / peak) as f32).collect()
}

/// Mono to interleaved stereo with different content per channel (the right
/// channel is delayed by 3 ms and attenuated), so a stereo coder has real
/// side information to code.
pub fn to_stereo(x: &[f32], rate: usize) -> Vec<f32> {
    let d = rate * 3 / 1000;
    let mut out = Vec::with_capacity(x.len() * 2);
    for i in 0..x.len() {
        let r = if i >= d { 0.6 * x[i - d] } else { 0.0 };
        out.push(x[i]);
        out.push(r);
    }
    out
}

// ---------------------------------------------------------------------------
// libopus C API wrappers
// ---------------------------------------------------------------------------

pub const MODE_SILK: i32 = 1000;
pub const MODE_HYBRID: i32 = 1001;
pub const MODE_CELT: i32 = 1002;
/// `OPUS_SET_FORCE_MODE` (opus_private.h): not part of the public header.
pub const OPUS_SET_FORCE_MODE_REQUEST: i32 = 11002;

pub const BW_NB: i32 = sys::OPUS_BANDWIDTH_NARROWBAND;
pub const BW_MB: i32 = sys::OPUS_BANDWIDTH_MEDIUMBAND;
pub const BW_WB: i32 = sys::OPUS_BANDWIDTH_WIDEBAND;
pub const BW_SWB: i32 = sys::OPUS_BANDWIDTH_SUPERWIDEBAND;
pub const BW_FB: i32 = sys::OPUS_BANDWIDTH_FULLBAND;

/// C libopus encoder.
pub struct CEnc {
    st: *mut sys::OpusEncoder,
    pub channels: usize,
    pub rate: usize,
}

impl CEnc {
    pub fn new(rate: usize, channels: usize, application: i32) -> Self {
        let mut err = 0;
        let st = unsafe {
            sys::opus_encoder_create(rate as i32, channels as i32, application, &mut err)
        };
        assert_eq!(err, 0, "opus_encoder_create({rate}, {channels})");
        Self { st, channels, rate }
    }

    pub fn voip(rate: usize, channels: usize) -> Self {
        Self::new(rate, channels, sys::OPUS_APPLICATION_VOIP)
    }

    pub fn audio(rate: usize, channels: usize) -> Self {
        Self::new(rate, channels, sys::OPUS_APPLICATION_AUDIO)
    }

    /// Any "set" control that takes one `opus_int32`.
    pub fn ctl(&mut self, request: i32, value: i32) -> &mut Self {
        let ret = unsafe { sys::opus_encoder_ctl(self.st, request, value) };
        assert_eq!(ret, 0, "opus_encoder_ctl({request}, {value}) = {ret}");
        self
    }

    pub fn bitrate(&mut self, bps: i32) -> &mut Self {
        self.ctl(sys::OPUS_SET_BITRATE_REQUEST, bps)
    }
    pub fn force_mode(&mut self, mode: i32) -> &mut Self {
        self.ctl(OPUS_SET_FORCE_MODE_REQUEST, mode)
    }
    pub fn bandwidth(&mut self, bw: i32) -> &mut Self {
        self.ctl(sys::OPUS_SET_BANDWIDTH_REQUEST, bw)
    }
    pub fn max_bandwidth(&mut self, bw: i32) -> &mut Self {
        self.ctl(sys::OPUS_SET_MAX_BANDWIDTH_REQUEST, bw)
    }
    pub fn complexity(&mut self, c: i32) -> &mut Self {
        self.ctl(sys::OPUS_SET_COMPLEXITY_REQUEST, c)
    }
    pub fn vbr(&mut self, on: bool) -> &mut Self {
        self.ctl(sys::OPUS_SET_VBR_REQUEST, on as i32)
    }
    pub fn fec(&mut self, on: bool, loss_perc: i32) -> &mut Self {
        self.ctl(sys::OPUS_SET_INBAND_FEC_REQUEST, on as i32);
        self.ctl(sys::OPUS_SET_PACKET_LOSS_PERC_REQUEST, loss_perc)
    }
    pub fn dtx(&mut self, on: bool) -> &mut Self {
        self.ctl(sys::OPUS_SET_DTX_REQUEST, on as i32)
    }
    pub fn force_channels(&mut self, ch: i32) -> &mut Self {
        self.ctl(sys::OPUS_SET_FORCE_CHANNELS_REQUEST, ch)
    }

    /// Encode one frame of interleaved `f32` PCM.
    pub fn encode(&mut self, pcm: &[f32]) -> Vec<u8> {
        let frame = pcm.len() / self.channels;
        let mut out = vec![0u8; 4000];
        let n = unsafe {
            sys::opus_encode_float(
                self.st,
                pcm.as_ptr(),
                frame as i32,
                out.as_mut_ptr(),
                out.len() as i32,
            )
        };
        assert!(n > 0, "opus_encode_float failed: {n}");
        out.truncate(n as usize);
        out
    }

    /// Encode a whole signal in frames of `frame` samples per channel. A
    /// trailing partial frame is dropped.
    pub fn encode_all(&mut self, pcm: &[f32], frame: usize) -> Vec<Vec<u8>> {
        pcm.chunks_exact(frame * self.channels)
            .map(|f| self.encode(f))
            .collect()
    }
}

impl Drop for CEnc {
    fn drop(&mut self) {
        unsafe { sys::opus_encoder_destroy(self.st) }
    }
}

/// C libopus decoder.
pub struct CDec {
    st: *mut sys::OpusDecoder,
    pub channels: usize,
    pub rate: usize,
}

impl CDec {
    pub fn new(rate: usize, channels: usize) -> Self {
        let mut err = 0;
        let st = unsafe { sys::opus_decoder_create(rate as i32, channels as i32, &mut err) };
        assert_eq!(err, 0, "opus_decoder_create({rate}, {channels})");
        Self { st, channels, rate }
    }

    /// Decode a packet (`None` = lost) into at most `frame_size` samples per
    /// channel. Returns the interleaved samples, or the libopus error code.
    pub fn decode(
        &mut self,
        packet: Option<&[u8]>,
        frame_size: usize,
        fec: bool,
    ) -> Result<Vec<f32>, i32> {
        let mut pcm = vec![0f32; frame_size * self.channels];
        let (ptr, len) = match packet {
            Some(p) => (p.as_ptr(), p.len() as i32),
            None => (std::ptr::null(), 0),
        };
        let n = unsafe {
            sys::opus_decode_float(
                self.st,
                ptr,
                len,
                pcm.as_mut_ptr(),
                frame_size as i32,
                fec as i32,
            )
        };
        if n < 0 {
            return Err(n);
        }
        pcm.truncate(n as usize * self.channels);
        Ok(pcm)
    }
}

impl Drop for CDec {
    fn drop(&mut self) {
        unsafe { sys::opus_decoder_destroy(self.st) }
    }
}

/// Merge several packets of the same configuration into one with the libopus
/// repacketizer. Two frames give frame-count code 1 (equal sizes) or code 2
/// (different sizes); three or more give code 3.
pub fn repacketize(packets: &[&[u8]]) -> Vec<u8> {
    unsafe {
        let rp = sys::opus_repacketizer_create();
        for p in packets {
            let ret = sys::opus_repacketizer_cat(rp, p.as_ptr(), p.len() as i32);
            assert_eq!(ret, 0, "opus_repacketizer_cat");
        }
        let mut out = vec![0u8; 8000];
        let n = sys::opus_repacketizer_out(rp, out.as_mut_ptr(), out.len() as i32);
        sys::opus_repacketizer_destroy(rp);
        assert!(n > 0, "opus_repacketizer_out = {n}");
        out.truncate(n as usize);
        out
    }
}

/// Pad a packet to `new_len` bytes with `opus_packet_pad` (the result is a
/// frame-count code 3 packet with the padding flag set).
pub fn pad_packet(packet: &[u8], new_len: usize) -> Vec<u8> {
    let mut out = packet.to_vec();
    out.resize(new_len, 0);
    let ret =
        unsafe { sys::opus_packet_pad(out.as_mut_ptr(), packet.len() as i32, new_len as i32) };
    assert_eq!(ret, 0, "opus_packet_pad");
    out
}

// ---------------------------------------------------------------------------
// Table-of-contents byte
// ---------------------------------------------------------------------------

/// What the first byte of a packet says (RFC 6716 section 3.1).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Toc {
    pub mode: &'static str,
    pub bandwidth: &'static str,
    /// Frame duration in tenths of a millisecond (25 = 2.5 ms).
    pub frame_tenth_ms: u32,
    pub stereo: bool,
    /// Frame-count code 0..3.
    pub code: u8,
}

pub fn toc(b: u8) -> Toc {
    let config = b >> 3;
    let (mode, bandwidth, frame_tenth_ms) = match config {
        0..=11 => (
            "SILK",
            ["NB", "MB", "WB"][(config / 4) as usize],
            [100, 200, 400, 600][(config % 4) as usize],
        ),
        12..=15 => (
            "Hybrid",
            ["SWB", "FB"][((config - 12) / 2) as usize],
            [100, 200][(config % 2) as usize],
        ),
        _ => (
            "CELT",
            ["NB", "WB", "SWB", "FB"][((config - 16) / 4) as usize],
            [25, 50, 100, 200][(config % 4) as usize],
        ),
    };
    Toc {
        mode,
        bandwidth,
        frame_tenth_ms,
        stereo: b & 4 != 0,
        code: b & 3,
    }
}

impl std::fmt::Display for Toc {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}-{} {}ms {} c{}",
            self.mode,
            self.bandwidth,
            self.frame_tenth_ms as f32 / 10.0,
            if self.stereo { "st" } else { "mono" },
            self.code
        )
    }
}

/// The distinct table-of-contents values in a stream, in order of first use.
pub fn toc_summary(packets: &[Vec<u8>]) -> String {
    let mut seen: Vec<(Toc, usize)> = Vec::new();
    for p in packets {
        if p.is_empty() {
            continue;
        }
        let t = toc(p[0]);
        match seen.iter_mut().find(|(s, _)| *s == t) {
            Some((_, n)) => *n += 1,
            None => seen.push((t, 1)),
        }
    }
    seen.iter()
        .map(|(t, n)| format!("{t} x{n}"))
        .collect::<Vec<_>>()
        .join(", ")
}

// ---------------------------------------------------------------------------
// Comparison
// ---------------------------------------------------------------------------

/// Result of comparing a test signal against a reference signal.
#[derive(Clone, Copy, Debug)]
pub struct Cmp {
    /// Signal-to-noise ratio over the whole signal, in dB (reference power
    /// over error power). 200 means bit-identical.
    pub snr_db: f64,
    /// The lowest SNR of any frame whose reference is not near-silent.
    pub worst_frame_snr_db: f64,
    /// Index of that frame.
    pub worst_frame: usize,
    /// Largest absolute sample difference.
    pub max_abs_diff: f64,
    /// RMS(test) / RMS(reference).
    pub level_ratio: f64,
    /// Shift (in samples per channel) of the test signal that best matches
    /// the reference. Positive: the test signal is late.
    pub lag: isize,
    /// Lowest correlation of any 100 ms window (at lag 0).
    pub worst_corr: f64,
}

pub fn snr_db(sig: f64, err: f64) -> f64 {
    if err <= 0.0 {
        200.0
    } else if sig <= 0.0 {
        -200.0
    } else {
        (10.0 * (sig / err).log10()).min(200.0)
    }
}

/// The shift of `test` against `reference` (within +-`max_lag` samples per
/// channel) that gives the highest correlation.
pub fn best_lag(reference: &[f32], test: &[f32], channels: usize, max_lag: isize) -> isize {
    let n = reference.len().min(test.len()) / channels;
    let mut best = (f64::NEG_INFINITY, 0isize);
    for lag in -max_lag..=max_lag {
        let mut dot = 0f64;
        let mut tt = 0f64;
        let mut rr = 0f64;
        for i in 0..n {
            let j = i as isize + lag;
            if j < 0 || j as usize >= n {
                continue;
            }
            for c in 0..channels {
                let r = reference[i * channels + c] as f64;
                let t = test[j as usize * channels + c] as f64;
                dot += r * t;
                tt += t * t;
                rr += r * r;
            }
        }
        let corr = dot / (rr * tt).sqrt().max(1e-30);
        // Prefer lag 0 on ties so identical signals report lag 0.
        if corr > best.0 + 1e-12 || (lag == 0 && corr >= best.0 - 1e-12) {
            best = (corr, lag);
        }
    }
    best.1
}

/// Compare `test` against `reference`, both interleaved, frame by frame
/// (`frame` samples per channel) with no time alignment applied. The lag is
/// reported separately.
pub fn compare(reference: &[f32], test: &[f32], channels: usize, frame: usize, rate: usize) -> Cmp {
    let n = reference.len().min(test.len());
    let (reference, test) = (&reference[..n], &test[..n]);
    let (mut sig, mut err, mut tt) = (0f64, 0f64, 0f64);
    let mut max_abs_diff = 0f64;
    let mut worst = (f64::INFINITY, 0usize);
    for (fi, (r, t)) in reference
        .chunks(frame * channels)
        .zip(test.chunks(frame * channels))
        .enumerate()
    {
        let (mut fs, mut fe) = (0f64, 0f64);
        for (a, b) in r.iter().zip(t) {
            let (a, b) = (*a as f64, *b as f64);
            fs += a * a;
            fe += (a - b) * (a - b);
            tt += b * b;
            max_abs_diff = max_abs_diff.max((a - b).abs());
        }
        sig += fs;
        err += fe;
        // Skip frames where the reference is below -70 dBFS: their SNR says
        // nothing about audible quality.
        if fs / r.len() as f64 > 1e-7 || fe / r.len() as f64 > 1e-7 {
            let s = snr_db(fs, fe);
            if s < worst.0 {
                worst = (s, fi);
            }
        }
    }
    if worst.0 == f64::INFINITY {
        worst = (snr_db(sig, err), 0);
    }
    // Worst 100 ms window correlation.
    let win = rate / 10 * channels;
    let mut worst_corr = 1.0f64;
    for (r, t) in reference.chunks_exact(win).zip(test.chunks_exact(win)) {
        let (mut d, mut rr, mut t2) = (0f64, 0f64, 0f64);
        for (a, b) in r.iter().zip(t) {
            d += *a as f64 * *b as f64;
            rr += *a as f64 * *a as f64;
            t2 += *b as f64 * *b as f64;
        }
        if rr / win as f64 > 1e-7 {
            worst_corr = worst_corr.min(d / (rr * t2).sqrt().max(1e-30));
        }
    }
    Cmp {
        snr_db: snr_db(sig, err),
        worst_frame_snr_db: worst.0,
        worst_frame: worst.1,
        max_abs_diff,
        level_ratio: if sig > 0.0 { (tt / sig).sqrt() } else { 1.0 },
        lag: best_lag(reference, test, channels, 60),
        worst_corr,
    }
}

/// SNR of every frame (`frame` samples per channel) of `test` against
/// `reference`, paired with the reference frame's mean power. No time
/// alignment.
pub fn frame_snrs(
    reference: &[f32],
    test: &[f32],
    channels: usize,
    frame: usize,
) -> Vec<(f64, f64)> {
    reference
        .chunks(frame * channels)
        .zip(test.chunks(frame * channels))
        .map(|(r, t)| {
            let (mut fs, mut fe) = (0f64, 0f64);
            for (a, b) in r.iter().zip(t) {
                let (a, b) = (*a as f64, *b as f64);
                fs += a * a;
                fe += (a - b) * (a - b);
            }
            (snr_db(fs, fe), fs / r.len() as f64)
        })
        .collect()
}

impl std::fmt::Display for Cmp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "snr {:6.1} dB, worst frame {:6.1} dB (#{}), max diff {:.2e}, level {:.3}, lag {}, worst corr {:.4}",
            self.snr_db,
            self.worst_frame_snr_db,
            self.worst_frame,
            self.max_abs_diff,
            self.level_ratio,
            self.lag,
            self.worst_corr
        )
    }
}

/// SNR of `test` against `reference` after shifting `test` by the best lag
/// within +-`max_lag` samples. Used for end-to-end quality against the
/// original input, where the codec delay is not known in advance.
#[allow(clippy::needless_range_loop)]
pub fn aligned_snr(reference: &[f32], test: &[f32], max_lag: isize) -> (f64, isize) {
    let n = reference.len().min(test.len());
    let mut best = (f64::NEG_INFINITY, 0isize);
    for lag in -max_lag..=max_lag {
        let (mut sig, mut err) = (0f64, 0f64);
        for i in 0..n {
            let j = i as isize + lag;
            if j < 0 || j as usize >= n {
                continue;
            }
            let r = reference[i] as f64;
            let t = test[j as usize] as f64;
            sig += r * r;
            err += (r - t) * (r - t);
        }
        let s = snr_db(sig, err);
        if s > best.0 {
            best = (s, lag);
        }
    }
    best
}

// ---------------------------------------------------------------------------
// Decoding a packet stream with both decoders
// ---------------------------------------------------------------------------

/// Decode `packets` with opus1 at `rate` / `channels`. Stops at the first
/// error and returns it together with the index of the failing packet.
pub fn rs_decode_all(
    packets: &[Vec<u8>],
    rate: usize,
    channels: usize,
) -> Result<Vec<f32>, (usize, opus1::Error)> {
    let mut dec = opus1::OpusDecoder::new(rate as i32, channels).unwrap();
    let max = rate * 120 / 1000;
    let mut pcm = vec![0f32; max * channels];
    let mut out = Vec::new();
    for (i, p) in packets.iter().enumerate() {
        let n = dec.decode(p, max, &mut pcm).map_err(|e| (i, e))?;
        out.extend_from_slice(&pcm[..n * channels]);
    }
    Ok(out)
}

/// Decode `packets` with C libopus at `rate` / `channels`.
pub fn c_decode_all(packets: &[Vec<u8>], rate: usize, channels: usize) -> Vec<f32> {
    let mut dec = CDec::new(rate, channels);
    let max = rate * 120 / 1000;
    let mut out = Vec::new();
    for p in packets {
        out.extend(dec.decode(Some(p), max, false).expect("libopus decode"));
    }
    out
}
