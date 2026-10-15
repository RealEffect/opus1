//! Shared helpers for the `silkenc_*` tests: thin wrappers over the raw C
//! libopus API (`audiopus_sys`), test signals and comparison functions.

#![allow(dead_code)]

use audiopus_sys as sys;

/// Private libopus request (opus_private.h).
pub const OPUS_SET_FORCE_MODE_REQUEST: i32 = 11002;
pub const OPUS_GET_FINAL_RANGE_REQUEST: i32 = 4031;
pub const MODE_SILK: i32 = 1000;

pub const BW_NB: i32 = sys::OPUS_BANDWIDTH_NARROWBAND;
pub const BW_MB: i32 = sys::OPUS_BANDWIDTH_MEDIUMBAND;
pub const BW_WB: i32 = sys::OPUS_BANDWIDTH_WIDEBAND;

pub const APP_VOIP: i32 = sys::OPUS_APPLICATION_VOIP;
pub const APP_AUDIO: i32 = sys::OPUS_APPLICATION_AUDIO;

/// C libopus encoder.
pub struct CEncoder {
    st: *mut sys::OpusEncoder,
    pub channels: usize,
}

impl CEncoder {
    pub fn new(rate: usize, channels: usize, application: i32) -> Self {
        let mut err = 0;
        let st = unsafe {
            sys::opus_encoder_create(rate as i32, channels as i32, application, &mut err)
        };
        assert_eq!(err, 0, "opus_encoder_create({rate}, {channels})");
        Self { st, channels }
    }

    /// A "set" control that takes one `opus_int32`.
    pub fn set(&mut self, request: i32, value: i32) -> &mut Self {
        let ret = unsafe { sys::opus_encoder_ctl(self.st, request, value) };
        assert_eq!(ret, 0, "opus_encoder_ctl({request}, {value}) = {ret}");
        self
    }

    pub fn bitrate(&mut self, bps: i32) -> &mut Self {
        self.set(sys::OPUS_SET_BITRATE_REQUEST, bps)
    }
    pub fn silk_only(&mut self) -> &mut Self {
        self.set(OPUS_SET_FORCE_MODE_REQUEST, MODE_SILK)
    }
    pub fn bandwidth(&mut self, bw: i32) -> &mut Self {
        self.set(sys::OPUS_SET_BANDWIDTH_REQUEST, bw)
    }
    pub fn max_bandwidth(&mut self, bw: i32) -> &mut Self {
        self.set(sys::OPUS_SET_MAX_BANDWIDTH_REQUEST, bw)
    }
    pub fn complexity(&mut self, c: i32) -> &mut Self {
        self.set(sys::OPUS_SET_COMPLEXITY_REQUEST, c)
    }
    pub fn vbr(&mut self, on: bool) -> &mut Self {
        self.set(sys::OPUS_SET_VBR_REQUEST, on as i32)
    }
    pub fn fec(&mut self, on: bool, loss_perc: i32) -> &mut Self {
        self.set(sys::OPUS_SET_INBAND_FEC_REQUEST, on as i32);
        self.set(sys::OPUS_SET_PACKET_LOSS_PERC_REQUEST, loss_perc)
    }
    pub fn force_channels(&mut self, ch: i32) -> &mut Self {
        self.set(sys::OPUS_SET_FORCE_CHANNELS_REQUEST, ch)
    }

    /// `OPUS_GET_FINAL_RANGE` after the last `encode`.
    pub fn final_range(&mut self) -> u32 {
        let mut value = 0u32;
        let ret = unsafe {
            sys::opus_encoder_ctl(self.st, OPUS_GET_FINAL_RANGE_REQUEST, &mut value as *mut u32)
        };
        assert_eq!(ret, 0, "OPUS_GET_FINAL_RANGE");
        value
    }

    /// Encode one frame of interleaved `f32` PCM into at most `max_bytes`.
    pub fn encode_max(&mut self, pcm: &[f32], max_bytes: usize) -> Vec<u8> {
        let frame = pcm.len() / self.channels;
        let mut out = vec![0u8; max_bytes];
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

    pub fn encode(&mut self, pcm: &[f32]) -> Vec<u8> {
        self.encode_max(pcm, 4000)
    }
}

impl Drop for CEncoder {
    fn drop(&mut self) {
        unsafe { sys::opus_encoder_destroy(self.st) }
    }
}

/// C libopus decoder.
pub struct CDecoder {
    st: *mut sys::OpusDecoder,
    pub channels: usize,
}

impl CDecoder {
    pub fn new(rate: usize, channels: usize) -> Self {
        let mut err = 0;
        let st = unsafe { sys::opus_decoder_create(rate as i32, channels as i32, &mut err) };
        assert_eq!(err, 0, "opus_decoder_create({rate}, {channels})");
        Self { st, channels }
    }

    /// `opus_decode_float`. `None` is a lost packet. With `fec` the decoder
    /// rebuilds the frame before `packet` from the copy `packet` carries.
    /// Returns the interleaved samples, or the libopus error code.
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
            sys::opus_decode_float(self.st, ptr, len, pcm.as_mut_ptr(), frame_size as i32, fec as i32)
        };
        if n < 0 {
            return Err(n);
        }
        pcm.truncate(n as usize * self.channels);
        Ok(pcm)
    }

    /// `OPUS_GET_FINAL_RANGE` after the last decode call.
    pub fn final_range(&mut self) -> u32 {
        let mut value = 0u32;
        let ret = unsafe {
            sys::opus_decoder_ctl(self.st, OPUS_GET_FINAL_RANGE_REQUEST, &mut value as *mut u32)
        };
        assert_eq!(ret, 0, "OPUS_GET_FINAL_RANGE");
        value
    }
}

impl Drop for CDecoder {
    fn drop(&mut self) {
        unsafe { sys::opus_decoder_destroy(self.st) }
    }
}

/// A small deterministic random number generator (xorshift32).
pub struct Rng(pub u32);

impl Rng {
    pub fn next(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        x
    }

    /// Uniform in -1.0..1.0.
    pub fn unit(&mut self) -> f32 {
        (self.next() >> 8) as f32 / 8_388_608.0 - 1.0
    }
}

/// `n` samples per channel of a speech-like signal at `fs` Hz, interleaved:
/// voiced stretches (a glottal pulse train with a moving pitch through two
/// resonances) that alternate with weaker noise and short pauses. The right
/// channel of a stereo signal is the left one delayed, at another level, with
/// its own noise.
pub fn speech_like(fs: usize, channels: usize, n: usize, seed: u32) -> Vec<f32> {
    synth(fs, channels, n, seed, true)
}

/// Like [`speech_like`] without the noise and the pauses: voiced throughout,
/// with a level that rises and falls, so that SILK finds voice activity in
/// every frame.
pub fn voiced_like(fs: usize, channels: usize, n: usize, seed: u32) -> Vec<f32> {
    synth(fs, channels, n, seed, false)
}

fn synth(fs: usize, channels: usize, n: usize, seed: u32, pauses: bool) -> Vec<f32> {
    let mut rng = Rng(seed | 1);
    let mut mono = vec![0f32; n];
    let mut phase = 0f64;
    // Two-pole resonators (formants) at 700 Hz and 1800 Hz.
    let mut res = [[0f64; 2]; 2];
    let coef = |f: f64, bw: f64| {
        let r = (-std::f64::consts::PI * bw / fs as f64).exp();
        (2.0 * r * (2.0 * std::f64::consts::PI * f / fs as f64).cos(), -r * r)
    };
    let formants = [coef(700.0, 130.0), coef(1800.0, 200.0)];
    for (i, out) in mono.iter_mut().enumerate() {
        let t = i as f64 / fs as f64;
        // A 600 ms pattern: 380 ms voiced, 140 ms noise, 80 ms pause.
        let pos = (t * 1000.0) % 600.0;
        let pitch = 110.0 + 40.0 * (2.0 * std::f64::consts::PI * 0.7 * t).sin();
        phase += pitch / fs as f64;
        let mut excitation = 0.0;
        if phase >= 1.0 {
            phase -= 1.0;
            excitation = 1.0;
        }
        let noise = rng.unit() as f64;
        let x = if !pauses {
            let env = 0.65 + 0.35 * (2.0 * std::f64::consts::PI * 1.3 * t).sin();
            env * (excitation + 0.01 * noise)
        } else if pos < 380.0 {
            let env = (std::f64::consts::PI * pos / 380.0).sin().max(0.0).sqrt();
            env * (excitation + 0.01 * noise)
        } else if pos < 520.0 {
            0.02 * noise
        } else {
            0.0005 * noise
        };
        let mut y = x;
        for (state, &(a1, a2)) in res.iter_mut().zip(&formants) {
            let v = y + a1 * state[0] + a2 * state[1];
            state[1] = state[0];
            state[0] = v;
            y = v;
        }
        *out = (0.02 * y).clamp(-0.9, 0.9) as f32;
    }
    if channels == 1 {
        return mono;
    }
    let delay = fs / 2000;
    let mut out = vec![0f32; 2 * n];
    for i in 0..n {
        out[2 * i] = mono[i];
        let delayed = if i >= delay { mono[i - delay] } else { 0.0 };
        out[2 * i + 1] = 0.6 * delayed + 0.002 * rng.unit();
    }
    out
}

/// `fixtures/answer_16k.wav`: 18 s of mono speech, 16-bit, 16 kHz.
pub fn speech_16k() -> Vec<f32> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/answer_16k.wav");
    let bytes = std::fs::read(path).expect("fixtures/answer_16k.wav");
    // Find the "data" chunk instead of assuming a 44-byte header.
    let mut pos = 12;
    while pos + 8 <= bytes.len() {
        let len = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into().unwrap()) as usize;
        if &bytes[pos..pos + 4] == b"data" {
            let data = &bytes[pos + 8..(pos + 8 + len).min(bytes.len())];
            return data
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| i16::from_le_bytes(*b) as f32 / 32768.0)
                .collect();
        }
        pos += 8 + len + (len & 1);
    }
    panic!("no data chunk in answer_16k.wav");
}

/// `n` samples per channel of real speech at `fs` Hz, interleaved, starting
/// `start_s` seconds into the fixture. The right channel of a stereo signal
/// is the left one delayed by 0.5 ms at 0.6 times the level, plus weak noise.
pub fn speech(fs: usize, channels: usize, n: usize, start_s: f64) -> Vec<f32> {
    let src = speech_16k();
    let start = (start_s * 16000.0) as usize;
    // Windowed-sinc interpolation of the 16 kHz fixture at `fs`.
    let ratio = 16000.0 / fs as f64;
    let cutoff = (fs as f64 / 16000.0).min(1.0) * 0.92;
    let half = 24i64;
    let mono: Vec<f32> = (0..n)
        .map(|i| {
            let t = start as f64 + i as f64 * ratio;
            let base = t.floor() as i64;
            let mut acc = 0f64;
            for k in base - half + 1..=base + half {
                if k < 0 || k as usize >= src.len() {
                    continue;
                }
                let x = t - k as f64;
                let sinc = if x.abs() < 1e-9 {
                    cutoff
                } else {
                    (std::f64::consts::PI * cutoff * x).sin() / (std::f64::consts::PI * x)
                };
                let w = 0.5 + 0.5 * (std::f64::consts::PI * x / half as f64).cos();
                acc += src[k as usize] as f64 * sinc * w;
            }
            acc as f32
        })
        .collect();
    if channels == 1 {
        return mono;
    }
    let mut rng = Rng(0x2545_f491);
    let delay = fs / 2000;
    let mut out = vec![0f32; 2 * n];
    for i in 0..n {
        out[2 * i] = mono[i];
        let delayed = if i >= delay { mono[i - delay] } else { 0.0 };
        out[2 * i + 1] = 0.6 * delayed + 0.001 * rng.unit();
    }
    out
}

/// Signal-to-noise ratio of `test` against `reference` in dB, over the
/// samples both have. 200 means identical.
pub fn snr(reference: &[f32], test: &[f32]) -> f64 {
    let (mut sig, mut err) = (0f64, 0f64);
    for (a, b) in reference.iter().zip(test) {
        sig += (*a as f64).powi(2);
        err += ((*a - *b) as f64).powi(2);
    }
    if err <= 0.0 {
        200.0
    } else if sig <= 0.0 {
        -200.0
    } else {
        (10.0 * (sig / err).log10()).min(200.0)
    }
}

/// Energy of `x` in dB relative to full scale; -200 for silence.
pub fn level_db(x: &[f32]) -> f64 {
    let e: f64 = x.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / x.len().max(1) as f64;
    if e <= 0.0 { -200.0 } else { 10.0 * e.log10() }
}

/// The best signal-to-noise ratio of `test` against `reference` over delays
/// of `test` from 0 to `max_delay` samples per channel, and that delay.
pub fn best_snr(reference: &[f32], test: &[f32], channels: usize, max_delay: usize) -> (f64, usize) {
    let mut best = (-200.0, 0);
    for d in 0..=max_delay {
        let off = d * channels;
        if off >= test.len() {
            break;
        }
        let n = reference.len().min(test.len() - off);
        let s = snr(&reference[..n], &test[off..off + n]);
        if s > best.0 {
            best = (s, d);
        }
    }
    best
}
