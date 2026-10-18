//! Signals: random numbers, WAV input and output, the resampler, and the
//! synthetic items of the built-in sets.
//!
//! Every generator has a fixed seed, so the items never change between
//! runs.

use std::f64::consts::{PI, TAU};
use std::path::Path;

/// The rate of every reference signal.
pub const REF_RATE: u32 = 48_000;

/// Peak level of the synthetic items (-6 dBFS).
const PEAK: f64 = 0.5;

/// xorshift64 generator with a fixed seed.
#[derive(Clone)]
pub struct XorShift(pub u64);

impl XorShift {
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    /// Uniform value in [-1, 1).
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (2.0 / (1u64 << 53) as f64) - 1.0
    }

    /// Uniform integer in [0, n).
    pub fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n.max(1)
    }
}

/// White noise through three one-pole low-pass filters (Paul Kellet's
/// "economy" filter). The spectrum falls by about 3 dB per octave.
pub struct PinkNoise {
    rng: XorShift,
    state: [f64; 3],
}

impl PinkNoise {
    pub fn new(seed: u64) -> Self {
        Self { rng: XorShift(seed), state: [0.0; 3] }
    }

    pub fn next(&mut self) -> f64 {
        let white = self.rng.uniform();
        self.state[0] = 0.99765 * self.state[0] + white * 0.099_046;
        self.state[1] = 0.96300 * self.state[1] + white * 0.296_516_4;
        self.state[2] = 0.57000 * self.state[2] + white * 1.052_691_3;
        self.state[0] + self.state[1] + self.state[2] + white * 0.1848
    }
}

// ---------------------------------------------------------------------------
// WAV files
// ---------------------------------------------------------------------------

/// Audio read from a WAV file: interleaved samples scaled to [-1, 1).
pub struct Wav {
    pub rate: u32,
    pub channels: usize,
    pub samples: Vec<f64>,
}

/// Read a WAV file: PCM with 16 or 24 bits, or 32-bit float, with one or
/// two channels, at 8 to 48 kHz. Chunks other than `fmt ` and `data` (such
/// as the `LIST` chunks of the TSP files) are skipped.
pub fn read_wav(path: &Path) -> Result<Wav, String> {
    let name = path.display();
    let bytes = std::fs::read(path).map_err(|e| format!("{name}: {e}"))?;
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(format!("{name}: not a RIFF/WAVE file"));
    }
    let mut pos = 12;
    let mut format: Option<(u16, usize, u32, u16)> = None;
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let size = u32::from_le_bytes([bytes[pos + 4], bytes[pos + 5], bytes[pos + 6], bytes[pos + 7]])
            as usize;
        let body = &bytes[pos + 8..(pos + 8).saturating_add(size).min(bytes.len())];
        if id == b"fmt " && body.len() >= 16 {
            let mut tag = u16::from_le_bytes([body[0], body[1]]);
            let channels = u16::from_le_bytes([body[2], body[3]]) as usize;
            let rate = u32::from_le_bytes([body[4], body[5], body[6], body[7]]);
            let bits = u16::from_le_bytes([body[14], body[15]]);
            // WAVE_FORMAT_EXTENSIBLE: the real tag is the first two bytes
            // of the sub-format GUID.
            if tag == 0xFFFE && body.len() >= 26 {
                tag = u16::from_le_bytes([body[24], body[25]]);
            }
            format = Some((tag, channels, rate, bits));
        } else if id == b"data" {
            let Some((tag, channels, rate, bits)) = format else {
                return Err(format!("{name}: data chunk before fmt chunk"));
            };
            if !(1..=2).contains(&channels) || !(8000..=48000).contains(&rate) {
                return Err(format!("{name}: {channels} channel(s) at {rate} Hz is not supported"));
            }
            let samples: Vec<f64> = match (tag, bits) {
                (1, 16) => body.as_chunks::<2>().0.iter().map(|&b| i16::from_le_bytes(b) as f64 / 32768.0).collect(),
                (1, 24) => body
                    .as_chunks::<3>()
                    .0
                    .iter()
                    .map(|b| (i32::from_le_bytes([0, b[0], b[1], b[2]]) >> 8) as f64 / 8_388_608.0)
                    .collect(),
                (3, 32) => body.as_chunks::<4>().0.iter().map(|&b| f32::from_le_bytes(b) as f64).collect(),
                _ => return Err(format!("{name}: format {tag} with {bits} bits is not supported")),
            };
            let frames = samples.len() / channels;
            let mut samples = samples;
            samples.truncate(frames * channels);
            return Ok(Wav { rate, channels, samples });
        }
        pos += 8 + size + (size & 1);
    }
    Err(format!("{name}: no data chunk"))
}

/// Write interleaved 16-bit samples as a PCM WAV file.
pub fn write_wav(path: &Path, rate: u32, channels: usize, samples: &[i16]) -> Result<(), String> {
    let data_len = (samples.len() * 2) as u32;
    let mut out = Vec::with_capacity(44 + samples.len() * 2);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_len).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&(channels as u16).to_le_bytes());
    out.extend_from_slice(&rate.to_le_bytes());
    out.extend_from_slice(&(rate * 2 * channels as u32).to_le_bytes());
    out.extend_from_slice(&(2 * channels as u16).to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    for s in samples {
        out.extend_from_slice(&s.to_le_bytes());
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
    }
    std::fs::write(path, out).map_err(|e| format!("{}: {e}", path.display()))
}

/// Round to 16 bits, saturating at the limits.
pub fn to_i16(x: f64) -> i16 {
    (x * 32768.0).round().clamp(-32768.0, 32767.0) as i16
}

/// Split interleaved samples into channels.
pub fn deinterleave(samples: &[f64], channels: usize) -> Vec<Vec<f64>> {
    (0..channels)
        .map(|c| samples.iter().skip(c).step_by(channels).copied().collect())
        .collect()
}

/// Interleave channels of equal length.
pub fn interleave(channels: &[Vec<f64>]) -> Vec<f64> {
    let n = channels.iter().map(Vec::len).min().unwrap_or(0);
    let mut out = Vec::with_capacity(n * channels.len());
    for i in 0..n {
        for ch in channels {
            out.push(ch[i]);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Resampler
// ---------------------------------------------------------------------------

fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// Modified Bessel function of the first kind, order zero (for the Kaiser
/// window).
fn bessel_i0(x: f64) -> f64 {
    let q = x * x / 4.0;
    let mut sum = 1.0;
    let mut term = 1.0;
    for k in 1..500 {
        term *= q / (k * k) as f64;
        sum += term;
        if term < 1e-17 * sum {
            break;
        }
    }
    sum
}

/// Polyphase resampler with a Kaiser-windowed sinc low-pass filter.
///
/// The filter is designed for the comparison of section 4.6 of the design
/// note: `opus_compare` at a rate R looks at the band up to R/2 - 300 Hz,
/// so the filter must be flat there. It passes up to N - 300 Hz and stops
/// from N, where N is the lower of the two Nyquist frequencies. The design
/// asks for at least 90 dB of attenuation from N; the Kaiser window is
/// designed for 96 dB (beta 9.62), so that the edge itself has a margin.
/// Its length is what the Kaiser formula gives for a 300 Hz transition
/// (about 980 taps at 48 kHz). The pass-band ripple is below 0.001 dB. The
/// filter is symmetric, so it adds no delay.
pub struct Resampler {
    up: usize,
    down: usize,
    /// Coefficients per output phase.
    taps: usize,
    /// `up` rows of `taps` coefficients.
    table: Vec<f64>,
}

impl Resampler {
    pub fn new(from: u32, to: u32) -> Self {
        const ATTENUATION_DB: f64 = 96.0;
        const TRANSITION_HZ: f64 = 300.0;
        let g = gcd(from as u64, to as u64);
        let up = (to as u64 / g) as usize;
        let down = (from as u64 / g) as usize;
        // The filter runs at the upsampled rate.
        let fs = from as f64 * up as f64;
        let nyquist = from.min(to) as f64 / 2.0;
        let cutoff = (nyquist - TRANSITION_HZ / 2.0) / fs; // cycles per sample
        let width = TAU * TRANSITION_HZ / fs; // radians per sample
        let beta = 0.1102 * (ATTENUATION_DB - 8.7);
        let length = ((ATTENUATION_DB - 8.0) / (2.285 * width)).ceil() as usize;
        let taps = length.div_ceil(up).next_multiple_of(2).max(2);
        let half = taps / 2;
        // Half-length of the window at the upsampled rate.
        let reach = (half * up) as f64;
        let i0_beta = bessel_i0(beta);
        let mut table = vec![0.0; up * taps];
        for (phase, row) in table.chunks_exact_mut(taps).enumerate() {
            for (j, coef) in row.iter_mut().enumerate() {
                // Distance from the output instant to input sample
                // `base - half + 1 + j`, at the upsampled rate.
                let t = phase as f64 + (half as f64 - 1.0 - j as f64) * up as f64;
                let u = t / reach;
                let window = if u.abs() < 1.0 { bessel_i0(beta * (1.0 - u * u).sqrt()) / i0_beta } else { 0.0 };
                let x = PI * 2.0 * cutoff * t;
                let sinc = if x.abs() < 1e-12 { 1.0 } else { x.sin() / x };
                *coef = 2.0 * cutoff * sinc * window;
            }
            // Unity gain at 0 Hz for every phase.
            let sum: f64 = row.iter().sum();
            for coef in row.iter_mut() {
                *coef /= sum;
            }
        }
        Self { up, down, taps, table }
    }

    /// Resample one channel. The output has `input.len() * to / from`
    /// samples (rounded down) and is aligned with the input: output sample
    /// n lies at input time `n * from / to`.
    pub fn run(&self, input: &[f64]) -> Vec<f64> {
        if self.up == 1 && self.down == 1 {
            return input.to_vec();
        }
        let half = self.taps / 2;
        let mut padded = vec![0.0; input.len() + self.taps + 2];
        padded[half..half + input.len()].copy_from_slice(input);
        let out_len = input.len() * self.up / self.down;
        let mut out = Vec::with_capacity(out_len);
        for n in 0..out_len {
            let position = n * self.down;
            let base = position / self.up;
            let row = &self.table[(position % self.up) * self.taps..][..self.taps];
            // Input sample `base - half + 1 + j` is `padded[base + 1 + j]`.
            let window = &padded[base + 1..base + 1 + self.taps];
            out.push(dot(row, window));
        }
        out
    }
}

/// Dot product with eight independent partial sums, so that the compiler
/// can use vector instructions.
fn dot(a: &[f64], b: &[f64]) -> f64 {
    let mut acc = [0.0f64; 8];
    let (ca, ra) = a.as_chunks::<8>();
    let (cb, rb) = b.as_chunks::<8>();
    for (x, y) in ca.iter().zip(cb) {
        for k in 0..8 {
            acc[k] += x[k] * y[k];
        }
    }
    let mut sum = ra.iter().zip(rb).map(|(x, y)| x * y).sum::<f64>();
    for v in acc {
        sum += v;
    }
    sum
}

/// Resample interleaved audio from `from` to `to` Hz.
pub fn resample_interleaved(samples: &[f64], channels: usize, from: u32, to: u32) -> Vec<f64> {
    if from == to {
        return samples.to_vec();
    }
    let resampler = Resampler::new(from, to);
    let parts: Vec<Vec<f64>> = deinterleave(samples, channels).iter().map(|c| resampler.run(c)).collect();
    interleave(&parts)
}

// ---------------------------------------------------------------------------
// Synthetic items (48 kHz)
// ---------------------------------------------------------------------------

fn peak_of(channels: &[Vec<f64>]) -> f64 {
    channels.iter().flat_map(|ch| ch.iter()).fold(0.0f64, |m, &v| m.max(v.abs()))
}

/// Scale to the peak level and round to 16 bits, interleaved.
pub fn finish(mut channels: Vec<Vec<f64>>) -> Vec<i16> {
    let peak = peak_of(&channels);
    if peak > 0.0 {
        let gain = PEAK / peak;
        for v in channels.iter_mut().flat_map(|ch| ch.iter_mut()) {
            *v *= gain;
        }
    }
    interleave(&channels).iter().map(|&v| to_i16(v)).collect()
}

fn add_pink_noise(channel: &mut [f64], seed: u64, rms: f64) {
    let mut generator = PinkNoise::new(seed);
    let noise: Vec<f64> = (0..channel.len()).map(|_| generator.next()).collect();
    let power = noise.iter().map(|v| v * v).sum::<f64>() / noise.len().max(1) as f64;
    let gain = if power > 0.0 { rms / power.sqrt() } else { 0.0 };
    for (dst, n) in channel.iter_mut().zip(&noise) {
        *dst += gain * n;
    }
}

/// Constant-power pan: 0 is hard left, 1 is hard right.
fn pan_gains(pan: f64) -> (f64, f64) {
    let angle = pan * std::f64::consts::FRAC_PI_2;
    (angle.cos(), angle.sin())
}

/// Synthetic music in stereo, `seconds` long, made of 250 ms steps.
///
/// It is the music signal of `examples/perf_harness.rs` with a seed:
/// five voices play harmonic notes of a chord sequence, with percussion
/// (bass drum, snare, high-pass noise bursts) and a pink noise floor. The
/// seed chooses the order of the chords, the melody, the positions of the
/// voices, how long the notes ring and whether percussion plays. Seed 0
/// gives the chord order and positions of `perf_harness`.
pub fn music(seed: u64, seconds: f64) -> Vec<Vec<f64>> {
    const CHORDS: [[u8; 4]; 8] = [
        [45, 52, 60, 64],
        [41, 53, 57, 60],
        [48, 55, 60, 64],
        [43, 50, 59, 62],
        [50, 57, 62, 65],
        [45, 52, 57, 64],
        [40, 52, 56, 59],
        [45, 57, 60, 69],
    ];
    const VOICES: [(f64, f64, f64, f64); 5] = [
        (0.50, 1.0, 0.50, 1.00),
        (0.08, 1.2, 0.35, 0.70),
        (0.92, 1.2, 0.35, 0.70),
        (0.25, 1.0, 0.30, 0.60),
        (0.80, 0.8, 0.12, 0.60),
    ];
    const ATTACK: f64 = 0.004;
    const RELEASE: f64 = 0.010;

    let rate = REF_RATE;
    let fs = rate as f64;
    let step_len = (rate / 4) as usize;
    let steps = ((seconds * 4.0).ceil() as usize).max(1);
    let step_seconds = step_len as f64 / fs;
    let n = step_len * steps;
    let max_frequency = (0.45 * fs).min(20_000.0);
    let mut left = vec![0.0f64; n];
    let mut right = vec![0.0f64; n];
    let edges = |t: f64| (t / ATTACK).min(1.0) * ((step_seconds - t) / RELEASE).clamp(0.0, 1.0);

    let mut choice = XorShift(0x1234_5678_9ABC_DEF1 ^ seed.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    let chord_offset = if seed == 0 { 0 } else { choice.below(8) as usize };
    let chord_stride = if seed == 0 { 1 } else { [1, 3, 5, 7][choice.below(4) as usize] };
    // Ring longer for some seeds (sustained notes), shorter for others.
    let decay_scale = if seed == 0 { 1.0 } else { [0.5, 1.0, 2.0, 4.0][choice.below(4) as usize] };
    let percussion = seed == 0 || choice.below(4) != 0;
    let pans: Vec<f64> = VOICES
        .iter()
        .map(|v| if seed == 0 { v.0 } else { (v.0 + 0.3 * choice.uniform()).clamp(0.0, 1.0) })
        .collect();

    let mut melody = XorShift(0x9E37_79B9_7F4A_7C15 ^ seed);
    for step in 0..steps {
        let chord = CHORDS[(chord_offset + step * chord_stride) % CHORDS.len()];
        let melody_note = chord[(melody.next_u64() % 4) as usize] + 24 + 12 * (melody.next_u64() % 2) as u8;
        let melody_rests = melody.next_u64().is_multiple_of(4);
        let notes = [chord[0], chord[1], chord[2], chord[3], melody_note];
        let base = step * step_len;
        for (voice, &(_, roll_off, decay, gain)) in VOICES.iter().enumerate() {
            if voice == 4 && melody_rests {
                continue;
            }
            let f0 = 440.0 * 2f64.powf((notes[voice] as f64 - 69.0) / 12.0);
            let harmonics = ((max_frequency / f0) as usize).min(64);
            if harmonics == 0 {
                continue;
            }
            let weights: Vec<f64> = (1..=harmonics).map(|h| (h as f64).powf(-roll_off)).collect();
            let (gain_left, gain_right) = pan_gains(pans[voice]);
            let omega = TAU * f0 / fs;
            let decay = decay * decay_scale;
            for i in 0..step_len {
                let t = i as f64 / fs;
                let x = omega * i as f64;
                let two_cos = 2.0 * x.cos();
                let mut previous = 0.0;
                let mut current = x.sin();
                let mut sum = weights[0] * current;
                for &weight in &weights[1..] {
                    let next = two_cos * current - previous;
                    sum += weight * next;
                    previous = current;
                    current = next;
                }
                let y = gain * edges(t) * (-t / decay).exp() * sum;
                left[base + i] += gain_left * y;
                right[base + i] += gain_right * y;
            }
        }
    }

    if percussion {
        let mut noise_left = XorShift(0x0123_4567_89AB_CDEF ^ seed);
        let mut noise_right = XorShift(0xFEDC_BA98_7654_3211 ^ seed);
        let eighth = step_len / 2;
        for hit in 0..steps * 2 {
            let base = hit * eighth;
            let step = hit / 2;
            let on_beat = hit.is_multiple_of(2);
            let (decay, pan) = if on_beat { (0.012, 0.15) } else { (0.035, 0.85) };
            let (gain_left, gain_right) = pan_gains(pan);
            let (mut last_left, mut last_right) = (0.0, 0.0);
            for i in 0..eighth {
                let t = i as f64 / fs;
                let envelope = 0.30 * (-t / decay).exp();
                let (white_left, white_right) = (noise_left.uniform(), noise_right.uniform());
                left[base + i] += gain_left * envelope * (white_left - last_left);
                right[base + i] += gain_right * envelope * (white_right - last_right);
                last_left = white_left;
                last_right = white_right;
            }
            if on_beat && step % 4 == 0 {
                for i in 0..step_len.min(n - base) {
                    let t = i as f64 / fs;
                    let phase = TAU * (50.0 * t + 100.0 * 0.03 * (1.0 - (-t / 0.03).exp()));
                    let y = 0.9 * edges(t) * (-t / 0.12).exp() * phase.sin();
                    left[base + i] += std::f64::consts::FRAC_1_SQRT_2 * y;
                    right[base + i] += std::f64::consts::FRAC_1_SQRT_2 * y;
                }
            }
            if on_beat && step % 4 == 2 {
                for i in 0..step_len.min(n - base) {
                    let t = i as f64 / fs;
                    let fade = edges(t);
                    let noise = 0.5 * fade * (-t / 0.05).exp();
                    let tone = 0.4 * fade * (-t / 0.03).exp() * (TAU * 180.0 * t).sin();
                    left[base + i] += 0.8 * noise * noise_left.uniform() + 0.75 * tone;
                    right[base + i] += 0.6 * noise * noise_right.uniform() + 0.5 * tone;
                }
            }
        }
    }

    let mut out = vec![left, right];
    let noise_rms = 0.006 * peak_of(&out);
    add_pink_noise(&mut out[0], 0xA5A5_5A5A_1234_5678 ^ seed, noise_rms);
    add_pink_noise(&mut out[1], 0x0F0F_F0F0_8765_4321 ^ seed, noise_rms);
    trim(out, seconds)
}

/// Cut every channel to `seconds`.
fn trim(mut channels: Vec<Vec<f64>>, seconds: f64) -> Vec<Vec<f64>> {
    let n = (seconds * REF_RATE as f64).round() as usize;
    for ch in channels.iter_mut() {
        ch.truncate(n);
    }
    channels
}

/// Castanets in stereo: short bursts of high-passed noise every 250 ms that
/// decay within a few milliseconds, alternating between left and right,
/// over a pink noise floor 60 dB down. A test of pre-echo.
pub fn castanets(seconds: f64) -> Vec<Vec<f64>> {
    let fs = REF_RATE as f64;
    let n = (seconds * fs).round() as usize;
    let mut out = vec![vec![0.0; n], vec![0.0; n]];
    let mut rng = XorShift(0xCA57_A2E7_0000_0001);
    let period = REF_RATE as usize / 4;
    for (hit, base) in (0..n).step_by(period).enumerate() {
        let pan = if hit % 2 == 0 { 0.2 } else { 0.8 };
        let (gl, gr) = pan_gains(pan);
        // Two clicks 15 ms apart, as a pair of castanet shells.
        for (offset, gain) in [(0usize, 1.0), ((0.015 * fs) as usize, 0.6)] {
            let mut last = 0.0;
            for i in 0..(0.03 * fs) as usize {
                let k = base + offset + i;
                if k >= n {
                    break;
                }
                let t = i as f64 / fs;
                let white = rng.uniform();
                let y = gain * (-t / 0.003).exp() * (white - last);
                last = white;
                out[0][k] += gl * y;
                out[1][k] += gr * y;
            }
        }
    }
    let floor = 0.001 * peak_of(&out);
    add_pink_noise(&mut out[0], 0xCA57_0000_0000_0002, floor);
    add_pink_noise(&mut out[1], 0xCA57_0000_0000_0003, floor);
    out
}

/// Three steady sinusoids (220, 1250 and 3100 Hz) with a slow vibrato of
/// 5 Hz, in mono. A test of the pitch filter and of tonal content.
pub fn tones(seconds: f64) -> Vec<Vec<f64>> {
    let fs = REF_RATE as f64;
    let n = (seconds * fs).round() as usize;
    let parts = [(220.0, 1.0, 0.004), (1250.0, 0.5, 0.003), (3100.0, 0.25, 0.002)];
    let mut out = vec![0.0; n];
    for &(f, gain, depth) in &parts {
        let mut phase = 0.0f64;
        for (i, v) in out.iter_mut().enumerate() {
            let t = i as f64 / fs;
            let freq = f * (1.0 + depth * (TAU * 5.0 * t).sin());
            phase += TAU * freq / fs;
            *v += gain * phase.sin();
        }
    }
    vec![out]
}

/// Pink noise in stereo, with independent channels.
pub fn pink_noise(seconds: f64) -> Vec<Vec<f64>> {
    let n = (seconds * REF_RATE as f64).round() as usize;
    let mut left = PinkNoise::new(0x9172_0000_0000_0001);
    let mut right = PinkNoise::new(0x9172_0000_0000_0002);
    vec![(0..n).map(|_| left.next()).collect(), (0..n).map(|_| right.next()).collect()]
}

/// Two-pole resonator with centre frequency `f` and bandwidth `bw` (Hz).
struct Resonator {
    a1: f64,
    a2: f64,
    gain: f64,
    y1: f64,
    y2: f64,
}

impl Resonator {
    fn new(f: f64, bw: f64) -> Self {
        let mut r = Self { a1: 0.0, a2: 0.0, gain: 0.0, y1: 0.0, y2: 0.0 };
        r.set(f, bw);
        r
    }

    fn set(&mut self, f: f64, bw: f64) {
        let fs = REF_RATE as f64;
        let radius = (-PI * bw / fs).exp();
        self.a1 = 2.0 * radius * (TAU * f / fs).cos();
        self.a2 = -radius * radius;
        self.gain = 1.0 - radius;
    }

    fn run(&mut self, x: f64) -> f64 {
        let y = self.gain * x + self.a1 * self.y1 + self.a2 * self.y2;
        self.y2 = self.y1;
        self.y1 = y;
        y
    }
}

/// Synthetic fullband vowels in mono: a glottal pulse train with a pitch
/// that glides between 100 and 220 Hz, through three formant resonators
/// that change vowel every 300 ms, with bursts of noise above 4 kHz (as
/// fricatives) every 700 ms.
pub fn vowels(seconds: f64) -> Vec<Vec<f64>> {
    const FORMANTS: [[f64; 3]; 5] = [
        [730.0, 1090.0, 2440.0],
        [270.0, 2290.0, 3010.0],
        [530.0, 1840.0, 2480.0],
        [570.0, 840.0, 2410.0],
        [300.0, 870.0, 2240.0],
    ];
    let fs = REF_RATE as f64;
    let n = (seconds * fs).round() as usize;
    let mut out = vec![0.0; n];
    let mut filters = [Resonator::new(730.0, 90.0), Resonator::new(1090.0, 110.0), Resonator::new(2440.0, 170.0)];
    let mut phase = 0.0f64;
    let mut rng = XorShift(0x50E1_0000_0000_0001);
    let mut hp_last = 0.0;
    let mut hp_state = 0.0;
    for (i, v) in out.iter_mut().enumerate() {
        let t = i as f64 / fs;
        if i % (0.3 * fs) as usize == 0 {
            let vowel = FORMANTS[(i / (0.3 * fs) as usize) % FORMANTS.len()];
            for (k, f) in filters.iter_mut().enumerate() {
                f.set(vowel[k], [90.0, 110.0, 170.0][k]);
            }
        }
        let f0 = 160.0 + 60.0 * (TAU * 0.7 * t).sin();
        phase += f0 / fs;
        // A glottal pulse: a decaying spike at each period, with a slight
        // spectral tilt, so the harmonics reach up to 20 kHz.
        let source = if phase >= 1.0 {
            phase -= 1.0;
            1.0
        } else {
            0.0
        };
        let mut y = 0.0;
        for (k, f) in filters.iter_mut().enumerate() {
            y += [1.0, 0.6, 0.3][k] * f.run(source);
        }
        // Fricative: high-passed noise for 120 ms every 700 ms.
        let in_burst = (t % 0.7) < 0.12;
        let white = rng.uniform();
        hp_state = 0.9 * hp_state + white - hp_last;
        hp_last = white;
        let fricative = if in_burst { 0.02 * hp_state } else { 0.0 };
        *v = y + fricative;
    }
    vec![out]
}
