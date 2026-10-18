//! Test signals for the `eval_*` tests.
//!
//! Every sample is computed with integer arithmetic and the four basic
//! floating-point operations only (no `sin` of the platform's math
//! library), so that every platform generates the same samples. The tests
//! hash the inputs and check the hash against the recorded one.

use std::f64::consts::TAU;

/// Deterministic random numbers (xorshift64).
pub struct XorShift(pub u64);

impl XorShift {
    pub fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// Uniform in `[-amp, amp]`.
    pub fn uniform(&mut self, amp: i32) -> i32 {
        if amp <= 0 {
            return 0;
        }
        (self.next() % (2 * amp as u64 + 1)) as i32 - amp
    }
}

/// `sin(x)` from the basic operations only: reduce to [-pi, pi], then to
/// [-pi/2, pi/2], then a Taylor polynomial of degree 17 (error below 1e-11).
pub fn det_sin(x: f64) -> f64 {
    let mut r = x - (x / TAU).round() * TAU;
    let half_pi = TAU / 4.0;
    if r > half_pi {
        r = TAU / 2.0 - r;
    } else if r < -half_pi {
        r = -TAU / 2.0 - r;
    }
    let r2 = r * r;
    let mut term = r;
    let mut sum = r;
    for k in 1..=8 {
        term = -term * r2 / ((2 * k) as f64 * (2 * k + 1) as f64);
        sum += term;
    }
    sum
}

/// FNV-1a, 64 bits.
pub fn fnv1a64(parts: &[&[u8]]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for part in parts {
        for &b in *part {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

/// The little-endian bytes of 16-bit samples, as in a `.sw` file.
pub fn le_bytes(s: &[i16]) -> Vec<u8> {
    s.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// Round to 16 bits with saturation.
pub fn to_i16(v: f64) -> i16 {
    v.round().clamp(-32768.0, 32767.0) as i16
}

/// A voiced sound: harmonics of `f0` up to `max_hz` with amplitudes
/// falling as 1/k, under a syllable-like envelope of `env_hz`. `t` in
/// seconds. Peak about `gain`.
pub fn voice(t: f64, f0: f64, max_hz: f64, env_hz: f64, gain: f64) -> f64 {
    let env = 0.55 + 0.45 * det_sin(TAU * env_hz * t);
    let mut v = 0.0;
    let mut k = 1;
    while k as f64 * f0 <= max_hz {
        v += det_sin(TAU * k as f64 * f0 * t + 0.3 * k as f64) / k as f64;
        k += 1;
    }
    gain * env * v / 2.5
}

/// The two channels of the standard test sound at time `t`: two voices with
/// content up to 3.5 kHz, so that every rate holds all of it.
pub fn stereo_voice(t: f64, gain: f64) -> (f64, f64) {
    let l = voice(t, 180.0, 3500.0, 3.0, gain);
    let r = voice(t, 247.0, 3500.0, 4.3, gain * 0.7) + 0.3 * l;
    (l, r)
}

/// Render `frames` frames at `rate` of a stereo sound, with `channels` 1
/// (the mean of both channels) or 2.
pub fn render(rate: u32, frames: usize, channels: usize, sound: &dyn Fn(f64) -> (f64, f64)) -> Vec<i16> {
    let mut out = Vec::with_capacity(frames * channels);
    for i in 0..frames {
        let (l, r) = sound(i as f64 / rate as f64);
        if channels == 2 {
            out.push(to_i16(l * 32767.0));
            out.push(to_i16(r * 32767.0));
        } else {
            out.push(to_i16((l + r) * 0.5 * 32767.0));
        }
    }
    out
}

/// Add uniform noise of amplitude `amp` with a fixed seed.
pub fn add_noise(s: &[i16], amp: i32, seed: u64) -> Vec<i16> {
    let mut rng = XorShift(seed);
    s.iter().map(|&v| (v as i32 + rng.uniform(amp)).clamp(-32768, 32767) as i16).collect()
}

/// One input of `opus_compare`.
pub struct CompareCase {
    pub name: &'static str,
    /// 48 kHz stereo (the C program's first file).
    pub reference: Vec<i16>,
    /// `channels` interleaved channels at `rate` (the second file).
    pub test: Vec<i16>,
    pub channels: usize,
    pub rate: u32,
}

impl CompareCase {
    /// FNV-1a of the inputs, to notice when the generator changes.
    pub fn hash(&self) -> u64 {
        fnv1a64(&[
            &self.rate.to_le_bytes(),
            &[self.channels as u8],
            &le_bytes(&self.reference),
            &le_bytes(&self.test),
        ])
    }
}

/// About 40 comparisons: every rate in mono and stereo, identical signals,
/// silence, noise at several levels, a low-passed and a shifted signal,
/// clipping, one frame only, the weighting near 8 kHz, the bins left out at
/// lower rates, and the error paths of the C program.
pub fn compare_cases() -> Vec<CompareCase> {
    let mut cases = Vec::new();
    let voice_sound = |t: f64| stereo_voice(t, 0.5);
    // 0.5 s.
    let frames48 = 24000;
    let reference = render(48000, frames48, 2, &voice_sound);
    let mut push = |name, reference: &Vec<i16>, test: Vec<i16>, channels, rate| {
        cases.push(CompareCase { name, reference: reference.clone(), test, channels, rate });
    };

    // The same sound at each rate, with a little noise (it passes), and
    // with more noise. The sound has nothing above 3.5 kHz, so noise there
    // weighs heavily.
    let names_40: [[&'static str; 2]; 5] = [
        ["voice_n40_48k_m", "voice_n40_48k_s"],
        ["voice_n40_24k_m", "voice_n40_24k_s"],
        ["voice_n40_16k_m", "voice_n40_16k_s"],
        ["voice_n40_12k_m", "voice_n40_12k_s"],
        ["voice_n40_8k_m", "voice_n40_8k_s"],
    ];
    for (r, &rate) in [48000u32, 24000, 16000, 12000, 8000].iter().enumerate() {
        for ch in [1, 2] {
            let n = frames48 * rate as usize / 48000;
            let clean = render(rate, n, ch, &voice_sound);
            push(names_40[r][ch - 1], &reference, add_noise(&clean, 3, 11 + r as u64 * 2 + ch as u64), ch, rate);
        }
    }
    for (name, rate, ch, amp) in [
        ("voice_n30_24k_m", 24000u32, 1usize, 12),
        ("voice_n30_24k_s", 24000, 2, 12),
        ("voice_n30_12k_s", 12000, 2, 12),
        ("voice_n30_8k_s", 8000, 2, 12),
        ("voice_n20_48k_m", 48000, 1, 150),
        ("voice_n20_48k_s", 48000, 2, 150),
        ("voice_n20_16k_m", 16000, 1, 150),
        ("voice_n20_16k_s", 16000, 2, 150),
    ] {
        let n = frames48 * rate as usize / 48000;
        let clean = render(rate, n, ch, &voice_sound);
        push(name, &reference, add_noise(&clean, amp, rate as u64 + ch as u64), ch, rate);
    }

    // Identical signals: error 0, quality 100. The mono case needs equal
    // channels, so that the mix is the channel itself.
    push("identical_48k_s", &reference, reference.clone(), 2, 48000);
    let mono_ref = render(48000, frames48, 2, &|t| {
        let (l, _) = voice_sound(t);
        (l, l)
    });
    let mono_test: Vec<i16> = mono_ref.iter().step_by(2).copied().collect();
    push("identical_48k_m", &mono_ref, mono_test, 1, 48000);

    // Silence against silence.
    push("silence_48k_s", &vec![0; 9600], vec![0; 9600], 2, 48000);
    push("silence_16k_m", &vec![0; 9600], vec![0; 1600], 1, 16000);

    // A reference with noise over the whole band, against a low-passed copy
    // (mean of 8 samples).
    let noisy = add_noise(&reference, 2000, 5);
    let lowpassed: Vec<i16> = (0..noisy.len())
        .map(|i| {
            let ch = i % 2;
            let frame = i / 2;
            let sum: i32 = (0..8).map(|k| noisy[2 * frame.saturating_sub(k) + ch] as i32).sum();
            (sum / 8) as i16
        })
        .collect();
    push("lowpass_48k_s", &noisy, lowpassed, 2, 48000);

    // Shifted by 1 ms.
    let shifted_m: Vec<i16> = {
        let clean = render(48000, frames48, 1, &voice_sound);
        let mut s = vec![0i16; 48];
        s.extend_from_slice(&clean[..frames48 - 48]);
        s
    };
    push("shift_48k_m", &reference, shifted_m, 1, 48000);
    let shifted_s: Vec<i16> = {
        let clean = render(16000, frames48 / 3, 2, &voice_sound);
        let mut s = vec![0i16; 32];
        s.extend_from_slice(&clean[..clean.len() - 32]);
        s
    };
    push("shift_16k_s", &reference, shifted_s, 2, 16000);

    // Full scale: a clipped loud reference against the unclipped shape at
    // full scale.
    let loud = render(48000, frames48, 2, &|t| stereo_voice(t, 1.8));
    let full = render(48000, frames48, 2, &|t| stereo_voice(t, 1.0));
    push("clip_48k_s", &loud, full, 2, 48000);

    // Exactly one window.
    let one = reference[..960].to_vec();
    push("one_frame_48k_s", &one, add_noise(&one, 100, 3), 2, 48000);
    let one8 = render(8000, 80, 1, &voice_sound);
    push("one_frame_8k_m", &one, one8, 1, 8000);

    // A tone near 8 kHz, where bins 79 to 81 weigh less.
    let tone8k = |rate: u32, ch: usize, n: usize| -> Vec<i16> {
        render(rate, n, ch, &|t| {
            let (l, r) = voice_sound(t);
            let tone = 0.05 * det_sin(TAU * 8000.0 * t);
            (l + tone, r + tone)
        })
    };
    push("tone8k_48k_m", &reference, tone8k(48000, 1, frames48), 1, 48000);
    push("tone8k_24k_s", &reference, tone8k(24000, 2, frames48 / 2), 2, 24000);

    // Tones in the bins a lower rate leaves out: 5.75 kHz at 12 kHz (bin
    // 57) and 7.85 kHz at 16 kHz (bin 78).
    let with_tone = |rate: u32, ch: usize, hz: f64| -> Vec<i16> {
        render(rate, frames48 * rate as usize / 48000, ch, &|t| {
            let (l, r) = voice_sound(t);
            let tone = 0.1 * det_sin(TAU * hz * t);
            (l + tone, r + tone)
        })
    };
    push("top_bins_12k_m", &reference, with_tone(12000, 1, 5750.0), 1, 12000);
    push("top_bins_16k_s", &reference, with_tone(16000, 2, 7850.0), 2, 16000);

    // 1 s of chords up to 12 kHz with a little noise.
    let chords = |t: f64| {
        let a = voice(t, 110.0, 12000.0, 0.7, 0.4) + voice(t, 164.8, 12000.0, 1.1, 0.3);
        let b = voice(t, 138.6, 12000.0, 0.9, 0.35) + voice(t, 220.0, 12000.0, 1.3, 0.25);
        (a, b)
    };
    let music = render(48000, 48000, 2, &chords);
    push("music_1s_48k_s", &music, add_noise(&music, 30, 9), 2, 48000);

    // Independent full-scale noise.
    let mut rng = XorShift(77);
    let noise_a: Vec<i16> = (0..19200).map(|_| rng.uniform(32767) as i16).collect();
    let noise_b: Vec<i16> = (0..19200).map(|_| rng.uniform(32767) as i16).collect();
    push("noise_full_48k_s", &noise_a, noise_b, 2, 48000);

    // An odd sample count: the last half frame of the reference is not read.
    let mut odd = reference[..4801].to_vec();
    odd[4800] = 12345;
    push("partial_frame_48k_s", &odd, add_noise(&reference[..4800], 50, 4), 2, 48000);

    // The error paths of the C program.
    push("bad_rate_44100", &reference, reference.clone(), 2, 44100);
    push("mismatch_48k_s", &reference, reference[..reference.len() - 2].to_vec(), 2, 48000);
    let r16 = render(16000, frames48 / 3, 1, &voice_sound);
    push("mismatch_16k_m", &reference, r16[..r16.len() - 1].to_vec(), 1, 16000);
    push("too_short_48k_s", &reference[..958].to_vec(), reference[..958].to_vec(), 2, 48000);
    push("too_short_8k_m", &reference[..948].to_vec(), r16[..79].to_vec(), 1, 8000);
    cases
}

/// One input of MOC: 16 kHz mono.
pub struct MocCase {
    pub name: &'static str,
    pub reference: Vec<i16>,
    pub test: Vec<i16>,
}

impl MocCase {
    pub fn hash(&self) -> u64 {
        fnv1a64(&[&le_bytes(&self.reference), &le_bytes(&self.test)])
    }
}

/// About 20 MOC inputs: noise at several levels, a low-passed and a
/// shifted signal, identity, pauses (for the activity detection), different
/// lengths, and too short inputs.
pub fn moc_cases() -> Vec<MocCase> {
    let rate = 16000u32;
    let speechy = |t: f64| {
        // A voice whose pitch steps every 0.25 s, with pauses of 0.2 s
        // every second.
        let f0 = [150.0, 190.0, 170.0, 210.0][((t * 4.0) as usize) % 4];
        let pause = (t % 1.0) > 0.8;
        if pause { (0.0, 0.0) } else { (voice(t, f0, 7800.0, 3.0, 0.45), 0.0) }
    };
    let n = 3 * rate as usize;
    let clean: Vec<i16> = render(rate, n, 2, &speechy).iter().step_by(2).copied().collect();
    let mut cases = Vec::new();
    let mut push = |name, reference: &Vec<i16>, test: Vec<i16>| {
        cases.push(MocCase { name, reference: reference.clone(), test });
    };
    for (name, amp) in [("noise_a20", 20), ("noise_a100", 100), ("noise_a400", 400), ("noise_a1600", 1600), ("noise_a6000", 6000)] {
        push(name, &clean, add_noise(&clean, amp, amp as u64));
    }
    push("identical", &clean, clean.clone());
    let lowpass: Vec<i16> =
        (0..clean.len()).map(|i| ((0..4).map(|k| clean[i.saturating_sub(k)] as i32).sum::<i32>() / 4) as i16).collect();
    push("lowpass_4tap", &clean, lowpass);
    let mut shifted = vec![0i16; 16];
    shifted.extend_from_slice(&clean[..clean.len() - 16]);
    push("shift_1ms", &clean, shifted);
    let quiet: Vec<i16> = clean.iter().map(|&v| v / 8).collect();
    push("gain_minus18db", &clean, quiet);
    // The test is longer than the reference: both are cut.
    let mut longer = add_noise(&clean, 200, 3);
    longer.extend_from_slice(&[1000; 777]);
    push("longer_test", &clean, longer);
    // Short and odd lengths.
    push("len_280", &clean[..280].to_vec(), add_noise(&clean[..280], 300, 6));
    push("len_1600", &clean[1000..2600].to_vec(), add_noise(&clean[1000..2600], 300, 7));
    push("len_4321", &clean[..4321].to_vec(), add_noise(&clean[..4321], 300, 8));
    // A reference with long silence, so that the activity detection drops
    // frames; noise only where the reference is silent.
    let mut gated = clean.clone();
    for v in gated[16000..32000].iter_mut() {
        *v = 0;
    }
    let mut noisy_gap = gated.clone();
    let mut rng = XorShift(12);
    for v in noisy_gap[16000..32000].iter_mut() {
        *v = rng.uniform(300) as i16;
    }
    push("silence_gap_noise", &gated, noisy_gap);
    // Very quiet everywhere but one burst.
    let mut burst = vec![0i16; 16000];
    burst[8000..8800].copy_from_slice(&clean[2000..2800]);
    push("burst", &burst, add_noise(&burst, 30, 13));
    // Independent noise.
    let mut rng = XorShift(31);
    let a: Vec<i16> = (0..8000).map(|_| rng.uniform(10000) as i16).collect();
    let b: Vec<i16> = (0..8000).map(|_| rng.uniform(10000) as i16).collect();
    push("independent_noise", &a, b);
    // Too short.
    push("len_279", &clean[..279].to_vec(), clean[..279].to_vec());
    push("len_1599", &clean[..1599].to_vec(), add_noise(&clean[..1599], 300, 9));
    cases
}
