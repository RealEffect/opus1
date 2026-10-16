//! `hp_cutoff` rounds its float input to 16 bits and filters it. These tests
//! require that the result equals the 16-bit entry point `hp_cutoff_i16` on
//! the rounded samples, for every length.

use opus1::hp_cutoff::{hp_cutoff, hp_cutoff_i16};

/// xorshift64 generator with a fixed seed, so the test data never changes.
struct XorShift(u64);

impl XorShift {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    /// Uniform value in [-1, 1).
    fn uniform(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32 / (1u64 << 23) as f32) - 1.0
    }
}

/// The conversion `hp_cutoff` applies: round half up, limit to 16 bits.
fn to_i16(x: f32) -> i16 {
    (x * 32768.0 + 0.5).floor().clamp(-32768.0, 32767.0) as i16
}

/// Run both entry points over `calls` consecutive frames of `frames` sample
/// frames each and require equal output and equal filter state after every
/// call.
fn check(
    rng: &mut XorShift,
    frames: usize,
    channels: usize,
    fs: i32,
    cutoff_hz: i32,
    gain: f32,
    calls: usize,
) {
    let n = frames * channels;
    let mut mem_f32 = [0i32; 4];
    let mut mem_i16 = [0i32; 4];
    for call in 0..calls {
        let input: Vec<f32> = (0..n).map(|_| rng.uniform() * gain).collect();
        let input_i16: Vec<i16> = input.iter().map(|&x| to_i16(x)).collect();
        // A marker value shows that nothing past the frame is written.
        let mut out_f32 = vec![0x5a5ai16; n + 3];
        let mut out_i16 = vec![0x5a5ai16; n + 3];
        hp_cutoff(
            &input,
            cutoff_hz,
            &mut out_f32[..n],
            &mut mem_f32,
            frames,
            channels,
            fs,
        );
        hp_cutoff_i16(
            &input_i16,
            cutoff_hz,
            &mut out_i16[..n],
            &mut mem_i16,
            frames,
            channels,
            fs,
        );
        assert_eq!(
            out_f32, out_i16,
            "output: frames {frames}, channels {channels}, fs {fs}, call {call}"
        );
        assert_eq!(
            mem_f32, mem_i16,
            "state: frames {frames}, channels {channels}, fs {fs}, call {call}"
        );
        assert_eq!(&out_f32[n..], &[0x5a5a; 3]);
    }
}

#[test]
fn block_wise_equals_single_pass_for_every_length() {
    let mut rng = XorShift(0x9E37_79B9_7F4A_7C15);
    // Every length around the block size of 480 samples and its multiples,
    // and the frame sizes the encoder uses (2.5 ms to 120 ms at 8 to 48 kHz).
    let mut lengths: Vec<usize> = (0..=40).collect();
    for centre in [240usize, 480, 960, 1440, 1920] {
        lengths.extend(centre - 3..=centre + 3);
    }
    lengths.extend([20, 40, 80, 120, 160, 320, 640, 2880, 5760]);
    for &frames in &lengths {
        for channels in [1usize, 2] {
            for (fs, cutoff_hz) in [(8000, 60), (16000, 80), (48000, 100)] {
                check(&mut rng, frames, channels, fs, cutoff_hz, 0.5, 3);
            }
        }
    }
}

#[test]
fn block_wise_equals_single_pass_for_extreme_input() {
    let mut rng = XorShift(0x0123_4567_89AB_CDEF);
    // Full scale and far beyond it: the conversion limits to 16 bits and the
    // filter output saturates.
    for gain in [1.0f32, 1.5, 8.0, 1.0e6, 1.0e-6] {
        for channels in [1usize, 2] {
            check(&mut rng, 961, channels, 48000, 60, gain, 4);
            check(&mut rng, 160, channels, 8000, 100, gain, 4);
        }
    }

    // Constant extreme values, and values that are not numbers.
    for value in [
        1.0f32,
        -1.0,
        32767.0 / 32768.0,
        f32::MAX,
        f32::MIN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        0.0,
        -0.0,
        0.499_999 / 32768.0,
        0.5 / 32768.0,
        -0.5 / 32768.0,
    ] {
        for channels in [1usize, 2] {
            let frames = 1000usize;
            let n = frames * channels;
            let input = vec![value; n];
            let input_i16: Vec<i16> = input.iter().map(|&x| to_i16(x)).collect();
            let mut mem_f32 = [0i32; 4];
            let mut mem_i16 = [0i32; 4];
            let mut out_f32 = vec![0i16; n];
            let mut out_i16 = vec![0i16; n];
            hp_cutoff(
                &input,
                60,
                &mut out_f32,
                &mut mem_f32,
                frames,
                channels,
                48000,
            );
            hp_cutoff_i16(
                &input_i16,
                60,
                &mut out_i16,
                &mut mem_i16,
                frames,
                channels,
                48000,
            );
            assert_eq!(out_f32, out_i16, "value {value}, channels {channels}");
            assert_eq!(mem_f32, mem_i16, "value {value}, channels {channels}");
        }
    }
}

#[test]
fn mono_stops_at_the_shorter_of_input_and_output() {
    let mut rng = XorShift(0xDEAD_BEEF_CAFE_F00D);
    let input: Vec<f32> = (0..1000).map(|_| rng.uniform() * 0.5).collect();
    let input_i16: Vec<i16> = input.iter().map(|&x| to_i16(x)).collect();
    for out_len in [0usize, 1, 479, 480, 481, 999, 1000, 1200] {
        let mut mem_f32 = [0i32; 2];
        let mut mem_i16 = [0i32; 2];
        let mut out_f32 = vec![0x5a5ai16; out_len];
        let mut out_i16 = vec![0x5a5ai16; out_len];
        hp_cutoff(&input, 60, &mut out_f32, &mut mem_f32, 1000, 1, 16000);
        hp_cutoff_i16(&input_i16, 60, &mut out_i16, &mut mem_i16, 1000, 1, 16000);
        assert_eq!(out_f32, out_i16, "output length {out_len}");
        assert_eq!(mem_f32, mem_i16, "output length {out_len}");
    }
}
