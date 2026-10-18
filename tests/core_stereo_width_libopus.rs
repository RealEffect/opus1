//! The stereo width estimate of the encoder (libopus
//! `compute_stereo_width`, src/opus_encoder.c) against the C function,
//! which the static libopus exports. The width moves the bitrate at which
//! the encoder of stereo input changes from SILK to CELT-only.
//!
//! The width and the whole state must be libopus's bit for bit after every
//! frame. The C build rounds the way clang does on 64-bit ARM, so the test
//! is exact there and only prints elsewhere.

mod k24_common;

use opus1::internals::stereo_width::{State, compute};

const EXACT: bool = cfg!(target_arch = "aarch64");

unsafe extern "C" {
    fn compute_stereo_width(pcm: *const f32, frame_size: i32, fs: i32, mem: *mut State) -> f32;
}

/// A small random number generator (64-bit linear congruential).
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u64 << 23) as f32) - 1.0
    }
}

/// Stereo test signals of `seconds` at `fs`, interleaved.
fn signals(fs: usize, seconds: f64) -> Vec<(&'static str, Vec<f32>)> {
    let n = (seconds * fs as f64) as usize;
    let speech = k24_common::speech_fullband(fs, seconds);
    let music = k24_common::multitone(fs, n);
    let mut rng = Lcg(5);
    let noise: Vec<f32> = (0..2 * n).map(|_| 0.3 * rng.next()).collect();
    let quiet: Vec<f32> = (0..2 * n).map(|_| 1e-3 * rng.next()).collect();
    // The right channel nearly silent: the product of the two powers is
    // tiny, and the 1e-15 of the correlation matters.
    let faint: Vec<f32> = speech.iter().flat_map(|a| [*a, 1e-6 * *a]).collect();
    vec![
        ("speech, music", speech.iter().zip(&music).flat_map(|(a, b)| [*a, *b]).collect()),
        ("delayed copy", k24_common::to_stereo(&music, fs)),
        ("same in both", speech.iter().flat_map(|a| [*a, *a]).collect()),
        ("noise", noise),
        ("quiet noise", quiet),
        ("opposite", music.iter().flat_map(|a| [*a, -0.7 * *a]).collect()),
        ("left only", speech.iter().flat_map(|a| [*a, 0.0]).collect()),
        ("right faint", faint),
    ]
}

#[test]
fn stereo_width_is_libopus() {
    let (mut frames, mut failures) = (0, Vec::new());
    for fs in [8_000usize, 12_000, 16_000, 24_000, 48_000] {
        for (name, pcm) in signals(fs, 3.0) {
            for frame_ms_x2 in [5, 10, 20, 40, 80, 120] {
                let frame = fs * frame_ms_x2 / 2000;
                let (mut rs, mut c) = (State::default(), State::default());
                for (k, f) in pcm.chunks_exact(2 * frame).enumerate() {
                    let got = compute(f, frame, fs as i32, &mut rs);
                    // SAFETY: `f` holds `frame` stereo samples and `c` is a
                    // `StereoWidthState` with the C layout.
                    let want = unsafe { compute_stereo_width(f.as_ptr(), frame as i32, fs as i32, &mut c) };
                    frames += 1;
                    if got.to_bits() != want.to_bits() || rs != c {
                        failures.push(format!(
                            "{fs} Hz {name}, {frame} samples, frame {k}: width {got:e} against {want:e}, state {rs:?} against {c:?}"
                        ));
                        break;
                    }
                }
            }
        }
    }
    println!("{frames} frames compared");
    for f in failures.iter().take(10) {
        println!("  {f}");
    }
    if EXACT {
        assert!(failures.is_empty(), "{} streams differ from libopus", failures.len());
    }
}
