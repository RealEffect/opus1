//! `silk_resampler_down2_3` and `silk_resampler_private_ar2` against the
//! same functions inside C libopus (silk/resampler_down2_3.c,
//! silk/resampler_private_AR2.c).
//!
//! The SILK encoder uses `silk_resampler_down2_3` once per frame at 12 kHz
//! (mediumband): the pitch search works on the signal at 8 kHz. Both
//! functions are fixed-point, so the outputs and the filter states must be
//! identical.

use opus_rs::silk::resampler::{
    SILK_RESAMPLER_2_3_COEFS_LQ, silk_resampler_down2_3, silk_resampler_private_ar2,
};

unsafe extern "C" {
    #[link_name = "silk_resampler_down2_3"]
    fn c_silk_resampler_down2_3(s: *mut i32, out: *mut i16, input: *const i16, in_len: i32);
    fn silk_resampler_private_AR2(
        s: *mut i32,
        out_q8: *mut i32,
        input: *const i16,
        a_q14: *const i16,
        len: i32,
    );
}

/// Makes sure the test binary links libopus: the functions above are found
/// in the static library that `audiopus_sys` brings along.
fn link_libopus() {
    let version = unsafe { audiopus_sys::opus_get_version_string() };
    assert!(!version.is_null());
}

/// Small deterministic pseudo-random generator.
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
    fn range(&mut self, lo: i32, hi: i32) -> i32 {
        lo + (self.next() as u64 % (hi as i64 - lo as i64 + 1) as u64) as i32
    }
}

/// Test signals. `t` is the sample position in the stream.
#[derive(Clone, Copy, Debug)]
enum Signal {
    /// Noise at about a quarter of full scale.
    Noise,
    /// Noise in which a third of the samples sit at a limit of 16 bits.
    NoiseAtLimits,
    /// A sine of 200 Hz at 12 kHz, as a voiced sound would have it.
    Sine,
    /// A square wave between the limits of 16 bits with this period.
    Square(usize),
    /// A constant at the negative limit.
    Constant,
}

impl Signal {
    fn sample(self, rng: &mut Lcg, t: usize) -> i16 {
        match self {
            Signal::Noise => rng.range(-8000, 8000) as i16,
            Signal::NoiseAtLimits => match rng.next() % 6 {
                0 => i16::MAX,
                1 => i16::MIN,
                _ => rng.next() as i16,
            },
            Signal::Sine => {
                (12000.0 * (2.0 * std::f64::consts::PI * 200.0 * t as f64 / 12000.0).sin()) as i16
            }
            Signal::Square(period) => {
                if (t / (period / 2).max(1)).is_multiple_of(2) { i16::MAX } else { i16::MIN }
            }
            Signal::Constant => i16::MIN,
        }
    }
}

const SIGNALS: [Signal; 7] = [
    Signal::Noise,
    Signal::NoiseAtLimits,
    Signal::Sine,
    Signal::Square(2),
    Signal::Square(6),
    Signal::Square(37),
    Signal::Constant,
];

/// One stream through both down-samplers in calls of varying length. The
/// states are carried from call to call on both sides.
#[test]
fn down2_3_matches_libopus() {
    link_libopus();
    let mut rng = Lcg(31);
    let mut samples = 0;
    for signal in SIGNALS {
        let mut rs_state = [0i32; 6];
        let mut c_state = [0i32; 6];
        let mut t = 0;
        // Call lengths from nothing to more than two batches of 480, and
        // the 20 ms frame of the pitch analysis (240 samples at 12 kHz).
        for len in [0usize, 1, 2, 3, 4, 5, 7, 12, 240, 240, 241, 479, 480, 481, 483, 960, 1203, 240] {
            let input: Vec<i16> = (0..len).map(|k| signal.sample(&mut rng, t + k)).collect();
            t += len;
            // libopus writes floor(2 * len / 3) samples per batch of 480.
            let cap = 2 * len / 3 + 2;
            let mut rs_out = vec![0x5555i16; cap];
            let mut c_out = vec![0x5555i16; cap];
            silk_resampler_down2_3(&mut rs_state, &mut rs_out, &input, len as i32);
            // SAFETY: the state has 6 words, the input `len` samples, and
            // the output room for the at most `2 * len / 3` samples that
            // libopus writes.
            unsafe {
                c_silk_resampler_down2_3(c_state.as_mut_ptr(), c_out.as_mut_ptr(), input.as_ptr(), len as i32);
            }
            assert_eq!(rs_out, c_out, "{signal:?}: output of a call of {len} samples at {t}");
            assert_eq!(rs_state, c_state, "{signal:?}: state after a call of {len} samples at {t}");
            samples += len;
        }
    }
    assert!(samples > 30_000);
}

#[test]
fn ar2_matches_libopus() {
    link_libopus();
    let mut rng = Lcg(32);
    for signal in SIGNALS {
        let mut rs_state = [0i32; 2];
        let mut c_state = [0i32; 2];
        let mut t = 0;
        for len in [0usize, 1, 2, 3, 7, 80, 240, 480, 481] {
            let input: Vec<i16> = (0..len).map(|k| signal.sample(&mut rng, t + k)).collect();
            t += len;
            let mut rs_out = vec![0x5555_5555i32; len + 1];
            let mut c_out = vec![0x5555_5555i32; len + 1];
            silk_resampler_private_ar2(
                &mut rs_state,
                &mut rs_out,
                &input,
                &SILK_RESAMPLER_2_3_COEFS_LQ,
                len as i32,
            );
            // SAFETY: the state has 2 words, input and output `len`
            // elements, the coefficient table 6.
            unsafe {
                silk_resampler_private_AR2(
                    c_state.as_mut_ptr(),
                    c_out.as_mut_ptr(),
                    input.as_ptr(),
                    SILK_RESAMPLER_2_3_COEFS_LQ.as_ptr(),
                    len as i32,
                );
            }
            assert_eq!(rs_out, c_out, "{signal:?}: output of a call of {len} samples");
            assert_eq!(rs_state, c_state, "{signal:?}: state after a call of {len} samples");
        }
    }
}
