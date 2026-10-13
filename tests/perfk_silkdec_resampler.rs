//! The SILK resampler against the same functions inside C libopus.
//!
//! libopus is linked as a static library, so its internal functions can be
//! called directly. The resampler is fixed-point: every test requires
//! bit-identical output.
//!
//! `tests/k24_silk_internals.rs` already compares every rate pair in 10 ms
//! and 20 ms calls. The tests here add what the rewritten loops need: call
//! lengths from 1 ms to 60 ms in one stream (the kernels work in batches of
//! 10 ms and compute four output samples per step, so every batch and tail
//! length occurs), and signals at the limits of 16 bits, where the filters
//! saturate.

use opus_rs::silk::resampler::{
    SilkResampler, silk_resampler_down2, silk_resampler_private_up2_hq,
};
use std::os::raw::c_int;

unsafe extern "C" {
    fn silk_resampler_init(state: *mut u8, fs_in: i32, fs_out: i32, for_enc: c_int) -> c_int;
    fn silk_resampler(state: *mut u8, out: *mut i16, input: *const i16, in_len: i32) -> c_int;
    #[link_name = "silk_resampler_down2"]
    fn c_silk_resampler_down2(s: *mut i32, out: *mut i16, input: *const i16, in_len: i32);
    fn silk_resampler_private_up2_HQ(s: *mut i32, out: *mut i16, input: *const i16, len: i32);
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
    /// A square wave between the limits of 16 bits with this period in
    /// samples; the short periods are near the resonances of the filters.
    Square(usize),
    /// One full-scale sample every 97 samples.
    Impulses,
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
            Signal::Square(period) => {
                if (t / (period / 2).max(1)) % 2 == 0 { i16::MAX } else { i16::MIN }
            }
            Signal::Impulses => {
                if t % 97 == 0 { i16::MAX } else { 0 }
            }
            Signal::Constant => i16::MIN,
        }
    }
}

const SIGNALS: [Signal; 8] = [
    Signal::Noise,
    Signal::NoiseAtLimits,
    Signal::Square(2),
    Signal::Square(4),
    Signal::Square(6),
    Signal::Square(64),
    Signal::Impulses,
    Signal::Constant,
];

/// Every rate pair of the decoder direction and of the encoder direction.
fn rate_pairs() -> Vec<(i32, i32, bool)> {
    let mut pairs = Vec::new();
    for fs_in in [8000, 12000, 16000] {
        for fs_out in [8000, 12000, 16000, 24000, 48000] {
            pairs.push((fs_in, fs_out, false));
        }
    }
    for fs_in in [8000, 12000, 16000, 24000, 48000] {
        for fs_out in [8000, 12000, 16000] {
            pairs.push((fs_in, fs_out, true));
        }
    }
    pairs
}

#[test]
fn resampler_matches_libopus_for_every_call_length_and_signal() {
    link_libopus();
    // Call lengths in milliseconds, in one stream: the filter state carries
    // over from each call to the next.
    const CALLS_MS: [usize; 16] = [20, 1, 2, 10, 3, 5, 40, 1, 60, 7, 20, 9, 11, 30, 13, 20];
    let mut rng = Lcg(31);
    let mut compared = 0usize;
    for (fs_in, fs_out, for_enc) in rate_pairs() {
        for signal in SIGNALS {
            // libopus state: opaque, far smaller than this buffer.
            let mut c_state = vec![0u8; 4096];
            let c_ret = unsafe {
                silk_resampler_init(c_state.as_mut_ptr(), fs_in, fs_out, for_enc as c_int)
            };
            assert_eq!(c_ret, 0);
            let mut rs = SilkResampler::default();
            let ret = if for_enc { rs.init_for_enc(fs_in, fs_out) } else { rs.init(fs_in, fs_out) };
            assert_eq!(ret, 0, "init {fs_in} -> {fs_out} for_enc={for_enc}");

            let mut t = 0usize;
            for (call, ms) in CALLS_MS.into_iter().enumerate() {
                let n_in = fs_in as usize / 1000 * ms;
                let n_out = fs_out as usize / 1000 * ms;
                let input: Vec<i16> = (0..n_in).map(|i| signal.sample(&mut rng, t + i)).collect();
                t += n_in;

                let mut c_out = vec![0i16; n_out];
                unsafe {
                    silk_resampler(
                        c_state.as_mut_ptr(),
                        c_out.as_mut_ptr(),
                        input.as_ptr(),
                        n_in as i32,
                    )
                };
                let mut rs_out = vec![0i16; n_out];
                assert_eq!(rs.process(&mut rs_out, &input, n_in as i32), 0);
                assert_eq!(
                    rs_out, c_out,
                    "{fs_in} -> {fs_out}, for_enc={for_enc}, {signal:?}, call {call} ({ms} ms)"
                );
                compared += n_out;
            }
        }
    }
    println!("{compared} output samples identical");
}

/// An output slice that is too short gets the first samples of the full
/// output and nothing is written behind it.
#[test]
fn resampler_with_a_short_output_writes_a_prefix() {
    let mut rng = Lcg(32);
    for (fs_in, fs_out, for_enc) in rate_pairs() {
        let init = |rs: &mut SilkResampler| {
            if for_enc { rs.init_for_enc(fs_in, fs_out) } else { rs.init(fs_in, fs_out) }
        };
        let n_in = fs_in as usize / 50;
        let n_out = fs_out as usize / 50;
        let input: Vec<i16> = (0..n_in).map(|_| rng.range(-20000, 20000) as i16).collect();

        let mut full = SilkResampler::default();
        assert_eq!(init(&mut full), 0);
        let mut expect = vec![0i16; n_out];
        full.process(&mut expect, &input, n_in as i32);

        for short in [n_out - 1, n_out - 3, n_out / 2 + 1, n_out / 3] {
            let mut rs = SilkResampler::default();
            assert_eq!(init(&mut rs), 0);
            let mut out = vec![0x5555i16; n_out];
            rs.process(&mut out[..short], &input, n_in as i32);
            // The 2x up-sampler writes whole pairs only.
            let written = if fs_out == 2 * fs_in { short & !1 } else { short };
            assert_eq!(
                out[..written],
                expect[..written],
                "{fs_in} -> {fs_out}, for_enc={for_enc}, {short} of {n_out} samples"
            );
            assert!(out[short..].iter().all(|&v| v == 0x5555));
        }
    }
}

#[test]
fn up2_hq_matches_libopus() {
    link_libopus();
    let mut rng = Lcg(33);
    for signal in SIGNALS {
        let mut c_state = [0i32; 6];
        let mut rs_state = [0i32; 6];
        let mut t = 0usize;
        // Every length from 0 to 40 (both parities, every tail), then long ones.
        for len in (0..=40).chain([79, 80, 81, 160, 240, 479, 480]) {
            let input: Vec<i16> = (0..len).map(|i| signal.sample(&mut rng, t + i)).collect();
            t += len;
            let mut c_out = vec![0i16; 2 * len];
            unsafe {
                silk_resampler_private_up2_HQ(
                    c_state.as_mut_ptr(),
                    c_out.as_mut_ptr(),
                    input.as_ptr(),
                    len as i32,
                )
            };
            let mut rs_out = vec![0i16; 2 * len];
            silk_resampler_private_up2_hq(&mut rs_state, &mut rs_out, &input, len as i32);
            assert_eq!(rs_out, c_out, "{signal:?}, {len} samples");
            assert_eq!(rs_state, c_state, "state after {signal:?}, {len} samples");
        }
    }
}

#[test]
fn down2_matches_libopus() {
    link_libopus();
    let mut rng = Lcg(34);
    for signal in SIGNALS {
        let mut c_state = [0i32; 2];
        let mut rs_state = [0i32; 2];
        let mut t = 0usize;
        for len in (0..=40).chain([79, 80, 81, 160, 320, 479, 480]) {
            let input: Vec<i16> = (0..len).map(|i| signal.sample(&mut rng, t + i)).collect();
            t += len;
            let mut c_out = vec![0i16; len / 2];
            unsafe {
                c_silk_resampler_down2(
                    c_state.as_mut_ptr(),
                    c_out.as_mut_ptr(),
                    input.as_ptr(),
                    len as i32,
                )
            };
            let mut rs_out = vec![0i16; len / 2];
            silk_resampler_down2(&mut rs_state, &mut rs_out, &input, len as i32);
            assert_eq!(rs_out, c_out, "{signal:?}, {len} samples");
            assert_eq!(rs_state, c_state, "state after {signal:?}, {len} samples");
        }
    }
}
