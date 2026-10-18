#![cfg(feature = "test-libopus-dnn")]
#![cfg(feature = "dnn")]
//! The signal processing of the neural network code (`dnn/freq.c`,
//! `dnn/burg.c`, `dnn/lpcnet_tables.c`) against the same functions inside a
//! libopus 1.6.1 built with them.
//!
//! The reference is the build without loop vectorization (plan.md
//! section 1): `scripts/android/build-libopus.sh --host --rtcd --dnn` with
//! `OPUS_EXTRA_CFLAGS="-fno-vectorize -fno-slp-vectorize"`. Every output
//! must have the same bits. Run with
//!
//! ```text
//! LIBOPUS_LIB_DIR=target/android/libopus-host-1.6.1-dnn-novec LIBOPUS_NO_PKG=1 \
//!   LIBOPUS_STATIC=1 CARGO_TARGET_DIR=target/dnn-ref \
//!   cargo test --release --features test-libopus-dnn,dnn --test dnn_freq_burg
//! ```
//!
//! The static functions of `freq.c` (`idct`, `inverse_transform`,
//! `lpc_from_bands`, `lpcn_lpc`, `compute_burg_cepstrum`, the inverse band
//! energy) cannot be called from here. `lpc_from_cepstrum` and
//! `burg_cepstral_analysis` run all of them.
//!
//! The signals are the speech of `fixtures/answer_16k.wav` and numbers from
//! fixed integer generators. No signal is made with `sin` or `cos`, whose
//! last bits differ between C libraries.

use opus1::internals::dnn::burg::silk_burg_analysis;
use opus1::internals::dnn::freq::{
    self, EBAND5MS, FRAME_SIZE, FREQ_SIZE, LPC_ORDER, NB_BANDS, WINDOW_SIZE,
};
use opus1::internals::dnn::lpcnet_tables::{
    DCT_TABLE, FFT_BITREV, FFT_TWIDDLES, HALF_WINDOW, KFFT_FACTORS, KFFT_NFFT, KFFT_SCALE,
    KFFT_SHIFT,
};
use opus1::internals::kiss_fft::KissCpx;
use std::os::raw::{c_int, c_void};

/// The C declarations, in their own module so that their names do not mix
/// with the Rust functions of the same names.
mod c {
    use super::*;

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Default)]
    pub struct Cpx {
        pub r: f32,
        pub i: f32,
    }

    /// `kiss_fft_state` of the float build (celt/kiss_fft.h).
    #[repr(C)]
    pub struct FftState {
        pub nfft: c_int,
        pub scale: f32,
        pub shift: c_int,
        pub factors: [i16; 16],
        pub bitrev: *const i16,
        pub twiddles: *const Cpx,
        pub arch_fft: *const c_void,
    }

    unsafe extern "C" {
        pub static kfft: FftState;
        pub static half_window: [f32; 160];
        pub static dct_table: [f32; 324];
        pub fn lpcn_compute_band_energy(band_e: *mut f32, x: *const Cpx);
        pub fn burg_cepstral_analysis(ceps: *mut f32, x: *const f32);
        pub fn apply_window(x: *mut f32);
        pub fn dct(out: *mut f32, input: *const f32);
        pub fn forward_transform(out: *mut Cpx, input: *const f32);
        pub fn lpc_from_cepstrum(lpc: *mut f32, cepstrum: *const f32) -> f32;
        pub fn silk_burg_analysis(
            a: *mut f32,
            x: *const f32,
            min_inv_gain: f32,
            subfr_length: c_int,
            nb_subfr: c_int,
            d: c_int,
        ) -> f32;
    }
}

/// Makes sure the test binary links libopus: the functions above are in
/// the static library that `audiopus_sys` brings along.
fn link_libopus() {
    let version = unsafe { audiopus_sys::opus_get_version_string() };
    assert!(!version.is_null());
}

/// A fixed generator of uniform numbers in [-1, 1).
struct Lcg(u64);

impl Lcg {
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 32) as u32
    }

    fn uniform(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / (1u32 << 23) as f32 - 1.0
    }
}

/// The 16-bit samples of the speech fixture (16 kHz, mono).
fn speech() -> Vec<i16> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/answer_16k.wav");
    let bytes = std::fs::read(path).expect("fixtures/answer_16k.wav");
    let data = bytes
        .windows(4)
        .position(|w| w == b"data")
        .expect("a data chunk");
    let len = u32::from_le_bytes(bytes[data + 4..data + 8].try_into().unwrap()) as usize;
    let (samples, _) = bytes[data + 8..data + 8 + len].as_chunks::<2>();
    samples.iter().map(|&b| i16::from_le_bytes(b)).collect()
}

/// A sine made without `sin`: the recursion `x[n] = 2c·x[n-1] - x[n-2]`
/// in `f64`, with `c` the cosine of the step, then rounded to `f32`.
fn sine(len: usize, two_cos: f64, amplitude: f64) -> Vec<f32> {
    let mut out = Vec::with_capacity(len);
    let (mut x1, mut x2) = (0.0f64, -amplitude * 0.3);
    for _ in 0..len {
        let x = two_cos * x1 - x2;
        x2 = x1;
        x1 = x;
        out.push(x as f32);
    }
    out
}

/// Panics with the first index where the bits differ.
#[track_caller]
fn assert_same_bits(what: &str, rust: &[f32], c: &[f32]) {
    assert_eq!(rust.len(), c.len());
    for (i, (a, b)) in rust.iter().zip(c.iter()).enumerate() {
        assert!(
            a.to_bits() == b.to_bits(),
            "{what}: value {i} differs: Rust {a:e} ({:#010x}), C {b:e} ({:#010x})",
            a.to_bits(),
            b.to_bits()
        );
    }
}

#[track_caller]
fn assert_same_cpx(what: &str, rust: &[KissCpx], c: &[c::Cpx]) {
    let r: Vec<f32> = rust.iter().flat_map(|v| [v.r, v.i]).collect();
    let cc: Vec<f32> = c.iter().flat_map(|v| [v.r, v.i]).collect();
    assert_same_bits(what, &r, &cc);
}

/// Windows of 320 samples: the speech every 160 samples at two scales
/// (16-bit values, and divided by 32768 as deep PLC does), random numbers
/// at several scales, impulses, silence and a full-scale square wave.
fn windows() -> Vec<[f32; WINDOW_SIZE]> {
    let mut out = Vec::new();
    let pcm = speech();
    for frame in pcm.windows(WINDOW_SIZE).step_by(FRAME_SIZE) {
        let mut w = [0.0f32; WINDOW_SIZE];
        let mut v = [0.0f32; WINDOW_SIZE];
        for i in 0..WINDOW_SIZE {
            w[i] = frame[i] as f32;
            v[i] = frame[i] as f32 / 32768.0;
        }
        out.push(w);
        out.push(v);
    }
    let mut rng = Lcg(1);
    for scale in [1e-6f32, 1e-3, 1.0, 32768.0, 1e6] {
        for _ in 0..200 {
            let mut w = [0.0f32; WINDOW_SIZE];
            for x in w.iter_mut() {
                *x = scale * rng.uniform();
            }
            out.push(w);
        }
    }
    for pos in [0, 1, 17, 159, 160, 319] {
        let mut w = [0.0f32; WINDOW_SIZE];
        w[pos] = 32767.0;
        out.push(w);
    }
    out.push([0.0; WINDOW_SIZE]);
    let mut square = [0.0f32; WINDOW_SIZE];
    for (n, x) in square.iter_mut().enumerate() {
        *x = if (n / 5) % 2 == 0 { 32767.0 } else { -32768.0 };
    }
    out.push(square);
    out
}

#[test]
fn tables_and_fft_plan_equal_those_of_libopus() {
    link_libopus();
    // SAFETY: `kfft`, `half_window` and `dct_table` are constant tables of
    // the library; the pointers of `kfft` point to 320 values each.
    let (st, window, table) = unsafe { (&c::kfft, &c::half_window, &c::dct_table) };
    assert_eq!(st.nfft as usize, KFFT_NFFT);
    assert_eq!(st.scale.to_bits(), KFFT_SCALE.to_bits());
    assert_eq!(st.shift, KFFT_SHIFT);
    assert_eq!(st.factors, KFFT_FACTORS);
    let bitrev = unsafe { std::slice::from_raw_parts(st.bitrev, KFFT_NFFT) };
    assert_eq!(bitrev, &FFT_BITREV[..]);
    let twiddles = unsafe { std::slice::from_raw_parts(st.twiddles, KFFT_NFFT) };
    assert_same_cpx("fft_twiddles", &FFT_TWIDDLES, twiddles);
    assert_same_bits("half_window", &HALF_WINDOW, window);
    assert_same_bits("dct_table", &DCT_TABLE, table);
}

#[test]
fn forward_transform_matches_libopus() {
    link_libopus();
    for (n, w) in windows().iter().enumerate() {
        let mut rust = [KissCpx::default(); FREQ_SIZE];
        freq::forward_transform(&mut rust, w);
        let mut c_out = [c::Cpx::default(); FREQ_SIZE];
        unsafe { c::forward_transform(c_out.as_mut_ptr(), w.as_ptr()) };
        assert_same_cpx(&format!("forward_transform, window {n}"), &rust, &c_out);
    }
}

#[test]
fn apply_window_matches_libopus() {
    link_libopus();
    for (n, w) in windows().iter().enumerate() {
        let mut rust = *w;
        freq::apply_window(&mut rust);
        let mut c_out = *w;
        unsafe { c::apply_window(c_out.as_mut_ptr()) };
        assert_same_bits(&format!("apply_window, window {n}"), &rust, &c_out);
    }
}

/// Spectra: those of the windows (after the analysis window), and random
/// complex values.
fn spectra() -> Vec<[KissCpx; FREQ_SIZE]> {
    let mut out = Vec::new();
    for mut w in windows() {
        freq::apply_window(&mut w);
        let mut spec = [KissCpx::default(); FREQ_SIZE];
        freq::forward_transform(&mut spec, &w);
        out.push(spec);
    }
    let mut rng = Lcg(2);
    for scale in [1e-3f32, 1.0, 1000.0] {
        for _ in 0..100 {
            let mut spec = [KissCpx::default(); FREQ_SIZE];
            for b in spec.iter_mut() {
                *b = KissCpx::new(scale * rng.uniform(), scale * rng.uniform());
            }
            out.push(spec);
        }
    }
    out
}

fn to_c(spec: &[KissCpx; FREQ_SIZE]) -> [c::Cpx; FREQ_SIZE] {
    let mut out = [c::Cpx::default(); FREQ_SIZE];
    for (o, v) in out.iter_mut().zip(spec.iter()) {
        *o = c::Cpx { r: v.r, i: v.i };
    }
    out
}

#[test]
fn band_energy_matches_libopus() {
    link_libopus();
    for (n, spec) in spectra().iter().enumerate() {
        let mut rust = [0.0f32; NB_BANDS];
        freq::lpcn_compute_band_energy(&mut rust, spec);
        let mut c_out = [0.0f32; NB_BANDS];
        let c_spec = to_c(spec);
        unsafe { c::lpcn_compute_band_energy(c_out.as_mut_ptr(), c_spec.as_ptr()) };
        assert_same_bits(&format!("lpcn_compute_band_energy, spectrum {n}"), &rust, &c_out);
    }
}

/// Cepstra as the feature analysis makes them (`dnn/lpcnet_enc.c:146-152`):
/// the log band energies with the same floor, through the DCT, minus 4.
/// Then random cepstra.
fn cepstra() -> Vec<[f32; NB_BANDS]> {
    let mut out = Vec::new();
    for spec in spectra() {
        let mut ex = [0.0f32; NB_BANDS];
        freq::lpcn_compute_band_energy(&mut ex, &spec);
        let mut ly = [0.0f32; NB_BANDS];
        let (mut log_max, mut follow) = (-2.0f32, -2.0f32);
        for i in 0..NB_BANDS {
            let l = (1e-2 + ex[i] as f64).log10() as f32;
            let l = l.max(log_max - 8.0).max(follow - 1.5);
            log_max = log_max.max(l);
            follow = (follow - 1.5).max(l);
            ly[i] = l;
        }
        let mut ceps = [0.0f32; NB_BANDS];
        freq::dct(&mut ceps, &ly);
        ceps[0] -= 4.0;
        out.push(ceps);
    }
    let mut rng = Lcg(3);
    for scale in [0.1f32, 1.0, 3.0] {
        for _ in 0..200 {
            let mut ceps = [0.0f32; NB_BANDS];
            for v in ceps.iter_mut() {
                *v = scale * rng.uniform();
            }
            ceps[0] += 4.0 * rng.uniform();
            out.push(ceps);
        }
    }
    out.push([0.0; NB_BANDS]);
    let mut flat = [0.0f32; NB_BANDS];
    flat[0] = -4.0;
    out.push(flat);
    out
}

#[test]
fn dct_matches_libopus() {
    link_libopus();
    let mut inputs = cepstra();
    let mut rng = Lcg(4);
    for scale in [1e-6f32, 1.0, 1e4] {
        for _ in 0..100 {
            let mut v = [0.0f32; NB_BANDS];
            for x in v.iter_mut() {
                *x = scale * rng.uniform();
            }
            inputs.push(v);
        }
    }
    for (n, input) in inputs.iter().enumerate() {
        let mut rust = [0.0f32; NB_BANDS];
        freq::dct(&mut rust, input);
        let mut c_out = [0.0f32; NB_BANDS];
        unsafe { c::dct(c_out.as_mut_ptr(), input.as_ptr()) };
        assert_same_bits(&format!("dct, input {n}"), &rust, &c_out);
    }
}

#[test]
fn lpc_from_cepstrum_matches_libopus() {
    link_libopus();
    let all = cepstra();
    for (n, ceps) in all.iter().enumerate() {
        let mut rust = [0.0f32; LPC_ORDER];
        let rust_err = freq::lpc_from_cepstrum(&mut rust, ceps);
        let mut c_out = [0.0f32; LPC_ORDER];
        let c_err = unsafe { c::lpc_from_cepstrum(c_out.as_mut_ptr(), ceps.as_ptr()) };
        let what = format!("lpc_from_cepstrum, cepstrum {n}");
        assert_same_bits(&what, &rust, &c_out);
        assert_same_bits(&what, &[rust_err], &[c_err]);
    }
    // Make sure the inputs reach the early stop of the Levinson recursion
    // (30 dB of prediction gain) and also run all 16 steps.
    let early = all
        .iter()
        .filter(|ceps| {
            let mut lpc = [0.0f32; LPC_ORDER];
            freq::lpc_from_cepstrum(&mut lpc, ceps);
            lpc[LPC_ORDER - 1] == 0.0
        })
        .count();
    assert!(early > 0 && early < all.len(), "{early} of {}", all.len());
}

/// Frames of 160 samples for `burg_cepstral_analysis`: every 10 ms of the
/// speech, divided by 32768 as deep PLC does (`dnn/lpcnet_plc.c:150`) and
/// in 16-bit values; random numbers; silence; impulses; full-scale squares;
/// sines that drive the Burg analysis to its maximum gain.
fn burg_frames() -> Vec<[f32; FRAME_SIZE]> {
    let mut out = Vec::new();
    let pcm = speech();
    for frame in pcm.as_chunks::<FRAME_SIZE>().0 {
        let mut a = [0.0f32; FRAME_SIZE];
        let mut b = [0.0f32; FRAME_SIZE];
        for i in 0..FRAME_SIZE {
            a[i] = frame[i] as f32 / 32768.0;
            b[i] = frame[i] as f32;
        }
        out.push(a);
        out.push(b);
    }
    let mut rng = Lcg(5);
    for scale in [1e-4f32, 1.0, 32768.0] {
        for _ in 0..200 {
            let mut f = [0.0f32; FRAME_SIZE];
            for x in f.iter_mut() {
                *x = scale * rng.uniform();
            }
            out.push(f);
        }
    }
    out.push([0.0; FRAME_SIZE]);
    for pos in [0, 1, 40, 79, 80, 81, 159] {
        let mut f = [0.0f32; FRAME_SIZE];
        f[pos] = 1.0;
        out.push(f);
    }
    for period in [2, 8, 33] {
        let mut f = [0.0f32; FRAME_SIZE];
        for (n, x) in f.iter_mut().enumerate() {
            *x = if (n / period) % 2 == 0 { 32767.0 } else { -32767.0 };
        }
        out.push(f);
    }
    for (two_cos, amplitude) in [(1.9, 1.0), (1.2, 30000.0), (-0.5, 0.5), (1.999, 1000.0)] {
        let s = sine(FRAME_SIZE, two_cos, amplitude);
        out.push(s.try_into().unwrap());
    }
    out
}

#[test]
fn burg_cepstral_analysis_matches_libopus() {
    link_libopus();
    for (n, frame) in burg_frames().iter().enumerate() {
        let mut rust = [0.0f32; 2 * NB_BANDS];
        freq::burg_cepstral_analysis(&mut rust, frame);
        let mut c_out = [0.0f32; 2 * NB_BANDS];
        unsafe { c::burg_cepstral_analysis(c_out.as_mut_ptr(), frame.as_ptr()) };
        assert_same_bits(&format!("burg_cepstral_analysis, frame {n}"), &rust, &c_out);
    }
}

/// `silk_burg_analysis` in the shape `compute_burg_cepstrum` uses it, and in
/// others: several subframes, every order, other gain limits.
#[test]
fn silk_burg_analysis_matches_libopus() {
    link_libopus();
    let frames = burg_frames();
    let mut checked = 0;
    let mut max_gain = 0;
    for (n, frame) in frames.iter().enumerate() {
        // Pre-emphasis as `compute_burg_cepstrum` does it, on both halves.
        let mut shapes: Vec<(Vec<f32>, f32, usize, usize, usize)> = Vec::new();
        for half in frame.as_chunks::<{ FRAME_SIZE / 2 }>().0 {
            let x: Vec<f32> = half.windows(2).map(|w| (-0.85f32).mul_add(w[0], w[1])).collect();
            shapes.push((x, 1e-3, 79, 1, LPC_ORDER));
        }
        shapes.push((frame.to_vec(), 1e-3, 40, 4, LPC_ORDER));
        shapes.push((frame.to_vec(), 1e-2, 80, 2, 10));
        shapes.push((frame.to_vec(), 0.5, 160, 1, 1 + n % 16));
        shapes.push((frame.to_vec(), 1e-4, 20, 8, 16));
        shapes.push((frame.to_vec(), 1e-3, 160, 1, 0));
        for (x, min_inv_gain, subfr_length, nb_subfr, d) in shapes {
            let mut rust = [0.0f32; 16];
            let rust_nrg =
                silk_burg_analysis(&mut rust, &x, min_inv_gain, subfr_length, nb_subfr, d);
            let mut c_out = [0.0f32; 16];
            let c_nrg = unsafe {
                c::silk_burg_analysis(
                    c_out.as_mut_ptr(),
                    x.as_ptr(),
                    min_inv_gain,
                    subfr_length as c_int,
                    nb_subfr as c_int,
                    d as c_int,
                )
            };
            let what = format!("silk_burg_analysis, frame {n}, {subfr_length} x {nb_subfr}, order {d}");
            assert_same_bits(&what, &rust[..d], &c_out[..d]);
            assert_same_bits(&what, &[rust_nrg], &[c_nrg]);
            checked += 1;
            // With the gain limit reached, the last coefficients are 0.
            if d == LPC_ORDER && rust[d - 1] == 0.0 {
                max_gain += 1;
            }
        }
    }
    assert!(max_gain > 0, "no input reached the maximum prediction gain");
    println!("{checked} analyses, {max_gain} stopped at the gain limit");
}

/// The band edges of `freq.rs` are those of libopus (`eband5ms` is static
/// in C): the band energies above would differ otherwise. This only checks
/// the shape of the table.
#[test]
fn band_edges_cover_0_to_8_khz() {
    assert_eq!(EBAND5MS[0], 0);
    assert_eq!(EBAND5MS[NB_BANDS - 1] as usize * 4, 160);
    assert!(EBAND5MS.windows(2).all(|w| w[0] < w[1]));
}
