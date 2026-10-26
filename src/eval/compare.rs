//! Port of the libopus test program `opus_compare` (`src/opus_compare.c`,
//! libopus 1.6.1).
//!
//! `opus_compare` decides whether a decoder passes the conformance test of
//! RFC 6716 and RFC 8251. It compares a decoded signal (the test signal)
//! with a reference signal and returns a weighted spectral error:
//!
//! 1. Both signals are cut into windows of 10 ms (480 samples at 48 kHz),
//!    one every 2.5 ms, with a Hann window. A direct discrete Fourier
//!    transform gives the power of each 100 Hz bin, plus 100000.
//! 2. The band energies of the reference (21 bands) are spread up and down
//!    across bands, over time and between the channels. A tenth of the
//!    result is added to every bin of both signals ("masking").
//! 3. Each frame's bins get the bins of the frame before added.
//! 4. Per bin, with `r` = test / reference, the distortion is
//!    `r - ln(r) - 1`. Bins 79 to 81 count a tenth, bin 80 a hundredth.
//!    The mean per band is squared and summed over the bands, divided by
//!    21, and raised to the 4th power. The result `err` is the 16th root of
//!    the mean over the frames.
//! 5. The quality `Q = 100 * (1 - 0.5 * ln(1 + err) / ln(1.13))`. The test
//!    passes when `Q >= 0`, that is when `err` is at most about 0.2769.
//!
//! The port gives the bits of the C program built with clang `-O2` for the
//! same processor:
//!
//! - It rounds as clang compiles the C code. On 64-bit ARM, clang fuses
//!   `a + b * c` into one fused multiply-add at the places where
//!   `mathops::mac` is used here. Elsewhere it rounds twice, and so
//!   does `mac`.
//! - It calls the same functions of the platform's math library: `cos` and
//!   `sin` for the tables, `log` and `pow` for the error. Each table is
//!   built in its own function, so that the compiler cannot merge a `cos`
//!   and a `sin` of the same angle into one `sincos` call, which can round
//!   differently.
//! - The constant `ln(1.13)` is the double clang folds it to.
//!
//! The C program stores the spectra of all frames. This port runs frame by
//! frame and keeps only the frame before, which computes the same numbers
//! with a few kilobytes of memory. Its transform runs several bins side by
//! side in vector lanes. Each lane does the same operations in the same
//! order as the C loop, so the result does not change.

use crate::mathops::{mac, mac_f64};
use core::fmt;

/// Window length at 48 kHz (`TEST_WIN_SIZE`): 10 ms.
pub const TEST_WIN_SIZE: usize = 480;
/// Hop between two windows at 48 kHz (`TEST_WIN_STEP`): 2.5 ms.
pub const TEST_WIN_STEP: usize = 120;
/// Bands of the comparison (`NBANDS`).
pub const NBANDS: usize = 21;
/// Bins of a 48 kHz spectrum (`NFREQS`), 100 Hz each.
pub const NFREQS: usize = 240;
/// First bin of each band (`BANDS`): bark-like bands of the CELT layer.
pub const BANDS: [usize; NBANDS + 1] = [
    0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 56, 68, 80, 96, 120, 156, 200,
];
/// The sampling rates the C program accepts for the test signal.
pub const RATES: [u32; 5] = [48000, 24000, 16000, 12000, 8000];

/// `log(1.13)` as clang folds it in `opus_compare.c`. Using the constant
/// keeps the result independent of the run-time `log` of the platform.
const LOG_1_13: f64 = f64::from_bits(0x3FBF_49A7_9F8B_9187);

/// `2*OPUS_PI` of the C program in float. `OPUS_PI` is `3.14159265F`,
/// which rounds to the same float as `f32::consts::PI`.
const TWO_PI: f32 = 2.0 * core::f32::consts::PI;

/// The reference signal. The C program always reads it as 16-bit stereo at
/// 48 kHz (its first file). For a mono comparison it mixes the two channels
/// as `.5*(L+R)`.
#[derive(Clone, Copy, Debug)]
pub enum Reference<'a> {
    /// Interleaved left and right samples, as the C program reads them.
    Stereo(&'a [i16]),
    /// One channel. It gives the same results as a stereo reference with
    /// these samples on both sides, because `.5*(x+x)` is exactly `x`.
    Mono(&'a [i16]),
}

impl Reference<'_> {
    /// Samples per channel. A stereo slice of odd length loses its last
    /// value, as the C program drops a partial last frame.
    fn len(&self) -> usize {
        match *self {
            Reference::Stereo(s) => s.len() / 2,
            Reference::Mono(s) => s.len(),
        }
    }
}

/// A test signal (the C program's second file): interleaved 16-bit samples
/// with the channel count of the comparison, at `rate`.
#[derive(Clone, Copy, Debug)]
pub struct TestSignal<'a> {
    pub samples: &'a [i16],
    pub rate: u32,
}

/// The result of one comparison.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Comparison {
    /// Frames compared (`nframes`): one every 2.5 ms.
    pub frames: usize,
    /// The "internal weighted error". 0 for identical spectra.
    pub err: f64,
    /// The quality metric `Q`, in percent. Negative when the test fails.
    pub quality: f32,
    /// The frame with the largest error. Not printed by the C program; it
    /// helps to find where two signals differ.
    pub worst_frame: usize,
    /// The error of that frame: the sum over the bands of the squared mean
    /// band error, divided by 21 (`Ef` before it is raised to the 4th
    /// power).
    pub worst_frame_error: f64,
}

impl Comparison {
    /// The C program's verdict: "Test vector PASSES" unless `Q < 0`.
    pub fn passes(&self) -> bool {
        // As in C (`if(Q<0)`), a NaN quality passes.
        self.quality >= 0.0 || self.quality.is_nan()
    }
}

impl fmt::Display for Comparison {
    /// The lines the C program prints to `stderr`, with the same rounding
    /// (`%.1f` and `%f` of C).
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.passes() {
            writeln!(f, "Test vector PASSES")?;
            writeln!(
                f,
                "Opus quality metric: {:.1} % (internal weighted error is {:.6})",
                f64::from(self.quality),
                self.err
            )
        } else {
            writeln!(f, "Test vector FAILS")?;
            writeln!(f, "Internal weighted error is {:.6}", self.err)
        }
    }
}

/// Why two signals cannot be compared. The messages are those of the C
/// program.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompareError {
    /// The rate of the test signal is not one of [`RATES`].
    BadRate(u32),
    /// The channel count is not 1 or 2. The C program has no such message:
    /// its `-s` switch only chooses between 1 and 2.
    BadChannels(usize),
    /// The reference does not have 48000 / rate samples per sample of the
    /// test signal. Both counts are at 48 kHz.
    LengthMismatch { reference: usize, test_scaled: usize },
    /// The reference is shorter than one window (480 samples).
    TooShort { samples: usize },
}

impl fmt::Display for CompareError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            CompareError::BadRate(_) => {
                write!(f, "Sampling rate must be 8000, 12000, 16000, 24000, or 48000")
            }
            CompareError::BadChannels(n) => write!(f, "Channel count must be 1 or 2, not {n}"),
            CompareError::LengthMismatch { reference, test_scaled } => {
                write!(f, "Sample counts do not match ({reference}!={test_scaled}).")
            }
            CompareError::TooShort { samples } => {
                write!(f, "Insufficient sample data ({samples}<{TEST_WIN_SIZE}).")
            }
        }
    }
}

impl std::error::Error for CompareError {}

/// `opus_compare [-s] -r <rate> reference test`: compare `test`, which has
/// `channels` interleaved channels at `rate`, with the 48 kHz reference.
pub fn opus_compare(
    reference: Reference<'_>,
    test: &[i16],
    channels: usize,
    rate: u32,
) -> Result<Comparison, CompareError> {
    let tests = [TestSignal { samples: test, rate }];
    opus_compare_many(&[reference], &tests, channels).remove(0).remove(0)
}

/// Compare every test signal with every reference in one pass: the result
/// for reference `r` and test `t` is `[r][t]`. Each result is bit for bit
/// the result of [`opus_compare`] with that pair.
///
/// The spectrum of each signal is computed once, which is where nearly all
/// the time goes. Test signals may have different rates. All have
/// `channels` channels.
pub fn opus_compare_many(
    references: &[Reference<'_>],
    tests: &[TestSignal<'_>],
    channels: usize,
) -> Vec<Vec<Result<Comparison, CompareError>>> {
    let mut results: Vec<Vec<Result<Comparison, CompareError>>> =
        vec![vec![Err(CompareError::BadChannels(channels)); tests.len()]; references.len()];
    if channels != 1 && channels != 2 {
        return results;
    }
    let nch = channels;

    // The checks of the C program, in its order: the rate, then the
    // lengths.
    let params: Vec<Option<RateParams>> = tests.iter().map(|t| RateParams::new(t.rate)).collect();
    let mut pairs = Vec::new();
    for (r, reference) in references.iter().enumerate() {
        let xlength = reference.len();
        for (t, test) in tests.iter().enumerate() {
            let result = &mut results[r][t];
            let Some(p) = params[t] else {
                *result = Err(CompareError::BadRate(test.rate));
                continue;
            };
            let test_scaled = test.samples.len() / nch * p.downsample;
            if xlength != test_scaled {
                *result = Err(CompareError::LengthMismatch { reference: xlength, test_scaled });
            } else if xlength < TEST_WIN_SIZE {
                *result = Err(CompareError::TooShort { samples: xlength });
            } else {
                pairs.push(Pair::new(r, t, p, frame_count(xlength), nch));
            }
        }
    }
    if pairs.is_empty() {
        return results;
    }

    // The transforms: the reference's at 48 kHz over all 21 bands, and one
    // per rate of the test signals over the bands that rate compares.
    let mut dfts: Vec<Dft> = Vec::new();
    let mut dft_of = |downsample: usize, nbins: usize| -> usize {
        let w = TEST_WIN_SIZE / downsample;
        if let Some(i) = dfts.iter().position(|d| d.w == w && d.nbins == nbins) {
            return i;
        }
        dfts.push(Dft::new(w, nbins));
        dfts.len() - 1
    };
    let mut refs: Vec<Option<RefState>> = references.iter().map(|_| None).collect();
    let mut sigs: Vec<Option<TestState>> = tests.iter().map(|_| None).collect();
    for pair in &pairs {
        if refs[pair.r].is_none() {
            let source = Source::reference(references[pair.r], nch);
            let dft = dft_of(1, BANDS[NBANDS]);
            refs[pair.r] = Some(RefState::new(source, dft, nch));
        }
        if sigs[pair.t].is_none() {
            let p = params[pair.t].expect("checked above");
            let source = Source::Interleaved { samples: tests[pair.t].samples, nch };
            let dft = dft_of(p.downsample, BANDS[p.ybands]);
            sigs[pair.t] = Some(TestState::new(source, dft, p, nch));
        }
    }

    let mut scratch = Vec::new();
    let total = pairs.iter().map(|p| p.frames).max().unwrap_or(0);
    let mut start = 0;
    while start < total {
        let end = (start + BATCH).min(total);
        // The spectra of a few frames of each signal at once, so that the
        // transform has more independent work (see `Dft::power`).
        for state in refs.iter_mut().flatten() {
            let stop = end.min(state.frames);
            if start < stop {
                let dft = &dfts[state.dft];
                dft.spectra(state.source, TEST_WIN_STEP, start..stop, 1.0, &mut state.ps, &mut scratch);
            }
        }
        for state in sigs.iter_mut().flatten() {
            let stop = end.min(state.frames);
            if start < stop {
                let dft = &dfts[state.dft];
                let step = TEST_WIN_STEP / state.params.downsample;
                let ds = state.params.downsample as f32;
                dft.spectra(state.source, step, start..stop, ds, &mut state.ps, &mut scratch);
            }
        }
        for i in start..end {
            let f = i - start;
            for state in refs.iter_mut().flatten() {
                if i < state.frames {
                    state.next_frame(i, f);
                }
            }
            for pair in pairs.iter_mut().filter(|p| i < p.frames) {
                let reference = refs[pair.r].as_ref().expect("set above");
                let test = sigs[pair.t].as_ref().expect("set above");
                pair.next_frame(i, f, reference, test);
            }
        }
        start = end;
    }

    for pair in &pairs {
        results[pair.r][pair.t] = Ok(pair.finish());
    }
    results
}

/// Frames compared for a reference of `xlength` samples (`nframes`).
fn frame_count(xlength: usize) -> usize {
    (xlength - TEST_WIN_SIZE + TEST_WIN_STEP) / TEST_WIN_STEP
}

/// Frames whose spectra are computed together.
const BATCH: usize = 4;

/// What the rate of the test signal changes.
#[derive(Clone, Copy, Debug)]
struct RateParams {
    /// 48000 / rate.
    downsample: usize,
    /// Bands of the test signal (`ybands`): those below its Nyquist
    /// frequency.
    ybands: usize,
    /// Bins below this one are compared (`max_compare`). Below 48 kHz the
    /// top 300 Hz are left out, except at 12 kHz, where the last band
    /// already stops 400 Hz below 6 kHz.
    max_compare: usize,
}

impl RateParams {
    fn new(rate: u32) -> Option<Self> {
        let (downsample, ybands) = match rate {
            48000 => (1, 21),
            24000 => (2, 19),
            16000 => (3, 17),
            12000 => (4, 15),
            8000 => (6, 13),
            _ => return None,
        };
        let max_compare = match rate {
            48000 => BANDS[NBANDS],
            12000 => BANDS[ybands],
            _ => BANDS[ybands] - 3,
        };
        Some(Self { downsample, ybands, max_compare })
    }
}

/// Where the samples of a signal come from.
#[derive(Clone, Copy, Debug)]
enum Source<'a> {
    /// Interleaved channels.
    Interleaved { samples: &'a [i16], nch: usize },
    /// `.5*(L+R)` of an interleaved stereo signal: the mono reference.
    MonoMix(&'a [i16]),
    /// One channel, used as both channels of a stereo reference.
    Duplicated(&'a [i16]),
}

impl Source<'_> {
    fn reference(reference: Reference<'_>, nch: usize) -> Source<'_> {
        match (reference, nch) {
            (Reference::Stereo(s), 1) => Source::MonoMix(s),
            (Reference::Stereo(s), _) => Source::Interleaved { samples: s, nch: 2 },
            (Reference::Mono(s), 1) => Source::Interleaved { samples: s, nch: 1 },
            (Reference::Mono(s), _) => Source::Duplicated(s),
        }
    }

    /// Sample `i` of channel `ch`, as the C program holds it in float.
    #[inline(always)]
    fn sample(&self, i: usize, ch: usize) -> f32 {
        match *self {
            Source::Interleaved { samples, nch } => f32::from(samples[i * nch + ch]),
            // C: `x[xi]=.5*(x[2*xi]+x[2*xi+1])`. The float sum of two
            // 16-bit values and its half are exact.
            Source::MonoMix(s) => (f32::from(s[2 * i]) + f32::from(s[2 * i + 1])) * 0.5,
            Source::Duplicated(s) => f32::from(s[i]),
        }
    }
}

/// The tables of `band_energy` for one window length, and the transform.
struct Dft {
    /// Window length.
    w: usize,
    /// Bins computed: the first bin of the band after the last band.
    nbins: usize,
    /// Hann window.
    window: Vec<f32>,
    /// `cos` and `sin` of the bins' angles, in the order the transform
    /// reads them: `[block of 4 bins][k][bin of the block]`, where entry
    /// `k` of bin `j` is the C table at `(j*k) mod w`.
    cb: Vec<f32>,
    sb: Vec<f32>,
}

impl Dft {
    fn new(w: usize, nbins: usize) -> Self {
        // C: `window[xj]=0.5F-0.5F*OPUS_COSF((2*OPUS_PI/(_window_sz-1))*xj)`.
        // clang fuses it into `0.5 + c*(-0.5)`; `0.5*c` is exact, so both
        // round the same.
        let window = cos_table(w, w - 1).iter().map(|&c| 0.5 - 0.5 * c).collect();
        let c = cos_table(w, w);
        let s = sin_table(w, w);
        let nblocks = nbins.div_ceil(4);
        let mut cb = vec![0.0; nblocks * w * 4];
        let mut sb = vec![0.0; nblocks * w * 4];
        for j in 0..nblocks * 4 {
            let (block, lane) = (j / 4, j % 4);
            // C: `ti+=xj; if(ti>=_window_sz)ti-=_window_sz;`.
            let mut ti = 0;
            for k in 0..w {
                cb[(block * w + k) * 4 + lane] = c[ti];
                sb[(block * w + k) * 4 + lane] = s[ti];
                ti += j % w;
                if ti >= w {
                    ti -= w;
                }
            }
        }
        Self { w, nbins, window, cb, sb }
    }

    /// The power of bins `0..nbins` (`_ps` of `band_energy`, plus the
    /// `re*=_downsample` scaling) for frames `frames` of `source`, every
    /// channel. Frame `f` of the range goes to `ps[(f*NFREQS + j)*nch +
    /// ch]`.
    fn spectra(
        &self,
        source: Source<'_>,
        step: usize,
        frames: core::ops::Range<usize>,
        downsample: f32,
        ps: &mut [f32],
        scratch: &mut Vec<f32>,
    ) {
        let w = self.w;
        let nch = match source {
            Source::Interleaved { nch, .. } => nch,
            Source::MonoMix(_) => 1,
            Source::Duplicated(_) => 2,
        };
        let count = frames.len() * nch;
        scratch.clear();
        scratch.resize(count * w, 0.0);
        for (f, i) in frames.enumerate() {
            for ch in 0..nch {
                let x = &mut scratch[(f * nch + ch) * w..][..w];
                // C: `x[...]=window[xk]*_in[(xi*_step+xk)*_nchannels+ci]`.
                for (k, (x, &win)) in x.iter_mut().zip(&self.window).enumerate() {
                    *x = win * source.sample(i * step + k, ch);
                }
            }
        }
        let mut n = 0;
        while n + 2 <= count {
            let xs = [&scratch[n * w..][..w], &scratch[(n + 1) * w..][..w]];
            self.power(xs, [n, n + 1], nch, downsample, ps);
            n += 2;
        }
        if n < count {
            self.power([&scratch[n * w..][..w]], [n], nch, downsample, ps);
        }
    }

    /// The bins of `N` windowed frames at once. `slots[n]` is
    /// `frame*nch + ch` of input `n`.
    fn power<const N: usize>(
        &self,
        xs: [&[f32]; N],
        slots: [usize; N],
        nch: usize,
        downsample: f32,
        ps: &mut [f32],
    ) {
        let nblocks = self.nbins.div_ceil(4);
        let mut block = 0;
        // Groups of blocks whose sums are independent of each other, so that
        // the processor can overlap their multiply-adds.
        while block + 4 <= nblocks {
            self.group::<N, 4>(&xs, block, slots, nch, downsample, ps);
            block += 4;
        }
        while block + 2 <= nblocks {
            self.group::<N, 2>(&xs, block, slots, nch, downsample, ps);
            block += 2;
        }
        while block < nblocks {
            self.group::<N, 1>(&xs, block, slots, nch, downsample, ps);
            block += 1;
        }
    }

    /// `B` blocks of 4 bins, for `N` inputs.
    #[inline(always)]
    fn group<const N: usize, const B: usize>(
        &self,
        xs: &[&[f32]; N],
        block: usize,
        slots: [usize; N],
        nch: usize,
        downsample: f32,
        ps: &mut [f32],
    ) {
        let mut re = [[[0.0f32; 4]; B]; N];
        let mut im = [[[0.0f32; 4]; B]; N];
        let w = self.w;
        let cb = &self.cb[block * w * 4..(block + B) * w * 4];
        let sb = &self.sb[block * w * 4..(block + B) * w * 4];
        dft_blocks::<N, B>(xs, cb, sb, w, &mut re, &mut im);
        for n in 0..N {
            let (frame, ch) = (slots[n] / nch, slots[n] % nch);
            let out = &mut ps[frame * NFREQS * nch..];
            for b in 0..B {
                for lane in 0..4 {
                    let j = (block + b) * 4 + lane;
                    if j < self.nbins {
                        out[j * nch + ch] = bin_power(re[n][b][lane], im[n][b][lane], downsample);
                    }
                }
            }
        }
    }
}

/// C: `re*=_downsample; im*=_downsample; _ps[...]=re*re+im*im+100000`.
/// clang computes `im*im` first and fuses `re*re` with the sum.
#[inline(always)]
fn bin_power(re: f32, im: f32, downsample: f32) -> f32 {
    let re = re * downsample;
    let im = im * downsample;
    mac(im * im, re, re) + 100000.0
}

/// `(float)cos((double)((2*OPUS_PI/period)*k))` for `k` in `0..w`.
///
/// Not inlined, and the angle passes through `black_box`, so that the
/// compiler neither merges this `cos` with the `sin` of the same angle
/// (`sincos` can round differently) nor folds it at compile time.
#[inline(never)]
fn cos_table(w: usize, period: usize) -> Vec<f32> {
    let step = TWO_PI / period as f32;
    (0..w).map(|k| f64::from(core::hint::black_box(step * k as f32)).cos() as f32).collect()
}

/// `(float)sin((double)((2*OPUS_PI/period)*k))` for `k` in `0..w`. See
/// [`cos_table`].
#[inline(never)]
fn sin_table(w: usize, period: usize) -> Vec<f32> {
    let step = TWO_PI / period as f32;
    (0..w).map(|k| f64::from(core::hint::black_box(step * k as f32)).sin() as f32).collect()
}

/// The transform of `B` blocks of 4 bins for `N` inputs of `w` samples: for
/// each bin, `re += c*x` and `im -= s*x` over the samples, in the order of
/// the C loop. `cb` and `sb` hold the `B` blocks of the tables.
///
/// 64-bit ARM: one fused multiply-add (`vfmaq_f32`) or fused
/// multiply-subtract (`vfmsq_f32`) per bin and sample, as clang compiles
/// the C loop (`fmadd`, `fmsub`).
///
/// The loop keeps its pointer loads and stores. With the blocks of the
/// tables taken from `as_chunks` and loaded through `simd_mem`, the
/// compiler assigned the registers of `opus_compare_many` differently on
/// Android, and loops there got longer.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn dft_blocks<const N: usize, const B: usize>(
    xs: &[&[f32]; N],
    cb: &[f32],
    sb: &[f32],
    w: usize,
    re_out: &mut [[[f32; 4]; B]; N],
    im_out: &mut [[[f32; 4]; B]; N],
) {
    use core::arch::aarch64::*;
    assert!(cb.len() >= B * w * 4 && sb.len() >= B * w * 4);
    assert!(xs.iter().all(|x| x.len() >= w));
    // SAFETY: NEON is part of every 64-bit ARM processor. The reads stay in
    // `cb`, `sb` (B blocks of w*4 values) and `xs` (w values each), as the
    // asserts above check.
    unsafe {
        let mut re = [[vdupq_n_f32(0.0); B]; N];
        let mut im = [[vdupq_n_f32(0.0); B]; N];
        let (cp, sp) = (cb.as_ptr(), sb.as_ptr());
        for k in 0..w {
            let mut x = [vdupq_n_f32(0.0); N];
            for n in 0..N {
                x[n] = vld1q_dup_f32(xs[n].as_ptr().add(k));
            }
            for b in 0..B {
                let c = vld1q_f32(cp.add((b * w + k) * 4));
                let s = vld1q_f32(sp.add((b * w + k) * 4));
                for n in 0..N {
                    re[n][b] = vfmaq_f32(re[n][b], c, x[n]);
                    im[n][b] = vfmsq_f32(im[n][b], s, x[n]);
                }
            }
        }
        for n in 0..N {
            for b in 0..B {
                vst1q_f32(re_out[n][b].as_mut_ptr(), re[n][b]);
                vst1q_f32(im_out[n][b].as_mut_ptr(), im[n][b]);
            }
        }
    }
}

/// The same on other processors, with [`mac`] (a separate multiply and
/// add, as clang compiles the C loop for x86-64 without `-mfma`).
#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
fn dft_blocks<const N: usize, const B: usize>(
    xs: &[&[f32]; N],
    cb: &[f32],
    sb: &[f32],
    w: usize,
    re_out: &mut [[[f32; 4]; B]; N],
    im_out: &mut [[[f32; 4]; B]; N],
) {
    let mut re = [[[0.0f32; 4]; B]; N];
    let mut im = [[[0.0f32; 4]; B]; N];
    for k in 0..w {
        for b in 0..B {
            let c = &cb[(b * w + k) * 4..][..4];
            let s = &sb[(b * w + k) * 4..][..4];
            for n in 0..N {
                let x = xs[n][k];
                for lane in 0..4 {
                    re[n][b][lane] = mac(re[n][b][lane], c[lane], x);
                    im[n][b][lane] = mac(im[n][b][lane], -s[lane], x);
                }
            }
        }
    }
    *re_out = re;
    *im_out = im;
}

/// Index of `xb` and of a spectrum: band or bin `j`, channel `ch`. The
/// stride is 2 whatever the channel count.
#[inline(always)]
fn at(j: usize, ch: usize) -> usize {
    j * 2 + ch
}

/// One reference: its spectra and the masking state.
struct RefState<'a> {
    source: Source<'a>,
    dft: usize,
    nch: usize,
    frames: usize,
    /// Spectra of the current batch: `[frame][bin][ch]`.
    ps: Vec<f32>,
    /// Masked band energies (`xb`) of the last frame processed.
    xb: [f32; NBANDS * 2],
    /// Masked spectrum of the previous frame, before averaging.
    xm_prev: [f32; NFREQS * 2],
    /// Masked and averaged spectrum of the current frame.
    xa: [f32; NFREQS * 2],
}

impl<'a> RefState<'a> {
    fn new(source: Source<'a>, dft: usize, nch: usize) -> Self {
        let len = match source {
            Source::Interleaved { samples, nch } => samples.len() / nch,
            Source::MonoMix(s) => s.len() / 2,
            Source::Duplicated(s) => s.len(),
        };
        Self {
            source,
            dft,
            nch,
            frames: frame_count(len),
            ps: vec![0.0; BATCH * NFREQS * nch],
            xb: [0.0; NBANDS * 2],
            xm_prev: [0.0; NFREQS * 2],
            xa: [0.0; NFREQS * 2],
        }
    }

    /// Frame `i`, which is frame `f` of the batch: band energies, masking
    /// and averaging (C lines 254-328 for one frame).
    fn next_frame(&mut self, i: usize, f: usize) {
        let nch = self.nch;
        let ps = &self.ps[f * NFREQS * nch..][..NFREQS * nch];
        // C `band_energy`: the mean power per band.
        let mut xb = [0.0f32; NBANDS * 2];
        for b in 0..NBANDS {
            let mut p = [0.0f32; 2];
            for j in BANDS[b]..BANDS[b + 1] {
                for ch in 0..nch {
                    p[ch] += ps[j * nch + ch];
                }
            }
            let width = (BANDS[b + 1] - BANDS[b]) as f32;
            for ch in 0..nch {
                xb[at(b, ch)] = p[ch] / width;
            }
        }
        // Frequency masking, low to high (10 dB/Bark) and high to low
        // (15 dB/Bark).
        for b in 1..NBANDS {
            for ch in 0..nch {
                xb[at(b, ch)] = mac(xb[at(b, ch)], 0.1, xb[at(b - 1, ch)]);
            }
        }
        for b in (0..NBANDS - 1).rev() {
            for ch in 0..nch {
                xb[at(b, ch)] = mac(xb[at(b, ch)], 0.03, xb[at(b + 1, ch)]);
            }
        }
        // Temporal masking (-3 dB per 2.5 ms), from the fully processed
        // previous frame.
        if i > 0 {
            for b in 0..NBANDS {
                for ch in 0..nch {
                    xb[at(b, ch)] = mac(xb[at(b, ch)], 0.5, self.xb[at(b, ch)]);
                }
            }
        }
        // Some cross-talk between the channels, from the old values.
        if nch == 2 {
            for b in 0..NBANDS {
                let (l, r) = (xb[at(b, 0)], xb[at(b, 1)]);
                xb[at(b, 0)] = mac(l, 0.01, r);
                xb[at(b, 1)] = mac(r, 0.01, l);
            }
        }
        // Masking of the reference spectrum (all bands; a comparison at a
        // lower rate reads only its own), then the sum with the previous
        // frame.
        for b in 0..NBANDS {
            for j in BANDS[b]..BANDS[b + 1] {
                for ch in 0..nch {
                    let xm = mac(ps[j * nch + ch], 0.1, xb[at(b, ch)]);
                    self.xa[at(j, ch)] = if i > 0 { xm + self.xm_prev[at(j, ch)] } else { xm };
                    self.xm_prev[at(j, ch)] = xm;
                }
            }
        }
        self.xb = xb;
    }
}

/// One test signal: its spectra.
struct TestState<'a> {
    source: Source<'a>,
    dft: usize,
    params: RateParams,
    frames: usize,
    /// Spectra of the current batch: `[frame][bin][ch]`.
    ps: Vec<f32>,
}

impl<'a> TestState<'a> {
    fn new(source: Source<'a>, dft: usize, params: RateParams, nch: usize) -> Self {
        let len = match source {
            Source::Interleaved { samples, nch } => samples.len() / nch,
            _ => unreachable!("a test signal is interleaved"),
        };
        Self {
            source,
            dft,
            params,
            frames: frame_count(len * params.downsample),
            ps: vec![0.0; BATCH * NFREQS * nch],
        }
    }
}

/// One comparison: the masking state of the test signal and the sum of
/// the error.
struct Pair {
    r: usize,
    t: usize,
    params: RateParams,
    frames: usize,
    nch: usize,
    /// Masked spectrum of the test signal in the previous frame, before
    /// averaging.
    ym_prev: [f32; NFREQS * 2],
    err: f64,
    worst_frame: usize,
    worst_frame_error: f64,
}

impl Pair {
    fn new(r: usize, t: usize, params: RateParams, frames: usize, nch: usize) -> Self {
        Self {
            r,
            t,
            params,
            frames,
            nch,
            ym_prev: [0.0; NFREQS * 2],
            err: 0.0,
            worst_frame: 0,
            worst_frame_error: f64::NEG_INFINITY,
        }
    }

    /// Frame `i` (frame `f` of the batch): masking and averaging of the test
    /// spectrum, then the error of the frame (C lines 295-365 for one
    /// frame).
    fn next_frame(&mut self, i: usize, f: usize, reference: &RefState<'_>, test: &TestState<'_>) {
        let nch = self.nch;
        let RateParams { ybands, max_compare, .. } = self.params;
        let ps = &test.ps[f * NFREQS * nch..][..NFREQS * nch];
        let mut ya = [0.0f32; NFREQS * 2];
        for b in 0..ybands {
            for j in BANDS[b]..BANDS[b + 1] {
                for ch in 0..nch {
                    let ym = mac(ps[j * nch + ch], 0.1, reference.xb[at(b, ch)]);
                    ya[at(j, ch)] = if i > 0 { ym + self.ym_prev[at(j, ch)] } else { ym };
                    self.ym_prev[at(j, ch)] = ym;
                }
            }
        }
        let xa = &reference.xa;
        let mut ef = 0.0f64;
        for b in 0..ybands {
            let mut eb = 0.0f64;
            for j in BANDS[b]..BANDS[b + 1].min(max_compare) {
                for ch in 0..nch {
                    let re = ya[at(j, ch)] / xa[at(j, ch)];
                    // C: `im=re-log(re)-1`, in double, rounded to float.
                    let r = f64::from(re);
                    let mut im = (r - r.ln() - 1.0) as f32;
                    // Less weight around the SILK/CELT cross-over at 8 kHz,
                    // where the modes may filter differently.
                    if (79..=81).contains(&j) {
                        im *= 0.1;
                    }
                    if j == 80 {
                        im *= 0.1;
                    }
                    eb += f64::from(im);
                }
            }
            // The full band width, also where `max_compare` cut the band.
            eb /= ((BANDS[b + 1] - BANDS[b]) * nch) as f64;
            ef = mac_f64(ef, eb, eb);
        }
        // The fixed band count, also at lower rates.
        ef /= NBANDS as f64;
        if ef > self.worst_frame_error {
            self.worst_frame = i;
            self.worst_frame_error = ef;
        }
        ef *= ef;
        self.err = mac_f64(self.err, ef, ef);
    }

    fn finish(&self) -> Comparison {
        let err = (self.err / self.frames as f64).powf(1.0 / 16.0);
        let quality = (100.0 * (1.0 - 0.5 * (1.0 + err).ln() / LOG_1_13)) as f32;
        Comparison {
            frames: self.frames,
            err,
            quality,
            worst_frame: self.worst_frame,
            worst_frame_error: self.worst_frame_error,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A direct transcription of `main` and `band_energy` of
    /// `opus_compare.c`, with the whole signal in memory, for checking the
    /// frame-by-frame version. Only the rounding helpers (`mac`, `mac_f64`)
    /// stand for what clang contracts.
    fn oracle(x_stereo: &[i16], y: &[i16], nchannels: usize, rate: u32) -> Option<(f64, f32)> {
        let (downsample, ybands) = match rate {
            48000 => (1, 21),
            24000 => (2, 19),
            16000 => (3, 17),
            12000 => (4, 15),
            8000 => (6, 13),
            _ => return None,
        };
        let yfreqs = NFREQS / downsample;
        let xlength = x_stereo.len() / 2;
        let mut x: Vec<f32> = x_stereo[..xlength * 2].iter().map(|&s| f32::from(s)).collect();
        if nchannels == 1 {
            for xi in 0..xlength {
                x[xi] = (0.5 * f64::from(x[2 * xi] + x[2 * xi + 1])) as f32;
            }
        }
        let ylength = y.len() / nchannels;
        let yv: Vec<f32> = y[..ylength * nchannels].iter().map(|&s| f32::from(s)).collect();
        if xlength != ylength * downsample || xlength < TEST_WIN_SIZE {
            return None;
        }
        let nframes = (xlength - TEST_WIN_SIZE + TEST_WIN_STEP) / TEST_WIN_STEP;
        let mut xb = vec![0f32; nframes * NBANDS * nchannels];
        let mut xs = vec![0f32; nframes * NFREQS * nchannels];
        let mut ys = vec![0f32; nframes * yfreqs * nchannels];
        band_energy_c(Some(&mut xb), &mut xs, NBANDS, &x, nchannels, nframes, TEST_WIN_SIZE, TEST_WIN_STEP, 1);
        band_energy_c(
            None,
            &mut ys,
            ybands,
            &yv,
            nchannels,
            nframes,
            TEST_WIN_SIZE / downsample,
            TEST_WIN_STEP / downsample,
            downsample,
        );
        let nc = nchannels;
        for xi in 0..nframes {
            for bi in 1..NBANDS {
                for ci in 0..nc {
                    let v = xb[(xi * NBANDS + bi - 1) * nc + ci];
                    xb[(xi * NBANDS + bi) * nc + ci] = mac(xb[(xi * NBANDS + bi) * nc + ci], 0.1, v);
                }
            }
            for bi in (0..NBANDS - 1).rev() {
                for ci in 0..nc {
                    let v = xb[(xi * NBANDS + bi + 1) * nc + ci];
                    xb[(xi * NBANDS + bi) * nc + ci] = mac(xb[(xi * NBANDS + bi) * nc + ci], 0.03, v);
                }
            }
            if xi > 0 {
                for bi in 0..NBANDS {
                    for ci in 0..nc {
                        let v = xb[((xi - 1) * NBANDS + bi) * nc + ci];
                        xb[(xi * NBANDS + bi) * nc + ci] = mac(xb[(xi * NBANDS + bi) * nc + ci], 0.5, v);
                    }
                }
            }
            if nc == 2 {
                for bi in 0..NBANDS {
                    let l = xb[(xi * NBANDS + bi) * nc];
                    let r = xb[(xi * NBANDS + bi) * nc + 1];
                    xb[(xi * NBANDS + bi) * nc] = mac(l, 0.01, r);
                    xb[(xi * NBANDS + bi) * nc + 1] = mac(r, 0.01, l);
                }
            }
            for bi in 0..ybands {
                for xj in BANDS[bi]..BANDS[bi + 1] {
                    for ci in 0..nc {
                        let m = xb[(xi * NBANDS + bi) * nc + ci];
                        let xk = (xi * NFREQS + xj) * nc + ci;
                        xs[xk] = mac(xs[xk], 0.1, m);
                        let yk = (xi * yfreqs + xj) * nc + ci;
                        ys[yk] = mac(ys[yk], 0.1, m);
                    }
                }
            }
        }
        for bi in 0..ybands {
            for xj in BANDS[bi]..BANDS[bi + 1] {
                for ci in 0..nc {
                    let mut xtmp = xs[xj * nc + ci];
                    let mut ytmp = ys[xj * nc + ci];
                    for xi in 1..nframes {
                        let xtmp2 = xs[(xi * NFREQS + xj) * nc + ci];
                        let ytmp2 = ys[(xi * yfreqs + xj) * nc + ci];
                        xs[(xi * NFREQS + xj) * nc + ci] += xtmp;
                        ys[(xi * yfreqs + xj) * nc + ci] += ytmp;
                        xtmp = xtmp2;
                        ytmp = ytmp2;
                    }
                }
            }
        }
        let max_compare = if rate == 48000 {
            BANDS[NBANDS]
        } else if rate == 12000 {
            BANDS[ybands]
        } else {
            BANDS[ybands] - 3
        };
        let mut err = 0f64;
        for xi in 0..nframes {
            let mut ef = 0f64;
            for bi in 0..ybands {
                let mut eb = 0f64;
                let mut xj = BANDS[bi];
                while xj < BANDS[bi + 1] && xj < max_compare {
                    for ci in 0..nc {
                        let re = ys[(xi * yfreqs + xj) * nc + ci] / xs[(xi * NFREQS + xj) * nc + ci];
                        let mut im = (f64::from(re) - f64::from(re).ln() - 1.0) as f32;
                        if (79..=81).contains(&xj) {
                            im *= 0.1;
                        }
                        if xj == 80 {
                            im *= 0.1;
                        }
                        eb += f64::from(im);
                    }
                    xj += 1;
                }
                eb /= ((BANDS[bi + 1] - BANDS[bi]) * nc) as f64;
                ef = mac_f64(ef, eb, eb);
            }
            ef /= NBANDS as f64;
            ef *= ef;
            err = mac_f64(err, ef, ef);
        }
        err = (err / nframes as f64).powf(1.0 / 16.0);
        let q = (100.0 * (1.0 - 0.5 * (1.0 + err).ln() / 1.13f64.ln())) as f32;
        Some((err, q))
    }

    #[allow(clippy::too_many_arguments)]
    fn band_energy_c(
        mut out: Option<&mut [f32]>,
        ps: &mut [f32],
        nbands: usize,
        input: &[f32],
        nchannels: usize,
        nframes: usize,
        window_sz: usize,
        step: usize,
        downsample: usize,
    ) {
        let window: Vec<f32> = (0..window_sz)
            .map(|xj| 0.5 - 0.5 * (f64::from((TWO_PI / (window_sz - 1) as f32) * xj as f32).cos() as f32))
            .collect();
        let c: Vec<f32> =
            (0..window_sz).map(|xj| f64::from((TWO_PI / window_sz as f32) * xj as f32).cos() as f32).collect();
        let s: Vec<f32> = (0..window_sz)
            .map(|xj| f64::from(core::hint::black_box((TWO_PI / window_sz as f32) * xj as f32)).sin() as f32)
            .collect();
        let ps_sz = window_sz / 2;
        let mut x = vec![0f32; nchannels * window_sz];
        for xi in 0..nframes {
            for ci in 0..nchannels {
                for xk in 0..window_sz {
                    x[ci * window_sz + xk] = window[xk] * input[(xi * step + xk) * nchannels + ci];
                }
            }
            let mut xj = 0;
            for bi in 0..nbands {
                let mut p = [0f32; 2];
                while xj < BANDS[bi + 1] {
                    for ci in 0..nchannels {
                        let (mut re, mut im) = (0f32, 0f32);
                        let mut ti = 0;
                        for xk in 0..window_sz {
                            re = mac(re, c[ti], x[ci * window_sz + xk]);
                            im = mac(im, -s[ti], x[ci * window_sz + xk]);
                            ti += xj;
                            if ti >= window_sz {
                                ti -= window_sz;
                            }
                        }
                        re *= downsample as f32;
                        im *= downsample as f32;
                        ps[(xi * ps_sz + xj) * nchannels + ci] = mac(im * im, re, re) + 100000.0;
                        p[ci] += ps[(xi * ps_sz + xj) * nchannels + ci];
                    }
                    xj += 1;
                }
                if let Some(out) = out.as_deref_mut() {
                    let width = (BANDS[bi + 1] - BANDS[bi]) as f32;
                    out[(xi * nbands + bi) * nchannels] = p[0] / width;
                    if nchannels == 2 {
                        out[(xi * nbands + bi) * nchannels + 1] = p[1] / width;
                    }
                }
            }
        }
    }

    struct XorShift(u64);
    impl XorShift {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn noise(&mut self, n: usize, amp: i32) -> Vec<i16> {
            (0..n).map(|_| ((self.next() % (2 * amp as u64 + 1)) as i32 - amp) as i16).collect()
        }
    }

    /// A 48 kHz stereo reference and a test signal at `rate` made from it
    /// (every `48000/rate`-th sample, plus noise).
    fn signals(seed: u64, frames48: usize, rate: u32, nch: usize) -> (Vec<i16>, Vec<i16>) {
        let mut rng = XorShift(seed);
        let ds = 48000 / rate as usize;
        let mut reference = vec![0i16; frames48 * 2];
        let mut phase = 0.0f64;
        for i in 0..frames48 {
            phase += 0.02 + 0.01 * (i as f64 * 1e-3).sin();
            let v = (6000.0 * phase.sin()) as i32;
            reference[2 * i] = (v + (rng.next() % 801) as i32 - 400) as i16;
            reference[2 * i + 1] = (v / 2 + (rng.next() % 401) as i32 - 200) as i16;
        }
        let noise = rng.noise(frames48 / ds * nch, 300);
        let mut test = vec![0i16; frames48 / ds * nch];
        for i in 0..frames48 / ds {
            for ch in 0..nch {
                let src = if nch == 2 {
                    reference[2 * i * ds + ch] as i32
                } else {
                    (reference[2 * i * ds] as i32 + reference[2 * i * ds + 1] as i32) / 2
                };
                test[i * nch + ch] = (src + noise[i * nch + ch] as i32).clamp(-32768, 32767) as i16;
            }
        }
        (reference, test)
    }

    #[test]
    fn two_pi_is_the_c_constant() {
        assert_eq!(TWO_PI.to_bits(), 0x40C9_0FDB);
        assert_eq!(LOG_1_13, 1.13f64.ln());
    }

    #[test]
    fn matches_the_transcription_of_the_c_program() {
        for (seed, &rate) in RATES.iter().enumerate() {
            for nch in [1, 2] {
                // 0.25 s plus a part of a hop, so that the last samples are
                // not in any window.
                let (reference, test) = signals(seed as u64 + 7, 12000 + 48 * 6, rate, nch);
                let got = opus_compare(Reference::Stereo(&reference), &test, nch, rate).unwrap();
                let (err, q) = oracle(&reference, &test, nch, rate).unwrap();
                assert_eq!(got.err.to_bits(), err.to_bits(), "rate {rate}, {nch} channels");
                assert_eq!(got.quality.to_bits(), q.to_bits(), "rate {rate}, {nch} channels");
            }
        }
    }

    #[test]
    fn many_equals_one_at_a_time() {
        let (ref_a, _) = signals(3, 9600, 48000, 2);
        let (ref_b, _) = signals(4, 9600, 48000, 2);
        for nch in [1, 2] {
            let tests_owned: Vec<(Vec<i16>, u32)> = RATES
                .iter()
                .enumerate()
                .map(|(i, &rate)| (signals(10 + i as u64, 9600, rate, nch).1, rate))
                .collect();
            let tests: Vec<TestSignal<'_>> =
                tests_owned.iter().map(|(s, rate)| TestSignal { samples: s, rate: *rate }).collect();
            let refs = [Reference::Stereo(&ref_a), Reference::Stereo(&ref_b), Reference::Mono(&ref_a[..9600])];
            let many = opus_compare_many(&refs, &tests, nch);
            for (r, reference) in refs.iter().enumerate() {
                for (t, test) in tests.iter().enumerate() {
                    let one = opus_compare(*reference, test.samples, nch, test.rate);
                    assert_eq!(many[r][t], one, "reference {r}, test {t}, {nch} channels");
                }
            }
        }
    }

    #[test]
    fn vector_kernel_equals_the_scalar_loop() {
        let mut rng = XorShift(99);
        for (w, nbins) in [(480, 200), (240, 120), (160, 80), (120, 56), (80, 40), (480, 6)] {
            let dft = Dft::new(w, nbins);
            let c: Vec<f32> = (0..w).map(|k| f64::from((TWO_PI / w as f32) * k as f32).cos() as f32).collect();
            let s: Vec<f32> = (0..w).map(|k| f64::from((TWO_PI / w as f32) * k as f32).sin() as f32).collect();
            for nch in [1, 2] {
                let frames = 3;
                let samples: Vec<i16> = rng.noise((frames * 40 + w) * nch, 20000);
                let source = Source::Interleaved { samples: &samples, nch };
                let mut ps = vec![0.0; BATCH * NFREQS * nch];
                let mut scratch = Vec::new();
                dft.spectra(source, 40, 0..frames, 3.0, &mut ps, &mut scratch);
                for f in 0..frames {
                    for ch in 0..nch {
                        let x: Vec<f32> =
                            (0..w).map(|k| dft.window[k] * source.sample(f * 40 + k, ch)).collect();
                        for j in 0..nbins {
                            let (mut re, mut im, mut ti) = (0f32, 0f32, 0);
                            for k in 0..w {
                                re = mac(re, c[ti], x[k]);
                                im = mac(im, -s[ti], x[k]);
                                ti += j;
                                if ti >= w {
                                    ti -= w;
                                }
                            }
                            let want = bin_power(re, im, 3.0);
                            let got = ps[(f * NFREQS + j) * nch + ch];
                            assert_eq!(got.to_bits(), want.to_bits(), "w {w}, bin {j}, frame {f}, ch {ch}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn identical_signals_give_zero_error() {
        let (reference, _) = signals(5, 4800, 48000, 2);
        let got = opus_compare(Reference::Stereo(&reference), &reference, 2, 48000).unwrap();
        assert_eq!(got.err, 0.0);
        assert_eq!(got.quality, 100.0);
        assert!(got.passes());
    }

    #[test]
    fn errors_follow_the_c_checks() {
        let x = vec![0i16; 960];
        let short = vec![0i16; 400];
        assert_eq!(opus_compare(Reference::Stereo(&x), &x, 2, 44100), Err(CompareError::BadRate(44100)));
        assert_eq!(opus_compare(Reference::Stereo(&x), &x, 3, 48000), Err(CompareError::BadChannels(3)));
        assert_eq!(
            opus_compare(Reference::Stereo(&x), &x[..100], 2, 48000),
            Err(CompareError::LengthMismatch { reference: 480, test_scaled: 50 })
        );
        assert_eq!(
            opus_compare(Reference::Stereo(&short), &short, 2, 48000),
            Err(CompareError::TooShort { samples: 200 })
        );
        assert_eq!(
            CompareError::LengthMismatch { reference: 480, test_scaled: 50 }.to_string(),
            "Sample counts do not match (480!=50)."
        );
        assert_eq!(CompareError::TooShort { samples: 200 }.to_string(), "Insufficient sample data (200<480).");
    }
}
