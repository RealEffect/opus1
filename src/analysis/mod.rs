//! The speech and music analysis of the libopus encoder (`src/analysis.c`
//! and `src/mlp.c` of libopus 1.6.1).
//!
//! For every 20 ms of input, the analysis estimates:
//!
//! - how tonal the signal is,
//! - how likely it is music rather than speech,
//! - how likely it holds any sound at all (its activity),
//! - which audio bandwidth it really uses,
//! - how much each band leaks into its neighbours.
//!
//! The float build of libopus runs it from complexity 7 on, for input at
//! 16, 24 or 48 kHz. The encoder reads the result to choose the mode and the
//! bandwidth, and the CELT encoder reads it for its bit allocation.
//!
//! The types of this file always exist, so that the encoder can carry an
//! [`AnalysisInfo`] without the analysis. The analysis itself needs the
//! cargo feature `analysis` (on by default). Without it, every
//! `AnalysisInfo` stays invalid and the encoder behaves as if the
//! complexity were below 7.
//!
//! The arithmetic follows the libopus 1.6.1 library of Homebrew on 64-bit
//! ARM, which behaves as Apple clang at `-Os`: the same products are fused
//! into one rounding (see [`crate::mathops::mac`]), and the same sums run in
//! the same order. The results are then the same bits in every field. On
//! other processors the products are rounded separately, as libopus builds
//! without fused multiply-add round them.
//!
//! The busiest loops have vector versions: NEON on 64-bit ARM (`neon.rs`),
//! SSE2 on x86-64 (`x86.rs`). Both give the bits of the scalar code of
//! their processor.

#[cfg(feature = "analysis")]
pub mod mlp;
#[cfg(feature = "analysis")]
pub mod mlp_data;
#[cfg(all(feature = "analysis", target_arch = "aarch64"))]
mod neon;
#[cfg(feature = "analysis")]
mod tables;
#[cfg(feature = "analysis")]
mod tonality;
#[cfg(all(feature = "analysis", target_arch = "x86_64", target_feature = "sse2"))]
mod x86;

#[cfg(feature = "analysis")]
pub use tonality::{
    ANALYSIS_BUF_SIZE, ANALYSIS_COUNT_MAX, DETECT_SIZE, NB_FRAMES, NB_TBANDS,
    TonalityAnalysisState, fast_atan2f,
};

/// Number of bands of `AnalysisInfo::leak_boost` (libopus `LEAK_BANDS`,
/// celt/celt.h).
pub const LEAK_BANDS: usize = 19;

/// The result of the analysis for one frame (libopus `AnalysisInfo`,
/// celt/celt.h:65-79).
///
/// Every reader checks `valid` first. The other fields hold the estimates:
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct AnalysisInfo {
    /// The analysis ran on this frame and its numbers can be used.
    pub valid: bool,
    /// How tonal the signal is, 0 to about 1.
    pub tonality: f32,
    /// Whether the tonality sits in the high bands (positive) or the low
    /// bands (negative).
    pub tonality_slope: f32,
    /// How noise-like the phase of the spectrum is.
    pub noisiness: f32,
    /// A measure of the activity from the noisiness and the band energies.
    pub activity: f32,
    /// Probability that the signal is music, 0 to 1.
    pub music_prob: f32,
    /// The threshold for a switch from speech to music (see
    /// `tonality_get_info` of libopus).
    pub music_prob_min: f32,
    /// The threshold for a switch from music to speech.
    pub music_prob_max: f32,
    /// The highest band that holds signal: 0 to 20, in the bands of the
    /// analysis, which end at the CELT band edges. 20 means fullband.
    pub bandwidth: i32,
    /// Probability that the frame holds speech or music rather than noise
    /// or silence, 0 to 1.
    pub activity_probability: f32,
    /// The energy below 6 kHz over the energy above it, at most 1. The CELT
    /// pre-filter scales its gain by it.
    pub max_pitch_ratio: f32,
    /// The extra precision each band needs because of the leakage between
    /// bands, in steps of 1/64 (libopus stores it in Q6).
    pub leak_boost: [u8; LEAK_BANDS],
}

/// The caller's samples, as libopus hands `analysis_pcm` and a downmix
/// function to `run_analysis`.
#[derive(Clone, Copy, Debug)]
pub enum PcmRef<'a> {
    /// Float samples in -1 to 1 (libopus `downmix_float`): scaled by 32768;
    /// the sum of the channels is then limited to ±65536, and a NaN becomes
    /// 0.
    F32(&'a [f32]),
    /// 16-bit samples (libopus `downmix_int`): converted to `f32` and
    /// summed, without a limit.
    I16(&'a [i16]),
    /// 24-bit samples in 32-bit integers (libopus `downmix_int24`):
    /// converted to `f32`, divided by 256 and summed, without a limit.
    I32(&'a [i32]),
}

/// Which channels the analysis adds to channel `c1` (libopus `c2`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SecondChannel {
    /// C `c2 == -1`: channel `c1` alone (a mono stream of a multistream
    /// encoder).
    None,
    /// C `c2 >= 0`: channel `c1` plus this channel, halved (a coupled
    /// stream of a multistream encoder).
    Channel(usize),
    /// C `c2 == -2`: channel `c1` plus the channels 1 to `channels - 1`
    /// (`OpusEncoder`, where `c1` is 0, so all channels). The sum is halved
    /// when there are exactly two channels.
    AllOthers,
}

/// The input of one call of the analysis.
#[derive(Clone, Copy, Debug)]
pub struct AnalysisPcm<'a> {
    /// The interleaved samples.
    pub pcm: PcmRef<'a>,
    /// C `C`: the channels of the interleaved buffer.
    pub channels: usize,
    /// C `c1`: the first channel the analysis reads.
    pub c1: usize,
    /// C `c2`: the channels the analysis adds to `c1`.
    pub c2: SecondChannel,
    /// C `analysis_frame_size`: the samples per channel of the caller's
    /// buffer. It can be longer than the frame that is coded, when the
    /// caller sets an expert frame duration. The buffer must hold at least
    /// `frame_size * channels` samples.
    pub frame_size: usize,
}

/// When the encoder runs the analysis. libopus has no such control: its
/// float build runs the analysis from complexity 7 on (`FloatBuild`), its
/// fixed-point build from complexity 10 on (`FixedPointBuild`). Both run it
/// only for input at 16, 24 or 48 kHz.
///
/// This is not part of the public interface. Tests that compare opus1
/// with a fixed-point build of libopus, and evaluations that measure what
/// the analysis is worth, set it through `opus1::internals::analysis`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AnalysisGate {
    /// From complexity 7 on, as the float build of libopus. The default.
    #[default]
    FloatBuild,
    /// At complexity 10 only, as the fixed-point build of libopus. opus1
    /// still runs the float analysis there.
    FixedPointBuild,
    /// Never: the encoder decides as libopus does below complexity 7.
    Off,
}

impl AnalysisGate {
    /// The lowest complexity at which the analysis runs, or `None` when it
    /// never runs.
    #[cfg_attr(not(feature = "analysis"), allow(dead_code))]
    pub(crate) fn min_complexity(self) -> Option<i32> {
        match self {
            AnalysisGate::FloatBuild => Some(7),
            AnalysisGate::FixedPointBuild => Some(10),
            AnalysisGate::Off => None,
        }
    }
}

/// libopus `is_digital_silence` of the float build
/// (src/opus_encoder.c:1060-1077): whether no sample of `pcm` has a
/// magnitude above `2^-lsb_depth`.
///
/// libopus finds the largest and the smallest sample with the ternaries of
/// `celt_maxabs16` (celt/mathops.h), and the libopus build keeps them as one
/// scalar loop. The ternaries treat a NaN in their own way: `max > x ? max
/// : x` takes a NaN, and the next sample replaces it. A NaN thus hides the
/// samples before it, and only a NaN in the last place makes the result NaN
/// (not silence). A search for the largest magnitude that skips NaN would
/// give another answer there.
///
/// Without a NaN, the answer depends only on the largest magnitude, which a
/// vector search finds faster. So on 64-bit ARM and x86-64 a vector search
/// runs first, and the loop of libopus runs only when the input holds a
/// NaN.
pub fn is_digital_silence(pcm: &[f32], lsb_depth: i32) -> bool {
    let threshold = 1.0 / (1i32 << lsb_depth) as f32;
    #[cfg(any(target_arch = "aarch64", all(target_arch = "x86_64", target_feature = "sse2")))]
    if let Some(max_abs) = max_abs_without_nan(pcm) {
        return max_abs <= threshold;
    }
    is_digital_silence_scalar(pcm, threshold)
}

/// `is_digital_silence` as libopus computes it, one sample at a time.
fn is_digital_silence_scalar(pcm: &[f32], threshold: f32) -> bool {
    let mut max = 0.0f32;
    let mut min = 0.0f32;
    for &x in pcm {
        max = if max > x { max } else { x };
        min = if min < x { min } else { x };
    }
    let neg_min = -min;
    let max_abs = if max > neg_min { max } else { neg_min };
    max_abs <= threshold
}

/// The largest magnitude in `pcm` (at least 0), or `None` when `pcm` holds a
/// NaN.
#[cfg(target_arch = "aarch64")]
fn max_abs_without_nan(pcm: &[f32]) -> Option<f32> {
    use core::arch::aarch64::*;
    let (blocks, tail) = pcm.as_chunks::<8>();
    // SAFETY: NEON is part of every 64-bit ARM processor; each block holds
    // 8 values, read as two vectors of 4.
    let (mut max, mut min) = unsafe {
        let zero = vdupq_n_f32(0.0);
        let (mut max0, mut max1, mut min0, mut min1) = (zero, zero, zero, zero);
        for b in blocks {
            let a = vld1q_f32(b.as_ptr());
            let c = vld1q_f32(b.as_ptr().add(4));
            // FMAX and FMIN give a NaN when either input is a NaN.
            max0 = vmaxq_f32(max0, a);
            max1 = vmaxq_f32(max1, c);
            min0 = vminq_f32(min0, a);
            min1 = vminq_f32(min1, c);
        }
        (vmaxvq_f32(vmaxq_f32(max0, max1)), vminvq_f32(vminq_f32(min0, min1)))
    };
    for &x in tail {
        max = if x > max || x.is_nan() { x } else { max };
        min = if x < min { x } else { min };
    }
    if max.is_nan() || min.is_nan() {
        return None;
    }
    Some(if max > -min { max } else { -min })
}

/// The largest magnitude in `pcm` (at least 0), or `None` when `pcm` holds a
/// NaN. The SSE version of the function above: it keeps the largest
/// magnitude instead of the largest and the smallest value, which gives the
/// same answer, and a mask of the lanes that saw a NaN.
#[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
fn max_abs_without_nan(pcm: &[f32]) -> Option<f32> {
    use core::arch::x86_64::*;
    let (blocks, tail) = pcm.as_chunks::<8>();
    // SAFETY: SSE is part of this target (see the `cfg`); each block holds
    // 8 values, read as two vectors of 4.
    let (max, nan) = unsafe {
        let sign = _mm_set1_ps(-0.0);
        let zero = _mm_setzero_ps();
        let (mut max0, mut max1, mut nan0, mut nan1) = (zero, zero, zero, zero);
        for b in blocks {
            let a = _mm_loadu_ps(b.as_ptr());
            let c = _mm_loadu_ps(b.as_ptr().add(4));
            // `_mm_max_ps` skips a NaN in its first operand, so the NaN
            // test is separate: a NaN is unordered with itself.
            max0 = _mm_max_ps(_mm_andnot_ps(sign, a), max0);
            max1 = _mm_max_ps(_mm_andnot_ps(sign, c), max1);
            nan0 = _mm_or_ps(nan0, _mm_cmpunord_ps(a, a));
            nan1 = _mm_or_ps(nan1, _mm_cmpunord_ps(c, c));
        }
        let max = core::mem::transmute::<__m128, [f32; 4]>(_mm_max_ps(max0, max1));
        (max, _mm_movemask_ps(_mm_or_ps(nan0, nan1)) != 0)
    };
    if nan || tail.iter().any(|x| x.is_nan()) {
        return None;
    }
    let max = max.into_iter().chain(tail.iter().map(|x| x.abs())).fold(0.0f32, f32::max);
    Some(max)
}

#[cfg(all(
    test,
    feature = "std",
    any(target_arch = "aarch64", all(target_arch = "x86_64", target_feature = "sse2"))
))]
mod tests {
    use super::*;

    /// The vector search gives the answer of the loop of libopus whenever
    /// it gives one, and declines exactly when a NaN is present.
    #[test]
    fn vector_silence_test_matches_the_scalar_loop() {
        let mut seed = 0x1234_5678_9abc_def0u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let specials = [0.0, -0.0, f32::INFINITY, f32::NEG_INFINITY, f32::NAN, 1e-40, -1e-40, f32::MAX];
        for len in 0..80 {
            for _ in 0..200 {
                let lsb = 8 + (next() % 17) as i32;
                let threshold = 1.0 / (1i32 << lsb) as f32;
                let pcm: Vec<f32> = (0..len)
                    .map(|_| match next() % 16 {
                        0 => specials[(next() % specials.len() as u64) as usize],
                        1 => threshold,
                        2 => -threshold,
                        3 => f32::from_bits(threshold.to_bits() + 1),
                        _ => {
                            let scale = threshold * 4.0;
                            ((next() >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * scale
                        }
                    })
                    .collect();
                let scalar = is_digital_silence_scalar(&pcm, threshold);
                match max_abs_without_nan(&pcm) {
                    Some(m) => {
                        assert!(!pcm.iter().any(|x| x.is_nan()));
                        assert_eq!(m <= threshold, scalar, "{pcm:?} lsb {lsb}");
                    }
                    None => assert!(pcm.iter().any(|x| x.is_nan())),
                }
                assert_eq!(is_digital_silence(&pcm, lsb), scalar, "{pcm:?} lsb {lsb}");
            }
        }
    }
}
