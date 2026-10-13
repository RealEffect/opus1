//! Surround masking analysis for the multistream encoder
//! (libopus `surround_analysis` in opus_multistream_encoder.c, float build).
//!
//! For every input channel and every CELT band the analysis says how far the
//! channel lies above what the other channels mask. The result is in log2
//! units (1.0 is 6 dB), 21 values per channel. The multistream encoder hands
//! these values to the stream encoders as their energy mask, and CELT then
//! spends fewer bits on bands that other channels cover.
//!
//! libopus runs the analysis only for the surround mapping type, which
//! exists for 3 to 8 channels.

use super::ChannelSource;
use crate::bands::compute_band_energies;
use crate::modes::{self, CeltMode};
use crate::pitch::inner_prod;

/// Number of bands the analysis covers. The mask has this many values per
/// channel.
const NB_BANDS: usize = 21;
/// Longest frame the analysis accepts, in samples at 48 kHz (120 ms).
const MAX_FRAME_48K: usize = 5760;
/// libopus `CELT_SIG_SCALE`: CELT works on samples in the 16-bit range.
const CELT_SIG_SCALE: f32 = 32768.0;

/// libopus `resampling_factor`: how many samples at 48 kHz, the internal rate
/// of CELT, one input sample stands for. `None` for a rate CELT does not
/// take. (`celt::resampling_factor` is private.)
fn resampling_factor(sampling_rate: i32) -> Option<usize> {
    match sampling_rate {
        48000 => Some(1),
        24000 => Some(2),
        16000 => Some(3),
        12000 => Some(4),
        8000 => Some(6),
        _ => None,
    }
}

/// libopus `channel_pos`: where each channel of a surround layout sits in the
/// mix. 0: not mixed (the low-frequency-effects channel), 1: left,
/// 2: centre, 3: right.
fn channel_pos(channels: usize) -> [u8; 8] {
    match channels {
        4 => [1, 3, 1, 3, 0, 0, 0, 0],
        3 | 5 | 6 => [1, 2, 3, 1, 3, 0, 0, 0],
        7 => [1, 2, 3, 1, 3, 2, 0, 0],
        8 => [1, 2, 3, 1, 3, 1, 3, 0],
        _ => [0; 8],
    }
}

/// libopus `logSum`: the level of two signals of the levels `a` and `b`
/// added together. A level is the log2 of an amplitude, so the exact value
/// is `log2(sqrt(4^a + 4^b))`. This is a rough approximation of it.
#[allow(clippy::neg_cmp_op_on_partial_ord)] // the negated test also catches NaN, as in C
fn log_sum(a: f32, b: f32) -> f32 {
    // 0.5 * log2(1 + 4^-d) for d = 0, 0.5, 1, ... 4. The entries after that
    // are 0.
    const DIFF_TABLE: [f32; 17] = [
        0.5, 0.2924813, 0.160964, 0.0849625, 0.0437314, 0.0221971, 0.0111839, 0.0056136, 0.0028123,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ];
    let (max, diff) = if a > b { (a, a - b) } else { (b, b - a) };
    if !(diff < 8.0) {
        return max;
    }
    let low = (2.0 * diff).floor() as usize;
    let frac = 2.0 * diff - low as f32;
    max + DIFF_TABLE[low] + frac * (DIFF_TABLE[low + 1] - DIFF_TABLE[low])
}

/// libopus `celt_preemphasis` for one channel, without clipping.
///
/// `pcm` holds the input at the API rate with full scale 1.0. `out` receives
/// `pcm.len() * upsample` samples at 48 kHz in the 16-bit range: the input
/// with `upsample - 1` zeros after every sample, filtered by
/// `y[n] = x[n] - coef0 * x[n - 1]`. `mem` carries `coef0 * x[n - 1]` from
/// one frame to the next.
fn preemphasis(pcm: &[f32], out: &mut [f32], upsample: usize, coef0: f32, mem: &mut f32) {
    let mut m = *mem;
    if upsample == 1 {
        for (o, &s) in out.iter_mut().zip(pcm) {
            let x = CELT_SIG_SCALE * s;
            *o = x - m;
            m = coef0 * x;
        }
    } else {
        out.fill(0.0);
        for (o, &s) in out.iter_mut().step_by(upsample).zip(pcm) {
            *o = CELT_SIG_SCALE * s;
        }
        for o in out.iter_mut() {
            let x = *o;
            *o = x - m;
            m = coef0 * x;
        }
    }
    *mem = m;
}

/// State and working buffers of the surround masking analysis.
pub(crate) struct SurroundAnalysis {
    mode: &'static CeltMode,
    channels: usize,
    /// Samples at 48 kHz per input sample. 0 when the analysis cannot run
    /// for this channel count or sampling rate; `run` then returns `false`.
    upsample: usize,
    /// C `ms_get_window_mem`: the last `overlap` filtered samples of each
    /// channel. They are the start of the next frame's first MDCT window.
    window_mem: Vec<f32>,
    /// C `ms_get_preemph_mem`: the pre-emphasis filter memory of each channel.
    preemph_mem: Vec<f32>,
    /// One channel of the input frame, at the API rate (C `x`).
    pcm: Vec<f32>,
    /// `window_mem` of the channel followed by the filtered frame (C `in`).
    input: Vec<f32>,
    /// MDCT coefficients of one channel (C `freq`).
    freq: Vec<f32>,
}

impl SurroundAnalysis {
    pub(crate) fn new(channels: usize, sampling_rate: i32) -> Self {
        let mode = modes::default_mode();
        // libopus only has the surround mapping type, and so only runs the
        // analysis, for 3 to 8 channels.
        let upsample = match resampling_factor(sampling_rate) {
            Some(upsample) if (3..=8).contains(&channels) => upsample,
            _ => 0,
        };
        let enabled = usize::from(upsample != 0);
        Self {
            mode,
            channels,
            upsample,
            window_mem: vec![0.0; enabled * channels * mode.overlap],
            preemph_mem: vec![0.0; enabled * channels],
            pcm: vec![0.0; enabled * MAX_FRAME_48K],
            input: vec![0.0; enabled * (MAX_FRAME_48K + mode.overlap)],
            freq: vec![0.0; enabled * (mode.short_mdct_size << mode.max_lm)],
        }
    }

    /// Clears the state carried from frame to frame.
    pub(crate) fn reset(&mut self) {
        self.window_mem.fill(0.0);
        self.preemph_mem.fill(0.0);
    }

    /// Fills `band_smr[..21 * channels]` with the mask of each channel and
    /// returns `true`. Returns `false` and writes nothing when no mask is
    /// available.
    ///
    /// `frame_size` is in samples per channel at the sampling rate given to
    /// `new`. `src` supplies `frame_size` samples of each channel with full
    /// scale 1.0.
    ///
    /// No mask is available when the channel count is not 3 to 8, when the
    /// sampling rate is not 8, 12, 16, 24 or 48 kHz, when the frame is not
    /// 2.5, 5, 10 or 20 ms or a multiple of 20 ms up to 120 ms, or when
    /// `band_smr` is too short.
    #[allow(clippy::neg_cmp_op_on_partial_ord)] // the negated test also catches NaN, as in C
    pub(crate) fn run<S: ChannelSource>(
        &mut self,
        src: &S,
        frame_size: usize,
        band_smr: &mut [f32],
    ) -> bool {
        let mode = self.mode;
        let channels = self.channels;
        let upsample = self.upsample;
        let overlap = mode.overlap;
        if upsample == 0 || frame_size > MAX_FRAME_48K / upsample {
            return false;
        }
        let Some(band_smr) = band_smr.get_mut(..NB_BANDS * channels) else {
            return false;
        };
        // The frame at 48 kHz.
        let n = frame_size * upsample;

        // C 252-257: LM = log2(n / 120). A frame that is not 2.5, 5 or 10 ms
        // is analysed in parts of 20 ms.
        let mut lm = 0;
        while lm < mode.max_lm && (mode.short_mdct_size << lm) != n {
            lm += 1;
        }
        let freq_size = mode.short_mdct_size << lm;
        let nb_frames = n / freq_size;
        if nb_frames == 0 || nb_frames * freq_size != n {
            return false;
        }

        let pos = channel_pos(channels);
        // What the left, centre and right of the mix hold in each band.
        let mut mask_log_e = [[-28.0f32; NB_BANDS]; 3];

        for c in 0..channels {
            let input = &mut self.input[..n + overlap];
            input[..overlap].copy_from_slice(&self.window_mem[c * overlap..(c + 1) * overlap]);
            let pcm = &mut self.pcm[..frame_size];
            src.copy(pcm, 1, c, frame_size);
            preemphasis(
                pcm,
                &mut input[overlap..],
                upsample,
                mode.preemph[0],
                &mut self.preemph_mem[c],
            );
            // C 277-289: drop a channel that holds NaN or absurdly large
            // values, which would give NaN further down.
            let sum = inner_prod(input, input, n + overlap);
            if !(sum < 1e18) || sum.is_nan() {
                input.fill(0.0);
                self.preemph_mem[c] = 0.0;
            }

            // Band amplitudes. With several MDCTs in the frame, the largest
            // amplitude of each band counts.
            let mut band_e = [0.0f32; NB_BANDS];
            let freq = &mut self.freq[..freq_size];
            for frame in 0..nb_frames {
                let mut tmp_e = [0.0f32; NB_BANDS];
                mode.mdct.forward(
                    &input[freq_size * frame..],
                    freq,
                    mode.window,
                    overlap,
                    mode.max_lm - lm,
                    1,
                );
                if upsample != 1 {
                    // Only the part of the spectrum below the input's
                    // Nyquist frequency holds signal. The zeros inserted
                    // between the samples divided its level by `upsample`.
                    let bound = freq_size / upsample;
                    for v in &mut freq[..bound] {
                        *v *= upsample as f32;
                    }
                    freq[bound..].fill(0.0);
                }
                compute_band_energies(mode, freq, &mut tmp_e, NB_BANDS, 1, lm);
                for (e, &t) in band_e.iter_mut().zip(&tmp_e) {
                    if t > *e {
                        *e = t;
                    }
                }
            }

            // libopus `amp2Log2(mode, 21, 21, ..)`: log2 of the amplitude
            // above the mean level of the band. `bands::amp2log2` is not
            // used because it limits the amplitude to 1e-10 from below,
            // which libopus does not do.
            let band_log_e = &mut band_smr[NB_BANDS * c..NB_BANDS * (c + 1)];
            for i in 0..NB_BANDS {
                band_log_e[i] = band_e[i].log2() - mode.e_means[i];
            }
            // Masking spreads to the neighbouring bands: it falls by 6 dB
            // per band going up and by 12 dB per band going down.
            for i in 1..NB_BANDS {
                let spread = band_log_e[i - 1] - 1.0;
                if spread > band_log_e[i] {
                    band_log_e[i] = spread;
                }
            }
            for i in (0..NB_BANDS - 1).rev() {
                let spread = band_log_e[i + 1] - 2.0;
                if spread > band_log_e[i] {
                    band_log_e[i] = spread;
                }
            }
            // Add the channel to the side of the mix it plays on. A centre
            // channel counts half (3 dB less) on both sides.
            match pos[c] {
                1 => {
                    for i in 0..NB_BANDS {
                        mask_log_e[0][i] = log_sum(mask_log_e[0][i], band_log_e[i]);
                    }
                }
                3 => {
                    for i in 0..NB_BANDS {
                        mask_log_e[2][i] = log_sum(mask_log_e[2][i], band_log_e[i]);
                    }
                }
                2 => {
                    for i in 0..NB_BANDS {
                        mask_log_e[0][i] = log_sum(mask_log_e[0][i], band_log_e[i] - 0.5);
                        mask_log_e[2][i] = log_sum(mask_log_e[2][i], band_log_e[i] - 0.5);
                    }
                }
                _ => {}
            }
            self.window_mem[c * overlap..(c + 1) * overlap].copy_from_slice(&input[n..]);
        }

        // The centre is masked by the quieter of the two sides.
        for i in 0..NB_BANDS {
            mask_log_e[1][i] = if mask_log_e[0][i] < mask_log_e[2][i] {
                mask_log_e[0][i]
            } else {
                mask_log_e[2][i]
            };
        }
        let channel_offset = 0.5 * (2.0 / (channels - 1) as f32).log2();
        for mask in &mut mask_log_e {
            for v in mask.iter_mut() {
                *v += channel_offset;
            }
        }

        // What is left is the level of each channel above the mask of its
        // position. A channel that is not part of the mix gets 0.
        for c in 0..channels {
            let band_log_e = &mut band_smr[NB_BANDS * c..NB_BANDS * (c + 1)];
            if pos[c] != 0 {
                let mask = &mask_log_e[pos[c] as usize - 1];
                for i in 0..NB_BANDS {
                    band_log_e[i] -= mask[i];
                }
            } else {
                band_log_e.fill(0.0);
            }
        }
        true
    }
}

/// Input for [`SurroundAnalysis`] from interleaved samples: sample `i` of
/// channel `c` is `pcm[i * channels + c]`.
struct Interleaved<'a> {
    pcm: &'a [f32],
    channels: usize,
}

impl ChannelSource for Interleaved<'_> {
    fn copy(&self, dst: &mut [f32], dst_stride: usize, src_channel: usize, frame_size: usize) {
        for i in 0..frame_size {
            dst[i * dst_stride] = self.pcm[i * self.channels + src_channel];
        }
    }
}

/// The surround masking analysis on interleaved samples. It exists for the
/// tests in `tests/`, which cannot reach the crate-internal
/// `SurroundAnalysis`. It is not part of the supported API.
#[doc(hidden)]
pub struct SurroundProbe {
    analysis: SurroundAnalysis,
}

impl SurroundProbe {
    /// A new analysis for `channels` input channels at `sampling_rate` Hz.
    pub fn new(channels: usize, sampling_rate: i32) -> Self {
        Self {
            analysis: SurroundAnalysis::new(channels, sampling_rate),
        }
    }

    /// Clears the state carried from frame to frame.
    pub fn reset(&mut self) {
        self.analysis.reset();
    }

    /// Analyses one frame. `pcm` holds `frame_size` samples per channel,
    /// interleaved, with full scale 1.0. Fills `band_smr[..21 * channels]`
    /// and returns `true`, or returns `false` and writes nothing when the
    /// analysis cannot run or `pcm` is too short.
    pub fn run(&mut self, pcm: &[f32], frame_size: usize, band_smr: &mut [f32]) -> bool {
        let channels = self.analysis.channels;
        match frame_size.checked_mul(channels) {
            Some(len) if pcm.len() >= len => {}
            _ => return false,
        }
        self.analysis
            .run(&Interleaved { pcm, channels }, frame_size, band_smr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_sum_approximates_the_level_of_a_sum() {
        // The table has steps of 0.5, with straight lines between them: the
        // error stays below 0.012.
        let mut worst = 0f64;
        for i in 0..=1700 {
            let d = i as f32 * 0.005;
            for &(a, b) in &[(3.0f32, 3.0 - d), (-7.5 - d, -7.5), (d, 0.0)] {
                let exact = 0.5 * (4f64.powf(a as f64) + 4f64.powf(b as f64)).log2();
                worst = worst.max((log_sum(a, b) as f64 - exact).abs());
                assert_eq!(log_sum(a, b), log_sum(b, a));
            }
        }
        assert!(worst < 0.012, "largest error {worst}");
        // Equal levels add 3 dB, which is 0.5.
        assert_eq!(log_sum(2.0, 2.0), 2.5);
        // 8 or more apart: the larger one, exactly.
        assert_eq!(log_sum(-28.0, -20.0), -20.0);
        assert_eq!(log_sum(1.0, -51.0), 1.0);
        // Just below 8 the last table entries are 0, so nothing is added.
        assert_eq!(log_sum(0.0, -7.99), 0.0);
        // libopus returns the second argument when one of them is NaN.
        assert_eq!(log_sum(f32::NAN, 1.0), 1.0);
        assert!(log_sum(1.0, f32::NAN).is_nan());
    }

    #[test]
    fn channel_positions_are_those_of_libopus() {
        assert_eq!(channel_pos(3)[..3], [1, 2, 3]);
        assert_eq!(channel_pos(4)[..4], [1, 3, 1, 3]);
        assert_eq!(channel_pos(5)[..5], [1, 2, 3, 1, 3]);
        assert_eq!(channel_pos(6)[..6], [1, 2, 3, 1, 3, 0]);
        assert_eq!(channel_pos(7)[..7], [1, 2, 3, 1, 3, 2, 0]);
        assert_eq!(channel_pos(8)[..8], [1, 2, 3, 1, 3, 1, 3, 0]);
        for channels in [0, 1, 2, 9, 255] {
            assert_eq!(channel_pos(channels), [0; 8]);
        }
    }

    #[test]
    fn preemphasis_inserts_zeros_and_keeps_its_memory() {
        let coef = 0.5;
        // 48 kHz: y[n] = x[n] - coef * x[n - 1], on samples times 32768.
        let mut mem = 0.0;
        let mut out = [0f32; 3];
        preemphasis(&[1.0, 1.0, -1.0], &mut out, 1, coef, &mut mem);
        assert_eq!(out, [32768.0, 16384.0, -49152.0]);
        assert_eq!(mem, -16384.0);
        // 16 kHz: two zeros after every sample. The filter runs over the
        // zeros too, so each sample is followed by its echo.
        let mut mem = 0.0;
        let mut out = [9f32; 6];
        preemphasis(&[1.0, -1.0], &mut out, 3, coef, &mut mem);
        assert_eq!(out, [32768.0, -16384.0, 0.0, -32768.0, 16384.0, 0.0]);
        assert_eq!(mem, 0.0);
        // The memory carries over: one call for two frames gives the same.
        let (mut a, mut b) = ([0f32; 4], [0f32; 4]);
        let mut mem = 0.0;
        preemphasis(&[0.25, 0.5, -0.75, 1.0], &mut a, 1, coef, &mut mem);
        let mut mem = 0.0;
        preemphasis(&[0.25, 0.5], &mut b[..2], 1, coef, &mut mem);
        preemphasis(&[-0.75, 1.0], &mut b[2..], 1, coef, &mut mem);
        assert_eq!(a, b);
    }

    /// Interleaved input that notes every call of `copy`: the length of the
    /// destination, the stride, the channel and the frame size.
    struct Recording<'a> {
        input: Interleaved<'a>,
        calls: std::cell::RefCell<Vec<(usize, usize, usize, usize)>>,
    }

    impl ChannelSource for Recording<'_> {
        fn copy(&self, dst: &mut [f32], dst_stride: usize, src_channel: usize, frame_size: usize) {
            self.calls
                .borrow_mut()
                .push((dst.len(), dst_stride, src_channel, frame_size));
            self.input.copy(dst, dst_stride, src_channel, frame_size);
        }
    }

    /// Input with the channels one after the other: sample `i` of channel
    /// `c` is `pcm[c * frame_size + i]`.
    struct Planar<'a> {
        pcm: &'a [f32],
        frame_size: usize,
    }

    impl ChannelSource for Planar<'_> {
        fn copy(&self, dst: &mut [f32], dst_stride: usize, src_channel: usize, frame_size: usize) {
            for i in 0..frame_size {
                dst[i * dst_stride] = self.pcm[src_channel * self.frame_size + i];
            }
        }
    }

    /// Samples between -0.25 and 0.25 from a small generator.
    fn test_signal(len: usize) -> Vec<f32> {
        let mut state = 0x1234_5678u32;
        (0..len)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (state >> 8) as f32 / (1u32 << 25) as f32 - 0.25
            })
            .collect()
    }

    #[test]
    fn run_asks_the_source_for_every_channel_once() {
        // The multistream encoder writes its sources against this: one call
        // per input channel, in channel order, with room for exactly the
        // frame and a stride of 1. A call that gives no mask asks for
        // nothing.
        for (channels, rate, frame_size) in
            [(6usize, 48_000, 960usize), (8, 16_000, 40), (3, 8_000, 960)]
        {
            let pcm = test_signal(frame_size * channels);
            let src = Recording {
                input: Interleaved {
                    pcm: &pcm,
                    channels,
                },
                calls: Default::default(),
            };
            let mut analysis = SurroundAnalysis::new(channels, rate);
            let mut mask = vec![0f32; NB_BANDS * channels];
            assert!(analysis.run(&src, frame_size, &mut mask));
            let want: Vec<_> = (0..channels)
                .map(|c| (frame_size, 1, c, frame_size))
                .collect();
            assert_eq!(*src.calls.borrow(), want);

            src.calls.borrow_mut().clear();
            assert!(!analysis.run(&src, frame_size + 1, &mut mask));
            assert!(!analysis.run(&src, 0, &mut mask));
            assert!(!analysis.run(&src, frame_size, &mut mask[1..]));
            assert!(src.calls.borrow().is_empty());
        }
        // An analysis that cannot run never asks.
        let src = Recording {
            input: Interleaved {
                pcm: &[],
                channels: 2,
            },
            calls: Default::default(),
        };
        let mut mask = [0f32; 2 * NB_BANDS];
        assert!(!SurroundAnalysis::new(2, 48_000).run(&src, 960, &mut mask));
        assert!(!SurroundAnalysis::new(6, 44_100).run(&src, 882, &mut mask));
        assert!(src.calls.borrow().is_empty());
    }

    #[test]
    fn the_mask_does_not_depend_on_how_the_source_stores_the_samples() {
        let (channels, rate, frame_size) = (7usize, 24_000, 480usize);
        let interleaved = test_signal(2 * frame_size * channels);
        let mut a = SurroundAnalysis::new(channels, rate);
        let mut b = SurroundAnalysis::new(channels, rate);
        for frame in interleaved.chunks_exact(frame_size * channels) {
            let mut planar = vec![0f32; frame.len()];
            for (i, x) in frame.chunks_exact(channels).enumerate() {
                for (c, &v) in x.iter().enumerate() {
                    planar[c * frame_size + i] = v;
                }
            }
            let mut mask_a = vec![0f32; NB_BANDS * channels];
            let mut mask_b = vec![1f32; NB_BANDS * channels];
            assert!(a.run(
                &Interleaved {
                    pcm: frame,
                    channels
                },
                frame_size,
                &mut mask_a
            ));
            assert!(b.run(
                &Planar {
                    pcm: &planar,
                    frame_size
                },
                frame_size,
                &mut mask_b
            ));
            assert_eq!(mask_a, mask_b);
        }
    }
}
