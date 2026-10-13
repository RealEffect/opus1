//! Surround masking analysis for the multistream encoder
//! (libopus `surround_analysis` in opus_multistream_encoder.c).
//!
//! This is the interface only. `run` returns `false` until the analysis is
//! ported; the encoder then sets no energy mask.

use super::ChannelSource;

/// Per-channel state of the surround masking analysis.
#[allow(dead_code)] // used once the multistream encoder is in
pub(crate) struct SurroundAnalysis {
    _channels: usize,
    _sampling_rate: i32,
}

#[allow(dead_code)]
impl SurroundAnalysis {
    pub(crate) fn new(channels: usize, sampling_rate: i32) -> Self {
        Self { _channels: channels, _sampling_rate: sampling_rate }
    }

    /// Clears the state carried from frame to frame.
    pub(crate) fn reset(&mut self) {}

    /// Fills `band_smr[..21 * channels]` with the mask of each channel and
    /// returns `true`. Returns `false` and writes nothing when no mask is
    /// available.
    pub(crate) fn run<S: ChannelSource>(
        &mut self,
        _src: &S,
        _frame_size: usize,
        _band_smr: &mut [f32],
    ) -> bool {
        false
    }
}
