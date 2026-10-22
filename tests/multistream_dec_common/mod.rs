//! Shared helpers of the `multistream_dec_*` tests.
//!
//! Each test encodes a signal with the C libopus multistream encoder and
//! decodes the packets twice: with the C multistream decoder and with
//! `opus1::multistream::MultistreamDecoder`. The helpers here build the
//! encoder, describe the layouts under test, and compare two decoded signals.
//!
//! Use with `mod ms_support; mod multistream_dec_common;`.

#![allow(dead_code)]

use crate::ms_support::*;
use opus1::SampleRate;
use opus1::multistream::MultistreamDecoder;

/// Sampling rate of every encoder in these tests.
pub const FS: i32 = 48_000;

/// A layout under test.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layout {
    /// The layout libopus picks for a channel count and a mapping family.
    /// The decoders use the same layout.
    Family { channels: usize, family: i32 },
    /// A hand-made layout. The encoder has 3 input channels: a stereo
    /// stream (channels 0 and 1) and a mono stream (channel 2). The decoders
    /// have 5 output channels with the mapping `[2, 0, 255, 1, 2]`: the mono
    /// stream feeds outputs 0 and 4, and output 2 is silent.
    Generic,
}

/// Mapping of the decoders for [`Layout::Generic`].
pub const GENERIC_DECODER_MAPPING: [u8; 5] = [2, 0, 255, 1, 2];

pub const MONO: Layout = Layout::Family { channels: 1, family: 1 };
pub const STEREO: Layout = Layout::Family { channels: 2, family: 1 };
pub const QUAD: Layout = Layout::Family { channels: 4, family: 1 };
pub const SURROUND_5_1: Layout = Layout::Family { channels: 6, family: 1 };
pub const SURROUND_7_1: Layout = Layout::Family { channels: 8, family: 1 };
pub const DISCRETE_3: Layout = Layout::Family { channels: 3, family: 255 };
pub const DISCRETE_5: Layout = Layout::Family { channels: 5, family: 255 };
pub const AMBISONICS_4: Layout = Layout::Family { channels: 4, family: 2 };
pub const AMBISONICS_6: Layout = Layout::Family { channels: 6, family: 2 };

/// Every layout of the tests.
pub const ALL_LAYOUTS: [Layout; 10] = [
    MONO,
    STEREO,
    QUAD,
    SURROUND_5_1,
    SURROUND_7_1,
    DISCRETE_3,
    DISCRETE_5,
    AMBISONICS_4,
    AMBISONICS_6,
    Layout::Generic,
];

/// Test signal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Signal {
    /// One steady sine per channel (`ms_support::tones`).
    Tones,
    /// A speech-like signal with a pitch per channel (`ms_support::voiced`).
    Voiced,
}

/// Settings of the C encoder.
#[derive(Clone, Copy, Debug)]
pub struct Settings {
    /// `APP_AUDIO` or `APP_VOIP`.
    pub application: i32,
    /// In-band forward error correction with 20 % expected loss.
    pub fec: bool,
    /// Variable bitrate. `false` is constant bitrate, which pads packets.
    pub vbr: bool,
}

/// Music settings: the encoder mostly picks CELT.
pub const AUDIO: Settings = Settings { application: APP_AUDIO, fec: false, vbr: true };
/// Speech settings with error correction: the encoder picks SILK and Hybrid
/// at low and middle bitrates.
pub const VOIP_FEC: Settings = Settings { application: APP_VOIP, fec: true, vbr: true };
/// Constant bitrate: packets carry padding.
pub const AUDIO_CBR: Settings = Settings { application: APP_AUDIO, fec: false, vbr: false };

/// The packets of one encoded signal, and the layout to decode them with.
pub struct Encoded {
    /// Output channels of the decoders.
    pub channels: usize,
    pub streams: usize,
    pub coupled: usize,
    /// Mapping of the decoders, one entry per output channel.
    pub mapping: Vec<u8>,
    /// Mapping family for `ms_support::tone_hz`.
    pub family: i32,
    /// Input channels of the encoder.
    pub input_channels: usize,
    pub packets: Vec<Vec<u8>>,
    /// The encoder's final range after each packet.
    pub ranges: Vec<u32>,
    /// What was encoded, for failure messages.
    pub label: String,
}

impl Encoded {
    /// Whether a stereo stream of some packet is coded as mono. libopus does
    /// that at low bitrates. A decoder has to turn such a packet into two
    /// channels.
    pub fn has_mono_coded_stereo_stream(&self) -> bool {
        self.packets.iter().any(|p| {
            let parts = split_streams(p, self.streams).expect("the C encoder writes valid packets");
            parts[..self.coupled].iter().any(|s| !toc(s.toc).stereo)
        })
    }

    /// A C decoder and an opus1 decoder for these packets at `fs`.
    pub fn decoders(&self, fs: i32) -> (CMsDecoder, MultistreamDecoder) {
        let c = CMsDecoder::new(fs, self.channels, self.streams, self.coupled, &self.mapping)
            .unwrap_or_else(|e| panic!("{}: C decoder: {e}", self.label));
        let r = MultistreamDecoder::with_layout(
            SampleRate::try_from(fs).unwrap(),
            self.channels,
            self.streams,
            self.coupled,
            &self.mapping,
        )
        .unwrap_or_else(|e| panic!("{}: opus1 decoder: {e}", self.label));
        (c, r)
    }
}

/// The C encoder for `layout`, and the mapping family for the test tones.
pub fn c_encoder(layout: Layout, settings: Settings) -> (CMsEncoder, i32) {
    let (mut enc, family) = match layout {
        Layout::Family { channels, family } => (
            CMsEncoder::surround(FS, channels, family, settings.application).expect("C encoder"),
            family,
        ),
        Layout::Generic => (
            CMsEncoder::generic(FS, 3, 2, 1, &[0, 1, 2], settings.application).expect("C encoder"),
            255,
        ),
    };
    if settings.fec {
        enc.set_inband_fec(true).set_packet_loss_perc(20);
    }
    enc.set_vbr(settings.vbr);
    (enc, family)
}

/// The input signal of `encode`: `samples` samples per channel.
pub fn input_signal(signal: Signal, channels: usize, family: i32, samples: usize, gain: f32) -> Vec<f32> {
    let mut pcm = match signal {
        Signal::Tones => tones(FS, channels, family, samples),
        Signal::Voiced => voiced(FS, channels, samples),
    };
    if gain != 1.0 {
        for v in pcm.iter_mut() {
            *v *= gain;
        }
    }
    pcm
}

/// Encodes `milliseconds` of `signal` with the C encoder.
///
/// - `kbps_per_channel`: the total bitrate is this times the number of
///   input channels.
/// - `frame`: frame size in samples at 48 kHz, 120 (2.5 ms) to 2880 (60 ms).
/// - `gain`: factor on the signal. Above 4 the decoded signal goes beyond
///   full scale.
pub fn encode(
    layout: Layout,
    settings: Settings,
    kbps_per_channel: i32,
    frame: usize,
    signal: Signal,
    milliseconds: usize,
    gain: f32,
) -> Encoded {
    let (mut enc, family) = c_encoder(layout, settings);
    let input_channels = enc.channels;
    enc.set_bitrate(kbps_per_channel * 1000 * input_channels as i32);
    let samples = FS as usize * milliseconds / 1000 / frame * frame;
    let pcm = input_signal(signal, input_channels, family, samples, gain);
    let mut packets = Vec::new();
    let mut ranges = Vec::new();
    for chunk in pcm.chunks_exact(frame * input_channels) {
        let packet = enc
            .encode_float(chunk, frame, max_packet_bytes(enc.streams))
            .unwrap_or_else(|e| panic!("C encoder failed: {e}"));
        ranges.push(enc.final_range());
        packets.push(packet);
    }
    let (channels, mapping) = match layout {
        Layout::Family { .. } => (input_channels, enc.mapping.clone()),
        Layout::Generic => (GENERIC_DECODER_MAPPING.len(), GENERIC_DECODER_MAPPING.to_vec()),
    };
    Encoded {
        channels,
        streams: enc.streams,
        coupled: enc.coupled,
        mapping,
        family,
        input_channels,
        packets,
        ranges,
        label: format!(
            "{layout:?} {settings:?} {kbps_per_channel} kb/s per channel, {} ms frames, {signal:?}",
            frame as f64 / 48.0
        ),
    }
}

/// Signal-to-noise ratio in dB of `test` against `reference` for every
/// channel of two interleaved signals of the same length. Two silent
/// channels give 200 dB.
pub fn channel_snrs(reference: &[f32], test: &[f32], channels: usize) -> Vec<f64> {
    assert_eq!(reference.len(), test.len(), "the signals have different lengths");
    (0..channels)
        .map(|c| {
            let (mut signal, mut noise) = (0f64, 0f64);
            for (r, t) in reference.iter().skip(c).step_by(channels).zip(test.iter().skip(c).step_by(channels)) {
                let (r, t) = (*r as f64, *t as f64);
                signal += r * r;
                noise += (r - t) * (r - t);
            }
            snr_db(signal, noise)
        })
        .collect()
}

/// The smallest value of `channel_snrs`.
pub fn min_channel_snr(reference: &[f32], test: &[f32], channels: usize) -> f64 {
    channel_snrs(reference, test, channels).into_iter().fold(f64::INFINITY, f64::min)
}

/// Sum of squares of a signal.
pub fn energy(x: &[f32]) -> f64 {
    x.iter().map(|&v| v as f64 * v as f64).sum()
}

/// Largest absolute difference between two signals of the same length.
pub fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "the signals have different lengths");
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

/// Deterministic pseudo-random numbers for the tests.
pub struct Rng(pub u32);

impl Rng {
    pub fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
        self.0 >> 8
    }

    /// A number below `n`.
    pub fn below(&mut self, n: usize) -> usize {
        self.next() as usize % n
    }
}
