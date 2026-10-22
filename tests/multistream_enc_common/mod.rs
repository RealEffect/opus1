//! Shared helpers of the multistream encoder tests
//! (`tests/multistream_enc_*.rs`).
//!
//! A test file uses it with `mod ms_support; mod multistream_enc_common;`.
//! The helpers build the opus1 encoder and the C libopus encoder for the
//! same layout and settings, so that a test compares like with like.

#![allow(dead_code)]

use crate::ms_support::*;
use opus1::multistream::MultistreamEncoder;
use opus1::{Application, Bitrate, OpusEncoder, SampleRate};

/// The bitrate a stream encoder was given, in bits per second. The
/// multistream encoder gives every stream an explicit bitrate when it
/// encodes a frame.
pub fn bitrate_set(enc: &OpusEncoder) -> i32 {
    match enc.bitrate_setting() {
        Bitrate::BitsPerSecond(bps) => bps,
        other => panic!("the stream encoder has no explicit bitrate: {other:?}"),
    }
}

/// How the layout of a test case is chosen.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// The surround constructor with this mapping family.
    Family(u8),
    /// The generic constructor with these streams, coupled streams and
    /// mapping.
    Generic(usize, usize, &'static [u8]),
}

/// One layout to test.
#[derive(Debug, Clone, Copy)]
pub struct Case {
    pub name: &'static str,
    pub channels: usize,
    pub kind: Kind,
}

impl Case {
    pub const fn family(name: &'static str, channels: usize, family: u8) -> Self {
        Self {
            name,
            channels,
            kind: Kind::Family(family),
        }
    }

    pub const fn generic(
        name: &'static str,
        channels: usize,
        streams: usize,
        coupled: usize,
        mapping: &'static [u8],
    ) -> Self {
        Self {
            name,
            channels,
            kind: Kind::Generic(streams, coupled, mapping),
        }
    }

    /// The mapping family that `tones` needs to place the 80 Hz tone of the
    /// LFE channel. A generic layout has no LFE channel.
    pub fn tone_family(&self) -> i32 {
        match self.kind {
            Kind::Family(f) => f as i32,
            Kind::Generic(..) => 255,
        }
    }

    /// The input channels the encoder codes. A generic layout can leave a
    /// channel out (mapping 255) or map two channels to one stream; the
    /// encoder then reads the first of them.
    pub fn coded_channels(&self) -> Vec<usize> {
        match self.kind {
            Kind::Family(_) => (0..self.channels).collect(),
            Kind::Generic(_, _, mapping) => (0..self.channels)
                .filter(|&c| mapping[c] != 255 && !mapping[..c].contains(&mapping[c]))
                .collect(),
        }
    }
}

/// The layouts every matrix test runs: the surround layouts of the design,
/// ambisonics, and layouts only the generic constructor can make.
pub const CASES: &[Case] = &[
    Case::family("mono", 1, 1),
    Case::family("stereo", 2, 1),
    Case::family("quad", 4, 1),
    Case::family("5.1", 6, 1),
    Case::family("7.1", 8, 1),
    Case::family("family 255, 3 channels", 3, 255),
    Case::family("family 2, 4 channels", 4, 2),
    Case::family("family 2, 6 channels", 6, 2),
    // The 3-channel layout of family 1, without surround mode.
    Case::generic("generic 3 channels", 3, 2, 1, &[0, 2, 1]),
    // A mono stream first in channel order, then a stereo pair.
    Case::generic("generic 4 channels", 4, 3, 1, &[2, 0, 1, 3]),
    // Channel 2 is not coded. Channels 3 and 4 both name the mono stream.
    Case::generic("generic 5 channels", 5, 2, 1, &[0, 1, 255, 2, 2]),
];

/// The same layouts without the two smallest and without duplicates of a
/// stream shape. For tests that are too slow for all of `CASES`.
pub const SURROUND_CASES: &[Case] = &[
    Case::family("quad", 4, 1),
    Case::family("5.1", 6, 1),
    Case::family("7.1", 8, 1),
];

/// The application value of the C API.
pub fn c_application(application: Application) -> i32 {
    match application {
        Application::Voip => APP_VOIP,
        Application::Audio => APP_AUDIO,
        Application::RestrictedLowDelay => APP_LOWDELAY,
    }
}

/// The opus1 encoder of a case.
pub fn rust_encoder(case: &Case, fs: i32, application: Application) -> MultistreamEncoder {
    match case.kind {
        Kind::Family(family) => {
            MultistreamEncoder::new_surround(SampleRate::try_from(fs).unwrap(), case.channels, family, application)
        }
        Kind::Generic(streams, coupled, mapping) => {
            MultistreamEncoder::new(SampleRate::try_from(fs).unwrap(), case.channels, streams, coupled, mapping, application)
        }
    }
    .unwrap_or_else(|e| panic!("{}: opus1 encoder: {e}", case.name))
}

/// The C encoder of a case.
pub fn c_encoder(case: &Case, fs: i32, application: Application) -> CMsEncoder {
    let app = c_application(application);
    match case.kind {
        Kind::Family(family) => CMsEncoder::surround(fs, case.channels, family as i32, app),
        Kind::Generic(streams, coupled, mapping) => {
            CMsEncoder::generic(fs, case.channels, streams, coupled, mapping, app)
        }
    }
    .unwrap_or_else(|e| panic!("{}: C encoder: {e}", case.name))
}

/// Both encoders of a case. Checks that they chose the same layout.
pub fn encoders(
    case: &Case,
    fs: i32,
    application: Application,
) -> (MultistreamEncoder, CMsEncoder) {
    let rust = rust_encoder(case, fs, application);
    let c = c_encoder(case, fs, application);
    let layout = rust.layout();
    assert_eq!(
        (
            layout.channels(),
            layout.streams(),
            layout.coupled_streams()
        ),
        (c.channels, c.streams, c.coupled),
        "{}: layout",
        case.name
    );
    assert_eq!(layout.mapping(), &c.mapping[..], "{}: mapping", case.name);
    (rust, c)
}

/// A C decoder with the layout and sampling rate of the opus1 encoder.
pub fn c_decoder(enc: &MultistreamEncoder) -> CMsDecoder {
    let layout = enc.layout();
    CMsDecoder::new(
        enc.sampling_rate().hz() as i32,
        layout.channels(),
        layout.streams(),
        layout.coupled_streams(),
        layout.mapping(),
    )
    .unwrap_or_else(|e| panic!("C decoder: {e}"))
}

/// The bitrate setting of the C API as the opus1 value.
pub fn rust_bitrate(c_value: i32) -> Bitrate {
    match c_value {
        OPUS_AUTO => Bitrate::Auto,
        OPUS_BITRATE_MAX => Bitrate::Max,
        bps => Bitrate::BitsPerSecond(bps),
    }
}

/// Sets the same bitrate and the same VBR switch on both encoders.
/// `bitrate` is a value of the C API: bits per second, `OPUS_AUTO` or
/// `OPUS_BITRATE_MAX`.
pub fn configure(rust: &mut MultistreamEncoder, c: &mut CMsEncoder, bitrate: i32, vbr: bool) {
    rust.set_bitrate(rust_bitrate(bitrate)).unwrap();
    rust.set_vbr(vbr);
    c.set_bitrate(bitrate);
    c.set_vbr(vbr);
}

/// Encodes a whole signal with the opus1 encoder, frame by frame, with
/// room for the largest packet. A trailing partial frame is dropped.
pub fn rust_encode_all(
    enc: &mut MultistreamEncoder,
    pcm: &[f32],
    frame_size: usize,
) -> Vec<Vec<u8>> {
    let channels = enc.layout().channels();
    let mut out = vec![0u8; max_packet_bytes(enc.layout().streams())];
    pcm.chunks_exact(frame_size * channels)
        .enumerate()
        .map(|(f, frame)| {
            let n = enc
                .encode(frame, frame_size, &mut out)
                .unwrap_or_else(|e| panic!("frame {f}: opus1 encode failed: {e}"));
            out[..n].to_vec()
        })
        .collect()
}

/// The frame sizes of 2.5, 5, 10, 20, 40 and 60 ms at `fs`, in samples.
pub fn frame_sizes(fs: i32) -> [usize; 6] {
    let fs = fs as usize;
    [fs / 400, fs / 200, fs / 100, fs / 50, fs / 25, 3 * fs / 50]
}

/// Text for a frame size, for failure messages.
pub fn ms(frame_size: usize, fs: i32) -> String {
    format!("{} ms", frame_size as f64 * 1000.0 / fs as f64)
}
