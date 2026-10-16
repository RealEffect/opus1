//! The channel layouts of `opus1::multistream` (`src/multistream/layout.rs`)
//! against C libopus.
//!
//! - `ChannelLayout::surround` against `opus_multistream_surround_encoder_create`:
//!   same accept or reject, same streams, coupled streams and mapping. The
//!   mapping type and the LFE stream are not readable from the C encoder, so
//!   they are compared through what the C encoder does with them.
//! - `ChannelLayout::new` against `opus_multistream_decoder_create`, and
//!   `validate_for_encoder` against `opus_multistream_encoder_create`.
//! - `left_channel`, `right_channel` and `mono_channel` against the routing
//!   of the C decoder (all matching channels) and of the C encoder (the first
//!   matching channel).
#![cfg(feature = "std")]

mod ms_support;

use audiopus_sys as sys;
use ms_support::*;
use opus1::multistream::{ChannelLayout, MappingType, SILENT_CHANNEL, validate_ambisonics};

const FS: i32 = 48_000;
/// 20 ms at 48 kHz.
const FRAME: usize = 960;

/// Compares `ChannelLayout::surround` with the C surround constructor for one
/// channel count and mapping family. Returns whether both accepted.
fn check_surround(channels: usize, family: u8) -> bool {
    let c = CMsEncoder::surround(FS, channels, family as i32, APP_AUDIO);
    let rs = ChannelLayout::surround(channels, family);
    assert_eq!(
        rs.is_ok(),
        c.is_ok(),
        "family {family}, {channels} channels: opus1 {:?}, libopus error {:?}",
        rs.as_ref().err(),
        c.as_ref().err()
    );
    let (Ok((layout, _, _)), Ok(c)) = (rs, c) else {
        return false;
    };
    let what = format!("family {family}, {channels} channels");
    assert_eq!(layout.channels(), channels, "{what}");
    assert_eq!(layout.streams(), c.streams, "{what}: streams");
    assert_eq!(
        layout.coupled_streams(),
        c.coupled,
        "{what}: coupled streams"
    );
    assert_eq!(layout.mapping(), &c.mapping[..], "{what}: mapping");
    // What the surround constructor builds, both generic C constructors accept.
    assert!(
        CMsEncoder::generic(FS, channels, c.streams, c.coupled, &c.mapping, APP_AUDIO).is_ok(),
        "{what}"
    );
    assert!(
        CMsDecoder::new(FS, channels, c.streams, c.coupled, &c.mapping).is_ok(),
        "{what}"
    );
    true
}

#[test]
fn surround_layouts_match_libopus() {
    let mut accepted = 0;
    let mut cases = 0;
    for family in [0u8, 1, 2, 255, 3] {
        for channels in 1..=20 {
            cases += 1;
            accepted += check_surround(channels, family) as usize;
        }
    }
    // Family 0: 2 layouts. Family 1: 8. Family 2: 1, 3, 4, 6, 9, 11, 16, 18.
    // Family 255: 20. Family 3: none.
    assert_eq!((cases, accepted), (100, 2 + 8 + 8 + 20));
}

#[test]
fn surround_layouts_match_libopus_for_every_channel_count() {
    // Beyond the grid above: every channel count libopus could be asked for,
    // with the two families that go up to 255 channels, and a few families
    // that do not exist.
    let mut accepted = 0;
    for family in [2u8, 255] {
        for channels in 0..=257 {
            accepted += check_surround(channels, family) as usize;
        }
    }
    // Family 2: 15 square numbers and 15 square numbers plus 2, up to 227.
    assert_eq!(accepted, 30 + 255);
    for family in [0u8, 1] {
        for channels in [0usize, 21, 100, 255, 256, 257] {
            assert!(!check_surround(channels, family));
        }
    }
    for family in [4u8, 5, 100, 254] {
        for channels in [1usize, 2, 4, 6] {
            assert!(!check_surround(channels, family));
        }
    }
    // The stream counts of family 2 as a function of their own.
    for channels in 0..=257 {
        let c = CMsEncoder::surround(FS, channels, 2, APP_AUDIO)
            .ok()
            .map(|e| (e.streams, e.coupled));
        assert_eq!(validate_ambisonics(channels), c, "{channels} channels");
    }
}

#[test]
fn surround_layouts_match_libopus_for_every_mapping_family() {
    // Every value the mapping family byte of an Ogg Opus header can have,
    // with every channel count from 0 to 256. Families 2 and 255 are left to
    // the test above: they accept many channel counts, and each accepted
    // layout costs up to 255 C stream encoders.
    let mut accepted = 0;
    let mut cases = 0;
    for family in (0..=255u8).filter(|&f| f != 2 && f != 255) {
        for channels in 0..=256 {
            cases += 1;
            accepted += check_surround(channels, family) as usize;
        }
    }
    // Family 0 has 2 layouts and family 1 has 8. No other family exists.
    assert_eq!((cases, accepted), (254 * 257, 2 + 8));
}

/// What the C encoder shows of the mapping type and the LFE stream of a
/// surround layout, after it has encoded a few frames.
struct CBehaviour {
    /// Coupled streams on which the encoder forced stereo coding. libopus
    /// does that on every coupled stream in surround mode, and nowhere else.
    forced_stereo: Vec<usize>,
    /// Whether every stream used CELT although the settings favour SILK.
    /// libopus forces CELT on every stream in ambisonics mode.
    all_celt: bool,
    /// Streams coded narrowband although the bitrate allows fullband. libopus
    /// does that to the LFE stream only.
    narrowband: Vec<usize>,
}

fn c_behaviour(channels: usize, family: u8) -> CBehaviour {
    // Speech settings at 12 kb/s per channel: a stream that is free to choose
    // codes this signal with SILK.
    let mut enc = CMsEncoder::surround(FS, channels, family as i32, APP_VOIP).unwrap();
    enc.set_bitrate(12_000 * channels as i32);
    let pcm = voiced(FS, channels, 5 * FRAME);
    let packets = enc.encode_all(&pcm, FRAME);
    let parts = split_streams(packets.last().unwrap(), enc.streams).unwrap();
    let all_celt = parts.iter().all(|p| toc(p.toc).mode == Mode::Celt);
    let forced_stereo = (0..enc.streams)
        .filter(|&s| enc.stream_get(s, sys::OPUS_GET_FORCE_CHANNELS_REQUEST) == 2)
        .collect();

    // Music settings at 64 kb/s per channel: every stream is fullband unless
    // something holds it back.
    let mut enc = CMsEncoder::surround(FS, channels, family as i32, APP_AUDIO).unwrap();
    enc.set_bitrate(64_000 * channels as i32);
    let pcm = noise(channels, 5 * FRAME, 0.1, 3);
    let packets = enc.encode_all(&pcm, FRAME);
    let parts = split_streams(packets.last().unwrap(), enc.streams).unwrap();
    let narrowband = (0..enc.streams)
        .filter(|&s| toc(parts[s].toc).bandwidth == Bandwidth::Nb)
        .collect();
    CBehaviour {
        forced_stereo,
        all_celt,
        narrowband,
    }
}

#[test]
fn mapping_type_and_lfe_stream_match_libopus_behaviour() {
    let cases: Vec<(usize, u8)> = (1..=8)
        .map(|ch| (ch, 1u8))
        .chain([
            (1, 0),
            (2, 0),
            (1, 255),
            (3, 255),
            (6, 255),
            (1, 2),
            (3, 2),
            (4, 2),
            (6, 2),
            (9, 2),
        ])
        .collect();
    let mut seen = [0usize; 3];
    for (channels, family) in cases {
        let (layout, mapping_type, lfe_stream) = ChannelLayout::surround(channels, family).unwrap();
        let c = c_behaviour(channels, family);
        let what = format!("family {family}, {channels} channels");

        // The LFE stream is the one stream libopus keeps narrowband.
        let want_narrowband: Vec<usize> = lfe_stream.into_iter().collect();
        assert_eq!(c.narrowband, want_narrowband, "{what}: narrowband streams");

        match mapping_type {
            MappingType::Surround => {
                seen[0] += 1;
                let coupled: Vec<usize> = (0..layout.coupled_streams()).collect();
                assert!(!coupled.is_empty(), "{what}");
                assert_eq!(
                    c.forced_stereo, coupled,
                    "{what}: streams with forced stereo"
                );
            }
            MappingType::Ambisonics => {
                seen[1] += 1;
                assert!(c.forced_stereo.is_empty(), "{what}");
                assert!(c.all_celt, "{what}: libopus did not force CELT");
            }
            MappingType::None => {
                seen[2] += 1;
                assert!(c.forced_stereo.is_empty(), "{what}");
                assert!(!c.all_celt, "{what}: libopus forced CELT");
            }
        }
    }
    assert_eq!(seen, [6, 5, 7]);
}

/// Compares the two Rust checks with the two generic C constructors for one
/// layout. Returns (decoder accepts, encoder accepts).
fn check_generic(channels: usize, streams: usize, coupled: usize, mapping: &[u8]) -> (bool, bool) {
    let what =
        format!("{channels} channels, {streams} streams, {coupled} coupled, mapping {mapping:?}");
    let rs = ChannelLayout::new(channels, streams, coupled, mapping);
    let c_dec = CMsDecoder::new(FS, channels, streams, coupled, mapping);
    assert_eq!(
        rs.is_ok(),
        c_dec.is_ok(),
        "{what}: ChannelLayout::new {:?}, C decoder error {:?}",
        rs.as_ref().err(),
        c_dec.as_ref().err()
    );
    let rs_enc = rs.clone().and_then(|l| l.validate_for_encoder());
    let c_enc = CMsEncoder::generic(FS, channels, streams, coupled, mapping, APP_AUDIO);
    assert_eq!(
        rs_enc.is_ok(),
        c_enc.is_ok(),
        "{what}: validate_for_encoder {:?}, C encoder error {:?}",
        rs_enc.as_ref().err(),
        c_enc.as_ref().err()
    );
    // An encoder layout is always a decoder layout.
    assert!(c_dec.is_ok() || c_enc.is_err(), "{what}");
    if let Ok(layout) = &rs {
        assert_eq!(layout.channels(), channels, "{what}");
        assert_eq!(layout.streams(), streams, "{what}");
        assert_eq!(layout.coupled_streams(), coupled, "{what}");
        assert_eq!(layout.mapping(), &mapping[..channels], "{what}");
    }
    (c_dec.is_ok(), c_enc.is_ok())
}

/// A layout and what libopus says about it: channels, streams, coupled
/// streams, mapping, whether a decoder accepts it, whether an encoder does.
type LayoutCase<'a> = (usize, usize, usize, &'a [u8], bool, bool);

#[test]
fn hand_made_layouts_match_libopus() {
    let identity: Vec<u8> = (0..=255).collect();
    let mut silent_255 = vec![SILENT_CHANNEL; 255];
    silent_255[0] = 254;
    let cases: &[LayoutCase] = &[
        // Plain valid layouts.
        (1, 1, 0, &[0], true, true),
        (2, 1, 1, &[0, 1], true, true),
        (2, 1, 1, &[1, 0], true, true),
        (2, 2, 0, &[0, 1], true, true),
        (3, 2, 1, &[0, 1, 2], true, true),
        (3, 3, 0, &[2, 0, 1], true, true),
        (4, 2, 2, &[0, 1, 2, 3], true, true),
        (6, 4, 2, &[0, 4, 1, 2, 3, 5], true, true),
        // A channel that is not coded, and a decoded channel used twice.
        (3, 1, 1, &[0, 1, 255], true, true),
        (4, 2, 1, &[255, 0, 1, 2], true, true),
        (5, 2, 1, &[0, 1, 2, 255, 2], true, true),
        (4, 1, 1, &[0, 1, 0, 1], true, true),
        // Valid for a decoder only: a stream without an input channel.
        (3, 2, 1, &[0, 1, 0], true, false),
        (2, 1, 1, &[0, 0], true, false),
        (2, 1, 1, &[1, 1], true, false),
        (1, 1, 0, &[255], true, false),
        (3, 3, 0, &[0, 1, 1], true, false),
        (4, 2, 1, &[0, 1, 255, 255], true, false),
        // Valid for a decoder only: more decoded channels than channels.
        (3, 3, 1, &[0, 1, 2], true, false),
        (2, 2, 1, &[0, 1], true, false),
        (1, 1, 1, &[0], true, false),
        (1, 2, 0, &[1], true, false),
        (2, 128, 127, &[0, 254], true, false),
        (2, 255, 0, &[0, 254], true, false),
        // Invalid for both: a mapping entry that names no decoded channel.
        (1, 1, 0, &[5], false, false),
        (1, 1, 0, &[1], false, false),
        (3, 2, 1, &[0, 1, 3], false, false),
        (3, 2, 1, &[0, 1, 254], false, false),
        (4, 2, 2, &[0, 1, 2, 4], false, false),
        // Invalid for both: the counts.
        (1, 0, 0, &[0], false, false),
        (2, 1, 2, &[0, 1], false, false),
        (2, 0, 1, &[0, 1], false, false),
        (0, 1, 0, &[0], false, false),
        (4, 128, 128, &[0, 1, 2, 3], false, false),
        (4, 256, 0, &[0, 1, 2, 3], false, false),
        (4, 200, 56, &[0, 1, 2, 3], false, false),
        (256, 1, 0, &identity, false, false),
        (300, 2, 1, &[0; 300], false, false),
        // The largest layouts.
        (255, 255, 0, &identity, true, true),
        (255, 128, 127, &identity, true, true),
        (255, 128, 127, &silent_255, true, false),
        (255, 1, 0, &[0; 255], true, true),
    ];
    for &(channels, streams, coupled, mapping, want_dec, want_enc) in cases {
        let got = check_generic(channels, streams, coupled, mapping);
        assert_eq!(
            got,
            (want_dec, want_enc),
            "{channels} channels, {streams} streams, {coupled} coupled, mapping {:?}",
            &mapping[..mapping.len().min(8)]
        );
    }
}

#[test]
fn random_layouts_match_libopus() {
    let mut seed = 0x1234_5678u32;
    let mut next = move |n: usize| {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        (seed >> 8) as usize % n
    };
    // Counts of (decoder accepts, encoder accepts): neither, decoder only, both.
    let mut seen = [0usize; 3];
    for _ in 0..4000 {
        let channels = next(10);
        let streams = next(6);
        let coupled = next(5);
        let decoded_channels = streams + coupled;
        let mapping: Vec<u8> = (0..channels.max(1))
            .map(|_| match next(12) {
                // Now and then: silent, or just outside the valid range.
                0 => SILENT_CHANNEL,
                1 => decoded_channels as u8,
                _ => next(decoded_channels.max(1)) as u8,
            })
            .collect();
        match check_generic(channels, streams, coupled, &mapping) {
            (false, false) => seen[0] += 1,
            (true, false) => seen[1] += 1,
            (true, true) => seen[2] += 1,
            (false, true) => unreachable!(),
        }
    }
    println!(
        "random layouts: {} invalid, {} decoder only, {} valid for both",
        seen[0], seen[1], seen[2]
    );
    assert!(seen.iter().all(|&n| n >= 100), "{seen:?}");
}

#[test]
fn small_layouts_match_libopus_exhaustively() {
    // Every layout with 0 to 4 channels, 0 to 4 streams and 0 to 4 coupled
    // streams. A mapping entry is each decoded channel, the first two values
    // past the last decoded channel, 254, or the silent value 255.
    // Counts of layouts that are invalid, valid for a decoder only, and valid
    // for both.
    let mut seen = [0usize; 3];
    for channels in 0..=4usize {
        for streams in 0..=4usize {
            for coupled in 0..=4usize {
                let decoded_channels = (streams + coupled) as u8;
                let mut values: Vec<u8> = (0..=decoded_channels + 1).collect();
                values.extend([254, SILENT_CHANNEL]);
                // The C constructors read one entry even for 0 channels.
                let entries = channels.max(1);
                let mut digits = vec![0usize; entries];
                loop {
                    let mapping: Vec<u8> = digits.iter().map(|&d| values[d]).collect();
                    match check_generic(channels, streams, coupled, &mapping) {
                        (false, false) => seen[0] += 1,
                        (true, false) => seen[1] += 1,
                        (true, true) => seen[2] += 1,
                        (false, true) => unreachable!(),
                    }
                    // Next mapping: count up in base `values.len()`.
                    let mut i = 0;
                    while i < entries {
                        digits[i] += 1;
                        if digits[i] < values.len() {
                            break;
                        }
                        digits[i] = 0;
                        i += 1;
                    }
                    if i == entries {
                        break;
                    }
                }
            }
        }
    }
    // An encoder accepts a layout when no entry is out of range and every
    // decoded channel has an input channel. Counting those mappings by hand
    // gives 1, 7, 43 and 307 layouts for 1 to 4 channels.
    assert_eq!(seen[2], 1 + 7 + 43 + 307);
    // The other two counts were measured with libopus 1.6.1. Together the
    // three are the size of the grid.
    assert_eq!(seen, [134_982, 23_700, 358]);
}

#[test]
fn short_mapping_is_an_error() {
    // C reads `channels` bytes behind the pointer whatever their number. The
    // Rust constructor must refuse a slice that is too short.
    assert!(ChannelLayout::new(3, 2, 1, &[0, 1]).is_err());
    assert!(ChannelLayout::new(1, 1, 0, &[]).is_err());
    assert!(ChannelLayout::new(3, 2, 1, &[0, 1, 2]).is_ok());
}

/// Builds the output the C multistream decoder must give for one frame, from
/// the samples of each stream and the three Rust lookups. The loops are those
/// of libopus `opus_multistream_decode_native`: every matching channel gets
/// the samples, and a channel with the silent value gets zeros.
fn route(layout: &ChannelLayout, stream_pcm: &[Vec<f32>], frame_size: usize) -> Vec<f32> {
    let channels = layout.channels();
    // A value no decoder produces marks the samples nobody wrote.
    let mut out = vec![f32::NAN; frame_size * channels];
    let mut copy = |chan: usize, src: &[f32], offset: usize, stride: usize| {
        for i in 0..frame_size {
            out[i * channels + chan] = src[i * stride + offset];
        }
    };
    assert_eq!(stream_pcm.len(), layout.streams());
    for (s, pcm) in stream_pcm.iter().enumerate() {
        if s < layout.coupled_streams() {
            let mut prev = None;
            while let Some(chan) = layout.left_channel(s, prev) {
                copy(chan, pcm, 0, 2);
                prev = Some(chan);
            }
            let mut prev = None;
            while let Some(chan) = layout.right_channel(s, prev) {
                copy(chan, pcm, 1, 2);
                prev = Some(chan);
            }
        } else {
            let mut prev = None;
            while let Some(chan) = layout.mono_channel(s, prev) {
                copy(chan, pcm, 0, 1);
                prev = Some(chan);
            }
        }
    }
    for (c, &m) in layout.mapping().iter().enumerate() {
        if m == SILENT_CHANNEL {
            for i in 0..frame_size {
                out[i * channels + c] = 0.0;
            }
        }
    }
    out
}

#[test]
fn routing_lookups_match_the_c_decoder() {
    // (streams, coupled) of the coded packet, and decoder mappings for it.
    // The mappings repeat decoded channels, skip some and have silent output
    // channels.
    let cases: [(usize, usize, &[&[u8]]); 4] = [
        (
            2,
            1,
            &[
                &[0, 1, 2],
                &[2, 0, 1, 0, 255, 2, 1, 2],
                &[255, 255, 2, 2, 1],
                &[1],
                &[2, 2, 2, 2],
            ],
        ),
        (
            4,
            2,
            &[
                &[0, 4, 1, 2, 3, 5],
                &[5, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5],
                &[3, 255, 3],
                &[255],
            ],
        ),
        (3, 0, &[&[0, 1, 2], &[2, 2, 0, 255, 1, 0], &[1]]),
        (2, 2, &[&[0, 1, 2, 3], &[3, 1, 255, 3, 1], &[2]]),
    ];
    let mut checked = 0;
    for (streams, coupled, mappings) in cases {
        // Encode with the plain layout: one input channel per decoded channel.
        let coded_channels = streams + coupled;
        let identity: Vec<u8> = (0..coded_channels as u8).collect();
        let mut enc =
            CMsEncoder::generic(FS, coded_channels, streams, coupled, &identity, APP_AUDIO)
                .unwrap();
        enc.set_bitrate(48_000 * coded_channels as i32);
        let pcm = tones(FS, coded_channels, 255, 8 * FRAME);
        let packets = enc.encode_all(&pcm, FRAME);

        for mapping in mappings {
            let channels = mapping.len();
            let layout = ChannelLayout::new(channels, streams, coupled, mapping).unwrap();
            let mut dec = CMsDecoder::new(FS, channels, streams, coupled, mapping).unwrap();
            let mut stream_decoders: Vec<CDecoder> = (0..streams)
                .map(|s| CDecoder::new(FS, if s < coupled { 2 } else { 1 }))
                .collect();
            for (f, packet) in packets.iter().enumerate() {
                let c_out = dec.decode_float(Some(packet), FRAME, false).unwrap();
                let parts = split_streams(packet, streams).unwrap();
                let stream_pcm: Vec<Vec<f32>> = parts
                    .iter()
                    .zip(stream_decoders.iter_mut())
                    .map(|(part, d)| d.decode_float(Some(&part.packet), FRAME).unwrap())
                    .collect();
                let want = route(&layout, &stream_pcm, FRAME);
                assert!(
                    want.iter().all(|v| !v.is_nan()),
                    "mapping {mapping:?}: the lookups left an output channel unwritten"
                );
                assert_eq!(c_out, want, "mapping {mapping:?}, frame {f}");
                checked += 1;
            }
            // The last frame is not silence, so the comparison means something.
            let last = dec
                .decode_float(Some(packets.last().unwrap()), FRAME, false)
                .unwrap();
            let audible = (0..channels)
                .filter(|&c| channel(&last, channels, c).iter().any(|&v| v != 0.0))
                .count();
            let not_silent = mapping.iter().filter(|&&m| m != SILENT_CHANNEL).count();
            assert_eq!(audible, not_silent, "mapping {mapping:?}");
        }
    }
    assert_eq!(checked, (5 + 4 + 3 + 3) * 8);
}

#[test]
fn routing_lookups_match_the_c_encoder() {
    // Encoder layouts in which several input channels name the same decoded
    // channel. libopus reads the first of them and ignores the others.
    // (channels, streams, coupled, mapping)
    let cases: [(usize, usize, usize, &[u8]); 6] = [
        (4, 1, 1, &[0, 1, 0, 1]),
        (4, 1, 1, &[1, 1, 0, 0]),
        (4, 2, 0, &[1, 0, 1, 0]),
        (5, 2, 1, &[2, 255, 1, 0, 2]),
        (6, 2, 1, &[255, 2, 0, 0, 1, 2]),
        (6, 4, 2, &[0, 4, 1, 2, 3, 5]),
    ];
    for (channels, streams, coupled, mapping) in cases {
        let layout = ChannelLayout::new(channels, streams, coupled, mapping).unwrap();
        layout.validate_for_encoder().unwrap();
        let mut enc =
            CMsEncoder::generic(FS, channels, streams, coupled, mapping, APP_AUDIO).unwrap();
        let coded_channels = streams + coupled;
        enc.set_bitrate(64_000 * coded_channels as i32);
        // Every input channel has its own tone.
        let pcm = tones(FS, channels, 255, 30 * FRAME);
        let packets = enc.encode_all(&pcm, FRAME);
        // Decode to one output channel per decoded channel.
        let identity: Vec<u8> = (0..coded_channels as u8).collect();
        let mut dec = CMsDecoder::new(FS, coded_channels, streams, coupled, &identity).unwrap();
        let out = dec.decode_all(&packets);

        for s in 0..streams {
            // (decoded channel, input channel the Rust lookup names for it)
            let sources: Vec<(usize, usize)> = if s < coupled {
                vec![
                    (2 * s, layout.left_channel(s, None).unwrap()),
                    (2 * s + 1, layout.right_channel(s, None).unwrap()),
                ]
            } else {
                vec![(s + coupled, layout.mono_channel(s, None).unwrap())]
            };
            for (decoded, input) in sources {
                let x = channel(&out, coded_channels, decoded);
                let x = &x[10 * FRAME..];
                // Which input channel's tone is in this decoded channel?
                let mut powers: Vec<(usize, f64)> = (0..channels)
                    .map(|k| (k, goertzel_power(x, tone_hz(k, channels, 255), FS)))
                    .collect();
                powers.sort_by(|a, b| b.1.total_cmp(&a.1));
                assert_eq!(
                    powers[0].0, input,
                    "mapping {mapping:?}, stream {s}, decoded channel {decoded}: libopus coded input channel {}, the lookup says {input}",
                    powers[0].0
                );
                assert!(
                    powers[0].1 > 100.0 * powers[1].1,
                    "mapping {mapping:?}, decoded channel {decoded}: tone powers {powers:?}"
                );
            }
        }
    }
}
