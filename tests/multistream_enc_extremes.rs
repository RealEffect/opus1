//! The opus1 multistream encoder at the ends of what its arguments allow,
//! against C libopus.
//!
//! - The largest layouts: 255 mono streams, 127 stereo streams, and the
//!   largest ambisonics layout, with buffers from the smallest packet up.
//! - Empty input, an empty output buffer, and frame sizes and bitrates at
//!   the ends of their types. Nothing may panic, and the two encoders must
//!   agree on what they accept.
#![cfg(feature = "std")]

mod ms_support;
mod multistream_enc_common;

use ms_support::*;
use multistream_enc_common::*;
use opus1::{Application, Bitrate};
use opus1::multistream::MultistreamEncoder;

/// The channels 0 to 253 in pairs: 127 stereo streams and no mono stream.
static PAIRS_254: [u8; 254] = {
    let mut mapping = [0u8; 254];
    let mut i = 0;
    while i < 254 {
        mapping[i] = i as u8;
        i += 1;
    }
    mapping
};

/// The largest layouts libopus takes.
fn largest_layouts() -> [Case; 3] {
    [
        // 255 mono streams.
        Case::family("255 mono streams", 255, 255),
        // 127 stereo streams: 254 coded channels.
        Case::generic("127 stereo streams", 254, 127, 127, &PAIRS_254),
        // Ambisonics of order 14 with a stereo stream: 225 + 2 channels in
        // 226 streams.
        Case::family("ambisonics, 227 channels", 227, 2),
    ]
}

#[test]
fn the_largest_layouts_encode_like_libopus() {
    let mut packets = 0;
    for case in largest_layouts() {
        let channels = case.channels;
        for (fs, application) in [
            (48_000, Application::RestrictedLowDelay),
            (8_000, Application::Audio),
        ] {
            let (mut rust, mut c) = encoders(&case, fs, application);
            let streams = c.streams;
            let smallest = 2 * streams - 1;
            let mut dec = c_decoder(&rust);
            let frame_size = fs as usize / 50;
            let pcm = noise(channels, 2 * frame_size, 0.2, 5);
            // A frame that is not a multiple of 20 ms as well.
            let short = fs as usize / 200;
            let settings = [
                (OPUS_AUTO, true),
                (12_000 * channels as i32, true),
                (12_000 * channels as i32, false),
                (OPUS_BITRATE_MAX, true),
                // The lowest and the highest bitrate that can be set.
                (1, false),
                (i32::MAX, true),
            ];
            for (bitrate, vbr) in settings {
                configure(&mut rust, &mut c, bitrate, vbr);
                for len in [
                    smallest,
                    smallest + 1,
                    3 * streams,
                    4_000,
                    max_packet_bytes(streams),
                ] {
                    for (frame, n_samples) in [(&pcm[..], frame_size), (&pcm[..], short)] {
                        let label = format!(
                            "{} {fs} Hz {application:?} {bitrate} b/s vbr {vbr}, {n_samples} \
                             samples into {len} bytes",
                            case.name
                        );
                        let mut out = vec![0u8; len];
                        let n = rust
                            .encode(frame, n_samples, &mut out)
                            .unwrap_or_else(|e| panic!("{label}: {e}"));
                        let c_packet = c
                            .encode_float(frame, n_samples, len)
                            .unwrap_or_else(|e| panic!("{label}: C error {e}"));
                        assert!(n <= len, "{label}: {n} bytes");
                        if !vbr {
                            assert_eq!(n, c_packet.len(), "{label}: packet size");
                        }
                        // `bitrate()` depends on which streams had room for
                        // audio, and under constant bitrate on what the
                        // streams left the last one. Both are the same in
                        // the two encoders when every stream is CELT (the
                        // low-delay application) and its size is fixed:
                        // by a constant bitrate, by a buffer with room for
                        // everything, or by one with room for nothing.
                        let fixed_sizes =
                            !vbr || len == smallest || len == max_packet_bytes(streams);
                        if application == Application::RestrictedLowDelay && fixed_sizes {
                            assert_eq!(rust.bitrate(), c.bitrate(), "{label}: bitrate()");
                        }
                        let decoded = dec
                            .decode_float(Some(&out[..n]), n_samples, false)
                            .unwrap_or_else(|e| panic!("{label}: C decode error {e}"));
                        assert_eq!(decoded.len(), n_samples * channels, "{label}");
                        assert_eq!(
                            rust.final_range(),
                            dec.final_range(),
                            "{label}: final range"
                        );
                        packets += 1;
                    }
                }
                // One byte less than the smallest packet.
                let mut out = vec![0u8; smallest - 1];
                assert!(rust.encode(&pcm, frame_size, &mut out).is_err());
                assert!(c.encode_float(&pcm, frame_size, smallest - 1).is_err());
            }
        }
    }
    println!("{packets} packets");
}

/// One more channel or stream than the largest layout.
#[test]
fn layouts_beyond_the_largest_are_rejected_like_libopus() {
    let full: Vec<u8> = (0..=255).collect();
    // (channels, streams, coupled streams).
    for (channels, streams, coupled) in [
        (255usize, 255usize, 0usize),
        (256, 255, 0),
        (256, 256, 0),
        (255, 128, 127),
        (255, 128, 128),
        (255, 129, 127),
        (254, 127, 127),
        (254, 128, 127),
        (255, 127, 128),
        (usize::MAX, 1, 0),
        (2, usize::MAX, 0),
        (2, 1, usize::MAX),
    ] {
        let rust = MultistreamEncoder::new(
            48_000,
            channels,
            streams,
            coupled,
            &full,
            Application::Audio,
        );
        // The C API takes `int`; a count that does not fit cannot be given.
        let fits = [channels, streams, coupled]
            .iter()
            .all(|&v| v <= i32::MAX as usize);
        if fits {
            let c = CMsEncoder::generic(48_000, channels, streams, coupled, &full, APP_AUDIO);
            assert_eq!(
                rust.is_ok(),
                c.is_ok(),
                "{channels} channels, {streams} streams, {coupled} coupled: opus1 {:?}, C {:?}",
                rust.as_ref().err(),
                c.as_ref().err()
            );
        } else {
            assert!(
                rust.is_err(),
                "{channels} channels, {streams} streams, {coupled} coupled"
            );
        }
    }
    for (channels, family) in [
        (usize::MAX, 255u8),
        (usize::MAX, 1),
        (0, 0),
        (0, 255),
        (3, 0),
    ] {
        assert!(
            MultistreamEncoder::new_surround(48_000, channels, family, Application::Audio).is_err(),
            "{channels} channels, family {family}"
        );
    }
    // An empty mapping, and one that is a channel short.
    assert!(MultistreamEncoder::new(48_000, 1, 1, 0, &[], Application::Audio).is_err());
    assert!(
        MultistreamEncoder::new(48_000, 255, 255, 0, &full[..254], Application::Audio).is_err()
    );
}

#[test]
fn empty_input_and_empty_output_are_errors() {
    for case in CASES {
        for fs in [48_000, 8_000] {
            let (mut rust, mut c) = encoders(case, fs, Application::Audio);
            let frame_size = fs as usize / 50;
            let channels = case.channels;
            let pcm = voiced(fs, channels, frame_size);
            let pcm16 = to_i16(&pcm);
            let mut out = vec![0u8; max_packet_bytes(c.streams)];
            let name = case.name;

            // No input at all, and input of every length short of a frame
            // near both ends.
            for len in [0, 1, channels, frame_size * channels - 1] {
                assert!(
                    rust.encode(&pcm[..len], frame_size, &mut out).is_err(),
                    "{name}: {len}"
                );
                assert!(
                    rust.encode_i16(&pcm16[..len], frame_size, &mut out)
                        .is_err(),
                    "{name}: {len}"
                );
            }
            // No output buffer. libopus: OPUS_BUFFER_TOO_SMALL.
            assert!(rust.encode(&pcm, frame_size, &mut []).is_err(), "{name}");
            assert!(
                rust.encode_i16(&pcm16, frame_size, &mut []).is_err(),
                "{name}"
            );
            assert_eq!(
                c.encode_float(&pcm, frame_size, 0),
                Err(OPUS_BUFFER_TOO_SMALL)
            );
            // A frame of no samples. libopus: OPUS_BAD_ARG.
            assert!(rust.encode(&pcm, 0, &mut out).is_err(), "{name}");
            assert!(rust.encode(&[], 0, &mut out).is_err(), "{name}");
            assert!(rust.encode_i16(&[], 0, &mut []).is_err(), "{name}");
            assert_eq!(c.encode_float(&pcm, 0, out.len()), Err(OPUS_BAD_ARG));

            // None of this moved the encoder: its first packet is that of a
            // new encoder.
            let mut fresh = rust_encoder(case, fs, Application::Audio);
            let mut want = vec![0u8; out.len()];
            let n = rust.encode(&pcm, frame_size, &mut out).unwrap();
            let m = fresh.encode(&pcm, frame_size, &mut want).unwrap();
            assert_eq!(
                &out[..n],
                &want[..m],
                "{name}: first packet after the errors"
            );

            // Input longer than a frame: the rest is ignored.
            let long: Vec<f32> = pcm.iter().chain(&[0.9f32; 77]).copied().collect();
            let mut a = rust_encoder(case, fs, Application::Audio);
            let n = a.encode(&long, frame_size, &mut out).unwrap();
            assert_eq!(&out[..n], &want[..m], "{name}: input longer than a frame");
        }
    }
}

#[test]
fn settings_at_the_ends_of_their_types() {
    let case = Case::family("7.1", 8, 1);
    let (mut rust, mut c) = encoders(&case, 48_000, Application::Audio);
    for value in [i32::MIN, i32::MIN + 1, -1, 0, 1, i32::MAX - 1, i32::MAX] {
        assert_eq!(
            rust.set_bitrate(Bitrate::BitsPerSecond(value)).is_ok(),
            value > 0,
            "bitrate {value}"
        );
        assert_eq!(
            rust.set_complexity(value).is_ok(),
            c.try_ctl(audiopus_sys::OPUS_SET_COMPLEXITY_REQUEST, value) == 0,
            "complexity {value}"
        );
        assert_eq!(
            rust.set_packet_loss_perc(value).is_ok(),
            c.try_ctl(audiopus_sys::OPUS_SET_PACKET_LOSS_PERC_REQUEST, value) == 0,
            "loss {value}"
        );
    }
    // What was refused changed nothing.
    assert_eq!(rust.bitrate_setting(), Bitrate::BitsPerSecond(8 * 750_000));
    assert_eq!(
        rust.complexity(),
        c.get(audiopus_sys::OPUS_GET_COMPLEXITY_REQUEST)
    );
    assert_eq!(
        rust.packet_loss_perc(),
        c.get(audiopus_sys::OPUS_GET_PACKET_LOSS_PERC_REQUEST)
    );
    // The encoder still works, at the highest bitrate.
    c.set_bitrate(i32::MAX);
    let pcm = voiced(48_000, 8, 960);
    let mut out = vec![0u8; max_packet_bytes(c.streams)];
    let n = rust.encode(&pcm, 960, &mut out).unwrap();
    c.encode_float(&pcm, 960, out.len()).unwrap();
    assert_eq!(rust.bitrate(), c.bitrate());
    let decoded = c_decoder(&rust)
        .decode_float(Some(&out[..n]), 960, false)
        .unwrap();
    assert_eq!(decoded.len(), 960 * 8);
    // A stream that does not exist.
    for stream in [c.streams, c.streams + 1, usize::MAX] {
        assert!(rust.stream_encoder(stream).is_none());
        assert!(rust.stream_encoder_mut(stream).is_none());
    }
}
