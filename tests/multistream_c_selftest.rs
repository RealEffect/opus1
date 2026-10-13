//! Self-test of `tests/ms_support`.
//!
//! Every test here runs C libopus against C libopus, or a helper against a C
//! function or a known value. Nothing uses opus-rs. When this file passes, a
//! failure in another multistream test is not a fault of the helpers.
//!
//! What is checked:
//!
//! - the encoder and decoder wrappers: a 5.1 round trip routes every channel
//!   to the right place, and both sides report the same final range;
//! - the per-stream encoder state: stream bitrates can be read and add up;
//! - the packet reader: stream count, TOC bytes, sizes, and that each stream
//!   it cuts out decodes to exactly what the multistream decoder outputs;
//! - packet loss: concealment, and rebuilding a lost frame from the forward
//!   error correction data of the next packet;
//! - the private requests: each of them changes what the C encoder does;
//! - the signals and the measures, on values that are known in advance.

mod ms_support;

use audiopus_sys as sys;
use ms_support::*;

const FS: i32 = 48_000;
/// 20 ms at 48 kHz.
const FRAME: usize = 960;

/// Encodes `pcm` with `enc` and decodes every packet with a new C decoder of
/// the same layout. Checks that both sides agree on the final range of every
/// packet. Returns the packets and the decoded signal.
fn round_trip(enc: &mut CMsEncoder, pcm: &[f32], frame_size: usize) -> (Vec<Vec<u8>>, Vec<f32>) {
    let mut dec = CMsDecoder::for_encoder(enc);
    let mut packets = Vec::new();
    let mut out = Vec::new();
    for (f, frame) in pcm.chunks_exact(frame_size * enc.channels).enumerate() {
        let packet = enc
            .encode_float(frame, frame_size, max_packet_bytes(enc.streams))
            .unwrap_or_else(|e| panic!("frame {f}: encode failed: {e}"));
        let decoded = dec
            .decode_float(Some(&packet), frame_size, false)
            .unwrap_or_else(|e| panic!("frame {f}: decode failed: {e}"));
        assert_eq!(
            decoded.len(),
            frame_size * enc.channels,
            "frame {f}: decoded length"
        );
        assert_eq!(
            enc.final_range(),
            dec.final_range(),
            "frame {f}: final range"
        );
        packets.push(packet);
        out.extend_from_slice(&decoded);
    }
    (packets, out)
}

/// The decoded channel (stream, channel inside the stream) that output
/// channel `j` takes its samples from, or `None` for a silent channel.
fn source_of(mapping: &[u8], coupled: usize, j: usize) -> Option<(usize, usize)> {
    let m = mapping[j] as usize;
    if m == 255 {
        None
    } else if m < 2 * coupled {
        Some((m / 2, m % 2))
    } else {
        Some((m - coupled, 0))
    }
}

#[test]
fn c_5_1_round_trip_routes_every_channel() {
    let mut enc = CMsEncoder::surround(FS, 6, 1, APP_AUDIO).unwrap();
    assert_eq!((enc.streams, enc.coupled), (4, 2));
    assert_eq!(enc.mapping, [0, 4, 1, 2, 3, 5]);
    assert_eq!(enc.lookahead(), 312);
    enc.set_bitrate(256_000);

    let pcm = tones(FS, 6, 1, 50 * FRAME);
    let (packets, out) = round_trip(&mut enc, &pcm, FRAME);
    assert_eq!(packets.len(), 50);

    // The strongest test tone on channel c is the tone of channel c.
    let skip = 10 * FRAME;
    let dominant = dominant_tones(&out, FS, 6, 1, skip);
    for (c, &(tone, ratio)) in dominant.iter().enumerate() {
        println!(
            "channel {c}: strongest tone is that of channel {tone}, {ratio:.0} times the next"
        );
        assert_eq!(tone, c, "channel {c} carries the tone of channel {tone}");
        assert!(
            ratio >= 1000.0,
            "channel {c}: tone only {ratio:.1} times stronger than the next"
        );
    }
    // The tones come back at their level. A sine of amplitude a has the power a * a / 2.
    let want = (TONE_AMPLITUDE * TONE_AMPLITUDE / 2.0) as f64;
    for c in 0..6 {
        let x = channel(&out, 6, c);
        let power = goertzel_power(&x[skip..], tone_hz(c, 6, 1), FS);
        assert!(
            (power / want - 1.0).abs() < 0.1,
            "channel {c}: tone power {power:.5}, expected {want:.5}"
        );
    }
    // The decoded signal follows the input.
    for c in 0..6 {
        let (snr, lag) = best_lag_snr(&channel(&pcm, 6, c), &channel(&out, 6, c), 400, skip);
        println!("channel {c}: SNR {snr:.1} dB at lag {lag}");
        assert!(snr > 25.0, "channel {c}: SNR {snr:.1} dB");
    }
}

#[test]
fn c_decoder_delay_is_the_encoder_lookahead() {
    // A signal that never repeats has one best lag: the delay of the codec.
    let mut enc = CMsEncoder::surround(FS, 4, 1, APP_AUDIO).unwrap();
    enc.set_bitrate(256_000);
    let lookahead = enc.lookahead() as isize;
    let pcm = voiced(FS, 4, 50 * FRAME);
    let (_, out) = round_trip(&mut enc, &pcm, FRAME);
    for c in 0..4 {
        let (snr, lag) = best_lag_snr(&channel(&pcm, 4, c), &channel(&out, 4, c), 400, 10 * FRAME);
        println!("channel {c}: SNR {snr:.1} dB at lag {lag}");
        // Coding noise can move the best match by one sample.
        assert!(
            (lag - lookahead).abs() <= 1,
            "channel {c}: lag {lag}, lookahead {lookahead}"
        );
        assert!(snr > 10.0, "channel {c}: SNR {snr:.1} dB");
    }
}

#[test]
fn c_5_1_packet_structure() {
    let mut enc = CMsEncoder::surround(FS, 6, 1, APP_AUDIO).unwrap();
    enc.set_bitrate(256_000);
    let pcm = tones(FS, 6, 1, 20 * FRAME);
    for packet in enc.encode_all(&pcm, FRAME) {
        let parts = split_streams(&packet, 4).unwrap();
        assert_eq!(parts.len(), 4);
        assert_eq!(parts.iter().map(|p| p.used).sum::<usize>(), packet.len());
        for (s, part) in parts.iter().enumerate() {
            let t = toc(part.toc);
            assert_eq!(part.samples(FS), FRAME, "stream {s}: {part}");
            assert_eq!(part.frames(), 1, "stream {s}: {part}");
            // In surround mode libopus forces CELT stereo on coupled streams.
            assert_eq!(t.stereo, s < 2, "stream {s}: {part}");
            if s < 2 {
                assert_eq!(t.mode, Mode::Celt, "stream {s}: {part}");
                assert_eq!(t.bandwidth, Bandwidth::Fb, "stream {s}: {part}");
            }
            // Every stream but the last has a length field of 1 or 2 bytes.
            let header = part.used - part.payload_len - part.padding;
            let want_header = if s == 3 {
                1
            } else if part.payload_len < 252 {
                2
            } else {
                3
            };
            assert_eq!(header, want_header, "stream {s}: {part}");
        }
        // The LFE stream is the last one: CELT, narrowband, mono, and small.
        let lfe = toc(parts[3].toc);
        assert_eq!(
            (lfe.mode, lfe.bandwidth, lfe.stereo),
            (Mode::Celt, Bandwidth::Nb, false)
        );
        assert!(parts[3].payload_len < parts[2].payload_len);
        assert_eq!(
            describe_packet(&packet, 4).matches('[').count(),
            4,
            "{}",
            describe_packet(&packet, 4)
        );
    }
}

#[test]
fn c_stream_bitrates_are_readable() {
    let mut enc = CMsEncoder::surround(FS, 6, 1, APP_AUDIO).unwrap();
    enc.set_bitrate(256_000);
    let pcm = tones(FS, 6, 1, FRAME);
    enc.encode_float(&pcm, FRAME, max_packet_bytes(4)).unwrap();
    // The values of design probe P4/P9 for 5.1 at 256 kb/s, 48 kHz, 20 ms.
    assert_eq!(enc.stream_bitrates(), [95_120, 95_120, 57_560, 8_195]);
    assert_eq!(enc.bitrate(), 95_120 + 95_120 + 57_560 + 8_195);
    // Two new encoders give the same first packet, stream by stream.
    let first = enc.stream_final_range(0);
    let mut again = CMsEncoder::surround(FS, 6, 1, APP_AUDIO).unwrap();
    again.set_bitrate(256_000);
    again
        .encode_float(&pcm, FRAME, max_packet_bytes(4))
        .unwrap();
    assert_eq!(again.stream_final_range(0), first);
    let combined = (0..4).fold(0u32, |acc, s| acc ^ again.stream_final_range(s));
    assert_eq!(combined, again.final_range());
    // A control on one stream reaches only that stream.
    assert_eq!(enc.stream_ctl(3, sys::OPUS_SET_COMPLEXITY_REQUEST, 3), 0);
    assert_eq!(enc.stream_get(3, sys::OPUS_GET_COMPLEXITY_REQUEST), 3);
    assert_eq!(enc.stream_get(2, sys::OPUS_GET_COMPLEXITY_REQUEST), 9);
    // A control on the multistream encoder reaches every stream.
    enc.set_complexity(5);
    assert!((0..4).all(|s| enc.stream_get(s, sys::OPUS_GET_COMPLEXITY_REQUEST) == 5));
    assert_eq!(
        enc.try_ctl(sys::OPUS_SET_COMPLEXITY_REQUEST, 11),
        OPUS_BAD_ARG
    );
    assert_eq!(enc.try_ctl(sys::OPUS_SET_BITRATE_REQUEST, 0), OPUS_BAD_ARG);
}

#[test]
fn c_cbr_packets_have_one_size() {
    for (channels, family, bitrate, want) in [
        (6usize, 1, 256_000, 640usize),
        (8, 1, 320_000, 800),
        (3, 255, 96_000, 240),
    ] {
        let mut enc = CMsEncoder::surround(FS, channels, family, APP_AUDIO).unwrap();
        enc.set_bitrate(bitrate).set_vbr(false);
        let pcm = tones(FS, channels, family, 20 * FRAME);
        let (packets, _) = round_trip(&mut enc, &pcm, FRAME);
        for p in &packets {
            assert_eq!(p.len(), want, "{channels} channels at {bitrate} b/s");
            // The streams fill the packet exactly.
            let parts = split_streams(p, enc.streams).unwrap();
            assert_eq!(parts.iter().map(|s| s.used).sum::<usize>(), want);
        }
    }
}

/// Encodes `pcm` and decodes every packet twice: with the C multistream
/// decoder, and stream by stream with single-stream C decoders after
/// `split_streams`. Both must give the same samples and the same final
/// range. Returns how many streams used each way of packing frames; see
/// `PACKINGS` for the order.
fn check_split_against_ms_decoder(
    enc: &mut CMsEncoder,
    pcm: &[f32],
    frame_size: usize,
) -> [usize; 7] {
    let (streams, coupled, channels) = (enc.streams, enc.coupled, enc.channels);
    let mut dec = CMsDecoder::for_encoder(enc);
    let mut stream_decoders: Vec<CDecoder> = (0..streams)
        .map(|s| CDecoder::new(FS, if s < coupled { 2 } else { 1 }))
        .collect();
    let mut codes = [0usize; 7];
    for (f, frame) in pcm.chunks_exact(frame_size * channels).enumerate() {
        let packet = enc
            .encode_float(frame, frame_size, max_packet_bytes(streams))
            .unwrap();
        let out = dec.decode_float(Some(&packet), frame_size, false).unwrap();
        assert_eq!(out.len(), frame_size * channels, "frame {f}");
        assert_eq!(
            enc.final_range(),
            dec.final_range(),
            "frame {f}: final range"
        );

        let parts = split_streams(&packet, streams)
            .unwrap_or_else(|e| panic!("frame {f}: {e}: {:02x?}", &packet[..packet.len().min(16)]));
        assert_eq!(
            parts.iter().map(|p| p.used).sum::<usize>(),
            packet.len(),
            "frame {f}"
        );
        let mut stream_pcm = Vec::new();
        let mut range = 0u32;
        for (s, part) in parts.iter().enumerate() {
            let packing = match toc(part.toc).code {
                3 if part.packet[1] & 0x80 != 0 => 4,
                code => code as usize,
            };
            codes[packing] += 1;
            if part.padding > 0 {
                codes[5] += 1;
            }
            if part.frame_sizes.iter().any(|&n| n >= 252) {
                codes[6] += 1;
            }
            assert_eq!(part.samples(FS), frame_size, "frame {f} stream {s}: {part}");
            // The C parser reads the stream packet the same way.
            let mut c_toc = 0u8;
            let mut c_sizes = [0i16; 48];
            let mut payload_offset = 0;
            let count = unsafe {
                sys::opus_packet_parse(
                    part.packet.as_ptr(),
                    part.packet.len() as i32,
                    &mut c_toc,
                    std::ptr::null_mut(),
                    c_sizes.as_mut_ptr(),
                    &mut payload_offset,
                )
            };
            assert_eq!(
                count as usize,
                part.frames(),
                "frame {f} stream {s}: {part}"
            );
            assert_eq!(c_toc, part.toc);
            let c_sizes: Vec<usize> = c_sizes[..count as usize]
                .iter()
                .map(|&v| v as usize)
                .collect();
            assert_eq!(c_sizes, part.frame_sizes, "frame {f} stream {s}: {part}");
            assert_eq!(
                parse_stream(&part.packet).unwrap().frame_sizes,
                part.frame_sizes
            );

            let decoded = stream_decoders[s]
                .decode_float(Some(&part.packet), frame_size)
                .unwrap_or_else(|e| {
                    panic!("frame {f} stream {s}: single-stream decode failed: {e}")
                });
            assert_eq!(decoded.len(), frame_size * stream_decoders[s].channels);
            range ^= stream_decoders[s].final_range();
            stream_pcm.push(decoded);
        }
        assert_eq!(
            range,
            dec.final_range(),
            "frame {f}: final range of the split streams"
        );
        for j in 0..channels {
            for i in 0..frame_size {
                let want = match source_of(&enc.mapping, coupled, j) {
                    Some((s, ch)) => stream_pcm[s][i * stream_decoders[s].channels + ch],
                    None => 0.0,
                };
                assert_eq!(
                    out[i * channels + j],
                    want,
                    "frame {f} channel {j} sample {i}"
                );
            }
        }
    }
    codes
}

/// The ways of packing frames that `check_split_against_ms_decoder` counts.
const PACKINGS: [&str; 7] = [
    "code 0 (one frame)",
    "code 1 (two equal frames)",
    "code 2 (two different frames)",
    "code 3 with equal frames",
    "code 3 with different frames",
    "padding",
    "a frame of 252 bytes or more",
];

#[test]
fn split_streams_decode_like_the_c_multistream_decoder() {
    let mut codes = [0usize; 7];
    // (channels, family, bitrate)
    for (channels, family, bitrate) in [
        (6usize, 1, 256_000),
        (8, 1, 320_000),
        (3, 255, 96_000),
        (6, 2, 192_000),
        (6, 1, 700_000),
    ] {
        for frame_size in [FRAME / 2, FRAME, 2 * FRAME, 3 * FRAME] {
            for vbr in [true, false] {
                for gated in [false, true] {
                    let mut enc = CMsEncoder::surround(FS, channels, family, APP_AUDIO).unwrap();
                    enc.set_bitrate(bitrate).set_vbr(vbr);
                    let mut pcm = voiced(FS, channels, 12 * FRAME);
                    if gated {
                        // Silence in every second 20 ms: with variable bitrate
                        // the frames of one packet then differ in size.
                        for block in pcm.chunks_mut(FRAME * channels).skip(1).step_by(2) {
                            block.fill(0.0);
                        }
                    }
                    let seen = check_split_against_ms_decoder(&mut enc, &pcm, frame_size);
                    for (total, n) in codes.iter_mut().zip(seen) {
                        *total += n;
                    }
                }
            }
        }
    }
    for (name, n) in PACKINGS.iter().zip(codes) {
        println!("{n:5} streams with {name}");
    }
    // The cases above must cover every way of packing frames.
    assert!(codes.iter().all(|&n| n > 0), "packings seen: {codes:?}");
}

#[test]
fn split_streams_reads_hand_made_packets() {
    // Each first stream below is self-delimited; the second stream is one
    // CELT fullband 20 ms mono frame of 2 bytes. The first stream uses CELT
    // fullband 20 ms frames too, except where two or three frames must fit.
    let celt_fb_20 = 0xF8u8; // config 31, mono, code 0
    let celt_fb_10 = 0xF0u8; // config 30
    let celt_fb_5 = 0xE8u8; // config 29
    let last = [celt_fb_20, 0x55, 0x66];
    struct Case {
        name: &'static str,
        first: Vec<u8>,
        last_toc: u8,
        frame_sizes: Vec<usize>,
        padding: usize,
        standalone: Vec<u8>,
    }
    let cases = [
        Case {
            name: "code 0",
            first: vec![celt_fb_20, 3, 1, 2, 3],
            last_toc: celt_fb_20,
            frame_sizes: vec![3],
            padding: 0,
            standalone: vec![celt_fb_20, 1, 2, 3],
        },
        Case {
            name: "code 0, empty frame",
            first: vec![celt_fb_20, 0],
            last_toc: celt_fb_20,
            frame_sizes: vec![0],
            padding: 0,
            standalone: vec![celt_fb_20],
        },
        Case {
            name: "code 1",
            first: vec![celt_fb_10 | 1, 2, 1, 2, 3, 4],
            last_toc: celt_fb_20,
            frame_sizes: vec![2, 2],
            padding: 0,
            standalone: vec![celt_fb_10 | 1, 1, 2, 3, 4],
        },
        Case {
            name: "code 2",
            first: vec![celt_fb_10 | 2, 1, 3, 9, 1, 2, 3],
            last_toc: celt_fb_20,
            frame_sizes: vec![1, 3],
            padding: 0,
            standalone: vec![celt_fb_10 | 2, 1, 9, 1, 2, 3],
        },
        Case {
            name: "code 3, equal sizes",
            first: vec![celt_fb_5 | 3, 4, 2, 1, 2, 3, 4, 5, 6, 7, 8],
            last_toc: celt_fb_20,
            frame_sizes: vec![2, 2, 2, 2],
            padding: 0,
            standalone: vec![celt_fb_5 | 3, 4, 1, 2, 3, 4, 5, 6, 7, 8],
        },
        Case {
            name: "code 3, equal sizes, padding",
            first: vec![celt_fb_10 | 3, 0x40 | 2, 3, 2, 1, 2, 3, 4, 0, 0, 0],
            last_toc: celt_fb_20,
            frame_sizes: vec![2, 2],
            padding: 3,
            standalone: vec![celt_fb_10 | 3, 0x40 | 2, 3, 1, 2, 3, 4, 0, 0, 0],
        },
        Case {
            name: "code 3, different sizes",
            first: vec![celt_fb_5 | 3, 0x80 | 4, 1, 2, 0, 3, 9, 8, 8, 7, 7, 7],
            last_toc: celt_fb_20,
            frame_sizes: vec![1, 2, 0, 3],
            padding: 0,
            standalone: vec![celt_fb_5 | 3, 0x80 | 4, 1, 2, 0, 9, 8, 8, 7, 7, 7],
        },
        Case {
            name: "code 3, different sizes, padding",
            first: vec![celt_fb_10 | 3, 0xC0 | 2, 1, 2, 3, 9, 9, 8, 8, 8, 0],
            last_toc: celt_fb_20,
            frame_sizes: vec![2, 3],
            padding: 1,
            standalone: vec![celt_fb_10 | 3, 0xC0 | 2, 1, 2, 9, 9, 8, 8, 8, 0],
        },
    ];
    // Padding of 257 bytes: its length is written as 255 (meaning 254 and
    // "more follows") and 3.
    let mut long_padding = vec![celt_fb_20 | 3, 0x40 | 1, 255, 3, 2, 1, 2];
    long_padding.extend_from_slice(&[0; 257]);
    let mut long_padding_alone = vec![celt_fb_20 | 3, 0x40 | 1, 255, 3, 1, 2];
    long_padding_alone.extend_from_slice(&[0; 257]);
    // A frame of 300 bytes: its length takes two bytes, 252 + 300 % 4 and (300 - 252) / 4.
    let mut long_frame = vec![celt_fb_20, 252, 12];
    long_frame.extend((0..300).map(|i| i as u8));
    let mut long_frame_alone = vec![celt_fb_20];
    long_frame_alone.extend((0..300).map(|i| i as u8));
    let cases = cases.into_iter().chain([
        Case {
            name: "code 3, long padding",
            first: long_padding,
            last_toc: celt_fb_20,
            frame_sizes: vec![2],
            padding: 257,
            standalone: long_padding_alone,
        },
        Case {
            name: "code 0, length of two bytes",
            first: long_frame,
            last_toc: celt_fb_20,
            frame_sizes: vec![300],
            padding: 0,
            standalone: long_frame_alone,
        },
    ]);
    let mut dec = CMsDecoder::new(FS, 2, 2, 0, &[0, 1]).unwrap();
    for case in cases {
        let mut packet = case.first.clone();
        packet.extend_from_slice(&last);
        let parts = split_streams(&packet, 2).unwrap_or_else(|e| panic!("{}: {e}", case.name));
        assert_eq!(parts[0].toc, case.first[0], "{}", case.name);
        assert_eq!(parts[0].frame_sizes, case.frame_sizes, "{}", case.name);
        assert_eq!(
            parts[0].payload_len,
            case.frame_sizes.iter().sum::<usize>(),
            "{}",
            case.name
        );
        assert_eq!(parts[0].padding, case.padding, "{}", case.name);
        assert_eq!(parts[0].used, case.first.len(), "{}", case.name);
        assert_eq!(parts[0].packet, case.standalone, "{}", case.name);
        assert_eq!(parts[1].toc, case.last_toc, "{}", case.name);
        assert_eq!(parts[1].packet, last, "{}", case.name);
        assert_eq!(parts[1].frame_sizes, [2], "{}", case.name);
        // The C decoder accepts the same bytes as two streams of 20 ms.
        let decoded = dec.decode_float(Some(&packet), MAX_FRAME, false);
        assert_eq!(decoded.map(|d| d.len()), Ok(2 * FRAME), "{}", case.name);
        // The C parser accepts the stream packet and finds the same frames.
        let mut c_sizes = [0i16; 48];
        let count = unsafe {
            sys::opus_packet_parse(
                parts[0].packet.as_ptr(),
                parts[0].packet.len() as i32,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                c_sizes.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        assert_eq!(count as usize, case.frame_sizes.len(), "{}", case.name);
        let c_sizes: Vec<usize> = c_sizes[..count as usize]
            .iter()
            .map(|&v| v as usize)
            .collect();
        assert_eq!(c_sizes, case.frame_sizes, "{}", case.name);
    }
    // Streams of different durations are rejected by both.
    let packet = [celt_fb_10, 1, 7, celt_fb_20, 0x55];
    assert!(split_streams(&packet, 2).is_err());
    assert_eq!(
        dec.decode_float(Some(&packet), MAX_FRAME, false),
        Err(OPUS_INVALID_PACKET)
    );
}

#[test]
fn split_streams_rejects_what_the_c_decoder_rejects() {
    let celt_fb_20 = 0xF8u8;
    // A valid first stream: one frame of 20 ms in one byte. Every last
    // stream below lasts 20 ms too, or is invalid for another reason.
    let first = [celt_fb_20, 1, 0x11];
    let frames = |toc: u8, count: u8, bytes: usize| -> Vec<u8> {
        let mut v = vec![toc | 3, count];
        v.resize(2 + bytes, 0x22);
        v
    };
    // (name, first stream, last stream, valid)
    let cases: Vec<(&str, Vec<u8>, Vec<u8>, bool)> = vec![
        ("plain", first.to_vec(), vec![celt_fb_20, 0x22], true),
        (
            "last stream is only a TOC byte",
            first.to_vec(),
            vec![celt_fb_20],
            true,
        ),
        ("no last stream", first.to_vec(), vec![], false),
        (
            "first stream longer than the packet",
            vec![celt_fb_20, 9, 0x11],
            vec![celt_fb_20, 0x22],
            false,
        ),
        (
            "first stream has no length",
            vec![celt_fb_20],
            vec![],
            false,
        ),
        (
            "first stream length cut in half",
            vec![celt_fb_20, 252],
            vec![],
            false,
        ),
        // The largest frame is 1275 bytes.
        (
            "frame of 1275 bytes",
            first.to_vec(),
            [vec![celt_fb_20], vec![0x22; 1275]].concat(),
            true,
        ),
        (
            "frame of 1276 bytes",
            first.to_vec(),
            [vec![celt_fb_20], vec![0x22; 1276]].concat(),
            false,
        ),
        // Two equal frames need an even number of bytes.
        (
            "code 1, even",
            first.to_vec(),
            vec![0xF0 | 1, 1, 2, 3, 4],
            true,
        ),
        (
            "code 1, odd",
            first.to_vec(),
            vec![0xF0 | 1, 1, 2, 3],
            false,
        ),
        // The first of two frames cannot be longer than the stream.
        ("code 2", first.to_vec(), vec![0xF0 | 2, 2, 1, 2, 3], true),
        (
            "code 2, first frame too long",
            first.to_vec(),
            vec![0xF0 | 2, 4, 1, 2, 3],
            false,
        ),
        // A packet lasts at most 120 ms: 6 frames of 20 ms.
        (
            "6 frames of 20 ms",
            [vec![celt_fb_20 | 3, 6, 1], vec![0x11; 6]].concat(),
            frames(celt_fb_20, 6, 6),
            true,
        ),
        (
            "7 frames of 20 ms",
            [vec![celt_fb_20 | 3, 7, 1], vec![0x11; 7]].concat(),
            frames(celt_fb_20, 7, 7),
            false,
        ),
        ("no frames", first.to_vec(), frames(celt_fb_20, 0, 4), false),
        // Equal frames must divide the bytes evenly.
        (
            "code 3, 2 frames in 4 bytes",
            first.to_vec(),
            frames(0xF0, 2, 4),
            true,
        ),
        (
            "code 3, 2 frames in 5 bytes",
            first.to_vec(),
            frames(0xF0, 2, 5),
            false,
        ),
        // Padding cannot be longer than the stream.
        (
            "padding of 2 bytes",
            first.to_vec(),
            vec![celt_fb_20 | 3, 0x40 | 1, 2, 0x22, 0, 0],
            true,
        ),
        (
            "padding of 9 bytes in 3",
            first.to_vec(),
            vec![celt_fb_20 | 3, 0x40 | 1, 9, 0x22, 0, 0],
            false,
        ),
        (
            "padding length missing",
            first.to_vec(),
            vec![celt_fb_20 | 3, 0x40 | 1],
            false,
        ),
        // Frame sizes of a code 3 packet cannot be longer than the stream.
        (
            "code 3, sizes fit",
            first.to_vec(),
            vec![0xE8 | 3, 0x80 | 4, 1, 1, 1, 7, 8, 9, 5],
            true,
        ),
        (
            "code 3, sizes too long",
            first.to_vec(),
            vec![0xE8 | 3, 0x80 | 4, 1, 1, 9, 7, 8, 9, 5],
            false,
        ),
    ];
    let mut dec = CMsDecoder::new(FS, 2, 2, 0, &[0, 1]).unwrap();
    for (name, first, last, valid) in cases {
        let packet = [first, last].concat();
        let c = dec.decode_float(Some(&packet), MAX_FRAME, false);
        assert_eq!(
            c.is_ok(),
            valid,
            "{name}: the C decoder gives {:?}",
            c.map(|pcm| pcm.len())
        );
        let mine = split_streams(&packet, 2);
        assert_eq!(
            mine.is_ok(),
            valid,
            "{name}: split_streams gives {:?}",
            mine.map(|p| p.len())
        );
    }
}

#[test]
fn split_streams_accepts_what_the_c_decoder_accepts() {
    // Damage real packets and compare accept or reject with the C decoder.
    let mut seed = 0x9E37_79B9u32;
    let mut next = move || {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        (seed >> 8) as usize
    };
    let (mut accepted, mut rejected) = (0usize, 0usize);
    for (channels, family, bitrate, frame_size) in [
        (6usize, 1, 64_000, FRAME),
        (6, 1, 96_000, 3 * FRAME),
        (3, 255, 48_000, 2 * FRAME),
    ] {
        let mut enc = CMsEncoder::surround(FS, channels, family, APP_AUDIO).unwrap();
        enc.set_bitrate(bitrate);
        let mut dec = CMsDecoder::for_encoder(&enc);
        let pcm = voiced(FS, channels, 6 * frame_size);
        let packets = enc.encode_all(&pcm, frame_size);
        let packet = packets.last().unwrap();
        let mut check = |bytes: &[u8], what: &str| {
            let mine = split_streams(bytes, enc.streams);
            let c = dec.decode_float(Some(bytes), MAX_FRAME, false);
            assert_eq!(
                mine.is_ok(),
                c.is_ok(),
                "{what}: split_streams gives {:?}, the C decoder gives {:?}",
                mine.as_ref().map(|p| p.len()),
                c.as_ref().map(|p| p.len())
            );
            if let (Ok(parts), Ok(pcm)) = (&mine, &c) {
                assert_eq!(
                    parts[0].samples(FS) * channels,
                    pcm.len(),
                    "{what}: duration"
                );
            }
            if c.is_ok() {
                accepted += 1;
            } else {
                rejected += 1;
            }
        };
        // Every truncation. An empty packet means "lost" to the C decoder, so start at 1.
        for len in 1..=packet.len() {
            check(
                &packet[..len],
                &format!("{channels} channels, first {len} bytes"),
            );
        }
        // One changed byte, mostly near the start where the headers are.
        for _ in 0..600 {
            let mut bytes = packet.clone();
            let at = if next() % 4 == 0 {
                next() % bytes.len()
            } else {
                next() % bytes.len().min(12)
            };
            bytes[at] = next() as u8;
            check(
                &bytes,
                &format!("{channels} channels, byte {at} set to {:#04x}", bytes[at]),
            );
        }
        // Wrong stream count on our side must not panic.
        for streams in 0..=enc.streams + 2 {
            let _ = split_streams(packet, streams);
        }
    }
    println!("damaged packets: {accepted} accepted and {rejected} rejected by both");
    assert!(
        accepted > 100 && rejected > 100,
        "{accepted} accepted, {rejected} rejected"
    );
}

#[test]
fn c_lost_packets_are_concealed() {
    let mut enc = CMsEncoder::surround(FS, 6, 1, APP_AUDIO).unwrap();
    enc.set_bitrate(128_000);
    let mut dec = CMsDecoder::for_encoder(&enc);
    let pcm = tones(FS, 6, 1, 10 * FRAME);
    for packet in enc.encode_all(&pcm, FRAME) {
        dec.decode_float(Some(&packet), FRAME, false).unwrap();
    }
    let concealed = dec.decode_float(None, FRAME, false).unwrap();
    assert_eq!(concealed.len(), FRAME * 6);
    assert!(concealed.iter().all(|v| v.is_finite()));
    assert!(concealed.iter().any(|&v| v != 0.0));
    // The 16-bit entry points work too.
    let mut enc = CMsEncoder::surround(FS, 2, 1, APP_AUDIO).unwrap();
    let mut dec = CMsDecoder::for_encoder(&enc);
    let pcm16 = to_i16(&tones(FS, 2, 1, FRAME));
    let packet = enc.encode_i16(&pcm16, FRAME, max_packet_bytes(1)).unwrap();
    assert_eq!(
        dec.decode_i16(Some(&packet), FRAME, false).unwrap().len(),
        FRAME * 2
    );
    assert_eq!(dec.stream_get(0, sys::OPUS_GET_SAMPLE_RATE_REQUEST), FS);
}

#[test]
fn split_streams_agrees_with_the_c_decoder_on_random_bytes() {
    // Packets that no encoder made: random bytes, with more of the values
    // that matter to the parser (frame counts, padding flags, the lengths
    // around 252 where a length field grows to two bytes).
    let mut seed = 0x0BAD_5EEDu32;
    let mut next = move |n: usize| {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        (seed >> 8) as usize % n
    };
    let (mut accepted, mut rejected) = (0usize, 0usize);
    for streams in 1..=4usize {
        let mapping: Vec<u8> = (0..streams as u8).collect();
        let mut dec = CMsDecoder::new(FS, streams, streams, 0, &mapping).unwrap();
        for _ in 0..20_000 {
            let len = 1 + next(24);
            let mut bytes: Vec<u8> = (0..len)
                .map(|_| match next(6) {
                    0 => [0u8, 1, 2, 3, 4, 251, 252, 253, 254, 255][next(10)],
                    1 => next(8) as u8,
                    _ => next(256) as u8,
                })
                .collect();
            // Mostly CELT streams, whose frames are short enough for 120 ms.
            if next(3) != 0 {
                bytes[0] |= 0x80;
            }
            let mine = split_streams(&bytes, streams);
            let c = dec.decode_float(Some(&bytes), MAX_FRAME, false);
            // libopus checks the packet structure before it decodes. A
            // payload it cannot decode gives another error than this one.
            let c_structure_ok = c != Err(OPUS_INVALID_PACKET);
            assert_eq!(
                mine.is_ok(),
                c_structure_ok,
                "{streams} streams, bytes {bytes:02x?}: split_streams gives {:?}, the C decoder gives {:?}",
                mine.as_ref().map(|p| p.len()),
                c.as_ref().map(|p| p.len())
            );
            if let (Ok(parts), Ok(pcm)) = (&mine, &c) {
                assert_eq!(parts.len(), streams);
                assert_eq!(parts.iter().map(|p| p.used).sum::<usize>(), bytes.len());
                assert_eq!(
                    parts[0].samples(FS) * streams,
                    pcm.len(),
                    "bytes {bytes:02x?}: duration"
                );
            }
            if c_structure_ok {
                accepted += 1;
            } else {
                rejected += 1;
            }
        }
    }
    println!("random packets: {accepted} accepted and {rejected} rejected by both");
    assert!(
        accepted > 5_000 && rejected > 5_000,
        "{accepted} accepted, {rejected} rejected"
    );
}

/// Decodes `packets` up to and including the frame `lost`, which did not
/// arrive, and returns the output for that frame. With `use_fec` the frame is
/// rebuilt from the forward error correction data of the packet after it;
/// otherwise the decoder conceals the loss on its own.
fn decode_with_one_loss(
    enc: &CMsEncoder,
    packets: &[Vec<u8>],
    lost: usize,
    frame_size: usize,
    use_fec: bool,
) -> Vec<f32> {
    let mut dec = CMsDecoder::for_encoder(enc);
    for packet in &packets[..lost] {
        dec.decode_float(Some(packet), frame_size, false).unwrap();
    }
    let out = if use_fec {
        dec.decode_float(Some(&packets[lost + 1]), frame_size, true)
    } else {
        dec.decode_float(None, frame_size, false)
    };
    out.unwrap()
}

#[test]
fn c_fec_rebuilds_a_lost_frame() {
    // Two mono streams of speech at 16 kHz. With forward error correction
    // (FEC) on, every SILK packet also carries a coarse copy of the frame
    // before it.
    let (fs, frame_size, channels) = (16_000, 320usize, 2usize);
    let pcm = voiced(fs, channels, 60 * frame_size);
    let lost_frames = [20usize, 25, 30, 35, 40, 45, 50];
    // Mean SNR in dB of the lost frames against the decode without loss.
    let mean_snr = |enc: &CMsEncoder, packets: &[Vec<u8>], use_fec: bool| -> f64 {
        let mut dec = CMsDecoder::for_encoder(enc);
        let clean: Vec<Vec<f32>> = packets
            .iter()
            .map(|p| dec.decode_float(Some(p), frame_size, false).unwrap())
            .collect();
        let mut sum = 0.0;
        for &lost in &lost_frames {
            let got = decode_with_one_loss(enc, packets, lost, frame_size, use_fec);
            assert_eq!(got.len(), frame_size * channels);
            for c in 0..channels {
                let reference = channel(&clean[lost], channels, c);
                sum += snr_at_lag(&reference, &channel(&got, channels, c), 0, 0);
            }
        }
        sum / (lost_frames.len() * channels) as f64
    };

    let mut enc = CMsEncoder::surround(fs, channels, 255, APP_VOIP).unwrap();
    enc.set_bitrate(32_000 * channels as i32)
        .set_inband_fec(true)
        .set_packet_loss_perc(20);
    assert_eq!(enc.get(sys::OPUS_GET_INBAND_FEC_REQUEST), 1);
    assert_eq!(enc.get(sys::OPUS_GET_PACKET_LOSS_PERC_REQUEST), 20);
    let packets = enc.encode_all(&pcm, frame_size);
    // FEC data exists only in SILK and Hybrid packets.
    let parts = split_streams(&packets[30], enc.streams).unwrap();
    assert!(
        parts.iter().all(|p| toc(p.toc).mode == Mode::Silk),
        "{}",
        describe_packet(&packets[30], enc.streams)
    );
    let concealed = mean_snr(&enc, &packets, false);
    let rebuilt = mean_snr(&enc, &packets, true);
    println!("lost frames: concealed {concealed:.1} dB, rebuilt from FEC data {rebuilt:.1} dB");
    // Measured with libopus 1.6.1: 7.6 dB concealed, 18.0 dB rebuilt.
    assert!(
        rebuilt > 14.0,
        "FEC rebuilt the frames with only {rebuilt:.1} dB"
    );
    assert!(
        rebuilt > concealed + 6.0,
        "FEC {rebuilt:.1} dB, concealment {concealed:.1} dB"
    );

    // Without FEC the encoder picks CELT for this bitrate, and CELT packets
    // carry no FEC data. A decode call that asks for it then conceals the
    // loss, with exactly the samples of a call without a packet.
    let mut enc = CMsEncoder::surround(fs, channels, 255, APP_VOIP).unwrap();
    enc.set_bitrate(32_000 * channels as i32);
    assert_eq!(enc.get(sys::OPUS_GET_INBAND_FEC_REQUEST), 0);
    let packets = enc.encode_all(&pcm, frame_size);
    let parts = split_streams(&packets[30], enc.streams).unwrap();
    assert!(parts.iter().all(|p| toc(p.toc).mode == Mode::Celt));
    for &lost in &lost_frames {
        assert_eq!(
            decode_with_one_loss(&enc, &packets, lost, frame_size, true),
            decode_with_one_loss(&enc, &packets, lost, frame_size, false),
            "frame {lost}"
        );
    }
}

/// TOC of every stream of the last packet when `enc` encodes `pcm`.
fn last_tocs(enc: &mut CMsEncoder, pcm: &[f32]) -> Vec<Toc> {
    let packets = enc.encode_all(pcm, FRAME);
    split_streams(packets.last().unwrap(), enc.streams)
        .unwrap()
        .iter()
        .map(|p| toc(p.toc))
        .collect()
}

/// A C single-stream encoder that is freed when dropped.
struct CEncoder(*mut sys::OpusEncoder);

impl Drop for CEncoder {
    fn drop(&mut self) {
        unsafe { sys::opus_encoder_destroy(self.0) }
    }
}

#[test]
fn private_requests_reach_the_c_stream_encoders() {
    // Three mono streams. Mapping family 255 leaves every stream encoder to
    // its own settings, so a control on one stream stays in force.
    let pcm = noise(3, 12 * FRAME, 0.2, 21);
    let new_encoder = || {
        let mut enc = CMsEncoder::surround(FS, 3, 255, APP_AUDIO).unwrap();
        enc.set_bitrate(3 * 48_000);
        enc
    };
    let default_modes: Vec<Mode> = last_tocs(&mut new_encoder(), &pcm)
        .iter()
        .map(|t| t.mode)
        .collect();

    // OPUS_SET_FORCE_MODE: one stream per mode. SILK alone codes at most
    // wideband, so that stream is limited to wideband first.
    let mut enc = new_encoder();
    let force = |enc: &mut CMsEncoder, s: usize, mode: i32| {
        assert_eq!(enc.stream_ctl(s, OPUS_SET_FORCE_MODE_REQUEST, mode), 0);
    };
    assert_eq!(
        enc.stream_ctl(
            2,
            sys::OPUS_SET_MAX_BANDWIDTH_REQUEST,
            sys::OPUS_BANDWIDTH_WIDEBAND
        ),
        0
    );
    force(&mut enc, 0, MODE_CELT_ONLY);
    force(&mut enc, 1, MODE_HYBRID);
    force(&mut enc, 2, MODE_SILK_ONLY);
    let tocs = last_tocs(&mut enc, &pcm);
    assert_eq!(
        [tocs[0].mode, tocs[1].mode, tocs[2].mode],
        [Mode::Celt, Mode::Hybrid, Mode::Silk]
    );
    // The other way round on the first two streams: the mode follows the
    // control, not the stream.
    force(&mut enc, 0, MODE_HYBRID);
    force(&mut enc, 1, MODE_CELT_ONLY);
    let tocs = last_tocs(&mut enc, &pcm);
    assert_eq!(
        [tocs[0].mode, tocs[1].mode, tocs[2].mode],
        [Mode::Hybrid, Mode::Celt, Mode::Silk]
    );
    // OPUS_AUTO gives the choice back to the encoder.
    force(&mut enc, 0, OPUS_AUTO);
    force(&mut enc, 1, OPUS_AUTO);
    let tocs = last_tocs(&mut enc, &pcm);
    assert_eq!([tocs[0].mode, tocs[1].mode], default_modes[..2]);
    // libopus checks the value, and it does not know a request near this one.
    assert_eq!(
        enc.stream_ctl(0, OPUS_SET_FORCE_MODE_REQUEST, 999),
        OPUS_BAD_ARG
    );
    assert_eq!(
        enc.stream_ctl(0, OPUS_SET_FORCE_MODE_REQUEST + 997, MODE_CELT_ONLY),
        OPUS_UNIMPLEMENTED
    );

    // OPUS_SET_LFE: the stream turns to narrowband CELT, the others do not.
    let mut enc = new_encoder();
    assert_eq!(enc.stream_ctl(2, OPUS_SET_LFE_REQUEST, 1), 0);
    let tocs = last_tocs(&mut enc, &pcm);
    assert_eq!(
        (tocs[2].mode, tocs[2].bandwidth),
        (Mode::Celt, Bandwidth::Nb)
    );
    assert!(tocs[..2].iter().all(|t| t.bandwidth == Bandwidth::Fb));

    // OPUS_SET_ENERGY_MASK takes a pointer to 21 values per channel, which
    // libopus keeps and reads in every later encode call. A mask of -2 in
    // every band says that other channels hide most of this one, and the
    // encoder then spends fewer bits.
    let mono = channel(&pcm, 3, 0);
    let bytes_with_mask = |mask: Option<&[f32; 21]>| -> i32 {
        let mut err = 0;
        let enc = CEncoder(unsafe { sys::opus_encoder_create(FS, 1, APP_AUDIO, &mut err) });
        assert_eq!(err, 0);
        let mut out = [0u8; 1500];
        let mut total = 0;
        unsafe {
            assert_eq!(
                sys::opus_encoder_ctl(enc.0, sys::OPUS_SET_BITRATE_REQUEST, 64_000),
                0
            );
            if let Some(mask) = mask {
                assert_eq!(
                    sys::opus_encoder_ctl(enc.0, OPUS_SET_ENERGY_MASK_REQUEST, mask.as_ptr()),
                    0
                );
            }
            for frame in mono.as_chunks::<FRAME>().0 {
                let n = sys::opus_encode_float(
                    enc.0,
                    frame.as_ptr(),
                    FRAME as i32,
                    out.as_mut_ptr(),
                    out.len() as i32,
                );
                assert!(n > 0, "opus_encode_float = {n}");
                total += n;
            }
        }
        total
    };
    let plain = bytes_with_mask(None);
    let masked = bytes_with_mask(Some(&[-2.0; 21]));
    println!("12 frames: {plain} bytes without a mask, {masked} bytes with a mask of -2");
    // Measured with libopus 1.6.1: 2092 and 1932 bytes.
    assert!(
        masked * 100 < plain * 96,
        "{masked} bytes with the mask, {plain} without"
    );
}

#[test]
fn c_constructors_report_errors() {
    // Mapping family 3 needs the projection API.
    assert_eq!(
        CMsEncoder::surround(FS, 4, 3, APP_AUDIO).err(),
        Some(OPUS_UNIMPLEMENTED)
    );
    assert_eq!(
        CMsEncoder::surround(FS, 9, 1, APP_AUDIO).err(),
        Some(OPUS_UNIMPLEMENTED)
    );
    assert_eq!(
        CMsEncoder::surround(FS, 5, 2, APP_AUDIO).err(),
        Some(OPUS_BAD_ARG)
    );
    assert_eq!(
        CMsEncoder::surround(FS, 0, 1, APP_AUDIO).err(),
        Some(OPUS_BAD_ARG)
    );
    assert_eq!(
        CMsEncoder::surround(FS, 256, 255, APP_AUDIO).err(),
        Some(OPUS_BAD_ARG)
    );
    // Stream 1 has no input channel: an encoder rejects that, a decoder does not.
    assert_eq!(
        CMsEncoder::generic(FS, 3, 2, 1, &[0, 1, 0], APP_AUDIO).err(),
        Some(OPUS_BAD_ARG)
    );
    assert!(CMsDecoder::new(FS, 3, 2, 1, &[0, 1, 0]).is_ok());
    assert_eq!(CMsDecoder::new(FS, 1, 1, 0, &[5]).err(), Some(OPUS_BAD_ARG));
    // The generic constructor keeps the layout it was given.
    let enc = CMsEncoder::generic(FS, 3, 2, 1, &[0, 1, 2], APP_AUDIO).unwrap();
    assert_eq!((enc.channels, enc.streams, enc.coupled), (3, 2, 1));
    assert_eq!(enc.mapping, [0, 1, 2]);
    // An output buffer below one byte per stream is an error, not a crash.
    let mut enc = CMsEncoder::surround(FS, 6, 1, APP_AUDIO).unwrap();
    let pcm = tones(FS, 6, 1, FRAME);
    assert_eq!(enc.encode_float(&pcm, FRAME, 3), Err(OPUS_BUFFER_TOO_SMALL));
}

#[test]
fn toc_reader_agrees_with_libopus() {
    for byte in 0..=255u8 {
        let t = toc(byte);
        let data = [byte, 0];
        let bandwidth = unsafe { sys::opus_packet_get_bandwidth(data.as_ptr()) };
        let want_bandwidth = match t.bandwidth {
            Bandwidth::Nb => sys::OPUS_BANDWIDTH_NARROWBAND,
            Bandwidth::Mb => sys::OPUS_BANDWIDTH_MEDIUMBAND,
            Bandwidth::Wb => sys::OPUS_BANDWIDTH_WIDEBAND,
            Bandwidth::Swb => sys::OPUS_BANDWIDTH_SUPERWIDEBAND,
            Bandwidth::Fb => sys::OPUS_BANDWIDTH_FULLBAND,
        };
        assert_eq!(bandwidth, want_bandwidth, "TOC {byte:#04x}");
        let channels = unsafe { sys::opus_packet_get_nb_channels(data.as_ptr()) };
        assert_eq!(channels, if t.stereo { 2 } else { 1 }, "TOC {byte:#04x}");
        let samples = unsafe { sys::opus_packet_get_samples_per_frame(data.as_ptr(), 48_000) };
        assert_eq!(samples as usize, t.frame_48k, "TOC {byte:#04x}");
        assert_eq!(t.code, byte & 3);
        // Bits 3 to 7 are the configuration: 0-11 SILK, 12-15 Hybrid, 16-31 CELT.
        let want_mode = match byte >> 3 {
            0..=11 => Mode::Silk,
            12..=15 => Mode::Hybrid,
            _ => Mode::Celt,
        };
        assert_eq!(t.mode, want_mode, "TOC {byte:#04x}");
    }
    assert_eq!(toc(0xFC).to_string(), "CELT/FB/20ms/stereo/c0");
    assert_eq!(toc(0x80).to_string(), "CELT/NB/2.5ms/mono/c0");
    assert_eq!(toc(0x7B).to_string(), "HYBR/FB/20ms/mono/c3");
    assert_eq!(toc(0x19).to_string(), "SILK/NB/60ms/mono/c1");
}

#[test]
fn tones_have_one_frequency_per_channel() {
    // 5.1: the last channel is the LFE channel and gets the low tone.
    assert_eq!(lfe_channel(6, 1), Some(5));
    assert_eq!(lfe_channel(8, 1), Some(7));
    assert_eq!(lfe_channel(5, 1), None);
    assert_eq!(lfe_channel(6, 255), None);
    assert_eq!(tone_hz(0, 6, 1), 400.0);
    assert_eq!(tone_hz(4, 6, 1), 1000.0);
    assert_eq!(tone_hz(5, 6, 1), 80.0);
    assert_eq!(tone_hz(5, 6, 255), 1150.0);

    for fs in [48_000, 16_000] {
        let n = fs as usize / 2;
        let pcm = tones(fs, 6, 1, n);
        assert_eq!(pcm.len(), n * 6);
        let want = (TONE_AMPLITUDE * TONE_AMPLITUDE / 2.0) as f64;
        for c in 0..6 {
            let x = channel(&pcm, 6, c);
            assert_eq!(x.len(), n);
            for k in 0..6 {
                let power = goertzel_power(&x, tone_hz(k, 6, 1), fs);
                if k == c {
                    assert!(
                        (power / want - 1.0).abs() < 1e-3,
                        "fs {fs} channel {c}: {power}"
                    );
                } else {
                    assert!(
                        power < want * 1e-6,
                        "fs {fs} channel {c}, tone {k}: {power}"
                    );
                }
            }
        }
        let dominant = dominant_tones(&pcm, fs, 6, 1, 0);
        assert!(
            dominant
                .iter()
                .enumerate()
                .all(|(c, &(tone, ratio))| tone == c && ratio > 1e6)
        );
        // Swapping two channels shows up in the measure.
        let mut swapped = pcm.clone();
        for i in 0..n {
            swapped.swap(i * 6 + 1, i * 6 + 4);
        }
        let dominant = dominant_tones(&swapped, fs, 6, 1, 0);
        assert_eq!((dominant[1].0, dominant[4].0), (4, 1));
    }
}

/// Pitch of `x` in Hz, between 90 and 250: the lag at which the signal
/// matches itself best.
fn pitch_hz(x: &[f32], fs: i32) -> f64 {
    let (lo, hi) = (fs as usize / 250, fs as usize / 90);
    let n = x.len() - hi;
    let dot = |lag: usize| -> f64 { (0..n).map(|i| x[i] as f64 * x[i + lag] as f64).sum() };
    let energy = |lag: usize| -> f64 { (0..n).map(|i| (x[i + lag] as f64).powi(2)).sum() };
    let best = (lo..=hi)
        .max_by(|&a, &b| (dot(a) / energy(a).sqrt()).total_cmp(&(dot(b) / energy(b).sqrt())))
        .unwrap();
    fs as f64 / best as f64
}

#[test]
fn voiced_noise_and_silence() {
    let n = 48_000;
    let pcm = voiced(FS, 4, n);
    assert_eq!(pcm.len(), n * 4);
    // The pitch of channel c stays within 15 Hz of 110 + 23 * c.
    for c in 0..8 {
        let centre = 110.0 + 23.0 * c as f64;
        let pitches: Vec<f64> = (0..2000)
            .map(|k| voiced_pitch_hz(c, k as f64 / 1000.0))
            .collect();
        let (lo, hi) = pitches
            .iter()
            .fold((f64::MAX, f64::MIN), |(lo, hi), &p| (lo.min(p), hi.max(p)));
        assert!(
            (lo - (centre - 15.0)).abs() < 0.01 && (hi - (centre + 15.0)).abs() < 0.01,
            "channel {c}: {lo} to {hi}"
        );
    }
    for c in 0..4 {
        let x = channel(&pcm, 4, c);
        let peak = x.iter().fold(0f32, |m, v| m.max(v.abs()));
        let rms = (x.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / n as f64).sqrt();
        assert!(
            peak < 0.5 && rms > 0.02,
            "channel {c}: peak {peak}, rms {rms}"
        );
        // The channel has the pitch that `voiced_pitch_hz` documents. The
        // estimate covers about 60 ms, in which the pitch drifts a little.
        for start in [5_000usize, 20_000, 40_000] {
            let estimate = pitch_hz(&x[start..start + 3_000], FS);
            let want = voiced_pitch_hz(c, (start + 1_500) as f64 / FS as f64);
            assert!(
                (estimate - want).abs() < 4.0,
                "channel {c} at {start}: {estimate:.1} Hz, expected {want:.1}"
            );
        }
    }
    // No two channels are alike.
    let (snr, _) = best_lag_snr(&channel(&pcm, 4, 0), &channel(&pcm, 4, 1), 400, 0);
    assert!(snr < 3.0, "channels 0 and 1 match with {snr:.1} dB");

    assert!(silence(3, 100).iter().all(|&v| v == 0.0));
    assert_eq!(silence(3, 100).len(), 300);

    let a = noise(2, 10_000, 0.5, 7);
    assert_eq!(a, noise(2, 10_000, 0.5, 7));
    assert_ne!(a, noise(2, 10_000, 0.5, 8));
    assert!(a.iter().all(|v| v.abs() <= 0.5));
    let mean = a.iter().map(|&v| v as f64).sum::<f64>() / a.len() as f64;
    let rms = (a.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / a.len() as f64).sqrt();
    // Uniform noise of amplitude 0.5 has the RMS 0.5 / sqrt(3) = 0.289.
    assert!(
        mean.abs() < 0.01 && (rms - 0.2887).abs() < 0.01,
        "mean {mean}, rms {rms}"
    );
    let (snr, _) = best_lag_snr(&channel(&a, 2, 0), &channel(&a, 2, 1), 100, 0);
    assert!(snr < 1.0, "the two noise channels match with {snr:.1} dB");

    assert_eq!(
        to_i16(&[0.0, 0.5, -0.5, 1.0, -1.0, 2.0, 1.5 / 32768.0, 2.5 / 32768.0]),
        [0, 16384, -16384, 32767, -32768, 32767, 2, 2]
    );
}

#[test]
fn best_lag_snr_finds_delay_and_noise_level() {
    let n = 30_000;
    let reference = channel(&noise(1, n, 0.5, 1), 1, 0);
    // The same signal, 37 samples late, with noise 40 dB below it.
    let added = noise(1, n, 0.005, 2);
    let mut late = vec![0f32; n];
    for i in 37..n {
        late[i] = reference[i - 37] + added[i];
    }
    let (snr, lag) = best_lag_snr(&reference, &late, 200, 0);
    assert_eq!(lag, 37);
    assert!((snr - 40.0).abs() < 0.5, "SNR {snr:.2} dB");
    assert!((snr_at_lag(&reference, &late, 37, 0) - snr).abs() < 1e-9);
    assert!(snr_at_lag(&reference, &late, 36, 0) < 1.0);
    // An early signal has a negative lag.
    let (snr, lag) = best_lag_snr(&late, &reference, 200, 0);
    assert_eq!(lag, -37);
    assert!((snr - 40.0).abs() < 0.5, "SNR {snr:.2} dB");
    // Equal signals: lag 0 and the upper limit of the scale.
    assert_eq!(best_lag_snr(&reference, &reference, 200, 0), (200.0, 0));
    // A gain error of 10 % alone gives 20 dB.
    let scaled: Vec<f32> = reference.iter().map(|v| v * 0.9).collect();
    let (snr, lag) = best_lag_snr(&reference, &scaled, 50, 1000);
    assert_eq!(lag, 0);
    assert!((snr - 20.0).abs() < 0.01, "SNR {snr:.3} dB");
    assert_eq!(snr_db(1.0, 0.0), 200.0);
    assert_eq!(snr_db(0.0, 1.0), -200.0);
    assert!((snr_db(100.0, 1.0) - 20.0).abs() < 1e-12);
    // Degenerate input does not panic.
    assert_eq!(best_lag_snr(&[], &[], 10, 0).1, 0);
    assert_eq!(best_lag_snr(&reference[..5], &reference[..5], 10, 100).1, 0);
}
