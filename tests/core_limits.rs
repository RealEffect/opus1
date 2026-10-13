//! Arguments at and beyond the limits: nothing panics, wrong arguments are
//! errors, and what the encoder writes under extreme settings is a packet
//! that C libopus decodes.

mod core_common;
mod k24_common;

use core_common::*;
use k24_common::{chirp, to_stereo};
use opus_rs::{Application, Bandwidth, ForcedMode, OpusDecoder, OpusEncoder, packet, pcm_soft_clip};

/// Frame sizes no buffer can hold. Several are chosen so that
/// `frame_size * 2` overflows to a small number, or so that their lowest 32
/// bits are a valid frame size at 48 kHz.
const HUGE: [usize; 9] = [
    usize::MAX,
    usize::MAX / 2 + 1,
    (1 << 63) + 112,
    (1 << 63) + 120,
    (1 << 63) + 960,
    (1 << 62) + 960,
    1 << 32,
    (1 << 32) + 960,
    i32::MAX as usize + 1,
];

/// A lost packet of a duration that overflows is an error. A packet with
/// more room than it needs decodes as usual: `frame_size` is only an upper
/// limit then.
#[test]
fn decoder_frame_sizes_beyond_any_buffer_are_errors() {
    for channels in [1usize, 2] {
        let pcm = chirp(48000, 48000 / 2);
        let pcm = if channels == 2 { to_stereo(&pcm, 48000) } else { pcm };
        let mut enc = CEncoder::new(48000, channels, APP_VOIP);
        enc.force_mode(MODE_SILK).bandwidth(BW_WB).bitrate(32000).fec(true, 20);
        let packets = enc.encode_all(&pcm, 960);
        let mut dec = OpusDecoder::new(48000, channels).unwrap();
        let mut out = vec![0f32; 5760 * 2];
        let mut out16 = vec![0i16; 5760 * 2];
        for (i, &frame_size) in HUGE.iter().enumerate() {
            let what = format!("{channels} ch, frame_size {frame_size:#x}");
            assert!(dec.decode(&[], frame_size, &mut out).is_err(), "{what}");
            assert!(dec.decode_fec(&packets[i + 1], frame_size, &mut out).is_err(), "{what}");
            assert!(dec.decode_i16(&[], frame_size, &mut out16).is_err(), "{what}");
            assert!(dec.decode_fec_i16(&packets[i + 1], frame_size, &mut out16).is_err(), "{what}");
            assert_eq!(dec.decode(&packets[i], frame_size, &mut out), Ok(960), "{what}");
            assert_eq!(dec.decode_i16(&packets[i + 1], frame_size, &mut out16), Ok(960), "{what}");
        }
        // Nothing to write to, and nothing asked for.
        assert!(dec.decode(&packets[0], 960, &mut []).is_err());
        assert!(dec.decode(&packets[0], 0, &mut out).is_err());
        assert!(dec.decode(&[], 0, &mut out).is_err());
        assert!(dec.decode(&[], 960, &mut []).is_err());
        assert!(dec.decode_fec(&packets[0], 0, &mut out).is_err());
        assert!(dec.decode_fec(&packets[0], 960, &mut []).is_err());
        assert!(dec.decode_i16(&packets[0], 960, &mut []).is_err());
        assert!(dec.decode_i16(&[], 0, &mut out16).is_err());
        assert!(dec.decode_fec_i16(&packets[0], 960, &mut []).is_err());
        // The decoder still works.
        assert_eq!(dec.decode(&packets[12], 960, &mut out), Ok(960));
    }
}

/// The encoder rejects the same frame sizes, for float and 16-bit input.
#[test]
fn encoder_frame_sizes_beyond_any_buffer_are_errors() {
    for channels in [1usize, 2] {
        let mut enc = OpusEncoder::new(48000, channels, Application::Audio).unwrap();
        let pcm = vec![0.1f32; 5760 * 2];
        let pcm16 = vec![1000i16; 5760 * 2];
        let mut out = vec![0u8; 4000];
        for frame_size in HUGE {
            assert!(enc.encode(&pcm, frame_size, &mut out).is_err(), "{channels} ch, {frame_size:#x}");
            assert!(enc.encode_i16(&pcm16, frame_size, &mut out).is_err(), "{channels} ch, {frame_size:#x}");
            assert_eq!(enc.final_range(), 0);
        }
        assert!(enc.encode(&pcm, 0, &mut out).is_err());
        assert!(enc.encode(&[], 960, &mut out).is_err());
        assert!(enc.encode(&pcm, 960, &mut []).is_err());
        assert!(enc.encode(&pcm, 960, &mut out[..1]).is_err());
        assert!(enc.encode(&pcm, 960, &mut out).is_ok());
    }
}

/// `pcm_soft_clip` and the packet functions with counts and lengths that do
/// not fit: nothing is changed, nothing panics.
#[test]
fn helper_functions_take_counts_that_do_not_fit() {
    let original = [1.5f32, -1.5, 1.5, -1.5];
    for (frame_size, channels) in [(1usize << 63, 2usize), (usize::MAX, 2), (usize::MAX / 2 + 1, 2), (2, usize::MAX), (usize::MAX, usize::MAX), (3, 2)] {
        let mut pcm = original;
        let mut mem = [0f32; 2];
        pcm_soft_clip(&mut pcm, frame_size, channels, &mut mem);
        assert_eq!(pcm, original, "frame_size {frame_size:#x}, channels {channels:#x}");
    }

    let mut data = [0x4Cu8, 1, 2, 3, 0, 0, 0, 0];
    for new_len in [usize::MAX, 9, 3, 0] {
        assert!(packet::pad(&mut data, 4, new_len).is_err(), "pad to {new_len:#x}");
        assert!(packet::multistream_pad(&mut data, 4, new_len, 1).is_err(), "multistream_pad to {new_len:#x}");
    }
    assert!(packet::pad(&mut data, usize::MAX, usize::MAX).is_err());
    assert!(packet::pad(&mut data, 9, 9).is_err());
    assert!(packet::multistream_pad(&mut data, 4, 8, usize::MAX).is_err());
    assert!(packet::multistream_validate(&data[..4], usize::MAX, 48000).is_err());
    assert!(packet::multistream_validate(&data[..4], 0, 48000).is_err());
    assert_eq!(data, [0x4C, 1, 2, 3, 0, 0, 0, 0], "a rejected call leaves the packet alone");
    // Sampling rates that are none: no duration, and no panic.
    for rate in [0, -1, i32::MIN, i32::MAX] {
        let _ = packet::samples_per_frame(0xFF, rate);
        let _ = packet::nb_samples(&data[..4], rate);
        let _ = packet::multistream_validate(&data[..4], 1, rate);
    }
    assert!(packet::nb_samples(&data[..4], -48000).is_err());
}

/// `opus_packet_parse` on a copy of `data`: the frame sizes, or `None` when
/// libopus rejects the packet.
fn c_parse_sizes(data: &[u8]) -> Option<Vec<usize>> {
    let mut sizes = [0i16; 48];
    let count = unsafe {
        audiopus_sys::opus_packet_parse(
            data.as_ptr(),
            data.len() as i32,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            sizes.as_mut_ptr(),
            std::ptr::null_mut(),
        )
    };
    (count >= 0).then(|| sizes[..count as usize].iter().map(|&s| s as usize).collect())
}

/// A frame holds 1275 bytes at most. Where the size of the last frame is
/// not coded, a long packet can imply a longer one: libopus rejects it, and
/// so does `packet::parse`. Every frame-count code, at the limit and one
/// byte beyond.
#[test]
fn frames_longer_than_1275_bytes_are_rejected_like_libopus() {
    let filled = |header: &[u8], payload: usize| {
        let mut p = header.to_vec();
        p.extend((0..payload).map(|i| (i * 7 + 3) as u8));
        p
    };
    // (header, payload bytes): the last frame gets what the header leaves.
    let mut cases = Vec::new();
    for extra in [0usize, 1, 2, 300] {
        // Code 0: one frame.
        cases.push(filled(&[0x48], 1275 + extra));
        // Code 1: two frames of one size.
        cases.push(filled(&[0x49], 2 * (1275 + extra)));
        // Code 2: a first frame of 10 bytes, the second takes the rest.
        cases.push(filled(&[0x4A, 10], 10 + 1275 + extra));
        // Code 2: a first frame of 1275 bytes (255 + 4 * 255), coded in
        // full, and a second one of 5 bytes or a little more. Always valid.
        cases.push(filled(&[0x4A, 255, 255], 1275 + 5 + extra));
        // Code 3, CBR, 3 frames.
        cases.push(filled(&[0x4B, 3], 3 * (1275 + extra)));
        // Code 3, VBR, 3 frames: 4 and 6 bytes, the third takes the rest.
        cases.push(filled(&[0x4B, 0x83, 4, 6], 10 + 1275 + extra));
        // Code 3, CBR, 2 frames, with 5 bytes of padding.
        cases.push(filled(&[0x4B, 0x42, 5], 2 * (1275 + extra) + 5));
    }
    let (mut accepted, mut rejected) = (0, 0);
    for data in &cases {
        let ours = packet::parse(data, false);
        match c_parse_sizes(data) {
            Some(sizes) => {
                let p = ours.unwrap_or_else(|e| panic!("libopus accepts {:02x?} + {} bytes, opus-rs: {e}", &data[..4], data.len() - 4));
                assert_eq!(p.sizes[..p.count].iter().map(|&s| s as usize).collect::<Vec<_>>(), sizes);
                assert!(sizes.iter().all(|&s| s <= packet::MAX_FRAME_BYTES));
                accepted += 1;
            }
            None => {
                assert!(ours.is_err(), "libopus rejects {:02x?} + {} bytes, opus-rs accepts", &data[..4], data.len() - 4);
                rejected += 1;
            }
        }
        // The decoder follows: an error, or the packet's duration.
        let mut dec = OpusDecoder::new(16000, 1).unwrap();
        let mut out = vec![0f32; 1920];
        assert_eq!(dec.decode(data, 1920, &mut out).is_ok(), ours.is_ok());
    }
    // The seven layouts at the limit, and the one whose long frame has a
    // coded size at the three longer lengths.
    assert_eq!((accepted, rejected), (10, 18));
    // `write` takes no frame beyond the limit either.
    let long = vec![1u8; packet::MAX_FRAME_BYTES + 1];
    let mut out = vec![0u8; 4000];
    assert!(packet::write(0x48, &[&long], &mut out, false, false).is_err());
    assert_eq!(packet::write(0x48, &[&long[..1275]], &mut out, false, false), Ok(1276));
}

/// A forced mode and bandwidth with bitrates from the lowest to the highest
/// possible, VBR and CBR, into output buffers from 0 bytes up: every call
/// returns a packet or an error. A packet fits the buffer, the C decoder
/// decodes it to the frame duration, and its final range is the encoder's.
///
/// One kind of packet is decoded but its final range is not compared:
/// stereo Hybrid, see `final_range_equals_the_decoders_for_stereo_hybrid`
/// in `tests/core_encoder.rs`.
#[test]
fn encoder_at_extreme_settings_writes_packets_libopus_decodes() {
    let modes = [None, Some(ForcedMode::SilkOnly), Some(ForcedMode::Hybrid), Some(ForcedMode::CeltOnly)];
    let bandwidths = [Bandwidth::Auto, Bandwidth::Narrowband, Bandwidth::Mediumband, Bandwidth::Superwideband, Bandwidth::Fullband];
    let (mut packets, mut errors) = (0usize, 0usize);
    for (rate, channels) in [(8000usize, 2usize), (12000, 1), (24000, 1), (48000, 2)] {
        let mono = chirp(rate, rate / 4);
        let pcm = if channels == 2 { to_stereo(&mono, rate) } else { mono };
        for (m, mode) in modes.into_iter().enumerate() {
            for (b, bandwidth) in bandwidths.into_iter().enumerate() {
                for (f, frame_2_5ms) in [1usize, 4, 8, 24].into_iter().enumerate() {
                    // Each combination takes one of the bitrates in turn.
                    let bitrate = [i32::MIN, 500, 6000, 510_000, i32::MAX][(m + b + f) % 5];
                    for cbr in [false, true] {
                        for room in [0usize, 1, 2, 3, 9, 1276, 4000] {
                            let frame = rate / 400 * frame_2_5ms;
                            let what = format!("{rate} Hz, {channels} ch, {mode:?}, {bandwidth:?}, {} ms, {bitrate} b/s, CBR {cbr}, {room} bytes", frame_2_5ms as f32 * 2.5);
                            let mut enc = OpusEncoder::new(rate as i32, channels, Application::Audio).unwrap();
                            enc.set_force_mode(mode);
                            enc.set_bandwidth(bandwidth);
                            enc.bitrate_bps = bitrate;
                            enc.use_cbr = cbr;
                            let mut c_dec = CDecoder::new(rate, channels);
                            let mut buf = vec![0u8; room];
                            for input in pcm.chunks_exact(frame * channels).take(3) {
                                let Ok(n) = enc.encode(input, frame, &mut buf) else {
                                    assert_eq!(enc.final_range(), 0, "{what}: final range after an error");
                                    errors += 1;
                                    continue;
                                };
                                assert!(n >= 1 && n <= room, "{what}: {n} bytes");
                                let decoded = c_dec.decode(Some(&buf[..n]), frame, false).unwrap_or_else(|e| panic!("{what}: libopus error {e}"));
                                assert_eq!(decoded.len(), frame * channels, "{what}: duration");
                                let stereo_hybrid = channels == 2 && buf[0] & 0xE0 == 0x60;
                                if !stereo_hybrid {
                                    assert_eq!(enc.final_range(), c_dec.final_range(), "{what}: final range");
                                }
                                packets += 1;
                            }
                        }
                    }
                }
            }
        }
    }
    println!("{packets} packets, {errors} errors");
    assert!(packets > 5000 && errors > 1000, "{packets} packets, {errors} errors");
}

/// CELT-only narrowband stereo at 10 ms with far more bits than the 13 bands
/// can use (128 kb/s and up), on a signal that starts abruptly in one
/// channel. The encoder's final range is the one the two decoders reach.
///
/// It used to differ for the first packet: the decoders read more of the
/// largest pulse vectors (2 samples, 128 pulses) near the end of the frame
/// than the encoder had written.
#[test]
fn large_celt_narrowband_stereo_packets_have_the_decoders_final_range() {
    let (rate, frame) = (8000usize, 80usize);
    let pcm = to_stereo(&chirp(rate, rate / 2), rate);
    for bitrate in [96000, 128000, 256000, 510000] {
        for cbr in [false, true] {
            let mut enc = OpusEncoder::new(rate as i32, 2, Application::Audio).unwrap();
            enc.set_force_mode(Some(ForcedMode::CeltOnly));
            enc.bitrate_bps = bitrate;
            enc.use_cbr = cbr;
            let mut c_dec = CDecoder::new(rate, 2);
            let mut dec = OpusDecoder::new(rate as i32, 2).unwrap();
            let mut buf = vec![0u8; 1276];
            let mut out = vec![0f32; frame * 2];
            for (i, input) in pcm.chunks_exact(frame * 2).enumerate().take(20) {
                let n = enc.encode(input, frame, &mut buf).unwrap();
                c_dec.decode(Some(&buf[..n]), frame, false).unwrap();
                dec.decode(&buf[..n], frame, &mut out).unwrap();
                assert_eq!(c_dec.final_range(), dec.final_range(), "{bitrate} b/s, CBR {cbr}, packet {i}: the decoders agree");
                assert_eq!(enc.final_range(), c_dec.final_range(), "{bitrate} b/s, CBR {cbr}, packet {i} ({n} bytes)");
            }
        }
    }
}

/// `opus_packet_pad` on a copy of `data`: the padded packet, or `None` when
/// libopus rejects it.
fn c_pad(data: &[u8], new_len: usize) -> Option<Vec<u8>> {
    let mut out = data.to_vec();
    out.resize(new_len, 0xEE);
    let ret = unsafe { audiopus_sys::opus_packet_pad(out.as_mut_ptr(), data.len() as i32, new_len as i32) };
    (ret == 0).then_some(out)
}

/// Padding a packet that is padded already, with random bytes in the old
/// padding. This measures the one known difference from `opus_packet_pad`
/// (packet extensions are not ported) and holds it to its limits:
///
/// - old padding of zeros: the bytes are libopus's;
/// - any old padding: `pad` succeeds, the frames are unchanged, and the new
///   padding is zeros. libopus keeps old padding it can read as extensions
///   and rejects the packet otherwise.
#[test]
fn pad_of_a_packet_with_padding_content() {
    let mut seed = 77u64;
    let mut next = move || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (seed >> 33) as u32
    };
    let (mut same, mut c_keeps_content, mut c_rejects) = (0, 0, 0);
    for round in 0..3000 {
        let frame_len = 1 + next() as usize % 20;
        let pad_len = 1 + next() as usize % 12;
        let zeros = round % 3 == 0;
        // Code 3, one frame, padding flag set, `pad_len` bytes of padding.
        let mut data = vec![0x4B, 0x41, pad_len as u8];
        data.extend((0..frame_len).map(|_| next() as u8));
        data.extend((0..pad_len).map(|_| if zeros || next() % 4 == 0 { 0 } else { next() as u8 }));
        let new_len = data.len() + 1 + next() as usize % 20;

        let mut ours = data.clone();
        ours.resize(new_len, 0xEE);
        packet::pad(&mut ours, data.len(), new_len).unwrap_or_else(|e| panic!("round {round}: {e}"));
        let parsed = packet::parse(&ours, false).unwrap();
        assert_eq!(parsed.frame(&ours, 0), &data[3..3 + frame_len], "round {round}: the frame");
        assert_eq!(parsed.padding_len + parsed.payload_offset + frame_len, new_len, "round {round}");
        assert!(ours[new_len - parsed.padding_len..].iter().all(|&b| b == 0), "round {round}: new padding");

        let all_zero = data[3 + frame_len..].iter().all(|&b| b == 0);
        match c_pad(&data, new_len) {
            Some(expected) if expected == ours => same += 1,
            Some(_) => {
                assert!(!all_zero, "round {round}: bytes differ from libopus without padding content");
                c_keeps_content += 1;
            }
            None => {
                assert!(!all_zero, "round {round}: libopus rejects a packet with zero padding");
                c_rejects += 1;
            }
        }
    }
    println!("{same} identical to libopus; libopus keeps the old padding in {c_keeps_content} and rejects {c_rejects}");
    assert!(same >= 1000, "{same} identical");
}
