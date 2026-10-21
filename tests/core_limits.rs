//! Arguments at and beyond the limits: nothing panics, wrong arguments are
//! errors, and what the encoder writes under extreme settings is a packet
//! that C libopus decodes.

mod core_common;
mod k24_common;

use core_common::*;
use k24_common::{chirp, to_stereo};
use opus1::{Application, Bandwidth, Bitrate, ForcedMode, OpusDecoder, OpusEncoder, packet, pcm_soft_clip};

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
        // One byte is enough for a packet without audio, as in libopus.
        assert_eq!(enc.encode(&pcm, 960, &mut out[..1]), Ok(1));
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
                let p = ours.unwrap_or_else(|e| panic!("libopus accepts {:02x?} + {} bytes, opus1: {e}", &data[..4], data.len() - 4));
                assert_eq!(p.sizes[..p.count].iter().map(|&s| s as usize).collect::<Vec<_>>(), sizes);
                assert!(sizes.iter().all(|&s| s <= packet::MAX_FRAME_BYTES));
                accepted += 1;
            }
            None => {
                assert!(ours.is_err(), "libopus rejects {:02x?} + {} bytes, opus1 accepts", &data[..4], data.len() - 4);
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
                    let bitrate = [1, 500, 6000, 510_000, i32::MAX][(m + b + f) % 5];
                    for cbr in [false, true] {
                        for room in [0usize, 1, 2, 3, 9, 1276, 4000] {
                            let frame = rate / 400 * frame_2_5ms;
                            let what = format!("{rate} Hz, {channels} ch, {mode:?}, {bandwidth:?}, {} ms, {bitrate} b/s, CBR {cbr}, {room} bytes", frame_2_5ms as f32 * 2.5);
                            let mut enc = OpusEncoder::new(rate as i32, channels, Application::Audio).unwrap();
                            enc.set_force_mode(mode);
                            enc.set_bandwidth(bandwidth);
                            enc.set_bitrate(Bitrate::BitsPerSecond(bitrate)).unwrap();
                            enc.set_vbr(!cbr);
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
            enc.set_bitrate(Bitrate::BitsPerSecond(bitrate)).unwrap();
            enc.set_vbr(!cbr);
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

/// `opus_packet_pad` on a copy of `data`: the padded packet, or the error
/// code when libopus rejects it.
fn c_pad(data: &[u8], new_len: usize) -> Result<Vec<u8>, i32> {
    let mut out = data.to_vec();
    out.resize(new_len, 0xEE);
    let ret = unsafe { audiopus_sys::opus_packet_pad(out.as_mut_ptr(), data.len() as i32, new_len as i32) };
    if ret == 0 { Ok(out) } else { Err(ret) }
}

/// Padding a packet that is padded already, with random bytes in the old
/// padding. libopus reads the old padding as packet extensions: it keeps
/// the extensions it finds and rejects the packet ("Internal error") when
/// the bytes are not valid extensions. `pad` must do the same, byte for
/// byte.
#[test]
fn pad_of_a_packet_with_padding_content() {
    let mut seed = 77u64;
    let mut next = move || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (seed >> 33) as u32
    };
    let (mut zeros_out, mut kept, mut rejected) = (0, 0, 0);
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
        let result = packet::pad(&mut ours, data.len(), new_len);
        match c_pad(&data, new_len) {
            Ok(expected) => {
                assert_eq!(result, Ok(()), "round {round}");
                assert_eq!(ours, expected, "round {round}: the bytes of libopus");
                let parsed = packet::parse(&ours, false).unwrap();
                assert_eq!(parsed.frame(&ours, 0), &data[3..3 + frame_len], "round {round}: the frame");
                if ours[new_len - parsed.padding_len..].iter().all(|&b| b == 0) {
                    zeros_out += 1;
                } else {
                    kept += 1;
                }
            }
            Err(code) => {
                assert_eq!(code, -3, "round {round}");
                assert_eq!(result, Err("Internal error"), "round {round}");
                rejected += 1;
            }
        }
    }
    println!("new padding of zeros {zeros_out}, extensions kept {kept}, rejected {rejected}");
    assert!(zeros_out >= 1000 && kept >= 100 && rejected >= 100, "{zeros_out} {kept} {rejected}");
}

/// The level of a signal in dB, 0 dB for a full-scale square wave.
fn level_db(x: &[f32]) -> f64 {
    let e = x.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / x.len().max(1) as f64;
    10.0 * e.max(1e-12).log10()
}

/// What two frames of a stream are replaced by.
#[derive(Clone, Copy, Debug)]
enum Insane {
    /// A square wave at the highest frequency whose squares add up to
    /// `sum` over the frame.
    Loud { sum: f64 },
    NotANumber,
    Infinite,
}

/// Float input that is not a sane signal: libopus clears the filtered
/// frame and the filter state when the squares of the frame add up to 1e9
/// or more, or to something that is not a number (opus_encoder.c:
/// 2016-2028). Both coders then code silence.
///
/// For each case the test returns the level of every decoded frame of the
/// opus1 stream and of the libopus stream, both decoded by libopus.
fn insane_frames(application: (i32, Application), mode: Option<(i32, ForcedMode)>, rate: usize, insane: Insane) -> Vec<(f64, f64)> {
    let frame = rate / 50;
    let mut pcm = chirp(rate, frame * 12);
    for i in 4..6 {
        let part = &mut pcm[i * frame..(i + 1) * frame];
        match insane {
            Insane::Loud { sum } => {
                let amplitude = (sum / frame as f64).sqrt() as f32;
                for (k, v) in part.iter_mut().enumerate() {
                    *v = if k % 2 == 0 { amplitude } else { -amplitude };
                }
            }
            Insane::NotANumber => part[frame / 3] = f32::NAN,
            Insane::Infinite => part[frame / 3] = f32::INFINITY,
        }
    }
    let mut c_enc = CEncoder::new(rate, 1, application.0);
    c_enc.bitrate(32000).complexity(5);
    let mut enc = OpusEncoder::new(rate as i32, 1, application.1).unwrap();
    enc.set_bitrate(Bitrate::BitsPerSecond(32000)).unwrap();
    enc.set_complexity(5).unwrap();
    if let Some((c_mode, forced)) = mode {
        c_enc.force_mode(c_mode);
        enc.set_force_mode(Some(forced));
    }
    let (mut dec, mut c_dec) = (CDecoder::new(rate, 1), CDecoder::new(rate, 1));
    let mut buf = vec![0u8; 1500];
    pcm.chunks_exact(frame)
        .enumerate()
        .map(|(i, input)| {
            let c_packet = c_enc.encode(input);
            let n = enc.encode(input, frame, &mut buf).unwrap_or_else(|e| panic!("frame {i}: {e}"));
            assert_eq!(buf[0] & 0xFC, c_packet[0] & 0xFC, "frame {i}: first byte");
            let decoded = dec.decode(Some(&buf[..n]), frame, false).unwrap();
            let c_decoded = c_dec.decode(Some(&c_packet), frame, false).unwrap();
            assert_eq!(enc.final_range(), dec.final_range(), "frame {i}: final range");
            assert!(decoded.iter().all(|v| v.is_finite()), "frame {i}");
            (level_db(&decoded), level_db(&c_decoded))
        })
        .collect()
}

/// The limit is 1e9: a frame just under it is coded as it is, a frame at
/// or over it as silence, and so is a frame with a sample that is not a
/// number or is infinite. Two such frames follow each other; the second
/// one decodes as silence when the frames were cleared.
///
/// The sums 0.995e9 and 1.005e9 lie on either side of the limit, within the
/// 1 % where opus1 adds the squares up a second time, one after the other.
///
/// The cleared frame is a silent CELT frame, the same 3 bytes as libopus's.
/// What it decodes to is the tail of the frame before. With the CELT
/// encoder of the fidelity merge (eecefa7) that tail decoded at -57.5 dB
/// against libopus's -62.0 dB (low delay, 48 kHz, sum 1.005e9), 4.5 dB
/// apart. Since the CELT corrections of wave 4 (merged as 584fea2) it
/// decodes at -62.0 dB, as libopus's; before the fidelity merge (1653bce)
/// it was -63.1 dB.
#[test]
fn float_input_that_is_not_a_sane_signal_is_coded_as_silence_like_libopus() {
    for (name, application, mode, rate) in [
        ("low delay, 48 kHz", (APP_LOWDELAY, Application::RestrictedLowDelay), None, 48000usize),
        ("audio, CELT-only, 48 kHz", (APP_AUDIO, Application::Audio), Some((MODE_CELT, ForcedMode::CeltOnly)), 48000),
        ("audio, SILK-only, 16 kHz", (APP_AUDIO, Application::Audio), Some((MODE_SILK, ForcedMode::SilkOnly)), 16000),
        ("VoIP, SILK-only, 16 kHz", (APP_VOIP, Application::Voip), Some((MODE_SILK, ForcedMode::SilkOnly)), 16000),
        ("audio, Hybrid, 48 kHz", (APP_AUDIO, Application::Audio), Some((MODE_HYBRID, ForcedMode::Hybrid)), 48000),
    ] {
        for (insane, cleared) in [
            (Insane::Loud { sum: 0.9e9 }, false),
            (Insane::Loud { sum: 0.995e9 }, false),
            (Insane::Loud { sum: 1.005e9 }, true),
            (Insane::Loud { sum: 4e9 }, true),
            (Insane::NotANumber, true),
            (Insane::Infinite, true),
        ] {
            let what = format!("{name}, {insane:?}");
            let levels = insane_frames(application, mode, rate, insane);
            // Frames 4 and 5 are the ones replaced.
            let (first, c_first) = levels[4];
            let (second, c_second) = levels[5];
            if cleared {
                // Silence, but for what the frame before leaves behind.
                assert!(c_second < -50.0, "{what}: libopus {c_second:.1} dB");
                assert!(second < -50.0, "{what}: {second:.1} dB, libopus {c_second:.1} dB");
                for (i, (level, c_level)) in levels.iter().enumerate().skip(4).take(4) {
                    assert!((level - c_level).abs() < 4.0, "{what}: frame {i}: {level:.1} dB, libopus {c_level:.1} dB");
                }
            } else {
                // Coded: far louder than a cleared frame, as in libopus.
                assert!(c_first > -9.0, "{what}: libopus {c_first:.1} dB");
                assert!((first - c_first).abs() < 4.0, "{what}: {first:.1} dB, libopus {c_first:.1} dB");
            }
        }
    }
}

/// A frame whose filtered samples have a sum of squares at the limit of
/// libopus's test for insane input (1e9) is cleared exactly when libopus
/// clears it. libopus takes that sum with `celt_inner_prod`, whose vector
/// version adds in another order than one sample after the other; the order
/// decides frames within a few parts in ten million of the limit. For each
/// noise frame the scale is bisected until libopus's decision flips, and
/// opus1 must decide as libopus on both sides of the flip.
#[test]
fn the_limit_of_the_insane_input_test_is_that_of_libopus() {
    fn noise(n: usize, seed: u64, scale: f64) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                (scale * (((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0)) as f32
            })
            .collect()
    }
    // Whether libopus and opus1 clear the frame (a cleared frame is coded
    // as digital silence, in a few bytes).
    fn cleared(channels: usize, frame: usize, pcm: &[f32]) -> (bool, bool) {
        let mut rs = OpusEncoder::new(48_000, channels, Application::RestrictedLowDelay).unwrap();
        rs.set_force_mode(Some(ForcedMode::CeltOnly));
        let mut c = CEncoder::new(48_000, channels, APP_LOWDELAY);
        c.force_mode(MODE_CELT);
        let c_packet = c.encode_max(pcm, 1276);
        let mut out = vec![0u8; 1276];
        let len = rs.encode(pcm, frame, &mut out).unwrap();
        (c_packet.len() < 10, len < 10)
    }
    let (mut flips, mut failures) = (0, Vec::new());
    for channels in [1usize, 2] {
        for frame in [120usize, 480, 960] {
            let n = frame * channels;
            // Uniform noise in [-s, s] has a sum of squares of about n*s^2/3.
            let s0 = (3.0e9f64 / n as f64).sqrt();
            for seed in 0..20u64 {
                let (mut lo, mut hi) = (0.9 * s0, 1.1 * s0);
                if !cleared(channels, frame, &noise(n, seed, hi)).0 || cleared(channels, frame, &noise(n, seed, lo)).0 {
                    continue;
                }
                for _ in 0..60 {
                    let mid = 0.5 * (lo + hi);
                    if cleared(channels, frame, &noise(n, seed, mid)).0 {
                        hi = mid;
                    } else {
                        lo = mid;
                    }
                }
                flips += 1;
                for scale in [lo, hi] {
                    let (c, rs) = cleared(channels, frame, &noise(n, seed, scale));
                    if c != rs {
                        failures.push(format!("{channels} ch, {frame} samples, seed {seed}, scale {scale}: libopus clears {c}, opus1 {rs}"));
                    }
                }
            }
        }
    }
    println!("{flips} flip points");
    for f in failures.iter().take(10) {
        println!("  {f}");
    }
    assert!(flips > 50, "too few flip points: {flips}");
    if cfg!(target_arch = "aarch64") {
        assert!(failures.is_empty(), "{} decisions differ from libopus", failures.len());
    }
}
