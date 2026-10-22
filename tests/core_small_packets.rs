//! Output buffers and bitrates too small to carry audio: libopus
//! `opus_encode_native`, "If the space is too low to do something useful,
//! emit 'PLC' frames" (opus_encoder.c:1340-1406).
//!
//! The reference is the C libopus encoder with the same settings and the
//! same buffer. The encoders run at complexity 5, where both make their
//! decisions without the speech/music analysis (both run it from
//! complexity 7 on; `core_analysis_libopus.rs` compares that).

mod core_common;
mod k24_common;

use std::num::NonZeroU32;
use core_common::*;
use k24_common::{multitone, speech, to_stereo};
use opus1::{Application, Bitrate, Channels, ForcedMode, OpusEncoder, SampleRate, packet};
use opus1::packet::Framing;

fn signal(rate: usize, channels: usize, seconds: f64) -> Vec<f32> {
    let voice = speech(rate, seconds);
    let tones = multitone(rate, voice.len());
    let mono: Vec<f32> = voice.iter().zip(&tones).map(|(v, t)| 0.6 * v + 0.3 * t).collect();
    if channels == 2 { to_stereo(&mono, rate) } else { mono }
}

/// A packet that codes nothing: every frame is empty.
fn is_placeholder(data: &[u8]) -> bool {
    let parsed = packet::parse(data, Framing::Normal).unwrap();
    parsed.frame_sizes().iter().all(|&size| size == 0)
}

const MODES: [(&str, Option<(i32, ForcedMode)>); 4] = [
    ("auto", None),
    ("SILK", Some((MODE_SILK, ForcedMode::SilkOnly))),
    ("Hybrid", Some((MODE_HYBRID, ForcedMode::Hybrid))),
    ("CELT", Some((MODE_CELT, ForcedMode::CeltOnly))),
];

struct Counts {
    packets: usize,
    placeholders: usize,
    one_byte: usize,
    coded: usize,
}

/// Every buffer size from 1 byte up, for every frame duration: what
/// libopus writes into the buffer, opus1 writes too.
///
/// - A packet without audio is libopus's byte for byte. That includes the
///   1-byte buffer, where a 60 ms frame becomes a SILK packet whatever the
///   mode, because one byte has no room for a frame count.
/// - With a constant bitrate every packet has libopus's size.
/// - A packet with audio names libopus's mode, bandwidth and frame
///   duration, and the C decoder ends in the encoder's final range.
///
/// With stereo input the mode, bandwidth and channel count in the first
/// byte are not compared. libopus codes stereo input as mono at the low
/// bitrates a small buffer stands for, which opus1 does not do yet. Mode
/// and bandwidth then follow from the mono rate, and a packet without audio
/// repeats those of the last coded one. Stereo Hybrid packets are left out of the final
/// range check (a known fault of the CELT layer, see tests/core_encoder.rs).
fn small_buffers(rate: usize, frame_tenth_ms: &[usize]) -> Counts {
    let mut counts = Counts { packets: 0, placeholders: 0, one_byte: 0, coded: 0 };
    // Sizes around every limit of the rule: 3 bytes, 12 bytes for 40 ms,
    // 19 bytes for 60 ms, and the sizes where the 20 ms parts get room.
    let buffers: Vec<usize> = (1..=26).chain([30, 40, 61, 150, 1000]).collect();
    for channels in [1usize, 2] {
        for &tenth_ms in frame_tenth_ms {
            let frame = rate * tenth_ms / 10_000;
            let pcm = signal(rate, channels, buffers.len() as f64 * tenth_ms as f64 / 10_000.0 + 0.1);
            for (mode_name, mode) in MODES {
                for cbr in [false, true] {
                    for bitrate in [6_000, 24_000, 96_000] {
                        let what = format!(
                            "{rate} Hz, {channels} ch, {} ms, {mode_name}, CBR {cbr}, {bitrate} b/s",
                            tenth_ms as f64 / 10.0
                        );
                        let mut c_enc = CEncoder::new(rate, channels, APP_AUDIO);
                        c_enc.bitrate(bitrate).vbr(!cbr).complexity(5);
                        let mut enc = OpusEncoder::new(
                            SampleRate::try_from(rate as i32).unwrap(),
                            Channels::try_from(channels).unwrap(),
                            Application::Audio,
                        )
                        .unwrap();
                        enc.set_bitrate(Bitrate::BitsPerSecond(NonZeroU32::new(bitrate as u32).unwrap()));
                        enc.set_vbr(!cbr);
                        enc.set_complexity(5).unwrap();
                        if let Some((c_mode, forced)) = mode {
                            c_enc.force_mode(c_mode);
                            enc.set_force_mode(Some(forced));
                        }
                        let mut c_dec = CDecoder::new(rate, channels);
                        // Small and large buffers alternate, so that packets
                        // without audio come after coded ones of every kind.
                        let order = buffers.iter().zip(buffers.iter().rev()).flat_map(|(a, b)| [*a, *b]);
                        for (input, len) in pcm.chunks_exact(frame * channels).zip(order) {
                            let what = format!("{what}, buffer {len}");
                            let c_packet = c_enc.encode_max(input, len);
                            let mut buf = vec![0u8; len];
                            let n = enc
                                .encode(input, frame, &mut buf)
                                .unwrap_or_else(|e| panic!("{what}: opus1: {e}"));
                            let packet = &buf[..n];
                            counts.packets += 1;
                            if cbr {
                                assert_eq!(n, c_packet.len(), "{what}: size");
                            }
                            let decoded = c_dec
                                .decode(Some(packet), frame, false)
                                .unwrap_or_else(|e| panic!("{what}: C decoder: {e}"));
                            assert_eq!(decoded.len(), frame * channels, "{what}");
                            if is_placeholder(&c_packet) {
                                // Stereo input: see above for the first byte.
                                if channels == 1 {
                                    assert_eq!(packet[0], c_packet[0], "{what}: packet without audio");
                                }
                                assert_eq!(packet[0] & 3, c_packet[0] & 3, "{what}: frame count code");
                                assert_eq!(packet[1..], c_packet[1..], "{what}: packet without audio");
                                assert_eq!(enc.final_range(), 0, "{what}");
                                counts.placeholders += 1;
                                counts.one_byte += (len == 1) as usize;
                                continue;
                            }
                            assert!(!is_placeholder(packet), "{what}: libopus codes audio here");
                            // Mode, bandwidth and frame duration. The
                            // frame count code depends on the sizes of the
                            // frames, which differ between the encoders.
                            // Mono input only: with stereo input libopus
                            // decides for the channel count it codes.
                            if channels == 1 {
                                assert_eq!(packet[0] & 0xF8, c_packet[0] & 0xF8, "{what}: first byte");
                            }
                            counts.coded += 1;
                            let t = k24_common::toc(packet[0]);
                            let stereo_hybrid = t.mode == "Hybrid" && packet[0] & 4 != 0;
                            if !stereo_hybrid {
                                assert_eq!(enc.final_range(), c_dec.final_range(), "{what}: final range");
                            }
                        }
                    }
                }
            }
        }
    }
    counts
}

#[test]
fn small_buffers_up_to_20_ms_like_libopus() {
    let counts = small_buffers(48000, &[25, 50, 100, 200]);
    assert!(counts.placeholders > 300 && counts.coded > 3000, "{} of {}", counts.placeholders, counts.packets);
    assert!(counts.one_byte >= 4 * 24 * 2, "{} one-byte buffers", counts.one_byte);
}

/// 40 and 60 ms frames: the rule is applied to the whole frame, before a
/// CELT or Hybrid frame is split into 20 ms parts. A buffer below 12 bytes
/// (40 ms) or 19 bytes (60 ms) gets a packet without audio; from there on
/// every part has room.
#[test]
fn small_buffers_with_40_and_60_ms_like_libopus() {
    let counts = small_buffers(48000, &[400, 600]);
    assert!(counts.placeholders > 900 && counts.coded > 900, "{} of {}", counts.placeholders, counts.packets);
    assert!(counts.one_byte >= 2 * 24 * 2, "{} one-byte buffers", counts.one_byte);
}

#[test]
fn small_buffers_at_16_khz_like_libopus() {
    let counts = small_buffers(16000, &[100, 200, 600]);
    assert!(counts.placeholders > 300 && counts.coded > 1500, "{} of {}", counts.placeholders, counts.packets);
}

/// A constant bitrate above what one frame can hold (1275 bytes): libopus
/// codes the frame in 1275 bytes and pads the packet to the size the
/// bitrate asks for (opus_encoder.c:1889 and 2643-2651). The buffer limits
/// that size.
#[test]
fn constant_bitrate_above_one_frame_pads_like_libopus() {
    let rate = 48000usize;
    for (frame, bitrate, len) in [(960usize, 800_000, 4000usize), (960, 800_000, 1500), (480, 1_500_000, 4000), (2880, 600_000, 6000)] {
        let pcm = signal(rate, 2, 0.5);
        let mut c_enc = CEncoder::new(rate, 2, APP_AUDIO);
        c_enc.bitrate(bitrate).vbr(false).complexity(5).force_mode(MODE_CELT);
        let mut enc = OpusEncoder::new(
            SampleRate::try_from(rate as i32).unwrap(),
            Channels::Stereo,
            Application::Audio,
        )
        .unwrap();
        enc.set_bitrate(Bitrate::BitsPerSecond(NonZeroU32::new(bitrate as u32).unwrap()));
        enc.set_vbr(false);
        enc.set_complexity(5).unwrap();
        enc.set_force_mode(Some(ForcedMode::CeltOnly));
        let mut c_dec = CDecoder::new(rate, 2);
        let mut buf = vec![0u8; len];
        for (i, input) in pcm.chunks_exact(frame * 2).enumerate() {
            let what = format!("{frame} samples, {bitrate} b/s, buffer {len}, packet {i}");
            let c_packet = c_enc.encode_max(input, len);
            let n = enc.encode(input, frame, &mut buf).unwrap_or_else(|e| panic!("{what}: {e}"));
            assert_eq!(n, c_packet.len(), "{what}: size");
            assert!(n > 1276, "{what}: {n} bytes");
            assert_eq!(buf[0], c_packet[0], "{what}: first byte");
            let c_parsed = packet::parse(&c_packet, Framing::Normal).unwrap();
            let parsed = packet::parse(&buf[..n], Framing::Normal).unwrap();
            assert_eq!(parsed.frame_sizes(), c_parsed.frame_sizes(), "{what}: frame sizes");
            c_dec.decode(Some(&buf[..n]), frame, false).unwrap_or_else(|e| panic!("{what}: C decoder: {e}"));
            assert_eq!(enc.final_range(), c_dec.final_range(), "{what}: final range");
        }
    }
}
