//! Every stream of a multistream packet is the packet a single
//! `OpusEncoder` writes for the channels of that stream.
//!
//! The multistream encoder adds three things to its stream encoders: the
//! bitrate of each stream, the settings a surround or ambisonics layout
//! calls for, and the framing of the packet. This test builds one
//! `OpusEncoder` per stream by hand, gives it those settings as libopus
//! defines them, feeds it the channels of its stream, and compares bytes.
//! It runs through changes of the bitrate and of the frame size and through
//! a reset, so it also shows that the multistream encoder carries no state
//! of its own from frame to frame that reaches the packets, other than that
//! of the surround analysis.
//!
//! In surround mode every stream gets an energy mask
//! (opus_multistream_encoder.c:989-1014). The test computes it with an
//! analysis of its own, a `SurroundProbe` that sees the same frames and the
//! same reset, and hands each hand-made encoder the mask of the channels of
//! its stream: left then right for a coupled stream. It does not take the
//! mask from the multistream encoder. So the bytes also show that the
//! analysis runs on every frame with the frame's size, that its result
//! reaches the right streams, and that a reset clears its state.
//!
//! tests/multistream_enc_quality.rs does the same for the streams of the
//! 7.1 layout at the bitrate of the quality measurements.
#![cfg(feature = "std")]

mod ms_support;
mod multistream_enc_common;

use ms_support::*;
use multistream_enc_common::*;
use opus1::multistream::encoder::MultistreamEncoder;
use opus1::multistream::layout::MappingType;
use opus1::multistream::surround::SurroundProbe;
use opus1::{Application, Bandwidth, ForcedMode, OpusEncoder};

/// A rising sine with a little noise, different on every channel.
fn sweeps(fs: i32, channels: usize, samples: usize) -> Vec<f32> {
    let mut state = 4242u32;
    let mut pcm = vec![0f32; samples * channels];
    let mut phase = vec![0f64; channels];
    for (i, frame) in pcm.chunks_exact_mut(channels).enumerate() {
        let t = i as f64 / fs as f64;
        for (c, s) in frame.iter_mut().enumerate() {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            let noise = (state >> 8) as f64 / (1u32 << 24) as f64 - 0.5;
            let hz = (120.0 + 70.0 * c as f64) * (1.0 + 2.0 * t);
            phase[c] += 2.0 * std::f64::consts::PI * hz.min(0.4 * fs as f64) / fs as f64;
            *s = (0.25 * phase[c].sin() + 0.02 * noise) as f32;
        }
    }
    pcm
}

/// The audio bandwidth libopus gives all streams in surround mode
/// (opus_multistream_encoder.c:941-952), from the total bitrate.
fn surround_bandwidth(bitrate: i32, channels: usize, frame_size: usize, fs: i32) -> Bandwidth {
    let channels = channels as i32;
    let mut equiv_rate = bitrate;
    if frame_size as i32 * 50 < fs {
        equiv_rate -= 60 * (fs / frame_size as i32 - 50) * channels;
    }
    if equiv_rate > 10_000 * channels {
        Bandwidth::Fullband
    } else if equiv_rate > 7_000 * channels {
        Bandwidth::Superwideband
    } else if equiv_rate > 5_000 * channels {
        Bandwidth::Wideband
    } else {
        Bandwidth::Narrowband
    }
}

/// The input channels of every stream, read off the mapping: left and right
/// of a stereo stream, or the channel of a mono stream. When two channels
/// name the same coded channel, the first one is read.
fn stream_channels(enc: &MultistreamEncoder) -> Vec<(usize, Option<usize>)> {
    let layout = enc.layout();
    let coupled = layout.coupled_streams();
    let find = |coded: usize| {
        layout
            .mapping()
            .iter()
            .position(|&m| m as usize == coded)
            .expect("every coded channel has an input")
    };
    (0..layout.streams())
        .map(|s| {
            if s < coupled {
                (find(2 * s), Some(find(2 * s + 1)))
            } else {
                (find(s + coupled), None)
            }
        })
        .collect()
}

/// The most bytes libopus lets stream `s` write
/// (opus_multistream_encoder.c:1016-1024). `limit` is the size limit of the
/// whole packet and `used` what the streams before `s` take.
fn stream_limit(limit: usize, used: usize, streams: usize, s: usize) -> usize {
    let after = (streams - s - 1) as i64;
    // One byte is kept for the last stream and two for each other one.
    let mut curr_max = limit as i64 - used as i64 - (2 * after - 1).max(0);
    curr_max = curr_max.min(6 * 1275 + 12);
    // The length field of every stream but the last.
    if after != 0 {
        curr_max -= if curr_max > 253 { 2 } else { 1 };
    }
    curr_max as usize
}

/// The energy mask of a stream: the 21 values of its left or only channel,
/// then those of its right channel (opus_multistream_encoder.c:989-1008).
/// `band_smr` holds 21 values per input channel.
fn stream_mask(band_smr: &[f32], left: usize, right: Option<usize>) -> Vec<f32> {
    let mut mask = band_smr[21 * left..21 * left + 21].to_vec();
    if let Some(right) = right {
        mask.extend_from_slice(&band_smr[21 * right..21 * right + 21]);
    }
    mask
}

/// The frames of an ordinary Opus packet, without header and padding.
fn frames_of(part: &StreamPacket) -> &[u8] {
    let end = part.packet.len() - part.padding;
    &part.packet[end - part.payload_len..end]
}

/// One multistream encoder and one hand-made encoder per stream, through a
/// run of frames with changing settings.
fn compare(case: &Case, fs: i32, application: Application, vbr: bool) -> usize {
    let channels = case.channels;
    let mut enc = rust_encoder(case, fs, application);
    enc.set_vbr(vbr);
    let streams = enc.layout().streams();
    let inputs = stream_channels(&enc);
    let mut singles: Vec<OpusEncoder> = inputs
        .iter()
        .enumerate()
        .map(|(s, (_, right))| {
            let stream_channels = if right.is_some() { 2 } else { 1 };
            let mut single = OpusEncoder::new(fs, stream_channels, application).unwrap();
            // libopus flags the LFE stream when the encoder is made.
            single.set_lfe(Some(s) == enc.lfe_stream());
            single
        })
        .collect();

    // libopus analyses the input for the surround mapping type only
    // (opus_multistream_encoder.c:910).
    let surround = enc.mapping_type() == MappingType::Surround;
    let mut probe = surround.then(|| SurroundProbe::new(channels, fs));
    let mut band_smr = vec![0f32; 21 * channels];

    let unit = fs as usize / 400;
    // (bitrate per channel, frame size in units of 2.5 ms, frames, reset
    // before the first of them).
    let steps: [(i32, usize, usize, bool); 6] = [
        (48_000, 8, 5, false),
        (6_500, 8, 5, false),
        (20_000, 4, 4, false),
        (64_000, 24, 2, false),
        (32_000, 8, 4, true),
        (9_000, 1, 6, false),
    ];
    let pcm = sweeps(fs, channels, 40 * 24 * unit);
    let mut out = vec![0u8; max_packet_bytes(streams)];
    let mut single_out = vec![0u8; 6 * 1275 + 12];
    let mut stream_pcm = vec![0f32; 2 * 24 * unit];
    let mut pos = 0;
    let mut compared = 0;
    for (k, (per_channel, units, frames, reset)) in steps.into_iter().enumerate() {
        let frame_size = units * unit;
        let bitrate = per_channel * channels as i32;
        enc.set_bitrate(rust_bitrate(bitrate)).unwrap();
        if reset {
            enc.reset_state();
            if let Some(probe) = probe.as_mut() {
                probe.reset();
            }
            for single in &mut singles {
                single.reset_state();
            }
        }
        for f in 0..frames {
            let label = format!(
                "{} {fs} Hz {application:?} vbr {vbr}: step {k}, frame {f}",
                case.name
            );
            let frame = &pcm[pos..pos + frame_size * channels];
            pos += frame_size * channels;
            let n = enc
                .encode(frame, frame_size, &mut out)
                .unwrap_or_else(|e| panic!("{label}: {e}"));
            let parts = split_streams(&out[..n], streams).unwrap();
            if let Some(probe) = probe.as_mut() {
                assert!(
                    probe.run(frame, frame_size, &mut band_smr),
                    "{label}: the analysis gave no mask"
                );
            }
            // The size limit of the packet: the buffer, and under constant
            // bitrate the bytes of the bitrate (never below the smallest
            // packet).
            let limit = if vbr {
                out.len()
            } else {
                let bits = bitrate as usize * frame_size / fs as usize;
                ((bits + 4) / 8).max(2 * streams - 1)
            };
            if !vbr {
                assert_eq!(n, limit, "{label}: packet size");
            }
            let mut used = 0;
            for (s, single) in singles.iter_mut().enumerate() {
                let stream = enc.stream_encoder(s).unwrap();
                // The bitrate of the allocation, or under constant bitrate
                // what was left for the last stream.
                single.bitrate_bps = stream.bitrate_bps;
                single.use_cbr = !vbr;
                let (left, right) = inputs[s];
                // What a surround or ambisonics layout sets on a stream
                // (opus_multistream_encoder.c:939-962).
                match enc.mapping_type() {
                    MappingType::Surround => {
                        single.set_bandwidth(surround_bandwidth(bitrate, channels, frame_size, fs));
                        if s < enc.layout().coupled_streams() {
                            // CELT-only and two coded channels
                            // (opus_multistream_encoder.c:953-958).
                            single.set_force_mode(Some(ForcedMode::CeltOnly));
                            single.set_force_channels(Some(2)).unwrap();
                        }
                        // Every stream, the LFE stream too, gets the mask
                        // of its own channels.
                        let mask = stream_mask(&band_smr, left, right);
                        assert_eq!(
                            stream.energy_mask(),
                            Some(&mask[..]),
                            "{label}: stream {s}: energy mask"
                        );
                        single.set_energy_mask(Some(&mask));
                    }
                    MappingType::Ambisonics => single.set_force_mode(Some(ForcedMode::CeltOnly)),
                    MappingType::None => {}
                }
                if !surround {
                    assert_eq!(
                        stream.energy_mask(),
                        None,
                        "{label}: stream {s}: a mask outside surround mode"
                    );
                }
                // The channels of the stream.
                let width = if right.is_some() { 2 } else { 1 };
                for (i, x) in frame.chunks_exact(channels).enumerate() {
                    stream_pcm[i * width] = x[left];
                    if let Some(right) = right {
                        stream_pcm[i * width + 1] = x[right];
                    }
                }
                // The stream may use what the streams before it left,
                // less the room kept for those after it.
                let room = stream_limit(limit, used, streams, s).clamp(1, single_out.len());
                used += parts[s].used;
                let m = single
                    .encode(
                        &stream_pcm[..frame_size * width],
                        frame_size,
                        &mut single_out[..room],
                    )
                    .unwrap_or_else(|e| panic!("{label}: stream {s} alone: {e}"));
                let alone = parse_stream(&single_out[..m]).unwrap();
                let part = &parts[s];
                // The same configuration, frame sizes and frame bytes. The
                // frame count code and the padding belong to the framing:
                // the multistream encoder writes the frames anew.
                assert_eq!(
                    part.toc & 0xFC,
                    alone.toc & 0xFC,
                    "{label}: stream {s}: TOC"
                );
                assert_eq!(
                    part.frame_sizes, alone.frame_sizes,
                    "{label}: stream {s}: frame sizes"
                );
                assert_eq!(
                    frames_of(part),
                    frames_of(&alone),
                    "{label}: stream {s}: frames"
                );
                assert_eq!(
                    stream.final_range(),
                    single.final_range(),
                    "{label}: stream {s}: final range"
                );
                // Only the last stream of a constant-bitrate packet is
                // padded.
                if vbr || s + 1 != streams {
                    assert_eq!(part.padding, 0, "{label}: stream {s}: padding");
                }
                compared += 1;
            }
        }
    }
    compared
}

#[test]
fn every_stream_is_what_its_own_encoder_writes_vbr() {
    let mut compared = 0;
    for case in CASES {
        compared += compare(case, 48_000, Application::Audio, true);
        compared += compare(case, 16_000, Application::Voip, true);
    }
    println!("{compared} stream packets compared");
    assert!(compared > 1500);
}

#[test]
fn every_stream_is_what_its_own_encoder_writes_cbr() {
    let mut compared = 0;
    for case in CASES {
        compared += compare(case, 48_000, Application::Audio, false);
        compared += compare(case, 24_000, Application::RestrictedLowDelay, false);
    }
    println!("{compared} stream packets compared");
    assert!(compared > 1500);
}
