//! Round trip inside opus1: `MultistreamEncoder` to `MultistreamDecoder`,
//! for 5.1 and 7.1.
//!
//! The decoder is the one in src/multistream/mod.rs; only its `new` and
//! `decode` are used. Its output is checked three ways:
//!
//! - it has the frame size;
//! - every channel carries the tone that went into that channel;
//! - it equals what the C multistream decoder makes of the same packets, so
//!   the encoder writes nothing only one of the two decoders understands.
#![cfg(feature = "std")]

mod ms_support;
mod multistream_enc_common;

use ms_support::*;
use multistream_enc_common::*;
use opus1::Application;
use opus1::multistream::MultistreamEncoder;
use opus1::multistream::{ChannelMappingTable, MultistreamDecoder};

const FS: i32 = 48_000;

/// The opus1 decoder for the layout of `enc`, built from the bytes an Ogg
/// Opus header carries after the mapping family: stream count, coupled
/// stream count, and the mapping.
fn rust_decoder(enc: &MultistreamEncoder) -> MultistreamDecoder {
    let layout = enc.layout();
    let mut tail = vec![layout.streams() as u8, layout.coupled_streams() as u8];
    tail.extend_from_slice(layout.mapping());
    let table = ChannelMappingTable::parse(1, layout.channels() as u8, &tail)
        .expect("channel mapping table");
    MultistreamDecoder::from_mapping_table(enc.sampling_rate(), &table).expect("opus1 multistream decoder")
}

/// Encodes `pcm` and decodes every packet with the opus1 decoder and with
/// the C decoder. Returns both outputs.
fn round_trip(
    enc: &mut MultistreamEncoder,
    pcm: &[f32],
    frame_size: usize,
    label: &str,
) -> (Vec<f32>, Vec<f32>) {
    let channels = enc.layout().channels();
    let mut rust_dec = rust_decoder(enc);
    let mut c_dec = c_decoder(enc);
    let mut packet = vec![0u8; max_packet_bytes(enc.layout().streams())];
    let mut frame_out = vec![0f32; frame_size * channels];
    let (mut rust_out, mut c_out) = (Vec::new(), Vec::new());
    for (f, frame) in pcm.chunks_exact(frame_size * channels).enumerate() {
        let n = enc.encode(frame, frame_size, &mut packet).unwrap();
        let decoded = rust_dec
            .decode(&packet[..n], frame_size, &mut frame_out)
            .unwrap_or_else(|e| panic!("{label}: frame {f}: opus1 decode: {e}"));
        assert_eq!(decoded, frame_size, "{label}: frame {f}: decoded samples");
        rust_out.extend_from_slice(&frame_out);
        c_out.extend(
            c_dec
                .decode_float(Some(&packet[..n]), frame_size, false)
                .unwrap_or_else(|e| panic!("{label}: frame {f}: C decode: {e}")),
        );
    }
    (rust_out, c_out)
}

#[test]
fn surround_round_trip_routes_every_channel() {
    for (channels, bitrate) in [(6usize, 256_000), (8, 320_000)] {
        let case = Case::family("surround", channels, 1);
        for (frame_size, vbr) in [(960usize, true), (960, false), (480, true), (2880, true)] {
            let label = format!("{channels} channels, frame {frame_size}, vbr {vbr}");
            let mut enc = rust_encoder(&case, FS, Application::Audio);
            enc.set_bitrate(rust_bitrate(bitrate));
            enc.set_vbr(vbr);
            let lookahead = enc.lookahead() as isize;
            let pcm = tones(FS, channels, 1, FS as usize);
            let (rust_out, c_out) = round_trip(&mut enc, &pcm, frame_size, &label);

            // Every channel carries its own tone.
            let skip = FS as usize / 5;
            let dominant = dominant_tones(&rust_out, FS, channels, 1, skip);
            for (ch, &(tone, ratio)) in dominant.iter().enumerate() {
                assert_eq!(
                    tone, ch,
                    "{label}: channel {ch} carries the tone of channel {tone}"
                );
                assert!(
                    ratio >= 10.0,
                    "{label}: channel {ch}: tone ratio {ratio:.1}"
                );
            }
            // The signal comes back at the delay of the codec. The LFE
            // channel is left out: its quality has its own test in
            // tests/multistream_enc_quality.rs.
            for ch in 0..channels - 1 {
                let snr = snr_at_lag(
                    &channel(&pcm, channels, ch),
                    &channel(&rust_out, channels, ch),
                    lookahead,
                    skip,
                );
                assert!(
                    snr >= 20.0,
                    "{label}: channel {ch}: SNR {snr:.1} dB against the input"
                );
            }
            // Both decoders make the same of the packets.
            for ch in 0..channels {
                let snr = snr_at_lag(
                    &channel(&c_out, channels, ch),
                    &channel(&rust_out, channels, ch),
                    0,
                    0,
                );
                println!("{label}: channel {ch}: opus1 decoder against C decoder {snr:.1} dB");
                assert!(
                    snr >= 90.0,
                    "{label}: channel {ch}: {snr:.1} dB against the C decoder"
                );
            }
        }
    }
}

#[test]
fn surround_round_trip_of_a_speech_like_signal() {
    for (channels, bitrate) in [(6usize, 256_000), (8, 320_000)] {
        let case = Case::family("surround", channels, 1);
        let label = format!("{channels} channels");
        let mut enc = rust_encoder(&case, FS, Application::Audio);
        enc.set_bitrate(rust_bitrate(bitrate));
        let lookahead = enc.lookahead() as isize;
        let pcm = voiced(FS, channels, 2 * FS as usize);
        let (rust_out, c_out) = round_trip(&mut enc, &pcm, 960, &label);
        let skip = FS as usize / 5;
        for ch in 0..channels {
            let against_c = snr_at_lag(
                &channel(&c_out, channels, ch),
                &channel(&rust_out, channels, ch),
                0,
                0,
            );
            let against_input = snr_at_lag(
                &channel(&pcm, channels, ch),
                &channel(&rust_out, channels, ch),
                lookahead,
                skip,
            );
            println!(
                "{label}: channel {ch}: {against_input:.1} dB against the input, \
                 {against_c:.1} dB against the C decoder"
            );
            assert!(
                against_c >= 90.0,
                "{label}: channel {ch}: {against_c:.1} dB against C"
            );
            // The LFE channel keeps only the lowest part of this signal.
            if ch != channels - 1 {
                assert!(
                    against_input >= 7.0,
                    "{label}: channel {ch}: {against_input:.1} dB"
                );
            }
        }
    }
}
