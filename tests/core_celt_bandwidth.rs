//! A CELT-only stream whose audio bandwidth narrows for one packet and
//! then widens again, as it does when the bitrate or the output buffer
//! drops for a moment.
//!
//! libopus clears the energies of the bands outside the coded range at the
//! end of every frame, in the encoder and in the decoder ("In case start or
//! end were to change", celt_encoder.c and celt_decoder.c). The first frame
//! with the wider bandwidth then predicts the bands that come back from
//! zero on both sides.
//!
//! The CELT encoder of opus1 (src/celt.rs, `CeltEncoder::encode_impl`)
//! clears them too since 1947a49. Before, it predicted the bands that come
//! back from the energies they had before the bandwidth narrowed, and the
//! decoder from zero. From then on the encoder's picture of the decoder
//! differed from libopus's, and so did its decisions and the level of the
//! decoded signal.
//!
//! The reference is the C libopus encoder with the same settings and the
//! same buffers. Both streams are decoded by the C decoder.

mod core_common;
mod k24_common;

use core_common::*;
use k24_common::{multitone, speech};
use opus1::{Application, Bandwidth, OpusEncoder};

fn energy_db(x: &[f32]) -> f64 {
    let e = x.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / x.len().max(1) as f64;
    10.0 * e.max(1e-12).log10()
}

/// A constant-bitrate stream of 2.5 ms frames. Every 16th packet gets a
/// buffer of 1 byte and carries no audio, and the one after it gets `small`
/// bytes: that packet is narrowband, the others are super-wideband.
/// Returns the largest distance in level, in dB, between the opus1
/// stream and the libopus stream over the three packets after each narrow
/// one.
fn level_distance_after_a_narrow_packet(small: usize) -> f64 {
    let rate = 48000usize;
    let frame = rate / 400;
    let voice = speech(rate, 1.0);
    let tones = multitone(rate, voice.len());
    let pcm: Vec<f32> = voice.iter().zip(&tones).map(|(v, t)| 0.7 * v + 0.2 * t).collect();

    let mut c_enc = CEncoder::new(rate, 1, APP_AUDIO);
    c_enc.bitrate(56_000).vbr(false).complexity(0).max_bandwidth(BW_SWB);
    let mut enc = OpusEncoder::new(rate as i32, 1, Application::Audio).unwrap();
    enc.bitrate_bps = 56_000;
    enc.use_cbr = true;
    enc.complexity = 0;
    enc.set_max_bandwidth(Bandwidth::Superwideband).unwrap();
    let (mut dec, mut c_dec) = (CDecoder::new(rate, 1), CDecoder::new(rate, 1));

    let mut worst = 0f64;
    let mut narrow_packets = 0;
    for (i, input) in pcm.chunks_exact(frame).take(160).enumerate() {
        let buffer = match i % 16 {
            8 => 1,
            9 => small,
            _ => 1500,
        };
        let c_packet = c_enc.encode_max(input, buffer);
        let mut buf = vec![0u8; buffer];
        let n = enc.encode(input, frame, &mut buf).unwrap();
        assert_eq!(n, c_packet.len(), "packet {i}: size");
        assert_eq!(buf[0] & 0xFC, c_packet[0] & 0xFC, "packet {i}: first byte");
        // Configuration 16: CELT-only, narrowband, 2.5 ms.
        narrow_packets += (i % 16 == 9 && buf[0] >> 3 == 16) as usize;
        let decoded = dec.decode(Some(&buf[..n]), frame, false).unwrap();
        let c_decoded = c_dec.decode(Some(&c_packet), frame, false).unwrap();
        assert_eq!(enc.final_range(), dec.final_range(), "packet {i}: final range");
        if (10..13).contains(&(i % 16)) {
            worst = worst.max((energy_db(&decoded) - energy_db(&c_decoded)).abs());
        }
    }
    assert_eq!(narrow_packets, 10);
    worst
}

/// After a packet with a narrower bandwidth, the stream decodes at the
/// level of libopus's stream.
///
/// Measured with the fault: up to 35 dB from libopus in the packets after
/// the narrow one. In a longer run with changing settings the same fault
/// gave three packets up to 24 dB above the input, at full scale (24 kHz,
/// 2.5 ms frames, 56 kb/s constant bitrate). With the energies cleared as
/// in libopus (already at the tip of integ6, 4726c16): 0.0, 2.8 and
/// 0.0 dB for narrow packets of 3, 4 and 6 bytes. Since aefbec9, which
/// rounds dc_reject and the tone detection as libopus does on 64-bit ARM:
/// 0.0 dB for all three.
#[test]
fn celt_only_stream_recovers_from_a_narrow_packet_like_libopus() {
    for small in [3usize, 4, 6] {
        let worst = level_distance_after_a_narrow_packet(small);
        if std::env::var("CORE_PRINT").is_ok() {
            println!("a packet of {small} bytes: {worst:.1} dB from libopus");
        }
        assert!(worst < 3.0, "a packet of {small} bytes: {worst:.1} dB from libopus's level afterwards");
    }
}
