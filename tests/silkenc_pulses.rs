//! The excitation of a SILK frame whose length is not a multiple of 16
//! samples: 10 ms at 12 kHz, 120 samples in 8 shell blocks of 16.
//!
//! libopus's `silk_encode_pulses` zeroes the 8 samples after the frame
//! before it codes the last block (silk/encode_pulses.c). opus-rs coded the
//! least significant bits and the signs of whatever an earlier, longer frame
//! had left there. The decoder reads those 8 samples as well and then drops
//! them, so the packet decoded, but the encoder's final range was not the
//! decoder's and the extra bits were wasted.

// `RangeCoder::finish` needs the `std` feature.
#![cfg(feature = "std")]
mod silkenc_common;
use silkenc_common::*;

use opus_rs::range_coder::RangeCoder;
use opus_rs::silk::encode_pulses::silk_encode_pulses;
use opus_rs::{Application, Bandwidth, OpusEncoder};

/// Mediumband 10 ms packets after a longer frame have the final range that
/// the libopus decoder reaches.
///
/// - 12 kHz input: one 40 ms packet (two 240-sample frames), then 10 ms.
/// - 16 kHz input: one 40 ms packet, one 2.5 ms packet (CELT), then 10 ms
///   packets with the bandwidth set to mediumband.
#[test]
fn mediumband_10_ms_after_a_longer_frame_has_the_decoders_final_range() {
    let mut differ = Vec::new();
    let mut checked = 0;
    for (fs, celt_frame) in [(12_000usize, false), (16_000, true)] {
        for application in [Application::Voip, Application::Audio] {
            for bitrate in [12_000, 20_000, 32_000] {
                let mut enc = OpusEncoder::new(fs as i32, 1, application).unwrap();
                enc.bitrate_bps = bitrate;
                enc.complexity = 3;
                let mut dec = CDecoder::new(fs, 1);
                let pcm = speech_like(fs, 1, fs, 7);
                let mut out = vec![0u8; 1500];
                let mut pos = 0;
                let mut sizes = vec![fs / 25];
                if celt_frame {
                    sizes.push(fs / 400);
                }
                sizes.extend([fs / 100; 30]);
                for (f, &n) in sizes.iter().enumerate() {
                    if n == fs / 100 {
                        enc.set_bandwidth(Bandwidth::Mediumband);
                    }
                    let len = enc.encode(&pcm[pos..pos + n], n, &mut out).unwrap();
                    pos += n;
                    dec.decode(Some(&out[..len]), n, false).unwrap();
                    checked += 1;
                    if enc.final_range() != dec.final_range() {
                        differ.push(format!(
                            "{fs} Hz {application:?} {bitrate} b/s: packet {f} ({len} bytes, TOC {:#04x})",
                            out[0]
                        ));
                    }
                }
            }
        }
    }
    assert!(
        differ.is_empty(),
        "{} of {checked} packets: {differ:#?}",
        differ.len()
    );
}

/// The coded excitation of a 120-sample frame does not depend on what lies
/// after the frame in the buffer, and the 8 samples after it are cleared, as
/// libopus clears them.
#[test]
fn samples_after_a_120_sample_frame_are_not_coded() {
    let mut rng = Rng(99);
    for signal_type in 0..3 {
        for quant_offset_type in 0..2 {
            for round in 0..50 {
                let mut clean = [0i8; 320];
                // Large values in some rounds, so that blocks need the
                // least significant bits coded.
                let amp = if round % 3 == 0 { 20 } else { 3 };
                for p in clean.iter_mut().take(120) {
                    *p = ((rng.next() % (2 * amp + 1)) as i32 - amp as i32) as i8;
                }
                let mut stale = clean;
                for p in stale.iter_mut().skip(120) {
                    *p = ((rng.next() % 41) as i32 - 20) as i8;
                }
                let code = |pulses: &mut [i8; 320]| {
                    let mut rc = RangeCoder::new_encoder(1275);
                    silk_encode_pulses(&mut rc, signal_type, quant_offset_type, pulses, 120);
                    let (bits, rng) = (rc.tell(), rc.rng);
                    (bits, rng, rc.finish())
                };
                let want = code(&mut clean);
                let got = code(&mut stale);
                assert_eq!(
                    got, want,
                    "signal type {signal_type}, offset {quant_offset_type}, round {round}"
                );
                assert_eq!(stale[120..136], [0; 16]);
            }
        }
    }
}
