//! Stereo input at low bitrates (opus_encoder.c): the stereo width
//! reduction of `opus_encode_frame_native` (`stereo_fade`, 2321-2349), and
//! the decision of `opus_encode_native` to code stereo input as mono
//! (1429-1451).
//!
//! The reference is the C libopus encoder with the same settings, at
//! complexity 5, where both encoders decide without the speech/music
//! analysis (both run it from complexity 7 on; `core_analysis_libopus.rs`
//! compares that).

mod core_common;
mod k24_common;

use core_common::*;
use k24_common::{speech_fullband, to_stereo};
use opus1::{Application, Bitrate, Channels, ForcedMode, OpusEncoder, SampleRate};

/// Level of the side signal (left minus right) against the mid signal (left
/// plus right), in dB. It goes down as a stereo image gets narrower.
fn side_level_db(pcm: &[f32]) -> f64 {
    let (mut mid, mut side) = (0f64, 0f64);
    for lr in pcm.as_chunks::<2>().0 {
        mid += ((lr[0] + lr[1]) as f64).powi(2);
        side += ((lr[0] - lr[1]) as f64).powi(2);
    }
    10.0 * (side.max(1e-12) / mid.max(1e-12)).log10()
}

fn snr(reference: &[f32], test: &[f32]) -> f64 {
    let (mut sig, mut err) = (0f64, 0f64);
    for (r, t) in reference.iter().zip(test) {
        sig += (*r as f64).powi(2);
        err += (*r as f64 - *t as f64).powi(2);
    }
    10.0 * (sig.max(1e-12) / err.max(1e-12)).log10()
}

struct Decoded {
    /// The decoded signal without the encoder's look-ahead.
    pcm: Vec<f32>,
    /// How many packets say stereo in their first byte.
    stereo_packets: usize,
    packets: usize,
}

/// Encode with opus1 and with libopus, CELT-only, and decode both streams
/// with libopus.
fn celt_streams(rate: usize, bitrate: i32, cbr: bool, input: &[f32]) -> (Decoded, Decoded) {
    let frame = rate / 50;
    let mut c_enc = CEncoder::new(rate, 2, APP_AUDIO);
    c_enc.bitrate(bitrate).complexity(5).vbr(!cbr).force_mode(MODE_CELT);
    let mut enc = OpusEncoder::new(
        SampleRate::try_from(rate as i32).unwrap(),
        Channels::Stereo,
        Application::Audio,
    )
    .unwrap();
    enc.set_bitrate(Bitrate::BitsPerSecond(bitrate)).unwrap();
    enc.set_complexity(5).unwrap();
    enc.set_vbr(!cbr);
    enc.set_force_mode(Some(ForcedMode::CeltOnly));
    let (mut dec, mut c_dec) = (CDecoder::new(rate, 2), CDecoder::new(rate, 2));
    let mut buf = vec![0u8; 4000];
    let (mut out, mut c_out) = (Vec::new(), Vec::new());
    let (mut stereo, mut c_stereo, mut packets) = (0, 0, 0);
    for pcm in input.chunks_exact(2 * frame) {
        let n = enc.encode(pcm, frame, &mut buf).unwrap();
        let c_packet = c_enc.encode(pcm);
        out.extend(dec.decode(Some(&buf[..n]), frame, false).unwrap());
        assert_eq!(enc.final_range(), dec.final_range(), "{rate} Hz, {bitrate} b/s, packet {packets}");
        c_out.extend(c_dec.decode(Some(&c_packet), frame, false).unwrap());
        stereo += (buf[0] & 4 != 0) as usize;
        c_stereo += (c_packet[0] & 4 != 0) as usize;
        packets += 1;
    }
    let lag = 2 * enc.lookahead();
    (
        Decoded { pcm: out[lag..].to_vec(), stereo_packets: stereo, packets },
        Decoded { pcm: c_out[lag..].to_vec(), stereo_packets: c_stereo, packets },
    )
}

/// CELT-only stereo between 18 and 40 kb/s, where libopus still codes two
/// channels but narrows the stereo image: to 40 % of the side signal at
/// 18 kb/s, 87 % at 24 kb/s, and all of it from 34 kb/s on.
///
/// The decoded side signal is as loud as in libopus's stream, within
/// 0.7 dB (measured: 0.1 dB). Without the width reduction it is 8 dB
/// louder at 18 kb/s and 3.4 dB at 20 kb/s. The quality against the input
/// is within 1 dB of libopus's (measured: 0.5 dB).
#[test]
fn celt_stereo_width_follows_the_bitrate_like_libopus() {
    for rate in [48000usize, 24000] {
        let input = to_stereo(&speech_fullband(rate, 3.0), rate);
        let input_side = side_level_db(&input);
        let mut narrowed = 0;
        for bitrate in [18000, 20000, 24000, 28000, 40000] {
            for cbr in [false, true] {
                let what = format!("{rate} Hz, {bitrate} b/s, CBR {cbr}");
                let (rust, c) = celt_streams(rate, bitrate, cbr, &input);
                // At 18 kb/s with a constant bitrate libopus codes mono, and
                // so does opus1 (see the test below).
                assert_eq!(rust.stereo_packets, c.stereo_packets, "{what}: stereo packets");
                if c.stereo_packets == 0 {
                    assert!(cbr && bitrate == 18000, "{what}: libopus codes mono");
                    continue;
                }
                assert_eq!(c.stereo_packets, c.packets, "{what}: libopus changes the channel count");
                let (side, c_side) = (side_level_db(&rust.pcm), side_level_db(&c.pcm));
                let n = rust.pcm.len();
                let (quality, c_quality) = (snr(&input[..n], &rust.pcm), snr(&input[..n], &c.pcm));
                println!("{what}: side {side:.1} dB, libopus {c_side:.1} dB, input {input_side:.1} dB; SNR {quality:.1} dB, libopus {c_quality:.1} dB");
                assert!((side - c_side).abs() <= 0.7, "{what}: side {side:.1} dB, libopus {c_side:.1} dB");
                assert!(quality >= c_quality - 1.0, "{what}: {quality:.1} dB, libopus {c_quality:.1} dB");
                narrowed += (c_side < input_side - 3.0) as usize;
            }
        }
        // The low bitrates do narrow the image in libopus.
        assert!(narrowed >= 3, "{rate} Hz: {narrowed} settings narrowed by 3 dB");
    }
}

/// A bitrate sweep over stereo input: opus1 codes the channel count
/// libopus codes. libopus goes to mono below about 17 kb/s for music and
/// 19 kb/s for speech, with 1 kb/s of hysteresis, and SILK takes two frames
/// to fade the side signal out before the first mono packet.
#[test]
fn stereo_input_at_low_bitrates_has_the_channel_count_of_libopus() {
    for (rate, application, c_application) in [
        (48000usize, Application::Audio, APP_AUDIO),
        (16000, Application::Voip, APP_VOIP),
    ] {
        let frame = rate / 50;
        let input = to_stereo(&speech_fullband(rate, 6.0), rate);
        let mut c_enc = CEncoder::new(rate, 2, c_application);
        c_enc.complexity(5);
        let mut enc = OpusEncoder::new(
            SampleRate::try_from(rate as i32).unwrap(),
            Channels::Stereo,
            application,
        )
        .unwrap();
        enc.set_complexity(5).unwrap();
        let mut c_dec = CDecoder::new(rate, 2);
        let mut buf = vec![0u8; 4000];
        let (mut differ, mut mono, mut packets) = (0, 0, 0);
        for (i, pcm) in input.chunks_exact(2 * frame).enumerate() {
            // Down from 40 to 8 kb/s and up again, 1 kb/s every 80 ms.
            let step = (i / 4) % 64;
            let bitrate = 8000 + 1000 * step.abs_diff(32) as i32;
            c_enc.bitrate(bitrate);
            enc.set_bitrate(Bitrate::BitsPerSecond(bitrate)).unwrap();
            let n = enc.encode(pcm, frame, &mut buf).unwrap();
            let c_packet = c_enc.encode(pcm);
            c_dec.decode(Some(&buf[..n]), frame, false).unwrap();
            differ += (buf[0] & 4 != c_packet[0] & 4) as usize;
            mono += (c_packet[0] & 4 == 0) as usize;
            packets += 1;
        }
        assert!(mono * 5 >= packets, "{rate} Hz: libopus codes {mono} of {packets} packets as mono");
        assert!(differ * 50 <= packets, "{rate} Hz: the channel count differs in {differ} of {packets} packets");
    }
}
