//! The multistream encoder with frames of 80, 100 and 120 ms, and its
//! expert frame duration control, against C libopus 1.6.1.
//!
//! - 5.1 surround at 16, 24 and 48 kHz: libopus's multistream decoder
//!   decodes every packet to the whole duration and ends in the encoder's
//!   final range. With CBR every packet has the size of libopus's packet.
//!   The decoded audio is within 0.5 dB of what libopus's multistream
//!   encoder gives, or of opus1's distance to it at 20 to 60 ms.
//! - The expert frame duration: libopus's multistream encoder stores any
//!   number and then refuses every encode call when the number is no
//!   duration. opus1 takes a `FrameDuration`, which holds only durations.
//!   For each of them both encoders take and refuse the same calls and code
//!   the same duration.
#![cfg(feature = "std")]

mod ms_support;
mod multistream_enc_common;

use audiopus_sys as sys;
use ms_support::*;
use multistream_enc_common::*;
use opus1::{Application, FrameDuration};

const SET: i32 = sys::OPUS_SET_EXPERT_FRAME_DURATION_REQUEST;
const GET: i32 = sys::OPUS_GET_EXPERT_FRAME_DURATION_REQUEST;

/// Mean SNR in dB over the channels of `decoded` against `input`, both
/// interleaved, with the decoded signal `lag` samples late. The first
/// 100 ms are left out.
fn mean_snr(input: &[f32], decoded: &[f32], channels: usize, lag: usize, fs: i32) -> f64 {
    (0..channels)
        .map(|c| {
            let reference = channel(input, channels, c);
            let test = channel(decoded, channels, c);
            snr_at_lag(&reference, &test, lag as isize, fs as usize / 10)
        })
        .sum::<f64>()
        / channels as f64
}

/// Encodes `pcm` (5.1, interleaved) in frames of `ms` with both encoders,
/// set up by `configure`. libopus's multistream decoder decodes both. Every
/// packet of opus1 must decode to the whole frame and end in the encoder's
/// final range, and with CBR have the size of libopus's packet. Returns the
/// mean SNR over the channels for opus1 and for libopus.
fn encode_5_1(fs: i32, app: Application, bitrate: i32, vbr: bool, ms: usize, pcm: &[f32]) -> (f64, f64) {
    let case = Case::family("5.1", 6, 1);
    let label = format!("5.1 {fs} Hz {app:?} {ms} ms vbr {vbr} {bitrate} b/s");
    let frame = fs as usize * ms / 1000;
    let (mut rust, mut c) = encoders(&case, fs, app);
    configure(&mut rust, &mut c, bitrate, vbr);
    let mut rust_dec = c_decoder(&rust);
    let mut c_dec = CMsDecoder::for_encoder(&c);
    let mut out = vec![0u8; max_packet_bytes(c.streams)];
    let (mut rust_pcm, mut c_pcm) = (Vec::new(), Vec::new());
    for (k, input) in pcm.chunks_exact(frame * 6).enumerate() {
        let n = rust.encode(input, frame, &mut out).unwrap_or_else(|e| panic!("{label}: packet {k}: {e}"));
        let c_packet = c.encode_float(input, frame, out.len()).unwrap();
        let decoded = rust_dec
            .decode_float(Some(&out[..n]), MAX_FRAME, false)
            .unwrap_or_else(|e| panic!("{label}: libopus rejects packet {k}: {e}"));
        assert_eq!(decoded.len(), frame * 6, "{label}: packet {k} duration");
        assert_eq!(rust_dec.final_range(), rust.final_range(), "{label}: packet {k} final range");
        if !vbr {
            assert_eq!(n, c_packet.len(), "{label}: packet {k} size");
        }
        rust_pcm.extend(decoded);
        c_pcm.extend(c_dec.decode_float(Some(&c_packet), MAX_FRAME, false).unwrap());
    }
    let lag = rust.lookahead();
    (mean_snr(pcm, &rust_pcm, 6, lag, fs), mean_snr(pcm, &c_pcm, 6, lag, fs))
}

#[test]
fn surround_5_1_long_frames_decode_in_libopus() {
    // The quality rule: opus1 is within 0.5 dB of libopus, or within 0.5 dB
    // of what it gives against libopus with the same settings in frames of
    // 20 to 60 ms. libopus runs its speech and music analysis in each
    // stream, which opus1 lacks; that alone puts opus1 up to 0.7 dB below
    // libopus at every duration (24 kHz, `Audio`, 256 kb/s CBR).
    for fs in [48_000, 24_000, 16_000] {
        let pcm = voiced(fs, 6, 12 * fs as usize / 5);
        for app in [Application::Audio, Application::Voip] {
            for vbr in [true, false] {
                for bitrate in [96_000, 256_000] {
                    let mut part = None;
                    for ms in [80usize, 100, 120] {
                        let label = format!("5.1 {fs} Hz {app:?} {ms} ms vbr {vbr} {bitrate} b/s");
                        let (rust_snr, c_snr) = encode_5_1(fs, app, bitrate, vbr, ms, &pcm);
                        println!("{label}: opus1 {rust_snr:.2} dB, libopus {c_snr:.2} dB");
                        let difference = rust_snr - c_snr;
                        if difference < -0.5 {
                            let part = *part.get_or_insert_with(|| {
                                [20, 40, 60]
                                    .map(|ms| {
                                        let (a, b) = encode_5_1(fs, app, bitrate, vbr, ms, &pcm);
                                        a - b
                                    })
                                    .into_iter()
                                    .fold(0.0f64, f64::min)
                            });
                            assert!(
                                difference >= part - 0.5,
                                "{label}: opus1 {rust_snr:.2} dB, libopus {c_snr:.2} dB, at 20 to 60 ms {part:+.2} dB"
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn expert_frame_duration_matches_libopus() {
    let case = Case::family("5.1", 6, 1);
    for fs in [48_000, 16_000] {
        let unit = fs as usize / 400;
        let pcm = voiced(fs, 6, 50 * unit + 1);
        let sizes: Vec<usize> =
            (0..=50).map(|k| k * unit).chain([1, unit + 1, 8 * unit + 1, 48 * unit - 1]).collect();
        let values = [i32::MIN, -1, 0, 4999].into_iter().chain(5000..=5009).chain([5010, 6000]);
        for value in values {
            let (mut rust, mut c) = encoders(&case, fs, Application::Audio);
            configure(&mut rust, &mut c, 256_000, false);
            // libopus's multistream encoder takes any number.
            assert_eq!(c.try_ctl(SET, value), 0, "value {value}");
            assert_eq!(c.get(GET), value);
            let duration = FrameDuration::try_from(value);
            if let Ok(d) = duration {
                rust.set_expert_frame_duration(d);
                assert_eq!(rust.expert_frame_duration(), d);
            }
            let mut out = vec![0u8; max_packet_bytes(c.streams)];
            for &size in &sizes {
                let label = format!("{fs} Hz, value {value}, frame size {size}");
                let c_result = c.encode_float(&pcm, size, out.len());
                let Ok(d) = duration else {
                    // A number that is no duration: libopus refuses every
                    // call, and opus1 cannot hold the number.
                    assert_eq!(c_result, Err(OPUS_BAD_ARG), "{label}");
                    continue;
                };
                match (rust.encode(&pcm, size, &mut out), c_result) {
                    (Ok(n), Ok(c_packet)) => {
                        let samples = |p: &[u8]| split_streams(p, c.streams).unwrap()[0].samples(fs);
                        assert_eq!(samples(&out[..n]), samples(&c_packet), "{label}");
                        assert_eq!(samples(&c_packet), d.samples(fs).unwrap_or(size), "{label}");
                        assert_eq!(n, c_packet.len(), "{label}: CBR size");
                    }
                    (Err(_), Err(e)) => assert_eq!(e, OPUS_BAD_ARG, "{label}"),
                    (rust_result, c_result) => {
                        panic!("{label}: opus1 {rust_result:?}, libopus {:?}", c_result.map(|p| p.len()))
                    }
                }
            }
        }
    }
}

#[test]
fn the_setting_is_kept_by_the_multistream_encoder() {
    // libopus keeps the setting in the multistream encoder and never hands
    // it to the stream encoders: they get the chosen frame size directly.
    let case = Case::family("5.1", 6, 1);
    let (mut rust, mut c) = encoders(&case, 48_000, Application::Audio);
    rust.set_expert_frame_duration(FrameDuration::Ms40);
    c.ctl(SET, sys::OPUS_FRAMESIZE_40_MS);
    for s in 0..c.streams {
        assert_eq!(c.stream_get(s, GET), sys::OPUS_FRAMESIZE_ARG);
        assert_eq!(rust.stream_encoder(s).unwrap().expert_frame_duration(), FrameDuration::Argument);
    }
    // A duration set on a stream encoder plays no part.
    rust.stream_encoder_mut(0).unwrap().set_expert_frame_duration(FrameDuration::Ms10);
    let pcm = voiced(48_000, 6, 5760);
    let mut out = vec![0u8; max_packet_bytes(c.streams)];
    let n = rust.encode(&pcm, 5760, &mut out).unwrap();
    for stream in split_streams(&out[..n], c.streams).unwrap() {
        assert_eq!(stream.samples(48_000), 1920);
    }
    rust.reset_state();
    c.ctl(sys::OPUS_RESET_STATE, 0);
    assert_eq!(rust.expert_frame_duration(), FrameDuration::Ms40);
    assert_eq!(c.get(GET), sys::OPUS_FRAMESIZE_40_MS);
}
