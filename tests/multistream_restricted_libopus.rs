//! The multistream encoder with the applications `RestrictedSilk` and
//! `RestrictedCelt` of libopus 1.6 (2052 and 2053), against libopus 1.6.1:
//! generic layouts, surround layouts of mapping family 1 (with the
//! low-frequency-effects stream from 6 channels on), ambisonics (family 2)
//! and family 255.
//!
//! - `RestrictedCelt`: every stream is CELT-only. The packets and final
//!   ranges are compared byte for byte (on 64-bit ARM), and they are those
//!   of `RestrictedLowDelay`.
//! - `RestrictedSilk`: every stream is SILK-only, at most wideband, and the
//!   surround layouts run no surround masking analysis, as in libopus.
//!   opus1's SILK encoder is not the float one of the linked libopus (see
//!   tests/core_restricted_libopus.rs), so each stream's configuration and
//!   stereo flag, the size of constant-bitrate packets and the final range
//!   of a decoder are compared.
//! - Both: the frame sizes an encode call takes, the look-ahead and the
//!   application change.
//!
//! Every stream gets an explicit bitrate. With the automatic or the maximum
//! bitrate, a surround encoder chooses another bandwidth than libopus on
//! purpose (difference D1 of docs/10).
#![cfg(feature = "std")]

mod ms_support;

use std::num::NonZeroU32;
use audiopus_sys as sys;
use ms_support::*;
use opus1::multistream::MultistreamEncoder;
use opus1::{Application, Bitrate, ErrorKind, SampleRate};

/// libopus `OPUS_APPLICATION_RESTRICTED_SILK` and
/// `OPUS_APPLICATION_RESTRICTED_CELT`, which `audiopus_sys` does not name.
const APP_SILK: i32 = 2052;
const APP_CELT: i32 = 2053;

/// How an encoder gets its layout.
#[derive(Debug, Clone, Copy)]
enum Layout {
    /// `new`: channels, streams, coupled streams and mapping.
    Generic(usize, usize, usize, &'static [u8]),
    /// `new_surround`: channels and mapping family.
    Family(usize, u8),
}

const LAYOUTS: [Layout; 13] = [
    Layout::Generic(5, 3, 2, &[0, 1, 2, 3, 4]),
    Layout::Generic(3, 2, 1, &[0, 1, 2]),
    Layout::Generic(2, 2, 0, &[0, 1]),
    Layout::Family(1, 1),
    Layout::Family(2, 1),
    Layout::Family(3, 1),
    Layout::Family(4, 1),
    Layout::Family(5, 1),
    Layout::Family(6, 1),
    Layout::Family(7, 1),
    Layout::Family(8, 1),
    Layout::Family(4, 2),
    Layout::Family(3, 255),
];

fn pair(layout: Layout, fs: i32, c_app: i32, app: Application) -> (MultistreamEncoder, CMsEncoder) {
    let rate = SampleRate::try_from(fs).unwrap();
    match layout {
        Layout::Generic(channels, streams, coupled, mapping) => (
            MultistreamEncoder::new(rate, channels, streams, coupled, mapping, app).unwrap(),
            CMsEncoder::generic(fs, channels, streams, coupled, mapping, c_app).unwrap(),
        ),
        Layout::Family(channels, family) => (
            MultistreamEncoder::new_surround(rate, channels, family, app).unwrap(),
            CMsEncoder::surround(fs, channels, family as i32, c_app).unwrap(),
        ),
    }
}

fn bps(bitrate: i32) -> Bitrate {
    Bitrate::BitsPerSecond(NonZeroU32::new(bitrate as u32).unwrap())
}

#[cfg(feature = "analysis")]
fn same_bytes(a: &[u8], b: &[u8], label: &str) {
    if a != b {
        let i = a.iter().zip(b).position(|(x, y)| x != y).unwrap_or(a.len().min(b.len()));
        panic!("{label}: {} bytes in opus1, {} in libopus, first difference at {i}", a.len(), b.len());
    }
}

/// The frame sizes Opus has at `fs`: 2.5, 5, 10, 20, 40, 60, 80, 100 and
/// 120 ms.
fn frame_sizes(fs: i32) -> [usize; 9] {
    [1, 2, 4, 8, 16, 24, 32, 40, 48].map(|k| k * fs as usize / 400)
}

/// Voiced sounds on the first half of the channels and tones on the
/// others, so that the streams differ.
fn signal(fs: i32, channels: usize, samples: usize) -> Vec<f32> {
    let v = voiced(fs, channels, samples);
    let t = tones(fs, channels, 1, samples);
    let mut pcm = v.clone();
    for i in 0..samples {
        for c in channels / 2..channels {
            pcm[i * channels + c] = 0.5 * t[i * channels + c];
        }
    }
    pcm
}

/// The look-ahead, the application and its change, and the frame sizes,
/// for both applications and every layout: the encoder takes and refuses
/// what libopus takes and refuses. `RestrictedSilk` refuses frames of 2.5
/// and 5 ms with `OPUS_BAD_ARG`.
#[test]
fn settings_and_frame_sizes_follow_libopus() {
    let fs = 48_000;
    for (c_app, app) in [(APP_SILK, Application::RestrictedSilk), (APP_CELT, Application::RestrictedCelt)] {
        for layout in LAYOUTS {
            let label = format!("{app:?}, {layout:?}");
            let (mut rust, mut c) = pair(layout, fs, c_app, app);
            assert_eq!(rust.application().to_raw(), c.get(sys::OPUS_GET_APPLICATION_REQUEST), "{label}");
            assert_eq!(rust.lookahead() as i32, c.lookahead(), "{label}: look-ahead");
            for (c_to, to) in [
                (APP_AUDIO, Application::Audio),
                (APP_LOWDELAY, Application::RestrictedLowDelay),
                (APP_SILK, Application::RestrictedSilk),
                (APP_CELT, Application::RestrictedCelt),
            ] {
                let c_ok = c.try_ctl(sys::OPUS_SET_APPLICATION_REQUEST, c_to) == 0;
                let result = rust.set_application(to);
                assert!(!c_ok, "{label}: libopus changes to {to:?}");
                assert_eq!(result.map_err(|e| e.kind()), Err(ErrorKind::BadArg), "{label}: to {to:?}");
                assert_eq!(rust.application().to_raw(), c.get(sys::OPUS_GET_APPLICATION_REQUEST), "{label}");
            }
            let channels = rust.layout().channels();
            rust.set_bitrate(bps(32_000 * channels as i32));
            c.set_bitrate(32_000 * channels as i32);
            let pcm = signal(fs, channels, 6 * fs as usize / 50);
            let mut out = vec![0u8; max_packet_bytes(c.streams)];
            for size in frame_sizes(fs).into_iter().chain([3 * fs as usize / 400, fs as usize / 100 + 1]) {
                let label = format!("{label}, frame of {size} samples");
                let input = &pcm[..size * channels];
                let c_result = c.encode_float(input, size, out.len());
                let result = rust.encode(input, size, &mut out);
                match (&result, &c_result) {
                    (Ok(_), Ok(_)) => {}
                    (Err(e), Err(c_err)) => {
                        assert_eq!(*c_err, OPUS_BAD_ARG, "{label}");
                        assert_eq!(e.kind(), ErrorKind::BadArg, "{label}");
                    }
                    _ => panic!("{label}: opus1 {:?}, libopus {:?}", result.map(|_| ()), c_result.map(|_| ())),
                }
                let silk_refuses = app == Application::RestrictedSilk && size < fs as usize / 100;
                assert_eq!(result.is_err(), silk_refuses || !frame_sizes(fs).contains(&size), "{label}");
            }
        }
    }
}

/// One stream of `RestrictedCelt` multistream packets against libopus's:
/// every packet byte for byte, and the final range. The packets are also
/// those of an opus1 `RestrictedLowDelay` encoder. Returns the number of
/// packets.
#[cfg(feature = "analysis")]
fn check_celt(layout: Layout, fs: i32, frame: usize, bitrate_per_channel: i32, complexity: u8, vbr: bool) -> usize {
    let label = format!("{layout:?}, {fs} Hz, {frame} samples, {bitrate_per_channel} b/s per channel, c{complexity}, vbr {vbr}");
    let (mut rust, mut c) = pair(layout, fs, APP_CELT, Application::RestrictedCelt);
    let (mut low_delay, _) = pair(layout, fs, APP_LOWDELAY, Application::RestrictedLowDelay);
    let channels = rust.layout().channels();
    let bitrate = bitrate_per_channel * channels as i32;
    for enc in [&mut rust, &mut low_delay] {
        enc.set_bitrate(bps(bitrate));
        enc.set_complexity(complexity).unwrap();
        enc.set_vbr(vbr);
    }
    c.set_bitrate(bitrate).set_complexity(complexity as i32).set_vbr(vbr);
    let mut c_dec = CMsDecoder::for_encoder(&c);
    let pcm = signal(fs, channels, fs as usize / 2);
    let mut out = vec![0u8; max_packet_bytes(c.streams)];
    let mut out_low_delay = out.clone();
    let mut packets = 0;
    for (i, f) in pcm.chunks_exact(frame * channels).enumerate() {
        let label = format!("{label}: packet {i}");
        let expected = c.encode_float(f, frame, out.len()).unwrap();
        let n = rust.encode(f, frame, &mut out).unwrap();
        same_bytes(&out[..n], &expected, &label);
        assert_eq!(rust.final_range(), c.final_range(), "{label}: final range");
        c_dec.decode_float(Some(&expected), frame, false).unwrap();
        assert_eq!(c_dec.final_range(), c.final_range(), "{label}: final range of the decoder");
        let m = low_delay.encode(f, frame, &mut out_low_delay).unwrap();
        assert!(out[..n] == out_low_delay[..m], "{label}: the RestrictedLowDelay packet differs");
        packets += 1;
    }
    packets
}

/// `RestrictedCelt` multistream packets at 48 and 16 kHz, frames of 2.5 to
/// 60 ms, two bitrates, complexities 0 and 10, VBR (up to 20 ms) and CBR,
/// over every layout, surround ones included. At complexity 10 the speech
/// and music analysis runs, so the test needs the feature `analysis`.
#[test]
#[cfg(feature = "analysis")]
#[cfg_attr(not(target_arch = "aarch64"), ignore = "bit-exact with libopus only on 64-bit ARM (docs/10, Platforms)")]
fn restricted_celt_packets_and_final_ranges_are_libopus() {
    let debug = cfg!(debug_assertions);
    let layouts: &[Layout] = if debug { &LAYOUTS[..4] } else { &LAYOUTS };
    let mut packets = 0;
    for &layout in layouts {
        for fs in [48_000, 16_000] {
            for frame in frame_sizes(fs).into_iter().take(6) {
                for (bitrate, complexity) in [(24_000, 10), (64_000, 0)] {
                    packets += check_celt(layout, fs, frame, bitrate, complexity, false);
                    if frame <= fs as usize / 50 {
                        packets += check_celt(layout, fs, frame, bitrate, complexity, true);
                    }
                }
            }
        }
    }
    println!("{packets} packets, all equal to libopus's");
}

/// `RestrictedSilk` multistream packets against libopus's at 48 and
/// 16 kHz, frames of 10 to 120 ms, two bitrates, VBR (up to 60 ms) and
/// CBR, over every layout:
///
/// - every stream of every packet is SILK-only, with libopus's
///   configuration and stereo flag; the low-frequency-effects stream is
///   narrowband;
/// - no stream gets an energy mask, also in surround mode;
/// - with CBR every packet has libopus's size;
/// - libopus's decoder ends opus1's packet in opus1's final range.
#[test]
fn restricted_silk_streams_follow_libopus() {
    let debug = cfg!(debug_assertions);
    let layouts: &[Layout] = if debug { &LAYOUTS[..4] } else { &LAYOUTS };
    let (mut packets, mut streams_checked) = (0, 0);
    for &layout in layouts {
        for fs in [48_000, 16_000] {
            for frame in frame_sizes(fs).into_iter().skip(2) {
                for bitrate_per_channel in [12_000, 24_000] {
                    for vbr in [true, false] {
                        if vbr && frame > 3 * fs as usize / 50 {
                            // opus1 leaves out libopus's limit of each frame
                            // of such a packet under VBR on purpose (docs/10).
                            continue;
                        }
                        let label = format!("{layout:?}, {fs} Hz, {frame} samples, {bitrate_per_channel} b/s per channel, vbr {vbr}");
                        let (mut rust, mut c) = pair(layout, fs, APP_SILK, Application::RestrictedSilk);
                        let channels = rust.layout().channels();
                        let bitrate = bitrate_per_channel * channels as i32;
                        rust.set_bitrate(bps(bitrate));
                        rust.set_vbr(vbr);
                        rust.set_complexity(5).unwrap();
                        c.set_bitrate(bitrate).set_vbr(vbr).set_complexity(5);
                        let mut c_dec = CMsDecoder::for_encoder(&c);
                        let pcm = signal(fs, channels, 3 * fs as usize / 5);
                        let mut out = vec![0u8; max_packet_bytes(c.streams)];
                        for (i, f) in pcm.chunks_exact(frame * channels).enumerate() {
                            let label = format!("{label}: packet {i}");
                            let expected = c.encode_float(f, frame, out.len()).unwrap();
                            let n = rust.encode(f, frame, &mut out).unwrap();
                            if !vbr {
                                assert_eq!(n, expected.len(), "{label}: CBR size");
                            }
                            let ours = split_streams(&out[..n], c.streams).unwrap();
                            let theirs = split_streams(&expected, c.streams).unwrap();
                            for (s, (a, b)) in ours.iter().zip(&theirs).enumerate() {
                                let t = toc(a.toc);
                                assert_eq!(t.mode, Mode::Silk, "{label}: stream {s}: {t}");
                                // With CBR the last stream gets the bytes the
                                // others left, and the SILK encoders use
                                // different numbers of bytes.
                                let last = s + 1 == c.streams;
                                if vbr || !last {
                                    assert_eq!(a.toc & 0xFC, b.toc & 0xFC, "{label}: stream {s}: {t} against {}", toc(b.toc));
                                    streams_checked += 1;
                                }
                                if rust.lfe_stream() == Some(s) {
                                    assert_eq!(t.bandwidth, Bandwidth::Nb, "{label}: LFE stream {s}");
                                }
                                assert_eq!(rust.stream_encoder(s).unwrap().energy_mask(), None, "{label}: stream {s}");
                            }
                            c_dec.decode_float(Some(&out[..n]), frame, false).unwrap();
                            assert_eq!(c_dec.final_range(), rust.final_range(), "{label}: final range");
                            packets += 1;
                        }
                    }
                }
            }
        }
    }
    println!("{packets} packets, {streams_checked} stream packets with libopus's configuration");
}
