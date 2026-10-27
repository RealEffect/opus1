//! The multistream controls that opus1 0.2 added, against libopus 1.6.1:
//! the encoder's forwarding of the forced channel count, the prediction
//! switch, phase inversion disabling, the application change and the voice
//! ratio; the decoder's output gain and phase inversion disabling; the
//! getters that libopus answers from the first stream (bandwidth and last
//! packet duration); and the 24-bit calls `opus_multistream_encode24` and
//! `opus_multistream_decode24`.
//!
//! Packets are compared byte for byte for CELT-only streams, where opus1's
//! encoder is libopus's byte for byte. Decoded samples are compared bit for
//! bit. Each stream's setting is read back from both encoders.
#![cfg(feature = "std")]

mod ms_support;

use std::num::NonZeroU32;
use audiopus_sys as sys;
use ms_support::*;
use opus1::multistream::{MultistreamDecoder, MultistreamEncoder};
use opus1::{Application, Channels, ForcedMode, SampleRate};

const FS: i32 = 48_000;
const FRAME: usize = 960;

/// A layout: channel count, streams, coupled streams and mapping.
type Layout = (usize, usize, usize, &'static [u8]);

/// Two coupled streams and a mono stream; one coupled and one mono; two
/// mono streams.
const LAYOUTS: [Layout; 3] = [(5, 3, 2, &[0, 1, 2, 3, 4]), (3, 2, 1, &[0, 1, 2]), (2, 2, 0, &[0, 1])];

/// Music whose odd channels are mostly the inverted even channel before
/// them, so that coupled streams use intensity stereo with phase inversion.
fn anti_phase(channels: usize, samples: usize) -> Vec<f32> {
    let t = tones(FS, channels, 0, samples);
    let mut pcm = vec![0f32; samples * channels];
    for i in 0..samples {
        for c in 0..channels {
            let even = t[i * channels + (c & !1)];
            pcm[i * channels + c] = if c % 2 == 1 { -0.8 * even + 0.1 * t[i * channels + c] } else { even };
        }
    }
    pcm
}

fn same_bytes(a: &[u8], b: &[u8], label: &str) {
    if a != b {
        let i = a.iter().zip(b).position(|(x, y)| x != y).unwrap_or(a.len().min(b.len()));
        panic!("{label}: {} bytes in opus1, {} in libopus, first difference at {i}", a.len(), b.len());
    }
}

fn same_f32(a: &[f32], b: &[f32], label: &str) {
    assert_eq!(a.len(), b.len(), "{label}: length");
    if let Some(i) = a.iter().zip(b).position(|(x, y)| x.to_bits() != y.to_bits()) {
        panic!("{label}: sample {i} is {} in opus1 and {} in libopus", a[i], b[i]);
    }
}

fn pair(layout: Layout, app: i32) -> (MultistreamEncoder, CMsEncoder) {
    let (channels, streams, coupled, mapping) = layout;
    let application = match app {
        APP_VOIP => Application::Voip,
        APP_LOWDELAY => Application::RestrictedLowDelay,
        _ => Application::Audio,
    };
    let rust = MultistreamEncoder::new(
        SampleRate::try_from(FS).unwrap(),
        channels,
        streams,
        coupled,
        mapping,
        application,
    )
    .unwrap();
    let c = CMsEncoder::generic(FS, channels, streams, coupled, mapping, app).unwrap();
    (rust, c)
}

/// Every stream's setting of `request`, from both encoders.
fn assert_streams(rust: &MultistreamEncoder, c: &mut CMsEncoder, request: i32, label: &str) {
    for s in 0..c.streams {
        let enc = rust.stream_encoder(s).unwrap();
        let value = match request {
            sys::OPUS_GET_FORCE_CHANNELS_REQUEST => enc.force_channels().map_or(OPUS_AUTO, |n| n.count() as i32),
            sys::OPUS_GET_PREDICTION_DISABLED_REQUEST => enc.prediction_disabled() as i32,
            sys::OPUS_GET_PHASE_INVERSION_DISABLED_REQUEST => enc.phase_inversion_disabled() as i32,
            sys::OPUS_GET_APPLICATION_REQUEST => enc.application() as i32,
            _ => unreachable!(),
        };
        assert_eq!(value, c.stream_get(s, request), "{label}: stream {s}, request {request}");
    }
}

/// `set_force_channels` goes to every stream. For `None` and mono libopus
/// does the same: every stream's value and the multistream getter agree
/// with libopus after each call. For stereo libopus sets the coupled
/// streams and then refuses the value at the first mono stream; opus1 sets
/// every stream, and a mono stream codes mono whatever is set. After stereo
/// only the coupled streams are compared with libopus.
#[test]
fn forced_channels_are_forwarded_as_libopus_does() {
    for layout in LAYOUTS {
        let label = format!("layout {layout:?}");
        let (mut rust, mut c) = pair(layout, APP_AUDIO);
        let get = sys::OPUS_GET_FORCE_CHANNELS_REQUEST;
        for (value, setting) in [
            (1, Some(Channels::Mono)),
            (OPUS_AUTO, None),
            (2, Some(Channels::Stereo)),
            (1, Some(Channels::Mono)),
        ] {
            let label = format!("{label}, force {value}");
            let c_ok = c.try_ctl(sys::OPUS_SET_FORCE_CHANNELS_REQUEST, value) == 0;
            rust.set_force_channels(setting);
            for s in 0..c.streams {
                assert_eq!(rust.stream_encoder(s).unwrap().force_channels(), setting, "{label}: stream {s}");
            }
            if setting == Some(Channels::Stereo) && c.coupled < c.streams {
                assert!(!c_ok, "{label}: libopus refuses stereo on a mono stream");
                for s in 0..c.coupled {
                    assert_eq!(c.stream_get(s, get), 2, "{label}: stream {s}");
                }
                continue;
            }
            assert!(c_ok, "{label}");
            assert_streams(&rust, &mut c, get, &label);
            assert_eq!(rust.force_channels().map_or(OPUS_AUTO, |n| n.count() as i32), c.get(get), "{label}");
        }
    }
    // In surround mode the coupled streams are forced to two channels on
    // every frame, over the caller's setting, in both. (Surround packets
    // are not compared: opus1 chooses the bandwidth of surround streams
    // differently on purpose, difference D1 of docs/guide/10.)
    let mut rust = MultistreamEncoder::new_surround(
        SampleRate::try_from(FS).unwrap(),
        6,
        1,
        Application::Audio,
    )
    .unwrap();
    let mut c = CMsEncoder::surround(FS, 6, 1, APP_AUDIO).unwrap();
    rust.set_force_channels(Some(Channels::Mono));
    c.ctl(sys::OPUS_SET_FORCE_CHANNELS_REQUEST, 1);
    let pcm = tones(FS, 6, 1, FRAME * 4);
    let mut out = vec![0u8; max_packet_bytes(c.streams)];
    for (i, f) in pcm.chunks_exact(c.channels * FRAME).enumerate() {
        c.encode_float(f, FRAME, out.len()).unwrap();
        rust.encode(f, FRAME, &mut out).unwrap();
        assert_streams(&rust, &mut c, sys::OPUS_GET_FORCE_CHANNELS_REQUEST, &format!("surround, frame {i}"));
    }
}

/// Prediction and phase inversion disabling reach every stream, and the
/// CELT-only packets equal libopus's with each setting. Disabling phase
/// inversion changes the packets of the coupled streams.
#[test]
#[cfg_attr(not(target_arch = "aarch64"), ignore = "bit-exact with libopus only on 64-bit ARM (docs/guide/10, Platforms)")]
fn prediction_and_phase_inversion_are_forwarded_as_libopus_does() {
    for layout in LAYOUTS {
        let (channels, streams, coupled, _) = layout;
        let pcm = anti_phase(channels, FRAME * 25);
        let mut by_setting = Vec::new();
        for (prediction_disabled, inversion_disabled) in [(false, false), (true, false), (false, true), (true, true)] {
            let label = format!("layout {layout:?}, prediction off {prediction_disabled}, inversion off {inversion_disabled}");
            let (mut rust, mut c) = pair(layout, APP_AUDIO);
            let bitrate = 28_000 * streams as i32;
            rust.set_bitrate(opus1::Bitrate::BitsPerSecond(NonZeroU32::new(bitrate as u32).unwrap()));
            c.set_bitrate(bitrate);
            rust.set_force_mode(Some(ForcedMode::CeltOnly));
            c.ctl(OPUS_SET_FORCE_MODE_REQUEST, MODE_CELT_ONLY);
            rust.set_prediction_disabled(prediction_disabled);
            c.ctl(sys::OPUS_SET_PREDICTION_DISABLED_REQUEST, prediction_disabled as i32);
            rust.set_phase_inversion_disabled(inversion_disabled);
            c.ctl(sys::OPUS_SET_PHASE_INVERSION_DISABLED_REQUEST, inversion_disabled as i32);
            for request in [sys::OPUS_GET_PREDICTION_DISABLED_REQUEST, sys::OPUS_GET_PHASE_INVERSION_DISABLED_REQUEST] {
                assert_streams(&rust, &mut c, request, &label);
            }
            let get = sys::OPUS_GET_PHASE_INVERSION_DISABLED_REQUEST;
            assert_eq!(rust.phase_inversion_disabled() as i32, c.get(get), "{label}");
            let get = sys::OPUS_GET_PREDICTION_DISABLED_REQUEST;
            assert_eq!(rust.prediction_disabled() as i32, c.get(get), "{label}");
            let mut out = vec![0u8; max_packet_bytes(streams)];
            let mut packets = Vec::new();
            for (i, f) in pcm.chunks_exact(channels * FRAME).enumerate() {
                let expected = c.encode_float(f, FRAME, out.len()).unwrap();
                let n = rust.encode(f, FRAME, &mut out).unwrap();
                same_bytes(&out[..n], &expected, &format!("{label}: packet {i}"));
                assert_eq!(rust.final_range(), c.final_range(), "{label}: range {i}");
                packets.push(expected);
            }
            by_setting.push(packets);
        }
        if coupled > 0 {
            assert!(by_setting[0] != by_setting[2], "layout {layout:?}: phase inversion changed nothing");
        }
        assert!(by_setting[0] != by_setting[1], "layout {layout:?}: prediction changed nothing");
    }
}

/// `set_application` before the first frame reaches every stream and
/// gives libopus's packets and look-ahead; after the first frame only the
/// current application is accepted. The voice ratio of the first stream
/// equals libopus's after every frame.
#[test]
#[cfg_attr(not(target_arch = "aarch64"), ignore = "bit-exact with libopus only on 64-bit ARM (docs/guide/10, Platforms)")]
fn application_and_voice_ratio_follow_libopus() {
    let apps = [
        (APP_VOIP, Application::Voip),
        (APP_AUDIO, Application::Audio),
        (APP_LOWDELAY, Application::RestrictedLowDelay),
    ];
    for layout in LAYOUTS {
        let (channels, streams, _, _) = layout;
        let pcm = voiced(FS, channels, FRAME * 10);
        for (c_to, to) in apps {
            let label = format!("layout {layout:?}, to {to:?}");
            let (mut rust, mut c) = pair(layout, APP_AUDIO);
            let ret = c.try_ctl(sys::OPUS_SET_APPLICATION_REQUEST, c_to);
            assert_eq!(rust.set_application(to).is_ok(), ret == 0, "{label}");
            assert_eq!(rust.application() as i32, c.get(sys::OPUS_GET_APPLICATION_REQUEST), "{label}");
            assert_streams(&rust, &mut c, sys::OPUS_GET_APPLICATION_REQUEST, &label);
            assert_eq!(rust.lookahead() as i32, c.lookahead(), "{label}: look-ahead");
            rust.set_force_mode(Some(ForcedMode::CeltOnly));
            c.ctl(OPUS_SET_FORCE_MODE_REQUEST, MODE_CELT_ONLY);
            let mut out = vec![0u8; max_packet_bytes(streams)];
            for (i, f) in pcm.chunks_exact(channels * FRAME).enumerate() {
                let expected = c.encode_float(f, FRAME, out.len()).unwrap();
                let n = rust.encode(f, FRAME, &mut out).unwrap();
                same_bytes(&out[..n], &expected, &format!("{label}: packet {i}"));
                let get = OPUS_GET_VOICE_RATIO_REQUEST;
                assert_eq!(rust.voice_ratio().map_or(-1, i32::from), c.get(get), "{label}: voice ratio after {i}");
            }
            for (c_other, other) in apps {
                let c_ok = c.try_ctl(sys::OPUS_SET_APPLICATION_REQUEST, c_other) == 0;
                assert_eq!(rust.set_application(other).is_ok(), c_ok, "{label}, then {other:?}");
                assert_eq!(c_ok, other == to, "{label}, then {other:?}");
            }
        }
    }
}

/// The output gain and phase inversion disabling of every stream decoder,
/// over CELT, SILK and Hybrid packets with losses, with float and 24-bit
/// output. The defaults are libopus's: phase inversion is disabled at the
/// start only when the first stream is mono.
#[test]
#[cfg_attr(not(target_arch = "aarch64"), ignore = "bit-exact with libopus only on 64-bit ARM (docs/guide/10, Platforms)")]
fn decoder_gain_and_phase_inversion_follow_libopus() {
    for layout in LAYOUTS {
        let (channels, streams, coupled, mapping) = layout;
        let pcm = anti_phase(channels, FRAME * 30);
        let mut c_enc = CMsEncoder::generic(FS, channels, streams, coupled, mapping, APP_AUDIO).unwrap();
        c_enc.set_bitrate(24_000 * streams as i32);
        let mut packets = Vec::new();
        for (i, f) in pcm.chunks_exact(channels * FRAME).enumerate() {
            let mode = [MODE_CELT_ONLY, MODE_SILK_ONLY, MODE_HYBRID][(i / 5) % 3];
            c_enc.ctl(OPUS_SET_FORCE_MODE_REQUEST, mode);
            packets.push(c_enc.encode_float(f, FRAME, max_packet_bytes(streams)).unwrap());
        }
        for (gain, inversion) in [(None, None), (Some(-3000i16), Some(true)), (Some(32767), Some(false)), (Some(-32768), None), (Some(700), Some(true))] {
            let label = format!("layout {layout:?}, gain {gain:?}, inversion off {inversion:?}");
            let mut rust = MultistreamDecoder::new(
                SampleRate::try_from(FS).unwrap(),
                channels,
                streams,
                coupled,
                mapping,
            )
            .unwrap();
            let mut c = CMsDecoder::new(FS, channels, streams, coupled, mapping).unwrap();
            let get_inv = sys::OPUS_GET_PHASE_INVERSION_DISABLED_REQUEST;
            assert_eq!(rust.phase_inversion_disabled() as i32, c.get(get_inv), "{label}: default");
            assert_eq!(rust.phase_inversion_disabled(), coupled == 0, "{label}: default");
            if let Some(g) = gain {
                assert_eq!(c.try_ctl(sys::OPUS_SET_GAIN_REQUEST, i32::from(g)), 0);
                rust.set_gain(g);
            }
            if let Some(off) = inversion {
                assert_eq!(c.try_ctl(sys::OPUS_SET_PHASE_INVERSION_DISABLED_REQUEST, off as i32), 0);
                rust.set_phase_inversion_disabled(off);
            }
            assert_eq!(i32::from(rust.gain()), c.get(sys::OPUS_GET_GAIN_REQUEST), "{label}");
            assert_eq!(rust.phase_inversion_disabled() as i32, c.get(get_inv), "{label}");
            let mut out = vec![0f32; MAX_FRAME * channels];
            let mut out24 = vec![0i32; MAX_FRAME * channels];
            let mut rust24 = MultistreamDecoder::new(
                SampleRate::try_from(FS).unwrap(),
                channels,
                streams,
                coupled,
                mapping,
            )
            .unwrap();
            let mut c24 = CMsDecoder::new(FS, channels, streams, coupled, mapping).unwrap();
            if let Some(g) = gain {
                assert_eq!(c24.try_ctl(sys::OPUS_SET_GAIN_REQUEST, i32::from(g)), 0);
                rust24.set_gain(g);
            }
            if let Some(off) = inversion {
                assert_eq!(c24.try_ctl(sys::OPUS_SET_PHASE_INVERSION_DISABLED_REQUEST, off as i32), 0);
                rust24.set_phase_inversion_disabled(off);
            }
            for (i, p) in packets.iter().enumerate() {
                let label = format!("{label}, packet {i}");
                let lost = i % 7 == 3;
                let (data, size) = if lost { (None, FRAME) } else { (Some(&p[..]), MAX_FRAME) };
                let expected = c.decode_float(data, size, false).unwrap();
                let n = rust.decode(data.unwrap_or(&[]), size, &mut out).unwrap();
                same_f32(&out[..n * channels], &expected, &label);
                let expected = c24.decode24(data, size, false).unwrap();
                let n = rust24.decode24(data.unwrap_or(&[]), size, &mut out24).unwrap();
                assert!(out24[..n * channels] == expected[..], "{label}: 24-bit");
            }
        }
        // Gains outside the range of an `i16`, which opus1's `set_gain`
        // cannot be given: libopus refuses them, and the setting stays.
        let mut rust = MultistreamDecoder::new(
            SampleRate::try_from(FS).unwrap(),
            channels,
            streams,
            coupled,
            mapping,
        )
        .unwrap();
        let mut c = CMsDecoder::new(FS, channels, streams, coupled, mapping).unwrap();
        rust.set_gain(-100);
        assert_eq!(c.try_ctl(sys::OPUS_SET_GAIN_REQUEST, -100), 0);
        for v in [i32::MIN, -32769, 32768, i32::MAX] {
            assert_eq!(c.try_ctl(sys::OPUS_SET_GAIN_REQUEST, v), OPUS_BAD_ARG, "gain {v}");
            assert_eq!(i32::from(rust.gain()), c.get(sys::OPUS_GET_GAIN_REQUEST), "gain {v}");
        }
    }
}

/// `encode24` and `decode24` of the multistream encoder and decoder against
/// libopus, CELT-only, with input at and beyond the 24-bit limits.
#[test]
#[cfg_attr(not(target_arch = "aarch64"), ignore = "bit-exact with libopus only on 64-bit ARM (docs/guide/10, Platforms)")]
fn encode24_and_decode24_follow_libopus() {
    for layout in LAYOUTS {
        let (mut rust, mut c) = pair(layout, APP_AUDIO);
        let (channels, streams, coupled, mapping) = layout;
        let label = format!("layout {layout:?}");
        rust.set_force_mode(Some(ForcedMode::CeltOnly));
        c.ctl(OPUS_SET_FORCE_MODE_REQUEST, MODE_CELT_ONLY);
        let t = tones(FS, channels, 0, FRAME * 12);
        let pcm: Vec<i32> = t
            .iter()
            .enumerate()
            .map(|(i, &v)| match (i / (channels * FRAME)) % 4 {
                2 => [8_388_607, -8_388_608, 9_000_001, i32::MIN][i % 4],
                _ => ((v * 12_000_000.0) as i32) | 1,
            })
            .collect();
        let mut out = vec![0u8; max_packet_bytes(streams)];
        let mut rust_dec = MultistreamDecoder::new(
            SampleRate::try_from(FS).unwrap(),
            channels,
            streams,
            coupled,
            mapping,
        )
        .unwrap();
        let mut c_dec = CMsDecoder::for_encoder(&c);
        let mut out24 = vec![0i32; MAX_FRAME * channels];
        for (i, f) in pcm.chunks_exact(channels * FRAME).enumerate() {
            let expected = c.encode24(f, FRAME, out.len()).unwrap();
            let n = rust.encode24(f, FRAME, &mut out).unwrap();
            same_bytes(&out[..n], &expected, &format!("{label}: packet {i}"));
            let decoded = c_dec.decode24(Some(&expected), MAX_FRAME, false).unwrap();
            let n = rust_dec.decode24(&expected, MAX_FRAME, &mut out24).unwrap();
            assert!(out24[..n * channels] == decoded[..], "{label}: decoded {i}");
        }
    }
}

/// In surround mode, in every coding mode, `encode24` codes what `encode`
/// codes for the samples divided by 2^23, as libopus's float build does.
/// The input stays within 24 bits.
#[test]
fn surround_encode24_codes_as_encode_of_the_scaled_samples() {
    for (channels, family) in [(6, 1), (8, 1), (4, 2)] {
        let label = format!("{channels} channels, family {family}");
        let mut a = MultistreamEncoder::new_surround(
            SampleRate::try_from(FS).unwrap(),
            channels,
            family,
            Application::Audio,
        )
        .unwrap();
        let mut b = MultistreamEncoder::new_surround(
            SampleRate::try_from(FS).unwrap(),
            channels,
            family,
            Application::Audio,
        )
        .unwrap();
        let x = voiced(FS, channels, FRAME * 20);
        let pcm: Vec<i32> = x.iter().map(|&v| ((v * 7_000_000.0) as i32) | 1).collect();
        let scaled: Vec<f32> = pcm.iter().map(|&v| v as f32 * (1.0 / 8_388_608.0)).collect();
        let mut out_a = vec![0u8; max_packet_bytes(a.layout().streams())];
        let mut out_b = out_a.clone();
        for (i, (p24, pf)) in pcm.chunks_exact(channels * FRAME).zip(scaled.chunks_exact(channels * FRAME)).enumerate() {
            let na = a.encode24(p24, FRAME, &mut out_a).unwrap();
            let nb = b.encode(pf, FRAME, &mut out_b).unwrap();
            same_bytes(&out_a[..na], &out_b[..nb], &format!("{label}: packet {i}"));
        }
    }
}

/// The getters that libopus answers from the first stream: the encoder's
/// `OPUS_GET_BANDWIDTH`, and the decoder's `OPUS_GET_BANDWIDTH` and
/// `OPUS_GET_LAST_PACKET_DURATION`. They equal libopus's before the first
/// frame and after every call, over packets of every coding mode at several
/// bitrates, with losses, FEC calls and a reset. The encoder's setting
/// getters (`bandwidth_setting`, `force_mode`) keep what was set. Packets
/// with SILK are not compared byte for byte (see the top of this file); the
/// decoder reads libopus's packets.
#[test]
#[cfg_attr(not(target_arch = "aarch64"), ignore = "bit-exact with libopus only on 64-bit ARM (docs/guide/10, Platforms)")]
fn getters_of_the_first_stream_follow_libopus() {
    let get_bw = sys::OPUS_GET_BANDWIDTH_REQUEST;
    let get_dur = sys::OPUS_GET_LAST_PACKET_DURATION_REQUEST;
    for layout in LAYOUTS {
        let (channels, streams, coupled, mapping) = layout;
        let label = format!("layout {layout:?}");
        let (mut rust, mut c) = pair(layout, APP_AUDIO);
        assert_eq!(rust.channels(), channels, "{label}");
        assert_eq!(rust.bandwidth().to_raw(), c.get(get_bw), "{label}: encoder before the first frame");
        let pcm = voiced(FS, channels, FRAME * 36);
        let mut out = vec![0u8; max_packet_bytes(streams)];
        let mut packets = Vec::new();
        for (i, f) in pcm.chunks_exact(channels * FRAME).enumerate() {
            // Low to high bitrates in each coding mode, so that the
            // bandwidth changes.
            let (mode, forced) = [
                (MODE_CELT_ONLY, ForcedMode::CeltOnly),
                (MODE_SILK_ONLY, ForcedMode::SilkOnly),
                (MODE_HYBRID, ForcedMode::Hybrid),
            ][(i / 12) % 3];
            let bitrate = [8_000, 14_000, 24_000, 64_000][(i / 3) % 4] * streams as i32;
            rust.set_force_mode(Some(forced));
            c.ctl(OPUS_SET_FORCE_MODE_REQUEST, mode);
            rust.set_bitrate(opus1::Bitrate::BitsPerSecond(NonZeroU32::new(bitrate as u32).unwrap()));
            c.set_bitrate(bitrate);
            assert_eq!(rust.force_mode(), Some(forced), "{label}: frame {i}");
            let expected = c.encode_float(f, FRAME, out.len()).unwrap();
            rust.encode(f, FRAME, &mut out).unwrap();
            assert_eq!(rust.bandwidth().to_raw(), c.get(get_bw), "{label}: encoder after frame {i}");
            assert_eq!(rust.bandwidth(), rust.stream_encoder(0).unwrap().bandwidth(), "{label}: frame {i}");
            packets.push(expected);
        }
        assert_eq!(rust.bandwidth_setting(), None, "{label}");

        let mut rust_dec =
            MultistreamDecoder::new(SampleRate::try_from(FS).unwrap(), channels, streams, coupled, mapping).unwrap();
        let mut c_dec = CMsDecoder::new(FS, channels, streams, coupled, mapping).unwrap();
        assert_eq!(rust_dec.channels(), channels, "{label}");
        let check = |rust_dec: &MultistreamDecoder, c_dec: &mut CMsDecoder, what: &str| {
            assert_eq!(rust_dec.bandwidth().map_or(0, |b| b.to_raw()), c_dec.get(get_bw), "{label}: decoder {what}");
            assert_eq!(rust_dec.last_packet_duration() as i32, c_dec.get(get_dur), "{label}: decoder {what}");
        };
        check(&rust_dec, &mut c_dec, "before the first packet");
        let mut pcm_out = vec![0f32; MAX_FRAME * channels];
        for (i, p) in packets.iter().enumerate() {
            if i == 20 {
                rust_dec.reset_state();
                assert_eq!(c_dec.try_ctl(sys::OPUS_RESET_STATE, 0), 0);
                check(&rust_dec, &mut c_dec, "after a reset");
            }
            let what = format!("after packet {i}");
            match i % 6 {
                // A loss of 10 ms.
                2 => {
                    let expected = c_dec.decode_float(None, FRAME / 2, false).unwrap();
                    let n = rust_dec.decode(&[], FRAME / 2, &mut pcm_out).unwrap();
                    same_f32(&pcm_out[..n * channels], &expected, &format!("{label}: {what}"));
                }
                // The packet before this one was lost: rebuild it from this
                // one, then decode this one.
                4 => {
                    let expected = c_dec.decode_float(Some(p), FRAME, true).unwrap();
                    let n = rust_dec.decode_fec(p, FRAME, &mut pcm_out).unwrap();
                    same_f32(&pcm_out[..n * channels], &expected, &format!("{label}: FEC {what}"));
                    check(&rust_dec, &mut c_dec, &format!("FEC {what}"));
                    let expected = c_dec.decode_float(Some(p), MAX_FRAME, false).unwrap();
                    let n = rust_dec.decode(p, MAX_FRAME, &mut pcm_out).unwrap();
                    same_f32(&pcm_out[..n * channels], &expected, &format!("{label}: {what}"));
                }
                _ => {
                    let expected = c_dec.decode_float(Some(p), MAX_FRAME, false).unwrap();
                    let n = rust_dec.decode(p, MAX_FRAME, &mut pcm_out).unwrap();
                    same_f32(&pcm_out[..n * channels], &expected, &format!("{label}: {what}"));
                }
            }
            check(&rust_dec, &mut c_dec, &what);
            assert_eq!(rust_dec.bandwidth(), rust_dec.stream_decoder(0).unwrap().bandwidth(), "{label}: {what}");
        }
    }

    // A 5.1 surround encoder with an explicit bitrate, where the encoder
    // sets the bandwidth of every stream on every frame (docs/guide/10, D1 is
    // about the automatic bitrate only).
    let mut rust =
        MultistreamEncoder::new_surround(SampleRate::try_from(FS).unwrap(), 6, 1, Application::Audio).unwrap();
    let mut c = CMsEncoder::surround(FS, 6, 1, APP_AUDIO).unwrap();
    assert_eq!(rust.bandwidth().to_raw(), c.get(get_bw), "surround before the first frame");
    let pcm = tones(FS, 6, 1, FRAME * 12);
    let mut out = vec![0u8; max_packet_bytes(c.streams)];
    for (i, f) in pcm.chunks_exact(c.channels * FRAME).enumerate() {
        let bitrate = [24_000, 40_000, 56_000, 128_000][i % 4];
        rust.set_bitrate(opus1::Bitrate::BitsPerSecond(NonZeroU32::new(bitrate).unwrap()));
        c.set_bitrate(bitrate as i32);
        c.encode_float(f, FRAME, out.len()).unwrap();
        rust.encode(f, FRAME, &mut out).unwrap();
        assert_eq!(rust.bandwidth().to_raw(), c.get(get_bw), "surround after frame {i}");
        // The coupled streams were forced to CELT-only; the setting is
        // still the caller's.
        assert_eq!(rust.stream_encoder(0).unwrap().force_mode(), Some(ForcedMode::CeltOnly));
        assert_eq!(rust.force_mode(), None, "surround frame {i}");
    }
}
