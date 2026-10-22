//! The expert frame duration control of `OpusEncoder` against C libopus
//! 1.6.1: `OPUS_SET_EXPERT_FRAME_DURATION` and
//! `OPUS_GET_EXPERT_FRAME_DURATION`.
//!
//! With a duration set, an encode call codes a frame of that duration out
//! of the samples it is given (libopus `frame_size_select`), and refuses
//! the call when they are fewer. Checked here:
//!
//! - The same numbers are accepted and refused, and the getter reports
//!   the same value.
//! - For every duration and every frame size passed, both encoders accept
//!   or refuse the call alike, and code the same duration. CELT-only
//!   packets with CBR at complexity 5, which opus1 writes as libopus does,
//!   are the same bytes, for `f32` and for 16-bit input.
//! - The setting stays across a reset.

mod core_common;

use audiopus_sys as sys;
use core_common::*;
use opus1::{Application, Bitrate, FrameDuration, OpusEncoder, packet};

const SET: i32 = sys::OPUS_SET_EXPERT_FRAME_DURATION_REQUEST;
const GET: i32 = sys::OPUS_GET_EXPERT_FRAME_DURATION_REQUEST;

#[test]
fn values_are_those_of_libopus() {
    let constants = [
        (FrameDuration::Argument, sys::OPUS_FRAMESIZE_ARG),
        (FrameDuration::Ms2_5, sys::OPUS_FRAMESIZE_2_5_MS),
        (FrameDuration::Ms5, sys::OPUS_FRAMESIZE_5_MS),
        (FrameDuration::Ms10, sys::OPUS_FRAMESIZE_10_MS),
        (FrameDuration::Ms20, sys::OPUS_FRAMESIZE_20_MS),
        (FrameDuration::Ms40, sys::OPUS_FRAMESIZE_40_MS),
        (FrameDuration::Ms60, sys::OPUS_FRAMESIZE_60_MS),
        (FrameDuration::Ms80, sys::OPUS_FRAMESIZE_80_MS),
        (FrameDuration::Ms100, sys::OPUS_FRAMESIZE_100_MS),
        (FrameDuration::Ms120, sys::OPUS_FRAMESIZE_120_MS),
    ];
    assert_eq!(FrameDuration::ALL.map(|d| d.to_raw()), constants.map(|(_, c)| c));
    for (d, c) in constants {
        assert_eq!(d.to_raw(), c, "{d:?}");
        assert_eq!(FrameDuration::try_from(c), Ok(d));
    }
}

#[test]
fn accepted_and_refused_values_match_libopus() {
    for app in [APP_VOIP, APP_AUDIO, APP_LOWDELAY] {
        let mut c = CEncoder::new(48000, 2, app);
        let mut rs = OpusEncoder::new(48000, 2, Application::Audio).unwrap();
        // The default.
        assert_eq!(c.get(GET), sys::OPUS_FRAMESIZE_ARG);
        assert_eq!(rs.expert_frame_duration(), FrameDuration::Argument);
        let values = [i32::MIN, -5000, -1, 0, 1, 20, 960, 4040, 4999]
            .into_iter()
            .chain(5000..=5009)
            .chain([5010, 5011, 5100, 6000, i32::MAX]);
        for value in values {
            let c_ok = c.try_set(SET, value) == 0;
            let rs_value = FrameDuration::try_from(value);
            assert_eq!(rs_value.is_ok(), c_ok, "value {value}");
            if let Ok(d) = rs_value {
                rs.set_expert_frame_duration(d);
            }
            // A refused value leaves the setting as it was, in both.
            assert_eq!(c.get(GET), rs.expert_frame_duration().to_raw(), "after value {value}");
        }
    }
}

/// The frame sizes passed to the encoders at `fs`: every multiple of
/// 2.5 ms up to 125 ms, and sizes that are no such multiple.
fn caller_sizes(fs: usize) -> Vec<usize> {
    let unit = fs / 400;
    (0..=50).map(|k| k * unit).chain([1, unit - 1, unit + 1, 8 * unit + 1, 48 * unit - 1]).collect()
}

/// A tone with a slow vibrato and some noise, `samples` per channel.
fn test_signal(fs: usize, channels: usize, samples: usize) -> Vec<f32> {
    let mut seed = 0x2468_ace1u32;
    (0..samples * channels)
        .map(|k| {
            let t = (k / channels) as f32 / fs as f32;
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            let noise = (seed >> 9) as f32 / (1u32 << 23) as f32 - 0.5;
            let f = 330.0 + 40.0 * (6.0 * t).sin() + 110.0 * (k % channels) as f32;
            0.4 * (std::f32::consts::TAU * f * t).sin() + 0.05 * noise
        })
        .collect()
}

/// For each duration setting: the two encoders take and refuse the same
/// calls, and code the same duration. With `same_bytes` the packets must
/// be libopus's byte for byte.
fn check_calls(fs: usize, channels: usize, app: (i32, Application), cbr: bool, same_bytes: bool, i16_input: bool) {
    let sizes = caller_sizes(fs);
    let pcm = test_signal(fs, channels, 50 * fs / 400 + 1);
    let pcm16: Vec<i16> = pcm.iter().map(|&x| (x * 32768.0).round() as i16).collect();
    for d in FrameDuration::ALL {
        let mut c = CEncoder::new(fs, channels, app.0);
        c.bitrate(64_000).vbr(!cbr).complexity(5).set(SET, d.to_raw());
        let mut rs = OpusEncoder::new(fs as i32, channels, app.1).unwrap();
        rs.set_bitrate(Bitrate::BitsPerSecond(64_000)).unwrap();
        rs.set_vbr(!cbr);
        rs.set_complexity(5).unwrap();
        rs.set_expert_frame_duration(d);
        let mut buf = vec![0u8; 4000];
        let (mut accepted, mut refused) = (0, 0);
        for &size in &sizes {
            let label = format!("{fs} Hz {channels} ch app {} cbr {cbr} i16 {i16_input}, {d:?}, frame size {size}", app.0);
            let (rs_result, c_result) = if i16_input {
                (rs.encode_i16(&pcm16, size, &mut buf), c.try_encode_i16(&pcm16, size, 4000))
            } else {
                (rs.encode(&pcm, size, &mut buf), c.try_encode(&pcm, size, 4000))
            };
            match (rs_result, c_result) {
                (Ok(n), Ok(c_packet)) => {
                    let rs_samples = packet::nb_samples(&buf[..n], fs as i32).unwrap();
                    let c_samples = packet::nb_samples(&c_packet, fs as i32).unwrap();
                    assert_eq!(rs_samples, c_samples, "{label}");
                    let want = d.samples(fs as i32).unwrap_or(size);
                    assert_eq!(rs_samples, want, "{label}");
                    if same_bytes {
                        assert_eq!(&buf[..n], &c_packet[..], "{label}");
                    }
                    accepted += 1;
                }
                (Err(_), Err(e)) => {
                    assert_eq!(e, sys::OPUS_BAD_ARG, "{label}");
                    refused += 1;
                }
                (rs_result, c_result) => panic!("{label}: opus1 {rs_result:?}, libopus {:?}", c_result.map(|p| p.len())),
            }
        }
        assert!(accepted > 0 && refused > 0, "{fs} Hz {d:?}: {accepted} accepted, {refused} refused");
    }
}

#[test]
#[cfg_attr(not(target_arch = "aarch64"), ignore = "bit-exact with libopus only on 64-bit ARM (docs/10, Platforms)")]
fn the_same_frames_are_coded_as_by_libopus() {
    for fs in [8000, 12000, 16000, 24000, 48000] {
        for channels in [1, 2] {
            // CELT-only with CBR at complexity 5: libopus's packets.
            check_calls(fs, channels, (APP_LOWDELAY, Application::RestrictedLowDelay), true, true, false);
            check_calls(fs, channels, (APP_LOWDELAY, Application::RestrictedLowDelay), true, true, true);
            // The other applications, with VBR: the same calls taken and
            // the same durations coded.
            check_calls(fs, channels, (APP_VOIP, Application::Voip), false, false, false);
            check_calls(fs, channels, (APP_AUDIO, Application::Audio), false, false, true);
        }
    }
}

#[test]
fn input_must_hold_the_frame_size_passed() {
    // The caller passes `frame_size` samples per channel; libopus reads
    // them all for its analysis. opus1 asks for them too, even when it
    // codes fewer.
    let mut rs = OpusEncoder::new(48000, 2, Application::Audio).unwrap();
    rs.set_expert_frame_duration(FrameDuration::Ms20);
    let pcm = vec![0.1f32; 2 * 2880];
    let mut buf = vec![0u8; 4000];
    assert!(rs.encode(&pcm[..2 * 2880 - 1], 2880, &mut buf).is_err());
    let n = rs.encode(&pcm, 2880, &mut buf).unwrap();
    assert_eq!(packet::nb_samples(&buf[..n], 48000), Ok(960));
    assert!(rs.encode_i16(&vec![0i16; 2 * 2880 - 1], 2880, &mut buf).is_err());
    // Less than the duration that is set is refused.
    assert!(rs.encode(&pcm, 480, &mut buf).is_err());
}

#[test]
fn the_setting_stays_across_a_reset() {
    let mut c = CEncoder::new(16000, 1, APP_VOIP);
    c.set(SET, sys::OPUS_FRAMESIZE_100_MS);
    c.reset();
    let mut rs = OpusEncoder::new(16000, 1, Application::Voip).unwrap();
    rs.set_expert_frame_duration(FrameDuration::Ms100);
    rs.reset_state();
    assert_eq!(c.get(GET), sys::OPUS_FRAMESIZE_100_MS);
    assert_eq!(rs.expert_frame_duration(), FrameDuration::Ms100);
}
