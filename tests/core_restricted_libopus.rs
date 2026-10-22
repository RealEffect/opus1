//! The applications `RestrictedSilk` and `RestrictedCelt` of libopus 1.6
//! (`OPUS_APPLICATION_RESTRICTED_SILK`, 2052, and
//! `OPUS_APPLICATION_RESTRICTED_CELT`, 2053), against libopus 1.6.1.
//!
//! - `RestrictedCelt` codes CELT-only packets. opus1's CELT encoder is
//!   libopus's byte for byte, so the packets and the final ranges are
//!   compared byte for byte (on 64-bit ARM, see docs/10, Platforms). They
//!   are also the packets of `RestrictedLowDelay`.
//! - `RestrictedSilk` codes SILK-only packets. opus1's SILK encoder is a
//!   port of the fixed-point SILK encoder, and the linked libopus is the
//!   float build, so the bytes after the first one differ (docs/10, SILK).
//!   The tests compare what the two encoders must share: the configuration
//!   and the stereo flag of every packet (its first byte without the frame
//!   count code), the size of every constant-bitrate packet, the state a
//!   decoder of the packet ends in (the final range), and the quality.
//! - For both: the frame sizes an encode call takes, the application
//!   change, the look-ahead, and the other controls whose answer depends on
//!   the application.

mod core_common;
mod k24_common;

use std::num::NonZeroU32;
use audiopus_sys as sys;
use core_common::*;
use k24_common::{multitone, speech, speech_fullband, to_stereo};
use opus1::{
    Application, Bandwidth, Bitrate, Channels, ErrorKind, ForcedMode, FrameDuration, OpusDecoder, OpusEncoder,
    SampleRate,
};

/// libopus `OPUS_APPLICATION_RESTRICTED_SILK` and
/// `OPUS_APPLICATION_RESTRICTED_CELT`, which `audiopus_sys` does not name.
const APP_SILK: i32 = 2052;
const APP_CELT: i32 = 2053;

const RATES: [usize; 5] = [8000, 12000, 16000, 24000, 48000];

/// Every application, with its libopus number.
const APPS: [(i32, Application); 5] = [
    (APP_VOIP, Application::Voip),
    (APP_AUDIO, Application::Audio),
    (APP_LOWDELAY, Application::RestrictedLowDelay),
    (APP_SILK, Application::RestrictedSilk),
    (APP_CELT, Application::RestrictedCelt),
];

fn encoder(rate: usize, channels: usize, app: Application) -> OpusEncoder {
    OpusEncoder::new(SampleRate::try_from(rate as i32).unwrap(), Channels::try_from(channels).unwrap(), app)
        .unwrap()
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

/// Speech with energy up to the Nyquist frequency, then music, `seconds`
/// in all, in `channels` channels.
fn signal(rate: usize, channels: usize, seconds: f64) -> Vec<f32> {
    let half = seconds / 2.0;
    let mut mono = speech_fullband(rate, half);
    mono.extend(multitone(rate, (half * rate as f64) as usize));
    if channels == 2 { to_stereo(&mono, rate) } else { mono }
}

/// The frame sizes Opus has at `rate`: 2.5, 5, 10, 20, 40, 60, 80, 100
/// and 120 ms.
fn frame_sizes(rate: usize) -> [usize; 9] {
    [1, 2, 4, 8, 16, 24, 32, 40, 48].map(|k| k * rate / 400)
}

/// The first byte of a packet names a SILK-only configuration (0 to 11)
/// at narrowband, mediumband or wideband: every configuration below 12.
fn is_silk_only(toc: u8) -> bool {
    toc >> 3 < 12
}

// --- Creation, application change, look-ahead ---

/// Both applications exist at every sampling rate and channel count, with
/// libopus's application number, look-ahead and start bandwidth.
#[test]
fn restricted_applications_are_created_like_libopus() {
    for (c_app, app) in [(APP_SILK, Application::RestrictedSilk), (APP_CELT, Application::RestrictedCelt)] {
        assert_eq!(Application::try_from(c_app), Ok(app));
        assert_eq!(app.to_raw(), c_app);
        for rate in RATES {
            for channels in [1, 2] {
                let label = format!("{app:?}, {rate} Hz, {channels} channels");
                let mut c = CEncoder::new(rate, channels, c_app);
                let rust = encoder(rate, channels, app);
                assert_eq!(rust.application().to_raw(), c.get(sys::OPUS_GET_APPLICATION_REQUEST), "{label}");
                assert_eq!(rust.lookahead() as i32, c.lookahead(), "{label}: look-ahead");
                assert_eq!(rust.bandwidth().to_raw(), c.get(sys::OPUS_GET_BANDWIDTH_REQUEST), "{label}");
            }
        }
    }
}

/// `set_application` refuses a change to `RestrictedSilk` or
/// `RestrictedCelt`, and any call on an encoder created with one of them,
/// before the first frame, after it and after a reset, as libopus does.
/// The other applications change as before.
#[test]
fn set_application_refuses_the_restricted_applications_like_libopus() {
    let rate = 48_000;
    let frame = rate / 50;
    let pcm = signal(rate, 1, 0.2);
    for (c_from, from) in APPS {
        for (c_to, to) in APPS {
            let label = format!("{from:?} to {to:?}");
            let mut c = CEncoder::new(rate, 1, c_from);
            let mut rust = encoder(rate, 1, from);
            // Below complexity 7 no speech and music analysis runs.
            c.complexity(5);
            rust.set_complexity(5).unwrap();
            let check = |rust: &mut OpusEncoder, c: &mut CEncoder, when: &str| {
                let c_ok = c.try_set(sys::OPUS_SET_APPLICATION_REQUEST, c_to) == 0;
                let result = rust.set_application(to);
                assert_eq!(result.is_ok(), c_ok, "{label}, {when}");
                if let Err(e) = result {
                    assert_eq!(e.kind(), ErrorKind::BadArg, "{label}, {when}");
                }
                assert_eq!(rust.application().to_raw(), c.get(sys::OPUS_GET_APPLICATION_REQUEST), "{label}, {when}");
                assert_eq!(rust.lookahead() as i32, c.lookahead(), "{label}, {when}: look-ahead");
            };
            check(&mut rust, &mut c, "before the first frame");
            let mut out = vec![0u8; 1500];
            rust.encode(&pcm[..frame], frame, &mut out).unwrap();
            let c_packet = c.encode(&pcm[..frame]);
            assert_eq!(out[0] >> 3 >= 16, c_packet[0] >> 3 >= 16, "{label}: CELT-only or not");
            check(&mut rust, &mut c, "after a frame");
            rust.reset_state();
            c.reset();
            check(&mut rust, &mut c, "after a reset");
        }
    }
}

// --- Frame sizes ---

/// A `RestrictedSilk` encoder refuses frames of 2.5 and 5 ms, also when
/// they come from the expert frame duration, with `OPUS_BAD_ARG` as
/// libopus does. `RestrictedCelt` takes every frame size. The
/// configuration and stereo flag of every packet are libopus's.
#[test]
fn frame_sizes_are_taken_and_refused_like_libopus() {
    for (c_app, app) in [(APP_SILK, Application::RestrictedSilk), (APP_CELT, Application::RestrictedCelt)] {
        for rate in RATES {
            for channels in [1, 2] {
                let label = format!("{app:?}, {rate} Hz, {channels} channels");
                let pcm = signal(rate, channels, 0.5);
                let mut c = CEncoder::new(rate, channels, c_app);
                c.complexity(5).bitrate(32_000 * channels as i32);
                let mut rust = encoder(rate, channels, app);
                rust.set_complexity(5).unwrap();
                rust.set_bitrate(bps(32_000 * channels as i32));
                let mut out = vec![0u8; 1500 * 6];
                // The sizes Opus has, then sizes it does not have: 7.5 ms,
                // a sample less than 2.5 ms, and a sample more than 10 ms.
                let sizes = frame_sizes(rate).into_iter().chain([3 * rate / 400, rate / 400 - 1, rate / 100 + 1]);
                let mut pos = 0;
                for size in sizes {
                    let label = format!("{label}, frame of {size} samples");
                    let input = &pcm[pos..pos + size * channels];
                    let c_result = c.try_encode(input, size, out.len());
                    let result = rust.encode(input, size, &mut out);
                    match (&result, &c_result) {
                        (Ok(n), Ok(c_packet)) => {
                            let config = config_and_stereo(out[0]);
                            assert_eq!(config, config_and_stereo(c_packet[0]), "{label}: configuration");
                            assert!(*n >= 1);
                            if app == Application::RestrictedSilk {
                                assert!(is_silk_only(out[0]), "{label}: SILK-only");
                            } else {
                                assert!(out[0] >> 3 >= 16, "{label}: CELT-only");
                            }
                        }
                        (Err(e), Err(c_err)) => {
                            assert_eq!(*c_err, sys::OPUS_BAD_ARG, "{label}");
                            assert_eq!(e.kind(), ErrorKind::BadArg, "{label}");
                        }
                        _ => panic!("{label}: opus1 {:?}, libopus {:?}", result.map(|_| ()), c_result.map(|_| ())),
                    }
                    let silk_refuses = app == Application::RestrictedSilk && size < rate / 100;
                    assert_eq!(result.is_err(), silk_refuses || !frame_sizes(rate).contains(&size), "{label}");
                    pos += size * channels;
                }
                // The expert frame duration, with calls of 120 ms.
                let call = 3 * rate / 25;
                for duration in FrameDuration::ALL {
                    let label = format!("{label}, {duration:?}");
                    c.set(sys::OPUS_SET_EXPERT_FRAME_DURATION_REQUEST, duration.to_raw());
                    rust.set_expert_frame_duration(duration);
                    let input = &pcm[..call * channels];
                    let c_result = c.try_encode(input, call, out.len());
                    let result = rust.encode(input, call, &mut out);
                    assert_eq!(result.is_ok(), c_result.is_ok(), "{label}");
                    if let (Ok(n), Ok(c_packet)) = (result, &c_result) {
                        let config = config_and_stereo(out[0]);
                        assert_eq!(config, config_and_stereo(c_packet[0]), "{label}: configuration");
                        let samples = opus1::packet::nb_samples(&out[..n], SampleRate::try_from(rate as i32).unwrap());
                        assert_eq!(samples, Ok(duration.samples(SampleRate::try_from(rate as i32).unwrap()).unwrap_or(call)), "{label}");
                    }
                    let too_short = matches!(duration, FrameDuration::Ms2_5 | FrameDuration::Ms5);
                    assert_eq!(result.is_err(), app == Application::RestrictedSilk && too_short, "{label}");
                }
            }
        }
    }
}

// --- Packets without audio ---

/// When the bitrate or the output buffer is too small to code a frame, the
/// encoder writes a packet without audio (libopus's "PLC" packet). Its TOC
/// byte names the mode of the last coded frame, and before the first coded
/// frame and after a reset libopus's initial mode, Hybrid at fullband, also
/// for the restricted applications (the documentation of `Application`
/// says so). Every such packet is libopus's byte for byte: at 500 b/s with
/// a variable and a constant bitrate, and into a buffer of 2 bytes before
/// a coded frame, after one and after a reset.
#[test]
fn packets_without_audio_are_those_of_libopus() {
    let restricted = [
        (APP_LOWDELAY, Application::RestrictedLowDelay),
        (APP_SILK, Application::RestrictedSilk),
        (APP_CELT, Application::RestrictedCelt),
    ];
    for (c_app, app) in restricted {
        // Packets whose TOC byte names Hybrid.
        let mut hybrid_tocs = 0;
        for rate in [16_000, 48_000] {
            for channels in [1, 2] {
                let pcm = signal(rate, channels, 0.5);
                for frame in frame_sizes(rate) {
                    if app == Application::RestrictedSilk && frame < rate / 100 {
                        continue;
                    }
                    let input = &pcm[..frame * channels];
                    let label = format!("{app:?}, {rate} Hz, {channels} channels, frame {frame}");
                    let mut out = vec![0u8; 1500];
                    // A bitrate too low to code any frame.
                    for vbr in [true, false] {
                        let label = format!("{label}, 500 b/s, VBR {vbr}");
                        let mut c = CEncoder::new(rate, channels, c_app);
                        c.bitrate(500).vbr(vbr);
                        let mut rust = encoder(rate, channels, app);
                        rust.set_bitrate(bps(500));
                        rust.set_vbr(vbr);
                        for i in 0..3 {
                            let c_packet = c.try_encode(input, frame, 1500).unwrap();
                            let n = rust.encode(input, frame, &mut out).unwrap();
                            assert_eq!(&out[..n], &c_packet[..], "{label}: packet {i}");
                            hybrid_tocs += usize::from(!is_silk_only(out[0]) && out[0] >> 3 < 16);
                        }
                    }
                    // A buffer of 2 bytes: before the first coded frame,
                    // after a coded frame, and after a reset.
                    let mut c = CEncoder::new(rate, channels, c_app);
                    let mut rust = encoder(rate, channels, app);
                    for when in ["new", "coded", "reset"] {
                        let label = format!("{label}, 2 bytes, {when}");
                        match when {
                            "coded" => {
                                let c_packet = c.try_encode(input, frame, 1500).unwrap();
                                rust.encode(input, frame, &mut out).unwrap();
                                assert_eq!(config_and_stereo(out[0]), config_and_stereo(c_packet[0]), "{label}");
                            }
                            "reset" => {
                                c.reset();
                                rust.reset_state();
                            }
                            _ => {}
                        }
                        let c_packet = c.try_encode(input, frame, 2);
                        let packet = rust.encode(input, frame, &mut out[..2]).map(|n| out[..n].to_vec());
                        assert_eq!(packet.map_err(|e| e.to_raw()), c_packet, "{label}");
                    }
                }
            }
        }
        // The Hybrid TOC byte of the initial state does occur.
        assert!(hybrid_tocs > 0, "{app:?}: no Hybrid TOC byte");
    }
}

// --- RestrictedCelt: packets byte for byte ---

/// One stream of `RestrictedCelt` packets against libopus's: every packet,
/// byte for byte, and every final range. The packets are also those of an
/// opus1 `RestrictedLowDelay` encoder with the same settings. Returns the
/// number of packets.
#[cfg(feature = "analysis")]
#[allow(clippy::too_many_arguments)]
fn check_celt_stream(
    rate: usize,
    channels: usize,
    pcm: &[f32],
    frame: usize,
    bitrate: i32,
    complexity: i32,
    vbr: bool,
    constrained: bool,
) -> usize {
    let label = format!(
        "{rate} Hz, {channels} ch, {frame} samples, {bitrate} b/s, complexity {complexity}, vbr {vbr}, constrained {constrained}"
    );
    let mut c = CEncoder::new(rate, channels, APP_CELT);
    c.bitrate(bitrate).complexity(complexity).vbr(vbr);
    c.set(sys::OPUS_SET_VBR_CONSTRAINT_REQUEST, constrained as i32);
    let mut rust = encoder(rate, channels, Application::RestrictedCelt);
    let mut low_delay = encoder(rate, channels, Application::RestrictedLowDelay);
    for enc in [&mut rust, &mut low_delay] {
        enc.set_bitrate(bps(bitrate));
        enc.set_complexity(complexity as u8).unwrap();
        enc.set_vbr(vbr);
        enc.set_vbr_constraint(constrained);
    }
    let mut out = vec![0u8; 1275 * 6];
    let mut out_low_delay = vec![0u8; 1275 * 6];
    let mut packets = 0;
    for (i, f) in pcm.chunks_exact(frame * channels).enumerate() {
        let label = format!("{label}: packet {i}");
        let c_packet = c.encode_max(f, out.len());
        let n = rust.encode(f, frame, &mut out).unwrap();
        same_bytes(&out[..n], &c_packet, &label);
        assert_eq!(rust.final_range(), c.final_range(), "{label}: final range");
        let m = low_delay.encode(f, frame, &mut out_low_delay).unwrap();
        assert!(out[..n] == out_low_delay[..m], "{label}: the RestrictedLowDelay packet differs");
        packets += 1;
    }
    packets
}

/// `RestrictedCelt` against libopus at every sampling rate, mono and
/// stereo, frames of 2.5 to 120 ms, low to high bitrates and complexities
/// 0, 5 and 10. Frames of up to 20 ms are coded with VBR, constrained VBR
/// and CBR. Longer packets hold several 20 ms frames, which opus1 codes
/// without libopus's limit of each frame under VBR on purpose (docs/10,
/// Deliberate differences), so they are compared with CBR.
///
/// At complexity 10 the speech and music analysis runs, so the test needs
/// the feature `analysis`.
#[test]
#[cfg(feature = "analysis")]
#[cfg_attr(not(target_arch = "aarch64"), ignore = "bit-exact with libopus only on 64-bit ARM (docs/10, Platforms)")]
fn restricted_celt_packets_and_final_ranges_are_libopus() {
    // A debug build codes a part of the matrix.
    let debug = cfg!(debug_assertions);
    let rates: &[usize] = if debug { &[16000, 48000] } else { &RATES };
    let seconds = if debug { 0.25 } else { 0.6 };
    let mut packets = 0;
    for &rate in rates {
        for channels in [1usize, 2] {
            let pcm = signal(rate, channels, seconds);
            // Bitrate and complexity: low, middle and high.
            let settings = [(12_000, 10), (32_000, 0), (32_000, 5), (96_000, 10)]
                .map(|(b, c)| (b * channels as i32, c));
            for frame in frame_sizes(rate) {
                let short = frame <= rate / 50;
                for (bitrate, complexity) in settings {
                    packets += check_celt_stream(rate, channels, &pcm, frame, bitrate, complexity, false, false);
                    if short {
                        packets += check_celt_stream(rate, channels, &pcm, frame, bitrate, complexity, true, true);
                        packets += check_celt_stream(rate, channels, &pcm, frame, bitrate, complexity, true, false);
                    }
                }
            }
        }
    }
    println!("{packets} packets, all equal to libopus's");
}

// --- RestrictedSilk: what the SILK encoders share ---

/// What one stream of `RestrictedSilk` packets gave.
struct SilkStream {
    packets: usize,
    /// The packets whose configuration or stereo flag is not libopus's.
    configs_differ: usize,
    /// SNR of opus1's and of libopus's decoded stream against the input.
    snr: f64,
    c_snr: f64,
}

/// The configuration and the stereo flag of a packet: its first byte
/// without the frame count code. In a packet of several frames that code
/// says whether the frames have the same size, which depends on the SILK
/// bytes.
fn config_and_stereo(toc: u8) -> u8 {
    toc & 0xFC
}

/// SNR of the decoded signal against the input, which it lags by
/// `delay` samples per channel.
fn snr_against_input(input: &[f32], decoded: &[f32], delay: usize, channels: usize) -> f64 {
    let lag = delay * channels;
    let n = input.len().min(decoded.len().saturating_sub(lag));
    snr(&input[..n], &decoded[lag..lag + n])
}

/// One stream of `RestrictedSilk` packets against libopus's: the
/// configuration and stereo flag of every packet, the size of every packet
/// with CBR, and the final range. opus1's final range must be the one libopus's decoder ends in
/// for opus1's packet, and opus1's decoder must end in it too. Both streams
/// are decoded by libopus and measured against the input.
#[allow(clippy::too_many_arguments)]
fn check_silk_stream(
    rate: usize,
    channels: usize,
    pcm: &[f32],
    frame: usize,
    bitrate: i32,
    complexity: i32,
    vbr: bool,
) -> SilkStream {
    let label = format!("{rate} Hz, {channels} ch, {frame} samples, {bitrate} b/s, complexity {complexity}, vbr {vbr}");
    let mut c = CEncoder::new(rate, channels, APP_SILK);
    c.bitrate(bitrate).complexity(complexity).vbr(vbr);
    let mut rust = encoder(rate, channels, Application::RestrictedSilk);
    rust.set_bitrate(bps(bitrate));
    rust.set_complexity(complexity as u8).unwrap();
    rust.set_vbr(vbr);
    let mut c_dec = CDecoder::new(rate, channels);
    let mut c_dec_own = CDecoder::new(rate, channels);
    let mut rust_dec = OpusDecoder::new(SampleRate::try_from(rate as i32).unwrap(), Channels::try_from(channels).unwrap()).unwrap();
    let mut out = vec![0u8; 1275 * 3];
    let mut decoded = vec![0f32; frame * channels];
    let (mut ours, mut theirs) = (Vec::new(), Vec::new());
    let mut stream = SilkStream { packets: 0, configs_differ: 0, snr: 0.0, c_snr: 0.0 };
    for (i, f) in pcm.chunks_exact(frame * channels).enumerate() {
        let label = format!("{label}: packet {i}");
        let c_packet = c.encode_max(f, out.len());
        let n = rust.encode(f, frame, &mut out).unwrap();
        let packet = &out[..n];
        assert!(is_silk_only(packet[0]), "{label}: not SILK-only: {:#04x}", packet[0]);
        if config_and_stereo(packet[0]) != config_and_stereo(c_packet[0]) {
            stream.configs_differ += 1;
        }
        if !vbr {
            assert_eq!(n, c_packet.len(), "{label}: CBR size");
        }
        ours.extend(c_dec.decode(Some(packet), frame, false).unwrap());
        assert_eq!(rust.final_range(), c_dec.final_range(), "{label}: final range");
        let m = rust_dec.decode(packet, frame, &mut decoded).unwrap();
        assert_eq!(m, frame, "{label}");
        assert_eq!(rust_dec.final_range(), c_dec.final_range(), "{label}: final range of opus1's decoder");
        theirs.extend(c_dec_own.decode(Some(&c_packet), frame, false).unwrap());
        assert_eq!(c.final_range(), c_dec_own.final_range(), "{label}: libopus's own final range");
        stream.packets += 1;
    }
    let delay = rust.lookahead();
    stream.snr = snr_against_input(pcm, &ours, delay, channels);
    stream.c_snr = snr_against_input(pcm, &theirs, delay, channels);
    stream
}

/// `RestrictedSilk` against libopus at every sampling rate, mono and
/// stereo, frames of 10 to 120 ms, low to high bitrates, complexities 0, 5
/// and 10, VBR and CBR.
///
/// Every packet has libopus's configuration (mode, bandwidth, frame
/// duration) and stereo flag, and with CBR libopus's size. The SNR of the
/// decoded speech is on average 0.2 dB below libopus's, at worst 3 dB
/// (12 kHz stereo, 60 ms, 32 kb/s CBR). That is the difference between
/// opus1's SILK encoder, a port of the fixed-point one, and the float one
/// of the linked libopus: opus1's `Audio` encoder forced to SILK-only
/// wideband gives the same packets and the same SNR (see
/// `restricted_silk_codes_as_forced_silk_audio_where_libopus_does`).
#[test]
fn restricted_silk_packets_follow_libopus() {
    let debug = cfg!(debug_assertions);
    let rates: &[usize] = if debug { &[16000, 48000] } else { &RATES };
    let seconds = if debug { 0.5 } else { 2.4 };
    let (mut packets, mut differ, mut streams) = (0, 0, 0);
    let (mut sum, mut c_sum) = (0.0, 0.0);
    let mut worst = (f64::INFINITY, String::new());
    for &rate in rates {
        for channels in [1usize, 2] {
            let mono = speech(rate, seconds);
            let pcm = if channels == 2 { to_stereo(&mono, rate) } else { mono };
            let settings = [(8_000, 10), (16_000, 0), (16_000, 5), (32_000, 10)]
                .map(|(b, c)| (b * channels as i32, c));
            for frame in frame_sizes(rate).into_iter().skip(2) {
                for (bitrate, complexity) in settings {
                    for vbr in [true, false] {
                        let s = check_silk_stream(rate, channels, &pcm, frame, bitrate, complexity, vbr);
                        packets += s.packets;
                        differ += s.configs_differ;
                        streams += 1;
                        sum += s.snr;
                        c_sum += s.c_snr;
                        let deficit = s.snr - s.c_snr;
                        if deficit < worst.0 {
                            worst = (deficit, format!("{rate} Hz, {channels} ch, {frame} samples, {bitrate} b/s, c{complexity}, vbr {vbr}: {:.2} against {:.2} dB", s.snr, s.c_snr));
                        }
                    }
                }
            }
        }
    }
    let (mean, c_mean) = (sum / streams as f64, c_sum / streams as f64);
    println!(
        "{streams} streams, {packets} packets, {differ} configurations differ; mean SNR {mean:.2} dB, libopus {c_mean:.2} dB; worst {:.2} dB ({})",
        worst.0, worst.1
    );
    assert_eq!(differ, 0, "packets whose configuration or stereo flag is not libopus's");
    assert!(mean > c_mean - 0.5, "mean SNR {mean:.2} dB, libopus {c_mean:.2} dB");
    assert!(worst.0 > -4.0, "worst stream: {:.2} dB ({})", worst.0, worst.1);
}

/// libopus codes `RestrictedSilk` as `Audio` forced to SILK-only with at
/// most wideband, as long as nothing that only one of them does happens.
/// Without the speech and music analysis (complexity 0 to 6) the first
/// such event is a change of SILK's internal rate: `Audio` adds a
/// redundant CELT frame there, `RestrictedSilk` does not.
///
/// This test runs both setups in libopus and in opus1, with a bitrate
/// that drops to a quarter and comes back every 250 ms, so that SILK
/// changes its rate. In every stream the first packet where the two setups
/// differ is the same in opus1 as in libopus, or there is none in both.
/// Packets of 80 to 120 ms are coded with CBR only: with VBR opus1 leaves
/// out libopus's limit of each of their frames on purpose (docs/10,
/// Deliberate differences), which changes SILK's internal rate at low
/// bitrates. Every packet of opus1's `RestrictedSilk` stream also has the
/// configuration and stereo flag of libopus's.
#[test]
fn restricted_silk_codes_as_forced_silk_audio_where_libopus_does() {
    let debug = cfg!(debug_assertions);
    let rates: &[usize] = if debug { &[16000, 48000] } else { &RATES };
    let seconds = if debug { 1.0 } else { 2.4 };
    let (mut streams, mut diverging) = (0, 0);
    for &rate in rates {
        for channels in [1usize, 2] {
            let mono = speech(rate, seconds);
            let pcm = if channels == 2 { to_stereo(&mono, rate) } else { mono };
            for frame in frame_sizes(rate).into_iter().skip(2) {
                for (bitrate, complexity) in [(8_000, 0), (16_000, 5), (32_000, 3), (64_000, 6)] {
                    let bitrate = bitrate * channels as i32;
                    for vbr in [true, false] {
                        if vbr && frame > 3 * rate / 50 {
                            continue;
                        }
                        let label = format!("{rate} Hz, {channels} ch, {frame} samples, {bitrate} b/s, c{complexity}, vbr {vbr}");
                        let mut c_silk = CEncoder::new(rate, channels, APP_SILK);
                        let mut c_audio = CEncoder::new(rate, channels, APP_AUDIO);
                        for c in [&mut c_silk, &mut c_audio] {
                            c.complexity(complexity).vbr(vbr);
                        }
                        c_audio.force_mode(MODE_SILK).max_bandwidth(BW_WB);
                        let mut silk = encoder(rate, channels, Application::RestrictedSilk);
                        let mut audio = encoder(rate, channels, Application::Audio);
                        for enc in [&mut silk, &mut audio] {
                            enc.set_complexity(complexity as u8).unwrap();
                            enc.set_vbr(vbr);
                        }
                        audio.set_force_mode(Some(ForcedMode::SilkOnly));
                        audio.set_max_bandwidth(Bandwidth::Wideband);
                        let (mut out_silk, mut out_audio) = (vec![0u8; 1275 * 6], vec![0u8; 1275 * 6]);
                        let (mut c_first, mut first) = (None, None);
                        for (i, f) in pcm.chunks_exact(frame * channels).enumerate() {
                            let quarter = (i * frame * 4 / rate) % 2 == 1;
                            let b = if quarter { (bitrate / 4).max(6000) } else { bitrate };
                            for c in [&mut c_silk, &mut c_audio] {
                                c.bitrate(b);
                            }
                            for enc in [&mut silk, &mut audio] {
                                enc.set_bitrate(bps(b));
                            }
                            let c_a = c_silk.encode_max(f, out_silk.len());
                            let c_b = c_audio.encode_max(f, out_audio.len());
                            let n = silk.encode(f, frame, &mut out_silk).unwrap();
                            let m = audio.encode(f, frame, &mut out_audio).unwrap();
                            assert_eq!(
                                config_and_stereo(out_silk[0]),
                                config_and_stereo(c_a[0]),
                                "{label}: configuration of packet {i}"
                            );
                            if c_first.is_none() && c_a != c_b {
                                c_first = Some(i);
                            }
                            if first.is_none() && out_silk[..n] != out_audio[..m] {
                                first = Some(i);
                            }
                        }
                        assert_eq!(first, c_first, "{label}: first packet where the setups differ");
                        streams += 1;
                        diverging += first.is_some() as usize;
                    }
                }
            }
        }
    }
    println!("{streams} streams, {diverging} with a packet where the setups differ");
    // Both cases occur.
    assert!(diverging > 0 && diverging < streams);
}

// --- Controls whose answer depends on the application ---

/// Gives an opus1 encoder and a libopus encoder the same setting.
#[cfg(feature = "analysis")]
type Setup = fn(&mut OpusEncoder, &mut CEncoder);

/// A `RestrictedSilk` encoder codes SILK-only and at most wideband, also
/// when CELT-only, Hybrid, fullband or LFE is asked for. It ignores phase
/// inversion disabling and runs no speech and music analysis (the voice
/// ratio stays unknown at complexity 10). A `RestrictedCelt` encoder codes
/// CELT-only when SILK is asked for. The first byte of every packet and
/// the getters are libopus's. At complexity 10 the `RestrictedCelt`
/// encoder runs the speech and music analysis, so the test needs the
/// feature `analysis`.
#[test]
#[cfg(feature = "analysis")]
fn controls_of_the_restricted_applications_follow_libopus() {
    let rate = 48_000;
    let frame = rate / 50;
    for channels in [1usize, 2] {
        let pcm = signal(rate, channels, 1.0);
        let cases: [(&str, Setup); 6] = [
            ("fullband", |r, c| {
                r.set_bandwidth(Some(Bandwidth::Fullband));
                c.bandwidth(BW_FB);
            }),
            ("forced CELT-only", |r, c| {
                r.set_force_mode(Some(ForcedMode::CeltOnly));
                c.force_mode(MODE_CELT);
            }),
            ("forced SILK-only", |r, c| {
                r.set_force_mode(Some(ForcedMode::SilkOnly));
                c.force_mode(MODE_SILK);
            }),
            ("forced Hybrid", |r, c| {
                r.set_force_mode(Some(ForcedMode::Hybrid));
                c.force_mode(MODE_HYBRID);
            }),
            ("LFE", |r, c| {
                r.set_lfe(true);
                c.lfe(true);
            }),
            ("phase inversion disabled", |r, c| {
                r.set_phase_inversion_disabled(true);
                c.set(sys::OPUS_SET_PHASE_INVERSION_DISABLED_REQUEST, 1);
            }),
        ];
        for (c_app, app) in [(APP_SILK, Application::RestrictedSilk), (APP_CELT, Application::RestrictedCelt)] {
            for (name, setup) in cases {
                let label = format!("{app:?}, {channels} channels, {name}");
                let mut c = CEncoder::new(rate, channels, c_app);
                c.bitrate(64_000).complexity(10);
                let mut rust = encoder(rate, channels, app);
                rust.set_bitrate(bps(64_000));
                rust.set_complexity(10).unwrap();
                setup(&mut rust, &mut c);
                let get_inv = sys::OPUS_GET_PHASE_INVERSION_DISABLED_REQUEST;
                assert_eq!(rust.phase_inversion_disabled() as i32, c.get(get_inv), "{label}");
                let mut out = vec![0u8; 1500];
                for (i, f) in pcm.chunks_exact(frame * channels).enumerate() {
                    let label = format!("{label}: packet {i}");
                    let c_packet = c.encode(f);
                    rust.encode(f, frame, &mut out).unwrap();
                    assert_eq!(out[0], c_packet[0], "{label}: first byte");
                    assert_eq!(rust.bandwidth().to_raw(), c.get(sys::OPUS_GET_BANDWIDTH_REQUEST), "{label}: bandwidth");
                    let ratio = c.get(OPUS_GET_VOICE_RATIO_REQUEST);
                    assert_eq!(rust.voice_ratio().map_or(-1, i32::from), ratio, "{label}: voice ratio");
                    if app == Application::RestrictedSilk {
                        assert!(is_silk_only(out[0]), "{label}: SILK-only, at most wideband");
                        assert_eq!(ratio, -1, "{label}: no analysis");
                    } else {
                        assert!(out[0] >> 3 >= 16, "{label}: CELT-only");
                    }
                }
            }
        }
    }
}

