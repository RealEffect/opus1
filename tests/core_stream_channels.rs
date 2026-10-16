//! Stereo input coded as mono (opus_encoder.c): the choice of the channel
//! count in `opus_encode_native` (1428-1451), the packet that leads over to
//! mono (1560-1571), and what SILK and CELT do with stereo input that a
//! packet codes as one channel (enc_API.c:296-332, `CELT_SET_CHANNELS`).
//!
//! The reference is the C libopus encoder with the same settings, at
//! complexity 5, where libopus decides without the speech/music analysis
//! that opus1 does not have. Both streams are decoded by libopus.

mod core_common;
mod k24_common;

use core_common::*;
use k24_common::{multitone, speech, speech_fullband, to_stereo};
use opus1::{Application, OpusEncoder};

/// Stereo test inputs, interleaved.
#[derive(Clone, Copy, Debug)]
enum Source {
    /// The same speech in both channels, the right one 1 dB quieter: a
    /// stereo width near 0.
    MonoLike,
    /// Speech in the left channel and music in the right: a wide image.
    Wide,
    /// Speech and a delayed, quieter copy (k24_common::to_stereo).
    Delayed,
}

fn source(kind: Source, rate: usize, seconds: f64) -> Vec<f32> {
    let n = (rate as f64 * seconds) as usize;
    match kind {
        Source::MonoLike => {
            let x = speech_fullband(rate, seconds);
            x.iter().flat_map(|&v| [v, 0.89 * v]).collect()
        }
        Source::Wide => {
            let left = speech(rate, seconds);
            let right = multitone(rate, n);
            left.iter().zip(&right).flat_map(|(&l, &r)| [l, 0.5 * r]).collect()
        }
        Source::Delayed => to_stereo(&speech_fullband(rate, seconds), rate),
    }
}

fn snr(reference: &[f32], test: &[f32]) -> f64 {
    let (mut sig, mut err) = (0f64, 0f64);
    for (r, t) in reference.iter().zip(test) {
        sig += (*r as f64).powi(2);
        err += (*r as f64 - *t as f64).powi(2);
    }
    10.0 * (sig.max(1e-12) / err.max(1e-12)).log10()
}

struct Sweep {
    /// Channel count in the first byte of each packet, opus1 and libopus.
    channels: Vec<(usize, usize)>,
    /// Mode of each packet (0 SILK-only, 1 Hybrid, 2 CELT-only), opus1 and
    /// libopus.
    modes: Vec<(u8, u8)>,
    /// Configuration of each packet (mode, bandwidth and frame duration, the
    /// upper five bits of the first byte), opus1 and libopus.
    configs: Vec<(u8, u8)>,
    /// Signal-to-noise ratio of the two decoded streams against the input.
    snr: (f64, f64),
}

fn mode_of(toc: u8) -> u8 {
    if toc & 0x80 != 0 {
        2
    } else if toc & 0x60 == 0x60 {
        1
    } else {
        0
    }
}

/// Both encoders on `input` with a bitrate from `bitrate(packet)`, and
/// `setup` applied to each.
fn sweep(
    rate: usize,
    frame_ms_x10: usize,
    application: (i32, Application),
    cbr: bool,
    input: &[f32],
    bitrate: &dyn Fn(usize) -> i32,
    setup: &dyn Fn(&mut OpusEncoder, &mut CEncoder),
) -> Sweep {
    let frame = rate * frame_ms_x10 / 10000;
    let mut c_enc = CEncoder::new(rate, 2, application.0);
    c_enc.complexity(5).vbr(!cbr);
    let mut enc = OpusEncoder::new(rate as i32, 2, application.1).unwrap();
    enc.complexity = 5;
    enc.use_cbr = cbr;
    setup(&mut enc, &mut c_enc);
    let (mut dec, mut c_dec) = (CDecoder::new(rate, 2), CDecoder::new(rate, 2));
    let mut buf = vec![0u8; 4000];
    let (mut out, mut c_out) = (Vec::new(), Vec::new());
    let (mut channels, mut modes, mut configs) = (Vec::new(), Vec::new(), Vec::new());
    for (i, pcm) in input.chunks_exact(2 * frame).enumerate() {
        c_enc.bitrate(bitrate(i));
        enc.bitrate_bps = bitrate(i);
        let n = enc.encode(pcm, frame, &mut buf).unwrap_or_else(|e| panic!("packet {i}: {e}"));
        let c_packet = c_enc.encode(pcm);
        out.extend(dec.decode(Some(&buf[..n]), frame, false).unwrap());
        assert_eq!(enc.final_range(), dec.final_range(), "packet {i}: final range");
        c_out.extend(c_dec.decode(Some(&c_packet), frame, false).unwrap());
        channels.push((1 + (buf[0] >> 2 & 1) as usize, 1 + (c_packet[0] >> 2 & 1) as usize));
        modes.push((mode_of(buf[0]), mode_of(c_packet[0])));
        configs.push((buf[0] >> 3, c_packet[0] >> 3));
    }
    let lag = 2 * enc.lookahead();
    let n = out.len() - lag;
    Sweep { channels, modes, configs, snr: (snr(&input[..n], &out[lag..]), snr(&input[..n], &c_out[lag..])) }
}

/// Down from 40 to 8 kb/s and up again, 1 kb/s every 80 ms.
fn down_and_up(frame_ms_x10: usize) -> impl Fn(usize) -> i32 {
    move |i| {
        let step = (i * frame_ms_x10 / 800) % 64;
        8000 + 1000 * if step < 32 { 32 - step } else { step - 32 } as i32
    }
}

/// A bitrate sweep over three kinds of stereo input, at three sampling
/// rates and four packet durations, with a variable and a constant
/// bitrate: every packet codes the channel count libopus's codes, with the
/// mode and the bandwidth libopus's has. The streams decode within 1 dB of
/// libopus's.
///
/// What goes on in libopus, and now in opus1:
///
/// - Stereo input is coded as mono when the rate for two channels is under
///   17 kb/s for music and 19 kb/s for speech, with 1 kb/s of hysteresis.
/// - In SILK-only and Hybrid packets of 10 and 20 ms, the packet before the
///   first mono one still codes two channels, with SILK's side signal
///   faded out (`toMono`).
/// - CELT codes the mean of the two channels as one.
///
/// Before the port opus1 coded every packet in stereo: the channel count
/// differed in 30 to 60 % of the packets.
#[test]
fn a_bitrate_sweep_codes_the_channel_count_of_libopus() {
    const AUDIO: (i32, Application) = (APP_AUDIO, Application::Audio);
    const VOIP: (i32, Application) = (APP_VOIP, Application::Voip);
    for kind in [Source::MonoLike, Source::Wide, Source::Delayed] {
        for (rate, application) in [(48000usize, AUDIO), (16000, VOIP), (24000, AUDIO)] {
            let input = source(kind, rate, 10.5);
            for frame_ms_x10 in [100usize, 200, 400, 600] {
                for cbr in [false, true] {
                    let what = format!(
                        "{kind:?}, {rate} Hz, {:?}, {} ms, CBR {cbr}",
                        application.1,
                        frame_ms_x10 / 10
                    );
                    let s = sweep(rate, frame_ms_x10, application, cbr, &input, &down_and_up(frame_ms_x10), &|_, _| {});
                    let mono = s.channels.iter().filter(|c| c.1 == 1).count();
                    let differ: Vec<usize> =
                        (0..s.channels.len()).filter(|&i| s.channels[i].0 != s.channels[i].1).collect();
                    let modes_differ = s.modes.iter().filter(|m| m.0 != m.1).count();
                    if std::env::var("CORE_STREAM_CHANNELS_PRINT").is_ok() {
                        println!(
                            "{what}: {mono}/{} mono in libopus, channels differ at {differ:?}, modes differ {modes_differ}, SNR {:.2}/{:.2} dB",
                            s.channels.len(),
                            s.snr.0,
                            s.snr.1
                        );
                    }
                    assert!(mono * 10 >= s.channels.len(), "{what}: libopus codes {mono} packets as mono");
                    assert!(differ.is_empty(), "{what}: channel count differs at packets {differ:?}");
                    let configs_differ: Vec<usize> =
                        (0..s.configs.len()).filter(|&i| s.configs[i].0 != s.configs[i].1).collect();
                    assert!(configs_differ.is_empty(), "{what}: mode or bandwidth differs at packets {configs_differ:?}");
                    assert!(s.snr.0 >= s.snr.1 - 1.0, "{what}: {:.2} dB, libopus {:.2} dB", s.snr.0, s.snr.1);
                }
            }
        }
    }
}

/// CELT-only streams of stereo input: the restricted low delay application
/// and a forced CELT-only mode, down to 2.5 ms packets. CELT codes the mean
/// of the two channels when the packet is mono (`CELT_SET_CHANNELS`), and
/// there is no lead-over packet. Every packet codes libopus's channel
/// count, and the streams decode within 1 dB of libopus's.
#[test]
fn celt_only_streams_code_the_channel_count_of_libopus() {
    const LOWDELAY: (i32, Application) = (APP_LOWDELAY, Application::RestrictedLowDelay);
    const AUDIO: (i32, Application) = (APP_AUDIO, Application::Audio);
    let force_celt = |enc: &mut OpusEncoder, c_enc: &mut CEncoder| {
        enc.set_force_mode(Some(opus1::ForcedMode::CeltOnly));
        c_enc.force_mode(MODE_CELT);
    };
    for kind in [Source::MonoLike, Source::Wide, Source::Delayed] {
        for (rate, application, frame_ms_x10, forced) in [
            (48000usize, LOWDELAY, 200usize, false),
            (48000, LOWDELAY, 50, false),
            (48000, AUDIO, 25, false),
            (24000, AUDIO, 100, true),
            (16000, AUDIO, 200, true),
        ] {
            let input = source(kind, rate, 10.5);
            for cbr in [false, true] {
                let what = format!("{kind:?}, {rate} Hz, {:?}, {} ms, CBR {cbr}", application.1, frame_ms_x10 as f64 / 10.0);
                let setup: &dyn Fn(&mut OpusEncoder, &mut CEncoder) = if forced { &force_celt } else { &|_, _| {} };
                let s = sweep(rate, frame_ms_x10, application, cbr, &input, &down_and_up(frame_ms_x10), setup);
                let mono = s.channels.iter().filter(|c| c.1 == 1).count();
                let differ: Vec<usize> =
                    (0..s.channels.len()).filter(|&i| s.channels[i].0 != s.channels[i].1).collect();
                if std::env::var("CORE_STREAM_CHANNELS_PRINT").is_ok() {
                    println!("{what}: {mono}/{} mono in libopus, channels differ at {} packets, SNR {:.2}/{:.2} dB", s.channels.len(), differ.len(), s.snr.0, s.snr.1);
                }
                assert!(s.modes.iter().all(|m| *m == (2, 2)), "{what}: not CELT-only");
                assert!(mono * 10 >= s.channels.len(), "{what}: libopus codes {mono} packets as mono");
                assert!(differ.is_empty(), "{what}: channel count differs at packets {differ:?}");
                assert!(s.configs.iter().all(|c| c.0 == c.1), "{what}: bandwidth differs");
                assert!(s.snr.0 >= s.snr.1 - 1.0, "{what}: {:.2} dB, libopus {:.2} dB", s.snr.0, s.snr.1);
            }
        }
    }
}

/// A 40 or 60 ms Hybrid packet is coded as 20 ms frames. When it would be
/// the packet that leads over to mono, libopus codes it in stereo without
/// SILK's fade and sets its own `force_channels` to mono
/// (opus_encoder.c:1763-1765): from the next packet on the stream is mono,
/// whatever the bitrate, until the caller sets the channel count again.
/// opus1 does the same, and reports the setting.
#[test]
fn a_long_hybrid_packet_that_leads_over_to_mono_keeps_the_stream_mono_like_libopus() {
    let rate = 48000;
    for frame_ms in [40usize, 60] {
        let input = source(Source::Delayed, rate, 6.0);
        let frame = rate * frame_ms / 1000;
        let mut c_enc = CEncoder::new(rate, 2, APP_AUDIO);
        c_enc.complexity(5).force_mode(MODE_HYBRID).bandwidth(BW_FB);
        let mut enc = OpusEncoder::new(rate as i32, 2, Application::Audio).unwrap();
        enc.complexity = 5;
        enc.set_force_mode(Some(opus1::ForcedMode::Hybrid));
        enc.set_bandwidth(opus1::Bandwidth::Fullband);
        let mut buf = vec![0u8; 4000];
        let mut channels = Vec::new();
        for (i, pcm) in input.chunks_exact(2 * frame).enumerate() {
            // 40 kb/s for 1.2 s, then 12 kb/s for 1.2 s, then 64 kb/s.
            let ms = i * frame_ms;
            let bitrate = if ms < 1200 { 40000 } else if ms < 2400 { 12000 } else { 64000 };
            c_enc.bitrate(bitrate);
            enc.bitrate_bps = bitrate;
            if ms == 4800 {
                // The caller lets the bitrate decide again.
                c_enc.force_channels(-1000);
                enc.set_force_channels(None).unwrap();
            }
            enc.encode(pcm, frame, &mut buf).unwrap();
            let c_packet = c_enc.encode(pcm);
            assert_eq!(mode_of(buf[0]), mode_of(c_packet[0]), "{frame_ms} ms, packet {i}: mode");
            channels.push((1 + (buf[0] >> 2 & 1), 1 + (c_packet[0] >> 2 & 1), enc.force_channels(), ms));
        }
        for &(got, want, forced, ms) in &channels {
            assert_eq!(got, want, "{frame_ms} ms, at {ms} ms: channels");
            // Stereo at first, mono from the low bitrate on, also at 64 kb/s,
            // and stereo again once the caller clears the setting.
            let expected = if ms < 1200 || ms >= 4800 { 2 } else if ms >= 1200 + 2 * frame_ms { 1 } else { want };
            assert_eq!(want, expected, "{frame_ms} ms, at {ms} ms: libopus");
            if (1200 + 2 * frame_ms..4800).contains(&ms) {
                assert_eq!(forced, Some(1), "{frame_ms} ms, at {ms} ms");
            }
        }
    }
}

/// Stereo input coded as mono from the first packet, then in stereo at
/// once together with a change of SILK's internal rate: narrowband SILK at
/// 10 kb/s, then fullband Hybrid at 64 kb/s. The side encoder has never
/// coded a frame. libopus starts it afresh for the first stereo packet
/// (enc_API.c:194-211) and its control call then sets up its resampler
/// rather than carrying a history over to the new rate
/// (control_codec.c:140-142). opus1 carried the side's history from a
/// rate of 0 and failed the packet ("Unsupported SILK rate switch").
#[test]
fn stereo_after_mono_with_a_new_silk_rate_codes_like_libopus() {
    for (rate, frame_ms) in [(48000usize, 20usize), (24000, 10), (48000, 60)] {
        let input = source(Source::Delayed, rate, 3.0);
        let frame = rate * frame_ms / 1000;
        let mut c_enc = CEncoder::new(rate, 2, APP_AUDIO);
        c_enc.complexity(5);
        let mut enc = OpusEncoder::new(rate as i32, 2, Application::Audio).unwrap();
        enc.complexity = 5;
        let mut dec = CDecoder::new(rate, 2);
        let mut buf = vec![0u8; 4000];
        for (i, pcm) in input.chunks_exact(2 * frame).enumerate() {
            let what = format!("{rate} Hz, {frame_ms} ms, packet {i}");
            let (bitrate, bandwidth) = if i * frame_ms < 1000 {
                (10000, (BW_NB, opus1::Bandwidth::Narrowband))
            } else {
                (64000, (BW_FB, opus1::Bandwidth::Fullband))
            };
            c_enc.bitrate(bitrate).bandwidth(bandwidth.0).force_mode(if bitrate > 10000 { MODE_HYBRID } else { MODE_SILK });
            enc.bitrate_bps = bitrate;
            enc.set_bandwidth(bandwidth.1);
            enc.set_force_mode(Some(if bitrate > 10000 { opus1::ForcedMode::Hybrid } else { opus1::ForcedMode::SilkOnly }));
            let n = enc.encode(pcm, frame, &mut buf).unwrap_or_else(|e| panic!("{what}: {e}"));
            let c_packet = c_enc.encode(pcm);
            assert_eq!(buf[0] & 0xFC, c_packet[0] & 0xFC, "{what}: first byte");
            dec.decode(Some(&buf[..n]), frame, false).unwrap();
            assert_eq!(enc.final_range(), dec.final_range(), "{what}: final range");
        }
    }
}

/// Peak and level (RMS) of a signal, in dB.
fn peak_and_level_db(x: &[f32]) -> (f64, f64) {
    let peak = x.iter().fold(0f32, |m, v| m.max(v.abs())) as f64;
    let rms = (x.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / x.len().max(1) as f64).sqrt();
    (20.0 * peak.max(1e-9).log10(), 20.0 * rms.max(1e-9).log10())
}

/// Loud stereo input coded as mono by SILK. The float build of libopus
/// mixes the two channels down as `RES2INT16(left + right)` halved
/// (enc_API.c:309-312): the sum is clipped to 16 bits before it is halved,
/// so where left plus right exceed full scale the mono signal stops at half
/// of it. The input here has the same speech, peaking at 0.9, in both
/// channels: libopus's decoded stream peaks at -4.3 dB (16 kHz, 20 ms).
///
/// opus1 halved the sum of the 16-bit channels, as the fixed-point build
/// does, which does not clip: its stream peaked at -0.1 dB, 4.2 dB above
/// libopus's. The VoIP application keeps the fixed-point rule: SILK's input
/// for it is the fixed-point build's (see `OpusEncoder::encode_impl`).
#[test]
fn loud_stereo_coded_as_mono_clips_the_sum_like_libopus() {
    for (rate, frame_ms, bitrate) in [(16000usize, 20usize, 12000), (48000, 20, 12000), (16000, 10, 14000)] {
        let what = format!("{rate} Hz, {frame_ms} ms, {bitrate} b/s");
        let mono = speech(rate, 3.0);
        let peak = mono.iter().fold(0f32, |m, v| m.max(v.abs()));
        let input: Vec<f32> = mono.iter().flat_map(|&v| [0.9 * v / peak, 0.9 * v / peak]).collect();
        let s = sweep(rate, frame_ms * 10, (APP_AUDIO, Application::Audio), false, &input, &|_| bitrate, &silk_wideband);
        assert!(s.channels.iter().all(|&c| c == (1, 1)), "{what}: mono packets");
        assert!(s.modes.iter().all(|&m| m == (0, 0)), "{what}: SILK-only packets");
        let frame = rate * frame_ms / 1000;
        let (decoded, c_decoded) = decode_both(rate, frame, bitrate, &input);
        let ((peak, level), (c_peak, c_level)) = (peak_and_level_db(&decoded), peak_and_level_db(&c_decoded));
        if std::env::var("CORE_STREAM_CHANNELS_PRINT").is_ok() {
            println!("{what}: peak {peak:.1} dB, level {level:.1} dB; libopus {c_peak:.1} dB, {c_level:.1} dB");
        }
        assert!(c_peak < -4.0, "{what}: libopus peaks at {c_peak:.1} dB");
        assert!((peak - c_peak).abs() < 1.0, "{what}: peak {peak:.1} dB, libopus {c_peak:.1} dB");
        assert!((level - c_level).abs() < 0.5, "{what}: level {level:.1} dB, libopus {c_level:.1} dB");
    }
}

/// SILK-only wideband in both encoders.
fn silk_wideband(enc: &mut OpusEncoder, c_enc: &mut CEncoder) {
    enc.set_force_mode(Some(opus1::ForcedMode::SilkOnly));
    enc.set_bandwidth(opus1::Bandwidth::Wideband);
    c_enc.force_mode(MODE_SILK).bandwidth(BW_WB);
}

/// The SILK-only wideband streams of both encoders for `input` at
/// `bitrate`, decoded by libopus.
fn decode_both(rate: usize, frame: usize, bitrate: i32, input: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let mut c_enc = CEncoder::new(rate, 2, APP_AUDIO);
    c_enc.complexity(5).bitrate(bitrate);
    let mut enc = OpusEncoder::new(rate as i32, 2, Application::Audio).unwrap();
    enc.complexity = 5;
    enc.bitrate_bps = bitrate;
    silk_wideband(&mut enc, &mut c_enc);
    let (mut dec, mut c_dec) = (CDecoder::new(rate, 2), CDecoder::new(rate, 2));
    let mut buf = vec![0u8; 1500];
    let (mut out, mut c_out) = (Vec::new(), Vec::new());
    for pcm in input.chunks_exact(2 * frame) {
        let n = enc.encode(pcm, frame, &mut buf).unwrap();
        out.extend(dec.decode(Some(&buf[..n]), frame, false).unwrap());
        c_out.extend(c_dec.decode(Some(&c_enc.encode(pcm)), frame, false).unwrap());
    }
    (out, c_out)
}

/// A caller that asks for mono packets (`OPUS_SET_FORCE_CHANNELS(1)`) also
/// turns libopus's stereo width estimate off: the width is 0 and its
/// smoothing state stands still (opus_encoder.c:1321-1324). The width moves
/// the bitrate at which the mode changes from SILK or Hybrid to CELT
/// (C 1494-1499): for wide input it lies 16 kb/s lower for speech (VoIP)
/// and 2.8 kb/s lower for music (Audio) than for a width of 0.
///
/// Wide stereo input is coded with mono forced for 3 s, over a bitrate
/// ramp through those thresholds, then with the bitrate deciding again.
/// Every packet has libopus's mode and channel count. With the width
/// measured while mono was forced, opus1 chose CELT below libopus's
/// threshold, and its smoothed width was not libopus's after the setting
/// was cleared.
#[test]
fn forced_mono_turns_the_stereo_width_off_like_libopus() {
    const AUDIO: (i32, Application) = (APP_AUDIO, Application::Audio);
    const VOIP: (i32, Application) = (APP_VOIP, Application::Voip);
    for (rate, application, low, high) in
        [(48000usize, VOIP, 30000, 80000), (24000, VOIP, 30000, 80000), (48000, AUDIO, 10000, 26000)]
    {
        let frame = rate / 50;
        let input = source(Source::Wide, rate, 6.0);
        let mut c_enc = CEncoder::new(rate, 2, application.0);
        c_enc.complexity(5);
        let mut enc = OpusEncoder::new(rate as i32, 2, application.1).unwrap();
        enc.complexity = 5;
        let mut buf = vec![0u8; 4000];
        let packets = input.len() / (2 * frame);
        for (i, pcm) in input.chunks_exact(2 * frame).enumerate() {
            let what = format!("{rate} Hz, {:?}, packet {i}", application.1);
            // Up from `low` to `high` and down again, twice.
            let phase = (i % (packets / 2)) as i32;
            let half = (packets / 4) as i32;
            let bitrate = low + (high - low) * half.min(phase).min(2 * half - phase).max(0) / half;
            if i == 0 {
                c_enc.force_channels(1);
                enc.set_force_channels(Some(1)).unwrap();
            } else if i == packets / 2 {
                c_enc.force_channels(-1000);
                enc.set_force_channels(None).unwrap();
            }
            c_enc.bitrate(bitrate);
            enc.bitrate_bps = bitrate;
            enc.encode(pcm, frame, &mut buf).unwrap_or_else(|e| panic!("{what}: {e}"));
            let c_packet = c_enc.encode(pcm);
            assert_eq!(mode_of(buf[0]), mode_of(c_packet[0]), "{what}: mode at {bitrate} b/s");
            assert_eq!(buf[0] & 4, c_packet[0] & 4, "{what}: channels at {bitrate} b/s");
        }
    }
}
