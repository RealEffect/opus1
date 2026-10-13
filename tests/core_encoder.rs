//! `OpusEncoder` controls against the C libopus encoder under the same
//! settings: forced mode, forced and maximum bandwidth, the LFE flag, the
//! final range, the lookahead and resetting.
//!
//! The first byte of a packet (the table-of-contents or TOC byte) names the
//! mode, the audio bandwidth, the frame duration and mono or stereo. Where
//! this file says that the TOC bytes are equal, opus-rs made the same
//! decisions as libopus for every packet.

mod core_common;
mod k24_common;

use core_common::*;
use k24_common::{aligned_snr, speech_fullband, to_stereo, toc_summary};
use opus_rs::{Application, Bandwidth, ForcedMode, OpusDecoder, OpusEncoder};

const MODES: [(&str, i32, ForcedMode); 3] = [
    ("SILK", MODE_SILK, ForcedMode::SilkOnly),
    ("Hybrid", MODE_HYBRID, ForcedMode::Hybrid),
    ("CELT", MODE_CELT, ForcedMode::CeltOnly),
];

const BANDWIDTHS: [(&str, i32, Bandwidth); 5] = [
    ("NB", BW_NB, Bandwidth::Narrowband),
    ("MB", BW_MB, Bandwidth::Mediumband),
    ("WB", BW_WB, Bandwidth::Wideband),
    ("SWB", BW_SWB, Bandwidth::Superwideband),
    ("FB", BW_FB, Bandwidth::Fullband),
];

const APPS: [(&str, i32, Application); 3] = [
    ("voip", APP_VOIP, Application::Voip),
    ("audio", APP_AUDIO, Application::Audio),
    ("low delay", APP_LOWDELAY, Application::RestrictedLowDelay),
];

/// Speech with content up to the Nyquist frequency, mono or stereo.
fn speech(rate: usize, channels: usize, seconds: f64) -> Vec<f32> {
    let mono = speech_fullband(rate, seconds);
    if channels == 2 { to_stereo(&mono, rate) } else { mono }
}

/// Encode `pcm` in frames of `frame` samples per channel.
fn encode_all(enc: &mut OpusEncoder, pcm: &[f32], frame: usize) -> Vec<Vec<u8>> {
    let mut buf = vec![0u8; 4000];
    pcm.chunks_exact(frame * enc.channels())
        .enumerate()
        .map(|(i, f)| {
            let n = enc.encode(f, frame, &mut buf).unwrap_or_else(|e| panic!("packet {i}: {e}"));
            buf[..n].to_vec()
        })
        .collect()
}

/// Decode with the C decoder. Every packet must decode to `frame` samples.
fn c_decode_all(packets: &[Vec<u8>], rate: usize, channels: usize, frame: usize) -> Vec<f32> {
    let mut dec = CDecoder::new(rate, channels);
    let mut out = Vec::new();
    for (i, p) in packets.iter().enumerate() {
        let pcm = dec.decode(Some(p), frame, false).unwrap_or_else(|e| panic!("packet {i}: libopus error {e}"));
        assert_eq!(pcm.len(), frame * channels, "packet {i}: duration");
        out.extend(pcm);
    }
    out
}

/// SNR in dB of each channel of `decoded` against `input`, at the delay
/// that fits best, without the first 200 ms.
fn channel_snrs(input: &[f32], decoded: &[f32], rate: usize, channels: usize) -> Vec<f64> {
    let skip = rate / 5;
    (0..channels)
        .map(|c| {
            let pick = |x: &[f32]| x.iter().skip(c).step_by(channels).copied().collect::<Vec<f32>>();
            aligned_snr(&pick(input)[skip..], &pick(decoded)[skip..], (rate / 50) as isize).0
        })
        .collect()
}

/// Number of packets whose TOC byte differs in the mode, the bandwidth, the
/// frame duration or the stereo flag. The two lowest bits are left out:
/// they say how the frames of a 40 or 60 ms packet are laid out, which
/// follows from the sizes of the coded frames.
fn toc_mismatches(a: &[Vec<u8>], b: &[Vec<u8>]) -> usize {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).filter(|(x, y)| x[0] & 0xFC != y[0] & 0xFC).count()
}

fn is_mode(toc: u8, mode: i32) -> bool {
    match mode {
        MODE_CELT => toc & 0x80 != 0,
        MODE_HYBRID => toc & 0xE0 == 0x60,
        _ => toc & 0x80 == 0 && toc & 0x60 != 0x60,
    }
}

// ---------------------------------------------------------------------------
// Forced mode
// ---------------------------------------------------------------------------

/// Every forced mode at one input rate, mono and stereo:
///
/// - every packet has the TOC byte libopus writes under the same forcing;
/// - that is the forced mode wherever the format has it (Hybrid needs
///   super-wideband, so 24 or 48 kHz input; below that libopus codes
///   SILK-only, and so does opus-rs);
/// - the C decoder decodes the packets, to a quality within 2 dB of what
///   it gets from the C encoder's packets.
fn forced_modes_at(rate: usize) {
    for channels in [1usize, 2] {
        for (name, c_mode, mode) in MODES {
            let pcm = speech(rate, channels, 1.5);
            let frame = rate / 50;
            let bitrate = match c_mode {
                MODE_SILK => 20000,
                MODE_HYBRID => 32000,
                _ => 48000,
            } * channels as i32;
            // SILK-only reaches wideband at most. Without this, forced SILK
            // from 24 or 48 kHz input becomes Hybrid, in libopus too.
            let limit_to_wideband = c_mode == MODE_SILK && rate > 16000;

            let mut c_enc = CEncoder::new(rate, channels, APP_AUDIO);
            c_enc.force_mode(c_mode).bitrate(bitrate);
            let mut enc = OpusEncoder::new(rate as i32, channels, Application::Audio).unwrap();
            enc.set_force_mode(Some(mode));
            enc.bitrate_bps = bitrate;
            if limit_to_wideband {
                c_enc.bandwidth(BW_WB);
                enc.set_bandwidth(Bandwidth::Wideband);
            }
            let c_packets = c_enc.encode_all(&pcm, frame);
            let packets = encode_all(&mut enc, &pcm, frame);
            let what = format!("{rate} Hz, {channels} ch, forced {name}");

            assert_eq!(toc_mismatches(&c_packets, &packets), 0, "{what}: libopus [{}], opus-rs [{}]", toc_summary(&c_packets), toc_summary(&packets));
            let reachable = c_mode != MODE_HYBRID || rate >= 24000;
            assert_eq!(packets.iter().all(|p| is_mode(p[0], c_mode)), reachable, "{what}: [{}]", toc_summary(&packets));

            let c_snr = channel_snrs(&pcm, &c_decode_all(&c_packets, rate, channels, frame), rate, channels);
            let snr = channel_snrs(&pcm, &c_decode_all(&packets, rate, channels, frame), rate, channels);
            println!("{what} [{}]: SNR libopus {c_snr:.1?} dB, opus-rs {snr:.1?} dB", toc_summary(&packets));
            for c in 0..channels {
                assert!(snr[c] > c_snr[c] - 2.0, "{what}, channel {c}: opus-rs {:.1} dB, libopus {:.1} dB", snr[c], c_snr[c]);
            }
        }
    }
}

#[test]
fn forced_modes_at_8_khz() {
    forced_modes_at(8000);
}
#[test]
fn forced_modes_at_12_khz() {
    forced_modes_at(12000);
}
#[test]
fn forced_modes_at_16_khz() {
    forced_modes_at(16000);
}
#[test]
fn forced_modes_at_24_khz() {
    forced_modes_at(24000);
}
#[test]
fn forced_modes_at_48_khz() {
    forced_modes_at(48000);
}

/// A forced mode is a request. Frames under 10 ms are CELT-only whatever is
/// asked, a `RestrictedLowDelay` encoder codes CELT-only, and 40 and 60 ms
/// of CELT or Hybrid go out as 20 ms frames in one packet. `None` gives the
/// choice back to the encoder.
#[test]
fn forced_mode_yields_to_the_frame_size_and_the_application() {
    let rate = 48000;
    let pcm = speech(rate, 1, 0.6);
    for (name, c_mode, mode) in MODES {
        for frame_2_5ms in [1usize, 2, 4, 8, 16, 24] {
            for (app_name, c_app, app) in APPS {
                let frame = rate / 400 * frame_2_5ms;
                let mut c_enc = CEncoder::new(rate, 1, c_app);
                c_enc.force_mode(c_mode).bitrate(32000).complexity(5);
                let mut enc = OpusEncoder::new(rate as i32, 1, app).unwrap();
                enc.set_force_mode(Some(mode));
                enc.bitrate_bps = 32000;
                enc.complexity = 5;
                let c_packets = c_enc.encode_all(&pcm, frame);
                let packets = encode_all(&mut enc, &pcm, frame);
                let what = format!("forced {name}, {} ms, {app_name}", frame_2_5ms as f32 * 2.5);
                assert_eq!(toc_mismatches(&c_packets, &packets), 0, "{what}: libopus [{}], opus-rs [{}]", toc_summary(&c_packets), toc_summary(&packets));
                c_decode_all(&packets, rate, 1, frame);
            }
        }
    }

    // Clearing the forced mode: the packets are those of an encoder that
    // never had one.
    let mut cleared = OpusEncoder::new(48000, 1, Application::Voip).unwrap();
    cleared.set_force_mode(Some(ForcedMode::CeltOnly));
    cleared.set_force_mode(None);
    let mut plain = OpusEncoder::new(48000, 1, Application::Voip).unwrap();
    assert_eq!(encode_all(&mut cleared, &pcm, 960), encode_all(&mut plain, &pcm, 960));
}

// ---------------------------------------------------------------------------
// Bandwidth
// ---------------------------------------------------------------------------

/// Narrowband, mediumband and wideband SILK from 24 and 48 kHz input: SILK
/// then runs at 8, 12 or 16 kHz behind a 3:1 to 6:1 resampler. The TOC is
/// libopus's, and the C decoder gets the quality it gets from libopus's
/// packets.
#[test]
fn forced_silk_bandwidths_from_24_and_48_khz_input() {
    for rate in [24000usize, 48000] {
        for channels in [1usize, 2] {
            for (bw_name, c_bw, bw) in &BANDWIDTHS[..3] {
                for use_max in [false, true] {
                    let pcm = speech(rate, channels, 1.2);
                    let frame = rate / 50;
                    let bitrate = 18000 * channels as i32;
                    let mut c_enc = CEncoder::new(rate, channels, APP_VOIP);
                    c_enc.force_mode(MODE_SILK).bitrate(bitrate);
                    let mut enc = OpusEncoder::new(rate as i32, channels, Application::Voip).unwrap();
                    enc.set_force_mode(Some(ForcedMode::SilkOnly));
                    enc.bitrate_bps = bitrate;
                    if use_max {
                        c_enc.max_bandwidth(*c_bw);
                        enc.set_max_bandwidth(*bw).unwrap();
                    } else {
                        c_enc.bandwidth(*c_bw);
                        enc.set_bandwidth(*bw);
                    }
                    let c_packets = c_enc.encode_all(&pcm, frame);
                    let packets = encode_all(&mut enc, &pcm, frame);
                    let what = format!("{rate} Hz, {channels} ch, SILK, {} {bw_name}", if use_max { "maximum" } else { "bandwidth" });

                    assert_eq!(toc_mismatches(&c_packets, &packets), 0, "{what}: libopus [{}], opus-rs [{}]", toc_summary(&c_packets), toc_summary(&packets));
                    let expected = format!("SILK-{bw_name}");
                    assert!(toc_summary(&packets).starts_with(&expected), "{what}: [{}]", toc_summary(&packets));

                    let c_snr = channel_snrs(&pcm, &c_decode_all(&c_packets, rate, channels, frame), rate, channels);
                    let snr = channel_snrs(&pcm, &c_decode_all(&packets, rate, channels, frame), rate, channels);
                    println!("{what}: SNR libopus {c_snr:.1?} dB, opus-rs {snr:.1?} dB");
                    for c in 0..channels {
                        assert!(snr[c] > c_snr[c] - 2.0, "{what}, channel {c}: opus-rs {:.1} dB, libopus {:.1} dB", snr[c], c_snr[c]);
                    }
                }
            }
        }
    }
}

/// A forced bandwidth and a maximum bandwidth, with each forced mode and
/// with the automatic mode choice, for one input rate: the TOC byte of every
/// packet is libopus's. This covers the rules that turn a request into what
/// is coded: nothing above the input's Nyquist frequency, no mediumband in
/// CELT, SILK above wideband is Hybrid, Hybrid at wideband or below is
/// SILK, and a forced bandwidth is not limited by the maximum.
///
/// Both encoders run at complexity 5. From complexity 7 libopus also
/// listens to the signal (speech or music, and how wide), which opus-rs
/// does not do.
///
/// Returns the number of cases compared.
fn bandwidth_settings_at(rate: usize, with_automatic_mode: bool) -> usize {
    let mut cases = 0;
    let modes = MODES.iter().map(|&(name, c_mode, mode)| (name, Some(c_mode), Some(mode)));
    let modes: Vec<_> = modes.chain(with_automatic_mode.then_some(("automatic", None, None))).collect();
    for channels in [1usize, 2] {
        let pcm = speech(rate, channels, 0.4);
        let frame = rate / 50;
        for (app_name, c_app, app) in &APPS[..2] {
            for &(mode_name, c_mode, mode) in &modes {
                for bitrate in [12000, 32000] {
                    for (bw_name, c_bw, bw) in BANDWIDTHS {
                        for use_max in [false, true] {
                            let mut c_enc = CEncoder::new(rate, channels, *c_app);
                            c_enc.bitrate(bitrate * channels as i32).complexity(5);
                            let mut enc = OpusEncoder::new(rate as i32, channels, *app).unwrap();
                            enc.bitrate_bps = bitrate * channels as i32;
                            enc.complexity = 5;
                            enc.set_force_mode(mode);
                            if let Some(c_mode) = c_mode {
                                c_enc.force_mode(c_mode);
                            }
                            if use_max {
                                c_enc.max_bandwidth(c_bw);
                                enc.set_max_bandwidth(bw).unwrap();
                            } else {
                                c_enc.bandwidth(c_bw);
                                enc.set_bandwidth(bw);
                            }
                            let c_packets = c_enc.encode_all(&pcm, frame);
                            let packets = encode_all(&mut enc, &pcm, frame);
                            assert_eq!(
                                toc_mismatches(&c_packets, &packets),
                                0,
                                "{rate} Hz, {channels} ch, {app_name}, {mode_name} mode, {bitrate} b/s per channel, {} {bw_name}: libopus [{}], opus-rs [{}]",
                                if use_max { "maximum" } else { "bandwidth" },
                                toc_summary(&c_packets),
                                toc_summary(&packets)
                            );
                            cases += 1;
                        }
                    }
                }
            }
        }
    }
    cases
}

#[test]
fn bandwidth_settings_at_8_khz() {
    assert_eq!(bandwidth_settings_at(8000, false), 240);
}
#[test]
fn bandwidth_settings_at_12_khz() {
    assert_eq!(bandwidth_settings_at(12000, false), 240);
}
#[test]
fn bandwidth_settings_at_16_khz() {
    assert_eq!(bandwidth_settings_at(16000, false), 240);
}
#[test]
fn bandwidth_settings_at_24_khz() {
    assert_eq!(bandwidth_settings_at(24000, true), 320);
}
#[test]
fn bandwidth_settings_at_48_khz() {
    assert_eq!(bandwidth_settings_at(48000, true), 320);
}

/// The automatic mode choice from 8, 12 and 16 kHz input. libopus takes
/// CELT-only when the bitrate is high for speech (`opus_encoder.c:1513`,
/// `st->mode = (equiv_rate >= threshold) ? MODE_CELT_ONLY : MODE_SILK_ONLY`),
/// at any input rate, and so does opus-rs. The forced modes are not
/// affected.
#[test]
fn automatic_mode_below_24_khz_input_follows_libopus() {
    let mut failures = Vec::new();
    for rate in [8000usize, 12000, 16000] {
        for channels in [1usize, 2] {
            for (app_name, c_app, app) in &APPS[..2] {
                for bitrate in [12000, 32000] {
                    let pcm = speech(rate, channels, 0.6);
                    let frame = rate / 50;
                    let mut c_enc = CEncoder::new(rate, channels, *c_app);
                    c_enc.bitrate(bitrate * channels as i32).complexity(5);
                    let mut enc = OpusEncoder::new(rate as i32, channels, *app).unwrap();
                    enc.bitrate_bps = bitrate * channels as i32;
                    enc.complexity = 5;
                    let c_packets = c_enc.encode_all(&pcm, frame);
                    let packets = encode_all(&mut enc, &pcm, frame);
                    if toc_mismatches(&c_packets, &packets) != 0 {
                        failures.push(format!("{rate} Hz, {channels} ch, {app_name}, {bitrate} b/s per channel: libopus [{}], opus-rs [{}]", toc_summary(&c_packets), toc_summary(&packets)));
                    }
                }
            }
        }
    }
    println!("{}", failures.join("\n"));
    assert!(failures.is_empty(), "{} of 24 cases differ", failures.len());
}

/// The bitrate from which the automatic choice takes CELT-only depends on
/// how wide the stereo image is (`opus_encoder.c:1495-1499`, with
/// `compute_stereo_width`): for speech it falls from 64 kb/s, when both
/// channels carry the same signal, to 44 kb/s for wide stereo.
///
/// VoIP at 56 kb/s lies between the two. Three kinds of stereo input:
///
/// - the same signal in both channels: SILK (Hybrid from 48 kHz) throughout;
/// - wide stereo: the width estimate rises from 0 over the first packets,
///   and the stream goes over to CELT-only;
/// - the same signal for a second, then wide stereo: it goes over later.
///
/// The packets have libopus's first byte, but for the packet at the change:
/// libopus keeps the old mode for one packet more and appends a 5 ms CELT
/// frame, which opus-rs does not write yet.
#[test]
fn automatic_mode_follows_the_stereo_width() {
    let mut failures = Vec::new();
    for rate in [16000usize, 48000] {
        let mono = speech(rate, 1, 2.4);
        let wide = to_stereo(&mono, rate);
        let same: Vec<f32> = mono.iter().flat_map(|&v| [v, v]).collect();
        let mut later = same[..2 * rate].to_vec();
        later.extend_from_slice(&wide[2 * rate..]);
        for (name, pcm, changes) in [("the same signal in both channels", &same, false), ("wide stereo", &wide, true), ("wide stereo after one second", &later, true)] {
            let frame = rate / 50;
            let mut c_enc = CEncoder::new(rate, 2, APP_VOIP);
            // libopus would also choose between mono and stereo packets.
            c_enc.bitrate(56000).complexity(5).force_channels(2);
            let mut enc = OpusEncoder::new(rate as i32, 2, Application::Voip).unwrap();
            enc.bitrate_bps = 56000;
            enc.complexity = 5;
            let c_packets = c_enc.encode_all(pcm, frame);
            let packets = encode_all(&mut enc, pcm, frame);
            let celt = |p: &[Vec<u8>]| p.iter().filter(|x| is_mode(x[0], MODE_CELT)).count();
            let first_celt = |p: &[Vec<u8>]| p.iter().position(|x| is_mode(x[0], MODE_CELT));
            let mismatches = toc_mismatches(&c_packets, &packets);
            let what = format!(
                "{rate} Hz, {name}: {mismatches} first bytes differ; CELT-only from packet {:?} (libopus {:?})\n  libopus [{}]\n  opus-rs [{}]",
                first_celt(&packets),
                first_celt(&c_packets),
                toc_summary(&c_packets),
                toc_summary(&packets)
            );
            println!("{what}");
            // The stream is what the case is meant to show.
            assert_eq!(celt(&c_packets) > 0, changes, "{what}");
            if changes {
                assert!(celt(&c_packets) < c_packets.len(), "{what}");
            }
            if mismatches > 1 {
                failures.push(what);
            }
        }
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

/// A packet cannot be larger than the caller's buffer. libopus takes that
/// into account before it decides anything (`user_bitrate_to_bitrate`,
/// opus_encoder.c:1325): with a variable bitrate, the bitrate it goes by is
/// the one asked for, or the rate the buffer stands for when that is lower.
/// With 128 kb/s asked for and room for 25 bytes per 20 ms it codes like a
/// 10 kb/s stream: SILK, not CELT at the full bandwidth of the input.
///
/// Every setting here asks for more than the buffer holds. The first bytes
/// must be libopus's, no packet may exceed the buffer, and the mean packet
/// size must be within 10 % of libopus's.
#[test]
fn a_small_output_buffer_lowers_the_bitrate_the_decisions_go_by() {
    let mut failures = Vec::new();
    let (mut cases, mut silk_cases, mut lowest, mut highest) = (0, 0, 1f64, 1f64);
    for rate in [8000usize, 16000, 48000] {
        for channels in [1usize, 2] {
            let pcm = speech(rate, channels, 0.6);
            for (app_name, c_app, app) in &APPS {
                for bitrate in [32000, 64000, 128000] {
                    for frame_ms in [10usize, 20, 60] {
                        for max_bytes in [12usize, 25, 60] {
                            let frame = rate * frame_ms / 1000;
                            let mut c_enc = CEncoder::new(rate, channels, *c_app);
                            c_enc.bitrate(bitrate).complexity(5);
                            if channels == 2 {
                                // libopus would also choose between mono and
                                // stereo packets.
                                c_enc.force_channels(2);
                            }
                            let mut enc = OpusEncoder::new(rate as i32, channels, *app).unwrap();
                            enc.bitrate_bps = bitrate;
                            enc.complexity = 5;
                            let mut buf = vec![0u8; max_bytes];
                            let (mut c_packets, mut packets) = (Vec::new(), Vec::new());
                            for f in pcm.chunks_exact(frame * channels) {
                                c_packets.push(c_enc.encode_max(f, max_bytes));
                                let n = enc.encode(f, frame, &mut buf).unwrap();
                                packets.push(buf[..n].to_vec());
                            }
                            let mean = |p: &[Vec<u8>]| p.iter().map(|x| x.len()).sum::<usize>() as f64 / p.len() as f64;
                            let ratio = mean(&packets) / mean(&c_packets);
                            cases += 1;
                            silk_cases += is_mode(c_packets[c_packets.len() / 2][0], MODE_SILK) as usize;
                            (lowest, highest) = (lowest.min(ratio), highest.max(ratio));
                            if toc_mismatches(&c_packets, &packets) != 0 || !(0.9..=1.1).contains(&ratio) {
                                failures.push(format!(
                                    "{rate} Hz, {channels} ch, {app_name}, {bitrate} b/s, {frame_ms} ms, {max_bytes} bytes: libopus [{}] {:.1} bytes, opus-rs [{}] {:.1} bytes",
                                    toc_summary(&c_packets), mean(&c_packets), toc_summary(&packets), mean(&packets)
                                ));
                            }
                        }
                    }
                }
            }
        }
    }
    println!("{cases} settings, {silk_cases} of them SILK in libopus: mean packet size {lowest:.3} to {highest:.3} of libopus's");
    // The buffer must change the mode in many of them: asked for 32 kb/s and
    // more, the low-delay application aside, libopus would not code SILK.
    assert!(silk_cases >= 100, "{silk_cases} settings are SILK");
    assert!(failures.is_empty(), "{} of {cases} settings differ:\n{}", failures.len(), failures.join("\n"));
}

/// The bandwidth is changed while the stream runs. A running SILK encoder
/// does not jump: it moves one step (16, 12, 8 kHz) at a time, and only in
/// a pause or after 5 seconds. opus-rs follows the requests the way libopus
/// does. The step can come a few packets earlier or later, because it
/// hangs on SILK's speech activity measure, so a small share of the packets
/// may differ.
#[test]
fn bandwidth_changes_in_a_running_stream_follow_libopus() {
    let schedule = [BANDWIDTHS[2], BANDWIDTHS[0], BANDWIDTHS[1], BANDWIDTHS[2], BANDWIDTHS[0], BANDWIDTHS[2]];
    for (rate, channels) in [(16000usize, 1usize), (48000, 1), (24000, 2)] {
        let pcm = speech(rate, channels, 12.0);
        let frame = rate / 50;
        let mut c_enc = CEncoder::new(rate, channels, APP_VOIP);
        c_enc.force_mode(MODE_SILK).bitrate(20000 * channels as i32).complexity(5);
        let mut enc = OpusEncoder::new(rate as i32, channels, Application::Voip).unwrap();
        enc.set_force_mode(Some(ForcedMode::SilkOnly));
        enc.bitrate_bps = 20000 * channels as i32;
        enc.complexity = 5;
        let (mut c_packets, mut packets) = (Vec::new(), Vec::new());
        let mut buf = vec![0u8; 4000];
        for (i, f) in pcm.chunks_exact(frame * channels).enumerate() {
            // A new request every 2 seconds.
            if i % 100 == 0 {
                let (_, c_bw, bw) = schedule[(i / 100) % schedule.len()];
                c_enc.bandwidth(c_bw);
                enc.set_bandwidth(bw);
            }
            c_packets.push(c_enc.encode(f));
            let n = enc.encode(f, frame, &mut buf).unwrap();
            packets.push(buf[..n].to_vec());
        }
        let differ = toc_mismatches(&c_packets, &packets);
        println!("{rate} Hz, {channels} ch: {differ} of {} TOC bytes differ\n  libopus [{}]\n  opus-rs [{}]", packets.len(), toc_summary(&c_packets), toc_summary(&packets));
        assert!(differ * 100 <= 8 * packets.len(), "{rate} Hz, {channels} ch: {differ} of {} TOC bytes differ", packets.len());
        // Both streams use the same bandwidths, and more than one.
        let kinds = |p: &[Vec<u8>]| p.iter().map(|x| x[0]).collect::<std::collections::BTreeSet<u8>>();
        assert_eq!(kinds(&c_packets), kinds(&packets), "{rate} Hz, {channels} ch");
        assert!(kinds(&packets).len() >= 2, "{rate} Hz, {channels} ch: the stream changes bandwidth");
        c_decode_all(&packets, rate, channels, frame);
    }
}

#[test]
fn the_maximum_bandwidth_must_be_a_bandwidth() {
    let mut enc = OpusEncoder::new(48000, 1, Application::Audio).unwrap();
    assert!(enc.set_max_bandwidth(Bandwidth::Auto).is_err());
    assert!(enc.set_max_bandwidth(Bandwidth::Wideband).is_ok());
    // `Auto` clears a forced bandwidth: the packets are those of an encoder
    // that never had one.
    let pcm = speech(48000, 1, 0.5);
    let mut cleared = OpusEncoder::new(48000, 1, Application::Audio).unwrap();
    cleared.set_bandwidth(Bandwidth::Narrowband);
    cleared.set_bandwidth(Bandwidth::Auto);
    let mut plain = OpusEncoder::new(48000, 1, Application::Audio).unwrap();
    assert_eq!(encode_all(&mut cleared, &pcm, 960), encode_all(&mut plain, &pcm, 960));
}

// ---------------------------------------------------------------------------
// LFE
// ---------------------------------------------------------------------------

/// An encoder with the LFE flag codes CELT-only narrowband, whatever mode
/// and bandwidth are asked for, as libopus does. The C decoder decodes the
/// packets.
#[test]
fn lfe_is_celt_only_narrowband() {
    for rate in [16000usize, 24000, 48000] {
        for (app_name, c_app, app) in APPS {
            for (mode_name, c_mode, mode) in MODES {
                for frame_2_5ms in [4usize, 8, 24] {
                    let frame = rate / 400 * frame_2_5ms;
                    // A low-frequency-effects channel: an 60 Hz tone that
                    // comes and goes.
                    let pcm: Vec<f32> = (0..rate / 2)
                        .map(|i| {
                            let t = i as f32 / rate as f32;
                            0.5 * (2.0 * std::f32::consts::PI * 60.0 * t).sin() * (3.0 * t).sin().abs()
                        })
                        .collect();
                    let mut c_enc = CEncoder::new(rate, 1, c_app);
                    c_enc.lfe(true).force_mode(c_mode).bandwidth(BW_FB).bitrate(16000);
                    let mut enc = OpusEncoder::new(rate as i32, 1, app).unwrap();
                    enc.set_lfe(true);
                    assert!(enc.lfe());
                    enc.set_force_mode(Some(mode));
                    enc.set_bandwidth(Bandwidth::Fullband);
                    enc.bitrate_bps = 16000;
                    let c_packets = c_enc.encode_all(&pcm, frame);
                    let packets = encode_all(&mut enc, &pcm, frame);
                    let what = format!("{rate} Hz, {app_name}, forced {mode_name}, {} ms", frame_2_5ms as f32 * 2.5);
                    assert_eq!(toc_mismatches(&c_packets, &packets), 0, "{what}: libopus [{}], opus-rs [{}]", toc_summary(&c_packets), toc_summary(&packets));
                    // CELT-only narrowband mono is configuration 16 to 19.
                    assert!(packets.iter().all(|p| (16..20).contains(&(p[0] >> 3)) && p[0] & 4 == 0), "{what}: [{}]", toc_summary(&packets));
                    let decoded = c_decode_all(&packets, rate, 1, frame);
                    assert!(decoded.iter().all(|v| v.is_finite()));
                }
            }
        }
    }
    // Switching the flag off gives the normal choice back.
    let pcm = speech(48000, 1, 0.3);
    let mut enc = OpusEncoder::new(48000, 1, Application::Audio).unwrap();
    enc.set_lfe(true);
    enc.set_lfe(false);
    let mut plain = OpusEncoder::new(48000, 1, Application::Audio).unwrap();
    assert_eq!(encode_all(&mut enc, &pcm, 960), encode_all(&mut plain, &pcm, 960));
}

/// The energy mask is stored, 21 values per channel, and it changes the
/// packets. tests/core_lfe_mask.rs compares the packet sizes with libopus.
#[test]
fn energy_mask_is_stored() {
    let pcm = speech(48000, 2, 0.3);
    let mut enc = OpusEncoder::new(48000, 2, Application::Audio).unwrap();
    assert_eq!(enc.energy_mask(), None);
    let mask: Vec<f32> = (0..50).map(|i| -0.1 * i as f32).collect();
    enc.set_energy_mask(Some(&mask));
    assert_eq!(enc.energy_mask(), Some(&mask[..42]));
    let mut plain = OpusEncoder::new(48000, 2, Application::Audio).unwrap();
    assert_ne!(encode_all(&mut enc, &pcm, 960), encode_all(&mut plain, &pcm, 960));
    // Too short for two channels: the mask is cleared, nothing panics.
    enc.set_energy_mask(Some(&mask[..41]));
    assert_eq!(enc.energy_mask(), None);
    enc.set_energy_mask(Some(&mask[..42]));
    enc.set_energy_mask(None);
    assert_eq!(enc.energy_mask(), None);
    // A mono encoder keeps 21 values.
    let mut mono = OpusEncoder::new(48000, 1, Application::Audio).unwrap();
    mono.set_energy_mask(Some(&mask));
    assert_eq!(mono.energy_mask(), Some(&mask[..21]));
    // A reset clears the mask, as in libopus.
    mono.reset_state();
    assert_eq!(mono.energy_mask(), None);
}

// ---------------------------------------------------------------------------
// Final range
// ---------------------------------------------------------------------------

/// One stream for the final range tests: (rate, channels, mode, bandwidth,
/// bitrate, frame duration in units of 2.5 ms, CBR, FEC).
type RangeStream = (usize, usize, ForcedMode, Bandwidth, i32, usize, bool, bool);

/// Encode a stream and decode every packet with the C decoder and with the
/// opus-rs decoder. Returns the number of packets, and for how many of them
/// the encoder's final range differs from the C decoder's and from the
/// opus-rs decoder's.
fn final_range_run(stream: RangeStream) -> (usize, usize, usize) {
    let (rate, channels, mode, bandwidth, bitrate, frame_2_5ms, cbr, fec) = stream;
    let pcm = speech(rate, channels, 1.5);
    let frame = rate / 400 * frame_2_5ms;
    let mut enc = OpusEncoder::new(rate as i32, channels, Application::Audio).unwrap();
    enc.set_force_mode(Some(mode));
    enc.set_bandwidth(bandwidth);
    enc.bitrate_bps = bitrate;
    enc.use_cbr = cbr;
    enc.use_inband_fec = fec;
    enc.packet_loss_perc = if fec { 15 } else { 0 };
    assert_eq!(enc.final_range(), 0, "before the first packet");
    let mut c_dec = CDecoder::new(rate, channels);
    let mut dec = OpusDecoder::new(rate as i32, channels).unwrap();
    let mut buf = vec![0u8; 4000];
    let mut out = vec![0f32; frame * channels];
    let (mut packets, mut differs_from_c, mut differs_from_rs) = (0, 0, 0);
    for f in pcm.chunks_exact(frame * channels) {
        let n = enc.encode(f, frame, &mut buf).unwrap();
        let range = enc.final_range();
        assert_ne!(range, 0, "{stream:?}: every packet carries coded data");
        c_dec.decode(Some(&buf[..n]), frame, false).unwrap_or_else(|e| panic!("{stream:?}: libopus error {e}"));
        dec.decode(&buf[..n], frame, &mut out).unwrap();
        differs_from_c += (c_dec.final_range() != range) as usize;
        differs_from_rs += (dec.final_range() != range) as usize;
        packets += 1;
    }
    (packets, differs_from_c, differs_from_rs)
}

/// The encoder's final range equals what the C decoder and the opus-rs
/// decoder report after decoding the packet, for every packet: the decoders
/// read exactly the symbols the encoder wrote. SILK-only, Hybrid and
/// CELT-only; 2.5 to 60 ms, which includes packets of several frames; VBR
/// and CBR; with and without in-band FEC.
#[test]
fn final_range_equals_the_decoders() {
    let streams: [RangeStream; 12] = [
        (8000, 1, ForcedMode::SilkOnly, Bandwidth::Narrowband, 12000, 8, false, false),
        (16000, 1, ForcedMode::SilkOnly, Bandwidth::Wideband, 24000, 8, true, false),
        (16000, 2, ForcedMode::SilkOnly, Bandwidth::Wideband, 36000, 8, false, true),
        (48000, 1, ForcedMode::SilkOnly, Bandwidth::Mediumband, 20000, 4, false, false),
        (24000, 1, ForcedMode::SilkOnly, Bandwidth::Wideband, 24000, 24, false, true),
        (24000, 1, ForcedMode::Hybrid, Bandwidth::Superwideband, 32000, 8, false, false),
        (48000, 1, ForcedMode::Hybrid, Bandwidth::Fullband, 32000, 4, true, false),
        (48000, 1, ForcedMode::Hybrid, Bandwidth::Fullband, 40000, 16, false, true),
        (48000, 1, ForcedMode::CeltOnly, Bandwidth::Fullband, 64000, 8, false, false),
        (48000, 2, ForcedMode::CeltOnly, Bandwidth::Fullband, 96000, 1, true, false),
        (24000, 2, ForcedMode::CeltOnly, Bandwidth::Superwideband, 64000, 2, false, false),
        (16000, 1, ForcedMode::CeltOnly, Bandwidth::Wideband, 48000, 16, true, false),
    ];
    for stream in streams {
        let (packets, differs_from_c, differs_from_rs) = final_range_run(stream);
        assert!(packets >= 25, "{stream:?}");
        assert_eq!((differs_from_c, differs_from_rs), (0, 0), "{stream:?}: the final range differs from the C decoder's on {differs_from_c} and from the opus-rs decoder's on {differs_from_rs} of {packets} packets");
    }
}

/// Stereo Hybrid. The final range shows a fault in the CELT layer of these
/// packets: the two decoders agree with each other and disagree with the
/// encoder on every packet.
///
/// The cause is in the CELT encoder (src/celt.rs). It limits the first
/// intensity-stereo band with `self.intensity.clamp(0, end)`. libopus
/// limits it to the coded bands, `IMIN(st->end, IMAX(st->start,
/// st->intensity))` (celt_encoder.c). A Hybrid frame starts at band 17, so
/// a band below 17 is written as a negative offset and the rest of the
/// frame decodes to noise (see the next test).
#[test]
#[ignore = "stereo Hybrid packets have a corrupt CELT layer: src/celt.rs limits the intensity band to 0..end, libopus to start..end"]
fn final_range_equals_the_decoders_for_stereo_hybrid() {
    let streams: [RangeStream; 4] = [
        (48000, 2, ForcedMode::Hybrid, Bandwidth::Fullband, 64000, 8, false, false),
        (48000, 2, ForcedMode::Hybrid, Bandwidth::Fullband, 64000, 4, true, false),
        (48000, 2, ForcedMode::Hybrid, Bandwidth::Superwideband, 56000, 24, false, false),
        (24000, 2, ForcedMode::Hybrid, Bandwidth::Superwideband, 48000, 8, false, true),
    ];
    for stream in streams {
        let (packets, differs_from_c, differs_from_rs) = final_range_run(stream);
        assert_eq!((differs_from_c, differs_from_rs), (0, 0), "{stream:?}: the final range differs from the C decoder's on {differs_from_c} and from the opus-rs decoder's on {differs_from_rs} of {packets} packets");
    }
}

/// Power of the component of `x` at `freq` Hz.
fn tone_power(x: &[f32], rate: usize, freq: f64) -> f64 {
    let (mut re, mut im) = (0f64, 0f64);
    for (i, &v) in x.iter().enumerate() {
        let phase = 2.0 * std::f64::consts::PI * freq * i as f64 / rate as f64;
        re += v as f64 * phase.cos();
        im += v as f64 * phase.sin();
    }
    (re * re + im * im) / (x.len() as f64).powi(2)
}

/// What the fault above does to the sound: a tone at 9 kHz, in the CELT
/// layer of a stereo Hybrid stream, comes back 30 dB or more too quiet in
/// the left channel, where libopus loses about 1 dB. With the intensity
/// band limited as in libopus it is within 2 dB of libopus. Only the left
/// channel is measured: the right one also depends on where intensity
/// stereo starts, which opus-rs still chooses differently.
#[test]
#[ignore = "stereo Hybrid packets have a corrupt CELT layer: src/celt.rs limits the intensity band to 0..end, libopus to start..end"]
fn stereo_hybrid_keeps_the_band_above_8_khz() {
    let rate = 48000;
    let frame = 960;
    let speech = speech_fullband(rate, 2.0);
    let pcm: Vec<f32> = speech
        .iter()
        .enumerate()
        .flat_map(|(i, &s)| {
            let tone = 0.05 * (2.0 * std::f32::consts::PI * 9000.0 * i as f32 / rate as f32).sin();
            [0.5 * s + tone, 0.5 * s + 0.6 * tone]
        })
        .collect();
    let mut c_enc = CEncoder::new(rate, 2, APP_AUDIO);
    c_enc.force_mode(MODE_HYBRID).bandwidth(BW_FB).bitrate(64000);
    let mut enc = OpusEncoder::new(rate as i32, 2, Application::Audio).unwrap();
    enc.set_force_mode(Some(ForcedMode::Hybrid));
    enc.set_bandwidth(Bandwidth::Fullband);
    enc.bitrate_bps = 64000;
    let left = |x: &[f32]| x.iter().step_by(2).copied().collect::<Vec<f32>>();
    let level = |packets: &[Vec<u8>]| {
        let decoded = c_decode_all(packets, rate, 2, frame);
        10.0 * (tone_power(&left(&decoded), rate, 9000.0) / tone_power(&left(&pcm), rate, 9000.0)).log10()
    };
    let c_level = level(&c_enc.encode_all(&pcm, frame));
    let rs_level = level(&encode_all(&mut enc, &pcm, frame));
    println!("9 kHz tone, left channel, against the input: libopus {c_level:+.1} dB, opus-rs {rs_level:+.1} dB");
    assert!(rs_level > c_level - 2.0, "9 kHz tone: libopus {c_level:+.1} dB, opus-rs {rs_level:+.1} dB");
}

/// A packet too small to code anything carries no range-coded data: its
/// final range is 0, in the encoder and in both decoders (they conceal it).
#[test]
fn final_range_is_zero_without_coded_data() {
    for cbr in [false, true] {
        let mut enc = OpusEncoder::new(16000, 1, Application::Voip).unwrap();
        enc.use_cbr = cbr;
        let pcm = speech(16000, 1, 0.4);
        let mut c_dec = CDecoder::new(16000, 1);
        let mut buf = vec![0u8; 400];
        for (i, f) in pcm.as_chunks::<320>().0.iter().enumerate() {
            // Normal packets first, then packets of 1 or 2 bytes.
            let starved = i >= 10;
            enc.bitrate_bps = if starved { 600 } else { 24000 };
            let n = enc.encode(f, 320, &mut buf).unwrap();
            c_dec.decode(Some(&buf[..n]), 320, false).unwrap();
            assert_eq!(enc.final_range(), c_dec.final_range(), "CBR {cbr}, packet {i} ({n} bytes)");
            assert_eq!(enc.final_range() == 0, starved, "CBR {cbr}, packet {i} ({n} bytes)");
        }
    }
}

// ---------------------------------------------------------------------------
// Reset, lookahead, accessors
// ---------------------------------------------------------------------------

/// After `reset_state` the encoder writes exactly the packets a new encoder
/// with the same settings writes.
#[test]
fn reset_state_gives_a_new_encoder_with_the_same_settings() {
    for (rate, channels, app) in [(16000usize, 1usize, Application::Voip), (48000, 2, Application::Audio), (24000, 1, Application::RestrictedLowDelay)] {
        for (mode, bandwidth, max, lfe) in [
            (None, Bandwidth::Auto, Bandwidth::Fullband, false),
            (Some(ForcedMode::SilkOnly), Bandwidth::Auto, Bandwidth::Wideband, false),
            (Some(ForcedMode::CeltOnly), Bandwidth::Wideband, Bandwidth::Fullband, false),
            (Some(ForcedMode::Hybrid), Bandwidth::Auto, Bandwidth::Fullband, channels == 1),
        ] {
            let setup = |enc: &mut OpusEncoder| {
                enc.bitrate_bps = 28000 * channels as i32;
                enc.complexity = 7;
                enc.use_cbr = rate == 16000;
                enc.use_inband_fec = true;
                enc.packet_loss_perc = 10;
                enc.set_force_mode(mode);
                enc.set_bandwidth(bandwidth);
                enc.set_max_bandwidth(max).unwrap();
                enc.set_lfe(lfe);
            };
            let pcm = speech(rate, channels, 1.0);
            let frame = rate / 50;
            let (first, second) = pcm.split_at(pcm.len() / 2);
            let mut used = OpusEncoder::new(rate as i32, channels, app).unwrap();
            setup(&mut used);
            encode_all(&mut used, first, frame);
            used.reset_state();
            assert_eq!(used.final_range(), 0);
            assert_eq!((used.sampling_rate(), used.channels(), used.lfe()), (rate as i32, channels, lfe));
            let mut fresh = OpusEncoder::new(rate as i32, channels, app).unwrap();
            setup(&mut fresh);
            assert_eq!(
                encode_all(&mut used, second, frame),
                encode_all(&mut fresh, second, frame),
                "{rate} Hz, {channels} ch, {app:?}, {mode:?}, {bandwidth:?}, maximum {max:?}, LFE {lfe}"
            );
        }
    }
}

/// `lookahead` is `OPUS_GET_LOOKAHEAD`, for every sampling rate and
/// application.
#[test]
fn lookahead_matches_libopus() {
    for rate in [8000usize, 12000, 16000, 24000, 48000] {
        for channels in [1usize, 2] {
            for (app_name, c_app, app) in APPS {
                let mut c_enc = CEncoder::new(rate, channels, c_app);
                let enc = OpusEncoder::new(rate as i32, channels, app).unwrap();
                assert_eq!(enc.lookahead(), c_enc.lookahead() as usize, "{rate} Hz, {app_name}");
                assert_eq!((enc.sampling_rate(), enc.channels()), (rate as i32, channels));
            }
        }
    }
}
