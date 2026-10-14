//! What the opus-rs multistream encoder puts into a packet, against the C
//! libopus encoder under the same settings.
//!
//! - Routing: every input channel comes out of the C decoder on its own
//!   output channel.
//! - Stream structure in surround mode: the coupled streams are CELT stereo,
//!   the LFE stream is CELT narrowband mono, and every stream has the audio
//!   bandwidth of the C encoder's stream. A stream that is left too few
//!   bytes for audio is without audio in the C packet too, and has the same
//!   TOC byte.
//! - Ambisonics: every stream is CELT.
//! - Bitrates: every stream encoder is set to the bitrate the C encoder sets
//!   on its stream encoder, and `bitrate()` equals `OPUS_GET_BITRATE`.
//! - Constant bitrate: every packet has exactly the size of the C packet.
#![cfg(feature = "std")]

mod ms_support;
mod multistream_enc_common;

use ms_support::Bandwidth as TocBandwidth;
use ms_support::*;
use multistream_enc_common::*;
use opus_rs::Application;
use opus_rs::multistream::encoder::{Bitrate, MultistreamEncoder};
use opus_rs::multistream::layout::MappingType;

/// The output channel `j` of a decoder with the layout of `case` carries
/// this input channel, or `None` when it is silent.
///
/// A generic layout can name one stream on two channels. The encoder reads
/// the first of them and the decoder writes the stream to both.
fn source_channel(case: &Case, j: usize) -> Option<usize> {
    match case.kind {
        Kind::Family(_) => Some(j),
        Kind::Generic(_, _, mapping) => {
            if mapping[j] == 255 {
                None
            } else {
                mapping.iter().position(|&m| m == mapping[j])
            }
        }
    }
}

#[test]
fn every_channel_reaches_its_own_output() {
    let fs = 48_000;
    for case in CASES {
        for frame_size in [fs as usize / 400, fs as usize / 50, 3 * fs as usize / 50] {
            let mut enc = rust_encoder(case, fs, Application::Audio);
            // A 2.5 ms frame resolves 400 Hz and the tones are 150 Hz
            // apart. Separating them takes more bits than in a long frame;
            // at 48 kb/s per channel the C encoder does not manage either.
            let per_channel = if frame_size < fs as usize / 100 {
                96_000
            } else {
                48_000
            };
            enc.set_bitrate(Bitrate::BitsPerSecond(per_channel * case.channels as i32))
                .unwrap();
            let mut dec = c_decoder(&enc);
            let family = case.tone_family();
            let pcm = tones(fs, case.channels, family, fs as usize);
            let packets = rust_encode_all(&mut enc, &pcm, frame_size);
            let out = dec.decode_all(&packets);
            assert_eq!(
                out.len(),
                packets.len() * frame_size * case.channels,
                "{}: decoded length",
                case.name
            );

            let skip = fs as usize / 5;
            let dominant = dominant_tones(&out, fs, case.channels, family, skip);
            let lfe = lfe_channel(case.channels, family);
            for (j, &(tone, ratio)) in dominant.iter().enumerate() {
                let label = format!("{} {} output channel {j}", case.name, ms(frame_size, fs));
                // The LFE stream has about 6 bytes for a 2.5 ms frame at
                // this bitrate. Its 80 Hz tone barely stands out then, with
                // the C encoder as well.
                if lfe == Some(j) && frame_size < fs as usize / 100 {
                    continue;
                }
                match source_channel(case, j) {
                    Some(want) => {
                        assert_eq!(tone, want, "{label}: carries the tone of channel {tone}");
                        // A channel that two inputs could have fed holds one
                        // tone only: the other input was not coded.
                        assert!(
                            ratio >= 10.0,
                            "{label}: its tone is only {ratio:.1} times the next"
                        );
                        // The tone comes back at its level. The level of
                        // the LFE channel is a matter of quality and is
                        // measured in tests/multistream_enc_quality.rs.
                        if lfe == Some(j) {
                            continue;
                        }
                        let x = channel(&out, case.channels, j);
                        let power =
                            goertzel_power(&x[skip..], tone_hz(want, case.channels, family), fs);
                        let nominal = (TONE_AMPLITUDE * TONE_AMPLITUDE / 2.0) as f64;
                        assert!(
                            (power / nominal - 1.0).abs() < 0.2,
                            "{label}: tone power {power:.5}, input {nominal:.5}"
                        );
                    }
                    None => {
                        let x = channel(&out, case.channels, j);
                        assert!(
                            x.iter().all(|&v| v == 0.0),
                            "{label}: a channel that is not coded must be silent"
                        );
                    }
                }
            }
        }
    }
}

/// The audio bandwidth libopus gives all streams in surround mode
/// (opus_multistream_encoder.c, `equiv_rate`), from the total bitrate.
fn surround_bandwidth(
    bitrate: i32,
    channels: usize,
    frame_size: usize,
    fs: i32,
) -> opus_rs::Bandwidth {
    let channels = channels as i32;
    let mut equiv_rate = bitrate;
    if frame_size as i32 * 50 < fs {
        equiv_rate -= 60 * (fs / frame_size as i32 - 50) * channels;
    }
    if equiv_rate > 10_000 * channels {
        opus_rs::Bandwidth::Fullband
    } else if equiv_rate > 7_000 * channels {
        opus_rs::Bandwidth::Superwideband
    } else if equiv_rate > 5_000 * channels {
        opus_rs::Bandwidth::Wideband
    } else {
        opus_rs::Bandwidth::Narrowband
    }
}

/// The bandwidth a CELT stream signals when it is asked for `bandwidth` at
/// the sampling rate `fs`: nothing above what the sampling rate carries.
fn celt_toc_bandwidth(bandwidth: opus_rs::Bandwidth, fs: i32) -> TocBandwidth {
    let asked = match bandwidth {
        opus_rs::Bandwidth::Narrowband => TocBandwidth::Nb,
        opus_rs::Bandwidth::Mediumband | opus_rs::Bandwidth::Wideband => TocBandwidth::Wb,
        opus_rs::Bandwidth::Superwideband => TocBandwidth::Swb,
        _ => TocBandwidth::Fb,
    };
    let widest = match fs {
        8_000 => TocBandwidth::Nb,
        12_000 | 16_000 => TocBandwidth::Wb,
        24_000 => TocBandwidth::Swb,
        _ => TocBandwidth::Fb,
    };
    asked.min(widest)
}

/// Checks the streams of one surround packet. `c_packet` is the packet the C
/// encoder wrote for the same frame, or `None` when there is none to
/// compare with. Returns what is wrong.
fn check_surround_packet(
    enc: &MultistreamEncoder,
    packet: &[u8],
    c_packet: Option<&[u8]>,
    bandwidth: opus_rs::Bandwidth,
    frame_size: usize,
) -> Vec<String> {
    let fs = enc.sampling_rate();
    let streams = enc.layout().streams();
    let coupled = enc.layout().coupled_streams();
    let mut wrong = Vec::new();
    let parts = split_streams(packet, streams).expect("opus-rs packet");
    let c_parts = c_packet.map(|p| split_streams(p, streams).expect("C packet"));
    for (s, part) in parts.iter().enumerate() {
        let t = toc(part.toc);
        if part.samples(fs) != frame_size {
            wrong.push(format!("stream {s}: {} samples", part.samples(fs)));
        }
        let c = c_parts.as_ref().map(|c_parts| &c_parts[s]);
        // A stream that is left too few bytes carries no audio. Its TOC byte
        // then names the mode and bandwidth of the last frame the stream
        // coded, or Hybrid fullband when it has coded none
        // (opus_encoder.c:1344-1385). What the frame was set to does not
        // show, so such a stream is only compared with the C stream: that
        // one is without audio too and has the same TOC byte.
        if part.payload_len == 0 {
            if let Some(c) = c
                && (c.payload_len != 0 || c.toc != part.toc)
            {
                wrong.push(format!(
                    "stream {s} without audio is {t}, C has {} with {} bytes",
                    toc(c.toc),
                    c.payload_len
                ));
            }
            continue;
        }
        if s < coupled {
            // To keep the spatial image: CELT, two channels, and the
            // bandwidth of the whole layout.
            let want = celt_toc_bandwidth(bandwidth, fs);
            if t.mode != Mode::Celt || !t.stereo || t.bandwidth != want {
                wrong.push(format!(
                    "coupled stream {s} is {t}, expected CELT stereo {want:?}"
                ));
            }
        } else if Some(s) == enc.lfe_stream() {
            if t.mode != Mode::Celt || t.stereo || t.bandwidth != TocBandwidth::Nb {
                wrong.push(format!(
                    "LFE stream {s} is {t}, expected CELT narrowband mono"
                ));
            }
        } else if t.stereo {
            wrong.push(format!("mono stream {s} is {t}"));
        }
        // Against the C encoder.
        if let Some(c) = c {
            let ct = toc(c.toc);
            if c.payload_len == 0 {
                wrong.push(format!("stream {s} is {t}, C has no audio"));
                continue;
            }
            if (t.bandwidth, t.stereo) != (ct.bandwidth, ct.stereo) {
                wrong.push(format!("stream {s} is {t}, C has {ct}"));
            }
            // The mode of a plain mono stream is the stream encoder's own
            // choice. libopus listens to the signal for it; opus-rs does
            // not. The forced streams must agree.
            if (s < coupled || Some(s) == enc.lfe_stream()) && t.mode != ct.mode {
                wrong.push(format!("stream {s} is {t}, C has {ct}"));
            }
        }
    }
    wrong
}

#[test]
fn surround_streams_have_the_structure_of_libopus() {
    let mut wrong = Vec::new();
    let mut packets = 0;
    // Streams without audio. They are compared with C byte for byte.
    let mut empty_streams = 0;
    for fs in [48_000, 24_000, 16_000] {
        for channels in 3..=8usize {
            let case = Case::family("surround", channels, 1);
            for per_channel in [2_000, 5_500, 8_000, 12_000, 64_000] {
                let bitrate = per_channel * channels as i32;
                for frame_size in frame_sizes(fs) {
                    let (mut rust, mut c) = encoders(&case, fs, Application::Audio);
                    assert_eq!(rust.mapping_type(), MappingType::Surround);
                    assert_eq!(rust.lfe_stream(), (channels >= 6).then(|| c.streams - 1));
                    configure(&mut rust, &mut c, bitrate, true);
                    let samples = fs as usize / 4;
                    let pcm = if per_channel == 8_000 {
                        tones(fs, channels, 1, samples)
                    } else {
                        voiced(fs, channels, samples)
                    };
                    let bandwidth = surround_bandwidth(bitrate, channels, frame_size, fs);
                    let rust_packets = rust_encode_all(&mut rust, &pcm, frame_size);
                    let c_packets = c.encode_all(&pcm, frame_size);
                    for (f, (r, cp)) in rust_packets.iter().zip(&c_packets).enumerate() {
                        packets += 1;
                        empty_streams += split_streams(r, c.streams)
                            .unwrap()
                            .iter()
                            .filter(|part| part.payload_len == 0)
                            .count();
                        for w in check_surround_packet(&rust, r, Some(cp), bandwidth, frame_size) {
                            wrong.push(format!(
                                "{fs} Hz, {channels} channels, {bitrate} b/s, {}, frame {f}: {w}",
                                ms(frame_size, fs)
                            ));
                        }
                    }
                }
            }
        }
    }
    println!(
        "{packets} packets, {empty_streams} streams without audio, {} differences",
        wrong.len()
    );
    for w in wrong.iter().take(30) {
        println!("{w}");
    }
    assert!(wrong.is_empty(), "{} differences", wrong.len());
    // The settings must keep reaching streams without audio: 4785 of them
    // when this was written, most of them LFE streams at a low bitrate or
    // with a short frame.
    assert!(empty_streams > 4000, "{empty_streams} streams without audio");
}

/// With an automatic or a maximum bitrate libopus codes every stream of a
/// surround layout narrowband: it compares the raw setting, -1000 or -1,
/// with the bandwidth thresholds. opus-rs uses the bitrate the allocation
/// chose (difference D1 of the design).
#[test]
fn automatic_bitrate_in_surround_mode_is_not_narrowband() {
    let fs = 48_000;
    let frame_size = 960;
    for (channels, setting) in [(6, OPUS_AUTO), (8, OPUS_AUTO), (6, OPUS_BITRATE_MAX)] {
        let case = Case::family("surround", channels, 1);
        let (mut rust, mut c) = encoders(&case, fs, Application::Audio);
        configure(&mut rust, &mut c, setting, true);
        let pcm = voiced(fs, channels, 10 * frame_size);
        let rust_packets = rust_encode_all(&mut rust, &pcm, frame_size);
        let c_packets = c.encode_all(&pcm, frame_size);
        // The allocation is the same in both encoders.
        assert_eq!(rust.bitrate(), c.bitrate(), "{channels} channels: bitrate");
        let allocated: i32 = (0..c.streams)
            .map(|s| rust.stream_encoder(s).unwrap().bitrate_bps)
            .sum();
        let bandwidth = surround_bandwidth(allocated, channels, frame_size, fs);
        assert_eq!(bandwidth, opus_rs::Bandwidth::Fullband);
        for (f, (r, cp)) in rust_packets.iter().zip(&c_packets).enumerate() {
            let wrong = check_surround_packet(&rust, r, None, bandwidth, frame_size);
            assert!(
                wrong.is_empty(),
                "{channels} channels, frame {f}: {wrong:?}"
            );
            // The libopus behaviour this departs from. A stream without
            // audio does not show it.
            for (s, part) in split_streams(cp, c.streams).unwrap().iter().enumerate() {
                if part.payload_len != 0 {
                    assert_eq!(
                        toc(part.toc).bandwidth,
                        TocBandwidth::Nb,
                        "{channels} channels, frame {f}: C stream {s} is no longer narrowband; \
                         check whether D1 is still a difference"
                    );
                }
            }
        }
    }
}

#[test]
fn ambisonics_streams_are_celt() {
    for fs in [48_000, 24_000, 16_000] {
        for channels in [1usize, 4, 6, 9, 11] {
            let case = Case::family("ambisonics", channels, 2);
            for per_channel in [6_000, 32_000] {
                for frame_size in frame_sizes(fs) {
                    let (mut rust, mut c) = encoders(&case, fs, Application::Audio);
                    assert_eq!(rust.mapping_type(), MappingType::Ambisonics);
                    assert_eq!(rust.lfe_stream(), None);
                    configure(&mut rust, &mut c, per_channel * channels as i32, true);
                    let pcm = voiced(fs, channels, fs as usize / 5);
                    let rust_packets = rust_encode_all(&mut rust, &pcm, frame_size);
                    let c_packets = c.encode_all(&pcm, frame_size);
                    for (f, (r, cp)) in rust_packets.iter().zip(&c_packets).enumerate() {
                        let label = format!(
                            "{fs} Hz, {channels} channels, {per_channel} b/s per channel, {}, \
                             frame {f}",
                            ms(frame_size, fs)
                        );
                        let parts = split_streams(r, c.streams).unwrap();
                        let c_parts = split_streams(cp, c.streams).unwrap();
                        for (s, (part, c_part)) in parts.iter().zip(&c_parts).enumerate() {
                            let (t, ct) = (toc(part.toc), toc(c_part.toc));
                            assert_eq!(t.mode, Mode::Celt, "{label}: stream {s} is {t}");
                            assert_eq!(t.stereo, s < c.coupled, "{label}: stream {s} is {t}");
                            assert_eq!(part.samples(fs), frame_size, "{label}: stream {s}");
                            if c_part.payload_len == 0 {
                                continue;
                            }
                            // The bandwidth is each stream encoder's own
                            // choice here, so it is not compared. libopus
                            // can also code the stereo stream as mono at a
                            // low bitrate; opus-rs codes it as stereo.
                            assert_eq!(ct.mode, Mode::Celt, "{label}: C stream {s} is {ct}");
                        }
                    }
                }
            }
        }
    }
}

/// What a C stream encoder reports for `OPUS_GET_BITRATE` when it was set to
/// `rate` and its last packet has frames of `coded_frame` samples: at most
/// what 1276 bytes per frame carry (opus_encoder.c,
/// `user_bitrate_to_bitrate`).
fn reported(rate: i32, fs: i32, coded_frame: usize) -> i32 {
    let cap = (1276i64 * 8 * (6 * fs as i64 / coded_frame as i64) / 6) as i32;
    rate.min(cap)
}

#[test]
fn stream_bitrates_are_those_of_libopus() {
    let mut frames = 0;
    for fs in [48_000, 24_000, 16_000] {
        for case in CASES {
            let channels = case.channels as i32;
            let settings = [
                3_000 * channels,
                8_000 * channels,
                24_000 * channels,
                64_000 * channels,
                300_000 * channels,
                OPUS_AUTO,
                OPUS_BITRATE_MAX,
            ];
            for bitrate in settings {
                for frame_size in frame_sizes(fs) {
                    let (mut rust, mut c) = encoders(case, fs, Application::Audio);
                    configure(&mut rust, &mut c, bitrate, true);
                    let label =
                        format!("{} {fs} Hz {bitrate} b/s {}", case.name, ms(frame_size, fs));
                    let pcm = voiced(fs, case.channels, 3 * frame_size);
                    let mut out = vec![0u8; max_packet_bytes(c.streams)];
                    for (f, frame) in pcm.chunks_exact(frame_size * case.channels).enumerate() {
                        let n = rust.encode(frame, frame_size, &mut out).unwrap();
                        let c_packet = c.encode_float(frame, frame_size, out.len()).unwrap();
                        let parts = split_streams(&out[..n], c.streams).unwrap();
                        let c_parts = split_streams(&c_packet, c.streams).unwrap();
                        let mut same_frames = true;
                        for s in 0..c.streams {
                            let set = rust.stream_encoder(s).unwrap().bitrate_bps;
                            // What libopus reports depends on the frames of
                            // its own packet.
                            let c_frame = c_parts[s].samples(fs) / c_parts[s].frames();
                            assert_eq!(
                                reported(set, fs, c_frame),
                                c.stream_bitrate(s),
                                "{label}: frame {f}: stream {s}"
                            );
                            let frame = parts[s].samples(fs) / parts[s].frames();
                            same_frames &= frame == c_frame
                                || reported(set, fs, frame) == reported(set, fs, c_frame);
                        }
                        // The sum is the same whenever both encoders split a
                        // long frame the same way. They can differ in that
                        // (SILK codes 60 ms as one frame, CELT as three), and
                        // the limit of a stream depends on it.
                        if same_frames {
                            assert_eq!(
                                rust.bitrate(),
                                c.bitrate(),
                                "{label}: frame {f}: bitrate()"
                            );
                        }
                        frames += 1;
                    }
                }
            }
        }
    }
    println!("{frames} frames compared");
}

#[test]
fn bitrate_before_the_first_frame_is_that_of_libopus() {
    for fs in [48_000, 24_000, 16_000, 12_000, 8_000] {
        for case in CASES {
            for application in [
                Application::Audio,
                Application::Voip,
                Application::RestrictedLowDelay,
            ] {
                let (mut rust, mut c) = encoders(case, fs, application);
                assert_eq!(
                    rust.bitrate(),
                    c.bitrate(),
                    "{} {fs} Hz: new encoder",
                    case.name
                );
                assert_eq!(
                    rust.lookahead() as i32,
                    c.lookahead(),
                    "{} {fs} Hz: lookahead",
                    case.name
                );
                // A bitrate that is set but not yet used does not show.
                configure(&mut rust, &mut c, 20_000 * case.channels as i32, true);
                assert_eq!(
                    rust.bitrate(),
                    c.bitrate(),
                    "{} {fs} Hz: after set_bitrate",
                    case.name
                );
                // After a reset the streams keep the bitrate of the last
                // frame, without the limit of its frame size.
                let frame_size = 3 * fs as usize / 50;
                let pcm = voiced(fs, case.channels, frame_size);
                let mut out = vec![0u8; max_packet_bytes(c.streams)];
                configure(&mut rust, &mut c, OPUS_BITRATE_MAX, true);
                rust.encode(&pcm, frame_size, &mut out).unwrap();
                c.encode_float(&pcm, frame_size, out.len()).unwrap();
                rust.reset_state();
                assert_eq!(c.try_ctl(audiopus_sys::OPUS_RESET_STATE, 0), 0);
                assert_eq!(
                    rust.bitrate(),
                    c.bitrate(),
                    "{} {fs} Hz: after reset",
                    case.name
                );
            }
        }
    }
}

#[test]
fn constant_bitrate_packets_have_the_size_of_libopus() {
    let mut packets = 0;
    for fs in [48_000, 24_000, 16_000] {
        for case in CASES {
            let channels = case.channels as i32;
            for bitrate in [
                8_000 * channels,
                24_000 * channels,
                64_000 * channels,
                OPUS_AUTO,
            ] {
                for frame_size in frame_sizes(fs) {
                    let (mut rust, mut c) = encoders(case, fs, Application::Audio);
                    configure(&mut rust, &mut c, bitrate, false);
                    let label =
                        format!("{} {fs} Hz {bitrate} b/s {}", case.name, ms(frame_size, fs));
                    let pcm = voiced(fs, case.channels, 6 * frame_size.max(fs as usize / 100));
                    let rust_packets = rust_encode_all(&mut rust, &pcm, frame_size);
                    let c_packets = c.encode_all(&pcm, frame_size);
                    for (f, (r, cp)) in rust_packets.iter().zip(&c_packets).enumerate() {
                        assert_eq!(r.len(), cp.len(), "{label}: frame {f}: packet size");
                        if bitrate != OPUS_AUTO {
                            // bitrate * duration / 8, rounded to a byte, but
                            // never below the smallest packet.
                            let bits = bitrate as i64 * frame_size as i64 / fs as i64;
                            let nominal = ((bits + 4) / 8).max(2 * c.streams as i64 - 1);
                            assert_eq!(r.len() as i64, nominal, "{label}: frame {f}");
                        }
                        packets += 1;
                    }
                }
            }
        }
    }
    println!("{packets} packets compared");
}

/// The last stream of a constant-bitrate packet is set to the bitrate that
/// fills what the other streams left, and is padded to the byte.
#[test]
fn constant_bitrate_gives_the_last_stream_what_is_left() {
    let fs = 48_000;
    let frame_size = 960;
    for case in CASES.iter().filter(|c| c.channels > 2) {
        let (mut rust, mut c) = encoders(case, fs, Application::Audio);
        let bitrate = 20_000 * case.channels as i32;
        configure(&mut rust, &mut c, bitrate, false);
        let streams = c.streams;
        let pcm = voiced(fs, case.channels, 20 * frame_size);
        let total = (bitrate / 400) as usize;
        for (f, packet) in rust_encode_all(&mut rust, &pcm, frame_size)
            .iter()
            .enumerate()
        {
            assert_eq!(packet.len(), total, "{}: frame {f}", case.name);
            let parts = split_streams(packet, streams).unwrap();
            let before: usize = parts[..streams - 1].iter().map(|p| p.used).sum();
            let last = &parts[streams - 1];
            assert_eq!(before + last.used, total, "{}: frame {f}", case.name);
            // 400 bits per second for every byte of a 20 ms frame.
            let left = (total - before) as i32;
            let channels = if streams == c.coupled { 2 } else { 1 };
            assert_eq!(
                rust.stream_encoder(streams - 1).unwrap().bitrate_bps,
                (left * 400).clamp(500, 750_000 * channels),
                "{}: frame {f}: bitrate of the last stream",
                case.name
            );
            // Streams before the last carry no padding: all of it is in the
            // last stream.
            for (s, part) in parts[..streams - 1].iter().enumerate() {
                assert_eq!(part.padding, 0, "{}: frame {f}: stream {s}", case.name);
            }
        }
    }
}
