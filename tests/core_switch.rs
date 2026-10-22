//! `OpusEncoder` controls in the cases `tests/core_encoder.rs` leaves out:
//! settings changed while the stream runs, 16-bit input, and constant
//! bitrate. The reference is the C libopus encoder under the same settings.

mod core_common;
mod k24_common;

use std::num::NonZeroU32;
use core_common::*;
use k24_common::{aligned_snr, multitone, speech, to_stereo, toc_summary};
use opus1::{
    Application, Bandwidth, Bitrate, Channels, ForcedMode, OpusDecoder, OpusEncoder, SampleRate,
};
use opus1::packet::Framing;

const MODES: [(&str, i32, ForcedMode); 3] = [
    ("SILK", MODE_SILK, ForcedMode::SilkOnly),
    ("Hybrid", MODE_HYBRID, ForcedMode::Hybrid),
    ("CELT", MODE_CELT, ForcedMode::CeltOnly),
];

/// Music-like tones on top of speech, mono or stereo: a signal with content
/// in every band, unlike plain speech.
fn tones_and_speech(rate: usize, channels: usize, seconds: f64) -> Vec<f32> {
    let voice = speech(rate, seconds);
    let tones = multitone(rate, voice.len());
    let mono: Vec<f32> = voice.iter().zip(&tones).map(|(v, t)| 0.6 * v + 0.3 * t).collect();
    if channels == 2 { to_stereo(&mono, rate) } else { mono }
}

fn to_i16(pcm: &[f32]) -> Vec<i16> {
    pcm.iter().map(|&v| (v * 32768.0).round().clamp(-32768.0, 32767.0) as i16).collect()
}

/// The forced mode changes every half second, through every order of two
/// modes. For every packet:
///
/// - the C decoder and the opus1 decoder decode it, and both end in the
///   encoder's final range;
/// - the TOC byte is libopus's. That includes the first packet after a
///   switch to CELT-only, which both encoders still code in the old mode,
///   with a 5 ms CELT frame appended (the "redundancy" frame).
///
/// Mono only: stereo Hybrid packets have a known fault, see
/// `tests/core_encoder.rs`.
#[test]
fn mode_changes_in_a_running_stream_keep_the_final_range() {
    // Every mode follows every other mode once.
    let schedule = [0usize, 2, 1, 2, 0, 1, 0];
    for (rate, bitrate) in [(48000usize, 32000), (24000, 28000), (16000, 24000)] {
        let pcm = tones_and_speech(rate, 1, 3.5);
        let frame = rate / 50;
        let mut c_enc = CEncoder::new(rate, 1, APP_AUDIO);
        c_enc.bitrate(bitrate).complexity(5);
        let mut enc = OpusEncoder::new(
            SampleRate::try_from(rate as i32).unwrap(),
            Channels::Mono,
            Application::Audio,
        )
        .unwrap();
        enc.set_bitrate(Bitrate::BitsPerSecond(NonZeroU32::new(bitrate as u32).unwrap()));
        enc.set_complexity(5).unwrap();
        let mut c_dec = CDecoder::new(rate, 1);
        let mut dec =
            OpusDecoder::new(SampleRate::try_from(rate as i32).unwrap(), Channels::Mono).unwrap();
        let mut buf = vec![0u8; 4000];
        let mut out = vec![0f32; frame];
        let (mut packets, mut c_packets) = (Vec::new(), Vec::new());
        for (i, input) in pcm.chunks_exact(frame).enumerate() {
            let step = (i / 25) % schedule.len();
            let (_, c_mode, mode) = MODES[schedule[step]];
            let switched = i % 25 == 0;
            if switched {
                c_enc.force_mode(c_mode);
                enc.set_force_mode(Some(mode));
            }
            let c_packet = c_enc.encode(input);
            let n = enc.encode(input, frame, &mut buf).unwrap();
            let what = format!("{rate} Hz, packet {i}");
            c_dec.decode(Some(&buf[..n]), frame, false).unwrap_or_else(|e| panic!("{what}: libopus error {e}"));
            assert_eq!(dec.decode(&buf[..n], frame, &mut out), Ok(frame), "{what}");
            assert_ne!(enc.final_range(), 0, "{what}");
            assert_eq!(enc.final_range(), c_dec.final_range(), "{what}: final range, C decoder");
            assert_eq!(enc.final_range(), dec.final_range(), "{what}: final range, opus1 decoder");
            assert_eq!(buf[0] & 0xFC, c_packet[0] & 0xFC, "{what}: TOC");
            packets.push(buf[..n].to_vec());
            c_packets.push(c_packet);
        }
        println!("{rate} Hz: {} packets\n  libopus [{}]\n  opus1 [{}]", packets.len(), toc_summary(&c_packets), toc_summary(&packets));
        assert!(packets.len() >= 170, "{rate} Hz");
    }
}

/// The LFE flag, the maximum bandwidth and the forced bandwidth switched on
/// and off while the stream runs, with the mode left to the encoder: the
/// packets stay decodable, the final range is the decoders', and the stream
/// uses the same packet configurations as libopus's.
#[test]
fn bandwidth_limits_and_lfe_changed_in_a_running_stream() {
    let rate = 48000;
    let frame = 960;
    let pcm = tones_and_speech(rate, 1, 3.6);
    let mut c_enc = CEncoder::new(rate, 1, APP_AUDIO);
    c_enc.bitrate(40000).complexity(5);
    let mut enc = OpusEncoder::new(
        SampleRate::try_from(rate as i32).unwrap(),
        Channels::Mono,
        Application::Audio,
    )
    .unwrap();
    enc.set_bitrate(Bitrate::BitsPerSecond(NonZeroU32::new(40000).unwrap()));
    enc.set_complexity(5).unwrap();
    let mut c_dec = CDecoder::new(rate, 1);
    let mut buf = vec![0u8; 4000];
    let (mut packets, mut c_packets) = (Vec::new(), Vec::new());
    for (i, input) in pcm.chunks_exact(frame).enumerate() {
        if i % 30 == 0 {
            match (i / 30) % 6 {
                0 => {
                    c_enc.lfe(false).bandwidth(audiopus_sys::OPUS_AUTO).max_bandwidth(BW_FB);
                    enc.set_lfe(false);
                    enc.set_bandwidth(None);
                    enc.set_max_bandwidth(Bandwidth::Fullband);
                }
                1 => {
                    c_enc.max_bandwidth(BW_WB);
                    enc.set_max_bandwidth(Bandwidth::Wideband);
                }
                2 => {
                    c_enc.lfe(true);
                    enc.set_lfe(true);
                }
                3 => {
                    c_enc.lfe(false).max_bandwidth(BW_SWB);
                    enc.set_lfe(false);
                    enc.set_max_bandwidth(Bandwidth::Superwideband);
                }
                4 => {
                    // A forced bandwidth is not limited by the maximum.
                    c_enc.bandwidth(BW_FB);
                    enc.set_bandwidth(Some(Bandwidth::Fullband));
                }
                _ => {
                    c_enc.bandwidth(BW_NB);
                    enc.set_bandwidth(Some(Bandwidth::Narrowband));
                }
            }
        }
        c_packets.push(c_enc.encode(input));
        let n = enc.encode(input, frame, &mut buf).unwrap();
        c_dec.decode(Some(&buf[..n]), frame, false).unwrap_or_else(|e| panic!("packet {i}: libopus error {e}"));
        assert_eq!(enc.final_range(), c_dec.final_range(), "packet {i}: final range");
        packets.push(buf[..n].to_vec());
    }
    let differ = packets.iter().zip(&c_packets).filter(|(a, b)| a[0] & 0xFC != b[0] & 0xFC).count();
    println!("{differ} of {} TOC bytes differ\n  libopus [{}]\n  opus1 [{}]", packets.len(), toc_summary(&c_packets), toc_summary(&packets));
    let kinds = |p: &[Vec<u8>]| p.iter().map(|x| x[0] & 0xFC).collect::<std::collections::BTreeSet<u8>>();
    assert_eq!(kinds(&packets), kinds(&c_packets), "packet configurations used");
    assert!(kinds(&packets).len() >= 4, "the stream changes configuration");
    // A change takes effect on the packet libopus changes on, or the one
    // after a switch to or from CELT-only.
    assert!(differ * 100 <= 4 * packets.len(), "{differ} of {} TOC bytes differ", packets.len());
}

/// One forced-mode stream from 16-bit input at a constant bitrate, coded by
/// both encoders and decoded by the C decoder.
struct CbrRun {
    /// TOC byte of the first packet.
    toc: u8,
    /// SNR per channel against the input: opus1 packets, libopus packets.
    snr: Vec<(f64, f64)>,
}

/// Encode `pcm` with `encode_i16` under a forced mode at a constant bitrate.
/// Checks on every packet that the TOC byte and the length are libopus's
/// and that the C decoder ends in the encoder's final range.
fn cbr_run(rate: usize, channels: usize, mode: usize, frame_2_5ms: usize, pcm: &[f32]) -> CbrRun {
    let (name, c_mode, forced) = MODES[mode];
    let frame = rate / 400 * frame_2_5ms;
    let bitrate = [24000, 36000, 56000][mode] * channels as i32;
    let what = format!("{rate} Hz, {channels} ch, forced {name}, {} ms", frame_2_5ms as f32 * 2.5);
    let mut c_enc = CEncoder::new(rate, channels, APP_AUDIO);
    c_enc.force_mode(c_mode).bitrate(bitrate).vbr(false).complexity(5);
    let mut enc = OpusEncoder::new(
        SampleRate::try_from(rate as i32).unwrap(),
        Channels::try_from(channels).unwrap(),
        Application::Audio,
    )
    .unwrap();
    enc.set_force_mode(Some(forced));
    enc.set_bitrate(Bitrate::BitsPerSecond(NonZeroU32::new(bitrate as u32).unwrap()));
    enc.set_vbr(false);
    enc.set_complexity(5).unwrap();
    if c_mode == MODE_SILK && rate > 16000 {
        // SILK-only reaches wideband at most.
        c_enc.max_bandwidth(BW_WB);
        enc.set_max_bandwidth(Bandwidth::Wideband);
    }
    let c_packets = c_enc.encode_all(pcm, frame);
    let pcm16 = to_i16(pcm);
    let mut c_dec = CDecoder::new(rate, channels);
    let mut c_dec_ref = CDecoder::new(rate, channels);
    let mut buf = vec![0u8; 4000];
    let (mut decoded, mut c_decoded) = (Vec::new(), Vec::new());
    let mut packets = Vec::new();
    for (i, input) in pcm16.chunks_exact(frame * channels).enumerate() {
        let n = enc.encode_i16(input, frame, &mut buf).unwrap_or_else(|e| panic!("{what}, packet {i}: {e}"));
        assert_eq!(n, c_packets[i].len(), "{what}, packet {i}: length");
        decoded.extend(c_dec.decode(Some(&buf[..n]), frame, false).unwrap_or_else(|e| panic!("{what}, packet {i}: libopus error {e}")));
        // Stereo Hybrid packets have a known fault, see tests/core_encoder.rs.
        let stereo_hybrid = channels == 2 && buf[0] & 0xE0 == 0x60;
        if !stereo_hybrid {
            assert_eq!(enc.final_range(), c_dec.final_range(), "{what}, packet {i}: final range");
        }
        c_decoded.extend(c_dec_ref.decode(Some(&c_packets[i]), frame, false).unwrap());
        packets.push(buf[..n].to_vec());
    }
    let differ = packets.iter().zip(&c_packets).filter(|(a, b)| a[0] & 0xFC != b[0] & 0xFC).count();
    assert_eq!(differ, 0, "{what}: libopus [{}], opus1 [{}]", toc_summary(&c_packets), toc_summary(&packets));
    let skip = rate / 5;
    let snr = (0..channels)
        .map(|c| {
            let pick = |x: &[f32]| x.iter().skip(c).step_by(channels).copied().collect::<Vec<f32>>();
            let against_input = |x: &[f32]| aligned_snr(&pick(pcm)[skip..], &pick(x)[skip..], (rate / 100) as isize).0;
            (against_input(&decoded), against_input(&c_decoded))
        })
        .collect();
    CbrRun { toc: packets[0][0], snr }
}

/// Every forced mode with 16-bit input (`encode_i16`) at a constant
/// bitrate, for every input rate: mono at 10, 20 and 40 ms, stereo at
/// 20 ms.
///
/// - The TOC byte and the packet length are libopus's on every packet.
/// - The C decoder ends in the encoder's final range.
/// - Where the packets are CELT-only, the quality the C decoder gets is
///   within 2 dB of what it gets from libopus's packets. For packets with a
///   SILK layer see the next test.
#[test]
#[allow(clippy::needless_range_loop)]
fn forced_modes_with_16_bit_input_at_a_constant_bitrate() {
    let (mut streams, mut compared) = (0, 0);
    for rate in [8000usize, 12000, 16000, 24000, 48000] {
        for channels in [1usize, 2] {
            let pcm = tones_and_speech(rate, channels, 0.7);
            for mode in 0..MODES.len() {
                for frame_2_5ms in [4usize, 8, 16] {
                    if channels == 2 && frame_2_5ms != 8 {
                        continue;
                    }
                    let run = cbr_run(rate, channels, mode, frame_2_5ms, &pcm);
                    streams += 1;
                    let celt_only = run.toc & 0x80 != 0;
                    assert_eq!(celt_only, mode == 2, "{rate} Hz, forced {}", MODES[mode].0);
                    if !celt_only {
                        continue;
                    }
                    for (c, (ours, theirs)) in run.snr.iter().enumerate() {
                        assert!(
                            ours > &(theirs - 2.0),
                            "{rate} Hz, {channels} ch, forced CELT, {} ms, channel {c}: opus1 {ours:.1} dB, libopus {theirs:.1} dB",
                            frame_2_5ms as f32 * 2.5
                        );
                    }
                    compared += 1;
                }
            }
        }
    }
    assert_eq!((streams, compared), (60, 20));
}

/// The same streams where the packets have a SILK layer: forced SILK and
/// forced Hybrid.
///
/// The packets have libopus's TOC byte and length, and on this signal
/// (steady tones under speech) the C decoder gets from them within 2 dB of
/// what it gets from libopus's packets.
///
/// This signal is the hard one for the SILK pitch analysis. Until
/// `silk_pitch_analysis_core` searched the lags libopus searches
/// (silk/fixed/pitch_analysis_core_FIX.c), these streams were 2 to 10 dB
/// below libopus.
#[test]
#[allow(clippy::needless_range_loop)]
fn forced_silk_and_hybrid_on_tonal_input_are_within_2_db_of_libopus() {
    let mut failures = Vec::new();
    for rate in [8000usize, 12000, 16000, 24000, 48000] {
        let pcm = tones_and_speech(rate, 1, 0.7);
        for mode in 0..2 {
            for frame_2_5ms in [4usize, 8, 16] {
                let run = cbr_run(rate, 1, mode, frame_2_5ms, &pcm);
                let (ours, theirs) = run.snr[0];
                if ours <= theirs - 2.0 {
                    failures.push(format!("{rate} Hz, forced {}, {} ms: opus1 {ours:.1} dB, libopus {theirs:.1} dB", MODES[mode].0, frame_2_5ms as f32 * 2.5));
                }
            }
        }
    }
    println!("{}", failures.join("\n"));
    assert!(failures.is_empty(), "{} streams with a SILK layer are more than 2 dB below libopus", failures.len());
}

/// The bitrate drops, again and again, so low that a packet cannot code
/// anything (opus_encoder.c:1340-1406). libopus then writes a placeholder
/// that a decoder conceals. Its first byte names the mode and bandwidth of
/// the last packet that was coded, or Hybrid and fullband when none was,
/// and the placeholder leaves the encoder's state alone.
///
/// Every placeholder is libopus's byte for byte, with a variable and a
/// constant bitrate. The packets in between have libopus's first byte.
#[test]
fn packets_too_small_to_code_name_the_last_coded_mode() {
    // (name, mode, bandwidth, LFE)
    let streams = [
        ("SILK-WB", Some((MODE_SILK, ForcedMode::SilkOnly)), Some((BW_WB, Bandwidth::Wideband)), false),
        ("SILK-NB", Some((MODE_SILK, ForcedMode::SilkOnly)), Some((BW_NB, Bandwidth::Narrowband)), false),
        ("Hybrid-SWB", Some((MODE_HYBRID, ForcedMode::Hybrid)), Some((BW_SWB, Bandwidth::Superwideband)), false),
        ("Hybrid-FB", Some((MODE_HYBRID, ForcedMode::Hybrid)), Some((BW_FB, Bandwidth::Fullband)), false),
        ("CELT-WB", Some((MODE_CELT, ForcedMode::CeltOnly)), Some((BW_WB, Bandwidth::Wideband)), false),
        ("CELT-FB", Some((MODE_CELT, ForcedMode::CeltOnly)), Some((BW_FB, Bandwidth::Fullband)), false),
        ("LFE", None, None, true),
    ];
    let rate = 48000usize;
    let pcm = tones_and_speech(rate, 1, 2.4);
    let mut placeholders = std::collections::BTreeSet::new();
    for (name, mode, bandwidth, lfe) in streams {
        for frame_tenth_ms in [25usize, 100, 200, 600] {
            let frame = rate * frame_tenth_ms / 10_000;
            for cbr in [false, true] {
                // The low rate starts the stream in half of the runs, so
                // that placeholders also come before any coded packet.
                for low_first in [false, true] {
                    let what = format!("{name}, {} ms, CBR {cbr}, low bitrate first {low_first}", frame_tenth_ms as f64 / 10.0);
                    let mut c_enc = CEncoder::new(rate, 1, APP_AUDIO);
                    c_enc.vbr(!cbr).complexity(5).lfe(lfe);
                    let mut enc = OpusEncoder::new(
                        SampleRate::try_from(rate as i32).unwrap(),
                        Channels::Mono,
                        Application::Audio,
                    )
                    .unwrap();
                    enc.set_vbr(!cbr);
                    enc.set_complexity(5).unwrap();
                    enc.set_lfe(lfe);
                    if let Some((c_mode, forced)) = mode {
                        c_enc.force_mode(c_mode);
                        enc.set_force_mode(Some(forced));
                    }
                    if let Some((c_bw, bw)) = bandwidth {
                        c_enc.bandwidth(c_bw);
                        enc.set_bandwidth(Some(bw));
                    }
                    let mut buf = vec![0u8; 4000];
                    let (mut seen, mut coded) = (0, 0);
                    for (i, input) in pcm.chunks_exact(frame).enumerate() {
                        // 200 ms at a rate that codes, 100 ms at 600 b/s.
                        let ms = i * frame_tenth_ms / 10;
                        let low = (ms % 300 >= 200) != low_first;
                        let bitrate = if low { 600 } else { 32000 };
                        c_enc.bitrate(bitrate);
                        enc.set_bitrate(Bitrate::BitsPerSecond(NonZeroU32::new(bitrate as u32).unwrap()));
                        let c_packet = c_enc.encode(input);
                        let n = enc.encode(input, frame, &mut buf).unwrap_or_else(|e| panic!("{what}, packet {i}: {e}"));
                        let packet = &buf[..n];
                        let parsed = opus1::packet::parse(&c_packet, Framing::Normal).unwrap();
                        if parsed.frame_sizes().iter().all(|&size| size == 0) {
                            assert!(low, "{what}, packet {i}: a placeholder at 32 kb/s");
                            assert_eq!(packet, &c_packet[..], "{what}, packet {i}: placeholder");
                            assert_eq!(enc.final_range(), 0, "{what}, packet {i}");
                            placeholders.insert(c_packet[0] & 0xFC);
                            seen += 1;
                        } else {
                            assert!(!low, "{what}, packet {i}: data at 600 b/s");
                            assert_eq!(packet[0] & 0xFC, c_packet[0] & 0xFC, "{what}, packet {i}: first byte");
                            coded += 1;
                        }
                    }
                    assert!(seen >= 3 && coded >= 6, "{what}: {seen} placeholders, {coded} coded packets");
                }
            }
        }
    }
    // Placeholders of many kinds were compared: SILK, Hybrid and CELT ones.
    assert!(placeholders.len() >= 12, "{} kinds of placeholder", placeholders.len());
}

/// The frame size changes together with the bitrate: 20 ms packets that
/// code, then frames of 2.5 or 5 ms at a bitrate too low to code. The
/// placeholders are CELT-only, the only mode with such frames, at the
/// bandwidth of the last coded packet. CELT has no mediumband, and a
/// placeholder after mediumband SILK packets says narrowband
/// (opus_encoder.c:1381-1382). Every placeholder is libopus's.
#[test]
fn placeholders_in_short_frames_after_longer_coded_packets() {
    let rate = 48000usize;
    let pcm = tones_and_speech(rate, 1, 1.0);
    for (name, c_mode, forced, c_bw, bw, placeholder) in [
        ("SILK-MB", MODE_SILK, ForcedMode::SilkOnly, BW_MB, Bandwidth::Mediumband, "CELT-NB"),
        ("SILK-NB", MODE_SILK, ForcedMode::SilkOnly, BW_NB, Bandwidth::Narrowband, "CELT-NB"),
        ("SILK-WB", MODE_SILK, ForcedMode::SilkOnly, BW_WB, Bandwidth::Wideband, "CELT-WB"),
        ("Hybrid-SWB", MODE_HYBRID, ForcedMode::Hybrid, BW_SWB, Bandwidth::Superwideband, "CELT-SWB"),
        ("CELT-FB", MODE_CELT, ForcedMode::CeltOnly, BW_FB, Bandwidth::Fullband, "CELT-FB"),
    ] {
        for short in [120usize, 240] {
            for cbr in [false, true] {
                let what = format!("{name}, then {} ms frames, CBR {cbr}", short as f64 / 48.0);
                let mut c_enc = CEncoder::new(rate, 1, APP_AUDIO);
                c_enc.force_mode(c_mode).bandwidth(c_bw).vbr(!cbr).complexity(5);
                let mut enc = OpusEncoder::new(
                    SampleRate::try_from(rate as i32).unwrap(),
                    Channels::Mono,
                    Application::Audio,
                )
                .unwrap();
                enc.set_force_mode(Some(forced));
                enc.set_bandwidth(Some(bw));
                enc.set_vbr(!cbr);
                enc.set_complexity(5).unwrap();
                let mut buf = vec![0u8; 4000];
                let mut at = 0;
                let mut seen = 0;
                // 200 ms in 20 ms packets that code, 50 ms in short frames
                // at 600 b/s, and again.
                while at + 960 <= pcm.len() {
                    let coding = (at / 480) % 25 < 20;
                    let (frame, bitrate) = if coding { (960, 32000) } else { (short, 600) };
                    c_enc.bitrate(bitrate);
                    enc.set_bitrate(Bitrate::BitsPerSecond(NonZeroU32::new(bitrate as u32).unwrap()));
                    let input = &pcm[at..at + frame];
                    at += frame;
                    let c_packet = c_enc.encode(input);
                    let n = enc.encode(input, frame, &mut buf).unwrap_or_else(|e| panic!("{what}: {e}"));
                    if coding {
                        assert_eq!(buf[0] & 0xFC, c_packet[0] & 0xFC, "{what}: first byte of a coded packet");
                    } else {
                        assert_eq!(&buf[..n], &c_packet[..], "{what}: placeholder");
                        let t = k24_common::toc(buf[0]);
                        assert_eq!(format!("{}-{}", t.mode, t.bandwidth), placeholder, "{what}");
                        seen += 1;
                    }
                }
                assert!(seen >= 20, "{what}: {seen} placeholders");
            }
        }
    }
}
