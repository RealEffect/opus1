//! The speech and music analysis in `MultistreamEncoder` against libopus
//! (`opus_multistream_encode_float` and `opus_multistream_encode`) at
//! complexity 7 to 10.
//!
//! libopus gives every stream encoder the whole interleaved input and the
//! channels of the stream (opus_multistream_encoder.c:978-1028): `c1` and
//! `c2` are the left and right channel of a coupled stream, and `c1` the
//! channel of a mono stream with `c2 = -1`. Each stream runs its own
//! analysis on them. What is checked:
//!
//! 1. With every stream CELT-only, every multistream packet is libopus's
//!    byte for byte: every layout of `multistream_enc_common::CASES`
//!    (surround, ambisonics, family 255, generic layouts with an uncoded
//!    channel and with two channels on one stream), 16, 24 and 48 kHz,
//!    float and 16-bit input, VBR and CBR.
//! 2. The analysis each stream coded its last frame with is what libopus's
//!    `run_analysis` gives for that stream's channels, in every mode.
//! 3. The controls `set_signal`, `set_lsb_depth` and `set_inband_fec_mode`
//!    reach every stream, as libopus forwards them.
//! 4. With an expert frame duration the analysis reads the caller's whole
//!    buffer, as libopus's does.
#![cfg(all(feature = "std", feature = "analysis"))]
// With `test-libopus-dnn` the tests link a libopus built with its neural
// network code and without vector loops. Its analysis rounds two sums of
// the cepstrum differently from the Homebrew build that opus1 follows
// (docs/10-differences-from-libopus.md), so these tests run only without
// that feature.
#![cfg(not(feature = "test-libopus-dnn"))]

mod analysis_common;
mod k24_common;
mod ms_support;
mod multistream_enc_common;

use analysis_common::{CAnalysis, Layout, Pcm, info_difference};
use ms_support::*;
use multistream_enc_common::*;
use opus1::internals::analysis::last_analysis_info;
use opus1::multistream::MultistreamEncoder;
use opus1::{Application, ForcedMode, FrameDuration, InbandFec, SampleRate, Signal};

/// Whether the packets must be libopus's: the analysis rounds as the
/// libopus the tests link only on 64-bit ARM.
const EXACT: bool = cfg!(target_arch = "aarch64");

/// `channels` channels of test signal, interleaved: speech in the even
/// channels and music in the odd ones, each channel delayed and scaled
/// differently, so that every stream has its own input.
fn signal(fs: i32, channels: usize, seconds: f64) -> Vec<f32> {
    let fs_u = fs as usize;
    let n = (seconds * fs as f64) as usize;
    let speech = k24_common::speech_fullband(fs_u, seconds + 0.1);
    let music = k24_common::multitone(fs_u, n + fs_u / 10);
    let mut pcm = vec![0.0f32; n * channels];
    for c in 0..channels {
        let delay = 37 * c;
        let gain = 1.0 - 0.08 * c as f32;
        let src = if c % 2 == 0 { &speech } else { &music };
        for i in 0..n {
            pcm[i * channels + c] = gain * src[i + delay];
        }
    }
    pcm
}

fn to_i16(pcm: &[f32]) -> Vec<i16> {
    pcm.iter()
        .map(|&x| (x * 32768.0).round().clamp(-32768.0, 32767.0) as i16)
        .collect()
}

/// libopus's `c1` and `c2` of each stream of `enc`.
fn stream_channels(enc: &MultistreamEncoder) -> Vec<Layout> {
    let layout = enc.layout();
    (0..layout.streams())
        .map(|s| {
            if s < layout.coupled_streams() {
                Layout {
                    channels: layout.channels(),
                    c1: layout.left_channel(s, None).unwrap(),
                    c2: layout.right_channel(s, None).unwrap() as i32,
                }
            } else {
                Layout {
                    channels: layout.channels(),
                    c1: layout.mono_channel(s, None).unwrap(),
                    c2: -1,
                }
            }
        })
        .collect()
}

/// The packets of both encoders for `pcm`, `frame_size` samples per channel
/// per call, with float or 16-bit input.
fn encode_both(
    rust: &mut MultistreamEncoder,
    c: &mut CMsEncoder,
    pcm: &[f32],
    frame_size: usize,
    i16_input: bool,
) -> (Vec<Vec<u8>>, Vec<Vec<u8>>) {
    let channels = rust.layout().channels();
    let max_bytes = max_packet_bytes(rust.layout().streams());
    let pcm16 = to_i16(pcm);
    let step = frame_size * channels;
    let mut out = vec![0u8; max_bytes];
    let (mut rs, mut cp) = (Vec::new(), Vec::new());
    for k in 0..pcm.len() / step {
        let range = k * step..(k + 1) * step;
        let n = if i16_input {
            cp.push(c.encode_i16(&pcm16[range.clone()], frame_size, max_bytes).unwrap());
            rust.encode_i16(&pcm16[range], frame_size, &mut out).unwrap()
        } else {
            cp.push(c.encode_float(&pcm[range.clone()], frame_size, max_bytes).unwrap());
            rust.encode(&pcm[range], frame_size, &mut out).unwrap()
        };
        rs.push(out[..n].to_vec());
    }
    (rs, cp)
}

/// A small random number generator (64-bit linear congruential).
struct Lcg(u64);

impl Lcg {
    fn below(&mut self, n: usize) -> usize {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) % n as u64) as usize
    }
}

/// Every stream CELT-only: the multistream packets are libopus's.
#[test]
fn celt_only_packets_are_libopus() {
    let mut rng = Lcg(31);
    let (mut streams, mut packets, mut failures) = (0, 0, Vec::new());
    for fs in [16_000, 24_000, 48_000] {
        for case in CASES {
            let pcm = signal(fs, case.channels, 1.0);
            for (per_channel, vbr) in [(24_000, true), (48_000, false), (96_000, true)] {
                let complexity = 7 + rng.below(4) as i32;
                let i16_input = rng.below(2) == 0;
                let (mut rust, mut c) = encoders(case, fs, Application::Audio);
                configure(&mut rust, &mut c, per_channel * case.channels as i32, vbr);
                rust.set_complexity(complexity).unwrap();
                c.set_complexity(complexity);
                rust.set_force_mode(Some(ForcedMode::CeltOnly));
                c.ctl(OPUS_SET_FORCE_MODE_REQUEST, MODE_CELT_ONLY);
                let frame_size = fs as usize / 50;
                let (rs, cp) = encode_both(&mut rust, &mut c, &pcm, frame_size, i16_input);
                streams += 1;
                packets += rs.len();
                if let Some(k) = rs.iter().zip(&cp).position(|(a, b)| a != b) {
                    failures.push(format!(
                        "{} {fs} Hz {per_channel} b/s per channel vbr {vbr} c{complexity} i16 {i16_input}: packet {k}",
                        case.name
                    ));
                }
            }
        }
    }
    println!("{streams} streams, {packets} packets");
    for f in failures.iter().take(20) {
        println!("  {f}");
    }
    if EXACT {
        assert!(failures.is_empty(), "{} streams are not libopus's", failures.len());
    }
}

/// The analysis each stream coded its last frame with is libopus's
/// analysis of the stream's channels, in every mode: an analysis state per
/// stream, called as `opus_encode_native` calls it.
#[test]
fn the_analysis_of_each_stream_is_libopus() {
    let mut rng = Lcg(5);
    let (mut compared, mut failures) = (0, Vec::new());
    for fs in [16_000, 24_000, 48_000] {
        for case in CASES {
            for frame_ms in [10, 20, 40, 60] {
                let application = [Application::Audio, Application::Voip][rng.below(2)];
                let per_channel = [12_000, 24_000, 64_000][rng.below(3)];
                let i16_input = rng.below(2) == 0;
                let (mut rust, _) = encoders(case, fs, application);
                rust.set_bitrate(rust_bitrate(per_channel * case.channels as i32)).unwrap();
                rust.set_complexity(7 + rng.below(4) as i32).unwrap();
                let layouts = stream_channels(&rust);
                let mut c: Vec<CAnalysis> = layouts.iter().map(|_| CAnalysis::new(fs)).collect();
                let frame_size = fs as usize * frame_ms / 1000;
                let channels = case.channels;
                let pcm = signal(fs, channels, 1.0);
                let pcm16 = to_i16(&pcm);
                let mut out = vec![0u8; max_packet_bytes(layouts.len())];
                let step = frame_size * channels;
                'packets: for k in 0..pcm.len() / step {
                    let range = k * step..(k + 1) * step;
                    let (input, n) = if i16_input {
                        let n = rust.encode_i16(&pcm16[range.clone()], frame_size, &mut out).unwrap();
                        (Pcm::I16(pcm16[range].to_vec()), n)
                    } else {
                        let n = rust.encode(&pcm[range.clone()], frame_size, &mut out).unwrap();
                        (Pcm::F32(pcm[range].to_vec()), n)
                    };
                    let parts = split_streams(&out[..n], layouts.len()).unwrap();
                    for (s, (layout, part)) in layouts.iter().zip(&parts).enumerate() {
                        // The frames of this stream's packet: SILK frames of
                        // 40 and 60 ms are one frame, CELT and Hybrid ones
                        // 20 ms each.
                        let silk = part.toc < 0x60;
                        let part_size = if silk || frame_size <= fs as usize / 50 {
                            frame_size
                        } else {
                            fs as usize / 50
                        };
                        let lsb = if i16_input { 16 } else { 24 };
                        let st = &mut c[s];
                        let cursor = (st.st.read_pos, st.st.read_subframe);
                        let mut info = st.run(&input, *layout, frame_size, frame_size, lsb);
                        if part_size < frame_size {
                            (st.st.read_pos, st.st.read_subframe) = cursor;
                            for _ in 0..frame_size / part_size {
                                info = st.get_info(part_size);
                            }
                        }
                        // A packet without audio codes no frame.
                        if part.payload_len == 0 {
                            continue;
                        }
                        compared += 1;
                        let got = last_analysis_info(rust.stream_encoder(s).unwrap());
                        if let Some(d) = info_difference(&got, &info) {
                            failures.push(format!(
                                "{} {fs} Hz {frame_ms} ms {application:?} {per_channel} b/s i16 {i16_input}: packet {k} stream {s}: {d}",
                                case.name
                            ));
                            break 'packets;
                        }
                    }
                }
            }
        }
    }
    println!("{compared} stream frames compared");
    for f in failures.iter().take(20) {
        println!("  {f}");
    }
    if EXACT {
        assert!(failures.is_empty(), "{} runs with another analysis", failures.len());
    }
}

/// The controls of the analysis reach every stream, as libopus forwards
/// `OPUS_SET_SIGNAL`, `OPUS_SET_LSB_DEPTH` and `OPUS_SET_INBAND_FEC`.
/// Forced to CELT-only at 16 kb/s per channel, where the signal type moves
/// the choice between mono and stereo coding and the bandwidth, the packets
/// are libopus's.
#[test]
fn controls_reach_every_stream() {
    let case = Case::family("5.1", 6, 1);
    let mut rust = rust_encoder(&case, 48_000, Application::Audio);
    assert_eq!(rust.signal(), Signal::Auto);
    assert_eq!(rust.lsb_depth(), 24);
    assert_eq!(rust.inband_fec_mode(), InbandFec::Off);
    rust.set_signal(Signal::Voice);
    rust.set_lsb_depth(12).unwrap();
    assert!(rust.set_lsb_depth(25).is_err());
    assert!(rust.set_lsb_depth(7).is_err());
    rust.set_inband_fec_mode(InbandFec::OnUnlessMusic);
    for s in 0..rust.layout().streams() {
        let enc = rust.stream_encoder(s).unwrap();
        assert_eq!(enc.signal(), Signal::Voice);
        assert_eq!(enc.lsb_depth(), 12);
        assert_eq!(enc.inband_fec_mode(), InbandFec::OnUnlessMusic);
    }
    assert_eq!(rust.signal(), Signal::Voice);
    assert_eq!(rust.lsb_depth(), 12);
    assert!(rust.inband_fec());

    let mut failures = Vec::new();
    for case in [Case::family("family 255, 3 channels", 3, 255), Case::family("5.1", 6, 1)] {
        let pcm = signal(48_000, case.channels, 1.0);
        for signal_type in [Signal::Auto, Signal::Voice, Signal::Music] {
            for (lsb, fec) in [(24, InbandFec::Off), (12, InbandFec::On), (16, InbandFec::OnUnlessMusic)] {
                let (mut rust, mut c) = encoders(&case, 48_000, Application::Audio);
                configure(&mut rust, &mut c, 16_000 * case.channels as i32, true);
                rust.set_force_mode(Some(ForcedMode::CeltOnly));
                c.ctl(OPUS_SET_FORCE_MODE_REQUEST, MODE_CELT_ONLY);
                rust.set_signal(signal_type);
                c.ctl(
                    audiopus_sys::OPUS_SET_SIGNAL_REQUEST,
                    match signal_type {
                        Signal::Auto => OPUS_AUTO,
                        Signal::Voice => audiopus_sys::OPUS_SIGNAL_VOICE,
                        Signal::Music => audiopus_sys::OPUS_SIGNAL_MUSIC,
                    },
                );
                rust.set_lsb_depth(lsb).unwrap();
                c.ctl(audiopus_sys::OPUS_SET_LSB_DEPTH_REQUEST, lsb);
                rust.set_inband_fec_mode(fec);
                rust.set_packet_loss_perc(10).unwrap();
                c.ctl(audiopus_sys::OPUS_SET_INBAND_FEC_REQUEST, fec as i32);
                c.set_packet_loss_perc(10);
                let (rs, cp) = encode_both(&mut rust, &mut c, &pcm, 960, false);
                if let Some(k) = rs.iter().zip(&cp).position(|(a, b)| a != b) {
                    failures.push(format!("{} {signal_type:?} lsb {lsb} {fec:?}: packet {k}", case.name));
                }
            }
        }
    }
    for f in &failures {
        println!("  {f}");
    }
    if EXACT {
        assert!(failures.is_empty(), "{} streams are not libopus's", failures.len());
    }
}

/// With an expert frame duration the caller's buffer is longer than the
/// coded frame. Each stream's analysis reads all of it, as in libopus, and
/// the CELT-only packets stay libopus's.
#[test]
fn expert_frame_duration_reads_the_whole_buffer() {
    let mut failures = Vec::new();
    for case in [Case::family("5.1", 6, 1), Case::family("family 255, 3 channels", 3, 255)] {
        let fs = 48_000;
        let pcm = signal(fs, case.channels, 1.0);
        for (expert, c_value, caller) in [
            (FrameDuration::Ms10, audiopus_sys::OPUS_FRAMESIZE_10_MS, 1920),
            (FrameDuration::Ms20, audiopus_sys::OPUS_FRAMESIZE_20_MS, 2880),
        ] {
            let (mut rust, mut c) = encoders(&case, fs, Application::Audio);
            configure(&mut rust, &mut c, 32_000 * case.channels as i32, true);
            rust.set_force_mode(Some(ForcedMode::CeltOnly));
            c.ctl(OPUS_SET_FORCE_MODE_REQUEST, MODE_CELT_ONLY);
            rust.set_expert_frame_duration(expert);
            c.ctl(audiopus_sys::OPUS_SET_EXPERT_FRAME_DURATION_REQUEST, c_value);
            let coded = expert.samples(SampleRate::Hz48000).unwrap();
            let max_bytes = max_packet_bytes(rust.layout().streams());
            let mut out = vec![0u8; max_bytes];
            let mut at = 0;
            let mut k = 0;
            while at + caller * case.channels <= pcm.len() {
                let buf = &pcm[at..at + caller * case.channels];
                let cp = c.encode_float(buf, caller, max_bytes).unwrap();
                let n = rust.encode(buf, caller, &mut out).unwrap();
                if out[..n] != cp[..] {
                    failures.push(format!("{} {expert:?}: packet {k}", case.name));
                    break;
                }
                at += coded * case.channels;
                k += 1;
            }
        }
    }
    for f in &failures {
        println!("  {f}");
    }
    if EXACT {
        assert!(failures.is_empty(), "{} streams are not libopus's", failures.len());
    }
}
