//! Every packet of the opus1 multistream encoder is one C libopus decodes.
//!
//! The matrix: every layout of `multistream_enc_common::CASES`, variable and
//! constant bitrate, 48, 24 and 16 kHz, frames of 2.5 to 60 ms, and three
//! bitrates per layout. For every packet
//!
//! - `opus_multistream_decode_float` must accept it and return the frame
//!   size;
//! - the final range of the C decoder must equal
//!   `MultistreamEncoder::final_range`. The value is the XOR of the range
//!   coder states of all streams, so it shows that the decoder read every
//!   stream to the last bit the encoder wrote.
//!
//! 16-bit input is checked against float input: `encode_i16` must write the
//! bytes `encode` writes for the same samples.
//!
//! One fault outside the multistream encoder used to show here. A stereo
//! stream in Hybrid mode had a CELT layer the decoder read differently from
//! how it was written: src/celt.rs limited the intensity band to `0..end`
//! where libopus uses `start..end`. The packet decoded, but the final range
//! differed. That is corrected, and the matrix holds such packets to the
//! same rule as all others. `final_range_of_packets_with_a_stereo_hybrid_stream`
//! makes sure the matrix still reaches them. Surround and ambisonics layouts
//! never produce such a stream: their stereo streams are CELT-only.
#![cfg(feature = "std")]

mod ms_support;
mod multistream_enc_common;

use ms_support::*;
use multistream_enc_common::*;
use opus1::Application;

/// Bitrates of the matrix in bits per second per input channel. The lowest
/// brings SILK and narrow bandwidths into play, the highest is CELT fullband.
const RATES_PER_CHANNEL: [i32; 3] = [8_000, 24_000, 64_000];

/// Signal length per case in milliseconds. Every case starts a new encoder,
/// so the first frames, where an encoder has no history, are always covered.
const CASE_MS: usize = 240;

/// Whether a stream of the packet is stereo and in Hybrid mode.
fn has_stereo_hybrid_stream(packet: &[u8], streams: usize) -> bool {
    split_streams(packet, streams).is_ok_and(|parts| {
        parts.iter().any(|p| {
            let t = toc(p.toc);
            t.mode == Mode::Hybrid && t.stereo
        })
    })
}

/// What `run_case` found.
#[derive(Default)]
struct Outcome {
    /// A description of every packet that failed.
    failures: Vec<String>,
    /// Packets with a stereo Hybrid stream.
    stereo_hybrid: usize,
    /// Those among them whose final range differs from the C decoder's.
    stereo_hybrid_range_differs: usize,
}

impl Outcome {
    fn add(&mut self, other: Outcome) {
        self.failures.extend(other.failures);
        self.stereo_hybrid += other.stereo_hybrid;
        self.stereo_hybrid_range_differs += other.stereo_hybrid_range_differs;
    }
}

/// Encodes `pcm` with a new encoder and decodes every packet with C.
/// `force_stereo` forces every coupled stream to code two channels.
#[allow(clippy::too_many_arguments)]
fn run_case(
    case: &Case,
    fs: i32,
    application: Application,
    frame_size: usize,
    bitrate: i32,
    vbr: bool,
    force_stereo: bool,
    pcm: &[f32],
) -> Outcome {
    let mut enc = rust_encoder(case, fs, application);
    enc.set_bitrate(rust_bitrate(bitrate)).unwrap();
    enc.set_vbr(vbr);
    if force_stereo {
        // OPUS_SET_FORCE_CHANNELS(2) on every coupled stream encoder.
        for s in 0..enc.layout().coupled_streams() {
            enc.stream_encoder_mut(s).unwrap().set_force_channels(Some(2)).unwrap();
        }
    }
    let mut dec = c_decoder(&enc);
    let streams = enc.layout().streams();
    let label = format!(
        "{} {fs} Hz {application:?} {} {bitrate} b/s {}",
        case.name,
        ms(frame_size, fs),
        if vbr { "VBR" } else { "CBR" }
    );
    let mut outcome = Outcome::default();
    let mut out = vec![0u8; max_packet_bytes(streams)];
    for (f, frame) in pcm.chunks_exact(frame_size * case.channels).enumerate() {
        let n = match enc.encode(frame, frame_size, &mut out) {
            Ok(n) => n,
            Err(e) => {
                outcome
                    .failures
                    .push(format!("{label}: frame {f}: encode error: {e}"));
                break;
            }
        };
        let packet = &out[..n];
        match dec.decode_float(Some(packet), frame_size, false) {
            Ok(decoded) if decoded.len() == frame_size * case.channels => {}
            Ok(decoded) => {
                outcome.failures.push(format!(
                    "{label}: frame {f}: C decoded {} samples per channel: {}",
                    decoded.len() / case.channels,
                    describe_packet(packet, streams)
                ));
                break;
            }
            Err(e) => {
                outcome.failures.push(format!(
                    "{label}: frame {f}: C decode error {e}: {}",
                    describe_packet(packet, streams)
                ));
                break;
            }
        }
        let same_range = enc.final_range() == dec.final_range();
        if has_stereo_hybrid_stream(packet, streams) {
            outcome.stereo_hybrid += 1;
            outcome.stereo_hybrid_range_differs += !same_range as usize;
        }
        if !same_range {
            outcome.failures.push(format!(
                "{label}: frame {f}: final range {:08x}, C decoder {:08x}: {}",
                enc.final_range(),
                dec.final_range(),
                describe_packet(packet, streams)
            ));
        }
    }
    outcome
}

/// The whole matrix for one sampling rate and one bitrate mode.
/// `force_stereo` forces every coupled stream to code two channels.
fn run_matrix(fs: i32, vbr: bool, force_stereo: bool) -> Outcome {
    let mut outcome = Outcome::default();
    let mut cases = 0;
    for case in CASES {
        let samples = fs as usize * CASE_MS / 1000;
        let pcm = voiced(fs, case.channels, samples);
        for frame_size in frame_sizes(fs) {
            for per_channel in RATES_PER_CHANNEL {
                let bitrate = per_channel * case.channels as i32;
                outcome.add(run_case(
                    case,
                    fs,
                    Application::Audio,
                    frame_size,
                    bitrate,
                    vbr,
                    force_stereo,
                    &pcm,
                ));
                cases += 1;
            }
        }
    }
    println!(
        "{fs} Hz {}: {cases} cases, {} failures; {} packets with a stereo Hybrid stream, \
         final range differs in {}",
        if vbr { "VBR" } else { "CBR" },
        outcome.failures.len(),
        outcome.stereo_hybrid,
        outcome.stereo_hybrid_range_differs
    );
    for failure in outcome.failures.iter().take(40) {
        println!("{failure}");
    }
    outcome
}

fn assert_matrix(fs: i32, vbr: bool) {
    let outcome = run_matrix(fs, vbr, false);
    assert!(
        outcome.failures.is_empty(),
        "{} packets failed",
        outcome.failures.len()
    );
}

#[test]
fn c_decodes_every_packet_48k_vbr() {
    assert_matrix(48_000, true);
}

#[test]
fn c_decodes_every_packet_48k_cbr() {
    assert_matrix(48_000, false);
}

#[test]
fn c_decodes_every_packet_24k_vbr() {
    assert_matrix(24_000, true);
}

#[test]
fn c_decodes_every_packet_24k_cbr() {
    assert_matrix(24_000, false);
}

#[test]
fn c_decodes_every_packet_16k_vbr() {
    assert_matrix(16_000, true);
}

#[test]
fn c_decodes_every_packet_16k_cbr() {
    assert_matrix(16_000, false);
}

/// The matrix with every coupled stream forced to two channels. Left to
/// themselves, both encoders code most stereo streams of the matrix as mono
/// at 8 kb/s per channel. Measured on this matrix at 48 and 24 kHz,
/// packets with a stereo Hybrid stream, forced to two channels: 421 in
/// opus1 (93, 91, 111 and 126 in the four runs below), as in libopus (310
/// at 8 kb/s and 111 at 24 kb/s per channel). Before opus1 ran the speech
/// and music analysis, it coded the higher rates as CELT and had 220.
#[test]
fn final_range_of_packets_with_a_stereo_hybrid_stream() {
    let mut packets = 0;
    let mut differs = 0;
    for fs in [48_000, 24_000] {
        for vbr in [true, false] {
            let outcome = run_matrix(fs, vbr, true);
            packets += outcome.stereo_hybrid;
            differs += outcome.stereo_hybrid_range_differs;
        }
    }
    // The matrix must reach such packets, or this test says nothing.
    assert!(
        packets > 100,
        "only {packets} packets with a stereo Hybrid stream"
    );
    assert_eq!(
        differs, 0,
        "final range differs in {differs} of {packets} packets"
    );
}

/// The other two applications, on a smaller matrix.
#[test]
fn c_decodes_every_packet_voip_and_low_delay() {
    let fs = 48_000;
    let mut outcome = Outcome::default();
    for application in [Application::Voip, Application::RestrictedLowDelay] {
        for case in CASES {
            let pcm = voiced(fs, case.channels, fs as usize * CASE_MS / 1000);
            for frame_size in frame_sizes(fs) {
                for (per_channel, vbr) in [(10_000, true), (32_000, false)] {
                    let bitrate = per_channel * case.channels as i32;
                    outcome.add(run_case(
                        case,
                        fs,
                        application,
                        frame_size,
                        bitrate,
                        vbr,
                        false,
                        &pcm,
                    ));
                }
            }
        }
    }
    for failure in outcome.failures.iter().take(40) {
        println!("{failure}");
    }
    assert!(
        outcome.failures.is_empty(),
        "{} packets failed",
        outcome.failures.len()
    );
}

/// 8 and 12 kHz input, the automatic and the maximum bitrate.
#[test]
fn c_decodes_every_packet_at_low_sampling_rates_and_special_bitrates() {
    let mut outcome = Outcome::default();
    for (fs, bitrate) in [
        (8_000, 12_000),
        (12_000, 16_000),
        (8_000, OPUS_AUTO),
        (12_000, OPUS_BITRATE_MAX),
        (48_000, OPUS_AUTO),
        (48_000, OPUS_BITRATE_MAX),
        (16_000, OPUS_AUTO),
        (24_000, OPUS_BITRATE_MAX),
    ] {
        for case in CASES {
            let pcm = voiced(fs, case.channels, fs as usize * CASE_MS / 1000);
            for frame_size in frame_sizes(fs) {
                for vbr in [true, false] {
                    // An explicit bitrate is per channel.
                    let bitrate = if bitrate > 0 {
                        bitrate * case.channels as i32
                    } else {
                        bitrate
                    };
                    outcome.add(run_case(
                        case,
                        fs,
                        Application::Audio,
                        frame_size,
                        bitrate,
                        vbr,
                        false,
                        &pcm,
                    ));
                }
            }
        }
    }
    for failure in outcome.failures.iter().take(40) {
        println!("{failure}");
    }
    assert!(
        outcome.failures.is_empty(),
        "{} packets failed",
        outcome.failures.len()
    );
}

/// Full-scale noise and silence: the loudest and the quietest input.
#[test]
fn c_decodes_noise_and_silence() {
    let fs = 48_000;
    let mut outcome = Outcome::default();
    for case in CASES {
        let samples = fs as usize * CASE_MS / 1000;
        let signals = [
            noise(case.channels, samples, 1.0, 7),
            silence(case.channels, samples),
        ];
        for pcm in &signals {
            for frame_size in [fs as usize / 400, fs as usize / 50, 3 * fs as usize / 50] {
                for (per_channel, vbr) in [(12_000, true), (48_000, true), (24_000, false)] {
                    let bitrate = per_channel * case.channels as i32;
                    outcome.add(run_case(
                        case,
                        fs,
                        Application::Audio,
                        frame_size,
                        bitrate,
                        vbr,
                        false,
                        pcm,
                    ));
                }
            }
        }
    }
    for failure in outcome.failures.iter().take(40) {
        println!("{failure}");
    }
    assert!(
        outcome.failures.is_empty(),
        "{} packets failed",
        outcome.failures.len()
    );
}

#[test]
fn sixteen_bit_input_gives_the_bytes_of_float_input() {
    let mut packets = 0;
    for fs in [48_000, 16_000] {
        for case in CASES {
            let samples = fs as usize * CASE_MS / 1000;
            // Samples that are exactly a 16-bit value divided by 32768.
            let pcm16 = to_i16(&voiced(fs, case.channels, samples));
            let pcm: Vec<f32> = pcm16.iter().map(|&s| s as f32 / 32768.0).collect();
            for frame_size in frame_sizes(fs) {
                for (per_channel, vbr) in [(8_000, true), (24_000, false), (64_000, true)] {
                    let mut float_enc = rust_encoder(case, fs, Application::Audio);
                    let mut int_enc = rust_encoder(case, fs, Application::Audio);
                    for enc in [&mut float_enc, &mut int_enc] {
                        enc.set_bitrate(rust_bitrate(per_channel * case.channels as i32))
                            .unwrap();
                        enc.set_vbr(vbr);
                    }
                    // 16-bit input is coded at a depth of 16 bits, as libopus
                    // `opus_multistream_encode` codes it.
                    float_enc.set_lsb_depth(16).unwrap();
                    let size = max_packet_bytes(float_enc.layout().streams());
                    let (mut a, mut b) = (vec![0u8; size], vec![0u8; size]);
                    let step = frame_size * case.channels;
                    for (f, (frame, frame16)) in pcm
                        .chunks_exact(step)
                        .zip(pcm16.chunks_exact(step))
                        .enumerate()
                    {
                        let n = float_enc.encode(frame, frame_size, &mut a).unwrap();
                        let m = int_enc.encode_i16(frame16, frame_size, &mut b).unwrap();
                        assert_eq!(
                            &a[..n],
                            &b[..m],
                            "{} {fs} Hz {} {per_channel} b/s per channel: frame {f}",
                            case.name,
                            ms(frame_size, fs)
                        );
                        assert_eq!(float_enc.final_range(), int_enc.final_range());
                        packets += 1;
                    }
                }
            }
        }
    }
    println!("{packets} packets compared");
}
