//! `MultistreamDecoder` against the C libopus multistream decoder: the same
//! packets, from the C multistream encoder, must decode to the same samples.
//!
//! What is checked:
//!
//! - float output, for 10 layouts, bitrates from 6 to 64 kb/s per channel,
//!   frames from 2.5 to 60 ms and decoder rates from 8 to 48 kHz: every
//!   channel is within 100 dB of the C decoder's output, and the final range
//!   after every packet equals the C encoder's;
//! - 16-bit output, with and without a signal beyond full scale;
//! - decoder layouts that differ from the encoder's: channels in another
//!   order, left out, repeated and silent.
//!
//! libopus codes a stereo stream as mono at low bitrates. The single-stream
//! decoder of opus-rs does not decode such a packet into two channels yet.
//! The cases with such packets are in tests of their own, which are ignored
//! until it does.

mod ms_support;
mod multistream_dec_common;

use ms_support::*;
use multistream_dec_common::*;
use opus_rs::multistream::MultistreamDecoder;

/// The limit for "the same samples as the C decoder". Measured: 124 dB or
/// more in every case of this file.
const MIN_SNR_DB: f64 = 100.0;

/// Which cases of the matrix a test takes.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Part {
    /// Every stereo stream of every packet is coded as stereo.
    StereoCoded,
    /// Some packet has a stereo stream that is coded as mono.
    MonoCoded,
}

/// Decodes the packets of `e` at `fs` with both decoders and compares them.
/// Returns the smallest SNR over the channels in dB.
fn check_float(e: &Encoded, fs: i32) -> f64 {
    let (mut c_dec, mut r_dec) = e.decoders(fs);
    let max_frame = fs as usize / 25 * 3;
    let mut out = vec![0f32; max_frame * e.channels];
    let (mut c_all, mut r_all) = (Vec::new(), Vec::new());
    for (i, packet) in e.packets.iter().enumerate() {
        let what = || format!("{} at {fs} Hz, packet {i} ({})", e.label, describe_packet(packet, e.streams));
        let c = c_dec
            .decode_float(Some(packet), max_frame, false)
            .unwrap_or_else(|err| panic!("{}: C decoder: {err}", what()));
        let n = r_dec
            .decode(packet, max_frame, &mut out)
            .unwrap_or_else(|err| panic!("{}: opus-rs decoder: {err}", what()));
        assert_eq!(n * e.channels, c.len(), "{}: samples per channel", what());
        assert_eq!(r_dec.final_range(), e.ranges[i], "{}: final range", what());
        assert_eq!(c_dec.final_range(), e.ranges[i], "{}: final range of the C decoder", what());
        c_all.extend_from_slice(&c);
        r_all.extend_from_slice(&out[..n * e.channels]);
    }
    let snrs = channel_snrs(&c_all, &r_all, e.channels);
    for (channel, &snr) in snrs.iter().enumerate() {
        assert!(
            snr >= MIN_SNR_DB,
            "{} at {fs} Hz: channel {channel} is {snr:.1} dB from the C decoder (all channels: {snrs:.0?})",
            e.label
        );
    }
    snrs.into_iter().fold(f64::INFINITY, f64::min)
}

/// Runs the matrix for one layout and returns the number of encoded signals
/// that belong to `part` and were checked.
///
/// The matrix has three blocks:
///
/// 1. music settings: 2 signals, 7 bitrates, 6 frame sizes, 4 decoder rates;
/// 2. speech settings with error correction, which give SILK and Hybrid
///    packets: 7 bitrates, 6 frame sizes, decoder rates 12 and 48 kHz;
/// 3. constant bitrate, which gives padded packets: 3 bitrates, 6 frame
///    sizes, decoder rates 16 and 48 kHz.
fn run_matrix(layout: Layout, part: Part) -> usize {
    /// One block of the matrix.
    struct Block {
        settings: Settings,
        signals: &'static [Signal],
        /// In kb/s per channel.
        bitrates: &'static [i32],
        decoder_rates: &'static [i32],
    }
    const BITRATES: [i32; 7] = [6, 12, 16, 24, 32, 48, 64];
    const FRAMES: [usize; 6] = [120, 240, 480, 960, 1920, 2880];
    let blocks = [
        Block {
            settings: AUDIO,
            signals: &[Signal::Tones, Signal::Voiced],
            bitrates: &BITRATES,
            decoder_rates: &[8000, 16000, 24000, 48000],
        },
        Block { settings: VOIP_FEC, signals: &[Signal::Voiced], bitrates: &BITRATES, decoder_rates: &[12000, 48000] },
        Block { settings: AUDIO_CBR, signals: &[Signal::Tones], bitrates: &[12, 32, 64], decoder_rates: &[16000, 48000] },
    ];
    let mut checked = 0;
    let mut worst = f64::INFINITY;
    for Block { settings, signals, bitrates, decoder_rates } in blocks {
        for &signal in signals {
            for &kbps in bitrates {
                for frame in FRAMES {
                    let e = encode(layout, settings, kbps, frame, signal, 600, 1.0);
                    let wanted = match part {
                        Part::StereoCoded => !e.has_mono_coded_stereo_stream(),
                        Part::MonoCoded => e.has_mono_coded_stereo_stream(),
                    };
                    if !wanted {
                        continue;
                    }
                    for &fs in decoder_rates {
                        worst = worst.min(check_float(&e, fs));
                    }
                    checked += 1;
                }
            }
        }
    }
    println!("{layout:?}: {checked} encoded signals checked, smallest SNR against the C decoder {worst:.0} dB");
    checked
}

/// Number of encoded signals in the matrix of `run_matrix`.
const MATRIX_SIZE: usize = 2 * 7 * 6 + 7 * 6 + 3 * 6;

#[test]
fn mono_matches_libopus() {
    assert_eq!(run_matrix(MONO, Part::StereoCoded), MATRIX_SIZE);
}

#[test]
fn quad_matches_libopus() {
    assert_eq!(run_matrix(QUAD, Part::StereoCoded), MATRIX_SIZE);
}

#[test]
fn surround_5_1_matches_libopus() {
    assert_eq!(run_matrix(SURROUND_5_1, Part::StereoCoded), MATRIX_SIZE);
}

#[test]
fn surround_7_1_matches_libopus() {
    assert_eq!(run_matrix(SURROUND_7_1, Part::StereoCoded), MATRIX_SIZE);
}

#[test]
fn family_255_with_3_channels_matches_libopus() {
    assert_eq!(run_matrix(DISCRETE_3, Part::StereoCoded), MATRIX_SIZE);
}

#[test]
fn family_255_with_5_channels_matches_libopus() {
    assert_eq!(run_matrix(DISCRETE_5, Part::StereoCoded), MATRIX_SIZE);
}

#[test]
fn ambisonics_with_4_channels_matches_libopus() {
    assert_eq!(run_matrix(AMBISONICS_4, Part::StereoCoded), MATRIX_SIZE);
}

// The three layouts below have a stereo stream that libopus is free to code
// as mono. Their matrix is split in two tests. The numbers of cases were
// counted with libopus 1.6.1.

#[test]
fn stereo_matches_libopus() {
    let checked = run_matrix(STEREO, Part::StereoCoded);
    assert!(checked >= 100, "only {checked} cases without a mono-coded packet");
}

#[test]
fn stereo_with_mono_coded_packets_matches_libopus() {
    let checked = run_matrix(STEREO, Part::MonoCoded);
    assert!(checked >= 20, "only {checked} cases with a mono-coded packet");
}

#[test]
fn ambisonics_with_6_channels_matches_libopus() {
    let checked = run_matrix(AMBISONICS_6, Part::StereoCoded);
    assert!(checked >= 80, "only {checked} cases without a mono-coded packet");
}

#[test]
fn ambisonics_with_6_channels_and_mono_coded_packets_matches_libopus() {
    let checked = run_matrix(AMBISONICS_6, Part::MonoCoded);
    assert!(checked >= 20, "only {checked} cases with a mono-coded packet");
}

#[test]
fn generic_layout_matches_libopus() {
    let checked = run_matrix(Layout::Generic, Part::StereoCoded);
    assert!(checked >= 100, "only {checked} cases without a mono-coded packet");
}

#[test]
fn generic_layout_with_mono_coded_packets_matches_libopus() {
    let checked = run_matrix(Layout::Generic, Part::MonoCoded);
    assert!(checked >= 20, "only {checked} cases with a mono-coded packet");
}

/// Every input channel has a sine of its own frequency. After decoding,
/// each output channel must carry the sine of the input channel it stands
/// for, at the level it went in with, and none of the others.
#[test]
fn every_channel_carries_its_own_tone() {
    for layout in ALL_LAYOUTS {
        let e = encode(layout, AUDIO, 48, 960, Signal::Tones, 1000, 1.0);
        assert!(!e.has_mono_coded_stereo_stream(), "{}", e.label);
        // The input channel behind each output channel.
        let source: Vec<Option<usize>> = match layout {
            Layout::Family { channels, .. } => (0..channels).map(Some).collect(),
            Layout::Generic => vec![Some(2), Some(0), None, Some(1), Some(2)],
        };
        let (_, mut dec) = e.decoders(FS);
        let mut out = vec![0f32; 960 * e.channels];
        let mut decoded = Vec::new();
        for packet in &e.packets {
            assert_eq!(dec.decode(packet, 960, &mut out), Ok(960), "{}", e.label);
            decoded.extend_from_slice(&out);
        }
        // Leave out the first 200 ms, where the codec starts up.
        let skip = FS as usize / 5;
        let want = (TONE_AMPLITUDE * TONE_AMPLITUDE / 2.0) as f64;
        for (c, &src) in source.iter().enumerate() {
            let x = channel(&decoded, e.channels, c);
            let x = &x[skip..];
            let Some(src) = src else {
                assert!(x.iter().all(|&v| v == 0.0), "{}: output {c} is not silent", e.label);
                continue;
            };
            let power = |input: usize| goertzel_power(x, tone_hz(input, e.input_channels, e.family), FS);
            let own = power(src);
            let strongest_other = (0..e.input_channels).filter(|&k| k != src).map(power).fold(0.0, f64::max);
            assert!(
                own >= 1000.0 * strongest_other,
                "{}: output {c}: its tone has the power {own:.2e}, another tone {strongest_other:.2e}",
                e.label
            );
            assert!(
                (own / want - 1.0).abs() < 0.1,
                "{}: output {c}: tone power {own:.5}, expected {want:.5}",
                e.label
            );
        }
    }
}

/// In the hand-made layout the mono stream feeds outputs 0 and 4, and output
/// 2 is silent.
#[test]
fn generic_layout_repeats_one_stream_and_keeps_one_output_silent() {
    for fs in [8000, 48000] {
        let e = encode(Layout::Generic, AUDIO, 48, 960, Signal::Tones, 600, 1.0);
        assert_eq!(e.mapping, [2, 0, 255, 1, 2]);
        let (_, mut dec) = e.decoders(fs);
        let max_frame = fs as usize / 25 * 3;
        // The output starts with a value that is not silence.
        let ch = e.channels;
        let mut out = vec![0.5f32; max_frame * ch];
        let mut heard = 0f64;
        for packet in &e.packets {
            out.fill(0.5);
            let n = dec.decode(packet, max_frame, &mut out).unwrap();
            for sample in out[..n * ch].chunks_exact(ch) {
                assert_eq!(sample[0].to_bits(), sample[4].to_bits(), "outputs 0 and 4 differ");
                assert_eq!(sample[2].to_bits(), 0f32.to_bits(), "output 2 is not silent");
                heard += (sample[0] as f64).powi(2);
            }
            // Nothing is written behind the decoded samples.
            assert!(out[n * ch..].iter().all(|&v| v == 0.5));
        }
        assert!(heard > 1.0, "the repeated channel is silent too");
    }
}

/// Decodes the packets of `e` with the layout `mapping` in both decoders and
/// returns both outputs.
fn decode_with_mapping(e: &Encoded, fs: i32, mapping: &[u8]) -> (Vec<f32>, Vec<f32>) {
    let channels = mapping.len();
    let mut c_dec = CMsDecoder::new(fs, channels, e.streams, e.coupled, mapping).unwrap();
    let mut r_dec = MultistreamDecoder::with_layout(fs, channels, e.streams, e.coupled, mapping).unwrap();
    let max_frame = fs as usize / 25 * 3;
    let mut out = vec![0f32; max_frame * channels];
    let (mut c_all, mut r_all) = (Vec::new(), Vec::new());
    for (i, packet) in e.packets.iter().enumerate() {
        let c = c_dec.decode_float(Some(packet), max_frame, false).unwrap();
        let n = r_dec.decode(packet, max_frame, &mut out).unwrap();
        assert_eq!(n * channels, c.len());
        // The final range does not depend on the mapping.
        assert_eq!(r_dec.final_range(), e.ranges[i], "{}: packet {i}", e.label);
        c_all.extend_from_slice(&c);
        r_all.extend_from_slice(&out[..n * channels]);
    }
    (c_all, r_all)
}

/// A decoder may use any mapping onto the streams of a packet. These
/// mappings differ from the one the 5.1 encoder was built with.
#[test]
fn decoder_layouts_that_differ_from_the_encoders_match_libopus() {
    let e = encode(SURROUND_5_1, AUDIO, 32, 960, Signal::Tones, 600, 1.0);
    // 4 streams, 2 of them stereo: decoded channels 0 to 5.
    assert_eq!((e.streams, e.coupled), (4, 2));
    let mappings: [&[u8]; 6] = [
        // The decoded channels in their own order, and backwards.
        &[0, 1, 2, 3, 4, 5],
        &[5, 4, 3, 2, 1, 0],
        // Only the first stereo stream. The other streams are decoded and
        // not used.
        &[0, 1],
        // One channel.
        &[4],
        // Repeated and silent channels.
        &[0, 0, 1, 255, 2, 3, 3, 255, 4, 5, 5, 4],
        // Nothing but silence.
        &[255, 255, 255],
    ];
    for mapping in mappings {
        for fs in [16000, 48000] {
            let channels = mapping.len();
            let (c, r) = decode_with_mapping(&e, fs, mapping);
            let snrs = channel_snrs(&c, &r, channels);
            assert!(
                snrs.iter().all(|&snr| snr >= MIN_SNR_DB),
                "mapping {mapping:?} at {fs} Hz: {snrs:.0?} dB against the C decoder"
            );
            // Channels with the same mapping entry carry the same samples,
            // and a silent channel carries zeros.
            for frame in r.chunks_exact(channels) {
                for (a, &ma) in mapping.iter().enumerate() {
                    if ma == 255 {
                        assert_eq!(frame[a].to_bits(), 0f32.to_bits(), "mapping {mapping:?}: channel {a}");
                    }
                    for (b, &mb) in mapping.iter().enumerate().skip(a + 1) {
                        if ma == mb {
                            assert_eq!(frame[a].to_bits(), frame[b].to_bits(), "mapping {mapping:?}: {a} and {b}");
                        }
                    }
                }
            }
        }
    }
}

/// The widest layout libopus accepts: 255 output channels on 1 stream, and
/// 255 mono streams.
#[test]
fn layouts_with_255_channels_match_libopus() {
    // 255 outputs: stream channel 0, 1 or silence in turn.
    let e = encode(STEREO, AUDIO, 64, 960, Signal::Tones, 200, 1.0);
    let mapping: Vec<u8> = (0..255).map(|c| [0u8, 1, 255][c % 3]).collect();
    let (c, r) = decode_with_mapping(&e, 48000, &mapping);
    let snrs = channel_snrs(&c, &r, 255);
    assert!(snrs.iter().all(|&snr| snr >= MIN_SNR_DB), "{snrs:.0?}");

    // 255 mono streams, a few frames.
    let e = encode(Layout::Family { channels: 255, family: 255 }, AUDIO, 8, 960, Signal::Voiced, 100, 1.0);
    assert_eq!((e.streams, e.coupled), (255, 0));
    let (c, r) = decode_with_mapping(&e, 16000, &e.mapping);
    let snrs = channel_snrs(&c, &r, 255);
    assert!(snrs.iter().all(|&snr| snr >= MIN_SNR_DB), "{snrs:.0?}");
}

/// 16-bit output against the C decoder's 16-bit output.
///
/// libopus soft-clips each stream before it rounds to 16 bits. With `gain`
/// 6 the decoded signal goes beyond full scale, so the soft clip has work.
/// The float outputs of the two decoders differ by rounding in the last bit,
/// which can move a sample by one step of 16 bits.
#[test]
fn sixteen_bit_output_matches_libopus() {
    let layouts = [MONO, QUAD, SURROUND_5_1, SURROUND_7_1, DISCRETE_3, AMBISONICS_4, Layout::Generic];
    let mut worst_fraction = 0f64;
    for layout in layouts {
        for gain in [1.0f32, 6.0] {
            for (kbps, frame) in [(16, 960), (48, 480), (64, 2880)] {
                let e = encode(layout, AUDIO, kbps, frame, Signal::Tones, 600, gain);
                if e.has_mono_coded_stereo_stream() {
                    continue;
                }
                for fs in [8000, 24000, 48000] {
                    let (mut c_dec, mut r_dec) = e.decoders(fs);
                    // A second decoder with float output, for comparison.
                    let (_, mut r_float) = e.decoders(fs);
                    let max_frame = fs as usize / 25 * 3;
                    let mut out = vec![0i16; max_frame * e.channels];
                    let mut out_float = vec![0f32; max_frame * e.channels];
                    let (mut differing, mut total, mut at_full_scale, mut unlike_hard_clip) = (0usize, 0usize, 0usize, 0usize);
                    for (i, packet) in e.packets.iter().enumerate() {
                        let c = c_dec.decode_i16(Some(packet), max_frame, false).unwrap();
                        let n = r_dec.decode_i16(packet, max_frame, &mut out).unwrap();
                        assert_eq!(n * e.channels, c.len(), "{} at {fs} Hz, packet {i}", e.label);
                        assert_eq!(r_dec.final_range(), e.ranges[i], "{} at {fs} Hz, packet {i}", e.label);
                        let n_float = r_float.decode(packet, max_frame, &mut out_float).unwrap();
                        assert_eq!(n_float, n);
                        let hard_clipped = to_i16(&out_float[..n * e.channels]);
                        for ((&c, &r), &h) in c.iter().zip(&out[..c.len()]).zip(&hard_clipped) {
                            let d = (c as i32 - r as i32).abs();
                            assert!(d <= 1, "{} at {fs} Hz, gain {gain}, packet {i}: C {c}, opus-rs {r}", e.label);
                            differing += (d != 0) as usize;
                            at_full_scale += (c == i16::MAX || c == i16::MIN) as usize;
                            unlike_hard_clip += (r != h) as usize;
                        }
                        total += c.len();
                    }
                    let fraction = differing as f64 / total as f64;
                    worst_fraction = worst_fraction.max(fraction);
                    assert!(
                        fraction <= 0.01,
                        "{} at {fs} Hz, gain {gain}: {differing} of {total} samples differ from the C decoder",
                        e.label
                    );
                    if gain == 1.0 {
                        // Nothing to clip: 16-bit output is the float output, rounded.
                        assert_eq!(at_full_scale, 0, "{} at {fs} Hz", e.label);
                        assert_eq!(unlike_hard_clip, 0, "{} at {fs} Hz: decode_i16 is not decode, rounded", e.label);
                    } else {
                        // The signal clips, and the soft clip changed samples
                        // that plain rounding would have left alone.
                        assert!(at_full_scale > 0, "{} at {fs} Hz: the signal did not reach full scale", e.label);
                        assert!(unlike_hard_clip > 0, "{} at {fs} Hz: no trace of the soft clip", e.label);
                    }
                }
            }
        }
    }
    println!("largest share of samples that differ by one step: {:.3} %", 100.0 * worst_fraction);
}

/// `reset_state` brings a decoder back to its state after construction.
#[test]
fn reset_state_gives_the_output_of_a_new_decoder() {
    for layout in [SURROUND_5_1, DISCRETE_3] {
        let e = encode(layout, VOIP_FEC, 16, 960, Signal::Voiced, 400, 1.0);
        let decode_all = |dec: &mut MultistreamDecoder| {
            let mut out = vec![0f32; 5760 * e.channels];
            let mut all = Vec::new();
            for packet in &e.packets {
                let n = dec.decode(packet, 5760, &mut out).unwrap();
                all.extend_from_slice(&out[..n * e.channels]);
            }
            all
        };
        let (_, mut dec) = e.decoders(48000);
        let first = decode_all(&mut dec);
        assert_ne!(dec.final_range(), 0);
        dec.reset_state();
        assert_eq!(dec.final_range(), 0, "{}: final range after a reset", e.label);
        let second = decode_all(&mut dec);
        assert!(first.iter().zip(&second).all(|(a, b)| a.to_bits() == b.to_bits()), "{}", e.label);
        // Without the reset the second pass starts from another state.
        let third = decode_all(&mut dec);
        assert!(first.iter().zip(&third).any(|(a, b)| a.to_bits() != b.to_bits()), "{}", e.label);
    }
}

/// The stream decoders can be reached, and the final range is the XOR of
/// theirs.
#[test]
fn final_range_is_the_xor_of_the_stream_decoders() {
    let e = encode(SURROUND_7_1, AUDIO, 32, 960, Signal::Tones, 200, 1.0);
    let (_, mut dec) = e.decoders(48000);
    assert_eq!(dec.layout().streams(), 5);
    assert_eq!(dec.sampling_rate(), 48000);
    let mut out = vec![0f32; 960 * 8];
    for (i, packet) in e.packets.iter().enumerate() {
        assert_eq!(dec.decode(packet, 960, &mut out), Ok(960));
        let mut xor = 0;
        for s in 0..5 {
            let stream = dec.stream_decoder(s).unwrap();
            assert_eq!(stream.channels(), if s < 3 { 2 } else { 1 });
            assert_eq!(stream.sampling_rate(), 48000);
            assert_ne!(stream.final_range(), 0);
            xor ^= stream.final_range();
        }
        assert_eq!(dec.final_range(), xor);
        assert_eq!(dec.final_range(), e.ranges[i]);
    }
    assert!(dec.stream_decoder(5).is_none());
    assert!(dec.stream_decoder_mut(5).is_none());
    // A change to one stream decoder shows in the multistream decoder.
    let before = dec.final_range();
    let last = dec.stream_decoder(4).unwrap().final_range();
    dec.stream_decoder_mut(4).unwrap().reset_state();
    assert_eq!(dec.final_range(), before ^ last);
}
