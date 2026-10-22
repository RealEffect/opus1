//! A wide search for differences between the decoded output of opus1 and
//! of libopus: every decode call is compared sample by sample, in float and
//! in 16 bits, and the final range after every decoded packet.
//!
//! The tests are ignored because they take minutes. Run them with
//!
//! ```text
//! cargo test --release --test core_decode_search -- --ignored --nocapture
//! ```
//!
//! On 64-bit ARM every sample must be libopus's. Elsewhere the tests print
//! what differs and do not fail.
//!
//! - `streams_of_libopus_and_opus1` needs the TSP speech corpus in
//!   `$OPUS1_CORPUS/tsp48k/48k` (`OPUS1_CORPUS` defaults to
//!   `<repo>/corpus`).
//! - `rfc8251_test_vectors` needs `$OPUS1_CORPUS/rfc8251/opus_newvectors`.
//! - `multistream_streams` needs nothing.
//!
//! A test whose corpus is missing says so and passes.

mod core_common;
mod k24_common;
mod ms_support;
mod multistream_dec_common;

use core_common::*;
use k24_common::resample;
use opus1::{Application, Bitrate, Channels, ForcedMode, OpusDecoder, OpusEncoder, SampleRate};
use std::f64::consts::PI;
use std::path::{Path, PathBuf};

/// The corpus root: `OPUS1_CORPUS`, else `<repo>/corpus`.
fn corpus_dir() -> PathBuf {
    match std::env::var_os("OPUS1_CORPUS") {
        Some(dir) if !dir.is_empty() => PathBuf::from(dir),
        _ => Path::new(env!("CARGO_MANIFEST_DIR")).join("corpus"),
    }
}

/// Fails on 64-bit ARM when anything differed.
fn conclude(what: &str, runs: usize, differing: usize) {
    println!("{what}: {differing} of {runs} decodes differ from libopus");
    if cfg!(target_arch = "aarch64") {
        assert_eq!(differing, 0, "{what}: {differing} of {runs} decodes differ from libopus");
    }
}

// ---------------------------------------------------------------------------
// Signals
// ---------------------------------------------------------------------------

/// The WAV files of the TSP corpus, sorted, or `None` when it is missing.
fn tsp_files() -> Option<Vec<PathBuf>> {
    let root = corpus_dir().join("tsp48k").join("48k");
    let mut files = Vec::new();
    for dir in std::fs::read_dir(&root).ok()?.flatten() {
        if dir.path().is_dir() {
            for file in std::fs::read_dir(dir.path()).ok()?.flatten() {
                if file.path().extension().is_some_and(|e| e == "wav") {
                    files.push(file.path());
                }
            }
        }
    }
    files.sort();
    (!files.is_empty()).then_some(files)
}

/// The samples of a 16-bit WAV file.
fn wav_samples(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).expect("read a WAV file");
    let mut pos = 12;
    while pos + 8 <= bytes.len() {
        let len = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into().unwrap()) as usize;
        if &bytes[pos..pos + 4] == b"data" {
            let data = &bytes[pos + 8..(pos + 8 + len).min(bytes.len())];
            return data.as_chunks::<2>().0.iter().map(|b| i16::from_le_bytes(*b) as f32 / 32768.0).collect();
        }
        pos += 8 + len + (len & 1);
    }
    panic!("{}: no data chunk", path.display());
}

/// `seconds` of 48 kHz speech, files of the corpus one after the other
/// from file `first` on, resampled to `rate`.
fn speech(files: &[PathBuf], rate: usize, seconds: f64, first: usize) -> Vec<f32> {
    let need = (seconds * 48000.0) as usize;
    let mut out = Vec::new();
    let mut k = first;
    while out.len() < need {
        out.extend(wav_samples(&files[k % files.len()]));
        k += 7;
    }
    out.truncate(need);
    resample(&out, 48000, rate)
}

/// Speech mixed with other content, changing every 1.5 s: speech, speech
/// with tones, tones with noise, near silence, and speech 8 dB louder than
/// full scale allows.
fn mixed(files: &[PathBuf], rate: usize, channels: usize, seconds: f64, seed: usize) -> Vec<f32> {
    let speech = speech(files, rate, seconds, seed);
    let n = speech.len();
    let mut state = 0x9e37_79b9u32.wrapping_add(seed as u32);
    let mut out = vec![0f32; n * channels];
    for i in 0..n {
        let t = i as f64 / rate as f64;
        let part = ((t / 1.5) as usize + seed) % 5;
        for c in 0..channels {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            let white = (state >> 8) as f32 / (1u32 << 23) as f32 - 1.0;
            let tone = (0.3 * (2.0 * PI * (330.0 + 110.0 * c as f64) * t).sin()
                + 0.2 * (2.0 * PI * (2900.0 + 700.0 * c as f64) * t).sin()) as f32;
            let s = speech[(i + c * 37) % n];
            out[i * channels + c] = match part {
                0 => s,
                1 => 0.5 * s + 0.5 * tone,
                2 => tone + 0.02 * white,
                3 => 0.0001 * white,
                _ => 2.5 * s,
            };
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Encoders
// ---------------------------------------------------------------------------

/// What a stream is coded with.
#[derive(Clone, Debug)]
struct Coding {
    rate: usize,
    channels: usize,
    application: i32,
    /// Frame durations in units of 2.5 ms, one after the other.
    durations: Vec<usize>,
    /// Bitrates in b/s; the bitrate changes every 10 packets.
    bitrates: Vec<i32>,
    fec: bool,
    /// Discontinuous transmission (libopus only).
    dtx: bool,
    force_hybrid: bool,
}

/// Packets of libopus's encoder.
fn c_encode(coding: &Coding, pcm: &[f32]) -> Vec<Vec<u8>> {
    let mut enc = CEncoder::new(coding.rate, coding.channels, coding.application);
    if coding.fec {
        enc.fec(true, 20);
    }
    if coding.dtx {
        enc.set(audiopus_sys::OPUS_SET_DTX_REQUEST, 1);
    }
    if coding.force_hybrid {
        enc.force_mode(MODE_HYBRID);
    }
    encode_frames(coding, pcm, |frame, bitrate| {
        enc.bitrate(bitrate);
        enc.encode(frame)
    })
}

/// Packets of opus1's encoder.
fn rs_encode(coding: &Coding, pcm: &[f32]) -> Vec<Vec<u8>> {
    let application = match coding.application {
        APP_VOIP => Application::Voip,
        APP_AUDIO => Application::Audio,
        _ => Application::RestrictedLowDelay,
    };
    let mut enc = OpusEncoder::new(
        SampleRate::try_from(coding.rate as i32).unwrap(),
        Channels::try_from(coding.channels).unwrap(),
        application,
    )
    .unwrap();
    if coding.fec {
        enc.set_inband_fec(true);
        enc.set_packet_loss_perc(20).unwrap();
    }
    if coding.force_hybrid {
        enc.set_force_mode(Some(ForcedMode::Hybrid));
    }
    let mut buf = vec![0u8; 4000];
    let channels = coding.channels;
    encode_frames(coding, pcm, |frame, bitrate| {
        enc.set_bitrate(Bitrate::BitsPerSecond(bitrate)).unwrap();
        let len = enc.encode(frame, frame.len() / channels, &mut buf).unwrap();
        buf[..len].to_vec()
    })
}

/// Cuts `pcm` into the frames of `coding` and codes each with `encode`.
fn encode_frames(coding: &Coding, pcm: &[f32], mut encode: impl FnMut(&[f32], i32) -> Vec<u8>) -> Vec<Vec<u8>> {
    let mut packets = Vec::new();
    let mut pos = 0;
    for k in 0.. {
        let n = coding.durations[k % coding.durations.len()] * coding.rate / 400 * coding.channels;
        if pos + n > pcm.len() {
            break;
        }
        packets.push(encode(&pcm[pos..pos + n], coding.bitrates[k / 10 % coding.bitrates.len()]));
        pos += n;
    }
    packets
}

// ---------------------------------------------------------------------------
// Decoding with losses
// ---------------------------------------------------------------------------

/// How packets are lost.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Loss {
    None,
    /// About one packet in ten, concealed.
    Random,
    /// About one packet in ten, rebuilt from the FEC data of the next one.
    Fec,
    /// Three packets in a row out of every 23, concealed.
    Burst,
    /// About one packet in six, concealed in pieces of 2.5 ms, 7.5 ms and
    /// the rest.
    Pieces,
    /// Two packets in a row, now and then, rebuilt by one FEC call for the
    /// duration of both.
    Fec2,
    /// 8 packets in a row out of every 40, concealed in one call.
    Long,
}

const LOSSES: [Loss; 7] = [Loss::None, Loss::Random, Loss::Fec, Loss::Burst, Loss::Pieces, Loss::Fec2, Loss::Long];

/// One decode call.
#[derive(Clone, Copy, Debug)]
enum Call {
    /// Decode packet `k`.
    Packet(usize),
    /// Conceal this many samples per channel.
    Conceal(usize),
    /// Rebuild this many samples per channel from the FEC data of packet
    /// `k`.
    Fec(usize, usize),
}

/// The calls that decode `packets` at `rate` with `loss`.
fn calls(packets: &[Vec<u8>], rate: usize, loss: Loss, seed: u32) -> Vec<Call> {
    let duration = |k: usize| opus1::packet::nb_samples(&packets[k], SampleRate::try_from(rate as u32).unwrap()).unwrap();
    let mut state = seed;
    let mut calls = Vec::new();
    let mut k = 0;
    while k < packets.len() {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        let r = state >> 24;
        let n = duration(k);
        let mut skip = 1;
        match loss {
            Loss::Random if k > 2 && r < 26 => calls.push(Call::Conceal(n)),
            Loss::Fec if k > 2 && r < 26 && k + 1 < packets.len() => calls.push(Call::Fec(k + 1, n)),
            Loss::Burst if k > 2 && k % 23 >= 20 => calls.push(Call::Conceal(n)),
            Loss::Pieces if k > 2 && r < 40 => {
                let unit = rate / 400;
                let mut left = n;
                for piece in [unit, 3 * unit] {
                    if left > piece {
                        calls.push(Call::Conceal(piece));
                        left -= piece;
                    }
                }
                calls.push(Call::Conceal(left));
            }
            Loss::Fec2 if k > 2 && r < 30 && k + 2 < packets.len() => {
                calls.push(Call::Fec(k + 2, n + duration(k + 1)));
                skip = 2;
            }
            Loss::Long if k > 2 && k % 40 == 20 && k + 8 < packets.len() => {
                calls.push(Call::Conceal((k..k + 8).map(duration).sum()));
                skip = 8;
            }
            _ => calls.push(Call::Packet(k)),
        }
        k += skip;
    }
    calls
}

/// Decodes `packets` with libopus and opus1, in float and in 16 bits.
/// Returns the first difference.
fn first_difference(packets: &[Vec<u8>], rate: usize, channels: usize, loss: Loss, seed: u32) -> Option<String> {
    let mut c = CDecoder::new(rate, channels);
    let mut c16 = CDecoder::new(rate, channels);
    let mut ours = OpusDecoder::new(
        SampleRate::try_from(rate as i32).unwrap(),
        Channels::try_from(channels).unwrap(),
    )
    .unwrap();
    let mut ours16 = OpusDecoder::new(
        SampleRate::try_from(rate as i32).unwrap(),
        Channels::try_from(channels).unwrap(),
    )
    .unwrap();
    // 8 packets of 120 ms.
    let mut out = vec![0f32; rate * channels];
    let mut out16 = vec![0i16; rate * channels];
    for call in calls(packets, rate, loss, seed) {
        let (what, reference, reference16, got, got16) = match call {
            Call::Packet(k) => {
                let p = &packets[k];
                let n = opus1::packet::nb_samples(p, SampleRate::try_from(rate as u32).unwrap()).unwrap();
                (
                    format!("packet {k} (TOC {:02x})", p[0]),
                    c.decode(Some(p), n, false),
                    c16.decode_i16(Some(p), n, false),
                    ours.decode(p, n, &mut out),
                    ours16.decode_i16(p, n, &mut out16),
                )
            }
            Call::Conceal(n) => (
                format!("{n} concealed samples"),
                c.decode(None, n, false),
                c16.decode_i16(None, n, false),
                ours.decode(&[], n, &mut out),
                ours16.decode_i16(&[], n, &mut out16),
            ),
            Call::Fec(k, n) => {
                let p = &packets[k];
                (
                    format!("{n} samples rebuilt from packet {k}"),
                    c.decode(Some(p), n, true),
                    c16.decode_i16(Some(p), n, true),
                    ours.decode_fec(p, n, &mut out),
                    ours16.decode_fec_i16(p, n, &mut out16),
                )
            }
        };
        match (reference, got) {
            (Ok(reference), Ok(n)) => {
                let ours = &out[..n * channels];
                if reference.len() != ours.len() || reference.iter().zip(ours).any(|(a, b)| a.to_bits() != b.to_bits()) {
                    return Some(format!("{what}: float output"));
                }
            }
            (reference, got) if reference.is_ok() != got.is_ok() => return Some(format!("{what}: libopus {reference:?}, opus1 {got:?}")),
            _ => {}
        }
        match (reference16, got16) {
            (Ok(reference), Ok(n)) if reference[..] != out16[..n * channels] => return Some(format!("{what}: 16-bit output")),
            (reference, got) if reference.is_ok() != got.is_ok() => return Some(format!("{what}: 16 bits: libopus {reference:?}, opus1 {got:?}")),
            _ => {}
        }
        if matches!(call, Call::Packet(_)) && (c.final_range() != ours.final_range() || c16.final_range() != ours16.final_range()) {
            return Some(format!("{what}: final range"));
        }
    }
    None
}

/// Decodes `packets` at every output rate with 1 and 2 channels and each
/// loss pattern. Returns (decodes, decodes that differ).
fn decode_everywhere(label: &str, packets: &[Vec<u8>], fec: bool, seed: u32) -> (usize, usize) {
    let (mut runs, mut differing) = (0, 0);
    for rate in [8000, 12000, 16000, 24000, 48000] {
        for channels in [1, 2] {
            for loss in LOSSES {
                if !fec && matches!(loss, Loss::Fec | Loss::Fec2) {
                    continue;
                }
                runs += 1;
                if let Some(what) = first_difference(packets, rate, channels, loss, seed) {
                    differing += 1;
                    println!("DIFF {label}, decoded at {rate} Hz, {channels} ch, {loss:?}: {what}");
                }
            }
        }
    }
    (runs, differing)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Streams coded by libopus and by opus1: 8 s of mixed content at every
/// rate, mono and stereo, for each application, with a bitrate that
/// changes every 10 packets so that the mode switches, frames of 2.5 to
/// 120 ms, FEC, DTX (libopus only) and forced Hybrid. 228 streams, 13,160
/// decodes.
#[test]
#[ignore = "takes about 4 minutes; needs the TSP corpus"]
fn streams_of_libopus_and_opus1() {
    let Some(files) = tsp_files() else {
        println!("skipped: the TSP corpus is not in {}", corpus_dir().display());
        return;
    };
    let mut codings = Vec::new();
    for rate in [8000, 12000, 16000, 24000, 48000] {
        for channels in [1usize, 2] {
            let ch = channels as i32;
            let coding = |application, durations: &[usize], bitrates: &[i32], fec, dtx| Coding {
                rate,
                channels,
                application,
                durations: durations.to_vec(),
                bitrates: bitrates.to_vec(),
                fec,
                dtx,
                force_hybrid: false,
            };
            for application in [APP_VOIP, APP_AUDIO, APP_LOWDELAY] {
                codings.push(coding(application, &[8], &[8000 * ch, 16000 * ch, 32000 * ch, 64000 * ch, 12000 * ch, 24000 * ch], false, false));
                codings.push(coding(application, &[8, 4, 8, 16, 24, 2, 1], &[10000 * ch, 20000 * ch, 40000 * ch, 80000 * ch, 14000 * ch], false, false));
                if application != APP_LOWDELAY {
                    codings.push(coding(application, &[8], &[12000 * ch, 20000 * ch, 32000 * ch, 16000 * ch], true, false));
                    codings.push(coding(application, &[8, 24, 16], &[9000 * ch, 28000 * ch, 18000 * ch], true, true));
                }
            }
            codings.push(coding(APP_VOIP, &[32, 40, 48, 8], &[6000, 16000 * ch, 40000 * ch], true, false));
            codings.push(coding(APP_AUDIO, &[32, 48, 16, 4, 1], &[64000 * ch, 9000, 24000 * ch], false, false));
            if rate >= 24000 {
                let mut hybrid = coding(APP_AUDIO, &[4, 8], &[24000 * ch, 48000], true, false);
                hybrid.force_hybrid = true;
                codings.push(hybrid);
            }
        }
    }
    let (mut runs, mut differing, mut streams) = (0, 0, 0);
    for (i, coding) in codings.iter().enumerate() {
        let pcm = mixed(&files, coding.rate, coding.channels, 8.0, i);
        let mut encoded = vec![("libopus", c_encode(coding, &pcm))];
        if !coding.dtx {
            encoded.push(("opus1", rs_encode(coding, &pcm)));
        }
        for (encoder, packets) in encoded {
            streams += 1;
            let label = format!("{encoder} {coding:?}");
            let (r, d) = decode_everywhere(&label, &packets, coding.fec, 77 + streams);
            runs += r;
            differing += d;
        }
    }
    println!("{streams} streams");
    conclude("streams of libopus and opus1", runs, differing);
}

/// The packets of an `opus_demo` bitstream: a 4-byte big-endian length, a
/// 4-byte final range, then the packet.
fn read_bitstream(path: &Path) -> Vec<Vec<u8>> {
    let bytes = std::fs::read(path).expect("read a test vector");
    let mut packets = Vec::new();
    let mut pos = 0;
    while pos + 8 <= bytes.len() {
        let len = u32::from_be_bytes(bytes[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 8;
        packets.push(bytes[pos..pos + len].to_vec());
        pos += len;
    }
    packets
}

/// The 12 test vectors of RFC 8251, decoded with the same losses: 840
/// decodes.
#[test]
#[ignore = "takes about 2 minutes; needs the RFC 8251 test vectors"]
fn rfc8251_test_vectors() {
    let dir = corpus_dir().join("rfc8251").join("opus_newvectors");
    if !dir.join("testvector01.bit").exists() {
        println!("skipped: the RFC 8251 test vectors are not in {}", dir.display());
        return;
    }
    let (mut runs, mut differing) = (0, 0);
    for v in 1..=12u32 {
        let packets = read_bitstream(&dir.join(format!("testvector{v:02}.bit")));
        // The losses come from the patterns, not from empty packets.
        let packets: Vec<Vec<u8>> = packets.into_iter().filter(|p| !p.is_empty()).collect();
        let (r, d) = decode_everywhere(&format!("test vector {v}"), &packets, true, 5 + v);
        runs += r;
        differing += d;
    }
    conclude("RFC 8251 test vectors", runs, differing);
}

/// Multistream streams of libopus: every layout, 3 encoder settings, 5
/// bitrates, 5 frame durations and 2 signals, some loud enough for the soft
/// clipper. Each is decoded at every output rate with concealed and rebuilt
/// packets: 7,500 decodes.
#[test]
#[ignore = "takes about 4 minutes"]
fn multistream_streams() {
    use multistream_dec_common::*;
    let same = |a: &[f32], b: &[f32]| a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits());
    let (mut runs, mut differing) = (0, 0);
    for layout in ALL_LAYOUTS {
        for settings in [AUDIO, VOIP_FEC, AUDIO_CBR] {
            for kbps in [12, 16, 24, 32, 64] {
                for frame in [120usize, 480, 960, 1920, 2880] {
                    for signal in [Signal::Voiced, Signal::Tones] {
                        let gain = if kbps == 64 { 5.0 } else { 1.0 };
                        let e = encode(layout, settings, kbps, frame, signal, 1500, gain);
                        for fs in [8000, 12000, 16000, 24000, 48000] {
                            runs += 1;
                            let (mut c, mut ours) = e.decoders(fs);
                            let (mut c16, mut ours16) = e.decoders(fs);
                            let ch = e.channels;
                            let n = frame * fs as usize / 48000;
                            let mut out = vec![0f32; n * ch];
                            let mut out16 = vec![0i16; n * ch];
                            let lost = |i: usize| i % 5 == 4 || i % 20 == 13 || (30..33).contains(&i);
                            let mut first = None;
                            let mut check = |what: String, reference: Vec<f32>, got: usize, out: &[f32], reference16: Vec<i16>, got16: usize, out16: &[i16]| {
                                if first.is_none() && (!same(&reference, &out[..got * ch]) || reference16[..] != out16[..got16 * ch]) {
                                    first = Some(what);
                                }
                            };
                            for (i, p) in e.packets.iter().enumerate() {
                                if lost(i) {
                                    continue;
                                }
                                if i > 0 && lost(i - 1) {
                                    // Conceal all lost packets but the last,
                                    // then rebuild the last from FEC data
                                    // (or conceal it without FEC).
                                    let mut j = i - 1;
                                    while j > 0 && lost(j - 1) {
                                        j -= 1;
                                    }
                                    for _ in j..i - 1 {
                                        check(
                                            format!("concealed before packet {i}"),
                                            c.decode_float(None, n, false).unwrap(),
                                            ours.decode(&[], n, &mut out).unwrap(),
                                            &out,
                                            c16.decode_i16(None, n, false).unwrap(),
                                            ours16.decode_i16(&[], n, &mut out16).unwrap(),
                                            &out16,
                                        );
                                    }
                                    if settings.fec {
                                        check(
                                            format!("rebuilt from packet {i}"),
                                            c.decode_float(Some(p), n, true).unwrap(),
                                            ours.decode_fec(p, n, &mut out).unwrap(),
                                            &out,
                                            c16.decode_i16(Some(p), n, true).unwrap(),
                                            ours16.decode_fec_i16(p, n, &mut out16).unwrap(),
                                            &out16,
                                        );
                                    } else {
                                        check(
                                            format!("concealed before packet {i}"),
                                            c.decode_float(None, n, false).unwrap(),
                                            ours.decode(&[], n, &mut out).unwrap(),
                                            &out,
                                            c16.decode_i16(None, n, false).unwrap(),
                                            ours16.decode_i16(&[], n, &mut out16).unwrap(),
                                            &out16,
                                        );
                                    }
                                }
                                check(
                                    format!("packet {i}"),
                                    c.decode_float(Some(p), n, false).unwrap(),
                                    ours.decode(p, n, &mut out).unwrap(),
                                    &out,
                                    c16.decode_i16(Some(p), n, false).unwrap(),
                                    ours16.decode_i16(p, n, &mut out16).unwrap(),
                                    &out16,
                                );
                            }
                            if let Some(what) = first {
                                differing += 1;
                                println!("DIFF {} at {fs} Hz: {what}", e.label);
                            }
                        }
                    }
                }
            }
        }
    }
    conclude("multistream streams", runs, differing);
}
