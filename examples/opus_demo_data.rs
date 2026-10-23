//! Data generator for the Opus explainer page (docs/opus-explained/).
//!
//! It encodes and decodes real audio with opus1 and writes everything the
//! page visualizes into one JavaScript file:
//!
//!     docs/opus-explained/data/opus-demo-data.js
//!     window.OPUS_DEMO_DATA = { meta, speech, music, mode_map };
//!
//! Audio is stored as base64 of little-endian int16 samples, packets as
//! base64 of their raw bytes. The output depends only on the source files and
//! the git commit it records (with "-dirty" when those sources have
//! uncommitted changes): the music is synthesized with a fixed-seed noise
//! generator and no step reads the clock.
//!
//! What it produces:
//!   A. speech: a 3.0 s excerpt of fixtures/answer_16k.wav, coded at six
//!      bitrates (SILK up to 16 kb/s; at 24 and 32 kb/s the encoder codes most
//!      packets with CELT); per-frame SILK parameters read back from the real
//!      16 kb/s bitstream; excitation detail for three frames; a packet-loss
//!      run.
//!   B. music: a synthetic 48 kHz signal (pad chord, plucked melody,
//!      percussive hits, high-frequency air), coded at six bitrates; per-frame
//!      CELT side information read back from the real bitstream.
//!   C. mode_map: which mode / bandwidth the encoder picks per bitrate, for
//!      opus1 and, as a reference, for the C libopus linked as the `opus`
//!      dev-dependency.
//!
//! How the per-frame parameters are read: the SILK and CELT decoders keep
//! them in local variables, so this program replays the bitstream parsing
//! with the crate's internal functions (`opus1::internals`: `silk_decode_indices`,
//! `unquant_coarse_energy`, `clt_compute_allocation`, `quant_all_bands`, ...)
//! in a copy of the decoder state. Each replayed frame is checked against the
//! real decoder: the range decoder must end in the same state, and the
//! replayed SILK samples must equal the real output. If any check fails, the
//! data file is left unchanged.
//!
//! Run from the crate root:
//!     cargo run --release --example opus_demo_data
//!
//! The `opus` dev-dependency links the C libopus: build it with cmake, or set
//! LIBOPUS_LIB_DIR to an installed libopus and LIBOPUS_NO_PKG=1.

// A few loops index several arrays at once, as the C reference does.
#![allow(clippy::needless_range_loop)]

use opus1::internals::bands::{anti_collapse, denormalise_bands, log2amp, quant_all_bands};
use opus1::internals::celt::{CeltDecoder, PcmLayout};
use opus1::internals::modes::{
    CeltMode, SPREAD_ICDF, TAPSET_ICDF, TF_SELECT_TABLE, TRIM_ICDF, default_mode,
};
use opus1::internals::quant_bands::{
    unquant_coarse_energy, unquant_energy_finalise, unquant_fine_energy,
};
use opus1::internals::range_coder::RangeCoder;
use opus1::internals::rate::{BITRES, clt_compute_allocation};
use opus1::internals::silk::dec_api::SilkDecoder;
use opus1::internals::silk::decode_core::silk_decode_core;
use opus1::internals::silk::decode_frame::FLAG_DECODE_NORMAL;
use opus1::internals::silk::decode_indices::silk_decode_indices;
use opus1::internals::silk::decode_parameters::silk_decode_parameters;
use opus1::internals::silk::decode_pulses::silk_decode_pulses;
use opus1::internals::silk::decoder_structs::{SilkDecControl, SilkDecoderControl};
use opus1::internals::silk::define::{CODE_INDEPENDENTLY, MAX_FRAME_LENGTH};
use opus1::internals::silk::init_decoder::silk_decoder_set_fs;
use opus1::{Application, Bitrate, Channels, OpusDecoder, OpusEncoder, SampleRate};
use std::fmt::Write as _;
use std::fs;
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------

/// Start of the speech excerpt in fixtures/answer_16k.wav, in seconds.
/// Chosen from the energy envelope and the SILK frame types. The 3 s from
/// here hold strongly periodic voiced speech, a few unvoiced frames, and two
/// quiet gaps (about 2.44-2.64 s and 2.76-2.94 s in the file; the second
/// holds the quietest frames of the whole file). At 16 kb/s SILK labels the
/// 150 frames 118 voiced, 5 unvoiced and 27 inactive. A start of 9.50 s gives
/// more unvoiced frames (about 27) but weaker voicing and less level
/// contrast between speech and pauses.
const SPEECH_START_S: f64 = 1.30;
const SPEECH_DUR_S: f64 = 3.0;
const SPEECH_RATE: i32 = 16000;
const SPEECH_BITRATES: [i32; 6] = [6000, 9000, 12000, 16000, 24000, 32000];
/// The speech stream whose SILK parameters and packet loss are traced.
const SPEECH_TRACE_BITRATE: i32 = 16000;
/// Lost packets (frame indices into the 16 kb/s speech stream).
/// One single loss, a burst of two and a burst of three, all in voiced
/// speech (SILK frame types of the 3 s excerpt at 16 kb/s).
const PLC_LOST_FRAMES: [usize; 6] = [12, 44, 45, 100, 101, 102];

const MUSIC_RATE: i32 = 48000;
const MUSIC_DUR_S: f64 = 2.5;
const MUSIC_BITRATES: [i32; 6] = [12000, 24000, 48000, 64000, 96000, 128000];
/// The music stream whose CELT side information is traced.
const MUSIC_TRACE_BITRATE: i32 = 64000;

const MODE_MAP_KBPS: [i32; 13] = [6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64, 96, 128];

const FRAME_MS: usize = 20;
const COMPLEXITY: u8 = 9;
/// Silent frames appended to each encode so the codec delay can be flushed
/// and the decoded output covers the whole input after alignment.
const FLUSH_FRAMES: usize = 3;
/// Largest codec delay searched by cross-correlation, in samples.
const MAX_DELAY_SEARCH: usize = 1500;

// ---------------------------------------------------------------------------
// Small JSON writer (the crate has no serde; this keeps the example std-only)
// ---------------------------------------------------------------------------

enum J {
    Null,
    Bool(bool),
    Int(i64),
    /// A float written with at most this many decimals.
    Num(f64, usize),
    Str(String),
    /// A string written like `Str`, but whose length does not count against
    /// `INLINE_MAX`, so the object holding it can still stay on one line.
    Blob(String),
    Arr(Vec<J>),
    Obj(Vec<(String, J)>),
}

fn obj(fields: Vec<(&str, J)>) -> J {
    J::Obj(
        fields
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect(),
    )
}
fn s(v: &str) -> J {
    J::Str(v.to_string())
}
fn int(v: i64) -> J {
    J::Int(v)
}
fn num(v: f64, decimals: usize) -> J {
    J::Num(v, decimals)
}
fn ints<T: Copy + Into<i64>>(v: &[T]) -> J {
    J::Arr(v.iter().map(|&x| J::Int(x.into())).collect())
}
fn nums(v: &[f64], decimals: usize) -> J {
    J::Arr(v.iter().map(|&x| J::Num(x, decimals)).collect())
}
fn opt<T>(v: Option<T>, f: impl FnOnce(T) -> J) -> J {
    v.map(f).unwrap_or(J::Null)
}

fn write_num(out: &mut String, v: f64, decimals: usize) {
    if !v.is_finite() {
        out.push_str("null");
        return;
    }
    let mut t = format!("{:.*}", decimals, v);
    if t.contains('.') {
        while t.ends_with('0') {
            t.pop();
        }
        if t.ends_with('.') {
            t.pop();
        }
    }
    if t == "-0" {
        t = "0".to_string();
    }
    out.push_str(&t);
}

fn write_str(out: &mut String, v: &str) {
    out.push('"');
    for c in v.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

fn is_scalar(j: &J) -> bool {
    !matches!(j, J::Arr(_) | J::Obj(_))
}

/// Objects at most this long (written compactly) go on one line.
const INLINE_MAX: usize = 1200;

/// Compact one-line form of `j`, or `None` if it would exceed `INLINE_MAX`.
fn compact(j: &J) -> Option<String> {
    // `blob` counts the characters of `J::Blob` strings, which are exempt.
    fn go(out: &mut String, j: &J, blob: &mut usize) -> bool {
        match j {
            J::Arr(items) => {
                out.push('[');
                for (i, it) in items.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    if !go(out, it, blob) {
                        return false;
                    }
                }
                out.push(']');
            }
            J::Obj(fields) => {
                out.push('{');
                for (i, (k, v)) in fields.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    write_str(out, k);
                    out.push_str(": ");
                    if !go(out, v, blob) {
                        return false;
                    }
                }
                out.push('}');
            }
            J::Str(v) if v.len() > INLINE_MAX => return false,
            J::Blob(v) => {
                write_str(out, v);
                *blob += v.len() + 2;
            }
            scalar => write_json(out, scalar, 0),
        }
        out.len() <= INLINE_MAX + *blob
    }
    let mut out = String::new();
    let mut blob = 0;
    go(&mut out, j, &mut blob).then_some(out)
}

/// Objects go one field per line; arrays of scalars stay on one line.
fn write_json(out: &mut String, j: &J, indent: usize) {
    match j {
        J::Null => out.push_str("null"),
        J::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        J::Int(v) => {
            let _ = write!(out, "{}", v);
        }
        J::Num(v, d) => write_num(out, *v, *d),
        J::Str(v) | J::Blob(v) => write_str(out, v),
        J::Arr(items) => {
            if items.iter().all(is_scalar) {
                out.push('[');
                for (i, it) in items.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    write_json(out, it, indent);
                }
                out.push(']');
            } else {
                out.push_str("[\n");
                for (i, it) in items.iter().enumerate() {
                    out.push_str(&" ".repeat(indent + 2));
                    write_json(out, it, indent + 2);
                    if i + 1 < items.len() {
                        out.push(',');
                    }
                    out.push('\n');
                }
                out.push_str(&" ".repeat(indent));
                out.push(']');
            }
        }
        J::Obj(fields) => {
            if let Some(line) = compact(j) {
                out.push_str(&line);
                return;
            }
            out.push_str("{\n");
            for (i, (k, v)) in fields.iter().enumerate() {
                out.push_str(&" ".repeat(indent + 2));
                write_str(out, k);
                out.push_str(": ");
                write_json(out, v, indent + 2);
                if i + 1 < fields.len() {
                    out.push(',');
                }
                out.push('\n');
            }
            out.push_str(&" ".repeat(indent));
            out.push('}');
        }
    }
}

fn json_len(j: &J) -> usize {
    let mut t = String::new();
    write_json(&mut t, j, 0);
    t.len()
}

// ---------------------------------------------------------------------------
// Base64 and PCM helpers
// ---------------------------------------------------------------------------

fn base64(bytes: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b = [
            chunk[0],
            *chunk.get(1).unwrap_or(&0),
            *chunk.get(2).unwrap_or(&0),
        ];
        let n = ((b[0] as u32) << 16) | ((b[1] as u32) << 8) | b[2] as u32;
        out.push(T[(n >> 18) as usize & 63] as char);
        out.push(T[(n >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 {
            T[(n >> 6) as usize & 63] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            T[n as usize & 63] as char
        } else {
            '='
        });
    }
    out
}

fn b64_pcm(pcm: &[i16]) -> J {
    let bytes: Vec<u8> = pcm.iter().flat_map(|v| v.to_le_bytes()).collect();
    J::Str(base64(&bytes))
}

/// Float decoder output (scale 1.0 = 32768) to int16, rounding like
/// libopus `FLOAT2INT16`.
fn f32_to_i16(v: f32) -> i16 {
    (v * 32768.0).round().clamp(-32768.0, 32767.0) as i16
}

fn rms(x: &[i16]) -> f64 {
    if x.is_empty() {
        return 0.0;
    }
    (x.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / x.len() as f64).sqrt()
}

fn snr_db(reference: &[i16], test: &[i16]) -> f64 {
    let mut sig = 0.0;
    let mut err = 0.0;
    for (&r, &t) in reference.iter().zip(test) {
        sig += (r as f64).powi(2);
        err += (r as f64 - t as f64).powi(2);
    }
    10.0 * (sig / err.max(1e-9)).log10()
}

/// Segmental SNR over 20 ms frames, each frame clamped to [-10, 35] dB.
fn seg_snr_db(reference: &[i16], test: &[i16], frame: usize) -> f64 {
    let mut total = 0.0;
    let mut count = 0usize;
    for (r, t) in reference.chunks(frame).zip(test.chunks(frame)) {
        if rms(r) < 1.0 {
            continue;
        }
        total += snr_db(r, t).clamp(-10.0, 35.0);
        count += 1;
    }
    if count == 0 {
        0.0
    } else {
        total / count as f64
    }
}

/// Codec delay: the lag that maximizes the normalized cross-correlation
/// between the input and the decoded output (decoded[n + lag] ~ input[n]).
fn find_delay(input: &[i16], decoded: &[i16]) -> usize {
    let mut best = (f64::NEG_INFINITY, 0usize);
    for lag in 0..=MAX_DELAY_SEARCH {
        let n = input.len().min(decoded.len().saturating_sub(lag));
        let mut xy = 0.0;
        let mut yy = 1e-9;
        for i in 0..n {
            let y = decoded[i + lag] as f64;
            xy += input[i] as f64 * y;
            yy += y * y;
        }
        let r = xy / yy.sqrt();
        if r > best.0 {
            best = (r, lag);
        }
    }
    best.1
}

/// Normalized cross-correlation at one lag (1.0 = identical shape).
fn ncc_at(input: &[i16], decoded: &[i16], lag: usize) -> f64 {
    let n = input.len().min(decoded.len().saturating_sub(lag));
    let (mut xy, mut xx, mut yy) = (0.0, 1e-9, 1e-9);
    for i in 0..n {
        let x = input[i] as f64;
        let y = decoded[i + lag] as f64;
        xy += x * y;
        xx += x * x;
        yy += y * y;
    }
    xy / (xx * yy).sqrt()
}

// ---------------------------------------------------------------------------
// WAV input
// ---------------------------------------------------------------------------

fn read_wav_mono16(path: &Path) -> (u32, Vec<i16>) {
    let data = fs::read(path).unwrap_or_else(|e| panic!("cannot read {:?}: {}", path, e));
    assert!(
        &data[0..4] == b"RIFF" && &data[8..12] == b"WAVE",
        "not a WAV file"
    );
    let mut pos = 12;
    let mut rate = 0u32;
    let mut format_ok = false;
    while pos + 8 <= data.len() {
        let id = &data[pos..pos + 4];
        let size = u32::from_le_bytes(data[pos + 4..pos + 8].try_into().unwrap()) as usize;
        let body = &data[pos + 8..(pos + 8 + size).min(data.len())];
        if id == b"fmt " {
            let format = u16::from_le_bytes([body[0], body[1]]);
            let channels = u16::from_le_bytes([body[2], body[3]]);
            rate = u32::from_le_bytes(body[4..8].try_into().unwrap());
            let bits = u16::from_le_bytes([body[14], body[15]]);
            assert!(
                format == 1 && channels == 1 && bits == 16,
                "need mono 16-bit PCM"
            );
            format_ok = true;
        } else if id == b"data" {
            assert!(format_ok, "data chunk before fmt chunk");
            let pcm = body
                .as_chunks::<2>()
                .0
                .iter()
                .map(|&c| i16::from_le_bytes(c))
                .collect();
            return (rate, pcm);
        }
        pos += 8 + size + (size & 1);
    }
    panic!("no data chunk in {:?}", path);
}

// ---------------------------------------------------------------------------
// ToC byte (RFC 6716 section 3.1)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct Toc {
    config: u8,
    mode: &'static str,
    bandwidth: &'static str,
    frame_ms: f64,
    stereo: bool,
    code: u8,
}

fn parse_toc(toc: u8) -> Toc {
    let config = toc >> 3;
    let (mode, bandwidth, frame_ms) = match config {
        0..=11 => (
            "SILK",
            ["NB", "MB", "WB"][config as usize / 4],
            [10.0, 20.0, 40.0, 60.0][config as usize % 4],
        ),
        12..=15 => (
            "Hybrid",
            if config < 14 { "SWB" } else { "FB" },
            [10.0, 20.0][config as usize % 2],
        ),
        _ => (
            "CELT",
            ["NB", "WB", "SWB", "FB"][(config as usize - 16) / 4],
            [2.5, 5.0, 10.0, 20.0][config as usize % 4],
        ),
    };
    Toc {
        config,
        mode,
        bandwidth,
        frame_ms,
        stereo: toc & 0x04 != 0,
        code: toc & 0x03,
    }
}

/// Last CELT band coded for a bandwidth (libopus `OPUS_SET_BANDWIDTH`).
fn celt_end_band(bandwidth: &str) -> usize {
    match bandwidth {
        "NB" => 13,
        "MB" | "WB" => 17,
        "SWB" => 19,
        _ => 21,
    }
}

// ---------------------------------------------------------------------------
// Encode / decode helpers
// ---------------------------------------------------------------------------

struct Stream {
    rate: i32,
    bitrate: i32,
    /// One packet per 20 ms frame, including `FLUSH_FRAMES` trailing ones.
    packets: Vec<Vec<u8>>,
    /// Decoded output, not yet aligned (codec delay still in front).
    decoded_raw: Vec<i16>,
    delay: usize,
}

impl Stream {
    fn aligned(&self, n: usize) -> &[i16] {
        &self.decoded_raw[self.delay..self.delay + n]
    }
}

fn pad_for_flush(pcm: &[i16], frame: usize) -> Vec<i16> {
    let mut padded = pcm.to_vec();
    let target = (pcm.len().div_ceil(frame) + FLUSH_FRAMES) * frame;
    padded.resize(target, 0);
    padded
}

fn sample_rate(rate: i32) -> SampleRate {
    SampleRate::try_from(rate).expect("an Opus sample rate")
}

fn encode_opus1(pcm: &[i16], rate: i32, app: Application, bitrate: i32) -> Vec<Vec<u8>> {
    let frame = rate as usize * FRAME_MS / 1000;
    let mut enc = OpusEncoder::new(sample_rate(rate), Channels::Mono, app).expect("encoder");
    enc.set_bitrate(Bitrate::BitsPerSecond(
        NonZeroU32::new(bitrate as u32).expect("bitrate"),
    ));
    enc.set_complexity(COMPLEXITY).expect("complexity");
    enc.set_vbr(true);
    let mut out = Vec::new();
    for chunk in pad_for_flush(pcm, frame).chunks(frame) {
        let mut buf = vec![0u8; 1500];
        let n = enc.encode_i16(chunk, frame, &mut buf).expect("encode");
        buf.truncate(n);
        out.push(buf);
    }
    out
}

/// Decode a packet sequence. An index in `lost` is passed to the decoder as
/// an empty packet, which is how the API signals a lost packet.
fn decode_opus1(packets: &[Vec<u8>], rate: i32, lost: &[usize]) -> Vec<i16> {
    let frame = rate as usize * FRAME_MS / 1000;
    let mut dec = OpusDecoder::new(sample_rate(rate), Channels::Mono).expect("decoder");
    let mut out = Vec::with_capacity(packets.len() * frame);
    let mut buf = vec![0f32; frame];
    for (i, p) in packets.iter().enumerate() {
        let input: &[u8] = if lost.contains(&i) { &[] } else { p };
        let n = dec.decode(input, frame, &mut buf).expect("decode");
        out.extend(buf[..n].iter().map(|&v| f32_to_i16(v)));
    }
    out
}

fn make_stream(pcm: &[i16], rate: i32, app: Application, bitrate: i32) -> Stream {
    let packets = encode_opus1(pcm, rate, app, bitrate);
    let decoded_raw = decode_opus1(&packets, rate, &[]);
    let delay = find_delay(pcm, &decoded_raw);
    assert!(delay + pcm.len() <= decoded_raw.len(), "flush too short");
    Stream {
        rate,
        bitrate,
        packets,
        decoded_raw,
        delay,
    }
}

fn mode_counts(packets: &[Vec<u8>]) -> (usize, usize, usize) {
    let mut c = (0, 0, 0);
    for p in packets {
        match parse_toc(p[0]).mode {
            "SILK" => c.0 += 1,
            "Hybrid" => c.1 += 1,
            _ => c.2 += 1,
        }
    }
    c
}

/// Most common value of `key` over the packets.
fn majority(packets: &[Vec<u8>], key: impl Fn(&Toc) -> String) -> String {
    let mut counts: Vec<(String, usize)> = Vec::new();
    for p in packets {
        let k = key(&parse_toc(p[0]));
        match counts.iter_mut().find(|(v, _)| *v == k) {
            Some((_, n)) => *n += 1,
            None => counts.push((k, 1)),
        }
    }
    counts.sort_by_key(|c| std::cmp::Reverse(c.1));
    counts[0].0.clone()
}

/// The same encode done by the C libopus (`opus` dev-dependency) and decoded
/// by libopus: what the reference picks and the quality it reaches.
fn libopus_reference(
    input: &[i16],
    rate: i32,
    app: Application,
    bitrate: i32,
    n_frames: usize,
) -> J {
    let frame = rate as usize * FRAME_MS / 1000;
    let packets = encode_libopus(input, rate, app, bitrate);
    let decoded = decode_libopus(&packets, rate, &[]);
    let delay = find_delay(input, &decoded);
    let aligned = &decoded[delay..delay + input.len()];
    let coded = &packets[..n_frames];
    let bytes: usize = coded.iter().map(|p| p.len()).sum();
    obj(vec![
        ("mode", J::Str(majority(coded, |t| t.mode.to_string()))),
        (
            "bandwidth",
            J::Str(majority(coded, |t| t.bandwidth.to_string())),
        ),
        (
            "avg_kbps",
            num(bytes as f64 * 8.0 / (n_frames * FRAME_MS) as f64, 2),
        ),
        ("delay_samples", int(delay as i64)),
        ("snr_db", num(snr_db(input, aligned), 2)),
        ("seg_snr_db", num(seg_snr_db(input, aligned, frame), 2)),
    ])
}

/// The JSON record of one encode: packets, sizes, decoded audio, quality.
fn stream_json(st: &Stream, input: &[i16], n_frames: usize, application: &str, reference: J) -> J {
    let frame = st.rate as usize * FRAME_MS / 1000;
    let coded = &st.packets[..n_frames];
    let aligned = st.aligned(input.len());
    let bytes: usize = coded.iter().map(|p| p.len()).sum();
    let avg_kbps = bytes as f64 * 8.0 / (n_frames as f64 * FRAME_MS as f64);
    let (silk, hybrid, celt) = mode_counts(coded);
    // Flag encodes whose actual rate is far from the target (none of the
    // current encodes; the page explains such a stream if one appears).
    let rate_note = if (avg_kbps * 1000.0 / st.bitrate as f64 - 1.0).abs() > 0.25 {
        s("actual rate differs from the target by more than 25 %")
    } else {
        J::Null
    };
    obj(vec![
        ("bitrate", int(st.bitrate as i64)),
        ("application", s(application)),
        ("mode", J::Str(majority(coded, |t| t.mode.to_string()))),
        (
            "bandwidth",
            J::Str(majority(coded, |t| t.bandwidth.to_string())),
        ),
        (
            "mode_counts",
            obj(vec![
                ("SILK", int(silk as i64)),
                ("Hybrid", int(hybrid as i64)),
                ("CELT", int(celt as i64)),
            ]),
        ),
        ("n_frames", int(n_frames as i64)),
        ("flush_frames", int((st.packets.len() - n_frames) as i64)),
        (
            "packets",
            J::Arr(st.packets.iter().map(|p| J::Str(base64(p))).collect()),
        ),
        (
            "sizes",
            J::Arr(st.packets.iter().map(|p| int(p.len() as i64)).collect()),
        ),
        (
            "toc",
            J::Arr(st.packets.iter().map(|p| int(p[0] as i64)).collect()),
        ),
        ("delay_samples", int(st.delay as i64)),
        ("decoded", b64_pcm(aligned)),
        ("snr_db", num(snr_db(input, aligned), 2)),
        ("seg_snr_db", num(seg_snr_db(input, aligned, frame), 2)),
        ("rms_ratio", num(rms(aligned) / rms(input), 3)),
        ("ncc", num(ncc_at(input, &st.decoded_raw, st.delay), 4)),
        ("avg_bytes", num(bytes as f64 / n_frames as f64, 2)),
        ("avg_kbps", num(avg_kbps, 2)),
        ("rate_note", rate_note),
        ("libopus_reference", reference),
    ])
}

// ---------------------------------------------------------------------------
// SILK replay: per-frame parameters read from the real bitstream
// ---------------------------------------------------------------------------

struct SilkFrame {
    vad: bool,
    signal_type: i32,
    quant_offset_type: i32,
    seed: i32,
    pitch_lags: Option<[i32; 4]>,
    gains_q16: [i32; 4],
    /// [first half, second half] prediction coefficients, Q12.
    lpc_q12: [[i16; 16]; 2],
    lpc_order: usize,
    nlsf_q15: [i16; 16],
    nlsf_interp_q2: i32,
    ltp_q14: Option<[i16; 20]>,
    ltp_scale_q14: i32,
    per_index: i32,
    pulses: Vec<i16>,
    exc_q14: Vec<i32>,
    /// SILK core output of this frame (before the output delay line).
    core_out: Vec<i16>,
    /// Range-coder position (1/8 bit) after the header, the side info and
    /// the pulses; and the payload size in bits.
    tell_q3: [i32; 3],
    payload_bits: i32,
    /// The replay ended with the same range-decoder state as the real
    /// decoder, and produced the same samples.
    rc_match: bool,
    pcm_match: bool,
}

/// The SILK sampling rate of a SILK-only packet: 8, 12 or 16 kHz for a
/// narrowband, mediumband or wideband packet.
fn silk_rate(bandwidth: &str) -> i32 {
    match bandwidth {
        "NB" => 8000,
        "MB" => 12000,
        _ => 16000,
    }
}

/// Decode a SILK-only 20 ms mono stream with a standalone `SilkDecoder`
/// and, for every packet, replay the frame in a copy of the decoder state to
/// capture its parameters. Returns the frames and the standalone decoder's
/// output (to compare with `OpusDecoder`).
fn trace_silk(packets: &[Vec<u8>], rate: i32) -> (Vec<SilkFrame>, Vec<i16>) {
    let out_len = (rate / 50) as usize;
    let mut dec = SilkDecoder::new();

    let mut frames = Vec::new();
    let mut output = Vec::new();
    for p in packets {
        let toc = parse_toc(p[0]);
        assert!(
            toc.mode == "SILK" && toc.frame_ms == 20.0 && toc.code == 0 && !toc.stereo,
            "trace_silk expects mono 20 ms SILK-only code-0 packets"
        );
        let internal_rate = silk_rate(toc.bandwidth);
        let fs_khz = internal_rate / 1000;
        let frame_len = (internal_rate / 50) as usize;
        let payload = &p[1..];
        // Copy of the state *before* this packet, for the replay.
        let mut shadow = dec.channel_state[0].clone();

        // Real decode, with the controls OpusDecoder passes for this packet.
        // It resamples to the output rate.
        let control = SilkDecControl {
            n_channels_api: 1,
            n_channels_internal: 1,
            api_sample_rate: rate,
            internal_sample_rate: internal_rate,
            payload_size_ms: 20,
        };
        let mut rc = RangeCoder::new_decoder(payload);
        let mut pcm = vec![0i16; out_len];
        let n = dec
            .decode_with_control(&control, &mut rc, &mut pcm, FLAG_DECODE_NORMAL, true)
            .expect("SILK decode");
        assert_eq!(n, out_len);
        output.extend_from_slice(&pcm);
        // The decoder keeps the frame it just decoded at the end of its
        // output history, before comfort noise and the concealment glue
        // touch it (neither changes a good frame after a good frame).
        let st = &dec.channel_state[0];
        let mv_len = (st.ltp_mem_length - st.frame_length) as usize;
        let real_core = st.out_buf[mv_len..mv_len + frame_len].to_vec();

        // Replay: the same steps as SilkDecoder::decode_with_control and
        // silk_decode_frame for a single-frame mono packet, keeping the
        // control struct.
        shadow.n_frames_decoded = 0;
        shadow.n_frames_per_packet = 1;
        shadow.nb_subfr = 4;
        silk_decoder_set_fs(&mut shadow, fs_khz, rate);
        let mut rc2 = RangeCoder::new_decoder(payload);
        let vad = rc2.decode_bit_logp(1);
        let lbrr = rc2.decode_bit_logp(1);
        assert!(!lbrr, "in-band FEC is off, so no LBRR data is expected");
        shadow.vad_flags[0] = vad as i32;
        let t_header = rc2.tell_frac();
        silk_decode_indices(
            &mut shadow,
            &mut rc2,
            0,
            FLAG_DECODE_NORMAL,
            CODE_INDEPENDENTLY,
        );
        let t_side = rc2.tell_frac();
        let mut pulses = [0i16; MAX_FRAME_LENGTH];
        silk_decode_pulses(
            &mut rc2,
            &mut pulses,
            shadow.indices.signal_type as i32,
            shadow.indices.quant_offset_type as i32,
            shadow.frame_length,
        );
        let t_pulses = rc2.tell_frac();
        let mut ctrl = SilkDecoderControl::new();
        silk_decode_parameters(&mut shadow, &mut ctrl, CODE_INDEPENDENTLY);
        let mut core = vec![0i16; frame_len];
        silk_decode_core(&mut shadow, &mut ctrl, &mut core, &pulses);

        let pcm_match = core == real_core;
        let rc_match = rc2.rng == rc.rng && rc2.val == rc.val && rc2.tell_frac() == rc.tell_frac();

        let voiced = shadow.indices.signal_type == 2;
        let mut lags = [0i32; 4];
        lags.copy_from_slice(&ctrl.pitch_l[..4]);
        let mut ltp = [0i16; 20];
        ltp.copy_from_slice(&ctrl.ltp_coef_q14[..20]);
        let mut nlsf = [0i16; 16];
        nlsf.copy_from_slice(&shadow.prev_nlsf_q15[..16]);
        let mut lpc = [[0i16; 16]; 2];
        lpc[0].copy_from_slice(&ctrl.pred_coef_q12[0][..16]);
        lpc[1].copy_from_slice(&ctrl.pred_coef_q12[1][..16]);
        let mut gains = [0i32; 4];
        gains.copy_from_slice(&ctrl.gains_q16[..4]);
        frames.push(SilkFrame {
            vad,
            signal_type: shadow.indices.signal_type as i32,
            quant_offset_type: shadow.indices.quant_offset_type as i32,
            seed: shadow.indices.seed as i32,
            pitch_lags: voiced.then_some(lags),
            gains_q16: gains,
            lpc_q12: lpc,
            lpc_order: shadow.lpc_order as usize,
            nlsf_q15: nlsf,
            nlsf_interp_q2: shadow.indices.nlsf_interp_coef_q2 as i32,
            ltp_q14: voiced.then_some(ltp),
            ltp_scale_q14: ctrl.ltp_scale_q14,
            per_index: shadow.indices.per_index as i32,
            pulses: pulses[..frame_len].to_vec(),
            exc_q14: shadow.exc_q14[..frame_len].to_vec(),
            core_out: core,
            tell_q3: [t_header, t_side, t_pulses],
            payload_bits: payload.len() as i32 * 8,
            rc_match,
            pcm_match,
        });
    }
    (frames, output)
}

/// Offset s such that SILK core sample j of frame k appears at decoded
/// output index k*frame + j + s (the mono delay line plus the resampler's
/// input delay). Found by direct comparison, not assumed.
fn silk_core_offset(frames: &[SilkFrame], decoded: &[i16], frame: usize) -> Option<usize> {
    (0..64).find(|&off| {
        frames.iter().enumerate().take(40).skip(5).all(|(k, f)| {
            let start = k * frame + off;
            let n = frame - off;
            start + n <= decoded.len() && f.core_out[..n] == decoded[start..start + n]
        })
    })
}

fn silk_frame_json(k: usize, f: &SilkFrame, start_sample: i64) -> J {
    let order = f.lpc_order;
    let q = |v: i32, shift: i32| v as f64 / (1i64 << shift) as f64;
    let bits = |t: i32| t as f64 / 8.0;
    obj(vec![
        ("index", int(k as i64)),
        ("start_sample", int(start_sample)),
        ("vad", J::Bool(f.vad)),
        ("signal_type", int(f.signal_type as i64)),
        ("quant_offset_type", int(f.quant_offset_type as i64)),
        ("seed", int(f.seed as i64)),
        ("pitch_lags", opt(f.pitch_lags, |l| ints(&l))),
        ("gains", nums(&f.gains_q16.map(|g| q(g, 16)), 2)),
        (
            "lpc_a",
            nums(
                &f.lpc_q12[1][..order]
                    .iter()
                    .map(|&a| q(a as i32, 12))
                    .collect::<Vec<_>>(),
                4,
            ),
        ),
        (
            "lpc_a_first_half",
            if f.nlsf_interp_q2 < 4 {
                nums(
                    &f.lpc_q12[0][..order]
                        .iter()
                        .map(|&a| q(a as i32, 12))
                        .collect::<Vec<_>>(),
                    4,
                )
            } else {
                J::Null
            },
        ),
        (
            "nlsf",
            nums(
                &f.nlsf_q15[..order]
                    .iter()
                    .map(|&v| q(v as i32, 15))
                    .collect::<Vec<_>>(),
                4,
            ),
        ),
        ("nlsf_interp_q2", int(f.nlsf_interp_q2 as i64)),
        (
            "ltp_coefs",
            opt(f.ltp_q14, |c| {
                J::Arr(
                    c.chunks(5)
                        .map(|sub| {
                            nums(&sub.iter().map(|&v| q(v as i32, 14)).collect::<Vec<_>>(), 4)
                        })
                        .collect(),
                )
            }),
        ),
        (
            "ltp_scale",
            if f.signal_type == 2 {
                num(q(f.ltp_scale_q14, 14), 4)
            } else {
                J::Null
            },
        ),
        (
            "periodicity_index",
            if f.signal_type == 2 {
                int(f.per_index as i64)
            } else {
                J::Null
            },
        ),
        (
            "pulse_sum",
            int(f.pulses.iter().map(|&p| (p as i64).abs()).sum::<i64>()),
        ),
        (
            "bits",
            obj(vec![
                ("header", num(bits(f.tell_q3[0]), 2)),
                ("side_info", num(bits(f.tell_q3[1] - f.tell_q3[0]), 2)),
                ("excitation", num(bits(f.tell_q3[2] - f.tell_q3[1]), 2)),
                ("unused", num(bits(f.payload_bits * 8 - f.tell_q3[2]), 2)),
                ("payload", int(f.payload_bits as i64)),
            ]),
        ),
        ("pulses_b64", J::Blob(base64(&pulse_bytes(&f.pulses)))),
    ])
}

/// The decoded excitation pulses as int8 bytes (they always fit).
fn pulse_bytes(pulses: &[i16]) -> Vec<u8> {
    pulses
        .iter()
        .map(|&p| i8::try_from(p).expect("SILK pulse outside the int8 range") as u8)
        .collect()
}

// ---------------------------------------------------------------------------
// CELT replay: per-frame side information read from the real bitstream
// ---------------------------------------------------------------------------

const NB_EBANDS: usize = 21;

struct CeltFrame {
    silence: bool,
    postfilter: Option<(usize, f32, usize)>,
    transient: bool,
    intra: bool,
    lm: usize,
    end_band: usize,
    tf_res: Vec<i32>,
    tf_select: i32,
    spread: i32,
    boost_q3: Vec<i32>,
    alloc_trim: i32,
    coded_bands: i32,
    pvq_bits_q3: Vec<i32>,
    fine_bits: Vec<i32>,
    fine_priority: Vec<i32>,
    anti_collapse: bool,
    /// Final quantized band energies, log2 of the band amplitude.
    band_log2: Vec<f32>,
    /// Range-coder positions (1/8 bit) after each step, in this order:
    /// header, coarse energy, tf, spread, dynalloc, trim, fine, pvq, end.
    tells_q3: [i32; 9],
    payload_bits: i32,
    /// Decoded MDCT spectrum of the frame (bands denormalized).
    mdct: Vec<f32>,
    rc_match: bool,
}

/// Copy of the CELT decoder state that the bitstream parsing depends on
/// (mono). It is advanced exactly like `CeltDecoder` advances its own.
struct CeltShadow {
    old_e: Vec<f32>,
    old_e2: Vec<f32>,
    old_e3: Vec<f32>,
    rng: u32,
}

/// The energy of a band the decoder has no history for: libopus
/// `oldLogE` after a reset (`OLD_LOG_E_RESET` in src/celt.rs).
const OLD_LOG_E_RESET: f32 = -28.0;

impl CeltShadow {
    fn new() -> Self {
        // As CeltDecoder::new and reset_state(): the last frame's energies
        // at 0, the two before at OLD_LOG_E_RESET.
        Self {
            old_e: vec![0.0; NB_EBANDS],
            old_e2: vec![OLD_LOG_E_RESET; NB_EBANDS],
            old_e3: vec![OLD_LOG_E_RESET; NB_EBANDS],
            rng: 0,
        }
    }
}

/// Port of the private `tf_decode` in src/celt.rs, for start band 0.
fn tf_decode(
    end: usize,
    is_transient: bool,
    tf_res: &mut [i32],
    lm: usize,
    rc: &mut RangeCoder,
) -> i32 {
    let mut curr = 0;
    let mut tf_changed = 0;
    let mut logp = if is_transient { 2 } else { 4 };
    let budget = rc.storage as i32 * 8;
    let mut tell = rc.tell();
    let tf_select_rsv = if lm > 0 && tell + logp < budget { 1 } else { 0 };
    let budget = budget - tf_select_rsv;
    for r in tf_res[..end].iter_mut() {
        if tell + logp <= budget {
            curr ^= rc.decode_bit_logp(logp as u32) as i32;
            tell = rc.tell();
            tf_changed |= curr;
        }
        *r = curr;
        logp = if is_transient { 4 } else { 5 };
    }
    let t = is_transient as usize;
    let mut tf_select = 0;
    if tf_select_rsv > 0
        && TF_SELECT_TABLE[lm][4 * t + tf_changed as usize]
            != TF_SELECT_TABLE[lm][4 * t + 2 + tf_changed as usize]
    {
        tf_select = rc.decode_bit_logp(1) as i32;
    }
    for r in tf_res[..end].iter_mut() {
        *r = TF_SELECT_TABLE[lm][4 * t + 2 * tf_select as usize + *r as usize] as i32;
    }
    tf_select
}

/// Replay one mono CELT-only frame: the same parsing steps, in the same
/// order, as `CeltDecoder::decode` (src/celt.rs), but keeping
/// every decoded decision. Stops before the MDCT synthesis.
fn replay_celt_frame(
    st: &mut CeltShadow,
    mode: &CeltMode,
    payload: &[u8],
    frame_size: usize,
    end: usize,
) -> (CeltFrame, RangeCoder) {
    let lm = (0..=mode.max_lm)
        .find(|&l| mode.short_mdct_size << l == frame_size)
        .expect("frame size");
    let total_bits = payload.len() as i32 * 8;
    let mut rc = RangeCoder::new_decoder(payload);
    let mut tells = [0i32; 9];

    let tell = rc.tell();
    let silence = if tell >= total_bits {
        true
    } else if tell == 1 {
        rc.decode_bit_logp(15)
    } else {
        false
    };
    if silence {
        rc.nbits_total += total_bits - rc.tell();
    }

    let mut postfilter = None;
    if !silence && rc.tell() + 16 <= total_bits && rc.decode_bit_logp(1) {
        let octave = rc.dec_uint(6);
        let period = ((16 << octave) + rc.dec_bits(4 + octave)) as usize - 1;
        let qg = rc.dec_bits(3);
        let tapset = if rc.tell() + 2 <= total_bits {
            rc.decode_icdf(&TAPSET_ICDF, 2) as usize
        } else {
            0
        };
        postfilter = Some((period, 0.09375 * (qg as f32 + 1.0), tapset));
    }
    let transient = lm > 0 && rc.tell() + 3 <= total_bits && rc.decode_bit_logp(3);
    let intra = rc.tell() + 3 <= total_bits && rc.decode_bit_logp(3);
    tells[0] = rc.tell_frac();

    unquant_coarse_energy(mode, 0, end, &mut st.old_e, intra, &mut rc, 1, lm);
    tells[1] = rc.tell_frac();

    let mut tf_res = vec![0i32; NB_EBANDS];
    let tf_select = tf_decode(end, transient, &mut tf_res, lm, &mut rc);
    tells[2] = rc.tell_frac();

    let spread = if rc.tell() + 4 <= total_bits {
        rc.decode_icdf(&SPREAD_ICDF, 5)
    } else {
        2 // SPREAD_NORMAL
    };
    tells[3] = rc.tell_frac();

    let mut cap = vec![0i32; NB_EBANDS];
    for i in 0..NB_EBANDS {
        let n = ((mode.e_bands[i + 1] - mode.e_bands[i]) as i32) << lm;
        cap[i] = ((mode.cache.caps[NB_EBANDS * (2 * lm) + i] as i32 + 64) * n) >> 2;
    }
    let mut offsets = vec![0i32; NB_EBANDS];
    let mut dynalloc_logp = 6i32;
    let mut total_bits_q3 = total_bits << BITRES;
    let mut tell_frac = rc.tell_frac();
    for i in 0..end {
        let width = ((mode.e_bands[i + 1] - mode.e_bands[i]) as i32) << lm;
        let quanta = (width << BITRES).min((6 << BITRES).max(width));
        let mut loop_logp = dynalloc_logp;
        let mut boost = 0;
        while tell_frac + (loop_logp << BITRES) < total_bits_q3 && boost < cap[i] {
            let flag = rc.decode_bit_logp(loop_logp as u32);
            tell_frac = rc.tell_frac();
            if !flag {
                break;
            }
            boost += quanta;
            total_bits_q3 -= quanta;
            loop_logp = 1;
        }
        offsets[i] = boost;
        if boost > 0 {
            dynalloc_logp = (dynalloc_logp - 1).max(2);
        }
    }
    tells[4] = rc.tell_frac();

    let alloc_trim = if rc.tell_frac() + (6 << BITRES) <= total_bits_q3 {
        rc.decode_icdf(&TRIM_ICDF, 7)
    } else {
        5
    };
    tells[5] = rc.tell_frac();

    let anti_collapse_rsv = if transient && lm >= 2 {
        let remaining = (total_bits << BITRES) - rc.tell_frac() - 1;
        if remaining >= ((lm as i32 + 2) << BITRES) {
            1 << BITRES
        } else {
            0
        }
    } else {
        0
    };
    let mut intensity = 0;
    let mut dual_stereo = 0;
    let mut balance = 0;
    let mut pulses = vec![0i32; NB_EBANDS];
    let mut ebits = vec![0i32; NB_EBANDS];
    let mut fine_priority = vec![0i32; NB_EBANDS];
    let alloc_bits = (total_bits << BITRES) - rc.tell_frac() - 1 - anti_collapse_rsv;
    let coded_bands = clt_compute_allocation(
        mode,
        0,
        end,
        &offsets,
        &cap,
        alloc_trim,
        &mut intensity,
        &mut dual_stereo,
        alloc_bits,
        &mut balance,
        &mut pulses,
        &mut ebits,
        &mut fine_priority,
        1,
        lm as i32,
        &mut rc,
        false,
        0,
        end as i32 - 1,
    );
    unquant_fine_energy(mode, 0, end, &mut st.old_e, &ebits, &mut rc, 1);
    tells[6] = rc.tell_frac();

    // PVQ band shapes. The shapes are needed for the decoded spectrum; the
    // bits they consume must be read anyway to reach the final fine bits.
    let pad = 352; // MAX_PVQ_N: quant_all_bands may read past the frame
    let mut x = vec![0f32; frame_size + pad];
    let mut collapse_masks = vec![0u32; NB_EBANDS];
    let mut band_amp = vec![0f32; NB_EBANDS];
    log2amp(mode, NB_EBANDS, &mut band_amp, &st.old_e, 1);
    let mut dual = false;
    quant_all_bands(
        false,
        mode,
        0,
        end,
        &mut x[..frame_size],
        None,
        &mut collapse_masks,
        &band_amp,
        &pulses,
        transient,
        spread,
        &mut dual,
        intensity as usize,
        &tf_res,
        (total_bits << BITRES) - anti_collapse_rsv,
        &mut balance,
        &mut rc,
        lm as i32,
        coded_bands,
        true,
        false,
        &mut st.rng,
    );
    tells[7] = rc.tell_frac();

    let anti_collapse_on = anti_collapse_rsv > 0 && rc.dec_bits(1) != 0;
    unquant_energy_finalise(
        mode,
        0,
        end,
        &mut st.old_e,
        &ebits,
        &fine_priority,
        total_bits - rc.tell(),
        &mut rc,
        1,
    );
    tells[8] = rc.tell_frac();
    if anti_collapse_on {
        st.rng = anti_collapse(
            mode,
            &mut x,
            &collapse_masks,
            lm as i32,
            1,
            frame_size,
            0,
            NB_EBANDS,
            &st.old_e,
            &st.old_e2,
            &st.old_e3,
            &pulses,
            st.rng,
        );
    }
    if silence {
        st.old_e.fill(-28.0);
    }

    let mut mdct = vec![0f32; frame_size];
    if !silence {
        log2amp(mode, NB_EBANDS, &mut band_amp, &st.old_e, 1);
        denormalise_bands(mode, &x, &mut mdct, &band_amp, 0, end, 1, 1 << lm);
    }
    let band_log2 = (0..NB_EBANDS)
        .map(|i| st.old_e[i] + mode.e_means[i])
        .collect();

    // End-of-frame state updates (as at the end of CeltDecoder::decode).
    if !transient {
        st.old_e3.copy_from_slice(&st.old_e2);
        st.old_e2.copy_from_slice(&st.old_e);
    } else {
        for i in 0..NB_EBANDS {
            st.old_e2[i] = st.old_e2[i].min(st.old_e[i]);
        }
    }
    // Bands this frame did not code carry no energy history.
    for i in end..NB_EBANDS {
        st.old_e[i] = 0.0;
        st.old_e2[i] = OLD_LOG_E_RESET;
        st.old_e3[i] = OLD_LOG_E_RESET;
    }
    st.rng = rc.rng;

    let frame = CeltFrame {
        silence,
        postfilter,
        transient,
        intra,
        lm,
        end_band: end,
        tf_res,
        tf_select,
        spread,
        boost_q3: offsets,
        alloc_trim,
        coded_bands,
        pvq_bits_q3: pulses,
        fine_bits: ebits,
        fine_priority,
        anti_collapse: anti_collapse_on,
        band_log2,
        tells_q3: tells,
        payload_bits: total_bits,
        mdct,
        rc_match: false,
    };
    (frame, rc)
}

/// Decode a mono 48 kHz stream with a standalone `CeltDecoder` and replay
/// every CELT-only frame. Non-CELT packets get `None`. Returns the frames and
/// the standalone decoder's output.
fn trace_celt(packets: &[Vec<u8>]) -> (Vec<Option<CeltFrame>>, Vec<i16>) {
    let mode = default_mode();
    let mut dec = CeltDecoder::new(mode, 1, MUSIC_RATE);
    let mut shadow = CeltShadow::new();
    let mut prev_mode = None;
    let mut frames = Vec::new();
    let mut output = Vec::new();
    for p in packets {
        let toc = parse_toc(p[0]);
        if toc.mode != "CELT" || toc.code != 0 || toc.stereo {
            frames.push(None);
            output.extend(std::iter::repeat_n(0i16, (MUSIC_RATE / 50) as usize));
            prev_mode = Some(toc.mode);
            continue;
        }
        // OpusDecoder resets CELT when the mode changes. A mode switch also
        // brings a redundancy frame, which this replay does not decode: the
        // traced stream is CELT-only, and a switch would fail the checks.
        if prev_mode.is_some_and(|m| m != "CELT") {
            dec.reset_state();
            shadow = CeltShadow::new();
        }
        prev_mode = Some("CELT");
        let frame_size = (MUSIC_RATE as f64 * toc.frame_ms / 1000.0) as usize;
        let end = celt_end_band(toc.bandwidth);
        let payload = &p[1..];

        let mut rc = RangeCoder::new_decoder(payload);
        let mut pcm = vec![0f32; frame_size];
        dec.set_end_band(end).expect("end band");
        let n = dec.decode(
            &mut rc,
            payload.len() as i32 * 8,
            frame_size,
            &mut pcm,
            PcmLayout::Planar,
        );
        assert_eq!(n, frame_size);
        output.extend(pcm.iter().map(|&v| f32_to_i16(v.clamp(-1.0, 1.0))));

        let (mut f, rc2) = replay_celt_frame(&mut shadow, mode, payload, frame_size, end);
        f.rc_match = rc2.rng == rc.rng && rc2.val == rc.val && rc2.tell_frac() == rc.tell_frac();
        frames.push(Some(f));
    }
    (frames, output)
}

fn celt_frame_json(k: usize, f: &CeltFrame, start_sample: i64, bytes: usize) -> J {
    let b = |t: i32| t as f64 / 8.0;
    let t = &f.tells_q3;
    let e = f.end_band;
    obj(vec![
        ("index", int(k as i64)),
        ("start_sample", int(start_sample)),
        ("bytes", int(bytes as i64)),
        ("silence", J::Bool(f.silence)),
        (
            "postfilter",
            match f.postfilter {
                Some((period, gain, tapset)) => obj(vec![
                    ("period", int(period as i64)),
                    ("gain", num(gain as f64, 5)),
                    ("tapset", int(tapset as i64)),
                ]),
                None => J::Null,
            },
        ),
        ("transient", J::Bool(f.transient)),
        ("intra", J::Bool(f.intra)),
        ("lm", int(f.lm as i64)),
        ("end_band", int(e as i64)),
        (
            "band_log2_amp",
            nums(
                &f.band_log2.iter().map(|&v| v as f64).collect::<Vec<_>>(),
                3,
            ),
        ),
        ("tf_res", ints(&f.tf_res[..e])),
        ("tf_select", int(f.tf_select as i64)),
        ("spread", int(f.spread as i64)),
        (
            "boost_bits",
            nums(
                &f.boost_q3[..e].iter().map(|&v| b(v)).collect::<Vec<_>>(),
                3,
            ),
        ),
        ("alloc_trim", int(f.alloc_trim as i64)),
        ("coded_bands", int(f.coded_bands as i64)),
        (
            "pvq_bits",
            nums(
                &f.pvq_bits_q3[..e].iter().map(|&v| b(v)).collect::<Vec<_>>(),
                3,
            ),
        ),
        ("fine_bits", ints(&f.fine_bits[..e])),
        ("fine_priority", ints(&f.fine_priority[..e])),
        ("anti_collapse", J::Bool(f.anti_collapse)),
        (
            "bits",
            obj(vec![
                ("header", num(b(t[0]), 2)),
                ("coarse_energy", num(b(t[1] - t[0]), 2)),
                ("tf", num(b(t[2] - t[1]), 2)),
                ("spread", num(b(t[3] - t[2]), 2)),
                ("dynalloc", num(b(t[4] - t[3]), 2)),
                ("trim", num(b(t[5] - t[4]), 2)),
                ("fine_energy", num(b(t[6] - t[5]), 2)),
                ("pvq", num(b(t[7] - t[6]), 2)),
                ("final_fine", num(b(t[8] - t[7]), 2)),
                ("unused", num(b(f.payload_bits * 8 - t[8]), 2)),
                ("payload", int(f.payload_bits as i64)),
            ]),
        ),
    ])
}

// ---------------------------------------------------------------------------
// Signals: 3x upsampler and the synthetic music
// ---------------------------------------------------------------------------

fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0;
    let mut term = 1.0;
    for k in 1..50 {
        term *= (x / (2.0 * k as f64)).powi(2);
        sum += term;
    }
    sum
}

/// 16 kHz -> 48 kHz: zero-stuff by 3 and low-pass with a Kaiser-windowed
/// sinc (145 taps, cutoff 7.6 kHz, beta 8). Zero phase: no added delay.
fn upsample3(x: &[i16]) -> Vec<i16> {
    const HALF: i64 = 72;
    let fc = 7600.0 / 48000.0;
    let beta = 8.0;
    let h: Vec<f64> = (-HALF..=HALF)
        .map(|n| {
            let t = n as f64;
            let sinc = if n == 0 {
                1.0
            } else {
                (2.0 * std::f64::consts::PI * fc * t).sin() / (2.0 * std::f64::consts::PI * fc * t)
            };
            let w = bessel_i0(beta * (1.0 - (t / HALF as f64).powi(2)).sqrt()) / bessel_i0(beta);
            3.0 * 2.0 * fc * sinc * w
        })
        .collect();
    let n_out = x.len() * 3;
    let mut y = vec![0i16; n_out];
    for m in 0..n_out as i64 {
        let mut acc = 0.0;
        let k_lo = (m - HALF).max(0).div_euclid(3) + if (m - HALF).max(0) % 3 != 0 { 1 } else { 0 };
        let k_hi = ((m + HALF) / 3).min(x.len() as i64 - 1);
        for k in k_lo..=k_hi {
            let d = m - 3 * k;
            if d.abs() <= HALF {
                acc += x[k as usize] as f64 * h[(d + HALF) as usize];
            }
        }
        y[m as usize] = acc.round().clamp(-32768.0, 32767.0) as i16;
    }
    y
}

/// Deterministic noise source (xorshift64*), uniform in [-1, 1).
struct Noise(u64);
impl Noise {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let v = self.0.wrapping_mul(0x2545_F491_4F6C_DD1D);
        (v >> 11) as f64 / (1u64 << 52) as f64 - 1.0
    }
}

/// RBJ biquad (Direct Form I).
struct Biquad {
    b: [f64; 3],
    a: [f64; 2],
    x: [f64; 2],
    y: [f64; 2],
}
impl Biquad {
    fn new(kind: &str, f0: f64, q: f64, fs: f64) -> Self {
        let w = 2.0 * std::f64::consts::PI * f0 / fs;
        let (sw, cw) = w.sin_cos();
        let alpha = sw / (2.0 * q);
        let (b, a0, a1, a2) = match kind {
            "lp" => (
                [(1.0 - cw) / 2.0, 1.0 - cw, (1.0 - cw) / 2.0],
                1.0 + alpha,
                -2.0 * cw,
                1.0 - alpha,
            ),
            "hp" => (
                [(1.0 + cw) / 2.0, -(1.0 + cw), (1.0 + cw) / 2.0],
                1.0 + alpha,
                -2.0 * cw,
                1.0 - alpha,
            ),
            _ => ([alpha, 0.0, -alpha], 1.0 + alpha, -2.0 * cw, 1.0 - alpha), // band-pass
        };
        Self {
            b: [b[0] / a0, b[1] / a0, b[2] / a0],
            a: [a1 / a0, a2 / a0],
            x: [0.0; 2],
            y: [0.0; 2],
        }
    }
    fn run(&mut self, v: f64) -> f64 {
        let out = self.b[0] * v + self.b[1] * self.x[0] + self.b[2] * self.x[1]
            - self.a[0] * self.y[0]
            - self.a[1] * self.y[1];
        self.x = [v, self.x[0]];
        self.y = [out, self.y[0]];
        out
    }
}

/// A music event, for the page's annotations.
struct Event {
    t_s: f64,
    kind: &'static str,
    detail: String,
}

/// 2.5 s of synthetic music at 48 kHz, peak -3 dBFS:
/// - a soft Cmaj9 pad (C3 G3 E4 B4 D5) with 5 Hz vibrato,
/// - a plucked pentatonic melody (Karplus-Strong),
/// - three percussive hits that trigger CELT short blocks,
/// - quiet 9-15 kHz "air" noise so the fullband part is never empty.
fn synth_music(fs: usize, dur_s: f64) -> (Vec<i16>, Vec<Event>) {
    let n = (fs as f64 * dur_s) as usize;
    let fsf = fs as f64;
    let tau = 2.0 * std::f64::consts::PI;
    let mut mix = vec![0f64; n];
    let mut events = Vec::new();
    let mut noise = Noise(0x0F0F_5EED_1234_ABCD);

    // Pad: two slightly detuned voices per note, harmonics up to 9 kHz.
    let pad_notes = [
        (130.81, 1.0),
        (196.00, 0.8),
        (329.63, 0.7),
        (493.88, 0.5),
        (587.33, 0.45),
    ];
    for (ni, &(f0, gain)) in pad_notes.iter().enumerate() {
        for (vi, detune) in [0.9985f64, 1.0015].iter().enumerate() {
            let mut phase = vec![0f64; 64];
            let vib_phase = ni as f64 * 1.3 + vi as f64 * 2.1;
            for i in 0..n {
                let t = i as f64 / fsf;
                let f = f0 * detune * (1.0 + 0.003 * (tau * 5.2 * t + vib_phase).sin());
                let mut v = 0.0;
                for k in 1..64 {
                    let fk = f * k as f64;
                    if fk > 9000.0 {
                        break;
                    }
                    phase[k] = (phase[k] + tau * fk / fsf) % tau;
                    v += phase[k].sin() * (k as f64).powf(-1.3) * (-fk / 5000.0).exp();
                }
                mix[i] += 0.055 * gain * v;
            }
        }
    }
    events.push(Event {
        t_s: 0.0,
        kind: "pad",
        detail: "Cmaj9 pad: C3 G3 E4 B4 D5, 5.2 Hz vibrato".into(),
    });

    // Plucked melody: Karplus-Strong strings.
    let melody = [
        (0.30, 659.26, "E5"),
        (0.55, 783.99, "G5"),
        (0.80, 880.00, "A5"),
        (1.05, 1046.50, "C6"),
        (1.55, 880.00, "A5"),
        (1.80, 783.99, "G5"),
        (2.05, 659.26, "E5"),
    ];
    for &(t0, f0, name) in &melody {
        let period = (fsf / f0).round() as usize;
        let mut line: Vec<f64> = (0..period).map(|_| noise.next()).collect();
        // Soften the excitation a little (one averaging pass).
        let first = line[0];
        for j in 0..period - 1 {
            line[j] = 0.5 * (line[j] + line[j + 1]);
        }
        line[period - 1] = 0.5 * (line[period - 1] + first);
        // Remove the excitation's mean, so the string carries no DC offset.
        let mean = line.iter().sum::<f64>() / period as f64;
        for v in line.iter_mut() {
            *v -= mean;
        }
        let start = (t0 * fsf) as usize;
        let len = ((0.9 * fsf) as usize).min(n - start);
        let mut idx = 0;
        for i in 0..len {
            let next = (idx + 1) % period;
            let v = line[idx];
            line[idx] = 0.996 * 0.5 * (line[idx] + line[next]);
            idx = next;
            mix[start + i] += 0.22 * v;
        }
        events.push(Event {
            t_s: t0,
            kind: "pluck",
            detail: format!("{} {:.2} Hz", name, f0),
        });
    }

    // Percussion: castanet clicks (band-passed noise, 4 ms decay) and an
    // open hi-hat (high-passed noise above 7 kHz, 45 ms decay).
    let hit = |mix: &mut Vec<f64>, t0: f64, kind: &'static str, noise: &mut Noise| {
        let start = (t0 * fsf) as usize;
        match kind {
            "castanet" => {
                let mut bp = Biquad::new("bp", 3200.0, 1.1, fsf);
                let mut hp = Biquad::new("hp", 1500.0, 0.7, fsf);
                for i in 0..(0.04 * fsf) as usize {
                    let t = i as f64 / fsf;
                    let v = bp.run(noise.next()) * 3.0 + if i < 2 { 0.6 } else { 0.0 };
                    mix[start + i] += 0.55 * hp.run(v) * (-t / 0.004).exp();
                }
            }
            _ => {
                let mut hp1 = Biquad::new("hp", 7000.0, 0.7, fsf);
                let mut hp2 = Biquad::new("hp", 7000.0, 0.7, fsf);
                for i in 0..(0.3 * fsf) as usize {
                    let t = i as f64 / fsf;
                    let v = hp2.run(hp1.run(noise.next()));
                    mix[start + i] += 0.45 * v * (-t / 0.045).exp();
                }
            }
        }
    };
    for &(t0, kind, detail) in &[
        (0.68, "castanet", "band-passed noise click, 4 ms decay"),
        (1.30, "hihat", "open hi-hat, noise above 7 kHz, 45 ms decay"),
        (1.93, "castanet", "double click (1.930 s and 1.955 s)"),
    ] {
        hit(&mut mix, t0, kind, &mut noise);
        if t0 == 1.93 {
            hit(&mut mix, 1.955, kind, &mut noise);
        }
        events.push(Event {
            t_s: t0,
            kind,
            detail: detail.into(),
        });
    }

    // Air: quiet 9-15 kHz noise.
    let mut hp = Biquad::new("hp", 9000.0, 0.7, fsf);
    let mut lp = Biquad::new("lp", 15000.0, 0.7, fsf);
    for v in mix.iter_mut() {
        *v += 0.012 * lp.run(hp.run(noise.next()));
    }
    events.push(Event {
        t_s: 0.0,
        kind: "air",
        detail: "quiet band-limited noise, 9-15 kHz".into(),
    });

    // Fade in/out (pad attack 0.35 s, release 0.4 s) and peak -3 dBFS.
    for i in 0..n {
        let t = i as f64 / fsf;
        let fade_in = (t / 0.35).min(1.0);
        let fade_out = ((dur_s - t) / 0.4).min(1.0);
        let g = (0.5 - 0.5 * (std::f64::consts::PI * fade_in).cos())
            * (0.5 - 0.5 * (std::f64::consts::PI * fade_out).cos());
        mix[i] *= g;
    }
    let peak = mix.iter().fold(0f64, |m, v| m.max(v.abs()));
    let scale = 10f64.powf(-3.0 / 20.0) * 32767.0 / peak;
    let pcm = mix
        .iter()
        .map(|v| (v * scale).round().clamp(-32768.0, 32767.0) as i16)
        .collect();
    (pcm, events)
}

// ---------------------------------------------------------------------------
// libopus reference (the `opus` dev-dependency) for the mode map
// ---------------------------------------------------------------------------

fn encode_libopus(pcm: &[i16], rate: i32, app: Application, bitrate: i32) -> Vec<Vec<u8>> {
    let frame = rate as usize * FRAME_MS / 1000;
    let app = match app {
        Application::Voip => opus::Application::Voip,
        Application::Audio => opus::Application::Audio,
        Application::RestrictedLowDelay => opus::Application::LowDelay,
        other => panic!("no libopus application for {other:?}"),
    };
    let mut enc = opus::Encoder::new(rate as u32, opus::Channels::Mono, app).expect("libopus enc");
    enc.set_bitrate(opus::Bitrate::Bits(bitrate)).unwrap();
    enc.set_vbr(true).unwrap();
    enc.set_complexity(i32::from(COMPLEXITY)).unwrap();
    pad_for_flush(pcm, frame)
        .chunks(frame)
        .map(|c| {
            let mut buf = vec![0u8; 1500];
            let n = enc.encode(c, &mut buf).expect("libopus encode");
            buf.truncate(n);
            buf
        })
        .collect()
}

fn decode_libopus(packets: &[Vec<u8>], rate: i32, lost: &[usize]) -> Vec<i16> {
    let frame = rate as usize * FRAME_MS / 1000;
    let mut dec = opus::Decoder::new(rate as u32, opus::Channels::Mono).expect("libopus dec");
    let mut out = Vec::new();
    let mut buf = vec![0i16; frame];
    for (i, p) in packets.iter().enumerate() {
        let input: &[u8] = if lost.contains(&i) { &[] } else { p };
        let n = dec.decode(input, &mut buf, false).expect("libopus decode");
        out.extend_from_slice(&buf[..n]);
    }
    out
}

fn libopus_lookahead(rate: i32, app: opus::Application) -> i32 {
    let mut enc = opus::Encoder::new(rate as u32, opus::Channels::Mono, app).unwrap();
    enc.get_lookahead().unwrap()
}

fn mode_map_entry(packets: &[Vec<u8>], n_frames: usize, kbps: i32) -> J {
    let coded = &packets[..n_frames];
    let (silk, hybrid, celt) = mode_counts(coded);
    let bytes: usize = coded.iter().map(|p| p.len()).sum();
    obj(vec![
        ("bitrate_kbps", int(kbps as i64)),
        ("mode", J::Str(majority(coded, |t| t.mode.to_string()))),
        (
            "bandwidth",
            J::Str(majority(coded, |t| t.bandwidth.to_string())),
        ),
        (
            "frame_ms",
            num(
                majority(coded, |t| format!("{}", t.frame_ms))
                    .parse()
                    .unwrap(),
                1,
            ),
        ),
        (
            "mode_counts",
            obj(vec![
                ("SILK", int(silk as i64)),
                ("Hybrid", int(hybrid as i64)),
                ("CELT", int(celt as i64)),
            ]),
        ),
        ("avg_bytes", num(bytes as f64 / n_frames as f64, 2)),
        (
            "avg_kbps",
            num(bytes as f64 * 8.0 / (n_frames * FRAME_MS) as f64, 2),
        ),
    ])
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn git_commit(root: &Path) -> String {
    let run = |args: &[&str]| {
        std::process::Command::new("git")
            .args(args)
            .current_dir(root)
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
    };
    match run(&["rev-parse", "--short", "HEAD"]) {
        None => "unknown".into(),
        Some(hash) => {
            // Only the generator's inputs count, never the output file.
            let inputs = [
                "status",
                "--porcelain",
                "--",
                "src",
                "fixtures",
                "Cargo.toml",
                "examples/opus_demo_data.rs",
            ];
            let dirty = run(&inputs).is_some_and(|s| !s.is_empty());
            if dirty { hash + "-dirty" } else { hash }
        }
    }
}

fn main() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let out_path = root.join("docs/opus-explained/data/opus-demo-data.js");
    let mut validation: Vec<(&str, J)> = Vec::new();
    let mut all_ok = true;

    // ---------------- A. Speech ----------------
    println!("== Speech ==");
    let (wav_rate, wav) = read_wav_mono16(&root.join("fixtures/answer_16k.wav"));
    assert_eq!(wav_rate as i32, SPEECH_RATE);
    let start = (SPEECH_START_S * SPEECH_RATE as f64).round() as usize;
    let len = (SPEECH_DUR_S * SPEECH_RATE as f64).round() as usize;
    let speech = wav[start..start + len].to_vec();
    let sframe = SPEECH_RATE as usize / 50;
    let n_speech_frames = len / sframe;

    let mut speech_streams = Vec::new();
    for &br in &SPEECH_BITRATES {
        let st = make_stream(&speech, SPEECH_RATE, Application::Voip, br);
        let aligned = st.aligned(len);
        let (si, hy, ce) = mode_counts(&st.packets[..n_speech_frames]);
        println!(
            "  {:>6} b/s: delay {} smp, SNR {:6.2} dB, segSNR {:6.2} dB, rms ratio {:.3}, {:.2} kb/s, SILK/Hybrid/CELT {}/{}/{}, first packet {} {}",
            br,
            st.delay,
            snr_db(&speech, aligned),
            seg_snr_db(&speech, aligned, sframe),
            rms(aligned) / rms(&speech),
            st.packets[..n_speech_frames]
                .iter()
                .map(|p| p.len())
                .sum::<usize>() as f64
                * 8.0
                / (n_speech_frames * FRAME_MS) as f64,
            si,
            hy,
            ce,
            parse_toc(st.packets[0][0]).mode,
            parse_toc(st.packets[0][0]).bandwidth,
        );
        speech_streams.push(st);
    }
    let trace = speech_streams
        .iter()
        .find(|s| s.bitrate == SPEECH_TRACE_BITRATE)
        .unwrap();

    // SILK parameters of the traced stream, checked against OpusDecoder.
    let (silk_frames, silk_replay_out) = trace_silk(&trace.packets, SPEECH_RATE);
    let replay_matches = silk_replay_out == trace.decoded_raw;
    let rc_ok = silk_frames.iter().all(|f| f.rc_match);
    let pcm_ok = silk_frames.iter().all(|f| f.pcm_match);
    let core_offset = silk_core_offset(&silk_frames, &trace.decoded_raw, sframe);
    println!(
        "  SILK replay: output == OpusDecoder: {}, range coder states equal: {}, core samples equal: {}, core->output offset: {:?}",
        replay_matches, rc_ok, pcm_ok, core_offset
    );
    all_ok &= replay_matches && rc_ok && pcm_ok && core_offset.is_some();
    let core_offset = core_offset.unwrap_or(0);
    let class: String = silk_frames[..n_speech_frames]
        .iter()
        .map(|f| ['_', 'u', 'V'][f.signal_type as usize])
        .collect();
    println!("  SILK frame types (_ inactive, u unvoiced, V voiced):");
    for (i, chunk) in class.as_bytes().chunks(50).enumerate() {
        println!("    {:3} {}", i * 50, std::str::from_utf8(chunk).unwrap());
    }
    validation.push((
        "silk_replay",
        obj(vec![
            (
                "standalone_output_equals_opus_decoder",
                J::Bool(replay_matches),
            ),
            ("range_decoder_state_equal_all_frames", J::Bool(rc_ok)),
            ("core_samples_equal_all_frames", J::Bool(pcm_ok)),
        ]),
    ));

    let frame_start = |k: usize| (k * sframe + core_offset) as i64 - trace.delay as i64;
    let silk_frames_json = J::Arr(
        silk_frames
            .iter()
            .enumerate()
            .map(|(k, f)| silk_frame_json(k, f, frame_start(k)))
            .collect(),
    );

    // Three representative frames: the loudest voiced, the loudest unvoiced
    // and the quietest frame (an inactive one if the VAD marked any).
    let energy = |f: &SilkFrame| f.core_out.iter().map(|&v| (v as f64).powi(2)).sum::<f64>();
    let pick = |want: i32, loudest: bool| -> Option<usize> {
        let mut best: Option<(usize, f64)> = None;
        for (k, f) in silk_frames[..n_speech_frames].iter().enumerate().skip(2) {
            if f.signal_type != want {
                continue;
            }
            let e = if loudest { energy(f) } else { -energy(f) };
            if best.is_none_or(|(_, b)| e > b) {
                best = Some((k, e));
            }
        }
        best.map(|b| b.0)
    };
    let quietest = (2..n_speech_frames)
        .min_by(|&a, &b| energy(&silk_frames[a]).total_cmp(&energy(&silk_frames[b])))
        .unwrap();
    let detail_picks = [
        ("voiced", pick(2, true)),
        ("unvoiced", pick(1, true)),
        ("inactive", pick(0, false).or(Some(quietest))),
    ];
    let silk_detail = J::Arr(
        detail_picks
            .iter()
            .filter_map(|&(role, k)| k.map(|k| (role, k)))
            .map(|(role, k)| {
                let f = &silk_frames[k];
                let sub = f.core_out.len() / 4;
                let exc: Vec<f64> = f.exc_q14.iter().map(|&v| v as f64 / 16384.0).collect();
                let exc_scaled: Vec<i64> = f
                    .exc_q14
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| {
                        (v as f64 / 16384.0 * f.gains_q16[i / sub] as f64 / 65536.0).round() as i64
                    })
                    .collect();
                obj(vec![
                    ("role", s(role)),
                    ("index", int(k as i64)),
                    ("signal_type", int(f.signal_type as i64)),
                    ("start_sample", int(frame_start(k))),
                    ("pulses", ints(&f.pulses)),
                    ("excitation", nums(&exc, 3)),
                    (
                        "excitation_scaled",
                        J::Arr(exc_scaled.into_iter().map(J::Int).collect()),
                    ),
                    ("output", ints(&f.core_out)),
                ])
            })
            .collect(),
    );
    println!(
        "  detail frames: {:?}",
        detail_picks.iter().map(|(r, k)| (r, k)).collect::<Vec<_>>()
    );

    // Packet loss on the traced stream.
    let plc_raw = decode_opus1(&trace.packets, SPEECH_RATE, &PLC_LOST_FRAMES);
    let plc_aligned = &plc_raw[trace.delay..trace.delay + len];
    // The same loss pattern through the libopus decoder. Compared over the
    // concealed frames, and separately over the first good frame after each
    // loss (where the decoders start to differ).
    let lib_plc = decode_libopus(&trace.packets, SPEECH_RATE, &PLC_LOST_FRAMES);
    let compare = |frames: &[usize]| {
        let (mut max_diff, mut sig, mut err) = (0i32, 0f64, 0f64);
        for &k in frames {
            for i in k * sframe..(k + 1) * sframe {
                let d = lib_plc[i] as i32 - plc_raw[i] as i32;
                max_diff = max_diff.max(d.abs());
                sig += (lib_plc[i] as f64).powi(2);
                err += (d as f64).powi(2);
            }
        }
        (max_diff, 10.0 * (sig / err.max(1e-9)).log10())
    };
    let after_loss: Vec<usize> = PLC_LOST_FRAMES
        .iter()
        .map(|&k| k + 1)
        .filter(|k| !PLC_LOST_FRAMES.contains(k))
        .collect();
    let (plc_max_diff, plc_vs_lib_snr) = compare(&PLC_LOST_FRAMES);
    let (rec_max_diff, rec_vs_lib_snr) = compare(&after_loss);
    println!(
        "  PLC: lost {:?} (types {:?}); SNR vs input {:.2} dB; vs libopus: concealed frames max diff {}, \
         SNR {:.1} dB; first good frames {:?} max diff {}, SNR {:.1} dB",
        PLC_LOST_FRAMES,
        PLC_LOST_FRAMES
            .iter()
            .map(|&k| silk_frames[k].signal_type)
            .collect::<Vec<_>>(),
        snr_db(&speech, plc_aligned),
        plc_max_diff,
        plc_vs_lib_snr,
        after_loss,
        rec_max_diff,
        rec_vs_lib_snr
    );

    // libopus decodes opus1 packets: does it agree with opus1's decoder?
    let mut cross = Vec::new();
    for st in &speech_streams {
        let lib = decode_libopus(&st.packets, SPEECH_RATE, &[]);
        let max_diff = lib
            .iter()
            .zip(&st.decoded_raw)
            .map(|(a, b)| (*a as i32 - *b as i32).abs())
            .max()
            .unwrap_or(0);
        cross.push(obj(vec![
            ("signal", s("speech")),
            ("bitrate", int(st.bitrate as i64)),
            ("max_abs_diff", int(max_diff as i64)),
        ]));
    }

    let speech_json = obj(vec![
        ("source", s("fixtures/answer_16k.wav")),
        (
            "note",
            s(
                "Telephone-band recording: almost no energy above ~3.8 kHz, and no true silence \
               (quietest 20 ms frames are about -27 dBFS).",
            ),
        ),
        ("start_s", num(SPEECH_START_S, 3)),
        ("dur_s", num(SPEECH_DUR_S, 3)),
        ("sample_rate", int(SPEECH_RATE as i64)),
        ("n_samples", int(len as i64)),
        ("frame_samples", int(sframe as i64)),
        ("pcm", b64_pcm(&speech)),
        ("rms", num(rms(&speech), 1)),
        (
            "encodes",
            J::Arr(
                speech_streams
                    .iter()
                    .map(|st| {
                        let r = libopus_reference(
                            &speech,
                            SPEECH_RATE,
                            Application::Voip,
                            st.bitrate,
                            n_speech_frames,
                        );
                        stream_json(st, &speech, n_speech_frames, "voip", r)
                    })
                    .collect(),
            ),
        ),
        (
            "silk_frames",
            obj(vec![
                ("bitrate", int(SPEECH_TRACE_BITRATE as i64)),
                ("lpc_order", int(silk_frames[0].lpc_order as i64)),
                ("core_to_output_offset", int(core_offset as i64)),
                ("frames", silk_frames_json),
            ]),
        ),
        ("silk_detail", silk_detail),
        (
            "plc",
            obj(vec![
                ("bitrate", int(SPEECH_TRACE_BITRATE as i64)),
                ("lost_frames", ints(&PLC_LOST_FRAMES.map(|v| v as i64))),
                (
                    "lost_start_samples",
                    ints(&PLC_LOST_FRAMES.map(|k| (k * sframe) as i64 - trace.delay as i64)),
                ),
                ("decoded", b64_pcm(plc_aligned)),
                ("snr_db", num(snr_db(&speech, plc_aligned), 2)),
            ]),
        ),
    ]);

    // ---------------- B. Music ----------------
    println!("== Music ==");
    let (music, events) = synth_music(MUSIC_RATE as usize, MUSIC_DUR_S);
    let mlen = music.len();
    let mframe = MUSIC_RATE as usize / 50;
    let n_music_frames = mlen / mframe;
    println!(
        "  synthesized {} samples, peak {}, rms {:.1}",
        mlen,
        music.iter().map(|v| (*v as i32).abs()).max().unwrap(),
        rms(&music)
    );
    let mut music_streams = Vec::new();
    for &br in &MUSIC_BITRATES {
        let st = make_stream(&music, MUSIC_RATE, Application::Audio, br);
        let aligned = st.aligned(mlen);
        let (si, hy, ce) = mode_counts(&st.packets[..n_music_frames]);
        println!(
            "  {:>6} b/s: delay {} smp, SNR {:6.2} dB, segSNR {:6.2} dB, rms ratio {:.3}, {:.2} kb/s, SILK/Hybrid/CELT {}/{}/{}",
            br,
            st.delay,
            snr_db(&music, aligned),
            seg_snr_db(&music, aligned, mframe),
            rms(aligned) / rms(&music),
            st.packets[..n_music_frames]
                .iter()
                .map(|p| p.len())
                .sum::<usize>() as f64
                * 8.0
                / (n_music_frames * FRAME_MS) as f64,
            si,
            hy,
            ce
        );
        let lib = decode_libopus(&st.packets, MUSIC_RATE, &[]);
        let max_diff = lib
            .iter()
            .zip(&st.decoded_raw)
            .map(|(a, b)| (*a as i32 - *b as i32).abs())
            .max()
            .unwrap_or(0);
        cross.push(obj(vec![
            ("signal", s("music")),
            ("bitrate", int(br as i64)),
            ("max_abs_diff", int(max_diff as i64)),
        ]));
        music_streams.push(st);
    }
    let mtrace = music_streams
        .iter()
        .find(|s| s.bitrate == MUSIC_TRACE_BITRATE)
        .unwrap();
    let (celt_frames, celt_out) = trace_celt(&mtrace.packets);
    let celt_out_equal = celt_out == mtrace.decoded_raw;
    let celt_rc_ok = celt_frames.iter().flatten().all(|f| f.rc_match);
    println!(
        "  CELT replay: standalone output == OpusDecoder: {}, range coder states equal: {} ({} CELT frames)",
        celt_out_equal,
        celt_rc_ok,
        celt_frames.iter().flatten().count()
    );
    all_ok &= celt_rc_ok && celt_out_equal;
    validation.push((
        "celt_replay",
        obj(vec![
            (
                "standalone_output_equals_opus_decoder",
                J::Bool(celt_out_equal),
            ),
            ("range_decoder_state_equal_all_frames", J::Bool(celt_rc_ok)),
        ]),
    ));
    let transients: Vec<usize> = celt_frames
        .iter()
        .enumerate()
        .filter(|(_, f)| f.as_ref().is_some_and(|f| f.transient))
        .map(|(k, _)| k)
        .collect();
    println!("  transient (short-block) frames: {:?}", transients);
    let celt_start = |k: usize| (k * mframe) as i64 - mtrace.delay as i64;
    let celt_frames_json = J::Arr(
        celt_frames
            .iter()
            .enumerate()
            .map(|(k, f)| match f {
                Some(f) => celt_frame_json(k, f, celt_start(k), mtrace.packets[k].len()),
                None => J::Null,
            })
            .collect(),
    );

    // Detail frames: the first transient frame, a tonal frame with the
    // post-filter on during the pad + melody, and the hi-hat frame.
    let frame_at = |t: f64| ((t * MUSIC_RATE as f64) as usize + mtrace.delay) / mframe;
    let tonal = (frame_at(0.4)..frame_at(0.65))
        .find(|&k| {
            celt_frames[k]
                .as_ref()
                .is_some_and(|f| f.postfilter.is_some() && !f.transient)
        })
        .unwrap_or(frame_at(0.45));
    let castanet = frame_at(0.68);
    let transient = transients
        .iter()
        .copied()
        .min_by_key(|&k| k.abs_diff(castanet))
        .unwrap_or(castanet);
    let detail_list = [
        ("tonal", tonal),
        ("transient", transient),
        ("hihat", frame_at(1.32)),
    ];
    let celt_detail = J::Arr(
        detail_list
            .iter()
            .filter_map(|&(role, k)| celt_frames[k].as_ref().map(|f| (role, k, f)))
            .map(|(role, k, f)| {
                let start = k * mframe;
                obj(vec![
                    ("role", s(role)),
                    ("index", int(k as i64)),
                    ("start_sample", int(celt_start(k))),
                    ("transient", J::Bool(f.transient)),
                    (
                        "mdct",
                        nums(&f.mdct.iter().map(|&v| v as f64).collect::<Vec<_>>(), 1),
                    ),
                    ("output", ints(&mtrace.decoded_raw[start..start + mframe])),
                ])
            })
            .collect(),
    );
    println!(
        "  detail frames: {:?}",
        detail_list.iter().collect::<Vec<_>>()
    );

    let music_json = obj(vec![
        (
            "source",
            s("synthetic (examples/opus_demo_data.rs, synth_music)"),
        ),
        ("sample_rate", int(MUSIC_RATE as i64)),
        ("dur_s", num(MUSIC_DUR_S, 3)),
        ("n_samples", int(mlen as i64)),
        ("frame_samples", int(mframe as i64)),
        ("pcm", b64_pcm(&music)),
        ("rms", num(rms(&music), 1)),
        (
            "events",
            J::Arr(
                events
                    .iter()
                    .map(|e| {
                        obj(vec![
                            ("t_s", num(e.t_s, 3)),
                            ("kind", s(e.kind)),
                            ("detail", s(&e.detail)),
                        ])
                    })
                    .collect(),
            ),
        ),
        (
            "encodes",
            J::Arr(
                music_streams
                    .iter()
                    .map(|st| {
                        let r = libopus_reference(
                            &music,
                            MUSIC_RATE,
                            Application::Audio,
                            st.bitrate,
                            n_music_frames,
                        );
                        stream_json(st, &music, n_music_frames, "audio", r)
                    })
                    .collect(),
            ),
        ),
        (
            "celt_frames",
            obj(vec![
                ("bitrate", int(MUSIC_TRACE_BITRATE as i64)),
                ("frames", celt_frames_json),
            ]),
        ),
        ("celt_detail", celt_detail),
    ]);

    // ---------------- C. Mode map ----------------
    println!("== Mode map ==");
    let speech48 = upsample3(&speech);
    let up_check = {
        // The upsampled signal, decimated back by 3, should match the input.
        let back: Vec<i16> = speech48.iter().step_by(3).copied().collect();
        snr_db(&speech, &back)
    };
    println!(
        "  upsampler check: 48k->16k decimation vs input SNR {:.1} dB",
        up_check
    );
    let series = [
        (
            "speech_voip",
            "speech, upsampled to 48 kHz",
            &speech48,
            Application::Voip,
            "voip",
        ),
        (
            "speech_audio",
            "speech, upsampled to 48 kHz",
            &speech48,
            Application::Audio,
            "audio",
        ),
        (
            "music_audio",
            "synthetic music",
            &music,
            Application::Audio,
            "audio",
        ),
    ];
    let mut mode_series = Vec::new();
    for (name, signal, pcm, app, app_name) in series {
        let n_frames = pcm.len() / mframe;
        let mut rs_rows = Vec::new();
        let mut lib_rows = Vec::new();
        let mut line_rs = String::new();
        let mut line_lib = String::new();
        for &kbps in &MODE_MAP_KBPS {
            let p_rs = encode_opus1(pcm, MUSIC_RATE, app, kbps * 1000);
            let p_lib = encode_libopus(pcm, MUSIC_RATE, app, kbps * 1000);
            let short = |p: &[Vec<u8>]| {
                format!(
                    "{}{}",
                    &majority(&p[..n_frames], |t| t.mode.to_string())[..1],
                    majority(&p[..n_frames], |t| t.bandwidth.to_string())
                )
            };
            let _ = write!(line_rs, "{:>6}", short(&p_rs));
            let _ = write!(line_lib, "{:>6}", short(&p_lib));
            rs_rows.push(mode_map_entry(&p_rs, n_frames, kbps));
            lib_rows.push(mode_map_entry(&p_lib, n_frames, kbps));
        }
        println!("  {:<13} opus1:   {}", name, line_rs);
        println!("  {:<13} libopus: {}", name, line_lib);
        mode_series.push(obj(vec![
            ("name", s(name)),
            ("signal", s(signal)),
            ("sample_rate", int(MUSIC_RATE as i64)),
            ("application", s(app_name)),
            ("opus1", J::Arr(rs_rows)),
            ("libopus", J::Arr(lib_rows)),
        ]));
    }
    let mode_map_json = obj(vec![
        ("bitrates_kbps", ints(&MODE_MAP_KBPS)),
        ("frame_ms", int(FRAME_MS as i64)),
        (
            "upsampler",
            s("3x zero-stuffing + 145-tap Kaiser-windowed sinc (beta 8, cutoff 7.6 kHz)"),
        ),
        ("series", J::Arr(mode_series)),
    ]);

    // ---------------- D. Meta ----------------
    let mut lib_delay = Vec::new();
    for (name, rate, app, oapp, pcm) in [
        (
            "speech_16k_voip",
            SPEECH_RATE,
            Application::Voip,
            opus::Application::Voip,
            &speech,
        ),
        (
            "music_48k_audio",
            MUSIC_RATE,
            Application::Audio,
            opus::Application::Audio,
            &music,
        ),
    ] {
        let p = encode_libopus(
            pcm,
            rate,
            app,
            if rate == SPEECH_RATE { 16000 } else { 64000 },
        );
        let d = decode_libopus(&p, rate, &[]);
        let measured = find_delay(pcm, &d);
        let lookahead = libopus_lookahead(rate, oapp);
        println!(
            "  libopus {}: lookahead {} smp, measured delay {} smp",
            name, lookahead, measured
        );
        lib_delay.push(obj(vec![
            ("stream", s(name)),
            ("lookahead_samples", int(lookahead as i64)),
            ("measured_delay_samples", int(measured as i64)),
        ]));
    }
    validation.push(("libopus_decodes_opus1_packets", J::Arr(cross)));
    validation.push((
        "plc_vs_libopus_decoder",
        obj(vec![
            ("concealed_max_abs_diff", int(plc_max_diff as i64)),
            ("concealed_snr_db", num(plc_vs_lib_snr, 2)),
            (
                "first_good_frames",
                ints(&after_loss.iter().map(|&k| k as i64).collect::<Vec<_>>()),
            ),
            ("first_good_max_abs_diff", int(rec_max_diff as i64)),
            ("first_good_snr_db", num(rec_vs_lib_snr, 2)),
        ]),
    ));
    validation.push(("upsampler_roundtrip_snr_db", num(up_check, 1)));

    let mode = default_mode();
    let celt_meta = obj(vec![
        (
            "band_edges_hz",
            ints(
                &mode
                    .e_bands
                    .iter()
                    .map(|&e| e as i64 * 200)
                    .collect::<Vec<_>>(),
            ),
        ),
        (
            "band_edges_bins_20ms",
            ints(
                &mode
                    .e_bands
                    .iter()
                    .map(|&e| e as i64 * 8)
                    .collect::<Vec<_>>(),
            ),
        ),
        (
            "e_means",
            nums(
                &mode.e_means[..NB_EBANDS]
                    .iter()
                    .map(|&v| v as f64)
                    .collect::<Vec<_>>(),
                4,
            ),
        ),
        ("short_mdct_size", int(mode.short_mdct_size as i64)),
        ("overlap", int(mode.overlap as i64)),
        ("preemph", num(mode.preemph[0] as f64, 4)),
        ("lm_20ms", int(3)),
    ]);
    let toc_configs = J::Arr(
        (0u8..32)
            .map(|c| {
                let t = parse_toc(c << 3);
                obj(vec![
                    ("config", int(t.config as i64)),
                    ("mode", s(t.mode)),
                    ("bandwidth", s(t.bandwidth)),
                    ("frame_ms", num(t.frame_ms, 1)),
                ])
            })
            .collect(),
    );
    let bandwidths = obj(vec![
        (
            "NB",
            obj(vec![("audio_hz", int(4000)), ("sample_rate", int(8000))]),
        ),
        (
            "MB",
            obj(vec![("audio_hz", int(6000)), ("sample_rate", int(12000))]),
        ),
        (
            "WB",
            obj(vec![("audio_hz", int(8000)), ("sample_rate", int(16000))]),
        ),
        (
            "SWB",
            obj(vec![("audio_hz", int(12000)), ("sample_rate", int(24000))]),
        ),
        (
            "FB",
            obj(vec![("audio_hz", int(20000)), ("sample_rate", int(48000))]),
        ),
    ]);
    let sizes = obj(vec![
        ("speech_bytes", int(json_len(&speech_json) as i64)),
        ("music_bytes", int(json_len(&music_json) as i64)),
        ("mode_map_bytes", int(json_len(&mode_map_json) as i64)),
    ]);
    let meta = obj(vec![
        ("generator", s("examples/opus_demo_data.rs")),
        (
            "regenerate",
            s("cargo run --release --example opus_demo_data"),
        ),
        ("crate", s(env!("CARGO_PKG_NAME"))),
        ("crate_version", s(env!("CARGO_PKG_VERSION"))),
        ("git_commit", J::Str(git_commit(&root))),
        ("libopus_version", s(opus::version())),
        (
            "encoding",
            obj(vec![
                ("pcm", s("base64 of little-endian int16, mono")),
                (
                    "packets",
                    s("base64 of the raw Opus packet bytes (ToC byte first)"),
                ),
            ]),
        ),
        ("frame_ms", int(FRAME_MS as i64)),
        ("complexity", int(COMPLEXITY as i64)),
        ("vbr", J::Bool(true)),
        (
            "codec_delay",
            obj(vec![
                ("speech_16k_voip_samples", int(trace.delay as i64)),
                ("music_48k_audio_samples", int(mtrace.delay as i64)),
                ("libopus", J::Arr(lib_delay)),
            ]),
        ),
        (
            "validation",
            J::Obj(
                validation
                    .into_iter()
                    .map(|(k, v)| (k.to_string(), v))
                    .collect(),
            ),
        ),
        (
            "notes",
            J::Arr(
                [
                    "Both encoders choose the bandwidth from the bitrate \
                     (opus_encoder.c, *_bandwidth_thresholds). At complexity 7 and \
                     above both also run the speech/music and bandwidth analysis \
                     (libopus analysis.c; opus1 src/analysis/, cargo feature \
                     `analysis`, on by default). At the complexity used here (9) the \
                     two mode_map rows pick the same mode and bandwidth at all 13 \
                     bitrates.",
                    "SILK-only packets have trailing zero bytes stripped by the encoder \
                     (libopus opus_encoder.c, 'strip off trailing zero bytes'); the range \
                     decoder reads zeros past the end, so silk bits.unused can be slightly \
                     negative.",
                    "The speech fixture is telephone-band (no energy above ~3.8 kHz), so \
                     libopus's bandwidth detection and speech/music analysis treat it \
                     differently from wideband speech.",
                ]
                .iter()
                .map(|v| s(v))
                .collect(),
            ),
        ),
        ("toc_configs", toc_configs),
        ("bandwidths", bandwidths),
        ("celt", celt_meta),
        ("sizes", sizes),
    ]);

    let root_json = obj(vec![
        ("meta", meta),
        ("speech", speech_json),
        ("music", music_json),
        ("mode_map", mode_map_json),
    ]);
    let mut text = String::from(
        "// Generated by examples/opus_demo_data.rs - do not edit by hand.\n\
         // Regenerate: cargo run --release --example opus_demo_data\n\
         window.OPUS_DEMO_DATA = ",
    );
    write_json(&mut text, &root_json, 0);
    text.push_str(";\n");
    fs::create_dir_all(out_path.parent().unwrap()).unwrap();
    if !all_ok {
        // Do not touch the committed data file when a replay check failed.
        let rejected = root.join("target/opus-demo-data.rejected.js");
        fs::create_dir_all(rejected.parent().unwrap()).unwrap();
        fs::write(&rejected, &text).unwrap();
        println!(
            "WARNING: a replay check failed; see the lines above. {:?} is unchanged; the new data is in {:?}.",
            out_path, rejected
        );
        std::process::exit(1);
    }
    // Write a temporary file next to the output, then rename it, so that a
    // reader never sees a half-written file.
    let tmp_path = out_path.with_extension("js.tmp");
    fs::write(&tmp_path, &text).unwrap();
    fs::rename(&tmp_path, &out_path).unwrap();
    println!("Wrote {:?} ({} bytes)", out_path, text.len());
}
