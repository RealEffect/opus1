//! Shared code of the end-to-end multistream tests
//! (`tests/multistream_e2e_*.rs`).
//!
//! A test file uses it with
//! `mod ms_support; mod multistream_enc_common; mod multistream_e2e_common;`.
//!
//! `measure` runs one setting through four paths and returns what it saw:
//!
//! | encoder | decoder | what it shows                                   |
//! |---------|---------|-------------------------------------------------|
//! | opus1 | C       | the opus1 encoder, judged by the reference    |
//! | C       | C       | the reference for every number                  |
//! | opus1 | opus1 | the path an application of opus1 uses         |
//! | C       | opus1 | the opus1 decoder on packets of the reference |

#![allow(dead_code)]

use crate::ms_support::*;
use crate::multistream_enc_common::*;
use opus1::Application;
use opus1::multistream::{MappingType, MultistreamDecoder, MultistreamEncoder};

/// The layouts of the end-to-end tests: every channel count of mapping
/// family 1, two layouts of family 255 and three of family 2.
pub const LAYOUTS: &[Case] = &[
    Case::family("family 1, mono", 1, 1),
    Case::family("family 1, stereo", 2, 1),
    Case::family("family 1, 3 channels", 3, 1),
    Case::family("family 1, quad", 4, 1),
    Case::family("family 1, 5 channels", 5, 1),
    Case::family("family 1, 5.1", 6, 1),
    Case::family("family 1, 6.1", 7, 1),
    Case::family("family 1, 7.1", 8, 1),
    Case::family("family 255, 3 channels", 3, 255),
    Case::family("family 255, 6 channels", 6, 255),
    Case::family("family 2, 4 channels", 4, 2),
    Case::family("family 2, 6 channels", 6, 2),
    Case::family("family 2, 9 channels", 9, 2),
];

/// Bitrates of the matrix in bits per second per input channel.
pub const RATES_PER_CHANNEL: [i32; 3] = [12_000, 32_000, 64_000];

/// The kind of stream an input channel is coded in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum StreamKind {
    /// A mono stream.
    Mono,
    /// One of the two channels of a stereo stream.
    Coupled,
    /// The stream of the low-frequency-effects channel.
    Lfe,
}

impl StreamKind {
    pub const ALL: [StreamKind; 3] = [StreamKind::Mono, StreamKind::Coupled, StreamKind::Lfe];

    pub fn name(self) -> &'static str {
        match self {
            StreamKind::Mono => "mono",
            StreamKind::Coupled => "coupled",
            StreamKind::Lfe => "LFE",
        }
    }
}

/// The kind of stream of stream `s`.
pub fn stream_kind(enc: &MultistreamEncoder, s: usize) -> StreamKind {
    if s < enc.layout().coupled_streams() {
        StreamKind::Coupled
    } else if Some(s) == enc.lfe_stream() {
        StreamKind::Lfe
    } else {
        StreamKind::Mono
    }
}

/// The stream that codes input channel `channel`.
pub fn stream_of_channel(enc: &MultistreamEncoder, channel: usize) -> usize {
    let layout = enc.layout();
    let coded = layout.mapping()[channel] as usize;
    if coded < 2 * layout.coupled_streams() {
        coded / 2
    } else {
        coded - layout.coupled_streams()
    }
}

/// Whether libopus fixes the coding mode of stream `s` from the multistream
/// layer (opus_multistream_encoder.c:953-962): the coupled streams of a
/// surround layout and every stream of an ambisonics layout are CELT-only.
/// The LFE stream is CELT-only by its own flag (opus_encoder.c:1538).
pub fn mode_is_fixed(enc: &MultistreamEncoder, s: usize) -> bool {
    match enc.mapping_type() {
        MappingType::Surround => stream_kind(enc, s) != StreamKind::Mono,
        MappingType::Ambisonics => true,
        MappingType::None => false,
    }
}

/// Amplitude of the noise in `program`: 54 dB below full scale.
pub const NOISE_FLOOR: f32 = 0.002;

/// The test signal: the speech-like signal of `voiced` on every channel,
/// with its own pitch per channel, over a little white noise. The LFE
/// channel holds what such a channel carries: two low tones, 45 and 80 Hz,
/// that swell and fade.
///
/// The noise is there because `voiced` has nothing above 2.5 kHz. The
/// analysis of libopus notices that and codes a narrower audio bandwidth
/// than the bitrate allows (opus_encoder.c:1652-1673), which opus1,
/// without the analysis, does not. With the noise both encoders code the
/// same bandwidth, and the tests compare the rest.
/// `a_signal_without_high_frequencies_gets_a_narrower_bandwidth_in_libopus`
/// shows the signal without it.
pub fn program(fs: i32, case: &Case, samples: usize) -> Vec<f32> {
    let mut pcm = program_without_noise(fs, case, samples);
    for (s, n) in pcm
        .iter_mut()
        .zip(noise(case.channels, samples, NOISE_FLOOR, 7))
    {
        *s += n;
    }
    pcm
}

/// `program` without its noise: no channel has anything above 2.5 kHz.
pub fn program_without_noise(fs: i32, case: &Case, samples: usize) -> Vec<f32> {
    let channels = case.channels;
    let mut pcm = voiced(fs, channels, samples);
    if let Some(lfe) = lfe_channel(channels, case.tone_family()) {
        let two_pi = 2.0 * std::f64::consts::PI;
        for (i, frame) in pcm.chunks_exact_mut(channels).enumerate() {
            let t = i as f64 / fs as f64;
            let env = 0.6 + 0.4 * (two_pi * 1.5 * t).sin();
            let v = 0.2 * (two_pi * 45.0 * t).sin() + 0.15 * (two_pi * 80.0 * t).sin();
            frame[lfe] = (env * v) as f32;
        }
    }
    pcm
}

/// One setting of the matrix.
#[derive(Debug, Clone, Copy)]
pub struct Setting {
    pub case: Case,
    pub fs: i32,
    pub application: Application,
    /// Samples per channel.
    pub frame_size: usize,
    pub vbr: bool,
    /// Bits per second for all channels together.
    pub bitrate: i32,
    /// 0 to 10. Both encoders start at 9.
    pub complexity: i32,
}

impl Setting {
    pub fn label(&self) -> String {
        format!(
            "{}, {} Hz, {}, {}, {} b/s",
            self.case.name,
            self.fs,
            ms(self.frame_size, self.fs),
            if self.vbr { "VBR" } else { "CBR" },
            self.bitrate
        )
    }
}

/// The two encoders, in the order of every pair in `Measured`.
pub const RUST: usize = 0;
pub const C: usize = 1;

/// The four paths, in the order of `ChannelSnr::snr`.
pub const RUST_TO_C: usize = 0;
pub const C_TO_C: usize = 1;
pub const RUST_TO_RUST: usize = 2;
pub const C_TO_RUST: usize = 3;

/// SNR of one input channel on the four paths, in dB.
#[derive(Debug, Clone, Copy)]
pub struct ChannelSnr {
    pub kind: StreamKind,
    pub snr: [f64; 4],
}

/// How the streams of the opus1 packets differ from those of the C
/// packets, frame by frame and stream by stream. Every field counts stream
/// packets.
#[derive(Debug, Clone, Copy, Default)]
pub struct Structure {
    /// Stream packets compared.
    pub compared: usize,
    /// Both without audio: the TOC byte differs.
    pub empty_toc: usize,
    /// One has audio and the other has none.
    pub audio: usize,
    /// Another coding mode, in a stream whose mode libopus fixes.
    pub fixed_mode: usize,
    /// Another coding mode, in a stream that chooses its mode itself.
    pub free_mode: usize,
    /// The same mode and another audio bandwidth.
    pub bandwidth: usize,
    /// Another number of coded channels.
    pub channels: usize,
}

impl Structure {
    pub fn add(&mut self, other: &Structure) {
        self.compared += other.compared;
        self.empty_toc += other.empty_toc;
        self.audio += other.audio;
        self.fixed_mode += other.fixed_mode;
        self.free_mode += other.free_mode;
        self.bandwidth += other.bandwidth;
        self.channels += other.channels;
    }
}

/// What one stream holds over the whole signal.
#[derive(Debug, Clone, Copy)]
pub struct StreamRate {
    pub kind: StreamKind,
    /// Bits per second the encoder set on the stream for the last frame.
    pub set: [i32; 2],
    /// Bits per second the stream took in the packets, on average.
    pub used: [f64; 2],
}

/// The result of `measure`.
#[derive(Debug, Clone)]
pub struct Measured {
    pub setting: Setting,
    /// Bits per second of the packets, on average: opus1, C.
    pub rate: [f64; 2],
    pub streams: Vec<StreamRate>,
    pub structure: Structure,
    pub channels: Vec<ChannelSnr>,
    /// The first difference of each kind, for failure messages.
    pub examples: Vec<String>,
}

/// An opus1 decoder with the layout of the encoder.
pub fn rust_decoder(enc: &MultistreamEncoder) -> MultistreamDecoder {
    let layout = enc.layout();
    MultistreamDecoder::with_layout(
        enc.sampling_rate(),
        layout.channels(),
        layout.streams(),
        layout.coupled_streams(),
        layout.mapping(),
    )
    .expect("opus1 multistream decoder")
}

/// Decodes a sequence of packets with the opus1 decoder.
pub fn rust_decode_all(
    dec: &mut MultistreamDecoder,
    packets: &[Vec<u8>],
    frame_size: usize,
    label: &str,
) -> Vec<f32> {
    let channels = dec.layout().channels();
    let mut frame = vec![0f32; frame_size * channels];
    let mut out = Vec::with_capacity(packets.len() * frame.len());
    for (f, packet) in packets.iter().enumerate() {
        let n = dec
            .decode(packet, frame_size, &mut frame)
            .unwrap_or_else(|e| panic!("{label}: frame {f}: opus1 decode: {e}"));
        assert_eq!(n, frame_size, "{label}: frame {f}: decoded samples");
        out.extend_from_slice(&frame);
    }
    out
}

/// Codes `seconds` of the test signal with both encoders under the same
/// setting and decodes both packet sequences with both decoders.
///
/// The SNR of a channel is measured against the input at the delay of the
/// codec, which is the encoder's lookahead. The first 200 ms are left out.
pub fn measure(setting: Setting, seconds: f64) -> Measured {
    let frames = ((seconds * setting.fs as f64) as usize / setting.frame_size).max(1);
    let pcm = program(setting.fs, &setting.case, frames * setting.frame_size);
    measure_signal(setting, &pcm)
}

/// `measure` for a signal of the caller: whole frames of interleaved
/// samples.
pub fn measure_signal(setting: Setting, pcm: &[f32]) -> Measured {
    let Setting {
        case,
        fs,
        application,
        frame_size,
        vbr,
        bitrate,
        complexity,
    } = setting;
    let label = setting.label();
    let channels = case.channels;
    let (mut rust, mut c) = encoders(&case, fs, application);
    configure(&mut rust, &mut c, bitrate, vbr);
    rust.set_complexity(complexity).unwrap();
    c.set_complexity(complexity);
    let streams = c.streams;

    let frames = pcm.len() / (frame_size * channels);
    assert_eq!(pcm.len(), frames * frame_size * channels, "{label}");
    let packets = [
        rust_encode_all(&mut rust, pcm, frame_size),
        c.encode_all(pcm, frame_size),
    ];
    assert_eq!(packets[RUST].len(), frames, "{label}");
    assert_eq!(packets[C].len(), frames, "{label}");

    // Structure and bytes of every stream.
    let mut structure = Structure::default();
    let mut examples = Vec::new();
    let mut example = |count: &mut usize, text: String| {
        if *count == 0 {
            examples.push(text);
        }
        *count += 1;
    };
    let mut stream_bytes = vec![[0usize; 2]; streams];
    for (f, (rp, cp)) in packets[RUST].iter().zip(&packets[C]).enumerate() {
        let r_parts =
            split_streams(rp, streams).unwrap_or_else(|e| panic!("{label}: frame {f}: {e}"));
        let c_parts =
            split_streams(cp, streams).unwrap_or_else(|e| panic!("{label}: frame {f}: C: {e}"));
        for (s, (r, cs)) in r_parts.iter().zip(&c_parts).enumerate() {
            stream_bytes[s][RUST] += r.used;
            stream_bytes[s][C] += cs.used;
            structure.compared += 1;
            assert_eq!(r.samples(fs), frame_size, "{label}: frame {f}: stream {s}");
            let (t, ct) = (toc(r.toc), toc(cs.toc));
            let at = format!("{label}: frame {f}: stream {s}: opus1 {t}, C {ct}");
            if r.payload_len == 0 || cs.payload_len == 0 {
                if r.payload_len != cs.payload_len {
                    example(&mut structure.audio, format!("{at}: audio in one only"));
                } else if r.toc != cs.toc {
                    example(&mut structure.empty_toc, format!("{at}: without audio"));
                }
                continue;
            }
            if t.mode != ct.mode {
                if mode_is_fixed(&rust, s) {
                    example(&mut structure.fixed_mode, format!("{at}: fixed mode"));
                } else {
                    example(&mut structure.free_mode, format!("{at}: mode"));
                }
            } else if t.bandwidth != ct.bandwidth {
                example(&mut structure.bandwidth, format!("{at}: bandwidth"));
            }
            if t.stereo != ct.stereo {
                example(&mut structure.channels, format!("{at}: channels"));
            }
        }
    }
    let duration = (frames * frame_size) as f64 / fs as f64;
    let total = |packets: &[Vec<u8>]| {
        8.0 * packets.iter().map(|p| p.len()).sum::<usize>() as f64 / duration
    };
    let rate = [total(&packets[RUST]), total(&packets[C])];
    let stream_rates = (0..streams)
        .map(|s| StreamRate {
            kind: stream_kind(&rust, s),
            set: [
                bitrate_set(rust.stream_encoder(s).unwrap()),
                c.stream_bitrate(s),
            ],
            used: [
                8.0 * stream_bytes[s][RUST] as f64 / duration,
                8.0 * stream_bytes[s][C] as f64 / duration,
            ],
        })
        .collect();

    // The four paths.
    let decoded = [
        c_decoder(&rust).decode_all(&packets[RUST]),
        c_decoder(&rust).decode_all(&packets[C]),
        rust_decode_all(&mut rust_decoder(&rust), &packets[RUST], frame_size, &label),
        rust_decode_all(&mut rust_decoder(&rust), &packets[C], frame_size, &label),
    ];
    assert_eq!(rust.lookahead() as i32, c.lookahead(), "{label}: lookahead");
    let lag = rust.lookahead() as isize;
    let skip = fs as usize / 5;
    let channel_snrs = (0..channels)
        .map(|ch| {
            let input = channel(pcm, channels, ch);
            let mut snr = [0f64; 4];
            for (path, out) in decoded.iter().enumerate() {
                assert_eq!(out.len(), pcm.len(), "{label}: path {path}: decoded length");
                snr[path] = snr_at_lag(&input, &channel(out, channels, ch), lag, skip);
            }
            ChannelSnr {
                kind: stream_kind(&rust, stream_of_channel(&rust, ch)),
                snr,
            }
        })
        .collect();

    Measured {
        setting,
        rate,
        streams: stream_rates,
        structure,
        channels: channel_snrs,
        examples,
    }
}

/// Smallest, mean and largest of a set of numbers.
#[derive(Debug, Clone, Copy)]
pub struct Spread {
    pub min: f64,
    pub mean: f64,
    pub max: f64,
    pub count: usize,
}

impl Spread {
    pub fn of(values: impl IntoIterator<Item = f64>) -> Option<Self> {
        let mut s = Spread {
            min: f64::INFINITY,
            mean: 0.0,
            max: f64::NEG_INFINITY,
            count: 0,
        };
        for v in values {
            s.min = s.min.min(v);
            s.max = s.max.max(v);
            s.mean += v;
            s.count += 1;
        }
        (s.count > 0).then(|| {
            s.mean /= s.count as f64;
            s
        })
    }
}

impl std::fmt::Display for Spread {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:+.1} ({:+.1} to {:+.1})", self.mean, self.min, self.max)
    }
}

/// The difference in dB between two paths for every channel of one kind, in
/// all of `results`.
pub fn snr_differences(
    results: &[Measured],
    kind: StreamKind,
    path: usize,
    reference: usize,
) -> Option<Spread> {
    Spread::of(results.iter().flat_map(|m| {
        m.channels
            .iter()
            .filter(move |ch| ch.kind == kind)
            .map(move |ch| ch.snr[path] - ch.snr[reference])
    }))
}

/// The measurements of one layout in a few lines of text.
pub fn summary(name: &str, results: &[Measured]) -> String {
    let mut text = format!("{name}: {} settings\n", results.len());
    let percent = |which: usize| {
        Spread::of(
            results
                .iter()
                .map(|m| 100.0 * (m.rate[which] / m.setting.bitrate as f64 - 1.0)),
        )
        .unwrap()
    };
    text += &format!(
        "  rate against the request in %: opus1 {}, libopus {}\n",
        percent(RUST),
        percent(C)
    );
    let streams = results[0].streams.len();
    for s in 0..streams {
        let ratio = Spread::of(results.iter().map(|m| {
            100.0 * (m.streams[s].used[RUST] / m.streams[s].used[C].max(1.0) - 1.0)
        }))
        .unwrap();
        let same_setting = results
            .iter()
            .filter(|m| m.streams[s].set[RUST] == m.streams[s].set[C])
            .count();
        text += &format!(
            "  stream {s} ({}): bitrate set as in libopus in {same_setting} of {}; bytes against libopus in %: {ratio}\n",
            results[0].streams[s].kind.name(),
            results.len()
        );
    }
    let mut structure = Structure::default();
    for m in results {
        structure.add(&m.structure);
    }
    text += &format!(
        "  structure, of {} stream packets: {} fixed mode, {} own mode, {} bandwidth, {} channel count, {} audio in one only, {} TOC without audio\n",
        structure.compared,
        structure.fixed_mode,
        structure.free_mode,
        structure.bandwidth,
        structure.channels,
        structure.audio,
        structure.empty_toc
    );
    for kind in StreamKind::ALL {
        let Some(c) = Spread::of(results.iter().flat_map(|m| {
            m.channels
                .iter()
                .filter(move |ch| ch.kind == kind)
                .map(|ch| ch.snr[C_TO_C])
        })) else {
            continue;
        };
        text += &format!(
            "  {} channels, SNR in dB: libopus {:.1} ({:.1} to {:.1}); opus1 encoder against it {}; opus1 both ends against it {}; opus1 decoder on libopus packets against it {}\n",
            kind.name(),
            c.mean,
            c.min,
            c.max,
            snr_differences(results, kind, RUST_TO_C, C_TO_C).unwrap(),
            snr_differences(results, kind, RUST_TO_RUST, C_TO_C).unwrap(),
            snr_differences(results, kind, C_TO_RUST, C_TO_C).unwrap(),
        );
    }
    text
}
