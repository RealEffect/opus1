//! The encoders keep the channel order: every input channel carries a tone
//! of its own frequency, and after encoding and decoding no output channel
//! may hold the tone of another input channel.
//!
//! The input chooses a sampling rate, an application, a frame size, a
//! bitrate (automatic, or a value per channel), constant or variable
//! bitrate, and the complexity. Then it chooses either a stereo
//! `OpusEncoder`, with a forced coding mode or none, or a
//! `MultistreamEncoder`: the layout of a mapping family of the Ogg Opus
//! header, or channels, streams, coupled streams and a mapping. libopus
//! encodes the same tones with the same settings, and libopus's decoders
//! decode both streams, 0.4 s each.
//!
//! Each tone lies in a CELT band of its own below 4 kHz, so every
//! bandwidth carries it. The low-frequency-effects (LFE) channel of a
//! surround layout gets a tone in the lowest band. For each output channel
//! the check measures, after the first 100 ms, the strongest tone of
//! another input channel, relative to the level that tone has at the
//! input: the leakage, in dB. A channel on the wrong output comes out with
//! a leakage near 0 dB, and two channels mixed into one with -6 dB.
//!
//! At low bitrates and with short frames both encoders can mix the channels
//! a little or lose a tone, and opus1 and libopus differ there in detail.
//! So the check applies where libopus keeps the channels apart: wherever
//! libopus's leakage is -20 dB or less, opus1's must be below -6 dB. Over
//! 9,000 random stereo settings of every mode (14,596 channels), opus1's
//! leakage was at most -10.4 dB wherever libopus's was -20 dB or less. A
//! lost tone raises no leakage, so the check does not depend on it.
//!
//! Left out: the channels of a stereo stream that either encoder codes as
//! mono in some packet (the stereo flag of the first byte is clear), since
//! then the two channels are the same signal; and the channels of a stereo
//! stream that SILK codes (SILK-only or Hybrid) with a constant bitrate.
//! There the rate control of the fixed-point SILK encoder, which opus1
//! ports, can collapse the stereo image of tones, in libopus's fixed-point
//! build too. The reference here is libopus's float build. Measured at
//! 12 kHz, 160 kb/s, 20 ms frames: one output channel holds the other
//! channel's tone 8.8 dB above its own in fixed-point libopus and 15.5 dB
//! above its own in opus1, while the float build keeps its own tone at
//! least 78 dB above the other (docs/guide/10-differences-from-libopus.md).
//!
//! What the layout maps to the same coded channel comes from the first
//! input channel that names it, in both encoders. An output channel the
//! mapping leaves silent is not checked. A surround layout always gets an
//! explicit bitrate: with the automatic one the encoders choose other
//! bandwidths on purpose (D1 in docs/guide/10-differences-from-libopus.md).

#![no_main]

mod common;

use std::num::NonZeroU32;
use audiopus_sys as sys;
use common::*;
use libfuzzer_sys::arbitrary::{Result, Unstructured};
use libfuzzer_sys::fuzz_target;
use opus1::multistream::{MappingType, MultistreamEncoder, split_self_delimited};
use opus1::{Application, Bitrate, Channels, ForcedMode, OpusEncoder, SampleRate};

fuzz_target!(|data: &[u8]| {
    let _ = run(data);
});

/// The tone of input channel `c`: one CELT band each, below 4 kHz.
const TONES: [f32; 12] = [
    300.0, 500.0, 700.0, 900.0, 1100.0, 1300.0, 1500.0, 1800.0, 2200.0, 2600.0, 3000.0, 3600.0,
];
/// The tone of the LFE channel, in the lowest CELT band.
const LFE_TONE: f32 = 100.0;

/// How long each encoder codes the tones, and how much of the start the
/// check leaves out.
const SECONDS: f32 = 0.4;
const SKIP_SECONDS: f32 = 0.1;

/// The level of a tone at the input, as `power_db` measures it: a sine of
/// amplitude 0.25 has the power 0.25² / 4.
const TONE_DB: f64 = -18.061_799_739_838_87;
/// Where libopus's leakage shows that it keeps the channels apart.
const C_LEAK_DB: f64 = -20.0;
/// The largest leakage opus1 may have there.
const MAX_LEAK_DB: f64 = -6.0;

/// The settings both encoders get.
#[derive(Debug, Clone, Copy)]
struct Settings {
    fs: i32,
    app: Application,
    frame_size: usize,
    /// Bits per second for all channels, or `None` for the automatic
    /// bitrate.
    bitrate: Option<i32>,
    vbr: bool,
    complexity: u8,
}

impl Settings {
    fn new(u: &mut Unstructured) -> Result<Self> {
        let fs = *u.choose(&RATES)?;
        Ok(Self {
            fs,
            app: *u.choose(&[Application::Voip, Application::Audio, Application::RestrictedLowDelay])?,
            frame_size: *u.choose(&encoder_frame_sizes(fs))?,
            // Per channel; the caller multiplies.
            bitrate: if u.ratio(1, 6)? { None } else { Some(u.int_in_range(6_000..=128_000)?) },
            vbr: u.arbitrary()?,
            complexity: u.int_in_range(0..=10)?,
        })
    }

    fn bitrate_for(&self, channels: usize) -> Bitrate {
        match self.bitrate {
            None => Bitrate::Auto,
            Some(per_channel) => Bitrate::BitsPerSecond(NonZeroU32::new((per_channel * channels as i32) as u32).unwrap()),
        }
    }

    fn c_bitrate_for(&self, channels: usize) -> i32 {
        match self.bitrate {
            None => sys::OPUS_AUTO,
            Some(per_channel) => per_channel * channels as i32,
        }
    }
}

/// The tones of `channels` input channels, interleaved, for `frames`
/// frames of `frame_size` samples.
fn tones(tone: &[f32], fs: i32, frame_size: usize, frames: usize) -> Vec<f32> {
    let channels = tone.len();
    let mut pcm = vec![0f32; frames * frame_size * channels];
    for (t, frame) in pcm.chunks_exact_mut(channels).enumerate() {
        let time = t as f64 / fs as f64;
        for (x, &f) in frame.iter_mut().zip(tone) {
            *x = (0.25 * (2.0 * std::f64::consts::PI * f as f64 * time).sin()) as f32;
        }
    }
    pcm
}

/// The power of frequency `f` in `x` (Goertzel's algorithm), in dB.
fn power_db(x: &[f32], f: f32, fs: i32) -> f64 {
    let w = 2.0 * std::f64::consts::PI * f as f64 / fs as f64;
    let coef = 2.0 * w.cos();
    let (mut s1, mut s2) = (0.0f64, 0.0f64);
    for &v in x {
        let s0 = v as f64 + coef * s1 - s2;
        s2 = s1;
        s1 = s0;
    }
    let p = (s1 * s1 + s2 * s2 - coef * s1 * s2) / (x.len() as f64 * x.len() as f64);
    10.0 * (p + 1e-30).log10()
}

/// The leakage of each output channel: the strongest tone of another input
/// channel, in dB relative to its level at the input. `source[c]` is the
/// input channel output channel `c` carries, `None` for a silent one.
fn leakages(out: &[f32], channels: usize, tone: &[f32], source: &[Option<usize>], fs: i32) -> Vec<Option<f64>> {
    let skip = (SKIP_SECONDS * fs as f32) as usize;
    (0..channels)
        .map(|c| {
            let own = source[c]?;
            let x: Vec<f32> = out.chunks_exact(channels).skip(skip).map(|frame| frame[c]).collect();
            let other_db = source
                .iter()
                .flatten()
                .filter(|&&k| tone[k] != tone[own])
                .map(|&k| power_db(&x, tone[k], fs))
                .fold(f64::NEG_INFINITY, f64::max);
            Some(other_db - TONE_DB)
        })
        .collect()
}

/// The check: where libopus keeps a channel apart, opus1 must too.
/// `skip[c]` leaves output channel `c` out.
fn compare(what: &str, rust: &[Option<f64>], c: &[Option<f64>], skip: &[bool]) {
    for (ch, ((r, c), &skip)) in rust.iter().zip(c).zip(skip).enumerate() {
        if let (Some(r), Some(c), false) = (r, c, skip) {
            if *c <= C_LEAK_DB {
                assert!(
                    *r < MAX_LEAK_DB,
                    "{what}: output channel {ch}: another channel's tone at {r:.1} dB of its input level, libopus {c:.1} dB"
                );
            }
        }
    }
}

/// The first byte of each stream of a multistream packet.
fn stream_tocs(packet: &[u8], streams: usize, scratch: &mut Vec<u8>) -> Vec<u8> {
    let mut tocs = Vec::with_capacity(streams);
    let mut rest = packet;
    for _ in 0..streams - 1 {
        let (stream, used) = split_self_delimited(rest, scratch).expect("a stream of a multistream packet");
        tocs.push(stream[0]);
        rest = &rest[used..];
    }
    tocs.push(rest[0]);
    tocs
}

/// Whether the first byte names a SILK-only or Hybrid packet.
fn silk_coded(toc: u8) -> bool {
    toc >> 3 < 16
}

/// Whether the first byte names a packet coded as mono.
fn mono_coded(toc: u8) -> bool {
    toc & 0x04 == 0
}

fn run(data: &[u8]) -> Result<()> {
    let mut u = Unstructured::new(data);
    let s = Settings::new(&mut u)?;
    if u.arbitrary()? { stereo(&mut u, s) } else { multistream(&mut u, s) }
}

/// One stereo stream.
fn stereo(u: &mut Unstructured, s: Settings) -> Result<()> {
    let mode = *u.choose(&[None, Some(ForcedMode::SilkOnly), Some(ForcedMode::Hybrid), Some(ForcedMode::CeltOnly)])?;
    let mut rust =
        OpusEncoder::new(SampleRate::try_from(s.fs).unwrap(), Channels::Stereo, s.app).unwrap();
    rust.set_bitrate(s.bitrate_for(2));
    rust.set_vbr(s.vbr);
    rust.set_complexity(s.complexity).unwrap();
    rust.set_force_mode(mode);
    let mut c = CEncoder::new(s.fs, 2, s.app.to_raw());
    c.set(sys::OPUS_SET_BITRATE_REQUEST, s.c_bitrate_for(2));
    c.set(sys::OPUS_SET_VBR_REQUEST, s.vbr as i32);
    c.set(sys::OPUS_SET_COMPLEXITY_REQUEST, i32::from(s.complexity));
    if let Some(mode) = mode {
        c.set(OPUS_SET_FORCE_MODE_REQUEST, mode.to_raw());
    }

    let tone = [TONES[0], TONES[7]];
    let frames = ((SECONDS * s.fs as f32) as usize).div_ceil(s.frame_size);
    let pcm = tones(&tone, s.fs, s.frame_size, frames);
    let mut rust_dec = CDecoder::new(s.fs, 2);
    let mut c_dec = CDecoder::new(s.fs, 2);
    let mut packet = vec![0u8; 4000];
    let mut pcm_out = vec![0f32; s.frame_size * 2];
    let (mut rust_out, mut c_out) = (Vec::new(), Vec::new());
    let (mut silk, mut mono) = (false, false);
    for (n, frame) in pcm.chunks_exact(s.frame_size * 2).enumerate() {
        let len = rust.encode(frame, s.frame_size, &mut packet).unwrap_or_else(|e| panic!("frame {n}: {e}"));
        silk |= silk_coded(packet[0]);
        mono |= mono_coded(packet[0]);
        let got = rust_dec.decode_float(&packet[..len], &mut pcm_out, s.frame_size, false);
        assert_eq!(got, s.frame_size as i32, "frame {n}: libopus decodes opus1's packet");
        rust_out.extend_from_slice(&pcm_out);
        let len = c.encode(frame, s.frame_size, &mut packet);
        assert!(len > 0, "frame {n}: libopus encode {len}");
        silk |= silk_coded(packet[0]);
        mono |= mono_coded(packet[0]);
        let got = c_dec.decode_float(&packet[..len as usize], &mut pcm_out, s.frame_size, false);
        assert_eq!(got, s.frame_size as i32, "frame {n}: libopus decodes its packet");
        c_out.extend_from_slice(&pcm_out);
    }
    let source = [Some(0), Some(1)];
    let skip = [mono || (silk && !s.vbr); 2];
    let what = format!("stereo, {s:?}, {mode:?}");
    compare(&what, &leakages(&rust_out, 2, &tone, &source, s.fs), &leakages(&c_out, 2, &tone, &source, s.fs), &skip);
    Ok(())
}

/// A multistream encoder.
fn multistream(u: &mut Unstructured, s: Settings) -> Result<()> {
    let c_app = s.app.to_raw();
    let (rust, c, lfe) = if u.arbitrary()? {
        let family = *u.choose(&[0u8, 1, 1, 2, 255])?;
        let channels = match family {
            2 => *u.choose(&[1usize, 4, 6, 9, 11])?,
            255 => u.int_in_range(1..=TONES.len())?,
            _ => u.int_in_range(1..=8)?,
        };
        let Ok(rust) = MultistreamEncoder::new_surround(
            SampleRate::try_from(s.fs).unwrap(),
            channels,
            family,
            s.app,
        ) else {
            return Ok(());
        };
        let (c, streams, coupled, mapping) = CMsEncoder::surround(s.fs, channels, family, c_app)
            .unwrap_or_else(|e| panic!("libopus refuses family {family} with {channels} channels: {e}"));
        let l = rust.layout();
        assert_eq!((l.streams(), l.coupled_streams(), l.mapping()), (streams, coupled, &mapping[..]));
        // libopus codes the last channel of a 5.1 or 7.1 layout as LFE.
        let lfe = rust.lfe_stream().map(|_| channels - 1);
        (rust, c, lfe)
    } else {
        let channels = u.int_in_range(1..=8)?;
        let streams = u.int_in_range(1..=channels)?;
        let coupled = u.int_in_range(0..=streams.min(channels - streams))?;
        let mut mapping = vec![0u8; channels];
        for m in mapping.iter_mut() {
            *m = if u.ratio(1, 10)? { 255 } else { u.int_in_range(0..=(streams + coupled) as u8 - 1)? };
        }
        let Ok(rust) = MultistreamEncoder::new(
            SampleRate::try_from(s.fs).unwrap(),
            channels,
            streams,
            coupled,
            &mapping,
            s.app,
        ) else {
            return Ok(());
        };
        let c = CMsEncoder::new(s.fs, channels, streams, coupled, &mapping, c_app)
            .unwrap_or_else(|e| panic!("libopus refuses {channels} channels, {streams} streams, {coupled} coupled, {mapping:?}: {e}"));
        (rust, c, None)
    };
    let mut rust = rust;
    let mut c = c;
    let layout = rust.layout().clone();
    let channels = layout.channels();
    // With the automatic bitrate in surround mode the two encoders choose
    // other audio bandwidths on purpose (D1 in docs/guide/10: libopus compares the
    // raw setting and codes narrowband), so the allocation within each
    // stream differs. These layouts get an explicit bitrate.
    let s = if rust.mapping_type() == MappingType::Surround && s.bitrate.is_none() {
        Settings { bitrate: Some(32_000), ..s }
    } else {
        s
    };
    rust.set_bitrate(s.bitrate_for(channels));
    rust.set_vbr(s.vbr);
    rust.set_complexity(s.complexity).unwrap();
    c.set(sys::OPUS_SET_BITRATE_REQUEST, s.c_bitrate_for(channels));
    c.set(sys::OPUS_SET_VBR_REQUEST, s.vbr as i32);
    c.set(sys::OPUS_SET_COMPLEXITY_REQUEST, i32::from(s.complexity));

    let tone: Vec<f32> = (0..channels).map(|ch| if Some(ch) == lfe { LFE_TONE } else { TONES[ch] }).collect();
    // The input channel each output channel carries: the first one that
    // the mapping sends to the same coded channel.
    let mapping = layout.mapping();
    let source: Vec<Option<usize>> = mapping
        .iter()
        .map(|&m| (m != 255).then(|| mapping.iter().position(|&n| n == m)).flatten())
        .collect();
    let frames = ((SECONDS * s.fs as f32) as usize).div_ceil(s.frame_size);
    let pcm = tones(&tone, s.fs, s.frame_size, frames);
    let (streams, coupled) = (layout.streams(), layout.coupled_streams());
    let mut rust_dec = CMsDecoder::new(s.fs, channels, streams, coupled, mapping).unwrap();
    let mut c_dec = CMsDecoder::new(s.fs, channels, streams, coupled, mapping).unwrap();
    let mut packet = vec![0u8; 4000 * streams];
    let mut pcm_out = vec![0f32; s.frame_size * channels];
    let (mut rust_out, mut c_out) = (Vec::new(), Vec::new());
    // The streams that SILK coded, and those coded as mono, in a packet of
    // either encoder.
    let mut silk = vec![false; streams];
    let mut mono = vec![false; streams];
    let mut scratch = Vec::new();
    for (n, frame) in pcm.chunks_exact(s.frame_size * channels).enumerate() {
        let len = rust.encode(frame, s.frame_size, &mut packet).unwrap_or_else(|e| panic!("frame {n}: {e}"));
        for (k, toc) in stream_tocs(&packet[..len], streams, &mut scratch).into_iter().enumerate() {
            silk[k] |= silk_coded(toc);
            mono[k] |= mono_coded(toc);
        }
        let got = rust_dec.decode_float(&packet[..len], &mut pcm_out, s.frame_size, false);
        assert_eq!(got, s.frame_size as i32, "frame {n}: libopus decodes opus1's packet");
        rust_out.extend_from_slice(&pcm_out);
        let len = c.encode(frame, s.frame_size, &mut packet);
        assert!(len > 0, "frame {n}: libopus encode {len}");
        for (k, toc) in stream_tocs(&packet[..len as usize], streams, &mut scratch).into_iter().enumerate() {
            silk[k] |= silk_coded(toc);
            mono[k] |= mono_coded(toc);
        }
        let got = c_dec.decode_float(&packet[..len as usize], &mut pcm_out, s.frame_size, false);
        assert_eq!(got, s.frame_size as i32, "frame {n}: libopus decodes its packet");
        c_out.extend_from_slice(&pcm_out);
    }
    // Coded channels 2s and 2s + 1 belong to stereo stream s, coded channel
    // coupled + s to mono stream s.
    let skip: Vec<bool> = mapping
        .iter()
        .map(|&m| {
            let m = m as usize;
            m != 255 && m < 2 * coupled && (mono[m / 2] || (silk[m / 2] && !s.vbr))
        })
        .collect();
    let what = format!("multistream {layout:?}, LFE channel {lfe:?}, {s:?}");
    compare(
        &what,
        &leakages(&rust_out, channels, &tone, &source, s.fs),
        &leakages(&c_out, channels, &tone, &source, s.fs),
        &skip,
    );
    Ok(())
}
