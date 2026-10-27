//! Discontinuous transmission (DTX) in `MultistreamEncoder` against libopus
//! (`OPUS_SET_DTX` on `opus_multistream_encoder_ctl`, which libopus passes
//! to every stream encoder).
//!
//! Each stream encoder decides DTX on its own. A stream it leaves out is
//! its TOC byte, with a length byte of 0 when it is not the last stream.
//! What is checked:
//!
//! 1. `set_dtx` reaches every stream, and `dtx` reads the first one.
//! 2. With every stream CELT-only, the multistream packets are libopus's
//!    byte for byte, for every layout of `multistream_enc_common::CASES`,
//!    below complexity 7 (only digital silence is left out) and from 7 on
//!    (the analysis of each stream decides).
//! 3. With the `Audio` application and the mode left to each stream, the
//!    frames each stream leaves out are libopus's. The SILK frames differ
//!    in their bytes (opus1's SILK is the fixed-point one), but DTX decides
//!    on inputs that are the same (see `tests/core_dtx_libopus.rs`).
//! 4. The opus1 decoder and the libopus decoder give the same samples for
//!    opus1's multistream DTX packets.
#![cfg(all(feature = "std", feature = "analysis"))]

mod k24_common;
mod ms_support;
mod multistream_enc_common;

use std::num::NonZeroU32;
use ms_support::*;
use multistream_enc_common::*;
use opus1::multistream::{MultistreamDecoder, MultistreamEncoder};
use opus1::{Application, ForcedMode, SampleRate};

/// Whether the packets must be libopus's: the analysis rounds as the
/// libopus the tests link only on 64-bit ARM.
const EXACT: bool = cfg!(target_arch = "aarch64");

/// libopus `OPUS_SET_DTX` and `OPUS_GET_DTX`.
const OPUS_SET_DTX_REQUEST: i32 = 4016;
const OPUS_GET_DTX_REQUEST: i32 = 4017;

/// White noise of amplitude `level`.
fn noise(n: usize, level: f32, seed: u32) -> Vec<f32> {
    let mut seed = seed;
    (0..n)
        .map(|_| {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            level * ((seed >> 8) as f32 / (1u32 << 23) as f32 - 1.0)
        })
        .collect()
}

/// `channels` channels, interleaved: speech in the even channels and music
/// in the odd ones, each delayed and scaled differently, with pauses at the
/// same time in every channel: digital silence, then quiet noise.
fn signal(fs: i32, channels: usize) -> Vec<f32> {
    let fs_u = fs as usize;
    let n = |s: f64| (s * fs as f64) as usize;
    let speech = k24_common::speech_fullband(fs_u, 2.7);
    let music = k24_common::multitone(fs_u, n(2.7));
    let len = n(1.2) + n(0.8) + n(0.6) + n(1.0) + n(0.8);
    let mut pcm = vec![0.0f32; len * channels];
    for c in 0..channels {
        let delay = 37 * c;
        let gain = 1.0 - 0.08 * c as f32;
        let src = if c % 2 == 0 { &speech } else { &music };
        let quiet = noise(n(1.0), 5e-4, 3 + c as u32);
        let mut x = Vec::with_capacity(len);
        x.extend(src[delay..delay + n(1.2)].iter().map(|v| gain * v));
        x.extend(std::iter::repeat_n(0.0, n(0.8)));
        x.extend(src[delay + n(1.2)..delay + n(1.8)].iter().map(|v| gain * v));
        x.extend(quiet);
        x.extend(src[delay + n(1.8)..delay + n(2.6)].iter().map(|v| gain * v));
        for (i, v) in x.iter().enumerate() {
            pcm[i * channels + c] = *v;
        }
    }
    pcm
}

/// The packets of both encoders for `pcm`, `frame_size` samples per channel
/// per call.
fn encode_both(
    rust: &mut MultistreamEncoder,
    c: &mut CMsEncoder,
    pcm: &[f32],
    frame_size: usize,
) -> (Vec<Vec<u8>>, Vec<Vec<u8>>) {
    let channels = rust.layout().channels();
    let max_bytes = max_packet_bytes(rust.layout().streams());
    let step = frame_size * channels;
    let mut out = vec![0u8; max_bytes];
    let (mut rs, mut cp) = (Vec::new(), Vec::new());
    for k in 0..pcm.len() / step {
        let range = k * step..(k + 1) * step;
        cp.push(c.encode_float(&pcm[range.clone()], frame_size, max_bytes).unwrap());
        let n = rust.encode(&pcm[range], frame_size, &mut out).unwrap();
        rs.push(out[..n].to_vec());
    }
    (rs, cp)
}

/// For each packet and stream: which frames DTX left out.
fn dtx_frames(packets: &[Vec<u8>], streams: usize) -> Vec<Vec<bool>> {
    packets
        .iter()
        .flat_map(|p| {
            split_streams(p, streams)
                .unwrap()
                .into_iter()
                .map(|s| s.frame_sizes.iter().map(|&n| n == 0).collect::<Vec<_>>())
        })
        .collect()
}

fn count(frames: &[Vec<bool>]) -> usize {
    frames.iter().flatten().filter(|&&b| b).count()
}

/// Both encoders of `case` with DTX on.
fn encoders_with_dtx(case: &Case, fs: i32, application: Application) -> (MultistreamEncoder, CMsEncoder) {
    let (mut rust, mut c) = encoders(case, fs, application);
    rust.set_dtx(true);
    c.ctl(OPUS_SET_DTX_REQUEST, 1);
    (rust, c)
}

#[test]
fn set_dtx_reaches_every_stream() {
    let case = Case::family("5.1", 6, 1);
    let (mut rust, mut c) = encoders(&case, 48_000, Application::Audio);
    assert!(!rust.dtx());
    rust.set_dtx(true);
    c.ctl(OPUS_SET_DTX_REQUEST, 1);
    assert!(rust.dtx());
    for s in 0..rust.layout().streams() {
        assert!(rust.stream_encoder(s).unwrap().dtx(), "stream {s}");
        assert_eq!(c.stream_get(s, OPUS_GET_DTX_REQUEST), 1, "stream {s} of libopus");
    }
    rust.reset_state();
    assert!(rust.dtx());
    rust.set_dtx(false);
    assert!((0..rust.layout().streams()).all(|s| !rust.stream_encoder(s).unwrap().dtx()));
}

/// Every stream CELT-only: the multistream packets are libopus's.
#[test]
fn celt_only_packets_are_libopus() {
    let (mut streams, mut packets, mut dtx, mut failures) = (0, 0, 0, Vec::new());
    for fs in [16_000, 48_000] {
        for case in CASES {
            let pcm = signal(fs, case.channels);
            for (per_channel, vbr, complexity) in [(24_000, true, 5), (48_000, false, 9), (32_000, true, 10)] {
                let (mut rust, mut c) = encoders_with_dtx(case, fs, Application::Audio);
                configure(&mut rust, &mut c, per_channel * case.channels as i32, vbr);
                rust.set_complexity(complexity).unwrap();
                c.set_complexity(i32::from(complexity));
                rust.set_force_mode(Some(ForcedMode::CeltOnly));
                c.ctl(OPUS_SET_FORCE_MODE_REQUEST, MODE_CELT_ONLY);
                let (rs, cp) = encode_both(&mut rust, &mut c, &pcm, fs as usize / 50);
                streams += 1;
                packets += rs.len();
                dtx += count(&dtx_frames(&cp, rust.layout().streams()));
                if let Some(k) = rs.iter().zip(&cp).position(|(a, b)| a != b) {
                    failures.push(format!(
                        "{} {fs} Hz {per_channel} b/s per channel vbr {vbr} c{complexity}: packet {k}",
                        case.name
                    ));
                }
            }
        }
    }
    println!("{streams} streams, {packets} packets, {dtx} stream frames left out");
    for f in failures.iter().take(20) {
        println!("  {f}");
    }
    assert!(dtx > 1000, "too few frames left out: {dtx}");
    if EXACT {
        assert!(failures.is_empty(), "{} streams are not libopus's", failures.len());
    }
}

/// Each stream picks its mode (the coupled streams of a surround layout are
/// CELT-only, as libopus forces them): the frames each stream leaves out
/// are libopus's.
#[test]
fn each_stream_leaves_out_the_frames_of_libopus() {
    let (mut runs, mut dtx, mut failures) = (0, 0, Vec::new());
    for fs in [16_000, 24_000, 48_000] {
        for case in CASES {
            let pcm = signal(fs, case.channels);
            for (per_channel, vbr, complexity, frame_ms) in
                [(12_000, true, 4, 20), (20_000, false, 9, 20), (16_000, false, 6, 60), (32_000, true, 8, 10)]
            {
                let (mut rust, mut c) = encoders_with_dtx(case, fs, Application::Audio);
                configure(&mut rust, &mut c, per_channel * case.channels as i32, vbr);
                rust.set_complexity(complexity).unwrap();
                c.set_complexity(i32::from(complexity));
                let frame_size = fs as usize * frame_ms / 1000;
                let (rs, cp) = encode_both(&mut rust, &mut c, &pcm, frame_size);
                let streams = rust.layout().streams();
                let (rd, cd) = (dtx_frames(&rs, streams), dtx_frames(&cp, streams));
                runs += 1;
                dtx += count(&cd);
                // With CBR the packets keep libopus's size too.
                let sizes_differ = !vbr && rs.iter().zip(&cp).any(|(a, b)| a.len() != b.len());
                if rd != cd || sizes_differ {
                    let k = rd.iter().zip(&cd).position(|(a, b)| a != b).map(|i| (i / streams, i % streams));
                    failures.push(format!(
                        "{} {fs} Hz {frame_ms} ms {per_channel} b/s per channel vbr {vbr} c{complexity}: \
                         frames left out {} / {} (libopus / opus1), first other (packet, stream) {k:?}, \
                         sizes differ {sizes_differ}",
                        case.name,
                        count(&cd),
                        count(&rd)
                    ));
                }
            }
        }
    }
    println!("{runs} runs, {dtx} stream frames left out");
    for f in failures.iter().take(20) {
        println!("  {f}");
    }
    assert!(dtx > 1000, "too few frames left out: {dtx}");
    if EXACT {
        assert!(failures.is_empty(), "{} of {runs} runs leave out other frames", failures.len());
    }
}

/// opus1's multistream DTX packets decode to the same samples in the opus1
/// decoder and in libopus's.
///
/// Every packet decodes bit for bit, and so does every stream frame DTX
/// left out. Before the cross-fade of the decoder (`smooth_fade`) fused one
/// of its products as clang does, a few Hybrid frames that announce a
/// switch to CELT-only decoded with a difference in the last bit of some
/// samples of their last 2.5 ms (8 of 960 samples of packet 27 of the mono
/// layout, Audio, 20 ms, complexity 9).
#[test]
#[cfg_attr(not(target_arch = "aarch64"), ignore = "bit-exact with libopus only on 64-bit ARM (docs/guide/10, Platforms)")]
fn opus1_dtx_packets_decode_like_libopus() {
    let (mut packets_checked, mut dtx, mut dtx_checked, mut not_identical) = (0, 0, 0, 0);
    let mut failures = Vec::new();
    let fs = 48_000;
    for case in CASES {
        let pcm = signal(fs, case.channels);
        for (application, frame_ms, complexity) in
            [(Application::Audio, 20, 9), (Application::Voip, 20, 5), (Application::Audio, 60, 3)]
        {
            let label = format!("{} {application:?} {frame_ms} ms c{complexity}", case.name);
            let mut rust = rust_encoder(case, fs, application);
            rust.set_dtx(true);
            rust.set_complexity(complexity).unwrap();
            rust.set_bitrate(opus1::Bitrate::BitsPerSecond(NonZeroU32::new(20_000 * case.channels as u32).unwrap()));
            let frame_size = fs as usize * frame_ms / 1000;
            let packets = rust_encode_all(&mut rust, &pcm, frame_size);
            let layout = rust.layout();
            let (channels, streams, coupled) = (layout.channels(), layout.streams(), layout.coupled_streams());
            // The stream each output channel comes from, if any.
            let stream_of: Vec<Option<usize>> = layout
                .mapping()
                .iter()
                .map(|&m| match m as usize {
                    255 => None,
                    m if m < 2 * coupled => Some(m / 2),
                    m => Some(m - coupled),
                })
                .collect();
            let mut r_dec = MultistreamDecoder::new(
                SampleRate::Hz48000,
                channels,
                streams,
                coupled,
                layout.mapping(),
            )
            .unwrap();
            let mut c_dec = c_decoder(&rust);
            let max = 5760;
            let mut out = vec![0f32; max * channels];
            // Per stream: its channels decoded to the same samples so far.
            let mut in_step = vec![true; streams];
            for (k, p) in packets.iter().enumerate() {
                let n = r_dec.decode(p, max, &mut out).unwrap();
                let got = &out[..n * channels];
                let want = c_dec.decode_float(Some(p), max, false).unwrap();
                let left_out = dtx_frames(std::slice::from_ref(p), streams);
                dtx += count(&left_out);
                packets_checked += 1;
                not_identical += (got != &want[..]) as usize;
                // Per stream and frame: whether its channels are the same.
                let frames = left_out[0].len();
                let mut same = vec![vec![true; frames]; streams];
                for (c, stream) in stream_of.iter().enumerate() {
                    let Some(stream) = *stream else { continue };
                    for (f, same) in same[stream].iter_mut().enumerate() {
                        let span = f * n / frames..(f + 1) * n / frames;
                        *same &= span.into_iter().all(|i| got[i * channels + c] == want[i * channels + c]);
                    }
                }
                for s in 0..streams {
                    for f in 0..frames {
                        if in_step[s] && left_out[s][f] {
                            dtx_checked += 1;
                            if !same[s][f] {
                                failures.push(format!("{label}: packet {k}, stream {s}, frame {f} left out decodes otherwise"));
                            }
                        }
                        in_step[s] &= same[s][f];
                    }
                }
                if got != &want[..] {
                    let agreement = snr(&want, got);
                    failures.push(format!("{label}: packet {k} decodes otherwise, {agreement:.1} dB"));
                }
            }
        }
    }
    println!(
        "{packets_checked} packets, {dtx} stream frames left out, {dtx_checked} of them checked and identical, \
         {not_identical} packets not identical"
    );
    for f in &failures {
        println!("  {f}");
    }
    assert!(dtx_checked > 2000, "too few frames left out: {dtx}, {dtx_checked} checked");
    assert!(failures.is_empty(), "{} packets decode otherwise", failures.len());
}

/// Signal-to-noise ratio of `test` against `reference` in dB. 200 means
/// identical.
fn snr(reference: &[f32], test: &[f32]) -> f64 {
    let (mut sig, mut err) = (0f64, 0f64);
    for (a, b) in reference.iter().zip(test) {
        sig += (*a as f64).powi(2);
        err += ((*a - *b) as f64).powi(2);
    }
    if err <= 0.0 { 200.0 } else { (10.0 * (sig / err).log10()).min(200.0) }
}
