//! The CELT encoder with the speech and music analysis of the Opus layer
//! (`CeltEncoder::set_analysis`) against libopus's CELT encoder with
//! `CELT_SET_ANALYSIS`.
//!
//! Both encoders get the same samples, the same settings and, before every
//! frame, the same `AnalysisInfo`. On 64-bit ARM every packet must then be
//! the packet of libopus, byte for byte. A valid analysis changes five
//! decisions of the encoder (celt_encoder.c of libopus 1.6.1):
//!
//! 1. the allocation trim (`tonality_slope`, C 935-939);
//! 2. the dynamic allocation (`leak_boost`, C 1223-1231);
//! 3. the gain of the pitch pre-filter (`max_pitch_ratio`, C 1492-1495);
//! 4. the VBR target (`activity`, `tonality` and the pitch change,
//!    C 1630-1633, 1655-1673, 2042-2044);
//! 5. the highest band the allocation may skip to (`bandwidth`,
//!    C 2603-2624).
//!
//! The analyses come from two places: the analysis of opus1 run on the same
//! signal (what the Opus layer hands over), and random values that also
//! reach the limits of each field.
#![cfg(feature = "analysis")]

mod celtenc_common;

use celtenc_common::c_celt::{OPUS_SET_LFE, Stream};
use celtenc_common::*;
use opus1::internals::analysis::{
    AnalysisInfo, AnalysisPcm, LEAK_BANDS, PcmRef, SecondChannel, TonalityAnalysisState,
};

/// `seconds` of speech at `fs` Hz (16, 24 or 48 kHz), mono.
fn speech_at(fs: usize, seconds: usize) -> Vec<f32> {
    match fs {
        24_000 => speech(48_000, seconds).iter().step_by(2).copied().collect(),
        _ => speech(fs, seconds),
    }
}

/// A sine of `freq` Hz: a pure tone, which takes the tone path of the
/// pre-filter.
fn tone(fs: usize, seconds: usize, freq: f64) -> Vec<f32> {
    (0..fs * seconds)
        .map(|i| (0.3 * (std::f64::consts::TAU * freq * i as f64 / fs as f64).sin()) as f32)
        .collect()
}

/// The test signals for `channels` input channels at `fs` Hz, interleaved.
fn signals(fs: usize, channels: usize, seconds: usize) -> Vec<(&'static str, Vec<f32>)> {
    let s = speech_at(fs, seconds);
    let m0 = music(fs, seconds, 0);
    let t = tone(fs, seconds, 440.0);
    if channels == 1 {
        return vec![("speech", s), ("music", m0), ("tone", t)];
    }
    let m1 = music(fs, seconds, 1);
    vec![
        ("speech", stereo_of(&s, &m1, 0.1)),
        ("music", stereo_of(&m0, &m1, 0.3)),
        ("tone", stereo_of(&t, &t, 0.0)),
    ]
}

/// The analysis of opus1 for each frame of `pcm`, as the Opus layer runs
/// it: the whole frame of all channels, at 24-bit depth.
fn analysis_of(st: &Stream, pcm: &[f32]) -> Vec<AnalysisInfo> {
    let mut state = TonalityAnalysisState::new(st.fs as i32);
    pcm.chunks_exact(st.frame * st.channels)
        .map(|frame| {
            let input = AnalysisPcm {
                pcm: PcmRef::F32(frame),
                channels: st.channels,
                c1: 0,
                c2: SecondChannel::AllOthers,
                frame_size: st.frame,
            };
            state.run(input, st.frame, 24)
        })
        .collect()
}

/// A small random number generator (64-bit linear congruential).
struct Lcg(u64);

impl Lcg {
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }

    /// Uniform in [0, 1).
    fn unit(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32
    }

    fn below(&mut self, n: u32) -> u32 {
        self.next_u32() % n
    }

    /// A value in [lo, hi], with each end taken one time in eight.
    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        match self.below(8) {
            0 => lo,
            1 => hi,
            _ => lo + (hi - lo) * self.unit(),
        }
    }
}

/// A random analysis: valid in 7 of 8 frames, every field in its range or
/// at one of its ends.
fn random_info(rng: &mut Lcg) -> AnalysisInfo {
    let mut leak_boost = [0u8; LEAK_BANDS];
    for v in &mut leak_boost {
        *v = match rng.below(4) {
            0 => 0,
            1 => 255,
            _ => rng.below(256) as u8,
        };
    }
    AnalysisInfo {
        valid: rng.below(8) != 0,
        tonality: rng.range(0.0, 1.0),
        tonality_slope: rng.range(-1.0, 1.0),
        noisiness: rng.range(0.0, 1.0),
        activity: rng.range(0.0, 1.0),
        music_prob: rng.range(0.0, 1.0),
        music_prob_min: rng.range(0.0, 1.0),
        music_prob_max: rng.range(0.0, 1.0),
        bandwidth: rng.below(21) as i32,
        activity_probability: rng.range(0.0, 1.0),
        max_pitch_ratio: rng.range(0.0, 1.0),
        leak_boost,
    }
}

/// Codes `pcm` with both encoders, handing both `infos[k]` before frame
/// `k`. Returns the payloads of libopus and of opus1.
fn encode_both(st: &Stream, pcm: &[f32], infos: &[AnalysisInfo]) -> (Vec<Vec<u8>>, Vec<Vec<u8>>) {
    st.enc_both_with(pcm, |k, c_enc, rs_enc| {
        c_enc.set_analysis(&infos[k]);
        rs_enc.set_analysis(&infos[k]);
    })
}

/// The label of a stream in a failure message.
fn label(st: &Stream, signal: &str) -> String {
    format!(
        "{} kHz, {} ch coded as {}, {}/2 ms, {} b/s, c{}, {}, start band {}, {signal}",
        st.fs / 1000,
        st.channels,
        st.stream_channels,
        st.frame * 2000 / st.fs,
        st.bitrate,
        st.complexity,
        match (st.vbr, st.constrained_vbr) {
            (false, _) => "CBR",
            (true, false) => "VBR",
            (true, true) => "CVBR",
        },
        st.start_band,
    )
}

/// The rows of a run that are not the packets of libopus, with the first
/// frame that differs.
#[derive(Default)]
struct Tally {
    streams: usize,
    packets: usize,
    failures: Vec<String>,
}

impl Tally {
    fn check(&mut self, what: String, c: &[Vec<u8>], rs: &[Vec<u8>]) {
        self.streams += 1;
        self.packets += c.len();
        if let Some(k) = c.iter().zip(rs).position(|(a, b)| a != b) {
            let same = Stream::same_frames(c, rs);
            self.failures.push(format!("{what}: frame {k} differs, {same} of {} same", c.len()));
        }
    }

    fn finish(self, name: &str) {
        println!("{name}: {} streams, {} packets", self.streams, self.packets);
        for f in self.failures.iter().take(20) {
            println!("  {f}");
        }
        // Elsewhere the crate may round differently from the libopus the
        // tests link (see `celtenc_gap_report`), so only 64-bit ARM must
        // match.
        if cfg!(target_arch = "aarch64") {
            assert!(
                self.failures.is_empty(),
                "{} of {} streams differ from libopus",
                self.failures.len(),
                self.streams
            );
        }
    }
}

/// The bitrates per channel of the matrix for a frame duration: from what
/// a 3-byte packet carries up to 255 kb/s per channel, the most CELT takes
/// of 510 kb/s in stereo.
fn bitrates(frame_ms_x2: usize) -> &'static [i32] {
    match frame_ms_x2 {
        5 => &[16_000, 32_000, 64_000, 128_000, 255_000],
        10 => &[8_000, 24_000, 48_000, 96_000, 255_000],
        _ => &[6_000, 12_000, 24_000, 48_000, 96_000, 255_000],
    }
}

/// With the analysis of opus1 on the same signal, every packet is
/// libopus's: 16, 24 and 48 kHz; mono, stereo and stereo coded as mono;
/// 2.5 to 20 ms; CBR, VBR and constrained VBR; complexity 0 to 10;
/// speech, music and a tone; 6 to 255 kb/s per channel.
#[test]
fn analysis_of_the_signal_gives_the_packets_of_libopus() {
    let mut tally = Tally::default();
    let mut pick = Lcg(7);
    for fs in [16_000usize, 24_000, 48_000] {
        for (channels, stream_channels) in [(1usize, 1usize), (2, 2), (2, 1)] {
            let sigs = signals(fs, channels, 2);
            for frame_ms_x2 in [5usize, 10, 20, 40] {
                for &per_channel in bitrates(frame_ms_x2) {
                    for (vbr, constrained) in [(false, false), (true, false), (true, true)] {
                        // Two complexities per row keep the run short; over
                        // the rows every complexity comes up many times.
                        for _ in 0..2 {
                            let complexity = pick.below(11) as i32;
                            let (name, pcm) = &sigs[pick.below(sigs.len() as u32) as usize];
                            let mut st = Stream::new(
                                fs,
                                channels,
                                frame_ms_x2,
                                per_channel * stream_channels as i32,
                            );
                            st.stream_channels = stream_channels;
                            st.complexity = complexity;
                            st.vbr = vbr;
                            st.constrained_vbr = constrained;
                            // Shorter frames need fewer seconds for as many
                            // frames.
                            let seconds = if frame_ms_x2 <= 10 { 1 } else { 2 };
                            let pcm = &pcm[..fs * channels * seconds];
                            let infos = analysis_of(&st, pcm);
                            let (c, rs) = encode_both(&st, pcm, &infos);
                            tally.check(label(&st, name), &c, &rs);
                        }
                    }
                }
            }
        }
    }
    tally.finish("analysis of the signal");
}

/// With random analyses, which reach the ends of every field, every packet
/// is libopus's. Also 8 and 12 kHz input, where the Opus layer runs no
/// analysis but the CELT encoder takes one.
#[test]
fn random_analyses_give_the_packets_of_libopus() {
    let mut tally = Tally::default();
    let mut rng = Lcg(99);
    for fs in [8_000usize, 12_000, 16_000, 24_000, 48_000] {
        for (channels, stream_channels) in [(1usize, 1usize), (2, 2), (2, 1)] {
            let sigs = if fs >= 16_000 {
                signals(fs, channels, 1)
            } else {
                // The speech exists at 16 and 48 kHz only.
                let (m, t) = (music(fs, 1, 0), tone(fs, 1, 440.0));
                if channels == 1 {
                    vec![("music", m), ("tone", t)]
                } else {
                    vec![("music", stereo_of(&m, &m, 0.0)), ("tone", stereo_of(&t, &t, 0.0))]
                }
            };
            for frame_ms_x2 in [5usize, 10, 20, 40] {
                for &per_channel in bitrates(frame_ms_x2) {
                    for (vbr, constrained) in [(false, false), (true, false), (true, true)] {
                        let complexity = rng.below(11) as i32;
                        let (name, pcm) = &sigs[rng.below(sigs.len() as u32) as usize];
                        let mut st = Stream::new(
                            fs,
                            channels,
                            frame_ms_x2,
                            per_channel * stream_channels as i32,
                        );
                        st.stream_channels = stream_channels;
                        st.complexity = complexity;
                        st.vbr = vbr;
                        st.constrained_vbr = constrained;
                        let frames = pcm.len() / (st.frame * channels);
                        let infos: Vec<AnalysisInfo> =
                            (0..frames).map(|_| random_info(&mut rng)).collect();
                        let (c, rs) = encode_both(&st, pcm, &infos);
                        tally.check(label(&st, name), &c, &rs);
                    }
                }
            }
        }
    }
    tally.finish("random analyses");
}

/// The other places where the analysis meets the encoder's state:
///
/// - a reset in the middle of a stream makes the analysis invalid until
///   the next `set_analysis`;
/// - the CELT part of Hybrid frames (start band 17), which reads the trim
///   and the bandwidth but not the VBR terms;
/// - the low-frequency-effects mode, which keeps its own rules;
/// - prediction off (no pre-filter, intra energy).
#[test]
fn reset_hybrid_lfe_and_prediction() {
    let mut tally = Tally::default();
    let mut rng = Lcg(5);
    let fs = 48_000;
    for channels in [1usize, 2] {
        let sigs = signals(fs, channels, 2);
        for (name, pcm) in &sigs {
            for complexity in [5, 7, 9, 10] {
                for vbr in [false, true] {
                    // A reset after frame 30, with the analysis set before
                    // it, and one with no analysis set after it for a frame.
                    let mut st = Stream::new(fs, channels, 20, 64_000 * channels as i32);
                    st.complexity = complexity;
                    st.vbr = vbr;
                    let infos = analysis_of(&st, pcm);
                    let (c, rs) = st.enc_both_with(pcm, |k, c_enc, rs_enc| {
                        if k != 31 {
                            c_enc.set_analysis(&infos[k]);
                            rs_enc.set_analysis(&infos[k]);
                        }
                        if k == 30 || k == 31 {
                            c_enc.reset();
                            rs_enc.reset_state();
                        }
                    });
                    tally.check(format!("reset, {}", label(&st, name)), &c, &rs);

                    // Hybrid: CELT codes bands 17 to 20.
                    let mut st = Stream::new(fs, channels, 20, 20_000 * channels as i32);
                    st.complexity = complexity;
                    st.vbr = vbr;
                    st.constrained_vbr = false;
                    st.start_band = 17;
                    let infos: Vec<AnalysisInfo> =
                        (0..pcm.len() / (st.frame * channels)).map(|_| random_info(&mut rng)).collect();
                    let (c, rs) = encode_both(&st, pcm, &infos);
                    tally.check(format!("hybrid, {}", label(&st, name)), &c, &rs);

                    // Prediction off.
                    let mut st = Stream::new(fs, channels, 20, 48_000 * channels as i32);
                    st.complexity = complexity;
                    st.vbr = vbr;
                    st.prediction = 0;
                    let infos = analysis_of(&st, pcm);
                    let (c, rs) = encode_both(&st, pcm, &infos);
                    tally.check(format!("prediction 0, {}", label(&st, name)), &c, &rs);
                }
            }
        }
        // LFE: mono, narrowband.
        if channels == 1 {
            for (name, pcm) in &sigs {
                for vbr in [false, true] {
                    let mut st = Stream::new(fs, 1, 20, 24_000);
                    st.complexity = 9;
                    st.vbr = vbr;
                    st.end_band = 13;
                    let infos: Vec<AnalysisInfo> =
                        (0..pcm.len() / st.frame).map(|_| random_info(&mut rng)).collect();
                    let (c, rs) = st.enc_both_with(pcm, |k, c_enc, rs_enc| {
                        if k == 0 {
                            c_enc.ctl(OPUS_SET_LFE, 1);
                            rs_enc.set_lfe(true);
                        }
                        c_enc.set_analysis(&infos[k]);
                        rs_enc.set_analysis(&infos[k]);
                    });
                    tally.check(format!("LFE, {}", label(&st, name)), &c, &rs);
                }
            }
        }
    }
    tally.finish("reset, Hybrid, LFE and prediction");
}

/// Each field the encoder reads changes packets on its own. Without this,
/// the tests above could pass with a term that does nothing on both sides.
///
/// The neutral analysis below changes no decision, so its packets are those
/// of an invalid analysis. For each field, a stream where only that field
/// moves away from the neutral value must give other packets, and they
/// must be libopus's.
#[test]
fn each_field_changes_the_packets() {
    let fs = 48_000;
    let neutral = AnalysisInfo {
        valid: true,
        tonality: 0.27,
        tonality_slope: -0.05,
        activity: 1.0,
        bandwidth: 20,
        max_pitch_ratio: 1.0,
        ..AnalysisInfo::default()
    };
    type Change = fn(&mut AnalysisInfo);
    let changes: [(&str, Change); 6] = [
        ("tonality_slope", |a| a.tonality_slope = 0.9),
        ("leak_boost", |a| a.leak_boost = [200; LEAK_BANDS]),
        ("max_pitch_ratio", |a| a.max_pitch_ratio = 0.3),
        ("activity", |a| a.activity = 0.0),
        ("tonality", |a| a.tonality = 1.0),
        ("bandwidth", |a| a.bandwidth = 8),
    ];
    let pcm = speech(fs, 2);
    let mut st = Stream::new(fs, 1, 20, 32_000);
    st.complexity = 9;
    st.vbr = true;
    let frames = pcm.len() / st.frame;
    let (c0, rs0) = encode_both(&st, &pcm, &vec![neutral; frames]);
    let invalid = AnalysisInfo { valid: false, ..neutral };
    let (c_invalid, _) = encode_both(&st, &pcm, &vec![invalid; frames]);
    assert_eq!(c0, c_invalid, "the neutral analysis changes packets of libopus");
    if cfg!(target_arch = "aarch64") {
        assert_eq!(c0, rs0, "neutral analysis");
    }
    for (field, change) in changes {
        let mut info = neutral;
        change(&mut info);
        let (c, rs) = encode_both(&st, &pcm, &vec![info; frames]);
        let moved = Stream::same_frames(&c, &c0);
        println!("{field}: {} of {frames} packets change", frames - moved);
        assert!(moved < frames, "{field} changes no packet of libopus");
        if cfg!(target_arch = "aarch64") {
            assert_eq!(c, rs, "{field}");
        }
    }
}
