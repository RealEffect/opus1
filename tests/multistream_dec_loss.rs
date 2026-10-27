//! `MultistreamDecoder` on a stream with lost packets, against the C libopus
//! multistream decoder.
//!
//! What is checked:
//!
//! - loss concealment (an empty packet): every 10th packet lost, and bursts
//!   of 3 and 6 packets. The concealed signal follows the C decoder's in
//!   level and from sample to sample, and the stream recovers afterwards;
//! - lost durations that are not the frame size of the stream, and loss
//!   before the first packet;
//! - forward error correction (FEC): the lost frame is rebuilt from the
//!   packet after it, as `opus_multistream_decode_float` does with
//!   `decode_fec = 1`;
//! - the multistream decoder gives exactly what one `OpusDecoder` per stream
//!   gives, for decoded, concealed and rebuilt frames and for 16-bit output.
//!
//! How close the comparison with C can be depends on the stream. A concealed
//! SILK frame is the same in both decoders, sample for sample. A concealed
//! CELT frame repeats the last pitch period of the signal, and it is not
//! bit-exact with libopus: the difference is 25 to 100 dB below the signal
//! on speech-like input.
// The multistream API needs the `std` feature.
#![cfg(feature = "std")]

mod ms_support;
mod multistream_dec_common;

use ms_support::*;
use multistream_dec_common::*;
use opus1::{Channels, OpusDecoder, SampleRate, pcm_soft_clip};

/// Which packets of a stream of `count` packets are lost: every 10th, a
/// burst of 3 and a burst of 6. Nothing is lost in the last `tail` packets.
fn is_lost(i: usize, count: usize, tail: usize) -> bool {
    i + tail < count && (i % 10 == 9 || (23..26).contains(&i) || (41..47).contains(&i))
}

/// What one run through a stream with losses measured, as the worst value
/// over all channels and all lost packets.
#[derive(Debug, Clone, Copy)]
struct LossResult {
    /// Largest difference in level between a concealed frame and the C
    /// decoder's, in dB, over the first two lost packets of every burst.
    level_db: f64,
    /// The same over the later packets of every burst.
    deep_level_db: f64,
    /// Largest step from one sample to the next in and right after a
    /// concealed frame, divided by the C decoder's largest step there, over
    /// the first two lost packets of every burst.
    step_ratio: f64,
    /// The same over the later packets of every burst.
    deep_step_ratio: f64,
    /// Smallest SNR against the C decoder over the end of the signal, where
    /// nothing is lost, in dB.
    tail_snr: f64,
    /// Smallest SNR against the C decoder over the concealed frames, in dB.
    lost_snr: f64,
}

impl LossResult {
    const BEST: Self = Self {
        level_db: 0.0,
        deep_level_db: 0.0,
        step_ratio: 0.0,
        deep_step_ratio: 0.0,
        tail_snr: f64::INFINITY,
        lost_snr: f64::INFINITY,
    };

    fn worst_of(self, other: Self) -> Self {
        Self {
            level_db: self.level_db.max(other.level_db),
            deep_level_db: self.deep_level_db.max(other.deep_level_db),
            step_ratio: self.step_ratio.max(other.step_ratio),
            deep_step_ratio: self.deep_step_ratio.max(other.deep_step_ratio),
            tail_snr: self.tail_snr.min(other.tail_snr),
            lost_snr: self.lost_snr.min(other.lost_snr),
        }
    }
}

/// Largest step between neighbouring samples of `x[from..to]`, counting the
/// step into `x[from]`.
fn largest_step(x: &[f32], from: usize, to: usize) -> f64 {
    (from.max(1)..to.min(x.len())).map(|i| (x[i] - x[i - 1]).abs() as f64).fold(0.0, f64::max)
}

/// Difference in level of two signals in dB. A level below -80 dB relative
/// to full scale counts as silence.
fn level_difference_db(reference: &[f32], test: &[f32]) -> f64 {
    let floor = reference.len() as f64 * 1e-8;
    (10.0 * ((energy(test) + floor) / (energy(reference) + floor)).log10()).abs()
}

/// Decodes `e` at `fs` with both decoders, with the packets of `is_lost`
/// concealed, and compares the outputs. `frame` is the frame size of the
/// encoder at 48 kHz.
fn run_with_losses(e: &Encoded, fs: i32, frame: usize) -> LossResult {
    let (mut c_dec, mut r_dec) = e.decoders(fs);
    let ch = e.channels;
    // The frame size at the rate of the decoders.
    let frame_dec = frame * fs as usize / FS as usize;
    let max_frame = fs as usize / 25 * 3;
    // Half a second without loss at the end.
    let tail = FS as usize / 2 / frame;
    let mut out = vec![0f32; max_frame * ch];
    let (mut c_all, mut r_all) = (Vec::new(), Vec::new());
    // The lost packets, each with its place in its burst, from 1.
    let mut lost_packets = Vec::new();
    let mut run = 0;
    for (i, packet) in e.packets.iter().enumerate() {
        let what = || format!("{} at {fs} Hz, packet {i}", e.label);
        let lost = is_lost(i, e.packets.len(), tail);
        let (c, n) = if lost {
            let c = c_dec.decode_float(None, frame_dec, false).unwrap();
            let n = r_dec.decode(&[], frame_dec, &mut out).unwrap_or_else(|err| panic!("{}: {err}", what()));
            (c, n)
        } else {
            let c = c_dec.decode_float(Some(packet), max_frame, false).unwrap();
            let n = r_dec.decode(packet, max_frame, &mut out).unwrap_or_else(|err| panic!("{}: {err}", what()));
            (c, n)
        };
        assert_eq!(n, frame_dec, "{}: samples per channel", what());
        assert_eq!(c.len(), n * ch, "{}: samples of the C decoder", what());
        assert_eq!(r_dec.final_range(), c_dec.final_range(), "{}: final range", what());
        if lost {
            // The final range is 0 after a concealed packet.
            assert_eq!(r_dec.final_range(), 0, "{}: final range after a loss", what());
            run += 1;
            lost_packets.push((i, run));
        } else {
            assert_eq!(r_dec.final_range(), e.ranges[i], "{}: final range", what());
            run = 0;
        }
        assert!(out[..n * ch].iter().all(|v| v.is_finite()), "{}: output is not finite", what());
        c_all.extend_from_slice(&c);
        r_all.extend_from_slice(&out[..n * ch]);
    }
    assert!(lost_packets.len() >= 13, "{}: only {} packets lost", e.label, lost_packets.len());

    let mut result = LossResult::BEST;
    for c in 0..ch {
        let c_ch = channel(&c_all, ch, c);
        let r_ch = channel(&r_all, ch, c);
        let (mut lost_signal, mut lost_noise) = (0f64, 0f64);
        for &(i, place) in &lost_packets {
            let (from, to) = (i * frame_dec, (i + 1) * frame_dec);
            let level = level_difference_db(&c_ch[from..to], &r_ch[from..to]);
            // The concealed frame and the 2.5 ms after it, where the next
            // frame joins.
            let join = to + fs as usize / 400;
            let (c_step, r_step) = (largest_step(&c_ch, from, join), largest_step(&r_ch, from, join));
            let step = r_step / (c_step + 1e-4);
            if place <= 2 {
                result.level_db = result.level_db.max(level);
                result.step_ratio = result.step_ratio.max(step);
            } else {
                result.deep_level_db = result.deep_level_db.max(level);
                result.deep_step_ratio = result.deep_step_ratio.max(step);
            }
            lost_signal += energy(&c_ch[from..to]);
            lost_noise += c_ch[from..to].iter().zip(&r_ch[from..to]).map(|(a, b)| ((a - b) as f64).powi(2)).sum::<f64>();
        }
        result.lost_snr = result.lost_snr.min(snr_db(lost_signal, lost_noise));
        // The last quarter second.
        let from = c_ch.len() - fs as usize / 4;
        result.tail_snr = result.tail_snr.min(snr_at_lag(&c_ch[from..], &r_ch[from..], 0, 0));
    }
    result
}

/// Runs the loss matrix for one layout and one signal and returns the worst
/// results: music and speech settings, 3 bitrates, 5 frame sizes, decoder
/// rates 16 and 48 kHz.
fn loss_matrix(layout: Layout, signal: Signal) -> LossResult {
    let mut worst = LossResult::BEST;
    let mut checked = 0;
    for settings in [AUDIO, VOIP_FEC] {
        for kbps in [12, 24, 64] {
            for frame in [120usize, 240, 480, 960, 2880] {
                // 60 packets with losses, then 750 ms without.
                let milliseconds = 60 * frame / 48 + 750;
                let e = encode(layout, settings, kbps, frame, signal, milliseconds, 1.0);
                if e.has_mono_coded_stereo_stream() {
                    continue;
                }
                for fs in [16000, 48000] {
                    worst = worst.worst_of(run_with_losses(&e, fs, frame));
                }
                checked += 1;
            }
        }
    }
    println!("{layout:?} {signal:?}: {checked} encoded signals, {worst:.2?}");
    assert!(checked >= 15, "{layout:?}: only {checked} cases without a mono-coded packet");
    worst
}

/// Speech-like input has one clear pitch period, so both decoders conceal a
/// frame in the same way.
///
/// Measured over all layouts: level within 0.41 dB, largest step 1.04 times
/// the C decoder's, concealed frames at least 19 dB and the end of the
/// signal at least 132 dB from the C decoder's.
fn check_voiced(layout: Layout) {
    let r = loss_matrix(layout, Signal::Voiced);
    assert!(r.level_db <= 1.5, "{layout:?}: a concealed frame is {:.2} dB off the C decoder's level", r.level_db);
    assert!(r.deep_level_db <= 1.5, "{layout:?}: a frame late in a burst is {:.2} dB off the C decoder's level", r.deep_level_db);
    assert!(r.step_ratio <= 1.3, "{layout:?}: a step {:.2} times the C decoder's largest", r.step_ratio);
    assert!(r.deep_step_ratio <= 1.3, "{layout:?}: late in a burst, a step {:.2} times the C decoder's largest", r.deep_step_ratio);
    assert!(r.lost_snr >= 10.0, "{layout:?}: concealed frames {:.1} dB from the C decoder's", r.lost_snr);
    assert!(r.tail_snr >= 100.0, "{layout:?}: {:.1} dB from the C decoder after the losses", r.tail_snr);
}

/// A steady tone fits several pitch periods equally well, and libopus and
/// opus1 do not always pick the same one for the concealment. The first
/// two lost frames of a burst have the same level either way. Later in a
/// burst the level falls at a speed that depends on the period, so it is
/// not compared (measured: up to 18.9 dB apart in the sixth lost frame;
/// libopus itself differs that much between its 16 kHz and its 48 kHz
/// output of the same packets). The largest step of a tone is proportional
/// to its level, so it is not compared late in a burst either.
///
/// How open the choice is, measured on the mono 400 Hz tone in 5 ms CELT
/// frames at 64 kb/s: the period of that tone is exactly 120 samples, so
/// 240, 360, 480 and 600 fit just as well. Over the 13 losses of the stream
/// libopus takes 120, 240 and 480 in turn, and its 16 kHz and 48 kHz
/// decoders take different periods at 3 of the 13 losses for the same
/// packets. opus1 differs from libopus at 5 to 7 of the 13. When the two
/// differ at the start of the burst of 6, the sixth frame is 18.5 dB apart
/// and its largest step 2.1 times libopus's, and the first loss after the
/// burst 1.2 dB apart: the same numbers as between libopus's own two
/// decoders.
///
/// Measured over all layouts, for the first two lost frames of every burst:
/// level within 1.22 dB, largest step 1.09 times the C decoder's. Concealed
/// frames are at least 16 dB and the end of the signal at least 104 dB from
/// the C decoder's.
fn check_tones(layout: Layout) {
    let r = loss_matrix(layout, Signal::Tones);
    assert!(r.level_db <= 1.5, "{layout:?}: a concealed frame is {:.2} dB off the C decoder's level", r.level_db);
    assert!(r.step_ratio <= 1.5, "{layout:?}: a step {:.2} times the C decoder's largest", r.step_ratio);
    assert!(r.lost_snr >= 10.0, "{layout:?}: concealed frames {:.1} dB from the C decoder's", r.lost_snr);
    assert!(r.tail_snr >= 60.0, "{layout:?}: {:.1} dB from the C decoder after the losses", r.tail_snr);
}

#[test]
fn mono_conceals_lost_speech_like_libopus() {
    check_voiced(MONO);
}

#[test]
fn mono_conceals_lost_tones_like_libopus() {
    check_tones(MONO);
}

#[test]
fn stereo_conceals_lost_speech_like_libopus() {
    check_voiced(STEREO);
}

#[test]
fn stereo_conceals_lost_tones_like_libopus() {
    check_tones(STEREO);
}

#[test]
fn quad_conceals_lost_speech_like_libopus() {
    check_voiced(QUAD);
}

#[test]
fn quad_conceals_lost_tones_like_libopus() {
    check_tones(QUAD);
}

#[test]
#[cfg_attr(not(target_arch = "aarch64"), ignore = "bit-exact with libopus only on 64-bit ARM (docs/guide/10, Platforms)")]
fn surround_5_1_conceals_lost_speech_like_libopus() {
    check_voiced(SURROUND_5_1);
}

#[test]
#[cfg_attr(not(target_arch = "aarch64"), ignore = "bit-exact with libopus only on 64-bit ARM (docs/guide/10, Platforms)")]
fn surround_5_1_conceals_lost_tones_like_libopus() {
    check_tones(SURROUND_5_1);
}

#[test]
#[cfg_attr(not(target_arch = "aarch64"), ignore = "bit-exact with libopus only on 64-bit ARM (docs/guide/10, Platforms)")]
fn surround_7_1_conceals_lost_speech_like_libopus() {
    check_voiced(SURROUND_7_1);
}

#[test]
fn surround_7_1_conceals_lost_tones_like_libopus() {
    check_tones(SURROUND_7_1);
}

#[test]
fn family_255_with_3_channels_conceals_lost_speech_like_libopus() {
    check_voiced(DISCRETE_3);
}

#[test]
fn family_255_with_3_channels_conceals_lost_tones_like_libopus() {
    check_tones(DISCRETE_3);
}

#[test]
fn ambisonics_with_4_channels_conceals_lost_speech_like_libopus() {
    check_voiced(AMBISONICS_4);
}

#[test]
fn ambisonics_with_4_channels_conceals_lost_tones_like_libopus() {
    check_tones(AMBISONICS_4);
}

#[test]
fn generic_layout_conceals_lost_speech_like_libopus() {
    check_voiced(Layout::Generic);
}

#[test]
fn generic_layout_conceals_lost_tones_like_libopus() {
    check_tones(Layout::Generic);
}

/// A lost duration does not have to be the frame size of the stream: any
/// multiple of 2.5 ms is concealed, before the first packet too.
#[test]
fn any_lost_duration_is_concealed_like_libopus() {
    // Speech settings at 12 kb/s per channel give SILK streams, which both
    // decoders conceal alike. Music settings give CELT streams.
    for (layout, settings, kbps, min_snr) in [(DISCRETE_3, VOIP_FEC, 12, 100.0), (SURROUND_5_1, AUDIO, 32, 15.0)] {
        let e = encode(layout, settings, kbps, 960, Signal::Voiced, 1500, 1.0);
        for fs in [8000, 48000] {
            let (mut c_dec, mut r_dec) = e.decoders(fs);
            let ch = e.channels;
            let unit = fs as usize / 400;
            let max_frame = fs as usize / 25 * 3;
            let mut out = vec![0f32; max_frame * ch];
            let (mut c_all, mut r_all) = (Vec::new(), Vec::new());
            // Lost durations in units of 2.5 ms, each one before the packet
            // with this index. 48 units are 120 ms, the longest one call
            // conceals; 1000 units count as 48.
            let losses = [(0usize, 1usize), (10, 24), (20, 48), (30, 3), (31, 1), (40, 9), (50, 1000)];
            for (i, packet) in e.packets.iter().enumerate() {
                for &(_, units) in losses.iter().filter(|&&(at, _)| at == i) {
                    let what = format!("{} at {fs} Hz, {units} units lost before packet {i}", e.label);
                    let lost = units * unit;
                    let c = c_dec.decode_float(None, lost, false).unwrap();
                    // The output only has to hold what is concealed.
                    let n = r_dec.decode(&[], lost, &mut out).unwrap_or_else(|err| panic!("{what}: {err}"));
                    assert_eq!(n, units.min(48) * unit, "{what}: concealed samples");
                    assert_eq!(c.len(), n * ch, "{what}: samples of the C decoder");
                    assert_eq!(r_dec.final_range(), 0, "{what}: final range");
                    assert_eq!(c_dec.final_range(), 0, "{what}: final range of the C decoder");
                    let level = level_difference_db(&c, &out[..n * ch]);
                    assert!(level <= 1.5, "{what}: {level:.2} dB off the C decoder's level");
                    c_all.extend_from_slice(&c);
                    r_all.extend_from_slice(&out[..n * ch]);
                }
                let c = c_dec.decode_float(Some(packet), max_frame, false).unwrap();
                let n = r_dec.decode(packet, max_frame, &mut out).unwrap();
                assert_eq!(c.len(), n * ch);
                assert_eq!(r_dec.final_range(), e.ranges[i]);
                c_all.extend_from_slice(&c);
                r_all.extend_from_slice(&out[..n * ch]);
            }
            let snrs = channel_snrs(&c_all, &r_all, ch);
            println!("{} at {fs} Hz: {snrs:.0?} dB", e.label);
            assert!(snrs.iter().all(|&snr| snr >= min_snr), "{} at {fs} Hz: {snrs:.0?} dB", e.label);
        }
    }
}

/// A lost duration that is not a multiple of 2.5 ms is an error in both
/// decoders, and it leaves the decoder as it was.
#[test]
fn a_lost_duration_must_be_a_multiple_of_2_5_ms() {
    let e = encode(SURROUND_5_1, AUDIO, 32, 960, Signal::Voiced, 400, 1.0);
    let (mut c_dec, mut r_dec) = e.decoders(48000);
    let (_, mut untouched) = e.decoders(48000);
    let mut out = vec![0f32; 5760 * 6];
    let mut out_untouched = vec![0f32; 5760 * 6];
    for (i, packet) in e.packets.iter().enumerate() {
        for bad in [1usize, 119, 121, 959, 5759] {
            assert_eq!(c_dec.decode_float(None, bad, false), Err(OPUS_BAD_ARG), "{bad} samples");
            assert!(r_dec.decode(&[], bad, &mut out).is_err(), "{bad} samples");
            assert!(r_dec.decode_fec(&[], bad, &mut out).is_err(), "{bad} samples");
            assert!(r_dec.decode_fec(packet, bad, &mut out).is_err(), "{bad} samples");
        }
        let n = r_dec.decode(packet, 5760, &mut out).unwrap();
        assert_eq!(untouched.decode(packet, 5760, &mut out_untouched), Ok(n));
        assert!(
            out[..n * 6].iter().zip(&out_untouched[..n * 6]).all(|(a, b)| a.to_bits() == b.to_bits()),
            "packet {i}: a rejected call changed the decoder"
        );
        assert_eq!(r_dec.final_range(), e.ranges[i]);
    }
}

/// The mode of each stream of a packet.
fn stream_modes(packet: &[u8], streams: usize) -> Vec<Mode> {
    split_streams(packet, streams).unwrap().iter().map(|s| toc(s.toc).mode).collect()
}

/// What `run_with_fec` counted and measured.
#[derive(Debug)]
struct FecResult {
    /// Number of `decode_fec` calls.
    calls: usize,
    /// Calls whose output is within 100 dB of the C decoder's.
    exact_calls: usize,
    /// Calls on a packet with a Hybrid stream whose final range differs
    /// from the C decoder's.
    hybrid_range_mismatches: usize,
    /// Smallest SNR of a rebuilt frame against the C decoder's, in dB.
    fec_snr: f64,
    /// Largest difference in level of a rebuilt frame to the C decoder's,
    /// in dB.
    fec_level_db: f64,
    /// Smallest SNR of a decoded packet against the C decoder's, in dB.
    decoded_snr: f64,
}

impl FecResult {
    const NONE: Self = Self {
        calls: 0,
        exact_calls: 0,
        hybrid_range_mismatches: 0,
        fec_snr: f64::INFINITY,
        fec_level_db: 0.0,
        decoded_snr: f64::INFINITY,
    };

    fn add(&mut self, other: &Self) {
        self.calls += other.calls;
        self.exact_calls += other.exact_calls;
        self.hybrid_range_mismatches += other.hybrid_range_mismatches;
        self.fec_snr = self.fec_snr.min(other.fec_snr);
        self.fec_level_db = self.fec_level_db.max(other.fec_level_db);
        self.decoded_snr = self.decoded_snr.min(other.decoded_snr);
    }
}

/// Decodes `e` with every 5th packet lost and rebuilt from the packet after
/// it. Every 20th loss takes two packets: the first one is concealed, the
/// second one rebuilt. Compares both decoders call by call.
///
/// With `exact`, every rebuilt frame must be the C decoder's, sample for
/// sample, with the same final range. That holds for streams that stay
/// SILK-only. Elsewhere CELT conceals a part of the frame: the band above
/// 8 kHz of a Hybrid stream, a CELT stream as a whole, and the frames around
/// a change of the coding mode.
fn run_with_fec(e: &Encoded, fs: i32, frame: usize, exact: bool) -> FecResult {
    let (mut c_dec, mut r_dec) = e.decoders(fs);
    let ch = e.channels;
    let frame_dec = frame * fs as usize / FS as usize;
    let max_frame = fs as usize / 25 * 3;
    let mut out = vec![0f32; max_frame * ch];
    let mut result = FecResult::NONE;
    let lost = |i: usize| i % 5 == 4 || i % 20 == 13;
    for (i, packet) in e.packets.iter().enumerate() {
        let what = || format!("{} at {fs} Hz, packet {i}", e.label);
        if lost(i) {
            continue;
        }
        if i > 0 && lost(i - 1) {
            if i > 1 && lost(i - 2) {
                // Two packets were lost. Nothing is known about the first.
                let c = c_dec.decode_float(None, frame_dec, false).unwrap();
                let n = r_dec.decode(&[], frame_dec, &mut out).unwrap();
                assert_eq!(c.len(), n * ch, "{}: concealed samples", what());
            }
            let c = c_dec.decode_float(Some(packet), frame_dec, true).unwrap_or_else(|err| panic!("{}: C FEC: {err}", what()));
            let n = r_dec.decode_fec(packet, frame_dec, &mut out).unwrap_or_else(|err| panic!("{}: FEC: {err}", what()));
            assert_eq!(n, frame_dec, "{}: samples from FEC", what());
            assert_eq!(c.len(), n * ch, "{}: samples of the C decoder", what());
            let rebuilt = &out[..n * ch];
            assert!(rebuilt.iter().all(|v| v.is_finite()), "{}: output is not finite", what());
            let snr = min_channel_snr(&c, rebuilt, ch);
            result.calls += 1;
            result.exact_calls += (snr >= 100.0) as usize;
            result.fec_snr = result.fec_snr.min(snr);
            result.fec_level_db = result.fec_level_db.max(level_difference_db(&c, rebuilt));
            if r_dec.final_range() != c_dec.final_range() {
                let modes = stream_modes(packet, e.streams);
                assert!(modes.contains(&Mode::Hybrid), "{}: final range after FEC without a Hybrid stream", what());
                result.hybrid_range_mismatches += 1;
            }
            if exact {
                assert!(snr >= 100.0, "{}: FEC frame {snr:.1} dB from the C decoder's", what());
                assert_eq!(r_dec.final_range(), c_dec.final_range(), "{}: final range after FEC", what());
            }
        }
        let c = c_dec.decode_float(Some(packet), max_frame, false).unwrap();
        let n = r_dec.decode(packet, max_frame, &mut out).unwrap_or_else(|err| panic!("{}: {err}", what()));
        assert_eq!(c.len(), n * ch, "{}: samples per channel", what());
        assert_eq!(r_dec.final_range(), e.ranges[i], "{}: final range", what());
        result.decoded_snr = result.decoded_snr.min(min_channel_snr(&c, &out[..n * ch], ch));
    }
    result
}

/// Runs the FEC matrix: speech settings with error correction, 4 bitrates,
/// 4 frame sizes, decoder rates 16 and 48 kHz.
///
/// With `silk_at_12k`, the layout has mono streams only, and at 12 kb/s per
/// channel libopus keeps them SILK-only. Those cases are checked exactly.
fn fec_matrix(layouts: &[Layout], silk_at_12k: bool) -> FecResult {
    let mut total = FecResult::NONE;
    let mut exact_cases = 0;
    for &layout in layouts {
        for kbps in [12, 16, 24, 32] {
            for frame in [480usize, 960, 1920, 2880] {
                let e = encode(layout, VOIP_FEC, kbps, frame, Signal::Voiced, 2000, 1.0);
                if e.has_mono_coded_stereo_stream() {
                    continue;
                }
                let exact = silk_at_12k && kbps == 12;
                if exact {
                    let all_silk = e.packets.iter().all(|p| stream_modes(p, e.streams).iter().all(|&m| m == Mode::Silk));
                    assert!(all_silk, "{}: not SILK-only", e.label);
                    exact_cases += 1;
                }
                for fs in [16000, 48000] {
                    total.add(&run_with_fec(&e, fs, frame, exact));
                }
            }
        }
    }
    println!("{layouts:?}: {total:.2?}");
    if silk_at_12k {
        assert_eq!(exact_cases, 4 * layouts.len());
    }
    total
}

/// Layouts with mono streams only. At 12 kb/s per channel every stream is
/// SILK-only and every rebuilt frame must be the C decoder's.
#[test]
fn fec_rebuilds_lost_frames_like_libopus() {
    let r = fec_matrix(&[MONO, DISCRETE_3, DISCRETE_5], true);
    assert!(r.calls >= 1500, "{r:?}");
    // Nearly all rebuilt frames are the C decoder's, sample for sample
    // (measured: 1739 of 1752). The others lie next to a change of the
    // coding mode, where CELT conceals a part.
    assert!(r.exact_calls * 100 >= r.calls * 95, "{r:?}");
    assert!(r.fec_level_db <= 1.0, "{r:?}");
    assert!(r.fec_snr >= 10.0, "{r:?}");
    assert!(r.decoded_snr >= 15.0, "{r:?}");
}

/// Layouts with a stereo stream, and ambisonics. libopus codes most of
/// their streams with CELT, so most of a lost frame is concealed.
#[test]
fn fec_with_stereo_streams_follows_libopus() {
    let r = fec_matrix(&[STEREO, AMBISONICS_4, Layout::Generic], false);
    assert!(r.calls >= 1000, "{r:?}");
    assert!(r.fec_level_db <= 1.0, "{r:?}");
    assert!(r.fec_snr >= 10.0, "{r:?}");
    assert!(r.decoded_snr >= 15.0, "{r:?}");
}

#[test]
fn fec_in_surround_layouts_follows_libopus() {
    let r = fec_matrix(&[SURROUND_5_1, SURROUND_7_1], false);
    assert!(r.calls >= 1000, "{r:?}");
    assert!(r.fec_level_db <= 1.0, "{r:?}");
    assert!(r.fec_snr >= 10.0, "{r:?}");
    assert!(r.decoded_snr >= 15.0, "{r:?}");
}

/// After `decode_fec` on a Hybrid packet, `OpusDecoder::final_range` is not
/// the value libopus reports (see its documentation). The multistream value
/// is the XOR of the stream values, so it differs too.
#[test]
fn final_range_after_fec_of_a_hybrid_packet_matches_libopus() {
    let r = fec_matrix(&[MONO, DISCRETE_3], true);
    assert!(r.calls >= 1000, "{r:?}");
    assert_eq!(r.hybrid_range_mismatches, 0, "{r:?}");
}

/// The lost duration given to `decode_fec` may be longer than the frame the
/// FEC data stands for. The start is then concealed. An empty packet
/// conceals everything.
#[test]
fn fec_with_a_longer_lost_duration_matches_libopus() {
    // 12 kb/s per channel: SILK streams, the same in both decoders.
    let e = encode(DISCRETE_3, VOIP_FEC, 12, 960, Signal::Voiced, 2000, 1.0);
    for fs in [16000, 48000] {
        let (mut c_dec, mut r_dec) = e.decoders(fs);
        let unit = fs as usize / 400;
        let max_frame = fs as usize / 25 * 3;
        let mut out = vec![0f32; max_frame * 3];
        let mut calls = 0;
        for (i, packet) in e.packets.iter().enumerate() {
            assert!(stream_modes(packet, 3).iter().all(|&m| m == Mode::Silk), "{}: packet {i}", e.label);
            // Before every 7th packet: a loss of 20, 40, 60 or 120 ms, or
            // of 20 ms without a packet to rebuild it from.
            if i % 7 == 6 {
                let units = [8usize, 16, 24, 48, 8][(i / 7) % 5];
                let fec_packet: &[u8] = if (i / 7) % 5 == 4 { &[] } else { packet };
                let what = format!("{} at {fs} Hz, {units} units before packet {i}", e.label);
                let c = c_dec.decode_float((!fec_packet.is_empty()).then_some(fec_packet), units * unit, true).unwrap();
                let n = r_dec.decode_fec(fec_packet, units * unit, &mut out).unwrap_or_else(|err| panic!("{what}: {err}"));
                assert_eq!(n, units * unit, "{what}: samples");
                assert_eq!(c.len(), n * 3, "{what}: samples of the C decoder");
                let snr = min_channel_snr(&c, &out[..n * 3], 3);
                assert!(snr >= 100.0, "{what}: {snr:.1} dB from the C decoder");
                assert_eq!(r_dec.final_range(), c_dec.final_range(), "{what}: final range");
                calls += 1;
            }
            let c = c_dec.decode_float(Some(packet), max_frame, false).unwrap();
            let n = r_dec.decode(packet, max_frame, &mut out).unwrap();
            let snr = min_channel_snr(&c, &out[..n * 3], 3);
            assert!(snr >= 100.0, "{} at {fs} Hz, packet {i}: {snr:.1} dB from the C decoder", e.label);
            assert_eq!(r_dec.final_range(), e.ranges[i]);
        }
        assert!(calls >= 10);
    }
}

/// One stream of a multistream packet as an ordinary packet, and its
/// channel count.
fn stream_packets(e: &Encoded, packet: &[u8]) -> Vec<(Vec<u8>, usize)> {
    split_streams(packet, e.streams)
        .unwrap()
        .into_iter()
        .enumerate()
        .map(|(s, part)| (part.packet, if s < e.coupled { 2 } else { 1 }))
        .collect()
}

/// Copies the outputs of the stream decoders to the output channels that
/// the mapping names. `decoded[s]` holds `samples` samples of stream `s`,
/// interleaved.
fn route<T: Copy + Default>(e: &Encoded, decoded: &[Vec<T>], samples: usize) -> Vec<T> {
    let mut out = vec![T::default(); samples * e.channels];
    for (c, &m) in e.mapping.iter().enumerate() {
        let m = m as usize;
        if m == 255 {
            continue;
        }
        // Decoded channels: 2 per stereo stream first, then 1 per mono stream.
        let (stream, stream_channels, index) = if m < 2 * e.coupled { (m / 2, 2, m % 2) } else { (m - e.coupled, 1, 0) };
        for i in 0..samples {
            out[i * e.channels + c] = decoded[stream][i * stream_channels + index];
        }
    }
    out
}

/// What is done with a packet in `multistream_output_is_one_decoder_per_stream`.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Step {
    Decode,
    /// The packet is lost and concealed.
    Conceal,
    /// The packet before was lost: rebuild it from this one, then decode
    /// this one.
    FecThenDecode,
}

/// The multistream decoder adds nothing of its own: its output is the
/// output of one `OpusDecoder` per stream, copied to the right channels,
/// bit for bit, and its final range is the XOR of theirs.
///
/// For 16-bit output the reference is built from the float output of the
/// stream decoders, as libopus `opus_multistream_decode` builds it: each
/// decoded packet of a stream is soft-clipped with the memory of that
/// stream, and each sample is rounded with `FLOAT2INT16`. One
/// `OpusDecoder::decode_i16` per stream is not the reference: like
/// `opus_decode`, it rounds halfway values in blocks of 16 on ARM.
#[test]
fn multistream_output_is_one_decoder_per_stream() {
    let cases = [
        (SURROUND_5_1, AUDIO, 24, 960, 1.0f32),
        (SURROUND_7_1, VOIP_FEC, 16, 480, 1.0),
        (DISCRETE_5, VOIP_FEC, 16, 960, 1.0),
        (AMBISONICS_6, AUDIO, 48, 1920, 1.0),
        (Layout::Generic, VOIP_FEC, 32, 2880, 1.0),
        // Beyond full scale, for the soft clip of the 16-bit output.
        (SURROUND_5_1, AUDIO, 48, 960, 6.0),
        (Layout::Generic, AUDIO, 64, 240, 6.0),
    ];
    for (layout, settings, kbps, frame, gain) in cases {
        let e = encode(layout, settings, kbps, frame, Signal::Voiced, 1500, gain);
        assert!(!e.has_mono_coded_stereo_stream(), "{}", e.label);
        for fs in [16000, 48000] {
            let frame_dec = frame * fs as usize / FS as usize;
            let max_frame = fs as usize / 25 * 3;
            let new_stream_decoders = || -> Vec<OpusDecoder> {
                (0..e.streams).map(|s| OpusDecoder::new(SampleRate::try_from(fs).unwrap(), Channels::try_from(if s < e.coupled { 2 } else { 1 }).unwrap()).unwrap()).collect()
            };
            // Float output.
            let (_, mut ms) = e.decoders(fs);
            let mut singles = new_stream_decoders();
            // 16-bit output, and the soft clipper's memory of each stream.
            let (_, mut ms16) = e.decoders(fs);
            let mut singles16 = new_stream_decoders();
            let mut clip_memory = vec![[0f32; 2]; e.streams];
            let mut out = vec![0f32; max_frame * e.channels];
            let mut out16 = vec![0i16; max_frame * e.channels];
            let (mut clipped, mut concealed, mut rebuilt) = (0usize, 0usize, 0usize);
            for (i, packet) in e.packets.iter().enumerate() {
                let what = format!("{} at {fs} Hz, packet {i}", e.label);
                let step = if i % 6 == 4 || i % 17 == 9 {
                    Step::Conceal
                } else if i % 6 == 5 {
                    Step::FecThenDecode
                } else {
                    Step::Decode
                };
                let parts = stream_packets(&e, packet);
                // (FEC, packet, frame size) for each call of this step.
                let calls: &[(bool, bool)] = match step {
                    Step::Decode => &[(false, true)],
                    Step::Conceal => &[(false, false)],
                    Step::FecThenDecode => &[(true, true), (false, true)],
                };
                for &(fec, with_packet) in calls {
                    let ms_packet: &[u8] = if with_packet { packet } else { &[] };
                    let size = if fec || !with_packet { frame_dec } else { max_frame };
                    // Float.
                    let n = if fec { ms.decode_fec(ms_packet, size, &mut out) } else { ms.decode(ms_packet, size, &mut out) }
                        .unwrap_or_else(|err| panic!("{what}: {err}"));
                    let mut range = 0;
                    let decoded: Vec<Vec<f32>> = singles
                        .iter_mut()
                        .zip(&parts)
                        .map(|(dec, (stream_packet, stream_channels))| {
                            let stream_packet: &[u8] = if with_packet { stream_packet } else { &[] };
                            let mut buf = vec![0f32; size * stream_channels];
                            let got = if fec { dec.decode_fec(stream_packet, size, &mut buf) } else { dec.decode(stream_packet, size, &mut buf) }
                                .unwrap_or_else(|err| panic!("{what}: stream decoder: {err}"));
                            assert_eq!(got, n, "{what}: samples of a stream");
                            range ^= dec.final_range();
                            buf
                        })
                        .collect();
                    let want = route(&e, &decoded, n);
                    assert!(
                        out[..n * e.channels].iter().zip(&want).all(|(a, b)| a.to_bits() == b.to_bits()),
                        "{what}: float output differs from the stream decoders'"
                    );
                    assert_eq!(ms.final_range(), range, "{what}: final range");

                    // 16 bits.
                    let n16 = if fec { ms16.decode_fec_i16(ms_packet, size, &mut out16) } else { ms16.decode_i16(ms_packet, size, &mut out16) }
                        .unwrap_or_else(|err| panic!("{what}: 16 bits: {err}"));
                    assert_eq!(n16, n, "{what}: samples of the 16-bit output");
                    let decoded: Vec<Vec<i16>> = singles16
                        .iter_mut()
                        .zip(&parts)
                        .zip(clip_memory.iter_mut())
                        .map(|((dec, (stream_packet, stream_channels)), memory)| {
                            let stream_packet: &[u8] = if with_packet { stream_packet } else { &[] };
                            let mut buf = vec![0f32; size * stream_channels];
                            let got = if fec { dec.decode_fec(stream_packet, size, &mut buf) } else { dec.decode(stream_packet, size, &mut buf) }
                                .unwrap_or_else(|err| panic!("{what}: stream decoder, 16 bits: {err}"));
                            assert_eq!(got, n, "{what}: samples of a stream");
                            // Concealed and rebuilt frames are not
                            // soft-clipped, and keep the memory.
                            if with_packet && !fec {
                                pcm_soft_clip(&mut buf[..n * stream_channels], *stream_channels, memory).unwrap();
                            }
                            to_i16(&buf[..n * stream_channels])
                        })
                        .collect();
                    let want16 = route(&e, &decoded, n);
                    assert_eq!(&out16[..n * e.channels], &want16[..], "{what}: 16-bit output differs from the stream decoders'");
                    clipped += want16.iter().filter(|&&v| v == i16::MAX || v == i16::MIN).count();
                    concealed += !with_packet as usize;
                    rebuilt += fec as usize;
                }
            }
            assert!(concealed >= 5 && rebuilt >= 3, "{}: {concealed} concealed, {rebuilt} rebuilt", e.label);
            if gain > 1.0 {
                assert!(clipped > 0, "{}: the signal did not reach full scale", e.label);
            }
        }
    }
}
