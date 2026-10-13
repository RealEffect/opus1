//! CELT loss concealment against C libopus.
//!
//! Both decoders get the same packets with the same losses. A lost packet is
//! `opus_decode_float(NULL)` for libopus and `decode(&[])` for opus-rs. The
//! outputs are compared frame by frame.
//!
//! libopus conceals a lost CELT frame in one of two ways
//! (`celt_decode_lost`, celt/celt_decoder.c):
//!
//! - It repeats the last pitch period of its output. This is used for the
//!   first 100 ms of a loss in a CELT-only stream.
//! - It plays noise with the spectrum of the last frame, decaying towards
//!   the background noise level. This is used after 100 ms, then until two
//!   packets in a row have arrived, and always for the CELT layer of a
//!   Hybrid stream.
//!
//! The noise comes from the decoder's random seed, so a correct port
//! reproduces it to the last bits (more than 120 dB).
//!
//! The pitch repetition cannot agree that closely, because it is built from
//! the decoder's own output, and that output already differs from libopus's
//! in the last bit (good frames agree to about 135 dB). Three steps amplify
//! the difference:
//!
//! 1. A 24th-order linear predictor is fitted to the output. Changing the
//!    last bit of the output changes a concealed frame at about -70 dB.
//!    Concealed frames therefore agree to 70 to 100 dB.
//! 2. The fit stops at the first order that reaches 30 dB of prediction
//!    gain. When the gain passes that limit slowly, the last bit decides
//!    the order, and the frame agrees to 30 dB or less. This is rare for
//!    full-band output. It happens in a few percent of the frames when the
//!    decoder runs at 8 to 16 kHz, because band-limited output is easy to
//!    predict.
//! 3. The pitch search first picks two candidates at a quarter of the
//!    rate. When the second and third candidate are within 0.01 percent,
//!    the last bit decides which one is examined, and the two decoders can
//!    end up with different pitch periods. A badly matched frame then
//!    disturbs the search for the next loss if that loss follows within
//!    40 ms.
//!
//! An error in the algorithm looks different: a wrong pitch, fade factor or
//! energy leaves 20 dB or less in every frame of its kind. The limits below
//! are therefore on the median, and on the share of frames below 40 dB.
//!
//! Run with `--nocapture` to see the measured numbers.

mod celtdec_common;
mod k24_common;
use celtdec_common::*;
use k24_common::*;

/// What a stream with losses must reach against libopus. SNR values in dB.
#[derive(Clone, Copy)]
struct Limits {
    /// Median SNR of the concealed frames.
    concealed_median: f64,
    /// Median SNR of the first good frame after a loss.
    first_good_median: f64,
    /// Largest share of the concealed frames, and of the first good frames,
    /// that may be below 40 dB. One such frame is always allowed.
    poor_share: f64,
    /// Lowest SNR of any concealed or first good frame. Only noise
    /// concealment can keep such a limit.
    worst: f64,
    /// Lowest SNR of a frame that starts 40 to 120 ms after a loss.
    recovered_worst: f64,
}

/// The limits for pitch repetition were measured on 64-bit ARM. There
/// opus-rs rounds the sums of the concealment the way libopus does (see
/// `plc_mac` in src/celt.rs). On other processors the two round differently,
/// so more frames are badly matched. The limits are wider there. They have
/// not been measured.
const ARM64: bool = cfg!(target_arch = "aarch64");

/// Losses concealed by pitch repetition, full-band output.
const PITCH: Limits = Limits {
    concealed_median: if ARM64 { 65.0 } else { 55.0 },
    first_good_median: if ARM64 { 70.0 } else { 60.0 },
    poor_share: if ARM64 { 0.02 } else { 0.1 },
    worst: f64::NEG_INFINITY,
    recovered_worst: if ARM64 { 85.0 } else { 60.0 },
};

/// Losses concealed by pitch repetition, band-limited output (a decoder
/// below 48 kHz, or a stream narrower than fullband). Up to one frame in
/// ten is badly matched here, see the top of this file.
const PITCH_BAND_LIMITED: Limits = Limits {
    concealed_median: if ARM64 { 60.0 } else { 45.0 },
    first_good_median: if ARM64 { 65.0 } else { 50.0 },
    poor_share: if ARM64 { 0.25 } else { 0.6 },
    worst: f64::NEG_INFINITY,
    recovered_worst: if ARM64 { 60.0 } else { 40.0 },
};

/// Losses concealed with noise only.
const NOISE: Limits = Limits {
    concealed_median: 120.0,
    first_good_median: 120.0,
    poor_share: 0.0,
    worst: 110.0,
    recovered_worst: 110.0,
};

/// Lowest SNR of a single frame that libopus and opus-rs conceal with noise.
const NOISE_FRAME_DB: f64 = 110.0;
/// SNR below which a pitch-concealed frame counts as badly matched.
const POOR_DB: f64 = 40.0;

/// Every `period`-th packet is lost, starting with packet `period - 3`.
fn single_losses(n: usize, period: usize) -> Vec<bool> {
    (0..n).map(|i| i % period == period - 3).collect()
}

/// Runs of `burst` lost packets, each followed by `gap` good ones. The
/// first `gap` packets are good.
fn bursts(n: usize, burst: usize, gap: usize) -> Vec<bool> {
    (0..n).map(|i| i % (burst + gap) >= gap).collect()
}

fn signal(rate: usize, channels: usize, seconds: f64, music: bool) -> Vec<f32> {
    let mono = if music {
        multitone(rate, (rate as f64 * seconds) as usize)
    } else {
        speech_fullband(rate, seconds)
    };
    if channels == 2 {
        to_stereo(&mono, rate)
    } else {
        mono
    }
}

/// Encode with the C encoder in the given mode.
fn encode(
    rate: usize,
    channels: usize,
    frame: usize,
    mode: i32,
    bandwidth: i32,
    bitrate: i32,
    pcm: &[f32],
) -> Vec<Vec<u8>> {
    let mut enc = CEnc::audio(rate, channels);
    enc.force_mode(mode).bandwidth(bandwidth).bitrate(bitrate);
    enc.encode_all(pcm, frame)
}

/// Collects the cases of one test and fails at the end with all of them.
struct Report(Vec<String>);

impl Report {
    fn new() -> Self {
        Self(Vec::new())
    }

    fn check(&mut self, name: &str, s: LossStats, limits: Limits) {
        println!("{name}: {s}");
        let mut fail = |what: &str| self.0.push(format!("{name}: {what}: {s}"));
        if s.lost == 0 || s.concealed.frames == 0 {
            fail("nothing was concealed");
        }
        let too_many_poor = |g: Group| g.poor > 1 && g.poor_share() > limits.poor_share;
        if s.concealed.median < limits.concealed_median || too_many_poor(s.concealed) {
            fail("concealed frames");
        }
        if s.first_good.median < limits.first_good_median || too_many_poor(s.first_good) {
            fail("first good frame after a loss");
        }
        if s.concealed.worst < limits.worst || s.first_good.worst < limits.worst {
            fail("worst frame");
        }
        if s.recovered.worst < limits.recovered_worst {
            fail("frames 40 ms and more after a loss");
        }
        // A badly matched frame can also differ in level. The others must
        // not: the level check is on the stream as a whole.
        if s.concealed.poor == 0 && s.level_diff_db > 0.2 {
            fail("level of the concealed frames");
        }
    }

    fn finish(self) {
        assert!(self.0.is_empty(), "\n{}", self.0.join("\n"));
    }
}

/// Single losses in CELT-only streams at 2.5, 5, 10 and 20 ms, mono and
/// stereo, speech and music. Every loss is concealed by pitch repetition.
#[test]
fn celt_only_single_losses_at_every_frame_size() {
    let rate = 48000;
    let mut report = Report::new();
    for channels in [1usize, 2] {
        for music in [false, true] {
            let pcm = signal(rate, channels, 2.0, music);
            for frame in [120usize, 240, 480, 960] {
                let packets = encode(
                    rate,
                    channels,
                    frame,
                    MODE_CELT,
                    BW_FB,
                    64000 * channels as i32,
                    &pcm,
                );
                let lost = single_losses(packets.len(), 10);
                let (c, r) = decode_with_loss(&packets, &lost, rate, channels, frame);
                let s = loss_stats(&c, &r, &lost, rate, channels, frame);
                let name = format!(
                    "CELT {} ms, {channels} ch, {}",
                    frame as f64 / 48.0,
                    if music { "music" } else { "speech" }
                );
                assert!(s.loudest_concealed_dbfs > -40.0, "{name}: {s}");
                report.check(&name, s, PITCH);
            }
        }
    }
    report.finish();
}

/// The same stream decoded at every output rate. The concealment always
/// runs at 48 kHz inside the decoder, and the output keeps one sample in
/// two, three, four or six. The losses are at least 75 ms apart, so that
/// one badly matched frame cannot disturb the next.
#[test]
fn celt_only_losses_at_every_decoder_rate() {
    let mut report = Report::new();
    for channels in [1usize, 2] {
        let pcm = signal(48000, channels, 3.0, false);
        for enc_frame in [120usize, 480, 960] {
            let packets = encode(
                48000,
                channels,
                enc_frame,
                MODE_CELT,
                BW_FB,
                64000 * channels as i32,
                &pcm,
            );
            let lost = single_losses(packets.len(), (3600 / enc_frame).max(10));
            for rate in [8000usize, 12000, 16000, 24000, 48000] {
                let frame = enc_frame * rate / 48000;
                let (c, r) = decode_with_loss(&packets, &lost, rate, channels, frame);
                let s = loss_stats(&c, &r, &lost, rate, channels, frame);
                let name = format!(
                    "CELT {} ms, {channels} ch, decoder at {rate} Hz",
                    enc_frame as f64 / 48.0
                );
                let limits = if rate == 48000 {
                    PITCH
                } else {
                    PITCH_BAND_LIMITED
                };
                report.check(&name, s, limits);
            }
        }
    }
    report.finish();
}

/// Streams coded at 8, 16 and 24 kHz with the matching CELT bandwidth
/// (narrowband, wideband, super-wideband), each decoded at its own rate.
/// The band range of the concealment ends where the stream's does.
#[test]
fn celt_only_losses_in_narrower_bandwidths() {
    let mut report = Report::new();
    for (rate, bw, name) in [
        (8000usize, BW_NB, "NB"),
        (16000, BW_WB, "WB"),
        (24000, BW_SWB, "SWB"),
    ] {
        for channels in [1usize, 2] {
            let pcm = signal(rate, channels, 3.0, false);
            let frame = rate / 50;
            let packets = encode(
                rate,
                channels,
                frame,
                MODE_CELT,
                bw,
                32000 * channels as i32,
                &pcm,
            );
            assert!(packets.iter().all(|p| toc(p[0]).mode == "CELT"));
            // A burst keeps the predictor of its first frame, so one badly
            // matched fit costs all five pitch-concealed frames of a burst.
            let burst_limits = Limits {
                poor_share: PITCH_BAND_LIMITED.poor_share + 0.15,
                ..PITCH_BAND_LIMITED
            };
            for (pattern, lost, limits) in [
                (
                    "single losses",
                    single_losses(packets.len(), 10),
                    PITCH_BAND_LIMITED,
                ),
                ("bursts of 7", bursts(packets.len(), 7, 10), burst_limits),
            ] {
                let (c, r) = decode_with_loss(&packets, &lost, rate, channels, frame);
                let s = loss_stats(&c, &r, &lost, rate, channels, frame);
                report.check(
                    &format!("CELT {name} at {rate} Hz, {channels} ch, {pattern}"),
                    s,
                    limits,
                );
            }
        }
    }
    report.finish();
}

/// Bursts of 2 to 10 lost packets. With 20 ms frames the first five lost
/// frames are pitch repetition, each quieter than the one before, and the
/// rest are noise. With 2.5 ms frames ten lost frames are all pitch
/// repetition, so that case also has a burst of 50.
#[test]
fn celt_only_bursts() {
    let rate = 48000;
    let mut report = Report::new();
    for channels in [1usize, 2] {
        let pcm = signal(rate, channels, 4.0, false);
        for (frame, burst_lengths) in [
            (960usize, vec![2usize, 3, 4, 5, 6, 8, 10]),
            (480, vec![3, 10, 12]),
            (240, vec![2, 7, 25]),
            (120, vec![2, 10, 50]),
        ] {
            let packets = encode(
                rate,
                channels,
                frame,
                MODE_CELT,
                BW_FB,
                64000 * channels as i32,
                &pcm,
            );
            for burst in burst_lengths {
                let gap = (7200 / frame).max(8);
                let lost = bursts(packets.len(), burst, gap);
                let (c, r) = decode_with_loss(&packets, &lost, rate, channels, frame);
                let s = loss_stats(&c, &r, &lost, rate, channels, frame);
                let name = format!(
                    "CELT {} ms, {channels} ch, bursts of {burst}",
                    frame as f64 / 48.0
                );
                report.check(&name, s, PITCH);
            }
        }
    }
    report.finish();
}

/// A burst long enough to reach the noise concealment must end in noise at
/// the level libopus produces, not in silence. Of eight lost 20 ms frames
/// the sixth is the first noise frame. Its start overlaps the last
/// pitch-concealed frame. The seventh and eighth are noise only.
#[test]
fn long_burst_ends_in_noise_like_libopus() {
    let (rate, frame) = (48000, 960);
    let pcm = signal(rate, 1, 3.0, false);
    let packets = encode(rate, 1, frame, MODE_CELT, BW_FB, 64000, &pcm);
    let lost = bursts(packets.len(), 8, 12);
    let (c, r) = decode_with_loss(&packets, &lost, rate, 1, frame);
    let mut checked = 0;
    for i in 7..packets.len() {
        if !lost[i] {
            continue;
        }
        // Number of lost frames right before this one.
        let before = (0..i).rev().take_while(|&k| lost[k]).count();
        if before < 5 {
            continue;
        }
        let (cf, rf) = (
            &c[i * frame..(i + 1) * frame],
            &r[i * frame..(i + 1) * frame],
        );
        let db = snr(cf, rf);
        println!(
            "lost frame {} of a burst: libopus {:.1} dBFS, opus-rs {:.1} dBFS, {db:.1} dB",
            before + 1,
            dbfs(cf),
            dbfs(rf)
        );
        assert!(
            dbfs(cf) > -90.0,
            "frame {i}: libopus is silent, the test is void"
        );
        assert!(
            (dbfs(cf) - dbfs(rf)).abs() < 0.1,
            "frame {i}: level differs"
        );
        let limit = if before == 5 { POOR_DB } else { NOISE_FRAME_DB };
        assert!(db > limit, "frame {i}: {db:.1} dB");
        checked += 1;
    }
    assert!(checked >= 18, "only {checked} noise frames");
}

/// After a noise-concealed frame the pitch repetition stays off until two
/// packets in a row have arrived (libopus `skip_plc`). Pattern: a long
/// burst, one good packet, one lost (noise), two good, one lost (pitch).
///
/// The lost packet after a single good one must be noise in both decoders.
/// Noise is reproduced to the last bits and pitch repetition is not, so the
/// SNR of that frame tells which of the two opus-rs used.
#[test]
fn pitch_concealment_returns_after_two_good_packets() {
    let (rate, frame) = (48000, 960);
    let pcm = signal(rate, 1, 3.0, false);
    let packets = encode(rate, 1, frame, MODE_CELT, BW_FB, 64000, &pcm);
    let mut lost = vec![false; packets.len()];
    let starts: Vec<usize> = (10..packets.len() - 20).step_by(25).collect();
    for &start in &starts {
        for k in 0..7 {
            lost[start + k] = true;
        }
        lost[start + 8] = true;
        lost[start + 11] = true;
    }
    let (c, r) = decode_with_loss(&packets, &lost, rate, 1, frame);
    let f = |i: usize| {
        snr(
            &c[i * frame..(i + 1) * frame],
            &r[i * frame..(i + 1) * frame],
        )
    };
    let mut pitch_frames = Vec::new();
    for &start in &starts {
        let (after_one, after_two) = (f(start + 8), f(start + 11));
        println!(
            "lost after one good packet {after_one:.1} dB, lost after two good packets {after_two:.1} dB"
        );
        assert!(after_one > NOISE_FRAME_DB, "noise frame: {after_one:.1} dB");
        pitch_frames.push(after_two);
    }
    // The frames after two good packets are pitch repetition: close to
    // libopus, but not to the last bits.
    pitch_frames.sort_by(|a, b| a.total_cmp(b));
    let median = pitch_frames[pitch_frames.len() / 2];
    assert!(
        median > PITCH.concealed_median && median < NOISE_FRAME_DB,
        "pitch frames: median {median:.1} dB"
    );
    let s = loss_stats(&c, &r, &lost, rate, 1, frame);
    let mut report = Report::new();
    report.check("burst, 1 good, 1 lost, 2 good, 1 lost", s, PITCH);
    report.finish();
}

/// A loss before any packet has arrived gives silence, and the stream that
/// follows decodes as if nothing had happened.
#[test]
fn loss_on_the_first_packet() {
    for (mode, bw, bitrate, name) in [
        (MODE_CELT, BW_FB, 64000, "CELT"),
        (MODE_HYBRID, BW_FB, 32000, "Hybrid"),
    ] {
        let (rate, frame) = (48000, 960);
        let pcm = signal(rate, 1, 1.0, false);
        let packets = encode(rate, 1, frame, mode, bw, bitrate, &pcm);
        let mut lost = vec![false; packets.len()];
        lost[0] = true;
        lost[1] = true;
        let (c, r) = decode_with_loss(&packets, &lost, rate, 1, frame);
        assert!(
            r[..2 * frame].iter().all(|v| *v == 0.0),
            "{name}: output before the first packet"
        );
        assert!(
            c[..2 * frame].iter().all(|v| *v == 0.0),
            "{name}: libopus output before the first packet"
        );
        let db = snr(&c[2 * frame..], &r[2 * frame..]);
        println!("{name}: after two lost packets at the start {db:.1} dB");
        assert!(db > 120.0, "{name}: {db:.1} dB");
    }
}

/// Lost frames in Hybrid streams: super-wideband and fullband, mono and
/// stereo, 10 and 20 ms, single losses and bursts. The SILK layer has its
/// own concealment. The CELT layer, bands 17 and up, is always noise.
#[test]
fn hybrid_losses() {
    let mut report = Report::new();
    for (rate, bw, bw_name) in [
        (24000usize, BW_SWB, "SWB"),
        (48000, BW_SWB, "SWB"),
        (48000, BW_FB, "FB"),
    ] {
        for channels in [1usize, 2] {
            let pcm = signal(rate, channels, 3.0, false);
            for frame_ms in [10usize, 20] {
                let frame = rate * frame_ms / 1000;
                let bitrate = if channels == 1 { 32000 } else { 48000 };
                let packets = encode(rate, channels, frame, MODE_HYBRID, bw, bitrate, &pcm);
                assert!(packets.iter().all(|p| toc(p[0]).mode == "Hybrid"));
                for (pattern, lost) in [
                    ("single losses", single_losses(packets.len(), 10)),
                    ("bursts of 3", bursts(packets.len(), 3, 10)),
                    ("bursts of 8", bursts(packets.len(), 8, 10)),
                ] {
                    let (c, r) = decode_with_loss(&packets, &lost, rate, channels, frame);
                    let s = loss_stats(&c, &r, &lost, rate, channels, frame);
                    let name = format!(
                        "Hybrid {bw_name} at {rate} Hz, {frame_ms} ms, {channels} ch, {pattern}"
                    );
                    report.check(&name, s, NOISE);
                }
            }
        }
    }
    report.finish();
}

/// The CELT layer of a concealed Hybrid frame is not silent: it carries
/// noise above 8 kHz. This looks at the high band of the lost frames.
#[test]
fn hybrid_concealment_keeps_the_high_band() {
    let (rate, frame) = (48000usize, 960usize);
    // White noise: plenty of energy above 8 kHz.
    let mut seed = 7u32;
    let pcm: Vec<f32> = (0..rate * 2)
        .map(|_| {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            ((seed >> 8) as f32 / (1u32 << 23) as f32 - 1.0) * 0.1
        })
        .collect();
    let packets = encode(rate, 1, frame, MODE_HYBRID, BW_FB, 40000, &pcm);
    let lost = single_losses(packets.len(), 10);
    let (c, r) = decode_with_loss(&packets, &lost, rate, 1, frame);
    // The second difference is a crude high-pass: it removes most of the
    // SILK band and keeps the CELT layer.
    let high = |x: &[f32]| -> f64 {
        let d: Vec<f32> = x.windows(3).map(|w| w[0] - 2.0 * w[1] + w[2]).collect();
        dbfs(&d)
    };
    let mut checked = 0;
    for i in (0..packets.len()).filter(|&i| lost[i]) {
        let (cf, rf) = (
            &c[i * frame..(i + 1) * frame],
            &r[i * frame..(i + 1) * frame],
        );
        println!(
            "lost frame {i}: high band libopus {:.1} dB, opus-rs {:.1} dB",
            high(cf),
            high(rf)
        );
        assert!(
            high(cf) > -60.0,
            "frame {i}: libopus has no high band, the test is void"
        );
        assert!(
            (high(cf) - high(rf)).abs() < 0.1,
            "frame {i}: high band level differs"
        );
        checked += 1;
    }
    assert!(checked >= 8);
}

/// A CELT-only stream followed by SILK-only packets from another encoder.
/// The switch carries no redundancy frame, so the decoder bridges it with
/// 5 ms of CELT concealment, cross-faded into the SILK frame (libopus
/// `opus_decode_frame`, `transition`).
#[test]
fn switch_from_celt_to_silk_is_bridged_with_concealment() {
    let (rate, frame) = (48000usize, 960usize);
    let pcm = signal(rate, 1, 2.0, false);
    let half = pcm.len() / 2 / frame * frame;
    let mut packets = encode(rate, 1, frame, MODE_CELT, BW_FB, 64000, &pcm[..half]);
    let switch_at = packets.len();
    packets.extend(encode(
        rate,
        1,
        frame,
        MODE_SILK,
        BW_WB,
        24000,
        &pcm[half..],
    ));
    let lost = vec![false; packets.len()];
    let (c, r) = decode_with_loss(&packets, &lost, rate, 1, frame);
    let at = switch_at * frame;
    let bridge = snr(&c[at..at + frame], &r[at..at + frame]);
    let before = snr(&c[..at], &r[..at]);
    let after = snr(&c[at + frame..], &r[at + frame..]);
    println!(
        "before the switch {before:.1} dB, first SILK frame {bridge:.1} dB, after {after:.1} dB"
    );
    assert!(
        dbfs(&c[at..at + 240]) > -60.0,
        "libopus is silent in the bridge, the test is void"
    );
    assert!(
        before > 120.0 && after > 120.0,
        "before {before:.1} dB, after {after:.1} dB"
    );
    assert!(
        bridge > POOR_DB,
        "first frame after the switch: {bridge:.1} dB"
    );
}
