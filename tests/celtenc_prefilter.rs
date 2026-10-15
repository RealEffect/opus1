//! The pitch pre-filter of the CELT encoder against libopus
//! (celt_encoder.c: `run_prefilter` and its call in `celt_encode_with_ec`).
//!
//! Each frame says in its header whether the decoder has to run the pitch
//! post-filter, and with which period, gain and tap set. The tests read
//! these fields from the packets of both encoders, which got the same
//! samples and settings.

mod celtenc_common;

use celtenc_common::c_celt::Stream;
use celtenc_common::*;

/// The post-filter fields of every frame of a stream.
fn postfilters(st: &Stream, payloads: &[Vec<u8>]) -> Vec<Option<(u32, u32, i32)>> {
    let lm = st.format().lm();
    payloads
        .iter()
        .map(|p| parse_header(p, st.stream_channels, st.end_band, lm).postfilter)
        .collect()
}

/// libopus looks for a pitch only when the packet has more than 12 bytes
/// per channel (`enabled` in celt_encode_with_ec). At 32 kb/s a 2.5 ms
/// frame has 9 bytes: the filter parameters would take 2 of them. opus-rs
/// ran the filter there, and the music signal came out 2.8 dB worse than
/// from libopus.
#[test]
fn no_prefilter_in_packets_of_12_bytes_or_less() {
    let pcm = music(48_000, 4, 0);
    for (bitrate, payload) in [(32_000, 9), (41_600, 12)] {
        let mut st = Stream::new(48_000, 1, 5, bitrate);
        st.complexity = 5;
        assert_eq!(st.cbr_bytes(), payload);
        let rs = st.enc_rs(&pcm);
        let on = postfilters(&st, &rs).iter().filter(|p| p.is_some()).count();
        assert_eq!(on, 0, "{bitrate} b/s: {on} frames with a pre-filter");
        let c = st.enc_c(&pcm);
        assert_eq!(postfilters(&st, &c).iter().filter(|p| p.is_some()).count(), 0);
        let (snr_rs, snr_c) = (st.snr_db(&pcm, &rs), st.snr_db(&pcm, &c));
        println!("{bitrate} b/s: opus-rs {snr_rs:.2} dB, libopus {snr_c:.2} dB");
        assert!(snr_rs > snr_c - 0.5, "{bitrate} b/s: {snr_rs:.2} dB against {snr_c:.2} dB");
    }
}

/// How the pre-filter decisions of two streams compare.
#[derive(Debug)]
struct Agreement {
    frames: usize,
    /// Frames with the filter on, in the C stream and in the opus-rs stream.
    on_c: usize,
    on_rs: usize,
    /// Frames in which both encoders took the same decision, on or off.
    same_decision: usize,
    /// Frames with the same decision and the same parameters.
    same: usize,
}

fn agreement(st: &Stream, c: &[Vec<u8>], rs: &[Vec<u8>]) -> Agreement {
    let c = postfilters(st, c);
    let rs = postfilters(st, rs);
    Agreement {
        frames: c.len(),
        on_c: c.iter().filter(|p| p.is_some()).count(),
        on_rs: rs.iter().filter(|p| p.is_some()).count(),
        same_decision: c
            .iter()
            .zip(&rs)
            .filter(|(a, b)| a.is_some() == b.is_some())
            .count(),
        same: c.iter().zip(&rs).filter(|(a, b)| a == b).count(),
    }
}

/// Two channels with the same talker, the right one quieter and 0.25 ms
/// late, at 48 kHz.
fn stereo_speech(seconds: usize) -> Vec<f32> {
    let s = speech(48_000, seconds);
    let late: Vec<f32> = (0..s.len())
        .map(|i| if i >= 12 { 0.6 * s[i - 12] } else { 0.0 })
        .collect();
    stereo_of(&s, &late, 0.0)
}

/// libopus runs the pre-filter for one and for two channels
/// (run_prefilter: a common pitch from the mix of both channels, the comb
/// filter on each channel). opus-rs ran it for mono only, which cost
/// stereo speech 0.5 to 0.8 dB at complexity 5 and above.
#[test]
fn stereo_frames_get_a_prefilter() {
    let pcm = stereo_speech(6);
    for (frame_ms_x2, per_channel) in [(40, 24_000), (40, 48_000), (20, 32_000), (10, 48_000)] {
        let mut st = Stream::new(48_000, 2, frame_ms_x2, 2 * per_channel);
        st.complexity = 5;
        let (c, rs) = (st.enc_c(&pcm), st.enc_rs(&pcm));
        let a = agreement(&st, &c, &rs);
        let (snr_rs, snr_c) = (st.snr_db(&pcm, &rs), st.snr_db(&pcm, &c));
        println!(
            "{} ms, {per_channel} b/s per channel: filter on in {} (libopus {}) of {} frames, \
             same decision in {}, same parameters in {}, opus-rs {snr_rs:.2} dB, libopus {snr_c:.2} dB",
            frame_ms_x2 as f32 / 2.0,
            a.on_rs,
            a.on_c,
            a.frames,
            a.same_decision,
            a.same,
        );
        assert!(a.on_c * 5 > a.frames, "the signal does not use the filter: {a:?}");
        // The filter is on in about as many frames as in libopus, and
        // mostly in the same ones. Period and gain are not compared: they
        // come from the pitch search in src/pitch.rs.
        assert!(a.on_rs * 10 >= a.on_c * 8 && a.on_rs * 8 <= a.on_c * 10, "{a:?}");
        assert!(a.same_decision * 100 >= a.frames * 85, "{a:?}");
        assert!(snr_rs > snr_c - 0.45, "{snr_rs:.2} dB against {snr_c:.2} dB");
    }
}

/// A weakly periodic signal at 48 kHz, mono: 20 harmonics of 240 Hz (a
/// period of 200 samples) under white noise at `noise` times their level.
fn weak_pitch(seconds: usize, noise: f64) -> Vec<f32> {
    let mut rng = Noise(5);
    (0..48_000 * seconds)
        .map(|i| {
            let t = i as f64 / 48_000.0;
            let mut v = 0.0;
            for h in 1..=20 {
                v += (std::f64::consts::TAU * 240.0 * h as f64 * t).sin() / h as f64;
            }
            (0.1 * (v + noise * f64::from(rng.next()))) as f32
        })
        .collect()
}

/// libopus remembers the pitch period of a frame even when it leaves the
/// filter off (celt_encoder.c 2770). A frame whose pitch continues that
/// period needs a gain of 0.2 to switch the filter on. Any other pitch
/// needs 0.4 (run_prefilter: `pf_threshold`). opus-rs forgot the period
/// with the filter, so a weak but steady pitch never got above the higher
/// threshold: the filter stayed off for the whole stream.
#[test]
fn a_weak_steady_pitch_switches_the_filter_on() {
    for noise in [0.6, 0.7, 0.8] {
        let pcm = weak_pitch(4, noise);
        let mut st = Stream::new(48_000, 1, 40, 64_000);
        st.complexity = 5;
        let (c, rs) = (st.enc_c(&pcm), st.enc_rs(&pcm));
        let a = agreement(&st, &c, &rs);
        println!("noise {noise}: {a:?}");
        assert!(a.on_c * 10 >= a.frames * 9, "libopus does not use the filter: {a:?}");
        assert!(a.on_rs * 10 >= a.on_c * 9, "{a:?}");
    }
}

/// For stereo, libopus keeps the filter only when no channel gets clearly
/// worse and at least one gets clearly better: by more than a quarter of
/// the gain of its own level, plus 1 % of the other channel's level
/// (run_prefilter, celt_encoder.c 1562-1569). For mono it only asks that
/// the channel got better.
///
/// Here the left channel has a weak pitch and the right one is quiet
/// noise. The filter makes the left channel a little better, but not by
/// that much, so libopus leaves it off. With the mono rule for both
/// channels it would be on in every frame.
#[test]
fn a_stereo_filter_must_clearly_help_one_channel() {
    let left = weak_pitch(4, 0.6);
    let mut rng = Noise(77);
    let right: Vec<f32> = (0..left.len()).map(|_| 0.005 * rng.next()).collect();
    let pcm = stereo_of(&left, &right, 0.0);
    let mut st = Stream::new(48_000, 2, 40, 128_000);
    st.complexity = 5;
    let (c, rs) = (st.enc_c(&pcm), st.enc_rs(&pcm));
    let a = agreement(&st, &c, &rs);
    println!("{a:?}");
    assert!(a.on_c * 10 < a.frames, "libopus uses the filter: {a:?}");
    assert!(a.same_decision * 100 >= a.frames * 95, "{a:?}");
}

/// `OPUS_RESET_STATE` clears the pitch period of the last frame
/// (prefilter_period 0). The first frame after the reset then needs the
/// higher gain threshold, as for a pitch that does not continue an
/// earlier one (run_prefilter: `pf_threshold`), and its weak pitch leaves
/// the filter off. The frame after it continues the period and switches
/// the filter on again.
#[test]
fn a_reset_forgets_the_pitch_period() {
    for noise in [0.6, 0.7, 0.8] {
        let pcm = weak_pitch(3, noise);
        let mut st = Stream::new(48_000, 1, 40, 64_000);
        st.complexity = 5;
        let at = 100;
        let (c, rs) = st.enc_both_with(&pcm, |k, c_enc, rs_enc| {
            if k == at {
                c_enc.reset();
                rs_enc.reset_state();
            }
        });
        let on = |p: &[Vec<u8>]| -> Vec<bool> {
            postfilters(&st, p).iter().map(|x| x.is_some()).collect()
        };
        let (oc, or) = (on(&c), on(&rs));
        println!("noise {noise}: libopus {:?}, opus-rs {:?}", &oc[at..at + 3], &or[at..at + 3]);
        assert_eq!(oc[at..at + 3], [false, true, true], "libopus");
        assert_eq!(or[at..at + 3], oc[at..at + 3]);
    }
}

/// Print the post-filter fields of both encoders side by side.
/// `CELTENC_PF=frame_ms_x2,bitrate,channels,signal,from,count`.
#[test]
#[ignore = "report"]
fn print_prefilter_decisions() {
    let Ok(spec) = std::env::var("CELTENC_PF") else {
        return;
    };
    let f: Vec<&str> = spec.split(',').collect();
    let num = |i: usize| f[i].parse::<usize>().expect("number");
    let channels = num(2);
    let mut st = Stream::new(48_000, channels, num(0), num(1) as i32);
    st.complexity = 5;
    let m = music(48_000, 4, 0);
    let s = speech(48_000, 4);
    let mono = if f[3] == "music" { m } else { s };
    let pcm = if channels == 2 {
        stereo_of(&mono, &music(48_000, 4, 1), 0.3)
    } else {
        mono
    };
    let c = postfilters(&st, &st.enc_c(&pcm));
    let rs = postfilters(&st, &st.enc_rs(&pcm));
    for k in num(4)..num(4) + num(5) {
        let mark = if c[k] == rs[k] { ' ' } else { '*' };
        println!("{k:5} {mark} C {:?}  rs {:?}", c[k], rs[k]);
    }
}
