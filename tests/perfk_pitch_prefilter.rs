//! The pitch pre-filter of the CELT encoder against libopus, at the level of
//! whole streams: both encoders code the same mono input, and the
//! post-filter parameters they write into each frame header are compared.
//! Those parameters are the result of the pitch search of the pre-filter
//! (pitch_downsample, pitch_search, remove_doubling).

mod celtenc_common;
use celtenc_common::*;

/// A voiced sound: harmonics of a pitch that glides slowly, with a formant
/// around 700 Hz that is much stronger than the fine structure, and a little
/// noise. Without spectral flattening a pitch search follows the formant.
fn voiced_signal(frames: usize, pitch_hz: f64, seed: u32) -> Vec<f32> {
    let mut noise = Noise(seed);
    let mut phase = 0.0f64;
    (0..frames * FRAME)
        .map(|i| {
            let t = i as f64 / FS as f64;
            let f0 = pitch_hz * (1.0 + 0.03 * (std::f64::consts::TAU * 0.4 * t).sin());
            phase += std::f64::consts::TAU * f0 / FS as f64;
            let mut s = 0.0f64;
            for h in 1..=30 {
                let f = f0 * h as f64;
                if f > 7000.0 {
                    break;
                }
                // Formant at 700 Hz, 150 Hz wide.
                let d = (f - 700.0) / 150.0;
                let weight = 0.08 + 1.0 / (1.0 + d * d);
                s += weight * (phase * h as f64).sin();
            }
            (0.08 * s) as f32 + 0.002 * noise.next()
        })
        .collect()
}

/// The encoder switches the post-filter on in the frames where libopus
/// does, and with the same pitch period.
///
/// Counted over voiced sounds with pitches from 90 to 400 Hz, 100 frames
/// each, without the first 10 frames:
/// - frames where both encoders agree on whether the filter is on,
/// - among the frames where both have it on, the ones with the same period,
///   and the ones where period, gain and tap set are all the same.
///
/// Without the spectral flattening of pitch_downsample the period differed
/// in 32 of 626 such frames and one of the three parameters in 122.
#[test]
fn post_filter_parameters_follow_libopus() {
    let (mut frames, mut same_switch, mut both_on, mut same_period, mut same_all) = (0, 0, 0, 0, 0);
    for (k, pitch_hz) in [90.0f64, 117.0, 150.0, 205.0, 260.0, 330.0, 400.0].into_iter().enumerate() {
        let pcm = voiced_signal(100, pitch_hz, 11 + k as u32);
        let cfg = Cfg::new(1, 64_000);
        let rs = headers(&enc_rs(&cfg, &pcm), 1, 21, 3);
        let c = headers(&enc_c(&cfg, &pcm), 1, 21, 3);
        let (mut on_rs, mut on_c, mut agree, mut period) = (0, 0, 0, 0);
        for (a, b) in rs.iter().zip(&c).skip(10) {
            frames += 1;
            on_rs += a.postfilter.is_some() as usize;
            on_c += b.postfilter.is_some() as usize;
            if a.postfilter.is_some() == b.postfilter.is_some() {
                same_switch += 1;
                agree += 1;
            }
            if let (Some(pa), Some(pb)) = (a.postfilter, b.postfilter) {
                both_on += 1;
                if pa.0 == pb.0 {
                    same_period += 1;
                    period += 1;
                }
                if pa == pb {
                    same_all += 1;
                }
            }
        }
        println!(
            "{pitch_hz} Hz: filter on in {on_rs} frames (libopus {on_c}) of 90, same switch in \
             {agree}, same period in {period}"
        );
    }
    println!(
        "{frames} frames: same switch in {same_switch}; both on in {both_on}, of these the same \
         period in {same_period} and all three parameters equal in {same_all}"
    );
    assert!(same_switch * 100 >= frames * 99, "switch: {same_switch} of {frames}");
    assert!(same_period * 100 >= both_on * 99, "period: {same_period} of {both_on}");
    assert!(same_all * 100 >= both_on * 97, "period, gain and tap set: {same_all} of {both_on}");
}
