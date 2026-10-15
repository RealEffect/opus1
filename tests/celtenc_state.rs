//! What the CELT encoder keeps from one frame to the next, against libopus
//! (celt_encoder.c: the end of `celt_encode_with_ec`, `OPUS_RESET_STATE`).

mod celtenc_common;

use celtenc_common::c_celt::{CELT_SET_END_BAND, Stream};
use celtenc_common::*;

/// Sum over the bands from `from` on of the difference between the coarse
/// energy steps two encoders wrote for one frame.
fn coarse_distance(a: &Header, b: &Header, from: usize) -> i32 {
    a.coarse[from..21]
        .iter()
        .zip(&b.coarse[from..21])
        .map(|(x, y)| (x - y).abs())
        .sum()
}

/// After each frame libopus clears the energy history of the bands the
/// frame did not code: "In case start or end were to change"
/// (celt_encoder.c 2789). When the bandwidth goes up again, the new bands
/// are predicted from an energy of zero. opus-rs kept the energies these
/// bands had when they were last coded, and predicted from those.
///
/// The stream here is fullband, then narrowband for half a second, then
/// fullband again. The signal gets 30 dB quieter while the stream is
/// narrowband, so the stale energies are far off.
#[test]
fn bands_outside_the_coded_range_lose_their_history() {
    let frames_per_part = 25;
    let frame = 960;
    let mut pcm = music(48_000, 2, 0);
    pcm.truncate(3 * frames_per_part * frame);
    for (i, s) in pcm.iter_mut().enumerate() {
        if i >= frames_per_part * frame {
            *s *= 0.03;
        }
    }
    let back = 2 * frames_per_part;
    for bitrate in [32_000, 64_000] {
        let st = Stream::new(48_000, 1, 40, bitrate);
        let (c, rs) = st.enc_both_with(&pcm, |k, c_enc, rs_enc| {
            let end = if (frames_per_part..back).contains(&k) { 13 } else { 21 };
            c_enc.ctl(CELT_SET_END_BAND, end);
            rs_enc.set_end_band(end as usize);
        });
        let snr = |payloads: &[Vec<u8>]| {
            let packets: Vec<Vec<u8>> = payloads
                .iter()
                .enumerate()
                .map(|(k, p)| {
                    let end = if (frames_per_part..back).contains(&k) { 13 } else { 21 };
                    let mut pkt = vec![celt_toc_lm(end, 1, 3)];
                    pkt.extend_from_slice(p);
                    pkt
                })
                .collect();
            let decoded = dec_c(&packets, 1);
            // The last part, from the switch back to fullband on.
            snr_db_delay(&pcm, &decoded, 1, back * frame, CELT_DELAY)
        };
        println!(
            "{bitrate} b/s, last part: opus-rs {:.2} dB, libopus {:.2} dB",
            snr(&rs),
            snr(&c)
        );
        // The first fullband frame after the narrowband part.
        let hc = parse_header(&c[back], 1, 21, 3);
        let hr = parse_header(&rs[back], 1, 21, 3);
        println!("{bitrate} b/s, libopus: intra {}, coarse {:?}", hc.intra, &hc.coarse[13..]);
        println!("{bitrate} b/s, opus-rs: intra {}, coarse {:?}", hr.intra, &hr.coarse[13..]);
        assert_eq!(hr.intra, hc.intra);
        let d = coarse_distance(&hc, &hr, 13);
        assert!(d <= 2, "bands 13 to 20 differ by {d} steps in all");
    }
}

/// libopus counts the transient frames in a row (`consec_transient`), and
/// a transient frame after two or more of them has no anti-collapse. A
/// frame that cannot signal a transient counts too: every 2.5 ms frame,
/// and a frame without room for the flag (celt_encoder.c 2809-2812,
/// `transient_got_disabled`). opus-rs counted only real transients, so
/// the first transient frame after a run of 2.5 ms frames had the flag on.
///
/// The stream is mono, 48 kHz, 64 kb/s: four 2.5 ms frames of quiet noise,
/// then 10 ms frames, each with a click, except one in the middle.
#[test]
fn frames_without_a_transient_flag_count_for_anti_collapse() {
    let short = Stream::new(48_000, 1, 5, 64_000);
    let long = Stream::new(48_000, 1, 20, 64_000);
    let mut noise = Noise(7);
    let mut quiet = |n: usize| -> Vec<f32> { (0..n).map(|_| 0.01 * noise.next()).collect() };
    let click = |mut x: Vec<f32>| {
        for (i, s) in x.iter_mut().enumerate().skip(300).take(40) {
            *s += if i % 2 == 0 { 0.8 } else { -0.8 };
        }
        x
    };
    let mut c_enc = long.c_encoder();
    let mut rs_enc = long.rs_encoder();
    for _ in 0..4 {
        let pcm = quiet(short.frame);
        short.c_frame(&mut c_enc, &pcm);
        short.rs_frame(&mut rs_enc, &pcm);
    }
    let frames = [
        click(quiet(long.frame)),
        quiet(long.frame),
        click(quiet(long.frame)),
    ];
    let mut flags = Vec::new();
    for (k, pcm) in frames.iter().enumerate() {
        let c = long.c_frame(&mut c_enc, pcm);
        let rs = long.rs_frame(&mut rs_enc, pcm);
        let (fc, fr) = (anti_collapse_flag(&c, 1, 21, 2), anti_collapse_flag(&rs, 1, 21, 2));
        println!("10 ms frame {k}: anti-collapse libopus {fc:?}, opus-rs {fr:?}");
        flags.push((fc, fr));
    }
    // libopus: the first click follows four 2.5 ms frames, the second one
    // a frame without a transient.
    assert_eq!(flags[0].0, Some(false));
    assert_eq!(flags[2].0, Some(true));
    for (k, (fc, fr)) in flags.iter().enumerate() {
        assert_eq!(fr, fc, "10 ms frame {k}");
    }
}

/// When a frame has no room left for the spreading decision, libopus
/// stores the normal spreading (celt_encoder.c 2349-2351), whatever the
/// complexity. The next frame that decides by analysis starts its
/// hysteresis from that value. opus-rs stored "no spreading" for such a
/// frame at complexity 0.
///
/// Mono, 48 kHz, 20 ms: ten frames at complexity 0 in packets of 2 bytes,
/// then complexity 5 at 64 kb/s.
#[test]
fn a_frame_without_room_for_spreading_stores_the_normal_one() {
    use celtenc_common::c_celt::OPUS_SET_COMPLEXITY;
    // A steady signal: three tones in faint noise, no transients. Its
    // spreading measure lies where the start value of the hysteresis
    // decides between normal and light spreading.
    let mut rng = Noise(21);
    let pcm: Vec<f32> = (0..96_000)
        .map(|i| {
            let t = i as f32 / 48_000.0;
            let tones: f32 = [330.0f32, 1250.0, 4100.0]
                .iter()
                .map(|f| (std::f32::consts::TAU * f * t).sin())
                .sum();
            0.1 * tones + 0.003 * rng.next()
        })
        .collect();
    let normal = Stream::new(48_000, 1, 40, 64_000);
    let mut tiny = normal.clone();
    // 3-byte packets: the TOC byte and 2 bytes of CELT payload.
    tiny.bitrate = 1_200;
    assert_eq!(tiny.cbr_bytes(), 2);
    let mut c_enc = normal.c_encoder();
    let mut rs_enc = normal.rs_encoder();
    let mut spreads = Vec::new();
    for (k, frame) in pcm.chunks_exact(normal.frame).enumerate().take(40) {
        let st = if k < 10 { &tiny } else { &normal };
        let complexity = if k < 10 { 0 } else { 5 };
        c_enc.ctl(OPUS_SET_COMPLEXITY, complexity);
        rs_enc.complexity = complexity;
        let (c, rs) = (st.c_frame(&mut c_enc, frame), st.rs_frame(&mut rs_enc, frame));
        let (hc, hr) = (parse_header(&c, 1, 21, 3), parse_header(&rs, 1, 21, 3));
        if k < 10 {
            assert_eq!(hc.spread, None, "frame {k} has room for the spreading decision");
        }
        spreads.push((hc.spread, hr.spread));
    }
    println!("{:?}", &spreads[10..20]);
    for (k, (c, rs)) in spreads.iter().enumerate().skip(10) {
        assert_eq!(rs, c, "frame {k}");
    }
}
