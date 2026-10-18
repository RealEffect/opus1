//! `CeltEncoder::set_energy_mask` against C libopus, measured at the points
//! where the two encoders can be compared exactly.
//!
//! `celtenc_energy_mask.rs` compares mean packet sizes and the error per
//! frequency range. Those follow libopus within about 10 %, which leaves
//! room for a wrong constant. The tests here are sharper:
//!
//! - From 96 kb/s per channel up, the VBR target of opus1 without a mask
//!   is within a few bytes of libopus's in every frame (below that rate
//!   libopus adds its temporal VBR, which opus1 does not have). The
//!   number of bytes a mask adds to or takes from each packet must then be
//!   the same in both encoders.
//! - The allocation trim and the dynamic allocation boosts are written at
//!   the start of each packet. `parse_header` reads them back, from the
//!   packets of both encoders.
//!
//! A mask value is in log2 units: 0 means the band is not masked by the
//! other surround channels, -2 means it is deeply masked.

mod celtenc_common;
use celtenc_common::*;
use opus1::internals::celt::CeltEncoder;
use opus1::internals::modes::default_mode;
use opus1::internals::range_coder::RangeCoder;

/// A mask: its name and the value for (channel, band).
type Mask = (&'static str, fn(usize, usize) -> f32);

fn mask_values(channels: usize, value: fn(usize, usize) -> f32) -> Vec<f32> {
    (0..channels)
        .flat_map(|c| (0..21).map(move |i| value(c, i)))
        .collect()
}

fn packet_sizes(packets: &[Vec<u8>]) -> Vec<i32> {
    packets.iter().map(|p| p.len() as i32).collect()
}

/// Largest difference, over the frames, between the bytes a mask adds to a
/// packet of opus1 and the bytes it adds to the packet of libopus.
/// Also returns the mean number of bytes it adds in libopus.
fn size_effect(rs_none: &[i32], c_none: &[i32], rs_mask: &[i32], c_mask: &[i32]) -> (i32, f64) {
    let mut largest = 0;
    let mut c_sum = 0;
    for k in 0..rs_none.len() {
        let rs_effect = rs_mask[k] - rs_none[k];
        let c_effect = c_mask[k] - c_none[k];
        largest = largest.max((rs_effect - c_effect).abs());
        c_sum += c_effect;
    }
    (largest, f64::from(c_sum) / rs_none.len() as f64)
}

/// The masks of the packet size test. Together they reach every branch of
/// the mask analysis (celt_encoder.c 2114-2188):
/// - values outside -2 to 0.25, which are limited;
/// - positive values, which count half;
/// - slopes below and above the limit of 0.031 per band;
/// - bands 0 and 1 against the rest: the first frame of a stream looks at
///   those two bands only;
/// - bands that are masked less than the rest ("holes"), fewer than 3 and
///   3 or more of them;
/// - masks that differ between the two channels of a stereo stream.
const SIZE_MASKS: [Mask; 20] = [
    ("flat -3", |_, _| -3.0),
    ("flat -2", |_, _| -2.0),
    ("flat -1", |_, _| -1.0),
    ("flat -0.5", |_, _| -0.5),
    ("flat 0", |_, _| 0.0),
    ("flat +0.2", |_, _| 0.2),
    ("flat +0.25", |_, _| 0.25),
    ("flat +1", |_, _| 1.0),
    ("slope +0.02", |_, i| -1.0 + 0.02 * i as f32),
    ("slope +0.04", |_, i| -1.2 + 0.04 * i as f32),
    ("slope -0.04", |_, i| -0.4 - 0.04 * i as f32),
    ("slope +0.1", |_, i| -2.0 + 0.1 * i as f32),
    ("bands 0 and 1 open", |_, i| if i < 2 { 0.0 } else { -2.0 }),
    (
        "bands 0 and 1 masked",
        |_, i| if i < 2 { -2.0 } else { 0.0 },
    ),
    ("four holes", |_, i| if i % 5 == 2 { 0.0 } else { -1.5 }),
    (
        "two holes",
        |_, i| if i == 6 || i == 14 { 0.0 } else { -1.5 },
    ),
    (
        "left masked, right open",
        |c, _| if c == 0 { -2.0 } else { 0.0 },
    ),
    (
        "left open, right masked",
        |c, _| if c == 0 { 0.0 } else { -2.0 },
    ),
    ("holes in the left channel", |c, i| {
        if c == 0 && i % 5 == 2 { 0.0 } else { -1.5 }
    }),
    ("holes in the right channel", |c, i| {
        if c == 1 && i % 5 == 2 { 0.0 } else { -1.5 }
    }),
];

/// A mask changes the size of every packet by the number of bytes it does
/// in libopus, to within 4 bytes, where the masks move the size by up to
/// 320 bytes.
///
/// Measured: the largest difference in any frame is 2 bytes for mono and 4
/// bytes for stereo.
///
/// Two stereo masks are left out, "flat -3" and "flat -2". They halve the
/// rate. opus1, whose stereo packets are about 9 bytes smaller than
/// libopus's without any mask, then drops the highest band in frames where
/// libopus keeps it, and the next frame's target follows the number of
/// coded bands. That difference is not one of the mask.
#[test]
fn mask_changes_each_packet_size_as_in_libopus() {
    for (channels, bitrate) in [(1usize, 128_000), (2, 256_000)] {
        let pcm = broadband(40, channels);
        let mut cfg = Cfg::new(channels, bitrate);
        cfg.constrained_vbr = false;
        let rs_none = packet_sizes(&enc_rs(&cfg, &pcm));
        let c_none = packet_sizes(&enc_c(&cfg, &pcm));
        let mut effects = Vec::new();
        for (name, value) in SIZE_MASKS {
            if channels == 2 && (name == "flat -3" || name == "flat -2") {
                continue;
            }
            cfg.energy_mask = Some(mask_values(channels, value));
            let rs = packet_sizes(&enc_rs(&cfg, &pcm));
            let c = packet_sizes(&enc_c(&cfg, &pcm));
            let (largest, c_mean) = size_effect(&rs_none, &c_none, &rs, &c);
            println!(
                "{channels} ch, {name}: libopus packets change by {c_mean:+.1} bytes, \
                 opus1 differs from that by at most {largest} in any frame"
            );
            assert!(
                largest <= 4,
                "{channels} ch, {name}: the mask changes a packet by {largest} bytes more or \
                 less than in libopus"
            );
            effects.push((name, c_mean));
        }
        let effect = |wanted: &str| effects.iter().find(|(name, _)| *name == wanted).unwrap().1;
        // The test compares something: the masks move the size a lot, and
        // the limits at -2 and 0.25 are in force.
        let per_channel = channels as f64;
        assert!(effect("flat -1") < -50.0 * per_channel);
        assert!(effect("flat 0") > 30.0 * per_channel);
        assert_eq!(effect("flat +0.25"), effect("flat +1"));
        if channels == 1 {
            assert_eq!(effect("flat -3"), effect("flat -2"));
            assert!(effect("flat -2") < -120.0);
        }
    }
}

/// The slope of a mask moves the allocation trim by the same number of
/// steps as in libopus, in every frame.
///
/// The trim is 2 without a mask on this signal (mono, 64 kb/s). A mask
/// that rises by 0.04 per band lowers it to 1, one that rises by 0.1 per
/// band lowers it to 0: the slope counts half and is limited to 0.031 per
/// band, which is 2 steps. Falling masks raise the trim by as much.
#[test]
fn mask_slope_moves_the_allocation_trim_as_in_libopus() {
    const SLOPES: [Mask; 9] = [
        ("flat -1", |_, _| -1.0),
        ("slope +0.02", |_, i| -1.0 + 0.02 * i as f32),
        ("slope +0.04", |_, i| -1.2 + 0.04 * i as f32),
        ("slope +0.1", |_, i| -2.0 + 0.1 * i as f32),
        ("slope -0.02", |_, i| -0.6 - 0.02 * i as f32),
        ("slope -0.04", |_, i| -0.4 - 0.04 * i as f32),
        ("slope -0.1", |_, i| -0.1 * i as f32),
        ("low bands masked", |_, i| if i < 10 { -2.0 } else { 0.0 }),
        ("high bands masked", |_, i| if i < 10 { 0.0 } else { -2.0 }),
    ];
    for (channels, bitrate, vbr) in [
        (1usize, 64_000, false),
        (2, 128_000, false),
        (1, 64_000, true),
        (2, 128_000, true),
    ] {
        let pcm = broadband(50, channels);
        let mut cfg = Cfg::new(channels, bitrate);
        cfg.vbr = vbr;
        let trims = |cfg: &Cfg| {
            let of = |packets: &[Vec<u8>]| -> Vec<i32> {
                headers(packets, channels, 21, 3)
                    .iter()
                    .map(|h| h.trim.expect("a packet this large has a trim"))
                    .collect()
            };
            (of(&enc_rs(cfg, &pcm)), of(&enc_c(cfg, &pcm)))
        };
        let mean = |v: &[i32]| v.iter().sum::<i32>() as f64 / v.len() as f64;
        let (rs_none, c_none) = trims(&cfg);
        assert_eq!(
            rs_none, c_none,
            "{channels} ch, vbr {vbr}: trim without a mask"
        );
        let mut shifts = Vec::new();
        for (name, value) in SLOPES {
            cfg.energy_mask = Some(mask_values(channels, value));
            let (rs, c) = trims(&cfg);
            let same = rs.iter().zip(&c).filter(|(a, b)| a == b).count();
            let shift = mean(&c) - mean(&c_none);
            println!(
                "{channels} ch, vbr {vbr}, {name}: trim {:.2}, libopus {:.2} (no mask {:.2}), \
                 the same in {same} of {} frames",
                mean(&rs),
                mean(&c),
                mean(&c_none),
                rs.len()
            );
            assert!(
                same * 100 >= rs.len() * 96,
                "{channels} ch, vbr {vbr}, {name}: trim differs from libopus in {} of {} frames",
                rs.len() - same,
                rs.len()
            );
            shifts.push((name, shift));
        }
        let shift = |wanted: &str| shifts.iter().find(|(name, _)| *name == wanted).unwrap().1;
        // What libopus does, so that the comparison above has a meaning.
        assert_eq!(shift("flat -1"), 0.0);
        assert!((shift("slope +0.04") + 1.0).abs() < 0.3);
        assert!((shift("slope +0.1") + 2.0).abs() < 0.3);
        assert!((shift("slope -0.04") - 1.0).abs() < 0.3);
        assert!((shift("slope -0.1") - 2.0).abs() < 0.3);
        assert!(shift("low bands masked") < -1.5 && shift("high bands masked") > 1.5);
    }
}

/// Bands that are masked much less than the rest get the dynamic
/// allocation boost libopus gives them (`surround_dynalloc`), band by band
/// and frame by frame. In a stereo stream the less masked channel decides.
#[test]
fn unmasked_bands_get_the_boost_libopus_gives_them() {
    const HOLES: [Mask; 6] = [
        ("flat -1.5", |_, _| -1.5),
        (
            "two holes",
            |_, i| if i == 6 || i == 14 { 0.0 } else { -1.5 },
        ),
        ("four holes", |_, i| if i % 5 == 2 { 0.0 } else { -1.5 }),
        ("holes in the left channel", |c, i| {
            if c == 0 && i % 5 == 2 { 0.0 } else { -1.5 }
        }),
        ("holes in the right channel", |c, i| {
            if c == 1 && i % 5 == 2 { 0.0 } else { -1.5 }
        }),
        ("deep holes", |_, i| {
            if i == 3 || i == 9 || i == 16 {
                0.0
            } else {
                -2.0
            }
        }),
    ];
    for (channels, bitrate, vbr) in [
        (1usize, 64_000, false),
        (2, 128_000, false),
        (1, 64_000, true),
        (2, 128_000, true),
    ] {
        let pcm = broadband(50, channels);
        let mut cfg = Cfg::new(channels, bitrate);
        cfg.vbr = vbr;
        let boosts = |cfg: &Cfg| {
            let of = |packets: &[Vec<u8>]| -> Vec<Vec<i32>> {
                headers(packets, channels, 21, 3)
                    .into_iter()
                    .map(|h| h.boost)
                    .collect()
            };
            (of(&enc_rs(cfg, &pcm)), of(&enc_c(cfg, &pcm)))
        };
        // Boost steps per band, summed over the frames.
        let total = |b: &[Vec<i32>]| -> Vec<i32> {
            (0..21).map(|i| b.iter().map(|f| f[i]).sum()).collect()
        };
        let (rs_none, c_none) = boosts(&cfg);
        let mut steps_added = Vec::new();
        for (name, value) in HOLES {
            cfg.energy_mask = Some(mask_values(channels, value));
            let (rs, c) = boosts(&cfg);
            let same = rs.iter().zip(&c).filter(|(a, b)| a == b).count();
            println!(
                "{channels} ch, vbr {vbr}, {name}: boosts the same in {same} of {} frames\n    \
                 opus1 {:?}\n    libopus {:?}",
                rs.len(),
                total(&rs),
                total(&c)
            );
            assert!(
                same * 100 >= rs.len() * 96,
                "{channels} ch, vbr {vbr}, {name}: boosts differ from libopus in {} of {} frames",
                rs.len() - same,
                rs.len()
            );
            let added: i32 = total(&c).iter().sum::<i32>() - total(&c_none).iter().sum::<i32>();
            steps_added.push((name, added));
        }
        assert_eq!(
            total(&rs_none),
            total(&c_none),
            "{channels} ch: boosts without a mask"
        );
        let added = |wanted: &str| {
            steps_added
                .iter()
                .find(|(name, _)| *name == wanted)
                .unwrap()
                .1
        };
        // What libopus does: a flat mask boosts nothing, holes do.
        assert_eq!(added("flat -1.5"), 0);
        assert!(added("two holes") >= 50 && added("deep holes") >= 50);
        if channels == 2 {
            // One open channel is enough for the boost.
            assert!(added("holes in the left channel") >= 100);
            assert_eq!(
                added("holes in the left channel"),
                added("holes in the right channel")
            );
        }
    }
}

/// The multistream encoder gives every frame its own mask. A mask that
/// changes from frame to frame changes each packet as in libopus, which
/// reads the caller's values again on every frame.
#[test]
fn mask_given_frame_by_frame_is_followed_frame_by_frame() {
    let frames = 60;
    for (channels, bitrate) in [(1usize, 128_000), (2, 256_000)] {
        let pcm = broadband(frames, channels);
        // The level swings between -1.4 and 0.2 with a period of 16 frames.
        // The slope turns over every 5 frames, and one band is open in
        // every third frame. The right channel is 0.3 less masked.
        let masks: Vec<Vec<f32>> = (0..frames)
            .map(|k| {
                let level = -0.6 + 0.8 * (k as f32 * std::f32::consts::TAU / 16.0).sin();
                let slope = if (k / 5) % 2 == 0 { 0.03 } else { -0.03 };
                (0..channels)
                    .flat_map(|c| {
                        (0..21).map(move |i| {
                            if k % 3 == 0 && i == 11 {
                                0.0
                            } else {
                                level + slope * (i as f32 - 10.0) + 0.3 * c as f32
                            }
                        })
                    })
                    .collect()
            })
            .collect();
        let mut cfg = Cfg::new(channels, bitrate);
        cfg.constrained_vbr = false;
        let rs_none = packet_sizes(&enc_rs(&cfg, &pcm));
        let c_none = packet_sizes(&enc_c(&cfg, &pcm));
        let rs = packet_sizes(&enc_rs_fmt(&cfg, Format::DEFAULT, &pcm, Some(&masks)));
        let c = packet_sizes(&enc_c_fmt(&cfg, Format::DEFAULT, &pcm, Some(&masks)));
        let (largest, _) = size_effect(&rs_none, &c_none, &rs, &c);
        let c_effect: Vec<i32> = c.iter().zip(&c_none).map(|(a, b)| a - b).collect();
        println!(
            "{channels} ch: libopus packets change by {} to {} bytes, opus1 differs from that \
             by at most {largest} in any frame",
            c_effect.iter().min().unwrap(),
            c_effect.iter().max().unwrap()
        );
        assert!(largest <= 4, "{channels} ch: {largest} bytes");
        // The mask does swing the size from frame to frame.
        let swing = c_effect.iter().max().unwrap() - c_effect.iter().min().unwrap();
        assert!(swing > 80 * channels as i32, "{channels} ch: swing {swing}");

        // A mask that stays in the encoder is the mask of every later
        // frame: setting it once is the same as setting it every frame.
        let constant = vec![masks[3].clone(); frames];
        cfg.energy_mask = Some(masks[3].clone());
        assert_eq!(
            enc_rs(&cfg, &pcm),
            enc_rs_fmt(
                &Cfg {
                    energy_mask: None,
                    ..cfg.clone()
                },
                Format::DEFAULT,
                &pcm,
                Some(&constant)
            )
        );
    }
}

/// The same at 10 ms and 5 ms frames, where the mask covers the same 21
/// bands with fewer coefficients each.
#[test]
fn mask_changes_packet_sizes_as_in_libopus_at_shorter_frames() {
    const MASKS: [Mask; 4] = [
        ("flat -1", |_, _| -1.0),
        ("flat 0", |_, _| 0.0),
        ("slope +0.04", |_, i| -1.2 + 0.04 * i as f32),
        ("four holes", |_, i| if i % 5 == 2 { 0.0 } else { -1.5 }),
    ];
    for fmt in [Format::new(48_000, 20), Format::new(48_000, 10)] {
        let per_20_ms = FRAME / fmt.frame;
        let pcm = broadband(30, 1);
        let mut cfg = Cfg::new(1, 160_000);
        cfg.constrained_vbr = false;
        let rs_none = packet_sizes(&enc_rs_fmt(&cfg, fmt, &pcm, None));
        let c_none = packet_sizes(&enc_c_fmt(&cfg, fmt, &pcm, None));
        for (name, value) in MASKS {
            cfg.energy_mask = Some(mask_values(1, value));
            let rs = packet_sizes(&enc_rs_fmt(&cfg, fmt, &pcm, None));
            let c = packet_sizes(&enc_c_fmt(&cfg, fmt, &pcm, None));
            let (largest, c_mean) = size_effect(&rs_none, &c_none, &rs, &c);
            println!(
                "{} samples, {name}: libopus packets change by {c_mean:+.1} bytes, opus1 \
                 differs from that by at most {largest} in any frame",
                fmt.frame
            );
            assert!(
                largest <= 4,
                "{} samples, {name}: {largest} bytes",
                fmt.frame
            );
            if name == "flat -1" {
                assert!(c_mean < -50.0 / per_20_ms as f64);
            }
        }
    }
}

/// LFE mode ignores the mask under constrained VBR and CBR as well.
/// `celtenc_lfe.rs` checks unconstrained VBR. Under constrained VBR a mask
/// switches off the pull towards the target rate for ordinary streams, but
/// not for LFE (compute_vbr: `(!has_surround_mask||lfe) && constrained_vbr`).
///
/// That pull moves a packet by a byte at most, and only where the frames
/// differ in what they need. "bursts" and the short frames are here for
/// that: with the LFE flag missing in the VBR target, the mask changes 6
/// to 39 of their packets.
#[test]
fn lfe_ignores_the_mask_in_every_rate_mode() {
    for fmt in [Format::DEFAULT, Format::new(FS, 10)] {
        for (name, pcm) in lfe_signals(FS, 2).into_iter().take(3) {
            for (vbr, constrained_vbr) in [(true, true), (true, false), (false, false)] {
                for bytes_per_frame in [8usize, 15, 30, 60] {
                    let mut cfg = Cfg::lfe((bytes_per_frame * 8 * FS / fmt.frame) as i32);
                    cfg.vbr = vbr;
                    cfg.constrained_vbr = constrained_vbr;
                    let rs_none = enc_rs_fmt(&cfg, fmt, &pcm, None);
                    let c_none = enc_c_fmt(&cfg, fmt, &pcm, None);
                    for level in [-2.0f32, 0.0] {
                        cfg.energy_mask = Some(vec![level; 21]);
                        let what = format!(
                            "{name}, {} samples, {bytes_per_frame} bytes, vbr \
                             {vbr}/{constrained_vbr}, mask {level}",
                            fmt.frame
                        );
                        assert!(enc_rs_fmt(&cfg, fmt, &pcm, None) == rs_none, "{what}");
                        assert!(
                            enc_c_fmt(&cfg, fmt, &pcm, None) == c_none,
                            "libopus, {what}"
                        );
                    }
                }
            }
        }
    }
    // The same stream without the LFE flag does change with the mask, so
    // the comparison above is not empty.
    let pcm = lfe_signal(60);
    let mut cfg = Cfg::lfe(16_000);
    cfg.lfe = false;
    let without = enc_rs(&cfg, &pcm);
    cfg.energy_mask = Some(vec![-2.0; 21]);
    assert_ne!(enc_rs(&cfg, &pcm), without);
}

/// The mask has no effect on the CELT part of a Hybrid frame (bands 17 to
/// 20): libopus skips the mask analysis there (celt_encoder.c 2114) and
/// does not use `compute_vbr` for Hybrid.
///
/// A Hybrid frame cannot be made with the C encoder at this level, so this
/// checks opus1 against itself.
#[test]
fn mask_has_no_effect_on_hybrid_frames() {
    let pcm = broadband(20, 1);
    let encode = |mask: Option<&[f32]>, start_band: usize, vbr: bool| -> Vec<Vec<u8>> {
        let mut enc = CeltEncoder::with_sampling_rate(default_mode(), 1, FS as i32);
        enc.complexity = COMPLEXITY;
        enc.set_bitrate(24_000).unwrap();
        enc.set_vbr(vbr);
        enc.set_energy_mask(mask);
        enc.set_start_band(start_band).unwrap();
        pcm.as_chunks::<FRAME>().0.iter()
            .map(|frame| {
                let mut rc = RangeCoder::new_encoder(60);
                enc.encode(frame, FRAME, &mut rc);
                rc.done();
                assert_eq!(rc.error, 0);
                rc.buf[..rc.storage as usize].to_vec()
            })
            .collect()
    };
    // A flat mask acts on the VBR target only. A sloped one also moves the
    // allocation trim, under CBR too.
    let flat = [-2.0f32; 21];
    let sloped: Vec<f32> = (0..21).map(|i| -0.1 * i as f32).collect();
    for vbr in [false, true] {
        let without = encode(None, 17, vbr);
        assert_eq!(
            encode(Some(&flat), 17, vbr),
            without,
            "Hybrid, vbr {vbr}, flat mask"
        );
        assert_eq!(
            encode(Some(&sloped), 17, vbr),
            without,
            "Hybrid, vbr {vbr}, sloped mask"
        );
        // From band 0 the sloped mask does change the frames.
        assert_ne!(
            encode(Some(&sloped), 0, vbr),
            encode(None, 0, vbr),
            "vbr {vbr}"
        );
    }
}
