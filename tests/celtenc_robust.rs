//! LFE mode and the energy mask on settings and input they were not made
//! for. Nothing may panic, and what C libopus accepts must come out as a
//! packet that C libopus decodes.

mod celtenc_common;
use celtenc_common::*;
use opus_rs::celt::CeltEncoder;
use opus_rs::modes::default_mode;
use opus_rs::range_coder::RangeCoder;

/// One 20 ms test frame per channel, one channel after the other.
fn planar_frame(channels: usize, frame: usize) -> Vec<f32> {
    (0..frame * channels)
        .map(|i| 0.3 * (i as f32 * 0.013).sin())
        .collect()
}

/// Budgets from 0 to 6 bytes, at every frame size, mono and stereo, CBR and
/// VBR, with LFE mode, with a mask, and with both.
///
/// The C CELT encoder takes 2 bytes or more (celt_encoder.c:
/// `nbCompressedBytes<2` is an argument error). From 2 bytes on, opus-rs
/// must write a packet that libopus decodes. Below that it must not panic
/// and must not write more than it was given.
#[test]
fn tiny_budgets_do_not_panic_and_decode() {
    let mask = [-1.0f32; 42];
    let mut decoded = 0;
    for lm in 0..=3usize {
        let fmt = Format {
            fs: FS,
            frame: 120 << lm,
        };
        for channels in [1usize, 2] {
            let pcm = planar_frame(channels, fmt.frame);
            for (lfe, masked) in [(true, false), (false, true), (true, true)] {
                for vbr in [false, true] {
                    for payload in 0..=6usize {
                        let mut enc =
                            CeltEncoder::with_sampling_rate(default_mode(), channels, FS as i32);
                        enc.complexity = COMPLEXITY;
                        enc.set_end_band(13);
                        enc.set_bitrate(8000);
                        enc.set_vbr(vbr);
                        enc.set_lfe(lfe);
                        if masked {
                            enc.set_energy_mask(Some(&mask[..21 * channels]));
                        }
                        let mut packets = Vec::new();
                        for _ in 0..6 {
                            let mut rc = RangeCoder::new_encoder(payload as u32);
                            enc.encode_with_budget_vbr(
                                &pcm,
                                fmt.frame,
                                &mut rc,
                                0,
                                (payload * 8) as i32,
                                vbr,
                            );
                            rc.done();
                            let n = rc.storage as usize;
                            assert!(n <= payload.max(1), "{n} bytes from a budget of {payload}");
                            if payload >= 2 {
                                assert_eq!(rc.error, 0, "budget {payload}: range coder error");
                                let mut packet = vec![celt_toc_lm(13, channels, lm)];
                                packet.extend_from_slice(&rc.buf[..n]);
                                packets.push(packet);
                            }
                        }
                        if !packets.is_empty() {
                            // `dec_c_fmt` panics when libopus rejects a packet.
                            let out = dec_c_fmt(&packets, channels, fmt);
                            assert!(out.iter().all(|v| v.is_finite()));
                            decoded += packets.len();
                        }
                    }
                }
            }
        }
    }
    println!("libopus decoded {decoded} packets");
}

/// `set_energy_mask` takes a slice of any length. Too short is no mask;
/// values after the first `21 * channels` are not read.
#[test]
fn mask_slices_of_any_length() {
    for channels in [1usize, 2] {
        let pcm = broadband(12, channels);
        let mut cfg = Cfg::new(channels, 64_000 * channels as i32);
        cfg.constrained_vbr = false;
        let needed = 21 * channels;
        // Sloped, so that every value matters, and different from the
        // values that follow the first `needed`.
        let values: Vec<f32> = (0..1000)
            .map(|i| {
                if i < needed {
                    -0.05 * (i % 21) as f32
                } else {
                    7.0
                }
            })
            .collect();
        let with = |len: usize| {
            let mut enc = rs_encoder(&cfg);
            enc.set_energy_mask(Some(&values[..len]));
            enc_rs_with(&mut enc, &cfg, &pcm)
        };
        let none = enc_rs(&cfg, &pcm);
        let exact = with(needed);
        assert_ne!(exact, none, "{channels} ch: the mask changes the stream");
        for len in [0, 1, 20, needed - 1] {
            assert_eq!(with(len), none, "{channels} ch: {len} values are no mask");
        }
        for len in [needed + 1, 2 * needed, 1000] {
            assert_eq!(with(len), exact, "{channels} ch: {len} values");
        }
    }
}

/// LFE mode on a stereo encoder, which the multistream encoder never asks
/// for. libopus has no rule against it. Both encoders use the same number
/// of bytes, and their frame headers keep to the LFE rules.
///
/// Measured: for mono the SNR is within 0.1 dB of libopus. For stereo
/// opus-rs is 2.6 dB (constrained VBR) and 5 dB (CBR) above libopus. The
/// reason was not looked for; the test only asks that opus-rs is not worse.
#[test]
fn lfe_on_a_stereo_encoder() {
    let mono = lfe_signal(30);
    for channels in [1usize, 2] {
        // The right channel is the left one, a quarter as loud.
        let pcm: Vec<f32> = mono
            .iter()
            .flat_map(|&v| [v, 0.25 * v].into_iter().take(channels))
            .collect();
        let mut cfg = Cfg::lfe(12_000 * channels as i32);
        cfg.channels = channels;
        for vbr in [true, false] {
            cfg.vbr = vbr;
            let what = format!("{channels} ch, vbr {vbr}");
            let rs = enc_rs(&cfg, &pcm);
            let c = enc_c(&cfg, &pcm);
            // libopus codes an LFE stream narrowband whatever bandwidth is
            // asked for (opus_encoder.c 1683), as `Cfg::lfe` does.
            assert!(rs.iter().zip(&c).all(|(a, b)| a[0] == b[0]), "{what}: TOC");
            let (rs_snr, c_snr) = (
                snr_db(&pcm, &dec_c(&rs, channels), channels, 5),
                snr_db(&pcm, &dec_c(&c, channels), channels, 5),
            );
            println!(
                "{what}: bytes {:.1} / {:.1}, SNR {rs_snr:.2} / {c_snr:.2} dB",
                mean_bytes(&rs, 5),
                mean_bytes(&c, 5)
            );
            assert!(
                (mean_bytes(&rs, 5) - mean_bytes(&c, 5)).abs() <= 0.5,
                "{what}: bytes"
            );
            assert!(
                rs_snr >= c_snr - 1.5,
                "{what}: SNR {rs_snr:.2}, libopus {c_snr:.2}"
            );
            for (who, packets) in [("opus-rs", &rs), ("libopus", &c)] {
                for (k, h) in headers(packets, channels, 13, 3).iter().enumerate() {
                    let frame = format!("{who}, {what}, frame {k}");
                    assert!(!h.transient, "{frame}: transient");
                    assert!(h.tf.iter().all(|&v| v == 0), "{frame}: tf");
                    assert!(h.spread.is_none_or(|s| s == 2), "{frame}: spread");
                    assert!(h.trim.is_none_or(|t| t == 5), "{frame}: trim");
                    assert!(h.boost[1..].iter().all(|&b| b == 0), "{frame}: boost");
                }
            }
        }
    }
}

/// LFE mode with end bands and complexities that libopus never combines
/// with it: libopus always codes an LFE stream narrowband (13 bands). There
/// is nothing to compare with, but nothing may panic, and libopus must
/// decode the packets.
#[test]
fn lfe_with_any_end_band_and_complexity() {
    let pcm = lfe_signal(8);
    for end_band in [1usize, 2, 3, 13, 17, 19, 21] {
        for complexity in [0, 1, 2, 4, 5, 8, 10] {
            for vbr in [true, false] {
                let mut cfg = Cfg::lfe(8000);
                cfg.end_band = end_band;
                cfg.vbr = vbr;
                let mut enc = rs_encoder(&cfg);
                enc.complexity = complexity;
                // The TOC byte knows four bandwidths. For an end band
                // below 13 the decoder expects 13 bands and reads the
                // missing ones from the unused room of the packet.
                let out = dec_c(&enc_rs_with(&mut enc, &cfg, &pcm), 1);
                assert!(
                    out.iter().all(|v| v.is_finite()),
                    "end band {end_band}, complexity {complexity}"
                );
            }
        }
    }
}

/// Samples that are not numbers, infinite or far out of range must not
/// panic the encoder, with LFE mode and with a mask. libopus removes such
/// samples before CELT sees them, so there is nothing to compare.
#[test]
fn non_finite_input_does_not_panic() {
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 1e30, -1e30] {
        for (lfe, masked) in [(true, false), (false, true)] {
            let mut enc = CeltEncoder::with_sampling_rate(default_mode(), 1, FS as i32);
            enc.complexity = COMPLEXITY;
            enc.set_bitrate(32_000);
            enc.set_vbr(true);
            enc.set_lfe(lfe);
            if lfe {
                enc.set_end_band(13);
            }
            if masked {
                enc.set_energy_mask(Some(&[-1.0f32; 21]));
            }
            let mut pcm = planar_frame(1, FRAME);
            for k in 0..6 {
                // Bad samples in the frames 1 and 2, clean frames after.
                pcm[100] = if k == 1 || k == 2 { bad } else { 0.1 };
                let mut rc = RangeCoder::new_encoder(200);
                enc.encode_with_budget_vbr(&pcm, FRAME, &mut rc, 0, 1600, true);
                rc.done();
                assert!(rc.storage <= 200);
            }
        }
    }
}

/// LFE mode switched on and off in the middle of a stream, and a mask that
/// comes and goes. The encoder keeps its state across the switches: every
/// frame decodes, the LFE frames keep to the LFE rules from the first one
/// on, and each stretch of the stream is as good as with libopus, which
/// gets the same switches.
#[test]
fn lfe_and_mask_switched_during_a_stream() {
    let frames = 120;
    let pcm = lfe_signal(frames);
    // 15 frames on, 15 frames off.
    let lfe: Vec<bool> = (0..frames).map(|k| (k / 15) % 2 == 0).collect();
    // A mask in the frames 40 to 79, none before and after. libopus keeps
    // its pointer, so "none" is given to both as a mask that changes
    // nothing for a CBR stream: all zeros has no slope and no holes.
    let masks: Vec<Vec<f32>> = (0..frames)
        .map(|k| {
            if (40..80).contains(&k) {
                (0..21).map(|i| -1.5 + 0.1 * i as f32).collect()
            } else {
                vec![0.0; 21]
            }
        })
        .collect();
    let mut cfg = Cfg::lfe(16_000);
    cfg.vbr = false;
    let rs = enc_rs_schedule(
        &mut rs_encoder(&cfg),
        &cfg,
        Format::DEFAULT,
        &pcm,
        Some(&masks),
        Some(&lfe),
    );
    let c = enc_c_schedule(&cfg, Format::DEFAULT, &pcm, Some(&masks), Some(&lfe));
    let (rs_out, c_out) = (dec_c(&rs, 1), dec_c(&c, 1));

    for (who, packets) in [("opus-rs", &rs), ("libopus", &c)] {
        for (k, h) in headers(packets, 1, 13, 3).iter().enumerate() {
            if lfe[k] {
                assert!(!h.transient, "{who}, frame {k}: transient");
                assert_eq!(h.trim, Some(5), "{who}, frame {k}: trim");
                assert_eq!(h.spread, Some(2), "{who}, frame {k}: spread");
                assert!(h.tf.iter().all(|&v| v == 0), "{who}, frame {k}: tf");
                assert_eq!(h.boost[0], 8, "{who}, frame {k}: boost of band 0");
                assert!(
                    h.boost[1..].iter().all(|&b| b == 0),
                    "{who}, frame {k}: boost"
                );
                assert!(
                    h.coarse[2..].iter().all(|&q| q <= 0),
                    "{who}, frame {k}: energy"
                );
            }
        }
    }
    // The SNR of each stretch of 15 frames, the first one left out.
    for stretch in 1..frames / 15 {
        let range = stretch * 15 * FRAME..(stretch + 1) * 15 * FRAME - CELT_DELAY;
        let snr = |out: &[f32]| {
            let (mut s, mut e) = (0f64, 0f64);
            for i in range.clone() {
                let (r, d) = (f64::from(pcm[i]), f64::from(out[i + CELT_DELAY]));
                s += r * r;
                e += (r - d) * (r - d);
            }
            10.0 * (s / e.max(1e-30)).log10()
        };
        let (rs_snr, c_snr) = (snr(&rs_out), snr(&c_out));
        println!(
            "frames {} to {} (LFE {}): SNR {rs_snr:.2} dB, libopus {c_snr:.2} dB",
            stretch * 15,
            stretch * 15 + 14,
            lfe[stretch * 15]
        );
        assert!(
            rs_snr >= c_snr - 1.5,
            "frames from {}: SNR {rs_snr:.2} dB, libopus {c_snr:.2} dB",
            stretch * 15
        );
    }
}

/// Under constrained VBR a frame may use twice the target rate less what
/// earlier frames took too much. At a rate below 2 bytes per frame that
/// allowance runs out after a few frames. libopus still gives a frame that
/// starts an empty packet 2 bytes (celt_encoder.c 1953: `tell==1?2:0`),
/// the least a CELT frame can have.
///
/// The Opus encoder of libopus does not call CELT at such rates, so this
/// checks opus-rs against that rule and lets libopus decode the result.
#[test]
fn constrained_vbr_gives_an_empty_packet_2_bytes_at_any_rate() {
    for lm in 0..=3usize {
        let fmt = Format {
            fs: FS,
            frame: 120 << lm,
        };
        let pcm = planar_frame(1, fmt.frame);
        for lfe in [true, false] {
            // 1, 1.5 and 2 bytes per frame.
            for bits_per_frame in [8usize, 12, 16] {
                let mut enc = CeltEncoder::with_sampling_rate(default_mode(), 1, FS as i32);
                enc.complexity = COMPLEXITY;
                enc.set_end_band(13);
                enc.set_bitrate((bits_per_frame * FS / fmt.frame) as i32);
                enc.set_vbr(true);
                enc.set_constrained_vbr(true);
                enc.set_lfe(lfe);
                let mut packets = Vec::new();
                for k in 0..40 {
                    let mut rc = RangeCoder::new_encoder(MAX_PAYLOAD as u32);
                    enc.encode_with_budget_vbr(
                        &pcm,
                        fmt.frame,
                        &mut rc,
                        0,
                        (MAX_PAYLOAD * 8) as i32,
                        true,
                    );
                    rc.done();
                    let n = rc.storage as usize;
                    let what = format!(
                        "{} samples, LFE {lfe}, {bits_per_frame} bits per frame, frame {k}",
                        fmt.frame
                    );
                    assert_eq!(rc.error, 0, "{what}: range coder error");
                    assert!(n >= 2, "{what}: {n} bytes");
                    // Once the allowance is used up, every frame gets the
                    // 2 bytes and no more.
                    if k >= 10 {
                        assert_eq!(n, 2, "{what}");
                    }
                    let mut packet = vec![celt_toc_lm(13, 1, lm)];
                    packet.extend_from_slice(&rc.buf[..n]);
                    packets.push(packet);
                }
                let out = dec_c_fmt(&packets, 1, fmt);
                assert!(out.iter().all(|v| v.is_finite()));
            }
        }
    }
}
