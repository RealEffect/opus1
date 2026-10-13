//! What an LFE frame signals, symbol by symbol, against C libopus.
//!
//! The decoded audio shows only the sum of the encoder's decisions. These
//! tests read the decisions themselves from the start of each packet: the
//! transient flag, the time-frequency resolution, the spreading, the
//! dynamic allocation and the allocation trim. libopus fixes all of them in
//! LFE mode (celt_encoder.c, every `st->lfe` condition), so the expected
//! values do not depend on the signal. The same checks run on the packets
//! of the C encoder, which shows that the reading is right.

mod celtenc_common;
use celtenc_common::*;

/// Spreading decision "normal" (libopus `SPREAD_NORMAL`).
const SPREAD_NORMAL: i32 = 2;
/// The first band of a narrowband stream that LFE mode holds down.
const FIRST_HELD_BAND: usize = 2;

/// Checks one LFE frame header against the rules of libopus. `who` names
/// the encoder and the stream in a failure message.
fn check_lfe_header(h: &Header, effective_bytes: usize, lm: usize, who: &str) {
    if h.silence {
        return;
    }
    // celt_encoder.c 2027: no transient analysis.
    assert!(!h.transient, "{who}: transient frame");
    // 2245: no time-frequency analysis, so no band changes its resolution.
    assert!(h.tf.iter().all(|&v| v == 0), "{who}: tf {:?}", h.tf);
    // 2307-2310: normal spreading.
    assert!(
        h.spread.is_none_or(|s| s == SPREAD_NORMAL),
        "{who}: spread {:?}",
        h.spread
    );
    // 1123 and 2354: no dynamic allocation except the fixed boost of band 0.
    assert!(
        h.boost[1..].iter().all(|&b| b == 0),
        "{who}: boost {:?}",
        h.boost
    );
    // 2414: the neutral trim.
    assert!(h.trim.is_none_or(|t| t == 5), "{who}: trim {:?}", h.trim);
    // quant_bands.c 224: the energy of the bands that are held down never
    // steps up.
    assert!(
        h.coarse[FIRST_HELD_BAND..].iter().all(|&q| q <= 0),
        "{who}: coarse energy steps {:?}",
        h.coarse
    );
    // 2354: band 0 asks for min(8, effectiveBytes / 3) boost steps. It gets
    // all of them in a 20 ms frame that is not cut short by the budget. A
    // frame that still had room for the trim was not cut short. In shorter
    // frames the cap of the band can end the boost earlier.
    let asked = (effective_bytes / 3).min(8);
    assert!(h.boost[0] as usize <= asked, "{who}: boost of band 0");
    if h.trim.is_some() && lm == 3 {
        assert_eq!(h.boost[0] as usize, asked, "{who}: boost of band 0");
    }
}

/// Every LFE frame carries the decisions libopus fixes for LFE, at every
/// frame size, at 48, 24 and 16 kHz, under constrained VBR and CBR, from
/// 10 to 45 bytes per frame, on four signals.
#[test]
fn lfe_frame_header_follows_the_rules_of_libopus() {
    let formats = [
        Format::new(48_000, 40),
        Format::new(48_000, 20),
        Format::new(48_000, 10),
        Format::new(48_000, 5),
        Format::new(24_000, 40),
        Format::new(16_000, 20),
    ];
    let mut frames = 0usize;
    let mut frames_with_trim = 0usize;
    for fmt in formats {
        for (name, pcm) in lfe_signals(fmt.fs, 2) {
            for vbr in [true, false] {
                for bytes_per_frame in [10usize, 20, 45] {
                    let bitrate = (bytes_per_frame * 8 * fmt.fs / fmt.frame) as i32;
                    let mut cfg = Cfg::lfe(bitrate);
                    cfg.vbr = vbr;
                    let effective_bytes = if vbr {
                        bytes_per_frame
                    } else {
                        fmt.cbr_packet_bytes(bitrate) - 1
                    };
                    let what = format!(
                        "{name}, {} Hz, {} samples, {bytes_per_frame} bytes, vbr {vbr}",
                        fmt.fs, fmt.frame
                    );
                    let rs = headers(&enc_rs_fmt(&cfg, fmt, &pcm, None), 1, 13, fmt.lm());
                    let c = headers(&enc_c_fmt(&cfg, fmt, &pcm, None), 1, 13, fmt.lm());
                    for (k, (r, c)) in rs.iter().zip(&c).enumerate() {
                        let lm = fmt.lm();
                        check_lfe_header(
                            r,
                            effective_bytes,
                            lm,
                            &format!("opus-rs, {what}, frame {k}"),
                        );
                        check_lfe_header(
                            c,
                            effective_bytes,
                            lm,
                            &format!("libopus, {what}, frame {k}"),
                        );
                        // Both encoders see the same samples, so they find
                        // the same frames silent.
                        assert_eq!(r.silence, c.silence, "{what}, frame {k}: silence flag");
                        frames += 1;
                        if r.trim.is_some() && c.trim.is_some() {
                            // Neither frame was cut short: the same boost.
                            assert_eq!(
                                r.boost[0], c.boost[0],
                                "{what}, frame {k}: boost of band 0"
                            );
                            frames_with_trim += 1;
                        }
                    }
                }
            }
        }
    }
    println!("{frames} frames, {frames_with_trim} with a trim");
    // The checks above must not pass for want of frames to look at.
    assert!(frames_with_trim * 10 > frames * 7);
}

/// An LFE packet with 3 bytes of CELT data or less carries no pitch
/// post-filter (celt_encoder.c 2041). With 4 bytes and more it does, as
/// often as in libopus.
///
/// The signal, 100 Hz and 200 Hz, is so periodic that both encoders switch
/// the filter on in every frame that may have one.
#[test]
fn lfe_post_filter_needs_more_than_3_bytes() {
    let pcm: Vec<f32> = (0..100 * FRAME)
        .map(|i| {
            let t = std::f64::consts::TAU * i as f64 / FS as f64;
            (0.3 * (100.0 * t).sin() + 0.2 * (200.0 * t).sin()) as f32
        })
        .collect();
    for payload_bytes in 2..=8usize {
        let mut cfg = Cfg::lfe((payload_bytes as i32 + 1) * 400);
        cfg.vbr = false;
        let rs = enc_rs(&cfg, &pcm);
        let c = enc_c(&cfg, &pcm);
        assert!(rs.iter().chain(&c).all(|p| p.len() == payload_bytes + 1));
        let on = |packets: &[Vec<u8>]| {
            headers(packets, 1, 13, 3)
                .iter()
                .skip(2)
                .filter(|h| h.postfilter.is_some())
                .count()
        };
        let (rs_on, c_on) = (on(&rs), on(&c));
        println!("{payload_bytes} bytes of CELT data: filter on in {rs_on} frames, libopus {c_on}");
        if payload_bytes <= 3 {
            assert_eq!((rs_on, c_on), (0, 0), "{payload_bytes} bytes");
        } else {
            assert!(c_on >= 95, "{payload_bytes} bytes: libopus {c_on}");
            assert!(
                rs_on.abs_diff(c_on) <= 2,
                "{payload_bytes} bytes: {rs_on}, libopus {c_on}"
            );
        }
    }
}

/// The same rule under constrained VBR, the default of the multistream
/// encoder, where the packet size is not known when the frame starts. What
/// counts is what the packet has left after the constrained-VBR bound
/// (libopus `nbAvailableBytes`, celt_encoder.c 1936-1961), not the size of
/// the caller's buffer.
///
/// At short frames the LFE stream of a surround stream is that small. For
/// 5.1 at 256 kb/s the multistream encoder gives it 3.7 bytes per packet
/// in 2.5 ms frames, and at 96 kb/s 4.3 bytes in 5 ms frames
/// (opus_multistream_encoder.c, `surround_rate_allocation`). Measured
/// before the bound was used: the filter was on in 600 of 799 packets with
/// 3 bytes of CELT data, and in none of libopus's.
#[test]
fn lfe_post_filter_needs_more_than_3_bytes_under_constrained_vbr() {
    let mut small_packets = 0;
    for (fmt, rates) in [
        (Format::new(FS, 5), [9600, 11_200, 12_800, 16_000]),
        (Format::new(FS, 10), [4800, 5600, 6400, 8000]),
    ] {
        let pcm: Vec<f32> = (0..2 * FS)
            .map(|i| {
                let t = std::f64::consts::TAU * i as f64 / FS as f64;
                (0.3 * (100.0 * t).sin() + 0.2 * (200.0 * t).sin()) as f32
            })
            .collect();
        for rate in rates {
            let cfg = Cfg::lfe(rate);
            assert!(cfg.vbr && cfg.constrained_vbr);
            let what = format!("{} samples, {rate} b/s", fmt.frame);
            let rs = enc_rs_fmt(&cfg, fmt, &pcm, None);
            let c = enc_c_fmt(&cfg, fmt, &pcm, None);
            let mut on = [0usize; 2];
            for (who, packets) in [&rs, &c].into_iter().enumerate() {
                for (k, (p, h)) in packets
                    .iter()
                    .zip(headers(packets, 1, 13, fmt.lm()))
                    .enumerate()
                {
                    let filter = h.postfilter.is_some();
                    on[who] += filter as usize;
                    if p.len() - 1 <= 3 {
                        small_packets += 1;
                        assert!(
                            !filter,
                            "{what}, frame {k}, encoder {who}: post-filter in a packet with {} \
                             bytes of CELT data",
                            p.len() - 1
                        );
                    }
                }
            }
            println!(
                "{what}: mean packet {:.2} / {:.2} bytes, filter on in {} / {} of {} frames",
                mean_bytes(&rs, 5),
                mean_bytes(&c, 5),
                on[0],
                on[1],
                rs.len()
            );
            assert!(
                (mean_bytes(&rs, 5) - mean_bytes(&c, 5)).abs() <= 0.1,
                "{what}: bytes"
            );
            // The first frame of a stream is left to chance: the two pitch
            // searches start from different states.
            assert!(
                on[0].abs_diff(on[1]) <= 2,
                "{what}: filter on in {on:?} frames"
            );
        }
    }
    // The rule was put to the test.
    assert!(small_packets > 2000, "{small_packets} small packets");
}
