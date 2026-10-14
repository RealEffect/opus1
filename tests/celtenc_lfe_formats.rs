//! `CeltEncoder::set_lfe` against C libopus at the frame sizes and sampling
//! rates that `celtenc_lfe.rs` leaves out, and on other signals.
//!
//! A surround stream may use any Opus frame size and sampling rate, and the
//! LFE stream follows it. Both encoders get the same input and the same
//! settings; C libopus decodes both streams.

mod celtenc_common;
use celtenc_common::*;

/// Bit use and decoded quality are those of libopus at 2.5, 5, 10 and 20 ms
/// and at 48, 24, 16 and 8 kHz, under constrained VBR and CBR.
///
/// The packets have 20 or 40 bytes. Smaller packets are left to
/// `celtenc_lfe.rs` and `celtenc_known_gaps.rs`: with 2 to 4 pulses in a
/// band the pulse search of opus-rs is worse than libopus's.
///
/// Measured: the mean packet size is within 0.1 byte of libopus, and the
/// SNR within 0.4 dB. Under VBR single packets differ by up to 3 bytes: the
/// two pitch searches do not always pick the same post-filter, and its
/// parameters take a different number of bits.
#[test]
fn lfe_bit_use_and_quality_match_libopus_at_other_frame_sizes_and_rates() {
    let formats = [
        Format::new(48_000, 20),
        Format::new(48_000, 10),
        Format::new(48_000, 5),
        Format::new(24_000, 40),
        Format::new(24_000, 10),
        Format::new(16_000, 20),
        Format::new(8_000, 40),
    ];
    let mut worst = (0f64, 0f64);
    for fmt in formats {
        // The steady tone has its own test below. "white noise" is no LFE
        // signal: LFE mode removes most of it, and the SNR says nothing.
        // `celtenc_lfe.rs` measures its spectrum.
        for (name, pcm) in lfe_signals(fmt.fs, 2) {
            if name != "thumps" && name != "bursts" {
                continue;
            }
            for vbr in [true, false] {
                for bytes_per_frame in [20usize, 40] {
                    let bitrate = (bytes_per_frame * 8 * fmt.fs / fmt.frame) as i32;
                    let mut cfg = Cfg::lfe(bitrate);
                    cfg.vbr = vbr;
                    let what = format!(
                        "{name}, {} Hz, {} samples, {bytes_per_frame} bytes, vbr {vbr}",
                        fmt.fs, fmt.frame
                    );
                    let rs = enc_rs_fmt(&cfg, fmt, &pcm, None);
                    let c = enc_c_fmt(&cfg, fmt, &pcm, None);
                    // The same TOC byte: CELT-only, narrowband, mono, this
                    // frame size.
                    assert!(rs.iter().zip(&c).all(|(a, b)| a[0] == b[0]), "{what}: TOC");

                    let skip = fmt.fs / 10;
                    let snr = |p: &[Vec<u8>]| {
                        snr_db_delay(&pcm, &dec_c_fmt(p, 1, fmt), 1, skip, fmt.delay())
                    };
                    let (rs_snr, c_snr) = (snr(&rs), snr(&c));
                    let (rs_mean, c_mean) = (mean_bytes(&rs, 5), mean_bytes(&c, 5));
                    let largest_difference = rs
                        .iter()
                        .zip(&c)
                        .map(|(a, b)| a.len().abs_diff(b.len()))
                        .max()
                        .unwrap();
                    println!(
                        "{what}: bytes {rs_mean:.2} / {c_mean:.2} (largest difference \
                         {largest_difference}), SNR {rs_snr:.2} / {c_snr:.2} dB"
                    );
                    worst.0 = worst.0.max((rs_mean - c_mean).abs());
                    worst.1 = worst.1.max(c_snr - rs_snr);
                    if vbr {
                        assert!(
                            (rs_mean - c_mean).abs() <= 0.3,
                            "{what}: mean packet size {rs_mean:.2}, libopus {c_mean:.2}"
                        );
                        assert!(largest_difference <= 4, "{what}: packet sizes");
                    } else {
                        assert_eq!(largest_difference, 0, "{what}: packet sizes");
                    }
                    assert!(
                        rs_snr >= c_snr - 1.0,
                        "{what}: SNR {rs_snr:.2} dB, libopus {c_snr:.2} dB"
                    );
                }
            }
        }
    }
    println!(
        "largest difference: {:.2} bytes, {:.2} dB",
        worst.0, worst.1
    );
}

/// Steady tones from 31 to 120 Hz, the content of a sustained bass note, at
/// 5, 10 and 20 ms frames: on average as good as with libopus.
///
/// A steady tone gives nearly the same spectrum in every frame, so a
/// difference between the two encoders repeats in every frame instead of
/// averaging out. That is why the test allows 2 dB for a single tone.
///
/// Measured: every tone is within 0.03 dB of libopus at all three frame
/// sizes. Before `pvq_search` searched a band of 2 coefficients pulse by
/// pulse, as libopus does (it rounded a projection instead, see
/// `celtenc_known_gaps.rs`), a single tone was up to 1.5 dB off in either
/// direction: 50 Hz in 10 ms frames 1.46 dB below libopus, 80 Hz in 20 ms
/// frames 0.81 dB above.
#[test]
fn lfe_steady_tones_match_libopus_on_average() {
    let tones = [
        31.0f64, 37.0, 40.0, 45.0, 50.0, 53.0, 60.0, 67.0, 75.0, 80.0, 93.0, 100.0, 110.0, 120.0,
    ];
    for fmt in [
        Format::new(48_000, 40),
        Format::new(48_000, 20),
        Format::new(48_000, 10),
    ] {
        let mut cfg = Cfg::lfe((30 * 8 * fmt.fs / fmt.frame) as i32);
        cfg.vbr = false;
        let mut line = String::new();
        let mut sum = 0.0;
        for hz in tones {
            let pcm: Vec<f32> = (0..2 * fmt.fs)
                .map(|i| {
                    (0.4 * (std::f64::consts::TAU * hz * i as f64 / fmt.fs as f64).sin()) as f32
                })
                .collect();
            let snr = |packets: &[Vec<u8>]| {
                let out = dec_c_fmt(packets, 1, fmt);
                snr_db_delay(&pcm, &out, 1, fmt.fs / 5, fmt.delay())
            };
            let rs_snr = snr(&enc_rs_fmt(&cfg, fmt, &pcm, None));
            let c_snr = snr(&enc_c_fmt(&cfg, fmt, &pcm, None));
            line += &format!(" {hz} Hz {:+.2}", rs_snr - c_snr);
            sum += rs_snr - c_snr;
            assert!(c_snr > 15.0, "{hz} Hz: libopus reaches {c_snr:.2} dB only");
            assert!(
                rs_snr >= c_snr - 2.0,
                "{} samples, {hz} Hz: SNR {rs_snr:.2} dB, libopus {c_snr:.2} dB",
                fmt.frame
            );
        }
        let mean = sum / tones.len() as f64;
        println!(
            "{} samples, SNR against libopus in dB:{line}; mean {mean:+.2}",
            fmt.frame
        );
        assert!(mean >= -0.5, "{} samples: mean {mean:+.2} dB", fmt.frame);
    }
}

/// At a few kb/s and short frames the VBR target of a frame is below zero:
/// the rate does not even pay for the fixed cost that libopus subtracts
/// (celt_encoder.c 2451: 60 bits for a mono frame). libopus then writes the
/// smallest packet that holds what is coded so far (`min_allowed`).
///
/// opus-rs used to turn the negative target into an unsigned byte count,
/// which gave the largest packet instead: 960 bytes on average, where
/// libopus writes 11, for an LFE stream of 9.6 kb/s in 2.5 ms frames under
/// unconstrained VBR. Under constrained VBR the bound hid it.
///
/// Measured now: the mean packet size is within 0.2 byte of libopus for
/// LFE. Without the LFE flag it is within 1.3 bytes; the largest difference
/// is on "bursts", where the two encoders do not find the same transients.
#[test]
fn vbr_target_below_zero_gives_small_packets_as_in_libopus() {
    for lfe in [true, false] {
        for (fmt, rates) in [
            (Format::new(FS, 5), [9600, 16_000]),
            (Format::new(FS, 10), [4800, 9600]),
            (Format::new(FS, 20), [2400, 4800]),
            (Format::new(FS, 40), [1200, 2400]),
        ] {
            for (name, pcm) in lfe_signals(FS, 2).into_iter().take(3) {
                for rate in rates {
                    let mut cfg = Cfg::lfe(rate);
                    cfg.lfe = lfe;
                    cfg.constrained_vbr = false;
                    let what = format!("LFE {lfe}, {name}, {} samples, {rate} b/s", fmt.frame);
                    let rs = enc_rs_fmt(&cfg, fmt, &pcm, None);
                    let c = enc_c_fmt(&cfg, fmt, &pcm, None);
                    let largest = |p: &[Vec<u8>]| p.iter().map(|x| x.len()).max().unwrap();
                    let (rs_mean, c_mean) = (mean_bytes(&rs, 5), mean_bytes(&c, 5));
                    println!(
                        "{what}: mean {rs_mean:.2} / {c_mean:.2} bytes, largest {} / {}",
                        largest(&rs),
                        largest(&c)
                    );
                    assert!(
                        largest(&c) <= 16,
                        "{what}: libopus writes {} bytes",
                        largest(&c)
                    );
                    assert!(
                        largest(&rs) <= largest(&c) + 4,
                        "{what}: a packet of {} bytes, libopus at most {}",
                        largest(&rs),
                        largest(&c)
                    );
                    let allowed = if lfe { 0.5 } else { 1.5 };
                    assert!(
                        (rs_mean - c_mean).abs() <= allowed,
                        "{what}: mean packet size {rs_mean:.2}, libopus {c_mean:.2}"
                    );
                    // Both streams decode.
                    dec_c_fmt(&rs, 1, fmt);
                }
            }
        }
    }
}
