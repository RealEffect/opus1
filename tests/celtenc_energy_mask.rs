//! `CeltEncoder::set_energy_mask` against C libopus with
//! `OPUS_SET_ENERGY_MASK`.
//!
//! libopus does not export what it computes from the mask, so these tests
//! look at what the mask does to the stream. Both encoders get the same
//! mask, the same input and the same settings: CELT-only, fullband, 20 ms.
//! C libopus decodes both streams.
//!
//! A mask value is in log2 units. 0 means the band is not masked by the
//! other surround channels, -2 means it is deeply masked.

mod celtenc_common;
use celtenc_common::*;

/// One mask value per band, the same for every channel.
fn mask(channels: usize, value: impl Fn(usize) -> f32) -> Option<Vec<f32>> {
    Some((0..channels).flat_map(|_| (0..21).map(&value)).collect())
}

/// Mean packet size of both encoders with `energy_mask`, under unconstrained
/// VBR. Under constrained VBR the rate control pulls every stream back to
/// the same mean, with or without mask.
fn sizes(pcm: &[f32], channels: usize, bitrate: i32, energy_mask: Option<Vec<f32>>) -> (f64, f64) {
    let mut cfg = Cfg::new(channels, bitrate);
    cfg.constrained_vbr = false;
    cfg.energy_mask = energy_mask;
    (
        mean_bytes(&enc_rs(&cfg, pcm), 10),
        mean_bytes(&enc_c(&cfg, pcm), 10),
    )
}

/// The same mask on every band changes the packet size as it does in
/// libopus: the deeper the mask, the smaller the packets.
///
/// Measured, mono 64 kb/s, opus1 / libopus mean bytes: mask -2: 70 / 65,
/// -1: 108 / 105, -0.5: 142 / 144, 0: 190 / 191, +0.5: 202 / 204.
/// Stereo 96 kb/s: 98 / 90, 145 / 149, 213 / 218, 282 / 296, 311 / 316.
///
/// The largest difference, 9 %, is at the deepest mask. There the size
/// depends most on how many bands the previous frame coded, and opus1
/// does not smooth that count over frames as libopus does.
#[test]
fn flat_mask_scales_the_vbr_packet_size_like_libopus() {
    for (channels, bitrate) in [(1usize, 64_000), (2, 96_000)] {
        let pcm = broadband(80, channels);
        let (rs_none, c_none) = sizes(&pcm, channels, bitrate, None);
        let levels = [-2.0f32, -1.0, -0.5, 0.0, 0.5];
        let mut previous = (0.0, 0.0);
        for level in levels {
            let (rs, c) = sizes(&pcm, channels, bitrate, mask(channels, |_| level));
            println!(
                "{channels} ch, {bitrate} b/s, mask {level:+.1}: opus1 {rs:.1} bytes, \
                 libopus {c:.1} (no mask: {rs_none:.1}, {c_none:.1})"
            );
            assert!(
                (rs - c).abs() <= 0.15 * c,
                "{channels} ch, mask {level}: {rs:.1} bytes, libopus {c:.1}"
            );
            // A less masked stream gets more bits, in both encoders.
            assert!(
                rs > previous.0 && c > previous.1,
                "{channels} ch, mask {level}: sizes do not grow with the mask value"
            );
            previous = (rs, c);
        }
        // A deep mask saves more than a third of the bits.
        let (rs_deep, c_deep) = sizes(&pcm, channels, bitrate, mask(channels, |_| -2.0));
        assert!(rs_deep < 0.65 * rs_none && c_deep < 0.65 * c_none);
        // A mask of all zeros is not the same as no mask: it raises the rate.
        let (rs_zero, c_zero) = sizes(&pcm, channels, bitrate, mask(channels, |_| 0.0));
        assert!(rs_zero > 1.1 * rs_none && c_zero > 1.1 * c_none);
    }
}

/// Masks that differ from band to band: the packet size follows libopus.
///
/// "holes" has a few bands that are not masked among bands that are. Those
/// bands get a boost from the dynamic allocation (libopus
/// `surround_dynalloc`).
#[test]
fn shaped_mask_scales_the_vbr_packet_size_like_libopus() {
    type Shape = fn(usize) -> f32;
    let shapes: [(&str, Shape); 5] = [
        ("low bands masked", |i| if i < 10 { -2.0 } else { 0.0 }),
        ("high bands masked", |i| if i < 10 { 0.0 } else { -2.0 }),
        ("rising", |i| -2.0 + 0.1 * i as f32),
        ("falling", |i| -0.1 * i as f32),
        ("holes", |i| if i % 5 == 2 { 0.0 } else { -1.5 }),
    ];
    for (channels, bitrate) in [(1usize, 64_000), (2, 96_000)] {
        let pcm = broadband(80, channels);
        for (name, shape) in shapes {
            let (rs, c) = sizes(&pcm, channels, bitrate, mask(channels, shape));
            println!(
                "{channels} ch, {bitrate} b/s, {name}: opus1 {rs:.1} bytes, libopus {c:.1}"
            );
            assert!(
                (rs - c).abs() <= 0.10 * c,
                "{channels} ch, {name}: {rs:.1} bytes, libopus {c:.1}"
            );
        }
    }
}

/// A mask that slopes over frequency moves bits towards the less masked
/// end of the spectrum, by as much as in libopus (the allocation trim).
///
/// Constrained VBR keeps the packets of both masks at the same size, so
/// the difference in the error per frequency range comes from the split of
/// the bits alone. Measured, mono, SNR below 800 Hz and from 6 to 20 kHz:
/// low bands masked 16.3 and 5.0 dB, high bands masked 20.9 and 1.4 dB;
/// libopus 17.3 and 4.9 dB, 22.4 and 1.3 dB.
#[test]
fn sloped_mask_moves_bits_between_low_and_high_bands_like_libopus() {
    let edges = [0.0, 800.0, 2400.0, 6000.0, 20_000.0];
    for (channels, bitrate) in [(1usize, 64_000), (2, 96_000)] {
        let pcm = broadband(80, channels);
        // Per encoder and mask: SNR in each range, channel 0.
        let measure = |low_masked: bool| {
            let mut cfg = Cfg::new(channels, bitrate);
            cfg.energy_mask = mask(channels, |i| if (i < 10) == low_masked { -2.0 } else { 0.0 });
            let (rs, c) = (enc_rs(&cfg, &pcm), enc_c(&cfg, &pcm));
            assert!((mean_bytes(&rs, 10) - mean_bytes(&c, 10)).abs() < 2.0);
            (
                range_snr_db(&pcm, &dec_c(&rs, channels), channels, 0, 10, &edges),
                range_snr_db(&pcm, &dec_c(&c, channels), channels, 0, 10, &edges),
            )
        };
        let (rs_low, c_low) = measure(true);
        let (rs_high, c_high) = measure(false);
        println!("{channels} ch, low bands masked:  opus1 {rs_low:.1?}, libopus {c_low:.1?}");
        println!("{channels} ch, high bands masked: opus1 {rs_high:.1?}, libopus {c_high:.1?}");

        // The error in each range is close to libopus's.
        for r in 0..4 {
            assert!(
                (rs_low[r] - c_low[r]).abs() < 2.5 && (rs_high[r] - c_high[r]).abs() < 2.5,
                "{channels} ch, range {r}: SNR differs from libopus by 2.5 dB or more"
            );
        }
        // Masking the low bands instead of the high bands costs the lowest
        // range SNR and gives it to the highest range.
        let (bottom, top) = (0, 3);
        let rs_shift = [rs_low[bottom] - rs_high[bottom], rs_low[top] - rs_high[top]];
        let c_shift = [c_low[bottom] - c_high[bottom], c_low[top] - c_high[top]];
        println!("{channels} ch, shift: opus1 {rs_shift:.1?}, libopus {c_shift:.1?}");
        assert!(rs_shift[0] < -3.0 && c_shift[0] < -3.0, "{channels} ch: low range");
        assert!(rs_shift[1] > 0.0 && c_shift[1] > 0.0, "{channels} ch: high range");
        assert!(
            (rs_shift[0] - c_shift[0]).abs() < 1.5 && (rs_shift[1] - c_shift[1]).abs() < 1.5,
            "{channels} ch: bits move by {rs_shift:.1?} dB, in libopus by {c_shift:.1?} dB"
        );
    }
}

/// `None` removes a mask, and a slice that is too short to be a mask does
/// the same. The stream is then the one of an encoder that never had a
/// mask, byte for byte.
#[test]
fn removed_or_short_mask_is_no_mask() {
    let pcm = broadband(30, 2);
    let mut cfg = Cfg::new(2, 96_000);
    cfg.constrained_vbr = false;
    let never = enc_rs(&cfg, &pcm);

    // Set and removed again.
    let mut enc = rs_encoder(&cfg);
    enc.set_energy_mask(Some(&[-2.0f32; 42][..]));
    enc.set_energy_mask(None);
    assert_eq!(enc_rs_with(&mut enc, &cfg, &pcm), never);

    // 41 values are one short of a stereo mask.
    let mut enc = rs_encoder(&cfg);
    enc.set_energy_mask(Some(&[-2.0f32; 42][..]));
    enc.set_energy_mask(Some(&[-2.0f32; 41][..]));
    assert_eq!(enc_rs_with(&mut enc, &cfg, &pcm), never);

    // The mask that was removed above does change the stream.
    let mut enc = rs_encoder(&cfg);
    enc.set_energy_mask(Some(&[-2.0f32; 42][..]));
    assert_ne!(enc_rs_with(&mut enc, &cfg, &pcm), never);
}

/// Mask values that are not numbers, or infinite, or far out of range must
/// not break the encoder: every packet still decodes in libopus.
#[test]
fn unusual_mask_values_do_not_break_the_stream() {
    let pcm = broadband(20, 2);
    let patterns: [fn(usize) -> f32; 5] = [
        |_| f32::NAN,
        |_| f32::INFINITY,
        |_| f32::NEG_INFINITY,
        |i| if i % 2 == 0 { 1e30 } else { -1e30 },
        |i| if i % 3 == 0 { f32::NAN } else { -1.0 },
    ];
    for (k, pattern) in patterns.into_iter().enumerate() {
        for constrained_vbr in [true, false] {
            let mut cfg = Cfg::new(2, 96_000);
            cfg.constrained_vbr = constrained_vbr;
            cfg.energy_mask = mask(2, pattern);
            let packets = enc_rs(&cfg, &pcm);
            let out = dec_c(&packets, 2);
            assert!(out.iter().all(|v| v.is_finite()), "pattern {k}");
            let snr = snr_db(&pcm, &out, 2, 5);
            println!("pattern {k}, constrained {constrained_vbr}: SNR {snr:.1} dB");
            assert!(snr > 3.0, "pattern {k}: SNR {snr:.1} dB");
        }
    }
}
