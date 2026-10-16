//! `CeltEncoder::set_lfe` against C libopus with `OPUS_SET_LFE(1)`.
//!
//! The stream under test is the one the multistream encoder makes of an LFE
//! channel: mono, CELT-only, narrowband, 20 ms, a few kb/s. Both encoders
//! get the same input and the same settings. C libopus decodes both
//! streams.

mod celtenc_common;
use celtenc_common::*;

/// Smallest and largest packet of a stream, in bytes.
fn size_range(packets: &[Vec<u8>]) -> (usize, usize) {
    let sizes = packets.iter().map(|p| p.len());
    (sizes.clone().min().unwrap(), sizes.max().unwrap())
}

/// Bit use per packet and decoded quality are those of libopus, from 3 to
/// 16 kb/s, under constrained VBR (the multistream default) and under CBR.
///
/// Measured on this signal: the packet sizes are the same, and the SNR is
/// within 0.2 dB of libopus at every rate listed.
#[test]
fn lfe_bit_use_and_quality_match_libopus() {
    let pcm = lfe_signal(100);
    for vbr in [true, false] {
        for rate in [3200, 6000, 8000, 12000, 16000] {
            let mut cfg = Cfg::lfe(rate);
            cfg.vbr = vbr;
            let rs = enc_rs(&cfg, &pcm);
            let c = enc_c(&cfg, &pcm);
            let what = format!("LFE {rate} b/s, vbr {vbr}");

            // Both streams say CELT-only, narrowband, mono, one 20 ms frame.
            assert!(rs.iter().chain(&c).all(|p| p[0] == 0x98), "{what}: TOC");

            let (rs_mean, c_mean) = (mean_bytes(&rs, 5), mean_bytes(&c, 5));
            let (rs_snr, c_snr) = (
                snr_db(&pcm, &dec_c(&rs, 1), 1, 5),
                snr_db(&pcm, &dec_c(&c, 1), 1, 5),
            );
            println!(
                "{what}: bytes opus1 {rs_mean:.2} {:?}, libopus {c_mean:.2} {:?}; \
                 SNR opus1 {rs_snr:.2} dB, libopus {c_snr:.2} dB",
                size_range(&rs),
                size_range(&c),
            );
            assert!(
                (rs_mean - c_mean).abs() <= 0.25,
                "{what}: mean packet size {rs_mean:.2} bytes, libopus {c_mean:.2}"
            );
            let (rs_range, c_range) = (size_range(&rs), size_range(&c));
            assert!(
                rs_range.0.abs_diff(c_range.0) <= 1 && rs_range.1.abs_diff(c_range.1) <= 1,
                "{what}: packet sizes {rs_range:?}, libopus {c_range:?}"
            );
            assert!(
                rs_snr >= c_snr - 1.0,
                "{what}: SNR {rs_snr:.2} dB, libopus {c_snr:.2} dB"
            );
        }
    }
}

/// With room to spare, the packet still decodes to the same quality.
///
/// From about 35 bytes on, an LFE frame cannot use all of its packet: the
/// two bands it codes are full. The encoder used to pad the unused room in
/// a way that moved the last raw bits of the frame, and libopus decoded
/// 1 dB of SNR where its own encoder reaches 24 dB.
#[test]
fn lfe_frame_that_does_not_fill_its_packet_decodes() {
    let pcm = lfe_signal(60);
    for rate in [14400, 24000, 64000] {
        let mut cfg = Cfg::lfe(rate);
        cfg.vbr = false;
        let rs_snr = snr_db(&pcm, &dec_c(&enc_rs(&cfg, &pcm), 1), 1, 5);
        let c_snr = snr_db(&pcm, &dec_c(&enc_c(&cfg, &pcm), 1), 1, 5);
        println!("LFE CBR {rate} b/s: SNR opus1 {rs_snr:.2} dB, libopus {c_snr:.2} dB");
        assert!(
            rs_snr >= c_snr - 1.0,
            "LFE CBR {rate} b/s: SNR {rs_snr:.2} dB, libopus {c_snr:.2} dB"
        );
    }
}

/// LFE mode holds the bands from band 2 up (above 400 Hz) down, as libopus
/// does.
///
/// White noise goes in. The level of the decoded signal above 400 Hz,
/// relative to the level below 400 Hz, is the same as with libopus
/// (measured: -12.8 dB for both). Without the flag the same encoder keeps
/// the higher bands (measured: +9.6 dB).
#[test]
fn lfe_holds_down_the_bands_above_band_1_like_libopus() {
    let mut noise = Noise(1);
    let pcm: Vec<f32> = (0..60 * FRAME).map(|_| 0.2 * noise.next()).collect();
    let edges = [0.0, 400.0, 4000.0];
    // Level above 400 Hz relative to the level below it, in dB.
    let high_to_low = |packets: &[Vec<u8>]| {
        let out = dec_c(packets, 1);
        let e = range_energies(&out[5 * FRAME..], &edges);
        10.0 * (e[1] / e[0]).log10()
    };
    let cfg = Cfg::lfe(8000);
    let rs = high_to_low(&enc_rs(&cfg, &pcm));
    let c = high_to_low(&enc_c(&cfg, &pcm));
    let mut plain = cfg.clone();
    plain.lfe = false;
    let rs_plain = high_to_low(&enc_rs(&plain, &pcm));
    println!(
        "above 400 Hz against below: LFE opus1 {rs:.1} dB, LFE libopus {c:.1} dB, \
         opus1 without LFE {rs_plain:.1} dB"
    );
    assert!(
        (rs - c).abs() < 2.0,
        "above 400 Hz: opus1 {rs:.1} dB, libopus {c:.1} dB"
    );
    assert!(
        rs_plain > rs + 15.0,
        "the LFE flag changes too little: {rs_plain:.1} dB without, {rs:.1} dB with"
    );
}

/// The smallest packets an LFE stream can get: 3 to 13 bytes. libopus
/// decodes every one of them, and to about what it decodes from its own
/// encoder's packets of the same size. This covers the packets with 3
/// bytes of CELT data and less, where libopus switches the pitch
/// pre-filter off.
///
/// The SNR means little at these sizes: up to 7 bytes there are no bits for
/// the shape of the spectrum, only for its level, and both encoders are
/// near -3 dB.
#[test]
fn lfe_tiny_packets_decode_like_libopus() {
    let pcm = lfe_signal(50);
    for packet_bytes in 3..=13usize {
        let mut cfg = Cfg::lfe(packet_bytes as i32 * 400);
        cfg.vbr = false;
        assert_eq!(cfg.cbr_packet_bytes(), packet_bytes);
        let rs = enc_rs(&cfg, &pcm);
        let c = enc_c(&cfg, &pcm);
        assert!(rs.iter().chain(&c).all(|p| p.len() == packet_bytes));
        // `dec_c` panics when libopus rejects a packet.
        let (rs_out, c_out) = (dec_c(&rs, 1), dec_c(&c, 1));
        let peak = |x: &[f32]| x.iter().fold(0f32, |m, v| m.max(v.abs()));
        let (rs_snr, c_snr) = (snr_db(&pcm, &rs_out, 1, 5), snr_db(&pcm, &c_out, 1, 5));
        println!(
            "LFE {packet_bytes}-byte packets: SNR opus1 {rs_snr:.2} dB, libopus {c_snr:.2} dB; \
             peak opus1 {:.2}, libopus {:.2}",
            peak(&rs_out),
            peak(&c_out)
        );
        assert!(rs_out.iter().all(|v| v.is_finite()));
        assert!(
            peak(&rs_out) <= 1.5 * peak(&c_out).max(0.5),
            "{packet_bytes}-byte packets: peak {} against {} with libopus",
            peak(&rs_out),
            peak(&c_out)
        );
        // 4 dB, not 1: with 2 to 4 pulses in a band the pulse search of
        // opus1 is worse than libopus's, which costs up to 2.8 dB in the
        // packets of 11 to 13 bytes (see celtenc_known_gaps.rs).
        assert!(
            rs_snr >= c_snr - 4.0,
            "{packet_bytes}-byte packets: SNR {rs_snr:.2} dB, libopus {c_snr:.2} dB"
        );
    }
}

/// An energy mask does not change an LFE stream: libopus skips the surround
/// masking for LFE.
#[test]
fn lfe_ignores_the_energy_mask() {
    let pcm = lfe_signal(40);
    let mut cfg = Cfg::lfe(8000);
    cfg.constrained_vbr = false;
    let without = enc_rs(&cfg, &pcm);
    cfg.energy_mask = Some(vec![-2.0; 21]);
    let with = enc_rs(&cfg, &pcm);
    assert_eq!(without, with);
    // libopus agrees with itself in the same way.
    let c_with = enc_c(&cfg, &pcm);
    cfg.energy_mask = None;
    assert_eq!(enc_c(&cfg, &pcm), c_with);
}
