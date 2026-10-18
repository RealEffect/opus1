//! Hybrid mode (SILK up to 8 kHz plus CELT from 8 to 12 kHz) from the opus1
//! encoder at a 24 kHz input rate, decoded by C libopus. Hybrid is what
//! speech at 24 kHz gets from about 14 kb/s up.

mod k24_enc_common;
use k24_enc_common::*;

use audiopus_sys::OPUS_BANDWIDTH_SUPERWIDEBAND;

/// Both encoders forced to Hybrid super-wideband. Without forcing, both code
/// these signals the same way, partly as SILK wideband or CELT-only: their
/// speech and music analysis finds no audio above 8 kHz in the speech, and
/// moves the higher bitrates to CELT.
fn hybrid_cfg(bitrate: i32, cbr: bool) -> Cfg {
    let mut cfg = Cfg::new(App::Voip, bitrate);
    cfg.cbr = cbr;
    cfg.c_force_mode = Some(MODE_HYBRID);
    cfg.c_bandwidth = Some(OPUS_BANDWIDTH_SUPERWIDEBAND);
    cfg.rs_force_mode = Some(opus1::ForcedMode::Hybrid);
    cfg.rs_bandwidth = Some(opus1::Bandwidth::Superwideband);
    cfg
}

fn assert_all_hybrid(packets: &[Vec<u8>], who: &str) {
    for (k, p) in packets.iter().enumerate() {
        let t = toc(p);
        assert_eq!((t.mode, t.bw), ("HYB", "SWB"), "{who} packet {k}");
    }
}

/// The loudest moment of the 8-12 kHz band, in dB: its energy over every
/// 1024-sample stretch, one per packet.
fn loudest_top_band(pcm: &[f32]) -> f64 {
    let mut worst = f64::NEG_INFINITY;
    let mut a = 0;
    while a + 1024 <= pcm.len() {
        let e = band_energies(&pcm[a..a + 1024], FS, &[(8000.0, 12000.0)])[0];
        worst = worst.max(10.0 * e.max(1e-12).log10());
        a += 480;
    }
    worst
}

/// A Hybrid VBR stream must come out at the bitrate asked for.
///
/// Three things made opus1's Hybrid VBR overshoot, the more the lower the
/// bitrate (16 kb/s came out at 23.9 kb/s, 12 kb/s at 20.4 kb/s, 24 kb/s at
/// 28.1 kb/s; libopus's Hybrid: 16.9, 13.0 and 24.7 kb/s):
///
/// 1. CELT's share had a floor of 8 kb/s. libopus gives CELT what SILK's
///    base rate leaves of the bitrate (opus_encoder.c:2453), which is about
///    2 kb/s at 16 kb/s.
/// 2. CELT sized its decisions by the largest packet the caller allows
///    instead of by its target rate (`effectiveBytes`,
///    celt_encoder.c:1904-1927). In Hybrid VBR that is the caller's whole
///    buffer, so the dynamic allocation boosted the two Hybrid bands as if
///    it had 500 kb/s to spend.
/// 3. SILK's base rate left out libopus's 300 b/s for super-wideband and
///    100 b/s for CBR (compute_silk_rate_for_hybrid, opus_encoder.c:973).
#[test]
fn vbr_stream_keeps_the_requested_bitrate() {
    let wide = speech_fast_24k()[FS..7 * FS].to_vec();
    let narrow = speech_24k()[FS..7 * FS].to_vec();
    for (name, pcm) in [("speech with content to 12 kHz", &wide), ("speech", &narrow)] {
        for bitrate in [14_000, 16_000, 20_000, 24_000, 32_000, 40_000] {
            let cfg = hybrid_cfg(bitrate, false);
            let (rs, c) = (enc_rs(&cfg, pcm).unwrap(), enc_c(&cfg, pcm));
            assert_all_hybrid(&rs, "opus1");
            assert_all_hybrid(&c, "libopus");
            let (rs_rate, c_rate) = (mean_bitrate(&rs, cfg.frame), mean_bitrate(&c, cfg.frame));
            let q = |p: &[Vec<u8>]| measure(pcm, &dec_c(p, FS, 1, cfg.frame).unwrap(), FS, FS / 5);
            let (rs_q, c_q) = (q(&rs), q(&c));
            let what = format!(
                "{name}, {} kb/s: opus1 {:.1} kb/s at {:.2} dB, libopus {:.1} kb/s at {:.2} dB",
                bitrate / 1000,
                rs_rate / 1000.0,
                rs_q.snr,
                c_rate / 1000.0,
                c_q.snr
            );
            assert!((rs_rate - c_rate).abs() <= 0.04 * c_rate, "{what}");
            assert!(rs_q.snr >= c_q.snr - 1.0, "{what}");
            assert!((rs_q.level_db - c_q.level_db).abs() < 0.5, "{what}");
        }
    }
}

/// The case the 24 kHz rate is used for most: mono speech, VoIP, 20 ms,
/// 16-40 kb/s, nothing forced on either side. The stream must not cost
/// noticeably more than was asked for.
#[test]
fn default_voip_stream_keeps_the_requested_bitrate() {
    let pcm = speech_fast_24k()[FS..7 * FS].to_vec();
    for bitrate in [16_000, 20_000, 24_000, 32_000, 40_000] {
        let cfg = Cfg::new(App::Voip, bitrate);
        let rs = enc_rs(&cfg, &pcm).unwrap();
        let rate = mean_bitrate(&rs, cfg.frame);
        assert!(
            rate <= 1.08 * bitrate as f64,
            "{} kb/s asked, {:.1} kb/s coded ({})",
            bitrate / 1000,
            rate / 1000.0,
            toc_summary(&rs)
        );
    }
}

/// A Hybrid packet that SILK nearly fills must still decode in libopus.
///
/// After the SILK data, a Hybrid packet carries a flag that says whether a
/// redundant CELT frame follows. libopus writes it only when the packet has
/// room for the flag and for the redundancy header that could follow, 37
/// bits in all (opus_encoder.c:2351-2355), and its decoder reads it under
/// the same test (opus_decoder.c:509-514). opus1 wrote and read the flag
/// always. Whenever SILK left fewer than 37 bits, libopus's decoder took
/// the flag for the first CELT symbol: the 8-12 kHz band of that frame came
/// out as a burst of noise, up to 35 dB above the input's.
///
/// The speech here has nothing above 8 kHz, so anything loud in the top
/// band is such a burst. CBR at 16-18 kb/s leaves SILK little room.
#[test]
fn packet_nearly_filled_by_silk_stays_in_sync() {
    let pcm = speech_24k()[FS..7 * FS].to_vec();
    for bitrate in [16_000, 17_000, 18_000] {
        let cfg = hybrid_cfg(bitrate, true);
        let (rs, c) = (enc_rs(&cfg, &pcm).unwrap(), enc_c(&cfg, &pcm));
        assert_all_hybrid(&rs, "opus1");
        assert_all_hybrid(&c, "libopus");
        let top = |p: &[Vec<u8>]| loudest_top_band(&dec_c(p, FS, 1, cfg.frame).unwrap());
        let (rs_top, c_top) = (top(&rs), top(&c));
        assert!(
            rs_top <= c_top + 6.0,
            "{} kb/s CBR: loudest 8-12 kHz moment {rs_top:.1} dB from opus1's packets, {c_top:.1} dB from libopus's",
            bitrate / 1000
        );
    }
}
