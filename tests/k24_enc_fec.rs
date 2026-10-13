//! In-band forward error correction (FEC) from the opus-rs encoder at a
//! 24 kHz input rate, decoded by C libopus.
//!
//! With FEC on, each packet also carries a low-bitrate copy of the previous
//! frame (SILK calls it LBRR, low bit-rate redundancy). When a packet is lost
//! the decoder rebuilds its frame from the copy in the next packet.

mod k24_enc_common;
use k24_enc_common::*;

/// 3 s of the speech fixture at 24 kHz.
fn speech() -> Vec<f32> {
    speech_24k()[FS..4 * FS].to_vec()
}

/// FEC that lowers the bandwidth must leave a stream libopus decodes.
///
/// When the bitrate cannot pay for FEC at the current bandwidth, libopus's
/// `decide_fec` lowers the bandwidth for the frame. libopus then turns
/// Hybrid at wideband or below into SILK-only (opus_encoder.c:1675-1695).
/// opus-rs made that mode decision before `decide_fec`, so the frame stayed
/// Hybrid with a wideband or narrower bandwidth: CELT had no band left to
/// code and `interp_bits2pulses` divided by zero. From 24 kHz, VoIP with FEC
/// panicked at 24 kb/s with 10 % loss, and at 16 and 24 kb/s with 20 %.
#[test]
fn fec_that_lowers_the_bandwidth_does_not_panic() {
    let pcm = speech();
    for frame in [480, 960, 1440] {
        for cbr in [false, true] {
            for (bitrate, loss) in [(24_000, 10), (16_000, 20), (24_000, 20), (20_000, 15)] {
                let mut cfg = Cfg::new(App::Voip, bitrate);
                cfg.fec = true;
                cfg.loss = loss;
                cfg.cbr = cbr;
                cfg.frame = frame;
                let what = format!(
                    "{} kb/s, {loss} % loss, {}, {} ms",
                    bitrate / 1000,
                    if cbr { "CBR" } else { "VBR" },
                    frame / 24
                );
                let rs = enc_rs(&cfg, &pcm).unwrap_or_else(|e| panic!("{what}: {e}"));
                let c = enc_c(&cfg, &pcm);
                let q = |p: &[Vec<u8>]| {
                    let out = dec_c(p, FS, 1, frame).unwrap_or_else(|e| panic!("{what}: {e}"));
                    measure(&pcm, &out, FS, FS / 5)
                };
                let (rs_q, c_q) = (q(&rs), q(&c));
                // The level may be off by as much as libopus's own, plus 1 dB.
                let level_limit = c_q.level_db.abs().max(1.0) + 1.0;
                assert!(
                    rs_q.snr >= c_q.snr - 3.0 && rs_q.level_db.abs() < level_limit,
                    "{what}: opus-rs {:.2} dB (level {:+.2} dB, {}), libopus {:.2} dB (level {:+.2} dB, {})",
                    rs_q.snr,
                    rs_q.level_db,
                    toc_summary(&rs),
                    c_q.snr,
                    c_q.level_db,
                    toc_summary(&c)
                );
            }
        }
    }
}

/// The redundant copy of a frame must decode to that frame.
///
/// SILK's delayed-decision quantizer chooses the seed of its winning path
/// (`psIndices->Seed = psDD->SeedInit`, NSQ_del_dec.c), and the decoder
/// derives the signs of the excitation from that seed. The redundant frame
/// was written with the main frame's seed instead of its own, so three
/// redundant frames in four decoded with scrambled signs: recovering a lost
/// frame from FEC was no better than concealing it (-2.3 to +1.2 dB against
/// the loss-free decode, where libopus's encoder gives +6.6 to +7.4 dB).
#[test]
fn redundant_frames_recover_lost_audio_like_libopus() {
    let pcm = speech();
    for cbr in [false, true] {
        for (bitrate, loss) in [(40_000, 10), (48_000, 10), (40_000, 20), (48_000, 20)] {
            let mut cfg = Cfg::new(App::Voip, bitrate);
            cfg.fec = true;
            cfg.loss = loss;
            cfg.cbr = cbr;
            let (rs_fec, rs_plc) = fec_recovery_db(&enc_rs(&cfg, &pcm).unwrap(), 1, cfg.frame);
            let (c_fec, _) = fec_recovery_db(&enc_c(&cfg, &pcm), 1, cfg.frame);
            let what = format!(
                "{} kb/s, {loss} % loss, {}: opus-rs FEC {rs_fec:+.2} dB (concealment {rs_plc:+.2} dB), libopus FEC {c_fec:+.2} dB",
                bitrate / 1000,
                if cbr { "CBR" } else { "VBR" }
            );
            assert!(rs_fec >= rs_plc + 3.0, "FEC recovers no better than concealment: {what}");
            assert!(rs_fec >= c_fec - 2.5, "FEC recovers worse than libopus's: {what}");
        }
    }
}
