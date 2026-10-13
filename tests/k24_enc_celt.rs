//! CELT-only coding from the opus-rs encoder at a 24 kHz input rate, decoded
//! by C libopus and compared with libopus's own encoder.

mod k24_enc_common;
use k24_enc_common::*;

fn tone(hz: f64, seconds: usize) -> Vec<f32> {
    (0..seconds * FS)
        .map(|i| (0.4 * (std::f64::consts::TAU * hz * i as f64 / FS as f64).sin()) as f32)
        .collect()
}

/// SNR and worst 20 ms window of a stream, skipping the first half second.
fn quality(pcm: &[f32], packets: &[Vec<u8>], lag: usize) -> (f64, f64) {
    let out = dec_c(packets, FS, 1, 480).unwrap();
    let m = measure_at(&pcm[FS / 2..], &out[FS / 2..], FS, 0, lag);
    (m.snr, m.worst_snr20)
}

/// The comb filter's tap shape ("tapset") must be the one the packet signals.
///
/// CELT's pitch pre-filter is a comb filter with one of three tap shapes.
/// Each frame cross-fades from the previous frame's shape to its own, and
/// the decoder's post-filter repeats the same cross-fade. Two things were
/// wrong (celt_encoder.c:2043, 2343, 2772):
///
/// 1. At the end of a frame the encoder remembered `tapset_decision` as the
///    shape it had used. By then the spreading analysis had already updated
///    that field for the *next* frame, so the next frame cross-faded from a
///    shape the decoder never saw.
/// 2. The high-frequency average behind `tapset_decision` was updated on
///    every 20 ms frame. libopus updates it only on long-block frames that
///    use the comb filter, so the shape changed far more often than it does
///    in libopus.
///
/// From 24 kHz input a tone above about 8.8 kHz hit both: every third frame
/// decoded 14 to 17 dB worse than its neighbours. A steady 9 kHz tone at
/// 128 kb/s measured 31.1 dB (worst 20 ms window 25.2 dB) against libopus's
/// 42.0 dB (41.6 dB), and a 100 Hz - 11 kHz chirp 35.9 dB against 41.6 dB.
///
/// Complexity 5 keeps libopus's tonality analysis off (it starts at
/// complexity 7 and opus-rs does not port it), so both sides run the same
/// algorithms.
#[test]
fn comb_filter_tap_shape_matches_the_decoder() {
    for (app, hz, lag) in [(App::Audio, 9000.0, 156), (App::LowDelay, 9300.0, 60)] {
        let pcm = tone(hz, 2);
        let mut cfg = Cfg::new(app, 128_000);
        cfg.complexity = 5;
        let (rs, rs_worst) = quality(&pcm, &enc_rs(&cfg, &pcm).unwrap(), lag);
        let (c, c_worst) = quality(&pcm, &enc_c(&cfg, &pcm), lag);
        let what = format!(
            "{hz} Hz tone, {}: opus-rs {rs:.1} dB (worst 20 ms {rs_worst:.1}), libopus {c:.1} dB (worst 20 ms {c_worst:.1})",
            app.name()
        );
        assert!(rs >= c - 2.0, "{what}");
        assert!(rs_worst >= c_worst - 3.0, "{what}");
    }

    let pcm = chirp(4 * FS, 100.0, 11_000.0, 0.4);
    let mut cfg = Cfg::new(App::Audio, 128_000);
    cfg.complexity = 5;
    let (rs, _) = quality(&pcm, &enc_rs(&cfg, &pcm).unwrap(), 156);
    let (c, _) = quality(&pcm, &enc_c(&cfg, &pcm), 156);
    assert!(
        rs >= c - 3.0,
        "100 Hz - 11 kHz chirp at 128 kb/s: opus-rs {rs:.1} dB, libopus {c:.1} dB"
    );
}

/// 2.5 ms frames must decode as well as libopus's.
///
/// A 2.5 ms CELT frame has no transient flag, so libopus drops the transient
/// decision for it (`isTransient = 0`, celt_encoder.c:2071-2074). opus-rs
/// kept it and coded the time-frequency resolution as for a transient frame,
/// which the decoder parses as non-transient: every such frame lost sync.
/// 2.5 ms speech at 64 kb/s measured 12-17 dB (worst 100 ms correlation
/// 0.60-0.88) against libopus's 26-27 dB (0.998).
#[test]
fn frames_of_2_5_ms_stay_in_sync() {
    let pcm = speech_24k()[FS..4 * FS].to_vec();
    for app in [App::LowDelay, App::Audio] {
        for bitrate in [64_000, 96_000] {
            let mut cfg = Cfg::new(app, bitrate);
            cfg.frame = 60;
            // Keep libopus in CELT super-wideband, which is what opus-rs codes.
            cfg.c_bandwidth = Some(audiopus_sys::OPUS_BANDWIDTH_SUPERWIDEBAND);
            let q = |packets: &[Vec<u8>]| {
                for p in packets {
                    let t = toc(p);
                    assert_eq!((t.mode, t.bw, t.frame_tenth_ms), ("CELT", "SWB", 25));
                }
                let out = dec_c(packets, FS, 1, cfg.frame).unwrap();
                measure(&pcm, &out, FS, FS / 5)
            };
            let (rs, c) = (q(&enc_rs(&cfg, &pcm).unwrap()), q(&enc_c(&cfg, &pcm)));
            let what = format!(
                "{} {} kb/s 2.5 ms: opus-rs {:.1} dB (worst 100 ms correlation {:.3}), libopus {:.1} dB ({:.3})",
                app.name(),
                bitrate / 1000,
                rs.snr,
                rs.worst_corr,
                c.snr,
                c.worst_corr
            );
            assert_eq!(rs.lag, c.lag, "{what}");
            assert!(rs.snr >= c.snr - 2.0, "{what}");
            assert!(rs.worst_corr >= c.worst_corr - 0.02, "{what}");
        }
    }
}

/// A constant-bitrate packet of two bytes must be one libopus accepts.
///
/// 6 kb/s in 2.5 ms packets is 15 bits a packet: two bytes. Nothing can be
/// coded in that, so the encoder sends a packet with an empty frame, which a
/// decoder conceals, and pads it to the constant size (opus_encoder.c:
/// 1339-1392). At exactly two bytes there is nothing to pad. opus-rs still
/// set the "padding follows" bit in the second byte, with no padding length
/// behind it. libopus's decoder rejects such a packet as corrupt
/// (OPUS_INVALID_PACKET), where it should play a concealed frame.
#[test]
fn two_byte_cbr_packets_are_valid() {
    let pcm = speech_24k()[FS..2 * FS].to_vec();
    for app in [App::LowDelay, App::Voip, App::Audio] {
        let mut cfg = Cfg::new(app, 6_000);
        cfg.frame = 60;
        cfg.cbr = true;
        let rs = enc_rs(&cfg, &pcm).unwrap();
        let c = enc_c(&cfg, &pcm);
        for (k, (p, q)) in rs.iter().zip(&c).enumerate() {
            assert_eq!(p.len(), 2, "{} packet {k}", app.name());
            assert_eq!(q.len(), 2, "{} libopus packet {k}", app.name());
            // Code 3, one frame, no padding: libopus writes the same.
            assert_eq!((p[0] & 3, p[1]), (3, 1), "{} packet {k}", app.name());
            assert_eq!((q[0] & 3, q[1]), (3, 1), "{} libopus packet {k}", app.name());
        }
        // The libopus decoder plays every packet as 2.5 ms.
        dec_c(&rs, FS, 1, 60).unwrap_or_else(|e| panic!("{}: {e}", app.name()));
    }
}

/// What issue #54 left open: from 24 kHz, CELT at 64 kb/s on the issue's
/// chirp measured 36.2 dB against libopus's 38.8 dB.
///
/// The gap is libopus's tonality analysis and nothing else. libopus runs it
/// from complexity 7 up (opus_encoder.c:1252) and no other CELT-only
/// decision changes between complexity 6 and 7. Both streams are CELT-only
/// at the same bitrate, so the gain comes from where the bits go, not from
/// more bits. With the analysis off, opus-rs is level with libopus.
#[test]
fn issue_54_gap_is_the_tonality_analysis() {
    // The issue's chirp: 200 Hz rising at 1200 Hz/s for 1.3 s, amplitude 0.4.
    let n = FS * 13 / 10;
    let mut pcm: Vec<f32> = (0..n)
        .map(|i| {
            let t = i as f64 / FS as f64;
            ((std::f64::consts::TAU * (200.0 * t + 600.0 * t * t)).sin() * 0.4) as f32
        })
        .collect();
    pcm.resize(n.div_ceil(480) * 480 + 480, 0.0);

    let run = |complexity: i32, c_complexity: i32| -> (f64, f64, f64, f64) {
        let mut cfg = Cfg::new(App::Audio, 64_000);
        cfg.complexity = complexity;
        cfg.c_complexity = Some(c_complexity);
        cfg.c_force_mode = Some(MODE_CELT_ONLY);
        let (rs, c) = (enc_rs(&cfg, &pcm).unwrap(), enc_c(&cfg, &pcm));
        for p in rs.iter().chain(&c) {
            assert_eq!(toc(p).mode, "CELT");
        }
        let snr = |p: &[Vec<u8>]| measure(&pcm, &dec_c(p, FS, 1, 480).unwrap(), FS, 0).snr;
        (snr(&rs), snr(&c), mean_bitrate(&rs, 480), mean_bitrate(&c, 480))
    };

    // Analysis off in libopus (complexity 6): opus-rs is not behind.
    let (rs6, c6, rs6_rate, c6_rate) = run(6, 6);
    assert!((rs6_rate - c6_rate).abs() < 1000.0, "bitrates {rs6_rate} / {c6_rate}");
    assert!(
        rs6 >= c6 - 0.5,
        "analysis off: opus-rs {rs6:.2} dB, libopus {c6:.2} dB"
    );
    // Analysis on in libopus (complexity 7): libopus gains about 4 dB at the
    // same bitrate; opus-rs has no analysis and does not move.
    let (rs7, c7, rs7_rate, c7_rate) = run(7, 7);
    assert!((rs7_rate - c7_rate).abs() < 1000.0, "bitrates {rs7_rate} / {c7_rate}");
    assert!((rs7 - rs6).abs() < 0.1, "opus-rs changed at complexity 7: {rs6:.2} -> {rs7:.2}");
    assert!(
        c7 >= c6 + 2.5,
        "libopus's analysis no longer explains the gap: {c6:.2} dB -> {c7:.2} dB"
    );
    // The gap that remains is no larger than what the analysis gives libopus.
    assert!(
        c7 - rs7 <= c7 - c6 + 0.5,
        "opus-rs {rs7:.2} dB is further behind libopus {c7:.2} dB than the analysis explains ({c6:.2} dB without it)"
    );
}
