//! Which mode and audio bandwidth the opus1 encoder picks at a 24 kHz
//! input rate, against C libopus.
//!
//! An Opus packet is SILK-only (speech, up to 8 kHz), CELT-only, or Hybrid
//! (SILK up to 8 kHz plus CELT above). Its bandwidth is narrowband (4 kHz),
//! wideband (8 kHz) or, from 24 kHz input, super-wideband (12 kHz). libopus
//! picks both from the bitrate: a low bitrate is better spent on a narrower
//! band.
//!
//! libopus also runs a tonality analysis from complexity 7 up that detects
//! music and the bandwidth actually present in the input, and can move these
//! choices. opus1 runs the same analysis. Most comparisons here use
//! complexity 5, where both decide from the bitrate alone;
//! `low_bitrates_are_coded_narrower_and_as_well_as_libopus` runs at the
//! default complexity 9.

mod k24_enc_common;
use k24_enc_common::*;
use opus1::Bitrate;

const BITRATES: [i32; 10] = [
    6_000, 8_000, 12_000, 16_000, 24_000, 32_000, 48_000, 64_000, 96_000, 128_000,
];

/// 3 s of the speech-like signal with content up to 12 kHz.
fn wide_speech() -> Vec<f32> {
    speech_fast_24k()[FS..4 * FS].to_vec()
}

/// The share of packets whose mode and bandwidth are the same in both streams.
fn agreement(a: &[Vec<u8>], b: &[Vec<u8>]) -> f64 {
    let same = a
        .iter()
        .zip(b)
        .filter(|(x, y)| {
            let (x, y) = (toc(x), toc(y));
            (x.mode, x.bw, x.frame_tenth_ms, x.frames) == (y.mode, y.bw, y.frame_tenth_ms, y.frames)
        })
        .count();
    same as f64 / a.len().max(1) as f64
}

/// The mode and bandwidth must be libopus's at every bitrate.
///
/// opus1 coded 24 kHz input super-wideband at every bitrate: Hybrid or
/// CELT, never SILK-only, never narrowband or wideband. At 6 and 8 kb/s
/// that spread the bits over 12 kHz where libopus codes 4 kHz: speech
/// measured 0.8 and 2.2 dB against libopus's 3.9 and 5.2 dB while the
/// stream came out at 16-17 kb/s instead of 6.6 and 8.0. The choice now
/// follows libopus's rate thresholds (opus_encoder.c:1583-1650).
#[test]
fn mode_and_bandwidth_are_libopus_s_at_every_bitrate() {
    let pcm = wide_speech();
    for app in [App::Voip, App::Audio, App::LowDelay] {
        for cbr in [false, true] {
            for bitrate in BITRATES {
                let mut cfg = Cfg::new(app, bitrate);
                cfg.complexity = 5;
                cfg.cbr = cbr;
                let (rs, c) = (enc_rs(&cfg, &pcm).unwrap(), enc_c(&cfg, &pcm));
                let share = agreement(&rs, &c);
                assert!(
                    share >= 0.97,
                    "{} {} kb/s {}: {:.0} % of packets agree; opus1 {}; libopus {}",
                    app.name(),
                    bitrate / 1000,
                    if cbr { "CBR" } else { "VBR" },
                    100.0 * share,
                    toc_summary(&rs),
                    toc_summary(&c)
                );
            }
        }
    }
}

/// The same for 10, 40 and 60 ms packets.
#[test]
fn mode_and_bandwidth_are_libopus_s_at_every_packet_duration() {
    let pcm = wide_speech();
    for frame in [240, 960, 1440] {
        for app in [App::Voip, App::Audio] {
            for bitrate in [8_000, 12_000, 16_000, 24_000, 32_000, 64_000, 96_000] {
                let mut cfg = Cfg::new(app, bitrate);
                cfg.complexity = 5;
                cfg.frame = frame;
                let (rs, c) = (enc_rs(&cfg, &pcm).unwrap(), enc_c(&cfg, &pcm));
                let share = agreement(&rs, &c);
                assert!(
                    share >= 0.95,
                    "{} {} kb/s {} ms: {:.0} % of packets agree; opus1 {}; libopus {}",
                    app.name(),
                    bitrate / 1000,
                    frame / 24,
                    100.0 * share,
                    toc_summary(&rs),
                    toc_summary(&c)
                );
            }
        }
    }
}

/// Low bitrates must sound as libopus's do, at the bitrate asked for.
#[cfg(feature = "analysis")]
#[test]
fn low_bitrates_are_coded_narrower_and_as_well_as_libopus() {
    let pcm = speech_24k()[FS..7 * FS].to_vec();
    for app in [App::Voip, App::Audio] {
        for (bitrate, mode) in [
            (6_000, ("SILK", "NB")),
            (8_000, ("SILK", "NB")),
            (10_000, ("SILK", "WB")),
            (12_000, ("SILK", "WB")),
        ] {
            // Complexity 9 on both sides: this is what a caller gets. The
            // fixture has no content above 8 kHz, which both encoders'
            // speech and music analysis finds: at 12 kb/s with `Audio`
            // they code SILK wideband, where without the analysis they
            // code Hybrid.
            let cfg = Cfg::new(app, bitrate);
            let (rs, c) = (enc_rs(&cfg, &pcm).unwrap(), enc_c(&cfg, &pcm));
            assert_eq!(main_mode(&rs), mode, "{} {bitrate}", app.name());
            assert_eq!(main_mode(&rs), main_mode(&c), "{} {bitrate}", app.name());
            let q = |p: &[Vec<u8>]| measure(&pcm, &dec_c(p, FS, 1, cfg.frame).unwrap(), FS, FS / 5);
            let (rs_q, c_q) = (q(&rs), q(&c));
            let (rs_rate, c_rate) = (mean_bitrate(&rs, cfg.frame), mean_bitrate(&c, cfg.frame));
            let what = format!(
                "{} {} kb/s: opus1 {:.1} kb/s at {:.2} dB, libopus {:.1} kb/s at {:.2} dB",
                app.name(),
                bitrate / 1000,
                rs_rate / 1000.0,
                rs_q.snr,
                c_rate / 1000.0,
                c_q.snr
            );
            assert!((rs_rate - c_rate).abs() <= 0.06 * c_rate, "{what}");
            assert!(rs_q.snr >= c_q.snr - 1.0, "{what}");
        }
    }
}

/// When the bitrate changes while SILK is running, the bandwidth must follow
/// as libopus's does.
///
/// A running SILK encoder changes its internal rate one step at a time (8,
/// 12, 16 kHz), and only when its speech activity allows it or 5 s have
/// passed. Going down it first closes a low-pass filter over 2.56 s; going
/// up it opens one over 5.12 s (control_audio_bandwidth.c). Without this, a
/// stream that started at 8 kb/s stayed narrowband for good when the
/// bitrate went to 32 kb/s, and one that started at 32 kb/s kept coding
/// 16 kHz at 8 kb/s.
#[test]
fn bandwidth_follows_a_bitrate_change() {
    let all = speech_fast_24k();
    let pcm = &all[..12 * FS];
    for schedule in [[8_000, 32_000, 32_000], [32_000, 8_000, 8_000], [8_000, 12_000, 20_000]] {
        let mut cfg = Cfg::new(App::Voip, schedule[0]);
        cfg.complexity = 5;
        let mut rs_enc = rs_encoder(&cfg);
        let mut c_enc = CEncoder::new(&cfg);
        let (mut rs, mut c) = (Vec::new(), Vec::new());
        for (k, &bitrate) in schedule.iter().enumerate() {
            let part = &pcm[k * 4 * FS..(k + 1) * 4 * FS];
            rs_enc.set_bitrate(Bitrate::BitsPerSecond(bitrate)).unwrap();
            c_enc.ctl(audiopus_sys::OPUS_SET_BITRATE_REQUEST, bitrate);
            rs.extend(enc_rs_with(&mut rs_enc, &cfg, part).unwrap());
            c.extend(c_enc.encode(&cfg, part));
        }
        let what = format!("bitrate per 4 s {schedule:?}");

        // Second by second, the streams are in the same mode and bandwidth.
        let mut same_seconds = 0;
        for s in 0..12 {
            let (a, b) = (&rs[s * 50..(s + 1) * 50], &c[s * 50..(s + 1) * 50]);
            same_seconds += (main_mode(a) == main_mode(b)) as usize;
        }
        assert!(
            same_seconds >= 11,
            "{what}: only {same_seconds} of 12 seconds agree; opus1 {}; libopus {}",
            toc_summary(&rs),
            toc_summary(&c)
        );
        // Both end where the last bitrate belongs.
        assert_eq!(main_mode(&rs[550..]), main_mode(&c[550..]), "{what}");

        // The switches are no rougher than libopus's.
        let q = |p: &[Vec<u8>]| measure(pcm, &dec_c(p, FS, 1, cfg.frame).unwrap(), FS, FS / 5);
        let (rs_q, c_q) = (q(&rs), q(&c));
        assert!(
            rs_q.snr >= c_q.snr - 1.0 && rs_q.worst_snr20 >= c_q.worst_snr20 - 3.0,
            "{what}: opus1 {:.2} dB (worst 20 ms {:.2}), libopus {:.2} dB (worst 20 ms {:.2})",
            rs_q.snr,
            rs_q.worst_snr20,
            c_q.snr,
            c_q.worst_snr20
        );
    }
}

/// CELT-only frames end a SILK rate switch that was under way.
///
/// At 8 kb/s after 32 kb/s, SILK starts closing its low-pass filter to go
/// down a rate. If the bitrate then jumps to CELT-only and comes back to
/// 32 kb/s, SILK restarts from scratch in libopus (silk_InitEncoder). A
/// filter left half closed would muffle the speech and hold the stream at
/// wideband SILK instead of Hybrid.
#[test]
fn celt_only_frames_end_a_rate_switch() {
    let all = speech_fast_24k();
    let pcm = &all[..12 * FS];
    let schedule = [32_000, 8_000, 96_000, 32_000];
    let mut cfg = Cfg::new(App::Voip, schedule[0]);
    cfg.complexity = 5;
    let mut rs_enc = rs_encoder(&cfg);
    let mut c_enc = CEncoder::new(&cfg);
    let (mut rs, mut c) = (Vec::new(), Vec::new());
    for (k, &bitrate) in schedule.iter().enumerate() {
        let part = &pcm[k * 3 * FS..(k + 1) * 3 * FS];
        rs_enc.set_bitrate(Bitrate::BitsPerSecond(bitrate)).unwrap();
        c_enc.ctl(audiopus_sys::OPUS_SET_BITRATE_REQUEST, bitrate);
        rs.extend(enc_rs_with(&mut rs_enc, &cfg, part).unwrap());
        c.extend(c_enc.encode(&cfg, part));
    }
    // The last 3 s at 32 kb/s, leaving out the first second after the switch.
    let (rs_end, c_end) = (&rs[500..], &c[500..]);
    assert!(
        rs_end.iter().all(|p| toc(p).mode == "HYB") && c_end.iter().all(|p| toc(p).mode == "HYB"),
        "back at 32 kb/s: opus1 {}; libopus {}",
        toc_summary(rs_end),
        toc_summary(c_end)
    );
    let top = |p: &[Vec<u8>]| {
        let out = dec_c(p, FS, 1, cfg.frame).unwrap();
        let lag = best_lag(pcm, &out, FS / 25);
        (
            band_gain_db(&pcm[10 * FS..], &out[10 * FS + lag..], FS, 4000.0, 8000.0),
            band_gain_db(&pcm[10 * FS..], &out[10 * FS + lag..], FS, 8000.0, 12000.0),
        )
    };
    let ((rs_mid, rs_top), (c_mid, c_top)) = (top(&rs), top(&c));
    assert!(
        (rs_mid - c_mid).abs() < 1.5 && (rs_top - c_top).abs() < 1.5,
        "back at 32 kb/s, decoded over input: 4-8 kHz {rs_mid:+.1} dB (libopus {c_mid:+.1}), 8-12 kHz {rs_top:+.1} dB (libopus {c_top:+.1})"
    );
}

/// SILK-only packets that follow CELT-only packets restart SILK once.
///
/// When SILK takes over from CELT-only, libopus restarts the SILK encoder and
/// feeds it the last 10 ms of input (opus_encoder.c:1575-1580, 2191-2210).
/// It does so once: `prev_mode` is set at the end of every coded frame
/// (opus_encoder.c:2555-2558). opus1 did not record the mode of a
/// variable-bitrate SILK-only packet, so the mode before it stayed
/// "CELT-only" and SILK restarted on every packet from then on. The speech
/// below at 12 kb/s decoded at 0.3 dB (worst 20 ms window -6.9 dB) and cost
/// 12.6 kb/s, where libopus's stream gives 6.4 dB (+1.0 dB) at 12.2 kb/s.
///
/// The stream reaches SILK-only straight from CELT-only in two ways here:
///
/// 1. The bitrate goes 12 -> 96 -> 12 kb/s and the first change falls in a
///    pause. SILK then still allows a bandwidth change when it comes back.
/// 2. The bitrate stays at 12 kb/s and the packets go 20 -> 5 -> 20 ms. 5 ms
///    packets are CELT-only and wideband at this bitrate.
#[test]
fn silk_only_after_celt_only_restarts_once() {
    let speech = speech_24k();
    // 2 s of speech, 1 s of silence, 6 s of speech.
    let mut pcm = speech[FS..3 * FS].to_vec();
    pcm.resize(3 * FS, 0.0);
    pcm.extend_from_slice(&speech[3 * FS..9 * FS]);
    for (high_bitrate, middle_frame) in [(96_000, 480), (12_000, 120)] {
        // (first sample, last sample, bitrate, samples per packet)
        let parts = [
            (0, 5 * FS / 2, 12_000, 480),
            (5 * FS / 2, 7 * FS / 2, high_bitrate, middle_frame),
            (7 * FS / 2, 9 * FS, 12_000, 480),
        ];
        let mut cfg = Cfg::new(App::Voip, 12_000);
        cfg.complexity = 5;
        let mut rs_enc = rs_encoder(&cfg);
        let mut c_enc = CEncoder::new(&cfg);
        let (mut rs, mut c) = (Vec::new(), Vec::new());
        let (mut rs_last, mut c_last) = (Vec::new(), Vec::new());
        for &(from, to, bitrate, frame) in &parts {
            cfg.frame = frame;
            rs_enc.set_bitrate(Bitrate::BitsPerSecond(bitrate)).unwrap();
            c_enc.ctl(audiopus_sys::OPUS_SET_BITRATE_REQUEST, bitrate);
            rs_last = enc_rs_with(&mut rs_enc, &cfg, &pcm[from..to]).unwrap();
            c_last = c_enc.encode(&cfg, &pcm[from..to]);
            rs.extend(rs_last.iter().cloned());
            c.extend(c_last.iter().cloned());
        }
        let what = format!(
            "12 kb/s 20 ms, then {} kb/s {} ms, then 12 kb/s 20 ms",
            high_bitrate / 1000,
            middle_frame / 24
        );
        assert_eq!(main_mode(&rs_last), ("SILK", "WB"), "{what}: {}", toc_summary(&rs_last));
        assert_eq!(main_mode(&c_last), ("SILK", "WB"), "{what}: {}", toc_summary(&c_last));
        // The last 4.5 s, one second after SILK took over again.
        let from = 9 * FS / 2;
        let q = |packets: &[Vec<u8>]| {
            let out = dec_c_any(packets);
            let lag = best_lag(&pcm[from..], &out[from..], FS / 25);
            measure_at(&pcm[from..], &out[from..], FS, 0, lag)
        };
        let (rs_q, c_q) = (q(&rs), q(&c));
        let (rs_rate, c_rate) = (mean_bitrate(&rs_last, 480), mean_bitrate(&c_last, 480));
        let numbers = format!(
            "{what}: opus1 {:.1} kb/s at {:.2} dB (worst 20 ms {:.2}), libopus {:.1} kb/s at {:.2} dB (worst 20 ms {:.2})",
            rs_rate / 1000.0,
            rs_q.snr,
            rs_q.worst_snr20,
            c_rate / 1000.0,
            c_q.snr,
            c_q.worst_snr20
        );
        assert!(rs_q.snr >= c_q.snr - 1.0, "{numbers}");
        assert!(rs_q.worst_snr20 >= c_q.worst_snr20 - 4.0, "{numbers}");
        assert!((rs_rate - c_rate).abs() <= 0.06 * c_rate, "{numbers}");
    }
}

/// Constant-bitrate packets of 40 and 60 ms have the constant size.
///
/// CELT and Hybrid frames are 20 ms at most, so such a packet carries two or
/// three 20 ms parts. libopus sizes the parts so that they and the packet
/// header fill the constant size exactly, padding the rest
/// (opus_encoder.c:1735-1840, repacketizer.c). opus1 coded each part at
/// the full 20 ms size and put the header on top: a 60 ms packet came out
/// 1 byte over the constant size (121 bytes at 16 kb/s), and 3 bytes over
/// when a part reached 252 bytes (963 bytes at 128 kb/s).
#[test]
fn cbr_packets_of_40_and_60_ms_have_the_constant_size() {
    let pcm = wide_speech();
    for frame in [960, 1440] {
        // Hybrid, then CELT-only with parts below and above 252 bytes.
        for (app, bitrate) in [
            (App::Voip, 16_000),
            (App::Voip, 24_000),
            (App::Voip, 40_000),
            (App::LowDelay, 24_000),
            (App::LowDelay, 64_000),
            (App::LowDelay, 128_000),
        ] {
            let mut cfg = Cfg::new(app, bitrate);
            cfg.complexity = 5;
            cfg.frame = frame;
            cfg.cbr = true;
            let (rs, c) = (enc_rs(&cfg, &pcm).unwrap(), enc_c(&cfg, &pcm));
            let what = format!("{} {} kb/s {} ms", app.name(), bitrate / 1000, frame / 24);
            assert!(
                agreement(&rs, &c) >= 0.95,
                "{what}: {}; libopus {}",
                toc_summary(&rs),
                toc_summary(&c)
            );
            let nominal = bitrate as usize * frame / FS / 8;
            assert_eq!(size_range(&c), (nominal, nominal), "{what}: libopus");
            assert_eq!(size_range(&rs), (nominal, nominal), "{what}: opus1");
            let q = |p: &[Vec<u8>]| measure(&pcm, &dec_c(p, FS, 1, frame).unwrap(), FS, FS / 5);
            let (rs_q, c_q) = (q(&rs), q(&c));
            assert!(
                rs_q.snr >= c_q.snr - 1.0,
                "{what}: opus1 {:.2} dB, libopus {:.2} dB",
                rs_q.snr,
                c_q.snr
            );
        }
    }
}

/// A variable-bitrate packet of 40 or 60 ms fits the caller's buffer.
///
/// Each 20 ms part was coded as if it had 1276 bytes to itself, whatever
/// the size of the caller's buffer. A buffer of the nominal packet size,
/// which libopus fills without complaint, then gave "Output buffer too
/// small" as soon as the parts together needed more.
///
/// The bitrate itself is not compared with libopus here. libopus 1.6 limits
/// each part to the size the bitrate gives 20 ms under variable bitrate too,
/// and so comes out 8 to 11 % under the bitrate asked for. opus1 does not
/// port that limit; see `encode_multiframe`.
#[test]
fn vbr_packets_of_40_and_60_ms_fit_the_buffer() {
    let pcm = wide_speech();
    for frame in [960, 1440] {
        for bitrate in [16_000, 24_000, 40_000] {
            let nominal = bitrate as usize * frame / FS / 8;
            let cfg = Cfg::new(App::Voip, bitrate);
            let mut enc = rs_encoder(&cfg);
            let mut buf = vec![0u8; nominal];
            let mut packets = Vec::new();
            for (k, chunk) in pcm.chunks_exact(frame).enumerate() {
                let n = enc.encode(chunk, frame, &mut buf).unwrap_or_else(|e| {
                    panic!("{} kb/s {} ms, packet {k}: {e}", bitrate / 1000, frame / 24)
                });
                packets.push(buf[..n].to_vec());
            }
            let out = dec_c(&packets, FS, 1, frame).unwrap();
            let m = measure(&pcm, &out, FS, FS / 5);
            assert!(
                m.snr > 6.0,
                "{} kb/s {} ms in a {nominal}-byte buffer: {:.2} dB ({})",
                bitrate / 1000,
                frame / 24,
                m.snr,
                toc_summary(&packets)
            );
        }
    }
}

/// 2.5 and 5 ms packets are CELT-only whatever the application.
///
/// SILK frames are 10 ms or longer, so libopus switches to CELT-only for
/// shorter packets (opus_encoder.c:1532-1537). opus1 kept SILK or Hybrid
/// and failed with "Unsupported frame size for Hybrid mode" for VoIP at any
/// bitrate up to about 64 kb/s and for Audio up to about 24 kb/s.
#[test]
fn short_packets_are_celt_only() {
    let pcm = wide_speech();
    for frame in [60, 120] {
        for app in [App::Voip, App::Audio] {
            for bitrate in [12_000, 24_000, 32_000, 64_000] {
                let mut cfg = Cfg::new(app, bitrate);
                cfg.complexity = 5;
                cfg.frame = frame;
                let what = format!("{} {} kb/s {} ms", app.name(), bitrate / 1000, frame as f32 / 24.0);
                let rs = enc_rs(&cfg, &pcm).unwrap_or_else(|e| panic!("{what}: {e}"));
                let c = enc_c(&cfg, &pcm);
                assert!(rs.iter().all(|p| toc(p).mode == "CELT"), "{what}: {}", toc_summary(&rs));
                let share = agreement(&rs, &c);
                assert!(
                    share >= 0.95,
                    "{what}: {:.0} % of packets agree; opus1 {}; libopus {}",
                    100.0 * share,
                    toc_summary(&rs),
                    toc_summary(&c)
                );
                dec_c(&rs, FS, 1, frame).unwrap_or_else(|e| panic!("{what}: {e}"));
            }
        }
    }
}

/// With FEC on and loss declared, speech stays in SILK or Hybrid.
///
/// Only SILK carries in-band FEC, so libopus keeps SILK when FEC is on and
/// the declared loss is high enough for the application
/// (opus_encoder.c:1515-1518). opus1 went CELT-only from about 64 kb/s
/// (VoIP) or 24 kb/s (Audio) and sent no FEC at all there.
#[test]
fn fec_keeps_the_stream_in_silk_or_hybrid() {
    let pcm = wide_speech();
    for (app, loss) in [(App::Voip, 5), (App::Voip, 20), (App::Audio, 10), (App::Audio, 20)] {
        for bitrate in [32_000, 64_000, 96_000] {
            let mut cfg = Cfg::new(app, bitrate);
            cfg.complexity = 5;
            cfg.fec = true;
            cfg.loss = loss;
            let (rs, c) = (enc_rs(&cfg, &pcm).unwrap(), enc_c(&cfg, &pcm));
            let what = format!(
                "{} {} kb/s, {loss} % loss: opus1 {}; libopus {}",
                app.name(),
                bitrate / 1000,
                toc_summary(&rs),
                toc_summary(&c)
            );
            assert!(rs.iter().all(|p| toc(p).mode != "CELT"), "{what}");
            assert!(agreement(&rs, &c) >= 0.95, "{what}");
            let (rs_fec, rs_plc) = fec_recovery_db(&rs, 1, cfg.frame);
            let (c_fec, _) = fec_recovery_db(&c, 1, cfg.frame);
            assert!(
                rs_fec >= rs_plc + 3.0 && rs_fec >= c_fec - 2.5,
                "{what}: FEC recovers {rs_fec:+.2} dB (concealment {rs_plc:+.2} dB), libopus {c_fec:+.2} dB"
            );
        }
    }
}
