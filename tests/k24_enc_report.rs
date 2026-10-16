//! Measurement tables for the opus1 encoder at a 24 kHz input rate, next to
//! C libopus under the same settings. Both streams are decoded by C libopus.
//!
//! These print tables and assert nothing, so they are `#[ignore]`d. Run one
//! with, for example:
//!
//! ```text
//! cargo test --release --test k24_enc_report -- --ignored --nocapture modes
//! ```

mod k24_enc_common;
use k24_enc_common::*;
use opus1::Bitrate;

/// One table row: the stream of each encoder decoded at `rate`.
fn row(label: &str, cfg: &Cfg, pcm: &[f32], rate: usize) -> String {
    let ch = cfg.channels;
    let reference_24k = channel(pcm, ch, 0);
    let reference = if rate == FS {
        reference_24k.clone()
    } else {
        resample(&reference_24k, rate / FS, 1)
    };
    let skip = rate / 5; // leave the first 200 ms out of the whole-signal numbers
    let side = |packets: Result<Vec<Vec<u8>>, String>| -> String {
        let packets = match packets {
            Ok(p) => p,
            Err(e) => return format!("ENCODE ERROR {e}"),
        };
        let out = match dec_c(&packets, rate, ch, cfg.frame) {
            Ok(o) => o,
            Err(e) => return format!("DECODE ERROR {e}"),
        };
        let m = measure(&reference, &channel(&out, ch, 0), rate, skip);
        let (lo, hi) = size_range(&packets);
        format!(
            "{:>6.1}kb {:>3}-{:<4} lag{:>4} snr{:>6.2} wsnr{:>7.2} wcorr{:>6.3} lvl{:>+6.2} | {}",
            mean_bitrate(&packets, cfg.frame) / 1000.0,
            lo,
            hi,
            m.lag,
            m.snr,
            m.worst_snr20,
            m.worst_corr,
            m.level_db,
            toc_summary(&packets)
        )
    };
    let rs = side(enc_rs(cfg, pcm));
    let c = side(Ok(enc_c(cfg, pcm)));
    format!("{label}\n    rs: {rs}\n    c : {c}")
}

fn speech_segment(seconds: usize) -> Vec<f32> {
    let s = speech_24k();
    s[FS..FS + seconds * FS].to_vec()
}

fn speech_fast_segment(seconds: usize) -> Vec<f32> {
    let s = speech_fast_24k();
    s[FS..FS + seconds * FS].to_vec()
}

const BITRATES: [i32; 10] = [
    6_000, 8_000, 12_000, 16_000, 24_000, 32_000, 48_000, 64_000, 96_000, 128_000,
];

/// Mode, bandwidth, packet size and quality at every bitrate, mono, 20 ms.
#[test]
#[ignore = "report: prints a table"]
fn modes() {
    let speech = speech_segment(6);
    for app in [App::Voip, App::Audio, App::LowDelay] {
        for cbr in [false, true] {
            println!("== {} mono 20 ms {} speech, decoded at 24 kHz", app.name(), if cbr { "CBR" } else { "VBR" });
            for b in BITRATES {
                let mut cfg = Cfg::new(app, b);
                cfg.cbr = cbr;
                println!("{}", row(&format!("{:>3} kb/s", b / 1000), &cfg, &speech, FS));
            }
        }
    }
}

/// The same streams decoded by libopus at 48 kHz instead of 24 kHz, against
/// the input resampled to 48 kHz.
#[test]
#[ignore = "report: prints a table"]
fn modes_decoded_at_48k() {
    let speech = speech_fast_segment(4);
    for app in [App::Voip, App::Audio, App::LowDelay] {
        println!(
            "== {} mono 20 ms VBR, speech with content to 12 kHz, decoded at 48 kHz",
            app.name()
        );
        for b in BITRATES {
            let cfg = Cfg::new(app, b);
            println!("{}", row(&format!("{:>3} kb/s", b / 1000), &cfg, &speech, 48_000));
        }
    }
}

/// The same with the speech-like signal that has content up to 12 kHz, and
/// how much of the 8-12 kHz band survives.
#[test]
#[ignore = "report: prints a table"]
fn modes_wide_speech() {
    let speech = speech_fast_segment(6);
    for app in [App::Voip, App::Audio] {
        println!("== {} mono 20 ms VBR, speech with content to 12 kHz, decoded at 24 kHz", app.name());
        for b in BITRATES {
            let cfg = Cfg::new(app, b);
            println!("{}", row(&format!("{:>3} kb/s", b / 1000), &cfg, &speech, FS));
            let bands = |packets: &[Vec<u8>]| -> String {
                let out = dec_c(packets, FS, 1, cfg.frame).unwrap();
                let lag = best_lag(&speech, &out, FS / 25);
                let o = &out[lag..];
                format!(
                    "0-4k {:+.1} dB, 4-8k {:+.1} dB, 8-12k {:+.1} dB",
                    band_gain_db(&speech, o, FS, 50.0, 4000.0),
                    band_gain_db(&speech, o, FS, 4000.0, 8000.0),
                    band_gain_db(&speech, o, FS, 8000.0, 12000.0)
                )
            };
            match enc_rs(&cfg, &speech) {
                Ok(p) => println!("    rs bands: {}", bands(&p)),
                Err(e) => println!("    rs bands: {e}"),
            }
            println!("    c  bands: {}", bands(&enc_c(&cfg, &speech)));
        }
    }
}

/// Decoded energy over input energy in three bands, as text.
fn bands_text(reference: &[f32], packets: &[Vec<u8>], frame: usize) -> String {
    let out = match dec_c(packets, FS, 1, frame) {
        Ok(o) => o,
        Err(e) => return e,
    };
    let lag = best_lag(reference, &out, FS / 25);
    let o = &out[lag..];
    format!(
        "0-4k {:+.1} dB, 4-8k {:+.1} dB, 8-12k {:+.1} dB",
        band_gain_db(reference, o, FS, 50.0, 4000.0),
        band_gain_db(reference, o, FS, 4000.0, 8000.0),
        band_gain_db(reference, o, FS, 8000.0, 12000.0)
    )
}

/// Hybrid super-wideband against Hybrid super-wideband: libopus is forced to
/// the mode and bandwidth opus1 uses, so the coding itself is compared.
#[test]
#[ignore = "report: prints a table"]
fn hybrid_like_for_like() {
    for (name, speech) in [
        ("speech (to 8 kHz)", speech_segment(6)),
        ("fast speech (to 12 kHz)", speech_fast_segment(6)),
    ] {
        for cbr in [false, true] {
            println!(
                "== voip mono 20 ms {} {name}; libopus forced to Hybrid SWB",
                if cbr { "CBR" } else { "VBR" }
            );
            for b in [12_000, 16_000, 20_000, 24_000, 32_000, 40_000, 48_000] {
                let mut cfg = Cfg::new(App::Voip, b);
                cfg.cbr = cbr;
                cfg.c_force_mode = Some(MODE_HYBRID);
                cfg.c_bandwidth = Some(audiopus_sys::OPUS_BANDWIDTH_SUPERWIDEBAND);
                println!("{}", row(&format!("{:>3} kb/s", b / 1000), &cfg, &speech, FS));
                match enc_rs(&cfg, &speech) {
                    Ok(p) => println!("    rs bands: {}", bands_text(&speech, &p, cfg.frame)),
                    Err(e) => println!("    rs bands: {e}"),
                }
                println!(
                    "    c  bands: {}",
                    bands_text(&speech, &enc_c(&cfg, &speech), cfg.frame)
                );
            }
        }
    }
}

/// Every frame duration, mono speech.
#[test]
#[ignore = "report: prints a table"]
fn frame_durations() {
    let speech = speech_segment(6);
    for app in [App::Voip, App::Audio, App::LowDelay] {
        for (frame, name) in [
            (60, "2.5"),
            (120, "5"),
            (240, "10"),
            (480, "20"),
            (960, "40"),
            (1440, "60"),
        ] {
            println!("== {} mono {name} ms VBR speech", app.name());
            for b in [12_000, 24_000, 32_000, 64_000] {
                let mut cfg = Cfg::new(app, b);
                cfg.frame = frame;
                println!("{}", row(&format!("{:>3} kb/s", b / 1000), &cfg, &speech, FS));
            }
        }
    }
}

/// Stereo: two related channels and two unrelated sources.
#[test]
#[ignore = "report: prints a table"]
fn stereo() {
    let speech = speech_segment(6);
    let fast = speech_fast_segment(6);
    let related = to_stereo(&speech);
    let unrelated = two_sources(&speech, &fast);
    for (name, pcm) in [("related L/R", &related), ("two sources", &unrelated)] {
        for app in [App::Voip, App::Audio] {
            println!("== {} stereo 20 ms VBR {name} (left channel measured)", app.name());
            for b in [
                12_000, 16_000, 24_000, 32_000, 48_000, 64_000, 96_000, 128_000, 256_000,
            ] {
                let mut cfg = Cfg::new(app, b);
                cfg.channels = 2;
                println!("{}", row(&format!("{:>3} kb/s", b / 1000), &cfg, pcm, FS));
            }
        }
    }
}

/// Complexity 0, 5 and 10.
#[test]
#[ignore = "report: prints a table"]
fn complexity() {
    let speech = speech_segment(6);
    for app in [App::Voip, App::Audio] {
        for cx in [0, 5, 10] {
            println!("== {} mono 20 ms VBR speech, complexity {cx}", app.name());
            for b in [12_000, 16_000, 24_000, 32_000, 64_000, 96_000] {
                let mut cfg = Cfg::new(app, b);
                cfg.complexity = cx;
                println!("{}", row(&format!("{:>3} kb/s", b / 1000), &cfg, &speech, FS));
            }
        }
    }
}

/// Chirp, music-like tones, silence and full-scale input.
#[test]
#[ignore = "report: prints a table"]
fn signals() {
    let n = 4 * FS;
    let loud_speech: Vec<f32> = speech_segment(4)
        .iter()
        .map(|&x| (x * 6.0).clamp(-1.0, 1.0))
        .collect();
    let full_sine: Vec<f32> = (0..n)
        .map(|i| (std::f64::consts::TAU * 1000.0 * i as f64 / FS as f64).sin() as f32)
        .collect();
    let signals: Vec<(&str, Vec<f32>)> = vec![
        ("chirp 100 Hz - 11 kHz", chirp(n, 100.0, 11_000.0, 0.4)),
        ("music-like chords", music(n)),
        ("full-scale 1 kHz sine", full_sine),
        ("speech x6, clipped at full scale", loud_speech),
    ];
    for (name, pcm) in &signals {
        for app in [App::Voip, App::Audio] {
            println!("== {} mono 20 ms VBR, {name}", app.name());
            for b in [16_000, 24_000, 32_000, 64_000, 128_000] {
                let cfg = Cfg::new(app, b);
                println!("{}", row(&format!("{:>3} kb/s", b / 1000), &cfg, pcm, FS));
            }
        }
    }
    println!("== silence, 2 s: packet sizes and the loudest decoded sample");
    let silence = vec![0.0f32; 2 * FS];
    for app in [App::Voip, App::Audio, App::LowDelay] {
        for b in [16_000, 32_000, 64_000] {
            for cbr in [false, true] {
                let mut cfg = Cfg::new(app, b);
                cfg.cbr = cbr;
                let side = |packets: Result<Vec<Vec<u8>>, String>| -> String {
                    match packets {
                        Err(e) => e,
                        Ok(p) => {
                            let out = dec_c(&p, FS, 1, cfg.frame).unwrap();
                            let peak = out.iter().fold(0f32, |m, &v| m.max(v.abs()));
                            let (lo, hi) = size_range(&p);
                            format!("{lo}-{hi} bytes, peak {peak:.6} | {}", toc_summary(&p))
                        }
                    }
                };
                println!(
                    "{} {} kb/s {}\n    rs: {}\n    c : {}",
                    app.name(),
                    b / 1000,
                    if cbr { "CBR" } else { "VBR" },
                    side(enc_rs(&cfg, &silence)),
                    side(Ok(enc_c(&cfg, &silence)))
                );
            }
        }
    }
}

/// Delay: where an isolated impulse comes out, in samples at 24 kHz.
#[test]
#[ignore = "report: prints a table"]
fn impulse_delay() {
    let at = FS / 2 + 137;
    let pcm = impulse(FS, at, 0.5);
    for app in [App::Voip, App::Audio, App::LowDelay] {
        for b in [16_000, 32_000, 96_000] {
            let cfg = Cfg::new(app, b);
            let side = |packets: Result<Vec<Vec<u8>>, String>| -> String {
                match packets {
                    Err(e) => e,
                    Ok(p) => {
                        let out = dec_c(&p, FS, 1, cfg.frame).unwrap();
                        let (mut peak, mut pos) = (0f32, 0usize);
                        for (i, &v) in out.iter().enumerate() {
                            if v.abs() > peak {
                                peak = v.abs();
                                pos = i;
                            }
                        }
                        // Energy-weighted centre of the response.
                        let (mut e, mut c) = (0f64, 0f64);
                        for (i, &v) in out.iter().enumerate() {
                            e += f64::from(v) * f64::from(v);
                            c += i as f64 * f64::from(v) * f64::from(v);
                        }
                        format!(
                            "peak at +{} ({:.3}), energy centre +{:.1} | {}",
                            pos as isize - at as isize,
                            peak,
                            c / e.max(1e-30) - at as f64,
                            toc_summary(&p)
                        )
                    }
                }
            };
            println!(
                "{} {} kb/s\n    rs: {}\n    c : {}",
                app.name(),
                b / 1000,
                side(enc_rs(&cfg, &pcm)),
                side(Ok(enc_c(&cfg, &pcm)))
            );
        }
    }
}

/// `encode` (f32) against `encode_i16`: are the packets the same?
#[test]
#[ignore = "report: prints a table"]
fn i16_entry_point() {
    // Quantize first, so both entry points see the same 16-bit audio.
    let quantize = |x: &[f32]| -> Vec<f32> {
        to_i16(x).iter().map(|&s| s as f32 / 32768.0).collect()
    };
    let speech = speech_segment(4);
    for app in [App::Voip, App::Audio, App::LowDelay] {
        for b in [16_000, 24_000, 32_000, 64_000] {
            for ch in [1, 2] {
                let pcm = quantize(&if ch == 2 { to_stereo(&speech) } else { speech.clone() });
                let mut cfg = Cfg::new(app, b);
                cfg.channels = ch;
                let f = enc_rs(&cfg, &pcm);
                cfg.i16_input = true;
                let i = enc_rs(&cfg, &pcm);
                let verdict = match (&f, &i) {
                    (Ok(f), Ok(i)) => {
                        let same = f.iter().zip(i).filter(|(a, b)| a == b).count();
                        format!("{same} of {} packets byte-identical", f.len())
                    }
                    (f, i) => format!("f32: {:?} i16: {:?}", f.as_ref().err(), i.as_ref().err()),
                };
                // libopus's own two entry points, for comparison.
                cfg.i16_input = false;
                let cf = enc_c(&cfg, &pcm);
                cfg.i16_input = true;
                let ci = enc_c(&cfg, &pcm);
                let c_same = cf.iter().zip(&ci).filter(|(a, b)| a == b).count();
                println!(
                    "{} {} kb/s {ch} ch: opus1 {verdict}; libopus {c_same} of {}",
                    app.name(),
                    b / 1000,
                    cf.len()
                );
                if let Ok(i) = &i {
                    cfg.i16_input = true;
                    let reference = channel(&pcm, ch, 0);
                    let m = |p: &[Vec<u8>]| {
                        let out = dec_c(p, FS, ch, cfg.frame).unwrap();
                        measure(&reference, &channel(&out, ch, 0), FS, FS / 5)
                    };
                    let (a, b2) = (m(i), m(&ci));
                    println!(
                        "    i16 path: rs snr {:.2} wcorr {:.3} lvl {:+.2} | c snr {:.2} wcorr {:.3} lvl {:+.2}",
                        a.snr, a.worst_corr, a.level_db, b2.snr, b2.worst_corr, b2.level_db
                    );
                }
            }
        }
    }
}

/// The bitrate changes every second, so the mode switches between SILK-only,
/// Hybrid and CELT-only. The worst 20 ms window in the 100 ms after each
/// change shows whether the switch is clean.
#[test]
#[ignore = "report: prints a table"]
fn bitrate_switch() {
    let speech = speech_segment(8);
    for app in [App::Voip, App::Audio] {
        for schedule in [
            [16_000, 96_000, 24_000, 96_000, 12_000, 64_000, 32_000, 128_000],
            [96_000, 16_000, 96_000, 8_000, 96_000, 24_000, 128_000, 40_000],
        ] {
            println!("== {} mono 20 ms VBR, bitrate per second: {schedule:?}", app.name());
            let cfg = Cfg::new(app, schedule[0]);
            let rs: Result<Vec<Vec<u8>>, String> = (|| {
                let mut enc = rs_encoder(&cfg);
                let mut all = Vec::new();
                for (k, &b) in schedule.iter().enumerate() {
                    enc.set_bitrate(Bitrate::BitsPerSecond(b)).unwrap();
                    all.extend(enc_rs_with(&mut enc, &cfg, &speech[k * FS..(k + 1) * FS])?);
                }
                Ok(all)
            })();
            let c = {
                let mut enc = CEncoder::new(&cfg);
                let mut all = Vec::new();
                for (k, &b) in schedule.iter().enumerate() {
                    enc.ctl(audiopus_sys::OPUS_SET_BITRATE_REQUEST, b);
                    all.extend(enc.encode(&cfg, &speech[k * FS..(k + 1) * FS]));
                }
                all
            };
            let side = |name: &str, packets: Result<Vec<Vec<u8>>, String>| {
                let packets = match packets {
                    Ok(p) => p,
                    Err(e) => {
                        println!("  {name}: {e}");
                        return;
                    }
                };
                let out = match dec_c(&packets, FS, 1, cfg.frame) {
                    Ok(o) => o,
                    Err(e) => {
                        println!("  {name}: {e}");
                        return;
                    }
                };
                let m = measure(&speech, &out, FS, FS / 5);
                println!(
                    "  {name}: lag {} snr {:.2} wsnr {:.2} wcorr {:.3}",
                    m.lag, m.snr, m.worst_snr20, m.worst_corr
                );
                for k in 0..schedule.len() {
                    let seg = &packets[k * 50..(k + 1) * 50];
                    // Worst 20 ms window in the 100 ms after the change, and
                    // in the rest of the second, at the stream's lag.
                    let lag = m.lag as usize;
                    let win = |from: usize, to: usize| -> f64 {
                        let mut worst = f64::INFINITY;
                        let mut s = from;
                        while s + 480 <= to && s + 480 + lag <= out.len() {
                            let (mut e, mut sig) = (0f64, 0f64);
                            for i in s..s + 480 {
                                sig += f64::from(speech[i]).powi(2);
                                e += f64::from(speech[i] - out[i + lag]).powi(2);
                            }
                            if sig / 480.0 > 1e-5 {
                                worst = worst.min(10.0 * (sig / e.max(1e-30)).log10());
                            }
                            s += 240;
                        }
                        worst
                    };
                    println!(
                        "    second {k} ({:>3} kb/s): first 100 ms worst {:>6.2} dB, rest worst {:>6.2} dB | {}",
                        schedule[k] / 1000,
                        win(k * FS, k * FS + FS / 10 + 480),
                        win(k * FS + FS / 10, (k + 1) * FS),
                        toc_summary(seg)
                    );
                }
            };
            side("rs", rs);
            side("c ", Ok(c));
        }
    }
}

/// A 30 s stream: drift (lag at the start against the end), level over time,
/// the worst 20 ms window, and whether the error piles up at packet edges.
#[test]
#[ignore = "report: prints a table"]
fn long_stream() {
    let base = speech_24k();
    let mut speech = base.clone();
    speech.extend_from_slice(&base);
    speech.truncate(30 * FS);
    for (app, b) in [
        (App::Voip, 16_000),
        (App::Voip, 24_000),
        (App::Voip, 32_000),
        (App::Voip, 40_000),
        (App::Audio, 32_000),
        (App::Audio, 64_000),
    ] {
        let cfg = Cfg::new(app, b);
        println!("== {} mono 20 ms VBR {} kb/s, 30 s", app.name(), b / 1000);
        let side = |name: &str, packets: Result<Vec<Vec<u8>>, String>| {
            let packets = match packets {
                Ok(p) => p,
                Err(e) => {
                    println!("  {name}: {e}");
                    return;
                }
            };
            let out = dec_c(&packets, FS, 1, cfg.frame).unwrap();
            let m = measure(&speech, &out, FS, FS / 5);
            let lag_first = best_lag(&speech[..5 * FS], &out[..5 * FS + FS / 25], FS / 25);
            let lag_last = best_lag(&speech[25 * FS..29 * FS], &out[25 * FS..], FS / 25);
            // Level of each second against the input, where the input is loud enough.
            let lag = m.lag as usize;
            let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
            for s in 0..29 {
                let r = &speech[s * FS..(s + 1) * FS];
                let o = &out[s * FS + lag..(s + 1) * FS + lag];
                let (er, eo): (f64, f64) = (
                    r.iter().map(|&v| f64::from(v).powi(2)).sum(),
                    o.iter().map(|&v| f64::from(v).powi(2)).sum(),
                );
                if er / FS as f64 > 1e-5 {
                    let d = 10.0 * (eo / er).log10();
                    lo = lo.min(d);
                    hi = hi.max(d);
                }
            }
            // Error energy by position inside the 20 ms packet, in 1 ms bins.
            let mut bins = [0f64; 20];
            for i in FS..29 * FS {
                let e = f64::from(speech[i] - out[i + lag]);
                bins[((i + lag) % 480) / 24] += e * e;
            }
            let mean = bins.iter().sum::<f64>() / 20.0;
            let peak = bins.iter().cloned().fold(0f64, f64::max);
            println!(
                "  {name}: {:.1} kb/s, lag {} (first 5 s {lag_first}, last 5 s {lag_last}), snr {:.2}, worst 20 ms {:.2} dB, worst 100 ms corr {:.3}, level per second {:+.2}..{:+.2} dB, error in the worst 1 ms slot of the packet {:.2}x the mean | {}",
                mean_bitrate(&packets, cfg.frame) / 1000.0,
                m.lag,
                m.snr,
                m.worst_snr20,
                m.worst_corr,
                lo,
                hi,
                peak / mean.max(1e-30),
                toc_summary(&packets)
            );
        };
        side("rs", enc_rs(&cfg, &speech));
        side("c ", Ok(enc_c(&cfg, &speech)));
    }
}

/// The first 200 ms of a stream that starts in the middle of a word.
#[test]
#[ignore = "report: prints a table"]
fn startup() {
    let all = speech_24k();
    // Start where the signal is loud.
    let speech = all[3 * FS + 2400..5 * FS + 2400].to_vec();
    for app in [App::Voip, App::Audio, App::LowDelay] {
        for b in [16_000, 24_000, 32_000, 64_000] {
            let cfg = Cfg::new(app, b);
            let side = |packets: Result<Vec<Vec<u8>>, String>| -> String {
                let packets = match packets {
                    Ok(p) => p,
                    Err(e) => return e,
                };
                let out = dec_c(&packets, FS, 1, cfg.frame).unwrap();
                let lag = best_lag(&speech, &out, FS / 25);
                let mut s = String::new();
                for w in 0..5 {
                    let (a, z) = (w * 960, (w + 1) * 960);
                    let m = measure_at(&speech[a..z], &out[a..], FS, 0, lag);
                    s += &format!(" [{:>5.2} dB, corr {:.3}, lvl {:+.1}]", m.snr, m.corr, m.level_db);
                }
                format!("lag {lag}{s} | {}", toc_summary(&packets[..10]))
            };
            println!(
                "{} {} kb/s: 40 ms windows from 0 to 200 ms\n    rs: {}\n    c : {}",
                app.name(),
                b / 1000,
                side(enc_rs(&cfg, &speech)),
                side(Ok(enc_c(&cfg, &speech)))
            );
        }
    }
}

/// The gap issue #54 left: CELT-only from 24 kHz against libopus, with
/// libopus's tonality analysis on (complexity 7 and above) and off.
#[test]
#[ignore = "report: prints a table"]
fn issue54_gap() {
    // The issue's chirp: 200 Hz rising at 1200 Hz/s for 1.3 s, amplitude 0.4.
    let n = FS * 13 / 10;
    let mut issue_chirp: Vec<f32> = (0..n)
        .map(|i| {
            let t = i as f64 / FS as f64;
            ((std::f64::consts::TAU * (200.0 * t + 600.0 * t * t)).sin() * 0.4) as f32
        })
        .collect();
    issue_chirp.resize(n.div_ceil(480) * 480 + 480, 0.0);
    let signals: Vec<(&str, Vec<f32>)> = vec![
        ("issue #54 chirp (200-1760 Hz)", issue_chirp),
        ("chirp 100 Hz - 11 kHz", chirp(4 * FS, 100.0, 11_000.0, 0.4)),
        ("speech", speech_segment(4)),
        ("music-like chords", music(4 * FS)),
    ];
    // libopus is forced to CELT-only: with its analysis on it would otherwise
    // leave CELT at 32 kb/s, and the comparison is about CELT.
    for (name, pcm) in &signals {
        for cbr in [false, true] {
            println!(
                "== audio mono 20 ms {}, {name}; libopus forced to CELT-only",
                if cbr { "CBR" } else { "VBR" }
            );
            println!("   kb/s  cx | opus1: SNR dB, actual kb/s | libopus: SNR dB, actual kb/s");
            for b in [32_000, 64_000, 128_000] {
                for cx in [5, 6, 7, 9, 10] {
                    let mut cfg = Cfg::new(App::Audio, b);
                    cfg.complexity = cx;
                    cfg.cbr = cbr;
                    cfg.c_force_mode = Some(MODE_CELT_ONLY);
                    let snr = |packets: &[Vec<u8>]| -> f64 {
                        let out = dec_c(packets, FS, 1, cfg.frame).unwrap();
                        measure(pcm, &out, FS, 0).snr
                    };
                    let rs = enc_rs(&cfg, pcm).unwrap();
                    let c = enc_c(&cfg, pcm);
                    let non_celt =
                        |p: &[Vec<u8>]| p.iter().filter(|p| toc(p).mode != "CELT").count();
                    assert_eq!(non_celt(&rs) + non_celt(&c), 0, "a stream left CELT-only");
                    println!(
                        "   {:>4} {:>3} | {:>7.2}  {:>6.1} | {:>7.2}  {:>6.1}",
                        b / 1000,
                        cx,
                        snr(&rs),
                        mean_bitrate(&rs, cfg.frame) / 1000.0,
                        snr(&c),
                        mean_bitrate(&c, cfg.frame) / 1000.0,
                    );
                }
            }
        }
    }
}

/// In-band FEC with 10 % and 20 % declared loss: stream size, mode, and the
/// quality when every 7th packet is lost and recovered from the next one.
#[test]
#[ignore = "report: prints a table"]
fn fec() {
    let speech = speech_segment(6);
    for cbr in [false, true] {
        for loss in [10, 20] {
            println!(
                "== voip mono 20 ms {} speech, FEC on, declared loss {loss} %",
                if cbr { "CBR" } else { "VBR" }
            );
            for b in [16_000, 24_000, 32_000, 40_000, 48_000] {
                let mut cfg = Cfg::new(App::Voip, b);
                cfg.fec = true;
                cfg.loss = loss;
                cfg.cbr = cbr;
                // The opus1 encoder can panic here; keep the table going.
                let rs = std::panic::catch_unwind(|| enc_rs(&cfg, &speech));
                if rs.is_err() {
                    println!("{:>3} kb/s\n    rs: PANIC in the encoder", b / 1000);
                    continue;
                }
                println!("{}", row(&format!("{:>3} kb/s", b / 1000), &cfg, &speech, FS));
                let lossy = |packets: &[Vec<u8>]| -> String {
                    let lost = |k: usize| k % 7 == 3;
                    let fec = dec_c_lossy(packets, FS, 1, cfg.frame, &lost, true).unwrap();
                    let plc = dec_c_lossy(packets, FS, 1, cfg.frame, &lost, false).unwrap();
                    let clean = dec_c(packets, FS, 1, cfg.frame).unwrap();
                    // Against the loss-free decode of the same stream, over the lost frames.
                    let err = |x: &[f32]| -> f64 {
                        let (mut s, mut e) = (0f64, 0f64);
                        for k in (0..packets.len()).filter(|&k| lost(k)) {
                            for i in k * cfg.frame..(k + 1) * cfg.frame {
                                s += f64::from(clean[i]).powi(2);
                                e += f64::from(clean[i] - x[i]).powi(2);
                            }
                        }
                        10.0 * (s / e.max(1e-30)).log10()
                    };
                    format!(
                        "lost frames vs loss-free decode: with FEC {:+.2} dB, concealment only {:+.2} dB",
                        err(&fec),
                        err(&plc)
                    )
                };
                match enc_rs(&cfg, &speech) {
                    Ok(p) => println!("    rs {}", lossy(&p)),
                    Err(e) => println!("    rs {e}"),
                }
                println!("    c  {}", lossy(&enc_c(&cfg, &speech)));
            }
        }
    }
}
