//! Tests of the paired-comparison harness `examples/eval`, whose modules
//! are included here by path:
//!
//! - the degradation self-test: on a reduced set (6 conditions, 4 items),
//!   libopus against libopus and opus1 against opus1 are exactly neutral,
//!   and every degradation of the opus1 side (low-pass, noise, a 1 ms
//!   shift, a lower and a higher bitrate, an energy mask that moves bits
//!   between bands) makes at least one gate fail that passes without it;
//! - the built-in `quick` set runs without a job error (its gates only
//!   warn, as decided for the evaluation track);
//! - the resampler that makes the encoder input is flat up to 300 Hz below
//!   the lower Nyquist frequency and attenuates by at least 90 dB above it;
//! - the JSON output reads back, set files and the corpus index work, and
//!   the external-metric merge pairs the values.
//!
//! The items of both sets are in the repository (`fixtures/tsp/` and
//! synthetic signals), so nothing is skipped. Run with
//! `cargo test --release --test eval_selftest -- --nocapture` to see the
//! reports.

#![cfg(feature = "std")]
#![allow(dead_code)]

#[path = "../examples/eval/corpus.rs"]
mod corpus;
#[path = "../examples/eval/degrade.rs"]
mod degrade;
#[path = "../examples/eval/libopus.rs"]
mod libopus;
#[path = "../examples/eval/pipeline.rs"]
mod pipeline;
#[path = "../examples/eval/report.rs"]
mod report;
#[path = "../examples/eval/sets.rs"]
mod sets;
#[path = "../examples/eval/signals.rs"]
mod signals;
#[path = "../examples/eval/stats.rs"]
mod stats;
#[path = "../examples/eval/tier2.rs"]
mod tier2;

use pipeline::{Codec, Inputs, Row};
use std::path::Path;

fn threads() -> usize {
    std::thread::available_parallelism().map_or(1, |n| n.get())
}

/// Run a set as `eval run` does.
fn run(set: &sets::EvalSet, items: &[corpus::Item]) -> Vec<Row> {
    let jobs = set.jobs(items);
    let inputs = Inputs::new(jobs.iter().map(|&(c, i)| (i, set.conditions[c].rate, set.conditions[c].channels)));
    pipeline::parallel_map(jobs.len(), threads(), "", |j| {
        let (c, i) = jobs[j];
        let cond = &set.conditions[c];
        let input = inputs.get(items, i, cond.rate, cond.channels);
        let result = pipeline::run_side(Codec::Opus1, Codec::Libopus, cond, &items[i], input, None, false).and_then(|a| {
            let b = pipeline::run_side(Codec::Libopus, Codec::Libopus, cond, &items[i], input, None, false)?;
            pipeline::pair(a, b)
        });
        Row { cond: c, item: i, result }
    })
}

#[test]
fn every_degradation_trips_a_gate() {
    let set = sets::selftest();
    let items = corpus::load_all(&set.items, threads()).expect("items");
    let runs = degrade::run_selftest(&set, &items, threads());
    let result = runs.evaluate(&set, &items);
    println!("{}", result.report());
    assert!(result.base_errors.is_empty(), "job errors: {:?}", result.base_errors);
    for (name, problems) in &result.identity {
        assert!(problems.is_empty(), "identity check {name}: {problems:?}");
    }
    for c in &result.checks {
        assert!(c.errors.is_empty(), "{}: {:?}", c.name, c.errors);
        assert!(!c.tripped.is_empty(), "the degradation {} trips no gate", c.name);
    }
    assert_eq!(result.checks.len(), degrade::Degrade::all().len());
    assert!(result.passed());

    // The self-test fails when the gates are too weak to see anything:
    // every limit out of reach, the gates of `--baseline` included.
    let mut weak = sets::selftest();
    for gate in weak.groups.iter_mut().flat_map(|g| g.gates.iter_mut()) {
        gate.limit = if gate.kind.upper() { f64::INFINITY } else { f64::NEG_INFINITY };
    }
    let weak_result = runs.evaluate(&weak, &items);
    assert!(!weak_result.passed());
    for c in &weak_result.checks {
        assert!(c.tripped.is_empty() && !c.passed(), "{} still trips {:?}", c.name, c.tripped.first().map(|g| g.line()));
    }
    // The identity checks do not depend on the limits.
    assert!(weak_result.identity.iter().all(|(_, p)| p.is_empty()));

    // A gate that already fails without a degradation does not count for
    // any degradation.
    let first = result.checks[0].tripped[0].clone();
    let mut failing = sets::selftest();
    let group = failing.groups.iter_mut().find(|g| g.name == first.group).expect("group");
    let gate = group.gates.iter_mut().find(|g| g.kind == first.kind).expect("gate");
    gate.limit = if gate.kind.upper() { f64::NEG_INFINITY } else { f64::INFINITY };
    let failing_result = runs.evaluate(&failing, &items);
    assert!(failing_result.base_gates.iter().any(|g| g.name == first.name && !g.pass), "{}", first.name);
    for c in &failing_result.checks {
        assert!(c.tripped.iter().all(|g| g.name != first.name), "{} counts {}", c.name, first.name);
    }
}

#[test]
fn quick_set_runs_without_errors() {
    let start = std::time::Instant::now();
    let set = sets::quick();
    let items = corpus::load_all(&set.items, threads()).expect("items");
    let rows = run(&set, &items);
    let aggregates = stats::aggregate(&set, &items, &rows);
    let gates = stats::evaluate_gates(&set, &items, &rows);
    let info = report::RunInfo {
        set: set.name.clone(),
        corpus_root: None,
        items_fnv64: corpus::items_hash(&items),
        decoder: "libopus".into(),
        degrade: None,
        threads: threads(),
        seconds: start.elapsed().as_secs_f64(),
    };
    println!("{}", report::text_report(&info, &set, &items, &rows, &aggregates, &gates));
    let errors: Vec<String> = rows
        .iter()
        .filter_map(|r| r.result.as_ref().err().map(|e| format!("{} {}: {e}", set.conditions[r.cond].name, items[r.item].name)))
        .collect();
    assert!(errors.is_empty(), "job errors: {errors:?}");
    assert_eq!(rows.len(), set.jobs(&items).len());
    // Every job checked the final range of the decoder against opus1's
    // encoder on most packets.
    for r in &rows {
        let p = r.result.as_ref().unwrap();
        assert!(p.opus1.stream.range_checks + 2 >= p.opus1.stream.packets - p.opus1.stream.lost * 2);
    }

    // The JSON file reads back: rows, aggregates and the stored values of
    // `--baseline`.
    let json = report::run_json(&info, &set, &items, &rows, &aggregates, &gates);
    let parsed = report::parse_json(&json).expect("valid JSON");
    assert_eq!(parsed.get("rows").map(|r| r.as_arr().len()), Some(rows.len()));
    assert_eq!(parsed.get("aggregates").map(|a| a.as_arr().len()), Some(aggregates.len()));
    let path = std::env::temp_dir().join(format!("opus1-eval-quick-{}.json", std::process::id()));
    std::fs::write(&path, &json).unwrap();
    let stored = report::read_stored(path.to_str().unwrap()).expect("stored");
    let _ = std::fs::remove_file(&path);
    assert_eq!(stored.len(), rows.len());
    for r in &rows {
        let p = r.result.as_ref().unwrap();
        let s = stored[&(set.conditions[r.cond].name.clone(), items[r.item].name.clone())];
        // The shortest decimal reads back to the same bits.
        assert_eq!(s.err_opus1.to_bits(), p.opus1.err.to_bits());
        assert_eq!(s.err_libopus.to_bits(), p.libopus.err.to_bits());
    }
    // Against itself, every baseline gate passes.
    for g in stats::baseline_gates(&set, &items, &rows, &stored) {
        assert!(g.pass, "{}", g.line());
    }
}

/// The two encoders of a pair report the same settings for every condition
/// of every built-in set: application, rate, channels, bitrate, bitrate
/// mode, complexity, FEC, expected loss, look-ahead, frame duration and the
/// controls the harness leaves at their defaults (signal type, bandwidth
/// limit, bit depth, discontinuous transmission, phase inversion).
#[test]
fn both_encoders_get_the_same_settings() {
    let sets = [
        sets::quick(),
        sets::selftest(),
        sets::music(),
        sets::tsp(Path::new("/corpus")).expect("tsp set"),
        sets::tsp_rates(Path::new("/corpus")).expect("tsp_rates set"),
    ];
    for set in &sets {
        for cond in &set.conditions {
            let opus1 = pipeline::encoder_settings(Codec::Opus1, cond).expect("opus1 encoder");
            let libopus = pipeline::encoder_settings(Codec::Libopus, cond).expect("libopus encoder");
            assert_eq!(opus1, libopus, "{} of set {}", cond.name, set.name);
            assert_eq!(opus1.bitrate, cond.bitrate_at(0.0), "{}", cond.name);
            assert_eq!(opus1.complexity, cond.complexity, "{}", cond.name);
            assert_eq!(opus1.vbr, !cond.cbr, "{}", cond.name);
        }
    }
}

/// Amplitude (dB) of a sine at `f` after resampling from 48 kHz to `rate`,
/// measured away from the edges by projection on sine and cosine.
fn sine_gain_db(rate: u32, f: f64, up: bool) -> f64 {
    let (from, to) = if up { (rate, 48_000) } else { (48_000, rate) };
    let n = from as usize; // 1 s
    let x: Vec<f64> = (0..n).map(|i| (std::f64::consts::TAU * f * i as f64 / from as f64).sin()).collect();
    let y = signals::Resampler::new(from, to).run(&x);
    assert_eq!(y.len(), n * to as usize / from as usize);
    let edge = y.len() / 5;
    let (mut s, mut c) = (0.0, 0.0);
    for (k, v) in y.iter().enumerate().take(y.len() - edge).skip(edge) {
        let w = std::f64::consts::TAU * f * k as f64 / to as f64;
        s += v * w.sin();
        c += v * w.cos();
    }
    let count = (y.len() - 2 * edge) as f64;
    let amplitude = 2.0 * (s * s + c * c).sqrt() / count;
    20.0 * amplitude.max(1e-30).log10()
}

#[test]
fn resampler_is_flat_in_the_compared_band() {
    for rate in [8_000u32, 12_000, 16_000, 24_000] {
        let nyquist = rate as f64 / 2.0;
        for up in [false, true] {
            // Pass band: up to 300 Hz below the Nyquist frequency.
            for f in [100.0, 1000.0, nyquist / 2.0, nyquist - 700.0, nyquist - 300.0] {
                let g = sine_gain_db(rate, f, up);
                assert!(g.abs() <= 0.1, "{rate} Hz {}: {f} Hz has gain {g:.4} dB", if up { "up" } else { "down" });
            }
        }
        // Stop band of the downsampler: a sine above the new Nyquist
        // frequency must not alias back into the output.
        for f in [nyquist + 10.0, nyquist + 500.0, nyquist * 1.5] {
            let alias = rate as f64 - f;
            let x: Vec<f64> = (0..48_000).map(|i| (std::f64::consts::TAU * f * i as f64 / 48_000.0).sin()).collect();
            let y = signals::Resampler::new(48_000, rate).run(&x);
            let edge = y.len() / 5;
            let (mut s, mut c) = (0.0, 0.0);
            for (k, v) in y.iter().enumerate().take(y.len() - edge).skip(edge) {
                let w = std::f64::consts::TAU * alias * k as f64 / rate as f64;
                s += v * w.sin();
                c += v * w.cos();
            }
            let g = 20.0 * (2.0 * (s * s + c * c).sqrt() / (y.len() - 2 * edge) as f64).max(1e-30).log10();
            assert!(g <= -90.0, "{rate} Hz: {f} Hz aliases at {g:.1} dB");
        }
    }
    // The same rate is a copy.
    let x = [0.25, -0.5, 0.125];
    assert_eq!(signals::Resampler::new(48_000, 48_000).run(&x), x);
}

#[test]
fn items_and_inputs() {
    let set = sets::quick();
    let items = corpus::load_all(&set.items, threads()).expect("items");
    for item in &items {
        // Whole samples at every rate opus_compare accepts.
        assert_eq!(item.frames() % 12, 0, "{}", item.name);
    }
    // At 48 kHz the encoder input is the reference scaled by 1/32768,
    // exactly; a mono condition on a stereo item codes (L+R)/2.
    let music = items.iter().find(|i| i.channels == 2).expect("a stereo item");
    let stereo = pipeline::encoder_input(music, 48_000, 2);
    assert!(stereo.iter().zip(&music.reference).all(|(&x, &r)| x == r as f32 / 32768.0));
    let mono = pipeline::encoder_input(music, 48_000, 1);
    assert!(mono.iter().zip(music.reference.as_chunks::<2>().0).all(|(&x, p)| x == (p[0] as f32 + p[1] as f32) / 65536.0));
    assert_eq!(pipeline::encoder_input(music, 16_000, 1).len(), music.frames() / 3);
    // The fixture excerpts are speech of 2.5 s.
    let speech: Vec<_> = items.iter().filter(|i| i.class == corpus::Class::Speech).collect();
    assert!(speech.len() >= 4);
    assert!(speech.iter().any(|i| i.name == "tsp/FA01_05" && i.frames() == 120_000));
}

#[test]
fn degradations_parse_and_act() {
    for d in degrade::Degrade::all() {
        assert_eq!(degrade::Degrade::parse(&d.name()).unwrap(), d);
    }
    assert!(degrade::Degrade::parse("lowpass:5000").is_ok());
    assert!(degrade::Degrade::parse("bogus").is_err());
    // A 1 ms shift at 48 kHz delays by 48 samples per channel.
    let mut pcm: Vec<i16> = (0..200).map(|i| i as i16).collect();
    degrade::Degrade::Shift(1.0).apply_post(&mut pcm, 48_000, 2);
    assert!(pcm[..96].iter().all(|&s| s == 0));
    assert_eq!(pcm[96], 0);
    assert_eq!(pcm[97], 1);
    // The low-pass removes a 12 kHz tone and keeps a 1 kHz tone.
    let tone = |f: f64| -> Vec<i16> {
        (0..48_000).map(|i| (10_000.0 * (std::f64::consts::TAU * f * i as f64 / 48_000.0).sin()) as i16).collect()
    };
    let rms = |x: &[i16]| (x[4800..43200].iter().map(|&s| (s as f64).powi(2)).sum::<f64>() / 38400.0).sqrt();
    let (mut low, mut high) = (tone(1000.0), tone(12_000.0));
    degrade::Degrade::Lowpass(None).apply_post(&mut low, 48_000, 1);
    degrade::Degrade::Lowpass(None).apply_post(&mut high, 48_000, 1);
    assert!((rms(&low) / rms(&tone(1000.0)) - 1.0).abs() < 0.01);
    // A 4th-order Butterworth run twice: about 37 dB down at 12 kHz.
    assert!(rms(&high) < 0.02 * rms(&tone(12_000.0)));
    // Noise 20 dB below the signal.
    let mut noisy = tone(1000.0);
    degrade::Degrade::Noise(20.0).apply_post(&mut noisy, 48_000, 1);
    let clean = tone(1000.0);
    let noise: f64 = noisy.iter().zip(&clean).map(|(&a, &b)| ((a - b) as f64).powi(2)).sum::<f64>() / 48_000.0;
    let snr = 10.0 * (rms(&clean).powi(2) / noise).log10();
    assert!((snr - 20.0).abs() < 0.5, "{snr}");
}

#[test]
fn set_files_and_corpus_index() {
    let fixtures = corpus::fixtures_dir();
    let text = corpus::corpus_index(&fixtures.join("tsp"), &fixtures, 2, corpus::Class::Speech).unwrap();
    let specs = corpus::parse_set(&text, &fixtures).unwrap();
    // One directory, two of its four files, evenly spaced.
    let names: Vec<&str> = specs.iter().map(|s| s.name.as_str()).collect();
    assert_eq!(names, ["tsp/FA01_05.wav", "tsp/MD19_04.wav"]);
    let items = corpus::load_all(&specs, 2).unwrap();
    assert_eq!(items[0].frames(), 120_000);
    // Start and duration.
    let part = corpus::parse_set("tsp/MD19_04.wav\tspeech\t0.5\t1.0\n# comment\n", &fixtures).unwrap();
    let item = corpus::load(&part[0]).unwrap();
    assert_eq!(item.frames(), 48_000);
    let whole = corpus::load(&corpus::parse_set("tsp/MD19_04.wav", &fixtures).unwrap()[0]).unwrap();
    assert_eq!(item.reference[..], whole.reference[24_000..72_000]);
    assert!(corpus::parse_set("x.wav\tnoise", &fixtures).is_err());
    // The split is a fixed function of the name.
    assert_eq!(corpus::split_of("tsp/FA01_05"), corpus::split_of("tsp/FA01_05"));
    let holdout = (0..400).filter(|i| corpus::split_of(&format!("item{i}")) == corpus::Split::Holdout).count();
    assert!((60..140).contains(&holdout), "{holdout}");
    // The committed TSP list has 222 items from 25 directories.
    let list = std::fs::read_to_string(sets::tsp_set_file()).unwrap();
    let tsp = corpus::parse_set(&list, Path::new("/corpus")).unwrap();
    assert_eq!(tsp.len(), 222);
}

#[test]
fn statistics() {
    assert!((stats::geomean(&[1.0, 4.0]) - 2.0).abs() < 1e-12);
    let values: Vec<f64> = (1..=50).map(|i| 1.0 + i as f64 / 100.0).collect();
    let (lo, hi) = stats::bootstrap_ci(&values);
    let g = stats::geomean(&values);
    assert!(lo < g && g < hi && hi - lo < 0.1, "{lo} {g} {hi}");
    // Deterministic.
    assert_eq!(stats::bootstrap_ci(&values), (lo, hi));
    assert_eq!(stats::bootstrap_ci(&[1.5]), (1.5, 1.5));
}

#[test]
fn merge_pairs_external_metrics() {
    let dir = std::env::temp_dir().join(format!("opus1-eval-merge-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let run = dir.join("run.json");
    std::fs::write(
        &run,
        "{\"rows\":[\n\
         {\"condition\":\"c1\",\"item\":\"a\",\"opus1\":{\"err\":0.2},\"libopus\":{\"err\":0.1}},\n\
         {\"condition\":\"c1\",\"item\":\"b\",\"opus1\":{\"err\":0.1},\"libopus\":{\"err\":0.2}}]}",
    )
    .unwrap();
    let tier2 = dir.join("tier2.jsonl");
    std::fs::write(
        &tier2,
        "{\"condition\":\"c1\",\"item\":\"a\",\"codec\":\"opus1\",\"metric\":\"visqol\",\"value\":4.0,\"higher_is_better\":true}\n\
         {\"condition\":\"c1\",\"item\":\"a\",\"codec\":\"libopus\",\"metric\":\"visqol\",\"value\":3.5,\"higher_is_better\":true}\n\
         {\"condition\":\"c1\",\"item\":\"b\",\"codec\":\"opus1\",\"metric\":\"visqol\",\"value\":4.5,\"higher_is_better\":true}\n\
         {\"condition\":\"c1\",\"item\":\"b\",\"codec\":\"libopus\",\"metric\":\"visqol\",\"value\":4.0,\"higher_is_better\":true}\n",
    )
    .unwrap();
    let (text, json) = tier2::merge(run.to_str().unwrap(), tier2.to_str().unwrap()).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    let parsed = report::parse_json(&json).unwrap();
    let metrics = parsed.get("metrics").unwrap().as_arr();
    let visqol = metrics.iter().find(|m| m.get("metric").and_then(|v| v.as_str()) == Some("visqol")).unwrap();
    assert_eq!(visqol.get("mean_diff").and_then(|v| v.as_f64()), Some(0.5));
    // Item a: ViSQOL prefers opus1, opus_compare prefers libopus.
    let disagree = parsed.get("disagreements").unwrap().as_arr();
    assert_eq!(disagree.len(), 1, "{text}");
    assert_eq!(disagree[0].get("item").and_then(|v| v.as_str()), Some("a"));
}
