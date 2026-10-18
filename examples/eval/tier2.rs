//! The bridge to optional external metrics ("tier 2": ViSQOL, PESQ,
//! SCOREQ, PLCMOS and others). None of them is a dependency; the user runs
//! whichever they have installed.
//!
//! 1. `eval run --write-wav DIR` writes, per job, the reference, the
//!    encoder input and both decoded signals (and 16 kHz versions for
//!    speech in mono), plus `DIR/manifest.jsonl` with one line per job.
//! 2. `scripts/eval/tier2.py DIR/manifest.jsonl > tier2.jsonl` runs the
//!    installed tools and writes one line per (condition, item, codec,
//!    metric, value).
//! 3. `eval merge RUN.json tier2.jsonl` adds the paired differences
//!    (opus1 - libopus) per condition and metric, with bootstrap intervals,
//!    and lists the items where the metrics disagree on which codec is
//!    better: these are the items to listen to.

use crate::corpus::{Class, Item};
use crate::pipeline::{Condition, Pair};
use crate::report::{Json, json_string, num, parse_json, parse_json_lines};
use crate::signals::{self, REF_RATE, Resampler};
use crate::stats;
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::Path;

fn safe(name: &str) -> String {
    name.chars().map(|c| if c.is_ascii_alphanumeric() || c == '.' || c == '-' { c } else { '_' }).collect()
}

/// Write the signals of one job under `dir` and return its manifest line.
/// The pair must have kept its decoded signals.
pub fn write_job(dir: &Path, cond: &Condition, item: &Item, input: &[f32], pair: &Pair) -> Result<String, String> {
    let (Some(a), Some(b)) = (&pair.opus1.pcm, &pair.libopus.pcm) else {
        return Err("the decoded signals were not kept".into());
    };
    let item_dir = dir.join("items");
    let reference = item_dir.join(format!("{}.wav", safe(&item.name)));
    if !reference.exists() {
        signals::write_wav(&reference, REF_RATE, item.channels, &item.reference)?;
    }
    let job = dir.join(safe(&cond.name)).join(safe(&item.name));
    let input16: Vec<i16> = input.iter().map(|&v| signals::to_i16(v as f64)).collect();
    let paths = [("input", &input16), ("opus1", a), ("libopus", b)];
    let mut fields = vec![
        format!("\"condition\":{}", json_string(&cond.name)),
        format!("\"group\":{}", json_string(&cond.group)),
        format!("\"item\":{}", json_string(&item.name)),
        format!("\"class\":{}", json_string(item.class.name())),
        format!("\"split\":{}", json_string(item.split.name())),
        format!("\"rate\":{}", cond.rate),
        format!("\"channels\":{}", cond.channels),
        format!("\"decode_loss\":{}", cond.decode_loss),
        format!("\"reference_48k\":{}", json_string(&reference.display().to_string())),
    ];
    for (name, pcm) in paths {
        let path = job.join(format!("{name}.wav"));
        signals::write_wav(&path, cond.rate, cond.channels, pcm)?;
        fields.push(format!("\"{name}\":{}", json_string(&path.display().to_string())));
    }
    if item.class == Class::Speech && cond.channels == 1 {
        let reference16: Vec<i16> = item.moc_reference().iter().map(|&v| signals::to_i16(v as f64)).collect();
        let to16 = |pcm: &[i16]| -> Vec<i16> {
            let x: Vec<f64> = pcm.iter().map(|&s| s as f64 / 32768.0).collect();
            let y = if cond.rate == 16_000 { x } else { Resampler::new(cond.rate, 16_000).run(&x) };
            y.iter().map(|&v| signals::to_i16(v)).collect()
        };
        for (name, pcm) in [("reference_16k", reference16), ("opus1_16k", to16(a)), ("libopus_16k", to16(b))] {
            let path = job.join(format!("{name}.wav"));
            signals::write_wav(&path, 16_000, 1, &pcm)?;
            fields.push(format!("\"{name}\":{}", json_string(&path.display().to_string())));
        }
    }
    Ok(format!("{{{}}}", fields.join(",")))
}

/// `eval merge RUN.json TIER2.jsonl`: the paired differences of every
/// external metric, and the items where the metrics disagree. Returns the
/// text report and a JSON summary.
pub fn merge(run_path: &str, tier2_path: &str) -> Result<(String, String), String> {
    let run_text = std::fs::read_to_string(run_path).map_err(|e| format!("{run_path}: {e}"))?;
    let run = parse_json(&run_text).map_err(|e| format!("{run_path}: {e}"))?;
    let tier2_text = std::fs::read_to_string(tier2_path).map_err(|e| format!("{tier2_path}: {e}"))?;
    let lines = parse_json_lines(&tier2_text).map_err(|e| format!("{tier2_path}: {e}"))?;

    // (condition, item, metric) -> (opus1, libopus); and whether higher is
    // better for each metric.
    type Key = (String, String, String);
    let mut values: BTreeMap<Key, (Option<f64>, Option<f64>)> = BTreeMap::new();
    let mut higher_better: BTreeMap<String, bool> = BTreeMap::new();
    for l in &lines {
        let s = |k: &str| l.get(k).and_then(Json::as_str).map(str::to_string);
        let (Some(c), Some(i), Some(codec), Some(metric)) = (s("condition"), s("item"), s("codec"), s("metric")) else {
            continue;
        };
        let Some(v) = l.get("value").and_then(Json::as_f64) else { continue };
        higher_better.insert(metric.clone(), !matches!(l.get("higher_is_better"), Some(Json::Bool(false))));
        let e = values.entry((c, i, metric)).or_default();
        match codec.as_str() {
            "opus1" => e.0 = Some(v),
            "libopus" => e.1 = Some(v),
            _ => {}
        }
    }
    // opus_compare's err from the run counts as one more metric (lower is
    // better).
    higher_better.insert("opus_compare_err".into(), false);
    for row in run.get("rows").map_or(&[][..], Json::as_arr) {
        let s = |k: &str| row.get(k).and_then(Json::as_str).map(str::to_string);
        let err = |side: &str| row.get(side).and_then(|x| x.get("err")).and_then(Json::as_f64);
        if let (Some(c), Some(i)) = (s("condition"), s("item")) {
            values.insert((c, i, "opus_compare_err".into()), (err("opus1"), err("libopus")));
        }
    }

    // Paired differences per (condition, metric), and per job the sign of
    // "opus1 is better" for each metric.
    let mut diffs: BTreeMap<(String, String), Vec<f64>> = BTreeMap::new();
    let mut verdicts: BTreeMap<(String, String), Vec<(String, f64)>> = BTreeMap::new();
    for ((c, i, m), (a, b)) in &values {
        let (Some(a), Some(b)) = (a, b) else { continue };
        let d = a - b;
        diffs.entry((c.clone(), m.clone())).or_default().push(d);
        let better = if higher_better[m] { d } else { -d };
        verdicts.entry((c.clone(), i.clone())).or_default().push((m.clone(), better));
    }

    let mut text = String::new();
    let mut json = String::from("{\"tool\":\"opus1 eval merge\",\"schema\":1,\"metrics\":[\n");
    let _ = writeln!(text, "{:<44} {:<20} {:>5} {:>11} {:>25}", "condition", "metric", "n", "opus1-lib", "95% CI");
    let mut parts = Vec::new();
    for ((c, m), d) in &diffs {
        let mean = stats::mean(d);
        let ci = stats::bootstrap_mean_ci(d);
        let _ = writeln!(text, "{c:<44} {m:<20} {:>5} {mean:>11.4} [{:>11.4}, {:>11.4}]", d.len(), ci.0, ci.1);
        parts.push(format!(
            "{{\"condition\":{},\"metric\":{},\"higher_is_better\":{},\"n\":{},\"mean_diff\":{},\"ci95\":[{},{}]}}",
            json_string(c),
            json_string(m),
            higher_better[m],
            d.len(),
            num(mean),
            num(ci.0),
            num(ci.1)
        ));
    }
    json += &parts.join(",\n");
    json += "],\n\"disagreements\":[\n";
    let mut parts = Vec::new();
    let _ = writeln!(text, "\nitems where the metrics disagree on which codec is better:");
    for ((c, i), v) in &verdicts {
        let says_better = v.iter().any(|(_, b)| *b > 0.0);
        let says_worse = v.iter().any(|(_, b)| *b < 0.0);
        if says_better && says_worse {
            let detail: Vec<String> = v.iter().map(|(m, b)| format!("{m} {}", if *b > 0.0 { "opus1" } else { "libopus" })).collect();
            let _ = writeln!(text, "  {c} {i}: {}", detail.join(", "));
            parts.push(format!("{{\"condition\":{},\"item\":{}}}", json_string(c), json_string(i)));
        }
    }
    json += &parts.join(",\n");
    json += "]}\n";
    Ok((text, json))
}
