//! The commands of `eval` (see `main.rs`).

use crate::pipeline::{Codec, Inputs, Row};
use crate::{conformance, corpus, degrade, pipeline, report, sets, signals, stats, tier2};
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Default)]
struct Opts {
    positional: Vec<String>,
    set: Option<String>,
    items: Option<String>,
    corpus: Option<String>,
    filter: Option<String>,
    item_filter: Option<String>,
    max_items: Option<usize>,
    json: Option<String>,
    md: Option<String>,
    text: Option<String>,
    baseline: Option<String>,
    degrade: Option<String>,
    write_wav: Option<String>,
    decoder: Option<String>,
    threads: Option<usize>,
    strict: bool,
    vectors: Option<String>,
    apply_vad: bool,
    root: Option<String>,
    out: Option<String>,
    per_group: Option<usize>,
    class: Option<String>,
}

impl Opts {
    fn parse(args: &[String]) -> Result<Self, String> {
        let mut o = Opts::default();
        let mut i = 0;
        while i < args.len() {
            let a = args[i].as_str();
            let mut value = || -> Result<String, String> {
                i += 1;
                args.get(i).cloned().ok_or_else(|| format!("{a} needs a value"))
            };
            let number = |v: String| v.parse::<usize>().map_err(|_| format!("{a}: not a number: {v}"));
            match a {
                "--set" => o.set = Some(value()?),
                "--items" => o.items = Some(value()?),
                "--corpus" => o.corpus = Some(value()?),
                "--filter" => o.filter = Some(value()?),
                "--item-filter" => o.item_filter = Some(value()?),
                "--max-items" => o.max_items = Some(number(value()?)?),
                "--json" => o.json = Some(value()?),
                "--md" => o.md = Some(value()?),
                "--text" => o.text = Some(value()?),
                "--baseline" => o.baseline = Some(value()?),
                "--degrade" => o.degrade = Some(value()?),
                "--write-wav" => o.write_wav = Some(value()?),
                "--decoder" => o.decoder = Some(value()?),
                "--threads" => o.threads = Some(number(value()?)?),
                "--strict" => o.strict = true,
                "--vectors" => o.vectors = Some(value()?),
                "--apply-vad" => o.apply_vad = true,
                "--root" => o.root = Some(value()?),
                "--out" => o.out = Some(value()?),
                "--per-group" => o.per_group = Some(number(value()?)?),
                "--class" => o.class = Some(value()?),
                s if s.starts_with("--") => return Err(format!("unknown option {s}")),
                s => o.positional.push(s.to_string()),
            }
            i += 1;
        }
        Ok(o)
    }

    fn threads(&self) -> usize {
        self.threads.unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |n| n.get()))
    }

    fn corpus(&self) -> PathBuf {
        self.corpus.as_ref().map_or_else(corpus::corpus_root, PathBuf::from)
    }
}

fn write_file(path: &str, text: &str) -> Result<(), String> {
    if let Some(parent) = Path::new(path).parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
    }
    std::fs::write(path, text).map_err(|e| format!("{path}: {e}"))
}

/// The set named by the options, with the item list, filters and item
/// limit applied.
fn load_set(o: &Opts, default: &str) -> Result<sets::EvalSet, String> {
    let root = o.corpus();
    let mut set = sets::builtin(o.set.as_deref().unwrap_or(default), &root)?;
    if let Some(file) = &o.items {
        set.items = corpus::read_set_file(Path::new(file), &root)?;
        set.name += "+items";
    }
    if let Some(f) = &o.filter {
        set.filter_conditions(f);
    }
    if let Some(f) = &o.item_filter {
        set.items.retain(|s| f.split(',').any(|p| s.name.contains(p)));
    }
    if let Some(n) = o.max_items {
        // The first n items of each class.
        let mut seen = std::collections::HashMap::new();
        set.items.retain(|s| {
            let k = seen.entry(s.class).or_insert(0usize);
            *k += 1;
            *k <= n
        });
    }
    Ok(set)
}

fn cmd_list(o: &Opts) -> Result<i32, String> {
    let set = load_set(o, "quick")?;
    println!("set {}: {} conditions, {} items", set.name, set.conditions.len(), set.items.len());
    for g in &set.groups {
        let classes: Vec<&str> = g.classes.iter().map(|c| c.name()).collect();
        println!("\n[{}] items of class {}", g.name, classes.join(", "));
        for c in set.conditions.iter().filter(|c| c.group == g.name) {
            let tags = if c.tags.is_empty() { String::new() } else { format!("  ({})", c.tags.join(", ")) };
            println!("  {}{tags}", c.name);
        }
        for gate in &g.gates {
            let rate = gate.bitrate.map_or(String::new(), |b| format!(" {b}"));
            let note = if gate.kind.needs_stored() { "  (with --baseline)" } else { "" };
            println!("  gate {}{rate} {} {}{note}", gate.kind.name(), if gate.kind.upper() { "<=" } else { ">=" }, gate.limit);
        }
    }
    println!("\nitems:");
    for s in &set.items {
        let source = match &s.source {
            corpus::Source::File { path, .. } => path.display().to_string(),
            corpus::Source::Generated(_) => "generated".into(),
        };
        println!("  {:<40} {:<10} {:<8} {source}", s.name, s.class.name(), corpus::split_of(&s.name).name());
    }
    Ok(0)
}

fn cmd_run(o: &Opts) -> Result<i32, String> {
    let start = Instant::now();
    let threads = o.threads();
    let set = load_set(o, "quick")?;
    let degrade = o.degrade.as_deref().map(degrade::Degrade::parse).transpose()?;
    let decoder = match o.decoder.as_deref() {
        None | Some("libopus") => Codec::Libopus,
        Some("opus1") => Codec::Opus1,
        Some(d) => return Err(format!("unknown decoder {d}")),
    };
    let stored = o.baseline.as_deref().map(report::read_stored).transpose()?;
    pipeline::check_settings(&set.conditions)?;
    let items = corpus::load_all(&set.items, threads)?;
    let jobs = set.jobs(&items);
    eprintln!("set {}: {} conditions, {} items, {} jobs, {threads} threads", set.name, set.conditions.len(), items.len(), jobs.len());
    let inputs = Inputs::new(jobs.iter().map(|&(c, i)| (i, set.conditions[c].rate, set.conditions[c].channels)));
    let wav_dir = o.write_wav.as_ref().map(PathBuf::from);
    let manifest = std::sync::Mutex::new(Vec::new());
    let rows: Vec<Row> = pipeline::parallel_map(jobs.len(), threads, "run", |j| {
        let (c, i) = jobs[j];
        let cond = &set.conditions[c];
        let item = &items[i];
        let input = inputs.get(&items, i, cond.rate, cond.channels);
        let d = degrade.as_ref().filter(|d| d.applies_to(cond));
        let keep = wav_dir.is_some();
        let mut result = pipeline::run_side(Codec::Opus1, decoder, cond, item, input, d, keep).and_then(|a| {
            let b = pipeline::run_side(Codec::Libopus, decoder, cond, item, input, None, keep)?;
            pipeline::pair(a, b)
        });
        if let (Some(dir), Ok(p)) = (&wav_dir, &mut result) {
            match tier2::write_job(dir, cond, item, input, p) {
                Ok(line) => manifest.lock().unwrap().push((j, line)),
                Err(e) => eprintln!("{} {}: {e}", cond.name, item.name),
            }
            p.opus1.pcm = None;
            p.libopus.pcm = None;
        }
        pipeline::Row { cond: c, item: i, result }
    });
    let aggregates = stats::aggregate(&set, &items, &rows);
    let mut gates = stats::evaluate_gates(&set, &items, &rows);
    if let Some(stored) = &stored {
        gates.extend(stats::baseline_gates(&set, &items, &rows, stored));
    }
    let info = report::RunInfo {
        set: set.name.clone(),
        corpus_root: Some(o.corpus().display().to_string()),
        items_fnv64: corpus::items_hash(&items),
        decoder: decoder.name().into(),
        degrade: degrade.as_ref().map(|d| d.name()),
        threads,
        seconds: start.elapsed().as_secs_f64(),
    };
    let text = report::text_report(&info, &set, &items, &rows, &aggregates, &gates);
    print!("{text}");
    if let Some(path) = &o.text {
        write_file(path, &text)?;
    }
    if let Some(path) = &o.json {
        write_file(path, &report::run_json(&info, &set, &items, &rows, &aggregates, &gates))?;
        eprintln!("wrote {path}");
    }
    if let Some(path) = &o.md {
        write_file(path, &report::markdown(&set, &aggregates, &gates))?;
    }
    if let Some(dir) = &wav_dir {
        let mut lines = manifest.into_inner().unwrap();
        lines.sort();
        let text: String = lines.into_iter().map(|(_, l)| l + "\n").collect();
        write_file(&dir.join("manifest.jsonl").display().to_string(), &text)?;
    }
    let errors = rows.iter().filter(|r| r.result.is_err()).count();
    let gates_fail = gates.iter().any(|g| !g.pass);
    Ok(if errors > 0 || (o.strict && gates_fail) { 1 } else { 0 })
}

fn cmd_selftest(o: &Opts) -> Result<i32, String> {
    let set = load_set(o, "selftest")?;
    let threads = o.threads();
    let items = corpus::load_all(&set.items, threads)?;
    let start = Instant::now();
    let result = degrade::selftest(&set, &items, threads);
    print!("{}", result.report());
    eprintln!("{:.1} s", start.elapsed().as_secs_f64());
    Ok(if result.passed() { 0 } else { 1 })
}

fn cmd_conformance(o: &Opts) -> Result<i32, String> {
    let Some(dir) = conformance::find_vectors(o.vectors.as_deref()) else {
        return Err("RFC 8251 test vectors not found (give --vectors DIR or set OPUS1_TESTVECTORS or OPUS1_CORPUS)".into());
    };
    println!("vectors: {}", dir.display());
    let (text, pass) = conformance::conformance(&dir, o.threads());
    print!("{text}");
    Ok(if pass { 0 } else { 1 })
}

fn cmd_moc(o: &Opts) -> Result<i32, String> {
    let [reference, test] = &o.positional[..] else {
        return Err("usage: eval moc REF.wav TEST.wav [--apply-vad]".into());
    };
    let read = |p: &str| -> Result<Vec<f32>, String> {
        let w = signals::read_wav(Path::new(p))?;
        if w.rate != 16_000 || w.channels != 1 {
            return Err(format!("{p}: MOC needs 16 kHz mono, found {} Hz with {} channel(s)", w.rate, w.channels));
        }
        Ok(w.samples.iter().map(|&v| v as f32).collect())
    };
    let v = opus1::internals::eval::moc::moc(&read(reference)?, &read(test)?, o.apply_vad).map_err(|e| e.to_string())?;
    println!("MOC: {v}");
    Ok(0)
}

fn cmd_corpus_index(o: &Opts) -> Result<i32, String> {
    let (Some(root), Some(out)) = (&o.root, &o.out) else {
        return Err("usage: eval corpus-index --root DIR --out FILE [--per-group N] [--class CLASS] [--corpus DIR]".into());
    };
    let class = match &o.class {
        Some(c) => corpus::Class::parse(c).ok_or_else(|| format!("unknown class {c}"))?,
        None => corpus::Class::Speech,
    };
    let text = corpus::corpus_index(Path::new(root), &o.corpus(), o.per_group.unwrap_or(10), class)?;
    write_file(out, &text)?;
    println!("wrote {out}: {} items", text.lines().filter(|l| !l.starts_with('#')).count());
    Ok(0)
}

fn cmd_merge(o: &Opts) -> Result<i32, String> {
    let [run, tier2] = &o.positional[..] else {
        return Err("usage: eval merge RUN.json TIER2.jsonl [--json PATH]".into());
    };
    let (text, json) = tier2::merge(run, tier2)?;
    print!("{text}");
    if let Some(path) = &o.json {
        write_file(path, &json)?;
    }
    Ok(0)
}

pub fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(command) = args.first() else {
        eprintln!("usage: eval list|run|selftest|conformance|moc|corpus-index|merge [options]");
        std::process::exit(2);
    };
    let result = Opts::parse(&args[1..]).and_then(|o| match command.as_str() {
        "list" => cmd_list(&o),
        "run" => cmd_run(&o),
        "selftest" => cmd_selftest(&o),
        "conformance" => cmd_conformance(&o),
        "moc" => cmd_moc(&o),
        "corpus-index" => cmd_corpus_index(&o),
        "merge" => cmd_merge(&o),
        other => Err(format!("unknown command {other}")),
    });
    match result {
        Ok(code) => std::process::exit(code),
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(2);
        }
    }
}
