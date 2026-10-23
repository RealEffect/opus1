//! Report of an A/B timing: two builds of opus1 ran `perf_harness bench` in
//! turns on one computer (scripts/x86/ab.sh and scripts/x86/ab.ps1). A is
//! the baseline, B the branch.
//!
//! ```text
//! rustc --edition 2024 -O -o ab_report scripts/x86/ab_report.rs
//! ab_report [--a NAME] [--b NAME] [--order TEXT] [--summary PATH]
//!           A1.json B1.json [A2.json B2.json ...]
//! ```
//!
//! The `bench --json` files come in pairs, A first. The two runs of a pair
//! ran one right after the other, so the computer was in the same state for
//! both. For each scenario and each pair the report divides the median time
//! per frame of B by that of A. This is `B/A`: below 1 means B is faster.
//!
//! It does this for the four timed operations: opus1 encode, opus1 decode,
//! libopus encode and libopus decode. libopus is the same library in both
//! programs, so its B/A is a control and should stay near 1. The harness
//! alternates opus1 and libopus within each run. So the opus1 B/A divided by
//! the libopus B/A of the same scenario and pair (`opus1/libopus B/A`) no
//! longer contains most of a change of the computer's speed between the two
//! runs.
//!
//! It prints Markdown on standard output. With `--summary` it also writes the
//! geometric means as a few lines of plain text. The output is ASCII only, so
//! that every console shows it.
//!
//! The program needs only the standard library, so rustc alone builds it on
//! Linux, macOS and Windows.

use std::fmt::Write as _;
use std::process::ExitCode;

const USAGE: &str = "usage: ab_report [--a NAME] [--b NAME] [--order TEXT] [--summary PATH] \
A1.json B1.json [A2.json B2.json ...]

The bench --json files come in pairs, the baseline (A) first.";

/// The four timed operations, as the keys of a bench JSON record name them.
const OPERATIONS: [&str; 4] = ["rust_enc", "c_enc", "rust_dec", "c_dec"];

/// What the report divides, B by A. The order of `ratios`.
const QUANTITIES: [&str; 6] = [
    "opus1 encode",
    "opus1 decode",
    "libopus encode (control)",
    "libopus decode (control)",
    "opus1/libopus encode",
    "opus1/libopus decode",
];

/// Groups of scenarios: a label and the start of the scenario names. The
/// empty start takes every scenario.
const GROUPS: [(&str, &str); 4] = [("all", ""), ("SILK", "silk_"), ("Hybrid", "hybrid_"), ("CELT", "celt_")];

/// One scenario in one bench JSON file.
struct Row {
    name: String,
    /// Median time per frame in nanoseconds, in the order of `OPERATIONS`.
    medians: [f64; 4],
    /// Bit rate of the opus1 stream in kb/s.
    kbps: Option<f64>,
    /// The harness marked the timing unsteady (its list `unsteady` is not
    /// empty).
    unsteady: bool,
}

impl Row {
    /// Time of opus1 divided by the time of libopus: encode, decode.
    fn levels(&self) -> [f64; 2] {
        [self.medians[0] / self.medians[1], self.medians[2] / self.medians[3]]
    }
}

/// B/A of each of the `QUANTITIES` for one scenario in one pair.
fn ratios(a: &Row, b: &Row) -> [f64; 6] {
    let r = |operation: usize| b.medians[operation] / a.medians[operation];
    [r(0), r(2), r(1), r(3), r(0) / r(1), r(2) / r(3)]
}

/// One scenario over all pairs.
struct Scenario<'a> {
    name: &'a str,
    /// Per quantity, B/A of each pair.
    ratios: [Vec<f64>; 6],
    /// opus1/libopus of A and of B (encode, decode), the median over the
    /// pairs.
    levels_a: [f64; 2],
    levels_b: [f64; 2],
    kbps_a: Option<f64>,
    kbps_b: Option<f64>,
}

/// Value of a `"key":"text"` field in one line of a bench JSON file.
fn text_field(line: &str, key: &str) -> Option<String> {
    let rest = &line[line.find(&format!("\"{key}\":\""))? + key.len() + 4..];
    Some(rest[..rest.find('"')?].to_string())
}

/// Value of a `"key":number` field in one line of a bench JSON file.
fn number_field(line: &str, key: &str) -> Option<f64> {
    let rest = &line[line.find(&format!("\"{key}\":"))? + key.len() + 3..];
    let end = rest.find([',', '}']).unwrap_or(rest.len());
    rest[..end].trim().parse().ok()
}

fn read_rows(path: &str) -> Result<Vec<Row>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    let mut rows = Vec::new();
    for line in text.lines() {
        let Some(name) = text_field(line, "name") else {
            continue;
        };
        let mut medians = [0.0; 4];
        for (median, operation) in medians.iter_mut().zip(OPERATIONS) {
            *median = number_field(line, &format!("{operation}_med_ns"))
                .filter(|value| *value > 0.0)
                .ok_or_else(|| format!("{path}: {name} has no {operation}_med_ns"))?;
        }
        rows.push(Row {
            kbps: number_field(line, "rust_kbps"),
            unsteady: line.contains("\"unsteady\":[") && !line.contains("\"unsteady\":[]"),
            name,
            medians,
        });
    }
    if rows.is_empty() {
        return Err(format!("{path}: no scenario in this file"));
    }
    Ok(rows)
}

fn find<'a>(rows: &'a [Row], name: &str) -> Option<&'a Row> {
    rows.iter().find(|row| row.name == name)
}

fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let n = sorted.len();
    if n % 2 == 1 { sorted[n / 2] } else { (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0 }
}

fn geomean(values: impl Iterator<Item = f64>) -> f64 {
    let (mut log_sum, mut count) = (0.0, 0usize);
    for value in values {
        log_sum += value.ln();
        count += 1;
    }
    (log_sum / count as f64).exp()
}

fn lowest(values: &[f64]) -> f64 {
    values.iter().copied().fold(f64::INFINITY, f64::min)
}

fn highest(values: &[f64]) -> f64 {
    values.iter().copied().fold(f64::NEG_INFINITY, f64::max)
}

/// "0.912 (0.905 to 0.920)": a value, and the lowest and highest of
/// `range`; only the value when there is one pair.
fn with_range(value: f64, range: &[f64]) -> String {
    if range.len() < 2 {
        format!("{value:.3}")
    } else {
        format!("{value:.3} ({:.3} to {:.3})", lowest(range), highest(range))
    }
}

/// `faster` when B/A is below 1 in every pair, `slower` when above 1 in
/// every pair. Nothing when there is only one pair.
fn sign(values: &[f64]) -> &'static str {
    if values.len() < 2 {
        ""
    } else if highest(values) < 1.0 {
        "faster"
    } else if lowest(values) > 1.0 {
        "slower"
    } else {
        ""
    }
}

fn file_name(path: &str) -> &str {
    path.rsplit(['/', '\\']).next().unwrap_or(path)
}

fn next_value(args: &mut impl Iterator<Item = String>, option: &str) -> Result<String, String> {
    args.next().ok_or_else(|| format!("{option} needs a value\n\n{USAGE}"))
}

fn run() -> Result<(), String> {
    let mut name_a = "A".to_string();
    let mut name_b = "B".to_string();
    let mut order = None;
    let mut summary_path = None;
    let mut files = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--a" => name_a = next_value(&mut args, "--a")?,
            "--b" => name_b = next_value(&mut args, "--b")?,
            "--order" => order = Some(next_value(&mut args, "--order")?),
            "--summary" => summary_path = Some(next_value(&mut args, "--summary")?),
            "--help" | "-h" => {
                println!("{USAGE}");
                return Ok(());
            }
            _ if arg.starts_with("--") => return Err(format!("unknown option {arg}\n\n{USAGE}")),
            _ => files.push(arg),
        }
    }
    if files.len() < 2 || files.len() % 2 != 0 {
        return Err(format!("give the bench JSON files in pairs, A first\n\n{USAGE}"));
    }
    let runs = files.iter().map(|path| read_rows(path)).collect::<Result<Vec<_>, _>>()?;
    let pairs = runs.len() / 2;

    // The scenarios that are in every file, in the order of the first file.
    let mut scenarios = Vec::new();
    let mut left_out = Vec::new();
    for row in &runs[0] {
        let found: Option<Vec<(&Row, &Row)>> = (0..pairs)
            .map(|pair| Some((find(&runs[2 * pair], &row.name)?, find(&runs[2 * pair + 1], &row.name)?)))
            .collect();
        let Some(found) = found else {
            left_out.push(row.name.as_str());
            continue;
        };
        let mut per_quantity: [Vec<f64>; 6] = Default::default();
        let mut levels: [[Vec<f64>; 2]; 2] = Default::default();
        for (a, b) in &found {
            for (values, ratio) in per_quantity.iter_mut().zip(ratios(a, b)) {
                values.push(ratio);
            }
            for (side, row) in [*a, *b].into_iter().enumerate() {
                for (values, level) in levels[side].iter_mut().zip(row.levels()) {
                    values.push(level);
                }
            }
        }
        let level_medians = |side: usize| [median(&levels[side][0]), median(&levels[side][1])];
        scenarios.push(Scenario {
            name: row.name.as_str(),
            levels_a: level_medians(0),
            levels_b: level_medians(1),
            ratios: per_quantity,
            kbps_a: found[0].0.kbps,
            kbps_b: found[0].1.kbps,
        });
    }
    if scenarios.is_empty() {
        return Err("no scenario is in every file".to_string());
    }

    let mut md = String::new();
    let _ = writeln!(md, "# A/B timing: B against A\n");
    let _ = writeln!(md, "- A (baseline): {name_a}");
    let _ = writeln!(md, "- B (branch): {name_b}");
    if let Some(order) = &order {
        let _ = writeln!(md, "- order of the runs: {order}");
    }
    let _ = writeln!(
        md,
        "- {pairs} {}, {} scenarios; a pair is a run of A and a run of B, one right after the other",
        if pairs == 1 { "pair" } else { "pairs" },
        scenarios.len()
    );
    let _ = writeln!(
        md,
        "- each run is one `perf_harness bench`; the time of an operation is its median time per frame in that run"
    );
    let _ = writeln!(
        md,
        "- B/A = time of B / time of A, in the same pair. Below 1 means B is faster: 0.900 means 10 % less time"
    );
    let _ = writeln!(
        md,
        "- libopus is the same library in A and B, so its B/A is a control and should stay near 1. \
When it moves, the computer ran faster or slower during one of the two runs"
    );
    let _ = writeln!(
        md,
        "- opus1/libopus B/A = opus1 B/A divided by libopus B/A of the same scenario and pair. \
The harness alternates opus1 and libopus within a run, so this removes most of a change of the computer's speed. \
Use it when the control moved"
    );

    // Geometric means per group.
    let groups: Vec<(&str, Vec<&Scenario>)> = GROUPS
        .iter()
        .map(|&(label, start)| (label, scenarios.iter().filter(|s| s.name.starts_with(start)).collect::<Vec<_>>()))
        .filter(|(label, members)| *label == "all" || !members.is_empty())
        .collect();
    // Per group and quantity: the geometric mean of the per-scenario
    // medians, and the geometric mean of each single pair.
    let means = |members: &[&Scenario], quantity: usize| -> (f64, Vec<f64>) {
        let of_medians = geomean(members.iter().map(|s| median(&s.ratios[quantity])));
        let per_pair = (0..pairs).map(|pair| geomean(members.iter().map(|s| s.ratios[quantity][pair]))).collect();
        (of_medians, per_pair)
    };
    let _ = writeln!(md, "\n## Geometric means of B/A\n");
    let _ = writeln!(
        md,
        "Over the scenarios of each group: the geometric mean of the per-scenario medians over the pairs; \
in brackets the lowest and the highest geometric mean of a single pair.\n"
    );
    let mut header = "| |".to_string();
    let mut rule = "|---|".to_string();
    for (label, members) in &groups {
        let _ = write!(header, " {label} ({}) |", members.len());
        rule.push_str("---:|");
    }
    let _ = writeln!(md, "{header}\n{rule}");
    for (quantity, label) in QUANTITIES.iter().enumerate() {
        let mut line = format!("| {label} |");
        for (_, members) in &groups {
            let (value, per_pair) = means(members, quantity);
            let _ = write!(line, " {} |", with_range(value, &per_pair));
        }
        let _ = writeln!(md, "{line}");
    }

    let _ = writeln!(md, "\n## Where A and B stand against libopus\n");
    let _ = writeln!(
        md,
        "Time of opus1 / time of libopus in the same run (below 1 means opus1 is faster): \
the geometric mean over the scenarios of each scenario's median over its runs.\n"
    );
    let _ = writeln!(md, "{header}\n{rule}");
    for (index, label) in ["encode", "decode"].iter().enumerate() {
        for (side, side_name) in ["A", "B"].iter().enumerate() {
            let mut line = format!("| opus1/libopus {label}, {side_name} |");
            for (_, members) in &groups {
                let value = geomean(members.iter().map(|s| if side == 0 { s.levels_a[index] } else { s.levels_b[index] }));
                let _ = write!(line, " {value:.3} |");
            }
            let _ = writeln!(md, "{line}");
        }
    }

    let _ = writeln!(md, "\n## Per scenario\n");
    let _ = writeln!(
        md,
        "B/A: the median over the pairs; in brackets the lowest and the highest pair. \
`faster` or `slower`: below or above 1 in every pair (with two pairs or more). \
The last two columns give opus1/libopus of A and of B (median over the runs).\n"
    );
    let _ = writeln!(
        md,
        "| scenario | opus1 encode | | opus1 decode | | libopus encode | libopus decode | opus1/libopus encode | opus1/libopus decode | opus1/libopus encode, A and B | opus1/libopus decode, A and B |"
    );
    let _ = writeln!(md, "|---|---:|---|---:|---|---:|---:|---:|---:|---:|---:|");
    for s in &scenarios {
        let cell = |quantity: usize| with_range(median(&s.ratios[quantity]), &s.ratios[quantity]);
        let _ = writeln!(
            md,
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {:.3}, {:.3} | {:.3}, {:.3} |",
            s.name,
            cell(0),
            sign(&s.ratios[0]),
            cell(1),
            sign(&s.ratios[1]),
            cell(2),
            cell(3),
            cell(4),
            cell(5),
            s.levels_a[0],
            s.levels_b[0],
            s.levels_a[1],
            s.levels_b[1],
        );
    }

    let _ = writeln!(md, "\n## Runs\n");
    let _ = writeln!(
        md,
        "Scenarios with unsteady timing: the harness found the median more than 3 % above the minimum in every round \
(other programs disturbed the run).\n"
    );
    let _ = writeln!(md, "| pair | A | B | unsteady in A | unsteady in B |\n|---:|---|---|---:|---:|");
    for pair in 0..pairs {
        let unsteady = |rows: &[Row]| rows.iter().filter(|row| row.unsteady).count();
        let _ = writeln!(
            md,
            "| {} | {} | {} | {} | {} |",
            pair + 1,
            file_name(&files[2 * pair]),
            file_name(&files[2 * pair + 1]),
            unsteady(&runs[2 * pair]),
            unsteady(&runs[2 * pair + 1]),
        );
    }

    // Different bit rates mean the two builds encode different streams, so
    // they do not do the same work.
    let mut notes = Vec::new();
    for s in &scenarios {
        if let (Some(a), Some(b)) = (s.kbps_a, s.kbps_b)
            && a > 0.0
            && (b / a - 1.0).abs() > 0.01
        {
            notes.push(format!(
                "- {}: the opus1 bit rate differs, {a:.2} kb/s in A and {b:.2} kb/s in B, so the two builds encode different streams",
                s.name
            ));
        }
    }
    if !left_out.is_empty() {
        notes.push(format!("- left out, because some file does not have them: {}", left_out.join(", ")));
    }
    if !notes.is_empty() {
        let _ = writeln!(md, "\n## Notes\n");
        for note in &notes {
            let _ = writeln!(md, "{note}");
        }
    }
    print!("{md}");

    if let Some(path) = summary_path {
        let all = &groups[0].1;
        let mut text = format!(
            "A/B timing, B/A (below 1 = B faster): geometric mean over {} scenarios{}\n",
            all.len(),
            if pairs > 1 { format!(", in brackets the lowest and highest of {pairs} pairs") } else { ", 1 pair".to_string() }
        );
        let _ = writeln!(text, "  A = {name_a}");
        let _ = writeln!(text, "  B = {name_b}");
        for (quantity, label) in QUANTITIES.iter().enumerate() {
            let (value, per_pair) = means(all, quantity);
            let _ = writeln!(text, "  {label:<26} {}", with_range(value, &per_pair));
        }
        std::fs::write(&path, text).map_err(|e| format!("{path}: {e}"))?;
    }
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("error: {message}");
            ExitCode::from(2)
        }
    }
}
