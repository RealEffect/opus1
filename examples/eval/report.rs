//! Output: the JSON file of a run, a short text report and Markdown
//! tables, and a small JSON reader for `--baseline` and `merge`.
//!
//! The JSON file has one result row, one aggregate and one gate per line,
//! so that line-based tools can read it too.

use crate::corpus::Item;
use crate::pipeline::{Condition, Mode, Row, Side};
use crate::sets::EvalSet;
use crate::stats::{Aggregate, GateResult, Stored, StoredJob};
use std::fmt::Write as _;

/// What a run reports about itself.
pub struct RunInfo {
    pub set: String,
    pub corpus_root: Option<String>,
    pub items_fnv64: u64,
    pub decoder: String,
    pub degrade: Option<String>,
    pub threads: usize,
    pub seconds: f64,
}

pub fn json_string(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 2);
    out.push('"');
    for ch in text.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// A number as JSON: the shortest decimal that reads back to the same
/// value, or `null` when it is not finite.
pub fn num(v: f64) -> String {
    if v.is_finite() { format!("{v}") } else { "null".into() }
}

fn opt(v: Option<f64>) -> String {
    v.map_or("null".into(), num)
}

fn side_json(s: &Side) -> String {
    format!(
        "{{\"bytes\":{},\"kbps\":{},\"packets\":{},\"toc\":{},\"lookahead\":{},\"lost\":{},\"fec_calls\":{},\"range_checks\":{},\
         \"err\":{},\"q\":{},\"worst_frame\":{},\"snr_db\":{},\"best_shift\":{},\"moc\":{},\"moc_vad\":{}}}",
        s.stream.bytes,
        num((s.stream.kbps * 1000.0).round() / 1000.0),
        s.stream.packets,
        json_string(&s.stream.toc),
        s.lookahead,
        s.stream.lost,
        s.stream.fec_calls,
        s.stream.range_checks,
        num(s.err),
        if s.q.is_finite() { format!("{}", s.q) } else { "null".into() },
        s.worst_frame,
        num(s.snr_db),
        s.best_shift,
        opt(s.moc),
        opt(s.moc_vad)
    )
}

fn condition_json(c: &Condition) -> String {
    let mode = match c.mode {
        None => "null".to_string(),
        Some(Mode::Silk) => json_string("silk"),
        Some(Mode::Hybrid) => json_string("hybrid"),
        Some(Mode::Celt) => json_string("celt"),
    };
    let schedule: Vec<String> = c.schedule.iter().map(|(t, b)| format!("[{},{b}]", num(*t))).collect();
    let tags: Vec<String> = c.tags.iter().map(|t| json_string(t)).collect();
    format!(
        "{{\"name\":{},\"group\":{},\"application\":{},\"rate\":{},\"channels\":{},\"bitrate\":{},\"cbr\":{},\
         \"complexity\":{},\"frame_us\":{},\"fec\":{},\"loss_perc\":{},\"decode_loss\":{},\"mode\":{mode},\
         \"schedule\":[{}],\"tags\":[{}]}}",
        json_string(&c.name),
        json_string(&c.group),
        json_string(c.app.name()),
        c.rate,
        c.channels,
        c.bitrate,
        c.cbr,
        c.complexity,
        c.frame_us,
        c.fec,
        c.loss_perc,
        c.decode_loss,
        schedule.join(","),
        tags.join(",")
    )
}

fn row_json(set: &EvalSet, items: &[Item], row: &Row, degrade: Option<&str>) -> String {
    let c = &set.conditions[row.cond];
    let item = &items[row.item];
    let tags: Vec<String> = c.tags.iter().map(|t| json_string(t)).collect();
    let mut s = format!(
        "{{\"condition\":{},\"group\":{},\"tags\":[{}],\"item\":{},\"class\":{},\"split\":{},\"seconds\":{},",
        json_string(&c.name),
        json_string(&c.group),
        tags.join(","),
        json_string(&item.name),
        json_string(item.class.name()),
        json_string(item.split.name()),
        num(item.seconds())
    );
    match &row.result {
        Ok(p) => {
            let _ = write!(
                s,
                "\"opus1\":{},\"libopus\":{},\"ratio_err\":{},\"delta_snr_db\":{},\"ratio_bytes\":{},\"moc_d\":{},",
                side_json(&p.opus1),
                side_json(&p.libopus),
                num(p.ratio_err),
                num(p.delta_snr_db),
                num(p.ratio_bytes),
                opt(p.moc_d)
            );
        }
        Err(e) => {
            let _ = write!(s, "\"error\":{},", json_string(e));
        }
    }
    let _ = write!(s, "\"degrade\":{}}}", degrade.map_or("null".into(), json_string));
    s
}

fn aggregate_json(set: &EvalSet, a: &Aggregate) -> String {
    let c = &set.conditions[a.cond];
    let worst: Vec<String> = a.worst.iter().map(|(i, r)| format!("[{},{}]", json_string(i), num(*r))).collect();
    format!(
        "{{\"condition\":{},\"group\":{},\"bitrate\":{},\"frame_us\":{},\"complexity\":{},\"channels\":{},\"items\":{},\"errors\":{},\
         \"geomean_ratio_err\":{},\"ci95\":[{},{}],\"max_ratio_err\":{},\"max_item\":{},\"worst\":[{}],\
         \"geomean_err_opus1\":{},\"geomean_err_libopus\":{},\"geomean_ratio_bytes\":{},\"min_ratio_bytes\":{},\"max_ratio_bytes\":{},\
         \"kbps_opus1\":{},\"kbps_libopus\":{},\"mean_delta_snr_db\":{},\"mean_moc_d\":{},\"min_moc_d\":{},\
         \"geomean_tune\":{},\"geomean_holdout\":{},\"holdout_items\":{}}}",
        json_string(&c.name),
        json_string(&c.group),
        c.bitrate,
        c.frame_us,
        c.complexity,
        c.channels,
        a.items,
        a.errors,
        num(a.geomean_ratio_err),
        num(a.ci95.0),
        num(a.ci95.1),
        num(a.max_ratio_err),
        json_string(&a.max_item),
        worst.join(","),
        num(a.geomean_err_opus1),
        num(a.geomean_err_libopus),
        num(a.geomean_ratio_bytes),
        num(a.min_ratio_bytes),
        num(a.max_ratio_bytes),
        num(a.kbps_opus1),
        num(a.kbps_libopus),
        num(a.mean_delta_snr_db),
        opt(a.mean_moc_d),
        opt(a.min_moc_d),
        opt(a.geomean_tune),
        opt(a.geomean_holdout),
        a.holdout_items
    )
}

fn gate_json(g: &GateResult) -> String {
    format!(
        "{{\"name\":{},\"group\":{},\"rule\":{},\"limit\":{},\"value\":{},\"pass\":{},\"detail\":{}}}",
        json_string(&g.name),
        json_string(&g.group),
        json_string(g.kind.name()),
        num(g.limit),
        num(g.value),
        g.pass,
        json_string(&g.detail)
    )
}

pub fn git_revision() -> String {
    let run = |args: &[&str]| {
        std::process::Command::new("git")
            .args(["-C", env!("CARGO_MANIFEST_DIR")])
            .args(args)
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
    };
    match run(&["rev-parse", "--short", "HEAD"]) {
        Some(head) => {
            let dirty = run(&["status", "--porcelain", "--untracked-files=no"]).is_some_and(|s| !s.is_empty());
            if dirty { format!("{head}+changes") } else { head }
        }
        None => "unknown".into(),
    }
}

/// The JSON file of a run.
pub fn run_json(
    info: &RunInfo,
    set: &EvalSet,
    items: &[Item],
    rows: &[Row],
    aggregates: &[Aggregate],
    gates: &[GateResult],
) -> String {
    let errors = rows.iter().filter(|r| r.result.is_err()).count();
    let pass = errors == 0 && gates.iter().all(|g| g.pass);
    let mut s = String::new();
    let _ = writeln!(
        s,
        "{{\"tool\":\"opus1 eval\",\"schema\":1,\"opus1\":{},\"git\":{},\"libopus\":{},\"libopus_lib_dir\":{},\
         \"target\":{},\"set\":{},\"corpus_root\":{},\"items\":{},\"items_fnv64\":\"{:016x}\",\"decoder\":{},\"degrade\":{},\
         \"threads\":{},\"seconds\":{},\"jobs\":{},\"errors\":{errors},",
        json_string(env!("CARGO_PKG_VERSION")),
        json_string(&git_revision()),
        json_string(&crate::libopus::version()),
        json_string(option_env!("LIBOPUS_LIB_DIR").unwrap_or("system")),
        json_string(&format!("{}-{}", std::env::consts::ARCH, std::env::consts::OS)),
        json_string(&info.set),
        info.corpus_root.as_deref().map_or("null".into(), json_string),
        items.len(),
        info.items_fnv64,
        json_string(&info.decoder),
        info.degrade.as_deref().map_or("null".into(), json_string),
        info.threads,
        num((info.seconds * 10.0).round() / 10.0),
        rows.len()
    );
    s.push_str("\"conditions\":[\n");
    let conds: Vec<String> = set.conditions.iter().map(condition_json).collect();
    s.push_str(&conds.join(",\n"));
    s.push_str("],\n\"rows\":[\n");
    let lines: Vec<String> = rows.iter().map(|r| row_json(set, items, r, info.degrade.as_deref())).collect();
    s.push_str(&lines.join(",\n"));
    s.push_str("],\n\"aggregates\":[\n");
    let lines: Vec<String> = aggregates.iter().map(|a| aggregate_json(set, a)).collect();
    s.push_str(&lines.join(",\n"));
    s.push_str("],\n\"gates\":[\n");
    let lines: Vec<String> = gates.iter().map(gate_json).collect();
    s.push_str(&lines.join(",\n"));
    let _ = writeln!(s, "],\n\"pass\":{pass}}}");
    s
}

/// The short text report.
pub fn text_report(
    info: &RunInfo,
    set: &EvalSet,
    items: &[Item],
    rows: &[Row],
    aggregates: &[Aggregate],
    gates: &[GateResult],
) -> String {
    let errors: Vec<&Row> = rows.iter().filter(|r| r.result.is_err()).collect();
    let mut s = String::new();
    let _ = writeln!(
        s,
        "opus1 eval: set {}, {} items, {} jobs, {} errors, {} threads, {:.1} s",
        info.set,
        items.len(),
        rows.len(),
        errors.len(),
        info.threads,
        info.seconds
    );
    let _ = writeln!(
        s,
        "{} ({}), opus1 {} at {}, decoder {}{}",
        crate::libopus::version(),
        option_env!("LIBOPUS_LIB_DIR").unwrap_or("system"),
        env!("CARGO_PKG_VERSION"),
        git_revision(),
        info.decoder,
        info.degrade.as_ref().map_or(String::new(), |d| format!(", degradation {d} on opus1"))
    );
    let _ = writeln!(s, "ratio_err = (err_opus1 + 1e-6) / (err_libopus + 1e-6) of opus_compare; geometric means over items.");
    let _ = writeln!(
        s,
        "\n{:<44} {:>4} {:>8} {:>19} {:>7}  {:<26} {:>7} {:>7} {:>7} {:>7}",
        "condition", "n", "ratio", "95% CI", "max", "worst item", "bytes", "dSNR", "MOC D", "err lib"
    );
    let mut last_group = "";
    for a in aggregates {
        let c = &set.conditions[a.cond];
        if c.group != last_group {
            let _ = writeln!(s, "[{}]", c.group);
            last_group = &c.group;
        }
        let worst: String = a.max_item.chars().rev().take(26).collect::<Vec<_>>().into_iter().rev().collect();
        let _ = writeln!(
            s,
            "{:<44} {:>4} {:>8.4} [{:>7.4}, {:>7.4}] {:>7.3}  {:<26} {:>7.4} {:>7.2} {:>7} {:>7.4}{}",
            c.name,
            a.items - a.errors,
            a.geomean_ratio_err,
            a.ci95.0,
            a.ci95.1,
            a.max_ratio_err,
            worst,
            a.geomean_ratio_bytes,
            a.mean_delta_snr_db,
            a.mean_moc_d.map_or("-".into(), |v| format!("{v:.4}")),
            a.geomean_err_libopus,
            if a.errors > 0 { format!("  ({} errors)", a.errors) } else { String::new() }
        );
    }
    let failed: Vec<&GateResult> = gates.iter().filter(|g| !g.pass).collect();
    let _ = writeln!(s, "\ngates: {} pass, {} warn", gates.len() - failed.len(), failed.len());
    for g in failed {
        let _ = writeln!(s, "  WARN {}", g.line());
    }
    let mut ranked: Vec<(&Row, f64)> = rows.iter().filter_map(|r| r.result.as_ref().ok().map(|p| (r, p.ratio_err))).collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
    let _ = writeln!(s, "\nworst jobs by ratio_err:");
    for (r, v) in ranked.iter().take(10) {
        let p = r.result.as_ref().unwrap();
        let _ = writeln!(
            s,
            "  {v:.4}  {} {}  (err {:.4} vs {:.4}, bytes {:.4})",
            set.conditions[r.cond].name,
            items[r.item].name,
            p.opus1.err,
            p.libopus.err,
            p.ratio_bytes
        );
    }
    for r in errors.iter().take(20) {
        let _ = writeln!(
            s,
            "  ERROR {} {}: {}",
            set.conditions[r.cond].name,
            items[r.item].name,
            r.result.as_ref().unwrap_err()
        );
    }
    s
}

/// One Markdown table per group.
pub fn markdown(set: &EvalSet, aggregates: &[Aggregate], gates: &[GateResult]) -> String {
    let mut s = String::new();
    for group in &set.groups {
        let mine: Vec<&Aggregate> = aggregates.iter().filter(|a| set.conditions[a.cond].group == group.name).collect();
        if mine.is_empty() {
            continue;
        }
        let _ = writeln!(s, "### {}\n", group.name);
        let _ = writeln!(
            s,
            "| condition | items | ratio_err | 95 % CI | max | worst item | bytes ratio | kb/s opus1 | kb/s libopus | ΔSNR dB | MOC D |\n\
             |---|---:|---:|---|---:|---|---:|---:|---:|---:|---:|"
        );
        for a in mine {
            let _ = writeln!(
                s,
                "| {} | {} | {:.4} | {:.4} to {:.4} | {:.3} | {} | {:.4} | {:.2} | {:.2} | {:.2} | {} |",
                set.conditions[a.cond].name,
                a.items - a.errors,
                a.geomean_ratio_err,
                a.ci95.0,
                a.ci95.1,
                a.max_ratio_err,
                a.max_item,
                a.geomean_ratio_bytes,
                a.kbps_opus1,
                a.kbps_libopus,
                a.mean_delta_snr_db,
                a.mean_moc_d.map_or("-".into(), |v| format!("{v:.4}"))
            );
        }
        let mine: Vec<&GateResult> = gates.iter().filter(|g| g.group == group.name).collect();
        if !mine.is_empty() {
            let _ = writeln!(s, "\n| gate | limit | value | result |\n|---|---:|---:|---|");
            for g in mine {
                let _ = writeln!(s, "| {} | {} | {:.4} | {} |", g.name, g.limit, g.value, if g.pass { "pass" } else { "warn" });
            }
        }
        s.push('\n');
    }
    s
}

// ---------------------------------------------------------------------------
// JSON reader
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

impl Json {
    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Obj(fields) => fields.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        if let Json::Num(v) = self { Some(*v) } else { None }
    }

    pub fn as_str(&self) -> Option<&str> {
        if let Json::Str(s) = self { Some(s) } else { None }
    }

    pub fn as_arr(&self) -> &[Json] {
        if let Json::Arr(v) = self { v } else { &[] }
    }
}

pub fn parse_json(text: &str) -> Result<Json, String> {
    let mut p = Parser { s: text.as_bytes(), i: 0 };
    let v = p.value()?;
    p.ws();
    if p.i != p.s.len() {
        return Err(format!("JSON: text after the value at byte {}", p.i));
    }
    Ok(v)
}

/// Parse a file of one JSON value per line (blank lines skipped).
pub fn parse_json_lines(text: &str) -> Result<Vec<Json>, String> {
    text.lines()
        .enumerate()
        .filter(|(_, l)| !l.trim().is_empty())
        .map(|(n, l)| parse_json(l).map_err(|e| format!("line {}: {e}", n + 1)))
        .collect()
}

struct Parser<'a> {
    s: &'a [u8],
    i: usize,
}

impl Parser<'_> {
    fn ws(&mut self) {
        while self.i < self.s.len() && self.s[self.i].is_ascii_whitespace() {
            self.i += 1;
        }
    }

    fn err<T>(&self, what: &str) -> Result<T, String> {
        Err(format!("JSON: {what} at byte {}", self.i))
    }

    fn eat(&mut self, word: &str) -> bool {
        if self.s[self.i..].starts_with(word.as_bytes()) {
            self.i += word.len();
            true
        } else {
            false
        }
    }

    fn value(&mut self) -> Result<Json, String> {
        self.ws();
        let Some(&c) = self.s.get(self.i) else { return self.err("unexpected end") };
        match c {
            b'{' => {
                self.i += 1;
                let mut fields = Vec::new();
                self.ws();
                if self.eat("}") {
                    return Ok(Json::Obj(fields));
                }
                loop {
                    self.ws();
                    let Json::Str(key) = self.value()? else { return self.err("expected a key") };
                    self.ws();
                    if !self.eat(":") {
                        return self.err("expected ':'");
                    }
                    let v = self.value()?;
                    fields.push((key, v));
                    self.ws();
                    if self.eat("}") {
                        return Ok(Json::Obj(fields));
                    }
                    if !self.eat(",") {
                        return self.err("expected ',' or '}'");
                    }
                }
            }
            b'[' => {
                self.i += 1;
                let mut items = Vec::new();
                self.ws();
                if self.eat("]") {
                    return Ok(Json::Arr(items));
                }
                loop {
                    items.push(self.value()?);
                    self.ws();
                    if self.eat("]") {
                        return Ok(Json::Arr(items));
                    }
                    if !self.eat(",") {
                        return self.err("expected ',' or ']'");
                    }
                }
            }
            b'"' => {
                self.i += 1;
                let mut out = String::new();
                loop {
                    let Some(&c) = self.s.get(self.i) else { return self.err("unterminated string") };
                    self.i += 1;
                    match c {
                        b'"' => return Ok(Json::Str(out)),
                        b'\\' => {
                            let Some(&e) = self.s.get(self.i) else { return self.err("bad escape") };
                            self.i += 1;
                            match e {
                                b'n' => out.push('\n'),
                                b't' => out.push('\t'),
                                b'r' => out.push('\r'),
                                b'b' => out.push('\u{8}'),
                                b'f' => out.push('\u{c}'),
                                b'u' => {
                                    let hex = std::str::from_utf8(self.s.get(self.i..self.i + 4).unwrap_or(&[]))
                                        .ok()
                                        .and_then(|h| u32::from_str_radix(h, 16).ok());
                                    let Some(code) = hex else { return self.err("bad \\u escape") };
                                    self.i += 4;
                                    out.push(char::from_u32(code).unwrap_or('\u{fffd}'));
                                }
                                other => out.push(other as char),
                            }
                        }
                        _ => {
                            // Copy one UTF-8 sequence.
                            let start = self.i - 1;
                            let mut end = self.i;
                            while end < self.s.len() && (self.s[end] & 0xC0) == 0x80 {
                                end += 1;
                            }
                            out.push_str(std::str::from_utf8(&self.s[start..end]).unwrap_or("\u{fffd}"));
                            self.i = end;
                        }
                    }
                }
            }
            b't' if self.eat("true") => Ok(Json::Bool(true)),
            b'f' if self.eat("false") => Ok(Json::Bool(false)),
            b'n' if self.eat("null") => Ok(Json::Null),
            _ => {
                let start = self.i;
                while self.i < self.s.len() && matches!(self.s[self.i], b'-' | b'+' | b'.' | b'e' | b'E' | b'0'..=b'9') {
                    self.i += 1;
                }
                let text = std::str::from_utf8(&self.s[start..self.i]).unwrap_or("");
                match text.parse::<f64>() {
                    Ok(v) => Ok(Json::Num(v)),
                    Err(_) => self.err("bad value"),
                }
            }
        }
    }
}

/// The `err` values of a stored run (`--baseline`).
pub fn read_stored(path: &str) -> Result<Stored, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    let json = parse_json(&text).map_err(|e| format!("{path}: {e}"))?;
    let mut out = Stored::new();
    for row in json.get("rows").map_or(&[][..], Json::as_arr) {
        let field = |k: &str| row.get(k).and_then(Json::as_str).map(str::to_string);
        let value = |side: &str, k: &str| row.get(side).and_then(|s| s.get(k)).and_then(Json::as_f64);
        if let (Some(c), Some(i), Some(a), Some(b)) =
            (field("condition"), field("item"), value("opus1", "err"), value("libopus", "err"))
        {
            let snr = value("opus1", "snr_db").unwrap_or(f64::NAN);
            out.insert((c, i), StoredJob { err_opus1: a, err_libopus: b, snr_opus1: snr });
        }
    }
    Ok(out)
}
