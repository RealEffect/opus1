//! Statistics over the paired results: geometric means, bootstrap
//! confidence intervals, aggregates per condition, and gates.
//!
//! A gate is a rule with a limit for one group of conditions. Gates only
//! warn: a run whose gates fail still exits with success, unless
//! `--strict` is given. The self-test uses them to show that each
//! degradation is detected.

use crate::corpus::{Item, Split};
use crate::pipeline::Row;
use crate::sets::EvalSet;
use crate::signals::XorShift;
use std::collections::BTreeMap;

/// `exp(mean(ln v))`. NaN for an empty slice.
pub fn geomean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    (values.iter().map(|v| v.ln()).sum::<f64>() / values.len() as f64).exp()
}

pub fn mean(values: &[f64]) -> f64 {
    if values.is_empty() { f64::NAN } else { values.iter().sum::<f64>() / values.len() as f64 }
}

/// Resamples of the bootstrap.
pub const BOOTSTRAP_RESAMPLES: usize = 2000;
const BOOTSTRAP_SEED: u64 = 0xB007_5742_0000_0001;

/// The 95 % confidence interval of the geometric mean: the 2.5 % and
/// 97.5 % points of the geometric means of 2000 resamples (drawn with
/// replacement, fixed seed).
pub fn bootstrap_ci(values: &[f64]) -> (f64, f64) {
    if values.len() < 2 {
        let g = geomean(values);
        return (g, g);
    }
    let logs: Vec<f64> = values.iter().map(|v| v.ln()).collect();
    let mut rng = XorShift(BOOTSTRAP_SEED);
    let n = logs.len();
    let mut means: Vec<f64> = (0..BOOTSTRAP_RESAMPLES)
        .map(|_| (0..n).map(|_| logs[rng.below(n as u64) as usize]).sum::<f64>() / n as f64)
        .collect();
    means.sort_by(f64::total_cmp);
    let at = |q: f64| means[((q * (BOOTSTRAP_RESAMPLES - 1) as f64).round() as usize).min(BOOTSTRAP_RESAMPLES - 1)].exp();
    (at(0.025), at(0.975))
}

/// The summary of one condition.
pub struct Aggregate {
    pub cond: usize,
    pub items: usize,
    pub errors: usize,
    pub geomean_ratio_err: f64,
    pub ci95: (f64, f64),
    pub max_ratio_err: f64,
    pub max_item: String,
    /// The five items with the largest `ratio_err`.
    pub worst: Vec<(String, f64)>,
    pub geomean_err_opus1: f64,
    pub geomean_err_libopus: f64,
    pub geomean_ratio_bytes: f64,
    pub min_ratio_bytes: f64,
    pub max_ratio_bytes: f64,
    pub kbps_opus1: f64,
    pub kbps_libopus: f64,
    pub mean_delta_snr_db: f64,
    pub mean_moc_d: Option<f64>,
    pub min_moc_d: Option<f64>,
    pub geomean_tune: Option<f64>,
    pub geomean_holdout: Option<f64>,
    pub holdout_items: usize,
}

/// One aggregate per condition, in the order of the set.
pub fn aggregate(set: &EvalSet, items: &[Item], rows: &[Row]) -> Vec<Aggregate> {
    let mut out = Vec::new();
    for (ci, _) in set.conditions.iter().enumerate() {
        let mine: Vec<&Row> = rows.iter().filter(|r| r.cond == ci).collect();
        if mine.is_empty() {
            continue;
        }
        let ok: Vec<(&Row, &crate::pipeline::Pair)> =
            mine.iter().filter_map(|r| r.result.as_ref().ok().map(|p| (*r, p))).collect();
        let ratios: Vec<f64> = ok.iter().map(|(_, p)| p.ratio_err).collect();
        let mut ranked: Vec<(String, f64)> = ok.iter().map(|(r, p)| (items[r.item].name.clone(), p.ratio_err)).collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        let bytes: Vec<f64> = ok.iter().map(|(_, p)| p.ratio_bytes).collect();
        let mocs: Vec<f64> = ok.iter().filter_map(|(_, p)| p.moc_d).collect();
        let split_geomean = |split: Split| -> Option<f64> {
            let v: Vec<f64> = ok.iter().filter(|(r, _)| items[r.item].split == split).map(|(_, p)| p.ratio_err).collect();
            (!v.is_empty()).then(|| geomean(&v))
        };
        let (max_item, max_ratio_err) = ranked.first().cloned().unwrap_or((String::new(), f64::NAN));
        out.push(Aggregate {
            cond: ci,
            items: mine.len(),
            errors: mine.len() - ok.len(),
            geomean_ratio_err: geomean(&ratios),
            ci95: bootstrap_ci(&ratios),
            max_ratio_err,
            max_item,
            worst: ranked.iter().take(5).cloned().collect(),
            geomean_err_opus1: geomean(&ok.iter().map(|(_, p)| p.opus1.err + 1e-6).collect::<Vec<_>>()),
            geomean_err_libopus: geomean(&ok.iter().map(|(_, p)| p.libopus.err + 1e-6).collect::<Vec<_>>()),
            geomean_ratio_bytes: geomean(&bytes),
            min_ratio_bytes: bytes.iter().copied().fold(f64::INFINITY, f64::min),
            max_ratio_bytes: bytes.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            kbps_opus1: mean(&ok.iter().map(|(_, p)| p.opus1.stream.kbps).collect::<Vec<_>>()),
            kbps_libopus: mean(&ok.iter().map(|(_, p)| p.libopus.stream.kbps).collect::<Vec<_>>()),
            mean_delta_snr_db: mean(&ok.iter().map(|(_, p)| p.delta_snr_db).collect::<Vec<_>>()),
            mean_moc_d: (!mocs.is_empty()).then(|| mean(&mocs)),
            min_moc_d: mocs.iter().copied().reduce(f64::min),
            geomean_tune: split_geomean(Split::Tune),
            geomean_holdout: split_geomean(Split::Holdout),
            holdout_items: ok.iter().filter(|(r, _)| items[r.item].split == Split::Holdout).count(),
        });
    }
    out
}

/// The rule of a gate. Each applies to the jobs of one group.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GateKind {
    /// At each bitrate of the group: the geometric mean of `ratio_err` is
    /// at most the limit.
    GeomeanErr,
    /// The largest `ratio_err` of any job is at most the limit.
    MaxErr,
    /// The largest `ratio_bytes` of VBR jobs with frames below 40 ms is at
    /// most the limit.
    BytesMax,
    /// The same for VBR jobs with frames of 40 ms and more, where opus1
    /// deliberately differs (docs/guide/10, "Deliberate differences").
    BytesMaxLong,
    /// The smallest `ratio_bytes` of VBR jobs is at least the limit.
    BytesMin,
    /// The largest `|ratio_bytes - 1|` of CBR jobs is at most the limit.
    BytesCbr,
    /// The smallest MOC D is at least the limit (the per-clip rule of the
    /// OSCE draft's relative test).
    MocClip,
    /// The mean MOC D is at least the limit.
    MocMean,
    /// Jobs where opus1's output is best aligned at a shift other than 0
    /// while libopus's is aligned at 0: at most the limit.
    Align,
    /// The mean of `delta_snr_db` (opus1 - libopus) is at least the limit.
    /// Only for groups where both codecs code the waveform (CELT at
    /// middle and high bitrates); there the signal-to-noise ratio shows a
    /// lost band or added noise that `opus_compare` hardly sees.
    DeltaSnr,
    /// With `--baseline`: the geometric mean over the jobs of a group of
    /// opus1's `err` now divided by opus1's stored `err` (each plus 1e-6).
    BaselineGeomean,
    /// With `--baseline`: the largest such ratio.
    BaselineMax,
    /// With `--baseline`: jobs whose libopus `err` differs from the stored
    /// one. The comparison with a stored run is valid only when this is 0.
    LibopusChanged,
    /// With `--baseline`, for groups with a `DeltaSnr` gate: the mean of
    /// opus1's signal-to-noise ratio now minus the stored one (dB).
    BaselineSnr,
}

impl GateKind {
    pub fn name(self) -> &'static str {
        match self {
            GateKind::GeomeanErr => "geomean_err",
            GateKind::MaxErr => "max_err",
            GateKind::BytesMax => "bytes",
            GateKind::BytesMaxLong => "bytes_long",
            GateKind::BytesMin => "bytes_min",
            GateKind::BytesCbr => "bytes_cbr",
            GateKind::MocClip => "moc_clip",
            GateKind::MocMean => "moc_mean",
            GateKind::Align => "align",
            GateKind::DeltaSnr => "delta_snr",
            GateKind::BaselineGeomean => "baseline_geomean",
            GateKind::BaselineMax => "baseline_max",
            GateKind::LibopusChanged => "libopus_changed",
            GateKind::BaselineSnr => "baseline_snr",
        }
    }

    /// Whether the gate compares with a stored run (`--baseline`).
    pub fn needs_stored(self) -> bool {
        matches!(
            self,
            GateKind::BaselineGeomean | GateKind::BaselineMax | GateKind::LibopusChanged | GateKind::BaselineSnr
        )
    }

    /// Whether the value must stay at or below the limit (otherwise at or
    /// above).
    pub fn upper(self) -> bool {
        !matches!(
            self,
            GateKind::BytesMin | GateKind::MocClip | GateKind::MocMean | GateKind::DeltaSnr | GateKind::BaselineSnr
        )
    }
}

#[derive(Clone, Debug)]
pub struct Gate {
    pub kind: GateKind,
    pub limit: f64,
    /// For `GeomeanErr`: the bitrate this limit is for. A gate without a
    /// bitrate covers the bitrates that have no gate of their own.
    pub bitrate: Option<i32>,
}

impl Gate {
    pub fn new(kind: GateKind, limit: f64) -> Self {
        Self { kind, limit, bitrate: None }
    }
}

#[derive(Clone, Debug)]
pub struct GateResult {
    pub name: String,
    pub group: String,
    pub kind: GateKind,
    pub limit: f64,
    pub value: f64,
    pub pass: bool,
    /// The job that decides the value, where there is one.
    pub detail: String,
}

impl GateResult {
    pub fn line(&self) -> String {
        let op = if self.pass {
            if self.kind.upper() { "<=" } else { ">=" }
        } else if self.kind.upper() {
            ">"
        } else {
            "<"
        };
        let mut s = format!("{}: {:.4} {op} {}", self.name, self.value, self.limit);
        if !self.detail.is_empty() {
            s += &format!(" ({})", self.detail);
        }
        s
    }
}

/// The job with the largest (or smallest) value of `f` among `jobs`. A tie
/// keeps the first.
fn extreme<'a>(
    jobs: &[(&'a Row, &'a crate::pipeline::Pair)],
    f: &dyn Fn(&crate::pipeline::Pair) -> f64,
    largest: bool,
) -> Option<(f64, &'a Row)> {
    jobs.iter().map(|(r, p)| (f(p), *r)).reduce(|a, b| if (b.0 > a.0) == largest && b.0 != a.0 { b } else { a })
}

/// Evaluate the gates of every group over the rows that succeeded. A gate
/// with no job to look at is left out.
pub fn evaluate_gates(set: &EvalSet, items: &[Item], rows: &[Row]) -> Vec<GateResult> {
    let mut out = Vec::new();
    for group in &set.groups {
        let mine: Vec<(&Row, &crate::pipeline::Pair)> = rows
            .iter()
            .filter(|r| set.conditions[r.cond].group == group.name)
            .filter_map(|r| r.result.as_ref().ok().map(|p| (r, p)))
            .collect();
        if mine.is_empty() {
            continue;
        }
        let label = |r: &Row| format!("{} {}", set.conditions[r.cond].name, items[r.item].name);
        for gate in &group.gates {
            let mut push = |name: String, value: f64, detail: String| {
                let pass = if gate.kind.upper() { value <= gate.limit } else { value >= gate.limit };
                // NaN (no value) passes.
                let pass = pass || value.is_nan();
                out.push(GateResult { name, group: group.name.clone(), kind: gate.kind, limit: gate.limit, value, pass, detail });
            };
            let base = format!("{} {}", group.name, gate.kind.name());
            match gate.kind {
                GateKind::GeomeanErr => {
                    let mut by_rate: BTreeMap<i32, Vec<f64>> = BTreeMap::new();
                    for (r, p) in &mine {
                        by_rate.entry(set.conditions[r.cond].bitrate).or_default().push(p.ratio_err);
                    }
                    // A gate for one bitrate, or the general one for the
                    // bitrates without their own.
                    let own = |rate: i32| group.gates.iter().any(|g| g.kind == GateKind::GeomeanErr && g.bitrate == Some(rate));
                    by_rate.retain(|&rate, _| match gate.bitrate {
                        Some(b) => b == rate,
                        None => !own(rate),
                    });
                    for (rate, v) in by_rate {
                        push(format!("{base} {rate}"), geomean(&v), format!("{} jobs", v.len()));
                    }
                }
                GateKind::MaxErr => {
                    if let Some((v, r)) = extreme(&mine, &|p| p.ratio_err, true) {
                        push(base, v, label(r));
                    }
                }
                GateKind::BytesMax | GateKind::BytesMaxLong | GateKind::BytesMin => {
                    let long = gate.kind == GateKind::BytesMaxLong;
                    let jobs: Vec<_> = mine
                        .iter()
                        .copied()
                        .filter(|(r, _)| {
                            let c = &set.conditions[r.cond];
                            !c.cbr && (gate.kind == GateKind::BytesMin || (c.frame_us >= 40_000) == long)
                        })
                        .collect();
                    let largest = gate.kind != GateKind::BytesMin;
                    if let Some((v, r)) = extreme(&jobs, &|p| p.ratio_bytes, largest) {
                        push(base, v, label(r));
                    }
                }
                GateKind::BytesCbr => {
                    let jobs: Vec<_> = mine.iter().copied().filter(|(r, _)| set.conditions[r.cond].cbr).collect();
                    if let Some((v, r)) = extreme(&jobs, &|p| (p.ratio_bytes - 1.0).abs(), true) {
                        push(base, v, label(r));
                    }
                }
                GateKind::MocClip | GateKind::MocMean => {
                    let jobs: Vec<_> = mine.iter().copied().filter(|(_, p)| p.moc_d.is_some()).collect();
                    if gate.kind == GateKind::MocClip {
                        if let Some((v, r)) = extreme(&jobs, &|p| p.moc_d.unwrap_or(0.0), false) {
                            push(base, v, label(r));
                        }
                    } else if !jobs.is_empty() {
                        let v: Vec<f64> = jobs.iter().filter_map(|(_, p)| p.moc_d).collect();
                        push(base, mean(&v), format!("{} jobs", v.len()));
                    }
                }
                GateKind::DeltaSnr => {
                    let v: Vec<f64> = mine.iter().map(|(_, p)| p.delta_snr_db).collect();
                    push(base, mean(&v), format!("{} jobs", v.len()));
                }
                GateKind::BaselineGeomean | GateKind::BaselineMax | GateKind::LibopusChanged | GateKind::BaselineSnr => {}
                GateKind::Align => {
                    let bad: Vec<_> = mine.iter().filter(|(_, p)| p.opus1.best_shift != 0 && p.libopus.best_shift == 0).collect();
                    let detail = bad.first().map_or(String::new(), |(r, p)| format!("{}, shift {}", label(r), p.opus1.best_shift));
                    push(base, bad.len() as f64, detail);
                }
            }
        }
    }
    out
}

/// The 95 % bootstrap confidence interval of a mean (2000 resamples, fixed
/// seed).
pub fn bootstrap_mean_ci(values: &[f64]) -> (f64, f64) {
    if values.len() < 2 {
        let m = mean(values);
        return (m, m);
    }
    let mut rng = XorShift(BOOTSTRAP_SEED);
    let n = values.len();
    let mut means: Vec<f64> = (0..BOOTSTRAP_RESAMPLES)
        .map(|_| (0..n).map(|_| values[rng.below(n as u64) as usize]).sum::<f64>() / n as f64)
        .collect();
    means.sort_by(f64::total_cmp);
    let at = |q: f64| means[((q * (BOOTSTRAP_RESAMPLES - 1) as f64).round() as usize).min(BOOTSTRAP_RESAMPLES - 1)];
    (at(0.025), at(0.975))
}

/// What a stored run gives for one job, by (condition, item).
#[derive(Clone, Copy, Debug)]
pub struct StoredJob {
    pub err_opus1: f64,
    pub err_libopus: f64,
    pub snr_opus1: f64,
}

pub type Stored = std::collections::HashMap<(String, String), StoredJob>;

/// The stored values of the rows of a run.
pub fn stored_of(set: &EvalSet, items: &[Item], rows: &[Row]) -> Stored {
    rows.iter()
        .filter_map(|r| {
            let p = r.result.as_ref().ok()?;
            let key = (set.conditions[r.cond].name.clone(), items[r.item].name.clone());
            Some((key, StoredJob { err_opus1: p.opus1.err, err_libopus: p.libopus.err, snr_opus1: p.opus1.snr_db }))
        })
        .collect()
}

/// Gates against a stored run (`--baseline`), per group, with the limits of
/// the group's `Baseline*` and `LibopusChanged` gates (by default: opus1's
/// `err` now against its stored `err` with a geometric mean of at most 1.02
/// and a largest ratio of at most 1.10, libopus's `err` unchanged, and for
/// groups with a `DeltaSnr` gate opus1's signal-to-noise ratio at most
/// 0.5 dB lower on average).
pub fn baseline_gates(set: &EvalSet, items: &[Item], rows: &[Row], stored: &Stored) -> Vec<GateResult> {
    let mut out = Vec::new();
    for group in &set.groups {
        let mut ratios = Vec::new();
        let mut worst: Option<(f64, String)> = None;
        let mut changed = Vec::new();
        let mut snr = Vec::new();
        for r in rows.iter().filter(|r| set.conditions[r.cond].group == group.name) {
            let Ok(p) = &r.result else { continue };
            let key = (set.conditions[r.cond].name.clone(), items[r.item].name.clone());
            let Some(old) = stored.get(&key) else { continue };
            let (old_opus1, old_libopus) = (old.err_opus1, old.err_libopus);
            snr.push(p.opus1.snr_db - old.snr_opus1);
            let ratio = (p.opus1.err + 1e-6) / (old_opus1 + 1e-6);
            ratios.push(ratio);
            if worst.as_ref().is_none_or(|w| ratio > w.0) {
                worst = Some((ratio, format!("{} {}", key.0, key.1)));
            }
            if p.libopus.err != old_libopus {
                changed.push(format!("{} {}", key.0, key.1));
            }
        }
        if ratios.is_empty() {
            continue;
        }
        for gate in &group.gates {
            let (value, detail) = match gate.kind {
                GateKind::BaselineGeomean => (geomean(&ratios), format!("{} jobs", ratios.len())),
                GateKind::BaselineMax => worst.clone().unwrap_or((f64::NAN, String::new())),
                GateKind::LibopusChanged => (
                    changed.len() as f64,
                    changed.first().map_or(String::new(), |c| format!("libopus changed: {c}")),
                ),
                GateKind::BaselineSnr => (mean(&snr), format!("{} jobs", snr.len())),
                _ => continue,
            };
            let pass = if gate.kind.upper() { value <= gate.limit } else { value >= gate.limit };
            out.push(GateResult {
                name: format!("{} {}", group.name, gate.kind.name()),
                group: group.name.clone(),
                kind: gate.kind,
                limit: gate.limit,
                value,
                pass,
                detail,
            });
        }
    }
    out
}
