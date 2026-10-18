//! Degradations of the opus1 side, and the self-test that checks that each
//! of them makes at least one gate fail.
//!
//! A degradation changes the opus1 side only, from outside the codec:
//!
//! - `lowpass[:HZ]`: a 4th-order Butterworth low-pass on the decoded
//!   signal, run forward and backward so that it adds no delay. Without a
//!   frequency: 7 kHz at 24 kHz and above, 3.5 kHz at 12 and 16 kHz, 2 kHz
//!   at 8 kHz.
//! - `noise:DB`: white noise DB below the RMS level of the decoded signal.
//! - `shift:MS`: the decoded signal is delayed by MS milliseconds.
//! - `underspend:F` and `overspend:F`: opus1 codes at F times the
//!   condition's bitrate.
//! - `alloc:mask`: an energy mask that moves bits from the low bands to the
//!   high bands (-2.0 on bands 0 to 10, +0.25 on bands 11 to 20). It applies
//!   to CELT CBR conditions only: the stream stays valid and the size stays
//!   the same; only the band allocation changes.

use crate::corpus::Item;
use crate::pipeline::{self, Codec, Condition, Inputs, Pair, Row};
use crate::sets::EvalSet;
use crate::signals::XorShift;
use crate::stats::{self, GateResult};

#[derive(Clone, Debug, PartialEq)]
pub enum Degrade {
    Lowpass(Option<f64>),
    Noise(f64),
    Shift(f64),
    Underspend(f64),
    Overspend(f64),
    AllocMask,
}

impl Degrade {
    pub fn parse(text: &str) -> Result<Self, String> {
        let (kind, arg) = text.split_once(':').map_or((text, None), |(k, a)| (k, Some(a)));
        let number = |default: Option<f64>| -> Result<f64, String> {
            match arg {
                Some(a) => a.trim_end_matches("ms").parse::<f64>().map_err(|_| format!("bad value in {text}")),
                None => default.ok_or_else(|| format!("{text} needs a value")),
            }
        };
        Ok(match kind {
            "lowpass" => Degrade::Lowpass(arg.map(|_| number(None)).transpose()?),
            "noise" => Degrade::Noise(number(None)?),
            "shift" => Degrade::Shift(number(Some(1.0))?),
            "underspend" => Degrade::Underspend(number(Some(0.75))?),
            "overspend" => Degrade::Overspend(number(Some(1.10))?),
            "alloc" if arg == Some("mask") => Degrade::AllocMask,
            _ => return Err(format!("unknown degradation {text}")),
        })
    }

    pub fn name(&self) -> String {
        match self {
            Degrade::Lowpass(None) => "lowpass".into(),
            Degrade::Lowpass(Some(f)) => format!("lowpass:{f}"),
            Degrade::Noise(db) => format!("noise:{db}"),
            Degrade::Shift(ms) => format!("shift:{ms}ms"),
            Degrade::Underspend(f) => format!("underspend:{f}"),
            Degrade::Overspend(f) => format!("overspend:{f}"),
            Degrade::AllocMask => "alloc:mask".into(),
        }
    }

    /// The degradations of the self-test.
    pub fn all() -> Vec<Degrade> {
        vec![
            Degrade::Lowpass(None),
            Degrade::Noise(30.0),
            Degrade::Noise(20.0),
            Degrade::Shift(1.0),
            Degrade::Underspend(0.75),
            Degrade::Overspend(1.10),
            Degrade::AllocMask,
        ]
    }

    pub fn applies_to(&self, cond: &Condition) -> bool {
        match self {
            Degrade::AllocMask => cond.cbr && cond.group.starts_with("celt"),
            _ => true,
        }
    }

    /// Whether the degradation acts on the decoded signal (so the packets
    /// need not be coded again).
    pub fn is_post(&self) -> bool {
        matches!(self, Degrade::Lowpass(_) | Degrade::Noise(_) | Degrade::Shift(_))
    }

    pub fn bitrate_scale(&self) -> f64 {
        match self {
            Degrade::Underspend(f) | Degrade::Overspend(f) => *f,
            _ => 1.0,
        }
    }

    /// The energy mask for opus1's encoder: 21 values per channel.
    pub fn energy_mask(&self, cond: &Condition) -> Option<Vec<f32>> {
        (*self == Degrade::AllocMask && self.applies_to(cond)).then(|| {
            let one: Vec<f32> = (0..21).map(|b| if b <= 10 { -2.0 } else { 0.25 }).collect();
            one.repeat(cond.channels)
        })
    }

    /// Apply a degradation that acts on the decoded signal. Others do
    /// nothing here.
    pub fn apply_post(&self, pcm: &mut [i16], rate: u32, channels: usize) {
        match *self {
            Degrade::Lowpass(f) => {
                let f = f.unwrap_or(match rate {
                    r if r >= 24_000 => 7000.0,
                    r if r >= 12_000 => 3500.0,
                    _ => 2000.0,
                });
                for c in 0..channels {
                    let mut x: Vec<f64> = pcm.iter().skip(c).step_by(channels).map(|&s| s as f64).collect();
                    lowpass_zero_phase(&mut x, f, rate as f64);
                    for (dst, v) in pcm.iter_mut().skip(c).step_by(channels).zip(&x) {
                        *dst = v.round().clamp(-32768.0, 32767.0) as i16;
                    }
                }
            }
            Degrade::Noise(db) => {
                let power = pcm.iter().map(|&s| (s as f64).powi(2)).sum::<f64>() / pcm.len().max(1) as f64;
                // A uniform value in [-1, 1) has power 1/3.
                let gain = (power * 3.0).sqrt() * 10f64.powf(-db / 20.0);
                let mut rng = XorShift(0x4015_E000_0000_0001);
                for s in pcm.iter_mut() {
                    let v = *s as f64 + gain * rng.uniform();
                    *s = v.round().clamp(-32768.0, 32767.0) as i16;
                }
            }
            Degrade::Shift(ms) => {
                let shift = ((ms * rate as f64 / 1000.0).round() as usize * channels).min(pcm.len());
                pcm.copy_within(..pcm.len() - shift, shift);
                pcm[..shift].fill(0);
            }
            _ => {}
        }
    }
}

/// One biquad section (direct form II transposed).
struct Biquad {
    b: [f64; 3],
    a: [f64; 2],
}

impl Biquad {
    /// Low-pass of the RBJ audio EQ cookbook.
    fn lowpass(f: f64, q: f64, fs: f64) -> Self {
        let w = std::f64::consts::TAU * f / fs;
        let alpha = w.sin() / (2.0 * q);
        let cos = w.cos();
        let a0 = 1.0 + alpha;
        Self {
            b: [(1.0 - cos) / 2.0 / a0, (1.0 - cos) / a0, (1.0 - cos) / 2.0 / a0],
            a: [-2.0 * cos / a0, (1.0 - alpha) / a0],
        }
    }

    fn run(&self, x: &mut [f64]) {
        let (mut z1, mut z2) = (0.0, 0.0);
        for v in x.iter_mut() {
            let input = *v;
            let y = self.b[0] * input + z1;
            z1 = self.b[1] * input - self.a[0] * y + z2;
            z2 = self.b[2] * input - self.a[1] * y;
            *v = y;
        }
    }
}

/// 4th-order Butterworth low-pass (two biquads with Q 0.5412 and 1.3066),
/// forward and then backward.
fn lowpass_zero_phase(x: &mut [f64], f: f64, fs: f64) {
    let sections = [Biquad::lowpass(f, 0.541_196_1, fs), Biquad::lowpass(f, 1.306_563, fs)];
    for s in &sections {
        s.run(x);
    }
    x.reverse();
    for s in &sections {
        s.run(x);
    }
    x.reverse();
}

/// What the self-test found for one degradation.
pub struct Check {
    pub name: String,
    /// Gates that pass on the undegraded run and fail with the degradation.
    pub tripped: Vec<GateResult>,
    /// The geometric mean of `ratio_err` over the jobs the degradation
    /// applies to, before and after.
    pub geomean_before: f64,
    pub geomean_after: f64,
    pub jobs: usize,
    pub errors: Vec<String>,
}

impl Check {
    pub fn passed(&self) -> bool {
        !self.tripped.is_empty() && self.errors.is_empty()
    }
}

/// The result of the self-test.
pub struct SelfTest {
    pub base_gates: Vec<GateResult>,
    pub base_errors: Vec<String>,
    /// Identity checks: libopus against libopus, opus1 against opus1. Each
    /// lists what was wrong; empty means it passed.
    pub identity: Vec<(String, Vec<String>)>,
    pub checks: Vec<Check>,
}

impl SelfTest {
    pub fn passed(&self) -> bool {
        self.base_errors.is_empty() && self.identity.iter().all(|(_, e)| e.is_empty()) && self.checks.iter().all(Check::passed)
    }

    pub fn report(&self) -> String {
        let mut out = String::new();
        out += &format!("undegraded run: {} job error(s)\n", self.base_errors.len());
        for e in &self.base_errors {
            out += &format!("  ERROR {e}\n");
        }
        for g in self.base_gates.iter().filter(|g| !g.pass) {
            out += &format!("  WARN gate fails without degradation: {}\n", g.line());
        }
        for (name, problems) in &self.identity {
            out += &format!("identity {name}: {}\n", if problems.is_empty() { "pass" } else { "FAIL" });
            for p in problems.iter().take(10) {
                out += &format!("  {p}\n");
            }
        }
        for c in &self.checks {
            out += &format!(
                "{:<18} {} ({} jobs, geometric mean of ratio_err {:.4} -> {:.4}, {} gate(s) tripped)\n",
                c.name,
                if c.passed() { "pass" } else { "FAIL" },
                c.jobs,
                c.geomean_before,
                c.geomean_after,
                c.tripped.len()
            );
            for g in c.tripped.iter().take(6) {
                out += &format!("  tripped {}\n", g.line());
            }
            for e in c.errors.iter().take(5) {
                out += &format!("  ERROR {e}\n");
            }
        }
        out += if self.passed() { "self-test: pass\n" } else { "self-test: FAIL\n" };
        out
    }
}

/// Run the self-test on a set: the undegraded run, the two identity checks
/// and every degradation of [`Degrade::all`], then evaluate the gates.
pub fn selftest(set: &EvalSet, items: &[Item], threads: usize) -> SelfTest {
    run_selftest(set, items, threads).evaluate(set, items)
}

/// The coded and measured jobs of a self-test, before any gate is
/// evaluated. [`SelfTestRuns::evaluate`] can then apply the gates of a set
/// with other limits without coding again.
pub struct SelfTestRuns {
    /// The undegraded run.
    pub base: Vec<Row>,
    /// The identity checks: a name and the rows of a fresh encoding by the
    /// same codec against the stored side.
    pub identity: Vec<(String, Vec<Row>)>,
    /// The rows of each degradation.
    pub degraded: Vec<(Degrade, Vec<Row>)>,
}

/// Code and measure every job of the self-test. The rows do not keep the
/// decoded signals.
pub fn run_selftest(set: &EvalSet, items: &[Item], threads: usize) -> SelfTestRuns {
    let jobs = set.jobs(items);
    let inputs = Inputs::new(jobs.iter().map(|&(c, i)| (i, set.conditions[c].rate, set.conditions[c].channels)));
    let input = |c: usize, i: usize| inputs.get(items, i, set.conditions[c].rate, set.conditions[c].channels);
    let side = |codec: Codec, c: usize, i: usize, d: Option<&Degrade>| {
        pipeline::run_side(codec, Codec::Libopus, &set.conditions[c], &items[i], input(c, i), d, true)
    };
    let to_rows = |pairs: Vec<Result<Pair, String>>| -> Vec<Row> {
        pairs
            .into_iter()
            .zip(&jobs)
            .map(|(result, &(cond, item))| {
                let result = result.map(|mut p| {
                    p.opus1.pcm = None;
                    p.libopus.pcm = None;
                    p
                });
                Row { cond, item, result }
            })
            .collect()
    };

    // Both sides of every job, once, with their decoded signals.
    let base: Vec<(Result<pipeline::Side, String>, Result<pipeline::Side, String>)> =
        pipeline::parallel_map(jobs.len(), threads, "", |j| {
            let (c, i) = jobs[j];
            (side(Codec::Opus1, c, i, None), side(Codec::Libopus, c, i, None))
        });
    let base_rows = to_rows(
        base.iter()
            .map(|(a, b)| match (a, b) {
                (Ok(a), Ok(b)) => pipeline::pair(a.clone(), b.clone()),
                (Err(e), _) | (_, Err(e)) => Err(e.clone()),
            })
            .collect(),
    );

    // Identity checks: a fresh encoding by the same codec in the tested
    // slot.
    let mut identity = Vec::new();
    for (codec, name) in [(Codec::Libopus, "libopus against libopus"), (Codec::Opus1, "opus1 against opus1")] {
        let rows = to_rows(pipeline::parallel_map(jobs.len(), threads, "", |j| {
            let (c, i) = jobs[j];
            let reference = if codec == Codec::Libopus { &base[j].1 } else { &base[j].0 };
            let fresh = side(codec, c, i, None)?;
            pipeline::pair(fresh, reference.clone()?)
        }));
        identity.push((name.to_string(), rows));
    }

    // Each degradation on the opus1 side.
    let mut degraded = Vec::new();
    for d in Degrade::all() {
        let rows = to_rows(pipeline::parallel_map(jobs.len(), threads, "", |j| {
            let (c, i) = jobs[j];
            let cond = &set.conditions[c];
            let (a, b) = &base[j];
            let b = b.clone()?;
            if !d.applies_to(cond) {
                return pipeline::pair(a.clone()?, b);
            }
            let a = if d.is_post() {
                pipeline::degrade_side(a.as_ref().map_err(Clone::clone)?, &d, cond, &items[i], input(c, i))?
            } else {
                side(Codec::Opus1, c, i, Some(&d))?
            };
            pipeline::pair(a, b)
        }));
        degraded.push((d, rows));
    }
    SelfTestRuns { base: base_rows, identity, degraded }
}

impl SelfTestRuns {
    /// Evaluate the gates of `set` (the set the runs were made with, or the
    /// same set with other limits).
    ///
    /// - The undegraded run must have no job error.
    /// - An identity check must be exactly neutral (`ratio_err` and
    ///   `ratio_bytes` 1, `delta_snr_db` and MOC D 0) and pass every gate.
    /// - A degradation must make at least one gate fail that passes on the
    ///   undegraded run, in a group it applies to. The gates include those
    ///   of `--baseline`, with the undegraded run as the stored one.
    pub fn evaluate(&self, set: &EvalSet, items: &[Item]) -> SelfTest {
        let errors_of = |rows: &[Row]| -> Vec<String> {
            rows.iter()
                .filter_map(|r| {
                    r.result.as_ref().err().map(|e| format!("{} {}: {e}", set.conditions[r.cond].name, items[r.item].name))
                })
                .collect()
        };
        let base_errors = errors_of(&self.base);
        let stored = stats::stored_of(set, items, &self.base);
        let gates_of = |rows: &[Row]| -> Vec<GateResult> {
            let mut g = stats::evaluate_gates(set, items, rows);
            g.extend(stats::baseline_gates(set, items, rows, &stored));
            g
        };
        let base_gates = gates_of(&self.base);

        let mut identity = Vec::new();
        for (name, rows) in &self.identity {
            let mut problems = errors_of(rows);
            for r in rows {
                let Ok(p) = &r.result else { continue };
                let label = format!("{} {}", set.conditions[r.cond].name, items[r.item].name);
                if p.ratio_err != 1.0 || p.ratio_bytes != 1.0 || p.delta_snr_db != 0.0 {
                    problems.push(format!(
                        "{label}: ratio_err {}, ratio_bytes {}, delta_snr_db {}",
                        p.ratio_err, p.ratio_bytes, p.delta_snr_db
                    ));
                }
                if p.moc_d.is_some_and(|d| d != 0.0) {
                    problems.push(format!("{label}: moc_d {:?}", p.moc_d));
                }
            }
            for g in stats::evaluate_gates(set, items, rows).iter().filter(|g| !g.pass) {
                problems.push(format!("gate fails: {}", g.line()));
            }
            identity.push((name.clone(), problems));
        }

        let mut checks = Vec::new();
        for (d, rows) in &self.degraded {
            let applied: Vec<usize> = (0..rows.len()).filter(|&j| d.applies_to(&set.conditions[rows[j].cond])).collect();
            let ratios = |rows: &[Row]| -> Vec<f64> {
                applied.iter().filter_map(|&j| rows[j].result.as_ref().ok().map(|p| p.ratio_err)).collect()
            };
            let groups: Vec<&str> = applied.iter().map(|&j| set.conditions[rows[j].cond].group.as_str()).collect();
            let tripped: Vec<GateResult> = gates_of(rows)
                .into_iter()
                .filter(|g| !g.pass && groups.contains(&g.group.as_str()))
                .filter(|g| base_gates.iter().any(|b| b.name == g.name && b.pass))
                .collect();
            checks.push(Check {
                name: d.name(),
                tripped,
                geomean_before: stats::geomean(&ratios(&self.base)),
                geomean_after: stats::geomean(&ratios(rows)),
                jobs: applied.len(),
                errors: errors_of(rows),
            });
        }
        SelfTest { base_gates, base_errors, identity, checks }
    }
}
