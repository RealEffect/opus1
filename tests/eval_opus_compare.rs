//! The port of `opus_compare` (`opus1::internals::eval::compare`) against
//! the C program of libopus 1.6.1.
//!
//! - `eval_opus_compare.expected` holds, for about 40 inputs, what the C
//!   program printed on 64-bit ARM macOS: the exit code, the bits of the
//!   internal weighted error, the bits of the quality metric `Q` when the
//!   test passes (the C program prints `Q` only then), or the error
//!   message. The port must give the same bits on that platform, and agree
//!   to a relative 1e-6 elsewhere, where the math library may round
//!   differently. (Measured on 2026-10-18 on an Android 14 phone, 64-bit
//!   ARM, NDK 27.3: all 55 recorded values came out with the same bits.)
//! - With `OPUS1_OPUS_COMPARE_C` naming the program that
//!   `scripts/eval/check-opus-compare.sh` builds (`opus_compare_exact`), a
//!   second test runs it on the same inputs and compares exactly. When the
//!   plain `opus_compare` lies next to it, the printed lines must also
//!   equal the port's. With `OPUS1_EVAL_RECORD=1` as well, the test
//!   rewrites the `.expected` file.
//!
//! The file is embedded with `include_str!`, so that the test also runs
//! where the source tree is not present (on the phone).

#![cfg(feature = "std")]

mod eval_common;

use eval_common::cases::{CompareCase, compare_cases, le_bytes};
use eval_common::{exact_platform, record, repo_dir, scratch_dir};
use opus1::internals::eval::compare::{Reference, opus_compare};
use proptest::prelude::*;
use std::path::{Path, PathBuf};
use std::process::Command;

const EXPECTED: &str = include_str!("eval_opus_compare.expected");

/// What one run printed, reduced to what the test compares.
#[derive(Clone, Debug, PartialEq)]
enum Outcome {
    /// The comparison ran. `q` is present when the test vector passes.
    Ran { exit: i32, err: f64, q: Option<f32> },
    /// The C program stopped with this message.
    Failed { message: String },
}

impl Outcome {
    fn to_line(&self, name: &str, hash: u64) -> String {
        match self {
            Outcome::Ran { exit, err, q } => format!(
                "{name}\t{hash:016x}\t{exit}\t{:016x}\t{}\t-",
                err.to_bits(),
                q.map_or("-".to_string(), |q| format!("{:08x}", q.to_bits()))
            ),
            Outcome::Failed { message } => format!("{name}\t{hash:016x}\t1\t-\t-\t{message}"),
        }
    }

    fn parse(fields: &[&str]) -> Self {
        if fields[3] == "-" {
            return Outcome::Failed { message: fields[5].to_string() };
        }
        let err = f64::from_bits(u64::from_str_radix(fields[3], 16).expect("err bits"));
        let q = (fields[4] != "-").then(|| f32::from_bits(u32::from_str_radix(fields[4], 16).expect("q bits")));
        Outcome::Ran { exit: fields[2].parse().expect("exit code"), err, q }
    }

    /// Whether `self` (the port) agrees with `want` (the C program): bit for
    /// bit, or within a relative `tolerance`.
    fn agrees(&self, want: &Outcome, tolerance: f64) -> bool {
        let close = |a: f64, b: f64| a.to_bits() == b.to_bits() || (a - b).abs() <= tolerance * b.abs();
        match (self, want) {
            (Outcome::Ran { exit: e1, err: a, q: q1 }, Outcome::Ran { exit: e2, err: b, q: q2 }) => {
                e1 == e2
                    && close(*a, *b)
                    && match (q1, q2) {
                        (Some(x), Some(y)) => close(f64::from(*x), f64::from(*y)),
                        (None, None) => true,
                        _ => false,
                    }
            }
            _ => self == want,
        }
    }
}

/// The port's outcome for one case.
fn port(case: &CompareCase) -> Outcome {
    match opus_compare(Reference::Stereo(&case.reference), &case.test, case.channels, case.rate) {
        Ok(c) => Outcome::Ran {
            exit: if c.passes() { 0 } else { 1 },
            err: c.err,
            q: c.passes().then_some(c.quality),
        },
        Err(e) => Outcome::Failed { message: e.to_string() },
    }
}

/// Write the two input files of a case.
fn write_inputs(case: &CompareCase, dir: &Path) -> (PathBuf, PathBuf) {
    let reference = dir.join(format!("{}.ref.sw", case.name));
    let test = dir.join(format!("{}.test.sw", case.name));
    std::fs::write(&reference, le_bytes(&case.reference)).expect("write the reference");
    std::fs::write(&test, le_bytes(&case.test)).expect("write the test signal");
    (reference, test)
}

/// Run a C `opus_compare` on a case: its exit code and what it printed to
/// stderr.
fn run_c(program: &Path, case: &CompareCase, dir: &Path) -> (i32, String) {
    let (reference, test) = write_inputs(case, dir);
    let mut cmd = Command::new(program);
    if case.channels == 2 {
        cmd.arg("-s");
    }
    cmd.arg("-r").arg(case.rate.to_string()).arg(&reference).arg(&test);
    let out = cmd.output().expect("run the C opus_compare");
    (out.status.code().unwrap_or(-1), String::from_utf8_lossy(&out.stderr).into_owned())
}

/// Read the full-precision output of `opus_compare_exact`.
fn parse_exact(exit: i32, stderr: &str) -> Outcome {
    let number = |s: &str| -> f64 { s.trim().parse().unwrap_or_else(|_| panic!("number in {stderr:?}")) };
    if let Some(line) = stderr.lines().find(|l| l.starts_with("Opus quality metric: ")) {
        // "Opus quality metric: <Q> % (internal weighted error is <err>)"
        let rest = &line["Opus quality metric: ".len()..];
        let (q, rest) = rest.split_once(" % (internal weighted error is ").expect("quality line");
        let err = rest.trim_end_matches(')');
        return Outcome::Ran { exit, err: number(err), q: Some(number(q) as f32) };
    }
    if let Some(line) = stderr.lines().find(|l| l.starts_with("Internal weighted error is ")) {
        return Outcome::Ran { exit, err: number(&line["Internal weighted error is ".len()..]), q: None };
    }
    Outcome::Failed { message: stderr.lines().next().unwrap_or("").to_string() }
}

#[test]
#[cfg_attr(not(target_arch = "aarch64"), ignore = "bit-exact with libopus only on 64-bit ARM (docs/10, Platforms)")]
fn port_matches_the_recorded_c_values() {
    let cases = compare_cases();
    let expected: Vec<Vec<&str>> = EXPECTED
        .lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
        .map(|l| l.split('\t').collect())
        .collect();
    assert_eq!(expected.len(), cases.len(), "the .expected file has another case count; record it again");
    let tolerance = if exact_platform() { 0.0 } else { 1e-6 };
    let mut failures = Vec::new();
    let (mut passing, mut failing, mut errors) = (0, 0, 0);
    // How far the port is from the recorded values, for platforms other
    // than the one they were recorded on.
    let (mut same_bits, mut largest) = (0, 0.0f64);
    for (case, fields) in cases.iter().zip(&expected) {
        assert_eq!(fields.len(), 6, "line of {}", case.name);
        assert_eq!(fields[0], case.name, "case order");
        assert_eq!(fields[1], format!("{:016x}", case.hash()), "the input of {} changed", case.name);
        let want = Outcome::parse(fields);
        let got = port(case);
        match &want {
            Outcome::Ran { q: Some(_), .. } => passing += 1,
            Outcome::Ran { q: None, .. } => failing += 1,
            Outcome::Failed { .. } => errors += 1,
        }
        if let (Outcome::Ran { err: a, q: qa, .. }, Outcome::Ran { err: b, q: qb, .. }) = (&got, &want) {
            let mut pairs = vec![(*a, *b)];
            if let (Some(x), Some(y)) = (qa, qb) {
                pairs.push((f64::from(*x), f64::from(*y)));
            }
            for (x, y) in pairs {
                same_bits += (x.to_bits() == y.to_bits()) as usize;
                if y != 0.0 {
                    largest = largest.max((x - y).abs() / y.abs());
                }
            }
        }
        if !got.agrees(&want, tolerance) {
            failures.push(format!("{}: port {got:?}, C {want:?}", case.name));
        }
    }
    println!("{} cases: {passing} pass, {failing} fail, {errors} errors; tolerance {tolerance}", cases.len());
    println!("{same_bits} values with the recorded bits; largest relative difference {largest:e}");
    assert!(failures.is_empty(), "differences:\n{}", failures.join("\n"));
    assert!(passing >= 10 && failing >= 5 && errors >= 4, "the cases should cover every outcome");
}

#[test]
fn port_matches_the_c_program() {
    let Some(program) = std::env::var_os("OPUS1_OPUS_COMPARE_C").map(PathBuf::from) else {
        println!("skipped: set OPUS1_OPUS_COMPARE_C (built by scripts/eval/check-opus-compare.sh)");
        return;
    };
    let plain = program.with_file_name("opus_compare");
    let dir = scratch_dir("opus-compare");
    let cases = compare_cases();
    let mut lines = vec![
        "# Outputs of libopus 1.6.1 src/opus_compare.c (scripts/eval/opus_compare_exact.c) for".to_string(),
        "# tests/eval_common/cases.rs::compare_cases. Fields: case, FNV-1a of the inputs, exit code,".to_string(),
        "# bits of the internal weighted error, bits of Q (passing cases only), error message.".to_string(),
        format!("# Recorded on {}-{} with OPUS1_EVAL_RECORD=1.", std::env::consts::ARCH, std::env::consts::OS),
    ];
    let mut failures = Vec::new();
    for case in &cases {
        let (exit, stderr) = run_c(&program, case, &dir);
        let want = parse_exact(exit, &stderr);
        let got = port(case);
        if got != want {
            failures.push(format!("{}: port {got:?}, C {want:?}", case.name));
        }
        if plain.exists() {
            // The lines of the plain program, with C's rounding.
            let (exit, text) = run_c(&plain, case, &dir);
            let ours = match opus_compare(Reference::Stereo(&case.reference), &case.test, case.channels, case.rate) {
                Ok(c) => (if c.passes() { 0 } else { 1 }, c.to_string()),
                Err(e) => (1, format!("{e}\n")),
            };
            if (exit, text.clone()) != ours {
                failures.push(format!("{}: printed {ours:?}, C printed {:?}", case.name, (exit, text)));
            }
        }
        lines.push(want.to_line(case.name, case.hash()));
    }
    let _ = std::fs::remove_dir_all(&dir);
    println!("{} cases compared with {}", cases.len(), program.display());
    if record() && failures.is_empty() {
        let path = repo_dir().join("tests/eval_opus_compare.expected");
        std::fs::write(&path, lines.join("\n") + "\n").expect("write the .expected file");
        println!("wrote {}", path.display());
    }
    assert!(failures.is_empty(), "differences:\n{}", failures.join("\n"));
}

/// A short random stereo signal and a test signal at `rate` for the
/// property tests.
fn random_pair(seed: u64, frames48: usize, rate: u32, channels: usize, amp: i32) -> (Vec<i16>, Vec<i16>) {
    let mut rng = eval_common::cases::XorShift(seed | 1);
    let reference: Vec<i16> = (0..frames48 * 2).map(|_| rng.uniform(amp) as i16).collect();
    let n = frames48 / (48000 / rate as usize) * channels;
    let test: Vec<i16> = (0..n).map(|_| rng.uniform(amp) as i16).collect();
    (reference, test)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(48))]

    /// Random inputs: no panic, a non-negative error.
    #[test]
    fn random_inputs_give_a_valid_error(seed in any::<u64>(), frames in 480usize..2400, rate_index in 0usize..5,
                                        channels in 1usize..3, amp in 0i32..32768) {
        let rate = [48000u32, 24000, 16000, 12000, 8000][rate_index];
        let frames = frames / 48 * 48;
        let (reference, test) = random_pair(seed, frames, rate, channels, amp);
        let c = opus_compare(Reference::Stereo(&reference), &test, channels, rate).unwrap();
        prop_assert!(c.err >= 0.0 && c.err.is_finite());
    }

    /// The reference against itself gives 0, and twice the reference does
    /// not.
    #[test]
    fn identity_and_scaling(seed in any::<u64>(), frames in 480usize..2000, amp in 100i32..16000) {
        let (reference, _) = random_pair(seed, frames, 48000, 2, amp);
        let same = opus_compare(Reference::Stereo(&reference), &reference, 2, 48000).unwrap();
        prop_assert_eq!(same.err, 0.0);
        prop_assert_eq!(same.quality, 100.0);
        let double: Vec<i16> = reference.iter().map(|&v| v.saturating_mul(2)).collect();
        let scaled = opus_compare(Reference::Stereo(&reference), &double, 2, 48000).unwrap();
        prop_assert!(scaled.err > 0.0);
    }

    /// A mono reference equals a stereo reference with the same samples on
    /// both sides, bit for bit.
    #[test]
    fn mono_reference_equals_duplicated_stereo(seed in any::<u64>(), frames in 480usize..1500, channels in 1usize..3) {
        let (reference, test) = random_pair(seed, frames, 48000, channels, 8000);
        let mono: Vec<i16> = reference.iter().step_by(2).copied().collect();
        let doubled: Vec<i16> = mono.iter().flat_map(|&v| [v, v]).collect();
        let a = opus_compare(Reference::Mono(&mono), &test, channels, 48000).unwrap();
        let b = opus_compare(Reference::Stereo(&doubled), &test, channels, 48000).unwrap();
        prop_assert_eq!(a.err.to_bits(), b.err.to_bits());
        prop_assert_eq!(a.quality.to_bits(), b.quality.to_bits());
    }
}
