//! The MOC port (`opus1::internals::eval::moc`) against the numpy
//! reference `scripts/eval/moc_reference.py`, a transcription of `moc.py`
//! of the libopus git repository.
//!
//! - `eval_moc.expected` holds the reference's values for about 20 inputs,
//!   each with and without the activity detection, with 17 significant
//!   digits. The port must agree to `1e-9·|value| + 1e-12`: numpy uses an
//!   FFT and its own summation order, so the last bits differ. Identical
//!   inputs must give exactly 0. Where `moc.py` fails (an exception or NaN),
//!   the port must return an error.
//! - With `OPUS1_EVAL_PYTHON` naming a Python 3 with numpy, a second test
//!   runs the script on the same inputs. With `OPUS1_MOC_PY` naming the
//!   original `moc.py` as well, the script also checks its transcription
//!   against it. With `OPUS1_EVAL_RECORD=1`, the test rewrites the
//!   `.expected` file.

#![cfg(feature = "std")]

mod eval_common;

use eval_common::cases::{MocCase, le_bytes, moc_cases};
use eval_common::{record, repo_dir, scratch_dir};
use opus1::internals::eval::moc::{MocError, moc_i16, moc_relative};
use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Command;

const EXPECTED: &str = include_str!("eval_moc.expected");

/// The tolerance of the comparison with numpy.
fn close(got: f64, want: f64) -> bool {
    (got - want).abs() <= 1e-9 * want.abs() + 1e-12
}

/// The reference's value of one case: a number, or a failure of `moc.py`.
#[derive(Debug, PartialEq)]
enum Want {
    Value(f64),
    Fails(String),
}

fn parse_value(s: &str) -> Want {
    match s.parse::<f64>() {
        Ok(v) if !v.is_nan() => Want::Value(v),
        _ => Want::Fails(s.to_string()),
    }
}

/// Check the port against the reference values `want[(name, vad)]`.
fn check(cases: &[MocCase], want: &HashMap<(String, bool), Want>) -> Vec<String> {
    let mut failures = Vec::new();
    let mut worst = 0.0f64;
    for case in cases {
        for vad in [false, true] {
            let got = moc_i16(&case.reference, &case.test, vad);
            let Some(w) = want.get(&(case.name.to_string(), vad)) else {
                failures.push(format!("{} vad={vad}: no reference value", case.name));
                continue;
            };
            match (&got, w) {
                (Ok(g), Want::Value(v)) => {
                    if *v != 0.0 {
                        worst = worst.max((g - v).abs() / v.abs());
                    }
                    if !close(*g, *v) || (*v == 0.0 && *g != 0.0) {
                        failures.push(format!("{} vad={vad}: port {g:?}, numpy {v:?}", case.name));
                    }
                }
                (Err(_), Want::Fails(_)) => {}
                _ => failures.push(format!("{} vad={vad}: port {got:?}, numpy {w:?}", case.name)),
            }
        }
    }
    println!("largest relative difference from numpy: {worst:e}");
    failures
}

#[test]
fn port_matches_the_recorded_numpy_values() {
    let cases = moc_cases();
    let mut want = HashMap::new();
    let hashes: HashMap<&str, u64> = cases.iter().map(|c| (c.name, c.hash())).collect();
    for line in EXPECTED.lines().filter(|l| !l.starts_with('#') && !l.trim().is_empty()) {
        let f: Vec<&str> = line.split('\t').collect();
        assert_eq!(f.len(), 4, "line {line:?}");
        let hash = hashes.get(f[0]).unwrap_or_else(|| panic!("unknown case {}", f[0]));
        assert_eq!(f[2], format!("{hash:016x}"), "the input of {} changed; record again", f[0]);
        want.insert((f[0].to_string(), f[1] == "1"), parse_value(f[3]));
    }
    assert_eq!(want.len(), 2 * cases.len(), "the .expected file has another case count; record it again");
    let failures = check(&cases, &want);
    assert!(failures.is_empty(), "differences:\n{}", failures.join("\n"));
}

#[test]
fn port_matches_the_numpy_script() {
    let Some(python) = std::env::var_os("OPUS1_EVAL_PYTHON").map(PathBuf::from) else {
        println!("skipped: set OPUS1_EVAL_PYTHON to a python3 with numpy (and OPUS1_MOC_PY to moc.py)");
        return;
    };
    let dir = scratch_dir("moc");
    let cases = moc_cases();
    let mut list = String::new();
    for case in &cases {
        std::fs::write(dir.join(format!("{}.ref.s16", case.name)), le_bytes(&case.reference)).unwrap();
        std::fs::write(dir.join(format!("{}.test.s16", case.name)), le_bytes(&case.test)).unwrap();
        for vad in [0, 1] {
            list += &format!("{0} {vad} {0}.ref.s16 {0}.test.s16\n", case.name);
        }
    }
    std::fs::write(dir.join("cases.txt"), list).unwrap();
    let mut cmd = Command::new(&python);
    cmd.arg("-I").arg(repo_dir().join("scripts/eval/moc_reference.py")).arg("--cases").arg(&dir);
    if let Some(moc_py) = std::env::var_os("OPUS1_MOC_PY") {
        cmd.arg("--check-against").arg(moc_py);
    }
    let out = cmd.output().expect("run moc_reference.py");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    println!("{}", stderr.trim());
    assert!(out.status.success(), "moc_reference.py failed:\n{stderr}");
    let _ = std::fs::remove_dir_all(&dir);

    let mut want = HashMap::new();
    let mut lines = vec![
        "# MOC values of scripts/eval/moc_reference.py (numpy) for tests/eval_common/cases.rs::moc_cases.".to_string(),
        "# Fields: case, activity detection (0 or 1), FNV-1a of the inputs, value (17 digits) or the".to_string(),
        "# failure of moc.py. Recorded with OPUS1_EVAL_RECORD=1.".to_string(),
    ];
    let hashes: HashMap<&str, u64> = cases.iter().map(|c| (c.name, c.hash())).collect();
    for line in stdout.lines() {
        let f: Vec<&str> = line.split_whitespace().collect();
        assert_eq!(f.len(), 3, "output line {line:?}");
        want.insert((f[0].to_string(), f[1] == "1"), parse_value(f[2]));
        lines.push(format!("{}\t{}\t{:016x}\t{}", f[0], f[1], hashes[f[0]], f[2]));
    }
    let failures = check(&cases, &want);
    if record() && failures.is_empty() {
        let path = repo_dir().join("tests/eval_moc.expected");
        std::fs::write(&path, lines.join("\n") + "\n").expect("write the .expected file");
        println!("wrote {}", path.display());
    }
    assert!(failures.is_empty(), "differences:\n{}", failures.join("\n"));
}

#[test]
fn errors_and_the_relative_distance() {
    let x = vec![1000i16; 279];
    assert_eq!(moc_i16(&x, &x, false), Err(MocError::TooShort { samples: 279 }));
    let x = vec![1000i16; 1599];
    assert_eq!(moc_i16(&x, &x, true), Err(MocError::VadTooShort { samples: 1599 }));
    assert_eq!(moc_i16(&x, &x, false), Ok(0.0));
    // D of the OSCE draft: positive when the tested decoder is closer.
    assert!(moc_relative(0.4, 0.3) > 0.0);
    assert!(moc_relative(0.4, 0.5) < 0.0);
    assert_eq!(moc_relative(0.4, 0.4), 0.0);
    assert!((moc_relative(0.9, 0.4) - 0.5).abs() < 1e-15);
}
