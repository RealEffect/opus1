//! Shared parts of the `eval_*` tests: the test signals (`cases`) and
//! libopus behind the `opus_demo` decoding loop (`libopus_demo`).

#![allow(dead_code)]

pub mod cases;
pub mod libopus_demo;

use std::path::PathBuf;

/// The directory of the repository (`CARGO_MANIFEST_DIR`).
pub fn repo_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// The corpus root: `OPUS1_CORPUS`, else `<repo>/corpus`.
pub fn corpus_dir() -> PathBuf {
    match std::env::var_os("OPUS1_CORPUS") {
        Some(dir) if !dir.is_empty() => PathBuf::from(dir),
        _ => repo_dir().join("corpus"),
    }
}

/// A new empty directory for the files of one test run, under the system's
/// temporary directory.
pub fn scratch_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("opus1-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create the scratch directory");
    dir
}

/// Whether `OPUS1_EVAL_RECORD=1` asks the tests to rewrite their
/// `.expected` files from the C or Python reference.
pub fn record() -> bool {
    std::env::var("OPUS1_EVAL_RECORD").is_ok_and(|v| v == "1")
}

/// Whether recorded values must match bit for bit here: on the platform
/// they were recorded on (64-bit ARM macOS). Elsewhere the math library may
/// differ in the last bit.
pub fn exact_platform() -> bool {
    cfg!(all(target_arch = "aarch64", target_os = "macos"))
}
