//! Paired comparison of opus1 with libopus 1.6.1.
//!
//! Each item is coded by both encoders with the same settings; libopus
//! decodes both streams; `opus_compare` (and MOC for speech) scores each
//! decoded signal against the clean reference; the tool reports the ratio
//! of the two errors per item, geometric means per condition with bootstrap
//! confidence intervals, the worst items, and the bytes each encoder spent.
//! Design: section 4.4 of the evaluation design note (A0).
//!
//! ```text
//! cargo run --release --example eval -- list [--set NAME]
//! cargo run --release --example eval -- run [--set NAME] [--items FILE] [--corpus DIR]
//!     [--filter S] [--item-filter S] [--max-items N] [--json PATH] [--md PATH]
//!     [--text PATH] [--baseline JSON] [--degrade KIND] [--write-wav DIR]
//!     [--decoder libopus|opus1] [--threads N] [--strict]
//! cargo run --release --example eval -- selftest [--set NAME] [--threads N]
//! cargo run --release --example eval -- conformance [--vectors DIR] [--threads N]
//! cargo run --release --example eval -- moc REF.wav TEST.wav [--apply-vad]
//! cargo run --release --example eval -- corpus-index --root DIR --out FILE
//!     [--per-group N] [--class CLASS] [--corpus DIR]
//! cargo run --release --example eval -- merge RUN.json TIER2.jsonl [--json PATH]
//! ```
//!
//! The sets are `quick` (default; no download needed), `selftest`, `tsp`,
//! `tsp_rates`, `music` and `baseline` (see `sets.rs`). The corpus root is `--corpus`,
//! else `OPUS1_CORPUS`, else `<repository>/corpus`.
//!
//! Gates only warn: `run` exits with 1 when a job fails (for example on a
//! final range mismatch), and with `--strict` also when a gate fails.

#[cfg(feature = "std")]
mod conformance;
#[cfg(feature = "std")]
mod corpus;
#[cfg(feature = "std")]
mod degrade;
#[cfg(feature = "std")]
mod libopus;
#[cfg(feature = "std")]
mod pipeline;
#[cfg(feature = "std")]
mod report;
#[cfg(feature = "std")]
mod sets;
#[cfg(feature = "std")]
mod signals;
#[cfg(feature = "std")]
mod stats;
#[cfg(feature = "std")]
mod tier2;
#[cfg(feature = "std")]
mod cli;

#[cfg(feature = "std")]
fn main() {
    cli::main();
}

#[cfg(not(feature = "std"))]
fn main() {
    println!("this example needs the `std` feature");
}
