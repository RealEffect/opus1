# opus1 documentation and project index

These documents explain the Opus audio codec and `opus1`. Opus is defined in RFC 6716 (Request for Comments 6716, the Internet standard for Opus). `opus1` is the pure-Rust implementation of Opus in this repository, and it follows libopus 1.6. The documents assume no knowledge of audio coding beyond "sample rate" and "bitrate". Each one defines its terms when it first uses them, and the [glossary](guide/glossary.md) collects them in one place. The repository's [README](../README.md) has the short overview, the release notes and the latest timings.

This page is also the index of the whole repository, for people who are new to the code and for AI coding tools (Claude, GPT and others). It lists the documents and a reading order, says where the code, tests, scripts and tools are, and gives the commands that build and test each part. [The architecture of opus1](framework.md) describes how the code is built.

File counts and test counts on this page were taken at commit `9891020` (2026-10-27).

## The project in brief

- **What it is.** `opus1` is a port of libopus 1.6.1, the reference implementation of Opus in C. A "port" here means Rust code that computes what the C file computes, function by function.
- **Crate facts.** Name `opus1`, version `0.3.0` (2026-10-29; the first release was 0.2.1, 2026-10-27), Rust edition 2024, minimum Rust version 1.98, licence BSD-3-Clause ([COPYING](../COPYING)). Repository: <https://github.com/RealEffect/opus1>.
- **History.** It is a hard fork of [opus-rs](https://github.com/restsend/opus-rs), taken at version 0.1.37. The two projects no longer share changes. Code that used `opus_rs::` now uses `opus1::`.
- **Not ported.** The projection encoder and decoder (mapping family 3), the floating-point SILK encoder, the scalable quality extension (QEXT) and Opus Custom. [Differences from libopus](guide/10-differences-from-libopus.md) lists every gap.

## The documents

The numbered documents and the glossary are in the folder [`guide/`](guide).

| Document | What it covers |
|---|---|
| [01 What Opus is](guide/01-what-is-opus.md) | The two coding layers (SILK, a speech coder, and CELT, the Constrained Energy Lapped Transform coder), the three modes, bandwidths, sampling rates, frame durations, delay, packet loss tools, Ogg Opus, and how `opus1` relates to libopus. |
| [02 Opus packets](guide/02-packets.md) | The bytes of a packet: the table-of-contents (TOC) byte, the 32 configurations, the four frame count codes, limits, self-delimited framing, the `opus1::packet` module and the packet extensions. |
| [03 The encoder](guide/03-encoder.md) | One call of `OpusEncoder::encode`: how the mode, bandwidth and channel count are chosen, the stages of the SILK and CELT encoders, and every encoder setting. |
| [04 The decoder](guide/04-decoder.md) | One call of `OpusDecoder::decode`: packet parsing, SILK and CELT decoding, mode switches, packet loss concealment, in-band forward error correction (FEC), deep redundancy (DRED), 16-bit and 24-bit output and the final range. |
| [05 Multichannel audio](guide/05-multichannel.md) | Multistream packets, channel layouts and mapping families, how the encoder shares the bitrate between streams, surround mode, and the application programming interface (API) of `opus1::multistream`. |
| [06 Using the library](guide/06-using-the-library.md) | Recipes for calls, voice agents and music; choosing frame duration, bitrate and complexity; packet loss; the neural network features of the decoder; the `#![no_std]` build; errors, threads and memory; build settings for speed; building and testing without cmake. |
| [07 Architecture](guide/07-architecture.md) | Every source file and the libopus file it ports, the main types, floating point and fixed point, feature flags, processor-specific code, `unsafe` code, error handling, and the layout of the tests and tools. |
| [08 Performance](guide/08-performance.md) | How speed is measured against libopus, the latest results, the changes that made the code fast, what did not help, how to reproduce the numbers, and the results on an Android phone and on x86-64. |
| [09 Testing](guide/09-testing.md) | How libopus is linked for the tests, what the tests compare, the groups of test files, the evaluation tools (the RFC 8251 conformance test, and the comparison of decoded audio with libopus's), the documentation examples, fuzzing, the `#![no_std]` checks, the tests on an Android phone and on an x86-64 computer, and the command for each part. |
| [10 Differences from libopus](guide/10-differences-from-libopus.md) | What `opus1` does not implement, where it differs on purpose, and how far its output differs from libopus, with the measured figures and the results of the evaluation tools; and the platforms it was built and tested on. |
| [Glossary](guide/glossary.md) | The terms used in these documents, in alphabetical order. |
| [The architecture of opus1](framework.md) | The overall design: the layers of the codec and how one encode and one decode call pass through them, the public interface, a map of the source files, the feature flags and the rules the code follows. |
| [Changes in 0.2.1](changelog-0.2.1.md) | The detailed list of changes from opus-rs 0.1.37 to opus1 0.2.1, with the measurements behind each fix. The README has the short release note. |

Other parts of the repository have their own README:

| Document | What it covers |
|---|---|
| [fuzz/README.md](../fuzz/README.md) | What each fuzz target checks and how to run it. |
| [tools/dnn-gen/README.md](../tools/dnn-gen/README.md) | How the neural network weights are generated and updated to a later libopus release. |
| [dnn-weights/README.md](../dnn-weights/README.md) | Which weight file belongs to which model and feature. |
| [docs/opus-explained/README.md](opus-explained/README.md) | How the explainer web page is built and how to edit it in both languages. |

## Reading order for users

1. [01 What Opus is](guide/01-what-is-opus.md), for the ideas and the terms.
2. [06 Using the library](guide/06-using-the-library.md), to write code that encodes and decodes.
3. [05 Multichannel audio](guide/05-multichannel.md), when you have more than two channels.
4. [10 Differences from libopus](guide/10-differences-from-libopus.md), before you depend on a libopus control or on bit-exact output.
5. [02 Opus packets](guide/02-packets.md), when you store, send or inspect packets yourself.

## Reading order for maintainers

1. [01 What Opus is](guide/01-what-is-opus.md) and [02 Opus packets](guide/02-packets.md), for the format.
2. [The architecture of opus1](framework.md), for the overall design.
3. [07 Architecture](guide/07-architecture.md), to find the code behind each part.
4. [03 The encoder](guide/03-encoder.md) and [04 The decoder](guide/04-decoder.md), for the code paths in the order they run.
5. [05 Multichannel audio](guide/05-multichannel.md), for the multistream layer.
6. [09 Testing](guide/09-testing.md), before you change code.
7. [08 Performance](guide/08-performance.md), before you change code for speed.
8. [10 Differences from libopus](guide/10-differences-from-libopus.md), for the known gaps and their sizes.

## Where to look for common questions

| Question | Where to look |
|---|---|
| How do I encode and decode? | [README quick start](../README.md#quick-start), [06](guide/06-using-the-library.md) |
| Which mode, bandwidth and bitrate does the encoder choose, and why? | [03, section 1.4](guide/03-encoder.md#14-mode-bandwidth-and-channels-decide_mode_and_bandwidth), `decide_mode_and_bandwidth` in `src/lib.rs` |
| What happens when a packet is lost? | [04, section 8](guide/04-decoder.md#8-packet-loss-concealment), [06, Handling packet loss](guide/06-using-the-library.md#handling-packet-loss) |
| How do the parts of the codec fit together? | [The architecture of opus1](framework.md) |
| Where is the Rust code for a libopus C file or function? | The tables in [07, Source layout](guide/07-architecture.md#source-layout). Or search `src/` for the C function name in lower case: most ports keep it, for example `silk_NSQ_del_dec` is `silk_nsq_del_dec` |
| Does `opus1` support a libopus feature or control? | [10, Not implemented](guide/10-differences-from-libopus.md#not-implemented) |
| Why does a result differ from libopus? | [10, Where the numbers differ](guide/10-differences-from-libopus.md#where-the-numbers-differ) |
| How do I build for a microcontroller or WebAssembly? | [README, `no_std`](../README.md#no_std--no-alloc-support), [06, Using the crate without the standard library](guide/06-using-the-library.md#using-the-crate-without-the-standard-library) |
| How much memory do the encoder and decoder use? | [06, Size of the encoder and the decoder](guide/06-using-the-library.md#size-of-the-encoder-and-the-decoder) |
| How fast is it, and how is speed measured? | [README, Performance](../README.md#performance), [08](guide/08-performance.md) |
| Which test covers a part, and how do I run it? | [09, Groups of test files](guide/09-testing.md#groups-of-test-files), [09, How to run each part](guide/09-testing.md#how-to-run-each-part) |
| How do I name branches and commits, and what must pass before a push? | [CONTRIBUTING.md](../CONTRIBUTING.md) |
| What does a term mean? | [Glossary](guide/glossary.md) |

## Repository layout

| Path | What it holds |
|---|---|
| [`Cargo.toml`](../Cargo.toml) | The package. Its comments explain every feature flag and every development dependency. |
| [`README.md`](../README.md) | Overview, quick start, `no_std` notes, latest timings and release notes. |
| [`AGENTS.md`](../AGENTS.md) | Working rules for AI coding tools. |
| [`CONTRIBUTING.md`](../CONTRIBUTING.md) | Branch names, commit messages, merge rules and the checks before a push. |
| [`src/`](../src) | The library. See the [source map](framework.md#source-map). |
| [`tests/`](../tests) | 289 integration test files and 17 helper folders. Most tests compare with libopus. See [Tests](#tests). |
| [`examples/`](../examples) | Runnable programs: the timing harness, the quality comparison harness, WAV round trip, multistream demo, and others. See [Examples](#examples-benchmarks-and-fuzzing). |
| [`benches/`](../benches) | Criterion benchmarks (`cargo bench`). |
| [`fuzz/`](../fuzz) | A separate package, `opus1-fuzz`, with 24 libFuzzer targets. See [fuzz/README.md](../fuzz/README.md). |
| [`tools/dnn-gen/`](../tools/dnn-gen) | A separate package that writes the neural network weights and model files from a libopus release. See [tools/dnn-gen/README.md](../tools/dnn-gen/README.md). |
| [`tools/libopus-fixed/`](../tools/libopus-fixed) | A test-only crate that builds the fixed-point libopus 1.6.1 from source with `cc`, with every symbol renamed, so the tests can call it next to the floating-point libopus. |
| [`scripts/`](../scripts) | Shell, PowerShell and Python scripts: `no_std` checks, x86 tests and timing, Android phone tests and timing, evaluation corpora, neural network test helpers. See [Scripts](#scripts). |
| [`dnn-weights/`](../dnn-weights) | The weights of the libopus neural network models, one blob per model. Generated; do not edit. See [dnn-weights/README.md](../dnn-weights/README.md). |
| [`fixtures/`](../fixtures) | Test audio: `answer_16k.wav` and four 48 kHz speech excerpts in `tsp/` (licence in `tsp/Licence.txt`). |
| [`docs/`](.) | The documentation: the numbered documents 01 to 10 and the glossary in [`docs/guide/`](guide), this page, the architecture and the changelog. |
| [`docs/opus-explained/`](opus-explained) | An interactive web page that explains Opus, in Chinese and English. Not part of the crate. See [its README](opus-explained/README.md). |
| [`bench_quick.sh`](../bench_quick.sh) | A short run of the timing harness, saved to a Markdown file. |

Folders that are ignored by git but that you may find in a working copy:

- `target/`: build output. Scripts also keep libopus builds, reports and phone results under `target/` (for example `target/android/`, `target/eval/`, `target/perf/`).
- `opus-1.6.1/`: an unpacked libopus 1.6.1 source tree, if a maintainer put one there.
- `corpus/`: downloaded test corpora for the evaluation tools.
- `.cargo/`: each maintainer's own cargo settings (for example where libopus is, and `RUST_MIN_STACK`).
- `fuzz/target/`, `fuzz/corpus/`, `fuzz/artifacts/`: fuzzing output.

## Tests

### How the tests compare with libopus

- Most integration tests run the same input through `opus1` and through libopus 1.6.1, then compare packets, decoded samples, internal values or the final range coder state.
- libopus is reached through two development dependencies: the `opus` crate (safe wrapper) and `audiopus_sys` (raw C interface). Both link the same library.
- Use libopus **1.6.1**; the expected values were made with it. The build script of `audiopus_sys` reads `LIBOPUS_LIB_DIR` and `LIBOPUS_NO_PKG`. On macOS with Homebrew: `LIBOPUS_LIB_DIR=/opt/homebrew` and `LIBOPUS_NO_PKG=1`. If neither pkg-config nor `LIBOPUS_LIB_DIR` finds a libopus, it builds the libopus 1.3 that it ships, with cmake. That is the wrong version for these tests. See [09, How libopus is linked](guide/09-testing.md#how-libopus-is-linked) and [06, Building and testing on a machine without cmake](guide/06-using-the-library.md#building-and-testing-on-a-machine-without-cmake).
- The Homebrew libopus is a floating-point build. `opus1`'s SILK encoder ports the fixed-point one. So the tests that check SILK encoding call the fixed-point libopus 1.6.1 of [`tools/libopus-fixed`](../tools/libopus-fixed/src/lib.rs). That crate finds the libopus source in `LIBOPUS_SRC_1_6_1`, else in `opus-1.6.1/`, else in `target/perf/libopus-1.6.1-src/opus-1.6.1/`. It must be the official 1.6.1 release, not libopus's main branch. Without the source the crate still builds, and its functions panic with the reason.
- Tests of the neural network code against libopus need a libopus built with that code (`scripts/android/build-libopus.sh --dnn`) and the feature `test-libopus-dnn`. See [09, Tests of the neural network code](guide/09-testing.md#tests-of-the-neural-network-code).
- Builds without `heap` keep large states on the stack. Set `RUST_MIN_STACK=33554432` for `cargo test --no-default-features`.

### Test files by name prefix

The file name prefix says what a test covers. Counts are files in `tests/`.

| Prefix | Files | What they cover |
|---|---:|---|
| `core_` | 39 | `OpusEncoder` and `OpusDecoder` entry points and controls against libopus: loss, FEC, mode switches, limits, analysis, DTX (discontinuous transmission), packets, the fixed-point reference |
| `multistream_` | 38 | Multistream encoder and decoder, end to end, surround analysis |
| `perfk_` | 33 | Single kernels after speed work, against the same libopus function |
| `dnn_` | 23 | Neural network code; the `_libopus` files need `test-libopus-dnn` |
| `k24_` | 21 | The 24 kHz path |
| `celtenc_`, `celtdec_`, `celt_` | 19, 4, 8 | CELT encoder and decoder |
| `silkenc_`, `silkdec_`, `silk_` | 6, 10, 14 | SILK encoder and decoder |
| `mdct_` | 16 | The MDCT |
| `issue_`, `issues_` | 14, 1 | Regression tests named after GitHub issues |
| `verify_`, `pvq_`, `hybrid_` | 5, 4, 4 | Band, PVQ, quantization and SILK functions; Hybrid mode |
| `eval_` | 5 | The evaluation tools, the RFC 8251 conformance test |
| `analysis_` | 3 | The speech and music analysis |
| others | 22 | Single topics, for example `stack_usage_test.rs`, `panic_surface_repro.rs`, `oracle_celt.rs`, `quality_regression_test.rs` |

Helper code lives in folders named after the tests that use them (`tests/core_common/`, `tests/k24_common/` and 15 others). Each holds a `mod.rs` that a test includes with `mod <name>;`. Two files of recorded reference values remain: `tests/eval_moc.expected` and `tests/eval_opus_compare.expected`.

Many test files start with a comment that says what they compare and which libopus function they follow. Read it first when a test fails. Unit tests also live inside `src/` in `#[cfg(test)]` modules; many compare a vector kernel with its portable version.

## Examples, benchmarks and fuzzing

| Path | What it is |
|---|---|
| `examples/perf_harness.rs` | The timing and comparison tool against libopus. Commands: `list`, `bench`, `run`, `golden`, `modes`, `settings`, `signals`, `resampler`, `spread`, `pairs`, `analysis`, `output`, `plc`. `golden` checks output hashes. |
| `examples/eval/` | The paired-comparison harness: `opus1` and libopus code the same items, and `opus_compare` and MOC score both. Also the RFC 8251 conformance check and a self-test. |
| `examples/opus_compare.rs` | libopus's `opus_compare` program, on top of `src/eval/compare.rs`. |
| `examples/wav_test.rs` | Encodes and decodes a WAV file (default `fixtures/answer_16k.wav`). |
| `examples/multistream.rs` | Encodes and decodes a 5.1 test signal and prints the quality of each channel. |
| `examples/opus_demo_data.rs` | Writes the data file of the explainer page. |
| `benches/opus_bench.rs` | Criterion benchmarks of whole encoding and of single kernels. |
| `benches/dnn_bench.rs`, `benches/deep_plc_bench.rs` | Neural network kernels and deep packet loss concealment. |
| `fuzz/fuzz_targets/` | 24 targets. 8 compare with libopus (wrappers in `fuzz/fuzz_targets/common/mod.rs`); 16 run `opus1` alone. Run with `fuzz/fuzz.sh` on the stable toolchain. |

## Scripts

| Path | What it does |
|---|---|
| `scripts/check_no_std.sh` | Builds without `std` for `thumbv7em-none-eabi`, `wasm32-unknown-unknown` (when installed) and the host, then runs the tests without `std`. |
| `scripts/x86/test.sh`, `scripts/x86/test.ps1` | On an x86-64 computer (Linux or WSL2, or Windows with MSVC): download and build libopus 1.6.1, run the whole test suite, the neural network kernel tests and the golden check, and pack the results into one file. See [09, Tests on an x86-64 computer](guide/09-testing.md#tests-on-an-x86-64-computer). |
| `scripts/x86/ab.sh`, `scripts/x86/ab.ps1`, `scripts/x86/ab_report.rs` | Time two builds of `opus1` in turns on one computer and report how much faster or slower the branch is. |
| `scripts/android/` | Tests and timing on a 64-bit ARM Android phone over adb: `env.sh` (shared settings), `build-libopus.sh` (libopus references, also `--dnn`), `test.sh`, `golden.sh`, `bench.sh`, `ab.sh`, `build-variant.sh`, `profile.sh`, `rustc-wrapper.sh`, `dump_weights_blob_all.c`. |
| `scripts/eval/` | Evaluation helpers: `fetch-corpus.sh` (download corpora), `check-opus-compare.sh` and `opus_compare_exact.c` (C reference of `opus_compare`), `moc_reference.py`, `tier2.py`. |
| `scripts/dnn/` | `build-osce-shim.sh` and `osce_test_shim.c`: a C helper library that `tests/dnn_osce_libopus.rs` loads. |
| `scripts/gen_analysis_mlp_data.py` | Writes `src/analysis/mlp_data.rs` from libopus's `mlp_data.c`. |

## Commands

Run tests with `--release`: they encode and decode seconds of audio per case.

| Task | Command |
|---|---|
| Build the library | `cargo build --release` |
| Build without `std` (bare metal) | `cargo build --release --no-default-features --features libm` |
| Run every test | `cargo test --release` (with libopus 1.6.1 found as above) |
| Run one test file | `cargo test --release --test core_encoder` |
| Run unit tests only | `cargo test --release --lib` |
| Run the document examples | `cargo test --release --doc` |
| Run an ignored report test | `cargo test --release --test <name> -- --ignored --nocapture` |
| Lint | `cargo clippy --all-targets -- -D warnings` |
| Build the API documentation with the features docs.rs uses (docs.rs also passes `--cfg docsrs`, which adds the labels of the features each item needs; on a stable compiler `RUSTC_BOOTSTRAP=1 RUSTDOCFLAGS="--cfg docsrs"` before the command shows them) | `cargo doc --no-deps --features dred,osce-bwe` |
| Check the `no_std` builds | `scripts/check_no_std.sh` |
| Time against libopus | `cargo run --release --example perf_harness -- bench` |
| Short timing run | `./bench_quick.sh <label> [filter]` |
| Check output hashes | `cargo run --release --example perf_harness -- golden --check <file>` |
| Quality comparison with libopus | `cargo run --release --example eval -- run` |
| Fuzz one target | `fuzz/fuzz.sh <target> <seconds>` |
| Regenerate the explainer data | `cargo run --release --example opus_demo_data` |
| Regenerate the neural network weights | see [tools/dnn-gen/README.md](../tools/dnn-gen/README.md#running) |

[09, How to run each part](guide/09-testing.md#how-to-run-each-part) has the full table, with the phone and evaluation commands.

## Rules for the documents

1. **Documents are tested.** `src/lib.rs` includes `README.md` and `docs/guide/01` to `docs/guide/10` as documentation tests. In those files, every code block tagged `rust` or without a tag is compiled and run (rustdoc treats an untagged block as Rust). Tag other blocks `text`, `toml`, `bash` and so on, or `rust,ignore` for Rust that should not run. This page, the architecture, the glossary and the changelog are not included.
2. **Plain writing.** Each document defines its terms when it first uses them and expands each acronym. Prefer short sentences and plain words.
