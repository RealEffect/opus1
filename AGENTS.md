# AGENTS.md

Rules for AI coding agents in this repository. They apply to every task unless the user overrides them.
Bias: caution over speed on non-trivial work. Use judgment on trivial tasks.

## 1. Think Before Coding

State assumptions explicitly. If uncertain, ask rather than guess.
Present the interpretations when a request is ambiguous.
Push back when a simpler approach exists.
Stop when confused. Name what is unclear.

## 2. Simplicity First

Use the minimum code that solves the problem. Nothing speculative.
No features beyond what was asked. No abstractions for single-use code.
Test: would a senior engineer say this is overcomplicated? If yes, simplify.

## 3. Surgical Changes

Touch only what you must. Clean up only your own mess.
Do not "improve" adjacent code, comments, or formatting.
Do not refactor what is not broken.
If you notice unrelated dead code or a likely bug, mention it; do not fix it in the same change.
Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

Define success criteria. Loop until verified.
Transform vague tasks into verifiable goals: "fix the bug" becomes "write a test that reproduces it, then make it pass".
If the area has no tests, verify another way (run it, check the output).
If your tool has a task list, keep the plan there instead of repeating it in messages.

## 5. Read Before You Write

Before adding code, read the file, its callers, and the shared helpers it uses.
Here that includes the libopus C file the Rust file ports: `docs/guide/07-architecture.md` names it for every file.
If you do not understand why existing code is structured a certain way, ask before adding to it.

## 6. Match the Codebase, Even If You Disagree

Inside the codebase, conformance beats taste.
Ported functions follow the structure of the C code and keep its names in lower case (`silk_NSQ_del_dec` is `silk_nsq_del_dec`), so a reader can check the Rust against the C.
The code is not formatted with rustfmt. Match the style around your change; do not run `cargo fmt` on the whole tree.
Generated files are regenerated, never edited by hand ([list](./docs/framework.md#generated-files-do-not-edit-by-hand)).
If two existing patterns contradict, pick the more recent or better-tested one, say why, and flag the other.
If a convention is genuinely harmful, surface it. Do not fork it silently.

## 7. Keep the Output Exact

On 64-bit ARM the output stays bit-exact with libopus 1.6.1, and the tests compare it.
On x86-64 the goal is correct audio and speed, not bit-exactness.
The neural network kernels give the 64-bit ARM bits on every processor.
A change that alters the output on purpose is recorded, with its measured size, in [`docs/guide/10-differences-from-libopus.md`](./docs/guide/10-differences-from-libopus.md).
The other rules for the code (`unsafe` comments, no allocation below the Opus layer, error handling) are in [`docs/framework.md`](./docs/framework.md#rules-the-code-follows).

## 8. Tests Verify Intent, Not Just Behavior

Most tests run the same input through `opus1` and through the linked libopus and compare the results.
A test that cannot fail when the logic changes is weak or wrong.
Do not treat shallow passing tests as proof of correctness.
Run tests with `--release`: they encode and decode seconds of audio per case.
Tests need these settings, which the git-ignored `.cargo/config.toml` of the main checkout holds; a new git worktree starts without them ([details](./docs/README.md#tests)):
- `LIBOPUS_LIB_DIR=/opt/homebrew` and `LIBOPUS_NO_PKG=1`: link Homebrew's libopus 1.6.1. Without them `audiopus_sys` builds the libopus 1.3 it ships, and the tests compare with the wrong version.
- `RUST_MIN_STACK=33554432`: builds without `heap` keep the codec state on the stack.
- `LIBOPUS_SRC_1_6_1=<libopus 1.6.1 source>`: the fixed-point libopus that `tools/libopus-fixed` builds for the SILK encoder tests.

## 9. Fail Loud

"Completed" is wrong if anything was skipped silently.
"Tests pass" is wrong if any relevant tests were skipped.
Name the checks you ran and their results, and the ones you could not run here (for example the x86 tests, or the tests against a libopus built with its neural network code).
Default to surfacing uncertainty, not hiding it.

## 10. Verification

Before calling a task done, run the checks for what you touched. This is the same list as [`CONTRIBUTING.md`](CONTRIBUTING.md) §4; keep the two in sync.

- `cargo clippy --all-targets -- -D warnings`.
- `cargo test --release`, with the settings of rule 8. Not `--all-features`: `test-libopus-dnn` needs a libopus built with its neural network code.
- Codec code: the golden check. Write a golden file before the change and check after it (`cargo run --release --example perf_harness -- golden --write <file>`, then `golden --check <file>`).
- Speed changes: timings with `perf_harness` ([docs/guide/08-performance.md](docs/guide/08-performance.md)).
- Neural network code: also its tests with its features.
- Feature flags or `no_std` code: `scripts/check_no_std.sh`.
- Documents: `cargo test --release --doc`.
- x86 code: `scripts/x86/` on an x86-64 computer. The tests on an Apple silicon Mac do not run it.

## 11. Git

See [`CONTRIBUTING.md`](CONTRIBUTING.md): branch names, Conventional Commits in English, rebase-only merges, the checks before a push.

## 12. Project Overview

See [`docs/README.md`](./docs/README.md) (documentation and project index) and [`docs/framework.md`](./docs/framework.md) (overall architecture).
