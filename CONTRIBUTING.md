# Contributing Guidelines

Branch naming, commit message, and merge rules for this repo.

## 1. Branch Naming

**The rule governs remote branch names only. Local branch names are never regulated** — name local branches (including git worktree branches) anything you like; when the names differ, push with an explicit refspec: `git push origin <local-name>:feat/short-description`.

Remote branches: `<type>/<short-description>` in kebab-case, optionally `<type>/<issue-id>-<short-description>`. E.g. `perf/x86-simd`, `fix/54-celt-prefilter-cancel-fade`.

Types (same set as commit types): `feat/` `fix/` `refactor/` `perf/` `docs/` `test/` `build/` `ci/` `chore/`.

Exempt remote branches: `release/*`, `hotfix/*`.

## 2. Commit Messages

Strictly [Conventional Commits](https://www.conventionalcommits.org/): `<type>(<scope>): <description>`, optional body, optional footers.

- **English only** — subject and body.
- Types: `feat` `fix` `refactor` `perf` `docs` `test` `build` `ci` `chore` `revert`.
- Scope = the area of the code. The repo is one crate, so the scopes are its parts (the [source map](docs/framework.md#source-map) lists the files):

  | Scope | What it covers |
  |---|---|
  | `opus` | The Opus layer and the shared code at the top of `src/`: `lib.rs`, `packet.rs`, `extensions.rs`, `error.rs`, `range_coder.rs`, `simd_mem.rs` and the other files of that group |
  | `celt` | CELT: `celt.rs`, `bands.rs`, `pvq.rs`, `mdct.rs`, `kiss_fft.rs`, `pitch.rs`, `rate.rs`, `quant_bands.rs`, `modes.rs`, `celt_lpc.rs` |
  | `silk` | `src/silk/` |
  | `analysis` | `src/analysis/` |
  | `multistream` | `src/multistream/` |
  | `dnn` | `src/dnn/` and `dnn-weights/` |
  | `eval` | `src/eval/` and `examples/eval/` |
  | `x86`, `aarch64` | Processor-specific code across several areas |
  | `examples` | `examples/`, except `examples/eval/` |
  | `bench` | `benches/` |
  | `fuzz` | `fuzz/` |
  | `tools` | `tools/` (`dnn-gen`, `libopus-fixed`) |
  | `scripts` | `scripts/` |
  | `docs` | `docs/` and `README.md` |

  A change to tests only takes the type `test` and the scope of the code under test (`test(silk): …`). Comma-separate when a change genuinely spans areas (`fix(celt,opus): …`); propose new scopes in the pull request instead of inventing them.
- Subject: imperative mood, lowercase after the colon, no trailing period, ≤72 chars.
- Body: wrap at 72 columns (git convention); explain what/why. Say whether the codec's output changes: on 64-bit ARM it must stay bit-exact with libopus 1.6.1, so give the result of the golden check (e.g. `Golden check: 0 of 137 differ`) or the reason the bits change. Give the timings for speed changes, and call out required follow-ups (e.g. regenerating the [generated files](docs/framework.md#generated-files-do-not-edit-by-hand)).
- Breaking changes: `!` after the scope (`feat(opus)!: …`) + impact in the body or a `BREAKING CHANGE:` footer. Changes that break callers of the public API count as breaking: the crate root, `packet`, `extensions`, `multistream`, the neural network types (`DnnBlob`, `DnnModels`, the DRED types) and the feature flags. The module `internals` and the test-only features `__internals` and `test-libopus-dnn` are not part of the API.
- Trailers (`Co-Authored-By:` …) go last, one per line.

## 3. Merge Strategy: Rebase Only

- Pull requests merge by rebase — linear history, no merge commits. Rebase your branch onto the latest target, resolve conflicts locally, then `git push --force-with-lease` your own branch; the target branch only fast-forwards.
- Never rebase or force-push shared branches (`main`, `release/*`) — rewriting history that others pull from and base work on breaks every downstream clone.
- Every commit that lands must individually follow §2; squash WIP/fixup commits with an interactive rebase before merging.

## 4. Before You Push

Run the checks for what you touched. [docs/guide/09-testing.md](docs/guide/09-testing.md#how-to-run-each-part) lists the command for each part.

- `cargo clippy --all-targets -- -D warnings` passes.
- `cargo test --release` passes. Always test with `--release`: the tests encode and decode seconds of audio per case. Don't use `--all-features`: `test-libopus-dnn` needs a libopus built with its neural network code. The tests link libopus, so set these first, e.g. in the `[env]` table of a git-ignored `.cargo/config.toml` ([how libopus is linked](docs/guide/09-testing.md#how-libopus-is-linked)):
  - `LIBOPUS_LIB_DIR=/opt/homebrew` and `LIBOPUS_NO_PKG=1`: link Homebrew's libopus on macOS instead of building it with CMake.
  - `RUST_MIN_STACK=33554432`: the `no_std` build keeps the whole codec state inline, so the tests need a large stack.
  - `LIBOPUS_SRC_1_6_1=<libopus 1.6.1 source>`: the fixed-point reference that `tools/libopus-fixed` builds. A git worktree needs it.
- Codec code: on 64-bit ARM the output stays bit-exact with libopus 1.6.1. Write a golden file before the change (`cargo run --release --example perf_harness -- golden --write <file>`) and check against it after (`golden --check <file>`). See [golden hashes](docs/guide/09-testing.md#golden-hashes-from-the-performance-harness).
- Speed changes: time them with `perf_harness`, as [docs/guide/08-performance.md](docs/guide/08-performance.md) describes.
- Neural network code: also run [its tests](docs/guide/09-testing.md#tests-of-the-neural-network-code) with its features.
- Feature flags or `no_std` code: `scripts/check_no_std.sh` passes.
- Documents: `README.md` and `docs/guide/01` to `docs/guide/10` are documentation tests, so `cargo test --release --doc` passes.
- x86 code: the tests on an Apple silicon Mac run the aarch64 code, not the x86 code. Run `scripts/x86/test.sh` (Linux, or WSL2, the Windows Subsystem for Linux) or `scripts/x86/test.ps1` (Windows) on an x86-64 computer. There the output must be correct, not bit-exact.
- Formatting: the code is not formatted with rustfmt, and `cargo fmt --check` reports differences in most files. Don't run `cargo fmt` on the whole tree; it would rewrite unrelated code. Match the style of the code around your change.
- The repo has no CI (continuous integration) yet, so run these checks yourself before you push.
