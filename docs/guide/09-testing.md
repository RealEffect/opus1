# Testing

This document explains how opus1 is tested. Most tests run the C library libopus in the same process and compare its output with the output of opus1. The sections below cover how libopus is linked, what the tests check, and the groups of test files. They then cover the checks against a fixed-point build of libopus, and the evaluation tools: the conformance test of RFC 8251 and the comparison of decoded audio with the original. Then come fuzzing, the `#![no_std]` checks, the ignored tests, the documentation examples, and the tests on an Android phone and on an x86-64 computer. The last section gives the command for each part.

## Terms used here

- **Opus** is the audio codec defined in RFC 6716 (Request for Comments 6716, the Internet standard for Opus). **libopus** is its reference implementation in C.
- **SILK** is the speech coder inside Opus. **CELT** (Constrained Energy Lapped Transform) is the transform coder inside Opus. **Hybrid** is the mode in which SILK codes the low band and CELT the high band.
- **Bit-exact** means that two implementations produce exactly the same bits: the same packet bytes, or the same output samples.
- The **signal-to-noise ratio (SNR)** of one signal against another, in decibels (dB), measures how close they are. A higher value means closer signals. Identical signals have no noise at all.
- **Packet loss concealment (PLC)** is what a decoder outputs for a packet that did not arrive.
- **Forward error correction (FEC)** in Opus is a low-bitrate copy of the previous frame inside a SILK packet. libopus calls the copy LBRR (low bitrate redundancy).
- The **TOC byte** (table-of-contents byte) is the first byte of an Opus packet. It names the mode, the audio bandwidth, the frame duration and the channel count.
- The **final range** is the state of the range coder after the last symbol of a packet. The range coder is the entropy coder of Opus.
- A **fixed-point** implementation computes with integers. A **floating-point** one computes with `f32` or `f64`.
- **NEON** is the vector instruction set of 64-bit ARM processors.

## The test suite in numbers

Counted with `cargo test --release -- --list` on 64-bit ARM macOS (`aarch64-apple-darwin`) with the default features, at commit `9891020`:

| Part | Files | Tests |
|---|---:|---:|
| Integration tests in `tests/*.rs` | 289 | 1302 |
| Unit tests inside `src/` | - | 408 |
| Documentation tests: the doc comments of `src/` (15) and the Rust examples of `README.md` and `docs/guide/` (55) | - | 70 |
| **Total** | | **1780** |

29 of these tests are ignored, and the listing shows one more documentation example apart, which is not run (see [Ignored tests](#ignored-tests)). Some unit tests exist only on 64-bit ARM, because they test NEON code. So the count differs on other targets. `tests/` also has 17 directories of shared helper code (for example `tests/k24_common/` and `tests/analysis_common/`) and 2 files of recorded reference values (`tests/eval_moc.expected` and `tests/eval_opus_compare.expected`).

A full run, `cargo test --release` with the RFC 8251 test vectors present (see [Corpora](#corpora)), passed all 1645 tests that are not ignored on Apple M4 Pro (commit `5e899cd`). On the branch `next-dred` (commit `8aedd12`), `cargo test --release` passed all 1659 tests that are not ignored. On the branch `next-gaps`, with the tests of phase inversion disabling, the decoder gain and pitch, the 24-bit calls and the multistream controls (`core_controls_libopus`, `multistream_controls_libopus`), it passed all 1676 tests that are not ignored (30 ignored). On the branch `next-release` (commit `3ecd7e0`, the x86 work included), `cargo test --release` without the corpora passed all 1676 tests that are not ignored (30 ignored, the documentation example included). On the branch `release/0.2.0-api`, which reviewed the public interface for the first release, `cargo test --release` passed all 1742 tests that are not ignored (30 ignored, the documentation example included). At commit `9891020`, with the fixed-point build of `libopus-fixed` compiled from the 1.6.1 release, `cargo test --release` without the corpora passed all 1751 tests that are not ignored (30 ignored, the documentation example included) on Apple M4 Pro.

The 23 files of the neural network code (`tests/dnn_*.rs`) have no tests in the default build: they need a model feature, and 12 of them also need `test-libopus-dnn` (see [Tests of the neural network code](#tests-of-the-neural-network-code)).

With the neural network features the suite must be linked with a libopus built with them (see [Builds with the neural network features](#builds-with-the-neural-network-features)). With deep PLC compiled, opus1 decodes losses at 16 kHz as libopus built with deep PLC does, and that differs from the Homebrew libopus also at complexity 0 (the fade-in after a loss). With OSCE compiled, a sample of -32768 in a 20 ms SILK frame at 16 kHz becomes -32767 at every complexity, as in libopus built with OSCE. Linked with `target/android/libopus-host-1.6.1-dnn-novec`, `cargo test --release --features test-libopus-dnn,osce-bwe` passed all 1811 tests (33 ignored) on Apple M4 Pro, with deep PLC, LACE, NoLACE and the BWE in the decoder. Linked with the Homebrew libopus, `--features osce-bwe` passes 1696 tests and fails 86 tests in 33 files (31 ignored): comparisons of decoded audio with losses (for example `core_loss.rs`, `core_fec.rs`, `k24_silk_loss_*.rs`, `silkdec_*.rs`, `multistream_dec_loss.rs`) and of loud signals that reach -32768 (for example `silkdec_saturation.rs`, `core_decode_i16.rs`). All of them pass against the novec build.

## How the tests reach internal code

Many tests call internal parts of the codec, such as the range coder or a SILK function, through the hidden module `opus1::internals` (see [Architecture](07-architecture.md#the-public-interface-and-internals)). That module exists only with the cargo feature `__internals`. The feature is for the repository only: it is outside semantic versioning, and applications do not enable it.

The tests, benchmarks and examples get the feature from a development dependency of opus1 on itself, in `Cargo.toml`:

```toml
[dev-dependencies]
opus1 = { path = ".", default-features = false, features = ["__internals"] }
```

Cargo merges the features of both requests for opus1. So `cargo test`, `cargo bench` and `cargo build --examples` build the library with `__internals`, and `cargo build` builds it without. `default-features = false` keeps the other features as the command line chose them: `cargo test --no-default-features --features libm` still tests a `no_std` build. `cargo package` removes this dependency from the published crate, because it has only a `path`. So the benchmarks and the examples that use `internals` list `__internals` in their `required-features`: cargo skips them in the published crate instead of failing to compile them. The fuzz targets turn the feature on in `fuzz/Cargo.toml`.

## How libopus is linked

Three development dependencies give the tests access to libopus:

- `opus` 0.3: a safe Rust wrapper around the libopus encoder and decoder.
- `audiopus_sys` 0.2: the raw C interface of libopus. The tests need it for controls that `opus` does not expose. Examples are forcing the coding mode (`OPUS_SET_FORCE_MODE`), decoding with the FEC flag (`opus_decode_float` with `decode_fec = 1`), the repacketizer and the multistream functions.
- `libopus-fixed`: a test-only crate of this repository (`tools/libopus-fixed/`). It compiles the fixed-point build of libopus 1.6.1 from its source, for the tests of the SILK encoder (see [Checks against the fixed-point build of libopus](#checks-against-the-fixed-point-build-of-libopus)).

`opus` uses `audiopus_sys`, so both reach the same libopus. Some tests also call internal functions of libopus directly, such as its fast Fourier transform (FFT), its pitch search and its comb filter (`tests/perfk_fft_libopus.rs`, `tests/perfk_pitch_libopus.rs`, the module `comb_filter_libopus_tests` in `src/celt.rs`). That works because libopus is linked as a static library, which is the default of `audiopus_sys` on macOS.

A rounding difference of one unit in the last place changes a result only near a decision, and ordinary test signals seldom come that close. Some tests therefore build such cases on purpose. `tests/perfk_pitch_libopus.rs` uses signals that repeat exactly, so that two pitch lags tie, quiet signals, where `1 + xx * yy` in the pitch gain is near 2^24, and a bisection that finds where `remove_doubling` switches between two periods. Each of them fails when one fused multiply-add of libopus is replaced by a separate multiply and add. The other tests do not notice that change.

The build script of `audiopus_sys` looks for libopus in this order:

1. With `pkg-config`, unless the environment variable `LIBOPUS_NO_PKG` (or `OPUS_NO_PKG`) is set.
2. In the directory named by `LIBOPUS_LIB_DIR` (or `OPUS_LIB_DIR`). It links from the `lib` folder inside that directory.
3. Otherwise it builds the copy of libopus 1.3 that it ships. This needs `cmake`.

The tests and the expected values in them were made with **libopus 1.6.1**. Use that version. On macOS with Homebrew, for example:

```text
brew install opus
export LIBOPUS_LIB_DIR=/opt/homebrew
export LIBOPUS_NO_PKG=1
cargo test --release
```

The maintainers keep the same settings in a local `.cargo/config.toml`, which is not in git:

```toml
[env]
LIBOPUS_LIB_DIR = "/opt/homebrew"
LIBOPUS_NO_PKG = "1"
```

The libopus that Homebrew ships is a floating-point build. Its SILK encoder is the floating-point one. The SILK encoder of opus1 is a port of the fixed-point one, so SILK encoder packets are compared against a separate fixed-point build, which `libopus-fixed` compiles. It needs the source of libopus 1.6.1 (see [Checks against the fixed-point build of libopus](#checks-against-the-fixed-point-build-of-libopus)).

## What the tests check

Most tests drive opus1 and libopus with the same input and settings, then compare. The checks fall into these kinds:

1. **Same packets, same decoded output.** libopus encodes a stream. Both decoders decode it, often with the same lost packets. The SILK decoder must give identical 16-bit samples. On 64-bit ARM the CELT and Hybrid output is libopus's too (see [Differences from libopus](10-differences-from-libopus.md#decoder)). Older tests still require only an SNR against libopus's output above a stated bound.
2. **Same settings, same decisions.** Both encoders code the same input with the same settings. The test compares the TOC bytes (so the mode, bandwidth and frame size), the packet sizes, or the packets byte for byte.
3. **Same quality.** Where packets differ, both encoders' packets are decoded by libopus and compared with the input. The test bounds the difference in SNR.
4. **The same final range.** After each packet, the encoder and the decoder both report their final range (`OpusEncoder::final_range`, `OpusDecoder::final_range`). Equal values show that the decoder read exactly the bits the encoder wrote. The libopus test program `opus_demo` makes the same check: it reports "Range coder state mismatch" when the two values differ.
5. **Properties without libopus.** Round trips, damaged and random packets (nothing may panic), extreme arguments, no heap allocation per frame, and the stack use of the default build.

The final-range check, written as a test of the kind the suite contains:

```rust
use std::num::NonZeroU32;
use opus1::{Application, Bitrate, Channels, OpusDecoder, OpusEncoder, SampleRate};

let mut encoder = OpusEncoder::new(SampleRate::Hz16000, Channels::Mono, Application::Voip).unwrap();
encoder.set_bitrate(Bitrate::BitsPerSecond(NonZeroU32::new(20_000).unwrap()));
let mut decoder = OpusDecoder::new(SampleRate::Hz16000, Channels::Mono).unwrap();

let frame = 320; // 20 ms at 16 kHz
let mut packet = vec![0u8; 1276];
let mut decoded = vec![0.0f32; frame];
for n in 0..50 {
    // A 440 Hz tone; each frame continues the one before.
    let pcm: Vec<f32> = (0..frame)
        .map(|i| {
            let t = (n * frame + i) as f32 / 16_000.0;
            0.3 * (2.0 * std::f32::consts::PI * 440.0 * t).sin()
        })
        .collect();
    let len = encoder.encode(&pcm, frame, &mut packet).unwrap();
    let samples = decoder.decode(&packet[..len], frame, &mut decoded).unwrap();
    assert_eq!(samples, frame);
    // The decoder read exactly the bits the encoder wrote.
    assert_eq!(decoder.final_range(), encoder.final_range());
}
```

Many test files start with a comment that explains what they compare and which libopus function they follow. Read that comment first when a test fails.

## Groups of test files

A test file's name starts with a prefix that says what it covers. The counts are from the listing above.

| Prefix | Files | Tests | What the files cover |
|---|---:|---:|---|
| `multistream_` | 38 | 300 | The multistream encoder and decoder (several Opus streams in one packet, for surround sound) against libopus's multistream functions: layouts (`_c_layout`), rates, the decoder (`_dec_*`), the encoder (`_enc_*`), end to end (`_e2e_*`), the surround analysis (`_surround_*`), and the speech and music analysis and DTX of each stream (`_analysis_libopus`, `_analysis_walk_libopus`, `_dtx_libopus`). |
| `core_` | 39 | 251 | The top layer, `OpusEncoder` and `OpusDecoder`, against libopus: controls, the error codes (`core_error_kinds_libopus`), channel counts, packet loss, FEC, mode switches and their redundant frames, packet framing, limits, 16-bit output, the high-pass filter in front of SILK, the stereo width estimate (`core_stereo_width_libopus`, bit for bit against libopus's `compute_stereo_width`), the speech and music analysis (`core_analysis_libopus`, and with settings that change during a stream `core_analysis_walk_libopus`), DTX (`core_dtx_libopus`), the decoder controls of the neural network features, and the packets of the fixed-point build (`core_fixed_point`). |
| `dnn_` | 23 | 0 | The neural network code: kernels, the layers of every network, weights, signal analysis, the models of deep PLC and of OSCE. Their tests compile only with a model feature (see [Tests of the neural network code](#tests-of-the-neural-network-code)). |
| `perfk_` | 33 | 114 | Guards for the speed work. Each rewritten kernel is compared with the same function of the linked libopus or of its fixed-point build, or with the code from before the rewrite (`tests/perfk_nsq_replay.rs` keeps the old quantizers for that). |
| `celtenc_` | 19 | 93 | Decisions of the CELT encoder against libopus: band energies, stereo, the pitch pre-filter, the low-frequency effects (LFE) channel, the energy mask, mode switching, the rate rules (`celtenc_rate_rules.rs`), the results of the speech and music analysis (`celtenc_analysis_info_libopus.rs`). |
| `issue_`, `issues_` | 15 | 89 | Regression tests named after GitHub issues, most of them against libopus (for example `issue_42_stereo_channels.rs`). `issues_repro.rs` holds 14 of the tests. |
| `k24_` | 21 | 72 | The 24 kilohertz (kHz) path: decoding at every output rate, encoding in each mode, SILK loss handling. `k24_main_use.rs` lets libopus choose the mode, as in a real call. |
| `silk_` | 14 | 67 | SILK parts tested one by one: the bitstream, Burg analysis, linear predictive coding (LPC), the normalized line spectral frequencies (NLSF), the quantizer, and packets of the fixed-point build. |
| `celtdec_` | 4 | 38 | The CELT decoder: channel conversions, edge cases, loss concealment and robustness, against libopus. |
| `silkdec_` | 10 | 31 | The SILK decoder alone (`SilkDecoder`), against libopus with SILK-only streams. These require identical 16-bit samples. |
| `eval_` | 5 | 26 | The evaluation tools: the ports of `opus_compare`, MOC and the `opus_demo` loop against their C and Python originals, the RFC 8251 conformance test, and the paired-comparison harness (see [Evaluation tools](#evaluation-tools)). |
| `silkenc_` | 6 | 25 | The SILK encoder: control, FEC data, mono coding of stereo input, pulses, restarts, robustness of its packet functions. |
| `analysis_` | 3 | 21 | The speech and music analysis alone against libopus, and its properties (see [Checks of the speech and music analysis and of DTX](#checks-of-the-speech-and-music-analysis-and-of-dtx)). |
| `hybrid_` | 4 | 19 | Hybrid mode encoding and decoding. |
| `celt_` | 8 | 17 | CELT round trips and regressions. |
| `mdct_` | 16 | 17 | The modified discrete cosine transform (MDCT): identity, gain and round trips. |
| `verify_` | 5 | 17 | Band energies, normalisation, energy quantization, PVQ (pyramid vector quantization), the SILK decoder and the SILK resampler against the same libopus functions on the same input; and a SILK round trip. Since commit `5c833c1` they call libopus instead of comparing with values that small C programs once printed. The exact comparison of band results is ignored off 64-bit ARM, where one energy differs in its last bit; a comparison within a few units in the last place runs everywhere. |
| `pvq_` | 4 | 7 | The PVQ codebook: its tables, its signs, and agreement between encoder and decoder. |
| other | 22 | 98 | Single files, for example `stack_usage_test.rs` (struct sizes and stack use), `panic_surface_repro.rs` (inputs that used to panic), `fuzz_toc_sequences.rs`, `proptest_encode_indices.rs` and `oracle_celt.rs`. |

Two files compare against reference files that are not in the repository. `tests/c_interop_diff.rs` reads `/tmp/opus_ref/`. `tests/multistream_interop.rs` reads `tmp/opus_ref/` under the crate root. Without those files their tests print a message and pass without checking anything.

## Bit-exactness checks

### Golden hashes from the performance harness

`examples/perf_harness.rs` (see [Performance](08-performance.md#the-harness)) can record hashes of everything the codec produces, for 137 streams. The file it writes is called a golden file: a stored record of the expected output that later runs are checked against. Per stream, a line holds these fields:

| Field | Hash of |
|---|---|
| `enc` | opus1 packets from `encode` |
| `enc_i16` | opus1 packets from `encode_i16` on the same audio |
| `dec_rust` | opus1 decoder output for the opus1 packets |
| `c_pk` | libopus packets; it changes only when libopus or the input changes |
| `dec_c` | opus1 decoder output for the libopus packets |
| `dec_c_plc` | the same, with one packet in ten lost |
| `dec_c_burst` | the same, with three more packets in a row lost out of every 25 |
| `fec_silk` | SILK decoder output with lost packets rebuilt from FEC data |
| `fec_frames` | how many frames that rebuild used (a count, not a hash) |
| `errs` | packets the opus1 decoder rejected (a count) |

Write the hashes before a change and check them after it:

```text
cargo run --release --example perf_harness -- golden --write target/golden_before.txt
cargo run --release --example perf_harness -- golden --check target/golden_before.txt
```

`--check` exits with status 1 when a field differs and names the fields that changed. The commit messages of the speed changes report the result of this check, for example "Golden check: 0 of 137 differ". A change that alters the output on purpose says so. For example, commit `390ab40` changed the FFT output on 64-bit ARM on purpose, and its message reports that 87 of 137 streams differ.

The speech and music analysis changed the output on purpose. Against the golden file of round 7 (`target/perf/golden_r7.txt`), 56 of 137 streams differ, and `target/perf/golden_analysis.txt` holds the new hashes:

- 44 CELT streams at complexity 7 to 10 and 16 to 48 kHz (`enc`, `enc_i16`, `dec_rust`): the analysis now runs. In 42 of them `enc` now equals `c_pk`, the packets of libopus. The other two are VBR packets of 40 and 60 ms, where opus1 leaves out libopus's limit of each frame on purpose. Over all 137 streams, `enc` equals `c_pk` in 49 (7 before).
- 4 Hybrid streams at 48 kHz (`enc_i16` only): 16-bit input is now coded with an input precision of 16 bits, as libopus `opus_encode` codes it. This changes CELT's silence threshold and noise floor in the high band.
- 8 mode-switching streams (`enc`, `dec_rust`, and in 5 of them the mix of modes): they now choose the modes, bandwidths and channel counts of libopus in all 8.

No field of libopus's own packets (`c_pk`) and of the decoding of libopus's packets changed. The NEON code of the analysis changed nothing: its steps were checked against the golden file of the scalar analysis, 0 of 137 differ. On the phone, the harness wrote a golden file identical in every field to the one of the Mac.

### Vector kernels against scalar code

The vector kernels have unit tests that compare them with scalar versions of the same code, bit for bit where the two must agree. Examples from `src/`:

- `kiss_fft::tests::test_neon_butterflies_match_scalar`
- `mdct::bit_exact_tests::neon_fold_matches_scalar`
- `pitch::tests::neon::xcorr_by_sample_neon_matches_scalar`
- `celt::comb_inplace_tests::const_part_neon_matches_portable_version`
- `analysis::neon::tests::tonality_bins_match_the_scalar_code`, and the other tests of `src/analysis/neon.rs` (downmix, window, network) and `analysis::tests::vector_silence_test_matches_the_scalar_loop`

Run them with `cargo test --release --lib`. The NEON tests run only on 64-bit ARM, and the x86 tests only on x86-64. The x86-64 vector code runs in the scripts of [Tests on an x86-64 computer](#tests-on-an-x86-64-computer). Its output is correct but not bit-exact with libopus (see [Differences from libopus](10-differences-from-libopus.md#platforms)).

## Checks against the fixed-point build of libopus

The SILK encoder of opus1 is a port of the fixed-point SILK encoder of libopus (`silk/fixed`). The linked libopus is a floating-point build, so it does not contain that encoder. The tests therefore also link a fixed-point build of libopus, and call it on the same input as opus1. The test-only crate `libopus-fixed` (`tools/libopus-fixed/`) compiles that build when the tests are built:

1. Its build script (`tools/libopus-fixed/build.rs`) compiles the source of the official libopus 1.6.1 release with the `cc` crate, without `configure` or cmake. The C files are the ones that `opus_sources.mk`, `celt_sources.mk` and `silk_sources.mk` list (`OPUS_SOURCES`, `OPUS_SOURCES_FLOAT`, `CELT_SOURCES`, `SILK_SOURCES` and `SILK_SOURCES_FIXED`), so the build has no vector code and no neural network code. The flags are `-O2 -DFIXED_POINT -DOPUS_BUILD -DVAR_ARRAYS -DHAVE_LRINTF -DHAVE_LRINT`. MSVC (Microsoft's C compiler) has no variable-length arrays and gets `USE_ALLOCA` instead. That changes where the scratch memory is, not the results.
2. Both builds of libopus define the same symbols, such as `opus_encode` and `silk_NSQ_c`. So the build script gives every global symbol of the fixed-point build the prefix `libopus_fixed_`, and checks that no symbol is left without it. Then the two builds link into one test program.
3. `tools/libopus-fixed/shim.c` adds plain C entry points, named `fixed_ref_*`. They cover an encoder with its controls, and the SILK functions that take the encoder state, whose layout depends on how libopus was built. `tools/libopus-fixed/src/lib.rs` wraps them for the tests: `Encoder`, `burg_modified`, `pitch_analysis_core`, `hp_variable_cutoff`, `control_audio_bandwidth`, `nsq` and `process_nlsfs`.

The build script looks for the source in this order:

1. the folder that the environment variable `LIBOPUS_SRC_1_6_1` names;
2. `opus-1.6.1/` in the repository;
3. `target/perf/libopus-1.6.1-src/opus-1.6.1/`, where `scripts/android/build-libopus.sh` looks too (see [Which libopus source to build](#which-libopus-source-to-build)).

A git worktree has its own `target/` folder, so a test run there needs `LIBOPUS_SRC_1_6_1`, for example in `.cargo/config.toml`:

```toml
[env]
LIBOPUS_SRC_1_6_1 = "/path/to/opus-1.6.1"
```

Without the source, or when its build fails, cargo prints a warning and builds the crate without the library. Each of its functions then panics with the reason. So the tests that need the fixed-point build fail and say why, and the other tests still run. After you put the source in place, run `cargo clean -p libopus-fixed` so that cargo runs the build script again. `scripts/x86/test.sh` and `test.ps1` unpack the release they download and set `LIBOPUS_SRC_1_6_1` themselves.

These tests call the fixed-point build:

| File | What it compares |
|---|---|
| `silk_fixed_point_compare.rs` | An encoder of the fixed-point build next to `OpusEncoder`: SILK-only, `Voip`, mono, 16-bit input. Every packet, byte for byte. |
| `core_fixed_point.rs` | The same for more settings: stereo input, complexity changes, digital silence, prediction disabled, SILK rate switches, and DTX decided by SILK (every packet and `in_dtx`). For Hybrid packets and rate switches it compares the SILK layers only. |
| `core_voip_input_filter.rs` | With the integer high-pass filter, opus1's SILK-only packets are those of the fixed-point build. |
| `perfk_silk_burg.rs`, `perfk_silk_pitch_core.rs`, `perfk_silk_nsq_plain.rs`, `perfk_silk_process_nlsfs.rs` | Burg's method, the core of the pitch analysis, the plain noise-shaping quantizer and the NLSF processing, on the same input. The quantizer test compares the whole quantizer state after every frame. |
| `silk_nsq_test.rs` | `silk_NSQ_c` and `silk_NSQ_del_dec_c` from the same state: the pulses, the whole quantizer state and the seed, with 1 to 4 states of the delayed-decision quantizer. |
| `silk_hp_variable_cutoff.rs` | `silk_HP_variable_cutoff` on the same sequence of encoder states. |
| `silkenc_control.rs` | `silk_control_audio_bandwidth`, called on both sides. |

Until commit `9218041` (2026-10-26) these tests held values recorded once from C programs: packet bytes as hexadecimal text, hashes of packets, and four `.expected` tables. They now call the build itself.

Do not point `LIBOPUS_SRC_1_6_1` at the libopus source that the crate `opus-head-sys` vendors. That source is libopus's main branch (commit `228a0f85`). Its fixed-point SILK encoder adds 1 to the frame energy of the Burg analysis (`silk/fixed/burg_modified_FIX.c`, line 72), which the 1.6.1 release does not. With it, the packets of every stream in `tests/core_fixed_point.rs` differ from 1.6.1's from packet 0 to 13 on, and 622 of the 1000 cases of `tests/perfk_silk_burg.rs` differ. Until October 2026 several of these tests held values from that source (see [Differences from libopus](10-differences-from-libopus.md#libopus-main-branch-and-the-161-release)).

These tests require identical results: the same packets, or the same integers. In the streams of these tests, the SILK-only packets of opus1 with the `Voip` application and 16-bit input (`OpusEncoder::encode_i16`) are byte for byte those of the fixed-point build. The same holds for 47 of the 49 SILK streams of the golden file. At 48 kHz opus1's SILK reads the float high-pass filter, not the fixed-point build's integer one (see [Differences from libopus](10-differences-from-libopus.md#deliberate-differences)). A test that compares packets at 48 kHz first calls the test control `internals::silk_input::set_integer_hp_at_48k`, which brings back the integer filter. In rare frames opus1's SILK also takes a quantization gain from the residual energy of the unscaled signal, where the fixed-point build reads a residual energy with a large rounding error (see [Differences from libopus](10-differences-from-libopus.md#silk)). None of the streams of these tests has such a frame. A test that needs the fixed-point build's gains in such frames calls the test control `internals::silk_input::set_fixed_point_res_nrg`.

The fixed-point build runs libopus's speech and music analysis only at complexity 10, for input at 16 to 48 kHz (the float build runs it from complexity 7). opus1 has no such analysis. In the streams of these tests at complexity 10 with 16 kHz input, and in `silk_wb_16k_mono_20k_vbr_c10_20ms` of the golden file, the packets are equal all the same. [Differences from libopus](10-differences-from-libopus.md#silk) lists the cases where the packets of longer streams start to differ.

## Evaluation tools

The tests above ask whether opus1 gives libopus's bits. The evaluation tools ask a different question: how far is the decoded audio from the original, with opus1's encoder and with libopus's, for the same input? They also run the conformance test of the Opus standard. They are:

- three ports, in `src/eval/`: libopus's test program `opus_compare`, the MOC distance of `moc.py`, and the decoding loop of libopus's test program `opus_demo`;
- the paired-comparison harness `examples/eval/`;
- the program `examples/opus_compare.rs`, with the command line of libopus's `opus_compare`;
- the scripts in `scripts/eval/`;
- the tests `tests/eval_*.rs`, with their shared code in `tests/eval_common/`.

The ports are reached as `opus1::internals::eval`. Like the rest of `internals`, this module is not part of the public application programming interface (API) (see [Architecture](07-architecture.md#the-public-interface-and-internals)). It is compiled only with the `std` feature, because it allocates and calls the platform's math library, and with the test-only feature `__internals` (see [How the tests reach internal code](#how-the-tests-reach-internal-code)).

Terms of this section:

- An **item** is one audio clip. A **condition** is one set of encoder settings. A **job** codes one item under one condition, once with opus1's encoder and once with libopus's.
- A **corpus** is a collection of items that is not in the repository.
- The **internal weighted error** (`err`) is the result of `opus_compare`: 0 for identical spectra, larger for a larger difference. Its quality value is `Q = 100 * (1 - 0.5 * ln(1 + err) / ln(1.13))`. A decoded signal passes the conformance test when `Q` is at least 0, that is when `err` is at most about 0.2769.
- **MOC** ("modified opus_compare") is the distance that the objective test of the Opus Speech Coding Enhancement draft (OSCE, `draft-ietf-mlcodec-opus-speech-coding-enhancement`) uses.
- A **geometric mean** of ratios is `exp(mean(ln r))`. A **bootstrap** confidence interval is computed by drawing the items again, with replacement, many times.

### The ports in `opus1::internals::eval`

| Module | Port of | What it offers |
|---|---|---|
| `compare` | `src/opus_compare.c` of libopus 1.6.1 | `opus_compare` (one test signal against one reference), `opus_compare_many` (several of each; each spectrum is computed once), `Comparison` (error, quality, frame count, worst frame) |
| `moc` | `compare` in `moc.py` of the libopus git repository (`dnn/torch/osce/stndrd/evaluation/moc.py`, BSD licence) | `moc`, `moc_i16`, and `moc_relative`, the value D of the draft's relative test |
| `demo` | the decoding loop of `opus_demo -d` (`src/opus_demo.c`) | `parse_bitstream`, `decode_bitstream` over the `DemoDecoder` trait, `opus_demo_s16` (one sample of the 16-bit output of `opus_demo`) |

**`opus_compare`.** The C program cuts both signals into 10 ms windows, one every 2.5 ms. It computes the power of each 100 Hz bin, spreads the reference's band energies across bands, over time and between the channels, and adds a tenth of that to both spectra. Per bin it takes `r - ln(r) - 1`, with `r` the ratio of the two powers. Bins near 8 kHz count less. The error is pooled over the 21 bands and over the frames. At a test rate below 48 kHz, only the bands below that rate's Nyquist frequency are compared, and the top 300 Hz are left out (except at 12 kHz).

The port gives the bits of the C program built with clang at `-O2` for the same processor:

- On 64-bit ARM, clang fuses some products and sums of the C code into one fused multiply-add. The port uses `mathops::mac` and `mac_f64` at exactly these places. On other processors both round twice.
- It calls the platform's `cos`, `sin`, `log` and `pow`, as the C program does. Each table is built in its own function, so that the compiler cannot merge a `cos` and a `sin` of the same angle into one `sincos` call, which can round differently.
- `ln(1.13)` is the constant that clang folds it to (`0x3FBF49A79F8B9187`).

The C program keeps the spectra of all frames. The port keeps only the frame before, so it needs a few kilobytes whatever the length. Its transform computes several bins side by side in NEON lanes. Each lane does the operations of the C loop in the same order, so the result does not change. On 20 s of stereo audio at 48 kHz the port took 0.18 s of processor time and the C program 3.25 s. Both gave the error 0.30854032158860312.

How the port is checked:

- `tests/eval_opus_compare.rs` holds 40 cases with values recorded from the C program (`tests/eval_opus_compare.expected`): every rate, mono and stereo, identical signals, silence, noise at several levels, a low-pass, a 1 ms shift, clipping, exactly one window, a partial last frame, a tone near 8 kHz, and 5 cases of the error messages. On 64-bit ARM macOS, the error and the quality must have the recorded bits. On other platforms they must agree to a relative 1e-6, because their math libraries can round `cos`, `sin`, `log` and `pow` differently. On the Android phone all 55 recorded values had the bits of the Mac.
- With `OPUS1_OPUS_COMPARE_C` naming the C program, the same test runs it on the cases and compares the printed lines too. `scripts/eval/check-opus-compare.sh --src <libopus 1.6.1 source>` builds it twice: as libopus ships it, and through `scripts/eval/opus_compare_exact.c`, which prints the numbers with 17 digits. The script checks that both builds contain the same number of fused multiply-adds (33) and that neither calls `sincos`.
- Property tests (with `proptest`) check that random inputs give an error of at least 0, that identical signals give 0, and that a mono reference equals a stereo reference with the same samples on both sides, bit for bit.
- Unit tests in `src/eval/compare.rs` compare the port with a direct transcription of the C code, and the vector kernel with the scalar loop, bit for bit.

**MOC.** MOC compares two 16 kHz mono signals in the way of `opus_compare`, with these differences:

- 10 ms windows every 2.5 ms with a Hamming window, and 81 bins of 100 Hz;
- 17 bands up to 7.5 kHz;
- the distortion of a bin is `ln(r)^2`;
- the error of a frame is the mean over the bands, and the result is the 6th root of the mean of its cube;
- with the activity detection, frames where the reference is quiet are left out.

The port computes in `f64`, and in `f32` where numpy keeps `float32` (numpy 2 rules). Its transform is a direct discrete Fourier transform instead of numpy's fast Fourier transform (FFT). So it agrees with numpy to about 1e-15, not to the bit. `tests/eval_moc.rs` compares 18 inputs, each with and without the activity detection, with values recorded from `scripts/eval/moc_reference.py`, a numpy-only transcription of `moc.py`. The limit is a relative 1e-9. With `OPUS1_EVAL_PYTHON` naming a Python 3 with numpy, the test runs the script itself. With `OPUS1_MOC_PY` naming the original `moc.py` as well, the script first checks its transcription against that file. The largest difference measured was a relative 9.8e-16, and 8.2e-16 on the phone.

`moc.py` comes from a snapshot of the libopus git repository that is older than libopus 1.6. Draft 04 of the OSCE draft replaced it by a C program, `osce_compare.c`, which is not available here and may compute other numbers.

**The `opus_demo` loop.** An `opus_demo` bitstream is a sequence of records. Each record holds a 4-byte big-endian length, the encoder's final range in 4 bytes, and the packet. A length of 0 marks a lost packet. `decode_bitstream` decodes such a stream as `opus_demo -d <rate> <channels> -ignore_extensions` does:

1. A lost packet is decoded at the next packet that arrived. The last lost packet is recovered with forward error correction (FEC) when that packet carries a redundant copy (`opus1::packet::has_lbrr`). The others are concealed. Both use the duration of the last packet (`OpusDecoder::last_packet_duration`).
2. A loss at the end of the stream is not decoded, as in `opus_demo`.
3. The decoder writes 24-bit integers, as libopus's `opus_decode24` does, and the loop rounds them to 16 bits as `opus_demo` does.
4. After each packet that arrived, when the packet before it also arrived, the decoder's final range must equal the one in the record. Otherwise the loop stops, as `opus_demo` does.

The `DemoDecoder` trait is implemented for `OpusDecoder`. The tests implement it for libopus as well (`tests/eval_common/libopus_demo.rs`), so the same loop drives both decoders. `tests/eval_demo.rs` checks the loop with both decoders: CELT, SILK, Hybrid and freely chosen modes, with and without FEC, with single losses and bursts, at 4 output configurations; a damaged final range; an invalid packet after a loss; losses before the first packet and at the end; and a cut bitstream. It also compares `packet::has_lbrr` with `opus_packet_has_lbrr` on 15,076 packets, and `last_packet_duration` with `OPUS_GET_LAST_PACKET_DURATION` over 27 steps at 3 rates. Streams without losses must give libopus's 16-bit output on every platform. Streams with losses are compared bit for bit only on 64-bit ARM macOS, and only reported elsewhere: against a libopus built at `-O2`, as on the phone, concealed CELT frames differ in their last bits (see [Results on the phone](#results-on-the-phone)).

### The conformance test of RFC 8251

The test vectors of RFC 8251 are 12 bitstreams in the format of `opus_demo` (`testvectorNN.bit`), with two sets of reference outputs: `testvectorNN.dec`, and `testvectorNNm.dec`, which is decoded without the phase inversion of stereo CELT. libopus's script `tests/run_vectors.sh` decodes each bitstream at 48, 24, 16, 12 and 8 kHz, in mono and in stereo, which makes 120 runs. It compares each output with both reference sets with `opus_compare`. A run passes when either comparison passes. The script then prints a mean quality per rate and channel count. For each reference set this is the mean over the 12 vectors, counted only when all 12 passed against that set. The script prints the larger of the two means.

`tests/eval_rfc8251.rs` does the same with opus1's decoder. It is stricter than the script: a final range mismatch, a failed decode call, a missing file or a cut bitstream fails the test. A second test decodes every vector with libopus through the same loop and requires opus1's 16-bit output to be identical. The command `cargo run --release --example eval -- conformance` prints the same table.

Results on 2026-10-18, the same on the Mac and on the phone:

- All 120 runs pass. The final range was checked about 200,000 times, with no mismatch and no failed decode call.
- All 120 outputs are identical to those of libopus 1.6.1 through the same loop.
- The mean quality, as `run_vectors.sh` computes it:

  | Output | 48 kHz | 24 kHz | 16 kHz | 12 kHz | 8 kHz | From |
  |---|---:|---:|---:|---:|---:|---|
  | mono | 97.4083 | 91.6250 | 84.7167 | 81.9500 | 73.2250 | `m.dec` |
  | stereo | 99.7000 | 91.9083 | 84.6500 | 82.1417 | 73.5167 | `.dec` |

- Mono output passes against `m.dec` in all 60 runs. Against `.dec` it fails in 20 runs (vectors 01, 05 and 06 at 48 and 24 kHz, 07 and 08 at every rate, 09 at every rate but 8 kHz). There the phase inversion of stereo CELT meets the mono mix. Stereo output passes against `.dec` in all 60 runs, and fails against `m.dec` in the 5 runs of vector 07. libopus's outputs are the same, so it shows the same pattern.
- The test takes 2.3 s on Apple M4 Pro and 30.8 s on the phone.

### Corpora

The corpora are not in the repository. The tools look for them under the directory that `OPUS1_CORPUS` names. Without it they look in `corpus/` at the root of the crate, which git ignores and the crate package leaves out. A worktree has its own crate root, so set `OPUS1_CORPUS` there. A test that needs a corpus prints "skipped" and passes when the corpus is missing.

| Corpus | Archive | Layout under the corpus root | Used by |
|---|---|---|---|
| The RFC 8251 test vectors | 74,624,664 bytes, from `https://opus-codec.org/docs/opus_testvectors-rfc8251.tar.gz` (libopus's `README`) | `rfc8251/opus_newvectors/testvectorNN.bit`, `.dec`, `m.dec` (the tools also accept `rfc8251/` itself, or `OPUS1_TESTVECTORS`) | the conformance test |
| The TSP speech database at 48 kHz, McGill University | 253,816,919 bytes; the page is `https://www.mmsp.ece.mcgill.ca/Documents/Data/` | `tsp48k/48k/<talker>/<file>.wav`: 1,444 files of English speech in 25 talker directories, with the simplified BSD licence in `48k/Licence.txt` | the `tsp` set of the harness |

`scripts/eval/fetch-corpus.sh` gets and checks them:

```text
scripts/eval/fetch-corpus.sh rfc8251
scripts/eval/fetch-corpus.sh tsp <URL of the 48 kHz archive>
scripts/eval/fetch-corpus.sh --verify rfc8251
```

- It asks before it downloads, unless `--yes` is given. It accepts `https:` and `file:` URLs only.
- It checks the size and the SHA-256 (Secure Hash Algorithm, 256 bits) sum of the archive. For the test vectors the sum is the one in libopus's `releases.sha2`. After unpacking, it also checks the SHA-1 sum of all 36 vector files against the list in `draft-ietf-codec-opus-update`, the draft of RFC 8251. The TSP archive has no published sum, so the script holds the size and sum of the download of 2026-10-18 and compares later downloads with them.
- It refuses archive entries with absolute paths or `..`, and it never runs anything from an archive.
- It writes the sums into `SHA256SUMS` in the corpus root. `--verify` checks a corpus that is already there.

`fixtures/tsp/` holds the first 2.5 s of four TSP files (`FA01_05`, `MD19_04`, `FJ55_02`, `MH43_04`; 960,176 bytes in all), with the licence text unchanged and a `README.txt`. The built-in `quick` and `selftest` sets use them, so these sets run without a corpus.

### The paired-comparison harness

The harness is the multi-file example `examples/eval/`. One job of the harness:

1. The item is a 16-bit reference at 48 kHz, mono or stereo. Its length is cut to a multiple of 12 samples, so that it divides into whole samples at every rate that `opus_compare` accepts.
2. The encoder input is the item at the condition's rate and channel count. Below 48 kHz it is resampled with a Kaiser-windowed sinc filter. The filter is flat within 0.1 dB up to 300 Hz below the lower Nyquist frequency, the band that `opus_compare` compares, and it attenuates by at least 90 dB above that frequency. A mono condition on a stereo item codes `(L+R)/2`, the mix that `opus_compare` compares with.
3. opus1's encoder and libopus's `opus_encode_float` code the input with the same settings. Before any job, `eval run` reads the settings back from both encoders and stops when they differ: application, rate, channels, bitrate, bitrate mode and its constraint, complexity, FEC, expected loss, forced channels, frame duration, prediction, look-ahead, and the controls that the harness leaves at their defaults (bandwidth limit, signal type, bit depth, discontinuous transmission, phase inversion). Nothing is forced unless the condition forces a coding mode or an audio bandwidth, which it then forces on both encoders. libopus codes a forced SILK mode as Hybrid when the bandwidth is wider than wideband, so a SILK condition at 24 or 48 kHz also forces wideband.
4. libopus's decoder decodes both packet streams with the `opus_demo` loop. `--decoder opus1` uses opus1's decoder instead. The loop compares the decoder's final range with the one of opus1's encoder after every packet whose predecessor arrived, so a packet that libopus reads differently from how opus1 wrote it stops the job. Conditions with FEC drop 10 % of the packets before decoding (fixed seed, the same packets on both sides).
5. Both encoders must report the same look-ahead. The first look-ahead samples of the output are dropped, and the rest is cut to the length of the input.
6. Each side is measured: `opus_compare` against the 48 kHz reference (error, quality, worst frame); the signal-to-noise ratio against the encoder input; the time shift within 1 ms that correlates best with the input; for speech in mono, MOC with and without the activity detection, after both signals are resampled to 16 kHz; bytes, kilobits per second, and how often each configuration of the table-of-contents byte occurs.
7. The pair gives `ratio_err = (err_opus1 + 1e-6) / (err_libopus + 1e-6)`, where a value below 1 means that opus1's output is closer to the reference. It also gives `delta_snr_db` (opus1's SNR minus libopus's), `ratio_bytes` (opus1's bytes divided by libopus's), and `moc_d`, the D of the OSCE draft with libopus as the reference decoder.

The alignment by the look-ahead was measured on the TSP excerpts, music and pink noise. CELT-only streams of the `Audio` and `RestrictedLowDelay` applications are best aligned at a shift of 0, for both encoders, at each rate measured (8, 16, 24 and 48 kHz). In `Voip` the high-pass filter shifts the phase of the low frequencies, and SILK output at 24 and 48 kHz comes 1 or 2 samples early with any application. Both encoders show the same shift there, and `opus_compare`'s 10 ms windows hardly notice it.

The built-in sets (`examples/eval/sets.rs`):

| Set | Items | Conditions | Jobs | Time on M4 Pro |
|---|---|---|---:|---:|
| `quick` (default) | 11 in the repository: the 4 TSP excerpts, speech with pauses, quiet speech, music, castanets, tones, pink noise, vowels | 30 in 6 groups: `silk_low`, `speech_mid`, `celt_mid`, `celt_high`, `lowdelay`, `switch` | 239 | about 4 s |
| `selftest` | 4 | 6 | 18 | 1.3 s |
| `tsp` | 222 TSP files, 9 per talker (`examples/eval/sets/tsp_speech.txt`) | 40: `Voip`, mono, 48 kHz, 6 to 128 kb/s, 20 and 60 ms, complexity 5 and 10, in the groups `tsp_low`, `tsp_mid` and `tsp_high` | 8,880 | 169 to 212 s |
| `tsp_rates` | the items of `tsp` | 64: `Voip`, mono, 16 and 24 kHz, 6 to 64 kb/s, 20 and 60 ms, complexity 5 and 10, in the same three groups. A control for changes meant for 48 kHz only. | 14,208 | 227 to 240 s |
| `tsp_stereo` | 111 stereo items: the items of `tsp` in pairs, one on each channel, cut to the shorter one | 12 in the group `tsp_stereo`: `Voip`, stereo, 16, 24 and 48 kHz, 20 ms. Complexity 5 and 9 with the bitrate schedule of the mode-switching streams of `perf_harness` (8 to 128 kb/s, one step every 200 ms), and complexity 5 at 24 and 32 kb/s | 1,332 | 9.2 s |
| `music` | 11 synthetic stereo items of 6 s | 16: `Audio`, stereo, 48 kHz, 64 to 160 kb/s, 20 and 60 ms, complexity 5 and 10, in the group `music_high` | 176 | |
| `baseline` | `tsp` and `music` | 56 | 9,056 | 171.8 s |

The synthetic items are not perceptually representative. They show sensitivity and regressions only. A condition at complexity 7 or more and at 16 kHz or more carries the tag `analysis`: there libopus runs its speech and music analysis, which opus1 does not have. An item belongs to the hold-out split when the FNV-1a hash of its name is a multiple of 4. The reports give the tuning and the hold-out items separately. Work that tunes the encoder should look at the tuning items only.

### Gates

A gate is a rule with a limit for one group of conditions. `eval list` prints the gates of each group.

| Gate | Rule | Default limit |
|---|---|---|
| `geomean_err` | The geometric mean of `ratio_err` at each bitrate of the group. | at most 1.20; 1.35 for `silk_low`, `speech_mid`, `tsp_low` and `tsp_mid` |
| `max_err` | The largest `ratio_err` of any job. | at most 1.50; 2.00 for the same four groups |
| `bytes` | The largest `ratio_bytes` of the jobs with a variable bitrate and frames below 40 ms. | at most 1.02; 1.05 (`speech_mid`, `tsp_mid`); 1.10 (`silk_low`, `tsp_low`) |
| `bytes_long` | The same for frames of 40 ms and more, where opus1 differs on purpose (see [Differences from libopus](10-differences-from-libopus.md#deliberate-differences)). | at most 1.10 |
| `bytes_min` | The smallest `ratio_bytes` of the jobs with a variable bitrate. | at least 0.95; 0.90 for the same four groups |
| `bytes_cbr` | The largest `abs(ratio_bytes - 1)` of the jobs with a constant bitrate. | 0 |
| `moc_clip` | The smallest `moc_d` (speech only). | at least -0.1 |
| `moc_mean` | The mean `moc_d` (speech only). | at least -0.025 |
| `align` | Jobs where opus1's best shift is not 0 while libopus's is (CELT, music, low delay). | 0 |
| `delta_snr` | The mean `delta_snr_db` (CELT, music, low delay, `tsp_high`). | at least -1 dB |
| `baseline_geomean` | With `--baseline`: the geometric mean of opus1's `err` now divided by its `err` in the stored run. | at most 1.02 |
| `baseline_max` | With `--baseline`: the largest such ratio. | at most 1.10 |
| `libopus_changed` | With `--baseline`: jobs whose libopus `err` differs from the stored one. A comparison with a stored run is valid only when this is 0, that is with the same libopus build. | 0 |
| `baseline_snr` | With `--baseline`, for groups with `delta_snr`: the mean change of opus1's SNR. | at least -0.5 dB |

The error limits start from values that the research notes behind these tools proposed. Nobody calibrated them by listening. The MOC limits are those of the OSCE draft's relative test. Where opus1 exceeds a default limit, the list `CALIBRATED` in `examples/eval/sets.rs` sets the limit about 10 % above the measured value. The list was measured again at commit `c7744f6`, after the speech and music analysis and the fix of the low-frequency burst at 48 kHz VoIP. It now has 12 entries, down from 28: the error limits of the `speech_mid`, `tsp_mid` and `tsp_high` geometric means are no longer needed, because opus1 now passes the defaults there. Each entry carries the measured value in a comment. These limits warn when a gap grows. They are not quality targets. The results of that run are in `target/perf/eval/final_c7744f6/` (not in git).

Gates only warn. `eval run` exits with 1 when a job fails, for example on a final range mismatch, a failed decode call or a different look-ahead. With `--strict` it also exits with 1 when a gate fails. In `cargo test` the gates of the `quick` set only warn too. The conformance test is the one quality check that makes the test suite fail.

### The self-test

The self-test shows that the gates see a degradation. It runs on the `selftest` set (6 conditions, 4 items). It runs every job once without a degradation, and then:

- Two identity checks: libopus against libopus and opus1 against opus1. Each pair must be exactly neutral (`ratio_err` and `ratio_bytes` 1, `delta_snr_db` and `moc_d` 0), and pass every gate.
- Seven degradations of the opus1 side, applied from outside the codec. Each must make at least one gate fail that passes without it, in a group it applies to. The gates of `--baseline` count, with the undegraded run as the stored run.

| Degradation | What it does | Gates it trips | For example |
|---|---|---:|---|
| `lowpass` | A 4th-order Butterworth low-pass on the decoded signal, run forward and backward: 7 kHz at 24 and 48 kHz, 3.5 kHz at 12 and 16 kHz, 2 kHz at 8 kHz. | 8 | `celt_high delta_snr` -4.15 dB; `silk_low moc_mean` -0.206 |
| `noise:30`, `noise:20` | White noise 30 or 20 dB below the level of the decoded signal. | 17, 25 | `celt_high geomean_err 96000` 2.23; `silk_low max_err` 7.31 |
| `shift:1ms` | The decoded signal is delayed by 1 ms. | 12 | `celt_mid align` 4 jobs; `celt_mid delta_snr` -16.4 dB |
| `underspend:0.75` | opus1 codes at 75 % of the bitrate. | 19 | `silk_low bytes_min` 0.74; `celt_mid bytes_cbr` 0.25 |
| `overspend:1.10` | opus1 codes at 110 % of the bitrate. | 4 | `celt_high bytes` 1.101; `celt_mid bytes_cbr` 0.10 |
| `alloc:mask` | CELT with a constant bitrate only: an energy mask of -2.0 on bands 0 to 10 and +0.25 on bands 11 to 20. The packets keep their size, only the allocation of bits to bands changes. | 3 | `celt_mid baseline_geomean` 1.0222; `baseline_snr` -0.86 dB |

What the self-test shows about the gates:

- `opus_compare` hardly sees the low-pass: the geometric mean of `ratio_err` over the 18 jobs only moves from 1.1503 to 1.1513. Removing the high band of CELT also removes its coding noise. The gates of the signal-to-noise ratio and of MOC catch it.
- The energy mask raises `err` by only 2.2 %. Only the gates of `--baseline` catch it, and by small margins. With their limits loosened to 1.03, 1.11 and -1.0 dB, it goes unseen and the self-test fails. Loosening the bytes limits by 0.15 lets `overspend:1.10` through in the same way. Without the MOC gates, or without the gates of the signal-to-noise ratio and the alignment, every degradation is still caught by another gate.
- `tests/eval_selftest.rs` also evaluates the same runs with every limit out of reach, and requires the self-test to fail. It checks as well that a gate that fails without a degradation is not counted for any degradation.

### Running the harness

```text
cargo run --release --example eval -- list [--set NAME]
cargo run --release --example eval -- run [--set NAME] [--json PATH] [--md PATH] [--text PATH]
    [--baseline JSON] [--degrade KIND] [--write-wav DIR] [--decoder libopus|opus1]
    [--filter TEXT] [--items FILE] [--item-filter TEXT] [--max-items N] [--corpus DIR]
    [--threads N] [--strict]
cargo run --release --example eval -- selftest
cargo run --release --example eval -- conformance [--vectors DIR]
cargo run --release --example eval -- moc REF.wav TEST.wav [--apply-vad]
cargo run --release --example eval -- corpus-index --root DIR --out FILE [--per-group N] [--class CLASS]
cargo run --release --example eval -- merge RUN.json TIER2.jsonl [--json PATH]
```

- `run` prints a text report: per condition the geometric mean of `ratio_err` with its 95 % bootstrap interval, the largest ratio and its item, the bytes ratio, `delta_snr_db`, the mean `moc_d`, and libopus's error. Then the gates and the worst jobs.
- `--json` writes one result row per line, with the settings of each condition, the git commit, the libopus version and library directory, and a hash of the items. `--baseline` reads such a file and adds the gates of a comparison with it. `--md` writes one Markdown table per group.
- `--filter` keeps the conditions whose name or group contains one of its comma-separated parts.
- `corpus-index` writes a set file from a directory of WAV files: one item per line, `path class start duration`, with N files per directory, evenly spaced.
- `--write-wav DIR` writes, for each job, the reference, the input and both decoded signals, plus 16 kHz versions of speech in mono, and a `manifest.jsonl`. `scripts/eval/tier2.py DIR/manifest.jsonl` runs the external metrics that are installed (ViSQOL, PESQ wideband, SCOREQ, PLCMOS) and installs nothing. `merge` then adds their paired differences, with bootstrap intervals, and lists the items where a metric and `opus_compare` disagree on which side is better.

The run of the `baseline` set on 2026-10-18 (233 items, 9,056 jobs, no job error, 171.8 s on 12 threads; at commit `5e07998`, whose codec is that of `d541821`) is kept outside git in `target/perf/eval/baseline_d541821.json`, with its text and Markdown reports. [Differences from libopus](10-differences-from-libopus.md#paired-comparison-with-the-evaluation-harness) summarizes what it found.

### The evaluation tests

| Test file | Tests | Needs |
|---|---:|---|
| `eval_opus_compare.rs` | 5 | nothing; the C program for its live part |
| `eval_moc.rs` | 3 | nothing; Python with numpy for its live part |
| `eval_demo.rs` | 7 | nothing |
| `eval_rfc8251.rs` | 2 | the RFC 8251 vectors |
| `eval_selftest.rs` | 9 | nothing: the `quick` and `selftest` sets use the fixtures |

`eval_selftest.rs` runs the self-test, the `quick` set (whose gates only warn), the settings check over the 156 conditions of the built-in sets, the resampler's pass band and stop band, the items and degradations, the statistics, the set files and the merge of external metrics. It takes about 5 s on Apple M4 Pro. `OPUS1_EVAL_RECORD=1` lets `eval_opus_compare.rs` and `eval_moc.rs` rewrite their `.expected` files from the C program or the Python script. Do this on 64-bit ARM macOS only, where the values are exact.

## Checks of the speech and music analysis and of DTX

The analysis (`src/analysis/`) is compared with libopus at three levels. The Homebrew library is static, and the analysis functions of libopus are not `static`, so the tests call them directly: `tonality_analysis_init`, `run_analysis`, `tonality_get_info`, `analysis_compute_dense`, `analysis_compute_gru` and the weight tables `layer0` to `layer2`. `tests/analysis_common/mod.rs` declares them, with `#[repr(C)]` copies of `TonalityAnalysisState` (14,004 bytes) and `AnalysisInfo` (64 bytes) whose sizes the tests check.

| File | Tests | What it checks |
|---|---:|---|
| `analysis_libopus.rs` | 13 | `TonalityAnalysisState::run` against `run_analysis`: every `AnalysisInfo` and the whole C state after every call, bit for bit. 16, 24 and 48 kHz; frames of 2.5 to 120 ms; mono, stereo and the channel layouts of the multistream encoder up to 8 channels; float and 16-bit input; input precision 8 to 24 bits; buffers longer than the coded frame; NaN, infinities, huge and subnormal input; band energies of 1e9 and more; resets; 205 s of audio, which drives the block counter to its limit. |
| `analysis_mlp_libopus.rs` | 5 | The weight tables equal the C arrays byte for byte. Each layer of the network equals the C layer on 200,000 random inputs and over a run of 10,000 steps; random int8 layers and a sweep of the activation functions too. |
| `analysis_properties.rs` | 3 | Property tests (`proptest`, 1000 cases each): any `f32` input and call pattern gives no panic and no allocation, valid results have no NaN and stay in range, 40 and 60 ms calls leave the state of 20 ms calls, and `reset` gives the state of `new`. |
| `celtenc_analysis_info_libopus.rs` | 4 | The CELT encoder given an `AnalysisInfo`, against libopus's CELT encoder given the same values (`CELT_SET_ANALYSIS`): 2,328 streams and 464,700 packets, all byte for byte, from 8 to 48 kHz, 2.5 to 20 ms, CBR and VBR, complexity 0 to 10, with opus1's own analysis and with random and extreme values. Another test shows that each field of `AnalysisInfo` changes libopus's packets. |
| `core_analysis_libopus.rs` | 12 | `OpusEncoder` with the analysis against `opus_encode_float` and `opus_encode`: CELT-only packets at complexity 7 to 10 byte for byte in 948 streams (16 to 48 kHz, 2.5 to 60 ms, the three applications, the three signal types, 6 to 510 kb/s, float and 16-bit input); the `AnalysisInfo` of every frame against C `run_analysis`; expert frame durations; complexity changes; `reset_state`; the first bytes of streams where the encoder chooses the mode. One more test, ignored, prints the SNR of SILK and Hybrid streams against libopus. |
| `core_analysis_walk_libopus.rs` | 3 | `OpusEncoder` against libopus while one setting changes every 2 to 20 packets: bitrate (with the automatic and the maximum bitrate), complexity across 7, signal type, bandwidth, maximum bandwidth, CBR/VBR/CVBR, input precision, packet loss and FEC, prediction, forced channels, DTX, expert frame duration, frame size from 2.5 to 120 ms, float or 16-bit input, output buffers of 1 to 12 bytes, and `reset_state`. 90 CELT-only streams at 8 to 48 kHz (54,000 packets): every packet, final range and `in_dtx` is libopus's. Mono streams that switch between CELT-only, SILK-only and Hybrid, forced or chosen by the analysis: every CELT-only packet and every redundant CELT frame of a switch is libopus's. |
| `multistream_analysis_libopus.rs` | 4 | The analysis of each stream of the multistream encoder against `run_analysis` with the stream's channels, and CELT-only multistream packets byte for byte for every layout of the test matrix. |
| `multistream_analysis_walk_libopus.rs` | 1 | `MultistreamEncoder` with every stream CELT-only while one setting changes every few packets, for every layout of the test matrix at 16, 24 and 48 kHz: every packet and final range is libopus's (33 streams, 9,900 packets). |
| `core_dtx_libopus.rs` | 14 | DTX of `OpusEncoder` against libopus: the 1-byte packets, `in_dtx` and the CBR sizes in the same places. CELT-only DTX in 840 streams (920,530 packets) byte for byte; streams with SILK and Hybrid; packets of several frames; decoding of opus1's DTX streams. |
| `multistream_dtx_libopus.rs` | 4 | DTX of `MultistreamEncoder` against libopus for every layout of the test matrix. |

`core_fixed_point.rs` also compares six streams with DTX decided by SILK with the fixed-point build: every packet and `in_dtx`, for 20 to 120 ms, mono and stereo, VBR and CBR, complexity 0 to 9. The tests that compare with the fixed-point build run opus1's analysis at complexity 10 only, as that build does (`set_analysis_gate(AnalysisGate::FixedPointBuild)`).

Each kind of difference these tests guard was put into a copy of the code once, and the tests failed. For example: a sum of the band tonality without the fused rounding of the Homebrew build, the downmix without its fused sum, the input precision of 16-bit input, the read position of packets of several frames, the speech estimate kept across a reset, the symmetric distance cache of the spectral variability, and the CBR padding of packets whose frames were all left out by DTX.

The NEON code of the analysis gives the bits of the scalar code. Its unit tests in `src/analysis/neon.rs` compare the two on random input, on the special values where the arithmetic changes branch, and on real blocks, and the libopus comparisons above run the NEON code.

## Fuzzing

`fuzz/` is a separate package (`opus1-fuzz`) of 24 targets for libFuzzer, the coverage-guided fuzzing engine of LLVM. libFuzzer calls a target with generated inputs. It keeps the inputs that reach new code and changes them further. The targets build and run on the stable toolchain:

```text
fuzz/fuzz.sh list                       # the targets
fuzz/fuzz.sh fuzz_decoder_sequence 600  # fuzz one target for 600 seconds
fuzz/fuzz.sh all 300                    # every target, 300 seconds each
```

`fuzz/fuzz.sh` builds the targets with cargo and the coverage options that cargo-fuzz uses without a sanitizer. Debug assertions and overflow checks are on, so an integer overflow stops a run like a panic. The corpus of each target is kept in `fuzz/corpus/<target>/`. An input that fails is written to `fuzz/artifacts/<target>/`. `fuzz/README.md` gives the build command, how to reproduce and minimize a failure, and the `cargo +nightly fuzz` command, which still works.

Eight targets compare opus1 with libopus, which is linked as for the tests:

| Target | What must hold |
|---|---|
| `fuzz_packet` | The `packet` module gives libopus's results: `parse` (plain and self-delimited), `nb_frames` and `nb_samples` against `opus_packet_parse_impl` and `opus_packet_get_nb_*`, `pad` and `multistream_pad` against `opus_packet_pad` and `opus_multistream_packet_pad` byte for byte, and `write` against the libopus repacketizer byte for byte. `multistream_validate` accepts what libopus's multistream decoder accepts. Written and padded packets parse to the frames they were made of. |
| `fuzz_decoder_sequence` | A stream of packets into `OpusDecoder` and into libopus's decoder. The packets come from a libopus encoder whose settings change (2.5 to 120 ms, FEC, discontinuous transmission, forced modes and bandwidths), or from the input. Some are damaged or lost. The calls are `decode`, `decode_i16`, `decode_fec` and `decode_fec_i16`, on decoders of either channel count. After every call both decoders accept or refuse alike, return the same number of samples, and end in the same final range. While no packet was lost or damaged, the output is within 50 dB of libopus's. opus1's output buffer sometimes has more or less room than the frame size. More room changes nothing. With too little room opus1 may refuse, and then it keeps its state. |
| `fuzz_multistream_decoder` | A layout and packets into `MultistreamDecoder` and into libopus's `opus_multistream_decoder`. Both accept or refuse the layout alike. After every call they accept or refuse alike, return the same number of samples, and end in the same final range. While no packet was lost or damaged, the output is within 50 dB of libopus's. opus1's output buffer sometimes has more or less room than the frame size. More room changes nothing. With too little room opus1 may refuse, and then it keeps its state. A decoder built from the layout as the channel mapping table of an Ogg Opus header is accepted too. opus1 writes only the samples it returns, with `f32` and with 16-bit output: the rest of its buffer keeps a marker value. |
| `fuzz_multistream_encoder` | `MultistreamEncoder` with a layout, settings and audio from the input. libopus accepts and refuses the same layouts, and picks the same streams and mapping for a mapping family. Every packet decodes in libopus's multistream decoder to one frame, in the encoder's final range. Until a frame is concealed, opus1's decoder gives output within 50 dB of libopus's. A second encoder with the same settings and an input precision of 16 bits (`set_lsb_depth(16)`), given `f32` samples where the first gets the same audio as `i16`, writes the same packets. DTX is switched on and off between frames. One sample too few is refused, and the encoder keeps its state. With a constant bitrate that is explicit or the maximum, a packet has the length of libopus's rule (`opus_multistream_encoder.c` lines 918-928): the bytes of the bitrate, at least the smallest packet and at most the buffer, or the whole buffer at the maximum bitrate. |
| `fuzz_encoder_settings` | `OpusEncoder` with settings that change between frames: bitrate (also automatic and maximum), variable or constant bitrate and the constraint of variable bitrate, complexity, forced mode, bandwidth, forced channel count, FEC, loss percentage, DTX, the low-frequency-effects (LFE) flag, energy mask, prediction and frame size. A value that libopus refuses, and that opus1's types can hold, is refused, and the setting stays: a complexity above 10, a loss percentage above 100 and an energy mask of the wrong length. The one exception is a forced stereo channel count on a mono encoder, which opus1 keeps without effect. Every valid frame is encoded. With a constant bitrate every packet has the length of libopus's rule in `opus_encode_native` (`cbr_packet_len` in `fuzz/fuzz_targets/common/mod.rs`). Every packet decodes in libopus's decoder to one frame, in the encoder's final range. Until a frame is concealed, opus1's decoder gives output within 50 dB of libopus's. A second encoder with the same settings and an input precision of 16 bits, given `f32` samples where the first gets the same audio as `i16`, writes the same packets. One sample too few is refused, and the encoder keeps its state. |
| `fuzz_channel_order` | A stereo `OpusEncoder`, or a `MultistreamEncoder` with the layout of a mapping family or with channels, streams and a mapping, gets a tone of its own in every input channel. libopus's encoder codes the same tones with the same settings, and libopus's decoders decode both. For each output channel the check measures the strongest tone of another input channel, relative to its level at the input (the leakage). Wherever libopus's leakage is -20 dB or less, opus1's must be below -6 dB. A channel on the wrong output gives a leakage of about 0 dB. Stereo streams that an encoder codes as mono, and those that SILK codes with a constant bitrate, are left out, see below. |
| `fuzz_extensions` | The `extensions` module against libopus's `src/extensions.c`. The input is the padding of a packet of 1 to 48 frames, or a list of extensions, some invalid and some repeated in every frame. `count`, `count_ext`, `parse` (also into a list that is too short), `parse_ext`, the `ExtensionIterator` (`next`, `find_id`, `set_frame_max`, `reset`), `generate` and `generated_len` give libopus's extensions or bytes, or its error. After an error the iterator ends, where libopus returns the error again. Extensions that are written fit the buffer, fill it with `Padding::FillBuffer`, and read back as the same extensions in frame order. |
| `fuzz_repacketizer` | `Repacketizer` and the padding functions of `packet` on packets whose padding holds extensions, against libopus's `repacketizer.c`. Up to 4 packets from the input (raw, written by `packet::write`, or code 3 packets whose padding is raw bytes or extensions), then up to 12 steps: `cat`, `reset`, `out` and `out_range_ext` (random range, buffer length, framing, padding and added extensions), `pad`, `pad_with_extensions` and `unpad` of one packet, and `multistream_pad` and `multistream_unpad`. Each gives libopus's result and bytes; when `len == new_len`, `pad_with_extensions` writes the extensions it is given, where libopus leaves them out. A written packet fits its buffer and parses to the frames it was made of. |

The output is compared only while both decoders have decoded the same packets in full. opus1's decoder differs from libopus's only in rounding, so 50 dB leaves a wide margin. After a lost or damaged packet the output is no longer compared. The concealment searches the decoded signal for its pitch, and a rounding difference can change what it finds. A frame of 0 or 1 bytes is concealed as well, so it also ends the comparison. So does a packet that switches from CELT-only to SILK or Hybrid. Unless such a packet carries a redundant CELT frame, the decoder conceals 5 ms in CELT and fades them into the new frame. The targets do not look for the redundant frame, so they stop comparing after every such switch. The fuzzer found a switch without a redundant frame at 8 kHz. On the CELT packets before it, the two decoders agreed to 127 dB or better. On the 20 ms SILK frame of the switch they agreed to 33 dB. The difference lay in the first 5 ms of that frame, which hold the concealed CELT audio. There the decoders agreed to 27 dB, and on the other 15 ms to 294 dB.

The other 16 targets run opus1 alone. Thirteen of them (`fuzz_encoder`, `fuzz_decoder`, `fuzz_roundtrip`, `fuzz_silk_encoder`, `fuzz_celt_encoder`, `fuzz_overflow`, `fuzz_celt_large_frame`, `fuzz_celt_decoder_boundary`, `fuzz_silk_decoder`, `fuzz_hybrid_mode`, `fuzz_encoder_boundary`, `fuzz_multi_frame`, `fuzz_packet_parsing`) use extreme settings, frame sizes and packets. Nothing may panic. Those that encode and decode also check that the decoder returns one frame and ends in the encoder's final range. The other three test the neural network features and need a feature of the fuzz package: `fuzz_dnn_blob` (feature `dnn`) loads weight blobs, `fuzz_dred` (feature `dred`) parses and decodes DRED data, and `fuzz_decoder_dnn` (feature `dred`) decodes with deep PLC, LACE, NoLACE, the bandwidth extension and DRED. `fuzz/README.md` describes what each target checks.

With the speech and music analysis and DTX, `fuzz_encoder_settings` and `fuzz_multistream_encoder` ran 150 s each (158,650 and 52,775 inputs) without a failure.

When fuzzing finds a failure in opus1, the fix follows what libopus does. The minimized input becomes a test in `tests/fuzz_regressions.rs`, as a byte array in the test, so the test needs no file. The runs below found no such failure, so that file does not exist yet.

Two files of the normal test suite also try many generated inputs. `tests/fuzz_toc_sequences.rs` feeds one decoder long, repeatable sequences of packets with varied TOC bytes. `tests/proptest_encode_indices.rs` uses the `proptest` crate, which generates random inputs and reduces a failing one to a small example.

### Fuzzing results for this release

The 18 targets in the table below ran on Apple M4 Pro with the build of `fuzz/fuzz.sh`, so with overflow checks and debug assertions. Several targets ran at the same time, one process each, unless the table says otherwise.

| Target | Time | Inputs run |
|---|---|---:|
| `fuzz_decoder_sequence` | 2 processes of 60 minutes | 1,302,529 |
| `fuzz_multistream_decoder` | 2 processes of 60 minutes | 1,540,925 |
| `fuzz_multistream_encoder` | 2 processes of 60 minutes | 405,544 |
| `fuzz_encoder_settings` | 2 processes of 60 minutes | 1,433,471 |
| `fuzz_packet` | 45 and 20 minutes | 404,488,807 |
| `fuzz_decoder` | 10 and 20 minutes | 22,145,096 |
| `fuzz_celt_encoder` | 10 and 20 minutes | 15,965,324 |
| `fuzz_roundtrip` | 10 and 20 minutes | 11,113,767 |
| `fuzz_encoder` | 10 and 20 minutes | 10,316,391 |
| `fuzz_celt_large_frame` | 10 and 20 minutes | 5,863,464 |
| `fuzz_overflow` | 10 and 20 minutes | 5,617,102 |
| `fuzz_silk_encoder` | 10 and 20 minutes | 4,233,591 |
| `fuzz_celt_decoder_boundary` | 10 and 20 minutes | 1,722,722 |
| `fuzz_multi_frame` | 10 and 20 minutes | 1,623,467 |
| `fuzz_hybrid_mode` | 10 and 20 minutes | 1,570,078 |
| `fuzz_silk_decoder` | 10 and 20 minutes | 778,344 |
| `fuzz_encoder_boundary` | 10 and 20 minutes | 39,352 |
| `fuzz_packet_parsing` | 10 and 20 minutes | 11,795 |

The table does not cover the other six targets: `fuzz_channel_order` (its runs are in [Checks added with the 0.2.1 interface](#checks-added-with-the-021-interface)), `fuzz_extensions`, `fuzz_repacketizer`, `fuzz_dnn_blob`, `fuzz_dred` and `fuzz_decoder_dnn`.

The first four targets also ran for about two hours of processor time each in earlier versions, before the checks of output, buffer room and short input were added. `fuzz_encoder_boundary` and `fuzz_packet_parsing` are slow by design: each input runs every rate, channel count and first byte.

A second check ran the five targets that compare with libopus once more, from an empty corpus and with a new seed. Each ran as one process for 15 minutes, all five at the same time:

| Target | Inputs run |
|---|---:|
| `fuzz_packet` | 88,909,063 |
| `fuzz_decoder_sequence` | 693,499 |
| `fuzz_multistream_decoder` | 909,403 |
| `fuzz_multistream_encoder` | 354,285 |
| `fuzz_encoder_settings` | 760,903 |

These runs found no failure.

What the runs found:

- No defect in opus1. No panic, no integer overflow, and no difference from libopus in accept or reject, in the number of samples or in the final range.
- One failure of the output check, in `fuzz_encoder_settings`: the switch from CELT-only to SILK described above. It is a rounding difference, so the check was changed, not the code.

The targets do find defects. Defects were put into a copy of the code on purpose, one at a time. Each run started from an empty corpus, with one process, and stopped at the first failure:

| Defect put in | Target | What failed | Inputs run |
|---|---|---|---:|
| `packet::write` counted one byte too many for the padding length when the padding is a multiple of 255 bytes. | `fuzz_packet` | a debug assertion of `write` | about 1,600 |
| `packet::parse` lost the check that all frames of a self-delimited constant-bitrate packet fit in it. | `fuzz_packet` | `split_self_delimited` and `parse` disagreed | about 3,500 |
| The decoder's final range left out the redundant CELT frame. | `fuzz_decoder_sequence` | final range | about 50,000 |
| `decode` took a packet longer than `frame_size` when the output buffer had room for it. | `fuzz_decoder_sequence` | accept or refuse | about 94,000 |
| `decode_fec` decoded the FEC data after a CELT-only packet. libopus conceals the frame then. | `fuzz_decoder_sequence` | final range | about 97,000 |
| `MultistreamDecoder` copied the left channel of a stereo stream to the right output channel. | `fuzz_multistream_decoder` | output | about 9,300 |
| `MultistreamDecoder` passed the FEC flag only to the mono streams. | `fuzz_multistream_decoder` | number of samples | about 16,000 |
| `MultistreamEncoder::final_range` left out the LFE stream. | `fuzz_multistream_encoder` | final range | about 6,600 |
| The encoder's final range left out the redundant CELT frame. | `fuzz_encoder_settings` | final range | about 74,000 |

Every run stopped within 20 seconds.

### Checks added with the 0.2.1 interface

Four checks came with the interface changes of 0.2.1:

1. `fuzz_multistream_decoder` fills opus1's output buffer with a marker value before each call. After a call, the samples behind the ones returned must still hold it, and the samples returned must fit the buffer. This holds for `f32` and for 16-bit output. Before, only the `f32` path checked the fit.
2. `fuzz_encoder_settings` compares the length of every constant-bitrate packet with libopus's rule (`cbr_packet_len` in `fuzz/fuzz_targets/common/mod.rs`, after `opus_encode_native`). It also requires that a complexity or loss percentage that libopus refuses is refused. No bitrate can be refused: `Bitrate::BitsPerSecond` holds a `NonZeroU32`, libopus refuses only 0 and negative numbers, and both limit the others to 500 to 750,000 b/s per channel. The automatic and the maximum bitrate and the constraint of the variable bitrate are among the settings it changes. Its output buffers have 1 to 12,000 bytes. Only a buffer of more than 7,500 bytes shows one part of the rule: at the maximum bitrate a 40 or 60 ms packet is padded to the whole buffer (opus_encoder.c:1749). With a smaller buffer the bitrate of the buffer is below the 1.5 Mb/s of the maximum bitrate, so the constant size is the buffer anyway.
3. `fuzz_multistream_encoder` compares the length of every constant-bitrate packet with libopus's rule for an explicit and for the maximum bitrate (`opus_multistream_encoder.c` lines 918-928). The rule for the automatic bitrate depends on the allocation, which the target does not know.
4. `fuzz_channel_order` is new. It is described in its own comment and in the table above.

Each check ran for 12 minutes as one process, the first three at the same time and `fuzz_channel_order` in its final form after them, on Apple M4 Pro while other work ran on the machine. None failed:

| Target | Inputs run |
|---|---:|
| `fuzz_encoder_settings` | 632,010 |
| `fuzz_multistream_encoder` | 338,192 |
| `fuzz_multistream_decoder` | 783,536 |
| `fuzz_channel_order` | 140,527 |

`fuzz_channel_order` failed four times while its check was being shaped, each time on a difference between the encoders that is not a channel order fault:

- A stereo stream that SILK coded with a constant bitrate at 12 kHz lost its stereo image in opus1. The fixed-point build of libopus, which opus1's SILK encoder ports, loses it too (built with `cc`, see [Checks against the fixed-point build](#checks-against-the-fixed-point-build-of-libopus)); the float build keeps it. Such streams are left out of the check.
- 7.1 sound at 24 kHz with the automatic bitrate: the encoders chose other bandwidths on purpose (D1 in [Differences from libopus](10-differences-from-libopus.md)). Surround layouts get an explicit bitrate since.
- A stereo stream that both encoders coded as mono. Such streams are left out.
- Three mono streams at 24 kHz in 2.5 ms frames at complexity 7: opus1 lost the tone of one channel where libopus kept it. libopus ran its speech and music analysis there, which opus1 did not have then. The check now measures how loud another channel's tone comes out, so a lost tone no longer counts.

Defects put into a copy of the code show that the new checks find what they look for. Each run started from a copy of the target's corpus from the runs above and failed while it replayed the saved inputs:

| Defect put in | Target | What failed | Time |
|---|---|---|---|
| A constant-bitrate packet of one stream took `(bits + 7) / 8` bytes instead of `(bits + 4) / 8`. | `fuzz_encoder_settings` | packet length | 3 s |
| The same rounding in the multistream encoder's constant-bitrate budget. | `fuzz_multistream_encoder` | packet length | 33 s |
| `MultistreamDecoder::decode_i16` wrote one sample behind those it returned. | `fuzz_multistream_decoder` | marker value | 1 s |
| The multistream encoder fed the left input channel of a coupled stream to its right channel and the right to its left. | `fuzz_channel_order` | another channel's tone at 0.1 dB of its input level | 6 s |
| SILK's stereo coding took the side signal as right minus left. | `fuzz_channel_order` | another channel's tone at 0.0 dB of its input level | 1 s |

A review of the checks put in five other defects, each into the code as it is in the repository, and fuzzed each target from an empty corpus directory with the saved inputs of the runs above as seeds. Every defect was found:

| Defect put in | Target | What failed | Time |
|---|---|---|---|
| `MultistreamDecoder::decode_fec_i16` wrote one sample behind those it returned. | `fuzz_multistream_decoder` | marker value | 0 s |
| A 40 or 60 ms packet at the maximum bitrate and a constant bitrate was padded to the size of 1.5 Mb/s instead of the whole buffer. | `fuzz_encoder_settings` | packet length (7,500 bytes for a buffer of 8,336) | 49 s |
| A packet too small to code audio was not padded to the constant size. | `fuzz_encoder_settings` | packet length | 2 s |
| The multistream encoder limited a packet at the maximum bitrate and a constant bitrate to the allocated bitrate instead of the buffer. | `fuzz_multistream_encoder` | packet length | 14 s |
| `OpusEncoder` swapped the left and the right channel of the input before CELT coded it. | `fuzz_channel_order` | another channel's tone at -4.7 dB of its input level | 0 s |

The second defect could not be found before the buffers went above 4,000 bytes, and `core_encoder_settings::maximum_bitrate_gives_the_packets_of_libopus` did not find it either before it got a buffer of 10,000 bytes.

The four targets then ran again without a defect, 10.5 minutes each, as four processes at the same time on Apple M4 Pro while other work ran on the machine. None failed:

| Target | Inputs run |
|---|---:|
| `fuzz_encoder_settings` (buffers up to 12,000 bytes) | 194,364 |
| `fuzz_multistream_encoder` | 109,976 |
| `fuzz_multistream_decoder` | 277,128 |
| `fuzz_channel_order` | 99,048 |

## `#![no_std]` checks

Without its default features, opus1 builds as `#![no_std]` with no heap allocation (see the [README](../../README.md)). `scripts/check_no_std.sh` checks this:

It does each step three times: with `--features libm`, with `--features libm,analysis`, which adds the speech and music analysis, and with `--features alloc,libm`, which adds the `multistream` module for a target that has a global allocator.

1. It builds with `--no-default-features` and those features for `thumbv7em-none-eabi` (ARM Cortex-M) and `wasm32-unknown-unknown` (WebAssembly). A target that is not installed is skipped with a message. Add it with `rustup target add <target>`.
2. It builds the same configuration for the host.
3. It runs `cargo test --no-default-features` with those features on the host with `RUST_MIN_STACK=33554432` (32 mebibytes). Without the `heap` feature, the encoder and decoder hold all their buffers inline: 127,936 bytes (153,904 with `analysis`) and 52,072 bytes on aarch64 (see [Size of the encoder and the decoder](06-using-the-library.md#size-of-the-encoder-and-the-decoder)). So test threads need a large stack.

Then it builds `--features heap,libm` (the codec state on the heap, without `std`) for the same targets. It runs the tests of building in place in a release build: `multistream_small_stack`, `encoder_init`, `decoder_init` and `encoder_reset`, with `alloc,libm`, `alloc,libm,analysis` and `heap,libm,analysis`. They build the codecs on a thread with a 32 KB stack, which only an optimized build can check (a debug build gives those threads 1 MiB), and check that a reset allocates nothing.

The multistream module needs the `alloc` feature, so the first two runs leave it out. The run with `alloc` compiles it and runs its unit tests. The multistream test files in `tests/` (36 of 37; the other one tests only libopus) and the multistream tests in other files are compiled only with the `std` feature, so every run leaves them out. At git commit `f952b46`, the run with `--features libm,analysis` passed 985 tests (26 ignored), in a debug build. On the branch `release/0.2.0-api`, the three runs passed 1007 tests (27 ignored) with `libm`, 1057 (28 ignored) with `libm,analysis` and 1064 (27 ignored) with `alloc,libm`. Tests that compare encoded packets with libopus at complexities where libopus runs its speech and music analysis need the `analysis` feature. Without `alloc`, the comparisons of packet extensions skip the calls that opus1 refuses with `AllocFail` (see [Differences from libopus](10-differences-from-libopus.md#packet-padding-extensions)). `tests/stack_usage_test.rs` checks the sizes for both builds: small structs with `heap`, large ones without it.

## Ignored tests

29 tests are marked `#[ignore]`, and one documentation example is not run.

- 23 print report tables or are tools, not checks: `celtenc_bands_libopus` (2), `celtenc_gap_report` (3), `celtenc_prefilter` (1), `core_analysis_libopus` (1, the SNR of SILK and Hybrid streams against libopus), `k24_enc_report` (15) and `perfk_rc_timing` (1).
- 3 unit tests inside `src/` measure the speed of one function and print it: `celt::deemph_segments_tests::deemphasis_timing`, `mdct::bit_exact_tests::transform_timing` and `pitch::tests::prefilter_timing`.
- 3 search for differences in the decoded output, against libopus, over many streams (`core_decode_search`: streams of libopus and opus1, the RFC 8251 test vectors, multistream streams). They take about 4 minutes. On 64-bit ARM they fail on any difference; the first needs the TSP corpus and the second the test vectors (see [Corpora](#corpora)). [Differences from libopus](10-differences-from-libopus.md#decoder) gives what they cover.
- No test is ignored because of a known gap. The two that were, both because of the missing speech and music analysis, pass since its port: `oracle_celt::impulse_48k_mono_64k_is_the_packet_of_libopus` and `multistream_enc_signals::average_rate_at_16_khz_and_for_voip_is_within_the_design_limits`. [Differences from libopus](10-differences-from-libopus.md#tests-that-track-these-gaps) gives their figures.
- With a neural network feature, `dnn_frecpe::software_frecpe_matches_the_instruction_on_all_inputs` checks all 2^32 inputs of the software FRECPE (about 10 s on Apple M4, 30 s on the Cortex-A76, 84 s on the Cortex-A55). It is ignored in the default run of those tests and not compiled without the features.
- With `test-libopus-dnn`, the `timing` tests of `dnn_plc_libopus.rs` and `dnn_plc_decoder_libopus.rs` print the time of deep PLC calls and of decode calls next to libopus's (see [Performance](08-performance.md#deep-plc-models)). The `timing` test of `dnn_osce_decoder_libopus.rs` prints the time of decode calls with LACE, NoLACE and the BWE (see [Using the library](06-using-the-library.md#lace-nolace-and-the-bandwidth-extension)).
- 1 documentation example is marked `rust,ignore`: the fields of `OpusEncoder` with and without the `heap` feature in [Architecture](07-architecture.md#feature-flags-and-where-state-is-stored). It shows part of a struct, so it cannot run on its own.

## Documentation examples

`src/lib.rs` includes `README.md` and each file of `docs/guide/` that has Rust examples. Each file becomes the documentation of a private item under `#[cfg(all(doctest, feature = "std"))]`. So `cargo test --doc` compiles and runs every `rust` block of these files. An example must be a complete program fragment that uses `unwrap` or `expect`. Hidden lines (lines that start with `# `) are not used, because GitHub would show them. The examples need the `std` feature, because some use the `multistream` module. They also run without `heap`: `cargo test --doc --no-default-features --features std`.

## Tests on an Android phone

The scripts in `scripts/android/` build the tests and the harness for 64-bit ARM Android (`aarch64-linux-android`). They copy the programs to a phone over USB and run them there. Each script explains its options at its top.

They were written on, and first run on, a HONOR CLK-AN00: MediaTek MT6833, Android 14, six Cortex-A55 cores at 2.0 GHz and two Cortex-A76 cores at 2.4 GHz. These cores have NEON, DotProd (dot product), RDM (rounding doubling multiply) and FP16 (16-bit floating point). They do not have FEAT_FCMA, the complex-number instructions. For this target the Rust compiler assumes only NEON. So the FFT runs its version without FEAT_FCMA, which `kiss_fft.rs` selects at run time.

### What the scripts need

- The Android NDK (Native Development Kit), named by `ANDROID_NDK_HOME`. NDK 27.3 was used.
- `cargo-ndk` and the Rust target: `rustup target add aarch64-linux-android`.
- `adb` (Android Debug Bridge), on `PATH` or named by `ADB`.
- A phone with USB debugging allowed. The scripts install nothing and change no setting. Everything goes to `/data/local/tmp/opus1` (`DEVICE_DIR`), where the shell user can write without root.

### The scripts

| Script | What it does |
|---|---|
| `build-libopus.sh --src DIR` | Builds the libopus source tree DIR as a static library with the NDK's clang (see [Which libopus source to build](#which-libopus-source-to-build)). It uses the settings that libopus's own `configure` picks for this target: `-O2`, NEON intrinsics presumed, run-time detection for the DotProd code. They were checked against what `configure` of libopus 1.6.1 writes. The script writes every flag to `target/android/libopus/build-flags.txt`. `--dnn` adds the neural network features (see [Builds with the neural network features](#builds-with-the-neural-network-features)). |
| `test.sh` | Builds every test program with `cargo ndk -t arm64-v8a test --release --no-run` against that libopus. It copies the programs and `fixtures/` to the phone and runs them one after the other. Then it prints one line per program and the totals. `--build-only` stops after the build, and `--no-build` runs the programs of the last build. So a phone that several people use is taken only while the tests run. |
| `golden.sh` | Writes the golden hashes on the phone. With `--check FILE` it also checks them against a golden file from another machine. |
| `bench.sh` | Runs `perf_harness bench`, pinned with `taskset` to one CPU of the fast cluster and one of the slow cluster (see [Performance](08-performance.md#on-an-android-phone)). |
| `profile.sh` | Counts hardware events per frame with `simpleperf stat` and samples the hot functions with `simpleperf record`, for one operation of the harness. |
| `build-variant.sh NAME` | Builds the harness with the compiler settings of the environment (for example `RUSTFLAGS`, `CARGO_PROFILE_RELEASE_LTO` or `PERF_HARNESS_PAD`) as the program `target/android/bin/perf_harness_NAME`. It records the settings and the address of the code section beside it. |
| `ab.sh BASE VARIANT...` | Runs such programs in turns on the phone, as one job: each pass runs every program once on each chosen CPU, and the order turns by one place from pass to pass. At the end it prints `perf_harness pairs` of BASE against each other program (see [Comparing two builds](08-performance.md#comparing-two-builds)). |

`env.sh` holds the settings that all scripts share. The commands of a first run:

```text
export ANDROID_NDK_HOME=$HOME/Library/Android/sdk/ndk/27.3.13750724
scripts/android/build-libopus.sh --src opus-1.6.1
scripts/android/test.sh
scripts/android/golden.sh --check target/perf/golden_r6.txt
scripts/android/bench.sh --runs 2
scripts/android/profile.sh --little celt_fb_48k_mono_64k_vbr_c10_20ms enc rust
```

To compare two builds, for example the default build and one for the Cortex-A55, in three passes on CPU 5 and CPU 7:

```text
scripts/android/build-variant.sh base
RUSTFLAGS="-C target-cpu=cortex-a55" scripts/android/build-variant.sh a55
scripts/android/ab.sh --passes 3 --cpus "5 7" base a55 -- \
  --filter silk_wb_16k_mono_20k_vbr_c10_20ms,celt_fb_48k_mono_64k_vbr_c10_20ms
```

`build-libopus.sh --host` builds the same libopus for the computer it runs on, when that is a 64-bit ARM computer such as a Mac with Apple silicon. Built from the release 1.6.1, it is the C reference at `-O2` of [The C reference](08-performance.md#the-c-reference).

Some details:

- The tests open `fixtures/answer_16k.wav` through a path that is fixed when they are compiled (`env!("CARGO_MANIFEST_DIR")`). While cargo compiles opus1, `rustc-wrapper.sh` sets that variable to `DEVICE_DIR`. The scripts copy `fixtures/` there.
- Each test program runs with `RUST_MIN_STACK=33554432`, as on macOS.
- The conformance test of RFC 8251 needs the test vectors on the phone. The default corpus root there is `corpus/` under `DEVICE_DIR`. Without the vectors the test passes with a "skipped" message (see [Corpora](#corpora)).
- The documentation tests do not run on the phone. Stable Rust cannot run them for another target.
- The programs run on the phone detached from adb (`setsid`). A dropped USB connection does not stop them. `test.sh --collect` waits for such a run and collects its logs.
- `test.sh` copies the logs of each run to a new directory, `target/android/test-logs/<date>-<time>/`, so a later run does not delete them. `test-logs/latest` points at the newest one.

### Which libopus source to build

`build-libopus.sh` builds the libopus source tree that `--src DIR` (or the variable `OPUS_SRC`) names. Use the release 1.6.1. The tests were written against it, and Homebrew ships it. Download it from Xiph and check it before you build it:

```text
curl -LO https://downloads.xiph.org/releases/opus/opus-1.6.1.tar.gz
shasum -a 256 opus-1.6.1.tar.gz
# 6ffcb593207be92584df15b32466ed64bbec99109f007c82205f0194572411a1, the sum
# that Xiph lists in SHA256SUMS.txt next to the release
tar xzf opus-1.6.1.tar.gz
scripts/android/build-libopus.sh --src opus-1.6.1
```

Homebrew's formula for libopus names the same file with the same SHA-256 (Secure Hash Algorithm, 256 bits) sum.

Without `--src` and `OPUS_SRC`, the script looks for the release at its default place, `target/perf/libopus-1.6.1-src/opus-1.6.1` under the repository root. From a git worktree it also looks under the main checkout (the parent of the git common directory). To put the release there, create `target/perf/libopus-1.6.1-src` and unpack the checked tarball into it (`tar xzf opus-1.6.1.tar.gz -C target/perf/libopus-1.6.1-src`). When that tree is missing too, the script falls back to the copy of libopus that the crate `opus-head-sys` 0.3.3 vendors, and it prints a warning. That copy is commit `228a0f85` of the libopus main branch, from 2026-08-28. It codes many SILK, Hybrid and LFE (low-frequency effects) streams differently from 1.6.1: in 86 of the 137 golden streams, libopus writes other packets. The phone runs up to round 7 linked this copy. On the phone, 11 tests more fail against it than against 1.6.1 (git commit `56ca2dd`). They compare packets with libopus: streams with an LFE channel, CBR with settings changed between packets, the rate limits and the switch of mode at a change of bitrate.

The version string of libopus shows which source a program links. `perf_harness` writes it into its records and into the header of a golden file: `libopus 1.6.1` for the release, `libopus git-228a0f855ce2` for the vendored copy.

The optimization level matters as well:

- `-O2` is the level of libopus's own build, and of the script. Use it for timings.
- At `-O2`, clang turns some loops of libopus into vector code. Their last bits can differ from those of the scalar code. The Homebrew library behaves like a build at `-Os` and has no such loops (see [The C reference](08-performance.md#the-c-reference)).
- So a comparison at the bit level with the results of the Homebrew build needs a build at `-Os` (`OPUS_EXTRA_CFLAGS=-Os`) or a build without vector loops (`OPUS_EXTRA_CFLAGS="-fno-vectorize -fno-slp-vectorize"`). [Results on the phone](#results-on-the-phone) lists the tests that see the difference.

`build-libopus.sh --host` builds the same way for a Mac with Apple silicon. There, `configure` builds no run-time CPU detection, and libopus's SILK encoder then calls the C version of its delayed-decision quantizer, as the Homebrew library does. `--rtcd` adds the detection, so the NEON version runs, as on the phone (see [The C reference](08-performance.md#the-c-reference)). The outputs are the same. Linking the harness or the tests against such a build is described at the top of the script.

### Builds with the neural network features

libopus 1.5 and later have decoder features that run small neural networks (DNN, deep neural network). A default build leaves them out. opus1 has the code they share, their weights and their decoder controls, and its decoder runs all of them:

- Decoder complexity 5 and higher turns on deep packet loss concealment (deep PLC).
- Complexity 6 turns on the speech enhancement LACE, and complexity 7 and higher NoLACE. Both are part of OSCE (Opus Speech Coding Enhancement), which works on SILK frames.
- `OPUS_SET_OSCE_BWE(1)` turns on the bandwidth extension (BWE) of OSCE. It extends 16 kHz SILK frames to 48 kHz output.

`build-libopus.sh --dnn` builds a libopus with these features. It adds the defines and sources that `./configure --enable-osce` of libopus 1.6.1 adds: `ENABLE_DEEP_PLC`, `ENABLE_OSCE` and `ENABLE_OSCE_BWE`. It also compiles the weights of the two networks of DRED (deep redundancy), so that tests can use them, but not DRED itself. DRED would change the encoder. The options `--deep-plc`, `--osce`, `--dred`, `--rdovae-weights` and `--dnn-weights-file` build single parts. The top of the script explains each option and gives the commands.

These four builds of the release 1.6.1 are the references for that work. All use run-time CPU detection, so the network code takes its DotProd (dot product) kernels on processors that have the instruction:

| Directory in `target/android/` | Computer | Compiler flags | Use |
|---|---|---|---|
| `libopus-host-1.6.1-dnn-novec` | Mac | `-O2 -fno-vectorize -fno-slp-vectorize` | The bit-exact reference for opus1's neural network code |
| `libopus-host-1.6.1-dnn` | Mac | `-O2` | Timings, and tests with a tolerance |
| `libopus-1.6.1-dnn-novec` | phone | `-O2 -fno-vectorize -fno-slp-vectorize` | The same reference on the phone |
| `libopus-1.6.1-dnn` | phone | `-O2` | Timings on the phone |

The port of DRED (deep redundancy, the cargo feature `dred`) has three more references, built with the same options and `--dred`: `libopus-host-1.6.1-dred-novec` (Mac, the bit-exact reference of DRED), `libopus-host-1.6.1-dred` (Mac, `-O2`) and `libopus-1.6.1-dred-novec` (phone). A small C program checked each one: a 16 kHz VoIP encoder at 32 kb/s with 20 % expected loss and `OPUS_SET_DRED_DURATION(100)` wrote the DRED extension (ID 126) into 99 of 100 packets, and `opus_dred_parse` read it from each of them.

With the decoder side of DRED (git branch `next-dred`), `cargo test --release --features test-libopus-dnn,dred,osce-bwe` linked against `libopus-host-1.6.1-dred-novec` passed all 1841 tests (33 ignored) on Apple M4 Pro. `dnn_osce_libopus.rs` needs the OSCE test shim of that reference: `scripts/dnn/build-osce-shim.sh target/android/libopus-host-1.6.1-dred-novec`. On the phone, `scripts/android/test.sh` with `CARGO_FEATURES=test-libopus-dnn,dred,osce-bwe` and `LIBOPUS_PREFIX` set to `libopus-1.6.1-dred-novec` passed the 14 tests of the three DRED test programs, and the other 23 programs whose names start with `dnn_`, `core_loss`, `core_fec` or `celtdec_plc` passed too, except `dnn_osce_libopus.rs`, whose shim was not built for that reference.

With the encoder side of DRED, the same command passed all 1863 tests (34 ignored) on Apple M4 Pro. On the phone, against `libopus-1.6.1-dred-novec`, the unit tests (`opus1`) and the five programs whose names start with `dnn_dred` passed all 473 tests (4 ignored), `dnn_dred_encode_libopus.rs` included: the DRED encoder gives libopus's packets there too. A later run of `opus1`, the `dnn_dred` programs, `core_packet`, `core_packet_extensions_libopus` and `core_limits` on the phone passed all 497 tests (4 ignored). With `--features dred` and DRED off, `perf_harness golden` gives the encoder fields (`enc`, `enc_i16`, `c_pk`) and `dec_rust` of `golden_r10.txt` in all 137 scenarios. It reports 63 scenarios as different, as with `--features deep-plc` alone: deep PLC changes the fields of decoding with losses (`dec_c_plc`, `dec_c_burst`, and `fec_silk` in 4 of them).

After the interface review for the first release (git branch `release/0.2.0-api`), the same command passed all 1941 tests (34 ignored) on Apple M4 Pro, with `OPUS1_CORPUS` set and the OSCE test shim built for `libopus-host-1.6.1-dred-novec`. Give each reference build its own `CARGO_TARGET_DIR`, for example `target/dnn-ref` for `libopus-host-1.6.1-dnn-novec` and `target/dnn-ref-dred` for `libopus-host-1.6.1-dred-novec`. The build script of `audiopus_sys` does not run again when `LIBOPUS_LIB_DIR` changes, so a target directory keeps the libopus it linked first. Linked with the `dnn-novec` build, which has no DRED, `dnn_dred_libopus.rs` then fails to link.

Why the builds without vector loops are the reference: at `-O2`, clang turns some loops into vector code that rounds products before it adds them. The networks turn such last-bit differences into large ones. A small C program decoded the same packets with each build (a 16 kb/s SILK wideband stream and a 32 kb/s CELT stream, with losses, at 12 settings of complexity, output rate and BWE):

- The Mac and the phone builds without vector loops gave the same output for all 12 settings, on the Cortex-A76 and on the Cortex-A55.
- The two `-O2` builds gave different output for 7 of the 12 settings: those with deep PLC, LACE or NoLACE. They agree at complexity 0 and 4, also with BWE at complexity 4.

Tests that call such a library are compiled only with the test-only cargo feature `test-libopus-dnn`. Their files start with `#![cfg(feature = "test-libopus-dnn")]`. They run in a cargo directory of their own, because cargo does not notice that another libopus is linked:

```text
LIBOPUS_LIB_DIR=$PWD/target/android/libopus-host-1.6.1-dnn-novec LIBOPUS_NO_PKG=1 \
  LIBOPUS_STATIC=1 CARGO_TARGET_DIR=target/dnn-ref \
  cargo test --release --features test-libopus-dnn,<model features> --test <name>
```

The whole suite of git commit `d541821`, linked against `libopus-host-1.6.1-dnn-novec`, gave 1425 passed, 75 failed and 28 ignored tests (in 28 test files). The 7 tests that fail against the `-O2` build of [Results on the phone](#results-on-the-phone) pass, as with every build without vector loops. The 75 failures came from two changes that the DNN defines make in libopus at every decoder complexity, also at 0. opus1 now makes both when the matching feature is compiled:

- With `ENABLE_DEEP_PLC`, `silk_PLC_glue_frames` skips its fade-in for 16 kHz frames (`silk/PLC.c` 477-480). So the first good 16 kHz SILK frame after a loss differs. This affects 71 tests.
- With `ENABLE_OSCE`, every 16 kHz SILK frame of 20 ms passes through `osce_enhance_frame`, which changes a sample of -32768 to -32767 (`dnn/osce.c` 1672-1679). The resampler and the stereo conversion spread the change to the neighbouring samples. This affects 12 tests, 4 of them only for this reason.

A build of the same source with both changes undone passed all 174 tests of the 28 files. On branch `next-wave2`, with deep PLC, LACE, NoLACE and the BWE in the decoder and both changes made, `cargo test --release --features test-libopus-dnn,osce-bwe` linked against `libopus-host-1.6.1-dnn-novec` passed all 1752 tests (35 ignored). A build with these features must be tested this way: linked against the Homebrew libopus, which has neither deep PLC nor OSCE, 84 tests fail.

With `--host`, the script also builds two programs that write weight blobs. A weight blob is the file format that `OPUS_SET_DNN_BLOB` loads, for a library built with `--dnn-weights-file`, which has no built-in weights:

- `bin/dump_weights_blob` is libopus's own `dnn/write_lpcnet_weights.c`. Its blob has every network except BBWENet, the network of the BWE. With that blob, `OPUS_SET_DNN_BLOB` returns 1, and LACE and NoLACE stay unloaded too. With BWE on, libopus 1.6.1 then aborts at the first SILK frame that it decodes at 48 kHz: a memory copy in `compute_generic_conv1d` runs past its buffer, which the check of `_FORTIFY_SOURCE` catches.
- `bin/dump_weights_blob_all` (`scripts/android/dump_weights_blob_all.c`) writes the same records followed by those of BBWENet. With its blob, the library decodes exactly as with built-in weights.

The blobs of both programs are in `target/android/dnn-blob-1.6.1/`, with a note of how they were made.

### Tests of the neural network code

The neural network code of `src/dnn/` (see [Architecture](07-architecture.md#neural-network-code-srcdnn)) has tests of its own. They need a model feature, for example `--features osce-bwe`, which includes all networks. The files whose names end in `_libopus`, and `dnn_freq_burg.rs`, also need `test-libopus-dnn` and a libopus built with `--dnn` (see above). Every comparison with libopus is bit for bit; two NaN values (not-a-number) count as equal.

| File | What it checks | Needs libopus with DNN |
|---|---|---|
| `dnn_kernels_libopus.rs` | The kernels and layer functions on random and edge inputs, against libopus's own functions (`compute_linear_c`, `compute_linear_dotprod`, `compute_activation_*`, `compute_conv2d_*`, `compute_generic_dense`, `_gru`, `_conv1d`, `_conv1d_dilation`, `compute_glu`), for every `DnnArch` the processor runs. The edge cases of `dnn_edges.rs` (sparse tables, int8 extremes, special inputs, odd GRU and GLU sizes) are compared with libopus too. | yes |
| `dnn_layers_libopus.rs` | Every layer of the six networks of 1.6.1 (PitchDNN, PLC, FARGAN, LACE, NoLACE, BBWENet), with the real weights on both sides and inputs from the speech of `fixtures/answer_16k.wav`. Each network runs as libopus runs it, call by call: activations, input sizes, convolution memories, recurrent states. Check 1 compares every call, passing libopus's result on. Check 2 runs each side on its own outputs and compares every layer output. The pitch computed from PitchDNN's output must equal the pitch of libopus's own `compute_pitchdnn`. | yes |
| `dnn_weights_libopus.rs` | The blobs of `dnn-weights/` against libopus's built-in arrays, record by record; their concatenation against the blobs that libopus's `dump_weights_blob` writes; each network's layers against those of libopus's `init_*` functions; the parser against libopus's `parse_weights` on about 90 changed blobs. | yes |
| `dnn_dred_libopus.rs` | DRED against a libopus built with `--dred`: the tables of DRED's quantizer against those of `dnn/dred_rdovae_stats_data.c`. It needs the features `test-libopus-dnn,dred` and the library `libopus-host-1.6.1-dred-novec`. | yes, with DRED |
| `dnn_dred_decode_libopus.rs` | The DRED decoder against a libopus built with `--dred`, exactly. `opus_dred_parse` and `opus_dred_process` (in place and into another state, at once and deferred) on the packets of libopus's encoder with DRED (6 settings: 16, 24 and 48 kHz, mono and stereo, 10 to 60 ms packets, 20 to 96 kb/s, complexity 3 to 10, 15 to 50 % expected loss, DRED durations 20 to 104): the result, `dred_end`, the latent vectors, the initial state and every feature. The same on cut, damaged and random DRED payloads and on 2,000 random packets. `opus_decoder_dred_decode`, `_float` and `24` for bursts of 1 to 50 lost packets rebuilt from the packet after the burst (as `opus_demo` does), at 16 and 48 kHz, mono and stereo, complexity 0 to 10, and at random positions with lengths from 2.5 ms to 1.1 s; every sample of the losses and of the packets after them. It needs the features `test-libopus-dnn,dred,osce-bwe` and `libopus-host-1.6.1-dred-novec`. | yes |
| `dnn_dred_decoder.rs` | The DRED decoder without libopus: random DRED payloads, frame numbers, rates, offsets and lengths never panic and give finite features and audio; the argument rules of `parse`, `process` and `dred_decode`; a DRED state that is not processed conceals as an empty packet; loading the RDOVAE decoder from a blob; parsing, processing and the three decode calls allocate nothing (losses of 200 ms included); `OpusDredDecoder` and `OpusDred` are `Send` and `Sync`, and `OpusDred` is 11,236 bytes. | no |
| `dnn_dred_encode_libopus.rs` | The DRED encoder against a libopus built with `--dred`. CELT-only packets with DRED, byte for byte: 48 kHz mono and stereo, 2.5 to 20 ms, VBR and CBR, with and without FEC, 24 to 128 kb/s, 5 to 50 % loss, DRED of 10 to 104 frames, complexity 0 to 10 (at 7 to 10 at least 99 % equal, because of the speech and music analysis); 40 and 60 ms with CBR; 40, 60 and 120 ms at 510 kb/s with CBR; 100 ms at the largest bitrate with CBR into 20,000 bytes; at the largest bitrate with VBR, 40 ms, 80 and 120 ms with DTX leaving out the first frames of a packet (in some packet the first coded frame after them carries the DRED data), and 60 to 100 ms into 20,000 bytes; 8 to 24 kHz. 120 ms at the largest bitrate with VBR into 20,000 bytes, where the packets differ on purpose: libopus decodes every one, and some are longer than 6 × 1276 bytes. The DRED data of SILK-only and Hybrid packets with `Audio` and VBR (8 to 48 kHz, 10 to 60 ms, mono and stereo). `OPUS_SET_DRED_DURATION`, `OPUS_GET_DRED_DURATION` and `OPUS_RESET_STATE`, and packets after a reset and after DRED is switched off and on. The multistream encoder with DRED set on each stream. libopus's DRED decoder and opus1's on bursts of 1 to 49 lost packets, from opus1's packets and from libopus's: the same samples, closer to the input than plain concealment. Two report tests print how often the DRED data is equal where it cannot always be (Hybrid with `Voip` or CBR; 40 and 60 ms with VBR). An ignored test measures the time per frame (`encoder_cost_of_dred`). It needs the features `test-libopus-dnn,dred,osce-bwe` and `libopus-host-1.6.1-dred-novec`. | yes |
| `dnn_dred_encoder.rs` | The DRED encoder without libopus: the duration takes 0 to 104 and a reset clears it; DRED data at every rate, mono and stereo, in every mode and in packets of up to 120 ms starts with `'D'` and version 12 and opus1's DRED decoder reads it; no DRED data with VBR and no expected loss; CBR packets keep their size; a duration of 0 gives the packets of an encoder without DRED; `set_dnn_blob` loads the RDOVAE encoder and PitchDNN together or keeps the old models; 120 encoders with random settings and inputs (NaN and huge values included) do not panic; a counting allocator shows that encoding with DRED allocates nothing. | no |
| `dnn_freq_burg.rs` | The signal analysis (`freq.rs`, `burg.rs`, the tables) against libopus on speech, noise and test signals. | yes |
| `dnn_plc_libopus.rs` | The models of deep PLC (`src/dnn/plc/`) against libopus's own functions, whole models at a time, for every `DnnArch`: the feature analysis (`lpcnet_compute_single_frame_features_float`), PitchDNN (`compute_pitchdnn`), FARGAN (`fargan_cont`, `fargan_synthesize`, `fargan_synthesize_int`, over whole signals on its own output), the state machine (`lpcnet_plc_init`, `_reset`, `_update`, `_conceal`, `_fec_add`, `_fec_clear`, which run `compute_plc_pred`) and `update_plc_state`. The C states are `#[repr(C)]` copies of the libopus structs, so the whole state is compared after every call. The sequences of good and lost 10 ms frames cover losses at the start, single losses after 1 to 40 good frames, bursts of 2 to 100 frames, alternating losses, resets in the middle and right after a burst, the DRED queue (with a pitch that gives FARGAN the period 0) and CELT updates; random sequences add bursts, resets and DRED vectors. The signals are `fixtures/answer_16k.wav` and the TSP corpus resampled to 16 kHz (a subset by default, all files with `OPUS1_DNN_PLC_ALL=1`). The ignored test `timing` prints the cost of each call for opus1 and libopus. | yes |
| `dnn_osce_libopus.rs` | LACE, NoLACE, the bandwidth extension, their features, the cross-fades and the filters of `nndsp.c` against libopus's own functions (`osce_enhance_frame`, `osce_reset`, `osce_calculate_features`, `osce_bwe`, `osce_bwe_reset`, `osce_bwe_calculate_features`, `adacomb_process_frame` and the others), through the C shim described below. Real speech is encoded by opus1 as SILK-only mono packets (10 to 60 ms, 6 to 32 kb/s, with stretches of narrowband and mediumband) and decoded frame by frame with opus1's SILK decoder functions, which give the parameters that OSCE reads; the decoded signal is checked against `OpusDecoder`. The streams switch the method, reset the channel as lost frames and decoder resets do, and unload the models for a while; the bandwidth extension runs in 10 and 20 ms calls with resets. Every output sample is compared, for every `DnnArch`. With `OPUS1_OSCE_TIME=1` it prints the time per frame of both. | yes, and the shim |
| `dnn_plc_decoder_libopus.rs` | Deep PLC inside the decoder: `OpusDecoder` and `MultistreamDecoder` against `opus_decode_float`, `opus_decode` and the multistream functions of the linked libopus, every float sample, every 16-bit sample and every final range. The streams come from libopus's and opus1's encoders: SILK at 8, 12 and 16 kHz (10 to 60 ms), Hybrid, CELT (2.5 to 20 ms), mono and stereo, in-band FEC, DTX, mode switches with redundancy frames, and packets of two encoders spliced together (switches without redundancy, where deep PLC conceals 5 ms of a packet that arrived, and SILK rate changes). Each is decoded at 16 and 48 kHz (and 8, 12, 24 kHz with fewer patterns), into one and two channels, at complexities 0, 4, 5, 6 and 10, with no loss, single losses, bursts, losses longer than 200 ms and of 2.2 s, losses concealed in pieces, FEC calls, decoder resets, complexity changes inside losses and losses at the start. From complexity 6 the comparison includes LACE and NoLACE; the BWE stays off. A test checks that libopus's output at complexity 5 differs from complexity 0, so that the comparison means something. `OPUS1_DNN_PLC_ALL=1` adds the TSP corpus, every complexity and every pattern at every rate; `OPUS1_DNN_PLC_STREAMS` keeps the streams whose name contains its value. With `OPUS1_DNN_REF_O2=1` the test `distance_to_an_o2_build` measures the distance to an `-O2` build. The ignored test `timing` prints the time of decode calls of both. | yes |
| `dnn_osce_decoder_libopus.rs` | LACE, NoLACE and the bandwidth extension (BWE) inside the decoder: `OpusDecoder` and `MultistreamDecoder` against the linked libopus, every float sample, every 16-bit sample and every final range. The streams: SILK wideband of 10 to 60 ms at low and high bitrates (stereo with mid-only frames), with FEC and DTX; SILK narrowband and mediumband; Hybrid; VoIP streams that switch between SILK and Hybrid; music streams that switch between SILK, Hybrid and CELT; and spliced streams (SILK wideband with Hybrid, with CELT, with narrowband, with 10 ms SILK, and mono with stereo SILK). Each is decoded at 48 and 16 kHz (8, 12 and 24 kHz for some streams), into one and two channels, at complexities 0, 4, 6, 7 and 10 with the BWE on and 6 and 7 with it off, with no loss, single losses, bursts, losses concealed in pieces, FEC, resets, losses at the start, and switches of the complexity and of the BWE in the middle of a stream. libopus 1.6.1 outputs silence for CELT-only frames after frames with the BWE; when a run reaches such a frame, the test checks that libopus's output is that silence and stops the run there. A test checks that LACE, NoLACE and the BWE each change libopus's output, and that the linked libopus has the defect. A 5.1 multistream decoder is compared at complexities 4 to 10 with the BWE on and off. `OPUS1_DNN_OSCE_ALL=1` adds the TSP corpus, every output rate, every complexity with the BWE on and off, and every event; `OPUS1_DNN_OSCE_STREAMS` keeps the streams whose name contains its value. With `OPUS1_DNN_REF_O2=1` the test `distance_to_an_o2_build` measures the distance to an `-O2` build. The ignored test `timing` prints the time of decode calls of both. | yes |
| `dnn_decoder_controls_libopus.rs` | The decoder controls against a libopus with the features: complexity, the BWE switch and its values, and that libopus has no `OPUS_SET_DNN_BLOB` when it has built-in weights. | yes |
| `dnn_weights.rs` | The blobs and the loader without libopus: the SHA-256 sums, every network loads with the shapes of its descriptors, groups load on their own, and malformed blobs give errors (property tests included). | no |
| `dnn_props.rs` | Properties of the kernels (`proptest`): every `DnnArch` gives the bits of `Portable`, activations stay in range, quantization rounds as specified. | no |
| `dnn_edges.rs` | Edge cases of the kernels on every `DnnArch`: sparse index tables with empty row blocks and repeated or falling positions, int8 weights of ±127 with inputs that quantize to -128 and 127 (up to 2048 inputs), NaN, infinities and subnormal numbers as inputs, GRU and GLU layers of odd sizes, layers without inputs or outputs, and invalid layers and short slices, which must panic. It also checks the arithmetic of the x86 kernels in scalar form (their FRECPE and their int8 quantization), because they cannot run here. | no |
| `dnn_frecpe.rs` | The software FRECPE against the ARM instruction: about 2^17 inputs; all 2^32 inputs in an ignored test. | no |
| `dnn_no_alloc.rs` | The layer functions allocate nothing. | no |
| `dnn_plc_props.rs` | Properties of deep PLC: finite output on silence, full-scale squares and impulses, noise, garbage DRED features (NaN, infinities, the periods 0 and the largest `int`) and garbage CELT history; a loss of 1.5 s fades by more than 20 dB; every `DnnArch` gives the same frames; a reset clears everything but FARGAN, as in libopus; the DRED queue drops vectors beyond its 104 places; no allocation; a first lost frame fits in a 64 KB stack (release) and 256 KB (debug); the size of the state. | no |
| `dnn_dsp_props.rs` | Properties of the signal analysis: energy kept by the transform, stable filters. | no |
| `dnn_plc_decoder_props.rs` | Properties of deep PLC inside the decoder: decode calls allocate nothing (losses, FEC, resets, 16-bit output, multistream), decode calls with a first deep PLC loss fit in a 64 KB stack (release) and 256 KB (debug), without loss the complexity changes nothing, complexity 4 conceals as 0 and complexity 5 differently, a reset decoder decodes as a new one until deep PLC has run (FARGAN's state survives a reset), and a 2 s loss stays finite and no louder than the speech before it. | no |
| `dnn_osce_decoder_props.rs` | Properties of OSCE inside the decoder: decode calls allocate nothing (losses, FEC, resets, complexity and BWE changes, 16-bit output, multistream); decode calls with NoLACE, the BWE and deep PLC fit in a 64 KB stack (release) and 256 KB (debug); LACE and NoLACE change only SILK frames of 20 ms at 16 kHz; the BWE changes only SILK wideband at 48 kHz output from complexity 4; CELT-only frames after frames with the BWE are decoded as with the BWE off (the fix of libopus main); the final range does not depend on OSCE; `MultistreamDecoder` forwards the complexity and the BWE switch to every stream. | no |
| `core_decoder_dnn_controls.rs` | The decoder controls against the Homebrew libopus (part of the default suite): complexity range and errors, the BWE switch with `osce-bwe` (without it the methods do not exist), multistream forwarding, blob loading (with `deep-plc`), and, in builds without `deep-plc` and `osce`, that the complexity does not change the decoded output. | no |

Commands (from the repository root; `target/dnn-ref` keeps this build apart from the normal one):

```text
# Without libopus:
cargo test --release --features osce-bwe --test dnn_weights --test dnn_props --test dnn_no_alloc \
  --test dnn_edges --test dnn_dsp_props --test dnn_frecpe --test core_decoder_dnn_controls \
  --test dnn_plc_props --test dnn_plc_decoder_props --test dnn_osce_decoder_props --lib dnn::

# Against the bit-exact reference (OPUS1_CORPUS names the corpus for dnn_plc_libopus):
OPUS1_CORPUS=$PWD/corpus \
LIBOPUS_LIB_DIR=$PWD/target/android/libopus-host-1.6.1-dnn-novec LIBOPUS_NO_PKG=1 LIBOPUS_STATIC=1 \
  CARGO_TARGET_DIR=target/dnn-ref cargo test --release --features test-libopus-dnn,osce-bwe \
  --test dnn_kernels_libopus --test dnn_layers_libopus --test dnn_weights_libopus \
  --test dnn_freq_burg --test dnn_decoder_controls_libopus --test dnn_plc_libopus \
  --test dnn_plc_decoder_libopus --test dnn_osce_decoder_libopus

# The OSCE test loads a C shim built against the reference (see below):
scripts/dnn/build-osce-shim.sh target/android/libopus-host-1.6.1-dnn-novec
LIBOPUS_LIB_DIR=$PWD/target/android/libopus-host-1.6.1-dnn-novec LIBOPUS_NO_PKG=1 LIBOPUS_STATIC=1 \
  CARGO_TARGET_DIR=target/dnn-ref cargo test --release --features test-libopus-dnn,osce-bwe \
  --test dnn_osce_libopus

# Against the -O2 build, measuring instead of requiring equal bits:
OPUS1_DNN_REF_O2=1 LIBOPUS_LIB_DIR=$PWD/target/android/libopus-host-1.6.1-dnn LIBOPUS_NO_PKG=1 \
  LIBOPUS_STATIC=1 CARGO_TARGET_DIR=target/dnn-ref-o2 cargo test --release \
  --features test-libopus-dnn,osce-bwe --test dnn_layers_libopus -- --nocapture --test-threads=1
```

The OSCE functions of libopus take its whole SILK decoder state, so `tests/dnn_osce_libopus.rs` calls them through a small C library, `scripts/dnn/osce_test_shim.c`. It keeps a `silk_decoder_state` and a `silk_decoder_control`, fills the fields OSCE reads, and calls the functions. It also sets `model->loaded`, which `osce_load_models` leaves at 0 (only `silk_LoadOSCEModels` sets it); without it, `osce_enhance_frame` would silently fall back to method NONE. `scripts/dnn/build-osce-shim.sh <reference build>` compiles the shim with the defines and flags recorded in the build's `build-flags.txt` and links it with the build's `libopus.a` into a shared library, `target/osce-shim/<build name>/libosce_shim.dylib`. The test loads that library at run time (`dlopen`) and finds it from `LIBOPUS_LIB_DIR`, or at `OPUS1_OSCE_SHIM`. So one test binary can compare with several reference builds, and a run against a libopus without the shim fails with instructions instead of failing to link. `OPUS1_OSCE_FILES` sets how many files of the TSP corpus each stream uses (48 by default, `all` for the whole corpus).

`OPUS1_DNN_REF_O2=1` works for `dnn_kernels_libopus.rs` and `dnn_layers_libopus.rs`, and for `dnn_osce_libopus.rs` with `OPUS1_OSCE_SHIM` pointing at a shim built from the `-O2` build: there the test prints the frames that differ, the largest difference and the signal-to-noise ratio of the difference. With it, these tests require equal bits except where the `-O2` build sums in another order, and print the largest difference. `dnn_plc_libopus.rs` has the mode too: it then prints how far the outputs are apart and requires only that the concealed frames stay within 8 dB signal-to-noise ratio of libopus's. In `dnn_plc_decoder_libopus.rs` and `dnn_osce_decoder_libopus.rs` the variable runs the test `distance_to_an_o2_build`, which measures the decoder's output in the same way (link the `-O2` build in its own `CARGO_TARGET_DIR`, for example `target/dnn-ref-o2`). `dnn_freq_burg.rs` has no such mode: against an `-O2` build it fails ([Differences from libopus](10-differences-from-libopus.md#neural-network-code) gives the differences).

Results on the Mac (Apple M4 Pro):

- Against `libopus-host-1.6.1-dnn-novec`: all pass. `dnn_layers_libopus.rs` makes 357,192 uses of a layer (a GRU step uses two) on 1804 frames of speech, each through libopus and the three variants `Portable`, `Neon` and `NeonDotprod`, and every output, state and memory is equal. The pitch equals that of `compute_pitchdnn` in all 1804 frames.
- Against `libopus-host-1.6.1-dnn` with `OPUS1_DNN_REF_O2=1`: only float layers with one output differ, as expected. [Differences from libopus](10-differences-from-libopus.md#neural-network-code) gives the measured differences.
- `dnn_plc_libopus.rs` against `libopus-host-1.6.1-dnn-novec`, with all 1,444 files of the TSP corpus and the fixture (`OPUS1_DNN_PLC_ALL=1`, 217 s): all pass, for `Portable`, `Neon` and `NeonDotprod`. That is 343,656 frames of feature analysis and of PitchDNN, 336,431 frames of FARGAN, and random sequences of good and lost frames on 1,445 signals, with every concealed frame and the compared fields of the state (positions, history, features, network states, FARGAN state) equal after every call. The default subset (182 signals) takes 19 s.
- `dnn_plc_libopus.rs` against `libopus-host-1.6.1-dnn` with `OPUS1_DNN_REF_O2=1`: the concealed frames differ at 15.7 dB SNR on the scripted sequence, 10.9 dB on the random sequences of the default subset (182 signals, 7,335 concealed frames) and 13.2 dB on those of all 1,445 signals (56,800 concealed frames). The test requires at least 8 dB. Builds of libopus differ from each other by about as much: the `-O2`, `-Os` and Android `-O2` builds conceal the same losses at 10.0 to 15.4 dB from each other.
- `dnn_osce_libopus.rs` against `libopus-host-1.6.1-dnn-novec`, with the whole TSP corpus (`OPUS1_OSCE_FILES=all`, 1444 files joined into one stream per test, about 196,000 SILK frames each): every output sample is equal for `Portable`, `Neon` and `NeonDotprod`. That is 139,852 frames enhanced by LACE, 140,994 by NoLACE, a stream that switches between NONE, LACE and NoLACE (45,237, 39,624 and 54,720 frames) with resets and unloaded models, 231,721 calls of the bandwidth extension (164 million output samples), and NoLACE followed by the bandwidth extension. The features of 4909 frames and 2000 calls of each filter of `nndsp.c` are equal bit for bit. The run took 304 s with 4 test threads. With `OPUS1_OSCE_TIME=1` (5200 frames of 20 ms, best of 5 runs, `NeonDotprod`), opus1 took 17 µs per frame for LACE, 58 µs for NoLACE and 54 µs for the bandwidth extension; the novec libopus took 23, 76 and 80 µs, and the `-O2` libopus 20, 71 and 72 µs. With method NONE, libopus spends 5 to 6 µs per frame on the features, which opus1 skips.
- `dnn_plc_decoder_libopus.rs` against `libopus-host-1.6.1-dnn-novec` (2026-10-18, 14 threads): every decode call equal, in float, in 16 bits and in the final range. The default run, with complexities 6 and 10 compared for every stream since LACE and NoLACE run in the decoder, decodes 62 streams of 4 s in 8,184 runs with about 2.16 million decode calls (379,000 of them concealed or rebuilt), and 64 multistream runs, in 46 s. Before OSCE was connected, with `OPUS1_DNN_PLC_ALL=1` (8 s streams with corpus speech, the complexities then compared, every output rate and pattern): 54,700 runs, 35,968,210 decode calls (6,152,394 concealed or rebuilt), 412 s. A mutation check made the cross-fade of the first neural CELT frame unfused: 1,156 of 3,576 runs then differed.
- `dnn_osce_decoder_libopus.rs` against `libopus-host-1.6.1-dnn-novec` (2026-10-19, 14 threads): every decode call equal, in float, in 16 bits and in the final range, up to the first CELT-only frame after frames with the BWE, where the test checks libopus 1.6.1's silence and stops the run. The default run decodes 53 streams of 4 s in 6,784 runs with 1,498,346 decode calls (188,087 concealed or rebuilt; 367 runs stopped at that defect), the other output rates in 2,880 runs with 307,350 calls, and 128 multistream runs, in about 110 s. With `OPUS1_DNN_OSCE_ALL=1` (8 s streams with corpus speech, every output rate, every complexity with the BWE on and off, every event), run in 8 groups of streams that cover all 53 streams (4 of them twice): 100,320 runs, 45,869,221 decode calls, 1,464 runs stopped at the defect, 0 differ, about 30 minutes in all.
- `dnn_plc_decoder_libopus.rs` against `libopus-host-1.6.1-dnn` with `OPUS1_DNN_REF_O2=1` (`distance_to_an_o2_build`, mono 48 kHz, complexity 5, single losses, bursts and long losses): the concealed audio is 13.5 dB SNR from the `-O2` build's (12.1 dB with the corpus, 20,370 concealed calls, of which 15,967 differ), and the audio decoded after the losses 26.9 dB (27.3 dB). With the corpus, single runs range from 0.9 dB to 111 dB, median 13.7 dB; a run with few concealed calls can be far off. In no run was the loudest concealed call more than 0.8 dB louder than the `-O2` build's loudest.

Results on the phone (HONOR CLK-AN00, 2026-10-18), on a Cortex-A76 (CPU 7) and a Cortex-A55 (CPU 5), with the 11 files of the table other than `dnn_edges.rs` (added later) and the unit tests of `src/dnn/`, built with `--features test-libopus-dnn,osce-bwe`:

- Against `libopus-1.6.1-dnn-novec`: all pass on both cores. `dnn_layers_libopus.rs` took 14 s on the Cortex-A76 and 51 s on the Cortex-A55.
- Against `libopus-1.6.1-dnn` (NDK clang 18 at `-O2`): every layer of the six networks is bit-exact, on both cores. Two checks fail, as on the Mac: 4 of the 9 tests of `dnn_freq_burg.rs` (band energies, Burg's method, the Burg cepstrum, `lpc_from_cepstrum`), and the pitch check of `dnn_layers_libopus.rs`. With `OPUS1_DNN_REF_O2=1` the pitch differs in 375 of 1804 frames, by at most 4.8e-7.

`dnn_plc_libopus.rs` and `dnn_plc_props.rs` on the same phone (2026-10-18), on the Cortex-A76 (CPU mask 80) and the Cortex-A55 (CPU mask 01), against `libopus-1.6.1-dnn-novec`, with the fixture and 50 files of the TSP corpus (two per speaker, `OPUS1_DNN_PLC_ALL=1`): all pass on both cores, for `Portable`, `Neon` and `NeonDotprod`. That is 13,792 frames of feature analysis and of PitchDNN, 13,537 frames of FARGAN, and 2,417 concealed frames, each equal to libopus's. The runs took 34 s on the Cortex-A76 and 131 s on the Cortex-A55. They were built with `cargo ndk -t arm64-v8a -P 21 test --release --no-run --features test-libopus-dnn,deep-plc --test dnn_plc_libopus --test dnn_plc_props` and the settings of `cargo_android`.

`scripts/android/test.sh` does not pass cargo features yet. This run used `cargo ndk -t arm64-v8a -P 21 test --release --no-run --features test-libopus-dnn,osce-bwe` with the settings of `cargo_android` in `scripts/android/env.sh` (`LIBOPUS_LIB_DIR`, `LIBOPUS_STATIC=1`, `RUSTC_WRAPPER` and `OPUS1_DEVICE_DIR`), and copied `fixtures/`, `dnn-weights/` and the blobs of `target/android/dnn-blob-1.6.1/` (named by `OPUS1_DNN_BLOB_DIR`) to the phone.

### Results on the phone

Run on 2026-10-18 at git commit `f952b46` (the speech and music analysis and DTX), against libopus 1.6.1 built by `build-libopus.sh` at `-O2` with the NDK's clang 18: 243 test programs with 1553 tests. 1498 passed, 29 failed and 26 were ignored. The run took 796 seconds, with the tests already built, including copying them to the phone. The logs are in `target/perf/analysis_c7/phone_tests_O2/`.

The same programs, built against libopus 1.6.1 at `-Os` (`OPUS_EXTRA_CFLAGS=-Os`), passed every one of the 28 failing tests that compare with libopus (logs in `target/perf/analysis_c7/phone_tests_Os/`). Only `issue_15_loss_concealment` still failed there; it makes its input with the C library's sine (see below). So the 28 failures come from the code that clang generates for libopus at `-O2`, not from opus1:

| Failing tests | Cause |
|---|---|
| The 8 of the earlier runs: `celtdec_plc` (5), `celtenc_bands_libopus::stereo_angles_are_the_ones_libopus_computes`, `perfk_pitch_libopus::periodic_signals_give_the_lag_of_libopus`, `issue_15_loss_concealment::silk_concealment_extrapolates_the_signal` | The vector code of libopus's loss concealment, `stereo_itheta` and pitch search, and the C library's sine. The table below explains each. |
| 11 tests of `analysis_libopus`, 3 of `core_analysis_libopus` (the analysis of each frame, 124 streams; the expert frame duration, 3 of 72 streams; CELT-only at 48 kHz, 11 of 252 streams), `multistream_analysis_libopus::the_analysis_of_each_stream_is_libopus` (132 runs) and `core_dtx_libopus::celt_only_streams_are_libopus_at_48_khz` (7 of 168 streams) | The speech and music analysis at `-O2`: clang's vector code rounds some products of the band sums before it adds them (analysis.c, lines 657 and 658, and in the NDK's build also line 864), where the Homebrew build fuses them (see [Differences from libopus](10-differences-from-libopus.md#what-the-figures-were-measured-against)). The first difference is in the last bit of a probability, a slope or a state value, for example `tonality_slope` 0x3c48abd0 against 0x3c48abd1. Where a test compares packets, a few streams then take another decision. |
| `celtenc_analysis::the_decaying_output_of_the_dc_filter_is_coded_as_libopus_codes_it`, `core_lowdelay::the_decaying_output_of_the_dc_filter_is_coded_as_libopus_codes_it` and `core_dtx_libopus::celt_only_streams_are_libopus_at_8_and_12_khz` (3 of 168 streams) | CELT encoding of signals at very low levels, where libopus's `EPSILON` terms matter, at `-O2`. These tests run no analysis (complexity 5, the CELT encoder alone, or 8 and 12 kHz). The two tests of the DC filter came after the last phone run. They pass against the `-Os` build, so the code of libopus's CELT encoder at `-O2` is the cause; which loop was not looked for. |
| `core_dtx_libopus::opus1_dtx_streams_decode_like_libopus` (85 packets) and `multistream_dtx_libopus::opus1_dtx_packets_decode_like_libopus` (149 packets) | The decoders conceal the 1-byte packets of DTX. This is the vector code of libopus's loss concealment, the cause of the `celtdec_plc` failures. |

The run of 2026-10-17 at git commit `56ca2dd` (round 8 and the frames of 80 to 120 ms merged) had 235 test programs with 1470 tests: 1435 passed, 8 failed and 27 were ignored (logs in `target/perf/merge_56ca2dd/phone_tests/`). The same 8 tests had failed at `c96e936`, before round 8.

The 8 failures of the earlier runs, in detail. None of them comes from opus1. Each failing test compares opus1 with libopus at the last bit, or makes its input with the C library's sine function. The difference is on that side:

| Failing tests | Cause | How it was found |
|---|---|---|
| `celtenc_bands_libopus::stereo_angles_are_the_ones_libopus_computes` | Vector code in libopus's `celt/vq.c`. At `-O2`, clang turns the loop of `stereo_itheta` that sums `m*m` and `s*s` into vector code. The vector code rounds each product before it adds it. The scalar code does both in one fused multiply-add. So the last bit of the angle can differ. | See below the table. |
| `perfk_pitch_libopus::periodic_signals_give_the_lag_of_libopus` | Vector code in libopus's `celt/pitch.c`, where clang vectorizes loops of `pitch_downsample` and `pitch_search`. For signals that repeat exactly, `pitch_search` of opus1 then finds libopus's lag in 1165 of 1168 cases instead of all. | See below the table. |
| 5 tests of `celtdec_plc`: `celt_only_bursts`, `celt_only_losses_at_every_decoder_rate`, `celt_only_losses_in_narrower_bandwidths`, `celt_only_single_losses_at_every_frame_size` and `pitch_concealment_returns_after_two_good_packets` | Vector code in libopus's loss concealment: in `celt/celt_lpc.c` (the linear prediction and its filters) and in `celt_decode_lost` (`celt/celt_decoder.c`). On 64-bit ARM these tests require every concealed frame to be the one libopus writes (a signal-to-noise ratio of 200 dB). Against the `-O2` build, some concealed frames match at about 60 to 120 dB only, so they differ in the last bits. | See below the table. Each of the two files alone at `-O2` makes all 5 tests fail. |
| `issue_15_loss_concealment::silk_concealment_extrapolates_the_signal` | The test signal. It is made with `f32::sin`, which calls the C library's `sinf`. Android's and Apple's `sinf` give different last bits for some inputs. With the phone's signal, the concealed 48 kHz frame has 0.0093 of the energy of the last real frame. The test requires more than 0.01. | Of 2,000,000 inputs, Apple's `sinf` differs from the sine computed in double precision and then rounded in 151,818 cases, Android's in 25,754. A copy of the test, run on the Mac with that rounded double-precision sine, got exactly the phone's numbers. libopus's decoder, given the same packets, also conceals the frame at 0.0093 of the energy. |

How the vector code was found to cause the first 7 failures:

1. On the phone, all 7 pass against a build of 1.6.1 with `OPUS_EXTRA_CFLAGS="-fno-vectorize -fno-slp-vectorize"`. Only the test of `issue_15_loss_concealment` still fails.
2. On the Mac, the whole suite against the `-O2` build of 1.6.1 (Apple clang 21) fails the same 7 tests. They pass against the `-Os` build, against the build without vector loops and against the Homebrew library.
3. On the Mac, a library was put together from the objects of the build without vector loops, with one file taken from the `-O2` build. `celt/vq.c` from the `-O2` build makes the first test fail, `celt/pitch.c` the second, and `celt/celt_lpc.c` or `celt/celt_decoder.c` the 5 tests of `celtdec_plc`. None of the other 10 files of `celt/` whose machine code the vector code changes makes one of these tests fail. (Records: `target/perf/round8/ref/mac_bisect.txt`.)

The earlier runs linked the main branch of libopus (see [Which libopus source to build](#which-libopus-source-to-build)). They had 10 failures at commit `bd080de` and 18 at commit `0090242`. 10 tests failed only because that source codes some streams differently from 1.6.1. They pass with 1.6.1:

- 8 tests of LFE streams and mode switches: `celtenc_known_gaps` (2), `celtenc_lfe`, `celtenc_lfe_formats`, `celtenc_lfe_header`, `multistream_e2e_matrix` (2) and `core_redundancy`.
- `celtenc_gap_report::sample_of_rows_is_the_packets_of_libopus` and `celtenc_rate_rules::bitrate_is_limited_to_750_kbps_per_channel`.

At git commit `f952b46`, the harness on the phone wrote a golden file identical in every field to `target/perf/golden_analysis.txt` of the Mac, against the `-O2` build of 1.6.1. Both encoders run the speech and music analysis in 104 of these streams (complexity 7 to 10 at 16 to 48 kHz, among them the 8 mode-switching streams), and the last bits of libopus's analysis at `-O2` changed none of libopus's packets there. Earlier, `golden.sh --check target/perf/golden_r6.txt` found 0 of 137 streams that differ, against the `-O2` build of 1.6.1 and against the build without vector loops. The golden file of the phone is identical in every field to the one of the Mac, whether the Mac links the Homebrew library or a build of 1.6.1 at `-O2` or `-Os`. So neither the vector code of `-O2` nor Android's C library changes a hash of these streams. Android's `acos`, `cos`, `exp` and `log` give other last bits than Apple's for some inputs. With the main branch, 86 of the 137 streams differed in the fields that depend on libopus's packets: `c_pk`, `dec_c`, `dec_c_plc` and `dec_c_burst`, and in 7 of them also `fec_silk`.

## Tests on an x86-64 computer

The scripts in `scripts/x86/` test and time opus1 on a 64-bit x86 computer. Each script explains its options and settings at its top.

| Script | What it does |
|---|---|
| `test.sh` | Tests opus1 under Linux, also inside WSL2 (Windows Subsystem for Linux) on Windows. It needs a C compiler, `make`, `curl` and Rust. It writes `x86-results-<date>.tar.gz` in the repository root. |
| `test.ps1` | The same natively on 64-bit Windows, with Rust's MSVC toolchain. It builds libopus with CMake and Visual Studio. It writes `x86-windows-results-<date>.zip` in the repository root. |
| `ab.sh`, `ab.ps1` | Time two source trees of opus1 in turns on one computer: A, the baseline, named on the command line, and B, the tree that holds the script. `ab.sh` runs on Linux, WSL2 and macOS; `ab.ps1` on Windows. |
| `ab_report.rs` | Writes the report of an A/B timing. It needs only `rustc`. |

`test.sh` and `test.ps1` take the same steps:

1. They record the CPU (central processing unit), its vector instruction sets (SSE2, AVX, AVX2, FMA, AVX-512, AVX-VNNI), the operating system and the tool versions.
2. They download the libopus 1.6.1 release, or take the file that `OPUS_TARBALL` names, and check its SHA-256 sum. They build it twice as a static library: the default build, and a build with deep PLC, OSCE and DRED. `test.sh` uses libopus's `configure` script, at `-O2` and with libopus's run-time CPU detection. `test.ps1` uses CMake and Visual Studio, in the Release configuration. They also unpack the release for `libopus-fixed` and set `LIBOPUS_SRC_1_6_1` (see [Checks against the fixed-point build of libopus](#checks-against-the-fixed-point-build-of-libopus)).
3. They run the whole test suite, linked with the default build.
4. They run the tests that compare every x86 kernel variant of the neural networks that the CPU has (AVX2 with FMA, AVX-VNNI, AVX-512 VNNI) with the portable code. Then they run the tests against the libopus built with its neural network code.
5. They check the golden file of 64-bit ARM and record how many streams differ. On Windows every stream differs: the golden file also holds libopus's packets (`c_pk`), and the libopus that Visual Studio builds gives other packets. Then they run `perf_harness modes`, which runs both encoders and both decoders on the same input and writes `modes.md`. It holds how far the two decoders agree, in decibels (dB), and the coding error of each encoder when libopus decodes its packets.
6. They time encoding and decoding against the default libopus build.

Their options:

- `--quick` (`-Quick`) skips the timing of step 6.
- `--no-dnn` (`-NoDnn`) skips the libopus build with the neural network code and the tests against it. The kernel tests of step 4 still run.
- `--ab BASELINE_DIR` (`-AbBaseline DIR`) replaces the timing of step 6 with an A/B timing against the source tree `BASELINE_DIR`, by `ab.sh` or `ab.ps1`. `--passes N` (`-Passes N`) sets the number of runs of each build, 5 by default. With `--quick` the A/B step only builds the two programs.
- `OPUS1_CORPUS` names a corpus folder with `rfc8251/`, for the conformance tests. Without it they are skipped (see [Corpora](#corpora)).

On x86-64 the output of opus1 is correct but not bit-exact with libopus (see [Differences from libopus](10-differences-from-libopus.md#platforms)). The tests that require libopus's bits are ignored there, with the reason "bit-exact with libopus only on 64-bit ARM (docs/guide/10, Platforms)". So on x86 the golden file of 64-bit ARM is only a record of how many streams differ. To see whether a change altered the output of opus1, write a golden file on the same computer with the tree before the change (`perf_harness golden --write FILE`), then check the tree after the change against it (`golden --check FILE`). `modes.md` shows whether the output is still correct.

Each run of the test suite writes its cargo log (`NAME.txt`), the result line of each test program (`NAME-summary.txt`) and the failed tests (`NAME-failed.txt`). `summary.txt` collects the passed, failed and ignored counts of all runs. A test program that crashes, for example with a stack overflow, prints no result line. The scripts name such a program in `summary.txt` and `NAME-failed.txt`, with the reason that cargo gives, and count it as one failed test. If cargo fails and the log shows no failed test, the summary counts one failed test and says so.

On Windows a thread cannot use all of its stack. Rust's standard library keeps 20 KB of each thread's stack for its stack overflow handler, and the end of the stack holds guard pages. So the four tests that run the neural network code on a thread of 64 KB give that thread 32 KB more on Windows (`thread_stack_size` in `tests/dnn_plc_common/mod.rs`, and the unit test `the_osce_path_runs_on_64_kb_of_stack`). On other systems the thread gets the size that the test names.

The A/B timing runs the two builds in turns because the clock and the temperature of a computer change during a session. Two runs one right after the other see the same state, so each such pair gives one comparison. The order turns from pair to pair: with 5 passes it is A B B A A B B A A B. One run is the bench set of `perf_harness`, about 3 minutes on an Intel Core i7-12700. Both builds link the same libopus, so the libopus times are a control: their ratio B/A should stay near 1. The output folder holds `ab-report.md` (time of B divided by time of A, per scenario and as geometric means, for opus1 encode and decode and for libopus), `ab-summary.txt`, `pairs.md`, `system.txt`, `builds.txt` and the records of each run in `runs/`. `ab.sh` can also build for another target, for example `CARGO_BUILD_TARGET=x86_64-apple-darwin` under Rosetta 2 on a Mac.

```text
scripts/x86/test.sh
scripts/x86/test.sh --ab ../opus1-base
scripts/x86/ab.sh --passes 3 ../opus1-base -- --filter celt_fb_48k_mono --reps 3 --procs 1
```

`test.sh` and `test.ps1` ran on an Intel Core i7-12700 (AVX2, FMA and AVX-VNNI, no AVX-512) on 2026-10-21, under Linux in WSL2 and natively on Windows. [Performance](08-performance.md#on-x86-64) gives the timings. The tests that call the fixed-point build were also checked there with MSVC (commit `9218041`).

## How to run each part

| What | Command |
|---|---|
| Everything | `cargo test --release` |
| Unit tests only | `cargo test --release --lib` |
| Documentation examples only | `cargo test --release --doc` |
| One test file | `cargo test --release --test core_encoder` |
| Tests whose name contains a word | `cargo test --release --test core_encoder -- final_range` |
| One group of files | `for f in tests/k24_*.rs; do cargo test --release --test "$(basename "$f" .rs)"; done` |
| Show the figures a test prints | add `-- --nocapture` |
| A report table | `cargo test --release --test k24_enc_report -- --ignored --nocapture` |
| Golden hashes | `cargo run --release --example perf_harness -- golden --check <file>` |
| `no_std` builds and tests | `scripts/check_no_std.sh` |
| Fuzzing (stable toolchain) | `fuzz/fuzz.sh <target> <seconds>` |
| RFC 8251 conformance | `OPUS1_CORPUS=<corpus> cargo test --release --test eval_rfc8251 -- --nocapture` |
| `opus_compare` against the C program | `scripts/eval/check-opus-compare.sh --src <libopus 1.6.1 source>`, then `OPUS1_OPUS_COMPARE_C=target/eval/opus_compare/opus_compare_exact cargo test --release --test eval_opus_compare` |
| Paired comparison with libopus | `cargo run --release --example eval -- run` (the `quick` set) |
| The self-test of the gates | `cargo run --release --example eval -- selftest` |
| Get a corpus | `scripts/eval/fetch-corpus.sh rfc8251` |
| Tests on an Android phone | `scripts/android/build-libopus.sh --src <libopus 1.6.1 source>`, then `scripts/android/test.sh` |
| Tests of the neural network code | see [Tests of the neural network code](#tests-of-the-neural-network-code) |
| Golden hashes on a phone | `scripts/android/golden.sh --check <file>` |
| Timings on a phone | `scripts/android/bench.sh --runs 2` |
| Hardware counters and hot functions on a phone | `scripts/android/profile.sh --little <scenario> enc rust` |
| Tests that call the fixed-point build | `LIBOPUS_SRC_1_6_1=<libopus 1.6.1 source> cargo test --release --test core_fixed_point` |
| Tests on an x86-64 computer | `scripts/x86/test.sh` (Linux, WSL2) or `scripts/x86/test.ps1` (Windows) |
| A/B timing of two source trees | `scripts/x86/ab.sh <baseline tree>` (Linux, WSL2, macOS) or `scripts/x86/ab.ps1` (Windows) |

Run the tests with `--release`. The tests encode and decode seconds of audio per case. All timings in this document are for release builds.

## See also

- [Performance](08-performance.md): the harness and its timing commands.
- [Differences from libopus](10-differences-from-libopus.md): what the tests show to differ.
- [Architecture](07-architecture.md): where the code under test lives.
- [Glossary](glossary.md): the terms used in these documents.
- The documentation index: [README.md](../README.md).
