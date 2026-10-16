# Testing

This document explains how opus1 is tested. Most tests run the C library libopus in the same process and compare its output with the output of opus1. The sections below cover how libopus is linked, what the tests check, and the groups of test files. They then cover the checks against a fixed-point build of libopus, fuzzing, the `#![no_std]` checks, the ignored tests and the documentation examples. The last section gives the command for each part.

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

Counted with `cargo test --release -- --list` on 64-bit ARM macOS (`aarch64-apple-darwin`) with the default features:

| Part | Files | Tests |
|---|---:|---:|
| Integration tests in `tests/*.rs` | 228 | 1094 |
| Unit tests inside `src/` | - | 312 |
| Documentation tests: the module `multistream` (2) and the Rust examples of `README.md` and `docs/` (47) | - | 49 |
| **Total** | | **1455** |

25 of these tests are ignored (see [Ignored tests](#ignored-tests)). Some unit tests exist only on 64-bit ARM, because they test NEON code. So the count differs on other targets. `tests/` also has 11 directories of shared helper code (for example `tests/k24_common/`) and 4 files of recorded reference values (`*.expected`).

A full run, `cargo test --release`, passed all 1430 tests that are not ignored on Apple M4 Pro. It took 129 seconds, with the tests already built.

## How libopus is linked

Two development dependencies give the tests access to libopus:

- `opus` 0.3: a safe Rust wrapper around the libopus encoder and decoder.
- `audiopus_sys` 0.2: the raw C interface of libopus. The tests need it for controls that `opus` does not expose. Examples are forcing the coding mode (`OPUS_SET_FORCE_MODE`), decoding with the FEC flag (`opus_decode_float` with `decode_fec = 1`), the repacketizer and the multistream functions.

`opus` uses `audiopus_sys`, so both reach the same libopus. Some tests also call internal functions of libopus directly, such as its fast Fourier transform (FFT) and its pitch search (`tests/perfk_fft_libopus.rs`, `tests/perfk_pitch_libopus.rs`). That works because libopus is linked as a static library, which is the default of `audiopus_sys` on macOS.

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

The libopus that Homebrew ships is a floating-point build. Its SILK encoder is the floating-point one. The SILK encoder of opus1 is a port of the fixed-point one, so SILK encoder packets are compared against a separate fixed-point build (see [Checks against the fixed-point build of libopus](#checks-against-the-fixed-point-build-of-libopus)).

## What the tests check

Most tests drive opus1 and libopus with the same input and settings, then compare. The checks fall into these kinds:

1. **Same packets, same decoded output.** libopus encodes a stream. Both decoders decode it, often with the same lost packets. The SILK decoder must give identical 16-bit samples. Where the output is not bit-exact (most CELT streams, see [Differences from libopus](10-differences-from-libopus.md#decoder)), the test requires an SNR against libopus's output above a stated bound.
2. **Same settings, same decisions.** Both encoders code the same input with the same settings. The test compares the TOC bytes (so the mode, bandwidth and frame size), the packet sizes, or the packets byte for byte.
3. **Same quality.** Where packets differ, both encoders' packets are decoded by libopus and compared with the input. The test bounds the difference in SNR.
4. **The same final range.** After each packet, the encoder and the decoder both report their final range (`OpusEncoder::final_range`, `OpusDecoder::final_range`). Equal values show that the decoder read exactly the bits the encoder wrote. The libopus test program `opus_demo` makes the same check: it reports "Range coder state mismatch" when the two values differ.
5. **Properties without libopus.** Round trips, damaged and random packets (nothing may panic), extreme arguments, no heap allocation per frame, and the stack use of the default build.

The final-range check, written as a test of the kind the suite contains:

```rust
use opus1::{Application, OpusDecoder, OpusEncoder};

let mut encoder = OpusEncoder::new(16_000, 1, Application::Voip).unwrap();
encoder.bitrate_bps = 20_000;
let mut decoder = OpusDecoder::new(16_000, 1).unwrap();

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
| `multistream_` | 31 | 276 | The multistream encoder and decoder (several Opus streams in one packet, for surround sound) against libopus's multistream functions: layouts (`_c_layout`), rates, the decoder (`_dec_*`), the encoder (`_enc_*`), end to end (`_e2e_*`) and the surround analysis (`_surround_*`). |
| `core_` | 22 | 142 | The top layer, `OpusEncoder` and `OpusDecoder`, against libopus: controls, channel counts, packet loss, FEC, mode switches and their redundant frames, packet framing, limits, 16-bit output. |
| `perfk_` | 33 | 113 | Guards for the speed work. Each rewritten kernel is compared with the libopus function, with reference values recorded from libopus, or with the code from before the rewrite (`tests/perfk_nsq_replay.rs` keeps the old quantizers for that). |
| `celtenc_` | 16 | 78 | Decisions of the CELT encoder against libopus: band energies, stereo, the pitch pre-filter, the low-frequency effects (LFE) channel, the energy mask, mode switching. |
| `issue_` | 14 | 75 | Regression tests named after GitHub issues, most of them against libopus (for example `issue_42_stereo_channels.rs`). `issues_repro.rs` adds 14 more tests. |
| `k24_` | 21 | 72 | The 24 kilohertz (kHz) path: decoding at every output rate, encoding in each mode, SILK loss handling. `k24_main_use.rs` lets libopus choose the mode, as in a real call. |
| `silk_` | 14 | 66 | SILK parts tested one by one: the bitstream, Burg analysis, linear predictive coding (LPC), the normalized line spectral frequencies (NLSF), the quantizer, and packets of the fixed-point build. |
| `celtdec_` | 4 | 38 | The CELT decoder: channel conversions, edge cases, loss concealment and robustness, against libopus. |
| `silkdec_` | 10 | 31 | The SILK decoder alone (`SilkDecoder`), against libopus with SILK-only streams. These require identical 16-bit samples. |
| `silkenc_` | 6 | 25 | The SILK encoder: control, FEC data, mono coding of stereo input, pulses, restarts, robustness of its packet functions. |
| `hybrid_` | 4 | 19 | Hybrid mode encoding and decoding. |
| `celt_` | 8 | 17 | CELT round trips and regressions. |
| `mdct_` | 16 | 17 | The modified discrete cosine transform (MDCT): identity, gain and round trips. |
| `verify_` | 5 | 16 | Band, PVQ (pyramid vector quantization), energy and SILK results against values from small C reference programs. |
| `pvq_` | 4 | 7 | The PVQ codebook: its tables, its signs, and agreement between encoder and decoder. |
| other | 19 | 88 | Single files, for example `stack_usage_test.rs` (struct sizes and stack use), `panic_surface_repro.rs` (inputs that used to panic), `fuzz_toc_sequences.rs`, `proptest_encode_indices.rs` and `oracle_celt.rs`. |

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

### Vector kernels against scalar code

The vector kernels have unit tests that compare them with scalar versions of the same code, bit for bit where the two must agree. Examples from `src/`:

- `kiss_fft::tests::test_neon_butterflies_match_scalar`
- `mdct::bit_exact_tests::neon_fold_matches_scalar`
- `pitch::tests::neon::xcorr_by_sample_neon_matches_scalar`
- `celt::comb_inplace_tests::const_part_neon_matches_portable_version`

Run them with `cargo test --release --lib`. The NEON tests run only on 64-bit ARM. The x86-64 vector code was only compiled in this release, not run (see [Differences from libopus](10-differences-from-libopus.md#platforms)).

## Checks against the fixed-point build of libopus

The SILK encoder of opus1 is a port of the fixed-point SILK encoder of libopus (`silk/fixed`). The linked libopus is a floating-point build, so it does not contain that encoder. The fixed-point comparisons therefore use values recorded from a separate build:

1. The libopus 1.6.1 C source is compiled with `cc` and the defines `-DFIXED_POINT -DOPUS_BUILD -DVAR_ARRAYS -DHAVE_LRINTF -DHAVE_LRINT`.
2. A small C program drives it: either `opus_encode` with the controls a test names, or one internal function (for example `silk_burg_modified_c`).
3. Its output is recorded into the test files:
   - packet bytes as hexadecimal text (`tests/silk_fixed_point_compare.rs`, 280 packets),
   - a 32-bit FNV-1a (Fowler-Noll-Vo) hash of every packet (`tests/core_fixed_point.rs`),
   - function results in `tests/perfk_silk_burg.expected`, `tests/perfk_silk_nsq_plain.expected`, `tests/perfk_silk_pitch_core.expected` and `tests/perfk_silk_process_nlsfs.expected`.
4. The C program and the Rust test make their input with the same integer-only generator. Each recorded row carries a checksum of its input, so a test can tell that both used the same samples.

The fixed-point build and its driver program are kept outside git. Only the recorded values are in the repository, so the tests run without the C build.

These tests require identical results: the same packets, or the same integers. In the recorded streams, the SILK-only packets of opus1 with the `Voip` application and 16-bit input (`OpusEncoder::encode_i16`) are byte for byte those of the fixed-point build. [Differences from libopus](10-differences-from-libopus.md#silk) lists the cases where the packets of longer streams start to differ.

## Fuzzing

`fuzz/` is a separate package (`opus1-fuzz`) of 18 targets for libFuzzer, the coverage-guided fuzzing engine of LLVM. libFuzzer calls a target with generated inputs. It keeps the inputs that reach new code and changes them further. The targets build and run on the stable toolchain:

```text
fuzz/fuzz.sh list                       # the targets
fuzz/fuzz.sh fuzz_decoder_sequence 600  # fuzz one target for 600 seconds
fuzz/fuzz.sh all 300                    # every target, 300 seconds each
```

`fuzz/fuzz.sh` builds the targets with cargo and the coverage options that cargo-fuzz uses without a sanitizer. Debug assertions and overflow checks are on, so an integer overflow stops a run like a panic. The corpus of each target is kept in `fuzz/corpus/<target>/`. An input that fails is written to `fuzz/artifacts/<target>/`. `fuzz/README.md` gives the build command, how to reproduce and minimize a failure, and the `cargo +nightly fuzz` command, which still works.

Five targets compare opus1 with libopus, which is linked as for the tests:

| Target | What must hold |
|---|---|
| `fuzz_packet` | The `packet` module gives libopus's results: `parse` (plain and self-delimited), `nb_frames` and `nb_samples` against `opus_packet_parse_impl` and `opus_packet_get_nb_*`, `pad` and `multistream_pad` against `opus_packet_pad` and `opus_multistream_packet_pad` byte for byte, and `write` against the libopus repacketizer byte for byte. `multistream_validate` accepts what libopus's multistream decoder accepts. Written and padded packets parse to the frames they were made of. |
| `fuzz_decoder_sequence` | A stream of packets into `OpusDecoder` and into libopus's decoder. The packets come from a libopus encoder whose settings change (2.5 to 120 ms, FEC, discontinuous transmission, forced modes and bandwidths), or from the input. Some are damaged or lost. The calls are `decode`, `decode_i16`, `decode_fec` and `decode_fec_i16`, on decoders of either channel count. After every call both decoders accept or refuse alike, return the same number of samples, and end in the same final range. While no packet was lost or damaged, the output is within 50 dB of libopus's. opus1's output buffer sometimes has more or less room than the frame size. More room changes nothing. With too little room opus1 may refuse, and then it keeps its state. |
| `fuzz_multistream_decoder` | A layout and packets into `MultistreamDecoder` and into libopus's `opus_multistream_decoder`. Both accept or refuse the layout alike. After every call they accept or refuse alike, return the same number of samples, and end in the same final range. While no packet was lost or damaged, the output is within 50 dB of libopus's. opus1's output buffer sometimes has more or less room than the frame size. More room changes nothing. With too little room opus1 may refuse, and then it keeps its state. A decoder built from the layout as the channel mapping table of an Ogg Opus header is accepted too. |
| `fuzz_multistream_encoder` | `MultistreamEncoder` with a layout, settings and audio from the input. libopus accepts and refuses the same layouts, and picks the same streams and mapping for a mapping family. Every packet decodes in libopus's multistream decoder to one frame, in the encoder's final range. Until a frame is concealed, opus1's decoder gives output within 50 dB of libopus's. A second encoder with the same settings, given `f32` samples where the first gets the same audio as `i16`, writes the same packets. One sample too few is refused, and the encoder keeps its state. |
| `fuzz_encoder_settings` | `OpusEncoder` with settings that change between frames: bitrate, variable or constant bitrate, complexity, forced mode, bandwidth, forced channel count, FEC, loss percentage, the low-frequency-effects (LFE) flag, energy mask, prediction and frame size. Every valid frame is encoded. Every packet decodes in libopus's decoder to one frame, in the encoder's final range. Until a frame is concealed, opus1's decoder gives output within 50 dB of libopus's. A second encoder with the same settings, given `f32` samples where the first gets the same audio as `i16`, writes the same packets. One sample too few is refused, and the encoder keeps its state. |

The output is compared only while both decoders have decoded the same packets in full. opus1's decoder differs from libopus's only in rounding, so 50 dB leaves a wide margin. After a lost or damaged packet the output is no longer compared. The concealment searches the decoded signal for its pitch, and a rounding difference can change what it finds. A frame of 0 or 1 bytes is concealed as well, so it also ends the comparison. So does a packet that switches from CELT-only to SILK or Hybrid. Unless such a packet carries a redundant CELT frame, the decoder conceals 5 ms in CELT and fades them into the new frame. The targets do not look for the redundant frame, so they stop comparing after every such switch. The fuzzer found a switch without a redundant frame at 8 kHz. On the CELT packets before it, the two decoders agreed to 127 dB or better. On the 20 ms SILK frame of the switch they agreed to 33 dB. The difference lay in the first 5 ms of that frame, which hold the concealed CELT audio. There the decoders agreed to 27 dB, and on the other 15 ms to 294 dB.

The other 13 targets (`fuzz_encoder`, `fuzz_decoder`, `fuzz_roundtrip`, `fuzz_silk_encoder`, `fuzz_celt_encoder`, `fuzz_overflow`, `fuzz_celt_large_frame`, `fuzz_celt_decoder_boundary`, `fuzz_silk_decoder`, `fuzz_hybrid_mode`, `fuzz_encoder_boundary`, `fuzz_multi_frame`, `fuzz_packet_parsing`) run opus1 alone, with extreme settings, frame sizes and packets. Nothing may panic. Those that encode and decode also check that the decoder returns one frame and ends in the encoder's final range.

When fuzzing finds a failure in opus1, the fix follows what libopus does. The minimized input becomes a test in `tests/fuzz_regressions.rs`, as a byte array in the test, so the test needs no file. The runs below found no such failure, so that file does not exist yet.

Two files of the normal test suite also try many generated inputs. `tests/fuzz_toc_sequences.rs` feeds one decoder long, repeatable sequences of packets with varied TOC bytes. `tests/proptest_encode_indices.rs` uses the `proptest` crate, which generates random inputs and reduces a failing one to a small example.

### Fuzzing results for this release

All 18 targets ran on Apple M4 Pro with the build of `fuzz/fuzz.sh`, so with overflow checks and debug assertions. Several targets ran at the same time, one process each, unless the table says otherwise.

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

## `#![no_std]` checks

Without its default features, opus1 builds as `#![no_std]` with no heap allocation (see the [README](../README.md)). `scripts/check_no_std.sh` checks this:

1. It builds with `--no-default-features --features libm` for `thumbv7em-none-eabi` (ARM Cortex-M) and `wasm32-unknown-unknown` (WebAssembly). A target that is not installed is skipped with a message. Add it with `rustup target add <target>`.
2. It builds the same configuration for the host.
3. It runs `cargo test --no-default-features --features libm` on the host with `RUST_MIN_STACK=33554432` (32 mebibytes). Without the `heap` feature, the encoder and decoder hold all their buffers inline: 135,192 and 67,432 bytes on aarch64 (see [Size of the encoder and the decoder](06-using-the-library.md#size-of-the-encoder-and-the-decoder)). So test threads need a large stack.

The multistream module needs `std`. The multistream test files that use it (30 of 31) are compiled only with the `std` feature, so they are left out of this run. `tests/stack_usage_test.rs` checks the sizes for both builds: small structs with `heap`, large ones without it.

## Ignored tests

24 tests are marked `#[ignore]`, and one documentation example is not run.

- 22 print report tables or are tools, not checks: `celtenc_bands_libopus` (2), `celtenc_gap_report` (3), `celtenc_prefilter` (1), `k24_enc_report` (15) and `perfk_rc_timing` (1).
- 2 describe known gaps, both caused by the missing speech and music analysis:
  - `oracle_celt::impulse_48k_mono_64k_is_the_packet_of_libopus`
  - `multistream_enc_signals::average_rate_at_16_khz_and_for_voip_is_within_the_design_limits`

  [Differences from libopus](10-differences-from-libopus.md) gives their figures.

- 1 documentation example is marked `rust,ignore`: the fields of `OpusEncoder` with and without the `heap` feature in [Architecture](07-architecture.md#feature-flags-and-where-state-is-stored). It shows part of a struct, so it cannot run on its own.

## Documentation examples

`src/lib.rs` includes `README.md` and each file of `docs/` that has Rust examples. Each file becomes the documentation of a private item under `#[cfg(all(doctest, feature = "std"))]`. So `cargo test --doc` compiles and runs every `rust` block of these files. An example must be a complete program fragment that uses `unwrap` or `expect`. Hidden lines (lines that start with `# `) are not used, because GitHub would show them. The examples need the `std` feature, because some use the `multistream` module. They also run without `heap`: `cargo test --doc --no-default-features --features std`.

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
| `build-libopus.sh` | Builds libopus as a static library with the NDK's clang. It uses the settings that libopus's own `configure` picks for this target: `-O2`, NEON intrinsics presumed, run-time detection for the DotProd code. It writes every flag to `target/android/libopus/build-flags.txt`. |
| `test.sh` | Builds every test program with `cargo ndk -t arm64-v8a test --release --no-run` against that libopus. It copies the programs and `fixtures/` to the phone and runs them one after the other. Then it prints one line per program and the totals. |
| `golden.sh` | Writes the golden hashes on the phone. With `--check FILE` it also checks them against a golden file from another machine. |
| `bench.sh` | Runs `perf_harness bench`, pinned with `taskset` to one CPU of the fast cluster and one of the slow cluster (see [Performance](08-performance.md#on-an-android-phone)). |
| `profile.sh` | Counts hardware events per frame with `simpleperf stat` and samples the hot functions with `simpleperf record`, for one operation of the harness. |

`env.sh` holds the settings that all scripts share. The commands of a first run:

```text
export ANDROID_NDK_HOME=$HOME/Library/Android/sdk/ndk/27.3.13750724
scripts/android/build-libopus.sh
scripts/android/test.sh
scripts/android/golden.sh --check target/perf/golden_final.txt
scripts/android/bench.sh --runs 2
scripts/android/profile.sh --little celt_fb_48k_mono_64k_vbr_c10_20ms enc rust
```

Some details:

- The tests open `fixtures/answer_16k.wav` through a path that is fixed when they are compiled (`env!("CARGO_MANIFEST_DIR")`). While cargo compiles opus1, `rustc-wrapper.sh` sets that variable to `DEVICE_DIR`. The scripts copy `fixtures/` there.
- Each test program runs with `RUST_MIN_STACK=33554432`, as on macOS.
- The documentation tests do not run on the phone. Stable Rust cannot run them for another target.
- The programs run on the phone detached from adb (`setsid`). A dropped USB connection does not stop them. `test.sh --collect` waits for such a run and collects its logs.
- `test.sh` copies the logs of each run to a new directory, `target/android/test-logs/<date>-<time>/`, so a later run does not delete them. `test-logs/latest` points at the newest one.

### Which libopus the phone links

`build-libopus.sh` builds, by default, the libopus source that the crate `opus-head-sys` 0.3.3 vendors. That source is commit `228a0f85` of the libopus main branch, from 2026-08-28. It is newer than the release 1.6.1, which the tests were written against and which Homebrew ships. No copy of the 1.6.1 source was available for the first run. Set `OPUS_SRC` to an unpacked 1.6.1 release to link the same version as on macOS.

### Results of the first run

Run on 2026-10-16 at git commit `bd080de`: 229 test programs with 1406 tests. 1372 passed, 10 failed and 24 were ignored. Two runs took 580 and 591 seconds, with the tests already built (129 seconds on Apple M4 Pro). A check run later that day, from a new build directory and a new directory on the phone, gave the same totals and the same 10 failures in 578 seconds.

None of the 10 failures comes from opus1. On the phone, opus1 writes the same packets and decodes the same samples as on Apple M4 Pro (see the golden check below). Each failing test compares opus1 with libopus, or makes its input with the C library's sine function. The difference is on that side:

| Failing tests | Cause | How it was found |
|---|---|---|
| 8 tests of LFE (low-frequency effects) streams and mode switches: `celtenc_known_gaps` (2), `celtenc_lfe`, `celtenc_lfe_formats`, `celtenc_lfe_header`, `multistream_e2e_matrix` (2) and `core_redundancy` | The libopus source. The main-branch libopus codes these streams differently from 1.6.1. | The same 8 tests fail on Apple M4 Pro when they link the same source, built there with Apple clang at `-O2` or `-Os`. On the phone, the opus1 packets of the 50 Hz LFE tone of `celtenc_known_gaps` are byte for byte those of the Mac. libopus differs from the first packet on. No build setting of that source writes the packets of the Homebrew 1.6.1: on the Mac, `golden --check` against `golden_final.txt` found the same 86 streams different at `-O2`, at `-Os` and without `FLOAT_APPROX`, and 100 without the NEON code. |
| `celtenc_bands_libopus::stereo_angles_are_the_ones_libopus_computes` | Vector code in libopus. At `-O2`, clang turns the loop of `stereo_itheta` (celt/vq.c) that sums `m*m` and `s*s` into vector code. The vector code rounds each product before it adds it. The scalar code does both in one fused multiply-add. So the last bit of the angle can differ. The Homebrew library has no such vector loops. | The test passes when libopus is built with `-fno-vectorize -fno-slp-vectorize`. On the Mac it fails with an `-O2` build of the same source and passes with an `-Os` build. |
| `issue_15_loss_concealment::silk_concealment_extrapolates_the_signal` | The test signal. It is made with `f32::sin`, which calls the C library's `sinf`. Android's and Apple's `sinf` give different last bits for some inputs. With the phone's signal, the concealed 48 kHz frame has 0.0093 of the energy of the last real frame. The test requires more than 0.01. | Of 2,000,000 inputs, Apple's `sinf` differs from the sine computed in double precision and then rounded in 151,818 cases, Android's in 25,754. A copy of the test, run on the Mac with that rounded double-precision sine, got exactly the phone's numbers. libopus's decoder, given the same packets, also conceals the frame at 0.0093 of the energy. |

The 10 failures are at `tests/celtenc_known_gaps.rs:165` and `:213`, `tests/celtenc_lfe.rs:56`, `tests/celtenc_lfe_formats.rs:204`, `tests/celtenc_lfe_header.rs:236`, `tests/multistream_e2e_matrix.rs:316` (2 tests), `tests/core_redundancy.rs:241`, `tests/celtenc_bands_libopus.rs:165` and `tests/issue_15_loss_concealment.rs:304`.

`golden.sh --check target/perf/golden_final.txt` found 86 of 137 streams that differ. In all 86, only fields that depend on libopus's packets differ: `c_pk`, `dec_c`, `dec_c_plc` and `dec_c_burst`, and in 7 of them also `fec_silk`. The fields of opus1's own output (`enc`, `enc_i16`, `dec_rust`, `fec_frames` and `errs`) are equal in all 137 streams. The same 86 streams differ on the Mac when the harness links the same libopus source. The golden file of the phone is identical to that Mac golden file in every field.

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
| Tests on an Android phone | `scripts/android/build-libopus.sh`, then `scripts/android/test.sh` |
| Golden hashes on a phone | `scripts/android/golden.sh --check <file>` |
| Timings on a phone | `scripts/android/bench.sh --runs 2` |
| Hardware counters and hot functions on a phone | `scripts/android/profile.sh --little <scenario> enc rust` |

Run the tests with `--release`. The tests encode and decode seconds of audio per case. All timings in this document are for release builds.

## See also

- [Performance](08-performance.md): the harness and its timing commands.
- [Differences from libopus](10-differences-from-libopus.md): what the tests show to differ.
- [Architecture](07-architecture.md): where the code under test lives.
- [Glossary](glossary.md): the terms used in these documents.
- The documentation index: [README.md](README.md).
