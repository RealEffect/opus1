# Performance

This document explains how the speed of opus1 is measured against the C library libopus. It gives the latest results and the changes that made the code fast. It also lists the ideas that did not help, the build settings an application can use, and the commands to reproduce the numbers. All numbers were measured on Apple M4 Pro (64-bit ARM), except those of [On an Android phone](#on-an-android-phone). Each section names the record and the state of the code that its numbers come from. The records are in `target/perf/`, a folder of the maintainers' working tree that is not in git.

## Terms used here

- **Opus** is the audio codec defined in RFC 6716 (Request for Comments 6716, the Internet standard for Opus).
- **libopus** is the reference implementation of Opus, written in C by Xiph.Org. opus1 follows libopus 1.6.
- **SILK** is the speech coder inside Opus. It is based on linear predictive coding (LPC), which predicts each sample from the samples before it.
- **CELT** (Constrained Energy Lapped Transform) is the transform coder inside Opus. It codes the spectrum of the signal, computed with the modified discrete cosine transform (MDCT).
- **Hybrid** is the Opus mode in which SILK codes the audio below 8 kilohertz (kHz) and CELT codes the audio above it.
- A **frame** is the block of audio that one encoder call codes: 2.5, 5, 10, 20, 40 or 60 milliseconds (ms).
- **Complexity** is an encoder setting from 0 to 10. Higher values spend more processor time on searches that improve quality.
- **VBR** (variable bitrate) lets the packet size change from frame to frame. **CBR** (constant bitrate) keeps every packet the same size.
- **NEON** is the 128-bit vector instruction set of 64-bit ARM processors. One vector instruction works on four 32-bit values at once.
- The MDCT is computed with a **fast Fourier transform (FFT)**.
- A **fused multiply-add (FMA)** computes `a * b + c` with a single rounding step.
- The **median** of a list of times is the middle value after sorting. It is less affected by a few slow runs than the mean.
- Times are given in nanoseconds (ns) and microseconds (µs). Bitrates are in kilobits per second (kb/s). Memory sizes are in kilobytes (KB, 1024 bytes).

## How speed is measured

### The harness

The program `examples/perf_harness.rs`, called the harness below, measures opus1 and libopus in the same process. It links libopus through the development dependency `audiopus_sys` (see [Testing](09-testing.md#how-libopus-is-linked)). Run it with:

```text
cargo run --release --example perf_harness -- <command> [options]
```

The commands are:

| Command | What it does |
|---|---|
| `list` | Prints every scenario with its settings. |
| `bench` | Times four operations per scenario: opus1 encode, libopus encode, opus1 decode and libopus decode. |
| `run` | Repeats one operation in a loop (10 s by default), so that a sampling profiler can attach to the process. |
| `golden` | Writes, or checks, hashes of everything the codec produces. The file of hashes is called a golden file. It tells whether a change altered the output. |
| `modes` | Compares the opus1 and libopus packet streams, and the two decoders, scenario by scenario. |
| `settings` | Encodes every scenario and prints the encoder settings both libraries report afterwards. libopus is asked with its `OPUS_GET_*` requests. |
| `signals` | Prints the level and the spectrum of every test signal. `--dump DIR` writes them as WAV (Waveform Audio File Format) files. |
| `resampler` | Checks the resampler that makes the speech signals, using sine tones. |
| `spread` | Compares several `bench --json` files. |

`cargo run --release --example perf_harness -- help` prints every option.

### Scenarios and their names

A scenario is one fixed set of encoder settings and one test signal. The harness has 129 scenarios. The `bench` command times 28 of them by default (the "bench set"). `--all` times all of them. The `golden` command uses all 129, plus 8 streams that switch between modes. So a golden file has 137 entries after its header line.

A scenario name lists its settings. For example, `hybrid_fb_48k_stereo_40k_cbr_c9_20ms_music` means:

| Part | Meaning |
|---|---|
| `hybrid` | Coding mode: `silk`, `hybrid` or `celt`. Both encoders are forced to it. |
| `fb` | Audio bandwidth, chosen from the input rate: `nb` narrowband (8 kHz input), `mb` mediumband (12 kHz), `wb` wideband (16 kHz), `swb` super-wideband (24 kHz), `fb` fullband (48 kHz). |
| `48k` | Input sample rate in kHz. |
| `stereo` | Channel count: `mono` or `stereo`. |
| `40k` | Target bitrate in kb/s. |
| `cbr` | Rate control: `vbr` or `cbr`. |
| `c9` | Complexity. |
| `20ms` | Frame duration (`2p5ms` is 2.5 ms). |
| `_music` | Optional suffixes for settings that differ from the default. They name the signal (`_speech`, `_music`), the application (`_voip`, `_audio`, `_lowdelay`), in-band forward error correction with an expected loss (`_fec10`), pauses of digital silence (`_pauses`), and a decoder output rate that differs from the input rate (`_dec16k`). |

By default, CELT scenarios use the `Audio` application and a synthetic music signal. SILK and Hybrid scenarios use the `Voip` application and speech. The speech comes from `fixtures/answer_16k.wav` (16 kHz mono) and is resampled to the other rates. The harness generates the music: five voices of harmonic notes, a bass drum, a snare and noise. Each voice has its own position between left and right.

### Both encoders do the same kind of work

The C encoder is forced to the mode, the audio bandwidth and the channel count that the scenario names. libopus is set with `OPUS_SET_FORCE_MODE`, `OPUS_SET_BANDWIDTH` and `OPUS_SET_FORCE_CHANNELS`. opus1 is set with `OpusEncoder::set_force_mode`, `set_bandwidth` and `set_force_channels`.

The reason is that an encoder that chooses its own mode can pick a different one than the other encoder. For example, one might code a frame with SILK while the other codes it with CELT. Their times would then measure different work. The mode-switching streams of `golden` and `modes` are the exception: there only the channel count is forced.

Two differences remain, and the harness reports them:

1. The libopus that the harness links is a floating-point build. Its SILK encoder is the floating-point one (`silk/float`). The SILK encoder of opus1 is a port of the fixed-point one (`silk/fixed`).
2. libopus runs its speech and music analysis (`run_analysis` in `analysis.c`) at complexity 7 and above when the input rate is 16 kHz or more. opus1 has no port of it. The C encode time of such a scenario includes the analysis, so its encode ratio favours opus1. The `bench` table names this step in the column `only C runs`.

Both decoders decode the same packets: the packets that libopus encoded.

### Repetitions, processes and medians

One repetition is one pass over the whole clip (6 s by default). The time per frame is the time of the pass divided by the number of frames.

1. Each measuring process first runs one warm-up repetition, which is not counted. It then runs 5 counted repetitions.
2. Inside each repetition, the four operations run one after the other. opus1 and libopus alternate, and the order within each pair is reversed on every repetition. So slow periods of the machine affect both libraries equally.
3. Each scenario is measured in 3 fresh processes, so one "round" has 15 repetitions per operation. A fresh process makes the result independent of the scenarios measured before. Several processes cover effects that are fixed for the life of a process, such as the random addresses the operating system gives to code and data. Both effects were seen to change a result by up to 10 %.
4. A round is "quiet" when the median of every operation is at most 3 % above its minimum. Otherwise another program disturbed the round, and the scenario is measured again, up to 3 rounds in total.
5. The table reports, per operation, the lowest minimum of all rounds and the lowest median of any one round.

On macOS, the measuring threads ask for the scheduling class `QOS_CLASS_USER_INTERACTIVE`, which asks the system to run them on a performance core. Medians repeat better than minimums on a machine that other programs use, so the ratios below are ratios of medians.

### Code placement

The speed of some functions depends on the address at which the linker places them. On Apple M4 Pro, the same machine code of `silk_warped_autocorrelation_fix` was measured in a fast and a slow state, about 1.7 times apart. This changes the SILK encode time at complexity 5 by 11 to 17 %. Any change to the library or to the harness can move the code. So a difference of that size between two builds may have nothing to do with the change.

To tell the two apart, build the harness with the environment variable `PERF_HARNESS_PAD=<bytes>`. It inserts that many bytes of padding before the library code (macOS only). Then compare the results of several paddings with `spread`:

```text
for pad in 0 20 40 60; do
  PERF_HARNESS_PAD=$pad cargo run --release --example perf_harness -- \
    bench --json target/perf/pad_$pad.json
done
cargo run --release --example perf_harness -- spread target/perf/pad_*.json
```

## Latest results

Measured on 2026-10-16 at git commit `b63e8c3` (opus1 0.2.0), after the refactoring described in [Effect of the refactoring](#effect-of-the-refactoring). Machine: Apple M4 Pro, macOS. C reference: libopus 1.6.1, the prebuilt Homebrew library (a floating-point build). opus1 was built with the default release profile. Source: `target/perf/release_bench.md`.

Other programs on the machine kept the load average near 4 during the measurement. The bench set was run three times, and this is the run with the fewest operations marked unsteady and the fewest repeated rounds. Across the three runs, the median time of an operation varied by up to 8.5 %. The opus1 / libopus ratio of the medians varied by at most 2.1 % for encode and 2.9 % for decode, because the two libraries run interleaved and a busy machine slows both (`perf_harness spread`).

Times are the median per frame in microseconds (µs). The ratio is opus1 / libopus. A ratio below 1 means opus1 is faster.

| Scenario | opus1 encode µs | libopus encode µs | Encode ratio | opus1 decode µs | libopus decode µs | Decode ratio | Only C runs |
|---|---:|---:|---:|---:|---:|---:|---|
| `silk_nb_8k_mono_12k_vbr_c0_20ms` | 9.0 | 14.5 | 0.620 | 2.0 | 3.2 | 0.618 | - |
| `silk_nb_8k_mono_12k_vbr_c5_20ms` | 20.2 | 31.9 | 0.632 | 2.1 | 3.3 | 0.628 | - |
| `silk_nb_8k_mono_12k_vbr_c10_20ms` | 26.1 | 57.7 | 0.453 | 2.0 | 3.3 | 0.618 | - |
| `silk_wb_16k_mono_20k_vbr_c0_20ms` | 14.8 | 24.8 | 0.595 | 3.5 | 6.1 | 0.575 | - |
| `silk_wb_16k_mono_20k_vbr_c5_20ms` | 35.0 | 58.1 | 0.603 | 3.8 | 6.5 | 0.582 | - |
| `silk_wb_16k_mono_20k_vbr_c10_20ms` | 45.9 | 119.0 | 0.386 | 3.8 | 6.6 | 0.583 | analysis |
| `silk_wb_16k_mono_20k_vbr_c9_10ms` | 25.3 | 63.2 | 0.400 | 2.0 | 3.4 | 0.591 | analysis |
| `silk_wb_16k_stereo_32k_vbr_c9_20ms` | 91.2 | 226.1 | 0.403 | 6.8 | 12.0 | 0.571 | analysis |
| `silk_wb_16k_mono_24k_vbr_c9_20ms_fec10` | 60.5 | 157.7 | 0.384 | 4.3 | 7.1 | 0.610 | analysis |
| `hybrid_swb_24k_mono_24k_cbr_c9_20ms` | 63.7 | 166.6 | 0.382 | 7.2 | 13.4 | 0.534 | analysis |
| `hybrid_swb_24k_stereo_36k_cbr_c9_20ms` | 125.5 | 312.5 | 0.402 | 12.0 | 23.1 | 0.518 | analysis |
| `hybrid_fb_48k_mono_32k_cbr_c9_20ms` | 67.1 | 158.3 | 0.424 | 8.3 | 16.1 | 0.515 | analysis |
| `hybrid_fb_48k_stereo_40k_cbr_c9_20ms` | 132.6 | 320.1 | 0.414 | 13.5 | 27.0 | 0.502 | analysis |
| `celt_fb_48k_mono_64k_vbr_c0_20ms` | 12.2 | 20.3 | 0.602 | 8.4 | 11.1 | 0.757 | - |
| `celt_fb_48k_mono_64k_vbr_c5_20ms` | 21.6 | 45.0 | 0.479 | 9.8 | 14.8 | 0.665 | - |
| `celt_fb_48k_mono_64k_vbr_c10_20ms` | 21.8 | 56.2 | 0.388 | 10.0 | 15.3 | 0.655 | analysis |
| `celt_fb_48k_stereo_128k_vbr_c0_20ms` | 23.6 | 41.4 | 0.571 | 16.7 | 21.0 | 0.794 | - |
| `celt_fb_48k_stereo_128k_vbr_c5_20ms` | 37.5 | 76.8 | 0.488 | 19.7 | 30.1 | 0.657 | - |
| `celt_fb_48k_stereo_128k_vbr_c10_20ms` | 58.6 | 121.5 | 0.482 | 19.8 | 29.8 | 0.664 | analysis |
| `celt_fb_48k_stereo_128k_vbr_c9_2p5ms` | 11.5 | 18.7 | 0.616 | 3.2 | 4.2 | 0.766 | analysis |
| `celt_fb_48k_stereo_128k_vbr_c9_5ms` | 19.2 | 34.5 | 0.558 | 5.8 | 7.8 | 0.735 | analysis |
| `celt_fb_48k_stereo_128k_vbr_c9_10ms` | 29.4 | 58.6 | 0.501 | 10.4 | 15.1 | 0.685 | analysis |
| `celt_swb_24k_mono_48k_vbr_c9_20ms` | 18.2 | 45.8 | 0.398 | 8.0 | 11.3 | 0.704 | analysis |
| `celt_swb_24k_stereo_96k_vbr_c9_20ms` | 51.0 | 102.3 | 0.498 | 15.0 | 23.0 | 0.650 | analysis |
| `celt_fb_48k_mono_24k_vbr_c9_20ms` | 16.4 | 47.3 | 0.347 | 5.9 | 11.1 | 0.527 | analysis |
| `celt_fb_48k_stereo_256k_vbr_c9_20ms` | 69.9 | 123.9 | 0.564 | 28.8 | 35.0 | 0.822 | analysis |
| `hybrid_fb_48k_mono_32k_cbr_c9_20ms_music` | 59.1 | 140.1 | 0.422 | 8.3 | 16.2 | 0.510 | analysis |
| `hybrid_fb_48k_stereo_40k_cbr_c9_20ms_music` | 117.4 | 275.7 | 0.426 | 13.4 | 26.8 | 0.499 | analysis |

How to read the table:

- **Decoding** takes 0.499 to 0.822 of libopus's time in every scenario. Both decoders do the same work, so this is a like-for-like comparison.
- **Encoding at complexity 0 and 5** takes 0.479 to 0.632 of libopus's time. At these settings libopus runs none of the large encoder steps that opus1 lacks.
- **Encoding at complexity 9 and 10** shows ratios of 0.347 to 0.616. Part of the gap is the speech and music analysis that only libopus runs (`analysis` in the last column).
- SILK encode ratios compare the fixed-point SILK encoder of opus1 with the floating-point one of libopus.
- The opus1 encode time of the two SILK complexity-5 rows depends on code placement (see [Code placement](#code-placement)). Two states about 16 % apart were measured.
- Five operations were marked unsteady in this run. They are the opus1 decode of `silk_nb_8k_mono_12k_vbr_c0_20ms`, `silk_nb_8k_mono_12k_vbr_c10_20ms`, `silk_wb_16k_mono_20k_vbr_c0_20ms` and `silk_wb_16k_mono_20k_vbr_c5_20ms`, and the libopus decode of `silk_wb_16k_mono_20k_vbr_c10_20ms`. For them the median is the number to use, and the table above uses medians.

### Effect of the refactoring

A refactoring merged duplicate code paths (merge commit `419d5bb`), for example into one `CeltEncoder::encode` that reads interleaved input. The output did not change. At `b63e8c3`, `perf_harness golden --check` against the golden file written at commit `5c8e05e` (`target/perf/golden_final.txt`) finds 0 of 137 entries that differ.

The bench set was also measured at `5c8e05e`, before the refactoring (`target/perf/final_bench.md`). Besides the refactoring, one more change to the code lies between the two commits: `bits2pulses` and `pulses2bits` now check their table indexes (commit `49c2bb4`). That commit measured -1.5 % to +0.4 % for encode and -2.1 % to +0.9 % for decode on the 13 CELT scenarios, within the noise of the machine. From `5c8e05e` to `b63e8c3`, the opus1 / libopus ratio of the medians changed as follows:

- CELT encode: -1.3 % to -14.0 %. The change is -12.8 % (mono) and -14.0 % (stereo) at complexity 0, and -6.0 % to -8.6 % at complexity 5 and 10.
- Decode: -2.4 % to +0.7 %.
- SILK and Hybrid encode: -1.2 % to +0.3 %.

The changes of decode and of SILK and Hybrid encode are within the variation between runs given in [Latest results](#latest-results). A measurement right after the merge, at commit `a9c1a54` (`target/perf/simplify_bench.md`), found -1.6 % to -12.0 % for CELT encode.

### Against opus-rs 0.1.37

The same harness measured the code before the performance work: git commit `08214a8` on 2026-10-13, same machine and same libopus (`target/perf/baseline_bench.md`). The table compares the opus1 / libopus ratio of the medians in the two records. libopus is the same build in both, so its time changes only with the state of the machine. The change of the ratio is therefore the change of the opus1 time, with most of the effect of other programs removed.

| Group | Change of opus1 time against libopus time |
|---|---|
| SILK and Hybrid encode (15 scenarios) | 32.7 % to 55.5 % less |
| Mono CELT encode (5 scenarios) | 29.3 % to 49.1 % less |
| Stereo CELT encode, complexity 0 and 5 (2 scenarios) | 23.8 % and 30.0 % less |
| Stereo CELT encode, complexity 9 and 10 (6 scenarios) | 2.5 % to 21.6 % more |
| Decode (all 28 scenarios) | 35.3 % to 62.3 % less |

Stereo CELT encoding at complexity 9 and 10 takes more time because the encoder now does more of libopus's work. Since 0.1.37 it runs the pitch pre-filter on stereo input (complexity 5 and above) and `theta_rdo` (complexity 8 and above). The commit that added `theta_rdo` measured +34 % to +73 % encode time on the six stereo bench scenarios at complexity 9 and 10. Its ratio to libopus stayed between 0.53 and 0.63. In [Latest results](#latest-results), these six scenarios take 0.482 to 0.616 of libopus's time.

In 0.1.37 decoding was slower than libopus in every scenario (ratio 1.033 to 1.333). It is now faster in every scenario.

## What made the code fast

Every change below keeps the codec output bit for bit the same, unless the text says otherwise. That was checked with `perf_harness golden --check` (137 streams). The measured effects come from the commit messages and from the profile reports in `target/perf/prof/`. Each is the effect of one change, measured against the code just before it.

### Loops in which each step no longer waits for the one before

A modern processor can run several independent operations at the same time. A loop is slow when each step needs the result of the step before, because the processor then waits. Several loops that take much of the codec's time were rewritten so that independent work runs at the same time.

- **SILK noise-shaping quantizer.** The noise-shaping quantizer (NSQ) chooses the quantized excitation of SILK. Its delayed-decision version keeps 2 to 4 candidate states and took 31 % to 60 % of the SILK encode time at complexity 5 and above. Its filter is a chain of 12 to 24 dependent steps per state, and the states were processed one after the other. Now each filter step is computed for all states before the next step, so the chains of the states run at the same time (commit `466f79e`). SILK and Hybrid encode at complexity 9 and 10 took 13.4 % to 23.5 % less time.
- **CELT de-emphasis.** De-emphasis is the last filter of the CELT decoder. Each output sample needs the one before it. The frame is now cut into eight segments that are filtered at the same time in two NEON vectors (commit `a9d3d06`). Each segment is checked against the end of the segment before it. When they do not match, the segment is computed again one sample at a time. Mono CELT and Hybrid decode took 6.6 % to 15.1 % less time. Before that, the two channels of a stereo frame were put into one loop (commit `7c835d9`): stereo decode took 4.2 % to 10.3 % less time.
- **SILK warped autocorrelation.** A NEON version keeps the filter state in registers (commit `2784fdd`). In isolation it is 3.9 to 5.3 times faster than the fast state of the old code, and it has the same speed at every code placement (`target/perf/prof/silk-enc/report.md`).

### Vector kernels that really use vectors

A kernel is a small function that does the inner loop of one step. A review of the processor (CPU) features (`target/perf/cpu_features.md`) found kernels that were named NEON or AVX (Advanced Vector Extensions, the vector instructions of x86-64) but ran scalar code, one value at a time. It also found kernels that added everything into one register, so each addition waited for the one before. They were replaced with kernels that keep several independent sums.

- **Pitch cross-correlation of the CELT pre-filter.** It now handles 64 lags per pass on 64-bit ARM (commit `6218f08`). One call of 240 samples by 244 lags went from 7921 ns to 1319 ns. CELT mono encode at complexity 5 and above took 15.7 % to 20.1 % less time.
- **CELT decoder post-filter.** The post-filter is a comb filter that restores the pitch structure. A NEON version computes four samples per step (commit `d539004`). Stereo CELT decode at complexity 10 took 4.6 % to 6.4 % less time.
- **FFT butterflies.** A butterfly is the basic step of an FFT that combines a few complex values. NEON butterflies for radix 3, 4 and 5, with one table of twiddle factors (the complex constants a stage multiplies by) per stage, made the FFT 1.55 to 1.92 times faster (commit `26572d7`). Over the 19 CELT and Hybrid bench scenarios, decode took 2.1 % to 8.0 % less time and encode 0.7 % to 5.4 % less.

### Complex-number instructions in the FFT

FEAT_FCMA is an optional ARMv8.3 extension with two complex-number instructions: `fcmla` (complex multiply-add) and `fcadd` (complex add with a rotation). The FFT of opus1 uses them on 64-bit ARM (commit `390ab40`). Their Rust intrinsics are not stable in Rust 1.98, so the code uses `core::arch::asm!`. On `aarch64-apple-darwin` (macOS on Apple silicon) FEAT_FCMA is part of the target, so there is no run-time check. Other 64-bit ARM targets detect it at run time with `std`. Without `std`, or on a processor without the feature, a plain NEON version with `fmla` runs.

This change also changed the output on 64-bit ARM on purpose. clang compiles libopus with fused multiply-adds where a product and a sum stand in one expression. The new butterflies fuse exactly those products, so the FFT now gives the same bits as the arm64 build of libopus. Measured in a micro-benchmark: the FFT is 18 % to 21 % faster and the 20 ms MDCT 21 % faster than before this commit.

### Less copying and zero-filling

The memory review (`target/perf/prof/memory-build/report.md`) found that the decoder moved 5 to 10 times more bytes through `memmove` and `memset` than libopus. For a 48 kHz stereo CELT frame that was 109.8 KB against 13.4 KB. These calls took 2.9 % to 5.9 % of the decode time. Examples of the fixes:

- The CELT decoder keeps its output history in a larger buffer. The start of the history moves forward by one frame, and the samples are moved back to the start of the buffer only when the history reaches its end. Before, all of the history was moved for every frame (commit `938478a`). 2.5 ms stereo decode took 4.1 % to 5.7 % less time.
- The CELT encoder keeps 120 samples of input history instead of 2168 (commit `6f155fb`). Stereo encode with 2.5 ms and 5 ms frames took 0.3 % to 3.9 % less time.
- The decoder writes its output through slices and fills transition buffers only on a mode switch (commit `22cc99d`). Buffers that are overwritten anyway are no longer cleared first.

### Removed computations whose results were not used

For stereo at complexity 8 and above, the CELT encoder turned the coded bands back into audio (`denormalise_bands` and two inverse MDCTs) into a buffer that nothing read. Removing this step (commit `25cb5df`) made stereo CELT encode take 5.0 % to 8.7 % less time in the affected scenarios. (The later port of `theta_rdo` needs the decoded bands inside `quant_all_bands_enc` again, but not the conversion back to audio.)

### Fewer bounds checks

Rust checks every slice index at run time unless the compiler can prove it is in range. Before the work, the 40 functions that took the most time contained 1,102 places that can panic, 570 of them inside loops. A check costs more than the compare itself when a loop keeps its state in slice elements. The compiler must then store that state back to memory before each check, because a panic could happen there.

- The pulse-vector decoder `cwrsi` turns an index back into the pulses of a band. It now checks once per call that every table read is in range, then reads the table as libopus does (commit `ca0c101`). In a replay of recorded calls, `cwrsi` took 17 % to 38 % less time. The opus1 / libopus decode ratio of the CELT scenarios fell by 1.5 % to 8.7 % (median over four code placements).
- The SILK 2x up-sampler read and wrote its six filter states through a slice, with a check on each access. Its states are now local variables, and on 64-bit ARM its two filter chains share one vector (commit `b4cc829`). Hybrid decode took 5.2 % to 8.2 % less time.

## What was tried and did not help

These results come from `target/perf/cpu_features.md` and the profile reports.

| Idea | Result |
|---|---|
| SME streaming mode | Apple M4 has the Scalable Matrix Extension (SME). A 16-bit dot product in streaming mode was 5 to 17 times slower than NEON for 80 to 480 samples. Entering and leaving streaming mode costs 9.9 ns per pair. |
| RDM instructions | FEAT_RDM (rounding doubling multiply) adds `sqrdmlah`, a rounding multiply-accumulate. SILK's `silk_SMLAWB` truncates instead of rounding, so 319 or 320 of 320 samples came out different. The baseline NEON instruction `sqdmulh` gives the exact result and is within 5 % to 13 % of the RDM speed. |
| DotProd instructions | FEAT_DotProd (dot product, `sdot`) takes 8-bit inputs only. SILK works on 16-bit samples. libopus itself uses these instructions only in its neural-network code. |
| `-C target-cpu=native` | On Apple M4 Pro it only adds `bf16`, `bti` and `i8mm`, which the codec does not use. FEAT_FCMA, RDM, DotProd and FP16 (16-bit floating point) are already part of the `aarch64-apple-darwin` target. Measured: encode 0.999, decode 1.001 of the default build's time. |
| Port of libopus's NEON delayed-decision quantizer | Not needed. On the same recorded inputs it took 38.4 to 39.5 µs per wideband frame; a plain-Rust reordering of the existing loop took 37.0 µs. |
| Faster NEON pulse search for bands of up to 16 coefficients | Two attempts brought nothing. |
| FFT stage tables aligned to 64 bytes | No change. |
| FMA in the comb filter | No consistent gain, and the output would differ. |

## Build settings for applications

Cargo takes the release profile from the top-level package, so opus1 cannot set it for you. An application can add this to its own `Cargo.toml`:

```toml
[profile.release]
codegen-units = 1
lto = "fat"
panic = "abort"   # only if the application does not catch panics
```

`codegen-units = 1` compiles each crate as one unit, so more functions can be inlined. `lto = "fat"` (link-time optimization) lets the compiler optimize across crates. `panic = "abort"` removes the code that unwinds the stack after a panic.

The table gives the measured effect of each setting. For each of the 28 bench scenarios, the opus1 / libopus time ratio with the setting is divided by the same ratio with the default profile. The table shows the geometric mean of these values. Below 1 means faster. Two passes agreed within 0.001. This was measured on 2026-10-13 at commit `2cf93bf`, before most of the performance work above (`target/perf/prof/memory-build/report.md`, section 6).

| Setting | Encode | Decode |
|---|---:|---:|
| `codegen-units = 1` | 1.004 | 0.973 |
| `lto = "fat"` | 0.992 | 0.980 |
| `lto = "fat"`, `codegen-units = 1` | 0.995 | 0.972 |
| `lto = "fat"`, `codegen-units = 1`, `panic = "abort"` | 0.992 | 0.968 |
| `lto = "thin"` | 1.007 | 0.981 |
| `-C target-cpu=native` | 0.999 | 1.001 |

So the recommended profile makes decoding about 3 % faster. The effect on encoding is below 1 % and within what code placement alone can move.

The `heap` feature (on by default) moves the large codec state to the heap when an encoder or decoder is created. It does not allocate per frame. Measured at commit `2cf93bf`: `OpusEncoder::new` made 11 allocations and `OpusDecoder::new` made 7. Then 300 calls each of `encode`, `encode_i16`, `decode`, and `decode` with lost packets made none. In the current code `OpusEncoder::new` makes 7 allocations for mono input and 8 for stereo input, and `OpusDecoder::new` makes 5. One of the decoder's is the float buffer of `decode_i16`, so no decode call allocates. [Size of the encoder and the decoder](06-using-the-library.md#size-of-the-encoder-and-the-decoder) gives them with their sizes.

## Reproducing the numbers

1. Install libopus 1.6.1 and make it visible to `audiopus_sys`. See [How libopus is linked](09-testing.md#how-libopus-is-linked). The harness prints the version it linked, and the golden file header records it.
2. Time the bench set and keep the tables:

   ```text
   cargo run --release --example perf_harness -- bench \
     --md target/perf/my_bench.md --json target/perf/my_bench.json
   ```

   `--filter silk_wb,celt_fb_48k_mono` limits the run to scenarios whose names contain one of the given parts. `--compare target/perf/old.json` also prints the change against an earlier `--json` file.
3. Close other programs while it runs. Rows marked `UNSTEADY` were disturbed in every round; use their median.
4. To check that a speed change kept the output the same, write a golden file before the change and check it after:

   ```text
   cargo run --release --example perf_harness -- golden --write target/golden_before.txt
   # ... change the code ...
   cargo run --release --example perf_harness -- golden --check target/golden_before.txt
   ```

   `golden --check` exits with status 1 when any hash differs and prints the fields that changed.
5. To profile one operation, start it with `run` and attach a sampling profiler to the printed process id, for example `sample <pid> 5` on macOS:

   ```text
   cargo run --release --example perf_harness -- run \
     --scenario celt_fb_48k_stereo_128k_vbr_c10_20ms --op dec --impl rust
   ```

To time opus1 inside your own program, measure many frames and divide, as the harness does:

```rust
use opus1::{Application, OpusEncoder};
use std::time::Instant;

let mut encoder = OpusEncoder::new(48_000, 2, Application::Audio).unwrap();
encoder.bitrate_bps = 128_000;
encoder.complexity = 5;

let frame = 960; // 20 ms at 48 kHz, samples per channel
let pcm: Vec<f32> = (0..frame * 2).map(|i| (i as f32 * 0.03).sin() * 0.5).collect();
let mut packet = vec![0u8; 1276];

let frames = 200;
let start = Instant::now();
for _ in 0..frames {
    let len = encoder.encode(&pcm, frame, &mut packet).unwrap();
    assert!(len > 0);
}
let per_frame = start.elapsed() / frames;
println!("{per_frame:?} per 20 ms frame");
```

Build such a program with `--release`. All numbers in this document are for release builds.

## On an Android phone

Measured on 2026-10-16 at git commit `bd080de`, with the scripts in `scripts/android/` (see [Testing](09-testing.md#tests-on-an-android-phone)). The phone is a HONOR CLK-AN00 with a MediaTek MT6833 and Android 14. `bench.sh` pinned the harness with `taskset` to one core of each kind:

- CPU 7, a Cortex-A76 at up to 2.4 GHz. It reorders instructions, so independent ones run at the same time.
- CPU 5, a Cortex-A55 at up to 2.0 GHz. It runs instructions in program order, at most two per cycle.

opus1 was built with the default release profile. The C reference is libopus built by `scripts/android/build-libopus.sh`: the NDK's clang 18 at `-O2`, the default of libopus's own build. Its source is commit `228a0f85` of the libopus main branch, not the release 1.6.1 (see [Testing](09-testing.md#which-libopus-the-phone-links)).

The bench set ran twice on each core. Between the two runs, the median time of an operation changed by at most 1.2 % on the A76 and 0.9 % on the A55. The opus1 / libopus ratio changed by at most 0.8 %. Every round was quiet, so no scenario was measured again. The cluster of the pinned core spent 89 % to 90 % of each run at its highest clock (cpufreq statistics). The processor reached at most 52.3 °C. The records are in `target/perf/android/`: `bench_cpu7_run1.json` to `bench_cpu5_run2.json` and the `.md` files beside them.

A check later on the same day ran the bench set again with a harness built in another directory and run from another directory of the phone. It ran the set twice on each core. The records are in `target/perf/android/check/`.

- In the first run, the phone kept the A76 cluster at 2.118 GHz and the A55 cluster at 1.75 GHz, while the processor was at 43 °C to 50 °C. The times grew by the ratio of the clocks: 1.13 times on the A76 and 1.14 times on the A55. The geometric means of the ratios were within 0.007 of those in the next table. `bench.sh` now prints a warning when a cluster spends less than 75 % of a run at its highest clock.
- In the second run, at the full clocks, every A55 time and ratio was within 1.2 % of [Results per scenario](#results-per-scenario).
- On the A76, the geometric means of the ratios were 0.686 for encoding and 0.784 for decoding, against 0.692 and 0.790. In SILK and Hybrid scenarios, libopus took up to 3.8 % more time to encode and opus1 up to 2.2 % less time to decode. The earlier and the new program then ran in turns in one session. They kept these differences in two of these scenarios, and showed none in a CELT scenario. The two programs have the same code: a longer directory name compiled into the new one moves its code 16 bytes further. So on the A76, a difference of about 3 % in one scenario between two builds is not a real change.

### The C reference matters

The Homebrew libopus of [Latest results](#latest-results) behaves like an `-Os` build. The phone's libopus source, built on the Mac with `-Os`, decodes at the speed of the Homebrew library, within 2.1 %. Built with `-O2`, as on the phone, it takes 0.80 of the Homebrew library's encode time and 0.89 of its decode time (geometric means over the 28 scenarios, `target/perf/android/mac_vendored_bench.json`). So the ratios of [Latest results](#latest-results) compare opus1 with a slower libopus than libopus's own default build.

Ratio opus1 / libopus of the median times, over the 28 scenarios of the bench set. Below 1 means opus1 is faster.

| | M4 Pro, Homebrew libopus | M4 Pro, the phone's libopus | Cortex-A76 | Cortex-A55 |
|---|---:|---:|---:|---:|
| Encode, geometric mean | 0.472 | 0.578 | 0.692 | 0.755 |
| Encode, range | 0.347 to 0.632 | 0.403 to 0.723 | 0.468 to 0.841 | 0.591 to 0.890 |
| Decode, geometric mean | 0.620 | 0.695 | 0.790 | 0.864 |
| Decode, range | 0.499 to 0.822 | 0.557 to 0.873 | 0.690 to 0.901 | 0.759 to 0.962 |

### Results per scenario

Times are opus1's median per frame in microseconds (µs), the lower of the two runs. The ratio is opus1 / libopus.

| Scenario | A76 encode µs | A76 encode ratio | A55 encode µs | A55 encode ratio | A76 decode µs | A76 decode ratio | A55 decode µs | A55 decode ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `silk_nb_8k_mono_12k_vbr_c0_20ms` | 34.5 | 0.841 | 103.9 | 0.754 | 6.0 | 0.841 | 16.0 | 0.944 |
| `silk_nb_8k_mono_12k_vbr_c5_20ms` | 65.7 | 0.794 | 191.5 | 0.728 | 6.1 | 0.841 | 16.3 | 0.938 |
| `silk_nb_8k_mono_12k_vbr_c10_20ms` | 92.6 | 0.783 | 269.7 | 0.859 | 6.1 | 0.841 | 16.2 | 0.940 |
| `silk_wb_16k_mono_20k_vbr_c0_20ms` | 56.0 | 0.824 | 160.6 | 0.725 | 10.4 | 0.740 | 27.8 | 0.815 |
| `silk_wb_16k_mono_20k_vbr_c5_20ms` | 112.5 | 0.764 | 315.3 | 0.689 | 11.1 | 0.746 | 29.7 | 0.813 |
| `silk_wb_16k_mono_20k_vbr_c10_20ms` | 160.8 | 0.651 | 457.5 | 0.704 | 11.1 | 0.744 | 29.5 | 0.813 |
| `silk_wb_16k_mono_20k_vbr_c9_10ms` | 90.2 | 0.673 | 253.5 | 0.714 | 6.0 | 0.748 | 17.2 | 0.836 |
| `silk_wb_16k_stereo_32k_vbr_c9_20ms` | 322.0 | 0.713 | 908.9 | 0.769 | 21.1 | 0.752 | 57.5 | 0.846 |
| `silk_wb_16k_mono_24k_vbr_c9_20ms_fec10` | 206.4 | 0.666 | 581.5 | 0.742 | 12.3 | 0.774 | 32.5 | 0.844 |
| `hybrid_swb_24k_mono_24k_cbr_c9_20ms` | 216.9 | 0.645 | 628.7 | 0.725 | 23.3 | 0.722 | 75.2 | 0.801 |
| `hybrid_swb_24k_stereo_36k_cbr_c9_20ms` | 434.4 | 0.716 | 1235.5 | 0.791 | 42.3 | 0.702 | 135.4 | 0.777 |
| `hybrid_fb_48k_mono_32k_cbr_c9_20ms` | 217.2 | 0.663 | 628.2 | 0.723 | 26.8 | 0.713 | 85.0 | 0.777 |
| `hybrid_fb_48k_stereo_40k_cbr_c9_20ms` | 448.6 | 0.720 | 1271.2 | 0.782 | 47.3 | 0.695 | 152.6 | 0.761 |
| `celt_fb_48k_mono_64k_vbr_c0_20ms` | 37.3 | 0.754 | 114.8 | 0.833 | 22.4 | 0.870 | 63.5 | 0.928 |
| `celt_fb_48k_mono_64k_vbr_c5_20ms` | 64.0 | 0.683 | 242.6 | 0.843 | 26.3 | 0.818 | 77.6 | 0.894 |
| `celt_fb_48k_mono_64k_vbr_c10_20ms` | 64.2 | 0.517 | 244.0 | 0.630 | 26.6 | 0.818 | 78.7 | 0.896 |
| `celt_fb_48k_stereo_128k_vbr_c0_20ms` | 72.0 | 0.733 | 228.5 | 0.828 | 43.2 | 0.901 | 120.7 | 0.962 |
| `celt_fb_48k_stereo_128k_vbr_c5_20ms` | 110.9 | 0.712 | 386.6 | 0.827 | 51.8 | 0.844 | 150.2 | 0.917 |
| `celt_fb_48k_stereo_128k_vbr_c10_20ms` | 166.3 | 0.671 | 551.1 | 0.744 | 51.9 | 0.844 | 150.2 | 0.918 |
| `celt_fb_48k_stereo_128k_vbr_c9_2p5ms` | 35.8 | 0.771 | 140.5 | 0.890 | 9.6 | 0.843 | 33.5 | 0.906 |
| `celt_fb_48k_stereo_128k_vbr_c9_5ms` | 58.2 | 0.740 | 212.7 | 0.847 | 16.1 | 0.847 | 52.8 | 0.922 |
| `celt_fb_48k_stereo_128k_vbr_c9_10ms` | 86.7 | 0.685 | 301.9 | 0.773 | 28.2 | 0.863 | 86.3 | 0.943 |
| `celt_swb_24k_mono_48k_vbr_c9_20ms` | 56.2 | 0.520 | 220.9 | 0.640 | 22.0 | 0.826 | 67.2 | 0.882 |
| `celt_swb_24k_stereo_96k_vbr_c9_20ms` | 149.8 | 0.681 | 507.5 | 0.754 | 40.9 | 0.831 | 119.0 | 0.857 |
| `celt_fb_48k_mono_24k_vbr_c9_20ms` | 50.3 | 0.468 | 207.0 | 0.591 | 17.9 | 0.751 | 60.5 | 0.855 |
| `celt_fb_48k_stereo_256k_vbr_c9_20ms` | 203.6 | 0.719 | 644.5 | 0.796 | 69.4 | 0.877 | 181.2 | 0.931 |
| `hybrid_fb_48k_mono_32k_cbr_c9_20ms_music` | 197.3 | 0.692 | 570.8 | 0.740 | 26.8 | 0.713 | 86.0 | 0.781 |
| `hybrid_fb_48k_stereo_40k_cbr_c9_20ms_music` | 401.6 | 0.740 | 1139.4 | 0.787 | 47.2 | 0.690 | 153.5 | 0.759 |

How to read the table:

- opus1 is faster than libopus in every scenario, for encoding and decoding, on both cores.
- The slowest operation on the A55, the encoding of 48 kHz stereo Hybrid, takes 1.27 ms per 20 ms frame. That is 6.4 % of one core.
- From the M4 Pro to the A76, the time of opus1 grows 3.21 times for encoding and 2.89 times for decoding (geometric means). The time of the same libopus grows 2.68 and 2.54 times.
- From the A76 to the A55, the time of opus1 grows 3.16 times for encoding and 2.94 times for decoding. The time of libopus grows 2.90 and 2.69 times.
- The A55 costs opus1 most in CELT encoding at complexity 5 and above: 3.2 to 4.1 times its A76 time, against 2.9 to 3.4 times for libopus.
- SILK encoding at complexity 0 and 5 is the other way round: opus1 grows 2.8 to 3.0 times, libopus 3.1 to 3.4 times.

### Where the time goes on the phone

`scripts/android/profile.sh` counted hardware events per frame with `simpleperf stat` and sampled the hot functions with `simpleperf record`, for opus1 and libopus on both cores. The records are in `target/perf/android/profile/`. The program for this keeps line tables and symbols. Its times per frame are within 3 % of the bench for CELT and for decoding, and 2 % to 11 % above it for SILK and Hybrid encoding, for both libraries. In the table, IPC is the number of instructions per cycle.

| Scenario and operation | instructions, opus1 / libopus | cycles on A76, opus1 / libopus | cycles on A55, opus1 / libopus | IPC on A76, opus1 and libopus | IPC on A55, opus1 and libopus |
|---|---:|---:|---:|---:|---:|
| SILK 16 kHz mono c10, encode (`silk_wb_16k_mono_20k_vbr_c10_20ms`) | 0.74 | 0.63 | 0.70 | 2.43 and 2.06 | 1.02 and 0.95 |
| Hybrid 24 kHz mono c9, encode (`hybrid_swb_24k_mono_24k_cbr_c9_20ms`) | 0.75 | 0.64 | 0.73 | 2.41 and 2.04 | 1.00 and 0.97 |
| Hybrid 24 kHz mono c9, decode (`hybrid_swb_24k_mono_24k_cbr_c9_20ms`) | 0.63 | 0.71 | 0.79 | 2.10 and 2.38 | 0.77 and 0.97 |
| CELT 48 kHz mono c10, encode (`celt_fb_48k_mono_64k_vbr_c10_20ms`) | 0.57 | 0.52 | 0.63 | 2.25 and 2.04 | 0.71 and 0.78 |
| CELT 48 kHz mono c10, decode (`celt_fb_48k_mono_64k_vbr_c10_20ms`) | 0.81 | 0.81 | 0.90 | 2.10 and 2.11 | 0.85 and 0.94 |
| CELT 48 kHz stereo c10, encode (`celt_fb_48k_stereo_128k_vbr_c10_20ms`) | 0.71 | 0.67 | 0.75 | 2.10 and 1.99 | 0.76 and 0.80 |
| CELT 48 kHz stereo c10, decode (`celt_fb_48k_stereo_128k_vbr_c10_20ms`) | 0.82 | 0.84 | 0.92 | 2.02 and 2.09 | 0.83 and 0.94 |

What the counters show:

- opus1 runs fewer instructions than libopus: 0.57 to 0.82 of them.
- On the A76, opus1 runs more instructions per cycle than libopus when it encodes (2.10 to 2.43 against 1.99 to 2.06), and fewer when it decodes (2.02 to 2.10 against 2.09 to 2.38).
- On the A55, opus1 runs fewer instructions per cycle than libopus in CELT encoding and in all decoding (0.71 to 0.85 against 0.78 to 0.97). Only in SILK and Hybrid encoding is its IPC a little higher (1.02 and 1.00 against 0.95 and 0.97).
- Several changes of [What made the code fast](#what-made-the-code-fast) let independent operations overlap. An in-order core overlaps them only when they stand next to each other in the program, so these changes gain less on the A55:
  - The pitch cross-correlation of the CELT encoder (`pitch::pitch_xcorr`, 64 lags per pass) takes 5.1 to 5.4 times its A76 cycles on the A55. libopus's NEON kernel takes 2.6 to 2.9 times. Against libopus, the kernel of opus1 takes 0.37 of the cycles on the A76 and 0.71 on the A55 (48 kHz mono, complexity 10).
  - The post-filter of the CELT decoder (`celt::comb_filter_inplace`) takes 5.5 times its A76 cycles on the A55. libopus's `comb_filter` takes 2.2 times.
  - The SILK noise-shaping quantizer (`quantize_subframe_neon::<4>`) takes 0.68 of the cycles of libopus's NEON quantizer on the A76, and 0.99 on the A55.
- The decoder of opus1 misses the level-1 caches more often than libopus's. Per frame of 48 kHz stereo CELT on the A76 there are 423 refills of the level-1 data cache, against 95, and 413 refills of the level-1 instruction cache, against 41. On the A55, the front end of the core (instruction fetch and decode) delivers nothing in 7.1 % to 11.1 % of opus1's decode cycles, against 4.7 % to 5.9 % of libopus's.
- `memmove` takes 29,600 cycles per frame of opus1's 48 kHz stereo CELT encoding on the A55, 2.7 % of the total and 5.2 times its A76 cycles.

### Building for one kind of core

The release build for `aarch64-linux-android` does not name a core. `-C target-cpu=cortex-a55` or `-C target-cpu=cortex-a76` (in `RUSTFLAGS`) tells the compiler which core to order the instructions for. It also turns on the extensions of ARMv8.2, so the program no longer runs on processors of ARMv8.0. Each build ran the bench set once on each core (`target/perf/android/tuning/`). Both give the same output as the default build: `golden --check` against the phone's golden file found 0 of 137 streams that differ.

Change of the opus1 / libopus ratio against the default build, geometric mean over the 28 scenarios. libopus is the same library in all builds.

| Build | A76 encode | A76 decode | A55 encode | A55 decode |
|---|---:|---:|---:|---:|
| `-C target-cpu=cortex-a55` | +0.0 % | -1.5 % | -2.6 % | -1.4 % |
| `-C target-cpu=cortex-a76` | -0.6 % | -1.1 % | +16.0 % | +8.5 % |

So ordering the instructions for the in-order Cortex-A55 makes opus1 2.6 % faster in encoding and 1.4 % faster in decoding on that core, and loses nothing on the Cortex-A76. Ordering them for the Cortex-A76 makes opus1 16.0 % and 8.5 % slower on the A55. Android can move a thread from one kind of core to the other, so an application should not build for the big core alone. These are single runs; the two runs of the default build differed by at most 0.8 % in their ratios.

The check later that day built both programs again. On the A55 it ran each in turns with the default build, twice, for a few scenarios at complexity 9 and 10 (SILK 16 kHz, Hybrid 24 kHz, CELT 48 kHz). The `cortex-a55` build took 1.8 % to 4.2 % less time to encode and 1.5 % to 2.9 % less to decode, in four scenarios. The `cortex-a76` build took 16.2 % to 18.4 % more time to encode and 6.5 % to 8.3 % more to decode, in three scenarios. Both builds again gave the golden hashes of the default build. The records are in `target/perf/android/check/ab/`.

## Earlier measurement on x86-64

The README keeps an older measurement on AMD Ryzen 7 5700X with the earlier `opus_vs_c` benchmark (opus-rs 0.1.x, mono encode, complexity 0). In the release described here, the x86-64 code paths were only compiled, not run. See [Differences from libopus](10-differences-from-libopus.md#platforms).

## See also

- [Testing](09-testing.md): how the output is checked against libopus.
- [Differences from libopus](10-differences-from-libopus.md): what opus1 does not do, and where its output differs.
- [Architecture](07-architecture.md): where the code of each part lives.
- [Glossary](glossary.md): the terms used in these documents.
- The documentation index: [README.md](README.md).
