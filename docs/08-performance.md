# Performance

This document explains how the speed of opus1 is measured against the C library libopus. It gives the latest results and the changes that made the code fast. It also lists the ideas that did not help, the build settings an application can use, and the commands to reproduce the numbers. All numbers were measured on Apple M4 Pro (64-bit ARM), except those of [On an Android phone](#on-an-android-phone) and the phone columns of [Code placement](#code-placement), [Round 8: more work for in-order cores](#round-8-more-work-for-in-order-cores), [Round 7: work for in-order cores](#round-7-work-for-in-order-cores) and [Build settings for applications](#build-settings-for-applications). Each section names the record and the state of the code that its numbers come from. The records are in `target/perf/`, a folder of the maintainers' working tree that is not in git.

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

Two differences remained, and the harness reports them:

1. The libopus that the harness links is a floating-point build. Its SILK encoder is the floating-point one (`silk/float`). The SILK encoder of opus1 is a port of the fixed-point one (`silk/fixed`).
2. Up to commit `d541821`, libopus ran its speech and music analysis (`run_analysis` in `analysis.c`) at complexity 7 and above when the input rate is 16 kHz or more, and opus1 had no port of it. The `bench` table named this step in the column `only C runs`, and the encode ratios of these scenarios favoured opus1. Now opus1 runs the analysis too (cargo feature `analysis`, on by default), so the column is empty and the encode ratios compare the same work. A harness built without the feature names the step again.

Both decoders decode the same packets: the packets that libopus encoded.

### The C reference

The speed of libopus depends on how it was compiled. Two builds of the release 1.6.1 serve as the reference on the Mac. Both come from the release file of Xiph, `opus-1.6.1.tar.gz`, whose SHA-256 sum matches Xiph's list and Homebrew's formula:

1. **libopus 1.6.1 at `-O2`.** `scripts/android/build-libopus.sh --host --rtcd --src <1.6.1 source>` builds it with Apple clang 21. It uses the flags that libopus's own `configure` picks for 64-bit ARM: `-O2`, `FLOAT_APPROX`, and the NEON code presumed (the script lists every flag). `--rtcd` adds the run-time CPU detection that `configure` turns on for Android. With it, the SILK encoder calls libopus's NEON delayed-decision quantizer, as on the phone (see below). This build is the main reference from round 8 on.
2. **The Homebrew libopus 1.6.1.** The tests link it (see [Testing](09-testing.md#how-libopus-is-linked)). Homebrew builds it with `./configure` and its own compiler wrapper. The result behaves like a build at `-Os` (optimized for size):
   - It runs at the speed of the release built by the script at `-Os`. See the table below.
   - Its machine code (the `__text` section) has 238,144 bytes. The script's build of the release has 231,660 bytes at `-Os` and 353,628 bytes at `-O2`. Of the 396 functions found in both libraries, 284 have exactly the size of the Homebrew function at `-Os`, and 138 at `-O2`.
   - In `stereo_itheta` it has no vector arithmetic, like the `-Os` build. The `-O2` build has 20 vector instructions there.
   - The 7 tests that see the vector code of an `-O2` build pass against it, as against the `-Os` build (see [Testing](09-testing.md#results-on-the-phone)).

Run-time CPU detection decides which SILK quantizer libopus runs on 64-bit ARM:

- Apple clang compiles the DotProd instructions for every Mac with Apple silicon. So `configure` presumes them and builds no run-time CPU detection.
- Without run-time detection, `silk/arm/NSQ_del_dec_arm.h` selects the C version of the delayed-decision quantizer, `silk_NSQ_del_dec_c`. It asks for the macro `OPUS_ARM_PRESUME_NEON`, which `configure` sets only for the assembly code of 32-bit ARM. The Homebrew library calls the C version too (checked in its machine code).
- For Android, `configure` turns on run-time detection, and the NEON version `silk_NSQ_del_dec_neon` runs.
- The NEON version handles 3 and 4 states, which SILK keeps from complexity 6 on. Below that both builds run the same code.

Each libopus build was linked into its own build of the same harness, and the programs ran in turns: three passes of the bench set (records in `target/perf/round8/ref/mac_bench2/`, and for the last two rows `mac_bench/` and `mac_bench_prelim_rtcd/`). The table compares the time of libopus between two builds: the geometric mean over the scenarios of the median of the three passes, with the range of the single scenarios in brackets. "Detection" means run-time CPU detection.

| Comparison | Scenarios | Encode | Decode |
|---|---:|---:|---:|
| `-O2` against `-Os`, both without detection | 28 | 0.911 (0.835 to 0.962) | 0.896 (0.782 to 0.966) |
| With detection against without, both at `-O2`: SILK and Hybrid at complexity 9 and 10 | 11 | 0.762 (0.744 to 0.781) | 0.993 (0.985 to 0.997) |
| With detection against without, both at `-O2`: the other scenarios | 17 | 1.000 (0.986 to 1.018) | 0.994 (0.982 to 1.014) |
| The Homebrew library against the release at `-Os` without detection | 28 | 1.011 (0.989 to 1.039) | 1.000 (0.977 to 1.029) |
| Main branch `228a0f85` against the release, both at `-O2` without detection | 28 | 1.003 (0.983 to 1.026) | 1.003 (0.989 to 1.016) |
| Main branch `228a0f85` against the release, both at `-O2` with detection | 28 | 0.985 (0.964 to 1.003) | 0.987 (0.970 to 1.002) |

So:

- `-O2` makes libopus about 10 % faster than `-Os`, in encoding and in decoding.
- The NEON quantizer makes libopus's SILK and Hybrid encoding at complexity 9 and 10 about 24 % faster on the M4 Pro. It changes nothing else.
- Together, the main reference takes 0.808 of the encode time and 0.889 of the decode time of the Homebrew library (geometric means; single scenarios 0.696 to 0.945 and 0.760 to 0.970).
- The Homebrew library runs at the speed of the `-Os` build, within 4 % in every scenario.
- The main branch of libopus and the release 1.6.1 run at the same speed, within 4 % in every scenario, when they are built the same way.

Round 7 used the main branch at `-O2`, with run-time detection, as its `-O2` reference. It concluded that the main branch encodes SILK and Hybrid at complexity 9 and 10 in about 0.78 of the time of 1.6.1. That was wrong. Its builds of the main branch had run-time detection, and the Homebrew library has none, so the difference was the quantizer and not the source. The main branch does write other packets than 1.6.1, in 86 of the 137 golden streams (see [Testing](09-testing.md#which-libopus-source-to-build)).

### Repetitions, processes and medians

One repetition is one pass over the whole clip (6 s by default). The time per frame is the time of the pass divided by the number of frames.

1. Each measuring process first runs one warm-up repetition, which is not counted. It then runs 5 counted repetitions.
2. Inside each repetition, the four operations run one after the other. opus1 and libopus alternate, and the order within each pair is reversed on every repetition. So slow periods of the machine affect both libraries equally.
3. Each scenario is measured in 3 fresh processes, so one "round" has 15 repetitions per operation. A fresh process makes the result independent of the scenarios measured before. Several processes cover effects that are fixed for the life of a process, such as the random addresses the operating system gives to code and data. Both effects were seen to change a result by up to 10 %.
4. A round is "quiet" when the median of every operation is at most 3 % above its minimum. Otherwise another program disturbed the round, and the scenario is measured again, up to 3 rounds in total.
5. The table reports, per operation, the lowest minimum of all rounds and the lowest median of any one round.

On macOS, the measuring threads ask for the scheduling class `QOS_CLASS_USER_INTERACTIVE`, which asks the system to run them on a performance core. Medians repeat better than minimums on a machine that other programs use, so the ratios below are ratios of medians.

### Code placement

The speed of some functions depends on the address at which the linker places them. Any change to the library or to the harness moves the code that follows it. So a difference between two builds can come from the new addresses alone and have nothing to do with the change.

Earlier, on Apple M4 Pro, the same machine code of `silk_warped_autocorrelation_fix` was measured in a fast and a slow state, about 1.7 times apart. That changed the SILK encode time at complexity 5 by 11 to 17 %. The NEON version of that function (commit `2784fdd`) has the same speed at every placement.

To see how much placement moves a time, build the harness with the environment variable `PERF_HARNESS_PAD=<bytes>`. It inserts that many bytes of padding before the library code (on macOS, Linux and Android). Then run the builds in turns and compare them with `pairs`, or compare the records of several paddings with `spread`:

```text
for pad in 0 20 40 60; do
  PERF_HARNESS_PAD=$pad cargo run --release --example perf_harness -- \
    bench --json target/perf/pad_$pad.json
done
cargo run --release --example perf_harness -- spread target/perf/pad_*.json
```

#### How much placement moves the times

Measured on 2026-10-17 with the code of commit `9f462a0`. Four builds that differ only in their padding ran in turns, three passes. The padding moved the library code by 0, 16, 32 and 48 bytes on the phone, and by 0, 12, 28 and 44 bytes on the Mac. So the code moved within its 64-byte cache lines. For each scenario and pass, the spread is how far the highest of the four times lies above the lowest. Each scenario then gets the median of its three spreads. The table gives the median of these values over the scenarios, and the largest one. The Mac ran the 28 scenarios of the bench set. The A76 ran the four scenarios of [Build settings for applications](#build-settings-for-applications), and the A55 two of them: SILK 16 kHz and CELT 48 kHz stereo (records: `target/perf/round7/guide/place/` for the Mac, `target/perf/round7/guide/android/place_a76/` and `place_a55/` for the phone).

- On the Mac other programs changed the load, so the spread there is that of the opus1 / libopus ratio. libopus is a separate library there and does not move.
- On the phone, libopus is linked into the harness and moves with the padding. So the table also gives its spread.

| Core | opus1 encode | opus1 decode | libopus encode | libopus decode |
|---|---:|---:|---:|---:|
| Apple M4 Pro | 0.7 %, largest 2.0 % | 1.0 %, largest 2.3 % | - | - |
| Cortex-A76 | 0.5 %, largest 1.4 % | 0.5 %, largest 0.7 % | 1.9 %, largest 4.0 % | 0.4 %, largest 0.6 % |
| Cortex-A55 | 0.3 %, largest 0.3 % | 0.4 %, largest 0.5 % | 0.2 %, largest 0.3 % | 0.3 %, largest 0.3 % |

What this shows:

- On Apple M4 Pro, no scenario moved by more than 2.3 %. The large effect described above is gone.
- On the Cortex-A76, shifts within a cache line move libopus's SILK and Hybrid encoding by 3.3 % to 4.0 %. A shift by a whole line (64 bytes) moved it by at most 0.1 %. This explains the 3.8 % that a 16-byte shift caused in the check of the phone numbers ([On an Android phone](#on-an-android-phone)).
- opus1 itself moves less on the Cortex-A76, but not never. A second set of four such builds, made from a later version of the harness, moved its SILK decoding by 2.3 % and its Hybrid decoding by 1.1 % (the `lal` builds below).
- On the Cortex-A55, no time moved by more than 0.5 %. The in-order core hardly depends on where the code lies.

#### Aligning the functions

Stable Rust has no attribute for the alignment of a function:

- `#[repr(align(N))]` applies to types only.
- The function attribute `#[align(N)]` (feature `fn_align`) and the flag `-Z min-function-alignment` exist only on nightly Rust.
- `-C llvm-args=...` is stable. It passes options to LLVM, and LLVM has four for code alignment: `-align-all-functions=<log2>`, `-align-all-nofallthru-blocks=<log2>` (only blocks that are reached by a jump, so the padding never runs), `-align-all-blocks=<log2>` and `-align-loops=<bytes>`. These are internal options of LLVM. A newer LLVM may rename or remove them.
- `-C target-cpu` sets the preferred alignment of the named core. In the measured builds, the functions were aligned to 4 bytes by default, on `aarch64-apple-darwin` and on `aarch64-linux-android`. They were aligned to 16 bytes with `cortex-a55` or `cortex-a53`, and to 32 bytes with `cortex-a76`.
- For C, clang has `-falign-functions=<bytes>` and `-falign-loops=<bytes>`.

`RUSTFLAGS="-C llvm-args=-align-all-functions=6"` aligns every opus1 function to 64 bytes. A change in one function then moves the functions after it by whole cache lines, and the code inside each function keeps its place within the lines. Four such builds, with paddings that moved the code by 0, 64, 128 and 192 bytes, ran in turns like the builds above. The libopus in these builds also moved by whole lines only.

| Core | opus1 encode | opus1 decode | libopus encode | libopus decode | Time against the unaligned builds, encode and decode |
|---|---:|---:|---:|---:|---:|
| Apple M4 Pro | 0.7 %, largest 2.1 % | 0.7 %, largest 2.5 % | - | - | -0.9 % and +0.2 % |
| Cortex-A76 | 0.1 %, largest 0.4 % | 0.3 %, largest 0.7 % | 0.2 %, largest 0.3 % | 0.3 %, largest 0.4 % | -0.3 % and -0.1 % |
| Cortex-A55 | 0.2 %, largest 0.3 % | 0.3 %, largest 0.4 % | 0.2 %, largest 0.3 % | 0.3 %, largest 0.4 % | +0.1 % and -0.4 % |

What this shows:

- On the Cortex-A76, the aligned builds are steadier. The largest spread of opus1's encode time fell from 1.4 % to 0.4 %, and that of libopus's encode time from 4.0 % to 0.3 %. opus1's decode time moved by at most 0.7 % in both sets, but by 2.3 % in the second set of unaligned builds.
- On the M4 Pro and on the Cortex-A55, alignment changed neither the spread nor the speed by more than 1 %. Both were steady before.
- The aligned builds run at the speed of the unaligned ones, within 0.9 %. On the Mac, `golden --check` finds 0 of 137 streams that differ for the aligned build.

In a real comparison, a change to opus1 also moves libopus, by any number of bytes. The Rust option does not align libopus, which is C code. To keep libopus steady as well, build it with `-falign-functions=64`. In four builds with such a libopus and with opus1 not aligned, the padding moved opus1 by 0, 16, 32 and 48 bytes and libopus by 0 or 64 bytes (records in `target/perf/round7/guide/android/lal_a76/` and `lal_a55/`). On the A76 the spread of libopus's encode time was then at most 0.3 % (median 0.2 %), against 4.0 % before. On the A55 every spread stayed at or below 0.3 %. Use such a libopus for comparisons only: its times can differ a little from those of the default build, so its ratios do not belong in the same table as ratios against the default build.

So, for comparing two builds on the phone, build both with aligned functions:

```text
# once: a libopus with aligned functions, in a place both checkouts use
export LIBOPUS_PREFIX=$HOME/libopus-android-al64
OPUS_EXTRA_CFLAGS=-falign-functions=64 scripts/android/build-libopus.sh
export RUSTFLAGS="-C llvm-args=-align-all-functions=6"
scripts/android/build-variant.sh new        # in the checkout with the change
# the same in the checkout of the base, then:
scripts/android/ab.sh --cpus "5 7" ../base/target/android/bin/perf_harness_base new -- --filter ...
```

On the Mac this is not needed. Applications gain nothing from it, so it is not part of [Build settings for speed](06-using-the-library.md#build-settings-for-speed).

### Comparing two builds

To measure a change, build the code before it (the base) and after it, and run the two programs in turns in one session: base, new, then new, base, then base, new. Both builds then see the same state of the machine: the same load, temperature and clock. Each pair of runs gives one comparison. `perf_harness pairs` reads the `--json` records of the pairs, the base of each pair first:

```text
perf_harness pairs base_1.json new_1.json base_2.json new_2.json base_3.json new_3.json
```

For each scenario and operation it prints the change of the median time from the base to the new build, as the lowest and the highest change over the pairs. A change with the same sign in every pair is marked `faster` or `slower`. Below the table it prints, per pair, the geometric mean over the scenarios of each operation and of the opus1 / libopus ratios.

The libopus columns are a control. libopus is the same code in both builds, so its time should not change. On the Mac the harness links the Homebrew libopus as a separate library, and a change there comes from the state of the machine. The ratio removes it. On the phone libopus is linked into the program, so a new build also moves libopus to other addresses. A change of the libopus time then also shows how much code placement alone moves a time on that core.

A change is real when:

1. it has the same sign in every pair, on the core it is meant for, and
2. it is larger than the difference between two placements of the same code ([Code placement](#code-placement)). With the default build, that difference reached 2.3 % for opus1 on the M4 Pro and on the Cortex-A76, 4.0 % for libopus on the Cortex-A76, and 0.5 % on the Cortex-A55. Builds with aligned functions keep it below 0.7 % on the Cortex-A76.

On the phone, `scripts/android/build-variant.sh` builds the programs and `scripts/android/ab.sh` runs them in turns as one job and prints the `pairs` table (see [Testing](09-testing.md#tests-on-an-android-phone)).

## Latest results

Measured on 2026-10-18 with the code of git commit `f952b46` (branch `next-analysis`: the speech and music analysis, DTX and their NEON code, on `d541821`; see [The speech and music analysis](#the-speech-and-music-analysis)). Machine: Apple M4 Pro, macOS. opus1 was built with the default release profile. The C references are the two of [The C reference](#the-c-reference): libopus 1.6.1 at `-O2` with run-time CPU detection, and the Homebrew libopus 1.6.1. The code before the analysis (`d541821`) and after it ran in turns, each linked with each of the two references: four harness programs, three passes of the bench set and of the two scenarios at complexity 7. Other programs kept the load average of the machine between 7 and 53, so single repetitions were often disturbed; the medians below held. Source: `target/perf/analysis_c7/mac_bench/`.

Times are medians per frame in microseconds (µs). The opus1 time is the median over the six runs of the code after the analysis, three with each reference. The libopus time is the median over the six runs with the `-O2` reference. A ratio is opus1 / libopus, the median over the three passes. A ratio below 1 means opus1 is faster. Both encoders now run the speech and music analysis at complexity 7 to 10 (the column `only C runs` of earlier tables is gone), so every ratio compares the same work, except that the SILK encoders are the fixed-point one in opus1 and the floating-point one in libopus.

| Scenario | opus1 encode µs | libopus -O2 encode µs | Encode ratio, -O2 | Encode ratio, Homebrew | opus1 decode µs | libopus -O2 decode µs | Decode ratio, -O2 | Decode ratio, Homebrew |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `silk_nb_8k_mono_12k_vbr_c0_20ms` | 9.1 | 12.9 | 0.711 | 0.625 | 2.0 | 3.1 | 0.636 | 0.623 |
| `silk_nb_8k_mono_12k_vbr_c5_20ms` | 20.0 | 31.0 | 0.654 | 0.593 | 2.1 | 3.3 | 0.649 | 0.628 |
| `silk_nb_8k_mono_12k_vbr_c10_20ms` | 25.3 | 43.2 | 0.579 | 0.412 | 2.1 | 3.2 | 0.648 | 0.629 |
| `silk_wb_16k_mono_20k_vbr_c0_20ms` | 14.9 | 22.2 | 0.676 | 0.590 | 3.6 | 6.0 | 0.607 | 0.581 |
| `silk_wb_16k_mono_20k_vbr_c5_20ms` | 34.7 | 56.3 | 0.621 | 0.582 | 3.9 | 6.4 | 0.615 | 0.587 |
| `silk_wb_16k_mono_20k_vbr_c10_20ms` | 46.7 | 89.8 | 0.516 | 0.385 | 3.9 | 6.4 | 0.617 | 0.588 |
| `silk_wb_16k_mono_20k_vbr_c9_10ms` | 25.8 | 48.1 | 0.532 | 0.393 | 2.1 | 3.3 | 0.627 | 0.598 |
| `silk_wb_16k_stereo_32k_vbr_c9_20ms` | 90.7 | 165.2 | 0.546 | 0.392 | 7.0 | 11.6 | 0.611 | 0.570 |
| `silk_wb_16k_mono_24k_vbr_c9_20ms_fec10` | 60.3 | 116.9 | 0.513 | 0.371 | 4.5 | 6.9 | 0.649 | 0.615 |
| `hybrid_swb_24k_mono_24k_cbr_c9_20ms` | 62.9 | 124.1 | 0.505 | 0.368 | 7.3 | 12.1 | 0.604 | 0.533 |
| `hybrid_swb_24k_stereo_36k_cbr_c9_20ms` | 121.1 | 224.0 | 0.538 | 0.379 | 12.3 | 21.9 | 0.562 | 0.515 |
| `hybrid_fb_48k_mono_32k_cbr_c9_20ms` | 68.1 | 120.5 | 0.566 | 0.420 | 8.5 | 14.4 | 0.589 | 0.506 |
| `hybrid_fb_48k_stereo_40k_cbr_c9_20ms` | 130.3 | 229.8 | 0.564 | 0.397 | 13.9 | 24.8 | 0.561 | 0.493 |
| `celt_fb_48k_mono_64k_vbr_c0_20ms` | 11.8 | 19.2 | 0.619 | 0.574 | 8.3 | 10.8 | 0.752 | 0.701 |
| `celt_fb_48k_mono_64k_vbr_c5_20ms` | 21.2 | 40.5 | 0.525 | 0.453 | 9.6 | 12.9 | 0.727 | 0.612 |
| `celt_fb_48k_mono_64k_vbr_c10_20ms` | 24.8 | 49.6 | 0.501 | 0.428 | 9.8 | 13.1 | 0.734 | 0.608 |
| `celt_fb_48k_stereo_128k_vbr_c0_20ms` | 22.7 | 38.5 | 0.596 | 0.545 | 16.1 | 20.2 | 0.779 | 0.722 |
| `celt_fb_48k_stereo_128k_vbr_c5_20ms` | 35.7 | 66.0 | 0.544 | 0.457 | 19.2 | 24.9 | 0.748 | 0.602 |
| `celt_fb_48k_stereo_128k_vbr_c10_20ms` | 58.0 | 101.0 | 0.577 | 0.463 | 19.2 | 24.9 | 0.751 | 0.602 |
| `celt_fb_48k_stereo_128k_vbr_c9_2p5ms` | 11.1 | 16.3 | 0.682 | 0.585 | 3.2 | 3.9 | 0.807 | 0.737 |
| `celt_fb_48k_stereo_128k_vbr_c9_5ms` | 18.6 | 29.4 | 0.637 | 0.530 | 5.6 | 6.9 | 0.799 | 0.690 |
| `celt_fb_48k_stereo_128k_vbr_c9_10ms` | 29.6 | 49.5 | 0.598 | 0.489 | 10.1 | 12.9 | 0.768 | 0.632 |
| `celt_swb_24k_mono_48k_vbr_c9_20ms` | 20.0 | 42.1 | 0.477 | 0.415 | 7.8 | 10.5 | 0.734 | 0.649 |
| `celt_swb_24k_stereo_96k_vbr_c9_20ms` | 49.5 | 87.8 | 0.567 | 0.467 | 14.8 | 19.9 | 0.726 | 0.609 |
| `celt_fb_48k_mono_24k_vbr_c9_20ms` | 19.5 | 40.9 | 0.477 | 0.395 | 5.7 | 8.9 | 0.617 | 0.487 |
| `celt_fb_48k_stereo_256k_vbr_c9_20ms` | 69.9 | 112.1 | 0.624 | 0.549 | 27.5 | 34.0 | 0.803 | 0.756 |
| `hybrid_fb_48k_mono_32k_cbr_c9_20ms_music` | 60.1 | 103.0 | 0.581 | 0.417 | 8.5 | 14.2 | 0.593 | 0.503 |
| `hybrid_fb_48k_stereo_40k_cbr_c9_20ms_music` | 115.8 | 197.9 | 0.583 | 0.411 | 13.5 | 24.5 | 0.558 | 0.494 |
| `silk_wb_16k_mono_20k_vbr_c7_20ms` | 41.6 | 81.4 | 0.508 | 0.423 | 3.9 | 6.3 | 0.616 | 0.588 |
| `celt_fb_48k_stereo_128k_vbr_c7_20ms` | 38.8 | 75.1 | 0.524 | 0.440 | 19.1 | 24.7 | 0.750 | 0.602 |

How to read the table:

- **Decoding** takes 0.558 to 0.807 of the time of libopus at `-O2`, and 0.487 to 0.756 of the time of the Homebrew build.
- **Encoding at complexity 0 and 5** takes 0.525 to 0.711 of the time of libopus at `-O2` (Homebrew: 0.453 to 0.625).
- **Encoding at complexity 7 to 10** takes 0.477 to 0.682 of the time of libopus at `-O2` (Homebrew: 0.368 to 0.585). Both encoders run the analysis here. [The speech and music analysis](#the-speech-and-music-analysis) gives the ratios from before the port.
- Geometric means over the 28 scenarios of the bench set: encoding 0.572 against libopus at `-O2` and 0.461 against Homebrew; decoding 0.669 and 0.598. Before the analysis (`d541821`, in the same passes) encoding was 0.546 and 0.443, with the analysis in libopus's time only.
- SILK encode ratios compare the fixed-point SILK encoder of opus1 with the floating-point one of libopus.
- The harness marked some operations unsteady in single runs, because other programs ran. The table uses medians, which are the numbers to use for them.

### The speech and music analysis

From complexity 7 on, for input at 16 to 48 kHz, both encoders run the speech and music analysis of libopus (`src/analysis.c` and `src/mlp.c`; in opus1 `src/analysis/`, see [The encoder](03-encoder.md#13-the-speech-and-music-analysis)). It runs once per 20 ms of input and stream. The command `perf_harness analysis` times it alone: opus1's `TonalityAnalysisState::run` against libopus's `run_analysis`, on the same 20 ms calls of 10 s of the harness's speech and music signals, 9 or 15 passes that alternate the two libraries.

Time of one 20 ms call, median, and the ratio opus1 / libopus. The Mac links the Homebrew libopus. The phone links libopus 1.6.1 built by `build-libopus.sh` at `-O2` and runs pinned to CPU 7 (Cortex-A76) and CPU 5 (Cortex-A55). Measured on 2026-10-18 with the code of commit `1856a50`, three interleaved passes on the phone and four on the Mac, where other programs kept the load average at about 25:

| Input | M4 Pro opus1 µs | M4 Pro libopus µs | Ratio | A76 opus1 µs | A76 libopus µs | Ratio | A55 opus1 µs | A55 libopus µs | Ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 kHz mono speech | 2.82 | 10.13 | 0.279 | 10.49 | 27.30 | 0.384 | 32.99 | 84.10 | 0.392 |
| 24 kHz mono music | 1.81 | 7.00 | 0.258 | 8.88 | 24.76 | 0.359 | 29.47 | 78.99 | 0.373 |
| 48 kHz mono music | 2.85 | 10.26 | 0.278 | 11.59 | 29.10 | 0.398 | 34.69 | 90.53 | 0.383 |
| 48 kHz mono speech | 2.89 | 10.68 | 0.271 | 11.64 | 29.23 | 0.398 | 34.82 | 90.50 | 0.385 |
| 48 kHz stereo music | 2.93 | 10.97 | 0.271 | 12.07 | 30.25 | 0.399 | 36.72 | 95.56 | 0.384 |
| 48 kHz stereo speech | 2.96 | 11.33 | 0.262 | 12.07 | 30.39 | 0.397 | 36.73 | 95.79 | 0.383 |
| the network alone | 0.21 | 1.64 | 0.126 | 0.54 | 4.86 | 0.111 | 2.71 | 22.59 | 0.120 |

In these short sessions the A76 cluster ran mostly at 2.0 GHz instead of 2.4 GHz (cpufreq statistics), so its times can be up to 20 % high. Its ratios hold, because both libraries ran in the same passes.

The first port was scalar and followed the C code line by line. It took 0.59 to 0.72 of libopus's time on the Mac, 0.81 to 0.86 on the A76 and 0.79 to 0.88 on the A55. Then the busiest loops got NEON versions and caches. Each step keeps the arithmetic of the scalar code, lane by lane, so the results stay the bits of libopus; unit tests compare each kernel with the scalar code, and the libopus comparison tests ran after each step. Each step was kept only if it was faster in interleaved runs against the code before it:

| Step (commit) | What changed | M4 Pro | Cortex-A76 | Cortex-A55 |
|---|---|---:|---:|---:|
| Input (`68681b9`, `f5756b4`) | The digital silence test searches the largest sample four at a time and runs libopus's loop only when a NaN is present. The downmix runs four or eight samples at a time. The 16 kHz path makes three outputs from two samples instead of computing the index of each repeated sample. | -12.5 to -22.9 % | -10.3 to -18.3 % | -7.0 to -17.2 % |
| Bins (`035055a`) | The phase tonality of the FFT bins, four bins per vector, with one division per `fast_atan2f` instead of a branch. The energy of each bin is computed once instead of twice. | -33.8 to -44.7 % | -34.4 to -40.6 % | -29.8 to -35.0 % |
| Caches (`b96e886`) | The state keeps the square roots of the band energies and the distances between the stored blocks: 18 square roots and 7 distances per block instead of 144 and 56. | -1.7 to -5.8 % | -3.3 to -6.6 % | -7.4 to -10.0 % |
| Window (`6e4896b`) | The window, the scaling and the bit-reversed order of the FFT input in one NEON pass. | not measured alone | -1.5 to -3.8 % | -1.5 to -6.3 % |
| Network (`fba40c0`) | The recurrent layer keeps the 18 sums of its three gates in vector registers; the activations run four at a time. | network 0.32 to 0.20 µs | -5.4 to -9.4 % | -7.5 to -11.4 % |
| Bins (`6266595`) | Two groups of four bins per pass. | not measured alone | -3.0 to -4.5 % | -1.2 to +0.2 % |
| Decimator (`745fc74`, `1856a50`) | The 48 kHz decimator makes four outputs per pass, so the compiler interleaves the sums of one output with the recursive filters of the next. 48 kHz cases only. | not measured alone | within ±2 % | -5.4 to -7.4 % |

The ranges are over the six input cases (four for the decimator) and the passes. A pass in which the libopus control moved by more than 10 % was left out (two of them, both from other programs on the machine). On the A55 the time of the network alone moved by up to 12 % between builds whose network code was the same, so the A55 is not free of code placement effects for single small loops.

What did not change: the FFT is opus1's FFT of the CELT encoder, and the 20 logarithms in double precision (19 natural, one to base 10) and the recursive filters of the decimator are kept as libopus computes them. The filters of the decimator are chains in which each step waits for the one before (a subtraction, a product and a sum per sample), so they need at least three times the latency of one operation per output sample: on the A55, with 4 cycles per operation, about 5,800 cycles for the 480 outputs of a frame.

#### The encoder before and after the analysis

The measurement of [Latest results](#latest-results) ran the code before the port (`d541821`) and after it in turns. The table gives the scenarios in which the analysis now runs (complexity 7 to 10, 16 to 48 kHz): opus1's median encode time per frame before and after, the time the analysis added, and the encode ratios against both references. Before the port only libopus ran the analysis, so the ratios before favoured opus1; the ratios after compare the same work.

| Scenario | opus1 encode µs before | after | Added µs | Encode ratio, -O2, before | after | Encode ratio, Homebrew, before | after |
|---|---:|---:|---:|---:|---:|---:|---:|
| `silk_wb_16k_mono_20k_vbr_c10_20ms` | 44.1 | 46.7 | +2.6 | 0.490 | 0.516 | 0.362 | 0.385 |
| `silk_wb_16k_mono_20k_vbr_c9_10ms` | 24.6 | 25.8 | +1.2 | 0.509 | 0.532 | 0.379 | 0.393 |
| `silk_wb_16k_stereo_32k_vbr_c9_20ms` | 87.8 | 90.7 | +2.9 | 0.534 | 0.546 | 0.380 | 0.392 |
| `silk_wb_16k_mono_24k_vbr_c9_20ms_fec10` | 57.4 | 60.3 | +2.9 | 0.493 | 0.513 | 0.355 | 0.371 |
| `hybrid_swb_24k_mono_24k_cbr_c9_20ms` | 60.8 | 62.9 | +2.1 | 0.490 | 0.505 | 0.355 | 0.368 |
| `hybrid_swb_24k_stereo_36k_cbr_c9_20ms` | 118.9 | 121.1 | +2.3 | 0.533 | 0.538 | 0.372 | 0.379 |
| `hybrid_fb_48k_mono_32k_cbr_c9_20ms` | 64.9 | 68.1 | +3.2 | 0.539 | 0.566 | 0.398 | 0.420 |
| `hybrid_fb_48k_stereo_40k_cbr_c9_20ms` | 126.6 | 130.3 | +3.7 | 0.551 | 0.564 | 0.387 | 0.397 |
| `celt_fb_48k_mono_64k_vbr_c10_20ms` | 20.9 | 24.8 | +3.9 | 0.422 | 0.501 | 0.365 | 0.428 |
| `celt_fb_48k_stereo_128k_vbr_c10_20ms` | 54.1 | 58.0 | +3.8 | 0.533 | 0.577 | 0.437 | 0.463 |
| `celt_fb_48k_stereo_128k_vbr_c9_2p5ms` | 10.9 | 11.1 | +0.3 | 0.667 | 0.682 | 0.571 | 0.585 |
| `celt_fb_48k_stereo_128k_vbr_c9_5ms` | 17.9 | 18.6 | +0.7 | 0.610 | 0.637 | 0.510 | 0.530 |
| `celt_fb_48k_stereo_128k_vbr_c9_10ms` | 27.6 | 29.6 | +2.0 | 0.561 | 0.598 | 0.462 | 0.489 |
| `celt_swb_24k_mono_48k_vbr_c9_20ms` | 17.5 | 20.0 | +2.5 | 0.421 | 0.477 | 0.372 | 0.415 |
| `celt_swb_24k_stereo_96k_vbr_c9_20ms` | 46.9 | 49.5 | +2.6 | 0.534 | 0.567 | 0.445 | 0.467 |
| `celt_fb_48k_mono_24k_vbr_c9_20ms` | 15.8 | 19.5 | +3.7 | 0.386 | 0.477 | 0.324 | 0.395 |
| `celt_fb_48k_stereo_256k_vbr_c9_20ms` | 66.0 | 69.9 | +3.9 | 0.584 | 0.624 | 0.522 | 0.549 |
| `hybrid_fb_48k_mono_32k_cbr_c9_20ms_music` | 56.7 | 60.1 | +3.4 | 0.554 | 0.581 | 0.397 | 0.417 |
| `hybrid_fb_48k_stereo_40k_cbr_c9_20ms_music` | 112.6 | 115.8 | +3.2 | 0.571 | 0.583 | 0.398 | 0.411 |
| `silk_wb_16k_mono_20k_vbr_c7_20ms` | 38.7 | 41.6 | +3.0 | 0.474 | 0.508 | 0.393 | 0.423 |
| `celt_fb_48k_stereo_128k_vbr_c7_20ms` | 35.2 | 38.8 | +3.6 | 0.466 | 0.524 | 0.397 | 0.440 |
ratio change per pass, analysis scenarios: min -28.7% max +55.1% positive 119 of 126
ratio change per pass, other scenarios: min -4.8% max +4.7%

- The analysis adds 2.1 to 3.9 µs per 20 ms frame and stream on Apple M4 Pro. With frames of 2.5, 5 and 10 ms it adds about an eighth, a quarter and a half of that per frame, because it runs once per 20 ms of input. A little more than the analysis alone (about 2.9 µs at 48 kHz) goes to what the encoder does with its result, for example the energy of each frame and the other bits that CELT spends.
- Geometric means over these 21 scenarios: encoding takes 0.551 of the time of libopus at `-O2` (0.516 before) and 0.431 of the Homebrew build (0.404 before). opus1's own encode time grew by 6.9 %.
- Over the 9 other scenarios, where neither encoder runs the analysis, opus1's encode time moved by +0.6 % and the ratios by at most 4.8 % in single passes, the code placement and the load of the machine.
- In 119 of the 126 single passes (21 scenarios, three passes, two references) the ratio after was higher than before. Single passes moved by up to 55 % in both directions while other programs loaded the machine, so only the medians are given.

The phone measured the same before and after: see [On an Android phone](#on-an-android-phone).

### Deep PLC models

The ignored test `timing` of `tests/dnn_plc_libopus.rs` measures each call next to libopus's own function (`-- --ignored --nocapture timing`). It runs the two sides in turns, in batches, and takes the median. Apple M4 Pro, 2026-10-18, opus1 with `NeonDotprod`, libopus 1.6.1 with run-time CPU detection (arch 4, DotProd); the results did not change with a load average of 25 to 44 from other jobs. Microseconds per call:

| Call | opus1 | libopus without vector loops | libopus `-O2` |
|---|---:|---:|---:|
| Feature analysis of 10 ms, PitchDNN included | 5.5 | 11.8 | 9.4 |
| PitchDNN | 1.9 | 4.0 | 2.2 |
| `fargan_cont` | 42.2 | 48.0 | 45.7 |
| `fargan_synthesize`, 10 ms | 39.3 | 44.3 | 42.1 |
| `update`, 10 ms | 0.09 | 0.14 | 0.09 |
| `conceal`, first lost frame after 40 good frames | 311 | 476 | 421 |
| `conceal`, each further lost frame | 44.9 | 53.4 | 51.2 |
| `update_from_celt`, stereo (libopus `update_plc_state`) | 4.1 | 13.0 | 7.2 |

The `-O2` column comes from a separate run against `libopus-host-1.6.1-dnn`, in which opus1 took the same times.

On the phone (HONOR CLK-AN00, the same test against `libopus-1.6.1-dnn-novec`, pinned with `taskset`), microseconds per call:

| Call | Cortex-A76 opus1 | Cortex-A76 libopus | Cortex-A55 opus1 | Cortex-A55 libopus |
|---|---:|---:|---:|---:|
| Feature analysis of 10 ms, PitchDNN included | 19.4 | 35.3 | 84.9 | 148.6 |
| PitchDNN | 7.2 | 16.5 | 36.9 | 82.0 |
| `fargan_cont` | 177 | 190 | 729 | 906 |
| `fargan_synthesize`, 10 ms | 163 | 171 | 662 | 824 |
| `conceal`, first lost frame after 40 good frames | 1,337 | 1,779 | 5,765 | 7,829 |
| `conceal`, each further lost frame | 263 | 277 | 1,091 | 1,313 |
| `update_from_celt`, stereo | 9.8 | 34.0 | 37.6 | 111.8 |

The first lost frame of a loss analyses up to 15 frames of history and starts FARGAN, so it costs about seven times a later one. `cargo bench --features deep-plc --bench deep_plc_bench` measures the opus1 side alone with Criterion.

### Deep PLC in the decoder

The ignored test `timing` of `tests/dnn_plc_decoder_libopus.rs` times whole decode calls of `OpusDecoder` and of libopus's `opus_decode_float` at complexity 5, mono, 48 kHz output, on 10 s of speech coded by libopus, with 3 packets lost after every 22 that arrive (`-- --ignored --nocapture --test-threads=1 timing`). The first lost packet conceals after 440 ms of good audio, so deep PLC analyses its whole history. Apple M4 Pro, 2026-10-18, load average 5 to 10 from other jobs. Medians of 5 runs, microseconds per call:

| Stream | Call | opus1 | libopus without vector loops | libopus `-O2` |
|---|---|---:|---:|---:|
| SILK wideband, 20 ms | first lost packet | 358 | 525 | 464 |
| SILK wideband, 20 ms | further lost packet | 95 | 111 | 105 |
| SILK wideband, 20 ms | packet that arrived | 4.6 | 14.7 | 12.9 |
| CELT fullband, 20 ms | first lost packet | 327 | 461 | 406 |
| CELT fullband, 20 ms | further lost packet | 91 | 116 | 107 |
| CELT fullband, 20 ms | packet that arrived | 8.1 | 11.1 | 11.2 |
| CELT fullband, 10 ms | first lost packet | 250 | 358 | 317 |
| CELT fullband, 10 ms | further lost packet | 46 | 60 | 55 |
| CELT fullband, 10 ms | packet that arrived | 4.3 | 5.7 | 5.5 |

opus1 took the same times in the run against the `-O2` build. A lost SILK packet of 20 ms is two calls of deep PLC; a lost CELT packet of 20 ms is two, and three on the first lost packet, which also feeds the CELT history into deep PLC. libopus spends more on a SILK packet that arrived because its build has OSCE, which computes the features of each 20 ms frame also at complexity 5; opus1 skips them as long as the method is NONE.

These calls were not timed on the phone. From the model measurements above (two calls of `conceal` per 20 ms), the deep PLC part of a first lost 20 ms packet takes about 1.6 ms on a Cortex-A76 and 6.9 ms on a Cortex-A55, and that of each further lost 20 ms packet about 0.5 ms and 2.2 ms.

### Round 8: more work for in-order cores

Round 8 is the eighth round of speed work on opus1. It made the release libopus 1.6.1 the C reference on the Mac and on the phone (see [The C reference](#the-c-reference)). It also went on with the work for in-order cores, in the SILK quantizer, the PVQ search and the bit allocation of CELT, the pitch analysis and the CELT decoder. Five parts were done separately and then merged (`git log --first-parent c96e936..b7a8b64`):

| Part (branch) | Files | What changed |
|---|---|---|
| C reference (`round8-ref`) | `scripts/android/`, documents | `build-libopus.sh --src` builds the release 1.6.1 from Xiph's release file, with the defines that its own `configure` writes. `--rtcd` gives a Mac build the run-time CPU detection of Android. No change to the codec. |
| SILK quantizer (`round8-nsq`) | `src/silk/nsq_del_dec.rs` | On Apple targets, the first 5 taps of the shaping filter run for both candidates of every state while the survivors of the last sample are chosen. On other targets, the shaping coefficients stay in 6 vectors for 2 and 4 states. Before, the compiler kept 16 to 24 vectors and reloaded them from the stack in every sample. |
| PVQ search and allocation (`round8-pvq`) | `src/pvq.rs`, `src/rate.rs`, `src/bands.rs` | The check of each pulse of the PVQ search takes 5 instead of 9 instructions. The next pulse is placed in advance (not on Apple targets), and the next candidate comes from one maximum over keys. `bits2pulses` counts the costs below the budget with three vector compares instead of a bisection. The two allocation searches sum the bands with NEON, for frames of 9 bands or more. The cosines of the spreading rotation come from inline copies of the cosine kernels of the C library, not from a call. `isqrt32` takes the square root in double precision. The encoder takes the gain of the resynthesis from the energy that the search returns, before it codes the pulses. |
| Pitch analysis (`round8-pitch`) | `src/pitch.rs`, `src/celt.rs` | On Apple targets, the window energies of `remove_doubling` are summed in the gaps of `pitch_search`. On all targets, about 2500 values per frame are no longer set to zero before they are written. |
| CELT decoder (`round8-decode`) | `src/celt.rs`, `src/kiss_fft.rs`, `src/mdct.rs` | The synthesis builds the spectrum in place, so the 7.7 KB buffer `w_freq` is gone. The de-emphasis runs its two groups of filters step by step side by side. The radix-4 and radix-2 passes of the FFT and the pre-rotation of the inverse MDCT handle two independent groups per step. |

The merge also changed one comparison in `nsq_del_dec.rs` (commit `b7a8b64`). On targets other than Apple's, `FIRST_TAPS > 0` is always false, and clippy refused to build it. The machine code did not change.

Every part keeps the codec output the same:

- On the Mac, `perf_harness golden --check` against `target/perf/golden_r6.txt` finds 0 of 137 streams that differ after each merge. On the phone, the program of `b7a8b64` finds 0 of 137 as well.
- The full test suite passes after each merge: 1469 tests at the end, and 924 in the `no_std` build.
- On the phone, against libopus 1.6.1 at `-O2`, 1413 tests pass. The same 8 tests fail as before round 8, with the same values. Each of them compares with the last bits of libopus's vector code or builds its input with the C library's sine (see [Testing](09-testing.md#results-on-the-phone)).
- The new vector kernels and the code around the new `unsafe` blocks also ran on inputs placed right before and right after memory that may be neither read nor written, on the Mac and on the phone. No kernel read or wrote outside its slices. The scratch tests are in `target/perf/round8/merge/guard_tests.patch`.

The code before round 8 (`c96e936`) and after it (`b7a8b64`) ran in turns ([Comparing two builds](#comparing-two-builds)). On the Mac they ran three pairs of the bench set with each of the two C references, six pairs in all. On the phone they ran two pairs on each kind of core, against libopus 1.6.1 at `-O2`. The A76 ran both pairs in one session. The A55 ran one pair in each of two sessions, in opposite orders. The clusters spent 86.8 % (A76) and 94.2 % to 94.3 % (A55) of the sessions at their highest clock. The records are in `target/perf/round8/merge/` (`mac/bench/` and `android/`). The table gives the change of opus1's median time: the geometric mean over the scenarios of the group, lowest and highest over the pairs, and in brackets the range of the single scenarios over all pairs. Below 0 means faster.

| Group | Apple M4 Pro | Cortex-A76 | Cortex-A55 |
|---|---:|---:|---:|
| Encode, all 28 scenarios | -3.5 to -3.1 % (-5.2 to +0.7 %) | -1.5 % (-5.2 to +1.3 %) | -1.8 to -1.5 % (-3.9 to +0.4 %) |
| Encode, SILK (9) | -2.8 to -2.3 % (-4.8 to +0.7 %) | +0.5 to +0.6 % (+0.0 to +1.3 %) | -0.9 to -0.7 % (-1.8 to +0.4 %) |
| Encode, Hybrid (6) | -3.9 to -3.3 % (-4.4 to -1.9 %) | -0.1 to -0.0 % (-0.2 to +0.2 %) | -1.6 to -1.2 % (-1.8 to -1.1 %) |
| Encode, CELT (13) | -4.1 to -3.5 % (-5.2 to -1.1 %) | -3.5 % (-5.2 to -1.8 %) | -2.5 to -2.3 % (-3.9 to -0.8 %) |
| Decode, all 28 scenarios | -2.2 to -1.9 % (-5.8 to +1.3 %) | -1.4 % (-4.2 to +1.6 %) | -1.8 to -1.7 % (-4.6 to +1.1 %) |
| Decode, SILK (9) | -0.5 to -0.1 % (-2.1 to +1.3 %) | +1.0 to +1.1 % (+0.4 to +1.6 %) | +0.5 % (+0.0 to +1.1 %) |
| Decode, Hybrid (6) | -0.7 to -0.1 % (-1.6 to +1.2 %) | -0.7 to -0.5 % (-1.5 to +0.1 %) | -1.9 to -1.7 % (-2.5 to -1.3 %) |
| Decode, CELT (13) | -4.2 to -3.6 % (-5.8 to -2.1 %) | -3.4 % (-4.2 to -2.5 %) | -3.3 to -3.1 % (-4.6 to -2.4 %) |

What the table shows:

- Every CELT scenario was faster in every pair on all three cores, in encoding and in decoding. So was every Hybrid scenario on the A55, and Hybrid encoding on the M4 Pro.
- libopus is the same code in both builds. Its geometric means changed by -0.7 to +0.8 % on the Mac, by +0.0 to +0.5 % on the A76 and by +0.1 to +0.8 % on the A55.
- The code of the SILK decoder did not change. Its time moved by +0.0 to +1.6 % on the phone, and libopus's SILK decoding moved with it (+0.2 to +1.0 %). That is code placement, not an effect of round 8.
- On the A76, SILK encoding took 0.0 to 1.3 % more time. The plain quantizer of complexity 0 did not change and took 1.2 to 1.3 % more. On the A76 code placement alone moves single scenarios by up to 2.3 % ([Code placement](#code-placement)). See also the steps below.
- The gains are smaller than those of round 7, which took 7.5 % to 11.1 % off encoding on the phone.

The bench set has no scenario at complexity 6 or 7. There SILK keeps 3 states in its delayed-decision quantizer, and round 7 had left encoding at complexity 7 0.9 % to 1.6 % slower on the M4 Pro. For this measurement only, the harness got six more scenarios, in the base and in the merged code alike (patch `target/perf/round8/merge/c67_harness.patch`, not in git). They ran in turns like the bench set: three pairs with each C reference on the Mac, and two pairs on each core of the phone in one session. Change of opus1's median time, lowest to highest over the pairs, and the encode ratio opus1 / libopus of the merged code (`-O2` reference on the Mac):

| Scenario | M4 Pro encode | A76 encode | A55 encode | M4 Pro decode | A76 decode | A55 decode | Encode ratio M4 Pro, A76, A55 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `silk_wb_16k_mono_20k_vbr_c6_20ms` | -6.4 to -5.5 % | +0.4 to +0.5 % | +0.1 to +0.4 % | -0.4 to +1.2 % | +1.4 to +1.7 % | +0.1 to +0.4 % | 0.522, 0.681, 0.695 |
| `silk_wb_16k_mono_20k_vbr_c7_20ms` | -6.4 to -5.9 % | +0.5 to +0.7 % | -0.0 to +0.3 % | -0.2 to +0.5 % | +1.4 to +1.5 % | -0.1 to +0.3 % | 0.471, 0.588, 0.584 |
| `hybrid_swb_24k_mono_24k_cbr_c6_20ms` | -6.1 to -5.9 % | +0.2 to +0.3 % | -0.1 to +0.7 % | -0.6 to +0.2 % | +0.0 to +0.2 % | -1.8 % | 0.507, 0.629, 0.669 |
| `hybrid_swb_24k_mono_24k_cbr_c7_20ms` | -6.2 to -3.8 % | +0.1 to +0.7 % | +0.2 to +0.3 % | -0.7 to +2.4 % | +0.1 % | -2.5 to -2.1 % | 0.477, 0.571, 0.596 |
| `celt_fb_48k_stereo_128k_vbr_c6_20ms` | -3.5 to -3.4 % | -3.5 to -3.3 % | -2.3 to -2.2 % | -4.1 to -3.5 % | -3.7 to -3.5 % | -3.2 to -2.9 % | 0.536, 0.646, 0.707 |
| `celt_fb_48k_stereo_128k_vbr_c7_20ms` | -4.6 to -3.4 % | -3.4 to -3.3 % | -2.4 to -2.3 % | -4.9 to -3.5 % | -3.4 to -3.2 % | -3.6 to -3.1 % | 0.469, 0.539, 0.580 |

What the table shows:

- On the M4 Pro, SILK and Hybrid encoding at complexity 6 and 7 take 3.8 % to 6.4 % less time. This is the first commit of the SILK quantizer part, which is used on Apple targets only. The slowdown that round 7 left at complexity 7 is gone.
- On the phone, the quantizer with 3 states did not change: its new form for other targets is used for 2 and 4 states only. SILK and Hybrid encoding moved by -0.1 to +0.7 %, as libopus did (+0.1 to +0.5 %). The SILK decoder did not change either; its +1.4 to +1.7 % on the A76 is code placement.
- CELT encoding and decoding at complexity 6 and 7 gain as at the other settings.
- opus1 does the same work at complexity 6 and 7. libopus runs its speech and music analysis at 7, so its ratios at 7 are lower.
- The session alternated the two cores. So the clock statistics of each cluster include the time it waited for the other one, and they cannot show whether the clock was lowered during a run. Runs in turns on one core still see the same state of the phone.

#### Which part each gain comes from

The four parts that change the codec were merged one after the other. A program was built at each merge, and the five programs (the code before round 8 and the four merges) ran in turns: three passes of the bench set on the Mac (Homebrew libopus), and two passes of five scenarios on each core of the phone. The table gives the change of opus1's median time that each merge brought, lowest to highest over the passes. Records: `target/perf/round8/merge/mac/steps/` and `target/perf/round8/merge/android/steps_a76/` and `steps_a55/`.

| Part | Operation and scenario | Apple M4 Pro | Cortex-A76 | Cortex-A55 |
|---|---|---:|---:|---:|
| SILK quantizer | encode, SILK 16 kHz mono c10 | -4.0 to -3.7 % | -0.1 to +0.1 % | -1.3 % |
| | encode, Hybrid 24 kHz mono c9 | -3.9 % | +0.0 to +0.2 % | -1.6 % |
| PVQ search and allocation | encode, CELT 48 kHz mono c10 | -2.6 to -2.4 % | -2.2 to -1.6 % | -1.8 % |
| | encode, CELT 48 kHz stereo c10 | -4.4 to -4.1 % | -4.2 to -4.1 % | -3.4 to -3.1 % |
| | encode, CELT 48 kHz stereo 2.5 ms c9 | -3.0 to -1.0 % | -3.0 to -2.9 % | -2.5 % |
| | decode, CELT 48 kHz mono c10 | -4.2 to -3.9 % | -2.5 to -2.4 % | -1.5 to -1.3 % |
| | decode, CELT 48 kHz stereo c10 | -4.3 to -4.0 % | -2.5 to -2.3 % | -1.5 to -1.2 % |
| | decode, CELT 48 kHz stereo 2.5 ms c9 | -3.2 to -0.6 % | -3.6 to -3.5 % | -3.6 % |
| Pitch analysis | encode, CELT 48 kHz mono c10 | -0.4 to +0.0 % | -0.9 % | -0.3 % |
| | encode, CELT 48 kHz stereo c10 | -0.4 to +0.0 % | -0.6 to -0.5 % | -0.2 to -0.0 % |
| | encode, CELT 48 kHz stereo 2.5 ms c9 | -4.0 to -1.4 % | -0.4 to -0.0 % | -0.5 to -0.3 % |
| CELT decoder | decode, CELT 48 kHz mono c10 | +0.1 to +0.4 % | -1.1 to -1.0 % | -1.5 to -1.4 % |
| | decode, CELT 48 kHz stereo c10 | -0.2 to -0.0 % | -1.0 to -0.7 % | -1.7 % |
| | decode, CELT 48 kHz stereo 2.5 ms c9 | +0.1 % | +0.0 to +0.3 % | +0.3 to +0.4 % |
| | encode, CELT 48 kHz stereo c10 | -0.5 to -0.2 % | -0.1 to +0.0 % | -0.3 to -0.2 % |

What the table shows:

- **The PVQ search and the allocation** bring most of the gain of round 8, on all three cores. The allocation searches and `bits2pulses` run in the decoder too, so this part also brings most of the gain in CELT decoding.
- **The SILK quantizer** brings the whole gain of SILK and Hybrid encoding on the M4 Pro and on the A55. On the A76 it changed SILK and Hybrid encoding by -0.1 to +0.2 %. The form for other targets runs there too, and the form for Apple cores does not.
- **The CELT decoder part** brings 0.7 to 1.7 % in CELT decoding of 20 ms frames on the phone, and nothing on the M4 Pro. Its FFT passes also run in the encoder's MDCT.
- **The pitch analysis** brings 1.4 to 4.0 % in the encoding of 2.5 ms frames on the M4 Pro, where its chain of energies is used. On the phone only its second change runs (no zero-filling). There it saved 0.0 to 0.9 %, which is within code placement.
- Where a part's code does not run, its merge moved a time by at most 1.2 % on the A55, 1.0 % on the A76 and 2.8 % on the M4 Pro, in single passes. That is code placement ([Code placement](#code-placement)). On the A55 it moved Hybrid decoding by -0.9 to -1.2 % twice, in the merges of the SILK quantizer and of the pitch analysis, which do not touch the decoder. So the -1.7 to -1.9 % of Hybrid decoding on the A55 in the table above comes mostly from placement.
- libopus is the same code in all five programs. Between two steps its time changed by at most 1.7 % on the A55 and 3.9 % on the A76 (SILK and Hybrid encoding), as [Code placement](#code-placement) describes.

On the Mac the step runs covered the whole bench set. Over all scenarios of a group (geometric means, lowest to highest over the passes): the SILK quantizer took 2.6 to 2.7 % off SILK encoding and 3.5 to 3.7 % off Hybrid encoding. The PVQ search and allocation took 3.1 to 3.3 % off CELT encoding and 3.9 to 4.2 % off CELT decoding. The pitch analysis took 0.5 to 0.8 % off CELT encoding. The CELT decoder part took 0.3 to 0.4 % off CELT encoding and changed CELT decoding by -0.2 to -0.0 %.

### Round 7: work for in-order cores

Round 7 is the seventh round of speed work on opus1. The measurements on the phone in round 6 ([On an Android phone](#on-an-android-phone)) showed that opus1 lost more of its lead on the in-order Cortex-A55 than on the Cortex-A76 or the M4 Pro. From the A76 to the A55 its encoding grew 3.16 times slower, libopus's 2.90 times. Round 7 worked on the functions where the A55 lost most. Five parts were done separately and then merged (`git log --first-parent 9f462a0..0090242`):

| Part (branch) | Files | What changed |
|---|---|---|
| Pitch analysis (`round7-pitch`) | `src/pitch.rs` | The pitch analysis of the CELT pre-filter: more independent work for `pitch_downsample`, window energies from a table in `find_best_pitch`, a cross-correlation kernel whose groups of lags share windows of the signal (an assembly loop, used on 64-bit ARM except on Apple targets), and the energies of `remove_doubling` summed during its other passes. |
| Decoder post-filter (`round7-decode`) | `src/celt.rs` | The comb filter of the CELT decoder filters both channels in one pass. De-emphasis writes interleaved stereo output directly. |
| SILK quantizers (`round7-nsq`) | `src/silk/nsq.rs`, `src/silk/nsq_del_dec.rs` | NEON for the shaping filter of all states of the delayed-decision quantizer, and a sample loop without branches. NEON for the short-term prediction and the shaping state of the plain quantizer. |
| CELT encoder (`round7-celtenc`) | `src/bands.rs`, `src/celt.rs`, `src/pvq.rs`, `src/rate.rs` | A cache of the cosines of the spreading rotation, allocation searches without data-dependent branches, fewer copies in the pre-filter and the stereo search of `theta_rdo`, and one loop per pulse in the PVQ search of bands of 8 to 64 values. |
| Measurement (`round7-guide`) | `examples/perf_harness.rs`, `scripts/android/`, documents | `perf_harness pairs`, scripts to run builds in turns on the phone, the `-O2` C reference, build settings. No change to the codec. |

Every part keeps the codec output the same. On the Mac, `perf_harness golden --check` against the golden file of `9f462a0` finds 0 of 137 streams that differ after each merge. On the phone, the program of `0090242` writes the same golden file as the program of `9f462a0`, field for field (records in `target/perf/round7/merge/android/golden/`).

The code before round 7 (`9f462a0`) and after it (`0090242`) ran in turns ([Comparing two builds](#comparing-two-builds)): on the Mac three pairs of the bench set with the Homebrew libopus (records in `target/perf/round7/merge/mac/`), on the phone two pairs on each kind of core (records in `target/perf/round7/merge/android/`). The table gives the change of opus1's median time: the geometric mean over the scenarios of the group, lowest and highest over the pairs, and in brackets the range of the single scenarios over all pairs. Below 0 means faster.

| Group | Apple M4 Pro | Cortex-A76 | Cortex-A55 |
|---|---:|---:|---:|
| Encode, all 28 scenarios | -2.8 to -2.5 % (-5.0 to -0.4 %) | -7.5 % (-14.0 to -3.8 %) | -11.1 to -10.8 % (-18.5 to -3.9 %) |
| Encode, SILK (9) | -2.2 to -1.9 % (-3.3 to -0.4 %) | -9.3 to -9.2 % (-14.0 to -4.2 %) | -11.4 to -10.7 % (-18.5 to -4.4 %) |
| Encode, Hybrid (6) | -2.6 to -2.3 % (-2.9 to -0.8 %) | -10.7 to -10.6 % (-11.9 to -9.2 %) | -13.9 to -13.6 % (-15.6 to -11.6 %) |
| Encode, CELT (13) | -3.4 to -2.8 % (-5.0 to -0.4 %) | -4.8 to -4.7 % (-6.0 to -3.8 %) | -9.6 to -9.5 % (-14.5 to -3.9 %) |
| Decode, all 28 scenarios | +0.1 to +0.2 % (-3.9 to +2.1 %) | -1.0 % (-1.7 to -0.2 %) | -2.0 to -1.9 % (-4.4 to +0.7 %) |
| Decode, SILK (9) | -0.4 to +0.6 % (-3.9 to +1.1 %) | -0.7 to -0.6 % (-1.6 to -0.2 %) | -1.7 to -1.6 % (-2.8 to -0.6 %) |
| Decode, Hybrid (6) | +0.4 to +0.5 % (-0.1 to +1.4 %) | -1.3 % (-1.6 to -0.9 %) | -0.3 to +0.0 % (-0.8 to +0.7 %) |
| Decode, CELT (13) | -0.2 to +0.5 % (-0.9 to +2.1 %) | -1.1 % (-1.7 to -0.4 %) | -3.0 % (-4.4 to -1.4 %) |

Every encode scenario was faster in every pair on all three cores. libopus is the same code in both builds: its geometric means changed by -0.3 to +0.1 % on the Mac, +0.1 to +0.5 % on the A76 and +0.0 to +0.5 % on the A55. The code of the SILK decoder did not change, so its change of -0.6 to -1.7 % on the phone and -0.4 to +0.6 % on the Mac comes from the new places of the code and is not a gain of round 7. The other decode changes on the Mac are within the 2.3 % that code placement alone moves a time there ([Code placement](#code-placement)).

#### Which part each gain comes from

The four parts that change the codec were merged one after the other. A program was built at each merge, with the harness of round 7, and the five programs (the code before round 7 and the four merges) ran in turns: three passes of the bench set on the Mac, and two passes of five scenarios on each core of the phone. The table gives the change of opus1's median time that each merge brought, lowest to highest over the passes. Records: `target/perf/round7/merge/mac_steps/` and `target/perf/round7/merge/android/a76_steps/` and `a55_steps/`.

| Part | Operation and scenario | Apple M4 Pro | Cortex-A76 | Cortex-A55 |
|---|---|---:|---:|---:|
| Pitch analysis | encode, CELT 48 kHz mono c10 | -0.3 to +0.0 % | -0.9 to -0.7 % | -9.4 to -9.2 % |
| | encode, CELT 48 kHz stereo c10 | -0.4 to -0.1 % | -0.3 to -0.1 % | -4.3 to -4.0 % |
| | encode, CELT 48 kHz stereo 2.5 ms c9 | +0.2 to +0.5 % | -1.6 to -1.2 % | -4.5 to -4.2 % |
| Decoder post-filter | decode, CELT 48 kHz mono c10 | -0.2 to +0.3 % | -1.2 to -1.0 % | -1.8 to -1.7 % |
| | decode, CELT 48 kHz stereo c10 | -0.6 to -0.4 % | -0.9 % | -3.8 to -3.7 % |
| | decode, CELT 48 kHz stereo 2.5 ms c9 | -1.4 to -1.0 % | +0.2 to +0.3 % | -0.6 to -0.5 % |
| SILK quantizers | encode, SILK 16 kHz mono c10 | -3.2 to -2.9 % | -11.1 to -10.9 % | -14.5 % |
| | encode, Hybrid 24 kHz mono c9 | -3.2 to -2.9 % | -11.5 to -11.4 % | -14.7 % |
| CELT encoder | encode, CELT 48 kHz mono c10 | -2.2 to -2.0 % | -3.6 to -3.2 % | -4.0 to -3.7 % |
| | encode, CELT 48 kHz stereo c10 | -4.9 to -4.5 % | -5.5 to -5.4 % | -6.2 to -5.8 % |
| | encode, CELT 48 kHz stereo 2.5 ms c9 | -4.9 to -4.6 % | -3.5 to -3.0 % | -5.2 to -5.0 % |
| | encode, Hybrid 24 kHz mono c9 | -0.4 to -0.1 % | -0.3 to -0.2 % | -0.8 to -0.7 % |
| | decode, CELT 48 kHz mono c10 | -0.2 to +0.5 % | +0.2 to +0.4 % | -1.0 % |
| | decode, CELT 48 kHz stereo 2.5 ms c9 | +0.5 to +0.6 % | -0.5 to +0.0 % | -2.7 to -2.6 % |

What the table shows:

- **On the A55,** every part gains where it should. The pitch analysis brings most of the gain of CELT encoding at 20 ms in mono, the CELT encoder part most of it in stereo. The SILK quantizers bring the whole gain of SILK and Hybrid encoding. The decoder post-filter brings most of the gain of CELT decoding at 20 ms, and the allocation search of the CELT encoder part (it is shared with the decoder) most of the gain at 2.5 ms.
- **On the A76 and the M4 Pro,** most of the gain in encoding comes from the SILK quantizers and the CELT encoder part. The pitch analysis was written for the A55. On the M4 Pro its new kernel is not used (see `USE_SKEW` in `src/pitch.rs`), and its other changes are within the noise there.
- Where a part's code does not run, for example the SILK quantizers in a CELT scenario, its merge changed the time by at most 0.9 % on the A55, 1.2 % on the M4 Pro and 2.1 % on the A76. That is code placement. On the A76, single steps below about 2 % can therefore be placement, as [Code placement](#code-placement) describes.

The five scenarios of the phone are those of the profiles in [Where the time goes on the phone](#where-the-time-goes-on-the-phone), plus 2.5 ms stereo frames.

### Effect of the refactoring

A refactoring merged duplicate code paths (merge commit `419d5bb`), for example into one `CeltEncoder::encode` that reads interleaved input. The output did not change. At `b63e8c3`, `perf_harness golden --check` against the golden file written at commit `5c8e05e` (`target/perf/golden_final.txt`) finds 0 of 137 entries that differ.

The bench set was also measured at `5c8e05e`, before the refactoring (`target/perf/final_bench.md`). Besides the refactoring, one more change to the code lies between the two commits: `bits2pulses` and `pulses2bits` now check their table indexes (commit `49c2bb4`). That commit measured -1.5 % to +0.4 % for encode and -2.1 % to +0.9 % for decode on the 13 CELT scenarios, within the noise of the machine. From `5c8e05e` to `b63e8c3`, the opus1 / libopus ratio of the medians changed as follows:

- CELT encode: -1.3 % to -14.0 %. The change is -12.8 % (mono) and -14.0 % (stereo) at complexity 0, and -6.0 % to -8.6 % at complexity 5 and 10.
- Decode: -2.4 % to +0.7 %.
- SILK and Hybrid encode: -1.2 % to +0.3 %.

The changes of decode and of SILK and Hybrid encode are within the variation between runs of that measurement: up to 2.1 % for encode ratios and 2.9 % for decode ratios. A measurement right after the merge, at commit `a9c1a54` (`target/perf/simplify_bench.md`), found -1.6 % to -12.0 % for CELT encode.

### Against opus-rs 0.1.37

The same harness measured the code before the performance work: git commit `08214a8` on 2026-10-13, same machine and same libopus (`target/perf/baseline_bench.md`). The table compares the opus1 / libopus ratio of the medians in the two records. libopus is the same build in both, so its time changes only with the state of the machine. The change of the ratio is therefore the change of the opus1 time, with most of the effect of other programs removed.

| Group | Change of opus1 time against libopus time |
|---|---|
| SILK and Hybrid encode (15 scenarios) | 32.7 % to 55.5 % less |
| Mono CELT encode (5 scenarios) | 29.3 % to 49.1 % less |
| Stereo CELT encode, complexity 0 and 5 (2 scenarios) | 23.8 % and 30.0 % less |
| Stereo CELT encode, complexity 9 and 10 (6 scenarios) | 2.5 % to 21.6 % more |
| Decode (all 28 scenarios) | 35.3 % to 62.3 % less |

Stereo CELT encoding at complexity 9 and 10 takes more time because the encoder now does more of libopus's work. Since 0.1.37 it runs the pitch pre-filter on stereo input (complexity 5 and above) and `theta_rdo` (complexity 8 and above). The commit that added `theta_rdo` measured +34 % to +73 % encode time on the six stereo bench scenarios at complexity 9 and 10. Its ratio to libopus stayed between 0.53 and 0.63. In [Latest results](#latest-results), these six scenarios take 0.455 to 0.594 of the time of the Homebrew libopus.

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

### Work for in-order cores

An in-order core such as the Cortex-A55 starts its instructions in program order. When an instruction needs the result of the one before, the core waits, unless independent work stands between the two in the program. Round 7 rewrote the hot loops for such cores ([Round 7: work for in-order cores](#round-7-work-for-in-order-cores)):

- **Independent chains side by side.** `fir5` in `pitch_downsample` handles four blocks per step. The decimation runs one block ahead of the autocorrelation chain. The energy sums of `remove_doubling` run inside its other passes. The comb filter of the decoder filters both channels in one pass.
- **Fewer loads.** The A55 takes 2 cycles for a 16-byte load and does not start it together with a vector operation. The new cross-correlation kernel lets groups of lags share each window of the signal and holds 64 lags in 16 sums. Its steady loop is written in assembly, because the compiler put dependent multiply-adds next to each other. The comb filter for stereo frames loads each sample once and takes the shifted taps from registers with `vextq_f32`.
- **No branches in the quantizer loop.** The delayed-decision quantizer of SILK runs its sample loop without data-dependent branches, and the shaping filter of all its states in NEON registers. The plain quantizer keeps its shaping state in NEON registers for a whole subframe.
- **Less work.** The encoder keeps the cosines of the spreading rotation in a cache. The history of the pre-filter is moved once per frame instead of copied twice. The PVQ search checks the position of one pulse and searches the position of the next in one loop.

Round 8 went on in the same way ([Round 8: more work for in-order cores](#round-8-more-work-for-in-order-cores)):

- **Searches without chains of dependent steps.** `bits2pulses` counted the costs of a row by bisection, and each step waited for its load. Now three vector compares count all costs below the budget at once. The two allocation searches add up the bits of all bands with NEON. The scalar loop of libopus decides band after band, from the last one down.
- **Fewer instructions per pulse.** The check of each pulse of the PVQ search takes 5 instead of 9 instructions, and the A55 was limited by the instruction count. The next pulse is placed in advance, so the first pass of the next search needs no extra instructions for it.
- **Values that stay in registers.** In the delayed-decision quantizer the compiler had made one vector per shaping coefficient and reloaded 16 to 24 of them from the stack in every sample. An empty `asm!` statement keeps the 6 packed vectors instead (not on Apple targets).
- **Two independent groups per step.** The radix-4 and radix-2 passes of the FFT, the pre-rotation of the inverse MDCT and the de-emphasis filters each handle two groups that do not depend on each other, so the A55 works on one while the other waits.
- **No calls into the C library.** The cosines of the spreading rotation come from inline copies of the cosine kernels of the C library. The decoder spent 2 % of its time on the A55 in `cos`.

Each change keeps the order of the floating-point operations and the integer results, so the output stays the same.

## What was tried and did not help

These results come from `target/perf/cpu_features.md`, the profile reports and, for the rows of round 8, the reports of its parts.

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
| Round 8: the first taps of the shaping filter for both candidates on the phone | The form that removes the slowdown at complexity 7 on Apple M4 made the Cortex-A55 1 % to 6 % slower and changed the A76 by -2 to +2 %. The cores of the phone do not wait for the chain of filter, decision and survivors. It is used on Apple targets only. |
| Round 8: a 64-bit shaping filter for 2 states | Cortex-A55 8 % to 11 % slower, Cortex-A76 3.2 % to 3.7 % faster. The A55 waits on dependent instructions, not on the width of the vectors. |
| Round 8: a first pass of the PVQ search without division | Cross products per lane instead of the quotient: 15 % slower on Apple M4. |
| Round 8: other forms of `icwrs` and `cwrsi` | Two-phase, unrolled and branchless versions were 5 % to 20 % slower on both cores of the phone; the branchless `cwrsi` was 4 times slower on Apple M4. |
| Round 8: the energies of `remove_doubling` inside `pitch_search` on the phone | Search and doubling 8 % to 20 % slower on the Cortex-A55 and A76. 12 % to 16 % faster on Apple M4 for 2.5 to 10 ms frames, so it is used on Apple targets only. |
| Round 8: an inverse MDCT in place | No change in the level-1 data cache refills of the Cortex-A55; the Cortex-A76 got 0.3 % to 0.9 % slower. |
| Round 8: cold paths marked with `#[cold]` in the decoder | The loss concealment and other rare paths moved out of the hot functions. Cortex-A55 0.2 % to 0.4 % slower, Hybrid 1.3 % to 1.5 %. |

## Build settings for applications

Cargo takes the release profile from the top-level package, so opus1 cannot set it for you. [Build settings for speed](06-using-the-library.md#build-settings-for-speed) gives the recommendations. This section gives the measurements behind them.

Measured on 2026-10-17 with the code of commit `9f462a0`. Each build ran in turns with the default build, three passes (see [Comparing two builds](#comparing-two-builds)). The records are in `target/perf/round7/guide/prof/` (Mac) and `target/perf/round7/guide/android/` (phone: `prof_a55`, `prof_a76`, `a76_on_a55`, `v80_a55`, `v80_a76`, `v80lto_a55`).

- **Mac** (Apple M4 Pro): the bench set, 28 scenarios, against the Homebrew libopus. Other programs changed the load between runs, so the table gives the change of the opus1 / libopus ratio. libopus is a separate library there, and the build of the harness does not change it.
- **Phone** (Cortex-A76 on CPU 7, Cortex-A55 on CPU 5): four scenarios, `silk_wb_16k_mono_20k_vbr_c10_20ms`, `hybrid_swb_24k_mono_24k_cbr_c9_20ms`, `celt_fb_48k_mono_64k_vbr_c10_20ms` and `celt_fb_48k_stereo_128k_vbr_c10_20ms`. The table gives the change of the opus1 time. Runs in turns see the same clock.

The release profile in the table is:

```toml
[profile.release]
codegen-units = 1
lto = "fat"
panic = "abort"   # only if the application does not catch panics
```

`codegen-units = 1` compiles each crate as one unit, so more functions can be inlined. `lto = "fat"` (link-time optimization) lets the compiler optimize across crates. `panic = "abort"` removes the code that unwinds the stack after a panic.

"v8.0" names `-C target-cpu=cortex-a55` together with `-C target-feature=-lse,-rdm,-dotprod,-fp16,-rcpc,-ras,-lor,-pan,-vh,-dpb,-crc,-aes,-sha2`. These features remove the extensions that `cortex-a55` adds to ARMv8.0-A. The compiler still orders the instructions for the Cortex-A55 and aligns the functions to 16 bytes.

Change of the time against the default build: the geometric mean over the scenarios, lowest to highest over the three passes. Below 0 means faster.

| Build | M4 Pro encode | M4 Pro decode | A76 encode | A76 decode | A55 encode | A55 decode |
|---|---:|---:|---:|---:|---:|---:|
| Release profile | -0.2 to +0.1 % | -1.7 to -1.6 % | -1.5 to -1.1 % | -0.3 to +0.2 % | -0.6 to -0.4 % | -0.7 to -0.4 % |
| `-C target-cpu=native` | -0.1 to +0.3 % | -0.1 to +0.3 % | | | | |
| `-C target-cpu=native`, release profile | -0.1 to +0.3 % | -1.5 to -1.3 % | | | | |
| `-C target-cpu=cortex-a55` | | | +0.0 to +0.5 % | -0.7 to -0.2 % | -4.8 to -4.2 % | -2.2 to -1.5 % |
| `-C target-cpu=cortex-a55`, release profile | | | -1.3 % | -0.6 to -0.3 % | -4.4 to -4.3 % | -2.0 to -1.7 % |
| v8.0 | | | +0.5 to +0.7 % | -1.4 to -1.1 % | -5.1 to -4.6 % | -2.0 to -1.7 % |
| v8.0, release profile | | | -0.7 to -0.5 % | -1.4 to -1.1 % | -4.5 to -4.4 % | -1.5 to -1.3 % |
| `-C target-cpu=cortex-a53` | | | +0.6 to +1.3 % | -0.7 to -0.0 % | -2.3 to -1.8 % | -1.0 to -0.5 % |
| `-C target-cpu=cortex-a76` | | | -0.9 to -0.4 % | +0.1 to +0.7 % | +15.2 to +15.6 % | +7.4 to +7.8 % |

How to read the table:

- The release profile helps a little everywhere. The largest effects are 1.7 % less decode time on the M4 Pro and 1.1 % to 1.5 % less encode time on the Cortex-A76.
- `-C target-cpu=native` changes nothing on the M4 Pro: the target `aarch64-apple-darwin` already includes the instructions that the codec uses.
- Ordering the instructions for the in-order Cortex-A55 makes encoding 4 % to 5 % faster and decoding about 2 % faster on that core, in every scenario and every pass. The `cortex-a55` rows of the A55 come from two sessions.
- The v8.0 build gains as much on the A55 as `cortex-a55`. Its program has none of the instructions of the LSE (atomics), RCPC, RDM and DotProd extensions, which the `cortex-a55` build with the release profile contains 13 times (checked with `llvm-objdump`). So the gain comes from the order of the instructions and the alignment of the functions, not from newer instructions. Such a program also runs on cores of ARMv8.0-A, such as the Cortex-A53. Two of the v8.0 results come from a second set of builds, made after a change to the harness: the A76 columns of both rows and the A55 columns of the row with the release profile. Their base is the default build of that set.
- `cortex-a53` orders the instructions for the older in-order core. It gains less than half as much on the A55, and its encoding is slower on the A76.
- `cortex-a76` makes the A55 15 % slower in encoding and 7.6 % slower in decoding (two scenarios: SILK 16 kHz and CELT 48 kHz mono). It gains less than 1 % on the A76.
- On the A76, changes below about 2 % in one scenario can be code placement (see [Code placement](#code-placement)). In the same `cortex-a55` and `cortex-a76` builds, libopus's SILK and Hybrid encoding took 3.4 % to 5.2 % more time, although its code did not change.

The settings do not change the output. On the Mac, `golden --check` found 0 of 137 streams that differ for the release profile and for `native` with it. On the phone, the v8.0 build with the release profile reports exactly the same differences against the Mac golden file as the default build. At that time the phone linked the main branch of libopus, so the fields that depend on libopus's packets differed (see [Testing](09-testing.md#which-libopus-source-to-build)). The fields of opus1's own output were equal.

An earlier measurement at commit `2cf93bf`, before most of the performance work, found the release profile about 3 % faster in decoding on the M4 Pro (`target/perf/prof/memory-build/report.md`, section 6).

The `heap` feature (on by default) moves the large codec state to the heap when an encoder or decoder is created. It does not allocate per frame. Measured at commit `2cf93bf`: `OpusEncoder::new` made 11 allocations and `OpusDecoder::new` made 7. Then 300 calls each of `encode`, `encode_i16`, `decode`, and `decode` with lost packets made none. In the current code `OpusEncoder::new` makes 7 allocations for mono input and 8 for stereo input, and `OpusDecoder::new` makes 5. One of the decoder's is the float buffer of `decode_i16`, so no decode call allocates. [Size of the encoder and the decoder](06-using-the-library.md#size-of-the-encoder-and-the-decoder) gives them with their sizes.

## Reproducing the numbers

1. Install libopus 1.6.1 and make it visible to `audiopus_sys`. See [How libopus is linked](09-testing.md#how-libopus-is-linked). The harness prints the version it linked, and the golden file header records it. For the `-O2` reference of [The C reference](#the-c-reference), build the release with `scripts/android/build-libopus.sh --host --rtcd --src <1.6.1 source>` and link a harness in its own cargo directory against it, as the top of the script shows.
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
use std::num::NonZeroU32;
use opus1::{Application, Bitrate, Channels, OpusEncoder, SampleRate};
use std::time::Instant;

let mut encoder = OpusEncoder::new(SampleRate::Hz48000, Channels::Stereo, Application::Audio).unwrap();
encoder.set_bitrate(Bitrate::BitsPerSecond(NonZeroU32::new(128_000).unwrap()));
encoder.set_complexity(5).unwrap();

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

Measured on 2026-10-18 with the code of git commit `f952b46` (the speech and music analysis and DTX on `d541821`), with the scripts in `scripts/android/` (see [Testing](09-testing.md#tests-on-an-android-phone)). The phone is a HONOR CLK-AN00 with a MediaTek MT6833 and Android 14. The harness was pinned with `taskset` to one core of each kind:

- CPU 7, a Cortex-A76 at up to 2.4 GHz. It reorders instructions, so independent ones run at the same time.
- CPU 5, a Cortex-A55 at up to 2.0 GHz. It runs instructions in program order, at most two per cycle.

opus1 was built with the default release profile. The C reference is libopus 1.6.1, built by `scripts/android/build-libopus.sh --src <1.6.1 source>` with the NDK's clang 18. The script uses the flags that libopus's own `configure` picks for Android: `-O2`, the NEON code presumed, and run-time CPU detection. So the SILK encoder runs libopus's NEON quantizer, as the `-O2` reference on the Mac does (see [The C reference](#the-c-reference)).

The code before the analysis (`d541821`) and after it (`f952b46`) ran in turns as one job, in the way of `ab.sh`: two passes of the bench set and of the two scenarios at complexity 7, on each core. The tables below give the runs of the code after the analysis. While a core ran, its cluster was at its highest clock: the A76 cluster spent 24.1 % of the session at 2.4 GHz and nearly all the rest idle at 725 MHz while the A55 ran, and the A55 cluster spent 70.0 % at 2.0 GHz. The processor was at 32 °C when the session started and reached at most 45.7 °C. Between the two runs of the same code on one core, the opus1 / libopus ratio of a scenario changed by at most 0.9 % (A76) and 0.6 % (A55). The records are in `target/perf/analysis_c7/phone_bench/`.

Encoding at complexity 7 to 10, before and after the analysis. Before, only libopus ran the analysis; after, both do. Times are opus1's median per frame (the lower of the two runs), the ratio is opus1 / libopus (the mean of the two runs):

| Scenario | A76 µs before | after | added | A76 ratio before | after | A55 µs before | after | added | A55 ratio before | after |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `silk_wb_16k_mono_20k_vbr_c10_20ms` | 142.8 | 155.8 | +13.0 | 0.580 | 0.633 | 384.1 | 427.9 | +43.8 | 0.591 | 0.659 |
| `silk_wb_16k_mono_20k_vbr_c9_10ms` | 81.4 | 88.1 | +6.7 | 0.609 | 0.659 | 218.2 | 241.8 | +23.6 | 0.617 | 0.683 |
| `silk_wb_16k_stereo_32k_vbr_c9_20ms` | 286.2 | 300.7 | +14.6 | 0.636 | 0.668 | 767.8 | 812.8 | +45.0 | 0.650 | 0.689 |
| `silk_wb_16k_mono_24k_vbr_c9_20ms_fec10` | 177.4 | 190.7 | +13.3 | 0.572 | 0.617 | 465.8 | 509.5 | +43.7 | 0.595 | 0.650 |
| `hybrid_swb_24k_mono_24k_cbr_c9_20ms` | 191.0 | 202.3 | +11.3 | 0.570 | 0.605 | 523.5 | 561.5 | +38.0 | 0.603 | 0.647 |
| `hybrid_swb_24k_stereo_36k_cbr_c9_20ms` | 380.4 | 392.8 | +12.4 | 0.630 | 0.651 | 1028.8 | 1063.3 | +34.5 | 0.660 | 0.683 |
| `hybrid_fb_48k_mono_32k_cbr_c9_20ms` | 195.1 | 210.4 | +15.2 | 0.598 | 0.646 | 538.8 | 581.0 | +42.2 | 0.622 | 0.672 |
| `hybrid_fb_48k_stereo_40k_cbr_c9_20ms` | 395.4 | 411.7 | +16.3 | 0.637 | 0.664 | 1065.2 | 1109.0 | +43.8 | 0.656 | 0.686 |
| `celt_fb_48k_mono_64k_vbr_c10_20ms` | 58.8 | 73.2 | +14.4 | 0.473 | 0.588 | 199.8 | 247.5 | +47.7 | 0.515 | 0.638 |
| `celt_fb_48k_stereo_128k_vbr_c10_20ms` | 147.8 | 163.7 | +15.9 | 0.597 | 0.658 | 471.6 | 519.6 | +48.1 | 0.636 | 0.704 |
| `celt_fb_48k_stereo_128k_vbr_c9_2p5ms` | 31.5 | 34.1 | +2.6 | 0.680 | 0.730 | 120.7 | 129.9 | +9.1 | 0.766 | 0.819 |
| `celt_fb_48k_stereo_128k_vbr_c9_5ms` | 51.5 | 56.3 | +4.7 | 0.656 | 0.713 | 184.0 | 199.0 | +15.0 | 0.735 | 0.792 |
| `celt_fb_48k_stereo_128k_vbr_c9_10ms` | 77.4 | 86.3 | +8.9 | 0.612 | 0.678 | 260.1 | 287.2 | +27.1 | 0.666 | 0.736 |
| `celt_swb_24k_mono_48k_vbr_c9_20ms` | 51.2 | 62.4 | +11.2 | 0.473 | 0.575 | 178.4 | 218.5 | +40.1 | 0.516 | 0.633 |
| `celt_swb_24k_stereo_96k_vbr_c9_20ms` | 133.0 | 145.8 | +12.8 | 0.604 | 0.661 | 433.1 | 475.0 | +42.0 | 0.642 | 0.706 |
| `celt_fb_48k_mono_24k_vbr_c9_20ms` | 46.4 | 60.3 | +14.0 | 0.429 | 0.558 | 166.5 | 212.0 | +45.4 | 0.476 | 0.609 |
| `celt_fb_48k_stereo_256k_vbr_c9_20ms` | 184.0 | 199.5 | +15.5 | 0.653 | 0.706 | 572.2 | 618.8 | +46.6 | 0.706 | 0.766 |
| `hybrid_fb_48k_mono_32k_cbr_c9_20ms_music` | 178.2 | 192.6 | +14.4 | 0.628 | 0.679 | 493.8 | 536.2 | +42.5 | 0.641 | 0.698 |
| `hybrid_fb_48k_stereo_40k_cbr_c9_20ms_music` | 359.7 | 375.9 | +16.2 | 0.662 | 0.693 | 977.7 | 1022.3 | +44.6 | 0.675 | 0.708 |
| `silk_wb_16k_mono_20k_vbr_c7_20ms` | 123.2 | 136.5 | +13.3 | 0.588 | 0.652 | 336.1 | 379.4 | +43.3 | 0.584 | 0.659 |
| `celt_fb_48k_stereo_128k_vbr_c7_20ms` | 100.3 | 116.1 | +15.7 | 0.538 | 0.619 | 331.3 | 379.2 | +47.9 | 0.581 | 0.667 |

- The analysis adds 11.2 to 16.3 µs per 20 ms frame and stream on the A76 and 34.5 to 48.1 µs on the A55. The analysis alone takes about 11.6 and 34.7 µs at 48 kHz (see [The speech and music analysis](#the-speech-and-music-analysis)).
- Over these 21 scenarios opus1's encode time grew by 10.5 % (A76) and 10.7 % (A55), and the geometric mean of the encode ratio went from 0.588 to 0.649 (A76) and from 0.622 to 0.689 (A55). Over the 9 other scenarios opus1's time moved by +0.6 % and +1.1 %, code placement.

The paragraphs below describe earlier measurements on the phone.

Round 8 (`b7a8b64`, 2026-10-17): `ab.sh` ran the code before round 8 (`c96e936`) and after it in turns, two pairs of the bench set on each core (see [Round 8: more work for in-order cores](#round-8-more-work-for-in-order-cores)). The A76 ran both pairs in one session, the A55 one pair in each of two sessions. The cluster of the pinned core spent 86.8 % (A76) and 94.2 % to 94.3 % (A55) of each session at its highest clock (cpufreq statistics). Most of the rest falls in the pauses between the measuring processes. The processor was at 32 °C to 35 °C when a session started and reached at most 47.8 °C. Between the two runs of the merged code on one core, the median time of an operation changed by at most 0.9 % on the A76 and 1.1 % on the A55, and the opus1 / libopus ratio by at most 1.2 % on both. The records are in `target/perf/round8/merge/android/` (`a76/`, `a55_p1/` and `a55_p2/`).

Before the merge, the code of `c96e936` was measured with `bench.sh`, two runs on each core (records in `target/perf/round8/ref/phone_bench/`). It gave the encode ratios 0.628 (A76) and 0.663 (A55) and the decode ratios 0.773 and 0.842. In the runs of `c96e936` in turns with the merged code, they were 0.635 and 0.663, and 0.774 and 0.842. The A76 encode ratio differs by 1.1 % between two builds of the same code: on the A76, code placement alone moves libopus's SILK and Hybrid encoding by up to 4 % ([Code placement](#code-placement)).

Round 7 measured the code of `c96e936` with the main branch of libopus (commit `228a0f85`) as the reference, built by the same script at `-O2` with run-time detection (records in `target/perf/round7/merge/android/`). From that session to the `bench.sh` session of `c96e936`, libopus's median times changed by -0.7 % to +1.1 % per scenario on the A55. On the A76 they changed by -1.1 % to +0.4 %, except SILK and Hybrid encoding at complexity 9 and 10, which took 1.3 % to 3.9 % more time. On the A76, code placement alone moves a libopus time by up to 4 % (see [Code placement](#code-placement)). opus1's times changed by -1.3 % to +0.8 % on the A76 and -0.6 % to +1.6 % on the A55. So the source of libopus makes no measurable difference to its speed on the phone either.

The measurement of round 6, at commit `bd080de`, was checked later on the same day with a harness built in another directory and run from another directory of the phone (records in `target/perf/android/check/`):

- In the first run, the phone kept the A76 cluster at 2.118 GHz and the A55 cluster at 1.75 GHz, while the processor was at 43 °C to 50 °C. The times grew by the ratio of the clocks: 1.13 times on the A76 and 1.14 times on the A55. The geometric means of the ratios stayed within 0.007. `bench.sh` now prints a warning when a cluster spends less than 75 % of a run at its highest clock, and `ab.sh` prints the share of each clock.
- In SILK and Hybrid scenarios on the A76, libopus took up to 3.8 % more time to encode and opus1 up to 2.2 % less time to decode, although the code was the same: a longer directory name compiled into the new program moved its code 16 bytes further. So on the A76, a difference of about 3 % in one scenario between two builds is not a real change (see [Code placement](#code-placement)).

### The C reference matters

The Homebrew libopus behaves like a build at `-Os`. The phone's libopus and the `-O2` reference on the Mac are the same release, built at `-O2` with run-time detection (see [The C reference](#the-c-reference)). So the table gives the ratios on the Mac against both. The Mac columns are those of [Latest results](#latest-results).

Ratio opus1 / libopus of the median times, over the 28 scenarios of the bench set. Below 1 means opus1 is faster.

| | M4 Pro, Homebrew libopus | M4 Pro, libopus 1.6.1 at `-O2` | Cortex-A76 | Cortex-A55 |
|---|---:|---:|---:|---:|
| Encode, geometric mean | 0.461 | 0.572 | 0.668 | 0.698 |
| Encode, range | 0.368 to 0.625 | 0.477 to 0.711 | 0.558 to 0.775 | 0.609 to 0.819 |
| Decode, geometric mean | 0.598 | 0.669 | 0.768 | 0.824 |
| Decode, range | 0.487 to 0.756 | 0.558 to 0.807 | 0.675 to 0.853 | 0.731 to 0.935 |
| Encode, geometric mean before the analysis | 0.443 | 0.546 | 0.625 | 0.650 |
| Encode, geometric mean in round 8 | 0.442 | 0.544 | 0.625 | 0.651 |
| Decode, geometric mean in round 8 | 0.607 | 0.680 | 0.759 | 0.821 |
| Encode, geometric mean before round 8 | 0.458 | 0.565 | 0.635 | 0.663 |
| Decode, geometric mean before round 8 | 0.619 | 0.695 | 0.774 | 0.842 |

The first four rows are the code after the analysis, in which both encoders run the analysis from complexity 7 on. A check at commit `c7744f6`, after the neural network features were merged (default features, so none of them is built), gave the same picture on the M4 Pro in two runs of the bench set: encode 0.566 and 0.568, decode 0.684 and 0.685 against `-O2`; encode 0.461 and 0.462, decode 0.606 against Homebrew. Records: `target/perf/final/`. The row "before the analysis" comes from the runs of `d541821` in the same sessions, in which only libopus ran it. The rows of round 8 come from the sessions of round 8: "in round 8" is `b7a8b64`, "before round 8" is `c96e936`. Round 7 measured that code against the main branch of libopus, built with run-time detection: 0.567 and 0.694 on the Mac at `-O2`, 0.636 and 0.771 on the A76, 0.663 and 0.840 on the A55. Round 6 had measured 0.692 and 0.755 for encoding and 0.790 and 0.864 for decoding on the phone, at commit `bd080de`, before the work of round 7.

### Results per scenario

Times are opus1's median per frame in microseconds (µs), the lower of the two runs of the code after the analysis (`f952b46`). The ratio is opus1 / libopus, the mean of the two runs.

| Scenario | A76 encode µs | A76 encode ratio | A55 encode µs | A55 encode ratio | A76 decode µs | A76 decode ratio | A55 decode µs | A55 decode ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `silk_nb_8k_mono_12k_vbr_c0_20ms` | 31.7 | 0.775 | 98.4 | 0.717 | 5.9 | 0.846 | 15.8 | 0.935 |
| `silk_nb_8k_mono_12k_vbr_c5_20ms` | 63.0 | 0.761 | 180.0 | 0.687 | 6.1 | 0.843 | 16.2 | 0.929 |
| `silk_nb_8k_mono_12k_vbr_c10_20ms` | 83.4 | 0.708 | 233.4 | 0.743 | 6.0 | 0.845 | 16.2 | 0.932 |
| `silk_wb_16k_mono_20k_vbr_c0_20ms` | 49.7 | 0.730 | 144.9 | 0.657 | 10.3 | 0.740 | 27.7 | 0.814 |
| `silk_wb_16k_mono_20k_vbr_c5_20ms` | 107.2 | 0.728 | 291.4 | 0.638 | 11.0 | 0.744 | 29.9 | 0.812 |
| `silk_wb_16k_mono_20k_vbr_c10_20ms` | 155.8 | 0.633 | 427.9 | 0.659 | 11.1 | 0.745 | 29.9 | 0.812 |
| `silk_wb_16k_mono_20k_vbr_c9_10ms` | 88.1 | 0.659 | 241.8 | 0.683 | 5.9 | 0.749 | 17.2 | 0.831 |
| `silk_wb_16k_stereo_32k_vbr_c9_20ms` | 300.7 | 0.668 | 812.8 | 0.689 | 21.0 | 0.754 | 57.2 | 0.842 |
| `silk_wb_16k_mono_24k_vbr_c9_20ms_fec10` | 190.7 | 0.617 | 509.5 | 0.650 | 12.3 | 0.774 | 32.6 | 0.839 |
| `hybrid_swb_24k_mono_24k_cbr_c9_20ms` | 202.3 | 0.605 | 561.5 | 0.647 | 22.9 | 0.709 | 72.6 | 0.767 |
| `hybrid_swb_24k_stereo_36k_cbr_c9_20ms` | 392.8 | 0.651 | 1063.3 | 0.683 | 41.2 | 0.684 | 130.2 | 0.746 |
| `hybrid_fb_48k_mono_32k_cbr_c9_20ms` | 210.4 | 0.646 | 581.0 | 0.672 | 26.1 | 0.697 | 82.1 | 0.747 |
| `hybrid_fb_48k_stereo_40k_cbr_c9_20ms` | 411.7 | 0.664 | 1109.0 | 0.686 | 46.1 | 0.679 | 147.4 | 0.735 |
| `celt_fb_48k_mono_64k_vbr_c0_20ms` | 34.9 | 0.707 | 108.7 | 0.790 | 21.4 | 0.836 | 59.8 | 0.868 |
| `celt_fb_48k_mono_64k_vbr_c5_20ms` | 59.0 | 0.630 | 201.3 | 0.701 | 24.9 | 0.776 | 71.6 | 0.821 |
| `celt_fb_48k_mono_64k_vbr_c10_20ms` | 73.2 | 0.588 | 247.5 | 0.638 | 25.3 | 0.778 | 72.5 | 0.824 |
| `celt_fb_48k_stereo_128k_vbr_c0_20ms` | 66.5 | 0.677 | 214.5 | 0.779 | 40.9 | 0.853 | 113.5 | 0.901 |
| `celt_fb_48k_stereo_128k_vbr_c5_20ms` | 101.9 | 0.654 | 334.7 | 0.717 | 49.2 | 0.799 | 137.1 | 0.836 |
| `celt_fb_48k_stereo_128k_vbr_c10_20ms` | 163.7 | 0.658 | 519.6 | 0.704 | 49.3 | 0.800 | 136.9 | 0.835 |
| `celt_fb_48k_stereo_128k_vbr_c9_2p5ms` | 34.1 | 0.730 | 129.9 | 0.819 | 9.0 | 0.788 | 31.1 | 0.832 |
| `celt_fb_48k_stereo_128k_vbr_c9_5ms` | 56.3 | 0.713 | 199.0 | 0.792 | 15.4 | 0.808 | 49.6 | 0.861 |
| `celt_fb_48k_stereo_128k_vbr_c9_10ms` | 86.3 | 0.678 | 287.2 | 0.736 | 26.7 | 0.812 | 79.7 | 0.867 |
| `celt_swb_24k_mono_48k_vbr_c9_20ms` | 62.4 | 0.575 | 218.5 | 0.633 | 21.0 | 0.786 | 61.9 | 0.808 |
| `celt_swb_24k_stereo_96k_vbr_c9_20ms` | 145.8 | 0.661 | 475.0 | 0.706 | 39.1 | 0.793 | 112.2 | 0.807 |
| `celt_fb_48k_mono_24k_vbr_c9_20ms` | 60.3 | 0.558 | 212.0 | 0.609 | 16.9 | 0.708 | 54.5 | 0.765 |
| `celt_fb_48k_stereo_256k_vbr_c9_20ms` | 199.5 | 0.706 | 618.8 | 0.766 | 65.5 | 0.828 | 168.2 | 0.863 |
| `hybrid_fb_48k_mono_32k_cbr_c9_20ms_music` | 192.6 | 0.679 | 536.2 | 0.698 | 26.2 | 0.700 | 82.7 | 0.752 |
| `hybrid_fb_48k_stereo_40k_cbr_c9_20ms_music` | 375.9 | 0.693 | 1022.3 | 0.708 | 45.9 | 0.675 | 148.0 | 0.731 |
| `silk_wb_16k_mono_20k_vbr_c7_20ms` | 136.5 | 0.652 | 379.4 | 0.659 | 11.1 | 0.744 | 29.9 | 0.815 |
| `celt_fb_48k_stereo_128k_vbr_c7_20ms` | 116.1 | 0.619 | 379.2 | 0.667 | 49.5 | 0.803 | 137.7 | 0.836 |

How to read the table:

- opus1 is faster than libopus in every scenario, for encoding and decoding, on both cores, also at complexity 7 to 10, where both run the analysis.
- The slowest operation on the A55, the encoding of 48 kHz stereo Hybrid, takes 1.11 ms per 20 ms frame. That is 5.5 % of one core.
- From the M4 Pro to the A76, the time of opus1 grows 3.10 times for encoding and 2.83 times for decoding (geometric means over the bench set; the M4 Pro times of [Latest results](#latest-results)). The time of the `-O2` libopus build grows 2.66 and 2.47 times.
- From the A76 to the A55, the time of opus1 grows 3.03 times for encoding and 2.90 times for decoding. The time of libopus grows 2.90 and 2.71 times.
- The A55 still costs opus1 most in CELT encoding at complexity 5 and above: 3.1 to 3.8 times its A76 time, against 2.9 to 3.4 times for libopus.
- SILK encoding at complexity 0 and 5 is the other way round: opus1 grows 2.7 to 3.1 times, libopus 3.1 to 3.4 times.

### Where the time goes on the phone

`scripts/android/profile.sh` counted hardware events per frame with `simpleperf stat` and sampled the hot functions with `simpleperf record`, for opus1 and libopus on both cores. Measured on 2026-10-17 with the code of commit `c96e936`, before round 8, and libopus 1.6.1, the builds of the `bench.sh` measurement of that code. The records are in `target/perf/round8/ref/profile/`. The program for this keeps line tables and symbols. Its times per frame are within 1.6 % of the bench for CELT and for decoding, and 0.7 % below to 8.2 % above it for SILK and Hybrid encoding, for both libraries. One profile was measured again: libopus's CELT stereo encoding on the A55. In its first run the cluster averaged 1.86 GHz instead of 1.99 GHz while the events were counted, so its counts per frame came out 5 % too low. In the table, IPC is the number of instructions per cycle.

| Scenario and operation | instructions, opus1 / libopus | cycles on A76, opus1 / libopus | cycles on A55, opus1 / libopus | IPC on A76, opus1 and libopus | IPC on A55, opus1 and libopus |
|---|---:|---:|---:|---:|---:|
| SILK 16 kHz mono c10, encode (`silk_wb_16k_mono_20k_vbr_c10_20ms`) | 0.58 | 0.58 | 0.60 | 2.13 and 2.14 | 0.93 and 0.96 |
| Hybrid 24 kHz mono c9, encode (`hybrid_swb_24k_mono_24k_cbr_c9_20ms`) | 0.56 | 0.58 | 0.61 | 2.06 and 2.13 | 0.89 and 0.98 |
| Hybrid 24 kHz mono c9, decode (`hybrid_swb_24k_mono_24k_cbr_c9_20ms`) | 0.63 | 0.70 | 0.79 | 2.14 and 2.40 | 0.78 and 0.97 |
| CELT 48 kHz mono c10, encode (`celt_fb_48k_mono_64k_vbr_c10_20ms`) | 0.54 | 0.48 | 0.53 | 2.27 and 2.03 | 0.80 and 0.78 |
| CELT 48 kHz mono c10, decode (`celt_fb_48k_mono_64k_vbr_c10_20ms`) | 0.79 | 0.80 | 0.86 | 2.10 and 2.12 | 0.87 and 0.94 |
| CELT 48 kHz stereo c10, encode (`celt_fb_48k_stereo_128k_vbr_c10_20ms`) | 0.68 | 0.62 | 0.66 | 2.16 and 1.97 | 0.82 and 0.80 |
| CELT 48 kHz stereo c10, decode (`celt_fb_48k_stereo_128k_vbr_c10_20ms`) | 0.80 | 0.82 | 0.87 | 2.04 and 2.09 | 0.87 and 0.94 |

The same table was measured in round 6, at commit `bd080de`, with the main branch of libopus (records in `target/perf/android/profile/`). Since then:

- opus1 runs 23 % fewer instructions per frame in SILK encoding and 26 % fewer in Hybrid encoding, 4 % to 5 % fewer in CELT encoding and up to 2 % fewer in decoding. libopus 1.6.1 runs within 1 % of the instructions of the main branch.
- So opus1 now runs 0.54 to 0.80 of libopus's instructions (round 6: 0.57 to 0.82). It needs 0.48 to 0.82 of libopus's cycles on the A76 (round 6: 0.52 to 0.84) and 0.53 to 0.87 on the A55 (round 6: 0.63 to 0.92).

What the counters show:

- On the A76, opus1 runs more instructions per cycle than libopus in CELT encoding (2.16 to 2.27 against 1.97 to 2.03). It runs about as many in SILK and Hybrid encoding (2.13 and 2.06 against 2.14 and 2.13) and fewer in decoding (2.04 to 2.14 against 2.09 to 2.40).
- On the A55, opus1 runs a few more instructions per cycle than libopus in CELT encoding (0.80 and 0.82 against 0.78 and 0.80). In round 6 it ran fewer (0.71 and 0.76). In SILK and Hybrid encoding (0.93 and 0.89 against 0.96 and 0.98) and in decoding (0.78 to 0.87 against 0.94 to 0.97) it runs fewer.
- The cycles of single functions below are their share of the samples times the cycles per frame:
  - The SILK delayed-decision quantizer of opus1 (`quantize_subframe_neon::<4, 24>`) takes 0.58 of the cycles of libopus's NEON quantizer (`silk_NSQ_del_dec_neon`) on the A76, and 0.64 on the A55. In round 6 it took 0.68 and 0.99.
  - The pitch analysis of the CELT encoder (the functions of `pitch.rs`, against libopus's `celt_pitch_xcorr_float_neon`, `pitch_search`, `pitch_downsample`, `remove_doubling` and their inner products) takes 0.51 of libopus's cycles on the A76 and 0.46 on the A55 (48 kHz mono, complexity 10). On the A55 it takes 3.1 times its A76 cycles, libopus's 3.4 times. In round 6, the cross-correlation of opus1 alone took 5.1 to 5.4 times its A76 cycles on the A55.
  - The post-filter of the CELT decoder (`celt::comb_filter_inplace::<2>`, 48 kHz stereo) takes 0.16 of the cycles of libopus's `comb_filter` on the A76 and 0.27 on the A55. On the A55 it takes 3.7 times its A76 cycles, libopus's 2.2 times. In round 6 it took 5.5 times.
  - `memmove` takes 19,300 cycles per frame of opus1's 48 kHz stereo CELT encoding on the A55. That is 2.0 % of the total and 3.7 times its A76 cycles. In round 6 it took 29,600 cycles, 2.7 % and 5.2 times.
- The decoder of opus1 still misses the level-1 caches more often than libopus's. Per frame of 48 kHz stereo CELT on the A76, there are 518 refills of the level-1 data cache, against 177, and 194 refills of the level-1 instruction cache, against 47. On the A55 there are 1071 against 808 and 585 against 400. On the A55, the front end of the core (instruction fetch and decode) delivers nothing in 7.0 % to 10.7 % of opus1's decode cycles, against 4.8 % to 5.7 % of libopus's.

Round 7 ([Round 7: work for in-order cores](#round-7-work-for-in-order-cores)) worked on the findings of the profile of round 6. Its parts measured the following on their own branches, each against the code before it:

- The pitch analysis of the CELT encoder: the steady loop of the new cross-correlation kernel takes 1.24 cycles per multiply-add on the A55, against 3.5 before. In the profile of 48 kHz mono CELT encoding at complexity 10 on the A55, the pitch functions fell from 27.7 % to 18.4 % of the cycles.
- The post-filter of the CELT decoder: one 20 ms stereo frame took 5.5 µs on the A55, against 10.1 µs before, and 1.39 µs on the A76, against 1.46 µs.
- The delayed-decision quantizer with 4 states, in a micro-benchmark of the quantizer alone: 135 µs against 203 µs on the A55 (-33 %), and 24 % less time on the A76.
- `memmove` fell from 2.27 % to 1.86 % of the cycles of 48 kHz stereo CELT encoding on the A55. The C library's `cos`, which took 1.5 % there, no longer shows in the profile: the encoder keeps the cosines of the spreading rotation.
- The level-1 cache refills of the decoder did not change beyond what code placement alone does: on the A55, 578 instead of 590 instruction-cache refills and 1066 instead of 1000 data-cache refills per frame of 48 kHz stereo CELT (libopus: 342 and 722). About half of the instruction-cache refills come from `bands.rs`, `pvq.rs` and `rate.rs`, and the largest source of data-cache refills is the backward MDCT.

### Building for one kind of core

The release build for `aarch64-linux-android` does not name a core. `-C target-cpu=cortex-a55` or `-C target-cpu=cortex-a76` (in `RUSTFLAGS`) tells the compiler which core to order the instructions for. It also turns on the extensions of ARMv8.2-A, so the program no longer runs on processors of ARMv8.0-A.

The first measurement at commit `bd080de` ran each build once on each core (`target/perf/android/tuning/`). Change of the opus1 / libopus ratio against the default build, geometric mean over the 28 scenarios:

| Build | A76 encode | A76 decode | A55 encode | A55 decode |
|---|---:|---:|---:|---:|
| `-C target-cpu=cortex-a55` | +0.0 % | -1.5 % | -2.6 % | -1.4 % |
| `-C target-cpu=cortex-a76` | -0.6 % | -1.1 % | +16.0 % | +8.5 % |

A check later that day ran the builds in turns with the default build on the A55 (`target/perf/android/check/ab/`). The `cortex-a55` build took 1.8 % to 4.2 % less time to encode and 1.5 % to 2.9 % less to decode. The `cortex-a76` build took 16.2 % to 18.4 % more time to encode and 6.5 % to 8.3 % more to decode.

The measurement of 2026-10-17 ran each build in turns with the default build, three passes, on both cores. It also measured the release profile, `cortex-a53`, and a `cortex-a55` build without the ARMv8.2-A extensions. Its results are in [Build settings for applications](#build-settings-for-applications). In short: ordering the instructions for the Cortex-A55 makes encoding 4 % to 5 % and decoding 1.3 % to 2.2 % faster on that core. Removing the ARMv8.2-A extensions keeps that gain, and the program then also runs on older cores. Ordering them for the Cortex-A76 makes the A55 15 % slower in encoding. All builds give the golden hashes of the default build.

## On x86-64

The bench set ran once on x86-64, on an Intel Core i7-12700 (AVX2, FMA and AVX-VNNI, no AVX-512), on 2026-10-21, against libopus 1.6.1 at `-O2`. Geometric means over the bench set: opus1 took 0.90 of libopus's time to encode and 0.81 to decode under Linux in WSL2, and 0.88 and 0.80 natively on Windows. Low-complexity SILK encoding took 1.04 to 1.10 of libopus's time. On x86-64 the output is correct but not bit-exact with libopus. See [Differences from libopus](10-differences-from-libopus.md#platforms).

The README also keeps an older measurement on AMD Ryzen 7 5700X with the earlier `opus_vs_c` benchmark (opus-rs 0.1.x, mono encode, complexity 0).

## See also

- [Testing](09-testing.md): how the output is checked against libopus.
- [Differences from libopus](10-differences-from-libopus.md): what opus1 does not do, and where its output differs.
- [Architecture](07-architecture.md): where the code of each part lives.
- [Glossary](glossary.md): the terms used in these documents.
- The documentation index: [README.md](README.md).
