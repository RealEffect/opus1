# Performance

This document explains how the speed of opus1 is measured against the C library libopus. It gives the latest results and the changes that made the code fast. It also lists the ideas that did not help, the build settings an application can use, and the commands to reproduce the numbers. All numbers were measured on Apple M4 Pro (64-bit ARM), except those of [On an Android phone](#on-an-android-phone) and the phone columns of [Code placement](#code-placement), [Round 7: work for in-order cores](#round-7-work-for-in-order-cores) and [Build settings for applications](#build-settings-for-applications). Each section names the record and the state of the code that its numbers come from. The records are in `target/perf/`, a folder of the maintainers' working tree that is not in git.

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

### The C reference

The speed of libopus depends on how it was compiled. Two builds serve as the reference on the Mac:

1. **libopus at `-O2`.** libopus's own build (`configure`) compiles with `-O2`. `scripts/android/build-libopus.sh --host` builds it that way on a Mac with Apple silicon, with Apple clang. It uses the flags that `configure` picks for 64-bit ARM: `-O2`, `FLOAT_APPROX`, and the NEON code presumed (the script lists every flag). Its source is the copy that the crate `opus-head-sys` 0.3.3 vendors: commit `228a0f85` of the libopus main branch, from 2026-08-28. No copy of the source of the release 1.6.1 was available offline. This build is the main reference from 2026-10-17 on.
2. **The Homebrew libopus 1.6.1.** The tests link it (see [Testing](09-testing.md#how-libopus-is-linked)). Homebrew builds it with `./configure` and its own compiler wrapper, and the result behaves like a build at `-Os` (optimized for size). Its machine code (the `__text` section) has 238,144 bytes. The vendored source has 232,236 bytes of machine code at `-Os` and 354,260 bytes at `-O2`.

The vendored source was also built at `-Os`, to separate the effect of the optimization level from that of the source version. Each of the three libopus builds was linked into its own build of the same harness, and the three programs ran in turns: three passes of the bench set (records in `target/perf/round7/guide/cref/`). The table compares the time of libopus between the builds, as the geometric mean over the scenarios, with the range in brackets.

| Comparison | Encode | Decode |
|---|---:|---:|
| Same source, `-O2` against `-Os` (28 scenarios) | 0.898 (0.824 to 0.958) | 0.885 (0.769 to 0.961) |
| Main branch at `-Os` against the Homebrew 1.6.1: SILK and Hybrid at complexity 9 and 10 (11 scenarios) | 0.776 (0.747 to 0.809) | 1.002 (0.992 to 1.022) |
| Main branch at `-Os` against the Homebrew 1.6.1: the other 17 scenarios | 0.996 (0.953 to 1.090) | 0.993 (0.955 to 1.044) |

So:

- `-O2` makes libopus about 10 % faster than `-Os`, in encoding and in decoding.
- The two versions of the source run at about the same speed (0.95 to 1.09 of the time in encoding, 0.96 to 1.04 in decoding), with one exception. The main branch encodes SILK and Hybrid at complexity 9 and 10 in about 0.78 of the time of 1.6.1. In these 11 scenarios the `-O2` reference is therefore faster than a build of 1.6.1 at `-O2` would be. The main branch also writes other packets than 1.6.1 in 86 of the 137 golden streams (see [Testing](09-testing.md#which-libopus-the-phone-links)).

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

Measured on 2026-10-17 with the code of git commit `0090242` (opus1 0.2.0, with the work of round 7 merged; see [Round 7: work for in-order cores](#round-7-work-for-in-order-cores)). Machine: Apple M4 Pro, macOS. opus1 was built with the default release profile. The C references are the two of [The C reference](#the-c-reference): libopus at `-O2` (main branch `228a0f85`) and the Homebrew libopus 1.6.1. Four harness programs ran in turns, three passes of the bench set: the code before round 7 (`9f462a0`) and the code of `0090242`, each linked once with each libopus. The load average of the machine fell from 37 to 6 during the first pass, because a build had just ended, and stayed between 2 and 4 after it. Source: `target/perf/round7/merge/mac/`.

Times are medians per frame in microseconds (µs). The opus1 time is the median over the six runs of `0090242`, the libopus time the median over the three runs of the `-O2` build. A ratio is opus1 / libopus, the median over the three passes. A ratio below 1 means opus1 is faster. Between the three passes, the `-O2` ratio of a scenario changed by at most 1.2 % for encoding and 2.2 % for decoding, and the geometric means by at most 0.002.

| Scenario | opus1 encode µs | libopus -O2 encode µs | Encode ratio, -O2 | Encode ratio, Homebrew | opus1 decode µs | libopus -O2 decode µs | Decode ratio, -O2 | Decode ratio, Homebrew | Only C runs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `silk_nb_8k_mono_12k_vbr_c0_20ms` | 8.8 | 12.5 | 0.705 | 0.622 | 2.0 | 3.1 | 0.641 | 0.620 | - |
| `silk_nb_8k_mono_12k_vbr_c5_20ms` | 19.9 | 30.0 | 0.664 | 0.631 | 2.0 | 3.2 | 0.647 | 0.621 | - |
| `silk_nb_8k_mono_12k_vbr_c10_20ms` | 25.5 | 42.1 | 0.605 | 0.445 | 2.1 | 3.2 | 0.645 | 0.627 | - |
| `silk_wb_16k_mono_20k_vbr_c0_20ms` | 14.4 | 21.6 | 0.671 | 0.583 | 3.5 | 5.9 | 0.598 | 0.577 | - |
| `silk_wb_16k_mono_20k_vbr_c5_20ms` | 34.5 | 54.8 | 0.628 | 0.595 | 3.8 | 6.2 | 0.607 | 0.586 | - |
| `silk_wb_16k_mono_20k_vbr_c10_20ms` | 44.5 | 88.1 | 0.504 | 0.376 | 3.8 | 6.2 | 0.610 | 0.585 | analysis |
| `silk_wb_16k_mono_20k_vbr_c9_10ms` | 24.6 | 47.0 | 0.522 | 0.392 | 2.0 | 3.2 | 0.618 | 0.592 | analysis |
| `silk_wb_16k_stereo_32k_vbr_c9_20ms` | 88.6 | 161.0 | 0.550 | 0.394 | 6.9 | 11.4 | 0.605 | 0.571 | analysis |
| `silk_wb_16k_mono_24k_vbr_c9_20ms_fec10` | 58.4 | 113.4 | 0.514 | 0.371 | 4.3 | 6.7 | 0.642 | 0.611 | analysis |
| `hybrid_swb_24k_mono_24k_cbr_c9_20ms` | 61.8 | 120.3 | 0.514 | 0.372 | 7.1 | 11.7 | 0.609 | 0.536 | analysis |
| `hybrid_swb_24k_stereo_36k_cbr_c9_20ms` | 120.9 | 217.8 | 0.554 | 0.391 | 12.0 | 21.3 | 0.564 | 0.519 | analysis |
| `hybrid_fb_48k_mono_32k_cbr_c9_20ms` | 65.4 | 116.1 | 0.563 | 0.414 | 8.3 | 13.9 | 0.599 | 0.517 | analysis |
| `hybrid_fb_48k_stereo_40k_cbr_c9_20ms` | 128.4 | 223.6 | 0.574 | 0.405 | 13.5 | 24.0 | 0.563 | 0.503 | analysis |
| `celt_fb_48k_mono_64k_vbr_c0_20ms` | 11.8 | 18.6 | 0.634 | 0.587 | 8.4 | 10.4 | 0.812 | 0.758 | - |
| `celt_fb_48k_mono_64k_vbr_c5_20ms` | 20.9 | 39.4 | 0.531 | 0.459 | 9.8 | 12.3 | 0.792 | 0.661 | - |
| `celt_fb_48k_mono_64k_vbr_c10_20ms` | 21.0 | 47.5 | 0.443 | 0.375 | 9.9 | 12.5 | 0.796 | 0.654 | analysis |
| `celt_fb_48k_stereo_128k_vbr_c0_20ms` | 22.5 | 36.9 | 0.612 | 0.559 | 16.3 | 19.1 | 0.856 | 0.788 | - |
| `celt_fb_48k_stereo_128k_vbr_c5_20ms` | 35.4 | 63.8 | 0.555 | 0.465 | 19.3 | 23.5 | 0.822 | 0.650 | - |
| `celt_fb_48k_stereo_128k_vbr_c10_20ms` | 55.2 | 97.7 | 0.564 | 0.455 | 19.4 | 23.6 | 0.823 | 0.652 | analysis |
| `celt_fb_48k_stereo_128k_vbr_c9_2p5ms` | 11.0 | 16.0 | 0.688 | 0.595 | 3.2 | 3.8 | 0.842 | 0.765 | analysis |
| `celt_fb_48k_stereo_128k_vbr_c9_5ms` | 18.3 | 28.6 | 0.639 | 0.534 | 5.7 | 6.7 | 0.852 | 0.735 | analysis |
| `celt_fb_48k_stereo_128k_vbr_c9_10ms` | 28.0 | 47.8 | 0.587 | 0.481 | 10.2 | 12.3 | 0.834 | 0.679 | analysis |
| `celt_swb_24k_mono_48k_vbr_c9_20ms` | 17.8 | 40.8 | 0.436 | 0.384 | 7.9 | 10.0 | 0.790 | 0.693 | analysis |
| `celt_swb_24k_stereo_96k_vbr_c9_20ms` | 48.1 | 85.0 | 0.565 | 0.470 | 15.0 | 19.1 | 0.784 | 0.651 | analysis |
| `celt_fb_48k_mono_24k_vbr_c9_20ms` | 15.9 | 39.6 | 0.401 | 0.331 | 5.8 | 8.4 | 0.679 | 0.518 | analysis |
| `celt_fb_48k_stereo_256k_vbr_c9_20ms` | 67.0 | 108.4 | 0.618 | 0.545 | 28.3 | 32.8 | 0.862 | 0.808 | analysis |
| `hybrid_fb_48k_mono_32k_cbr_c9_20ms_music` | 57.4 | 99.6 | 0.576 | 0.413 | 8.3 | 13.7 | 0.605 | 0.516 | analysis |
| `hybrid_fb_48k_stereo_40k_cbr_c9_20ms_music` | 113.9 | 191.5 | 0.595 | 0.416 | 13.4 | 23.8 | 0.560 | 0.502 | analysis |

How to read the table:

- **Decoding** takes 0.560 to 0.862 of the time of libopus at `-O2`, and 0.502 to 0.808 of the time of the Homebrew build. Both decoders do the same work, so this is a like-for-like comparison.
- **Encoding at complexity 0 and 5** takes 0.531 to 0.705 of the time of libopus at `-O2` (Homebrew: 0.459 to 0.631). At these settings libopus runs none of the large encoder steps that opus1 lacks.
- **Encoding at complexity 9 and 10** takes 0.401 to 0.688 of the time of libopus at `-O2` (Homebrew: 0.331 to 0.595). Part of the gap is the speech and music analysis that only libopus runs (`analysis` in the last column). In the SILK and Hybrid rows, the main branch of libopus is itself faster than 1.6.1 (see [The C reference](#the-c-reference)).
- Geometric means over the 28 scenarios: encoding 0.567 against libopus at `-O2` and 0.458 against Homebrew; decoding 0.694 and 0.619. The code before round 7 gave 0.583 and 0.470 for encoding and 0.695 and 0.617 for decoding in the same session.
- SILK encode ratios compare the fixed-point SILK encoder of opus1 with the floating-point one of libopus.
- The harness marked some operations unsteady in single runs, mostly the decoding of SILK at 8 and 16 kHz, which takes 2 to 4 µs per frame. The table uses medians, which are the numbers to use for them.

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

The code before round 7 (`9f462a0`) and after it (`0090242`) ran in turns ([Comparing two builds](#comparing-two-builds)): on the Mac three pairs of the bench set (the runs of [Latest results](#latest-results) with the Homebrew libopus), on the phone two pairs on each kind of core (the runs of [On an Android phone](#on-an-android-phone)). The table gives the change of opus1's median time: the geometric mean over the scenarios of the group, lowest and highest over the pairs, and in brackets the range of the single scenarios over all pairs. Below 0 means faster.

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

Stereo CELT encoding at complexity 9 and 10 takes more time because the encoder now does more of libopus's work. Since 0.1.37 it runs the pitch pre-filter on stereo input (complexity 5 and above) and `theta_rdo` (complexity 8 and above). The commit that added `theta_rdo` measured +34 % to +73 % encode time on the six stereo bench scenarios at complexity 9 and 10. Its ratio to libopus stayed between 0.53 and 0.63. In [Latest results](#latest-results), these six scenarios take 0.476 to 0.621 of the time of the Homebrew libopus.

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

Each change keeps the order of the floating-point operations and the integer results, so the output stays the same.

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

The settings do not change the output. On the Mac, `golden --check` found 0 of 137 streams that differ for the release profile and for `native` with it. On the phone, the v8.0 build with the release profile reports exactly the same differences against the Mac golden file as the default build (only the fields that depend on the phone's libopus differ, see [Testing](09-testing.md#which-libopus-the-phone-links)).

An earlier measurement at commit `2cf93bf`, before most of the performance work, found the release profile about 3 % faster in decoding on the M4 Pro (`target/perf/prof/memory-build/report.md`, section 6).

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
use opus1::{Application, Bitrate, OpusEncoder};
use std::time::Instant;

let mut encoder = OpusEncoder::new(48_000, 2, Application::Audio).unwrap();
encoder.set_bitrate(Bitrate::BitsPerSecond(128_000)).unwrap();
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

Measured on 2026-10-17 with the code of git commit `0090242`, with the scripts in `scripts/android/` (see [Testing](09-testing.md#tests-on-an-android-phone)). The phone is a HONOR CLK-AN00 with a MediaTek MT6833 and Android 14. `ab.sh` pinned the harness with `taskset` to one core of each kind:

- CPU 7, a Cortex-A76 at up to 2.4 GHz. It reorders instructions, so independent ones run at the same time.
- CPU 5, a Cortex-A55 at up to 2.0 GHz. It runs instructions in program order, at most two per cycle.

opus1 was built with the default release profile. The C reference is libopus built by `scripts/android/build-libopus.sh`: the NDK's clang 18 at `-O2`, the default of libopus's own build. Its source is commit `228a0f85` of the libopus main branch, not the release 1.6.1 (see [Testing](09-testing.md#which-libopus-the-phone-links)).

The code before round 7 (`9f462a0`) and the code of `0090242` ran in turns, two pairs of the bench set on each core: one session on the A76 (base, new, new, base) and two sessions on the A55 (base, new, then new, base). Both programs were built with the same settings and the same phone directory, so the strings compiled into them have the same length. The cluster of the pinned core spent 88 % (A76) and 96 % (A55) of each session at its highest clock (cpufreq statistics). The processor was at 34 °C to 38 °C when a session started and reached at most 47.6 °C. Between the two runs of `0090242` on one core, the median time of an operation changed by at most 0.6 % on the A76 and 1.0 % on the A55, and the opus1 / libopus ratio by at most 0.8 % and 1.4 %. Every round was quiet, so no scenario was measured again. The records are in `target/perf/round7/merge/android/`: `a76/`, `a55_s1/` and `a55_s2/`, with `head_*` for `0090242` and `base_*` for `9f462a0`.

The measurement of round 6, at commit `bd080de`, was checked later on the same day with a harness built in another directory and run from another directory of the phone (records in `target/perf/android/check/`):

- In the first run, the phone kept the A76 cluster at 2.118 GHz and the A55 cluster at 1.75 GHz, while the processor was at 43 °C to 50 °C. The times grew by the ratio of the clocks: 1.13 times on the A76 and 1.14 times on the A55. The geometric means of the ratios stayed within 0.007. `bench.sh` now prints a warning when a cluster spends less than 75 % of a run at its highest clock, and `ab.sh` prints the share of each clock.
- In SILK and Hybrid scenarios on the A76, libopus took up to 3.8 % more time to encode and opus1 up to 2.2 % less time to decode, although the code was the same: a longer directory name compiled into the new program moved its code 16 bytes further. So on the A76, a difference of about 3 % in one scenario between two builds is not a real change (see [Code placement](#code-placement)).

### The C reference matters

The Homebrew libopus behaves like a build at `-Os`, and the phone's libopus is built at `-O2` (see [The C reference](#the-c-reference)). So the table gives the ratios on the Mac against both. The Mac columns are those of [Latest results](#latest-results): the same code, the Homebrew library, and the phone's libopus source built on the Mac with `-O2`.

Ratio opus1 / libopus of the median times, over the 28 scenarios of the bench set. Below 1 means opus1 is faster.

| | M4 Pro, Homebrew libopus | M4 Pro, the phone's libopus | Cortex-A76 | Cortex-A55 |
|---|---:|---:|---:|---:|
| Encode, geometric mean | 0.458 | 0.567 | 0.636 | 0.663 |
| Encode, range | 0.331 to 0.631 | 0.401 to 0.705 | 0.441 to 0.774 | 0.487 to 0.796 |
| Decode, geometric mean | 0.619 | 0.694 | 0.771 | 0.840 |
| Decode, range | 0.502 to 0.808 | 0.560 to 0.862 | 0.676 to 0.890 | 0.753 to 0.931 |
| Encode, geometric mean before round 7 | 0.470 | 0.583 | 0.689 | 0.745 |
| Decode, geometric mean before round 7 | 0.617 | 0.695 | 0.782 | 0.860 |

The rows "before round 7" are the runs of `9f462a0` in the same sessions. Round 6 had measured 0.692 and 0.755 for encoding and 0.790 and 0.864 for decoding on the phone, at commit `bd080de`.

### Results per scenario

Times are opus1's median per frame in microseconds (µs), the lower of the two runs of `0090242`. The ratio is opus1 / libopus, the mean of the two runs.

| Scenario | A76 encode µs | A76 encode ratio | A55 encode µs | A55 encode ratio | A76 decode µs | A76 decode ratio | A55 decode µs | A55 decode ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `silk_nb_8k_mono_12k_vbr_c0_20ms` | 31.6 | 0.774 | 97.3 | 0.710 | 5.7 | 0.813 | 15.8 | 0.928 |
| `silk_nb_8k_mono_12k_vbr_c5_20ms` | 62.8 | 0.760 | 179.2 | 0.686 | 5.8 | 0.808 | 16.0 | 0.921 |
| `silk_nb_8k_mono_12k_vbr_c10_20ms` | 83.1 | 0.705 | 234.8 | 0.751 | 5.8 | 0.811 | 16.0 | 0.923 |
| `silk_wb_16k_mono_20k_vbr_c0_20ms` | 49.5 | 0.727 | 143.6 | 0.653 | 10.1 | 0.719 | 27.5 | 0.807 |
| `silk_wb_16k_mono_20k_vbr_c5_20ms` | 106.9 | 0.726 | 291.3 | 0.641 | 10.7 | 0.719 | 29.4 | 0.805 |
| `silk_wb_16k_mono_20k_vbr_c10_20ms` | 143.1 | 0.580 | 388.8 | 0.600 | 10.7 | 0.721 | 29.2 | 0.805 |
| `silk_wb_16k_mono_20k_vbr_c9_10ms` | 81.3 | 0.608 | 220.4 | 0.623 | 5.8 | 0.724 | 17.0 | 0.822 |
| `silk_wb_16k_stereo_32k_vbr_c9_20ms` | 287.3 | 0.637 | 777.7 | 0.659 | 20.6 | 0.734 | 56.9 | 0.838 |
| `silk_wb_16k_mono_24k_vbr_c9_20ms_fec10` | 177.5 | 0.572 | 473.7 | 0.606 | 11.9 | 0.748 | 32.0 | 0.834 |
| `hybrid_swb_24k_mono_24k_cbr_c9_20ms` | 191.3 | 0.569 | 532.8 | 0.615 | 22.6 | 0.703 | 73.7 | 0.783 |
| `hybrid_swb_24k_stereo_36k_cbr_c9_20ms` | 382.3 | 0.630 | 1042.6 | 0.668 | 41.0 | 0.680 | 133.3 | 0.765 |
| `hybrid_fb_48k_mono_32k_cbr_c9_20ms` | 196.6 | 0.600 | 545.5 | 0.628 | 25.9 | 0.692 | 84.1 | 0.767 |
| `hybrid_fb_48k_stereo_40k_cbr_c9_20ms` | 397.4 | 0.636 | 1079.9 | 0.666 | 46.0 | 0.676 | 151.6 | 0.758 |
| `celt_fb_48k_mono_64k_vbr_c0_20ms` | 35.7 | 0.724 | 109.6 | 0.796 | 22.2 | 0.867 | 62.0 | 0.905 |
| `celt_fb_48k_mono_64k_vbr_c5_20ms` | 60.1 | 0.643 | 203.7 | 0.710 | 25.8 | 0.803 | 74.2 | 0.857 |
| `celt_fb_48k_mono_64k_vbr_c10_20ms` | 60.5 | 0.487 | 204.8 | 0.529 | 26.2 | 0.803 | 75.2 | 0.858 |
| `celt_fb_48k_stereo_128k_vbr_c0_20ms` | 69.0 | 0.703 | 218.0 | 0.791 | 42.7 | 0.890 | 116.5 | 0.931 |
| `celt_fb_48k_stereo_128k_vbr_c5_20ms` | 104.8 | 0.673 | 339.6 | 0.727 | 50.9 | 0.830 | 141.7 | 0.865 |
| `celt_fb_48k_stereo_128k_vbr_c10_20ms` | 155.5 | 0.629 | 489.3 | 0.661 | 51.1 | 0.831 | 142.1 | 0.868 |
| `celt_fb_48k_stereo_128k_vbr_c9_2p5ms` | 32.6 | 0.705 | 122.6 | 0.776 | 9.4 | 0.826 | 32.1 | 0.867 |
| `celt_fb_48k_stereo_128k_vbr_c9_5ms` | 54.1 | 0.689 | 190.0 | 0.756 | 15.9 | 0.837 | 51.0 | 0.889 |
| `celt_fb_48k_stereo_128k_vbr_c9_10ms` | 81.1 | 0.641 | 268.8 | 0.689 | 27.7 | 0.844 | 81.9 | 0.895 |
| `celt_swb_24k_mono_48k_vbr_c9_20ms` | 52.9 | 0.489 | 182.8 | 0.530 | 21.6 | 0.807 | 63.8 | 0.837 |
| `celt_swb_24k_stereo_96k_vbr_c9_20ms` | 140.2 | 0.638 | 449.6 | 0.668 | 40.3 | 0.820 | 116.2 | 0.838 |
| `celt_fb_48k_mono_24k_vbr_c9_20ms` | 47.3 | 0.441 | 169.7 | 0.487 | 17.4 | 0.732 | 56.8 | 0.803 |
| `celt_fb_48k_stereo_256k_vbr_c9_20ms` | 193.8 | 0.685 | 592.9 | 0.733 | 68.5 | 0.867 | 172.7 | 0.890 |
| `hybrid_fb_48k_mono_32k_cbr_c9_20ms_music` | 178.8 | 0.628 | 500.1 | 0.650 | 26.1 | 0.695 | 84.9 | 0.771 |
| `hybrid_fb_48k_stereo_40k_cbr_c9_20ms_music` | 361.4 | 0.665 | 989.5 | 0.687 | 46.1 | 0.676 | 152.2 | 0.753 |

How to read the table:

- opus1 is faster than libopus in every scenario, for encoding and decoding, on both cores.
- The slowest operation on the A55, the encoding of 48 kHz stereo Hybrid, takes 1.08 ms per 20 ms frame. That is 5.4 % of one core.
- From the M4 Pro to the A76, the time of opus1 grows 3.07 times for encoding and 2.87 times for decoding (geometric means). The time of the same libopus grows 2.74 and 2.58 times.
- From the A76 to the A55, the time of opus1 grows 3.02 times for encoding and 2.94 times for decoding. The time of libopus grows 2.90 and 2.69 times. Before round 7, in the same sessions, opus1 grew 3.14 and 2.96 times.
- The A55 still costs opus1 most in CELT encoding at complexity 5 and above: 3.1 to 3.8 times its A76 time, against 2.9 to 3.4 times for libopus. Before round 7, in the same sessions, opus1 took 3.1 to 4.0 times its A76 time.
- SILK encoding at complexity 0 and 5 is the other way round: opus1 grows 2.7 to 3.1 times, libopus 3.1 to 3.4 times.

### Where the time goes on the phone

`scripts/android/profile.sh` counted hardware events per frame with `simpleperf stat` and sampled the hot functions with `simpleperf record`, for opus1 and libopus on both cores. These counters were measured in round 6, at commit `bd080de`, before the work of round 7. The records are in `target/perf/android/profile/`. The program for this keeps line tables and symbols. Its times per frame are within 3 % of the bench for CELT and for decoding, and 2 % to 11 % above it for SILK and Hybrid encoding, for both libraries. In the table, IPC is the number of instructions per cycle.

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

Round 7 ([Round 7: work for in-order cores](#round-7-work-for-in-order-cores)) worked on these findings. Its parts measured the following on their own branches, each against the code before it:

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

## Earlier measurement on x86-64

The README keeps an older measurement on AMD Ryzen 7 5700X with the earlier `opus_vs_c` benchmark (opus-rs 0.1.x, mono encode, complexity 0). In the release described here, the x86-64 code paths were only compiled, not run. See [Differences from libopus](10-differences-from-libopus.md#platforms).

## See also

- [Testing](09-testing.md): how the output is checked against libopus.
- [Differences from libopus](10-differences-from-libopus.md): what opus1 does not do, and where its output differs.
- [Architecture](07-architecture.md): where the code of each part lives.
- [Glossary](glossary.md): the terms used in these documents.
- The documentation index: [README.md](README.md).
