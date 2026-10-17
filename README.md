# opus1

A pure-Rust implementation of the [Opus audio codec](https://opus-codec.org/) (RFC 6716), ported from the reference C implementation (libopus 1.6). The name says what it follows: the Opus 1.x bitstream and the behaviour of libopus 1.x.

This crate was published as `opus-rs` up to version 0.1.37. From 0.2.0 it is published as `opus1`:

```toml
[dependencies]
opus1 = "0.2"
```

A systematic introduction to Opus and to this library is in [docs/](docs/README.md).

> **Production-ready**

## Features

- **Pure Rust** — no C dependencies
- **High performance:** on Apple M4 Pro, decoding takes 0.56-0.86x and encoding (complexity 0-5, where both do the same work) 0.53-0.71x the time of C libopus 1.6.1 built at `-O2`, the optimization level of libopus's own build. Against the Homebrew build of libopus 1.6.1, which behaves like a build at `-Os` and takes about 24 % more time to encode and 12 % more to decode, the ranges are 0.50-0.81x and 0.46-0.63x. opus1 is also faster than libopus on the big and the small cores of an Android phone (see [Performance](#performance))
- **Multichannel:** multistream encoder and decoder (mapping families 0, 1, 2 and 255), interoperable with libopus
- **libopus controls:** the encoder settings, their defaults and the values they refuse follow libopus's `OPUS_SET_*` requests (automatic bitrate, VBR and its constraint, complexity, in-band FEC, packet loss, forced mode, bandwidth and channels)
- **`#![no_std]` + no `alloc`:** fully heap-free — runs on bare metal, RTOS, and WebAssembly with **no global allocator**

## Quick Start

```rust
use opus1::{Application, Bitrate, OpusDecoder, OpusEncoder};

// Encode. Without `set_bitrate` the encoder uses libopus's automatic
// bitrate (`Bitrate::Auto`, 19 kb/s for 16 kHz mono in 20 ms frames).
// Each setter refuses what libopus refuses, for example a complexity of 11.
let mut encoder = OpusEncoder::new(16000, 1, Application::Voip).unwrap();
encoder.set_bitrate(Bitrate::BitsPerSecond(16000)).unwrap();
encoder.set_vbr(false); // constant bitrate: every packet has 40 bytes
assert!(encoder.set_complexity(11).is_err());

let input = vec![0.0f32; 320]; // 20ms frame at 16kHz
let mut output = vec![0u8; 256];
let bytes = encoder.encode(&input, 320, &mut output).unwrap();
assert_eq!(bytes, 40);

// Encode PCM16 directly. The output is byte-for-byte identical to
// `encode()` called with the same audio converted via
// `sample as f32 / 32768.0`.
let input_i16 = vec![0i16; 320]; // 20ms frame at 16kHz
let bytes = encoder.encode_i16(&input_i16, 320, &mut output).unwrap();

// Decode. `decode` writes the samples it returns and nothing after them.
let mut decoder = OpusDecoder::new(16000, 1).unwrap();
let mut pcm = vec![0.0f32; 320];
let samples = decoder.decode(&output[..bytes], 320, &mut pcm).unwrap();
assert_eq!(samples, 320);
```

The public API is the crate root (`OpusEncoder`, `OpusDecoder`, `Application`,
`Bandwidth`, `Bitrate`, `ForcedMode`, `pcm_soft_clip`) and the modules `packet`
and `multistream`. [docs/06](docs/06-using-the-library.md) shows the settings and
recipes for calls, voice agents and music.

Decoding to 16-bit PCM, and recovering a lost packet from the in-band
forward error correction (FEC) data of the next one:

```rust
use opus1::{Application, Bitrate, OpusDecoder, OpusEncoder};

// Three 20 ms packets of a 180 Hz tone at 16 kHz. Each packet after the
// first also carries a low-bitrate copy of the frame before it (in-band FEC).
let mut encoder = OpusEncoder::new(16000, 1, Application::Voip).unwrap();
encoder.set_bitrate(Bitrate::BitsPerSecond(24000)).unwrap();
encoder.set_inband_fec(true);
encoder.set_packet_loss_perc(10).unwrap();
let tone: Vec<f32> = (0..3 * 320)
    .map(|i| 0.3 * (2.0 * std::f32::consts::PI * 180.0 * i as f32 / 16000.0).sin())
    .collect();
let mut packets = Vec::new();
for frame in tone.chunks_exact(320) {
    let mut packet = vec![0u8; 256];
    let len = encoder.encode(frame, 320, &mut packet).unwrap();
    packet.truncate(len);
    packets.push(packet);
}

let mut decoder = OpusDecoder::new(16000, 1).unwrap();
let mut pcm16 = vec![0i16; 320];
decoder.decode_i16(&packets[0], 320, &mut pcm16).unwrap(); // soft-clipped like opus_decode
// packets[1] is lost. packets[2] has arrived and carries a copy of it.
let mut lost = vec![0.0f32; 320];
decoder.decode_fec(&packets[2], 320, &mut lost).unwrap(); // the frame before packets[2]
decoder.decode_i16(&packets[2], 320, &mut pcm16).unwrap();
```

Float output is not clipped to [-1, 1], as in libopus's `opus_decode_float`.

### Multichannel

```rust
use opus1::{Application, Bitrate};
use opus1::multistream::{MultistreamEncoder, MultistreamDecoder};

// 5.1 at 48 kHz in Vorbis channel order (mapping family 1).
let mut encoder = MultistreamEncoder::new_surround(48_000, 6, 1, Application::Audio).unwrap();
encoder.set_bitrate(Bitrate::BitsPerSecond(256_000)).unwrap();
let layout = encoder.layout().clone(); // streams, coupled streams and mapping for the Ogg header

let pcm = vec![0.0f32; 960 * 6]; // 20 ms, interleaved
let mut packet = vec![0u8; 4000];
let len = encoder.encode(&pcm, 960, &mut packet).unwrap();

let mut decoder = MultistreamDecoder::with_layout(
    48_000, 6, layout.streams(), layout.coupled_streams(), layout.mapping()).unwrap();
let mut out = vec![0.0f32; 960 * 6];
decoder.decode(&packet[..len], 960, &mut out).unwrap();
```

`cargo run --release --example multistream` encodes and decodes a 5.1 test
signal and prints the signal-to-noise ratio of each channel.

### WAV Roundtrip

```bash
# Rust encoder/decoder
cargo run --example wav_test
```

## `#![no_std]` + no-`alloc` Support

By default opus1 runs on `std` with the **`heap` feature**: the large codec state
(`OpusEncoder`/`OpusDecoder` working buffers) is allocated with `Box`, so the public
structs are small (about 2.5 KB for `OpusEncoder`, 0.1 KB for `OpusDecoder`) and
constructing them uses almost no stack.

For `#![no_std]`, disable the default features. The crate then builds as
`#![no_std]` with **no `alloc`** — fully heap-free, no global allocator required —
and every working buffer is an inline, fixed-size array (sized to the Opus worst
case); growable state uses an internal `FixedVec<T, N>` (a tiny `Vec`-like type over
`[MaybeUninit<T>; N]`). In that configuration the structs are large, so place them
in a `static` (wrap init in your own `Once`-like guard) or a dedicated buffer.

```toml
[dependencies.opus1]
version = "0.2"
default-features = false
features = ["libm"]
```

The public API (`OpusEncoder`/`OpusDecoder` with `encode`/`decode` writing into caller-provided buffers) is identical in both configurations.

### Feature flags

| Feature  | Default | Description |
|----------|---------|-------------|
| `std`    | yes     | Enables OS-backed runtime x86 SIMD detection (CPUID). Without it the crate is `#![no_std]`. |
| `heap`   | yes     | Allocates the large codec state on the heap (`Box`), shrinking `OpusEncoder`/`OpusDecoder` to a few KB so they can live on ordinary stacks. Requires `std`. |
| `libm`   | no      | Required for `#![no_std]` builds — provides float math via a pure-Rust port of musl `libm`. Not needed when `std` is on. |

> Runtime dependency footprint: `std` build → **0 deps**; `no_std` build → **1 dep** (`libm`, pure Rust).

### Notes for `no_std` users

- **Large structs:** without `heap`, every working buffer is inline, so
  `OpusEncoder`/`OpusDecoder` are large (~135 KB / ~67 KB). Place them in a
  `static` (wrap init in your own `Once`-like guard) or a dedicated buffer rather
  than on the stack. The test suite of this build needs big test-thread stacks:
  `scripts/check_no_std.sh` sets `RUST_MIN_STACK=33554432` (32 MiB).
- **x86 SIMD:** without `std`, AVX/AVX2 dispatch falls back to compile-time detection. Build with `RUSTFLAGS="-C target-feature=+avx2"` to enable it. aarch64 NEON is unconditional.
- **Frames are capped** at the RFC 6716 maximum of 1275 bytes (the heap-free range-coder buffer is sized accordingly). A packet of 40 to 120 ms holds up to six such frames.
- **Verified targets:** `thumbv7em-none-eabi` (Cortex-M), `wasm32-unknown-unknown`, and any Linux no_std target. Check with `scripts/check_no_std.sh`.

## Performance

Measured with `examples/perf_harness.rs` (`cargo run --release --example perf_harness -- bench`):
28 scenarios, 6 s clips of speech and of a synthetic full-band music signal,
15 repetitions per operation with opus1 and libopus interleaved. Both decoders
decode the same libopus-encoded packets. The libopus encoder is forced to the
same mode, bandwidth and channel count as opus1. The table gives the median
time per 20 ms frame (10 ms or 2.5 ms where named) and the ratio opus1 / libopus
(below 1 means opus1 is faster).

### vs C libopus on Apple M4 Pro (aarch64), default release profile

Measured on 2026-10-17 with the code of commit `b7a8b64` (the work of round 8
merged on `c96e936`). Both C references are the release libopus 1.6.1. The main one is
built at `-O2`, the optimization level of libopus's own build:
`scripts/android/build-libopus.sh --host --rtcd --src <1.6.1 source>` builds
it with Apple clang and the flags of libopus's `configure`, plus the run-time
CPU detection that `configure` uses on Android. With that detection libopus's
SILK encoder uses its NEON quantizer, as on a phone. The second ratio column
is the Homebrew libopus 1.6.1, which the tests link. It behaves like a build at
`-Os` and calls the C version of that quantizer. The `-O2` build takes 0.81 of
its encode time and 0.89 of its decode time (geometric means). The code before
and after round 8 ran in turns, each linked with each libopus build: four
harness programs, three passes. Between the passes the ratio of a scenario
changed by at most 2.4 %.

| Scenario | opus1 encode µs | Ratio vs `-O2` | Ratio vs Homebrew | opus1 decode µs | Ratio vs `-O2` | Ratio vs Homebrew |
|---|---:|---:|---:|---:|---:|---:|
| SILK NB 8 kHz mono 12 kb/s, complexity 0 | 8.9 | 0.71 | 0.62 | 2.0 | 0.64 | 0.62 |
| SILK WB 16 kHz mono 20 kb/s, complexity 10 | 42.9 | 0.49* | 0.36* | 3.8 | 0.61 | 0.59 |
| SILK WB 16 kHz stereo 32 kb/s, complexity 9 | 85.3 | 0.53* | 0.38* | 6.9 | 0.60 | 0.57 |
| Hybrid SWB 24 kHz mono 24 kb/s, complexity 9 | 59.3 | 0.49* | 0.36* | 7.2 | 0.61 | 0.54 |
| Hybrid FB 48 kHz stereo 40 kb/s, complexity 9 | 123.5 | 0.55* | 0.39* | 13.5 | 0.56 | 0.50 |
| CELT FB 48 kHz mono 64 kb/s, complexity 0 | 11.3 | 0.60 | 0.57 | 8.0 | 0.77 | 0.72 |
| CELT FB 48 kHz mono 64 kb/s, complexity 10 | 20.3 | 0.42* | 0.36* | 9.5 | 0.76 | 0.63 |
| CELT FB 48 kHz stereo 128 kb/s, complexity 5 | 34.1 | 0.53 | 0.45 | 18.6 | 0.79 | 0.63 |
| CELT FB 48 kHz stereo 128 kb/s, complexity 10 | 52.6 | 0.53* | 0.43* | 18.7 | 0.79 | 0.63 |
| CELT FB 48 kHz stereo 128 kb/s, 2.5 ms, complexity 9 | 10.6 | 0.66* | 0.57* | 3.1 | 0.82 | 0.74 |

\* libopus runs its speech/music analysis (`analysis.c`) at complexity 7 and
above when the input rate is 16 kHz or more; opus1 has no port of it, so these
encode ratios favour opus1.
The C SILK encoder in a float build of libopus is the floating-point one;
opus1 ports the fixed-point one.

Over all 28 scenarios of the bench set (geometric means), opus1 takes 0.544 of
the encode time and 0.680 of the decode time of libopus 1.6.1 at `-O2`, and
0.442 and 0.607 of the Homebrew build (before round 8: 0.565 and 0.695, 0.458
and 0.619). The full table is in
[docs/08-performance.md](docs/08-performance.md#latest-results), and
[The C reference](docs/08-performance.md#the-c-reference) explains the builds.

On an Android phone (MediaTek MT6833), against libopus 1.6.1 built with the
NDK's clang at `-O2` and the flags of libopus's `configure` for Android, opus1
is faster in all 28 scenarios on both kinds of core (geometric means, commit
`b7a8b64`): encoding 0.625 and decoding 0.759 of libopus's time on the
Cortex-A76, 0.651 and 0.821 on the in-order Cortex-A55. See
[docs/08-performance.md](docs/08-performance.md#on-an-android-phone).

The latest speed work ("round 8" in the performance document) made the
release libopus 1.6.1, built from Xiph's release file, the C reference on the
Mac and on the phone. It also went on with the work for in-order cores, in
these parts:

- the SILK delayed-decision quantizer;
- the PVQ search and the bit allocation of CELT;
- the pitch analysis of the CELT encoder;
- the FFT, the inverse MDCT and the de-emphasis of the CELT decoder.

The output did not change. Against the code before it, run in turns
(geometric means over the bench set):

| | Apple M4 Pro | Cortex-A76 | Cortex-A55 |
|---|---:|---:|---:|
| Encoding | 3.1-3.5 % less time | 1.5 % less | 1.5-1.8 % less |
| Decoding | 1.9-2.2 % less | 1.4 % less | 1.7-1.8 % less |

CELT encoding and decoding take 2.3-4.2 % less time on all three cores. Most
of that comes from the PVQ search and the allocation. On Apple M4 Pro, SILK and
Hybrid encoding at complexity 6 and 7 take 3.8-6.4 % less time. That removes
the slowdown that round 7 left there. See
[docs/08-performance.md](docs/08-performance.md#round-8-more-work-for-in-order-cores).

Round 7 before it was aimed at the in-order Cortex-A55. It rewrote the pitch
analysis of the CELT encoder, the
post-filter of the CELT decoder, the SILK noise-shaping quantizers, and the PVQ
search, the spreading rotation and the bit allocation of the CELT encoder. The
output did not change. Against the code before it, run in turns on the same
machine (geometric means over the bench set), encoding takes 10.8-11.1 % less
time on the Cortex-A55, 7.5 % less on the Cortex-A76 and 2.5-2.8 % less on
Apple M4 Pro. Decoding takes 1.9-2.0 % less on the Cortex-A55, 1.0 % less on
the Cortex-A76 and about the same time (within 0.4 %) on Apple M4 Pro. From
the Cortex-A76 to the Cortex-A55, opus1's encoding now grows 3.02 times slower
and libopus's 2.90 times; before this work opus1's grew 3.14 times. See
[docs/08-performance.md](docs/08-performance.md#round-7-work-for-in-order-cores).

Against opus-rs 0.1.37 on the same machine (change of the opus1 / libopus
ratio, Homebrew libopus), encoding takes 33-56 % less time for SILK and Hybrid
and 29-49 % less for mono CELT; decoding takes 35-62 % less in every scenario.
Stereo CELT encoding takes 24-30 % less time at complexity 0 and 5. From
complexity 5 it now runs libopus's stereo pitch pre-filter and, from complexity
8, its second quantization pass per stereo band (`theta_rdo`). So at complexity
9 and 10 it takes 2-22 % more time than in 0.1.37, while staying at 0.46-0.59
of the Homebrew libopus's time.

Build settings that help (measured, see
[docs/06-using-the-library.md](docs/06-using-the-library.md#build-settings-for-speed)):
`lto = "fat"`, `codegen-units = 1` and `panic = "abort"` in the application's
release profile take 1.7 % off decoding on Apple M4 Pro and 1.1-1.5 % off
encoding on the Cortex-A76. For Android, `-C target-cpu=cortex-a55` orders the
instructions for the in-order Cortex-A55. With the ARMv8.2-A extensions turned
off again (so the program runs on every 64-bit ARM phone) and with that
profile, it takes 4.4-4.5 % off encoding and 1.3-1.5 % off decoding on the
Cortex-A55, and 0.5-1.4 % off both on the Cortex-A76. Building for the
Cortex-A76 instead makes the Cortex-A55 15 % slower. `-C target-cpu=native`
changes nothing measurable on Apple silicon.

On aarch64 the FFT uses the FEAT_FCMA complex multiply-add instructions (part of
the Apple aarch64 baseline). Other aarch64 targets detect them at run time with
`std`. On aarch64 the FFT and MDCT give the same bits as the arm64 build of libopus.

### Earlier measurement: x86-64 (AVX2/FMA), opus-rs 0.1.x

Measured on AMD Ryzen 7 5700X with the earlier `opus_vs_c` benchmark (mono encode only,
complexity 0). The x86-64 code paths were only compile-checked in the work described
under Unreleased below; they were not run.

| Config | Pure Rust | C Opus | Ratio |
|--------|-----------|--------|-------|
| 8 kHz / 20 ms VoIP | **39.9 ms** | 40.6 ms | 0.98× (**Rust 2% faster**) |
| 16 kHz / 20 ms VoIP | **66.8 ms** | 67.1 ms | 1.00× (**Rust 0.5% faster**) |
| 16 kHz / 10 ms VoIP | 73.2 ms | **72.5 ms** | 1.01× (within noise) |
| 48 kHz / 20 ms Audio | **25.1 ms** | 28.4 ms | 0.88× (**Rust 12% faster**) |
| 48 kHz / 10 ms Audio | **29.7 ms** | 31.2 ms | 0.95× (**Rust 5% faster**) |

## Release Notes

### 0.2.0 (unreleased)

The crate is renamed from `opus-rs` to `opus1`. Code that used `opus_rs::` now uses `opus1::`.

This release was checked against libopus 1.6.1 throughout: every fix below has a
test that compares opus1 with the C library under the same settings.

**Breaking changes to the public interface**
- `OpusEncoder::enable_hybrid_mode` is removed. It only set the audio bandwidth,
  so at higher bitrates the encoder still chose CELT. To ask for Hybrid packets,
  call `set_force_mode(Some(ForcedMode::Hybrid))`. Also call
  `set_bandwidth(Bandwidth::Superwideband)` or `set_bandwidth(Bandwidth::Fullband)`,
  so that low bitrates do not fall back to wideband SILK.
- The public field `OpusDecoder::hybrid_skip_celt` is removed. The decoder never
  read it.
- The settings of `OpusEncoder` are methods instead of public fields:
  `set_bitrate` with `bitrate_setting` and `bitrate`, `set_complexity` and
  `complexity`, `set_vbr` and `vbr` (instead of `use_cbr`, with the opposite
  meaning), `set_inband_fec` and `inband_fec`, `set_packet_loss_perc` and
  `packet_loss_perc`. Each setter checks its value as libopus's `OPUS_SET_*`
  request does. A bitrate of 0 or less, a complexity outside 0-10 and a packet
  loss outside 0-100 % are errors, and the setting keeps its old value. A
  bitrate is raised to at least 500 b/s and limited to at most 750,000 b/s per
  channel. The encoder no longer
  clamps the settings on every `encode` call.
- The default bitrate of `OpusEncoder` is libopus's automatic one,
  `Bitrate::Auto`: 60 × rate / frame size + rate × channels b/s, for example
  19 kb/s for 16 kHz mono and 99 kb/s for 48 kHz stereo in 20 ms frames. It was
  a fixed 64 kb/s. An encoder that is given a bitrate writes the same packets as
  before.
- `Bitrate` moved from `opus1::multistream` to the crate root.
  `OpusEncoder::set_bitrate` takes it too, so a single stream now also has
  `Bitrate::Auto` (libopus `OPUS_AUTO`) and `Bitrate::Max` (`OPUS_BITRATE_MAX`).
- New: `set_vbr_constraint` on `OpusEncoder` and `MultistreamEncoder` (libopus
  `OPUS_SET_VBR_CONSTRAINT`). Turning the constraint off gives CELT-only packets
  an unconstrained variable bitrate.
- The internal modules are private. The public API is the crate root
  (`OpusEncoder`, `OpusDecoder`, `Application`, `Bandwidth`, `Bitrate`,
  `ForcedMode`, `pcm_soft_clip`) and the modules `packet` and `multistream`. The
  modules `bands`, `celt`, `celt_lpc`, `hp_cutoff`, `kiss_fft`, `mdct`, `modes`,
  `pitch`, `pvq`, `quant_bands`, `range_coder`, `rate` and `silk`, the types
  `CeltEncoder`, `CeltDecoder`, `SilkResampler`, `SilkResamplerDown1_3` and
  `SilkResamplerDown1_6` at the crate root, and the macro `tell_frac_inline!`
  are no longer exported. The submodules of
  `multistream` are private too; their public items, `validate_ambisonics`
  included, are in `opus1::multistream`. The tests of this repository reach
  internal items through `opus1::internals`, which is hidden from the
  documentation and not part of the stable API.
- `OpusDecoder::decode` no longer sets the part of `output` after the decoded
  samples to zero. As in libopus, it writes only the samples it returns. A program
  that relied on the zeros must clear the buffer itself.

**24 kHz audio**
- **Fix: Hybrid super-wideband packets decoded with 21 CELT bands instead of 19.**
  Every 24 kHz speech stream at the usual VoIP bitrates is Hybrid super-wideband,
  so its 8-12 kHz band decoded to unrelated content. Now within 124 dB of libopus.
- **Fix: the encoder chooses mode and bandwidth from the bitrate, as libopus does.**
  24 kHz input was always coded super-wideband; Hybrid VBR coded about 25 kb/s
  when 14 kb/s was asked for; in-band forward error correction (FEC) that lowered
  the bandwidth divided by zero.
- Other decoder fixes found on the way, none specific to 24 kHz: 10 ms SILK and
  Hybrid frames decoded to noise; mode switches and redundancy frames; SILK output
  at 8 and 12 kHz used the wrong resampler; SILK packet loss concealment (four
  differences, plus comfort noise); two CELT band rules for 2.5 and 5 ms frames.

**New features**
- `multistream::MultistreamEncoder` (surround families 0, 1, 2 and 255, rate
  allocation, LFE stream, surround masking) and a general `MultistreamDecoder`
  (any valid layout, 16-bit output, loss concealment, FEC). Packets interoperate
  with libopus's multistream encoder and decoder in both directions.
- `OpusDecoder`: `decode_i16` (soft clipping as `opus_decode`), `decode_fec`,
  `decode_fec_i16`, `final_range`, `reset_state`; packets whose channel count
  differs from the decoder's; CELT packet loss concealment ported from libopus.
- `OpusEncoder`: `set_force_mode`, `set_bandwidth`, `set_max_bandwidth`,
  `set_force_channels`, `set_lfe`, `set_energy_mask`, `final_range`,
  `reset_state`, `lookahead`.
- Frames of 80, 100 and 120 ms in `OpusEncoder` and `MultistreamEncoder`,
  coded as libopus 1.6.1 codes them: CELT and Hybrid as four to six 20 ms
  frames, SILK as two 40 ms, five 20 ms or two 60 ms frames, in one packet.
  With CBR every packet has libopus's size, and CELT-only packets at
  complexity 0 to 5 are libopus's byte for byte. With VBR a part is not cut
  to its share of the bitrate, as with 40 and 60 ms (with in-band FEC that
  limit cost libopus about 20 dB in the streams measured).
- `set_expert_frame_duration` and `expert_frame_duration` on `OpusEncoder`
  and `MultistreamEncoder`, with the new type `FrameDuration` (libopus
  `OPUS_SET_EXPERT_FRAME_DURATION`): each call codes one frame of the duration
  that is set. The multistream encoder no longer differs from libopus in its
  frame sizes (difference D3).
- `packet` module: parse, write and pad packets, self-delimited framing.

**Behaviour changes**
- Float output is no longer clamped to [-1, 1] (libopus does not clip float output).
  Use `decode_i16` for 16-bit output.
- The automatic mode choice follows libopus. At 64 kb/s a 16 kHz VoIP encoder
  now codes CELT, as libopus does at that bitrate. At speech rates (for example
  16-24 kb/s, or the automatic 19 kb/s of 20 ms frames) it codes SILK.
- Stereo input is coded as mono at low bitrates, as libopus does.
- Encoder output changed to follow libopus in many places (stereo pitch
  pre-filter, `theta_rdo` at complexity 8 and above, SILK pitch analysis, pulse
  search, band energies, FFT and MDCT rounding on aarch64). A voiced 300-packet
  SILK stream is byte-identical to fixed-point libopus.

**Closer to libopus** (aarch64, against libopus 1.6.1)
- The CELT encoder writes the packets of libopus: all 1,080,000 packets of the
  960 streams of `celtenc_gap_report` (mono and stereo, 16 and 48 kHz, 2.5 to
  20 ms, CBR and VBR, complexity 4 to 9), up from 91.2 %. Through `OpusEncoder`
  the remaining differences at complexity 7 and above come from the missing
  speech and music analysis. Stereo input whose channels are scaled copies of
  each other still gives a few other packets at 128 kb/s per channel and more.
- The CELT decoder gives the samples of libopus: 51 of 51 CELT streams and 28 of
  29 Hybrid streams of `perf_harness modes` (8 and 25 before), and every
  concealed frame of the CELT loss tests and the multistream loss tests.
- What changed: the pitch analysis of the pre-filter and the loss concealment
  (one shared, libopus-rounded set of functions; the concealment's copies are
  gone), the comb filter (four taps had too few digits, and its rounding), the
  coarse energy prediction, a dynalloc boost that only libopus's main branch
  has, the 750 kb/s per channel limit and the silent VBR frame of the CELT
  encoder, `celt_log2` in the surround masking, and one `silk_lin2log` that
  gives libopus's result for 0.
- Speed: the pitch analysis got new kernels (the decimation and lag 4 inside
  the autocorrelation chain, eight samples per step in the correlation, energy
  sums inside the correlation passes), and the comb filter became faster.
  Against the state before these changes, in the 13 CELT scenarios of the
  bench, encode time changed by -1.4 to +0.9 % and decode time by -2.9 to
  +0.5 % (three alternating runs on a busy machine, where single rows vary by
  about 2 %). Frames of 20 ms are faster. Frames of 2.5 ms are slightly
  slower: the exact pitch analysis costs about 0.3 µs more per frame, and six
  more alternating runs measured +0.6 % encode time for stereo 128 kb/s c9
  with 2.5 ms frames.

**Performance** (Apple M4 Pro, see [Performance](#performance))
- Decoding takes 35-62 % less time than in 0.1.37 and is now faster than libopus
  in every measured scenario (it was 3-33 % slower).
- Encoding takes 33-56 % less time for SILK and Hybrid and 29-49 % less for
  mono CELT. Stereo CELT encoding takes 24-30 % less at complexity 0 and 5, and
  2-22 % more at complexity 9 and 10, where it now runs libopus's stereo pitch
  pre-filter and `theta_rdo`.
- The main changes: NEON kernels for the delayed-decision quantizer, warped
  autocorrelation, cross-correlations, comb filters and de-emphasis; FFT with
  FEAT_FCMA instructions; fewer copies and no zero-filling of buffers that are
  overwritten.
- Work for in-order cores such as the Cortex-A55: the pitch analysis of the
  CELT encoder, the post-filter of the CELT decoder, the SILK noise-shaping
  quantizers, and the PVQ search, the spreading rotation and the bit allocation
  of the CELT encoder now give such a core independent work at each step. The output is the same.
  Against the code before, run in turns (geometric means over the bench set),
  encoding takes 10.8-11.1 % less time on the Cortex-A55 of an Android phone,
  7.5 % less on its Cortex-A76 and 2.5-2.8 % less on Apple M4 Pro. Decoding
  takes 1.9-2.0 % less on the Cortex-A55 and 1.0 % less on the Cortex-A76.
- A second round of this work, again with the same output:
  - The PVQ search places the next pulse in advance and checks each pulse
    with fewer instructions.
  - `bits2pulses` and the two allocation searches use vector compares and
    sums instead of chains of dependent steps.
  - The cosines of the spreading rotation no longer call the C library.
  - The delayed-decision quantizer keeps its shaping coefficients in
    registers. On Apple cores it starts its shaping filter before the
    survivors are chosen.
  - The CELT decoder builds its spectrum in place. Its FFT, inverse MDCT and
    de-emphasis work on two independent groups per step.

  Against the code before, run in turns, encoding takes 3.1-3.5 % less time
  on Apple M4 Pro, 1.5 % less on the Cortex-A76 and 1.5-1.8 % less on the
  Cortex-A55. Decoding takes 1.9-2.2 %, 1.4 % and 1.7-1.8 % less.
- The C reference for these numbers is the release libopus 1.6.1, built from
  Xiph's release file with the settings of its own `configure`
  (`scripts/android/build-libopus.sh --src`). Earlier phone runs linked the
  libopus main branch, which writes other packets than 1.6.1 for many SILK,
  Hybrid and LFE streams.
- Fixes found in review: several vector kernels read past the end of short
  slices; `compute_theta` wrote through a shared reference, so its output depended
  on inlining.

**Fuzzing**
- The 19 fuzz targets in `fuzz/` build and run on the stable toolchain
  (`fuzz/fuzz.sh`), with overflow checks on. Six of them compare with
  libopus: packet parsing, padding and writing; a decoder fed packet
  streams with losses and FEC; the multistream decoder and encoder; an
  encoder whose settings change between frames; and the channel order of
  both encoders. They also check the packet length of a constant bitrate
  against libopus's rule, and that the multistream decoder writes only the
  samples it returns.

**Known gaps** (measured against libopus 1.6.1)
- libopus's speech/music analysis (`analysis.c`) and discontinuous transmission
  are not ported.
- x86-64 code was only compile-checked in this release, not run.
- Two FEC streams of the decoder agree with libopus to 185 and 186 dB instead of
  bit for bit.
- SILK stereo with a constant bitrate at 12 kHz can lose its stereo image on
  tones, as the fixed-point build of libopus does; the float build keeps it
  ([docs/10](docs/10-differences-from-libopus.md#silk)).

### Unreleased (earlier)

- **Fix: SILK CBR with in-band FEC busted packets into concealed frames
  (issue #51).** SILK's frame target now leaves room for the packet's LBRR
  section, as libopus's does. This ports `enc_API.c`'s per-frame target, for
  mono and stereo: the moving average of LBRR bits (`nBitsUsedLBRR`), the
  bit reservoir across packets (`nBitsExceeded`) and the balance within a
  multi-frame packet (`bitsBalance`). The frame cap no longer takes the
  LBRR bits off a second time, since the rate control already counts them.
  At 16-48 kb/s, 20 ms and 40% loss, mono busts 0-1 packets in 50 (was up
  to 8; libopus 0-2) and stereo 0-2 (was up to 15). Stereo carries LBRR in
  22-31 packets (was 5-20; libopus 29-30). Mono's main frame codes as well
  as without FEC (worst window 4.6 dB, was 0.3). Hybrid's SILK layer also
  pays back its overshoot of the target rate, as libopus's does.
- **Stereo SILK/Hybrid codes the side, as libopus does (issue #48).** L and R
  are resampled separately and converted to mid/side at the internal rate by
  a bit-exact port of libopus's `silk_stereo_LR_to_MS`: predictor estimation
  and quantization, stereo-width smoothing, the panned-mono and reduced-width
  decisions, and the mid/side bit split. A second SILK channel codes the
  side, with its own VAD, LBRR (in-band FEC) and rate control, and libopus's
  side reset after mid-only frames. On the #42 signal each channel now
  measures within 0.3 dB of libopus (R: +3.8 → +5.1 dB, libopus +5.1).
  Anti-phase input, which decoded as silence, measures +7.0 dB per channel,
  as libopus does. Two unrelated voices go from +1.9/+2.8 to +6.0/+9.3 dB
  (libopus +5.4/+8.0). Hybrid splits its SILK rate per channel, as libopus's
  `compute_silk_rate_for_hybrid` does. Mono output is unchanged, byte for
  byte. Under CBR with in-band FEC the side's LBRR at first crowded LBRR
  out of many packets; #51 (above) fixes that.
- **Fix: stereo SILK/Hybrid decoded as L ≈ −0.63·mid, R ≈ 2.63·mid (issue
  #42).** The encoder codes stereo SILK as the mid only, but wrote stereo
  predictor index 0, which dequantizes to `pred_Q13 = (0, −13364)`. The
  decoder applies the predictor to mid-only frames too, so the left channel
  came out phase-inverted and the right ~8 dB hot. It now writes the zero
  predictor, joint symbol 12, as libopus does for its `toMono` frames. Hybrid
  also fed SILK the interleaved L,R input as if it were mono; it is now
  downmixed like SILK-only, which lifts the 24/48 kHz Hybrid mid from −2.4 dB
  to +5.8 dB SNR (libopus +5.9 dB). Both channels now carry the mid
  (L = R ≈ 0.85·mid, SNR +5.2/+3.8 dB, libopus +5.1/+5.1 dB with real side
  coding), until #48 (above) adds the side.
- **Fix: the SILK decoder's stereo predictor levels match libopus.** The
  half-step constant was `(1 << 16) / 10` = 6553 instead of libopus's
  `SILK_FIX_CONST(0.1, 16)` = 6554, so 45 of the 75 levels came out a few
  Q13 units off and libopus stereo streams decoded at ~65 dB against libopus
  instead of bit-exactly. The stereo state is no longer cleared on frames
  whose predictor is zero (libopus clears it only on a mono→stereo switch),
  and a lost frame keeps the last mid-only flag and conceals the side only
  if the last good frame coded one (dec_API.c).
- **Fix: SILK-only VBR ignored `bitrate_bps` (issue #43).** The SILK target
  rate came from the packet budget, which under VBR is the caller's output
  buffer: a 1500-byte buffer asked SILK for ~510 kb/s, so every bitrate coded
  the same ~87-byte packets (16 kHz, 20 ms). The rate now follows libopus
  1.6's `bits_target` / `total_bitRate` (opus_encoder.c): the requested
  bitrate, capped by the buffer, less the TOC byte. The buffer stays the hard
  cap (`maxBits`). At 12-64 kb/s, 16 kHz VBR now codes 21-69 bytes a packet,
  against libopus's 27-73. CBR and Hybrid streams are byte-identical to
  before. Known gap: at wideband, opus-rs codes fewer excitation bits than
  libopus at the same gains (0.78x libopus's size at 12 kb/s, 1.01x at
  narrowband). That is a separate SILK encoder divergence, tracked for
  follow-up.
- **Fix: mediumband (12 kHz) SILK decoded to noise (issue #53).** The decoder
  gave 12 kHz the NB/MB NLSF codebook (order 10) but the wideband LPC order
  (16), so it built a 16-tap synthesis filter from ten decoded NLSFs. Every
  mediumband packet, from libopus or from opus-rs, decoded to loud noise at
  every output rate (-14 to -18 dB against libopus). The order and the
  codebook now come from one branch, as in libopus's `decoder_set_fs.c`: mono
  mediumband decodes bit-exactly against libopus, and stereo matches it as
  far as the stereo predictor allows (issue #42).

### 0.1.37

- **Fix: starved CBR budgets (issues #45, #46).** Budgets below 3 bytes per
  frame (and other starved budgets) now emit libopus's TOC-only "PLC" packet
  — padded under CBR, with the 40/60 ms multiframe TOC rules — instead of
  failing with a range-coder error (debug builds panicked in
  `RangeCoder::shrink`). Budgets below ~6 kb/s (9 kb/s above 20 ms) switch to
  CELT-only mode and code small valid packets, as libopus does
  (opus_encoder.c:1226-1286, 1525-1527). Known gap: at 5-6 byte CELT packets
  the payload's tail bits differ from libopus's, costing some SNR — tracked
  for follow-up.
- **Fix: 60 ms SILK packets cap the first two frames at 2/5 and 3/4 of the
  packet (enc_API.c parity, from PR #44).** Without the caps a 60 ms packet's
  early frames could leave the last one under its no-pulse fallback cost.
- **CELT transients are analysed on unfiltered history (PR #47).** The
  tone/transient analysis saw a prefiltered overlap head against an
  unfiltered body, reading a comb-filter step as a transient on steady
  pitched input; 48 kHz Audio at 20-24 kbps lost its first ~200 ms. Now
  lag 0, correlation 1.00 from the first window, and only the onset frame
  is coded with short blocks.

### Unreleased

- **Fix: CELT from 24 kHz input stuck at ~20 dB SNR (issue #54).** When
  `run_prefilter` cancelled the pitch pre-filter for a frame (libopus 1.6's
  `cancel_pitch`), it restored the unfiltered input over the whole frame.
  libopus still fades the previous frame's comb filter out over the overlap
  (celt_encoder.c:1572-1584), and the decoder's post-filter always applies
  that fade. So every cancel right after an "on" frame left 120 samples the
  decoder post-filtered but the encoder never pre-filtered. From 24 kHz the
  zero-stuffed input is dominated by its image above 12 kHz, so the filter
  was cancelled whenever the pitch period was odd, in 18 of 66 frames on the
  issue's chirp. The SNR was 20.1 / 20.9 / 21.1 dB at 32 / 64 / 128 kbps and
  is now 29.4 / 36.2 / 41.9 dB. libopus 1.6.1 scores 38.8 / 41.9 dB at 64 /
  128 kbps (35.2 / 41.1 with its tonality analysis off, which opus-rs does
  not port). The 48 kHz chirp packets are byte-identical.
- **Fix: CELT flagged transients on steady pitched input (issue #38).** The
  encoder ran `tone_detect` and `transient_analysis` on a buffer whose
  overlap head was the previous frame's *prefiltered* signal while the body
  was unfiltered; libopus 1.6 analyses the unfiltered `prefilter_mem` tail
  (celt_encoder.c:2017). Wherever the comb filter was active, the step at the
  head read as a transient: on the issue's chirp, 30 of 65 frames were coded
  with short blocks at every bitrate (libopus: only the onset frame). With
  bits to spare this was inaudible, but 48 kHz `Audio` at 20-24 kbps
  (CELT-only) lost its first ~200 ms (correlation 0.56-0.66); it is now lag 0
  and correlation 1.00 from the first window, like libopus's CELT.
  `prefilter_mem` is also kept current while the comb filter is off (stereo,
  hybrid, complexity < 5), as libopus's always-run `run_prefilter` does.

### 0.1.36

- **Fix: in-band FEC emitted no LBRR at CBR (issue #36).** The LBRR payload
  reused the main frame's quantized pulses with only the first gain index
  raised, so it cost as many bits as the main frame and never fit a CBR
  packet. `silk_LBRR_encode` is now ported faithfully: the excitation is
  re-quantized through the noise-shaping quantizer at the raised LBRR gain
  with its own NSQ state and gain chain, gated on speech activity, with
  `silk_setup_LBRR` (gain increase 7, then `max(7 − 0.2·loss, 2)`) and
  opus_encoder.c's `decide_fec` (rate thresholds + hysteresis + bandwidth
  reduction). CBR mono now recovers the previous packet through libopus
  `decode_fec` at 3.8 dB median (previously 0.0 dB: no LBRR at all).
- **Fix: 48 kHz SILK/hybrid encoder delay and startup (issue #38).** The
  encoder input was resampled with the stateless `down2`/`down2_3` decimators
  plus an internal-rate delay buffer, lagging libopus's pre-skip by up to 17
  samples. The encoder now runs the libopus `silk_resampler` (encoder
  direction: `delay_matrix_enc` + private down-FIR, ported with the
  `silk_Resampler_*_COEFS` tables) on the API-rate input, and prefills the
  SILK encoder on CELT→SILK restarts. Measured with the issue's harness:
  Voip 32 kbps lag −5..−2 (libopus's own encoder: identical), Audio 24 kbps
  startup correlation 0.56 → ≥ 0.77.
- **Fix: panics found by fuzzing.** 2.5 ms CELT frames aborted in the
  encoder's lm search (off-by-one: lm=0 was excluded from the valid set);
  2.5/5 ms frames reached the SILK encoder, whose VAD band split is only
  valid for ≥ 10 ms frames — SILK-only now rejects sub-10 ms frame sizes
  (they are CELT-only durations in libopus) and the VAD bails out on
  non-multiple-of-8 lengths.
- **no_std:** the crate builds and the full test suite runs with
  `--no-default-features --features libm` (`RUST_MIN_STACK=33554432`, see
  `scripts/check_no_std.sh`). 13 fuzz targets run clean.

### 0.1.35

- **Fix: SIGILL on AVX-without-FMA CPUs (issue #30, PR #31).** Nine kernels
  compiled with `#[target_feature(enable = "avx,fma")]` / `"avx2,fma"` were
  dispatched after checking only AVX/AVX2. New `compat::x86_has_avx_fma()` /
  `x86_has_avx2_fma()` probes gate them, so Sandy/Ivy Bridge and VMs that mask
  FMA no longer fault on the first `vfmadd`.
- **Fix: frame-loss concealment (issue #15, PR #32).** Packets of 0 or 1 bytes
  were decoded using the placeholder byte's ToC, which switched the decoder's
  mode/bandwidth/SILK rate and could panic in the resampler. Losses now conceal
  in the previous mode, as libopus does; an empty packet conceals `frame_size`
  samples instead of erroring, and a mono marker no longer errors on a stereo
  decoder.
- **Fix: in-band FEC LBRR desync (issue #27, PR #33).** LBRR frames were coded
  with `CODE_INDEPENDENTLY_NO_LTP_SCALING` while every decoder reads
  `CODE_INDEPENDENTLY`, so a voiced LBRR frame desynced the main frame that
  followed it. The LBRR gain bump now raises only index 0 of independently
  coded frames, as libopus does.
- **Fix: SILK decoder alignment with libopus (issue #34).** Mono SILK output
  read from `w_silk_buf[0][2..]` instead of libopus's `[1..]`, running one
  internal-rate sample early, and equal API/SILK-internal rates bypassed
  `silk_resampler`, dropping its `inputDelay`. Mono SILK now matches libopus at
  lag 0 at 16 kHz and 48 kHz.
- **Fix: 40/60 ms frames (issue #35).** Frame sizes were validated by
  `sampling_rate % frame_size == 0`, which rejected 60 ms (16.67 frames/s)
  before encoding. Validation is now duration-based. SILK-only carries a
  60 ms frame directly (3 internal SILK frames); CELT and Hybrid, which cap a
  frame at 20 ms, repacketize 20 ms sub-frames into one code 1/2/3 packet, as
  libopus does. This also fixes the code-3 VBR packet parser, which read frame
  lengths and frame data interleaved instead of all lengths first (libopus
  layout) — a latent bug that only surfaced once the encoder emitted code 3.
- **Fix: 24 kHz CELT decoded to garbage (issue #37).** The CELT encoder always
  coded all 21 bands, while the decoder derives its end band from the TOC
  (SWB = 19 at 24 kHz); coding one band more than the decoder reads
  desynchronised the range coder, turning every 24 kHz CELT packet into noise.
  The encoder now takes `end_band` from the Opus bandwidth
  (`CELT_SET_END_BAND`). It also zero-stuffs the API-rate input by
  `resampling_factor(Fs)` and scales/zeroes the MDCT output for non-48 kHz
  rates, as libopus `celt_preemphasis` / `compute_mdcts` do. 24 kHz `Audio`
  now matches libopus (per-window correlation ~1.0, was ~0.04).
- **Tests:** new libopus-oracle suites for SILK multi-frame/FEC (#27),
  frame-loss concealment (#15), mono alignment (#34), 40/60 ms frames (#35)
  and 24 kHz encoding (#37).

### 0.1.34

- **New: `OpusEncoder::encode_i16()` — native PCM16 entry point (issue #28).**
  Callers holding PCM16 audio no longer need to convert to f32 before encoding.
  In the SILK/VoIP path the integer high-pass biquad (`hp_cutoff_i16`) is fed
  directly from the input — no f32 round-trip, no `[i16; 11520]` temporary
  conversion buffer. Output is byte-for-byte identical to `encode()` called
  with the same audio converted via `sample as f32 / 32768.0` (SILK-only,
  Hybrid and CELT-only modes; verified by a parity test matrix across
  8/12/16/24/48 kHz, mono/stereo, complexity 0-10, CBR/VBR, in-band FEC).
  The float `encode()` API is unchanged.
- **Fix: `no_std` builds (`--no-default-features --features libm`) compile
  again.** The `multistream` decode module (added in 0.1.33) uses `Vec`-based
  plumbing and is now gated behind the default `std` feature; the core codec
  remains heap-free `no_std`.
- **Zero warnings:** `cargo check` and `cargo clippy --all-targets` are clean
  for the default and `no_std` feature sets (dead CELT state fields removed,
  tests/examples/benches lint-clean).

### 0.1.29

- **Default build is now heap-backed (issue #12).** A new `heap` feature (on by
  default, requires `std`) allocates the large codec state with `Box`, shrinking
  `OpusEncoder`/`OpusDecoder` from ~250 KB / ~175 KB to ~1 KB each. Constructing,
  encoding and decoding now fit on a 768 KiB stack (previously `OpusEncoder::new`
  alone needed ~850 KiB in release / ~2 MiB in debug, overflowing small-stack
  threads such as Windows' 1 MiB main thread). Disable default features for the
  heap-free `#![no_std]` build, which keeps the inline layout.
- **Tests:** add `tests/stack_usage_test.rs` asserting the structs stay <4 KiB
  with `heap`, >100 KiB without it, and that a full construct + encode + decode
  round-trip fits in a 768 KiB stack thread.

### 0.1.28

- **Fix: high-bitrate bitstream interop with libopus (issue #11).** The ported `BAND_ALLOCATION` table was missing one `200` in its final row (7 entries instead of 8), shifting the whole row. That row is only selected by the allocator at high bitrates, so encoded stereo streams above ~160 kbps decoded to garbage in libopus (and the crate decoder mis-decoded libopus streams at the same rates). Table now byte-identical to libopus.
- **Fix: `celt_pvq_u`/`celt_pvq_v` panic for `k >= 129`.** `compute_u`'s fixed-size buffer is now domain-checked (`k <= MAX_PVQ_K = 128`); out-of-domain calls return `u32::MAX` instead of panicking or silently wrapping.
- **Fix: silent u32 overflow in PVQ codebook sizes.** `compute_u`/`unext`/`celt_pvq_v` use saturating arithmetic, so an out-of-domain `(n, k)` can never produce a plausible-but-wrong codebook size.
- **Fix: `celt_pvq_u_lookup` row-extent aliasing.** Lookups whose `max(n, k)` exceeds a row's actual coverage now compute instead of reading the next row's block.
- **Fix: `OpusEncoder::encode` now returns an error when the range coder overflows its packet budget** (previously bytes were silently dropped).
- **Tests:** add full-domain PVQ table verification vs exact big-int recurrence, out-of-domain saturation checks, and a high-bitrate (128–192 kbps) stereo interop test that cross-checks against libopus.

### 0.1.27

- **`#![no_std]` with no `alloc`**: the crate is now fully heap-free — no global allocator is required. All `Vec`/`Box` working buffers were replaced with an internal `FixedVec<T, N>` (inline `[MaybeUninit<T>; N]`); `std::sync::LazyLock` was replaced with a heap-free `OnceCell`; no_std float math (`sin`/`cos`/`sqrt`/…) is routed through an optional `libm` feature (pure-Rust musl port).
  - Verified on `thumbv7em-none-eabi` (Cortex-M), `wasm32-unknown-unknown`, and Linux no_std. `scripts/check_no_std.sh` reproduces.
  - Feature flags: `std` (default) → 0 runtime deps; `default-features = false, features = ["libm"]` → 1 dep (`libm`).
  - Encoder/decoder structs are inline and large (~250 KB / ~175 KB); place them in a `static` or dedicated buffer on constrained targets.
- **Performance:** unchanged vs 0.1.26 (measured within ±0.5% on `opus_real`; no per-frame allocation overhead).
- **Conformance:** RFC 6716 1276-byte packet cap enforced on the Opus path; the standalone `RangeCoder`/CELT buffer supports up to 2048 bytes.
- `wav_test` example now tags output files by build (`_std` / `_nostd`) for A/B listening across build modes.

## License

See [COPYING](COPYING) for the original Opus license (BSD-3-Clause).

## Links

- **RustPBX**: <https://github.com/restsend/rustpbx>
- **RustRTC**: <https://github.com/restsend/rustrtc>
- **SIP Stack**: <https://github.com/restsend/rsipstack>
- **Rust Voice Agent**: <https://github.com/restsend/active-call>
