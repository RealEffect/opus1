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
- **High performance:** on Apple M4 Pro, decoding takes 0.56-0.81x and encoding 0.48-0.71x the time of C libopus 1.6.1 built at `-O2`, the optimization level of libopus's own build, at every complexity, with the speech and music analysis in both from complexity 7 on. Against the Homebrew build of libopus 1.6.1, which behaves like a build at `-Os` and takes about 24 % more time to encode and 12 % more to decode, the ranges are 0.49-0.76x and 0.37-0.63x. opus1 is also faster than libopus on the big and the small cores of an Android phone, and on average over the bench set on an Intel Core i7-12700 (x86-64) (see [Performance](#performance))
- **Multichannel:** multistream encoder and decoder (mapping families 0, 1, 2 and 255), interoperable with libopus
- **libopus controls:** the encoder settings, their defaults and the values they refuse follow libopus's `OPUS_SET_*` requests (automatic bitrate, VBR and its constraint, complexity, in-band FEC, packet loss, forced mode, bandwidth and channels, signal type, input bit depth, DTX, frame duration from 2.5 to 120 ms)
- **Packets:** the `packet` module parses, writes and pads packets and has libopus's repacketizer. The `extensions` module reads and writes the padding extensions of libopus 1.5 and later
- **64-bit ARM and x86-64:** on 64-bit ARM the output is checked bit for bit against libopus 1.6.1. On x86-64 it is correct but not bit-exact with libopus: opus1's x86 vector kernels do not follow libopus's, and libopus's own x86 output depends on the C compiler (see [docs/10](docs/10-differences-from-libopus.md#platforms))
- **The speech and music analysis of libopus:** from complexity 7 on, the encoder estimates whether the input is speech or music and which bandwidth it uses, as libopus does, with the same results to the last bit on 64-bit ARM. CELT-only packets at complexity 7 to 10 are libopus's byte for byte
- **The neural decoder features of libopus 1.5 and 1.6 (optional):** deep packet loss concealment, the LACE and NoLACE speech enhancement of SILK, and the bandwidth extension of SILK wideband to fullband. Each is compiled only with its cargo feature and runs only when the decoder complexity is raised, as in libopus: `set_complexity(5)` for deep PLC (packet loss concealment), 6 for LACE, 7 for NoLACE, and 4 with `set_osce_bwe(true)` for the bandwidth extension. On 64-bit ARM the output is that of libopus 1.6.1 built with these features (without loop vectorization) to the last bit.
- **Deep redundancy (DRED, optional):** with the cargo feature `dred`, the encoder adds a compressed description of up to about one second of earlier speech to each packet (`set_dred_duration`, libopus `OPUS_SET_DRED_DURATION`), and the decoder rebuilds a burst of lost packets from the next packet that arrives (`OpusDredDecoder`, `OpusDred`, `OpusDecoder::dred_decode`). CELT-only packets with DRED, and the DRED data in SILK and Hybrid packets with the `Audio` application and VBR, are those of libopus 1.6.1 built with DRED, byte for byte. Decoding gives libopus's output to the last bit.
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

// Encode PCM16 directly. As libopus `opus_encode` does, this codes with an
// input precision of 16 bits: the output is byte-for-byte identical to
// `encode()` after `set_lsb_depth(16)`, called with the same audio
// converted via `sample as f32 / 32768.0`.
let input_i16 = vec![0i16; 320]; // 20ms frame at 16kHz
let bytes = encoder.encode_i16(&input_i16, 320, &mut output).unwrap();

// Decode. `decode` writes the samples it returns and nothing after them.
let mut decoder = OpusDecoder::new(16000, 1).unwrap();
let mut pcm = vec![0.0f32; 320];
let samples = decoder.decode(&output[..bytes], 320, &mut pcm).unwrap();
assert_eq!(samples, 320);
```

The public API is the crate root (`OpusEncoder`, `OpusDecoder`, `Application`,
`Bandwidth`, `Bitrate`, `ForcedMode`, `FrameDuration`, `Signal`, `InbandFec`,
`pcm_soft_clip`) and the modules `packet`, `extensions` and `multistream` (which
needs `std`). The neural network features add `DnnBlob` and `DnnModels`, and
`dred` adds `OpusDredDecoder`, `OpusDred` and `DredParse`. [docs/06](docs/06-using-the-library.md) shows the settings and
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
structs are small (about 2.6 KB for `OpusEncoder`, 0.1 KB for `OpusDecoder`) and
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
features = ["libm", "analysis"]
```

The public API (`OpusEncoder`/`OpusDecoder` with `encode`/`decode` writing into caller-provided buffers) is identical in both configurations.

### Feature flags

| Feature  | Default | Description |
|----------|---------|-------------|
| `std`    | yes     | Enables OS-backed runtime x86 SIMD detection (CPUID). Without it the crate is `#![no_std]`. |
| `heap`   | yes     | Allocates the large codec state on the heap (`Box`), shrinking `OpusEncoder`/`OpusDecoder` to a few KB so they can live on ordinary stacks. Requires `std`. |
| `analysis` | yes   | The speech and music analysis of the encoder (libopus `analysis.c`), which runs from complexity 7 on. Without it the encoder decides as libopus does below complexity 7, and its state is about 26 KB smaller. Works with and without `std`. |
| `deep-plc` | no    | Deep packet loss concealment (PitchDNN, the PLC network, FARGAN) with its built-in weights (1,432,768 bytes). Runs at decoder complexity 5 and above. |
| `osce`   | no      | LACE and NoLACE, the speech enhancement of SILK (decoder complexity 6 and 7; weights 1,573,440 bytes). Turns on `deep-plc`, as libopus's `--enable-osce` does. |
| `osce-bwe` | no    | The bandwidth extension of SILK wideband to 48 kHz fullband output, switched on with `set_osce_bwe` (weights 685,888 bytes). Turns on `osce`. |
| `dred`   | no      | Deep redundancy (DRED): the encoder writes DRED data (`OpusEncoder::set_dred_duration`), the decoder rebuilds lost packets from it (`OpusDredDecoder`, `OpusDred`, `OpusDecoder::dred_decode`). Embeds the RDOVAE encoder and decoder (weights 565,440 bytes). Turns on `deep-plc`, as libopus's `--enable-dred` does. |
| `dnn`    | no      | The code that the neural network features share (layer kernels, weight blob parser). Not meant to be selected alone: `deep-plc`, `osce`, `osce-bwe` and `dred` turn it on. |
| `test-libopus-dnn` | no | Test only: compiles the tests that compare with a libopus built with its neural network code (see [docs/09](docs/09-testing.md#tests-of-the-neural-network-code)). |
| `libm`   | no      | Required for `#![no_std]` builds — provides float math via a pure-Rust port of musl `libm`. Not needed when `std` is on. |

> Runtime dependency footprint: `std` build → **0 deps**; `no_std` build → **1 dep** (`libm`, pure Rust).

### Notes for `no_std` users

- **Large structs:** without `heap`, every working buffer is inline, so
  `OpusEncoder` and `OpusDecoder` are large: on aarch64 the encoder takes
  about 154 KB with the `analysis` feature and 128 KB without it, and the
  decoder about 60 KB. Place them in a
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

Measured on 2026-10-18 with the code of commit `f952b46` (the speech and music
analysis and DTX on `d541821`). Both C references are the release libopus
1.6.1. The main one is built at `-O2`, the optimization level of libopus's own
build: `scripts/android/build-libopus.sh --host --rtcd --src <1.6.1 source>`
builds it with Apple clang and the flags of libopus's `configure`, plus the
run-time CPU detection that `configure` uses on Android. With that detection
libopus's SILK encoder uses its NEON quantizer, as on a phone. The second ratio
column is the Homebrew libopus 1.6.1, which the tests link. It behaves like a
build at `-Os` and calls the C version of that quantizer. The code before and
after the analysis ran in turns, each linked with each libopus build: four
harness programs, three passes, while other programs loaded the machine.

| Scenario | opus1 encode µs | Ratio vs `-O2` | Ratio vs Homebrew | opus1 decode µs | Ratio vs `-O2` | Ratio vs Homebrew |
|---|---:|---:|---:|---:|---:|---:|
| SILK NB 8 kHz mono 12 kb/s, complexity 0 | 9.1 | 0.71 | 0.63 | 2.0 | 0.64 | 0.62 |
| SILK WB 16 kHz mono 20 kb/s, complexity 10 | 46.7 | 0.52 | 0.39 | 3.9 | 0.62 | 0.59 |
| SILK WB 16 kHz stereo 32 kb/s, complexity 9 | 90.7 | 0.55 | 0.39 | 7.0 | 0.61 | 0.57 |
| Hybrid SWB 24 kHz mono 24 kb/s, complexity 9 | 62.9 | 0.51 | 0.37 | 7.3 | 0.60 | 0.53 |
| Hybrid FB 48 kHz stereo 40 kb/s, complexity 9 | 130.3 | 0.56 | 0.40 | 13.9 | 0.56 | 0.49 |
| CELT FB 48 kHz mono 64 kb/s, complexity 0 | 11.8 | 0.62 | 0.57 | 8.3 | 0.75 | 0.70 |
| CELT FB 48 kHz mono 64 kb/s, complexity 10 | 24.8 | 0.50 | 0.43 | 9.8 | 0.73 | 0.61 |
| CELT FB 48 kHz stereo 128 kb/s, complexity 5 | 35.7 | 0.54 | 0.46 | 19.2 | 0.75 | 0.60 |
| CELT FB 48 kHz stereo 128 kb/s, complexity 10 | 58.0 | 0.58 | 0.46 | 19.2 | 0.75 | 0.60 |
| CELT FB 48 kHz stereo 128 kb/s, 2.5 ms, complexity 9 | 11.1 | 0.68 | 0.59 | 3.2 | 0.81 | 0.74 |

From complexity 7 on, for input at 16 kHz or more, both encoders run the speech
and music analysis, so every ratio compares the same work. opus1's analysis
takes 0.26 to 0.28 of the time of libopus's on this machine and adds 2.1 to
3.9 µs per 20 ms frame. The C SILK encoder in a float build of libopus is the
floating-point one; opus1 ports the fixed-point one.

Over all 28 scenarios of the bench set (geometric means), opus1 takes 0.572 of
the encode time and 0.669 of the decode time of libopus 1.6.1 at `-O2`, and
0.461 and 0.598 of the Homebrew build. Before the analysis, when only libopus
ran it, encoding took 0.546 and 0.443 in the same passes. The full table is in
[docs/08-performance.md](docs/08-performance.md#latest-results), and
[The C reference](docs/08-performance.md#the-c-reference) explains the builds.

On an Android phone (MediaTek MT6833), against libopus 1.6.1 built with the
NDK's clang at `-O2` and the flags of libopus's `configure` for Android, opus1
is faster in all 30 scenarios on both kinds of core, also where both encoders
run the speech and music analysis (geometric means over the bench set, commit
`f952b46`): encoding 0.668 and decoding 0.768 of libopus's time on the
Cortex-A76, 0.698 and 0.824 on the in-order Cortex-A55. Before the analysis,
when only libopus ran it, encoding took 0.625 and 0.650. opus1's analysis
alone takes 0.36 to 0.40 of the time of libopus's on these cores. See
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

### x86-64 (Intel Core i7-12700)

Measured on 2026-10-21 on an Intel Core i7-12700 (AVX2, FMA and AVX-VNNI, no
AVX-512) against libopus 1.6.1 at `-O2` (geometric means over the bench set):
opus1 takes 0.90 of libopus's time to encode and 0.81 to decode under Linux in
WSL2, and 0.88 and 0.80 natively on Windows. Low-complexity SILK encoding took
1.04 to 1.10 of libopus's time. On x86-64 the output is correct but not
bit-exact with libopus (see
[docs/10](docs/10-differences-from-libopus.md#platforms)).

### Earlier measurement: x86-64 (AVX2/FMA), opus-rs 0.1.x

Measured on AMD Ryzen 7 5700X with the earlier `opus_vs_c` benchmark (mono encode only,
complexity 0).

| Config | Pure Rust | C Opus | Ratio |
|--------|-----------|--------|-------|
| 8 kHz / 20 ms VoIP | **39.9 ms** | 40.6 ms | 0.98× (**Rust 2% faster**) |
| 16 kHz / 20 ms VoIP | **66.8 ms** | 67.1 ms | 1.00× (**Rust 0.5% faster**) |
| 16 kHz / 10 ms VoIP | 73.2 ms | **72.5 ms** | 1.01× (within noise) |
| 48 kHz / 20 ms Audio | **25.1 ms** | 28.4 ms | 0.88× (**Rust 12% faster**) |
| 48 kHz / 10 ms Audio | **29.7 ms** | 31.2 ms | 0.95× (**Rust 5% faster**) |

## Release Notes

### 0.2.0 (unreleased)

The crate is renamed from `opus-rs` to `opus1`: code that used `opus_rs::` now
uses `opus1::`. This release follows libopus 1.6.1 and was checked against it
throughout: every fix has a test that compares opus1 with the C library under
the same settings. The full list of changes, with the measurements behind each
fix, is in [docs/changelog-0.2.0.md](docs/changelog-0.2.0.md).

**Breaking changes**
- The crate name: `opus1` instead of `opus-rs`.
- The settings of `OpusEncoder` are methods instead of public fields:
  `set_bitrate`, `set_complexity`, `set_vbr` (instead of `use_cbr`, with the
  opposite meaning), `set_inband_fec` and `set_packet_loss_perc`, each with its
  getter. Each setter refuses the values that libopus refuses, and the setting
  then keeps its old value.
- The default bitrate is libopus's automatic one, `Bitrate::Auto` (for example
  19 kb/s for 16 kHz mono in 20 ms frames), instead of a fixed 64 kb/s.
  `Bitrate` moved from `opus1::multistream` to the crate root.
- `OpusEncoder::enable_hybrid_mode` and the field `OpusDecoder::hybrid_skip_celt`
  are removed. To ask for Hybrid packets, call
  `set_force_mode(Some(ForcedMode::Hybrid))` and `set_bandwidth`.
- The internal modules are private. The public API is the crate root and the
  modules `packet`, `extensions` and `multistream`.
- `OpusDecoder::decode` writes only the samples it returns and no longer sets
  the rest of `output` to zero. Float output is no longer clamped to [-1, 1].
- The encoder now chooses mode, bandwidth and channel count as libopus does, so
  its packets change: for example a 16 kHz `Voip` encoder codes CELT at 64 kb/s
  and SILK at speech rates. `encode_i16` codes with an input precision of 16
  bits, as libopus `opus_encode` does.

**New features**
- Multichannel: `multistream::MultistreamEncoder` (mapping families 0, 1, 2 and
  255, with libopus's rate allocation, LFE stream and surround masking) and
  `MultistreamDecoder` (any valid layout). Packets interoperate with libopus in
  both directions.
- Every frame duration from 2.5 to 120 ms: frames of 80, 100 and 120 ms are new,
  and `set_expert_frame_duration` with the type `FrameDuration` sets one.
- The speech and music analysis of libopus (feature `analysis`, on by default),
  which runs from complexity 7 on, and discontinuous transmission (DTX):
  `set_dtx`, `dtx` and `in_dtx`.
- The neural decoder features of libopus 1.5 and 1.6 (features `deep-plc`,
  `osce`, `osce-bwe`): deep packet loss concealment, the LACE and NoLACE speech
  enhancement of SILK, and the bandwidth extension (BWE) of SILK wideband to
  fullband.
- Deep redundancy (DRED, feature `dred`): `OpusEncoder::set_dred_duration`,
  and `OpusDredDecoder`, `OpusDred` and `OpusDecoder::dred_decode` on the
  decoder side.
- Packet extensions of libopus 1.5 and later: the module `extensions`,
  `packet::Repacketizer`, `packet::pad_with_extensions`, `packet::unpad` and
  `packet::multistream_unpad`.
- New controls of libopus. Decoder: `decode_i16`, `decode_fec`, `final_range`,
  `reset_state`, `last_packet_duration`, `set_gain`, `pitch` and
  `set_phase_inversion_disabled`. Encoder: `set_force_mode`, `set_bandwidth`,
  `set_max_bandwidth`, `set_force_channels`, `set_signal`, `set_lsb_depth`,
  `set_inband_fec_mode`, `set_vbr_constraint`, `set_application`,
  `set_voice_ratio`, `set_phase_inversion_disabled` and `lookahead`. The
  multistream encoder and decoder forward the controls that libopus's forward.
- The 24-bit integer calls of libopus 1.6: `OpusEncoder::encode24`,
  `OpusDecoder::decode24` and `decode_fec24`, and their multistream versions.

**Quality fixes**
- 48 kHz `Voip`: SILK now reads the float high-pass filter's output, as the
  float build of libopus does. The integer filter added a constant of about -9
  to the 16-bit signal, which SILK coded as a burst of low-frequency energy in
  7.6 to 9.2 % of the 10 ms frames of speech. On 222 TSP speech files the error
  against the original, relative to libopus's, fell from between 1.21 and 3.28
  to between 0.98 and 1.09.
- 8 to 24 kHz `Voip`: the first SILK frame after CELT-only frames no longer
  codes energy below 200 Hz.
- 24 kHz: Hybrid super-wideband packets decoded with 21 CELT bands instead of
  19, so the 8-12 kHz band of every 24 kHz speech stream at the usual VoIP
  bitrates decoded to unrelated content.
- Stereo SILK and Hybrid: the decoded channels had the wrong sign and level
  (issue #42), and the side signal is now coded as libopus codes it (issue #48).
- SILK: VBR ignored the bitrate (issue #43); CBR with in-band FEC turned
  packets into concealed frames (issue #51); 12 kHz (mediumband) packets
  decoded to noise (issue #53); 10 ms SILK and Hybrid frames decoded to noise.
- Closer to libopus on 64-bit ARM: the CELT encoder writes libopus's packets in
  all 1,080,000 packets of the 960 streams of `celtenc_gap_report` (up from
  91.2 %), and the CELT decoder gives libopus's samples in 51 of 51 CELT
  streams and 28 of 29 Hybrid streams of `perf_harness modes`.

**Speed** (opus1's time divided by libopus 1.6.1's; below 1 means opus1 is
faster; geometric means over the bench set; see [Performance](#performance))

| | Encoding | Decoding |
|---|---:|---:|
| Apple M4 Pro, libopus at `-O2` | 0.572 | 0.669 |
| Apple M4 Pro, Homebrew libopus | 0.461 | 0.598 |
| Cortex-A76 (Android phone, MediaTek MT6833) | 0.668 | 0.768 |
| Cortex-A55 (same phone) | 0.698 | 0.824 |
| Intel Core i7-12700, Linux in WSL2 | 0.90 | 0.81 |
| Intel Core i7-12700, Windows | 0.88 | 0.80 |

On x86-64, low-complexity SILK encoding took 1.04 to 1.10 of libopus's time.
Against opus-rs 0.1.37 on Apple M4 Pro, decoding takes 35-62 % less time, and
encoding 33-56 % less for SILK and Hybrid and 29-49 % less for mono CELT.
Stereo CELT encoding at complexity 9 and 10 takes 2-22 % more time, because it
now runs libopus's stereo pitch pre-filter and `theta_rdo`.

**Platforms**
- 64-bit ARM is the reference platform. There the output is checked bit for
  bit against libopus 1.6.1, and every test that requires libopus's bits
  passes. [docs/10](docs/10-differences-from-libopus.md) lists the remaining
  differences and their sizes.
- x86-64 gives correct output but is not bit-exact with libopus. opus1's x86
  vector kernels do not follow libopus's x86 kernels, and libopus's own x86
  output depends on the C compiler. About 40 tests that require libopus's bits
  are ignored on targets other than 64-bit ARM. x86-64 was run on an Intel
  Core i7-12700 (AVX2, FMA and AVX-VNNI), under Linux in WSL2 and natively on
  Windows ([docs/10](docs/10-differences-from-libopus.md#platforms)).
- The minimum Rust version is 1.98: the 64-bit ARM dot-product intrinsics of
  the neural network code are stable from that version on.

**Known gaps** (see [docs/10](docs/10-differences-from-libopus.md#not-implemented))
- The scalable quality extension of libopus 1.6 (QEXT) is not ported.
- Mapping family 3 (ambisonics mixed through a matrix, libopus's
  `opus_projection_*` functions) is not implemented.
- A few controls have no equivalent: the decoder's `OPUS_GET_BANDWIDTH` (the
  `packet` functions read the bandwidth from the packet's first byte) and
  `OPUS_SET_IGNORE_EXTENSIONS`, and the multistream decoder's
  `OPUS_GET_BANDWIDTH` and `OPUS_GET_LAST_PACKET_DURATION`.
- SILK stereo with a constant bitrate at 12 kHz can lose its stereo image on
  tones, as the fixed-point build of libopus does.

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

### 0.1.37 (earlier changes)

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

opus1 is released under the BSD-3-Clause license; the full text is in [COPYING](COPYING). It is a port of libopus, so the copyright of the libopus authors (Xiph.Org, Skype Limited, Octasic, Jean-Marc Valin, Timothy B. Terriberry, CSIRO, Gregory Maxwell, Mark Borgerding, Erik de Castro Lopo, Mozilla, Amazon) applies together with that of restsend.com (opus-rs) and gaojiangjie (opus1). The neural network weights in `dnn-weights/` come from libopus 1.6.1 under the same license ([dnn-weights/COPYING](dnn-weights/COPYING)).

## Links

- **RustPBX**: <https://github.com/restsend/rustpbx>
- **RustRTC**: <https://github.com/restsend/rustrtc>
- **SIP Stack**: <https://github.com/restsend/rsipstack>
- **Rust Voice Agent**: <https://github.com/restsend/active-call>
