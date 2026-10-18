# Fuzz testing for opus1

This directory is a separate package, `opus1-fuzz`. Each target is a program for libFuzzer, the coverage-guided fuzzing engine of LLVM. libFuzzer calls the target with generated inputs, keeps the inputs that reach new code, and changes them further. The targets look for:

- panics, including integer overflow (the build turns on overflow checks and debug assertions);
- results that differ from libopus, the C reference implementation of Opus;
- packets that libopus cannot decode, or decodes to another final range (the state of the range coder after the packet).

## What you need

- The stable Rust toolchain. Neither nightly nor `cargo-fuzz` is needed.
- libopus 1.6.1, found as for the tests of the main crate. On macOS with Homebrew:

  ```bash
  brew install opus
  export LIBOPUS_LIB_DIR=/opt/homebrew
  export LIBOPUS_NO_PKG=1
  ```

  [docs/09-testing.md](../docs/09-testing.md#how-libopus-is-linked) explains how the C library is found.

## Running

`fuzz.sh` builds the targets and runs them:

```bash
fuzz/fuzz.sh list                          # the targets
fuzz/fuzz.sh build                         # build all targets
fuzz/fuzz.sh fuzz_decoder_sequence 600     # fuzz one target for 600 seconds
fuzz/fuzz.sh fuzz_packet 600 -fork=4       # the same with 4 processes
fuzz/fuzz.sh all 300                       # every target for 300 seconds each
```

- The corpus of a target (the inputs worth keeping) is in `fuzz/corpus/<target>/`. A later run starts from it.
- An input that makes a target fail is written to `fuzz/artifacts/<target>/`, as `crash-<hash>` for a panic.
- Arguments after the number of seconds go to libFuzzer. `-fork=N` runs N processes. `-help=1` lists all options.

### How the build works

A target is built by cargo as an ordinary binary, with the options cargo-fuzz uses when it builds without a sanitizer:

```bash
RUSTFLAGS="-Cpasses=sancov-module \
  -Cllvm-args=-sanitizer-coverage-level=4 \
  -Cllvm-args=-sanitizer-coverage-inline-8bit-counters \
  -Cllvm-args=-sanitizer-coverage-pc-table \
  -Cllvm-args=-sanitizer-coverage-trace-compares \
  --cfg fuzzing -Cdebug-assertions -Coverflow-checks" \
CARGO_TARGET_DIR=fuzz/target \
cargo build --release --target "$(rustc -vV | sed -n 's/^host: //p')" \
  --manifest-path fuzz/Cargo.toml --bins
```

- The `-C` options make the compiler count which code each input reaches. libFuzzer, linked in by the `libfuzzer-sys` crate, reads those counts.
- `--target` is needed. Without it, `RUSTFLAGS` also applies to build scripts, which then fail to link.
- Run the command from the root of the repository. Its paths are relative to it. cargo also reads `.cargo/config.toml` from the directory it runs in, so settings kept there apply only then. `fuzz.sh` runs cargo in `fuzz/`, so it can be called from any directory.
- The programs are in `fuzz/target/<host>/release/`. One runs as `fuzz_packet fuzz/corpus/fuzz_packet -max_total_time=60`.
- `--bins` leaves out `fuzz_dnn_blob`, which requires the package feature `dnn`. Build it with `--bin fuzz_dnn_blob --features dnn` in place of `--bins`, as `fuzz.sh` does.

### With cargo-fuzz on nightly

cargo-fuzz still works on the nightly toolchain. By default it also builds with the address sanitizer, which checks the memory accesses of the Rust code, the `unsafe` vector code included:

```bash
cargo install cargo-fuzz
cargo +nightly fuzz run fuzz_decoder_sequence --release -- -max_total_time=600
```

## Reproducing and minimizing a failure

```bash
fuzz/fuzz.sh repro fuzz_packet fuzz/artifacts/fuzz_packet/crash-<hash>
fuzz/fuzz.sh minimize fuzz_packet fuzz/artifacts/fuzz_packet/crash-<hash>
```

`repro` runs the one input and prints the panic message. `minimize` runs libFuzzer with `-minimize_crash=1`. It writes a smaller input that fails the same way to `<file>.min`.

A fixed failure becomes a test in `tests/fuzz_regressions.rs`, with the minimized input as a byte array in the test.

## Targets

### Targets that compare with libopus

| Target | What must hold |
|---|---|
| `fuzz_packet` | `packet::parse` (plain and self-delimited), `nb_frames`, `nb_samples` and `samples_per_frame` give libopus's results. `pad` and `multistream_pad` give the bytes of `opus_packet_pad` and `opus_multistream_packet_pad`. `write` gives the bytes of the libopus repacketizer, plain, self-delimited and padded. `multistream_validate` accepts what libopus's multistream decoder accepts. Written and padded packets parse to the frames they were made of. |
| `fuzz_decoder_sequence` | A stream of packets from a libopus encoder with changing settings (2.5 to 120 ms, FEC, DTX, forced modes), with losses, damaged packets, `decode_fec`, `decode_i16` and decoders of either channel count. After every call, opus1 and libopus's decoder accept or refuse alike, return the same number of samples, and end in the same final range. While no packet was lost or damaged, the output is within 50 dB of libopus's. opus1's output buffer sometimes has more or less room than the frame size. More room changes nothing. With too little room opus1 may refuse, and then it keeps its state. |
| `fuzz_multistream_decoder` | A layout and packets from the input, or from libopus encoders joined into multistream packets. libopus's `opus_multistream_decoder` and opus1 accept or refuse the layout alike. After every call they accept or refuse alike, return the same number of samples, and end in the same final range. While no packet was lost or damaged, the output is within 50 dB of libopus's. opus1's output buffer sometimes has more or less room than the frame size. More room changes nothing. With too little room opus1 may refuse, and then it keeps its state. A decoder built from the layout as the channel mapping table of an Ogg Opus header is accepted too. opus1 writes only the samples it returns, with `f32` and with 16-bit output. |
| `fuzz_multistream_encoder` | A layout, settings and input from the input. libopus accepts and refuses the same layouts and picks the same streams for a mapping family. Every packet decodes in libopus's multistream decoder to one frame, in the encoder's final range. Until a frame is concealed, opus1's decoder gives output within 50 dB of libopus's. A second encoder with the same settings, given `f32` samples where the first gets the same audio as `i16`, writes the same packets. One sample too few is refused, and the encoder keeps its state. With a constant bitrate that is explicit or the maximum, a packet has the length of libopus's rule (`opus_multistream_encoder.c` 918-928). |
| `fuzz_encoder_settings` | Bitrate (also automatic and maximum), VBR or CBR and the VBR constraint, complexity, forced mode, bandwidth, channels, FEC, loss percentage, LFE, energy mask, prediction and frame size change between frames. A setting libopus refuses is refused. Every valid frame is encoded. With CBR every packet has the length of libopus's rule (`cbr_packet_len` in `fuzz_targets/common/mod.rs`). Every packet decodes in libopus to one frame, in the encoder's final range. Until a frame is concealed, opus1's decoder gives output within 50 dB of libopus's. A second encoder with the same settings, given `f32` samples where the first gets the same audio as `i16`, writes the same packets. One sample too few is refused, and the encoder keeps its state. |
| `fuzz_channel_order` | A stereo `OpusEncoder` or a `MultistreamEncoder` (mapping family or channels, streams and mapping) and libopus's encoder with the same settings code a tone of its own per input channel. Both are decoded by libopus. Wherever libopus keeps the tones of the other input channels out of an output channel (at least 20 dB below their input level), opus1's output channel may hold none of them at more than -6 dB of its input level. A channel on the wrong output gives about 0 dB. Stereo streams that an encoder codes as mono, and those that SILK codes with a constant bitrate, are left out (see the comment at the top of the target). |

FEC is forward error correction, DTX discontinuous transmission, VBR and CBR variable and constant bitrate, and LFE the low-frequency-effects channel.

The output check allows 50 dB because opus1's decoder differs from libopus's only in rounding: on real streams the two agree to 136 dB or better ([docs/10](../docs/10-differences-from-libopus.md#decoder)). After a lost or damaged packet the output is not compared. The concealment of a loss searches the decoded signal for its pitch, and a rounding difference can change what it finds. These count as a loss as well:

- a packet with a frame of 0 or 1 bytes, which the decoder conceals;
- a packet that takes the stream from CELT-only to SILK or Hybrid. The decoder conceals 5 ms in CELT and fades them into the new frame, unless the packet carries a redundant CELT frame.

### Other targets

| Target | What it does |
|---|---|
| `fuzz_encoder` | Encodes one frame of samples from the input. 10, 20 and 40 ms frames must be encoded, 30 ms frames refused. |
| `fuzz_decoder` | Decodes the input as a packet. |
| `fuzz_roundtrip` | Encodes one frame and decodes it. The decoder must return the frame and end in the encoder's final range. |
| `fuzz_silk_encoder` | Encodes at 8 to 16 kHz with SILK settings. |
| `fuzz_celt_encoder` | Encodes with CELT settings and signals. |
| `fuzz_overflow` | Encodes with extreme settings and samples. A bitrate, complexity or loss percentage that libopus refuses must be refused. A packet that is written must decode in the encoder's final range. |
| `fuzz_celt_large_frame` | Decodes CELT packets with frame sizes near the decoder's buffer limits. |
| `fuzz_celt_decoder_boundary` | Decodes short, invalid and long packets with small and large frame sizes. |
| `fuzz_silk_decoder` | Decodes the input as SILK packets at every rate. |
| `fuzz_hybrid_mode` | Encodes and decodes Hybrid frames at 24 and 48 kHz, in the encoder's final range, and decodes the input. |
| `fuzz_encoder_boundary` | Encodes with every rate, channel count and application, extreme settings and output buffers of 1 to 4000 bytes. |
| `fuzz_multi_frame` | Encodes and decodes a run of frames. Each must decode in the encoder's final range to finite samples. |
| `fuzz_packet_parsing` | Decodes the input with all 256 first bytes at every rate. |
| `fuzz_dnn_blob` | Loads weight blobs of the neural network models: the input as a blob, or a built-in blob with changed headers and data, cut short or not. Nothing may panic. Every model that loads must have the layer shapes of its descriptors, and every array it reads must lie inside the blob. The same bytes, copied with `DnnBlob::from_bytes` and loaded with `OpusDecoder::set_dnn_blob`, must either load some groups of networks or change nothing, and the decoder must then still conceal a lost packet at complexity 10. Built with the package feature `dnn`, which turns on opus1's `osce-bwe`; `fuzz.sh` builds it in a cargo call of its own, so the other targets keep opus1's default features. |

## Adding a target

1. Create `fuzz_targets/<name>.rs`:

   ```rust
   #![no_main]
   use libfuzzer_sys::fuzz_target;

   fuzz_target!(|data: &[u8]| {
       // Call opus1 with `data` and assert what must hold.
   });
   ```

   `fuzz_targets/common/mod.rs` has wrappers for the libopus encoder, decoder, multistream encoder, multistream decoder and repacketizer. A target uses them with `mod common;`.

2. Add the target to `Cargo.toml`:

   ```toml
   [[bin]]
   name = "<name>"
   path = "fuzz_targets/<name>.rs"
   test = false
   doc = false
   bench = false
   ```
