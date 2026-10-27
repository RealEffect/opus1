# The architecture of opus1

This page describes how `opus1` is built, as an overview:

- the parts of the codec and how they depend on each other;
- the path of one encode call and of one decode call;
- the public interface, a map of the source files and the feature flags;
- the rules the code follows.

For the details, [07 Architecture](guide/07-architecture.md) lists every source file with the libopus file it ports. [03 The encoder](guide/03-encoder.md) and [04 The decoder](guide/04-decoder.md) follow the code paths step by step. The [documentation and project index](README.md) lists the documents, the tests, the scripts and the commands.

## The design in brief

- **A port of libopus.** `opus1` is a port of libopus 1.6.1, the reference implementation of Opus in C. Each Rust function computes what the C function computes. The code follows the structure of the C code and mostly keeps its function names.
- **No C at run time.** The library has two optional Rust dependencies (`libm`, `portable-atomic`) and no C code. libopus is linked only by the tests, examples, benchmarks and fuzz targets, to compare results.
- **How close it is to libopus.**
  - On 64-bit ARM (aarch64), the output is checked bit for bit against libopus 1.6.1.
  - On x86-64, the output is correct but not bit-exact. Its vector code does not follow libopus's x86 vector code.
  - The neural network code gives the 64-bit ARM bits of libopus on every processor.
- **Number formats.** SILK (the speech coder) is a port of libopus's fixed-point SILK, in integers. CELT (Constrained Energy Lapped Transform, the music coder) is a port of libopus's floating-point build.
- **Embedded use.** The crate builds as `#![no_std]`, and also without a heap allocator. Below the Opus layer no code allocates memory.
- **aarch64 needs NEON.** On 64-bit ARM the vector code uses NEON without a run-time check. A `compile_error!` stops the build for aarch64 targets without NEON, such as `aarch64-unknown-none-softfloat`.

## The parts of the codec

```text
application
  |
  +-- opus1::multistream     MultistreamEncoder, MultistreamDecoder (feature alloc)
  |     |                    splits the channels into streams of 1 or 2 channels
  |     v
  +-- OpusEncoder, OpusDecoder         the Opus layer (src/lib.rs)
  |     |
  |     +-- speech and music analysis  src/analysis/   encoder only (feature analysis)
  |     +-- SILK                       src/silk/       speech coder, fixed point, 8, 12 or 16 kHz
  |     +-- CELT                       top of src/     transform coder, floating point, 48 kHz
  |     +-- RangeCoder                 range_coder.rs  one bitstream per frame, shared by SILK and CELT
  |     +-- neural network code        src/dnn/        deep PLC, OSCE, DRED (optional features)
  |
  +-- opus1::packet, opus1::extensions   read and write packets, without a codec state
```

- **The Opus layer** (`OpusEncoder` and `OpusDecoder` in `src/lib.rs`) chooses, for each packet, the mode, the audio bandwidth and the channel count. It runs SILK, CELT or both, and builds the packet: the table-of-contents (TOC) byte, the frames and the padding.
- **The three modes.**
  - SILK-only: SILK codes the audio.
  - CELT-only: CELT codes the audio.
  - Hybrid: SILK runs at 16 kHz and codes the audio up to 8 kHz. CELT codes only its bands from 8 kHz up. Both write into the same range coder, SILK first.

  When the mode changes, the encoder adds a 5 ms redundant CELT frame, and the decoder cross-fades with it. [01, The three modes](guide/01-what-is-opus.md#the-three-modes) explains the modes.
- **The speech and music analysis** runs in the encoder at complexity 7 to 10, for input at 16, 24 or 48 kHz. It estimates whether the input is speech or music and which bandwidth it really uses. The choices of mode, channel count, bandwidth and FEC use these estimates, and so does the CELT encoder.
- **Multistream** is built on top of the Opus layer. `MultistreamEncoder` holds one `OpusEncoder` per stream, and `MultistreamDecoder` holds one `OpusDecoder` per stream. Each stream has 1 or 2 channels. The multistream layer adds the channel mapping, the sharing of the bitrate between streams and the surround analysis. See [05 Multichannel audio](guide/05-multichannel.md).
- **The neural network code** is optional. It plugs into the codec in three places:
  - Deep packet loss concealment (PLC), feature `deep-plc`: the SILK and CELT decoders conceal a lost packet with it, at decoder complexity 5 and up.
  - OSCE (Opus Speech Coding Enhancement), feature `osce`: it improves the speech that the SILK decoder outputs, with LACE at decoder complexity 6 and NoLACE at 7 and up. With `osce-bwe` and `set_osce_bwe(true)`, a bandwidth extension takes SILK wideband to 48 kHz in place of the resampler.
  - DRED (deep redundancy), feature `dred`: the encoder writes a compressed description of the past speech into the padding of each packet, as the packet extension with ID 126. A decoder parses it with `OpusDredDecoder` and rebuilds lost audio with `OpusDecoder::dred_decode`, through deep PLC.
- **The packet modules** (`opus1::packet` and `opus1::extensions`) parse, write, pad and repacketize packets, and read the packet extensions. They hold no codec state.

## One encode call

`OpusEncoder::encode` (or `encode_i16`, `encode24`) runs these steps. [03, Data flow](guide/03-encoder.md#data-flow) has the full diagram.

1. `encode_impl` checks the arguments and takes the bitrate of this packet from the size of the output buffer. It runs the digital silence test and, at complexity 7 to 10 with input at 16, 24 or 48 kHz, the speech and music analysis.
2. `decide_mode_and_bandwidth` chooses the channel count, the mode, the bandwidth, FEC (forward error correction) and whether a redundant frame is needed.
3. `encode_frame` codes one frame:
   - It filters the input: a high-pass filter for the `Voip` application, a DC (direct current) removal for the others.
   - In SILK-only and Hybrid mode, the SILK encoder codes the frame, resampled to 8, 12 or 16 kHz.
   - In Hybrid and CELT-only mode, the CELT encoder codes it, into the same range coder after SILK.
   - On a change of mode it adds the redundant CELT frame.
4. `finish_frame` ends the packet. With the feature `dred`, it puts the DRED data into the padding. With a constant bitrate, it pads the packet to its full size.

Packets of 40 to 120 ms in CELT-only or Hybrid mode, and of 80 to 120 ms in SILK-only mode, go through `encode_multiframe`. It codes 2 to 6 frames with `encode_frame` and joins them into one packet with `packet::write`.

## One decode call

`OpusDecoder::decode` (or `decode_i16`, `decode24`) runs these steps. [04, Data flow](guide/04-decoder.md#data-flow) has the full diagram.

1. `decode_native` takes the packet. An empty packet means a lost packet, which it conceals. `decode_fec` takes the next packet and decodes the FEC copy of the lost frame from it, or conceals the frame when the packet has no copy. For a packet to decode, `decode_native` reads the TOC byte, and `packet::parse` finds the frames and their sizes.
2. For each frame, `decode_or_conceal_frame` either conceals it (a frame of 0 or 1 bytes) or calls `decode_frame`.
3. `decode_frame` decodes one frame:
   - In SILK-only and Hybrid mode, the SILK decoder decodes it at 8, 12 or 16 kHz and resamples it to the output rate.
   - In Hybrid and CELT-only mode, the CELT decoder decodes its bands and runs the inverse MDCT (modified discrete cosine transform) at 48 kHz. In Hybrid mode its output is added to the SILK output.
   - On a change of mode it cross-fades with the redundant frame, or with 5 ms of concealment.
4. The result is float samples, not clipped. `decode_i16` soft-clips them and rounds them to 16 bits.

A lost packet is concealed in the mode of the last frame: by SILK in `src/silk/plc.rs`, or by CELT in `celt.rs`. In Hybrid mode SILK conceals the audio below 8 kHz and CELT adds noise above 8 kHz, and the two are added. See [04, Packet loss concealment](guide/04-decoder.md#8-packet-loss-concealment).

## Public API

The public interface is the crate root and three modules. Everything else is private.

- **Crate root** ([`src/lib.rs`](../src/lib.rs)):
  - Codecs: `OpusEncoder` and `OpusDecoder`.
  - Setting types: `SampleRate`, `Channels`, `Application`, `Bandwidth`, `Bitrate`, `ForcedMode`, `FrameDuration`, `Signal`, `InbandFec`.
  - Errors: `Error` and `ErrorKind` ([`src/error.rs`](../src/error.rs)).
  - Other items: `pcm_soft_clip`, `VERSION`, `LIBOPUS_VERSION`.
  - With `deep-plc` (or a feature that turns it on): `DnnBlob`, `DnnModels`.
  - With `dred`: `OpusDred`, `OpusDredDecoder`, `DredParse`, `DredProcessing`.
- **`opus1::packet`** ([`src/packet.rs`](../src/packet.rs)): parse, write and pad packets; the repacketizer; self-delimited framing; multistream packet checks.
- **`opus1::extensions`** ([`src/extensions.rs`](../src/extensions.rs)): the packet extensions stored in the padding of a packet.
- **`opus1::multistream`** ([`src/multistream/`](../src/multistream)), feature `alloc`: `MultistreamEncoder`, `MultistreamDecoder`, `ChannelLayout`, `ChannelMappingTable` and the mapping families 0, 1, 2 and 255.

Main methods:

- `OpusEncoder`: `new`, `init` (build in caller-provided memory), `encode` (`f32`), `encode_i16`, `encode24`, `reset_state`, `final_range`, and one setter and getter per libopus `OPUS_SET_*` control (`set_bitrate`, `set_complexity`, `set_vbr`, `set_inband_fec`, `set_packet_loss_perc`, `set_dtx`, `set_force_mode`, `set_bandwidth`, `set_dred_duration` and others).
- `OpusDecoder`: `new`, `init`, `decode`, `decode_i16`, `decode24`, `decode_fec` (and its `_i16` and `24` forms), `dred_decode` (and its `_i16` and `24` forms), `final_range`, `reset_state`, `set_complexity`, `set_gain`, `set_osce_bwe`, `set_dnn_blob`.

The hidden module `opus1::internals` ([`src/internals.rs`](../src/internals.rs)) exposes internal items to this repository's tests, benchmarks, examples and fuzz targets. It exists only with the test-only feature `__internals`, which is outside semantic versioning. A test that needs a new internal item gets it through a re-export in `src/internals.rs`.

Where to read more: [Using the library](guide/06-using-the-library.md) for recipes and settings, [Multichannel audio](guide/05-multichannel.md) for `multistream`, [Opus packets](guide/02-packets.md) for `packet` and `extensions`. The README's quick start shows encode and decode in a few lines.

## Source map

[Architecture](guide/07-architecture.md#source-layout) lists every file under `src/` with the libopus file it ports. This section groups the files by the question you may have.

### The Opus layer and shared code (top of `src/`)

| File | What it does |
|---|---|
| `lib.rs` | `OpusEncoder` and `OpusDecoder`, and the public setting types. Encoder path: `encode_impl` → `decide_mode_and_bandwidth` → `encode_frame` → `finish_frame`; `encode_multiframe` for packets of 40 to 120 ms. Decoder path: `decode_native` → `decode_frame`. Also the size limits (`OPUS_MAX_CHANNELS`, `OPUS_MAX_FRAME` and others) and the includes of the documents as documentation tests. |
| `error.rs` | `Error` and `ErrorKind`, one kind per libopus error code. |
| `packet.rs`, `extensions.rs` | The public `packet` and `extensions` modules. |
| `range_coder.rs` | `RangeCoder`: the entropy coder that SILK and CELT both write into. |
| `hp_cutoff.rs` | The encoder's input filters: high-pass for speech, DC (direct current) removal for other audio. |
| `mathops.rs` | Float helpers that round as libopus does on the target processor (`mac`, `celt_log2`, `celt_exp2`). |
| `fixedvec.rs` | `FixedVec<T, N>`: a growable buffer with fixed capacity, stored inline. CELT, the FFT, the MDCT and the range coder keep their buffers in it. |
| `in_place.rs` | Tools to build large states directly in their final memory (`write_fields!`, `write_zero`, `boxed`). |
| `simd_mem.rs` | Safe loads and stores of vector registers for the NEON and x86 kernels (`load_f32x4`, `store_i16x4_uninit` and others). Each function takes a reference to an array of exactly the values one instruction reads or writes, and holds the only `unsafe` block that access needs. Also `array_at` and `array_at_mut`, which take such an array from a slice. |
| `compat.rs` | `OnceCell` without `std`, a stable insertion sort, x86 feature probes (`x86_has_avx2` and others), and the `Math` trait that maps float functions to `libm` without `std`. |
| `deep_plc.rs`, `osce_args.rs` | The arguments that `OpusDecoder` passes down to SILK and CELT for the neural network features. Empty types when the feature is off. |
| `internals.rs` | The test-only module described above. |

### CELT, the transform coder (top of `src/`, floating point)

| File | What it does |
|---|---|
| `celt.rs` | `CeltEncoder` and `CeltDecoder`: transient analysis, pitch pre-filter and post-filter, allocation decisions, synthesis, CELT packet loss concealment. |
| `modes.rs` | `CeltMode`: band layout, window and tables, built once. |
| `bands.rs` | Band energies, normalisation, coding of all bands (`quant_all_bands`), stereo, spreading, anti-collapse. |
| `quant_bands.rs` | Coarse and fine quantization of band energies. |
| `rate.rs` | Bit allocation between bands (integer arithmetic). |
| `pvq.rs` | The pyramid vector quantizer (PVQ): pulse search and pulse coding. |
| `mdct.rs`, `kiss_fft.rs` | The modified discrete cosine transform (MDCT) and the fast Fourier transform (FFT) behind it. |
| `pitch.rs` | Pitch search, cross-correlations, inner products. |
| `celt_lpc.rs` | Linear prediction (LPC) and the filters of CELT loss concealment. |

### SILK, the speech coder (`src/silk/`, fixed point)

| Group | Files |
|---|---|
| Entry points | `enc_api.rs` (`SilkEncoder`, `silk_encode_frame`), `dec_api.rs` (`SilkDecoder`) |
| State and constants | `structs.rs` (encoder state), `decoder_structs.rs` (decoder state), `define.rs`, `tuning_parameters.rs`, `init_encoder.rs`, `init_decoder.rs`, `control_codec.rs`, `control_fixed.rs`, `control_snr.rs` |
| Encoder analysis | `vad.rs` (voice activity detection), `ana_filt_bank_1.rs`, `hp_variable_cutoff.rs`, `lp_variable_cutoff.rs`, `pitch_analysis.rs`, `lpc_analysis.rs`, `lpc_fit.rs`, `ltp_analysis.rs` (long-term prediction), `noise_shape_analysis.rs`, `stereo_lr_to_ms.rs` |
| Quantization | `nsq.rs` (noise-shaping quantizer, complexity 0 and 1), `nsq_del_dec.rs` (delayed-decision quantizer, complexity 2 and up), `gain_quant.rs`, `vq_wmat_ec.rs`, `nlsf.rs`, `nlsf_encode.rs`, `nlsf_del_dec_quant.rs` (NLSF: normalized line spectral frequencies) |
| Bitstream | `encode_indices.rs`, `encode_pulses.rs`, `shell_coder.rs`, `decode_indices.rs`, `decode_pulses.rs`, `nlsf_decode.rs`, `nlsf_unpack.rs`, `decode_parameters.rs` |
| Decoder synthesis | `decode_frame.rs`, `decode_core.rs`, `plc.rs` (packet loss concealment), `cng.rs` (comfort noise), `stereo_ms_to_lr.rs` |
| Signal processing and tables | `resampler.rs`, `sigproc_fix.rs`, `macros.rs` (fixed-point arithmetic helpers), `interpolate.rs`, `lin2log.rs`, `log2lin.rs`, `sigm.rs`, `sort.rs`, `tables.rs`, `tables_nlsf.rs` |

### The speech and music analysis (`src/analysis/`, feature `analysis`)

The encoder runs it from complexity 7 on. It estimates whether the input is speech or music and which bandwidth it really uses. `tonality.rs` is the analysis, `mlp.rs` and `mlp_data.rs` are its small neural network, `neon.rs` and `x86.rs` are vector kernels, `tables.rs` holds constants, and `mod.rs` holds the result type `AnalysisInfo`.

### Multistream (`src/multistream/`, feature `alloc`)

`encoder.rs` (`MultistreamEncoder`), `decoder.rs` (`MultistreamDecoder`), `layout.rs` (channel layouts and mapping families), `rate_alloc.rs` (how the bitrate is shared between streams), `surround.rs` (surround masking analysis), `mod.rs` (`ChannelMappingTable` and helpers).

### Neural network code (`src/dnn/`, features `dnn`, `deep-plc`, `osce`, `osce-bwe`, `dred`)

| Part | Files |
|---|---|
| Layers and kernels | `nnet.rs` (layer functions), `arch.rs` (`DnnArch`: which kernel variant runs), `vec_portable.rs` (the reference kernels), `vec_neon.rs`, `vec_x86.rs` |
| Weights and models | `weights.rs` (blob parser), `builtin.rs` (embedded weights), `holder.rs` (`DnnBlob`, `DnnModels`, `DecoderDnn`), `models/` (one generated file per network; `models/mod.rs` is written by hand) |
| Signal analysis | `freq.rs`, `burg.rs`, `lpcnet_tables.rs` |
| Deep packet loss concealment | `plc/` (`features.rs`, `pitchdnn.rs`, `fargan.rs`, `lpcnet_plc.rs`) |
| Speech enhancement (OSCE: Opus Speech Coding Enhancement) | `osce/` (`mod.rs`, `features.rs`, `nndsp.rs`, `lace.rs`, `nolace.rs`, `bbwenet.rs`, `math.rs`) |
| Deep redundancy (DRED) | `dred/` (`encoder.rs`, `decoder.rs`, `rdovae_enc.rs`, `rdovae_dec.rs`, `coding.rs`) |

### Evaluation tools (`src/eval/`, test-only, needs `std` and `__internals`)

Ports of libopus test programs, used by the tests and by `examples/eval`: `compare.rs` (`opus_compare`), `moc.rs` (the MOC distance), `demo.rs` (the decoding loop of `opus_demo`). They are not part of the codec.

### Generated files: do not edit by hand

| Files | Written by |
|---|---|
| `dnn-weights/*`, `src/dnn/models/*.rs` except `mod.rs`, `tests/dnn_gen/mod.rs` | `tools/dnn-gen` |
| `src/analysis/mlp_data.rs` | `scripts/gen_analysis_mlp_data.py` |
| `docs/opus-explained/data/opus-demo-data.js` | `cargo run --release --example opus_demo_data` |

## Feature flags

[`Cargo.toml`](../Cargo.toml) explains each flag in comments. [Using the library](guide/06-using-the-library.md#feature-flags) explains them for users, and [Architecture](guide/07-architecture.md#feature-flags-and-where-state-is-stored) explains what each changes inside the crate.

| Feature | Default | What it does |
|---|---|---|
| `std` | on | Run-time x86 processor checks. Turns on `alloc`. Off means `#![no_std]`. |
| `heap` | on | Keeps the large codec state in `Box`es, so the public structs are small. Turns on `alloc`. |
| `analysis` | on | The speech and music analysis of the encoder. |
| `alloc` | through `std` or `heap` | Needs a global allocator, not an operating system. Adds `multistream`. |
| `libm` | off | Required without `std`: float math functions from the `libm` crate. |
| `portable-atomic` | off | For targets without atomic compare-and-swap, such as Cortex-M0. |
| `dnn` | off | The shared neural network layer. No public API on its own. |
| `deep-plc` | off | Deep packet loss concealment, at decoder complexity 5 and up. |
| `osce` | off | Speech enhancement of SILK: LACE at decoder complexity 6, NoLACE at 7 and up. Turns on `deep-plc`. |
| `osce-bwe` | off | Bandwidth extension of SILK wideband to 48 kHz, at decoder complexity 4 and up after `set_osce_bwe(true)`. Turns on `osce`. |
| `dred` | off | Deep redundancy in the encoder and the decoder. Turns on `deep-plc`. |
| `test-libopus-dnn` | off | Test-only. Compiles the tests that call a libopus built with its neural network code. |
| `__internals` | off | Test-only. Adds `opus1::internals`. Applications never enable it. |

## Rules the code follows

Keep these when you change the code. The linked sections give the reasons.

1. **Match libopus.** A port follows the structure of the C code and mostly keeps its function names. On 64-bit ARM the tests require libopus 1.6.1's output bit for bit. Where `opus1` differs on purpose, [10](guide/10-differences-from-libopus.md) records the difference and its size. On x86-64 the aim is correct, clear audio and speed, not bit-exactness.
2. **One safe entry point per vector kernel.** A safe function checks slice lengths first, then calls the NEON, SSE or AVX version, or the portable one. The kernels themselves are safe functions with `#[target_feature]`. Their loads and stores go through `simd_mem.rs`. See [07, Code for specific processors](guide/07-architecture.md#code-for-specific-processors).
3. **Neural network kernels give the same bits on every processor:** those of libopus on 64-bit ARM. `vec_portable.rs` is the reference. See [07, The kernels of the neural networks](guide/07-architecture.md#the-kernels-of-the-neural-networks).
4. **`unsafe` code is documented.** Every `unsafe` block in library code has its own `// SAFETY:` comment, and the Clippy lint `undocumented_unsafe_blocks` is on. The crate does not allow `unsafe_op_in_unsafe_fn`, so the body of an `unsafe fn` also puts its unsafe operations in `unsafe` blocks. The one exception is the x86 module of `src/silk/nsq_del_dec.rs`. Uninitialized memory is never viewed as a slice of `T` before it is written. See [07, Unsafe code](guide/07-architecture.md#unsafe-code).
5. **No allocation below the Opus layer.** CELT, SILK and the range coder keep every buffer inline, with sizes fixed at compile time (`FixedVec`, arrays). Only `heap` puts the large states in `Box`es. See [07, Feature flags and where state is stored](guide/07-architecture.md#feature-flags-and-where-state-is-stored).
6. **Errors.** Public functions return `Result<T, Error>`. Input is checked before state changes, so a rejected packet leaves the decoder unchanged. SILK functions keep libopus's `i32` error codes inside. See [07, Error handling](guide/07-architecture.md#error-handling).
