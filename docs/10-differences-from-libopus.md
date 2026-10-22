# Differences from libopus

opus1 follows libopus 1.6, the reference implementation of Opus in C. This document lists where it does not. It names the features opus1 does not implement and the places where it differs on purpose. It then gives where the numeric output differs, and by how much, and ends with notes on platforms. Every figure was measured against a named build of libopus. Read this document before you depend on bit-for-bit agreement with libopus, or on a libopus control that opus1 may lack.

## Terms used here

- **Opus** is the audio codec defined in RFC 6716 (Request for Comments 6716, the Internet standard for Opus). **libopus** is its reference implementation in C.
- **SILK** is the speech coder inside Opus. **CELT** (Constrained Energy Lapped Transform) is the transform coder inside Opus. **Hybrid** is the mode in which SILK codes the audio below 8 kilohertz (kHz) and CELT the audio above it.
- The **modified discrete cosine transform (MDCT)** turns a block of CELT audio into its spectrum. It is computed with a **fast Fourier transform (FFT)**.
- **Voice over Internet Protocol (VoIP)** means calls over the Internet. `Voip` is also the name of the opus1 application setting for speech.
- Bitrates are in kilobits per second (kb/s) or bits per second (b/s).
- Durations are in milliseconds (ms) or seconds (s).
- **Complexity** is an encoder setting from 0 to 10. Higher values spend more processor time to improve quality. "c9" means complexity 9.
- **VBR** (variable bitrate) lets the packet size change from frame to frame. **CBR** (constant bitrate) keeps it fixed.
- **Fixed-point** code computes with integers. **Floating-point** code computes with `f32` or `f64`. libopus can be built either way.
- The **signal-to-noise ratio (SNR)**, in decibels (dB), measures how close a decoded signal is to a reference. Higher is closer.
- **Forward error correction (FEC)** in Opus is a low-bitrate copy of the previous frame inside a SILK packet.
- A **fused multiply-add (FMA)** computes `a * b + c` with a single rounding step instead of two.
- **Multistream** is the Opus format that packs several streams into one packet, for example for 5.1 surround sound.
- The **neural network features** of libopus 1.5 and later are deep packet loss concealment (deep PLC), the speech enhancers LACE and NoLACE, and the bandwidth extension (BWE). LACE, NoLACE and the BWE make up OSCE (Opus Speech Coding Enhancement). A **weight blob** is a file of network weights in libopus's format.
- A **novec** build of libopus is built at `-O2` with `-fno-vectorize -fno-slp-vectorize`, so the compiler turns no loop into vector code.

## What the figures were measured against

- The reference is **libopus 1.6.1 as built by Homebrew**, on Apple M4 Pro (64-bit ARM, macOS). It is a floating-point build.
- The C source of libopus's main branch differs from 1.6.1 in a few places. opus1 follows 1.6.1, checked against the arm64 object code of the Homebrew build and, for the fixed-point SILK encoder, against a fixed-point build of the 1.6.1 release. Two examples: `remove_doubling` has no second check of very short periods, and `dynalloc_analysis` gives band 0 no extra boost above 320 bytes per frame. [libopus main branch and the 1.6.1 release](#libopus-main-branch-and-the-161-release) lists every difference.
- Other builds of the release 1.6.1 give the same packets and samples as the Homebrew build in all 137 streams of the golden file, on the Mac and on an Android phone. The main branch of libopus, which the phone linked up to round 7, writes other packets in 86 of them (see [Testing](09-testing.md#which-libopus-source-to-build)).
- A build of 1.6.1 at `-O2`, the optimization level of libopus's own build, still differs from the Homebrew build in the last bit of some results. At `-O2`, clang turns loops of `celt/vq.c`, `celt/pitch.c`, `celt/celt_lpc.c` and `celt/celt_decoder.c` into vector code, which rounds some products before it adds them. opus1 follows the scalar code of the Homebrew build, which behaves like a build at `-Os`. So against an `-O2` build, 7 tests that require libopus's exact stereo angles, pitch lags and concealed frames fail, on the Mac and on the phone (see [Testing](09-testing.md#results-on-the-phone)). These are differences between two builds of libopus, not differences of opus1.
- **The speech and music analysis** (`src/analysis.c`, `src/mlp.c`) follows the Homebrew build too, in every field of every frame. Its arithmetic equals that of clang at `-Os`. Other builds of the same source round some of its sums differently:
  - At `-O2`, clang turns the sums of the tonality and the noisiness of each band (analysis.c, lines 657 and 658) into vector code that rounds each product before it adds it. The Homebrew build fuses them. A build without vector loops fuses two other sums, those of the cepstral values (lines 864 and 871), which the Homebrew build rounds.
  - Measured before the port, on 3,804 frames of speech and music at 16, 24 and 48 kHz: a copy of `analysis.c` built at `-O2` differs from the Homebrew build in 76 % to 97 % of the frames, by at most 7.0e-6 in a probability. The band of the detected bandwidth, the leakage values, the pitch ratio and the integer speech estimate did not differ in any frame.
  - So against libopus built at `-O2`, for example on an Android phone, the analysis can differ in the last bits of its probabilities, which can change a decision of the encoder now and then. In the 137 streams of the golden file it changed none: on the phone, the packets of libopus 1.6.1 built at `-O2` by the NDK's clang are those of the Homebrew build in all of them, also at complexity 7 to 10. [Testing](09-testing.md#results-on-the-phone) gives the tests that see the difference.
- Comparisons of the SILK encoder at the packet level also use **a fixed-point build of libopus** (`-DFIXED_POINT`, see [Testing](09-testing.md#checks-against-the-fixed-point-build-of-libopus)). It is built from the source of the official 1.6.1 release. Before October 2026 it was built from the source that the `opus-head-sys` dependency vendors, which is libopus's main branch, and opus1 followed one line of the main branch in its Burg analysis (see [SILK](#silk)).
- Most figures come from `perf_harness modes` (see [Performance](08-performance.md#the-harness)) over its 129 scenarios with 5 s clips, at the final integration state of this release. In those runs, "SNR" is the decoded audio against the encoder input. The opus1 encoder is compared with the libopus encoder under the same forced settings, and libopus decodes both streams.
- Figures marked **(earlier)** come from earlier stages of the work and were not measured again in the final state.
- An "SNR difference" below is opus1's SNR minus libopus's SNR. A positive value means opus1 is closer to the input.
- A "deficit" is the other way round: libopus's SNR minus opus1's SNR. A negative value means opus1 is closer.

## Not implemented

### Encoder features and controls

- The fixed-point build of libopus runs its own integer version of the speech and music analysis, at complexity 10 only. opus1 ports the float version, which the float build runs from complexity 7 on. Tests that compare with the fixed-point build run opus1's analysis at complexity 10 only, through the internal setting `opus1::internals::analysis::set_analysis_gate`.

### Decoder controls

The decoder decodes every valid packet, up to 120 ms. The decoder requests of libopus 1.6.1 have equivalents in `OpusDecoder` (phase inversion disabling, the output gain and the pitch since opus1 0.2), with two exceptions: `OPUS_GET_BANDWIDTH`, the bandwidth of the last packet, which `packet` functions read from the packet's first byte instead, and `OPUS_SET_IGNORE_EXTENSIONS` (see [Packet padding extensions](#packet-padding-extensions)). The decoder complexity (`OPUS_SET_COMPLEXITY`), the BWE switch (`OPUS_SET_OSCE_BWE`) and weight blobs (`OPUS_SET_DNN_BLOB`) act as in libopus when the matching feature is compiled: deep PLC from complexity 5, LACE at 6, NoLACE from 7, and the BWE from 4 (see [Optional libopus features](#optional-libopus-features)).

### Multichannel

- **Mapping family 3** (ambisonics mixed through a matrix, libopus's `opus_projection_*` functions) is not implemented. Families 0, 1, 2 and 255 are.
- The multistream encoder forwards every request of libopus's multistream encoder ctl as libopus does (since opus1 0.2 also the forced channel count, prediction, phase inversion, the application and `OPUS_GET_VOICE_RATIO`), except the quality extension (`OPUS_SET_QEXT`), which is not ported. libopus has no multistream request for `OPUS_GET_IN_DTX`; `OpusEncoder::in_dtx` of each stream encoder (`MultistreamEncoder::stream_encoder`) tells whether that stream is in DTX. The module documentation in `src/multistream/encoder.rs` names this former difference D5, and the surround bandwidth choice D1 (see [Deliberate differences](#deliberate-differences)).
- The multistream decoder forwards the complexity, the gain and phase inversion disabling, as libopus does, but has no getter for `OPUS_GET_BANDWIDTH` and `OPUS_GET_LAST_PACKET_DURATION`, which libopus answers from the first stream. `MultistreamDecoder::stream_decoder(0)` gives the first stream's `last_packet_duration`.

### Packet padding extensions

libopus 1.5 and later can store extensions in the padding of a packet. opus1 reads and writes them as libopus 1.6.1 does (`src/extensions.rs`), and its repacketizer, `pad`, `unpad` and their multistream versions carry or drop them as libopus's do (see [Packets](02-packets.md#packet-extensions)). The remaining differences:

- Without the feature `std`, a call that has to keep more than 32 extensions, or to copy more than 1275 bytes of old padding in `pad`, returns "Internal error". libopus allocates room for any number on its stack.
- The decoder control `OPUS_SET_IGNORE_EXTENSIONS` is not offered. In libopus 1.6.1 only the quality extension (QEXT), which is not ported, reads extensions in the decoder, so the control changes nothing there.

### Optional libopus features

libopus has features that are compiled only when it is configured with them. They are off by default:

- deep packet loss concealment (`--enable-deep-plc`);
- Deep REDundancy (DRED, `--enable-dred`);
- Opus Speech Coding Enhancement (OSCE, `--enable-osce`);
- the scalable quality extension (`--enable-qext`).

These options are listed in libopus's `configure.ac`. For deep PLC and OSCE, opus1 has the cargo features `deep-plc`, `osce` and `osce-bwe` (see [Using the library](06-using-the-library.md#neural-network-features-of-the-decoder)). Deep PLC, LACE, NoLACE and the bandwidth extension run in the decoder, and the decoded output is that of libopus 1.6.1 built with them (see [Neural network code](#neural-network-code)), with one exception: CELT-only frames that follow frames with the bandwidth extension. libopus 1.6.1 skips their CELT decoding and outputs silence until the next SILK frame; opus1 decodes them, as libopus main does (see [Deliberate differences](#deliberate-differences)).

A libopus built with the features also changes two things at every decoder complexity, even at 0. opus1 makes both changes when the feature is compiled:

- With deep PLC compiled, `silk_PLC_glue_frames` skips its fade-in for 16 kHz SILK frames (`silk/PLC.c` lines 477-480).
- With OSCE compiled, a sample of -32768 in a 20 ms SILK frame at 16 kHz becomes -32767 (`dnn/osce.c` lines 1672-1679).

With the feature `dred`, opus1 decodes DRED as libopus 1.6.1 built with `--enable-dred` does, bit for bit (see [DRED decoding](#dred-decoding)). Its encoder writes DRED data (`set_dred_duration`, libopus `OPUS_SET_DRED_DURATION`) as libopus does where the inputs of DRED are the same (see [DRED encoding](#dred-encoding)). opus1 does not implement the scalable quality extension.

## Deliberate differences

These differences are intended. Most are listed in the comment above `OpusEncoder::encode_impl` in `src/lib.rs`. The multistream ones are in `src/multistream/encoder.rs`.

| What | opus1 | libopus | Why |
|---|---|---|---|
| SILK encoder | Port of the fixed-point encoder (`silk/fixed`). | The floating-point build uses `silk/float`. | By design. |
| SILK input with the `Voip` application | At 8 to 24 kHz SILK gets the output of the integer high-pass filter `hp_cutoff` of the fixed-point build. That filter has its own state and runs on every frame, CELT-only frames included, as in the fixed-point build. At 48 kHz SILK gets the float filter's output, rounded to 16 bits. | The floating-point build filters the float signal and rounds it to 16 bits. The fixed-point build uses the integer filter at every rate. | Below 48 kHz opus1 follows the fixed-point build, whose SILK encoder it ports. At 48 kHz the integer filter adds a constant of about -9 to the 16-bit signal, which SILK codes as low-frequency energy in quiet frames (see [Paired comparison](#paired-comparison-with-the-evaluation-harness)). |
| SILK quantization gains | When the residual energy of a subframe that libopus computes gives more than twice the gain that the residual energy of the unscaled signal gives, the subframe takes the smaller gain. | The fixed-point build always takes the energy it computes from its signal scaled by the inverse subframe gains and rounded to 16 bits. The float build computes the energy without that rounding. | In a frame with a subframe near digital silence the rounding makes the energy several times too large. At a low SNR the gain is then too large, and the decoder plays a burst of energy below 200 Hz. See [SILK](#silk). |
| Packets of several frames with VBR: CELT and Hybrid at 40 to 120 ms, SILK at 80 to 120 ms | No limit of each frame to its share of the bitrate (libopus `opus_encoder.c`, line 1785). | Limits each frame. | Kept on purpose. It uses more bits, and with in-band FEC libopus loses much quality to the limit; see [Hybrid and mode switching](#hybrid-and-mode-switching). |
| Hybrid CELT bitrate | Never below 501 b/s. | No such floor. | On purpose. |
| Switch to CELT-only without room for the redundant frame | The CELT encoder is still reset. | Goes on from its last CELT frame. | The decoder then starts the CELT-only packets from the same reset state. |
| Energy mask (surround) | Copied, and applied again on the next frame after a CELT reset. | Keeps a pointer, which is lost after a CELT reset until the caller sets it again. | Rust ownership. The values are the same. |
| Multistream bandwidth with the automatic or maximum bitrate in surround mode (D1) | Chosen from the bitrate that was allocated. Codes fullband. | Compares the raw setting and always chooses narrowband. | Narrowband 5.1 at a high bitrate is not useful. |
| BWE switch and weight blobs on a multistream decoder | `MultistreamDecoder::set_osce_bwe` and `set_dnn_blob` set every stream. | The multistream decoder answers `OPUS_SET_OSCE_BWE` and `OPUS_SET_DNN_BLOB` with `OPUS_UNIMPLEMENTED`; each stream must be set through `OPUS_MULTISTREAM_GET_DECODER_STATE`. | Convenience. `stream_decoder_mut` still sets one stream. |
| CELT-only frames after frames with the bandwidth extension (BWE) | Decoded. | libopus 1.6.1 keeps the extended mode of the last SILK frame, so with the BWE on it skips the CELT decoding of the following CELT-only frames (and of their concealment): the output is silence until the next SILK or Hybrid frame, and the range decoder is not read. The final range of such a frame is the state of the range decoder before it reads the frame. | A defect, fixed in libopus main by commit `228a0f85` (2026-08-28), which sets the extended mode to CELT-only for those frames. opus1 follows main. The CELT-only frames then decode as with the BWE off. |
| The BWE switch | `set_osce_bwe` takes a `bool`. | `OPUS_SET_OSCE_BWE` takes an integer and refuses values other than 0 and 1 with `OPUS_BAD_ARG`. | A `bool` has no other values. |
| Expert frame duration of the multistream encoder | `set_expert_frame_duration` takes a `FrameDuration`, which holds only the ten values libopus defines. | `OPUS_SET_EXPERT_FRAME_DURATION` stores any number, and every later encode call fails when it is no duration. | A `FrameDuration` cannot hold another number. For the ten values both encoders take and refuse the same calls. |

### Neural network weights

| What | opus1 | libopus 1.6.1 | Why |
|---|---|---|---|
| Where `set_dnn_blob` exists | In every build with a neural network feature. A new decoder has the built-in weights; a blob replaces them. | `OPUS_SET_DNN_BLOB` exists only in a build with `USE_WEIGHTS_FILE`, which has no built-in weights. Otherwise it returns `OPUS_UNIMPLEMENTED`. | One build serves both uses. `DnnBlob::builtin()` brings back the built-in weights. |
| What a blob loads | Each group of networks on its own: deep PLC (PitchDNN, PLC and FARGAN), LACE, NoLACE, BBWENet. A group is replaced only when all its arrays load; the others keep their weights. | One request loads everything. A blob without BBWENet (such as the one of libopus's own `dump_weights_blob`) leaves LACE and NoLACE unloaded too in a build with the BWE, and with the BWE switched on libopus 1.6.1 then aborts at the first SILK frame at 48 kHz (a buffer overrun in `compute_generic_conv1d` that `_FORTIFY_SOURCE` stops). | The official blob works. |
| A blob that fails | Returns an error string and changes nothing. | Returns 1, which is not an Opus error code. The models loaded before the failing one stay replaced, and the failing layer keeps the arrays found before the failure. | No half-loaded network. |
| A malformed blob | Refused by `DnnBlob::from_bytes` or `from_static`, without reading out of bounds. | `parse_weights` fails, but every caller ignores that: the decoder request then crashes in `lpcnet_plc_load_model`. A blob without the index table of a sparse layer makes libopus read uninitialized memory. | Safety. |
| What the layer loader also refuses | Arrays that do not start on a 4-byte boundary; int8 weights equal to -128; a layer with neither int8 nor float weights; int8 layers whose output count is not a multiple of 8 or whose input count is not a multiple of 4 or above 2048; a diagonal array whose size is not 3 times the input count; negative counts or positions in a sparse index table; sizes whose byte count overflows. | Accepts all of these. Some of them make its kernels read or write out of bounds later; with -128 its kernel variants give different results. | Safety. The 1.6.1 weights have none of these. |
| `reset_state` | Keeps every loaded network. | `OPUS_RESET_STATE` reinitializes PitchDNN from the built-in arrays. In a build with `USE_WEIGHTS_FILE`, PitchDNN is then empty while the decoder counts deep PLC as loaded. | A reset should not lose weights. |

### DRED decoding

The DRED decoder (feature `dred`) gives libopus's results for every valid call. It differs where libopus has no rule or would crash:

| What | opus1 | libopus 1.6.1 | Why |
|---|---|---|---|
| `sampling_rate` of `parse` not positive | `"Invalid argument"`. | `opus_dred_parse` divides by it: 0 stops the program, a negative rate gives meaningless numbers. | No crash. |
| A DRED payload longer than 2,048 bytes | The range decoder reads the bytes after the first 2,048 as zeros. | Reads them. | opus1's range decoder holds at most 2,048 bytes. libopus's encoder writes at most 1,000 bytes of DRED, so only crafted packets reach this. |
| Output too short for `frame_size` in `dred_decode` | `"Output buffer too small for packet"`, before any change of the decoder. | Takes no buffer length: the caller must give room for `frame_size` samples per channel. | Memory safety. |
| A new `OpusDred` | Holds nothing: it is neither parsed nor processed, so `process` refuses it and decoding with it conceals as an empty packet. | `opus_dred_alloc` leaves the memory uninitialized. | Defined behavior. |
| Processing into another state | `process(&src, &mut dst)`; `process_in_place(&mut dred)` for `src == dst`. | One function, `opus_dred_process(dec, src, dst)`, with `src` and `dst` the same or not. | Rust's borrow rules. |
| Loading weights | `OpusDredDecoder::set_dnn_blob` in every build; a blob without the RDOVAE decoder gives an error and keeps the weights. | `OPUS_SET_DNN_BLOB` only with `USE_WEIGHTS_FILE`; a failure returns `OPUS_BAD_ARG` and can leave the model half replaced. | As for the other networks ([Neural network weights](#neural-network-weights)). |
| 24-bit output out of range | `dred_decode24` gives the nearest 32-bit value, and 0 for NaN, as libopus does on 64-bit ARM (`vcvtns_s32_f32`). | On x86 (`_mm_cvt_ss2si`) both give -2^31. | Concealed audio stays far inside the range, so this is not reached. |

### DRED encoding

The DRED encoder (feature `dred`) was compared with libopus 1.6.1 built with `--dred` without loop vectorization (`libopus-host-1.6.1-dred-novec`, a float build), on the speech fixture with pauses of digital silence and quiet noise (`tests/dnn_dred_encode_libopus.rs`). The same input and settings give:

| Packets | Equal to libopus | With DRED data |
|---|---|---|
| CELT-only, 48 kHz, mono and stereo, 2.5 to 20 ms, VBR and CBR, 24 to 128 kb/s, 5 to 50 % loss, with and without FEC, DRED of 10 to 104 frames, complexity 0 to 6 (80 streams) | 60,000 of 60,000 packets, byte for byte | 21,848 (packets of 2.5 and 5 ms at low bitrates have no room) |
| The same at complexity 7 to 10, with the speech and music analysis (5 streams) | 3,000 of 3,000 packets | 1,618 |
| CELT-only packets of 40 and 60 ms with CBR (4 streams) | 500 of 500 packets | 291 |
| CELT-only at 8, 12, 16 and 24 kHz (8 streams) | 2,400 of 2,400 packets | 2,010 |
| Hybrid, 48 kHz, `Audio`, VBR, complexity 0 to 6 (5 streams) | the DRED data of 2,100 of 2,100 packets | 1,875 |
| SILK-only, 8 to 48 kHz, 10 to 60 ms, `Audio`, VBR (6 streams) | the DRED data of 1,400 of 1,400 packets | 1,166 |
| Hybrid, `Voip` with VBR at complexity 5, and `Audio` with VBR at complexity 9 (one stream each) | the DRED data of 500 of 500 packets in each | 487 in each |
| Hybrid with CBR: `Audio` at complexity 5, `Voip` at complexity 9 | the DRED data of 208 and 207 of 500 packets | libopus 466 and 468, opus1 466 and 471 |
| 40 and 60 ms with VBR: Hybrid | the DRED data of 120 of 250 and 78 of 166 packets | libopus 198 and 137, opus1 243 and 161 |
| 40 and 60 ms with VBR: CELT-only | the DRED data of 146 of 250 and 97 of 166 packets | libopus 208 and 154, opus1 243 and 161 |
| Multistream (a stereo and a mono stream, CELT-only, 96 kb/s), DRED set on each stream | 200 of 200 packets | 198 in each stream |

The SILK layer of opus1 is the fixed-point one, so SILK-only and Hybrid packets differ from those of the float build in their main frames. The DRED data reads only the filtered input, the voice activity and the bitrate, so it can still be libopus's. It differs in these cases:

- **A constant bitrate with SILK.** The room for DRED data is what the main frame leaves of the packet. opus1's SILK codes other bytes, so the room differs, and the DRED data stops after another latent vector or is not written at all.
- **Packets of several frames with VBR.** opus1 leaves out libopus's limit of each frame to its share of the bitrate on purpose (see [Deliberate differences](#deliberate-differences)). The first frame then has more room for DRED data, and libopus cuts its data shorter or leaves it out more often.
- **`Voip` with SILK.** The high-pass filter before DRED follows the pitch analysis of SILK, and at 8 to 24 kHz SILK's voice activity detector reads the output of opus1's integer filter (see [Deliberate differences](#deliberate-differences)). Both can change the DRED data. In the stream above, at 48 kHz, nothing changed.
- **The speech and music analysis.** opus1's analysis follows the Homebrew build of libopus, which rounds some sums otherwise than the build without vector loops (see [What the figures were measured against](#what-the-figures-were-measured-against)). The activity it gives DRED could then differ. No packet of the tests changed.

The recovered audio was measured too. With bursts of 1 to 49 lost packets, each rebuilt from the DRED data of the packet after the burst, libopus's decoder and opus1's give the same samples, from opus1's packets and from libopus's. The energy of the rebuilt audio in each 10 ms is closer to that of a decode without loss than plain concealment: 5.57 against 7.40 dB on average (opus1's Hybrid packets, 32 kb/s), 6.29 against 26.64 dB (opus1's CELT-only stereo packets, 64 kb/s CBR), and 6.18 against 7.29 dB (libopus's Hybrid packets, 24 kb/s).

The time DRED takes was measured on the Mac (Apple M4 Pro) with the ignored test `encoder_cost_of_dred`: `Voip`, 32 kb/s, 20 % expected loss, complexity 9, the mode chosen by the encoder, 20 s of the speech above, the best of 5 runs. Per 20 ms frame:

| Input | opus1, DRED off | opus1, DRED of 100 frames | libopus `-O2`, off | libopus `-O2`, 100 frames | libopus novec, off | libopus novec, 100 frames |
|---|---|---|---|---|---|---|
| 48 kHz mono | 25.6 µs | 79.8 µs | 46.3 µs | 136.0 µs | 49.8 µs | 142.8 µs |
| 16 kHz mono | 18.3 µs | 64.8 µs | 36.9 µs | 114.8 µs | 39.2 µs | 122.1 µs |

So DRED adds about 54 µs per 20 ms frame to opus1's encoder at 48 kHz and 47 µs at 16 kHz, and about 90 µs and 78 µs to libopus's (`libopus-host-1.6.1-dred`, `-O2`). The opus1 and `-O2` columns come from one run; the novec columns from a run against `libopus-host-1.6.1-dred-novec`, in which opus1 measured 25.6, 79.6, 18.2 and 64.7 µs.

Where the DRED encoder itself differs from libopus:

| What | opus1 | libopus 1.6.1 | Why |
|---|---|---|---|
| The voice activity of latent vector 51 | Counts the 8 bytes after the activity memory as inactive. | `dred_voice_active` reads 8 bytes past `activity_mem` when it skips silence with a full buffer of latent vectors: the encoder fields `nonfinal_frame` and `rangeFinal`. | Memory safety. With latent vector 51 or 52 there is nothing left to code either way, so no packet changes. |
| A reset in a build with `USE_WEIGHTS_FILE` | `reset_state` keeps the loaded models. | `dred_encoder_reset` calls `lpcnet_encoder_init`, which clears PitchDNN and loads it again only from built-in weights. With a weights file PitchDNN is then empty while the encoder counts DRED as loaded. | A reset should not lose weights. Both turn DRED off. |
| Loading weights | `OpusEncoder::set_dnn_blob` in every build. The RDOVAE encoder and PitchDNN load together; a blob without one of them gives an error and keeps the old models. | `OPUS_SET_DNN_BLOB` only with `USE_WEIGHTS_FILE`. When PitchDNN fails, the RDOVAE encoder stays replaced and `OPUS_BAD_ARG` is returned. | As for the other networks ([Neural network weights](#neural-network-weights)). |
| One frame of a packet of several frames | At most 1,275 bytes, DRED data included. | The first frame may take more when the DRED bytes are added to its share. | Only at bitrates near the largest a frame holds. |
| The multistream encoder | Has no DRED setting; set it on each stream encoder (`stream_encoder_mut`). | Answers `OPUS_SET_DRED_DURATION` with `OPUS_UNIMPLEMENTED`; set it on each stream (`OPUS_MULTISTREAM_GET_ENCODER_STATE`). | The same. |

## Where the numbers differ

### CELT encoder

- **The CELT encoder alone** (`CeltEncoder`, without the speech and music analysis of the Opus layer): every packet is identical to libopus. `celtenc_gap_report` has 960 streams with 1,080,000 packets: mono and stereo, 16 and 48 kHz input, frames of 2.5 to 20 ms, 24 to 96 kb/s per channel, CBR and VBR, complexity 4, 5 and 9. `celtenc_rate_rules` adds VBR at bitrates that are not a whole number of bits per frame, bitrates above 750 kb/s per channel, silent frames, and the silence flag of a stereo input coded as one channel. Before this release round 91.2 % of the 1,080,000 packets were identical.
- **Stereo input whose two channels are scaled copies of each other** (one source panned to one side), and **signals at very low levels**: every packet is libopus's. The decaying output of the Opus layer's DC filter after a signal stops is both: its two channels decay at the same rate, at peak levels from about 1e-3 down to 6e-8. Before, a few packets per stream differed there, for example 6 of 600 in 48 kHz stereo at 510 kb/s, and 30 of 103,500 packets of panned tones from -134 to -20 dBFS at 128 to 750 kb/s per channel. Three causes, all fixed:
  - `intensity_stereo` (celt/bands.c, lines 379-403) left out libopus's `EPSILON` (1e-15) under the square root of the band energies and on the root, clamped the energies to 1e-10, and rounded each product. When the two channels are scaled copies, every band has the angle 0 and is coded from this mix.
  - The transient analysis divided its energies by 16. In the float build `PSHR32` does not shift, and `EPSILON` is added to the mean energy and to the envelope, so the scale matters when the energies are tiny. It also rounded products that clang fuses and used single precision where libopus computes in double precision (`mean*maxE*.5*len2` and the `tf_estimate` formula, whose constants are doubles).
  - `stereo_fade` and `gain_fade` of the Opus layer rounded the gain `w*g2 + (1-w)*g1` in two steps. clang rounds `(1-w)*g1` and fuses `w*g2` into the sum.

  Tests: `celtenc_stereo::a_source_panned_to_one_side_is_coded_as_libopus_codes_it`, `celtenc_analysis::the_decaying_output_of_the_dc_filter_is_coded_as_libopus_codes_it` (108,000 frames), `celtenc_bands_libopus::stereo_bands_that_are_scaled_copies_at_low_levels_are_coded_as_libopus_codes_them` and, through the Opus encoder, `core_lowdelay::the_decaying_output_of_the_dc_filter_is_coded_as_libopus_codes_it` (43,200 packets).
- **Digital silence after a signal, when a stereo encoder codes one channel**: the silence flag is libopus's. CELT sets a silence flag when the peak level of the frame is at most 2^-24. libopus measures that peak over `C*(N-overlap)` samples (celt_encoder.c, lines 1969-1971). `C` is the number of coded channels, but the input holds both input channels. So when a stereo encoder codes a mono stream, libopus measures only the first half of the frame, and opus1 now does the same. Before, opus1 measured the whole frame, and one or a few packets in 3 s of speech with silent stretches differed (8 kHz stereo, `RestrictedLowDelay`, 510 kb/s VBR into buffers of 10 to 100 bytes). Tests: `celtenc_rate_rules::silence_flag_of_stereo_input_coded_as_mono` and `core_long_frames::silence_flag_of_stereo_input_coded_as_mono_is_libopus`.
- **CELT-only streams of the Opus encoder, at every complexity:** with the speech and music analysis, the packets are libopus's. In the 51 CELT scenarios of `perf_harness modes` (8 to 48 kHz, 2.5 to 20 ms, complexity 0 to 10, CBR and VBR, mono and stereo up to 510 kb/s, `RestrictedLowDelay`), every packet is identical: 26,750 of 26,958 packets. The other 208 are the two streams of packets of 40 and 60 ms with VBR, where opus1 leaves out the limit of each frame on purpose (see [Deliberate differences](#deliberate-differences)); their SNR differs by -0.2 and +0.2 dB. Before the analysis was ported, 3,942 of the 26,958 packets were identical: at complexity 7 and above only 4 to 24 of 250 per stream. The tests `core_analysis_libopus` (948 streams, about 290,000 packets) and `celtenc_analysis_info_libopus` (2,328 streams, 464,700 packets) check this byte for byte.
- **Impulse at 48 kHz, mono, 64 kb/s CBR, c10:** 160 of 160 bytes equal libopus (7 before the analysis was ported). Test: `oracle_celt::impulse_48k_mono_64k_is_the_packet_of_libopus`.
- **Rounding on 64-bit ARM.** clang fuses some products and sums in libopus into one FMA. opus1 copies this in the pitch analysis of the pre-filter (`src/pitch.rs`, `src/celt_lpc.rs`), the comb filter, the prediction of the coarse energy, the transient analysis, `intensity_stereo` and the fades of the Opus layer, among others. A few such places are not copied in opus1. No case was found where they change a packet:
  - The fused rounding of the two-bin side-sign product is not ported. It matters only at the angles 0 and 16384, where no sign is coded (earlier).
  - Others, for example `importance` (0.5 + 13 * exp2): no case found where they change a packet (earlier).
  - The pre-filter compares the sum of the absolute sample values before and after filtering. opus1 adds them in four lanes, libopus one value after the other.

### SILK

- **Against floating-point libopus:** 44 of 12,290 SILK packets and 20 of 7,583 Hybrid packets are identical (CELT: 26,750 of 26,958). This is expected, because the two SILK encoders compute differently. SNR difference over the 39 SILK streams: mean -0.04 dB, worst -0.9 dB (`silk_wb_16k_mono_20k_vbr_c9_20ms_audio`), best +0.6 dB (`silk_mb_12k_stereo_24k_vbr_c9_20ms`). Over the 22 Hybrid streams: mean -0.03 dB, worst -0.4 dB. Over all 100 streams with an SNR: mean -0.02 dB.
- **Streams that mix SILK or Hybrid with CELT, at complexity 7 and above:** the speech and music analysis gives the decisions of libopus, but SILK's feedback (its voice activity, its signal type, the bits it used) reaches later decisions. So such a stream follows libopus until its first SILK packet, and after that it can take other decisions. In 180 SILK and Hybrid streams at 16 to 48 kHz (`core_analysis_libopus`, report test), the first byte of 19,500 of 19,500 packets was libopus's, and the SNR was 12.36 dB against libopus's 12.33 dB (11.72 dB without the analysis). In the 8 mode-switching streams of `perf_harness modes`, every stream has libopus's mix of modes, bandwidths and channel counts.
- **Against fixed-point libopus** (packets compared byte for byte):
  - The reference is the fixed-point build of the 1.6.1 release. opus1's Burg analysis used to add 1 to the energy of the frame (`C0_64 = ... + 1`, "Make sure the noise corresponds to at least a unit pulse"), as libopus's main branch does in silk/fixed/burg_modified_FIX.c, line 72. The 1.6.1 release has no such line, and opus1 no longer has it. On the 49 SILK streams of the golden file (5 s each, 16-bit input, the settings of each stream) opus1 now equals the 1.6.1 build in every packet in 47 streams (11,574 of 12,290 packets). With the `+ 1` it equalled the main-branch build in 46. The other two:
    - `silk_wb_16k_mono_20k_vbr_c9_20ms_audio`: with the `Audio` application opus1's SILK reads the frame that the float `dc_reject` filtered, as in the float build. The fixed-point build gives SILK the output of its integer `dc_reject`. 0 of 250 packets equal.
    - `silk_wb_16k_mono_24k_vbr_c9_10ms_fec10`: the stream changes SILK's internal rate from 12 to 16 kHz at packet 18. Packet 17 carries the redundant CELT frame of the switch, and the prefill of packet 18 reads the float delay buffer (both below). The packets differ from packet 17 on (34 of 500 equal).
  - 8 s of speech, Voip, SILK-only wideband, c9, 16-bit input, mono and stereo at 12, 24 and 32 kb/s, CBR and VBR: 12 of 12 streams equal the 1.6.1 build in 400 of 400 packets. With the `+ 1`, against the main-branch build, mono 32 kb/s CBR differed from packet 294 on and stereo 32 kb/s CBR from packet 163 on.
  - NLSF weights (fixed in October 2026). libopus's `silk_NLSF_VQ_weights_laroia` takes the weight of the last coefficient from its distance to 1, `(1 << 15) - NLSF[D-1]`. opus1 took the distance to 32767, so for some vectors that weight was 1 too large: 121 instead of 120 when the last NLSF is 30,619. The NLSF quantizer weighs its choices by these weights, and rarely chose other indices than libopus. Stereo speech at 32 kb/s with VBR differed from packet 274 on (8 s, 20 ms frames), and the golden stream `silk_wb_16k_stereo_32k_vbr_c9_10ms` from packet 322 on, both without a change of mode or rate. Both now equal the 1.6.1 build in every packet (400 and 500). Tests: `silk::nlsf::laroia_tests` (the weights that the fixed-point build computed) and `core_fixed_point::stereo_speech_at_32_kbps_codes_the_packets_of_libopus`.
  - Frames of digital silence: libopus tells SILK that such a frame has no voice activity (`is_digital_silence`, opus_encoder.c, lines 1246 and 1909-1914), and SILK then clears its voice activity flag. opus1 does the same. Before, the mono stream of the test `core_fixed_point::digital_silence_tells_silk_there_is_no_voice_activity_like_libopus` differed from packet 15 on, and against the float build 5 to 11 silent frames per stream had the flag set where libopus had it clear (`core_encoder::silk_voice_activity_flag_of_digital_silence_is_libopus`).
  - Packets of 80 to 120 ms with CBR, Voip, c9 (8 and 16 kHz, mono and stereo, 16 and 64 kb/s mono, 24 and 96 kb/s stereo, with and without FEC at 10 % loss, 3.6 s of speech, 48 streams): 40 of the 42 SILK streams equal the fixed-point build in every packet. The other six streams are CELT-only (stereo at 96 kb/s without FEC), where the fixed-point build's CELT encoder is not the float one. The two that differ (16 kHz stereo, 24 kb/s, FEC, 80 and 120 ms) do so from a change of SILK's internal rate on: the redundant frame of the switch is CELT, which is fixed-point in that build, and the prefill reads the float delay buffer (next item). So with CBR the distance to the float build of libopus at 80 to 120 ms comes from the two SILK encoders, not from the long frames.
  - The VoIP SILK prefill reads the float delay buffer. The fixed-point build reads its 16-bit `hp_cutoff` output. Stereo VoIP after a rate switch: 312 of 400 packets equal. Mono packets are equal again after 1 to 5 packets (earlier). In `silk_wb_16k_mono_24k_vbr_c9_10ms_fec10` (10 ms, FEC) they are not equal again within the 500 packets.
  - `silk_encode_prefill` has no `to_mono` argument. libopus passes `toMono` to its prefill. No failing case was found (earlier).
- **Quantization gains of frames with a subframe near digital silence** (a deliberate difference since October 2026).
  - The cause, in libopus's fixed-point encoder. `silk_find_pred_coefs_FIX` multiplies each subframe of 5 ms by the smallest noise shaping gain of the frame divided by the subframe's own gain, times 1/4, and rounds the result to 16 bits. `silk_residual_energy_FIX` computes the residual energy of each subframe from that signal and multiplies it back by the squared inverse factor. When one subframe is near digital silence, the factor of the other subframes is small, and the rounding error is about as large as the scaled signal. Their energies then come out several times too large. `silk_process_gains_FIX` sets each gain from the noise shaping gain and the residual energy, which weighs more the lower the target SNR is. At a low SNR the gain comes out too large. The decoder plays the subframe's few pulses and the quantizer's offset at that gain. The float build scales in float and has no such error.
  - Where it shows: a stereo stream whose bitrate drops below the stereo threshold. In the packet of the drop the stereo width falls to 0. SILK's side channel then fades out over the first 8 ms and holds zeros after that, and its target SNR is 0 dB. The decoder adds the side channel to the left output and subtracts it from the right one, so both get the burst.
  - Example: the TSP items `FD22_01` (left) and `FE25_04` (right), `Voip`, 48 kHz, complexity 5, the bitrate schedule of the mode-switching streams of `perf_harness`, packet 100, where the bitrate drops from 32 to 8 kb/s. The side channel's input has the mean squares 45, 21, 1 and 0 in its four subframes, in opus1 and in the float build alike. The values of the C builds were printed by copies of `silk/fixed/encode_frame_FIX.c` and `silk/float/encode_frame_FLP.c` with print statements added. The fixed-point copy gives the packets of the unchanged build. The fixed-point build reads its integer high-pass filter here, so its input differs a little (mean squares 82, 37, 2 and 0).

    | Side channel, packet 100 | Residual energies | Gains | Gain indices |
    |---|---|---|---|
    | Float build of libopus 1.6.1 | 2,840, 1,230, 88, 0 | 75, 64, 34, 18 | 26, 3, 0, 0 |
    | Fixed-point build of libopus 1.6.1 | 45,253, 46,464, 6,137, 0 | 227, 227, 121, 64 | 33, 4, 0, 0 |
    | opus1 before | 133,225, 72,900, 2,904, 0 | 428, 364, 194, 104 | 37, 3, 0, 0 |
    | opus1 now: energies of the unscaled signal | 2,874, 1,366, 79, 0 | 75, 64, 34, 18 | 26, 3, 0, 0 |

    Energy below 200 Hz in the 10 ms frames 199 and 200, left and right output, in dB: input 56.7 and 61.4, 69.1 and 59.1. Float build 67.5 and 64.8, 65.6 and 66.5. Fixed-point build 79.3 and 84.8, 82.1 and 84.8. opus1 before 84.3 and 81.3, 83.3 and 81.4. opus1 now 69.1 and 67.5, 66.4 and 68.2.
  - The change. opus1 also computes the residual energies from the unscaled signal (`silk_res_nrg_unscaled_fix` in `src/silk/control_fixed.rs`). When libopus's energy gives a subframe more than twice the gain that the unscaled energy gives, the subframe takes the smaller gain (`silk_process_gains_fix`). The unscaled energies are only computed in frames where libopus's energies give some subframe more than 1.5 times its noise shaping gain. The encoding time changed by -0.1 % to +0.3 % (four SILK streams of 44 s, 16 to 48 kHz, mono and stereo, complexity 5 to 10, with and without the change in turns).
  - Packets against the fixed-point build: the 137 streams of the golden file did not change, and the tests that compare packets with the fixed-point build pass. Mono speech at a constant bitrate gives the same packets with and without the change in most streams: for example the four TSP excerpts of `fixtures/tsp/` at complexity 9, at 16 kHz and 20 kb/s, 24 kHz and 24 kb/s, and 48 kHz and 32 kb/s. The test control `internals::silk_input::set_fixed_point_res_nrg` turns the change off.
  - Frames with a burst (10 ms frames whose energy below 200 Hz lies more than 10 dB above both the input's and that of the float build's output), before and now:
    - Stereo, the bitrate schedule of `perf_harness` (20 pairs of TSP items, 8,884 frames per stream kind): at 48 kHz 8 and 5 before (complexity 5 and 9), 2 and 1 now; at 24 kHz 7 and 7 before, 3 and 2 now; at 16 kHz 7 and 8 before, 0 and 4 now. In all, 42 before and 12 now. The float build: 0 to 2 per stream kind.
    - The same schedule on four other samples of 20 pairs (the item list started 1, 2, 3 and 4 items later): 16 before and 13 now, 19 and 17, 35 and 35, 5 and 6. In all, 75 before and 71 now. Most bursts in these samples have other causes (see the remaining bursts below). The change helps mainly where a subframe is close to digital silence, which the first sample has more often.
    - Mono, the same schedule (40 TSP items, 8 to 48 kHz, complexity 5 and 9, 9,575 frames per stream kind): 5 before, 2 now.
    - Forced SILK-only and CELT-only packets taking turns (see [Paired comparison](#paired-comparison-with-the-evaluation-harness)): 4 before, 5 now. The added frame lies at the start of a stereo stream at 24 kHz: 65.5 dB, against 64.3 dB before, where the threshold is 65.1 dB.
    - The four TSP excerpts of `fixtures/tsp/` in 12 stereo pairs, 32 and 8 kb/s taking turns every 200 ms (6,000 frames per stream kind; test `core_silk_residual_energy`): at 48 kHz 15 and 10 before (complexity 5 and 9), 5 and 2 now; at 24 kHz 18 and 13 before, 11 and 3 now; at 16 kHz 18 and 15 before, 6 and 7 now.
  - The paired comparison (`examples/eval`, before and now): see [Paired comparison](#paired-comparison-with-the-evaluation-harness).
  - What remains:
    - Subframes where libopus's energy gives a gain between 1.5 and 2 times the gain from the unscaled energy keep libopus's gain. Example: the TSP items `FL68_08` and `FL72_01`, 24 kHz, complexity 9, packet 100 of the same schedule. The side channel's gain index is 45 in opus1 and in the fixed-point build, and 41 in the float build. With a limit of 1.5 instead of 2, the 42 bursts of the stereo schedule fall to 8 instead of 12, but 3 of the 137 golden streams change.
    - The LPC analysis (`silk_find_LPC_FIX`) reads the same rounded signal. Tried on top of the change and not kept: in `silk_find_pred_coefs_fix`, take the larger of the frame's smallest gain and a quarter of its largest gain in place of the smallest gain, and limit each factor to 1/4. The 42 bursts then fell to 6. But 11 of the 137 golden streams changed, 1,331 of the 1,332 jobs of the set `tsp_stereo` changed, 5 of the 10 tests of `core_fixed_point` failed, and the error of `tsp_stereo` moved by -1.4 % to +1.0 % per condition, each change within its 95 % interval.
- **DTX of SILK with the `Voip` application** (`set_dtx`, below complexity 7 or at 8 and 12 kHz, where SILK's voice activity detector decides): SILK reads the output of the integer high-pass filter (see [Deliberate differences](#deliberate-differences)), the float build of libopus the float filter output. So the detector decides otherwise in a few frames: 16 of 9,202 frames at 8 and 12 kHz in the test matrix of `core_dtx_libopus`, which allows at most 1 %. Against the fixed-point build, whose SILK reads the same samples, 1,884 of 1,920 `Voip` streams with DTX (8 to 48 kHz, 10 to 60 ms, complexity 0 to 9) were byte for byte the same, including `in_dtx`. The other 36 differ at the same packet with DTX off. With the analysis (complexity 7 to 10 at 16 to 48 kHz) and with `Audio`, the DTX frames and the CBR sizes are libopus's.
- **Hard cases** (earlier):
  - `Audio` application, FEC with 20 % expected loss, CBR, 40 and 60 ms stereo: a SILK burst 12 dB above the input at packets 68 and 69. At 48 kHz with 60 ms frames the SNR is 1.23 dB against libopus's 3.36 dB. With 16-bit input the burst is gone (2.59 against 2.98 dB). The cause is SILK's rate control.
  - 24 kHz, VoIP, 40 ms, CBR, float input, after a step from 64 to 10 kb/s: 5.14 against 5.73 dB (two packets). With 16-bit input it equals the fixed-point build.
  - Low bitrates, largest excess of a decoded packet over the input, opus1 against libopus: 8.9 against 3.0 dB at 8 kb/s with 10 ms frames, 4.7 against 1.6 dB at 6 kb/s with 20 ms frames.
  - Input at -40 dB, wideband, 24 kb/s, 60 ms: 11.1 against 16.0 dB. This was measured before later changes to the pitch analysis.
  - Stereo tones at 12 kHz with a constant bitrate (found by the fuzz target `fuzz_channel_order`): the stereo image can collapse. The signal is a 300 Hz tone on the left and an 1800 Hz tone on the right, each at amplitude 0.3 (-16.5 dB), coded as SILK in 20 ms frames for 0.5 s. The tone levels are measured over the last 0.3 s, and mid and side over the last 0.25 s.
    - 160 kb/s, complexity 9 or 10: in opus1 the right output holds the left tone at -9.8 dB and its own tone at -25.3 dB. In the fixed-point build of libopus the left output holds the right tone at -9.1 dB and its own tone at -17.8 dB. The float build keeps every foreign tone at least 78 dB below the own one. Mid and side SNR: 19.7 and -10.8 dB in opus1, 8.5 and -14.7 dB in the fixed-point build, 42.8 and 42.9 dB in the float build.
    - 33 kb/s, complexity 10, own tone minus foreign tone in the left and the right output: -1.0 and 2.2 dB in opus1, 1.7 and 46.4 dB in the fixed-point build, 26.6 and 28.9 dB in the float build.
    - 25.5 kb/s, complexity 2, the same measure: -4.1 and 2.2 dB in opus1, -3.6 and 8.6 dB in the fixed-point build, 33.0 and 42.4 dB in the float build.
    - With a variable bitrate the image stays. At 160 kb/s the own tone is at least 52 dB above the foreign one in opus1, 50 dB in the fixed-point build and 67 dB in the float build.
    - The cause lies in the rate control of the fixed-point SILK encoder, which opus1 ports. At 33 and 25.5 kb/s opus1 keeps the channels apart less well than the fixed-point build.
- **Bandwidth switches** (test `core_redundancy::silk_bandwidth_switches_match_libopus`): SILK changes its internal rate on libopus's packets. The deficit in the worst window around each switch:
  - 48 kHz: mean -1.5 dB with CBR (worst switch 3.4 dB) and +1.8 dB with VBR (worst switch 5.5 dB).
  - 16 kHz: mean -1.4 dB with CBR (worst switch 1.9 dB) and +0.3 dB with VBR (worst switch 4.4 dB).
  - The switches fall into pauses of digital silence, so these windows compare how much each encoder leaves in the decoded silence (-88 to -111 dB against a reference energy of 1e-12). With the `+ 1` of the main branch in the Burg analysis the means were -0.9 and +2.3 dB at 48 kHz and -1.0 and +2.4 dB at 16 kHz. Before SILK was told about digital silence, the switches came one packet after libopus's, the windows were others, and the means were -1.0 and -0.7 dB at 48 kHz and +0.8 and -1.2 dB at 16 kHz.
  - The remaining gap is the difference between the fixed-point and the float SILK encoder. The fixed-point build of 1.6.1, given the same audio as 16-bit samples, is itself below the float build in these windows: mean 2.0 dB at 16 kHz with VBR (worst switch 7.0 dB), 0.2 dB with CBR, 0.4 and -0.7 dB at 48 kHz. The test uses the `Audio` application, where opus1's SILK reads the float `dc_reject` output, so its packets are not that build's (143 to 176 of 1500 equal). With `Voip` and 16-bit input, where they are (at 48 kHz with the integer filter, which opus1 used there until October 2026), 23 of the 24 worst windows of opus1 equal that build's to 0.1 dB. The other one (48 kHz, CBR) is 1.2 dB off, because the redundant CELT frames of the switch differ.

### Hybrid and mode switching

- **Fullband Hybrid at 48 kHz**, SNR against floating-point libopus, from `perf_harness modes`, with the delay in samples at which it was measured. The column "integer filter" is opus1 with SILK reading the integer `hp_cutoff`, as until October 2026:

  | Stream | opus1 | integer filter | libopus |
  |---|---:|---:|---:|
  | mono 32 kb/s, CBR, c9 | 8.6 dB @307 | 8.4 dB @308 | 8.7 dB @307 |
  | mono 32 kb/s, CBR, c5 | 8.5 dB @308 | 8.5 dB @308 | 8.7 dB @307 |
  | mono 32 kb/s, CBR, c0 | 8.5 dB @308 | 8.3 dB @308 | 8.5 dB @308 |
  | stereo 40 kb/s, CBR, c9 | 8.3 dB @307 | 8.3 dB @307 | 8.4 dB @307 |
  | stereo 40 kb/s, VBR, c9 | 8.2 dB @308 | 8.3 dB @308 | 8.6 dB @307 |
  | mono 32 kb/s, VBR, c9 | 8.4 dB @308 | 8.7 dB @307 | 8.7 dB @307 |

  Two causes explain these gaps:
  - The measurement. `perf_harness modes` takes the delay at which the decoded audio correlates best with the input in the loudest 250 ms. At 308 samples the SNR is about 0.2 dB lower than at 307. With the integer filter, the differences at the same delay of 307 samples were -0.03 dB (mono 32 kb/s CBR c5), +0.04 dB (c9), -0.13 dB (c0), -0.08 dB (stereo 40 kb/s CBR c9), -0.19 dB (stereo 40 kb/s VBR c9) and +0.01 dB (mono 32 kb/s VBR c9) (earlier). Over the 222 TSP items of the paired comparison, which measures the whole item, opus1's SNR changed by at most 0.042 dB per condition when the filter changed (see [Paired comparison](#paired-comparison-with-the-evaluation-harness)).
  - The SILK encoder. The test signal has almost no energy above 7.6 kHz, so the SNR measures the band that SILK codes. With the integer filter, the fixed-point build of libopus, whose SILK encoder opus1 ports, gave the same SNR as opus1 to 0.01 dB in all six streams (Voip, 16-bit input, measured at 307 samples).
- **40 and 60 ms Hybrid VBR uses more bits** than libopus with the same forced settings, because of the missing per-frame limit (see [Deliberate differences](#deliberate-differences)):

  | Stream | opus1 | libopus | More |
  |---|---:|---:|---:|
  | fullband 48 kHz mono 32 kb/s, 40 ms | 32.0 kb/s | 30.2 kb/s | +6.0 % |
  | super-wideband 24 kHz mono 24 kb/s, 60 ms | 24.4 kb/s | 22.5 kb/s | +8.4 % |
  | super-wideband stereo 36 kb/s, 40 ms | 34.7 kb/s | 31.9 kb/s | +8.8 % |

- **80 to 120 ms with VBR and in-band FEC:** the same missing limit keeps `opus1` far above libopus, whose parts are cut to their share of the bitrate. Measured with 2.4 s of speech, complexity 9, FEC with 10 % expected loss, decoded by libopus:

  | Stream | opus1 | libopus | libopus at 60 ms |
  |---|---:|---:|---:|
  | 8 kHz mono, `Audio`, 64 kb/s, 80 ms | 30.2 dB | 10.2 dB | 29.8 dB |
  | 8 kHz mono, `Audio`, 64 kb/s, 120 ms | 29.6 dB | 8.8 dB | 29.8 dB |
  | 16 kHz mono, `Audio`, 64 kb/s, 120 ms | 23.2 dB | 7.2 dB | 23.3 dB |

  The limit is the only cause. With it applied under VBR too (a trial, not kept), opus1's SILK-only packets of 80 to 120 ms equal those of the fixed-point build in 96 of 96 VBR and constrained VBR streams (8 and 16 kHz, mono and stereo, two bitrates, with and without FEC, 3.6 s of speech). Without it, 6 of 96 streams are equal: those where no part needs more than its share. The rule is libopus's, deterministic and a one-line change in `encode_multiframe` (apply `curr_max.min(nominal_part)` whatever the bitrate mode). It stays out because of what it costs with FEC (see [Deliberate differences](#deliberate-differences)).

  Over the 864 streams of `tests/core_long_frames.rs` (8 to 48 kHz, mono and stereo, the three applications, VBR, constrained VBR and CBR, with and without FEC, 80 to 120 ms, 3.6 s each), `opus1` is on average 0.8 dB above libopus. 21 of these streams are more than 0.5 dB below libopus, at worst 1.5 dB (51 before the analysis was ported); the same settings in frames of 20 to 60 ms are about as far below. They are CBR streams with FEC, where SILK's rate control differs. With CBR every packet has libopus's size, and CELT-only packets at complexity 0 and 5 are libopus's byte for byte.
- **Hybrid after SILK-only packets:** the high band of the first three Hybrid packets is at most 0.7 dB from libopus (test `core_redundancy::hybrid_after_silk_only_starts_its_celt_layer_afresh`).
- **Mode switches**, forced or driven by the bitrate (the other `core_redundancy` tests): the worst window around a switch is up to 4.2 dB below libopus (48 kHz stereo, 20 ms, CBR). The mean deficit per stream is -1.4 to +1.4 dB.

### Paired comparison with the evaluation harness

The harness `examples/eval` (see [Testing](09-testing.md#the-paired-comparison-harness)) codes the same items with both encoders, lets libopus decode both, and scores each output against the original with `opus_compare`. `ratio_err` is opus1's error divided by libopus's: below 1, opus1's output is closer to the original. The `baseline` set ran on 2026-10-18 with the codec of `d541821` (233 items, 9,056 jobs). The `tsp` set ran again after SILK's input with `Voip` at 48 kHz changed to the float filter (October 2026; "now" below). The figures below are geometric means over the items.

- **Speech** (222 items of the TSP database, `Voip`, mono, 48 kHz, variable bitrate, complexity 5, 20 ms frames). Both encoders code the same mode and bandwidth in every packet. The 95 % intervals are about ±5 %.

  | Bitrate (kb/s) | 6 | 9 | 12 | 16 | 24 | 32 | 48 | 64 | 96 | 128 |
  |---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
  | Mode and bandwidth | SILK NB | SILK NB | SILK WB | Hybrid FB | Hybrid FB | Hybrid FB | Hybrid FB | Hybrid FB | CELT FB | CELT FB |
  | `ratio_err` at `d541821` | 1.204 | 1.375 | 1.256 | 1.373 | 1.620 | 1.937 | 2.656 | 3.279 | 1.000 | 1.000 |
  | `ratio_err` now | 1.085 | 0.992 | 1.013 | 0.989 | 0.982 | 1.003 | 0.982 | 0.986 | 1.000 | 1.000 |

  (NB, WB and FB stand for narrowband, wideband and fullband.) At 96 and 128 kb/s the packets are identical. At complexity 10 the speech and music analysis that opus1 lacks changes the picture above 48 kb/s: at 64 kb/s libopus codes 9 % of the packets as Hybrid and opus1 none, and the ratio is 1.021; at 96 and 128 kb/s it is 0.989.
- **Low-frequency energy in quiet frames at 48 kHz (the cause of the high ratios at `d541821`).** With `d541821`, from 24 to 64 kb/s opus1's error stayed at 1.34 to 1.49, while libopus's fell from 0.92 to 0.41. In many quiet frames near speech, opus1's output held energy in the bins of 0 to 200 Hz that lay more than 10 dB above both the original and libopus's output.
  - The cause is the integer high-pass filter `hp_cutoff` of libopus's fixed-point build, which then fed opus1's SILK with `Voip` at every rate. The filter (`silk_biquad_alt_stride1`) keeps its state in Q12. Its products are truncated toward minus infinity (`silk_SMLAWB`), and its gain at 0 Hz is large: 15,690 at 48 kHz with the lowest cutoff, 60 Hz (3,926 at 24 kHz, 1,750 at 16 kHz). The truncation errors add up to a constant at the output. On `CB01_01` with the cutoff held at 60 Hz, the integer filter's output lies on average 8.7 below the float filter's output (in 16-bit units) at 48 kHz, 1.8 below at 24 kHz and 0.5 below at 16 kHz. SILK codes that constant. In quiet frames it is the strongest part of the low band.
  - The fixed-point build of libopus 1.6.1 shows the same frames. On `CB01_01` with `Voip` at 48 kHz, SILK wideband, 12 kb/s, complexity 5, its packets were opus1's in 180 of 180 packets, and 134 of the 355 frames of 10 ms held the energy in both outputs (the energy is that of the bins 0 to 4 of a discrete Fourier transform over a 20 ms Hann window). The float build of libopus shows none.
  - opus1's SILK now reads the float filter's output with `Voip` at 48 kHz, as the float build does. Frames with the energy, over the 222 TSP items of the `tsp` set (53,412 frames of 10 ms), `Voip` at 48 kHz, complexity 5 unless noted, before and now: SILK wideband at 12 kb/s 4,064 (7.61 %) and 6; fullband Hybrid at 32 kb/s 4,631 (8.67 %) and 3; automatic mode at 32 kb/s and complexity 10 4,919 (9.21 %) and 2; automatic mode at 24 kb/s with 60 ms frames 4,472 (8.37 %) and 2. libopus: 0 to 5. At 24 kHz (24 kb/s) 66 frames (0.12 %) and at 16 kHz (12 kb/s) 9 frames (0.02 %) have it, with the integer filter.
  - The `tsp` set before (at `c90ad3f`) and now (40 conditions): in each of the 30 conditions that code SILK or Hybrid, opus1's error fell, by a geometric mean of 9 % to 70 % per condition, `moc_d` rose by 0.003 to 0.065 (its 95 % interval excludes 0 in 29 of the 30), and the bytes ratio fell by 0.0004 to 0.015. opus1's signal-to-noise ratio changed by -0.023 to +0.042 dB per condition, +0.001 dB on average over the 30. In 2 of them (-0.012 and -0.016 dB) the 95 % interval of the change lies below 0, about as many as chance gives at that level. The CELT conditions did not change.
  - The `quick` set: the 7 conditions at 48 kHz in `Voip` changed, the other 23 did not. Their `ratio_err` fell from 1.55 to 3.07 to 0.96 to 1.23, and the bytes ratio fell in all 7. `moc_d` rose in the 6 mono conditions (MOC is not computed for stereo). The mean SNR of their 6 speech items changed by -0.10 to +0.11 dB. Most single items moved toward libopus's SNR. The three gate warnings of the run before (`silk_low` `max_err`, `speech_mid` `bytes` and `geomean_err` at 20 kb/s) are gone.
  - At 16 and 24 kHz the integer filter stays. The same change measured there (set `tsp_rates`, 32 conditions per rate) moved the geometric mean of `ratio_err` by -4.1 % to +4.7 % per condition at 16 kHz (17 conditions better, 13 worse, 2 unchanged, on average -0.04 %), and by -8.9 % to +3.9 % at 24 kHz (24 better, 6 worse, 2 unchanged). The 95 % intervals are about ±4 to 5 %. Some conditions got worse, so opus1 keeps the fixed-point build's packets at these rates.
  - At 8 to 24 kHz the integer filter runs on every frame, CELT-only frames included, as in the fixed-point build (since October 2026). Before, it ran only on frames that SILK coded. The first SILK frame after CELT-only frames then started from the filter state of the last SILK frame, and SILK coded that step as energy below 200 Hz. Measured on 40 TSP items (9,575 frames of 10 ms), `Voip`, mono, frames with the energy before and now:
    - The bitrate schedule of the `perf_harness` switch streams (8 to 128 kb/s, one step every 200 ms), complexity 5 and 9: 33 to 37 frames at each of 8, 12, 16 and 24 kHz before, 0 to 3 now. libopus: 0 or 1.
    - Forced SILK-only and CELT-only packets taking turns every 7 packets, at 8, 12, 16 and 24 kHz: 83 to 96 frames before, 0 or 1 now. Every 25 packets: 31 (24 kHz) and 32 (16 kHz) frames before, 1 and 0 now.
    - In stereo (20 pairs of items) the same two kinds of streams at 16 and 24 kHz had 35 to 63 such frames before and have 0 to 11 now. The bitrate schedule left 7 or 8 such frames per stream kind in stereo at 16 and 24 kHz, and 8 and 5 at 48 kHz, where SILK reads the float filter. These came from SILK's quantization gains, not from the filter, and 0 to 4 remain (see [SILK](#silk)).
    - Streams without CELT-only frames do not change: the 13,824 jobs of the `tsp_rates` set, run on 216 TSP items that are not in the `tsp` set, were identical. In the streams above, opus1's mean SNR changed by -0.03 to +0.04 dB per kind of stream, and the ratio of its bytes to libopus's by 0 to +0.0014.
  - Test: `core_voip_input_filter`. The test control `internals::silk_input::set_integer_hp_at_48k` brings back the integer filter at 48 kHz, for tests that compare packets with the fixed-point build.
- **SILK's quantization gains from the unscaled signal** (October 2026; see [SILK](#silk)). Before is the code of `a8c9916`. The figures are changes of the geometric mean of opus1's error per condition, with 95 % intervals.
  - The set `tsp_stereo` (111 stereo pairs of the `tsp` items, 12 conditions, see [Testing](09-testing.md#the-paired-comparison-harness)): 578 of 1,332 jobs changed. With the bitrate schedule (6 conditions) the error fell by 0.2 % to 1.25 % (intervals ±0.3 % to ±1.5 %). At a constant 24 and 32 kb/s (6 conditions) it changed by -0.5 % to +0.5 % (intervals up to ±1.1 %). opus1's SNR changed by -0.001 to +0.004 dB per condition, and in no condition does its 95 % interval lie below 0. The bytes ratio rose by at most 0.0006.
  - The set `tsp` (40 conditions): 698 of 8,880 jobs changed. At 6 kb/s the error fell by 0.2 % to 0.6 % (4 conditions). At 9 kb/s with complexity 5 and 60 ms frames it rose by 0.16 % (interval ±0.3 %). In the other 35 conditions it changed by less than 0.005 %. In no condition does the 95 % interval of the SNR change lie below 0.
  - The set `quick`: 3 of 239 jobs changed (SILK at 8 and 16 kHz, 8 and 12 kb/s). Each took 1 to 4 bytes more, and its error stayed the same in the first 10 digits.
- **Music** (11 synthetic stereo items, `Audio`, 48 kHz, 64 to 160 kb/s): at complexity 5 with 20 ms frames the packets are identical. At complexity 10, and with 60 ms frames, the ratio is 0.69 to 1.06.
- **Bytes**, opus1's divided by libopus's, now: with 20 ms frames, 1.004 to 1.018 for SILK speech and 1.009 to 1.013 for Hybrid speech. With 60 ms frames, 1.008 to 1.026 for SILK speech, 1.059 to 1.091 for Hybrid speech and 1.012 to 1.018 for CELT speech. Hybrid and CELT at 60 ms take more because opus1 leaves out the per-frame limit on purpose. At `d541821` the same five ranges were 1.006 to 1.024, 1.009 to 1.014, 1.012 to 1.033, 1.059 to 1.093 and 1.012 to 1.018.
- **MOC:** the value D of the OSCE draft's relative test, with libopus as the reference decoder, now has a mean of +0.0053 to +0.0085 for Hybrid speech with 20 ms frames at complexity 5, +0.038 to +0.065 for Hybrid speech with 60 ms frames, and +0.0030 to +0.0103 for SILK speech. At `d541821` it was -0.0075 (16 kb/s) to -0.0589 (64 kb/s) for Hybrid speech with 20 ms frames at complexity 5, below the draft's limit of -0.025 for the mean from 32 kb/s on (-0.0254).

### Multistream

- **Surround masking:** the band levels are taken with libopus's approximation of the logarithm (`celt_log2`), as libopus does. 98.7 % of the mask values are the bits of libopus's `surround_analysis` (`multistream_surround_c_direct`, 1,247,400 values; 72.8 % with the exact logarithm used before). The others differ by at most 3.8e-6, from the rounding of the MDCT.
- **Bitrate at 16 kHz and for VoIP:** in the 25 cases of `multistream_enc_signals::average_rate_at_16_khz_and_for_voip_is_within_the_design_limits`, opus1 codes -12.8 % to +4.1 % around the requested bitrate, libopus +0.3 % to +8.3 %. Before the analysis was ported, opus1 used 65 % to 80 % of it in four cases.
- **Stereo Hybrid streams** in the matrix of `tests/multistream_enc_validity.rs` (48 and 24 kHz, VBR and CBR), counted in packets that contain a stereo Hybrid stream: opus1 now has as many as libopus. Left to themselves, 239 (libopus: 128 at 8 kb/s and 111 at 24 kb/s per channel); with two channels forced, 421 (libopus: 310 and 111). Before the speech and music analysis was ported, opus1 had 0 and 220: it coded the streams of 24 kb/s per channel as CELT.

### Decoder

On 64-bit ARM the decoder gives the samples of libopus, in float (`decode`) and in 16 bits (`decode_i16`):

- **`perf_harness modes`:** all 137 streams decode to libopus's samples: 51 CELT, 29 Hybrid and 49 SILK streams, and the 8 streams that switch modes. Earlier in this release, only 8 of the CELT streams were identical and the others agreed to 136 to 146 dB. The causes were four filter taps of the post-filter written with too few digits, the rounding of the post-filter, and the rounding of the coarse energy prediction in frames shorter than 20 ms. After that fix, the 8 switching streams and two FEC streams (`silk_wb_16k_mono_24k_vbr_c9_10ms_fec10`, and `hybrid_swb_24k_mono_24k_cbr_c9_20ms_fec10`, which libopus codes as SILK) still agreed only to 172 to 192 dB, because of the rounding of `smooth_fade` (see below). The packets of opus1's encoder that switch modes decode identically too (`core_dtx_libopus`, `multistream_dtx_libopus`).
- **A wider search** (`tests/core_decode_search.rs`, ignored tests that take about 4 minutes) compared every decode call, sample by sample, in float and in 16 bits, and the final range after every packet:
  - 228 streams coded by libopus and by opus1: 8 s of speech from the TSP speech database (see [Testing](09-testing.md#corpora)) mixed with tones, near silence and loud passages; 8, 12, 16, 24 and 48 kHz; mono and stereo; the applications `Voip`, `Audio` and `RestrictedLowDelay`; a bitrate that changes every 10 packets, so that the mode switches; frames of 2.5 to 120 ms; FEC; DTX (discontinuous transmission, libopus streams only); forced Hybrid. Each was decoded at every output rate with 1 and 2 channels: without loss, with random and burst losses, with a loss concealed in pieces of 2.5 ms, 7.5 ms and the rest, with 8 packets concealed in one call, and with one or two lost packets rebuilt from FEC data. That makes 13,160 decodes.
  - The 12 test vectors of RFC 8251, decoded in the same ways: 840 decodes.
  - 1,500 multistream streams of libopus (10 layouts, 3 encoder settings, 5 bitrates, 5 frame sizes, 2 signals, some loud enough for the soft clipper), each decoded at every output rate with concealment and FEC: 7,500 decodes.

  Every sample and every final range was libopus's.

An agreement of 172 dB means the difference is 172 dB below the signal. That is far smaller than the error of rounding to 16-bit samples, which alone limits the SNR of a full-scale sine to about 98 dB.

Three last-bit differences were fixed in October 2026 to get there:

- **`smooth_fade`** (`src/opus_decoder.c`), the fade between SILK or Hybrid output and a CELT frame at a mode switch or a redundant CELT frame: `w*in2 + (1-w)*in1`. clang rounds `(1-w)*in1` and adds `w*in2` in one fused multiply-add. opus1 rounded both products. For example, libopus codes `fixtures/answer_16k.wav` (16 kHz mono, VoIP, 24 kb/s, 20 ms frames) as SILK for 28 packets, then as CELT. Packet 27 carries a redundant CELT frame, and 9 of its float samples differed at 16 kHz output (14 at 48 kHz). Test: `tests/core_decode_bitexact.rs`.
- **`opus_pcm_soft_clip`** (`src/opus.c`), the soft clipper of the 16-bit output: `x + a*x*x` is `a*x` rounded, then one fused multiply-add, and the boost `a + a*2.4e-7` is a fused multiply-add too. opus1 rounded every step, so a soft-clipped 16-bit sample could be one step off. `pcm_soft_clip` now gives the bits of `opus_pcm_soft_clip` (`tests/core_decode_i16.rs`).
- **The rounding of halfway values to 16 bits**, described next.

#### Rounding to 16 bits

`opus_decode` converts its float output to 16 bits with `celt_float2int16`. A value that lies exactly halfway between two integers after the scaling by 32768 (for example 9470.5) is rounded by a rule that depends on the processor. opus1 follows the rule of the libopus built for the same target:

| Target | libopus 1.6.1 | Halfway values |
|---|---|---|
| 64-bit ARM | `celt_float2int16_neon` (`celt/arm/celt_neon_intr.c`). libopus's build enables NEON (ARM vector instructions) by default, and every 64-bit ARM processor has them, so this version runs with and without run-time CPU detection. | Away from zero (`vcvtaq_s32_f32`) in whole blocks of 16 values from the start of the call. To the even integer (`vcvtns_s32_f32` in `FLOAT2INT16`) for the values after the last whole block. |
| 32-bit ARM | `celt_float2int16_neon` on a processor with NEON, chosen at run time or at build time. opus1 assumes a processor with NEON, as the Android NDK's default build for `armeabi-v7a` does. | In the blocks, ARMv7 adds 0.5 with the sign of the value and truncates: away from zero. ARMv8 in 32-bit mode uses `vcvtaq_s32_f32`; the two differ only for the scaled value ±0.49999997. opus1 follows ARMv7. After the blocks: to the even integer (`lrintf`). |
| x86 | `celt_float2int16_c`. libopus 1.6.1 has no x86 version of it. `FLOAT2INT16` rounds with `_mm_cvt_ss2si`. | To the even integer. |
| Other targets | `celt_float2int16_c`; `FLOAT2INT16` rounds with `lrintf`. A build without `lrintf` uses `floor(0.5 + x)`, which rounds halfway values up; opus1 does not follow that case. | To the even integer. |

Before, opus1 rounded every halfway value to the even integer. On 64-bit ARM, a sample in a block that libopus rounds away from zero was then one step off. For example, in the stream above (902 packets), 47 packets had such a sample at 16 kHz output (48 samples), and 129 packets at 48 kHz (137 samples).

The other integer outputs convert each sample on its own, as libopus does, and round halfway values to the even integer on every target:

- `MultistreamDecoder::decode_i16`: libopus `opus_multistream_decode` converts each sample with `FLOAT2INT16` (`RES2INT16`).
- The 24-bit output of the evaluation tools (`internals::eval::res2int24`, like libopus `opus_decode24`): `RES2INT24` is `float2int` of each sample. opus1 has no public 24-bit decode call.

`tests/core_decode_i16_ties.rs` decodes loud CELT and Hybrid streams at every output rate, mono and stereo, with losses, and finds the halfway values in the float output of libopus. It requires the 16-bit samples of libopus, with halfway values both in the blocks and after them, and does the same for the multistream decoder.

**Conformance (RFC 8251):** opus1's decoder passes all 120 runs of the test vectors (12 vectors, 48 to 8 kHz, mono and stereo), as libopus's `tests/run_vectors.sh` runs them. The 16-bit outputs are identical to libopus 1.6.1's in all 120 runs, on the Mac and on the phone. The mean quality and the details are in [Testing](09-testing.md#the-conformance-test-of-rfc-8251).

Concealment and FEC are identical too. The concealed frames of all 127 loss cases of `celtdec_plc` (2.5 to 20 ms, decoders at 8 to 48 kHz, CELT and Hybrid) are libopus's to the last bit, and so are the concealed and rebuilt frames of the wider search above, for single streams and for every multistream layout. Before, a small difference in the decoded signal could change the pitch the concealment found. Fuzzing found such a case ([Testing](09-testing.md#fuzzing)): an 8 kHz stream that switched from CELT-only to SILK without a redundant CELT frame, where the 5 ms that the decoder conceals in CELT agreed with libopus to 27 dB. That was measured before the CELT decoder became bit-exact.

## libopus main branch and the 1.6.1 release

The crate `opus-head-sys` 0.3.3, a dependency of the tests, vendors the C source of libopus's main branch at commit `228a0f85` (2026-08-28), not the 1.6.1 release. Earlier work sometimes read that source. Its `celt/`, `silk/`, `src/` and `include/` folders were compared with the official 1.6.1 tarball in October 2026: C files and headers, without `dnn/`, the build files and the tests. `include/` has no difference. The table lists every difference and which version opus1 follows. Line numbers are those of 1.6.1; "after N" means main inserts lines there.

| File | Lines (1.6.1) | What the main branch changes | opus1 |
|---|---|---|---|
| silk/fixed/burg_modified_FIX.c | 70-71 | Adds 1 to the frame energy `C0_64` ("Make sure the noise corresponds to at least a unit pulse"). | 1.6.1. opus1 had the `+ 1` until October 2026 (see [SILK](#silk)). |
| silk/fixed/x86/burg_modified_FIX_sse4_1.c | 78-79 | The same in the SSE4.1 version. | 1.6.1. opus1 has one Burg function for every target. |
| celt/celt_encoder.c | after 1231 | Above 320 bytes per frame, `dynalloc_analysis` raises `follower[0]`, the boost of band 0, by up to 1.5. | 1.6.1: no such boost. |
| celt/celt_encoder.c | 1969-1970 | The silence check measures `CC*(N-overlap)` samples, the input channels, instead of `C*(N-overlap)`, the coded channels. | 1.6.1 (see [CELT encoder](#celt-encoder)). |
| celt/pitch.c | after 536 | For a period shorter than twice the minimum, `remove_doubling` computes the pitch gain at 5/8 and at 6/8 of the period, and sets the gain to 0 when either is at least as large. | 1.6.1: no such check. |
| celt/bands.c | 1750-1753 | For bands at or above `effEBands`, `Y` points to `norm2` instead of `norm`, and an assertion is added. | Not reachable: opus1 has only the standard mode, where `effEBands` equals the 21 bands. |
| celt/bands.c, celt/celt_encoder.c, celt/celt_decoder.c, celt/rate.c, celt/vq.c, celt/modes.c | bands.c 847, 1091, 1675-1676; celt_encoder.c 2557, after 2590, 2659, 2688; celt_decoder.c 1474, 1507; rate.c 752-851; vq.c 526; modes.c 487 | The scalable quality extension (QEXT, `ENABLE_QEXT`): more extra bits per band (12 to 14), a new depth allocation, a cap of 1275 bytes without the extension, bits counted on the extension's range coder, `extra_quant[nbEBands+1]` corrected to `[nbEBands+i]`, `ec_enc_refine`, a comment. | Not ported: opus1 has no QEXT. |
| celt/celt_decoder.c | 644-645, after 671 | Deep PLC (`update_plc_state`): two buffers move from the C stack to `ALLOC`. Same output. | No change needed: `DeepPlc::update_from_celt` keeps them on the stack (8 KB and 1.3 KB), with the same output. |
| src/opus_decoder.c | after 496, after 497 | With the OSCE bandwidth extension, a CELT-only frame resets `osce_extended_mode`. In 1.6.1 a CELT-only frame after bandwidth-extended SILK frames skips its CELT decode, and the range decoder loses its place. | main (see [Deliberate differences](#deliberate-differences)). |
| src/opus_decoder.c | after 735, 1442, after 1530, 1562-1566, after 1597, 1615 | DRED (deep redundancy): rejects a DRED state that was not processed, returns the result of loading a model through `ret`, clears a new DRED decoder, changes how many features are decoded, adds an assertion on `nb_latents`, removes a blank line. | 1.6.1, with the checks listed in [DRED decoding](#dred-decoding). |
| src/repacketizer.c | 169 | A packet whose extensions cannot be parsed gives `OPUS_INVALID_PACKET` instead of `OPUS_INTERNAL_ERROR`. | 1.6.1: opus1's repacketizer and `pad` give "Internal error" (`OPUS_INTERNAL_ERROR`), as 1.6.1 does (see [Packet padding extensions](#packet-padding-extensions)). |
| src/repacketizer.c | 265, after 352, after 353 | `RESTORE_STACK` on two error paths. It matters only for builds with libopus's own stack (neither `VAR_ARRAYS` nor `alloca`). | Not applicable. |
| silk/x86/NSQ_del_dec_avx2.c | 100-106, 138, 518, 567, 901, 959, 1069 | The AVX2 quantizer rounds with `silk_RSHIFT_ROUND` instead of `(a + 2^(b-1)) >> b`. The two differ only when that addition overflows. | opus1 has no AVX2 quantizer. Its quantizer uses `silk_RSHIFT_ROUND`, as the C quantizer of both versions does. |
| silk/float/burg_modified_FLP.c | 69-70 | The float Burg analysis adds 0.1 instead of 1e-9 to the energy. | Not ported: opus1 has no float SILK encoder. The linked Homebrew libopus is 1.6.1 and adds 1e-9. |
| celt/celt.c, celt/celt_encoder.c, celt/celt_decoder.c, celt/float_cast.h, celt/mathops.h, silk/float/SigProc_FLP.h, silk/LPC_inv_pred_gain.c, silk/NLSF2A.c, src/analysis.c | celt.c after 34, after 52; celt_encoder.c after 143; celt_decoder.c after 180; float_cast.h 72; mathops.h after 46-47; SigProc_FLP.h after 146-147; LPC_inv_pred_gain.c after 141; NLSF2A.c after 40, after 140; analysis.c 57, 864, 871 | No change of output: a `celt_fatal` for builds with assertions, prototypes, a header for MSVC, `#ifndef PI` guards, `#undef` of macros, the name `dct_table` changed to `analysis_dct_table`. | Not applicable. |
| src/opus_demo.c | 1006-1026 | The demo program reads a short last frame correctly. | Not part of the library. |

Files in only one of the two trees: main has the tool `celt/dump_modes/`, the TI DSP headers `celt/fixed_c5x.h` and `celt/fixed_c6x.h`, the Xtensa headers in `silk/xtensa/`, and `src/tansig_table.h`, which no file includes. 1.6.1 has the generated assembly file `celt/arm/celt_pitch_xcorr_arm-gnu.S`. None of them is ported.

So opus1 followed the main branch in one place, the Burg energy, and now follows 1.6.1 there too. Some main changes fix memory errors: the cleared DRED decoder (1.6.1 leaves its memory uninitialized), the assertion on `nb_latents`, and `RESTORE_STACK`. Others fix wrong output: the range decoder after the bandwidth extension, the `extra_quant` index of QEXT, and `Y` beyond `effEBands`. Of these, only the fix after the bandwidth extension is in code that opus1 ports, and opus1 keeps it. It is the only main-branch change that opus1 follows.

### Neural network code

The code that the networks share (`src/dnn/`, see [Architecture](07-architecture.md#neural-network-code-srcdnn)) follows libopus 1.6.1 as built for 64-bit ARM at `-O2` without loop vectorization (novec), on every target. Against that build, on the Mac and on the phone (Cortex-A76 and Cortex-A55), every result is bit-exact:

- every kernel and layer function, for every kernel variant (`Portable`, `Neon`, `NeonDotprod`), on random and edge inputs;
- every layer of the six networks PitchDNN, PLC, FARGAN, LACE, NoLACE and BBWENet, with the real weights and inputs from speech: 357,192 uses of a layer on 1804 frames of 10 ms, call by call and also with each side running on its own outputs (`tests/dnn_layers_libopus.rs`);
- the pitch that PitchDNN gives, against libopus's `compute_pitchdnn`;
- the signal analysis of `dnn/freq.c` and `dnn/burg.c`.
- LACE, NoLACE and the bandwidth extension (`src/dnn/osce/`), with their features and filters: every output sample of real speech decoded by opus1's SILK decoder, over the whole TSP corpus, with method switches and resets (Mac only; see [Testing](09-testing.md#tests-of-the-neural-network-code)).

Builds of libopus that turn loops into vector code differ in the last bits. Measured with the same tests:

- **Apple clang 21 at `-O2` (Mac).** It sums the products of float layers with one output in another order. Only these layers differ: `sig_net_cond_gain_dense` of FARGAN; the gains of LACE (`lace_cf1_gain`, `lace_cf1_global_gain`, `lace_cf2_gain`, `lace_cf2_global_gain`, `lace_af1_gain`); the gains of NoLACE (`nolace_cf1_gain`, `nolace_cf1_global_gain`, `nolace_cf2_gain`, `nolace_cf2_global_gain`, `nolace_af4_gain`); and `bbwenet_af3_gain`. With the same inputs, the largest difference was 4.2e-6 (in `lace_cf2_global_gain`), or 1.04e-6 of the sum of the absolute products of that output. That is at most 0.068 of the bound `2·M·2^-24·Σ|w·x|` for a sum of `M` products. When FARGAN runs on its own output, as it does during a loss, such a difference grows: its gain enters the excitation it feeds back, and after a few frames the two runs differ by up to 1.8 in the states of its recurrent layers. The networks that do not feed back their output keep the single-call differences. The last step of `compute_pitchdnn`, which takes the weighted mean of 5 outputs around the maximum, also rounds differently: the pitch differs in 386 of 1804 frames, by at most 4.8e-7.
- **NDK clang 18 at `-O2` (phone).** Every layer of the six networks is bit-exact, on both cores. As on the Mac, the pitch of `compute_pitchdnn` differs: in 375 of 1804 frames, by at most 4.8e-7. `dnn/freq.c` and `dnn/burg.c` differ too, because their sums are vector code.
- **`dnn/freq.c` and `dnn/burg.c` at `-O2` on the Mac.** The band energies differ in 14,976 of 88,452 values, by at most 2.7e-7 of the value. Burg's method differs in 78 of 377,951 values, by one step of the float format. The Burg cepstrum differs in 42,246 of 152,028 values, by at most 5.3e-6 (the values reach 23). `lpc_from_cepstrum` differs in the last bits of 41,770 of 93,772 values; in one case one build stops the recursion early and the other does not. The FFT, the window and the DCT give the same bits.

- **OSCE against the `-O2` build of the Mac.** Besides the gain layers above, this build sums the kernel norm of `nndsp.c` and the filter banks of the features in another order. Measured with `tests/dnn_osce_libopus.rs` over the whole TSP corpus (one stream of 1444 files per model, decoded by opus1's SILK decoder):

  | Model | Frames that differ | Largest difference | SNR of the difference |
  |---|---:|---:|---:|
  | LACE | 4,280 of 195,082 | 142 LSB | 70.2 dB |
  | NoLACE | 8,362 of 194,504 | 89 LSB | 73.1 dB |
  | method switches (NONE, LACE, NoLACE) | 3,028 of 196,996 | 31 LSB | 84.4 dB |
  | bandwidth extension (10 and 20 ms calls) | 70,152 of 231,721 calls | 166 LSB | 66.4 dB |
  | NoLACE, then the bandwidth extension | 194,717 of 195,081 | 190 LSB | 60.8 dB |

  LSB is one step of a 16-bit sample. On shorter streams (48 files, about 4,400 enhanced frames per model) LACE differed in 133 frames and NoLACE in 216, each by at most 1 LSB (103.9 and 101.9 dB). The bandwidth extension differed in 2,519 of 7,742 calls by up to 130 LSB (63.5 dB).

The models of deep PLC (`src/dnn/plc/`: the feature analysis, PitchDNN, FARGAN and the state machine of `dnn/lpcnet_plc.c`) are bit-exact against the build without vector loops: every concealed frame and the state after every call, for every kernel variant, on the speech fixture and the 1,444 files of the TSP corpus on the Mac, and on the fixture and 50 corpus files on the phone (Cortex-A76 and Cortex-A55) (`tests/dnn_plc_libopus.rs`).

They run in the decoder, and the decoder with them is bit-exact against the same build: every float sample, every 16-bit sample and every final range of `OpusDecoder` and `MultistreamDecoder`, for streams of libopus's and opus1's encoders in every mode, at every output rate, with single losses, bursts, losses longer than 200 ms, FEC, resets and complexity changes (`tests/dnn_plc_decoder_libopus.rs`, Mac). This includes complexities 6 to 10, where LACE and NoLACE run as well.

LACE, NoLACE and the bandwidth extension run in the decoder too, and the decoder is bit-exact against the same build at every complexity, with the BWE on and off, at every output rate, in mono, stereo and multistream decoders, with losses, FEC, resets and switches of the complexity, of the BWE and of the coding mode (`tests/dnn_osce_decoder_libopus.rs`, Mac). The exception is the defect of libopus 1.6.1 with CELT-only frames after frames with the BWE ([Deliberate differences](#deliberate-differences)): a comparison stops at the first such frame. Against the `-O2` build of the Mac, the decoded output of the streams of that test without CELT-only packets (mono and stereo, 48 kHz, no loss; `distance_to_an_o2_build`) differs as follows:

| Setting | Decode calls that differ | Largest difference | SNR of the difference |
|---|---:|---:|---:|
| LACE (complexity 6) | 114 of 19,552 | 2 LSB | 96.9 dB |
| NoLACE (complexity 7) | 196 of 19,552 | 3 LSB | 94.1 dB |
| BWE (complexity 4) | 2,360 of 19,552 | 12 LSB | 71.3 dB |
| NoLACE and BWE (complexity 7) | 5,683 of 19,552 | 46 LSB | 64.6 dB |

With speech of the TSP corpus and streams of 8 s (`OPUS1_DNN_OSCE_ALL=1`, 39,108 calls per setting), LACE differed in 280 calls by at most 3 LSB (97.8 dB), NoLACE in 862 calls by at most 56 LSB (62.1 dB), the BWE in 7,155 calls by at most 14 LSB (71.8 dB), and NoLACE with the BWE in 14,091 calls by at most 99 LSB (60.5 dB).

Against the `-O2` build of the Mac, the differences above grow, because the networks feed back their outputs:

- The features differ in 343,519 of 343,656 frames (the fixture and the whole corpus); the largest difference, 2.56, is in the pitch.
- PitchDNN, run on its own state with the same inputs, gives another pitch in 74,764 of 343,656 frames.
- FARGAN, run on its own output for whole files, differs at 14.0 dB SNR.
- The concealed frames differ at 15.7 dB SNR (scripted losses on the fixture) and 13.2 dB (random losses on all 1,445 signals, 56,800 concealed frames). Builds of libopus differ from each other by about as much: 10.0 to 15.4 dB between the Mac `-O2`, the Mac `-Os` and the Android `-O2` builds.
- In the decoder (`OpusDecoder` at complexity 5, mono 48 kHz output, 62 streams of every mode, with single losses, bursts and long losses; `distance_to_an_o2_build` in `tests/dnn_plc_decoder_libopus.rs`), the concealed audio differs at 13.5 dB SNR (12.1 dB with speech of the TSP corpus, 20,370 concealed calls), and the audio decoded after the losses, which continues from the concealed state, at 26.9 dB (27.3 dB). The `-O2` build also rounds the filters of the CELT part of deep PLC (`update_plc_state` and the upsampler) and the pitch-based CELT concealment differently.

Two places of the deep PLC models are written for memory safety. Only DRED features (deep redundancy, feature `dred`) reach them:

- A DRED feature vector beyond the 104 places of the queue is dropped. libopus stops with an assertion, or, built without assertions, writes past the queue.
- A DRED pitch above 7.5 gives FARGAN the period 0. libopus then reads past the end of its pitch buffer, into the next field of `FARGANState`. opus1 reads the same values from that field, so the output is the same, without reading out of bounds.

Without `std`, the networks' `log10`, `pow`, `exp`, `exp2` and `log`, and the `sin` and `cos` of OSCE, come from the `libm` crate instead of the platform's C library. Their differences were not measured.

## Platforms

- **x86-64 was run once, on an Intel Core i7-12700 (AVX2, FMA and AVX-VNNI, no AVX-512), on 2026-10-21:** under Linux in WSL2 against libopus 1.6.1 built by its `configure` with GCC 11, and natively on Windows with Rust's MSVC toolchain against libopus built with CMake and Visual Studio (`scripts/x86/test.sh` and `test.ps1`).
  - The neural network kernels for AVX2 with FMA and for AVX-VNNI gave the bits of the portable code in every test (33 of 33). AVX-512 VNNI was not run.
  - About 40 tests that require libopus's bits failed on x86. opus1's x86 vector kernels do not follow libopus's x86 kernels (SSE4.1, AVX2, some with fused multiply-adds) the way its 64-bit ARM kernels follow libopus's NEON code, and some tests compare with values recorded on 64-bit ARM. These tests are marked `ignore` on targets other than 64-bit ARM, with the reason "bit-exact with libopus only on 64-bit ARM". The tests of the neural network code that compare with libopus are compiled for 64-bit ARM only.
  - libopus's own results on x86 depend on the compiler. With the same opus1 values, `celtenc_bands_libopus::stereo_angles_are_the_ones_libopus_computes` passed against libopus built by GCC 11 (WSL2) and failed against libopus built by Apple clang (x86-64 under Rosetta 2) and by MSVC (Windows): for one input, opus1 and the GCC build give the stereo angle 655179584, the other two 655179520. The band tests of that file are therefore among the ignored ones.
  - A memory error in an x86 kernel was found and fixed: the AVX2 pulse search (`pvq_search_avx2`) could add a pulse past the end of its output when every score was 0 (an all-zero band or NaN input). An audit of every raw-pointer access in the x86 kernels found no other reachable one; the kernels now check their slice lengths at entry.
  - One x86 kernel did not follow libopus: the AVX2 version of `haar1` computed `(a+b)*s` where libopus computes `s*a + s*b`. It is fixed.
  - The libopus that libopus's CMake files build differs from the `configure` build on x86: CMake leaves out `FLOAT_APPROX`, which `configure` turns on for x86, and `ENABLE_OSCE_BWE`. `scripts/x86/test.ps1` adds both.
  - Speed against that libopus at `-O2` (bench set, geometric means): encode 0.90 and decode 0.81 of libopus's time under WSL2, 0.88 and 0.80 natively on Windows. Low-complexity SILK encoding took 1.04 to 1.10 of libopus's time.
- 32-bit x86 (`i686-unknown-linux-gnu`) was compiled once, without `std`, with `libm` and every neural network feature, with and without the AVX target features. The Rust standard library for that target is not installed here, so `core` was built from the `rust-src` component for this check; it is not part of the regular checks. The vector kernels for the pulse search, the band energies and the renormalization are not ports of libopus's SSE2 code. The kernels for the inner products and correlations of the pitch analysis do not follow libopus's x86 kernels in every detail either. SSE2 (Streaming SIMD Extensions 2, where SIMD means single instruction, multiple data) is the baseline vector instruction set of x86-64. The [README](../README.md) keeps an older x86-64 speed measurement of opus-rs 0.1.x.
- **64-bit ARM rounding follows clang.** clang compiles C with `-ffp-contract=on` by default. Where a product and a sum stand in one expression, it emits one fused multiply-add. opus1 copies this in its FFT, its MDCT, the pitch analysis and the comb filter of CELT, the coarse energy and several other places. So on 64-bit ARM they give the same bits as the arm64 build of libopus. A libopus built without contraction, or built by GCC (the GNU Compiler Collection) with `-ffp-contract=fast`, rounds differently. On other targets, the FFT of opus1 rounds each product before the addition.
- **Speed cost of following libopus:**
  - The speech and music analysis runs from complexity 7 on, as in libopus. It adds about 2.9 µs per 20 ms frame and stream on Apple M4 Pro, 11.6 µs on a Cortex-A76 and 34.7 µs on a Cortex-A55 (48 kHz mono), which is 0.27 to 0.40 of the time of libopus's own analysis ([Performance](08-performance.md#the-speech-and-music-analysis)).
  - `theta_rdo` adds 34 % to 73 % encode time on the six stereo bench scenarios at c9 and c10. The opus1 / libopus encode time ratio stays 0.53 to 0.63.
  - The exact pulse search of libopus costs 3 % to 11 % CELT encode time (earlier).
  - The pitch analysis of the pre-filter adds its sums in libopus's order. The autocorrelation of `pitch_downsample` and the window energies of `remove_doubling` are then chains of fused multiply-adds in which each step waits for the one before. With 20 ms frames the new kernels make up for that: the three functions take about as long as before (4.0 to 4.1 µs per frame). With 2.5 ms frames they take 0.25 to 0.34 µs (13 % to 18 %) longer per frame, and with 5 ms frames 0.19 to 0.23 µs (9 % to 10 %). Encoding stereo 128 kb/s c9 with 2.5 ms frames took 0.6 % longer than before this release round (six alternating bench runs), even though the comb filter became faster.

## Tests that track these gaps

No test is ignored because of a gap. On targets other than 64-bit ARM, about 40 tests that require libopus's bits are ignored (see [Platforms](#platforms)). The two tests that were, both because of the missing speech and music analysis, pass now:

| Test | Measured |
|---|---|
| `oracle_celt::impulse_48k_mono_64k_is_the_packet_of_libopus` | 160 of 160 bytes equal (7 before) |
| `multistream_enc_signals::average_rate_at_16_khz_and_for_voip_is_within_the_design_limits` | -12.8 % to +4.1 % around the requested bitrate (65 % to 80 % of it before) |

The ignored tests print report tables (see [Testing](09-testing.md#ignored-tests)).

## See also

- [Testing](09-testing.md): how these comparisons are made.
- [Performance](08-performance.md): speed against libopus.
- [Using the library](06-using-the-library.md): the encoder and decoder calls that opus1 does offer.
- [Glossary](glossary.md): the terms used in these documents.
- The documentation index: [README.md](README.md).
