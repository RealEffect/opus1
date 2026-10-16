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

## What the figures were measured against

- The reference is **libopus 1.6.1 as built by Homebrew**, on Apple M4 Pro (64-bit ARM, macOS). It is a floating-point build.
- The C source of libopus's main branch differs from 1.6.1 in a few places. opus1 follows 1.6.1, checked against the arm64 object code of the Homebrew build. Two examples: `remove_doubling` has no second check of very short periods, and `dynalloc_analysis` gives band 0 no extra boost above 320 bytes per frame.
- Comparisons of the SILK encoder at the packet level also use **libopus 1.6.1 built with `-DFIXED_POINT`** (see [Testing](09-testing.md#checks-against-the-fixed-point-build-of-libopus)).
- Most figures come from `perf_harness modes` (see [Performance](08-performance.md#the-harness)) over its 129 scenarios with 5 s clips, at the final integration state of this release. In those runs, "SNR" is the decoded audio against the encoder input. The opus1 encoder is compared with the libopus encoder under the same forced settings, and libopus decodes both streams.
- Figures marked **(earlier)** come from earlier stages of the work and were not measured again in the final state.
- An "SNR difference" below is opus1's SNR minus libopus's SNR. A positive value means opus1 is closer to the input.
- A "deficit" is the other way round: libopus's SNR minus opus1's SNR. A negative value means opus1 is closer.

## Not implemented

### Speech and music analysis

libopus has an analysis step (`analysis.c`) that estimates whether the input is speech or music, how tonal it is, and which bandwidth it really uses. The floating-point build of libopus runs it at complexity 7 and above for input at 16 to 48 kHz. opus1 has no port of it. Instead:

- The estimated chance of speech is a constant per application: 115 (on a scale of 0 to 127) for `Voip`, 48 otherwise. libopus uses the same constants when it runs no analysis.
- The bandwidth that the analysis would detect is not applied.
- The CELT encoder gets no analysis results.

This causes several of the differences listed below: in CELT packets at complexity 7 and above, in the CELT impulse test, and in the multistream bitrate at 16 kHz and for VoIP.

### Encoder features and controls

- **Discontinuous transmission (DTX)** is not implemented. DTX lets the encoder send almost nothing during silence.
- **Frames of 80, 100 and 120 ms** are not accepted. The longest encoder frame is 60 ms. libopus's "expert frame duration" control is not offered either.
- **Phase inversion cannot be disabled.** CELT's intensity stereo codes one band for both channels and may invert the sign of one of them. libopus documents that disabling this improves a mono downmix of the stream and slightly lowers normal stereo quality. libopus has `OPUS_SET_PHASE_INVERSION_DISABLED` for it. opus1's CELT encoder always allows phase inversion: it calls `quant_all_bands_enc` with `disable_inv` set to `false`.
- These libopus encoder controls have no equivalent in `OpusEncoder`:
  - the signal type (`OPUS_SET_SIGNAL`),
  - the input bit depth (`OPUS_SET_LSB_DEPTH`; see [Deliberate differences](#deliberate-differences)).

The encoder refuses an 80 ms frame with an error:

```rust
use opus1::{Application, OpusEncoder};

let mut encoder = OpusEncoder::new(48_000, 1, Application::Audio).unwrap();
let mut packet = vec![0u8; 4000];

// 60 ms (2880 samples at 48 kHz) is the longest frame the encoder takes.
let pcm_60 = vec![0.0f32; 2880];
assert!(encoder.encode(&pcm_60, 2880, &mut packet).is_ok());

// 80 ms (3840 samples) is refused. libopus would accept it.
let pcm_80 = vec![0.0f32; 3840];
assert!(encoder.encode(&pcm_80, 3840, &mut packet).is_err());
```

### Decoder controls

The decoder decodes every valid packet, up to 120 ms. These libopus decoder controls have no equivalent in `OpusDecoder`:

- the output gain (`OPUS_SET_GAIN`),
- phase inversion disabling (`OPUS_SET_PHASE_INVERSION_DISABLED`). The decoder does what libopus does by default: it disables phase inversion only when it mixes a stereo stream down to mono output,
- the duration of the last packet (`OPUS_GET_LAST_PACKET_DURATION`),
- the pitch of the last frame (`OPUS_GET_PITCH`).

### Multichannel

- **Mapping family 3** (ambisonics mixed through a matrix, libopus's `opus_projection_*` functions) is not implemented. Families 0, 1, 2 and 255 are.
- The multistream encoder offers no 80, 100 and 120 ms frames and no expert frame duration control. The source names this difference D3. The labels used here (D1, D3, D5) are those of the module documentation in `src/multistream/encoder.rs`.
- The multistream encoder does not forward these controls: input bit depth, signal type, DTX, forced channel count, prediction and phase inversion (D5). The forced channel count and the prediction switch can be set on each stream's encoder through `MultistreamEncoder::stream_encoder_mut`.

### Packet padding extensions

libopus 1.5 and later can store extensions in the padding of a packet. The `packet` module does not read them. When a packet already has padding whose bytes are not all zero:

- libopus reads those bytes as extensions and copies them into the new padding. opus1's `pad` and `multistream_pad` write zeros.
- libopus rejects the packet when the bytes are not valid extensions. opus1 pads it.

For a packet without padding, or with zero padding, the result is libopus's byte for byte.

### Optional libopus features

libopus has features that are compiled only when it is configured with them. They are off by default:

- deep packet loss concealment (`--enable-deep-plc`);
- Deep REDundancy (DRED, `--enable-dred`);
- Opus Speech Coding Enhancement (OSCE, `--enable-osce`);
- the scalable quality extension (`--enable-qext`).

These options are listed in libopus's `configure.ac`. opus1 implements none of them.

## Deliberate differences

These differences are intended. Most are listed in the comment above `OpusEncoder::encode_impl` in `src/lib.rs`. The multistream ones are in `src/multistream/encoder.rs`.

| What | opus1 | libopus | Why |
|---|---|---|---|
| SILK encoder | Port of the fixed-point encoder (`silk/fixed`). | The floating-point build uses `silk/float`. | By design. |
| SILK input with the `Voip` application | SILK gets the output of the integer high-pass filter `hp_cutoff` of the fixed-point build. That filter has its own state and runs only on frames that SILK codes. | The floating-point build filters the float signal and rounds it to 16 bits. | Follows the fixed-point build, whose SILK encoder opus1 ports. |
| Input bit depth | 16-bit input (`encode_i16`) is coded with the silence threshold of 24-bit input. | Uses `lsb_depth` 16 for 16-bit input. | `encode` and `encode_i16` give the same packets for the same audio. |
| 40 and 60 ms CELT and Hybrid packets with VBR | No limit of each 20 ms frame to its share of the bitrate (libopus `opus_encoder.c`, line 1785). | Limits each frame. | Kept on purpose. It uses more bits; see [Hybrid and mode switching](#hybrid-and-mode-switching). |
| Hybrid CELT bitrate | Never below 501 b/s. | No such floor. | On purpose. |
| Switch to CELT-only without room for the redundant frame | The CELT encoder is still reset. | Goes on from its last CELT frame. | The decoder then starts the CELT-only packets from the same reset state. |
| Energy mask (surround) | Copied, and applied again on the next frame after a CELT reset. | Keeps a pointer, which is lost after a CELT reset until the caller sets it again. | Rust ownership. The values are the same. |
| Multistream bandwidth with the automatic or maximum bitrate in surround mode (D1) | Chosen from the bitrate that was allocated. Codes fullband. | Compares the raw setting and always chooses narrowband. | Narrowband 5.1 at a high bitrate is not useful. |

## Where the numbers differ

### CELT encoder

- **The CELT encoder alone** (`CeltEncoder`, without the speech and music analysis that libopus's Opus layer adds): every packet is identical to libopus. `celtenc_gap_report` has 960 streams with 1,080,000 packets: mono and stereo, 16 and 48 kHz input, frames of 2.5 to 20 ms, 24 to 96 kb/s per channel, CBR and VBR, complexity 4, 5 and 9. `celtenc_rate_rules` adds VBR at bitrates that are not a whole number of bits per frame, bitrates above 750 kb/s per channel, and silent frames. Before this release round 91.2 % of the 1,080,000 packets were identical.
- **Stereo input whose two channels are scaled copies of each other** (one source panned to one side), at 128 kb/s per channel and more: a few packets still differ from libopus. Found in the verification of this release round with tones from -134 to -40 dBFS, at 128 to 750 kb/s per channel and frames of 2.5 to 20 ms: up to 3 of 50 to 400 packets per stream differ. With the right channel inverted, or with another tone in it, no packet differed in the cases tried. The streams of `celtenc_gap_report` do not have such input: their right channel is delayed. Before this release round most of these streams differed in many packets.
- **Complexity 0 to 5:** packets are identical to libopus, 250 of 250 per stream.
- **Complexity 7 and above, 16 to 48 kHz:** identical packets drop to 4 of 250 (stereo 128 kb/s at c7 and c10) and 24 of 250 (mono 64 kb/s at c10). At 8 and 12 kHz, where libopus runs no analysis, the c9 low-delay streams stay at 250 of 250. SNR difference over the CELT streams: mean +0.08 dB, worst -0.2 dB.
- **Stereo at c9 and c10** (with `theta_rdo`, the second quantization pass per stereo band): identical packets range from 4 of 250 (128 kb/s VBR, c10) to 151 of 250 (128 kb/s CBR, c9) and 185 of 250 (510 kb/s VBR). SNR difference: mean +0.04 dB, worst -0.2 dB. The difference comes from the missing analysis: without it, the CELT encoder writes the packets of libopus (see the first item).
- **Impulse at 48 kHz, mono, 64 kb/s CBR, c10:** 7 of 160 bytes equal libopus. The first difference is at byte 7. The allocation trim differs, because the analysis is missing. Test: `oracle_celt::impulse_48k_mono_64k_is_the_packet_of_libopus` (ignored).
- **Rounding on 64-bit ARM.** clang fuses some products and sums in libopus into one FMA. opus1 copies this in the pitch analysis of the pre-filter (`src/pitch.rs`, `src/celt_lpc.rs`), the comb filter and the prediction of the coarse energy, among others. A few such places are not copied in opus1. No case was found where they change a packet:
  - `transient_analysis` rounds products separately and uses single precision. In libopus 1.6.1 on arm64 the high-pass filter, the energy of two samples and the smoothing are fused, and two square-root arguments are computed in double precision. Its decisions matched libopus in 300 of 300 frames (earlier).
  - `intensity_stereo` has no `EPSILON` terms and clamps with `max(1e-10)`. A 4000-frame probe found 0 differences (earlier).
  - The fused rounding of the two-bin side-sign product is not ported. It matters only at the angles 0 and 16384, where no sign is coded (earlier).
  - Others, for example `importance` (0.5 + 13 * exp2): no case found where they change a packet (earlier).
  - The pre-filter compares the sum of the absolute sample values before and after filtering. opus1 adds them in four lanes, libopus one value after the other.

### SILK

- **Against floating-point libopus:** 44 of 12290 SILK packets and 20 of 7583 Hybrid packets are identical (CELT: 3942 of 26958). This is expected, because the two SILK encoders compute differently. SNR difference over the 39 SILK streams: mean -0.04 dB, worst -0.9 dB (`silk_wb_16k_mono_20k_vbr_c9_20ms_audio`), best +0.6 dB (`silk_mb_12k_stereo_24k_vbr_c9_20ms`). Over all 100 streams with an SNR: mean +0.01 dB.
- **Against fixed-point libopus** (packets compared byte for byte):
  - VoIP CBR SILK-only packets first differ from those of the fixed-point build at packet 294 (mono, 16 kHz, 32 kb/s, c9). Stereo, and VBR capped at 80 bytes, first differ at packet 163 (earlier).
  - The VoIP SILK prefill reads the float delay buffer. The fixed-point build reads its 16-bit `hp_cutoff` output. Stereo VoIP after a rate switch: 312 of 400 packets equal. Mono packets are equal again after 1 to 5 packets (earlier).
  - `silk_encode_prefill` has no `to_mono` argument. libopus passes `toMono` to its prefill. No failing case was found (earlier).
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
- **Bandwidth switches** (test `core_redundancy::silk_bandwidth_switches_match_libopus`), deficit in the worst window around each switch:
  - 48 kHz: mean -1.0 dB with CBR and -0.7 dB with VBR, so opus1 is closer to the input than libopus.
  - 16 kHz: mean +0.8 dB with CBR, worst switch 6.7 dB. Mean -1.2 dB with VBR, worst switch 3.6 dB.

### Hybrid and mode switching

- **Fullband Hybrid at 48 kHz**, SNR against floating-point libopus (earlier). The first three rows use CBR, the last two VBR:

  | Stream | opus1 | libopus |
  |---|---:|---:|
  | mono 32 kb/s, CBR, c5 and c9 | 8.4 dB | 8.7 dB |
  | mono 32 kb/s, CBR, c0 | 8.4 dB | 8.5 dB |
  | stereo 40 kb/s, CBR, c9 | 8.1 dB | 8.4 dB |
  | stereo 40 kb/s, VBR, c9 | 8.2 dB | 8.6 dB |
  | mono 32 kb/s, VBR, c9 | 8.7 dB | 8.7 dB |

  This is not explained. libopus runs no analysis at c5, and the SILK layer equals the fixed-point build's.
- **40 and 60 ms Hybrid VBR uses more bits** than libopus with the same forced settings, because of the missing per-frame limit (see [Deliberate differences](#deliberate-differences)):

  | Stream | opus1 | libopus | More |
  |---|---:|---:|---:|
  | fullband 48 kHz mono 32 kb/s, 40 ms | 32.0 kb/s | 30.2 kb/s | +6.0 % |
  | super-wideband 24 kHz mono 24 kb/s, 60 ms | 24.4 kb/s | 22.5 kb/s | +8.4 % |
  | super-wideband stereo 36 kb/s, 40 ms | 34.7 kb/s | 31.9 kb/s | +8.8 % |

- **Hybrid after SILK-only packets:** the high band of the first three Hybrid packets is at most 0.7 dB from libopus (test `core_redundancy::hybrid_after_silk_only_starts_its_celt_layer_afresh`).
- **Mode switches**, forced or driven by the bitrate (the other `core_redundancy` tests): the worst window around a switch is up to 4.2 dB below libopus (48 kHz stereo, 20 ms, CBR). The mean deficit per stream is -1.4 to +1.4 dB.

### Multistream

- **Surround masking:** the band levels are taken with libopus's approximation of the logarithm (`celt_log2`), as libopus does. 98.7 % of the mask values are the bits of libopus's `surround_analysis` (`multistream_surround_c_direct`, 1,247,400 values; 72.8 % with the exact logarithm used before). The others differ by at most 3.8e-6, from the rounding of the MDCT.
- **Bitrate at 16 kHz and for VoIP:** opus1 uses 65 % to 80 % of the requested bitrate in four cases where libopus uses 101 % to 102 %. The cause is the missing speech and music analysis. Test: `multistream_enc_signals::average_rate_at_16_khz_and_for_voip_is_within_the_design_limits` (ignored).
- **Stereo Hybrid streams** in the matrix of `tests/multistream_enc_validity.rs` (48 and 24 kHz, VBR and CBR), counted in packets that contain a stereo Hybrid stream:
  - At 8 kb/s per channel: libopus 128, opus1 0. Both code most of them as mono. With two channels forced: libopus 310, opus1 220.
  - At 24 kb/s per channel: libopus 111, opus1 0, forced or not. opus1 picks CELT.

### Decoder

On 64-bit ARM the decoder gives the samples of libopus for the packets of libopus:

- **CELT:** 51 of 51 streams of `perf_harness modes` are identical. Before this release round 8 were; the others agreed to 136 to 146 dB. The causes were four filter taps of the post-filter written with too few digits, the rounding of the post-filter, and the rounding of the coarse energy prediction in frames shorter than 20 ms.
- **Hybrid:** 28 of 29 streams are identical. One FEC stream agrees to 185 dB.
- **SILK:** 48 of 49 streams are identical. One FEC stream agrees to 186 dB.

An agreement of 185 dB means the difference is 185 dB below the signal. That is far smaller than the error of rounding to 16-bit samples, which alone limits the SNR of a full-scale sine to about 98 dB.

Concealment is identical too. The concealed frames of all 127 loss cases of `celtdec_plc` (2.5 to 20 ms, decoders at 8 to 48 kHz, CELT and Hybrid) are libopus's to the last bit, and so are the concealed frames of `multistream_dec_loss` for every layout. (Its streams with FEC agree with libopus to 156 to 200 dB.) Before, a small difference in the decoded signal could change the pitch the concealment found. Fuzzing found such a case ([Testing](09-testing.md#fuzzing)): an 8 kHz stream that switched from CELT-only to SILK without a redundant CELT frame, where the 5 ms that the decoder conceals in CELT agreed with libopus to 27 dB. That was measured before the CELT decoder became bit-exact.

## Platforms

- **x86-64 was only compiled in this release, not run.** Its vector kernels for the pulse search, the band energies and the renormalization are not ports of libopus's SSE2 code. Its kernels for the inner products and correlations of the pitch analysis do not follow libopus's x86 kernels in every detail either. SSE2 (Streaming SIMD Extensions 2, where SIMD means single instruction, multiple data) is the baseline vector instruction set of x86-64. The [README](../README.md) keeps an older x86-64 speed measurement of opus-rs 0.1.x.
- **64-bit ARM rounding follows clang.** clang compiles C with `-ffp-contract=on` by default. Where a product and a sum stand in one expression, it emits one fused multiply-add. opus1 copies this in its FFT, its MDCT, the pitch analysis and the comb filter of CELT, the coarse energy and several other places. So on 64-bit ARM they give the same bits as the arm64 build of libopus. A libopus built without contraction, or built by GCC (the GNU Compiler Collection) with `-ffp-contract=fast`, rounds differently. On other targets, the FFT of opus1 rounds each product before the addition.
- **Speed cost of following libopus:**
  - `theta_rdo` adds 34 % to 73 % encode time on the six stereo bench scenarios at c9 and c10. The opus1 / libopus encode time ratio stays 0.53 to 0.63.
  - The exact pulse search of libopus costs 3 % to 11 % CELT encode time (earlier).
  - The pitch analysis of the pre-filter adds its sums in libopus's order. The autocorrelation of `pitch_downsample` and the window energies of `remove_doubling` are then chains of fused multiply-adds in which each step waits for the one before. With 20 ms frames the new kernels make up for that: the three functions take about as long as before (4.0 to 4.1 µs per frame). With 2.5 ms frames they take 0.25 to 0.34 µs (13 % to 18 %) longer per frame, and with 5 ms frames 0.19 to 0.23 µs (9 % to 10 %). Encoding stereo 128 kb/s c9 with 2.5 ms frames took 0.6 % longer than before this release round (six alternating bench runs), even though the comb filter became faster.

## Tests that track these gaps

Two tests stay ignored because of the gaps above. Both are caused by the missing speech and music analysis:

| Test | Measured |
|---|---|
| `oracle_celt::impulse_48k_mono_64k_is_the_packet_of_libopus` | 7 of 160 bytes equal |
| `multistream_enc_signals::average_rate_at_16_khz_and_for_voip_is_within_the_design_limits` | 65 % to 80 % of the requested bitrate, against libopus's 101 % to 102 % |

Run them with `cargo test --release --test oracle_celt -- --ignored` and `cargo test --release --test multistream_enc_signals -- --ignored`. The other ignored tests print report tables (see [Testing](09-testing.md#ignored-tests)).

## See also

- [Testing](09-testing.md): how these comparisons are made.
- [Performance](08-performance.md): speed against libopus.
- [Using the library](06-using-the-library.md): the encoder and decoder calls that opus1 does offer.
- [Glossary](glossary.md): the terms used in these documents.
- The documentation index: [README.md](README.md).
