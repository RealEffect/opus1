# Architecture

This document describes how the source code of `opus1` is organised, for maintainers and for readers who want to find the code behind a feature. It lists every file under `src/` with the libopus file it ports. It shows which types own which, and explains where the code uses floating point and where it uses fixed point. It then covers the three feature flags and how they change where state is stored, the code written for specific processors, and the rules for `unsafe` code. The last sections cover error handling, and the layout of the tests, examples, benchmarks and fuzz targets. How to call the library is in [Using the library](06-using-the-library.md), and the [documentation index](README.md) lists the other documents.

## Terms

- **libopus**: the reference implementation of Opus in C, from the Xiph.Org Foundation. `opus1` follows libopus 1.6. A "port" of a C file is Rust code that computes what that file computes, function by function.
- **SILK**: the speech coder inside Opus, based on linear predictive coding (LPC). Its code is in `src/silk/`.
- **CELT** (Constrained Energy Lapped Transform): the transform coder inside Opus, based on the modified discrete cosine transform (MDCT). Its code is in the files at the top of `src/`.
- **Opus layer**: the code that sits above SILK and CELT. It chooses the mode for each packet, runs one or both coders, and builds the packet. It is in `src/lib.rs`.
- **State**: everything an encoder or decoder keeps from one frame to the next, such as filter memories and the previous frame's band energies.
- **Kernel**: a small function that does the arithmetic of a frequently run loop, such as an inner product. Many kernels have one version per processor family.
- **Fixed point**: integers that stand for fractions. A value in Q15 format is an integer divided by 2^15. SILK uses 16-bit and 32-bit integers in this way.
- **TOC byte**: the table-of-contents (TOC) byte at the start of every packet. It names the mode, the audio bandwidth, the frame duration and the channel count ([Opus packets](02-packets.md)).
- **Range coder**: the entropy coder of Opus (sections 4.1 and 5.1 of RFC 6716, the Request for Comments that defines Opus). Both SILK and CELT write their symbols into it, and it produces the bytes of a frame.

## Source layout

### The Opus layer and shared code

| File | What it does | libopus counterpart |
|---|---|---|
| `src/lib.rs` | `OpusEncoder` and `OpusDecoder`: mode and bandwidth decisions, in-band forward error correction (FEC) decisions, mode switches with redundant CELT frames, packets of 40 and 60 milliseconds (ms), packet loss, `pcm_soft_clip`. | `src/opus_encoder.c`, `src/opus_decoder.c`, `opus_pcm_soft_clip` in `src/opus.c` |
| `src/packet.rs` | Reading, writing and padding packets; self-delimited framing; checks of multistream packets. | `opus_packet_parse_impl` in `src/opus.c`, `src/repacketizer.c`, `opus_multistream_packet_validate` in `src/opus_multistream_decoder.c` |
| `src/hp_cutoff.rs` | The encoder's input filters: the high-pass filter for speech (`hp_cutoff`, `hp_cutoff_i16`, `hp_cutoff_float`) and, for other audio, the filter that removes the constant (DC, direct current) part of the signal (`dc_reject_float`). The integer filters use the second-order filter of `src/silk/sigproc_fix.rs`. | `hp_cutoff` and `dc_reject` in `src/opus_encoder.c` |
| `src/range_coder.rs` | `RangeCoder`: the range encoder and decoder, bit counting (`tell`, `tell_frac`), Laplace-distributed symbols. | `celt/entenc.c`, `celt/entdec.c`, `celt/entcode.c`, `celt/laplace.c` |
| `src/fixedvec.rs` | `FixedVec<T, N>`: a growable buffer with a fixed capacity, stored inline. See [Feature flags and where state is stored](#feature-flags-and-where-state-is-stored). | none |
| `src/compat.rs` | `OnceCell` (a value built once, without the heap), a stable insertion sort (`sort_by`), the x86 feature probes (`x86_has_avx` and others), and the `Math` trait that maps float functions to `libm` without `std`. | none |
| `src/mathops.rs` | Float helpers that several modules share: `mac` and `mac_f64` (multiply-add, rounded as libopus rounds it on the target processor), `celt_log2` and `celt_exp2`. | `celt/mathops.h`, `celt/mathops.c` |

### The CELT layer

| File | What it does | libopus counterpart |
|---|---|---|
| `src/celt.rs` | `CeltEncoder` and `CeltDecoder`: frame analysis (transients, tone detection), the pitch pre-filter and post-filter (comb filters), band allocation decisions, synthesis, de-emphasis, CELT packet loss concealment (PLC). `PcmLayout` says how the decoder writes a frame: interleaved, added to interleaved samples already in the buffer, or one channel after the other. | `celt/celt_encoder.c`, `celt/celt_decoder.c`, `celt/celt.c` |
| `src/modes.rs` | `CeltMode`: the band layout, window and tables of the one mode Opus uses (48 kilohertz (kHz), 960-sample frames, 120-sample overlap), built once by `default_mode()`. | `celt/modes.c`, `celt/static_modes_float.h` |
| `src/bands.rs` | Band energies, normalisation, the coding of all bands (`quant_all_bands`), stereo angles, spreading, anti-collapse. | `celt/bands.c` |
| `src/quant_bands.rs` | Coarse and fine quantization of the band energies. | `celt/quant_bands.c` |
| `src/rate.rs` | Bit allocation between the bands (`clt_compute_allocation`) and the pulse-count cache lookups. | `celt/rate.c`, `celt/rate.h` |
| `src/pvq.rs` | The pyramid vector quantizer (PVQ): pulse search (`pvq_search`, `alg_quant`), `alg_unquant`, and the coding of pulse vectors as one number (`encode_pulses`, `decode_pulses`). | `celt/vq.c`, `celt/cwrs.c` |
| `src/mdct.rs` | `MdctLookup`: the forward and inverse MDCT. | `celt/mdct.c` |
| `src/kiss_fft.rs` | The complex fast Fourier transform (FFT) behind the MDCT, with a portable version, a NEON version (NEON: the vector instructions of 64-bit ARM) and a FEAT_FCMA version (see [The FFT and FEAT_FCMA](#the-fft-and-feat_fcma)). | `celt/kiss_fft.c` |
| `src/pitch.rs` | Pitch search for the pre-filter (`pitch_downsample`, `pitch_search`, `remove_doubling`), cross-correlations and inner products. | `celt/pitch.c`, `celt/pitch.h` |
| `src/celt_lpc.rs` | Linear prediction helpers of CELT: `lpc` and `autocorr`. | `celt/celt_lpc.c` |

### The SILK layer (`src/silk/`)

Encoder and decoder entry points:

| File | What it does | libopus counterpart |
|---|---|---|
| `enc_api.rs` | `SilkEncoder`: the SILK encoder of one Opus stream. It owns a state and an input resampler for each input channel, chooses the internal rate, resamples, and codes one packet with `silk_encode_packet` (mono or stereo, the low bitrate redundancy (LBRR) copy that carries FEC). It also holds `silk_encode_prefill` and the frame encoder `silk_encode_frame`. | `silk/enc_API.c`, `silk/fixed/encode_frame_FIX.c` |
| `dec_api.rs` | `SilkDecoder`: decodes a SILK packet, stereo to left/right, resampling to the output rate, FEC. | `silk/dec_API.c` |
| `structs.rs` | Encoder state: `SilkEncoderState`, `SilkEncoderStateCommon`, `SilkNSQState`, `SilkStereoState` and others. | `silk/structs.h`, `silk/fixed/structs_FIX.h` |
| `decoder_structs.rs` | Decoder state: `SilkDecoderState`, `SilkPLCState`, `SilkCNGState`, `SilkDecControl`. | `silk/structs.h`, `silk/control.h` |
| `define.rs` | Constants: frame lengths, limits, error codes. | `silk/define.h`, `silk/errors.h` |
| `tuning_parameters.rs` | Encoder tuning constants. | `silk/tuning_parameters.h` |
| `init_encoder.rs` | `silk_init_encoder`. | `silk/init_encoder.c` |
| `init_decoder.rs` | Decoder reset and the choice of internal rate. | `silk/init_decoder.c`, `silk/decoder_set_fs.c` |
| `control_codec.rs` | Encoder set-up for a rate, a packet duration and a complexity; LBRR set-up. | `silk/control_codec.c` |
| `control_fixed.rs` | Prediction coefficients, gain processing, long-term prediction (LTP) scaling. | `silk/fixed/find_pred_coefs_FIX.c`, `silk/fixed/process_gains_FIX.c`, `silk/fixed/LTP_scale_ctrl_FIX.c` |
| `control_snr.rs` | Target rate to signal-to-noise ratio. | `silk/control_SNR.c` |

Encoder analysis and quantization:

| File | What it does | libopus counterpart |
|---|---|---|
| `vad.rs` | Voice activity detection (VAD). | `silk/VAD.c` |
| `ana_filt_bank_1.rs` | The two-band split filter of the VAD. | `silk/ana_filt_bank_1.c` |
| `hp_variable_cutoff.rs` | Adapts the cutoff of the input high-pass filter to the pitch. | `silk/HP_variable_cutoff.c` |
| `lp_variable_cutoff.rs` | The low-pass filter that smooths a change of internal rate. | `silk/LP_variable_cutoff.c` |
| `pitch_analysis.rs` | Pitch lag search. | `silk/fixed/find_pitch_lags_FIX.c`, `silk/fixed/pitch_analysis_core_FIX.c` |
| `lpc_analysis.rs` | LPC analysis with Burg's method, residual energy, inverse prediction gain. | `silk/fixed/find_LPC_FIX.c`, `silk/fixed/burg_modified_FIX.c`, `silk/fixed/residual_energy_FIX.c`, `silk/LPC_inv_pred_gain.c` |
| `lpc_fit.rs` | Fits LPC coefficients into 16 bits. | `silk/LPC_fit.c` |
| `ltp_analysis.rs` | LTP analysis and filtering. | `silk/fixed/find_LTP_FIX.c`, `silk/fixed/LTP_analysis_filter_FIX.c` |
| `noise_shape_analysis.rs` | Noise-shaping filter design. | `silk/fixed/noise_shape_analysis_FIX.c` |
| `nsq.rs` | The noise-shaping quantizer (NSQ), used at complexity 0 and 1. | `silk/NSQ.c` |
| `nsq_del_dec.rs` | The delayed-decision noise-shaping quantizer, used from complexity 2. | `silk/NSQ_del_dec.c` |
| `gain_quant.rs` | Gain quantization and LTP gain quantization. | `silk/gain_quant.c`, `silk/quant_LTP_gains.c` |
| `vq_wmat_ec.rs` | Vector quantization of LTP coefficients. | `silk/VQ_WMat_EC.c` |
| `nlsf.rs` | Conversions between LPC coefficients and normalized line spectral frequencies (NLSF), NLSF stabilization, weights and processing. | `silk/A2NLSF.c`, `silk/NLSF2A.c`, `silk/NLSF_stabilize.c`, `silk/NLSF_VQ.c`, `silk/NLSF_VQ_weights_laroia.c`, `silk/process_NLSFs.c`, `silk/bwexpander_32.c` |
| `nlsf_encode.rs`, `nlsf_del_dec_quant.rs` | NLSF quantization. | `silk/NLSF_encode.c`, `silk/NLSF_del_dec_quant.c` |
| `stereo_lr_to_ms.rs` | Left/right to mid/side conversion with the stereo predictor. | `silk/stereo_LR_to_MS.c`, `silk/stereo_find_predictor.c`, `silk/stereo_quant_pred.c` |

Bitstream:

| File | What it does | libopus counterpart |
|---|---|---|
| `encode_indices.rs` | Writes the side information of a frame and the stereo predictor. | `silk/encode_indices.c`, `silk/stereo_encode_pred.c` |
| `encode_pulses.rs` | Writes the excitation pulses and their signs. | `silk/encode_pulses.c`, `silk/code_signs.c` |
| `shell_coder.rs` | The shell coder for pulse counts (encoder side). | `silk/shell_coder.c` |
| `decode_indices.rs` | Reads the side information and the stereo predictor. | `silk/decode_indices.c`, `silk/stereo_decode_pred.c` |
| `decode_pulses.rs` | Reads the pulses, with the shell decoder and the sign decoder. | `silk/decode_pulses.c`, `silk/shell_coder.c`, `silk/code_signs.c` |
| `nlsf_decode.rs`, `nlsf_unpack.rs` | NLSF decoding. | `silk/NLSF_decode.c`, `silk/NLSF_unpack.c` |
| `decode_parameters.rs` | Turns the indices into filter coefficients, pitch lags and gains. | `silk/decode_parameters.c`, `silk/decode_pitch.c` |

Decoder synthesis:

| File | What it does | libopus counterpart |
|---|---|---|
| `decode_frame.rs` | One frame: decode, or conceal. | `silk/decode_frame.c` |
| `decode_core.rs` | Excitation to speech: the LTP and LPC synthesis filters. | `silk/decode_core.c` |
| `plc.rs` | Packet loss concealment. | `silk/PLC.c` |
| `cng.rs` | Comfort noise generation (CNG). | `silk/CNG.c` |
| `stereo_ms_to_lr.rs` | Mid/side to left/right. | `silk/stereo_MS_to_LR.c` |

Signal processing and tables:

| File | What it does | libopus counterpart |
|---|---|---|
| `resampler.rs` | `SilkResampler`: the SILK resampler in both directions. `SilkResamplerDown1_3` and `SilkResamplerDown1_6` are extra fixed-ratio decimators with no libopus counterpart. | `silk/resampler.c`, `silk/resampler_down2.c`, `silk/resampler_down2_3.c`, `silk/resampler_private_*.c`, `silk/resampler_rom.c` |
| `sigproc_fix.rs` | Fixed-point helpers: autocorrelation, Schur recursion, reflection to prediction coefficients, energies, inner products, correlation matrices, sine window, warped autocorrelation, LPC analysis filter, cross-correlation, and the second-order filter of the input high-pass filter (`silk_biquad_alt_stride1`, `silk_biquad_alt_stride2`). | `silk/fixed/autocorr_FIX.c`, `schur_FIX.c`, `schur64_FIX.c`, `k2a_FIX.c`, `k2a_Q16_FIX.c`, `corrMatrix_FIX.c`, `apply_sine_window_FIX.c`, `warped_autocorrelation_FIX.c`, `vector_ops_FIX.c`; `silk/sum_sqr_shift.c`, `silk/inner_prod_aligned.c`, `silk/bwexpander.c`, `silk/LPC_analysis_filter.c`, `silk/biquad_alt.c` |
| `macros.rs` | The fixed-point arithmetic macros as functions (`silk_smulwb`, `silk_rshift_round`, `silk_div32_varq` and others). | `silk/SigProc_FIX.h`, `silk/macros.h`, `silk/Inlines.h` |
| `interpolate.rs`, `lin2log.rs`, `log2lin.rs`, `sigm.rs`, `sort.rs` | Small helpers. | `silk/interpolate.c`, `silk/lin2log.c`, `silk/log2lin.c`, `silk/sigm_Q15.c`, `silk/sort.c` |
| `tables.rs` | Probability and codebook tables. | `silk/tables_LTP.c`, `silk/tables_gain.c`, `silk/tables_other.c`, `silk/tables_pitch_lag.c`, `silk/tables_pulses_per_block.c`, `silk/table_LSF_cos.c`, `silk/pitch_est_tables.c` |
| `tables_nlsf.rs` | NLSF codebooks. | `silk/tables_NLSF_CB_NB_MB.c`, `silk/tables_NLSF_CB_WB.c` |
| `mod.rs` | Module list and re-exports. | none |

### Multistream (`src/multistream/`, needs `std`)

| File | What it does | libopus counterpart |
|---|---|---|
| `mod.rs` | Module documentation, `ChannelMappingTable` (the channel mapping table of an Ogg Opus header, RFC 7845 section 5.1.1), `split_self_delimited` (cuts one stream out of a multistream packet), and the internal types that copy interleaved samples into one stream's input (`Interleaved`, for `f32` and `i16`). | none |
| `layout.rs` | `ChannelLayout` and `MappingType`: streams, coupled streams, mapping, the layouts of mapping families 0, 1, 2 and 255. | `src/opus_multistream.c`, the surround layouts of `src/opus_multistream_encoder.c` |
| `encoder.rs` | `MultistreamEncoder` and `Bitrate`. | `src/opus_multistream_encoder.c` |
| `decoder.rs` | `MultistreamDecoder`. | `src/opus_multistream_decoder.c` |
| `rate_alloc.rs` | How the encoder divides the bitrate and the packet bytes among the streams. | `surround_rate_allocation`, `ambisonics_rate_allocation`, `rate_allocation` in `src/opus_multistream_encoder.c` |
| `surround.rs` | Surround masking analysis: how far each band of each channel lies above what the other channels mask. | `surround_analysis` in `src/opus_multistream_encoder.c` |

### What is not ported

[Differences from libopus](10-differences-from-libopus.md) lists these gaps together with the measured differences of the parts that are ported.

- The speech/music and bandwidth analysis of libopus (`src/analysis.c`, `src/mlp.c`), which libopus runs at complexity 7 and above for 16 to 48 kHz input.
- Discontinuous transmission (DTX).
- The projection encoder and decoder (`src/opus_projection_*.c`, `src/mapping_matrix.c`), which code ambisonics through a mixing matrix (mapping family 3). Mapping family 2, ambisonics without a matrix, is supported by the multistream code.
- The floating-point SILK encoder (`silk/float/`). `opus1` ports the fixed-point one.
- The neural-network parts of libopus, such as deep packet loss concealment and Deep REDundancy (DRED). libopus builds them only when asked to (`--enable-dred` and similar options).
- The scalable quality extension (QEXT), which libopus 1.6 builds only with `--enable-qext`.

## Main types and what they own

```text
OpusEncoder                                   src/lib.rs
├── celt_enc: CeltEncoder                     src/celt.rs
│     └── mode: &'static CeltMode             src/modes.rs, built once
├── silk: SilkEncoder                         src/silk/enc_api.rs
│     ├── mid: SilkChannel                    mono input, or the mid of stereo
│     ├── side: Option<SilkChannel>           the side, for stereo input only
│     │     (each SilkChannel: SilkEncoderState, SilkResampler, the packet
│     │      at the internal rate)
│     └── api_buf                             one input channel before resampling
├── rc: RangeCoder                            SILK and CELT write into it in turn
└── input and delay buffers: FixedVec<..>

OpusDecoder                                   src/lib.rs
├── celt_dec: CeltDecoder                     src/celt.rs
│     └── mode: &'static CeltMode
├── silk_dec: SilkDecoder                     src/silk/dec_api.rs
│     └── channel_state: [SilkDecoderState; 2]   each with its resampler, PLC and CNG state
└── output buffers: FixedVec<..>

MultistreamEncoder                            src/multistream/encoder.rs
├── layout: ChannelLayout
└── encoders: Vec<OpusEncoder>                one per stream; the first `coupled_streams` are stereo

MultistreamDecoder                            src/multistream/decoder.rs
├── layout: ChannelLayout
├── decoders: Vec<OpusDecoder>                one per stream
└── buf: Vec<f32>                             one stream's output, 120 ms of stereo
```

Notes:

- With the `heap` feature, `celt_enc`, `celt_dec`, `silk_dec`, each `SilkChannel`, `api_buf` and the large buffers each sit behind a `Box`. Without it they are stored inline (see [Feature flags and where state is stored](#feature-flags-and-where-state-is-stored)).
- A stereo encoder has two SILK channels even when its packets code mono, because the channel count of the packets can change from one packet to the next. A mono encoder has one. libopus always keeps two states (`state_Fxx[0]` and `state_Fxx[1]`).
- A Hybrid frame is one range-coded stream: SILK writes first, and CELT continues in the same `RangeCoder`. The decoder reads it in the same order.
- `CeltMode` is shared. `modes::default_mode()` builds it at the first call into a `static` `OnceCell` and returns `&'static CeltMode` after that. It is the only global value that is written at run time.

### The path of one frame

```text
encode / encode_i16  (8-48 kHz, 1 or 2 channels, interleaved)
  ├── decide_mode_and_bandwidth: mode, bandwidth, coded channels, FEC (once per packet)
  ├── CELT or Hybrid, 40/60 ms: encode_multiframe → two or three 20 ms frames → packet::write
  └── encode_frame
        ├── high-pass or DC filter                      hp_cutoff.rs
        ├── SILK-only or Hybrid: 16-bit samples
        │     └── SilkEncoder::encode: resample to 8/12/16 kHz → RangeCoder   silk/enc_api.rs
        ├── Hybrid or CELT-only: CeltEncoder → same RangeCoder        celt.rs
        ├── at a switch to or from CELT-only: a redundant 5 ms CELT frame
        └── TOC byte + range coder bytes → packet

decode / decode_i16 / decode_fec
  ├── empty packet → conceal in the previous mode (SILK PLC or CELT PLC)
  ├── packet::parse → frames
  └── decode_frame, for each frame
        ├── SILK-only or Hybrid: SilkDecoder → 16-bit samples at the output rate
        │     → f32, written into the output
        ├── Hybrid or CELT-only: CeltDecoder::decode reads on in the same range decoder
        │     → f32, written into the output (CELT-only) or added to it (Hybrid)
        ├── a redundant frame: cross-faded with the main frame
        └── decode_i16 only: pcm_soft_clip on a decoded packet, then rounding to 16 bits
```

## Floating point and fixed point

- **CELT is floating point.** `celt.rs`, `bands.rs`, `pvq.rs`, `mdct.rs`, `kiss_fft.rs`, `pitch.rs`, `celt_lpc.rs` and `quant_bands.rs` compute in `f32`, as the float build of libopus does. The bit allocation in `rate.rs` is integer arithmetic, as in libopus.
- **SILK is fixed point.** Everything under `src/silk/` works on `i16` and `i32` values in Q formats, as the fixed-point build of libopus (`silk/fixed/`) does. The float build of libopus uses the floating-point SILK encoder instead, so SILK packets from `opus1` and from a float build of libopus differ. Both are valid, and any Opus decoder reads them.
- **The Opus layer converts between them** in `src/lib.rs`:
  - Encoder: SILK needs 16-bit samples before its resampler. With `Application::Voip` they come from the integer high-pass filter of the fixed-point build: `hp_cutoff` takes `f32` input and writes `i16`. With the other applications the float signal is filtered and then rounded to 16 bits (`SilkInput::Float`). CELT gets the float signal with a 4 ms delay buffer in front, unless the application is `RestrictedLowDelay`.
  - Decoder: SILK's 16-bit output is scaled by 1/32768 to `f32` (`int16_to_res`) and written into the output. In a Hybrid frame CELT adds its output to it.
- `encode_i16` converts its input to float once, with `sample as f32 / 32768.0`, and then does what `encode` does. So `encode` and `encode_i16` write the same packets for the same audio. The conversion is exact, and the integer high-pass filter of `Voip` rounds the float samples back to the same 16-bit values.

## Feature flags and where state is stored

The crate has three features: `std` and `heap` (both on by default) and `libm`. [Using the library](06-using-the-library.md#feature-flags) describes them from the user's side. Inside the crate:

- **`std`** selects `#![no_std]` when it is off (`#![cfg_attr(not(feature = "std"), no_std)]` in `src/lib.rs`). It also enables the run-time processor checks in `src/compat.rs` and `src/kiss_fft.rs`, and the `multistream` module, which uses `Vec`.
- **`libm`** is required when `std` is off. `src/compat.rs` stops the build with `compile_error!` if both are missing. Its `Math` trait gives `f32` and `f64` the methods `sqrt`, `sin`, `exp2` and so on through the `libm` crate. Each module that needs them imports the trait under `#[cfg(not(feature = "std"))]`; with `std` the inherent methods of `f32` are used instead.
- **`heap`** changes `src/lib.rs` and `src/silk/enc_api.rs`. Each large field of `OpusEncoder` and `OpusDecoder` is declared twice:

  ```rust,ignore
  #[cfg(not(feature = "heap"))]
  celt_enc: CeltEncoder,
  #[cfg(feature = "heap")]
  celt_enc: Box<CeltEncoder>,
  ```

  The helper `state_mut` borrows such a field as the inner type in both builds, so the code that uses it needs no `#[cfg]`. `SilkEncoder` stores its channels and its input buffer in `Slot<T>`, which is `Box<T>` with `heap` and an inline value without it.

Below the Opus layer nothing allocates. The CELT, SILK and range coder states store every buffer inline, with a size fixed at compile time:

- SILK mostly uses plain arrays, sized by the largest frame (for example `x_buf: [i16; 2 * MAX_FRAME_LENGTH + LA_SHAPE_MAX]`).
- CELT, the FFT, the MDCT and the range coder use `FixedVec<T, N>` from `src/fixedvec.rs`. It stores up to `N` elements in `[MaybeUninit<T>; N]` and keeps a length. It offers the part of the `Vec` interface the codec uses (`push`, `resize`, `truncate`, `extend_from_slice`, slicing through `Deref`). Every operation that could go past the capacity checks this with `assert!`, and panics instead of going past it. It never drops its elements, so a `FixedVec` of a type with a destructor does not compile.
- The capacities are constants for the largest case: 2 channels, 120 ms at 48 kHz, 1276-byte packets (`OPUS_MAX_CHANNELS`, `OPUS_MAX_FRAME`, `OPUS_MAX_PACKET_BYTES` and others at the top of `src/lib.rs`). This is why the size of an encoder does not depend on its sampling rate or channel count.

The measured sizes of both builds are in [Using the library](06-using-the-library.md#size-of-the-encoder-and-the-decoder).

`src/compat.rs` also replaces what `no_std` without `alloc` lacks:

- `OnceCell<T>`: a value built once into inline storage, guarded by an atomic state (0 empty, 1 being built, 2 ready). It holds the `CeltMode`.
- `sort_by`: an insertion sort for the short slices the codec sorts, because the stable `sort` of slices needs `alloc`.

## Code for specific processors

The loops that take most of the run time have several versions. The rule is the same in every file:

1. A safe function is the only entry point, for example `pitch::inner_prod`. It checks the slice lengths, usually by slicing to the exact length (`&x[..n]`), so a short slice panics before any vector code runs.
2. On aarch64 it calls the NEON version. NEON is part of every aarch64 target, so there is no run-time check. These functions use `core::arch::aarch64` intrinsics; some carry `#[target_feature(enable = "neon")]`.
3. On x86 and x86-64 it checks for AVX or AVX2 (Advanced Vector Extensions) and FMA (fused multiply-add):
   - It asks `crate::compat::x86_has_avx()`, `x86_has_avx2()`, `x86_has_avx_fma()` or `x86_has_avx2_fma()`.
   - When the answer is yes, it calls an `unsafe fn` marked `#[target_feature(enable = "avx,fma")]` (or `"avx"`, `"avx2"`, `"avx2,fma"`).
   - Kernels that use FMA instructions run only when AVX and FMA are both present, because some processors have AVX without FMA.
   - Some kernels in `pitch.rs` and `celt.rs` also have an SSE version. SSE (Streaming SIMD Extensions, where SIMD means single instruction, multiple data) is the older set of x86 vector instructions, and every x86-64 processor has it.
4. Otherwise it runs the portable Rust version.

The check happens inside the public function, at every call. With `std` the probes use `std::arch::is_x86_feature_detected!`, which stores its answer after the first call. Without `std` they use `cfg!(target_feature = "...")`, which the compiler decides from the build flags.

Files with x86 versions: `bands.rs`, `celt.rs`, `kiss_fft.rs`, `mdct.rs`, `pitch.rs`, `pvq.rs`, `silk/sigproc_fix.rs`. Files with aarch64 versions: those, plus, under `src/silk/`, `decode_core.rs`, `lpc_analysis.rs`, `nlsf.rs`, `nsq_del_dec.rs`, `pitch_analysis.rs`, `resampler.rs` and `vq_wmat_ec.rs`.

`src/mathops.rs` also depends on the processor, for rounding rather than speed. On aarch64, `mac` and `mac_f64` compute `acc + a * b` with one fused multiply-add, which rounds once. The arm64 build of libopus does the same, because clang fuses such C expressions there. On other targets the product is rounded first.

For this release the x86 code was compiled but not run on an x86 processor (the README lists this among the known gaps). The aarch64 code runs in every test on Apple silicon.

### The FFT and FEAT_FCMA

FEAT_FCMA is an ARMv8.3 extension with two instructions for complex numbers: `fcmla` (complex multiply-add) and `fcadd` (complex add with a rotation by 90 or 270 degrees). Their Rust intrinsics are not stable, so `src/kiss_fft.rs` writes them with `core::arch::asm!` in the macros `cmul_fcma` and `pm_i_fcma`.

- The module `fused` builds the FFT stages twice from one macro, `kernels!`: `fused::fcma` (marked `#[target_feature(enable = "neon,fcma")]`, with `fcmla` and `fcadd`) and `fused::fma` (plain NEON with `fmla`). Both give the same bits.
- `has_fcma()` decides which one runs. It is `true` at compile time where the target includes FEAT_FCMA (Apple silicon). Elsewhere it asks `std::arch::is_aarch64_feature_detected!("fcma")` with `std`, and answers `false` without `std`.
- On aarch64 the FFT gives the bits of `opus_fft_impl_fused`, which match the arm64 build of libopus. On other targets it gives the bits of `opus_fft_impl_scalar`. Both functions are public, so tests can compare them on any machine.

## Unsafe code

- `src/lib.rs` and `src/multistream/` contain no `unsafe` code. The codec's control logic and the public types are safe Rust.
- `unsafe` appears in four kinds of places:
  - the kernels: vector intrinsics and raw pointer loads;
  - `FixedVec` and `OnceCell`, which manage uninitialized storage;
  - the range coder and `FixedVec`, which build large values in place to avoid copying them;
  - a few buffers that are written through `MaybeUninit`, to skip filling them with zeros first.
- Most `unsafe fn` declarations have a `# Safety` section that states what the caller must guarantee. Those without one are the register helpers `lo2`, `hi2` and `re_of_a_im_of_b` in `kiss_fft.rs` and `rev4` in `mdct.rs`, `forward_avx` and `backward_avx` in `mdct.rs`, `l1_metric_neon` in `celt.rs`, and the four AVX2 kernels in `src/silk/sigproc_fix.rs`.
- In library code (outside the test modules), every `unsafe` block has a `// SAFETY:` comment that says why the block is sound. Some comments cover several blocks, for example `// SAFETY (all three blocks): ...` in `src/pitch.rs`.
- Kernels check their preconditions where they are cheap: `assert!` in entry points (for example `opus_fft_impl` checks the buffer length), `debug_assert!` inside many `unsafe fn` kernels, and the `assert!` checks of `FixedVec`.
- The crate root allows `unsafe_op_in_unsafe_fn`, so an `unsafe fn` body may call unsafe operations without an inner `unsafe` block. Many functions still use inner blocks with their own comments.

## Error handling

- Every public function that can fail returns `Result<T, &'static str>`. There is no error enum. The strings are fixed messages such as `"Invalid packet"` or `"Output buffer too small for packet"`. [Using the library](06-using-the-library.md#errors) lists what the encoder and decoder reject.
- The SILK functions keep the libopus convention and return `i32` codes: 0 (`SILK_NO_ERROR`) for success, and negative codes such as `SILK_ENC_FS_NOT_SUPPORTED` (-102) otherwise. `SilkEncoder::encode` (`src/silk/enc_api.rs`) turns a non-zero code into a message, such as `"SILK encoding failed"`.
- The CELT encoder reports nothing itself. The range coder records an overflow in its `error` field, and `src/lib.rs` checks that field and the bit count after each frame.
- Input is checked before state changes. The decoder parses the packet and checks the buffer sizes before it updates its state, so a rejected packet leaves the decoder unchanged.
- Settings that are public fields (`bitrate_bps`, `complexity`, `packet_loss_perc`) are clamped at the next `encode`, not rejected.

## Tests, examples, benchmarks and fuzzing

### `tests/`

The repository holds 228 test files in `tests/`, plus helper modules in subdirectories. More than half of them run the same input through `opus1` and through C libopus. They compare packets, decoded samples or internal values. libopus is linked through the `opus` crate and `audiopus_sys` (see [building without cmake](06-using-the-library.md#building-and-testing-on-a-machine-without-cmake)). [Testing](09-testing.md) explains what the tests check and how to run each part. The file name prefix says what a test covers:

| Prefix | Files | What the tests cover |
|---|---:|---|
| `perfk_` | 33 | Single kernels after performance work, against the same function inside libopus or a reference version: FFT, PVQ, range coder, pitch, SILK analysis and quantizer. Four `.expected` files hold reference output. |
| `multistream_` | 31 | The multistream encoder and decoder: `multistream_c_*` against libopus's own functions, `multistream_enc_*` and `multistream_dec_*` for each side, `multistream_e2e_*` end to end, `multistream_surround_*` for the surround masking analysis. |
| `core_` | 22 | `OpusEncoder` and `OpusDecoder` entry points and controls against libopus: loss, FEC, mode switches, channel changes, limits. |
| `k24_` | 21 | The 24 kHz path: encoder modes, Hybrid super-wideband, SILK internals and loss, and the decoder matrix of every mode and bandwidth at every output rate. |
| `celtenc_`, `celtdec_`, `celt_` | 16, 4, 8 | The CELT encoder and decoder. |
| `silkenc_`, `silkdec_`, `silk_` | 6, 10, 14 | The SILK encoder and decoder. |
| `mdct_` | 16 | The MDCT. |
| `issue_`, `issues_` | 15 | Regression tests named after GitHub issues, each reproducing the reported fault. |
| `verify_`, `pvq_`, `hybrid_` | 5, 4, 4 | Checks of band, PVQ, quantization and SILK functions, and of Hybrid mode. |
| others | 19 | Single topics, for example `stack_usage_test.rs` (struct sizes and stack use), `panic_surface_repro.rs` (inputs that once panicked), `quality_regression_test.rs`, `proptest_encode_indices.rs` (property-based tests with `proptest`). |

Helper directories are named after the tests that use them: `celtdec_common/`, `celtenc_common/`, `core_common/`, `k24_common/`, `k24_enc_common/`, `ms_support/`, `multistream_dec_common/`, `multistream_e2e_common/`, `multistream_enc_common/`, `silkdec_common/`, `silkenc_common/`. Each holds a `mod.rs` that a test includes with `mod <name>;`. Several of them wrap the raw C application programming interface (API) of libopus (`audiopus_sys`) for the controls the `opus` crate does not offer, such as forcing the coding mode.

Tests marked `#[ignore]` either print measurement tables and assert nothing, or record a known difference from libopus. Run one with `cargo test --release --test <name> -- --ignored --nocapture`.

Unit tests also live inside the source files, in `#[cfg(test)]` modules. Many of them compare a vector kernel with its portable version.

### `examples/`

- `multistream.rs`: encodes and decodes two seconds of a 5.1 test signal and prints the layout, packet sizes and signal-to-noise ratio of every channel.
- `wav_test.rs`: encodes and decodes a WAV (Waveform Audio File Format) file (by default `fixtures/answer_16k.wav`) and writes the decoded files into `fixtures/`.
- `perf_harness.rs`: the timing and comparison tool against libopus. Its commands are `list`, `bench`, `run`, `golden`, `modes`, `settings`, `signals`, `resampler` and `spread`. `cargo run --release --example perf_harness -- bench` produces the timings in the README.

### `benches/`

`benches/opus_bench.rs` holds Criterion benchmarks (`cargo bench`) of whole encoding (SILK, CELT, 16-bit input) and of single kernels: FFT, MDCT, PVQ, the SILK noise-shaping quantizer, pitch analysis, autocorrelation, Burg's method and others. The script `bench_quick.sh` at the repository root runs a shorter `perf_harness bench` (6 repetitions in one round) and saves its table to a Markdown file. It takes a label and an optional scenario filter.

### `fuzz/`

`fuzz/` is a separate Cargo package of 18 targets for libFuzzer. `fuzz/fuzz.sh` builds them with the stable toolchain and runs them. Five targets compare opus1 with libopus: `fuzz_packet`, `fuzz_decoder_sequence`, `fuzz_multistream_decoder`, `fuzz_multistream_encoder` and `fuzz_encoder_settings`. Their wrappers of the libopus C functions are in `fuzz/fuzz_targets/common/mod.rs`. The other 13 targets run opus1 alone. `fuzz/README.md` and [Testing](09-testing.md#fuzzing) describe what each target checks.

### `scripts/`

`scripts/check_no_std.sh` builds the crate without `std` for `thumbv7em-none-eabi` and `wasm32-unknown-unknown` (when installed) and for the host, then runs the test suite without `std`.
