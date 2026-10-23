# Architecture

This document describes how the source code of `opus1` is organised, for maintainers and for readers who want to find the code behind a feature. It lists every file under `src/` with the libopus file it ports. It shows which types own which, and explains where the code uses floating point and where it uses fixed point. It then covers the feature flags and how they change where state is stored, the code written for specific processors, and the rules for `unsafe` code. The last sections cover error handling, and the layout of the tests, examples, benchmarks and fuzz targets. How to call the library is in [Using the library](06-using-the-library.md), and the [documentation index](README.md) lists the other documents.

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

### The public interface and `internals`

The public interface is the crate root and three modules:

- At the crate root: `OpusEncoder`, `OpusDecoder`, `SampleRate`, `Channels`, `Application`, `Bandwidth`, `Bitrate`, `ForcedMode`, `FrameDuration`, `Signal`, `InbandFec`, `Error`, `ErrorKind`, `pcm_soft_clip`, `VERSION` and `LIBOPUS_VERSION`. With `deep-plc` (or a feature that turns it on) also `DnnBlob` and `DnnModels` (see [Neural network code](#neural-network-code-srcdnn)), and with `dred` also `OpusDred`, `OpusDredDecoder`, `DredParse` and `DredProcessing`.
- `packet`: reading, writing and padding packets.
- `extensions`: the packet extensions stored in the padding of a packet.
- `multistream`: the multistream encoder and decoder and their channel layouts (feature `alloc`).

Every other module is private. The tests, benchmarks, examples and fuzz targets of the repository reach internal items through the module `opus1::internals` (`src/internals.rs`). That module exists only with the cargo feature `__internals`, which is for those tests only and outside semantic versioning. It is also hidden from the documentation. The tests, benchmarks and examples get the feature through a development dependency of opus1 on itself (`Cargo.toml`), and the fuzz targets through `fuzz/Cargo.toml`. Each module inside `internals` has the name and the place of an internal module, and re-exports only the items that code outside the crate uses, for example `opus1::internals::range_coder::RangeCoder`. So in a build with `__internals`, such as `cargo test`, the compiler reports an internal item that neither the codec nor a test uses as dead code. A build without it does not check that: there every item that only `internals` exposes is unused, so `src/lib.rs` allows dead code in that build.

### The Opus layer and shared code

| File | What it does | libopus counterpart |
|---|---|---|
| `src/lib.rs` | `OpusEncoder` and `OpusDecoder`: mode and bandwidth decisions, in-band forward error correction (FEC) decisions, mode switches with redundant CELT frames, packets of 40 to 120 milliseconds (ms), discontinuous transmission (DTX), packet loss, `pcm_soft_clip`. The encoder runs the speech and music analysis of `src/analysis/` and hands its result to these decisions and to the CELT encoder. | `src/opus_encoder.c`, `src/opus_decoder.c`, `opus_pcm_soft_clip` in `src/opus.c` |
| `src/internals.rs` | The internal items that the tests, benchmarks, examples and fuzz targets use, with the feature `__internals` (see above). | none |
| `src/packet.rs` | Reading, writing and padding packets; the repacketizer, which carries packet extensions; self-delimited framing; checks of multistream packets. | `opus_packet_parse_impl` in `src/opus.c`, `src/repacketizer.c`, `opus_multistream_packet_validate` in `src/opus_multistream_decoder.c` |
| `src/extensions.rs` | Packet extensions in the padding of a packet: reading them (`ExtensionIterator`, `count`, `parse`) and writing them (`generate`). | `src/extensions.c` |
| `src/hp_cutoff.rs` | The encoder's input filters: the high-pass filter for speech (`hp_cutoff`, `hp_cutoff_float`) and, for other audio, the filter that removes the constant (DC, direct current) part of the signal (`dc_reject_float`). The integer filters use the second-order filter of `src/silk/sigproc_fix.rs`. | `hp_cutoff` and `dc_reject` in `src/opus_encoder.c` |
| `src/range_coder.rs` | `RangeCoder`: the range encoder and decoder, bit counting (`tell`, `tell_frac`), Laplace-distributed symbols. | `celt/entenc.c`, `celt/entdec.c`, `celt/entcode.c`, `celt/laplace.c` |
| `src/fixedvec.rs` | `FixedVec<T, N>`: a growable buffer with a fixed capacity, stored inline. See [Feature flags and where state is stored](#feature-flags-and-where-state-is-stored). | none |
| `src/compat.rs` | `OnceCell` (a value built once, without the heap), a stable insertion sort (`sort_by`), the x86 feature probes (`x86_has_avx` and others), and the `Math` trait that maps float functions to `libm` without `std`. | none |
| `src/deep_plc.rs` | `DeepPlcArgs`: what `OpusDecoder` passes down to the SILK and CELT decoders so that they can run deep packet loss concealment (the state, the models and the kernel variant). Without the feature `deep-plc` it is a type without values, so the argument is always `None`. | the `LPCNetPLCState *lpcnet` argument of `silk_Decode`, `silk_decode_frame`, `silk_PLC` and `celt_decode_with_ec_dred` |
| `src/osce_args.rs` | `OsceArgs`: what `OpusDecoder` passes to the SILK decoder for each frame so that it can run OSCE: the method (LACE, NoLACE or none), the extended mode (whether the bandwidth extension replaces the resampler) and the layers. Without the feature `osce` it is a type without values. | the fields `osce_method`, `osce_extended_mode` of `silk_DecControlStruct` and the `OSCEModel` argument of `silk_decode_frame` |
| `src/mathops.rs` | Float helpers that several modules share: `mac` and `mac_f64` (multiply-add, rounded as libopus rounds it on the target processor), `celt_log2` and `celt_exp2`. | `celt/mathops.h`, `celt/mathops.c` |

### The speech and music analysis (`src/analysis/`)

For each 20 ms of input, the analysis estimates how tonal the signal is, how likely it is music rather than speech, how likely it holds any sound, and which audio bandwidth it really uses. The encoder runs it from complexity 7 on, for input at 16, 24 or 48 kHz, as the floating-point build of libopus does. [The encoder](03-encoder.md#13-the-speech-and-music-analysis) describes what it computes and where the encoder uses it.

| File | What it does | libopus counterpart |
|---|---|---|
| `src/analysis/mod.rs` | `AnalysisInfo` (the result for one frame), the description of the caller's samples (`PcmRef`, `SecondChannel`, `AnalysisPcm`), `AnalysisGate` (when the encoder runs the analysis, an internal setting for tests) and `is_digital_silence`. These items are compiled in every build, so the encoder can carry an invalid `AnalysisInfo` without the feature `analysis`. | `AnalysisInfo` in `celt/celt.h`, `is_digital_silence` in `src/opus_encoder.c` |
| `src/analysis/tonality.rs` | `TonalityAnalysisState`: the downmix and the resampling to 24 kHz, the 480-point FFT of each block, the tonality of each FFT bin from its phase, the band energies, the leakage between bands, the bandwidth, the 25 features of the network (`tonality_analysis`), and the result for the frame being coded (`get_info`, libopus `tonality_get_info`). It also keeps the square roots of the band energies and the distances between the stored blocks, so that a block does not compute them again. | `src/analysis.c` |
| `src/analysis/mlp.rs` | The network: a dense layer of 32 outputs, a gated recurrent unit (GRU) of 24 outputs, and a dense layer of 2 outputs. | `src/mlp.c` |
| `src/analysis/mlp_data.rs` | The int8 weights of the network, written by `scripts/gen_analysis_mlp_data.py` from libopus's file. `mlp.rs` turns them into `f32` tables when the crate is compiled. | `src/mlp_data.c` |
| `src/analysis/tables.rs` | The analysis window, the table of the cepstral transform and the band edges. | the tables of `src/analysis.c` |
| `src/analysis/neon.rs` | NEON versions of the downmix, the windowing of the FFT input, the tonality of the FFT bins (four bins per vector), the recurrent layer and the activations of the network (64-bit ARM only). They give the bits of the scalar code, which unit tests check. | none |

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
| `src/pitch.rs` | Pitch search for the pre-filter and the loss concealment (`pitch_downsample`, `pitch_search`, `remove_doubling`), cross-correlations and inner products (`celt_inner_prod`). | `celt/pitch.c`, `celt/pitch.h` |
| `src/celt_lpc.rs` | Linear prediction of CELT: `lpc`, `autocorr` and the two filters of the loss concealment (`celt_fir`, `celt_iir`). | `celt/celt_lpc.c` |

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

### Evaluation tools (`src/eval/`, needs `std`)

These files are not part of the codec. They port libopus's test programs, so that tests and the harness `examples/eval` can measure the decoded audio. They are reached as `opus1::internals::eval` (see [Testing](09-testing.md#evaluation-tools)).

| File | What it does | libopus counterpart |
|---|---|---|
| `mod.rs` | Module documentation and re-exports. | none |
| `compare.rs` | `opus_compare` and `opus_compare_many`: the weighted spectral error of the conformance test, bit for bit as the C program built with clang. | `src/opus_compare.c` |
| `moc.rs` | `moc`, `moc_i16`, `moc_relative`: the MOC distance of the Opus Speech Coding Enhancement draft. | `moc.py` of the libopus git repository (`dnn/torch/osce/stndrd/evaluation/`) |
| `demo.rs` | `parse_bitstream`, `decode_bitstream`, the `DemoDecoder` trait and `opus_demo_s16`: the decoding loop of `opus_demo -d` and its 16-bit output. | `src/opus_demo.c` |

### Multistream (`src/multistream/`, needs `alloc`)

| File | What it does | libopus counterpart |
|---|---|---|
| `mod.rs` | Module documentation, `ChannelMappingTable` (the channel mapping table of an Ogg Opus header, RFC 7845 section 5.1.1), `split_self_delimited` (cuts one stream out of a multistream packet), and the internal types that copy interleaved samples into one stream's input (`Interleaved`, for `f32` and `i16`). | none |
| `layout.rs` | `ChannelLayout` and `MappingType`: streams, coupled streams, mapping, the layouts of mapping families 0, 1, 2 and 255. | `src/opus_multistream.c`, the surround layouts of `src/opus_multistream_encoder.c` |
| `encoder.rs` | `MultistreamEncoder`. | `src/opus_multistream_encoder.c` |
| `decoder.rs` | `MultistreamDecoder`. | `src/opus_multistream_decoder.c` |
| `rate_alloc.rs` | How the encoder divides the bitrate and the packet bytes among the streams. | `surround_rate_allocation`, `ambisonics_rate_allocation`, `rate_allocation` in `src/opus_multistream_encoder.c` |
| `surround.rs` | Surround masking analysis: how far each band of each channel lies above what the other channels mask. | `surround_analysis` in `src/opus_multistream_encoder.c` |

### Neural network code (`src/dnn/`)

libopus 1.5 and later run small neural networks in the decoder: deep packet loss concealment (deep PLC) with the networks PitchDNN, PLC and FARGAN, the speech enhancers LACE and NoLACE, and the bandwidth extension (BWE) with the network BBWENet. LACE, NoLACE and the BWE make up OSCE (Opus Speech Coding Enhancement). libopus keeps this code in its folder `dnn/`. opus1 compiles `src/dnn/` only with the feature `dnn`, which the features `deep-plc`, `osce` and `osce-bwe` turn on (see [Feature flags and where state is stored](#feature-flags-and-where-state-is-stored)).

`src/dnn/` holds what the networks share: the layer functions, their kernels, the weights and their loader, and the signal processing that the networks' input needs. It also holds the models of deep PLC (`plc/`) and LACE, NoLACE and the bandwidth extension (`osce/`). The decoder runs all of them (see [Deep PLC in the decoder](#deep-plc-in-the-decoder) and [OSCE in the decoder](#osce-in-the-decoder)). `src/dnn/` computes the bits of libopus 1.6.1 for 64-bit ARM, built without loop vectorization (see [Code for specific processors](#code-for-specific-processors)).

| File | What it does | libopus counterpart |
|---|---|---|
| `mod.rs` | Module list, `Activation` (the activation functions of a layer) and the size limits of the layer functions. It stops a build for a big-endian target, because the weight blobs are little-endian. | `dnn/nnet.h` |
| `nnet.rs` | `LinearLayer` and `Conv2dLayer`, and the layer functions: `compute_linear`, `compute_activation`, `compute_generic_dense`, `compute_generic_gru` (gated recurrent unit), `compute_glu` (gated linear unit), `compute_generic_conv1d` with and without dilation, `compute_conv2d`. They check their slices with `assert!`, then run the kernels of the chosen `DnnArch`. | `dnn/nnet.c`, `dnn/nnet_arch.h` |
| `arch.rs` | `DnnArch`, the kernel variants: `Portable`, `Neon`, `NeonDotprod` (the `sdot` instruction of 64-bit ARM), and on x86-64 `Avx2Fma`, `AvxVnni`, `Avx512Vnni`. `detect()` picks the best one the processor runs; `usable()` replaces a variant it cannot run. | `opus_select_arch`, `dnn/arm/arm_dnn_map.c`, `dnn/x86/x86_dnn_map.c` |
| `vec_portable.rs` | The kernels in plain Rust: products of int8 and float weights (dense and sparse), the activations, the quantization of inputs to int8, and a software copy of the ARM reciprocal estimate (FRECPE). It is the specification that the other variants must match. | `dnn/vec_neon.h` (its numerics), `dnn/vec.h` |
| `vec_neon.rs` | The NEON and DotProd kernels, with intrinsics. | `dnn/vec_neon.h`, `dnn/arm/nnet_neon.c`, `dnn/arm/nnet_dotprod.c` |
| `vec_x86.rs` | The AVX2, AVX-VNNI and AVX-512 VNNI kernels (VNNI: vector neural network instructions). They compute the bits of the 64-bit ARM kernels, not those of libopus's x86 kernels. | none (libopus's `dnn/vec_avx.h` rounds differently) |
| `weights.rs` | `WeightList`, the parser of weight blobs, and `linear_init` and `conv2d_init`, which build a layer from the arrays of a list. They refuse what libopus refuses, and more (see [Differences from libopus](10-differences-from-libopus.md#neural-network-weights)). | `dnn/parse_lpcnet_weights.c` |
| `models/` | One file per network with its layer struct, its `init` function, the constants of its C header and `LAYERS`, a table of the arrays and the shape of each layer. `tools/dnn-gen` writes these files. `models/mod.rs`, written by hand, has the shared types and `ModelSet`, which loads the four groups (deep PLC, LACE, NoLACE, BBWENet) one by one. | `dnn/*_data.h`, the `init_*` functions of `dnn/*_data.c` |
| `builtin.rs` | The weights of libopus 1.6.1, embedded with `include_bytes!` in 64-byte aligned statics, each by the feature that needs it. `weights()` parses them once and `models()` builds their layers once per process. | the arrays of `dnn/*_data.c` |
| `holder.rs` | `DnnBlob` (a parsed blob: the built-in weights, a `'static` slice, or a heap copy), `DnnModels` (a set of groups) and `DecoderDnn`, what an `OpusDecoder` stores: the kernel variant and the layers of each loaded group. | `OPUS_SET_DNN_BLOB` in `src/opus_decoder.c` |
| `freq.rs` | The feature analysis of deep PLC: band energies, the Burg cepstrum, the discrete cosine transform (DCT) and the LPC (linear prediction) coefficients from a cepstrum, on a 320-point FFT. | `dnn/freq.c` |
| `burg.rs` | Burg's method for LPC in floating point (`silk_burg_analysis`), a copy of SILK's float version that deep PLC uses. | `dnn/burg.c` |
| `lpcnet_tables.rs` | The tables of `freq.rs`: the plan of the 320-point FFT (bit reversal, twiddles, factors), the half window and the DCT table. | `dnn/lpcnet_tables.c` |
| `plc/features.rs` | The feature analysis of deep PLC: 36 values per 10 ms of 16 kHz audio (cepstrum, pitch, pitch correlation, LPC coefficients), with the state `LpcnetEncState`. It has its own copies of the CELT helpers it needs (`celt_inner_prod`, `celt_pitch_xcorr`, `celt_fir`, `celt_log2`), with the rounding of 64-bit ARM on every target. | `dnn/lpcnet_enc.c` |
| `plc/pitchdnn.rs` | PitchDNN, the pitch estimator, with its state `PitchDnnState`. | `dnn/pitchdnn.c` |
| `plc/fargan.rs` | FARGAN, the vocoder that turns 20 features into 10 ms of audio (`fargan_cont`, `fargan_synthesize`, `fargan_synthesize_int`), with its state `FarganState`. | `dnn/fargan.c` |
| `plc/lpcnet_plc.rs` | `DeepPlc`, the state of deep PLC for one decoder: 150 ms of history at 16 kHz, the deferred analysis of good frames, the PLC network (`compute_plc_pred`), the attenuation of long losses, and the queue of DRED features. `update_from_celt` feeds it with CELT history (libopus `update_plc_state`). The weights and the `DnnArch` are arguments of `conceal`; the state holds no weights. | `dnn/lpcnet_plc.c`, `update_plc_state` of `celt/celt_decoder.c` |
| `osce/mod.rs` | The entry points of OSCE for the SILK decoder: `OsceFrame` (the values of one decoded SILK frame that OSCE reads), `OsceChannel` and `osce_enhance_frame` (LACE or NoLACE on one 20 ms frame at 16 kHz, with the reset and bypass rules), `OsceBweChannel` and `osce_bwe` (16 to 48 kHz), `osce_reset`, `osce_bwe_reset`, the two 10 ms cross-fades, `OsceMethod`, `OsceExtMode` and `OsceModels` (the layers a decoder holds). The module documentation lists the rules. | `dnn/osce.c` (`osce_enhance_frame`, `osce_bwe`, the resets), `dnn/osce.h`, `dnn/osce_structs.h` |
| `osce/features.rs` | The 93 features per 5 ms of LACE and NoLACE (LPC spectrum, cepstrum, correlation at the pitch lag, LTP coefficients, log gain), the 114 features per 10 ms of the bandwidth extension, and the cross-fades. | `dnn/osce_features.c`, `dnn/osce_config.h` |
| `osce/nndsp.rs` | The adaptive filters: AdaConv (convolution), AdaComb (comb filter at the pitch lag), AdaShape (temporal envelope), and the overlap windows as tables. | `dnn/nndsp.c` |
| `osce/lace.rs`, `osce/nolace.rs`, `osce/bbwenet.rs` | The three models: feature network, filter stages, state per channel. `bbwenet.rs` also has the resamplers (2x all-pass, 3/2 interpolation) and the non-linear function of BBWENet. It runs all steps for one 5 ms subframe before the next one, which gives libopus's results with smaller buffers. | `dnn/osce.c` |
| `osce/math.rs` | Fused multiply-adds, the calls into the C math library, and `celt_log2`, `celt_log`, `celt_cos_norm2` and `celt_sin` with the roundings of the 64-bit ARM build on every target. | `celt/mathops.h`, `celt/float_cast.h` |

The weights are files of the repository, not Rust code:

- `dnn-weights/` holds one blob per network in libopus's blob format (`pitchdnn.bin`, `plc.bin`, `fargan.bin`, `dred_rdovae_enc.bin`, `dred_rdovae_dec.bin`, `lace.bin`, `nolace.bin`, `bbwenet.bin`), their SHA-256 sums (Secure Hash Algorithm, 256 bits; file `SHA256SUMS`), the libopus version and the sums of the C files they come from (`VERSION`), and libopus's licence (`COPYING`).
- `tools/dnn-gen/` is a separate Cargo package, not published. It reads `dnn/*_data.c` and `dnn/*_data.h` of a libopus release and writes `dnn-weights/`, `src/dnn/models/<network>.rs` and `tests/dnn_gen/mod.rs` (the C structs and symbols that the tests against libopus use). Run again on the same release, it writes the same files. Its `README.md` explains how to update the weights to a later release.

The feature `dred` (deep redundancy) adds the two DRED networks, the RDOVAE encoder and decoder (`models/dred_rdovae_enc.rs`, `models/dred_rdovae_dec.rs`), the constants of the RDOVAE and the tables of DRED's quantizer (`models/dred_rdovae_stats.rs`, also generated), and the encoder and decoder sides of DRED in `dred/`. `ModelSet` does not load the DRED networks: in libopus the encoder and the DRED decoder hold them, not the Opus decoder.

- `builtin::rdovae_dec()` builds the built-in RDOVAE decoder once per process. `holder::DredDecoderDnn` holds it, or one loaded from a blob, for an `OpusDredDecoder`.
- `builtin::dred_enc()` builds the models of the DRED encoder once per process: the RDOVAE encoder and PitchDNN (`models::DredEncModels`), which load together as in libopus. `holder::DredEncoderDnn` holds them, or those loaded from a blob, for an `OpusEncoder`.

| File | What it does | libopus counterpart |
|---|---|---|
| `dred/mod.rs` | The constants of `dnn/dred_config.h`. | `dnn/dred_config.h` |
| `dred/decoder.rs` | `OpusDredDecoder` (the weights and the kernel variant), `OpusDred` (the DRED data of one packet: initial state, latent vectors, features), `DredParse`, the search for the DRED extension, and the reading of the range-coded payload. | `OpusDREDDecoder`, `opus_dred_parse`, `opus_dred_process`, `dred_find_payload` of `src/opus_decoder.c`; `dnn/dred_decoder.c` |
| `dred/rdovae_dec.rs` | The RDOVAE decoder network: the GRU states from the initial state, then four frames of features per latent vector. | `dnn/dred_rdovae_dec.c` |
| `dred/coding.rs` | `compute_quantizer`, the quantizer level of each latent vector. | `dnn/dred_coding.c` |
| `dred/encoder.rs` | `DredEncoder`, the DRED part of an `OpusEncoder`: the share of the bitrate, the conversion to 16 kHz, the feature analysis and the RDOVAE encoder every 20 ms, the latent vectors and the voice activity of the last second, and the quantized DRED data of a packet. | `dnn/dred_encoder.c`; `compute_dred_bitrate`, `estimate_dred_bitrate` and the DRED fields of `src/opus_encoder.c` |
| `dred/rdovae_enc.rs` | The RDOVAE encoder network: one latent vector and one initial state from the features of two 10 ms frames. | `dnn/dred_rdovae_enc.c` |

The range coder has four functions for DRED: `RangeCoder::decode_icdf16` and `laplace_decode_p0` for the decoder, `encode_icdf16` and `laplace_encode_p0` for the encoder (libopus `ec_dec_icdf16`, `ec_laplace_decode_p0`, `ec_enc_icdf16`, `ec_laplace_encode_p0`). The Opus encoder calls the DRED encoder from `encode_impl`, `encode_frame`, `finish_frame` and `encode_multiframe` in `src/lib.rs` ([The encoder](03-encoder.md#18-deep-redundancy-dred)).

### What is not ported

[Differences from libopus](10-differences-from-libopus.md) lists these gaps together with the measured differences of the parts that are ported.

- The projection encoder and decoder (`src/opus_projection_*.c`, `src/mapping_matrix.c`), which code ambisonics through a mixing matrix (mapping family 3). Mapping family 2, ambisonics without a matrix, is supported by the multistream code.
- The floating-point SILK encoder (`silk/float/`). `opus1` ports the fixed-point one.
- The scalable quality extension (QEXT), which libopus 1.6 builds only with `--enable-qext`.
- Opus Custom (`include/opus_custom.h`), which codes CELT at rates and frame sizes that standard Opus does not have. libopus builds it only with `--enable-custom-modes` or `--enable-opus-custom-api`.

The other optional parts of libopus are ported: deep packet loss concealment, LACE, NoLACE and the bandwidth extension, which run in the decoder, and Deep REDundancy (DRED) in the encoder and the decoder (see [Neural network code](#neural-network-code-srcdnn)). libopus builds them only with `--enable-deep-plc`, `--enable-osce` and `--enable-dred`.

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
├── analysis: TonalityAnalysisState          src/analysis/tonality.rs (feature `analysis`)
├── rc: RangeCoder                            SILK and CELT write into it in turn
└── input and delay buffers: FixedVec<..>

OpusDecoder                                   src/lib.rs
├── celt_dec: CeltDecoder                     src/celt.rs
│     └── mode: &'static CeltMode
├── silk_dec: SilkDecoder                     src/silk/dec_api.rs
│     ├── channel_state: [SilkDecoderState; 2]   each with its resampler, PLC and CNG state,
│     │                                       and with `osce` its OSCE state (`osce`) and
│     │                                       with `osce-bwe` its BWE state (`osce_bwe`)
│     └── prev_osce_extended_mode             with `osce-bwe`: the mode of the last SILK frame
├── output buffers: FixedVec<..>
├── complexity, osce_bwe                      decoder settings; a reset keeps them
├── dnn: DecoderDnn                           src/dnn/holder.rs, with a neural network feature
│     ├── arch: DnnArch                       chosen once, when the decoder is built
│     └── one slot per group of networks      &'static to the shared built-in layers,
│                                             or layers built from a blob plus a clone of the blob
└── deep_plc: DeepPlc                         src/dnn/plc/lpcnet_plc.rs, with `deep-plc`:
                                              150 ms of history at 16 kHz, the analysis,
                                              the PLC network and FARGAN states

MultistreamEncoder                            src/multistream/encoder.rs
├── layout: ChannelLayout
└── encoders: Vec<OpusEncoder>                one per stream; the first `coupled_streams` are stereo

MultistreamDecoder                            src/multistream/decoder.rs
├── layout: ChannelLayout
├── decoders: Vec<OpusDecoder>                one per stream
└── buf: Vec<f32>                             one stream's output, 120 ms of stereo
```

Notes:

- With the `heap` feature, `celt_enc`, `celt_dec`, `silk_dec`, `analysis`, each `SilkChannel`, `api_buf` and the large buffers each sit behind a `Box`. Without it they are stored inline (see [Feature flags and where state is stored](#feature-flags-and-where-state-is-stored)).
- A stereo encoder has two SILK channels even when its packets code mono, because the channel count of the packets can change from one packet to the next. A mono encoder has one. libopus always keeps two states (`state_Fxx[0]` and `state_Fxx[1]`).
- A Hybrid frame is one range-coded stream: SILK writes first, and CELT continues in the same `RangeCoder`. The decoder reads it in the same order.
- `CeltMode` is shared. `modes::default_mode()` builds it at the first call into a `static` `OnceCell` and returns `&'static CeltMode` after that. It is the only global value that is written at run time, apart from the layers of the built-in networks: with a neural network feature, `dnn::builtin::models()` builds them the same way when the first decoder is built, and every decoder points at them.

### The path of one frame

```text
encode / encode_i16  (8-48 kHz, 1 or 2 channels, interleaved)
  ├── is_digital_silence; at complexity 7-10 and 16-48 kHz the speech and
  │   music analysis of the caller's buffer                   analysis/
  ├── decide_mode_and_bandwidth: mode, bandwidth, coded channels, FEC (once per packet)
  ├── CELT or Hybrid 40-120 ms, SILK 80-120 ms: encode_multiframe → two to six parts → packet::write
  └── encode_frame
        ├── high-pass or DC filter                      hp_cutoff.rs
        ├── SILK-only or Hybrid: 16-bit samples
        │     └── SilkEncoder::encode: resample to 8/12/16 kHz → RangeCoder   silk/enc_api.rs
        ├── Hybrid or CELT-only: CeltEncoder → same RangeCoder        celt.rs
        ├── at a switch to or from CELT-only: a redundant 5 ms CELT frame
        ├── TOC byte + range coder bytes → packet
        └── DTX (set_dtx): a TOC byte alone while the input is inactive

decode / decode_i16 / decode_fec
  ├── empty packet → conceal in the previous mode (SILK PLC or CELT PLC;
  │     deep PLC at complexity 5 and above for SILK at 16 kHz and CELT-only)
  ├── packet::parse → frames
  └── decode_frame, for each frame
        ├── SILK-only or Hybrid: SilkDecoder → 16-bit samples at the output rate
        │     (with `osce`: LACE or NoLACE on each 20 ms frame at 16 kHz; with
        │      `osce-bwe` and BWE on: BBWENet in place of the resampler)
        │     → f32, written into the output
        ├── Hybrid or CELT-only: CeltDecoder::decode reads on in the same range decoder
        │     → f32, written into the output (CELT-only) or added to it (Hybrid)
        ├── a redundant frame: cross-faded with the main frame
        └── decode_i16 only: pcm_soft_clip on a decoded packet, then rounding to 16 bits
```

### Deep PLC in the decoder

With the feature `deep-plc`, `OpusDecoder` holds one `DeepPlc`. libopus passes its deep PLC state as a pointer from `opus_decode_frame` down to SILK and CELT. opus1 passes `Option<&mut DeepPlcArgs>` (`src/deep_plc.rs`) along the same calls. It holds the state, the models (`None` when they are not loaded) and the kernel variant. The public `SilkDecoder::decode`, `SilkDecoder::decode_with_control`, `CeltDecoder::decode` and `CeltDecoder::decode_lost` keep their signatures and pass `None`; crate-private variants (`SilkDecoder::decode_with_dnn`, `CeltDecoder::decode_with_deep_plc` and `decode_lost_with_deep_plc`) take the argument.

| Where | What happens | libopus |
|---|---|---|
| `OpusDecoder::decode_silk` (`src/lib.rs`) | Every SILK call gets the state and the switch `complexity >= 5`. | `src/opus_decoder.c:443`, `474-479` |
| `SilkDecoder::decode_to_silk_rate` (`src/silk/dec_api.rs`) | The switch goes into the PLC state of the mid channel. Only the mid channel gets the deep PLC state. | `silk/dec_API.c:330`, `355-362` |
| `silk_plc` (`src/silk/plc.rs`) | A decoded 16 kHz frame goes into the history, 10 ms at a time (`DeepPlc::update`). A lost 16 kHz frame is first concealed as without deep PLC. Then, when the models are loaded and deep PLC is switched on (or DRED features are queued), `DeepPlc::conceal` replaces it 10 ms at a time, and the LPC state of SILK is computed back from it. Otherwise the classic output goes into the history. | `silk/PLC.c:105-112`, `399-416` |
| `silk_plc_glue_frames` (`src/silk/plc.rs`) | The fade-in of the first good frame after a loss is skipped at 16 kHz, in every build with the feature and at every complexity. | `silk/PLC.c:477-480` |
| `CeltDecoder::decode_with_deep_plc` (`src/celt.rs`) | A frame with data ends the loss of deep PLC (`DeepPlc::clear_blend`). | `celt/celt_decoder.c:1288-1293` |
| `CeltDecoder::conceal` (`src/celt.rs`) | Deep PLC takes over when the start band is 0, the models are loaded, the complexity is 5 or more, the loss is shorter than 200 ms and noise is not forced (`skip_plc`). On the first such frame the last 2048 samples of the history go to `DeepPlc::update_from_celt` before the history moves. On later frames the pitch period repetition is skipped, because deep PLC overwrites all it would write. `conceal_neural` then fills `plc_pcm` with deep PLC output, upsamples it from 16 to 48 kHz, applies the pre-emphasis, copies channel 0 to channel 1, and on the first frame cross-fades from the pitch period repetition. | `celt/celt_decoder.c:726-736`, `820-831`, `1020-1075` |
| `OpusDecoder::queue_dred_features` (`src/lib.rs`, feature `dred`) | Before the concealment of `dred_decode`, the DRED features of the lost time go into the queue of `DeepPlc` (`fec_clear`, `fec_add`). Deep PLC takes them instead of its predictions. | `src/opus_decoder.c:735-758` |
| `CeltDecoder::conceal` with queued DRED features (feature `dred`) | Deep PLC conceals whatever the complexity and the length of the loss (`FrameType::Dred`, libopus `FRAME_DRED`). Such a frame sets `plc_duration` to 0 and clears `skip_plc`. | `celt/celt_decoder.c:731-734`, `1082-1087` |
| `OpusDecoder::reset_state` | `DeepPlc::reset` clears everything but FARGAN. `CeltDecoder::reset_state` clears `plc_pcm` and its fill level and memory; `set_complexity` passes the complexity to the CELT decoder, and a reset keeps it. | `src/opus_decoder.c:1064`, `1120`; `celt/celt_decoder.c:1729-1738` |

The CELT decoder gets the deep PLC state in the main call of a frame: for a frame with data, a lost frame, the concealment of the CELT layer in an FEC decode, and the 5 ms of concealment that bridge a mode switch without a redundancy frame. The redundancy frames and the 2.5 ms fade-out frame get `None`, as libopus passes `NULL` there.

### OSCE in the decoder

With the feature `osce`, each `SilkDecoderState` holds an `OsceChannel` (field `osce`), and with `osce-bwe` an `OsceBweChannel` (field `osce_bwe`), as libopus keeps `silk_OSCE_struct` and `silk_OSCE_BWE_struct` in `silk_decoder_state`. With `heap` each is a `Box` of its own (`OsceHeld`), so that `OpusDecoder::new` does not build them on the stack. The models stay in `OpusDecoder` (`DecoderDnn`). For each SILK frame, `OpusDecoder::decode_silk` builds an `OsceArgs` (`src/osce_args.rs`) with the method, the extended mode and the layers, and passes it to `SilkDecoder::decode_with_dnn`. The public `SilkDecoder::decode` and `decode_with_control` pass `None`, which is method NONE without BWE, as a libopus `silk_Decode` whose control has no OSCE settings.

| Where | What happens | libopus |
|---|---|---|
| `OpusDecoder::decode_silk` (`src/lib.rs`) | The method follows the complexity (`OsceMethod::for_complexity`: NONE below 6, LACE at 6, NoLACE from 7). The extended mode is `SilkBbwe` at complexity 4 or more with the BWE switch on, 48 kHz output, SILK at 16 kHz and a SILK-only frame or a frame without data; otherwise the mode of the frame. After a CELT-only frame the previous extended mode becomes `CeltOnly`. | `src/opus_decoder.c:444-466` |
| `OpusDecoder::conceal` (`src/lib.rs`) | A lost Hybrid frame in BBWE mode is not concealed by CELT; 2.5 ms of CELT silence are decoded on top of SILK's output instead (`silk_bwe_replaces_celt`). | `src/opus_decoder.c:595-627` |
| `SilkDecoder::decode_to_silk_rate` (`src/silk/dec_api.rs`) | Before each `silk_decode_frame`, a channel whose method differs from the frame's is reset to it (`osce_reset`). A side channel that a mid-only frame does not code is neither reset nor run. | `silk/dec_API.c:350-354` |
| `silk_decode_frame` (`src/silk/decode_frame.rs`) | The bits of the frame are counted from before `silk_decode_indices`. After the update of `out_buf` and before `silk_plc`, `osce_enhance_frame` runs with the frame's parameters (`OsceFrame`) at every complexity: method NONE only clips -32768 to -32767. A lost frame resets the channel's OSCE state with its method. | `silk/decode_frame.c:74-77`, `109-114`, `139-141` |
| `OutputStage` in `src/silk/dec_api.rs` | Per output channel: in BBWE mode `osce_bwe` instead of the resampler, with `osce_bwe_reset` when the previous frame was not BBWE and a 10 ms fade from the resampler's output after a SILK-only or Hybrid frame. Otherwise the resampler, with a 10 ms fade from `osce_bwe` after a BBWE frame when the SILK rate is still 16 kHz. Then `prev_osce_extended_mode` takes the frame's mode. The right channel of the first mono frame after stereo goes through its resampler, also in BBWE mode. | `silk/dec_API.c:389-454` |
| `silk_reset_decoder`, `silk_init_decoder` (`src/silk/init_decoder.rs`) | A reset sets the OSCE state to `OSCE_DEFAULT_METHOD` (NoLACE) and keeps the BWE state. Only `silk_init_decoder`, which runs for a new decoder and for channel 1 on a switch from mono to stereo, clears the BWE state to zero. | `silk/init_decoder.c:61-64`, `73-83`, `silk/dec_API.c:177-179` |
| `OpusDecoder::reset_state` | `SilkDecoder::reset` keeps the BWE states and `prev_osce_extended_mode`, which libopus keeps in `DecControl`. | `src/opus_decoder.c:1109-1121` |

libopus 1.6.1 sets the extended mode only in its SILK section, so after BBWE frames it skips the CELT layer of CELT-only frames. opus1 follows libopus main (commit `228a0f85`), which sets the extended mode to CELT-only for those frames. The extended mode is therefore computed for each frame and not stored in `OpusDecoder`: a frame with data is in BBWE mode only when it is SILK-only.

## Floating point and fixed point

- **CELT is floating point.** `celt.rs`, `bands.rs`, `pvq.rs`, `mdct.rs`, `kiss_fft.rs`, `pitch.rs`, `celt_lpc.rs` and `quant_bands.rs` compute in `f32`, as the float build of libopus does. The bit allocation in `rate.rs` is integer arithmetic, as in libopus.
- **SILK is fixed point.** Everything under `src/silk/` works on `i16` and `i32` values in Q formats, as the fixed-point build of libopus (`silk/fixed/`) does. The float build of libopus uses the floating-point SILK encoder instead, so SILK packets from `opus1` and from a float build of libopus differ. Both are valid, and any Opus decoder reads them.
- **The Opus layer converts between them** in `src/lib.rs`:
  - Encoder: SILK needs 16-bit samples before its resampler. With `Application::Voip` at 8 to 24 kHz they come from the integer high-pass filter of the fixed-point build: `hp_cutoff` takes `f32` input and writes `i16`. With `Voip` at 48 kHz, and with the other applications, the float signal is filtered and then rounded to 16 bits (`SilkInput::Float`). CELT gets the float signal with a 4 ms delay buffer in front, unless the application is one of the restricted ones (`RestrictedLowDelay`, `RestrictedCelt`; `RestrictedSilk` codes no CELT).
  - Decoder: SILK's 16-bit output is scaled by 1/32768 to `f32` (`int16_to_res`) and written into the output. In a Hybrid frame CELT adds its output to it.
- `encode_i16` converts its input to float once, with `sample as f32 / 32768.0`, and then does what `encode` does, with an input precision (`lsb_depth`) of at most 16 bits, as libopus `opus_encode` does. The conversion is exact, and the integer high-pass filter of `Voip` rounds the float samples back to the same 16-bit values. So `encode_i16` writes the packets that `encode` writes for the same audio after `set_lsb_depth(16)`.
- **The speech and music analysis is floating point**, as in the float build of libopus. The fixed-point build of libopus has its own integer version of it, which opus1 does not port.

## Feature flags and where state is stored

The crate has five features that decide how it is built: `std`, `heap` and `analysis` (on by default), `alloc` (on with `std` or `heap`) and `libm`. The neural network features (`dnn`, `deep-plc`, `osce`, `osce-bwe`, `dred`) add code, `test-libopus-dnn` only compiles tests, and `__internals` adds the module `internals` for the tests (see [The public interface and `internals`](#the-public-interface-and-internals)). [Using the library](06-using-the-library.md#feature-flags) describes them from the user's side. Inside the crate:

- **`std`** selects `#![no_std]` when it is off (`#![cfg_attr(not(feature = "std"), no_std)]` in `src/lib.rs`). It also enables the run-time processor checks in `src/compat.rs` and `src/kiss_fft.rs`, and it turns on `alloc`.
- **`alloc`** links the `alloc` crate (`extern crate alloc` in `src/lib.rs`), so it needs a global allocator but no operating system. It enables the `multistream` module, which keeps its stream encoders and decoders in a `Vec`. Each of them is built in place in the memory of the `Vec`, with `OpusEncoder::init` or `OpusDecoder::init`, so a large stream codec never passes through the stack. It also lets `ExtensionList` (`src/extensions.rs`) and `pad_keeping_extensions` (`src/packet.rs`) allocate when their fixed room of 32 extensions or 1275 bytes is too small. The files of `src/multistream/` import `Vec` and `vec!` from `alloc` under `#[cfg(not(feature = "std"))]`.
- **`libm`** is required when `std` is off. `src/compat.rs` stops the build with `compile_error!` if both are missing. Its `Math` trait gives `f32` and `f64` the methods `sqrt`, `sin`, `exp2` and so on through the `libm` crate. Each module that needs them imports the trait under `#[cfg(not(feature = "std"))]`; with `std` the inherent methods of `f32` are used instead.
- **`analysis`** compiles the speech and music analysis (`TonalityAnalysisState` and the network in `src/analysis/`). Without it the encoder decides as libopus does below complexity 7, at every complexity, and `OpusEncoder` is smaller. The types of `src/analysis/mod.rs` exist in both builds.
- **`heap`** changes `src/lib.rs` and `src/silk/enc_api.rs`. Each large field of `OpusEncoder` and `OpusDecoder` is declared twice:

  ```rust,ignore
  #[cfg(not(feature = "heap"))]
  celt_enc: CeltEncoder,
  #[cfg(feature = "heap")]
  celt_enc: Box<CeltEncoder>,
  ```

  The helper `state_mut` borrows such a field as the inner type in both builds, so the code that uses it needs no `#[cfg]`. `SilkEncoder` stores its channels and its input buffer in `Slot<T>`, which is `Box<T>` with `heap` and an inline value without it. `heap` turns on `alloc` and takes `Box` from it, so it also works without `std`.

  `OpusEncoder::reset_state` writes the new state over the old one with the same functions that build an encoder in place (`write_new`). With `heap` they get `HeapMemory::Reuse` (`src/in_place.rs`): each box keeps its memory and only its contents are written anew, so a reset allocates nothing, as libopus `OPUS_RESET_STATE`.

Below the Opus layer nothing allocates. The CELT, SILK and range coder states store every buffer inline, with a size fixed at compile time:

- SILK mostly uses plain arrays, sized by the largest frame (for example `x_buf: [i16; 2 * MAX_FRAME_LENGTH + LA_SHAPE_MAX]`).
- CELT, the FFT, the MDCT and the range coder use `FixedVec<T, N>` from `src/fixedvec.rs`. It stores up to `N` elements in `[MaybeUninit<T>; N]` and keeps a length. It offers the part of the `Vec` interface the codec uses (`push`, `resize`, `truncate`, `extend_from_slice`, slicing through `Deref`). Every operation that could go past the capacity checks this with `assert!`, and panics instead of going past it. It never drops its elements, so a `FixedVec` of a type with a destructor does not compile.
- The capacities are constants for the largest case: 2 channels, 120 ms at 48 kHz, 1276-byte packets (`OPUS_MAX_CHANNELS`, `OPUS_MAX_FRAME`, `OPUS_MAX_PACKET_BYTES` and others at the top of `src/lib.rs`). This is why the size of an encoder does not depend on its sampling rate or channel count.

The neural network features follow the same rules:

- `dnn` compiles `src/dnn/`, but adds no public API: `DnnBlob`, `DnnModels`, `set_dnn_blob` and `dnn_models` need `deep-plc`, the smallest feature with a model. `deep-plc`, `osce` and `osce-bwe` each turn on the one before and `dnn`. Each embeds the weights of its networks and adds its groups to `ModelSet` and `DecoderDnn`. `dred` turns on `deep-plc` and embeds the two RDOVAE networks; an `OpusDredDecoder` holds the RDOVAE decoder, not `ModelSet`. `#[cfg(feature = "...")]` on the fields and match arms keeps a build without a feature free of its code and its weights.
- With `heap`, `OpusDecoder` holds `DecoderDnn` in a `Box`, and the layers of a group loaded from a blob are in a `Box` too. Without `heap`, `DecoderDnn` is inline and has room for the layers of every group. Blobs then must be `'static`, because there is no heap to copy them to. `DnnBlob::from_bytes` shares its copy through an `Arc`, so it also needs a target with atomic compare-and-swap (`target_has_atomic = "ptr"`).
- `test-libopus-dnn` adds nothing to the library. Test files that call a libopus built with its neural network code start with `#![cfg(feature = "test-libopus-dnn")]` (see [Testing](09-testing.md#tests-of-the-neural-network-code)).

The measured sizes of both builds are in [Using the library](06-using-the-library.md#size-of-the-encoder-and-the-decoder).

`src/compat.rs` also replaces what `no_std` without `alloc` lacks:

- `OnceCell<T>`: a value built once into inline storage, guarded by an atomic state (0 empty, 1 being built, 2 ready). It holds the `CeltMode`.
- `sort_by`: an insertion sort for the short slices the codec sorts, because the stable `sort` of slices needs `alloc`.

## Code for specific processors

The loops that take most of the run time have several versions. The rule is the same in every file:

1. A safe function is the only entry point, for example `pitch::inner_prod`. It checks the slice lengths, usually by slicing to the exact length (`&x[..n]`), so a short slice panics before any vector code runs.
2. On aarch64 it calls the NEON version. NEON is part of every aarch64 target, so there is no run-time check. These functions use `core::arch::aarch64` intrinsics; some carry `#[target_feature(enable = "neon")]`.
3. On x86 and x86-64 it checks for SSE4.1, AVX or AVX2 (Advanced Vector Extensions) and FMA (fused multiply-add):
   - It asks `crate::compat::x86_has_sse41()`, `x86_has_avx()`, `x86_has_avx2()`, `x86_has_avx_fma()` or `x86_has_avx2_fma()`.
   - When the answer is yes, it calls an `unsafe fn` marked `#[target_feature(enable = "avx,fma")]` (or `"sse4.1"`, `"avx"`, `"avx2"`, `"avx2,fma"`).
   - The SILK noise-shaping quantizers (`silk/nsq.rs` and `silk/nsq_del_dec.rs`) need SSE4.1. The delayed-decision quantizer has a second copy of its sample loop for AVX2. Both check once per subframe.
   - The SILK analysis kernels check once per call: the warped autocorrelation (`silk/sigproc_fix.rs`) and the A2NLSF polynomial evaluation (`silk/nlsf.rs`) have an SSE4.1 and an AVX2 version, the LTP codebook search (`silk/vq_wmat_ec.rs`) an SSE4.1 version, and the batch division of the pitch analysis (`silk/pitch_analysis.rs`) an AVX2 version.
   - Some portable SILK encoder functions run in a second copy compiled with AVX2: the pitch analysis, Burg's method, the correlations of the LTP analysis, the autocorrelation, the voice activity detection, the LTP analysis filter and the frame loop of the plain quantizer. The body is an `#[inline(always)]` function, and the copy is an `unsafe fn` with `#[target_feature(enable = "avx2")]` that only calls it. The default x86-64 target has no 32-bit and no 32x32-to-64-bit vector multiplication, which fixed-point code needs. The source is the same, so the results are the same bits.
   - Kernels that use FMA instructions run only when AVX and FMA are both present, because some processors have AVX without FMA.
   - Some kernels in `pitch.rs` and `celt.rs` also have an SSE version. SSE (Streaming SIMD Extensions, where SIMD means single instruction, multiple data) is the older set of x86 vector instructions, and every x86-64 processor has it.
4. Otherwise it runs the portable Rust version.

The check happens inside the function that the rest of the codec calls, at every call. With `std` the probes use `std::arch::is_x86_feature_detected!`, which stores its answer after the first call. Without `std` they use `cfg!(target_feature = "...")`, which the compiler decides from the build flags.

Files with x86 versions: `bands.rs`, `celt.rs`, `kiss_fft.rs`, `mdct.rs`, `pitch.rs`, `pvq.rs` and, under `src/silk/`, `lpc_analysis.rs`, `ltp_analysis.rs`, `nlsf.rs`, `nsq.rs`, `nsq_del_dec.rs`, `pitch_analysis.rs`, `sigproc_fix.rs`, `vad.rs` and `vq_wmat_ec.rs`. Files with aarch64 versions: the same without `silk/ltp_analysis.rs` and `silk/vad.rs`, plus `src/analysis/mod.rs` (the search for the largest sample of the digital silence test), `src/analysis/neon.rs` and, under `src/silk/`, `decode_core.rs` and `resampler.rs`. The analysis has no x86 vector code: on x86 it runs its scalar code.

`src/mathops.rs` also depends on the processor, for rounding rather than speed. On aarch64, `mac` and `mac_f64` compute `acc + a * b` with one fused multiply-add, which rounds once. The arm64 build of libopus does the same, because clang fuses such C expressions there. On other targets the product is rounded first.

The x86 code ran once on an Intel Core i7-12700 (AVX2, FMA and AVX-VNNI), on 2026-10-21. Its output is correct but not bit-exact with libopus: its vector kernels do not follow libopus's x86 kernels (see [Differences from libopus](10-differences-from-libopus.md#platforms)). The aarch64 code runs in every test on Apple silicon.

### The kernels of the neural networks

`src/dnn/` follows another rule, because its results must not depend on the processor:

- Every kernel variant gives the same bits: those of libopus 1.6.1 on 64-bit ARM, built at `-O2` without loop vectorization (`-fno-vectorize -fno-slp-vectorize`). The tests compare every variant with that build of libopus and with `vec_portable.rs`.
- The variant is chosen once per decoder (`DnnArch`, from `arch::detect()`), as libopus chooses `arch` once. Every layer call takes it as an argument. `arch::usable()` replaces a variant the processor cannot run, so a wrong value cannot reach an instruction the processor lacks.
- On 64-bit ARM, `NeonDotprod` needs the DotProd extension. It is known at compile time on Apple targets. Elsewhere `std::arch::is_aarch64_feature_detected!("dotprod")` asks the operating system; without `std` the variant is `Neon`.
- On other targets the kernels compute the 64-bit ARM bits too: products and sums that clang fuses on 64-bit ARM use `mul_add`, and the ARM reciprocal estimate (FRECPE) is computed in software. So the x86 kernels do not follow libopus's x86 kernels, which round differently and differ between Intel and AMD processors.
- `freq.rs` and `burg.rs` use fused multiply-adds exactly where the libopus object code has them, which was read from the disassembly of the reference build.

### The FFT and FEAT_FCMA

FEAT_FCMA is an ARMv8.3 extension with two instructions for complex numbers: `fcmla` (complex multiply-add) and `fcadd` (complex add with a rotation by 90 or 270 degrees). Their Rust intrinsics are not stable, so `src/kiss_fft.rs` writes them with `core::arch::asm!` in the macros `cmul_fcma` and `pm_i_fcma`.

- The module `fused` builds the FFT stages twice from one macro, `kernels!`: `fused::fcma` (marked `#[target_feature(enable = "neon,fcma")]`, with `fcmla` and `fcadd`) and `fused::fma` (plain NEON with `fmla`). Both give the same bits.
- `has_fcma()` decides which one runs. It is `true` at compile time where the target includes FEAT_FCMA (Apple silicon). Elsewhere it asks `std::arch::is_aarch64_feature_detected!("fcma")` with `std`, and answers `false` without `std`.
- On aarch64 the FFT gives the bits of `opus_fft_impl_fused`, which match the arm64 build of libopus. On other targets it gives the bits of `opus_fft_impl_scalar`. The tests reach both through `opus1::internals::kiss_fft`, so they can compare them on any machine.

## Unsafe code

The library uses `unsafe` code where safe Rust would cost speed or stack memory. This section lists the kinds of `unsafe` code, where each kind is, what the public API exposes, and the rules the code follows.

### What applications see

- No function or method that an application can call is an `unsafe fn`. This covers the crate root, `packet`, `extensions`, `multistream` and the neural network types (`DnnBlob`, `DnnModels` and the DRED types).
- `OpusEncoder::init` and `OpusDecoder::init` build a codec in memory that the caller provides as `&mut MaybeUninit<Self>`. Both are safe functions. A caller who keeps that memory in a `static mut` needs an `unsafe` block of their own to borrow it, as the examples in [Using the library](06-using-the-library.md) show.
- The module `internals` is the exception. It exists only with the test-only feature `__internals`, which is outside semantic versioning. The module itself contains no `unsafe` code, but `internals::dnn` re-exports all of `src/dnn/`, so it includes the kernels of `vec_neon.rs` and `vec_x86.rs`, which are `pub unsafe fn`.

### The kinds of unsafe code

1. **Vector kernels.** Most of the `unsafe` code is here. These functions use the vector instructions of the processor: the intrinsics of `core::arch`, and on aarch64 also inline assembly for FEAT_FCMA in `kiss_fft.rs`. Some are portable code compiled for an extra instruction set with `#[target_feature]`. Many kernels load and store through raw pointers, without index checks.
   - A safe entry point checks the slice lengths before a kernel runs (see [Code for specific processors](#code-for-specific-processors)).
   - A kernel runs only on a processor that has its instructions. NEON is part of every aarch64 target, so it needs no check. SSE is part of every x86-64 target. On x86 the functions of `crate::compat` check for SSE4.1, AVX, AVX2 and FMA: at run time with `std`, and from the build flags without `std`. `has_fcma()` in `kiss_fft.rs` checks for FEAT_FCMA, and `src/dnn/arch.rs` checks for the DotProd extension that some neural network kernels use.
   - Files with aarch64 kernels: `src/lib.rs` (the conversion of decoded samples to 16 bits), `bands.rs`, `celt.rs`, `kiss_fft.rs`, `mathops.rs`, `mdct.rs`, `pitch.rs`, `pvq.rs` and `rate.rs`; `src/analysis/` (`mod.rs`, `neon.rs`, `tonality.rs`); `src/eval/compare.rs` (test-only, see item 6); under `src/silk/`, `decode_core.rs`, `lpc_analysis.rs`, `nlsf.rs`, `nsq.rs`, `nsq_del_dec.rs`, `pitch_analysis.rs`, `resampler.rs`, `sigproc_fix.rs` and `vq_wmat_ec.rs`; and `src/dnn/vec_neon.rs` and `src/dnn/vec_portable.rs`.
   - Files with x86 kernels: `bands.rs`, `celt.rs`, `kiss_fft.rs`, `mdct.rs`, `pitch.rs`, `pvq.rs`, `src/silk/nsq.rs`, `src/silk/nsq_del_dec.rs`, `src/silk/sigproc_fix.rs` and `src/dnn/vec_x86.rs`.
   - In `src/dnn/`, `nnet.rs` calls the kernels of `vec_neon.rs` and `vec_x86.rs` only after it has checked the slices and the index tables of sparse layers.
2. **Building large states in place.** Without `heap` an `OpusEncoder` holds about 150 KB inline and an `OpusDecoder` about 60 KB. A constructor that returns such a value builds it on the stack and copies it out. So the constructors write every field through a raw pointer into the final memory, which may be uninitialized.
   - `src/in_place.rs` holds the tools. The macro `write_fields!` writes the fields and checks at compile time that it writes all of them. `write_zero` fills a field with zero bytes, for the types of the trait `Zeroable`, whose all-zero bytes are a valid value. `boxed` and `write_box` fill the boxes of `heap` in place.
   - Each type built this way has an `unsafe fn write_new` (or `write_default`). They are `OpusEncoder` and `OpusDecoder` in `src/lib.rs`, `CeltEncoder` and `CeltDecoder` in `celt.rs`, the SILK encoder and decoder (`src/silk/enc_api.rs`, `dec_api.rs` and `decoder_structs.rs`), the analysis state in `src/analysis/tonality.rs`, `CeltMode` in `modes.rs`, `MdctLookup` in `mdct.rs`, the range coder in `range_coder.rs`, `FixedVec` in `fixedvec.rs`, and in `src/dnn/` the states of `holder.rs`, `models/mod.rs`, `plc/lpcnet_plc.rs`, `dred/encoder.rs` and `osce/mod.rs`.
   - `OpusEncoder::reset_state` writes a new encoder over the old one in the same way.
   - `src/multistream/encoder.rs` and `decoder.rs` build each stream codec with `OpusEncoder::init` or `OpusDecoder::init` in the spare capacity of a `Vec`, and then lengthen the `Vec` with `Vec::set_len`.
3. **Uninitialized storage.** Some buffers start without values, so that they need no fill with zeros.
   - `FixedVec` (`fixedvec.rs`) is a `Vec` with inline storage: an array of `MaybeUninit<T>` and a length. Its slices cover only the first `len` elements, which hold values.
   - `compat::OnceCell` (`compat.rs`) builds a shared table on first use, also without `std`. It holds the CELT mode (`modes.rs`), the FFT state of the neural network features (`src/dnn/freq.rs`) and the built-in neural network models (`src/dnn/builtin.rs`). An atomic state lets exactly one caller write the value, and the other callers wait until it is complete. An `unsafe impl` marks `OnceCell` as `Sync`.
   - Scratch buffers of `MaybeUninit` values: in `bands.rs`, the buffers of the band reordering and the copies of `quant_band_stereo_rdo`; in `pvq.rs`, the buffers of the pulse search and the pulse vector of a band, which the pulse decoder (`cwrsi_table`) and the pulse search write through the trait `PulseSlot`; in `pitch.rs`, the tables of window energies; in `mdct.rs`, the buffers before and inside the FFT (fast Fourier transform); in `src/silk/decode_core.rs`, the state of the long-term prediction; in `src/silk/resampler.rs`, the filter buffers; and in `src/dnn/vec_neon.rs` and `vec_x86.rs`, the quantized input of a layer.
   - The rule for this storage: the code writes values through `MaybeUninit` first (`write`, `write_copy_of_slice`, or `write_filled` of `pvq.rs`, which sets every value to one value). Only memory that holds values becomes a `&[T]` or `&mut [T]` (with `assume_init_ref` or `assume_init_mut`). Memory that is not written yet is reached through raw pointers or `MaybeUninit` only, never through a slice of `T`, even if it is written right after.
   - Where tracking which values are written would cost more than a fill, the buffer is set to 0 first. `norm` and `norm2` of `quant_all_bands` in `bands.rs` (the decoded bands kept for folding) are filled this way once per frame, and only when the frame resynthesizes its bands (the decoder always does).
4. **The weights of the neural networks.** `src/dnn/weights.rs` views the bytes of a blob as `f32`, `i32` or `i8` arrays with `align_to`, after it checked the alignment. In `src/dnn/holder.rs`, `DnnBlob::from_bytes` gives the parsed list of a heap copy the lifetime `'static`. That is sound because the copy is never moved, changed or freed while a clone of the blob exists, and `DecoderDnn` keeps a clone next to every layer it builds from it and lends the layers out only for the lifetime of a borrow of itself. The copy is owned by `SharedBlob` through a raw pointer and freed in `Drop`, not held in a `Box`: moving a `Box` asserts that no other pointer to its memory exists, which the parsed list contradicts. Two `unsafe impl` blocks mark `SharedBlob` as `Send` and `Sync`, because it never writes the copy after `from_bytes`.
5. **Lookups without an index check.** `get_unchecked` appears in three files, where an index check at every access cost speed. In `bands.rs`, `band_should_split` reads the pulse cache of the mode. In `pvq.rs`, `pvq_u_unchecked` reads the table of pulse vector counts; its callers check the range first, most of them with `pvq_table_covers`. In `src/dnn/vec_x86.rs`, the kernels of sparse layers read the index tables that `nnet.rs` checked. Besides these, `src/silk/nsq_del_dec.rs` reads groups of five and of three values through a pointer, at an index chosen without a branch.
6. **Test-only code.** The unit tests (the `#[cfg(test)]` code) use `unsafe` code too. For example, they call kernels directly, and the helper `test_guard` in `kiss_fft.rs` places inputs right before and right after memory that may be neither read nor written (`mmap` and `mprotect`). The tests of `pitch.rs`, `src/silk/decode_core.rs` and `src/silk/resampler.rs` view slices of values as slices of `MaybeUninit`. The library build does not contain this code. The test-only feature `__internals` (with `std`) adds the module `src/eval/`, the tools that measure decoded audio. Its `compare.rs` has an aarch64 kernel. Otherwise `__internals` adds no `unsafe` code: it only makes more items reachable, as described above.

### Rules

- Every `unsafe` block in library code (outside the test modules) has its own `// SAFETY:` comment that says why the block is sound. The crate root turns on the Clippy lint `undocumented_unsafe_blocks` for every build except the unit tests, so `cargo clippy` warns about a block without one.
- Most `unsafe fn` declarations have a `# Safety` section that states what the caller must guarantee. Those without one are:
  - register helpers that touch no memory: `lo2`, `hi2` and `re_of_a_im_of_b` in `kiss_fft.rs`, `rev4` in `mdct.rs`, and `reverse`, `atan2`, `to_f32`, `limit`, `tansig` and `sigmoid` in `src/analysis/neon.rs`;
  - `taps`, a helper inside a function of `celt.rs`, whose doc comment states its contract in a `SAFETY:` line;
  - `forward_avx` and `backward_avx` in `mdct.rs`, `l1_metric_neon` in `celt.rs`, and the four AVX2 kernels in `src/silk/sigproc_fix.rs`.
- Kernels check their preconditions where they are cheap: `assert!` in entry points (for example `opus_fft_impl` checks the buffer length), `debug_assert!` inside many `unsafe fn` kernels, and the `assert!` checks of `FixedVec`.
- The crate root allows `unsafe_op_in_unsafe_fn`, so an `unsafe fn` body may call unsafe operations without an inner `unsafe` block. Many functions still use inner blocks with their own comments.

## Error handling

- Every public function that can fail returns `Result<T, Error>`. `Error` (`src/error.rs`) holds an `ErrorKind`, one per libopus error code, and a fixed message such as `"invalid packet"` or `"output buffer too small"`. Each check returns the kind that libopus returns at the same place. The internal functions return `Error` too, so `?` passes an error up without a conversion. Shared errors are constants of `Error` (for example `Error::INVALID_PACKET`), so each has one wording. [Using the library](06-using-the-library.md#errors) lists what the encoder and decoder reject.
- The SILK functions keep the libopus convention and return `i32` codes: 0 (`SILK_NO_ERROR`) for success, and negative codes such as `SILK_ENC_FS_NOT_SUPPORTED` (-102) otherwise. `SilkEncoder::encode` (`src/silk/enc_api.rs`) turns a non-zero code into an error of kind `InternalError` with a message such as `"SILK encoding failed"`, as libopus turns a SILK error into `OPUS_INTERNAL_ERROR`.
- The CELT encoder reports nothing itself. The range coder records an overflow in its `error` field, and `src/lib.rs` checks that field and the bit count after each frame.
- Input is checked before state changes. The decoder parses the packet and checks the buffer sizes before it updates its state, so a rejected packet leaves the decoder unchanged.
- The setters of `OpusEncoder` check their values as libopus's `OPUS_SET_*` requests do. A refused value is an error and leaves the setting as it was. A positive bitrate outside the allowed range is limited instead, as in libopus.

## Tests, examples, benchmarks and fuzzing

### `tests/`

The repository holds 280 test files in `tests/`, plus helper modules in subdirectories. More than half of them run the same input through `opus1` and through C libopus. They compare packets, decoded samples or internal values. libopus is linked through the `opus` crate and `audiopus_sys` (see [building without cmake](06-using-the-library.md#building-and-testing-on-a-machine-without-cmake)). [Testing](09-testing.md) explains what the tests check and how to run each part. The file name prefix says what a test covers:

| Prefix | Files | What the tests cover |
|---|---:|---|
| `perfk_` | 33 | Single kernels after performance work, against the same function inside libopus or a reference version: FFT, PVQ, range coder, pitch, SILK analysis and quantizer. Four `.expected` files hold reference output. |
| `multistream_` | 35 | The multistream encoder and decoder: `multistream_c_*` against libopus's own functions, `multistream_enc_*` and `multistream_dec_*` for each side, `multistream_e2e_*` end to end, `multistream_surround_*` for the surround masking analysis, `multistream_analysis_libopus`, `multistream_analysis_walk_libopus` and `multistream_dtx_libopus` for the analysis and DTX of each stream. |
| `core_` | 36 | `OpusEncoder` and `OpusDecoder` entry points and controls against libopus: loss, FEC, mode switches, channel changes, limits, the speech and music analysis (`core_analysis_libopus`, `core_analysis_walk_libopus`), DTX (`core_dtx_libopus`), the high-pass filter in front of SILK (`core_voip_input_filter`), the decoder controls of the neural network features. `core_packet` and `core_packet_extensions_libopus` test the packet functions, the repacketizer and the packet extensions against libopus's own functions. |
| `dnn_` | 23 | The neural network code: kernels and their edge cases, layers of every network, weights, signal analysis, the models of deep PLC and OSCE, both inside the decoder, and the DRED decoder. The 10 files that end in `_libopus` and `dnn_freq_burg.rs` need a libopus built with its neural network code (`dnn_dred_libopus.rs` and `dnn_dred_decode_libopus.rs` one built with DRED) (see [Testing](09-testing.md#tests-of-the-neural-network-code)). |
| `analysis_` | 3 | The speech and music analysis alone against libopus's `run_analysis` and network, and its properties. |
| `k24_` | 21 | The 24 kHz path: encoder modes, Hybrid super-wideband, SILK internals and loss, and the decoder matrix of every mode and bandwidth at every output rate. |
| `celtenc_`, `celtdec_`, `celt_` | 19, 4, 8 | The CELT encoder and decoder. |
| `silkenc_`, `silkdec_`, `silk_` | 6, 10, 14 | The SILK encoder and decoder. |
| `mdct_` | 16 | The MDCT. |
| `issue_`, `issues_` | 15 | Regression tests named after GitHub issues, each reproducing the reported fault. |
| `verify_`, `pvq_`, `hybrid_` | 5, 4, 4 | Checks of band, PVQ, quantization and SILK functions, and of Hybrid mode. |
| `eval_` | 5 | The evaluation tools of `src/eval/` against the C program `opus_compare` and the Python script `moc.py`, the `opus_demo` decoding loop with both decoders, the RFC 8251 conformance test, and the paired-comparison harness. Two `.expected` files hold reference values. |
| others | 19 | Single topics, for example `stack_usage_test.rs` (struct sizes and stack use), `panic_surface_repro.rs` (inputs that once panicked), `quality_regression_test.rs`, `proptest_encode_indices.rs` (property-based tests with `proptest`). |

Helper directories are named after the tests that use them: `celtdec_common/`, `celtenc_common/`, `core_common/`, `dnn_dec_common/`, `dnn_gen/`, `dnn_plc_common/`, `eval_common/`, `k24_common/`, `k24_enc_common/`, `ms_support/`, `multistream_dec_common/`, `multistream_e2e_common/`, `multistream_enc_common/`, `silkdec_common/`, `silkenc_common/`. `dnn_gen/` is written by `tools/dnn-gen`. Each holds a `mod.rs` that a test includes with `mod <name>;`. Several of them wrap the raw C application programming interface (API) of libopus (`audiopus_sys`) for the controls the `opus` crate does not offer, such as forcing the coding mode.

Tests marked `#[ignore]` either print measurement tables and assert nothing, or record a known difference from libopus. Run one with `cargo test --release --test <name> -- --ignored --nocapture`.

Unit tests also live inside the source files, in `#[cfg(test)]` modules. Many of them compare a vector kernel with its portable version.

### `examples/`

- `multistream.rs`: encodes and decodes two seconds of a 5.1 test signal and prints the layout, packet sizes and signal-to-noise ratio of every channel.
- `wav_test.rs`: encodes and decodes a WAV (Waveform Audio File Format) file (by default `fixtures/answer_16k.wav`) and writes the decoded files into `fixtures/`.
- `eval/`: the paired-comparison harness. opus1 and libopus code the same items with the same settings, libopus decodes both, and `opus_compare` and MOC score each output against the original. It also runs the RFC 8251 conformance check and a self-test of its gates. See [Testing](09-testing.md#the-paired-comparison-harness). Its files:
  - `main.rs` and `cli.rs`: the commands and their options.
  - `sets.rs`: the built-in sets, their conditions and gates; `sets/tsp_speech.txt` lists the TSP items.
  - `corpus.rs`: items, set files, the corpus root and the `corpus-index` command.
  - `signals.rs`: WAV input and output, the resampler and the synthetic items.
  - `pipeline.rs`: one job: encode with both encoders, decode, align and measure.
  - `libopus.rs`: libopus's encoder and decoder through `audiopus_sys`.
  - `stats.rs`: geometric means, bootstrap intervals, aggregates and gates.
  - `report.rs`: the JSON, text and Markdown output, and the reader for `--baseline` and `merge`.
  - `degrade.rs`: the degradations of the self-test.
  - `conformance.rs`: the RFC 8251 table of `eval conformance`.
  - `tier2.rs`: the bridge to external quality metrics (`--write-wav`, `merge`).
- `opus_compare.rs`: libopus's test program `opus_compare`, with its command line and messages, on top of `src/eval/compare.rs`.
- `perf_harness.rs`: the timing and comparison tool against libopus. Its commands are `list`, `bench`, `run`, `golden`, `modes`, `settings`, `signals`, `resampler`, `spread`, `pairs` and `analysis` (the speech and music analysis alone). `cargo run --release --example perf_harness -- bench` produces the timings in the README.

### `benches/`

`benches/dnn_bench.rs` (it needs the feature `dnn`) times the layer kernels of the neural networks for every `DnnArch`, and with `test-libopus-dnn` also the same calls of libopus. `benches/opus_bench.rs` holds Criterion benchmarks (`cargo bench`) of whole encoding (SILK, CELT, 16-bit input) and of single kernels: FFT, MDCT, PVQ, the SILK noise-shaping quantizer, pitch analysis, autocorrelation, Burg's method, the speech and music analysis and its network, and others. The script `bench_quick.sh` at the repository root runs a shorter `perf_harness bench` (6 repetitions in one round) and saves its table to a Markdown file. It takes a label and an optional scenario filter.

### `fuzz/`

`fuzz/` is a separate Cargo package of 24 targets for libFuzzer. `fuzz/fuzz.sh` builds them with the stable toolchain and runs them. Eight targets compare opus1 with libopus: `fuzz_packet`, `fuzz_decoder_sequence`, `fuzz_multistream_decoder`, `fuzz_multistream_encoder`, `fuzz_encoder_settings`, `fuzz_channel_order`, `fuzz_extensions` and `fuzz_repacketizer`. Their wrappers of the libopus C functions are in `fuzz/fuzz_targets/common/mod.rs`. The other 16 targets run opus1 alone. Three of them test the neural network code and need a feature of the fuzz package: `fuzz_dnn_blob` (feature `dnn`) loads weight blobs, and `fuzz_dred` and `fuzz_decoder_dnn` (feature `dred`) decode DRED data and run the decoder with its networks. `fuzz/README.md` and [Testing](09-testing.md#fuzzing) describe what each target checks.

### `scripts/`

`scripts/android/build-libopus.sh` builds the libopus references for the phone and the Mac; with `--dnn` it builds them with the neural network code (see [Testing](09-testing.md#builds-with-the-neural-network-features)). `scripts/check_no_std.sh` builds the crate without `std` for `thumbv7em-none-eabi` and `wasm32-unknown-unknown` (when installed) and for the host, then runs the test suite without `std`.

`scripts/eval/` holds the scripts of the evaluation tools: `fetch-corpus.sh` (downloads and checks the corpora), `check-opus-compare.sh` and `opus_compare_exact.c` (build libopus's `opus_compare` as the reference of the port), `moc_reference.py` (the numpy reference of MOC) and `tier2.py` (runs external quality metrics that are installed). See [Testing](09-testing.md#evaluation-tools).
