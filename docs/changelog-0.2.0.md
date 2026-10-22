# Changes in opus1 0.2.0

This file keeps the detailed list of changes from opus-rs 0.1.37 to opus1 0.2.0, with the measurements behind each fix. The short release note is in the [README](../README.md#020-unreleased).

The crate is renamed from `opus-rs` to `opus1`. Code that used `opus_rs::` now uses `opus1::`.

This release was checked against libopus 1.6.1 throughout: every fix below has a
test that compares opus1 with the C library under the same settings.

## Breaking changes to the public interface

- Errors: every call that can fail returns `Result<T, opus1::Error>` instead of
  `Result<T, &'static str>`, and `TryFrom<i32> for FrameDuration` has
  `type Error = opus1::Error`. `Error::kind()` returns an `ErrorKind` with one
  value per libopus error code: `BadArg`, `BufferTooSmall`, `InternalError`,
  `InvalidPacket`, `Unimplemented`, `InvalidState` and `AllocFail`
  (`#[non_exhaustive]`). Each call returns the kind that libopus returns in the
  same case. `Error::to_raw()` and `ErrorKind::to_raw()` give the libopus code
  (-1 to -7). `Error::message()` and `Display` give a short text. The texts
  start with a lowercase letter, have no final period, and one error has one
  wording: for example every output buffer that is too small gives "output
  buffer too small". `Error` is `Copy` and implements `core::error::Error`.
  Code that compared message strings now compares `err.kind()`. Kinds that
  differ from what the old messages suggested: an empty output buffer of
  `encode` is `BadArg` (libopus `max_data_bytes <= 0`); without the `alloc`
  feature, more than 32 extensions or more than 1275 bytes of old padding in
  one call are `AllocFail`.
- Sampling rates and channel counts have their own types at the crate root.
  `SampleRate` has the values `Hz8000`, `Hz12000`, `Hz16000`, `Hz24000` and
  `Hz48000` (`#[non_exhaustive]`), `SampleRate::ALL`, `hz()`,
  `max_frame_size()` (the samples per channel of 120 ms), `TryFrom<u32>`,
  `TryFrom<i32>` and `From<SampleRate> for u32`. `Channels` has the values
  `Mono` and `Stereo`, `Channels::ALL`, `count()`, `TryFrom<usize>` and
  `From<Channels> for usize`. `try_from` refuses other numbers with
  `BadArg` ("invalid sampling rate", "invalid number of channels"). The
  calls that change:
  - `OpusEncoder::new(i32, usize, Application)` -> `new(SampleRate, Channels,
    Application)`; `OpusDecoder::new(i32, usize)` -> `new(SampleRate,
    Channels)`; `OpusDecoder::init(slot, i32, usize)` -> `init(slot,
    SampleRate, Channels)`. They no longer refuse a rate or a channel count.
  - `sampling_rate() -> i32` -> `sampling_rate() -> SampleRate` on
    `OpusEncoder`, `OpusDecoder`, `MultistreamEncoder` and
    `MultistreamDecoder`; `channels() -> usize` -> `channels() -> Channels` on
    `OpusEncoder` and `OpusDecoder`.
  - `set_force_channels(Option<usize>) -> Result<(), Error>` ->
    `set_force_channels(Option<Channels>)` on `OpusEncoder` and
    `MultistreamEncoder`; `force_channels() -> Option<usize>` ->
    `force_channels() -> Option<Channels>`. Stereo on a mono encoder, or on
    the mono streams of a multistream encoder, is kept and has no effect,
    where libopus refuses it.
  - The sampling rate of every constructor of `MultistreamEncoder` and
    `MultistreamDecoder` is a `SampleRate` (the constructors are renamed
    below). Their channel counts stay `usize` (1 to 255).
  - `packet::samples_per_frame(toc, i32)`, `packet::nb_samples(data, i32)`,
    `packet::multistream_validate(data, nb_streams, i32)` and
    `FrameDuration::samples(self, i32)` take a `SampleRate` instead.
  - `OpusDredDecoder::parse(dred, data, max_dred_samples, i32, defer)` takes
    the sampling rate as a `SampleRate`; its `BadArg` for a rate that is not
    positive is gone.
- Packets and packet extensions (modules `packet` and `extensions`):
  - New `packet::bandwidth(toc) -> Bandwidth` and
    `packet::nb_channels(toc) -> Channels` (libopus
    `opus_packet_get_bandwidth`, `opus_packet_get_nb_channels`).
  - The flags `self_delimited: bool` and `pad: bool` are the enums
    `packet::Framing` (`Normal`, `SelfDelimited`) and `packet::Padding`
    (`Shortest`, `FillBuffer`): `packet::parse(data, Framing)`,
    `packet::write(toc, frames, out, Framing, Padding)`,
    `Repacketizer::out_range_ext(begin, end, out, Framing, Padding,
    extensions)`, `packet::pad_with_extensions(data, len, new_len, Padding,
    extensions)` and `extensions::generate(out, extensions, nb_frames,
    Padding)`.
  - `ParsedPacket` has private fields and read-only methods: `toc()`,
    `nb_frames()` (was the field `count`), `frame_sizes()` (was
    `sizes[..count]`), `payload_offset()`, `packet_offset()` and
    `padding_len()`. Only `parse` makes one, so `frames()` and `padding()`
    cannot panic.
  - `Repacketizer` is `Clone` but no longer `Copy` (it holds about 1.6 KB).
  - `extensions::Extension` is `#[non_exhaustive]`; `Extension::new(id,
    frame, data)` makes one. Its fields stay public for reading.
  - `ExtensionIterator` implements `Iterator<Item = Result<Extension,
    Error>>`. After an error it returns `None` (libopus returns the error
    again). The inherent `next() -> Result<Option<Extension>, Error>` is
    gone, and `find(id)` is renamed `find_id(id)` so that `Iterator::find`
    stays callable.
  - No panics on bad arguments: `nb_frames` above 48 is `BadArg` in
    `ExtensionIterator` (on the first `next`), `extensions::count`,
    `count_ext`, `parse` and `parse_ext`, and an `nb_frame_exts` slice shorter
    than `nb_frames` is `BadArg` in `count_ext` and `parse_ext`.
    `extensions::count` and `count_ext` therefore return `Result<usize,
    Error>` instead of `usize`. Counts that do not match the padding no
    longer overflow in `parse_ext`.
- DRED (feature `dred`): `OpusDredDecoder::parse(dred, data,
  max_dred_samples, SampleRate, defer_processing: bool)` takes a
  `DredProcessing` (`Now`, `Deferred`, exported at the crate root) instead of
  the bool. `DredParse` is `#[non_exhaustive]`.
- Settings types and enums. The same changes apply to `OpusEncoder`,
  `OpusDecoder`, `MultistreamEncoder` and `MultistreamDecoder`:
  - `Application`, `ForcedMode`, `Signal`, `InbandFec` and
    `multistream::MappingType` are `#[non_exhaustive]`, like `ErrorKind` and
    `SampleRate`. A `match` on them needs a `_` arm.
  - New `Application::RestrictedSilk` (2052) and
    `Application::RestrictedCelt` (2053), the applications
    `OPUS_APPLICATION_RESTRICTED_SILK` and `OPUS_APPLICATION_RESTRICTED_CELT`
    of libopus 1.6, with their behaviour, also in `MultistreamEncoder`:
    SILK-only at most wideband in frames of 10 ms or more, and CELT-only
    with the packets of `RestrictedLowDelay`. `set_application` refuses to
    change to them or from them, as libopus does.
  - Every small enum derives `Debug`, `Clone`, `Copy`, `PartialEq`, `Eq` and
    `Hash`. `Application`, `Bandwidth`, `Signal`, `ForcedMode`, `InbandFec`
    and `FrameDuration` have `const fn to_raw(self) -> i32` (the libopus
    number) and `TryFrom<i32>` (`BadArg` for other numbers). `ForcedMode`
    has the numbers of libopus's `MODE_*` (1000 to 1002), `Signal::Auto` is
    `OPUS_AUTO` (-1000). `Bitrate`, `Signal`, `InbandFec` and
    `FrameDuration` implement `Default` (`Auto`, `Auto`, `Off`, `Argument`).
  - `Bandwidth::Auto` is removed: `Bandwidth` has the five real bandwidths.
    `set_bandwidth(Bandwidth)` -> `set_bandwidth(Option<Bandwidth>)`, where
    `None` is the automatic choice. New `bandwidth_setting() ->
    Option<Bandwidth>` on `OpusEncoder`; `MultistreamEncoder::bandwidth()`,
    which returned the setting, is renamed `bandwidth_setting()`.
    `set_max_bandwidth(Bandwidth)` no longer fails and returns `()`.
  - `Bitrate::BitsPerSecond(i32)` -> `Bitrate::BitsPerSecond(NonZeroU32)`.
    `set_bitrate` no longer fails and returns `()`: every value is limited to
    500 to 750,000 b/s per channel, as before.
  - Small integers: `set_complexity(u8)` and `complexity() -> u8` on all four
    codecs; `set_packet_loss_perc(u8)`, `packet_loss_perc() -> u8`,
    `set_lsb_depth(u8)` and `lsb_depth() -> u8` on both encoders;
    `OpusEncoder::set_voice_ratio(Option<u8>)` (`None` is libopus -1) and
    `voice_ratio() -> Option<u8>` (also on `MultistreamEncoder`);
    `OpusEncoder::set_dred_duration(u8)` and `dred_duration() -> u8`. They
    still refuse a complexity above 10, a loss or voice ratio above 100, a
    depth outside 8 to 24 and a DRED duration above 104. The decoders'
    `set_gain(i16)` no longer fails and returns `()`; `gain() -> i16`.
  - In-band FEC has one interface: `set_inband_fec(InbandFec)` and
    `inband_fec() -> InbandFec` on both encoders. `set_inband_fec(bool)`,
    `inband_fec() -> bool`, `set_inband_fec_mode` and `inband_fec_mode` are
    removed.
  - `OpusEncoder::set_energy_mask(Option<&[f32]>)` returns `Result<(),
    Error>`: a slice of another length than 21 values per channel is
    `BadArg` and keeps the mask. It used to clear the mask.
  - `pcm_soft_clip(pcm, frame_size, channels, declip_mem)` ->
    `pcm_soft_clip(pcm, channels, declip_mem) -> Result<(), Error>`. The
    number of samples per channel is `pcm.len() / channels`. No channels, a
    length that is not a multiple of `channels` and fewer memory values than
    channels are `BadArg`; they used to leave `pcm` unchanged silently.
- The single-stream encoder and decoder:
  - The DRED decode calls follow the names of the decode calls:
    `OpusDecoder::dred_decode_float` -> `dred_decode` (`f32` output, libopus
    `opus_decoder_dred_decode_float`), `dred_decode` -> `dred_decode_i16`
    (libopus `opus_decoder_dred_decode`). `dred_decode24` keeps its name.
  - New `OpusEncoder::init(slot, SampleRate, Channels, Application) ->
    Result<&mut OpusEncoder, Error>` builds an encoder in memory that the
    caller provides, as `OpusDecoder::init` builds a decoder. Without `heap`
    it needs less than 16 KB of stack, where `new` needs more than 256 KB.
    `OpusEncoder::reset_state` now writes the new state in place too.
  - New getters: `OpusEncoder::bandwidth() -> Bandwidth` (libopus
    `OPUS_GET_BANDWIDTH`: the bandwidth of the last packet, fullband before
    the first), `max_bandwidth() -> Bandwidth` (`OPUS_GET_MAX_BANDWIDTH`) and
    `force_mode() -> Option<ForcedMode>`; `OpusDecoder::bandwidth() ->
    Option<Bandwidth>` (`OPUS_GET_BANDWIDTH`, `None` before the first packet)
    and `nb_samples(packet) -> Result<usize, Error>`
    (`opus_decoder_get_nb_samples`).
  - A method that needs a cargo feature exists only with it. `set_osce_bwe`
    and `osce_bwe` of `OpusDecoder`, and `MultistreamDecoder::set_osce_bwe`,
    need `osce-bwe`; without it they used to return `Unimplemented`.
    `set_osce_bwe(bool)` returns `()` instead of `Result<(), Error>`, and
    `osce_bwe()` returns `bool` instead of `Result<bool, Error>`.
  - Every `set_dnn_blob` returns `Result<DnnModels, Error>`: those of
    `OpusEncoder` (feature `dred`) and `OpusDredDecoder` returned `Result<(),
    Error>`. New `DnnModels::DRED_ENCODER` (the RDOVAE encoder and PitchDNN)
    and `DnnModels::DRED_DECODER` (the RDOVAE decoder) are what they load.
  - `Debug` for `OpusEncoder`, `OpusDecoder`, `MultistreamEncoder`,
    `MultistreamDecoder`, `OpusDred` and `OpusDredDecoder`. It shows the
    settings, never the buffers. The `Debug` of `Repacketizer` no longer
    shows the frames it holds.
  - New constants `opus1::VERSION` (the crate version) and
    `opus1::LIBOPUS_VERSION` (`"1.6.1"`, the libopus release opus1 follows).
- Multistream (module `multistream`):
  - `MultistreamEncoder` and `MultistreamDecoder` have the same constructors:
    `new(rate, channels, streams, coupled_streams, mapping)`,
    `from_mapping_table(rate, &ChannelMappingTable)` and `from_layout(rate,
    &ChannelLayout)`, each with an `Application` last for the encoder. The
    encoder keeps `new_surround`. `MultistreamDecoder::new(rate,
    ChannelMappingTable)` -> `from_mapping_table(rate, &ChannelMappingTable)`;
    `MultistreamDecoder::with_layout(rate, channels, streams,
    coupled_streams, mapping)` -> `new(...)` with the same arguments.
    `MultistreamEncoder::from_mapping_table` and both `from_layout` are new;
    the encoders they build have no surround mode and no LFE stream, like
    those of `new`.
  - New getters: `channels() -> usize` on both;
    `MultistreamEncoder::bandwidth() -> Bandwidth` (libopus
    `OPUS_GET_BANDWIDTH`, the bandwidth the first stream coded last) and
    `force_mode() -> Option<ForcedMode>` (the setting);
    `MultistreamDecoder::bandwidth() -> Option<Bandwidth>` and
    `last_packet_duration() -> usize` (libopus `OPUS_GET_BANDWIDTH` and
    `OPUS_GET_LAST_PACKET_DURATION`, answered by the first stream as libopus
    does), `osce_bwe() -> bool` (feature `osce-bwe`) and `dnn_models() ->
    DnnModels` (feature `dnn`).
  - `ChannelMappingTable` is `#[non_exhaustive]`;
    `ChannelMappingTable::new(channels, stream_count, coupled_count,
    &mapping)` makes one. Its fields stay public.
  - New cargo feature `alloc`, which `std` turns on. The module
    `multistream` needs `alloc` instead of `std`, so it works in a `no_std`
    build with a global allocator (`default-features = false, features =
    ["alloc", "libm"]`). With `alloc`, the `packet` functions keep more than
    32 extensions and copy more than 1275 bytes of old padding, as with
    `std` before.
  - The documentation of `MultistreamEncoder::stream_encoder_mut` lists
    which settings of one stream can be changed, which the next `encode`
    sets again, and which break the agreement between the streams.
  - The multistream decoder writes the channels in the order of its mapping
    table, the Vorbis order for mapping family 1. opus1 does not reorder or
    downmix channels; [docs/05](05-multichannel.md#the-order-of-the-channels)
    compares the Vorbis order with the WAV order.
- `OpusEncoder::enable_hybrid_mode` is removed. It only set the audio bandwidth,
  so at higher bitrates the encoder still chose CELT. To ask for Hybrid packets,
  call `set_force_mode(Some(ForcedMode::Hybrid))`. Also call
  `set_bandwidth(Some(Bandwidth::Superwideband))` or
  `set_bandwidth(Some(Bandwidth::Fullband))`,
  so that low bitrates do not fall back to wideband SILK.
- The public field `OpusDecoder::hybrid_skip_celt` is removed. The decoder never
  read it.
- The settings of `OpusEncoder` are methods instead of public fields:
  `set_bitrate` with `bitrate_setting` and `bitrate`, `set_complexity` and
  `complexity`, `set_vbr` and `vbr` (instead of `use_cbr`, with the opposite
  meaning), `set_inband_fec` and `inband_fec`, `set_packet_loss_perc` and
  `packet_loss_perc`. Each setter checks its value as libopus's `OPUS_SET_*`
  request does. A complexity above 10 and a packet loss above 100 % are
  errors, and the setting keeps its old value. A bitrate is raised to at
  least 500 b/s and limited to at most 750,000 b/s per channel. The encoder no longer
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
  `ForcedMode`, `pcm_soft_clip`) and the modules `packet`, `extensions` and
  `multistream`. The modules `bands`, `celt`, `celt_lpc`, `hp_cutoff`,
  `kiss_fft`, `mdct`, `modes`,
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

## 24 kHz audio

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

## New features

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
- The speech and music analysis of libopus (`analysis.c`, `mlp.c`), in the
  new module `src/analysis/` (cargo feature `analysis`, on by default). From
  complexity 7 on, for input at 16 to 48 kHz, it estimates per packet how
  likely the input is speech or music, how tonal it is, whether it holds any
  sound and which bandwidth it uses. The mode, bandwidth, channel and FEC
  decisions, the activity given to SILK, and five decisions of the CELT
  encoder (allocation trim, extra bits for leaking bands, pre-filter gain, VBR
  target, coded bandwidth) use it as libopus does. On 64-bit ARM the analysis
  gives libopus's results in every field of every frame, and CELT-only packets
  at complexity 7 to 10 are libopus's byte for byte. Its NEON code takes 0.27 to
  0.40 of the time of libopus's analysis.
- Discontinuous transmission (DTX): `set_dtx`, `dtx` and `in_dtx` on
  `OpusEncoder`, `set_dtx` and `dtx` on `MultistreamEncoder` (libopus
  `OPUS_SET_DTX`, `OPUS_GET_IN_DTX`), with both kinds of DTX of libopus: that
  of the Opus layer, which uses the analysis, and SILK's.
- New encoder controls of libopus: `set_signal` with the type `Signal`
  (`OPUS_SET_SIGNAL`), `set_lsb_depth` (`OPUS_SET_LSB_DEPTH`) and the type
  `InbandFec` for `set_inband_fec` (`OPUS_SET_INBAND_FEC` 2, FEC that does
  not move music to SILK). The multistream encoder forwards
  them to every stream.
- The neural decoder features of libopus (cargo features `deep-plc`, `osce`, `osce-bwe`, off by default): deep packet loss concealment, LACE, NoLACE and the bandwidth extension, with the decoder controls `set_complexity`, `set_osce_bwe` and `set_dnn_blob` (weights loaded at run time, like `OPUS_SET_DNN_BLOB`); `MultistreamDecoder` forwards them. The shared int8 and float network kernels use the dot-product instructions of 64-bit ARM where the processor has them. libopus 1.6.1 does not decode CELT-only frames that follow frames with the bandwidth extension; opus1 applies the fix of libopus's main branch ([docs/10](10-differences-from-libopus.md)).
- `OpusDecoder::last_packet_duration()` and `packet::has_lbrr()`, as libopus has them.
- Phase inversion disabling: `set_phase_inversion_disabled` and
  `phase_inversion_disabled` on `OpusEncoder`, `OpusDecoder`,
  `MultistreamEncoder` and `MultistreamDecoder`
  (`OPUS_SET_PHASE_INVERSION_DISABLED`). CELT's intensity stereo then never
  inverts the second channel, which makes a mono downmix sound better.
- Decoder output gain and pitch: `OpusDecoder::set_gain` and `gain`
  (`OPUS_SET_GAIN`, in 1/256 dB, also on `MultistreamDecoder`) and
  `OpusDecoder::pitch` (`OPUS_GET_PITCH`).
- The 24-bit integer calls of libopus 1.6: `OpusEncoder::encode24`,
  `OpusDecoder::decode24` and `decode_fec24`, and the multistream versions
  (`opus_encode24`, `opus_decode24`, `opus_multistream_encode24`,
  `opus_multistream_decode24`).
- `OpusEncoder::set_application` and `application` (`OPUS_SET_APPLICATION`,
  before the first frame), and `set_voice_ratio` and `voice_ratio`
  (`OPUS_SET_VOICE_RATIO`).
- `MultistreamEncoder` forwards the forced channel count, the prediction and
  phase inversion switches, the application and the voice ratio as libopus
  does (difference D5 closed).
- Deep redundancy (DRED, cargo feature `dred`, off by default), as in libopus 1.5 and later: `OpusEncoder::set_dred_duration` and `dred_duration` (`OPUS_SET_DRED_DURATION`, `OPUS_GET_DRED_DURATION`), `OpusEncoder::set_dnn_blob`, and on the decoder side `OpusDredDecoder` (`opus_dred_decoder_*`, `opus_dred_parse`, `opus_dred_process`), `OpusDred` (`opus_dred_alloc`) and `OpusDecoder::dred_decode` (`f32` output, `opus_decoder_dred_decode_float`), `dred_decode_i16` (`opus_decoder_dred_decode`) and `dred_decode24` (`opus_decoder_dred_decode24`). Without the feature, or with a duration of 0, the packets do not change.
- Packet extensions of libopus 1.5 and later: the module `extensions` (libopus `src/extensions.c`), `packet::Repacketizer` (libopus `OpusRepacketizer`), which carries extensions, and `packet::pad_with_extensions`, `packet::unpad` and `packet::multistream_unpad`.
- Evaluation tools for the repository (not public API): exact ports of libopus's `opus_compare` and of the MOC distance of the OSCE draft, the RFC 8251 conformance check (all 12 test vectors pass at every rate), and a paired comparison of opus1 with libopus (`examples/eval`, see [docs/09](09-testing.md#evaluation-tools)).
- `OpusDecoder::init(slot, sampling_rate, channels)` builds a decoder in memory that the caller provides (`&mut MaybeUninit<OpusDecoder>`, for example a `static`) and returns `&mut OpusDecoder`. Upstream opus-rs added the same function in PR #65 (by Alix ANNERAUD); opus1 implements it in its own way, because its decoder also holds the state of the neural network features. Every part of the decoder is written in place, field by field: the CELT and SILK decoders, the OSCE and BWE states, deep PLC and the model holder. A macro, `write_fields!`, checks at compile time that every field of each struct is written. `new` now goes through the same code. The tables that every decoder shares (the CELT mode and the built-in models) are also built in place, on first use. Measured in a release build for aarch64 without `heap`: `init` needs less than 16 KB of stack, also with `osce-bwe` and `dred` and on the first call of a program, where `new` needs about 192 KB (512 KB with those features). With `heap`, `init` writes the boxed parts straight into their heap memory. Tested by `tests/decoder_init.rs`, adapted from upstream: the output of a decoder from `init` equals that of one from `new` (five settings, with lost packets, at complexity 0 and 10), and `init` and decoding run on a thread with a 32 KB stack. `OpusEncoder::init` builds an encoder in the same way: the CELT and SILK encoders, the speech and music analysis and the DRED encoder are written in place, and `tests/encoder_init.rs` checks that its packets equal those of `new` and that `init` and `reset_state` run on a 32 KB stack. The multistream encoder and decoder have no `init` yet.
- The feature `portable-atomic`: the crate builds for targets without atomic compare-and-swap (CAS), such as `thumbv6m-none-eabi` (Cortex-M0/M0+) and `riscv32imc-unknown-none-elf` (ESP32-C3), as upstream opus-rs PR #64 (by Alix ANNERAUD) does. The only compare-and-swap of the crate guards the lazily built shared tables (`compat::OnceCell`); with the feature its atomic comes from the `portable-atomic` crate. `Arc` is used only with `heap`, which needs `std`. On such a target the application also sets portable-atomic's `critical-section` feature, or `--cfg portable_atomic_unsafe_assume_single_core` on a single-core chip. Checked with `cargo build --target thumbv6m-none-eabi --no-default-features --features libm,portable-atomic` (also with `analysis`, and with `osce-bwe,dred`); `scripts/check_no_std.sh` now builds it.

## Behaviour changes

- Float output is no longer clamped to [-1, 1] (libopus does not clip float output).
  Use `decode_i16` for 16-bit output.
- `packet::pad` and `packet::multistream_pad` keep the extensions in the padding
  of a packet, and fail with `ErrorKind::InternalError` when the padding is not valid
  extension data, as libopus does. Before, they wrote zeros.
- The automatic mode choice follows libopus. At 64 kb/s a 16 kHz VoIP encoder
  now codes CELT, as libopus does at that bitrate. At speech rates (for example
  16-24 kb/s, or the automatic 19 kb/s of 20 ms frames) it codes SILK.
- Stereo input is coded as mono at low bitrates, as libopus does.
- `encode_i16` codes with an input precision of 16 bits, as libopus
  `opus_encode` does. From complexity 7 on its packets can differ from those
  of `encode` with the default precision of 24 bits; `set_lsb_depth(16)` makes
  them equal.
- From complexity 7 on, the speech and music analysis changes the packets of
  every stream at 16 to 48 kHz, towards those of libopus.
- Encoder output changed to follow libopus in many places (stereo pitch
  pre-filter, `theta_rdo` at complexity 8 and above, SILK pitch analysis, pulse
  search, band energies, FFT and MDCT rounding on aarch64). A voiced 300-packet
  SILK stream is byte-identical to fixed-point libopus.
- With `Voip` at 48 kHz, SILK now reads the float high-pass filter's output,
  as the float build of libopus does. The integer filter of the fixed-point
  build added a constant of about -9 to the 16-bit signal there, which SILK
  coded as low-frequency energy in 7.6 to 9.2 % of the 10 ms frames of speech.
  On 222 TSP speech files the error against the original, relative to
  libopus's, fell from between 1.21 and 3.28 to between 0.98 and 1.09
  (complexity 5, 20 ms, 6 to 64 kb/s). At 8 to 24 kHz SILK keeps the integer
  filter ([docs/10](10-differences-from-libopus.md#paired-comparison-with-the-evaluation-harness)).
- With `Voip` at 8 to 24 kHz the integer high-pass filter in front of SILK
  runs on every frame, as in the fixed-point build of libopus. It ran only on
  frames that SILK coded, so the first SILK frame after CELT-only frames
  started from an old filter state, which SILK coded as energy below 200 Hz.
  Streams that change between SILK and CELT-only no longer have that energy.
  Streams without CELT-only frames do not change.
- SILK's NLSF weights follow libopus for the last coefficient (distance to
  `1 << 15`, not to 32767). Stereo SILK streams that left the fixed-point
  build's packets after 274 and 322 packets now stay byte-identical.

## Closer to libopus (aarch64, against libopus 1.6.1)

- The CELT encoder writes the packets of libopus: all 1,080,000 packets of the
  960 streams of `celtenc_gap_report` (mono and stereo, 16 and 48 kHz, 2.5 to
  20 ms, CBR and VBR, complexity 4 to 9), up from 91.2 %. Through `OpusEncoder`,
  with the speech and music analysis, CELT-only packets at complexity 7 to 10
  are libopus's too. Stereo input whose channels are scaled copies of
  each other, and very quiet signals such as the decaying output of the DC
  filter after a signal stops, give the packets of libopus too
  (`intensity_stereo`, the transient analysis and the fades of the Opus layer
  now round as libopus does).
- Digital silence is handled as libopus handles it: CELT measures the silence
  flag of a stereo input coded as mono over the first half of the frame, and
  SILK is told that a frame of digital silence has no voice activity.
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

## Performance (Apple M4 Pro, see the [README](../README.md#performance))

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

## Fuzzing

- The 19 fuzz targets in `fuzz/` build and run on the stable toolchain
  (`fuzz/fuzz.sh`), with overflow checks on. Six of them compare with
  libopus: packet parsing, padding and writing; a decoder fed packet
  streams with losses and FEC; the multistream decoder and encoder; an
  encoder whose settings change between frames; and the channel order of
  both encoders. They also check the packet length of a constant bitrate
  against libopus's rule, and that the multistream decoder writes only the
  samples it returns.

## x86-64

- x86-64 was run once, on an Intel Core i7-12700, on 2026-10-21: under Linux in WSL2 and natively on Windows (`scripts/x86/test.sh` and `test.ps1`). The output is correct but not bit-exact with libopus on x86. About 40 tests that require libopus's bits are marked `ignore` on targets other than 64-bit ARM ([docs/10](10-differences-from-libopus.md#platforms)).
- A memory error in an x86 kernel was found and fixed: the AVX2 pulse search (`pvq_search_avx2`) could add a pulse past the end of its output when every score was 0 (an all-zero band or NaN input). The x86 kernels now check their slice lengths at entry.
- The AVX2 version of `haar1` computed `(a+b)*s` where libopus computes `s*a + s*b`. It now follows libopus.

## Known gaps (measured against libopus 1.6.1)

- SILK stereo with a constant bitrate at 12 kHz can lose its stereo image on
  tones, as the fixed-point build of libopus does; the float build keeps it
  ([docs/10](10-differences-from-libopus.md#silk)).

## Fixes made after 0.1.37, before the rename

These fixes were listed in the README as "Unreleased" while the crate was still named opus-rs.

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
