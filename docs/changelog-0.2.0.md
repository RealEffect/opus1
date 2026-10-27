# Changes in opus1 0.2.0

This file keeps the detailed list of changes from opus-rs 0.1.37 to opus1 0.2.0, with the measurements behind each fix. The short release note is in the [README](../README.md#020-unreleased).

The crate is renamed from `opus-rs` to `opus1`. Code that used `opus_rs::` now uses `opus1::`.

This release was checked against libopus 1.6.1 throughout: every fix below has a
test that compares opus1 with the C library under the same settings.

## Breaking changes to the public interface

This list is what code written for opus-rs 0.1.37 must change. The public
interface of 0.1.37 was small: `OpusEncoder` (`new`, `encode`,
`encode_i16`, `enable_hybrid_mode` and five public fields of settings),
`OpusDecoder` (`new`, `decode` and the public field `hybrid_skip_celt`),
the enums `Application` and `Bandwidth`, the internal modules, and with
`std` the module `multistream` (`ChannelMappingTable`,
`split_self_delimited`, and `MultistreamDecoder` with `new` and `decode`).
Everything else is new in 0.2.0. It is listed under
[New features](#new-features) in its 0.2.0 form, also where it changed
while 0.2.0 was developed.

- Errors: every call that can fail returns `Result<T, opus1::Error>` instead
  of `Result<T, &'static str>`. Of the calls of 0.1.37 these are
  `OpusEncoder::new`, `encode` and `encode_i16`, `OpusDecoder::new` and
  `decode`, `MultistreamDecoder::decode` and the constructor that replaces
  `MultistreamDecoder::new` (see below), and `split_self_delimited`.
  - `Error::kind()` returns an `ErrorKind` with one value per libopus error
    code: `BadArg`, `BufferTooSmall`, `InternalError`, `InvalidPacket`,
    `Unimplemented`, `InvalidState` and `AllocFail` (`#[non_exhaustive]`).
    Each call returns the kind that libopus returns in the same case.
  - `Error::to_raw()` and `ErrorKind::to_raw()` give the libopus code (-1
    to -7). `Error::message()` and `Display` give a short text. The texts
    start with a lowercase letter, have no final period, and one error has
    one wording: for example every output buffer that is too small gives
    "output buffer too small".
  - `Error` is `Copy` and implements `core::error::Error`.
  - Code that compared message strings now compares `err.kind()`. One kind
    differs from what the old message suggested: an empty output buffer of
    `encode` is `BadArg` (libopus `max_data_bytes <= 0`), where 0.1.37 said
    "Output buffer too small".
- Sampling rates and channel counts have their own types at the crate root.
  `OpusEncoder::new(i32, usize, Application)` -> `new(SampleRate, Channels,
  Application)`, and `OpusDecoder::new(i32, usize)` -> `new(SampleRate,
  Channels)`. They no longer refuse a rate or a channel count, because the
  types hold only valid ones.
  - `SampleRate` has the values `Hz8000`, `Hz12000`, `Hz16000`, `Hz24000`
    and `Hz48000` (`#[non_exhaustive]`), `SampleRate::ALL` (a slice,
    `&'static [SampleRate]`, so that a later rate does not change its
    type), `hz()`, `max_frame_size()` (the samples per channel of 120 ms),
    `TryFrom<u32>`, `TryFrom<i32>` and `From<SampleRate> for u32`.
  - `Channels` has the values `Mono` and `Stereo`, `Channels::ALL` (an
    array, `[Channels; 2]`), `count()`, `TryFrom<usize>` and
    `From<Channels> for usize`.
  - `try_from` refuses other numbers with `BadArg` ("invalid sampling
    rate", "invalid number of channels"). Both types derive `Debug`,
    `Clone`, `Copy`, `PartialEq`, `Eq`, `PartialOrd`, `Ord` and `Hash`.
  - The multistream decoder takes a `SampleRate` too (see below).
- The settings of `OpusEncoder` are methods instead of public fields. Each
  setter checks its value as libopus's `OPUS_SET_*` request does. A value
  that libopus refuses is an error of kind `BadArg`, and the setting keeps
  its old value. The encoder no longer clamps the settings on every
  `encode` call (0.1.37 limited the bitrate to 500 to 3,000,000 b/s there).
  - `bitrate_bps: i32` -> `set_bitrate(Bitrate)`, with the getters
    `bitrate_setting() -> Bitrate` (the setting) and `bitrate() -> i32`
    (libopus `OPUS_GET_BITRATE`: the bits per second the setting gives).
    `Bitrate` is `Auto` (libopus `OPUS_AUTO`, the default), `Max`
    (`OPUS_BITRATE_MAX`) or `BitsPerSecond(NonZeroU32)`. `set_bitrate`
    cannot fail: as in libopus, a bitrate is raised to at least 500 b/s and
    limited to at most 750,000 b/s per channel.
  - `complexity: i32` -> `set_complexity(u8) -> Result<(), Error>` and
    `complexity() -> u8`. A complexity above 10 is refused.
  - `use_cbr: bool` -> `set_vbr(bool)` and `vbr() -> bool`, with the
    opposite meaning.
  - `use_inband_fec: bool` -> `set_inband_fec(InbandFec)` and
    `inband_fec() -> InbandFec`. `InbandFec` is `Off` (the default), `On`
    or `OnUnlessMusic` (libopus `OPUS_SET_INBAND_FEC` 0, 1 and 2).
  - `packet_loss_perc: i32` -> `set_packet_loss_perc(u8) -> Result<(),
    Error>` and `packet_loss_perc() -> u8`. A loss above 100 % is refused.
- The default bitrate of `OpusEncoder` is libopus's automatic one,
  `Bitrate::Auto`: 60 × rate / frame size + rate × channels b/s, for example
  19 kb/s for 16 kHz mono and 99 kb/s for 48 kHz stereo in 20 ms frames. It
  was a fixed 64 kb/s. To keep 64 kb/s, call
  `set_bitrate(Bitrate::BitsPerSecond(NonZeroU32::new(64_000).unwrap()))`.
- `OpusEncoder::enable_hybrid_mode` is removed. It only set the audio
  bandwidth, so at higher bitrates the encoder still chose CELT. To ask for
  Hybrid packets, call `set_force_mode(Some(ForcedMode::Hybrid))`. Also call
  `set_bandwidth(Some(Bandwidth::Superwideband))` or
  `set_bandwidth(Some(Bandwidth::Fullband))`, so that low bitrates do not
  fall back to wideband SILK.
- The public field `OpusDecoder::hybrid_skip_celt` is removed. The decoder
  never read it.
- `OpusDecoder::decode` no longer sets the part of `output` after the
  decoded samples to zero. As in libopus, it writes only the samples it
  returns. A program that relied on the zeros must clear the buffer itself.
- `Application` is `#[non_exhaustive]` and has two more values,
  `RestrictedSilk` and `RestrictedCelt` (see [New features](#new-features)).
  A `match` on it needs a `_` arm.
- `Bandwidth::Auto` is removed: `Bandwidth` has the five real bandwidths.
  Where the automatic choice is meant, an `Option<Bandwidth>` is `None`, as
  in `OpusEncoder::set_bandwidth`.
- Multistream (module `multistream`):
  - `MultistreamDecoder::new(i32, ChannelMappingTable)` ->
    `MultistreamDecoder::from_mapping_table(SampleRate,
    &ChannelMappingTable)`. The table is checked as libopus checks a
    layout, and an invalid one is `BadArg`. The name `new` now takes the
    layout as numbers: `new(SampleRate, channels, streams, coupled_streams,
    &mapping)`.
  - `ChannelMappingTable` is `#[non_exhaustive]`, so code outside opus1
    cannot build it with a struct expression.
    `ChannelMappingTable::new(channels, stream_count, coupled_count,
    &mapping)` builds one, and `ChannelMappingTable::parse` reads one as
    before. Its fields stay public.
- The internal modules are private. The public interface is the crate root
  and the modules `packet`, `extensions` and `multistream`. The modules
  `bands`, `celt`, `celt_lpc`, `hp_cutoff`, `kiss_fft`, `mdct`, `modes`,
  `pitch`, `pvq`, `quant_bands`, `range_coder`, `rate` and `silk`, the types
  `CeltEncoder`, `CeltDecoder`, `SilkResampler`, `SilkResamplerDown1_3` and
  `SilkResamplerDown1_6` at the crate root, and the macro
  `tell_frac_inline!` are no longer exported.
  - The crate root holds `OpusEncoder`, `OpusDecoder`, `SampleRate`,
    `Channels`, `Application`, `Bandwidth`, `Bitrate`, `ForcedMode`,
    `FrameDuration`, `Signal`, `InbandFec`, `Error`, `ErrorKind`,
    `pcm_soft_clip`, `VERSION` and `LIBOPUS_VERSION`; with `deep-plc` (or a
    feature that turns it on) also `DnnBlob` and `DnnModels`, and with
    `dred` also `OpusDred`, `OpusDredDecoder`, `DredParse` and
    `DredProcessing`.
  - The tests, benchmarks, examples and fuzz targets of the repository
    reach internal items through the hidden module `opus1::internals`. It
    exists only with the cargo feature `__internals`, which is for them
    only and outside semantic versioning; applications never enable it.
    The other hidden test hooks need it too: `DnnBlob::weights()` and the
    implementation of the evaluation trait `DemoDecoder` for `OpusDecoder`.
- The feature `heap` turns on `alloc` instead of `std`. In 0.1.37 a build
  with `default-features = false, features = ["heap"]` was a `std` build.
  Now it is a `no_std` build, which needs `libm` as well (or `std`). The
  default features (`std`, `heap` and the new `analysis`) still build with
  `std`.
- The minimum Rust version is 1.98 (`rust-version` in Cargo.toml): the
  64-bit ARM dot-product intrinsics of the neural network code are stable
  from that version on. opus-rs 0.1.37 declared no minimum; its edition
  2024 needs Rust 1.85 or later.

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

Everything in this section is new since opus-rs 0.1.37, and is described in
its 0.2.0 form.

- Multichannel audio, module `multistream`: `MultistreamEncoder` (surround
  families 0, 1, 2 and 255, rate allocation, LFE stream, surround masking)
  and a general `MultistreamDecoder` (any valid layout, `f32`, 16-bit and
  24-bit output, loss concealment, FEC). Packets interoperate with libopus's
  multistream encoder and decoder in both directions.
  - Both have the same constructors: `new(rate, channels, streams,
    coupled_streams, &mapping)`, `from_mapping_table(rate,
    &ChannelMappingTable)` and `from_layout(rate, &ChannelLayout)`, each
    with an `Application` last for the encoder. The rate is a `SampleRate`,
    and the channel counts are `usize` (1 to 255). The encoder also has
    `new_surround(rate, channels, mapping_family, application)`, which
    chooses the streams of a mapping family as libopus does. The encoders
    of the other three constructors have no surround mode and no LFE
    stream.
  - The module needs the new cargo feature `alloc`, which `std` and `heap`
    turn on. So it also works in a `no_std` build with a global allocator
    (`default-features = false, features = ["alloc", "libm"]`).
  - The constructors build every stream encoder and decoder in place, in
    the memory of the multistream encoder or decoder (`OpusEncoder::init`,
    `OpusDecoder::init`). So they need less than 16 KB of stack, also
    without `heap`, where one stream encoder holds up to 210 KB. Encoding,
    decoding and `reset_state` allocate nothing.
  - Getters: `layout()`, `channels() -> usize` and `sampling_rate() ->
    SampleRate` on both; `MultistreamEncoder::bandwidth() -> Bandwidth`
    (libopus `OPUS_GET_BANDWIDTH`, the bandwidth the first stream coded
    last), `bandwidth_setting() -> Option<Bandwidth>` and `force_mode() ->
    Option<ForcedMode>` (the settings); `MultistreamDecoder::bandwidth() ->
    Option<Bandwidth>` and `last_packet_duration() -> usize` (libopus
    `OPUS_GET_BANDWIDTH` and `OPUS_GET_LAST_PACKET_DURATION`, answered by
    the first stream as libopus does), `osce_bwe() -> bool` (feature
    `osce-bwe`) and `dnn_models() -> DnnModels` (feature `deep-plc`).
  - `ChannelMappingTable::new(channels, stream_count, coupled_count,
    &mapping)` makes a table. `ChannelLayout`, `MappingType`
    (`#[non_exhaustive]`) and `validate_ambisonics` are in
    `opus1::multistream` too.
  - The documentation of `MultistreamEncoder::stream_encoder_mut` lists
    which settings of one stream can be changed, which the next `encode`
    sets again, and which break the agreement between the streams.
  - The multistream decoder writes the channels in the order of its mapping
    table, the Vorbis order for mapping family 1. opus1 does not reorder or
    downmix channels; [docs/guide/05](guide/05-multichannel.md#the-order-of-the-channels)
    compares the Vorbis order with the WAV order.
- `MultistreamEncoder` forwards the forced channel count, the prediction and
  phase inversion switches, the application and the voice ratio as libopus
  does (difference D5 closed).
- `OpusDecoder`: `decode_i16` (soft clipping as `opus_decode`), `decode_fec`,
  `decode_fec_i16`, `final_range`, `reset_state`; packets whose channel count
  differs from the decoder's; CELT packet loss concealment ported from libopus.
  New getters: `sampling_rate() -> SampleRate`, `channels() -> Channels`,
  `bandwidth() -> Option<Bandwidth>` (libopus `OPUS_GET_BANDWIDTH`, `None`
  before the first packet) and `nb_samples(packet) -> Result<usize, Error>`
  (`opus_decoder_get_nb_samples`).
- `OpusEncoder` controls of libopus, with their getters:
  - `set_force_mode(Option<ForcedMode>)` and `force_mode()`
    (`OPUS_SET_FORCE_MODE`). `ForcedMode` is `SilkOnly`, `Hybrid` or
    `CeltOnly`, with the numbers of libopus's `MODE_*` (1000 to 1002);
    `None` is the automatic choice.
  - `set_bandwidth(Option<Bandwidth>)` and `bandwidth_setting() ->
    Option<Bandwidth>` (`OPUS_SET_BANDWIDTH`; `None` is the automatic
    choice, the default). `bandwidth() -> Bandwidth` is
    `OPUS_GET_BANDWIDTH`: the bandwidth of the last packet, fullband before
    the first. `set_max_bandwidth(Bandwidth)` and `max_bandwidth()`
    (`OPUS_SET_MAX_BANDWIDTH`).
  - `set_force_channels(Option<Channels>)` and `force_channels() ->
    Option<Channels>` (`OPUS_SET_FORCE_CHANNELS`). Stereo on a mono
    encoder, or on the mono streams of a multistream encoder, is kept and
    has no effect, where libopus refuses it.
  - `set_lfe`, `final_range`, `reset_state`, `lookahead`,
    `sampling_rate() -> SampleRate` and `channels() -> Channels`.
  - `set_energy_mask(Option<&[f32]>) -> Result<(), Error>` takes 21 values
    per channel. A slice of another length is `BadArg` and keeps the mask.
- The settings types are enums with the numbers of libopus.
  `Application`, `Bandwidth`, `Signal`, `ForcedMode`, `InbandFec` and
  `FrameDuration` have `const fn to_raw(self) -> i32` (the libopus number)
  and `TryFrom<i32>` (`BadArg` for other numbers). `Signal::Auto` is
  `OPUS_AUTO` (-1000). `Bitrate`, `Signal`, `InbandFec` and
  `FrameDuration` implement `Default` (`Auto`, `Auto`, `Off`, `Argument`).
  Every small enum derives `Debug`, `Clone`, `Copy`, `PartialEq`, `Eq` and
  `Hash`. `Application`, `ForcedMode`, `Signal`, `InbandFec`,
  `multistream::MappingType`, `ErrorKind` and `SampleRate` are
  `#[non_exhaustive]`, so a `match` on them needs a `_` arm.
- `Bitrate` is at the crate root, and both encoders take it in
  `set_bitrate`: `Bitrate::Auto` (libopus `OPUS_AUTO`), `Bitrate::Max`
  (`OPUS_BITRATE_MAX`) and `Bitrate::BitsPerSecond(NonZeroU32)`.
- `set_vbr_constraint` on `OpusEncoder` and `MultistreamEncoder` (libopus
  `OPUS_SET_VBR_CONSTRAINT`). Turning the constraint off gives CELT-only
  packets an unconstrained variable bitrate.
- `Application::RestrictedSilk` (2052) and `Application::RestrictedCelt`
  (2053), the applications `OPUS_APPLICATION_RESTRICTED_SILK` and
  `OPUS_APPLICATION_RESTRICTED_CELT` of libopus 1.6, with their behaviour,
  also in `MultistreamEncoder`: SILK-only at most wideband in frames of
  10 ms or more, and CELT-only with the packets of `RestrictedLowDelay`.
  This holds for every packet with audio. A packet without audio, which the
  encoder writes when the bitrate or the output buffer is too small to code
  a frame, names the mode of the last coded frame in its TOC byte, as in
  libopus: Hybrid before the first coded frame and after `reset_state`
  ([docs/guide/02](guide/02-packets.md#packets-without-audio)). `set_application`
  refuses to change to them or from them, as libopus does.
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
- The module `packet`: parse, write and pad packets, with self-delimited
  framing (libopus `opus_packet_*` and the repacketizer).
  - `packet::parse(data, Framing) -> Result<ParsedPacket, Error>`.
    `ParsedPacket` has read-only methods: `toc()`, `nb_frames()`,
    `frame_sizes()`, `payload_offset()`, `packet_offset()`,
    `padding_len()`, `frames(data)`, `frame(data, i)` and `padding(data)`.
    Only `parse` makes one.
  - `packet::write(toc, frames, out, Framing, Padding)`, `pad(data, len,
    new_len)`, `unpad`, `multistream_pad`, `multistream_unpad` and
    `multistream_validate(data, nb_streams, SampleRate)`.
  - The TOC getters `samples_per_frame(toc, SampleRate)`,
    `bandwidth(toc) -> Bandwidth` and `nb_channels(toc) -> Channels`
    (libopus `opus_packet_get_samples_per_frame`,
    `opus_packet_get_bandwidth`, `opus_packet_get_nb_channels`), and
    `nb_frames(data)` and `nb_samples(data, SampleRate)`.
  - The enums `Framing` (`Normal`, `SelfDelimited`) and `Padding`
    (`Shortest`, `FillBuffer`) stand for libopus's flags `self_delimited`
    and `pad`.
  - `pad` and `multistream_pad` keep the extensions in the padding of a
    packet, and fail with `ErrorKind::InternalError` when the padding is not
    valid extension data, as libopus does.
- Packet extensions of libopus 1.5 and later: the module `extensions`
  (libopus `src/extensions.c`), `packet::Repacketizer` (libopus
  `OpusRepacketizer`), which carries extensions, and
  `packet::pad_with_extensions`, `packet::unpad` and
  `packet::multistream_unpad`.
  - `Extension` (`#[non_exhaustive]`) has the public fields `id`, `frame`
    and `data`; `Extension::new(id, frame, data)` makes one.
  - `ExtensionIterator` implements `Iterator<Item = Result<Extension,
    Error>>`. After an error it returns `None` (libopus returns the error
    again). `find_id(id)` looks for one ID; it is not called `find`, so
    that `Iterator::find` stays callable.
  - `extensions::count` and `count_ext` return `Result<usize, Error>`;
    `parse`, `parse_ext`, `generate(out, extensions, nb_frames, Padding)`
    and `generated_len` complete the module.
  - No call panics on bad arguments: `nb_frames` above 48 is `BadArg` in
    `ExtensionIterator` (on the first `next`), `count`, `count_ext`,
    `parse` and `parse_ext`, and an `nb_frame_exts` slice shorter than
    `nb_frames` is `BadArg` in `count_ext` and `parse_ext`. libopus checks
    these only with debug assertions, or not at all.
  - `Repacketizer::out_range_ext(begin, end, out, Framing, Padding,
    extensions)` has the options of libopus's internal
    `opus_repacketizer_out_range_impl`. `Repacketizer` is `Clone` but not
    `Copy` (it holds about 1.6 KB).
  - `packet::pad_with_extensions(data, len, new_len, Padding, extensions)`
    returns the length of the packet, also when there is nothing to do.
    Extensions it is given are always written, or the call fails with
    `BufferTooSmall` when they do not fit. Here it differs from libopus's
    internal `opus_packet_pad_impl`, which returns 0 when `len` equals
    `new_len` and leaves the extensions out.
  - Without the feature `alloc`, more than 32 extensions or more than 1275
    bytes of old padding in one call are `AllocFail`.
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
  (`OPUS_SET_SIGNAL`), `set_lsb_depth(u8)` (8 to 24, `OPUS_SET_LSB_DEPTH`)
  and `InbandFec::OnUnlessMusic` for `set_inband_fec` (`OPUS_SET_INBAND_FEC`
  2, FEC that does not move music to SILK). The multistream encoder forwards
  them to every stream.
- The neural decoder features of libopus (cargo features `deep-plc`, `osce`, `osce-bwe`, off by default): deep packet loss concealment, LACE, NoLACE and the bandwidth extension, with the decoder controls `set_complexity(u8)`, `set_osce_bwe(bool)` and `osce_bwe() -> bool` (feature `osce-bwe`; without it the methods do not exist) and `set_dnn_blob(&DnnBlob) -> Result<DnnModels, Error>` with `dnn_models()` (weights loaded at run time, like `OPUS_SET_DNN_BLOB`); `MultistreamDecoder` forwards them. `DnnBlob` and `DnnModels` need the feature `deep-plc`, which `osce`, `osce-bwe` and `dred` turn on; the feature `dnn` alone adds no public API. The shared int8 and float network kernels use the dot-product instructions of 64-bit ARM where the processor has them. libopus 1.6.1 does not decode CELT-only frames that follow frames with the bandwidth extension; opus1 applies the fix of libopus's main branch ([docs/guide/10](guide/10-differences-from-libopus.md)).
- `OpusDecoder::last_packet_duration()` and `packet::has_lbrr()`, as libopus has them.
- Phase inversion disabling: `set_phase_inversion_disabled` and
  `phase_inversion_disabled` on `OpusEncoder`, `OpusDecoder`,
  `MultistreamEncoder` and `MultistreamDecoder`
  (`OPUS_SET_PHASE_INVERSION_DISABLED`). CELT's intensity stereo then never
  inverts the second channel, which makes a mono downmix sound better.
- Decoder output gain and pitch: `OpusDecoder::set_gain(i16)` and
  `gain() -> i16` (`OPUS_SET_GAIN`, in 1/256 dB, also on
  `MultistreamDecoder`) and `OpusDecoder::pitch` (`OPUS_GET_PITCH`).
- The 24-bit integer calls of libopus 1.6: `OpusEncoder::encode24`,
  `OpusDecoder::decode24` and `decode_fec24`, and the multistream versions
  (`opus_encode24`, `opus_decode24`, `opus_multistream_encode24`,
  `opus_multistream_decode24`).
- `OpusEncoder::set_application` and `application` (`OPUS_SET_APPLICATION`,
  before the first frame), and `set_voice_ratio(Option<u8>)` and
  `voice_ratio() -> Option<u8>` (`OPUS_SET_VOICE_RATIO`, 0 to 100; `None` is
  libopus -1).
- `pcm_soft_clip(pcm, channels, declip_mem) -> Result<(), Error>` (libopus
  `opus_pcm_soft_clip`). The number of samples per channel is
  `pcm.len() / channels`. No channels, a length that is not a multiple of
  `channels` and fewer memory values than channels are `BadArg`.
- Deep redundancy (DRED, cargo feature `dred`, off by default), as in libopus 1.5 and later: `OpusEncoder::set_dred_duration(u8)` and `dred_duration() -> u8` (`OPUS_SET_DRED_DURATION`, `OPUS_GET_DRED_DURATION`, 0 to 104 frames of 10 ms), `OpusEncoder::set_dnn_blob`, and on the decoder side `OpusDredDecoder` (`opus_dred_decoder_*`, `opus_dred_parse`, `opus_dred_process`), `OpusDred` (`opus_dred_alloc`) and `OpusDecoder::dred_decode` (`f32` output, `opus_decoder_dred_decode_float`), `dred_decode_i16` (`opus_decoder_dred_decode`) and `dred_decode24` (`opus_decoder_dred_decode24`). `OpusDredDecoder::parse(dred, data, max_dred_samples, SampleRate, DredProcessing)` returns a `DredParse` (`#[non_exhaustive]`); `DredProcessing::Deferred` is libopus's `defer_processing`. Every `set_dnn_blob` returns `Result<DnnModels, Error>`; `DnnModels::DRED_ENCODER` (the RDOVAE encoder and PitchDNN) and `DnnModels::DRED_DECODER` (the RDOVAE decoder) are what the DRED encoder and decoder load. Without the feature, or with a duration of 0, the packets do not change.
- `Debug` for `OpusEncoder`, `OpusDecoder`, `MultistreamEncoder`,
  `MultistreamDecoder`, `OpusDred` and `OpusDredDecoder`. It shows the
  settings, never the buffers.
- The constants `opus1::VERSION` (the crate version) and
  `opus1::LIBOPUS_VERSION` (`"1.6.1"`, the libopus release opus1 follows).
- Evaluation tools for the repository (not public API): exact ports of libopus's `opus_compare` and of the MOC distance of the OSCE draft, the RFC 8251 conformance check (all 12 test vectors pass at every rate), and a paired comparison of opus1 with libopus (`examples/eval`, see [docs/guide/09](guide/09-testing.md#evaluation-tools)).
- `OpusDecoder::init(slot, sampling_rate, channels)` builds a decoder in memory that the caller provides (`&mut MaybeUninit<OpusDecoder>`, for example a `static`) and returns `&mut OpusDecoder`. Upstream opus-rs added the same function in PR #65 (by Alix ANNERAUD); opus1 implements it in its own way, because its decoder also holds the state of the neural network features. Every part of the decoder is written in place, field by field: the CELT and SILK decoders, the OSCE and BWE states, deep PLC and the model holder. A macro, `write_fields!`, checks at compile time that every field of each struct is written. `new` now goes through the same code. The tables that every decoder shares (the CELT mode and the built-in models) are also built in place, on first use. Measured in a release build for aarch64 without `heap`: `init` needs less than 16 KB of stack, also with `osce-bwe` and `dred` and on the first call of a program, where `new` needs about 192 KB (512 KB with those features). With `heap`, `init` writes the boxed parts straight into their heap memory. Tested by `tests/decoder_init.rs`, adapted from upstream: the output of a decoder from `init` equals that of one from `new` (five settings, with lost packets, at complexity 0 and 10), and `init` and decoding run on a thread with a 32 KB stack. `OpusEncoder::init(slot, SampleRate, Channels, Application) -> Result<&mut OpusEncoder, Error>` builds an encoder in the same way: the CELT and SILK encoders, the speech and music analysis and the DRED encoder are written in place, and `tests/encoder_init.rs` checks that its packets equal those of `new` and that `init` and `reset_state` run on a 32 KB stack. Without `heap` it needs less than 16 KB of stack, where `new` needs more than 256 KB. The multistream encoder and decoder have no `init` yet, but their constructors build each stream codec in place with these functions (`tests/multistream_small_stack.rs`).
- `OpusEncoder::reset_state` writes the new state in place, as libopus `OPUS_RESET_STATE`: with `heap` into the heap memory the encoder has, so a reset allocates nothing.
- The feature `portable-atomic`: the crate builds for targets without atomic compare-and-swap (CAS), such as `thumbv6m-none-eabi` (Cortex-M0/M0+) and `riscv32imc-unknown-none-elf` (ESP32-C3), as upstream opus-rs PR #64 (by Alix ANNERAUD) does. The only compare-and-swap of the crate guards the lazily built shared tables (`compat::OnceCell`); with the feature its atomic comes from the `portable-atomic` crate. `Arc` is used only by `DnnBlob::from_bytes` (with `heap`), which such a target does not have. On such a target the application also sets portable-atomic's `critical-section` feature, or `--cfg portable_atomic_unsafe_assume_single_core` on a single-core chip. Checked with `RUSTFLAGS="--cfg portable_atomic_unsafe_assume_single_core" cargo build --target thumbv6m-none-eabi --no-default-features --features libm,portable-atomic` (also with `analysis`, and with `osce-bwe,dred`); `scripts/check_no_std.sh` now builds it.
- The feature `heap` also works in a `no_std` build with a global allocator
  (`default-features = false, features = ["heap", "libm"]`). There
  `DnnBlob::from_bytes` needs a target with atomic compare-and-swap.
- The documentation on docs.rs covers the neural network features: it is
  built with `dred` and `osce-bwe`, which turn on the others
  (`[package.metadata.docs.rs]` in Cargo.toml). Each item that needs a
  feature is labelled with it.

## Behaviour changes

- Float output is no longer clamped to [-1, 1] (libopus does not clip float output).
  Use `decode_i16` for 16-bit output.
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
  filter ([docs/guide/10](guide/10-differences-from-libopus.md#paired-comparison-with-the-evaluation-harness)).
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

- The 24 fuzz targets in `fuzz/` build and run on the stable toolchain
  (`fuzz/fuzz.sh`), with overflow checks on. Eight of them compare with
  libopus: packet parsing, padding and writing; packet extensions; the
  repacketizer with extensions; a decoder fed packet streams with losses
  and FEC; the multistream decoder and encoder; an encoder whose settings
  change between frames; and the channel order of both encoders. They also
  check the packet length of a constant bitrate against libopus's rule, and
  that the multistream decoder writes only the samples it returns. Three
  targets test the neural network code and need a feature of the fuzz
  package: weight blobs, DRED data, and a decoder with deep PLC, LACE,
  NoLACE, the bandwidth extension and DRED.

## x86-64

- x86-64 was run once, on an Intel Core i7-12700, on 2026-10-21: under Linux in WSL2 and natively on Windows (`scripts/x86/test.sh` and `test.ps1`). The output is correct but not bit-exact with libopus on x86. 54 tests that require libopus's bits are marked `ignore` on targets other than 64-bit ARM ([docs/guide/10](guide/10-differences-from-libopus.md#platforms)).
- A memory error in an x86 kernel was found and fixed: the AVX2 pulse search (`pvq_search_avx2`) could add a pulse past the end of its output when every score was 0 (an all-zero band or NaN input). The x86 kernels now check their slice lengths at entry.
- The AVX2 version of `haar1` computed `(a+b)*s` where libopus computes `s*a + s*b`. It now follows libopus.

## Known gaps (measured against libopus 1.6.1)

- SILK stereo with a constant bitrate at 12 kHz can lose its stereo image on
  tones, as the fixed-point build of libopus does; the float build keeps it
  ([docs/guide/10](guide/10-differences-from-libopus.md#silk)).
- aarch64 targets without NEON, such as `aarch64-unknown-none-softfloat`,
  are not supported: the build stops with a `compile_error!`
  ([docs/guide/10](guide/10-differences-from-libopus.md#platforms)).

## Fixes made after 0.1.37, before the rename

These fixes were listed in the README as "Unreleased" while the crate was still named opus-rs. The fix of issue #54 came after the release of 0.1.37 (2026-10-07) but was listed under 0.1.37 by mistake.

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
  128 kbps (35.2 / 41.1 with its tonality analysis off, which opus-rs 0.1.37
  did not port). The 48 kHz chirp packets are byte-identical.
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
