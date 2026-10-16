# Multichannel audio

This document explains how Opus carries more than two channels, and how the `opus1::multistream` module encodes and decodes such audio. It covers the multistream packet, the channel layout and its mapping table, and the channel mapping families of the Ogg Opus header (RFC 7845, Request for Comments 7845). It then covers how the encoder shares the bitrate between streams, the low-frequency effects (LFE) mode and the surround masking analysis. It ends with the application programming interface (API) and complete examples. It assumes the packet structure of [Opus packets](02-packets.md). The [documentation index](README.md) lists the other documents.

The [encoder](03-encoder.md) and [decoder](04-decoder.md) documents describe the single-stream `OpusEncoder` and `OpusDecoder` that every stream uses. The `multistream` module needs the `std` feature, which is on by default. Memory is allocated when an encoder or a decoder is built. Encoding and decoding allocate nothing.

## Why multistream exists

One Opus stream carries one or two channels: the table-of-contents (TOC) byte of a packet has a single bit for mono or stereo. Surround sound has more channels, for example six for 5.1.

Multistream Opus solves this with several ordinary Opus streams that are coded in parallel:

- Each stream is coded by its own mono or stereo encoder (`OpusEncoder`).
- Each stream is decoded by its own mono or stereo decoder (`OpusDecoder`).
- One multistream packet holds one packet of every stream for the same stretch of time.

The codec itself never sees more than two channels. The multistream layer only splits the channels into streams, shares the bitrate, and joins the stream packets.

## Terms

- **Stream**: one ordinary Opus stream inside a multistream packet.
- **Coupled stream**: a stereo stream. It carries two channels.
- **Uncoupled stream**: a mono stream. It carries one channel.
- **Coded channel**: one channel as it is coded inside a stream. The decoder documentation in the code calls the same numbers "decoded channels".
- **Mapping table**: one byte per channel of the application. It says which coded channel each channel goes to (encoder) or comes from (decoder).
- **Layout**: the channel count, the stream count, the coupled stream count and the mapping table together. `opus1::multistream::ChannelLayout` holds one.
- **Channel mapping family**: a number in the Ogg Opus header that says what the channels mean and how the layout was chosen.
- **LFE channel**: the low-frequency effects channel of a surround layout. It carries only the lowest frequencies, for a subwoofer.
- **SILK and CELT**: the two coders inside Opus. SILK is a speech coder based on linear prediction. CELT (Constrained Energy Lapped Transform) is a transform coder for any sound. A packet is SILK-only, CELT-only, or Hybrid (both at once). [What Opus is](01-what-is-opus.md) explains them.

## The multistream packet

A multistream packet of `N` streams is `N` Opus packets written one after another:

```text
+---------------------+---------------------+-----+-------------------------+
| stream 0            | stream 1            | ... | stream N-1              |
| (self-delimited)    | (self-delimited)    |     | (ordinary packet)       |
+---------------------+---------------------+-----+-------------------------+
```

The rules:

- Every stream except the last is self-delimited (appendix B of RFC 6716, the Opus specification). It stores the size of its last frame, so that a reader finds where the next stream starts. The last stream ends where the multistream packet ends. [Opus packets](02-packets.md#self-delimited-framing) shows the bytes.
- The coupled streams come first, then the uncoupled streams.
- All streams of a packet last equally long.
- Each stream has its own TOC byte. Streams may differ in mode and bandwidth.
- Every stream takes at least 1 byte, and every stream except the last at least 2 bytes. So the smallest packet has `2 * N - 1` bytes.
- Padding, when there is any, goes into the last stream.
- A multistream packet does not store the layout. Encoder and decoder must agree on it in advance, for example through the Ogg Opus header.

`opus1::packet::multistream_validate` checks a multistream packet, and `opus1::packet::multistream_pad` pads one. `opus1::multistream::split_self_delimited` cuts the first stream out of a multistream packet as an ordinary packet:

```rust
use opus1::multistream::split_self_delimited;

// Stream 0: self-delimited, code 0, a frame of 2 bytes.
// Stream 1: an ordinary code 1 packet with two frames of 1 byte.
let data = [0xF8, 0x02, 0xA0, 0xA1, 0xF9, 0xB0, 0xB1];
let mut scratch = Vec::new();
let (first, next) = split_self_delimited(&data, &mut scratch).unwrap();
// The extra size field is gone: this is an ordinary packet.
assert_eq!(first, &[0xF8, 0xA0, 0xA1]);
// Stream 1 starts at byte 4.
assert_eq!(next, 4);
```

## The layout and the mapping table

### Numbering of the coded channels

A layout has `N` streams, of which the first `M` are coupled. Decoding all streams in order gives `N + M` coded channels, numbered like this:

- Coupled stream `s` (for `s` below `M`) carries coded channels `2 * s` (left) and `2 * s + 1` (right).
- Uncoupled stream `s` (for `s` from `M` to `N - 1`) carries coded channel `s + M`.

For example, 2 coupled streams and 2 uncoupled streams give coded channels 0 to 5:

```text
stream 0 (coupled)    coded channels 0 and 1
stream 1 (coupled)    coded channels 2 and 3
stream 2 (uncoupled)  coded channel  4
stream 3 (uncoupled)  coded channel  5
```

### The mapping table

The mapping table has one entry per channel of the application. Entry `j` is the coded channel for channel `j`, or 255 (`opus1::multistream::SILENT_CHANNEL`).

In the encoder, the table routes input channels to streams:

- Stream `s` reads its input from the first input channel whose entry names its coded channel.
- An input channel with entry 255 is not coded.
- Every stream needs an input channel. The encoder rejects a layout where a stream has none.
- There cannot be more coded channels than input channels: `N + M` is at most the channel count.

In the decoder, the table routes coded channels to output channels:

- Each coded channel, once decoded, goes to every output channel whose entry names it. So one coded channel can play on several outputs.
- An output channel with entry 255 is silent.
- A coded channel that no entry names is decoded and not played.

The 5.1 layout of mapping family 1 as an example:

```text
input channel     entry   coded channel       stream
0 front left   ->   0  -> left of stream 0    stream 0 (coupled): front left + front right
1 centre       ->   4  -> stream 2            stream 2 (uncoupled): centre
2 front right  ->   1  -> right of stream 0
3 rear left    ->   2  -> left of stream 1    stream 1 (coupled): rear left + rear right
4 rear right   ->   3  -> right of stream 1
5 LFE          ->   5  -> stream 3            stream 3 (uncoupled): LFE
```

A layout must also pass these checks (`ChannelLayout::new`):

- 1 to 255 channels.
- At least 1 stream, and no more coupled streams than streams.
- `N + M` is at most 255.
- Every entry is a coded channel below `N + M`, or 255.

## What an Ogg Opus header stores

An Ogg Opus file starts with an identification header called `OpusHead` (RFC 7845 section 5.1). All numbers in it are little-endian.

| Bytes | Field | Meaning |
|---|---|---|
| 0 to 7 | Magic signature | The 8 bytes `OpusHead`. |
| 8 | Version | 1. |
| 9 | Output channel count `C` | 1 to 255. |
| 10 and 11 | Pre-skip | Samples at 48 kilohertz (kHz) that a player drops at the start: the encoder's delay. |
| 12 to 15 | Input sample rate | The rate of the original audio in hertz (Hz). For information only. |
| 16 and 17 | Output gain | A signed gain in units of 1/256 decibel (dB), applied after decoding. |
| 18 | Channel mapping family | 0, 1, 2, 3 or 255. |
| 19 | Stream count `N` | Only when the family is not 0. |
| 20 | Coupled stream count `M` | Only when the family is not 0. |
| 21 to 20 + `C` | Channel mapping | One entry per output channel. Only when the family is not 0. |

Bytes 19 onward are the channel mapping table. `ChannelMappingTable::parse(family, channels, &head[19..])` reads them. `MultistreamDecoder::new` builds a decoder from the result.

Family 0 has no table. Its layout is implied: one stream, `M = C - 1`, and the mapping `[0]` for mono or `[0, 1]` for stereo. `ChannelMappingTable::parse` returns `None` for family 0. Build that decoder with `MultistreamDecoder::with_layout(rate, C, 1, C - 1, &[0, 1][..C])`, or use a plain `OpusDecoder`.

## Channel mapping families

`MultistreamEncoder::new_surround` and `ChannelLayout::surround` pick the layout of a family. These are the layouts libopus picks (`opus_multistream_surround_encoder_init`).

### Family 0: mono and stereo

One stream with 1 or 2 channels. This is ordinary Opus.

### Family 1: surround in Vorbis channel order

1 to 8 channels, in the channel order the Vorbis codec defined. The table gives the layouts. Short names: L left, R right, C centre, FL/FR front left/right, RL/RR rear left/right, SL/SR side left/right, RC rear centre, LFE low-frequency effects.

| Channels | Name | Channel order | `N` | `M` | Mapping | Streams in order | LFE stream |
|---:|---|---|---:|---:|---|---|---|
| 1 | mono | mono | 1 | 0 | `0` | mono | none |
| 2 | stereo | L, R | 1 | 1 | `0 1` | L+R | none |
| 3 | linear surround | L, C, R | 2 | 1 | `0 2 1` | L+R, C | none |
| 4 | quadraphonic | FL, FR, RL, RR | 2 | 2 | `0 1 2 3` | FL+FR, RL+RR | none |
| 5 | 5.0 surround | FL, C, FR, RL, RR | 3 | 2 | `0 4 1 2 3` | FL+FR, RL+RR, C | none |
| 6 | 5.1 surround | FL, C, FR, RL, RR, LFE | 4 | 2 | `0 4 1 2 3 5` | FL+FR, RL+RR, C, LFE | 3 |
| 7 | 6.1 surround | FL, C, FR, SL, SR, RC, LFE | 4 | 3 | `0 4 1 2 3 5 6` | FL+FR, SL+SR, C+RC, LFE | 3 |
| 8 | 7.1 surround | FL, C, FR, SL, SR, RL, RR, LFE | 5 | 3 | `0 6 1 2 3 4 5 7` | FL+FR, SL+SR, RL+RR, C, LFE | 4 |

With 3 to 8 channels the encoder works in surround mode (`MappingType::Surround`). The section [Surround mode](#surround-mode) explains what that changes. With 1 or 2 channels family 1 behaves like family 0.

### Family 2: ambisonics

Ambisonics describes a sound field around the listener instead of loudspeaker feeds. Family 2 and family 3 are defined in RFC 8486, which extends RFC 7845.

- An ambisonics signal of order `n` has `(n + 1)²` channels: 1, 4, 9, 16, and so on up to 225. The channels are in Ambisonic Channel Number (ACN) order.
- Two more channels may follow: a stereo pair that is not part of the sound field, for example music or commentary. RFC 8486 calls them non-diegetic channels.

So the valid channel counts are the square numbers up to 225, and the square numbers plus 2 up to 227. `opus1::multistream::validate_ambisonics` checks a count.

The layout: every ambisonics channel gets an uncoupled stream, in channel order. The extra stereo pair, if present, is coupled stream 0. Every stream is coded CELT-only, and every stream gets the same share of the bitrate.

### Family 3: projection (not supported)

Family 3 is ambisonics mixed through a matrix before coding. The header stores the matrix that undoes the mix. libopus implements it as `opus_projection_*`. `opus1` does not support family 3: `new_surround` and `ChannelLayout::surround` return an error for it.

### Family 255: no defined meaning

1 to 255 channels whose meaning the application defines. The header carries any valid mapping table, and `MultistreamDecoder` decodes any valid layout. `new_surround` with family 255 picks one uncoupled stream per channel and the mapping `0, 1, 2, ...`.

### Your own layout

`MultistreamEncoder::new` takes any layout that passes the checks above. Such an encoder has no surround mode and no LFE stream.

The mapping family in the header says what the output channels mean. The stream counts and the mapping table are stored in the header for every family except 0. So a custom layout fits these families:

- Family 255, when the channels have no standard meaning.
- Family 1, when there are 1 to 8 channels in the order of the family 1 table above. The stream counts and the mapping may differ from that table.

## How the encoder shares the bitrate

### The total bitrate

`MultistreamEncoder::set_bitrate` takes a `Bitrate` (`opus1::Bitrate`, the type `OpusEncoder::set_bitrate` takes too):

| Value | Meaning |
|---|---|
| `Bitrate::Auto` (the default) | The encoder picks the total from the sampling rate, the frame size and the layout. |
| `Bitrate::Max` | As many bits as the output buffer holds. |
| `Bitrate::BitsPerSecond(b)` | `b` bits per second for all channels together. It must be positive. It is limited to 500 to 750 000 bits per second per input channel. |

Before every frame the encoder splits the total between the streams. Every stream gets at least 500 bits per second.

### Surround allocation

All layouts except ambisonics use the rule of libopus `surround_rate_allocation`. For a total bitrate `B` and `F` frames per second (counted as at least 50), the rule is:

1. Each channel of a non-LFE stream gets a fixed part for its band energies: `40 * F` bits per second.
2. The LFE stream gets a small fixed part: the smaller of `B / 20` and 3000, plus `15 * F`.
3. Each non-LFE stream gets a starting rate. It is half of what is left per non-LFE channel after steps 1 and 2, and at most 20 000 bits per second.
4. What is left after that is split in shares. An uncoupled stream gets 1 share, a coupled stream 2 shares, and the LFE stream 1/8 of a share.

The result for 5.1 at 256 000 bits per second with frames of 20 milliseconds (ms) (`F` = 50):

| Stream | Fixed part | Starting rate | Shares | Total |
|---|---:|---:|---:|---:|
| 0: FL+FR | 2 × 2000 | 20 000 | 2 × 35 560 | 95 120 |
| 1: RL+RR | 2 × 2000 | 20 000 | 2 × 35 560 | 95 120 |
| 2: C | 2000 | 20 000 | 35 560 | 57 560 |
| 3: LFE | 3750 | none | 35 560 / 8 | 8 195 |

With `Bitrate::Auto` the total is `(40 * F + sampling rate + 10 000)` per non-LFE channel, plus 8000 for the LFE stream. For 5.1 at 48 kHz with 20 ms frames the streams get 307 998 bits per second together.

An encoder built with `MultistreamEncoder::new` has no LFE stream. The same rule then treats every uncoupled stream alike. With uncoupled streams only, the split is even.

### Ambisonics allocation

With family 2 every stream gets the same share: the total divided by the number of streams (libopus `ambisonics_rate_allocation`).

### Constant bitrate and the byte budget

With constant bitrate (`set_vbr(false)`) every packet has the same size: the bitrate times the frame duration, divided by 8 and rounded to a byte. The size is at least `2 * N - 1` bytes and at most the output buffer. With `Bitrate::Max` the packet fills the output buffer. For 5.1 at 256 000 bits per second and 20 ms frames, every packet has 640 bytes.

While it writes a packet, the encoder gives each stream a byte limit:

- It keeps 1 byte for the last stream and 2 bytes for each other stream still to come.
- It keeps 1 or 2 bytes for the stream's own size field, which every stream except the last has.

With constant bitrate the last stream gets the bitrate that fills the bytes that are left. Its packet is then padded, so the whole packet reaches its exact size.

The source file `src/multistream/rate_alloc.rs` holds these rules. After a frame, each stream encoder reports the bitrate it was given, and the multistream encoder reports their sum:

```rust
use opus1::multistream::{MappingType, MultistreamEncoder};
use opus1::{Application, Bitrate};

// 5.1 at 48 kHz (family 1): two coupled streams, the centre and the LFE.
let mut encoder = MultistreamEncoder::new_surround(48_000, 6, 1, Application::Audio).unwrap();
assert_eq!((encoder.mapping_type(), encoder.lfe_stream()), (MappingType::Surround, Some(3)));
let pcm = vec![0.0f32; 960 * 6]; // 20 ms
let mut packet = vec![0u8; 4000];
let stream_rates = |encoder: &MultistreamEncoder| -> Vec<i32> {
    (0..4).map(|s| encoder.stream_encoder(s).unwrap().bitrate()).collect()
};

// 256 kb/s.
encoder.set_bitrate(Bitrate::BitsPerSecond(256_000)).unwrap();
encoder.encode(&pcm, 960, &mut packet).unwrap();
assert_eq!(stream_rates(&encoder), [95_120, 95_120, 57_560, 8_195]);
assert_eq!(encoder.bitrate(), 255_995);

// The automatic bitrate.
encoder.set_bitrate(Bitrate::Auto).unwrap();
encoder.encode(&pcm, 960, &mut packet).unwrap();
assert_eq!(encoder.bitrate(), 307_998);

// The same layout without surround mode, as `MultistreamEncoder::new`
// builds it: no LFE stream, so both uncoupled streams are treated alike.
let layout = encoder.layout().clone();
let mut plain = MultistreamEncoder::new(
    48_000,
    6,
    layout.streams(),
    layout.coupled_streams(),
    layout.mapping(),
    Application::Audio,
)
.unwrap();
plain.set_bitrate(Bitrate::BitsPerSecond(256_000)).unwrap();
plain.encode(&pcm, 960, &mut packet).unwrap();
assert_eq!(stream_rates(&plain), [78_666, 78_666, 49_333, 49_333]);
```

## Surround mode

Family 1 with 3 to 8 channels puts the encoder in surround mode. On every frame the encoder then sets up the streams like this:

- **One bandwidth for all streams.** It comes from the bitrate per channel. For frames shorter than 20 ms, `60 * (F - 50)` bits per second per channel are taken off first, because short frames spend more on overhead.

  | Bitrate per channel | Bandwidth |
  |---|---|
  | above 10 000 bits per second | fullband |
  | above 7000, up to 10 000 | super-wideband |
  | above 5000, up to 7000 | wideband |
  | 5000 or less | narrowband |

- **Coupled streams are CELT-only and stereo.** This keeps the position of each sound between the two channels. Outside surround mode a stereo encoder may code its input as mono at low bitrates.
- **The LFE stream runs in LFE mode.** See below.
- **Every stream gets an energy mask** from the surround masking analysis. See below.

In surround mode the encoder sets the bandwidth of every stream, and the mode of the coupled streams, again on every frame. So `set_bandwidth` has no effect there, and `set_force_mode` has no effect on the coupled streams.

In ambisonics mode (family 2) every stream is coded CELT-only. Outside these two modes the stream encoders choose mode and bandwidth by themselves, as a single `OpusEncoder` does.

### The LFE mode

The stream of the LFE channel runs its encoder with `OpusEncoder::set_lfe(true)` (libopus `OPUS_SET_LFE`). In this mode:

- Every packet is CELT-only and narrowband.
- CELT spends its bits on the two lowest bands, 0 to 400 Hz.
- Every band from band 2 up is held at least 80 dB below band 0, so it costs almost nothing.
- The encoder skips the transient, time-frequency and dynamic allocation analyses.

Only family 1 with 6, 7 or 8 channels has an LFE stream. It is the last stream. `MultistreamEncoder::lfe_stream` returns its index.

### The surround masking analysis

A loud sound hides quieter sounds at nearby frequencies. This is called masking. In surround sound, one channel also masks parts of the channels around it. The surround masking analysis (libopus `surround_analysis`) estimates how much, so that the encoder spends fewer bits on what the listener would not hear.

On every frame the analysis does this:

1. For every input channel except the LFE channel, it measures the energy in each of the 21 CELT bands. Each energy becomes a level in units of 6 dB, counted from a fixed reference level of that band.
2. It spreads each level to the neighbouring bands. One band up, the level falls by 6 dB. One band down, it falls by 12 dB.
3. It adds the channels into three positions: left, centre and right. A centre channel counts half on both sides.

   | Channels | Position of each input channel |
   |---:|---|
   | 3 | left, centre, right |
   | 4 | left, right, left, right |
   | 5 | left, centre, right, left, right |
   | 6 | left, centre, right, left, right, none (LFE) |
   | 7 | left, centre, right, left, right, centre, none (LFE) |
   | 8 | left, centre, right, left, right, left, right, none (LFE) |

4. The centre position takes the quieter of the left and right levels.
5. It lowers all three levels a little more when there are more channels: by `0.5 * log2((channels - 1) / 2)` units.
6. The result for a channel is its own level minus the level at its position. A negative value means the band is masked. The LFE channel gets 0 everywhere.

Each stream encoder gets the values of its channels as its energy mask (`OpusEncoder::set_energy_mask`, 21 values per channel). With it, CELT spends fewer bits on masked bands. With variable bitrate it also spends fewer bits on the whole frame. With variable bitrate the SILK part of a SILK-only or Hybrid packet also gets a lower bitrate, by at most two thirds.

## The API

All fallible calls return `Result<_, &'static str>`. Samples are interleaved: sample `i` of channel `c` is at index `i * channels + c`. A frame size is in samples per channel.

### `MultistreamEncoder`

A port of libopus `OpusMSEncoder`.

| Call | What it does |
|---|---|
| `new(rate, channels, streams, coupled_streams, mapping, application)` | An encoder for a layout you choose. No surround mode, no LFE stream. libopus `opus_multistream_encoder_create`. |
| `new_surround(rate, channels, family, application)` | An encoder with the layout of mapping family 0, 1, 2 or 255. libopus `opus_multistream_surround_encoder_create`. |
| `encode(pcm, frame_size, output)` | Encodes one frame of `f32` samples (full scale ±1.0). Returns the packet length. |
| `encode_i16(pcm, frame_size, output)` | The same for 16-bit samples. Writes the packet `encode` writes for the samples `s as f32 / 32768.0`. |
| `set_bitrate(Bitrate)`, `bitrate_setting()` | Sets and reads the total bitrate setting. |
| `bitrate()` | The sum of the stream bitrates that the last frame used. |
| `set_vbr(bool)`, `vbr()` | Variable bitrate (`true`, the default) or constant bitrate. |
| `set_vbr_constraint(bool)`, `vbr_constraint()` | With variable bitrate, keeps CELT-only packets close to the bitrate (`true`, the default). |
| `set_complexity(0..=10)`, `complexity()` | Complexity of every stream. The default is 9. |
| `set_inband_fec(bool)`, `inband_fec()` | In-band forward error correction (FEC) on every stream. Only SILK frames carry it. |
| `set_packet_loss_perc(0..=100)`, `packet_loss_perc()` | Expected packet loss in percent. |
| `set_bandwidth(Bandwidth)`, `bandwidth()` | Fixes the bandwidth of every stream. No effect in surround mode. |
| `set_max_bandwidth(Bandwidth)`, `max_bandwidth()` | Limits the automatic bandwidth choice. |
| `set_force_mode(Option<ForcedMode>)` | Asks every stream for SILK-only, Hybrid or CELT-only. In surround mode the coupled streams stay CELT-only, and in ambisonics mode all streams do. |
| `layout()`, `mapping_type()`, `lfe_stream()` | The layout, how it was chosen, and the LFE stream. |
| `application()`, `sampling_rate()` | The settings given at construction. |
| `lookahead()` | The delay the encoder adds, in samples per channel. An Ogg Opus header stores it, converted to 48 kHz, as the pre-skip. |
| `final_range()` | The final range coder states of all streams, combined with exclusive or (XOR). A decoder that read the packet correctly reports the same value. |
| `reset_state()` | Forgets the audio coded so far. The settings stay. |
| `stream_encoder(s)`, `stream_encoder_mut(s)` | The `OpusEncoder` of stream `s`, to read or change a setting of one stream. |

`encode` takes these arguments:

- `rate`, given at construction, is 8000, 12000, 16000, 24000 or 48000.
- `pcm` holds at least `frame_size * channels` samples.
- `frame_size` is 2.5, 5, 10, 20, 40 or 60 ms. At 48 kHz that is 120, 240, 480, 960, 1920 or 2880.
- `output.len()` is the largest packet allowed. The smallest buffer that can work is `2 * streams - 1` bytes.

### `MultistreamDecoder`

A port of libopus `OpusMSDecoder`.

| Call | What it does |
|---|---|
| `new(rate, ChannelMappingTable)` | A decoder for the table of an Ogg Opus header. `table.mapping` must have exactly `table.channels` entries. |
| `with_layout(rate, channels, streams, coupled_streams, mapping)` | A decoder for a layout given as numbers. libopus `opus_multistream_decoder_create`. |
| `decode(packet, frame_size, out)` | Decodes one packet into `f32` samples. Returns the samples per channel written. The output is not clipped. |
| `decode_i16(packet, frame_size, out)` | The same into 16-bit samples. Each stream is soft-clipped before it is rounded, as in libopus. |
| `decode_fec(packet, frame_size, out)`, `decode_fec_i16(...)` | Rebuilds a lost packet from the packet after it. |
| `layout()`, `sampling_rate()` | The settings given at construction. |
| `final_range()` | The final range decoder states of all streams, combined with XOR. |
| `reset_state()` | Forgets the audio decoded so far. |
| `stream_decoder(s)`, `stream_decoder_mut(s)` | The `OpusDecoder` of stream `s`. |

`decode` works like this:

- `rate` is 8000, 12000, 16000, 24000 or 48000.
- `frame_size` is the room in `out` in samples per channel. It must be at least the duration of the packet. `out` must hold `frame_size * channels` samples. A `frame_size` above 120 ms counts as 120 ms.
- The decoder checks the whole packet before it decodes any stream. A malformed packet is an error and leaves the decoder as it was.
- An empty packet (`&[]`) stands for a lost packet. Every stream then conceals `frame_size` samples, which must be a multiple of 2.5 ms.

`decode_fec` takes the packet that arrived after the lost one, and the lost duration as `frame_size`. That duration is a multiple of 2.5 ms, and at least the duration of the packet, as in libopus. Exactly `frame_size` samples per channel are written. Each stream decides by itself. A stream whose packet carries a copy of the lost frame decodes that copy. Every other stream conceals the loss. Decode the packet itself with `decode` afterwards.

### `ChannelLayout` and helpers

| Item | What it does |
|---|---|
| `ChannelLayout::new(channels, streams, coupled_streams, mapping)` | Builds a layout and checks it as a decoder does. |
| `ChannelLayout::surround(channels, family)` | The layout of a mapping family, with its `MappingType` and LFE stream. |
| `validate_for_encoder()` | The extra checks an encoder needs. |
| `channels()`, `streams()`, `coupled_streams()`, `mapping()` | The four parts of the layout. |
| `left_channel(s, prev)`, `right_channel(s, prev)`, `mono_channel(s, prev)` | The channels whose entry names a coded channel of stream `s`. Pass `None` for the first one and the previous result for the next one. |
| `MappingType` | `None`, `Surround` or `Ambisonics`. |
| `SILENT_CHANNEL` | The mapping entry 255. |
| `ChannelMappingTable` | The table of an Ogg Opus header: `channels`, `stream_count`, `coupled_count`, `mapping`. |
| `validate_ambisonics(channels)` | The stream and coupled stream counts of an ambisonics channel count, or `None`. |
| `split_self_delimited(data, scratch)` | Cuts the first stream out of a multistream packet. |

```rust
use opus1::multistream::{ChannelLayout, MappingType, validate_ambisonics};

// 7.1 in family 1: 5 streams, the first 3 coupled, LFE in stream 4.
let (layout, mapping_type, lfe) = ChannelLayout::surround(8, 1).unwrap();
assert_eq!(mapping_type, MappingType::Surround);
assert_eq!((layout.streams(), layout.coupled_streams(), lfe), (5, 3, Some(4)));
// Stream 0 reads front left (input 0) and front right (input 2).
assert_eq!(layout.left_channel(0, None), Some(0));
assert_eq!(layout.right_channel(0, None), Some(2));
// Stream 3 is the centre (input 1), stream 4 the LFE channel (input 7).
assert_eq!(layout.mono_channel(3, None), Some(1));
assert_eq!(layout.mono_channel(4, None), Some(7));

// Ambisonics: 9 channels are order 2, 11 add a stereo pair, 10 is invalid.
assert_eq!(validate_ambisonics(9), Some((9, 0)));
assert_eq!(validate_ambisonics(11), Some((10, 1)));
assert_eq!(validate_ambisonics(10), None);
// Order 1 plus a stereo pair: the pair (coupled stream 0) is fed by the
// last two channels.
let (layout, mapping_type, _) = ChannelLayout::surround(6, 2).unwrap();
assert_eq!(mapping_type, MappingType::Ambisonics);
assert_eq!(layout.mapping(), &[2, 3, 4, 5, 0, 1]);

// Family 3 (projection) is not supported.
assert!(ChannelLayout::surround(4, 3).is_err());
```

## Examples

`cargo run --release --example multistream` encodes and decodes a longer 5.1 signal. It prints the packet sizes and the signal-to-noise ratio of every channel.

### Encoding 5.1

```rust
use opus1::multistream::MultistreamEncoder;
use opus1::{Application, Bitrate, packet};

// 5.1 at 48 kHz in the channel order of mapping family 1:
// front left, centre, front right, rear left, rear right, LFE.
let mut encoder = MultistreamEncoder::new_surround(48_000, 6, 1, Application::Audio).unwrap();
encoder.set_bitrate(Bitrate::BitsPerSecond(256_000)).unwrap();

// The layout the encoder chose. An Ogg Opus header stores these numbers.
let layout = encoder.layout();
assert_eq!((layout.streams(), layout.coupled_streams()), (4, 2));
assert_eq!(layout.mapping(), &[0, 4, 1, 2, 3, 5]);
assert_eq!(encoder.lfe_stream(), Some(3));

// One frame of 20 ms: 960 samples per channel, interleaved.
let pcm: Vec<f32> = (0..960 * 6)
    .map(|k| {
        let (i, c) = (k / 6, k % 6);
        0.2 * (i as f32 * 0.01 * (c + 1) as f32).sin()
    })
    .collect();
// The length of the buffer is the largest packet allowed.
let mut packet = vec![0u8; 4000];
let len = encoder.encode(&pcm, 960, &mut packet).unwrap();

// The bitrates the allocation gave the four streams for this frame.
assert_eq!(encoder.bitrate(), 95_120 + 95_120 + 57_560 + 8_195);

// Look inside the packet: four streams, all but the last self-delimited.
let mut rest = &packet[..len];
let mut tocs = Vec::new();
for stream in 0..4 {
    let parsed = packet::parse(rest, stream < 3).unwrap();
    tocs.push(parsed.toc);
    rest = &rest[parsed.packet_offset..];
}
assert!(rest.is_empty());
// The coupled streams are CELT-only (bit 7 set) and stereo (bit 2 set).
for toc in &tocs[..2] {
    assert!(toc & 0x80 != 0 && toc & 0x04 != 0);
}
// The LFE stream is configuration 19 (CELT-only, narrowband, 20 ms), mono.
assert_eq!(tocs[3] >> 3, 19);
assert_eq!(tocs[3] & 0x04, 0);

// With constant bitrate every packet has 256 000 * 0.020 / 8 = 640 bytes.
encoder.set_vbr(false);
let len = encoder.encode(&pcm, 960, &mut packet).unwrap();
assert_eq!(len, 640);
```

### Decoding 5.1 through an Ogg Opus header

```rust
use opus1::multistream::{ChannelMappingTable, MultistreamDecoder, MultistreamEncoder};
use opus1::{Application, Bitrate};

let mut encoder = MultistreamEncoder::new_surround(48_000, 6, 1, Application::Audio).unwrap();
encoder.set_bitrate(Bitrate::BitsPerSecond(256_000)).unwrap();
let layout = encoder.layout().clone();

// The writer builds the OpusHead header (RFC 7845 section 5.1).
// The input is at 48 kHz, so the lookahead is already the pre-skip.
let pre_skip = encoder.lookahead() as u16;
assert_eq!(pre_skip, 312); // 6.5 ms
let mut head = Vec::new();
head.extend_from_slice(b"OpusHead");
head.push(1); // version
head.push(layout.channels() as u8);
head.extend_from_slice(&pre_skip.to_le_bytes());
head.extend_from_slice(&48_000u32.to_le_bytes()); // input sample rate
head.extend_from_slice(&0i16.to_le_bytes()); // output gain
head.push(1); // channel mapping family
head.push(layout.streams() as u8);
head.push(layout.coupled_streams() as u8);
head.extend_from_slice(layout.mapping());
assert_eq!(head.len(), 21 + 6);

// The player reads the channel mapping table from bytes 19 onward and
// builds its decoder from it.
let table = ChannelMappingTable::parse(head[18], head[9], &head[19..]).unwrap();
assert_eq!((table.stream_count, table.coupled_count), (4, 2));
let mut decoder = MultistreamDecoder::new(48_000, table).unwrap();

let pcm = vec![0.1f32; 960 * 6];
let mut packet = vec![0u8; 4000];
let mut out = vec![0f32; 960 * 6];

// Frame 1 arrives and is decoded. Both sides end in the same range coder
// state, which shows that the decoder read every stream completely.
let len = encoder.encode(&pcm, 960, &mut packet).unwrap();
assert_eq!(decoder.decode(&packet[..len], 960, &mut out).unwrap(), 960);
assert_eq!(decoder.final_range(), encoder.final_range());

// Frame 2 is lost. An empty packet asks the decoder to conceal 20 ms.
encoder.encode(&pcm, 960, &mut packet).unwrap();
assert_eq!(decoder.decode(&[], 960, &mut out).unwrap(), 960);

// Frame 3 is lost, and frame 4 arrives. `decode_fec` rebuilds frame 3
// from frame 4 where a stream carries FEC data, and conceals it elsewhere.
// Then frame 4 itself is decoded, here to 16-bit samples.
encoder.encode(&pcm, 960, &mut packet).unwrap();
let len = encoder.encode(&pcm, 960, &mut packet).unwrap();
assert_eq!(decoder.decode_fec(&packet[..len], 960, &mut out).unwrap(), 960);
let mut out16 = vec![0i16; 960 * 6];
assert_eq!(decoder.decode_i16(&packet[..len], 960, &mut out16).unwrap(), 960);
```

### A custom layout

```rust
use opus1::Application;
use opus1::multistream::{ChannelLayout, MultistreamDecoder, MultistreamEncoder, SILENT_CHANNEL};

// Four input channels: a stereo programme (inputs 0 and 1), a mono
// commentary (input 2), and a guide track that is not sent (input 3).
// One coupled stream (coded channels 0 and 1), one uncoupled stream
// (coded channel 2).
let mapping = [0, 1, 2, SILENT_CHANNEL];
let mut encoder =
    MultistreamEncoder::new(48_000, 4, 2, 1, &mapping, Application::Audio).unwrap();
assert_eq!(encoder.lfe_stream(), None);

let mut pcm = vec![0f32; 960 * 4];
for (i, frame) in pcm.chunks_exact_mut(4).enumerate() {
    let phase = 2.0 * std::f32::consts::PI * i as f32 / 48_000.0;
    frame[0] = 0.3 * (440.0 * phase).sin();
    frame[1] = 0.3 * (660.0 * phase).sin();
    frame[2] = 0.3 * (220.0 * phase).sin();
    frame[3] = 0.5; // not coded
}
let mut packet = vec![0u8; 1500];
let len = encoder.encode(&pcm, 960, &mut packet).unwrap();

// The receiver has 5 outputs: the programme on 0 and 1, the commentary on
// both 2 and 3, and nothing on 4. Its mapping table differs from the
// encoder's. Only the stream counts must match.
let mut decoder =
    MultistreamDecoder::with_layout(48_000, 5, 2, 1, &[0, 1, 2, 2, SILENT_CHANNEL]).unwrap();
let mut out = vec![1f32; 960 * 5];
assert_eq!(decoder.decode(&packet[..len], 960, &mut out).unwrap(), 960);
for frame in out.chunks_exact(5) {
    assert_eq!(frame[2], frame[3]); // the same coded channel twice
    assert_eq!(frame[4], 0.0); // a silent output
}

// Layouts an encoder cannot use are rejected.
// Three coded channels, but only two input channels:
assert!(MultistreamEncoder::new(48_000, 2, 2, 1, &[0, 1], Application::Audio).is_err());
// No input channel feeds the uncoupled stream (coded channel 2):
assert!(MultistreamEncoder::new(48_000, 3, 2, 1, &[0, 1, 0], Application::Audio).is_err());
// A decoder accepts that layout. Coded channel 2 is decoded and not played.
assert!(ChannelLayout::new(3, 2, 1, &[0, 1, 0]).is_ok());
```

## Interoperability and differences from libopus

Packets of the `opus1` multistream encoder decode with the libopus 1.6.1 multistream decoder, and the `opus1` multistream decoder decodes packets of the libopus encoder. The tests `tests/multistream_*.rs` compare the two libraries directly. Among other things they check that:

- libopus accepts every packet of the `opus1` encoder over a matrix of layouts, bitrates, sampling rates and frame sizes, and reaches the same final range;
- the per-stream bitrates of `rate_allocation` equal those of the libopus encoder;
- on libopus packets, the output of the `opus1` decoder matches the output of the libopus decoder to a signal-to-noise ratio of at least 100 dB.

These differences from libopus 1.6.1 are known. The figures come from the project's measurements.

- **Bandwidth with an automatic or maximum bitrate in surround mode.** libopus compares the raw setting (−1000 or −1) with the thresholds, so it always chooses narrowband. `opus1` uses the allocated bitrate and codes fullband. This is deliberate.
- **Frame sizes.** 80, 100 and 120 ms frames and the expert frame duration control are not offered.
- **Controls.** The multistream encoder has no control for these: the least significant bit (LSB) depth, the signal type, discontinuous transmission (DTX), the forced channel count, prediction, and phase inversion. The forced channel count and the prediction switch can be set on each stream through `stream_encoder_mut`.
- **Rate use without the speech/music analysis.** libopus's speech/music analysis is not ported. At 16 kHz and with `Application::Voip`, the `opus1` multistream encoder used 65 % to 80 % of the requested rate in four test cases, where libopus used 101 % to 102 %. In the test matrix of `tests/multistream_enc_validity.rs`, libopus wrote some packets with a stereo Hybrid stream: 128 at 8 kilobits per second (kb/s) per channel, and 111 at 24 kb/s per channel. `opus1` wrote none.
- **Family 3 (projection)** is not supported.
