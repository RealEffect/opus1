# Opus packets

This document describes the bytes of an Opus packet, as section 3 of RFC 6716 (Request for Comments 6716, the Opus specification) defines them. It also describes the `opus1::packet` module, which reads and writes them. It covers the table-of-contents (TOC) byte and its 32 configurations, the four ways a packet can hold its frames, and the size limits. It then covers the self-delimited form (RFC 6716 appendix B) that multistream packets use, and what the decoder does with a packet that carries no audio. [Multichannel audio](05-multichannel.md) builds on the self-delimited form. The [documentation index](README.md) lists the other documents.

## Terms

- **Packet**: what one call of the encoder writes and one call of the decoder reads. A transport carries packets one at a time, for example the Real-time Transport Protocol (RTP) or an Ogg file. An Opus packet does not store its own length. The transport tells the decoder how many bytes the packet has.
- **Frame**: a piece of coded audio with a fixed duration: 2.5, 5, 10, 20, 40 or 60 milliseconds (ms). A packet holds 1 to 48 frames. All frames of a packet have the same configuration.
- **Mode**: the coder that coded the frames. Opus has three modes:
  - **SILK-only**: SILK is a speech coder based on linear predictive coding (LPC).
  - **CELT-only**: CELT (Constrained Energy Lapped Transform) is a transform coder based on the modified discrete cosine transform (MDCT). It codes music and any other sound.
  - **Hybrid**: SILK codes the audio below 8 kilohertz (kHz) and CELT codes the audio above 8 kHz.

  [What Opus is](01-what-is-opus.md) explains the two coders.
- **Audio bandwidth**: the highest audio frequency the frames code. Opus has five bandwidths, listed in the table below.
- **Configuration**: one combination of mode, audio bandwidth and frame duration. There are 32 of them, numbered 0 to 31.

The five audio bandwidths:

| Name | Short name | Audio bandwidth | Sampling rate needed | `opus1::Bandwidth` |
|---|---|---|---|---|
| Narrowband | NB | 4 kHz | 8 kHz | `Narrowband` |
| Mediumband | MB | 6 kHz | 12 kHz | `Mediumband` |
| Wideband | WB | 8 kHz | 16 kHz | `Wideband` |
| Super-wideband | SWB | 12 kHz | 24 kHz | `Superwideband` |
| Fullband | FB | 20 kHz | 48 kHz | `Fullband` |

The bandwidth of a packet does not fix the output sampling rate. A decoder can produce 8, 12, 16, 24 or 48 kHz from any packet.

## The TOC byte

Every packet starts with one TOC byte. It has three fields:

```text
  bit:  7   6   5   4   3   2   1   0
      +---+---+---+---+---+---+---+---+
      |      config       | s |   c   |
      +---+---+---+---+---+---+---+---+
```

- `config` (bits 7 to 3) is the configuration number, 0 to 31: `toc >> 3`.
- `s` (bit 2) is 1 for stereo and 0 for mono: `toc & 0x04`.
- `c` (bits 1 and 0) is the frame count code, 0 to 3: `toc & 0x03`. It says how the frames are laid out after the TOC byte.

```rust
use opus1::SampleRate;
// Split a TOC byte into its three fields.
let toc: u8 = 0xFC; // binary 1111 1100
let config = toc >> 3;
let stereo = toc & 0x04 != 0;
let code = toc & 0x03;
assert_eq!((config, stereo, code), (31, true, 0));

// Configuration 31 is CELT-only fullband with 20 ms frames:
// 960 samples per channel at 48 kHz.
assert_eq!(opus1::packet::samples_per_frame(toc, SampleRate::Hz48000), 960);
assert_eq!(opus1::packet::bandwidth(toc), opus1::Bandwidth::Fullband);
assert_eq!(opus1::packet::nb_channels(toc), opus1::Channels::Stereo);
```

### The 32 configurations

The second column is the TOC byte of a mono packet with frame count code 0, which is `config << 3`.

| Config | TOC byte | Mode | Bandwidth | Frame duration |
|---:|---|---|---|---:|
| 0 | `0x00` | SILK-only | NB | 10 ms |
| 1 | `0x08` | SILK-only | NB | 20 ms |
| 2 | `0x10` | SILK-only | NB | 40 ms |
| 3 | `0x18` | SILK-only | NB | 60 ms |
| 4 | `0x20` | SILK-only | MB | 10 ms |
| 5 | `0x28` | SILK-only | MB | 20 ms |
| 6 | `0x30` | SILK-only | MB | 40 ms |
| 7 | `0x38` | SILK-only | MB | 60 ms |
| 8 | `0x40` | SILK-only | WB | 10 ms |
| 9 | `0x48` | SILK-only | WB | 20 ms |
| 10 | `0x50` | SILK-only | WB | 40 ms |
| 11 | `0x58` | SILK-only | WB | 60 ms |
| 12 | `0x60` | Hybrid | SWB | 10 ms |
| 13 | `0x68` | Hybrid | SWB | 20 ms |
| 14 | `0x70` | Hybrid | FB | 10 ms |
| 15 | `0x78` | Hybrid | FB | 20 ms |
| 16 | `0x80` | CELT-only | NB | 2.5 ms |
| 17 | `0x88` | CELT-only | NB | 5 ms |
| 18 | `0x90` | CELT-only | NB | 10 ms |
| 19 | `0x98` | CELT-only | NB | 20 ms |
| 20 | `0xA0` | CELT-only | WB | 2.5 ms |
| 21 | `0xA8` | CELT-only | WB | 5 ms |
| 22 | `0xB0` | CELT-only | WB | 10 ms |
| 23 | `0xB8` | CELT-only | WB | 20 ms |
| 24 | `0xC0` | CELT-only | SWB | 2.5 ms |
| 25 | `0xC8` | CELT-only | SWB | 5 ms |
| 26 | `0xD0` | CELT-only | SWB | 10 ms |
| 27 | `0xD8` | CELT-only | SWB | 20 ms |
| 28 | `0xE0` | CELT-only | FB | 2.5 ms |
| 29 | `0xE8` | CELT-only | FB | 5 ms |
| 30 | `0xF0` | CELT-only | FB | 10 ms |
| 31 | `0xF8` | CELT-only | FB | 20 ms |

The table shows these rules:

- SILK-only goes up to wideband. Hybrid is super-wideband or fullband. CELT-only has no mediumband.
- Frames of 2.5 and 5 ms are always CELT-only.
- Frames of 40 and 60 ms are always SILK-only. A CELT or Hybrid encoder makes a packet of 40 to 120 ms out of 20 ms frames: two to six frames in one packet. A SILK encoder makes 80 ms out of two 40 ms frames, 100 ms out of five 20 ms frames and 120 ms out of two 60 ms frames.

The same table as code, checked against the library:

```rust
use opus1::SampleRate;
/// Mode, bandwidth and frame duration in tenths of a millisecond
/// of a configuration number.
fn describe(config: u8) -> (&'static str, &'static str, usize) {
    let i = config as usize;
    match config {
        0..=11 => ("SILK-only", ["NB", "MB", "WB"][i / 4], [100, 200, 400, 600][i % 4]),
        12..=15 => ("Hybrid", ["SWB", "FB"][(i - 12) / 2], [100, 200][i % 2]),
        _ => ("CELT-only", ["NB", "WB", "SWB", "FB"][(i - 16) / 4], [25, 50, 100, 200][i % 4]),
    }
}

assert_eq!(describe(9), ("SILK-only", "WB", 200));
assert_eq!(describe(15), ("Hybrid", "FB", 200));
assert_eq!(describe(28), ("CELT-only", "FB", 25));

// The durations agree with the library for all 32 configurations.
for config in 0..32u8 {
    let (_, _, tenths_of_ms) = describe(config);
    let samples_at_48k = tenths_of_ms * 48_000 / 10_000;
    assert_eq!(opus1::packet::samples_per_frame(config << 3, SampleRate::Hz48000), samples_at_48k);
}
```

## The frame length field

Codes 2 and 3 store the size of some frames in the packet. A size takes one or two bytes:

- A first byte from 0 to 251 is the size itself.
- A first byte from 252 to 255 is followed by a second byte. The size is `first + 4 * second`.

The largest size two bytes can hold is 255 + 4 × 255 = 1275. Examples:

| Size in bytes | Field |
|---:|---|
| 0 | `00` |
| 251 | `FB` |
| 252 | `FC 00` |
| 300 | `FC 0C` (252 + 4 × 12) |
| 1275 | `FF FF` (255 + 4 × 255) |

A size of 0 means a frame without data. The decoder conceals such a frame (see [Packets without audio](#packets-without-audio)).

## The four frame count codes

All examples below use TOC bytes of configuration 31 (CELT-only, fullband, 20 ms, mono). Bytes are in hexadecimal. The frame contents are made-up bytes; only the layout matters here.

### Code 0: one frame

```text
+-----+------------------------------+
| TOC | frame 1 (rest of the packet) |
+-----+------------------------------+
```

The frame is everything after the TOC byte.

```text
F8 | 01 02 03
TOC  frame 1 (3 bytes)
```

### Code 1: two frames of the same size

```text
+-----+---------+---------+
| TOC | frame 1 | frame 2 |
+-----+---------+---------+
```

Each frame is half of the bytes after the TOC byte. The number of bytes after the TOC byte must therefore be even.

```text
F9 | 01 02 03 | 04 05 06
TOC  frame 1    frame 2
```

### Code 2: two frames of different sizes

```text
+-----+------------------+---------+---------+
| TOC | size of frame 1  | frame 1 | frame 2 |
+-----+------------------+---------+---------+
```

The size of frame 1 is a frame length field. Frame 2 is the rest of the packet.

```text
FA | 03              | 01 02 03 | 07
TOC  size of frame 1   frame 1    frame 2
```

### Code 3: any number of frames

After the TOC byte comes a frame count byte:

```text
  bit:  7   6   5   4   3   2   1   0
      +---+---+---+---+---+---+---+---+
      | v | p |      M (frames)       |
      +---+---+---+---+---+---+---+---+
```

- `v` is 1 for variable bitrate (VBR): every frame has its own size. It is 0 for constant bitrate (CBR): all frames have the same size.
- `p` is 1 when the packet has padding.
- `M` is the number of frames, at least 1.

The whole packet is laid out in this order:

```text
+-----+-------+----------------+------------------+--------+---------+
| TOC | count | padding length | sizes of frames  | frames | padding |
|     | byte  | (only if p=1)  | 1 to M-1 (if v=1)|        |         |
+-----+-------+----------------+------------------+--------+---------+
```

With CBR the bytes between the header and the padding are split into `M` equal frames. With VBR the packet stores the sizes of the first `M - 1` frames. The last frame is what is left.

The padding length takes one or more bytes:

- A byte of 255 adds 254 bytes of padding, and another length byte follows.
- A byte from 0 to 254 adds that many bytes of padding and ends the length.

For example, 300 bytes of padding are written as `FF 2E` (254 + 46). The padding bytes themselves are at the end of the packet. The `opus1` encoder writes zeros there. libopus 1.5 and later can also store packet extensions in them (see [Packet extensions](#packet-extensions)). Padding lets an encoder bring a packet to an exact size. A constant bitrate needs this.

Three frames of 3 bytes each, CBR (count byte `03`: `v` = 0, `p` = 0, `M` = 3):

```text
FB | 03    | 01 02 03 | 04 05 06 | 01 02 03
TOC  count   frame 1    frame 2    frame 3
```

Three frames of 3, 1 and 3 bytes, VBR (count byte `83`: `v` = 1, `M` = 3):

```text
FB | 83    | 03 01           | 01 02 03 | 07      | 04 05 06
TOC  count   sizes of 1 and 2  frame 1    frame 2   frame 3
```

One frame of 4 bytes and 3 bytes of padding (count byte `41`: `p` = 1, `M` = 1):

```text
FB | 41    | 03             | 05 05 05 05 | 00 00 00
TOC  count   padding length   frame 1       padding
```

## Limits

A packet that breaks one of these rules is invalid. The decoder rejects it with an error of kind `ErrorKind::InvalidPacket`.

- A packet has at least 1 byte, the TOC byte.
- A frame has at most 1275 bytes (`opus1::packet::MAX_FRAME_BYTES`).
- A packet lasts at most 120 ms. So it holds at most 48 frames of 2.5 ms (`opus1::packet::MAX_FRAMES`), 6 frames of 20 ms, or 2 frames of 60 ms.
- Code 1: the bytes after the TOC byte split into two equal halves.
- Code 2: the size of frame 1 fits into the packet.
- Code 3: `M` is at least 1. The padding fits into the packet. With CBR the frame bytes split into `M` equal frames.
- The size of the last frame is not stored (frame 2 of code 2, the last frame of code 3 VBR, every frame of codes 1 and 3 CBR). It must still be at most 1275 bytes.

## Self-delimited framing

An ordinary packet gets its length from the transport, because the size of its last frame is not stored. This does not work when several packets are written one after another into one buffer. A multistream packet does exactly that (see [Multichannel audio](05-multichannel.md)).

RFC 6716 appendix B defines the self-delimited form for this case. It adds one more frame length field: the size of the last frame. The field comes after all other header bytes, just before the first frame. A reader then knows where the packet ends and where the next data starts.

- Code 0: the field is the size of the only frame.
- Code 1 and code 3 CBR: the field is the size of every frame.
- Code 2 and code 3 VBR: the field is the size of the last frame.

Examples, each followed by the first byte of the next packet (`F8`):

```text
code 0:  F8 | 02                    | 09 09                   || F8 ...
         TOC  size of frame 1         frame 1

code 1:  F9 | 03                    | 01 02 03 | 04 05 06     || F8 ...
         TOC  size of each frame      frame 1    frame 2

code 2:  FA | 03      | 01          | 01 02 03 | 07           || F8 ...
         TOC  size 1    size 2        frame 1    frame 2
```

The same examples, read with `opus1::packet::parse` and `Framing::SelfDelimited`. `packet_offset()` is where the next packet starts:

```rust
use opus1::packet::{self, Framing};

// Code 0, then the first byte of the next packet.
let data = [0xF8, 0x02, 0x09, 0x09, 0xF8];
let parsed = packet::parse(&data, Framing::SelfDelimited).unwrap();
assert_eq!(parsed.frame(&data, 0), &[0x09, 0x09]);
assert_eq!(parsed.packet_offset(), 4);

// Code 1: one size for both frames.
let data = [0xF9, 0x03, 1, 2, 3, 4, 5, 6, 0xF8];
let parsed = packet::parse(&data, Framing::SelfDelimited).unwrap();
assert_eq!(parsed.frame_sizes(), [3, 3]);
assert_eq!(parsed.packet_offset(), 8);

// Code 2: the sizes of both frames.
let data = [0xFA, 0x03, 0x01, 1, 2, 3, 7, 0xF8];
let parsed = packet::parse(&data, Framing::SelfDelimited).unwrap();
assert_eq!(parsed.frame_sizes(), [3, 1]);
assert_eq!(parsed.packet_offset(), 7);

// Read as an ordinary packet, the same bytes mean something else: the
// extra size field and the next packet become frame data.
let data = [0xF8, 0x02, 0x09, 0x09, 0xF8];
let parsed = packet::parse(&data, Framing::Normal).unwrap();
assert_eq!(parsed.frame(&data, 0), &[0x02, 0x09, 0x09, 0xF8]);
```

In a multistream packet every stream except the last uses this form. The last stream is an ordinary packet: it ends where the multistream packet ends.

## Packets without audio

Some packets carry no coded audio. The decoder fills the time they cover with concealment. Packet loss concealment (PLC) means that the decoder makes up audio that continues the signal it decoded before. [The decoder](04-decoder.md) describes it.

| What the decoder gets | What it means | What `OpusDecoder::decode` does |
|---|---|---|
| 0 bytes (`&[]`) | Not a packet. The application programming interface (API) of the decoder uses it to say "this packet was lost". | Conceals `frame_size` samples per channel. `frame_size` must be a multiple of 2.5 ms. |
| 1 byte, code 0 or 1 | A valid packet whose frames have 0 bytes. | Conceals the duration the TOC byte describes: one frame for code 0, two frames for code 1. |
| 1 byte, code 2 or 3 | An invalid packet: these codes need more bytes. | Returns an error of kind `InvalidPacket`. |
| A frame of 0 or 1 byte inside a longer packet | A frame without audio. | Conceals that frame. |

Concealment continues in the mode of the audio the decoder produced last, as in libopus. The mode and bandwidth in the TOC byte of a packet without audio are not used for it. Before the first decoded packet, concealment gives silence.

A packet that is only the TOC byte is called a TOC-only packet. Encoders write one in these cases:

- The `opus1` encoder (`OpusEncoder`) writes one when the bitrate or the output buffer is too small to code anything. These are the cases, as in libopus:
  - the output buffer has fewer than 3 bytes;
  - the bitrate gives less than 3 bytes per packet;
  - the packet is longer than 20 ms, and the bitrate is below 2400 bits per second or the output buffer holds fewer than 300 bytes per second of audio.

  For a 60 ms CELT-only or Hybrid packet the encoder adds a frame count byte (code 3, three frames of 0 bytes) when the output buffer has room for it. That packet then has 2 bytes. With CBR the encoder pads the packet to its constant size. `OpusEncoder::final_range` is 0 after such a packet.

  The TOC byte names the mode of the last frame the encoder coded, and its bandwidth where that mode allows it, as in libopus. Frames shorter than 10 ms are marked CELT-only, and frames of 60 ms or more in a 1-byte buffer SILK-only. Before the first coded frame, and after `reset_state`, the encoder has libopus's initial mode, Hybrid at fullband. So the TOC byte can name Hybrid even for an encoder with `Application::RestrictedSilk`, `RestrictedCelt` or `RestrictedLowDelay`, whose packets with audio are all SILK-only or all CELT-only.
- The encoder also writes TOC-only packets during discontinuous transmission (DTX). With DTX on (`set_dtx(true)`), it writes TOC-only packets while the input is inactive, as libopus does, and the application need not send them. With CBR they are not padded.
- A multistream encoder writes a TOC-only stream when the packet leaves that stream only 1 byte (see [Multichannel audio](05-multichannel.md)).

```rust
use opus1::{Channels, OpusDecoder, SampleRate};

let mut decoder = OpusDecoder::new(SampleRate::Hz48000, Channels::Mono).unwrap();
let mut out = vec![0.0f32; 2 * 960];

// An empty packet: the packet was lost. Conceal 20 ms.
assert_eq!(decoder.decode(&[], 960, &mut out).unwrap(), 960);

// A TOC-only packet with code 0: one frame of 20 ms (configuration 31)
// without data. The decoder conceals the duration the TOC byte gives,
// even though `out` has room for 40 ms.
assert_eq!(decoder.decode(&[0xF8], 2 * 960, &mut out).unwrap(), 960);

// A TOC-only packet with code 1: two frames of 20 ms.
assert_eq!(decoder.decode(&[0xF9], 2 * 960, &mut out).unwrap(), 2 * 960);

// Code 2 needs a size byte after the TOC byte.
assert!(decoder.decode(&[0xFA], 2 * 960, &mut out).is_err());
```

## The `opus1::packet` module

The module is a port of the framing code of libopus 1.6. It reads the frames out of a packet, writes frames into a packet, joins and splits packets, and pads packets. It is available in the `#![no_std]` build. It does not allocate memory, except in the rare cases of [Packet extensions](#packet-extensions). A parsed packet is a small value that holds offsets and sizes in the caller's buffer. It does not copy the frames. Errors are `opus1::Error` values of the kinds `InvalidPacket`, `BufferTooSmall`, `BadArg` and `InternalError` (see [Errors](06-using-the-library.md#errors)).

| Function | libopus function | What it does |
|---|---|---|
| `parse(data, framing)` | `opus_packet_parse` | Finds the frames of one packet. Returns a `ParsedPacket`. |
| `write(toc, frames, out, framing, padding)` | `opus_repacketizer_out_range_impl` | Writes frames into one packet. Returns its length. |
| `Repacketizer` | `OpusRepacketizer` | Joins the frames of several packets into one packet, or writes some of them as a packet. |
| `pad(data, len, new_len)` | `opus_packet_pad` | Pads a packet in place to `new_len` bytes. |
| `pad_with_extensions(data, len, new_len, padding, extensions)` | `opus_packet_pad_impl` | Rewrites a packet in place with packet extensions added. |
| `unpad(data, len)` | `opus_packet_unpad` | Removes the padding and the extensions of a packet, in place. |
| `nb_frames(data)` | `opus_packet_get_nb_frames` | The number of frames the packet says it holds. |
| `nb_samples(data, sampling_rate)` | `opus_packet_get_nb_samples` | The duration of the packet in samples per channel. |
| `samples_per_frame(toc, sampling_rate)` | `opus_packet_get_samples_per_frame` | The duration of one frame in samples per channel. |
| `bandwidth(toc)` | `opus_packet_get_bandwidth` | The audio bandwidth of the packet, a `Bandwidth`. |
| `nb_channels(toc)` | `opus_packet_get_nb_channels` | The number of channels the packet codes, a `Channels`. |
| `has_lbrr(data)` | `opus_packet_has_lbrr` | Whether the first frame carries a low-bitrate copy (LBRR) of the frame before it, for in-band FEC. `false` for CELT-only packets. |
| `multistream_validate(data, nb_streams, sampling_rate)` | `opus_multistream_packet_validate` | Checks a multistream packet. Returns its duration. |
| `multistream_pad(data, len, new_len, nb_streams)` | `opus_multistream_packet_pad` | Pads a multistream packet in place. The padding goes into the last stream. |
| `multistream_unpad(data, len, nb_streams)` | `opus_multistream_packet_unpad` | Removes the padding and the extensions of every stream, in place. |

Two small enums replace libopus's flags `self_delimited` and `pad`:

- `Framing::Normal` is an ordinary packet. `Framing::SelfDelimited` adds the size of the last frame (see [Self-delimited framing](#self-delimited-framing)).
- `Padding::Shortest` writes no more bytes than needed. `Padding::FillBuffer` adds padding until the packet takes the whole output buffer.

### `parse`

`parse(data, framing)` checks the framing of one packet and returns a `ParsedPacket`. Its methods:

| Method | Meaning |
|---|---|
| `toc()` | The TOC byte. |
| `nb_frames()` | The number of frames, 1 to 48. |
| `frame_sizes()` | The size of each frame in bytes, one value per frame. |
| `payload_offset()` | Where the first frame starts in `data`. |
| `packet_offset()` | How many bytes the packet takes, padding included. For a self-delimited packet this is where the next data starts. |
| `padding_len()` | The number of padding bytes at the end of the packet. |

`ParsedPacket::frames(data)` iterates over the frames, and `ParsedPacket::frame(data, i)` returns frame `i`. `ParsedPacket::padding(data)` returns the padding bytes, where packet extensions are stored. Pass the same `data` that was parsed. With `Framing::SelfDelimited`, `data` may continue after the packet. Only `parse` makes a `ParsedPacket`, so its sizes and offsets always agree with each other.

The worked examples of this document, checked with `parse`:

```rust
use opus1::packet::{self, Framing};

// Code 0: one frame.
let p = [0xF8, 0x01, 0x02, 0x03];
let parsed = packet::parse(&p, Framing::Normal).unwrap();
assert_eq!(parsed.nb_frames(), 1);
assert_eq!(parsed.frame(&p, 0), &[0x01, 0x02, 0x03]);

// Code 1: two frames of the same size.
let p = [0xF9, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06];
let parsed = packet::parse(&p, Framing::Normal).unwrap();
let frames: Vec<&[u8]> = parsed.frames(&p).collect();
assert_eq!(frames, [&[0x01, 0x02, 0x03][..], &[0x04, 0x05, 0x06][..]]);

// Code 2: the size of frame 1 is stored.
let p = [0xFA, 0x03, 0x01, 0x02, 0x03, 0x07];
let parsed = packet::parse(&p, Framing::Normal).unwrap();
assert_eq!(parsed.frame_sizes(), [3, 1]);
assert_eq!(parsed.payload_offset(), 2);

// Code 3, CBR: three frames of 3 bytes.
let p = [0xFB, 0x03, 1, 2, 3, 4, 5, 6, 1, 2, 3];
let parsed = packet::parse(&p, Framing::Normal).unwrap();
assert_eq!(parsed.frame_sizes(), [3, 3, 3]);

// Code 3, VBR: the sizes of frames 1 and 2 are stored.
let p = [0xFB, 0x83, 0x03, 0x01, 1, 2, 3, 7, 4, 5, 6];
let parsed = packet::parse(&p, Framing::Normal).unwrap();
assert_eq!(parsed.frame_sizes(), [3, 1, 3]);
assert_eq!(parsed.frame(&p, 2), &[4, 5, 6]);

// Code 3 with 3 bytes of padding.
let p = [0xFB, 0x41, 0x03, 5, 5, 5, 5, 0, 0, 0];
let parsed = packet::parse(&p, Framing::Normal).unwrap();
assert_eq!((parsed.nb_frames(), parsed.frame_sizes()[0], parsed.padding_len()), (1, 4, 3));
assert_eq!(parsed.packet_offset(), p.len());

// A frame size in two bytes: FC 0C is 252 + 4 * 12 = 300.
let mut p = vec![0xFA, 0xFC, 0x0C];
p.extend_from_slice(&[0x55; 300]); // frame 1
p.push(0x07); // frame 2
let parsed = packet::parse(&p, Framing::Normal).unwrap();
assert_eq!(parsed.frame_sizes(), [300, 1]);

// Invalid: code 1 with an odd number of bytes after the TOC byte.
assert!(packet::parse(&[0xF9, 0x01, 0x02, 0x03], Framing::Normal).is_err());
// Invalid: 7 frames of 20 ms are more than 120 ms.
assert!(packet::parse(&[0xFB, 0x07], Framing::Normal).is_err());
```

### `write`

`write(toc, frames, out, framing, padding)` writes one packet made of `frames` into `out` and returns its length.

- `toc` gives the configuration and the stereo bit. Its frame count code is replaced by the code that fits the frames:
  - code 0 for one frame;
  - code 1 for two frames of the same size;
  - code 2 for two frames of different sizes;
  - code 3 for more frames.
- `Framing::SelfDelimited` adds the size of the last frame.
- `Padding::FillBuffer` fills `out` to its full length with padding. The packet is then exactly `out.len()` bytes long. A packet of one or two frames that is only one byte short is written as code 3 without padding: the frame count byte is the extra byte.

```rust
use opus1::packet::{self, Framing, Padding};

let mut out = [0u8; 64];
let a = [1u8, 2, 3];
let b = [4u8, 5, 6];
let c = [7u8];

// One frame: code 0. The code 3 in the TOC byte given here is replaced.
let n = packet::write(0xFB, &[&a], &mut out, Framing::Normal, Padding::Shortest).unwrap();
assert_eq!(&out[..n], &[0xF8, 1, 2, 3]);

// Two frames of different sizes: code 2.
let n = packet::write(0xF8, &[&a, &c], &mut out, Framing::Normal, Padding::Shortest).unwrap();
assert_eq!(&out[..n], &[0xFA, 3, 1, 2, 3, 7]);

// Three frames of different sizes: code 3, VBR.
let n = packet::write(0xF8, &[&a, &c, &b], &mut out, Framing::Normal, Padding::Shortest).unwrap();
assert_eq!(&out[..n], &[0xFB, 0x83, 3, 1, 1, 2, 3, 7, 4, 5, 6]);

// Self-delimited: one more size field, for the last frame.
let n = packet::write(0xF8, &[&a, &c], &mut out, Framing::SelfDelimited, Padding::Shortest).unwrap();
assert_eq!(&out[..n], &[0xFA, 3, 1, 1, 2, 3, 7]);

// Padded to exactly 10 bytes.
let n = packet::write(0xF8, &[&[5, 5, 5, 5]], &mut out[..10], Framing::Normal, Padding::FillBuffer).unwrap();
assert_eq!(&out[..n], &[0xFB, 0x41, 3, 5, 5, 5, 5, 0, 0, 0]);
```

### `pad`

`pad(data, len, new_len)` turns the packet `data[..len]` into a packet of `new_len` bytes, in place. `data` must have room for `new_len` bytes. The result is a code 3 packet with zero padding. When the padding of the packet holds packet extensions, they are kept (see [Packet extensions](#packet-extensions)). When only one byte is missing, the frame count byte of code 3 is that byte, and the packet has no padding. The frames stay the same.

```rust
use opus1::packet;

// A code 2 packet of 5 bytes in a buffer of 10 bytes.
let mut data = [0xFA, 0x01, 0x09, 0x08, 0x07, 0, 0, 0, 0, 0];
packet::pad(&mut data, 5, 10).unwrap();
// Code 3, VBR, 2 frames, padding flag; padding length 3; size of frame 1;
// the two frames; 3 bytes of padding.
assert_eq!(data, [0xFB, 0xC2, 0x03, 0x01, 0x09, 0x08, 0x07, 0, 0, 0]);
```

### `nb_frames`, `nb_samples` and `samples_per_frame`

These read the duration of a packet from its first two bytes. They do not check the rest of the packet; `parse` does.

```rust
use opus1::{SampleRate, packet};

// Configuration 1: SILK-only narrowband, 20 ms frames. Code 1: two frames.
let p = [0x08 | 0x01, 0xAA, 0xBB];
assert_eq!(packet::nb_frames(&p).unwrap(), 2);
assert_eq!(packet::samples_per_frame(p[0], SampleRate::Hz16000), 320);
assert_eq!(packet::nb_samples(&p, SampleRate::Hz16000).unwrap(), 640); // 40 ms at 16 kHz

// Code 3 reads the count from the second byte: 3 frames of 20 ms.
assert_eq!(packet::nb_samples(&[0xFB, 0x03], SampleRate::Hz48000).unwrap(), 2880);
```

### `multistream_validate` and `multistream_pad`

A multistream packet holds one packet per stream. Every stream except the last is self-delimited.

- `multistream_validate(data, nb_streams, sampling_rate)` checks every stream and that all streams last equally long. It returns that duration in samples per channel.
- `multistream_pad(data, len, new_len, nb_streams)` pads the last stream so that the whole packet has `new_len` bytes.

```rust
use opus1::{SampleRate, packet};

// Stream 0: self-delimited, one frame of 1 byte. Stream 1: one frame of
// 2 bytes. Both are 20 ms long.
let mut data = [0xF8, 0x01, 0x07, 0xF8, 0x08, 0x09, 0, 0, 0];
assert_eq!(packet::multistream_validate(&data[..6], 2, SampleRate::Hz48000).unwrap(), 960);

// Pad the 6-byte packet to 9 bytes. Only the last stream changes.
packet::multistream_pad(&mut data, 6, 9, 2).unwrap();
assert_eq!(data, [0xF8, 0x01, 0x07, 0xFB, 0x41, 0x01, 0x08, 0x09, 0x00]);
assert_eq!(packet::multistream_validate(&data, 2, SampleRate::Hz48000).unwrap(), 960);

// Stream 1 lasts 2.5 ms (configuration 28): the durations differ.
let bad = [0xF8, 0x01, 0x07, 0xE0, 0x08, 0x09];
assert!(packet::multistream_validate(&bad, 2, SampleRate::Hz48000).is_err());
```

`opus1::multistream::split_self_delimited` (feature `alloc`, on by default through `std`) cuts the first stream out of a multistream packet. It returns that stream as an ordinary packet, which any decoder accepts, and the offset of the next stream.

### `Repacketizer`

A `Repacketizer` collects the frames of packets that have the same TOC byte apart from the frame count code, at most 120 ms in all. `cat(packet)` adds the frames of a packet. `out(out)` writes all frames as one packet, and `out_range(begin, end, out)` frames `begin..end`. The repacketizer borrows the packets, so they must stay alive until the packet is written.

```rust
use opus1::packet::Repacketizer;

// Two packets of one 20 ms CELT frame each.
let a = [0xF8, 1, 2];
let b = [0xF8, 3, 4];
let mut rp = Repacketizer::new();
rp.cat(&a).unwrap();
rp.cat(&b).unwrap();
let mut out = [0u8; 16];
// Both frames in one packet: code 1.
let n = rp.out(&mut out).unwrap();
assert_eq!(&out[..n], &[0xF9, 1, 2, 3, 4]);
// The second frame alone.
let n = rp.out_range(1, 2, &mut out).unwrap();
assert_eq!(&out[..n], &[0xF8, 3, 4]);
```

`out_range_ext(begin, end, out, framing, padding, extensions)` has the options of libopus's internal `opus_repacketizer_out_range_impl`: self-delimited framing, padding to `out.len()`, and extensions to add.

## Packet extensions

libopus 1.5 and later can store extensions in the padding of a code 3 packet. Deep redundancy (DRED) uses them. A decoder that does not know an extension skips it. An older decoder sees only padding. The module `opus1::extensions` is a port of `src/extensions.c` of libopus 1.6.1.

The padding is a sequence of entries. Each entry starts with one byte: the extension ID in the upper 7 bits, and a flag `L` in the lowest bit.

| ID | Meaning | `L` = 0 | `L` = 1 |
|---|---|---|---|
| 0 | Padding | The rest of the padding is padding. | This byte is padding. |
| 1 | Frame separator | The next entries belong to the next frame. | The next byte says how many frames to move on; 0 does nothing. |
| 2 | Repeat | Repeat the extensions since the last separator (or the last repeat, or the start) for every later frame. The last repeated long extension of the last frame has no length. Afterwards the frame number moves on by one. | The same, but every repeated long extension has its length, and the frame number stays. |
| 3 to 31 | Short extension | No payload. | One byte of payload. |
| 32 to 127 | Long extension | The payload runs to the end of the padding. | A length follows, then the payload. |

The length of a long extension is written like the padding length of a packet, but a byte of 255 adds 255 bytes. The entries before the first separator belong to frame 0. A separator that goes past the last frame of the packet makes the padding invalid. After a repeat, only the payloads of the repeated extensions follow, frame by frame. Their ID bytes are not written again.

| Function | libopus function | What it does |
|---|---|---|
| `ExtensionIterator::new(padding, nb_frames)`, `next()`, `find_id(id)`, `set_frame_max(f)`, `reset()` | `opus_extension_iterator_*` | Reads the extensions one by one, in the order of the bytes. Repeated extensions come right after their repeat. It is an `Iterator` of `Result<Extension, Error>`, which ends after an error. |
| `count(padding, nb_frames)` | `opus_packet_extensions_count` | The number of extensions. It stops at invalid data without an error. |
| `count_ext(padding, counts, nb_frames)` | `opus_packet_extensions_count_ext` | The same, and the number of extensions of each frame. |
| `parse(padding, out, nb_frames)` | `opus_packet_extensions_parse` | Reads the extensions into `out`, in the order of the bytes. |
| `parse_ext(padding, out, counts, nb_frames)` | `opus_packet_extensions_parse_ext` | Reads them in the order of the frames. |
| `generate(out, extensions, nb_frames, padding)` | `opus_packet_extensions_generate` | Writes extensions, with a repeat where the same extensions follow in every later frame. With `Padding::FillBuffer`, bytes `01` in front fill `out`. |
| `generated_len(max_len, extensions, nb_frames)` | the same with a null buffer | The number of bytes `generate` writes. |

An `Extension` has an `id`, a `frame` and its payload `data`, which can be read. `Extension::new(id, frame, data)` makes one. Errors: `InvalidPacket` for invalid padding, `BufferTooSmall` when the output is too short, and `BadArg` for an ID outside 3 to 127, a frame number past the last frame, a short extension with more than one byte, `nb_frames` above 48, or a `counts` slice shorter than `nb_frames`. libopus checks the last two only with debug assertions, or not at all. opus1 returns the error instead of panicking.

The packet functions use extensions as libopus does:

- `parse` does not read the padding. `ParsedPacket::padding` returns it.
- `Repacketizer` carries the extensions of its packets into the packets it writes. Their frame numbers then count from the first frame of the new packet. The extensions of a packet go with its first frame: when `out_range` leaves that frame out, they are dropped. When `out_range` keeps that frame but a packet's extension is for one of its frames at or after `end`, the call fails with `BadArg`, as in libopus.
- `pad` and `multistream_pad` keep the extensions of the packet. When its padding is not valid extension data, they fail with `InternalError`, as libopus does. Zeros are valid: ID 0 with `L` = 0 is padding to the end.
- `pad_with_extensions(data, len, new_len, padding, extensions)` adds extensions before those the packet has. With `Padding::Shortest` the packet takes only the room it needs. It returns the new length of the packet. When `len` equals `new_len` and there are no extensions, it changes nothing and returns `len`. Extensions it is given are always written, or the call fails with `BufferTooSmall` when they do not fit. Here it differs from libopus's internal `opus_packet_pad_impl`, which returns 0 when `len` equals `new_len` and leaves the extensions out.
- `unpad` and `multistream_unpad` drop the padding with its extensions.
- The decoder ignores extensions, as libopus 1.6.1 does without its quality extension (QEXT).
- The encoder of libopus writes an extension only for DRED, and so does the encoder of `opus1` (feature `dred`, see [The encoder](03-encoder.md#18-deep-redundancy-dred)).

Up to 32 extensions in one call need no memory allocation. More need the `alloc` feature (on by default through `std`), which allocates them. `pad` with extensions in the old padding also copies that padding, because the new packet may overwrite it. Without `alloc`, these cases fail with `AllocFail` when there are more than 32 extensions or the old padding is longer than 1275 bytes.

```rust
use opus1::extensions::{self, Extension};
use opus1::packet::{self, Framing, Padding};

// A short extension (ID 5, 1 byte) for frame 0 and a long one (ID 40,
// 3 bytes) for frame 1 of a packet of two frames.
let exts = [
    Extension::new(5, 0, &[0xAA]),
    Extension::new(40, 1, &[1, 2, 3]),
];
let mut padding = [0u8; 16];
let n = extensions::generate(&mut padding, &exts, 2, Padding::Shortest).unwrap();
// ID 5 with L = 1 and its byte; a separator; ID 40 with L = 0, because it
// is the last entry and runs to the end; its 3 bytes.
assert_eq!(&padding[..n], &[0x0B, 0xAA, 0x02, 0x50, 1, 2, 3]);

// The same extensions in a packet of two 20 ms CELT frames.
let mut data = [0u8; 32];
let len = packet::write(0xF8, &[&[1, 2], &[3, 4]], &mut data, Framing::Normal, Padding::Shortest).unwrap();
let len = packet::pad_with_extensions(&mut data, len, 32, Padding::Shortest, &exts).unwrap();
// Code 3, CBR, padding flag, 2 frames; 7 bytes of padding; the frames;
// the extensions.
assert_eq!(&data[..len], &[0xFB, 0x42, 0x07, 1, 2, 3, 4, 0x0B, 0xAA, 0x02, 0x50, 1, 2, 3]);

// Read them back.
let parsed = packet::parse(&data[..len], Framing::Normal).unwrap();
let mut found = [Extension::default(); 4];
let n = extensions::parse(parsed.padding(&data[..len]), &mut found, parsed.nb_frames()).unwrap();
assert_eq!(&found[..n], &exts);

// Remove them: the shortest packet of the two frames is code 1.
let len = packet::unpad(&mut data, len).unwrap();
assert_eq!(&data[..len], &[0xF9, 1, 2, 3, 4]);
```
