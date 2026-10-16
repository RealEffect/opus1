# What Opus is

This document explains what the Opus audio codec is and what it can do. It describes the two coding layers, SILK and CELT (Constrained Energy Lapped Transform), and the three modes built from them. It lists the audio bandwidths, sampling rates, frame durations, bitrates and channel counts that Opus supports. It then covers delay, the tools for networks that lose packets, the Ogg Opus file format, and the history of the reference library libopus. The last section explains how this library, `opus1`, relates to libopus. The [glossary](glossary.md) defines the terms used here, and the [documentation index](README.md) lists the other documents.

## What Opus is for

An audio codec compresses audio into small packets and turns the packets back into audio. The part that compresses is the encoder. The part that decompresses is the decoder.

Opus was designed for interactive audio over the internet:

- voice and video calls, and conferences;
- games and other real-time applications;
- live music and broadcasting.

Three properties make it fit that use:

1. **Low delay.** With the default frames of 20 milliseconds (ms), the codec adds 26.5 ms of delay. It can go down to 5 ms (see [Delay](#delay)).
2. **One codec for speech and music.** Its bitrates reach from about 6 kilobits per second (kb/s), enough for narrowband speech, up to 510 kb/s for stereo music.
3. **Changes between packets.** The encoder can change the bitrate, the audio bandwidth, the frame duration and the mode from one packet to the next. The decoder follows without any information from outside the packets.

Opus is free to use. It is royalty-free. libopus is open source, under the 3-clause Berkeley Software Distribution (BSD) license.

## Who defined it

The codec working group of the Internet Engineering Task Force (IETF) defined Opus. The IETF published it in September 2012 as Request for Comments (RFC) 6716, "Definition of the Opus Audio Codec". Its authors are Jean-Marc Valin, Koen Vos and Timothy B. Terriberry.

Two rules of RFC 6716 matter for anyone who implements it:

- **Only the decoder is normative.** Any encoder is allowed, as long as its packets decode correctly. This is why libopus, and this library, can change the encoder without changing the format.
- **The reference code takes precedence.** RFC 6716 contains the source code of a reference implementation. Where the English text and that code disagree, the code is followed.

Other RFCs build on it:

| RFC | Year | Content |
|---|---|---|
| RFC 7587 | 2015 | How to carry Opus in the Real-time Transport Protocol (RTP). |
| RFC 7845 | 2016 | How to store Opus in Ogg files, with channel mappings for more than two channels. |
| RFC 7874 | 2016 | Audio requirements of Web Real-Time Communication (WebRTC), the standard for calls in web browsers. Every WebRTC endpoint must support Opus. |
| RFC 8251 | 2017 | Small corrections to the reference decoder. The format stays the same. |
| RFC 8486 | 2018 | Ambisonics (a format for three-dimensional sound) in Ogg Opus. |

## Two coding layers

Opus contains two coders, called layers. Each one is good at different sounds.

### SILK: a speech coder

SILK came from Skype, where it was the speech codec. The name is not an abbreviation. SILK is a **linear predictive coding (LPC)** coder:

- A short filter predicts each sample from the samples just before it. The filter models the resonances of the human voice.
- **Long-term prediction (LTP)** predicts voiced speech from the signal one pitch period earlier.
- What the two predictions miss is the **excitation**. SILK quantizes it and codes it together with the filter parameters.

SILK works at a sampling rate of 8, 12 or 16 kilohertz (kHz). It codes frames of 10 or 20 ms, and splits each frame into **subframes** of 5 ms. It is very efficient for speech at low bitrates. It is not designed for music. It codes nothing above 8 kHz.

### CELT: a transform coder for music and general audio

CELT came from the Xiph.Org Foundation. It is a transform coder:

- A **modified discrete cosine transform (MDCT)** turns each frame into frequency coefficients.
- The coefficients are grouped into 21 **bands**.
- The energy of each band is coded explicitly. This keeps the overall spectrum right even at low bitrates.
- The shape of each band (how the energy is spread over its coefficients) is coded with **pyramid vector quantization (PVQ)**.

CELT works at 48 kHz internally. It codes frames of 2.5, 5, 10 or 20 ms. It handles any kind of sound. It codes frequencies up to 20 kHz. For speech at low bitrates it needs more bits than SILK.

### Why there are two

Neither layer is best everywhere:

- Linear prediction fits speech well, but not music.
- A transform coder fits any sound and allows very short frames, but it costs more bits for speech at low bitrates.

Opus uses both and chooses for each packet. Both layers write into one **range coder**, so their output forms one bitstream.

## The three modes

A **mode** says which layers code a packet. The mode table uses the five audio bandwidths of the [next section](#audio-bandwidths): narrowband (NB), mediumband (MB), wideband (WB), super-wideband (SWB) and fullband (FB).

| Mode | What codes the audio | Audio bandwidths | Frame durations | Typical use |
|---|---|---|---|---|
| SILK-only | SILK | NB, MB, WB | 10, 20, 40, 60 ms | Speech at low bitrates |
| Hybrid | SILK up to 8 kHz, CELT above 8 kHz | SWB, FB | 10, 20 ms | Speech at medium bitrates, with a wide audio bandwidth |
| CELT-only | CELT | NB, WB, SWB, FB | 2.5, 5, 10, 20 ms | Music and other audio, high bitrates, and the shortest delay |

In Hybrid mode, SILK runs at 16 kHz and codes the audio up to 8 kHz. CELT codes only its bands from 8 kHz up (band 17 onwards). Both write into the same range coder, SILK first:

```text
              +--> resample to 16 kHz --> SILK (0 to 8 kHz) ----+
input audio --+                                                 +--> range coder --> packet
              +--> 48 kHz --------------> CELT (8 to 20 kHz) ---+
```

### How the encoder chooses the mode

The encoder chooses the mode again for every packet. It looks at:

- the bitrate;
- the application: `Application::Voip` (speech), `Application::Audio` (music and general audio) or `Application::RestrictedLowDelay`;
- the frame duration: frames shorter than 10 ms are always CELT-only;
- packet loss and in-band forward error correction (FEC): they favour SILK, because only SILK carries FEC;
- the mode of the previous packet, so that the choice does not flip back and forth.

A `RestrictedLowDelay` encoder always codes CELT-only. `OpusEncoder::set_force_mode` asks for a mode instead of the automatic choice.

The floating-point build of libopus also runs a speech/music detector on the input. It runs at complexity 7 and above, for input at 16 kHz or more. opus1 has no such detector. It uses the fixed estimates that libopus uses when the detector does not run: `Voip` input counts as mostly speech, and `Audio` input as mostly music.

The table below shows what opus1 chooses for 48 kHz mono input. The settings are 20 ms frames, a variable bitrate and complexity 9. The input is a steady two-tone test signal. The table gives the packets after one second.

| Bitrate asked for | `Application::Voip` | `Application::Audio` |
|---|---|---|
| 6 and 8 kb/s | SILK-only, NB | SILK-only, NB |
| 10 kb/s | SILK-only, WB | SILK-only, WB |
| 12 kb/s | SILK-only, WB | Hybrid, SWB |
| 16 kb/s | Hybrid, FB | Hybrid, FB |
| 20 to 56 kb/s | Hybrid, FB | CELT-only, FB |
| 64 to 128 kb/s | CELT-only, FB | CELT-only, FB |

With 16 kHz input the bandwidth stops at wideband. `Voip` then codes SILK-only up to 56 kb/s and CELT-only from 64 kb/s. `Audio` codes SILK-only up to 16 kb/s and CELT-only from 20 kb/s. The exact switch points move with the frame duration, the complexity, the choice of constant or variable bitrate, the packet loss and the previous mode. [The encoder](03-encoder.md) gives the rules in full.

### Switching between modes

The decoder state of one layer is of no use to the other. So when the stream changes between CELT-only and SILK-only or Hybrid, the encoder adds a **redundant frame**: an extra 5 ms of CELT audio inside a packet. The encoder leaves the frame out when the packet has too little room for it. The decoder uses the frame to cross-fade from the old mode to the new one, so the switch is smooth.

## Audio bandwidths

The audio bandwidth is the highest audio frequency that a packet codes. Opus has five:

| Bandwidth | Short name | Audio up to | Internal sampling rate | `opus1::Bandwidth` | Modes |
|---|---|---|---|---|---|
| Narrowband | NB | 4 kHz | 8 kHz | `Narrowband` | SILK-only, CELT-only |
| Mediumband | MB | 6 kHz | 12 kHz | `Mediumband` | SILK-only |
| Wideband | WB | 8 kHz | 16 kHz | `Wideband` | SILK-only, CELT-only |
| Super-wideband | SWB | 12 kHz | 24 kHz | `Superwideband` | Hybrid, CELT-only |
| Fullband | FB | 20 kHz | 48 kHz | `Fullband` | Hybrid, CELT-only |

The internal sampling rate is the sampling rate that holds the bandwidth:

- SILK really runs at that rate.
- CELT always runs at 48 kHz. It codes no bands above the bandwidth. The **end band** marks where it stops: 13 for NB, 17 for WB, 19 for SWB and 21 for FB.

CELT has no mediumband. When mediumband is asked for in CELT-only mode, opus1 codes wideband, as libopus does.

The encoder chooses the bandwidth from the bitrate. It never goes above what the input sampling rate can hold: 16 kHz input gives at most wideband. `OpusEncoder::set_bandwidth` fixes the bandwidth. `OpusEncoder::set_max_bandwidth` caps the automatic choice.

## Sampling rates

An application gives the encoder audio at 8, 12, 16, 24 or 48 kHz. It gets audio from the decoder at one of the same five rates. `OpusEncoder::new` and `OpusDecoder::new` reject any other rate. Audio at 44.1 kHz, for example, must be resampled by the application first.

The sampling rate and the audio bandwidth are independent:

- The input sampling rate only sets an upper limit on the bandwidth.
- The decoder can produce any of the five rates from any packet. A narrowband packet decoded at 48 kHz gives 48 kHz audio with nothing above 4 kHz. A fullband packet decoded at 8 kHz gives only the audio below 4 kHz.

## Frames and packets

A **frame** is a piece of coded audio with a fixed duration. A **packet** is what one call of the encoder writes and one call of the decoder reads. It holds one or more frames. All frames of a packet have the same mode, bandwidth, duration and channel count.

| Mode | Frame durations in the format | How 40 and 60 ms are coded |
|---|---|---|
| SILK-only | 10, 20, 40, 60 ms | One frame that holds 2 or 3 SILK frames of 20 ms |
| Hybrid | 10, 20 ms | A packet of 2 or 3 frames of 20 ms |
| CELT-only | 2.5, 5, 10, 20 ms | A packet of 2 or 3 frames of 20 ms |

The format has these limits:

- A packet holds at most 48 frames.
- A packet holds at most 120 ms of audio.
- A frame holds at most 1275 bytes.

Every packet starts with a **table-of-contents (TOC) byte**. It has three fields:

- The top five bits are the configuration: one of 32 combinations of mode, bandwidth and frame duration.
- The next bit says whether the packet is mono or stereo.
- The lowest two bits are the **frame count code**. It says how many frames follow and how their sizes are stored.

[Opus packets](02-packets.md) describes these bytes in full.

In opus1:

- Each call of `OpusEncoder::encode` codes 2.5, 5, 10, 20, 40 or 60 ms of audio into one packet.
- The decoder reads any valid packet, up to 120 ms.
- The `packet` module reads, writes and pads packets without decoding them.

The example encodes 20 ms of a tone and reads the TOC byte, first by hand and then with the `packet` module.

```rust
use opus1::packet;
use opus1::{Application, Bitrate, OpusEncoder};

// 20 ms of a 440 Hz tone at 48 kHz, mono: 960 samples.
let pcm: Vec<f32> = (0..960)
    .map(|i| 0.3 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 48_000.0).sin())
    .collect();

let mut encoder = OpusEncoder::new(48_000, 1, Application::Audio).unwrap();
encoder.set_bitrate(Bitrate::BitsPerSecond(64_000)).unwrap();
let mut buf = vec![0u8; 1276];
let len = encoder.encode(&pcm, 960, &mut buf).unwrap();
let pkt = &buf[..len];

// The three fields of the TOC byte.
let toc = pkt[0];
let config = toc >> 3; // 0 to 31
let stereo = toc & 0x04 != 0;
let frame_count_code = toc & 0x03;
assert_eq!(config, 31); // CELT-only, fullband, 20 ms
assert!(!stereo);
assert_eq!(frame_count_code, 0); // one frame

// The same facts from the packet module.
assert_eq!(packet::nb_frames(pkt).unwrap(), 1);
assert_eq!(packet::samples_per_frame(toc, 48_000), 960);
assert_eq!(packet::nb_samples(pkt, 48_000).unwrap(), 960);
```

## Bitrates

Opus covers bitrates from about 6 kb/s to 510 kb/s. The top comes from the frame size limit: 1275 bytes in 20 ms is 510 kb/s.

There are three ways to control the bitrate:

- **Constant bitrate (CBR)**: every packet has the same size.
- **Variable bitrate (VBR)**: the packet size follows how hard the audio is to code. The average follows the bitrate asked for.
- **Constrained VBR**: VBR that stays close enough to the target that a link at the nominal bitrate never needs more than one frame of buffering.

In opus1:

- `OpusEncoder::set_bitrate` sets the bitrate: `Bitrate::BitsPerSecond(n)` in bits per second, `Bitrate::Auto` or `Bitrate::Max`. It starts at `Bitrate::Auto`, libopus's automatic bitrate of 60 × rate / frame size + rate × channels b/s: 19,000 b/s for 16 kHz mono and 99,000 b/s for 48 kHz stereo in 20 ms frames.
- `OpusEncoder::set_vbr` chooses VBR (`true`, the default) or CBR (`false`).
- VBR is constrained for CELT-only packets, unless `set_vbr_constraint(false)` turns the constraint off. In Hybrid packets the CELT layer uses unconstrained VBR. Both match libopus.
- The length of the output buffer passed to `encode` is a limit too. When the buffer is smaller than the bitrate needs, the encoder codes for the lower bitrate.

## Channels

One Opus stream carries one or two channels:

- **Mono**: one channel.
- **Stereo**: two channels.

Both layers have their own ways to code stereo:

- SILK codes **mid/side stereo**: the average of the two channels (mid) and half their difference (side).
- CELT codes each band as mid and side, or as left and right. Above a chosen band it can use **intensity stereo**.

At low bitrates the encoder codes stereo input as mono. The measurement of [the mode choice](#how-the-encoder-chooses-the-mode) was repeated with stereo input. It gave mono packets up to 16 kb/s and stereo packets from 20 kb/s.

A decoder accepts both kinds of packet. A stereo decoder writes a mono packet to both channels. A mono decoder mixes a stereo packet down.

More channels need **multistream** Opus:

- A multistream packet holds several ordinary Opus packets, one per stream.
- Each stream is mono or stereo. A stereo stream is called a **coupled stream**.
- A table says which stream each channel goes to.
- Multistream Opus carries up to 255 channels.

opus1 implements this in the `multistream` module, which needs the `std` feature. [Multichannel audio](05-multichannel.md) explains it.

## Delay

The algorithmic delay is the delay that the codec itself adds. It does not count processing time or the network. It has two parts:

1. **The frame.** The encoder must collect a whole frame before it can code it.
2. **The lookahead.** The decoder's output lags the encoder's input by this much more.

The lookahead has two parts:

- 2.5 ms for the overlap of CELT's MDCT windows.
- 4 ms that libopus adds so that SILK and CELT line up: 2.5 ms of SILK lookahead, and 1.5 ms for SILK's resamplers and stereo prediction. `RestrictedLowDelay` leaves these 4 ms out.

| Application | Lookahead | Delay with 20 ms frames | Delay with 2.5 ms frames |
|---|---|---|---|
| `Voip`, `Audio` | 6.5 ms | 26.5 ms | 9 ms |
| `RestrictedLowDelay` | 2.5 ms | 22.5 ms | 5 ms |

`OpusEncoder::lookahead` returns the lookahead in samples per channel, at the input sampling rate:

```rust
use opus1::{Application, OpusEncoder};

let voip = OpusEncoder::new(48_000, 1, Application::Voip).unwrap();
assert_eq!(voip.lookahead(), 312); // 6.5 ms at 48 kHz

let low_delay = OpusEncoder::new(48_000, 1, Application::RestrictedLowDelay).unwrap();
assert_eq!(low_delay.lookahead(), 120); // 2.5 ms at 48 kHz

let wideband = OpusEncoder::new(16_000, 1, Application::Voip).unwrap();
assert_eq!(wideband.lookahead(), 104); // 6.5 ms at 16 kHz
```

## Tools for networks that lose packets

Real-time audio usually travels over the User Datagram Protocol (UDP). UDP can lose packets and does not send them again. Opus has three tools for this.

| Tool | What it does | In opus1 |
|---|---|---|
| **Packet loss concealment (PLC)** | The decoder makes up audio for a lost packet, from the audio before it. | Yes. Pass an empty packet to `OpusDecoder::decode`. The SILK and CELT concealment are ports of libopus. |
| **In-band forward error correction (FEC)** | A SILK frame can carry a low-bitrate copy of the frame before it, called **low bitrate redundancy (LBRR)**. When a packet is lost, the decoder rebuilds it from the copy in the next packet. | Yes. Encoder: `use_inband_fec` and `packet_loss_perc`. Decoder: `OpusDecoder::decode_fec`. |
| **Discontinuous transmission (DTX)** | During silence the encoder sends almost nothing. The decoder fills the gaps with comfort noise. | Decoder only. The decoder conceals the missing packets and the 1-byte packets of a DTX stream, as libopus does. SILK's concealment adds comfort noise. The encoder has no DTX. |

Notes on these tools:

- FEC exists only in SILK-only and Hybrid packets. In a Hybrid packet the copy covers the audio below 8 kHz.
- FEC costs bits. libopus and opus1 add it only when the packet loss and the bitrate make it worthwhile.
- With FEC on and enough packet loss, the encoder prefers SILK, even at bitrates where it would otherwise choose CELT.
- `OpusEncoder::set_prediction_disabled` makes every frame decodable on its own, at a cost in quality.
- libopus 1.5 added further tools based on machine learning: deep PLC and Deep REDundancy (DRED). opus1 has neither.

[The decoder](04-decoder.md) explains how concealment and FEC work inside. The example below loses two packets. It rebuilds the first one from the FEC data of the next packet, and it conceals the second.

```rust
use opus1::{Application, Bitrate, OpusDecoder, OpusEncoder};

// 16 kHz mono with in-band FEC: 20 ms is 320 samples.
let mut encoder = OpusEncoder::new(16_000, 1, Application::Voip).unwrap();
encoder.set_bitrate(Bitrate::BitsPerSecond(24_000)).unwrap();
encoder.set_inband_fec(true);
encoder.set_packet_loss_perc(10).unwrap();

let mut packets = Vec::new();
for f in 0..5 {
    let pcm: Vec<f32> = (0..320)
        .map(|i| 0.3 * ((f * 320 + i) as f32 * 0.05).sin())
        .collect();
    let mut buf = vec![0u8; 1276];
    let len = encoder.encode(&pcm, 320, &mut buf).unwrap();
    buf.truncate(len);
    packets.push(buf);
}

let mut decoder = OpusDecoder::new(16_000, 1).unwrap();
let mut out = vec![0.0f32; 320];

// Packet 0 arrives.
assert_eq!(decoder.decode(&packets[0], 320, &mut out).unwrap(), 320);
// Packet 1 is lost and packet 2 arrives: rebuild packet 1 from the
// FEC data in packet 2, then decode packet 2 itself.
assert_eq!(decoder.decode_fec(&packets[2], 320, &mut out).unwrap(), 320);
assert_eq!(decoder.decode(&packets[2], 320, &mut out).unwrap(), 320);
// Packet 3 is lost and nothing has arrived after it: conceal it.
assert_eq!(decoder.decode(&[], 320, &mut out).unwrap(), 320);
```

## Ogg Opus files

To store Opus packets in a file, they need a container. The usual one is Ogg, as RFC 7845 defines it for Opus. Such files use the extension `.opus`. An Ogg Opus file starts with two headers:

- The first header, `OpusHead`, holds:
  - the channel count;
  - the **pre-skip**: the number of samples a player drops at the start, counted at 48 kHz;
  - the sampling rate of the original input, for information only;
  - an output gain;
  - the **mapping family**, with its channel mapping table.
- The second header, `OpusTags`, holds text comments such as the title.

After the headers, each Ogg packet holds one Opus packet. Positions in the stream (granule positions) count samples at 48 kHz, whatever the sampling rate of the input.

opus1 does not read or write Ogg. It gives a container program what it needs:

- `OpusEncoder::lookahead` and `MultistreamEncoder::lookahead` give the pre-skip, at the input sampling rate. Scale it to 48 kHz.
- `MultistreamEncoder::layout` gives the stream counts and the mapping to write into `OpusHead`.
- `multistream::ChannelMappingTable::parse` reads them back from `OpusHead`.
- The decoder does not apply the output gain. The application must apply it to the decoded samples.

## A short history

libopus is the reference implementation, maintained by the Xiph.Org Foundation. All releases share the bitstream format of RFC 6716. libopus 1.5 and later can add optional data to a packet. This data is stored as packet extensions in the padding of the packet, and other decoders skip it. The main changes:

- **1.0 (2012).** The first release, published together with RFC 6716.
- **1.1 (2013).** Speech/music detection and tonality analysis in the encoder. An encoder for surround sound.
- **1.2 (2017).** Better quality at low bitrates, for speech and for music.
- **1.3 (2018).** Speech/music detection with a recurrent neural network. Ambisonics (mapping families 2 and 3, RFC 8486).
- **1.4 (2023).** Improved in-band FEC.
- **1.5 (2024).** Optional tools based on machine learning: deep PLC, DRED, and Opus Speech Coding Enhancement (OSCE), which improves decoded speech. Packet extensions, stored in the padding of a packet.
- **1.6.** The release that opus1 follows. 1.6.1 is its bug-fix release. Its source also has two new controls:
  - `OPUS_SET_QEXT`, for a quality extension (QEXT) for very high bitrates;
  - `OPUS_SET_OSCE_BWE`, for a bandwidth extension (BWE) of wideband speech in the decoder.

## How this library relates to libopus

opus1 is a port of libopus 1.6 to Rust. It has no C code and no C dependency. Built without its default features and with the `libm` feature, it is `#![no_std]` and needs no memory allocator.

### What opus1 implements

- `OpusEncoder` and `OpusDecoder`: all three modes, all five bandwidths, all five sampling rates, mono and stereo.
- Packet loss concealment and in-band FEC.
- 16-bit output with **soft clipping** (`OpusDecoder::decode_i16`).
- The `multistream` module: `MultistreamEncoder` and `MultistreamDecoder`, for mapping families 0, 1, 2 and 255. Their packets work with the multistream encoder and decoder of libopus in both directions.
- The `packet` module: it parses, writes and pads packets, including **self-delimited framing**.

### What follows libopus bit for bit

- **The bitstream.** opus1 reads and writes the Opus format of RFC 6716. libopus decodes the packets of opus1, and opus1 decodes the packets of libopus.
- **SILK decoding.** libopus decodes SILK with integer arithmetic in every build, so an exact port is possible. In the project's comparison over 49 SILK scenarios, 48 decode to the same samples as libopus. In the remaining one, which uses FEC, the difference lies 186 decibels (dB) below the signal. `tests/k24_silk_bitexact.rs` checks SILK-only streams sample by sample.
- **The fast Fourier transform and the MDCT on arm64.** On aarch64 the fast Fourier transform (FFT) and the MDCT give the same bits as the arm64 build of libopus.

### What is a port with small numeric differences

- **The CELT decoder.** It uses floating point, like the default build of libopus. On most CELT streams its output is not bit-exact. The difference from libopus lies 136 to 146 dB below the signal.
- **The CELT encoder.** Over 960 CELT test streams, the mean quality gap to libopus is 0.02 dB.
- **The SILK encoder.** opus1 ports the fixed-point SILK encoder of libopus. The usual floating-point build of libopus uses a floating-point SILK encoder instead. Its packets therefore differ from those of opus1. The quality is about the same. Against the fixed-point build of libopus, a voiced stream of 300 SILK packets is byte-identical.

### What is missing

The main gaps:

- the speech/music and tonality analysis of the encoder (libopus `analysis.c`);
- DTX in the encoder;
- encoder frames of 80, 100 and 120 ms;
- some controls, for example phase inversion, signal type, and a decoder output gain;
- the machine-learning tools (deep PLC, DRED, OSCE), the quality extension, and packet extensions;
- ambisonics with a demixing matrix (mapping family 3);
- on x86-64, the code compiles but has not been run.

[Differences from libopus](10-differences-from-libopus.md) lists every known difference with its measured size. [Performance](08-performance.md) compares the speed of opus1 with that of libopus. To start using the library, read [Using the library](06-using-the-library.md).
