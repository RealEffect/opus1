# Using the library

This document shows how to use `opus1` in an application. It starts with one frame through the encoder and the decoder. It then explains the default settings and how the encoder chooses its coding mode. Three recipes follow: a voice over Internet Protocol (VoIP) call at 16 kilohertz (kHz), a voice agent at 24 kHz, and music at 48 kHz. Then come the settings that matter most (frame duration, bitrate, complexity), packet loss and 16-bit samples. The last sections cover the build without the standard library, errors, threads, memory, and how to build and test the crate on a machine without cmake. Readers new to Opus can start with [What Opus is](01-what-is-opus.md). The byte layout of a packet is in [Opus packets](02-packets.md). Audio with more than two channels is in [Multichannel audio](05-multichannel.md). The [documentation index](README.md) lists the other documents.

Every `rust` example below is a complete program fragment. `cargo test --doc` compiles and runs each of them against this crate.

## Terms

- **Sample**: one number that gives the level of the sound at one moment, for one channel. `opus1` takes and returns samples as `f32` in the range -1.0 to 1.0, or as `i16`.
- **Channel**: one audio signal, such as the left side of a stereo recording. An `OpusEncoder` or `OpusDecoder` has 1 or 2 channels.
- **Interleaved**: stereo samples stored as left, right, left, right, and so on. The encoder and the decoder take and return interleaved samples.
- **Frame size**: the number of samples per channel in one call. 20 milliseconds (ms) at 48 kHz is a frame size of 960, whatever the channel count.
- **Packet**: the bytes one `encode` call writes and one `decode` call reads.
- **Bitrate**: how many bits the packets take per second, given in bits per second (b/s) or kilobits per second (kb/s).
- **Mode**: the coder inside Opus that codes a packet. There are three modes:
  - SILK-only: SILK is a speech coder based on linear prediction.
  - CELT-only: CELT (Constrained Energy Lapped Transform) is a transform coder for music and any other sound.
  - Hybrid: SILK codes the audio below 8 kHz and CELT the audio above 8 kHz.

## Adding the dependency

```toml
[dependencies]
opus1 = "0.2"
```

The crate was published as `opus-rs` up to version 0.1.37. Code that used `opus_rs::` now uses `opus1::`.

## One frame through the encoder and the decoder

```rust
use opus1::{Application, OpusDecoder, OpusEncoder};

let rate = 16_000; // samples per second
let frame = 320; // 20 ms at 16 kHz

// A 440 Hz tone as input.
let pcm: Vec<f32> = (0..frame)
    .map(|i| 0.3 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / rate as f32).sin())
    .collect();

let mut encoder = OpusEncoder::new(rate, 1, Application::Voip).unwrap();
encoder.bitrate_bps = 20_000;

// The length of `packet` is the largest packet the encoder may write.
let mut packet = vec![0u8; 1276];
let len = encoder.encode(&pcm, frame, &mut packet).unwrap();

let mut decoder = OpusDecoder::new(rate, 1).unwrap();
let mut out = vec![0.0f32; frame];
let samples = decoder.decode(&packet[..len], frame, &mut out).unwrap();
assert_eq!(samples, frame);
```

The rules of `OpusEncoder::encode(input, frame_size, output)`:

- The sampling rate is 8000, 12000, 16000, 24000 or 48000 hertz (Hz). Other rates need a resampler before the encoder.
- `frame_size` is 2.5, 5, 10, 20, 40 or 60 ms of samples per channel.
- `input` holds at least `frame_size * channels` samples. The encoder reads exactly that many.
- The length of `output` is the largest packet allowed. It also limits the bitrate. When `output` holds fewer bytes than `bitrate_bps` gives the frame, the encoder codes the frame at the lower rate. Give it 1276 bytes for each 20 ms of the frame when the bitrate alone should decide.
- The result is the number of bytes written to `output`.

The rules of `OpusDecoder::decode(packet, frame_size, output)`:

- `frame_size` is the room in `output`, in samples per channel. The packet itself says how long it is.
- The result is the number of samples per channel written. The rest of `output` is set to zero.
- The decoder may have another channel count than the packet. A mono packet fills both channels of a stereo decoder. A stereo packet is mixed down by a mono decoder.
- The output sampling rate may differ from the input rate of the encoder. Any of the five rates decodes any packet.
- Float output is not clipped, as in libopus `opus_decode_float`. Loud signals can come back with peaks beyond 1.0.

## Default settings and the automatic choice of mode

These public fields of `OpusEncoder` can be set at any time. The next `encode` call uses them.

| Field | Default | Meaning |
|---|---|---|
| `bitrate_bps` | 64000 | Target bitrate in b/s, for all channels together. Values outside 500 to 3,000,000 are clamped. |
| `complexity` | 9 | 0 to 10. Higher values spend more processor time for better quality. Values outside are clamped. |
| `use_cbr` | `false` | `true` gives every packet the same size (constant bitrate, CBR). `false` lets the size follow the signal (variable bitrate, VBR). |
| `use_inband_fec` | `false` | `true` lets SILK packets carry a low-bitrate copy of the previous frame, so a lost packet can be rebuilt (in-band forward error correction, FEC). |
| `packet_loss_perc` | 0 | The packet loss the network is expected to have, 0 to 100 percent. |

The three `Application` values give the encoder a hint about the signal. They match libopus's `OPUS_APPLICATION_*` values.

- `Application::Voip`: speech. The encoder prefers SILK and Hybrid.
- `Application::Audio`: music and other sound. The encoder prefers CELT.
- `Application::RestrictedLowDelay`: every packet is CELT-only, and the encoder leaves out the 4 ms that align CELT with SILK. Its delay (`lookahead()`) is 2.5 ms instead of 6.5 ms.

For every packet the encoder chooses the mode, the audio bandwidth and the number of coded channels. It decides from the bitrate, the application, the sampling rate, the frame duration and the FEC settings. It uses the thresholds of libopus 1.6. libopus also runs a speech/music analysis at complexity 7 and above, for 16 to 48 kHz input. `opus1` does not have that analysis. So where libopus would look at the signal, the choice of `opus1` depends on the application alone. [The encoder](03-encoder.md) explains each step of the choice in its section 1.3.

### The default bitrate

The default `bitrate_bps` is 64,000 b/s for every sampling rate and channel count. libopus starts with a different default, `OPUS_AUTO`. It is 60 × rate / frame size + rate × channels, which gives 19,000 b/s for 16 kHz mono with 20 ms frames. `opus1` has no `OPUS_AUTO`.

64 kb/s is a high rate for speech. At this rate a 16 kHz VoIP encoder codes CELT, and libopus 1.6.1 does the same at 64 kb/s. Set `bitrate_bps` to a speech rate to get SILK:

```rust
use opus1::{Application, OpusEncoder};

let frame = 320; // 20 ms at 16 kHz
let pcm: Vec<f32> = (0..frame).map(|i| 0.2 * (i as f32 * 0.07).sin()).collect();
let mut packet = vec![0u8; 1276];

// The top 5 bits of the first byte are the packet's configuration
// number: 0-11 SILK, 12-15 Hybrid, 16-31 CELT.
let config = |packet: &[u8]| packet[0] >> 3;

let mut encoder = OpusEncoder::new(16_000, 1, Application::Voip).unwrap();
assert_eq!(encoder.bitrate_bps, 64_000);
encoder.encode(&pcm, frame, &mut packet).unwrap();
assert!(config(&packet) >= 16); // CELT

let mut encoder = OpusEncoder::new(16_000, 1, Application::Voip).unwrap();
encoder.bitrate_bps = 20_000;
encoder.encode(&pcm, frame, &mut packet).unwrap();
assert!(config(&packet) <= 11); // SILK
```

The first byte of a packet is its table-of-contents (TOC) byte. The [TOC byte](02-packets.md#the-toc-byte) section of the packet document lists all 32 configurations.

### What the encoder chose in a measurement

The table shows the configurations the encoder wrote for 50 packets of a synthetic speech-like signal. All rows use 20 ms frames, VBR, complexity 9 and an output buffer of 3828 bytes. "NB", "MB", "WB", "SWB" and "FB" are narrowband (4 kHz of audio), mediumband (6 kHz), wideband (8 kHz), super-wideband (12 kHz) and fullband (20 kHz).

| Input | Application | Bitrate | FEC and loss | Packets |
|---|---|---|---|---|
| 8 kHz mono | `Voip` | 12 kb/s | off | SILK NB |
| 16 kHz mono | `Voip` | 64 kb/s (default) | off | CELT WB |
| 16 kHz mono | `Voip` | 16, 20 or 24 kb/s | off | SILK WB |
| 16 kHz mono | `Voip` | 20 kb/s | on, 10 % | SILK MB (37) and SILK NB (13) |
| 16 kHz mono | `Voip` | 24 or 28 kb/s | on, 10 % | SILK WB |
| 16 kHz mono | `Audio` | 16 kb/s | off | SILK WB |
| 16 kHz mono | `Audio` | 20 kb/s | off | CELT WB |
| 24 kHz mono | `Voip` | 16, 24 or 32 kb/s | off | Hybrid SWB |
| 24 kHz mono | `Voip` | 24 kb/s | on, 10 % | SILK WB |
| 24 kHz mono | `Voip` | 32 kb/s | on, 10 % | Hybrid SWB |
| 24 kHz mono | `Audio` | 16 kb/s | off | Hybrid SWB |
| 24 kHz mono | `Audio` | 24 or 32 kb/s | off | CELT SWB |
| 48 kHz mono | `Voip` | 32 kb/s | off | Hybrid FB |
| 48 kHz mono | `Audio` | 32 kb/s | off | CELT FB |
| 48 kHz stereo | `Audio` | 24, 64, 96 or 128 kb/s | off | CELT FB |

The rules behind the table:

- In-band FEC needs bits. When the bitrate is too low for FEC at the chosen bandwidth and the expected loss is above 5 %, the encoder lowers the bandwidth until FEC fits, as libopus does.
- At complexity 9 with VBR and 20 ms frames, `Application::Voip` keeps a mono encoder on SILK or Hybrid below about 62 kb/s. `Application::Audio` switches to CELT from about 18 kb/s.
- Once a mode runs, its threshold moves by 4 kb/s in its favour, so that the encoder does not switch back and forth.

## Recipe: a VoIP call at 16 kHz mono

Settings and the reason for each:

- 16 kHz sampling rate: it carries speech up to 8 kHz (wideband), twice the audio bandwidth of a telephone line.
- `Application::Voip`: the encoder codes speech with SILK.
- 24 kb/s: SILK wideband with room for in-band FEC at 10 % loss. Without FEC, 16-20 kb/s is enough.
- 20 ms frames: the usual packet duration for calls. It balances delay against the cost of packet headers.
- In-band FEC on, with the loss you expect: a lost packet can be rebuilt from the next one (see [Handling packet loss](#handling-packet-loss)).
- Complexity: the default, 9. SILK encoding at 16 kHz takes about 46 microseconds (µs) per 20 ms frame at complexity 10 on an Apple M4 Pro. That is a small fraction of the 20 ms the frame lasts.

```rust
use opus1::{Application, OpusDecoder, OpusEncoder};

const RATE: i32 = 16_000;
const FRAME: usize = 320; // 20 ms

let mut encoder = OpusEncoder::new(RATE, 1, Application::Voip).unwrap();
encoder.bitrate_bps = 24_000;
encoder.use_inband_fec = true;
encoder.packet_loss_perc = 10;

let mut decoder = OpusDecoder::new(RATE, 1).unwrap();

// One second of 16-bit input, as a microphone delivers it.
let mic: Vec<i16> = (0..RATE as usize)
    .map(|i| (6000.0 * (2.0 * std::f32::consts::PI * 200.0 * i as f32 / RATE as f32).sin()) as i16)
    .collect();

let mut packet = [0u8; 1276];
let mut speaker = [0i16; FRAME];
for frame in mic.chunks_exact(FRAME) {
    let len = encoder.encode_i16(frame, FRAME, &mut packet).unwrap();
    // Configurations 8-11 are SILK wideband.
    assert!((8..=11).contains(&(packet[0] >> 3)));

    // ... send packet[..len], receive it on the other side ...

    let samples = decoder.decode_i16(&packet[..len], FRAME, &mut speaker).unwrap();
    assert_eq!(samples, FRAME);
}
```

## Recipe: a voice agent at 24 kHz mono

A voice agent sends the output of a text-to-speech (TTS) engine to a listener. This recipe assumes that the engine produces 16-bit mono samples at 24 kHz.

Settings and the reason for each:

- 24 kHz sampling rate: keep the rate of the TTS output. It carries audio up to 12 kHz (super-wideband).
- `Application::Voip`: at 16, 24 and 32 kb/s the encoder codes Hybrid super-wideband. SILK codes the speech below 8 kHz, and CELT codes the band from 8 to 12 kHz. With `Application::Audio` the encoder codes CELT from about 18 kb/s.
- 16-32 kb/s: 24 kb/s is a good start. With in-band FEC at 10 % loss, use 32 kb/s to keep super-wideband. At 24 kb/s the encoder drops to SILK wideband to make room for FEC.
- 20 ms frames for a conversation. A stream that is not interactive can use 40 or 60 ms frames. Fewer packets per second means less transport overhead per second.
- Encode the 16-bit samples directly with `encode_i16`.

```rust
use opus1::{Application, OpusEncoder};

const RATE: i32 = 24_000;
const FRAME: usize = 480; // 20 ms

let mut encoder = OpusEncoder::new(RATE, 1, Application::Voip).unwrap();
encoder.bitrate_bps = 24_000;

// One utterance from the TTS engine (a stand-in signal here). It does not
// end on a frame boundary.
let tts: Vec<i16> = (0..RATE as usize + 100)
    .map(|i| (5000.0 * (2.0 * std::f32::consts::PI * 150.0 * i as f32 / RATE as f32).sin()) as i16)
    .collect();

let mut packets = Vec::new();
let mut packet = [0u8; 1276];
let mut frame = [0i16; FRAME];
for chunk in tts.chunks(FRAME) {
    // The last frame is padded with silence.
    frame[..chunk.len()].copy_from_slice(chunk);
    frame[chunk.len()..].fill(0);
    let len = encoder.encode_i16(&frame, FRAME, &mut packet).unwrap();
    // Configurations 12-13 are Hybrid super-wideband.
    assert!((12..=13).contains(&(packet[0] >> 3)));
    packets.push(packet[..len].to_vec());
}
assert_eq!(packets.len(), 51);

// An Ogg Opus file stores the encoder delay at 48 kHz as its pre-skip.
let pre_skip = encoder.lookahead() * 48_000 / RATE as usize;
assert_eq!(pre_skip, 312);
```

`lookahead()` is the delay the encoder adds, in samples per channel at the input rate. A decoder's output lags the encoder's input by this much: 6.5 ms for `Voip` and `Audio`, 2.5 ms for `RestrictedLowDelay`.

## Recipe: music at 48 kHz stereo

Settings and the reason for each:

- 48 kHz stereo: the full audio band (20 kHz) and both channels.
- `Application::Audio`: the encoder codes CELT.
- 96-128 kb/s VBR: CELT fullband stereo. Below about 17 kb/s stereo input is coded as mono, as libopus does.
- Complexity 10: CELT stereo encoding takes about 58 µs per 20 ms frame at complexity 10 on an Apple M4 Pro, against 37 µs at complexity 5.
- 20 ms frames. Use 10, 5 or 2.5 ms frames when the delay must be lower. Shorter frames need more bits for the same quality.

```rust
use opus1::{Application, OpusDecoder, OpusEncoder};

const RATE: i32 = 48_000;
const FRAME: usize = 960; // 20 ms

let mut encoder = OpusEncoder::new(RATE, 2, Application::Audio).unwrap();
encoder.bitrate_bps = 128_000;
encoder.complexity = 10;
let mut decoder = OpusDecoder::new(RATE, 2).unwrap();

// Interleaved stereo: 440 Hz on the left, 660 Hz on the right.
let pcm: Vec<f32> = (0..FRAME * 2)
    .map(|k| {
        let t = (k / 2) as f32 / RATE as f32;
        let f = if k % 2 == 0 { 440.0 } else { 660.0 };
        0.25 * (2.0 * std::f32::consts::PI * f * t).sin()
    })
    .collect();

let mut packet = vec![0u8; 1276];
let len = encoder.encode(&pcm, FRAME, &mut packet).unwrap();
assert!(packet[0] >> 3 >= 28); // configurations 28-31: CELT fullband
assert_ne!(packet[0] & 0x04, 0); // the stereo bit of the TOC byte

let mut out = vec![0.0f32; FRAME * 2];
assert_eq!(decoder.decode(&packet[..len], FRAME, &mut out).unwrap(), FRAME);
```

## Choosing frame duration, bitrate and complexity

### Frame duration

| Duration | Frame size at 48 kHz | Modes | Use it when |
|---|---|---|---|
| 2.5 ms | 120 | CELT only | the delay must be as low as possible |
| 5 ms | 240 | CELT only | low delay, for example live music over a network |
| 10 ms | 480 | all | low delay with speech coding |
| 20 ms | 960 | all | in doubt: calls, voice agents, streaming |
| 40 ms | 1920 | all | fewer packets per second matter more than delay |
| 60 ms | 2880 | all | the fewest packets per second |

SILK codes 40 and 60 ms as one Opus frame, made of two or three 20 ms SILK frames. CELT and Hybrid frames are at most 20 ms long, so the encoder puts two or three 20 ms frames into one packet. Frames shorter than 10 ms are always coded with CELT, whatever the application. Frames of 80, 100 and 120 ms, which libopus offers, are not available.

### Bitrate

The table gives starting points for mono speech and for music. Raise the bitrate when the quality is not good enough. Lower it as long as the quality stays good enough.

| Use | Input | Application | Bitrate | Reason |
|---|---|---|---|---|
| Telephone-band speech | 8 kHz | `Voip` | 12 kb/s | SILK narrowband. |
| Wideband speech | 16 kHz | `Voip` | 16-24 kb/s | SILK wideband. From about 62 kb/s the encoder switches to CELT. |
| Wideband speech with FEC | 16 kHz | `Voip` | 24-28 kb/s | At 20 kb/s and 10 % loss FEC lowers the bandwidth to MB or NB. |
| Super-wideband speech | 24 kHz | `Voip` | 16-32 kb/s | Hybrid super-wideband. |
| Fullband speech | 48 kHz | `Voip` | 32 kb/s | Hybrid fullband. |
| Music, mono | 48 kHz | `Audio` | 48-64 kb/s | CELT fullband. |
| Music, stereo | 48 kHz | `Audio` | 96-128 kb/s | CELT fullband stereo. |

With `use_cbr = true` every packet has the size the bitrate gives the frame. With VBR (the default) the sizes vary around the target. The mode decision counts a constant bitrate as worth about 8 % less than the same variable bitrate, as libopus does.

### Complexity

`complexity` runs from 0 to 10. The default is 9, as in libopus. Lower values switch off search steps:

- SILK:
  - Complexity 0-1: the simple noise-shaping quantizer.
  - From 2: the delayed-decision quantizer, which tries 2 candidate sequences (complexity 2-5), 3 (6-7) or 4 (8-10).
  - The pitch analysis and the order of the noise-shaping filter also grow with complexity.
- CELT:
  - Complexity 0: no spreading and no transient analysis.
  - From 1: transient analysis.
  - From 2: time-frequency resolution analysis.
  - From 3: the spreading is chosen from the signal.
  - From 4: coarse band energies are coded both with and without prediction, and the cheaper result is kept.
  - From 5: the pitch pre-filter.
  - From 8: a second transform for transient frames and, in stereo, a second quantization pass per band.

Median encoding time per 20 ms frame, Apple M4 Pro, from `examples/perf_harness.rs` at git commit `b63e8c3` (the encoders were forced to the named mode; record `target/perf/release_bench.md`). [Performance](08-performance.md) has the full measurement.

| Scenario | Complexity 0 | Complexity 5 | Complexity 10 |
|---|---:|---:|---:|
| SILK WB, 16 kHz mono, 20 kb/s | 14.8 µs | 35.0 µs | 45.9 µs |
| CELT FB, 48 kHz mono, 64 kb/s | 12.2 µs | 21.6 µs | 21.8 µs |
| CELT FB, 48 kHz stereo, 128 kb/s | 23.6 µs | 37.5 µs | 58.6 µs |

A 20 ms frame lasts 20,000 µs, so on that processor even complexity 10 uses well under 1 % of one core. Lower the complexity on small embedded processors, or when one machine encodes many streams.

## Handling packet loss

A packet that does not arrive in time is lost. The decoder can fill the gap in two ways.

### Concealment

Pass an empty packet. The decoder then makes up `frame_size` samples per channel that continue the signal (packet loss concealment, PLC). `frame_size` is the lost duration and must be a multiple of 2.5 ms.

```rust
use opus1::OpusDecoder;

let mut decoder = OpusDecoder::new(48_000, 2).unwrap();
let mut out = vec![0.0f32; 960 * 2];
// 20 ms were lost.
let samples = decoder.decode(&[], 960, &mut out).unwrap();
assert_eq!(samples, 960);
```

Concealment needs nothing from the encoder. It works for every mode.

### In-band forward error correction

With in-band FEC each SILK or Hybrid packet can carry a low-bitrate copy of the frame before it. SILK calls this copy LBRR (low bitrate redundancy).

On the encoder:

- Set `use_inband_fec = true`.
- Set `packet_loss_perc` to the loss you expect. With 0 % no copy is coded. The higher the value, the more bits go to the copy.
- Give the encoder enough bitrate (see the table above).
- CELT-only packets carry no copy. With FEC on, the encoder chooses SILK or Hybrid instead of CELT when the expected loss is above 0 % (`Application::Voip`) or above 5 % (`Application::Audio`), as libopus does.

On the decoder, when packet `n` is lost and packet `n + 1` has arrived:

1. Call `decode_fec(packet_n_plus_1, frame_size, out)`. It rebuilds the lost frame from the copy in packet `n + 1`. When packet `n + 1` holds no copy, it conceals the frame instead.
2. Call `decode(packet_n_plus_1, frame_size, out)` as usual.

When several packets in a row are lost, conceal all but the last one with empty packets, then use `decode_fec` for the last one.

```rust
use opus1::{Application, OpusDecoder, OpusEncoder};

const RATE: i32 = 16_000;
const FRAME: usize = 320; // 20 ms

let mut encoder = OpusEncoder::new(RATE, 1, Application::Voip).unwrap();
encoder.bitrate_bps = 24_000;
encoder.use_inband_fec = true;
encoder.packet_loss_perc = 10;

// 30 packets of a test signal.
let signal: Vec<f32> = (0..FRAME * 30)
    .map(|i| 0.3 * (2.0 * std::f32::consts::PI * 180.0 * i as f32 / RATE as f32).sin())
    .collect();
let packets: Vec<Vec<u8>> = signal
    .chunks_exact(FRAME)
    .map(|frame| {
        let mut packet = vec![0u8; 1276];
        let len = encoder.encode(frame, FRAME, &mut packet).unwrap();
        packet.truncate(len);
        packet
    })
    .collect();

let mut decoder = OpusDecoder::new(RATE, 1).unwrap();
let mut out = vec![0.0f32; FRAME];
let mut decoded = 0;
for (n, packet) in packets.iter().enumerate() {
    if n == 10 {
        continue; // packet 10 is lost
    }
    if n == 11 {
        // Rebuild packet 10 from the copy inside packet 11.
        decoded += decoder.decode_fec(packet, FRAME, &mut out).unwrap();
    }
    decoded += decoder.decode(packet, FRAME, &mut out).unwrap();
}
assert_eq!(decoded, 30 * FRAME);
```

If packet 11 had not arrived either, the decoder would call `decoder.decode(&[], FRAME, &mut out)` for packet 10 instead.

## 16-bit input and output

- `encode_i16` takes interleaved `i16` samples. It converts every sample to float as `sample as f32 / 32768.0` and then encodes as `encode` does, so it writes the same packets.
- `decode_i16` writes interleaved `i16` samples, as libopus `opus_decode` does. It decodes to float first and then applies a soft clipper, which rounds off peaks beyond full scale instead of cutting them. The public function `opus1::pcm_soft_clip` does the same for float buffers.
- `decode_fec_i16` is `decode_fec` with 16-bit output.

```rust
use opus1::{Application, OpusDecoder, OpusEncoder};

let mut encoder = OpusEncoder::new(48_000, 1, Application::Audio).unwrap();
let mut decoder = OpusDecoder::new(48_000, 1).unwrap();

let pcm: Vec<i16> = (0..960).map(|i| ((i % 100) as i16 - 50) * 200).collect();
let mut packet = [0u8; 1276];
let len = encoder.encode_i16(&pcm, 960, &mut packet).unwrap();

// The same audio as float gives the same packet.
let pcm_f32: Vec<f32> = pcm.iter().map(|&s| s as f32 / 32768.0).collect();
let mut packet_f32 = [0u8; 1276];
let mut encoder_f32 = OpusEncoder::new(48_000, 1, Application::Audio).unwrap();
let len_f32 = encoder_f32.encode(&pcm_f32, 960, &mut packet_f32).unwrap();
assert_eq!(packet[..len], packet_f32[..len_f32]);

let mut out = [0i16; 960];
assert_eq!(decoder.decode_i16(&packet[..len], 960, &mut out).unwrap(), 960);
```

## More than two channels

`OpusEncoder` and `OpusDecoder` handle 1 or 2 channels. For 3 to 255 channels, such as 5.1 surround, use `opus1::multistream::MultistreamEncoder` and `MultistreamDecoder`. A multistream packet holds one ordinary Opus packet per stream. Each stream has its own `OpusEncoder` or `OpusDecoder`. The module needs the `std` feature. [Multichannel audio](05-multichannel.md) explains the channel layouts and the settings.

```rust
use opus1::Application;
use opus1::multistream::{MultistreamDecoder, MultistreamEncoder};

// 5.1 at 48 kHz, channel mapping family 1 (Vorbis channel order).
let mut encoder = MultistreamEncoder::new_surround(48_000, 6, 1, Application::Audio).unwrap();
let layout = encoder.layout().clone();
let mut decoder = MultistreamDecoder::with_layout(
    48_000,
    6,
    layout.streams(),
    layout.coupled_streams(),
    layout.mapping(),
)
.unwrap();

let pcm = vec![0.0f32; 960 * 6]; // 20 ms, interleaved
let mut packet = vec![0u8; 4000];
let len = encoder.encode(&pcm, 960, &mut packet).unwrap();
let mut out = vec![0.0f32; 960 * 6];
assert_eq!(decoder.decode(&packet[..len], 960, &mut out).unwrap(), 960);
```

## Using the crate without the standard library

### Feature flags

| Feature | Default | What it does |
|---|---|---|
| `std` | on | Links the standard library. On x86-64 it detects the vector instructions AVX and AVX2 (Advanced Vector Extensions) and FMA (fused multiply-add) at run time. On aarch64 it detects FEAT_FCMA (the complex-number instructions of ARMv8.3) at run time where the target does not include them already. Without `std` the crate is `#![no_std]`. |
| `heap` | on | Keeps the large codec state behind `Box`, so `OpusEncoder` and `OpusDecoder` themselves are small. Requires `std`. |
| `libm` | off | Provides `sin`, `cos`, `sqrt` and the other float functions through the `libm` crate, a pure-Rust port of the musl C library's math functions. A build without `std` needs it. |

Three builds are possible:

- Default (`std` and `heap`): no dependencies at run time.
- `std` without `heap` (`default-features = false, features = ["std"]`): the codec state is stored inline, and the `multistream` module is available.
- `#![no_std]` without `alloc` (`default-features = false, features = ["libm"]`): no allocator is needed at all. One dependency, `libm`. The `multistream` module is not available. A build with neither `std` nor `libm` stops with a compile error that names the missing feature.

```toml
[dependencies.opus1]
version = "0.2"
default-features = false
features = ["libm"]
```

Without `std`, the x86-64 code uses AVX, AVX2 and FMA only when the build enables them, for example with `RUSTFLAGS="-C target-feature=+avx2,+fma"`. NEON, the vector instruction set of 64-bit ARM, is part of every aarch64 target and needs no check. `scripts/check_no_std.sh` builds the `no_std` crate for `thumbv7em-none-eabi` (Cortex-M) and `wasm32-unknown-unknown` when those targets are installed, then runs the tests without `std`.

### Size of the encoder and the decoder

Measured with `size_of` and a counting allocator, release build, Rust 1.98 on aarch64-apple-darwin. The sizes do not depend on the sampling rate, because every buffer is sized for the largest case (48 kHz stereo). With `heap`, the heap use of an encoder depends on its channel count: a mono encoder has no SILK state for the side channel.

| Build | `OpusEncoder` | `OpusDecoder` |
|---|---|---|
| Default (`heap`) | 2,544 bytes, plus 122,784 bytes on the heap in 7 allocations (mono) or 132,712 bytes in 8 allocations (stereo) | 96 bytes, plus 113,464 bytes on the heap in 5 allocations |
| Without `heap` (with or without `std`) | 135,192 bytes | 67,432 bytes |

The heap of the decoder includes a 46,088-byte float buffer for `decode_i16` and `decode_fec_i16`. Without `heap`, these two calls use a buffer of the same size on the stack instead.

Without `heap`, `OpusEncoder::new` and `OpusDecoder::new` return the whole state by value, so building one takes stack room too. One measurement ran each case on a new thread with a given stack size (release build, Rust 1.98, aarch64-apple-darwin):

| Work on the thread | Without `heap` | With `heap` |
|---|---|---|
| Build a 48 kHz stereo encoder and encode one 20 ms frame | 416 kibibytes (KiB, 1024 bytes) was enough, 400 KiB was not | 128 KiB was enough, 112 KiB was not |
| Build a 48 kHz stereo decoder, conceal one frame with `decode`, decode one packet with `decode_i16` | 240 KiB was enough, 224 KiB was not | 96 KiB was enough, 80 KiB was not |

These figures depend on the code around the codec, and debug builds need more. A second program ran each case in a function of its own. Its encoder case needed 432 KiB without `heap` and 144 KiB with it. When its `decode_i16` call concealed a frame instead of decoding a packet, the decoder case needed 272 KiB without `heap` and 128 KiB with it. The repository's `tests/stack_usage_test.rs` checks that, with `heap`, building, encoding and decoding fit in 768 KiB.

### Placing the codec in a static

Without `heap`, keep the encoder and the decoder in a `static` (or in other memory that lives as long as they are used), not in a local variable. `OpusEncoder::new` is not a `const fn`. So the static holds uninitialized memory until the program builds the encoder at run time:

```rust
use core::cell::UnsafeCell;
use core::mem::MaybeUninit;
use opus1::{Application, OpusEncoder};

/// Room for one encoder, outside every stack.
struct EncoderSlot(UnsafeCell<MaybeUninit<OpusEncoder>>);

// SAFETY: only one thread uses the slot (see `init_encoder`).
unsafe impl Sync for EncoderSlot {}

static ENCODER: EncoderSlot = EncoderSlot(UnsafeCell::new(MaybeUninit::uninit()));

/// Builds the encoder in `ENCODER` and returns it.
///
/// # Safety
/// Call it once, before any other use of `ENCODER`, from the only thread
/// that uses the encoder.
unsafe fn init_encoder() -> &'static mut OpusEncoder {
    // SAFETY: the caller guarantees that nothing else refers to the slot.
    let slot = unsafe { &mut *ENCODER.0.get() };
    slot.write(OpusEncoder::new(16_000, 1, Application::Voip).unwrap())
}

// SAFETY: this is the first and only call.
let encoder = unsafe { init_encoder() };
encoder.bitrate_bps = 16_000;
let mut packet = [0u8; 1276];
let len = encoder.encode(&[0.0f32; 320], 320, &mut packet).unwrap();
assert!(len > 0);
```

An embedded framework often has its own way to do this, such as a cell that is initialized once. Any of them works, as long as the codec is not stored on a small stack.

## Errors

Every function that can fail returns `Result<T, &'static str>`. The string says what was wrong. There is no error enum. The strings are fixed messages, so a program can log them or compare them.

What is rejected:

| Call | Input | Error |
|---|---|---|
| `OpusEncoder::new`, `OpusDecoder::new` | a sampling rate other than 8000, 12000, 16000, 24000, 48000 | `"Invalid sampling rate"` |
| `OpusEncoder::new`, `OpusDecoder::new` | a channel count other than 1 or 2 | `"Invalid number of channels"` |
| `encode`, `encode_i16` | a frame size that is not 2.5, 5, 10, 20, 40 or 60 ms | `"Invalid frame size for sampling rate"` |
| `encode`, `encode_i16` | fewer than `frame_size * channels` input samples | `"Input buffer too small for frame"` |
| `encode`, `encode_i16` | an empty output buffer | `"Output buffer too small"` |
| `decode` and the other decode calls | a packet that breaks the rules of section 3 of RFC 6716 (Request for Comments 6716, the Opus specification) | `"Invalid packet"` |
| `decode` | a packet longer than `frame_size` | `"frame_size too small for packet"` |
| `decode` and the other decode calls | an output buffer shorter than the samples to be written | `"Output buffer too small for packet"` |
| `decode` with an empty packet, `decode_fec` | a `frame_size` that is not a multiple of 2.5 ms | `"Lost frame: frame_size must be a multiple of 2.5 ms"` |
| `set_max_bandwidth` | `Bandwidth::Auto` | `"Invalid maximum bandwidth"` |
| `set_force_channels` | more channels than the encoder has, or 0 | `"Invalid forced channel count"` |
| `enable_hybrid_mode` | an encoder at 8, 12 or 16 kHz | `"Hybrid mode requires 24kHz or 48kHz sampling rate"` |

What is not rejected: `bitrate_bps`, `complexity` and `packet_loss_perc` are public fields. The next `encode` call clamps them to 500-3,000,000, 0-10 and 0-100. libopus's `OPUS_SET_*` requests clamp them in the same way.

The decoder checks the packet and the buffer sizes before it changes its state. After an invalid packet or a buffer that is too small, the next packet decodes normally.

```rust
use opus1::{Application, OpusDecoder, OpusEncoder};

// 44.1 kHz is not an Opus sampling rate.
assert_eq!(
    OpusEncoder::new(44_100, 1, Application::Audio).err(),
    Some("Invalid sampling rate")
);

let mut encoder = OpusEncoder::new(48_000, 1, Application::Audio).unwrap();
let mut packet = [0u8; 1276];
// 441 samples is not a valid frame duration at 48 kHz.
assert_eq!(
    encoder.encode(&[0.0; 441], 441, &mut packet),
    Err("Invalid frame size for sampling rate")
);

// Out-of-range settings are clamped, not rejected.
encoder.complexity = 42;
encoder.encode(&[0.0; 960], 960, &mut packet).unwrap();
assert_eq!(encoder.complexity, 10);

let mut decoder = OpusDecoder::new(48_000, 1).unwrap();
let mut out = [0.0f32; 960];
// Frame count code 3 needs a second byte.
assert_eq!(decoder.decode(&[0x03], 960, &mut out), Err("Invalid packet"));
match decoder.decode(&[], 100, &mut out) {
    Ok(_) => unreachable!(),
    Err(message) => assert!(message.contains("2.5 ms")),
}
```

## Threads

`OpusEncoder`, `OpusDecoder`, `MultistreamEncoder` and `MultistreamDecoder` are all `Send` and `Sync`. The example below checks this at compile time. In practice:

- Each encoder or decoder holds the state of one audio stream. `encode` and `decode` take `&mut self`, so one thread uses it at a time.
- An encoder or decoder can move to another thread, for example into a worker thread or an async task.
- To share one between threads, put it in a `Mutex`.
- Separate streams need separate encoders and decoders. They do not share mutable state, so they can run in parallel on different threads.

Apart from constant tables, the crate has one global value: the CELT mode, which holds the band layout, the window and the transform tables of the CELT layer. It is built once, at first use, and then only read.

```rust
use opus1::multistream::{MultistreamDecoder, MultistreamEncoder};
use opus1::{Application, OpusDecoder, OpusEncoder};

fn is_send_and_sync<T: Send + Sync>() {}
is_send_and_sync::<OpusEncoder>();
is_send_and_sync::<OpusDecoder>();
is_send_and_sync::<MultistreamEncoder>();
is_send_and_sync::<MultistreamDecoder>();

let mut encoder = OpusEncoder::new(48_000, 2, Application::Audio).unwrap();
let worker = std::thread::spawn(move || {
    let mut packet = [0u8; 1276];
    encoder.encode(&[0.0f32; 1920], 960, &mut packet).unwrap()
});
assert!(worker.join().unwrap() > 0);
```

## Memory use

- `encode`, `encode_i16`, `decode`, `decode_fec`, `decode_i16`, `decode_fec_i16` and the multistream `encode` and `decode` allocate nothing. All memory is taken when the encoder or decoder is built.
- One exception in the default build: `OpusEncoder::reset_state` builds a fresh encoder with the same settings. So it allocates the 122,784 bytes (mono) or 132,712 bytes (stereo) of a new state and frees the old one. `OpusDecoder::reset_state` allocates nothing.
- The sizes of one encoder and one decoder are in [Size of the encoder and the decoder](#size-of-the-encoder-and-the-decoder).
- A multistream encoder or decoder holds one `OpusEncoder` or `OpusDecoder` per stream. For 5.1 sound (4 streams) a `MultistreamEncoder` allocated 605,822 bytes and a `MultistreamDecoder` 500,320 bytes in the measurement above. The repository's tests `multistream_enc_no_alloc` and `multistream_dec_no_alloc` check that they allocate nothing after they are built.

## Building and testing on a machine without cmake

The library itself is pure Rust. `cargo build` needs no C compiler and no cmake.

The tests, examples and benchmarks compare `opus1` with the C reference implementation, libopus. They link libopus through two development dependencies: the `opus` crate and `audiopus_sys`. When `audiopus_sys` finds no installed libopus, it builds its own copy with cmake. To use an installed libopus instead, set two environment variables:

- `LIBOPUS_LIB_DIR`: the directory whose `lib` subdirectory holds libopus. With Homebrew on Apple silicon this is `/opt/homebrew`.
- `LIBOPUS_NO_PKG`: any value. It skips the search through `pkg-config`, which would otherwise come first.

On macOS `audiopus_sys` links libopus statically, so `lib` must hold `libopus.a`. On Linux with the GNU C library it links dynamically. On any Unix system it first asks `pkg-config`. When that finds an installed libopus, no variable is needed.

The project's measurements use libopus 1.6.1 from Homebrew. A developer can keep the variables in `.cargo/config.toml`, which `.gitignore` excludes from the repository. A local file looks like this:

```toml
[env]
# Link the test reference (C libopus) against Homebrew instead of building it with cmake.
LIBOPUS_LIB_DIR = "/opt/homebrew"
LIBOPUS_NO_PKG = "1"
# Without `heap` the codec state is inline; tests need a big stack.
RUST_MIN_STACK = "33554432"
```

Then:

```text
cargo test --release                              # default build
cargo test --release --no-default-features --features libm   # no_std build
scripts/check_no_std.sh                           # no_std builds for Cortex-M and WebAssembly
cargo run --release --example perf_harness -- bench   # timings against libopus
```

`RUST_MIN_STACK` sets the stack size of test threads. The `no_std` build needs it because its encoders and decoders are large values on the stack (see above). [Testing](09-testing.md) explains what the tests check, and [Architecture](07-architecture.md) shows how the source and the tests are organised.
