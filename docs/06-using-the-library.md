# Using the library

This document shows how to use `opus1` in an application. It starts with one frame through the encoder and the decoder. It then explains the default settings and how the encoder chooses its coding mode. Three recipes follow: a voice over Internet Protocol (VoIP) call at 16 kilohertz (kHz), a voice agent at 24 kHz, and music at 48 kHz. Then come the settings that matter most (frame duration, bitrate, complexity), packet loss, the neural network features of the decoder and 16-bit samples. The last sections cover the build without the standard library, errors, threads, memory, the build settings that make the codec faster, and how to build and test the crate on a machine without cmake. Readers new to Opus can start with [What Opus is](01-what-is-opus.md). The byte layout of a packet is in [Opus packets](02-packets.md). Audio with more than two channels is in [Multichannel audio](05-multichannel.md). The [documentation index](README.md) lists the other documents.

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

The public interface is the crate root (`OpusEncoder`, `OpusDecoder`, `Application`, `Bandwidth`, `Bitrate`, `ForcedMode` and `pcm_soft_clip`, and with a neural network feature also `DnnBlob` and `DnnModels`) and the modules `packet`, `extensions` and `multistream`. The crate also has a module `internals` for the tests of its repository. That module is hidden from the documentation, and any release may change it.

## One frame through the encoder and the decoder

```rust
use std::num::NonZeroU32;
use opus1::{Application, Bitrate, Channels, OpusDecoder, OpusEncoder, SampleRate};

let rate = SampleRate::Hz16000;
let frame = 320; // 20 ms at 16 kHz

// A 440 Hz tone as input.
let pcm: Vec<f32> = (0..frame)
    .map(|i| 0.3 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / rate.hz() as f32).sin())
    .collect();

let mut encoder = OpusEncoder::new(rate, Channels::Mono, Application::Voip).unwrap();
encoder.set_bitrate(Bitrate::BitsPerSecond(NonZeroU32::new(20_000).unwrap()));

// The length of `packet` is the largest packet the encoder may write.
let mut packet = vec![0u8; 1276];
let len = encoder.encode(&pcm, frame, &mut packet).unwrap();

let mut decoder = OpusDecoder::new(rate, Channels::Mono).unwrap();
let mut out = vec![0.0f32; frame];
let samples = decoder.decode(&packet[..len], frame, &mut out).unwrap();
assert_eq!(samples, frame);
```

The rules of `OpusEncoder::encode(input, frame_size, output)`:

- The sampling rate is 8000, 12000, 16000, 24000 or 48000 hertz (Hz). Other rates need a resampler before the encoder.
- `frame_size` is 2.5, 5, 10, 20, 40, 60, 80, 100 or 120 ms of samples per channel.
- `input` holds at least `frame_size * channels` samples. The encoder reads exactly that many.
- With an expert frame duration set, the encoder codes only the first frame of that duration (see [Frame duration](#frame-duration)).
- The length of `output` is the largest packet allowed. It also limits the bitrate. When `output` holds fewer bytes than the bitrate gives the frame, the encoder codes the frame at the lower rate. Give it 1276 bytes for each 20 ms of the frame when the bitrate alone should decide.
- The result is the number of bytes written to `output`.

The rules of `OpusDecoder::decode(packet, frame_size, output)`:

- `frame_size` is the room in `output`, in samples per channel. The packet itself says how long it is.
- The result is the number of samples per channel written. The rest of `output` is left as it was, as in libopus.
- The decoder may have another channel count than the packet. A mono packet fills both channels of a stereo decoder. A stereo packet is mixed down by a mono decoder.
- The output sampling rate may differ from the input rate of the encoder. Any of the five rates decodes any packet.
- Float output is not clipped, as in libopus `opus_decode_float`. Loud signals can come back with peaks beyond 1.0.

## Default settings and the automatic choice of mode

These settings of `OpusEncoder` can be changed at any time. The next `encode` call uses them. The type of each setting holds only values that make sense, such as a `u8` for the complexity. Each setter checks its value as the matching `OPUS_SET_*` request of libopus does. A setter that can still refuse a value returns a `Result`. After an error the setting keeps its old value.

| Setter | Getter | Default | Meaning |
|---|---|---|---|
| `set_bitrate(Bitrate)` | `bitrate_setting()`, `bitrate()` | `Bitrate::Auto` | Target bitrate for all channels together: `Bitrate::BitsPerSecond(n)` with `n` a `NonZeroU32`, `Bitrate::Auto` or `Bitrate::Max`. `n` is raised to at least 500 b/s and limited to at most 750,000 b/s per channel, so the call never fails. `bitrate()` returns the rate in b/s, as libopus `OPUS_GET_BITRATE` does. |
| `set_complexity(u8)` | `complexity()` | 9 | 0 to 10. Higher values spend more processor time for better quality. A value above 10 is refused. |
| `set_vbr(bool)` | `vbr()` | `true` | `false` gives every packet the same size (constant bitrate, CBR). `true` lets the size follow the signal (variable bitrate, VBR). |
| `set_vbr_constraint(bool)` | `vbr_constraint()` | `true` | With VBR, `true` keeps the size of CELT-only packets close to the bitrate. `false` lets it follow the signal more freely. |
| `set_inband_fec(InbandFec)` | `inband_fec()` | `InbandFec::Off` | `On` lets SILK and Hybrid packets carry a low-bitrate copy of the previous frame, so a lost packet can be rebuilt (in-band forward error correction, FEC). The three values are those of libopus `OPUS_SET_INBAND_FEC`: `Off`, `On` and `OnUnlessMusic`, which does not move the encoder to SILK for FEC when the input is clearly music. |
| `set_signal(Signal)` | `signal()` | `Signal::Auto` | What the input is: `Voice`, `Music`, or `Auto` to let the speech and music analysis decide (from complexity 7 on). See below. |
| `set_lsb_depth(u8)` | `lsb_depth()` | 24 | The precision of the input in bits, 8 to 24. Other values are refused. Samples below `2^-depth` count as silence, and the bandwidth detection ignores noise at that level. `encode_i16` uses at most 16. |
| `set_dtx(bool)` | `dtx()`, `in_dtx()` | `false` | Discontinuous transmission (DTX): while the input is inactive, the encoder sends packets of one byte. `in_dtx()` says whether it is in such a pause. See below. |
| `set_packet_loss_perc(u8)` | `packet_loss_perc()` | 0 | The packet loss the network is expected to have, 0 to 100 percent. A value above 100 is refused. |
| `set_expert_frame_duration(FrameDuration)` | `expert_frame_duration()` | `FrameDuration::Argument` | The duration of the frame each `encode` call codes. `Argument` takes the `frame_size` of the call. See [Frame duration](#frame-duration). |

The three `Application` values give the encoder a hint about the signal. They match libopus's `OPUS_APPLICATION_*` values.

- `Application::Voip`: speech. The encoder prefers SILK and Hybrid.
- `Application::Audio`: music and other sound. The encoder prefers CELT.
- `Application::RestrictedLowDelay`: every packet is CELT-only, and the encoder leaves out the 4 ms that align CELT with SILK. Its delay (`lookahead()`) is 2.5 ms instead of 6.5 ms.

For every packet the encoder chooses the mode, the audio bandwidth and the number of coded channels. It decides from the bitrate, the application, the sampling rate, the frame duration and the FEC settings. It uses the thresholds of libopus 1.6.

From complexity 7 on (the default is 9), for input at 16, 24 or 48 kHz, the encoder also runs the speech and music analysis of libopus. It estimates for every packet how likely the input is speech or music, and which bandwidth the input really uses. The choice of mode and bandwidth then follows the signal, as in libopus: music goes to CELT sooner, and a narrowband recording is coded narrower. The CELT encoder also uses the analysis for its bit allocation. Below complexity 7, and at 8 and 12 kHz, the choice depends on the application instead. `set_signal(Signal::Voice)` or `Signal::Music` tells the encoder what the input is when you know it better than the analysis. [The encoder](03-encoder.md) explains the analysis in its section 1.3 and each step of the choice in its section 1.4.

With `set_dtx(true)` the encoder sends a packet of one byte instead of a coded frame while the input is inactive (silence or steady background noise), after 200 ms of it. After each 400 ms left out, one frame is coded again. The decoder conceals the one-byte packets like lost ones and plays comfort noise after SILK frames. This saves bits in calls with pauses.

### The default bitrate

The default bitrate is `Bitrate::Auto`, as in libopus (`OPUS_AUTO`). It is 60 × rate / frame size + rate × channels bits per second, so it depends on the input and on the frame duration:

| Input | Frame | Automatic bitrate |
|---|---|---|
| 8 kHz mono | 20 ms | 11,000 b/s |
| 16 kHz mono | 20 ms | 19,000 b/s |
| 24 kHz mono | 20 ms | 27,000 b/s |
| 48 kHz mono | 20 ms | 51,000 b/s |
| 48 kHz stereo | 20 ms | 99,000 b/s |
| 48 kHz stereo | 10 ms | 102,000 b/s |

`bitrate()` returns the rate that applies, for the duration of the last frame. The automatic rate is a speech rate at 8 and 16 kHz, so a 16 kHz VoIP encoder codes SILK. A bitrate that is set explicitly changes that. At 64 kb/s the same encoder codes CELT, and libopus 1.6.1 does the same at 64 kb/s:

```rust
use std::num::NonZeroU32;
use opus1::{Application, Bitrate, Channels, OpusEncoder, SampleRate};

let frame = 320; // 20 ms at 16 kHz
let pcm: Vec<f32> = (0..frame).map(|i| 0.2 * (i as f32 * 0.07).sin()).collect();
let mut packet = vec![0u8; 1276];

// The top 5 bits of the first byte are the packet's configuration
// number: 0-11 SILK, 12-15 Hybrid, 16-31 CELT.
let config = |packet: &[u8]| packet[0] >> 3;

let mut encoder = OpusEncoder::new(SampleRate::Hz16000, Channels::Mono, Application::Voip).unwrap();
assert_eq!(encoder.bitrate_setting(), Bitrate::Auto);
encoder.encode(&pcm, frame, &mut packet).unwrap();
assert_eq!(encoder.bitrate(), 19_000);
assert!(config(&packet) <= 11); // SILK

let mut encoder = OpusEncoder::new(SampleRate::Hz16000, Channels::Mono, Application::Voip).unwrap();
encoder.set_bitrate(Bitrate::BitsPerSecond(NonZeroU32::new(64_000).unwrap()));
encoder.encode(&pcm, frame, &mut packet).unwrap();
assert!(config(&packet) >= 16); // CELT
```

The first byte of a packet is its table-of-contents (TOC) byte. The [TOC byte](02-packets.md#the-toc-byte) section of the packet document lists all 32 configurations.

### What the encoder chose in a measurement

The table shows the configurations the encoder wrote for 50 packets of a synthetic speech-like signal. All rows use 20 ms frames, VBR, complexity 9 and an output buffer of 3828 bytes. "NB", "MB", "WB", "SWB" and "FB" are narrowband (4 kHz of audio), mediumband (6 kHz), wideband (8 kHz), super-wideband (12 kHz) and fullband (20 kHz).

| Input | Application | Bitrate | FEC and loss | Packets |
|---|---|---|---|---|
| 8 kHz mono | `Voip` | 12 kb/s | off | SILK NB |
| 16 kHz mono | `Voip` | 64 kb/s | off | CELT WB |
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
use std::num::NonZeroU32;
use opus1::{Application, Bitrate, Channels, InbandFec, OpusDecoder, OpusEncoder, SampleRate};

const RATE: SampleRate = SampleRate::Hz16000;
const FRAME: usize = 320; // 20 ms

let mut encoder = OpusEncoder::new(RATE, Channels::Mono, Application::Voip).unwrap();
encoder.set_bitrate(Bitrate::BitsPerSecond(NonZeroU32::new(24_000).unwrap()));
encoder.set_inband_fec(InbandFec::On);
encoder.set_packet_loss_perc(10).unwrap();

let mut decoder = OpusDecoder::new(RATE, Channels::Mono).unwrap();

// One second of 16-bit input, as a microphone delivers it.
let mic: Vec<i16> = (0..RATE.hz() as usize)
    .map(|i| (6000.0 * (2.0 * std::f32::consts::PI * 200.0 * i as f32 / RATE.hz() as f32).sin()) as i16)
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
- `Signal::Voice`: the input is speech. From complexity 7 on (the default is 9) the encoder otherwise estimates this from the input with its speech and music analysis, and may code a voice it takes for music as CELT.

```rust
use std::num::NonZeroU32;
use opus1::{Application, Bitrate, Channels, OpusEncoder, SampleRate, Signal};

const RATE: SampleRate = SampleRate::Hz24000;
const FRAME: usize = 480; // 20 ms

let mut encoder = OpusEncoder::new(RATE, Channels::Mono, Application::Voip).unwrap();
encoder.set_bitrate(Bitrate::BitsPerSecond(NonZeroU32::new(24_000).unwrap()));
encoder.set_signal(Signal::Voice);

// One utterance from the TTS engine (a stand-in signal here: a tone and
// some noise, which reaches up to 12 kHz as speech does). It does not end
// on a frame boundary.
let mut seed = 1u32;
let tts: Vec<i16> = (0..RATE.hz() as usize + 100)
    .map(|i| {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let noise = (seed >> 16) as f32 / 32768.0 - 1.0;
        let tone = (2.0 * std::f32::consts::PI * 150.0 * i as f32 / RATE.hz() as f32).sin();
        (5000.0 * tone + 500.0 * noise) as i16
    })
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
let pre_skip = encoder.lookahead() * 48_000 / RATE.hz() as usize;
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
use std::num::NonZeroU32;
use opus1::{Application, Bitrate, Channels, OpusDecoder, OpusEncoder, SampleRate};

const RATE: SampleRate = SampleRate::Hz48000;
const FRAME: usize = 960; // 20 ms

let mut encoder = OpusEncoder::new(RATE, Channels::Stereo, Application::Audio).unwrap();
encoder.set_bitrate(Bitrate::BitsPerSecond(NonZeroU32::new(128_000).unwrap()));
encoder.set_complexity(10).unwrap();
let mut decoder = OpusDecoder::new(RATE, Channels::Stereo).unwrap();

// Interleaved stereo: 440 Hz on the left, 660 Hz on the right.
let pcm: Vec<f32> = (0..FRAME * 2)
    .map(|k| {
        let t = (k / 2) as f32 / RATE.hz() as f32;
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
| 60 ms | 2880 | all | few packets per second |
| 80 ms | 3840 | all | fewer packets still, when a delay of more than 80 ms does not matter |
| 100 ms | 4800 | all | the same |
| 120 ms | 5760 | all | the fewest packets per second, for example for storage or a link that sends few packets |

A packet holds at most 120 ms. The encoder splits a long frame into parts, as libopus does:

- SILK codes 40 and 60 ms as one Opus frame, made of two or three 20 ms SILK frames.
- SILK frames are at most 60 ms long. 80 ms is two frames of 40 ms, 100 ms is five frames of 20 ms, and 120 ms is two frames of 60 ms, in one packet.
- CELT and Hybrid frames are at most 20 ms long. 40 to 120 ms is two to six frames of 20 ms in one packet.

The encoder decides the mode, the bandwidth and the channels once for the whole packet, and every part is coded with them. Frames shorter than 10 ms are always coded with CELT, whatever the application.

With VBR, libopus 1.6 limits each part of such a packet to the bytes the bitrate gives one part. `opus1` leaves that limit out, so a part can take more bytes when the signal needs them, as with frames of their own. With in-band FEC the limit costs libopus much quality. One measurement: 2.4 s of speech at 8 kHz mono, `Audio`, 64 kb/s VBR, complexity 9, FEC with 10 % expected loss, decoded by libopus. libopus's SNR falls from 29.8 dB in 60 ms frames to 10.2 dB in 80 ms frames and 8.8 dB in 120 ms frames. `opus1` gives 30.2 and 29.6 dB. [Differences from libopus](10-differences-from-libopus.md) lists this with the other deliberate differences.

```rust
use std::num::NonZeroU32;
use opus1::{Application, Bitrate, Channels, OpusDecoder, OpusEncoder, SampleRate};

const FRAME: usize = 5760; // 120 ms at 48 kHz
let mut encoder = OpusEncoder::new(SampleRate::Hz48000, Channels::Mono, Application::Audio).unwrap();
encoder.set_bitrate(Bitrate::BitsPerSecond(NonZeroU32::new(64_000).unwrap()));
let pcm: Vec<f32> = (0..FRAME).map(|i| 0.3 * (i as f32 * 0.03).sin()).collect();
let mut packet = vec![0u8; 6 * 1276];
let len = encoder.encode(&pcm, FRAME, &mut packet).unwrap();
// CELT-only at 64 kb/s: six 20 ms frames, frame count code 3.
assert_eq!(packet[0] & 0x03, 3);
assert_eq!(packet[1] & 0x3f, 6);

let mut decoder = OpusDecoder::new(SampleRate::Hz48000, Channels::Mono).unwrap();
let mut out = vec![0.0f32; FRAME];
assert_eq!(decoder.decode(&packet[..len], FRAME, &mut out).unwrap(), FRAME);
```

#### Expert frame duration

`set_expert_frame_duration` is libopus's `OPUS_SET_EXPERT_FRAME_DURATION`. With any value but `FrameDuration::Argument`, each `encode` call codes one frame of that duration:

- `frame_size` must be at least that long, or the call is refused. It need not be a frame size of Opus itself.
- `input` must still hold `frame_size * channels` samples. The encoder codes the first frame of them.
- The caller learns how much audio was coded from the packet (`packet::nb_samples`), and passes the rest again with the next call. libopus calls this "variable duration": the encoder may code less audio than the call passes. libopus 1.6.1 chooses no duration of its own; the one that is set is used.

The values are those of libopus: `FrameDuration::try_from(5000..=5009)` gives `Argument` (`OPUS_FRAMESIZE_ARG`, 5000) and the durations from 2.5 ms (5001) to 120 ms (5009). Any other number is refused, as libopus refuses it. The setting stays across `reset_state`.

```rust
use opus1::{Application, Channels, FrameDuration, OpusEncoder, SampleRate, packet};

let mut encoder = OpusEncoder::new(SampleRate::Hz48000, Channels::Mono, Application::Audio).unwrap();
encoder.set_expert_frame_duration(FrameDuration::Ms20);
assert_eq!(FrameDuration::try_from(5004), Ok(FrameDuration::Ms20));
assert!(FrameDuration::try_from(5010).is_err());

// 50 ms of audio, which is no frame size of Opus: the call codes the
// first 20 ms of it.
let pcm = vec![0.0f32; 2400];
let mut out = vec![0u8; 1500];
let len = encoder.encode(&pcm, 2400, &mut out).unwrap();
assert_eq!(packet::nb_samples(&out[..len], SampleRate::Hz48000).unwrap(), 960);

// Less than 20 ms is refused.
assert!(encoder.encode(&pcm, 480, &mut out).is_err());
```

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

With `set_vbr(false)` every packet has the size the bitrate gives the frame. With VBR (the default) the sizes vary around the target. The mode decision counts a constant bitrate as worth about 8 % less than the same variable bitrate, as libopus does.

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
use opus1::{Channels, OpusDecoder, SampleRate};

let mut decoder = OpusDecoder::new(SampleRate::Hz48000, Channels::Stereo).unwrap();
let mut out = vec![0.0f32; 960 * 2];
// 20 ms were lost.
let samples = decoder.decode(&[], 960, &mut out).unwrap();
assert_eq!(samples, 960);
```

Concealment needs nothing from the encoder. It works for every mode.

### In-band forward error correction

With in-band FEC each SILK or Hybrid packet can carry a low-bitrate copy of the frame before it. SILK calls this copy LBRR (low bitrate redundancy).

On the encoder:

- Call `set_inband_fec(InbandFec::On)`.
- Call `set_packet_loss_perc` with the loss you expect. With 0 % no copy is coded. The higher the value, the more bits go to the copy.
- Give the encoder enough bitrate (see the table above).
- CELT-only packets carry no copy. With FEC on, the encoder chooses SILK or Hybrid instead of CELT when the expected loss is above 0 % (`Application::Voip`) or above 5 % (`Application::Audio`), as libopus does.

On the decoder, when packet `n` is lost and packet `n + 1` has arrived:

1. Call `decode_fec(packet_n_plus_1, frame_size, out)`. It rebuilds the lost frame from the copy in packet `n + 1`. When packet `n + 1` holds no copy, it conceals the frame instead.
2. Call `decode(packet_n_plus_1, frame_size, out)` as usual.

When several packets in a row are lost, conceal all but the last one with empty packets, then use `decode_fec` for the last one.

```rust
use std::num::NonZeroU32;
use opus1::{Application, Bitrate, Channels, InbandFec, OpusDecoder, OpusEncoder, SampleRate};

const RATE: SampleRate = SampleRate::Hz16000;
const FRAME: usize = 320; // 20 ms

let mut encoder = OpusEncoder::new(RATE, Channels::Mono, Application::Voip).unwrap();
encoder.set_bitrate(Bitrate::BitsPerSecond(NonZeroU32::new(24_000).unwrap()));
encoder.set_inband_fec(InbandFec::On);
encoder.set_packet_loss_perc(10).unwrap();

// 30 packets of a test signal.
let signal: Vec<f32> = (0..FRAME * 30)
    .map(|i| 0.3 * (2.0 * std::f32::consts::PI * 180.0 * i as f32 / RATE.hz() as f32).sin())
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

let mut decoder = OpusDecoder::new(RATE, Channels::Mono).unwrap();
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

### Deep redundancy (DRED)

With the feature `dred`, the decoder can rebuild lost speech from deep redundancy (DRED) data. The encoder of opus1 adds this data to its packets after `set_dred_duration`, and so does a libopus encoder of version 1.5 or later after `OPUS_SET_DRED_DURATION`. Each packet with DRED describes up to about one second of the speech before it, in a few kb/s. So one packet that arrives after a burst of losses can rebuild the whole burst, where in-band FEC rebuilds only the last packet. DRED carries speech features, not audio: deep PLC synthesizes the audio from them. The result follows the lost speech closely, but it is not a copy of it.

The sender sets three things on its `OpusEncoder`:

- `set_dred_duration(n)`: how much audio before each packet the packet describes, in frames of 10 ms, from 1 to 104. 0, the default, turns DRED off.
- `set_packet_loss_perc(p)`: the expected loss in percent. DRED's share of the bitrate grows with it. With no expected loss DRED takes no bits, and with VBR it then writes nothing.
- A bitrate above 12 kb/s (20 kb/s with in-band FEC): DRED's share is a fraction of the bitrate above that. When the share does not pay for at least two latent vectors (40 ms), DRED takes nothing.

The share is taken from the bitrate of the audio, so the packets stay at about the bitrate that was set. In the tests, Hybrid packets of 20 ms at 24 kb/s with 20 % expected loss carried 26 bytes of DRED data on average, in the 287 of 300 packets that had DRED. `reset_state` turns DRED off, as libopus's `OPUS_RESET_STATE` does, so set the duration again after a reset. A `MultistreamEncoder` has no DRED setting, as in libopus: set it on each stream encoder (`stream_encoder_mut`). [The encoder](03-encoder.md#18-deep-redundancy-dred) explains the steps.

The receiver needs three objects per stream:

- the `OpusDecoder`;
- an `OpusDredDecoder`, which holds the weights of the RDOVAE decoder (the network that turns DRED data into features). It keeps nothing from one packet to the next, so one can serve every stream;
- an `OpusDred` (11,236 bytes), which holds the DRED data of one packet.

When the jitter buffer (the queue that holds received packets until it is time to play them) must play lost packets and a later packet has arrived:

1. Call `OpusDredDecoder::parse` on the first packet after the gap. Pass the duration from the start of the first lost packet to the start of that packet as `max_dred_samples` (more than one second does not help), and the decoder's rate as `sampling_rate`. The result's `samples` says how far back the data reaches; 0 means the packet carries no DRED.
2. For each lost packet, oldest first, call `dred_decode_float` (or `dred_decode` for 16-bit output, `dred_decode24` for 24-bit output). `dred_offset` is the number of samples from the start of that lost packet to the start of the DRED packet, and `frame_size` its duration. Audio older than the DRED data reaches is concealed by deep PLC as without DRED, so the call can be made for every lost packet of the burst.
3. Decode the DRED packet itself with `decode`, as usual.

```rust
use std::num::NonZeroU32;
#[cfg(feature = "dred")]
{
    use opus1::{
        Application, Bitrate, Channels, DredProcessing, OpusDecoder, OpusDred, OpusDredDecoder,
        OpusEncoder, SampleRate,
    };

    const RATE: SampleRate = SampleRate::Hz16000;
    const FRAME: usize = 320; // 20 ms

    /// Plays `lost` lost packets of FRAME samples, then `next`, the packet
    /// that arrived after them. Returns the samples played.
    fn play_after_loss(
        decoder: &mut OpusDecoder,
        dred_decoder: &OpusDredDecoder,
        dred: &mut OpusDred,
        next: &[u8],
        lost: usize,
        out: &mut [f32],
    ) -> usize {
        let gap = (lost * FRAME) as i32;
        // Only the DRED data that covers the gap is read.
        let reach = match dred_decoder.parse(dred, next, gap.min(RATE.hz() as i32), RATE, DredProcessing::Now) {
            Ok(found) => found.samples,
            Err(_) => 0, // not a valid packet: conceal
        };
        let mut played = 0;
        for k in 0..lost {
            // From the start of lost packet k to the start of `next`.
            let dred_offset = ((lost - k) * FRAME) as i32;
            played += if reach > 0 {
                decoder.dred_decode_float(dred, dred_offset, FRAME, out).unwrap()
            } else {
                decoder.decode(&[], FRAME, out).unwrap()
            };
        }
        played + decoder.decode(next, FRAME, out).unwrap()
    }

    // The sender: DRED of 1 s, with 20 % expected loss. Packets of a
    // libopus encoder with DRED are read the same way.
    let mut encoder = OpusEncoder::new(RATE, Channels::Mono, Application::Voip).unwrap();
    encoder.set_bitrate(Bitrate::BitsPerSecond(NonZeroU32::new(24_000).unwrap()));
    encoder.set_packet_loss_perc(20).unwrap();
    encoder.set_dred_duration(100).unwrap();
    let signal: Vec<f32> = (0..FRAME * 20)
        .map(|i| 0.3 * (2.0 * std::f32::consts::PI * 180.0 * i as f32 / RATE.hz() as f32).sin())
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

    let mut decoder = OpusDecoder::new(RATE, Channels::Mono).unwrap();
    let dred_decoder = OpusDredDecoder::new();
    let mut dred = OpusDred::new();
    let mut out = vec![0.0f32; FRAME];
    let mut played = 0;
    for packet in &packets[..10] {
        played += decoder.decode(packet, FRAME, &mut out).unwrap();
    }
    // Packets 10 to 14 are lost; packet 15 arrived. All its DRED data
    // (up to 1 s) reaches back over all of them.
    let found = dred_decoder.parse(&mut dred, &packets[15], RATE.hz() as i32, RATE, DredProcessing::Deferred).unwrap();
    assert!(found.samples >= (5 * FRAME) as i32);
    played += play_after_loss(&mut decoder, &dred_decoder, &mut dred, &packets[15], 5, &mut out);
    assert_eq!(played, 16 * FRAME);
}
```

Points for the jitter buffer:

- DRED helps only when the packet after the gap arrives before the lost audio must be played. A buffer that holds packets for a short delay often has it after a short burst. When it has not arrived, conceal with an empty packet.
- One parsed `OpusDred` serves every lost packet of the burst, because decoding does not change it. A new packet needs a new parse.
- `parse` with `DredProcessing::Deferred` only reads the bits, which is cheap. `process_in_place` then runs the network, the costly step. A receiver can parse each packet as it arrives, to know how far back it reaches, and process only the one it uses.
- `dred_offset` need not fall on the start of a packet. Only `frame_size` must be a multiple of 2.5 ms.
- In-band FEC and DRED combine. libopus's test program `opus_demo` uses `decode_fec` for the last lost packet when `packet::has_lbrr` says the next packet carries a copy, and DRED for the others.
- DRED works at every decoder complexity. It needs the deep PLC models, which are built in. [The decoder](04-decoder.md#10-deep-redundancy-dred-dred_decode) explains what the calls do.

## Neural network features of the decoder

libopus 1.5 and later can run small neural networks in the decoder. A default build of libopus leaves them out, and so does a default build of opus1. In opus1 they are cargo features, all off by default:

| Feature | What it adds | libopus `configure` option | Weights it embeds |
|---|---|---|---:|
| `deep-plc` | Deep packet loss concealment (deep PLC): the networks PitchDNN, PLC and FARGAN make up lost speech. | `--enable-deep-plc` | 1,432,768 bytes |
| `osce` | LACE and NoLACE, the two speech enhancers of OSCE (Opus Speech Coding Enhancement), for SILK frames. It turns on `deep-plc`, as `--enable-osce` does. | `--enable-osce` | 1,573,440 bytes more |
| `osce-bwe` | The bandwidth extension (BWE) of OSCE: the network BBWENet extends 16 kHz SILK speech to 48 kHz output. It turns on `osce`. | `--enable-osce` (only the `configure` build of libopus) | 685,888 bytes more |
| `dred` | Deep redundancy (DRED): the encoder writes DRED data (`OpusEncoder::set_dred_duration`), and the decoder rebuilds lost packets from it (`OpusDredDecoder`, `OpusDred` and `OpusDecoder::dred_decode`; see [Deep redundancy (DRED)](#deep-redundancy-dred)). It embeds the RDOVAE encoder and decoder of DRED. It turns on `deep-plc`, as `--enable-dred` does. | `--enable-dred` | 565,440 bytes more |
| `dnn` | The code that the networks share. Each feature above turns it on. It is not meant to be selected alone. | | |

The weights are those of libopus 1.6.1. They are files in the crate (`dnn-weights/`), and each feature embeds the files of its networks in the program. The features work with and without `std` and `heap`.

All of them run in the decoder; `dred` also runs in the encoder. With `dred`, the audio that `dred_decode` makes is that of libopus 1.6.1 built with `--enable-dred`, bit for bit. So are CELT-only packets with DRED data, and the DRED data of SILK and Hybrid packets with the `Audio` application and VBR ([Differences from libopus](10-differences-from-libopus.md#dred-encoding)). The decoded output is that of libopus 1.6.1 built with `--enable-osce` (which includes deep PLC), bit for bit, with one exception: opus1 decodes CELT-only frames that follow frames with the BWE, which libopus 1.6.1 replaces by silence (see [LACE, NoLACE and the bandwidth extension](#lace-nolace-and-the-bandwidth-extension)). [Testing](09-testing.md#tests-of-the-neural-network-code) says how this is checked.

### Deep packet loss concealment

With the feature `deep-plc`, the decoder conceals as libopus 1.6.1 built with `--enable-deep-plc` does, bit for bit (see [Testing](09-testing.md#tests-of-the-neural-network-code) for how this is checked):

- At complexity 5 and above, a lost SILK frame at the 16 kHz SILK rate is made up by deep PLC, 10 ms at a time. That covers wideband SILK and the SILK layer of Hybrid. Narrowband and mediumband SILK keep the classic concealment.
- At complexity 5 and above, a lost CELT-only frame is made up by deep PLC for the first 200 ms of a loss. After that the decoder turns to shaped noise, as without deep PLC after 100 ms. Deep PLC is mono: a stereo decoder gets the same concealed signal on both channels. The CELT layer of Hybrid keeps its noise concealment.
- A switch to or from CELT-only without a redundancy frame is bridged with 5 ms of concealment in the old mode. At complexity 5 and above deep PLC makes these 5 ms, so the output changes also on streams without any loss.
- Forward error correction (`decode_fec`) uses deep PLC wherever it has to conceal.
- At every complexity, each decoded 16 kHz SILK frame goes into deep PLC's history of 150 ms. And a build with the feature skips the fade-in of the first 16 kHz SILK frame after a loss. So a build with `deep-plc` decodes streams with losses differently from a build without it, also at complexity 0, as libopus builds with and without `--enable-deep-plc` do.
- `reset_state` clears deep PLC's state except the state of FARGAN, as libopus does. Concealment after a reset can therefore differ from that of a new decoder, once deep PLC has concealed a loss.
- Each decoder holds about 43 KB more (see [Size of the encoder and the decoder](#size-of-the-encoder-and-the-decoder)). Decoding still allocates nothing.

Deep PLC defers its work to the first lost frame: that frame analyses the history (up to 15 frames of 10 ms) before it conceals. On an Apple M4 Pro, the first lost 20 ms packet of a loss took about 0.36 ms (SILK) and 0.33 ms (CELT), and each further lost packet about 0.09 ms, against 0.005 to 0.008 ms for a packet that arrived. On a Cortex-A55, the deep PLC part of a first lost 20 ms packet takes about 6.9 ms, about 19 times as long as on the M4. [Performance](08-performance.md#deep-plc-in-the-decoder) gives the measurements.

### LACE, NoLACE and the bandwidth extension

With the feature `osce`, the decoder enhances SILK speech as libopus 1.6.1 built with `--enable-osce` does, bit for bit:

- LACE runs at complexity 6, NoLACE at 7 and above. Both are post-filters: a small network computes, every 5 ms, the coefficients of adaptive filters that reduce the coding noise of the decoded speech. NoLACE is larger and adds a non-linear stage.
- They enhance SILK frames of 20 ms at the 16 kHz SILK rate: wideband SILK-only packets of 20, 40 and 60 ms, and the SILK layer of Hybrid packets of 20 ms. Narrowband and mediumband SILK, 10 ms SILK frames and CELT are left as they are. A lost frame is not enhanced.
- The enhancement starts afresh after a frame it does not enhance, a lost frame, a decoder reset and a change of complexity that changes the method. The first frame that could be enhanced then comes out unenhanced, and the next one fades from the unenhanced to the enhanced signal over its first 10 ms. Stereo SILK enhances the mid and the side signal separately. A side signal that a packet does not code keeps its state until it is coded again.
- At every complexity, a build with the feature changes a sample of -32768 to -32767 in these frames, as libopus built with OSCE does.

With the feature `osce-bwe` and `set_osce_bwe(true)`, the bandwidth extension (BWE) makes the 48 kHz output of SILK at the 16 kHz SILK rate, with the band from 8 to 20 kHz that SILK does not code, in place of the resampler:

- It runs at complexity 4 and above, only when the decoder's output rate is 48 kHz. It takes the SILK frames (10 or 20 ms) of SILK-only wideband packets, and the concealment of lost SILK wideband and Hybrid frames (also DTX frames, which are concealed). It does not run on Hybrid packets that arrive: their CELT layer codes the upper band.
- When a Hybrid frame is lost, the BWE makes the whole band, and CELT does not conceal; 2.5 ms of CELT silence let the CELT layer fade out, as libopus does.
- A switch between the BWE and the resampler fades over 10 ms, as in libopus. After CELT-only frames, or when the SILK rate changes, there is no fade, and the BWE starts afresh.
- The BWE output is delayed by 21 samples (0.44 ms) against the resampler's.
- libopus 1.6.1 does not decode CELT-only frames that follow frames with the BWE: it outputs silence until the next SILK frame, and its range decoder loses its place. libopus main fixed this (commit `228a0f85`), and opus1 follows main. So a stream that switches from SILK wideband to CELT-only with the BWE on is not bit-exact against libopus 1.6.1 from the first CELT-only frame on.
- `reset_state` keeps the state of the BWE, as libopus does: after a reset in the middle of a stream with the BWE on, it continues from where it was.

Cost, measured on an Apple M4 Pro with `tests/dnn_osce_decoder_libopus.rs` (test `timing`: the median decode call of a 20 ms SILK wideband packet at 20 kb/s, no loss, mono 48 kHz output, release build, median of 5 runs), in microseconds (µs) per call. A 20 ms packet lasts 20,000 µs.

| Setting | opus1 | libopus 1.6.1, `-O2` without loop vectorization | libopus 1.6.1, `-O2` |
|---|---:|---:|---:|
| Complexity 5 (no enhancement) | 4.6 | 14.9 | 12.6 |
| LACE (complexity 6) | 21.3 | 32.0 | 27.6 |
| NoLACE (complexity 7) | 61.3 | 83.9 | 77.1 |
| BWE (complexity 4) | 56.2 | 93.3 | 84.5 |
| NoLACE and BWE (complexity 7) | 112.9 | 162.2 | 148.6 |

At complexities below 6 libopus still computes the features of LACE and NoLACE for every 20 ms frame at 16 kHz, although it does not use them; opus1 skips them, with the same output.

Cost on an Android phone (HONOR CLK-AN00, measured on 2026-10-20 at commit `94f10e3`): the same `timing` tests of `tests/dnn_osce_decoder_libopus.rs` and `tests/dnn_plc_decoder_libopus.rs`, run on one Cortex-A55 (cpu5) and on one Cortex-A76 (cpu7) with `taskset`. They ran twice, linked with libopus 1.6.1 built with its neural network code by the NDK's clang at `-O2` and without loop vectorization. The opus1 code is the same in both runs, so its two values show how much a run varies. Medians of 5 runs, in µs per decode call of a 20 ms packet (10 ms where named), mono 48 kHz output. Records: `target/perf/final/phone_timing/` (not in git).

| Setting | opus1, A55 | libopus `-O2`, A55 | opus1, A76 | libopus `-O2`, A76 |
|---|---:|---:|---:|---:|
| Complexity 5, SILK wideband, no loss | 53 to 54 | 147 | 17 | 48 |
| LACE (complexity 6) | 440 to 454 | 483 | 103 to 106 | 122 |
| NoLACE (complexity 7) | 1,630 to 1,712 | 1,924 | 335 to 345 | 389 |
| BWE (complexity 4) | 1,684 to 1,691 | 1,789 | 350 to 357 | 438 |
| NoLACE and BWE (complexity 7) | 3,335 to 3,378 | 3,556 | 704 to 725 | 788 |
| Deep PLC, SILK wideband: first lost packet | 6,892 to 8,086 | 9,388 | 1,711 to 1,908 | 2,059 |
| Deep PLC, SILK wideband: further lost packet | 2,138 to 2,619 | 2,951 | 530 to 593 | 576 |
| Deep PLC, CELT 20 ms: first lost packet | 7,532 to 7,979 | 8,678 | 1,729 to 1,761 | 1,987 |
| Deep PLC, CELT 20 ms: further lost packet | 2,488 to 2,883 | 2,917 | 534 to 536 | 592 |
| Deep PLC, CELT 10 ms: first lost packet | 6,195 to 6,279 | 6,975 | 1,259 to 1,305 | 1,492 |
| Deep PLC, CELT 10 ms: further lost packet | 1,436 to 1,531 | 1,636 | 282 to 285 | 315 |

- The heaviest case on the small core is NoLACE with the BWE: about 3.4 ms per 20 ms packet, 17 % of one Cortex-A55. On the Cortex-A76 it is about 0.7 ms (3.5 %).
- The first lost packet of a loss is the most expensive call: deep PLC first analyses up to 15 frames of history. On the Cortex-A55 it takes 6.2 to 8.1 ms, which fits in a 20 ms packet but not with much room on a busy core.
- libopus built without loop vectorization was slower than at `-O2` in every row but one (for example NoLACE and BWE: 3,763 µs on the A55, 840 µs on the A76). The exception is a further lost SILK packet on the A55: 2,655 µs without loop vectorization, 2,951 µs at `-O2`.
- At complexity 5 without loss, libopus also computes the features of LACE and NoLACE, which it does not use there; that is most of the difference in the first row.

Each SILK channel holds 14,344 bytes of OSCE state and 8,220 bytes of BWE state (on the heap with `heap`); see [Size of the encoder and the decoder](#size-of-the-encoder-and-the-decoder). Decoding still allocates nothing.

The controls follow libopus:

| Call | libopus request | What it does |
|---|---|---|
| `OpusDecoder::set_complexity`, `complexity` | `OPUS_SET_COMPLEXITY`, `OPUS_GET_COMPLEXITY` of the decoder | The decoder complexity, 0 to 10. The default is 0. It selects the networks: deep PLC from 5, LACE at 6, NoLACE from 7, and the BWE from 4 when it is switched on. It exists in every build; without the features it is only stored, as in libopus. |
| `OpusDecoder::set_osce_bwe`, `osce_bwe` | `OPUS_SET_OSCE_BWE`, `OPUS_GET_OSCE_BWE` | Switches the BWE on or off. The default is off. It acts at complexity 4 and above with 48 kHz output. Without the `osce-bwe` feature both return an error of kind `ErrorKind::Unimplemented`, as libopus returns `OPUS_UNIMPLEMENTED`. |
| `OpusDecoder::set_dnn_blob`, `dnn_models` | `OPUS_SET_DNN_BLOB` | Loads other weights from a blob in the libopus format, and tells which groups of networks are loaded. Only with a neural network feature. |
| `MultistreamDecoder::set_complexity`, `complexity` | `OPUS_SET_COMPLEXITY`, `OPUS_GET_COMPLEXITY` | Sets every stream; reads the first stream, as libopus does. |
| `MultistreamDecoder::set_osce_bwe`, `set_dnn_blob` | none | Set every stream. libopus's multistream decoder does not forward these requests. |

`reset_state` keeps the complexity, the BWE switch and the loaded weights.

A new decoder has the built-in weights of every network of its build. `set_dnn_blob` replaces them with the weights of a blob, such as the file that libopus's program `dump_weights_blob` writes. The networks load in four groups: deep PLC (PitchDNN, PLC and FARGAN together), LACE, NoLACE and BBWENet. A group is replaced only when the blob holds all of its arrays. The other groups keep their weights. A blob without any complete group is refused, and nothing changes. `DnnBlob::builtin()` brings back the built-in weights.

A `DnnBlob` is made in one of three ways:

- `DnnBlob::from_bytes(&bytes)` copies the bytes into memory that starts on a 64-byte boundary. It needs the `heap` feature.
- `DnnBlob::from_static(bytes)` uses bytes that live as long as the program, without a copy. Every array of the blob must start on a 4-byte boundary.
- `DnnBlob::builtin()` is the built-in weights (with `deep-plc`, `osce` or `osce-bwe`).

Cloning a `DnnBlob` does not copy the bytes. A decoder keeps a clone of the blob for as long as it uses its weights. `set_dnn_blob` allocates the layer tables of each group it loads (with `heap`); decoding allocates nothing.

```rust
use opus1::{Channels, ErrorKind, OpusDecoder, SampleRate};

let mut decoder = OpusDecoder::new(SampleRate::Hz48000, Channels::Mono).unwrap();
assert_eq!(decoder.complexity(), 0);
// Deep PLC and NoLACE, in a build with those features.
decoder.set_complexity(7).unwrap();
assert_eq!(decoder.set_complexity(11).unwrap_err().kind(), ErrorKind::BadArg);
assert_eq!(decoder.complexity(), 7);

// The bandwidth extension needs the `osce-bwe` feature.
match decoder.set_osce_bwe(true) {
    Ok(()) => assert_eq!(decoder.osce_bwe(), Ok(true)),
    Err(e) => assert_eq!(e.kind(), ErrorKind::Unimplemented),
}

#[cfg(feature = "deep-plc")]
{
    use opus1::{DnnBlob, DnnModels};
    assert!(decoder.dnn_models().contains(DnnModels::DEEP_PLC));
    // Bytes that are not a blob are refused.
    let err = DnnBlob::from_bytes(&[1, 2, 3]).err().unwrap();
    assert_eq!(err.kind(), ErrorKind::BadArg);
    // Back to the built-in weights, after another blob was loaded.
    let loaded = decoder.set_dnn_blob(&DnnBlob::builtin()).unwrap();
    assert!(loaded.contains(DnnModels::DEEP_PLC));
}
```

## 16-bit input and output

- `encode_i16` takes interleaved `i16` samples. It converts every sample to float as `sample as f32 / 32768.0` and then encodes as `encode` does, with an input precision of at most 16 bits, as libopus `opus_encode` does. So it writes the packets that `encode` writes for the same audio after `set_lsb_depth(16)`. The precision changes the silence threshold and the noise floor of the analysis, so from complexity 7 on `encode` with the default 24 bits can give other packets.
- `decode_i16` writes interleaved `i16` samples, as libopus `opus_decode` does. It decodes to float first and then applies a soft clipper, which rounds off peaks beyond full scale instead of cutting them. The public function `opus1::pcm_soft_clip` does the same for float buffers.
- `decode_fec_i16` is `decode_fec` with 16-bit output.

```rust
use opus1::{Application, Channels, OpusDecoder, OpusEncoder, SampleRate};

let mut encoder = OpusEncoder::new(SampleRate::Hz48000, Channels::Mono, Application::Audio).unwrap();
let mut decoder = OpusDecoder::new(SampleRate::Hz48000, Channels::Mono).unwrap();

let pcm: Vec<i16> = (0..960).map(|i| ((i % 100) as i16 - 50) * 200).collect();
let mut packet = [0u8; 1276];
let len = encoder.encode_i16(&pcm, 960, &mut packet).unwrap();

// The same audio as float, coded with the precision of 16-bit input,
// gives the same packet.
let pcm_f32: Vec<f32> = pcm.iter().map(|&s| s as f32 / 32768.0).collect();
let mut packet_f32 = [0u8; 1276];
let mut encoder_f32 = OpusEncoder::new(SampleRate::Hz48000, Channels::Mono, Application::Audio).unwrap();
encoder_f32.set_lsb_depth(16).unwrap();
let len_f32 = encoder_f32.encode(&pcm_f32, 960, &mut packet_f32).unwrap();
assert_eq!(packet[..len], packet_f32[..len_f32]);

let mut out = [0i16; 960];
assert_eq!(decoder.decode_i16(&packet[..len], 960, &mut out).unwrap(), 960);
```

## 24-bit input and output

libopus 1.6 added calls for 24-bit samples held in 32-bit integers, with full scale 2^23 (±8388608). opus1 has them with libopus's arithmetic:

- `encode24` divides each sample by 2^23 to a float and encodes as `encode` does, with 24 bits of precision. The speech and music analysis reads the integers. Values beyond 24 bits are accepted and not limited, as in libopus.
- `decode24` and `decode_fec24` decode as `decode` and `decode_fec` do, then multiply each sample by 2^23 and round it to the nearest integer. There is no soft clipping and no limit to 24 bits, as in libopus: a loud signal gives values beyond ±8388607.
- `MultistreamEncoder::encode24`, `MultistreamDecoder::decode24` and `decode_fec24` do the same for multistream packets.

```rust
use opus1::{Application, Channels, OpusDecoder, OpusEncoder, SampleRate};

let mut encoder = OpusEncoder::new(SampleRate::Hz48000, Channels::Mono, Application::Audio).unwrap();
let mut decoder = OpusDecoder::new(SampleRate::Hz48000, Channels::Mono).unwrap();

// A 24-bit tone, and the same tone as float.
let pcm: Vec<i32> = (0..960).map(|i| ((i % 96) as i32 - 48) * 100_000).collect();
let pcm_f32: Vec<f32> = pcm.iter().map(|&s| s as f32 / 8_388_608.0).collect();

let mut packet = [0u8; 1276];
let len = encoder.encode24(&pcm, 960, &mut packet).unwrap();
let mut packet_f32 = [0u8; 1276];
let mut encoder_f32 = OpusEncoder::new(SampleRate::Hz48000, Channels::Mono, Application::Audio).unwrap();
let len_f32 = encoder_f32.encode(&pcm_f32, 960, &mut packet_f32).unwrap();
assert_eq!(packet[..len], packet_f32[..len_f32]);

let mut out = [0i32; 960];
assert_eq!(decoder.decode24(&packet[..len], 960, &mut out).unwrap(), 960);
```

## Decoder output gain, pitch and phase inversion

- `OpusDecoder::set_gain(gain)` scales the decoded audio by `gain` in 1/256 dB, as libopus `OPUS_SET_GAIN` does. 256 is +1 dB. `gain` is an `i16`, and its whole range, -32768 to 32767 (about -128 to +128 dB), is the range of libopus, so the call never fails. The scaling comes before the soft clipping of `decode_i16` and the conversion of `decode24`. `gain()` reads it.
- `OpusDecoder::pitch()` (libopus `OPUS_GET_PITCH`) is the pitch period of the last frame in samples at 48 kHz, or 0: the post-filter period of CELT after a CELT-only frame, SILK's pitch lag otherwise.
- `set_phase_inversion_disabled(true)` on `OpusEncoder` or `OpusDecoder` (libopus `OPUS_SET_PHASE_INVERSION_DISABLED`) turns off the sign inversion in CELT's intensity stereo. On the encoder it makes a mono downmix of the decoded stereo sound better, at a small cost in stereo quality. A mono decoder has it on from the start, because it mixes stereo streams down; a stereo decoder has it off. The multistream encoder and decoder forward it to every stream.

```rust
use opus1::{Channels, OpusDecoder, SampleRate};

let mut decoder = OpusDecoder::new(SampleRate::Hz48000, Channels::Stereo).unwrap();
decoder.set_gain(-6 * 256); // -6 dB
assert_eq!(decoder.gain(), -1536);
assert!(!decoder.phase_inversion_disabled());
decoder.set_phase_inversion_disabled(true);
assert_eq!(decoder.pitch(), 0); // nothing decoded yet
```

## More than two channels

`OpusEncoder` and `OpusDecoder` handle 1 or 2 channels. For 3 to 255 channels, such as 5.1 surround, use `opus1::multistream::MultistreamEncoder` and `MultistreamDecoder`. A multistream packet holds one ordinary Opus packet per stream. Each stream has its own `OpusEncoder` or `OpusDecoder`. The module needs the `std` feature. [Multichannel audio](05-multichannel.md) explains the channel layouts and the settings.

```rust
use opus1::{Application, SampleRate};
use opus1::multistream::{MultistreamDecoder, MultistreamEncoder};

// 5.1 at 48 kHz, channel mapping family 1 (Vorbis channel order).
let mut encoder = MultistreamEncoder::new_surround(SampleRate::Hz48000, 6, 1, Application::Audio).unwrap();
let layout = encoder.layout().clone();
let mut decoder = MultistreamDecoder::with_layout(
    SampleRate::Hz48000,
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
| `analysis` | on | The speech and music analysis of the encoder. It adds about 28 KB of machine code and 24 KB of tables, most of them the weights of its network as `f32` (measured in a release build for aarch64-apple-darwin that encodes one frame). Without it the encoder decides at every complexity as at complexity 6, and `OpusEncoder` is 25,968 bytes smaller without `heap` (see below). It works with and without `std`. |
| `libm` | off | Provides `sin`, `cos`, `sqrt` and the other float functions through the `libm` crate, a pure-Rust port of the musl C library's math functions. A build without `std` needs it. |
| `portable-atomic` | off | For targets without atomic compare-and-swap (CAS) instructions, such as `thumbv6m-none-eabi` (Cortex-M0 and M0+). The shared tables are then built with the atomics of the `portable-atomic` crate. See [Targets without compare-and-swap](#targets-without-compare-and-swap). |
| `deep-plc`, `osce`, `osce-bwe`, `dred` | off | The neural network features of the decoder (`dred` also of the encoder), with the weights of libopus 1.6.1 (see [Neural network features of the decoder](#neural-network-features-of-the-decoder)). `osce` turns on `deep-plc`, `osce-bwe` turns on `osce`, and `dred` turns on `deep-plc`. Each turns on `dnn`. |
| `dnn` | off | The code that the neural network features share. The features above turn it on. |
| `test-libopus-dnn` | off | For the tests of this repository only. It adds nothing to the library. It compiles the tests that call a libopus built with its neural network code. `cargo test --all-features` fails to link against a libopus without that code, such as the Homebrew build. |

Three builds are possible:

- Default (`std`, `heap` and `analysis`): no dependencies at run time.
- `std` without `heap` (`default-features = false, features = ["std", "analysis"]`): the codec state is stored inline, and the `multistream` module is available.
- `#![no_std]` without `alloc` (`default-features = false, features = ["libm", "analysis"]`): no allocator is needed at all. One dependency, `libm` (two with `portable-atomic`). The `multistream` module is not available. A build with neither `std` nor `libm` stops with a compile error that names the missing feature.

```toml
[dependencies.opus1]
version = "0.2"
default-features = false
features = ["libm", "analysis"]
```

The neural network features work in all three builds. Without `heap`, `DnnBlob::from_bytes` is missing, and a blob must live as long as the program (`DnnBlob::from_static`). Without `std`, the networks take `exp`, `log` and the other functions they need from the `libm` crate, whose last bits can differ from those of the platform's C library.

Leave out `analysis` to save its code and tables and the 26 KB of its state, for example on a small processor that encodes at complexity 6 or below anyway.

Without `std`, the x86-64 code uses AVX, AVX2 and FMA only when the build enables them, for example with `RUSTFLAGS="-C target-feature=+avx2,+fma"`. NEON, the vector instruction set of 64-bit ARM, is part of every aarch64 target and needs no check. `scripts/check_no_std.sh` builds the `no_std` crate for `thumbv7em-none-eabi` (Cortex-M), `wasm32-unknown-unknown` and `thumbv6m-none-eabi` (Cortex-M0, with `portable-atomic`) when those targets are installed, then runs the tests without `std`.

### Building a decoder in place

Without `heap`, `OpusDecoder` holds all of its state inline: 59,752 bytes, and up to 160,752 bytes with the neural network features (see below). `OpusDecoder::new` returns the decoder by value, so it builds it on the stack and copies it out. In a release build for aarch64 that takes about 192 KB of stack, or 512 KB with `osce-bwe` and `dred`.

`OpusDecoder::init` builds the decoder directly in memory that the caller provides, a `MaybeUninit<OpusDecoder>`, for example in a `static`. It writes every field in place, so no large value is put on the stack. It returns a reference to the decoder, which is the same as the one of `new`. Upstream opus-rs added the same function in its PR #65.

```rust
use core::mem::MaybeUninit;
use opus1::{Channels, OpusDecoder, SampleRate};

static mut DECODER: MaybeUninit<OpusDecoder> = MaybeUninit::uninit();

// Run once at start-up.
let slot = &raw mut DECODER;
// SAFETY: nothing else uses `DECODER` at the same time.
let decoder: &'static mut OpusDecoder = OpusDecoder::init(unsafe { &mut *slot }, SampleRate::Hz48000, Channels::Stereo)?;

let mut pcm = [0.0f32; 960 * 2];
decoder.decode(&[], 960, &mut pcm)?; // conceals a lost frame
# Ok::<(), opus1::Error>(())
```

Measured in a release build for aarch64 without `heap`, with and without the neural network features:

- `init` needs less than 16 KB of stack. This includes the first call of a program, which also builds the tables that every decoder shares: the CELT mode and, with the neural network features, the built-in models.
- Decoding 48 kHz stereo music, with one packet in seven lost, needed less than 16 KB of stack, or 24 KB with the neural network features at complexity 10. Other signals and modes were not measured.

`init` overwrites what the slot holds without dropping it. A decoder without `heap` owns no other memory, so there is nothing to drop. With `heap`, `init` also works: it writes the boxed parts straight into their heap memory. Drop such a decoder with `MaybeUninit::assume_init_drop`.

The encoder and the multistream decoder have no `init` yet.

### Targets without compare-and-swap

The tables that all encoders and decoders share (the CELT mode and the built-in neural network models) are built on first use. A small atomic state guards each of them, and the first caller claims it with an atomic compare-and-swap (CAS). Some processors have no CAS instruction, for example the ARM Cortex-M0 and M0+ (`thumbv6m-none-eabi`) and the RISC-V cores without the atomic extension, such as the ESP32-C3 (`riscv32imc-unknown-none-elf`). For them, turn on the feature `portable-atomic`. It takes the atomics from the [`portable-atomic`](https://crates.io/crates/portable-atomic) crate, an idea of upstream opus-rs PR #64 by Alix ANNERAUD.

```toml
[dependencies.opus1]
version = "0.2"
default-features = false
features = ["libm", "portable-atomic"]
```

On such a target portable-atomic needs one more setting from the application:

- On a single-core chip, build with `RUSTFLAGS="--cfg portable_atomic_unsafe_assume_single_core"`. portable-atomic then makes the compare-and-swap safe by turning interrupts off around it.
- Otherwise, turn on portable-atomic's `critical-section` feature, and provide an implementation of the [`critical-section`](https://crates.io/crates/critical-section) crate (most hardware support crates have one).

Without one of these, portable-atomic stops the build with an error that says so. Targets that have CAS do not need the feature; with it, they use their own atomic instructions. The `heap` feature needs `std`, and with `std` the crate uses `Arc` for a weight blob loaded at run time; targets without CAS have no `std`, so this does not concern them.

`scripts/check_no_std.sh` builds the crate for `thumbv6m-none-eabi` with `portable-atomic` when that target is installed. The neural network features build there as well.

### Size of the encoder and the decoder

The encoder sizes were measured on 2026-10-18 at commit `1856a50` (the branch of the speech and music analysis), the decoder sizes on the branch that connects deep PLC and OSCE (`next-wave2`). Both use `size_of` and a counting allocator, a release build, and Rust 1.98 on aarch64-apple-darwin. The sizes do not depend on the sampling rate, because every buffer is sized for the largest case (48 kHz stereo). With `heap`, the heap use of an encoder depends on its channel count: a mono encoder has no SILK state for the side channel.

| Build | `OpusEncoder` |
|---|---|
| Default (`heap`, `analysis`) | 2,656 bytes, plus 141,384 bytes on the heap in 8 allocations (mono) or 151,320 bytes in 9 allocations (stereo) |
| `heap` without `analysis` | 2,648 bytes, plus 115,416 bytes in 7 allocations (mono) or 125,352 bytes in 8 allocations (stereo) |
| Without `heap` (with or without `std`), with `analysis` | 153,904 bytes |
| Without `heap` and without `analysis` | 127,936 bytes |
| Default with `dred` (measured on 2026-10-20, branch `next-dred`) | 2,672 bytes, plus 193,816 bytes on the heap in 9 allocations (mono) or 203,760 bytes in 10 allocations (stereo) |
| Without `heap`, with `analysis` and `dred` (the same day) | 210,176 bytes |

| Build | `OpusDecoder` |
|---|---|
| Default (`heap`) | 112 bytes, plus 105,768 bytes on the heap in 5 allocations |
| Default with `deep-plc` | 128 bytes, plus 149,228 bytes on the heap in 7 allocations |
| Default with `osce` | 128 bytes, plus 178,236 bytes on the heap in 9 allocations |
| Default with `osce-bwe` | 128 bytes, plus 194,844 bytes on the heap in 11 allocations |
| Without `heap` (with or without `std`) | 59,752 bytes |
| Without `heap`, with `deep-plc` | 107,288 bytes |
| Without `heap`, with `osce` | 142,136 bytes |
| Without `heap`, with `osce-bwe` | 160,752 bytes |

With `dred`, the encoder holds the state of the DRED encoder. With `heap` it is one allocation of 52,424 bytes. 31,200 bytes of it are the latent vectors and initial states of the last 104 steps of 20 ms; the rest is the 16 kHz input, the feature analysis and the state of the RDOVAE encoder. Without `heap` it takes 56,248 bytes inline, which include room for the layer tables of models loaded from a blob. On the same day the default encoder measured 2,664 bytes plus 141,392 bytes (mono) or 151,336 bytes (stereo), and 153,928 bytes without `heap`.

The state of the speech and music analysis takes 25,968 bytes: 14,004 bytes as in libopus, plus the FFT buffer and the per-bin values that libopus keeps on the stack, and two caches: the square roots of the band energies and the distances between the stored blocks.

With a neural network feature, the decoder holds references to the layer tables of the built-in networks, which every decoder shares. Without `heap`, it also has room for the layer tables of every group that a blob can load (12,584 bytes with `osce-bwe`). The weights themselves are not copied: the embedded ones stay in the program, and a blob's bytes stay in the blob.

With `deep-plc`, the decoder also holds the state of deep PLC: 42,012 bytes (on the heap with `heap`), and 1,292 bytes of 16 kHz samples and filter memory in the CELT decoder.

With `osce`, each of the two SILK channels holds the state of LACE or NoLACE and their features, 14,344 bytes; with `osce-bwe` also the state of the BWE, 8,220 bytes. With `heap` each of these states is an allocation of its own, so that building a decoder does not hold them on the stack.

The heap of the decoder includes a 46,088-byte float buffer for `decode_i16` and `decode_fec_i16`. Without `heap`, these two calls use a buffer of the same size on the stack instead.

Without `heap`, `OpusEncoder::new` and `OpusDecoder::new` return the whole state by value, so building one takes stack room too. One measurement ran each case on a new thread with a given stack size (release build, Rust 1.98, aarch64-apple-darwin):

| Work on the thread | Without `heap` | With `heap` |
|---|---|---|
| Build a 48 kHz stereo encoder and encode one 20 ms frame | 416 kibibytes (KiB, 1024 bytes) was enough, 400 KiB was not | 128 KiB was enough, 112 KiB was not |
| Build a 48 kHz stereo decoder, conceal one frame with `decode`, decode one packet with `decode_i16` | 240 KiB was enough, 224 KiB was not | 96 KiB was enough, 80 KiB was not |
| The same with `deep-plc` at complexity 5, after 40 packets (CELT-only or SILK wideband), so that deep PLC conceals the frame | 344 KiB was enough, 336 KiB was not (208 KiB and 192 KiB without `deep-plc`) | 80 KiB was enough, 64 KiB was not (the same without `deep-plc`) |
| The same with `osce-bwe` at complexity 10 with the BWE on, after 40 packets of stereo SILK wideband (NoLACE, BWE and deep PLC run) | 584 KiB was enough, 576 KiB was not | 84 KiB was enough, 80 KiB was not |

These figures depend on the code around the codec, and debug builds need more. A second program ran each case in a function of its own. Its encoder case needed 432 KiB without `heap` and 144 KiB with it. When its `decode_i16` call concealed a frame instead of decoding a packet, the decoder case needed 272 KiB without `heap` and 128 KiB with it. The repository's `tests/stack_usage_test.rs` checks that, with `heap`, building, encoding and decoding fit in 768 KiB.

### Placing the codec in a static

Without `heap`, keep the encoder and the decoder in a `static` (or in other memory that lives as long as they are used), not in a local variable. `OpusEncoder::new` is not a `const fn`. So the static holds uninitialized memory until the program builds the encoder at run time:

```rust
use std::num::NonZeroU32;
use core::cell::UnsafeCell;
use core::mem::MaybeUninit;
use opus1::{Application, Bitrate, Channels, OpusEncoder, SampleRate};

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
    slot.write(OpusEncoder::new(SampleRate::Hz16000, Channels::Mono, Application::Voip).unwrap())
}

// SAFETY: this is the first and only call.
let encoder = unsafe { init_encoder() };
encoder.set_bitrate(Bitrate::BitsPerSecond(NonZeroU32::new(16_000).unwrap()));
let mut packet = [0u8; 1276];
let len = encoder.encode(&[0.0f32; 320], 320, &mut packet).unwrap();
assert!(len > 0);
```

An embedded framework often has its own way to do this, such as a cell that is initialized once. Any of them works, as long as the codec is not stored on a small stack.

## Errors

Every function that can fail returns `Result<T, opus1::Error>`. An `Error` holds two things:

- `kind()` returns an `ErrorKind`, one per libopus error code. The kinds are `BadArg` (libopus `OPUS_BAD_ARG`), `BufferTooSmall` (`OPUS_BUFFER_TOO_SMALL`), `InternalError` (`OPUS_INTERNAL_ERROR`), `InvalidPacket` (`OPUS_INVALID_PACKET`), `Unimplemented` (`OPUS_UNIMPLEMENTED`), `InvalidState` (`OPUS_INVALID_STATE`) and `AllocFail` (`OPUS_ALLOC_FAIL`). opus1 returns the kind that libopus returns in the same case.
- `message()` returns a short text that says what was wrong, such as `"invalid complexity"`. `Display` writes the same text.

`to_raw()` gives the libopus number of the kind, for example -1 for `BadArg`. That helps where opus1 replaces libopus behind a C interface.

A program should decide what to do from the kind, not from the message. The messages are for logs. Their wording can change in any release. `ErrorKind` is `#[non_exhaustive]`: a later release may add kinds, so a `match` on it needs a `_` arm.

`Error` implements `core::error::Error`, which is `std::error::Error` in a `std` build. So `?` turns it into a `Box<dyn std::error::Error>` or the error type of an application. `Error` is `Copy` and holds no allocation, so it also works in `no_std` builds.

`InternalError` means that the codec reached a state that should not happen. Please report it with the input that caused it.

What the encoder and the decoder reject:

| Call | Input | Kind | Message |
|---|---|---|---|
| `SampleRate::try_from` | a sampling rate other than 8000, 12000, 16000, 24000, 48000 | `BadArg` | `invalid sampling rate` |
| `Channels::try_from` | a channel count other than 1 or 2 | `BadArg` | `invalid number of channels` |
| `encode`, `encode_i16`, `encode24` | a frame size that is not 2.5, 5, 10, 20, 40, 60, 80, 100 or 120 ms; with an expert frame duration, a frame size shorter than that duration | `BadArg` | `invalid frame size` |
| `encode`, `encode_i16`, `encode24` | fewer than `frame_size * channels` input samples | `BadArg` | `input buffer too small for frame` |
| `encode`, `encode_i16`, `encode24` | an empty output buffer | `BadArg` | `output buffer is empty` |
| `encode`, `encode_i16`, `encode24` | a 100 ms frame into a 1-byte buffer | `BufferTooSmall` | `output buffer too small` |
| `decode` and the other decode calls | a packet that breaks the rules of section 3 of RFC 6716 (Request for Comments 6716, the Opus specification) | `InvalidPacket` | `invalid packet` |
| `decode` and the other decode calls | a `frame_size` of 0 | `BadArg` | `invalid frame size` |
| `decode` | a packet longer than `frame_size` | `BufferTooSmall` | `output buffer too small: frame_size is shorter than the packet` |
| `decode` and the other decode calls | an output buffer shorter than the samples to be written | `BufferTooSmall` | `output buffer too small` |
| `decode` with an empty packet, `decode_fec` | a `frame_size` that is not a multiple of 2.5 ms | `BadArg` | `lost frame: frame_size must be a multiple of 2.5 ms` |
| `set_complexity` of the encoders and the decoders | a value above 10 | `BadArg` | `invalid complexity` |
| `OpusDecoder::set_osce_bwe`, `osce_bwe`, `MultistreamDecoder::set_osce_bwe` | a build without the `osce-bwe` feature | `Unimplemented` | `request not implemented` |
| `DnnBlob::from_bytes`, `DnnBlob::from_static` | bytes that are not a blob in the libopus format | `BadArg` | `invalid DNN blob` |
| `DnnBlob::from_static` | a blob with an array that does not start on a 4-byte boundary | `BadArg` | `DNN blob is not aligned` |
| `set_dnn_blob` | a blob that holds no complete group of networks of this build | `BadArg` | `DNN blob holds no model of this build` |
| `set_packet_loss_perc` | a value above 100 | `BadArg` | `invalid packet loss percentage` |
| `set_lsb_depth` | a value outside 8 to 24 | `BadArg` | `invalid LSB depth` |
| `OpusEncoder::set_dred_duration` | a value above 104 | `BadArg` | `invalid DRED duration` |
| `OpusEncoder::set_voice_ratio` | `Some` of a value above 100 | `BadArg` | `invalid voice ratio` |
| `OpusEncoder::set_energy_mask` | a slice whose length is not 21 values per channel | `BadArg` | `invalid energy mask length` |
| `pcm_soft_clip` | no channels, a length that is not a multiple of the channel count, or fewer memory values than channels | `BadArg` | `invalid argument` |
| `OpusEncoder::set_application`, `MultistreamEncoder::set_application` | another application after the first frame (before `reset_state`) | `BadArg` | `application cannot change after the first frame` |

The functions of the modules `packet`, `extensions` and `multistream` use the same kinds. Their documentation lists the errors of each function.

Several settings cannot fail, because their types hold only valid values: `set_bitrate`, `set_bandwidth`, `set_max_bandwidth`, `set_force_channels` and the decoder's `set_gain`. A bitrate below 500 b/s or above 750,000 b/s per channel is not rejected: `set_bitrate` moves it to the nearest limit, as libopus's `OPUS_SET_BITRATE` does.

The decoder checks the packet and the buffer sizes before it changes its state. After an invalid packet or a buffer that is too small, the next packet decodes normally.

```rust
use std::num::NonZeroU32;
use opus1::{Application, Bitrate, Channels, ErrorKind, OpusDecoder, OpusEncoder, SampleRate};

// 44.1 kHz is not an Opus sampling rate.
let err = SampleRate::try_from(44_100u32).unwrap_err();
assert_eq!(err.kind(), ErrorKind::BadArg);
assert_eq!(err.to_raw(), -1); // OPUS_BAD_ARG
assert_eq!(err.to_string(), "invalid sampling rate");

let mut encoder = OpusEncoder::new(SampleRate::Hz48000, Channels::Mono, Application::Audio).unwrap();
let mut packet = [0u8; 1276];
// 441 samples is not a valid frame duration at 48 kHz.
let err = encoder.encode(&[0.0; 441], 441, &mut packet).unwrap_err();
assert_eq!(err.kind(), ErrorKind::BadArg);

// A complexity above 10 is refused, and the old value stays.
assert_eq!(encoder.set_complexity(42).unwrap_err().kind(), ErrorKind::BadArg);
assert_eq!(encoder.complexity(), 9);
// A bitrate above 750 kb/s per channel is limited, not refused.
encoder.set_bitrate(Bitrate::BitsPerSecond(NonZeroU32::new(1_000_000).unwrap()));
assert_eq!(encoder.bitrate_setting(), Bitrate::BitsPerSecond(NonZeroU32::new(750_000).unwrap()));

let mut decoder = OpusDecoder::new(SampleRate::Hz48000, Channels::Mono).unwrap();
let mut out = [0.0f32; 960];
// Frame count code 3 needs a second byte.
let err = decoder.decode(&[0x03], 960, &mut out).unwrap_err();
assert_eq!(err.kind(), ErrorKind::InvalidPacket);
// A lost frame must last a whole number of 2.5 ms.
match decoder.decode(&[], 100, &mut out) {
    Ok(_) => unreachable!(),
    Err(e) if e.kind() == ErrorKind::BadArg => println!("refused: {e}"),
    Err(e) => panic!("unexpected error: {e}"),
}

// `?` turns an `Error` into a `Box<dyn std::error::Error>`.
fn first_packet(pcm: &[f32]) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let mut encoder = OpusEncoder::new(SampleRate::Hz48000, Channels::Mono, Application::Audio)?;
    let mut packet = vec![0u8; 1276];
    let len = encoder.encode(pcm, 960, &mut packet)?;
    packet.truncate(len);
    Ok(packet)
}
assert!(first_packet(&[0.0; 960]).is_ok());
assert!(first_packet(&[0.0; 100]).is_err());
```

## Threads

`OpusEncoder`, `OpusDecoder`, `MultistreamEncoder` and `MultistreamDecoder` are all `Send` and `Sync`. The example below checks this at compile time. In practice:

- Each encoder or decoder holds the state of one audio stream. `encode` and `decode` take `&mut self`, so one thread uses it at a time.
- An encoder or decoder can move to another thread, for example into a worker thread or an async task.
- To share one between threads, put it in a `Mutex`.
- Separate streams need separate encoders and decoders. They do not share mutable state, so they can run in parallel on different threads.

Apart from constant tables, the crate has one global value: the CELT mode, which holds the band layout, the window and the transform tables of the CELT layer. It is built once, at first use, and then only read. With a neural network feature there is a second one: the layer tables of the built-in networks. The first decoder builds them, and reads one byte of every 4 KiB page of the weights, so that the first use of a network does not wait for the system to load those pages. Then they are only read.

```rust
use opus1::multistream::{MultistreamDecoder, MultistreamEncoder};
use opus1::{Application, Channels, OpusDecoder, OpusEncoder, SampleRate};

fn is_send_and_sync<T: Send + Sync>() {}
is_send_and_sync::<OpusEncoder>();
is_send_and_sync::<OpusDecoder>();
is_send_and_sync::<MultistreamEncoder>();
is_send_and_sync::<MultistreamDecoder>();

let mut encoder = OpusEncoder::new(SampleRate::Hz48000, Channels::Stereo, Application::Audio).unwrap();
let worker = std::thread::spawn(move || {
    let mut packet = [0u8; 1276];
    encoder.encode(&[0.0f32; 1920], 960, &mut packet).unwrap()
});
assert!(worker.join().unwrap() > 0);
```

## Memory use

- `encode`, `encode_i16`, `decode`, `decode_fec`, `decode_i16`, `decode_fec_i16` and the multistream `encode` and `decode` allocate nothing. All memory is taken when the encoder or decoder is built.
- One exception in the default build: `OpusEncoder::reset_state` builds a fresh encoder with the same settings. So it allocates the 141,384 bytes (mono) or 151,320 bytes (stereo) of a new state and frees the old one. `OpusDecoder::reset_state` allocates nothing.
- `DnnBlob::from_bytes` allocates a copy of the blob, and `OpusDecoder::set_dnn_blob` allocates the layer tables of each group of networks it loads.
- The sizes of one encoder and one decoder are in [Size of the encoder and the decoder](#size-of-the-encoder-and-the-decoder).
- A multistream encoder or decoder holds one `OpusEncoder` or `OpusDecoder` per stream. For 5.1 sound (4 streams) a `MultistreamEncoder` allocated 703,694 bytes (with the analysis) and a `MultistreamDecoder` 469,600 bytes (643,504 bytes with `deep-plc`, 759,536 bytes with `osce` and 825,968 bytes with `osce-bwe`) in the measurements above. The repository's tests `multistream_enc_no_alloc` and `multistream_dec_no_alloc` check that they allocate nothing after they are built.

## Build settings for speed

opus1 is fast with the default release profile. An application can still gain a few percent with the settings below. None of them changes what the codec computes: the golden hashes of the harness stay the same. The measurements behind them are in [Build settings for applications](08-performance.md#build-settings-for-applications). They were made on Apple M4 Pro and on an Android phone with Cortex-A76 (big) and Cortex-A55 (small, in-order) cores.

### The release profile, on every platform

Cargo takes the release profile from the application's own `Cargo.toml`, so opus1 cannot set it. Add:

```toml
[profile.release]
codegen-units = 1
lto = "fat"
panic = "abort"   # only if the application does not catch panics
```

- `codegen-units = 1` compiles each crate as one unit, so more functions can be inlined.
- `lto = "fat"` (link-time optimization) lets the compiler optimize across crates.
- `panic = "abort"` removes the code that unwinds the stack after a panic.

Measured change of the time of opus1:

| Core | Encode | Decode |
|---|---:|---:|
| Apple M4 Pro | none (-0.2 % to +0.1 %) | 1.6 % to 1.7 % less |
| Cortex-A76 | 1.1 % to 1.5 % less | none (-0.3 % to +0.2 %) |
| Cortex-A55 | 0.4 % to 0.6 % less | 0.4 % to 0.7 % less |

### Android on 64-bit ARM

Most Android phones combine fast cores with slower in-order cores, such as the Cortex-A55. Android moves threads between them, so the codec often runs on a small core. There it takes about three times as long as on a big core. The default build for `aarch64-linux-android` names no core. Naming the Cortex-A55 lets the compiler order the instructions for that core and align the functions to 16 bytes. Put this in the application's `.cargo/config.toml`, and use the release profile above:

```toml
[target.aarch64-linux-android]
rustflags = [
    "-C", "target-cpu=cortex-a55",
    # Keep to the instructions of ARMv8.0-A, so that the program also runs
    # on phones with older cores.
    "-C", "target-feature=-lse,-rdm,-dotprod,-fp16,-rcpc,-ras,-lor,-pan,-vh,-dpb,-crc,-aes,-sha2",
]
```

Measured change of the time of opus1, against the default build with the default profile:

| Core | Encode | Decode |
|---|---:|---:|
| Cortex-A55 | 4.4 % to 4.5 % less | 1.3 % to 1.5 % less |
| Cortex-A76 | 0.5 % to 0.7 % less | 1.1 % to 1.4 % less |

Notes:

1. The second line matters. `target-cpu=cortex-a55` alone allows the instructions of ARMv8.2-A, and the compiler uses some of them: the SILK encoder then contains the dot-product instruction `sdot`. Such a program stops with an illegal-instruction error on a phone whose cores are older, for example the Cortex-A53 or the Cortex-A73. Without these instructions the measured gain was the same.
2. If every phone the application supports has ARMv8.2-A cores (Cortex-A55, Cortex-A75, Cortex-A76 or newer), the second line can go. `-C target-cpu=cortex-a55` alone, with the release profile, measured 4.3 % to 4.4 % less encode time and 1.7 % to 2.0 % less decode time on the Cortex-A55, and 1.3 % less encode time on the Cortex-A76.
3. `rustflags` apply to every crate of the application. A crate that decides at compile time whether to use the AES, SHA or CRC instructions (with `cfg(target_feature = "aes")`, for example) then uses its version without them.
4. Do not build for the big core alone. `-C target-cpu=cortex-a76` gained less than 1 % on the Cortex-A76, but made encoding 15 % and decoding 7.6 % slower on the Cortex-A55.

### macOS on Apple silicon

`-C target-cpu=native` changed nothing measurable on Apple M4 Pro (within 0.3 %). The default target `aarch64-apple-darwin` already includes the instructions the codec uses, FEAT_FCMA among them. Use only the release profile.

iOS was not measured. The target `aarch64-apple-ios` assumes an older core without FEAT_FCMA, so with `std` the FFT checks for it at run time.

### x86-64

These settings were not measured on x86-64 in this release. With `std`, opus1 detects AVX2 and FMA at run time. Without `std`, enable them at build time, for example with `RUSTFLAGS="-C target-feature=+avx2,+fma"` (see [Feature flags](#feature-flags)).

### Code alignment

Applications need no special code alignment. Aligning every function to 64 bytes (`-C llvm-args=-align-all-functions=6`) changed the speed of opus1 by less than 1 % on all three measured cores. It does make measurements on the Cortex-A76 steadier, so it is recommended for comparing two builds there (see [Code placement](08-performance.md#code-placement)).

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
