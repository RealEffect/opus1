# The encoder

This document follows one call of `OpusEncoder::encode` from the caller's samples to the finished packet. It covers the checks on the input, and how the encoder chooses the coding mode, the audio bandwidth and the channel count. It then covers how a frame is split between the SILK and CELT (Constrained Energy Lapped Transform) layers, and how the packet is put together. Next it lists the stages of the SILK encoder and of the CELT encoder in the order the code runs them, each with its source file. The last section lists the encoder settings, what each one changes, and the libopus control it corresponds to. The code follows `opus_encoder.c`, `silk/enc_API.c` and `celt/celt_encoder.c` of libopus 1.6. Comments in the code cite those files with line numbers ("C 1513" means line 1513 of the C file named nearby). The decoder is described in [04-decoder.md](04-decoder.md). The list of all documents is in [README.md](README.md).

## Terms used here

- **API rate**: the sampling rate of the samples that the caller passes through the application programming interface (API): 8, 12, 16, 24 or 48 kilohertz (kHz).
- **Frame**: a stretch of audio that the codec codes as one unit: 2.5, 5, 10, 20, 40 or 60 milliseconds (ms).
- **Packet**: what one call of `encode` returns. It starts with one table-of-contents (TOC) byte, which names the mode, the bandwidth, the frame duration, mono or stereo, and how many frames follow (section 3.1 of RFC 6716, the Request for Comments that defines Opus).
- **Mode**: which coder codes the frame. SILK-only uses the SILK coder alone. CELT-only uses the CELT coder alone. Hybrid codes the audio below 8 kHz with SILK and the audio above 8 kHz with CELT.
- **SILK**: the speech coder of Opus. It describes the signal with a linear predictor and codes the prediction error in the time domain.
- **CELT**: the transform coder of Opus. It codes the spectrum of the signal band by band.
- **Audio bandwidth**: the highest audio frequency a packet carries. Narrowband (NB) is 4 kHz, mediumband (MB) 6 kHz, wideband (WB) 8 kHz, super-wideband (SWB) 12 kHz and fullband (FB) 20 kHz.
- **Internal rate**: the sampling rate SILK works at: 8 kHz for narrowband, 12 kHz for mediumband and 16 kHz for wideband and above.
- **Range coder**: the entropy coder of Opus (RFC 6716 section 4.1). Every symbol of a frame, from SILK and CELT alike, goes into one range-coded byte stream. The type is `RangeCoder` in `src/range_coder.rs`.
- **Constant bitrate (CBR)** and **variable bitrate (VBR)**: with CBR every packet has the same size. With VBR the size follows the signal around a target.
- **Equivalent rate**: the bitrate that a 20 ms VBR frame at complexity 10 would need for the same quality (`compute_equiv_rate`). The mode, bandwidth and forward error correction (FEC) decisions compare it with thresholds.
- **In-band FEC**: a low-bitrate copy of a SILK frame that the next packet carries, so that a receiver can rebuild a lost packet. libopus and this crate call the copy "low bitrate redundancy" (LBRR).
- **Redundant frame**: a 5 ms CELT frame that a packet carries next to a change of mode, so that the decoder can join the two modes without a click.
- **Q notation**: SILK works in fixed point. A value "in Q15" is stored as the integer equal to the value times 2^15. Many field names end in their Q format, for example `speech_activity_q8`.

## Data flow

```text
caller samples: f32 (encode) or i16 (encode_i16), interleaved, API rate
  |
  v
encode_impl          checks, bitrate from the output buffer
  |                  buffer too small -> write_placeholder (TOC-only packet)
  v
decide_mode_and_bandwidth    channels, mode, bandwidth, FEC, redundancy
  |
  |  CELT-only or Hybrid, 40/60 ms --> encode_multiframe --> encode_frame x2 or x3
  |                                                     --> packet::write (one packet)
  v
encode_frame (one frame)
  |
  |  high-pass (Voip) or DC removal (others)
  |  pcm_buf = last 4 ms from delay_buffer + filtered frame
  |
  +--> SILK-only, Hybrid:  16-bit samples, resample to 8/12/16 kHz
  |                        SILK encoder --------------------------+
  |                                                               |  one range coder:
  +--> Hybrid, CELT-only:  pcm_buf (high-band gain, stereo width)  |  SILK symbols first,
  |                        CELT encoder --------------------------+  CELT symbols after
  |
  +--> change of mode:     5 ms redundant CELT frame (own range coder)
  |
  v
packet = TOC byte | main frame | redundant frame | padding (CBR)
```

## 1. One packet: `OpusEncoder::encode`

`encode` (float samples) and `encode_i16` (16-bit samples) both call `encode_impl` in `src/lib.rs`. It is the port of libopus `opus_encode_native`. `encode_i16` first converts its samples to float.

### 1.1 Input checks and conversion

`encode_impl` does these steps in this order:

1. It clears the final range (`final_range()` returns 0 until a frame is coded).
2. It takes the length of the output buffer as the largest packet, but no more than 6 × 1276 bytes. An empty buffer is an error.
3. It checks the frame size. A frame must last 2.5, 5, 10, 20, 40 or 60 ms at the API rate (`frame_rate_from_params`). Any other size is the error "Invalid frame size for sampling rate".
4. It checks that the input holds `frame_size * channels` samples.
5. For stereo input it updates the estimate of the stereo width (`compute_stereo_width`), unless the caller forced mono packets. The width is 0 when both channels carry the same signal and 1 when they differ completely.

The samples are interleaved: left, right, left, right for stereo. Float samples use the range -1.0 to 1.0. `encode_i16` converts every sample to float as `s as f32 / 32768.0` before `encode_impl` runs. Its packets are therefore the same bytes that `encode` gives for that converted input.

### 1.2 The bitrate of this packet

The packet is coded at the bitrate of the setting (`set_bitrate`), or at the bitrate that the size of the output buffer allows, when that is lower (libopus `user_bitrate_to_bitrate`). `Bitrate::Auto` stands for 60 × rate / frame size + rate × channels b/s, and `Bitrate::Max` for 1,500,000 b/s. A short output buffer therefore lowers the bitrate. Pass a buffer of 1276 bytes per 20 ms or more when the bitrate alone should decide.

With a constant bitrate (`set_vbr(false)`) the packet size is fixed: the number of whole bytes that the bitrate gives the frame, rounded to the nearest byte and capped by the buffer. The bitrate becomes the rate of those bytes.

When the packet has no room to code anything, the encoder writes a TOC-only packet instead (`write_placeholder`). That happens in these cases:

- the packet has fewer than 3 bytes;
- the bitrate is below 3 bytes per frame;
- the frame is 40 or 60 ms, and the packet allows fewer than 300 bytes per second or the bitrate is below 2400 bits per second (b/s).

A decoder conceals such a packet as if it were lost. Under CBR the packet is padded to its size.

### 1.3 Mode, bandwidth and channels: `decide_mode_and_bandwidth`

This function makes the decisions libopus makes once per packet (opus_encoder.c lines 1408-1695). It returns them in a `FrameDecisions` value, which every frame of the packet uses.

**Voice estimate.** libopus estimates how likely the input is speech with its speech/music analysis (`analysis.c`). That analysis is not ported. The encoder uses the fixed estimate libopus uses when the analysis does not run: 115 (in Q7, that is 0.90) for the `Voip` application and 48 (0.375) for the others. All thresholds below are interpolated between a "music" value and a "voice" value with the square of this estimate:

```text
threshold = music + voice_est * voice_est * (voice - music) / 2^14
```

**Channel count** (`decide_stream_channels`). A mono encoder codes mono. A stereo encoder codes the count set with `set_force_channels` when there is one. Otherwise it codes stereo when the equivalent rate for two channels is above a threshold between 17 000 b/s (music) and 19 000 b/s (voice). The threshold is 1000 b/s lower while the stream is stereo and 1000 b/s higher while it is mono, so that the choice does not flip on every packet.

**Mode.** The rules apply in this order:

1. `RestrictedLowDelay` is always CELT-only.
2. A mode set with `set_force_mode` is taken as it is, and rules 3 to 5 are skipped. Rules 6 and 7, and the last corrections at the end of this section, can still change it.
3. Otherwise the equivalent rate is compared with a threshold. At or above it the packet is CELT-only; below it, SILK.
4. With in-band FEC on and a packet loss above `(128 - voice_est) / 16` percent (above 0 % for `Voip`, above 5 % for the others), the packet is SILK: only SILK carries FEC.
5. A packet whose budget is below 6000 b/s (9000 b/s for frames shorter than 20 ms) is CELT-only, which can code such small packets.
6. Frames shorter than 10 ms are CELT-only. SILK frames are 10 ms or longer.
7. An encoder for a low-frequency-effects (LFE) channel (`set_lfe`) is CELT-only.

The threshold of step 3 (`compute_mode_threshold`):

| Part | Value |
|---|---|
| "voice" value | 64 000 b/s for mono, moving to 44 000 b/s as the stereo width goes to 1 |
| "music" value | 10 000 b/s |
| `Voip` application | +8000 b/s |
| previous packet CELT-only / not CELT-only | -4000 / +4000 b/s (none before the first packet) |
| `RestrictedLowDelay` | 0 |

For a mono `Voip` encoder the threshold is 10 000 + 43 588 + 8 000 = 61 588 b/s. For a mono `Audio` encoder it is 17 593 b/s.

For the channel and mode decisions, the equivalent rate (`compute_equiv_rate`) starts from the bitrate. Then:

- for frames shorter than 20 ms it takes off their overhead, `(40 * channels + 20) * (frame_rate - 50)` b/s,
- under CBR it takes off 1/12,
- it multiplies the result by `(90 + complexity) / 100`,
- for packet loss it takes off `rate * loss / (12 * loss + 20)`.

The bandwidth and FEC decisions use the equivalent rate for the mode just chosen. For SILK-only and Hybrid it takes off `rate * loss / (6 * loss + 10)` for packet loss, and 1/5 more below complexity 2. For CELT-only it takes off nothing for loss, and 1/10 below complexity 5.

The first example below shows the switch for a 16 kHz `Voip` encoder at complexity 9. At 62 000 b/s the equivalent rate is 61 380 b/s, below the threshold, and the packet is SILK. At 62 300 b/s it is 61 677 b/s, and the packet is CELT-only.

```rust
use opus1::{Application, Bitrate, OpusEncoder};

// The top five bits of the TOC byte are the configuration number
// (RFC 6716 section 3.1).
fn mode_of(toc: u8) -> &'static str {
    match toc >> 3 {
        0..=11 => "SILK-only",
        12..=15 => "Hybrid",
        _ => "CELT-only",
    }
}

// 20 ms of a 220 Hz tone at 16 kHz.
let pcm: Vec<f32> = (0..320)
    .map(|i| 0.3 * (2.0 * std::f32::consts::PI * 220.0 * i as f32 / 16_000.0).sin())
    .collect();
let mut packet = vec![0u8; 1276];
for (bitrate, expected) in [(62_000, "SILK-only"), (62_300, "CELT-only")] {
    let mut encoder = OpusEncoder::new(16_000, 1, Application::Voip).unwrap();
    encoder.set_bitrate(Bitrate::BitsPerSecond(bitrate)).unwrap();
    let len = encoder.encode(&pcm, 320, &mut packet).unwrap();
    assert!(len > 1);
    assert_eq!(mode_of(packet[0]), expected);
}
```

**Redundant frame and mode switches.** When the mode changes between CELT-only and SILK-only or Hybrid, the decision asks for a redundant 5 ms CELT frame:

- Leaving CELT-only (`celt_to_silk`): the first packet in the new mode carries the redundant frame. It continues the CELT signal of the packets before.
- Entering CELT-only (`to_celt`): this packet is still coded in the old mode and carries the redundant frame at its end. The next packet is CELT-only and continues from that redundant frame. A frame shorter than 10 ms cannot stay in the old mode, so it switches at once, without a redundant frame.

A stereo stream that changes to mono codes one more stereo packet with the SILK side signal faded out (`silk_to_mono`), as libopus does. This does not apply next to CELT-only packets, which change the channel count at once.

**Bandwidth.** The encoder chooses the bandwidth from the equivalent rate only for CELT-only packets, for the first packet, and when SILK allows a switch. SILK allows a switch while its speech activity is low, and after 5 seconds at the latest. The thresholds are equivalent rates in b/s (`VOICE_BANDWIDTH_THRESHOLDS`, `MUSIC_BANDWIDTH_THRESHOLDS`):

| Step | Music | Voice | Hysteresis |
|---|---:|---:|---:|
| NB to MB | 9 000 | 9 000 | 700 |
| MB to WB | 9 000 | 9 000 | 700 |
| WB to SWB | 11 000 | 13 500 | 1 000 |
| SWB to FB | 12 000 | 14 000 | 2 000 |

After the first packet, each threshold is lowered by its hysteresis when the last automatic choice was at least the wider of the two bandwidths, and raised by it otherwise. This keeps the choice from switching back and forth.

The search starts at fullband and steps down until the rate reaches the threshold. Then these rules apply, in order:

1. Mediumband becomes wideband. Mediumband is used only on request, or while SILK steps through 12 kHz.
2. A running SILK encoder does not go above wideband until it runs at 16 kHz and has finished its low-pass transition.
3. The bandwidth does not exceed the maximum set with `set_max_bandwidth`.
4. A bandwidth set with `set_bandwidth` replaces the result.
5. SILK and Hybrid packets that cannot exceed 15 000 b/s stay at wideband or below.
6. The bandwidth does not exceed half the API rate: 8 kHz input is narrowband, 12 kHz mediumband, 16 kHz wideband, 24 kHz super-wideband.

**In-band FEC** (`decide_fec`). FEC is off when `set_inband_fec(true)` was not called, when the packet loss is 0 %, or for CELT-only packets. Otherwise the equivalent rate must exceed a threshold for the bandwidth (`FEC_THRESHOLDS`):

| Bandwidth | NB | MB | WB | SWB | FB |
|---|---:|---:|---:|---:|---:|
| Threshold (b/s) | 12 000 | 14 000 | 16 000 | 20 000 | 22 000 |

The threshold is 1000 b/s lower when the last packet had FEC and 1000 b/s higher when it had none. It is then scaled by about `(125 - min(loss, 25)) / 100`. When the rate is too low and the loss is above 5 %, the bandwidth goes down one step at a time until FEC fits. If no bandwidth fits, FEC stays off and the bandwidth stays as it was.

**Last corrections.** A CELT-only packet has no mediumband and codes wideband instead. An LFE packet is narrowband. SILK above wideband becomes Hybrid. Hybrid at wideband or below becomes SILK-only.

### 1.4 One frame: `encode_frame`

`encode_frame` codes one frame into one packet. It is the port of libopus `opus_encode_frame_native`. Each mode takes these frame durations:

| Mode | Frame durations |
|---|---|
| CELT-only | 2.5, 5, 10, 20 ms |
| Hybrid | 10, 20 ms |
| SILK-only | 10, 20, 40, 60 ms |

#### The filtered frame and the look-ahead buffer

The input goes through a high-pass filter first:

- `Voip`: `hp_cutoff_float` (`src/hp_cutoff.rs`), a second-order high-pass filter. Its cutoff starts at 60 hertz (Hz) and follows the pitch that SILK finds (`silk_hp_variable_cutoff` updates it, and `encode_frame` smooths it).
- `Audio` and `RestrictedLowDelay`: `dc_reject_float`, a first-order filter with a 3 Hz cutoff that removes a constant offset.

A frame whose filtered samples have a sum of squares of 1e9 or more, or that holds a value that is not a number, is coded as silence.

The CELT layer codes the signal 4 ms late. This lines it up with SILK, whose resampler and analysis add their own delay. The encoder keeps 10 ms of filtered input in `delay_buffer`. It builds `pcm_buf` (the field `buf_celt_pcm`) from the last 4 ms of that buffer followed by the filtered frame. CELT codes the first `frame_size` samples of `pcm_buf`. `RestrictedLowDelay` has no 4 ms delay and no delay buffer. The total delay is `lookahead()`: 2.5 ms of CELT window overlap plus the 4 ms.

```rust
use opus1::{Application, OpusEncoder};

let encoder = OpusEncoder::new(48_000, 1, Application::Audio).unwrap();
assert_eq!(encoder.lookahead(), 312); // 2.5 ms + 4 ms at 48 kHz
let low_delay = OpusEncoder::new(48_000, 1, Application::RestrictedLowDelay).unwrap();
assert_eq!(low_delay.lookahead(), 120); // 2.5 ms
```

#### The SILK part (SILK-only and Hybrid)

`encode_frame` prepares the SILK call:

1. **Bit budget.** SILK codes at the packet's bitrate, capped by the packet size without the redundant frame, less 8 bits for the TOC byte (`bits_target`, `total_bitrate`).
2. **SILK bitrate.** In a SILK-only packet SILK gets the whole bitrate. In a Hybrid packet it gets the share that `compute_silk_rate_for_hybrid` gives (below), and CELT gets the rest.
3. **Bit limit** (`silk_max_bits`): the packet less the TOC byte, less the redundant frame and its signalling. In Hybrid CBR, SILK may take up to a quarter of the bits beyond its rate. In Hybrid VBR the limit is the SILK share of the largest packet allowed.
4. **Internal rates allowed.** The rate asked for is that of the bandwidth: 8, 12 or 16 kHz. Hybrid allows 16 kHz only. A SILK-only packet with few bytes lowers the highest rate allowed to 12 or 8 kHz.
5. **Input at 16 bits.** With `Voip` the SILK input comes from the integer high-pass filter `hp_cutoff`, as in the fixed-point build of libopus. With the other applications it is the float-filtered frame from `pcm_buf`, rounded to 16 bits, as in the float build.
6. **Restart.** After CELT-only packets the SILK encoder starts from a clean state. It is then fed the 10 ms of the delay buffer without coding them (C `prefill`). In those 10 ms everything before the last 6.5 ms is cleared, and the first 2.5 ms of the 6.5 ms fade in. The same restart happens for the first frame at a new SILK internal rate. In that case the state of the transition low-pass filter is kept. `encode_frame` passes the restart to SILK as a `SilkPrefill`.

The settings go to SILK in a `SilkControl`. `SilkEncoder::encode` (`src/silk/enc_api.rs`, the port of libopus `silk_Encode`) then does these steps:

1. **Internal rate** (`control_audio_bandwidth`, which calls `silk_control_audio_bandwidth` in `src/silk/control_codec.rs`). A fresh SILK encoder takes the rate asked for. A running one moves one step at a time, and only when SILK allows a switch.
2. **Restart**, when `encode_frame` asked for one: the coded channels are cleared.
3. **Control call** (`silk_control_encoder`, `src/silk/control_codec.rs`) for each coded channel: internal rate, packet duration, bitrate and complexity.
4. **Prefill**, after a restart: the 10 ms before the packet are resampled and analysed, but not coded (`silk_encode_prefill`).
5. **Resampling** of the packet to the internal rate (`SilkResampler::process`, `src/silk/resampler.rs`). Each input channel has its own resampler. SILK converts to mid and side at the internal rate. Stereo input coded as mono is mixed down to one channel first.
6. **SILK encoding** of the packet with `silk_encode_packet`, for one or two coded channels. Section 2 describes it.
7. **Report.** SILK reports its internal rate and its stereo width. It also reports whether it allows a bandwidth switch and whether it is ready to change its internal rate.

Then `encode_frame` uses the report. A SILK-only TOC names the bandwidth of the rate SILK really coded at. SILK allows a bandwidth switch while its speech activity is below a threshold. When SILK is ready to change its internal rate, this packet ends with a redundant frame and the next one switches.

SILK's share of a Hybrid packet, per channel (`compute_silk_rate_for_hybrid`; the 10 ms columns of the code have the same values as the 20 ms ones):

| Total rate per channel (b/s) | SILK rate, no FEC | SILK rate, FEC |
|---:|---:|---:|
| 12 000 | 10 000 | 11 000 |
| 16 000 | 13 500 | 15 000 |
| 20 000 | 16 000 | 18 000 |
| 24 000 | 18 000 | 21 000 |
| 32 000 | 22 000 | 28 000 |
| 64 000 | 38 000 | 50 000 |

Rates between two rows are interpolated linearly. Above 64 000 b/s SILK gets half of the extra bits. CBR adds 100 b/s and super-wideband 300 b/s. The result is multiplied by the channel count. Stereo then takes off 1000 b/s when the rate per channel is 12 000 b/s or more.

The fewer bits the CELT layer of a Hybrid frame gets, the more the encoder attenuates the band above 8 kHz before CELT codes it. The gain is `1 - 2^(-celt_rate / 1024)`, where `celt_rate` is the total rate less the SILK rate (`hb_gain`, applied with `gain_fade` over 2.5 ms). A surround stream with an energy mask skips this.

#### The CELT part (CELT-only and Hybrid)

Before CELT codes, `encode_frame` sets its controls for this frame. These are the same values libopus sends with `celt_encoder_ctl`:

- complexity,
- end band from the TOC bandwidth: 13 bands for narrowband, 17 for medium- and wideband, 19 for super-wideband, 21 for fullband,
- coded channel count, packet loss, LFE flag and energy mask,
- for Hybrid, the SILK frame's signal type and quantization offset (`set_silk_info`), which change CELT's bit target.

Then it adjusts the input and the rate:

- **Stereo width.** At low bitrates the stereo image of the CELT input is narrowed: full width above an equivalent rate of 32 000 b/s, none below 16 000 b/s (`stereo_fade`). A stereo Hybrid frame uses the width SILK chose.
- **Start band.** Hybrid starts at band 17 (8 kHz). CELT-only starts at band 0.
- **Bitrate.** CBR: no CELT bitrate limit; CELT fills the bytes the packet has left. CELT-only VBR: the packet's bitrate, with constrained VBR. Hybrid VBR: the bitrate less the SILK rate, at least 501 b/s, with unconstrained VBR.
- **Restart.** When the mode changes, the CELT encoder starts from a clean state. It first codes the 2.5 ms before the frame into 2 bytes that are thrown away, to fill its filter memories. The decoder resets its CELT state at the same place.

The CELT symbols go into the same range coder as the SILK symbols, after them. Section 3 describes the CELT encoder.

#### Redundant frames

A redundant frame is a 5 ms CELT frame that codes all bands at a constant bitrate with its own range coder (`encode_celt_redundancy`). Its size comes from `compute_redundancy_bytes`. That function adds the extra overhead of 5 ms frames to the packet's bitrate and raises the result by half. The size is the number of bytes this rate gives 5 ms of audio. It is capped by the room in the packet and by 257 bytes. A frame that would get 4 + 8 × channels bytes or fewer is not sent. The decoder then conceals instead.

The packet says whether a redundant frame follows, when there is room for the signalling (17 bits, 37 for Hybrid):

| Field | Hybrid | SILK-only |
|---|---|---|
| "redundant frame follows" | 1 bit, probability 1/4096 | implied: the bytes after the SILK data are the redundant frame |
| position (`celt_to_silk`) | 1 bit, probability 1/2 | 1 bit, probability 1/2 |
| length | `enc_uint(bytes - 2, 256)` | the rest of the packet |

The two directions are coded differently:

- **CELT-only to SILK or Hybrid** (`celt_to_silk`): the redundant frame is coded first, from the start of `pcm_buf`, so that it continues the old CELT stream. Then the CELT encoder is reset.
- **SILK or Hybrid to CELT-only**: the CELT encoder is reset, codes 2.5 ms into 2 thrown-away bytes, then codes the last 5 ms of the frame as the redundant frame. The CELT-only packets that follow continue from it.

#### Packet assembly

1. The range coder is finished (`RangeCoder::done`).
2. If the frame did not fit its bytes, the packet becomes the TOC byte and one zero byte. A decoder conceals such a packet. The libopus comment here reads "tell the decoder to call the PLC", where PLC is packet loss concealment.
3. Otherwise the main frame is trimmed. A SILK-only frame without a redundant frame drops its trailing zero bytes, because the range decoder reads missing bytes as zeros. A CELT frame keeps the bytes VBR used.
4. The packet is the TOC byte, the main frame and the redundant frame.
5. The final range is the range coder state after the main frame, combined by exclusive or (XOR) with that of the redundant frame (`final_range()`).
6. Under CBR the packet is padded to its fixed size (`packet::pad`).

### 1.5 40 and 60 ms packets

SILK-only codes a 40 or 60 ms frame as one Opus frame. Inside it SILK codes 2 or 3 frames of 20 ms. The TOC names the 40 or 60 ms duration, and the frame count code is 0.

CELT and Hybrid frames are 20 ms at most. `encode_multiframe` therefore codes a 40 or 60 ms CELT or Hybrid packet as 2 or 3 frames of 20 ms. Each goes through `encode_frame` with the decisions of the whole packet. `packet::write` then joins them into one packet with frame count code 1, 2 or 3 (libopus does this with its repacketizer). Each 20 ms frame gets its share of the packet. A switch to CELT-only is announced in the last frame only.

libopus 1.6 also limits each 20 ms frame of a VBR packet to what the bitrate gives 20 ms (opus_encoder.c line 1785). That limit is left out on purpose. In libopus it costs up to 23 decibels (dB) of signal-to-noise ratio with in-band FEC on.

```rust
use opus1::{Application, OpusEncoder};

let mut encoder = OpusEncoder::new(48_000, 1, Application::Audio).unwrap();
let pcm = vec![0.0f32; 2880]; // 60 ms at 48 kHz
let mut packet = vec![0u8; 1276];
let len = encoder.encode(&pcm, 2880, &mut packet).unwrap();
let parsed = opus1::packet::parse(&packet[..len], false).unwrap();
assert!(packet[0] >> 3 >= 16); // CELT-only at the automatic 49 kb/s
assert_eq!(encoder.bitrate(), 48_000 + 60 * 48_000 / 960); // of the last 20 ms frame
assert_eq!(packet[0] & 0x03, 3); // frame count code 3
assert_eq!(parsed.count, 3); // three 20 ms frames
```

## 2. The SILK encoder

SILK codes speech by predicting each sample from the samples before it and coding what the prediction misses. It works in fixed point and is a port of the fixed-point SILK encoder of libopus (`silk/fixed/`). A SILK frame is 10 ms (2 subframes of 5 ms) or 20 ms (4 subframes). The packet-level code is in `src/silk/enc_api.rs`: `SilkEncoder`, which owns the state of each channel, and `silk_encode_packet`, which codes one packet. The frame encoder is `silk_encode_frame` in the same file.

The table lists the stages in the order the code runs them for each frame.

| Stage | What it does | File | Main function |
|---|---|---|---|
| Resampling | API rate to the internal rate (8, 12 or 16 kHz); runs in `SilkEncoder::encode` before the packet is coded | `src/silk/resampler.rs` | `SilkResampler::init_for_enc`, `process` |
| Packet header and rate | voice activity flags and LBRR flags, the LBRR section, the bit target of each frame | `src/silk/enc_api.rs` | `silk_encode_packet`, `silk_frame_target_rate_bps` |
| High-pass cutoff | moves the cutoff of the input high-pass filter after voiced frames | `src/silk/hp_variable_cutoff.rs` | `silk_hp_variable_cutoff` |
| Signal-to-noise ratio (SNR) target | the SNR the frame aims for, from its bit target | `src/silk/control_snr.rs` | `silk_control_snr` |
| Stereo to mid/side | stereo only: mid and side signals and the predictor between them | `src/silk/stereo_lr_to_ms.rs` | `silk_stereo_lr_to_ms` |
| Voice activity detection (VAD) | speech activity, input tilt and band quality | `src/silk/vad.rs`, `ana_filt_bank_1.rs` | `silk_encode_do_vad`, `silk_vad_get_sa_q8` |
| Transition low-pass | fades the band above the new bandwidth during a bandwidth switch | `src/silk/lp_variable_cutoff.rs` | `silk_lp_variable_cutoff` |
| Pitch analysis | voiced or unvoiced, pitch lag of each subframe | `src/silk/pitch_analysis.rs` | `silk_find_pitch_lags_fix`, `silk_pitch_analysis_core` |
| Noise shaping analysis | shaping filters, subframe gains, quantization offset type | `src/silk/noise_shape_analysis.rs` | `silk_noise_shape_analysis_fix` |
| Long-term prediction (LTP) | voiced frames: 5-tap pitch predictor and its quantization | `src/silk/ltp_analysis.rs`, `gain_quant.rs`, `vq_wmat_ec.rs` | `silk_find_ltp_fix`, `silk_quant_ltp_gains`, `silk_vq_wmat_ec` |
| Linear predictive coding (LPC) analysis | short-term predictor with Burg's method | `src/silk/lpc_analysis.rs` | `silk_find_lpc_fix`, `silk_burg_modified_fix` |
| Normalized line spectral frequency (NLSF) quantization | predictor as NLSFs, two-stage quantization | `src/silk/nlsf.rs`, `nlsf_encode.rs`, `nlsf_del_dec_quant.rs` | `silk_process_nlsfs`, `silk_nlsf_encode`, `silk_nlsf_del_dec_quant` |
| Gain quantization | final subframe gains, quantized in the log domain | `src/silk/control_fixed.rs`, `gain_quant.rs` | `silk_process_gains_fix`, `silk_gains_quant` |
| In-band FEC (LBRR) | low-bitrate copy of the frame for the next packet | `src/silk/enc_api.rs` | `silk_lbrr_encode` |
| Noise shaping quantizer (NSQ) | the excitation pulses; delayed-decision search | `src/silk/nsq.rs`, `nsq_del_dec.rs` | `silk_nsq`, `silk_nsq_del_dec` |
| Range coding | parameters and pulses into the range coder | `src/silk/encode_indices.rs`, `encode_pulses.rs`, `shell_coder.rs` | `silk_encode_indices`, `silk_encode_pulses` |

`silk_find_pred_coefs_fix` in `src/silk/control_fixed.rs` runs the long-term prediction, the LPC analysis and the NLSF quantization, in that order.

### 2.1 Packet level and rate control

`silk_encode_packet` codes one SILK packet of 1, 2 or 3 frames into the range coder of the Opus frame. The packet starts with one voice activity flag per frame and one LBRR flag. Their values are only known at the end, so the encoder reserves the bits first and writes them last (`RangeCoder::patch_initial_bits`). Next comes the LBRR section: the low-bitrate copies that the previous packet's frames left. Then come the frames.

Each frame gets a bit target (`silk_frame_target_rate_bps`). The target is the frame's share of the bitrate, less the LBRR section. It is lowered further when earlier frames and packets used more bits than their targets. This store of extra bits is called the "bit reservoir", and it is paid back over `BITRESERVOIR_DECAY_TIME_MS`. In a 40 or 60 ms packet the first frames are capped at a share of the packet (2/5 and 3/4 for 60 ms, 3/5 for 40 ms), so that the last frame keeps room.

### 2.2 Stereo

For two coded channels, `silk_encode_packet` converts left and right to mid (the average) and side (the difference) for each frame (`silk_stereo_lr_to_ms`). It predicts the side from the mid and codes only what the prediction misses. The two quantized predictor values are written in front of the frame. When the side carries little, the frame codes the mid only and says so with a flag. The bits are split between mid and side from the frame's target. Each of the two channels has its own `SilkEncoderState`: the `mid` channel of `SilkEncoder` codes the mid, and its `side` channel codes the side. A mono encoder has no `side` channel.

### 2.3 Voice activity detection

The voice activity detector (VAD, `silk_vad_get_sa_q8`) splits the frame into 4 frequency bands with a filter bank (`ana_filt_bank_1.rs`). It compares the energy of each band with a running estimate of the background noise. The result is the speech activity in Q8 (0 to 255), the spectral tilt of the input, and a quality value per band. Below an activity of 13 (`SPEECH_ACTIVITY_DTX_THRES_Q8`) the frame is marked as having no voice activity. The speech activity is also an input to the pitch threshold, the noise shaping, the rate control, the LBRR decision and the bandwidth switch.

### 2.4 Pitch analysis

`silk_find_pitch_lags_fix` first whitens the signal (flattens its spectrum) with a short linear predictor (order 6 to 16, depending on the complexity). `silk_pitch_analysis_core` then searches the pitch lag in three stages: at 4 kHz, at 8 kHz, and at the internal rate with a codebook of lag contours over the subframes. Lags range from 2 to 18 ms. A frame whose correlation passes the threshold is voiced and gets a lag for each subframe. The others are unvoiced. Frames without voice activity, and the first frame after a reset, skip the search.

### 2.5 Noise shaping analysis

The quantizer does not try to make the coding noise as small as possible. It shapes the spectrum of the noise so that the speech masks it. The noise is louder where the speech spectrum is loud, and quieter between the pitch harmonics. `silk_noise_shape_analysis_fix` computes the filters that do this for each subframe:

- a shaping filter of order 12 to 24 from a warped autocorrelation (from complexity 4), so that the low frequencies get more detail,
- the subframe gains from the energy that filter leaves,
- the harmonic shaping gain, the spectral tilt and the low-frequency shaping,
- for unvoiced frames, the quantization offset type from how sparse the signal is.

### 2.6 Long-term prediction

For voiced frames the long-term predictor (LTP) predicts a sample from the samples one pitch period earlier, with 5 taps. `silk_find_ltp_fix` computes the correlations of the pitch-whitened signal. `silk_quant_ltp_gains` picks the 5 taps of each subframe from codebooks with a rate-distortion search (`silk_vq_wmat_ec`). For the first frame of a packet, which does not depend on the frame before, `silk_ltp_scale_ctrl_fix` chooses how far to scale down the prediction from the previous packet's signal. It scales down more when the packet loss is high. A scaled-down prediction recovers faster after a loss. `silk_ltp_analysis_filter_fix` then removes the long-term prediction from the signal before the LPC analysis.

### 2.7 LPC analysis and NLSF quantization

The short-term predictor uses linear predictive coding (LPC): it predicts a sample from the 10 (8 and 12 kHz) or 16 (16 kHz) samples before it. `silk_find_lpc_fix` computes the predictor with Burg's method (`silk_burg_modified_fix`). Burg's method finds the coefficients one order at a time from the forward and backward prediction errors of the signal. The input is the frame with the long-term prediction removed (voiced) or the frame scaled by the inverse subframe gains (unvoiced). For 20 ms frames at complexity 4 and above it also tests whether interpolating between the last frame's predictor and this one works better for the first half of the frame. The interpolation factor is coded.

The predictor is converted to normalized line spectral frequencies (NLSF, `silk_a2nlsf`). NLSFs are an equivalent form of the predictor that stays stable when it is quantized. `silk_nlsf_encode` quantizes them in two stages:

1. A vector codebook of the first stage gives a few candidates (the "survivors": 2 to 16 depending on the complexity).
2. For each candidate, the remaining error is quantized coefficient by coefficient with a delayed-decision search (`silk_nlsf_del_dec_quant`) that weighs error against bits.

The encoder keeps the best candidate. The quantized NLSFs are converted back to predictor coefficients (`silk_nlsf2a`), so that encoder and decoder use the same predictor.

### 2.8 Gains

`silk_process_gains_fix` adjusts the subframe gains for the target SNR and the residual energy, and quantizes them in the log domain with 64 levels (`silk_gains_quant`). The first gain of a frame coded on its own (the first frame of a packet) is coded as a level. Every other gain is coded as a step from the gain before. It also sets the quantization offset for voiced frames and the rate-distortion weight `lambda_q10` the quantizer uses.

### 2.9 In-band FEC (LBRR)

When FEC is on for the packet and the speech activity is above 0.3, `silk_lbrr_encode` quantizes the frame a second time, at a higher gain and therefore with fewer bits. For the first frame of an LBRR run the gain index goes up by `lbrr_gain_increases`: 7 for the first packet with FEC, then `max(7 - 0.2 * loss, 3)` (`silk_setup_lbrr`). It reuses all other parameters of the frame. The copy is stored in the encoder state, and the next packet writes it in its LBRR section. A decoder that lost this packet can then rebuild it from the next one.

### 2.10 Noise shaping quantizer

The noise shaping quantizer (NSQ) turns the frame into excitation pulses: small integers, one per sample, that drive the LTP and LPC synthesis filters. It runs the synthesis filters and the shaping filters sample by sample, so that the coding noise takes the shape computed in 2.5.

- `silk_nsq` (complexity 0 and 1) decides each sample at once.
- `silk_nsq_del_dec` (complexity 2 and above) keeps 2 to 4 candidate paths and decides each sample only 40 samples later (`DECISION_DELAY`). It keeps the path with the lowest weighted error. Each path also tries a different random seed, and the seed of the kept path is coded.

### 2.11 Rate control loop and range coding

`silk_encode_frame` runs the quantizer and the range coding in a loop. After each pass it counts the bits:

- With VBR, a first pass within the limit ends the loop.
- Otherwise, while the frame is over its limit, or under CBR more than 5 bits below it, the loop scales the gains, quantizes them again and codes the frame again.
- Once it has one pass above the limit and one below, it interpolates the gain scale between them.
- It stops after at most seven passes. When a pass fitted within the limit, the frame is coded as such a pass. If no pass fitted, the frame is coded with no pulses at all.

The range coding writes the parameters (`silk_encode_indices`: signal type, gains, NLSF indices, pitch lags and contour, LTP taps, seed) and then the pulses (`silk_encode_pulses`). The pulses are coded in blocks of 16 samples, in three parts:

1. the sum of magnitudes of each block;
2. how the sum splits between halves of the block (the shell coder, `shell_coder.rs`);
3. the low bits and the signs.

### 2.12 Complexity in SILK

`silk_setup_complexity` (`src/silk/control_codec.rs`) sets the analysis by complexity:

| Complexity | Pitch search | Pitch LPC order | Shaping LPC order | Quantizer paths | NLSF interpolation | NLSF survivors | Warped shaping |
|---|---|---:|---:|---:|---|---:|---|
| 0 | low | 6 | 12 | 1 | no | 2 | no |
| 1 | medium | 8 | 14 | 1 | no | 3 | no |
| 2 | low | 6 | 12 | 2 | no | 2 | no |
| 3 | medium | 8 | 14 | 2 | no | 4 | no |
| 4-5 | medium | 10 | 16 | 2 | yes | 6 | yes |
| 6-7 | medium | 12 | 20 | 3 | yes | 8 | yes |
| 8-10 | high | 16 | 24 | 4 | yes | 16 | yes |

The pitch LPC order is never above the predictor order (10 at 8 and 12 kHz).

## 3. The CELT encoder

CELT codes a frame in the frequency domain. It splits the spectrum into 21 bands. It codes the energy of each band, and then the shape of the spectrum inside each band as a vector of unit length. Because each band's energy is coded explicitly, the spectral envelope (the overall shape of the spectrum) is kept even at low bitrates. CELT works at 48 kHz in floating point. The encoder is `CeltEncoder` in `src/celt.rs`; its main function is `encode`. It reads the frame interleaved, as the caller passed it.

The band edges, in Hz (`EBAND_5MS` in `src/modes.rs`, in units of 200 Hz):

```text
band:      0    1    2    3    4    5    6    7    8    9   10   11   12   13   14   15   16   17   18    19    20
start Hz:  0  200  400  600  800 1000 1200 1400 1600 2000 2400 2800 3200 4000 4800 5600 6800 8000 9600 12000 15600
```

Band 20 ends at 20 000 Hz. The bands a frame codes depend on its bandwidth:

- narrowband: bands 0 to 12 (up to 4000 Hz);
- wideband: bands 0 to 16 (up to 8000 Hz);
- super-wideband: bands 0 to 18 (up to 12 000 Hz);
- fullband: all 21 bands.

The CELT layer of a Hybrid frame starts at band 17 (8000 Hz).

The table lists the stages in the order `encode` runs them.

| Stage | What it does | File | Main function |
|---|---|---|---|
| Pre-emphasis | input scaled to 16-bit range, filter `x[i] - 0.85 * x[i-1]`; zeros stuffed for input below 48 kHz | `src/celt.rs` | `preemphasis_input`, `celt_preemphasis` |
| Silence | flag for a frame whose peak is at most 2^-24 of full scale; such a frame codes nothing else | `src/celt.rs` | in `encode` |
| Tone and transient analysis | finds a pure tone, and attacks that need short blocks | `src/celt.rs` | `tone_detect`, `transient_passes`, `transient_decision` |
| Pitch pre-filter | comb filter that weakens the pitch harmonics before coding | `src/celt.rs`, `src/pitch.rs` | `run_prefilter`, `pitch_search`, `remove_doubling` |
| Modified discrete cosine transform (MDCT) | to the frequency domain: one long block or 2 to 8 short blocks | `src/mdct.rs`, `src/kiss_fft.rs` | `compute_mdcts`, `MdctLookup::forward` |
| Band energies | energy of each band; normalized band shapes | `src/bands.rs` | `compute_band_energies`, `normalise_bands` |
| Dynamic allocation analysis | bands that get extra bits | `src/celt.rs` | `dynalloc_analysis` |
| Time-frequency resolution | per band, more time or more frequency resolution | `src/celt.rs` | `tf_analysis`, `tf_encode` |
| Coarse energy | band energies to 6 dB steps, predicted from the last frame | `src/quant_bands.rs` | `quant_coarse_energy_advanced` |
| Stereo decisions | dual stereo or mid/side; intensity band | `src/celt.rs` | `stereo_analysis`, `hysteresis_decision` |
| Spreading | how much to spread the pulses in each band | `src/bands.rs` | `spreading_decision` |
| Boosts, trim, VBR target | coded band boosts, allocation trim, packet size | `src/celt.rs` | `alloc_trim_analysis`, `compute_vbr` |
| Bit allocation | bits for each band's shape and fine energy | `src/rate.rs` | `clt_compute_allocation` |
| Fine energy | more precision for the band energies | `src/quant_bands.rs` | `quant_fine_energy` |
| Band shapes and stereo | pyramid vector quantization of each band, mid/side angle | `src/bands.rs`, `src/pvq.rs` | `quant_all_bands_enc`, `quant_band_stereo`, `alg_quant` |
| Anti-collapse | 1 bit for transient frames | `src/celt.rs` | in `encode` |
| Energy finalise | leftover bits refine the energies | `src/quant_bands.rs` | `quant_energy_finalise` |

All symbols go through `RangeCoder` (`src/range_coder.rs`). They are written in the order RFC 6716 section 4.3 gives: silence, pre-filter, transient, coarse energy, time-frequency resolution, spreading, band boosts, allocation trim, allocation flags, fine energy, band shapes, anti-collapse, final energy bits.

### 3.1 Pre-emphasis and frame layout

CELT always works at 48 kHz. Input at a lower API rate is "zero-stuffed": each sample is followed by zeros up to the 48 kHz rate. The spectrum above the input's Nyquist frequency (half its sampling rate) is set to zero after the MDCT (`upsample_scale_zero`). The pre-emphasis filter raises the high frequencies. The decoder's de-emphasis lowers them again, and the coding noise with them. So the coding noise is quieter at high frequencies.

A frame of 2.5, 5, 10 or 20 ms is 120, 240, 480 or 960 samples at 48 kHz. The code calls the size index `lm` (0 to 3; the frame is `120 << lm` samples). Consecutive frames overlap by 120 samples (2.5 ms), using the window `WINDOW_120`.

### 3.2 Transient analysis and short blocks

`transient_passes` and `transient_decision` look for a sudden rise in energy within the frame (complexity 1 and above, not for LFE). A transient frame is coded with short blocks: `1 << lm` MDCTs of 120 samples instead of one long MDCT. This keeps the coding noise from spreading in time before the attack ("pre-echo"). From complexity 5, a frame whose band energies jump from the last frame is also switched to short blocks (`patch_transient_decision`).

### 3.3 Pitch pre-filter

The pre-filter is a comb filter. It reduces the energy at the pitch harmonics before coding, and the decoder's post-filter adds it back. That way the coding noise between the harmonics is less audible. `run_prefilter` searches the period (15 to 1022 samples) with `pitch_downsample`, `pitch_search` and `remove_doubling` in `src/pitch.rs`. The search runs from complexity 5, or at any complexity for a pure tone. It does not run for Hybrid, for silent frames, or for packets with 12 bytes per channel or fewer. The packet carries the period, a 3-bit gain (`0.09375 * (q + 1)`) and one of three tap shapes ("tapset"). The filter fades from the last frame's settings to the new ones over the 2.5 ms overlap.

### 3.4 MDCT and band energies

The modified discrete cosine transform (MDCT, `src/mdct.rs`) turns the overlapped, windowed frame into as many frequency values as the frame has samples. It is computed with a complex fast Fourier transform (FFT, `src/kiss_fft.rs`). A stereo input coded as mono uses half the sum of the two spectra.

`compute_band_energies` takes the square root of the energy in each band. `normalise_bands` divides each band by its energy, so that every band becomes a vector of length 1: its "shape". Energies are coded in the log2 domain (`amp2log2_enc`), so that one unit is 6 dB.

### 3.5 Energy quantization

The coarse energy (`quant_coarse_energy_advanced`, `src/quant_bands.rs`) codes each band's log energy in steps of 6 dB. Each band is predicted from the same band in the last frame and from the bands below it in this frame. The prediction error is coded with a Laplace distribution (`RangeCoder::laplace_encode`). An "intra" frame predicts only from the bands below it, so that it does not depend on the last frame. From complexity 4 the encoder codes both ways and keeps the better one, weighing the packet loss. The fine energy (`quant_fine_energy`) adds more bits of precision where the allocation gives them. Bits left at the end refine the energies once more (`quant_energy_finalise`).

### 3.6 Time-frequency resolution

Each band can trade frequency resolution for time resolution, or the other way round (`tf_res`, coded by `tf_encode`). In a transient frame a band can join its short blocks for better frequency resolution. In a long-block frame a band can be split for better time resolution. The change is made with Haar transforms on the band (`haar1` in `src/bands.rs`). `tf_analysis` chooses per band from complexity 2, for frames with at least 15 bytes per channel. It does not run for Hybrid, LFE or pure tones.

### 3.7 Bit allocation

`clt_compute_allocation` (`src/rate.rs`) splits the bits of the frame between the bands. It interpolates between rows of a fixed table of allocation vectors (`BAND_ALLOCATION` in `src/modes.rs`) until the total fits the bits that are left. The encoder can move the result:

- **Band boosts** (`dynalloc_analysis`): bands that stand out from their neighbours, or that the last frame coded much lower, get extra bits.
- **Allocation trim** (`alloc_trim_analysis`): tilts the allocation towards low or high bands, from 0 to 10, with 5 as neutral. Hybrid and LFE keep 5.
- **Band skipping**: the allocator may stop coding the top bands when it has too few bits. It keeps bands that earlier frames coded with a lower threshold, so that bands do not turn on and off from frame to frame.

The allocation gives each band a number of bits for its shape and a number of fine energy bits. Encoder and decoder run the same function, so they agree without sending the allocation.

Under VBR, `compute_vbr` sets the size of this frame around the target before the allocation runs. It gives more bits to transient frames, to tonal frames, and to frames louder than the running average ("temporal VBR"). Constrained VBR keeps a reservoir so that the average stays at the bitrate.

### 3.8 Band shapes: pyramid vector quantization

Each band's shape is a unit vector. CELT approximates it with an integer vector whose absolute values add up to a number of pulses `K`; this is pyramid vector quantization (PVQ). More bits give more pulses and a closer approximation. `quant_all_bands_enc` (`src/bands.rs`) codes the bands from low to high:

1. The bits of the band give `K` (`bits2pulses` in `src/rate.rs`).
2. A band with more bits than one codeword can carry is split in halves, recursively. The split codes the share of energy between the halves as an angle (`quant_partition`, `compute_theta`).
3. `alg_quant` (`src/pvq.rs`) rotates the vector to spread the pulses (`exp_rotation`, by the spreading decision). It then finds the best integer vector (`pvq_search`). Last, it codes the index of that vector in the codebook of all such vectors (`encode_pulses`, `icwrs`).
4. A band with no bits is filled from the shapes of lower bands ("folding") or with noise.

### 3.9 Stereo coding

With two coded channels, each band is coded in one of three ways:

- **Mid/side**: the band is coded as mid and side, and an angle `theta` codes how the energy splits between them (`quant_band_stereo`, `compute_theta`). This is the usual case.
- **Intensity stereo**: bands from the intensity band upwards code one shape for both channels. When there are bits for it, one more bit says whether the second channel is inverted. Each channel keeps its own band energy. The intensity band moves with the equivalent rate, with hysteresis (`INTEN_THRESHOLDS`, `INTEN_HYSTERESIS`).
- **Dual stereo**: left and right are coded as two independent channels. `stereo_analysis` chooses it for frames where it estimates that this costs fewer bits. 2.5 ms frames never use dual stereo.

At complexity 8 and above, each mid/side band below the intensity band is coded twice, with the angle rounded down and up, and the better one is kept (`theta_rdo`).

### 3.10 Anti-collapse

In a transient frame, a band of a short block can end up with no pulses at all. The decoder would then output silence in that block, which sounds like a gap ("collapse"). For transient frames of 10 and 20 ms the encoder reserves one bit. When the bit is set, the decoder fills empty short blocks with noise at a level taken from the previous frames (`anti_collapse` in `src/bands.rs`). The encoder sets it unless two transient frames came just before.

### 3.11 Complexity in CELT

| Complexity | What is enabled |
|---|---|
| 0 | no spreading (`SPREAD_NONE`) |
| 1 and above | transient analysis; normal spreading |
| 2 and above | time-frequency analysis |
| 3 and above | spreading decision from the signal |
| 4 and above | coarse energy coded intra and inter, the better one kept |
| 5 and above | pitch pre-filter search; late transient detection |
| 8 and above | a second, long-block MDCT on transient frames; `theta_rdo` for stereo |

## 4. Settings and what they change

All settings take effect with the next call of `encode`. Each one is a method of `OpusEncoder`, and most have a getter of the same name without `set_`. A setter checks its value as the libopus control does. One that can refuse a value returns a `Result`, and after an error the setting keeps its old value.

| Setting | libopus control | Default | What it changes |
|---|---|---|---|
| `set_bitrate(Bitrate)` | `OPUS_SET_BITRATE` | `Bitrate::Auto` (60 × rate / frame size + rate × channels b/s) | The target bitrate: `Bitrate::BitsPerSecond(n)`, `Bitrate::Auto` (`OPUS_AUTO`) or `Bitrate::Max` (`OPUS_BITRATE_MAX`). `n` of 0 or less is an error; other values are raised to at least 500 b/s and limited to at most 750 000 b/s per channel. It drives the channel, mode, bandwidth and FEC decisions and the SILK and CELT rates. The output buffer can lower it (1.2). `bitrate_setting()` returns the setting, `bitrate()` the rate in b/s (`OPUS_GET_BITRATE`). |
| `set_vbr(bool)` | `OPUS_SET_VBR` | `true` | CBR (`false`): every packet has the same size and is padded to it. In Hybrid, SILK still runs VBR with a cap and CELT fills the rest. VBR (`true`): see the next row. |
| `set_vbr_constraint(bool)` | `OPUS_SET_VBR_CONSTRAINT` | `true` | With VBR, CELT-only packets use constrained VBR, which stays close to the bitrate; `false` makes it unconstrained. Hybrid packets always use unconstrained VBR, as in libopus. |
| `set_complexity(i32)` | `OPUS_SET_COMPLEXITY` | 9 | 0..=10; other values are an error. More analysis for better quality at the same bitrate (tables 2.12 and 3.11). It also raises the equivalent rate. |
| `Application` (argument of `OpusEncoder::new`) | `application` of `opus_encoder_create`, `OPUS_SET_APPLICATION` | none | `Voip`: voice estimate 115, CELT threshold +8000 b/s, high-pass filter with a moving cutoff, SILK input from the integer filter. `Audio`: voice estimate 48, a 3 Hz filter that removes the constant (direct current, DC) offset. `RestrictedLowDelay`: always CELT-only, no 4 ms look-ahead. It cannot change after construction. |
| `set_force_mode(Option<ForcedMode>)` | `OPUS_SET_FORCE_MODE` (in libopus's private header `opus_private.h`) | `None` | Asks for SILK-only, Hybrid or CELT-only. The rules that keep a packet valid still apply (1.3): frames under 10 ms are CELT-only, SILK above wideband becomes Hybrid, Hybrid at wideband or below becomes SILK-only. |
| `set_bandwidth(Bandwidth)` | `OPUS_SET_BANDWIDTH` | `Bandwidth::Auto` | Codes this bandwidth instead of the automatic one. It never exceeds half the API rate. CELT codes mediumband as wideband. A running SILK encoder moves to it one step at a time. |
| `set_max_bandwidth(Bandwidth)` | `OPUS_SET_MAX_BANDWIDTH` | `Bandwidth::Fullband` | The widest bandwidth the automatic choice may take. `Bandwidth::Auto` is an error. |
| `set_force_channels(Option<usize>)` | `OPUS_SET_FORCE_CHANNELS` | `None` | Codes 1 or 2 channels instead of the automatic choice. More than the input has is an error. A 40 or 60 ms CELT or Hybrid packet that changes to mono sets it to `Some(1)` itself, as libopus does. |
| `set_inband_fec(bool)` | `OPUS_SET_INBAND_FEC` | `false` | Allows LBRR copies in SILK and Hybrid packets (1.3, 2.9). With packet loss above the voice-dependent limit it also makes the encoder prefer SILK. |
| `set_packet_loss_perc(i32)` | `OPUS_SET_PACKET_LOSS_PERC` | 0 | The expected packet loss, 0..=100; other values are an error. FEC needs it above 0. It lowers the equivalent rate, raises the LTP scaling, sets the LBRR gain step, and makes CELT code intra energy more often. |
| `set_lfe(bool)` | `OPUS_SET_LFE` (in libopus's `celt/celt.h`) | `false` | The encoder codes a low-frequency-effects channel of a surround stream: CELT-only, narrowband, only bands 0 and 1 matter. |
| `encode_i16` (method) | `opus_encode` (16-bit input) | | Encodes 16-bit samples. The packets are the same bytes as `encode` with `s as f32 / 32768.0`. |
| `set_prediction_disabled(bool)` | `OPUS_SET_PREDICTION_DISABLED` | `false` | Every frame can be decoded on its own: SILK codes each packet as the first after a reset, CELT turns off the pre-filter and inter-frame energy prediction. Costs quality. |
| `set_energy_mask(Option<&[f32]>)` | `OPUS_SET_ENERGY_MASK` (in `celt/celt.h`) | `None` | Used by the multistream encoder for surround ([05-multichannel.md](05-multichannel.md)): how much each band is masked by other channels. Lowers the bits of masked bands. |
| `reset_state()` | `OPUS_RESET_STATE` | | Forgets the signal coded so far. The settings stay. |
| `final_range()` | `OPUS_GET_FINAL_RANGE` | | The range coder state after the last packet, for checking a decoder (see [04-decoder.md](04-decoder.md)). |
| `lookahead()` | `OPUS_GET_LOOKAHEAD` | | The encoder delay in samples at the API rate. |

This example sets several controls at once and checks the packet they give. The output is a mono SILK wideband packet of exactly 80 bytes: 32 kilobits per second (kb/s) for 20 ms is 640 bits.

```rust
use opus1::{Application, Bandwidth, Bitrate, ForcedMode, OpusEncoder};

let mut encoder = OpusEncoder::new(48_000, 2, Application::Audio).unwrap();
encoder.set_bitrate(Bitrate::BitsPerSecond(32_000)).unwrap(); // OPUS_SET_BITRATE
encoder.set_vbr(false); // OPUS_SET_VBR(0)
encoder.set_complexity(5).unwrap(); // OPUS_SET_COMPLEXITY
encoder.set_inband_fec(true); // OPUS_SET_INBAND_FEC(1)
encoder.set_packet_loss_perc(10).unwrap(); // OPUS_SET_PACKET_LOSS_PERC
encoder.set_force_mode(Some(ForcedMode::SilkOnly)); // OPUS_SET_FORCE_MODE
encoder.set_bandwidth(Bandwidth::Wideband); // OPUS_SET_BANDWIDTH
encoder.set_force_channels(Some(1)).unwrap(); // OPUS_SET_FORCE_CHANNELS

let pcm = vec![0.0f32; 960 * 2]; // 20 ms of stereo silence at 48 kHz
let mut packet = vec![0u8; 1276];
let len = encoder.encode(&pcm, 960, &mut packet).unwrap();
assert_eq!(len, 80); // CBR pads the packet to its size
assert_eq!(packet[0] >> 3, 9); // configuration 9: SILK-only, wideband, 20 ms
assert_eq!(packet[0] & 0x04, 0); // the stereo bit is clear: one coded channel
```

This example shows that the two entry points give the same packets:

```rust
use opus1::{Application, Bitrate, OpusEncoder};

let pcm16: Vec<i16> = (0..320).map(|i| ((i as f32 * 0.07).sin() * 8000.0) as i16).collect();
let pcm32: Vec<f32> = pcm16.iter().map(|&s| s as f32 / 32768.0).collect();

let mut a = OpusEncoder::new(16_000, 1, Application::Voip).unwrap();
let mut b = OpusEncoder::new(16_000, 1, Application::Voip).unwrap();
a.set_bitrate(Bitrate::BitsPerSecond(20_000)).unwrap();
b.set_bitrate(Bitrate::BitsPerSecond(20_000)).unwrap();
let (mut packet_a, mut packet_b) = (vec![0u8; 400], vec![0u8; 400]);
let len_a = a.encode_i16(&pcm16, 320, &mut packet_a).unwrap();
let len_b = b.encode(&pcm32, 320, &mut packet_b).unwrap();
assert_eq!(packet_a[..len_a], packet_b[..len_b]);
```

## 5. Known differences from libopus

The encoder follows libopus 1.6 with these differences. The doc comment of `encode_impl` in `src/lib.rs` lists them with their reasons. [10-differences-from-libopus.md](10-differences-from-libopus.md) collects the differences of the whole library.

1. No speech/music analysis (`analysis.c`). The voice estimate is fixed per application. The bandwidth that analysis detects is not applied. CELT gets no analysis input. libopus runs the analysis at complexity 7 and above for 16 to 48 kHz input, so packets differ most there.
2. No discontinuous transmission (DTX): the encoder sends a packet for every frame, also in silence.
3. With `Voip`, SILK reads the output of the integer high-pass filter of the fixed-point build. With the other applications SILK follows the float build.
4. 40 and 60 ms CELT and Hybrid VBR packets have no per-frame limit (1.5).
5. The CELT bitrate of a Hybrid packet is at least 501 b/s.
6. 16-bit input uses the silence threshold of float input, so both entry points give the same packets.
7. No 80, 100 or 120 ms frames, and no controls for signal type or phase inversion.
8. A frame that announces the switch to CELT-only without room for the redundant frame still resets the CELT encoder.
9. The energy mask is copied, not kept as a pointer.
