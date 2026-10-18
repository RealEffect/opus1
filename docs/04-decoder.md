# The decoder

This document follows one packet through `OpusDecoder::decode` to the output samples. It covers how the packet is parsed, and how each frame is decoded by the SILK layer, the CELT (Constrained Energy Lapped Transform) layer or both. It also covers how the decoder joins frames of different modes. It then covers what the decoder does for a lost packet: concealment with the SILK and CELT methods, and recovery from the in-band forward error correction (FEC) data of the next packet. The last sections cover float and 16-bit output and the final range. The code follows `opus_decoder.c`, `silk/dec_API.c` and `celt/celt_decoder.c` of libopus 1.6. Comments in the code cite those files with line numbers ("C 581" means line 581 of the C file named nearby). The encoder is described in [03-encoder.md](03-encoder.md). The list of all documents is in [README.md](README.md).

## Terms used here

- **Output rate**: the sampling rate the caller asks for in `OpusDecoder::new`: 8, 12, 16, 24 or 48 kilohertz (kHz). It does not have to match the rate the stream was encoded at.
- **Packet**: one Opus packet. It starts with a table-of-contents (TOC) byte, followed by one or more frames (section 3 of RFC 6716, the Request for Comments that defines Opus).
- **Frame**: a stretch of audio coded as a unit: 2.5, 5, 10, 20, 40 or 60 milliseconds (ms). All frames of a packet have the same mode, bandwidth and duration.
- **Mode**: which coder coded the frame. SILK-only uses the SILK speech coder. CELT-only uses the CELT transform coder. Hybrid codes below 8 kHz with SILK and above 8 kHz with CELT.
- **Audio bandwidth**: the highest audio frequency the packet carries. Narrowband (NB) is 4 kHz, mediumband (MB) 6 kHz, wideband (WB) 8 kHz, super-wideband (SWB) 12 kHz and fullband (FB) 20 kHz.
- **Internal rate**: the rate SILK decodes at: 8 kHz (NB), 12 kHz (MB) or 16 kHz (WB and Hybrid).
- **Range decoder**: the entropy decoder of Opus (RFC 6716 section 4.1, `RangeCoder` in `src/range_coder.rs`). SILK and CELT read their symbols from one range-coded stream per frame.
- **Packet loss concealment (PLC)**: making up audio for a lost packet from the audio before it.
- **In-band FEC**: a low-bitrate copy of a SILK frame that the next packet carries. libopus calls it low bitrate redundancy (LBRR).
- **Redundant frame**: a 5 ms CELT frame that a SILK-only or Hybrid packet can carry next to a change of mode.
- **Modified discrete cosine transform (MDCT)**: the transform CELT codes in. The decoder runs the inverse MDCT.
- **Pyramid vector quantization (PVQ)**: how CELT codes the shape of the spectrum in a band.

## Data flow

```text
packet bytes (or an empty slice for a lost packet)
  |
  v
decode / decode_fec          decode_i16 / decode_fec_i16
  |                                  |
  |                                  v
  |                          decode_i16_native (float buffer, soft clip on)
  v                                  |
decode_native <----------------------+
  |  empty packet ------------------------------> conceal(frame_size)
  |  decode_fec ---> FEC copy in next packet? ---> decode_frame(FEC) or conceal
  |  TOC byte: mode, bandwidth, frame duration, channels
  |  packet::parse: frame count and sizes
  v
for each frame: decode_or_conceal_frame
  |  frame of 0 or 1 bytes ---------------------> conceal
  v
decode_frame
  |  SILK-only, Hybrid: decode_silk -> SilkDecoder::decode_with_control
  |                     (SILK frames at 8/12/16 kHz, mid/side to left/right,
  |                      resampled to the output rate), written into the output
  |  redundant frame flag and size
  |  Hybrid, CELT-only: CeltDecoder::decode (bands from 17 or 0 up to the TOC
  |                     bandwidth, inverse MDCT at 48 kHz, post-filter,
  |                     de-emphasis, keep 1 sample in 48000/output rate),
  |                     written into the output (CELT-only) or added to it (Hybrid)
  |  change of mode: cross-fade with the redundant frame or with 5 ms of PLC
  v
float samples, interleaved (not clipped)
  |
  +--> decode_i16, decode_fec_i16: soft clip (pcm_soft_clip) of a decoded
       packet, round to 16 bits
```

## 1. Entry points

All decoding goes through `decode_native` in `src/lib.rs`, the port of libopus `opus_decode_native`. The 16-bit methods call it through `decode_i16_native`, which decodes into a float buffer and turns on the soft clipper (section 10).

| Method | libopus function | What it does |
|---|---|---|
| `decode(data, frame_size, out)` | `opus_decode_float` | Decodes a packet to `f32`. An empty `data` conceals `frame_size` samples. |
| `decode_i16(data, frame_size, out)` | `opus_decode` | As `decode`, then soft clipping and rounding to 16 bits. |
| `decode_fec(data, frame_size, out)` | `opus_decode_float` with `decode_fec = 1` | Rebuilds the packet before `data` from the FEC copy in `data`. |
| `decode_fec_i16(data, frame_size, out)` | `opus_decode` with `decode_fec = 1` | As `decode_fec`, with 16-bit output. |
| `final_range()` | `OPUS_GET_FINAL_RANGE` | The range decoder state after the last frame. |
| `last_packet_duration()` | `OPUS_GET_LAST_PACKET_DURATION` | The samples per channel of the last successful decode, concealment or FEC call. 0 at the start and after `reset_state`. |
| `reset_state()` | `OPUS_RESET_STATE` | Forgets the stream decoded so far. |

All of them return the number of samples per channel written. The output is interleaved: left, right, left, right for two channels.

For a packet that arrived, `frame_size` is the room in `out` in samples per channel, and the packet decides how many samples come out. A packet longer than `frame_size` is the error "frame_size too small for packet". `decode` writes zeros after the decoded samples. To decode any packet, pass room for 120 ms: 5760 samples per channel at 48 kHz.

For a lost packet and for `decode_fec`, `frame_size` is the duration to produce. It must be a multiple of 2.5 ms.

## 2. Packet parsing

[02-packets.md](02-packets.md) describes the packet format in full. In short, the TOC byte has three fields:

```text
bit:   7  6  5  4  3 | 2 | 1  0
       configuration | s | frame count code
```

The configuration number (0 to 31) names the mode, the bandwidth and the frame duration (RFC 6716 section 3.1):

| Configuration | Mode | Bandwidth | Frame durations |
|---|---|---|---|
| 0-3 | SILK-only | NB | 10, 20, 40, 60 ms |
| 4-7 | SILK-only | MB | 10, 20, 40, 60 ms |
| 8-11 | SILK-only | WB | 10, 20, 40, 60 ms |
| 12-13 | Hybrid | SWB | 10, 20 ms |
| 14-15 | Hybrid | FB | 10, 20 ms |
| 16-19 | CELT-only | NB | 2.5, 5, 10, 20 ms |
| 20-23 | CELT-only | WB | 2.5, 5, 10, 20 ms |
| 24-27 | CELT-only | SWB | 2.5, 5, 10, 20 ms |
| 28-31 | CELT-only | FB | 2.5, 5, 10, 20 ms |

Bit `s` is set for a stereo packet. The frame count code says how the frames follow:

| Code | Frames |
|---|---|
| 0 | one frame |
| 1 | two frames of the same size |
| 2 | two frames; the size of the first is coded |
| 3 | 1 to 48 frames; a count byte follows, with optional padding and sizes |

`decode_native` reads the mode (`mode_from_toc`), the bandwidth (`bandwidth_from_toc`), the frame duration (`packet::samples_per_frame`) and the channel count (`channels_from_toc`) from the TOC byte. `packet::parse` (`src/packet.rs`, the port of `opus_packet_parse_impl`) checks the framing and returns the frame count and the size of each frame in a `ParsedPacket`. A packet that does not parse is an error, and the decoder state stays as it was.

The decoder state (mode, bandwidth, frame duration, channel count of the last packet) is updated only after these checks.

## 3. The per-frame loop

Each frame of the packet goes through `decode_or_conceal_frame`, the port of libopus `opus_decode_frame`:

- A frame of 0 or 1 bytes carries no audio. It is concealed like a lost frame (libopus: "Payloads of 1 (2 including ToC) or 0 trigger the PLC/DTX"; DTX is discontinuous transmission, where the encoder sends such frames during silence).
- Any other frame goes to `decode_frame`.

`decode_frame` decodes one frame in these steps:

1. **Transition check.** A switch to or from CELT-only needs audio in the old mode to cross-fade from, unless the last frame ended with a redundant frame for it (section 6). For a switch into CELT-only, 5 ms are concealed in the old mode first.
2. **SILK layer** (SILK-only and Hybrid): `decode_silk` writes the SILK signal into the frame's part of the output. After CELT-only frames the SILK decoder is reset first.
3. **Redundant frame**: its flag, position and size (section 6).
4. **CELT layer** (Hybrid and CELT-only). The CELT decoder is reset when the mode changed and the last frame had no redundant frame. CELT-only output is the CELT signal, written into the output (`PcmLayout::Interleaved`). In a Hybrid frame CELT adds its signal to the SILK signal already there (`PcmLayout::InterleavedAdd`).
5. **Fade-out of CELT**: after a Hybrid frame, a SILK-only frame gets 2.5 ms of CELT added, so that the CELT layer fades out (section 6).
6. **Cross-fades** with the redundant frame or with the concealed audio of step 1.
7. **Final range** and the state for the next frame (`prev_mode`, `prev_redundancy`).

## 4. SILK decoding

`decode_silk` (`src/lib.rs`) calls `SilkDecoder::decode_with_control` (`src/silk/dec_api.rs`, the port of `silk_Decode`) once per SILK frame. A SILK frame is 10 or 20 ms. A 40 or 60 ms Opus frame holds 2 or 3 SILK frames in one range-coded stream; only the first call is told that a new packet starts. The call gets a `SilkDecControl` with the output rate, the output channel count, the coded channel count, the internal rate and the frame duration.

For the first SILK frame of a packet, `decode_with_control` reads the packet header. The header holds one voice activity flag per frame and one LBRR flag for each coded channel. It then says which frames have an LBRR copy. In normal decoding it skips the LBRR data. Then each SILK frame goes through the steps below. `decode_with_control` reads the stereo header. The other steps run in `silk_decode_frame` (`src/silk/decode_frame.rs`), once for each coded channel.

| Step | What it does | File | Function |
|---|---|---|---|
| Stereo header | predictor from mid to side, and whether only the mid is coded | `src/silk/decode_indices.rs` | `silk_stereo_decode_pred`, `silk_stereo_decode_mid_only` |
| Parameters | signal type, gains, normalized line spectral frequency (NLSF) indices, pitch lags, long-term prediction (LTP) taps, seed | `src/silk/decode_indices.rs` | `silk_decode_indices` |
| Pulses | the excitation pulses (shell decoder, low bits, signs) | `src/silk/decode_pulses.rs` | `silk_decode_pulses` |
| Dequantization | gains; NLSFs to predictor coefficients, interpolated in the first half; pitch lags; LTP taps | `src/silk/decode_parameters.rs` | `silk_decode_parameters` |
| Synthesis | excitation from pulses, offset and random signs; long-term (pitch) synthesis; short-term synthesis with the linear predictive coding (LPC) filter; gain | `src/silk/decode_core.rs` | `silk_decode_core` |
| PLC state | remembers pitch, predictor and gains for a later loss | `src/silk/plc.rs` | `silk_plc` |
| Comfort noise | tracks the background noise; adds it to concealed frames | `src/silk/cng.rs` | `silk_cng` |
| Joining | smooths the first good frame after a loss | `src/silk/plc.rs` | `silk_plc_glue_frames` |

Then `decode_with_control`:

1. converts mid and side to left and right (`silk_stereo_ms_to_lr`, `src/silk/stereo_ms_to_lr.rs`) when the packet and the output are both stereo,
2. resamples each channel from the internal rate to the output rate (`SilkResampler::process`, `src/silk/resampler.rs`; `silk_decoder_set_fs` in `src/silk/init_decoder.rs` sets it up when a rate changes),
3. writes 16-bit samples.

`decode_silk` converts them to float as `v / 32768` (`int16_to_res`) and writes them into the output. As in libopus, the SILK decoder works in fixed point.

## 5. CELT decoding

The CELT decoder (`CeltDecoder` in `src/celt.rs`) decodes at 48 kHz. Its main function is `decode`. It writes the frame into the caller's buffer as a `PcmLayout` says. Before each call `decode_frame` sets the band range of the packet with `set_start_band` and `set_end_band`: from band 0 (CELT-only) or band 17 (Hybrid, 8 kHz) up to the band the TOC bandwidth ends at:

| TOC bandwidth | NB | MB, WB | SWB | FB |
|---|---:|---:|---:|---:|
| End band (`Bandwidth::celt_end_band`) | 13 | 17 | 19 | 21 |
| Highest frequency | 4 kHz | 8 kHz | 12 kHz | 20 kHz |

`decode` reads the symbols in the order the encoder wrote them:

| Step | What it reads or does | File | Function |
|---|---|---|---|
| Silence | a silent frame codes nothing else | `src/celt.rs` | in `decode` |
| Post-filter | pitch period, gain, tap shape (not in Hybrid) | `src/celt.rs` | in `decode` |
| Transient | long block or short blocks | `src/celt.rs` | in `decode` |
| Coarse energy | band energies in steps of 6 decibels (dB), predicted from the last frame (or not, for an "intra" frame) | `src/quant_bands.rs` | `unquant_coarse_energy` |
| Time-frequency resolution | per band | `src/celt.rs` | `tf_decode` |
| Spreading, band boosts, allocation trim | | `src/celt.rs` | in `decode` |
| Bit allocation | the same computation as the encoder | `src/rate.rs` | `clt_compute_allocation` |
| Fine energy | more precision for the energies | `src/quant_bands.rs` | `unquant_fine_energy` |
| Band shapes | PVQ codewords, mid/side angles, folding | `src/bands.rs`, `src/pvq.rs` | `quant_all_bands`, `alg_unquant` |
| Anti-collapse, final energy bits | | `src/bands.rs`, `src/quant_bands.rs` | `anti_collapse`, `unquant_energy_finalise` |
| Synthesis | shapes times energies; inverse MDCT; overlap-add with the last frame | `src/celt.rs`, `src/bands.rs`, `src/mdct.rs` | `synthesis`, `denormalise_bands`, `MdctLookup::backward` |
| Post-filter | comb filter that restores the pitch harmonics | `src/celt.rs` | `postfilter`, `comb_filter_inplace` |
| De-emphasis | the inverse of the encoder's pre-emphasis; scaled back to ±1; keeps 1 sample in `48000 / output rate`; writes or adds the samples as the `PcmLayout` says | `src/celt.rs` | `deemphasis` |

Some details:

- **Allocation.** Encoder and decoder run the same `clt_compute_allocation` on the same inputs, so the allocation itself is not sent. Only the band boosts, the trim and a few flags are.
- **Output rates below 48 kHz.** `synthesis` sets every MDCT value above the output's Nyquist frequency (half the output rate) to zero before the inverse MDCT. De-emphasis then keeps one sample in 2, 3, 4 or 6 without another filter.
- **Post-filter.** The filter changes smoothly from frame to frame. In a frame of 5 ms or more, the first 2.5 ms use the last frame's filter, the next 2.5 ms fade to this frame's filter, and the rest uses this frame's filter. A 2.5 ms frame fades between the filters of the two frames before it; its own filter takes effect in the next frame.
- **No saturation.** libopus saturates CELT samples only in its fixed-point build. The float build, and this crate, keep values beyond ±1.

## 6. Mode switches and redundant frames

The encoder adds a 5 ms CELT frame (the redundant frame) to a SILK-only or Hybrid packet next to a switch to or from CELT-only. The decoder finds it after the SILK data:

- **Hybrid**: when at least 37 bits of the frame are left after the SILK data, a flag (probability 1/4096) says whether a redundant frame follows. Then 1 bit gives its position, and `dec_uint(256) + 2` its size in bytes.
- **SILK-only**: when at least 17 bits of the frame are left after the SILK data, a redundant frame follows. 1 bit gives its position. Its size is the rest of the frame.

The redundant frame sits at the end of the frame's bytes. The main frame's CELT layer stops before it (`rc.storage -= redundancy_bytes`). A size that does not fit the frame is invalid. The decoder then ignores the redundant frame and the rest of the frame's bytes.

| Switch | What the decoder does |
|---|---|
| SILK or Hybrid to CELT-only, redundant frame at the end of the last SILK/Hybrid frame | Resets the CELT decoder and decodes the redundant frame. The last 2.5 ms of the frame fade into the second half of the redundant frame. The next CELT-only frame continues from the redundant frame's CELT state, without a reset. |
| CELT-only to SILK or Hybrid, redundant frame in the first SILK/Hybrid frame | Decodes the redundant frame first, continuing the last CELT frame. Its first 2.5 ms replace the start of the frame; the next 2.5 ms fade from it into the frame. The redundant frame is not used when the frame before was SILK-only without a redundant frame: then the first frame of the switch was lost. |
| Either direction without a redundant frame | Conceals 5 ms in the old mode. The first 2.5 ms of the concealment replace the start of the new frame, and the next 2.5 ms fade into the frame. A 2.5 ms frame fades from the concealment over its whole length. |
| Hybrid to SILK-only | Decodes 2.5 ms of CELT from a frame of two 0xFF bytes and adds it, so that the CELT layer's last window fades out. This is skipped when this frame starts with a redundant frame for a switch from CELT-only and the frame before ended with one for a switch to CELT-only. |
| SILK-only to Hybrid | Nothing extra. The CELT decoder starts from a reset state. |

The cross-fades use the squared CELT window (`smooth_fade`). Each sample is `w * in2 + (1 - w) * in1`, with the first product fused into the sum as the libopus build on 64-bit ARM computes it (`mac`). The redundant frame's final range is combined by exclusive or (XOR) into the frame's final range.

## 7. Packets with another channel count

A packet may code another channel count than the decoder outputs, and the count may change from packet to packet:

- **Mono packet, stereo decoder.** SILK writes the decoded channel to both outputs. On the first mono frame after stereo frames, the right output goes through the right channel's resampler once more. This outputs the last samples that resampler still holds. CELT gives both outputs the same spectrum.
- **Stereo packet, mono decoder.** SILK outputs the mid channel, which is the average of left and right. CELT mixes the two spectra into one (`0.5 * left + 0.5 * right`). It decodes without phase inversion (libopus `disable_inv`), because inverted bands would cancel in the mix.

Before every CELT call the decoder tells the CELT decoder the packet's channel count (`set_stream_channels`, libopus `CELT_SET_CHANNELS`). When the SILK stream goes from mono to stereo, the second SILK channel starts from a reset state.

```rust
use opus1::{Application, OpusDecoder, OpusEncoder};

let mut encoder = OpusEncoder::new(48_000, 1, Application::Audio).unwrap();
let mut decoder = OpusDecoder::new(48_000, 2).unwrap(); // stereo output
let mut packet = vec![0u8; 1276];
let mut stereo = vec![0.0f32; 960 * 2];
for k in 0..5 {
    let pcm: Vec<f32> = (0..960)
        .map(|i| 0.3 * (2.0 * std::f32::consts::PI * 440.0 * (k * 960 + i) as f32 / 48_000.0).sin())
        .collect();
    let len = encoder.encode(&pcm, 960, &mut packet).unwrap();
    assert_eq!(packet[0] & 0x04, 0); // a mono packet
    let n = decoder.decode(&packet[..len], 960, &mut stereo).unwrap();
    assert_eq!(n, 960);
    // Both output channels carry the mono signal.
    for pair in stereo.chunks_exact(2) {
        assert_eq!(pair[0], pair[1]);
    }
}
```

## 8. Packet loss concealment

The caller reports a lost packet by passing an empty slice to `decode` or `decode_i16`, with the duration to conceal as `frame_size`. A packet of only a TOC byte is concealed too, for the duration its TOC names. The concealment runs in the mode of the last frame (`conceal` in `src/lib.rs`). After a frame that ended with a redundant frame for a switch to CELT-only, that mode is CELT-only. Before the first packet the decoder outputs zeros.

The loss is concealed in pieces, as libopus does:

- One piece covers at most one frame of the last packet. A loss after 10 ms packets is concealed 10 ms at a time.
- A piece is at most 20 ms.
- A piece is 2.5, 5, 10 or 20 ms: when what is left lies between two of these, the shorter one is taken. SILK-only also takes 7.5 ms.

Each concealed piece sets the final range to 0.

### 8.1 SILK concealment

A lost SILK frame goes through `silk_decode_frame` with `FLAG_PACKET_LOST`, which calls `silk_plc` (`src/silk/plc.rs`, the port of `silk/PLC.c`):

1. After each good frame, `silk_plc_update` stores the pitch lag and LTP taps of the last voiced subframe, the predictor coefficients and the last two gains.
2. A lost frame is synthesized like a normal frame. Its excitation continues the last frame's excitation through the long-term predictor at the last pitch lag. Random values are added, taken from the excitation of the quieter of the last two subframes.
3. With each lost frame in a row, the pitch part and the random part get quieter (`HARM_ATT_Q15`, `PLC_RAND_ATTENUATE_V_Q15`, `PLC_RAND_ATTENUATE_UV_Q15`). The LPC predictor gets a bandwidth expansion (`BWE_COEF_Q16`), which makes its resonances a little wider. The pitch lag grows slowly.
4. Comfort noise generation (`silk_cng`, `src/silk/cng.rs`) adds noise shaped like the background noise of the good frames.
5. When the first good frame after a loss is louder than the concealment, `silk_plc_glue_frames` fades its gain up from the concealed level.

A SILK frame is 10 ms at least. A shorter piece is taken from the start of 10 ms of concealment.

After a lost SILK frame the decoder sets its last gain index back to the start value 10. Otherwise the first gain of the next good frame could not fall more than 16 steps below the gain before the loss. The level could then jump back up (dec_API.c: "remove the gain clamping").

### 8.2 CELT concealment

A lost CELT frame goes through `CeltDecoder::decode_lost` (`conceal` in `src/celt.rs`, the port of `celt_decode_lost` without the neural concealment of libopus). It uses one of two methods:

- **Pitch repetition** (CELT-only, for the first 100 ms of a loss):
  1. On the first lost frame, it searches the pitch period of the last output (`plc_pitch_search`, 100 to 720 samples at 48 kHz). It also fits a linear predictor of order 24 to the last 1024 output samples. The search and the fit use the functions of the encoder's pre-filter in `src/pitch.rs` and `src/celt_lpc.rs`, which round as libopus does on 64-bit ARM.
  2. It filters the last output with that predictor to get its excitation, and repeats the last pitch period of the excitation.
  3. If the excitation was getting quieter, the repetition keeps getting quieter at the same rate. Each later lost frame is also multiplied by 0.8.
  4. It filters the result back with the predictor, starting from the last decoded samples, so that the signal stays continuous.
  5. If the result has more energy than the signal it copies, it is scaled down to about that energy. If it has more than 5 times that energy, it is replaced by silence.
- **Shaped noise**: random values in every band, at the band energies of the last frame. The energies fall towards the estimated background noise, by 1.5 units of 6 dB on the first lost frame and by 0.5 units on each later one. This method is used in three cases:
  - after 100 ms of loss;
  - for the CELT layer of a Hybrid stream;
  - after a noise-concealed loss, until two frames in a row have been decoded.

The first frame decoded after a loss predicts its band energies from the frames before the loss. The decoder lowers those energies first, so that a wrong prediction cannot cause a loud click.

### 8.3 Hybrid concealment

A lost Hybrid frame is concealed by SILK below 8 kHz and by CELT noise above 8 kHz, and the two are added.

### Example: decoding a stream with lost packets

```rust
use opus1::{Application, OpusDecoder, OpusEncoder};

let rate = 48_000;
let frame = 960; // 20 ms
let mut encoder = OpusEncoder::new(rate, 1, Application::Audio).unwrap();
let mut decoder = OpusDecoder::new(rate, 1).unwrap();

// Encode 10 packets of a 440 Hz tone.
let mut packets = Vec::new();
for k in 0..10 {
    let pcm: Vec<f32> = (0..frame)
        .map(|i| 0.3 * (2.0 * std::f32::consts::PI * 440.0 * (k * frame + i) as f32 / rate as f32).sin())
        .collect();
    let mut packet = vec![0u8; 1276];
    let len = encoder.encode(&pcm, frame, &mut packet).unwrap();
    packet.truncate(len);
    packets.push(packet);
}

// Packets 3 and 4 are lost on the way.
let received: Vec<Option<&[u8]>> = packets
    .iter()
    .enumerate()
    .map(|(i, p)| if i == 3 || i == 4 { None } else { Some(p.as_slice()) })
    .collect();

let mut output = Vec::new();
let mut pcm = vec![0.0f32; frame];
for packet in received {
    let samples = match packet {
        Some(data) => decoder.decode(data, frame, &mut pcm).unwrap(),
        // An empty packet asks for `frame` samples of concealment.
        None => decoder.decode(&[], frame, &mut pcm).unwrap(),
    };
    assert_eq!(samples, frame);
    if packet.is_none() {
        assert_eq!(decoder.final_range(), 0); // nothing was decoded
    }
    output.extend_from_slice(&pcm[..samples]);
}
assert_eq!(output.len(), 10 * frame);
```

## 9. In-band FEC: `decode_fec`

A SILK-only or Hybrid packet can carry a low-bitrate copy (LBRR) of the SILK frames of the packet before it. The encoder adds it when `set_inband_fec(true)` turned FEC on and the packet loss percentage (`set_packet_loss_perc`) is above 0 (see [03-encoder.md](03-encoder.md), section 2.9). When packet `n` is lost and packet `n + 1` has arrived, the receiver can rebuild packet `n`:

1. Call `decode_fec(packet_n_plus_1, frame_size, out)`, with `frame_size` the duration of the lost packet. It writes exactly `frame_size` samples.
2. Then call `decode(packet_n_plus_1, ...)` as usual for packet `n + 1`.

`decode_fec` runs `decode_frame` on the first frame of the packet with the SILK lost flag `FLAG_DECODE_LBRR`. The SILK decoder then reads the LBRR copy instead of the frame. It conceals the SILK frames for which the packet has no copy.

The whole duration is concealed instead when:

- the packet is CELT-only, or the packet before the loss was CELT-only (neither carries FEC),
- `frame_size` is shorter than one frame of the packet.

When `frame_size` is longer than one frame of the packet, the start is concealed and the copy fills the end.

The copy codes SILK only. In a Hybrid stream the band above 8 kHz is concealed with CELT noise and added. No redundant frame is read during FEC decoding.

`packet::has_lbrr(packet_n_plus_1)` tells whether the packet carries a copy at all, and `last_packet_duration()` gives a duration to rebuild: libopus's test program `opus_demo` uses both for the last of the lost packets.

When the next packet has not arrived yet, a receiver conceals with an empty packet instead. Which of the two to use is up to the receiver's jitter buffer (the queue that holds received packets until it is time to play them).

### Example: using FEC when the next packet has arrived

```rust
use opus1::{Application, Bitrate, OpusDecoder, OpusEncoder};

let rate = 16_000;
let frame = 320; // 20 ms
let mut encoder = OpusEncoder::new(rate, 1, Application::Voip).unwrap();
encoder.set_bitrate(Bitrate::BitsPerSecond(24_000)).unwrap(); // SILK at 16 kHz
encoder.set_inband_fec(true); // OPUS_SET_INBAND_FEC(1)
encoder.set_packet_loss_perc(20).unwrap(); // OPUS_SET_PACKET_LOSS_PERC(20)

let mut packets = Vec::new();
for k in 0..20 {
    let pcm: Vec<f32> = (0..frame)
        .map(|i| 0.3 * (2.0 * std::f32::consts::PI * 220.0 * (k * frame + i) as f32 / rate as f32).sin())
        .collect();
    let mut packet = vec![0u8; 1276];
    let len = encoder.encode(&pcm, frame, &mut packet).unwrap();
    packet.truncate(len);
    packets.push(packet);
}

let lost = 10;
let mut decoder = OpusDecoder::new(rate, 1).unwrap();
let mut pcm = vec![0.0f32; frame];
let mut output = Vec::new();
for i in 0..packets.len() {
    let samples = if i == lost {
        // Packet 10 is missing, and packet 11 is already here:
        // rebuild packet 10 from the copy that packet 11 carries.
        decoder.decode_fec(&packets[i + 1], frame, &mut pcm).unwrap()
    } else {
        decoder.decode(&packets[i], frame, &mut pcm).unwrap()
    };
    assert_eq!(samples, frame);
    output.extend_from_slice(&pcm[..samples]);
}
assert_eq!(output.len(), packets.len() * frame);
```

## 10. Float and 16-bit output

`decode` and `decode_fec` write `f32` samples. Full scale is ±1.0. The output is **not** clipped, as in libopus `opus_decode_float`: a loud signal comes back with peaks beyond ±1.0. A caller that needs the range must limit the samples itself.

`decode_i16` and `decode_fec_i16` follow libopus `opus_decode` in its float build. Both run `decode_i16_native`:

1. The packet is decoded to float into a buffer for the whole packet (`decode_native` with `soft_clip` set). The buffer holds 120 ms, the longest packet. A longer loss is concealed 60 ms at a time. This gives the same samples as concealing it in one call.
2. A decoded packet is soft-clipped (`pcm_soft_clip`, the port of `opus_pcm_soft_clip`):
   - Where the signal goes beyond ±1, the samples between the zero crossings around that place are replaced by `x + a * x * x`. As in the libopus build on 64-bit ARM, `a * x` is rounded and the second product is fused into the sum, so the output is libopus's bit for bit.
   - The value `a` is chosen so that the peak lands on ±1.
   - When such a stretch reaches the end of the packet, a value per channel (`softclip_mem`) keeps its `a`. The next packet applies the same curve to its first samples, up to the first zero crossing. So packets join without a step.
   - Concealed and FEC output is not soft-clipped, as in libopus.
3. Each sample is scaled by 32768, limited to the 16-bit range, and rounded to the nearest integer, halves to even (`float2int16`, libopus `FLOAT2INT16`).

`pcm_soft_clip` is public and can be used on float output directly:

```rust
use opus1::pcm_soft_clip;

let mut pcm = vec![0.0f32, 0.8, 1.6, 0.8, 0.0, -0.5]; // one channel, a peak of 1.6
let mut memory = [0.0f32; 1]; // one value per channel, 0 at the start of a stream
pcm_soft_clip(&mut pcm, 6, 1, &mut memory);
assert!(pcm.iter().all(|x| (-1.0..=1.0).contains(x)));
assert!(pcm[2] > 0.99); // the peak lands just below 1
```

## 11. The final range

The range coder ends every frame in a state, a 32-bit number. An encoder and a decoder that read the same frame the same way end in the same state. `OpusDecoder::final_range()` (libopus `OPUS_GET_FINAL_RANGE`) returns it for the last frame decoded:

- For a frame with a CELT layer it is the CELT decoder's final state. For a SILK-only frame it is the range decoder's state after the SILK data.
- When the frame has a redundant frame, that frame's final state is combined in by XOR.
- It is 0 after a concealed frame and after a frame of 1 byte or less.
- After `decode_fec` of a Hybrid packet it is the random state of the CELT concealment, as in libopus.

The value is for testing. Comparing it with `OpusEncoder::final_range()` for the same packet shows whether the decoder read every symbol as the encoder wrote it. libopus's test tool `opus_demo` stores the encoder's final range with each packet. It reports "Range coder state mismatch" when the decoder ends in another state. A mismatch means a corrupted packet or a bug, even when the audio sounds right.

```rust
use opus1::{Application, Bitrate, OpusDecoder, OpusEncoder};

let mut encoder = OpusEncoder::new(16_000, 1, Application::Voip).unwrap();
encoder.set_bitrate(Bitrate::BitsPerSecond(20_000)).unwrap();
let mut decoder = OpusDecoder::new(16_000, 1).unwrap();
let mut packet = vec![0u8; 1276];
let mut pcm16 = vec![0i16; 320];
for k in 0..10 {
    let pcm: Vec<f32> = (0..320)
        .map(|i| 0.3 * (2.0 * std::f32::consts::PI * 300.0 * (k * 320 + i) as f32 / 16_000.0).sin())
        .collect();
    let len = encoder.encode(&pcm, 320, &mut packet).unwrap();
    let n = decoder.decode_i16(&packet[..len], 320, &mut pcm16).unwrap();
    assert_eq!(n, 320);
    assert_eq!(decoder.final_range(), encoder.final_range());
}
```

## 12. Known differences from libopus

These figures come from the project's comparison of both decoders on packets from the libopus encoder (Apple M4 Pro, libopus 1.6.1). [10-differences-from-libopus.md](10-differences-from-libopus.md) collects the differences of the whole library.

- SILK: 49 of 49 test streams of `perf_harness modes` decode to the same samples as libopus.
- Hybrid: 29 of 29 test streams decode to the same samples.
- CELT: 51 of 51 test streams decode to the same samples.
- Mode switches: the 8 switching test streams decode to the same samples.
- The neural concealment of libopus is not ported.
- Packet extensions that libopus 1.5 and later store in the padding are not read (`src/packet.rs`).
