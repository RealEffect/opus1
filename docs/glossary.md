# Glossary

This glossary defines, in alphabetical order, the terms that these documents use for Opus and for this library, `opus1`. Each entry gives a short, plain definition. Where it helps, it also names the related item in opus1 or in libopus. All terms are established names from RFC 6716 (the Opus standard), RFC 7845 (Ogg Opus), libopus (the reference implementation in C) or general signal processing. An abbreviation used inside a definition is written out in its own entry. [What Opus is](01-what-is-opus.md) introduces most of these terms in context, and the [documentation index](README.md) lists the other documents.

## A

- **Algorithmic delay**: The delay that the codec itself adds between input and output: one frame plus the lookahead. With frames of 20 milliseconds (ms) it is 26.5 ms, or 22.5 ms with `Application::RestrictedLowDelay`.
- **Ambisonics**: A format for three-dimensional sound that describes the sound field around the listener instead of the signal for each loudspeaker. Ogg Opus stores it with mapping family 2 (one mono stream per ambisonic channel, plus an optional stereo stream) or mapping family 3 (with a mixing matrix). RFC 8486 defines both families.
- **Anti-collapse**: A CELT tool for transient frames. When a band of one short block received no pulses, the decoder fills it with noise instead of silence.
- **Application**: The use an encoder is set up for (`opus1::Application`): `Voip` for speech, `Audio` for music and general audio, or `RestrictedLowDelay` for the lowest delay. It changes the decisions of the encoder, but not the format.

## B

- **Band**: A group of neighbouring MDCT coefficients that CELT codes together. The standard CELT mode has 21 bands, from 200 hertz (Hz) wide at the bottom to 4.4 kilohertz (kHz) wide at the top (15.6 to 20 kHz).
- **Band energy**: The energy of one CELT band in one frame, coded before the shape of the band. It is coded in two steps. First comes a coarse value in steps of 6 decibels (dB), predicted from the previous frame and from the band below. Then fine bits refine it.
- **Bandwidth (audio bandwidth)**: The highest audio frequency that a packet codes: narrowband (4 kHz), mediumband (6 kHz), wideband (8 kHz), super-wideband (12 kHz) or fullband (20 kHz). It is stored in the TOC byte and does not depend on the output sampling rate of the decoder (`opus1::Bandwidth`).
- **Bit allocation**: CELT's rule for dividing the bits of a frame among its bands. Encoder and decoder compute the same split from a fixed table and a few values in the frame, for example per-band boosts and the allocation trim. So the split itself costs few bits.
- **Bit-exact**: Producing exactly the same bits, or the same output samples, as a reference implementation. [What Opus is](01-what-is-opus.md#how-this-library-relates-to-libopus) says which parts of opus1 are bit-exact with libopus.
- **Bitrate**: The number of bits per second that the encoder spends, usually given in kilobits per second (kb/s). Opus covers about 6 to 510 kb/s. In opus1 it is set with `OpusEncoder::set_bitrate`.
- **Bitstream**: The exact sequence of bits that an encoder writes and a decoder reads. RFC 6716 fixes the format of the Opus bitstream. Encoders may differ in which bits they choose to write.

## C

- **CBR (constant bitrate)**: Every packet has the same size, whatever the audio. In opus1, `OpusEncoder::set_vbr(false)` selects it.
- **CELT (Constrained Energy Lapped Transform)**: The transform layer of Opus, from the Xiph.Org Foundation, for music and general audio. It codes MDCT coefficients band by band: the energy of each band explicitly, and the shape of each band with PVQ.
- **Channel mapping table**: See mapping family.
- **CNG (comfort noise generation)**: Noise that the SILK decoder adds while it conceals lost or missing frames, with the spectrum and level of the recent background noise. It keeps the background from dropping out during packet loss and during DTX.
- **Codec, encoder, decoder**: A codec is a pair of an encoder, which compresses audio into packets, and a decoder, which turns packets back into audio. Opus is an audio codec.
- **Comb filter (pitch pre-filter and post-filter)**: A filter that acts on the repeating part of a signal at a given pitch period. The CELT encoder applies it before coding to weaken the harmonics of a pitched sound (the pre-filter). The decoder applies the inverse after decoding (the post-filter). Together they lower the coding noise between the harmonics.
- **Complexity**: A setting from 0 to 10 that says how much computation the encoder may spend for better quality at the same bitrate. Both libopus and opus1 default to 9 (`OpusEncoder::complexity`).
- **Constrained VBR**: A variable bitrate whose packet sizes may vary only so far that a link at the nominal bitrate needs at most one frame of buffering. libopus uses it by default. opus1 uses it for CELT-only packets.
- **Coupled stream**: A stream of a multistream packet that carries two channels (a stereo pair). The coupled streams come first in a packet, and the other streams are mono.

## D

- **Decibel (dB)**: A logarithmic unit for a ratio of two powers or two amplitudes. A factor of two in amplitude is about 6 dB.
- **De-emphasis**: See pre-emphasis.
- **Deep PLC**: Packet loss concealment with a neural network, added to libopus 1.5 as a build option. opus1 does not have it.
- **Delayed decision**: A way of quantizing the SILK excitation. It keeps several candidate results and chooses between them only some samples later, which lowers the error. SILK uses 2 candidates from complexity 2, 3 from complexity 6 and 4 from complexity 8.
- **DRED (Deep REDundancy)**: Redundancy coded with a neural network, added to libopus 1.5 as a build option. opus1 does not have it. It carries a compact description of earlier speech in the padding of later packets.
- **DTX (discontinuous transmission)**: During silence or steady background noise, the encoder sends a packet only now and then. The decoder fills the gaps with comfort noise. The opus1 decoder plays DTX streams, but the opus1 encoder cannot produce them.
- **Dual stereo**: A CELT stereo option that codes the left and the right channel of a band separately, instead of as mid and side. It applies to the bands below the intensity band.

## E

- **End band**: The index one past the last CELT band that a frame codes. It follows from the bandwidth: 13 for narrowband, 17 for wideband, 19 for super-wideband and 21 for fullband.
- **Energy mask**: For each band of each channel, how far the band lies above what the other channels of a surround stream mask. The multistream encoder computes it for every frame and passes it to each stream encoder (`OpusEncoder::set_energy_mask`, libopus `OPUS_SET_ENERGY_MASK`). The stream encoder then spends fewer bits on masked bands.
- **Excitation**: In SILK, the part of the signal that linear prediction does not predict. SILK quantizes it to whole-number pulses and codes them, together with one gain per subframe.

## F

- **FEC (forward error correction)**: Redundant data that lets a receiver rebuild lost data without asking for it again. In Opus it is in-band FEC: a SILK frame carries a low-bitrate copy (LBRR) of the frame before it.
- **Final range**: The 32-bit state of the range coder after the last frame of a packet. An encoder and a decoder that handle the same packet in the same way end with the same value. So comparing the two checks a decoder (`final_range()` in opus1, `OPUS_GET_FINAL_RANGE` in libopus).
- **Fixed point, floating point**: Two ways to store fractions. Fixed point stores an integer with an agreed scale (see Q format). Floating point stores `f32` or `f64` values. opus1 computes SILK in fixed point and CELT in floating point.
- **Frame**: A piece of coded audio with a fixed duration: 2.5, 5, 10, 20, 40 or 60 ms. A packet holds 1 to 48 frames of the same configuration.
- **Frame count code**: The lowest two bits of the TOC byte. Code 0 means one frame, code 1 two frames of equal size, code 2 two frames of different sizes, and code 3 any number of frames, with an extra byte for the count.
- **Frame size (in function calls)**: In opus1 and libopus, the argument `frame_size` counts samples per channel, not bytes and not all channels together. At 48 kHz, 20 ms is a `frame_size` of 960.
- **Fullband (FB)**: The widest audio bandwidth: audio up to 20 kHz, held at a sampling rate of 48 kHz.
- **Fused multiply-add (FMA)**: A processor instruction that computes `a * b + c` with a single rounding instead of two. Whether a compiler uses it changes the last bits of floating-point results. This matters when two implementations are compared bit for bit.

## G

- **Granule position**: The position in an Ogg stream, stored with each Ogg page. In Ogg Opus it counts samples at 48 kHz, whatever the sampling rate of the input.

## H

- **Hybrid mode**: The mode in which SILK codes the audio up to 8 kHz and CELT codes the bands above 8 kHz (from band 17), both in the same frame. It is used for super-wideband and fullband speech at medium bitrates, with frames of 10 or 20 ms.

## I

- **Intensity stereo**: A CELT stereo tool for the bands from the intensity band up. It codes one shape per band for both channels, plus the energy of each channel. It saves bits but loses the differences in phase between the two channels.
- **Interleaved samples**: Samples of all channels stored in turn: left, right, left, right, and so on. The encoders and decoders of opus1 take and return interleaved samples.
- **Internal sampling rate**: The sampling rate at which a layer works: 8, 12 or 16 kHz for SILK, depending on the bandwidth, and always 48 kHz for CELT. It does not depend on the rate that the application uses.

## L

- **LBRR (low bitrate redundancy)**: SILK's name for its in-band FEC data: a copy of the previous frame, coded at a lower bitrate and carried in the next packet. `OpusDecoder::decode_fec` uses it to rebuild a lost packet.
- **LFE (low-frequency effects)**: The bass channel of a surround layout, the ".1" in 5.1. In a multistream surround stream it gets its own mono stream, which is always coded CELT-only and narrowband.
- **libopus**: The reference implementation of Opus, written in C and maintained by the Xiph.Org Foundation. opus1 follows libopus 1.6.
- **Lookahead**: How much the output of the decoder lags the input of the encoder, not counting the frame itself: 6.5 ms, or 2.5 ms with `RestrictedLowDelay` (`OpusEncoder::lookahead`, libopus `OPUS_GET_LOOKAHEAD`). An Ogg Opus file stores it as the pre-skip.
- **LPC (linear predictive coding)**: Coding a signal by predicting each sample as a weighted sum of the samples just before it, and coding the weights and the prediction error. SILK uses 10 weights (order 10) for narrowband and mediumband, and 16 for wideband.
- **LSF (line spectral frequencies)**: See NLSF.
- **LTP (long-term prediction)**: Predicting a sample of voiced speech from the signal one pitch period earlier. SILK uses a filter with 5 taps for each subframe of a voiced frame.

## M

- **Mapping family**: The number in an Ogg Opus header that says how the channels map to the streams of a multistream packet. The families and their support in opus1:
  - 0: mono or stereo in one stream (supported);
  - 1: 1 to 8 channels in the channel order of Vorbis, for surround sound (supported);
  - 2: ambisonics (supported);
  - 3: ambisonics with a mixing matrix (not supported);
  - 255: up to 255 channels without a defined meaning (supported).
- **MDCT (modified discrete cosine transform)**: A transform that turns a block of samples into frequency coefficients. The blocks overlap, so that the edges of frames cause no audible steps. CELT uses an MDCT with 960 coefficients for a 20 ms frame at 48 kHz, with an overlap of 2.5 ms.
- **Mediumband (MB)**: Audio up to 6 kHz, held at 12 kHz. Only SILK-only packets use it, because CELT has no mediumband.
- **Mid/side stereo**: Coding the average of the two channels (mid) and half their difference (side) instead of left and right. SILK always codes stereo this way, and CELT can choose it band by band.
- **Mode**: Which layers code a packet: SILK-only, Hybrid or CELT-only. It is stored in the TOC byte, and the encoder chooses it again for every packet.
- **Multistream**: A packet format that holds several Opus streams, each mono or stereo, to carry more than two channels (up to 255). opus1 implements it in the module `opus1::multistream`.

## N

- **Narrowband (NB)**: Audio up to 4 kHz, held at 8 kHz, as in a classic telephone call.
- **NLSF (normalized line spectral frequencies)**: Another form of the LPC weights: a set of frequencies, scaled to the range 0 to 1, from which the weights can be rebuilt. SILK quantizes its LPC filters in this form for two reasons. Small errors in it keep the filter stable. And it can be interpolated between frames.
- **Noise shaping**: Shaping the spectrum of the coding noise so that the signal masks it. The noise is moved near the peaks and harmonics of the spectrum of the signal. SILK does this in its noise shaping quantizer (NSQ), which quantizes the excitation.
- **Normative**: Part of a standard that every implementation must follow. In RFC 6716 the decoder is normative and the encoder is not.

## O

- **Ogg Opus**: The file format for Opus packets in the Ogg container, defined in RFC 7845, with the file extension `.opus`. Its header `OpusHead` holds the channel count, the pre-skip, the output gain and the channel mapping. Its header `OpusTags` holds text comments.
- **OSCE (Opus Speech Coding Enhancement)**: Decoder tools of libopus 1.5 and later, not in opus1, that improve decoded speech with neural networks. The libopus 1.6 source also has a related bandwidth extension (BWE) for wideband speech (`OPUS_SET_OSCE_BWE`).
- **Overlap**: The part of an MDCT block that it shares with the next frame. CELT overlaps frames by 2.5 ms (120 samples at 48 kHz). This overlap is part of the lookahead.

## P

- **Packet**: What one call of the encoder writes and one call of the decoder reads: a TOC byte and 1 to 48 frames, at most 120 ms in total. The packet does not store its own length. The transport, such as the Real-time Transport Protocol (RTP) or Ogg, carries it.
- **Packet extension**: Extra data that libopus 1.5 and later can store in the padding of a packet, for example DRED. Decoders that do not know an extension skip it. opus1 neither reads nor writes extensions.
- **Padding**: Extra bytes at the end of a packet that carry no audio, for example to bring a CBR packet to its exact size. Only packets with frame count code 3 can have padding.
- **Phase inversion**: In intensity stereo, CELT can invert the sign of one channel of a band, to code a band whose two channels are close to opposite. Such bands cancel when the decoded stereo is mixed down to mono. So libopus can turn the inversion off (`OPUS_SET_PHASE_INVERSION_DISABLED`). opus1 cannot.
- **Pitch, pitch period, pitch lag**: The pitch period is the length of one cycle of a periodic sound, such as voiced speech or a musical note. The pitch lag is the same period counted in samples. SILK estimates it for LTP, and the CELT encoder for its comb filter.
- **PLC (packet loss concealment)**: The decoder fills the gap of a lost packet with plausible audio, made from the audio before it. In opus1, pass an empty packet to `OpusDecoder::decode`.
- **Pre-emphasis, de-emphasis**: Pre-emphasis is a filter that boosts high frequencies before CELT codes the signal (`y[n] = x[n] - 0.85 * x[n-1]`). De-emphasis is its inverse after decoding. Together they make the coding noise lower at high frequencies relative to low ones.
- **Pre-skip**: The number of samples, at 48 kHz, that a player drops from the start of an Ogg Opus stream, because they come from the lookahead of the codec. The usual value is the lookahead of the encoder, scaled to 48 kHz.
- **PVQ (pyramid vector quantization)**: CELT's quantizer for the shape of a band. It picks a vector of whole numbers whose absolute values add up to a given number of pulses. It scales the vector to unit length, and codes its index with the range coder.

## Q

- **Q format**: A way to store a fraction as an integer. In format Qn the integer is the value times 2 to the power n, so 0.5 in Q15 is 16384. SILK's fixed-point code names values this way, for example `pred_Q13` in libopus and `pred_q13` in opus1.
- **QEXT (quality extension)**: An extension in the libopus 1.6 source for very high bitrates (`OPUS_SET_QEXT`). opus1 does not have it.

## R

- **Range coder**: The entropy coder of Opus. Each coded symbol narrows a range of numbers in proportion to its probability, so likely symbols cost fewer bits. SILK and CELT write into the same range coder within a frame.
- **Redundant frame**: An extra CELT frame of 5 ms that the encoder adds to a packet when the stream switches between CELT-only and SILK-only or Hybrid. The decoder uses it to cross-fade between the two modes.
- **Repacketizer**: The libopus tool (`OpusRepacketizer`) that joins the frames of several packets into one packet, or splits them again. opus1 offers the writing part as `packet::write`.
- **Resampler**: Code that converts audio from one sampling rate to another. SILK resamples between the rate of the application and its internal rate. CELT needs no separate resampler, because 48 kHz is a whole multiple of every rate that Opus accepts.
- **RFC (Request for Comments)**: A document series of the Internet Engineering Task Force (IETF), which includes its standards. Opus is RFC 6716, and Ogg Opus is RFC 7845.

## S

- **Sampling rate**: The number of samples per second of a signal. Opus accepts and produces 8, 12, 16, 24 and 48 kHz.
- **Self-delimited framing**: A variant of the packet format (RFC 6716 appendix B) that also stores the size of the last frame, so that a reader knows where the packet ends. Multistream packets use it for every stream but the last.
- **Short blocks**: See transient.
- **Signal type (encoder setting)**: A hint to the libopus encoder that the input is voice or music (`OPUS_SET_SIGNAL`). opus1 has no such setting and takes the hint from the application instead.
- **SILK**: The speech layer of Opus, based on linear prediction and derived from the SILK codec of Skype (the name is not an abbreviation). It works at 8, 12 or 16 kHz with frames of 10 or 20 ms, and codes audio up to 8 kHz.
- **SIMD (single instruction, multiple data)**: Processor instructions that work on several numbers at once, such as NEON on aarch64, and the Streaming SIMD Extensions (SSE) and Advanced Vector Extensions 2 (AVX2) on x86-64. opus1 uses them in its most frequent loops.
- **SNR (signal-to-noise ratio)**: The power of a reference signal divided by the power of the difference from it, in decibels. These documents use it to compare decoded audio with the input, or the output of opus1 with that of libopus.
- **Soft clipping**: Bringing sample values that go beyond full scale (±1.0) back into range with a smooth curve, instead of cutting them off. `OpusDecoder::decode_i16` does this before it rounds to 16 bits, as libopus's `opus_decode` does. The function itself is `opus1::pcm_soft_clip`.
- **Spreading**: A rotation that CELT applies to the shape of a band. It spreads a few pulses over more coefficients, so that the band does not sound tonal. The encoder picks one of four levels per frame: none, light, normal or aggressive.
- **Stream**: One sequence of Opus packets with one or two channels, from one encoder to one decoder.
- **Subframe**: A part of 5 ms of a SILK frame: four in a 20 ms frame, two in a 10 ms frame. SILK sets its gains, pitch lags and LTP filters once per subframe.
- **Super-wideband (SWB)**: Audio up to 12 kHz, held at 24 kHz.

## T

- **TF resolution (time-frequency resolution)**: The balance between detail in time and detail in frequency of the MDCT coefficients of a band. CELT can change it band by band (`tf_res` in the code). It uses Haar transforms across short blocks or across neighbouring coefficients.
- **TOC byte (table-of-contents byte)**: The first byte of every Opus packet: 5 bits of configuration (mode, bandwidth and frame duration), 1 stereo bit and the 2 bits of the frame count code. [Opus packets](02-packets.md) lists the 32 configurations.
- **Tonality analysis**: The libopus analysis of the encoder input (`analysis.c`) that estimates how tonal the signal is, whether it is speech or music, and its bandwidth. The floating-point build of libopus runs it at complexity 7 and above. opus1 does not have it.
- **Transient**: A sudden change in the signal, such as a drum hit. CELT codes a transient frame with several short MDCTs (short blocks) instead of one long one, for example eight MDCTs of 2.5 ms in a 20 ms frame.

## V

- **VAD (voice activity detection)**: SILK's estimate of whether, and how strongly, a frame contains speech, made from the energy in four frequency bands. It decides whether a frame counts as having voice activity. It is also used for DTX, FEC and the timing of bandwidth switches.
- **VBR (variable bitrate)**: The packet size follows how hard the audio is to code, while the average follows the bitrate asked for. It is the default in libopus and in opus1 (`set_vbr(true)`).
- **Voiced, unvoiced**: SILK's classes for a frame with speech. Voiced frames are periodic, with a pitch (vowels). Unvoiced frames are noise-like (sounds such as "s" or "f"). A third class marks frames without voice activity.

## W

- **Wideband (WB)**: Audio up to 8 kHz, held at 16 kHz.
