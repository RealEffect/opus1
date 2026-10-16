//! Multistream Opus encoder: several Opus streams in one packet.
//!
//! A multistream packet carries more than two channels. Each stream is an
//! ordinary mono or stereo Opus stream with its own [`OpusEncoder`]. The
//! [`ChannelLayout`] says which input channel goes to which stream.
//!
//! This is a port of libopus 1.6 opus_multistream_encoder.c:
//! `opus_multistream_encoder_init_impl`,
//! `opus_multistream_surround_encoder_init`, `opus_multistream_encode_native`
//! and `opus_multistream_encoder_ctl`. For every frame the encoder
//!
//! 1. splits the bitrate between the streams ([`super::rate_alloc`]);
//! 2. sets up every stream encoder for the kind of layout;
//! 3. copies the input channels of each stream out of the interleaved input,
//!    encodes them, and writes the stream into the packet. Every stream but
//!    the last is written with the size of its last frame ("self-delimited"
//!    framing, RFC 6716 appendix B), so a decoder finds the next stream.
//!
//! Nothing is allocated after construction.
//!
//! # Differences from libopus
//!
//! These are deliberate. The numbers are those of the design document.
//!
//! - **D1, bandwidth with an automatic or maximum bitrate.** In surround
//!   mode libopus picks the audio bandwidth of the streams from the bitrate
//!   setting. For the automatic and the maximum bitrate it uses the raw
//!   setting, -1000 or -1, and so always ends up with narrowband. This
//!   encoder uses the bitrate the allocation chose for the frame. With an
//!   explicit bitrate both do the same.
//! - **D3, frame sizes.** 80, 100 and 120 ms frames and the "expert frame
//!   duration" control are not offered: [`OpusEncoder`] has neither.
//! - **D5, controls.** Missing at the multistream level are the LSB depth,
//!   the constrained-VBR switch, the signal type, discontinuous
//!   transmission, the forced channel count, and the switches for
//!   prediction and phase inversion. [`OpusEncoder`] has the forced channel
//!   count and the prediction switch; they can be set on each stream
//!   through [`MultistreamEncoder::stream_encoder_mut`]. It has none of the
//!   others.
//! - **D6, energy mask.** libopus gives each stream encoder a pointer to the
//!   mask. Here the values are copied into the stream encoder.
//!
//! One thing libopus does has no counterpart here, because the stream
//! encoder has none: libopus gives each stream encoder the whole input and
//! the channels of the stream, for its analysis of speech against music.
//! [`OpusEncoder`] has no such analysis.

use super::{ChannelSource, Interleaved, Sample};
use super::layout::{ChannelLayout, MappingType};
use super::rate_alloc::{
    MAX_BITRATE_PER_CHANNEL, MS_FRAME_TMP, OPUS_AUTO, OPUS_BITRATE_MAX, bits_to_bitrate,
    cbr_last_stream_bitrate, cbr_max_data_bytes, clamp_user_bitrate, rate_allocation,
    smallest_packet, stream_max_bytes,
};
use super::surround::{MASK_BANDS, SurroundAnalysis};
use crate::packet;
use crate::{Application, Bandwidth, ForcedMode, OPUS_MAX_PACKET_BYTES, OpusEncoder};

const BUFFER_TOO_SMALL: &str = "Output buffer too small";

/// Total bitrate of a multistream encoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bitrate {
    /// libopus `OPUS_AUTO`: the encoder picks the bitrate from the sampling
    /// rate, the frame size and the layout.
    Auto,
    /// libopus `OPUS_BITRATE_MAX`: as many bits as the packet can hold.
    Max,
    /// Bits per second for all channels together.
    BitsPerSecond(i32),
}

/// The bitrate control of a stream encoder (libopus `opus_encoder_ctl`,
/// `OPUS_SET_BITRATE_REQUEST`): a positive value is limited to 500 bits per
/// second at least and 750 000 bits per second per channel at most.
/// [`OpusEncoder`] itself allows more, so the limit is applied here.
#[inline]
fn stream_bitrate(value: i32, stream_channels: usize) -> i32 {
    value.clamp(500, MAX_BITRATE_PER_CHANNEL * stream_channels as i32)
}

/// The input channels of stream `s` (C `c1` and `c2`): the left and the right
/// channel of a coupled stream, or the one channel of a mono stream. The
/// samples and the energy mask of a stream both go by this pair, so the two
/// cannot disagree on which channel comes first.
fn stream_input_channels(
    layout: &ChannelLayout,
    s: usize,
) -> Result<(usize, Option<usize>), &'static str> {
    if s < layout.coupled_streams() {
        let missing = "multistream encoder: a coupled stream has no input channel";
        let left = layout.left_channel(s, None).ok_or(missing)?;
        let right = layout.right_channel(s, None).ok_or(missing)?;
        Ok((left, Some(right)))
    } else {
        let chan = layout
            .mono_channel(s, None)
            .ok_or("multistream encoder: a mono stream has no input channel")?;
        Ok((chan, None))
    }
}

/// Builds the energy mask of one stream out of the result of the surround
/// analysis: the 21 values of input channel `c1`, then those of `c2` for a
/// coupled stream (C lines 991-995 and 1007-1008). `band_smr` holds the masks
/// of all input channels, one after the other. Returns the part of `dst`
/// that was written.
#[inline]
fn stream_mask<'a>(
    dst: &'a mut [f32; 2 * MASK_BANDS],
    band_smr: &[f32],
    c1: usize,
    c2: Option<usize>,
) -> &'a [f32] {
    dst[..MASK_BANDS].copy_from_slice(&band_smr[MASK_BANDS * c1..][..MASK_BANDS]);
    match c2 {
        Some(c2) => {
            dst[MASK_BANDS..].copy_from_slice(&band_smr[MASK_BANDS * c2..][..MASK_BANDS]);
            &dst[..]
        }
        None => &dst[..MASK_BANDS],
    }
}

/// Encoder for a multistream Opus packet. libopus `OpusMSEncoder`.
pub struct MultistreamEncoder {
    layout: ChannelLayout,
    mapping_type: MappingType,
    /// The stream of the low-frequency-effects (LFE) channel. Only a
    /// surround layout with 6, 7 or 8 channels has one.
    lfe_stream: Option<usize>,
    application: Application,
    sampling_rate: i32,
    /// C `st->bitrate_bps`: `OPUS_AUTO`, `OPUS_BITRATE_MAX` or bits per
    /// second for all channels together.
    bitrate_bps: i32,
    /// What `set_bandwidth` was given last.
    user_bandwidth: Bandwidth,
    /// What `set_max_bandwidth` was given last.
    max_bandwidth: Bandwidth,
    /// One encoder per stream. The first `coupled_streams` are stereo.
    encoders: Vec<OpusEncoder>,
    /// The input channels of each stream (`stream_input_channels`).
    stream_inputs: Vec<(usize, Option<usize>)>,
    /// The bitrate the allocation gave each stream for the current frame.
    rates: Vec<i32>,
    /// Per stream, the duration in samples of one frame of its last packet
    /// (C `prev_framesize` of the stream encoder). 0 until a frame is coded
    /// and after `reset_state`.
    stream_frame_size: Vec<usize>,
    /// The stream encoders have been given a bitrate, which the first
    /// `encode` does. Before that libopus stream encoders are at
    /// `OPUS_AUTO`.
    rates_assigned: bool,
    /// Input of one stream: one frame, interleaved when the stream is stereo.
    buf: Vec<f32>,
    /// Packet of one stream as its encoder wrote it.
    tmp: Vec<u8>,
    /// Surround masking analysis. Present only for a surround layout.
    surround: Option<SurroundAnalysis>,
    /// Output of the analysis: 21 values per input channel.
    band_smr: Vec<f32>,
}

impl MultistreamEncoder {
    /// Creates an encoder for a layout the caller chose. libopus
    /// `opus_multistream_encoder_create`.
    ///
    /// - `sampling_rate` is 8000, 12000, 16000, 24000 or 48000.
    /// - `channels` is the number of input channels, 1 to 255.
    /// - `streams` is the number of streams in a packet. The first
    ///   `coupled_streams` of them are stereo, the others mono.
    /// - `mapping` has one entry per input channel; more entries are ignored.
    ///   Entry `j` is the coded channel that input channel `j` feeds, or 255
    ///   for a channel that is not coded. Coded channels `2 * s` and
    ///   `2 * s + 1` are the left and right channel of stereo stream `s`.
    ///   Coded channel `s + coupled_streams` is mono stream `s`.
    ///
    /// Every stream needs an input channel, and there cannot be more coded
    /// channels than input channels.
    pub fn new(
        sampling_rate: i32,
        channels: usize,
        streams: usize,
        coupled_streams: usize,
        mapping: &[u8],
        application: Application,
    ) -> Result<Self, &'static str> {
        let layout = ChannelLayout::new(channels, streams, coupled_streams, mapping)?;
        Self::init(sampling_rate, layout, MappingType::None, None, application)
    }

    /// Creates an encoder with the layout of a mapping family of the Ogg
    /// Opus header. libopus `opus_multistream_surround_encoder_create`.
    ///
    /// - Family 0: 1 or 2 channels, one stream.
    /// - Family 1: 1 to 8 channels in Vorbis channel order. With more than 2
    ///   channels the encoder works in surround mode. With 6, 7 or 8
    ///   channels the last channel is the low-frequency-effects channel.
    /// - Family 255: 1 to 255 channels, one mono stream per channel.
    /// - Family 2: ambisonics. Every stream is coded CELT-only.
    ///
    /// Read the chosen layout back with [`layout`](Self::layout) to write
    /// the Ogg Opus header.
    pub fn new_surround(
        sampling_rate: i32,
        channels: usize,
        mapping_family: u8,
        application: Application,
    ) -> Result<Self, &'static str> {
        let (layout, mapping_type, lfe_stream) = ChannelLayout::surround(channels, mapping_family)?;
        Self::init(sampling_rate, layout, mapping_type, lfe_stream, application)
    }

    /// libopus `opus_multistream_encoder_init_impl`.
    fn init(
        sampling_rate: i32,
        layout: ChannelLayout,
        mapping_type: MappingType,
        lfe_stream: Option<usize>,
        application: Application,
    ) -> Result<Self, &'static str> {
        layout.validate_for_encoder()?;
        // The generic constructor has no LFE stream (C: `lfe_stream = -1`
        // unless the mapping type is surround).
        let lfe_stream = if mapping_type == MappingType::Surround {
            lfe_stream
        } else {
            None
        };
        let streams = layout.streams();
        let coupled = layout.coupled_streams();
        let channels = layout.channels();

        let stream_inputs = (0..streams)
            .map(|s| stream_input_channels(&layout, s))
            .collect::<Result<Vec<_>, _>>()?;
        let mut encoders = Vec::with_capacity(streams);
        for s in 0..streams {
            let stream_channels = if s < coupled { 2 } else { 1 };
            let mut enc = OpusEncoder::new(sampling_rate, stream_channels, application)?;
            if Some(s) == lfe_stream {
                enc.set_lfe(true);
            }
            encoders.push(enc);
        }

        // The longest frame is 60 ms. A stereo stream has two samples each.
        let max_frame = 3 * sampling_rate as usize / 50;
        let (surround, band_smr) = if mapping_type == MappingType::Surround {
            (
                Some(SurroundAnalysis::new(channels, sampling_rate)),
                vec![0.0; MASK_BANDS * channels],
            )
        } else {
            (None, Vec::new())
        };

        Ok(Self {
            layout,
            mapping_type,
            lfe_stream,
            application,
            sampling_rate,
            bitrate_bps: OPUS_AUTO,
            user_bandwidth: Bandwidth::Auto,
            max_bandwidth: Bandwidth::Fullband,
            encoders,
            stream_inputs,
            rates: vec![0; streams],
            stream_frame_size: vec![0; streams],
            rates_assigned: false,
            buf: vec![0.0; 2 * max_frame],
            tmp: vec![0; MS_FRAME_TMP],
            surround,
            band_smr,
        })
    }

    /// The channel layout: channels, streams, coupled streams and mapping.
    pub fn layout(&self) -> &ChannelLayout {
        &self.layout
    }

    /// How the layout was chosen. It decides how the streams are set up:
    /// surround mode, ambisonics, or nothing special.
    pub fn mapping_type(&self) -> MappingType {
        self.mapping_type
    }

    /// The stream that carries the low-frequency-effects channel, if the
    /// layout has one.
    pub fn lfe_stream(&self) -> Option<usize> {
        self.lfe_stream
    }

    /// The application the stream encoders were created for.
    pub fn application(&self) -> Application {
        self.application
    }

    /// The sampling rate of the input in Hz.
    pub fn sampling_rate(&self) -> i32 {
        self.sampling_rate
    }

    /// libopus `OPUS_GET_LOOKAHEAD`: the delay the encoder adds, in samples
    /// per channel at the input sampling rate. It is the same for every
    /// stream. An Ogg Opus header stores it, scaled to 48 kHz, as the
    /// pre-skip.
    pub fn lookahead(&self) -> usize {
        self.encoders.first().map_or(0, |enc| enc.lookahead())
    }

    /// Encodes one frame of interleaved `f32` samples, scale ±1.0.
    /// libopus `opus_multistream_encode_float`.
    ///
    /// - `pcm` holds `frame_size * channels` samples: sample `i` of channel
    ///   `c` is `pcm[i * channels + c]`. More samples are ignored.
    /// - `frame_size` is 2.5, 5, 10, 20, 40 or 60 ms in samples per channel.
    /// - `output.len()` is the largest packet allowed. The smallest buffer
    ///   that can work is `2 * streams - 1` bytes. A buffer that leaves a
    ///   stream too few bytes is an error ("Output buffer too small").
    ///
    /// Returns the length of the packet in bytes. With constant bitrate and
    /// an explicit bitrate the packet has `bitrate * duration / 8` bytes,
    /// rounded to a byte and at least `2 * streams - 1`, or `output.len()`
    /// when that is smaller. With constant bitrate and [`Bitrate::Max`] the
    /// packet fills `output`.
    pub fn encode(
        &mut self,
        pcm: &[f32],
        frame_size: usize,
        output: &mut [u8],
    ) -> Result<usize, &'static str> {
        self.check_input(pcm.len(), frame_size)?;
        let src = Interleaved {
            pcm,
            channels: self.layout.channels(),
        };
        self.encode_native(&src, frame_size, output)
    }

    /// Encodes one frame of interleaved 16-bit samples. libopus
    /// `opus_multistream_encode`.
    ///
    /// The arguments are those of [`encode`](Self::encode). The packet is
    /// the one `encode` writes for the samples `s as f32 / 32768.0`.
    pub fn encode_i16(
        &mut self,
        pcm: &[i16],
        frame_size: usize,
        output: &mut [u8],
    ) -> Result<usize, &'static str> {
        self.check_input(pcm.len(), frame_size)?;
        let src = Interleaved {
            pcm,
            channels: self.layout.channels(),
        };
        self.encode_native(&src, frame_size, output)
    }

    /// Checks the frame size (libopus `frame_size_select` without a frame
    /// duration override) and that the input holds one frame.
    fn check_input(&self, pcm_len: usize, frame_size: usize) -> Result<(), &'static str> {
        if crate::frame_rate_from_params(self.sampling_rate, frame_size).is_none() {
            // D3: libopus also takes 80, 100 and 120 ms.
            return Err("Invalid frame size");
        }
        if pcm_len < frame_size * self.layout.channels() {
            return Err("Input buffer too small for frame");
        }
        Ok(())
    }

    /// libopus `opus_multistream_encode_native`. `frame_size` has been
    /// checked, and `src` holds a frame of every input channel.
    fn encode_native<T: Sample>(
        &mut self,
        src: &Interleaved<'_, T>,
        frame_size: usize,
        output: &mut [u8],
    ) -> Result<usize, &'static str> {
        let fs = self.sampling_rate;
        let streams = self.layout.streams();
        let coupled = self.layout.coupled_streams();
        let surround_mode = self.mapping_type == MappingType::Surround;
        // libopus reads the VBR switch from the first stream.
        let vbr = self.vbr();

        // Smallest packet the encoder can produce.
        let smallest = smallest_packet(streams, frame_size, fs)?;
        if output.len() < smallest {
            return Err(BUFFER_TOO_SMALL);
        }

        // Surround masking: how far each band of each channel lies above
        // what the other channels mask (C 910-913). libopus runs the
        // analysis for the surround mapping type only, and only such an
        // encoder has one here. The analysis reads the input channels
        // through `src`. It gives a mask for every frame size and sampling
        // rate this function is reached with, so `have_mask` is then true.
        let have_mask = match self.surround.as_mut() {
            Some(analysis) => analysis.run(src, frame_size, &mut self.band_smr),
            None => false,
        };

        // Compute bitrate allocation between streams.
        let rate_sum = rate_allocation(
            &self.layout,
            self.mapping_type,
            self.lfe_stream,
            self.bitrate_bps,
            frame_size,
            fs,
            &mut self.rates,
        )?;

        // Constant bitrate: the packet gets exactly the bytes of the bitrate.
        let mut max_data_bytes = output.len();
        if !vbr {
            max_data_bytes = cbr_max_data_bytes(
                self.bitrate_bps,
                rate_sum,
                streams,
                frame_size,
                fs,
                max_data_bytes,
            )?;
        }

        // Surround mode: one audio bandwidth for all streams, from the
        // bitrate per channel (C `equiv_rate`).
        let surround_bandwidth = if surround_mode {
            // D1: libopus uses the raw setting here, which is -1000 or -1
            // for an automatic or maximum bitrate and always gives
            // narrowband. This uses the bitrate the allocation chose.
            let total_bitrate =
                if self.bitrate_bps == OPUS_AUTO || self.bitrate_bps == OPUS_BITRATE_MAX {
                    rate_sum
                } else {
                    self.bitrate_bps
                };
            let nb_channels = self.layout.channels() as i64;
            let (frame_size, fs) = (frame_size as i64, fs as i64);
            let mut equiv_rate = total_bitrate as i64;
            // Short frames spend more of the rate on overhead.
            if frame_size * 50 < fs {
                equiv_rate -= 60 * (fs / frame_size - 50) * nb_channels;
            }
            Some(if equiv_rate > 10000 * nb_channels {
                Bandwidth::Fullband
            } else if equiv_rate > 7000 * nb_channels {
                Bandwidth::Superwideband
            } else if equiv_rate > 5000 * nb_channels {
                Bandwidth::Wideband
            } else {
                Bandwidth::Narrowband
            })
        } else {
            None
        };

        // Settings of every stream encoder for this frame.
        for (s, enc) in self.encoders.iter_mut().enumerate() {
            enc.bitrate_bps = stream_bitrate(self.rates[s], enc.channels());
            if let Some(bandwidth) = surround_bandwidth {
                enc.set_bandwidth(bandwidth);
                if s < coupled {
                    // "To preserve the spatial image, force stereo CELT on
                    // coupled streams" (opus_multistream_encoder.c:953-958):
                    // CELT-only, and two coded channels, so that a stereo
                    // stream encoder does not code its input as mono at a
                    // low bitrate. Ambisonics streams are not forced to two
                    // channels.
                    enc.set_force_mode(Some(ForcedMode::CeltOnly));
                    let _ = enc.set_force_channels(Some(2));
                }
            } else if self.mapping_type == MappingType::Ambisonics {
                enc.set_force_mode(Some(ForcedMode::CeltOnly));
            }
        }
        self.rates_assigned = true;

        let mut band_log_e = [0.0f32; 2 * MASK_BANDS];
        let mut tot_size = 0usize;
        let streams_and_inputs = self.encoders.iter_mut().zip(&self.stream_inputs);
        for (s, (enc, &(c1, c2))) in streams_and_inputs.enumerate() {
            let last = s + 1 == streams;
            let stream_channels = enc.channels();
            // Copy the input of the stream: left and right interleaved for a
            // coupled stream.
            match c2 {
                Some(c2) => src.copy_pair(&mut self.buf, c1, c2, frame_size),
                None => src.copy(&mut self.buf, c1, frame_size),
            }
            if surround_mode {
                // libopus sets the mask of every stream on every frame in
                // surround mode: the mask of the left channel, then that of
                // the right. D6: the values are copied. Without a mask the
                // stream encoder is told so, which also drops the mask of an
                // earlier frame.
                enc.set_energy_mask(if have_mask {
                    Some(stream_mask(&mut band_log_e, &self.band_smr, c1, c2))
                } else {
                    None
                });
            }

            // Number of bytes left for this stream. It keeps one byte for
            // the last stream and two for each other stream still to come,
            // and the bytes of this stream's own length field.
            let curr_max = stream_max_bytes(max_data_bytes, tot_size, streams, s, frame_size, fs)?;
            if !vbr && last {
                // Constant bitrate: the last stream fills what is left.
                enc.bitrate_bps = stream_bitrate(
                    cbr_last_stream_bitrate(curr_max, frame_size, fs)?,
                    stream_channels,
                );
            }
            let len = enc.encode(
                &self.buf[..frame_size * stream_channels],
                frame_size,
                &mut self.tmp[..curr_max],
            )?;

            // The stream encoder can return more than one frame (a 60 ms
            // CELT packet is three 20 ms frames) and can pad its packet.
            // Like the libopus repacketizer, take the frames out and write
            // them anew: with the size of the last frame for every stream
            // but the last, and padded to the full size for the last stream
            // of a constant-bitrate packet.
            let stream_packet = &self.tmp[..len];
            let parsed = packet::parse(stream_packet, false)?;
            let mut frames: [&[u8]; packet::MAX_FRAMES] = [&[]; packet::MAX_FRAMES];
            let mut coded = false;
            for (slot, frame) in frames.iter_mut().zip(parsed.frames(stream_packet)) {
                coded |= !frame.is_empty();
                *slot = frame;
            }
            if coded {
                // For `bitrate()`. libopus does not record the frame size
                // of a packet without audio either.
                self.stream_frame_size[s] = packet::samples_per_frame(parsed.toc, fs);
            }
            let room = output
                .get_mut(tot_size..max_data_bytes)
                .ok_or(BUFFER_TOO_SMALL)?;
            let written = packet::write(
                parsed.toc,
                &frames[..parsed.count],
                room,
                !last,
                !vbr && last,
            )?;
            tot_size += written;
        }
        Ok(tot_size)
    }

    // --- Settings of the whole encoder ---

    /// Sets the total bitrate. libopus `OPUS_SET_BITRATE`.
    ///
    /// An explicit value must be positive. It is limited to 500 to 750 000
    /// bits per second per input channel. The default is [`Bitrate::Auto`].
    pub fn set_bitrate(&mut self, bitrate: Bitrate) -> Result<(), &'static str> {
        self.bitrate_bps = match bitrate {
            Bitrate::Auto => OPUS_AUTO,
            Bitrate::Max => OPUS_BITRATE_MAX,
            Bitrate::BitsPerSecond(bps) => {
                if bps <= 0 {
                    return Err("Invalid bitrate");
                }
                clamp_user_bitrate(bps, self.layout.channels())?
            }
        };
        Ok(())
    }

    /// The bitrate that was set, after its limits were applied.
    pub fn bitrate_setting(&self) -> Bitrate {
        match self.bitrate_bps {
            OPUS_AUTO => Bitrate::Auto,
            OPUS_BITRATE_MAX => Bitrate::Max,
            bps => Bitrate::BitsPerSecond(bps),
        }
    }

    /// The sum of the bitrates of the streams, in bits per second. libopus
    /// `OPUS_GET_BITRATE`.
    ///
    /// After [`encode`](Self::encode) this is what the allocation gave the
    /// streams for that frame. As in libopus, a stream counts with at most
    /// what a 1276-byte packet per frame carries. Before the first frame
    /// every stream counts with the bitrate a single-stream encoder picks
    /// by itself.
    pub fn bitrate(&self) -> i32 {
        let fs = self.sampling_rate;
        let mut sum = 0i64;
        for (enc, &frame_size) in self.encoders.iter().zip(&self.stream_frame_size) {
            // C `user_bitrate_to_bitrate(st, st->prev_framesize, 1276)`.
            let frame_size = if frame_size == 0 {
                fs as usize / 400
            } else {
                frame_size
            };
            let Ok(max_bitrate) = bits_to_bitrate(OPUS_MAX_PACKET_BYTES as i32 * 8, fs, frame_size) else {
                continue;
            };
            let user_bitrate = if self.rates_assigned {
                enc.bitrate_bps
            } else {
                // The stream encoder's own choice for `OPUS_AUTO`.
                60 * fs / frame_size as i32 + fs * enc.channels() as i32
            };
            sum += user_bitrate.min(max_bitrate) as i64;
        }
        sum.clamp(0, i32::MAX as i64) as i32
    }

    // --- Settings forwarded to every stream encoder ---

    /// Sets the complexity, 0 to 10, on every stream. libopus
    /// `OPUS_SET_COMPLEXITY`. The default is 9.
    pub fn set_complexity(&mut self, complexity: i32) -> Result<(), &'static str> {
        if !(0..=10).contains(&complexity) {
            return Err("Invalid complexity");
        }
        for enc in &mut self.encoders {
            enc.complexity = complexity;
        }
        Ok(())
    }

    /// The complexity of the first stream. libopus `OPUS_GET_COMPLEXITY`.
    pub fn complexity(&self) -> i32 {
        self.encoders.first().map_or(0, |enc| enc.complexity)
    }

    /// Selects variable bitrate (`true`, the default) or constant bitrate
    /// (`false`) on every stream. libopus `OPUS_SET_VBR`.
    ///
    /// With constant bitrate every packet has the same size for the same
    /// frame duration.
    pub fn set_vbr(&mut self, vbr: bool) {
        for enc in &mut self.encoders {
            enc.use_cbr = !vbr;
        }
    }

    /// Whether the first stream uses variable bitrate. libopus
    /// `OPUS_GET_VBR`. The encoder goes by this value for the whole packet.
    pub fn vbr(&self) -> bool {
        self.encoders.first().is_none_or(|enc| !enc.use_cbr)
    }

    /// Turns in-band forward error correction on or off on every stream.
    /// libopus `OPUS_SET_INBAND_FEC`. Only streams coded with SILK carry it.
    pub fn set_inband_fec(&mut self, enabled: bool) {
        for enc in &mut self.encoders {
            enc.use_inband_fec = enabled;
        }
    }

    /// Whether in-band forward error correction is on in the first stream.
    /// libopus `OPUS_GET_INBAND_FEC`.
    pub fn inband_fec(&self) -> bool {
        self.encoders.first().is_some_and(|enc| enc.use_inband_fec)
    }

    /// Sets the expected packet loss, 0 to 100 percent, on every stream.
    /// libopus `OPUS_SET_PACKET_LOSS_PERC`.
    pub fn set_packet_loss_perc(&mut self, percent: i32) -> Result<(), &'static str> {
        if !(0..=100).contains(&percent) {
            return Err("Invalid packet loss percentage");
        }
        for enc in &mut self.encoders {
            enc.packet_loss_perc = percent;
        }
        Ok(())
    }

    /// The expected packet loss of the first stream in percent. libopus
    /// `OPUS_GET_PACKET_LOSS_PERC`.
    pub fn packet_loss_perc(&self) -> i32 {
        self.encoders.first().map_or(0, |enc| enc.packet_loss_perc)
    }

    /// Sets the audio bandwidth every stream codes. libopus
    /// `OPUS_SET_BANDWIDTH`. `Bandwidth::Auto`, the default, lets each
    /// stream decide from its bitrate.
    ///
    /// In surround mode the encoder sets the bandwidth of every stream
    /// again on every frame, exactly like libopus, so this has no effect
    /// there. The LFE stream is always narrowband.
    pub fn set_bandwidth(&mut self, bandwidth: Bandwidth) {
        self.user_bandwidth = bandwidth;
        for enc in &mut self.encoders {
            enc.set_bandwidth(bandwidth);
        }
    }

    /// What [`set_bandwidth`](Self::set_bandwidth) was given last.
    ///
    /// This is the setting. libopus `OPUS_GET_BANDWIDTH` reports something
    /// else: the bandwidth the first stream coded last. The TOC byte of a
    /// stream of the packet tells that.
    pub fn bandwidth(&self) -> Bandwidth {
        self.user_bandwidth
    }

    /// Sets the widest audio bandwidth the automatic choice of a stream may
    /// take. libopus `OPUS_SET_MAX_BANDWIDTH`. `Bandwidth::Auto` is not a
    /// bandwidth and is rejected. The default is fullband.
    ///
    /// A bandwidth set with [`set_bandwidth`](Self::set_bandwidth), or by
    /// the encoder in surround mode, is not limited by it.
    pub fn set_max_bandwidth(&mut self, bandwidth: Bandwidth) -> Result<(), &'static str> {
        if bandwidth == Bandwidth::Auto {
            return Err("Invalid maximum bandwidth");
        }
        for enc in &mut self.encoders {
            enc.set_max_bandwidth(bandwidth)?;
        }
        self.max_bandwidth = bandwidth;
        Ok(())
    }

    /// What [`set_max_bandwidth`](Self::set_max_bandwidth) was given last.
    pub fn max_bandwidth(&self) -> Bandwidth {
        self.max_bandwidth
    }

    /// Asks every stream for a coding mode instead of its automatic choice.
    /// libopus `OPUS_SET_FORCE_MODE`. `None` gives the choice back.
    ///
    /// In surround mode the encoder sets the coupled streams to CELT-only on
    /// every frame. In ambisonics mode it does so for all streams.
    pub fn set_force_mode(&mut self, mode: Option<ForcedMode>) {
        for enc in &mut self.encoders {
            enc.set_force_mode(mode);
        }
    }

    // --- State ---

    /// The final range values of all streams after the last packet,
    /// combined with XOR. libopus `OPUS_GET_FINAL_RANGE`. A multistream
    /// decoder that read the packet correctly reports the same value.
    pub fn final_range(&self) -> u32 {
        self.encoders
            .iter()
            .fold(0, |acc, enc| acc ^ enc.final_range())
    }

    /// Forgets the signal coded so far. libopus `OPUS_RESET_STATE`. The
    /// settings stay.
    pub fn reset_state(&mut self) {
        if let Some(analysis) = self.surround.as_mut() {
            analysis.reset();
        }
        for enc in &mut self.encoders {
            enc.reset_state();
        }
        self.stream_frame_size.fill(0);
    }

    /// The encoder of stream `stream`. libopus
    /// `OPUS_MULTISTREAM_GET_ENCODER_STATE`. `None` when there is no such
    /// stream.
    pub fn stream_encoder(&self, stream: usize) -> Option<&OpusEncoder> {
        self.encoders.get(stream)
    }

    /// Like [`stream_encoder`](Self::stream_encoder), to change a setting of
    /// one stream. The next [`encode`](Self::encode) sets the bitrate of
    /// every stream again, and in surround and ambisonics mode also the
    /// bandwidth and the coding mode as described there.
    pub fn stream_encoder_mut(&mut self, stream: usize) -> Option<&mut OpusEncoder> {
        self.encoders.get_mut(stream)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stream_mask_holds_the_left_channel_then_the_right() {
        // The mask of channel c holds 100 * c + band.
        let band_smr: Vec<f32> = (0..4 * MASK_BANDS)
            .map(|k| (100 * (k / MASK_BANDS) + k % MASK_BANDS) as f32)
            .collect();
        // A coupled stream with channel 3 left and channel 1 right.
        let mut dst = [-1.0f32; 2 * MASK_BANDS];
        let mask = stream_mask(&mut dst, &band_smr, 3, Some(1));
        assert_eq!(mask.len(), 2 * MASK_BANDS);
        for band in 0..MASK_BANDS {
            assert_eq!(mask[band], (300 + band) as f32);
            assert_eq!(mask[MASK_BANDS + band], (100 + band) as f32);
        }
        // A mono stream: 21 values, and nothing else is written.
        let mut dst = [-1.0f32; 2 * MASK_BANDS];
        let mask = stream_mask(&mut dst, &band_smr, 2, None);
        assert_eq!(mask.len(), MASK_BANDS);
        assert_eq!(mask[0], 200.0);
        assert_eq!(mask[MASK_BANDS - 1], 220.0);
        assert!(dst[MASK_BANDS..].iter().all(|&v| v == -1.0));
    }

    #[test]
    fn no_mask_is_set_when_the_analysis_gives_none() {
        let mut enc = MultistreamEncoder::new_surround(48000, 6, 1, Application::Audio).unwrap();
        let pcm: Vec<f32> = (0..960 * 6)
            .map(|k| ((k * 7 % 97) as f32 - 48.0) / 200.0)
            .collect();
        let mut out = vec![0u8; 6000];
        enc.encode(&pcm, 960, &mut out).unwrap();
        for stream in &enc.encoders {
            let mask = stream.energy_mask().expect("a surround stream has a mask");
            assert_eq!(mask.len(), MASK_BANDS * stream.channels());
        }
        // No frame size or sampling rate that `encode` accepts leaves the
        // analysis without a mask. To reach that branch, the encoder gets an
        // analysis for a channel count libopus has no surround layout for.
        // It gives no mask, and every stream must then lose the mask of
        // the frame before.
        enc.surround = Some(SurroundAnalysis::new(2, 48000));
        enc.encode(&pcm, 960, &mut out).unwrap();
        for stream in &enc.encoders {
            assert_eq!(stream.energy_mask(), None);
        }
    }

    #[test]
    fn only_a_surround_layout_has_an_analysis() {
        // libopus runs the analysis for the surround mapping type only
        // (opus_multistream_encoder.c:910): 3 to 8 channels of family 1.
        for channels in 1..=8 {
            let enc =
                MultistreamEncoder::new_surround(48000, channels, 1, Application::Audio).unwrap();
            assert_eq!(enc.surround.is_some(), channels > 2, "{channels} channels");
            assert_eq!(enc.band_smr.len(), if channels > 2 { MASK_BANDS * channels } else { 0 });
        }
        for (channels, family) in [(3, 255), (6, 255), (4, 2), (6, 2), (9, 2)] {
            let enc = MultistreamEncoder::new_surround(48000, channels, family, Application::Audio)
                .unwrap();
            assert!(enc.surround.is_none(), "family {family}, {channels} channels");
        }
        let enc =
            MultistreamEncoder::new(48000, 6, 4, 2, &[0, 4, 1, 2, 3, 5], Application::Audio)
                .unwrap();
        assert!(enc.surround.is_none(), "the generic constructor");
        assert_eq!(enc.lfe_stream(), None);
    }

    #[test]
    fn stream_input_channels_follow_the_layout() {
        // 5.1 in Vorbis order: front left, centre, front right, rear left,
        // rear right, LFE. libopus codes it as mapping [0, 4, 1, 2, 3, 5].
        let enc = MultistreamEncoder::new_surround(48000, 6, 1, Application::Audio).unwrap();
        assert_eq!(enc.layout().mapping(), &[0, 4, 1, 2, 3, 5]);
        let channels = |s| stream_input_channels(enc.layout(), s).unwrap();
        assert_eq!(channels(0), (0, Some(2)));
        assert_eq!(channels(1), (3, Some(4)));
        assert_eq!(channels(2), (1, None));
        assert_eq!(channels(3), (5, None));
        // 7.1: mapping [0, 6, 1, 2, 3, 4, 5, 7].
        let enc = MultistreamEncoder::new_surround(48000, 8, 1, Application::Audio).unwrap();
        let channels = |s| stream_input_channels(enc.layout(), s).unwrap();
        assert_eq!(channels(0), (0, Some(2)));
        assert_eq!(channels(1), (3, Some(4)));
        assert_eq!(channels(2), (5, Some(6)));
        assert_eq!(channels(3), (1, None));
        assert_eq!(channels(4), (7, None));
        // A generic layout. Two channels name the mono stream: the first one
        // is read. Channel 2 is not coded.
        let enc = MultistreamEncoder::new(48000, 5, 2, 1, &[1, 0, 255, 2, 2], Application::Audio)
            .unwrap();
        let channels = |s| stream_input_channels(enc.layout(), s).unwrap();
        assert_eq!(channels(0), (1, Some(0)));
        assert_eq!(channels(1), (3, None));
    }

    #[test]
    fn stream_bitrate_follows_the_stream_encoder_control() {
        assert_eq!(stream_bitrate(1, 1), 500);
        assert_eq!(stream_bitrate(500, 2), 500);
        assert_eq!(stream_bitrate(64_000, 1), 64_000);
        assert_eq!(stream_bitrate(750_001, 1), 750_000);
        assert_eq!(stream_bitrate(2_000_000, 2), 1_500_000);
        assert_eq!(stream_bitrate(i32::MAX, 2), 1_500_000);
    }

    #[test]
    fn bitrate_setting_round_trips() {
        let mut enc = MultistreamEncoder::new_surround(48000, 6, 1, Application::Audio).unwrap();
        assert_eq!(enc.bitrate_setting(), Bitrate::Auto);
        enc.set_bitrate(Bitrate::Max).unwrap();
        assert_eq!(enc.bitrate_setting(), Bitrate::Max);
        enc.set_bitrate(Bitrate::BitsPerSecond(256_000)).unwrap();
        assert_eq!(enc.bitrate_setting(), Bitrate::BitsPerSecond(256_000));
        // Limited to 500 and 750 000 bits per second per channel.
        enc.set_bitrate(Bitrate::BitsPerSecond(1)).unwrap();
        assert_eq!(enc.bitrate_setting(), Bitrate::BitsPerSecond(3000));
        enc.set_bitrate(Bitrate::BitsPerSecond(i32::MAX)).unwrap();
        assert_eq!(enc.bitrate_setting(), Bitrate::BitsPerSecond(4_500_000));
        // The two special values of libopus are not bitrates here.
        for bad in [0, -1, -1000, i32::MIN] {
            assert!(enc.set_bitrate(Bitrate::BitsPerSecond(bad)).is_err());
            assert_eq!(enc.bitrate_setting(), Bitrate::BitsPerSecond(4_500_000));
        }
    }

    #[test]
    fn frame_sizes_are_those_of_the_stream_encoder() {
        for fs in [8000, 12000, 16000, 24000, 48000] {
            let enc = MultistreamEncoder::new_surround(fs, 2, 1, Application::Audio).unwrap();
            let ms = |tenths: usize| fs as usize * tenths / 10_000;
            for tenths in [25, 50, 100, 200, 400, 600] {
                assert!(
                    enc.check_input(usize::MAX, ms(tenths)).is_ok(),
                    "{fs} Hz, {tenths}"
                );
            }
            // 80, 100 and 120 ms, which libopus takes, and sizes that are
            // no frame at all.
            for tenths in [800, 1000, 1200, 150, 300] {
                assert!(
                    enc.check_input(usize::MAX, ms(tenths)).is_err(),
                    "{fs} Hz, {tenths}"
                );
            }
            for n in [0, 1, ms(25) - 1, ms(200) + 1, fs as usize + 1, usize::MAX] {
                assert!(
                    enc.check_input(usize::MAX, n).is_err(),
                    "{fs} Hz, {n} samples"
                );
            }
            // One sample short of a frame.
            assert!(enc.check_input(2 * ms(200) - 1, ms(200)).is_err());
            assert!(enc.check_input(2 * ms(200), ms(200)).is_ok());
        }
    }
}
