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
//! - **Expert frame duration.** libopus's `OPUS_SET_EXPERT_FRAME_DURATION`
//!   on a multistream encoder stores any number without a check, and every
//!   later encode call with a number that is no duration fails.
//!   [`MultistreamEncoder::set_expert_frame_duration`] takes a
//!   [`FrameDuration`], which holds only the numbers libopus accepts.
//! - **D5, controls (closed).** The forced channel count, the switches for
//!   prediction and phase inversion, the application and the voice ratio
//!   were missing at the multistream level. They are now forwarded as
//!   libopus forwards them: a setter goes to one stream after the other
//!   and stops at the first stream that refuses the value, and a getter
//!   reads the first stream. The forced channel count is the exception: no
//!   stream refuses a [`Channels`] value, so it is set on every stream,
//!   where libopus stops with an error at the first mono stream when it is
//!   stereo. `OPUS_GET_IN_DTX` has no multistream request in libopus;
//!   [`OpusEncoder::in_dtx`] of each stream tells it.
//! - **D6, energy mask.** libopus gives each stream encoder a pointer to the
//!   mask. Here the values are copied into the stream encoder.
//!
//! As in libopus, each stream encoder runs its own speech and music
//! analysis (from complexity 7 on) on the whole interleaved input: a
//! coupled stream on the mean of its two channels, a mono stream on its
//! one channel.

use crate::Error;
use super::{ChannelSource, Interleaved, Sample};
use super::layout::{ChannelLayout, MappingType};
use super::rate_alloc::{
    MAX_BITRATE_PER_CHANNEL, MS_FRAME_TMP, OPUS_AUTO, OPUS_BITRATE_MAX, cbr_last_stream_bitrate,
    cbr_max_data_bytes, clamp_user_bitrate, rate_allocation, smallest_packet, stream_max_bytes,
};
use super::surround::{MASK_BANDS, SurroundAnalysis};
use crate::analysis::{AnalysisPcm, SecondChannel};
use crate::packet;
use crate::{
    Application, Bandwidth, Bitrate, Channels, ForcedMode, FrameDuration, InbandFec, OpusEncoder,
    SampleRate, Signal,
};

/// The bitrate a stream encoder gets for `value` (libopus `opus_encoder_ctl`,
/// `OPUS_SET_BITRATE_REQUEST`): limited to 500 bits per second at least and
/// 750 000 bits per second per channel at most. A value of 0 or less, which
/// libopus refuses, becomes 500.
#[inline]
fn stream_bitrate(value: i32, stream_channels: usize) -> Bitrate {
    Bitrate::within_limits(value.clamp(500, MAX_BITRATE_PER_CHANNEL * stream_channels as i32))
}

/// The input channels of stream `s` (C `c1` and `c2`): the left and the right
/// channel of a coupled stream, or the one channel of a mono stream. The
/// samples and the energy mask of a stream both go by this pair, so the two
/// cannot disagree on which channel comes first.
fn stream_input_channels(
    layout: &ChannelLayout,
    s: usize,
) -> Result<(usize, Option<usize>), Error> {
    if s < layout.coupled_streams() {
        let missing = Error::bad_arg("multistream encoder: a coupled stream has no input channel");
        let left = layout.left_channel(s, None).ok_or(missing)?;
        let right = layout.right_channel(s, None).ok_or(missing)?;
        Ok((left, Some(right)))
    } else {
        let chan = layout
            .mono_channel(s, None)
            .ok_or(Error::bad_arg("multistream encoder: a mono stream has no input channel"))?;
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
    /// What `set_bandwidth` was given last: `None` for the automatic
    /// choice.
    user_bandwidth: Option<Bandwidth>,
    /// What `set_max_bandwidth` was given last.
    max_bandwidth: Bandwidth,
    /// One encoder per stream. The first `coupled_streams` are stereo.
    encoders: Vec<OpusEncoder>,
    /// The input channels of each stream (`stream_input_channels`).
    stream_inputs: Vec<(usize, Option<usize>)>,
    /// The bitrate the allocation gave each stream for the current frame.
    rates: Vec<i32>,
    /// C `st->variable_duration`: the expert frame duration
    /// ([`set_expert_frame_duration`](Self::set_expert_frame_duration)).
    variable_duration: FrameDuration,
    /// Input of one stream: one frame, interleaved when the stream is stereo.
    buf: Vec<f32>,
    /// Packet of one stream as its encoder wrote it.
    tmp: Vec<u8>,
    /// Surround masking analysis. Present only for a surround layout.
    surround: Option<SurroundAnalysis>,
    /// Output of the analysis: 21 values per input channel.
    band_smr: Vec<f32>,
}

impl core::fmt::Debug for MultistreamEncoder {
    /// The layout and the main settings, as their getters return them. The
    /// stream encoders and the buffers are left out.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("MultistreamEncoder")
            .field("sampling_rate", &self.sampling_rate())
            .field("layout", &self.layout)
            .field("mapping_type", &self.mapping_type)
            .field("application", &self.application)
            .field("bitrate_setting", &self.bitrate_setting())
            .field("complexity", &self.complexity())
            .field("vbr", &self.vbr())
            .field("bandwidth_setting", &self.user_bandwidth)
            .field("max_bandwidth", &self.max_bandwidth)
            .field("expert_frame_duration", &self.variable_duration)
            .finish_non_exhaustive()
    }
}

impl MultistreamEncoder {
    /// Creates an encoder for a layout the caller chose. libopus
    /// `opus_multistream_encoder_create`.
    ///
    /// - `sampling_rate` is the rate of the input.
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
        sampling_rate: SampleRate,
        channels: usize,
        streams: usize,
        coupled_streams: usize,
        mapping: &[u8],
        application: Application,
    ) -> Result<Self, Error> {
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
        sampling_rate: SampleRate,
        channels: usize,
        mapping_family: u8,
        application: Application,
    ) -> Result<Self, Error> {
        let (layout, mapping_type, lfe_stream) = ChannelLayout::surround(channels, mapping_family)?;
        Self::init(sampling_rate, layout, mapping_type, lfe_stream, application)
    }

    /// libopus `opus_multistream_encoder_init_impl`.
    fn init(
        sampling_rate: SampleRate,
        layout: ChannelLayout,
        mapping_type: MappingType,
        lfe_stream: Option<usize>,
        application: Application,
    ) -> Result<Self, Error> {
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
            let stream_channels = if s < coupled { Channels::Stereo } else { Channels::Mono };
            // A new stream encoder is at `Bitrate::Auto`, as in libopus.
            // `bitrate` reports that until the first frame sets the
            // bitrates of the streams.
            let mut enc = OpusEncoder::new(sampling_rate, stream_channels, application)?;
            if Some(s) == lfe_stream {
                enc.set_lfe(true);
            }
            encoders.push(enc);
        }

        let sampling_rate = sampling_rate.hz() as i32;
        // The longest frame is 120 ms. A stereo stream has two samples each.
        let max_frame = 6 * sampling_rate as usize / 50;
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
            user_bandwidth: None,
            max_bandwidth: Bandwidth::Fullband,
            encoders,
            stream_inputs,
            rates: vec![0; streams],
            variable_duration: FrameDuration::Argument,
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

    /// The application of the first stream. libopus
    /// `OPUS_GET_APPLICATION`.
    pub fn application(&self) -> Application {
        self.encoders.first().map_or(self.application, OpusEncoder::application)
    }

    /// Changes the application of every stream before the first frame.
    /// libopus `OPUS_SET_APPLICATION`. See
    /// [`OpusEncoder::set_application`]: after the first frame only the
    /// current application is accepted.
    pub fn set_application(&mut self, application: Application) -> Result<(), Error> {
        for enc in &mut self.encoders {
            enc.set_application(application)?;
        }
        self.application = application;
        Ok(())
    }

    /// The sampling rate of the input.
    pub fn sampling_rate(&self) -> SampleRate {
        SampleRate::from_valid_hz(self.sampling_rate)
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
    /// - `frame_size` is 2.5, 5, 10, 20, 40, 60, 80, 100 or 120 ms in
    ///   samples per channel. With an expert frame duration set
    ///   ([`set_expert_frame_duration`](Self::set_expert_frame_duration)),
    ///   only the first frame of that duration is coded, and `frame_size`
    ///   must be at least that long.
    /// - `output.len()` is the largest packet allowed. The smallest buffer that
    ///   can work is `2 * streams - 1` bytes. A buffer that leaves a stream too
    ///   few bytes is an error
    ///   ([`BufferTooSmall`](crate::ErrorKind::BufferTooSmall)).
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
    ) -> Result<usize, Error> {
        let selected = self.check_input(pcm.len(), frame_size)?;
        let src = Interleaved {
            pcm,
            channels: self.layout.channels(),
        };
        self.encode_native(&src, selected, frame_size, output)
    }

    /// Encodes one frame of interleaved 16-bit samples. libopus
    /// `opus_multistream_encode`.
    ///
    /// The arguments are those of [`encode`](Self::encode). The packet is
    /// the one `encode` writes for the samples `s as f32 / 32768.0` with an
    /// LSB depth of at most 16, as libopus `opus_multistream_encode` codes
    /// it (see [`OpusEncoder::encode_i16`]).
    pub fn encode_i16(
        &mut self,
        pcm: &[i16],
        frame_size: usize,
        output: &mut [u8],
    ) -> Result<usize, Error> {
        let selected = self.check_input(pcm.len(), frame_size)?;
        let src = Interleaved {
            pcm,
            channels: self.layout.channels(),
        };
        self.encode_native(&src, selected, frame_size, output)
    }

    /// Encodes one frame of interleaved 24-bit samples held in 32-bit
    /// integers: libopus `opus_multistream_encode24`. Full scale is 2^23;
    /// see [`OpusEncoder::encode24`]. `frame_size` and the errors are those
    /// of [`encode`](Self::encode).
    pub fn encode24(
        &mut self,
        pcm: &[i32],
        frame_size: usize,
        output: &mut [u8],
    ) -> Result<usize, Error> {
        let selected = self.check_input(pcm.len(), frame_size)?;
        let src = Interleaved {
            pcm,
            channels: self.layout.channels(),
        };
        self.encode_native(&src, selected, frame_size, output)
    }

    /// The frame to code: libopus `frame_size_select` with the application
    /// and the expert frame duration of this encoder. Also checks that the
    /// input holds the `frame_size` samples per channel the caller passed.
    /// Returns the frame size in samples per channel.
    fn check_input(&self, pcm_len: usize, frame_size: usize) -> Result<usize, Error> {
        let selected = crate::frame_size_select(
            self.application,
            frame_size,
            self.variable_duration,
            self.sampling_rate,
        )
        .ok_or(Error::INVALID_FRAME_SIZE)?;
        if pcm_len < frame_size.saturating_mul(self.layout.channels()) {
            return Err(Error::INPUT_TOO_SHORT);
        }
        Ok(selected)
    }

    /// libopus `opus_multistream_encode_native`. `frame_size` has been
    /// checked, and `src` holds `analysis_frame_size` samples of every
    /// input channel: the caller's frame size, which is at least
    /// `frame_size`.
    fn encode_native<T: Sample>(
        &mut self,
        src: &Interleaved<'_, T>,
        frame_size: usize,
        analysis_frame_size: usize,
        output: &mut [u8],
    ) -> Result<usize, Error> {
        let fs = self.sampling_rate;
        let streams = self.layout.streams();
        let coupled = self.layout.coupled_streams();
        let surround_mode = self.mapping_type == MappingType::Surround;
        // C 910, 989, 1005, 1013: surround masking needs CELT, so an encoder
        // with only SILK leaves it out.
        let masking = surround_mode && self.application != Application::RestrictedSilk;
        // libopus reads the VBR switch from the first stream.
        let vbr = self.vbr();

        // Smallest packet the encoder can produce.
        let smallest = smallest_packet(streams, frame_size, fs)?;
        if output.len() < smallest {
            return Err(Error::BUFFER_TOO_SMALL);
        }

        // Surround masking: how far each band of each channel lies above
        // what the other channels mask (C 910-913). libopus runs the
        // analysis for the surround mapping type only, and only such an
        // encoder has one here. The analysis reads the input channels
        // through `src`. It gives a mask for every frame size and sampling
        // rate this function is reached with, so `have_mask` is then true.
        let have_mask = match self.surround.as_mut() {
            Some(analysis) if masking => analysis.run(src, frame_size, &mut self.band_smr),
            _ => false,
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
            enc.set_bitrate(stream_bitrate(self.rates[s], enc.channels().count()));
            if let Some(bandwidth) = surround_bandwidth {
                enc.set_bandwidth(Some(bandwidth));
                if s < coupled {
                    // "To preserve the spatial image, force stereo CELT on
                    // coupled streams" (opus_multistream_encoder.c:953-958):
                    // CELT-only, and two coded channels, so that a stereo
                    // stream encoder does not code its input as mono at a
                    // low bitrate. Ambisonics streams are not forced to two
                    // channels.
                    enc.set_force_mode(Some(ForcedMode::CeltOnly));
                    enc.set_force_channels(Some(Channels::Stereo));
                }
            } else if self.mapping_type == MappingType::Ambisonics {
                enc.set_force_mode(Some(ForcedMode::CeltOnly));
            }
        }

        let mut band_log_e = [0.0f32; 2 * MASK_BANDS];
        let mut tot_size = 0usize;
        let streams_and_inputs = self.encoders.iter_mut().zip(&self.stream_inputs);
        for (s, (enc, &(c1, c2))) in streams_and_inputs.enumerate() {
            let last = s + 1 == streams;
            let stream_channels = enc.channels().count();
            // Copy the input of the stream: left and right interleaved for a
            // coupled stream.
            match c2 {
                Some(c2) => src.copy_pair(&mut self.buf, c1, c2, frame_size),
                None => src.copy(&mut self.buf, c1, frame_size),
            }
            if masking {
                // libopus sets the mask of every stream on every frame in
                // surround mode: the mask of the left channel, then that of
                // the right. D6: the values are copied. Without a mask the
                // stream encoder is told so, which also drops the mask of an
                // earlier frame.
                // The mask has 21 values per channel of the stream, the
                // length the stream encoder takes.
                let _ = enc.set_energy_mask(if have_mask {
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
                enc.set_bitrate(stream_bitrate(
                    cbr_last_stream_bitrate(curr_max, frame_size, fs)?,
                    stream_channels,
                ));
            }
            // libopus calls `opus_encode_native`: the frame size is chosen,
            // and an expert frame duration set on a stream encoder plays no
            // part. The stream's analysis reads the whole input (C
            // 1027-1028): channel `c1`, plus `c2` for a coupled stream,
            // whose sum it halves.
            let analysis_pcm = AnalysisPcm {
                pcm: T::analysis_pcm(src.pcm),
                channels: src.channels,
                c1,
                c2: match c2 {
                    Some(c2) => SecondChannel::Channel(c2),
                    None => SecondChannel::None,
                },
                frame_size: analysis_frame_size,
            };
            let len = enc.encode_impl(
                &self.buf[..frame_size * stream_channels],
                frame_size,
                &mut self.tmp[..curr_max],
                analysis_pcm,
                T::LSB_DEPTH,
                T::FLOAT_API,
            )?;

            // The stream encoder can return more than one frame (a 60 ms
            // CELT packet is three 20 ms frames, a 120 ms one six) and can
            // pad its packet.
            // As libopus does, pass it through the repacketizer, which
            // writes the frames anew with the extensions of the padding:
            // with the size of the last frame for every stream but the
            // last, and padded to the full size for the last stream of a
            // constant-bitrate packet.
            let mut rp = packet::Repacketizer::new();
            // C: a stream packet the repacketizer refuses is an internal
            // error.
            rp.cat(&self.tmp[..len]).map_err(|_| Error::INTERNAL_ERROR)?;
            let room = output
                .get_mut(tot_size..max_data_bytes)
                .ok_or(Error::BUFFER_TOO_SMALL)?;
            let framing = packet::Framing::from_flag(!last);
            let padding = packet::Padding::from_flag(!vbr && last);
            let written = rp.out_range_ext(0, rp.nb_frames(), room, framing, padding, &[])?;
            tot_size += written;
        }
        Ok(tot_size)
    }

    // --- Settings of the whole encoder ---

    /// Sets the total bitrate. libopus `OPUS_SET_BITRATE`.
    ///
    /// An explicit value is limited to 500 to 750 000 bits per second per
    /// input channel. The default is [`Bitrate::Auto`].
    pub fn set_bitrate(&mut self, bitrate: Bitrate) {
        self.bitrate_bps = match bitrate {
            Bitrate::Auto => OPUS_AUTO,
            Bitrate::Max => OPUS_BITRATE_MAX,
            // A layout has at least one channel and the value is positive,
            // so the limits apply and the clamp never fails.
            Bitrate::BitsPerSecond(bps) => {
                clamp_user_bitrate(Bitrate::bps_to_i32(bps), self.layout.channels())
                    .unwrap_or(self.bitrate_bps)
            }
        };
    }

    /// The bitrate that was set, after its limits were applied.
    pub fn bitrate_setting(&self) -> Bitrate {
        match self.bitrate_bps {
            OPUS_AUTO => Bitrate::Auto,
            OPUS_BITRATE_MAX => Bitrate::Max,
            bps => Bitrate::within_limits(bps),
        }
    }

    /// The sum of the bitrates of the streams, in bits per second. libopus
    /// `OPUS_GET_BITRATE`.
    ///
    /// After [`encode`](Self::encode) this is what the allocation gave the
    /// streams for that frame. As in libopus, it is the sum of
    /// [`OpusEncoder::bitrate`] over the streams: a stream counts with at
    /// most what a 1276-byte packet per frame carries. Before the first
    /// frame every stream is at [`Bitrate::Auto`] and counts with the
    /// bitrate a single-stream encoder picks by itself.
    pub fn bitrate(&self) -> i32 {
        let sum: i64 = self.encoders.iter().map(|enc| enc.bitrate() as i64).sum();
        sum.clamp(0, i32::MAX as i64) as i32
    }

    // --- Settings forwarded to every stream encoder ---

    /// Sets the complexity, 0 to 10, on every stream. libopus
    /// `OPUS_SET_COMPLEXITY`. The default is 9. A value above 10 is refused
    /// with [`BadArg`](crate::ErrorKind::BadArg) and no stream changes.
    pub fn set_complexity(&mut self, complexity: u8) -> Result<(), Error> {
        if complexity > 10 {
            return Err(Error::INVALID_COMPLEXITY);
        }
        for enc in &mut self.encoders {
            enc.set_complexity(complexity)?;
        }
        Ok(())
    }

    /// The complexity of the first stream. libopus `OPUS_GET_COMPLEXITY`.
    pub fn complexity(&self) -> u8 {
        self.encoders.first().map_or(0, |enc| enc.complexity())
    }

    /// Selects variable bitrate (`true`, the default) or constant bitrate
    /// (`false`) on every stream. libopus `OPUS_SET_VBR`.
    ///
    /// With constant bitrate every packet has the same size for the same
    /// frame duration.
    pub fn set_vbr(&mut self, vbr: bool) {
        for enc in &mut self.encoders {
            enc.set_vbr(vbr);
        }
    }

    /// Whether the first stream uses variable bitrate. libopus
    /// `OPUS_GET_VBR`. The encoder goes by this value for the whole packet.
    pub fn vbr(&self) -> bool {
        self.encoders.first().is_none_or(|enc| enc.vbr())
    }

    /// Keeps CELT-only packets of a variable bitrate close to the bitrate
    /// (`true`, the default) on every stream. libopus
    /// `OPUS_SET_VBR_CONSTRAINT`. See [`OpusEncoder::set_vbr_constraint`].
    pub fn set_vbr_constraint(&mut self, constrained: bool) {
        for enc in &mut self.encoders {
            enc.set_vbr_constraint(constrained);
        }
    }

    /// Whether the first stream constrains its variable bitrate. libopus
    /// `OPUS_GET_VBR_CONSTRAINT`.
    pub fn vbr_constraint(&self) -> bool {
        self.encoders.first().is_none_or(|enc| enc.vbr_constraint())
    }

    /// Sets in-band forward error correction (FEC) on every stream.
    /// libopus `OPUS_SET_INBAND_FEC`. Only streams coded with SILK carry it.
    /// See [`OpusEncoder::set_inband_fec`].
    pub fn set_inband_fec(&mut self, fec: InbandFec) {
        for enc in &mut self.encoders {
            enc.set_inband_fec(fec);
        }
    }

    /// The in-band FEC setting of the first stream. libopus
    /// `OPUS_GET_INBAND_FEC`.
    pub fn inband_fec(&self) -> InbandFec {
        self.encoders.first().map_or(InbandFec::Off, |enc| enc.inband_fec())
    }

    /// Says what the input is, on every stream. libopus `OPUS_SET_SIGNAL`.
    /// See [`OpusEncoder::set_signal`].
    pub fn set_signal(&mut self, signal: Signal) {
        for enc in &mut self.encoders {
            enc.set_signal(signal);
        }
    }

    /// The signal type of the first stream. libopus `OPUS_GET_SIGNAL`.
    pub fn signal(&self) -> Signal {
        self.encoders.first().map_or(Signal::Auto, |enc| enc.signal())
    }

    /// Sets the precision of the input in bits, 8 to 24, on every stream.
    /// libopus `OPUS_SET_LSB_DEPTH`. See [`OpusEncoder::set_lsb_depth`]. Any
    /// other value is refused with [`BadArg`](crate::ErrorKind::BadArg),
    /// and the setting stays.
    pub fn set_lsb_depth(&mut self, depth: u8) -> Result<(), Error> {
        if !(8..=24).contains(&depth) {
            return Err(Error::INVALID_LSB_DEPTH);
        }
        for enc in &mut self.encoders {
            enc.set_lsb_depth(depth)?;
        }
        Ok(())
    }

    /// The input precision of the first stream. libopus
    /// `OPUS_GET_LSB_DEPTH`.
    pub fn lsb_depth(&self) -> u8 {
        self.encoders.first().map_or(24, |enc| enc.lsb_depth())
    }

    /// Turns discontinuous transmission (DTX) on or off on every stream.
    /// libopus `OPUS_SET_DTX`. See [`OpusEncoder::set_dtx`]. Each stream
    /// decides on its own: a stream it leaves out takes 1 byte in the
    /// packet, 2 when it is not the last stream.
    pub fn set_dtx(&mut self, enabled: bool) {
        for enc in &mut self.encoders {
            enc.set_dtx(enabled);
        }
    }

    /// Whether DTX is on in the first stream. libopus `OPUS_GET_DTX`.
    ///
    /// libopus has no `OPUS_GET_IN_DTX` for a multistream encoder.
    /// [`OpusEncoder::in_dtx`] of each stream encoder
    /// ([`stream_encoder`](Self::stream_encoder)) tells it per stream.
    pub fn dtx(&self) -> bool {
        self.encoders.first().is_some_and(|enc| enc.dtx())
    }

    /// Sets the expected packet loss, 0 to 100 percent, on every stream.
    /// libopus `OPUS_SET_PACKET_LOSS_PERC`. A value above 100 is refused
    /// with [`BadArg`](crate::ErrorKind::BadArg) and no stream changes.
    pub fn set_packet_loss_perc(&mut self, percent: u8) -> Result<(), Error> {
        if percent > 100 {
            return Err(Error::INVALID_PACKET_LOSS);
        }
        for enc in &mut self.encoders {
            enc.set_packet_loss_perc(percent)?;
        }
        Ok(())
    }

    /// The expected packet loss of the first stream in percent. libopus
    /// `OPUS_GET_PACKET_LOSS_PERC`.
    pub fn packet_loss_perc(&self) -> u8 {
        self.encoders.first().map_or(0, |enc| enc.packet_loss_perc())
    }

    /// Sets the audio bandwidth every stream codes. libopus
    /// `OPUS_SET_BANDWIDTH`. `None`, the default, lets each stream decide
    /// from its bitrate (libopus `OPUS_AUTO`).
    ///
    /// In surround mode the encoder sets the bandwidth of every stream
    /// again on every frame, exactly like libopus, so this has no effect
    /// there. The LFE stream is always narrowband.
    pub fn set_bandwidth(&mut self, bandwidth: Option<Bandwidth>) {
        self.user_bandwidth = bandwidth;
        for enc in &mut self.encoders {
            enc.set_bandwidth(bandwidth);
        }
    }

    /// What [`set_bandwidth`](Self::set_bandwidth) was given last: `None`
    /// when each stream chooses the bandwidth itself.
    ///
    /// This is the setting. libopus `OPUS_GET_BANDWIDTH` reports something
    /// else: the bandwidth the first stream coded last. The TOC byte of a
    /// stream of the packet tells that.
    pub fn bandwidth_setting(&self) -> Option<Bandwidth> {
        self.user_bandwidth
    }

    /// Sets the widest audio bandwidth the automatic choice of a stream may
    /// take. libopus `OPUS_SET_MAX_BANDWIDTH`. The default is
    /// [`Bandwidth::Fullband`].
    ///
    /// A bandwidth set with [`set_bandwidth`](Self::set_bandwidth), or by
    /// the encoder in surround mode, is not limited by it.
    pub fn set_max_bandwidth(&mut self, bandwidth: Bandwidth) {
        for enc in &mut self.encoders {
            enc.set_max_bandwidth(bandwidth);
        }
        self.max_bandwidth = bandwidth;
    }

    /// What [`set_max_bandwidth`](Self::set_max_bandwidth) was given last.
    pub fn max_bandwidth(&self) -> Bandwidth {
        self.max_bandwidth
    }

    /// Sets the frame duration of the packets. libopus
    /// `OPUS_SET_EXPERT_FRAME_DURATION`, which the multistream encoder keeps
    /// for itself. See [`OpusEncoder::set_expert_frame_duration`]: with any
    /// value but [`FrameDuration::Argument`] each [`encode`](Self::encode)
    /// codes a frame of that duration, and its `frame_size` must be at
    /// least that long.
    ///
    /// The setting stays across [`reset_state`](Self::reset_state).
    pub fn set_expert_frame_duration(&mut self, duration: FrameDuration) {
        self.variable_duration = duration;
    }

    /// libopus `OPUS_GET_EXPERT_FRAME_DURATION`: the value set with
    /// [`set_expert_frame_duration`](Self::set_expert_frame_duration).
    pub fn expert_frame_duration(&self) -> FrameDuration {
        self.variable_duration
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

    /// Forces the number of coded channels of every stream. libopus
    /// `OPUS_SET_FORCE_CHANNELS`. See [`OpusEncoder::set_force_channels`].
    ///
    /// The value goes to every stream. A mono stream codes mono whatever
    /// is set, so `Some(Channels::Stereo)` forces stereo on the coupled
    /// streams only. libopus sets the coupled streams and then refuses
    /// `Some(Stereo)` at the first mono stream with `OPUS_BAD_ARG`; opus1
    /// sets every stream and does not fail.
    ///
    /// In surround mode the encoder forces two channels on every coupled
    /// stream on every frame, as libopus does.
    pub fn set_force_channels(&mut self, channels: Option<Channels>) {
        for enc in &mut self.encoders {
            enc.set_force_channels(channels);
        }
    }

    /// The forced channel count of the first stream. libopus
    /// `OPUS_GET_FORCE_CHANNELS`.
    pub fn force_channels(&self) -> Option<Channels> {
        self.encoders.first().and_then(|enc| enc.force_channels())
    }

    /// Turns off prediction from earlier frames in every stream. libopus
    /// `OPUS_SET_PREDICTION_DISABLED`. See
    /// [`OpusEncoder::set_prediction_disabled`].
    pub fn set_prediction_disabled(&mut self, disabled: bool) {
        for enc in &mut self.encoders {
            enc.set_prediction_disabled(disabled);
        }
    }

    /// Whether prediction is disabled in the first stream. libopus
    /// `OPUS_GET_PREDICTION_DISABLED`.
    pub fn prediction_disabled(&self) -> bool {
        self.encoders.first().is_some_and(|enc| enc.prediction_disabled())
    }

    /// Disables phase inversion in the intensity stereo of every stream.
    /// libopus `OPUS_SET_PHASE_INVERSION_DISABLED`. See
    /// [`OpusEncoder::set_phase_inversion_disabled`]. It changes only the
    /// coupled streams; a mono stream has no intensity stereo.
    pub fn set_phase_inversion_disabled(&mut self, disabled: bool) {
        for enc in &mut self.encoders {
            enc.set_phase_inversion_disabled(disabled);
        }
    }

    /// Whether phase inversion is disabled in the first stream. libopus
    /// `OPUS_GET_PHASE_INVERSION_DISABLED`.
    pub fn phase_inversion_disabled(&self) -> bool {
        self.encoders.first().is_some_and(|enc| enc.phase_inversion_disabled())
    }

    /// The voice ratio of the first stream, in percent, or `None` when it
    /// is not known (libopus -1). libopus `OPUS_GET_VOICE_RATIO`; see
    /// [`OpusEncoder::voice_ratio`]. libopus has no multistream
    /// `OPUS_SET_VOICE_RATIO`.
    pub fn voice_ratio(&self) -> Option<u8> {
        self.encoders.first().and_then(|enc| enc.voice_ratio())
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
        let mut enc = MultistreamEncoder::new_surround(
            SampleRate::Hz48000,
            6,
            1,
            Application::Audio,
        )
        .unwrap();
        let pcm: Vec<f32> = (0..960 * 6)
            .map(|k| ((k * 7 % 97) as f32 - 48.0) / 200.0)
            .collect();
        let mut out = vec![0u8; 6000];
        enc.encode(&pcm, 960, &mut out).unwrap();
        for stream in &enc.encoders {
            let mask = stream.energy_mask().expect("a surround stream has a mask");
            assert_eq!(mask.len(), MASK_BANDS * stream.channels().count());
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
                MultistreamEncoder::new_surround(SampleRate::Hz48000, channels, 1, Application::Audio).unwrap();
            assert_eq!(enc.surround.is_some(), channels > 2, "{channels} channels");
            assert_eq!(enc.band_smr.len(), if channels > 2 { MASK_BANDS * channels } else { 0 });
        }
        for (channels, family) in [(3, 255), (6, 255), (4, 2), (6, 2), (9, 2)] {
            let enc = MultistreamEncoder::new_surround(
                SampleRate::Hz48000,
                channels,
                family,
                Application::Audio,
            )
            .unwrap();
            assert!(enc.surround.is_none(), "family {family}, {channels} channels");
        }
        let enc =
            MultistreamEncoder::new(SampleRate::Hz48000, 6, 4, 2, &[0, 4, 1, 2, 3, 5], Application::Audio)
                .unwrap();
        assert!(enc.surround.is_none(), "the generic constructor");
        assert_eq!(enc.lfe_stream(), None);
    }

    #[test]
    fn stream_input_channels_follow_the_layout() {
        // 5.1 in Vorbis order: front left, centre, front right, rear left,
        // rear right, LFE. libopus codes it as mapping [0, 4, 1, 2, 3, 5].
        let enc = MultistreamEncoder::new_surround(
            SampleRate::Hz48000,
            6,
            1,
            Application::Audio,
        )
        .unwrap();
        assert_eq!(enc.layout().mapping(), &[0, 4, 1, 2, 3, 5]);
        let channels = |s| stream_input_channels(enc.layout(), s).unwrap();
        assert_eq!(channels(0), (0, Some(2)));
        assert_eq!(channels(1), (3, Some(4)));
        assert_eq!(channels(2), (1, None));
        assert_eq!(channels(3), (5, None));
        // 7.1: mapping [0, 6, 1, 2, 3, 4, 5, 7].
        let enc = MultistreamEncoder::new_surround(
            SampleRate::Hz48000,
            8,
            1,
            Application::Audio,
        )
        .unwrap();
        let channels = |s| stream_input_channels(enc.layout(), s).unwrap();
        assert_eq!(channels(0), (0, Some(2)));
        assert_eq!(channels(1), (3, Some(4)));
        assert_eq!(channels(2), (5, Some(6)));
        assert_eq!(channels(3), (1, None));
        assert_eq!(channels(4), (7, None));
        // A generic layout. Two channels name the mono stream: the first one
        // is read. Channel 2 is not coded.
        let enc = MultistreamEncoder::new(
            SampleRate::Hz48000,
            5,
            2,
            1,
            &[1, 0, 255, 2, 2],
            Application::Audio,
        )
        .unwrap();
        let channels = |s| stream_input_channels(enc.layout(), s).unwrap();
        assert_eq!(channels(0), (1, Some(0)));
        assert_eq!(channels(1), (3, None));
    }

    #[test]
    fn stream_bitrate_follows_the_stream_encoder_control() {
        let bps = |value| Bitrate::BitsPerSecond(core::num::NonZeroU32::new(value).unwrap());
        assert_eq!(stream_bitrate(1, 1), bps(500));
        assert_eq!(stream_bitrate(500, 2), bps(500));
        assert_eq!(stream_bitrate(64_000, 1), bps(64_000));
        assert_eq!(stream_bitrate(750_001, 1), bps(750_000));
        assert_eq!(stream_bitrate(2_000_000, 2), bps(1_500_000));
        assert_eq!(stream_bitrate(i32::MAX, 2), bps(1_500_000));
        // What the stream encoder keeps of it is the same value.
        let mut enc =
            OpusEncoder::new(SampleRate::Hz48000, Channels::Stereo, Application::Audio).unwrap();
        for value in [i32::MIN, 0, 1, 500, 64_000, 2_000_000, i32::MAX] {
            enc.set_bitrate(stream_bitrate(value, 2));
            assert_eq!(enc.bitrate_setting(), stream_bitrate(value, 2), "{value}");
        }
    }

    #[test]
    fn bitrate_setting_round_trips() {
        let mut enc = MultistreamEncoder::new_surround(
            SampleRate::Hz48000,
            6,
            1,
            Application::Audio,
        )
        .unwrap();
        let bps = |value| Bitrate::BitsPerSecond(core::num::NonZeroU32::new(value).unwrap());
        assert_eq!(enc.bitrate_setting(), Bitrate::Auto);
        enc.set_bitrate(Bitrate::Max);
        assert_eq!(enc.bitrate_setting(), Bitrate::Max);
        enc.set_bitrate(bps(256_000));
        assert_eq!(enc.bitrate_setting(), bps(256_000));
        // Limited to 500 and 750 000 bits per second per channel.
        enc.set_bitrate(bps(1));
        assert_eq!(enc.bitrate_setting(), bps(3000));
        enc.set_bitrate(bps(i32::MAX as u32));
        assert_eq!(enc.bitrate_setting(), bps(4_500_000));
        // Above the `opus_int32` of libopus the same limit applies.
        enc.set_bitrate(bps(u32::MAX));
        assert_eq!(enc.bitrate_setting(), bps(4_500_000));
    }

    #[test]
    fn frame_sizes_are_those_of_the_stream_encoder() {
        for &rate in SampleRate::ALL {
            let enc = MultistreamEncoder::new_surround(rate, 2, 1, Application::Audio).unwrap();
            let fs = rate.hz();
            let ms = |tenths: usize| fs as usize * tenths / 10_000;
            for tenths in [25, 50, 100, 200, 400, 600, 800, 1000, 1200] {
                assert_eq!(
                    enc.check_input(usize::MAX, ms(tenths)),
                    Ok(ms(tenths)),
                    "{fs} Hz, {tenths}"
                );
            }
            // Sizes that are no frame at all.
            for tenths in [150, 300, 1400] {
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

    #[test]
    fn the_expert_frame_duration_picks_the_frame() {
        let mut enc =
            MultistreamEncoder::new_surround(SampleRate::Hz48000, 2, 1, Application::Audio).unwrap();
        assert_eq!(enc.expert_frame_duration(), FrameDuration::Argument);
        enc.set_expert_frame_duration(FrameDuration::Ms20);
        // The input must hold what the caller passes; the frame is 20 ms.
        assert_eq!(enc.check_input(2 * 2880, 2880), Ok(960));
        assert!(enc.check_input(2 * 2880 - 1, 2880).is_err());
        // Less than 20 ms is refused.
        assert!(enc.check_input(usize::MAX, 480).is_err());
        // A size that is no frame is fine when 20 ms fit into it.
        assert_eq!(enc.check_input(usize::MAX, 1000), Ok(960));
        enc.reset_state();
        assert_eq!(enc.expert_frame_duration(), FrameDuration::Ms20);
    }
}
