//! Multistream decoder: a port of libopus 1.6 `opus_multistream_decoder.c`.
//!
//! A multistream packet holds one ordinary Opus packet per stream. Every
//! stream has its own [`OpusDecoder`] with 1 or 2 channels. This decoder
//! decodes the streams one after the other and copies each decoded channel
//! to the output channels that the [`ChannelLayout`] names.
//!
//! Memory is allocated in the constructor only. Decoding allocates nothing.

use crate::Error;
use super::layout::{ChannelLayout, ChannelsByEntry, SILENT_CHANNEL};
use super::{ChannelMappingTable, ChannelSink, Sample};
use crate::{Channels, OpusDecoder, SampleRate, packet};

/// Decoder for a multistream Opus stream: libopus `OpusMSDecoder`.
///
/// Samples are interleaved: sample `i` of output channel `c` is at index
/// `i * channels + c`. A frame size is always in samples per channel.
pub struct MultistreamDecoder {
    layout: ChannelLayout,
    /// The output channels of each decoded channel, and the silent ones.
    outputs: ChannelsByEntry,
    sampling_rate: SampleRate,
    /// One decoder per stream. The first `layout.coupled_streams()` of them
    /// have 2 channels, the others 1.
    decoders: Vec<OpusDecoder>,
    /// Output of one stream: room for 120 ms of stereo, the longest packet.
    buf: Vec<f32>,
}

/// Interleaved output: libopus `opus_copy_channel_out_float` and
/// `opus_copy_channel_out_short`. 16-bit samples are rounded and limited to
/// the 16-bit range.
struct InterleavedSink<'a, T> {
    out: &'a mut [T],
    channels: usize,
}

impl<T: Sample> ChannelSink for InterleavedSink<'_, T> {
    #[inline]
    fn put(&mut self, dst_channel: usize, src: Option<(&[f32], usize)>, frame_size: usize) {
        let Some(out) = self.out.get_mut(dst_channel..) else {
            return;
        };
        let dst = out.iter_mut().step_by(self.channels.max(1)).take(frame_size);
        match src {
            Some((samples, stride)) => {
                for (d, &s) in dst.zip(samples.iter().step_by(stride.max(1))) {
                    *d = T::from_f32(s);
                }
            }
            None => {
                for d in dst {
                    *d = T::ZERO;
                }
            }
        }
    }
}

impl MultistreamDecoder {
    /// Builds a decoder from the channel mapping table of an Ogg Opus
    /// header.
    ///
    /// The table is checked like the arguments of
    /// [`with_layout`](Self::with_layout). In addition, `table.mapping` must
    /// hold exactly `table.channels` entries.
    pub fn new(sampling_rate: SampleRate, table: ChannelMappingTable) -> Result<Self, Error> {
        if table.mapping.len() != table.channels as usize {
            return Err(Error::bad_arg(
                "multistream layout: the mapping table length does not match the channel count",
            ));
        }
        Self::with_layout(
            sampling_rate,
            table.channels as usize,
            table.stream_count as usize,
            table.coupled_count as usize,
            &table.mapping,
        )
    }

    /// libopus `opus_multistream_decoder_create`.
    ///
    /// - `sampling_rate`: output rate.
    /// - `channels`: number of output channels, 1 to 255.
    /// - `streams`: number of streams in a packet, at least 1.
    /// - `coupled_streams`: how many of the streams are stereo. They are the
    ///   first streams of a packet. `streams + coupled_streams` is at most
    ///   255.
    /// - `mapping`: one entry per output channel, see [`ChannelLayout`]. An
    ///   entry is the index of a decoded channel, or [`SILENT_CHANNEL`] for a
    ///   channel that stays silent. Several output channels may name the
    ///   same decoded channel.
    ///   Entries after the first `channels` are ignored.
    pub fn with_layout(
        sampling_rate: SampleRate,
        channels: usize,
        streams: usize,
        coupled_streams: usize,
        mapping: &[u8],
    ) -> Result<Self, Error> {
        let layout = ChannelLayout::new(channels, streams, coupled_streams, mapping)?;
        let mut decoders = Vec::with_capacity(layout.streams());
        for s in 0..layout.streams() {
            let stream_channels =
                if s < layout.coupled_streams() { Channels::Stereo } else { Channels::Mono };
            decoders.push(OpusDecoder::new(sampling_rate, stream_channels)?);
        }
        let max_frame = sampling_rate.max_frame_size();
        Ok(Self {
            outputs: ChannelsByEntry::new(&layout),
            layout,
            sampling_rate,
            decoders,
            buf: vec![0.0; 2 * max_frame],
        })
    }

    /// The channel layout this decoder was built with.
    pub fn layout(&self) -> &ChannelLayout {
        &self.layout
    }

    /// The sampling rate of the output.
    pub fn sampling_rate(&self) -> SampleRate {
        self.sampling_rate
    }

    /// Decodes one multistream packet into interleaved `f32` samples:
    /// libopus `opus_multistream_decode_float`.
    ///
    /// `frame_size` is the room in `out` in samples per channel. It must be
    /// at least the duration of the packet, and `out` must hold
    /// `frame_size * channels` samples. A `frame_size` above 120 ms counts
    /// as 120 ms. Returns the number of samples per channel written.
    ///
    /// An empty `packet` stands for a lost packet. Every stream then conceals
    /// `frame_size` samples, which must be a multiple of 2.5 ms.
    ///
    /// The packet is checked as a whole before any stream is decoded, so a
    /// malformed packet is an error that leaves the decoder as it was. The
    /// output is not clipped, as in [`OpusDecoder::decode`].
    pub fn decode(
        &mut self,
        packet: &[u8],
        frame_size: usize,
        out: &mut [f32],
    ) -> Result<usize, Error> {
        let frame_size = self.usable_frame_size(frame_size, out.len())?;
        let channels = self.layout.channels();
        self.decode_native(packet, frame_size, &mut InterleavedSink { out, channels }, false, false)
    }

    /// [`decode`](Self::decode) with 16-bit output: libopus
    /// `opus_multistream_decode`.
    ///
    /// As in libopus, each stream is soft-clipped before it is rounded to 16
    /// bits, so that a peak beyond full scale is rounded off instead of cut
    /// ([`pcm_soft_clip`](crate::pcm_soft_clip)).
    pub fn decode_i16(
        &mut self,
        packet: &[u8],
        frame_size: usize,
        out: &mut [i16],
    ) -> Result<usize, Error> {
        let frame_size = self.usable_frame_size(frame_size, out.len())?;
        let channels = self.layout.channels();
        self.decode_native(packet, frame_size, &mut InterleavedSink { out, channels }, false, true)
    }

    /// Rebuilds a lost packet from the forward error correction (FEC) data
    /// of the packet after it: libopus `opus_multistream_decode_float` with
    /// `decode_fec = 1`.
    ///
    /// `packet` is the packet received after the lost one. `frame_size` is
    /// the duration that was lost in samples per channel, a multiple of
    /// 2.5 ms. It must be at least the duration of `packet`, as in libopus.
    /// Exactly `frame_size` samples per channel are written.
    ///
    /// Each stream decides by itself: a stream whose packet carries a copy
    /// of the lost frame decodes it, every other stream conceals the loss
    /// (see [`OpusDecoder::decode_fec`]). Decode `packet` itself with
    /// [`decode`](Self::decode) afterwards.
    pub fn decode_fec(
        &mut self,
        packet: &[u8],
        frame_size: usize,
        out: &mut [f32],
    ) -> Result<usize, Error> {
        let frame_size = self.usable_frame_size(frame_size, out.len())?;
        let channels = self.layout.channels();
        self.decode_native(packet, frame_size, &mut InterleavedSink { out, channels }, true, false)
    }

    /// [`decode_fec`](Self::decode_fec) with 16-bit output: libopus
    /// `opus_multistream_decode` with `decode_fec = 1`.
    pub fn decode_fec_i16(
        &mut self,
        packet: &[u8],
        frame_size: usize,
        out: &mut [i16],
    ) -> Result<usize, Error> {
        let frame_size = self.usable_frame_size(frame_size, out.len())?;
        let channels = self.layout.channels();
        self.decode_native(packet, frame_size, &mut InterleavedSink { out, channels }, true, true)
    }

    /// libopus `OPUS_GET_FINAL_RANGE` of a multistream decoder: the final
    /// range values of all stream decoders combined with XOR. After a packet
    /// was decoded it equals the value the multistream encoder reports for
    /// that packet.
    /// Decodes one multistream packet into interleaved 24-bit samples held
    /// in 32-bit integers: libopus `opus_multistream_decode24`. Each sample
    /// is scaled by 2^23 and rounded, without soft clipping and without a
    /// limit to 24 bits; see [`OpusDecoder::decode24`].
    pub fn decode24(
        &mut self,
        packet: &[u8],
        frame_size: usize,
        out: &mut [i32],
    ) -> Result<usize, Error> {
        let frame_size = self.usable_frame_size(frame_size, out.len())?;
        let channels = self.layout.channels();
        self.decode_native(packet, frame_size, &mut InterleavedSink { out, channels }, false, false)
    }

    /// [`decode_fec`](Self::decode_fec) with 24-bit output: libopus
    /// `opus_multistream_decode24` with `decode_fec = 1`.
    pub fn decode_fec24(
        &mut self,
        packet: &[u8],
        frame_size: usize,
        out: &mut [i32],
    ) -> Result<usize, Error> {
        let frame_size = self.usable_frame_size(frame_size, out.len())?;
        let channels = self.layout.channels();
        self.decode_native(packet, frame_size, &mut InterleavedSink { out, channels }, true, false)
    }

    pub fn final_range(&self) -> u32 {
        self.decoders.iter().fold(0, |acc, dec| acc ^ dec.final_range())
    }

    /// libopus `OPUS_RESET_STATE`: every stream decoder forgets the stream
    /// decoded so far. The layout and the sampling rate stay.
    pub fn reset_state(&mut self) {
        for dec in &mut self.decoders {
            dec.reset_state();
        }
    }

    /// libopus `OPUS_SET_COMPLEXITY` of a multistream decoder: sets the decoder
    /// complexity of every stream decoder (see
    /// [`OpusDecoder::set_complexity`]). A value above 10 is refused with
    /// [`BadArg`](crate::ErrorKind::BadArg) and no stream changes.
    pub fn set_complexity(&mut self, complexity: u8) -> Result<(), Error> {
        if complexity > 10 {
            return Err(Error::INVALID_COMPLEXITY);
        }
        for dec in &mut self.decoders {
            dec.set_complexity(complexity)?;
        }
        Ok(())
    }

    /// libopus `OPUS_GET_COMPLEXITY` of a multistream decoder: the
    /// complexity of the first stream decoder.
    pub fn complexity(&self) -> u8 {
        self.decoders.first().map_or(0, OpusDecoder::complexity)
    }

    /// Sets the output gain of every stream, in 1/256 dB (Q8 dB). libopus
    /// `OPUS_SET_GAIN`. See [`OpusDecoder::set_gain`].
    pub fn set_gain(&mut self, gain: i16) {
        for dec in &mut self.decoders {
            dec.set_gain(gain);
        }
    }

    /// The output gain of the first stream. libopus `OPUS_GET_GAIN`.
    pub fn gain(&self) -> i16 {
        self.decoders.first().map_or(0, OpusDecoder::gain)
    }

    /// Disables phase inversion in the intensity stereo of every stream.
    /// libopus `OPUS_SET_PHASE_INVERSION_DISABLED`. See
    /// [`OpusDecoder::set_phase_inversion_disabled`].
    pub fn set_phase_inversion_disabled(&mut self, disabled: bool) {
        for dec in &mut self.decoders {
            dec.set_phase_inversion_disabled(disabled);
        }
    }

    /// Whether phase inversion is disabled in the first stream. libopus
    /// `OPUS_GET_PHASE_INVERSION_DISABLED`. At the start it is `false` when
    /// the first stream is coupled (stereo) and `true` when it is mono, as
    /// in libopus.
    pub fn phase_inversion_disabled(&self) -> bool {
        self.decoders.first().is_some_and(OpusDecoder::phase_inversion_disabled)
    }

    /// [`OpusDecoder::set_osce_bwe`] for every stream decoder. libopus's
    /// multistream decoder does not forward `OPUS_SET_OSCE_BWE`; there each
    /// stream decoder must be set on its own. Without the cargo feature
    /// `osce-bwe` this returns
    /// [`Unimplemented`](crate::ErrorKind::Unimplemented).
    pub fn set_osce_bwe(&mut self, enabled: bool) -> Result<(), Error> {
        for dec in &mut self.decoders {
            dec.set_osce_bwe(enabled)?;
        }
        Ok(())
    }

    /// [`OpusDecoder::set_dnn_blob`] for every stream decoder: each one loads
    /// the model groups of this build that `blob` holds completely. Returns
    /// those groups, or [`BadArg`](crate::ErrorKind::BadArg) when the blob
    /// holds no complete group of this build, and then no stream changes.
    /// libopus's multistream decoder does not forward `OPUS_SET_DNN_BLOB`;
    /// there each stream decoder must load the blob on its own.
    #[cfg(feature = "dnn")]
    pub fn set_dnn_blob(&mut self, blob: &crate::DnnBlob) -> Result<crate::DnnModels, Error> {
        // Every stream decoder has the same build, so the first one's
        // result is everyone's: when it fails, none would load.
        let mut loaded = Err(crate::dnn::models::ERR_NO_MODEL);
        for dec in &mut self.decoders {
            loaded = Ok(dec.set_dnn_blob(blob)?);
        }
        loaded
    }

    /// The decoder of stream `stream`: libopus
    /// `OPUS_MULTISTREAM_GET_DECODER_STATE`. `None` when there is no such
    /// stream.
    pub fn stream_decoder(&self, stream: usize) -> Option<&OpusDecoder> {
        self.decoders.get(stream)
    }

    /// Like [`stream_decoder`](Self::stream_decoder), for changing the
    /// decoder of one stream.
    pub fn stream_decoder_mut(&mut self, stream: usize) -> Option<&mut OpusDecoder> {
        self.decoders.get_mut(stream)
    }

    /// The first steps of libopus `opus_multistream_decode_native`: reject a
    /// frame size of 0 and limit it to 120 ms. Then check that the output,
    /// which has room for `out_len` samples, holds that many samples per
    /// channel.
    fn usable_frame_size(&self, frame_size: usize, out_len: usize) -> Result<usize, Error> {
        if frame_size == 0 {
            return Err(Error::INVALID_FRAME_SIZE);
        }
        // C: "Limit frame_size to avoid excessive stack allocations."
        let frame_size = frame_size.min(self.sampling_rate.max_frame_size());
        if out_len / self.layout.channels() < frame_size {
            return Err(Error::BUFFER_TOO_SMALL);
        }
        Ok(frame_size)
    }

    /// libopus `opus_multistream_decode_native`. `frame_size` has passed
    /// [`usable_frame_size`](Self::usable_frame_size).
    fn decode_native<K: ChannelSink>(
        &mut self,
        packet: &[u8],
        mut frame_size: usize,
        sink: &mut K,
        decode_fec: bool,
        soft_clip: bool,
    ) -> Result<usize, Error> {
        let Self { layout, outputs, sampling_rate, decoders, buf } = self;
        let streams = layout.streams();
        let coupled_streams = layout.coupled_streams();

        let do_plc = packet.is_empty();
        if !do_plc {
            // Every stream takes at least 1 byte, and every stream but the
            // last at least 2.
            if packet.len() < 2 * streams - 1 {
                return Err(Error::INVALID_PACKET);
            }
            // Nothing is decoded before the whole packet has been checked.
            let samples = packet::multistream_validate(packet, streams, *sampling_rate)?;
            if samples > frame_size {
                return Err(Error::FRAME_SIZE_TOO_SMALL);
            }
        }

        let mut rest = packet;
        for (s, dec) in decoders.iter_mut().enumerate() {
            let stream_channels = if s < coupled_streams { 2 } else { 1 };
            if !do_plc && rest.is_empty() {
                return Err(Error::internal(
                    "internal error: the packet ended before the last stream",
                ));
            }
            // Every stream but the last says how long it is.
            let self_delimited = s != streams - 1;
            let stream_buf = buf
                .get_mut(..frame_size * stream_channels)
                .ok_or(Error::internal("internal error: stream buffer too small"))?;
            let (samples, used) = dec.decode_native(
                rest,
                frame_size,
                stream_buf,
                decode_fec,
                self_delimited,
                soft_clip,
            )?;
            if !do_plc {
                rest = rest.get(used..).unwrap_or(&[]);
            }
            if samples == 0 {
                return Ok(0);
            }
            frame_size = samples;
            let stream_buf = &buf[..frame_size * stream_channels];
            // Copy each decoded channel to every output channel that uses
            // it: the left and the right channel of a coupled stream, or
            // the one channel of a mono stream.
            if s < coupled_streams {
                for &chan in outputs.with_entry(2 * s) {
                    sink.put(chan as usize, Some((stream_buf, 2)), frame_size);
                }
                for &chan in outputs.with_entry(2 * s + 1) {
                    sink.put(chan as usize, Some((&stream_buf[1..], 2)), frame_size);
                }
            } else {
                for &chan in outputs.with_entry(s + coupled_streams) {
                    sink.put(chan as usize, Some((stream_buf, 1)), frame_size);
                }
            }
        }
        // Channels that no stream feeds are silent.
        for &chan in outputs.with_entry(SILENT_CHANNEL as usize) {
            sink.put(chan as usize, None, frame_size);
        }
        Ok(frame_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table(channels: u8, stream_count: u8, coupled_count: u8, mapping: &[u8]) -> ChannelMappingTable {
        ChannelMappingTable { channels, stream_count, coupled_count, mapping: mapping.to_vec() }
    }

    #[test]
    fn new_builds_one_decoder_per_stream() {
        let ms = MultistreamDecoder::new(
            SampleRate::Hz48000,
            table(6, 4, 2, &[0, 4, 1, 2, 3, 5]),
        )
        .unwrap();
        assert_eq!(ms.decoders.len(), 4);
        let stream_channels: Vec<Channels> = ms.decoders.iter().map(|d| d.channels()).collect();
        use Channels::{Mono, Stereo};
        assert_eq!(stream_channels, [Stereo, Stereo, Mono, Mono]);
        assert_eq!(ms.layout().channels(), 6);
        assert_eq!(ms.layout().mapping(), &[0, 4, 1, 2, 3, 5]);
        assert_eq!(ms.sampling_rate(), SampleRate::Hz48000);
        assert_eq!(ms.buf.len(), 2 * 5760);
        assert!(ms.stream_decoder(3).is_some());
        assert!(ms.stream_decoder(4).is_none());
    }

    #[test]
    fn new_rejects_invalid_tables() {
        // More coupled streams than streams.
        assert!(MultistreamDecoder::new(SampleRate::Hz48000, table(2, 1, 2, &[0, 1])).is_err());
        // A mapping entry that names a decoded channel that does not exist.
        // The old decoder accepted this table and panicked while decoding.
        assert!(MultistreamDecoder::new(SampleRate::Hz48000, table(1, 1, 0, &[5])).is_err());
        // No stream, no channel.
        assert!(MultistreamDecoder::new(SampleRate::Hz48000, table(1, 0, 0, &[0])).is_err());
        assert!(MultistreamDecoder::new(SampleRate::Hz48000, table(0, 1, 0, &[])).is_err());
        // The mapping does not have one entry per channel.
        assert!(MultistreamDecoder::new(SampleRate::Hz48000, table(2, 1, 1, &[0])).is_err());
        assert!(MultistreamDecoder::new(SampleRate::Hz48000, table(2, 1, 1, &[0, 1, 0])).is_err());
        assert!(MultistreamDecoder::new(SampleRate::Hz48000, table(2, 1, 1, &[0, 1])).is_ok());
    }

    #[test]
    fn the_buffer_holds_120_ms_of_stereo_at_every_rate() {
        let wanted = [1920, 2880, 3840, 5760, 11520];
        for (fs, want) in SampleRate::ALL.iter().copied().zip(wanted) {
            let ms = MultistreamDecoder::with_layout(fs, 1, 1, 0, &[0]).unwrap();
            assert_eq!(ms.buf.len(), want, "{fs:?}");
        }
    }

    #[test]
    fn sinks_write_one_channel_and_leave_the_others() {
        let src = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut out = [9.0f32; 9];
        let mut sink = InterleavedSink { out: &mut out, channels: 3 };
        // Every second sample of `src`, to channel 1.
        sink.put(1, Some((&src, 2)), 3);
        assert_eq!(out, [9.0, 1.0, 9.0, 9.0, 3.0, 9.0, 9.0, 5.0, 9.0]);
        let mut sink = InterleavedSink { out: &mut out, channels: 3 };
        // The same samples starting at the second one, to channel 2.
        sink.put(2, Some((&src[1..], 2)), 3);
        sink.put(0, None, 2);
        assert_eq!(out, [0.0, 1.0, 2.0, 0.0, 3.0, 4.0, 9.0, 5.0, 6.0]);

        let src = [0.5f32, -0.5, 2.0, -2.0];
        let mut out = [7i16; 8];
        let mut sink = InterleavedSink { out: &mut out, channels: 2 };
        sink.put(0, Some((&src, 1)), 4);
        assert_eq!(out, [16384, 7, -16384, 7, 32767, 7, -32768, 7]);
        let mut sink = InterleavedSink { out: &mut out, channels: 2 };
        sink.put(1, None, 3);
        assert_eq!(out, [16384, 0, -16384, 0, 32767, 0, -32768, 7]);
    }

    #[test]
    fn sinks_do_not_panic_on_arguments_that_do_not_fit() {
        let src = [1.0f32; 4];
        let mut out = [0.0f32; 4];
        let mut sink = InterleavedSink { out: &mut out, channels: 2 };
        // A channel beyond the output, more samples than there are, stride 0.
        sink.put(9, Some((&src, 1)), 4);
        sink.put(1, Some((&src, 1)), 100);
        sink.put(0, Some((&src, 0)), 2);
        sink.put(0, Some((&[], 1)), 2);
        let mut out = [0i16; 4];
        let mut sink = InterleavedSink { out: &mut out, channels: 0 };
        sink.put(9, None, 4);
        sink.put(0, Some((&src, 0)), 100);
    }

    #[test]
    fn frame_size_and_output_length_are_checked() {
        let mut ms =
            MultistreamDecoder::with_layout(SampleRate::Hz48000, 2, 2, 0, &[0, 1]).unwrap();
        let mut out = vec![0.0f32; 2 * 5760];
        // A frame size of 0.
        assert!(ms.decode(&[], 0, &mut out).is_err());
        // The output does not hold frame_size samples per channel.
        assert!(ms.decode(&[], 960, &mut out[..2 * 960 - 1]).is_err());
        assert_eq!(ms.decode(&[], 960, &mut out[..2 * 960]), Ok(960));
        // A lost duration that is not a multiple of 2.5 ms.
        assert!(ms.decode(&[], 961, &mut out).is_err());
        // More than 120 ms counts as 120 ms.
        assert_eq!(ms.decode(&[], 1_000_000, &mut out), Ok(5760));
        assert_eq!(ms.decode(&[], usize::MAX, &mut out), Ok(5760));
        let mut out16 = vec![0i16; 2 * 5760];
        assert_eq!(ms.decode_i16(&[], usize::MAX, &mut out16), Ok(5760));
        assert_eq!(ms.decode_fec(&[], 480, &mut out), Ok(480));
        assert_eq!(ms.decode_fec_i16(&[], 480, &mut out16), Ok(480));
        assert!(ms.decode_fec_i16(&[], 480, &mut out16[..959]).is_err());
    }

    #[test]
    fn packets_shorter_than_the_stream_count_allows_are_rejected() {
        // 3 streams need at least 2 + 2 + 1 bytes.
        let mut ms =
            MultistreamDecoder::with_layout(SampleRate::Hz48000, 3, 3, 0, &[0, 1, 2]).unwrap();
        let mut out = vec![0.0f32; 3 * 960];
        for len in 1..5 {
            assert!(ms.decode(&[0u8; 8][..len], 960, &mut out).is_err(), "{len} bytes");
        }
        // Three SILK narrowband 10 ms streams without data: each is concealed.
        assert_eq!(ms.decode(&[0, 0, 0, 0, 0], 960, &mut out), Ok(480));
        // The room is smaller than the packet.
        assert!(ms.decode(&[0, 0, 0, 0, 0], 240, &mut out).is_err());
    }
}
