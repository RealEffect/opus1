//! Multistream decoder: a port of libopus 1.6 `opus_multistream_decoder.c`.
//!
//! A multistream packet holds one ordinary Opus packet per stream. Every
//! stream has its own [`OpusDecoder`] with 1 or 2 channels. This decoder
//! decodes the streams one after the other and copies each decoded channel
//! to the output channels that the [`ChannelLayout`] names.
//!
//! Memory is allocated in the constructor only. Decoding allocates nothing.

use super::layout::{ChannelLayout, SILENT_CHANNEL};
use super::{ChannelMappingTable, ChannelSink};
use crate::{OpusDecoder, float2int16, packet};

/// Decoder for a multistream Opus stream: libopus `OpusMSDecoder`.
///
/// Samples are interleaved: sample `i` of output channel `c` is at index
/// `i * channels + c`. A frame size is always in samples per channel.
pub struct MultistreamDecoder {
    layout: ChannelLayout,
    sampling_rate: i32,
    /// One decoder per stream. The first `layout.coupled_streams()` of them
    /// have 2 channels, the others 1.
    decoders: Vec<OpusDecoder>,
    /// Output of one stream: room for 120 ms of stereo, the longest packet.
    buf: Vec<f32>,
}

/// Interleaved `f32` output: libopus `opus_copy_channel_out_float`.
struct FloatSink<'a> {
    out: &'a mut [f32],
    channels: usize,
}

impl ChannelSink for FloatSink<'_> {
    #[inline]
    fn put(&mut self, dst_channel: usize, src: Option<(&[f32], usize)>, frame_size: usize) {
        let Some(out) = self.out.get_mut(dst_channel..) else {
            return;
        };
        let dst = out.iter_mut().step_by(self.channels.max(1)).take(frame_size);
        match src {
            Some((samples, stride)) => {
                for (d, &s) in dst.zip(samples.iter().step_by(stride.max(1))) {
                    *d = s;
                }
            }
            None => {
                for d in dst {
                    *d = 0.0;
                }
            }
        }
    }
}

/// Interleaved 16-bit output: libopus `opus_copy_channel_out_short`.
struct I16Sink<'a> {
    out: &'a mut [i16],
    channels: usize,
}

impl ChannelSink for I16Sink<'_> {
    #[inline]
    fn put(&mut self, dst_channel: usize, src: Option<(&[f32], usize)>, frame_size: usize) {
        let Some(out) = self.out.get_mut(dst_channel..) else {
            return;
        };
        let dst = out.iter_mut().step_by(self.channels.max(1)).take(frame_size);
        match src {
            Some((samples, stride)) => {
                for (d, &s) in dst.zip(samples.iter().step_by(stride.max(1))) {
                    *d = float2int16(s);
                }
            }
            None => {
                for d in dst {
                    *d = 0;
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
    pub fn new(sampling_rate: i32, table: ChannelMappingTable) -> Result<Self, &'static str> {
        if table.mapping.len() != table.channels as usize {
            return Err("channel mapping table length does not match declared channel count");
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
    /// - `sampling_rate`: output rate in Hz: 8000, 12000, 16000, 24000 or
    ///   48000.
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
        sampling_rate: i32,
        channels: usize,
        streams: usize,
        coupled_streams: usize,
        mapping: &[u8],
    ) -> Result<Self, &'static str> {
        let layout = ChannelLayout::new(channels, streams, coupled_streams, mapping)?;
        let mut decoders = Vec::with_capacity(layout.streams());
        for s in 0..layout.streams() {
            let stream_channels = if s < layout.coupled_streams() { 2 } else { 1 };
            decoders.push(OpusDecoder::new(sampling_rate, stream_channels)?);
        }
        // `OpusDecoder::new` has checked the sampling rate.
        let max_frame = sampling_rate as usize / 25 * 3;
        Ok(Self {
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

    /// The sampling rate of the output in Hz.
    pub fn sampling_rate(&self) -> i32 {
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
    ) -> Result<usize, &'static str> {
        let frame_size = self.usable_frame_size(frame_size, out.len())?;
        let channels = self.layout.channels();
        self.decode_native(packet, frame_size, &mut FloatSink { out, channels }, false, false)
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
    ) -> Result<usize, &'static str> {
        let frame_size = self.usable_frame_size(frame_size, out.len())?;
        let channels = self.layout.channels();
        self.decode_native(packet, frame_size, &mut I16Sink { out, channels }, false, true)
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
    ) -> Result<usize, &'static str> {
        let frame_size = self.usable_frame_size(frame_size, out.len())?;
        let channels = self.layout.channels();
        self.decode_native(packet, frame_size, &mut FloatSink { out, channels }, true, false)
    }

    /// [`decode_fec`](Self::decode_fec) with 16-bit output: libopus
    /// `opus_multistream_decode` with `decode_fec = 1`.
    pub fn decode_fec_i16(
        &mut self,
        packet: &[u8],
        frame_size: usize,
        out: &mut [i16],
    ) -> Result<usize, &'static str> {
        let frame_size = self.usable_frame_size(frame_size, out.len())?;
        let channels = self.layout.channels();
        self.decode_native(packet, frame_size, &mut I16Sink { out, channels }, true, true)
    }

    /// libopus `OPUS_GET_FINAL_RANGE` of a multistream decoder: the final
    /// range values of all stream decoders combined with XOR. After a packet
    /// was decoded it equals the value the multistream encoder reports for
    /// that packet.
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
    fn usable_frame_size(&self, frame_size: usize, out_len: usize) -> Result<usize, &'static str> {
        if frame_size == 0 {
            return Err("frame_size must not be 0");
        }
        // C: "Limit frame_size to avoid excessive stack allocations."
        let frame_size = frame_size.min(self.sampling_rate as usize / 25 * 3);
        if out_len / self.layout.channels() < frame_size {
            return Err("output buffer too small for frame_size * channels");
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
    ) -> Result<usize, &'static str> {
        let Self { layout, sampling_rate, decoders, buf } = self;
        let streams = layout.streams();
        let coupled_streams = layout.coupled_streams();

        let do_plc = packet.is_empty();
        if !do_plc {
            // Every stream takes at least 1 byte, and every stream but the
            // last at least 2.
            if packet.len() < 2 * streams - 1 {
                return Err("Invalid packet");
            }
            // Nothing is decoded before the whole packet has been checked.
            let samples = packet::multistream_validate(packet, streams, *sampling_rate)?;
            if samples > frame_size {
                return Err("frame_size too small for packet");
            }
        }

        let mut rest = packet;
        for (s, dec) in decoders.iter_mut().enumerate() {
            let stream_channels = if s < coupled_streams { 2 } else { 1 };
            if !do_plc && rest.is_empty() {
                return Err("Internal error: the packet ended before the last stream");
            }
            // Every stream but the last says how long it is.
            let self_delimited = s != streams - 1;
            let stream_buf = buf
                .get_mut(..frame_size * stream_channels)
                .ok_or("Internal error: stream buffer too small")?;
            let (samples, used) = if soft_clip {
                dec.decode_native_soft_clip(rest, frame_size, stream_buf, decode_fec, self_delimited)?
            } else {
                dec.decode_native(rest, frame_size, stream_buf, decode_fec, self_delimited)?
            };
            if !do_plc {
                rest = rest.get(used..).unwrap_or(&[]);
            }
            if samples == 0 {
                return Ok(0);
            }
            frame_size = samples;
            let stream_buf = &buf[..frame_size * stream_channels];
            if s < coupled_streams {
                // Copy the left channel to every output channel that uses it.
                let mut prev = None;
                while let Some(chan) = layout.left_channel(s, prev) {
                    sink.put(chan, Some((stream_buf, 2)), frame_size);
                    prev = Some(chan);
                }
                // The same for the right channel.
                let mut prev = None;
                while let Some(chan) = layout.right_channel(s, prev) {
                    sink.put(chan, Some((&stream_buf[1..], 2)), frame_size);
                    prev = Some(chan);
                }
            } else {
                let mut prev = None;
                while let Some(chan) = layout.mono_channel(s, prev) {
                    sink.put(chan, Some((stream_buf, 1)), frame_size);
                    prev = Some(chan);
                }
            }
        }
        // Channels that no stream feeds are silent.
        for (c, &m) in layout.mapping().iter().enumerate() {
            if m == SILENT_CHANNEL {
                sink.put(c, None, frame_size);
            }
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
        let ms = MultistreamDecoder::new(48000, table(6, 4, 2, &[0, 4, 1, 2, 3, 5])).unwrap();
        assert_eq!(ms.decoders.len(), 4);
        let stream_channels: Vec<usize> = ms.decoders.iter().map(|d| d.channels()).collect();
        assert_eq!(stream_channels, [2, 2, 1, 1]);
        assert_eq!(ms.layout().channels(), 6);
        assert_eq!(ms.layout().mapping(), &[0, 4, 1, 2, 3, 5]);
        assert_eq!(ms.sampling_rate(), 48000);
        assert_eq!(ms.buf.len(), 2 * 5760);
        assert!(ms.stream_decoder(3).is_some());
        assert!(ms.stream_decoder(4).is_none());
    }

    #[test]
    fn new_rejects_invalid_tables() {
        // More coupled streams than streams.
        assert!(MultistreamDecoder::new(48000, table(2, 1, 2, &[0, 1])).is_err());
        // A mapping entry that names a decoded channel that does not exist.
        // The old decoder accepted this table and panicked while decoding.
        assert!(MultistreamDecoder::new(48000, table(1, 1, 0, &[5])).is_err());
        // No stream, no channel.
        assert!(MultistreamDecoder::new(48000, table(1, 0, 0, &[0])).is_err());
        assert!(MultistreamDecoder::new(48000, table(0, 1, 0, &[])).is_err());
        // The mapping does not have one entry per channel.
        assert!(MultistreamDecoder::new(48000, table(2, 1, 1, &[0])).is_err());
        assert!(MultistreamDecoder::new(48000, table(2, 1, 1, &[0, 1, 0])).is_err());
        // A sampling rate Opus does not have.
        assert!(MultistreamDecoder::new(44100, table(2, 1, 1, &[0, 1])).is_err());
        assert!(MultistreamDecoder::new(48000, table(2, 1, 1, &[0, 1])).is_ok());
    }

    #[test]
    fn the_buffer_holds_120_ms_of_stereo_at_every_rate() {
        for (fs, want) in [(8000, 1920), (12000, 2880), (16000, 3840), (24000, 5760), (48000, 11520)] {
            let ms = MultistreamDecoder::with_layout(fs, 1, 1, 0, &[0]).unwrap();
            assert_eq!(ms.buf.len(), want, "{fs} Hz");
        }
    }

    #[test]
    fn sinks_write_one_channel_and_leave_the_others() {
        let src = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut out = [9.0f32; 9];
        let mut sink = FloatSink { out: &mut out, channels: 3 };
        // Every second sample of `src`, to channel 1.
        sink.put(1, Some((&src, 2)), 3);
        assert_eq!(out, [9.0, 1.0, 9.0, 9.0, 3.0, 9.0, 9.0, 5.0, 9.0]);
        let mut sink = FloatSink { out: &mut out, channels: 3 };
        // The same samples starting at the second one, to channel 2.
        sink.put(2, Some((&src[1..], 2)), 3);
        sink.put(0, None, 2);
        assert_eq!(out, [0.0, 1.0, 2.0, 0.0, 3.0, 4.0, 9.0, 5.0, 6.0]);

        let src = [0.5f32, -0.5, 2.0, -2.0];
        let mut out = [7i16; 8];
        let mut sink = I16Sink { out: &mut out, channels: 2 };
        sink.put(0, Some((&src, 1)), 4);
        assert_eq!(out, [16384, 7, -16384, 7, 32767, 7, -32768, 7]);
        let mut sink = I16Sink { out: &mut out, channels: 2 };
        sink.put(1, None, 3);
        assert_eq!(out, [16384, 0, -16384, 0, 32767, 0, -32768, 7]);
    }

    #[test]
    fn sinks_do_not_panic_on_arguments_that_do_not_fit() {
        let src = [1.0f32; 4];
        let mut out = [0.0f32; 4];
        let mut sink = FloatSink { out: &mut out, channels: 2 };
        // A channel beyond the output, more samples than there are, stride 0.
        sink.put(9, Some((&src, 1)), 4);
        sink.put(1, Some((&src, 1)), 100);
        sink.put(0, Some((&src, 0)), 2);
        sink.put(0, Some((&[], 1)), 2);
        let mut out = [0i16; 4];
        let mut sink = I16Sink { out: &mut out, channels: 0 };
        sink.put(9, None, 4);
        sink.put(0, Some((&src, 0)), 100);
    }

    #[test]
    fn frame_size_and_output_length_are_checked() {
        let mut ms = MultistreamDecoder::with_layout(48000, 2, 2, 0, &[0, 1]).unwrap();
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
        let mut ms = MultistreamDecoder::with_layout(48000, 3, 3, 0, &[0, 1, 2]).unwrap();
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
