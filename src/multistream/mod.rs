//! Multistream Opus: several Opus streams in one packet, for audio with more
//! than two channels (RFC 7845 section 5.1.1, libopus `opus_multistream_*`).
//!
//! A multistream packet holds one ordinary Opus packet per stream. The first
//! streams are stereo ("coupled" streams), the others are mono. Every stream
//! has its own [`OpusEncoder`](crate::OpusEncoder) or
//! [`OpusDecoder`](crate::OpusDecoder), so the codec itself never sees more
//! than 2 channels. A channel layout says which stream each channel of the
//! application goes to or comes from.
//!
//! What is here:
//!
//! - [`MultistreamEncoder`](crate::multistream::MultistreamEncoder):
//!   encodes interleaved samples into a multistream packet. A port of
//!   libopus `opus_multistream_encoder.c` (`src/multistream/encoder.rs`).
//!   Its surround masking analysis is in `src/multistream/surround.rs`, and
//!   how it divides a bitrate among the streams in
//!   `src/multistream/rate_alloc.rs`.
//! - [`MultistreamDecoder`](crate::multistream::MultistreamDecoder): decodes
//!   a multistream packet into interleaved samples. A port of libopus
//!   `opus_multistream_decoder.c` (`src/multistream/decoder.rs`).
//! - [`ChannelLayout`](crate::multistream::ChannelLayout): the layout, its
//!   checks, and the layouts of the mapping families of the Ogg Opus header
//!   (`src/multistream/layout.rs`).
//!   [`validate_ambisonics`](crate::multistream::validate_ambisonics) checks
//!   the channel count of an ambisonics layout.
//! - [`ChannelMappingTable`](crate::multistream::ChannelMappingTable): the
//!   channel mapping table as it is stored in an Ogg Opus header.
//! - [`split_self_delimited`](crate::multistream::split_self_delimited):
//!   cuts one stream out of a multistream packet, for code that wants to
//!   look at the streams itself. The decoder does not need it.
//!
//! The module needs the `std` feature. The number of streams is known only at
//! run time (up to 255), so the stream encoders and decoders live in a
//! `Vec`. Memory is allocated when an encoder or a decoder is built.
//! Encoding and decoding allocate nothing.
//!
//! # Encoding
//!
//! [`MultistreamEncoder::new_surround`](crate::multistream::MultistreamEncoder::new_surround)
//! picks the layout of a mapping family of the Ogg Opus header. Family 1 is
//! the one for surround sound: 1 to 8 channels in the channel order of
//! Vorbis. With more than 2 channels the encoder works in surround mode. It
//! then codes the stereo pairs with CELT and spends fewer bits on what
//! other channels mask. With 6, 7 or 8 channels the last channel is the
//! low-frequency-effects (LFE) channel, which gets a small stream of its
//! own.
//!
//! ```
//! use opus1::multistream::MultistreamEncoder;
//! use opus1::{Application, Bitrate};
//!
//! // 5.1 at 48 kHz: front left, centre, front right, rear left, rear
//! // right, LFE.
//! let mut encoder = MultistreamEncoder::new_surround(48_000, 6, 1, Application::Audio)?;
//! encoder.set_bitrate(Bitrate::BitsPerSecond(256_000))?;
//!
//! // Two stereo streams (front and rear), then the centre and the LFE
//! // channel as mono streams. An Ogg Opus header stores these numbers and
//! // the mapping.
//! let layout = encoder.layout();
//! assert_eq!((layout.streams(), layout.coupled_streams()), (4, 2));
//! assert_eq!(layout.mapping(), &[0, 4, 1, 2, 3, 5]);
//! assert_eq!(encoder.lfe_stream(), Some(3));
//!
//! // One frame of 20 ms: 960 samples per channel, interleaved. Sample `i`
//! // of channel `c` is `pcm[i * 6 + c]`.
//! let pcm: Vec<f32> = (0..960 * 6)
//!     .map(|k| 0.2 * ((k / 6) as f32 * 0.02 * (1 + k % 6) as f32).sin())
//!     .collect();
//! // The length of the buffer is the largest packet allowed.
//! let mut packet = vec![0u8; 4000];
//! let len = encoder.encode(&pcm, 960, &mut packet)?;
//! assert!(len > 0 && len <= packet.len());
//! # Ok::<(), &'static str>(())
//! ```
//!
//! # Decoding
//!
//! The decoder needs the layout the encoder used: the number of channels,
//! of streams and of coupled streams, and the mapping. A file carries them
//! in its header. A packet does not, so a decoder with another layout reads
//! it wrongly or rejects it.
//!
//! ```
//! use opus1::Application;
//! use opus1::multistream::{MultistreamDecoder, MultistreamEncoder};
//!
//! // A packet to decode: one 20 ms frame of quadraphonic sound.
//! let mut encoder = MultistreamEncoder::new_surround(48_000, 4, 1, Application::Audio)?;
//! let pcm = vec![0.1f32; 960 * 4];
//! let mut packet = vec![0u8; 4000];
//! let len = encoder.encode(&pcm, 960, &mut packet)?;
//!
//! // The decoder for the same layout.
//! let layout = encoder.layout();
//! let mut decoder = MultistreamDecoder::with_layout(
//!     48_000,
//!     layout.channels(),
//!     layout.streams(),
//!     layout.coupled_streams(),
//!     layout.mapping(),
//! )?;
//!
//! // `out` has room for 960 samples per channel. The result is the number
//! // of samples per channel that were written, interleaved like the input.
//! let mut out = vec![0f32; 960 * 4];
//! let samples = decoder.decode(&packet[..len], 960, &mut out)?;
//! assert_eq!(samples, 960);
//!
//! // An empty packet stands for a lost one: the decoder fills the gap.
//! let samples = decoder.decode(&[], 960, &mut out)?;
//! assert_eq!(samples, 960);
//! # Ok::<(), &'static str>(())
//! ```
//!
//! `examples/multistream.rs` encodes and decodes a longer 5.1 signal and
//! prints the packet sizes and the quality of every channel.

// The links in the text above are written with their full path on purpose.
// `src/lib.rs` puts a line of documentation in front of that text, and
// rustdoc then resolves the links of both from the crate root.

mod decoder;
mod encoder;
mod layout;
pub(crate) mod rate_alloc;
pub(crate) mod surround;

pub use decoder::MultistreamDecoder;
pub use encoder::MultistreamEncoder;
pub use layout::{ChannelLayout, MappingType, SILENT_CHANNEL, validate_ambisonics};

use crate::{float2int16, int16_to_res, packet};

/// Source of the input samples of one frame, by input channel.
/// libopus calls this `opus_copy_channel_in_func`.
pub(crate) trait ChannelSource {
    /// Writes `frame_size` samples of input channel `channel` to
    /// `dst[..frame_size]`.
    fn copy(&self, dst: &mut [f32], channel: usize, frame_size: usize);
}

/// A sample type of the encoder input and the decoder output: `f32` with
/// full scale 1.0, or `i16`.
pub(crate) trait Sample: Copy {
    /// Silence.
    const ZERO: Self;
    /// The precision libopus gives this input type: 24 bits for `f32`
    /// (`MAX_ENCODING_DEPTH`) and 16 for `i16`.
    const LSB_DEPTH: i32;
    /// The samples as the speech and music analysis reads them: libopus
    /// `downmix_float` for `f32` and `downmix_int` for `i16`.
    fn analysis_pcm(pcm: &[Self]) -> crate::analysis::PcmRef<'_>;
    /// The sample as `f32` with full scale 1.0.
    fn to_f32(self) -> f32;
    /// An `f32` sample with full scale 1.0 as this type.
    fn from_f32(v: f32) -> Self;
}

impl Sample for f32 {
    const ZERO: f32 = 0.0;
    const LSB_DEPTH: i32 = 24;

    #[inline(always)]
    fn analysis_pcm(pcm: &[f32]) -> crate::analysis::PcmRef<'_> {
        crate::analysis::PcmRef::F32(pcm)
    }

    #[inline(always)]
    fn to_f32(self) -> f32 {
        self
    }

    #[inline(always)]
    fn from_f32(v: f32) -> f32 {
        v
    }
}

impl Sample for i16 {
    const ZERO: i16 = 0;
    const LSB_DEPTH: i32 = 16;

    #[inline(always)]
    fn analysis_pcm(pcm: &[i16]) -> crate::analysis::PcmRef<'_> {
        crate::analysis::PcmRef::I16(pcm)
    }

    /// libopus `INT16TORES` in the float build.
    #[inline(always)]
    fn to_f32(self) -> f32 {
        int16_to_res(self)
    }

    /// libopus `FLOAT2INT16`: rounded to the nearest integer and limited to
    /// the 16-bit range.
    #[inline(always)]
    fn from_f32(v: f32) -> i16 {
        float2int16(v)
    }
}

/// Interleaved input: sample `i` of channel `c` is `pcm[i * channels + c]`.
/// libopus `opus_copy_channel_in_float` and `opus_copy_channel_in_short`.
///
/// A copy writes fewer samples when `pcm` or the destination is shorter
/// than the frame. It writes nothing for a channel the input does not have.
pub(crate) struct Interleaved<'a, T> {
    pub(crate) pcm: &'a [T],
    pub(crate) channels: usize,
}

impl<T: Sample> Interleaved<'_, T> {
    /// Writes `frame_size` samples of channels `left` and `right` to `dst`,
    /// interleaved: sample `i` of `left` to `dst[2 * i]` and of `right` to
    /// `dst[2 * i + 1]`. This is the input of a coupled stream.
    #[inline]
    pub(crate) fn copy_pair(&self, dst: &mut [f32], left: usize, right: usize, frame_size: usize) {
        if left >= self.channels || right >= self.channels {
            return;
        }
        let pairs = dst.as_chunks_mut::<2>().0.iter_mut();
        let frames = self.pcm.chunks_exact(self.channels);
        for (d, frame) in pairs.zip(frames).take(frame_size) {
            *d = [frame[left].to_f32(), frame[right].to_f32()];
        }
    }
}

impl<T: Sample> ChannelSource for Interleaved<'_, T> {
    #[inline]
    fn copy(&self, dst: &mut [f32], channel: usize, frame_size: usize) {
        if channel >= self.channels {
            return;
        }
        let frames = self.pcm.chunks_exact(self.channels);
        for (d, frame) in dst.iter_mut().zip(frames).take(frame_size) {
            *d = frame[channel].to_f32();
        }
    }
}

/// Destination of the decoded samples of one frame, by output channel.
/// libopus calls this `opus_copy_channel_out_func`.
pub(crate) trait ChannelSink {
    /// Writes `frame_size` samples to output channel `dst_channel`.
    /// `src` is `Some((samples, stride))`: sample `i` is `samples[i * stride]`.
    /// `None` writes silence.
    fn put(&mut self, dst_channel: usize, src: Option<(&[f32], usize)>, frame_size: usize);
}

/// The channel mapping table of an Ogg Opus header (RFC 7845 section 5.1.1):
/// the bytes that follow the fixed 19-byte part of `OpusHead` when the
/// mapping family is not 0.
///
/// A table read from a file is not checked.
/// [`MultistreamDecoder::new`] checks it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChannelMappingTable {
    /// Number of output channels.
    pub channels: u8,
    /// Number of streams in one packet.
    pub stream_count: u8,
    /// Number of stereo streams. They are the first streams of a packet; the
    /// others are mono.
    pub coupled_count: u8,
    /// One entry per output channel: the decoded channel it uses, or
    /// [`SILENT_CHANNEL`] for silence. Decoded channels are numbered as
    /// [`ChannelLayout`] describes: 2 per stereo stream first, then 1 per mono
    /// stream.
    pub mapping: Vec<u8>,
}

impl ChannelMappingTable {
    /// Reads the table. `tail` starts right after the output gain of
    /// `OpusHead`, at byte 19 of the header.
    ///
    /// Returns `None` for mapping family 0, which has no table, and when
    /// `tail` is too short.
    pub fn parse(mapping_family: u8, channels: u8, tail: &[u8]) -> Option<Self> {
        if mapping_family == 0 || tail.len() < 2 + channels as usize {
            return None;
        }
        Some(Self {
            channels,
            stream_count: tail[0],
            coupled_count: tail[1],
            mapping: tail[2..2 + channels as usize].to_vec(),
        })
    }
}

/// Cuts the first stream out of a multistream packet.
///
/// Every stream of a multistream packet but the last is "self-delimited"
/// (RFC 6716 appendix B): it carries one size field more than an ordinary
/// packet, for its last frame, so that a reader knows where the stream ends.
///
/// `data` starts with a self-delimited packet. The function copies it into
/// `scratch` without the extra size field. The result is an ordinary Opus
/// packet that any decoder accepts. Returns that packet and the number of
/// bytes the stream takes in `data`, which is where the next stream starts.
///
/// `scratch` is cleared first. Its memory is reused, so a caller that splits
/// many packets allocates once.
///
/// The framing is checked with [`packet::parse`], which follows libopus. A
/// packet that libopus rejects is an error.
pub fn split_self_delimited<'a>(
    data: &[u8],
    scratch: &'a mut Vec<u8>,
) -> Result<(&'a [u8], usize), &'static str> {
    const INVALID: &str = "Invalid packet";
    let parsed = packet::parse(data, true)?;
    // The extra field is the last thing before the first frame. It holds the
    // size of the last frame.
    let field_len = packet::size_field_len(parsed.sizes[parsed.count - 1]);
    let header_end = parsed.payload_offset.checked_sub(field_len).ok_or(INVALID)?;
    let header = data.get(..header_end).ok_or(INVALID)?;
    let frames_and_padding = data.get(parsed.payload_offset..parsed.packet_offset).ok_or(INVALID)?;
    scratch.clear();
    scratch.extend_from_slice(header);
    scratch.extend_from_slice(frames_and_padding);
    Ok((scratch.as_slice(), parsed.packet_offset))
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;

    fn toc_byte(config: u8, stereo: bool, code: u8) -> u8 {
        (config << 3) | ((stereo as u8) << 2) | (code & 0x03)
    }

    #[test]
    fn interleaved_input_gives_one_channel_or_a_pair() {
        // 3 channels, 4 samples: channel c holds 10 * c + i.
        let pcm: Vec<f32> = (0..12).map(|k| (10 * (k % 3) + k / 3) as f32).collect();
        let src = Interleaved {
            pcm: &pcm,
            channels: 3,
        };
        let mut mono = [-1.0f32; 6];
        src.copy(&mut mono, 2, 4);
        assert_eq!(mono, [20.0, 21.0, 22.0, 23.0, -1.0, -1.0]);
        let mut stereo = [-1.0f32; 10];
        src.copy_pair(&mut stereo, 0, 1, 4);
        assert_eq!(stereo, [0.0, 10.0, 1.0, 11.0, 2.0, 12.0, 3.0, 13.0, -1.0, -1.0]);
        // The right channel of a pair may come before the left one.
        src.copy_pair(&mut stereo, 2, 0, 2);
        assert_eq!(&stereo[..4], [20.0, 0.0, 21.0, 1.0]);
    }

    #[test]
    fn interleaved_input_never_reads_or_writes_out_of_range() {
        let pcm = [1.0f32; 7];
        let src = Interleaved {
            pcm: &pcm,
            channels: 2,
        };
        // More samples asked for than the input and the destination hold.
        let mut dst = [0.0f32; 2];
        src.copy(&mut dst, 1, 100);
        assert_eq!(dst, [1.0, 1.0]);
        let mut dst = [0.0f32; 5];
        src.copy_pair(&mut dst, 0, 1, 100);
        assert_eq!(dst, [1.0, 1.0, 1.0, 1.0, 0.0]);
        // A channel that does not exist copies nothing.
        let mut dst = [0.0f32; 4];
        src.copy(&mut dst, 2, 2);
        src.copy_pair(&mut dst, 0, 2, 2);
        src.copy_pair(&mut dst, 2, 1, 2);
        assert_eq!(dst, [0.0; 4]);
    }

    #[test]
    fn sixteen_bit_input_is_scaled_by_32768() {
        let pcm = [i16::MIN, -1, 0, 1, 16384, i16::MAX];
        let src = Interleaved {
            pcm: &pcm,
            channels: 1,
        };
        let mut dst = [9.0f32; 6];
        src.copy(&mut dst, 0, 6);
        assert_eq!(
            dst,
            [
                -1.0,
                -1.0 / 32768.0,
                0.0,
                1.0 / 32768.0,
                0.5,
                32767.0 / 32768.0
            ]
        );
    }

    #[test]
    fn mapping_table_parses_5_1_shape() {
        let tail = [4u8, 2, 0, 1, 4, 5, 2, 3];
        let table = ChannelMappingTable::parse(1, 6, &tail).unwrap();
        assert_eq!(table.stream_count, 4);
        assert_eq!(table.coupled_count, 2);
        assert_eq!(table.mapping, vec![0, 1, 4, 5, 2, 3]);
    }

    #[test]
    fn mapping_family_zero_has_no_table() {
        assert!(ChannelMappingTable::parse(0, 2, &[9, 9, 9, 9]).is_none());
    }

    #[test]
    fn mapping_table_rejects_truncated_tail() {
        assert!(ChannelMappingTable::parse(1, 6, &[4, 2, 0, 1]).is_none());
    }

    #[test]
    fn split_code0_single_frame() {
        let bytes = [toc_byte(0, false, 0), 3, 0xAA, 0xBB, 0xCC, 0xFF];
        let mut scratch = Vec::new();
        let (pkt, consumed) = split_self_delimited(&bytes, &mut scratch).unwrap();
        assert_eq!(consumed, 1 + 1 + 3);
        // reconstructed packet excises the length byte: TOC + frame data only
        assert_eq!(pkt, &[toc_byte(0, false, 0), 0xAA, 0xBB, 0xCC][..]);
    }

    #[test]
    fn split_code0_two_byte_length() {
        let mut bytes = vec![toc_byte(0, false, 0), 252, 0];
        bytes.extend(std::iter::repeat_n(0x7Fu8, 252));
        let mut scratch = Vec::new();
        let (_, consumed) = split_self_delimited(&bytes, &mut scratch).unwrap();
        assert_eq!(consumed, 1 + 2 + 252);
    }

    #[test]
    fn split_code1_equal_frames() {
        let bytes = [toc_byte(0, false, 1), 2, 0xAA, 0xBB, 0xCC, 0xDD, 0x11];
        let mut scratch = Vec::new();
        let (_, consumed) = split_self_delimited(&bytes, &mut scratch).unwrap();
        assert_eq!(consumed, 1 + 1 + 4);
    }

    #[test]
    fn split_code2_two_lengths() {
        let bytes = [toc_byte(0, false, 2), 3, 2, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE];
        let mut scratch = Vec::new();
        let (pkt, consumed) = split_self_delimited(&bytes, &mut scratch).unwrap();
        assert_eq!(consumed, 1 + 1 + 1 + 3 + 2);
        // inline length (3, for frame 1) is kept; the extra Appendix B
        // length (2, for frame 2) is excised
        assert_eq!(pkt, &[toc_byte(0, false, 2), 3, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE][..]);
    }

    #[test]
    fn split_code3_cbr_three_frames() {
        let bytes = [toc_byte(0, false, 3), 3, 2, 0x10, 0x11, 0x20, 0x21, 0x30, 0x31];
        let mut scratch = Vec::new();
        let (_, consumed) = split_self_delimited(&bytes, &mut scratch).unwrap();
        assert_eq!(consumed, 1 + 1 + 1 + 6);
    }

    #[test]
    fn split_code3_cbr_with_padding() {
        let fc = 0x40 | 2u8;
        let bytes = [
            toc_byte(0, false, 3), fc, 2, 3,
            0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF,
            0x55, 0x66,
        ];
        let mut scratch = Vec::new();
        let (_, consumed) = split_self_delimited(&bytes, &mut scratch).unwrap();
        assert_eq!(consumed, 12);
    }

    #[test]
    fn split_code3_vbr_three_frames() {
        let fc = 0x80 | 3u8;
        let bytes = [
            toc_byte(0, false, 3), fc, 2, 3, 1,
            0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF,
        ];
        let mut scratch = Vec::new();
        let (pkt, consumed) = split_self_delimited(&bytes, &mut scratch).unwrap();
        assert_eq!(consumed, 1 + 1 + 1 + 1 + 1 + 2 + 3 + 1);
        // inline lengths for frames 1-2 (2, 3) are kept; the extra
        // Appendix B length for frame 3 (1) is excised
        assert_eq!(pkt, &[toc_byte(0, false, 3), fc, 2, 3, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF][..]);
    }

    #[test]
    fn split_two_streams_chained() {
        let bytes = [
            toc_byte(0, false, 0), 2, 0xA0, 0xA1,
            toc_byte(0, false, 1), 1, 0xB0, 0xB1,
        ];
        let mut scratch = Vec::new();
        let a_consumed = {
            let (a, a_consumed) = split_self_delimited(&bytes, &mut scratch).unwrap();
            // reconstructed: TOC + frame data, length byte excised
            assert_eq!(a, &[toc_byte(0, false, 0), 0xA0, 0xA1][..]);
            a_consumed
        };
        assert_eq!(a_consumed, 4);
        let (_, b_consumed) = split_self_delimited(&bytes[a_consumed..], &mut scratch).unwrap();
        assert_eq!(b_consumed, 4);
    }

    #[test]
    fn split_truncated_frame_is_error() {
        let bytes = [toc_byte(0, false, 0), 4, 0xAA, 0xBB];
        let mut scratch = Vec::new();
        assert!(split_self_delimited(&bytes, &mut scratch).is_err());
    }

    #[test]
    fn split_code3_zero_frames_is_error() {
        let bytes = [toc_byte(0, false, 3), 0, 1, 0xAA];
        let mut scratch = Vec::new();
        assert!(split_self_delimited(&bytes, &mut scratch).is_err());
    }

    #[test]
    fn split_gives_the_packet_that_packet_write_gives() {
        // Write the same frames as a self-delimited and as an ordinary
        // packet. Splitting the first must give the second.
        let big = [0x5Au8; 600];
        let cases: [(&[&[u8]], bool); 9] = [
            (&[&big[..0]], false),
            (&[&big[..1]], false),
            (&[&big[..251]], false),
            (&[&big[..252]], false),
            (&[&big[..600]], false),
            (&[&big[..300], &big[..300]], false),
            (&[&big[..7], &big[..300]], false),
            (&[&big[..3], &big[..3], &big[..3]], true),
            (&[&big[..10], &big[..253], &big[..0], &big[..40]], true),
        ];
        // SILK narrowband 10 ms, so that 4 frames stay below 120 ms.
        let toc = toc_byte(0, false, 0);
        for (frames, pad) in cases {
            let mut delimited = [0u8; 2000];
            let mut ordinary = [0u8; 2000];
            let (len_d, len_o) = if pad {
                // Padding fills the whole buffer that is given.
                let total: usize = frames.iter().map(|f| f.len()).sum();
                (total + 40, total + 37)
            } else {
                (2000, 2000)
            };
            let n_d = packet::write(toc, frames, &mut delimited[..len_d], true, pad).unwrap();
            let n_o = packet::write(toc, frames, &mut ordinary[..len_o], false, pad).unwrap();
            // Bytes of another stream follow the self-delimited packet.
            delimited[n_d..n_d + 3].copy_from_slice(&[0xEE; 3]);
            let mut scratch = vec![1, 2, 3];
            let (pkt, consumed) = split_self_delimited(&delimited[..n_d + 3], &mut scratch).unwrap();
            assert_eq!(consumed, n_d, "{} frames", frames.len());
            let parsed = packet::parse(pkt, false).unwrap();
            assert_eq!(parsed.count, frames.len());
            for (i, frame) in frames.iter().enumerate() {
                assert_eq!(parsed.frame(pkt, i), *frame, "frame {i}");
            }
            if !pad {
                assert_eq!(pkt, &ordinary[..n_o], "{} frames", frames.len());
            }
        }
    }

    #[test]
    fn split_rejects_what_libopus_rejects() {
        let mut scratch = Vec::new();
        // No byte at all, and a TOC byte without its size field.
        assert!(split_self_delimited(&[], &mut scratch).is_err());
        assert!(split_self_delimited(&[toc_byte(0, false, 0)], &mut scratch).is_err());
        // A 2-byte size field cut after its first byte.
        assert!(split_self_delimited(&[toc_byte(0, false, 0), 252], &mut scratch).is_err());
        // 13 SILK 10 ms frames are more than 120 ms.
        let mut long = vec![toc_byte(0, false, 3), 13, 1];
        long.extend_from_slice(&[0u8; 13]);
        assert!(split_self_delimited(&long, &mut scratch).is_err());
    }
}
