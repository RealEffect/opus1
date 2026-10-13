//! Multistream Opus: several Opus streams in one packet, for audio with more
//! than two channels (RFC 7845 section 5.1.1, libopus `opus_multistream_*`).
//!
//! A multistream packet holds one ordinary Opus packet per stream. The first
//! streams are stereo ("coupled" streams), the others are mono. Every stream
//! has its own [`OpusDecoder`](crate::OpusDecoder), so the codec itself never
//! sees more than 2 channels. A channel layout says which decoded channel
//! each channel of the application uses.
//!
//! What is here:
//!
//! - [`MultistreamDecoder`](crate::multistream::MultistreamDecoder), in the
//!   module `decoder`: decodes a multistream packet into interleaved
//!   samples. A port of libopus `opus_multistream_decoder.c`.
//! - [`ChannelLayout`](crate::multistream::ChannelLayout), in the module
//!   `layout`: the layout, its checks, and the layouts of the mapping
//!   families of the Ogg Opus header.
//! - the module `rate_alloc`: how an encoder divides a bitrate among the
//!   streams.
//! - the modules `encoder` and `surround` belong to the multistream encoder.
//! - [`ChannelMappingTable`](crate::multistream::ChannelMappingTable): the
//!   channel mapping table as it is stored in an Ogg Opus header.
//! - [`split_self_delimited`](crate::multistream::split_self_delimited):
//!   cuts one stream out of a multistream packet, for code that wants to
//!   look at the streams itself. The decoder does not need it.
//!
//! The module needs the `std` feature. The number of streams is known only at
//! run time (up to 255), so the stream decoders live in a `Vec`. Memory is
//! allocated when a decoder is built. Decoding allocates nothing.

// The links in the text above are written with their full path on purpose.
// `src/lib.rs` puts a line of documentation in front of that text, and
// rustdoc then resolves the links of both from the crate root.

pub mod decoder;
pub mod encoder;
pub mod layout;
pub mod rate_alloc;
pub mod surround;

pub use decoder::MultistreamDecoder;
pub use encoder::{Bitrate, MultistreamEncoder};
pub use layout::{ChannelLayout, MappingType, SILENT_CHANNEL};

use crate::packet;

/// Source of the input samples of one frame, by input channel.
/// libopus calls this `opus_copy_channel_in_func`.
#[allow(dead_code)] // used once the multistream encoder is in
pub(crate) trait ChannelSource {
    /// Writes `frame_size` samples of input channel `src_channel` to
    /// `dst[0]`, `dst[dst_stride]`, `dst[2 * dst_stride]` and so on.
    fn copy(&self, dst: &mut [f32], dst_stride: usize, src_channel: usize, frame_size: usize);
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
    /// [`layout`] describes: 2 per stereo stream first, then 1 per mono
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
    // size of the last frame. A size below 252 takes 1 byte, any other 2.
    let last_size = parsed.sizes[parsed.count - 1];
    let field_len = if last_size < 252 { 1 } else { 2 };
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
