//! The decoding loop of the libopus test program `opus_demo` in decode mode
//! (`opus_demo -d <rate> <channels> -ignore_extensions in.bit out.pcm`,
//! `src/opus_demo.c` of libopus 1.6.1). The RFC 8251 conformance check
//! decodes its test vectors with it before it compares.
//!
//! An `opus_demo` bitstream is a sequence of records. Each record has a
//! 4-byte big-endian payload length, the encoder's 4-byte big-endian final
//! range, and the payload. A length of 0 marks a lost packet.
//!
//! The loop:
//!
//! - A lost packet is not decoded at once. At the next packet that arrived,
//!   each lost packet is concealed with the duration of the last packet
//!   (`OPUS_GET_LAST_PACKET_DURATION`). The last lost packet is instead
//!   recovered with forward error correction (FEC) when the new packet
//!   carries a redundant copy (LBRR). Then the new packet is decoded.
//! - A loss at the end of the file is never decoded (a "FIXME" of
//!   `opus_demo`).
//! - The decoder writes 24-bit integers (`opus_decode24`). `opus_demo`
//!   rounds them to 16 bits: clamp to ±`0x7fff00`, then `(s + 128) >> 8`.
//! - After each packet that arrived, when the packet before it also
//!   arrived, the decoder's final range must equal the one in the record
//!   (unless that is 0). Otherwise `opus_demo` stops. The first packet is
//!   never checked.
//! - A failed decode call is reported and the loop goes on.

use crate::packet;

/// Largest payload `opus_demo` accepts (`MAX_PACKET`).
pub const MAX_PACKET: usize = 15000;
/// The duration `opus_demo` asks for when it decodes a packet that arrived:
/// `MAX_SAMPLING_RATE*2` samples, 2 s at 48 kHz. The decoder writes only
/// what the packet holds.
pub const MAX_FRAME_SIZE: usize = 96000;

/// One record of an `opus_demo` bitstream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BitPacket<'a> {
    /// The packet. Empty for a lost packet.
    pub data: &'a [u8],
    /// The encoder's final range after this packet, or 0 when unknown.
    pub enc_final_range: u32,
}

/// Why reading a bitstream stopped before its end. `opus_demo` prints the
/// message, keeps what it decoded so far and exits with success.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BitstreamError {
    /// Record `index` gives a length above [`MAX_PACKET`] (or a negative
    /// one, as `opus_demo` reads it).
    InvalidLength { index: usize, len: u32 },
    /// Record `index` has fewer payload bytes than its length says.
    RanOutOfInput { index: usize, expected: usize, got: usize },
}

impl core::fmt::Display for BitstreamError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match *self {
            // C prints the length as a signed `int`.
            BitstreamError::InvalidLength { len, .. } => {
                write!(f, "Invalid payload length: {}", len as i32)
            }
            BitstreamError::RanOutOfInput { expected, got, .. } => {
                write!(f, "Ran out of input, expecting {expected} bytes got {got}")
            }
        }
    }
}

impl std::error::Error for BitstreamError {}

/// The records of a bitstream, and why reading stopped early, if it did.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Bitstream<'a> {
    pub packets: Vec<BitPacket<'a>>,
    pub error: Option<BitstreamError>,
}

/// Read the records of an `opus_demo` bitstream (`opus_demo.c` 960-981).
/// Reading stops without an error where a length or range field is cut
/// short, and with an error at an invalid length or a cut payload. The
/// records before that point are kept, as `opus_demo` decodes them.
pub fn parse_bitstream(bytes: &[u8]) -> Bitstream<'_> {
    let mut packets = Vec::new();
    let mut rest = bytes;
    let error = loop {
        let index = packets.len();
        let Some((len, tail)) = read_u32(rest) else {
            break None;
        };
        if len as usize > MAX_PACKET || (len as i32) < 0 {
            break Some(BitstreamError::InvalidLength { index, len });
        }
        let Some((enc_final_range, tail)) = read_u32(tail) else {
            break None;
        };
        let len = len as usize;
        if tail.len() < len {
            break Some(BitstreamError::RanOutOfInput { index, expected: len, got: tail.len() });
        }
        let (data, tail) = tail.split_at(len);
        packets.push(BitPacket { data, enc_final_range });
        rest = tail;
    };
    Bitstream { packets, error }
}

/// C `char_to_int`: a big-endian 32-bit value.
fn read_u32(bytes: &[u8]) -> Option<(u32, &[u8])> {
    let (head, tail) = bytes.split_first_chunk::<4>()?;
    Some((u32::from_be_bytes(*head), tail))
}

/// Append one record to a bitstream, as `opus_demo -e` writes it.
pub fn write_record(out: &mut Vec<u8>, data: &[u8], enc_final_range: u32) {
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    out.extend_from_slice(&enc_final_range.to_be_bytes());
    out.extend_from_slice(data);
}

/// What [`decode_bitstream`] needs from a decoder: the libopus calls that
/// `opus_demo` makes. It is implemented for [`crate::OpusDecoder`]; a test
/// can implement it for libopus, so that the same loop drives both.
pub trait DemoDecoder {
    /// Output channels.
    fn channels(&self) -> usize;
    /// libopus `opus_decode24`: decode `data` into `out`, as 24-bit
    /// integers (full scale is ±2^23). An empty `data` is a lost packet,
    /// concealed for `frame_size` samples. With `decode_fec`, the frame
    /// before `data` is recovered from it. Returns the samples per channel
    /// written. The error is a message, not a [`crate::Error`], because
    /// code outside the crate (a test that wraps libopus) cannot make an
    /// `Error`.
    fn decode24(
        &mut self,
        data: &[u8],
        frame_size: usize,
        decode_fec: bool,
        out: &mut [i32],
    ) -> Result<usize, &'static str>;
    /// libopus `OPUS_GET_LAST_PACKET_DURATION`.
    fn last_packet_duration(&self) -> usize;
    /// libopus `OPUS_GET_FINAL_RANGE`.
    fn final_range(&self) -> u32;
}

impl DemoDecoder for crate::OpusDecoder {
    fn channels(&self) -> usize {
        crate::OpusDecoder::channels(self).count()
    }

    /// `opus_decode24` of the float build: decode to float, without the
    /// soft clipper, and convert each sample with [`res2int24`].
    fn decode24(
        &mut self,
        data: &[u8],
        frame_size: usize,
        decode_fec: bool,
        out: &mut [i32],
    ) -> Result<usize, &'static str> {
        if frame_size == 0 {
            return Err(crate::Error::INVALID_FRAME_SIZE.message());
        }
        // C: a packet that arrived needs no more room than it holds.
        let frame_size = if !data.is_empty() && !decode_fec {
            frame_size.min(packet::nb_samples(data, self.sampling_rate()).map_err(|e| e.message())?)
        } else {
            frame_size
        };
        let channels = crate::OpusDecoder::channels(self).count();
        let mut pcm = vec![0.0f32; frame_size * channels];
        let n = if decode_fec {
            self.decode_fec(data, frame_size, &mut pcm)
        } else {
            self.decode(data, frame_size, &mut pcm)
        }
        .map_err(|e| e.message())?;
        for (o, &x) in out.iter_mut().zip(&pcm[..n * channels]) {
            *o = res2int24(x);
        }
        Ok(n)
    }

    fn last_packet_duration(&self) -> usize {
        crate::OpusDecoder::last_packet_duration(self)
    }

    fn final_range(&self) -> u32 {
        crate::OpusDecoder::final_range(self)
    }
}

/// The decoder's final range differs from the encoder's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RangeMismatch {
    /// Index of the record.
    pub packet: usize,
    pub encoder: u32,
    pub decoder: u32,
}

/// What `opus_demo -d` produces.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DemoOutput {
    /// Interleaved 16-bit samples: the output file of `opus_demo`.
    pub pcm: Vec<i16>,
    /// Records processed, the one with a range mismatch included.
    pub packets: usize,
    /// Records that were lost (no payload).
    pub lost: usize,
    /// Decode calls that recovered a lost packet with FEC.
    pub fec_calls: usize,
    /// Decode calls that failed ("error decoding frame"). `opus_demo`
    /// writes nothing for them and goes on.
    pub decode_errors: usize,
    /// Packets whose final range was compared with the encoder's.
    pub range_checks: usize,
    /// The final range check that stopped `opus_demo`, if one did.
    pub range_mismatch: Option<RangeMismatch>,
}

/// The decoding loop of `opus_demo -d <rate> <channels> -ignore_extensions`
/// (`opus_demo.c` 1116-1232) over the records of a bitstream, with a fresh
/// or a used `decoder`.
pub fn decode_bitstream<D: DemoDecoder + ?Sized>(
    decoder: &mut D,
    packets: &[BitPacket<'_>],
) -> DemoOutput {
    let channels = decoder.channels();
    let mut out = vec![0i32; MAX_FRAME_SIZE * channels];
    let mut result = DemoOutput::default();
    // C: `lost_prev` starts at 1, so the first packet is never checked.
    let mut lost_prev = true;
    let mut lost_count = 0usize;
    for (index, packet) in packets.iter().enumerate() {
        let data = packet.data;
        let lost = data.is_empty();
        result.packets += 1;
        let run_decoder = if lost {
            lost_count += 1;
            result.lost += 1;
            0
        } else {
            1 + lost_count
        };
        for fr in 0..run_decoder {
            let decoded = if fr + 1 == lost_count && packet::has_lbrr(data).unwrap_or(true) {
                // C uses the result of `opus_packet_has_lbrr` as a truth
                // value, so an invalid packet (a negative error code) also
                // starts an FEC decode, which then fails.
                result.fec_calls += 1;
                let n = decoder.last_packet_duration();
                decoder.decode24(data, n, true, &mut out)
            } else if fr < lost_count {
                let n = decoder.last_packet_duration();
                decoder.decode24(&[], n, false, &mut out)
            } else {
                decoder.decode24(data, MAX_FRAME_SIZE, false, &mut out)
            };
            match decoded {
                Ok(n) if n > 0 => {
                    result.pcm.extend(out[..n * channels].iter().map(|&s| s24_to_s16(s)));
                }
                _ => result.decode_errors += 1,
            }
        }
        if packet.enc_final_range != 0 && !lost && !lost_prev {
            result.range_checks += 1;
            let decoder_range = decoder.final_range();
            if decoder_range != packet.enc_final_range {
                result.range_mismatch = Some(RangeMismatch {
                    packet: index,
                    encoder: packet.enc_final_range,
                    decoder: decoder_range,
                });
                break;
            }
        }
        lost_prev = lost;
        if !lost {
            lost_count = 0;
        }
    }
    result
}

/// libopus `RES2INT24` of the float build: `float2int(32768.f*256.f*x)`,
/// a float sample scaled to 24 bits and rounded to the nearest integer,
/// ties to even.
///
/// Out of the `i32` range, `float2int` saturates on 64-bit ARM
/// (`vcvtns_s32_f32`). On x86 (`_mm_cvt_ss2si`) it gives `i32::MIN` there
/// and for NaN, and so does this function. Elsewhere it saturates.
pub fn res2int24(x: f32) -> i32 {
    let v = x * 8_388_608.0;
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if !(-2_147_483_648.0..2_147_483_648.0).contains(&v) {
        return i32::MIN;
    }
    v.round_ties_even() as i32
}

/// One sample of the 16-bit output of `opus_demo` (`opus_demo.c`
/// 1165-1175): clamp to ±`0x7fff00`, then `(s + 128) >> 8`. The result lies
/// in [-32767, 32767].
fn s24_to_s16(s: i32) -> i16 {
    let s = s.clamp(-0x7fff00, 0x7fff00);
    ((s + 128) >> 8) as i16
}

/// The 16-bit sample `opus_demo` writes for the float sample `x` of the
/// decoder: [`res2int24`], then the clamp and the rounding to 16 bits.
pub fn opus_demo_s16(x: f32) -> i16 {
    s24_to_s16(res2int24(x))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_round_trip() {
        let mut bytes = Vec::new();
        write_record(&mut bytes, &[1, 2, 3], 0xdead_beef);
        write_record(&mut bytes, &[], 0);
        write_record(&mut bytes, &[9; 300], 7);
        let parsed = parse_bitstream(&bytes);
        assert_eq!(parsed.error, None);
        assert_eq!(parsed.packets.len(), 3);
        assert_eq!(parsed.packets[0], BitPacket { data: &[1, 2, 3], enc_final_range: 0xdead_beef });
        assert_eq!(parsed.packets[1].data, &[] as &[u8]);
        assert_eq!(parsed.packets[2].enc_final_range, 7);
        // A cut header ends the stream without an error: the first two
        // records take 19 bytes, and 6 bytes of the third header follow.
        let cut = parse_bitstream(&bytes[..25]);
        assert_eq!(cut.error, None);
        assert_eq!(cut.packets.len(), 2);
    }

    #[test]
    fn bad_records_keep_the_prefix() {
        let mut bytes = Vec::new();
        write_record(&mut bytes, &[1, 2, 3], 1);
        let good = bytes.len();
        write_record(&mut bytes, &[5; 10], 2);
        let cut = parse_bitstream(&bytes[..bytes.len() - 4]);
        assert_eq!(cut.packets.len(), 1);
        assert_eq!(cut.error, Some(BitstreamError::RanOutOfInput { index: 1, expected: 10, got: 6 }));
        assert_eq!(cut.error.unwrap().to_string(), "Ran out of input, expecting 10 bytes got 6");
        let mut bad = bytes[..good].to_vec();
        bad.extend_from_slice(&15001u32.to_be_bytes());
        bad.extend_from_slice(&[0; 8]);
        let parsed = parse_bitstream(&bad);
        assert_eq!(parsed.packets.len(), 1);
        assert_eq!(parsed.error, Some(BitstreamError::InvalidLength { index: 1, len: 15001 }));
        let mut negative = bytes[..good].to_vec();
        negative.extend_from_slice(&0xffff_fff0u32.to_be_bytes());
        assert_eq!(parse_bitstream(&negative).error.unwrap().to_string(), "Invalid payload length: -16");
    }

    #[test]
    fn sixteen_bit_conversion() {
        assert_eq!(opus_demo_s16(1.0), 32767);
        assert_eq!(opus_demo_s16(-1.0), -32767);
        assert_eq!(opus_demo_s16(0.0), 0);
        assert_eq!(opus_demo_s16(0.5 / 32768.0), 1);
        assert_eq!(opus_demo_s16(-0.5 / 32768.0), 0);
        assert_eq!(opus_demo_s16(1.5 / 32768.0), 2);
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        {
            assert_eq!(opus_demo_s16(f32::NAN), 0);
            assert_eq!(opus_demo_s16(1e30), 32767);
            assert_eq!(opus_demo_s16(-1e30), -32767);
        }
        assert_eq!(res2int24(0.5 / 8_388_608.0), 0);
        assert_eq!(res2int24(1.5 / 8_388_608.0), 2);
    }
}
