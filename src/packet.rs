//! Opus packet framing (RFC 6716 section 3 and appendix B): reading the
//! frames out of a packet, writing frames into a packet, and padding.
//!
//! This is a port of the framing code of libopus 1.6: `opus_packet_parse_impl`
//! (opus.c), `opus_repacketizer_out_range_impl` and `opus_packet_pad`
//! (repacketizer.c), and `opus_multistream_packet_validate`
//! (opus_multistream_decoder.c).
//!
//! Nothing here allocates. A packet is described by a small [`ParsedPacket`]
//! value and the frames stay where they are, in the caller's buffer.
//!
//! A packet starts with a table-of-contents (TOC) byte. Its two lowest bits
//! are the frame-count code:
//!
//! - code 0: one frame.
//! - code 1: two frames of the same size.
//! - code 2: two frames of different sizes.
//! - code 3: any number of frames, of one size (CBR) or with a size each
//!   (VBR), and optional padding at the end.
//!
//! "Self-delimited" framing (RFC 6716 appendix B) adds the size of the last
//! frame, so that a decoder knows where the packet ends. A multistream packet
//! uses it for every stream but the last.
//!
//! Not ported: the packet extensions libopus 1.5 and later can store in the
//! padding. This shows in [`pad`] and [`multistream_pad`] when the packet
//! already has padding whose bytes are not all zero:
//!
//! - libopus reads those bytes as extensions and carries them over into the
//!   new padding. These functions write zeros.
//! - libopus rejects the packet when the bytes are not valid extensions.
//!   These functions pad it.
//!
//! For a packet without padding, or with zero padding, the result is
//! libopus's byte for byte.

/// The most frames a packet can hold: 120 ms of 2.5 ms frames.
pub const MAX_FRAMES: usize = 48;
/// The largest frame the size fields can express.
pub const MAX_FRAME_BYTES: usize = 1275;
/// The most audio one packet holds: 120 ms, in samples per channel at
/// 48 kHz.
pub(crate) const MAX_PACKET_SAMPLES: usize = max_packet_samples(48000);

/// The most audio one packet holds, 120 ms, in samples per channel at `fs`
/// samples per second.
#[inline]
pub(crate) const fn max_packet_samples(fs: i32) -> usize {
    fs as usize / 25 * 3
}

const INVALID_PACKET: &str = "Invalid packet";
const BUFFER_TOO_SMALL: &str = "Output buffer too small";
const BAD_ARG: &str = "Invalid argument";

/// Result of parsing one Opus packet. libopus `opus_packet_parse_impl`.
#[derive(Debug, Clone, Copy)]
pub struct ParsedPacket {
    /// The table-of-contents byte.
    pub toc: u8,
    /// Number of frames, 1..=48.
    pub count: usize,
    /// Size of each frame in bytes; only `sizes[..count]` is meaningful.
    pub sizes: [u16; MAX_FRAMES],
    /// Offset of the first frame's first byte from the start of the packet.
    pub payload_offset: usize,
    /// Bytes this packet occupies, padding included. For a self-delimited
    /// packet this is where the next stream starts.
    pub packet_offset: usize,
    /// Bytes of padding at the end. They are the last `padding_len` bytes
    /// before `packet_offset`.
    pub padding_len: usize,
}

impl ParsedPacket {
    /// The frames of the packet `data` this value was parsed from, in order.
    pub fn frames<'a>(&self, data: &'a [u8]) -> Frames<'a, '_> {
        Frames {
            data,
            sizes: &self.sizes[..self.count],
            offset: self.payload_offset,
        }
    }

    /// Frame `index` of the packet `data` this value was parsed from. An
    /// index past the last frame gives an empty slice.
    pub fn frame<'a>(&self, data: &'a [u8], index: usize) -> &'a [u8] {
        self.frames(data).nth(index).unwrap_or(&[])
    }
}

/// Iterator over the frames of a parsed packet, see [`ParsedPacket::frames`].
#[derive(Debug, Clone)]
pub struct Frames<'a, 'p> {
    data: &'a [u8],
    sizes: &'p [u16],
    offset: usize,
}

impl<'a> Iterator for Frames<'a, '_> {
    type Item = &'a [u8];

    fn next(&mut self) -> Option<&'a [u8]> {
        let (&size, rest) = self.sizes.split_first()?;
        self.sizes = rest;
        let start = self.offset;
        self.offset += size as usize;
        // `parse` checked that every frame lies inside the packet. The
        // fallback only matters when `data` is not that packet.
        Some(self.data.get(start..self.offset).unwrap_or(&[]))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.sizes.len(), Some(self.sizes.len()))
    }
}

impl ExactSizeIterator for Frames<'_, '_> {}

/// libopus `opus_packet_get_samples_per_frame`: the duration of one frame of
/// a packet with this TOC byte, in samples at `sampling_rate`.
pub fn samples_per_frame(toc: u8, sampling_rate: i32) -> usize {
    let fs = sampling_rate.max(0) as i64;
    let audiosize = if toc & 0x80 != 0 {
        // CELT-only: 2.5, 5, 10 or 20 ms.
        (fs << ((toc >> 3) & 0x3)) / 400
    } else if toc & 0x60 == 0x60 {
        // Hybrid: 10 or 20 ms.
        if toc & 0x08 != 0 { fs / 50 } else { fs / 100 }
    } else {
        // SILK-only: 10, 20, 40 or 60 ms.
        let shift = (toc >> 3) & 0x3;
        if shift == 3 { fs * 60 / 1000 } else { (fs << shift) / 100 }
    };
    audiosize as usize
}

/// libopus `opus_packet_get_nb_frames`: the number of frames a packet says
/// it holds. Only the first two bytes are looked at; [`parse`] checks the
/// rest.
pub fn nb_frames(data: &[u8]) -> Result<usize, &'static str> {
    let Some(&toc) = data.first() else {
        return Err(BAD_ARG);
    };
    match toc & 0x3 {
        0 => Ok(1),
        1 | 2 => Ok(2),
        _ => match data.get(1) {
            Some(&count) => Ok((count & 0x3F) as usize),
            None => Err(INVALID_PACKET),
        },
    }
}

/// libopus `opus_packet_get_nb_samples`: the duration of a packet in
/// samples per channel at `sampling_rate`. More than 120 ms is invalid.
pub fn nb_samples(data: &[u8], sampling_rate: i32) -> Result<usize, &'static str> {
    let samples = nb_frames(data)? * samples_per_frame(data[0], sampling_rate);
    if samples as i64 * 25 > sampling_rate as i64 * 3 {
        return Err(INVALID_PACKET);
    }
    Ok(samples)
}

/// libopus `encode_size` (opus.c): the number of bytes a frame size takes.
#[inline]
pub(crate) fn size_field_len(size: u16) -> usize {
    1 + (size >= 252) as usize
}

/// libopus `encode_size` (opus.c): write a frame size, return the number of
/// bytes written (1 or 2).
#[inline]
fn encode_size(size: u16, out: &mut [u8]) -> usize {
    if size < 252 {
        out[0] = size as u8;
        1
    } else {
        let first = 252 + (size & 0x3);
        out[0] = first as u8;
        out[1] = ((size - first) >> 2) as u8;
        2
    }
}

/// libopus `parse_size` (opus.c): read a frame size at `data[pos]` when
/// `len` bytes are left. Returns the number of bytes read and the size, or
/// `None` when the field is cut short.
#[inline]
fn parse_size(data: &[u8], pos: usize, len: i64) -> Option<(i64, i64)> {
    if len < 1 {
        return None;
    }
    let first = *data.get(pos)? as i64;
    if first < 252 {
        Some((1, first))
    } else if len < 2 {
        None
    } else {
        Some((2, 4 * (*data.get(pos + 1)? as i64) + first))
    }
}

/// Parse the framing of one Opus packet: libopus `opus_packet_parse_impl`
/// (opus.c).
///
/// With `self_delimited`, `data` starts with a self-delimited packet and may
/// continue with other data, such as the next stream of a multistream
/// packet. [`ParsedPacket::packet_offset`] then says where the packet ends.
///
/// Returns an error wherever libopus returns `OPUS_INVALID_PACKET`.
pub fn parse(data: &[u8], self_delimited: bool) -> Result<ParsedPacket, &'static str> {
    let Some(&toc) = data.first() else {
        return Err(INVALID_PACKET);
    };
    let framesize = samples_per_frame(toc, 48000);

    // `pos` is the next byte to read. `len` is the C variable of the same
    // name: the bytes left for sizes and frames. It is signed because the
    // padding of a corrupt packet can claim more bytes than there are.
    let mut pos = 1usize;
    let mut len = data.len() as i64 - 1;
    let mut last_size = len;
    let mut cbr = false;
    let mut pad = 0i64;
    let mut sizes = [0u16; MAX_FRAMES];
    let count: usize;

    match toc & 0x3 {
        // One frame.
        0 => count = 1,
        // Two frames of the same size.
        1 => {
            count = 2;
            cbr = true;
            if !self_delimited {
                if len & 0x1 != 0 {
                    return Err(INVALID_PACKET);
                }
                last_size = len / 2;
            }
        }
        // Two frames of different sizes.
        2 => {
            count = 2;
            let Some((bytes, size)) = parse_size(data, pos, len) else {
                return Err(INVALID_PACKET);
            };
            len -= bytes;
            if size > len {
                return Err(INVALID_PACKET);
            }
            pos += bytes as usize;
            sizes[0] = size as u16;
            last_size = len - size;
        }
        // Any number of frames, 0 to 120 ms in total.
        _ => {
            if len < 1 {
                return Err(INVALID_PACKET);
            }
            // The number of frames is in bits 0 to 5.
            let ch = data[pos];
            pos += 1;
            count = (ch & 0x3F) as usize;
            if count == 0 || framesize * count > MAX_PACKET_SAMPLES {
                return Err(INVALID_PACKET);
            }
            len -= 1;
            // Bit 6 says that padding follows.
            if ch & 0x40 != 0 {
                loop {
                    if len <= 0 {
                        return Err(INVALID_PACKET);
                    }
                    let p = data[pos];
                    pos += 1;
                    len -= 1;
                    let tmp = if p == 255 { 254 } else { p as i64 };
                    len -= tmp;
                    pad += tmp;
                    if p != 255 {
                        break;
                    }
                }
            }
            if len < 0 {
                return Err(INVALID_PACKET);
            }
            // Bit 7 says that the frames have a size each (VBR).
            cbr = ch & 0x80 == 0;
            if !cbr {
                last_size = len;
                for size_out in sizes.iter_mut().take(count - 1) {
                    let Some((bytes, size)) = parse_size(data, pos, len) else {
                        return Err(INVALID_PACKET);
                    };
                    len -= bytes;
                    if size > len {
                        return Err(INVALID_PACKET);
                    }
                    pos += bytes as usize;
                    *size_out = size as u16;
                    last_size -= bytes + size;
                }
                if last_size < 0 {
                    return Err(INVALID_PACKET);
                }
            } else if !self_delimited {
                last_size = len / count as i64;
                if last_size * count as i64 != len {
                    return Err(INVALID_PACKET);
                }
            }
        }
    }

    if self_delimited {
        // Self-delimited framing has an extra size for the last frame.
        let Some((bytes, size)) = parse_size(data, pos, len) else {
            return Err(INVALID_PACKET);
        };
        len -= bytes;
        if size > len {
            return Err(INVALID_PACKET);
        }
        pos += bytes as usize;
        if cbr {
            // For CBR packets the size applies to all the frames.
            if size * count as i64 > len {
                return Err(INVALID_PACKET);
            }
            sizes[..count].fill(size as u16);
        } else {
            if bytes + size > last_size {
                return Err(INVALID_PACKET);
            }
            sizes[count - 1] = size as u16;
        }
    } else {
        // The size of the last frame (of all frames for CBR) is not coded,
        // so it can come out larger than a frame may be.
        if last_size > MAX_FRAME_BYTES as i64 {
            return Err(INVALID_PACKET);
        }
        if cbr {
            sizes[..count].fill(last_size as u16);
        } else {
            sizes[count - 1] = last_size as u16;
        }
    }

    let frame_bytes: usize = sizes[..count].iter().map(|&s| s as usize).sum();
    Ok(ParsedPacket {
        toc,
        count,
        sizes,
        payload_offset: pos,
        packet_offset: pad as usize + pos + frame_bytes,
        padding_len: pad as usize,
    })
}

/// libopus `opus_multistream_packet_validate` (opus_multistream_decoder.c):
/// check the framing of a multistream packet of `nb_streams` streams.
///
/// Every stream but the last is self-delimited. All streams must have the
/// same duration. Returns that duration in samples per channel at
/// `sampling_rate`.
pub fn multistream_validate(
    data: &[u8],
    nb_streams: usize,
    sampling_rate: i32,
) -> Result<usize, &'static str> {
    if nb_streams == 0 {
        return Err(BAD_ARG);
    }
    let mut rest = data;
    let mut samples = 0usize;
    for s in 0..nb_streams {
        if rest.is_empty() {
            return Err(INVALID_PACKET);
        }
        let packet = parse(rest, s != nb_streams - 1)?;
        // opus_packet_get_nb_samples: no more than 120 ms.
        let tmp_samples = packet.count * samples_per_frame(packet.toc, sampling_rate);
        if tmp_samples as i64 * 25 > sampling_rate as i64 * 3 {
            return Err(INVALID_PACKET);
        }
        if s != 0 && samples != tmp_samples {
            return Err(INVALID_PACKET);
        }
        samples = tmp_samples;
        rest = rest.get(packet.packet_offset..).ok_or(INVALID_PACKET)?;
    }
    Ok(samples)
}

/// Where the parts of a packet go: the decisions
/// `opus_repacketizer_out_range_impl` makes before it writes anything.
struct Layout {
    /// Frame-count code, 0..=3.
    code: u8,
    /// Code 3 only: the frames have a size each.
    vbr: bool,
    /// Code 3 only: bytes of padding, counting the bytes that code its
    /// length.
    pad_amount: usize,
    /// Bytes before the first frame.
    header_len: usize,
    /// Length of the whole packet.
    total: usize,
}

/// Lay out a packet of frames with the given `sizes` in at most `maxlen`
/// bytes: the first half of `opus_repacketizer_out_range_impl`.
fn layout(
    sizes: &[u16],
    maxlen: usize,
    self_delimited: bool,
    pad: bool,
) -> Result<Layout, &'static str> {
    let count = sizes.len();
    let payload: usize = sizes.iter().map(|&s| s as usize).sum();
    // Self-delimited framing codes the size of the last frame as well.
    let last_size_len = if self_delimited { size_field_len(sizes[count - 1]) } else { 0 };

    let mut code = 3u8;
    let mut header_len = 0usize;
    if count == 1 {
        code = 0;
        header_len = 1;
    } else if count == 2 {
        if sizes[1] == sizes[0] {
            code = 1;
            header_len = 1;
        } else {
            code = 2;
            header_len = 1 + size_field_len(sizes[0]);
        }
    }
    header_len += last_size_len;
    if count <= 2 && header_len + payload > maxlen {
        return Err(BUFFER_TOO_SMALL);
    }

    let mut vbr = false;
    let mut pad_amount = 0usize;
    if count > 2 || (pad && header_len + payload < maxlen) {
        // Code 3. Padding needs it whatever the number of frames.
        code = 3;
        vbr = sizes[1..].iter().any(|&s| s != sizes[0]);
        header_len = 2 + last_size_len;
        if vbr {
            header_len += sizes[..count - 1].iter().map(|&s| size_field_len(s)).sum::<usize>();
        }
        if header_len + payload > maxlen {
            return Err(BUFFER_TOO_SMALL);
        }
        if pad {
            pad_amount = maxlen - header_len - payload;
        }
        if pad_amount != 0 {
            // The length of the padding takes `nb_255s + 1` bytes of it.
            header_len += (pad_amount - 1) / 255 + 1;
        }
    }
    let padding_zeros = if pad_amount != 0 { pad_amount - (pad_amount - 1) / 255 - 1 } else { 0 };
    Ok(Layout {
        code,
        vbr,
        pad_amount,
        header_len,
        total: header_len + payload + padding_zeros,
    })
}

/// Write the bytes before the first frame into `out[..plan.header_len]`:
/// the TOC byte, the frame count, the padding length and the frame sizes.
fn write_header(toc: u8, sizes: &[u16], plan: &Layout, self_delimited: bool, out: &mut [u8]) {
    let count = sizes.len();
    out[0] = (toc & 0xFC) | plan.code;
    let mut pos = 1usize;
    if plan.code == 2 {
        pos += encode_size(sizes[0], &mut out[pos..]);
    } else if plan.code == 3 {
        out[1] = count as u8 | if plan.vbr { 0x80 } else { 0 };
        pos = 2;
        if plan.pad_amount != 0 {
            out[1] |= 0x40;
            let nb_255s = (plan.pad_amount - 1) / 255;
            out[pos..pos + nb_255s].fill(255);
            pos += nb_255s;
            out[pos] = (plan.pad_amount - 255 * nb_255s - 1) as u8;
            pos += 1;
        }
        if plan.vbr {
            for &size in &sizes[..count - 1] {
                pos += encode_size(size, &mut out[pos..]);
            }
        }
    }
    if self_delimited {
        pos += encode_size(sizes[count - 1], &mut out[pos..]);
    }
    debug_assert_eq!(pos, plan.header_len);
}

/// Check the frames a packet is to be made of, and collect their sizes.
fn frame_sizes(toc: u8, frames: &[&[u8]]) -> Result<[u16; MAX_FRAMES], &'static str> {
    let count = frames.len();
    // The repacketizer takes no more than 120 ms of audio
    // (opus_repacketizer_cat_impl).
    if count == 0 || count > MAX_FRAMES || count * samples_per_frame(toc, 48000) > MAX_PACKET_SAMPLES {
        return Err(BAD_ARG);
    }
    let mut sizes = [0u16; MAX_FRAMES];
    for (size, frame) in sizes.iter_mut().zip(frames) {
        if frame.len() > MAX_FRAME_BYTES {
            return Err(BAD_ARG);
        }
        *size = frame.len() as u16;
    }
    Ok(sizes)
}

/// Write one packet made of `frames`: libopus
/// `opus_repacketizer_out_range_impl` (repacketizer.c) without extensions.
///
/// - `toc` supplies the configuration and the stereo flag; its frame-count
///   code is replaced by the one that fits the frames.
/// - `self_delimited` adds the size of the last frame.
/// - `pad` fills `out` to its full length with padding. The packet then is
///   exactly `out.len()` bytes long.
///
/// Returns the number of bytes written.
pub fn write(
    toc: u8,
    frames: &[&[u8]],
    out: &mut [u8],
    self_delimited: bool,
    pad: bool,
) -> Result<usize, &'static str> {
    let sizes = frame_sizes(toc, frames)?;
    let sizes = &sizes[..frames.len()];
    let plan = layout(sizes, out.len(), self_delimited, pad)?;
    write_header(toc, sizes, &plan, self_delimited, out);
    let mut pos = plan.header_len;
    for frame in frames {
        out[pos..pos + frame.len()].copy_from_slice(frame);
        pos += frame.len();
    }
    if pad {
        // Fill the padding with zeros.
        out[pos..].fill(0);
    }
    Ok(plan.total)
}

/// Pad the packet `data[..len]` in place to `new_len` bytes: libopus
/// `opus_packet_pad` (repacketizer.c).
///
/// The result is a code 3 packet with zero padding, unless one more byte is
/// all that is needed. `data` must hold at least `new_len` bytes.
///
/// Padding the packet already has is replaced by zeros: packet extensions
/// are not ported (see the module documentation).
pub fn pad(data: &mut [u8], len: usize, new_len: usize) -> Result<(), &'static str> {
    if len < 1 || len > data.len() {
        return Err(BAD_ARG);
    }
    if len == new_len {
        return Ok(());
    }
    if len > new_len || new_len > data.len() {
        return Err(BAD_ARG);
    }
    let packet = parse(&data[..len], false)?;
    let sizes = &packet.sizes[..packet.count];
    let plan = layout(sizes, new_len, false, true)?;
    // libopus copies the packet and writes the new one over the old. Here
    // the frames, which are one block of bytes, move to their new place
    // first. The new header is built from the parsed sizes alone, so it can
    // overwrite what is left of the old one.
    let payload: usize = sizes.iter().map(|&s| s as usize).sum();
    data.copy_within(packet.payload_offset..packet.payload_offset + payload, plan.header_len);
    write_header(packet.toc, sizes, &plan, false, data);
    data[plan.header_len + payload..new_len].fill(0);
    Ok(())
}

/// Pad the multistream packet `data[..len]` of `nb_streams` streams in place
/// to `new_len` bytes: libopus `opus_multistream_packet_pad`
/// (repacketizer.c). The padding goes into the last stream.
pub fn multistream_pad(
    data: &mut [u8],
    len: usize,
    new_len: usize,
    nb_streams: usize,
) -> Result<(), &'static str> {
    if len < 1 || len > data.len() {
        return Err(BAD_ARG);
    }
    if len == new_len {
        return Ok(());
    }
    if len > new_len || new_len > data.len() {
        return Err(BAD_ARG);
    }
    let amount = new_len - len;
    // Seek to the last stream.
    let mut start = 0usize;
    for _ in 1..nb_streams {
        if start >= len {
            return Err(INVALID_PACKET);
        }
        start += parse(&data[start..len], true)?.packet_offset;
    }
    let last_len = len.checked_sub(start).ok_or(INVALID_PACKET)?;
    pad(&mut data[start..], last_len, last_len + amount)
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;

    /// TOC byte of a CELT-only fullband 20 ms mono packet.
    const TOC_CELT_20: u8 = 0xF8;
    /// TOC byte of a CELT-only fullband 2.5 ms mono packet.
    const TOC_CELT_2_5: u8 = 0xE0;

    #[test]
    fn samples_per_frame_covers_every_configuration() {
        // SILK-only narrowband: 10, 20, 40, 60 ms.
        assert_eq!(samples_per_frame(0x00, 48000), 480);
        assert_eq!(samples_per_frame(0x08, 48000), 960);
        assert_eq!(samples_per_frame(0x10, 48000), 1920);
        assert_eq!(samples_per_frame(0x18, 48000), 2880);
        // Hybrid: 10 and 20 ms.
        assert_eq!(samples_per_frame(0x60, 16000), 160);
        assert_eq!(samples_per_frame(0x68, 16000), 320);
        // CELT-only: 2.5, 5, 10, 20 ms.
        assert_eq!(samples_per_frame(0x80, 8000), 20);
        assert_eq!(samples_per_frame(0x88, 8000), 40);
        assert_eq!(samples_per_frame(0x90, 8000), 80);
        assert_eq!(samples_per_frame(0x98, 8000), 160);
    }

    #[test]
    fn code_0_is_one_frame_to_the_end() {
        let p = parse(&[TOC_CELT_20, 1, 2, 3], false).unwrap();
        assert_eq!((p.count, p.sizes[0], p.payload_offset, p.packet_offset), (1, 3, 1, 4));
        // A TOC byte alone is one empty frame.
        let p = parse(&[TOC_CELT_20], false).unwrap();
        assert_eq!((p.count, p.sizes[0], p.packet_offset), (1, 0, 1));
        assert!(parse(&[], false).is_err());
    }

    #[test]
    fn code_1_needs_an_even_payload() {
        let p = parse(&[TOC_CELT_20 | 1, 1, 2, 3, 4], false).unwrap();
        assert_eq!((p.count, p.sizes[0], p.sizes[1]), (2, 2, 2));
        assert_eq!(p.frame(&[TOC_CELT_20 | 1, 1, 2, 3, 4], 1), &[3, 4]);
        assert!(parse(&[TOC_CELT_20 | 1, 1, 2, 3], false).is_err());
    }

    #[test]
    fn code_2_reads_the_first_size() {
        let data = [TOC_CELT_20 | 2, 1, 9, 8, 7];
        let p = parse(&data, false).unwrap();
        assert_eq!((p.count, p.sizes[0], p.sizes[1], p.payload_offset), (2, 1, 2, 2));
        let frames: Vec<&[u8]> = p.frames(&data).collect();
        assert_eq!(frames, [&[9u8][..], &[8, 7][..]]);
        // The first size runs past the end.
        assert!(parse(&[TOC_CELT_20 | 2, 4, 9, 8, 7], false).is_err());
        // The size field is missing.
        assert!(parse(&[TOC_CELT_20 | 2], false).is_err());
    }

    #[test]
    fn code_3_checks_count_duration_and_padding() {
        // No frames.
        assert!(parse(&[TOC_CELT_20 | 3, 0], false).is_err());
        // 7 frames of 20 ms are more than 120 ms.
        assert!(parse(&[TOC_CELT_20 | 3, 7], false).is_err());
        // 48 frames of 2.5 ms are exactly 120 ms.
        assert_eq!(parse(&[TOC_CELT_2_5 | 3, 48], false).unwrap().count, 48);
        assert!(parse(&[TOC_CELT_2_5 | 3, 49], false).is_err());
        // CBR: the payload must divide by the count.
        assert!(parse(&[TOC_CELT_20 | 3, 2, 1, 2, 3], false).is_err());
        // Padding: 2 bytes at the end are not part of a frame.
        let p = parse(&[TOC_CELT_20 | 3, 0x41, 2, 7, 0, 0], false).unwrap();
        assert_eq!((p.count, p.sizes[0], p.padding_len, p.packet_offset), (1, 1, 2, 6));
        // Padding longer than the packet.
        assert!(parse(&[TOC_CELT_20 | 3, 0x41, 9, 7], false).is_err());
        // The padding flag without a length byte.
        assert!(parse(&[TOC_CELT_20 | 3, 0x41], false).is_err());
    }

    #[test]
    fn self_delimited_packet_ends_before_the_next_stream() {
        // Code 0, size 2, then 3 bytes of another stream.
        let data = [TOC_CELT_20, 2, 5, 6, 0xAA, 0xBB, 0xCC];
        let p = parse(&data, true).unwrap();
        assert_eq!((p.count, p.sizes[0], p.payload_offset, p.packet_offset), (1, 2, 2, 4));
        // Code 1: the one size applies to both frames.
        let data = [TOC_CELT_20 | 1, 2, 5, 6, 7, 8, 0xAA];
        let p = parse(&data, true).unwrap();
        assert_eq!((p.count, p.sizes[0], p.sizes[1], p.packet_offset), (2, 2, 2, 6));
        // The size does not fit.
        assert!(parse(&[TOC_CELT_20, 3, 5, 6], true).is_err());
        assert!(parse(&[TOC_CELT_20 | 1, 2, 5, 6, 7], true).is_err());
    }

    #[test]
    fn a_frame_may_not_exceed_1275_bytes() {
        let mut data = vec![0u8; 1 + 1276];
        data[0] = TOC_CELT_20;
        assert!(parse(&data, false).is_err());
        assert_eq!(parse(&data[..1276], false).unwrap().sizes[0], 1275);
    }

    #[test]
    fn write_picks_the_frame_count_code() {
        let mut out = [0u8; 64];
        let a = [1u8, 2, 3];
        let b = [4u8, 5, 6];
        let c = [7u8];
        // One frame: code 0.
        let n = write(TOC_CELT_20 | 3, &[&a], &mut out, false, false).unwrap();
        assert_eq!(&out[..n], &[TOC_CELT_20, 1, 2, 3]);
        // Two frames of one size: code 1.
        let n = write(TOC_CELT_20, &[&a, &b], &mut out, false, false).unwrap();
        assert_eq!(&out[..n], &[TOC_CELT_20 | 1, 1, 2, 3, 4, 5, 6]);
        // Two frames of different sizes: code 2.
        let n = write(TOC_CELT_20, &[&a, &c], &mut out, false, false).unwrap();
        assert_eq!(&out[..n], &[TOC_CELT_20 | 2, 3, 1, 2, 3, 7]);
        // Three frames of one size: code 3, CBR.
        let n = write(TOC_CELT_20, &[&a, &b, &a], &mut out, false, false).unwrap();
        assert_eq!(&out[..n], &[TOC_CELT_20 | 3, 3, 1, 2, 3, 4, 5, 6, 1, 2, 3]);
        // Three frames of different sizes: code 3, VBR.
        let n = write(TOC_CELT_20, &[&a, &c, &b], &mut out, false, false).unwrap();
        assert_eq!(&out[..n], &[TOC_CELT_20 | 3, 0x83, 3, 1, 1, 2, 3, 7, 4, 5, 6]);
    }

    #[test]
    fn write_self_delimited_adds_the_last_size() {
        let mut out = [0u8; 16];
        let n = write(TOC_CELT_20, &[&[9, 9]], &mut out, true, false).unwrap();
        assert_eq!(&out[..n], &[TOC_CELT_20, 2, 9, 9]);
        let p = parse(&out[..n], true).unwrap();
        assert_eq!((p.count, p.sizes[0], p.packet_offset), (1, 2, n));
    }

    #[test]
    fn write_pads_to_the_exact_length() {
        let frame = [5u8; 4];
        // One byte short of the buffer: code 3 without padding.
        let mut out = [0xEEu8; 6];
        assert_eq!(write(TOC_CELT_20, &[&frame], &mut out, false, true), Ok(6));
        assert_eq!(out, [TOC_CELT_20 | 3, 1, 5, 5, 5, 5]);
        // The buffer fits exactly: the packet stays code 0.
        let mut out = [0xEEu8; 5];
        assert_eq!(write(TOC_CELT_20, &[&frame], &mut out, false, true), Ok(5));
        assert_eq!(out, [TOC_CELT_20, 5, 5, 5, 5]);
        // More room: a length byte and zeros.
        let mut out = [0xEEu8; 10];
        assert_eq!(write(TOC_CELT_20, &[&frame], &mut out, false, true), Ok(10));
        assert_eq!(out, [TOC_CELT_20 | 3, 0x41, 3, 5, 5, 5, 5, 0, 0, 0]);
        let p = parse(&out, false).unwrap();
        assert_eq!((p.count, p.sizes[0], p.padding_len), (1, 4, 3));
    }

    #[test]
    fn write_rejects_bad_input_without_panicking() {
        let mut out = [0u8; 8];
        assert!(write(TOC_CELT_20, &[], &mut out, false, false).is_err());
        assert!(write(TOC_CELT_20, &[&[0; 9]], &mut out, false, false).is_err());
        assert!(write(TOC_CELT_20, &[&[0; 1276]], &mut [0u8; 2000], false, false).is_err());
        // 7 frames of 20 ms are more than 120 ms.
        let f: &[u8] = &[1];
        assert!(write(TOC_CELT_20, &[f; 7], &mut [0u8; 64], false, false).is_err());
        assert!(write(TOC_CELT_20, &[f], &mut [], false, false).is_err());
    }

    #[test]
    fn pad_moves_the_frames_in_place() {
        let mut data = [TOC_CELT_20 | 2, 1, 9, 8, 7, 0, 0, 0, 0, 0];
        pad(&mut data, 5, 10).unwrap();
        // Code 3, VBR, 2 frames, padding: 1 length byte and 3 zeros.
        assert_eq!(data, [TOC_CELT_20 | 3, 0xC2, 3, 1, 9, 8, 7, 0, 0, 0]);
        let p = parse(&data, false).unwrap();
        assert_eq!((p.count, p.sizes[0], p.sizes[1], p.padding_len), (2, 1, 2, 3));
        // Nothing to do, and arguments that make no sense.
        assert_eq!(pad(&mut data, 10, 10), Ok(()));
        assert!(pad(&mut data, 10, 9).is_err());
        assert!(pad(&mut data, 0, 5).is_err());
        assert!(pad(&mut data, 5, 11).is_err());
    }

    #[test]
    fn multistream_validate_wants_equal_durations() {
        // Stream 0: self-delimited 20 ms. Stream 1: 20 ms.
        let ok = [TOC_CELT_20, 1, 7, TOC_CELT_20, 8, 9];
        assert_eq!(multistream_validate(&ok, 2, 48000), Ok(960));
        // Stream 1 is 2.5 ms.
        let bad = [TOC_CELT_20, 1, 7, TOC_CELT_2_5, 8, 9];
        assert!(multistream_validate(&bad, 2, 48000).is_err());
        // The second stream is missing.
        assert!(multistream_validate(&[TOC_CELT_20, 1, 7], 2, 48000).is_err());
        assert!(multistream_validate(&ok, 0, 48000).is_err());
    }

    #[test]
    fn multistream_pad_pads_the_last_stream() {
        let mut data = [TOC_CELT_20, 1, 7, TOC_CELT_20, 8, 9, 0, 0, 0];
        multistream_pad(&mut data, 6, 9, 2).unwrap();
        assert_eq!(data, [TOC_CELT_20, 1, 7, TOC_CELT_20 | 3, 0x41, 1, 8, 9, 0]);
        assert_eq!(multistream_validate(&data, 2, 48000), Ok(960));
    }
}
