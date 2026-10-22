//! Packet extensions: extra data stored in the padding of an Opus packet.
//!
//! This is a port of `src/extensions.c` of libopus 1.6.1. libopus 1.5 and
//! later can store extensions in the padding of a code 3 packet (see
//! [`crate::packet`]). Deep redundancy (DRED) uses them. A decoder that does
//! not know an extension skips it, and an old decoder sees plain padding.
//!
//! The padding is a sequence of entries. Each entry starts with one byte:
//! the extension ID in its upper 7 bits and a flag `L` in its lowest bit.
//!
//! - ID 0: padding. With `L = 1` it is one byte; with `L = 0` it fills the
//!   rest of the padding.
//! - ID 1: frame separator. The entries after it belong to the next frame
//!   (`L = 0`), or to the frame a following byte further on (`L = 1`; an
//!   increment of 0 does nothing). A separator past the last frame of the
//!   packet makes the padding invalid.
//! - ID 2: repeat. The extensions since the last separator (or since the
//!   start, or since the last repeat) are repeated for every later frame.
//!   Only their payloads follow, frame by frame; their ID bytes are not
//!   written again. With `L = 0`, the last repeated long extension of the
//!   last frame has no length (it runs to the end), and the frame number
//!   moves on by one after the repeats.
//! - IDs 3 to 31: short extensions. `L` is the payload length, 0 or 1 byte.
//! - IDs 32 to 127: long extensions. With `L = 1` a length follows, coded
//!   like the padding length of a packet: bytes of 255 add 255 each and
//!   continue, the first byte below 255 ends the length. With `L = 0` the
//!   payload runs to the end of the padding, except for the payload bytes of
//!   short extensions that a repeat adds after it.
//!
//! Frame numbers count the frames of the packet from 0. The entries before
//! the first separator belong to frame 0.
//!
//! Nothing here allocates, except `ExtensionList` beyond its fixed size
//! with the feature `alloc`.

use crate::Error;
use crate::packet::Padding;

/// The extension IDs 0, 1 and 2 are padding, separators and repeats. The
/// IDs of real extensions are 3 to 127.
pub const MIN_ID: u8 = 3;
/// The largest extension ID.
pub const MAX_ID: u8 = 127;

/// The most frames of a packet (120 ms of 2.5 ms frames).
const MAX_FRAMES: usize = 48;

/// `nb_frames` is above 48. libopus only asserts this.
const TOO_MANY_FRAMES: Error = Error::bad_arg("invalid argument: nb_frames is above 48");
/// `nb_frame_exts` holds fewer than `nb_frames` values.
const COUNTS_TOO_SHORT: Error = Error::bad_arg("invalid argument: nb_frame_exts is shorter than nb_frames");

/// One extension: libopus `opus_extension_data`.
///
/// The fields can be read. To make one, use [`Extension::new`]: more fields
/// may be added in a later release.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct Extension<'a> {
    /// The ID, 3 to 127. IDs 3 to 31 are short extensions with at most one
    /// byte of payload, IDs 32 to 127 long extensions.
    pub id: u8,
    /// The frame of the packet the extension belongs to, from 0.
    pub frame: usize,
    /// The payload.
    pub data: &'a [u8],
}

impl<'a> Extension<'a> {
    /// An extension with ID `id` for frame `frame` of a packet, with the
    /// payload `data`. It is not checked here: [`generate`] refuses an ID
    /// outside 3 to 127, a frame past the last frame, and a short extension
    /// (ID below 32) with more than one byte.
    pub const fn new(id: u8, frame: usize, data: &'a [u8]) -> Self {
        Extension { id, frame, data }
    }
}

/// Skips the payload of an extension whose ID byte was `id_byte`. The
/// payload starts at `pos`; `len` bytes are left from there. Returns the
/// position after the payload, the bytes left, and the bytes of the length
/// field of a long extension. `None` when the payload does not fit.
/// libopus `skip_extension_payload`.
///
/// A repeat (ID 2) has no payload here: its repeated payloads need the
/// iterator.
fn skip_extension_payload(
    data: &[u8],
    pos: usize,
    len: usize,
    id_byte: u8,
    trailing_short_len: usize,
) -> Option<(usize, usize, usize)> {
    let id = id_byte >> 1;
    let l = (id_byte & 1) as usize;
    let (mut pos, mut len) = (pos, len);
    let mut header_size = 0;
    if (id == 0 && l == 1) || id == 2 {
        // Nothing to do.
    } else if id > 0 && id < 32 {
        if len < l {
            return None;
        }
        pos += l;
        len -= l;
    } else if l == 0 {
        // The payload runs to the end, but for the short payloads that a
        // repeat adds after it.
        if len < trailing_short_len {
            return None;
        }
        pos += len - trailing_short_len;
        len = trailing_short_len;
    } else {
        let mut bytes = 0usize;
        loop {
            if len < 1 {
                return None;
            }
            let lacing = data[pos] as usize;
            pos += 1;
            bytes += lacing;
            header_size += 1;
            // libopus lets `len` go below zero here and fails afterwards.
            if len < lacing + 1 {
                return None;
            }
            len -= lacing + 1;
            if lacing != 255 {
                break;
            }
        }
        pos += bytes;
    }
    Some((pos, len, header_size))
}

/// Skips one extension, its ID byte included. Returns the position after it,
/// the bytes left and the bytes before its payload (the ID byte and any
/// length bytes). libopus `skip_extension`. A repeat only skips its ID byte.
fn skip_extension(data: &[u8], pos: usize, len: usize) -> Option<(usize, usize, usize)> {
    if len == 0 {
        return Some((pos, 0, 0));
    }
    let id_byte = data[pos];
    let (pos, len, header_size) = skip_extension_payload(data, pos + 1, len - 1, id_byte, 0)?;
    Some((pos, len, header_size + 1))
}

/// Reads the extensions of a padding one by one: libopus
/// `OpusExtensionIterator`.
///
/// It is an [`Iterator`] over `Result<Extension, Error>`. It returns every
/// extension but padding, separators and repeats, and includes the
/// extensions that a repeat adds. They come in the order of the bytes,
/// which is not always the order of the frames: repeated extensions come
/// right after the repeat.
///
/// Errors: [`InvalidPacket`](crate::ErrorKind::InvalidPacket) when the
/// padding is not valid, and [`BadArg`](crate::ErrorKind::BadArg) when
/// `nb_frames` is above 48. After an error the iterator ends: later calls of
/// `next` return `None`, until [`reset`](Self::reset). (libopus returns the
/// error again on every later call.)
///
/// ```
/// use opus1::extensions::{Extension, ExtensionIterator};
///
/// // ID 3 with one byte in frame 0, a separator, ID 4 without payload in
/// // frame 1.
/// let padding = [0x07, 0xAA, 0x02, 0x08];
/// let found: Vec<Extension> = ExtensionIterator::new(&padding, 2).collect::<Result<_, _>>().unwrap();
/// assert_eq!(found, [Extension::new(3, 0, &[0xAA]), Extension::new(4, 1, &[])]);
///
/// // A separator past the last frame makes the padding invalid.
/// let mut iter = ExtensionIterator::new(&padding, 1);
/// assert!(iter.next().unwrap().is_ok());
/// assert!(iter.next().unwrap().is_err());
/// assert!(iter.next().is_none());
/// ```
#[derive(Clone, Debug)]
pub struct ExtensionIterator<'a> {
    data: &'a [u8],
    /// Position of the next entry and bytes left from there.
    curr: usize,
    curr_len: usize,
    /// The padding was found invalid (libopus: `curr_len < 0`).
    failed: bool,
    /// Where the extensions that a repeat would repeat start.
    repeat_data: usize,
    /// While repeating: the bytes of those extensions.
    repeat_len: usize,
    /// The end of the last long extension since `repeat_data`.
    last_long: Option<usize>,
    /// While repeating: the next extension to repeat, and the bytes left.
    src: usize,
    src_len: usize,
    /// The payload bytes of short extensions since the last long one.
    trailing_short_len: usize,
    nb_frames: usize,
    frame_max: usize,
    curr_frame: usize,
    /// While repeating: the frame the repeated extensions are for; 0 when
    /// not repeating.
    repeat_frame: usize,
    /// The `L` flag of the repeat.
    repeat_l: u8,
}

impl<'a> ExtensionIterator<'a> {
    /// An iterator over the padding `data` of a packet of `nb_frames`
    /// frames. libopus `opus_extension_iterator_init`.
    ///
    /// When `nb_frames` is above 48, which libopus only asserts, the first
    /// call of `next` returns [`BadArg`](crate::ErrorKind::BadArg).
    pub fn new(data: &'a [u8], nb_frames: usize) -> Self {
        ExtensionIterator {
            data,
            curr: 0,
            curr_len: data.len(),
            failed: false,
            repeat_data: 0,
            repeat_len: 0,
            last_long: None,
            src: 0,
            src_len: 0,
            trailing_short_len: 0,
            nb_frames,
            frame_max: nb_frames,
            curr_frame: 0,
            repeat_frame: 0,
            repeat_l: 0,
        }
    }

    /// Starts again from the first extension. libopus
    /// `opus_extension_iterator_reset`.
    ///
    /// As in libopus, this also clears an error, and keeps the limit of
    /// [`set_frame_max`](Self::set_frame_max).
    pub fn reset(&mut self) {
        self.failed = false;
        self.repeat_data = 0;
        self.curr = 0;
        self.last_long = None;
        self.curr_len = self.data.len();
        self.repeat_frame = 0;
        self.curr_frame = 0;
        self.trailing_short_len = 0;
    }

    /// Returns no extension of frame `frame_max` or later, and stops early
    /// when only those are left. libopus
    /// `opus_extension_iterator_set_frame_max`.
    pub fn set_frame_max(&mut self, frame_max: usize) {
        self.frame_max = frame_max;
    }

    fn fail(&mut self) -> Error {
        self.failed = true;
        Error::INVALID_PACKET
    }

    /// The next repeated extension, or `None` when the repeats are done.
    /// libopus `opus_extension_iterator_next_repeat`.
    fn next_repeat(&mut self) -> Result<Option<Extension<'a>>, Error> {
        debug_assert!(self.repeat_frame > 0);
        while self.repeat_frame < self.nb_frames {
            while self.src_len > 0 {
                let mut repeat_id_byte = self.data[self.src];
                // These extensions were skipped before, so this cannot fail.
                let Some((src, src_len, _)) = skip_extension(self.data, self.src, self.src_len) else {
                    return Err(self.fail());
                };
                self.src = src;
                self.src_len = src_len;
                // Padding and separators with an increment of 0 are not
                // repeated.
                if repeat_id_byte <= 3 {
                    continue;
                }
                // With L = 0, the last repeated long extension of the last
                // frame runs to the end.
                if self.repeat_l == 0 && self.repeat_frame + 1 >= self.nb_frames && Some(self.src) == self.last_long {
                    repeat_id_byte &= !1;
                }
                let curr0 = self.curr;
                let Some((curr, curr_len, header_size)) =
                    skip_extension_payload(self.data, self.curr, self.curr_len, repeat_id_byte, self.trailing_short_len)
                else {
                    return Err(self.fail());
                };
                self.curr = curr;
                self.curr_len = curr_len;
                if self.repeat_frame >= self.frame_max {
                    continue;
                }
                return Ok(Some(Extension {
                    id: repeat_id_byte >> 1,
                    frame: self.repeat_frame,
                    data: &self.data[curr0 + header_size..self.curr],
                }));
            }
            // The extensions of this frame are done.
            self.src = self.repeat_data;
            self.src_len = self.repeat_len;
            self.repeat_frame += 1;
        }
        // The repeats are done.
        self.repeat_data = self.curr;
        self.last_long = None;
        // With L = 0, the frame number moves on, in case a long extension
        // with L = 0 did not take all the data.
        if self.repeat_l == 0 {
            self.curr_frame += 1;
            // Ignore further padding after the last frame.
            if self.curr_frame >= self.nb_frames {
                self.curr_len = 0;
            }
        }
        self.repeat_frame = 0;
        Ok(None)
    }

    /// The next extension, `None` at the end, or an error
    /// ([`InvalidPacket`](crate::ErrorKind::InvalidPacket)) when the padding is
    /// not valid. libopus `opus_extension_iterator_next`.
    fn next_extension(&mut self) -> Result<Option<Extension<'a>>, Error> {
        if self.repeat_frame > 0 {
            // In the middle of a repeat.
            if let Some(ext) = self.next_repeat()? {
                return Ok(Some(ext));
            }
        }
        // Checked here so that set_frame_max can be called at any time.
        if self.curr_frame >= self.frame_max {
            return Ok(None);
        }
        while self.curr_len > 0 {
            let curr0 = self.curr;
            let id = self.data[curr0] >> 1;
            let l = self.data[curr0] & 1;
            let Some((curr, curr_len, header_size)) = skip_extension(self.data, self.curr, self.curr_len) else {
                return Err(self.fail());
            };
            self.curr = curr;
            self.curr_len = curr_len;
            if id == 1 {
                if l == 0 {
                    self.curr_frame += 1;
                } else {
                    // An increment of 0 does nothing.
                    let increment = self.data[curr0 + 1] as usize;
                    if increment == 0 {
                        continue;
                    }
                    self.curr_frame += increment;
                }
                if self.curr_frame >= self.nb_frames {
                    return Err(self.fail());
                }
                // Skip the extensions of frames from frame_max on.
                if self.curr_frame >= self.frame_max {
                    self.curr_len = 0;
                }
                self.repeat_data = self.curr;
                self.last_long = None;
                self.trailing_short_len = 0;
            } else if id == 2 {
                self.repeat_l = l;
                self.repeat_frame = self.curr_frame + 1;
                self.repeat_len = curr0 - self.repeat_data;
                self.src = self.repeat_data;
                self.src_len = self.repeat_len;
                if let Some(ext) = self.next_repeat()? {
                    return Ok(Some(ext));
                }
            } else if id > 2 {
                if id >= 32 {
                    // Where the last long extension ends, for the L flag of
                    // a repeat with L = 0.
                    self.last_long = Some(self.curr);
                    self.trailing_short_len = 0;
                } else {
                    self.trailing_short_len += l as usize;
                }
                return Ok(Some(Extension {
                    id,
                    frame: self.curr_frame,
                    data: &self.data[curr0 + header_size..self.curr],
                }));
            }
        }
        Ok(None)
    }

    /// The next extension with ID `id`, `None` when no other extension has
    /// it, or the error that ends the iterator. libopus
    /// `opus_extension_iterator_find`.
    ///
    /// The name is not `find`, so that [`Iterator::find`] stays callable.
    pub fn find_id(&mut self, id: u8) -> Result<Option<Extension<'a>>, Error> {
        for ext in self.by_ref() {
            let ext = ext?;
            if ext.id == id {
                return Ok(Some(ext));
            }
        }
        Ok(None)
    }
}

impl<'a> Iterator for ExtensionIterator<'a> {
    type Item = Result<Extension<'a>, Error>;

    /// The next extension, `None` at the end, or an error. libopus
    /// `opus_extension_iterator_next`. After an error, `None`.
    fn next(&mut self) -> Option<Self::Item> {
        if self.failed {
            return None;
        }
        if self.nb_frames > MAX_FRAMES {
            self.failed = true;
            return Some(Err(TOO_MANY_FRAMES));
        }
        self.next_extension().transpose()
    }
}

/// The number of extensions in the padding `data` of a packet of
/// `nb_frames` frames, repeated ones included. libopus
/// `opus_packet_extensions_count`.
///
/// As in libopus, invalid padding is not an error: the count stops at the
/// first entry that cannot be read. The only error is
/// [`BadArg`](crate::ErrorKind::BadArg) when `nb_frames` is above 48.
pub fn count(data: &[u8], nb_frames: usize) -> Result<usize, Error> {
    if nb_frames > MAX_FRAMES {
        return Err(TOO_MANY_FRAMES);
    }
    Ok(ExtensionIterator::new(data, nb_frames).map_while(Result::ok).count())
}

/// Like [`count`], but an error when the padding is not valid.
pub(crate) fn count_checked(data: &[u8], nb_frames: usize) -> Result<usize, Error> {
    let mut n = 0;
    for ext in ExtensionIterator::new(data, nb_frames) {
        ext?;
        n += 1;
    }
    Ok(n)
}

/// Counts the extensions of each frame into `nb_frame_exts[..nb_frames]`
/// and returns their total. libopus `opus_packet_extensions_count_ext`.
/// Like [`count`], it stops at invalid padding.
///
/// Errors: [`BadArg`](crate::ErrorKind::BadArg) when `nb_frames` is above 48
/// or `nb_frame_exts` holds fewer than `nb_frames` values. `nb_frame_exts`
/// then stays as it was.
pub fn count_ext(data: &[u8], nb_frame_exts: &mut [usize], nb_frames: usize) -> Result<usize, Error> {
    if nb_frames > MAX_FRAMES {
        return Err(TOO_MANY_FRAMES);
    }
    let counts = nb_frame_exts.get_mut(..nb_frames).ok_or(COUNTS_TOO_SHORT)?;
    counts.fill(0);
    let mut n = 0;
    for ext in ExtensionIterator::new(data, nb_frames).map_while(Result::ok) {
        counts[ext.frame] += 1;
        n += 1;
    }
    Ok(n)
}

/// Reads the extensions of the padding `data` of a packet of `nb_frames`
/// frames into `out`, in the order of the bytes, and returns their number.
/// libopus `opus_packet_extensions_parse`.
///
/// Errors: [`InvalidPacket`](crate::ErrorKind::InvalidPacket) when the padding
/// is not valid, [`BufferTooSmall`](crate::ErrorKind::BufferTooSmall) when
/// `out` is too short. In both cases `out` holds the extensions read before.
/// [`BadArg`](crate::ErrorKind::BadArg) when `nb_frames` is above 48.
pub fn parse<'a>(data: &'a [u8], out: &mut [Extension<'a>], nb_frames: usize) -> Result<usize, Error> {
    if nb_frames > MAX_FRAMES {
        return Err(TOO_MANY_FRAMES);
    }
    let mut n = 0;
    for ext in ExtensionIterator::new(data, nb_frames) {
        let ext = ext?;
        let slot = out.get_mut(n).ok_or(Error::BUFFER_TOO_SMALL)?;
        *slot = ext;
        n += 1;
    }
    Ok(n)
}

/// Reads the extensions like [`parse`], but in the order of the frames:
/// those of frame 0 first, each frame in the order of the bytes.
/// `nb_frame_exts` must hold what [`count_ext`] counted for the same
/// padding. libopus `opus_packet_extensions_parse_ext`.
///
/// When `nb_frame_exts` does not hold the counts of `data`, the order of `out`
/// is unspecified, and the result may be
/// [`BufferTooSmall`](crate::ErrorKind::BufferTooSmall).
///
/// Errors: those of [`parse`], and [`BadArg`](crate::ErrorKind::BadArg) when
/// `nb_frames` is above 48 or `nb_frame_exts` holds fewer than `nb_frames`
/// values.
pub fn parse_ext<'a>(
    data: &'a [u8],
    out: &mut [Extension<'a>],
    nb_frame_exts: &[usize],
    nb_frames: usize,
) -> Result<usize, Error> {
    if nb_frames > MAX_FRAMES {
        return Err(TOO_MANY_FRAMES);
    }
    let counts = nb_frame_exts.get(..nb_frames).ok_or(COUNTS_TOO_SHORT)?;
    // Where the extensions of each frame start. Counts that are not those
    // of `data` may be huge: the sums saturate, and an index past `out` is
    // BufferTooSmall.
    let mut next = [0usize; MAX_FRAMES];
    let mut total = 0usize;
    for (start, &count) in next.iter_mut().zip(counts) {
        *start = total;
        total = total.saturating_add(count);
    }
    let mut n = 0;
    for ext in ExtensionIterator::new(data, nb_frames) {
        let ext = ext?;
        let idx = next[ext.frame];
        next[ext.frame] = idx.saturating_add(1);
        let slot = out.get_mut(idx).ok_or(Error::BUFFER_TOO_SMALL)?;
        *slot = ext;
        n += 1;
    }
    Ok(n)
}

/// Where [`generate`] writes: a buffer, or nowhere when it only measures.
struct Out<'o> {
    data: Option<&'o mut [u8]>,
    len: usize,
}

impl Out<'_> {
    fn put(&mut self, pos: usize, byte: u8) {
        if let Some(d) = self.data.as_deref_mut() {
            d[pos] = byte;
        }
    }

    fn put_all(&mut self, pos: usize, bytes: &[u8]) {
        if let Some(d) = self.data.as_deref_mut() {
            d[pos..pos + bytes.len()].copy_from_slice(bytes);
        }
    }
}

/// Writes the payload of `ext` at `pos`, with its length unless `last`.
/// libopus `write_extension_payload`.
fn write_extension_payload(out: &mut Out, pos: usize, ext: &Extension, last: bool) -> Result<usize, Error> {
    let mut pos = pos;
    let n = ext.data.len();
    if ext.id < 32 {
        if n > 1 {
            return Err(Error::BAD_ARG);
        }
        if n > 0 {
            if out.len - pos < n {
                return Err(Error::BUFFER_TOO_SMALL);
            }
            out.put(pos, ext.data[0]);
            pos += 1;
        }
    } else {
        let length_bytes = if last { 0 } else { 1 + n / 255 };
        if out.len - pos < length_bytes + n {
            return Err(Error::BUFFER_TOO_SMALL);
        }
        if !last {
            for _ in 0..n / 255 {
                out.put(pos, 255);
                pos += 1;
            }
            out.put(pos, (n % 255) as u8);
            pos += 1;
        }
        out.put_all(pos, ext.data);
        pos += n;
    }
    Ok(pos)
}

/// Writes `ext`, its ID byte first. libopus `write_extension`.
fn write_extension(out: &mut Out, pos: usize, ext: &Extension, last: bool) -> Result<usize, Error> {
    if out.len - pos < 1 {
        return Err(Error::BUFFER_TOO_SMALL);
    }
    // For a short extension the L flag is its length; libopus writes this
    // byte before it checks the length.
    let l = if ext.id < 32 { ext.data.len() as u8 } else { u8::from(!last) };
    out.put(pos, (ext.id << 1).wrapping_add(l));
    write_extension_payload(out, pos + 1, ext, last)
}

/// libopus `opus_packet_extensions_generate`, writing into `out.data` when
/// it is there.
fn generate_impl(
    mut out: Out,
    extensions: &[Extension],
    nb_frames: usize,
    pad: bool,
) -> Result<usize, Error> {
    if nb_frames > MAX_FRAMES {
        return Err(TOO_MANY_FRAMES);
    }
    let n = extensions.len();
    let len = out.len;
    // The first and one past the last index of the extensions of each frame,
    // so that the work is linear in the number of extensions when they come
    // in frame order (which is not required).
    let mut frame_min_idx = [n; MAX_FRAMES];
    let mut frame_max_idx = [0usize; MAX_FRAMES];
    for (i, ext) in extensions.iter().enumerate() {
        let f = ext.frame;
        if f >= nb_frames || ext.id < MIN_ID || ext.id > MAX_ID {
            return Err(Error::BAD_ARG);
        }
        frame_min_idx[f] = frame_min_idx[f].min(i);
        frame_max_idx[f] = frame_max_idx[f].max(i + 1);
    }
    let mut frame_repeat_idx = frame_min_idx;
    let mut curr_frame = 0usize;
    let mut pos = 0usize;
    let mut written = 0usize;
    for f in 0..nb_frames {
        let mut last_long_idx: Option<usize> = None;
        let mut repeat_count = 0usize;
        if f + 1 < nb_frames {
            for i in frame_min_idx[f]..frame_max_idx[f] {
                if extensions[i].frame != f {
                    continue;
                }
                // Can this extension be repeated in every later frame?
                let mut g = f + 1;
                while g < nb_frames {
                    if frame_repeat_idx[g] >= frame_max_idx[g] {
                        break;
                    }
                    let other = &extensions[frame_repeat_idx[g]];
                    if other.id != extensions[i].id {
                        break;
                    }
                    if other.id < 32 && other.data.len() != extensions[i].data.len() {
                        break;
                    }
                    g += 1;
                }
                if g < nb_frames {
                    break;
                }
                // It can. For a long extension, keep the index of its last
                // copy, whose L flag may change.
                if extensions[i].id >= 32 {
                    last_long_idx = Some(frame_repeat_idx[nb_frames - 1]);
                }
                // libopus always repeats when it can, although in a rare
                // case with long payloads this is not the shortest coding.
                for g in f + 1..nb_frames {
                    let mut j = frame_repeat_idx[g] + 1;
                    while j < frame_max_idx[g] && extensions[j].frame != g {
                        j += 1;
                    }
                    frame_repeat_idx[g] = j;
                }
                repeat_count += 1;
                // The repeat goes after this extension.
                frame_repeat_idx[f] = i;
            }
        }
        let mut i = frame_min_idx[f];
        while i < frame_max_idx[f] {
            if extensions[i].frame == f {
                // A separator when the frame changes.
                if f != curr_frame {
                    let diff = f - curr_frame;
                    if len - pos < 2 {
                        return Err(Error::BUFFER_TOO_SMALL);
                    }
                    if diff == 1 {
                        out.put(pos, 0x02);
                        pos += 1;
                    } else {
                        out.put(pos, 0x03);
                        out.put(pos + 1, diff as u8);
                        pos += 2;
                    }
                    curr_frame = f;
                }
                pos = write_extension(&mut out, pos, &extensions[i], written + 1 == n)?;
                written += 1;
                if repeat_count > 0 && frame_repeat_idx[f] == i {
                    // The repeat, then the payloads of the repeated
                    // extensions of each later frame.
                    let nb_repeated = repeat_count * (nb_frames - (f + 1));
                    let last = written + nb_repeated == n || (last_long_idx.is_none() && i + 1 >= frame_max_idx[f]);
                    if len - pos < 1 {
                        return Err(Error::BUFFER_TOO_SMALL);
                    }
                    out.put(pos, 0x04 + u8::from(!last));
                    pos += 1;
                    for g in f + 1..nb_frames {
                        let mut j = frame_min_idx[g];
                        while j < frame_repeat_idx[g] {
                            if extensions[j].frame == g {
                                pos = write_extension_payload(&mut out, pos, &extensions[j], last && Some(j) == last_long_idx)?;
                                written += 1;
                            }
                            j += 1;
                        }
                        frame_min_idx[g] = j;
                    }
                    if last {
                        curr_frame += 1;
                    }
                }
            }
            i += 1;
        }
    }
    debug_assert_eq!(written, n);
    // Padding goes in front, as bytes 0x01.
    if pad && pos < len {
        let padding = len - pos;
        if let Some(d) = out.data.as_deref_mut() {
            d.copy_within(0..pos, padding);
            d[..padding].fill(0x01);
        }
        pos += padding;
    }
    Ok(pos)
}

/// Writes `extensions` for a packet of `nb_frames` frames into `out` and
/// returns the bytes written. libopus `opus_packet_extensions_generate`.
///
/// The extensions may come in any order; within a frame they keep their
/// order. Where the same extensions follow in every later frame, a repeat
/// codes them. With [`Padding::FillBuffer`], bytes 0x01 in front fill `out`
/// to its length.
///
/// Errors: [`BadArg`](crate::ErrorKind::BadArg) for more than 48 frames, a
/// frame number of `nb_frames` or more, an ID outside 3 to 127, or a short
/// extension (ID below 32) with more than one byte;
/// [`BufferTooSmall`](crate::ErrorKind::BufferTooSmall) when `out` is too
/// short. `out` may then hold part of the result.
pub fn generate(out: &mut [u8], extensions: &[Extension], nb_frames: usize, padding: Padding) -> Result<usize, Error> {
    let len = out.len();
    generate_impl(Out { data: Some(out), len }, extensions, nb_frames, padding.fills_buffer())
}

/// The bytes [`generate`] would write into a buffer of `max_len` bytes
/// without padding, with its errors. libopus
/// `opus_packet_extensions_generate` with a null `data` pointer.
pub fn generated_len(max_len: usize, extensions: &[Extension], nb_frames: usize) -> Result<usize, Error> {
    generate_impl(Out { data: None, len: max_len }, extensions, nb_frames, false)
}

/// The extensions the repacketizer collects, on the stack while they are
/// few: libopus's `all_extensions` of `opus_repacketizer_out_range_impl`.
const INLINE_EXTENSIONS: usize = 32;

/// A list of extensions with room for a known number. Up to 32 live inside the
/// value. More need the feature `alloc`, which puts them on the heap; without it,
/// [`ExtensionList::with_capacity`] fails with
/// [`ErrorKind::AllocFail`](crate::ErrorKind::AllocFail).
pub(crate) struct ExtensionList<'a> {
    inline: [Extension<'a>; INLINE_EXTENSIONS],
    len: usize,
    #[cfg(feature = "alloc")]
    heap: alloc::vec::Vec<Extension<'a>>,
}

impl<'a> ExtensionList<'a> {
    pub(crate) fn with_capacity(capacity: usize) -> Result<Self, Error> {
        #[cfg(not(feature = "alloc"))]
        if capacity > INLINE_EXTENSIONS {
            return Err(Error::new(
                crate::ErrorKind::AllocFail,
                "more than 32 extensions need the alloc feature",
            ));
        }
        Ok(ExtensionList {
            inline: [Extension::default(); INLINE_EXTENSIONS],
            len: 0,
            #[cfg(feature = "alloc")]
            heap: if capacity > INLINE_EXTENSIONS { alloc::vec::Vec::with_capacity(capacity) } else { alloc::vec::Vec::new() },
        })
    }

    pub(crate) fn push(&mut self, ext: Extension<'a>) {
        #[cfg(feature = "alloc")]
        if self.heap.capacity() > 0 {
            self.heap.push(ext);
            return;
        }
        self.inline[self.len] = ext;
        self.len += 1;
    }

    pub(crate) fn as_slice(&self) -> &[Extension<'a>] {
        #[cfg(feature = "alloc")]
        if self.heap.capacity() > 0 {
            return &self.heap;
        }
        &self.inline[..self.len]
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;
    use crate::ErrorKind;

    /// ID, frame and payload of each extension.
    type Plain = (u8, usize, Vec<u8>);

    fn all(data: &[u8], nb_frames: usize) -> Result<Vec<Plain>, Error> {
        ExtensionIterator::new(data, nb_frames).map(|e| e.map(|e| (e.id, e.frame, e.data.to_vec()))).collect()
    }

    #[test]
    fn padding_and_short_and_long_extensions() {
        // 0x01 padding, ID 3 with one byte, ID 32 with a length, ID 33 to
        // the end.
        let data = [0x01, 0x07, 0xAA, 0x41, 2, 1, 2, 0x42, 9, 9, 9];
        let v = all(&data, 1).unwrap();
        assert_eq!(v, [(3, 0, vec![0xAA]), (32, 0, vec![1, 2]), (33, 0, vec![9, 9, 9])]);
        assert_eq!(count(&data, 1), Ok(3));
        // Zeros are padding to the end.
        assert_eq!(all(&[0, 0, 0], 1).unwrap(), []);
        // A length past the end.
        assert_eq!(all(&[0x41, 5, 1], 1), Err(Error::INVALID_PACKET));
    }

    #[test]
    fn separators_and_their_limits() {
        // ID 3 in frame 0, separator, ID 4 in frame 1, separator by 2 to
        // frame 3, ID 5.
        let data = [0x06, 0x02, 0x08, 0x03, 2, 0x0A];
        let v = all(&data, 4).unwrap();
        assert_eq!(v, [(3, 0, vec![]), (4, 1, vec![]), (5, 3, vec![])]);
        // A separator past the last frame.
        assert_eq!(all(&data, 3), Err(Error::INVALID_PACKET));
        let mut it = ExtensionIterator::new(&data, 4);
        it.set_frame_max(2);
        assert_eq!(it.next().map(|e| e.unwrap().frame), Some(0));
        assert_eq!(it.next().map(|e| e.unwrap().frame), Some(1));
        assert_eq!(it.next(), None);
    }

    #[test]
    fn the_iterator_ends_after_an_error_until_reset() {
        // ID 3, then a separator past the last frame of a 1-frame packet.
        let data = [0x06, 0x02, 0x08];
        let mut it = ExtensionIterator::new(&data, 1);
        assert_eq!(it.next(), Some(Ok(Extension::new(3, 0, &[]))));
        assert_eq!(it.next().map(|e| e.map_err(|e| e.kind())), Some(Err(ErrorKind::InvalidPacket)));
        assert_eq!(it.next(), None);
        assert_eq!(it.next(), None);
        it.reset();
        assert_eq!(it.next(), Some(Ok(Extension::new(3, 0, &[]))));
        // find_id reports the error once too.
        it.reset();
        assert_eq!(it.find_id(4).map_err(|e| e.kind()), Err(ErrorKind::InvalidPacket));
        assert_eq!(it.find_id(4), Ok(None));
        // With enough frames there is no error, and the iterator adapters
        // work.
        let mut it = ExtensionIterator::new(&data, 2);
        assert_eq!(it.find_id(4), Ok(Some(Extension::new(4, 1, &[]))));
        assert_eq!(it.next(), None);
        let ids: Result<Vec<u8>, Error> = ExtensionIterator::new(&data, 2).map(|e| e.map(|e| e.id)).collect();
        assert_eq!(ids, Ok(vec![3, 4]));
        assert_eq!(ExtensionIterator::new(&data, 2).find(|e| e.as_ref().is_ok_and(|e| e.frame == 1)), Some(Ok(Extension::new(4, 1, &[]))));
    }

    #[test]
    fn more_than_48_frames_is_an_error_not_a_panic() {
        let data = [0x07, 0xAA];
        // The iterator: BadArg once, then the end.
        let mut it = ExtensionIterator::new(&data, 49);
        assert_eq!(it.next().map(|e| e.map_err(|e| e.kind())), Some(Err(ErrorKind::BadArg)));
        assert_eq!(it.next(), None);
        assert_eq!(ExtensionIterator::new(&data, usize::MAX).find_id(3).map_err(|e| e.kind()), Err(ErrorKind::BadArg));
        // 48 frames are fine.
        assert_eq!(ExtensionIterator::new(&data, 48).next(), Some(Ok(Extension::new(3, 0, &[0xAA]))));
        assert_eq!(count(&data, 48), Ok(1));
        // The functions.
        let kind = |r: Result<usize, Error>| r.map_err(|e| e.kind());
        assert_eq!(kind(count(&data, 49)), Err(ErrorKind::BadArg));
        let mut counts = [7usize; 64];
        assert_eq!(kind(count_ext(&data, &mut counts, 49)), Err(ErrorKind::BadArg));
        assert_eq!(counts, [7; 64], "counts change on an error");
        let mut out = [Extension::default(); 4];
        assert_eq!(kind(parse(&data, &mut out, 49)), Err(ErrorKind::BadArg));
        assert_eq!(kind(parse_ext(&data, &mut out, &counts, 49)), Err(ErrorKind::BadArg));
        assert_eq!(kind(generate(&mut [0u8; 8], &[], 49, Padding::Shortest)), Err(ErrorKind::BadArg));
        assert_eq!(kind(generated_len(8, &[], 49)), Err(ErrorKind::BadArg));
    }

    #[test]
    fn too_few_frame_counts_is_an_error_not_a_panic() {
        // ID 3 in frame 0, a separator, ID 4 in frame 1.
        let data = [0x06, 0x02, 0x08];
        let mut counts = [9usize; 1];
        assert_eq!(count_ext(&data, &mut counts, 2).map_err(|e| e.kind()), Err(ErrorKind::BadArg));
        assert_eq!(counts, [9], "counts change on an error");
        let mut out = [Extension::default(); 4];
        assert_eq!(parse_ext(&data, &mut out, &[1], 2).map_err(|e| e.kind()), Err(ErrorKind::BadArg));
        assert_eq!(parse_ext(&data, &mut out, &[], 1).map_err(|e| e.kind()), Err(ErrorKind::BadArg));
        // Enough counts; the ones past nb_frames are not touched.
        let mut counts = [9usize; 3];
        assert_eq!(count_ext(&data, &mut counts, 2), Ok(2));
        assert_eq!(counts, [1, 1, 9]);
        assert_eq!(parse_ext(&data, &mut out, &counts, 2), Ok(2));
        assert_eq!(out[..2], [Extension::new(3, 0, &[]), Extension::new(4, 1, &[])]);
        // Counts that are not those of the padding, even huge ones, give
        // BufferTooSmall or some order, never a panic.
        for bad in [[usize::MAX, usize::MAX], [usize::MAX, 0], [0, 0], [0, usize::MAX], [3, 3]] {
            let r = parse_ext(&data, &mut out, &bad, 2);
            assert!(matches!(r.map_err(|e| e.kind()), Ok(2) | Err(ErrorKind::BufferTooSmall)), "{bad:?}: {r:?}");
        }
    }

    #[test]
    fn extension_new_sets_the_fields() {
        let payload = [1u8, 2];
        const EMPTY: Extension<'static> = Extension::new(5, 2, &[]);
        assert_eq!((EMPTY.id, EMPTY.frame, EMPTY.data), (5, 2, &[][..]));
        let ext = Extension::new(40, 1, &payload);
        assert_eq!((ext.id, ext.frame, ext.data), (40, 1, &payload[..]));
        assert_eq!(ext, Extension { id: 40, frame: 1, data: &payload });
    }

    #[test]
    fn generate_uses_repeats_and_parses_back() {
        let payload = [1u8, 2, 3];
        let mut exts = Vec::new();
        for f in 0..4 {
            exts.push(Extension::new(3, f, &[]));
            exts.push(Extension::new(40, f, &payload));
        }
        let mut buf = [0u8; 64];
        let n = generate(&mut buf, &exts, 4, Padding::Shortest).unwrap();
        assert_eq!(generated_len(64, &exts, 4), Ok(n));
        // Two ID bytes, the length, the payload, the repeat and three
        // payloads, the last without a length.
        assert_eq!(n, 1 + 1 + 1 + 3 + 1 + 4 + 4 + 3);
        let mut out = [Extension::default(); 8];
        let mut counts = [0usize; 4];
        assert_eq!(count_ext(&buf[..n], &mut counts, 4), Ok(8));
        assert_eq!(counts, [2, 2, 2, 2]);
        assert_eq!(parse_ext(&buf[..n], &mut out, &counts, 4), Ok(8));
        assert_eq!(&out[..], &exts[..]);
        // Padded to the end with 0x01 in front.
        let n2 = generate(&mut buf, &exts, 4, Padding::FillBuffer).unwrap();
        assert_eq!(n2, 64);
        assert_eq!(buf[0], 0x01);
        assert_eq!(count(&buf, 4), Ok(8));
        // Bad arguments.
        let bad = [Extension::new(2, 0, &[])];
        assert_eq!(generate(&mut buf, &bad, 1, Padding::Shortest), Err(Error::BAD_ARG));
        let bad = [Extension::new(3, 1, &[])];
        assert_eq!(generate(&mut buf, &bad, 1, Padding::Shortest), Err(Error::BAD_ARG));
        let bad = [Extension::new(3, 0, &[1, 2])];
        assert_eq!(generate(&mut buf, &bad, 1, Padding::Shortest), Err(Error::BAD_ARG));
        assert_eq!(generate(&mut buf[..5], &exts, 4, Padding::Shortest), Err(Error::BUFFER_TOO_SMALL));
    }
}
