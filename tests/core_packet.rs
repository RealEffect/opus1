//! `opus1::packet` against the packet functions of C libopus.
//!
//! The reference is the public C API only:
//!
//! - `opus_packet_parse` for normal framing,
//! - the repacketizer and `opus_packet_pad` for writing and padding,
//! - `opus_multistream_packet_pad` and `opus_multistream_packet_unpad` for
//!   self-delimited framing (they rewrite every stream of a packet),
//! - `opus_multistream_decode_float`, which validates the framing of every
//!   stream before it decodes anything.

use audiopus_sys as sys;
use opus1::{SampleRate, packet};

/// Frame sizes around the points where the size field changes: one byte up
/// to 251, two bytes from 252, and the largest frame, 1275 bytes.
const SIZES: [usize; 6] = [0, 1, 251, 252, 253, 1275];

/// TOC byte of a CELT-only fullband 2.5 ms mono packet: up to 48 frames fit
/// in one packet.
const TOC_2_5_MS: u8 = 0xE0;
/// TOC byte of a SILK-only wideband 20 ms stereo packet: up to 6 frames.
const TOC_20_MS: u8 = 0x4C;

/// A small deterministic random number generator.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    fn below(&mut self, n: usize) -> usize {
        self.next() as usize % n.max(1)
    }
    fn bytes(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| self.next() as u8).collect()
    }
}

// ---------------------------------------------------------------------------
// C libopus wrappers
// ---------------------------------------------------------------------------

/// `opus_packet_parse`: TOC byte, frame sizes and payload offset, or the
/// error code.
fn c_parse(data: &[u8]) -> Result<(u8, Vec<usize>, usize), i32> {
    let mut toc = 0u8;
    let mut sizes = [0i16; 48];
    let mut payload_offset = 0i32;
    let count = unsafe {
        sys::opus_packet_parse(
            data.as_ptr(),
            data.len() as i32,
            &mut toc,
            std::ptr::null_mut(),
            sizes.as_mut_ptr(),
            &mut payload_offset,
        )
    };
    if count < 0 {
        return Err(count);
    }
    let sizes = sizes[..count as usize].iter().map(|&s| s as usize).collect();
    Ok((toc, sizes, payload_offset as usize))
}

/// `opus_packet_pad` on a copy of `data`.
fn c_pad(data: &[u8], new_len: usize) -> Result<Vec<u8>, i32> {
    let mut out = data.to_vec();
    out.resize(new_len.max(data.len()), 0xEE);
    let ret = unsafe { sys::opus_packet_pad(out.as_mut_ptr(), data.len() as i32, new_len as i32) };
    if ret != 0 {
        return Err(ret);
    }
    out.truncate(new_len);
    Ok(out)
}

/// `opus_multistream_packet_pad` on a copy of `data`.
fn c_multistream_pad(data: &[u8], new_len: usize, streams: usize) -> Result<Vec<u8>, i32> {
    let mut out = data.to_vec();
    out.resize(new_len.max(data.len()), 0xEE);
    let ret = unsafe {
        sys::opus_multistream_packet_pad(
            out.as_mut_ptr(),
            data.len() as i32,
            new_len as i32,
            streams as i32,
        )
    };
    if ret != 0 {
        return Err(ret);
    }
    out.truncate(new_len);
    Ok(out)
}

/// `opus_multistream_packet_unpad` on a copy of `data`: every stream is
/// written again by the C repacketizer, without padding.
fn c_multistream_unpad(data: &[u8], streams: usize) -> Result<Vec<u8>, i32> {
    let mut out = data.to_vec();
    let ret = unsafe {
        sys::opus_multistream_packet_unpad(out.as_mut_ptr(), out.len() as i32, streams as i32)
    };
    if ret < 0 {
        return Err(ret);
    }
    out.truncate(ret as usize);
    Ok(out)
}

/// The packet the C repacketizer makes of `frames`.
fn c_repacketize(toc: u8, frames: &[&[u8]]) -> Result<Vec<u8>, i32> {
    // The repacketizer keeps pointers into the packets it is given.
    let singles: Vec<Vec<u8>> = frames
        .iter()
        .map(|f| {
            let mut p = vec![toc & 0xFC];
            p.extend_from_slice(f);
            p
        })
        .collect();
    unsafe {
        let rp = sys::opus_repacketizer_create();
        let mut result = Ok(Vec::new());
        for p in &singles {
            let ret = sys::opus_repacketizer_cat(rp, p.as_ptr(), p.len() as i32);
            if ret != 0 {
                result = Err(ret);
                break;
            }
        }
        if result.is_ok() {
            let mut out = vec![0u8; 48 * 1277 + 16];
            let n = sys::opus_repacketizer_out(rp, out.as_mut_ptr(), out.len() as i32);
            if n < 0 {
                result = Err(n);
            } else {
                out.truncate(n as usize);
                result = Ok(out);
            }
        }
        sys::opus_repacketizer_destroy(rp);
        result
    }
}

/// C multistream decoder of `streams` mono streams at 48 kHz.
struct CMsDecoder {
    st: *mut sys::OpusMSDecoder,
    pcm: Vec<f32>,
}

impl CMsDecoder {
    fn new(streams: usize) -> Self {
        let mapping: Vec<u8> = (0..streams as u8).collect();
        let mut err = 0;
        let st = unsafe {
            sys::opus_multistream_decoder_create(
                48000,
                streams as i32,
                streams as i32,
                0,
                mapping.as_ptr(),
                &mut err,
            )
        };
        assert_eq!(err, 0, "opus_multistream_decoder_create");
        Self { st, pcm: vec![0f32; 5760 * streams] }
    }

    /// Samples per channel, or the error code.
    fn decode(&mut self, data: &[u8]) -> Result<usize, i32> {
        let n = unsafe {
            sys::opus_multistream_decode_float(
                self.st,
                data.as_ptr(),
                data.len() as i32,
                self.pcm.as_mut_ptr(),
                5760,
                0,
            )
        };
        if n < 0 { Err(n) } else { Ok(n as usize) }
    }
}

impl Drop for CMsDecoder {
    fn drop(&mut self) {
        unsafe { sys::opus_multistream_decoder_destroy(self.st) }
    }
}

// ---------------------------------------------------------------------------
// Test packets
// ---------------------------------------------------------------------------

/// Frame layouts that reach every frame-count code: one frame, two equal,
/// two different, several equal, several different, and the largest count.
fn frame_layouts() -> Vec<(u8, Vec<usize>)> {
    let mut layouts = Vec::new();
    for &a in &SIZES {
        layouts.push((TOC_20_MS, vec![a]));
        layouts.push((TOC_20_MS, vec![a, a]));
        layouts.push((TOC_20_MS, vec![a; 3]));
        layouts.push((TOC_20_MS, vec![a; 6]));
        layouts.push((TOC_2_5_MS, vec![a; 48]));
        for &b in &SIZES {
            if a != b {
                layouts.push((TOC_20_MS, vec![a, b]));
                layouts.push((TOC_20_MS, vec![a, b, a]));
                layouts.push((TOC_20_MS, vec![b, a, a, a, b, a]));
            }
        }
    }
    // Every size once, and 48 frames that cycle through the sizes.
    layouts.push((TOC_20_MS, SIZES.to_vec()));
    layouts.push((TOC_2_5_MS, (0..48).map(|i| SIZES[i % SIZES.len()]).collect()));
    layouts
}

fn make_frames(rng: &mut Rng, sizes: &[usize]) -> Vec<Vec<u8>> {
    sizes.iter().map(|&n| rng.bytes(n)).collect()
}

fn as_slices(frames: &[Vec<u8>]) -> Vec<&[u8]> {
    frames.iter().map(|f| f.as_slice()).collect()
}

/// `packet::write` into a buffer of `room` bytes.
fn rs_write(toc: u8, frames: &[Vec<u8>], room: usize, self_delimited: bool, pad: bool) -> Vec<u8> {
    let mut out = vec![0xEEu8; room];
    let n = packet::write(toc, &as_slices(frames), &mut out, self_delimited, pad).expect("write");
    out.truncate(n);
    out
}

/// Check that `data` parses back into `frames`.
fn assert_parses_to(data: &[u8], self_delimited: bool, toc: u8, frames: &[Vec<u8>], what: &str) {
    let parsed = packet::parse(data, self_delimited).unwrap_or_else(|e| panic!("{what}: {e}"));
    assert_eq!(parsed.toc & 0xFC, toc & 0xFC, "{what}: TOC");
    assert_eq!(parsed.count, frames.len(), "{what}: frame count");
    for (i, (got, want)) in parsed.frames(data).zip(frames).enumerate() {
        assert_eq!(got, want.as_slice(), "{what}: frame {i}");
    }
    assert_eq!(parsed.frames(data).count(), frames.len(), "{what}: frames iterator");
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// `write` then `parse` returns the same frames, for every frame-count code,
/// with and without self-delimited framing and padding. Without padding the
/// bytes are those of the C repacketizer.
#[test]
fn write_then_parse_round_trips() {
    let mut rng = Rng(1);
    let layouts = frame_layouts();
    let mut codes = [0usize; 4];
    for (toc, sizes) in &layouts {
        let frames = make_frames(&mut rng, sizes);
        let what = format!("TOC {toc:#04x}, sizes {sizes:?}");
        let room = sizes.iter().sum::<usize>() + 2 * sizes.len() + 4;

        // Normal framing.
        let plain = rs_write(*toc, &frames, room, false, false);
        codes[(plain[0] & 3) as usize] += 1;
        assert_parses_to(&plain, false, *toc, &frames, &what);
        assert_eq!(plain, c_repacketize(*toc, &as_slices(&frames)).unwrap(), "{what}: C repacketizer");
        let (c_toc, c_sizes, c_offset) = c_parse(&plain).unwrap();
        let parsed = packet::parse(&plain, false).unwrap();
        assert_eq!(c_toc, parsed.toc, "{what}");
        assert_eq!(c_sizes, *sizes, "{what}: C sizes");
        assert_eq!(c_offset, parsed.payload_offset, "{what}: payload offset");
        assert_eq!(parsed.packet_offset, plain.len(), "{what}: packet offset");
        assert_eq!(parsed.padding_len, 0, "{what}");

        // Self-delimited framing, followed by bytes of another stream.
        let mut delimited = rs_write(*toc, &frames, room, true, false);
        let own_len = delimited.len();
        delimited.extend_from_slice(&[0xAB; 7]);
        assert_parses_to(&delimited, true, *toc, &frames, &what);
        assert_eq!(packet::parse(&delimited, true).unwrap().packet_offset, own_len, "{what}");

        // Padding to several exact lengths, in both framings.
        for extra in [0usize, 1, 2, 3, 100, 254, 255, 256, 257, 511, 700] {
            for self_delimited in [false, true] {
                let base = if self_delimited { own_len } else { plain.len() };
                let padded = rs_write(*toc, &frames, base + extra, self_delimited, true);
                assert_eq!(padded.len(), base + extra, "{what}: padded length");
                assert_parses_to(&padded, self_delimited, *toc, &frames, &what);
                let parsed = packet::parse(&padded, self_delimited).unwrap();
                assert_eq!(parsed.packet_offset, padded.len(), "{what}: padded packet offset");
                let padding = &padded[padded.len() - parsed.padding_len..];
                assert!(padding.iter().all(|&b| b == 0), "{what}: padding is zeros");
            }
        }

        // A buffer one byte too small is an error, not a panic.
        let mut small = vec![0u8; plain.len() - 1];
        assert!(packet::write(*toc, &as_slices(&frames), &mut small, false, false).is_err(), "{what}");
    }
    assert!(codes.iter().all(|&n| n > 0), "every frame-count code was written: {codes:?}");
}

/// `pad` gives the bytes of `opus_packet_pad`, also for a packet that is
/// padded already.
#[test]
fn pad_gives_the_bytes_of_opus_packet_pad() {
    let mut rng = Rng(2);
    let mut compared = 0;
    for (toc, sizes) in &frame_layouts() {
        let frames = make_frames(&mut rng, sizes);
        let room = sizes.iter().sum::<usize>() + 2 * sizes.len() + 4;
        let plain = rs_write(*toc, &frames, room, false, false);
        for extra in [0usize, 1, 2, 3, 4, 100, 253, 254, 255, 256, 257, 258, 509, 510, 511, 512, 1000] {
            let new_len = plain.len() + extra;
            let what = format!("TOC {toc:#04x}, sizes {sizes:?}, +{extra}");
            let expected = c_pad(&plain, new_len).unwrap_or_else(|e| panic!("{what}: C error {e}"));
            let mut ours = plain.clone();
            ours.resize(new_len, 0xEE);
            packet::pad(&mut ours, plain.len(), new_len).unwrap_or_else(|e| panic!("{what}: {e}"));
            assert_eq!(ours, expected, "{what}");
            assert_parses_to(&ours, false, *toc, &frames, &what);

            // Pad the padded packet again.
            let again_len = new_len + 1 + extra / 3;
            let expected = c_pad(&expected, again_len).unwrap();
            ours.resize(again_len, 0xEE);
            packet::pad(&mut ours, new_len, again_len).unwrap();
            assert_eq!(ours, expected, "{what}: padded twice");
            compared += 2;
        }
    }
    assert!(compared > 1000);

    // Both reject the same arguments.
    let packet = [TOC_20_MS, 1, 2, 3];
    assert!(c_pad(&packet, 3).is_err());
    assert!(packet::pad(&mut packet.to_vec(), 4, 3).is_err());
    assert!(packet::pad(&mut [], 0, 4).is_err());
    // A code 3 packet without its frame-count byte is invalid.
    let broken = [TOC_20_MS | 3];
    assert!(c_pad(&broken, 8).is_err());
    let mut ours = broken.to_vec();
    ours.resize(8, 0);
    assert!(packet::pad(&mut ours, 1, 8).is_err());
}

/// A multistream packet of `streams` streams. Returns the packet and the
/// frames of every stream.
fn multistream_packet(rng: &mut Rng, streams: usize, layouts: &[(u8, Vec<usize>)]) -> (Vec<u8>, Vec<Vec<Vec<u8>>>) {
    // All streams must have the same duration: take layouts with the TOC and
    // frame count of the first one.
    let (toc, first) = &layouts[rng.below(layouts.len())];
    let same: Vec<&Vec<usize>> = layouts
        .iter()
        .filter(|(t, s)| t == toc && s.len() == first.len())
        .map(|(_, s)| s)
        .collect();
    let mut data = Vec::new();
    let mut all_frames = Vec::new();
    for s in 0..streams {
        let sizes = same[rng.below(same.len())];
        let frames = make_frames(rng, sizes);
        let room = sizes.iter().sum::<usize>() + 2 * sizes.len() + 4;
        data.extend(rs_write(*toc, &frames, room, s != streams - 1, false));
        all_frames.push(frames);
    }
    (data, all_frames)
}

/// `multistream_pad` gives the bytes of `opus_multistream_packet_pad`, and
/// the self-delimited packets `write` makes are what the C repacketizer
/// writes (`opus_multistream_packet_unpad` writes every stream again).
#[test]
fn multistream_pad_gives_the_bytes_of_the_c_function() {
    let mut rng = Rng(3);
    let layouts = frame_layouts();
    for round in 0..400 {
        let streams = 1 + round % 4;
        let (data, frames) = multistream_packet(&mut rng, streams, &layouts);
        let what = format!("round {round}, {streams} streams");

        // The unpadded packet is already in the form C writes.
        assert_eq!(c_multistream_unpad(&data, streams).unwrap(), data, "{what}: C rewrite differs");

        // Every stream parses back, the last one with normal framing.
        let mut rest = &data[..];
        for (s, stream_frames) in frames.iter().enumerate() {
            let self_delimited = s != streams - 1;
            let toc = rest[0];
            assert_parses_to(rest, self_delimited, toc, stream_frames, &what);
            rest = &rest[packet::parse(rest, self_delimited).unwrap().packet_offset..];
        }
        assert!(rest.is_empty(), "{what}");

        let extra = [1usize, 2, 3, 17, 254, 255, 256, 600][round % 8];
        let new_len = data.len() + extra;
        let expected = c_multistream_pad(&data, new_len, streams).unwrap();
        let mut ours = data.clone();
        ours.resize(new_len, 0xEE);
        packet::multistream_pad(&mut ours, data.len(), new_len, streams).unwrap();
        assert_eq!(ours, expected, "{what}: +{extra}");
        // Removing the padding again gives the packet we started from.
        assert_eq!(c_multistream_unpad(&ours, streams).unwrap(), data, "{what}");
    }
    // A packet with fewer streams than asked for is invalid on both sides.
    let one = [TOC_20_MS, 9, 9];
    assert!(c_multistream_pad(&one, 20, 3).is_err());
    let mut ours = one.to_vec();
    ours.resize(20, 0);
    assert!(packet::multistream_pad(&mut ours, 3, 20, 3).is_err());
}

/// Change a packet at random: cut it short, flip bits, overwrite header
/// bytes, or append bytes.
fn mutate(rng: &mut Rng, data: &mut Vec<u8>) {
    for _ in 0..1 + rng.below(3) {
        match rng.below(6) {
            0 => {
                let n = rng.below(data.len() + 1);
                data.truncate(n);
            }
            1 if !data.is_empty() => {
                let i = rng.below(data.len());
                data[i] ^= 1 << rng.below(8);
            }
            2 if !data.is_empty() => {
                // The framing lives in the first bytes.
                let i = rng.below(data.len().min(8));
                data[i] = rng.next() as u8;
            }
            3 => {
                let n = rng.below(6);
                data.extend(rng.bytes(n));
            }
            4 if !data.is_empty() => {
                let i = rng.below(data.len());
                data[i] = [0, 1, 251, 252, 253, 254, 255][rng.below(7)];
            }
            _ => {
                let n = rng.below(40);
                *data = rng.bytes(n);
            }
        }
    }
}

/// Mutated single packets: opus1 and `opus_packet_parse` accept the same
/// ones and find the same frames. `pad` agrees with `opus_packet_pad` when
/// the packet holds no padding with content.
#[test]
fn mutated_packets_parse_like_libopus() {
    let mut rng = Rng(4);
    // Short layouts keep the header a large part of the packet.
    let layouts: Vec<(u8, Vec<usize>)> = frame_layouts()
        .into_iter()
        .filter(|(_, s)| s.iter().sum::<usize>() < 600)
        .collect();
    let (mut accepted, mut rejected) = (0, 0);
    for round in 0..6000 {
        let (toc, sizes) = &layouts[rng.below(layouts.len())];
        let frames = make_frames(&mut rng, sizes);
        let room = sizes.iter().sum::<usize>() + 2 * sizes.len() + 4 + rng.below(300);
        let mut data = rs_write(*toc, &frames, room, false, round % 3 == 0);
        mutate(&mut rng, &mut data);

        let ours = packet::parse(&data, false);
        if data.is_empty() {
            // C takes a null pointer for an empty packet; both reject it.
            assert!(ours.is_err());
            continue;
        }
        match c_parse(&data) {
            Err(code) => {
                assert!(ours.is_err(), "round {round}: C rejects ({code}), opus1 accepts {data:02x?}");
                rejected += 1;
            }
            Ok((c_toc, c_sizes, c_offset)) => {
                let p = ours.unwrap_or_else(|e| panic!("round {round}: C accepts, opus1: {e}: {data:02x?}"));
                let sizes: Vec<usize> = p.sizes[..p.count].iter().map(|&s| s as usize).collect();
                assert_eq!((p.toc, sizes, p.payload_offset), (c_toc, c_sizes, c_offset), "round {round}: {data:02x?}");
                assert!(p.packet_offset <= data.len() && p.padding_len <= p.packet_offset);
                assert_eq!(p.frames(&data).map(|f| f.len()).sum::<usize>() + p.payload_offset + p.padding_len, p.packet_offset);
                accepted += 1;
            }
        }

        // Padding the mutated packet never panics. libopus keeps "packet
        // extensions" it finds in existing padding, which this crate does
        // not have, so the bytes are compared only when the padding is empty
        // or all zeros.
        let new_len = data.len() + 1 + rng.below(300);
        let mut padded = data.clone();
        padded.resize(new_len, 0xEE);
        let result = packet::pad(&mut padded, data.len(), new_len);
        if let Ok(p) = ours {
            let padding = &data[p.packet_offset - p.padding_len..p.packet_offset];
            if padding.iter().all(|&b| b == 0) {
                assert_eq!(result.map(|()| padded).map_err(|_| ()), c_pad(&data, new_len).map_err(|_| ()), "round {round}: pad of {data:02x?}");
            }
        } else {
            assert!(result.is_err() && c_pad(&data, new_len).is_err(), "round {round}");
        }
    }
    println!("{accepted} accepted, {rejected} rejected");
    assert!(accepted > 1000 && rejected > 1000, "{accepted} accepted, {rejected} rejected");
}

/// Mutated multistream packets, also read with the wrong stream count:
/// `multistream_validate` accepts what the C multistream decoder accepts and
/// reports the same duration.
#[test]
fn mutated_multistream_packets_validate_like_libopus() {
    let mut rng = Rng(5);
    let layouts: Vec<(u8, Vec<usize>)> = frame_layouts()
        .into_iter()
        .filter(|(_, s)| s.iter().sum::<usize>() < 300)
        .collect();
    let mut decoders: Vec<CMsDecoder> = (1..=4).map(CMsDecoder::new).collect();
    let (mut accepted, mut rejected) = (0, 0);
    for round in 0..5000 {
        let streams = 1 + rng.below(4);
        let (mut data, _) = multistream_packet(&mut rng, streams, &layouts);
        // One round in three stays valid; the others are mutated.
        if round % 3 != 0 {
            mutate(&mut rng, &mut data);
        }
        // One round in five reads the packet with another stream count.
        let read_as = if round % 5 == 0 { 1 + rng.below(4) } else { streams };
        if data.is_empty() {
            // An empty packet means "lost" to the C decoder.
            assert!(packet::multistream_validate(&data, read_as, SampleRate::Hz48000).is_err());
            continue;
        }
        let ours = packet::multistream_validate(&data, read_as, SampleRate::Hz48000);
        match decoders[read_as - 1].decode(&data) {
            Ok(samples) => {
                assert_eq!(ours, Ok(samples), "round {round}, {read_as} streams: {data:02x?}");
                accepted += 1;
            }
            Err(code) => {
                assert!(ours.is_err(), "round {round}, {read_as} streams: C rejects ({code}), opus1 gives {ours:?}: {data:02x?}");
                rejected += 1;
            }
        }
        // Padding a damaged multistream packet never panics.
        let new_len = data.len() + 1 + rng.below(40);
        let mut padded = data.clone();
        padded.resize(new_len, 0);
        let _ = packet::multistream_pad(&mut padded, data.len(), new_len, read_as);
    }
    println!("{accepted} accepted, {rejected} rejected");
    assert!(accepted > 1000 && rejected > 1000, "{accepted} accepted, {rejected} rejected");
}

/// `samples_per_frame` is `opus_packet_get_samples_per_frame` for every TOC
/// byte and sampling rate.
#[test]
fn samples_per_frame_matches_libopus() {
    for &fs in SampleRate::ALL {
        for toc in 0..=255u8 {
            let expected = unsafe { sys::opus_packet_get_samples_per_frame(&toc, fs.hz() as i32) };
            assert_eq!(packet::samples_per_frame(toc, fs), expected as usize, "TOC {toc:#04x} at {fs:?}");
        }
    }
}

/// `nb_frames` and `nb_samples` are `opus_packet_get_nb_frames` and
/// `opus_packet_get_nb_samples` for every first and second byte.
#[test]
fn nb_frames_and_nb_samples_match_libopus() {
    for toc in 0..=255u8 {
        for second in 0..=255u8 {
            for len in [1usize, 2] {
                let bytes = [toc, second];
                let data = &bytes[..len];
                let c_frames = unsafe { sys::opus_packet_get_nb_frames(data.as_ptr(), len as i32) };
                let ours = packet::nb_frames(data);
                if c_frames < 0 {
                    assert!(ours.is_err(), "{data:02x?}: libopus error {c_frames}");
                } else {
                    assert_eq!(ours, Ok(c_frames as usize), "{data:02x?}");
                }
                for fs in [SampleRate::Hz8000, SampleRate::Hz48000] {
                    let c_samples = unsafe { sys::opus_packet_get_nb_samples(data.as_ptr(), len as i32, fs.hz() as i32) };
                    let ours = packet::nb_samples(data, fs);
                    if c_samples < 0 {
                        assert!(ours.is_err(), "{data:02x?} at {fs:?}: libopus error {c_samples}");
                    } else {
                        assert_eq!(ours, Ok(c_samples as usize), "{data:02x?} at {fs:?}");
                    }
                }
            }
        }
    }
    assert!(packet::nb_frames(&[]).is_err());
    assert!(packet::nb_samples(&[], SampleRate::Hz48000).is_err());
}
