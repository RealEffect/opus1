//! The `packet` module against libopus.
//!
//! The first byte of the input picks a test; the rest is its data.
//!
//! 1. A packet from the input. `parse` (plain and self-delimited),
//!    `nb_frames`, `nb_samples` and `samples_per_frame` must give what
//!    libopus `opus_packet_parse_impl`, `opus_packet_get_nb_frames`,
//!    `opus_packet_get_nb_samples` and `opus_packet_get_samples_per_frame`
//!    give. The frames of a packet written again with `write` must parse to
//!    the same frames. `pad` must give the bytes of `opus_packet_pad`, and
//!    `write` of the parsed frames the bytes of the libopus repacketizer.
//! 2. A multistream packet from the input. `multistream_validate` must
//!    accept the packets that the checks of libopus's multistream decoder
//!    accept, with the same duration. `multistream_pad` must give the bytes
//!    of `opus_multistream_packet_pad`. `split_self_delimited` must cut out
//!    the first stream with its frames unchanged.
//! 3. Frames from the input. `write` must give the bytes of the libopus
//!    repacketizer, plain, self-delimited and padded, and the result must
//!    parse to the same frames.
//!
//! libopus reads non-zero padding as packet extensions, which opus1 does not
//! port (see the `packet` module documentation). So before a comparison
//! with libopus the padding of the packet is set to zeros.

#![no_main]

mod common;

use common::*;
use libfuzzer_sys::arbitrary::{Result, Unstructured};
use libfuzzer_sys::fuzz_target;
use opus1::SampleRate;
use opus1::multistream::split_self_delimited;
use opus1::packet::{self, ParsedPacket};

const OPUS_BAD_ARG: i32 = -1;
const OPUS_INVALID_PACKET: i32 = -4;

fuzz_target!(|data: &[u8]| {
    let _ = run(data);
});

fn run(data: &[u8]) -> Result<()> {
    let Some((&which, rest)) = data.split_first() else {
        return Ok(());
    };
    match which % 3 {
        0 => single(rest),
        1 => multistream(rest),
        _ => frames(rest),
    }
}

/// opus1's result in the form of the C result: 0 for success, else the
/// libopus error code.
fn c_form<T>(r: &std::result::Result<T, opus1::Error>) -> i32 {
    match r {
        Ok(_) => 0,
        Err(e) => e.to_raw(),
    }
}

/// The frames of a parsed packet, copied.
fn frames_of(p: &ParsedPacket, data: &[u8]) -> Vec<Vec<u8>> {
    p.frames(data).map(|f| f.to_vec()).collect()
}

/// Compare `parse` with `opus_packet_parse_impl`.
fn compare_parse(data: &[u8], self_delimited: bool) -> Option<ParsedPacket> {
    let rust = packet::parse(data, self_delimited);
    let c = c_parse(data, self_delimited);
    match (&rust, &c) {
        (Ok(r), Ok(c)) => {
            let got = CParsed {
                toc: r.toc,
                sizes: r.sizes[..r.count].to_vec(),
                payload_offset: r.payload_offset,
                packet_offset: r.packet_offset,
                padding_len: r.padding_len,
            };
            assert_eq!(&got, c, "parse(sd={self_delimited}) differs from libopus for {data:02x?}");
            assert!(r.packet_offset <= data.len());
            Some(*r)
        }
        (Err(e), Err(c)) => {
            assert_eq!(e.to_raw(), *c, "parse(sd={self_delimited}) error for {data:02x?}");
            None
        }
        _ => panic!(
            "parse(sd={self_delimited}) accept/reject differs: opus1 {rust:?}, libopus {c:?} for {data:02x?}"
        ),
    }
}

/// Set the padding of the plain packet `data` to zeros, as libopus would
/// otherwise read it as extensions.
fn zero_padding(data: &mut [u8]) {
    if let Ok(p) = c_parse(data, false) {
        data[p.packet_offset - p.padding_len..p.packet_offset].fill(0);
    }
}

/// Compare `pad` with `opus_packet_pad` for the plain packet `data`
/// grown by `extra` bytes. `data` has zero padding.
fn compare_pad(data: &[u8], extra: usize) {
    let new_len = data.len() + extra;
    let mut rust_buf = data.to_vec();
    rust_buf.resize(new_len, 0xA5);
    let mut c_buf = rust_buf.clone();
    let rust = packet::pad(&mut rust_buf, data.len(), new_len);
    let c = unsafe {
        audiopus_sys::opus_packet_pad(c_buf.as_mut_ptr(), data.len() as i32, new_len as i32)
    };
    assert_eq!(c_form(&rust), c, "pad by {extra}: opus1 {rust:?}, libopus {c} for {data:02x?}");
    if c == 0 {
        assert_eq!(rust_buf, c_buf, "pad by {extra} gives other bytes than libopus for {data:02x?}");
        if extra > 0 {
            // The padded packet holds the same frames.
            let before = packet::parse(data, false).expect("padded a packet that does not parse");
            let after = packet::parse(&rust_buf, false).expect("the padded packet does not parse");
            assert_eq!(frames_of(&before, data), frames_of(&after, &rust_buf));
            assert_eq!(before.toc & 0xFC, after.toc & 0xFC);
            assert_eq!(after.packet_offset, new_len);
        }
    }
}

/// Compare `write` of `frames` with the libopus repacketizer, for an output
/// buffer of `maxlen` bytes.
fn compare_write(toc: u8, frames: &[&[u8]], maxlen: usize, self_delimited: bool, pad: bool) {
    let mut out = vec![0x5Au8; maxlen];
    let rust = packet::write(toc, frames, &mut out, self_delimited, pad);

    // libopus: one code-0 packet per frame into the repacketizer.
    let singles: Vec<Vec<u8>> = frames
        .iter()
        .map(|f| {
            let mut p = vec![toc & 0xFC];
            p.extend_from_slice(f);
            p
        })
        .collect();
    let mut rp = CRepacketizer::new();
    let mut cat_ok = !frames.is_empty();
    for p in &singles {
        if rp.cat(p) != 0 {
            cat_ok = false;
            break;
        }
    }
    if !cat_ok {
        // libopus does not take these frames (too many, too long, more
        // than 120 ms). Neither may opus1.
        assert_eq!(
            rust.map_err(|e| e.to_raw()),
            Err(OPUS_BAD_ARG),
            "write took frames the repacketizer refuses"
        );
        return;
    }
    let mut c_out = vec![0x5Au8; maxlen];
    let c = rp.out(&mut c_out, self_delimited, pad);
    match rust {
        Ok(n) => {
            assert_eq!(n as i32, c, "write(sd={self_delimited}, pad={pad}, maxlen={maxlen}) length");
            assert_eq!(out[..n], c_out[..n], "write(sd={self_delimited}, pad={pad}) bytes");
            if pad {
                assert_eq!(n, maxlen);
            }
            // The packet parses to the frames it was made of.
            let p = packet::parse(&out[..n], self_delimited).expect("a written packet does not parse");
            assert_eq!(p.packet_offset, n);
            assert_eq!(p.toc & 0xFC, toc & 0xFC);
            let got: Vec<&[u8]> = p.frames(&out[..n]).collect();
            assert_eq!(got, frames);
        }
        Err(e) => assert_eq!(e.to_raw(), c, "write(sd={self_delimited}, pad={pad}) error"),
    }
}

fn single(rest: &[u8]) -> Result<()> {
    let mut u = Unstructured::new(rest);
    let extra = *u.choose(&[0usize, 1, 2, 3, 10, 254, 255, 256, 600])?;
    let maxlen_extra = u.int_in_range(0..=300)?;
    let shrink = u.int_in_range(0..=4)?;
    let modes = u.int_in_range(0..=3u8)?;
    let data = u.take_rest();

    // The duration fields at every rate.
    for fs in RATES {
        let rate = SampleRate::try_from(fs).unwrap();
        if let Some(&toc) = data.first() {
            let c = unsafe { audiopus_sys::opus_packet_get_samples_per_frame(&toc, fs) };
            assert_eq!(packet::samples_per_frame(toc, rate) as i32, c, "samples_per_frame({toc:#x}, {fs})");
        }
        let rust = packet::nb_samples(data, rate);
        let c = c_nb_samples(data, fs);
        match rust {
            Ok(n) => assert_eq!(n as i32, c, "nb_samples at {fs}"),
            Err(e) => assert_eq!(e.to_raw(), c, "nb_samples error at {fs}"),
        }
    }
    let c = unsafe { audiopus_sys::opus_packet_get_nb_frames(data_ptr(data), data.len() as i32) };
    match packet::nb_frames(data) {
        Ok(n) => assert_eq!(n as i32, c, "nb_frames"),
        Err(e) => assert_eq!(e.to_raw(), c, "nb_frames error"),
    }

    let plain = compare_parse(data, false);
    let delimited = compare_parse(data, true);

    // Rust-only: the frames written again parse to the same frames, in no
    // more bytes than the packet took.
    for (p, sd) in [(plain, false), (delimited, true)] {
        let Some(p) = p else { continue };
        let frames = frames_of(&p, data);
        let refs: Vec<&[u8]> = frames.iter().map(|f| f.as_slice()).collect();
        let mut out = vec![0u8; p.packet_offset];
        let n = packet::write(p.toc, &refs, &mut out, sd, false)
            .expect("the frames of a parsed packet do not fit in its length");
        let again = packet::parse(&out[..n], sd).expect("a written packet does not parse");
        assert_eq!(frames_of(&again, &out[..n]), frames);
        assert_eq!(again.toc & 0xFC, p.toc & 0xFC);
    }

    // Padding: opus1 alone on the packet as it is, then against libopus on
    // the packet with zero padding.
    if plain.is_some() {
        let mut buf = data.to_vec();
        buf.resize(data.len() + extra, 0);
        if packet::pad(&mut buf, data.len(), data.len() + extra).is_ok() && extra > 0 {
            let p = packet::parse(&buf, false).expect("a padded packet does not parse");
            assert_eq!(p.packet_offset, data.len() + extra);
        }
    }
    let mut zeroed = data.to_vec();
    zero_padding(&mut zeroed);
    if !zeroed.is_empty() {
        compare_pad(&zeroed, extra);
    }
    // `pad` arguments that do not fit, against libopus: a shorter new
    // length, and lengths of 0.
    if zeroed.len() > shrink && shrink > 0 {
        compare_pad_lengths(&zeroed, zeroed.len(), zeroed.len() - shrink);
    }

    // The parsed frames written by opus1 and by the libopus repacketizer.
    if let Some(p) = packet::parse(&zeroed, false).ok() {
        let frames = frames_of(&p, &zeroed);
        let refs: Vec<&[u8]> = frames.iter().map(|f| f.as_slice()).collect();
        let maxlen = (p.packet_offset + maxlen_extra).saturating_sub(shrink);
        compare_write(p.toc, &refs, maxlen, modes & 1 != 0, modes & 2 != 0);
    }
    Ok(())
}

/// `pad` with a new length below the old one: both must refuse.
fn compare_pad_lengths(data: &[u8], len: usize, new_len: usize) {
    let mut rust_buf = data.to_vec();
    let mut c_buf = data.to_vec();
    let rust = packet::pad(&mut rust_buf, len, new_len);
    let c = unsafe { audiopus_sys::opus_packet_pad(c_buf.as_mut_ptr(), len as i32, new_len as i32) };
    assert_eq!(c_form(&rust), c, "pad({len} -> {new_len})");
    assert_eq!(rust_buf, c_buf);
}

/// libopus `opus_multistream_packet_validate` (a static function of
/// opus_multistream_decoder.c), built from the exported functions it calls.
fn c_ms_validate(data: &[u8], nb_streams: usize, fs: i32) -> i32 {
    let mut off = 0usize;
    let mut samples = 0;
    for s in 0..nb_streams {
        if data.len() <= off {
            return OPUS_INVALID_PACKET;
        }
        let p = match c_parse(&data[off..], s != nb_streams - 1) {
            Ok(p) => p,
            Err(e) => return e,
        };
        let tmp = c_nb_samples(&data[off..off + p.packet_offset], fs);
        if s != 0 && samples != tmp {
            return OPUS_INVALID_PACKET;
        }
        samples = tmp;
        off += p.packet_offset;
    }
    samples
}

/// Zero the padding of the last stream of a multistream packet.
fn zero_last_padding(data: &mut [u8], nb_streams: usize) {
    let mut off = 0usize;
    for _ in 1..nb_streams {
        if off >= data.len() {
            return;
        }
        match c_parse(&data[off..], true) {
            Ok(p) => off += p.packet_offset,
            Err(_) => return,
        }
    }
    if off < data.len() {
        zero_padding(&mut data[off..]);
    }
}

fn multistream(rest: &[u8]) -> Result<()> {
    let mut u = Unstructured::new(rest);
    let nb_streams = if u.ratio(1, 16)? { u.int_in_range(1..=255)? } else { u.int_in_range(1..=6)? };
    let fs = *u.choose(&RATES)?;
    let rate = SampleRate::try_from(fs).unwrap();
    let extra = *u.choose(&[0usize, 1, 2, 5, 255, 300])?;
    let data = u.take_rest();

    let rust = packet::multistream_validate(data, nb_streams, rate);
    let c = c_ms_validate(data, nb_streams, fs);
    match rust {
        Ok(n) => assert_eq!(n as i32, c, "multistream_validate({nb_streams} streams, {fs} Hz)"),
        Err(e) => assert!(c < 0, "multistream_validate refused ({e}), libopus gives {c}"),
    }

    // The first stream cut out as a plain packet.
    let mut scratch = Vec::new();
    match (split_self_delimited(data, &mut scratch), packet::parse(data, true)) {
        (Ok((plain, used)), Ok(p)) => {
            assert_eq!(used, p.packet_offset);
            let q = packet::parse(plain, false).expect("a cut-out stream does not parse");
            assert_eq!(frames_of(&q, plain), frames_of(&p, data));
            assert_eq!(q.toc, p.toc);
            assert_eq!(q.padding_len, p.padding_len);
        }
        (Err(_), Err(_)) => {}
        (a, b) => panic!("split_self_delimited {:?} but parse {:?}", a.map(|x| x.1), b.map(|p| p.count)),
    }

    // Padding the last stream, against libopus.
    let mut zeroed = data.to_vec();
    zero_last_padding(&mut zeroed, nb_streams);
    if !zeroed.is_empty() {
        let new_len = zeroed.len() + extra;
        let mut rust_buf = zeroed.clone();
        rust_buf.resize(new_len, 0xA5);
        let mut c_buf = rust_buf.clone();
        let rust = packet::multistream_pad(&mut rust_buf, zeroed.len(), new_len, nb_streams);
        let c = unsafe {
            audiopus_sys::opus_multistream_packet_pad(
                c_buf.as_mut_ptr(),
                zeroed.len() as i32,
                new_len as i32,
                nb_streams as i32,
            )
        };
        assert_eq!(c_form(&rust), c, "multistream_pad({nb_streams} streams) by {extra}: {rust:?}");
        if c == 0 {
            assert_eq!(rust_buf, c_buf, "multistream_pad gives other bytes than libopus");
            // Padding keeps the duration of a valid packet.
            if let Ok(n) = packet::multistream_validate(&zeroed, nb_streams, rate) {
                assert_eq!(packet::multistream_validate(&rust_buf, nb_streams, rate), Ok(n));
            }
        }
    }
    Ok(())
}

fn frames(rest: &[u8]) -> Result<()> {
    let mut u = Unstructured::new(rest);
    let toc: u8 = u.arbitrary()?;
    let count = if u.ratio(1, 8)? { u.int_in_range(0..=50)? } else { u.int_in_range(1..=6)? };
    let same = u.arbitrary::<bool>()?;
    let mut sizes = Vec::with_capacity(count);
    for i in 0..count {
        let size = if same && i > 0 {
            sizes[0]
        } else {
            match u.int_in_range(0..=9)? {
                0 => 0,
                1 => u.int_in_range(250..=260)?,
                2 => u.int_in_range(1270..=1280)?,
                _ => u.int_in_range(0..=40)?,
            }
        };
        sizes.push(size);
    }
    let fill: u8 = u.arbitrary()?;
    let frames: Vec<Vec<u8>> = sizes
        .iter()
        .enumerate()
        .map(|(i, &n)| (0..n).map(|k| fill.wrapping_add((i * 31 + k) as u8)).collect())
        .collect();
    let refs: Vec<&[u8]> = frames.iter().map(|f| f.as_slice()).collect();
    let payload: usize = sizes.iter().sum();
    let maxlen = match u.int_in_range(0..=3)? {
        0 => u.int_in_range(0..=payload + 8)?,
        1 => payload + u.int_in_range(0..=4)?,
        2 => payload + u.int_in_range(0..=600)?,
        _ => u.int_in_range(0..=4000)?,
    };
    for (sd, pad) in [(false, false), (true, false), (false, true), (true, true)] {
        compare_write(toc, &refs, maxlen, sd, pad);
    }
    Ok(())
}
