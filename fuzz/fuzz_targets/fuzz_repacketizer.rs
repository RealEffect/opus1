//! The repacketizer and the padding functions of `opus1::packet` on packets
//! whose padding holds extensions, against libopus (repacketizer.c).
//!
//! The input gives up to 4 packets: raw bytes, packets written by
//! `packet::write`, or code 3 packets whose padding is raw bytes or
//! extensions from `extensions::generate`. Most share one TOC byte. Then up
//! to 12 steps from the input, each compared with libopus:
//!
//! - `Repacketizer::cat` of every packet (and `reset`), against
//!   `opus_repacketizer_cat`: the same result and frame count.
//! - `out_range_ext` with a random range, buffer length, self-delimiting,
//!   padding and added extensions, against
//!   `opus_repacketizer_out_range_impl`; `out` against
//!   `opus_repacketizer_out`. The same result and bytes. A packet that is
//!   written fits the buffer, fills it with `pad`, and parses to the frames
//!   of the range.
//! - `pad` and `pad_with_extensions` of one packet against
//!   `opus_packet_pad` and `opus_packet_pad_impl`; `unpad` against
//!   `opus_packet_unpad`. The same result and bytes. A padded or unpadded
//!   packet keeps its frames; an unpadded packet has no padding.
//! - `multistream_pad` and `multistream_unpad` of a multistream packet
//!   (raw bytes, or packets joined by libopus with all but the last
//!   self-delimited) against `opus_multistream_packet_pad` and
//!   `opus_multistream_packet_unpad`. The same result and bytes.

#![no_main]

mod common;

use audiopus_sys as sys;
use common::*;
use libfuzzer_sys::arbitrary::{Result, Unstructured};
use libfuzzer_sys::fuzz_target;
use opus1::extensions::{self, Extension};
use opus1::packet::{self, Repacketizer};
use std::os::raw::c_int;

fuzz_target!(|data: &[u8]| {
    let mut u = Unstructured::new(data);
    let _ = run(&mut u);
});

/// The frame sizes of a code 3 packet: one byte below 252, else two.
fn push_size(out: &mut Vec<u8>, size: usize) {
    if size < 252 {
        out.push(size as u8);
    } else {
        let first = 252 + (size & 3);
        out.push(first as u8);
        out.push(((size - first) >> 2) as u8);
    }
}

/// A code 3 packet (VBR) of `frames` whose padding is `padding`.
fn code3(toc: u8, frames: &[&[u8]], padding: &[u8], padding_flag: bool) -> Vec<u8> {
    let mut out = vec![toc | 3, 0x80 | frames.len() as u8 | if padding_flag { 0x40 } else { 0 }];
    if padding_flag {
        let mut p = padding.len();
        while p > 254 {
            out.push(255);
            p -= 254;
        }
        out.push(p as u8);
    }
    for f in &frames[..frames.len() - 1] {
        push_size(&mut out, f.len());
    }
    for f in frames {
        out.extend_from_slice(f);
    }
    if padding_flag {
        out.extend_from_slice(padding);
    }
    out
}

fn arbitrary_packet(u: &mut Unstructured, toc: u8, pool: &[u8]) -> Result<Vec<u8>> {
    let kind = u.int_in_range(0..=5)?;
    if kind == 0 {
        let n = u.int_in_range(0..=300usize)?.min(u.len());
        return Ok(u.bytes(n)?.to_vec());
    }
    let max = 48 / (packet::samples_per_frame(toc, 48000) / 120);
    let count = if u.ratio(1, 8)? { u.int_in_range(1..=max)? } else { u.int_in_range(1..=max.min(4))? };
    let mut frames = Vec::with_capacity(count);
    for _ in 0..count {
        let size = match u.int_in_range(0..=15)? {
            0 => u.int_in_range(252..=1275)?,
            1 => u.int_in_range(0..=1280)?,
            _ => u.int_in_range(0..=24)?,
        };
        let size = size.min(pool.len());
        let start = u.int_in_range(0..=pool.len() - size)?;
        frames.push(&pool[start..start + size]);
    }
    if kind == 1 {
        let mut out = vec![0u8; 48 * 1277 + 8];
        return match packet::write(toc, &frames, &mut out, false, false) {
            Ok(n) => {
                out.truncate(n);
                Ok(out)
            }
            Err(_) => Ok(code3(toc, &frames, &[], false)),
        };
    }
    let padding: Vec<u8> = match u.int_in_range(0..=3)? {
        0 => {
            let n = u.int_in_range(0..=300usize)?.min(u.len());
            u.bytes(n)?.to_vec()
        }
        _ => {
            let exts: Vec<Extension> = arbitrary_extensions(u, pool, count, 40)?
                .into_iter()
                .filter(|e| (3..=127).contains(&e.id) && e.frame < count && (e.id >= 32 || e.data.len() <= 1))
                .collect();
            let mut buf = vec![0u8; 40 * 710 + 2 * 48];
            let n = extensions::generate(&mut buf, &exts, count, false).unwrap_or(0);
            buf.truncate(n);
            let fill = if u.ratio(1, 4)? { u.int_in_range(1..=3)? } else { 0 };
            let mut padding = vec![0x01; fill];
            padding.extend_from_slice(&buf);
            padding
        }
    };
    let flag = !padding.is_empty() || u.arbitrary()?;
    Ok(code3(toc, &frames, &padding, flag))
}

/// The frames of `data`, or nothing when it does not parse.
fn frames_of(data: &[u8], self_delimited: bool) -> Option<Vec<Vec<u8>>> {
    let p = packet::parse(data, self_delimited).ok()?;
    Some(p.frames(data).map(|f| f.to_vec()).collect())
}

fn run(u: &mut Unstructured) -> Result<()> {
    let pool_len = u.int_in_range(0..=2000usize)?.min(u.len());
    let pool = u.bytes(pool_len)?;
    let base_toc: u8 = u.arbitrary()?;
    let nb_packets = u.int_in_range(1..=4)?;
    let mut packets = Vec::with_capacity(nb_packets);
    for _ in 0..nb_packets {
        let toc = if u.ratio(1, 8)? { u.arbitrary()? } else { base_toc };
        packets.push(arbitrary_packet(u, toc, pool)?);
    }
    let packets = packets;

    // Both repacketizers take every packet; libopus keeps pointers into
    // `packets`, which live to the end.
    let mut rp = Repacketizer::new();
    let mut c_rp = CRepacketizer::new();
    let mut frames: Vec<&[u8]> = Vec::new();
    for (i, p) in packets.iter().enumerate() {
        let c_ret = c_rp.cat(p);
        let ret = rp.cat(p);
        match ret {
            Ok(()) => {
                assert_eq!(c_ret, 0, "cat of packet {i}");
                let p = packet::parse(p, false).expect("a packet that cat takes parses");
                frames.extend(p.frames(&packets[i]));
            }
            Err(e) => assert_eq!(e.to_raw(), c_ret, "cat of packet {i}"),
        }
        assert_eq!(rp.nb_frames() as i32, c_rp.nb_frames(), "frames after packet {i}");
        assert_eq!(rp.nb_frames(), frames.len());
    }

    for step in 0..u.int_in_range(1..=12)? {
        let what = format!("step {step}");
        match u.int_in_range(0..=5)? {
            0 | 1 => {
                let nb = rp.nb_frames();
                let begin = u.int_in_range(0..=nb + 1)?;
                let end = u.int_in_range(0..=nb + 1)?;
                let self_delimited = u.arbitrary()?;
                let pad = u.arbitrary()?;
                let extra =
                    if u.ratio(1, 2)? { arbitrary_extensions(u, pool, end.saturating_sub(begin), 20)? } else { Vec::new() };
                let maxlen = match u.int_in_range(0..=3)? {
                    0 => u.int_in_range(0..=200)?,
                    1 => u.int_in_range(0..=70_000)?,
                    _ => 1277 * 48 + 2000 + u.int_in_range(0..=3000)?,
                };
                let c_extra = to_c(&extra);
                let mut c_out = vec![0xEEu8; maxlen];
                let c_ret = unsafe {
                    opus_repacketizer_out_range_impl(
                        c_rp.0,
                        begin as c_int,
                        end as c_int,
                        c_out.as_mut_ptr(),
                        maxlen as i32,
                        self_delimited as c_int,
                        pad as c_int,
                        c_extra.as_ptr() as *const u8,
                        c_extra.len() as c_int,
                    )
                };
                let mut out = vec![0xEEu8; maxlen];
                let what = format!(
                    "{what}: out_range_ext {begin}..{end} of {nb}, {maxlen} bytes, self-delimited {self_delimited}, pad {pad}, {} extra",
                    extra.len()
                );
                match rp.out_range_ext(begin, end, &mut out, self_delimited, pad, &extra) {
                    Ok(n) => {
                        assert_eq!(n as i32, c_ret, "{what}");
                        assert!(n <= maxlen, "{what}: {n} bytes");
                        if pad {
                            assert_eq!(n, maxlen, "{what}: a padded packet fills the buffer");
                        }
                        assert_eq!(out[..n], c_out[..n], "{what}: bytes");
                        let back = frames_of(&out[..n], self_delimited).unwrap_or_else(|| panic!("{what}: does not parse"));
                        let want: Vec<Vec<u8>> = frames[begin..end].iter().map(|f| f.to_vec()).collect();
                        assert_eq!(back, want, "{what}: frames");
                    }
                    Err(e) => assert_eq!(e.to_raw(), c_ret, "{what}: error"),
                }
            }
            2 => {
                let maxlen = u.int_in_range(0..=70_000)?;
                let mut c_out = vec![0xEEu8; maxlen];
                let c_ret = unsafe { sys::opus_repacketizer_out(c_rp.0, c_out.as_mut_ptr(), maxlen as i32) };
                let mut out = vec![0xEEu8; maxlen];
                match rp.out(&mut out) {
                    Ok(n) => {
                        assert_eq!(n as i32, c_ret, "{what}: out into {maxlen}");
                        assert_eq!(out[..n], c_out[..n], "{what}: out bytes");
                    }
                    Err(e) => assert_eq!(e.to_raw(), c_ret, "{what}: out into {maxlen}"),
                }
            }
            3 => {
                let i = u.int_in_range(0..=packets.len() - 1)?;
                let data = &packets[i];
                let len = if u.ratio(1, 10)? { u.int_in_range(0..=data.len())? } else { data.len() };
                let new_len = match u.int_in_range(0..=3)? {
                    0 => u.int_in_range(0..=len + 2)?,
                    _ => len + u.int_in_range(0..=3000)?,
                };
                let pad = u.arbitrary()?;
                let count = packet::nb_frames(&data[..len]).unwrap_or(1);
                let with_extra = u.arbitrary()?;
                let extra = if with_extra { arbitrary_extensions(u, pool, count, 20)? } else { Vec::new() };
                let c_extra = to_c(&extra);
                let mut c_buf = data.clone();
                c_buf.resize(new_len.max(len) + 8, 0xEE);
                let mut buf = c_buf.clone();
                let c_ret = unsafe {
                    if with_extra {
                        opus_packet_pad_impl(c_buf.as_mut_ptr(), len as i32, new_len as i32, pad as c_int, c_extra.as_ptr(), c_extra.len() as c_int)
                    } else {
                        sys::opus_packet_pad(c_buf.as_mut_ptr(), len as i32, new_len as i32)
                    }
                };
                let ret = if with_extra {
                    packet::pad_with_extensions(&mut buf, len, new_len, pad, &extra)
                } else {
                    packet::pad(&mut buf, len, new_len).map(|_| 0)
                };
                let what = format!("{what}: pad packet {i} from {len} to {new_len}, pad {pad}, {} extra", extra.len());
                match ret {
                    Ok(n) => {
                        assert!(c_ret >= 0, "{what}: libopus gives {c_ret}");
                        if with_extra {
                            assert_eq!(n as i32, c_ret, "{what}");
                        }
                        // 0: the length did not change, and nothing was done.
                        let n = if with_extra && n > 0 { n } else { new_len };
                        assert!(n <= new_len, "{what}: {n} bytes");
                        assert_eq!(buf[..n], c_buf[..n], "{what}: bytes");
                        assert_eq!(frames_of(&buf[..n], false), frames_of(&data[..len], false), "{what}: frames");
                    }
                    Err(e) => assert_eq!(e.to_raw(), c_ret, "{what}: error"),
                }
            }
            4 => {
                let i = u.int_in_range(0..=packets.len() - 1)?;
                let data = &packets[i];
                let len = if u.ratio(1, 10)? { u.int_in_range(0..=data.len())? } else { data.len() };
                let mut c_buf = data.clone();
                let c_ret = unsafe { sys::opus_packet_unpad(c_buf.as_mut_ptr(), len as i32) };
                let mut buf = data.clone();
                let what = format!("{what}: unpad packet {i} of {len} bytes");
                match packet::unpad(&mut buf, len) {
                    Ok(n) => {
                        assert_eq!(n as i32, c_ret, "{what}");
                        assert!(n <= len, "{what}: {n} bytes");
                        assert_eq!(buf[..n], c_buf[..n], "{what}: bytes");
                        let p = packet::parse(&buf[..n], false).unwrap_or_else(|e| panic!("{what}: does not parse: {e}"));
                        assert_eq!(p.padding_len, 0, "{what}: padding left");
                        assert_eq!(frames_of(&buf[..n], false), frames_of(&data[..len], false), "{what}: frames");
                    }
                    Err(e) => assert_eq!(e.to_raw(), c_ret, "{what}: error"),
                }
            }
            _ => {
                // A multistream packet: raw bytes, or the packets joined by
                // libopus, all but the last self-delimited.
                let nb_streams = u.int_in_range(0..=6)?;
                let data: Vec<u8> = if u.ratio(1, 4)? {
                    let n = u.int_in_range(0..=400usize)?.min(u.len());
                    u.bytes(n)?.to_vec()
                } else {
                    let mut ms = Vec::new();
                    for (s, p) in packets.iter().enumerate() {
                        let mut one = CRepacketizer::new();
                        if one.cat(p) != 0 {
                            continue;
                        }
                        let mut out = vec![0u8; 48 * 1277 + 8];
                        let n = one.out(&mut out, s + 1 < packets.len(), false);
                        if n > 0 {
                            ms.extend_from_slice(&out[..n as usize]);
                        }
                    }
                    ms
                };
                let len = data.len();
                let new_len = match u.int_in_range(0..=3)? {
                    0 => u.int_in_range(0..=len + 2)?,
                    _ => len + u.int_in_range(0..=3000)?,
                };
                let mut c_buf = data.clone();
                c_buf.resize(new_len.max(len) + 8, 0xEE);
                let mut buf = c_buf.clone();
                let c_ret =
                    unsafe { sys::opus_multistream_packet_pad(c_buf.as_mut_ptr(), len as i32, new_len as i32, nb_streams as c_int) };
                let what_pad = format!("{what}: multistream_pad of {nb_streams} streams from {len} to {new_len}");
                match packet::multistream_pad(&mut buf, len, new_len, nb_streams) {
                    Ok(()) => {
                        assert_eq!(c_ret, 0, "{what_pad}");
                        assert_eq!(buf[..new_len], c_buf[..new_len], "{what_pad}: bytes");
                    }
                    Err(e) => assert_eq!(e.to_raw(), c_ret, "{what_pad}: error"),
                }
                let mut c_buf = data.clone();
                let c_ret = unsafe { sys::opus_multistream_packet_unpad(c_buf.as_mut_ptr(), len as i32, nb_streams as c_int) };
                let mut buf = data.clone();
                let what = format!("{what}: multistream_unpad of {nb_streams} streams, {len} bytes");
                match packet::multistream_unpad(&mut buf, len, nb_streams) {
                    Ok(n) => {
                        assert_eq!(n as i32, c_ret, "{what}");
                        assert!(n <= len, "{what}: {n} bytes");
                        assert_eq!(buf[..n], c_buf[..n], "{what}: bytes");
                    }
                    Err(e) => assert_eq!(e.to_raw(), c_ret, "{what}: error"),
                }
            }
        }
        if u.ratio(1, 16)? {
            // Start again: reset both and take the packets once more.
            rp.reset();
            c_rp.init();
            frames.clear();
            for p in &packets {
                let c_ret = c_rp.cat(p);
                match rp.cat(p) {
                    Ok(()) => {
                        assert_eq!(c_ret, 0, "cat after reset");
                        frames.extend(packet::parse(p, false).unwrap().frames(p));
                    }
                    Err(e) => assert_eq!(e.to_raw(), c_ret, "cat after reset"),
                }
            }
            assert_eq!(rp.nb_frames() as i32, c_rp.nb_frames(), "frames after reset");
        }
    }
    Ok(())
}
