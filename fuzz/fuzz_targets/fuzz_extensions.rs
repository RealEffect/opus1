//! Packet extensions (`opus1::extensions`) against libopus's extension code
//! (src/extensions.c).
//!
//! The first input byte picks a test; the rest is its data.
//!
//! 1. The data is the padding of a packet of 1 to 48 frames. `count`,
//!    `count_ext`, `parse` (into a list with room for all and into one that
//!    is too short), `parse_ext` and the iterator (`next`, `find_id`,
//!    `set_frame_max`, `reset`) must give libopus's results: the same
//!    extensions (ID, frame, payload) or the same error. When the padding
//!    is valid, `generate` of its extensions must give libopus's bytes,
//!    and they must read back as the same extensions in frame order.
//! 2. A list of extensions from the data, some of them invalid.
//!    `generate` and `generated_len` must give libopus's result, and
//!    `generate` its bytes. A list that is written must read back as the
//!    same extensions in frame order, and fit in the buffer; with `pad` it
//!    fills the buffer.

#![no_main]

mod common;

use common::*;
use libfuzzer_sys::arbitrary::{Result, Unstructured};
use libfuzzer_sys::fuzz_target;
use opus1::extensions::{self, Extension, ExtensionIterator};
use std::os::raw::c_int;

fuzz_target!(|data: &[u8]| {
    let Some((&test, rest)) = data.split_first() else { return };
    let mut u = Unstructured::new(rest);
    let _ = match test % 2 {
        0 => read_padding(&mut u),
        _ => write_list(&mut u),
    };
});

/// The extensions of `data` in frame order, from `count_ext` and
/// `parse_ext`, after checking both against libopus.
fn parse_in_frame_order<'a>(data: &'a [u8], nb_frames: usize, what: &str) -> Result<Vec<Extension<'a>>, opus1::Error> {
    let len = data.len() as i32;
    let nf = nb_frames as c_int;
    let mut c_counts = [0i32; 48];
    let c_total = unsafe { opus_packet_extensions_count_ext(data.as_ptr(), len, c_counts.as_mut_ptr(), nf) };
    let mut counts = [0usize; 48];
    let total = extensions::count_ext(data, &mut counts, nb_frames).unwrap_or_else(|e| panic!("{what}: count_ext: {e}"));
    assert_eq!(total, c_total as usize, "{what}: count_ext");
    for f in 0..nb_frames {
        assert_eq!(counts[f], c_counts[f] as usize, "{what}: count_ext of frame {f}");
    }
    let mut c_exts = vec![C_EXT_EMPTY; total + 1];
    let mut c_n = c_exts.len() as i32;
    let c_ret =
        unsafe { opus_packet_extensions_parse_ext(data.as_ptr(), len, c_exts.as_mut_ptr(), &mut c_n, c_counts.as_ptr(), nf) };
    let mut ours = vec![Extension::default(); total + 1];
    match extensions::parse_ext(data, &mut ours, &counts, nb_frames) {
        Ok(n) => {
            assert_eq!((c_ret, c_n as usize), (0, n), "{what}: parse_ext");
            let a: Vec<Plain> = ours[..n].iter().map(plain).collect();
            let b: Vec<Plain> = c_exts[..n].iter().map(c_plain).collect();
            assert_eq!(a, b, "{what}: parse_ext");
            assert!(a.windows(2).all(|w| w[0].1 <= w[1].1), "{what}: frame order");
            ours.truncate(n);
            Ok(ours)
        }
        Err(e) => {
            assert_eq!(e.to_raw(), c_ret, "{what}: parse_ext error");
            Err(e)
        }
    }
}

fn read_padding(u: &mut Unstructured) -> Result<()> {
    let nb_frames = u.int_in_range(1..=48)?;
    let frame_max = u.int_in_range(0..=49)?;
    let find_id: Option<u8> = u.arbitrary()?;
    let frame_max_step = u.int_in_range(0..=8)?;
    let reset_step = u.int_in_range(0..=16)?;
    let short_by = u.int_in_range(1..=8)?;
    let slack = u.int_in_range(0..=16)?;
    let pad = u.arbitrary::<bool>()?;
    let data = u.bytes(u.len())?;
    let what = format!("{} bytes of padding, {nb_frames} frames", data.len());
    let len = data.len() as i32;
    let nf = nb_frames as c_int;

    // count
    let c_count = unsafe { opus_packet_extensions_count(data.as_ptr(), len, nf) };
    let count = extensions::count(data, nb_frames).unwrap_or_else(|e| panic!("{what}: count: {e}"));
    assert_eq!(count, c_count as usize, "{what}: count");

    // parse, into a list with room for one more
    let mut c_exts = vec![C_EXT_EMPTY; count + 1];
    let mut c_n = c_exts.len() as i32;
    let c_ret = unsafe { opus_packet_extensions_parse(data.as_ptr(), len, c_exts.as_mut_ptr(), &mut c_n, nf) };
    let mut ours = vec![Extension::default(); count + 1];
    let parsed = extensions::parse(data, &mut ours, nb_frames);
    match parsed {
        Ok(n) => {
            assert_eq!((c_ret, c_n as usize), (0, n), "{what}: parse");
            assert_eq!(n, count, "{what}: parse and count");
            let a: Vec<Plain> = ours[..n].iter().map(plain).collect();
            let b: Vec<Plain> = c_exts[..n].iter().map(c_plain).collect();
            assert_eq!(a, b, "{what}: parse");
        }
        Err(e) => assert_eq!(e.to_raw(), c_ret, "{what}: parse error"),
    }

    // parse, into a list that is too short
    if count > 0 {
        let cap = count.saturating_sub(short_by);
        let mut c_n = cap as i32;
        let c_ret = unsafe { opus_packet_extensions_parse(data.as_ptr(), len, c_exts.as_mut_ptr(), &mut c_n, nf) };
        let ret = extensions::parse(data, &mut ours[..cap], nb_frames);
        assert_eq!(ret.map_err(|e| e.to_raw()), Err(c_ret), "{what}: parse into {cap}");
    }

    // count_ext and parse_ext
    let ordered = parse_in_frame_order(data, nb_frames, &what);
    assert_eq!(ordered.is_ok(), parsed.is_ok(), "{what}: parse and parse_ext disagree");

    // The iterator, with a frame limit, find and reset. After an error
    // libopus returns it again on every call, while opus1's iterator ends:
    // `None` from `next`, `Ok(None)` from `find_id`.
    let mut c_iter: CIter = unsafe { std::mem::zeroed() };
    unsafe { opus_extension_iterator_init(&mut c_iter, data.as_ptr(), len, nf) };
    let mut iter = ExtensionIterator::new(data, nb_frames);
    let mut failed: Option<i32> = None;
    for step in 0..2 * count + 8 {
        if step == frame_max_step {
            unsafe { opus_extension_iterator_set_frame_max(&mut c_iter, frame_max as c_int) };
            iter.set_frame_max(frame_max);
        }
        if step == reset_step {
            unsafe { opus_extension_iterator_reset(&mut c_iter) };
            iter.reset();
            failed = None;
        }
        let mut c_ext = C_EXT_EMPTY;
        let (c_ret, ret) = match find_id {
            Some(id) if step % 2 == 1 => {
                (unsafe { opus_extension_iterator_find(&mut c_iter, &mut c_ext, id as c_int) }, iter.find_id(id))
            }
            _ => (unsafe { opus_extension_iterator_next(&mut c_iter, &mut c_ext) }, iter.next().transpose()),
        };
        match ret {
            Ok(Some(e)) => {
                assert_eq!(c_ret, 1, "{what}: iterator step {step}");
                assert_eq!(plain(&e), c_plain(&c_ext), "{what}: iterator step {step}");
                assert!(e.frame < nb_frames, "{what}: frame {} of {nb_frames}", e.frame);
            }
            Ok(None) => assert_eq!(c_ret, failed.unwrap_or(0), "{what}: iterator step {step}"),
            Err(e) => {
                assert_eq!(e.to_raw(), c_ret, "{what}: iterator step {step}");
                failed = Some(c_ret);
            }
        }
    }

    // Write the extensions of a valid padding again.
    if let Ok(list) = ordered {
        let need = extensions::generated_len(usize::MAX >> 2, &list, nb_frames)
            .unwrap_or_else(|e| panic!("{what}: generated_len of a parsed list: {e}"));
        compare_generate(&list, nb_frames, need + slack, pad, &what);
    }
    Ok(())
}

fn write_list(u: &mut Unstructured) -> Result<()> {
    let nb_frames = if u.ratio(1, 10)? { u.int_in_range(0..=49)? } else { u.int_in_range(1..=8)? };
    let pool_len = u.int_in_range(0..=1500usize)?.min(u.len());
    let pool = u.bytes(pool_len)?;
    let exts = arbitrary_extensions(u, pool, nb_frames, 200)?;
    let pad = u.arbitrary::<bool>()?;
    let len = match extensions::generated_len(usize::MAX >> 2, &exts, nb_frames) {
        Ok(need) => match u.int_in_range(0..=3)? {
            0 => u.int_in_range(0..=need)?,
            1 => need,
            _ => need + u.int_in_range(0..=40)?,
        },
        Err(_) => u.int_in_range(0..=2000)?,
    };
    let what = format!("{} extensions, {nb_frames} frames, {len} bytes, pad {pad}", exts.len());
    compare_generate(&exts, nb_frames, len, pad, &what);
    Ok(())
}

/// `generate` and `generated_len` into `len` bytes against libopus; a list
/// that is written must read back in frame order.
fn compare_generate(exts: &[Extension], nb_frames: usize, len: usize, pad: bool, what: &str) {
    let c = to_c(exts);
    let mut c_out = vec![0xEEu8; len];
    let c_ret = unsafe {
        opus_packet_extensions_generate(c_out.as_mut_ptr(), len as i32, c.as_ptr(), c.len() as i32, nb_frames as c_int, pad as c_int)
    };
    let c_len =
        unsafe { opus_packet_extensions_generate(std::ptr::null_mut(), len as i32, c.as_ptr(), c.len() as i32, nb_frames as c_int, 0) };
    match extensions::generated_len(len, exts, nb_frames) {
        Ok(n) => assert_eq!(n as i32, c_len, "{what}: generated_len"),
        Err(e) => assert_eq!(e.to_raw(), c_len, "{what}: generated_len error"),
    }
    let mut out = vec![0xEEu8; len];
    match extensions::generate(&mut out, exts, nb_frames, padding_of(pad)) {
        Ok(n) => {
            assert_eq!(n as i32, c_ret, "{what}: generate");
            assert!(n <= len, "{what}: wrote {n} bytes into {len}");
            if pad {
                assert_eq!(n, len, "{what}: a padded list fills the buffer");
            }
            assert_eq!(out[..n], c_out[..n], "{what}: generate bytes");
            // It reads back as the same extensions, in frame order.
            let back = parse_in_frame_order(&out[..n], nb_frames, what)
                .unwrap_or_else(|e| panic!("{what}: a generated padding does not parse: {e}"));
            let mut want: Vec<Plain> = exts.iter().map(plain).collect();
            want.sort_by_key(|e| e.1);
            assert_eq!(back.iter().map(plain).collect::<Vec<_>>(), want, "{what}: read back");
        }
        Err(e) => assert_eq!(e.to_raw(), c_ret, "{what}: generate error"),
    }
}
