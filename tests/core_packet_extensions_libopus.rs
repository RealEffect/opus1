//! Packet extensions (`opus1::extensions`) and the packet functions that
//! carry them (`opus1::packet`) against libopus 1.6.1.
//!
//! The tests link libopus as a static library, which keeps the names of its
//! internal functions, so they call libopus's own extension code
//! (`src/extensions.c`): the iterator, `opus_packet_extensions_count`,
//! `_count_ext`, `_parse`, `_parse_ext` and `_generate`, and
//! `opus_packet_parse_impl`, `opus_repacketizer_out_range_impl` and
//! `opus_packet_pad_impl`. Every result must be the same: the return value
//! or error, every extension (ID, frame, payload) and every byte written.
//!
//! The inputs are random paddings (biased towards the bytes that mean
//! something: separators, repeats, short and long IDs, small lengths),
//! random lists of extensions (with repeatable patterns across frames and
//! some invalid entries), and packets that carry them: written by opus1's
//! encoder and by `packet::write`, and padded with extensions or with
//! random bytes.

use std::num::NonZeroU32;
use audiopus_sys as sys;
use opus1::SampleRate;
use opus1::extensions::{self, Extension, ExtensionIterator};
use opus1::packet::{self, Framing, Padding, Repacketizer};
use std::os::raw::{c_int, c_uchar};

/// Without the feature `alloc`, opus1 refuses with `AllocFail` a call that
/// has to keep more than 32 extensions or to copy more than 1275 bytes of
/// old padding, where libopus allocates (docs/10, "Packet padding
/// extensions"). Such a call is not compared.
fn allocation_refused(e: opus1::Error) -> bool {
    cfg!(not(feature = "alloc")) && e.kind() == opus1::ErrorKind::AllocFail
}

/// libopus `opus_extension_data` (src/opus_private.h).
#[repr(C)]
#[derive(Clone, Copy)]
struct CExt {
    id: c_int,
    frame: c_int,
    data: *const c_uchar,
    len: i32,
}

const EMPTY: CExt = CExt { id: 0, frame: 0, data: std::ptr::null(), len: 0 };

/// libopus `OpusExtensionIterator` (src/opus_private.h).
#[repr(C)]
struct CIter {
    data: *const c_uchar,
    curr_data: *const c_uchar,
    repeat_data: *const c_uchar,
    last_long: *const c_uchar,
    src_data: *const c_uchar,
    len: i32,
    curr_len: i32,
    repeat_len: i32,
    src_len: i32,
    trailing_short_len: i32,
    nb_frames: c_int,
    frame_max: c_int,
    curr_frame: c_int,
    repeat_frame: c_int,
    repeat_l: c_uchar,
}

unsafe extern "C" {
    fn opus_extension_iterator_init(iter: *mut CIter, data: *const c_uchar, len: i32, nb_frames: i32);
    fn opus_extension_iterator_reset(iter: *mut CIter);
    fn opus_extension_iterator_set_frame_max(iter: *mut CIter, frame_max: c_int);
    fn opus_extension_iterator_next(iter: *mut CIter, ext: *mut CExt) -> c_int;
    fn opus_extension_iterator_find(iter: *mut CIter, ext: *mut CExt, id: c_int) -> c_int;
    fn opus_packet_extensions_count(data: *const c_uchar, len: i32, nb_frames: c_int) -> i32;
    fn opus_packet_extensions_count_ext(data: *const c_uchar, len: i32, nb_frame_exts: *mut i32, nb_frames: c_int) -> i32;
    fn opus_packet_extensions_parse(data: *const c_uchar, len: i32, exts: *mut CExt, nb: *mut i32, nb_frames: c_int) -> i32;
    fn opus_packet_extensions_parse_ext(
        data: *const c_uchar,
        len: i32,
        exts: *mut CExt,
        nb: *mut i32,
        nb_frame_exts: *const i32,
        nb_frames: c_int,
    ) -> i32;
    fn opus_packet_extensions_generate(
        data: *mut c_uchar,
        len: i32,
        exts: *const CExt,
        nb: i32,
        nb_frames: c_int,
        pad: c_int,
    ) -> i32;
    fn opus_packet_parse_impl(
        data: *const c_uchar,
        len: i32,
        self_delimited: c_int,
        out_toc: *mut c_uchar,
        frames: *mut *const c_uchar,
        size: *mut i16,
        payload_offset: *mut c_int,
        packet_offset: *mut i32,
        padding: *mut *const c_uchar,
        padding_len: *mut i32,
    ) -> c_int;
    fn opus_repacketizer_out_range_impl(
        rp: *mut sys::OpusRepacketizer,
        begin: c_int,
        end: c_int,
        data: *mut c_uchar,
        maxlen: i32,
        self_delimited: c_int,
        pad: c_int,
        exts: *const CExt,
        nb: c_int,
    ) -> i32;
    fn opus_packet_pad_impl(data: *mut c_uchar, len: i32, new_len: i32, pad: c_int, exts: *const CExt, nb: c_int) -> i32;
}

/// What `packet::pad_with_extensions` does, in libopus terms:
/// `opus_packet_pad_impl`, with two changes when `len == new_len`.
/// Without extensions the result is `len`, not 0. With extensions the
/// packet is rewritten as libopus rewrites it for a longer `new_len`,
/// instead of being left as it is without them.
///
/// # Safety
/// `data` must hold at least `len` and `new_len` bytes.
unsafe fn c_pad_with_extensions(data: &mut [u8], len: usize, new_len: usize, pad: bool, exts: &[CExt]) -> i32 {
    unsafe {
        if len < 1 || len != new_len {
            return opus_packet_pad_impl(data.as_mut_ptr(), len as i32, new_len as i32, pad as c_int, exts.as_ptr(), exts.len() as c_int);
        }
        if exts.is_empty() {
            return len as i32;
        }
        // The rest of `opus_packet_pad_impl`.
        let copy = data[..len].to_vec();
        let rp = sys::opus_repacketizer_create();
        let mut ret = sys::opus_repacketizer_cat(rp, copy.as_ptr(), len as i32);
        if ret == 0 {
            let nb = sys::opus_repacketizer_get_nb_frames(rp);
            ret = opus_repacketizer_out_range_impl(rp, 0, nb, data.as_mut_ptr(), new_len as i32, 0, pad as c_int, exts.as_ptr(), exts.len() as c_int);
        }
        sys::opus_repacketizer_destroy(rp);
        ret
    }
}

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
    fn chance(&mut self, percent: usize) -> bool {
        self.below(100) < percent
    }
    fn bytes(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| self.next() as u8).collect()
    }
}

/// An extension as plain values: ID, frame, payload.
type Plain = (u8, usize, Vec<u8>);

fn plain(e: &Extension) -> Plain {
    (e.id, e.frame, e.data.to_vec())
}

fn c_plain(e: &CExt) -> Plain {
    let data = if e.len > 0 { unsafe { std::slice::from_raw_parts(e.data, e.len as usize) }.to_vec() } else { Vec::new() };
    (e.id as u8, e.frame as usize, data)
}

fn to_c(exts: &[Extension]) -> Vec<CExt> {
    exts.iter()
        .map(|e| CExt { id: e.id as c_int, frame: e.frame as c_int, data: e.data.as_ptr(), len: e.data.len() as i32 })
        .collect()
}

// ---------------------------------------------------------------------------
// Random inputs
// ---------------------------------------------------------------------------

/// A random padding: entries that mean something, with now and then a
/// random byte.
fn random_padding(rng: &mut Rng, nb_frames: usize) -> Vec<u8> {
    let mut v = Vec::new();
    let entries = rng.below(12);
    for _ in 0..entries {
        match rng.below(10) {
            0 => v.push(rng.next() as u8),
            1 => v.push(rng.below(2) as u8),       // padding
            2 => v.push(0x02),                     // next frame
            3 => {
                v.push(0x03);                      // frame increment
                v.push(rng.below(nb_frames.max(1) + 1) as u8);
            }
            4 => v.push(0x04 | rng.below(2) as u8), // repeat
            5 | 6 => {
                // Short extension, L = 0 or 1.
                let l = rng.below(2) as u8;
                v.push(((3 + rng.below(29)) as u8) << 1 | l);
                if l == 1 {
                    v.push(rng.next() as u8);
                }
            }
            _ => {
                // Long extension, L = 0 or 1, with a length that may run past
                // the end.
                let l = rng.below(4) != 0;
                v.push(((32 + rng.below(96)) as u8) << 1 | l as u8);
                let n = if rng.chance(5) { 255 + rng.below(300) } else { rng.below(8) };
                if l {
                    let mut m = n;
                    while m >= 255 {
                        v.push(255);
                        m -= 255;
                    }
                    v.push(m as u8);
                }
                let room = if rng.chance(10) { n.saturating_sub(1) } else { n };
                v.extend(rng.bytes(room));
            }
        }
    }
    v
}

/// A random list of extensions for `nb_frames` frames. Some lists repeat a
/// pattern in every frame, so that `generate` uses repeats. With `invalid`,
/// an entry may be invalid (ID, frame or payload length).
fn random_extensions<'a>(rng: &mut Rng, pool: &'a [u8], nb_frames: usize, invalid: bool) -> Vec<Extension<'a>> {
    let payload = |rng: &mut Rng, id: u8| -> &'a [u8] {
        let n = if id < 32 {
            rng.below(2)
        } else if rng.chance(5) {
            254 + rng.below(260)
        } else {
            rng.below(6)
        };
        let start = rng.below(pool.len() - n);
        &pool[start..start + n]
    };
    let mut exts = Vec::new();
    if nb_frames == 0 {
        return exts;
    }
    if rng.chance(50) {
        // A pattern in every frame, with the same short lengths.
        let pattern: Vec<u8> = (0..1 + rng.below(3))
            .map(|_| if rng.chance(50) { 3 + rng.below(29) as u8 } else { 32 + rng.below(96) as u8 })
            .collect();
        let short_len: Vec<usize> = pattern.iter().map(|_| rng.below(2)).collect();
        for f in 0..nb_frames {
            for (k, &id) in pattern.iter().enumerate() {
                let data = if id < 32 { &pool[k..k + short_len[k]] } else { payload(rng, id) };
                exts.push(Extension::new(id, f, data));
            }
            if rng.chance(20) {
                let id = 3 + rng.below(125) as u8;
                exts.push(Extension::new(id, f, payload(rng, id)));
            }
        }
        // Out of frame order now and then.
        if rng.chance(20) && exts.len() > 1 {
            let a = rng.below(exts.len());
            let b = rng.below(exts.len());
            exts.swap(a, b);
        }
    } else {
        for _ in 0..rng.below(8) {
            let id = 3 + rng.below(125) as u8;
            let frame = rng.below(nb_frames);
            exts.push(Extension::new(id, frame, payload(rng, id)));
        }
        if rng.chance(50) {
            exts.sort_by_key(|e| e.frame);
        }
    }
    if invalid && !exts.is_empty() && rng.chance(30) {
        let k = rng.below(exts.len());
        match rng.below(3) {
            0 => exts[k].id = [0, 1, 2, 128, 200][rng.below(5)],
            1 => exts[k].frame = nb_frames + rng.below(3),
            _ => {
                exts[k].id = 3 + rng.below(29) as u8;
                exts[k].data = &pool[..2 + rng.below(3)];
            }
        }
    }
    exts
}

// ---------------------------------------------------------------------------
// Reading extensions
// ---------------------------------------------------------------------------

/// opus1's framing for libopus's `self_delimited` flag.
fn framing_of(self_delimited: bool) -> Framing {
    if self_delimited { Framing::SelfDelimited } else { Framing::Normal }
}

/// opus1's padding for libopus's `pad` flag.
fn padding_of(pad: bool) -> Padding {
    if pad { Padding::FillBuffer } else { Padding::Shortest }
}

/// All the readers of libopus and opus1 on one padding.
fn compare_reading(data: &[u8], nb_frames: usize, rng: &mut Rng, what: &str) {
    let len = data.len() as i32;
    let nf = nb_frames as c_int;
    // count
    let c_count = unsafe { opus_packet_extensions_count(data.as_ptr(), len, nf) };
    assert_eq!(extensions::count(data, nb_frames), Ok(c_count as usize), "{what}: count");
    // parse
    let mut c_exts = vec![EMPTY; 3000];
    let mut c_n = c_exts.len() as i32;
    let c_ret = unsafe { opus_packet_extensions_parse(data.as_ptr(), len, c_exts.as_mut_ptr(), &mut c_n, nf) };
    let mut ours = vec![Extension::default(); 3000];
    let ret = extensions::parse(data, &mut ours, nb_frames);
    match ret {
        Ok(n) => {
            assert_eq!(c_ret, 0, "{what}: parse result");
            assert_eq!(n, c_n as usize, "{what}: parse count");
            let a: Vec<Plain> = ours[..n].iter().map(plain).collect();
            let b: Vec<Plain> = c_exts[..n].iter().map(c_plain).collect();
            assert_eq!(a, b, "{what}: parse");
        }
        Err(e) if allocation_refused(e) => {}
        Err(e) => assert_eq!(e.to_raw(), c_ret, "{what}: parse error"),
    }
    // parse into a list that is too short
    if c_count > 0 {
        let cap = rng.below(c_count as usize);
        let mut c_n = cap as i32;
        let c_ret = unsafe { opus_packet_extensions_parse(data.as_ptr(), len, c_exts.as_mut_ptr(), &mut c_n, nf) };
        let ret = extensions::parse(data, &mut ours[..cap], nb_frames);
        assert_eq!(ret.map_err(|e| e.to_raw()), Err(c_ret), "{what}: parse into {cap}");
    }
    // count_ext and parse_ext
    let mut c_counts = [0i32; 48];
    let c_total = unsafe { opus_packet_extensions_count_ext(data.as_ptr(), len, c_counts.as_mut_ptr(), nf) };
    let mut counts = [0usize; 48];
    let total = extensions::count_ext(data, &mut counts, nb_frames);
    assert_eq!(total, Ok(c_total as usize), "{what}: count_ext");
    for f in 0..nb_frames {
        assert_eq!(counts[f], c_counts[f] as usize, "{what}: count_ext of frame {f}");
    }
    if c_ret == 0 {
        let mut c_n = c_exts.len() as i32;
        let c_ret = unsafe {
            opus_packet_extensions_parse_ext(data.as_ptr(), len, c_exts.as_mut_ptr(), &mut c_n, c_counts.as_ptr(), nf)
        };
        let n = extensions::parse_ext(data, &mut ours, &counts, nb_frames).unwrap_or_else(|e| panic!("{what}: {e}"));
        assert_eq!((c_ret, c_n as usize), (0, n), "{what}: parse_ext");
        let a: Vec<Plain> = ours[..n].iter().map(plain).collect();
        let b: Vec<Plain> = c_exts[..n].iter().map(c_plain).collect();
        assert_eq!(a, b, "{what}: parse_ext");
        assert!(a.windows(2).all(|w| w[0].1 <= w[1].1), "{what}: frame order");
    }
    // The iterator with a frame limit, find, and reset. After an error
    // libopus returns it again on every call, while opus1's iterator ends:
    // `None` from `next`, `Ok(None)` from `find_id`.
    let frame_max = rng.below(nb_frames + 2);
    let find_id = if rng.chance(50) { 3 + rng.below(125) as u8 } else { 0 };
    let mut c_iter: CIter = unsafe { std::mem::zeroed() };
    unsafe { opus_extension_iterator_init(&mut c_iter, data.as_ptr(), len, nf) };
    let mut iter = ExtensionIterator::new(data, nb_frames);
    let mut failed: Option<i32> = None;
    for step in 0..2 * c_count as usize + 8 {
        if step == 3 {
            unsafe { opus_extension_iterator_set_frame_max(&mut c_iter, frame_max as c_int) };
            iter.set_frame_max(frame_max);
        }
        if step == 6 {
            unsafe { opus_extension_iterator_reset(&mut c_iter) };
            iter.reset();
            failed = None;
        }
        let mut c_ext = EMPTY;
        let (c_ret, ret) = if find_id != 0 && step % 2 == 1 {
            (unsafe { opus_extension_iterator_find(&mut c_iter, &mut c_ext, find_id as c_int) }, iter.find_id(find_id))
        } else {
            (unsafe { opus_extension_iterator_next(&mut c_iter, &mut c_ext) }, iter.next().transpose())
        };
        match ret {
            Ok(Some(e)) => {
                assert_eq!(c_ret, 1, "{what}: step {step}");
                assert_eq!(plain(&e), c_plain(&c_ext), "{what}: step {step}");
            }
            Ok(None) => assert_eq!(c_ret, failed.unwrap_or(0), "{what}: step {step}"),
            Err(e) => {
                assert_eq!(e.to_raw(), c_ret, "{what}: step {step}");
                failed = Some(c_ret);
            }
        }
    }
}

#[test]
fn random_paddings_read_like_libopus() {
    let mut rng = Rng(0x5EED_0001);
    let mut found = 0;
    let mut invalid = 0;
    for case in 0..30_000 {
        let nb_frames = if rng.chance(20) { rng.below(49) } else { 1 + rng.below(6) };
        let data = random_padding(&mut rng, nb_frames);
        compare_reading(&data, nb_frames, &mut rng, &format!("case {case}"));
        found += extensions::count(&data, nb_frames).unwrap();
        invalid += extensions::parse(&data, &mut [Extension::default(); 4000], nb_frames).is_err() as usize;
    }
    // The inputs reach both valid and invalid paddings.
    assert!(found > 30_000, "{found} extensions");
    assert!(invalid > 3_000 && invalid < 27_000, "{invalid} invalid paddings");
}

#[test]
fn edge_paddings_read_like_libopus() {
    let mut rng = Rng(7);
    let cases: &[(&[u8], usize)] = &[
        (&[], 1),
        (&[], 0),
        (&[0x06], 0),
        (&[0x00, 0x07], 1),
        (&[0x01, 0x01, 0x07, 0x55], 1),
        (&[0x02], 1),
        (&[0x02], 2),
        (&[0x03, 0x00, 0x06], 1),
        (&[0x03, 0x02, 0x06], 2),
        (&[0x03], 2),
        (&[0x06, 0x04], 3),
        (&[0x06, 0x05], 3),
        (&[0x41, 0x02, 1, 2, 0x04, 3, 4, 5, 6], 3),
        (&[0x41, 0x02, 1, 2, 0x05, 0x02, 3, 4, 0x02, 5, 6], 3),
        (&[0x41, 0x01, 9, 0x07, 1, 0x04, 0x02, 8, 2], 3),
        (&[0x40, 1, 2, 3], 1),
        (&[0x41, 255, 0, 7], 1),
        (&[0x41, 255], 1),
        (&[0x07], 1),
        (&[0x06, 0x02, 0x06, 0x02, 0x06, 0x02], 3),
        (&[0x06, 0x02, 0x06, 0x02, 0x06, 0x02], 4),
    ];
    for (k, &(data, nb_frames)) in cases.iter().enumerate() {
        compare_reading(data, nb_frames, &mut rng, &format!("edge case {k}"));
    }
    // Long payloads with lengths of several bytes.
    for n in [253usize, 254, 255, 256, 509, 510, 511, 1000] {
        let mut data = vec![0x41];
        let mut m = n;
        while m >= 255 {
            data.push(255);
            m -= 255;
        }
        data.push(m as u8);
        data.extend(rng.bytes(n));
        data.push(0x07);
        data.push(0x42);
        compare_reading(&data, 1, &mut rng, &format!("length {n}"));
        compare_reading(&data[..data.len() - 3], 1, &mut rng, &format!("length {n}, cut"));
    }
}

// ---------------------------------------------------------------------------
// Writing extensions
// ---------------------------------------------------------------------------

#[test]
fn generate_writes_the_bytes_of_libopus() {
    let mut rng = Rng(0x5EED_0002);
    let pool = rng.bytes(4096);
    let mut repeats = 0;
    for case in 0..20_000 {
        let nb_frames = if rng.chance(10) { rng.below(50) } else { 1 + rng.below(8) };
        let exts = random_extensions(&mut rng, &pool, nb_frames, true);
        let c = to_c(&exts);
        let full: usize = exts.iter().map(|e| 3 + e.data.len() + e.data.len() / 255).sum::<usize>() + 2 * nb_frames;
        let len = match rng.below(4) {
            0 => rng.below(full + 1),
            _ => full + rng.below(20),
        };
        let pad = rng.chance(30);
        let what = format!("case {case}: {} extensions, {nb_frames} frames, {len} bytes, pad {pad}", exts.len());
        let mut c_out = vec![0xEEu8; len];
        let c_ret = unsafe {
            opus_packet_extensions_generate(c_out.as_mut_ptr(), len as i32, c.as_ptr(), c.len() as i32, nb_frames as c_int, pad as c_int)
        };
        let mut out = vec![0xEEu8; len];
        let ret = extensions::generate(&mut out, &exts, nb_frames, padding_of(pad));
        let c_len = unsafe {
            opus_packet_extensions_generate(std::ptr::null_mut(), len as i32, c.as_ptr(), c.len() as i32, nb_frames as c_int, 0)
        };
        let ret_len = extensions::generated_len(len, &exts, nb_frames);
        match ret_len {
            Ok(n) => assert_eq!(n as i32, c_len, "{what}: length"),
            Err(e) if allocation_refused(e) => {}
            Err(e) => assert_eq!(e.to_raw(), c_len, "{what}: length error"),
        }
        match ret {
            Ok(n) => {
                assert_eq!(n as i32, c_ret, "{what}");
                assert_eq!(out[..n], c_out[..n], "{what}: bytes");
                // It reads back as the same extensions, in frame order.
                let mut back = vec![Extension::default(); exts.len()];
                let mut counts = [0usize; 48];
                extensions::count_ext(&out[..n], &mut counts, nb_frames).unwrap();
                let m = extensions::parse_ext(&out[..n], &mut back, &counts, nb_frames).unwrap();
                let mut want: Vec<Plain> = exts.iter().map(plain).collect();
                want.sort_by_key(|e| e.1);
                assert_eq!(back[..m].iter().map(plain).collect::<Vec<_>>(), want, "{what}: read back");
                repeats += out[..n].iter().any(|&b| b == 0x04 || b == 0x05) as usize;
            }
            Err(e) if allocation_refused(e) => {}
            Err(e) => assert_eq!(e.to_raw(), c_ret, "{what}: error"),
        }
    }
    assert!(repeats > 1000, "{repeats} lists used a repeat");
}

// ---------------------------------------------------------------------------
// Packets
// ---------------------------------------------------------------------------

/// A random packet of `count` frames with the TOC `toc`, written by
/// `packet::write`, whose padding is `padding` (code 3).
fn packet_with_padding(rng: &mut Rng, toc: u8, count: usize, padding: &[u8]) -> Vec<u8> {
    let frames: Vec<Vec<u8>> = (0..count).map(|_| {
        let n = if rng.chance(10) { 252 + rng.below(30) } else { rng.below(40) };
        rng.bytes(n)
    }).collect();
    let slices: Vec<&[u8]> = frames.iter().map(|f| f.as_slice()).collect();
    let mut out = vec![0u8; 16000];
    let n = packet::write(toc, &slices, &mut out, Framing::Normal, Padding::Shortest).unwrap();
    out.truncate(n);
    if padding.is_empty() && rng.chance(50) {
        return out;
    }
    // Pad to the padding's length plus its length bytes, then put it in.
    let target = n + 2 + padding.len() + padding.len() / 254 + 1;
    let mut buf = vec![0u8; target + 10];
    buf[..n].copy_from_slice(&out);
    let mut new_len = target;
    // Find the length whose zero padding is exactly `padding.len()` bytes.
    loop {
        let mut tmp = buf.clone();
        packet::write(toc, &slices, &mut tmp[..new_len], Framing::Normal, Padding::FillBuffer).unwrap();
        let p = packet::parse(&tmp[..new_len], Framing::Normal).unwrap();
        if p.padding_len() == padding.len() {
            let start = new_len - padding.len();
            tmp[start..new_len].copy_from_slice(padding);
            tmp.truncate(new_len);
            return tmp;
        }
        if p.padding_len() < padding.len() {
            new_len += 1;
        } else {
            new_len -= 1;
        }
    }
}

/// libopus's `opus_packet_parse_impl` padding: offset and length.
fn c_padding(data: &[u8]) -> Option<(usize, usize)> {
    let mut toc = 0u8;
    let mut frames = [std::ptr::null(); 48];
    let mut sizes = [0i16; 48];
    let mut padding: *const c_uchar = std::ptr::null();
    let mut padding_len = 0i32;
    let ret = unsafe {
        opus_packet_parse_impl(
            data.as_ptr(),
            data.len() as i32,
            0,
            &mut toc,
            frames.as_mut_ptr(),
            sizes.as_mut_ptr(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            &mut padding,
            &mut padding_len,
        )
    };
    (ret > 0).then(|| (if padding_len > 0 { padding as usize - data.as_ptr() as usize } else { 0 }, padding_len as usize))
}

/// Random packets: the padding of `opus_packet_parse_impl` and
/// `ParsedPacket::padding` agree, and so do pad and unpad, with and
/// without extensions passed in.
#[test]
fn pad_and_unpad_carry_extensions_like_libopus() {
    let mut rng = Rng(0x5EED_0003);
    let pool = rng.bytes(4096);
    let tocs = [0xE0u8, 0x4C, 0x08, 0xF8];
    let mut kept = 0;
    for case in 0..6_000 {
        let toc = tocs[rng.below(tocs.len())];
        let max = 48 / (packet::samples_per_frame(toc, SampleRate::Hz48000) / 120);
        let count = 1 + rng.below(max.min(6));
        let padding = match rng.below(4) {
            0 => Vec::new(),
            1 => random_padding(&mut rng, count),
            _ => {
                let exts = random_extensions(&mut rng, &pool, count, false);
                let mut buf = vec![0u8; 4000];
                let n = extensions::generate(&mut buf, &exts, count, Padding::Shortest).unwrap();
                buf.truncate(n);
                if rng.chance(30) {
                    buf.insert(0, 0x01);
                }
                buf
            }
        };
        let data = packet_with_padding(&mut rng, toc, count, &padding);
        let what = format!("case {case}: {count} frames, padding {}", padding.len());
        let parsed = packet::parse(&data, Framing::Normal).unwrap();
        let (off, plen) = c_padding(&data).unwrap();
        assert_eq!(parsed.padding_len(), plen, "{what}");
        if plen > 0 {
            assert_eq!(parsed.packet_offset() - parsed.padding_len(), off, "{what}");
        }
        assert_eq!(parsed.padding(&data), &padding[..], "{what}");

        // opus_packet_pad and opus_packet_pad_impl with extra extensions.
        // One packet in eight keeps its length, where
        // `pad_with_extensions` differs from libopus
        // (`c_pad_with_extensions`).
        let new_len = if rng.chance(12) { data.len() } else { data.len() + rng.below(3000) };
        let extra = if rng.chance(50) { random_extensions(&mut rng, &pool, count, true) } else { Vec::new() };
        let c_extra = to_c(&extra);
        for (pad_flag, with_extra) in [(true, false), (true, true), (false, true)] {
            let exts: &[Extension] = if with_extra { &extra } else { &[] };
            let mut c_buf = data.clone();
            c_buf.resize(new_len + 8, 0xEE);
            let mut buf = c_buf.clone();
            let c_ret = unsafe {
                if with_extra {
                    c_pad_with_extensions(&mut c_buf, data.len(), new_len, pad_flag, &c_extra)
                } else {
                    sys::opus_packet_pad(c_buf.as_mut_ptr(), data.len() as i32, new_len as i32)
                }
            };
            let ret = if with_extra {
                packet::pad_with_extensions(&mut buf, data.len(), new_len, padding_of(pad_flag), exts)
            } else {
                packet::pad(&mut buf, data.len(), new_len).map(|_| 0)
            };
            let w = format!("{what}: pad {pad_flag}, {} extra", exts.len());
            match ret {
                Ok(n) => {
                    assert!(c_ret >= 0, "{w}: libopus gives {c_ret}");
                    if with_extra {
                        assert_eq!(n as i32, c_ret, "{w}");
                    }
                    let n = if with_extra { n } else { new_len };
                    assert_eq!(buf[..n], c_buf[..n], "{w}: bytes");
                    kept += extensions::count(packet::parse(&buf[..n], Framing::Normal).unwrap().padding(&buf[..n]), count).unwrap();
                }
                Err(e) if allocation_refused(e) => {}
                Err(e) => assert_eq!(e.to_raw(), c_ret, "{w}: error"),
            }
        }

        // opus_packet_unpad
        let mut c_buf = data.clone();
        let c_ret = unsafe { sys::opus_packet_unpad(c_buf.as_mut_ptr(), data.len() as i32) };
        let mut buf = data.clone();
        let n = packet::unpad(&mut buf, data.len()).unwrap();
        assert_eq!(n as i32, c_ret, "{what}: unpad");
        assert_eq!(buf[..n], c_buf[..n], "{what}: unpad bytes");
    }
    assert!(kept > 5_000, "{kept} extensions kept");
}

/// The repacketizer on random packets with extensions: `cat`, then
/// `out_range_ext` on random ranges with every option, against
/// `opus_repacketizer_cat` and `opus_repacketizer_out_range_impl`.
#[test]
fn repacketizer_carries_extensions_like_libopus() {
    let mut rng = Rng(0x5EED_0004);
    let pool = rng.bytes(4096);
    let mut carried = 0;
    for case in 0..3_000 {
        let toc = [0xE0u8, 0x4C, 0xC8][rng.below(3)];
        let per = packet::samples_per_frame(toc, SampleRate::Hz48000);
        let mut packets = Vec::new();
        let mut total = 0;
        while packets.len() < 6 {
            let count = 1 + rng.below(3);
            if (total + count) * per > 5760 && !packets.is_empty() {
                break;
            }
            let padding = if rng.chance(70) {
                let exts = random_extensions(&mut rng, &pool, count, false);
                let mut buf = vec![0u8; 4000];
                let n = extensions::generate(&mut buf, &exts, count, Padding::Shortest).unwrap();
                buf.truncate(n);
                buf
            } else if rng.chance(10) {
                random_padding(&mut rng, count)
            } else {
                Vec::new()
            };
            // Now and then a packet of another configuration, or too long.
            let t = if rng.chance(3) { toc ^ 0x08 } else { toc };
            packets.push(packet_with_padding(&mut rng, t, count, &padding));
            total += count;
        }
        let what = format!("case {case}");
        let c_rp = unsafe { sys::opus_repacketizer_create() };
        let mut rp = Repacketizer::new();
        for (k, p) in packets.iter().enumerate() {
            let c_ret = unsafe { sys::opus_repacketizer_cat(c_rp, p.as_ptr(), p.len() as i32) };
            let ret = rp.cat(p);
            match ret {
                Ok(()) => assert_eq!(c_ret, 0, "{what}: cat {k}"),
                Err(e) if allocation_refused(e) => {}
                Err(e) => assert_eq!(e.to_raw(), c_ret, "{what}: cat {k}"),
            }
        }
        let nb = unsafe { sys::opus_repacketizer_get_nb_frames(c_rp) } as usize;
        assert_eq!(rp.nb_frames(), nb, "{what}");
        for _ in 0..6 {
            let begin = rng.below(nb + 1);
            let end = begin + rng.below(nb + 2 - begin);
            let self_delimited = rng.chance(30);
            let pad = rng.chance(30);
            let extra = if rng.chance(30) { random_extensions(&mut rng, &pool, end.saturating_sub(begin), true) } else { Vec::new() };
            let c_extra = to_c(&extra);
            let maxlen = if rng.chance(20) { rng.below(400) } else { 2000 + rng.below(8000) };
            let mut c_out = vec![0xEEu8; maxlen];
            let mut out = vec![0xEEu8; maxlen];
            let c_ret = unsafe {
                opus_repacketizer_out_range_impl(
                    c_rp,
                    begin as c_int,
                    end as c_int,
                    c_out.as_mut_ptr(),
                    maxlen as i32,
                    self_delimited as c_int,
                    pad as c_int,
                    c_extra.as_ptr(),
                    c_extra.len() as c_int,
                )
            };
            let ret = rp.out_range_ext(begin, end, &mut out, framing_of(self_delimited), padding_of(pad), &extra);
            let w = format!("{what}: frames {begin}..{end} of {nb}, sd {self_delimited}, pad {pad}, {} extra, {maxlen} bytes", extra.len());
            match ret {
                Ok(n) => {
                    assert_eq!(n as i32, c_ret, "{w}");
                    assert_eq!(out[..n], c_out[..n], "{w}: bytes");
                    let p = packet::parse(&out[..n], framing_of(self_delimited)).unwrap();
                    carried += extensions::count(p.padding(&out[..n]), p.nb_frames()).unwrap();
                }
                Err(e) if allocation_refused(e) => {}
                Err(e) => assert_eq!(e.to_raw(), c_ret, "{w}: error"),
            }
        }
        // The public calls without options.
        if nb > 0 {
            let mut c_out = vec![0u8; 20000];
            let mut out = vec![0u8; 20000];
            let c_ret = unsafe { sys::opus_repacketizer_out(c_rp, c_out.as_mut_ptr(), c_out.len() as i32) };
            match rp.out(&mut out) {
                Ok(n) => {
                    assert_eq!(n as i32, c_ret, "{what}: out");
                    assert_eq!(out[..n], c_out[..n], "{what}: out bytes");
                }
                Err(e) if allocation_refused(e) => {}
                Err(e) => assert_eq!(e.to_raw(), c_ret, "{what}: out error"),
            }
        }
        unsafe { sys::opus_repacketizer_destroy(c_rp) };
    }
    assert!(carried > 3_000, "{carried} extensions carried");
}

/// Multistream packets whose streams have extensions: pad and unpad.
#[test]
fn multistream_pad_and_unpad_like_libopus() {
    let mut rng = Rng(0x5EED_0005);
    let pool = rng.bytes(4096);
    let mut kept = 0;
    for case in 0..2_000 {
        let streams = 1 + rng.below(4);
        let mut data = Vec::new();
        for s in 0..streams {
            let count = 1 + rng.below(3);
            let exts = if rng.chance(60) { random_extensions(&mut rng, &pool, count, false) } else { Vec::new() };
            let mut padding = vec![0u8; 4000];
            let n = extensions::generate(&mut padding, &exts, count, Padding::Shortest).unwrap();
            padding.truncate(n);
            let single = packet_with_padding(&mut rng, 0xE0, count, &padding);
            if s + 1 < streams {
                // Self-delimited, with the same padding.
                let mut rp = Repacketizer::new();
                rp.cat(&single).unwrap();
                let mut out = vec![0u8; single.len() + 8];
                let n = rp.out_range_ext(0, count, &mut out, Framing::SelfDelimited, Padding::Shortest, &[]).unwrap();
                data.extend_from_slice(&out[..n]);
            } else {
                data.extend_from_slice(&single);
            }
        }
        let what = format!("case {case}: {streams} streams");
        let new_len = data.len() + rng.below(600);
        let mut c_buf = data.clone();
        c_buf.resize(new_len, 0);
        let mut buf = c_buf.clone();
        let c_ret = unsafe { sys::opus_multistream_packet_pad(c_buf.as_mut_ptr(), data.len() as i32, new_len as i32, streams as i32) };
        match packet::multistream_pad(&mut buf, data.len(), new_len, streams) {
            Ok(()) => {
                assert_eq!(c_ret, 0, "{what}: pad");
                assert_eq!(buf, c_buf, "{what}: pad bytes");
                kept += 1;
            }
            Err(e) if allocation_refused(e) => {}
            Err(e) => assert_eq!(e.to_raw(), c_ret, "{what}: pad error"),
        }
        let mut c_buf = data.clone();
        let mut buf = data.clone();
        let c_ret = unsafe { sys::opus_multistream_packet_unpad(c_buf.as_mut_ptr(), data.len() as i32, streams as i32) };
        match packet::multistream_unpad(&mut buf, data.len(), streams) {
            Ok(n) => {
                assert_eq!(n as i32, c_ret, "{what}: unpad");
                assert_eq!(buf[..n], c_buf[..n], "{what}: unpad bytes");
            }
            Err(e) if allocation_refused(e) => {}
            Err(e) => assert_eq!(e.to_raw(), c_ret, "{what}: unpad error"),
        }
    }
    assert!(kept > 1_900, "{kept} packets padded");
}

/// Malformed packets give libopus's errors in pad, unpad and the
/// repacketizer, and leave nothing half-done where libopus fails first.
#[test]
fn malformed_packets_give_the_errors_of_libopus() {
    let mut rng = Rng(0x5EED_0006);
    for case in 0..20_000 {
        let len = rng.below(40);
        let mut data = rng.bytes(len);
        if !data.is_empty() && rng.chance(50) {
            data[0] |= 3;
        }
        let what = format!("case {case}: {data:02x?}");
        // pad
        let new_len = len + rng.below(30);
        let mut c_buf = data.clone();
        c_buf.resize(new_len.max(1), 0);
        let mut buf = c_buf.clone();
        let c_ret = unsafe { sys::opus_packet_pad(c_buf.as_mut_ptr(), len as i32, new_len as i32) };
        match packet::pad(&mut buf, len, new_len) {
            Ok(()) => {
                assert_eq!(c_ret, 0, "{what}: pad");
                assert_eq!(buf[..new_len], c_buf[..new_len], "{what}: pad bytes");
            }
            Err(e) if allocation_refused(e) => {}
            Err(e) => assert_eq!(e.to_raw(), c_ret, "{what}: pad error"),
        }
        // unpad
        if len > 0 {
            let mut c_buf = data.clone();
            let mut buf = data.clone();
            let c_ret = unsafe { sys::opus_packet_unpad(c_buf.as_mut_ptr(), len as i32) };
            match packet::unpad(&mut buf, len) {
                Ok(n) => {
                    assert_eq!(n as i32, c_ret, "{what}: unpad");
                    assert_eq!(buf[..n], c_buf[..n], "{what}: unpad bytes");
                }
                Err(e) if allocation_refused(e) => {}
                Err(e) => assert_eq!(e.to_raw(), c_ret, "{what}: unpad error"),
            }
        }
        // cat and out
        if len > 0 {
            let c_rp = unsafe { sys::opus_repacketizer_create() };
            let mut rp = Repacketizer::new();
            let c_ret = unsafe { sys::opus_repacketizer_cat(c_rp, data.as_ptr(), len as i32) };
            match rp.cat(&data) {
                Ok(()) => {
                    assert_eq!(c_ret, 0, "{what}: cat");
                    let mut c_out = vec![0u8; 100];
                    let mut out = vec![0u8; 100];
                    let c_ret = unsafe { sys::opus_repacketizer_out(c_rp, c_out.as_mut_ptr(), 100) };
                    match rp.out(&mut out) {
                        Ok(n) => {
                            assert_eq!(n as i32, c_ret, "{what}: out");
                            assert_eq!(out[..n], c_out[..n], "{what}: out bytes");
                        }
                        Err(e) if allocation_refused(e) => {}
                        Err(e) => assert_eq!(e.to_raw(), c_ret, "{what}: out error"),
                    }
                }
                Err(e) if allocation_refused(e) => {}
                Err(e) => assert_eq!(e.to_raw(), c_ret, "{what}: cat error"),
            }
            unsafe { sys::opus_repacketizer_destroy(c_rp) };
        }
    }
}

/// Real packets of opus1's encoder (CBR, so they are padded) with
/// extensions added: libopus and opus1 decode them as the packets without
/// extensions, and the repacketizer gives libopus's bytes.
#[test]
#[cfg_attr(not(target_arch = "aarch64"), ignore = "bit-exact with libopus only on 64-bit ARM (docs/10, Platforms)")]
fn encoded_packets_with_extensions_decode_as_without() {
    use opus1::{Application, Bitrate, Channels, OpusDecoder, OpusEncoder, SampleRate};
    let mut enc = OpusEncoder::new(SampleRate::Hz48000, Channels::Mono, Application::Voip).unwrap();
    enc.set_bitrate(Bitrate::BitsPerSecond(NonZeroU32::new(24000).unwrap()));
    enc.set_vbr(false);
    let mut dec = OpusDecoder::new(SampleRate::Hz48000, Channels::Mono).unwrap();
    let mut dec_plain = OpusDecoder::new(SampleRate::Hz48000, Channels::Mono).unwrap();
    let mut err = 0;
    let c_dec = unsafe { sys::opus_decoder_create(48000, 1, &mut err) };
    assert_eq!(err, 0);
    let mut rng = Rng(9);
    let payload = rng.bytes(300);
    for n in 0..50 {
        let pcm: Vec<f32> = (0..960).map(|i| (((n * 960 + i) as f32) * 0.03).sin() * 0.3).collect();
        let mut buf = vec![0u8; 1500];
        let len = enc.encode(&pcm, 960, &mut buf[..200]).unwrap();
        let plain = buf[..len].to_vec();
        let exts = [
            Extension::new(5, 0, &payload[..1]),
            Extension::new(126, 0, &payload[..(n * 7) % 300]),
        ];
        let c_exts = to_c(&exts);
        let mut c_buf = buf.clone();
        let target = len + 320;
        let c_ret = unsafe { opus_packet_pad_impl(c_buf.as_mut_ptr(), len as i32, target as i32, 0, c_exts.as_ptr(), 2) };
        let m = packet::pad_with_extensions(&mut buf, len, target, Padding::Shortest, &exts).unwrap();
        assert_eq!(m as i32, c_ret, "packet {n}");
        assert_eq!(buf[..m], c_buf[..m], "packet {n}");
        let p = packet::parse(&buf[..m], Framing::Normal).unwrap();
        let mut found = [Extension::default(); 4];
        assert_eq!(extensions::parse(p.padding(&buf[..m]), &mut found, p.nb_frames()), Ok(2));
        assert_eq!(&found[..2], &exts[..]);
        // Decoding ignores the extensions.
        let mut out = vec![0f32; 960];
        let mut out_plain = vec![0f32; 960];
        let mut c_out = vec![0f32; 960];
        assert_eq!(dec.decode(&buf[..m], 960, &mut out).unwrap(), 960);
        assert_eq!(dec_plain.decode(&plain, 960, &mut out_plain).unwrap(), 960);
        let c_n = unsafe { sys::opus_decode_float(c_dec, buf.as_ptr(), m as i32, c_out.as_mut_ptr(), 960, 0) };
        assert_eq!(c_n, 960);
        assert_eq!(out, out_plain, "packet {n}");
        assert_eq!(out, c_out, "packet {n}");
    }
    unsafe { sys::opus_decoder_destroy(c_dec) };
}
