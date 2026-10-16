//! Tests for copying a `RangeCoder`: `save_state` / `restore_state`,
//! `clone` and `new_decoder`.
//!
//! None of them copies the whole 2048-byte buffer, so each test checks that
//! the copy still behaves like the original.

use opus1::internals::range_coder::{RANGE_BUF_MAX, RangeCoder};

/// Small deterministic generator (xorshift64*), so the tests need no crate.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        (self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 32) as u32
    }
}

const ICDF8: [u8; 8] = [224, 192, 160, 128, 96, 64, 32, 0];

/// The words of a coder, for comparing two coders.
fn words(rc: &RangeCoder) -> [i64; 11] {
    [
        rc.storage as i64,
        rc.end_offs as i64,
        rc.end_window as i64,
        rc.nend_bits as i64,
        rc.nbits_total as i64,
        rc.offs as i64,
        rc.rng as i64,
        rc.val as i64,
        rc.ext as i64,
        rc.rem as i64,
        rc.error as i64,
    ]
}

/// Encodes `n` range coder symbols drawn from `r`.
fn encode_symbols(rc: &mut RangeCoder, r: &mut Rng, n: usize) {
    for _ in 0..n {
        match r.next() % 4 {
            0 => rc.encode_icdf((r.next() % 8) as i32, &ICDF8, 8),
            1 => rc.encode_bit_logp(r.next() % 8 == 0, 3),
            2 => {
                let ft = 2 + r.next() % 1000;
                let fl = r.next() % ft;
                rc.encode(fl, fl + 1, ft);
            }
            // Symbol 255 of 256 gives 0xFF bytes whose carry stays open.
            _ => rc.encode(255, 256, 256),
        }
    }
}

/// Going back to a saved state and coding the same symbols again gives the
/// same coder; going back and coding other symbols gives what a coder gives
/// that never coded the first ones.
#[test]
fn restore_state_returns_to_the_saved_point() {
    for &storage in &[4u32, 40, 160, 1275, 2048] {
        for &(before, tried, after) in &[
            (0usize, 0usize, 0usize),
            (0, 1, 1),
            (0, 30, 5),
            (7, 3, 40),
            (25, 60, 60),
            (200, 200, 10),
            (900, 500, 500),
        ] {
            let what = format!("storage {storage}, {before} + {tried} / {after} symbols");
            let seed = 3 + storage as u64 * 17 + before as u64 * 5 + tried as u64;

            // The coder under test: code `before`, save, code `tried`,
            // restore, code `after`.
            let mut rc = RangeCoder::new_encoder(storage);
            rc.buf.fill(0xA5);
            let mut r = Rng(seed);
            encode_symbols(&mut rc, &mut r, before);
            rc.enc_bits(0x15, 5);
            let saved = rc.save_state();
            let saved_words = words(&rc);
            let mut r_tried = Rng(seed ^ 0xFFFF);
            encode_symbols(&mut rc, &mut r_tried, tried);
            rc.restore_state(&saved);
            assert_eq!(words(&rc), saved_words, "{what}");
            let mut r_after = Rng(seed ^ 0x7777);
            encode_symbols(&mut rc, &mut r_after, after);

            // The reference: a coder that never coded the `tried` symbols.
            let mut want = RangeCoder::new_encoder(storage);
            let mut r = Rng(seed);
            encode_symbols(&mut want, &mut r, before);
            want.enc_bits(0x15, 5);
            let mut r_after = Rng(seed ^ 0x7777);
            encode_symbols(&mut want, &mut r_after, after);

            assert_eq!(words(&rc), words(&want), "{what}");
            let offs = want.offs as usize;
            assert_eq!(rc.buf[..offs], want.buf[..offs], "{what}");

            rc.done();
            want.done();
            assert_eq!(rc.error, want.error, "{what}");
            if want.error == 0 {
                let s = want.storage as usize;
                assert_eq!(rc.buf[..s], want.buf[..s], "{what}");
            }
        }
    }
}

/// A later state can be put back too, when the bytes written between the two
/// states are put back with it. This is what the coarse energy coder does
/// with its intra pass.
#[test]
fn restore_state_with_saved_bytes_returns_to_a_later_point() {
    for &storage in &[40u32, 160, 1275] {
        for &(before, first, second) in &[(0usize, 5usize, 5usize), (10, 40, 3), (30, 3, 60)] {
            let what = format!("storage {storage}, {before} + {first} / {second} symbols");
            let seed = 9 + storage as u64 + before as u64 * 3 + first as u64;

            let mut rc = RangeCoder::new_encoder(storage);
            let mut r = Rng(seed);
            encode_symbols(&mut rc, &mut r, before);
            let start = rc.save_state();
            let start_offs = rc.offs as usize;

            // First attempt; keep its words and bytes.
            let mut r1 = Rng(seed ^ 0x1111);
            encode_symbols(&mut rc, &mut r1, first);
            let first_state = rc.save_state();
            let first_words = words(&rc);
            let first_offs = rc.offs as usize;
            let first_bytes = rc.buf[start_offs..first_offs].to_vec();
            let first_prefix = rc.buf[..first_offs].to_vec();

            // Second attempt from the same start; it overwrites the bytes.
            rc.restore_state(&start);
            let mut r2 = Rng(seed ^ 0x2222);
            encode_symbols(&mut rc, &mut r2, second);

            // Back to the first attempt.
            rc.restore_state(&first_state);
            rc.buf[start_offs..first_offs].copy_from_slice(&first_bytes);
            assert_eq!(words(&rc), first_words, "{what}");
            assert_eq!(rc.buf[..first_offs], first_prefix[..], "{what}");

            // The packet is the one of a coder that only made the first
            // attempt.
            let mut want = RangeCoder::new_encoder(storage);
            let mut r = Rng(seed);
            encode_symbols(&mut want, &mut r, before);
            let mut r1 = Rng(seed ^ 0x1111);
            encode_symbols(&mut want, &mut r1, first);
            rc.done();
            want.done();
            assert_eq!(rc.error, want.error, "{what}");
            if want.error == 0 {
                let s = want.storage as usize;
                assert_eq!(rc.buf[..s], want.buf[..s], "{what}");
            }
        }
    }
}

/// A clone has the same words and the same `buf.len()` bytes, and goes on
/// like the original, both as an encoder and as a decoder.
#[test]
fn clone_behaves_like_the_original() {
    for &storage in &[1u32, 2, 17, 160, 1275, 2048] {
        for &nsym in &[0usize, 1, 50, 400] {
            let what = format!("storage {storage}, {nsym} symbols");
            let seed = 21 + storage as u64 * 3 + nsym as u64;

            let mut rc = RangeCoder::new_encoder(storage);
            rc.buf.fill(0x3C);
            let mut r = Rng(seed);
            encode_symbols(&mut rc, &mut r, nsym);
            rc.enc_bits(0x5, 3);

            let mut copy = rc.clone();
            assert_eq!(words(&copy), words(&rc), "{what}");
            assert_eq!(copy.buf.len(), rc.buf.len(), "{what}");
            assert_eq!(copy.buf[..], rc.buf[..], "{what}");

            // Both go on with the same symbols and give the same packet.
            let mut r_a = Rng(seed ^ 0xABCD);
            let mut r_b = Rng(seed ^ 0xABCD);
            encode_symbols(&mut rc, &mut r_a, 20);
            encode_symbols(&mut copy, &mut r_b, 20);
            rc.done();
            copy.done();
            assert_eq!(words(&copy), words(&rc), "{what}");
            assert_eq!(copy.buf[..], rc.buf[..], "{what}");

            // A decoder cloned in the middle of the packet reads the same
            // symbols and raw bits as the original.
            if rc.error == 0 {
                let mut dec = RangeCoder::new_decoder(&rc.buf[..storage as usize]);
                for _ in 0..nsym.min(10) {
                    let _ = dec.decode_icdf(&ICDF8, 8);
                }
                let mut dec_copy = dec.clone();
                assert_eq!(words(&dec_copy), words(&dec), "{what}");
                assert_eq!(dec_copy.buf[..], dec.buf[..], "{what}");
                for _ in 0..40 {
                    assert_eq!(dec.decode_icdf(&ICDF8, 8), dec_copy.decode_icdf(&ICDF8, 8));
                    assert_eq!(dec.dec_bits(3), dec_copy.dec_bits(3));
                }
                assert_eq!(words(&dec_copy), words(&dec), "{what}");
            }
        }
    }
}

/// `new_decoder` copies the packet (at most `RANGE_BUF_MAX` bytes) and starts
/// with the words of libopus's `ec_dec_init`, worked out here step by step.
#[test]
fn new_decoder_matches_ec_dec_init() {
    let mut r = Rng(0xDEC0DE);
    let big: Vec<u8> = (0..RANGE_BUF_MAX + 300).map(|_| r.next() as u8).collect();
    for &len in &[
        0usize,
        1,
        2,
        3,
        4,
        5,
        15,
        16,
        17,
        63,
        64,
        65,
        255,
        1275,
        RANGE_BUF_MAX - 1,
        RANGE_BUF_MAX,
        RANGE_BUF_MAX + 1,
        RANGE_BUF_MAX + 300,
    ] {
        for first in [None, Some(0x00u8), Some(0x7F), Some(0x80), Some(0xFF)] {
            let mut data = big[..len].to_vec();
            if let (Some(b), true) = (first, len > 0) {
                data[0] = b;
            }
            let dec = RangeCoder::new_decoder(&data);
            let n = len.min(RANGE_BUF_MAX);
            assert_eq!(dec.buf.len(), n, "len {len}");
            assert_eq!(dec.buf[..], data[..n], "len {len}");
            assert_eq!(dec.storage as usize, n, "len {len}");

            // ec_dec_init: rng = 128, rem = first byte, val = 127 - (rem >> 1),
            // then three normalisation steps of one byte each. Bytes past the
            // end of the packet read as 0.
            let byte = |i: usize| if i < n { data[i] as u32 } else { 0 };
            let mut rem = byte(0);
            let mut rng = 128u32;
            let mut val = rng - 1 - (rem >> 1);
            let mut nbits_total = 9i32;
            let mut offs = 1usize;
            while rng <= 1 << 23 {
                nbits_total += 8;
                rng <<= 8;
                let sym = rem;
                rem = byte(offs);
                offs += 1;
                let sym = ((sym << 8) | rem) >> 1;
                val = ((val << 8) + (0xFF & !sym)) & 0x7FFF_FFFF;
            }
            assert_eq!(rng, 1 << 31);
            assert_eq!(dec.rng, rng, "len {len}");
            assert_eq!(dec.val, val, "len {len}");
            assert_eq!(dec.rem as u32, rem, "len {len}");
            assert_eq!(dec.nbits_total, nbits_total, "len {len}");
            assert_eq!(dec.offs as usize, offs.min(n), "len {len}");
            assert_eq!(dec.end_offs, 0);
            assert_eq!(dec.end_window, 0);
            assert_eq!(dec.nend_bits, 0);
            assert_eq!(dec.ext, 0);
            assert_eq!(dec.error, 0);
        }
    }
}

/// What the encoder writes, the decoder built by `new_decoder` reads back,
/// for packets from one byte to the largest buffer.
#[test]
fn new_decoder_reads_what_the_encoder_wrote() {
    for &storage in &[1u32, 2, 3, 8, 40, 160, 1275, 2048] {
        let mut enc = RangeCoder::new_encoder(storage);
        let mut r = Rng(storage as u64 + 5);
        // Stay well inside the buffer: 3 bits per symbol, 3 raw bits.
        let nsym = (storage as usize * 8 / 3).saturating_sub(12) / 2;
        let syms: Vec<i32> = (0..nsym).map(|_| (r.next() % 8) as i32).collect();
        for &s in &syms {
            enc.encode_icdf(s, &ICDF8, 8);
            enc.enc_bits(s as u32 ^ 5, 3);
        }
        enc.done();
        assert_eq!(enc.error, 0, "storage {storage}");

        let mut dec = RangeCoder::new_decoder(&enc.buf[..storage as usize]);
        for &s in &syms {
            assert_eq!(dec.decode_icdf(&ICDF8, 8), s, "storage {storage}");
            assert_eq!(dec.dec_bits(3), s as u32 ^ 5, "storage {storage}");
        }
        assert_eq!(dec.error, 0);
    }
}
