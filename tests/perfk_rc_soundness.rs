//! The range coder with arguments and public words that a correct caller
//! never produces. The coder must then panic or set its error flag. It must
//! not read or write outside its buffer or outside the caller's table.
//!
//! `storage`, `offs`, `end_offs` and the buffer are public fields, so safe
//! code can set them to anything. The encoder used to write its bytes with
//! unchecked indexing below `storage`, and `encode_icdf` read the caller's
//! table with an unchecked symbol.

use opus1::range_coder::{BITRES, RANGE_BUF_MAX, RangeCoder};

/// Small deterministic pseudo-random generator.
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
}

const ICDF4: [u8; 4] = [200, 120, 40, 0];

/// Codes far more symbols and raw bits than fit into `storage` bytes.
fn overfill(rc: &mut RangeCoder, rng: &mut Lcg) {
    for _ in 0..40_000 {
        match rng.next() % 4 {
            0 => rc.encode_icdf((rng.next() % 4) as i32, &ICDF4, 8),
            1 => rc.encode_bit_logp(rng.next() % 2 == 0, 1 + rng.next() % 3),
            2 => rc.enc_bits(rng.next() & 0xFFF, 12),
            _ => {
                let fl = rng.next() % 100;
                rc.encode(fl, fl + 1, 100);
            }
        }
    }
}

/// A coder whose `storage` says more than its buffer holds writes nothing
/// outside the buffer. It reports an error instead.
#[test]
fn encoder_with_storage_beyond_its_buffer_sets_the_error_flag() {
    let mut rng = Lcg(1);
    for (buf_len, storage) in [
        (16u32, 17u32),
        (16, 4000),
        (1, RANGE_BUF_MAX as u32 + 1),
        (300, 100_000),
        (RANGE_BUF_MAX as u32, u32::MAX / 2),
    ] {
        let mut rc = RangeCoder::new_encoder(buf_len);
        assert_eq!(rc.buf.len(), buf_len as usize);
        rc.storage = storage;
        overfill(&mut rc, &mut rng);
        assert_ne!(rc.error, 0, "buffer {buf_len}, storage {storage}");
        assert_eq!(rc.buf.len(), buf_len as usize);
        // Range coder bytes stop at the end of the buffer.
        assert!(rc.offs as usize <= rc.buf.len(), "offs {}", rc.offs);
    }
}

/// The same with a correct `storage`: the coder fills its buffer, sets the
/// error flag and keeps `offs + end_offs` within `storage`.
#[test]
fn full_encoder_sets_the_error_flag() {
    let mut rng = Lcg(2);
    for size in [1u32, 2, 7, 64, 1275, RANGE_BUF_MAX as u32] {
        let mut rc = RangeCoder::new_encoder(size);
        overfill(&mut rc, &mut rng);
        assert_ne!(rc.error, 0, "size {size}");
        assert!(rc.offs + rc.end_offs <= rc.storage, "size {size}");
        rc.done();
        assert_eq!(rc.buf.len(), size as usize);
    }
}

/// A buffer that safe code made shorter than `storage`.
#[test]
fn encoder_with_a_truncated_buffer_sets_the_error_flag() {
    let mut rng = Lcg(3);
    let mut rc = RangeCoder::new_encoder(200);
    rc.buf.truncate(5);
    overfill(&mut rc, &mut rng);
    assert_ne!(rc.error, 0);
    assert_eq!(rc.buf.len(), 5);
}

#[test]
#[should_panic]
fn encode_icdf_panics_on_a_symbol_past_the_table() {
    let mut rc = RangeCoder::new_encoder(64);
    rc.encode_icdf(4, &ICDF4, 8);
}

#[test]
#[should_panic]
fn encode_icdf_panics_on_a_negative_symbol() {
    let mut rc = RangeCoder::new_encoder(64);
    rc.encode_icdf(-1, &ICDF4, 8);
}

/// Every symbol of the table still codes and decodes.
#[test]
fn encode_icdf_round_trip_for_every_symbol() {
    let mut rng = Lcg(4);
    let symbols: Vec<i32> = (0..500).map(|i| if i < 4 { i } else { (rng.next() % 4) as i32 }).collect();
    let mut rc = RangeCoder::new_encoder(1000);
    for &s in &symbols {
        rc.encode_icdf(s, &ICDF4, 8);
    }
    rc.done();
    assert_eq!(rc.error, 0);
    let mut dec = RangeCoder::new_decoder(&rc.buf[..rc.offs as usize]);
    for &s in &symbols {
        assert_eq!(dec.decode_icdf(&ICDF4, 8), s);
    }
}

/// `tell_frac_inline!` gives the value of `tell_frac` for every range a
/// coder can have between two symbols (above 2^23).
#[test]
fn tell_frac_inline_matches_tell_frac() {
    let mut rng = Lcg(5);
    let mut rc = RangeCoder::new_encoder(64);
    let mut ranges: Vec<u32> = vec![(1 << 23) + 1, (1 << 24) - 1, 1 << 24, 1 << 31, u32::MAX];
    for shift in 0..8 {
        // Around each threshold of the correction table.
        for c in [35733u32, 38967, 42495, 46340, 50535, 55109, 60097, 65535] {
            for d in [0u32, 1] {
                ranges.push(((c + d) << 8 >> shift).max((1 << 23) + 1));
            }
        }
    }
    for _ in 0..20_000 {
        ranges.push(rng.next().max((1 << 23) + 1));
    }
    for rng_value in ranges {
        rc.rng = rng_value;
        rc.nbits_total = 33 + (rng.next() % 10_000) as i32;
        let inline = opus1::tell_frac_inline!(rc);
        assert_eq!(inline, rc.tell_frac(), "rng {rng_value:#x}");
    }
}

/// A range that no coder has between symbols (zero, or below 2^15) must not
/// make `tell_frac_inline!` read outside its table. The value itself has no
/// meaning; release builds must return and debug builds may panic on the
/// shift.
#[test]
fn tell_frac_inline_stays_inside_its_table_for_any_range() {
    let mut rc = RangeCoder::new_encoder(64);
    for rng_value in [0u32, 1, 255, 1 << 15, (1 << 16) - 1, 1 << 16, 1 << 22] {
        rc.rng = rng_value;
        let result = std::panic::catch_unwind(|| {
            let rc = rc.clone();
            opus1::tell_frac_inline!(rc)
        });
        if cfg!(not(debug_assertions)) {
            assert!(result.is_ok(), "rng {rng_value:#x}");
        }
    }
}
