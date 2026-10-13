//! Range coder tests for the buffer handling of `RangeCoder`.
//!
//! The encoder keeps its buffer between frames, so a new frame starts with
//! the bytes of the previous one. These tests fill the buffer with a marker
//! byte before encoding and require the same packet as a clean buffer gives.

use opus_rs::range_coder::RangeCoder;

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

/// Encodes `nsym` range coder symbols and `nraw` groups of raw bits, then
/// calls `done()`. Returns the whole buffer and the error flag.
fn encode_packet(rc: &mut RangeCoder, seed: u64, nsym: usize, nraw: usize) -> (Vec<u8>, i32) {
    let mut r = Rng(seed);
    for _ in 0..nsym {
        rc.encode_icdf((r.next() % 8) as i32, &ICDF8, 8);
    }
    for _ in 0..nraw {
        let bits = 1 + r.next() % 8;
        rc.enc_bits(r.next() & ((1 << bits) - 1), bits);
    }
    rc.done();
    (rc.buf[..rc.storage as usize].to_vec(), rc.error)
}

/// `done()` clears every byte between the range coder bytes and the raw bits,
/// whatever the buffer held before, and leaves all other bytes alone.
#[test]
fn done_clears_the_gap_of_a_dirty_buffer() {
    // Storage sizes around the sizes the codec uses, and symbol counts that
    // give an empty buffer, a large gap, a gap of a few bytes and no gap.
    for &storage in &[1u32, 2, 3, 7, 8, 9, 40, 160, 1275, 2048] {
        for &(nsym, nraw) in &[
            (0usize, 0usize),
            (1, 0),
            (0, 1),
            (3, 2),
            (20, 5),
            (100, 30),
            (400, 0),
            (0, 300),
            (3000, 0),
            (3000, 900),
        ] {
            let seed = 1 + storage as u64 * 131 + nsym as u64 * 7 + nraw as u64;

            let mut clean = RangeCoder::new_encoder(storage);
            let (want, want_error) = encode_packet(&mut clean, seed, nsym, nraw);

            let mut dirty = RangeCoder::new_encoder(storage);
            dirty.buf.fill(0xA5);
            let (got, got_error) = encode_packet(&mut dirty, seed, nsym, nraw);

            assert_eq!(
                got_error, want_error,
                "error flag: storage {storage}, {nsym} symbols, {nraw} raw groups"
            );
            if want_error == 0 {
                assert_eq!(
                    got, want,
                    "packet: storage {storage}, {nsym} symbols, {nraw} raw groups"
                );
                let gap = dirty.offs as usize..(dirty.storage - dirty.end_offs) as usize;
                assert!(
                    got[gap.start..gap.end.max(gap.start)]
                        .iter()
                        .enumerate()
                        // The last gap byte may hold the final raw bits.
                        .all(|(i, &b)| b == 0 || i + 1 == gap.len()),
                    "gap not cleared: storage {storage}, {nsym} symbols, {nraw} raw groups"
                );
            } else {
                // A full buffer has no gap, so `done()` must not have touched
                // the range coder bytes or the raw bits of the failed packet.
                let offs = dirty.offs as usize;
                assert_eq!(got[..offs], want[..offs]);
            }
        }
    }
}

/// A second frame in the same coder gives the packet a new coder gives.
#[test]
fn reused_encoder_gives_the_packet_of_a_new_one() {
    let mut reused = RangeCoder::new_encoder(1275);
    let _ = encode_packet(&mut reused, 11, 900, 200);
    for &(storage, nsym, nraw) in &[(1275u32, 60usize, 10usize), (200, 300, 40), (1275, 5, 0)] {
        reused.reset_for_encode(storage);
        let (got, got_error) = encode_packet(&mut reused, 77, nsym, nraw);
        let mut fresh = RangeCoder::new_encoder(storage);
        let (want, want_error) = encode_packet(&mut fresh, 77, nsym, nraw);
        assert_eq!(got_error, 0);
        assert_eq!(want_error, 0);
        assert_eq!(got, want, "storage {storage}, {nsym} symbols, {nraw} raw groups");
    }
}

/// The range encoder of libopus (celt/entenc.c), written out plainly: the
/// reference for `encoder_matches_the_entenc_model`.
///
/// One difference from the C code is kept on purpose: when the buffer is
/// full this port sets `error` to 1, where libopus sets it to -1.
struct EntencModel {
    buf: Vec<u8>,
    storage: u32,
    end_offs: u32,
    end_window: u32,
    nend_bits: i32,
    nbits_total: i32,
    offs: u32,
    rng: u32,
    val: u32,
    ext: u32,
    rem: i32,
    error: i32,
    /// Not part of the coder: how often a carry went into a waiting run of
    /// 0xFF bytes and turned it into 0x00 bytes.
    carries_into_run: u32,
}

impl EntencModel {
    fn new(storage: u32) -> Self {
        EntencModel {
            buf: vec![0; storage as usize],
            storage,
            end_offs: 0,
            end_window: 0,
            nend_bits: 0,
            nbits_total: 33,
            offs: 0,
            rng: 1 << 31,
            val: 0,
            ext: 0,
            rem: -1,
            error: 0,
            carries_into_run: 0,
        }
    }

    /// ec_write_byte
    fn write_byte(&mut self, value: u32) {
        if self.offs + self.end_offs >= self.storage {
            self.error = 1;
        } else {
            self.buf[self.offs as usize] = value as u8;
            self.offs += 1;
        }
    }

    /// ec_write_byte_at_end
    fn write_byte_at_end(&mut self, value: u32) {
        if self.offs + self.end_offs >= self.storage {
            self.error = 1;
        } else {
            self.end_offs += 1;
            self.buf[(self.storage - self.end_offs) as usize] = value as u8;
        }
    }

    /// ec_enc_carry_out
    fn carry_out(&mut self, c: i32) {
        if c != 0xFF {
            let carry = c >> 8;
            if self.rem >= 0 {
                self.write_byte((self.rem + carry) as u32);
            }
            if self.ext > 0 {
                self.carries_into_run += carry as u32;
                let sym = ((0xFF + carry) & 0xFF) as u32;
                while self.ext > 0 {
                    self.write_byte(sym);
                    self.ext -= 1;
                }
            }
            self.rem = c & 0xFF;
        } else {
            self.ext += 1;
        }
    }

    /// ec_enc_normalize
    fn normalize(&mut self) {
        while self.rng <= 1 << 23 {
            self.carry_out((self.val >> 23) as i32);
            self.val = (self.val << 8) & 0x7FFF_FFFF;
            self.rng <<= 8;
            self.nbits_total += 8;
        }
    }

    /// ec_encode
    fn encode(&mut self, fl: u32, fh: u32, ft: u32) {
        let r = self.rng / ft;
        if fl > 0 {
            self.val = self.val.wrapping_add(self.rng - r * (ft - fl));
            self.rng = r * (fh - fl);
        } else {
            self.rng -= r * (ft - fh);
        }
        self.normalize();
    }

    /// ec_enc_bit_logp
    fn encode_bit_logp(&mut self, bit: bool, logp: u32) {
        let s = self.rng >> logp;
        let r = self.rng - s;
        if bit {
            self.val = self.val.wrapping_add(r);
            self.rng = s;
        } else {
            self.rng = r;
        }
        self.normalize();
    }

    /// ec_enc_icdf
    fn encode_icdf(&mut self, s: usize, icdf: &[u8], ftb: u32) {
        let r = self.rng >> ftb;
        if s > 0 {
            self.val = self.val.wrapping_add(self.rng - r * icdf[s - 1] as u32);
            self.rng = r * (icdf[s - 1] - icdf[s]) as u32;
        } else {
            self.rng -= r * icdf[s] as u32;
        }
        self.normalize();
    }

    /// ec_enc_bits
    fn enc_bits(&mut self, value: u32, bits: u32) {
        let mut window = self.end_window;
        let mut used = self.nend_bits;
        if used as u32 + bits > 32 {
            while used >= 8 {
                self.write_byte_at_end(window & 0xFF);
                window >>= 8;
                used -= 8;
            }
        }
        window |= value << used;
        used += bits as i32;
        self.end_window = window;
        self.nend_bits = used;
        self.nbits_total += bits as i32;
    }

    fn words(&self) -> [i64; 11] {
        [
            self.storage as i64,
            self.end_offs as i64,
            self.end_window as i64,
            self.nend_bits as i64,
            self.nbits_total as i64,
            self.offs as i64,
            self.rng as i64,
            self.val as i64,
            self.ext as i64,
            self.rem as i64,
            self.error as i64,
        ]
    }
}

fn coder_words(rc: &RangeCoder) -> [i64; 11] {
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

/// After every symbol the encoder has the words of the libopus encoder
/// written out in `EntencModel`, and the same bytes at both ends of the
/// buffer.
///
/// The symbol mix has long runs of the top symbol of 256, which give 0xFF
/// bytes whose carry stays open, and symbols that resolve such a run with
/// and without a carry. Every buffer size from 1 to 40 bytes is used, and
/// raw bits take room from the other end, so the buffer becomes full at
/// every position: before a byte, in the middle of a run, and with the two
/// ends meeting. Coding goes on after the buffer is full, as the codec does.
#[test]
fn encoder_matches_the_entenc_model() {
    let mut open_carry_seen = 0;
    let mut carry_into_run_seen = 0;
    let mut full_seen = 0;
    for storage in (1..=40u32).chain([64, 160, 1275, 2048]) {
        for seed in 0..12u64 {
            let mut rc = RangeCoder::new_encoder(storage);
            // Old bytes in the buffer must not matter.
            rc.buf.fill(0xC3);
            let mut model = EntencModel::new(storage);
            let mut r = Rng(1 + storage as u64 * 1000 + seed);
            let steps = if storage <= 40 { 400 } else { 3000 };
            for step in 0..steps {
                let what = format!("storage {storage} seed {seed} step {step}");
                match r.next() % 16 {
                    // A run of the top symbol of 256: 0xFF bytes, carry open.
                    0..=2 => {
                        for _ in 0..(1 + r.next() % 6) {
                            rc.encode(255, 256, 256);
                            model.encode(255, 256, 256);
                        }
                    }
                    3..=5 => {
                        let s = (r.next() % 8) as usize;
                        rc.encode_icdf(s as i32, &ICDF8, 8);
                        model.encode_icdf(s, &ICDF8, 8);
                    }
                    6..=8 => {
                        let logp = 1 + r.next() % 15;
                        let bit = r.next() % 4 == 0;
                        rc.encode_bit_logp(bit, logp);
                        model.encode_bit_logp(bit, logp);
                    }
                    9..=12 => {
                        let ft = 2 + r.next() % 60000;
                        let fl = r.next() % ft;
                        let fh = fl + 1 + r.next() % (ft - fl);
                        rc.encode(fl, fh, ft);
                        model.encode(fl, fh, ft);
                    }
                    // The top symbol of a wide alphabet: the largest step of
                    // `val`, which is what carries into a waiting run.
                    13 => {
                        rc.encode(40000, 40001, 40001);
                        model.encode(40000, 40001, 40001);
                    }
                    _ => {
                        let bits = 1 + r.next() % 8;
                        let value = r.next() & ((1 << bits) - 1);
                        rc.enc_bits(value, bits);
                        model.enc_bits(value, bits);
                    }
                }
                assert_eq!(coder_words(&rc), model.words(), "{what}");
                let offs = model.offs as usize;
                assert_eq!(rc.buf[..offs], model.buf[..offs], "{what}");
                let raw = (model.storage - model.end_offs) as usize;
                assert_eq!(rc.buf[raw..], model.buf[raw..], "{what}");

                if model.ext > 0 {
                    open_carry_seen += 1;
                }
                if model.error != 0 {
                    full_seen += 1;
                }
            }
            carry_into_run_seen += model.carries_into_run;
        }
    }
    // The cases this test exists for did occur.
    assert!(open_carry_seen > 1000, "open carry in {open_carry_seen} steps");
    assert!(carry_into_run_seen > 20, "carry into a run {carry_into_run_seen} times");
    assert!(full_seen > 1000, "full buffer in {full_seen} steps");
}
