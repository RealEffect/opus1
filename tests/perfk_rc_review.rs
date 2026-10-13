//! Second set of tests for the range coder changes, written for the review
//! of that work. Each test checks one changed function against something
//! that does not share code with it:
//!
//! * `done()` against a model of libopus's `ec_enc_done`, on buffers that
//!   hold random bytes of earlier packets.
//! * `save_state` / `restore_state`: every word is saved and put back.
//! * The coarse energy coder against the decoder: the decoder must read from
//!   the finished packet exactly the energies the encoder kept, whichever of
//!   the two passes won.
//! * `new_decoder` against a model of libopus's range decoder, for every
//!   packet length from 0 to 300 bytes and around the largest buffer.

use opus_rs::modes::default_mode;
use opus_rs::quant_bands::{quant_coarse_energy_advanced, unquant_coarse_energy};
use opus_rs::range_coder::{RANGE_BUF_MAX, RangeCoder};

/// SplitMix64.
struct SplitMix(u64);

impl SplitMix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform in `0..n`.
    fn below(&mut self, n: u32) -> u32 {
        ((self.next() >> 32) * n as u64 >> 32) as u32
    }
    /// Uniform in [-1, 1).
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u32 << 23) as f32 - 1.0
    }
}

const ICDF5: [u8; 5] = [200, 150, 90, 20, 0];

/// The words of a coder in one array, in the order of the struct.
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

fn set_words(rc: &mut RangeCoder, w: [i64; 11]) {
    rc.storage = w[0] as u32;
    rc.end_offs = w[1] as u32;
    rc.end_window = w[2] as u32;
    rc.nend_bits = w[3] as i32;
    rc.nbits_total = w[4] as i32;
    rc.offs = w[5] as u32;
    rc.rng = w[6] as u32;
    rc.val = w[7] as u32;
    rc.ext = w[8] as u32;
    rc.rem = w[9] as i32;
    rc.error = w[10] as i32;
}

// ---------------------------------------------------------------------------
// done() against ec_enc_done
// ---------------------------------------------------------------------------

/// The range encoder of libopus (celt/entenc.c) with `ec_enc_done`.
///
/// Its buffer starts with the bytes the coder under test starts with. The
/// error values are the ones of this port: 1 when a byte does not fit
/// (libopus: -1), -1 where `ec_enc_done` sets -1.
struct EncModel {
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
}

impl EncModel {
    fn new(buf: Vec<u8>) -> Self {
        EncModel {
            storage: buf.len() as u32,
            buf,
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
        }
    }

    fn write_byte(&mut self, value: u32) {
        if self.offs + self.end_offs >= self.storage {
            self.error = 1;
        } else {
            self.buf[self.offs as usize] = value as u8;
            self.offs += 1;
        }
    }

    fn write_byte_at_end(&mut self, value: u32) {
        if self.offs + self.end_offs >= self.storage {
            self.error = 1;
        } else {
            self.end_offs += 1;
            self.buf[(self.storage - self.end_offs) as usize] = value as u8;
        }
    }

    fn carry_out(&mut self, c: i32) {
        if c != 0xFF {
            let carry = c >> 8;
            if self.rem >= 0 {
                self.write_byte((self.rem + carry) as u32);
            }
            while self.ext > 0 {
                self.write_byte(((0xFF + carry) & 0xFF) as u32);
                self.ext -= 1;
            }
            self.rem = c & 0xFF;
        } else {
            self.ext += 1;
        }
    }

    fn normalize(&mut self) {
        while self.rng <= 1 << 23 {
            self.carry_out((self.val >> 23) as i32);
            self.val = (self.val << 8) & 0x7FFF_FFFF;
            self.rng <<= 8;
            self.nbits_total += 8;
        }
    }

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

    /// ec_enc_done, statement by statement.
    fn done(&mut self) {
        let mut l = self.rng.leading_zeros() as i32;
        let mut msk = 0x7FFF_FFFFu32 >> l;
        let mut end = self.val.wrapping_add(msk) & !msk;
        if (end | msk) >= self.val.wrapping_add(self.rng) {
            l += 1;
            msk >>= 1;
            end = self.val.wrapping_add(msk) & !msk;
        }
        while l > 0 {
            self.carry_out((end >> 23) as i32);
            end = (end << 8) & 0x7FFF_FFFF;
            l -= 8;
        }
        if self.rem >= 0 || self.ext > 0 {
            self.carry_out(0);
        }
        let mut window = self.end_window;
        let mut used = self.nend_bits;
        while used >= 8 {
            self.write_byte_at_end(window & 0xFF);
            window >>= 8;
            used -= 8;
        }
        if self.error == 0 {
            // OPUS_CLEAR(buf + offs, storage - offs - end_offs), one byte at
            // a time.
            let mut i = self.offs;
            while i < self.storage - self.end_offs {
                self.buf[i as usize] = 0;
                i += 1;
            }
            if used > 0 {
                if self.end_offs >= self.storage {
                    self.error = -1;
                } else {
                    let l = -l;
                    if self.offs + self.end_offs >= self.storage && l < used {
                        window &= (1 << l) - 1;
                        self.error = -1;
                    }
                    self.buf[(self.storage - self.end_offs - 1) as usize] |= window as u8;
                }
            }
        }
    }
}

/// `done()` leaves the buffer `ec_enc_done` leaves: the same bytes at both
/// ends, zeros in between, and the last raw bits in the right byte. When the
/// packet did not fit, nothing is cleared and the buffer is still the same
/// as the model's.
///
/// One coder is used for all packets, as the codec does, so a packet starts
/// on the bytes of earlier ones; in every second round the whole buffer is
/// filled with random bytes first. Buffer sizes cover 1 to 48 bytes one by
/// one and the sizes around the largest buffers, in rising and in falling
/// order. The number of symbols goes from none to more than fit, so `done()`
/// runs with an empty buffer, with a large gap, with the two ends touching,
/// and after the buffer overflowed.
#[test]
fn done_matches_a_model_of_ec_enc_done() {
    let mut rng = SplitMix(0x0D15_EA5E);
    let mut rc = RangeCoder::new_encoder(RANGE_BUF_MAX as u32);
    for b in rc.buf.iter_mut() {
        *b = rng.next() as u8;
    }
    let sizes: Vec<u32> = (1..=48)
        .chain([97, 255, 256, 1000, 1274, 1275, 2047, 2048])
        .collect();
    let (mut fitted, mut overflowed, mut with_gap, mut partial_byte) = (0, 0, 0, 0);

    for round in 0..40 {
        for k in 0..sizes.len() {
            let storage = if round % 4 < 2 {
                sizes[k]
            } else {
                sizes[sizes.len() - 1 - k]
            };
            let what = format!("storage {storage} round {round}");
            rc.reset_for_encode(storage);
            if round % 2 == 1 {
                for b in rc.buf.iter_mut() {
                    *b = rng.next() as u8;
                }
            }
            let mut model = EncModel::new(rc.buf[..storage as usize].to_vec());

            // Between no symbol and about 1.5 times what fits.
            let capacity_bits = storage * 8;
            let target_bits = rng.below(capacity_bits * 3 / 2 + 2);
            let raw_share = rng.below(5);
            while (model.nbits_total as u32) < target_bits + 33 {
                if rng.below(4) < raw_share {
                    let bits = 1 + rng.below(13);
                    let value = rng.next() as u32 & ((1 << bits) - 1);
                    rc.enc_bits(value, bits);
                    model.enc_bits(value, bits);
                } else {
                    // Mostly the top symbol of a small alphabet, sometimes
                    // any symbol of a large one. The first gives runs of
                    // 0xFF bytes with an open carry.
                    let (fl, fh, ft) = if rng.below(3) == 0 {
                        let ft = 2 + rng.below(40000);
                        let fl = rng.below(ft);
                        (fl, fl + 1 + rng.below(ft - fl), ft)
                    } else {
                        let ft = 2 + rng.below(300);
                        (ft - 1, ft, ft)
                    };
                    rc.encode(fl, fh, ft);
                    model.encode(fl, fh, ft);
                }
            }
            assert_eq!(
                words(&rc)[..],
                model_words(&model)[..],
                "before done: {what}"
            );

            rc.done();
            model.done();

            // `done()` does not store the raw-bit window back, and neither
            // does libopus, so those two words are as before.
            assert_eq!(
                words(&rc)[..],
                model_words(&model)[..],
                "after done: {what}"
            );
            assert_eq!(rc.buf[..storage as usize], model.buf[..], "buffer: {what}");

            if model.error == 0 {
                fitted += 1;
                if model.offs + model.end_offs + 4 < storage {
                    with_gap += 1;
                }
                if model.nend_bits % 8 != 0 {
                    partial_byte += 1;
                }
            } else {
                overflowed += 1;
            }
        }
    }
    // The cases this test is for did occur.
    assert!(fitted > 500, "{fitted} packets fitted");
    assert!(overflowed > 300, "{overflowed} packets overflowed");
    assert!(with_gap > 300, "{with_gap} packets had a gap to clear");
    assert!(
        partial_byte > 200,
        "{partial_byte} packets ended in a partial raw byte"
    );
}

fn model_words(m: &EncModel) -> [i64; 11] {
    [
        m.storage as i64,
        m.end_offs as i64,
        m.end_window as i64,
        m.nend_bits as i64,
        m.nbits_total as i64,
        m.offs as i64,
        m.rng as i64,
        m.val as i64,
        m.ext as i64,
        m.rem as i64,
        m.error as i64,
    ]
}

// ---------------------------------------------------------------------------
// save_state / restore_state
// ---------------------------------------------------------------------------

/// Each of the eleven words is saved and put back, and the buffer is left
/// alone. The words are set directly, so the test does not depend on which
/// words an encoder happens to change.
#[test]
fn restore_state_puts_back_every_word() {
    let saved_words: [i64; 11] = [
        211,
        19,
        0xDEAD_BEEF,
        23,
        4321,
        57,
        0x8765_4321,
        0x1234_5678,
        9,
        0xAB,
        1,
    ];
    let other_words: [i64; 11] = [199, 5, 0x0BAD_F00D, 7, 99, 3, 0x8000_0001, 77, 1, -1, -1];

    let mut rc = RangeCoder::new_encoder(300);
    for (i, b) in rc.buf.iter_mut().enumerate() {
        *b = (i * 7 + 1) as u8;
    }
    set_words(&mut rc, saved_words);
    let saved = rc.save_state();
    assert_eq!(words(&rc), saved_words, "save_state changed the coder");

    // Change one word at a time, then all of them.
    for changed in 0..=11 {
        let mut w = saved_words;
        for (k, word) in w.iter_mut().enumerate() {
            if changed == k || changed == 11 {
                *word = other_words[k];
            }
        }
        set_words(&mut rc, w);
        for b in rc.buf.iter_mut() {
            *b = b.wrapping_add(1);
        }
        let buf_before: Vec<u8> = rc.buf.to_vec();

        rc.restore_state(&saved);
        assert_eq!(words(&rc), saved_words, "word {changed} not put back");
        assert_eq!(
            rc.buf[..],
            buf_before[..],
            "restore_state touched the buffer"
        );
    }

    // A saved state is a plain copy: restoring it twice is the same.
    let copy = saved;
    set_words(&mut rc, other_words);
    rc.restore_state(&copy);
    assert_eq!(words(&rc), saved_words);
}

/// A state of a coder with a larger buffer is refused. The encoder writes
/// below `storage` without an index check, so `storage` must never get
/// larger than the buffer.
#[test]
#[should_panic(expected = "larger buffer")]
fn restore_state_refuses_the_state_of_a_larger_coder() {
    let large = RangeCoder::new_encoder(100);
    let mut small = RangeCoder::new_encoder(10);
    small.restore_state(&large.save_state());
}

/// A state saved before `shrink` can be put back: the buffer keeps its
/// length when the coder is shrunk, so the saved `storage` still fits.
#[test]
fn restore_state_after_shrink_is_accepted() {
    let mut rc = RangeCoder::new_encoder(64);
    rc.encode(3, 4, 10);
    let saved = rc.save_state();
    rc.shrink(20);
    assert_eq!(rc.storage, 20);
    rc.restore_state(&saved);
    assert_eq!(rc.storage, 64);
}

// ---------------------------------------------------------------------------
// Coarse energy: encoder against decoder
// ---------------------------------------------------------------------------

/// The coarse energy coder tries an intra pass and an inter pass and keeps
/// the better one. Whichever it keeps, the packet must carry that pass: the
/// decoder reads the intra flag and the energies from the finished packet
/// and must arrive at the energies the encoder kept, bit for bit, and at the
/// same position in the packet.
///
/// A fault in putting the coder back (a word, or the bytes of the intra
/// pass) breaks this, because the decoder then reads other symbols.
///
/// The encoder is reused for all frames, so the buffer holds old bytes.
/// Symbols and raw bits are coded before and after the energies. Frames
/// run in sequences, so each frame predicts from the one before; every
/// sixth frame has unrelated energies, which is where the intra pass wins.
#[test]
fn coarse_energy_packet_decodes_to_the_energies_the_encoder_kept() {
    let mode = default_mode();
    let nb = mode.nb_ebands;
    let mut rng = SplitMix(0xC0A2_5E00);
    let mut rc = RangeCoder::new_encoder(RANGE_BUF_MAX as u32);
    for b in rc.buf.iter_mut() {
        *b = rng.next() as u8;
    }
    let (mut intra_frames, mut inter_frames, mut one_pass_frames) = (0, 0, 0);

    for channels in 1..=2usize {
        for lm in 0..4usize {
            for &(start, end) in &[(0usize, nb), (0, 13), (0, 19), (17, nb)] {
                for &storage in &[9u32, 23, 57, 111, 400, 1000] {
                    let budget = storage * 8;
                    let n = channels * nb;
                    let mut old_enc = [0f32; 42];
                    let mut old_dec = [0f32; 42];
                    let mut error = [0f32; 42];
                    let mut delayed_intra = 0f32;
                    let mut level = [0f32; 42];
                    for (i, l) in level.iter_mut().enumerate().take(n) {
                        *l = 6.0 - 0.25 * (i % nb) as f32 + 3.0 * rng.unit();
                    }

                    for frame in 0..14 {
                        let what = format!(
                            "channels {channels} lm {lm} bands {start}..{end} storage {storage} \
                             frame {frame}"
                        );
                        if frame % 6 == 5 {
                            for l in level.iter_mut().take(n) {
                                *l = 12.0 * rng.unit();
                            }
                        } else {
                            for l in level.iter_mut().take(n) {
                                *l = (*l + 0.9 * rng.unit()).clamp(-20.0, 25.0);
                            }
                        }
                        let two_pass = frame % 5 != 4;
                        let loss_rate = if frame % 3 == 2 { 25 } else { 0 };
                        let prefix_symbols = rng.below(4);
                        let prefix_raw = rng.below(3) == 0 && storage > 9;
                        let suffix_seed = rng.next();

                        // Encoder.
                        rc.reset_for_encode(storage);
                        for k in 0..prefix_symbols {
                            rc.encode_icdf(((frame + k as usize) % 5) as i32, &ICDF5, 8);
                        }
                        if prefix_raw {
                            rc.enc_bits(0x155, 9);
                        }
                        let flag_coded = rc.tell() + 3 <= budget as i32;
                        quant_coarse_energy_advanced(
                            mode,
                            start,
                            end,
                            end,
                            &level[..n],
                            &mut old_enc[..n],
                            budget,
                            &mut error[..n],
                            &mut rc,
                            channels,
                            lm,
                            storage as usize,
                            false,
                            &mut delayed_intra,
                            two_pass,
                            loss_rate,
                            false,
                        );
                        // Symbols after the energies, while they fit.
                        let mut suffix = SplitMix(suffix_seed);
                        let mut suffix_symbols = 0;
                        while suffix_symbols < 6 && rc.tell() + 12 <= budget as i32 {
                            rc.encode_icdf(suffix.below(5) as i32, &ICDF5, 8);
                            rc.enc_bits(suffix.below(8), 3);
                            suffix_symbols += 1;
                        }
                        let enc_tell = rc.tell();
                        assert!(enc_tell <= budget as i32, "{what}");
                        rc.done();
                        assert_eq!(rc.error, 0, "{what}");

                        // Decoder, on the finished packet.
                        let mut dec = RangeCoder::new_decoder(&rc.buf[..storage as usize]);
                        for k in 0..prefix_symbols {
                            let want = ((frame + k as usize) % 5) as i32;
                            assert_eq!(dec.decode_icdf(&ICDF5, 8), want, "{what}");
                        }
                        if prefix_raw {
                            assert_eq!(dec.dec_bits(9), 0x155, "{what}");
                        }
                        let intra = flag_coded && dec.decode_bit_logp(3);
                        unquant_coarse_energy(
                            mode,
                            start,
                            end,
                            &mut old_dec[..n],
                            intra,
                            &mut dec,
                            channels,
                            lm,
                        );
                        let mut suffix = SplitMix(suffix_seed);
                        for _ in 0..suffix_symbols {
                            assert_eq!(
                                dec.decode_icdf(&ICDF5, 8),
                                suffix.below(5) as i32,
                                "{what}"
                            );
                            assert_eq!(dec.dec_bits(3), suffix.below(8), "{what}");
                        }
                        assert_eq!(dec.tell(), enc_tell, "{what}");
                        assert_eq!(dec.error, 0, "{what}");

                        // Bands outside `start..end` are written by neither
                        // side, so the whole arrays are equal.
                        assert_eq!(
                            old_dec.map(f32::to_bits),
                            old_enc.map(f32::to_bits),
                            "energies: {what}"
                        );

                        if !two_pass || !flag_coded {
                            one_pass_frames += 1;
                        } else if intra {
                            intra_frames += 1;
                        } else {
                            inter_frames += 1;
                        }
                    }
                }
            }
        }
    }
    // Both results of the two-pass decision were exercised, and frames with
    // a single pass too.
    assert!(
        intra_frames > 300,
        "intra pass kept in {intra_frames} frames"
    );
    assert!(
        inter_frames > 300,
        "inter pass kept in {inter_frames} frames"
    );
    assert!(
        one_pass_frames > 300,
        "{one_pass_frames} frames with one pass"
    );
}

// ---------------------------------------------------------------------------
// new_decoder against the libopus range decoder
// ---------------------------------------------------------------------------

/// The range decoder of libopus (celt/entdec.c) reading from a slice.
struct DecModel<'a> {
    buf: &'a [u8],
    end_offs: usize,
    end_window: u32,
    nend_bits: i32,
    nbits_total: i32,
    offs: usize,
    rng: u32,
    val: u32,
    ext: u32,
    rem: u32,
}

impl<'a> DecModel<'a> {
    /// ec_dec_init
    fn new(buf: &'a [u8]) -> Self {
        let mut d = DecModel {
            buf,
            end_offs: 0,
            end_window: 0,
            nend_bits: 0,
            nbits_total: 9,
            offs: 0,
            rng: 1 << 7,
            val: 0,
            ext: 0,
            rem: 0,
        };
        d.rem = d.read_byte();
        d.val = d.rng - 1 - (d.rem >> 1);
        d.normalize();
        d
    }

    fn read_byte(&mut self) -> u32 {
        if self.offs < self.buf.len() {
            self.offs += 1;
            self.buf[self.offs - 1] as u32
        } else {
            0
        }
    }

    fn read_byte_from_end(&mut self) -> u32 {
        if self.end_offs < self.buf.len() {
            self.end_offs += 1;
            self.buf[self.buf.len() - self.end_offs] as u32
        } else {
            0
        }
    }

    fn normalize(&mut self) {
        while self.rng <= 1 << 23 {
            self.nbits_total += 8;
            self.rng <<= 8;
            let sym = self.rem;
            self.rem = self.read_byte();
            let sym = ((sym << 8) | self.rem) >> 1;
            self.val = ((self.val << 8) + (0xFF & !sym)) & 0x7FFF_FFFF;
        }
    }

    /// ec_decode followed by ec_dec_update for the symbol that was found.
    fn decode_uniform(&mut self, ft: u32) -> u32 {
        self.ext = self.rng / ft;
        let s = self.val / self.ext;
        let fs = ft - (s + 1).min(ft);
        let s = self.ext * (ft - (fs + 1));
        self.val -= s;
        self.rng = if fs > 0 { self.ext } else { self.rng - s };
        self.normalize();
        fs
    }

    /// ec_dec_bit_logp
    fn decode_bit_logp(&mut self, logp: u32) -> bool {
        let s = self.rng >> logp;
        let ret = self.val < s;
        if !ret {
            self.val -= s;
        }
        self.rng = if ret { s } else { self.rng - s };
        self.normalize();
        ret
    }

    /// ec_dec_icdf
    fn decode_icdf(&mut self, icdf: &[u8], ftb: u32) -> i32 {
        let mut s = self.rng;
        let d = self.val;
        let r = s >> ftb;
        let mut ret = 0;
        let mut t;
        loop {
            t = s;
            s = r * icdf[ret] as u32;
            if d >= s {
                break;
            }
            ret += 1;
        }
        self.val = d - s;
        self.rng = t - s;
        self.normalize();
        ret as i32
    }

    /// ec_dec_bits
    fn dec_bits(&mut self, bits: u32) -> u32 {
        let mut window = self.end_window;
        let mut available = self.nend_bits;
        if (available as u32) < bits {
            loop {
                window |= self.read_byte_from_end() << available;
                available += 8;
                if available > 24 {
                    break;
                }
            }
        }
        let ret = window & ((1u32 << bits) - 1);
        self.end_window = window >> bits;
        self.nend_bits = available - bits as i32;
        self.nbits_total += bits as i32;
        ret
    }
}

fn decoder_matches(dec: &RangeCoder, model: &DecModel, what: &str) {
    assert_eq!(dec.rng, model.rng, "rng: {what}");
    assert_eq!(dec.val, model.val, "val: {what}");
    assert_eq!(dec.ext, model.ext, "ext: {what}");
    assert_eq!(dec.rem as u32, model.rem, "rem: {what}");
    assert_eq!(dec.nbits_total, model.nbits_total, "nbits_total: {what}");
    assert_eq!(dec.offs as usize, model.offs, "offs: {what}");
    assert_eq!(dec.end_offs as usize, model.end_offs, "end_offs: {what}");
    assert_eq!(dec.end_window, model.end_window, "end_window: {what}");
    assert_eq!(dec.nend_bits, model.nend_bits, "nend_bits: {what}");
    assert_eq!(dec.error, 0, "error: {what}");
}

/// A decoder made by `new_decoder` holds the packet and reads it like the
/// libopus decoder, from both ends, up to the last byte and beyond it.
///
/// Packets are random bytes of every length from 0 to 300 and of lengths
/// around the largest buffer; a packet longer than the buffer is cut. The
/// packet is a part of a larger array and starts at every offset from 0 to
/// 7 in turn, so the copy is made from differently aligned sources.
#[test]
fn new_decoder_reads_like_the_libopus_decoder() {
    let mut rng = SplitMix(0xDEC0_DE01);
    let arena: Vec<u8> = (0..RANGE_BUF_MAX + 1100)
        .map(|_| rng.next() as u8)
        .collect();
    let lengths = (0..=300usize).chain([
        1274,
        1275,
        RANGE_BUF_MAX - 1,
        RANGE_BUF_MAX,
        RANGE_BUF_MAX + 1,
        RANGE_BUF_MAX + 1000,
    ]);
    for len in lengths {
        let offset = len % 8;
        let data = &arena[offset..offset + len];
        let n = len.min(RANGE_BUF_MAX);
        let what = format!("length {len}");

        let mut dec = RangeCoder::new_decoder(data);
        assert_eq!(dec.buf.len(), n, "{what}");
        assert_eq!(dec.buf[..], data[..n], "{what}");
        assert_eq!(dec.storage as usize, n, "{what}");
        assert_eq!(dec.ext, 0, "{what}");
        let mut model = DecModel::new(&data[..n]);
        decoder_matches(&dec, &model, &what);

        // Read more than the packet holds, so both ends run past each other
        // and the decoder goes on with zeros.
        let reads = if len <= 300 { 3 * len + 24 } else { 6000 };
        for step in 0..reads {
            let what = format!("length {len} step {step}");
            match rng.below(4) {
                0 => {
                    let ft = 2 + rng.below(1000);
                    let got = dec.decode(ft);
                    dec.update(got, got + 1, ft);
                    assert_eq!(got, model.decode_uniform(ft), "{what}");
                }
                1 => {
                    let logp = 1 + rng.below(8);
                    assert_eq!(
                        dec.decode_bit_logp(logp),
                        model.decode_bit_logp(logp),
                        "{what}"
                    );
                }
                2 => {
                    assert_eq!(
                        dec.decode_icdf(&ICDF5, 8),
                        model.decode_icdf(&ICDF5, 8),
                        "{what}"
                    );
                }
                _ => {
                    let bits = 1 + rng.below(16);
                    assert_eq!(dec.dec_bits(bits), model.dec_bits(bits), "{what}");
                }
            }
            decoder_matches(&dec, &model, &what);
        }
        // The packet copy is not changed by reading.
        assert_eq!(dec.buf[..], data[..n], "{what}");
    }
}
