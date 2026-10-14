use crate::fixedvec::FixedVec;
use core::mem::MaybeUninit;

pub const EC_SYM_BITS: u32 = 8;
pub const EC_CODE_BITS: u32 = 32;
pub const EC_SYM_MAX: u32 = (1 << EC_SYM_BITS) - 1;
pub const EC_CODE_SHIFT: u32 = EC_CODE_BITS - EC_SYM_BITS - 1;
pub const EC_CODE_TOP: u32 = 1 << (EC_CODE_BITS - 1);
pub const EC_CODE_BOT: u32 = EC_CODE_TOP >> EC_SYM_BITS;
pub const EC_CODE_EXTRA: u32 = (EC_CODE_BITS - 2) % EC_SYM_BITS + 1;
pub const BITRES: i32 = 3;

/// Maximum range-coder buffer capacity. Opus packets are bounded by 1276 bytes
/// (RFC 6716 §3.1); standalone CELT usage (e.g. tests) may use up to 2048, so the
/// heap-free buffer is sized for that. The Opus encoder additionally clamps its
/// per-frame budget to `OPUS_MAX_PACKET_BYTES = 1276` (see `lib.rs`).
pub const RANGE_BUF_MAX: usize = 2048;

#[macro_export]
macro_rules! tell_frac_inline {
    ($rc:expr) => {{
        static CORRECTION: [u32; 8] = [35733, 38967, 42495, 46340, 50535, 55109, 60097, 65535];
        let nbits = $rc.nbits_total << BITRES;
        let l = 32 - $rc.rng.leading_zeros() as i32;
        let r = $rc.rng >> (l - 16);
        let b = (r >> 12).wrapping_sub(8);

        // `rng` is above 2^23 between symbols, so `r` has 16 bits and `b` is
        // 0 to 7. The mask keeps the index inside the table for any `rng`.
        let correction = CORRECTION[(b & 7) as usize];
        let b = b + (r > correction) as u32;
        nbits - (l << 3) - b as i32
    }};
}

#[derive(Clone)]
pub struct RangeCoder {
    pub buf: FixedVec<u8, RANGE_BUF_MAX>,
    pub storage: u32,
    pub end_offs: u32,
    pub end_window: u32,
    pub nend_bits: i32,
    pub nbits_total: i32,
    pub offs: u32,
    pub rng: u32,
    pub val: u32,
    pub ext: u32,
    pub rem: i32,
    pub error: i32,
}

/// The words of a [`RangeCoder`] without its byte buffer (44 bytes).
///
/// An encoder that codes the same symbols in two ways and keeps the better
/// result does not need a copy of the whole coder for that. The words and
/// the bytes written in between are enough: range coder bytes below `offs`
/// never change once written. `quant_coarse_energy_advanced` works this way
/// (libopus: the `ec_enc` copies in `quant_coarse_energy`).
#[derive(Clone, Copy)]
pub struct RangeCoderState {
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

impl RangeCoder {
    /// Writes every field of a coder to `this`, which may be uninitialized
    /// memory: the buffer gets `bytes`, the words get `state`.
    ///
    /// Panics if `bytes.len() > RANGE_BUF_MAX`, before anything is written.
    ///
    /// # Safety
    /// `this` must be valid for writes of `Self`.
    #[inline(always)]
    unsafe fn write_parts(this: *mut Self, bytes: &[u8], state: &RangeCoderState) {
        // Every word of the state is named here. A word that is not written
        // below is reported by the compiler as an unused variable.
        let RangeCoderState {
            storage,
            end_offs,
            end_window,
            nend_bits,
            nbits_total,
            offs,
            rng,
            val,
            ext,
            rem,
            error,
        } = *state;
        // SAFETY: the caller guarantees that `this` is valid for writes. Each
        // field is written exactly once; none is read. The fields are the
        // buffer and the words of `RangeCoderState`; `save_state` fails to
        // compile when `RangeCoder` gets a field that is neither.
        unsafe {
            FixedVec::write_from_slice(&raw mut (*this).buf, bytes);
            (&raw mut (*this).storage).write(storage);
            (&raw mut (*this).end_offs).write(end_offs);
            (&raw mut (*this).end_window).write(end_window);
            (&raw mut (*this).nend_bits).write(nend_bits);
            (&raw mut (*this).nbits_total).write(nbits_total);
            (&raw mut (*this).offs).write(offs);
            (&raw mut (*this).rng).write(rng);
            (&raw mut (*this).val).write(val);
            (&raw mut (*this).ext).write(ext);
            (&raw mut (*this).rem).write(rem);
            (&raw mut (*this).error).write(error);
        }
    }

    /// Copies the words of the coder. The byte buffer is not copied.
    #[inline]
    pub fn save_state(&self) -> RangeCoderState {
        // The pattern names every field of the coder and has no `..`. A
        // field added to `RangeCoder` does not compile here until it is
        // added to `RangeCoderState` as well. `write_parts` relies on this:
        // it builds a coder from the buffer and these words alone.
        let RangeCoder {
            buf: _,
            storage,
            end_offs,
            end_window,
            nend_bits,
            nbits_total,
            offs,
            rng,
            val,
            ext,
            rem,
            error,
        } = *self;
        RangeCoderState {
            storage,
            end_offs,
            end_window,
            nend_bits,
            nbits_total,
            offs,
            rng,
            val,
            ext,
            rem,
            error,
        }
    }

    /// Puts back words saved with [`RangeCoder::save_state`].
    ///
    /// The byte buffer is not touched. Going back to an earlier state of an
    /// encoder is complete when only range coder symbols were coded since
    /// then: those write at and above the saved `offs`, and the encoder
    /// overwrites them from there. To return to a later state, the caller
    /// also puts back the bytes between the two `offs` values.
    ///
    /// Panics if the saved `storage` is larger than this coder's buffer,
    /// which means the state was saved from another coder. Such a coder
    /// could not write a valid packet: every write past the buffer would
    /// only set the error flag.
    #[inline]
    pub fn restore_state(&mut self, state: &RangeCoderState) {
        // As in `write_parts`: a word that is not put back is reported as an
        // unused variable.
        let RangeCoderState {
            storage,
            end_offs,
            end_window,
            nend_bits,
            nbits_total,
            offs,
            rng,
            val,
            ext,
            rem,
            error,
        } = *state;
        assert!(
            storage as usize <= self.buf.len(),
            "restore_state: the state belongs to a coder with a larger buffer"
        );
        self.storage = storage;
        self.end_offs = end_offs;
        self.end_window = end_window;
        self.nend_bits = nend_bits;
        self.nbits_total = nbits_total;
        self.offs = offs;
        self.rng = rng;
        self.val = val;
        self.ext = ext;
        self.rem = rem;
        self.error = error;
    }

    pub fn new_encoder(size: u32) -> Self {
        let size = (size as usize).clamp(1, RANGE_BUF_MAX);
        let buf = FixedVec::from_value(0u8, size);
        RangeCoder {
            buf,
            storage: size as u32,
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

    #[inline]
    pub fn reset_for_encode(&mut self, size: u32) {
        let size = (size as usize).clamp(1, RANGE_BUF_MAX);
        self.buf.resize(size, 0);
        self.storage = size as u32;
        self.end_offs = 0;
        self.end_window = 0;
        self.nend_bits = 0;
        self.nbits_total = 33;
        self.offs = 0;
        self.rng = 1 << 31;
        self.val = 0;
        self.ext = 0;
        self.rem = -1;
        self.error = 0;
    }

    /// Starts decoding `data`. At most `RANGE_BUF_MAX` bytes are used.
    ///
    /// The coder keeps its own copy of the packet. Only the packet bytes are
    /// copied: the coder is built in the caller's variable (see
    /// `init_decoder`), not in a local that is then moved.
    #[inline]
    pub fn new_decoder(data: &[u8]) -> Self {
        let mut rc = MaybeUninit::<Self>::uninit();
        // SAFETY: `rc` is valid for writes. `init_decoder` writes every field.
        unsafe {
            Self::init_decoder(rc.as_mut_ptr(), data);
            rc.assume_init()
        }
    }

    /// Writes a decoder for `data` to `this`.
    ///
    /// This is a separate call on purpose. A function that builds the coder
    /// in a local and returns it copies all 2104 bytes to the caller (twice,
    /// before this function existed), however short the packet is. A call
    /// that fills memory through a pointer, followed by one move of the
    /// result, is a pattern the compiler turns into "fill the caller's
    /// variable directly".
    ///
    /// # Safety
    /// `this` must be valid for writes of `Self`.
    #[inline(never)]
    unsafe fn init_decoder(this: *mut Self, data: &[u8]) {
        let n = data.len().min(RANGE_BUF_MAX);
        let start = RangeCoderState {
            storage: n as u32,
            end_offs: 0,
            end_window: 0,
            nend_bits: 0,
            nbits_total: (EC_CODE_BITS + 1
                - ((EC_CODE_BITS - EC_CODE_EXTRA) / EC_SYM_BITS) * EC_SYM_BITS)
                as i32,
            offs: 0,
            rng: 1 << EC_CODE_EXTRA,
            val: 0,
            ext: 0,
            rem: 0,
            error: 0,
        };
        // SAFETY: the caller guarantees that `this` is valid for writes.
        // `write_parts` writes every field (it cannot panic here, because
        // `n <= RANGE_BUF_MAX`), so the reference made afterwards points to
        // an initialized coder.
        let rc = unsafe {
            Self::write_parts(this, &data[..n], &start);
            &mut *this
        };

        rc.rem = rc.read_byte() as i32;
        rc.val = rc
            .rng
            .wrapping_sub(1)
            .wrapping_sub(rc.rem as u32 >> (EC_SYM_BITS - EC_CODE_EXTRA));

        rc.normalize_decoder();
    }

    #[inline(always)]
    fn normalize_decoder(&mut self) {
        let mut guard = 0u32;
        while self.rng <= EC_CODE_BOT {
            guard += 1;
            if guard > 100 {
                self.error = 1;
                self.rng = EC_CODE_BOT + 1;
                break;
            }
            self.nbits_total += EC_SYM_BITS as i32;
            self.rng <<= EC_SYM_BITS;

            let sym = self.rem;
            self.rem = self.read_byte() as i32;

            let combined_sym = ((sym << EC_SYM_BITS) | self.rem) >> (EC_SYM_BITS - EC_CODE_EXTRA);
            self.val = (self.val << EC_SYM_BITS).wrapping_add(EC_SYM_MAX & !combined_sym as u32)
                & (EC_CODE_TOP - 1);
        }
    }

    fn read_byte(&mut self) -> u8 {
        if self.offs < self.storage {
            let b = self.buf[self.offs as usize];
            self.offs += 1;
            b
        } else {
            0
        }
    }

    #[inline(always)]
    pub fn enc_uint(&mut self, fl: u32, ft: u32) {
        if ft > 1 {
            let ft_minus_1 = ft - 1;
            let ftb = 32 - ft_minus_1.leading_zeros() as i32;
            if ftb > 8 {
                let s = ftb - 8;
                let fl_low = fl & ((1u32 << s) - 1);
                let fl = fl >> s;
                let ft = (ft_minus_1 >> s) + 1;
                self.encode(fl, fl.wrapping_add(1), ft);
                self.enc_bits(fl_low, s as u32);
            } else {
                self.encode(fl, fl.wrapping_add(1), ft);
            }
        }
    }

    /// Decodes an integer from 0 to `ft - 1` (libopus: `ec_dec_uint`,
    /// celt/entdec.c).
    ///
    /// Above 256 values the integer is coded in two parts: the high bits
    /// with the range coder, the low bits as raw bits. In a damaged packet
    /// the two parts can combine to `ft` or more. As in libopus, the result
    /// is then `ft - 1` and the error flag of the coder is set.
    #[inline(always)]
    pub fn dec_uint(&mut self, ft: u32) -> u32 {
        if ft > 1 {
            let ft_minus_1 = ft - 1;
            let ftb = 32 - ft_minus_1.leading_zeros() as i32;
            if ftb > 8 {
                let s = ftb - 8;
                let ft = (ft_minus_1 >> s) + 1;
                let fs = self.decode(ft);
                self.update(fs, fs.wrapping_add(1), ft);
                let t = (fs << s) | self.dec_bits(s as u32);
                if t <= ft_minus_1 {
                    t
                } else {
                    self.error = 1;
                    ft_minus_1
                }
            } else {
                let fs = self.decode(ft);
                self.update(fs, fs.wrapping_add(1), ft);
                fs
            }
        } else {
            0
        }
    }

    #[inline(always)]
    pub fn enc_bits(&mut self, val: u32, bits: u32) {
        if bits == 0 {
            return;
        }
        let mut window = self.end_window;
        let mut used = self.nend_bits;
        if (used as u32) + bits > EC_CODE_BITS {
            while used >= EC_SYM_BITS as i32 {
                self.write_byte_at_end((window & EC_SYM_MAX) as u8);
                window >>= EC_SYM_BITS;
                used -= EC_SYM_BITS as i32;
            }
        }
        window |= (val & ((1 << bits) - 1)) << used;
        used += bits as i32;
        self.end_window = window;
        self.nend_bits = used;
        self.nbits_total += bits as i32;
    }

    pub fn dec_bits(&mut self, bits: u32) -> u32 {
        if bits == 0 {
            return 0;
        }
        let mut window = self.end_window;
        let mut used = self.nend_bits;
        if used < bits as i32 {
            loop {
                let byte = if self.end_offs < self.storage {
                    self.end_offs += 1;
                    self.buf[(self.storage - self.end_offs) as usize]
                } else {
                    0
                };
                window |= (byte as u32) << used;
                used += 8;
                if used > 32 - 8 {
                    break;
                }
            }
        }
        let ret = window & ((1 << bits) - 1);
        self.end_window = window >> bits;
        self.nend_bits = used - bits as i32;
        self.nbits_total += bits as i32;
        ret
    }

    #[inline(always)]
    pub fn tell_frac(&self) -> i32 {
        static CORRECTION: [u32; 8] = [35733, 38967, 42495, 46340, 50535, 55109, 60097, 65535];
        let nbits = self.nbits_total << BITRES;
        let l = 32 - self.rng.leading_zeros() as i32;
        let r = self.rng >> (l - 16);
        let b = (r >> 12).wrapping_sub(8);
        let b = b + (r > CORRECTION[b as usize]) as u32;
        nbits - (l << 3) - b as i32
    }

    /// Integer bit count matching C's `ec_tell()`: `nbits_total - EC_ILOG(rng)`.
    #[inline(always)]
    pub fn tell(&self) -> i32 {
        self.nbits_total - (32 - self.rng.leading_zeros() as i32)
    }

    #[inline(always)]
    pub fn tell_fast(&self) -> i32 {
        self.nbits_total
    }

    /// Shrink the range coder buffer, moving end-coded bytes to the new end.
    /// Equivalent to C's `ec_enc_shrink`.
    pub fn shrink(&mut self, new_size: u32) {
        debug_assert!(self.offs + self.end_offs <= new_size);
        if self.end_offs > 0 {
            let old_end_start = (self.storage - self.end_offs) as usize;
            let old_end_end = self.storage as usize;
            let new_end_start = (new_size - self.end_offs) as usize;
            self.buf
                .copy_within(old_end_start..old_end_end, new_end_start);
        }
        self.storage = new_size;
    }

    /// Writes the next range coder byte, or sets the error flag when the
    /// buffer is full (libopus: `ec_write_byte`).
    ///
    /// The second condition is always true for a coder whose `storage` is
    /// within its buffer. It is what keeps the write inside the buffer when
    /// safe code has set `storage`, `offs` or the buffer length to something
    /// else: all three are public.
    #[inline(always)]
    fn write_byte(&mut self, value: u8) {
        let offs = self.offs as usize;
        if self.offs + self.end_offs < self.storage && offs < self.buf.len() {
            self.buf[offs] = value;
            self.offs += 1;
        } else {
            self.error = 1;
        }
    }

    #[inline(always)]
    fn carry_out(&mut self, c: i32) {
        if c != EC_SYM_MAX as i32 {
            let carry = c >> EC_SYM_BITS;
            if self.rem >= 0 {
                self.write_byte((self.rem + carry) as u8);
            }
            if self.ext > 0 {
                let sym = (EC_SYM_MAX as i32 + carry) & EC_SYM_MAX as i32;

                let ext = self.ext as usize;
                for _j in 0..ext {
                    self.write_byte(sym as u8);
                }
                self.ext = 0;
            }
            self.rem = c & EC_SYM_MAX as i32;
        } else {
            self.ext += 1;
        }
    }

    #[inline(always)]
    fn celt_udiv(n: u32, d: u32) -> u32 {
        // assert! (not debug_assert!): d == 0 is a hard arithmetic fault in
        // release builds too; fail fast with a clear message instead
        // (issue #27 deep scan).
        assert!(d > 0, "celt_udiv: division by zero");
        n / d
    }

    #[inline(always)]
    pub fn encode(&mut self, fl: u32, fh: u32, ft: u32) {
        // assert! (not debug_assert!): ft == 0 hits celt_udiv's hard fault in
        // release as well (issue #27 deep scan).
        assert!(ft > 0, "encode: ft must be > 0");
        let r = Self::celt_udiv(self.rng, ft);
        if fl > 0 {
            self.val = self
                .val
                .wrapping_add(self.rng.wrapping_sub(r.wrapping_mul(ft.wrapping_sub(fl))));
            self.rng = r.wrapping_mul(fh.wrapping_sub(fl));
        } else {
            self.rng = self.rng.wrapping_sub(r.wrapping_mul(ft.wrapping_sub(fh)));
        }
        self.normalize_encoder();
    }

    #[inline(always)]
    fn normalize_encoder(&mut self) {
        while self.rng <= EC_CODE_BOT {
            let c = (self.val >> EC_CODE_SHIFT) as i32;
            if c != EC_SYM_MAX as i32 {
                let carry = c >> EC_SYM_BITS;
                if self.rem >= 0 {
                    self.write_byte((self.rem + carry) as u8);
                }
                if self.ext > 0 {
                    let sym = (EC_SYM_MAX as i32 + carry) & EC_SYM_MAX as i32;
                    let ext = self.ext as usize;
                    for _j in 0..ext {
                        self.write_byte(sym as u8);
                    }
                    self.ext = 0;
                }
                self.rem = c & EC_SYM_MAX as i32;
            } else {
                self.ext += 1;
            }
            self.val = (self.val << EC_SYM_BITS) & (EC_CODE_TOP - 1);
            self.rng <<= EC_SYM_BITS;
            self.nbits_total = self.nbits_total.wrapping_add(EC_SYM_BITS as i32);
        }
    }

    #[inline(always)]
    pub fn encode_bit_logp(&mut self, val: bool, logp: u32) {
        let s = self.rng >> logp;
        let r = self.rng.wrapping_sub(s);
        if val {
            self.val = self.val.wrapping_add(r);
            self.rng = s;
        } else {
            self.rng = r;
        }
        self.normalize_encoder();
    }

    #[inline(always)]
    pub fn encode_icdf(&mut self, s: i32, icdf: &[u8], ftb: u32) {
        // The table is indexed with a check: a symbol outside the table is
        // a bug in the caller and panics, it does not read other memory. A
        // negative symbol becomes a huge index. Entry `s` is read first, so
        // that one check covers entry `s - 1` as well.
        let s = s as usize;
        let lower = icdf[s] as u32;
        let r = self.rng >> ftb;
        if s > 0 {
            let val = icdf[s - 1] as u32;
            self.val = self
                .val
                .wrapping_add(self.rng.wrapping_sub(r.wrapping_mul(val)));
            self.rng = r.wrapping_mul(val.wrapping_sub(lower));
        } else {
            self.rng = self.rng.wrapping_sub(r.wrapping_mul(lower));
        }
        self.normalize_encoder();
    }

    #[inline(always)]
    pub fn decode_bit_logp(&mut self, logp: u32) -> bool {
        let s = self.rng >> logp;
        let ret = self.val < s;
        if !ret {
            self.val = self.val.wrapping_sub(s);
            self.rng = self.rng.wrapping_sub(s);
        } else {
            self.rng = s;
        }
        self.normalize_decoder();
        ret
    }

    #[inline(always)]
    pub fn decode_icdf(&mut self, icdf: &[u8], ftb: u32) -> i32 {
        let mut s = self.rng;
        let d = self.val;
        let r = s >> ftb;
        let mut ret = 0;
        let mut t;

        loop {
            t = s;
            s = r.wrapping_mul(icdf[ret] as u32);
            ret += 1;
            if d >= s {
                break;
            }
        }

        self.val = d.wrapping_sub(s);
        self.rng = t.wrapping_sub(s);
        self.normalize_decoder();
        (ret - 1) as i32
    }

    #[inline(always)]
    pub fn decode(&mut self, ft: u32) -> u32 {
        assert!(ft > 0, "decode: ft must be > 0");
        let r = Self::celt_udiv(self.rng, ft);
        self.ext = r;
        let s = self.val / r;
        ft - ft.min(s.wrapping_add(1))
    }

    #[inline(always)]
    pub fn update(&mut self, fl: u32, fh: u32, ft: u32) {
        let s = self.ext.wrapping_mul(ft.wrapping_sub(fh));
        self.val = self.val.wrapping_sub(s);
        self.rng = if fl > 0 {
            self.ext.wrapping_mul(fh.wrapping_sub(fl))
        } else {
            self.rng.wrapping_sub(s)
        };
        self.normalize_decoder();
    }

    pub fn laplace_encode(&mut self, value: &mut i32, fs: u32, decay: i32) {
        let mut val = *value;
        let mut fl = 0;
        let mut fs_val = fs;

        if val != 0 {
            let s = if val < 0 { -1 } else { 0 };
            val = (val + s) ^ s;
            fl = fs_val;
            fs_val = self.laplace_get_freq1(fs_val, decay);

            let mut i = 1;
            while fs_val > 0 && i < val {
                fs_val *= 2;
                fl += fs_val + 2;
                fs_val = ((fs_val as i32 * decay) >> 15) as u32;
                i += 1;
            }

            if fs_val == 0 {
                let ndi_max = 32768 - fl + 1 - 1;
                let ndi_max = (ndi_max as i32 - s) >> 1;
                let di = (val - i).min(ndi_max - 1);
                fl += (2 * di + 1 + s) as u32;
                fs_val = 1u32.min(32768 - fl);
                *value = (i + di + s) ^ s;
            } else {
                fs_val += 1;
                fl += fs_val & (!s as u32);
            }
        }
        self.encode(fl, fl.wrapping_add(fs_val), 1 << 15);
    }

    fn laplace_get_freq1(&self, fs0: u32, decay: i32) -> u32 {
        let ft = 32768 - (2 * 16) - fs0;
        ((ft as i32 * (16384 - decay)) >> 15) as u32
    }

    /// `#[inline]`: the coarse energy decoder calls this once per band and
    /// channel from another module. Without the attribute the default
    /// release build (16 code-generation units) inlines it in some builds
    /// and not in others, depending on how the crate happens to be split,
    /// and the 2.5 ms decode rows move by 1 to 2 % with it.
    ///
    /// `laplace_encode` is left without the attribute on purpose: inlined
    /// into the coarse energy encoder it measured 12 to 19 % slower there.
    #[inline]
    pub fn laplace_decode(&mut self, fs: u32, decay: i32) -> i32 {
        let fm = self.decode(1 << 15);
        let mut fl = 0;
        let mut fs_val = fs;
        let mut val = 0;

        if fm >= fs_val {
            val += 1;
            fl = fs_val;
            fs_val = self.laplace_get_freq1(fs_val, decay) + 1;

            while fs_val > 1 && fm >= fl + 2 * fs_val {
                fs_val *= 2;
                fl += fs_val;
                fs_val = (((fs_val as i32 - 2) * decay) >> 15) as u32 + 1;
                val += 1;
            }

            if fs_val <= 1 {
                let di = (fm - fl) >> 1;
                val += di as i32;
                fl += 2 * di;
            }

            if fm < fl + fs_val {
                val = -val;
            } else {
                fl += fs_val;
            }
        }

        self.update(fl, fl.wrapping_add(fs_val.min(32768 - fl)), 1 << 15);
        val
    }

    /// Writes the next byte of raw bits, from the end of the buffer
    /// downwards, or sets the error flag when the buffer is full (libopus:
    /// `ec_write_byte_at_end`). As in `write_byte`, the second condition
    /// only matters for a coder whose public words were set inconsistently.
    #[inline(always)]
    fn write_byte_at_end(&mut self, value: u8) {
        let idx = self.storage.wrapping_sub(self.end_offs).wrapping_sub(1) as usize;
        if self.offs + self.end_offs < self.storage && idx < self.buf.len() {
            self.end_offs += 1;
            self.buf[idx] = value;
        } else {
            self.error = 1;
        }
    }

    pub fn patch_initial_bits(&mut self, val: u32, nbits: u32) {
        let shift = EC_SYM_BITS - nbits;
        let mask = ((1u32 << nbits) - 1) << shift;
        if self.offs > 0 {
            self.buf[0] = ((self.buf[0] as u32 & !mask) | (val << shift)) as u8;
        } else if self.rem >= 0 {
            self.rem = ((self.rem as u32 & !mask) | (val << shift)) as i32;
        } else if self.rng <= (EC_CODE_TOP >> nbits) {
            let mask_shifted = mask << EC_CODE_SHIFT;
            self.val = (self.val & !mask_shifted) | (val << (EC_CODE_SHIFT + shift));
        } else {
            self.error = -1;
        }
    }

    pub fn done(&mut self) {
        let ilog = 32 - self.rng.leading_zeros();
        let mut l = (EC_CODE_BITS - ilog) as i32;
        let mut msk = (EC_CODE_TOP - 1) >> l;
        let mut end = (self.val.wrapping_add(msk)) & !msk;

        if (end | msk) >= self.val.wrapping_add(self.rng) {
            l += 1;
            msk >>= 1;
            end = (self.val.wrapping_add(msk)) & !msk;
        }

        while l > 0 {
            self.carry_out((end >> EC_CODE_SHIFT) as i32);
            end = (end << EC_SYM_BITS) & (EC_CODE_TOP - 1);
            l -= EC_SYM_BITS as i32;
        }

        if self.rem >= 0 || self.ext > 0 {
            self.carry_out(0);
        }

        let mut window = self.end_window;
        let mut used = self.nend_bits;
        while used >= EC_SYM_BITS as i32 {
            self.write_byte_at_end((window & EC_SYM_MAX) as u8);
            window >>= EC_SYM_BITS;
            used -= EC_SYM_BITS as i32;
        }

        if self.error == 0 {
            // Clear the unused bytes between the range coder bytes and the
            // raw bits. One slice fill instead of a store per byte: the gap
            // is about 1.2 kB in a SILK frame coded into a 1275-byte buffer.
            let gap_start = self.offs as usize;
            let gap_end = (self.storage - self.end_offs) as usize;
            if gap_start < gap_end {
                self.buf[gap_start..gap_end].fill(0);
            }

            if used > 0 {
                if self.end_offs >= self.storage {
                    self.error = -1;
                } else {
                    // If we've busted, don't add too many extra bits to the
                    // last byte; it would corrupt the range coder data.
                    if self.offs + self.end_offs >= self.storage && -l < used {
                        window &= (1u32 << (-l as u32)) - 1;
                        self.error = -1;
                    }
                    let idx = (self.storage - self.end_offs - 1) as usize;
                    self.buf[idx] |= window as u8;
                }
            }
        }
    }

    /// Flush pending output and return the encoded bytes.
    ///
    /// Allocates and is therefore only available with the `std` feature. The
    /// `#![no_std]` encoder path reads `self.buf[..self.offs]` directly instead.
    #[cfg(feature = "std")]
    pub fn finish(&mut self) -> Vec<u8> {
        self.done();

        let extra_end = if self.nend_bits > 0 && self.end_offs == 0 {
            1
        } else {
            0
        };
        let mut result = Vec::with_capacity((self.offs + self.end_offs + extra_end) as usize);
        result.extend_from_slice(&self.buf[0..self.offs as usize]);
        if extra_end > 0 {
            result.push(self.buf[(self.storage - 1) as usize]);
        }
        result.extend_from_slice(
            &self.buf[(self.storage - self.end_offs) as usize..self.storage as usize],
        );
        result
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;

    #[test]
    fn test_laplace() {
        let mut enc = RangeCoder::new_encoder(100);
        let mut val = -3;
        let fs = 100 << 7;
        let decay = 120 << 6;
        enc.laplace_encode(&mut val, fs, decay);
        enc.done();

        assert_eq!(enc.offs, 1);
        assert_eq!(enc.buf[0], 224);

        let mut dec = RangeCoder::new_decoder(&enc.buf[..enc.offs as usize]);
        let decoded_val = dec.laplace_decode(fs, decay);
        assert_eq!(decoded_val, -3);
    }

    #[test]
    fn test_icdf_consistency() {
        let mut enc = RangeCoder::new_encoder(1024);
        let icdf = [2, 1, 0];
        enc.encode_icdf(0, &icdf, 2);
        enc.encode_icdf(1, &icdf, 2);
        enc.encode_icdf(2, &icdf, 2);
        enc.done();
        let data = enc.buf[..enc.offs as usize].to_vec();

        let mut dec = RangeCoder::new_decoder(&data);
        let s0 = dec.decode_icdf(&icdf, 2);
        let s1 = dec.decode_icdf(&icdf, 2);
        let s2 = dec.decode_icdf(&icdf, 2);

        assert_eq!(s0, 0);
        assert_eq!(s1, 1);
        assert_eq!(s2, 2);
    }

    #[test]
    fn test_icdf_last_symbol_no_oob() {
        let icdf: &[u8] = &[170, 85, 0];
        let ftb = 8u32;

        for sym in 0..3i32 {
            let mut enc = RangeCoder::new_encoder(256);
            enc.encode_icdf(sym, icdf, ftb);
            enc.done();
            let data = enc.buf[..enc.offs as usize].to_vec();

            let mut dec = RangeCoder::new_decoder(&data);
            let decoded = dec.decode_icdf(icdf, ftb);
            assert_eq!(decoded, sym, "往返失败: 编码 symbol={sym} 解码得 {decoded}");
        }
    }

    #[test]
    fn test_icdf_decode_terminates() {
        let icdf: &[u8] = &[192, 128, 64, 0];
        let ftb = 8u32;

        let symbols = [0i32, 1, 2, 3];
        let mut enc = RangeCoder::new_encoder(256);
        for &s in &symbols {
            enc.encode_icdf(s, icdf, ftb);
        }
        enc.done();
        let data = enc.buf[..enc.offs as usize].to_vec();

        let mut dec = RangeCoder::new_decoder(&data);
        for &expected in &symbols {
            let got = dec.decode_icdf(icdf, ftb);
            assert_eq!(got, expected, "解码器输出 {got}，期望 {expected}");
        }
    }

    #[test]
    fn test_bits_only() {
        let mut enc = RangeCoder::new_encoder(1024);

        enc.enc_bits(1, 1);
        enc.enc_bits(5, 3);
        enc.enc_bits(7, 3);
        enc.enc_bits(0, 2);

        let data = enc.finish();
        let mut dec = RangeCoder::new_decoder(&data);

        let b1 = dec.dec_bits(1);
        let b2 = dec.dec_bits(3);
        let b3 = dec.dec_bits(3);
        let b4 = dec.dec_bits(2);

        assert_eq!(b1, 1);
        assert_eq!(b2, 5);
        assert_eq!(b3, 7);
        assert_eq!(b4, 0);
    }

    #[test]
    fn test_interleaved_bits_entropy() {
        let mut enc = RangeCoder::new_encoder(1024);

        enc.enc_bits(1, 1);

        enc.encode(10, 20, 100);

        enc.enc_bits(5, 3);

        enc.encode(50, 60, 100);

        let data = enc.finish();

        let mut dec = RangeCoder::new_decoder(&data);

        let b1 = dec.dec_bits(1);
        let d1 = dec.decode(100);
        dec.update(10, 20, 100);
        let b2 = dec.dec_bits(3);
        let d2 = dec.decode(100);
        dec.update(50, 60, 100);

        assert_eq!(b1, 1);
        assert!((10..20).contains(&d1), "d1={} expected in [10, 20)", d1);
        assert_eq!(b2, 5);
        assert!((50..60).contains(&d2), "d2={} expected in [50, 60)", d2);
    }
}
