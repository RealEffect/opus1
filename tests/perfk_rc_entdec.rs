//! The range decoder of opus-rs against the range decoder of libopus
//! (celt/entdec.c, celt/laplace.c), on packets that no encoder wrote.
//!
//! libopus is linked as a static library, so its internal functions can be
//! called directly. Both decoders read the same random bytes with the same
//! random sequence of calls. After every call the returned value and all
//! words of the two coders must be equal, the error flag included.
//!
//! A valid packet never shows how a decoder treats values that an encoder
//! cannot write. On random bytes every primitive meets them: an integer
//! whose two parts combine to more than its limit, raw bits read past the
//! start of the packet, range coder bytes read past its end.

use opus_rs::quant_bands::E_PROB_MODEL;
use opus_rs::range_coder::RangeCoder;
use std::os::raw::{c_int, c_uint};

/// `ec_ctx` of libopus (celt/entcode.h).
#[repr(C)]
struct EcCtx {
    buf: *mut u8,
    storage: u32,
    end_offs: u32,
    end_window: u32,
    nend_bits: c_int,
    nbits_total: c_int,
    offs: u32,
    rng: u32,
    val: u32,
    ext: u32,
    rem: c_int,
    error: c_int,
}

unsafe extern "C" {
    fn ec_dec_init(this: *mut EcCtx, buf: *mut u8, storage: u32);
    fn ec_decode(this: *mut EcCtx, ft: c_uint) -> c_uint;
    fn ec_dec_update(this: *mut EcCtx, fl: c_uint, fh: c_uint, ft: c_uint);
    fn ec_dec_bit_logp(this: *mut EcCtx, logp: c_uint) -> c_int;
    fn ec_dec_icdf(this: *mut EcCtx, icdf: *const u8, ftb: c_uint) -> c_int;
    fn ec_dec_uint(this: *mut EcCtx, ft: u32) -> u32;
    fn ec_dec_bits(this: *mut EcCtx, bits: c_uint) -> u32;
    fn ec_laplace_decode(this: *mut EcCtx, fs: c_uint, decay: c_int) -> c_int;
    fn ec_tell_frac(this: *mut EcCtx) -> u32;
}

/// Makes sure the test binary links libopus: the functions above are found
/// in the static library that `audiopus_sys` brings along.
fn link_libopus() {
    let version = unsafe { audiopus_sys::opus_get_version_string() };
    assert!(!version.is_null());
}

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
    fn below(&mut self, n: u32) -> u32 {
        ((self.next() as u64 * n as u64) >> 32) as u32
    }
}

/// The libopus decoder on its own copy of the packet.
struct CDec {
    ctx: EcCtx,
    _bytes: Vec<u8>,
}

impl CDec {
    fn new(packet: &[u8]) -> Self {
        let mut bytes = packet.to_vec();
        let mut ctx = EcCtx {
            buf: std::ptr::null_mut(),
            storage: 0,
            end_offs: 0,
            end_window: 0,
            nend_bits: 0,
            nbits_total: 0,
            offs: 0,
            rng: 0,
            val: 0,
            ext: 0,
            rem: 0,
            error: 0,
        };
        // SAFETY: `bytes` lives as long as `ctx` (both are moved into the
        // result; the heap buffer of the Vec does not move).
        unsafe { ec_dec_init(&mut ctx, bytes.as_mut_ptr(), bytes.len() as u32) };
        CDec { ctx, _bytes: bytes }
    }

    fn words(&self) -> [i64; 10] {
        let c = &self.ctx;
        [
            c.storage as i64,
            c.end_offs as i64,
            c.end_window as i64,
            c.nend_bits as i64,
            c.nbits_total as i64,
            c.offs as i64,
            c.rng as i64,
            c.val as i64,
            c.rem as i64,
            c.error as i64,
        ]
    }
}

/// The words both decoders must agree on. `ext` is left out: it is a
/// scratch value that only `decode` followed by `update` uses.
fn words(rc: &RangeCoder) -> [i64; 10] {
    [
        rc.storage as i64,
        rc.end_offs as i64,
        rc.end_window as i64,
        rc.nend_bits as i64,
        rc.nbits_total as i64,
        rc.offs as i64,
        rc.rng as i64,
        rc.val as i64,
        rc.rem as i64,
        rc.error as i64,
    ]
}

const ICDF2: [u8; 2] = [120, 0];
const ICDF3: [u8; 3] = [2, 1, 0];
const ICDF5: [u8; 5] = [200, 150, 90, 20, 0];
const ICDF16: [u8; 16] = [250, 240, 225, 200, 170, 140, 110, 85, 60, 40, 25, 15, 8, 4, 1, 0];

/// A limit for `dec_uint`: small, at a power of two, one above a power of
/// two, large, or anything.
fn uint_limit(rng: &mut Lcg) -> u32 {
    match rng.below(8) {
        0 => 2 + rng.below(255),
        1 => 257 + rng.below(4),
        2 => 1 << (9 + rng.below(23)),
        3 => (1 << (8 + rng.below(24))) + 1 + rng.below(3),
        4 => u32::MAX - rng.below(3),
        5 => 2 + rng.below(100_000),
        _ => rng.next().max(2),
    }
}

fn packet(rng: &mut Lcg, kind: u32) -> Vec<u8> {
    let len = match rng.below(6) {
        0 => 0,
        1 => 1 + rng.below(3),
        2 => 4 + rng.below(8),
        _ => 12 + rng.below(80),
    } as usize;
    (0..len)
        .map(|_| match kind % 4 {
            0 => rng.next() as u8,
            1 => 0xFF,
            2 => 0x00,
            // Mostly ones: the top values of every symbol.
            _ => {
                if rng.below(5) == 0 { rng.next() as u8 } else { 0xFF }
            }
        })
        .collect()
}

/// One random call on both decoders. Returns a description and the two
/// results.
fn step(rng: &mut Lcg, rc: &mut RangeCoder, c: &mut CDec, only_uint: bool) -> (String, i64, i64) {
    let ctx = &mut c.ctx as *mut EcCtx;
    let op = if only_uint { 0 } else { rng.below(7) };
    // SAFETY: `ctx` points to a decoder set up by `ec_dec_init` whose packet
    // is alive. The tables end with 0, as libopus requires.
    unsafe {
        match op {
            0 => {
                let ft = uint_limit(rng);
                (format!("dec_uint({ft})"), rc.dec_uint(ft) as i64, ec_dec_uint(ctx, ft) as i64)
            }
            1 => {
                let bits = 1 + rng.below(25);
                (format!("dec_bits({bits})"), rc.dec_bits(bits) as i64, ec_dec_bits(ctx, bits) as i64)
            }
            2 => {
                let logp = 1 + rng.below(15);
                (
                    format!("decode_bit_logp({logp})"),
                    rc.decode_bit_logp(logp) as i64,
                    ec_dec_bit_logp(ctx, logp) as i64,
                )
            }
            3 => {
                let (icdf, ftb): (&[u8], u32) = match rng.below(4) {
                    0 => (&ICDF2, 8),
                    1 => (&ICDF3, 2),
                    2 => (&ICDF5, 8),
                    _ => (&ICDF16, 8),
                };
                (
                    format!("decode_icdf(len {}, {ftb})", icdf.len()),
                    rc.decode_icdf(icdf, ftb) as i64,
                    ec_dec_icdf(ctx, icdf.as_ptr(), ftb) as i64,
                )
            }
            4 => {
                // The coarse energy model of the codec.
                let model = &E_PROB_MODEL[rng.below(4) as usize][rng.below(2) as usize];
                let i = 2 * rng.below(21) as usize;
                let (fs, decay) = ((model[i] as u32) << 7, (model[i + 1] as i32) << 6);
                (
                    format!("laplace_decode({fs}, {decay})"),
                    rc.laplace_decode(fs, decay) as i64,
                    ec_laplace_decode(ctx, fs, decay) as i64,
                )
            }
            5 => {
                // A symbol of `ft` values, then the update for an interval
                // that holds the decoded value.
                let ft = 2 + rng.below(40_000);
                let r = rc.decode(ft);
                let cv = ec_decode(ctx, ft);
                let fl = r - rng.below(r + 1).min(3);
                let fh = (r + 1 + rng.below(3)).min(ft);
                rc.update(fl, fh, ft);
                ec_dec_update(ctx, fl, fh, ft);
                (format!("decode({ft}) + update({fl}, {fh})"), r as i64, cv as i64)
            }
            _ => ("tell_frac".to_string(), rc.tell_frac() as i64, ec_tell_frac(ctx) as i64),
        }
    }
}

fn run(seed: u64, packets: usize, calls: usize, only_uint: bool) -> (usize, usize) {
    link_libopus();
    let mut rng = Lcg(seed);
    let mut total_calls = 0;
    let mut errors = 0;
    for p in 0..packets {
        let bytes = packet(&mut rng, p as u32);
        let mut rc = RangeCoder::new_decoder(&bytes);
        let mut c = CDec::new(&bytes);
        assert_eq!(words(&rc), c.words(), "packet {p}: after init");
        for k in 0..calls {
            let (what, rust, libopus) = step(&mut rng, &mut rc, &mut c, only_uint);
            assert_eq!(rust, libopus, "packet {p} ({} bytes), call {k}: {what}", bytes.len());
            assert_eq!(
                words(&rc),
                c.words(),
                "packet {p} ({} bytes), words after call {k}: {what}",
                bytes.len()
            );
            total_calls += 1;
        }
        errors += usize::from(rc.error != 0);
    }
    (total_calls, errors)
}

/// `dec_uint` alone: the value is limited to `ft - 1` and the error flag is
/// set, exactly when libopus does it.
#[test]
fn dec_uint_matches_ec_dec_uint_on_random_packets() {
    let (calls, errors) = run(1, 4000, 40, true);
    assert_eq!(calls, 160_000);
    // The limit must have been reached in many packets, or the test shows
    // nothing.
    assert!(errors > 500, "only {errors} packets reached the limit");
}

/// The smallest case: 257 values. On a packet of ones the high part decodes
/// to its top value 128 and the raw bit to 1, which combine to 257.
#[test]
fn dec_uint_of_257_on_a_packet_of_ones() {
    link_libopus();
    let bytes = [0xFFu8; 4];
    let mut rc = RangeCoder::new_decoder(&bytes);
    let mut c = CDec::new(&bytes);
    let libopus = unsafe { ec_dec_uint(&mut c.ctx, 257) };
    assert_eq!(libopus, 256);
    assert_eq!(c.ctx.error, 1);
    assert_eq!(rc.dec_uint(257), 256);
    assert_eq!(rc.error, 1);
    assert_eq!(words(&rc), c.words());
}

/// Every decoder primitive in random order.
#[test]
fn every_decoder_primitive_matches_libopus_on_random_packets() {
    let (calls, errors) = run(2, 6000, 60, false);
    assert_eq!(calls, 360_000);
    assert!(errors > 200, "only {errors} packets reached the limit of dec_uint");
}
