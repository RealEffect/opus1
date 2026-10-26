//! The pulse vector quantizer (`alg_quant`, `alg_unquant`, src/pvq.rs)
//! against the functions of the same name in libopus (celt/vq.c), called in
//! the static library the tests link.
//!
//! Both quantize the same band into range coders of 1024 bytes. The bytes,
//! the quantized band and the collapse mask must be equal. Then both decode
//! the bytes, which must give the band again.

use opus1::internals::pvq::{alg_quant, alg_unquant};
use opus1::internals::range_coder::RangeCoder;
use std::os::raw::c_int;

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
    fn ec_enc_init(this: *mut EcCtx, buf: *mut u8, size: u32);
    fn ec_enc_done(this: *mut EcCtx);
    fn ec_dec_init(this: *mut EcCtx, buf: *mut u8, size: u32);
    #[link_name = "alg_quant"]
    fn c_alg_quant(
        x: *mut f32,
        n: c_int,
        k: c_int,
        spread: c_int,
        b: c_int,
        enc: *mut EcCtx,
        gain: f32,
        resynth: c_int,
        arch: c_int,
    ) -> u32;
    #[link_name = "alg_unquant"]
    fn c_alg_unquant(x: *mut f32, n: c_int, k: c_int, spread: c_int, b: c_int, dec: *mut EcCtx, gain: f32) -> u32;
}

/// Size of the range coder buffers.
const BYTES: usize = 1024;

/// Quantizes and decodes `x` (`k` pulses, spreading `spread`, one block)
/// with opus1 and with libopus, and compares every result.
fn compare(x: &[f32], k: i32, spread: i32) {
    let n = x.len();
    // Makes sure the test binary links the static library that
    // `audiopus_sys` brings along.
    assert!(!unsafe { audiopus_sys::opus_get_version_string() }.is_null());

    let mut rc = RangeCoder::new_encoder(BYTES as u32);
    let mut x_quant = x.to_vec();
    let mask = alg_quant(&mut x_quant, n, k, spread, 1, &mut rc, 1.0, true);
    rc.done();

    let mut c_buf = vec![0u8; BYTES];
    let mut c_quant = x.to_vec();
    let mut ec: EcCtx = unsafe { std::mem::zeroed() };
    // SAFETY: the band holds `n` values and the buffer `BYTES` bytes.
    let c_mask = unsafe {
        ec_enc_init(&mut ec, c_buf.as_mut_ptr(), BYTES as u32);
        let mask = c_alg_quant(c_quant.as_mut_ptr(), n as c_int, k, spread, 1, &mut ec, 1.0, 1, 0);
        ec_enc_done(&mut ec);
        mask
    };
    assert_eq!(ec.error, 0, "libopus's coder failed");
    assert_eq!(&rc.buf[..BYTES], &c_buf[..], "bytes of N={n} K={k} spread={spread}");
    assert_eq!(x_quant, c_quant, "quantized band of N={n} K={k} spread={spread}");
    assert_eq!(mask, c_mask, "collapse mask");
    assert!(c_quant.iter().any(|&v| v != 0.0));

    let mut dec = RangeCoder::new_decoder(&c_buf);
    let mut x_unquant = vec![0.0f32; n];
    let unquant_mask = alg_unquant(&mut x_unquant, n, k, spread, 1, &mut dec, 1.0);
    let mut c_unquant = vec![0.0f32; n];
    // SAFETY: the band holds `n` values and the buffer `BYTES` bytes.
    let c_unquant_mask = unsafe {
        ec_dec_init(&mut ec, c_buf.as_mut_ptr(), BYTES as u32);
        c_alg_unquant(c_unquant.as_mut_ptr(), n as c_int, k, spread, 1, &mut ec, 1.0)
    };
    assert_eq!(x_unquant, c_unquant, "decoded band of N={n} K={k} spread={spread}");
    assert_eq!(unquant_mask, c_unquant_mask, "collapse mask of the decoder");
    // The decoder gives the band that the encoder quantized.
    assert_eq!(c_unquant, c_quant, "libopus's decoded band");
}

#[test]
fn test_pvq_match() {
    // No spreading.
    let x: Vec<f32> = (0..8).map(|i| (i as f32 * 0.5f32).sin()).collect();
    compare(&x, 10, 0);
}

#[test]
fn test_pvq_complex_match() {
    // Normal spreading (SPREAD_NORMAL), which rotates the band before and
    // after the search.
    let x: Vec<f32> = (0..16).map(|i| (i as f32 * 0.4f32).sin()).collect();
    compare(&x, 10, 2);
}
