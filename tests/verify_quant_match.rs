//! The coarse and fine energy quantizers (`quant_coarse_energy`,
//! `quant_fine_energy`, src/quant_bands.rs) against the functions of the
//! same name in libopus (celt/quant_bands.c), called in the static library
//! the tests link.
//!
//! Both get the same band energies, all previous energies at zero and a
//! range coder of 1000 bytes. The bytes of the two coders, the quantized
//! energies and the remaining errors must be equal.

use opus1::internals::modes::default_mode;
use opus1::internals::quant_bands::{quant_coarse_energy, quant_fine_energy};
use opus1::internals::range_coder::RangeCoder;
use std::os::raw::{c_int, c_void};

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
    fn opus_custom_mode_create(fs: i32, frame_size: c_int, error: *mut c_int) -> *const c_void;
    fn ec_enc_init(this: *mut EcCtx, buf: *mut u8, size: u32);
    fn ec_enc_done(this: *mut EcCtx);
    #[link_name = "quant_coarse_energy"]
    fn c_quant_coarse_energy(
        m: *const c_void,
        start: c_int,
        end: c_int,
        eff_end: c_int,
        e_bands: *const f32,
        old_e_bands: *mut f32,
        budget: u32,
        error: *mut f32,
        enc: *mut EcCtx,
        channels: c_int,
        lm: c_int,
        nb_available_bytes: c_int,
        force_intra: c_int,
        delayed_intra: *mut f32,
        two_pass: c_int,
        loss_rate: c_int,
        lfe: c_int,
    );
    #[link_name = "quant_fine_energy"]
    fn c_quant_fine_energy(
        m: *const c_void,
        start: c_int,
        end: c_int,
        old_e_bands: *mut f32,
        error: *mut f32,
        prev_quant: *const c_int,
        extra_quant: *const c_int,
        enc: *mut EcCtx,
        channels: c_int,
    );
}

/// The 48 kHz mode of libopus. The call also makes sure the test binary
/// links the static library that `audiopus_sys` brings along.
fn c_mode() -> *const c_void {
    let version = unsafe { audiopus_sys::opus_get_version_string() };
    assert!(!version.is_null());
    let mut err: c_int = -1;
    let mode = unsafe { opus_custom_mode_create(48000, 960, &mut err) };
    assert!(err == 0 && !mode.is_null(), "libopus has no 48 kHz mode: {err}");
    mode
}

#[test]
fn test_quant_energy_match() {
    let mode = default_mode();
    let nb_ebands = mode.nb_ebands;
    let e_bands: Vec<f32> = (0..nb_ebands).map(|i| 5.0 + (i as f32 * 0.5).sin() * 2.0).collect();
    let fine_quant: Vec<i32> = (0..nb_ebands).map(|i| (i % 3) as i32).collect();
    let (start, end, budget, channels, lm) = (0, nb_ebands, 10000u32, 1, 3);
    let nb_available_bytes = (budget / 8) as usize;
    // Without intra coding forced (`force_intra` 0 and `delayedIntra` 0
    // in libopus).
    let intra = false;

    let mut old_e_bands = vec![0.0f32; nb_ebands];
    let mut error = vec![0.0f32; nb_ebands];
    let mut enc = RangeCoder::new_encoder(1000);
    quant_coarse_energy(
        mode,
        start,
        end,
        &e_bands,
        &mut old_e_bands,
        budget,
        &mut error,
        &mut enc,
        channels,
        lm,
        intra,
        nb_available_bytes,
    );
    quant_fine_energy(mode, start, end, &mut old_e_bands, &mut error, &fine_quant, &mut enc, channels);
    enc.done();

    let mut c_old = vec![0.0f32; nb_ebands];
    let mut c_error = vec![0.0f32; nb_ebands];
    let mut c_buf = vec![0u8; 1000];
    let mut c_enc: EcCtx = unsafe { std::mem::zeroed() };
    let mut delayed_intra = 0.0f32;
    // SAFETY: the band arrays hold `nb_ebands` values for one channel, the
    // buffer 1000 bytes, and the mode is libopus's own.
    unsafe {
        let m = c_mode();
        ec_enc_init(&mut c_enc, c_buf.as_mut_ptr(), 1000);
        c_quant_coarse_energy(
            m,
            start as c_int,
            end as c_int,
            end as c_int,
            e_bands.as_ptr(),
            c_old.as_mut_ptr(),
            budget,
            c_error.as_mut_ptr(),
            &mut c_enc,
            channels as c_int,
            lm as c_int,
            nb_available_bytes as c_int,
            intra as c_int,
            &mut delayed_intra,
            0,
            0,
            0,
        );
        c_quant_fine_energy(
            m,
            start as c_int,
            end as c_int,
            c_old.as_mut_ptr(),
            c_error.as_mut_ptr(),
            std::ptr::null(),
            fine_quant.as_ptr(),
            &mut c_enc,
            channels as c_int,
        );
        ec_enc_done(&mut c_enc);
    }

    assert_eq!(c_enc.error, 0, "libopus's coder ran out of room");
    assert!(c_enc.offs > 0 && c_enc.end_offs > 0, "libopus wrote range coder bytes and raw bits");
    assert_eq!(&enc.buf[..1000], &c_buf[..], "bytes");
    assert_eq!((enc.offs, enc.end_offs), (c_enc.offs, c_enc.end_offs), "bytes written at the front and the end");
    assert_eq!(old_e_bands, c_old, "quantized energies");
    assert_eq!(error, c_error, "remaining errors");
}
