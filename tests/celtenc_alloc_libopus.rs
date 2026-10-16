//! The bit allocation of CELT (`clt_compute_allocation`) against the same
//! function of C libopus, called directly on the same input.
//!
//! The tests link libopus as a static library, which keeps the names of its
//! internal functions. Both sides get identical values, so every result must
//! be identical: the return value, every output array, and the bytes the
//! range coder writes or the values it reads.
//!
//! The inputs cover what the encoder and the decoder pass (channels, frame
//! sizes, band ranges of CELT and Hybrid, trims, dynamic offsets, caps and
//! budgets), and some values past that (budgets of 0 and very large ones).

mod celtenc_common;
use celtenc_common::Noise;

use opus1::internals::modes::{CeltMode, default_mode};
use opus1::internals::range_coder::RangeCoder;
use opus1::internals::rate;
use std::os::raw::{c_int, c_uchar, c_void};

/// libopus `ec_ctx` (celt/entcode.h).
#[repr(C)]
struct EcCtx {
    buf: *mut c_uchar,
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
    fn ec_enc_init(enc: *mut EcCtx, buf: *mut c_uchar, size: u32);
    fn ec_enc_done(enc: *mut EcCtx);
    fn ec_dec_init(dec: *mut EcCtx, buf: *mut c_uchar, size: u32);
    fn clt_compute_allocation(
        m: *const c_void,
        start: c_int,
        end: c_int,
        offsets: *const c_int,
        cap: *const c_int,
        alloc_trim: c_int,
        intensity: *mut c_int,
        dual_stereo: *mut c_int,
        total: i32,
        balance: *mut i32,
        pulses: *mut c_int,
        ebits: *mut c_int,
        fine_priority: *mut c_int,
        c: c_int,
        lm: c_int,
        ec: *mut EcCtx,
        encode: c_int,
        prev: c_int,
        signal_bandwidth: c_int,
    ) -> c_int;
}

/// The 48 kHz mode of libopus, the one its encoder uses for every rate.
fn c_mode() -> *const c_void {
    let mut err = 0;
    let m = unsafe { opus_custom_mode_create(48_000, 960, &mut err) };
    assert!(!m.is_null() && err == 0, "opus_custom_mode_create");
    m
}

/// The inputs of one call.
#[derive(Clone, Debug)]
struct Input {
    start: usize,
    end: usize,
    offsets: Vec<i32>,
    cap: Vec<i32>,
    alloc_trim: i32,
    intensity: i32,
    dual_stereo: i32,
    total: i32,
    channels: i32,
    lm: i32,
    prev: i32,
    signal_bandwidth: i32,
    bytes: usize,
}

/// What one call returns and writes.
#[derive(Debug, PartialEq)]
struct Output {
    coded_bands: i32,
    intensity: i32,
    dual_stereo: i32,
    balance: i32,
    pulses: Vec<i32>,
    ebits: Vec<i32>,
    fine_priority: Vec<i32>,
    /// The coder's bytes after `done` when encoding; the bits it consumed
    /// when decoding.
    coder: Vec<u8>,
    tell: i32,
}

fn uniform(noise: &mut Noise, lo: i32, hi: i32) -> i32 {
    // `Noise::next` is in [-1, 1).
    let u = (noise.next() + 1.0) * 0.5;
    (lo + (u * (hi - lo + 1) as f32) as i32).clamp(lo, hi)
}

/// A random input as the encoder or the decoder could pass it.
fn random_input(m: &CeltMode, noise: &mut Noise, round: usize) -> Input {
    let channels = 1 + (round % 2) as i32;
    let lm = ((round / 2) % 4) as i32;
    let hybrid = round % 7 == 3;
    let start = if hybrid { 17 } else { 0 };
    let end = if hybrid { 21 } else { [13, 17, 19, 21, 21, 21][round % 6] };
    // libopus `init_caps`, scaled up or down now and then.
    let cap: Vec<i32> = (0..21)
        .map(|i| {
            let n = (m.e_bands[i + 1] - m.e_bands[i]) as i32;
            let caps = m.cache.caps[21 * (2 * lm as usize + channels as usize - 1) + i] as i32;
            let cap = ((caps + 64) * channels * n) >> 2;
            match round % 11 {
                5 => cap / 3,
                6 => cap * 2,
                _ => cap,
            }
        })
        .collect();
    // Dynamic allocation boosts on a few bands.
    let offsets: Vec<i32> = (0..21)
        .map(|_| {
            if uniform(noise, 0, 5) == 0 {
                uniform(noise, 1, 8 * 6 * channels) << 3
            } else {
                0
            }
        })
        .collect();
    let bytes = uniform(noise, 2, 320) as usize;
    let used = uniform(noise, 1, bytes as i32 * 8 / 3);
    let total = match round % 13 {
        0 => 0,
        1 => uniform(noise, 0, 64),
        _ => ((bytes as i32 * 8) << 3) - (used << 3) - 1,
    };
    Input {
        start,
        end,
        offsets,
        cap,
        alloc_trim: uniform(noise, 0, 10),
        intensity: uniform(noise, start as i32, end as i32),
        dual_stereo: uniform(noise, 0, 1),
        total,
        channels,
        lm,
        prev: uniform(noise, 0, 21),
        signal_bandwidth: uniform(noise, 0, 20),
        bytes,
    }
}

fn run_rust(m: &CeltMode, input: &Input, encode: bool, data: &[u8]) -> Output {
    let mut rc = if encode {
        RangeCoder::new_encoder(input.bytes as u32)
    } else {
        RangeCoder::new_decoder(data)
    };
    let mut intensity = input.intensity;
    let mut dual_stereo = input.dual_stereo;
    let mut balance = 0;
    let mut pulses = vec![0; 21];
    let mut ebits = vec![0; 21];
    let mut fine_priority = vec![0; 21];
    let coded_bands = rate::clt_compute_allocation(
        m,
        input.start,
        input.end,
        &input.offsets,
        &input.cap,
        input.alloc_trim,
        &mut intensity,
        &mut dual_stereo,
        input.total,
        &mut balance,
        &mut pulses,
        &mut ebits,
        &mut fine_priority,
        input.channels,
        input.lm,
        &mut rc,
        encode,
        input.prev,
        input.signal_bandwidth,
    );
    let tell = rc.tell_frac();
    let coder = if encode {
        rc.done();
        rc.buf[..input.bytes].to_vec()
    } else {
        Vec::new()
    };
    Output {
        coded_bands,
        intensity,
        dual_stereo,
        balance,
        pulses,
        ebits,
        fine_priority,
        coder,
        tell,
    }
}

fn run_c(input: &Input, encode: bool, data: &[u8]) -> Output {
    let mut buf = if encode {
        vec![0u8; input.bytes]
    } else {
        data.to_vec()
    };
    let mut ec: EcCtx = unsafe { std::mem::zeroed() };
    unsafe {
        if encode {
            ec_enc_init(&mut ec, buf.as_mut_ptr(), buf.len() as u32);
        } else {
            ec_dec_init(&mut ec, buf.as_mut_ptr(), buf.len() as u32);
        }
    }
    let mut intensity = input.intensity;
    let mut dual_stereo = input.dual_stereo;
    let mut balance = 0;
    let mut pulses = vec![0; 21];
    let mut ebits = vec![0; 21];
    let mut fine_priority = vec![0; 21];
    let coded_bands = unsafe {
        clt_compute_allocation(
            c_mode(),
            input.start as c_int,
            input.end as c_int,
            input.offsets.as_ptr(),
            input.cap.as_ptr(),
            input.alloc_trim,
            &mut intensity,
            &mut dual_stereo,
            input.total,
            &mut balance,
            pulses.as_mut_ptr(),
            ebits.as_mut_ptr(),
            fine_priority.as_mut_ptr(),
            input.channels,
            input.lm,
            &mut ec,
            encode as c_int,
            input.prev,
            input.signal_bandwidth,
        )
    };
    // C `ec_tell_frac`.
    let tell = {
        const CORRECTION: [u32; 8] = [35733, 38967, 42495, 46340, 50535, 55109, 60097, 65535];
        let nbits = (ec.nbits_total as u32) << 3;
        let l = 32 - ec.rng.leading_zeros() as i32;
        let r = ec.rng >> (l - 16);
        let mut b = (r >> 12) - 8;
        b += (r > CORRECTION[b as usize]) as u32;
        let l = ((l as u32) << 3) + b;
        (nbits - l) as i32
    };
    let coder = if encode {
        unsafe { ec_enc_done(&mut ec) };
        buf.clone()
    } else {
        Vec::new()
    };
    Output {
        coded_bands,
        intensity,
        dual_stereo,
        balance,
        pulses,
        ebits,
        fine_priority,
        coder,
        tell,
    }
}

/// Encoding: the same allocation and the same bytes as libopus.
#[test]
fn allocation_encodes_as_libopus_does() {
    let m = default_mode();
    let mut noise = Noise(21);
    let mut coded = 0;
    for round in 0..20_000 {
        let input = random_input(m, &mut noise, round);
        let got = run_rust(m, &input, true, &[]);
        let want = run_c(&input, true, &[]);
        assert_eq!(got, want, "round {round}, {input:?}");
        coded += got.coded_bands as usize;
    }
    assert!(coded > 100_000);
}

/// Decoding random bytes: the same allocation and the same position in the
/// stream as libopus.
#[test]
fn allocation_decodes_as_libopus_does() {
    let m = default_mode();
    let mut noise = Noise(22);
    for round in 0..20_000 {
        let input = random_input(m, &mut noise, round);
        let data: Vec<u8> = (0..input.bytes).map(|_| uniform(&mut noise, 0, 255) as u8).collect();
        let got = run_rust(m, &input, false, &data);
        let want = run_c(&input, false, &data);
        assert_eq!(got, want, "round {round}, {input:?}");
    }
}
