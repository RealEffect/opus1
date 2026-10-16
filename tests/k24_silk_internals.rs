//! Differential tests of SILK decoder building blocks against the same
//! functions inside C libopus.
//!
//! libopus is linked as a static library, so its internal (non-`static`)
//! functions and tables can be called directly. Each test feeds the same
//! input to the libopus function and to the opus1 port and requires
//! bit-identical output, as SILK is fixed-point.

mod k24_common;

use opus1::silk::nlsf::silk_nlsf2a;
use opus1::silk::nlsf_decode::silk_nlsf_decode;
use opus1::silk::tables_nlsf::{SILK_NLSF_CB_NB_MB, SILK_NLSF_CB_WB};
use std::os::raw::{c_int, c_void};

unsafe extern "C" {
    static silk_NLSF_CB_NB_MB: u8;
    static silk_NLSF_CB_WB: u8;
    fn silk_NLSF_decode(nlsf_q15: *mut i16, indices: *mut i8, cb: *const c_void);
    fn silk_NLSF2A(a_q12: *mut i16, nlsf: *const i16, d: c_int, arch: c_int);
    static silk_LTP_vq_ptrs_Q7: [*const i8; 3];
}

/// libopus range decoder state (`struct ec_ctx`, celt/entcode.h).
#[repr(C)]
struct EcDec {
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
    fn ec_dec_init(dec: *mut EcDec, buf: *mut u8, storage: u32);
    fn silk_decode_pulses(
        dec: *mut EcDec,
        pulses: *mut i16,
        signal_type: c_int,
        quant_offset_type: c_int,
        frame_length: c_int,
    );
}

/// `silk_decode_pulses` on random bit streams: any byte string decodes to
/// some excitation, so random bytes reach every branch, including the rare
/// ones (extra least-significant bits for very large pulses).
#[test]
fn decode_pulses_matches_libopus() {
    use opus1::range_coder::RangeCoder;
    use opus1::silk::decode_pulses::silk_decode_pulses as rs_decode_pulses;
    let mut rng = Lcg(3);
    for trial in 0..20000 {
        // Bias some streams towards 0x00 / 0xFF bytes, which decode to the
        // ends of each probability table (large pulse counts).
        let style = trial % 4;
        let mut bytes: Vec<u8> = (0..200)
            .map(|_| match style {
                0 => rng.next() as u8,
                1 => (rng.next() % 8) as u8,
                2 => 255 - (rng.next() % 8) as u8,
                _ => if rng.next() % 3 == 0 { rng.next() as u8 } else { 0 },
            })
            .collect();
        let signal_type = rng.range(0, 2);
        let quant_offset_type = rng.range(0, 1);
        let frame_length = [80, 120, 160, 240, 320][(rng.next() % 5) as usize];

        let mut c_pulses = [0i16; 320 + 16];
        let mut dec = std::mem::MaybeUninit::<EcDec>::uninit();
        let c_tell = unsafe {
            ec_dec_init(dec.as_mut_ptr(), bytes.as_mut_ptr(), bytes.len() as u32);
            silk_decode_pulses(
                dec.as_mut_ptr(),
                c_pulses.as_mut_ptr(),
                signal_type,
                quant_offset_type,
                frame_length,
            );
            let d = dec.assume_init_ref();
            (d.nbits_total, d.rng)
        };

        let mut rs_pulses = [0i16; 320 + 16];
        let mut rc = RangeCoder::new_decoder(&bytes);
        rs_decode_pulses(&mut rc, &mut rs_pulses, signal_type, quant_offset_type, frame_length);
        let n = frame_length as usize;
        assert_eq!(
            rs_pulses[..n],
            c_pulses[..n],
            "pulses differ: trial {trial} type {signal_type} offset {quant_offset_type} length {frame_length}"
        );
        assert_eq!((rc.nbits_total, rc.rng), c_tell, "range decoder state differs: trial {trial}");
    }
}

unsafe extern "C" {
    fn silk_LPC_inverse_pred_gain_c(a_q12: *const i16, order: c_int) -> i32;
}

/// Prediction filter (Q12) with all poles at radius `r`: `pairs` complex
/// pole pairs at the given angles.
fn resonant_filter(r: f64, angles: &[f64]) -> Vec<i16> {
    // Polynomial 1 - sum a[k] z^-k = prod (1 - 2 r cos(w) z^-1 + r^2 z^-2).
    let mut poly = vec![1.0f64];
    for w in angles {
        let section = [1.0, -2.0 * r * w.cos(), r * r];
        let mut next = vec![0.0; poly.len() + 2];
        for (i, p) in poly.iter().enumerate() {
            for (j, s) in section.iter().enumerate() {
                next[i + j] += p * s;
            }
        }
        poly = next;
    }
    poly[1..]
        .iter()
        .map(|c| (-c * 4096.0).round().clamp(-32768.0, 32767.0) as i16)
        .collect()
}

/// `silk_LPC_inverse_pred_gain` decides whether a prediction filter is
/// stable. `silk_NLSF2A` widens the filter's bandwidth until this function
/// accepts it, so a different verdict changes the decoded filter.
///
/// The stability limit on each reflection coefficient is 0.99975 in Q24,
/// which is 16773022 (libopus `A_LIMIT`). opus1 had 16773043, so it
/// accepted filters with a reflection coefficient between the two values
/// that libopus rejects. A pure tone produces exactly such filters: a
/// narrowband chirp decoded to -3 dB against libopus from the first such
/// frame on.
#[test]
fn lpc_inverse_pred_gain_matches_libopus() {
    use opus1::silk::lpc_analysis::silk_lpc_inverse_pred_gain;
    // The filter of the first diverging frame of the chirp stream.
    let from_chirp: [i16; 10] = [5782, -8287, 4351, -2408, 157, -223, -45, -302, 104, -250];
    let c = unsafe { silk_LPC_inverse_pred_gain_c(from_chirp.as_ptr(), 10) };
    assert_eq!(silk_lpc_inverse_pred_gain(&from_chirp, 10), c, "chirp filter");

    let mut rng = Lcg(11);
    let (mut accepted, mut rejected) = (0, 0);
    for trial in 0..200_000 {
        let order = if trial % 2 == 0 { 10 } else { 16 };
        // Poles close to the unit circle, on both sides of the limit.
        let r = 0.9990 + (rng.next() % 1200) as f64 * 1e-6;
        let strong = 1 + (rng.next() % 2) as usize;
        let angles: Vec<f64> = (0..order / 2)
            .map(|i| (rng.next() % 3000) as f64 * 1e-3 + 0.05 + if i < strong { 0.0 } else { 0.01 })
            .collect();
        // Only the first pole pairs sit at radius r; the rest are damped.
        let mut a = resonant_filter(r, &angles[..strong]);
        let damped = resonant_filter(0.5, &angles[strong..]);
        // Combine by polynomial multiplication in floating point.
        let to_poly = |v: &[i16]| -> Vec<f64> {
            std::iter::once(1.0).chain(v.iter().map(|x| -(*x as f64) / 4096.0)).collect()
        };
        let (pa, pb) = (to_poly(&a), to_poly(&damped));
        let mut prod = vec![0.0; pa.len() + pb.len() - 1];
        for (i, x) in pa.iter().enumerate() {
            for (j, y) in pb.iter().enumerate() {
                prod[i + j] += x * y;
            }
        }
        a = prod[1..]
            .iter()
            .map(|c| (-c * 4096.0).round().clamp(-32768.0, 32767.0) as i16)
            .collect();
        assert_eq!(a.len(), order);
        let c = unsafe { silk_LPC_inverse_pred_gain_c(a.as_ptr(), order as c_int) };
        let rs = silk_lpc_inverse_pred_gain(&a, order);
        assert_eq!(rs, c, "trial {trial}: order {order} a = {a:?}");
        if c == 0 {
            rejected += 1;
        } else {
            accepted += 1;
        }
    }
    println!("{accepted} filters accepted, {rejected} rejected, all verdicts identical");
    assert!(accepted > 1000 && rejected > 1000, "test must cover both verdicts");
}

unsafe extern "C" {
    fn silk_resampler_init(state: *mut u8, fs_in: i32, fs_out: i32, for_enc: c_int) -> c_int;
    fn silk_resampler(state: *mut u8, out: *mut i16, input: *const i16, in_len: i32) -> c_int;
}

/// The SILK resampler, every rate pair of both directions, in 10 ms and
/// 20 ms calls, on noise and on a loud tone.
///
/// The decoder direction includes down-sampling (12 or 16 kHz SILK frames
/// into an 8 or 12 kHz decoder). opus1 set the down-sampler up for the
/// encoder only; the decoder's down-sampling ran through the up-sampling
/// method and produced a different signal.
#[test]
fn resampler_matches_libopus_for_every_rate_pair() {
    use opus1::silk::resampler::SilkResampler;
    let mut rng = Lcg(5);
    let mut pairs = Vec::new();
    for fs_in in [8000, 12000, 16000] {
        for fs_out in [8000, 12000, 16000, 24000, 48000] {
            pairs.push((fs_in, fs_out, false));
        }
    }
    for fs_in in [8000, 12000, 16000, 24000, 48000] {
        for fs_out in [8000, 12000, 16000] {
            pairs.push((fs_in, fs_out, true));
        }
    }
    for (fs_in, fs_out, for_enc) in pairs {
        for block_ms in [10usize, 20] {
            // libopus state: opaque, far smaller than this buffer.
            let mut c_state = vec![0u8; 4096];
            assert_eq!(
                unsafe { silk_resampler_init(c_state.as_mut_ptr(), fs_in, fs_out, for_enc as c_int) },
                0
            );
            let mut rs = SilkResampler::default();
            let ret = if for_enc { rs.init_for_enc(fs_in, fs_out) } else { rs.init(fs_in, fs_out) };
            assert_eq!(ret, 0, "init {fs_in} -> {fs_out} for_enc={for_enc}");

            let n_in = fs_in as usize / 1000 * block_ms;
            let n_out = fs_out as usize / 1000 * block_ms;
            for block in 0..30 {
                let input: Vec<i16> = (0..n_in)
                    .map(|i| {
                        if block < 15 {
                            rng.range(-20000, 20000) as i16
                        } else {
                            let t = (block * n_in + i) as f64 / fs_in as f64;
                            (30000.0 * (2.0 * std::f64::consts::PI * 3100.0 * t).sin()) as i16
                        }
                    })
                    .collect();
                let mut c_out = vec![0i16; n_out];
                unsafe { silk_resampler(c_state.as_mut_ptr(), c_out.as_mut_ptr(), input.as_ptr(), n_in as i32) };
                let mut rs_out = vec![0i16; n_out];
                rs.process(&mut rs_out, &input, n_in as i32);
                assert_eq!(
                    rs_out, c_out,
                    "{fs_in} -> {fs_out}, for_enc={for_enc}, {block_ms} ms blocks, block {block}"
                );
            }
        }
    }
}

/// Small deterministic pseudo-random generator (no dependency on `rand`).
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    fn range(&mut self, lo: i32, hi: i32) -> i32 {
        lo + (self.next() % (hi - lo + 1) as u32) as i32
    }
}

/// `silk_NLSF_decode` followed by `silk_NLSF2A`, for every first-stage
/// codebook entry and many residual index vectors, including the extreme
/// residual values (+-10) that only very tonal signals produce.
#[test]
fn nlsf_decode_and_nlsf2a_match_libopus() {
    let mut rng = Lcg(1);
    let mut checked = 0;
    for (wb, order) in [(false, 10usize), (true, 16usize)] {
        let (rs_cb, c_cb) = if wb {
            (&SILK_NLSF_CB_WB, unsafe { &silk_NLSF_CB_WB as *const u8 })
        } else {
            (&SILK_NLSF_CB_NB_MB, unsafe {
                &silk_NLSF_CB_NB_MB as *const u8
            })
        };
        for first in 0..32i8 {
            for trial in 0..400 {
                let amp = [1, 2, 4, 10][trial % 4];
                let mut indices = [0i8; 17];
                indices[0] = first;
                for v in indices[1..=order].iter_mut() {
                    *v = rng.range(-amp, amp) as i8;
                }
                let mut c_nlsf = [0i16; 16];
                let mut c_idx = indices;
                unsafe { silk_NLSF_decode(c_nlsf.as_mut_ptr(), c_idx.as_mut_ptr(), c_cb.cast()) };
                let mut rs_nlsf = [0i16; 16];
                silk_nlsf_decode(&mut rs_nlsf, &indices, rs_cb);
                assert_eq!(
                    rs_nlsf[..order],
                    c_nlsf[..order],
                    "silk_NLSF_decode differs: wb={wb} indices={:?}",
                    &indices[..=order]
                );

                let mut c_a = [0i16; 16];
                unsafe { silk_NLSF2A(c_a.as_mut_ptr(), c_nlsf.as_ptr(), order as c_int, 0) };
                let mut rs_a = [0i16; 16];
                silk_nlsf2a(&mut rs_a, &rs_nlsf, order);
                assert_eq!(
                    rs_a[..order],
                    c_a[..order],
                    "silk_NLSF2A differs: wb={wb} nlsf={:?}",
                    &rs_nlsf[..order]
                );
                checked += 1;
            }
        }
    }
    println!("{checked} NLSF vectors identical");
}

/// `silk_NLSF2A` on line-spectral-frequency vectors with clustered values,
/// which describe a filter with a sharp resonance (a pure tone). Those are
/// the vectors that reach the stabilising branches of the conversion.
#[test]
fn nlsf2a_matches_libopus_on_resonant_filters() {
    let mut rng = Lcg(7);
    for order in [10usize, 16] {
        for _ in 0..20000 {
            // A few tight clusters plus spread values, sorted and kept
            // strictly increasing as the decoder guarantees.
            let mut v: Vec<i32> = Vec::new();
            let centre = rng.range(300, 32000);
            while v.len() < order {
                let x = if rng.next() % 3 == 0 {
                    rng.range(100, 32600)
                } else {
                    centre + rng.range(-200, 200)
                };
                v.push(x.clamp(10, 32750));
            }
            v.sort();
            for i in 1..order {
                if v[i] <= v[i - 1] {
                    v[i] = v[i - 1] + 1 + (rng.next() % 4) as i32;
                }
            }
            if v[order - 1] > 32760 {
                continue;
            }
            let nlsf: Vec<i16> = v.iter().map(|x| *x as i16).collect();
            let mut c_a = [0i16; 16];
            unsafe { silk_NLSF2A(c_a.as_mut_ptr(), nlsf.as_ptr(), order as c_int, 0) };
            let mut rs_a = [0i16; 16];
            silk_nlsf2a(&mut rs_a, &nlsf, order);
            assert_eq!(rs_a[..order], c_a[..order], "silk_NLSF2A differs: nlsf={nlsf:?}");
        }
    }
}

/// The three long-term-prediction gain codebooks.
#[test]
fn ltp_gain_codebooks_match_libopus() {
    use opus1::silk::tables::SILK_LTP_VQ_PTRS_Q7;
    for (cb, size) in [8usize, 16, 32].into_iter().enumerate() {
        let c = unsafe { std::slice::from_raw_parts(silk_LTP_vq_ptrs_Q7[cb], size * 5) };
        for entry in 0..size {
            for tap in 0..5 {
                assert_eq!(
                    SILK_LTP_VQ_PTRS_Q7[cb][entry][tap],
                    c[entry * 5 + tap],
                    "codebook {cb} entry {entry} tap {tap}"
                );
            }
        }
    }
}
