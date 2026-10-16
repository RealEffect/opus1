use crate::mathops::mac;
use crate::silk::macros::*;
use crate::silk::sigproc_fix::{silk_biquad_alt_stride1, silk_biquad_alt_stride2};

#[cfg(not(feature = "std"))]
use crate::compat::Math;

/// libopus `SILK_FIX_CONST(1.5 * pi / 1000, 19)` of `hp_cutoff` (opus_encoder.c).
const SILK_FIX_CONST_19: i32 = silk_fix_const(1.5 * core::f64::consts::PI / 1000.0, 19);

/// The Q28 biquad coefficients `B_Q28` and `A_Q28` of libopus `hp_cutoff`
/// (opus_encoder.c) for `cutoff_hz` at the sampling rate `fs`. The 16-bit
/// and the float filters use the same ones.
fn compute_hp_coeffs(cutoff_hz: i32, fs: i32) -> ([i32; 3], [i32; 2]) {
    let fc_q19 = silk_div32_16(silk_smulbb(SILK_FIX_CONST_19, cutoff_hz), fs / 1000);

    let r_q28 = (1i32 << 28) - silk_mul(471, fc_q19);

    let b_q28 = [r_q28, -silk_lshift(r_q28, 1), r_q28];

    let r_q22 = silk_rshift(r_q28, 6);
    let a_q28 = [
        silk_smulww(r_q22, silk_smulww(fc_q19, fc_q19) - (2i32 << 22)),
        silk_smulww(r_q22, r_q22),
    ];
    (b_q28, a_q28)
}

/// The 16-bit high-pass filter of the fixed-point build of libopus
/// (`hp_cutoff`, opus_encoder.c), on 16-bit input.
///
/// Mono filters the shorter of `input` and `output`. Stereo filters `len`
/// interleaved sample pairs. `hp_mem` holds 2 state values per channel.
pub fn hp_cutoff_i16(
    input: &[i16],
    cutoff_hz: i32,
    output: &mut [i16],
    hp_mem: &mut [i32],
    len: usize,
    channels: usize,
    fs: i32,
) {
    // A rate below 1 kHz would divide by zero in the coefficients. The
    // encoder never passes one; the output is then left as it is.
    if fs < 1000 {
        return;
    }
    let n = filtered_len(input.len(), output.len(), len, channels);
    output[..n].copy_from_slice(&input[..n]);
    hp_cutoff_in_place(&mut output[..n], cutoff_hz, hp_mem, channels, fs);
}

/// [`hp_cutoff_i16`] on float input (1.0 = 32768), which is first rounded
/// to 16 bits: half up, then limited to the 16-bit range.
pub fn hp_cutoff(
    input: &[f32],
    cutoff_hz: i32,
    output: &mut [i16],
    hp_mem: &mut [i32],
    len: usize,
    channels: usize,
    fs: i32,
) {
    // As in `hp_cutoff_i16`.
    if fs < 1000 {
        return;
    }
    let n = filtered_len(input.len(), output.len(), len, channels);
    float_to_i16_round(&input[..n], &mut output[..n]);
    hp_cutoff_in_place(&mut output[..n], cutoff_hz, hp_mem, channels, fs);
}

/// The samples that [`hp_cutoff_i16`] and [`hp_cutoff`] filter.
fn filtered_len(input_len: usize, output_len: usize, len: usize, channels: usize) -> usize {
    if channels == 1 {
        input_len.min(output_len)
    } else {
        2 * len
    }
}

/// The 16-bit high-pass filter on `samples`, which hold one channel or two
/// interleaved ones. `fs` is at least 1000.
fn hp_cutoff_in_place(
    samples: &mut [i16],
    cutoff_hz: i32,
    hp_mem: &mut [i32],
    channels: usize,
    fs: i32,
) {
    let (b_q28, a_q28) = compute_hp_coeffs(cutoff_hz, fs);
    if channels == 1 {
        let len = samples.len();
        silk_biquad_alt_stride1(samples, &b_q28, &a_q28, &mut hp_mem[..2], len);
    } else {
        let len = samples.len() / 2;
        silk_biquad_alt_stride2(samples, &b_q28, &a_q28, &mut hp_mem[..4], len);
    }
}

/// Convert float samples (1.0 = 32768) to 16-bit: round half up, then limit
/// to the 16-bit range. `output` must be as long as `input`.
#[inline]
fn float_to_i16_round(input: &[f32], output: &mut [i16]) {
    debug_assert_eq!(input.len(), output.len());
    for (dst, &src) in output.iter_mut().zip(input) {
        *dst = (src * 32768.0 + 0.5).floor().clamp(-32768.0, 32767.0) as i16;
    }
}

/// Direct-form-II-transposed second-order ARMA filter with **float** in/out,
/// mirroring the float build of C `silk_biquad_res()` (opus_encoder.c). Used by
/// the float `hp_cutoff_float` path so CELT sees the exact same pre-emphasis
/// samples as libopus 1.6.
///
/// `B_Q28`/`A_Q28` are Q28 coefficients; the state `S` is 2 elements per
/// channel (stride pass). Output is written with the same `stride` as the input.
///
/// The steps round as in libopus 1.6.1 on 64-bit ARM, whose disassembly
/// shows fused multiply-adds (`mac`): `vout = S0 + B0*in` is one,
/// `S0 = S1 - vout*A0 + B1*in` is two, and in `S1 = -vout*A1 + B2*in +
/// VERY_SMALL` the first product is fused into the sum with `B2*in`.
fn silk_biquad_res_float(
    input: &[f32],
    b_q28: &[i32; 3],
    a_q28: &[i32; 2],
    s: &mut [f32; 2],
    output: &mut [f32],
    len: usize,
    stride: usize,
) {
    let inv = 1.0f32 / ((1i32 << 28) as f32);
    let a0 = a_q28[0] as f32 * inv;
    let a1 = a_q28[1] as f32 * inv;
    let b0 = b_q28[0] as f32 * inv;
    let b1 = b_q28[1] as f32 * inv;
    let b2 = b_q28[2] as f32 * inv;

    for k in 0..len {
        let inval = input[k * stride];
        let vout = mac(s[0], b0, inval);
        s[0] = mac(mac(s[1], -vout, a0), b1, inval);
        s[1] = mac(b2 * inval, -vout, a1) + 1e-30;
        output[k * stride] = vout;
    }
}

/// Floating-point high-pass `hp_cutoff` (C `hp_cutoff()` float path,
/// opus_encoder.c:441). Writes float output; state is 4 floats.
pub fn hp_cutoff_float(
    input: &[f32],
    cutoff_hz: i32,
    output: &mut [f32],
    hp_mem: &mut [f32; 4],
    len: usize,
    channels: usize,
    fs: i32,
) {
    // As in `hp_cutoff_i16`.
    if fs < 1000 {
        return;
    }
    let (b_q28, a_q28) = compute_hp_coeffs(cutoff_hz, fs);

    if channels == 1 {
        let mut s = [hp_mem[0], hp_mem[1]];
        silk_biquad_res_float(input, &b_q28, &a_q28, &mut s, output, len, 1);
        hp_mem[0] = s[0];
        hp_mem[1] = s[1];
    } else {
        let mut s0 = [hp_mem[0], hp_mem[1]];
        silk_biquad_res_float(input, &b_q28, &a_q28, &mut s0, output, len, channels);
        hp_mem[0] = s0[0];
        hp_mem[1] = s0[1];
        let mut s1 = [hp_mem[2], hp_mem[3]];
        silk_biquad_res_float(
            &input[1..],
            &b_q28,
            &a_q28,
            &mut s1,
            &mut output[1..],
            len,
            channels,
        );
        hp_mem[2] = s1[0];
        hp_mem[3] = s1[1];
    }
}

/// DC-reject first-order high-pass filter (C `dc_reject()` float path,
/// opus_encoder.c:507). `VERY_SMALL` matches `arch.h` float builds (1e-30f).
/// Used for the CELT/hybrid input when the application is not VOIP.
///
/// The state update `coef*x + VERY_SMALL + coef2*m0` is two fused
/// multiply-adds in libopus on 64-bit ARM (`mac`), as the
/// disassembly of libopus 1.6.1 shows.
pub fn dc_reject_float(
    input: &[f32],
    cutoff_hz: i32,
    output: &mut [f32],
    hp_mem: &mut [f32; 4],
    len: usize,
    channels: usize,
    fs: i32,
) {
    let coef = 6.3f32 * cutoff_hz as f32 / fs as f32;
    let coef2 = 1.0 - coef;
    if channels == 2 {
        let mut m0 = hp_mem[0];
        let mut m2 = hp_mem[2];
        for i in 0..len {
            let x0 = input[2 * i];
            let x1 = input[2 * i + 1];
            let out0 = x0 - m0;
            let out1 = x1 - m2;
            m0 = mac(mac(1e-30, coef, x0), coef2, m0);
            m2 = mac(mac(1e-30, coef, x1), coef2, m2);
            output[2 * i] = out0;
            output[2 * i + 1] = out1;
        }
        hp_mem[0] = m0;
        hp_mem[2] = m2;
    } else {
        let mut m0 = hp_mem[0];
        for i in 0..len {
            let x = input[i];
            let y = x - m0;
            m0 = mac(mac(1e-30, coef, x), coef2, m0);
            output[i] = y;
        }
        hp_mem[0] = m0;
    }
}
