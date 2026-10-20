//! The scalar arithmetic that OSCE shares: fused multiply-adds, the calls
//! into the C math library, and the fast approximations of
//! `celt/mathops.h` that `nndsp.c` and `osce.c` use.
//!
//! Every function computes the bits of libopus 1.6.1 on 64-bit ARM, built
//! with clang at `-O2 -fno-vectorize -fno-slp-vectorize`, on every target:
//!
//! - clang contracts `a * b + c` in one C expression into a fused
//!   multiply-add (`fma`). The callers use `fma` exactly where the object
//!   code of the reference build has `fmadd`, which was read from its
//!   disassembly.
//! - A `double` literal or a `double` libm call makes the C expression a
//!   `double` one. The callers convert to `f64` at the same places.
//! - `exp`, `log`, `sin`, `cos` are the double-precision functions of the
//!   platform's C library with `std`, and of the `libm` crate without it.
//!   The latter can differ in the last bit for rare inputs.

#[cfg(not(feature = "std"))]
use crate::compat::Math;

pub use crate::dnn::vec_portable::fma;

/// `a * b + c` in `f64` with one rounding.
#[inline(always)]
pub fn fma64(a: f64, b: f64, c: f64) -> f64 {
    crate::dnn::freq::fma(a, b, c)
}

/// C `exp` (double precision).
#[inline(always)]
pub fn exp(x: f64) -> f64 {
    x.exp()
}

/// C `log` (double precision).
#[inline(always)]
pub fn ln(x: f64) -> f64 {
    x.ln()
}

/// C `sin` (double precision).
#[inline(always)]
pub fn sin(x: f64) -> f64 {
    x.sin()
}

/// C `cos` (double precision).
#[inline(always)]
pub fn cos(x: f64) -> f64 {
    x.cos()
}

/// C `sqrt` (double precision). It is correctly rounded on every target.
#[inline(always)]
pub fn sqrt64(x: f64) -> f64 {
    x.sqrt()
}

/// C `sqrtf`. It is correctly rounded on every target.
#[inline(always)]
pub fn sqrt32(x: f32) -> f32 {
    x.sqrt()
}

/// C `floor` (double precision).
#[cfg(feature = "osce-bwe")]
#[inline(always)]
pub fn floor64(x: f64) -> f64 {
    x.floor()
}

/// libopus `float2int` on 64-bit ARM (`vcvtns_s32_f32`): rounds to the
/// nearest integer, ties to even. NaN gives 0, as the instruction does.
/// The callers clamp the value to ±32767 first.
#[inline(always)]
pub fn float2int(x: f32) -> i32 {
    x.round_ties_even() as i32
}

/// libopus `celt_log2` with `FLOAT_APPROX` (`celt/mathops.h:271-302`),
/// with the fused multiply-adds of the 64-bit ARM build on every target.
/// (`crate::mathops::celt_log2` fuses only on 64-bit ARM, as the CELT code
/// needs.) Like libopus, it does not handle 0, negative values, infinities
/// or NaN.
// The constants keep the digits of the C source.
#[allow(clippy::excessive_precision)]
#[inline]
pub fn celt_log2(x: f32) -> f32 {
    const X_NORM: [f32; 8] = [
        1.000000000000000000000000000,
        8.88888895511627197265625e-01,
        8.00000000000000000000000e-01,
        7.27272748947143554687500e-01,
        6.66666686534881591796875e-01,
        6.15384638309478759765625e-01,
        5.71428596973419189453125e-01,
        5.33333361148834228515625e-01,
    ];
    const Y_NORM: [f32; 8] = [
        0.0000000000000000000000000000,
        1.699250042438507080078125e-01,
        3.219280838966369628906250e-01,
        4.594316184520721435546875e-01,
        5.849624872207641601562500e-01,
        7.004396915435791015625000e-01,
        8.073549270629882812500000e-01,
        9.068905711174011230468750e-01,
    ];
    const A0: f32 = 8.74628424644470214843750000e-02;
    const A1: f32 = 1.357829570770263671875000000000;
    const A2: f32 = -6.3897705078125000000000000e-01;
    const A3: f32 = 4.01971250772476196289062500e-01;
    const A4: f32 = -2.8415444493293762207031250e-01;
    let bits = x.to_bits() as i32;
    let integer = (bits >> 23) - 127;
    let bits = bits.wrapping_sub(((integer as u32) << 23) as i32);
    let range_idx = ((bits >> 20) & 0x7) as usize;
    let m = fma(f32::from_bits(bits as u32), X_NORM[range_idx], -1.0625);
    let p = fma(m, A4, A3);
    let p = fma(m, p, A2);
    let p = fma(m, p, A1);
    let p = fma(m, p, A0);
    // `integer + in.f + log2_y_norm_coeff[range_idx]`: two plain additions.
    integer as f32 + p + Y_NORM[range_idx]
}

/// libopus `celt_log` with `FLOAT_APPROX`: `celt_log2(x) * 0.6931471805599453f`
/// (that literal is the float `LN_2`).
#[inline]
pub fn celt_log(x: f32) -> f32 {
    celt_log2(x) * core::f32::consts::LN_2
}

/// libopus `celt_cos_norm2` (`celt/mathops.h:192-220`): an approximation
/// of `cos(π/2 · x)` with a polynomial in `x²`.
///
/// The roundings of the reference build: `x + 1` in float; the product
/// with 0.25 and the floor in double; `x - 4·floor(..)` as a double fused
/// multiply-add, rounded to float; the polynomial as a chain of float fused
/// multiply-adds; the sign as a float product.
// The constants keep the digits of the C source.
#[allow(clippy::excessive_precision)]
#[cfg(feature = "osce-bwe")]
#[inline]
pub fn celt_cos_norm2(x: f32) -> f32 {
    const A0: f32 = 9.999999403953552246093750000000e-01;
    const A2: f32 = -1.233698248863220214843750000000000;
    const A4: f32 = 2.536507546901702880859375000000e-01;
    const A6: f32 = -2.08106283098459243774414062500e-02;
    const A8: f32 = 8.581906440667808055877685546875e-04;
    // Restrict x to [-1, 3].
    let fl = floor64(0.25 * (x + 1.0) as f64);
    let x = fma64(fl, -4.0, x as f64) as f32;
    // Negative sign for [1, 3], and restrict to [-1, 1].
    let above = x > 1.0;
    let output_sign: f32 = if above { -1.0 } else { 1.0 };
    let x = x - if above { 2.0 } else { 0.0 };
    let x_norm_sq = x * x;
    let p = fma(x_norm_sq, A8, A6);
    let p = fma(x_norm_sq, p, A4);
    let p = fma(x_norm_sq, p, A2);
    let p = fma(x_norm_sq, p, A0);
    p * output_sign
}

/// libopus `celt_sin(x)`: `celt_cos_norm2((0.5f*PI) * x - 1.0f)`, with the
/// argument computed as a double fused multiply-add (`PI` is a double).
/// Note that this is `cos(π/2 · (π/2 · x - 1))`, not `sin(x)`; BBWENet was
/// trained with it.
#[cfg(feature = "osce-bwe")]
#[inline]
pub fn celt_sin(x: f32) -> f32 {
    // `0.5f*PI`, with `PI` the double 3.1415926535897931.
    celt_cos_norm2(fma64(x as f64, core::f64::consts::FRAC_PI_2, -1.0) as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn float2int_rounds_ties_to_even() {
        assert_eq!(float2int(0.5), 0);
        assert_eq!(float2int(1.5), 2);
        assert_eq!(float2int(-2.5), -2);
        assert_eq!(float2int(-32767.0), -32767);
        assert_eq!(float2int(f32::NAN), 0);
    }

    #[test]
    fn celt_log2_is_close_to_log2() {
        for &x in &[1e-6f32, 0.001, 0.5, 1.0, 1.5, 2.0, 3.7, 1000.0, 65535.0] {
            let d = celt_log2(x) - x.log2();
            assert!(d.abs() < 1e-4, "{x}: {d}");
        }
    }

    #[cfg(feature = "osce-bwe")]
    #[test]
    fn celt_cos_norm2_is_close_to_cosine() {
        for i in -400..400 {
            let x = i as f32 / 100.0;
            let want = (core::f64::consts::FRAC_PI_2 * x as f64).cos() as f32;
            assert!((celt_cos_norm2(x) - want).abs() < 1e-5, "{x}");
        }
    }

    #[cfg(feature = "osce-bwe")]
    #[test]
    fn celt_sin_is_the_libopus_formula() {
        // cos(π/2 · (π/2 · y - 1)), not sin(y).
        for i in 1..200 {
            let y = i as f32 / 20.0 - 5.0;
            let want = (core::f64::consts::FRAC_PI_2 * (core::f64::consts::FRAC_PI_2 * y as f64 - 1.0)).cos();
            assert!((celt_sin(y) as f64 - want).abs() < 1e-5, "{y}");
        }
    }
}
