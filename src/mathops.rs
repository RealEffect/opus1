//! Numeric helpers of the float build that several modules use, named after
//! libopus celt/mathops.h.
//!
//! `mac` and `mac_f64` round as libopus does on the processor this crate is
//! built for. `pitch::mul_acc` and `kiss_fft::Arith` are separate on
//! purpose: their callers choose the rounding at compile time.

#[cfg(not(feature = "std"))]
use crate::compat::Math;

/// `acc + a * b`, rounded as libopus rounds the C expression on this
/// processor.
///
/// libopus writes these steps as plain C: `MAC16_16(acc, a, b)` of the float
/// build, or the steps of a polynomial. clang builds libopus on 64-bit ARM
/// with `-ffp-contract=on`, its default. It turns such an expression into
/// one fused multiply-add, which rounds once. The arm64 objects of libopus
/// 1.6.1 contain these `fmadd` instructions. x86-64 builds of libopus do not
/// fuse without `-mfma`, so other targets round the product first.
///
/// The rounding shows in the output. Measured on 64-bit ARM against libopus
/// 1.6.1: with separate rounding, 793 of 12600 band log energies of a stereo
/// stream were one step off, and the dynamic allocation differed in 17 of
/// 300 frames.
#[inline(always)]
pub(crate) fn mac(acc: f32, a: f32, b: f32) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        use core::arch::aarch64::{vdup_n_f32, vfma_f32, vget_lane_f32};
        // SAFETY: these vector instructions are part of every 64-bit ARM
        // processor. The compiler turns the three lanes into one scalar
        // `fmadd`.
        unsafe { vget_lane_f32::<0>(vfma_f32(vdup_n_f32(acc), vdup_n_f32(a), vdup_n_f32(b))) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        acc + a * b
    }
}

/// `acc + a * b` in double precision, rounded as libopus rounds it on this
/// processor: one fused multiply-add on 64-bit ARM (see `mac`).
#[inline(always)]
pub(crate) fn mac_f64(acc: f64, a: f64, b: f64) -> f64 {
    #[cfg(target_arch = "aarch64")]
    {
        use core::arch::aarch64::{vdup_n_f64, vfma_f64, vget_lane_f64};
        // SAFETY: these vector instructions are part of every 64-bit ARM
        // processor.
        unsafe { vget_lane_f64::<0>(vfma_f64(vdup_n_f64(acc), vdup_n_f64(a), vdup_n_f64(b))) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        acc + a * b
    }
}

/// libopus `celt_log2` of the float build with `FLOAT_APPROX`
/// (celt/mathops.h): the base-2 logarithm from the exponent bits and a
/// polynomial of degree 4 for the mantissa.
///
/// `FLOAT_APPROX` is what the autotools build of libopus turns on by default
/// for 64-bit ARM and x86, and what the libopus these tests compare with was
/// built with. The result is within about one float step of the exact
/// logarithm. Like libopus, this does not handle 0, negative values,
/// infinities or NaN.
///
/// Each step `c + a * b` is one `mac`.
// The constants are written with the digits libopus has.
#[allow(clippy::excessive_precision)]
#[inline]
pub(crate) fn celt_log2(x: f32) -> f32 {
    const X_NORM: [f32; 8] = [
        1.0,
        8.888_888_955_116_272e-1,
        8.0e-1,
        7.272_727_489_471_436e-1,
        6.666_666_865_348_816e-1,
        6.153_846_383_094_788e-1,
        5.714_285_969_734_192e-1,
        5.333_333_611_488_342e-1,
    ];
    const Y_NORM: [f32; 8] = [
        0.0,
        1.699_250_042_438_507e-1,
        3.219_280_838_966_369_6e-1,
        4.594_316_184_520_721_4e-1,
        5.849_624_872_207_642e-1,
        7.004_396_915_435_791e-1,
        8.073_549_270_629_883e-1,
        9.068_905_711_174_011e-1,
    ];
    let bits = x.to_bits() as i32;
    let integer = (bits >> 23) - 127;
    let bits = bits.wrapping_sub(((integer as u32) << 23) as i32);
    // "Normalize the mantissa range from [1, 2] to [1,1.125], and then
    // shift x by 1.0625 to [-0.0625, 0.0625]."
    let range_idx = ((bits >> 20) & 0x7) as usize;
    let m = mac(-1.0625, f32::from_bits(bits as u32), X_NORM[range_idx]);
    // The polynomial, innermost bracket first.
    let p = mac(4.019_712_507_724_762e-1, m, -2.841_544_449_329_376e-1);
    let p = mac(-6.389_770_507_812_5e-1, m, p);
    let p = mac(1.357_829_570_770_263_7, m, p);
    let p = mac(8.746_284_246_444_702e-2, m, p);
    integer as f32 + p + Y_NORM[range_idx]
}

/// libopus `celt_exp2` of the float build with `FLOAT_APPROX`
/// (celt/mathops.h): `2^x` from a polynomial of degree 5 for the fraction,
/// put into the exponent bits. Each step `c + frac * p` is one `mac`.
#[allow(clippy::excessive_precision)]
#[inline]
pub(crate) fn celt_exp2(x: f32) -> f32 {
    let integer = x.floor() as i32;
    if integer < -50 {
        return 0.0;
    }
    let frac = x - integer as f32;
    // The polynomial, innermost bracket first.
    let p = mac(8.989_339_694_380_76e-3, frac, 1.877_576_694_823_801_5e-3);
    let p = mac(5.582_631_751_894_951e-2, frac, p);
    let p = mac(2.401_536_107_063_293_5e-1, frac, p);
    let p = mac(6.931_530_833_244_324e-1, frac, p);
    let p = mac(9.999_999_403_953_552e-1, frac, p);
    let bits =
        (p.to_bits() as i32).wrapping_add(((integer as u32) << 23) as i32) as u32 & 0x7fff_ffff;
    f32::from_bits(bits)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Input, then the result of libopus 1.6 `celt_log2` (`FLOAT_APPROX`)
    /// built with clang on 64-bit ARM (fused multiply-adds), then built
    /// without fused multiply-adds, as on x86-64. All as float bits. These
    /// are inputs for which the two builds differ.
    const LIBOPUS_LOG2: [(u32, u32, u32); 12] = [
        (0x3eec39c4, 0xbf8ed85f, 0xbf8ed860),
        (0x2727f314, 0xc2426eb5, 0xc2426eb6),
        (0x3e959784, 0xbfe336e0, 0xbfe336e1),
        (0x4050a227, 0x3fda37c0, 0x3fda37c1),
        (0x3f62a7e9, 0xbe33dafc, 0xbe33daf8),
        (0x51295a74, 0x420d9d97, 0x420d9d96),
        (0x4161ad9b, 0x40745c20, 0x40745c21),
        (0x49b8b15d, 0x41a43b5c, 0x41a43b5b),
        (0x3fa73822, 0x3ec56d03, 0x3ec56d04),
        (0x40616760, 0x3fe87ec0, 0x3fe87ec1),
        (0x3e9179b3, 0xbfe85e01, 0xbfe85e00),
        (0x3f943826, 0x3e58ab00, 0x3e58ab02),
    ];

    /// The same for libopus `celt_exp2`.
    const LIBOPUS_EXP2: [(u32, u32, u32); 12] = [
        (0xc0b1d0a0, 0x3cae0a7d, 0x3cae0a7e),
        (0xc18e3c3c, 0x36952591, 0x36952590),
        (0xc1a2f360, 0x35463f8f, 0x35463f90),
        (0x411b021c, 0x444e3763, 0x444e3762),
        (0xc0806620, 0x3d7dcc24, 0x3d7dcc23),
        (0xbdea4600, 0x3f6c7c53, 0x3f6c7c52),
        (0x40933f40, 0x41c235ab, 0x41c235aa),
        (0x40eea958, 0x432fd8ad, 0x432fd8ae),
        (0x415dd50c, 0x46690d89, 0x46690d88),
        (0xc2143b7f, 0x2cf5e51d, 0x2cf5e51e),
        (0xc1c967a1, 0x32e2a985, 0x32e2a986),
        (0xc0f9bc38, 0x3b929a73, 0x3b929a74),
    ];

    /// `celt_log2` and `celt_exp2` give the bits of libopus on this
    /// processor: with fused multiply-adds on 64-bit ARM, without
    /// elsewhere (`mac`).
    #[test]
    fn log2_and_exp2_round_as_libopus() {
        let fused = cfg!(target_arch = "aarch64");
        for (f, table) in [
            (celt_log2 as fn(f32) -> f32, &LIBOPUS_LOG2),
            (celt_exp2, &LIBOPUS_EXP2),
        ] {
            for &(x, with_fma, without) in table.iter() {
                let want = if fused { with_fma } else { without };
                let got = f(f32::from_bits(x)).to_bits();
                assert_eq!(got, want, "input {x:08x}: {got:08x}, libopus {want:08x}");
            }
        }
    }
}
