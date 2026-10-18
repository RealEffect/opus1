//! The processor variants of the neural network kernels, and their
//! detection (libopus `opus_select_arch()` and the tables of
//! `dnn/arm/arm_dnn_map.c`).
//!
//! libopus selects its kernels at run time from the processor. opus1 does
//! the same, but every variant computes the same bits: the bits of libopus
//! on 64-bit ARM. A decoder detects its variant once, when it is created,
//! and tests can pass any variant of `available()`.
//!
//! Every value of `DnnArch` exists on every target. The layer functions
//! accept any of them: a variant that the processor cannot run is replaced
//! by the best one it can run (`usable`), which gives the same bits. So a
//! wrong value can make a call slower, never wrong or unsafe.

/// A set of kernels. Every value exists on every target, so tests and
/// stored settings need no `cfg`; `available()` says which ones the
/// processor can run.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DnnArch {
    /// Plain Rust (`vec_portable`). Runs everywhere.
    Portable,
    /// 64-bit ARM with NEON, int8 products without the dot product
    /// instructions (libopus arch 3).
    Neon,
    /// 64-bit ARM with NEON and the DotProd extension (`sdot`; libopus
    /// arch 4).
    NeonDotprod,
    /// x86-64 with AVX2 and FMA.
    Avx2Fma,
    /// x86-64 with AVX2, FMA and AVX-VNNI (vector neural network
    /// instructions on 256-bit registers).
    AvxVnni,
    /// x86-64 with AVX2, FMA and AVX-512 VNNI (and AVX-512 F and BW).
    Avx512Vnni,
}

/// Whether the processor has the DotProd extension (FEAT_DotProd).
///
/// Known at compile time where it is part of the target (Apple silicon).
/// Elsewhere it is a run-time check, which needs `std`; without `std` the
/// answer is no.
#[cfg(target_arch = "aarch64")]
#[inline]
pub(crate) fn has_dotprod() -> bool {
    #[cfg(target_feature = "dotprod")]
    {
        true
    }
    #[cfg(all(not(target_feature = "dotprod"), feature = "std"))]
    {
        std::arch::is_aarch64_feature_detected!("dotprod")
    }
    #[cfg(all(not(target_feature = "dotprod"), not(feature = "std")))]
    {
        false
    }
}

/// The x86 feature levels that the kernels use: (AVX2 and FMA, AVX-VNNI,
/// AVX-512 VNNI with F and BW). With `std` they are detected at run time;
/// without `std` they come from the compile-time target features.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
fn x86_levels() -> (bool, bool, bool) {
    #[cfg(feature = "std")]
    {
        let avx2 = std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma");
        let vnni = avx2 && std::arch::is_x86_feature_detected!("avxvnni");
        let vnni512 = avx2
            && std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512bw")
            && std::arch::is_x86_feature_detected!("avx512vnni");
        (avx2, vnni, vnni512)
    }
    #[cfg(not(feature = "std"))]
    {
        let avx2 = cfg!(all(target_feature = "avx2", target_feature = "fma"));
        let vnni = avx2 && cfg!(target_feature = "avxvnni");
        let vnni512 = avx2
            && cfg!(all(
                target_feature = "avx512f",
                target_feature = "avx512bw",
                target_feature = "avx512vnni"
            ));
        (avx2, vnni, vnni512)
    }
}

/// The fastest variant this processor can run.
///
/// - aarch64: `NeonDotprod` when DotProd is enabled at compile time or,
///   with `std`, detected at run time; otherwise `Neon`.
/// - x86 and x86-64: with `std`, run-time detection of AVX-512 VNNI, AVX-VNNI, then
///   AVX2 and FMA; without `std`, the same from the compile-time target
///   features. Otherwise `Portable`.
/// - other targets: `Portable`.
pub fn detect() -> DnnArch {
    #[cfg(target_arch = "aarch64")]
    {
        if has_dotprod() { DnnArch::NeonDotprod } else { DnnArch::Neon }
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        match x86_levels() {
            (true, _, true) => DnnArch::Avx512Vnni,
            (true, true, false) => DnnArch::AvxVnni,
            (true, false, false) => DnnArch::Avx2Fma,
            _ => DnnArch::Portable,
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86", target_arch = "x86_64")))]
    {
        DnnArch::Portable
    }
}

/// Every variant this processor can run, `Portable` first. Tests run each
/// of them and compare the bits.
pub fn available() -> &'static [DnnArch] {
    use DnnArch::*;
    #[cfg(target_arch = "aarch64")]
    {
        if has_dotprod() { &[Portable, Neon, NeonDotprod] } else { &[Portable, Neon] }
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        match x86_levels() {
            (true, true, true) => &[Portable, Avx2Fma, AvxVnni, Avx512Vnni],
            (true, false, true) => &[Portable, Avx2Fma, Avx512Vnni],
            (true, true, false) => &[Portable, Avx2Fma, AvxVnni],
            (true, false, false) => &[Portable, Avx2Fma],
            _ => &[Portable],
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86", target_arch = "x86_64")))]
    {
        &[Portable]
    }
}

/// `arch` if this processor can run it, else the best variant it can run
/// in its place. Every variant gives the same bits, so the replacement
/// only changes the speed.
///
/// The layer functions call this before they call a kernel, so the
/// `unsafe` kernels only ever run on a processor that has their
/// instructions. The checks are cached by `std` (one atomic load each) or
/// known at compile time.
#[inline]
pub(crate) fn usable(arch: DnnArch) -> DnnArch {
    match arch {
        DnnArch::Portable => DnnArch::Portable,
        #[cfg(target_arch = "aarch64")]
        DnnArch::Neon => DnnArch::Neon,
        #[cfg(target_arch = "aarch64")]
        DnnArch::NeonDotprod => {
            if has_dotprod() { DnnArch::NeonDotprod } else { DnnArch::Neon }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        DnnArch::Avx2Fma | DnnArch::AvxVnni | DnnArch::Avx512Vnni => {
            let (avx2, vnni, vnni512) = x86_levels();
            match arch {
                DnnArch::Avx512Vnni if vnni512 => DnnArch::Avx512Vnni,
                DnnArch::AvxVnni if vnni => DnnArch::AvxVnni,
                _ if avx2 => DnnArch::Avx2Fma,
                _ => DnnArch::Portable,
            }
        }
        // A variant of another processor family.
        #[allow(unreachable_patterns)]
        _ => detect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_is_available_and_usable() {
        let best = detect();
        assert!(available().contains(&best));
        assert_eq!(available()[0], DnnArch::Portable);
        for &a in available() {
            assert_eq!(usable(a), a);
        }
    }

    #[test]
    fn every_variant_maps_to_an_available_one() {
        use DnnArch::*;
        for a in [Portable, Neon, NeonDotprod, Avx2Fma, AvxVnni, Avx512Vnni] {
            assert!(available().contains(&usable(a)), "{a:?}");
        }
    }

    #[cfg(all(target_arch = "aarch64", target_vendor = "apple"))]
    #[test]
    fn apple_targets_have_dotprod() {
        assert_eq!(detect(), DnnArch::NeonDotprod);
    }
}
