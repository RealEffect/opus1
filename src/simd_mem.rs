//! Safe loads and stores of vector registers.
//!
//! The intrinsics of `core::arch` that read or write memory (`vld1q_f32`,
//! `_mm_loadu_ps`, `_mm256_storeu_ps`, ...) take raw pointers, so each call
//! needs an `unsafe` block. The functions here make these calls for the
//! kernels. Each takes a reference to an array of exactly as many values as
//! the instruction reads or writes. The reference already guarantees what
//! the intrinsic needs: the memory is valid for that many values, and for a
//! store no other reference reaches it. The instructions are the unaligned
//! forms. The x86 ones need no alignment at all; the NEON ones need only the
//! alignment of one value, which the reference has. So each function is
//! safe, and its one `unsafe` block is the only one a kernel needs for
//! memory.
//!
//! Each function is compiled with the target feature of its intrinsic. A
//! kernel compiled with that feature, or with one that includes it (AVX2
//! includes AVX, SSE2 and SSE), calls it without `unsafe`, just as it calls
//! the intrinsics that only compute on registers (`vaddq_f32`,
//! `_mm256_fmadd_ps`). The `unsafe` block of a kernel is then the call to
//! the kernel itself, from code without the feature, where the caller
//! states why the processor has the feature.
//!
//! How a kernel gets the arrays from a slice without an index check in its
//! loop:
//! - `as_chunks`, `as_chunks_mut`, `as_rchunks` and `chunks_exact` split a
//!   slice into arrays. A loop over them has no index check.
//! - [`array_at`] and [`array_at_mut`] take the array at an index. The
//!   compiler drops their check when it can tell that the range is inside
//!   the slice, for example in a loop over at most 32 values that it
//!   unrolls. In other loops, such as `while i + 4 <= steps`, it may not:
//!   when it cannot rule out that `i + 4` wraps around, a compare stays in
//!   the loop. So check the generated code of a hot loop.
//! - Two ranges of one slice that a loop reads and writes in turn, such as
//!   `x[i..i + 4]` and `x[i + stride..i + stride + 4]`, are views of the
//!   same `&[Cell<f32>]` (`Cell::from_mut(x).as_slice_of_cells()`), split
//!   with `as_chunks`. The `_cells` functions load and store them.
//!
//! The functions are `#[inline]`, not `#[inline(always)]`, which Rust does
//! not allow together with `#[target_feature]`. Each is one instruction, so
//! the compiler always inlines it.

/// The `N` values `s[i..i + N]` as an array.
///
/// Panics when `s` holds fewer than `i + N` values.
// The library uses this and `array_at_mut` only in x86 kernels so far.
#[cfg_attr(not(any(target_arch = "x86", target_arch = "x86_64")), allow(dead_code))]
#[inline(always)]
pub(crate) fn array_at<T, const N: usize>(s: &[T], i: usize) -> &[T; N] {
    s[i..i + N].try_into().unwrap()
}

/// The `N` values `s[i..i + N]` as a mutable array.
///
/// Panics when `s` holds fewer than `i + N` values.
#[cfg_attr(not(any(target_arch = "x86", target_arch = "x86_64")), allow(dead_code))]
#[inline(always)]
pub(crate) fn array_at_mut<T, const N: usize>(s: &mut [T], i: usize) -> &mut [T; N] {
    (&mut s[i..i + N]).try_into().unwrap()
}

/// Loads and stores for NEON (aarch64).
#[cfg(target_arch = "aarch64")]
pub(crate) mod neon {
    use core::arch::aarch64::*;
    use core::cell::Cell;

    // `f32` values.

    /// Loads four `f32` values.
    #[target_feature(enable = "neon")]
    #[inline]
    pub(crate) fn load_f32x4(a: &[f32; 4]) -> float32x4_t {
        // SAFETY: `a` is valid for reading four `f32` values, which is
        // what `vld1q_f32` reads. It needs only the alignment of `f32`.
        unsafe { vld1q_f32(a.as_ptr()) }
    }

    /// Stores four `f32` values.
    #[target_feature(enable = "neon")]
    #[inline]
    pub(crate) fn store_f32x4(a: &mut [f32; 4], v: float32x4_t) {
        // SAFETY: `a` is valid for writing four `f32` values, which is
        // what `vst1q_f32` writes, and no other reference reaches them. It
        // needs only the alignment of `f32`.
        unsafe { vst1q_f32(a.as_mut_ptr(), v) }
    }

    /// Loads four `f32` values from cells.
    #[target_feature(enable = "neon")]
    #[inline]
    pub(crate) fn load_f32x4_cells(a: &[Cell<f32>; 4]) -> float32x4_t {
        // SAFETY: `Cell<f32>` has the layout of `f32`, so `a` is valid for
        // reading four `f32` values. Nothing writes them during the load:
        // a `Cell` lends no reference to its value and stays in one
        // thread.
        unsafe { vld1q_f32(a.as_ptr().cast()) }
    }

    /// Stores four `f32` values into cells.
    #[target_feature(enable = "neon")]
    #[inline]
    pub(crate) fn store_f32x4_cells(a: &[Cell<f32>; 4], v: float32x4_t) {
        // SAFETY: `Cell<f32>` has the layout of `f32` and may be written
        // through a shared reference, so `a` is valid for writing four `f32`
        // values. Nothing else reads or writes them during the store: a
        // `Cell` lends no reference to its value and stays in one thread.
        unsafe { vst1q_f32(a.as_ptr().cast_mut().cast(), v) }
    }

    // `i32` and `u32` values.

    /// Loads four `i32` values.
    #[target_feature(enable = "neon")]
    #[inline]
    pub(crate) fn load_i32x4(a: &[i32; 4]) -> int32x4_t {
        // SAFETY: `a` is valid for reading four `i32` values, which is
        // what `vld1q_s32` reads. It needs only the alignment of `i32`.
        unsafe { vld1q_s32(a.as_ptr()) }
    }

    /// Loads four `u32` values.
    #[target_feature(enable = "neon")]
    #[inline]
    pub(crate) fn load_u32x4(a: &[u32; 4]) -> uint32x4_t {
        // SAFETY: `a` is valid for reading four `u32` values, which is
        // what `vld1q_u32` reads. It needs only the alignment of `u32`.
        unsafe { vld1q_u32(a.as_ptr()) }
    }
}

/// Loads and stores for SSE, SSE2, AVX and AVX-512 (x86 and x86-64).
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub(crate) mod x86 {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;
    use core::cell::Cell;

    // `f32` values.

    /// Loads four `f32` values (SSE).
    #[target_feature(enable = "sse")]
    #[inline]
    pub(crate) fn load_f32x4(a: &[f32; 4]) -> __m128 {
        // SAFETY: `a` is valid for reading four `f32` values, which is
        // what `_mm_loadu_ps` reads. It needs no alignment.
        unsafe { _mm_loadu_ps(a.as_ptr()) }
    }

    /// Loads four `f32` values from cells (SSE).
    #[target_feature(enable = "sse")]
    #[inline]
    pub(crate) fn load_f32x4_cells(a: &[Cell<f32>; 4]) -> __m128 {
        // SAFETY: `Cell<f32>` has the layout of `f32`, so `a` is valid for
        // reading four `f32` values. Nothing writes them during the load:
        // a `Cell` lends no reference to its value and stays in one
        // thread.
        unsafe { _mm_loadu_ps(a.as_ptr().cast()) }
    }

    /// Stores four `f32` values into cells (SSE).
    #[target_feature(enable = "sse")]
    #[inline]
    pub(crate) fn store_f32x4_cells(a: &[Cell<f32>; 4], v: __m128) {
        // SAFETY: `Cell<f32>` has the layout of `f32` and may be written
        // through a shared reference, so `a` is valid for writing four `f32`
        // values. Nothing else reads or writes them during the store: a
        // `Cell` lends no reference to its value and stays in one thread.
        unsafe { _mm_storeu_ps(a.as_ptr().cast_mut().cast(), v) }
    }

    /// Loads eight `f32` values (AVX).
    #[target_feature(enable = "avx")]
    #[inline]
    pub(crate) fn load_f32x8(a: &[f32; 8]) -> __m256 {
        // SAFETY: `a` is valid for reading eight `f32` values, which is
        // what `_mm256_loadu_ps` reads. It needs no alignment.
        unsafe { _mm256_loadu_ps(a.as_ptr()) }
    }

    /// Stores eight `f32` values (AVX).
    #[target_feature(enable = "avx")]
    #[inline]
    pub(crate) fn store_f32x8(a: &mut [f32; 8], v: __m256) {
        // SAFETY: `a` is valid for writing eight `f32` values, which is
        // what `_mm256_storeu_ps` writes, and no other reference reaches
        // them. It needs no alignment.
        unsafe { _mm256_storeu_ps(a.as_mut_ptr(), v) }
    }

    // `i32` values.

    /// Loads four `i32` values (SSE2).
    #[target_feature(enable = "sse2")]
    #[inline]
    pub(crate) fn load_i32x4(a: &[i32; 4]) -> __m128i {
        // SAFETY: `a` is valid for reading four `i32` values (16 bytes),
        // which is what `_mm_loadu_si128` reads. It needs no alignment.
        unsafe { _mm_loadu_si128(a.as_ptr().cast()) }
    }

    /// Stores four `i32` values (SSE2).
    #[target_feature(enable = "sse2")]
    #[inline]
    pub(crate) fn store_i32x4(a: &mut [i32; 4], v: __m128i) {
        // SAFETY: `a` is valid for writing four `i32` values (16 bytes),
        // which is what `_mm_storeu_si128` writes, and no other reference
        // reaches them. It needs no alignment.
        unsafe { _mm_storeu_si128(a.as_mut_ptr().cast(), v) }
    }

    /// Loads eight `i32` values (AVX).
    #[target_feature(enable = "avx")]
    #[inline]
    pub(crate) fn load_i32x8(a: &[i32; 8]) -> __m256i {
        // SAFETY: `a` is valid for reading eight `i32` values (32 bytes),
        // which is what `_mm256_loadu_si256` reads. It needs no alignment.
        unsafe { _mm256_loadu_si256(a.as_ptr().cast()) }
    }

    /// Stores eight `i32` values (AVX).
    #[target_feature(enable = "avx")]
    #[inline]
    pub(crate) fn store_i32x8(a: &mut [i32; 8], v: __m256i) {
        // SAFETY: `a` is valid for writing eight `i32` values (32 bytes),
        // which is what `_mm256_storeu_si256` writes, and no other
        // reference reaches them. It needs no alignment.
        unsafe { _mm256_storeu_si256(a.as_mut_ptr().cast(), v) }
    }
}
