//! The kernels for x86 and x86-64: AVX2 with FMA, AVX-VNNI and AVX-512 VNNI.
//!
//! These kernels do not copy the x86 code of libopus (`dnn/vec_avx.h`),
//! which computes other bits (unsigned inputs, `subias`, `rcp_ps`). They
//! give the bits of the functions of the same name in `vec_portable`, that
//! is, the bits of libopus on 64-bit ARM (`nnet.md` section 10.6):
//!
//! - int8 products with signed inputs: `maddubs(abs(x), sign(w, x))`, or
//!   `dpbusd` with VNNI. Nothing saturates while every weight lies in
//!   [-127, 127]: a pair of products is at most 2 · 128 · 127 = 32,512.
//! - input quantization with `cvtps_epi32` (ties to even in the default
//!   rounding mode). x86 gives `i32::MIN` for NaN and for values out of
//!   range; ARM gives 0 for NaN and `i32::MAX` for large positive values.
//!   Only the low byte is kept, and the low byte of `i32::MIN` is 0, so only
//!   the large positive values need a fix (to -1).
//! - activations with `fmadd`, a vector FRECPE, and `min`/`max` with the
//!   constant as the first operand, which returns the NaN of the input as
//!   ARM's FMIN and FMAX do.
//!
//! The kernels assume the default floating-point mode (MXCSR: round to
//! nearest, no flush to zero), the state of user programs.
//!
//! They can only be compile-checked on the development machine (an Apple
//! M4); a run on an x86 machine is an open item.

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;
use core::ptr;

use super::MAX_INPUTS;
use super::vec_portable;

// ---------------------------------------------------------------------------
// Float products
// ---------------------------------------------------------------------------

/// One pass of `sgemv` over `V` vectors of 8 rows.
///
/// # Safety
///
/// AVX2 and FMA. `out` has `8·V` values; `w` has `8·V` values at each
/// multiple of `stride` below `cols·stride`; `x` has `cols` values.
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn sgemv_pass<const V: usize>(out: *mut f32, w: *const f32, cols: usize, stride: usize, x: *const f32) {
    // SAFETY: the caller guarantees the lengths.
    unsafe {
        let mut acc = [_mm256_setzero_ps(); V];
        let mut wj = w;
        for j in 0..cols {
            let xj = _mm256_broadcast_ss(&*x.add(j));
            for (v, a) in acc.iter_mut().enumerate() {
                *a = _mm256_fmadd_ps(_mm256_loadu_ps(wj.add(8 * v)), xj, *a);
            }
            // After the last column this points past the weights; it is
            // not read then. `wrapping_add` makes that pointer allowed.
            wj = wj.wrapping_add(stride);
        }
        for (v, a) in acc.iter().enumerate() {
            _mm256_storeu_ps(out.add(8 * v), *a);
        }
    }
}

/// AVX2 `sgemv`. Same contract as `vec_portable::sgemv`.
///
/// # Safety
///
/// The processor must support AVX2 and FMA. The slices must have the
/// lengths that `vec_portable::sgemv` requires, and `col_stride >= rows`.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn sgemv_avx2(
    out: &mut [f32],
    weights: &[f32],
    rows: usize,
    cols: usize,
    col_stride: usize,
    x: &[f32],
) {
    // The reads of row `r` and column `j` are at `w[j * col_stride + r]`,
    // so they end at `(cols - 1) * col_stride + rows`. Checked without
    // wrapping around; `nnet` never calls with `cols == 0`.
    assert!(
        out.len() >= rows
            && x.len() >= cols
            && (rows == 0
                || cols > 0
                    && (cols - 1)
                        .checked_mul(col_stride)
                        .and_then(|n| n.checked_add(rows))
                        .is_some_and(|n| weights.len() >= n)),
        "sgemv_avx2: slices shorter than the layer"
    );
    let o = out.as_mut_ptr();
    let w = weights.as_ptr();
    let xp = x.as_ptr();
    let mut i = 0;
    // SAFETY: the caller guarantees the lengths; every pass stays below
    // `rows`.
    unsafe {
        while rows - i >= 32 {
            sgemv_pass::<4>(o.add(i), w.add(i), cols, col_stride, xp);
            i += 32;
        }
        match (rows - i) / 8 {
            3 => sgemv_pass::<3>(o.add(i), w.add(i), cols, col_stride, xp),
            2 => sgemv_pass::<2>(o.add(i), w.add(i), cols, col_stride, xp),
            1 => sgemv_pass::<1>(o.add(i), w.add(i), cols, col_stride, xp),
            _ => {}
        }
        i += (rows - i) & !7;
        if rows - i >= 4 {
            let mut acc = _mm_setzero_ps();
            let mut wj = w.add(i);
            for j in 0..cols {
                acc = _mm_fmadd_ps(_mm_loadu_ps(wj), _mm_set1_ps(*xp.add(j)), acc);
                wj = wj.wrapping_add(col_stride);
            }
            _mm_storeu_ps(o.add(i), acc);
            i += 4;
        }
        let r = rows - i;
        if r > 0 {
            let mut acc = [_mm_setzero_ps(); 3];
            let mut wj = w.add(i);
            for j in 0..cols {
                let xj = _mm_set_ss(*xp.add(j));
                for (k, a) in acc[..r].iter_mut().enumerate() {
                    *a = _mm_fmadd_ss(_mm_set_ss(*wj.add(k)), xj, *a);
                }
                wj = wj.wrapping_add(col_stride);
            }
            for (k, a) in acc[..r].iter().enumerate() {
                *o.add(i + k) = _mm_cvtss_f32(*a);
            }
        }
    }
}

/// AVX2 `sparse_sgemv8x4`. Same contract as
/// `vec_portable::sparse_sgemv8x4`.
///
/// # Safety
///
/// The processor must support AVX2 and FMA. The index table must be valid
/// for `rows` and the length of `x`, and `weights` must hold 32 floats for
/// each listed block, as `nnet` checks.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn sparse_sgemv8x4_avx2(
    out: &mut [f32],
    weights: &[f32],
    idx: &[i32],
    rows: usize,
    x: &[f32],
) {
    // Each block of 8 rows stores 8 values. `nnet` checks the index table
    // (`sparse_blocks`); the debug checks below repeat that per entry.
    assert!(rows.is_multiple_of(8) && out.len() >= rows);
    let w = weights.as_ptr();
    let xp = x.as_ptr();
    let o = out.as_mut_ptr();
    // The next entry of `idx` and the next block of `weights`.
    let mut p = 0usize;
    let mut wo = 0usize;
    // SAFETY: the caller guarantees the table and the lengths.
    unsafe {
        for i in (0..rows).step_by(8) {
            let mut y = _mm256_setzero_ps();
            let count = idx[p] as usize;
            p += 1;
            for _ in 0..count {
                debug_assert!(p < idx.len() && wo + 32 <= weights.len());
                let pos = *idx.get_unchecked(p) as usize;
                debug_assert!(pos + 4 <= x.len());
                p += 1;
                for c in 0..4 {
                    let xc = _mm256_set1_ps(*xp.add(pos + c));
                    y = _mm256_fmadd_ps(_mm256_loadu_ps(w.add(wo + 8 * c)), xc, y);
                }
                wo += 32;
            }
            _mm256_storeu_ps(o.add(i), y);
        }
    }
}

// ---------------------------------------------------------------------------
// int8 products
// ---------------------------------------------------------------------------

/// The quantized input of an int8 product.
#[repr(C, align(32))]
struct XBuf([i8; MAX_INPUTS]);

/// `xq[j] = quantize(x[j])` for `j < cols`, with the ARM results.
///
/// # Safety
///
/// AVX2. `x` has `cols` values and `xq` room for them.
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn quantize(xq: *mut i8, x: *const f32, cols: usize) {
    // SAFETY: the caller guarantees the lengths.
    unsafe {
        let c127 = _mm256_set1_ps(127.0);
        let two31 = _mm256_set1_ps(2_147_483_648.0);
        // Byte 0 of each 32-bit lane, in each 128-bit half.
        let pick = _mm256_setr_epi8(
            0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 4, 8, 12, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1,
        );
        let join = _mm256_setr_epi32(0, 4, 1, 1, 1, 1, 1, 1);
        let mut j = 0;
        while j + 8 <= cols {
            let p = _mm256_mul_ps(c127, _mm256_loadu_ps(x.add(j)));
            let i = _mm256_cvtps_epi32(p);
            // ARM saturates to i32::MAX, whose low byte is 0xff.
            let big = _mm256_castps_si256(_mm256_cmp_ps::<_CMP_GE_OQ>(p, two31));
            let i = _mm256_or_si256(i, big);
            let b = _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8(i, pick), join);
            _mm_storel_epi64(xq.add(j) as *mut __m128i, _mm256_castsi256_si128(b));
            j += 8;
        }
        while j < cols {
            *xq.add(j) = vec_portable::quantize(*x.add(j));
            j += 1;
        }
    }
}

/// The 4 quantized inputs at `q`, repeated in each 32-bit lane.
///
/// # Safety
///
/// AVX; `q` has 4 bytes.
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn load_x4(q: *const i8) -> __m256i {
    // SAFETY: the caller guarantees 4 readable bytes.
    unsafe { _mm256_set1_epi32(ptr::read_unaligned(q as *const i32)) }
}

/// The dot products of the 8 rows of one 8 × 4 block with 4 inputs,
/// exactly: `maddubs(|x|, sign(w, x))` gives 16-bit pairs, `madd` with ones
/// adds the pairs of each row.
///
/// # Safety
///
/// AVX2.
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn dot_avx2(acc: __m256i, w: __m256i, x: __m256i) -> __m256i {
    let t = _mm256_maddubs_epi16(_mm256_abs_epi8(x), _mm256_sign_epi8(w, x));
    _mm256_add_epi32(acc, _mm256_madd_epi16(t, _mm256_set1_epi16(1)))
}

/// `dot_avx2` with AVX-VNNI: `dpbusd` adds the 4 products of each row
/// without an intermediate 16-bit sum.
///
/// # Safety
///
/// AVX2 and AVX-VNNI.
#[target_feature(enable = "avx2,fma,avxvnni")]
#[inline]
unsafe fn dot_avxvnni(acc: __m256i, w: __m256i, x: __m256i) -> __m256i {
    _mm256_dpbusd_avx_epi32(acc, _mm256_abs_epi8(x), _mm256_sign_epi8(w, x))
}

/// `out[k] = scale[k] · acc[k]` for the 8 rows of one block.
///
/// # Safety
///
/// AVX; `out` and `scale` have 8 values.
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn store8(out: *mut f32, scale: *const f32, acc: __m256i) {
    // SAFETY: the caller guarantees the lengths.
    unsafe { _mm256_storeu_ps(out, _mm256_mul_ps(_mm256_loadu_ps(scale), _mm256_cvtepi32_ps(acc))) }
}

/// Defines a dense and a sparse int8 product on one 256-bit dot product.
macro_rules! int8_kernels {
    ($features:literal, $dot:ident, $dense:ident, $sparse:ident, $what:literal) => {
        #[doc = concat!($what, " `cgemv8x4`. Same contract as `vec_portable::cgemv8x4`.")]
        ///
        /// # Safety
        ///
        /// The processor must support the instructions named above. The
        /// slices must have the lengths that `vec_portable::cgemv8x4`
        /// requires.
        #[target_feature(enable = $features)]
        pub unsafe fn $dense(
            out: &mut [f32],
            weights: &[i8],
            scale: &[f32],
            rows: usize,
            cols: usize,
            x: &[f32],
        ) {
            // `quantize` writes `cols` bytes into a buffer of `MAX_INPUTS`;
            // block `b` stores rows `8 * b..8 * b + 8`; the weights are read
            // up to `rows * cols`.
            assert!(
                cols <= MAX_INPUTS
                    && x.len() >= cols
                    && out.len() >= rows
                    && scale.len() >= rows
                    && rows.checked_mul(cols).is_some_and(|n| weights.len() >= n),
                "cgemv8x4: slices shorter than the layer"
            );
            let mut xq = core::mem::MaybeUninit::<XBuf>::uninit();
            let q = xq.as_mut_ptr() as *mut i8;
            let o = out.as_mut_ptr();
            let s = scale.as_ptr();
            let groups = cols / 4;
            // SAFETY: the caller guarantees the lengths; only the `cols`
            // bytes written by `quantize` are read.
            unsafe {
                quantize(q, x.as_ptr(), cols);
                // One block of 8 rows at a time, so the weights are one
                // sequential stream (several streams were slower on 64-bit
                // ARM when the weights are not in the caches); 4 column
                // groups on separate accumulators.
                let mut w = weights.as_ptr();
                for b in 0..rows / 8 {
                    let mut acc = [_mm256_setzero_si256(); 4];
                    let mut g = 0;
                    while g + 4 <= groups {
                        for (u, a) in acc.iter_mut().enumerate() {
                            let vx = load_x4(q.add(4 * (g + u)));
                            *a = $dot(*a, _mm256_loadu_si256(w.add(32 * u) as *const __m256i), vx);
                        }
                        w = w.add(128);
                        g += 4;
                    }
                    while g < groups {
                        let vx = load_x4(q.add(4 * g));
                        acc[0] = $dot(acc[0], _mm256_loadu_si256(w as *const __m256i), vx);
                        w = w.add(32);
                        g += 1;
                    }
                    let a = _mm256_add_epi32(
                        _mm256_add_epi32(acc[0], acc[1]),
                        _mm256_add_epi32(acc[2], acc[3]),
                    );
                    store8(o.add(8 * b), s.add(8 * b), a);
                }
            }
        }

        #[doc = concat!($what, " `sparse_cgemv8x4`. Same contract as `vec_portable::sparse_cgemv8x4`.")]
        ///
        /// # Safety
        ///
        /// The processor must support the instructions named above. The
        /// index table must be valid for `rows` and `cols`, and `weights`
        /// must hold 32 bytes for each listed block, as `nnet` checks.
        #[target_feature(enable = $features)]
        pub unsafe fn $sparse(
            out: &mut [f32],
            weights: &[i8],
            idx: &[i32],
            scale: &[f32],
            rows: usize,
            cols: usize,
            x: &[f32],
        ) {
            // `quantize` writes `cols` bytes into a buffer of `MAX_INPUTS`;
            // each block of 8 rows stores 8 values. `nnet` checks the index
            // table (`sparse_blocks`); the debug checks below repeat that per
            // entry.
            assert!(
                rows.is_multiple_of(8)
                    && cols <= MAX_INPUTS
                    && x.len() >= cols
                    && out.len() >= rows
                    && scale.len() >= rows,
                "sparse_cgemv8x4: slices shorter than the layer"
            );
            let mut xq = core::mem::MaybeUninit::<XBuf>::uninit();
            let q = xq.as_mut_ptr() as *mut i8;
            let o = out.as_mut_ptr();
            let s = scale.as_ptr();
            let w = weights.as_ptr();
            // The next entry of `idx` and the next block of `weights`.
            let mut p = 0usize;
            let mut wo = 0usize;
            // SAFETY: the caller guarantees the table and the lengths, so
            // only bytes written by `quantize` are read.
            unsafe {
                quantize(q, x.as_ptr(), cols);
                for i in (0..rows).step_by(8) {
                    let mut acc = _mm256_setzero_si256();
                    let count = idx[p] as usize;
                    p += 1;
                    for _ in 0..count {
                        debug_assert!(p < idx.len() && wo + 32 <= weights.len());
                        let pos = *idx.get_unchecked(p) as usize;
                        debug_assert!(pos + 4 <= cols);
                        let vx = load_x4(q.add(pos));
                        acc = $dot(acc, _mm256_loadu_si256(w.add(wo) as *const __m256i), vx);
                        p += 1;
                        wo += 32;
                    }
                    store8(o.add(i), s.add(i), acc);
                }
            }
        }
    };
}

int8_kernels!("avx2,fma", dot_avx2, cgemv8x4_avx2, sparse_cgemv8x4_avx2, "AVX2 (`maddubs` and `madd`)");
int8_kernels!(
    "avx2,fma,avxvnni",
    dot_avxvnni,
    cgemv8x4_avxvnni,
    sparse_cgemv8x4_avxvnni,
    "AVX-VNNI (`_mm256_dpbusd_avx_epi32`)"
);

/// `dot_avx2` on two 8 × 4 blocks at a time with AVX-512 VNNI: the low
/// half of `w` and `x` is one column group, the high half the next one.
/// AVX-512 has no `sign_epi8`; the weights are negated where the input is
/// negative.
///
/// # Safety
///
/// AVX-512 F, BW and VNNI.
#[target_feature(enable = "avx2,fma,avx512f,avx512bw,avx512vnni")]
#[inline]
unsafe fn dot_avx512vnni(acc: __m512i, w: __m512i, x: __m512i) -> __m512i {
    let neg = _mm512_movepi8_mask(x);
    let sw = _mm512_mask_sub_epi8(w, neg, _mm512_setzero_si512(), w);
    _mm512_dpbusd_epi32(acc, _mm512_abs_epi8(x), sw)
}

/// The inputs of two column groups: 4 bytes at `q` in the low 8 lanes,
/// the next 4 in the high 8 lanes.
///
/// # Safety
///
/// AVX-512 F; `q` has 8 bytes.
#[target_feature(enable = "avx2,fma,avx512f,avx512bw,avx512vnni")]
#[inline]
unsafe fn load_x8(q: *const i8) -> __m512i {
    // SAFETY: the caller guarantees 8 readable bytes.
    unsafe {
        let lo = _mm256_set1_epi32(ptr::read_unaligned(q as *const i32));
        let hi = _mm256_set1_epi32(ptr::read_unaligned(q.add(4) as *const i32));
        _mm512_inserti64x4::<1>(_mm512_castsi256_si512(lo), hi)
    }
}

/// The sum of the two halves of a 512-bit accumulator: the 8 rows.
///
/// # Safety
///
/// AVX-512 F.
#[target_feature(enable = "avx2,fma,avx512f,avx512bw,avx512vnni")]
#[inline]
unsafe fn fold512(acc: __m512i) -> __m256i {
    _mm256_add_epi32(_mm512_castsi512_si256(acc), _mm512_extracti64x4_epi64::<1>(acc))
}

/// AVX-512 VNNI `cgemv8x4` (`_mm512_dpbusd_epi32` on two 8 × 4 blocks at
/// a time). Same contract as `vec_portable::cgemv8x4`.
///
/// # Safety
///
/// The processor must support AVX2, FMA, AVX-512 F, AVX-512 BW and
/// AVX-512 VNNI. The slices must have the lengths that
/// `vec_portable::cgemv8x4` requires.
#[target_feature(enable = "avx2,fma,avx512f,avx512bw,avx512vnni")]
pub unsafe fn cgemv8x4_avx512vnni(
    out: &mut [f32],
    weights: &[i8],
    scale: &[f32],
    rows: usize,
    cols: usize,
    x: &[f32],
) {
    // `quantize` writes `cols` bytes into a buffer of `MAX_INPUTS`;
    // block `b` stores rows `8 * b..8 * b + 8`; the weights are read
    // up to `rows * cols`.
    assert!(
        cols <= MAX_INPUTS
            && x.len() >= cols
            && out.len() >= rows
            && scale.len() >= rows
            && rows.checked_mul(cols).is_some_and(|n| weights.len() >= n),
        "cgemv8x4: slices shorter than the layer"
    );
    let mut xq = core::mem::MaybeUninit::<XBuf>::uninit();
    let q = xq.as_mut_ptr() as *mut i8;
    let o = out.as_mut_ptr();
    let s = scale.as_ptr();
    let bstride = cols * 8;
    let groups = cols / 4;
    // SAFETY: the caller guarantees the lengths; only the `cols` bytes
    // written by `quantize` are read.
    unsafe {
        quantize(q, x.as_ptr(), cols);
        for b in 0..rows / 8 {
            let w = weights.as_ptr().add(b * bstride);
            let mut acc = _mm512_setzero_si512();
            let mut g = 0;
            while g + 2 <= groups {
                let vx = load_x8(q.add(4 * g));
                acc = dot_avx512vnni(acc, _mm512_loadu_si512(w.add(32 * g) as *const _), vx);
                g += 2;
            }
            let mut acc = fold512(acc);
            if g < groups {
                let vx = load_x4(q.add(4 * g));
                acc = dot_avx2(acc, _mm256_loadu_si256(w.add(32 * g) as *const __m256i), vx);
            }
            store8(o.add(8 * b), s.add(8 * b), acc);
        }
    }
}

/// AVX-512 VNNI `sparse_cgemv8x4`. Same contract as
/// `vec_portable::sparse_cgemv8x4`.
///
/// # Safety
///
/// The processor must support AVX2, FMA, AVX-512 F, AVX-512 BW and
/// AVX-512 VNNI. The index table must be valid for `rows` and `cols`, and
/// `weights` must hold 32 bytes for each listed block, as `nnet` checks.
#[target_feature(enable = "avx2,fma,avx512f,avx512bw,avx512vnni")]
pub unsafe fn sparse_cgemv8x4_avx512vnni(
    out: &mut [f32],
    weights: &[i8],
    idx: &[i32],
    scale: &[f32],
    rows: usize,
    cols: usize,
    x: &[f32],
) {
    // `quantize` writes `cols` bytes into a buffer of `MAX_INPUTS`;
    // each block of 8 rows stores 8 values. `nnet` checks the index
    // table (`sparse_blocks`); the debug checks below repeat that per
    // entry.
    assert!(
        rows.is_multiple_of(8)
            && cols <= MAX_INPUTS
            && x.len() >= cols
            && out.len() >= rows
            && scale.len() >= rows,
        "sparse_cgemv8x4: slices shorter than the layer"
    );
    let mut xq = core::mem::MaybeUninit::<XBuf>::uninit();
    let q = xq.as_mut_ptr() as *mut i8;
    let o = out.as_mut_ptr();
    let s = scale.as_ptr();
    let w = weights.as_ptr();
    // The next entry of `idx` and the next block of `weights`.
    let mut p = 0usize;
    let mut wo = 0usize;
    // SAFETY: the caller guarantees the table and the lengths.
    unsafe {
        quantize(q, x.as_ptr(), cols);
        for i in (0..rows).step_by(8) {
            let mut acc = _mm512_setzero_si512();
            let count = idx[p] as usize;
            p += 1;
            let mut c = 0;
            while c + 2 <= count {
                debug_assert!(p + 1 < idx.len() && wo + 64 <= weights.len());
                let pos0 = *idx.get_unchecked(p) as usize;
                let pos1 = *idx.get_unchecked(p + 1) as usize;
                debug_assert!(pos0 + 4 <= cols && pos1 + 4 <= cols);
                let lo = _mm256_set1_epi32(ptr::read_unaligned(q.add(pos0) as *const i32));
                let hi = _mm256_set1_epi32(ptr::read_unaligned(q.add(pos1) as *const i32));
                let vx = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(lo), hi);
                acc = dot_avx512vnni(acc, _mm512_loadu_si512(w.add(wo) as *const _), vx);
                p += 2;
                wo += 64;
                c += 2;
            }
            let mut acc = fold512(acc);
            if c < count {
                // Checked: at most once per block of 8 rows.
                let pos = idx[p] as usize;
                debug_assert!(pos + 4 <= cols && wo + 32 <= weights.len());
                let vx = load_x4(q.add(pos));
                acc = dot_avx2(acc, _mm256_loadu_si256(w.add(wo) as *const __m256i), vx);
                p += 1;
                wo += 32;
            }
            store8(o.add(i), s.add(i), acc);
        }
    }
}

// ---------------------------------------------------------------------------
// Activations
// ---------------------------------------------------------------------------

/// FRECPE on 8 lanes (`vec_portable::frecpe_bits`).
///
/// The 8-bit estimate is `(b + 1) / 2` with `b = floor(2^19 / (2a + 1))`,
/// where `a` is 256 plus the top 8 fraction bits. The float division gives
/// the exact `floor`: `2a + 1` is odd and at most 1023, so the quotient is
/// never an integer and lies more than 1/1023 below the next one, far more
/// than the rounding error.
///
/// Normal inputs with an exponent field from 1 to 252, infinities and NaN
/// are computed here. Other lanes (zeros, subnormals, and inputs from 2^126
/// up, whose estimate is subnormal) are rare; when one occurs, the whole
/// vector goes through `vec_portable::frecpe_bits`.
///
/// # Safety
///
/// AVX2.
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn frecpe8(d: __m256) -> __m256 {
    let bits = _mm256_castps_si256(d);
    let sign = _mm256_and_si256(bits, _mm256_set1_epi32(i32::MIN));
    let e = _mm256_and_si256(_mm256_srli_epi32::<23>(bits), _mm256_set1_epi32(0xff));
    let a = _mm256_or_si256(
        _mm256_and_si256(_mm256_srli_epi32::<15>(bits), _mm256_set1_epi32(0xff)),
        _mm256_set1_epi32(0x100),
    );
    let den = _mm256_cvtepi32_ps(_mm256_add_epi32(_mm256_add_epi32(a, a), _mm256_set1_epi32(1)));
    let b = _mm256_cvttps_epi32(_mm256_div_ps(_mm256_set1_ps(524_288.0), den));
    let est = _mm256_srli_epi32::<1>(_mm256_add_epi32(b, _mm256_set1_epi32(1)));
    let rexp = _mm256_sub_epi32(_mm256_set1_epi32(253), e);
    let normal = _mm256_or_si256(
        _mm256_or_si256(sign, _mm256_slli_epi32::<23>(rexp)),
        _mm256_slli_epi32::<15>(_mm256_and_si256(est, _mm256_set1_epi32(0xff))),
    );
    // Exponent field 255: NaN (quieted) or infinity (zero of its sign).
    let is_max = _mm256_cmpeq_epi32(e, _mm256_set1_epi32(0xff));
    let frac_zero = _mm256_cmpeq_epi32(
        _mm256_and_si256(bits, _mm256_set1_epi32(0x007f_ffff)),
        _mm256_setzero_si256(),
    );
    let special = _mm256_blendv_epi8(
        _mm256_or_si256(bits, _mm256_set1_epi32(0x0040_0000)),
        sign,
        frac_zero,
    );
    // Lanes this path does not cover: exponent field 0, or 253 and 254.
    let low = _mm256_cmpeq_epi32(e, _mm256_setzero_si256());
    let high = _mm256_and_si256(
        _mm256_cmpgt_epi32(e, _mm256_set1_epi32(252)),
        _mm256_cmpgt_epi32(_mm256_set1_epi32(255), e),
    );
    if _mm256_movemask_epi8(_mm256_or_si256(low, high)) != 0 {
        let mut v = [0u32; 8];
        // SAFETY: `v` has 8 lanes.
        unsafe { _mm256_storeu_si256(v.as_mut_ptr() as *mut __m256i, bits) };
        for lane in v.iter_mut() {
            *lane = vec_portable::frecpe_bits(*lane);
        }
        // SAFETY: as above.
        return unsafe { _mm256_castsi256_ps(_mm256_loadu_si256(v.as_ptr() as *const __m256i)) };
    }
    _mm256_castsi256_ps(_mm256_blendv_epi8(normal, special, is_max))
}

/// `exp4_approx` on 8 lanes.
///
/// # Safety
///
/// AVX2 and FMA.
#[target_feature(enable = "avx2,fma")]
#[inline]
#[allow(clippy::excessive_precision, clippy::approx_constant)]
unsafe fn exp8(x: __m256) -> __m256 {
    // `min` and `max` with the constant first return the input when it is
    // NaN, like FMIN and FMAX.
    let x = _mm256_max_ps(_mm256_set1_ps(-88.0), _mm256_min_ps(_mm256_set1_ps(88.0), x));
    let x = _mm256_fmadd_ps(x, _mm256_set1_ps(1.44269504), _mm256_set1_ps(127.0));
    // The input lies in [0.04, 254] or is NaN. For NaN x86 gives i32::MIN
    // where ARM gives 0; both lead to a NaN result.
    let i = _mm256_cvttps_epi32(x);
    let x = _mm256_sub_ps(x, _mm256_cvtepi32_ps(i));
    let k0 = _mm256_set1_ps(0.99992522);
    let k1 = _mm256_set1_ps(0.69583354);
    let k2 = _mm256_set1_ps(0.22606716);
    let k3 = _mm256_set1_ps(0.078024523);
    let y = _mm256_fmadd_ps(x, _mm256_fmadd_ps(x, _mm256_fmadd_ps(k3, x, k2), k1), k0);
    _mm256_mul_ps(y, _mm256_castsi256_ps(_mm256_slli_epi32::<23>(i)))
}

/// `tanh4_approx` on 8 lanes.
///
/// # Safety
///
/// AVX2 and FMA.
#[target_feature(enable = "avx2,fma")]
#[inline]
#[allow(clippy::excessive_precision)]
unsafe fn tanh8(x: __m256) -> __m256 {
    let x2 = _mm256_mul_ps(x, x);
    let num = _mm256_fmadd_ps(
        x2,
        _mm256_fmadd_ps(_mm256_set1_ps(0.60863042), x2, _mm256_set1_ps(96.39235687)),
        _mm256_set1_ps(952.52801514),
    );
    let den = _mm256_fmadd_ps(
        x2,
        _mm256_fmadd_ps(_mm256_set1_ps(11.88600922), x2, _mm256_set1_ps(413.36801147)),
        _mm256_set1_ps(952.72399902),
    );
    let num = _mm256_mul_ps(num, x);
    // SAFETY: AVX2 is enabled.
    let num = _mm256_mul_ps(num, unsafe { frecpe8(den) });
    _mm256_max_ps(_mm256_set1_ps(-1.0), _mm256_min_ps(_mm256_set1_ps(1.0), num))
}

/// `sigmoid4_approx` on 8 lanes.
///
/// # Safety
///
/// AVX2 and FMA.
#[target_feature(enable = "avx2,fma")]
#[inline]
#[allow(clippy::excessive_precision)]
unsafe fn sigmoid8(x: __m256) -> __m256 {
    let x2 = _mm256_mul_ps(x, x);
    let num = _mm256_fmadd_ps(
        x2,
        _mm256_fmadd_ps(_mm256_set1_ps(0.00950985), x2, _mm256_set1_ps(6.02452230)),
        _mm256_set1_ps(238.13200378),
    );
    let den = _mm256_fmadd_ps(
        x2,
        _mm256_fmadd_ps(_mm256_set1_ps(0.74287558), x2, _mm256_set1_ps(103.34200287)),
        _mm256_set1_ps(952.72399902),
    );
    let num = _mm256_mul_ps(num, x);
    // SAFETY: AVX2 is enabled.
    let num = _mm256_fmadd_ps(num, unsafe { frecpe8(den) }, _mm256_set1_ps(0.5));
    // The result is never -0.0, so the order of `max` for two zeros does
    // not matter here.
    _mm256_max_ps(_mm256_setzero_ps(), _mm256_min_ps(_mm256_set1_ps(1.0), num))
}

/// AVX2 `vec_exp`. Same contract as `vec_portable::vec_exp`.
///
/// # Safety
///
/// The processor must support AVX2 and FMA.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn vec_exp_avx2(buf: &mut [f32]) {
    let n8 = buf.len() & !7;
    let p = buf.as_mut_ptr();
    // SAFETY: `i + 8 <= buf.len()`.
    unsafe {
        for i in (0..n8).step_by(8) {
            _mm256_storeu_ps(p.add(i), exp8(_mm256_loadu_ps(p.add(i))));
        }
    }
    for v in &mut buf[n8..] {
        *v = vec_portable::lpcnet_exp(*v);
    }
}

/// AVX2 `vec_tanh`. Same contract as `vec_portable::vec_tanh`.
///
/// # Safety
///
/// The processor must support AVX2 and FMA.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn vec_tanh_avx2(buf: &mut [f32]) {
    let groups = buf.len() & !3;
    let n8 = buf.len() & !7;
    let p = buf.as_mut_ptr();
    // SAFETY: `i + 8 <= buf.len()`.
    unsafe {
        for i in (0..n8).step_by(8) {
            _mm256_storeu_ps(p.add(i), tanh8(_mm256_loadu_ps(p.add(i))));
        }
    }
    // A last group of 4 with the group formula, then the tail formula.
    for v in &mut buf[n8..groups] {
        *v = vec_portable::tanh_approx(*v);
    }
    for v in &mut buf[groups..] {
        *v = vec_portable::tanh_tail(*v);
    }
}

/// AVX2 `vec_sigmoid`. Same contract as `vec_portable::vec_sigmoid`.
///
/// # Safety
///
/// The processor must support AVX2 and FMA.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn vec_sigmoid_avx2(buf: &mut [f32]) {
    let groups = buf.len() & !3;
    let n8 = buf.len() & !7;
    let p = buf.as_mut_ptr();
    // SAFETY: `i + 8 <= buf.len()`.
    unsafe {
        for i in (0..n8).step_by(8) {
            _mm256_storeu_ps(p.add(i), sigmoid8(_mm256_loadu_ps(p.add(i))));
        }
    }
    for v in &mut buf[n8..groups] {
        *v = vec_portable::sigmoid_approx(*v);
    }
    for v in &mut buf[groups..] {
        *v = vec_portable::sigmoid_tail(*v);
    }
}
