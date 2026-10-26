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
//!
//! The kernels are safe functions compiled with the target features they
//! use. They reach the slices through arrays (`as_chunks`,
//! `chunks_exact`), loaded and stored with the functions of
//! `crate::simd_mem`. Each kernel checks the lengths it needs and panics
//! when a slice is too short. The kernels of sparse layers also check the
//! index table (`nnet::check_sparse`); after that check they read the
//! weights and the inputs at the listed positions without index checks.
//! Only `quantize`, the conversion of the inputs of the int8 layers, keeps
//! the pointer loads and stores of its loop (see there).

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;
use core::mem::MaybeUninit;
use core::iter::Zip;
use core::slice::{self, ChunksExact};

use super::MAX_INPUTS;
use super::nnet::check_sparse;
use super::vec_portable;
use crate::simd_mem::array_at_unchecked;
use crate::simd_mem::x86::{
    load_f32x4, load_f32x8, load_i8x32, load_i8x64, load_i32x8, store_f32x4, store_f32x8,
    store_i32x8,
};

// ---------------------------------------------------------------------------
// Float products
// ---------------------------------------------------------------------------

/// The columns of an `sgemv` product before the last one, each with its
/// input: `(x[j], column j)`. Column `j` holds the weight of row `r` at
/// index `r` and has `stride` values. `sgemv` builds this iterator once
/// and gives each pass over the rows a copy.
type Columns<'a> = Zip<slice::Iter<'a, f32>, ChunksExact<'a, f32>>;

/// The products of one column for `V` vectors of 8 rows:
/// `acc[v] += w[v] · xj`, one fused multiply-add per lane.
#[target_feature(enable = "avx2,fma")]
#[inline]
fn sgemv_column<const V: usize>(acc: &mut [__m256; V], w: &[[f32; 8]; V], xj: &f32) {
    let xj = _mm256_broadcast_ss(xj);
    for (a, w) in acc.iter_mut().zip(w) {
        *a = _mm256_fmadd_ps(load_f32x8(w), xj, *a);
    }
}

/// One pass of `sgemv` over `V` vectors of 8 rows: `out` gets the rows
/// `i..i + 8·V`.
///
/// `columns` yields the columns before the last one with their inputs.
/// `last` holds the weights of these rows in the last column, and `x_last`
/// is its input.
///
/// Panics when `stride < i + 8·V`.
#[target_feature(enable = "avx2,fma")]
#[inline]
fn sgemv_pass<const V: usize>(
    out: &mut [[f32; 8]; V],
    i: usize,
    columns: Columns<'_>,
    stride: usize,
    last: &[[f32; 8]; V],
    x_last: &f32,
) {
    // Each column of `columns` holds `stride` values, so this check covers
    // the reads of all of them.
    assert!(i <= stride && 8 * V <= stride - i);
    let mut acc = [_mm256_setzero_ps(); V];
    for (xj, col) in columns {
        let w = col[i..][..8 * V].as_chunks::<8>().0.try_into().unwrap();
        sgemv_column(&mut acc, w, xj);
    }
    sgemv_column(&mut acc, last, x_last);
    for (o, a) in out.iter_mut().zip(&acc) {
        store_f32x8(o, *a);
    }
}

/// 4 rows of `sgemv` in one 128-bit register: `out` gets the rows
/// `i..i + 4`. The other arguments are those of `sgemv_pass`.
///
/// Panics when `stride < i + 4`.
#[target_feature(enable = "avx2,fma")]
#[inline]
fn sgemv_quad(
    out: &mut [f32; 4],
    i: usize,
    columns: Columns<'_>,
    stride: usize,
    last: &[f32; 4],
    x_last: &f32,
) {
    // As in `sgemv_pass`.
    assert!(i <= stride && 4 <= stride - i);
    let mut acc = _mm_setzero_ps();
    for (&xj, col) in columns {
        acc = _mm_fmadd_ps(load_f32x4(col[i..].first_chunk().unwrap()), _mm_set1_ps(xj), acc);
    }
    acc = _mm_fmadd_ps(load_f32x4(last), _mm_set1_ps(*x_last), acc);
    store_f32x4(out, acc);
}

/// The products of one column for the last 1 to 3 rows, one scalar chain
/// each: `acc[k] = fma(w[k], xj, acc[k])`.
///
/// Panics when `w` and `acc` differ in length.
#[target_feature(enable = "avx2,fma")]
#[inline]
fn sgemv_tail_column(acc: &mut [__m128], w: &[f32], xj: f32) {
    let xj = _mm_set_ss(xj);
    let w = &w[..acc.len()];
    for (a, &wk) in acc.iter_mut().zip(w) {
        *a = _mm_fmadd_ss(_mm_set_ss(wk), xj, *a);
    }
}

/// The last 1 to 3 rows of `sgemv`, one scalar chain each: `out` gets the
/// rows `i..i + out.len()`. The other arguments are those of
/// `sgemv_pass`.
///
/// Panics when `out` holds more than 3 values, when `last` and `out`
/// differ in length, or when `stride < i + out.len()`.
#[target_feature(enable = "avx2,fma")]
#[inline]
fn sgemv_tail(
    out: &mut [f32],
    i: usize,
    columns: Columns<'_>,
    stride: usize,
    last: &[f32],
    x_last: &f32,
) {
    let r = out.len();
    // As in `sgemv_pass`.
    assert!(i <= stride && r <= stride - i);
    let mut acc = [_mm_setzero_ps(); 3];
    let acc = &mut acc[..r];
    for (&xj, col) in columns {
        sgemv_tail_column(acc, &col[i..], xj);
    }
    sgemv_tail_column(acc, last, *x_last);
    for (o, a) in out.iter_mut().zip(acc.iter()) {
        *o = _mm_cvtss_f32(*a);
    }
}

/// AVX2 `sgemv`. Same contract as `vec_portable::sgemv`.
///
/// Panics when a slice is shorter than `vec_portable::sgemv` requires,
/// when `rows > 0` and `cols == 0`, and when `col_stride < rows`.
///
/// # Safety
///
/// The processor must support AVX2 and FMA.
#[target_feature(enable = "avx2,fma")]
pub fn sgemv_avx2(
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
                    && col_stride >= rows
                    && (cols - 1)
                        .checked_mul(col_stride)
                        .and_then(|n| n.checked_add(rows))
                        .is_some_and(|n| weights.len() >= n)),
        "sgemv_avx2: slices shorter than the layer"
    );
    if rows == 0 {
        return;
    }
    let out = &mut out[..rows];
    let Some((x_last, x_body)) = x[..cols].split_last() else {
        // Not reached: with `rows > 0` the check above needs `cols > 0`.
        return;
    };
    // The columns before the last one, `col_stride` values each, and the
    // last one, which only needs `rows` values. The passes over the rows
    // split `out` and the last column alike.
    let (body, last) = weights.split_at(x_body.len() * col_stride);
    let columns = x_body.iter().zip(body.chunks_exact(col_stride));
    let (o32, o_rest) = out.as_chunks_mut::<32>();
    let (l32, l_rest) = last[..rows].as_chunks::<32>();
    for (k, (o, l)) in o32.iter_mut().zip(l32).enumerate() {
        let o = o.as_chunks_mut::<8>().0.try_into().unwrap();
        let l = l.as_chunks::<8>().0.try_into().unwrap();
        sgemv_pass::<4>(o, 32 * k, columns.clone(), col_stride, l, x_last);
    }
    let mut i = 32 * o32.len();
    let (o8, o_rest) = o_rest.as_chunks_mut::<8>();
    let (l8, l_rest) = l_rest.as_chunks::<8>();
    let c = columns.clone();
    match o8.len() {
        3 => sgemv_pass::<3>(o8.try_into().unwrap(), i, c, col_stride, l8.try_into().unwrap(), x_last),
        2 => sgemv_pass::<2>(o8.try_into().unwrap(), i, c, col_stride, l8.try_into().unwrap(), x_last),
        1 => sgemv_pass::<1>(o8.try_into().unwrap(), i, c, col_stride, l8.try_into().unwrap(), x_last),
        _ => {}
    }
    i += 8 * o8.len();
    let (o4, o_tail) = o_rest.as_chunks_mut::<4>();
    let (l4, l_tail) = l_rest.as_chunks::<4>();
    if let (Some(o), Some(l)) = (o4.first_mut(), l4.first()) {
        sgemv_quad(o, i, columns.clone(), col_stride, l, x_last);
        i += 4;
    }
    if !o_tail.is_empty() {
        sgemv_tail(o_tail, i, columns, col_stride, l_tail, x_last);
    }
}

/// AVX2 `sparse_sgemv8x4`. Same contract as
/// `vec_portable::sparse_sgemv8x4`.
///
/// Panics when `rows` is not a multiple of 8, when `out` holds fewer than
/// `rows` values, or when the index table or the weights fail
/// `nnet::check_sparse` for `rows` rows and inputs of the length of `x`.
///
/// # Safety
///
/// The processor must support AVX2 and FMA.
#[target_feature(enable = "avx2,fma")]
pub fn sparse_sgemv8x4_avx2(
    out: &mut [f32],
    weights: &[f32],
    idx: &[i32],
    rows: usize,
    x: &[f32],
) {
    // Each block of 8 rows stores 8 values. `check_sparse` checks the index
    // table; the debug checks below repeat that per entry.
    assert!(rows.is_multiple_of(8) && out.len() >= rows);
    check_sparse(idx, rows, x.len(), weights.len());
    // The entries of the table in order, and the offset of the next block
    // of weights.
    let mut table = idx.iter();
    let mut wo = 0usize;
    for o in out[..rows].as_chunks_mut::<8>().0 {
        let mut y = _mm256_setzero_ps();
        let count = *table.next().unwrap() as usize;
        for _ in 0..count {
            debug_assert!(!table.as_slice().is_empty() && wo + 32 <= weights.len());
            // SAFETY: `check_sparse` checked that `count` positions follow
            // the count of this block.
            let pos = unsafe { *table.next().unwrap_unchecked() } as usize;
            debug_assert!(pos + 4 <= x.len());
            // SAFETY: `check_sparse` checked that each listed position has
            // `pos + 4 <= x.len()`, and that the weights hold a block of 32
            // values for each listed position, so the block at `wo` is
            // inside them.
            let (xa, w) =
                unsafe { (array_at_unchecked::<f32, 4>(x, pos), array_at_unchecked::<f32, 32>(weights, wo)) };
            for (&xc, w) in xa.iter().zip(w.as_chunks::<8>().0) {
                let xc = _mm256_set1_ps(xc);
                y = _mm256_fmadd_ps(load_f32x8(w), xc, y);
            }
            wo += 32;
        }
        store_f32x8(o, y);
    }
}

// ---------------------------------------------------------------------------
// int8 products
// ---------------------------------------------------------------------------

/// The quantized input of an int8 product.
#[repr(C, align(32))]
struct XBuf([MaybeUninit<i8>; MAX_INPUTS]);

/// `xq[j] = quantize(x[j])` for every `j`, with the ARM results. Returns
/// `xq`, which then holds values.
///
/// Panics when `xq` and `x` differ in length.
#[target_feature(enable = "avx2,fma")]
#[inline]
fn quantize<'a>(xq: &'a mut [MaybeUninit<i8>], x: &[f32]) -> &'a [i8] {
    assert_eq!(xq.len(), x.len());
    let c127 = _mm256_set1_ps(127.0);
    let two31 = _mm256_set1_ps(2_147_483_648.0);
    // Byte 0 of each 32-bit lane, in each 128-bit half.
    let pick = _mm256_setr_epi8(
        0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 4, 8, 12, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1,
    );
    let join = _mm256_setr_epi32(0, 4, 1, 1, 1, 1, 1, 1);
    let cols = x.len();
    let (q, xp) = (xq.as_mut_ptr().cast::<i8>(), x.as_ptr());
    // The loops keep the pointer loads and stores of the original code:
    // with the groups of eight taken from `as_chunks`, the compiler counted
    // the steps with an extra compare.
    // SAFETY: a step reads `x[j..j + 8]` and writes `xq[j..j + 8]` with
    // `j + 8 <= cols`, and the last loop reads and writes index `j < cols`;
    // both slices hold `cols` values (checked above). Writing a value into
    // `MaybeUninit` is always allowed.
    unsafe {
        let mut j = 0;
        while j + 8 <= cols {
            let p = _mm256_mul_ps(c127, _mm256_loadu_ps(xp.add(j)));
            let i = _mm256_cvtps_epi32(p);
            // ARM saturates to i32::MAX, whose low byte is 0xff.
            let big = _mm256_castps_si256(_mm256_cmp_ps::<_CMP_GE_OQ>(p, two31));
            let i = _mm256_or_si256(i, big);
            let b = _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8(i, pick), join);
            _mm_storel_epi64(q.add(j).cast(), _mm256_castsi256_si128(b));
            j += 8;
        }
        while j < cols {
            *q.add(j) = vec_portable::quantize(*xp.add(j));
            j += 1;
        }
    }
    // SAFETY: `xq` and `x` have the same length, so the two loops above
    // wrote every value of `xq`: the groups of 8, then the last values one
    // by one.
    unsafe { xq.assume_init_ref() }
}

/// The 4 quantized inputs `q` as one 32-bit value (in memory order).
#[inline(always)]
fn x4_bits(q: &[i8; 4]) -> i32 {
    i32::from_ne_bytes(q.map(|b| b as u8))
}

/// The 4 quantized inputs `q`, repeated in each 32-bit lane.
#[target_feature(enable = "avx2,fma")]
#[inline]
fn load_x4(q: &[i8; 4]) -> __m256i {
    _mm256_set1_epi32(x4_bits(q))
}

/// The dot products of the 8 rows of one 8 × 4 block with 4 inputs,
/// exactly: `maddubs(|x|, sign(w, x))` gives 16-bit pairs, `madd` with ones
/// adds the pairs of each row.
#[target_feature(enable = "avx2,fma")]
#[inline]
fn dot_avx2(acc: __m256i, w: __m256i, x: __m256i) -> __m256i {
    let t = _mm256_maddubs_epi16(_mm256_abs_epi8(x), _mm256_sign_epi8(w, x));
    _mm256_add_epi32(acc, _mm256_madd_epi16(t, _mm256_set1_epi16(1)))
}

/// `dot_avx2` with AVX-VNNI: `dpbusd` adds the 4 products of each row
/// without an intermediate 16-bit sum.
#[target_feature(enable = "avx2,fma,avxvnni")]
#[inline]
fn dot_avxvnni(acc: __m256i, w: __m256i, x: __m256i) -> __m256i {
    _mm256_dpbusd_avx_epi32(acc, _mm256_abs_epi8(x), _mm256_sign_epi8(w, x))
}

/// `out[k] = scale[k] · acc[k]` for the 8 rows of one block.
#[target_feature(enable = "avx2,fma")]
#[inline]
fn store8(out: &mut [f32; 8], scale: &[f32; 8], acc: __m256i) {
    store_f32x8(out, _mm256_mul_ps(load_f32x8(scale), _mm256_cvtepi32_ps(acc)))
}

/// Defines a dense and a sparse int8 product on one 256-bit dot product.
macro_rules! int8_kernels {
    ($features:literal, $dot:ident, $dense:ident, $sparse:ident, $what:literal) => {
        #[doc = concat!($what, " `cgemv8x4`. Same contract as `vec_portable::cgemv8x4`.")]
        ///
        /// Panics when a slice is shorter than `vec_portable::cgemv8x4`
        /// requires, or when `cols > MAX_INPUTS`.
        ///
        /// # Safety
        ///
        /// The processor must support the instructions named above.
        #[target_feature(enable = $features)]
        pub fn $dense(
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
            let mut xq = XBuf([MaybeUninit::uninit(); MAX_INPUTS]);
            let xq = quantize(&mut xq.0[..cols], &x[..cols]);
            let out = out[..rows].as_chunks_mut::<8>().0;
            let scale = scale[..rows].as_chunks::<8>().0;
            let groups = cols / 4;
            if groups == 0 {
                // No column group: every sum is 0.
                for (o, s) in out.iter_mut().zip(scale) {
                    store8(o, s, _mm256_setzero_si256());
                }
                return;
            }
            // One block of 8 rows at a time, so the weights are one
            // sequential stream (several streams were slower on 64-bit
            // ARM when the weights are not in the caches); 4 column
            // groups on separate accumulators.
            let (x16, x_rest) = xq.as_chunks::<16>();
            let x4 = x_rest.as_chunks::<4>().0;
            // The weights of a block of 8 rows: one 8 × 4 block (32 bytes)
            // for each column group.
            let wlen = 32 * groups;
            let wblocks = weights[..out.len() * wlen].chunks_exact(wlen);
            for ((o, s), w) in out.iter_mut().zip(scale).zip(wblocks) {
                let mut acc = [_mm256_setzero_si256(); 4];
                let (w128, w_rest) = w.as_chunks::<128>();
                for (xg, w) in x16.iter().zip(w128) {
                    let xg = xg.as_chunks::<4>().0;
                    let w = w.as_chunks::<32>().0;
                    for ((a, xg), w) in acc.iter_mut().zip(xg).zip(w) {
                        let vx = load_x4(xg);
                        *a = $dot(*a, load_i8x32(w), vx);
                    }
                }
                for (xg, w) in x4.iter().zip(w_rest.as_chunks::<32>().0) {
                    let vx = load_x4(xg);
                    acc[0] = $dot(acc[0], load_i8x32(w), vx);
                }
                let a = _mm256_add_epi32(
                    _mm256_add_epi32(acc[0], acc[1]),
                    _mm256_add_epi32(acc[2], acc[3]),
                );
                store8(o, s, a);
            }
        }

        #[doc = concat!($what, " `sparse_cgemv8x4`. Same contract as `vec_portable::sparse_cgemv8x4`.")]
        ///
        /// Panics when a slice is shorter than the layer, when `rows` is
        /// not a multiple of 8, when `cols > MAX_INPUTS`, or when the index
        /// table or the weights fail `nnet::check_sparse` for `rows` rows
        /// and `cols` inputs.
        ///
        /// # Safety
        ///
        /// The processor must support the instructions named above.
        #[target_feature(enable = $features)]
        pub fn $sparse(
            out: &mut [f32],
            weights: &[i8],
            idx: &[i32],
            scale: &[f32],
            rows: usize,
            cols: usize,
            x: &[f32],
        ) {
            // `quantize` writes `cols` bytes into a buffer of `MAX_INPUTS`;
            // each block of 8 rows stores 8 values. `check_sparse` checks
            // the index table; the debug checks below repeat that per
            // entry.
            assert!(
                rows.is_multiple_of(8)
                    && cols <= MAX_INPUTS
                    && x.len() >= cols
                    && out.len() >= rows
                    && scale.len() >= rows,
                "sparse_cgemv8x4: slices shorter than the layer"
            );
            check_sparse(idx, rows, cols, weights.len());
            let mut xq = XBuf([MaybeUninit::uninit(); MAX_INPUTS]);
            let xq = quantize(&mut xq.0[..cols], &x[..cols]);
            let mut out = out[..rows].as_chunks_mut::<8>().0;
            let mut scale = scale[..rows].as_chunks::<8>().0;
            // The entries of the table in order, and the offset of the
            // weights of the next block of 8 rows.
            let mut table = idx.iter();
            let mut wo = 0usize;
            // The blocks of the output and of the scales are taken apart
            // from the front, one step at a time. With a zip of the two,
            // the compiler addressed them by block number, which took
            // three more instructions per block.
            while let ([o, out_rest @ ..], [s, s_rest @ ..]) = (core::mem::take(&mut out), scale) {
                (out, scale) = (out_rest, s_rest);
                let mut acc = _mm256_setzero_si256();
                let count = *table.next().unwrap() as usize;
                debug_assert!(wo + 32 * count <= weights.len());
                // SAFETY: `check_sparse` checked that the weights hold a
                // block of 32 bytes for each listed position.
                let wrow = unsafe { weights.get_unchecked(wo..wo + 32 * count) };
                wo += 32 * count;
                for w in wrow.as_chunks::<32>().0 {
                    debug_assert!(!table.as_slice().is_empty());
                    // SAFETY: `check_sparse` checked that `count` positions
                    // follow the count of this block.
                    let pos = unsafe { *table.next().unwrap_unchecked() } as usize;
                    debug_assert!(pos + 4 <= cols);
                    // SAFETY: `check_sparse` checked that each listed
                    // position has `pos + 4 <= cols` (`xq` holds `cols`
                    // values).
                    let xg = unsafe { array_at_unchecked::<i8, 4>(xq, pos) };
                    let vx = load_x4(xg);
                    acc = $dot(acc, load_i8x32(w), vx);
                }
                store8(o, s, acc);
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
#[target_feature(enable = "avx2,fma,avx512f,avx512bw,avx512vnni")]
#[inline]
fn dot_avx512vnni(acc: __m512i, w: __m512i, x: __m512i) -> __m512i {
    let neg = _mm512_movepi8_mask(x);
    let sw = _mm512_mask_sub_epi8(w, neg, _mm512_setzero_si512(), w);
    _mm512_dpbusd_epi32(acc, _mm512_abs_epi8(x), sw)
}

/// The inputs of two column groups: the first 4 bytes of `q` in the low 8
/// lanes, the next 4 in the high 8 lanes.
#[target_feature(enable = "avx2,fma,avx512f,avx512bw,avx512vnni")]
#[inline]
fn load_x8(q: &[i8; 8]) -> __m512i {
    let (q, _) = q.as_chunks::<4>();
    let lo = _mm256_set1_epi32(x4_bits(&q[0]));
    let hi = _mm256_set1_epi32(x4_bits(&q[1]));
    _mm512_inserti64x4::<1>(_mm512_castsi256_si512(lo), hi)
}

/// The sum of the two halves of a 512-bit accumulator: the 8 rows.
#[target_feature(enable = "avx2,fma,avx512f,avx512bw,avx512vnni")]
#[inline]
fn fold512(acc: __m512i) -> __m256i {
    _mm256_add_epi32(_mm512_castsi512_si256(acc), _mm512_extracti64x4_epi64::<1>(acc))
}

/// AVX-512 VNNI `cgemv8x4` (`_mm512_dpbusd_epi32` on two 8 × 4 blocks at
/// a time). Same contract as `vec_portable::cgemv8x4`.
///
/// Panics when a slice is shorter than `vec_portable::cgemv8x4` requires,
/// or when `cols > MAX_INPUTS`.
///
/// # Safety
///
/// The processor must support AVX2, FMA, AVX-512 F, AVX-512 BW and
/// AVX-512 VNNI.
#[target_feature(enable = "avx2,fma,avx512f,avx512bw,avx512vnni")]
pub fn cgemv8x4_avx512vnni(
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
    let mut xq = XBuf([MaybeUninit::uninit(); MAX_INPUTS]);
    let xq = quantize(&mut xq.0[..cols], &x[..cols]);
    let out = out[..rows].as_chunks_mut::<8>().0;
    let scale = scale[..rows].as_chunks::<8>().0;
    let bstride = cols * 8;
    if bstride == 0 {
        // No inputs: every sum is 0.
        for (o, s) in out.iter_mut().zip(scale) {
            store8(o, s, _mm256_setzero_si256());
        }
        return;
    }
    // The inputs of 2 column groups at a time, then those of the last
    // group when their number is odd.
    let (x8, x_rest) = xq.as_chunks::<8>();
    let x_odd = x_rest.first_chunk::<4>();
    let wblocks = weights[..out.len() * bstride].chunks_exact(bstride);
    for ((o, s), w) in out.iter_mut().zip(scale).zip(wblocks) {
        let mut acc = _mm512_setzero_si512();
        let (w64, w_rest) = w.as_chunks::<64>();
        for (xg, w) in x8.iter().zip(w64) {
            let vx = load_x8(xg);
            acc = dot_avx512vnni(acc, load_i8x64(w), vx);
        }
        let mut acc = fold512(acc);
        if let (Some(xg), Some(w)) = (x_odd, w_rest.first_chunk::<32>()) {
            let vx = load_x4(xg);
            acc = dot_avx2(acc, load_i8x32(w), vx);
        }
        store8(o, s, acc);
    }
}

/// AVX-512 VNNI `sparse_cgemv8x4`. Same contract as
/// `vec_portable::sparse_cgemv8x4`.
///
/// Panics when a slice is shorter than the layer, when `rows` is not a
/// multiple of 8, when `cols > MAX_INPUTS`, or when the index table or the
/// weights fail `nnet::check_sparse` for `rows` rows and `cols` inputs.
///
/// # Safety
///
/// The processor must support AVX2, FMA, AVX-512 F, AVX-512 BW and
/// AVX-512 VNNI.
#[target_feature(enable = "avx2,fma,avx512f,avx512bw,avx512vnni")]
pub fn sparse_cgemv8x4_avx512vnni(
    out: &mut [f32],
    weights: &[i8],
    idx: &[i32],
    scale: &[f32],
    rows: usize,
    cols: usize,
    x: &[f32],
) {
    // `quantize` writes `cols` bytes into a buffer of `MAX_INPUTS`;
    // each block of 8 rows stores 8 values. `check_sparse` checks the
    // index table; the debug checks below repeat that per entry.
    assert!(
        rows.is_multiple_of(8)
            && cols <= MAX_INPUTS
            && x.len() >= cols
            && out.len() >= rows
            && scale.len() >= rows,
        "sparse_cgemv8x4: slices shorter than the layer"
    );
    check_sparse(idx, rows, cols, weights.len());
    let mut xq = XBuf([MaybeUninit::uninit(); MAX_INPUTS]);
    let xq = quantize(&mut xq.0[..cols], &x[..cols]);
    let out = out[..rows].as_chunks_mut::<8>().0;
    let scale = scale[..rows].as_chunks::<8>().0;
    // The entries of the table in order, and the offset of the weights of
    // the next block of 8 rows.
    let mut table = idx.iter();
    let mut wo = 0usize;
    for (o, s) in out.iter_mut().zip(scale) {
        let mut acc = _mm512_setzero_si512();
        let count = *table.next().unwrap() as usize;
        debug_assert!(wo + 32 * count <= weights.len());
        // SAFETY: `check_sparse` checked that the weights hold a block of 32
        // bytes for each listed position.
        let wrow = unsafe { weights.get_unchecked(wo..wo + 32 * count) };
        wo += 32 * count;
        // Two positions at a time, then the last one when `count` is odd.
        let (pairs, odd) = wrow.as_chunks::<64>();
        for w in pairs {
            debug_assert!(table.as_slice().len() >= 2);
            // SAFETY: `check_sparse` checked that `count` positions follow
            // the count of this block.
            let (pos0, pos1) = unsafe {
                (*table.next().unwrap_unchecked() as usize, *table.next().unwrap_unchecked() as usize)
            };
            debug_assert!(pos0 + 4 <= cols && pos1 + 4 <= cols);
            // SAFETY: `check_sparse` checked that each listed position has
            // `pos + 4 <= cols` (`xq` holds `cols` values).
            let (x0, x1) = unsafe { (array_at_unchecked::<i8, 4>(xq, pos0), array_at_unchecked::<i8, 4>(xq, pos1)) };
            let lo = _mm256_set1_epi32(x4_bits(x0));
            let hi = _mm256_set1_epi32(x4_bits(x1));
            let vx = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(lo), hi);
            acc = dot_avx512vnni(acc, load_i8x64(w), vx);
        }
        let mut acc = fold512(acc);
        if let Some(w) = odd.first_chunk::<32>() {
            // Checked: at most once per block of 8 rows.
            let pos = *table.next().unwrap() as usize;
            debug_assert!(pos + 4 <= cols);
            // SAFETY: as above.
            let xg = unsafe { array_at_unchecked::<i8, 4>(xq, pos) };
            let vx = load_x4(xg);
            acc = dot_avx2(acc, load_i8x32(w), vx);
        }
        store8(o, s, acc);
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
#[target_feature(enable = "avx2,fma")]
#[inline]
fn frecpe8(d: __m256) -> __m256 {
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
        // The bits of each lane, as `i32` (`frecpe_bits` takes them as
        // `u32`).
        let mut v = [0i32; 8];
        store_i32x8(&mut v, bits);
        for lane in v.iter_mut() {
            *lane = vec_portable::frecpe_bits(*lane as u32) as i32;
        }
        return _mm256_castsi256_ps(load_i32x8(&v));
    }
    _mm256_castsi256_ps(_mm256_blendv_epi8(normal, special, is_max))
}

/// `exp4_approx` on 8 lanes.
#[target_feature(enable = "avx2,fma")]
#[inline]
#[allow(clippy::excessive_precision, clippy::approx_constant)]
fn exp8(x: __m256) -> __m256 {
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
#[target_feature(enable = "avx2,fma")]
#[inline]
#[allow(clippy::excessive_precision)]
fn tanh8(x: __m256) -> __m256 {
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
    let num = _mm256_mul_ps(num, frecpe8(den));
    _mm256_max_ps(_mm256_set1_ps(-1.0), _mm256_min_ps(_mm256_set1_ps(1.0), num))
}

/// `sigmoid4_approx` on 8 lanes.
#[target_feature(enable = "avx2,fma")]
#[inline]
#[allow(clippy::excessive_precision)]
fn sigmoid8(x: __m256) -> __m256 {
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
    let num = _mm256_fmadd_ps(num, frecpe8(den), _mm256_set1_ps(0.5));
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
pub fn vec_exp_avx2(buf: &mut [f32]) {
    let n8 = buf.len() & !7;
    for v in buf[..n8].as_chunks_mut::<8>().0 {
        store_f32x8(v, exp8(load_f32x8(v)));
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
pub fn vec_tanh_avx2(buf: &mut [f32]) {
    let groups = buf.len() & !3;
    let n8 = buf.len() & !7;
    for v in buf[..n8].as_chunks_mut::<8>().0 {
        store_f32x8(v, tanh8(load_f32x8(v)));
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
pub fn vec_sigmoid_avx2(buf: &mut [f32]) {
    let groups = buf.len() & !3;
    let n8 = buf.len() & !7;
    for v in buf[..n8].as_chunks_mut::<8>().0 {
        store_f32x8(v, sigmoid8(load_f32x8(v)));
    }
    for v in &mut buf[n8..groups] {
        *v = vec_portable::sigmoid_approx(*v);
    }
    for v in &mut buf[groups..] {
        *v = vec_portable::sigmoid_tail(*v);
    }
}
