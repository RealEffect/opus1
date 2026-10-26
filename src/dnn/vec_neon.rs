//! The kernels for 64-bit ARM: NEON, and NEON with the DotProd extension
//! (libopus `dnn/vec_neon.h`, `dnn/arm/nnet_dotprod.c`).
//!
//! Each kernel gives the bits of the function of the same name in
//! `vec_portable`. `vfmaq_f32` stands wherever libopus writes `vmlaq_f32`:
//! libopus redefines `vmlaq_f32` as the fused `vfmaq_f32`, while the Rust
//! `vmlaq_f32` rounds the product first.
//!
//! How they are faster than libopus without changing a bit (`nnet.md`
//! section 10):
//!
//! - Float products: vector lanes stand for output rows, and each lane runs
//!   its own chain of fused multiply-adds in the order of the inputs, as
//!   every libopus variant does. A pass takes up to 32 rows (8 vectors), so
//!   8 independent chains hide the latency of `fmla`. libopus computes an
//!   output count that is not a multiple of 8 one row at a time.
//! - int8 products: integer sums do not depend on their order. Four column
//!   groups share one 16-byte load of the inputs (`sdot` by element) and
//!   run on 8 accumulators. The weights are read as one stream, as in
//!   libopus.
//! - Activations: 4 vectors per loop step.
//!
//! The kernels are safe functions compiled with the target features they
//! use. They reach the slices through arrays (`as_chunks`,
//! `chunks_exact`), loaded and stored with the functions of
//! `crate::simd_mem`. Each kernel checks the lengths it needs and panics
//! when a slice is too short. The kernels of sparse layers also check the
//! index table (`nnet::check_sparse`); after that check they read the
//! table, the weights and the inputs at the listed positions without index
//! checks.

use core::arch::aarch64::*;
use core::mem::MaybeUninit;
use core::iter::Zip;
use core::slice::{self, ChunksExact};

use super::MAX_INPUTS;
use super::nnet::check_sparse;
use super::vec_portable;
use crate::simd_mem::array_at_unchecked;
use crate::simd_mem::neon::{
    load_dup_f32x4, load_f32x4, load_i8x16, load_i8x16x4, store_f32x4, store_i8x4_uninit,
    store_i8x16_uninit,
};

// ---------------------------------------------------------------------------
// Float products
// ---------------------------------------------------------------------------

/// The columns of an `sgemv` product before the last one, each with its
/// input: `(x[j], column j)`. Column `j` holds the weight of row `r` at
/// index `r` and has `stride` values. `sgemv` builds this iterator once
/// and gives each pass over the rows a copy.
type Columns<'a> = Zip<slice::Iter<'a, f32>, ChunksExact<'a, f32>>;

/// The products of one column for `V` vectors of 4 rows:
/// `acc[v] += w[v] · xj`, one fused multiply-add per lane.
#[target_feature(enable = "neon")]
#[inline]
fn sgemv_column<const V: usize>(acc: &mut [float32x4_t; V], w: &[[f32; 4]; V], xj: &f32) {
    let xj = load_dup_f32x4(xj);
    for (a, w) in acc.iter_mut().zip(w) {
        *a = vfmaq_f32(*a, load_f32x4(w), xj);
    }
}

/// One pass of `sgemv` over `V` vectors of 4 rows: `out` gets the rows
/// `i..i + 4·V`.
///
/// `columns` yields the columns before the last one with their inputs.
/// `last` holds the weights of these rows in the last column, and `x_last`
/// is its input.
///
/// Panics when `stride < i + 4·V`.
#[target_feature(enable = "neon")]
#[inline]
fn sgemv_pass<const V: usize>(
    out: &mut [[f32; 4]; V],
    i: usize,
    columns: Columns<'_>,
    stride: usize,
    last: &[[f32; 4]; V],
    x_last: &f32,
) {
    // Each column of `columns` holds `stride` values, so this check covers
    // the reads of all of them.
    assert!(i <= stride && 4 * V <= stride - i);
    let mut acc = [vdupq_n_f32(0.0); V];
    for (xj, col) in columns {
        let w = col[i..][..4 * V].as_chunks::<4>().0.try_into().unwrap();
        sgemv_column(&mut acc, w, xj);
    }
    sgemv_column(&mut acc, last, x_last);
    for (o, a) in out.iter_mut().zip(&acc) {
        store_f32x4(o, *a);
    }
}

/// NEON `sgemv`. Same contract as `vec_portable::sgemv`.
///
/// Panics when a slice is shorter than `vec_portable::sgemv` requires, and
/// when `col_stride < rows` in a product with rows and columns.
///
/// # Safety
///
/// The processor must support NEON, as every 64-bit ARM processor does.
#[target_feature(enable = "neon")]
pub fn sgemv(
    out: &mut [f32],
    weights: &[f32],
    rows: usize,
    cols: usize,
    col_stride: usize,
    x: &[f32],
) {
    // The reads of row `r` and column `j` are at `w[j * col_stride + r]`,
    // so they end at `(cols - 1) * col_stride + rows`. Checked without
    // wrapping around.
    assert!(
        out.len() >= rows
            && x.len() >= cols
            && (rows == 0
                || cols == 0
                || col_stride >= rows
                    && (cols - 1)
                        .checked_mul(col_stride)
                        .and_then(|n| n.checked_add(rows))
                        .is_some_and(|n| weights.len() >= n)),
        "sgemv: slices shorter than the layer"
    );
    if rows == 0 {
        return;
    }
    let out = &mut out[..rows];
    let Some((x_last, x_body)) = x[..cols].split_last() else {
        // No products: every chain stays at +0.
        out.fill(0.0);
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
        let o = o.as_chunks_mut::<4>().0.try_into().unwrap();
        let l = l.as_chunks::<4>().0.try_into().unwrap();
        sgemv_pass::<8>(o, 32 * k, columns.clone(), col_stride, l, x_last);
    }
    let i = 32 * o32.len();
    let (o4, o_tail) = o_rest.as_chunks_mut::<4>();
    let (l4, l_tail) = l_rest.as_chunks::<4>();
    let c = columns.clone();
    match o4.len() {
        7 => sgemv_pass::<7>(o4.try_into().unwrap(), i, c, col_stride, l4.try_into().unwrap(), x_last),
        6 => sgemv_pass::<6>(o4.try_into().unwrap(), i, c, col_stride, l4.try_into().unwrap(), x_last),
        5 => sgemv_pass::<5>(o4.try_into().unwrap(), i, c, col_stride, l4.try_into().unwrap(), x_last),
        4 => sgemv_pass::<4>(o4.try_into().unwrap(), i, c, col_stride, l4.try_into().unwrap(), x_last),
        3 => sgemv_pass::<3>(o4.try_into().unwrap(), i, c, col_stride, l4.try_into().unwrap(), x_last),
        2 => sgemv_pass::<2>(o4.try_into().unwrap(), i, c, col_stride, l4.try_into().unwrap(), x_last),
        1 => sgemv_pass::<1>(o4.try_into().unwrap(), i, c, col_stride, l4.try_into().unwrap(), x_last),
        _ => {}
    }
    // The last 1 to 3 rows: one scalar chain each, side by side.
    let i = i + 4 * o4.len();
    let c = columns;
    match o_tail.len() {
        3 => sgemv_rows::<3>(o_tail.try_into().unwrap(), i, c, col_stride, l_tail.try_into().unwrap(), x_last),
        2 => sgemv_rows::<2>(o_tail.try_into().unwrap(), i, c, col_stride, l_tail.try_into().unwrap(), x_last),
        1 => sgemv_rows::<1>(o_tail.try_into().unwrap(), i, c, col_stride, l_tail.try_into().unwrap(), x_last),
        _ => {}
    }
}

/// `R` rows of `sgemv` (1 to 3) as scalar chains, side by side: `out`
/// gets the rows `i..i + R`. The chain of a row is fixed, so a layer with
/// one output runs at the latency of `fmadd`, as in libopus. The other
/// arguments are those of `sgemv_pass`.
///
/// Panics when `stride < i + R`.
#[inline(always)]
fn sgemv_rows<const R: usize>(
    out: &mut [f32; R],
    i: usize,
    columns: Columns<'_>,
    stride: usize,
    last: &[f32; R],
    x_last: &f32,
) {
    // Each column of `columns` holds `stride` values, so this check covers
    // the reads of all of them.
    assert!(i <= stride && R <= stride - i);
    let mut acc = [0.0f32; R];
    // `for_each` rather than a `for` loop: then the compiler counts the
    // columns down, as in the loop over pointers this replaces, and does
    // not compare a counter in each step.
    columns.for_each(|(&xj, col)| {
        sgemv_rows_column(&mut acc, col[i..].first_chunk().unwrap(), xj);
    });
    sgemv_rows_column(&mut acc, last, *x_last);
    *out = acc;
}

/// `acc[k] = fma(w[k], xj, acc[k])` for each of the `R` rows.
#[inline(always)]
fn sgemv_rows_column<const R: usize>(acc: &mut [f32; R], w: &[f32; R], xj: f32) {
    for (a, &w) in acc.iter_mut().zip(w) {
        *a = vec_portable::fma(w, xj, *a);
    }
}

/// NEON `sparse_sgemv8x4`. Same contract as `vec_portable::sparse_sgemv8x4`.
///
/// Panics when `out` holds fewer than `rows` values, or when the index
/// table or the weights fail `nnet::check_sparse` for `rows` rows and
/// inputs of the length of `x`.
///
/// # Safety
///
/// The processor must support NEON, as every 64-bit ARM processor does.
#[target_feature(enable = "neon")]
pub fn sparse_sgemv8x4(out: &mut [f32], weights: &[f32], idx: &[i32], rows: usize, x: &[f32]) {
    check_sparse(idx, rows, x.len(), weights.len());
    // The entries of the table in order, and the offset of the next block
    // of weights.
    let mut table = idx.iter();
    let mut wo = 0usize;
    for o in out[..rows].as_chunks_mut::<8>().0 {
        let mut y0 = vdupq_n_f32(0.0);
        let mut y1 = vdupq_n_f32(0.0);
        // SAFETY: `check_sparse` checked that the table holds an entry for
        // each block of 8 rows: a count, then that many positions.
        let count = unsafe { *table.next().unwrap_unchecked() } as usize;
        for _ in 0..count {
            // SAFETY: `check_sparse` checked the table: `count` positions
            // follow the count, each with `pos + 4 <= x.len()`, and the
            // weights hold a block of 32 values for each listed position,
            // so the block at `wo` is inside them.
            let (xa, w) = unsafe {
                let pos = *table.next().unwrap_unchecked() as usize;
                (array_at_unchecked::<f32, 4>(x, pos), array_at_unchecked::<f32, 32>(weights, wo))
            };
            let xv = load_f32x4(xa);
            let (w, _) = w.as_chunks::<4>();
            y0 = vfmaq_laneq_f32::<0>(y0, load_f32x4(&w[0]), xv);
            y1 = vfmaq_laneq_f32::<0>(y1, load_f32x4(&w[1]), xv);
            y0 = vfmaq_laneq_f32::<1>(y0, load_f32x4(&w[2]), xv);
            y1 = vfmaq_laneq_f32::<1>(y1, load_f32x4(&w[3]), xv);
            y0 = vfmaq_laneq_f32::<2>(y0, load_f32x4(&w[4]), xv);
            y1 = vfmaq_laneq_f32::<2>(y1, load_f32x4(&w[5]), xv);
            y0 = vfmaq_laneq_f32::<3>(y0, load_f32x4(&w[6]), xv);
            y1 = vfmaq_laneq_f32::<3>(y1, load_f32x4(&w[7]), xv);
            wo += 32;
        }
        let (o, _) = o.as_chunks_mut::<4>();
        store_f32x4(&mut o[0], y0);
        store_f32x4(&mut o[1], y1);
    }
}

// ---------------------------------------------------------------------------
// int8 products
// ---------------------------------------------------------------------------

/// The quantized input of an int8 product, aligned for 4-byte loads.
#[repr(C, align(16))]
struct XBuf([MaybeUninit<i8>; MAX_INPUTS]);

/// `xq[j] = quantize(x[j])` for every `j` (`dnn/vec_neon.h:385-392`):
/// `vcvtnq_s32_f32(127·x)`, then the low 16 and the low 8 bits. Returns
/// `xq`, which then holds values.
///
/// Panics when `xq` and `x` differ in length.
#[target_feature(enable = "neon")]
#[inline]
fn quantize<'a>(xq: &'a mut [MaybeUninit<i8>], x: &[f32]) -> &'a [i8] {
    assert_eq!(xq.len(), x.len());
    let c127 = vdupq_n_f32(127.0);
    let (q16, q_rest) = xq.as_chunks_mut::<16>();
    let (x16, x_rest) = x.as_chunks::<16>();
    for (q, x) in q16.iter_mut().zip(x16) {
        let (x, _) = x.as_chunks::<4>();
        let a = vcvtnq_s32_f32(vmulq_f32(c127, load_f32x4(&x[0])));
        let b = vcvtnq_s32_f32(vmulq_f32(c127, load_f32x4(&x[1])));
        let c = vcvtnq_s32_f32(vmulq_f32(c127, load_f32x4(&x[2])));
        let d = vcvtnq_s32_f32(vmulq_f32(c127, load_f32x4(&x[3])));
        let ab = vmovn_high_s32(vmovn_s32(a), b);
        let cd = vmovn_high_s32(vmovn_s32(c), d);
        store_i8x16_uninit(q, vmovn_high_s16(vmovn_s16(ab), cd));
    }
    let (q4, q_tail) = q_rest.as_chunks_mut::<4>();
    let (x4, x_tail) = x_rest.as_chunks::<4>();
    for (q, x) in q4.iter_mut().zip(x4) {
        let a = vcvtnq_s32_f32(vmulq_f32(c127, load_f32x4(x)));
        let a8 = vmovn_s16(vcombine_s16(vmovn_s32(a), vdup_n_s16(0)));
        store_i8x4_uninit(q, a8);
    }
    for (q, &v) in q_tail.iter_mut().zip(x_tail) {
        q.write(vec_portable::quantize(v));
    }
    // SAFETY: `xq` and `x` have the same length, so the three loops above
    // wrote every value of `xq`: the groups of 16, the groups of 4 of the
    // rest, and the last values one by one.
    unsafe { xq.assume_init_ref() }
}

/// The 4 quantized inputs `q`, repeated in each 32-bit lane.
#[target_feature(enable = "neon")]
#[inline]
fn load_x4(q: &[i8; 4]) -> int8x16_t {
    vreinterpretq_s8_s32(vdupq_n_s32(i32::from_ne_bytes(q.map(|b| b as u8))))
}

/// `out[k] = scale[k] · acc[k]` for the 8 rows of one block.
#[target_feature(enable = "neon")]
#[inline]
fn store8(out: &mut [f32; 8], scale: &[f32; 8], a0: int32x4_t, a1: int32x4_t) {
    let (o, _) = out.as_chunks_mut::<4>();
    let (s, _) = scale.as_chunks::<4>();
    store_f32x4(&mut o[0], vmulq_f32(load_f32x4(&s[0]), vcvtq_f32_s32(a0)));
    store_f32x4(&mut o[1], vmulq_f32(load_f32x4(&s[1]), vcvtq_f32_s32(a1)));
}

/// The dot products of 4 rows by 4 columns with the `sdot` instruction.
#[target_feature(enable = "neon,dotprod")]
#[inline]
fn dot_sdot(acc: int32x4_t, w: int8x16_t, x: int8x16_t) -> int32x4_t {
    vdotq_s32(acc, w, x)
}

/// The same dot products without DotProd (`dnn/vec_neon.h:373-376`):
/// products in 16 bits, pairs added in 16 bits, then added to the 32-bit
/// lanes. A pair cannot overflow while the weights lie in [-127, 127].
#[target_feature(enable = "neon")]
#[inline]
fn dot_neon(acc: int32x4_t, w: int8x16_t, x: int8x16_t) -> int32x4_t {
    vpadalq_s16(acc, vpaddq_s16(vmull_s8(vget_low_s8(w), vget_low_s8(x)), vmull_high_s8(w, x)))
}

/// The dot products of 4 rows by 4 columns with `sdot`, the inputs taken
/// from 32-bit lane `L` of `x` (`sdot` by element).
#[target_feature(enable = "neon,dotprod")]
#[inline]
fn dot_lane_sdot<const L: i32>(acc: int32x4_t, w: int8x16_t, x: int8x16_t) -> int32x4_t {
    vdotq_laneq_s32::<L>(acc, w, x)
}

/// `dot_neon` with the inputs taken from 32-bit lane `L` of `x`.
#[target_feature(enable = "neon")]
#[inline]
fn dot_lane_neon<const L: i32>(acc: int32x4_t, w: int8x16_t, x: int8x16_t) -> int32x4_t {
    dot_neon(acc, w, vreinterpretq_s8_s32(vdupq_laneq_s32::<L>(vreinterpretq_s32_s8(x))))
}

/// Defines a dense and a sparse int8 product on one dot product function
/// (`$dot`, inputs repeated in every lane) and its by-lane form (`$dotl`).
macro_rules! int8_kernels {
    ($features:literal, $dot:ident, $dotl:ident, $dense:ident, $sparse:ident, $what:literal) => {
        #[doc = concat!("NEON `cgemv8x4` ", $what, ". Same contract as `vec_portable::cgemv8x4`.")]
        ///
        /// One block of 8 rows at a time, so the weights are read as one
        /// sequential stream, as in libopus: with several blocks at a time
        /// (several streams), layers whose weights are not in the caches
        /// ran up to 1.6 times slower on the Cortex-A55 and 1.4 times on
        /// the Cortex-A76. Four column groups share one 16-byte load of
        /// the inputs and run on separate accumulators.
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
            debug_assert!(cols <= MAX_INPUTS && rows % 8 == 0 && cols % 4 == 0);
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
                    store8(o, s, vdupq_n_s32(0), vdupq_n_s32(0));
                }
                return;
            }
            // The inputs of 4 column groups at a time, then those of the
            // last 0 to 3 groups.
            let (x16, x_rest) = xq.as_chunks::<16>();
            let x4 = x_rest.as_chunks::<4>().0;
            // The weights of a block of 8 rows: one 8 × 4 block (32 bytes)
            // for each column group.
            let wlen = 32 * groups;
            let wblocks = weights[..out.len() * wlen].chunks_exact(wlen);
            for ((o, s), w) in out.iter_mut().zip(scale).zip(wblocks) {
                let mut acc = [vdupq_n_s32(0); 8];
                let (w128, w_rest) = w.as_chunks::<128>();
                for (xv, w) in x16.iter().zip(w128) {
                    let xv = load_i8x16(xv);
                    let (w, _) = w.as_chunks::<64>();
                    let w01 = load_i8x16x4(&w[0]);
                    let w23 = load_i8x16x4(&w[1]);
                    acc[0] = $dotl::<0>(acc[0], w01.0, xv);
                    acc[1] = $dotl::<0>(acc[1], w01.1, xv);
                    acc[2] = $dotl::<1>(acc[2], w01.2, xv);
                    acc[3] = $dotl::<1>(acc[3], w01.3, xv);
                    acc[4] = $dotl::<2>(acc[4], w23.0, xv);
                    acc[5] = $dotl::<2>(acc[5], w23.1, xv);
                    acc[6] = $dotl::<3>(acc[6], w23.2, xv);
                    acc[7] = $dotl::<3>(acc[7], w23.3, xv);
                }
                for (xg, w) in x4.iter().zip(w_rest.as_chunks::<32>().0) {
                    let vx = load_x4(xg);
                    let (w, _) = w.as_chunks::<16>();
                    acc[0] = $dot(acc[0], load_i8x16(&w[0]), vx);
                    acc[1] = $dot(acc[1], load_i8x16(&w[1]), vx);
                }
                let a0 = vaddq_s32(vaddq_s32(acc[0], acc[2]), vaddq_s32(acc[4], acc[6]));
                let a1 = vaddq_s32(vaddq_s32(acc[1], acc[3]), vaddq_s32(acc[5], acc[7]));
                store8(o, s, a0, a1);
            }
        }

        #[doc = concat!("NEON `sparse_cgemv8x4` ", $what, ". Same contract as `vec_portable::sparse_cgemv8x4`.")]
        ///
        /// Panics when a slice is shorter than the layer, when
        /// `cols > MAX_INPUTS`, or when the index table or the weights
        /// fail `nnet::check_sparse` for `rows` rows and `cols` inputs.
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
            debug_assert!(cols <= MAX_INPUTS && rows % 8 == 0);
            // `quantize` writes `cols` bytes into a buffer of `MAX_INPUTS`;
            // each block of 8 rows stores 8 values.
            assert!(
                cols <= MAX_INPUTS && x.len() >= cols && out.len() >= rows && scale.len() >= rows,
                "sparse_cgemv8x4: slices shorter than the layer"
            );
            check_sparse(idx, rows, cols, weights.len());
            let mut xq = XBuf([MaybeUninit::uninit(); MAX_INPUTS]);
            let xq = quantize(&mut xq.0[..cols], &x[..cols]);
            let out = out[..rows].as_chunks_mut::<8>().0;
            let scale = scale[..rows].as_chunks::<8>().0;
            // The entries of the table in order, and the offset of the
            // weights of the next block of 8 rows.
            let mut table = idx.iter();
            let mut wo = 0usize;
            for (o, s) in out.iter_mut().zip(scale) {
                let mut acc = [vdupq_n_s32(0); 4];
                // SAFETY: `check_sparse` checked that the table holds an
                // entry for each block of 8 rows (a count, then that many
                // positions), and that the weights hold a block of 32 bytes
                // for each listed position.
                let (count, wrow) = unsafe {
                    let count = *table.next().unwrap_unchecked() as usize;
                    (count, weights.get_unchecked(wo..wo + 32 * count))
                };
                wo += 32 * count;
                // Two positions at a time, then the last one when `count` is
                // odd.
                let (pairs, odd) = wrow.as_chunks::<64>();
                for w in pairs {
                    // SAFETY: `check_sparse` checked that `count` positions
                    // follow the count, each with `pos + 4 <= cols` (`xq`
                    // holds `cols` values).
                    let (x0, x1) = unsafe {
                        let pos0 = *table.next().unwrap_unchecked() as usize;
                        let pos1 = *table.next().unwrap_unchecked() as usize;
                        (array_at_unchecked::<i8, 4>(xq, pos0), array_at_unchecked::<i8, 4>(xq, pos1))
                    };
                    let vx0 = load_x4(x0);
                    let vx1 = load_x4(x1);
                    let (w, _) = w.as_chunks::<16>();
                    acc[0] = $dot(acc[0], load_i8x16(&w[0]), vx0);
                    acc[1] = $dot(acc[1], load_i8x16(&w[1]), vx0);
                    acc[2] = $dot(acc[2], load_i8x16(&w[2]), vx1);
                    acc[3] = $dot(acc[3], load_i8x16(&w[3]), vx1);
                }
                if let Some(w) = odd.first_chunk::<32>() {
                    // SAFETY: as above, for the last position of the block.
                    let xg = unsafe { array_at_unchecked::<i8, 4>(xq, *table.next().unwrap_unchecked() as usize) };
                    let vx = load_x4(xg);
                    let (w, _) = w.as_chunks::<16>();
                    acc[0] = $dot(acc[0], load_i8x16(&w[0]), vx);
                    acc[1] = $dot(acc[1], load_i8x16(&w[1]), vx);
                }
                let a0 = vaddq_s32(acc[0], acc[2]);
                let a1 = vaddq_s32(acc[1], acc[3]);
                store8(o, s, a0, a1);
            }
        }
    };
}

int8_kernels!(
    "neon",
    dot_neon,
    dot_lane_neon,
    cgemv8x4,
    sparse_cgemv8x4,
    "without DotProd: the dot products with `vmull_s8`, `vpaddq_s16` and `vpadalq_s16`, as libopus on processors without DotProd"
);
int8_kernels!(
    "neon,dotprod",
    dot_sdot,
    dot_lane_sdot,
    cgemv8x4_dotprod,
    sparse_cgemv8x4_dotprod,
    "with the `sdot` instruction"
);

// ---------------------------------------------------------------------------
// Activations
// ---------------------------------------------------------------------------

/// `exp4_approx` (`dnn/vec_neon.h:59-84`).
#[target_feature(enable = "neon")]
#[inline]
#[allow(clippy::excessive_precision, clippy::approx_constant)]
fn exp4(x: float32x4_t) -> float32x4_t {
    let x = vmaxq_f32(vminq_f32(x, vdupq_n_f32(88.0)), vdupq_n_f32(-88.0));
    let x = vfmaq_f32(vdupq_n_f32(127.0), x, vdupq_n_f32(1.44269504));
    let i = vcvtq_s32_f32(x);
    let x = vsubq_f32(x, vcvtq_f32_s32(i));
    let k0 = vdupq_n_f32(0.99992522);
    let k1 = vdupq_n_f32(0.69583354);
    let k2 = vdupq_n_f32(0.22606716);
    let k3 = vdupq_n_f32(0.078024523);
    let y = vfmaq_f32(k0, x, vfmaq_f32(k1, x, vfmaq_f32(k2, k3, x)));
    vmulq_f32(y, vreinterpretq_f32_s32(vshlq_n_s32::<23>(i)))
}

/// `tanh4_approx` (`dnn/vec_neon.h:86-104`).
#[target_feature(enable = "neon")]
#[inline]
#[allow(clippy::excessive_precision)]
fn tanh4(x: float32x4_t) -> float32x4_t {
    let x2 = vmulq_f32(x, x);
    let num = vfmaq_f32(
        vdupq_n_f32(952.52801514),
        x2,
        vfmaq_f32(vdupq_n_f32(96.39235687), vdupq_n_f32(0.60863042), x2),
    );
    let den = vfmaq_f32(
        vdupq_n_f32(952.72399902),
        x2,
        vfmaq_f32(vdupq_n_f32(413.36801147), vdupq_n_f32(11.88600922), x2),
    );
    let num = vmulq_f32(num, x);
    let num = vmulq_f32(num, vrecpeq_f32(den));
    vmaxq_f32(vdupq_n_f32(-1.0), vminq_f32(vdupq_n_f32(1.0), num))
}

/// `sigmoid4_approx` (`dnn/vec_neon.h:106-125`).
#[target_feature(enable = "neon")]
#[inline]
#[allow(clippy::excessive_precision)]
fn sigmoid4(x: float32x4_t) -> float32x4_t {
    let x2 = vmulq_f32(x, x);
    let num = vfmaq_f32(
        vdupq_n_f32(238.13200378),
        x2,
        vfmaq_f32(vdupq_n_f32(6.02452230), vdupq_n_f32(0.00950985), x2),
    );
    let den = vfmaq_f32(
        vdupq_n_f32(952.72399902),
        x2,
        vfmaq_f32(vdupq_n_f32(103.34200287), vdupq_n_f32(0.74287558), x2),
    );
    let num = vmulq_f32(num, x);
    let num = vfmaq_f32(vdupq_n_f32(0.5), num, vrecpeq_f32(den));
    vmaxq_f32(vdupq_n_f32(0.0), vminq_f32(vdupq_n_f32(1.0), num))
}

/// Applies the vector function `$f` to the first `4·floor(len/4)` values
/// of `$buf`, 4 vectors per loop step (16 values), so that an in-order
/// core such as the Cortex-A55 can overlap their chains of dependent
/// instructions. Evaluates to the number of values done.
macro_rules! map_groups {
    ($buf:expr, $f:ident) => {{
        let buf: &mut [f32] = $buf;
        let groups = buf.len() & !3;
        let (b16, b4) = buf[..groups].as_chunks_mut::<16>();
        for v in b16 {
            let (v, _) = v.as_chunks_mut::<4>();
            let a = $f(load_f32x4(&v[0]));
            let b = $f(load_f32x4(&v[1]));
            let c = $f(load_f32x4(&v[2]));
            let d = $f(load_f32x4(&v[3]));
            store_f32x4(&mut v[0], a);
            store_f32x4(&mut v[1], b);
            store_f32x4(&mut v[2], c);
            store_f32x4(&mut v[3], d);
        }
        for v in b4.as_chunks_mut::<4>().0 {
            store_f32x4(v, $f(load_f32x4(v)));
        }
        groups
    }};
}

/// NEON `vec_exp`. Same contract as `vec_portable::vec_exp`.
///
/// # Safety
///
/// The processor must support NEON, as every 64-bit ARM processor does.
#[target_feature(enable = "neon")]
pub fn vec_exp(buf: &mut [f32]) {
    let groups = map_groups!(buf, exp4);
    for v in &mut buf[groups..] {
        *v = vec_portable::lpcnet_exp(*v);
    }
}

/// NEON `vec_tanh` (with `vrecpeq_f32`). Same contract as
/// `vec_portable::vec_tanh`.
///
/// # Safety
///
/// The processor must support NEON, as every 64-bit ARM processor does.
#[target_feature(enable = "neon")]
pub fn vec_tanh(buf: &mut [f32]) {
    let groups = map_groups!(buf, tanh4);
    for v in &mut buf[groups..] {
        *v = vec_portable::tanh_tail(*v);
    }
}

/// NEON `vec_sigmoid` (with `vrecpeq_f32`). Same contract as
/// `vec_portable::vec_sigmoid`.
///
/// # Safety
///
/// The processor must support NEON, as every 64-bit ARM processor does.
#[target_feature(enable = "neon")]
pub fn vec_sigmoid(buf: &mut [f32]) {
    let groups = map_groups!(buf, sigmoid4);
    for v in &mut buf[groups..] {
        *v = vec_portable::sigmoid_tail(*v);
    }
}
