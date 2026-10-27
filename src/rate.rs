use crate::modes::CeltMode;
use crate::range_coder::RangeCoder;
use core::cmp::{max, min};
use core::hint::select_unpredictable;

const MAX_EBANDS: usize = 21;
pub const BITRES: i32 = 3;
pub const FINE_OFFSET: i32 = 21;
pub const MAX_FINE_BITS: i32 = 8;

pub const LOG2_FRAC_TABLE: [u8; 24] = [
    0, 8, 13, 16, 19, 21, 23, 24, 26, 27, 28, 29, 30, 31, 32, 32, 33, 34, 34, 35, 36, 36, 37, 37,
];

#[inline(always)]
pub fn get_pulses(i: i32) -> i32 {
    if i < 8 {
        i
    } else {
        let shift = (i >> 3) - 1;
        if shift >= 31 {
            return 0x7FFFFFFF;
        }
        (8 + (i & 7)) << shift
    }
}

/// The row of the pulse cache of `band` at frame size index `lm`: -1 to 3,
/// where -1 stands for half a band of a 2.5 ms frame. Its first entry is
/// the largest pseudo-pulse count `K` of the row, and entry `q` (1 to `K`)
/// is the cost of `q` pseudo-pulses in 1/8 bits, less one. `None` when the
/// band cannot hold pulses at that size.
///
/// Indexing is checked: a `band` or `lm` outside the mode's cache panics.
#[inline(always)]
fn pulse_cache_row(m: &CeltMode, band: usize, lm: i32) -> Option<&[u8]> {
    let idx = (lm + 1) as usize * m.nb_ebands + band;
    let cache_index = m.cache.index[idx];
    if cache_index < 0 {
        return None;
    }
    Some(&m.cache.bits[cache_index as usize..])
}

/// libopus `bits2pulses` (rate.h): the pseudo-pulse count whose cost in
/// the pulse cache lies closest to `bits` (1/8 bits) for `band` at frame
/// size index `lm`. Panics when `band` or `lm` lies outside the mode's
/// pulse cache.
///
/// On aarch64 the row is searched with `bits2pulses_count` when 48 bytes
/// from its start lie inside the cache, which holds for most rows. The
/// other rows, and other targets, use `bits2pulses_bisect`, the loop of
/// libopus.
#[inline(always)]
pub fn bits2pulses(m: &CeltMode, band: usize, lm: i32, bits: i32) -> i32 {
    let idx = (lm + 1) as usize * m.nb_ebands + band;
    let cache_index = m.cache.index[idx];
    if cache_index < 0 {
        return 0;
    }
    let start = cache_index as usize;
    #[cfg(target_arch = "aarch64")]
    if let Some(row) = m.cache.bits.get(start..start + 48)
        && let Ok(row) = <&[u8; 48]>::try_from(row)
        && let Some(q) = bits2pulses_count(row, bits)
    {
        return q;
    }
    bits2pulses_bisect(&m.cache.bits[start..], bits)
}

/// The search of libopus `bits2pulses` over one row of the pulse cache:
/// `row[0]` is the largest pseudo-pulse count `K` of the row, and `row[q]`
/// (1 to `K`) the cost of `q` pseudo-pulses in 1/8 bits, less one. Panics
/// when the row holds fewer than `K + 1` values.
#[inline(always)]
fn bits2pulses_bisect(cache: &[u8], bits: i32) -> i32 {
    let mut lo = 0usize;
    let mut hi = cache[0] as usize;
    let bits = bits - 1;
    // LOG_MAX_PSEUDO = 6 halvings of the row, rounding the middle up as C
    // does.
    for _ in 0..6 {
        let mid = (lo + hi + 1) >> 1;
        if cache[mid] as i32 >= bits {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    let lo_val = if lo == 0 { -1 } else { cache[lo] as i32 };
    let hi_val = cache[hi] as i32;
    if bits - lo_val <= hi_val - bits {
        lo as i32
    } else {
        hi as i32
    }
}

/// Lane numbers for `bits2pulses_count`. Lane 0 holds `K`, not a cost, so
/// it gets 255, which is above every `K` the function accepts.
#[cfg(target_arch = "aarch64")]
static PULSE_ROW_LANES: [u8; 48] = {
    let mut lanes = [0u8; 48];
    let mut i = 1;
    while i < 48 {
        lanes[i] = i as u8;
        i += 1;
    }
    lanes[0] = 255;
    lanes
};

/// `bits2pulses_bisect` for a row of which 48 bytes can be read, without a
/// chain of dependent steps. `None` when the row has more than 40 entries.
///
/// The costs in a row of the pulse cache do not decrease (a test checks
/// this for every row of the mode). The bisection then ends with `lo`
/// equal to the number of costs below `bits - 1`, and `hi = lo + 1`, or
/// `hi = lo = K` when all costs are below. Six halvings are enough for
/// `K <= 40`, libopus's `MAX_PSEUDO`. This function counts the costs below
/// `bits - 1` with three vector compares, so the six loads and compares
/// that depend on each other become independent work. On a Cortex-A55
/// each step of the bisection waits for its load.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn bits2pulses_count(row: &[u8; 48], bits: i32) -> Option<i32> {
    use crate::simd_mem::{array_at, neon::load_u8x16};
    use core::arch::aarch64::*;

    let k = row[0];
    if k > 40 {
        return None;
    }
    let b = bits - 1;
    // A cost is a byte. Every cost lies below a `b` above 255, and none
    // below a `b` of 0 or less, so `b` can be clamped to a byte when the
    // case above 255 is handled apart.
    let below_all = b > 255;
    let b8 = b.clamp(0, 255) as u8;
    // The vector code stays in this always-inlined function: a function
    // with `#[target_feature]` cannot be `#[inline(always)]`, and the
    // compiler did not inline one here.
    // SAFETY: NEON is available: src/lib.rs refuses to build for aarch64
    // without it. The block calls only NEON intrinsics and the load helpers of
    // `simd_mem`.
    let count = unsafe {
        let (vb, vk) = (vdupq_n_u8(b8), vdupq_n_u8(k));
        // All bits set in lanes 1 to `K` whose cost is below `b`, among
        // the 16 lanes from `o` on.
        let below = |o: usize| {
            vandq_u8(
                vcltq_u8(load_u8x16(array_at(row, o)), vb),
                vcleq_u8(load_u8x16(array_at(&PULSE_ROW_LANES, o)), vk),
            )
        };
        // Each set lane is -1; subtracting adds one per lane. At most 3
        // per lane, 40 in all, so the bytes do not overflow.
        let sum = vsubq_u8(
            vsubq_u8(vsubq_u8(vdupq_n_u8(0), below(0)), below(16)),
            below(32),
        );
        vaddvq_u8(sum) as usize
    };
    let lo = if below_all { k as usize } else { count };
    let hi = lo + (lo < k as usize) as usize;
    let lo_val = if lo == 0 { -1 } else { row[lo] as i32 };
    let hi_val = row[hi] as i32;
    Some(if b - lo_val <= hi_val - b {
        lo as i32
    } else {
        hi as i32
    })
}

/// libopus `pulses2bits` (rate.h): the cost in 1/8 bits of `pulses`
/// pseudo-pulses in `band` at frame size index `lm`. Panics when `band`,
/// `lm` or `pulses` lies outside the mode's pulse cache.
#[inline(always)]
pub fn pulses2bits(m: &CeltMode, band: usize, lm: i32, pulses: i32) -> i32 {
    if pulses == 0 {
        return 0;
    }
    match pulse_cache_row(m, band, lm) {
        Some(cache) => cache[pulses as usize] as i32 + 1,
        None => 0,
    }
}

/// What one band adds to the bit sum of an allocation search (libopus
/// `clt_compute_allocation` and `interp_bits2pulses`). The search visits
/// the bands from the last one down. A band at or above its threshold
/// `thresh` sets `done`. While `done` is set, a band adds its `bits` up to
/// its `cap`. Before that, a band adds `floor` if it reaches it, else
/// nothing.
///
/// The code has no branch that depends on the data. The searches run this
/// about 200 times per frame, and an in-order core such as the Cortex-A55
/// loses many cycles on each mispredicted branch.
#[inline(always)]
fn searched_bits(bits: i32, thresh: i32, cap: i32, floor: i32, done: &mut bool) -> i32 {
    *done |= bits >= thresh;
    let below = select_unpredictable(bits >= floor, floor, 0);
    select_unpredictable(*done, min(bits, cap), below)
}

/// The bits that one band allocation vector gives the bands of a frame:
/// the loop of the bisection over the vectors in libopus
/// `clt_compute_allocation`. Each slice has one entry per band, from the
/// first band of the frame on.
///
/// - `row`: the vector, in 1/32 bit per coefficient and channel.
/// - `width`: `(C * N) << LM`, where `N` is the width of the band at LM 0.
/// - `floor`: `C << BITRES`.
///
/// This form runs on targets other than aarch64 and for frames of few
/// bands, and is the reference for `SearchLanes::vector_bits` in the tests.
#[inline(always)]
fn vector_bits(
    row: &[u8],
    width: &[i32],
    trim_offset: &[i32],
    offsets: &[i32],
    thresh: &[i32],
    cap: &[i32],
    floor: i32,
) -> i32 {
    // One length for all slices, so the loop has no bounds checks.
    let len = width.len();
    let (row, trim_offset, offsets) = (&row[..len], &trim_offset[..len], &offsets[..len]);
    let (thresh, cap) = (&thresh[..len], &cap[..len]);
    let mut psum = 0;
    let mut done = false;
    for j in (0..len).rev() {
        // C: `(C*N*alloc) << LM >> 2`. The product does not overflow, so the
        // order of the factors and of the shift does not matter.
        let bits = (width[j] * row[j] as i32) >> 2;
        // C: `if (bitsj > 0) bitsj = IMAX(0, bitsj + trim_offset[j])`. The
        // value is not negative, so for 0 the trim is left out instead.
        let trim = select_unpredictable(bits > 0, trim_offset[j], 0);
        let bits = max(0, bits + trim) + offsets[j];
        psum += searched_bits(bits, thresh[j], cap[j], floor, &mut done);
    }
    psum
}

/// The sum of `interp_bits2pulses`'s search for the interpolation step
/// `mid`: band `j` has `bits1[j] + ((mid * bits2[j]) >> 6)` bits. The
/// slices hold the bands of the frame, from the first one on.
///
/// This form runs on targets other than aarch64 and for frames of few
/// bands, and is the reference for `SearchLanes::interp_bits` in the tests.
#[inline(always)]
fn interp_bits(
    bits1: &[i32],
    bits2: &[i32],
    mid: i32,
    thresh: &[i32],
    cap: &[i32],
    floor: i32,
) -> i32 {
    let len = bits1.len();
    let (bits2, thresh, cap) = (&bits2[..len], &thresh[..len], &cap[..len]);
    let mut psum = 0;
    let mut done = false;
    for j in (0..len).rev() {
        let tmp = bits1[j] + ((mid * bits2[j]) >> 6);
        psum += searched_bits(tmp, thresh[j], cap[j], floor, &mut done);
    }
    psum
}

/// Lanes for the bands of a frame, from its first band on: the mode's 21
/// bands rounded up to six vectors of four. The searches of
/// `clt_compute_allocation` and `interp_bits2pulses` run on these with
/// NEON.
#[cfg(target_arch = "aarch64")]
const BAND_LANES: usize = 24;

#[cfg(target_arch = "aarch64")]
const _: () = assert!(MAX_EBANDS <= BAND_LANES);

/// The vector searches pay for building their lanes when the frame has at
/// least this many bands. Hybrid frames code 2 to 4 bands (from band 17
/// on), and there the scalar loops were faster: on an Apple M4 the vector
/// form made Hybrid 24 kHz encoding 0.6 % to 2.2 % slower.
#[cfg(target_arch = "aarch64")]
const NEON_MIN_BANDS: usize = 9;

/// One value per band in lanes (see `BAND_LANES`).
#[cfg(target_arch = "aarch64")]
type BandLanes = [i32; BAND_LANES];

/// `src` in the first lanes and `pad` in the lanes after the frame.
/// Panics when `src` has more than `BAND_LANES` values.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn band_lanes(src: &[i32], pad: i32) -> BandLanes {
    let mut lanes = [pad; BAND_LANES];
    for (lane, &v) in lanes[..src.len()].iter_mut().zip(src) {
        *lane = v;
    }
    lanes
}

/// The values of one search, in lanes. The lanes after the frame hold a
/// threshold no band reaches and a cap of 0, so with 0 bits there they add
/// nothing to a sum.
#[cfg(target_arch = "aarch64")]
struct SearchLanes {
    thresh: BandLanes,
    cap: BandLanes,
}

#[cfg(target_arch = "aarch64")]
impl SearchLanes {
    fn new(thresh: &[i32], cap: &[i32]) -> Self {
        SearchLanes {
            thresh: band_lanes(thresh, i32::MAX),
            cap: band_lanes(cap, 0),
        }
    }

    /// The sum of `searched_bits` over the bands for `bits` in lanes, with
    /// 0 in the lanes after the frame.
    ///
    /// The scalar loop visits the bands from the last one down, and `done`
    /// is set from the first band (from the top) whose bits reach its
    /// threshold. So a band is `done` when its lane is at or below the
    /// highest lane that reaches its threshold. That lane is found first,
    /// then every lane adds its share; no step waits for the band above.
    #[target_feature(enable = "neon")]
    #[inline]
    fn sum(&self, bits: [core::arch::aarch64::int32x4_t; BAND_LANES / 4], floor: i32) -> i32 {
        use crate::simd_mem::{array_at, neon::load_i32x4};
        use core::arch::aarch64::*;
        let lanes = [0i32, 1, 2, 3];
        // The loads below read `[4 * g, 4 * g + 4)` of arrays of
        // `BAND_LANES` values with `g < BAND_LANES / 4`.
        let first = load_i32x4(&lanes);
        let none = vdupq_n_s32(-1);
        let mut top = none;
        for (g, &b) in bits.iter().enumerate() {
            let idx = vaddq_s32(first, vdupq_n_s32(4 * g as i32));
            let reached = vcgeq_s32(b, load_i32x4(array_at(&self.thresh, 4 * g)));
            top = vmaxq_s32(top, vbslq_s32(reached, idx, none));
        }
        let top = vdupq_n_s32(vmaxvq_s32(top));
        let vfloor = vdupq_n_s32(floor);
        let mut acc = vdupq_n_s32(0);
        for (g, &b) in bits.iter().enumerate() {
            let idx = vaddq_s32(first, vdupq_n_s32(4 * g as i32));
            let capped = vminq_s32(b, load_i32x4(array_at(&self.cap, 4 * g)));
            let below = vandq_s32(vreinterpretq_s32_u32(vcgeq_s32(b, vfloor)), vfloor);
            acc = vaddq_s32(acc, vbslq_s32(vcleq_s32(idx, top), capped, below));
        }
        vaddvq_s32(acc)
    }

    /// `vector_bits` for the vector `row` (one byte per band of the frame,
    /// at least as many as the frame has bands), with `width`, `trim` and
    /// `offsets` in lanes with 0 after the frame.
    #[target_feature(enable = "neon")]
    #[inline]
    fn vector_bits(
        &self,
        row: &[u8],
        width: &BandLanes,
        trim: &BandLanes,
        offsets: &BandLanes,
        len: usize,
        floor: i32,
    ) -> i32 {
        use crate::simd_mem::{array_at, neon::{load_i32x4, load_u8x8, load_u8x16}};
        use core::arch::aarch64::*;
        let mut bytes = [0u8; BAND_LANES];
        for (b, &r) in bytes.iter_mut().zip(&row[..len]) {
            *b = r;
        }
        // The loads read bytes 0 to 23 of `bytes`, and `[4 * g, 4 * g + 4)`
        // of arrays of `BAND_LANES` values with `g < BAND_LANES / 4`.
        let b16 = load_u8x16(array_at(&bytes, 0));
        let b8 = load_u8x8(array_at(&bytes, 16));
        let (lo, hi, top) = (vmovl_u8(vget_low_u8(b16)), vmovl_high_u8(b16), vmovl_u8(b8));
        let rows = [
            vmovl_u16(vget_low_u16(lo)),
            vmovl_high_u16(lo),
            vmovl_u16(vget_low_u16(hi)),
            vmovl_high_u16(hi),
            vmovl_u16(vget_low_u16(top)),
            vmovl_high_u16(top),
        ];
        let zero = vdupq_n_s32(0);
        let mut bits = [zero; BAND_LANES / 4];
        for (g, (out, &r)) in bits.iter_mut().zip(&rows).enumerate() {
            let w = load_i32x4(array_at(width, 4 * g));
            // C: `(C*N*alloc) << LM >> 2`, as in `vector_bits`.
            let b = vshrq_n_s32::<2>(vmulq_s32(w, vreinterpretq_s32_u32(r)));
            // C: the trim only for a positive value, then `IMAX(0, .)`.
            let t = vandq_s32(
                vreinterpretq_s32_u32(vcgtq_s32(b, zero)),
                load_i32x4(array_at(trim, 4 * g)),
            );
            let b = vmaxq_s32(vaddq_s32(b, t), zero);
            *out = vaddq_s32(b, load_i32x4(array_at(offsets, 4 * g)));
        }
        self.sum(bits, floor)
    }

    /// `interp_bits` with `bits1` and `bits2` in lanes with 0 after the
    /// frame.
    #[target_feature(enable = "neon")]
    #[inline]
    fn interp_bits(&self, bits1: &BandLanes, bits2: &BandLanes, mid: i32, floor: i32) -> i32 {
        use crate::simd_mem::{array_at, neon::load_i32x4};
        use core::arch::aarch64::*;
        let mut bits = [vdupq_n_s32(0); BAND_LANES / 4];
        for (g, out) in bits.iter_mut().enumerate() {
            // The loads read `[4 * g, 4 * g + 4)` of arrays of
            // `BAND_LANES` values with `g < BAND_LANES / 4`.
            let (b1, b2) = (load_i32x4(array_at(bits1, 4 * g)), load_i32x4(array_at(bits2, 4 * g)));
            *out = vaddq_s32(b1, vshrq_n_s32::<6>(vmulq_n_s32(b2, mid)));
        }
        self.sum(bits, floor)
    }
}

/// The bisection of `clt_compute_allocation` over the band allocation
/// vectors 1 to `last`, with `psum(v)` the bits that vector `v` gives. It
/// returns `lo` as the loop of libopus leaves it: the first vector whose
/// bits exceed `total`.
#[inline(always)]
fn bisect_vectors(last: i32, total: i32, psum: impl Fn(i32) -> i32) -> i32 {
    let (mut lo, mut hi) = (1, last);
    while lo <= hi {
        let mid = (lo + hi) >> 1;
        if psum(mid) > total {
            hi = mid - 1;
        } else {
            lo = mid + 1;
        }
    }
    lo
}

/// The six halvings of `interp_bits2pulses` over the interpolation steps 0
/// to 64, with `psum(step)` the bits that step gives. It returns `lo`, the
/// largest step tried whose bits do not exceed `total`.
#[inline(always)]
fn bisect_interp(total: i32, psum: impl Fn(i32) -> i32) -> i32 {
    let (mut lo, mut hi) = (0, 1 << 6);
    for _ in 0..6 {
        let mid = (lo + hi) >> 1;
        if psum(mid) > total {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    lo
}

#[allow(clippy::too_many_arguments)]
pub fn clt_compute_allocation(
    m: &CeltMode,
    start: usize,
    end: usize,
    offsets: &[i32],
    cap: &[i32],
    alloc_trim: i32,
    intensity: &mut i32,
    dual_stereo: &mut i32,
    mut total: i32,
    balance_out: &mut i32,
    pulses: &mut [i32],
    ebits: &mut [i32],
    fine_priority: &mut [i32],
    c: i32,
    lm: i32,
    rc: &mut RangeCoder,
    encode: bool,
    prev: i32,
    signal_bandwidth: i32,
) -> i32 {
    total = max(total, 0);
    let nb_ebands = m.nb_ebands;
    let mut skip_start = start;

    let skip_rsv = if total >= (1 << BITRES) {
        1 << BITRES
    } else {
        0
    };
    total -= skip_rsv;

    let mut intensity_rsv = 0;
    let mut dual_stereo_rsv = 0;
    if c == 2 {
        intensity_rsv = LOG2_FRAC_TABLE[end - start] as i32;
        if intensity_rsv > total {
            intensity_rsv = 0;
        } else {
            total -= intensity_rsv;
            dual_stereo_rsv = if total >= (1 << BITRES) {
                1 << BITRES
            } else {
                0
            };
            total -= dual_stereo_rsv;
        }
    }

    let mut thresh_buf = [0i32; MAX_EBANDS];
    let thresh = &mut thresh_buf[..nb_ebands];
    let mut trim_offset_buf = [0i32; MAX_EBANDS];
    let trim_offset = &mut trim_offset_buf[..nb_ebands];
    // `(c * n) << lm` per band, where `n` is the band width at LM 0.
    let mut width_buf = [0i32; MAX_EBANDS];
    let width = &mut width_buf[..nb_ebands];

    for j in start..end {
        let n = (m.e_bands[j + 1] - m.e_bands[j]) as i32;
        width[j] = (c * n) << lm;
        thresh[j] = max(c << BITRES, ((3 * n) << (lm + BITRES)) >> 4);
        trim_offset[j] =
            (c * n * (alloc_trim - 5 - lm) * (end - j - 1) as i32 * (1 << (lm + BITRES))) >> 6;
        if n << lm == 1 {
            trim_offset[j] -= c << BITRES;
        }
    }

    let floor = c << BITRES;
    let last = m.nb_alloc_vectors as i32 - 1;
    // The bits of the allocation vector `mid`.
    let scalar = |mid: i32| {
        let row = &m.alloc_vectors[mid as usize * m.alloc_stride..];
        vector_bits(
            &row[start..end],
            &width[start..end],
            &trim_offset[start..end],
            &offsets[start..end],
            &thresh[start..end],
            &cap[start..end],
            floor,
        )
    };
    #[cfg(target_arch = "aarch64")]
    let lo = if end - start >= NEON_MIN_BANDS {
        let search = SearchLanes::new(&thresh[start..end], &cap[start..end]);
        let (w, t, o) = (
            band_lanes(&width[start..end], 0),
            band_lanes(&trim_offset[start..end], 0),
            band_lanes(&offsets[start..end], 0),
        );
        bisect_vectors(last, total, |mid| {
            let row = &m.alloc_vectors[mid as usize * m.alloc_stride..];
            // SAFETY: NEON is available: src/lib.rs refuses to build for
            // aarch64 without it.
            unsafe { search.vector_bits(&row[start..end], &w, &t, &o, end - start, floor) }
        })
    } else {
        bisect_vectors(last, total, scalar)
    };
    #[cfg(not(target_arch = "aarch64"))]
    let lo = bisect_vectors(last, total, scalar);

    let hi_final = lo as usize;
    let lo_final = (lo - 1) as usize;

    let mut bits1_buf = [0i32; MAX_EBANDS];
    let bits1 = &mut bits1_buf[..nb_ebands];
    let mut bits2_buf = [0i32; MAX_EBANDS];
    let bits2 = &mut bits2_buf[..nb_ebands];

    for j in start..end {
        let n = (m.e_bands[j + 1] - m.e_bands[j]) as i32;
        let mut bits1j = (c * n * m.alloc_vectors[lo_final * m.alloc_stride + j] as i32) << lm >> 2;
        let mut bits2j = if hi_final >= m.nb_alloc_vectors {
            cap[j]
        } else {
            (c * n * m.alloc_vectors[hi_final * m.alloc_stride + j] as i32) << lm >> 2
        };

        if bits1j > 0 {
            bits1j = max(0, bits1j + trim_offset[j]);
        }
        if bits2j > 0 {
            bits2j = max(0, bits2j + trim_offset[j]);
        }
        if lo_final > 0 {
            bits1j += offsets[j];
        }
        bits2j += offsets[j];
        if offsets[j] > 0 {
            skip_start = j;
        }
        bits2j = max(0, bits2j - bits1j);
        bits1[j] = bits1j;
        bits2[j] = bits2j;
    }

    interp_bits2pulses(
        m,
        start,
        end,
        skip_start,
        bits1,
        bits2,
        thresh,
        cap,
        total,
        balance_out,
        skip_rsv,
        intensity,
        intensity_rsv,
        dual_stereo,
        dual_stereo_rsv,
        pulses,
        ebits,
        fine_priority,
        c,
        lm,
        rc,
        encode,
        prev,
        signal_bandwidth,
    )
}

#[allow(clippy::too_many_arguments)]
fn interp_bits2pulses(
    m: &CeltMode,
    start: usize,
    end: usize,
    skip_start: usize,
    bits1: &[i32],
    bits2: &[i32],
    thresh: &[i32],
    cap: &[i32],
    total: i32,
    balance_out: &mut i32,
    skip_rsv: i32,
    intensity: &mut i32,
    mut intensity_rsv: i32,
    dual_stereo: &mut i32,
    dual_stereo_rsv: i32,
    pulses: &mut [i32],
    ebits: &mut [i32],
    fine_priority: &mut [i32],
    c: i32,
    lm: i32,
    rc: &mut RangeCoder,
    encode: bool,
    prev: i32,
    signal_bandwidth: i32,
) -> i32 {
    let mut psum: i32;
    let alloc_floor = c << BITRES;
    let stereo = if c > 1 { 1 } else { 0 };
    let log_m = lm << BITRES;

    let mut bits_buf = [0i32; MAX_EBANDS];
    let bits = &mut bits_buf[..m.nb_ebands];

    // The bands `start..end`, all of one length, so the loops below have no
    // bounds checks.
    let len = end - start;
    let (bands_bits1, bands_bits2) = (&bits1[start..end][..len], &bits2[start..end][..len]);
    let (bands_thresh, bands_cap) = (&thresh[start..end][..len], &cap[start..end][..len]);
    let scalar = |mid: i32| {
        interp_bits(
            bands_bits1,
            bands_bits2,
            mid,
            bands_thresh,
            bands_cap,
            alloc_floor,
        )
    };
    #[cfg(target_arch = "aarch64")]
    let lo = if len >= NEON_MIN_BANDS {
        let search = SearchLanes::new(bands_thresh, bands_cap);
        let (b1, b2) = (band_lanes(bands_bits1, 0), band_lanes(bands_bits2, 0));
        // SAFETY: NEON is available: src/lib.rs refuses to build for aarch64
        // without it.
        bisect_interp(total, |mid| unsafe {
            search.interp_bits(&b1, &b2, mid, alloc_floor)
        })
    } else {
        bisect_interp(total, scalar)
    };
    #[cfg(not(target_arch = "aarch64"))]
    let lo = bisect_interp(total, scalar);
    psum = 0;
    let mut done = false;
    for (j, bits_j) in bits[start..end][..len].iter_mut().enumerate().rev() {
        let tmp = bands_bits1[j] + ((lo * bands_bits2[j]) >> 6);
        // Unlike the search, C caps the floor here too: a band below its
        // threshold keeps `min(floor, cap)`.
        let tmp = searched_bits(tmp, bands_thresh[j], i32::MAX, alloc_floor, &mut done);
        let tmp = min(tmp, bands_cap[j]);
        *bits_j = tmp;
        psum += tmp;
    }

    let mut coded_bands = end;
    let mut total_with_rsv = total;
    loop {
        if coded_bands <= start {
            break;
        }
        let j = coded_bands - 1;
        if j <= skip_start {
            total_with_rsv += skip_rsv;
            break;
        }

        let left = total_with_rsv - psum;
        let nb_samples = (m.e_bands[coded_bands] - m.e_bands[start]) as i32;
        let percoeff = left / nb_samples;
        let left_rem = left - nb_samples * percoeff;
        let rem = max(left_rem - (m.e_bands[j] - m.e_bands[start]) as i32, 0);
        let band_width = (m.e_bands[coded_bands] - m.e_bands[j]) as i32;
        let mut band_bits = bits[j] + percoeff * band_width + rem;

        if band_bits >= max(thresh[j], alloc_floor + (1 << BITRES)) {
            if encode {
                let depth_threshold = if coded_bands > 17 {
                    if (j as i32) < prev { 7 } else { 9 }
                } else {
                    0
                };
                if coded_bands <= start + 2
                    || (band_bits > ((depth_threshold * band_width) << lm << BITRES) >> 4
                        && (j as i32) <= signal_bandwidth)
                {
                    rc.encode_bit_logp(true, 1);
                    break;
                }
                rc.encode_bit_logp(false, 1);
            } else {
                let bit = rc.decode_bit_logp(1);
                if bit {
                    break;
                }
            }
            psum += 1 << BITRES;
            band_bits -= 1 << BITRES;
        }
        psum -= bits[j] + intensity_rsv;
        if intensity_rsv > 0 {
            intensity_rsv = LOG2_FRAC_TABLE[j - start] as i32;
        }
        psum += intensity_rsv;
        if band_bits >= alloc_floor {
            psum += alloc_floor;
            bits[j] = alloc_floor;
        } else {
            bits[j] = 0;
        }
        coded_bands -= 1;
    }

    if intensity_rsv > 0 {
        if encode {
            *intensity = min(*intensity, coded_bands as i32);
            rc.enc_uint(
                (*intensity - start as i32) as u32,
                (coded_bands + 1 - start) as u32,
            );
        } else {
            *intensity = start as i32 + rc.dec_uint((coded_bands + 1 - start) as u32) as i32;
        }
    } else {
        *intensity = 0;
    }

    let mut dual_stereo_rsv_final = dual_stereo_rsv;
    if *intensity <= start as i32 {
        total_with_rsv += dual_stereo_rsv_final;
        dual_stereo_rsv_final = 0;
    }
    if dual_stereo_rsv_final > 0 {
        if encode {
            rc.encode_bit_logp(*dual_stereo != 0, 1);
        } else {
            *dual_stereo = if rc.decode_bit_logp(1) { 1 } else { 0 };
        }
    } else {
        *dual_stereo = 0;
    }

    let mut left = total_with_rsv - psum;
    let nb_samples = (m.e_bands[coded_bands] - m.e_bands[start]) as i32;
    let percoeff = left / nb_samples;
    left -= nb_samples * percoeff;
    for (j, bits_j) in bits[start..coded_bands]
        .iter_mut()
        .enumerate()
        .map(|(i, v)| (i + start, v))
    {
        *bits_j += percoeff * (m.e_bands[j + 1] - m.e_bands[j]) as i32;
    }
    for (j, bits_j) in bits[start..coded_bands]
        .iter_mut()
        .enumerate()
        .map(|(i, v)| (i + start, v))
    {
        let tmp = min(left, (m.e_bands[j + 1] - m.e_bands[j]) as i32);
        *bits_j += tmp;
        left -= tmp;
    }

    let mut balance = 0;
    for j in start..coded_bands {
        let n0 = (m.e_bands[j + 1] - m.e_bands[j]) as i32;
        let n = n0 << lm;
        let bit = bits[j] + balance;

        let mut excess;
        if n > 1 {
            excess = max(bit - cap[j], 0);
            bits[j] = bit - excess;

            let den = c * n
                + (if c == 2 && n > 2 && *dual_stereo == 0 && (j as i32) < *intensity {
                    1
                } else {
                    0
                });
            let nc_log_n = den * (m.log_n[j] as i32 + log_m);
            let mut offset = (nc_log_n >> 1) - den * FINE_OFFSET;

            if n == 2 {
                offset += den << BITRES >> 2;
            }

            if bits[j] + offset < (den * 2) << BITRES {
                offset += nc_log_n >> 2;
            } else if bits[j] + offset < (den * 3) << BITRES {
                offset += nc_log_n >> 3;
            }

            ebits[j] = max(0, bits[j] + offset + (den << (BITRES - 1)));

            let num = ebits[j];
            if den > 0 {
                ebits[j] = ((num as u32 / den as u32) >> BITRES) as i32;
            } else {
                ebits[j] = 0;
            }

            if c * ebits[j] > (bits[j] >> BITRES) {
                ebits[j] = bits[j] >> stereo >> BITRES;
            }
            ebits[j] = min(ebits[j], MAX_FINE_BITS);
            fine_priority[j] = if ebits[j] * (den << BITRES) >= bits[j] + offset {
                1
            } else {
                0
            };
            bits[j] -= (c * ebits[j]) << BITRES;
        } else {
            excess = max(0, bit - (c << BITRES));
            bits[j] = bit - excess;
            ebits[j] = 0;
            fine_priority[j] = 1;
        }

        if excess > 0 {
            let extra_fine = min(excess >> (stereo + BITRES), MAX_FINE_BITS - ebits[j]);
            ebits[j] += extra_fine;
            let extra_bits = (extra_fine * c) << BITRES;
            fine_priority[j] = if extra_bits >= excess - balance { 1 } else { 0 };
            excess -= extra_bits;
        }
        balance = excess;
        pulses[j] = bits[j];
    }
    *balance_out = balance;

    for j in coded_bands..end {
        ebits[j] = bits[j] >> stereo >> BITRES;
        bits[j] = 0;
        fine_priority[j] = if ebits[j] < 1 { 1 } else { 0 };
        pulses[j] = 0;
    }

    coded_bands as i32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modes::default_mode;

    /// The starts of the rows of the pulse cache of the mode.
    fn row_starts() -> std::vec::Vec<usize> {
        let m = default_mode();
        let mut starts: std::vec::Vec<usize> = m
            .cache
            .index
            .iter()
            .filter(|&&i| i >= 0)
            .map(|&i| i as usize)
            .collect();
        starts.sort_unstable();
        starts.dedup();
        starts
    }

    /// `bits2pulses_count` relies on rows whose costs do not decrease and
    /// that have at most 40 entries.
    #[test]
    fn pulse_cache_rows_do_not_decrease() {
        let m = default_mode();
        for start in row_starts() {
            let k = m.cache.bits[start] as usize;
            assert!((1..=40).contains(&k), "row at {start} has {k} entries");
            let row = &m.cache.bits[start + 1..=start + k];
            assert!(
                row.windows(2).all(|w| w[0] <= w[1]),
                "row at {start} decreases"
            );
        }
    }

    /// `bits2pulses` gives the result of the libopus bisection for every
    /// band, frame size and bit count, on every target.
    #[test]
    fn bits2pulses_matches_the_bisection() {
        let m = default_mode();
        for lm in -1..=3 {
            for band in 0..m.nb_ebands {
                let idx = (lm + 1) as usize * m.nb_ebands + band;
                let start = m.cache.index[idx];
                for bits in -40..600 {
                    let want = if start < 0 {
                        0
                    } else {
                        bits2pulses_bisect(&m.cache.bits[start as usize..], bits)
                    };
                    assert_eq!(
                        bits2pulses(m, band, lm, bits),
                        want,
                        "band {band} lm {lm} bits {bits}"
                    );
                }
            }
        }
    }

    /// Small deterministic generator, so that the tests need no dependency.
    #[cfg(target_arch = "aarch64")]
    struct Lcg(u64);
    #[cfg(target_arch = "aarch64")]
    impl Lcg {
        fn below(&mut self, n: u32) -> i32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 33) % n as u64) as i32
        }
    }

    /// The vector forms of the two allocation searches give the sums of the
    /// scalar loops, for frames of 1 to 21 bands and values around the
    /// thresholds, the floor and the caps.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn allocation_searches_match_the_scalar_loops() {
        let mut rng = Lcg(3);
        for round in 0..20_000 {
            let len = 1 + (round % MAX_EBANDS);
            let floor = 8 << rng.below(2);
            let scale: u32 = [4, 40, 400, 4000][round % 4];
            let v = |rng: &mut Lcg| rng.below(scale) - (scale / 8) as i32;
            let thresh: std::vec::Vec<i32> = (0..len).map(|_| floor + rng.below(scale)).collect();
            let cap: std::vec::Vec<i32> = (0..len).map(|_| rng.below(2 * scale)).collect();
            let search = SearchLanes::new(&thresh, &cap);

            // `vector_bits`: bytes of a row, widths, trims and offsets.
            let row: std::vec::Vec<u8> = (0..len).map(|_| rng.below(256) as u8).collect();
            let width: std::vec::Vec<i32> = (0..len).map(|_| 1 + rng.below(352)).collect();
            let trim: std::vec::Vec<i32> = (0..len).map(|_| v(&mut rng)).collect();
            let offsets: std::vec::Vec<i32> = (0..len).map(|_| rng.below(scale)).collect();
            let want = vector_bits(&row, &width, &trim, &offsets, &thresh, &cap, floor);
            // SAFETY: NEON is available: src/lib.rs refuses to build for
            // aarch64 without it.
            let got = unsafe {
                search.vector_bits(
                    &row,
                    &band_lanes(&width, 0),
                    &band_lanes(&trim, 0),
                    &band_lanes(&offsets, 0),
                    len,
                    floor,
                )
            };
            assert_eq!(got, want, "vector_bits, round {round}");

            // `interp_bits`: every interpolation step of the bisection.
            let bits1: std::vec::Vec<i32> = (0..len).map(|_| rng.below(scale)).collect();
            let bits2: std::vec::Vec<i32> = (0..len).map(|_| rng.below(scale)).collect();
            let (l1, l2) = (band_lanes(&bits1, 0), band_lanes(&bits2, 0));
            for mid in 0..=64 {
                let want = interp_bits(&bits1, &bits2, mid, &thresh, &cap, floor);
                // SAFETY: as above.
                let got = unsafe { search.interp_bits(&l1, &l2, mid, floor) };
                assert_eq!(got, want, "interp_bits, round {round}, mid {mid}");
            }
        }
    }

    /// The counting form directly, for every row that has 48 readable bytes.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn bits2pulses_count_matches_the_bisection() {
        let m = default_mode();
        let mut rows = 0;
        for start in row_starts() {
            let Some(row) = m.cache.bits.get(start..start + 48) else {
                continue;
            };
            let row: &[u8; 48] = row.try_into().unwrap();
            rows += 1;
            for bits in -40..600 {
                let want = bits2pulses_bisect(&m.cache.bits[start..], bits);
                assert_eq!(
                    bits2pulses_count(row, bits),
                    Some(want),
                    "row {start} bits {bits}"
                );
            }
        }
        assert!(rows >= 15, "only {rows} rows checked");
    }
}
