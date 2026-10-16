//! The public functions of `src/pitch.rs` and `src/celt_lpc.rs` against plain
//! reference code.
//!
//! The reference code in this file is written one sample at a time. It follows
//! the order of additions that each function has on the target:
//!
//! - aarch64 adds most sums of products with fused multiply-add, in the two
//!   orders described at the top of `src/pitch.rs`;
//! - every other target that this file is compiled for multiplies and adds
//!   separately, one product after the other.
//!
//! The x86 kernels use other orders again, which this file does not describe,
//! so it is not compiled for x86.
//!
//! Every input and output slice is placed directly before or directly after
//! a memory page that must not be touched (see `Region`). A kernel that reads
//! or writes even one sample outside its slice stops the test with a memory
//! fault.

#![cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
// The reference functions repeat the index-based loops of the C code.
#![allow(clippy::needless_range_loop)]

use opus1::celt_lpc::autocorr;
use opus1::pitch::{
    dual_inner_prod, inner_prod, pitch_downsample, pitch_search, pitch_xcorr, remove_doubling,
};

/// True when the library uses its aarch64 kernels.
const NEON: bool = cfg!(target_arch = "aarch64");

// ---------------------------------------------------------------------------
// Memory with pages around it that must not be touched
// ---------------------------------------------------------------------------

#[cfg(all(
    any(target_os = "macos", target_os = "linux"),
    target_pointer_width = "64"
))]
mod region {
    use core::ffi::{c_int, c_void};

    unsafe extern "C" {
        fn mmap(
            addr: *mut c_void,
            len: usize,
            prot: c_int,
            flags: c_int,
            fd: c_int,
            offset: i64,
        ) -> *mut c_void;
        fn mprotect(addr: *mut c_void, len: usize, prot: c_int) -> c_int;
        fn munmap(addr: *mut c_void, len: usize) -> c_int;
    }

    /// 64 KiB: a multiple of every page size in use (4, 16 and 64 KiB).
    const SPAN: usize = 1 << 16;
    const PROT_NONE: c_int = 0;
    const PROT_READ_WRITE: c_int = 3;
    const MAP_PRIVATE: c_int = 2;
    #[cfg(target_os = "macos")]
    const MAP_ANON: c_int = 0x1000;
    #[cfg(target_os = "linux")]
    const MAP_ANON: c_int = 0x20;

    /// 64 KiB of memory between two blocks of 64 KiB that may be neither read
    /// nor written.
    pub struct Region {
        base: *mut u8,
    }

    impl Region {
        pub fn new() -> Self {
            // SAFETY: a new private mapping; the two calls of `mprotect`
            // cover its first and its last third.
            unsafe {
                let base = mmap(
                    core::ptr::null_mut(),
                    3 * SPAN,
                    PROT_READ_WRITE,
                    MAP_PRIVATE | MAP_ANON,
                    -1,
                    0,
                );
                assert!(base as isize != -1, "mmap failed");
                assert_eq!(mprotect(base, SPAN, PROT_NONE), 0);
                assert_eq!(mprotect(base.add(2 * SPAN), SPAN, PROT_NONE), 0);
                Region {
                    base: base as *mut u8,
                }
            }
        }

        /// A copy of `data` that ends at the protected block after the
        /// region (`at_end`), or starts at the one before it.
        pub fn place(&mut self, data: &[f32], at_end: bool) -> &mut [f32] {
            let bytes = core::mem::size_of_val(data);
            assert!(bytes <= SPAN);
            // SAFETY: the middle third of the mapping is readable and
            // writable, it is aligned for f32, and `bytes` fit in it. The
            // returned slice borrows `self`.
            unsafe {
                let start = if at_end {
                    self.base.add(2 * SPAN - bytes)
                } else {
                    self.base.add(SPAN)
                } as *mut f32;
                core::ptr::copy_nonoverlapping(data.as_ptr(), start, data.len());
                core::slice::from_raw_parts_mut(start, data.len())
            }
        }
    }

    impl Drop for Region {
        fn drop(&mut self) {
            // SAFETY: the mapping made in `new`.
            unsafe {
                munmap(self.base as *mut c_void, 3 * SPAN);
            }
        }
    }
}

/// Without `mmap`, a plain buffer of exactly the length asked for.
#[cfg(not(all(
    any(target_os = "macos", target_os = "linux"),
    target_pointer_width = "64"
)))]
mod region {
    pub struct Region {
        buf: Vec<f32>,
    }

    impl Region {
        pub fn new() -> Self {
            Region { buf: Vec::new() }
        }

        pub fn place(&mut self, data: &[f32], _at_end: bool) -> &mut [f32] {
            self.buf.clear();
            self.buf.extend_from_slice(data);
            &mut self.buf
        }
    }
}

use region::Region;

// ---------------------------------------------------------------------------
// Test inputs
// ---------------------------------------------------------------------------

/// SplitMix64.
struct Gen(u64);

impl Gen {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    /// Uniform in [-1, 1).
    fn signed(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 23) as f32 - 1.0
    }
}

#[derive(Clone, Copy, Debug)]
enum Fill {
    /// Bell-shaped noise at the level of 16-bit audio.
    Noise,
    /// Mostly zeros of both signs, with a few loud samples.
    Sparse,
    /// Any bit pattern: all magnitudes, subnormal numbers, infinities and
    /// not-a-number values.
    AnyBits,
    /// So large that squares and sums overflow.
    Overflow,
    /// So small that products are subnormal or zero.
    Underflow,
    /// Subnormal numbers and zeros of both signs. Halving or quartering such
    /// a number is not exact, so a fused multiply-add gives another result
    /// than a separate multiply and add even with the factors 0.5 and 0.25.
    Subnormal,
}

const FILLS: [Fill; 6] = [
    Fill::Noise,
    Fill::Sparse,
    Fill::AnyBits,
    Fill::Overflow,
    Fill::Underflow,
    Fill::Subnormal,
];

fn fill(g: &mut Gen, n: usize, kind: Fill) -> Vec<f32> {
    (0..n)
        .map(|_| match kind {
            Fill::Noise => 9000.0 * (g.signed() + g.signed() + g.signed()),
            Fill::Sparse => match g.below(8) {
                0 => 30000.0 * g.signed(),
                1 => -0.0,
                _ => 0.0,
            },
            Fill::AnyBits => f32::from_bits(g.next() as u32),
            Fill::Overflow => (1.0e18 + 3.0e19 * g.signed().abs()) * g.signed().signum(),
            Fill::Underflow => 1.0e-22 * g.signed(),
            Fill::Subnormal => f32::from_bits(g.next() as u32 & 0x807f_ffff),
        })
        .collect()
}

/// A signal with a period of `period` samples that drifts slowly, plus noise.
fn voiced(g: &mut Gen, n: usize, period: f32, noise: f32) -> Vec<f32> {
    let cycle: Vec<f32> = (0..64).map(|_| g.signed()).collect();
    let drift = 0.02 * g.signed();
    (0..n)
        .map(|i| {
            let phase = i as f32 / (period * (1.0 + drift * i as f32 / n.max(1) as f32));
            let pos = phase.fract() * 64.0;
            let (a, b) = (cycle[pos as usize % 64], cycle[(pos as usize + 1) % 64]);
            6000.0 * (a + (b - a) * pos.fract()) + noise * g.signed()
        })
        .collect()
}

/// Equal bits. Two not-a-number values count as equal.
fn same(a: f32, b: f32) -> bool {
    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
}

#[track_caller]
fn assert_same(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: lengths differ");
    for (i, (&a, &b)) in got.iter().zip(want).enumerate() {
        assert!(
            same(a, b),
            "{what}: element {i}: {a:e} ({:#010x}), expected {b:e} ({:#010x})",
            a.to_bits(),
            b.to_bits()
        );
    }
}

// ---------------------------------------------------------------------------
// Reference code
// ---------------------------------------------------------------------------

/// `inner_prod`.
///
/// aarch64: four running sums that take the samples 4m, 4m+1, 4m+2 and 4m+3
/// with fused multiply-add; they are added as (s0 + s1) + (s2 + s3); the last
/// `n % 4` products follow with a separate multiply and add.
/// Elsewhere: one running sum, a separate multiply and add per sample.
fn ref_inner(x: &[f32], y: &[f32], n: usize) -> f32 {
    if NEON {
        let whole = n / 4 * 4;
        let mut s = [0.0f32; 4];
        for j in 0..whole {
            s[j % 4] = x[j].mul_add(y[j], s[j % 4]);
        }
        let mut sum = (s[0] + s[1]) + (s[2] + s[3]);
        for j in whole..n {
            sum += x[j] * y[j];
        }
        sum
    } else {
        let mut sum = 0.0f32;
        for j in 0..n {
            sum += x[j] * y[j];
        }
        sum
    }
}

/// `pitch_xcorr`: `out[k] = sum over j < len of x[j] * y[j + k]`.
///
/// aarch64: with 32 lags or more, the lags below the largest multiple of four
/// use one running sum and a fused multiply-add per sample. All other lags
/// are computed like `inner_prod`.
/// Elsewhere: one running sum, a separate multiply and add per sample.
fn ref_xcorr(x: &[f32], y: &[f32], len: usize, lags: usize) -> Vec<f32> {
    let by_sample = if NEON && lags >= 32 { lags & !3 } else { 0 };
    (0..lags)
        .map(|k| {
            if k < by_sample {
                let mut sum = 0.0f32;
                for j in 0..len {
                    sum = x[j].mul_add(y[j + k], sum);
                }
                sum
            } else {
                ref_inner(x, &y[k..], len)
            }
        })
        .collect()
}

/// One plus the energy of `y`, the start value in `find_best_pitch`.
fn ref_energy_plus_one(y: &[f32]) -> f32 {
    if NEON {
        let whole = y.len() / 4 * 4;
        let mut s = [0.0f32; 4];
        for j in 0..whole {
            s[j % 4] = y[j].mul_add(y[j], s[j % 4]);
        }
        let mut sum = 1.0f32 + ((s[0] + s[1]) + (s[2] + s[3]));
        for &v in &y[whole..] {
            sum += v * v;
        }
        sum
    } else {
        let mut sum = 1.0f32;
        for &v in y {
            sum += v * v;
        }
        sum
    }
}

/// `keep_limit` is true for the real algorithm. With false the running energy
/// is not held at 1 or above; the tests use that to show that their inputs
/// reach the limit.
fn ref_find_best_pitch(
    xcorr: &[f32],
    y: &[f32],
    len: usize,
    max_pitch: usize,
    keep_limit: bool,
) -> [usize; 2] {
    let mut best_num = [-1.0f32, -1.0f32];
    let mut best_den = [0.0f32, 0.0f32];
    let mut best_pitch = [0usize, 1];
    let mut syy = ref_energy_plus_one(&y[..len]);
    for i in 0..max_pitch {
        if xcorr[i] > 0.0 {
            let xc16 = xcorr[i] * 1e-12;
            let num = xc16 * xc16;
            if num * best_den[1] > best_num[1] * syy {
                if num * best_den[0] > best_num[0] * syy {
                    best_num[1] = best_num[0];
                    best_den[1] = best_den[0];
                    best_pitch[1] = best_pitch[0];
                    best_num[0] = num;
                    best_den[0] = syy;
                    best_pitch[0] = i;
                } else {
                    best_num[1] = num;
                    best_den[1] = syy;
                    best_pitch[1] = i;
                }
            }
        }
        syy += y[i + len] * y[i + len] - y[i] * y[i];
        if keep_limit && syy < 1.0 {
            syy = 1.0;
        }
    }
    best_pitch
}

/// `pitch_search` as it was before the package: every step written out.
fn ref_pitch_search(
    x_lp: &[f32],
    y: &[f32],
    len: usize,
    max_pitch: usize,
    keep_limit: bool,
) -> usize {
    let max_pitch = max_pitch >> 1;
    let len = len >> 1;
    let lag = len + max_pitch;

    let x_lp4: Vec<f32> = (0..len >> 1).map(|j| x_lp[2 * j]).collect();
    let y_lp4: Vec<f32> = (0..lag >> 1).map(|j| y[2 * j]).collect();

    let mut xcorr = ref_xcorr(&x_lp4, &y_lp4, len >> 1, max_pitch >> 1);
    let mut best = ref_find_best_pitch(&xcorr, &y_lp4, len >> 1, max_pitch >> 1, keep_limit);

    xcorr.resize(max_pitch, 0.0);
    for i in 0..max_pitch {
        xcorr[i] = -1.0;
        if (i as i32 - 2 * best[0] as i32).abs() > 2 && (i as i32 - 2 * best[1] as i32).abs() > 2 {
            continue;
        }
        xcorr[i] = ref_inner(x_lp, &y[i..], len);
        if xcorr[i] < -1.0 {
            xcorr[i] = -1.0;
        }
    }
    best = ref_find_best_pitch(&xcorr, y, len, max_pitch, keep_limit);

    let mut offset = 0isize;
    if best[0] > 0 && best[0] < max_pitch - 1 {
        let a = xcorr[best[0] - 1];
        let b = xcorr[best[0]];
        let c = xcorr[best[0] + 1];
        if (c - a) > 0.7 * (b - a) {
            offset = 1;
        } else if (a - c) > 0.7 * (b - c) {
            offset = -1;
        }
    }
    ((2 * best[0]) as isize).wrapping_sub(offset) as usize
}

fn ref_pitch_gain(xy: f32, xx: f32, yy: f32) -> f32 {
    if xy <= 0.0 || xx <= 0.0 || yy <= 0.0 {
        return 0.0;
    }
    xy / (1.0 + xx * yy).sqrt()
}

/// `remove_doubling` as it was before the package: one inner product per
/// call, and the energy of every lag up to `max_period / 2`.
///
/// `keep_limit` is true for the real algorithm. With false the running energy
/// is not held at 0 or above (see `ref_find_best_pitch`).
#[allow(clippy::too_many_arguments)]
fn ref_remove_doubling(
    x: &[f32],
    max_period: usize,
    min_period: usize,
    n: usize,
    t0_ptr: &mut usize,
    prev_period: usize,
    prev_gain: f32,
    keep_limit: bool,
) -> f32 {
    const SECOND_CHECK: [usize; 16] = [0, 0, 3, 2, 3, 2, 5, 2, 3, 2, 3, 2, 5, 2, 3, 2];
    let min_period0 = min_period;
    let max_period = max_period / 2;
    let min_period = min_period / 2;
    let prev_period = prev_period / 2;
    let n = n / 2;
    let t0 = (*t0_ptr / 2).min(max_period - 1);
    let mut t = t0;
    let x_target = &x[max_period..];

    let xx = ref_inner(x_target, x_target, n);
    let xy = ref_inner(x_target, &x[max_period - t0..], n);
    let mut yy_lookup = vec![0.0f32; max_period + 1];
    yy_lookup[0] = xx;
    let mut yy_curr = xx;
    for i in 1..=max_period {
        yy_curr = yy_curr + x[max_period - i] * x[max_period - i]
            - x[max_period + n - i] * x[max_period + n - i];
        if keep_limit && yy_curr < 0.0 {
            yy_curr = 0.0;
        }
        yy_lookup[i] = yy_curr;
    }

    let mut best_xy = xy;
    let mut best_yy = yy_lookup[t0];
    let mut g = ref_pitch_gain(best_xy, xx, best_yy);
    let g0 = g;
    for k in 2..=15 {
        let t1 = (2 * t0 + k) / (2 * k);
        if t1 < min_period {
            break;
        }
        let t1b = if k == 2 {
            if t1 + t0 > max_period { t0 } else { t0 + t1 }
        } else {
            (2 * SECOND_CHECK[k] * t0 + k) / (2 * k)
        };
        let xy_a = ref_inner(x_target, &x[max_period - t1..], n);
        let xy_b = ref_inner(x_target, &x[max_period - t1b..], n);
        let xy_new = 0.5 * (xy_a + xy_b);
        let yy_new = 0.5 * (yy_lookup[t1] + yy_lookup[t1b]);
        let g1 = ref_pitch_gain(xy_new, xx, yy_new);

        let mut cont = 0.0f32;
        if (t1 as i32 - prev_period as i32).abs() <= 1 {
            cont = prev_gain;
        } else if (t1 as i32 - prev_period as i32).abs() <= 2 && 5 * k * k < t0 {
            cont = 0.5 * prev_gain;
        }
        let mut thresh = (0.7 * g0 - cont).max(0.3);
        if t1 < 3 * min_period {
            thresh = (0.85 * g0 - cont).max(0.4);
        } else if t1 < 2 * min_period {
            thresh = (0.9 * g0 - cont).max(0.5);
        }
        if g1 > thresh {
            best_xy = xy_new;
            best_yy = yy_new;
            t = t1;
            g = g1;
        }
    }

    best_xy = best_xy.max(0.0);
    let pg = if best_yy <= best_xy {
        1.0f32
    } else {
        best_xy / (best_yy + 1.0)
    };

    let mut xcorr = [0.0f32; 3];
    for (k, out) in xcorr.iter_mut().enumerate() {
        *out = ref_inner(x_target, &x[max_period - (t + k - 1)..], n);
    }
    let mut offset = 0i32;
    if (xcorr[2] - xcorr[0]) > 0.7 * (xcorr[1] - xcorr[0]) {
        offset = 1;
    } else if (xcorr[0] - xcorr[2]) > 0.7 * (xcorr[1] - xcorr[2]) {
        offset = -1;
    }

    *t0_ptr = ((2 * t as i32 + offset) as usize).max(min_period0);
    pg.min(g)
}

/// The spectral flattening at the end of `pitch_downsample` (libopus,
/// celt/pitch.c): an LPC analysis of order 4 and a 5-tap filter built from
/// it. The autocorrelation and the LPC recursion are the library's own; they
/// have their tests further down.
fn ref_flatten(x_lp: &mut [f32]) {
    let len = x_lp.len();
    if len < 5 {
        return;
    }
    let mut ac = [0.0f32; 5];
    opus1::celt_lpc::autocorr(x_lp, &mut ac, None, 0, 4, len);
    ac[0] *= 1.0001;
    for i in 1..=4 {
        let f = 0.008 * i as f32;
        ac[i] -= ac[i] * f * f;
    }
    let mut coef = [0.0f32; 4];
    opus1::celt_lpc::lpc(&mut coef, &ac, 4);
    let mut tmp = 1.0f32;
    for c in coef.iter_mut() {
        tmp *= 0.9;
        *c *= tmp;
    }
    let num = [
        coef[0] + 0.8,
        coef[1] + 0.8 * coef[0],
        coef[2] + 0.8 * coef[1],
        coef[3] + 0.8 * coef[2],
        0.8 * coef[3],
    ];
    // x[i] += num[0] * x[i - 1] + .. + num[4] * x[i - 5], on the old values,
    // with zeros before the first sample.
    let mut past = [0.0f32; 5];
    for v in x_lp.iter_mut() {
        let mut sum = *v;
        for j in 0..5 {
            sum += num[j] * past[j];
        }
        past.copy_within(0..4, 1);
        past[0] = *v;
        *v = sum;
    }
}

/// The decimation of `pitch_downsample`, before the spectral flattening.
fn ref_downsample(x: &[&[f32]], len: usize, factor: usize) -> Vec<f32> {
    let offset = factor / 2;
    let mut out = vec![0.0f32; len];
    for i in 1..len {
        let mut val = 0.0f32;
        for x_k in x {
            if factor * i + offset < x_k.len() {
                val += 0.25 * x_k[factor * i - offset]
                    + 0.25 * x_k[factor * i + offset]
                    + 0.5 * x_k[factor * i];
            }
        }
        out[i] = val;
    }
    let mut val = 0.0f32;
    for x_k in x {
        if offset < x_k.len() {
            val += 0.25 * x_k[offset] + 0.5 * x_k[0];
        }
    }
    out[0] = val;
    out
}

/// `autocorr`: the first `n - lag` products of every lag come from
/// `pitch_xcorr`, the others are added to that one by one.
fn ref_autocorr(x: &[f32], window: Option<&[f32]>, overlap: usize, lag: usize) -> Vec<f32> {
    let n = x.len();
    let mut xx = x.to_vec();
    if let Some(win) = window {
        for i in 0..overlap {
            xx[i] *= win[i];
            xx[n - 1 - i] *= win[i];
        }
    }
    let fast_n = n - lag;
    let mut ac = ref_xcorr(&xx, &xx, fast_n, lag + 1);
    for (k, out) in ac.iter_mut().enumerate() {
        let mut d = 0.0f32;
        for i in k + fast_n..n {
            d += xx[i] * xx[i - k];
        }
        *out += d;
    }
    ac
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn inner_products_follow_the_order_of_the_target() {
    let mut g = Gen(0xA11CE);
    let (mut rx, mut ry1, mut ry2) = (Region::new(), Region::new(), Region::new());
    for kind in FILLS {
        for n in (0..=70).chain([97, 255, 256, 481, 963]) {
            let (x, y1, y2) = (
                fill(&mut g, n, kind),
                fill(&mut g, n, kind),
                fill(&mut g, n, kind),
            );
            let want = [ref_inner(&x, &y1, n), ref_inner(&x, &y2, n)];
            for at_end in [true, false] {
                let what = format!("{kind:?} n {n} at_end {at_end}");
                let x = rx.place(&x, at_end);
                let y1 = ry1.place(&y1, at_end);
                let y2 = ry2.place(&y2, at_end);
                assert_same(&[inner_prod(x, y1, n)], &want[..1], &what);
                let (a, b) = dual_inner_prod(x, y1, y2, n);
                assert_same(&[a, b], &want, &what);
            }
        }
    }
}

#[test]
fn pitch_xcorr_follows_the_order_of_the_target() {
    let mut g = Gen(0xB0B);
    let (mut rx, mut ry, mut rout) = (Region::new(), Region::new(), Region::new());
    // Every pair of small sizes, then the sizes of the encoder (30 to 240
    // samples, 244 lags) and some that are larger or lopsided.
    let small = (1..=37).flat_map(|len| (1..=72).map(move |lags| (len, lags)));
    let large = [
        (30, 244),
        (60, 244),
        (120, 244),
        (240, 244),
        (240, 245),
        (239, 246),
        (238, 247),
        (61, 131),
        (479, 489),
        (5, 300),
        (1, 257),
        (1000, 3),
    ];
    let pairs: Vec<(usize, usize)> = small.chain(large).collect();
    for kind in FILLS {
        for &(len, lags) in &pairs {
            let x = fill(&mut g, len, kind);
            // Exactly as long as the last lag needs.
            let y = fill(&mut g, len + lags - 1, kind);
            let want = ref_xcorr(&x, &y, len, lags);
            for at_end in [true, false] {
                let x = rx.place(&x, at_end);
                let y = ry.place(&y, at_end);
                let out = rout.place(&vec![f32::NAN; lags], at_end);
                pitch_xcorr(x, y, out, len, lags);
                assert_same(
                    out,
                    &want,
                    &format!("{kind:?} len {len} lags {lags} at_end {at_end}"),
                );
            }
        }
    }
}

#[test]
fn pitch_xcorr_with_a_short_y_writes_only_the_lags_that_fit() {
    let mut g = Gen(0xC0FFEE);
    let (mut rx, mut ry, mut rout) = (Region::new(), Region::new(), Region::new());
    for (len, lags, y_len) in [
        (40, 60, 50),
        (9, 80, 70),
        (16, 4, 15),
        (16, 4, 16),
        (3, 100, 2),
        (0, 5, 9),
    ] {
        let x = fill(&mut g, len, Fill::Noise);
        let y = fill(&mut g, y_len, Fill::Noise);
        let fit = if len == 0 {
            0
        } else {
            lags.min((y_len + 1).saturating_sub(len))
        };
        let mut want = ref_xcorr(&x, &y, len, fit);
        want.resize(lags, 7.0);
        for at_end in [true, false] {
            let x = rx.place(&x, at_end);
            let y = ry.place(&y, at_end);
            let out = rout.place(&vec![7.0f32; lags], at_end);
            pitch_xcorr(x, y, out, len, lags);
            assert_same(
                out,
                &want,
                &format!("len {len} lags {lags} y {y_len} at_end {at_end}"),
            );
        }
    }
}

#[test]
fn pitch_downsample_matches_the_formula() {
    let mut g = Gen(0xD0E);
    let (mut r0, mut r1, mut r2, mut rout) =
        (Region::new(), Region::new(), Region::new(), Region::new());
    for kind in FILLS {
        // Every length around the vector width, and the lengths the encoder
        // uses for frames of 2.5 to 20 ms (572 to 992).
        for len in (1..=45).chain([572, 632, 752, 992, 1001]) {
            for (c, factor) in [(1, 2), (2, 2), (3, 2), (1, 1), (2, 3), (1, 4)] {
                // Channels of exactly `factor * len` samples (what the
                // encoder passes), and channels that are shorter.
                for cut in [
                    (0, 0),
                    (1, 0),
                    (0, 2),
                    (g.below(9), g.below(9)),
                    (factor * len, 0),
                ] {
                    let full = factor * len;
                    let x0 = fill(&mut g, full - cut.0.min(full), kind);
                    let x1 = fill(&mut g, full - cut.1.min(full), kind);
                    let x2 = fill(&mut g, full, kind);
                    let chans: [&[f32]; 3] = [&x0, &x1, &x2];
                    let mut want = ref_downsample(&chans[..c], len, factor);
                    ref_flatten(&mut want);
                    for at_end in [true, false] {
                        let p0 = &*r0.place(&x0, at_end);
                        let p1 = &*r1.place(&x1, at_end);
                        let p2 = &*r2.place(&x2, at_end);
                        let placed: [&[f32]; 3] = [p0, p1, p2];
                        let out = rout.place(&vec![f32::NAN; len], at_end);
                        pitch_downsample(&placed[..c], out, len, c, factor);
                        assert_same(
                            out,
                            &want,
                            &format!(
                                "{kind:?} len {len} c {c} factor {factor} cut {cut:?} at_end {at_end}"
                            ),
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn pitch_search_matches_the_old_algorithm() {
    let mut g = Gen(0x5EA2C4);
    let (mut rx, mut ry) = (Region::new(), Region::new());
    let mut results = std::collections::BTreeSet::new();
    // (len, max_pitch) as the caller passes them. The encoder uses a maximum
    // lag of 979 and 120 to 960 samples.
    let sizes = [
        (120usize, 979usize),
        (240, 979),
        (480, 979),
        (960, 979),
        (962, 977),
        (64, 130),
        (90, 201),
        (34, 67),
        (130, 64),
        (16, 12),
        (6, 4),
        (4, 9),
    ];
    for &(len, max_pitch) in &sizes {
        // What the function reads: `len / 2` samples of `x_lp`, and
        // `len / 2 + max_pitch / 2` samples of `y`.
        let (nx, ny) = (len >> 1, (len >> 1) + (max_pitch >> 1));
        for case in 0..40 {
            let (x_lp, y) = match case {
                // The target is a part of `y`, as in the encoder, with a
                // period the search can find or not.
                0..30 => {
                    let period = 3.0 + (max_pitch as f32) * 0.6 * g.signed().abs();
                    let noise = [0.0, 300.0, 3000.0, 20000.0][case % 4];
                    let y = voiced(&mut g, ny, period, noise);
                    let start = if case % 3 == 0 {
                        ny - nx
                    } else {
                        g.below(ny - nx + 1)
                    };
                    (y[start..start + nx].to_vec(), y)
                }
                30..35 => (vec![0.0; nx], vec![0.0; ny]),
                _ => {
                    let kind = FILLS[case % FILLS.len()];
                    (fill(&mut g, nx, kind), fill(&mut g, ny, kind))
                }
            };
            let want = ref_pitch_search(&x_lp, &y, len, max_pitch, true);
            for at_end in [true, false] {
                let x_lp = rx.place(&x_lp, at_end);
                let y = ry.place(&y, at_end);
                let got = pitch_search(x_lp, y, len, max_pitch);
                assert_eq!(
                    got, want,
                    "len {len} max_pitch {max_pitch} case {case} at_end {at_end}"
                );
            }
            if len == 960 {
                results.insert(want);
            }
        }
    }
    // The inputs must lead to many different lags.
    assert!(results.len() > 12, "only {} different lags", results.len());
}

#[test]
fn remove_doubling_matches_the_old_algorithm() {
    let mut g = Gen(0xD0B1E);
    let mut rx = Region::new();
    let mut changed_period = 0;
    let mut cases = 0;
    // (max_period, min_period, n). The encoder uses (1024, 15) with 120 to
    // 960 samples.
    let sizes = [
        (1024usize, 15usize, 120usize),
        (1024, 15, 240),
        (1024, 15, 480),
        (1024, 15, 960),
        (1024, 15, 90),
        (300, 10, 100),
        (64, 4, 40),
        (2046, 15, 46),
    ];
    for &(max_period, min_period, n) in &sizes {
        let total = max_period / 2 + n / 2;
        for case in 0..60 {
            let period = 4.0 + (max_period as f32 / 8.0) * g.signed().abs();
            let x = match case % 12 {
                0..8 => voiced(
                    &mut g,
                    total,
                    period,
                    [0.0, 200.0, 2000.0, 9000.0][case % 4],
                ),
                8 => vec![0.0; total],
                k => fill(&mut g, total, FILLS[(case + k) % FILLS.len()]),
            };
            // The period the search found: a multiple of the true one, any
            // value, or one of the largest. Below 2 it is not a valid input.
            let t0 = match case % 5 {
                0 => (2.0 * period * (1 + g.below(6)) as f32) as usize,
                1 => max_period - g.below(4),
                2 => max_period + g.below(100),
                _ => g.below(max_period),
            }
            .max(2);
            let prev_period = g.below(max_period);
            let prev_gain = g.signed().abs();

            let mut want_t = t0;
            let want = ref_remove_doubling(
                &x,
                max_period,
                min_period,
                n,
                &mut want_t,
                prev_period,
                prev_gain,
                true,
            );
            for at_end in [true, false] {
                let x = rx.place(&x, at_end);
                let mut got_t = t0;
                let got = remove_doubling(
                    x,
                    max_period,
                    min_period,
                    n,
                    &mut got_t,
                    prev_period,
                    prev_gain,
                );
                let what = format!(
                    "sizes {max_period} {min_period} {n} case {case} t0 {t0} at_end {at_end}"
                );
                assert_eq!(got_t, want_t, "{what}");
                assert!(same(got, want), "{what}: gain {got:e}, expected {want:e}");
            }
            cases += 1;
            if want_t + 3 < t0.min(max_period - 1) {
                changed_period += 1;
            }
        }
    }
    // The inputs must reach the branch that picks a shorter period.
    assert!(
        changed_period * 8 > cases,
        "{changed_period} of {cases} cases picked a shorter period"
    );
}

/// Quiet noise of level `quiet` with a loud sample at each index in `loud`.
fn quiet_with_loud_samples(g: &mut Gen, n: usize, quiet: f32, loud: &[usize]) -> Vec<f32> {
    let mut v: Vec<f32> = (0..n).map(|_| quiet * g.signed()).collect();
    for &i in loud {
        v[i] = (20000.0 + 12000.0 * g.signed().abs()) * g.signed().signum();
    }
    v
}

/// The window energy in `find_best_pitch` is a running sum. When loud samples
/// leave the window, what is left of the sum is a rounding error: zero, or
/// some tens above or below it. The energy is then held at 1. These inputs
/// reach that limit, in the coarse and in the fine search.
#[test]
fn pitch_search_holds_the_window_energy_at_one() {
    let mut g = Gen(0x11317);
    let (mut rx, mut ry) = (Region::new(), Region::new());
    let (mut cases, mut limit_decides) = (0, 0);
    for &(len, max_pitch) in &[
        (120usize, 979usize),
        (240, 979),
        (480, 979),
        (960, 979),
        (64, 130),
    ] {
        let (nx, ny) = (len >> 1, (len >> 1) + (max_pitch >> 1));
        for case in 0..60 {
            // The target is the part of `y` that starts at `lag`, so the
            // search has a lag to find. The loud samples are all before it:
            // they are in the first windows and have left when the window
            // reaches `lag`.
            let lag = 8 + g.below((max_pitch >> 1) - 8);
            let first = lag.min(nx);
            let loud: Vec<usize> = (0..1 + case % 4).map(|_| g.below(first)).collect();
            let quiet = [0.01, 0.3, 1.0][case % 3];
            let y = quiet_with_loud_samples(&mut g, ny, quiet, &loud);
            let x_lp = y[lag..lag + nx].to_vec();

            let want = ref_pitch_search(&x_lp, &y, len, max_pitch, true);
            for at_end in [true, false] {
                let got = pitch_search(
                    rx.place(&x_lp, at_end),
                    ry.place(&y, at_end),
                    len,
                    max_pitch,
                );
                assert_eq!(
                    got, want,
                    "len {len} max_pitch {max_pitch} case {case} at_end {at_end}"
                );
            }
            cases += 1;
            if ref_pitch_search(&x_lp, &y, len, max_pitch, false) != want {
                limit_decides += 1;
            }
        }
    }
    // Without the limit a good part of the results must be different, or
    // this test does not show that the limit is applied.
    assert!(
        limit_decides * 5 > cases,
        "the limit changed {limit_decides} of {cases} results"
    );
}

/// The energies in `remove_doubling` are a running sum too, held at 0 (see
/// `pitch_search_holds_the_window_energy_at_one`). Here the loud samples are
/// the last ones of the target, which leave the window first.
#[test]
fn remove_doubling_holds_the_energy_at_zero() {
    let mut g = Gen(0x2E20);
    let mut rx = Region::new();
    let (mut cases, mut limit_decides) = (0, 0);
    for &(max_period, min_period, n) in &[
        (1024usize, 15usize, 120usize),
        (1024, 15, 480),
        (1024, 15, 960),
        (300, 10, 100),
    ] {
        let total = max_period / 2 + n / 2;
        for case in 0..80 {
            let loud: Vec<usize> = (0..2 + case % 3).map(|_| total - 1 - g.below(6)).collect();
            let quiet = [0.01, 0.2, 1.0][case % 3];
            let x = quiet_with_loud_samples(&mut g, total, quiet, &loud);
            let t0 = (16 + g.below(max_period - 16)).max(2);
            let prev_period = g.below(max_period);
            let prev_gain = g.signed().abs();

            let mut want_t = t0;
            let want = ref_remove_doubling(
                &x,
                max_period,
                min_period,
                n,
                &mut want_t,
                prev_period,
                prev_gain,
                true,
            );
            for at_end in [true, false] {
                let mut got_t = t0;
                let got = remove_doubling(
                    rx.place(&x, at_end),
                    max_period,
                    min_period,
                    n,
                    &mut got_t,
                    prev_period,
                    prev_gain,
                );
                let what = format!(
                    "sizes {max_period} {min_period} {n} case {case} t0 {t0} at_end {at_end}"
                );
                assert_eq!(got_t, want_t, "{what}");
                assert!(same(got, want), "{what}: gain {got:e}, expected {want:e}");
            }
            cases += 1;
            let mut other_t = t0;
            let other = ref_remove_doubling(
                &x,
                max_period,
                min_period,
                n,
                &mut other_t,
                prev_period,
                prev_gain,
                false,
            );
            if other_t != want_t || !same(other, want) {
                limit_decides += 1;
            }
        }
    }
    assert!(
        limit_decides * 10 > cases,
        "the limit changed {limit_decides} of {cases} results"
    );
}

#[test]
fn autocorr_matches_the_old_algorithm() {
    let mut g = Gen(0xAC);
    let (mut rx, mut rac, mut rwin) = (Region::new(), Region::new(), Region::new());
    let win: Vec<f32> = (0..160)
        .map(|i| ((i as f32 + 0.5) / 160.0).sqrt())
        .collect();
    for kind in FILLS {
        let mut sizes: Vec<(usize, usize, usize)> = vec![
            // (n, lag, overlap); the last one is the largest signal that a
            // window accepts.
            (1, 0, 0),
            (2, 1, 1),
            (9, 8, 4),
            (40, 39, 20),
            (320, 24, 160),
            (997, 4, 0),
            (2880, 24, 120),
        ];
        for _ in 0..40 {
            let n = 2 + g.below(700);
            let lag = g.below(n.min(70));
            sizes.push((n, lag, g.below(n / 2 + 1).min(160)));
        }
        for (n, lag, overlap) in sizes {
            let x = fill(&mut g, n, kind);
            for windowed in [false, true] {
                let window = windowed.then_some(&win[..overlap]);
                let want = ref_autocorr(&x, window, overlap, lag);
                for at_end in [true, false] {
                    let x = rx.place(&x, at_end);
                    let ac = rac.place(&vec![f32::NAN; lag + 1], at_end);
                    let window: Option<&[f32]> = if windowed {
                        Some(rwin.place(&win[..overlap], at_end))
                    } else {
                        None
                    };
                    autocorr(x, ac, window, overlap, lag, n);
                    assert_same(
                        ac,
                        &want,
                        &format!(
                            "{kind:?} n {n} lag {lag} overlap {overlap} window {windowed} at_end {at_end}"
                        ),
                    );
                }
            }
        }
    }
}

// The kernels read through pointers. A slice that is shorter than the length
// passed with it must stop the call, not be read past its end.

#[test]
#[should_panic]
fn inner_prod_with_a_short_x_panics() {
    inner_prod(&[1.0; 7], &[1.0; 16], 16);
}

#[test]
#[should_panic]
fn inner_prod_with_a_short_y_panics() {
    inner_prod(&[1.0; 16], &[1.0; 15], 16);
}

#[test]
#[should_panic]
fn dual_inner_prod_with_a_short_second_signal_panics() {
    dual_inner_prod(&[1.0; 16], &[1.0; 16], &[1.0; 3], 16);
}

#[test]
#[should_panic]
fn pitch_xcorr_with_a_short_x_panics() {
    let mut out = [0.0f32; 40];
    pitch_xcorr(&[1.0; 8], &[1.0; 200], &mut out, 64, 40);
}
