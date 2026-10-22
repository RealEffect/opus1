#[cfg(not(feature = "std"))]
use crate::compat::Math;
use crate::fixedvec::FixedVec;

pub const MAXFACTORS: usize = 8;

/// Maximum FFT size (`nfft`) used by the MDCT plans. Bounded by `MAX_N4 = 480`.
pub const KISS_MAX_N: usize = 480;

/// Room for the twiddle tables of all stages of one plan, counted in `f32`.
///
/// A stage of radix `p` with `m` butterflies per group stores `(p - 1) * m`
/// complex twiddles. `(p - 1) * m` is the group length of this stage minus the
/// group length of the next one, so the sum over all stages is `nfft - 1`.
const STAGE_TW_MAX: usize = 2 * KISS_MAX_N;

#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
pub struct KissCpx {
    pub r: f32,
    pub i: f32,
}

impl KissCpx {
    #[inline(always)]
    pub const fn new(r: f32, i: f32) -> Self {
        Self { r, i }
    }
}

// SAFETY: zero bytes are the number 0 + 0i (two `f32` zeros, no padding),
// and the type has no destructor.
unsafe impl crate::in_place::Zeroable for KissCpx {}

/// How the portable code computes `a * b + c`.
///
/// libopus writes such an expression in C. A compiler may round the product
/// and then add (two operations), or compute product and sum with one
/// rounding (fused multiply-add). The two give different last bits, so the
/// portable code exists in both forms.
///
/// The expressions in this file and in `mdct.rs` are written the way clang
/// reads the C source with its default setting `-ffp-contract=on`: a product
/// is fused with a sum only when both are in one C expression, and of two
/// products it is the one on the left; the other one is rounded first.
/// `a * b - c * d` is therefore `mul_add(a, b, -(c * d))`.
pub(crate) trait Arith {
    fn mul_add(a: f32, b: f32, c: f32) -> f32;
}

/// The product is rounded, then the sum. A build of libopus without fused
/// multiply-add computes this, for example one for x86-64.
pub(crate) struct Separate;

/// Product and sum are rounded once. clang builds libopus for arm64 this
/// way.
pub(crate) struct Fused;

impl Arith for Separate {
    #[inline(always)]
    fn mul_add(a: f32, b: f32, c: f32) -> f32 {
        a * b + c
    }
}

impl Arith for Fused {
    #[inline(always)]
    fn mul_add(a: f32, b: f32, c: f32) -> f32 {
        a.mul_add(b, c)
    }
}

/// `a * b` as libopus writes it (`C_MUL`, celt/_kiss_fft_guts.h:112-114).
#[inline(always)]
fn c_mul<A: Arith>(a: &KissCpx, b: &KissCpx) -> KissCpx {
    KissCpx {
        r: A::mul_add(a.r, b.r, -(a.i * b.i)),
        i: A::mul_add(a.r, b.i, a.i * b.r),
    }
}

#[inline(always)]
fn c_sub(a: &KissCpx, b: &KissCpx) -> KissCpx {
    KissCpx {
        r: a.r - b.r,
        i: a.i - b.i,
    }
}

#[inline(always)]
fn c_add(a: &KissCpx, b: &KissCpx) -> KissCpx {
    KissCpx {
        r: a.r + b.r,
        i: a.i + b.i,
    }
}

/// One butterfly pass over the whole buffer.
///
/// The pass works on `groups` consecutive groups of `radix * m` samples.
/// `groups * radix * m` equals the FFT size for every stage.
#[derive(Clone, Copy, Debug, Default)]
struct FftStage {
    /// Radix of the butterflies: 2, 3, 4 or 5.
    radix: u8,
    /// Butterflies per group.
    m: u16,
    /// Number of groups.
    groups: u16,
    /// Start of this stage's twiddles in `KissFftState::stage_tw`.
    tw: u16,
}

impl FftStage {
    /// Number of `f32` twiddle values this stage reads.
    ///
    /// Radix 2 and radix 4 with `m == 1` use fixed factors and read nothing.
    #[inline(always)]
    fn tw_len(&self) -> usize {
        match (self.radix, self.m) {
            (3 | 5, m) => 2 * (self.radix as usize - 1) * m as usize,
            (4, m) if m > 1 => 6 * m as usize,
            _ => 0,
        }
    }
}

pub struct KissFftState {
    nfft: usize,
    scale: f32,
    /// The butterfly passes in the order `opus_fft_impl` runs them.
    stages: [FftStage; MAXFACTORS],
    nstages: usize,
    /// Where each input sample goes before the first stage: sample `i` is
    /// stored at place `bitrev[i]`. A plan that `new` or `new_sub` returned
    /// holds a permutation of `0..nfft` here.
    pub bitrev: FixedVec<i16, KISS_MAX_N>,
    /// Twiddle factors, one contiguous table per stage.
    ///
    /// A stage of radix `p` with `m` butterflies per group stores, for each
    /// `k` in `1..p`, the `m` values `w^(k * j)`, `j = 0..m`. `w` is the
    /// root of unity of that stage. The values are copies of entries of the
    /// table of the largest FFT size, which are a stride apart there.
    /// `StreamTw` describes how the `m` values are laid out.
    stage_tw: FixedVec<f32, STAGE_TW_MAX>,
}

fn kf_factor(n_orig: usize, factors: &mut [i16; 2 * MAXFACTORS]) -> bool {
    // The twiddle and bit-reversal tables hold KISS_MAX_N entries. A larger
    // size is rejected here, so that `KissFftState::new` returns None.
    if n_orig > KISS_MAX_N {
        return false;
    }
    let mut n = n_orig;
    let mut p: i32 = 4;
    let mut stages = 0;

    loop {
        while !n.is_multiple_of(p as usize) {
            p = match p {
                4 => 2,
                2 => 3,
                _ => p + 2,
            };
            if p > 32000 || (p as i64) * (p as i64) > n as i64 {
                p = n as i32;
            }
        }
        n /= p as usize;

        if p > 5 {
            return false;
        }

        factors[2 * stages] = p as i16;

        if p == 2 && stages > 1 {
            factors[2 * stages] = 4;
            factors[2] = 2;
        }
        stages += 1;

        if n <= 1 {
            break;
        }
    }

    for i in 0..(stages / 2) {
        factors.swap(2 * i, 2 * (stages - i - 1));
    }

    n = n_orig;
    for i in 0..stages {
        n /= factors[2 * i] as usize;
        factors[2 * i + 1] = n as i16;
    }

    true
}

fn compute_bitrev_table(
    fout: i32,
    f: &mut [i16],
    fstride: usize,
    in_stride: usize,
    factors: &[i16],
) {
    let p = factors[0] as i32;
    let m = factors[1] as i32;

    if m == 1 {
        for j in 0..p {
            let idx = (j as usize) * fstride * in_stride;
            f[idx] = (fout + j) as i16;
        }
    } else {
        let mut fout = fout;
        let mut f_offset = 0usize;
        for _ in 0..p {
            compute_bitrev_table(
                fout,
                &mut f[f_offset..],
                fstride * (p as usize),
                in_stride,
                &factors[2..],
            );
            f_offset += fstride * in_stride;
            fout += m;
        }
    }
}

/// True when `table` holds every number of `0..table.len()` exactly once.
fn is_permutation(table: &[i16]) -> bool {
    let n = table.len();
    if n > KISS_MAX_N {
        return false;
    }
    // One bit per number.
    let mut seen = [0u64; KISS_MAX_N.div_ceil(64)];
    for &v in table {
        // A negative entry becomes a number above `n` here.
        let v = v as usize;
        if v >= n {
            return false;
        }
        let bit = 1u64 << (v % 64);
        if seen[v / 64] & bit != 0 {
            return false;
        }
        seen[v / 64] |= bit;
    }
    true
}

/// The twiddle table of an FFT of `nfft` points: entry `i` is
/// `exp(-2 * pi * i / nfft)`.
///
/// For 480 points it is the table libopus has for its 48 kHz mode. For any
/// other size it is computed like libopus computes the table of a custom
/// mode (`compute_twiddles`, celt/kiss_fft.c:427-431): phase, cosine and sine
/// in double precision, then rounded to `f32`.
fn compute_twiddles(nfft: usize) -> FixedVec<KissCpx, KISS_MAX_N> {
    let mut out: FixedVec<KissCpx, KISS_MAX_N> = FixedVec::new();
    if nfft == KISS_MAX_N {
        for &t in FFT_TWIDDLES_480.iter() {
            out.push(t);
        }
        return out;
    }
    // The constant of libopus, 3.14159265358979323846264338327, is this
    // number in double precision.
    const PI: f64 = core::f64::consts::PI;
    for i in 0..nfft {
        let phase = (-2.0 * PI / nfft as f64) * i as f64;
        out.push(KissCpx::new(phase.cos() as f32, phase.sin() as f32));
    }
    out
}

impl KissFftState {
    pub fn new(nfft: usize) -> Option<Self> {
        if nfft == 0 || nfft > KISS_MAX_N {
            return None;
        }
        let twiddles = compute_twiddles(nfft);
        Self::build(nfft, &twiddles, 0)
    }

    pub fn new_sub(base: &KissFftState, nfft: usize) -> Option<Self> {
        if nfft == 0 || nfft > KISS_MAX_N {
            return None;
        }
        let mut shift = 0usize;
        while shift < 32 && (nfft << shift) != base.nfft {
            shift += 1;
        }
        if shift >= 32 {
            return None;
        }
        // The stage tables hold copies of entries of the base size's table.
        // That table is computed again here, because the base plan keeps only
        // its own stage tables.
        let twiddles = compute_twiddles(base.nfft);
        Self::build(nfft, &twiddles, shift)
    }

    /// The plan for `twiddles.len()` points, with a given twiddle table
    /// instead of one computed with `cos` and `sin`. Entry `i` of the table
    /// must be `exp(-2 * pi * i / n)`, `n = twiddles.len()`.
    ///
    /// libopus has such fixed tables, for example `kfft` of
    /// `dnn/lpcnet_tables.c` (320 points). The factors and the bit-reversal
    /// table are computed as for `new`; they are the ones of libopus.
    #[cfg(feature = "dnn")]
    pub(crate) fn with_twiddles(twiddles: &[KissCpx]) -> Option<Self> {
        let nfft = twiddles.len();
        if nfft == 0 || nfft > KISS_MAX_N {
            return None;
        }
        Self::build(nfft, twiddles, 0)
    }

    /// Builds the plan for `nfft` points.
    ///
    /// `twiddles` is the table of the FFT size `nfft << shift`. Entry `i` of
    /// it is `exp(-2 * pi * i / (nfft << shift))`.
    fn build(nfft: usize, twiddles: &[KissCpx], shift: usize) -> Option<Self> {
        let mut factors = [0i16; 2 * MAXFACTORS];
        if !kf_factor(nfft, &mut factors) {
            return None;
        }

        let mut bitrev = FixedVec::from_value(0i16, nfft);
        compute_bitrev_table(0, &mut bitrev, 1, 1, &factors);
        // The MDCT stores value `i` at place `bitrev[i]` of a buffer that it
        // does not initialise first. Every place must get a value, so a plan
        // exists only when the table is a permutation of `0..nfft`.
        if !is_permutation(&bitrev) {
            return None;
        }

        // `fstride[i]` is the product of the radices before factor `i`. It is
        // the number of groups of that stage, and the step through `twiddles`.
        let mut fstride = [0usize; MAXFACTORS + 1];
        fstride[0] = 1;
        let mut l = 0;
        loop {
            let p = factors[2 * l] as usize;
            let m = factors[2 * l + 1] as usize;
            fstride[l + 1] = fstride[l] * p;
            l += 1;
            if m == 1 {
                break;
            }
        }

        let mut stages = [FftStage::default(); MAXFACTORS];
        let mut stage_tw: FixedVec<f32, STAGE_TW_MAX> = FixedVec::new();
        // The transform runs the factors from the last one to the first one.
        for (stage, i) in stages.iter_mut().zip((0..l).rev()) {
            let p = factors[2 * i] as usize;
            let m = factors[2 * i + 1] as usize;
            let groups = fstride[i];
            if groups * p * m != nfft {
                return None;
            }
            // The radix-2 butterfly exists for m == 1 and m == 4 only, as in
            // libopus. `kf_factor` never asks for another one.
            if p == 2 && m != 1 && m != 4 {
                return None;
            }
            *stage = FftStage {
                radix: p as u8,
                m: m as u16,
                groups: groups as u16,
                tw: stage_tw.len() as u16,
            };
            if stage.tw_len() == 0 {
                continue;
            }
            let step = groups << shift;
            for k in 1..p {
                // The layout that `StreamTw` reads.
                #[cfg(target_arch = "aarch64")]
                for j in 0..m {
                    stage_tw.push(twiddles[k * j * step].r);
                    stage_tw.push(twiddles[k * j * step].i);
                }
                #[cfg(not(target_arch = "aarch64"))]
                {
                    for j in 0..m {
                        stage_tw.push(twiddles[k * j * step].r);
                    }
                    for j in 0..m {
                        stage_tw.push(twiddles[k * j * step].i);
                    }
                }
            }
        }

        Some(Self {
            nfft,
            scale: 1.0 / nfft as f32,
            stages,
            nstages: l,
            bitrev,
            stage_tw,
        })
    }

    #[inline]
    pub fn nfft(&self) -> usize {
        self.nfft
    }

    #[inline]
    pub fn scale(&self) -> f32 {
        self.scale
    }
}

// ---------------------------------------------------------------------------
// Portable butterflies.
//
// They are the loops of libopus (kiss_fft.c, float build). `A` says whether a
// product is fused with the sum it is written next to:
// - `Separate` is the code of every target without a vector version.
// - `Fused` is what the vector versions for aarch64 compute. They are tested
//   against it and must give the same bits.
// ---------------------------------------------------------------------------

// libopus reads these constants from its twiddle table (kiss_fft.c:201,
// 261-262): the imaginary part of `exp(-2 * pi * i / 3)`, then
// `exp(-2 * pi * i / 5)` and `exp(-4 * pi * i / 5)`. They are entries 160, 96
// and 192 of `FFT_TWIDDLES_480`; a unit test compares them.
const EPI3_I: f32 = -0.866_025_4;
const YA_R: f32 = 0.309_017;
const YA_I: f32 = -0.95105652;
const YB_R: f32 = -0.809_017;
const YB_I: f32 = -0.58778525;

/// The twiddles of one input stream of a stage: `m` complex values.
///
/// The stage table stores them in the layout that the kernels of the target
/// read fastest: pairs of real and imaginary part on aarch64, the `m` real
/// parts followed by the `m` imaginary parts elsewhere.
#[derive(Clone, Copy)]
struct StreamTw<'a> {
    #[cfg(target_arch = "aarch64")]
    pairs: &'a [[f32; 2]],
    #[cfg(not(target_arch = "aarch64"))]
    re: &'a [f32],
    #[cfg(not(target_arch = "aarch64"))]
    im: &'a [f32],
}

impl<'a> StreamTw<'a> {
    /// Stream `k` of a stage table with `m` butterflies per group. `k` is 0
    /// for the first stream that has twiddles.
    #[inline(always)]
    fn new(tw: &'a [f32], m: usize, k: usize) -> Self {
        let stream = &tw[2 * k * m..][..2 * m];
        #[cfg(target_arch = "aarch64")]
        {
            Self {
                pairs: stream.as_chunks::<2>().0,
            }
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            let (re, im) = stream.split_at(m);
            Self { re, im }
        }
    }

    /// Twiddle `j`, with `j < m`.
    #[inline(always)]
    fn get(&self, j: usize) -> KissCpx {
        #[cfg(target_arch = "aarch64")]
        {
            KissCpx::new(self.pairs[j][0], self.pairs[j][1])
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            KissCpx::new(self.re[j], self.im[j])
        }
    }
}

/// One radix-2 group with `m == 4`: eight samples, fixed twiddles.
///
/// No product stands next to a sum in the C source (kiss_fft.c:86-102), so
/// nothing is fused in any build.
#[inline(always)]
fn bfly2_m4_group(g: &mut [KissCpx; 8]) {
    let tw: f32 = core::f32::consts::FRAC_1_SQRT_2;

    let t = g[4];
    g[4] = c_sub(&g[0], &t);
    g[0] = c_add(&g[0], &t);

    let t = KissCpx::new((g[5].r + g[5].i) * tw, (g[5].i - g[5].r) * tw);
    g[5] = c_sub(&g[1], &t);
    g[1] = c_add(&g[1], &t);

    let t = KissCpx::new(g[6].i, -g[6].r);
    g[6] = c_sub(&g[2], &t);
    g[2] = c_add(&g[2], &t);

    let t = KissCpx::new((g[7].i - g[7].r) * tw, -(g[7].i + g[7].r) * tw);
    g[7] = c_sub(&g[3], &t);
    g[3] = c_add(&g[3], &t);
}

/// One radix-4 group with `m == 1`: four samples, no twiddles.
#[inline(always)]
fn bfly4_m1_group(g: &mut [KissCpx; 4]) {
    let scratch0 = c_sub(&g[0], &g[2]);
    let sum02 = c_add(&g[0], &g[2]);
    let scratch1 = c_add(&g[1], &g[3]);
    let diff13 = c_sub(&g[1], &g[3]);

    g[0] = c_add(&sum02, &scratch1);
    g[2] = c_sub(&sum02, &scratch1);
    g[1] = KissCpx::new(scratch0.r + diff13.i, scratch0.i - diff13.r);
    g[3] = KissCpx::new(scratch0.r - diff13.i, scratch0.i + diff13.r);
}

/// One radix-3 butterfly (kiss_fft.c:211-230). `w1` and `w2` are the
/// twiddles of `f1` and `f2`.
#[inline(always)]
fn bfly3_one<A: Arith>(
    f0: &mut KissCpx,
    f1: &mut KissCpx,
    f2: &mut KissCpx,
    w1: KissCpx,
    w2: KissCpx,
) {
    let scratch1 = c_mul::<A>(f1, &w1);
    let scratch2 = c_mul::<A>(f2, &w2);

    let scratch3 = c_add(&scratch1, &scratch2);
    let scratch0 = c_sub(&scratch1, &scratch2);

    // Fout[m] = Fout - HALF_OF(scratch[3]): the half is a product.
    let fout_m = KissCpx::new(
        A::mul_add(-scratch3.r, 0.5, f0.r),
        A::mul_add(-scratch3.i, 0.5, f0.i),
    );
    let scratch0_scaled = KissCpx::new(scratch0.r * EPI3_I, scratch0.i * EPI3_I);

    *f0 = c_add(f0, &scratch3);
    *f1 = KissCpx::new(fout_m.r - scratch0_scaled.i, fout_m.i + scratch0_scaled.r);
    *f2 = KissCpx::new(fout_m.r + scratch0_scaled.i, fout_m.i - scratch0_scaled.r);
}

/// One radix-4 butterfly (kiss_fft.c:153-170). `w1`, `w2`, `w3` are the
/// twiddles of `f1`, `f2`, `f3`.
#[inline(always)]
fn bfly4_one<A: Arith>(
    f0: &mut KissCpx,
    f1: &mut KissCpx,
    f2: &mut KissCpx,
    f3: &mut KissCpx,
    w1: KissCpx,
    w2: KissCpx,
    w3: KissCpx,
) {
    let scratch0 = c_mul::<A>(f1, &w1);
    let scratch1 = c_mul::<A>(f2, &w2);
    let scratch2 = c_mul::<A>(f3, &w3);

    let scratch5 = c_sub(f0, &scratch1);
    let sum = c_add(f0, &scratch1);

    let scratch3 = c_add(&scratch0, &scratch2);
    let scratch4 = c_sub(&scratch0, &scratch2);

    *f2 = c_sub(&sum, &scratch3);
    *f0 = c_add(&sum, &scratch3);

    *f1 = KissCpx::new(scratch5.r + scratch4.i, scratch5.i - scratch4.r);
    *f3 = KissCpx::new(scratch5.r - scratch4.i, scratch5.i + scratch4.r);
}

/// One radix-5 butterfly (kiss_fft.c:277-307). `w[k - 1]` is the twiddle of
/// `f[k]`.
#[inline(always)]
fn bfly5_one<A: Arith>(
    f0: &mut KissCpx,
    f1: &mut KissCpx,
    f2: &mut KissCpx,
    f3: &mut KissCpx,
    f4: &mut KissCpx,
    w: [KissCpx; 4],
) {
    let scratch0 = *f0;

    let scratch1 = c_mul::<A>(f1, &w[0]);
    let scratch2 = c_mul::<A>(f2, &w[1]);
    let scratch3 = c_mul::<A>(f3, &w[2]);
    let scratch4 = c_mul::<A>(f4, &w[3]);

    let scratch7 = c_add(&scratch1, &scratch4);
    let scratch10 = c_sub(&scratch1, &scratch4);
    let scratch8 = c_add(&scratch2, &scratch3);
    let scratch9 = c_sub(&scratch2, &scratch3);

    // libopus adds the two products first and the sample last
    // (kiss_fft.c:289-293, 301-302). The order decides the rounding.
    *f0 = KissCpx::new(
        scratch0.r + (scratch7.r + scratch8.r),
        scratch0.i + (scratch7.i + scratch8.i),
    );

    let scratch5 = KissCpx::new(
        scratch0.r + A::mul_add(scratch7.r, YA_R, scratch8.r * YB_R),
        scratch0.i + A::mul_add(scratch7.i, YA_R, scratch8.i * YB_R),
    );

    let scratch6 = KissCpx::new(
        A::mul_add(scratch10.i, YA_I, scratch9.i * YB_I),
        -A::mul_add(scratch10.r, YA_I, scratch9.r * YB_I),
    );

    *f1 = c_sub(&scratch5, &scratch6);
    *f4 = c_add(&scratch5, &scratch6);

    let scratch11 = KissCpx::new(
        scratch0.r + A::mul_add(scratch7.r, YB_R, scratch8.r * YA_R),
        scratch0.i + A::mul_add(scratch7.i, YB_R, scratch8.i * YA_R),
    );

    let scratch12 = KissCpx::new(
        A::mul_add(scratch9.i, YA_I, -(scratch10.i * YB_I)),
        A::mul_add(scratch10.r, YB_I, -(scratch9.r * YA_I)),
    );

    *f2 = c_add(&scratch11, &scratch12);
    *f3 = c_sub(&scratch11, &scratch12);
}

/// Radix-2 pass. `m` is 1 or 4.
#[inline(always)]
fn kf_bfly2_scalar(fout: &mut [KissCpx], m: usize, groups: usize) {
    if m == 1 {
        for g in fout.as_chunks_mut::<2>().0.iter_mut().take(groups) {
            let t = g[1];
            g[1] = c_sub(&g[0], &t);
            g[0] = c_add(&g[0], &t);
        }
    } else {
        for g in fout.as_chunks_mut::<8>().0.iter_mut().take(groups) {
            bfly2_m4_group(g);
        }
    }
}

/// Radix-4 pass with one butterfly per group.
#[inline(always)]
fn kf_bfly4_m1_scalar(fout: &mut [KissCpx], groups: usize) {
    for g in fout.as_chunks_mut::<4>().0.iter_mut().take(groups) {
        bfly4_m1_group(g);
    }
}

/// Radix-3 pass. `tw` is the stage table: `2 * 2 * m` values.
#[inline(always)]
fn kf_bfly3_scalar<A: Arith>(fout: &mut [KissCpx], tw: &[f32], m: usize, groups: usize) {
    if m == 0 {
        return;
    }
    let w1 = StreamTw::new(tw, m, 0);
    let w2 = StreamTw::new(tw, m, 1);

    for group in fout.chunks_exact_mut(3 * m).take(groups) {
        let (f0, rest) = group.split_at_mut(m);
        let (f1, f2) = rest.split_at_mut(m);
        let f2 = &mut f2[..m];
        for j in 0..m {
            bfly3_one::<A>(&mut f0[j], &mut f1[j], &mut f2[j], w1.get(j), w2.get(j));
        }
    }
}

/// Radix-4 pass with `m > 1`. `tw` is the stage table: `2 * 3 * m` values.
#[inline(always)]
fn kf_bfly4_scalar<A: Arith>(fout: &mut [KissCpx], tw: &[f32], m: usize, groups: usize) {
    if m == 0 {
        return;
    }
    let w1 = StreamTw::new(tw, m, 0);
    let w2 = StreamTw::new(tw, m, 1);
    let w3 = StreamTw::new(tw, m, 2);

    for group in fout.chunks_exact_mut(4 * m).take(groups) {
        let (f0, rest) = group.split_at_mut(m);
        let (f1, rest) = rest.split_at_mut(m);
        let (f2, f3) = rest.split_at_mut(m);
        let f3 = &mut f3[..m];
        for j in 0..m {
            bfly4_one::<A>(
                &mut f0[j],
                &mut f1[j],
                &mut f2[j],
                &mut f3[j],
                w1.get(j),
                w2.get(j),
                w3.get(j),
            );
        }
    }
}

/// Radix-5 pass. `tw` is the stage table: `2 * 4 * m` values.
#[inline(always)]
fn kf_bfly5_scalar<A: Arith>(fout: &mut [KissCpx], tw: &[f32], m: usize, groups: usize) {
    if m == 0 {
        return;
    }
    let w1 = StreamTw::new(tw, m, 0);
    let w2 = StreamTw::new(tw, m, 1);
    let w3 = StreamTw::new(tw, m, 2);
    let w4 = StreamTw::new(tw, m, 3);

    for group in fout.chunks_exact_mut(5 * m).take(groups) {
        let (f0, rest) = group.split_at_mut(m);
        let (f1, rest) = rest.split_at_mut(m);
        let (f2, rest) = rest.split_at_mut(m);
        let (f3, f4) = rest.split_at_mut(m);
        let f4 = &mut f4[..m];
        for j in 0..m {
            bfly5_one::<A>(
                &mut f0[j],
                &mut f1[j],
                &mut f2[j],
                &mut f3[j],
                &mut f4[j],
                [w1.get(j), w2.get(j), w3.get(j), w4.get(j)],
            );
        }
    }
}

/// Runs all stages with the portable butterflies.
#[inline(always)]
fn fft_stages_scalar<A: Arith>(st: &KissFftState, fout: &mut [KissCpx]) {
    for s in &st.stages[..st.nstages] {
        let m = s.m as usize;
        let groups = s.groups as usize;
        let tw = &st.stage_tw[s.tw as usize..s.tw as usize + s.tw_len()];
        match (s.radix, m) {
            (2, _) => kf_bfly2_scalar(fout, m, groups),
            (4, 1) => kf_bfly4_m1_scalar(fout, groups),
            (4, _) => kf_bfly4_scalar::<A>(fout, tw, m, groups),
            (3, _) => kf_bfly3_scalar::<A>(fout, tw, m, groups),
            (5, _) => kf_bfly5_scalar::<A>(fout, tw, m, groups),
            // A one-point transform has a single stage of radix 1. It
            // leaves the sample as it is.
            _ => {}
        }
    }
}

/// The portable stages, compiled with AVX enabled.
///
/// The code is the portable one; AVX only changes how the compiler encodes
/// and vectorizes it. The results are the same bits.
///
/// # Safety
/// The processor must support AVX.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx")]
unsafe fn fft_stages_avx(st: &KissFftState, fout: &mut [KissCpx]) {
    fft_stages_scalar::<Separate>(st, fout);
}

// ---------------------------------------------------------------------------
// NEON butterflies without twiddle tables (aarch64).
//
// These three passes have no product next to a sum, so no build of libopus
// fuses anything in them. Each one gives the same bits as its portable
// version: every lane does the same multiplies, additions and subtractions
// in the same order.
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
mod neon {
    use super::{KissCpx, c_add, c_sub};
    use core::arch::aarch64::*;

    /// The first sample of each of two groups: `[a.lo, b.lo]`.
    #[inline(always)]
    unsafe fn lo2(a: float32x4_t, b: float32x4_t) -> float32x4_t {
        vcombine_f32(vget_low_f32(a), vget_low_f32(b))
    }

    /// The second sample of each of two groups: `[a.hi, b.hi]`.
    #[inline(always)]
    unsafe fn hi2(a: float32x4_t, b: float32x4_t) -> float32x4_t {
        vcombine_f32(vget_high_f32(a), vget_high_f32(b))
    }

    /// Radix-2 pass with one butterfly per group.
    ///
    /// # Safety
    /// `fout` must hold at least `2 * groups` samples.
    pub(super) unsafe fn bfly2_m1(fout: &mut [KissCpx], groups: usize) {
        debug_assert!(2 * groups <= fout.len());
        let n = groups;
        let ptr = fout.as_mut_ptr() as *mut f32;

        // SAFETY: a group is 4 floats. The first loop touches the floats of
        // groups `i..i + 4` with `i + 4 <= n`, the second loop those of
        // groups `i..i + 2` with `i + 2 <= n`.
        let mut i = 0usize;

        while i + 4 <= n {
            let base = i * 4;
            let v0 = vld1q_f32(ptr.add(base));
            let v1 = vld1q_f32(ptr.add(base + 4));
            let v2 = vld1q_f32(ptr.add(base + 8));
            let v3 = vld1q_f32(ptr.add(base + 12));

            let r0 = vcombine_f32(
                vadd_f32(vget_low_f32(v0), vget_high_f32(v0)),
                vsub_f32(vget_low_f32(v0), vget_high_f32(v0)),
            );
            let r1 = vcombine_f32(
                vadd_f32(vget_low_f32(v1), vget_high_f32(v1)),
                vsub_f32(vget_low_f32(v1), vget_high_f32(v1)),
            );
            let r2 = vcombine_f32(
                vadd_f32(vget_low_f32(v2), vget_high_f32(v2)),
                vsub_f32(vget_low_f32(v2), vget_high_f32(v2)),
            );
            let r3 = vcombine_f32(
                vadd_f32(vget_low_f32(v3), vget_high_f32(v3)),
                vsub_f32(vget_low_f32(v3), vget_high_f32(v3)),
            );

            vst1q_f32(ptr.add(base), r0);
            vst1q_f32(ptr.add(base + 4), r1);
            vst1q_f32(ptr.add(base + 8), r2);
            vst1q_f32(ptr.add(base + 12), r3);

            i += 4;
        }

        while i + 2 <= n {
            let base = i * 4;
            let v0 = vld1q_f32(ptr.add(base));
            let v1 = vld1q_f32(ptr.add(base + 4));
            let r0 = vcombine_f32(
                vadd_f32(vget_low_f32(v0), vget_high_f32(v0)),
                vsub_f32(vget_low_f32(v0), vget_high_f32(v0)),
            );
            let r1 = vcombine_f32(
                vadd_f32(vget_low_f32(v1), vget_high_f32(v1)),
                vsub_f32(vget_low_f32(v1), vget_high_f32(v1)),
            );
            vst1q_f32(ptr.add(base), r0);
            vst1q_f32(ptr.add(base + 4), r1);
            i += 2;
        }

        while i < n {
            let idx = i * 2;
            let t = fout[idx + 1];
            fout[idx + 1] = c_sub(&fout[idx], &t);
            fout[idx] = c_add(&fout[idx], &t);
            i += 1;
        }
    }

    /// Radix-2 pass with four butterflies per group (fixed twiddles).
    ///
    /// # Safety
    /// `fout` must hold at least `8 * groups` samples.
    pub(super) unsafe fn bfly2_m4(fout: &mut [KissCpx], groups: usize) {
        debug_assert!(8 * groups <= fout.len());
        const SIGN: u32 = 0x8000_0000;
        const TW: f32 = core::f32::consts::FRAC_1_SQRT_2;
        // Lanes 0 and 1 hold a butterfly with twiddle 1 or -i: it has no
        // product. Multiplying by 1.0 leaves every value unchanged.
        let scale = vld1q_f32([1.0, 1.0, TW, TW].as_ptr());
        let hi_mask = vld1q_u32([0, 0, u32::MAX, u32::MAX].as_ptr());
        let sign_k0 = vld1q_u32([SIGN, SIGN, 0, SIGN].as_ptr());
        let sign_k1 = vld1q_u32([SIGN, SIGN, SIGN, 0].as_ptr());
        let sign_odd = vld1q_u32([0, SIGN, 0, SIGN].as_ptr());
        let ptr = fout.as_mut_ptr() as *mut f32;

        // One group: 16 floats at `p`.
        // SAFETY (for every call below): `p` points to a whole group.
        let group = |p: *mut f32| {
            let a0 = vld1q_f32(p); // f0 f1
            let a1 = vld1q_f32(p.add(4)); // f2 f3
            let b0 = vld1q_f32(p.add(8)); // f4 f5
            let b1 = vld1q_f32(p.add(12)); // f6 f7

            // t0 = [f4.r, f4.i, (f5.r + f5.i) * tw, (f5.i - f5.r) * tw].
            // k0 = [-0.0, -0.0, f5.i, -f5.r]; adding -0.0 changes nothing.
            let r0 = vrev64q_f32(b0);
            let k0 = vreinterpretq_f32_u32(veorq_u32(
                vandq_u32(vreinterpretq_u32_f32(r0), hi_mask),
                sign_k0,
            ));
            let t0 = vmulq_f32(vaddq_f32(b0, k0), scale);

            // t1 = [f6.i, -f6.r, (f7.i - f7.r) * tw, -(f7.i + f7.r) * tw].
            // k1 = [-0.0, -0.0, -f7.r, f7.i].
            let r1 = vrev64q_f32(b1);
            let k1 = vreinterpretq_f32_u32(veorq_u32(
                vandq_u32(vreinterpretq_u32_f32(b1), hi_mask),
                sign_k1,
            ));
            let y = vaddq_f32(r1, k1);
            let y = vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(y), sign_odd));
            let t1 = vmulq_f32(y, scale);

            vst1q_f32(p.add(8), vsubq_f32(a0, t0));
            vst1q_f32(p, vaddq_f32(a0, t0));
            vst1q_f32(p.add(12), vsubq_f32(a1, t1));
            vst1q_f32(p.add(4), vaddq_f32(a1, t1));
        };
        // Two groups per step, as in `bfly4_m1`.
        // SAFETY: a group is 16 floats and `g + 1 < groups`, then
        // `g < groups`.
        let mut g = 0;
        while g + 2 <= groups {
            let p = ptr.add(16 * g);
            group(p);
            group(p.add(16));
            g += 2;
        }
        if g < groups {
            group(ptr.add(16 * g));
        }
    }

    /// Radix-4 pass with one butterfly per group.
    ///
    /// # Safety
    /// `fout` must hold at least `4 * groups` samples.
    pub(super) unsafe fn bfly4_m1(fout: &mut [KissCpx], groups: usize) {
        debug_assert!(4 * groups <= fout.len());
        const SIGN: u32 = 0x8000_0000;
        let sign_odd = vld1q_u32([0, SIGN, 0, SIGN].as_ptr());
        let ptr = fout.as_mut_ptr() as *mut f32;

        // One group: 8 floats at `p`.
        // SAFETY (for every call below): `p` points to a whole group.
        let group = |p: *mut f32| {
            let v0 = vld1q_f32(p); // f0 f1
            let v1 = vld1q_f32(p.add(4)); // f2 f3

            // sum = [sum02, scratch1], diff = [scratch0, diff13]
            let sum = vaddq_f32(v0, v1);
            let diff = vsubq_f32(v0, v1);
            // The high half of `rot` is [diff13.i, -diff13.r]. Adding -x is
            // the same as subtracting x, bit for bit.
            let rot = vreinterpretq_f32_u32(veorq_u32(
                vreinterpretq_u32_f32(vrev64q_f32(diff)),
                sign_odd,
            ));
            let a = lo2(sum, diff); // sum02, scratch0
            let b = hi2(sum, rot); // scratch1, rotated diff13

            vst1q_f32(p, vaddq_f32(a, b));
            vst1q_f32(p.add(4), vsubq_f32(a, b));
        };
        // Two groups per step. They do not depend on each other, so a core
        // that runs the instructions in program order works on one while
        // the other waits for a result.
        // SAFETY: a group is 8 floats and `g + 1 < groups`, then `g < groups`.
        let mut g = 0;
        while g + 2 <= groups {
            let p = ptr.add(8 * g);
            group(p);
            group(p.add(8));
            g += 2;
        }
        if g < groups {
            group(ptr.add(8 * g));
        }
    }
}

// ---------------------------------------------------------------------------
// Butterflies with twiddles (aarch64).
//
// They compute what the portable butterflies compute with `Fused`, which is
// what the arm64 build of libopus computes. A vector holds two complex
// numbers as (re, im, re, im), the layout of the samples and of the stage
// tables, so nothing is shuffled between memory and the arithmetic.
//
// There are two versions with the same arithmetic and the same bits:
// - `fcma` uses the complex-number instructions of FEAT_FCMA (ARMv8.3):
//   `fcmla` for a complex product and `fcadd` for "a plus or minus i times
//   b". They are written with `core::arch::asm!`, because the intrinsics for
//   them are not stable.
// - `fma` uses plain NEON with `fmla`. It runs on processors without
//   FEAT_FCMA and needs more instructions.
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
mod fused {
    use super::{EPI3_I, KissCpx, KissFftState, YA_I, YA_R, YB_I, YB_R, neon};
    use core::arch::aarch64::*;
    use core::arch::asm;

    const SIGN: u32 = 0x8000_0000;

    /// `a * t` for two complex numbers, with `fcmla`.
    ///
    /// `fcmla #90` adds `-(a.im * t.im)` to the real part of the accumulator
    /// and `a.im * t.re` to its imaginary part. `fcmla #0` adds
    /// `a.re * t.re` and `a.re * t.im`. Every addition is fused.
    ///
    /// The accumulator starts at -0.0. Adding -0.0 changes no value, not even
    /// the sign of a zero, so the first step leaves the two rounded products
    /// and the second step fuses the other two onto them. That is `c_mul`
    /// with `Fused`.
    ///
    /// The block does not promise to keep the flags: the instructions set
    /// the cumulative exception bits of the floating-point status register.
    macro_rules! cmul_fcma {
        ($a:expr, $t:expr) => {{
            let a: float32x4_t = $a;
            let t: float32x4_t = $t;
            let mut acc = vdupq_n_f32(-0.0);
            asm!(
                "fcmla {acc:v}.4s, {a:v}.4s, {t:v}.4s, #90",
                "fcmla {acc:v}.4s, {a:v}.4s, {t:v}.4s, #0",
                acc = inout(vreg) acc,
                a = in(vreg) a,
                t = in(vreg) t,
                options(pure, nomem, nostack),
            );
            acc
        }};
    }

    /// `(a + i * b, a - i * b)` for two complex numbers, with `fcadd`.
    macro_rules! pm_i_fcma {
        ($a:expr, $b:expr) => {{
            let a: float32x4_t = $a;
            let b: float32x4_t = $b;
            let plus: float32x4_t;
            let minus: float32x4_t;
            asm!(
                "fcadd {plus:v}.4s, {a:v}.4s, {b:v}.4s, #90",
                "fcadd {minus:v}.4s, {a:v}.4s, {b:v}.4s, #270",
                // `plus` is written while `a` and `b` are still needed, so it
                // must not share a register with them. `minus` is written
                // last and may.
                plus = out(vreg) plus,
                minus = lateout(vreg) minus,
                a = in(vreg) a,
                b = in(vreg) b,
                options(pure, nomem, nostack),
            );
            (plus, minus)
        }};
    }

    /// `a * t` for two complex numbers, with `fmla`: the same four products
    /// and the same two roundings as `cmul_fcma`.
    macro_rules! cmul_fma {
        ($a:expr, $t:expr) => {{
            let a: float32x4_t = $a;
            let t: float32x4_t = $t;
            let a_re = vtrn1q_f32(a, a);
            let a_im = vtrn2q_f32(a, a);
            // i * t = (-t.im, t.re)
            let i_t = vreinterpretq_f32_u32(veorq_u32(
                vreinterpretq_u32_f32(vrev64q_f32(t)),
                vld1q_u32([SIGN, 0, SIGN, 0].as_ptr()),
            ));
            // (-(a.im * t.im), a.im * t.re) is rounded. Then a.re * t is
            // fused onto it.
            vfmaq_f32(vmulq_f32(a_im, i_t), a_re, t)
        }};
    }

    /// `(a + i * b, a - i * b)` for two complex numbers, with plain NEON.
    macro_rules! pm_i_fma {
        ($a:expr, $b:expr) => {{
            let a: float32x4_t = $a;
            let b_swapped = vreinterpretq_u32_f32(vrev64q_f32($b));
            // i * b = (-b.im, b.re) and -i * b = (b.im, -b.re). Adding -x
            // is the same as subtracting x, bit for bit.
            let i_b = veorq_u32(b_swapped, vld1q_u32([SIGN, 0, SIGN, 0].as_ptr()));
            let neg_i_b = veorq_u32(b_swapped, vld1q_u32([0, SIGN, 0, SIGN].as_ptr()));
            (
                vaddq_f32(a, vreinterpretq_f32_u32(i_b)),
                vaddq_f32(a, vreinterpretq_f32_u32(neg_i_b)),
            )
        }};
    }

    /// `(a.re, b.im)` for two complex numbers.
    #[inline(always)]
    unsafe fn re_of_a_im_of_b(a: float32x4_t, b: float32x4_t) -> float32x4_t {
        vbslq_f32(vld1q_u32([u32::MAX, 0, u32::MAX, 0].as_ptr()), a, b)
    }

    /// Two radix-3 butterflies. Returns the new `(f0, f1, f2)`.
    macro_rules! bfly3_step {
        ($cmul:ident, $pm_i:ident, $f0:expr, $f1:expr, $f2:expr, $w1:expr, $w2:expr) => {{
            let f0: float32x4_t = $f0;
            let s1 = $cmul!($f1, $w1);
            let s2 = $cmul!($f2, $w2);
            let s3 = vaddq_f32(s1, s2);
            let s0 = vsubq_f32(s1, s2);
            // fm = f0 - s3 * 0.5, fused.
            let fm = vfmsq_n_f32(f0, s3, 0.5);
            let s0 = vmulq_n_f32(s0, EPI3_I);
            // f1 = fm + i * s0 and f2 = fm - i * s0
            let (plus, minus) = $pm_i!(fm, s0);
            (vaddq_f32(f0, s3), plus, minus)
        }};
    }

    /// Two radix-4 butterflies. Returns the new `(f0, f1, f2, f3)`.
    macro_rules! bfly4_step {
        ($cmul:ident, $pm_i:ident, $f0:expr, $f1:expr, $f2:expr, $f3:expr,
         $w1:expr, $w2:expr, $w3:expr) => {{
            let f0: float32x4_t = $f0;
            let s0 = $cmul!($f1, $w1);
            let s1 = $cmul!($f2, $w2);
            let s2 = $cmul!($f3, $w3);
            let s5 = vsubq_f32(f0, s1);
            let sum = vaddq_f32(f0, s1);
            let s3 = vaddq_f32(s0, s2);
            let s4 = vsubq_f32(s0, s2);
            // f1 = s5 - i * s4 and f3 = s5 + i * s4
            let (plus, minus) = $pm_i!(s5, s4);
            (vaddq_f32(sum, s3), minus, vsubq_f32(sum, s3), plus)
        }};
    }

    /// Two radix-5 butterflies. Returns the new `(f0, f1, f2, f3, f4)`.
    ///
    /// The names are those of `bfly5_one`. Where that function has
    /// `mul_add(a, b, c * d)`, the product `c * d` is computed first with
    /// `fmul` and `a * b` is fused onto it with `fmla`.
    macro_rules! bfly5_step {
        ($cmul:ident, $pm_i:ident, $f0:expr, $f1:expr, $f2:expr, $f3:expr, $f4:expr,
         $w1:expr, $w2:expr, $w3:expr, $w4:expr) => {{
            let s0: float32x4_t = $f0;
            let s1 = $cmul!($f1, $w1);
            let s2 = $cmul!($f2, $w2);
            let s3 = $cmul!($f3, $w3);
            let s4 = $cmul!($f4, $w4);
            let s7 = vaddq_f32(s1, s4);
            let s10 = vsubq_f32(s1, s4);
            let s8 = vaddq_f32(s2, s3);
            let s9 = vsubq_f32(s2, s3);
            let o0 = vaddq_f32(s0, vaddq_f32(s7, s8));

            let s5 = vaddq_f32(s0, vfmaq_n_f32(vmulq_n_f32(s8, YB_R), s7, YA_R));
            // t = s10 * ya.i + s9 * yb.i. scratch6 is (t.im, -t.re), so
            // f1 = s5 - scratch6 = s5 + i * t and f4 = s5 - i * t.
            let t = vfmaq_n_f32(vmulq_n_f32(s9, YB_I), s10, YA_I);
            let (o1, o4) = $pm_i!(s5, t);

            let s11 = vaddq_f32(s0, vfmaq_n_f32(vmulq_n_f32(s8, YA_R), s7, YB_R));
            // scratch12 = (s9.im * ya.i - s10.im * yb.i,
            //              s10.re * yb.i - s9.re * ya.i).
            // Its two parts fuse different products, so they are computed
            // with their places swapped:
            //   v.re = s10.re * yb.i + (-(s9.re * ya.i))    = scratch12.im
            //   v.im = s9.im * ya.i + (-(s10.im * yb.i))    = scratch12.re
            let rounded = vmulq_f32(
                re_of_a_im_of_b(s9, s10),
                vld1q_f32([-YA_I, -YB_I, -YA_I, -YB_I].as_ptr()),
            );
            let v = vfmaq_f32(
                rounded,
                re_of_a_im_of_b(s10, s9),
                vld1q_f32([YB_I, YA_I, YB_I, YA_I].as_ptr()),
            );
            let s12 = vrev64q_f32(v);
            (o0, o1, vaddq_f32(s11, s12), vsubq_f32(s11, s12), o4)
        }};
    }

    /// One complex number in the low half of a vector, zero in the high half.
    ///
    /// # Safety
    /// `p[0..2]` must be readable.
    #[inline(always)]
    unsafe fn load1(p: *const f32) -> float32x4_t {
        vcombine_f32(vld1_f32(p), vdup_n_f32(0.0))
    }

    /// Stores the complex number in the low half of `v`.
    ///
    /// # Safety
    /// `p[0..2]` must be writable.
    #[inline(always)]
    unsafe fn store1(p: *mut f32, v: float32x4_t) {
        vst1_f32(p, vget_low_f32(v));
    }

    /// The kernels of one version. `$features` are the target features the
    /// functions are compiled with; `$cmul` and `$pm_i` are the two
    /// operations that differ between the versions.
    macro_rules! kernels {
        ($name:ident, $features:literal, $cmul:ident, $pm_i:ident) => {
            pub(super) mod $name {
                use super::*;

                /// Radix-3 pass. Two butterflies per iteration.
                ///
                /// # Safety
                /// The processor must have the target features of this
                /// function. `fout` must hold at least `3 * m * groups`
                /// samples and `tw` at least `4 * m` values.
                #[target_feature(enable = $features)]
                pub(crate) unsafe fn bfly3(
                    fout: &mut [KissCpx],
                    tw: &[f32],
                    m: usize,
                    groups: usize,
                ) {
                    debug_assert!(3 * m * groups <= fout.len());
                    debug_assert!(4 * m <= tw.len());
                    let w1 = tw.as_ptr();
                    let w2 = w1.add(2 * m);
                    let base = fout.as_mut_ptr() as *mut f32;

                    // SAFETY: group `g` is `3 * m` samples, 2 floats each.
                    // `p0`, `p1`, `p2` point to its three streams of `m`
                    // samples. A step at `j` touches samples `j` and `j + 1`
                    // of each stream and of each twiddle table, with
                    // `j + 2 <= m`. The last step of an odd `m` touches
                    // sample `j = m - 1` only.
                    for g in 0..groups {
                        let p0 = base.add(6 * m * g);
                        let p1 = p0.add(2 * m);
                        let p2 = p1.add(2 * m);
                        let step = |k: usize| {
                            let (o0, o1, o2) = bfly3_step!(
                                $cmul,
                                $pm_i,
                                vld1q_f32(p0.add(k)),
                                vld1q_f32(p1.add(k)),
                                vld1q_f32(p2.add(k)),
                                vld1q_f32(w1.add(k)),
                                vld1q_f32(w2.add(k))
                            );
                            vst1q_f32(p0.add(k), o0);
                            vst1q_f32(p1.add(k), o1);
                            vst1q_f32(p2.add(k), o2);
                        };
                        let mut j = 0usize;
                        while j + 2 <= m {
                            step(2 * j);
                            j += 2;
                        }
                        if j < m {
                            let k = 2 * j;
                            let (o0, o1, o2) = bfly3_step!(
                                $cmul,
                                $pm_i,
                                load1(p0.add(k)),
                                load1(p1.add(k)),
                                load1(p2.add(k)),
                                load1(w1.add(k)),
                                load1(w2.add(k))
                            );
                            store1(p0.add(k), o0);
                            store1(p1.add(k), o1);
                            store1(p2.add(k), o2);
                        }
                    }
                }

                /// Radix-4 pass with `m > 1`. Two butterflies per iteration.
                ///
                /// # Safety
                /// The processor must have the target features of this
                /// function. `fout` must hold at least `4 * m * groups`
                /// samples and `tw` at least `6 * m` values.
                #[target_feature(enable = $features)]
                pub(crate) unsafe fn bfly4(
                    fout: &mut [KissCpx],
                    tw: &[f32],
                    m: usize,
                    groups: usize,
                ) {
                    debug_assert!(4 * m * groups <= fout.len());
                    debug_assert!(6 * m <= tw.len());
                    let w1 = tw.as_ptr();
                    let w2 = w1.add(2 * m);
                    let w3 = w2.add(2 * m);
                    let base = fout.as_mut_ptr() as *mut f32;

                    // SAFETY: as in `bfly3`, with four streams and three
                    // twiddle tables.
                    for g in 0..groups {
                        let p0 = base.add(8 * m * g);
                        let p1 = p0.add(2 * m);
                        let p2 = p1.add(2 * m);
                        let p3 = p2.add(2 * m);
                        let step = |k: usize| {
                            let (o0, o1, o2, o3) = bfly4_step!(
                                $cmul,
                                $pm_i,
                                vld1q_f32(p0.add(k)),
                                vld1q_f32(p1.add(k)),
                                vld1q_f32(p2.add(k)),
                                vld1q_f32(p3.add(k)),
                                vld1q_f32(w1.add(k)),
                                vld1q_f32(w2.add(k)),
                                vld1q_f32(w3.add(k))
                            );
                            vst1q_f32(p0.add(k), o0);
                            vst1q_f32(p1.add(k), o1);
                            vst1q_f32(p2.add(k), o2);
                            vst1q_f32(p3.add(k), o3);
                        };
                        let mut j = 0usize;
                        while j + 2 <= m {
                            step(2 * j);
                            j += 2;
                        }
                        if j < m {
                            let k = 2 * j;
                            let (o0, o1, o2, o3) = bfly4_step!(
                                $cmul,
                                $pm_i,
                                load1(p0.add(k)),
                                load1(p1.add(k)),
                                load1(p2.add(k)),
                                load1(p3.add(k)),
                                load1(w1.add(k)),
                                load1(w2.add(k)),
                                load1(w3.add(k))
                            );
                            store1(p0.add(k), o0);
                            store1(p1.add(k), o1);
                            store1(p2.add(k), o2);
                            store1(p3.add(k), o3);
                        }
                    }
                }

                /// Radix-5 pass. Two butterflies per iteration.
                ///
                /// # Safety
                /// The processor must have the target features of this
                /// function. `fout` must hold at least `5 * m * groups`
                /// samples and `tw` at least `8 * m` values.
                #[target_feature(enable = $features)]
                pub(crate) unsafe fn bfly5(
                    fout: &mut [KissCpx],
                    tw: &[f32],
                    m: usize,
                    groups: usize,
                ) {
                    debug_assert!(5 * m * groups <= fout.len());
                    debug_assert!(8 * m <= tw.len());
                    let w1 = tw.as_ptr();
                    let w2 = w1.add(2 * m);
                    let w3 = w2.add(2 * m);
                    let w4 = w3.add(2 * m);
                    let base = fout.as_mut_ptr() as *mut f32;

                    // SAFETY: as in `bfly3`, with five streams and four
                    // twiddle tables.
                    for g in 0..groups {
                        let p0 = base.add(10 * m * g);
                        let p1 = p0.add(2 * m);
                        let p2 = p1.add(2 * m);
                        let p3 = p2.add(2 * m);
                        let p4 = p3.add(2 * m);
                        let step = |k: usize| {
                            let (o0, o1, o2, o3, o4) = bfly5_step!(
                                $cmul,
                                $pm_i,
                                vld1q_f32(p0.add(k)),
                                vld1q_f32(p1.add(k)),
                                vld1q_f32(p2.add(k)),
                                vld1q_f32(p3.add(k)),
                                vld1q_f32(p4.add(k)),
                                vld1q_f32(w1.add(k)),
                                vld1q_f32(w2.add(k)),
                                vld1q_f32(w3.add(k)),
                                vld1q_f32(w4.add(k))
                            );
                            vst1q_f32(p0.add(k), o0);
                            vst1q_f32(p1.add(k), o1);
                            vst1q_f32(p2.add(k), o2);
                            vst1q_f32(p3.add(k), o3);
                            vst1q_f32(p4.add(k), o4);
                        };
                        let mut j = 0usize;
                        while j + 2 <= m {
                            step(2 * j);
                            j += 2;
                        }
                        if j < m {
                            let k = 2 * j;
                            let (o0, o1, o2, o3, o4) = bfly5_step!(
                                $cmul,
                                $pm_i,
                                load1(p0.add(k)),
                                load1(p1.add(k)),
                                load1(p2.add(k)),
                                load1(p3.add(k)),
                                load1(p4.add(k)),
                                load1(w1.add(k)),
                                load1(w2.add(k)),
                                load1(w3.add(k)),
                                load1(w4.add(k))
                            );
                            store1(p0.add(k), o0);
                            store1(p1.add(k), o1);
                            store1(p2.add(k), o2);
                            store1(p3.add(k), o3);
                            store1(p4.add(k), o4);
                        }
                    }
                }

                /// Runs all stages.
                ///
                /// # Safety
                /// The processor must have the target features of this
                /// function. `fout` must hold exactly `st.nfft` samples.
                #[target_feature(enable = $features)]
                pub(crate) unsafe fn stages(st: &KissFftState, fout: &mut [KissCpx]) {
                    debug_assert!(fout.len() == st.nfft);
                    // SAFETY: `KissFftState::build` makes
                    // `groups * radix * m == nfft` for every stage, and gives
                    // each stage a table of `tw_len()` values.
                    for s in &st.stages[..st.nstages] {
                        let m = s.m as usize;
                        let groups = s.groups as usize;
                        let tw = &st.stage_tw[s.tw as usize..s.tw as usize + s.tw_len()];
                        match (s.radix, m) {
                            (2, 1) => neon::bfly2_m1(fout, groups),
                            (2, _) => neon::bfly2_m4(fout, groups),
                            (4, 1) => neon::bfly4_m1(fout, groups),
                            (4, _) => bfly4(fout, tw, m, groups),
                            (3, _) => bfly3(fout, tw, m, groups),
                            (5, _) => bfly5(fout, tw, m, groups),
                            // A one-point transform has a single stage of
                            // radix 1. It leaves the sample as it is.
                            _ => {}
                        }
                    }
                }
            }
        };
    }

    kernels!(fcma, "neon,fcma", cmul_fcma, pm_i_fcma);
    kernels!(fma, "neon", cmul_fma, pm_i_fma);
}

/// Whether the processor has the complex-number instructions (FEAT_FCMA).
///
/// Known at compile time where they are part of the target (Apple silicon).
/// Elsewhere it is a run-time check, which needs `std`; without `std` the
/// answer is no.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn has_fcma() -> bool {
    #[cfg(target_feature = "fcma")]
    {
        true
    }
    #[cfg(all(not(target_feature = "fcma"), feature = "std"))]
    {
        std::arch::is_aarch64_feature_detected!("fcma")
    }
    #[cfg(all(not(target_feature = "fcma"), not(feature = "std")))]
    {
        false
    }
}

/// Runs all stages with the aarch64 butterflies.
///
/// # Safety
/// `fout` must hold exactly `st.nfft` samples.
#[cfg(target_arch = "aarch64")]
unsafe fn fft_stages_neon(st: &KissFftState, fout: &mut [KissCpx]) {
    debug_assert!(fout.len() == st.nfft);
    if has_fcma() {
        // SAFETY: the processor has FEAT_FCMA.
        fused::fcma::stages(st, fout)
    } else {
        fused::fma::stages(st, fout)
    }
}

/// In-place FFT of `st.nfft()` samples that are already in bit-reversed order.
///
/// On aarch64 it gives the bits of `opus_fft_impl_fused`, which are the bits
/// of the arm64 build of libopus. On every other target it gives the bits of
/// `opus_fft_impl_scalar`.
pub fn opus_fft_impl(st: &KissFftState, fout: &mut [KissCpx]) {
    // The vector kernels read and write `nfft` values through pointers, so
    // the length is checked here.
    assert!(
        fout.len() >= st.nfft,
        "opus_fft_impl: fout.len() ({}) < nfft ({})",
        fout.len(),
        st.nfft
    );
    let fout = &mut fout[..st.nfft];

    // The choice of the implementation is made once per transform, not once
    // per stage.
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: `fout` holds exactly `nfft` samples.
        unsafe { fft_stages_neon(st, fout) }
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if crate::compat::x86_has_avx() {
            // SAFETY: the processor has AVX.
            unsafe { fft_stages_avx(st, fout) }
        } else {
            fft_stages_scalar::<Separate>(st, fout);
        }
    }
    #[cfg(not(any(
        target_arch = "aarch64",
        target_arch = "x86",
        target_arch = "x86_64"
    )))]
    fft_stages_scalar::<Separate>(st, fout);
}

/// libopus `opus_fft_c` (celt/kiss_fft.c:615-635): scales the input by
/// `1 / nfft` (`st.scale()`), puts it in bit-reversed order and transforms
/// it with `opus_fft_impl`. `fin` and `fout` hold `st.nfft()` samples each.
///
/// The speech and music analysis uses it, as libopus does.
pub fn opus_fft(st: &KissFftState, fin: &[KissCpx], fout: &mut [KissCpx]) {
    let scale = st.scale;
    let nfft = st.nfft;
    for (x, &rev) in fin[..nfft].iter().zip(st.bitrev.iter()) {
        fout[rev as usize] = KissCpx::new(x.r * scale, x.i * scale);
    }
    opus_fft_impl(st, fout);
}

/// The transform of `opus_fft_impl` with the portable butterflies and no
/// fused product, on every target.
///
/// It is the code of the targets without a vector version, and what a build
/// of libopus without fused multiply-add computes.
pub fn opus_fft_impl_scalar(st: &KissFftState, fout: &mut [KissCpx]) {
    assert!(
        fout.len() >= st.nfft,
        "opus_fft_impl_scalar: fout.len() ({}) < nfft ({})",
        fout.len(),
        st.nfft
    );
    fft_stages_scalar::<Separate>(st, &mut fout[..st.nfft]);
}

/// The transform of `opus_fft_impl` with the portable butterflies and the
/// products fused as clang fuses them in libopus for arm64, on every target.
///
/// It is the reference the aarch64 butterflies are tested against. On a
/// processor without a fused multiply-add instruction it is slow.
pub fn opus_fft_impl_fused(st: &KissFftState, fout: &mut [KissCpx]) {
    assert!(
        fout.len() >= st.nfft,
        "opus_fft_impl_fused: fout.len() ({}) < nfft ({})",
        fout.len(),
        st.nfft
    );
    fft_stages_scalar::<Fused>(st, &mut fout[..st.nfft]);
}

/// libopus `opus_fft_c` (celt/kiss_fft.c:615-635) with the products fused
/// as clang fuses them for arm64, on every target: each input value times
/// the plan's scale, stored at its bit-reversed place in `fout`, then the
/// transform of `opus_fft_impl_fused`.
///
/// The neural network code uses it, because it computes the bits of libopus
/// for arm64 on every target. On aarch64 it runs the vector butterflies of
/// `opus_fft_impl`, which give the same bits.
///
/// Panics when `fin` or `fout` holds fewer than `st.nfft()` values.
#[cfg(feature = "dnn")]
pub(crate) fn opus_fft_fused(st: &KissFftState, fin: &[KissCpx], fout: &mut [KissCpx]) {
    let n = st.nfft;
    let fin = &fin[..n];
    let fout = &mut fout[..n];
    let scale = st.scale;
    // `bitrev` is a permutation of `0..n` (checked in `build`), so every
    // value of `fout` is written.
    for (x, &place) in fin.iter().zip(st.bitrev.iter()) {
        fout[place as usize] = KissCpx::new(x.r * scale, x.i * scale);
    }
    #[cfg(target_arch = "aarch64")]
    opus_fft_impl(st, fout);
    #[cfg(not(target_arch = "aarch64"))]
    opus_fft_impl_fused(st, fout);
}

/// The twiddle table of the 480-point FFT: entry `i` is
/// `exp(-2 * pi * i / 480)`.
///
/// It is `fft_twiddles48000_960` of libopus 1.6
/// (celt/static_modes_float.h:132-373), digit for digit. The FFT sizes 240,
/// 120 and 60 read every second, fourth and eighth entry.
///
/// The numbers have eight significant digits. Some of them are not the `f32`
/// nearest to the exact cosine, so the table cannot be computed here; it has
/// to be copied.
#[allow(clippy::excessive_precision, clippy::approx_constant)]
pub static FFT_TWIDDLES_480: [KissCpx; KISS_MAX_N] = [
    KissCpx::new(1.0000000, -0.0000000), KissCpx::new(0.99991433, -0.013089596),
    KissCpx::new(0.99965732, -0.026176948), KissCpx::new(0.99922904, -0.039259816),
    KissCpx::new(0.99862953, -0.052335956), KissCpx::new(0.99785892, -0.065403129),
    KissCpx::new(0.99691733, -0.078459096), KissCpx::new(0.99580493, -0.091501619),
    KissCpx::new(0.99452190, -0.10452846), KissCpx::new(0.99306846, -0.11753740),
    KissCpx::new(0.99144486, -0.13052619), KissCpx::new(0.98965139, -0.14349262),
    KissCpx::new(0.98768834, -0.15643447), KissCpx::new(0.98555606, -0.16934950),
    KissCpx::new(0.98325491, -0.18223553), KissCpx::new(0.98078528, -0.19509032),
    KissCpx::new(0.97814760, -0.20791169), KissCpx::new(0.97534232, -0.22069744),
    KissCpx::new(0.97236992, -0.23344536), KissCpx::new(0.96923091, -0.24615329),
    KissCpx::new(0.96592583, -0.25881905), KissCpx::new(0.96245524, -0.27144045),
    KissCpx::new(0.95881973, -0.28401534), KissCpx::new(0.95501994, -0.29654157),
    KissCpx::new(0.95105652, -0.30901699), KissCpx::new(0.94693013, -0.32143947),
    KissCpx::new(0.94264149, -0.33380686), KissCpx::new(0.93819134, -0.34611706),
    KissCpx::new(0.93358043, -0.35836795), KissCpx::new(0.92880955, -0.37055744),
    KissCpx::new(0.92387953, -0.38268343), KissCpx::new(0.91879121, -0.39474386),
    KissCpx::new(0.91354546, -0.40673664), KissCpx::new(0.90814317, -0.41865974),
    KissCpx::new(0.90258528, -0.43051110), KissCpx::new(0.89687274, -0.44228869),
    KissCpx::new(0.89100652, -0.45399050), KissCpx::new(0.88498764, -0.46561452),
    KissCpx::new(0.87881711, -0.47715876), KissCpx::new(0.87249601, -0.48862124),
    KissCpx::new(0.86602540, -0.50000000), KissCpx::new(0.85940641, -0.51129309),
    KissCpx::new(0.85264016, -0.52249856), KissCpx::new(0.84572782, -0.53361452),
    KissCpx::new(0.83867057, -0.54463904), KissCpx::new(0.83146961, -0.55557023),
    KissCpx::new(0.82412619, -0.56640624), KissCpx::new(0.81664156, -0.57714519),
    KissCpx::new(0.80901699, -0.58778525), KissCpx::new(0.80125381, -0.59832460),
    KissCpx::new(0.79335334, -0.60876143), KissCpx::new(0.78531693, -0.61909395),
    KissCpx::new(0.77714596, -0.62932039), KissCpx::new(0.76884183, -0.63943900),
    KissCpx::new(0.76040597, -0.64944805), KissCpx::new(0.75183981, -0.65934582),
    KissCpx::new(0.74314483, -0.66913061), KissCpx::new(0.73432251, -0.67880075),
    KissCpx::new(0.72537437, -0.68835458), KissCpx::new(0.71630194, -0.69779046),
    KissCpx::new(0.70710678, -0.70710678), KissCpx::new(0.69779046, -0.71630194),
    KissCpx::new(0.68835458, -0.72537437), KissCpx::new(0.67880075, -0.73432251),
    KissCpx::new(0.66913061, -0.74314483), KissCpx::new(0.65934582, -0.75183981),
    KissCpx::new(0.64944805, -0.76040597), KissCpx::new(0.63943900, -0.76884183),
    KissCpx::new(0.62932039, -0.77714596), KissCpx::new(0.61909395, -0.78531693),
    KissCpx::new(0.60876143, -0.79335334), KissCpx::new(0.59832460, -0.80125381),
    KissCpx::new(0.58778525, -0.80901699), KissCpx::new(0.57714519, -0.81664156),
    KissCpx::new(0.56640624, -0.82412619), KissCpx::new(0.55557023, -0.83146961),
    KissCpx::new(0.54463904, -0.83867057), KissCpx::new(0.53361452, -0.84572782),
    KissCpx::new(0.52249856, -0.85264016), KissCpx::new(0.51129309, -0.85940641),
    KissCpx::new(0.50000000, -0.86602540), KissCpx::new(0.48862124, -0.87249601),
    KissCpx::new(0.47715876, -0.87881711), KissCpx::new(0.46561452, -0.88498764),
    KissCpx::new(0.45399050, -0.89100652), KissCpx::new(0.44228869, -0.89687274),
    KissCpx::new(0.43051110, -0.90258528), KissCpx::new(0.41865974, -0.90814317),
    KissCpx::new(0.40673664, -0.91354546), KissCpx::new(0.39474386, -0.91879121),
    KissCpx::new(0.38268343, -0.92387953), KissCpx::new(0.37055744, -0.92880955),
    KissCpx::new(0.35836795, -0.93358043), KissCpx::new(0.34611706, -0.93819134),
    KissCpx::new(0.33380686, -0.94264149), KissCpx::new(0.32143947, -0.94693013),
    KissCpx::new(0.30901699, -0.95105652), KissCpx::new(0.29654157, -0.95501994),
    KissCpx::new(0.28401534, -0.95881973), KissCpx::new(0.27144045, -0.96245524),
    KissCpx::new(0.25881905, -0.96592583), KissCpx::new(0.24615329, -0.96923091),
    KissCpx::new(0.23344536, -0.97236992), KissCpx::new(0.22069744, -0.97534232),
    KissCpx::new(0.20791169, -0.97814760), KissCpx::new(0.19509032, -0.98078528),
    KissCpx::new(0.18223553, -0.98325491), KissCpx::new(0.16934950, -0.98555606),
    KissCpx::new(0.15643447, -0.98768834), KissCpx::new(0.14349262, -0.98965139),
    KissCpx::new(0.13052619, -0.99144486), KissCpx::new(0.11753740, -0.99306846),
    KissCpx::new(0.10452846, -0.99452190), KissCpx::new(0.091501619, -0.99580493),
    KissCpx::new(0.078459096, -0.99691733), KissCpx::new(0.065403129, -0.99785892),
    KissCpx::new(0.052335956, -0.99862953), KissCpx::new(0.039259816, -0.99922904),
    KissCpx::new(0.026176948, -0.99965732), KissCpx::new(0.013089596, -0.99991433),
    KissCpx::new(6.1232340e-17, -1.0000000), KissCpx::new(-0.013089596, -0.99991433),
    KissCpx::new(-0.026176948, -0.99965732), KissCpx::new(-0.039259816, -0.99922904),
    KissCpx::new(-0.052335956, -0.99862953), KissCpx::new(-0.065403129, -0.99785892),
    KissCpx::new(-0.078459096, -0.99691733), KissCpx::new(-0.091501619, -0.99580493),
    KissCpx::new(-0.10452846, -0.99452190), KissCpx::new(-0.11753740, -0.99306846),
    KissCpx::new(-0.13052619, -0.99144486), KissCpx::new(-0.14349262, -0.98965139),
    KissCpx::new(-0.15643447, -0.98768834), KissCpx::new(-0.16934950, -0.98555606),
    KissCpx::new(-0.18223553, -0.98325491), KissCpx::new(-0.19509032, -0.98078528),
    KissCpx::new(-0.20791169, -0.97814760), KissCpx::new(-0.22069744, -0.97534232),
    KissCpx::new(-0.23344536, -0.97236992), KissCpx::new(-0.24615329, -0.96923091),
    KissCpx::new(-0.25881905, -0.96592583), KissCpx::new(-0.27144045, -0.96245524),
    KissCpx::new(-0.28401534, -0.95881973), KissCpx::new(-0.29654157, -0.95501994),
    KissCpx::new(-0.30901699, -0.95105652), KissCpx::new(-0.32143947, -0.94693013),
    KissCpx::new(-0.33380686, -0.94264149), KissCpx::new(-0.34611706, -0.93819134),
    KissCpx::new(-0.35836795, -0.93358043), KissCpx::new(-0.37055744, -0.92880955),
    KissCpx::new(-0.38268343, -0.92387953), KissCpx::new(-0.39474386, -0.91879121),
    KissCpx::new(-0.40673664, -0.91354546), KissCpx::new(-0.41865974, -0.90814317),
    KissCpx::new(-0.43051110, -0.90258528), KissCpx::new(-0.44228869, -0.89687274),
    KissCpx::new(-0.45399050, -0.89100652), KissCpx::new(-0.46561452, -0.88498764),
    KissCpx::new(-0.47715876, -0.87881711), KissCpx::new(-0.48862124, -0.87249601),
    KissCpx::new(-0.50000000, -0.86602540), KissCpx::new(-0.51129309, -0.85940641),
    KissCpx::new(-0.52249856, -0.85264016), KissCpx::new(-0.53361452, -0.84572782),
    KissCpx::new(-0.54463904, -0.83867057), KissCpx::new(-0.55557023, -0.83146961),
    KissCpx::new(-0.56640624, -0.82412619), KissCpx::new(-0.57714519, -0.81664156),
    KissCpx::new(-0.58778525, -0.80901699), KissCpx::new(-0.59832460, -0.80125381),
    KissCpx::new(-0.60876143, -0.79335334), KissCpx::new(-0.61909395, -0.78531693),
    KissCpx::new(-0.62932039, -0.77714596), KissCpx::new(-0.63943900, -0.76884183),
    KissCpx::new(-0.64944805, -0.76040597), KissCpx::new(-0.65934582, -0.75183981),
    KissCpx::new(-0.66913061, -0.74314483), KissCpx::new(-0.67880075, -0.73432251),
    KissCpx::new(-0.68835458, -0.72537437), KissCpx::new(-0.69779046, -0.71630194),
    KissCpx::new(-0.70710678, -0.70710678), KissCpx::new(-0.71630194, -0.69779046),
    KissCpx::new(-0.72537437, -0.68835458), KissCpx::new(-0.73432251, -0.67880075),
    KissCpx::new(-0.74314483, -0.66913061), KissCpx::new(-0.75183981, -0.65934582),
    KissCpx::new(-0.76040597, -0.64944805), KissCpx::new(-0.76884183, -0.63943900),
    KissCpx::new(-0.77714596, -0.62932039), KissCpx::new(-0.78531693, -0.61909395),
    KissCpx::new(-0.79335334, -0.60876143), KissCpx::new(-0.80125381, -0.59832460),
    KissCpx::new(-0.80901699, -0.58778525), KissCpx::new(-0.81664156, -0.57714519),
    KissCpx::new(-0.82412619, -0.56640624), KissCpx::new(-0.83146961, -0.55557023),
    KissCpx::new(-0.83867057, -0.54463904), KissCpx::new(-0.84572782, -0.53361452),
    KissCpx::new(-0.85264016, -0.52249856), KissCpx::new(-0.85940641, -0.51129309),
    KissCpx::new(-0.86602540, -0.50000000), KissCpx::new(-0.87249601, -0.48862124),
    KissCpx::new(-0.87881711, -0.47715876), KissCpx::new(-0.88498764, -0.46561452),
    KissCpx::new(-0.89100652, -0.45399050), KissCpx::new(-0.89687274, -0.44228869),
    KissCpx::new(-0.90258528, -0.43051110), KissCpx::new(-0.90814317, -0.41865974),
    KissCpx::new(-0.91354546, -0.40673664), KissCpx::new(-0.91879121, -0.39474386),
    KissCpx::new(-0.92387953, -0.38268343), KissCpx::new(-0.92880955, -0.37055744),
    KissCpx::new(-0.93358043, -0.35836795), KissCpx::new(-0.93819134, -0.34611706),
    KissCpx::new(-0.94264149, -0.33380686), KissCpx::new(-0.94693013, -0.32143947),
    KissCpx::new(-0.95105652, -0.30901699), KissCpx::new(-0.95501994, -0.29654157),
    KissCpx::new(-0.95881973, -0.28401534), KissCpx::new(-0.96245524, -0.27144045),
    KissCpx::new(-0.96592583, -0.25881905), KissCpx::new(-0.96923091, -0.24615329),
    KissCpx::new(-0.97236992, -0.23344536), KissCpx::new(-0.97534232, -0.22069744),
    KissCpx::new(-0.97814760, -0.20791169), KissCpx::new(-0.98078528, -0.19509032),
    KissCpx::new(-0.98325491, -0.18223553), KissCpx::new(-0.98555606, -0.16934950),
    KissCpx::new(-0.98768834, -0.15643447), KissCpx::new(-0.98965139, -0.14349262),
    KissCpx::new(-0.99144486, -0.13052619), KissCpx::new(-0.99306846, -0.11753740),
    KissCpx::new(-0.99452190, -0.10452846), KissCpx::new(-0.99580493, -0.091501619),
    KissCpx::new(-0.99691733, -0.078459096), KissCpx::new(-0.99785892, -0.065403129),
    KissCpx::new(-0.99862953, -0.052335956), KissCpx::new(-0.99922904, -0.039259816),
    KissCpx::new(-0.99965732, -0.026176948), KissCpx::new(-0.99991433, -0.013089596),
    KissCpx::new(-1.0000000, -1.2246468e-16), KissCpx::new(-0.99991433, 0.013089596),
    KissCpx::new(-0.99965732, 0.026176948), KissCpx::new(-0.99922904, 0.039259816),
    KissCpx::new(-0.99862953, 0.052335956), KissCpx::new(-0.99785892, 0.065403129),
    KissCpx::new(-0.99691733, 0.078459096), KissCpx::new(-0.99580493, 0.091501619),
    KissCpx::new(-0.99452190, 0.10452846), KissCpx::new(-0.99306846, 0.11753740),
    KissCpx::new(-0.99144486, 0.13052619), KissCpx::new(-0.98965139, 0.14349262),
    KissCpx::new(-0.98768834, 0.15643447), KissCpx::new(-0.98555606, 0.16934950),
    KissCpx::new(-0.98325491, 0.18223553), KissCpx::new(-0.98078528, 0.19509032),
    KissCpx::new(-0.97814760, 0.20791169), KissCpx::new(-0.97534232, 0.22069744),
    KissCpx::new(-0.97236992, 0.23344536), KissCpx::new(-0.96923091, 0.24615329),
    KissCpx::new(-0.96592583, 0.25881905), KissCpx::new(-0.96245524, 0.27144045),
    KissCpx::new(-0.95881973, 0.28401534), KissCpx::new(-0.95501994, 0.29654157),
    KissCpx::new(-0.95105652, 0.30901699), KissCpx::new(-0.94693013, 0.32143947),
    KissCpx::new(-0.94264149, 0.33380686), KissCpx::new(-0.93819134, 0.34611706),
    KissCpx::new(-0.93358043, 0.35836795), KissCpx::new(-0.92880955, 0.37055744),
    KissCpx::new(-0.92387953, 0.38268343), KissCpx::new(-0.91879121, 0.39474386),
    KissCpx::new(-0.91354546, 0.40673664), KissCpx::new(-0.90814317, 0.41865974),
    KissCpx::new(-0.90258528, 0.43051110), KissCpx::new(-0.89687274, 0.44228869),
    KissCpx::new(-0.89100652, 0.45399050), KissCpx::new(-0.88498764, 0.46561452),
    KissCpx::new(-0.87881711, 0.47715876), KissCpx::new(-0.87249601, 0.48862124),
    KissCpx::new(-0.86602540, 0.50000000), KissCpx::new(-0.85940641, 0.51129309),
    KissCpx::new(-0.85264016, 0.52249856), KissCpx::new(-0.84572782, 0.53361452),
    KissCpx::new(-0.83867057, 0.54463904), KissCpx::new(-0.83146961, 0.55557023),
    KissCpx::new(-0.82412619, 0.56640624), KissCpx::new(-0.81664156, 0.57714519),
    KissCpx::new(-0.80901699, 0.58778525), KissCpx::new(-0.80125381, 0.59832460),
    KissCpx::new(-0.79335334, 0.60876143), KissCpx::new(-0.78531693, 0.61909395),
    KissCpx::new(-0.77714596, 0.62932039), KissCpx::new(-0.76884183, 0.63943900),
    KissCpx::new(-0.76040597, 0.64944805), KissCpx::new(-0.75183981, 0.65934582),
    KissCpx::new(-0.74314483, 0.66913061), KissCpx::new(-0.73432251, 0.67880075),
    KissCpx::new(-0.72537437, 0.68835458), KissCpx::new(-0.71630194, 0.69779046),
    KissCpx::new(-0.70710678, 0.70710678), KissCpx::new(-0.69779046, 0.71630194),
    KissCpx::new(-0.68835458, 0.72537437), KissCpx::new(-0.67880075, 0.73432251),
    KissCpx::new(-0.66913061, 0.74314483), KissCpx::new(-0.65934582, 0.75183981),
    KissCpx::new(-0.64944805, 0.76040597), KissCpx::new(-0.63943900, 0.76884183),
    KissCpx::new(-0.62932039, 0.77714596), KissCpx::new(-0.61909395, 0.78531693),
    KissCpx::new(-0.60876143, 0.79335334), KissCpx::new(-0.59832460, 0.80125381),
    KissCpx::new(-0.58778525, 0.80901699), KissCpx::new(-0.57714519, 0.81664156),
    KissCpx::new(-0.56640624, 0.82412619), KissCpx::new(-0.55557023, 0.83146961),
    KissCpx::new(-0.54463904, 0.83867057), KissCpx::new(-0.53361452, 0.84572782),
    KissCpx::new(-0.52249856, 0.85264016), KissCpx::new(-0.51129309, 0.85940641),
    KissCpx::new(-0.50000000, 0.86602540), KissCpx::new(-0.48862124, 0.87249601),
    KissCpx::new(-0.47715876, 0.87881711), KissCpx::new(-0.46561452, 0.88498764),
    KissCpx::new(-0.45399050, 0.89100652), KissCpx::new(-0.44228869, 0.89687274),
    KissCpx::new(-0.43051110, 0.90258528), KissCpx::new(-0.41865974, 0.90814317),
    KissCpx::new(-0.40673664, 0.91354546), KissCpx::new(-0.39474386, 0.91879121),
    KissCpx::new(-0.38268343, 0.92387953), KissCpx::new(-0.37055744, 0.92880955),
    KissCpx::new(-0.35836795, 0.93358043), KissCpx::new(-0.34611706, 0.93819134),
    KissCpx::new(-0.33380686, 0.94264149), KissCpx::new(-0.32143947, 0.94693013),
    KissCpx::new(-0.30901699, 0.95105652), KissCpx::new(-0.29654157, 0.95501994),
    KissCpx::new(-0.28401534, 0.95881973), KissCpx::new(-0.27144045, 0.96245524),
    KissCpx::new(-0.25881905, 0.96592583), KissCpx::new(-0.24615329, 0.96923091),
    KissCpx::new(-0.23344536, 0.97236992), KissCpx::new(-0.22069744, 0.97534232),
    KissCpx::new(-0.20791169, 0.97814760), KissCpx::new(-0.19509032, 0.98078528),
    KissCpx::new(-0.18223553, 0.98325491), KissCpx::new(-0.16934950, 0.98555606),
    KissCpx::new(-0.15643447, 0.98768834), KissCpx::new(-0.14349262, 0.98965139),
    KissCpx::new(-0.13052619, 0.99144486), KissCpx::new(-0.11753740, 0.99306846),
    KissCpx::new(-0.10452846, 0.99452190), KissCpx::new(-0.091501619, 0.99580493),
    KissCpx::new(-0.078459096, 0.99691733), KissCpx::new(-0.065403129, 0.99785892),
    KissCpx::new(-0.052335956, 0.99862953), KissCpx::new(-0.039259816, 0.99922904),
    KissCpx::new(-0.026176948, 0.99965732), KissCpx::new(-0.013089596, 0.99991433),
    KissCpx::new(-1.8369702e-16, 1.0000000), KissCpx::new(0.013089596, 0.99991433),
    KissCpx::new(0.026176948, 0.99965732), KissCpx::new(0.039259816, 0.99922904),
    KissCpx::new(0.052335956, 0.99862953), KissCpx::new(0.065403129, 0.99785892),
    KissCpx::new(0.078459096, 0.99691733), KissCpx::new(0.091501619, 0.99580493),
    KissCpx::new(0.10452846, 0.99452190), KissCpx::new(0.11753740, 0.99306846),
    KissCpx::new(0.13052619, 0.99144486), KissCpx::new(0.14349262, 0.98965139),
    KissCpx::new(0.15643447, 0.98768834), KissCpx::new(0.16934950, 0.98555606),
    KissCpx::new(0.18223553, 0.98325491), KissCpx::new(0.19509032, 0.98078528),
    KissCpx::new(0.20791169, 0.97814760), KissCpx::new(0.22069744, 0.97534232),
    KissCpx::new(0.23344536, 0.97236992), KissCpx::new(0.24615329, 0.96923091),
    KissCpx::new(0.25881905, 0.96592583), KissCpx::new(0.27144045, 0.96245524),
    KissCpx::new(0.28401534, 0.95881973), KissCpx::new(0.29654157, 0.95501994),
    KissCpx::new(0.30901699, 0.95105652), KissCpx::new(0.32143947, 0.94693013),
    KissCpx::new(0.33380686, 0.94264149), KissCpx::new(0.34611706, 0.93819134),
    KissCpx::new(0.35836795, 0.93358043), KissCpx::new(0.37055744, 0.92880955),
    KissCpx::new(0.38268343, 0.92387953), KissCpx::new(0.39474386, 0.91879121),
    KissCpx::new(0.40673664, 0.91354546), KissCpx::new(0.41865974, 0.90814317),
    KissCpx::new(0.43051110, 0.90258528), KissCpx::new(0.44228869, 0.89687274),
    KissCpx::new(0.45399050, 0.89100652), KissCpx::new(0.46561452, 0.88498764),
    KissCpx::new(0.47715876, 0.87881711), KissCpx::new(0.48862124, 0.87249601),
    KissCpx::new(0.50000000, 0.86602540), KissCpx::new(0.51129309, 0.85940641),
    KissCpx::new(0.52249856, 0.85264016), KissCpx::new(0.53361452, 0.84572782),
    KissCpx::new(0.54463904, 0.83867057), KissCpx::new(0.55557023, 0.83146961),
    KissCpx::new(0.56640624, 0.82412619), KissCpx::new(0.57714519, 0.81664156),
    KissCpx::new(0.58778525, 0.80901699), KissCpx::new(0.59832460, 0.80125381),
    KissCpx::new(0.60876143, 0.79335334), KissCpx::new(0.61909395, 0.78531693),
    KissCpx::new(0.62932039, 0.77714596), KissCpx::new(0.63943900, 0.76884183),
    KissCpx::new(0.64944805, 0.76040597), KissCpx::new(0.65934582, 0.75183981),
    KissCpx::new(0.66913061, 0.74314483), KissCpx::new(0.67880075, 0.73432251),
    KissCpx::new(0.68835458, 0.72537437), KissCpx::new(0.69779046, 0.71630194),
    KissCpx::new(0.70710678, 0.70710678), KissCpx::new(0.71630194, 0.69779046),
    KissCpx::new(0.72537437, 0.68835458), KissCpx::new(0.73432251, 0.67880075),
    KissCpx::new(0.74314483, 0.66913061), KissCpx::new(0.75183981, 0.65934582),
    KissCpx::new(0.76040597, 0.64944805), KissCpx::new(0.76884183, 0.63943900),
    KissCpx::new(0.77714596, 0.62932039), KissCpx::new(0.78531693, 0.61909395),
    KissCpx::new(0.79335334, 0.60876143), KissCpx::new(0.80125381, 0.59832460),
    KissCpx::new(0.80901699, 0.58778525), KissCpx::new(0.81664156, 0.57714519),
    KissCpx::new(0.82412619, 0.56640624), KissCpx::new(0.83146961, 0.55557023),
    KissCpx::new(0.83867057, 0.54463904), KissCpx::new(0.84572782, 0.53361452),
    KissCpx::new(0.85264016, 0.52249856), KissCpx::new(0.85940641, 0.51129309),
    KissCpx::new(0.86602540, 0.50000000), KissCpx::new(0.87249601, 0.48862124),
    KissCpx::new(0.87881711, 0.47715876), KissCpx::new(0.88498764, 0.46561452),
    KissCpx::new(0.89100652, 0.45399050), KissCpx::new(0.89687274, 0.44228869),
    KissCpx::new(0.90258528, 0.43051110), KissCpx::new(0.90814317, 0.41865974),
    KissCpx::new(0.91354546, 0.40673664), KissCpx::new(0.91879121, 0.39474386),
    KissCpx::new(0.92387953, 0.38268343), KissCpx::new(0.92880955, 0.37055744),
    KissCpx::new(0.93358043, 0.35836795), KissCpx::new(0.93819134, 0.34611706),
    KissCpx::new(0.94264149, 0.33380686), KissCpx::new(0.94693013, 0.32143947),
    KissCpx::new(0.95105652, 0.30901699), KissCpx::new(0.95501994, 0.29654157),
    KissCpx::new(0.95881973, 0.28401534), KissCpx::new(0.96245524, 0.27144045),
    KissCpx::new(0.96592583, 0.25881905), KissCpx::new(0.96923091, 0.24615329),
    KissCpx::new(0.97236992, 0.23344536), KissCpx::new(0.97534232, 0.22069744),
    KissCpx::new(0.97814760, 0.20791169), KissCpx::new(0.98078528, 0.19509032),
    KissCpx::new(0.98325491, 0.18223553), KissCpx::new(0.98555606, 0.16934950),
    KissCpx::new(0.98768834, 0.15643447), KissCpx::new(0.98965139, 0.14349262),
    KissCpx::new(0.99144486, 0.13052619), KissCpx::new(0.99306846, 0.11753740),
    KissCpx::new(0.99452190, 0.10452846), KissCpx::new(0.99580493, 0.091501619),
    KissCpx::new(0.99691733, 0.078459096), KissCpx::new(0.99785892, 0.065403129),
    KissCpx::new(0.99862953, 0.052335956), KissCpx::new(0.99922904, 0.039259816),
    KissCpx::new(0.99965732, 0.026176948), KissCpx::new(0.99991433, 0.013089596),
];

/// Test memory with pages around it that must not be touched.
///
/// A kernel that reads or writes even one value outside a slice placed here
/// stops the test with a memory fault. The tests of `mdct.rs` use it too.
#[cfg(all(test, feature = "std"))]
pub(crate) mod test_guard {
    #[cfg(all(
        any(target_os = "macos", target_os = "linux"),
        target_pointer_width = "64"
    ))]
    mod imp {
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

        /// 64 KiB of memory between two blocks of 64 KiB that may be neither
        /// read nor written.
        pub(crate) struct Region {
            base: *mut u8,
        }

        impl Region {
            pub(crate) fn new() -> Self {
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

            /// A copy of `data` that ends at the protected block behind the
            /// region (`at_end`), or starts right behind the protected block
            /// in front of it.
            pub(crate) fn place<T: Copy>(&mut self, data: &[T], at_end: bool) -> &mut [T] {
                let bytes = core::mem::size_of_val(data);
                assert!(bytes <= SPAN && SPAN.is_multiple_of(core::mem::align_of::<T>()));
                // SAFETY: the middle third of the mapping is readable and
                // writable and `bytes` fit in it. Its two ends are aligned
                // for `T`, and `bytes` is a multiple of the size of `T`. The
                // returned slice borrows `self`.
                unsafe {
                    let start = if at_end {
                        self.base.add(2 * SPAN - bytes)
                    } else {
                        self.base.add(SPAN)
                    } as *mut T;
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

    /// Without `mmap`: a plain buffer of exactly the length asked for.
    #[cfg(not(all(
        any(target_os = "macos", target_os = "linux"),
        target_pointer_width = "64"
    )))]
    mod imp {
        pub(crate) struct Region {
            /// `u128` gives the alignment of every type the tests place.
            buf: Vec<u128>,
        }

        impl Region {
            pub(crate) fn new() -> Self {
                Region { buf: Vec::new() }
            }

            pub(crate) fn place<T: Copy>(&mut self, data: &[T], _at_end: bool) -> &mut [T] {
                let bytes = core::mem::size_of_val(data);
                assert!(core::mem::align_of::<T>() <= core::mem::align_of::<u128>());
                self.buf.clear();
                self.buf.resize(bytes.div_ceil(16), 0);
                // SAFETY: the buffer holds at least `bytes` bytes and is
                // aligned for `T`. The returned slice borrows `self`.
                unsafe {
                    let start = self.buf.as_mut_ptr() as *mut T;
                    core::ptr::copy_nonoverlapping(data.as_ptr(), start, data.len());
                    core::slice::from_raw_parts_mut(start, data.len())
                }
            }
        }
    }

    pub(crate) use imp::Region;

    /// The two places the tests put each slice: ending at a protected block,
    /// and starting right behind one.
    pub(crate) const PLACEMENTS: [bool; 2] = [true, false];
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::test_guard::{PLACEMENTS, Region};
    use super::*;
    use core::f32::consts::PI;

    /// libopus `opus_ifft_c`: the inverse transform, without scaling.
    fn opus_ifft(st: &KissFftState, fin: &[KissCpx], fout: &mut [KissCpx]) {
        let nfft = st.nfft;

        for i in 0..nfft {
            let rev = st.bitrev[i] as usize;
            fout[rev] = KissCpx::new(fin[i].r, -fin[i].i);
        }

        opus_fft_impl(st, fout);

        for i in 0..nfft {
            fout[i].i = -fout[i].i;
        }
    }

    fn almost_equal(a: f32, b: f32, tolerance: f32) -> bool {
        (a - b).abs() < tolerance
    }

    fn cpx_almost_equal(a: &KissCpx, b: &KissCpx, tolerance: f32) -> bool {
        almost_equal(a.r, b.r, tolerance) && almost_equal(a.i, b.i, tolerance)
    }

    /// Deterministic pseudo-random numbers, so every run tests the same data.
    struct Lcg(u64);

    impl Lcg {
        fn next_u32(&mut self) -> u32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 33) as u32
        }

        /// Uniform in [-1, 1).
        fn unit(&mut self) -> f32 {
            (self.next_u32() as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
        }
    }

    /// Number of input classes `fill` knows.
    const CLASSES: usize = 5;

    /// The numbers of class 4.
    const ZEROS_AND_ONES: [f32; 4] = [0.0, -0.0, 1.0, -1.0];

    /// Fills `out` with test data.
    ///
    /// Class 0: audio-sized values. Class 1: values whose products overflow
    /// to infinity and then give NaN. Class 2: values whose products are
    /// subnormal. Class 3: signed zeros, infinities, the largest and the
    /// smallest numbers, mixed with ordinary values. Class 4: zeros of both
    /// signs, 1 and -1; many products and sums are then exactly zero, and
    /// the sign of the zero is all that can go wrong.
    fn fill(rng: &mut Lcg, out: &mut [KissCpx], class: usize) {
        const SPECIAL: [f32; 9] = [
            0.0,
            -0.0,
            f32::MAX,
            f32::MIN,
            f32::MIN_POSITIVE,
            1e-45,
            f32::INFINITY,
            f32::NEG_INFINITY,
            1.0,
        ];
        let value = |rng: &mut Lcg| match class {
            0 => rng.unit() * 32768.0,
            1 => rng.unit() * 1e30,
            2 => rng.unit() * 1e-38,
            4 => ZEROS_AND_ONES[rng.next_u32() as usize % 4],
            _ => {
                let pick = rng.next_u32() as usize % (2 * SPECIAL.len());
                if pick < SPECIAL.len() {
                    SPECIAL[pick]
                } else {
                    rng.unit()
                }
            }
        };
        for c in out.iter_mut() {
            *c = KissCpx::new(value(rng), value(rng));
        }
    }

    /// Bit-for-bit comparison. Any NaN equals any other NaN: the sign and the
    /// payload of a NaN are not part of what the codec computes.
    fn same_f32(a: f32, b: f32) -> bool {
        a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
    }

    fn first_difference(a: &[KissCpx], b: &[KissCpx]) -> Option<usize> {
        assert_eq!(a.len(), b.len());
        (0..a.len()).find(|&i| !same_f32(a[i].r, b[i].r) || !same_f32(a[i].i, b[i].i))
    }

    /// Every FFT size the plan builder accepts.
    fn valid_sizes() -> Vec<usize> {
        (1..=KISS_MAX_N)
            .filter(|&n| KissFftState::new(n).is_some())
            .collect()
    }

    /// The FFT as it was before the per-stage twiddle tables: the butterflies
    /// of libopus, reading one shared twiddle table with a stride. `A` says
    /// whether the products that clang fuses for arm64 are fused.
    mod strided {
        use super::super::*;

        fn bfly2(fout: &mut [KissCpx], m: usize, n: usize) {
            if m == 1 {
                for i in 0..n {
                    let idx = i * 2;
                    let t = fout[idx + 1];
                    fout[idx + 1] = c_sub(&fout[idx], &t);
                    fout[idx] = c_add(&fout[idx], &t);
                }
            } else {
                let tw: f32 = core::f32::consts::FRAC_1_SQRT_2;
                for i in 0..n {
                    let base = i * 8;

                    let t = fout[base + 4];
                    fout[base + 4] = c_sub(&fout[base], &t);
                    fout[base] = c_add(&fout[base], &t);

                    let t = KissCpx::new(
                        (fout[base + 5].r + fout[base + 5].i) * tw,
                        (fout[base + 5].i - fout[base + 5].r) * tw,
                    );
                    fout[base + 5] = c_sub(&fout[base + 1], &t);
                    fout[base + 1] = c_add(&fout[base + 1], &t);

                    let t = KissCpx::new(fout[base + 6].i, -fout[base + 6].r);
                    fout[base + 6] = c_sub(&fout[base + 2], &t);
                    fout[base + 2] = c_add(&fout[base + 2], &t);

                    let t = KissCpx::new(
                        (fout[base + 7].i - fout[base + 7].r) * tw,
                        -(fout[base + 7].i + fout[base + 7].r) * tw,
                    );
                    fout[base + 7] = c_sub(&fout[base + 3], &t);
                    fout[base + 3] = c_add(&fout[base + 3], &t);
                }
            }
        }

        fn bfly4<A: Arith>(
            fout: &mut [KissCpx],
            fstride: usize,
            twiddles: &[KissCpx],
            m: usize,
            n: usize,
            mm: usize,
        ) {
            if m == 1 {
                for i in 0..n {
                    let base = i * 4;

                    let scratch0 = c_sub(&fout[base], &fout[base + 2]);
                    let sum02 = c_add(&fout[base], &fout[base + 2]);
                    let scratch1 = c_add(&fout[base + 1], &fout[base + 3]);
                    let diff13 = c_sub(&fout[base + 1], &fout[base + 3]);

                    fout[base] = c_add(&sum02, &scratch1);
                    fout[base + 2] = c_sub(&sum02, &scratch1);
                    fout[base + 1] = KissCpx::new(scratch0.r + diff13.i, scratch0.i - diff13.r);
                    fout[base + 3] = KissCpx::new(scratch0.r - diff13.i, scratch0.i + diff13.r);
                }
            } else {
                let stride2 = fstride * 2;
                let stride3 = fstride * 3;
                let m2 = 2 * m;
                let m3 = 3 * m;
                for i in 0..n {
                    let base = i * mm;
                    let mut tw1 = 0usize;
                    let mut tw2 = 0usize;
                    let mut tw3 = 0usize;

                    for j in 0..m {
                        let idx = base + j;

                        let scratch0 = c_mul::<A>(&fout[idx + m], &twiddles[tw1]);
                        let scratch1 = c_mul::<A>(&fout[idx + m2], &twiddles[tw2]);
                        let scratch2 = c_mul::<A>(&fout[idx + m3], &twiddles[tw3]);

                        let scratch5 = c_sub(&fout[idx], &scratch1);
                        fout[idx] = c_add(&fout[idx], &scratch1);

                        let scratch3 = c_add(&scratch0, &scratch2);
                        let scratch4 = c_sub(&scratch0, &scratch2);

                        fout[idx + m2] = c_sub(&fout[idx], &scratch3);
                        fout[idx] = c_add(&fout[idx], &scratch3);

                        fout[idx + m] =
                            KissCpx::new(scratch5.r + scratch4.i, scratch5.i - scratch4.r);
                        fout[idx + m3] =
                            KissCpx::new(scratch5.r - scratch4.i, scratch5.i + scratch4.r);

                        tw1 += fstride;
                        tw2 += stride2;
                        tw3 += stride3;
                    }
                }
            }
        }

        fn bfly3<A: Arith>(
            fout: &mut [KissCpx],
            fstride: usize,
            twiddles: &[KissCpx],
            m: usize,
            n: usize,
            mm: usize,
        ) {
            let m2 = 2 * m;
            let epi3_i: f32 = -0.866_025_4;
            let stride2 = fstride * 2;

            for i in 0..n {
                let base = i * mm;
                let mut tw1 = 0usize;
                let mut tw2 = 0usize;

                for j in 0..m {
                    let idx = base + j;

                    let scratch1 = c_mul::<A>(&fout[idx + m], &twiddles[tw1]);
                    let scratch2 = c_mul::<A>(&fout[idx + m2], &twiddles[tw2]);

                    let scratch3 = c_add(&scratch1, &scratch2);
                    let scratch0 = c_sub(&scratch1, &scratch2);

                    let fout_m = KissCpx::new(
                        A::mul_add(-scratch3.r, 0.5, fout[idx].r),
                        A::mul_add(-scratch3.i, 0.5, fout[idx].i),
                    );

                    let scratch0_scaled = KissCpx::new(scratch0.r * epi3_i, scratch0.i * epi3_i);

                    fout[idx] = c_add(&fout[idx], &scratch3);

                    fout[idx + m] =
                        KissCpx::new(fout_m.r - scratch0_scaled.i, fout_m.i + scratch0_scaled.r);
                    fout[idx + m2] =
                        KissCpx::new(fout_m.r + scratch0_scaled.i, fout_m.i - scratch0_scaled.r);

                    tw1 += fstride;
                    tw2 += stride2;
                }
            }
        }

        fn bfly5<A: Arith>(
            fout: &mut [KissCpx],
            fstride: usize,
            twiddles: &[KissCpx],
            m: usize,
            n: usize,
            mm: usize,
        ) {
            let ya = KissCpx::new(0.309_017, -0.95105652);
            let yb = KissCpx::new(-0.809_017, -0.58778525);

            for i in 0..n {
                let base = i * mm;
                let stride2 = fstride * 2;
                let stride3 = fstride * 3;
                let stride4 = fstride * 4;

                let mut tw1 = 0usize;
                let mut tw2 = 0usize;
                let mut tw3 = 0usize;
                let mut tw4 = 0usize;

                for u in 0..m {
                    let idx0 = base + u;
                    let idx1 = idx0 + m;
                    let idx2 = idx0 + 2 * m;
                    let idx3 = idx0 + 3 * m;
                    let idx4 = idx0 + 4 * m;

                    let scratch0 = fout[idx0];

                    let scratch1 = c_mul::<A>(&fout[idx1], &twiddles[tw1]);
                    let scratch2 = c_mul::<A>(&fout[idx2], &twiddles[tw2]);
                    let scratch3 = c_mul::<A>(&fout[idx3], &twiddles[tw3]);
                    let scratch4 = c_mul::<A>(&fout[idx4], &twiddles[tw4]);

                    let scratch7 = c_add(&scratch1, &scratch4);
                    let scratch10 = c_sub(&scratch1, &scratch4);
                    let scratch8 = c_add(&scratch2, &scratch3);
                    let scratch9 = c_sub(&scratch2, &scratch3);

                    fout[idx0] = KissCpx::new(
                        scratch0.r + (scratch7.r + scratch8.r),
                        scratch0.i + (scratch7.i + scratch8.i),
                    );

                    let scratch5 = KissCpx::new(
                        scratch0.r + A::mul_add(scratch7.r, ya.r, scratch8.r * yb.r),
                        scratch0.i + A::mul_add(scratch7.i, ya.r, scratch8.i * yb.r),
                    );

                    let scratch6 = KissCpx::new(
                        A::mul_add(scratch10.i, ya.i, scratch9.i * yb.i),
                        -A::mul_add(scratch10.r, ya.i, scratch9.r * yb.i),
                    );

                    fout[idx1] = c_sub(&scratch5, &scratch6);
                    fout[idx4] = c_add(&scratch5, &scratch6);

                    let scratch11 = KissCpx::new(
                        scratch0.r + A::mul_add(scratch7.r, yb.r, scratch8.r * ya.r),
                        scratch0.i + A::mul_add(scratch7.i, yb.r, scratch8.i * ya.r),
                    );

                    let scratch12 = KissCpx::new(
                        A::mul_add(scratch9.i, ya.i, -(scratch10.i * yb.i)),
                        A::mul_add(scratch10.r, yb.i, -(scratch9.r * ya.i)),
                    );

                    fout[idx2] = c_add(&scratch11, &scratch12);
                    fout[idx3] = c_sub(&scratch11, &scratch12);

                    tw1 += fstride;
                    tw2 += stride2;
                    tw3 += stride3;
                    tw4 += stride4;
                }
            }
        }

        /// In-place FFT of `nfft` samples with the twiddle table of the size
        /// `nfft << shift`.
        pub fn fft<A: Arith>(nfft: usize, shift: usize, fout: &mut [KissCpx]) {
            let mut factors = [0i16; 2 * MAXFACTORS];
            assert!(kf_factor(nfft, &mut factors));
            let twiddles = compute_twiddles(nfft << shift);

            let mut fstride = [0usize; MAXFACTORS + 1];
            fstride[0] = 1;
            let mut l = 0;
            let mut m;
            loop {
                let p = factors[2 * l] as usize;
                m = factors[2 * l + 1] as usize;
                fstride[l + 1] = fstride[l] * p;
                l += 1;
                if m == 1 {
                    break;
                }
            }

            m = factors[2 * l - 1] as usize;
            for i in (0..l).rev() {
                let p = factors[2 * i] as usize;
                let fstride_i = fstride[i];
                let fstride_adjusted = fstride_i << shift;
                let m2 = if i > 0 {
                    factors[2 * i - 1] as usize
                } else {
                    1
                };
                match p {
                    2 => bfly2(fout, m, fstride_i),
                    4 => bfly4::<A>(fout, fstride_adjusted, &twiddles, m, fstride_i, m2),
                    3 => bfly3::<A>(fout, fstride_adjusted, &twiddles, m, fstride_i, m2),
                    5 => bfly5::<A>(fout, fstride_adjusted, &twiddles, m, fstride_i, m2),
                    _ => {}
                }
                m = m2;
            }
        }
    }

    /// The constants of the radix-3 and radix-5 butterflies are the table
    /// entries libopus uses for them.
    #[test]
    fn test_butterfly_constants_are_table_entries() {
        assert_eq!(EPI3_I.to_bits(), FFT_TWIDDLES_480[160].i.to_bits());
        assert_eq!(YA_R.to_bits(), FFT_TWIDDLES_480[96].r.to_bits());
        assert_eq!(YA_I.to_bits(), FFT_TWIDDLES_480[96].i.to_bits());
        assert_eq!(YB_R.to_bits(), FFT_TWIDDLES_480[192].r.to_bits());
        assert_eq!(YB_I.to_bits(), FFT_TWIDDLES_480[192].i.to_bits());
    }

    /// The copied table is a twiddle table: every entry is within one unit
    /// in the last place of the exact value.
    #[test]
    fn test_twiddle_table_values() {
        let mut inexact = 0;
        for (i, t) in FFT_TWIDDLES_480.iter().enumerate() {
            let phase = -2.0 * core::f64::consts::PI * i as f64 / 480.0;
            let (s, c) = phase.sin_cos();
            assert!((t.r as f64 - c).abs() <= 6.0e-8, "entry {i}");
            assert!((t.i as f64 - s).abs() <= 6.0e-8, "entry {i}");
            if t.r != c as f32 || t.i != s as f32 {
                inexact += 1;
            }
        }
        // Why the table is copied and not computed.
        assert!(inexact > 0);
        assert_eq!(FFT_TWIDDLES_480[0].i.to_bits(), (-0.0f32).to_bits());
    }

    #[test]
    fn test_kf_factor_60() {
        let st = KissFftState::new(60).unwrap();
        assert!(st.nstages > 0);
        assert_eq!(st.bitrev.len(), 60);
    }

    #[test]
    fn test_kf_factor_240() {
        let st = KissFftState::new(240).unwrap();
        assert!(st.nstages > 0);
        assert_eq!(st.bitrev.len(), 240);
    }

    #[test]
    fn test_plan_rejects_unusable_sizes() {
        assert!(KissFftState::new(0).is_none());
        assert!(KissFftState::new(7).is_none());
        assert!(KissFftState::new(KISS_MAX_N + 1).is_none());
        let base = KissFftState::new(480).unwrap();
        assert!(KissFftState::new_sub(&base, 0).is_none());
        assert!(KissFftState::new_sub(&base, 100).is_none());
    }

    /// Every plan holds a permutation of `0..nfft` in `bitrev`. The MDCT
    /// relies on it.
    #[test]
    fn test_bitrev_permutation() {
        let base = KissFftState::new(480).unwrap();
        let mut plans: Vec<KissFftState> = valid_sizes()
            .into_iter()
            .map(|n| KissFftState::new(n).unwrap())
            .collect();
        for n in [60, 120, 240] {
            plans.push(KissFftState::new_sub(&base, n).unwrap());
        }
        for st in &plans {
            let nfft = st.nfft();
            let mut sorted: Vec<i16> = st.bitrev.as_slice().to_vec();
            sorted.sort();
            let expected: Vec<i16> = (0..nfft).map(|x| x as i16).collect();
            assert_eq!(sorted, expected, "bitrev for nfft={nfft}");
        }
    }

    /// The permutation check of `build` rejects no size: the table is a
    /// permutation for every size that has a factorisation.
    #[test]
    fn test_bitrev_is_a_permutation_for_every_factorisation() {
        let mut sizes = 0;
        for nfft in 1..=KISS_MAX_N {
            let mut factors = [0i16; 2 * MAXFACTORS];
            if !kf_factor(nfft, &mut factors) {
                continue;
            }
            let mut bitrev = vec![0i16; nfft];
            compute_bitrev_table(0, &mut bitrev, 1, 1, &factors);
            assert!(is_permutation(&bitrev), "nfft {nfft}");
            sizes += 1;
        }
        assert!(sizes >= valid_sizes().len());
        assert!(valid_sizes().contains(&480));
    }

    #[test]
    fn test_is_permutation() {
        assert!(is_permutation(&[]));
        assert!(is_permutation(&[0]));
        assert!(is_permutation(&[2, 0, 1]));
        // A number twice, a number too large, a negative number.
        assert!(!is_permutation(&[0, 0]));
        assert!(!is_permutation(&[0, 2]));
        assert!(!is_permutation(&[1, -1]));
        // The last bit of a word and the first bit of the next one.
        let mut long: Vec<i16> = (0..130).collect();
        assert!(is_permutation(&long));
        long[63] = 64;
        assert!(!is_permutation(&long));
        assert!(!is_permutation(&vec![0i16; KISS_MAX_N + 1]));
    }

    #[test]
    fn test_sub_fft() {
        let nfft = 480;
        let base = KissFftState::new(nfft).unwrap();

        for &n in &[60, 120, 240] {
            let sub = KissFftState::new_sub(&base, n).unwrap();
            assert_eq!(sub.nfft(), n);
        }
    }

    /// Every stage covers the whole buffer, and the stage tables follow each
    /// other without gaps.
    #[test]
    fn test_stages_cover_the_buffer() {
        for nfft in valid_sizes() {
            let st = KissFftState::new(nfft).unwrap();
            let mut tw = 0usize;
            for s in &st.stages[..st.nstages] {
                assert_eq!(
                    s.groups as usize * s.radix as usize * s.m as usize,
                    nfft,
                    "nfft {nfft}"
                );
                assert_eq!(s.tw as usize, tw, "nfft {nfft}");
                tw += s.tw_len();
            }
            assert_eq!(tw, st.stage_tw.len(), "nfft {nfft}");
            assert!(tw <= 2 * (nfft - 1), "nfft {nfft}");
        }
    }

    /// The portable FFT with per-stage tables gives the bits of the FFT that
    /// read one shared table with a stride, with and without fused products.
    #[test]
    fn test_scalar_fft_matches_strided_tables() {
        let mut rng = Lcg(0x5eed_0001);
        let mut plans: Vec<(KissFftState, usize, usize)> = valid_sizes()
            .into_iter()
            .map(|n| (KissFftState::new(n).unwrap(), n, 0))
            .collect();
        let base = KissFftState::new(480).unwrap();
        for shift in 1..=3 {
            let n = 480 >> shift;
            plans.push((KissFftState::new_sub(&base, n).unwrap(), n, shift));
        }
        for (st, nfft, shift) in &plans {
            for class in 0..CLASSES {
                let mut a = vec![KissCpx::default(); *nfft];
                fill(&mut rng, &mut a, class);
                let mut b = a.clone();
                let mut c = a.clone();
                let mut d = a.clone();
                opus_fft_impl_scalar(st, &mut a);
                strided::fft::<Separate>(*nfft, *shift, &mut b);
                assert_eq!(
                    first_difference(&a, &b),
                    None,
                    "nfft {nfft} shift {shift} class {class}"
                );
                opus_fft_impl_fused(st, &mut c);
                strided::fft::<Fused>(*nfft, *shift, &mut d);
                assert_eq!(
                    first_difference(&c, &d),
                    None,
                    "fused, nfft {nfft} shift {shift} class {class}"
                );
            }
        }
    }

    /// The portable FFT whose bits `opus_fft_impl` gives on this target.
    fn reference_fft(st: &KissFftState, fout: &mut [KissCpx]) {
        if cfg!(target_arch = "aarch64") {
            opus_fft_impl_fused(st, fout);
        } else {
            opus_fft_impl_scalar(st, fout);
        }
    }

    /// Largest difference between two results, relative to the largest
    /// value.
    fn max_relative_difference(a: &[KissCpx], b: &[KissCpx]) -> f32 {
        let peak = b.iter().fold(0f32, |m, c| m.max(c.r.abs()).max(c.i.abs()));
        let diff = a
            .iter()
            .zip(b)
            .fold(0f32, |m, (x, y)| m.max((x.r - y.r).abs()).max((x.i - y.i).abs()));
        if peak == 0.0 { diff } else { diff / peak }
    }

    /// `opus_fft_impl` gives the bits of the portable FFT of this target for
    /// every size.
    #[test]
    fn test_fft_matches_reference_fft() {
        let mut rng = Lcg(0x5eed_0002);
        let mut plans: Vec<KissFftState> = valid_sizes()
            .into_iter()
            .map(|n| KissFftState::new(n).unwrap())
            .collect();
        let base = KissFftState::new(480).unwrap();
        for n in [60, 120, 240] {
            plans.push(KissFftState::new_sub(&base, n).unwrap());
        }
        let mut region = Region::new();
        for st in &plans {
            let nfft = st.nfft();
            for class in 0..CLASSES {
                for _ in 0..4 {
                    // Three extra samples behind the transform must stay as
                    // they are.
                    let mut a = vec![KissCpx::default(); nfft + 3];
                    fill(&mut rng, &mut a, class);
                    let mut b = a.clone();
                    // A buffer of exactly `nfft` samples between two blocks
                    // of memory that must not be touched.
                    for at_end in PLACEMENTS {
                        let c = region.place(&a[..nfft], at_end);
                        opus_fft_impl(st, c);
                        reference_fft(st, &mut b[..nfft]);
                        assert_eq!(first_difference(c, &b[..nfft]), None, "nfft {nfft} class {class}");
                        b[..nfft].copy_from_slice(&a[..nfft]);
                    }
                    opus_fft_impl(st, &mut a);
                    reference_fft(st, &mut b);
                    assert_eq!(first_difference(&a, &b), None, "nfft {nfft} class {class}");
                }
            }
        }
    }

    /// Fusing the products changes the last bits only: for audio-sized input
    /// the two portable transforms agree to 1e-5 of the largest value.
    #[test]
    fn test_fused_fft_is_close_to_the_separate_one() {
        let mut rng = Lcg(0x5eed_0004);
        for nfft in valid_sizes() {
            let st = KissFftState::new(nfft).unwrap();
            let mut a = vec![KissCpx::default(); nfft];
            fill(&mut rng, &mut a, 0);
            let mut b = a.clone();
            opus_fft_impl_fused(&st, &mut a);
            opus_fft_impl_scalar(&st, &mut b);
            let d = max_relative_difference(&a, &b);
            assert!(d < 1e-5, "nfft {nfft}: {d}");
        }
    }

    /// The two aarch64 versions of the transform give the bits of the
    /// portable fused one for every size. The `fcma` version runs only on a
    /// processor that has the instructions.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_fused_fft_versions_agree() {
        let mut rng = Lcg(0x5eed_0005);
        let mut plans: Vec<KissFftState> = valid_sizes()
            .into_iter()
            .map(|n| KissFftState::new(n).unwrap())
            .collect();
        let base = KissFftState::new(480).unwrap();
        for n in [60, 120, 240] {
            plans.push(KissFftState::new_sub(&base, n).unwrap());
        }
        for st in &plans {
            let nfft = st.nfft();
            for class in 0..CLASSES {
                let mut want = vec![KissCpx::default(); nfft];
                fill(&mut rng, &mut want, class);
                let mut a = want.clone();
                let mut b = want.clone();
                opus_fft_impl_fused(st, &mut want);
                unsafe { fused::fma::stages(st, &mut a) };
                assert_eq!(first_difference(&a, &want), None, "fma, nfft {nfft} class {class}");
                if has_fcma() {
                    unsafe { fused::fcma::stages(st, &mut b) };
                    assert_eq!(first_difference(&b, &want), None, "fcma, nfft {nfft} class {class}");
                }
            }
        }
    }

    /// Each aarch64 butterfly gives the bits of its portable version, for
    /// every number of butterflies left over after the vector loop.
    ///
    /// The samples and the twiddles have exactly the lengths the kernel
    /// needs, and lie between two blocks of memory that must not be touched.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_neon_butterflies_match_scalar() {
        let mut rng = Lcg(0x5eed_0003);
        let mut samples = Region::new();
        let mut twiddles = Region::new();
        let ms: Vec<usize> = (0..=13).chain([16, 24, 32, 96]).collect();
        for class in 0..CLASSES {
            for &m in &ms {
                for groups in 0..=3usize {
                    // Twiddles: any finite values, the kernels do not rely on
                    // them having magnitude 1. For the class of zeros and
                    // ones the twiddles are zeros and ones too, like the
                    // first twiddle of every stage, which is (1, -0).
                    let tw: Vec<f32> = (0..8 * m)
                        .map(|_| {
                            if class == 4 {
                                ZEROS_AND_ONES[rng.next_u32() as usize % 4]
                            } else {
                                rng.unit()
                            }
                        })
                        .collect();
                    for radix in [3usize, 4, 5] {
                        let n = radix * m * groups;
                        let mut a = vec![KissCpx::default(); n];
                        fill(&mut rng, &mut a, class);
                        let input = a.clone();
                        let tw = &tw[..2 * (radix - 1) * m];
                        match radix {
                            3 => kf_bfly3_scalar::<Fused>(&mut a, tw, m, groups),
                            4 => kf_bfly4_scalar::<Fused>(&mut a, tw, m, groups),
                            _ => kf_bfly5_scalar::<Fused>(&mut a, tw, m, groups),
                        }
                        for at_end in PLACEMENTS {
                            let what = format!("radix {radix} m {m} groups {groups} class {class}");
                            let b = samples.place(&input, at_end);
                            let tw = twiddles.place(tw, at_end);
                            match radix {
                                3 => unsafe { fused::fma::bfly3(b, tw, m, groups) },
                                4 => unsafe { fused::fma::bfly4(b, tw, m, groups) },
                                _ => unsafe { fused::fma::bfly5(b, tw, m, groups) },
                            }
                            assert_eq!(first_difference(&a, b), None, "fma, {what}");
                            if has_fcma() {
                                let b = samples.place(&input, at_end);
                                match radix {
                                    3 => unsafe { fused::fcma::bfly3(b, tw, m, groups) },
                                    4 => unsafe { fused::fcma::bfly4(b, tw, m, groups) },
                                    _ => unsafe { fused::fcma::bfly5(b, tw, m, groups) },
                                }
                                assert_eq!(first_difference(&a, b), None, "fcma, {what}");
                            }
                        }
                    }
                }
            }

            for groups in 0..=9usize {
                for (radix, m) in [(2usize, 1usize), (2, 4), (4, 1)] {
                    let mut a = vec![KissCpx::default(); radix * m * groups];
                    fill(&mut rng, &mut a, class);
                    let input = a.clone();
                    match (radix, m) {
                        (2, _) => kf_bfly2_scalar(&mut a, m, groups),
                        _ => kf_bfly4_m1_scalar(&mut a, groups),
                    }
                    for at_end in PLACEMENTS {
                        let b = samples.place(&input, at_end);
                        match (radix, m) {
                            (2, 1) => unsafe { neon::bfly2_m1(b, groups) },
                            (2, _) => unsafe { neon::bfly2_m4(b, groups) },
                            _ => unsafe { neon::bfly4_m1(b, groups) },
                        }
                        assert_eq!(
                            first_difference(&a, b),
                            None,
                            "radix {radix} m {m} groups {groups} class {class}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_fft_roundtrip_60() {
        let nfft = 60;
        let st = KissFftState::new(nfft).unwrap();

        let mut fin = vec![KissCpx::default(); nfft];
        let mut fout = vec![KissCpx::default(); nfft];
        let mut finv = vec![KissCpx::default(); nfft];

        fin[0] = KissCpx::new(1.0, 0.0);

        opus_fft(&st, &fin, &mut fout);
        opus_ifft(&st, &fout, &mut finv);

        for i in 0..nfft {
            let expected = if i == 0 { 1.0 } else { 0.0 };
            assert!(
                cpx_almost_equal(&finv[i], &KissCpx::new(expected, 0.0), 1e-5),
                "Roundtrip failed at index {}: got ({}, {}), expected ({}, 0)",
                i,
                finv[i].r,
                finv[i].i,
                expected
            );
        }
    }

    #[test]
    fn test_fft_roundtrip_120() {
        let nfft = 120;
        let st = KissFftState::new(nfft).unwrap();

        let mut fin = vec![KissCpx::default(); nfft];
        let mut fout = vec![KissCpx::default(); nfft];
        let mut finv = vec![KissCpx::default(); nfft];

        for i in 0..nfft {
            fin[i] = KissCpx::new((2.0 * PI * 3.0 * i as f32 / nfft as f32).sin(), 0.0);
        }

        opus_fft(&st, &fin, &mut fout);
        opus_ifft(&st, &fout, &mut finv);

        for i in 0..nfft {
            assert!(
                cpx_almost_equal(&finv[i], &fin[i], 1e-4),
                "Roundtrip failed at index {}: got ({}, {}), expected ({}, {})",
                i,
                finv[i].r,
                finv[i].i,
                fin[i].r,
                fin[i].i
            );
        }
    }

    #[test]
    fn test_fft_roundtrip_480() {
        let nfft = 480;
        let st = KissFftState::new(nfft).unwrap();

        let mut fin = vec![KissCpx::default(); nfft];
        let mut fout = vec![KissCpx::default(); nfft];
        let mut finv = vec![KissCpx::default(); nfft];

        for i in 0..nfft {
            fin[i] = KissCpx::new(
                (2.0 * PI * 7.0 * i as f32 / nfft as f32).sin(),
                (2.0 * PI * 11.0 * i as f32 / nfft as f32).cos(),
            );
        }

        opus_fft(&st, &fin, &mut fout);
        opus_ifft(&st, &fout, &mut finv);

        for i in 0..nfft {
            assert!(
                cpx_almost_equal(&finv[i], &fin[i], 1e-4),
                "Roundtrip failed at index {}: got ({}, {}), expected ({}, {})",
                i,
                finv[i].r,
                finv[i].i,
                fin[i].r,
                fin[i].i
            );
        }
    }

    #[test]
    fn test_fft_dc_component() {
        let nfft = 120;
        let st = KissFftState::new(nfft).unwrap();

        let mut fin = vec![KissCpx::default(); nfft];
        let mut fout = vec![KissCpx::default(); nfft];

        for i in 0..nfft {
            fin[i] = KissCpx::new(1.0, 0.0);
        }

        opus_fft(&st, &fin, &mut fout);

        assert!(almost_equal(fout[0].r, 1.0, 1e-5));
        assert!(almost_equal(fout[0].i, 0.0, 1e-5));

        for i in 1..nfft {
            assert!(
                cpx_almost_equal(&fout[i], &KissCpx::new(0.0, 0.0), 1e-5),
                "Non-DC component at index {} is ({}, {}), expected (0, 0)",
                i,
                fout[i].r,
                fout[i].i
            );
        }
    }

    #[test]
    fn test_fft_performance() {
        use std::time::Instant;

        let nfft = 480;
        let st = KissFftState::new(nfft).unwrap();

        let mut fin = vec![KissCpx::default(); nfft];
        let mut fout = vec![KissCpx::default(); nfft];
        let mut finv = vec![KissCpx::default(); nfft];

        for i in 0..nfft {
            fin[i] = KissCpx::new(
                (2.0 * PI * 7.0 * i as f32 / nfft as f32).sin(),
                (2.0 * PI * 11.0 * i as f32 / nfft as f32).cos(),
            );
        }

        for _ in 0..100 {
            opus_fft(&st, &fin, &mut fout);
            opus_ifft(&st, &fout, &mut finv);
        }

        let iterations = 10000;
        let start = Instant::now();

        for _ in 0..iterations {
            opus_fft(&st, &fin, &mut fout);
            opus_ifft(&st, &fout, &mut finv);
        }

        let elapsed = start.elapsed();
        let ns_per_iter = elapsed.as_nanos() as f64 / (iterations as f64 * 2.0);

        println!("\nFFT/IFFT performance (nfft={}):", nfft);
        println!(
            "  Total time for {} FFT+IFFT pairs: {:?}",
            iterations, elapsed
        );
        println!("  Time per FFT or IFFT: {:.2} ns", ns_per_iter);
        println!(
            "  Throughput: {:.2} M points/sec",
            (nfft as f64) / ns_per_iter * 1000.0
        );
    }
}
