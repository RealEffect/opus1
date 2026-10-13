use core::f32::consts::PI;

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

#[inline(always)]
fn c_mul(a: &KissCpx, b: &KissCpx) -> KissCpx {
    KissCpx {
        r: a.r * b.r - a.i * b.i,
        i: a.r * b.i + a.i * b.r,
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
    pub bitrev: FixedVec<i16, KISS_MAX_N>,
    /// Twiddle factors, one contiguous table per stage.
    ///
    /// A stage of radix `p` with `m` butterflies per group stores, for each
    /// `k` in `1..p`, first the `m` real parts and then the `m` imaginary
    /// parts of `w^(k * j)`, `j = 0..m`. `w` is the root of unity of that
    /// stage. The values are copies of the entries the butterflies used to
    /// read with a stride from the table of the largest FFT size.
    stage_tw: FixedVec<f32, STAGE_TW_MAX>,
}

fn kf_factor(n_orig: usize, factors: &mut [i16; 2 * MAXFACTORS]) -> bool {
    // The twiddle table (KissCpx, KISS_MAX_N entries) and the bit-reversal
    // table are sized for KISS_MAX_N; anything larger would overflow them on
    // push (fail-fast assert) — reject up front so `KissFftState::new`
    // returns None gracefully (issue #27 deep scan).
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

fn compute_twiddles(nfft: usize) -> FixedVec<KissCpx, KISS_MAX_N> {
    let two_pi_over_n = -2.0 * PI / nfft as f32;
    let mut out: FixedVec<KissCpx, KISS_MAX_N> = FixedVec::new();
    for i in 0..nfft {
        let phase = two_pi_over_n * i as f32;
        out.push(KissCpx::new(phase.cos(), phase.sin()));
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
                for j in 0..m {
                    stage_tw.push(twiddles[k * j * step].r);
                }
                for j in 0..m {
                    stage_tw.push(twiddles[k * j * step].i);
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
// They run on every target without a vector version. The vector versions are
// tested against them and must give the same bits. The order of the
// operations is the one of libopus (kiss_fft.c, float build); no product is
// fused with a sum.
// ---------------------------------------------------------------------------

const EPI3_I: f32 = -0.866_025_4;
const YA_R: f32 = 0.309_017;
const YA_I: f32 = -0.95105652;
const YB_R: f32 = -0.809_017;
const YB_I: f32 = -0.58778525;

/// One radix-2 group with `m == 4`: eight samples, fixed twiddles.
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

/// One radix-3 butterfly. `w1` and `w2` are the twiddles of `f1` and `f2`.
#[inline(always)]
fn bfly3_one(f0: &mut KissCpx, f1: &mut KissCpx, f2: &mut KissCpx, w1: KissCpx, w2: KissCpx) {
    let scratch1 = c_mul(f1, &w1);
    let scratch2 = c_mul(f2, &w2);

    let scratch3 = c_add(&scratch1, &scratch2);
    let scratch0 = c_sub(&scratch1, &scratch2);

    let half_scratch3 = KissCpx::new(scratch3.r * 0.5, scratch3.i * 0.5);
    let fout_m = KissCpx::new(f0.r - half_scratch3.r, f0.i - half_scratch3.i);
    let scratch0_scaled = KissCpx::new(scratch0.r * EPI3_I, scratch0.i * EPI3_I);

    *f0 = c_add(f0, &scratch3);
    *f1 = KissCpx::new(fout_m.r - scratch0_scaled.i, fout_m.i + scratch0_scaled.r);
    *f2 = KissCpx::new(fout_m.r + scratch0_scaled.i, fout_m.i - scratch0_scaled.r);
}

/// One radix-4 butterfly. `w1`, `w2`, `w3` are the twiddles of `f1`, `f2`, `f3`.
#[inline(always)]
fn bfly4_one(
    f0: &mut KissCpx,
    f1: &mut KissCpx,
    f2: &mut KissCpx,
    f3: &mut KissCpx,
    w1: KissCpx,
    w2: KissCpx,
    w3: KissCpx,
) {
    let scratch0 = c_mul(f1, &w1);
    let scratch1 = c_mul(f2, &w2);
    let scratch2 = c_mul(f3, &w3);

    let scratch5 = c_sub(f0, &scratch1);
    let sum = c_add(f0, &scratch1);

    let scratch3 = c_add(&scratch0, &scratch2);
    let scratch4 = c_sub(&scratch0, &scratch2);

    *f2 = c_sub(&sum, &scratch3);
    *f0 = c_add(&sum, &scratch3);

    *f1 = KissCpx::new(scratch5.r + scratch4.i, scratch5.i - scratch4.r);
    *f3 = KissCpx::new(scratch5.r - scratch4.i, scratch5.i + scratch4.r);
}

/// One radix-5 butterfly. `w[k - 1]` is the twiddle of `f[k]`.
#[inline(always)]
fn bfly5_one(
    f0: &mut KissCpx,
    f1: &mut KissCpx,
    f2: &mut KissCpx,
    f3: &mut KissCpx,
    f4: &mut KissCpx,
    w: [KissCpx; 4],
) {
    let scratch0 = *f0;

    let scratch1 = c_mul(f1, &w[0]);
    let scratch2 = c_mul(f2, &w[1]);
    let scratch3 = c_mul(f3, &w[2]);
    let scratch4 = c_mul(f4, &w[3]);

    let scratch7 = c_add(&scratch1, &scratch4);
    let scratch10 = c_sub(&scratch1, &scratch4);
    let scratch8 = c_add(&scratch2, &scratch3);
    let scratch9 = c_sub(&scratch2, &scratch3);

    *f0 = KissCpx::new(
        scratch0.r + scratch7.r + scratch8.r,
        scratch0.i + scratch7.i + scratch8.i,
    );

    let scratch5 = KissCpx::new(
        scratch0.r + scratch7.r * YA_R + scratch8.r * YB_R,
        scratch0.i + scratch7.i * YA_R + scratch8.i * YB_R,
    );

    let scratch6 = KissCpx::new(
        scratch10.i * YA_I + scratch9.i * YB_I,
        -(scratch10.r * YA_I + scratch9.r * YB_I),
    );

    *f1 = c_sub(&scratch5, &scratch6);
    *f4 = c_add(&scratch5, &scratch6);

    let scratch11 = KissCpx::new(
        scratch0.r + scratch7.r * YB_R + scratch8.r * YA_R,
        scratch0.i + scratch7.i * YB_R + scratch8.i * YA_R,
    );

    let scratch12 = KissCpx::new(
        scratch9.i * YA_I - scratch10.i * YB_I,
        scratch10.r * YB_I - scratch9.r * YA_I,
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
fn kf_bfly3_scalar(fout: &mut [KissCpx], tw: &[f32], m: usize, groups: usize) {
    if m == 0 {
        return;
    }
    let (w1r, tw) = tw.split_at(m);
    let (w1i, tw) = tw.split_at(m);
    let (w2r, tw) = tw.split_at(m);
    let w2i = &tw[..m];

    for group in fout.chunks_exact_mut(3 * m).take(groups) {
        let (f0, rest) = group.split_at_mut(m);
        let (f1, f2) = rest.split_at_mut(m);
        let f2 = &mut f2[..m];
        for j in 0..m {
            bfly3_one(
                &mut f0[j],
                &mut f1[j],
                &mut f2[j],
                KissCpx::new(w1r[j], w1i[j]),
                KissCpx::new(w2r[j], w2i[j]),
            );
        }
    }
}

/// Radix-4 pass with `m > 1`. `tw` is the stage table: `2 * 3 * m` values.
#[inline(always)]
fn kf_bfly4_scalar(fout: &mut [KissCpx], tw: &[f32], m: usize, groups: usize) {
    if m == 0 {
        return;
    }
    let (w1r, tw) = tw.split_at(m);
    let (w1i, tw) = tw.split_at(m);
    let (w2r, tw) = tw.split_at(m);
    let (w2i, tw) = tw.split_at(m);
    let (w3r, tw) = tw.split_at(m);
    let w3i = &tw[..m];

    for group in fout.chunks_exact_mut(4 * m).take(groups) {
        let (f0, rest) = group.split_at_mut(m);
        let (f1, rest) = rest.split_at_mut(m);
        let (f2, f3) = rest.split_at_mut(m);
        let f3 = &mut f3[..m];
        for j in 0..m {
            bfly4_one(
                &mut f0[j],
                &mut f1[j],
                &mut f2[j],
                &mut f3[j],
                KissCpx::new(w1r[j], w1i[j]),
                KissCpx::new(w2r[j], w2i[j]),
                KissCpx::new(w3r[j], w3i[j]),
            );
        }
    }
}

/// Radix-5 pass. `tw` is the stage table: `2 * 4 * m` values.
#[inline(always)]
fn kf_bfly5_scalar(fout: &mut [KissCpx], tw: &[f32], m: usize, groups: usize) {
    if m == 0 {
        return;
    }
    let (w1r, tw) = tw.split_at(m);
    let (w1i, tw) = tw.split_at(m);
    let (w2r, tw) = tw.split_at(m);
    let (w2i, tw) = tw.split_at(m);
    let (w3r, tw) = tw.split_at(m);
    let (w3i, tw) = tw.split_at(m);
    let (w4r, tw) = tw.split_at(m);
    let w4i = &tw[..m];

    for group in fout.chunks_exact_mut(5 * m).take(groups) {
        let (f0, rest) = group.split_at_mut(m);
        let (f1, rest) = rest.split_at_mut(m);
        let (f2, rest) = rest.split_at_mut(m);
        let (f3, f4) = rest.split_at_mut(m);
        let f4 = &mut f4[..m];
        for j in 0..m {
            bfly5_one(
                &mut f0[j],
                &mut f1[j],
                &mut f2[j],
                &mut f3[j],
                &mut f4[j],
                [
                    KissCpx::new(w1r[j], w1i[j]),
                    KissCpx::new(w2r[j], w2i[j]),
                    KissCpx::new(w3r[j], w3i[j]),
                    KissCpx::new(w4r[j], w4i[j]),
                ],
            );
        }
    }
}

/// Runs all stages with the portable butterflies.
#[inline(always)]
fn fft_stages_scalar(st: &KissFftState, fout: &mut [KissCpx]) {
    for s in &st.stages[..st.nstages] {
        let m = s.m as usize;
        let groups = s.groups as usize;
        let tw = &st.stage_tw[s.tw as usize..s.tw as usize + s.tw_len()];
        match (s.radix, m) {
            (2, _) => kf_bfly2_scalar(fout, m, groups),
            (4, 1) => kf_bfly4_m1_scalar(fout, groups),
            (4, _) => kf_bfly4_scalar(fout, tw, m, groups),
            (3, _) => kf_bfly3_scalar(fout, tw, m, groups),
            (5, _) => kf_bfly5_scalar(fout, tw, m, groups),
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
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx")]
unsafe fn fft_stages_avx(st: &KissFftState, fout: &mut [KissCpx]) {
    fft_stages_scalar(st, fout);
}

// ---------------------------------------------------------------------------
// NEON butterflies (aarch64).
//
// Each one gives the same bits as its portable version: every lane does the
// same multiplies, additions and subtractions in the same order, and no
// product is fused with a sum.
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
mod neon {
    use super::{
        EPI3_I, KissCpx, YA_I, YA_R, YB_I, YB_R, bfly3_one, bfly4_one, bfly5_one, c_add, c_sub,
    };
    use core::arch::aarch64::*;

    /// Product of four complex numbers `a` with four twiddles `w`.
    ///
    /// Real and imaginary parts are in separate vectors. The operations are
    /// those of `c_mul`, in the same order.
    #[inline(always)]
    unsafe fn cmul(
        ar: float32x4_t,
        ai: float32x4_t,
        wr: float32x4_t,
        wi: float32x4_t,
    ) -> (float32x4_t, float32x4_t) {
        (
            vsubq_f32(vmulq_f32(ar, wr), vmulq_f32(ai, wi)),
            vaddq_f32(vmulq_f32(ar, wi), vmulq_f32(ai, wr)),
        )
    }

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

        // SAFETY: a group is 16 floats and `g < groups`.
        for g in 0..groups {
            let p = ptr.add(16 * g);
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

        // SAFETY: a group is 8 floats and `g < groups`.
        for g in 0..groups {
            let p = ptr.add(8 * g);
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
        }
    }

    /// Radix-3 pass. Four butterflies per iteration.
    ///
    /// # Safety
    /// `fout` must hold at least `3 * m * groups` samples and `tw` at least
    /// `4 * m` values.
    pub(super) unsafe fn bfly3(fout: &mut [KissCpx], tw: &[f32], m: usize, groups: usize) {
        debug_assert!(3 * m * groups <= fout.len());
        debug_assert!(4 * m <= tw.len());
        let half = vdupq_n_f32(0.5);
        let epi3 = vdupq_n_f32(EPI3_I);
        let w = tw.as_ptr();
        let base = fout.as_mut_ptr();
        let m4 = m & !3;

        // SAFETY: group `g` is `3 * m` samples. `p0`, `p1`, `p2` point to its
        // three streams of `m` samples (`2 * m` floats) each. A vector step
        // at `j` reads and writes samples `j..j + 4` of each stream and
        // twiddles `j..j + 4` of each of the four `m`-long tables, and
        // `j + 4 <= m`. The tail touches sample `j < m` of each stream.
        for g in 0..groups {
            let c0 = base.add(3 * m * g);
            let c1 = c0.add(m);
            let c2 = c1.add(m);
            let p0 = c0 as *mut f32;
            let p1 = c1 as *mut f32;
            let p2 = c2 as *mut f32;

            let mut j = 0usize;
            while j < m4 {
                let a0 = vld2q_f32(p0.add(2 * j));
                let a1 = vld2q_f32(p1.add(2 * j));
                let a2 = vld2q_f32(p2.add(2 * j));
                let (s1r, s1i) = cmul(a1.0, a1.1, vld1q_f32(w.add(j)), vld1q_f32(w.add(m + j)));
                let (s2r, s2i) = cmul(
                    a2.0,
                    a2.1,
                    vld1q_f32(w.add(2 * m + j)),
                    vld1q_f32(w.add(3 * m + j)),
                );

                let s3r = vaddq_f32(s1r, s2r);
                let s3i = vaddq_f32(s1i, s2i);
                let s0r = vsubq_f32(s1r, s2r);
                let s0i = vsubq_f32(s1i, s2i);

                let fmr = vsubq_f32(a0.0, vmulq_f32(s3r, half));
                let fmi = vsubq_f32(a0.1, vmulq_f32(s3i, half));
                let s0r = vmulq_f32(s0r, epi3);
                let s0i = vmulq_f32(s0i, epi3);

                vst2q_f32(
                    p0.add(2 * j),
                    float32x4x2_t(vaddq_f32(a0.0, s3r), vaddq_f32(a0.1, s3i)),
                );
                vst2q_f32(
                    p1.add(2 * j),
                    float32x4x2_t(vsubq_f32(fmr, s0i), vaddq_f32(fmi, s0r)),
                );
                vst2q_f32(
                    p2.add(2 * j),
                    float32x4x2_t(vaddq_f32(fmr, s0i), vsubq_f32(fmi, s0r)),
                );
                j += 4;
            }

            while j < m {
                bfly3_one(
                    &mut *c0.add(j),
                    &mut *c1.add(j),
                    &mut *c2.add(j),
                    KissCpx::new(*w.add(j), *w.add(m + j)),
                    KissCpx::new(*w.add(2 * m + j), *w.add(3 * m + j)),
                );
                j += 1;
            }
        }
    }

    /// Radix-4 pass with `m > 1`. Four butterflies per iteration.
    ///
    /// # Safety
    /// `fout` must hold at least `4 * m * groups` samples and `tw` at least
    /// `6 * m` values.
    pub(super) unsafe fn bfly4(fout: &mut [KissCpx], tw: &[f32], m: usize, groups: usize) {
        debug_assert!(4 * m * groups <= fout.len());
        debug_assert!(6 * m <= tw.len());
        let w = tw.as_ptr();
        let base = fout.as_mut_ptr();
        let m4 = m & !3;

        // SAFETY: group `g` is `4 * m` samples. `p0..p3` point to its four
        // streams of `m` samples (`2 * m` floats) each. A vector step at `j`
        // reads and writes samples `j..j + 4` of each stream and twiddles
        // `j..j + 4` of each of the six `m`-long tables, and `j + 4 <= m`.
        // The tail touches sample `j < m` of each stream.
        for g in 0..groups {
            let c0 = base.add(4 * m * g);
            let c1 = c0.add(m);
            let c2 = c1.add(m);
            let c3 = c2.add(m);
            let p0 = c0 as *mut f32;
            let p1 = c1 as *mut f32;
            let p2 = c2 as *mut f32;
            let p3 = c3 as *mut f32;

            let mut j = 0usize;
            while j < m4 {
                let a0 = vld2q_f32(p0.add(2 * j));
                let a1 = vld2q_f32(p1.add(2 * j));
                let a2 = vld2q_f32(p2.add(2 * j));
                let a3 = vld2q_f32(p3.add(2 * j));
                let (s0r, s0i) = cmul(a1.0, a1.1, vld1q_f32(w.add(j)), vld1q_f32(w.add(m + j)));
                let (s1r, s1i) = cmul(
                    a2.0,
                    a2.1,
                    vld1q_f32(w.add(2 * m + j)),
                    vld1q_f32(w.add(3 * m + j)),
                );
                let (s2r, s2i) = cmul(
                    a3.0,
                    a3.1,
                    vld1q_f32(w.add(4 * m + j)),
                    vld1q_f32(w.add(5 * m + j)),
                );

                let s5r = vsubq_f32(a0.0, s1r);
                let s5i = vsubq_f32(a0.1, s1i);
                let sumr = vaddq_f32(a0.0, s1r);
                let sumi = vaddq_f32(a0.1, s1i);

                let s3r = vaddq_f32(s0r, s2r);
                let s3i = vaddq_f32(s0i, s2i);
                let s4r = vsubq_f32(s0r, s2r);
                let s4i = vsubq_f32(s0i, s2i);

                vst2q_f32(
                    p2.add(2 * j),
                    float32x4x2_t(vsubq_f32(sumr, s3r), vsubq_f32(sumi, s3i)),
                );
                vst2q_f32(
                    p0.add(2 * j),
                    float32x4x2_t(vaddq_f32(sumr, s3r), vaddq_f32(sumi, s3i)),
                );
                vst2q_f32(
                    p1.add(2 * j),
                    float32x4x2_t(vaddq_f32(s5r, s4i), vsubq_f32(s5i, s4r)),
                );
                vst2q_f32(
                    p3.add(2 * j),
                    float32x4x2_t(vsubq_f32(s5r, s4i), vaddq_f32(s5i, s4r)),
                );
                j += 4;
            }

            while j < m {
                bfly4_one(
                    &mut *c0.add(j),
                    &mut *c1.add(j),
                    &mut *c2.add(j),
                    &mut *c3.add(j),
                    KissCpx::new(*w.add(j), *w.add(m + j)),
                    KissCpx::new(*w.add(2 * m + j), *w.add(3 * m + j)),
                    KissCpx::new(*w.add(4 * m + j), *w.add(5 * m + j)),
                );
                j += 1;
            }
        }
    }

    /// Radix-5 pass. Four butterflies per iteration.
    ///
    /// # Safety
    /// `fout` must hold at least `5 * m * groups` samples and `tw` at least
    /// `8 * m` values.
    pub(super) unsafe fn bfly5(fout: &mut [KissCpx], tw: &[f32], m: usize, groups: usize) {
        debug_assert!(5 * m * groups <= fout.len());
        debug_assert!(8 * m <= tw.len());
        let ya_r = vdupq_n_f32(YA_R);
        let ya_i = vdupq_n_f32(YA_I);
        let yb_r = vdupq_n_f32(YB_R);
        let yb_i = vdupq_n_f32(YB_I);
        let w = tw.as_ptr();
        let base = fout.as_mut_ptr();
        let m4 = m & !3;

        // SAFETY: group `g` is `5 * m` samples. `p0..p4` point to its five
        // streams of `m` samples (`2 * m` floats) each. A vector step at `j`
        // reads and writes samples `j..j + 4` of each stream and twiddles
        // `j..j + 4` of each of the eight `m`-long tables, and `j + 4 <= m`.
        // The tail touches sample `j < m` of each stream.
        for g in 0..groups {
            let c0 = base.add(5 * m * g);
            let c1 = c0.add(m);
            let c2 = c1.add(m);
            let c3 = c2.add(m);
            let c4 = c3.add(m);
            let p0 = c0 as *mut f32;
            let p1 = c1 as *mut f32;
            let p2 = c2 as *mut f32;
            let p3 = c3 as *mut f32;
            let p4 = c4 as *mut f32;

            let mut j = 0usize;
            while j < m4 {
                let s0 = vld2q_f32(p0.add(2 * j));
                let a1 = vld2q_f32(p1.add(2 * j));
                let a2 = vld2q_f32(p2.add(2 * j));
                let a3 = vld2q_f32(p3.add(2 * j));
                let a4 = vld2q_f32(p4.add(2 * j));
                let (s1r, s1i) = cmul(a1.0, a1.1, vld1q_f32(w.add(j)), vld1q_f32(w.add(m + j)));
                let (s2r, s2i) = cmul(
                    a2.0,
                    a2.1,
                    vld1q_f32(w.add(2 * m + j)),
                    vld1q_f32(w.add(3 * m + j)),
                );
                let (s3r, s3i) = cmul(
                    a3.0,
                    a3.1,
                    vld1q_f32(w.add(4 * m + j)),
                    vld1q_f32(w.add(5 * m + j)),
                );
                let (s4r, s4i) = cmul(
                    a4.0,
                    a4.1,
                    vld1q_f32(w.add(6 * m + j)),
                    vld1q_f32(w.add(7 * m + j)),
                );

                let s7r = vaddq_f32(s1r, s4r);
                let s7i = vaddq_f32(s1i, s4i);
                let s10r = vsubq_f32(s1r, s4r);
                let s10i = vsubq_f32(s1i, s4i);
                let s8r = vaddq_f32(s2r, s3r);
                let s8i = vaddq_f32(s2i, s3i);
                let s9r = vsubq_f32(s2r, s3r);
                let s9i = vsubq_f32(s2i, s3i);

                vst2q_f32(
                    p0.add(2 * j),
                    float32x4x2_t(
                        vaddq_f32(vaddq_f32(s0.0, s7r), s8r),
                        vaddq_f32(vaddq_f32(s0.1, s7i), s8i),
                    ),
                );

                let s5r = vaddq_f32(
                    vaddq_f32(s0.0, vmulq_f32(s7r, ya_r)),
                    vmulq_f32(s8r, yb_r),
                );
                let s5i = vaddq_f32(
                    vaddq_f32(s0.1, vmulq_f32(s7i, ya_r)),
                    vmulq_f32(s8i, yb_r),
                );
                // scratch6 = (s6r, -s6i_neg). Subtracting -x is the same as
                // adding x, bit for bit, so the negation is not computed.
                let s6r = vaddq_f32(vmulq_f32(s10i, ya_i), vmulq_f32(s9i, yb_i));
                let s6i_neg = vaddq_f32(vmulq_f32(s10r, ya_i), vmulq_f32(s9r, yb_i));

                vst2q_f32(
                    p1.add(2 * j),
                    float32x4x2_t(vsubq_f32(s5r, s6r), vaddq_f32(s5i, s6i_neg)),
                );
                vst2q_f32(
                    p4.add(2 * j),
                    float32x4x2_t(vaddq_f32(s5r, s6r), vsubq_f32(s5i, s6i_neg)),
                );

                let s11r = vaddq_f32(
                    vaddq_f32(s0.0, vmulq_f32(s7r, yb_r)),
                    vmulq_f32(s8r, ya_r),
                );
                let s11i = vaddq_f32(
                    vaddq_f32(s0.1, vmulq_f32(s7i, yb_r)),
                    vmulq_f32(s8i, ya_r),
                );
                let s12r = vsubq_f32(vmulq_f32(s9i, ya_i), vmulq_f32(s10i, yb_i));
                let s12i = vsubq_f32(vmulq_f32(s10r, yb_i), vmulq_f32(s9r, ya_i));

                vst2q_f32(
                    p2.add(2 * j),
                    float32x4x2_t(vaddq_f32(s11r, s12r), vaddq_f32(s11i, s12i)),
                );
                vst2q_f32(
                    p3.add(2 * j),
                    float32x4x2_t(vsubq_f32(s11r, s12r), vsubq_f32(s11i, s12i)),
                );
                j += 4;
            }

            while j < m {
                bfly5_one(
                    &mut *c0.add(j),
                    &mut *c1.add(j),
                    &mut *c2.add(j),
                    &mut *c3.add(j),
                    &mut *c4.add(j),
                    [
                        KissCpx::new(*w.add(j), *w.add(m + j)),
                        KissCpx::new(*w.add(2 * m + j), *w.add(3 * m + j)),
                        KissCpx::new(*w.add(4 * m + j), *w.add(5 * m + j)),
                        KissCpx::new(*w.add(6 * m + j), *w.add(7 * m + j)),
                    ],
                );
                j += 1;
            }
        }
    }
}

/// Runs all stages with the NEON butterflies.
///
/// # Safety
/// `fout` must hold exactly `st.nfft` samples.
#[cfg(target_arch = "aarch64")]
unsafe fn fft_stages_neon(st: &KissFftState, fout: &mut [KissCpx]) {
    debug_assert!(fout.len() == st.nfft);
    // SAFETY: `KissFftState::build` makes `groups * radix * m == nfft` for
    // every stage, and gives each stage a table of `tw_len()` values.
    for s in &st.stages[..st.nstages] {
        let m = s.m as usize;
        let groups = s.groups as usize;
        let tw = &st.stage_tw[s.tw as usize..s.tw as usize + s.tw_len()];
        match (s.radix, m) {
            (2, 1) => neon::bfly2_m1(fout, groups),
            (2, _) => neon::bfly2_m4(fout, groups),
            (4, 1) => neon::bfly4_m1(fout, groups),
            (4, _) => neon::bfly4(fout, tw, m, groups),
            (3, _) => neon::bfly3(fout, tw, m, groups),
            (5, _) => neon::bfly5(fout, tw, m, groups),
            // A one-point transform has a single stage of radix 1. It
            // leaves the sample as it is.
            _ => {}
        }
    }
}

/// In-place FFT of `st.nfft()` samples that are already in bit-reversed order.
pub fn opus_fft_impl(st: &KissFftState, fout: &mut [KissCpx]) {
    // The butterfly kernels index up to nfft elements; the SIMD paths read
    // through raw pointers, so an undersized slice would be out-of-bounds
    // memory access rather than a panic — check up front (issue #27 deep
    // scan).
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
            fft_stages_scalar(st, fout);
        }
    }
    #[cfg(not(any(
        target_arch = "aarch64",
        target_arch = "x86",
        target_arch = "x86_64"
    )))]
    fft_stages_scalar(st, fout);
}

/// The same transform as `opus_fft_impl`, computed with the portable
/// butterflies on every target.
///
/// It is the reference the vector versions are tested against.
pub fn opus_fft_impl_scalar(st: &KissFftState, fout: &mut [KissCpx]) {
    assert!(
        fout.len() >= st.nfft,
        "opus_fft_impl_scalar: fout.len() ({}) < nfft ({})",
        fout.len(),
        st.nfft
    );
    fft_stages_scalar(st, &mut fout[..st.nfft]);
}

pub fn opus_fft(st: &KissFftState, fin: &[KissCpx], fout: &mut [KissCpx]) {
    let scale = st.scale;
    let nfft = st.nfft;

    for i in 0..nfft {
        let x = &fin[i];
        let rev = st.bitrev[i] as usize;
        fout[rev] = KissCpx::new(x.r * scale, x.i * scale);
    }

    opus_fft_impl(st, fout);
}

pub fn opus_ifft(st: &KissFftState, fin: &[KissCpx], fout: &mut [KissCpx]) {
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

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;

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
    const CLASSES: usize = 4;

    /// Fills `out` with test data.
    ///
    /// Class 0: audio-sized values. Class 1: values whose products overflow
    /// to infinity and then give NaN. Class 2: values whose products are
    /// subnormal. Class 3: signed zeros, infinities, the largest and the
    /// smallest numbers, mixed with ordinary values.
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
    /// of libopus, reading one shared twiddle table with a stride.
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

        fn bfly4(
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

                        let scratch0 = c_mul(&fout[idx + m], &twiddles[tw1]);
                        let scratch1 = c_mul(&fout[idx + m2], &twiddles[tw2]);
                        let scratch2 = c_mul(&fout[idx + m3], &twiddles[tw3]);

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

        fn bfly3(
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

                    let scratch1 = c_mul(&fout[idx + m], &twiddles[tw1]);
                    let scratch2 = c_mul(&fout[idx + m2], &twiddles[tw2]);

                    let scratch3 = c_add(&scratch1, &scratch2);
                    let scratch0 = c_sub(&scratch1, &scratch2);

                    let half_scratch3 = KissCpx::new(scratch3.r * 0.5, scratch3.i * 0.5);

                    let fout_m =
                        KissCpx::new(fout[idx].r - half_scratch3.r, fout[idx].i - half_scratch3.i);

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

        fn bfly5(
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

                    let scratch1 = c_mul(&fout[idx1], &twiddles[tw1]);
                    let scratch2 = c_mul(&fout[idx2], &twiddles[tw2]);
                    let scratch3 = c_mul(&fout[idx3], &twiddles[tw3]);
                    let scratch4 = c_mul(&fout[idx4], &twiddles[tw4]);

                    let scratch7 = c_add(&scratch1, &scratch4);
                    let scratch10 = c_sub(&scratch1, &scratch4);
                    let scratch8 = c_add(&scratch2, &scratch3);
                    let scratch9 = c_sub(&scratch2, &scratch3);

                    fout[idx0] = KissCpx::new(
                        scratch0.r + scratch7.r + scratch8.r,
                        scratch0.i + scratch7.i + scratch8.i,
                    );

                    let scratch5 = KissCpx::new(
                        scratch0.r + scratch7.r * ya.r + scratch8.r * yb.r,
                        scratch0.i + scratch7.i * ya.r + scratch8.i * yb.r,
                    );

                    let scratch6 = KissCpx::new(
                        scratch10.i * ya.i + scratch9.i * yb.i,
                        -(scratch10.r * ya.i + scratch9.r * yb.i),
                    );

                    fout[idx1] = c_sub(&scratch5, &scratch6);
                    fout[idx4] = c_add(&scratch5, &scratch6);

                    let scratch11 = KissCpx::new(
                        scratch0.r + scratch7.r * yb.r + scratch8.r * ya.r,
                        scratch0.i + scratch7.i * yb.r + scratch8.i * ya.r,
                    );

                    let scratch12 = KissCpx::new(
                        scratch9.i * ya.i - scratch10.i * yb.i,
                        scratch10.r * yb.i - scratch9.r * ya.i,
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
        pub fn fft(nfft: usize, shift: usize, fout: &mut [KissCpx]) {
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
                    4 => bfly4(fout, fstride_adjusted, &twiddles, m, fstride_i, m2),
                    3 => bfly3(fout, fstride_adjusted, &twiddles, m, fstride_i, m2),
                    5 => bfly5(fout, fstride_adjusted, &twiddles, m, fstride_i, m2),
                    _ => {}
                }
                m = m2;
            }
        }
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

    #[test]
    fn test_bitrev_permutation() {
        for &nfft in &[60, 120, 240, 480] {
            let st = KissFftState::new(nfft).unwrap();
            let mut sorted: Vec<i16> = st.bitrev.as_slice().to_vec();
            sorted.sort();
            let expected: Vec<i16> = (0..nfft).map(|x| x as i16).collect();
            assert_eq!(
                sorted,
                expected,
                "bitrev for nfft={} should be a permutation of 0..{}",
                nfft,
                nfft - 1
            );
        }
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
    /// read one shared table with a stride.
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
                opus_fft_impl_scalar(st, &mut a);
                strided::fft(*nfft, *shift, &mut b);
                assert_eq!(
                    first_difference(&a, &b),
                    None,
                    "nfft {nfft} shift {shift} class {class}"
                );
            }
        }
    }

    /// `opus_fft_impl` gives the bits of the portable FFT for every size.
    #[test]
    fn test_fft_matches_scalar_fft() {
        let mut rng = Lcg(0x5eed_0002);
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
                for _ in 0..4 {
                    // Three extra samples behind the transform must stay as
                    // they are.
                    let mut a = vec![KissCpx::default(); nfft + 3];
                    fill(&mut rng, &mut a, class);
                    let mut b = a.clone();
                    opus_fft_impl(st, &mut a);
                    opus_fft_impl_scalar(st, &mut b);
                    assert_eq!(first_difference(&a, &b), None, "nfft {nfft} class {class}");
                }
            }
        }
    }

    /// Each NEON butterfly gives the bits of its portable version, for every
    /// number of butterflies left over after the vector loop.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_neon_butterflies_match_scalar() {
        let mut rng = Lcg(0x5eed_0003);
        let ms: Vec<usize> = (0..=13).chain([16, 24, 32, 96]).collect();
        for class in 0..CLASSES {
            for &m in &ms {
                for groups in 0..=3usize {
                    // Twiddles: any finite values, the kernels do not rely on
                    // them having magnitude 1.
                    let tw: Vec<f32> = (0..8 * m).map(|_| rng.unit()).collect();
                    for radix in [3usize, 4, 5] {
                        let n = radix * m * groups;
                        let mut a = vec![KissCpx::default(); n];
                        fill(&mut rng, &mut a, class);
                        let mut b = a.clone();
                        let tw = &tw[..2 * (radix - 1) * m];
                        match radix {
                            3 => {
                                kf_bfly3_scalar(&mut a, tw, m, groups);
                                unsafe { neon::bfly3(&mut b, tw, m, groups) };
                            }
                            4 => {
                                kf_bfly4_scalar(&mut a, tw, m, groups);
                                unsafe { neon::bfly4(&mut b, tw, m, groups) };
                            }
                            _ => {
                                kf_bfly5_scalar(&mut a, tw, m, groups);
                                unsafe { neon::bfly5(&mut b, tw, m, groups) };
                            }
                        }
                        assert_eq!(
                            first_difference(&a, &b),
                            None,
                            "radix {radix} m {m} groups {groups} class {class}"
                        );
                    }
                }
            }

            for groups in 0..=9usize {
                for (radix, m) in [(2usize, 1usize), (2, 4), (4, 1)] {
                    let mut a = vec![KissCpx::default(); radix * m * groups];
                    fill(&mut rng, &mut a, class);
                    let mut b = a.clone();
                    match (radix, m) {
                        (2, 1) => {
                            kf_bfly2_scalar(&mut a, 1, groups);
                            unsafe { neon::bfly2_m1(&mut b, groups) };
                        }
                        (2, _) => {
                            kf_bfly2_scalar(&mut a, 4, groups);
                            unsafe { neon::bfly2_m4(&mut b, groups) };
                        }
                        _ => {
                            kf_bfly4_m1_scalar(&mut a, groups);
                            unsafe { neon::bfly4_m1(&mut b, groups) };
                        }
                    }
                    assert_eq!(
                        first_difference(&a, &b),
                        None,
                        "radix {radix} m {m} groups {groups} class {class}"
                    );
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
